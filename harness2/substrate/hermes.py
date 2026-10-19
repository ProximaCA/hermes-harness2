# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HermesSubstrate: the Hermes Agent CLI as a solver substrate.

Hermes reads project context from ``AGENTS.md`` in the working directory, so the four
always-on components (c1..c4) are composed into that one file when the tree is rendered, and
c7 helper scripts ride along in ``./scripts``. Components Hermes cannot load from a workspace
are REFUSED at materialise time instead of being staged and silently ignored:

    c5 sub_agents  Hermes delegates through its own tool; no per-workspace agent files.
    c6 skills      resolved from HERMES_HOME/skills and trusted project dirs only.
    c8 plugins     loaded from HERMES_HOME/plugins.

One rollout is one ``hermes chat --format stream-json`` process per task; that JSONL stream is
also the trajectory format this substrate renders.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping

from .. import config
from ..core import runmark as RM
from ..core import tree as HT
from ..core.tree import SurfaceSpec
from .base import Install, Launcher, RunRequest

HERMES_SURFACE = SurfaceSpec(
    name="hermes",
    always_on=(
        "systemprompt.md",
        "guardrails.md",
        "LongTermMEMORY.md",
        "tool_descriptions/tool_guidance.md",
    ),
    always_on_max=24_000,
    always_on_cap_reason="AGENTS.md is injected into every turn of the session",
    skill_dir="skills",
    skill_name_rule="must_match",
    trigger_hint=(
        "Fix it by adding a trigger line naming `{name}` to systemprompt.md or "
        'guardrails.md, e.g. "When <the situation this skill handles>, load the '
        '`{name}` skill."'
    ),
    autotrigger_file="guardrails.md",
    trigger_line="- When {desc}, load the `{name}` skill.\n",
    subagent_requires_mode=False,
    plugin_dir=None,
    hooks_file=None,
    stubs=(
        (
            "scripts/README.md",
            "<!-- c7: deterministic helpers, run as `python scripts/<name>.py`. "
            "Add one by emitting scripts/<name>.py. -->\n",
        ),
    ),
)

HERMES_INSTALL_MODE = "workspace_agents_md"

_TOKEN = re.compile(r"\{\{\s*([a-z_]+)\s*\}\}")
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_TRUNC_FMT = "…(+{n} chars — render truncation)"


def _clip(value, n: int) -> str:
    s = "" if value is None else str(value)
    s = s.replace("\x00", "")
    if n <= 0 or len(s) <= n:
        return s
    return s[:n] + _TRUNC_FMT.format(n=len(s) - n)


def _interp(text: str, ctx: Mapping[str, str]) -> str:
    """Substitute {{ name }} tokens only; single braces in prose stay untouched."""
    return _TOKEN.sub(lambda m: str(ctx.get(m.group(1), "")), text)


def _strip_comments(text: str) -> str:
    """Drop HTML comment blocks, so a comment-only component renders to nothing."""
    return _COMMENT.sub("", text).strip()


def _read(staged: Path, rel: str) -> str:
    p = staged / rel
    return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""


def _scripts(staged: Path) -> list[Path]:
    d = staged / "scripts"
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob("*.py") if not p.name.startswith("_"))


def _script_doc(p: Path) -> str:
    """First line of the script's module docstring, if any."""
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "helper script"
    m = re.search(r'^\s*(?:"""|\'\'\')(.+)', txt)
    return m.group(1).strip()[:120] if m else "helper script"


def compose_agents_md(staged: Path, ctx: Mapping[str, str]) -> str:
    """c1 prose + c6 guardrails + c4 tool guidance + c3 memory + a c7 script note.

    Order follows the tree: the always-on components as Hermes will read them back out of
    AGENTS.md, with the script index last.
    """
    parts: list[str] = []
    for rel, head in (
        ("systemprompt.md", ""),
        ("guardrails.md", "# Operating Guardrails\n\n"),
        ("tool_descriptions/tool_guidance.md", ""),
        ("LongTermMEMORY.md", "# Long-Term Memory\n\n"),
    ):
        body = _strip_comments(_interp(_read(staged, rel), ctx))
        if body:
            parts.append(head + body)
    scripts = _scripts(staged)
    if scripts:
        listing = "\n".join(f"- `python scripts/{s.name}` — {_script_doc(s)}" for s in scripts)
        parts.append(
            "# Helper scripts (run via Bash)\n\nDeterministic helpers are available in "
            "`./scripts`. Prefer them over hand-rolling the same logic:\n\n" + listing
        )
    return "\n\n".join(p for p in parts if p.strip()) + "\n"


def hermes_bin() -> str:
    return config.HERMES_BIN


class HermesSubstrate:
    """Hermes behind the one Substrate interface; the launcher body is benchmark-supplied."""

    name = "hm"

    def __init__(self, launcher: Launcher | None = None):
        self._launcher = launcher

    def surface(self) -> SurfaceSpec:
        return HERMES_SURFACE

    def materialise(self, tree_dir: Path, spec, ctx: Mapping[str, str]) -> Install:
        mode = spec.install_mode
        if mode != HERMES_INSTALL_MODE:
            raise ValueError(
                f"install_mode {mode!r} is not a hermes mode; this substrate supports "
                f"{HERMES_INSTALL_MODE!r} (the opencode modes and codex_root belong to "
                f"their own substrates)"
            )
        tree_dir = Path(tree_dir)
        if not HT.is_tree(HERMES_SURFACE, tree_dir):
            raise ValueError(f"{tree_dir} is not a hermes harness tree (no always-on file present)")
        self._refuse_unloadable(tree_dir)
        staged = Path(tempfile.mkdtemp(prefix=f"adapt-hm-{mode}-"))
        try:
            shutil.copytree(tree_dir, staged, dirs_exist_ok=True)
            for key, body in ctx.items():
                if key.startswith("file:"):
                    rel = HT.normalize_rel(key[len("file:") :])
                    p = staged / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(body, encoding="utf-8")
        except BaseException:
            shutil.rmtree(staged, ignore_errors=True)
            raise
        return Install(
            staging_dir=staged,
            fingerprint=RM.fingerprint(tree_dir),
            extras={"install_mode": mode},
        )

    @staticmethod
    def _refuse_unloadable(tree_dir: Path) -> None:
        """Refuse components Hermes cannot load from a workspace instead of staging dead weight."""
        for pattern, why in (
            (
                "skills/*/SKILL.md",
                "Hermes resolves skills from HERMES_HOME/skills and trusted project dirs, "
                "never from the workspace",
            ),
            (
                "sub_agents/*/AGENT.md",
                "Hermes delegates through its own tool and has no per-workspace agent files",
            ),
            ("plugin/*.ts", "Hermes loads plugins from HERMES_HOME/plugins"),
        ):
            found = sorted(str(p.relative_to(tree_dir)) for p in tree_dir.glob(pattern))
            if found:
                raise ValueError(
                    f"{', '.join(found)} cannot be loaded on the hermes substrate: {why}. "
                    f"Put that guidance in tool_descriptions/tool_guidance.md instead."
                )

    def run_argv(
        self,
        prompt_file: Path,
        workspace: Path,
        *,
        model: str = "",
        max_turns: int = 200,
        toolsets: str = "",
    ) -> list[str]:
        """The reference argv; the real launcher builds its own.

        `--query-file` carries the prompt so nothing is shell-interpreted and no argv limit
        applies; `--oneshot --yolo` is what makes a headless rollout possible at all.
        """
        argv = [
            hermes_bin(),
            "chat",
            "--query-file",
            str(prompt_file),
            "--format",
            "stream-json",
            "--oneshot",
            "--yolo",
            "--in",
            str(workspace),
            "--max-turns",
            str(int(max_turns)),
        ]
        if model and model != config.HERMES_DEFAULT_MODEL:
            argv += ["-m", model]
        if toolsets:
            argv += ["-t", toolsets]
        return argv

    def execute(
        self, task_ref: str, install: Install, run_ids: list[str], timeout: int
    ) -> list[Path]:
        if self._launcher is None:
            raise RuntimeError(
                "HermesSubstrate was constructed without a launcher body; the benchmark "
                "adapter must supply one (it owns workspace staging and read-back locations)"
            )
        return self._launcher(
            RunRequest(task_ref=task_ref, run_ids=tuple(run_ids), timeout=int(timeout)), install
        )

    def read_trajectory(
        self, stream_path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 400
    ) -> str:
        return render_stream_json(
            Path(stream_path), obs_trunc=obs_trunc, max_chars=max_chars, args_trunc=args_trunc
        )

    def preflight(self) -> None:
        """Checks the hermes CLI answers `--version`; starts no model call."""
        binary = hermes_bin()
        if not (Path(binary).is_file() or shutil.which(binary)):
            raise RuntimeError(
                f"hermes binary not found at {binary!r}: install the Hermes Agent CLI and put "
                f"it on PATH, or set HARNESS2_HERMES_BIN to its absolute path."
            )
        try:
            r = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"hermes --version failed: {type(exc).__name__}: {exc}") from None
        if r.returncode:
            raise RuntimeError(
                f"hermes --version exited {r.returncode}: {(r.stderr or '').strip()[-300:]}"
            )


def render_stream_json(path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 400) -> str:
    """Render `hermes chat --format stream-json` JSONL into the shared grammar.

    ``text`` events are DELTAS of one assistant message, so consecutive deltas are joined
    before emitting one ASSISTANT block; a line per delta would be one line per token.
    """
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"(failed to read trajectory: {e})"

    lines: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        if not pending:
            return
        text = "".join(pending).strip()
        pending.clear()
        if text:
            lines.append(f"[ASSISTANT] {_clip(text, obs_trunc)}")

    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            ev = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        etype = ev.get("type")
        if etype == "text":
            if isinstance(ev.get("text"), str):
                pending.append(ev["text"])
            continue
        flush()
        if etype == "tool_use":
            args = ev.get("input")
            rendered = json.dumps(args, default=str, ensure_ascii=False) if args is not None else ""
            lines.append(f"[ACTION] {ev.get('name') or 'tool'}({_clip(rendered, args_trunc)})")
        elif etype == "tool_result":
            out = _clip(ev.get("output"), obs_trunc)
            lines.append(
                f"[OBSERVATION ERROR] {out}" if ev.get("is_error") else f"[OBSERVATION] {out}"
            )
        elif etype == "result" and ev.get("error"):
            lines.append(f"[ERROR] {_clip(ev['error'], 300)}")
    flush()

    rendered_text = "\n".join(lines).strip()
    if not rendered_text:
        return "(no parseable events in trajectory)"
    if len(rendered_text) > max_chars:
        dropped = len(rendered_text) - max_chars
        rendered_text = rendered_text[:max_chars] + "\n" + _TRUNC_FMT.format(n=dropped)
    return rendered_text


__all__ = [
    "HERMES_SURFACE",
    "HERMES_INSTALL_MODE",
    "HermesSubstrate",
    "compose_agents_md",
    "hermes_bin",
    "render_stream_json",
]
