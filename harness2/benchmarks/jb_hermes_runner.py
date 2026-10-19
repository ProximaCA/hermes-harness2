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

"""JobBench Hermes launcher: stage a task workspace, run one Hermes session, keep the artifacts.

Mirrors jb_opencode_runner.py against `hermes chat --format stream-json`, with two differences
that are deliberate: the per-task wall clock is enforced by Python (macOS has no `timeout`
binary) and the tool surface is pinned with `-t` so a benchmark cell does not inherit whatever
toolsets the host profile happens to enable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

JOBBENCH = Path(__file__).resolve().parents[1]
TEMP_BASE = Path("/tmp/hevo_ws")
TRAJ_DIR_NAME = "model_traj"
EVENT_PREFIX = '{"type":'

PROMPT = """=== TASK FOLDER ===
{task_folder}

=== INSTRUCTIONS ===
1. Read the TASK_INSTRUCTIONS.txt file in the task folder above
2. Based on the Reference Files section in TASK_INSTRUCTIONS.txt, read the corresponding files from the same task folder using appropriate tools.
3. Complete the task as specified in TASK_INSTRUCTIONS.txt
4. Only save the final deliverables to the output directory specified below. Do not save any intermediate or temporary files.

=== OUTPUT DIRECTORY ===
{output_dir}

IMPORTANT:
- All reference files are in the task folder: {task_folder}
- Only save the final deliverables to the output directory {output_dir}. Do not save any intermediate or temporary files.
- You MUST only access files within {workspace} or search online for new reference files if you find needed. Do NOT access any files or directories in this system outside of this path.
- If you encounter ambiguous or conflicting information, analyze the conflict, explain your reasoning, and justify the approach you choose.
- If a file cannot be read directly (e.g., .xlsx, .docx, .db, .pptx), use appropriate tools, MCP servers, or code to extract and process its contents."""


@dataclass(frozen=True)
class Settings:
    task_ids: tuple[str, ...]
    label: str
    render_dir: Path
    tasks_base: Path
    hermes_bin: str
    model: str
    max_turns: int
    toolsets: str
    timeout_per_task: int
    max_retries: int
    log_dir: Path


def _settings() -> Settings:
    """Resolve the environment contract the adapter writes; a bad one exits non-zero."""
    env = os.environ
    missing = [n for n in ("LABEL", "HARNESS_RENDER_DIR", "TASK_IDS") if not env.get(n)]
    if missing:
        raise SystemExit(f"[ERROR] unset: {', '.join(missing)}")
    render_dir = Path(env["HARNESS_RENDER_DIR"])
    if not (render_dir / "AGENTS.md").is_file():
        raise SystemExit(f"[ERROR] HARNESS_RENDER_DIR has no AGENTS.md: {render_dir}")
    tasks_base = Path(env.get("TASKS_BASE_DIR") or JOBBENCH / "dataset" / "main")
    return Settings(
        task_ids=tuple(env["TASK_IDS"].split()),
        label=env["LABEL"],
        render_dir=render_dir,
        tasks_base=tasks_base,
        hermes_bin=env.get("HERMES_BIN") or "hermes",
        model=env.get("HERMES_MODEL") or "",
        max_turns=int(env.get("HERMES_MAX_TURNS") or 200),
        toolsets=env.get("HERMES_TOOLSETS") or "",
        timeout_per_task=int(env.get("TIMEOUT_PER_TASK") or "3600"),
        max_retries=int(env.get("MAX_RETRIES") or 2),
        log_dir=tasks_base / ".harness2-logs",
    )


def _hms() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _nonempty(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _log(log_file: Path, line: str) -> None:
    print(line, flush=True)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _reset_task(task_dir: Path, workspace: Path) -> None:
    """Restore a pristine task_folder and an empty output/ before a retry."""
    shutil.rmtree(workspace / "task_folder", ignore_errors=True)
    shutil.rmtree(workspace / "output", ignore_errors=True)
    shutil.copytree(task_dir, workspace / "task_folder")
    (workspace / "output").mkdir(parents=True, exist_ok=True)


def _stage(task_dir: Path, render_dir: Path, workspace: Path) -> None:
    shutil.rmtree(workspace, ignore_errors=True)
    (workspace / "output").mkdir(parents=True)
    shutil.copytree(task_dir, workspace / "task_folder")
    if (render_dir / "AGENTS.md").is_file():
        shutil.copy(render_dir / "AGENTS.md", workspace / "AGENTS.md")
    if (render_dir / "scripts").is_dir():
        shutil.copytree(render_dir / "scripts", workspace / "scripts")


def _command(cfg: Settings, workspace: Path, prompt_file: Path) -> list[str]:
    cmd = [
        cfg.hermes_bin,
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
        str(int(cfg.max_turns)),
    ]
    if cfg.model and cfg.model != "hermes/default":
        cmd += ["-m", cfg.model]
    if cfg.toolsets:
        cmd += ["-t", cfg.toolsets]
    return cmd


def _attempt(
    cfg: Settings, workspace: Path, traj_file: Path, log_file: Path, task_name: str, attempt: int
) -> int:
    """Run Hermes once; its event stream lands in `traj_file`, everything else in the log."""
    prompt = PROMPT.format(
        task_folder=workspace / "task_folder",
        output_dir=workspace / "output",
        workspace=workspace,
    )
    prompt_file = workspace / f".prompt_{task_name}_a{attempt}.txt"
    prompt_file.write_text(prompt, encoding="utf-8")

    handle, raw = tempfile.mkstemp(prefix=f".{task_name}_a{attempt}.stdout.", dir=traj_file.parent)
    os.close(handle)
    timed_out = False
    with open(raw, "w", encoding="utf-8") as out, log_file.open("a", encoding="utf-8") as err:
        try:
            code = subprocess.run(
                _command(cfg, workspace, prompt_file),
                cwd=workspace,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                timeout=cfg.timeout_per_task,
            ).returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            code = 124
    with open(raw, encoding="utf-8", errors="replace") as src:
        traj_file.write_text(
            "".join(ln for ln in src if ln.startswith(EVENT_PREFIX)), encoding="utf-8"
        )
    os.unlink(raw)
    prompt_file.unlink(missing_ok=True)
    return code if not timed_out else 124


def run_one(cfg: Settings, task_id: str) -> bool:
    """Roll one task; the existing-output skip mirrors the adapter's resume, for direct runs.

    True only when deliverables landed in `model_output/<label>/`. A rollout that dies on
    auth or a proxy still writes a trajectory, so the caller must not read one as evidence
    that the agent ran: `main` turns a False here into a non-zero exit.

    A rollout CUT by the wall clock is not evidence either, and is not the same thing as a
    model that spent its own budget: the timeout is a property of this machine, not of the
    harness being measured. So a timed-out attempt is discarded whole -- even when it left
    usable-looking files -- and the task is retried; if every attempt times out the task is
    reported as failed and left unmarked, so a truncated run can never become the evidence a
    proposer reasons from.
    """
    task_parent = cfg.tasks_base / task_id
    task_dir = task_parent / "task_folder"
    task_name = task_id.replace("/", "_")
    final_output_dir = task_parent / "model_output" / cfg.label
    traj_dir = task_parent / TRAJ_DIR_NAME / cfg.label
    log_file = cfg.log_dir / f"hevo_{task_name}_{cfg.label}_{_stamp()}.log"

    if not task_dir.is_dir():
        _log(log_file, f"[ERROR] no task_folder: {task_dir}")
        return False
    if _nonempty(final_output_dir):
        print(f"[{_hms()}] SKIP (done): {task_id} ({cfg.label})", flush=True)
        return True
    traj_dir.mkdir(parents=True, exist_ok=True)

    workspace = TEMP_BASE / f"{task_name}_{cfg.label}_{os.getpid()}"
    output_dir = workspace / "output"
    _stage(task_dir, cfg.render_dir, workspace)

    delivered = False
    for attempt in range(1, cfg.max_retries + 1):
        if attempt > 1:
            _reset_task(task_dir, workspace)
        traj_file = traj_dir / f"{task_name}_{cfg.label}_attempt{attempt}_{_stamp()}.jsonl"
        _log(log_file, f"[{_hms()}] run {task_id} ({cfg.label}) attempt {attempt} -> {traj_file}")
        code = _attempt(cfg, workspace, traj_file, log_file, task_name, attempt)
        if code == 124:
            _log(
                log_file,
                f"[{_hms()}] TIMEOUT {task_id} attempt {attempt}: cut at "
                f"{cfg.timeout_per_task}s -- discarded, a truncated run is not evidence",
            )
            continue
        if _nonempty(output_dir):
            delivered = True
            break
        _log(log_file, f"[{_hms()}] rc={code} {task_id} attempt {attempt} produced no output")

    if delivered:
        final_output_dir.mkdir(parents=True, exist_ok=True)
        for item in sorted(output_dir.iterdir()):
            if not item.name.startswith("."):
                shutil.move(str(item), str(final_output_dir / item.name))
        _log(log_file, f"[{_hms()}] output -> {final_output_dir}")
    else:
        _log(log_file, f"[{_hms()}] NO OUTPUT for {task_id} ({cfg.label})")
    shutil.rmtree(workspace, ignore_errors=True)
    return delivered and _nonempty(final_output_dir)


def main() -> int:
    cfg = _settings()
    print(
        f"=== harness eval: LABEL={cfg.label} HERMES={cfg.hermes_bin} "
        f"MODEL={cfg.model or '(profile default)'} ===",
        flush=True,
    )
    print(f"render: {cfg.render_dir}", flush=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    TEMP_BASE.mkdir(parents=True, exist_ok=True)
    failed = [task_id for task_id in cfg.task_ids if not run_one(cfg, task_id)]
    print(f"=== harness eval done: {cfg.label} ===", flush=True)
    if failed:
        print(f"no output for: {', '.join(failed)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
