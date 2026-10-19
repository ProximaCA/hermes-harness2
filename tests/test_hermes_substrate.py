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

"""The hermes substrate: surface, composition, refusals and the stream-json renderer.

The end-to-end rollout test spends a model call and is opt-in:
`HARNESS2_HERMES_E2E=1 python -m unittest tests.test_hermes_substrate`.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from harness2 import config
from harness2.benchmarks import jobbench
from harness2.core import tree as HT
from harness2.substrate.hermes import (
    HERMES_SURFACE,
    HermesSubstrate,
    compose_agents_md,
    render_stream_json,
)

HM_BASE = jobbench.HM_BASE_HARNESS


def _tree_dir(
    tmp: Path,
    *,
    system: str = "",
    guardrails: str = "",
    memory: str = "",
    guidance: str = "",
    scripts: dict | None = None,
) -> Path:
    d = tmp / "tree"
    files = {
        "systemprompt.md": system,
        "guardrails.md": guardrails,
        "LongTermMEMORY.md": memory,
        "tool_descriptions/tool_guidance.md": guidance,
    }
    for rel, body in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    for name, body in (scripts or {}).items():
        p = d / "scripts" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return d


class SurfaceTests(unittest.TestCase):
    def test_baseline_tree_is_a_valid_harness(self):
        tree = HT.read_tree(HERMES_SURFACE, HM_BASE)
        census = HT.validate(HERMES_SURFACE, tree, allow_subagents=False)
        self.assertGreater(census["always_on_chars"], 0)
        self.assertEqual(census["skills"], 0)

    def test_skills_are_not_loadable_and_plugins_are_off_the_surface(self):
        # `skills/` stays on the surface so the tree can see it, but materialise refuses it:
        # Hermes resolves skills from HERMES_HOME, never from the workspace.
        self.assertTrue(HT.is_allowed(HERMES_SURFACE, "skills/x/SKILL.md"))
        self.assertFalse(HT.is_allowed(HERMES_SURFACE, "plugin/x.ts"))
        self.assertTrue(HT.is_allowed(HERMES_SURFACE, "scripts/check.py"))
        self.assertTrue(HT.is_allowed(HERMES_SURFACE, "guardrails.md"))


class MaterialiseTests(unittest.TestCase):
    def test_staging_keeps_the_tree_and_interpolates_later(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _tree_dir(tmp, system="Focus on {{domain_focus}}.", guardrails="Rule one.")
            spec = jobbench.make_spec("hm")
            install = HermesSubstrate().materialise(src, spec, {})
            try:
                self.assertTrue((install.staging_dir / "systemprompt.md").is_file())
                self.assertFalse((install.staging_dir / "AGENTS.md").exists())
            finally:
                import shutil

                shutil.rmtree(install.staging_dir, ignore_errors=True)

    def test_composition_orders_components_and_indexes_scripts(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _tree_dir(
                tmp,
                system="SYS {{domain_focus}}",
                guardrails="GUARD",
                memory="MEM",
                guidance="TOOLS",
                scripts={"helper.py": '"""Validates the workbook."""\nprint(1)\n'},
            )
            body = compose_agents_md(src, {"domain_focus": "legal"})
            self.assertIn("SYS legal", body)
            self.assertLess(body.index("SYS legal"), body.index("GUARD"))
            self.assertLess(body.index("GUARD"), body.index("TOOLS"))
            self.assertLess(body.index("TOOLS"), body.index("MEM"))
            self.assertIn("python scripts/helper.py", body)
            self.assertIn("Validates the workbook.", body)

    def test_comments_render_to_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _tree_dir(tmp, guardrails="<!-- empty in v0 -->")
            self.assertEqual(compose_agents_md(src, {}).strip(), "")

    def test_unloadable_components_are_refused_loudly(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _tree_dir(tmp, system="x")
            skill = src / "skills" / "demo" / "SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text("---\nname: demo\ndescription: d\n---\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                HermesSubstrate().materialise(src, jobbench.make_spec("hm"), {})
            self.assertIn("HERMES_HOME/skills", str(ctx.exception))

    def test_foreign_install_mode_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            src = _tree_dir(Path(td), system="x")
            spec = jobbench.make_spec("cx")  # codex_root
            with self.assertRaises(ValueError):
                HermesSubstrate().materialise(src, spec, {})


class RenderTests(unittest.TestCase):
    def _render(self, events, **kw):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.jsonl"
            p.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
            return render_stream_json(
                p, obs_trunc=kw.get("obs_trunc", 1000), max_chars=kw.get("max_chars", 100_000)
            )

    def test_text_deltas_are_joined_into_one_block(self):
        out = self._render(
            [
                {"type": "system", "subtype": "init"},
                {"type": "text", "text": "Hel"},
                {"type": "text", "text": "lo "},
                {"type": "text", "text": "world"},
                {"type": "result", "text": "Hello world", "exit_code": 0},
            ]
        )
        self.assertEqual(out, "[ASSISTANT] Hello world")

    def test_tools_and_errors_map_to_the_shared_grammar(self):
        out = self._render(
            [
                {"type": "text", "text": "Reading."},
                {"type": "tool_use", "name": "terminal", "input": {"command": "ls"}},
                {"type": "tool_result", "name": "terminal", "output": "a.txt", "is_error": False},
                {"type": "tool_result", "name": "read_file", "output": "boom", "is_error": True},
                {"type": "result", "text": "", "error": "gateway down"},
            ]
        )
        self.assertEqual(
            out.splitlines(),
            [
                "[ASSISTANT] Reading.",
                '[ACTION] terminal({"command": "ls"})',
                "[OBSERVATION] a.txt",
                "[OBSERVATION ERROR] boom",
                "[ERROR] gateway down",
            ],
        )

    def test_unparseable_stream_is_reported_not_empty(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.jsonl"
            p.write_text("Session: local-abc\nnot json\n", encoding="utf-8")
            self.assertEqual(
                render_stream_json(p, obs_trunc=100, max_chars=1000),
                "(no parseable events in trajectory)",
            )

    def test_render_truncates_at_the_cap(self):
        out = self._render([{"type": "text", "text": "x" * 500}], max_chars=100)
        self.assertIn("render truncation", out)
        self.assertLess(len(out), 200)


class PreflightTests(unittest.TestCase):
    def test_hermes_cli_answers_version(self):
        try:
            HermesSubstrate().preflight()
        except RuntimeError as exc:  # pragma: no cover - only on a host without the CLI
            self.skipTest(f"hermes CLI unavailable: {exc}")


@unittest.skipUnless(os.environ.get("HARNESS2_HERMES_E2E"), "set HARNESS2_HERMES_E2E=1")
class EndToEndTests(unittest.TestCase):
    """One real Hermes rollout through the JobBench runner: costs one model call."""

    def test_runner_produces_deliverables(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            store = tmp / "store"
            task_dir = store / "demo" / "task1" / "task_folder"
            task_dir.mkdir(parents=True)
            (task_dir / "TASK_INSTRUCTIONS.txt").write_text(
                "Write the single line OK to output/result.txt and nothing else.\n",
                encoding="utf-8",
            )
            render = tmp / "render"
            render.mkdir()
            (render / "AGENTS.md").write_text(
                "Follow the task instructions in the task folder and write the final "
                "deliverables to the specified output directory.\n",
                encoding="utf-8",
            )
            env = {
                **os.environ,
                "TASK_IDS": "demo/task1",
                "LABEL": "e2e",
                "HARNESS_RENDER_DIR": str(render),
                "TASKS_BASE_DIR": str(store),
                "HERMES_BIN": config.HERMES_BIN,
                "HERMES_MODEL": config.HERMES_MODEL,
                "HERMES_MAX_TURNS": "12",
                "HERMES_TOOLSETS": "terminal,file",
                "TIMEOUT_PER_TASK": "600",
                "MAX_RETRIES": "1",
            }
            proc = subprocess.run(
                [sys.executable, str(jobbench.HM_RUNNER)],
                env=env,
                capture_output=True,
                text=True,
                timeout=900,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            out = store / "demo" / "task1" / "model_output" / "e2e"
            self.assertTrue(any(out.iterdir()), f"no deliverables in {out}")
            traj = list((store / "demo" / "task1" / "model_traj" / "e2e").glob("*.jsonl"))
            self.assertTrue(traj, "no trajectory written")
            rendered = render_stream_json(traj[0], obs_trunc=2000, max_chars=200_000)
            self.assertNotEqual(rendered, "(no parseable events in trajectory)")


if __name__ == "__main__":
    unittest.main()
