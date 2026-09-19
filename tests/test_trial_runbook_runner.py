"""The runner that executes a runbook's own commands (customer chain lane C5e).

``scripts/trial/run_runbook.py`` is a trial instrument: the trial's record is only worth what its expectations are, so
each property here has the control that shows a step can fail.
"""

from __future__ import annotations

import importlib.util
import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path, PureWindowsPath

_REPO = Path(__file__).resolve().parents[1]
_PY = shlex.quote(Path(sys.executable).as_posix())


def _tool():
    name = "_trial_run_runbook"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / "trial" / "run_runbook.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_RUNBOOK = f"""# A runbook

<!-- step: first -->
```bash
{_PY} -c "print('one')"
```

<!-- step: layer file: out/$HAND.yaml -->
```yaml
gripper:
  model: $HAND
```

<!-- trial-skip: needs a physical controller -->
<!-- step: bench -->
```bash
{_PY} -c "print('never')"
```

<!-- step: second -->
```bash
{_PY} -c \\
  "print('admitted')"
```
"""


class _Case(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = _tool()
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.root = self.tmp / "root"
        self.root.mkdir()
        self.logs = self.tmp / "logs"

    def runner(self, plan: dict, text: str = _RUNBOOK):
        return self.tool.Runner(runbook_path="runbook.md", steps=self.tool.parse_runbook(text), plan=plan,
                                root=self.root, logs=self.logs)


class ParsingTests(_Case):
    def test_steps_parse_in_document_order_and_malformed_markers_are_refused(self) -> None:
        steps = self.tool.parse_runbook(_RUNBOOK)
        self.assertEqual([(s.id, s.kind) for s in steps], [("first", "bash"), ("layer", "yaml"), ("bench", "bash"),
                                                           ("second", "bash")])
        self.assertEqual(steps[2].skip, "needs a physical controller")
        self.assertIn("print('admitted')", self.tool.command_of(steps[3]))

    def test_a_duplicate_id_and_an_empty_skip_reason_are_refused_naming_the_line(self) -> None:
        """⭐ THE CONTROL."""
        with self.assertRaises(self.tool.Malformed) as caught:
            self.tool.parse_runbook(_RUNBOOK + "\n<!-- step: first -->\n```bash\necho\n```\n")
        self.assertIn("line", str(caught.exception))
        with self.assertRaises(self.tool.Malformed) as caught:
            self.tool.parse_runbook("<!-- trial-skip:  -->\n<!-- step: a -->\n```bash\necho\n```\n")
        self.assertIn("line 1", str(caught.exception))


class RunningTests(_Case):
    def test_an_unbound_variable_runs_nothing(self) -> None:
        plan = {"phases": [{"name": "P", "steps": [{"runbook": "first", "expect": {"exit": 0}},
                                                   {"runbook": "layer"}]}]}
        with self.assertRaises(self.tool.Malformed) as caught:
            self.runner(plan).run()
        self.assertIn("HAND", str(caught.exception))
        self.assertFalse(self.logs.exists() and any(self.logs.iterdir()))
        # ⭐ THE CONTROL: bound, the same plan runs and writes the layer.
        plan["phases"][0]["bindings"] = {"HAND": "acme"}
        self.assertEqual(self.runner(plan).run(), 0)
        self.assertEqual((self.root / "out" / "acme.yaml").read_bytes(), b"gripper:\n  model: acme\n")

    def test_exit_zero_without_the_expected_text_is_a_failed_step(self) -> None:
        plan = {"phases": [{"name": "P", "steps": [
            {"runbook": "first", "expect": {"exit": 0, "stdout_has": ["admitted"]}},
            {"runbook": "second", "expect": {"exit": 0, "stdout_has": ["admitted"]}},
        ]}]}
        runner = self.runner(plan)
        self.assertEqual(runner.run(), 1)
        self.assertEqual(runner.record["stopped_at"], {"phase": "P", "id": "first"})
        self.assertEqual(len(runner.record["steps"]), 1, "a later step ran after the stop")
        # ⭐ THE CONTROL: the step that does print it passes.
        plan["phases"][0]["steps"] = plan["phases"][0]["steps"][1:]
        self.assertEqual(self.runner(plan).run(), 0)

    def test_a_hidden_file_comes_back_even_when_a_step_between_fails(self) -> None:
        precious = self.root / "evidence" / "file.json"
        precious.parent.mkdir()
        precious.write_bytes(b'{"b1": true}\n')
        plan = {"phases": [{"name": "P", "steps": [
            {"hide": "evidence/file.json"},
            {"id": "fails", "run": f"{_PY} -c \"raise SystemExit(3)\"", "expect": {"exit": 0}},
            {"restore": "evidence/file.json"},
        ]}]}
        runner = self.runner(plan)
        self.assertEqual(runner.run(), 1)
        self.assertEqual(precious.read_bytes(), b'{"b1": true}\n')
        self.assertTrue(json.loads((self.logs / "record.json").read_text(encoding="utf-8"))["stopped_at"])

    def test_a_restore_that_finds_other_bytes_fails_loudly(self) -> None:
        """⭐ THE CONTROL: a hidden copy that changed is not put back as if it were the file."""
        precious = self.root / "file.json"
        precious.write_bytes(b"one\n")
        runner = self.runner({"phases": []})
        runner._one({"name": "P"}, {"hide": "file.json"}, "hide", {}, {}, 1)
        place, _ = runner.hidden["file.json"]
        place.write_bytes(b"two\n")
        said = runner._restore("file.json")
        self.assertIn("other bytes", said)
        self.assertFalse(precious.exists())


class WindowsPathTests(_Case):
    def test_the_root_reaches_a_command_whole_on_windows(self) -> None:
        """MEASURED 2026-09-17: ROOT bound as ``D:\\willy_trial\\customer_hand`` lost its backslashes to the posix split,
        and the copy landed in the original tree as ``D:willy_trialcustomer_hand``, a path relative to the drive's
        current directory."""
        # The paths a Windows host holds, read as Windows paths on any host: a POSIX `Path` would keep the
        # backslashes inside one name, and `_bindings` only ever asks a path for `as_posix()`.
        runner = self.tool.Runner(runbook_path="runbook.md", steps=[], plan={"phases": []},
                                  root=PureWindowsPath("D:\\willy_trial\\customer_hand"),
                                  logs=PureWindowsPath("D:\\logs\\trial"))
        bound = runner._bindings({})
        words = shlex.split(self.tool.substitute("copy --copy $ROOT --out $LOGS/x.json", bound), posix=True)
        self.assertEqual(words[2:], ["D:/willy_trial/customer_hand", "--out", "D:/logs/trial/x.json"])

    def test_a_binding_holding_a_backslash_is_refused_before_anything_runs(self) -> None:
        """⭐ THE CONTROL: a plan that spells a path the way the split destroys is refused, not run."""
        plan = {"bindings": {"ORIG": "D:\\dev\\Workaholic-Willy"},
                "phases": [{"name": "P", "steps": [{"runbook": "first", "expect": {"exit": 0}}]}]}
        with self.assertRaises(self.tool.Malformed) as caught:
            self.runner(plan).run()
        self.assertIn("ORIG", str(caught.exception))


class TheHashTests(_Case):
    def test_the_steps_hash_moves_with_one_character(self) -> None:
        one = self.tool.steps_sha256(self.tool.parse_runbook(_RUNBOOK))
        self.assertEqual(one, self.tool.steps_sha256(self.tool.parse_runbook(_RUNBOOK)))
        changed = _RUNBOOK.replace("print('one')", "print('onf')")
        self.assertNotEqual(one, self.tool.steps_sha256(self.tool.parse_runbook(changed)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
