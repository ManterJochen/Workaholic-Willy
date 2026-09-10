"""The datagen command line, asked the three questions a first-time operator asks it.

MEASURED on 2026-09-10, against a tree with no dataset written yet:

* `python -m datagen why-no-jaw` ended in
  `TypeError: '<=' not supported between instances of 'NoneType' and 'int'`
  (`assets/service.py:60`). The command had NO working invocation without `--limit`, and the
  library twin `MeshPreparation.why_no_jaw` declares `limit: int = 40`, so a sane default existed
  and the CLI overrode it with `None`.
* `python -m datagen prompts` ended in
  `FileNotFoundError: [Errno 2] No such file or directory: 'data\\datagen\\v1\\provenance.json'`,
  and the traceback replaced the exit-code footer, so the run's own log had no line saying how it
  finished. `--name` defaults to `v1`, which is the README's example, so this is what the
  documented first invocation does on a fresh clone.
* `python -m datagen plan --config <a JSON array>` ended in
  `TypeError: datagen.config.DatagenConfig() argument after ** must be a mapping, not list`,
  past a guard whose own comment reads "A MISTYPED FLAG AND A CRASH MUST NOT LOOK ALIKE".

The property under test in all three is that one: a wrong request reaches the operator as one line
and an exit code, never as a stack trace, and every run says how it ended.
"""

from __future__ import annotations

import inspect
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from datagen.assets.service import MeshPreparation

#: The twin's own default, read from the twin. Naming 40 here as well would be the second copy that
#: the defect was made of.
TWIN_LIMIT = inspect.signature(MeshPreparation.why_no_jaw).parameters["limit"].default

#: A library big enough that a default-sized draw is a real subset of it.
LIBRARY = [(f"gso_{i:03d}", f"g{i}.obj") for i in range(200)]


def _diagnosis(meshes, **_kwargs):                        # type: ignore[no-untyped-def]
    """What `assets.diagnose.why_no_jaw` returns, with the keys `JawDiagnosis.render` reads."""
    return {"probed": len(meshes), "unreadable": 0, "with_no_label_in_any_pose": 0,
            "rescued_by_a_different_pose": 0, "reasons": {}}


class WhyNoJawHasAWorkingDefaultTests(unittest.TestCase):
    """`--limit` is SHARED with camera-probe and eval-grasps, where `None` means "the whole
    dataset". Forwarding that `None` to a command that samples MESHES is what removed the twin's
    default."""

    def _run(self, argv: list[str]) -> tuple[int, list]:
        from datagen.__main__ import main

        asked: list = []

        def capture(meshes, **kwargs):                     # type: ignore[no-untyped-def]
            asked.extend(meshes)
            return _diagnosis(meshes, **kwargs)

        with mock.patch.object(MeshPreparation, "entries", lambda self: list(LIBRARY)), \
                mock.patch("datagen.assets.diagnose.why_no_jaw", side_effect=capture), \
                redirect_stdout(io.StringIO()):
            code = main(argv)
        return code, asked

    def test_the_documented_bare_invocation_runs(self) -> None:
        code, asked = self._run(["why-no-jaw"])
        self.assertEqual(0, code)
        self.assertEqual(TWIN_LIMIT, len(asked),
                         "the CLI did not fall back to the twin's own default")

    def test_a_named_limit_still_decides(self) -> None:
        """The control: without it the test above passes for a CLI that ignores `--limit`."""
        code, asked = self._run(["why-no-jaw", "--limit", "7"])
        self.assertEqual(0, code)
        self.assertEqual(7, len(asked))

    def test_the_help_names_this_command_and_says_meshes(self) -> None:
        """The help named camera-probe and eval-grasps only, and said "scenes" for a command that
        samples meshes, so the one reader who would have found the default was told about two other
        commands instead."""
        from datagen.__main__ import build_parser

        lines = build_parser().format_help().splitlines()
        starts = [i for i, line in enumerate(lines) if line.strip().startswith("--limit")]
        self.assertTrue(starts, "no --limit entry in the help at all")
        block = [lines[starts[0]]]
        for line in lines[starts[0] + 1:]:
            if line.strip().startswith("--"):        # the next flag's entry, so this one is over
                break
            block.append(line)
        text = " ".join(block)
        self.assertIn("why-no-jaw", text, "the flag's help never mentions why-no-jaw")
        self.assertIn("MESH", text.upper(), "the flag's help still calls them scenes")


class AnAbsentDatasetIsARefusalNotACrashTests(unittest.TestCase):
    """MEASURED: `prompts`, `label-grasps`, `eval-grasps` and `grasp-gate` all reached the operator
    as a traceback when the dataset directory was absent, and each of them skipped the
    `<command> finished in N s: exit C` line that is the run's only record of its own outcome."""

    def _run(self, argv: list[str]) -> tuple[int, str]:
        from datagen.__main__ import main

        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            code = main(argv)
        return code, err.getvalue()

    def test_prompts_on_an_unwritten_dataset_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            code, err = self._run(["prompts", "--out", name])
        self.assertEqual(2, code)
        self.assertIn("provenance.json", err)
        self.assertNotIn("Traceback", err)

    def test_label_grasps_on_an_unwritten_dataset_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            code, err = self._run(["label-grasps", "--out", name])
        self.assertEqual(2, code)
        self.assertIn("provenance.json", err)

    def test_the_refusal_names_the_command_that_writes_the_dataset(self) -> None:
        """A path that does not exist is not an instruction. The one thing the operator needs is
        the step that would create it."""
        with tempfile.TemporaryDirectory() as name:
            _code, err = self._run(["prompts", "--out", name, "--name", "v9"])
        self.assertIn("datagen build", err)
        self.assertIn("v9", err)

    def test_the_run_still_says_how_it_finished(self) -> None:
        """The exit code is the whole verdict of a CLI and the log line is where it is kept."""
        with tempfile.TemporaryDirectory() as name:
            with self.assertLogs("datagen.cli", level="INFO") as caught:
                self._run(["prompts", "--out", name])
        self.assertTrue([line for line in caught.output if "finished" in line and "exit 2" in line],
                        f"no exit-code footer in {caught.output}")

    def test_a_command_that_already_refuses_keeps_ITS_OWN_code(self) -> None:
        """The first control. `verify` and `camera-probe` handle an absent dataset themselves and
        exit 1; a guard that turned those into a usage error would be rewriting verdicts it did not
        make."""
        with tempfile.TemporaryDirectory() as name:
            code, _err = self._run(["verify", "--out", name])
        self.assertEqual(1, code)

    def test_a_run_that_works_is_untouched(self) -> None:
        """The second control: without it every assertion above passes for a CLI that refuses
        everything."""
        code, _err = self._run(["plan"])
        self.assertEqual(0, code)


class ANonObjectConfigIsARefusalTests(unittest.TestCase):
    """The guard caught `ValidationError`, `OSError` and `json.JSONDecodeError`. A JSON file that
    parses but is not an object reached `DatagenConfig(**base)` as a `TypeError` and went straight
    past it."""

    def _config(self, payload: object) -> tuple[int, str]:
        from datagen.__main__ import main

        err = io.StringIO()
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with redirect_stderr(err), redirect_stdout(io.StringIO()):
                code = main(["plan", "--config", str(path)])
        return code, err.getvalue()

    def test_a_json_array_is_a_usage_error(self) -> None:
        code, err = self._config([1, 2, 3])
        self.assertEqual(2, code)
        self.assertIn("config:", err)
        self.assertNotIn("Traceback", err)

    def test_a_json_string_is_a_usage_error(self) -> None:
        code, _err = self._config("scenes=300")
        self.assertEqual(2, code)

    def test_a_json_object_is_still_accepted(self) -> None:
        """The control: the guard refuses the shape, not every config."""
        code, _err = self._config({"scenes": 4})
        self.assertEqual(0, code)


if __name__ == "__main__":
    unittest.main()
