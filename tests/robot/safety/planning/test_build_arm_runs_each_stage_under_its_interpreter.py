"""``scripts/arms/build_arm.py`` starts every stage of an arm's build under the interpreter it runs under (P0f F4b).

Three stages run under the cuRobo sidecar's interpreter and three under the project's, and until F4b the operator
had to know which, from a runbook. The orchestrator is judged here with its runner replaced, so nothing is baked or
installed: which command each stage is, in which order, what a verify leaves unwritten, and that the first failure
stops the build before the stages that depend on it.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.config import ConfigError
from src.contracts import UNSET

_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT = _ROOT / "scripts" / "arms" / "build_arm.py"


def _module():
    spec = importlib.util.spec_from_file_location("_build_arm_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Runner:
    """Stands in for subprocess.run: records each command and answers with the exit code a stage is given."""

    def __init__(self, fail: str = "") -> None:
        self.calls: list[list[str]] = []
        self.fail = fail

    def __call__(self, argv: list[str], **_: object) -> SimpleNamespace:
        self.calls.append(argv)
        failing = self.fail and Path(argv[1]).stem == self.fail
        return SimpleNamespace(returncode=1 if failing else 0)


class TheStagesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = _module()
        cls.folder = tempfile.TemporaryDirectory()
        cls.cu = str(Path(cls.folder.name) / "curobo_python.exe")
        Path(cls.cu).write_text("", encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.folder.cleanup()

    def build(self, **kwargs: object):
        return self.m.ArmBuild.from_model("ur3e", python="project-python", curobo_python=self.cu, **kwargs)

    def test_each_stage_runs_under_its_own_interpreter_in_order(self) -> None:
        report = self.build(write=True, hand="robotiq_hande", profile="ur3e").run(runner=_Runner())
        self.assertEqual([stage.name for stage in report.stages], list(self.m.STAGES))
        runs_under = {stage.name: stage.argv[0] for stage in report.stages}
        self.assertEqual(runs_under, {"fetch": self.cu, "bake": "project-python", "fit": "project-python",
                                      "descriptor": self.cu, "check": self.cu, "doctor": "project-python"})
        self.assertEqual(report.exit_code, 0)

    def test_every_stage_script_it_starts_is_on_disk(self) -> None:
        for stage in self.build(write=True, hand="robotiq_hande", profile="ur3e").stages():
            with self.subTest(stage=stage.name):
                if stage.name == "doctor":
                    self.assertEqual(stage.argv[1:4], ("-m", "src.robot.safety.planning", "--doctor"))
                    self.assertTrue((_ROOT / "src/robot/safety/planning/__main__.py").is_file())
                else:
                    self.assertTrue(Path(stage.argv[1]).is_file(), stage.argv[1])
                    self.assertEqual(Path(stage.argv[1]).parent, _SCRIPT.parent)

    def test_a_verify_writes_nothing(self) -> None:
        runner = _Runner()
        report = self.build(hand="robotiq_hande", profile="ur3e").run(runner=runner)
        self.assertFalse([argv for argv in runner.calls if "--write" in argv], "a verify passed --write")
        self.assertIn("--check", runner.calls[0], "the fetch of a verify goes to the network")
        descriptor = next(stage for stage in report.stages if stage.name == "descriptor")
        self.assertIn("--write", descriptor.skipped)
        self.assertNotIn(descriptor.argv, [tuple(argv) for argv in runner.calls])

    def test_a_write_bakes_fits_and_installs(self) -> None:
        runner = _Runner()
        self.build(write=True).run(runner=runner)
        by_stem = {Path(argv[1]).stem: argv for argv in runner.calls}
        self.assertIn("--write", by_stem["bake"])
        self.assertIn("--write", by_stem["fit"])
        self.assertNotIn("--check", by_stem["fetch"])
        self.assertEqual(by_stem["descriptor"][2:], ["ur3e"])

    def test_the_first_failure_stops_the_build(self) -> None:
        """THE CONTROL. A descriptor built after a failed fit would carry the spheres of the last good one."""
        runner = _Runner(fail="fit")
        report = self.build(write=True, hand="robotiq_hande", profile="ur3e").run(runner=runner)
        self.assertEqual([Path(argv[1]).stem for argv in runner.calls], ["fetch", "bake", "fit"])
        self.assertEqual(report.failed.name, "fit")
        self.assertEqual(report.exit_code, 1)
        self.assertEqual([stage.exit_code for stage in report.stages[3:]], [None, None, None])
        self.assertIn("stopped at fit, exit 1", report.render())

    def test_without_a_hand_or_a_cell_the_check_and_the_doctor_say_why(self) -> None:
        runner = _Runner()
        report = self.build(write=True).run(runner=runner)
        skipped = {stage.name: stage.skipped for stage in report.stages if stage.skipped}
        self.assertEqual(sorted(skipped), ["check", "doctor"])
        self.assertIn("--hand", skipped["check"])
        self.assertIn("--profile", skipped["doctor"])
        self.assertEqual(len(runner.calls), 4)

    def test_a_cell_that_names_its_hand_gives_it_to_the_check_and_the_doctor(self) -> None:
        stages = {stage.name: stage for stage in self.build(profile="ur3e,hande").stages()}
        self.assertEqual(stages["check"].argv[-2:], ("--hand", "robotiq_hande"))
        self.assertEqual(stages["doctor"].argv[-4:], ("--profile", "ur3e,hande", "--hand", "robotiq_hande"))

    def test_a_hand_that_contradicts_the_cell_is_refused_before_anything_runs(self) -> None:
        """The doctor's rows read the cell's hand, so a check planned with another would prove the wrong pair."""
        with self.assertRaises(ConfigError) as caught:
            self.build(profile="ur3e,hande", hand="robotiq_2f85")
        self.assertIn("robotiq_hande", str(caught.exception))
        self.assertIn("robotiq_2f85", str(caught.exception))

    def test_the_doctor_reads_the_cell_it_is_named_and_is_given_no_model(self) -> None:
        doctor = self.build(hand="robotiq_hande", profile="ur3e").stages()[-1]
        self.assertEqual(doctor.argv[1:], ("-m", "src.robot.safety.planning", "--doctor",
                                           "--profile", "ur3e", "--hand", "robotiq_hande"))
        self.assertNotIn("--model", doctor.argv)

    def test_without_a_profile_the_doctor_says_why(self) -> None:
        """THE CONTROL. Handed a model and a hand the doctor loads no cell and answers "missing" for every arm:
        the first run of this script in aurora (2026-09-23) stopped a healthy ur5e there, exit 1."""
        doctor = self.build(hand="robotiq_hande").stages()[-1]
        self.assertEqual(doctor.argv, ())
        self.assertIn("--profile", doctor.skipped)

    def test_a_profile_for_another_arm_is_refused_before_anything_runs(self) -> None:
        with self.assertRaises(ConfigError) as caught:
            self.m.ArmBuild.from_model("ur5e", profile="ur3e", curobo_python=self.cu)
        self.assertIn("ur3e", str(caught.exception))
        self.assertIn("ur5e", str(caught.exception))

    def test_a_missing_curobo_interpreter_refuses_before_anything_runs(self) -> None:
        runner = _Runner()
        build = self.m.ArmBuild.from_model("ur5e", python="project-python",
                                           curobo_python=str(Path(self.folder.name) / "absent.exe"))
        report = build.run(runner=runner)
        self.assertEqual(runner.calls, [])
        self.assertIn("absent.exe", report.refused)
        self.assertEqual(report.exit_code, 1)

    def test_an_arm_no_description_names_is_refused_before_anything_runs(self) -> None:
        with self.assertRaises(ConfigError):
            self.m.ArmBuild.from_model("ur99", curobo_python=self.cu)

    def test_the_report_describes_itself(self) -> None:
        report = self.build(hand=UNSET).run(runner=_Runner())
        text = report.render()
        self.assertTrue(text.isascii())
        self.assertFalse(text.endswith("\n"))
        self.assertEqual(report.to_dict()["stages"][0]["name"], "fetch")
        self.assertEqual(report.to_dict()["exit_code"], report.exit_code)


if __name__ == "__main__":
    unittest.main()
