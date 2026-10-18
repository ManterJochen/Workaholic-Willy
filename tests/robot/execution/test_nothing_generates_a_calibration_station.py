"""Nothing generates a calibration station, and a sweep that names none is refused.

The owner, 2026-09-24: every automatic station generator goes. What is left is a person's stations, a file or a
list of poses and joint stations (``fixed_poses``, ``--fixed-poses``) and the arm guided by hand (``freedrive``,
``--freedrive``). The generated sweep (``CalibrationRoutine.run_auto``, ``PoseProvider.generate``, ``order_by_bearing``,
the pose count ``SweepOptions.poses`` and ``--poses``) is gone, and so is the aimed sweep that planned
its own views round a marker (``CalibrationRoutine.run_aimed``, ``camera_aim``, ``SweepOptions.aim``, ``--aim-at``).
This file is the guard that both stay gone: a generator that came back would run poses nobody wrote down, in a cell
nobody measured them for.

Every hint that tells a person how to calibrate names ``--freedrive``, the way that needs no stations file.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[3]
#: The names of the generated sweep and of the aimed one; none may be defined or called anywhere in the code.
_GENERATORS = ("run_auto", "order_by_bearing", "generate_hemisphere_viewpoints", "DEFAULT_POSES", "_MIN_SPREAD_DEG",
               "run_aimed", "AimedSweep", "MarkerAim", "SkippedStation")


def _app() -> SimpleNamespace:
    from src.config.schema.camera import HandEyeConfig
    from src.config.schema.robot import RobotConfig
    from tests.camera.setup.image_taking.test_camera_boundaries import _rgbd_rig

    return SimpleNamespace(
        robot=RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": "10.253.253.51"}}),
        camera=SimpleNamespace(cameras=SimpleNamespace(rigs=[_rgbd_rig("overhead")], primary_rig_id="overhead"),
                               hand_eye=HandEyeConfig()))


class TheGeneratorsAreGoneTests(unittest.TestCase):

    def test_the_routine_and_the_pose_provider_generate_nothing(self) -> None:
        from src.robot.execution import pose_provider
        from src.robot.execution.calibration import CalibrationRoutine
        from src.robot.execution.pose_provider import PoseProvider

        self.assertFalse(hasattr(CalibrationRoutine, "run_auto"))
        self.assertFalse(hasattr(PoseProvider, "generate"))
        self.assertFalse(hasattr(pose_provider, "order_by_bearing"))
        self.assertFalse(hasattr(CalibrationRoutine, "run_aimed"))
        self.assertFalse(hasattr(pose_provider, "SkippedStation"))
        self.assertIsNone(importlib.util.find_spec("src.robot.execution.camera_aim"))

    def test_the_options_and_the_noun_count_no_generated_poses(self) -> None:
        from src.robot.execution import hand_eye
        from src.robot.execution.hand_eye import HandEyeCalibration, SweepOptions

        for noun in (SweepOptions, HandEyeCalibration):
            self.assertFalse({"poses", "aim"} & {f.name for f in dataclasses.fields(noun)}, noun.__name__)
        self.assertFalse(hasattr(hand_eye, "DEFAULT_POSES"))

    def test_the_command_line_takes_no_count_of_generated_poses(self) -> None:
        from src.robot.execution.real_cell import calibrate

        flags = {option for action in calibrate.build_parser()._actions for option in action.option_strings}  # noqa: SLF001
        self.assertFalse({"--poses", "--aim-at", "--aim-distance-mm", "--closing-axis"} & flags)
        self.assertIn("--fixed-poses", flags)
        self.assertIn("--freedrive", flags)

    def test_no_code_defines_or_calls_a_generator(self) -> None:
        """A walk over every module a customer runs: the names of the generated sweep, defined or reached for."""
        found: list[str] = []
        for top in ("src", "api", "willy", "scripts", "examples"):
            for path in sorted((_ROOT / top).rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    name = (node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else
                            node.attr if isinstance(node, ast.Attribute) else
                            node.id if isinstance(node, ast.Name) else "")
                    if name in _GENERATORS:
                        found.append(f"{path.relative_to(_ROOT).as_posix()}:{node.lineno}: {name}")
        self.assertEqual(found, [])


class ASweepNamesItsStationsTests(unittest.TestCase):

    def test_a_sweep_that_names_no_stations_and_no_hands_is_refused_at_the_check(self) -> None:
        from src.robot.execution.hand_eye import CalibrationOutcome, HandEyeCalibration

        calibration = HandEyeCalibration.from_config(_app(), rig_id="overhead", mode="eye_to_hand")
        check = calibration.check()
        self.assertFalse(check.ok)
        for said in ("a sweep needs its stations and none were named", "--fixed-poses PATH", "--freedrive",
                     "Nothing generates stations"):
            self.assertIn(said, check.refusal)
        with patch("src.robot.execution.robot.Robot.from_config",
                   side_effect=AssertionError("a sweep with no stations built the arm")):
            self.assertIs(calibration.run().outcome, CalibrationOutcome.REFUSED)

    def test_the_command_line_refuses_the_same_before_anything_is_built(self) -> None:
        from src.robot.execution.real_cell import calibrate

        printed = io.StringIO()
        with patch.object(calibrate, "_load", return_value=_app()), redirect_stdout(printed):
            code = calibrate.main(["--rig", "overhead", "--check"])
        self.assertEqual(code, calibrate._EXIT_CONFIG, printed.getvalue())  # noqa: SLF001
        self.assertIn("[config] REFUSED: a sweep needs its stations and none were named", printed.getvalue())

    def test_a_list_a_file_or_the_hands_each_name_the_stations(self) -> None:
        from src.geometry import Pose
        from src.robot.execution.hand_eye import HandEyeCalibration, SweepOptions

        for options in (SweepOptions(fixed_poses=[Pose.tool_down(400.0, 0.0, 350.0)]), SweepOptions(freedrive=True)):
            with self.subTest(options=options):
                check = HandEyeCalibration.from_config(_app(), rig_id="overhead", mode="eye_to_hand",
                                                       options=options).check()
                self.assertTrue(check.ok, check.refusal)


class TheGeneratorKeysLeftTheSchemaTests(unittest.TestCase):
    """The keys that only shaped generated poses left the schema, and a tree written before still writes them:
    refused at load with the sentence that says why, never ignored (``_removed.py``)."""

    _GONE = {
        "robot.calibration.orientation_spread_deg": "robot:\n  calibration:\n    orientation_spread_deg: 15.0\n",
        "robot.calibration.max_attempts_per_pose": "robot:\n  calibration:\n    max_attempts_per_pose: 200\n",
        "robot.calibration.pose_box_z_frac":
            "robot:\n  calibration:\n    pose_box_z_frac:\n      ground_truth: [0.5, 0.9]\n",
        "robot.sim.scene_setup.eih_viewpoints":
            "robot:\n  sim:\n    scene_setup:\n      eih_viewpoints: { radii_mm: [235.0] }\n",
    }

    def test_the_schema_holds_none_of_them(self) -> None:
        from src.config.schema.robot import RobotCalibrationConfig
        from src.config.schema.robot.sim_schema import SimSceneConfig

        for name in ("orientation_spread_deg", "max_attempts_per_pose", "pose_box_z_frac"):
            self.assertNotIn(name, RobotCalibrationConfig.model_fields)
        self.assertNotIn("eih_viewpoints", SimSceneConfig.model_fields)

    def test_each_has_the_sentence_that_says_where_it_went(self) -> None:
        from src.config.schema._removed import removed_key_sentence

        for key in self._GONE:
            with self.subTest(key):
                said = removed_key_sentence(key) or ""
                self.assertIn("nothing generates a calibration station any more", said)
                self.assertIn("Delete this key", said)

    def test_a_tree_that_still_writes_one_is_refused_with_its_sentence(self) -> None:
        import shutil
        import tempfile

        from src.config import load_config
        from src.config.loader import ConfigError, reload_config, set_active_profile
        from src.config.schema._removed import removed_key_sentence

        folder = Path(self.enterContext(tempfile.TemporaryDirectory())) / "data"
        shutil.copytree(_ROOT / "config", folder)
        self.addCleanup(reload_config)
        self.addCleanup(set_active_profile, None)
        for index, (key, layer) in enumerate(self._GONE.items()):
            with self.subTest(key):
                (folder / "robot" / f"robot.gone{index}.yaml").write_text(layer, encoding="utf-8")
                set_active_profile(f"sim,gone{index}")
                reload_config()
                with self.assertRaises(ConfigError) as caught:
                    load_config(folder)
                message = str(caught.exception)
                self.assertIn(key, message)
                self.assertIn("removed on purpose", message)
                self.assertIn(removed_key_sentence(key) or "<no sentence>", message)


class EveryHintNamesTheWayByHandTests(unittest.TestCase):
    """Where the code tells a person to run the calibration, it names ``--freedrive``: without it the command it
    prints is refused now."""

    def test_the_hints_that_print_the_command_name_freedrive(self) -> None:
        sources = {
            "src/calibration/rig_calibration.py": 4,
            "src/config/schema/camera/shared_schema.py": 3,
            "src/robot/execution/real_cell/preflight.py": 1,
            "src/robot/execution/wrist_bodies.py": 1,
        }
        for rel, count in sources.items():
            with self.subTest(rel):
                text = (_ROOT / rel).read_text(encoding="utf-8")
                self.assertGreaterEqual(text.count("--freedrive"), count, rel)

    def test_the_remedy_of_a_missing_calibration_names_it(self) -> None:
        from src.calibration.rig_calibration import _sweep_remedy

        said: Any = _sweep_remedy("overhead", "eye_to_hand")
        self.assertIn("--rig overhead --mode eye_to_hand --freedrive", said)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
