"""A station file is read against the range the arm's description rates each joint for (the owner, 2026-09-25: the
bound per joint from the description, where 85f082b's was a full turn on every arm).

``pose_provider.load_stations`` refused a ``joints_rad`` value past one full turn as degrees and a ``joints_deg`` value
past 360 as beyond any joint, and a :class:`JointStation` built in code past a full turn: right for every UR, whose
description rates each joint a full turn either way, and wrong for an arm rated otherwise. A file is read against the
rating of the arm that runs the sweep now, the bound the planned arm reads a joint target against
(``PlannedArm._read_as_degrees``), and a station built in code holds what it is given: the arm judges it against the
same rating when it moves.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from src.config.schema.robot import RobotConfig
from src.contracts import UNSET
from src.robot.core import JointPositions
from src.robot.execution.pose_provider import JointStation, load_stations, rated_joints_deg

_ROOT = Path(__file__).resolve().parents[1]
#: What every UR's description rates each of its six joints for.
_UR = ((-360.0, 360.0),) * 6
#: A wrist rated narrower than a turn, and one rated past it: no arm described today, the shape another vendor has.
_NARROW_WRIST = _UR[:5] + ((-170.0, 170.0),)
_WIDE_WRIST = _UR[:5] + ((-400.0, 400.0),)
_HOME_DEG = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]


class _Stopped(Exception):
    """Raised by a patched reader once it has recorded what it was asked with."""


class TheRatingIsTheArmsTests(unittest.TestCase):

    def test_every_ur_description_rates_each_joint_a_full_turn_either_way(self) -> None:
        models = sorted(path.stem for path in (_ROOT / "config" / "arms").glob("*.yaml"))
        self.assertIn("ur5e", models)
        for model in models:
            with self.subTest(model):
                self.assertEqual(rated_joints_deg(model), _UR)

    def test_an_arm_with_no_description_names_no_rating(self) -> None:
        self.assertIs(rated_joints_deg("dummy-6dof"), UNSET)
        self.assertIs(rated_joints_deg(""), UNSET)


class AFileIsReadAgainstTheRatingTests(unittest.TestCase):

    def setUp(self) -> None:
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def _file(self, record: dict[str, Any]) -> Path:
        path = self.folder / "stations.json"
        path.write_text(json.dumps([record]), encoding="utf-8")
        return path

    def test_a_joint_rated_narrower_than_a_turn_refuses_a_value_past_its_rating(self) -> None:
        degrees = self._file({"label": "w", "joints_deg": [*_HOME_DEG[:5], 200.0]})
        self.assertEqual(len(load_stations(degrees)), 1)
        with self.assertRaisesRegex(ValueError, r"joints_deg gives wrist 3 as 200, past -170 to 170 deg, the range the "
                                                r"arm's description rates the joint for"):
            load_stations(degrees, rated_deg=_NARROW_WRIST)
        radians = self._file({"label": "w", "joints_rad": [0.0, -1.57, 1.57, -1.57, -1.57, 3.2]})
        self.assertEqual(len(load_stations(radians)), 1)
        with self.assertRaisesRegex(ValueError, r"joints_rad gives wrist 3 as 3.2, past -170 to 170 deg, .*reads as "
                                                r"degrees"):
            load_stations(radians, rated_deg=_NARROW_WRIST)

    def test_a_joint_rated_past_a_turn_takes_a_value_past_a_turn(self) -> None:
        degrees = self._file({"label": "w", "joints_deg": [*_HOME_DEG[:5], 380.0]})
        with self.assertRaisesRegex(ValueError, r"wrist 3 as 380, past -360 to 360 deg, one full turn either way"):
            load_stations(degrees)
        (station,) = load_stations(degrees, rated_deg=_WIDE_WRIST)
        self.assertAlmostEqual(float(station.degrees[5]), 380.0, places=9)
        radians = self._file({"label": "w", "joints_rad": [0.0, -1.57, 1.57, -1.57, -1.57, 6.6]})
        with self.assertRaisesRegex(ValueError, r"wrist 3 as 6.6, past -360 to 360 deg, .*reads as degrees"):
            load_stations(radians)
        (station,) = load_stations(radians, rated_deg=_WIDE_WRIST)
        self.assertEqual(float(station.joints.values[5]), 6.6)

    def test_a_joint_station_built_in_code_holds_what_it_is_given(self) -> None:
        """The arm judges it: past its rating, the planned arm's destination gate refuses it as degrees."""
        station = JointStation(JointPositions([0.0, -90.0, 0.0, 0.0, 0.0, 0.0]), label="deg")
        self.assertEqual(float(station.joints.values[1]), -90.0)


class TheReadersNameTheirArmTests(unittest.TestCase):

    def setUp(self) -> None:
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path = self.folder / "stations.json"
        self.path.write_text(json.dumps([{"label": "a", "joints_deg": _HOME_DEG}]), encoding="utf-8")
        self.asked: list[dict[str, Any]] = []

    def _reader(self, *args: Any, **kwargs: Any) -> Any:
        self.asked.append(kwargs)
        raise _Stopped

    def test_the_routine_reads_a_file_against_the_arm_it_drives(self) -> None:
        from tests.test_calibration_joint_stations import _Arm, _routine

        arm = _Arm({})
        arm.capabilities = SimpleNamespace(model="ur3e")  # type: ignore[attr-defined]
        routine, _ = _routine(arm)
        with patch("src.robot.execution.calibration.load_stations", self._reader), \
                self.assertRaises(_Stopped):
            routine.run_from_json(self.path)
        self.assertEqual(self.asked, [{"rated_deg": rated_joints_deg("ur3e")}])

    def test_the_noun_reads_a_file_against_the_arm_its_tree_names(self) -> None:
        from tests.test_camera_boundaries import _rgbd_rig
        from src.config.schema.camera import HandEyeConfig
        from src.robot.execution.hand_eye import HandEyeCalibration, SweepOptions

        app = SimpleNamespace(
            robot=RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": "10.253.253.41", "model": "ur3e"}}),
            camera=SimpleNamespace(cameras=SimpleNamespace(rigs=[_rgbd_rig("overhead")]), hand_eye=HandEyeConfig()))
        noun = HandEyeCalibration.from_config(app, rig_id="overhead", mode="eye_to_hand",
                                              options=SweepOptions(fixed_poses=self.path))
        with patch("src.robot.execution.pose_provider.load_stations", self._reader), \
                self.assertRaises(_Stopped):
            noun._own_stations()
        self.assertEqual(self.asked, [{"rated_deg": rated_joints_deg("ur3e")}])


if __name__ == "__main__":
    unittest.main()
