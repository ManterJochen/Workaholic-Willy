"""L8.2 — CalibrationRoutine now threads motion_limits through RobotArm.move(vel=, acc=).

The ctor accepted ``motion_limits`` but never used it. It is now commanded on the typed motion path; with
no limits the call is unchanged (driver default). _move_to_pose only needs arm / _motion_limits / logger,
so these construct the routine via ``object.__new__`` to keep the test focused.
"""

from __future__ import annotations

import logging
import unittest

import numpy as np

from src.config.schema.robot.robot_schema import MotionLimitsConfig
from src.geometry import Frame, Pose
from src.robot.core.motion_result import MotionCommand, MotionResult
from src.robot.execution.calibration import CalibrationRoutine


class _RecordingArm:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def move(self, pose: Pose, **kwargs: object) -> MotionResult:
        self.calls.append(dict(kwargs))
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)


def _routine(motion_limits: MotionLimitsConfig | None) -> CalibrationRoutine:
    routine = object.__new__(CalibrationRoutine)
    routine.arm = _RecordingArm()  # type: ignore[attr-defined]
    routine._motion_limits = motion_limits  # type: ignore[attr-defined]
    routine.logger = logging.getLogger("test_calibration_motion_limits")  # type: ignore[attr-defined]
    return routine


def _pose() -> Pose:
    return Pose(
        position_mm=np.array([400.0, 0.0, 300.0]),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        frame=Frame.BASE,
        label="p",
    )


class CalibrationMotionLimitsTests(unittest.TestCase):
    def test_limits_are_threaded_into_move(self) -> None:
        routine = _routine(MotionLimitsConfig(max_velocity=2.0, max_acceleration=1.0))
        self.assertTrue(routine._move_to_pose(_pose()))
        call = routine.arm.calls[-1]  # type: ignore[attr-defined]
        self.assertEqual(call.get("vel"), 2.0)
        self.assertEqual(call.get("acc"), 1.0)
        self.assertIs(call.get("register"), False)

    def test_no_limits_keeps_driver_default(self) -> None:
        routine = _routine(None)
        self.assertTrue(routine._move_to_pose(_pose()))
        call = routine.arm.calls[-1]  # type: ignore[attr-defined]
        self.assertNotIn("vel", call)
        self.assertNotIn("acc", call)


if __name__ == "__main__":
    unittest.main()
