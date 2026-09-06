"""The pose-DIVERSITY check belongs to calibration sampling, not to the motion path.

MEASURED 2026-08-09: `WorkspaceGuard.validate()` = workspace box AND "this pose is too similar to one
already accepted", and both real vendor drivers called `validate()` on every commanded move while
`accept()` accumulated for the guard's whole lifetime. Driven with the ordinary shape of a pick:

    standoff  ACCEPTED
    descend   ACCEPTED
    retreat   REJECTED   "too similar to 'standoff' (dist=0.0 mm, angle=0.0 deg)"

The arm grasps and refuses to LIFT -- on every pick, on both drivers. And the symptom reads as a
gripper, planner or calibration fault, which is the expensive kind of bug: it looks like anything except
a sampling rule that leaked out of the calibration routine.

The rule itself is sound where it belongs: `execution/pose_provider.py` builds its OWN local guard to
keep auto-generated calibration poses spread out, and that is untouched.
"""

from __future__ import annotations

import inspect
import unittest

from src.config.schema.robot import RobotConfig
from src.robot.drivers.ur.pose import URPose
from src.robot.safety.workspace import WorkspaceGuard


def _pose(z: float, label: str) -> URPose:
    # A top-down pick keeps ONE orientation throughout, which is why the angle term never saves it.
    return URPose(x=400.0, y=0.0, z=z, rx=0.0, ry=3.14159, rz=0.0, label=label)


_PICK = (("standoff", 300.0), ("descend", 250.0), ("retreat", 300.0))


class DiversityIsNotOnTheMotionPathTests(unittest.TestCase):
    def test_a_pick_sequence_survives_the_box_check(self) -> None:
        """Twice through, because `accept()` accumulates across runs -- a cell does more than one pick."""
        guard = WorkspaceGuard(RobotConfig().workspace_limits)
        for run in range(2):
            for label, z in _PICK:
                with self.subTest(run=run, waypoint=label):
                    self.assertTrue(guard.is_inside_workspace(_pose(z, label)))

    def test_the_diversity_rule_still_rejects_that_same_retreat(self) -> None:
        """Pinning WHAT was removed from the motion path, not just that something was. If this ever
        stops rejecting, the finding above has been silently undone rather than relocated."""
        guard = WorkspaceGuard(RobotConfig().workspace_limits)
        guard.accept(_pose(300.0, "standoff"))
        guard.accept(_pose(250.0, "descend"))
        self.assertFalse(guard.is_diverse_enough(_pose(300.0, "retreat")))

    def test_neither_real_driver_calls_validate_on_the_motion_path(self) -> None:
        """The two vendor drivers that command real hardware. Source-level, because the alternative is
        a mock that would pass whichever method the driver happened to call."""
        from src.robot.drivers.kuka import arm as kuka_arm
        from src.robot.drivers.ur import motion as ur_motion

        for mod, fn in ((ur_motion, "move_to"), (kuka_arm, "move_to")):
            with self.subTest(module=mod.__name__):
                src = inspect.getsource(getattr(mod, "MotionController", getattr(mod, "KukaRobotArm", None)))
                self.assertIn("is_inside_workspace(pose)", src)
                self.assertNotIn("guard.validate(pose)", src.replace("_guard.validate(pose)", "guard.validate(pose)"))

    def test_calibration_pose_generation_keeps_its_own_diversity_guard(self) -> None:
        """The rule is not deleted -- it is back where it belongs. pose_provider builds a LOCAL guard so
        auto-generated calibration poses stay spread out."""
        from src.robot.execution import pose_provider

        src = inspect.getsource(pose_provider)
        self.assertIn("local_guard.validate(", src)
        self.assertIn("local_guard.accept(", src)


if __name__ == "__main__":
    unittest.main()
