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
keep a calibration sweep's stations spread out, and that is untouched.
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
        a mock that would pass whichever method the driver happened to call.

        Matched on the call, not on the name of its argument: the UR controller path boxes `boxed`, the
        grasp centre whenever its arm passes one as `workspace_pose`, and a check spelled
        `validate(pose)` would have let `validate(boxed)` straight through. The box call is looked for
        in the method that commands the move, because `move_home` carries one too and would answer for
        a `move_to` that had lost its own; `validate` is refused anywhere in the class."""
        from src.robot.drivers.kuka import arm as kuka_arm
        from src.robot.drivers.ur import motion as ur_motion

        for cls, commanding in ((ur_motion.MotionController, "move_to"),
                                (kuka_arm.KukaRobotArm, "_drive_pose")):
            with self.subTest(driver=cls.__name__):
                self.assertRegex(inspect.getsource(getattr(cls, commanding)), r"guard\.is_inside_workspace\(")
                self.assertNotRegex(inspect.getsource(cls), r"guard\.validate\(")

    def test_the_calibration_station_screen_keeps_its_own_diversity_guard(self) -> None:
        """The rule is not deleted -- it is back where it belongs. pose_provider builds a LOCAL guard per
        sweep so the stations it commands stay spread out, and nothing generates a station any more (the
        owner, 2026-09-24): the screen judges the stations somebody wrote down."""
        from src.geometry import Pose
        from src.robot.execution import pose_provider

        box = RobotConfig().workspace_limits
        x, y, z = (float(box.x_min + box.x_max) / 2, float(box.y_min + box.y_max) / 2, float(box.z_min + box.z_max) / 2)
        station, near = Pose.tool_down(x, y, z, label="kept"), Pose.tool_down(x + 10.0, y, z, label="near")
        shared = WorkspaceGuard(box, min_distance_mm=40.0, min_angle_deg=10.0)
        provider = pose_provider.PoseProvider(shared)
        screen = provider.screen()
        self.assertIsNone(screen.refusal(station))
        screen.keep(station)
        refused = screen.refusal(near)
        assert refused is not None
        self.assertEqual(refused[0], "too_similar")
        self.assertIsNone(provider.screen().refusal(near), "a second sweep starts with an empty history")
        self.assertEqual(shared.accepted_poses, [], "the guard handed in is never the one that remembers")


if __name__ == "__main__":
    unittest.main()
