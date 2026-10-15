"""Every route holds the caller's speed to robot.motion_limits, the cuRobo route too (P0e E12).

The joint drive and the ik route clamp a caller's ``vel`` and ``acc`` to ``robot.motion_limits`` before the
controller sees them (``clamp_to_motion_limits``). The cuRobo route did not: ``move(pose, vel=5.0)`` on a cuRobo cell
sent every waypoint at 5 rad/s whatever the cell declared, found on the P0e map (``curobo_motion.py`` execute). A limit
a caller can talk past is not a limit.

Since P0g slice 13 the arm is a planned arm over the fake transport, built here; it borrowed the UR tool-frame file's
builder by importing that file's TestCase class, which made pytest collect the class a second time in this file.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock

from tests._plan_end import pose_where_it_ends

_DECLINED = "unit double: this test exercises the planner on a cuRobo arm, and no camera world is wired to it"
_Q = [0.0, -1.2, 1.3, -0.4, 1.5, 0.2]
#: The 2F-85 on a flange as the Isaac driver hardcodes it: 132 mm along flange +Y, -90 deg about flange X.
_2F85 = {"offset_mm": (0.0, 132.0, 0.0), "rotation_quat_xyzw": (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)}


def _CUROBO_ARM(source: str) -> tuple[Any, MagicMock]:  # noqa: N802 (the name its callers used)
    """A cuRobo planned arm on tool frame ``source``, whose controller reports the flange at ``_Q`` whatever it is
    asked, its planner a MagicMock planning one waypoint at ``_Q``."""
    import numpy as np

    from src.geometry import Frame, Pose
    from src.robot.safety._ur_kinematics import ur_link_transforms_mm
    from tests._arm_kit import kit_tree, planned_arm
    from tests._fake_transport import fake_transport
    from tests._plan_end import OPEN_WORKSPACE

    config = kit_tree("curobo", workspace_limits=OPEN_WORKSPACE, safety={"payload": {"enforce": False}},
                      gripper={"tool_frame": {"source": source, **_2F85}})
    links = ur_link_transforms_mm("ur5e", np.asarray(_Q, dtype=np.float64))
    assert links is not None
    fake = fake_transport(joints=_Q)
    fake.always("controller_fk", Pose.from_matrix(links[-1], frame=Frame.BASE))
    arm = planned_arm(config, transport=fake)
    planner = MagicMock()
    planner.plan.return_value = [list(_Q)]
    arm._curobo = planner
    return arm, planner


class TheCuroboRouteIsHeldToTheLimitsTests(unittest.TestCase):
    def test_a_caller_asking_for_more_is_held_to_the_configured_limits(self) -> None:
        arm, planner = _CUROBO_ARM("willy")
        with arm.without_camera_world(_DECLINED):
            arm.move(pose_where_it_ends(arm, _Q), vel=5.0, acc=9.0)
        limits = arm.config.motion_limits
        self.assertEqual((limits.max_velocity, limits.max_acceleration),
                         (planner.execute.call_args.kwargs["vel"], planner.execute.call_args.kwargs["acc"]))

    def test_a_caller_asking_for_less_keeps_what_it_asked_for(self) -> None:
        arm, planner = _CUROBO_ARM("willy")
        with arm.without_camera_world(_DECLINED):
            arm.move(pose_where_it_ends(arm, _Q), vel=0.2, acc=0.1)
        self.assertEqual((0.2, 0.1), (planner.execute.call_args.kwargs["vel"], planner.execute.call_args.kwargs["acc"]))

    def test_a_joint_move_planned_around_its_line_is_held_to_the_limits_too(self) -> None:
        """A joint move whose straight line collides runs cuRobo's plan through the glue, clamped as the Cartesian route
        is."""
        from src.robot.core import JointPositions, MotionCommand, MotionResult
        from src.robot.safety.planning import JointCheckVerdict

        arm, planner = _CUROBO_ARM("willy")
        target = [_Q[0] + 0.1, *_Q[1:]]

        def check(samples: Any, *, refresh: bool = True, clearance_mm: float = 0.0) -> JointCheckVerdict:
            count = len(list(samples))
            if clearance_mm > 0.0:
                return JointCheckVerdict(valid=False, first_invalid=0, checked=count, reason="the line grazes the tote")
            return JointCheckVerdict(valid=True, first_invalid=None, checked=count, reason="clear")

        planner.check_joint_path.side_effect = check
        planner.plan_joint.return_value = [list(_Q), target]
        planner.last_refusal = None
        planner.execute.return_value = MotionResult.executed(MotionCommand.MOVE_JOINTS, message="curobo")
        with arm.without_camera_world(_DECLINED):
            result = arm.move_to_joints(JointPositions(target), velocity=5.0, acceleration=9.0)
        self.assertTrue(result.ok, result.message)
        limits = arm.config.motion_limits
        self.assertEqual((limits.max_velocity, limits.max_acceleration),
                         (planner.execute.call_args.kwargs["vel"], planner.execute.call_args.kwargs["acc"]))

    def test_a_caller_asking_for_nothing_leaves_the_planners_default(self) -> None:
        arm, planner = _CUROBO_ARM("willy")
        with arm.without_camera_world(_DECLINED):
            arm.move(pose_where_it_ends(arm, _Q))
        self.assertEqual((None, None), (planner.execute.call_args.kwargs["vel"], planner.execute.call_args.kwargs["acc"]))


if __name__ == "__main__":
    unittest.main()
