"""A grasp turned the cell's natural way round with no configuration on the arm's branch inside the cable window takes
its twin, half a turn about its approach, before it waits for its branch or the arm changes one (the owner's window gap,
2026-10-08: "passiert eh schon öfter").

``robot.natural_closing_axis: "-y"`` stays: with it the wrist camera faces -y, onto the work area, parallel to the mat,
and the owner set it on purpose. But that rule chooses one of a grasp's two equivalent wrist turns for every grasp, and
inside the half-turn cable window the one it chooses can have no configuration on the branch the arm holds. On the cell
(2026-10-07, 11:19, ``case1007.py`` of the stuck map) the natural turn of a grasp's line down needed wrist 3 at +90
degrees; the window about that day's home (wrist 3 at -88.6) ends at +86.4, so only shoulder + goals were offered and try
10 judged a shoulder and elbow flip, which the operator's halt stopped. Its twin, wrist 3 at -90 and the largest joint
102 degrees, was on the arm's branch.

The pick loop now asks the arm, on the closed form alone (``URRobotArm.has_a_goal_on_its_branch``: no screen, no
planner, no controller call), whether each grasp's standoff has a configuration on the arm's branch inside the window,
and takes the twin where the natural turn has none and the twin has one. Pinned here, on the 2026-10-07 numbers:

* the arm's answer, both ways round, and that it asks nothing of the controller;
* the loop takes the twin there; the same grasp from the bench's home of config B keeps its natural turn, though the twin
  turns wrist 3 less (R1, the shorter turn, is not built: "-y" stays wherever it has a configuration on the branch);
* the wrist camera's distance still decides after it, and ``both_faces`` and a closing axis a program names keep their
  own rules.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.config.schema.robot import RobotConfig
from src.geometry import Frame, Pose
from src.geometry.closing_axis import closing_axis_of
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.grasping.collision.gripper_model import ParallelJawGripperModel
from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.safety._ur_ik import nearest_goals, ur_branch, ur_flange_ik
from src.robot.safety._ur_kinematics import ur_link_transforms_mm
from tests._plan_end import OPEN_WORKSPACE
from tests.test_a_grasp_turns_its_wrist_camera_off_the_wall import _HOUSING, _camera_at, _wall

#: The home of 2026-10-07 and the bench's home of config B, degrees.
HOME_1007 = (-88.6, -52.5, -91.2, -127.7, 90.2, -88.6)
HOME_B = (-75.10, -83.00, -72.40, -137.31, 96.02, -76.28)
#: Sample 129 of 144 of the line down judged ahead on the cell, 2026-10-07 11:19:33, radians.
_LINE_SAMPLE = (1.2249, -1.3010, 1.9823, 0.6792, 1.4200, -2.2020)
#: The cell's flange to TCP: 157 mm along the flange axis, turned -90 degrees about it.
_TOOL = {"source": "polyscope", "offset_mm": [0.0, 0.0, 157.0], "rotation_quat_xyzw": [0.0, 0.0, -0.7071068, 0.7071068]}
_STANDOFF_MM = 80.0


def _radians(degrees: "tuple[float, ...]") -> list[float]:
    return [math.radians(v) for v in degrees]


def _ur10(home_deg: "tuple[float, ...]") -> URRobotArm:
    """The cell's UR10 on cuRobo with the cable window about ``home_deg``, on a controller double that records calls."""
    arm = URRobotArm(RobotConfig.model_validate({
        "vendor": "ur", "ur": {"model": "ur10", "motion_planner": "curobo"},
        "safety": {"payload": {"enforce": False}, "joint_limits": {"within_half_turn_of_home": True}},
        "gripper": {"model": "robotiq_2f85", "tool_frame": _TOOL}, "workspace_limits": OPEN_WORKSPACE,
        "home_joint_positions": _radians(home_deg),
    }))
    arm._conn = MagicMock()
    arm._conn.is_connected = True
    arm._conn.get_joint_positions.return_value = _radians(home_deg)
    return arm


def _natural_standoff(arm: URRobotArm) -> Pose:
    """The TCP where the line sample put it, the grasp's natural turn: tool +X nearer -y."""
    flange = np.asarray(ur_link_transforms_mm("ur10", np.asarray(_LINE_SAMPLE))[-1], dtype=np.float64)
    return Pose.from_matrix(flange @ arm._declared_tool_matrix(), frame=Frame.BASE, label="natural")


def _twin(pose: Pose) -> Pose:
    half = np.diag([-1.0, -1.0, 1.0, 1.0])
    return Pose.from_matrix(np.asarray(pose.to_matrix(), dtype=np.float64) @ half, frame=Frame.BASE, label="twin")


def _grasp(standoff: Pose) -> GraspPoint:
    """The grasp whose standoff, ``_STANDOFF_MM`` back along its approach, is ``standoff``, closing along its tool +X."""
    matrix = np.asarray(standoff.to_matrix(), dtype=np.float64)
    approach = matrix[:3, 2]
    return GraspPoint(position=matrix[:3, 3] + approach * _STANDOFF_MM, approach=approach, axis=matrix[:3, 0].copy(),
                      grip_width_mm=40.0, score=0.9, frame=GraspFrame.BASE, label="part")


def _loop(arm: URRobotArm, *, both_faces: bool = False, closing_axis: Any = None) -> Any:
    """The pick loop as ``_closing_along`` reads it: its arm, the real policy, the owner's natural "-y"."""
    from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator  # noqa: PLC0415

    loop = object.__new__(BinPickingOrchestrator)
    loop.arm = arm
    loop.policy = GraspExecutionPolicy(arm=arm, standoff_mm=_STANDOFF_MM, closing_axis=closing_axis)
    loop.natural_closing_axis = closing_axis_of("-y")
    loop.both_faces = both_faces
    loop.gripper_model = ParallelJawGripperModel(finger_width_mm=29.24, pad_length_mm=20.91, pad_ahead_mm=10.45)
    loop._wrist_housing_read = True
    loop._wrist_housing_cached = None
    loop.calculator_for = lambda camera_id: None
    return loop


def _result(*grasps: GraspPoint, metadata: "dict[str, Any] | None" = None) -> GraspResult:
    return GraspResult(candidates=tuple(grasps), reasons=(), telemetry={}, depth_confidence=1.0, mask_confidence=1.0,
                       top_score=grasps[0].score, metadata=metadata or {})


class TheArmSaysOnTheClosedFormTests(unittest.TestCase):
    def test_the_natural_turn_of_2026_10_07_has_no_goal_on_the_arms_branch_and_its_twin_has(self) -> None:
        arm = _ur10(HOME_1007)
        natural = _natural_standoff(arm)
        np.testing.assert_allclose(natural.to_matrix()[:3, 0], (-0.955, -0.296, 0.005), atol=1e-3)
        self.assertAlmostEqual(86.4, math.degrees(arm._goal_joint_window()[1][5]), places=6)

        self.assertIs(False, arm.has_a_goal_on_its_branch(natural, _radians(HOME_1007)))
        self.assertIs(True, arm.has_a_goal_on_its_branch(_twin(natural), _radians(HOME_1007)))
        self.assertEqual([], arm._conn.method_calls, "the controller was asked")
        self.assertIsNone(arm._curobo_ur, "a planner was built")

    def test_the_twin_is_the_same_branch_with_wrist_3_half_a_turn_on(self) -> None:
        """``case1007.py``: the twin's goal on the branch turns wrist 3 to -90 and no joint more than 102 degrees."""
        arm = _ur10(HOME_1007)
        here = _radians(HOME_1007)
        flange = arm._pose_to_flange(_twin(_natural_standoff(arm)))
        lower, upper = arm._goal_joint_window()
        goals = nearest_goals(ur_flange_ik("ur10", np.asarray(flange.to_matrix()), q6_if_singular=here[-1]),
                              current=here, lower=lower, upper=upper, velocity=0.5)
        held = ur_branch("ur10", here)
        assert held is not None
        (on_it,) = [goal for goal in goals if held.agrees_with(ur_branch("ur10", goal.joints))]
        self.assertAlmostEqual(-90.0, math.degrees(on_it.joints[5]), delta=1.0)
        self.assertAlmostEqual(102.0, math.degrees(on_it.largest_rad), delta=1.0)

    def test_from_config_bs_home_the_natural_turn_has_one(self) -> None:
        arm = _ur10(HOME_B)
        self.assertIs(True, arm.has_a_goal_on_its_branch(_natural_standoff(arm), _radians(HOME_B)))

    def test_what_it_cannot_tell_is_none_and_a_pose_out_of_reach_has_none(self) -> None:
        arm = _ur10(HOME_1007)
        natural = _natural_standoff(arm)
        in_camera = Pose(position_mm=natural.position_mm, quaternion_xyzw=natural.quaternion_xyzw, frame=Frame.CAMERA)
        far = Pose(position_mm=np.asarray(natural.position_mm) * 5.0, quaternion_xyzw=natural.quaternion_xyzw,
                   frame=Frame.BASE)

        self.assertIsNone(arm.has_a_goal_on_its_branch(in_camera, _radians(HOME_1007)))
        self.assertIsNone(arm.has_a_goal_on_its_branch(natural, _radians(HOME_1007)[:5]))
        self.assertIsNone(arm.has_a_goal_on_its_branch(natural, [float("nan")] * 6))
        self.assertIs(False, arm.has_a_goal_on_its_branch(far, _radians(HOME_1007)))


class ThePickTakesTheTwinTests(unittest.TestCase):
    def test_the_2026_10_07_grasp_is_taken_half_a_turn_round_on_the_arms_branch(self) -> None:
        """Red before: the grasp kept its natural turn, and only another branch reached its standoff."""
        arm = _ur10(HOME_1007)
        natural = _grasp(_natural_standoff(arm))

        with self.assertLogs("src.robot.grasping.loop.pick_loop", level="INFO") as said:
            kept = _loop(arm)._closing_along(_result(natural), "wrist")

        (taken,) = kept.candidates
        np.testing.assert_allclose(taken.axis, -natural.axis, atol=1e-12)
        np.testing.assert_array_equal(taken.approach, natural.approach)
        np.testing.assert_array_equal(taken.position, natural.position)
        self.assertEqual((natural.score, natural.grip_width_mm), (taken.score, taken.grip_width_mm))
        self.assertIs(True, arm.has_a_goal_on_its_branch(_loop(arm).policy.standoff_of(taken), _radians(HOME_1007)))
        self.assertTrue(any("their twins half a turn about the approach have one" in r.getMessage() for r in said.records))

    def test_minus_y_stays_wherever_its_turn_has_a_goal_on_the_branch(self) -> None:
        """From config B's home the natural turn reaches the branch, wrist 3 166 degrees round where the twin needs 112:
        the natural turn is kept. Choosing the shorter turn (R1) is not built."""
        arm = _ur10(HOME_B)
        natural = _grasp(_natural_standoff(arm))
        result = _result(natural)

        self.assertIs(result, _loop(arm)._closing_along(result, "wrist"))

    def test_a_grasp_neither_turn_of_which_has_one_keeps_its_natural_turn(self) -> None:
        arm = _ur10(HOME_1007)
        arm.has_a_goal_on_its_branch = lambda pose, here: False  # type: ignore[method-assign]
        result = _result(_grasp(_natural_standoff(arm)))

        self.assertIs(result, _loop(arm)._closing_along(result, "wrist"))

    def test_both_faces_grips_the_grasp_whose_faces_were_judged(self) -> None:
        arm = _ur10(HOME_1007)
        result = _result(_grasp(_natural_standoff(arm)))

        self.assertIs(result, _loop(arm, both_faces=True)._closing_along(result, "wrist"))

    def test_a_closing_axis_the_program_names_keeps_its_own_way_round(self) -> None:
        arm = _ur10(HOME_1007)
        natural = _grasp(_natural_standoff(arm))
        asked: list[Any] = []
        arm.has_a_goal_on_its_branch = lambda pose, here: asked.append(pose) or False  # type: ignore[method-assign]

        kept = _loop(arm, closing_axis="-x")._closing_along(_result(natural), "wrist")

        self.assertEqual([], asked)
        self.assertLess(float(kept.candidates[0].axis[0]), 0.0, "not the way round -x names")

    def test_an_arm_that_cannot_say_leaves_the_turns_as_chosen(self) -> None:
        arm = _ur10(HOME_1007)
        loop = _loop(arm)
        loop.arm = SimpleNamespace(get_joint_positions=arm.get_joint_positions)
        result = _result(_grasp(_natural_standoff(arm)))

        self.assertIs(result, loop._closing_along(result, "wrist"))

    def test_the_wrist_cameras_distance_decides_after_it(self) -> None:
        """A wall where the twin's camera would stand: the camera check turns the grasp back the natural way round."""
        arm = _ur10(HOME_1007)
        natural = _grasp(_natural_standoff(arm))
        twin = GraspPoint(position=natural.position, approach=natural.approach, axis=-natural.axis,
                          grip_width_mm=natural.grip_width_mm, score=natural.score, frame=natural.frame,
                          label=natural.label)
        camera = _camera_at(twin)
        wall = _wall(float(camera[0]), float(camera[1]), float(camera[2]), (30.0, 30.0, 30.0))
        loop = _loop(arm)
        loop._wrist_housing_cached = _HOUSING

        with self.assertLogs("src.robot.grasping.loop.pick_loop", level="INFO") as said:
            kept = loop._closing_along(_result(natural, metadata={"scene_seen_boxes": (wall,),
                                                                   "scene_seen_distance_mm": 3.0}), "wrist")

        np.testing.assert_allclose(kept.candidates[0].axis, natural.axis, atol=1e-12)
        lines = [record.getMessage() for record in said.records]
        self.assertTrue(any("the twins are taken" in line for line in lines), lines)
        self.assertTrue(any("at 1 only turned half a turn" in line for line in lines), lines)


if __name__ == "__main__":
    unittest.main()
