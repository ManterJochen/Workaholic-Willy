"""A grasp judged ahead is judged once, and the moves it judged run as they were judged (the judge chain, 2026-10-08).

The owner presented the cell all day on 2026-10-08, and a pick took about a minute. Between "try 2" and the arm moving,
11.4 s went to judging: the controller's inverse kinematics 92 times, five world refreshes, the exact guard on 2177
samples. Three of those judgements were repeats. The line down to the grasp was judged twice, the lift's world was rebuilt
for the question the line's world had just answered, and the route to the standoff judged ahead was judged again by the
move that ran it. After the close, the line up was judged again as the lift judged at the part had judged it. What this
file pins, on the real UR driver, its real guard and planner glue, over the test bench's camera and sidecar:

* ``grasp_refusal_ahead`` judges the line down, the lift and the route once each, with two refreshes, where the route
  ends where the line was judged from, and keeps that route for the move to the standoff;
* that move runs the route as it was judged, with no refresh and no second judgement, and is stamped PLANNED on the
  refresh it was judged in. A moved arm, a refresh, more than half a second, another pose, another hand, a held world, a
  halt and a second move each have it judged again;
* the line up from the part runs on the lift judged there as if the jaws held the part, inside the held world it was
  judged in, on a cell that models no carried part. A moved arm, a cell that models the part, the world no longer held
  and another pose each have it judged again.

Every motion still goes out through the same sending code: the halt latch read before every waypoint, the controller's
own checks before a moveL. Honesty bucket (2): the arm's real judgement on a stand-in controller whose inverse kinematics
is the DH chain's closed form, the solution nearest its seed.
"""

from __future__ import annotations

import dataclasses
import time
import unittest
from typing import Any
from unittest.mock import patch

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import JointPositions, MotionStatus, RobotKinematicsError
from src.robot.core.arm_capabilities import HaltState
from src.robot.core.camera_world import CameraWorldUse
from src.robot.drivers.ur import curobo_motion
from src.robot.drivers.ur.curobo_motion import UR_ARM_JOINT_NAMES
from src.robot.drivers.ur.pose_adapter import pose_to_urpose
from src.robot.safety._ur_ik import ur_flange_ik
from tests.test_a_bin_the_camera_saw_beside_the_base_is_the_exact_guards import HandKnownOpen
from tests.test_every_verb_meets_the_camera_world import _Camera, _Client, _mesh_backend_available, _ur, _world

#: A part on the bench in front of the arm: the grasp, its standoff 80 mm over it, the lift's top 100 mm over it, all tool
#: down, and where the arm stands before it leaves for it, 60 mm over the standoff.
_GRASP = Pose.tool_down(350.0, -150.0, 120.0, label="grasp")
_STANDOFF = Pose.tool_down(350.0, -150.0, 200.0, label="standoff")
_LIFT = Pose.tool_down(350.0, -150.0, 220.0, label="lift")
_ABOVE = Pose.tool_down(350.0, -150.0, 260.0, label="above")
#: The configuration of ``_ABOVE`` the arm stands in, by the solution nearest these joints (degrees).
_ABOVE_NEAR_DEG = (126.4, -86.2, 100.2, -14.0, -53.6, 180.0)
_WIDTH_MM = 40.0


def _shifted(pose: Pose, dz_mm: float) -> Pose:
    position = np.asarray(pose.position_mm, dtype=np.float64).copy()
    position[2] += float(dz_mm)
    return Pose(position_mm=position, quaternion_xyzw=np.asarray(pose.quaternion_xyzw, dtype=np.float64),
                frame=Frame.BASE, label=f"{pose.label} {dz_mm:+.0f} mm")


def _nearest(solutions: Any, start: "np.ndarray") -> "np.ndarray":
    """The solution nearest ``start``, each joint on its turn nearest it."""
    turned = [start + (np.asarray(q, dtype=np.float64) - start + np.pi) % (2.0 * np.pi) - np.pi for q in solutions]
    return min(turned, key=lambda q: float(np.abs(q - start).max()))


def _controller_ik(arm: Any, solved: "list[Pose]") -> Any:
    """The controller's inverse kinematics as the stand-in answers it: the closed form of the flange the pose puts there,
    the solution nearest the seed. Every pose asked is recorded in ``solved``."""
    model = str(arm.config.ur.model)

    def ik(pose: Pose, *, seed: "JointPositions | None" = None) -> JointPositions:
        solved.append(pose)
        start = np.asarray(seed.tolist() if seed is not None else arm._conn.get_joint_positions(), dtype=np.float64)
        flange = np.asarray(arm._pose_to_flange(pose).to_matrix(), dtype=np.float64)
        solutions = ur_flange_ik(model, flange, q6_if_singular=float(start[-1]))
        if not solutions:
            raise RobotKinematicsError(f"no solution for {pose.label or 'the pose'}")
        return JointPositions(tuple(float(v) for v in _nearest(solutions, start)))

    return ik


def _configuration(arm: Any, pose: Pose, near: "list[float]") -> "list[float]":
    model = str(arm.config.ur.model)
    flange = np.asarray(arm._pose_to_flange(pose).to_matrix(), dtype=np.float64)
    solutions = ur_flange_ik(model, flange, q6_if_singular=float(near[-1]))
    assert solutions, f"{pose.label} is out of reach"
    return [float(v) for v in _nearest(solutions, np.asarray(near, dtype=np.float64))]


def _stand(arm: Any, joints: "list[float]") -> None:
    """The controller reports the arm at ``joints``: its joints, and its TCP (the controller holds the tool here)."""
    arm._conn.get_joint_positions.return_value = [float(v) for v in joints]
    tcp = arm._tcp_at_joints_mm(JointPositions(tuple(joints)))
    arm._motion.get_current_pose.return_value = pose_to_urpose(Pose.from_matrix(tcp, frame=Frame.BASE))


class _CarryingClient(_Client):
    """The test bench's sidecar, taking a carried part as the glue hands it over and forgetting it on a detach."""

    def attach_payload(self, joints: Any, dims_m: Any, pose: Any) -> bool:
        return True

    def detach_payload(self) -> bool:
        return True


def _cell(client: "_Client | None" = None) -> "tuple[Any, list[str], list[Pose]]":
    """The cuRobo UR of ``test_every_verb_meets_the_camera_world`` (its real guard recording into ``events``, its live
    world on the bench camera, its sidecar double), its controller solving every pose on the closed form into
    ``solved``, standing over the part. No speed is asked of a move, so the configured maximum ranks its goals, as a
    grasp judged ahead ranks them."""
    events: list[str] = []
    arm = _ur(_world(_Camera(silent=False)), client or _Client(joint_names=UR_ARM_JOINT_NAMES), events)
    solved: list[Pose] = []
    arm.ik = _controller_ik(arm, solved)
    arm._motion._clamp.side_effect = lambda vel, acc: (vel, acc)
    _stand(arm, _configuration(arm, _ABOVE, [float(np.radians(v)) for v in _ABOVE_NEAR_DEG]))
    return arm, events, solved


def _judged_ahead(arm: Any) -> str:
    return arm.grasp_refusal_ahead(standoff=_STANDOFF, grasp=_GRASP, lift=_LIFT, grip_width_mm=_WIDTH_MM)


def _refreshes() -> Any:
    return patch.object(curobo_motion, "refresh_planner_world", wraps=curobo_motion.refresh_planner_world)


def _moved_1_mm(arm: Any) -> None:
    """The arm a millimetre from where it stood, in the path gate's own metric: wrist 3 turned that far."""
    here = [float(v) for v in arm._conn.get_joint_positions()]
    radii = arm._preflight.joint_radii_mm(arm)
    here[5] += 1.0 / float(radii[5])
    arm._conn.get_joint_positions.return_value = here


class AGraspJudgedAheadIsJudgedOnceTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_the_line_down_the_lift_and_the_route_are_judged_once_each_with_two_refreshes(self) -> None:
        """⭐ Red before: four refreshes and the line down judged twice, from the closed form's configuration and again
        from the route's end, which is the same configuration."""
        arm, events, solved = _cell()
        lines: list[Any] = []
        judge = arm._judge_linear_move

        def judged_line(pose: Pose, **asked: Any) -> Any:
            lines.append((pose.label, asked.get("start")))
            return judge(pose, **asked)

        arm._judge_linear_move = judged_line
        with _refreshes() as refresh:
            self.assertEqual("", _judged_ahead(arm))
        self.assertEqual(2, refresh.call_count, "the line down's world, the lift judged in it, the route's own")
        self.assertEqual(["grasp", "lift"], [label for label, _ in lines], "the line down once, the lift once")
        self.assertEqual(1, events.count("gate_planned_path"), "the route once")
        ahead = arm._judged_ahead
        assert ahead is not None, "the route was not kept for the move to the standoff"
        self.assertEqual([float(v) for v in ahead.route.waypoints[-1]], [float(v) for v in lines[0][1][1]],
                         "the route ends where the line down was judged from")

    def test_the_lift_is_judged_in_the_world_the_line_down_was_judged_in(self) -> None:
        arm, _events, _solved = _cell()
        held: list[str] = []
        carried = arm.carried_line_refusal

        def lift(pose: Pose, **asked: Any) -> Any:
            held.append(str(arm._held_world_reason))
            return carried(pose, **asked)

        arm.carried_line_refusal = lift
        self.assertEqual("", _judged_ahead(arm))
        self.assertEqual(1, len(held))
        self.assertIn("the world the line down to its grasp was judged in", held[0])
        self.assertIsNone(arm._held_world_reason, "the held world ends with the lift")


class TheMoveToTheStandoffRunsTheRouteJudgedAheadTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def _ready(self, client: "_Client | None" = None) -> "tuple[Any, list[str], Any]":
        arm, events, _solved = _cell(client)
        self.assertEqual("", _judged_ahead(arm))
        ahead = arm._judged_ahead
        self.assertIsNotNone(ahead)
        events.clear()
        return arm, events, ahead

    def _judged_again(self, arm: Any, events: "list[str]", pose: Pose = _STANDOFF, *, refreshes: int = 1) -> Any:
        """The move to ``pose`` judges its route as it runs: its own refresh (none in a held world) and its path gate."""
        with _refreshes() as refresh:
            result = arm.move(pose)
        self.assertEqual(refreshes, refresh.call_count, "the move did not refresh its world as it judged")
        self.assertIn("gate_planned_path", events, "the move ran a route nobody judged again")
        return result

    def test_the_move_runs_the_route_as_it_was_judged(self) -> None:
        """⭐ Red before: a third refresh and the exact guard on every sample of the same route a second time, 2.9 to
        3.3 s of a try on the cell."""
        arm, events, ahead = self._ready()
        with _refreshes() as refresh:
            result = arm.move(_STANDOFF)
        self.assertIs(MotionStatus.EXECUTED, result.status, result.message)
        self.assertEqual(0, refresh.call_count, "the move refreshed its world")
        self.assertNotIn("gate_planned_path", events, "the move judged its route a second time")
        self.assertNotIn("gate_joint_path", events)
        sent = [list(call.args[0]) for call in arm._conn.moveJ.call_args_list]
        self.assertEqual([[float(v) for v in w] for w in ahead.route.waypoints], sent,
                         "moveJ was not handed exactly the waypoints judged")
        stamp = result.camera_world
        self.assertIs(CameraWorldUse.PLANNED, stamp.use, stamp.render())
        self.assertEqual(ahead.refresh.captured_at_s, stamp.captured_at_s)
        self.assertIsNone(arm._judged_ahead, "the route judged ahead serves one move")

    def test_the_picks_own_sequence_runs_the_route_it_judged_ahead(self) -> None:
        """The pick loop asks the policy ``refusal_ahead``, then ``execute``s the grasp: the move to the standoff runs the
        route judged ahead, the line down refreshes its own world as ever, and the line up is judged in it."""
        from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy, PolicyOutcome
        from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint

        arm, events, _solved = _cell()
        policy = GraspExecutionPolicy(arm=arm, gripper=None, standoff_mm=80.0, retreat_mm=100.0)
        grasp = GraspPoint(position=np.array([350.0, -150.0, 120.0]), approach=np.array([0.0, 0.0, -1.0]),
                           axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=_WIDTH_MM, score=0.9, frame=GraspFrame.BASE,
                           label="part")
        self.assertEqual("", policy.refusal_ahead(grasp))
        self.assertIsNotNone(arm._judged_ahead)
        events.clear()
        with _refreshes() as refresh:
            report = policy.execute(grasp)
        self.assertIs(PolicyOutcome.EXECUTED, report.outcome, report.motion_message)
        self.assertEqual(1, refresh.call_count, "the line down's world alone")
        self.assertEqual(0, events.count("gate_planned_path"), "the route to the standoff was judged a second time")
        self.assertEqual(2, arm._conn.moveJ.call_count, "the route judged ahead, one moveJ a waypoint")
        self.assertEqual(2, arm._motion.move_to.call_count, "the moveL down and the moveL up")

    def test_a_second_move_judges_its_route(self) -> None:
        arm, events, _ahead = self._ready()
        self.assertTrue(arm.move(_STANDOFF).ok)
        events.clear()
        self.assertTrue(self._judged_again(arm, events).ok)

    def test_an_arm_moved_a_millimetre_has_its_route_judged_again(self) -> None:
        arm, events, _ahead = self._ready()
        _moved_1_mm(arm)
        result = self._judged_again(arm, events)
        self.assertTrue(result.ok, result.message)
        self.assertEqual([float(v) for v in arm._conn.get_joint_positions()],
                         list(arm._conn.moveJ.call_args_list[0].args[0]), "the route judged again starts where it stands")

    def test_a_refresh_in_between_has_the_route_judged_again(self) -> None:
        arm, events, _ahead = self._ready()
        arm._curobo_ur_planner().refresh_world(near_point_mm=[350.0, -150.0, 200.0])
        self.assertTrue(self._judged_again(arm, events).ok)

    def test_more_than_half_a_second_has_the_route_judged_again(self) -> None:
        arm, events, ahead = self._ready()
        arm._judged_ahead = dataclasses.replace(ahead, at=time.monotonic() - 0.6)
        self.assertTrue(self._judged_again(arm, events).ok)

    def test_another_pose_is_judged_and_the_route_judged_ahead_is_gone(self) -> None:
        arm, events, _ahead = self._ready()
        self.assertTrue(self._judged_again(arm, events, _shifted(_STANDOFF, 5.0)).ok)
        self.assertIsNone(arm._judged_ahead)
        events.clear()
        self.assertTrue(self._judged_again(arm, events).ok)

    def test_a_hand_that_changed_has_the_route_judged_again(self) -> None:
        arm, events, _solved = _cell()
        hand = HandKnownOpen()
        arm.set_hand(hand)
        self.assertEqual("", _judged_ahead(arm))
        events.clear()
        hand.jaws_closed = True
        self.assertTrue(self._judged_again(arm, events).ok)

    def test_a_held_world_has_the_route_judged_again_in_it(self) -> None:
        arm, events, _ahead = self._ready()
        with arm.held_world("a block that holds the world"):
            result = self._judged_again(arm, events, refreshes=0)
        self.assertTrue(result.ok, result.message)

    def test_a_halt_sends_nothing(self) -> None:
        arm, events, _ahead = self._ready()
        arm._conn.halt_state.return_value = HaltState(reason="the operator pressed halt now", requested_at=time.time())
        arm._conn.halt_requests = 1
        result = arm.move(_STANDOFF)
        self.assertIs(MotionStatus.CANCELLED, result.status, result.message)
        self.assertIn("the operator pressed halt now", result.message)
        arm._conn.moveJ.assert_not_called()

    def test_any_other_motion_verb_forgets_the_route(self) -> None:
        verbs = {
            "a line": lambda arm: arm.move(_STANDOFF, linear=True),
            "a joint move": lambda arm: arm.move_to_joints(JointPositions(tuple(arm._conn.get_joint_positions()))),
            "a refused move": lambda arm: arm.move(Pose(position_mm=_STANDOFF.position_mm,
                                                        quaternion_xyzw=_STANDOFF.quaternion_xyzw, frame=Frame.TOOL)),
        }
        for label, verb in verbs.items():
            with self.subTest(label):
                arm, _events, _ahead = self._ready()
                verb(arm)
                self.assertIsNone(arm._judged_ahead)


def _at_the_part(client: "_Client | None" = None) -> "tuple[Any, list[str], list[Pose]]":
    """The arm down its judged line at the part, the world that line was judged in the planner's last: standing at the
    standoff, the line down judged and sent, the controller then reporting the arm at the grasp."""
    arm, events, solved = _cell(client)
    at_standoff = _configuration(arm, _STANDOFF, [float(v) for v in arm._conn.get_joint_positions()])
    _stand(arm, at_standoff)
    down = arm.move(_GRASP, linear=True)
    assert down.ok, down.message
    _stand(arm, _configuration(arm, _GRASP, at_standoff))
    events.clear()
    solved.clear()
    return arm, events, solved


#: The held world a grasp opens at the part (``execution_policy._the_world_held_at_the_part``).
_HELD = "the way back up from the part, judged against the world the line down was judged in"


class TheLineUpRunsOnTheLiftJudgedAtThePartTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_the_line_up_runs_with_no_second_judgement(self) -> None:
        """⭐ Red before: the line up solved again at every sample and the exact guard on every one, about 1.5 s of a
        try on the cell, for the same line, in the same world, judged as the lift was."""
        arm, events, solved = _at_the_part()
        with arm.held_world(_HELD):
            self.assertIsNone(arm.carried_line_refusal(_LIFT, grip_width_mm=_WIDTH_MM))
            asked, gated = len(solved), events.count("gate_joint_path")
            self.assertGreater(asked, 0)
            self.assertEqual(1, gated)
            self.assertFalse(arm.attach_payload(_WIDTH_MM), "this cell models no carried part")
            with _refreshes() as refresh:
                result = arm.move(_LIFT, linear=True)
        self.assertIs(MotionStatus.EXECUTED, result.status, result.message)
        self.assertEqual(asked, len(solved), "the line up was solved a second time")
        self.assertEqual(gated, events.count("gate_joint_path"), "the exact guard ran a second time")
        self.assertEqual(0, refresh.call_count)
        self.assertEqual(2, arm._motion.move_to.call_count, "the moveL down and the moveL up")
        self.assertIs(CameraWorldUse.PLANNED, result.camera_world.use)

    def _judged_again(self, arm: Any, events: "list[str]", solved: "list[Pose]", pose: Pose = _LIFT) -> None:
        asked, gated = len(solved), events.count("gate_joint_path")
        result = arm.move(pose, linear=True)
        self.assertTrue(result.ok, result.message)
        self.assertGreater(len(solved), asked, "the line up was not solved again")
        self.assertEqual(gated + 1, events.count("gate_joint_path"), "the exact guard did not judge the line up")

    def test_an_arm_moved_a_millimetre_has_the_line_up_judged_again(self) -> None:
        arm, events, solved = _at_the_part()
        with arm.held_world(_HELD):
            self.assertIsNone(arm.carried_line_refusal(_LIFT, grip_width_mm=_WIDTH_MM))
            arm.attach_payload(_WIDTH_MM)
            _moved_1_mm(arm)
            self._judged_again(arm, events, solved)

    def test_another_pose_has_the_line_up_judged(self) -> None:
        arm, events, solved = _at_the_part()
        with arm.held_world(_HELD):
            self.assertIsNone(arm.carried_line_refusal(_LIFT, grip_width_mm=_WIDTH_MM))
            arm.attach_payload(_WIDTH_MM)
            self._judged_again(arm, events, solved, _shifted(_LIFT, 10.0))

    def test_outside_the_held_world_the_line_up_is_judged(self) -> None:
        arm, events, solved = _at_the_part()
        with arm.held_world(_HELD):
            self.assertIsNone(arm.carried_line_refusal(_LIFT, grip_width_mm=_WIDTH_MM))
            arm.attach_payload(_WIDTH_MM)
        with _refreshes() as refresh:
            self._judged_again(arm, events, solved)
        self.assertEqual(1, refresh.call_count)

    def test_a_cell_that_models_the_part_judges_the_line_up_carrying_it(self) -> None:
        """The lift judged at the part only stood in for the part the planner then carries: its line up is judged with
        it (``safety.planning_world.payload.length_mm``)."""
        arm, events, solved = _at_the_part(_CarryingClient(joint_names=UR_ARM_JOINT_NAMES))
        world = arm.config.safety.planning_world
        modelled = world.model_copy(update={"payload": world.payload.model_copy(update={"length_mm": 120.0})})
        arm.config = arm.config.model_copy(update={"safety": arm.config.safety.model_copy(
            update={"planning_world": modelled})})
        self.assertIsNone(arm.payload_declined_reason(), "this cell models the carried part")
        with arm.held_world(_HELD):
            self.assertIsNone(arm.carried_line_refusal(_LIFT, grip_width_mm=_WIDTH_MM))
            self.assertIsNone(arm._lift_judged)
            self.assertTrue(arm.attach_payload(_WIDTH_MM))
            self._judged_again(arm, events, solved)


if __name__ == "__main__":
    unittest.main()
