"""A leg judged ahead while the arm waited runs as it was judged only where nothing it was judged on changed (map4 C7).

The owner, 2026-10-09: judge the next leg while the jaws travel (``robot.motion.judge_next_leg``), in the world held
across the jaws' stroke (``safety.planning_world.hold.carry`` and ``.drop``: "altes Bild, weil sich in der Kiste so oder so
nichts verändert"), and during the line before it on a send a halt brakes (``in_settles_and_motion``). What the UR
driver keeps of such a judgement follows wave 1's rules for a route judged ahead: it serves one move, it runs only where
the arm stands within 0.5 mm of where it was judged from, in the world of the very refresh it was judged in, with the
carried part, the hand, the camera's boxes and the halts as they were; otherwise it is judged again as it runs. Pinned
here, on the real UR driver, its real guard and planner glue (the fixture of
``test_a_grasp_judged_ahead_runs_as_it_was_judged``):

* the line out of a place judged where the arm stands, in a held world (``judge_line_ahead``), runs with no second
  judgement; an arm moved a millimetre, another pose, or no held world has it judged again;
* the joint move a task declared next (``expecting_next``), judged from where the line before it ends
  (``judge_the_next_leg``), runs as judged on ``move_to_joints`` and ``move_to_home``, stamped with its refresh; a moved
  arm, a refresh, a halt, more than ten seconds, another target, and every switch off have it judged as it runs;
* judged during the line on a second thread, it asks the controller nothing from that thread, and runs as judged; an
  arm whose halt does not brake the line judges nothing during it; a judgement still running when its line ended long
  ago keeps nothing.

Honesty bucket (2): the arm's real judgement on a stand-in controller.
"""

from __future__ import annotations

import dataclasses
import threading
import time
import unittest
from typing import Any
from unittest.mock import patch

from src.config.schema.robot.safety_schema import HeldWorldConfig
from src.robot.core import JointPositions, MotionStatus
from src.robot.core.arm_capabilities import HaltState
from src.robot.core.camera_world import CameraWorldUse
from tests.test_a_grasp_judged_ahead_runs_as_it_was_judged import (
    _ABOVE,
    _HELD,
    _LIFT,
    _STANDOFF,
    _at_the_part,
    _configuration,
    _moved_1_mm,
    _refreshes,
    _shifted,
    _stand,
)
from tests.test_every_verb_meets_the_camera_world import _mesh_backend_available


def _judging(arm: Any, *, mode: str = "in_settles", carry: bool = True, drop: bool = True) -> None:
    """The arm judges next legs (``robot.motion.judge_next_leg``) and holds its world across the jaws' strokes."""
    arm._next_leg_mode = mode
    arm._holds = HeldWorldConfig(carry=carry, drop=drop)


def _look(arm: Any) -> JointPositions:
    """The look the pick left from, the joint move a task declares after the pick: the configuration over the part."""
    return JointPositions(tuple(_configuration(arm, _ABOVE, [float(v) for v in arm._conn.get_joint_positions()])))


class TheLineOutJudgedWhileTheJawsOpenTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_the_line_out_runs_with_no_second_judgement(self) -> None:
        """⭐ Red before: the line out was solved and judged again after the stroke, in a frame taken then."""
        arm, events, solved = _at_the_part()
        with arm.held_world("the drop"):
            self.assertIsNone(arm.judge_line_ahead(_STANDOFF))
            asked, gated = len(solved), events.count("gate_joint_path")
            self.assertEqual(1, gated)
            with _refreshes() as refresh:
                result = arm.move(_STANDOFF, linear=True)
        self.assertIs(MotionStatus.EXECUTED, result.status, result.message)
        self.assertEqual(asked, len(solved), "the line out was solved a second time")
        self.assertEqual(gated, events.count("gate_joint_path"), "the exact guard ran a second time")
        self.assertEqual(0, refresh.call_count)
        self.assertIs(CameraWorldUse.PLANNED, result.camera_world.use)

    def test_an_arm_moved_a_millimetre_has_the_line_out_judged_again(self) -> None:
        arm, events, solved = _at_the_part()
        with arm.held_world("the drop"):
            self.assertIsNone(arm.judge_line_ahead(_STANDOFF))
            _moved_1_mm(arm)
            gated = events.count("gate_joint_path")
            self.assertTrue(arm.move(_STANDOFF, linear=True).ok)
        self.assertEqual(gated + 1, events.count("gate_joint_path"))

    def test_another_pose_has_the_line_judged(self) -> None:
        arm, events, _solved = _at_the_part()
        with arm.held_world("the drop"):
            self.assertIsNone(arm.judge_line_ahead(_STANDOFF))
            gated = events.count("gate_joint_path")
            self.assertTrue(arm.move(_shifted(_STANDOFF, 10.0), linear=True).ok)
        self.assertEqual(gated + 1, events.count("gate_joint_path"))

    def test_outside_a_held_world_nothing_is_kept(self) -> None:
        arm, _events, _solved = _at_the_part()
        self.assertIsNone(arm.judge_line_ahead(_STANDOFF))
        self.assertIsNone(arm._lift_judged)

    def test_on_a_cell_that_models_the_part_the_line_out_judged_as_it_runs_runs_too(self) -> None:
        """Unlike the lift judged before the close, whose part only stood in for the one carried, the line out is judged
        empty-handed, as it runs."""
        arm, events, solved = _at_the_part()
        world = arm.config.safety.planning_world
        modelled = world.model_copy(update={"payload": world.payload.model_copy(update={"length_mm": 120.0})})
        arm.config = arm.config.model_copy(update={"safety": arm.config.safety.model_copy(
            update={"planning_world": modelled})})
        with arm.held_world("the drop"):
            self.assertIsNone(arm.judge_line_ahead(_STANDOFF))
            self.assertIsNotNone(arm._lift_judged)
            asked = len(solved)
            self.assertTrue(arm.move(_STANDOFF, linear=True).ok)
        self.assertEqual(asked, len(solved))


class TheJointMoveDeclaredNextTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def _ready(self, **switches: Any) -> "tuple[Any, list[str], JointPositions, Any]":
        """At the part, the carry to the look judged while the jaws close, and the arm then at the top of the lift."""
        arm, events, _solved = _at_the_part()
        _judging(arm, **switches)
        look = _look(arm)
        with arm.expecting_next(look), arm.held_world(_HELD):
            said = arm.judge_the_next_leg(_LIFT, junction="carry")
        judged = arm._next_leg_ahead
        if judged is not None:
            _stand(arm, list(judged.start))
        events.clear()
        return arm, events, look, said

    def test_the_carry_runs_as_it_was_judged_at_the_part(self) -> None:
        """⭐ Red before: a frame at the top of the lift, a world built for it, and the exact guard on every sample of
        the carry, 4.0 s on the cell after waves 1 and 2."""
        arm, events, look, said = self._ready()
        self.assertEqual("", said)
        judged = arm._next_leg_ahead
        assert judged is not None
        with _refreshes() as refresh:
            result = arm.move_to_joints(look)
        self.assertIs(MotionStatus.EXECUTED, result.status, result.message)
        self.assertEqual(0, refresh.call_count, "the carry refreshed its world")
        self.assertNotIn("gate_planned_path", events, "the carry was judged a second time")
        self.assertEqual([[float(v) for v in judged.joints.tolist()]],
                         [list(call.args[0]) for call in arm._conn.moveJ.call_args_list])
        self.assertIs(CameraWorldUse.PLANNED, result.camera_world.use)
        self.assertEqual(judged.refresh.captured_at_s, result.camera_world.captured_at_s)
        self.assertIsNone(arm._next_leg_ahead, "a move judged ahead serves one move")

    def _judged_again(self, arm: Any, events: "list[str]", look: JointPositions) -> None:
        with _refreshes() as refresh:
            result = arm.move_to_joints(look)
        self.assertTrue(result.ok, result.message)
        self.assertEqual(1, refresh.call_count, "the move did not refresh its world as it judged")
        self.assertIn("gate_planned_path", events)

    def test_an_arm_moved_a_millimetre_has_the_carry_judged_again(self) -> None:
        arm, events, look, _said = self._ready()
        _moved_1_mm(arm)
        self._judged_again(arm, events, look)

    def test_a_refresh_in_between_has_the_carry_judged_again(self) -> None:
        arm, events, look, _said = self._ready()
        arm._curobo_ur_planner().refresh_world(near_point_mm=[350.0, -150.0, 220.0])
        self._judged_again(arm, events, look)

    def test_a_halt_has_the_carry_judged_again(self) -> None:
        arm, events, look, _said = self._ready()
        arm._conn.halt_state.return_value = HaltState(reason="the operator pressed halt now", requested_at=time.time())
        self.assertIn("halted", arm._why_the_next_leg_is_judged_again(arm._next_leg_ahead))

    def test_more_than_ten_seconds_have_the_carry_judged_again(self) -> None:
        arm, events, look, _said = self._ready()
        arm._next_leg_ahead = dataclasses.replace(arm._next_leg_ahead, at=time.monotonic() - 10.5)
        self._judged_again(arm, events, look)

    def test_another_target_is_judged_and_the_carry_judged_ahead_is_gone(self) -> None:
        arm, events, look, _said = self._ready()
        other = JointPositions(tuple(v + (0.05 if i == 5 else 0.0) for i, v in enumerate(look.tolist())))
        self._judged_again(arm, events, other)
        self.assertIsNone(arm._next_leg_ahead)

    def test_a_move_to_a_pose_forgets_it(self) -> None:
        arm, _events, _look, _said = self._ready()
        self.assertTrue(arm.move(_shifted(_LIFT, 20.0)).ok)
        self.assertIsNone(arm._next_leg_ahead)

    def test_the_way_home_runs_as_it_was_judged(self) -> None:
        arm, events, look, _said = self._ready()
        arm._home_joints = list(look.tolist())
        with _refreshes() as refresh:
            result = arm.move_to_home()
        self.assertIs(MotionStatus.EXECUTED, result.status, result.message)
        self.assertEqual(0, refresh.call_count)
        self.assertNotIn("gate_planned_path", events)

    def test_nothing_is_judged_ahead_with_any_switch_off(self) -> None:
        cases = {
            "judge_next_leg off": {"mode": "off"},
            "hold.carry off": {"carry": False},
        }
        for label, switches in cases.items():
            with self.subTest(label):
                arm, _events, _look, said = self._ready(**switches)
                self.assertTrue(said)
                self.assertIsNone(arm._next_leg_ahead)

    def test_nothing_is_judged_ahead_where_nothing_was_declared_or_no_world_is_held(self) -> None:
        arm, _events, _solved = _at_the_part()
        _judging(arm)
        with arm.held_world(_HELD):
            self.assertIn("declared", arm.judge_the_next_leg(_LIFT, junction="carry"))
        with arm.expecting_next(_look(arm)):
            self.assertIn("no world is held", arm.judge_the_next_leg(_LIFT, junction="carry"))
        self.assertIsNone(arm._next_leg_ahead)


class TheJointMoveJudgedDuringTheLineBeforeItTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def _lift_judging_the_carry(self, *, brakes: bool = True) -> "tuple[Any, list[str], JointPositions, dict[str, Any]]":
        arm, events, _solved = _at_the_part()
        _judging(arm, mode="in_settles_and_motion")
        arm._conn.brake_on_halt = brakes
        look = _look(arm)
        seen: dict[str, Any] = {"main": threading.get_ident(), "judged on": None, "controller from": set()}
        judge = arm._judge_the_next_leg_from

        def judged_on(*asked: Any, **said: Any) -> str:
            seen["judged on"] = threading.get_ident()
            time.sleep(0.05)  # the line is still running
            return judge(*asked, **said)

        arm._judge_the_next_leg_from = judged_on
        for name in ("get_joint_positions", "wait_until_steady", "moveJ", "moveL", "getInverseKinematics"):
            method = getattr(arm._conn, name)
            wrapped = method.side_effect

            def asked(*args: Any, _wrapped: Any = wrapped, _method: Any = method, **kwargs: Any) -> Any:
                seen["controller from"].add(threading.get_ident())
                return _wrapped(*args, **kwargs) if _wrapped is not None else _method.return_value

            method.side_effect = asked
        ik = arm.ik

        def solved(*args: Any, **kwargs: Any) -> Any:
            seen["controller from"].add(threading.get_ident())
            return ik(*args, **kwargs)

        arm.ik = solved
        with arm.expecting_next(look), arm.held_world(_HELD):
            self.assertIsNone(arm.carried_line_refusal(_LIFT, grip_width_mm=40.0))
            arm.attach_payload(40.0)
            arm.judge_the_next_leg(_LIFT, junction="carry", now=False)
            result = arm.move(_LIFT, linear=True)
        self.assertTrue(result.ok, result.message)
        return arm, events, look, seen

    def test_the_carry_is_judged_on_a_second_thread_that_asks_the_controller_nothing(self) -> None:
        """⭐ The thread judges with what the sending thread read where the arm stood still; the controller is asked only
        by the thread that sends and watches the line."""
        arm, _events, look, seen = self._lift_judging_the_carry()
        self.assertIsNotNone(seen["judged on"])
        self.assertNotEqual(seen["main"], seen["judged on"])
        self.assertEqual({seen["main"]}, seen["controller from"], "the second thread asked the controller")
        judged = arm._next_leg_ahead
        assert judged is not None
        _stand(arm, list(judged.start))
        with _refreshes() as refresh:
            result = arm.move_to_joints(look)
        self.assertIs(MotionStatus.EXECUTED, result.status, result.message)
        self.assertEqual(0, refresh.call_count)

    def test_the_thread_that_watches_the_line_is_not_kept_waiting_for_the_gil(self) -> None:
        """At Python's 5 ms the judging thread kept the thread that watches the line, which reads the halt every 8 ms,
        waiting up to 451 ms on Windows: while it judges the switch interval is 0.5 ms, and it is put back after."""
        import sys

        from src.robot.drivers.ur import arm as arm_module

        before = sys.getswitchinterval()
        arm, _events, _solved = _at_the_part()
        _judging(arm, mode="in_settles_and_motion")
        arm._conn.brake_on_halt = True
        judge = arm._judge_the_next_leg_from
        seen: list[float] = []

        def judged(*asked: Any, **said: Any) -> str:
            seen.append(sys.getswitchinterval())
            return judge(*asked, **said)

        arm._judge_the_next_leg_from = judged
        with arm.expecting_next(_look(arm)), arm.held_world(_HELD):
            arm.judge_the_next_leg(_LIFT, junction="carry", now=False)
            self.assertTrue(arm.move(_LIFT, linear=True).ok)
        self.assertEqual([arm_module._WATCHED_SWITCH_S], seen)
        self.assertEqual(before, sys.getswitchinterval(), "the switch interval was not put back")

    def test_a_cell_that_models_the_part_with_none_carried_judges_nothing_during_the_line(self) -> None:
        """There the judgement reads the hand off the controller (a toggle's output), which the second thread never
        asks: the move is judged as it runs."""
        arm, _events, _solved = _at_the_part()
        world = arm.config.safety.planning_world
        modelled = world.model_copy(update={"payload": world.payload.model_copy(update={"length_mm": 120.0})})
        arm.config = arm.config.model_copy(update={"safety": arm.config.safety.model_copy(
            update={"planning_world": modelled})})
        _judging(arm, mode="in_settles_and_motion")
        arm._conn.brake_on_halt = True
        with arm.expecting_next(_look(arm)), arm.held_world(_HELD):
            self.assertEqual("", arm.judge_the_next_leg(_LIFT, junction="drop", now=False))
            self.assertIsNone(arm._judging_during(_LIFT, arm._next_leg_during))
        self.assertIsNone(arm._next_leg_ahead)

    def test_a_line_whose_halt_does_not_brake_it_judges_nothing_during_it(self) -> None:
        arm, _events, _look, seen = self._lift_judging_the_carry(brakes=False)
        self.assertIsNone(seen["judged on"])
        self.assertIsNone(arm._next_leg_ahead)

    def test_a_judgement_still_running_long_after_its_line_ended_keeps_nothing(self) -> None:
        arm, _events, _solved = _at_the_part()
        _judging(arm, mode="in_settles_and_motion")
        arm._conn.brake_on_halt = True
        judge = arm._judge_the_next_leg_from
        release = threading.Event()

        def slow(*asked: Any, **said: Any) -> str:
            release.wait(5.0)
            return judge(*asked, **said)

        arm._judge_the_next_leg_from = slow
        with patch.object(type(arm), "_NEXT_LEG_JOIN_S", 0.05), arm.expecting_next(_look(arm)), \
                arm.held_world(_HELD):
            arm.judge_the_next_leg(_LIFT, junction="carry", now=False)
            self.assertTrue(arm.move(_LIFT, linear=True).ok)
        release.set()
        time.sleep(0.5)
        self.assertIsNone(arm._next_leg_ahead, "a judgement its line no longer waited for was kept")


class ThePicksOwnFlowTests(unittest.TestCase):
    """The owner's cell end to end, from the grasp judged at the look to the carry: one frame for all of it."""

    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_one_refresh_from_the_judgement_ahead_to_the_carry_and_one_change_of_do0(self) -> None:
        """⭐ With every switch the owner set (2026-10-09): the grasp judged ahead in one world, the approach and the
        line down held in it, the lift run on its judgement at the part, the carry judged while the jaws close and
        run as judged at the top of the lift; the toggle changes its output once."""
        import numpy as np

        from src.geometry import Frame
        from src.robot.core.arm_capabilities import RobotMode, RobotStatus, SafetyMode
        from src.robot.drivers.ur.pose_adapter import urpose_to_pose
        from src.robot.execution.motion import expecting_next
        from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy, PolicyOutcome
        from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
        from tests.test_a_grasp_judged_ahead_runs_as_it_was_judged import _cell
        from tests.test_jaw_io_gripper import HIGH, CLOSE_PIN, _Timeline, _toggle

        arm, events, _solved = _cell()
        arm._next_leg_mode = "in_settles"
        arm._holds = HeldWorldConfig(standoff=True, carry=True, drop=True)

        def joint_moves(joints: Any, **_said: Any) -> bool:
            _stand(arm, [float(v) for v in joints])
            return True

        def lines(urpose: Any, **_said: Any) -> bool:
            here = [float(v) for v in arm._conn.get_joint_positions()]
            _stand(arm, _configuration(arm, urpose_to_pose(urpose, frame=Frame.BASE), here))
            return True

        arm._conn.moveJ.side_effect = joint_moves
        arm._motion.move_to.side_effect = lines
        arm.get_robot_status = lambda: RobotStatus(RobotMode.RUNNING, SafetyMode.NORMAL, False, False)
        io = _Timeline()
        hand = _toggle(io)
        hand.connect()
        io.events.clear()
        policy = GraspExecutionPolicy(arm=arm, gripper=hand, standoff_mm=80.0, retreat_mm=100.0)
        grasp = GraspPoint(position=np.array([350.0, -150.0, 120.0]), approach=np.array([0.0, 0.0, -1.0]),
                           axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=40.0, score=0.9, frame=GraspFrame.BASE,
                           label="part")
        look = _look(arm)
        with _refreshes() as refresh:
            self.assertEqual("", policy.refusal_ahead(grasp))
            with expecting_next(arm, look):
                report = policy.execute(grasp)
            self.assertIs(PolicyOutcome.EXECUTED, report.outcome, report.motion_message)
            self.assertIsNotNone(arm._next_leg_ahead, "the carry was not judged while the jaws closed")
            carry = arm.move_to_joints(look)
        self.assertIs(MotionStatus.EXECUTED, carry.status, carry.message)
        self.assertEqual(1, refresh.call_count, "a frame was taken between the look and the carry")
        self.assertEqual([HIGH], [event for event in io.events if event[0] == f"out{CLOSE_PIN}"],
                         "the close is one change of DO0")
        self.assertIs(CameraWorldUse.PLANNED, carry.camera_world.use)

    def test_one_refresh_from_the_route_to_the_drop_to_the_way_back_and_one_change_of_do0(self) -> None:
        """⭐ The place with the owner's switches: the line in held in its route's world, the release's one change, the
        line out and the way back judged while the jaws open, both run as judged."""
        from types import SimpleNamespace

        from src.geometry import Frame
        from src.robot.core.arm_capabilities import RobotMode, RobotStatus, SafetyMode
        from src.robot.drivers.ur.pose_adapter import urpose_to_pose
        from src.robot.execution import handling
        from src.robot.execution.handling import HandlingOutcome
        from src.robot.execution.motion import expecting_next
        from tests.test_a_grasp_judged_ahead_runs_as_it_was_judged import _GRASP, _cell
        from tests.test_jaw_io_gripper import CLOSE_PIN, LOW, _Timeline, _toggle

        arm, events, _solved = _cell()
        arm._next_leg_mode = "in_settles"
        arm._holds = HeldWorldConfig(standoff=True, carry=True, drop=True)

        def joint_moves(joints: Any, **_said: Any) -> bool:
            _stand(arm, [float(v) for v in joints])
            return True

        def lines(urpose: Any, **_said: Any) -> bool:
            here = [float(v) for v in arm._conn.get_joint_positions()]
            _stand(arm, _configuration(arm, urpose_to_pose(urpose, frame=Frame.BASE), here))
            return True

        arm._conn.moveJ.side_effect = joint_moves
        arm._motion.move_to.side_effect = lines
        arm.get_robot_status = lambda: RobotStatus(RobotMode.RUNNING, SafetyMode.NORMAL, False, False)
        io = _Timeline()
        hand = _toggle(io)
        hand.connect()
        hand.set_closed(True)
        io.events.clear()
        back = _look(arm)
        with _refreshes() as refresh:
            with expecting_next(arm, back):
                placed = handling.place(SimpleNamespace(arm=arm, gripper=hand), _GRASP, standoff_mm=80.0)
            self.assertIs(HandlingOutcome.EXECUTED, placed.outcome, placed.message)
            self.assertIsNotNone(arm._next_leg_ahead, "the way back was not judged while the jaws opened")
            home = arm.move_to_joints(back)
        self.assertIs(MotionStatus.EXECUTED, home.status, home.message)
        self.assertEqual(1, refresh.call_count, "a frame was taken after the route to the standoff")
        self.assertEqual([LOW], [event for event in io.events if event[0] == f"out{CLOSE_PIN}"],
                         "the release is one change of DO0")


if __name__ == "__main__":
    unittest.main()
