"""A motion is judged while the arm settles, and the steady gate waits right before it is sent (map4 motion C4).

On the cell the steady gate cost 0.54 to 0.61 s every time it directly followed a motion: the controller's
``is_steady()`` takes that long to say "at rest", and the judgement of the next motion waited behind it (map4 F5). With
``robot.safety.dwell.gate_at: send`` the cuRobo UR judges first and gates its own sends: it waits for steady right
before the ``moveJ`` or ``moveL``, and where the arm then stands more than 0.5 mm from where the motion was judged from,
it judges the motion again from there. Pinned here, on the real UR driver, its real guard and planner glue, over the
test bench's camera and sidecar (the fixture of ``test_a_grasp_judged_ahead_runs_as_it_was_judged``):

* the order: the judgement, then the gate, then the send, for a route, a line, a joint move and a joint path;
* an arm that came to rest more than 0.5 mm off is judged again; one within it is sent the judged motion;
* a timeout at the gate, and a halt during it, send nothing;
* a route whose first waypoint is where the gate found the arm skips that zero-length first ``moveJ``;
* the verbs leave the gate to such an arm and keep their own everywhere else; ``"verb"``, the default, is unchanged.

Honesty bucket (2): the arm's real judgement on a stand-in controller.
"""

from __future__ import annotations

import time
import unittest
from typing import Any
from unittest.mock import MagicMock

from src.config.schema.robot import RobotConfig
from src.robot.core import JointPositions, MotionCommand, MotionResult, MotionStatus
from src.robot.core.arm_capabilities import HaltState, LineMotion, LineReading
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.execution.motion import gates_its_own_sends, steady_timeout_of
from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
from tests.test_a_grasp_judged_ahead_runs_as_it_was_judged import (
    _GRASP,
    _STANDOFF,
    _cell,
    _configuration,
    _judged_ahead,
    _moved_1_mm,
    _stand,
)
from tests.test_every_verb_meets_the_camera_world import _mesh_backend_available


def _sending(arm: Any, events: "list[str]") -> None:
    """The arm gates its own sends (``gate_at: send``), and its controller's gate and sends are recorded in ``events``."""
    arm._gates_sends = True

    def steady(*_asked: Any) -> bool:
        events.append("steady")
        return True

    def move_j(*_asked: Any, **_said: Any) -> bool:
        events.append("moveJ")
        return True

    def move_l(*_asked: Any, **_said: Any) -> bool:
        events.append("moveL")
        return True

    arm._conn.wait_until_steady.side_effect = steady
    arm._conn.moveJ.side_effect = move_j
    arm._motion.move_to.side_effect = move_l


def _after(events: "list[str]", first: str, then: str) -> bool:
    """Whether the last ``first`` in ``events`` comes before the first ``then`` after it."""
    return first in events and then in events and max(i for i, e in enumerate(events) if e == first) < events.index(then)


class TheGateWaitsRightBeforeTheSendTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_a_route_is_judged_first_and_the_gate_waits_right_before_its_first_moveJ(self) -> None:
        """⭐ Red before: the gate waited before the judgement, behind 0.5 to 0.6 s of an arm settling."""
        arm, events, _solved = _cell()
        _sending(arm, events)
        result = arm.move(_STANDOFF)
        self.assertIs(MotionStatus.EXECUTED, result.status, result.message)
        self.assertTrue(_after(events, "gate_planned_path", "steady"), events)
        self.assertTrue(_after(events, "steady", "moveJ"), events)
        self.assertEqual(1, events.count("steady"))

    def test_a_route_runs_without_its_zero_length_first_moveJ(self) -> None:
        arm, events, _solved = _cell()
        _sending(arm, events)
        self.assertEqual("", _judged_ahead(arm))
        route = arm._judged_ahead.route
        events.clear()
        self.assertTrue(arm.move(_STANDOFF).ok)
        sent = [list(call.args[0]) for call in arm._conn.moveJ.call_args_list]
        self.assertEqual([[float(v) for v in w] for w in route.waypoints[1:]], sent,
                         "the moveJ to where the arm stands went out, or a judged waypoint did not")
        self.assertNotIn("gate_planned_path", events, "the route judged ahead was judged a second time")

    def test_verb_mode_sends_every_waypoint_and_gates_nothing_itself(self) -> None:
        arm, events, _solved = _cell()
        self.assertEqual("", _judged_ahead(arm))
        route = arm._judged_ahead.route
        self.assertTrue(arm.move(_STANDOFF).ok)
        self.assertEqual(len(route.waypoints), arm._conn.moveJ.call_count)
        arm._conn.wait_until_steady.assert_not_called()

    def test_an_arm_that_came_to_rest_half_a_millimetre_off_is_judged_again_from_there(self) -> None:
        arm, events, _solved = _cell()
        _sending(arm, events)
        moved = {"once": False}

        def settles_off(*_asked: Any) -> bool:
            events.append("steady")
            if not moved["once"]:
                moved["once"] = True
                _moved_1_mm(arm)
            return True

        arm._conn.wait_until_steady.side_effect = settles_off
        result = arm.move(_STANDOFF)
        self.assertIs(MotionStatus.EXECUTED, result.status, result.message)
        self.assertEqual(2, events.count("gate_planned_path"), "the route was not judged again from where it came to rest")
        self.assertEqual(2, events.count("steady"))
        here = [float(v) for v in arm._conn.get_joint_positions()]
        first = list(arm._conn.moveJ.call_args_list[0].args[0])
        self.assertNotEqual(here, first, "the zero-length first moveJ went out")

    def test_an_arm_that_came_to_rest_within_half_a_millimetre_runs_the_judged_route(self) -> None:
        arm, events, _solved = _cell()
        _sending(arm, events)
        radii = arm._preflight.joint_radii_mm(arm)

        def settles_near(*_asked: Any) -> bool:
            events.append("steady")
            here = [float(v) for v in arm._conn.get_joint_positions()]
            here[5] += 0.3 / float(radii[5])
            arm._conn.get_joint_positions.return_value = here
            return True

        arm._conn.wait_until_steady.side_effect = settles_near
        self.assertTrue(arm.move(_STANDOFF).ok)
        self.assertEqual(1, events.count("gate_planned_path"))

    def test_a_timeout_at_the_gate_sends_nothing(self) -> None:
        arm, events, _solved = _cell()
        _sending(arm, events)
        arm._conn.wait_until_steady.side_effect = lambda *_asked: False
        result = arm.move(_STANDOFF)
        self.assertIs(MotionStatus.TIMEOUT, result.status)
        self.assertIn("gate_at 'send'", result.message)
        self.assertIn("nothing was sent", result.message)
        arm._conn.moveJ.assert_not_called()

    def test_a_halt_during_the_gate_sends_nothing(self) -> None:
        arm, events, _solved = _cell()
        _sending(arm, events)

        def halted(*_asked: Any) -> bool:
            arm._conn.halt_state.return_value = HaltState(reason="the operator pressed halt now",
                                                          requested_at=time.time())
            arm._conn.halt_requests = 1
            return True

        arm._conn.wait_until_steady.side_effect = halted
        result = arm.move(_STANDOFF)
        self.assertIs(MotionStatus.CANCELLED, result.status, result.message)
        arm._conn.moveJ.assert_not_called()

    def test_an_arm_that_never_comes_to_rest_where_it_was_judged_is_refused(self) -> None:
        arm, events, _solved = _cell()
        _sending(arm, events)

        def keeps_moving(*_asked: Any) -> bool:
            _moved_1_mm(arm)
            return True

        arm._conn.wait_until_steady.side_effect = keeps_moving
        result = arm.move(_STANDOFF)
        self.assertIs(MotionStatus.TIMEOUT, result.status)
        self.assertIn("no judgement of it began where it stands", result.message)
        arm._conn.moveJ.assert_not_called()

    def test_a_line_is_judged_first_and_the_gate_waits_right_before_moveL(self) -> None:
        arm, events, _solved = _cell()
        _stand(arm, _configuration(arm, _STANDOFF, [float(v) for v in arm._conn.get_joint_positions()]))
        _sending(arm, events)
        result = arm.move(_GRASP, linear=True)
        self.assertIs(MotionStatus.EXECUTED, result.status, result.message)
        self.assertTrue(_after(events, "gate_joint_path", "steady"), events)
        self.assertTrue(_after(events, "steady", "moveL"), events)

    def test_a_line_whose_arm_came_to_rest_off_is_judged_again(self) -> None:
        arm, events, solved = _cell()
        _stand(arm, _configuration(arm, _STANDOFF, [float(v) for v in arm._conn.get_joint_positions()]))
        _sending(arm, events)
        moved = {"once": False}

        def settles_off(*_asked: Any) -> bool:
            events.append("steady")
            if not moved["once"]:
                moved["once"] = True
                _moved_1_mm(arm)
            return True

        arm._conn.wait_until_steady.side_effect = settles_off
        self.assertTrue(arm.move(_GRASP, linear=True).ok)
        self.assertEqual(2, events.count("gate_joint_path"), "the line was not judged again")
        self.assertEqual(1, events.count("moveL"))

    def test_a_joint_move_is_judged_first_and_the_gate_waits_right_before_moveJ(self) -> None:
        arm, events, _solved = _cell()
        _sending(arm, events)
        target = JointPositions(tuple(_configuration(arm, _STANDOFF, [float(v) for v in
                                                                        arm._conn.get_joint_positions()])))
        result = arm.move_to_joints(target)
        self.assertIs(MotionStatus.EXECUTED, result.status, result.message)
        self.assertTrue(_after(events, "gate_planned_path", "steady"), events)
        self.assertTrue(_after(events, "steady", "moveJ"), events)

    def test_a_joint_path_waits_for_the_gate_after_its_judgement(self) -> None:
        arm, events, _solved = _cell()
        _sending(arm, events)
        here = [float(v) for v in arm._conn.get_joint_positions()]
        there = list(here)
        there[5] += 0.2
        result = arm.move_through_joints_on_the_line([JointPositions(tuple(there)), JointPositions(tuple(here))])
        self.assertIs(MotionStatus.EXECUTED, result.status, result.message)
        self.assertTrue(_after(events, "gate_planned_path", "steady"), events)
        self.assertTrue(_after(events, "steady", "moveJ"), events)


class _GatedArm:
    """An arm that keeps its lines and gates its own sends, recording who waited for steady."""

    gates_its_own_sends = True

    def __init__(self) -> None:
        self.waited: list[float] = []

    def line_motion(self) -> LineReading:
        return LineReading(LineMotion.CHECKED, "judged")

    def wait_until_steady(self, timeout_s: float = 5.0) -> bool:
        self.waited.append(timeout_s)
        return True

    def move(self, pose: Any, *, linear: bool = False) -> MotionResult:
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)


class TheVerbsLeaveTheGateToAnArmThatGatesItsSendsTests(unittest.TestCase):
    def test_the_policy_asks_such_an_arm_with_no_gate_of_its_own(self) -> None:
        arm = _GatedArm()
        policy = GraspExecutionPolicy(arm=arm, require_steady_before_motion=True)  # type: ignore[arg-type]
        self.assertTrue(policy._drive_to(_STANDOFF).ok)
        self.assertEqual([], arm.waited)

    def test_the_policy_keeps_its_gate_before_any_other_arm(self) -> None:
        arm = _GatedArm()
        arm.gates_its_own_sends = False  # type: ignore[misc]
        policy = GraspExecutionPolicy(arm=arm, require_steady_before_motion=True)  # type: ignore[arg-type]
        self.assertTrue(policy._drive_to(_STANDOFF).ok)
        self.assertEqual([5.0], arm.waited)

    def test_the_robot_verbs_ask_such_an_arm_with_no_gate_of_their_own(self) -> None:
        from types import SimpleNamespace

        from src.robot.execution import motion
        from src.robot.execution.handling import HandlingVerb, _Handling

        for gates, waits in ((True, []), (False, [4.0])):
            with self.subTest(gates_its_own_sends=gates):
                arm = _GatedArm()
                arm.gates_its_own_sends = gates  # type: ignore[misc]
                arm.plans_paths = True  # type: ignore[attr-defined]
                arm.is_connected = True  # type: ignore[attr-defined]
                arm.config = SimpleNamespace(safety=SimpleNamespace(dwell=SimpleNamespace(  # type: ignore[attr-defined]
                    require_steady_before_motion=True, steady_timeout_s=4.0)))
                self.assertTrue(motion.move(arm, _STANDOFF).ok)
                self.assertEqual(waits, arm.waited)
                handling = _Handling(SimpleNamespace(arm=arm, gripper=None), HandlingVerb.PLACE, _STANDOFF, 80.0,
                                     camera_world=MagicMock())
                self.assertEqual(None if gates else 4.0, handling._steady_timeout())
                self.assertEqual(4.0, steady_timeout_of(arm), "what the tree asks stays what it asks, for the push")

    def test_a_double_that_answers_every_attribute_keeps_its_verbs_gate(self) -> None:
        self.assertFalse(gates_its_own_sends(MagicMock()))


class TheGateAtKeyReachesTheArmTests(unittest.TestCase):
    def _arm(self, **safety: Any) -> URRobotArm:
        planner = safety.pop("planner", "curobo")
        # No guard that reads a hand: what is read here is the key, not the guard.
        guards = {"payload": {"enforce": False}, "ik_quality": {"enforce": False}, "self_collision": {"enforce": False}}
        return URRobotArm(RobotConfig.model_validate({"vendor": "ur", "ur": {"motion_planner": planner},
                                                      "safety": {**guards, **safety}}))

    def test_send_on_a_curobo_arm_gates_its_own_sends(self) -> None:
        self.assertTrue(self._arm(dwell={"gate_at": "send"}).gates_its_own_sends)

    def test_verb_the_default_keeps_the_verbs_gate(self) -> None:
        self.assertFalse(self._arm().gates_its_own_sends)
        self.assertEqual("verb", RobotConfig().safety.dwell.gate_at)

    def test_no_gate_asked_is_no_gate_moved(self) -> None:
        self.assertFalse(self._arm(dwell={"gate_at": "send", "require_steady_before_motion": False}).gates_its_own_sends)

    def test_an_ik_arm_keeps_the_verbs_gate(self) -> None:
        """Only a cuRobo arm sends every motion through the sends that gate."""
        self.assertFalse(self._arm(dwell={"gate_at": "send"}, planner="ik").gates_its_own_sends)


if __name__ == "__main__":
    unittest.main()
