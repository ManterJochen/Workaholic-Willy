"""A halted arm is not a stopped controller: the controller stays operational, the arm does not move, and every refusal
says "halted", never "clear the stop".

Build plan 0.1 and L9. The halt latch folds into ``RobotStatus.is_operational``, so the six library gates that ask the
controller (handling, the pick loop, the grasp policy, the push, the locator, freedrive) refuse a halted arm without an
edit; the four outside handling and freedrive also read the latch first and say the halt's own words, as handling does
(the integration of commit 2). ``RobotStatus.controller_operational`` is the controller's four fields alone and stays
true while halted: the API
gates read it, and the halt latch before it, so a halt never reads as a protective stop. Handling's
``_controller_refusal``, which every hand verb and every pick start asks before the jaws are told anything, says the
halt's own sentence: a person confirms the cell is clear, then Restart. "Clear the stop where the arm is visible" was
the sentence a halt read as before, and there is no stop to clear.

``quick_robot_status()`` is what the ready bar polls: the four receive-stream reads and the latch, and no dashboard round
trip (``get_robot_status`` makes one on every call).

The halt is read strictly everywhere: a real ``HaltState``, or ``RobotStatus.halted`` in words. A double that answers
every attribute (a ``MagicMock`` arm) is not halted, and a status from before the halt with no such field reads as it
did, never raising past a refusal that promises a sentence.
"""

from __future__ import annotations

import unittest
from typing import Any

from src.robot.core import (
    HaltState,
    RobotEmergencyStop,
    RobotMode,
    RobotMotionRejected,
    RobotStatus,
    SafetyMode,
    SupportsHalt,
    SupportsRobotStatus,
)
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.drivers.ur.freedrive import raise_unless_operational
from src.robot.execution.handling import _controller_refusal
from src.robot.grasping.motion.execution_policy import _controller_refusal as policy_refusal
from tests._halt_fakes import RecordingRtde, ur_arm_on

_REASON = "the operator pressed halt now"
_RUNNING = RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.NORMAL, protective_stopped=False,
                       emergency_stopped=False)
_PROTECTIVE = RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.PROTECTIVE_STOP,
                          protective_stopped=True, emergency_stopped=False, message="Safetystatus: PROTECTIVE_STOP")


class _Arm:
    """An arm that reports its controller (``SupportsRobotStatus``) and carries the halt latch (``SupportsHalt``)."""

    def __init__(self, status: RobotStatus = _RUNNING) -> None:
        self._status = status
        self._halt: HaltState | None = None
        self.recovered = 0

    def get_robot_status(self) -> RobotStatus:
        reason = self._halt.reason if self._halt is not None else ""
        return RobotStatus(self._status.robot_mode, self._status.safety_mode, self._status.protective_stopped,
                           self._status.emergency_stopped, self._status.message, halted=reason)

    def recover_from_protective_stop(self) -> bool:  # pragma: no cover - never called, by design
        self.recovered += 1
        raise AssertionError("a halt is never cleared by clearing a protective stop")

    def halt(self, reason: str) -> HaltState:
        self._halt = self._halt or HaltState(reason=reason, requested_at=0.0)
        return self._halt

    def clear_halt(self) -> None:
        self._halt = None

    def halt_state(self) -> HaltState | None:
        return self._halt


class TheStatusTests(unittest.TestCase):
    def test_a_halted_status_keeps_its_controller_operational_and_is_not_operational(self) -> None:
        """Red before: RobotStatus had no halt in it, so a latch could not reach the six gates."""
        halted = RobotStatus(RobotMode.RUNNING, SafetyMode.NORMAL, False, False, halted=_REASON)
        self.assertTrue(halted.controller_operational)
        self.assertFalse(halted.is_operational)
        self.assertFalse(halted.is_stopped, "a halt is no controller stop")
        self.assertTrue(_RUNNING.is_operational and _RUNNING.controller_operational)
        self.assertEqual("", _RUNNING.halted)
        self.assertFalse(_PROTECTIVE.controller_operational)

    def test_the_positional_constructor_of_before_still_builds_the_same_status(self) -> None:
        status = RobotStatus(RobotMode.RUNNING, SafetyMode.REDUCED, False, False, "Reduced")
        self.assertEqual(("Reduced", ""), (status.message, status.halted))
        self.assertFalse(status.is_operational)


class TheUrArmReportsTheHaltBesideItsControllerTests(unittest.TestCase):
    def test_get_robot_status_carries_the_halt_and_reads_what_it_read_before(self) -> None:
        rtde = RecordingRtde()
        arm = ur_arm_on(rtde)
        arm.halt(_REASON)
        status = arm.get_robot_status()
        self.assertEqual(_REASON, status.halted)
        self.assertTrue(status.controller_operational)
        self.assertFalse(status.is_operational)
        self.assertEqual(["getRobotMode", "getSafetyMode", "isProtectiveStopped", "isEmergencyStopped", "safetystatus"],
                         [call[1] for call in rtde.calls])

    def test_quick_robot_status_reads_the_receive_stream_and_never_the_dashboard(self) -> None:
        """Red before: there was no such read, and get_robot_status costs a dashboard round trip per call."""
        rtde = RecordingRtde()
        arm = ur_arm_on(rtde)
        quick = arm.quick_robot_status()
        self.assertEqual((RobotMode.RUNNING, SafetyMode.NORMAL, False, False, "", ""),
                         (quick.robot_mode, quick.safety_mode, quick.protective_stopped, quick.emergency_stopped,
                          quick.message, quick.halted))
        self.assertEqual(["getRobotMode", "getSafetyMode", "isProtectiveStopped", "isEmergencyStopped"],
                         [call[1] for call in rtde.calls])
        self.assertEqual({"recv"}, {call[0] for call in rtde.calls})
        arm.halt(_REASON)
        self.assertEqual(_REASON, arm.quick_robot_status().halted)

    def test_quick_robot_status_of_a_closed_connection_is_refused(self) -> None:
        from src.robot.core import RobotConnectionError

        arm = ur_arm_on(RecordingRtde())
        arm._conn._ctrl = arm._conn._recv = None
        with self.assertRaises(RobotConnectionError):
            arm.quick_robot_status()

    def test_a_halted_arm_frees_nobody_and_says_why(self) -> None:
        """Freedrive is refused with the halt's own sentence, not the controller sentence it would have read."""
        arm = ur_arm_on(RecordingRtde())
        arm.halt(_REASON)
        with self.assertRaises(RobotMotionRejected) as raised:
            arm.freedrive()
        self.assertIn("halted", str(raised.exception))
        self.assertIn(_REASON, str(raised.exception))


class TheHandsAreRefusedWithTheHaltsSentenceTests(unittest.TestCase):
    def test_a_halted_arm_is_refused_as_halted_and_never_told_to_clear_a_stop(self) -> None:
        """Red before: a halted controller read operational, or with the latch folded in, "clear the stop"."""
        arm = _Arm()
        arm.halt(_REASON)
        why = _controller_refusal(arm)
        self.assertIn("halted", why)
        self.assertIn(_REASON, why)
        self.assertNotIn("clear the stop", why)
        self.assertIn("nothing was commanded, the jaws included", why)
        self.assertIn("the cell is clear", why)

    def test_a_halted_arm_that_reports_no_controller_is_refused_too(self) -> None:
        """The dummy halts and reports no controller: the hands still refuse it."""
        arm = DummyRobotArm()
        arm.connect()
        self.assertNotIsInstance(arm, SupportsRobotStatus)
        self.assertEqual("", _controller_refusal(arm))
        arm.halt(_REASON)
        self.assertIn("halted", _controller_refusal(arm))

    def test_a_protective_stop_still_reads_as_the_stop_it_is(self) -> None:
        arm = _Arm(_PROTECTIVE)
        why = _controller_refusal(arm)
        self.assertIn("clear the stop", why)
        self.assertNotIn("halted", why)

    def test_a_halt_on_a_stopped_controller_says_both_and_still_clears_nothing(self) -> None:
        arm = _Arm(_PROTECTIVE)
        arm.halt(_REASON)
        why = _controller_refusal(arm)
        self.assertIn("halted", why)
        self.assertIn("protective_stop", why)
        self.assertNotIn("clear the stop where the arm is visible", why)
        self.assertEqual(0, arm.recovered)

    def test_an_operational_arm_is_not_refused(self) -> None:
        self.assertEqual("", _controller_refusal(_Arm()))

    def test_a_halted_arm_whose_controller_cannot_be_read_is_refused_as_halted(self) -> None:
        arm = _Arm()
        arm.halt(_REASON)

        def unreadable() -> RobotStatus:
            raise RuntimeError("RTDE receive timed out")

        arm.get_robot_status = unreadable  # type: ignore[method-assign]
        why = _controller_refusal(arm)
        self.assertIn("halted", why)
        self.assertNotIn("could not be read", why)

    def test_a_mock_arm_is_not_read_as_halted(self) -> None:
        """A double that answers every attribute is not halted, as every other reader of the latch has it. Red before: its
        status's ``halted`` was a truthy mock, and the hands refused it as halted."""
        from unittest.mock import MagicMock, create_autospec

        from src.robot.drivers.ur.arm import URRobotArm

        self.assertEqual("", _controller_refusal(MagicMock()))
        self.assertEqual("", _controller_refusal(create_autospec(URRobotArm, instance=True)))

    def test_a_status_double_from_before_the_halt_is_read_as_before(self) -> None:
        """A status with only the fields it had before the halt: red before, reading ``halted`` off it raised past the
        refusal, which promises a sentence and never a raise."""
        from types import SimpleNamespace

        class _Before:
            def __init__(self, operational: bool) -> None:
                self._operational = operational

            def get_robot_status(self) -> Any:
                return SimpleNamespace(is_operational=self._operational, robot_mode=RobotMode.RUNNING,
                                       safety_mode=SafetyMode.PROTECTIVE_STOP, protective_stopped=True,
                                       emergency_stopped=False, message="")

            def recover_from_protective_stop(self) -> bool:  # pragma: no cover - never called, by design
                raise AssertionError("never asked")

        self.assertIsInstance(_Before(True), SupportsRobotStatus)
        self.assertEqual("", _controller_refusal(_Before(True)))
        self.assertIn("clear the stop", _controller_refusal(_Before(False)))

    def test_a_status_that_says_halted_in_something_other_than_words_is_not_halted(self) -> None:
        class _Odd(_Arm):
            def get_robot_status(self) -> Any:
                from types import SimpleNamespace

                return SimpleNamespace(is_operational=True, halted=1, controller_operational=True)

        self.assertEqual("", _controller_refusal(_Odd()))


class TheLatchReadersTests(unittest.TestCase):
    """What every gate reads of the latch: a real ``HaltState`` or nothing, and never a raise."""

    def test_halt_state_of_counts_only_a_real_latch(self) -> None:
        from src.robot.core.arm_capabilities import halt_state_of

        class _Raising(_Arm):
            def halt_state(self) -> HaltState | None:
                raise RuntimeError("the double broke")

        class _Answering(_Arm):
            def halt_state(self) -> Any:
                return "halted, says a double"

        self.assertIsNone(halt_state_of(object()))
        self.assertIsNone(halt_state_of(_Raising()))
        self.assertIsNone(halt_state_of(_Answering()))
        arm = _Arm()
        self.assertIsNone(halt_state_of(arm))
        arm.halt(_REASON)
        self.assertIs(arm.halt_state(), halt_state_of(arm))

    def test_brakes_in_motion_of_says_true_only_where_the_arm_says_so(self) -> None:
        from src.robot.core.arm_capabilities import brakes_in_motion_of

        class _Raising:
            def brakes_in_motion(self) -> bool:
                raise RuntimeError("the double broke")

        self.assertFalse(brakes_in_motion_of(object()))
        self.assertFalse(brakes_in_motion_of(_Raising()))
        self.assertFalse(brakes_in_motion_of(DummyRobotArm()))
        self.assertFalse(brakes_in_motion_of(ur_arm_on(RecordingRtde())))
        self.assertTrue(brakes_in_motion_of(ur_arm_on(RecordingRtde(), brake_on_halt=True)))

    def test_a_halt_state_says_what_became_of_the_move(self) -> None:
        idle = HaltState(reason=_REASON, requested_at=12.5)
        pending = HaltState(reason=_REASON, requested_at=12.5, in_motion=True)
        ran_out = HaltState(reason=_REASON, requested_at=12.5, in_motion=True, brake="ran_out")
        unconfirmed = HaltState(reason=_REASON, requested_at=12.5, in_motion=True, brake="unconfirmed")
        braked = HaltState(reason=_REASON, requested_at=12.5, in_motion=True, braked=True, brake_s=0.27)
        self.assertEqual(["none", "pending", "ran_out", "unconfirmed", "braked"],
                         [s.brake for s in (idle, pending, ran_out, unconfirmed, braked)],
                         "the outcome is derived from the other fields where it is not given")
        self.assertIn("no move was in flight", idle.render())
        self.assertIn("not ended yet", pending.render())
        self.assertIn("ended without a brake", ran_out.render())
        self.assertIn("press the emergency stop", unconfirmed.render())
        self.assertIn("braked under control", braked.render())
        self.assertIn("0.27 s", braked.render())
        self.assertEqual({"reason": _REASON, "requested_at": 12.5, "in_motion": True, "braked": True, "brake_s": 0.27,
                          "brake": "braked"}, braked.to_dict())
        with self.assertRaises(ValueError):
            HaltState(reason=_REASON, requested_at=12.5, brake="exploded")

    def test_a_ur_arm_whose_connection_latch_cannot_be_read_reads_as_not_halted(self) -> None:
        from unittest.mock import MagicMock

        arm = ur_arm_on(RecordingRtde())
        arm._conn = MagicMock()
        arm._conn.halt_state.side_effect = RuntimeError("the double broke")
        self.assertIsNone(arm.halt_state())


class TheGatesThatAskTheControllerRefuseAHaltedArmTests(unittest.TestCase):
    """The latch reaches the gates through ``is_operational``, with no edit to any of them."""

    def _halted(self) -> Any:
        arm = _Arm()
        arm.halt(_REASON)
        return arm

    def test_the_grasp_policy_refuses_a_halted_arm(self) -> None:
        self.assertNotEqual("", policy_refusal(self._halted()))

    def test_freedrive_refuses_a_halted_controller_status(self) -> None:
        with self.assertRaises(RobotMotionRejected) as raised:
            raise_unless_operational(self._halted().get_robot_status(), "freeing the arm")
        self.assertNotIsInstance(raised.exception, RobotEmergencyStop, "a halt is no emergency stop")

    def test_the_pick_loops_diagnosis_reads_a_halted_arm_as_unable_to_move(self) -> None:
        from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator

        loop = BinPickingOrchestrator.__new__(BinPickingOrchestrator)
        loop.arm = self._halted()
        self.assertIsNotNone(loop._controller_cannot_move())

    def test_the_push_refuses_a_halted_arm(self) -> None:
        """The fifth gate (``push_motion._controller_reading``): the latch reaches it through ``is_operational``."""
        from src.robot.grasping.recovery.push_motion import _controller_reading

        cannot, unreadable = _controller_reading(self._halted())
        self.assertNotEqual("", cannot)
        self.assertEqual("", unreadable)
        self.assertEqual(("", ""), _controller_reading(_Arm()))

    def test_the_locator_refuses_a_halted_arm(self) -> None:
        """The sixth gate (``locator._controller_cannot_move``)."""
        from src.robot.perception.locator import _controller_cannot_move

        self.assertNotEqual("", _controller_cannot_move(self._halted()))
        self.assertEqual("", _controller_cannot_move(_Arm()))

    def test_the_halted_arm_is_a_supports_halt(self) -> None:
        self.assertIsInstance(self._halted(), SupportsHalt)


class EveryGateSaysTheHaltsOwnWordsTests(unittest.TestCase):
    """The four gates that ask the controller outside handling (the grasp policy, the pick loop's diagnosis, the push,
    the locator) read the latch first (``halt_state_of``, then a status that says so in words) and refuse a halted arm
    in the halt's own words, as handling's ``_controller_refusal`` does: why it is halted, and that a person confirms
    the cell is clear, then Restart. Their own sentence is a stopped controller's, "clear the stop where the arm is
    visible", and it sent a person to a pendant with no stop on it.

    The dummy carries the latch and reports no controller, so these gates asked it nothing and let a halted dummy
    through to verbs that refuse it one by one; now each refuses it at the gate, as handling's does."""

    def _gates(self) -> "dict[str, Any]":
        from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator
        from src.robot.grasping.recovery.push_motion import _controller_reading
        from src.robot.perception.locator import _controller_cannot_move

        def diagnosis(arm: Any) -> Any:
            loop = BinPickingOrchestrator.__new__(BinPickingOrchestrator)
            loop.arm = arm
            return loop._controller_cannot_move()

        return {"the grasp policy": policy_refusal, "the pick loop's diagnosis": diagnosis,
                "the push": lambda arm: _controller_reading(arm)[0], "the locator": _controller_cannot_move}

    def _in_the_halts_words(self, words: Any) -> None:
        self.assertIsInstance(words, str)
        self.assertIn(f"halted ({_REASON})", words)
        self.assertIn("confirms the cell is clear", words)
        self.assertNotIn("clear the stop", words.lower())
        self.assertNotIn("clears the stop", words.lower())

    def test_a_halted_arm_is_refused_in_the_halts_words_at_every_gate(self) -> None:
        """Red before: every gate refused, in a stopped controller's words."""
        for name, gate in self._gates().items():
            arm = _Arm()
            arm.halt(_REASON)
            with self.subTest(gate=name):
                self._in_the_halts_words(gate(arm))

    def test_a_halted_dummy_is_refused_at_every_gate(self) -> None:
        """Red before: the dummy reports no controller, so every gate let it through."""
        for name, gate in self._gates().items():
            arm = DummyRobotArm()
            arm.connect()
            arm.halt(_REASON)
            with self.subTest(gate=name):
                self._in_the_halts_words(gate(arm))

    def test_a_controller_status_that_says_halted_is_read_where_the_arm_answers_no_latch(self) -> None:
        class _StatusOnly:
            def get_robot_status(self) -> RobotStatus:
                return RobotStatus(RobotMode.RUNNING, SafetyMode.NORMAL, False, False, halted=_REASON)

            def recover_from_protective_stop(self) -> bool:  # pragma: no cover - never called, by design
                raise AssertionError("a halt is never cleared by clearing a protective stop")

        for name, gate in self._gates().items():
            with self.subTest(gate=name):
                self._in_the_halts_words(gate(_StatusOnly()))

    def test_a_protective_stop_keeps_its_own_words_at_every_gate(self) -> None:
        for name, gate in self._gates().items():
            with self.subTest(gate=name):
                words = gate(_Arm(_PROTECTIVE))
                self.assertIn("stop", str(words).lower())
                self.assertNotIn("halted", str(words))

    def test_an_arm_that_can_move_passes_every_gate(self) -> None:
        connected = DummyRobotArm()
        connected.connect()
        for name, gate in self._gates().items():
            for arm in (_Arm(), DummyRobotArm(), connected):
                with self.subTest(gate=name, arm=type(arm).__name__):
                    self.assertFalse(gate(arm))

    def test_freedrive_refuses_a_halted_controller_in_the_halts_words(self) -> None:
        """Freedrive's own check, asked again inside an open session before the arm is freed (``free()``). Red before:
        it refused, in the controller's words ("the controller reports robot mode running and safety mode normal")."""
        halted = RobotStatus(RobotMode.RUNNING, SafetyMode.NORMAL, False, False, halted=_REASON)
        with self.assertRaises(RobotMotionRejected) as raised:
            raise_unless_operational(halted, "freeing the arm")
        self.assertNotIsInstance(raised.exception, RobotEmergencyStop, "a halt is no emergency stop")
        self._in_the_halts_words(str(raised.exception))
        self.assertIn("freeing the arm", str(raised.exception))

    def test_freedrive_says_both_where_a_halted_arm_stands_at_a_protective_stop(self) -> None:
        both = RobotStatus(RobotMode.RUNNING, SafetyMode.PROTECTIVE_STOP, True, False,
                           "Safetystatus: PROTECTIVE_STOP", halted=_REASON)
        with self.assertRaises(RobotEmergencyStop) as raised:
            raise_unless_operational(both, "freeing the arm")
        said = str(raised.exception)
        self.assertIn(f"halted ({_REASON})", said)
        self.assertIn("protective_stop", said.lower().replace(" ", "_"))
        self.assertIn("Clear the stop at the pendant", said)

    def test_freedrive_still_says_a_stopped_controller_as_before(self) -> None:
        with self.assertRaises(RobotEmergencyStop) as raised:
            raise_unless_operational(_PROTECTIVE, "freeing the arm")
        self.assertTrue(str(raised.exception).startswith(
            "freeing the arm is refused: the controller reports robot mode running and safety mode protective_stop"),
            str(raised.exception))
        self.assertNotIn("halted", str(raised.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
