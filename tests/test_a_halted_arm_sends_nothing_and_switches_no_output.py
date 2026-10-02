"""A halted arm sends nothing and switches no output until a person says the cell is clear.

"Sofort anhalten" in the console (``POST /v1/cell/brake``, build plan 1.3.5) latches the arm: ``arm.halt(reason)``. From
then on every motion verb refuses CANCELLED with nothing sent, the bool verbs answer False and the raising verbs raise
``RobotMotionRejected`` carrying that result, and every output is refused with ``ArmHalted`` before anything reaches the
controller, so a toggle hand's DO0 does not change. Only ``clear_halt()``, which the console calls when a person confirms
"Zelle ist frei", ends it.

One write goes out on a halted arm, and only through its own door: ``end_output_pulse`` drives an output low to end a
pulse that began before the halt (a bistable valve's coil, a vacuum blow-off), because a coil left energised until the
cell is cleared is what cooks it. Never for a toggle hand, whose every change of DO0 moves the jaws.

The latch lives on the UR connection object, which a Disconnect and a Connect of the same built arm keep, and on the dummy
arm itself. ``halt()`` sends nothing from the thread that calls it: with ``robot.ur.brake_on_halt`` off (Q1 = A, as
shipped) the move in flight ends as it would have, and nothing after it is sent. ``stop()`` on the UR arm latches it as
well; before, it sent ``stopJ`` and ``stopL`` from whatever thread called it, which did nothing to a synchronous move in
flight. On the dummy ``stop()`` stays the no-op it was.

Each test says what the code before the halt did.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import (
    ArmHalted,
    HaltState,
    JointPositions,
    MotionResult,
    MotionStatus,
    RobotMotionRejected,
    SupportsHalt,
)
from src.robot.core.arm_capabilities import DigitalIOPort
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.drivers.ur import connection as connection_module
from src.robot.drivers.ur.connection import MoveEnd, URConnection
from src.robot.grippers.jaw_io import JawIOGripper
from tests._halt_fakes import RecordingRtde, SimulatedUr, VirtualClock, ur_arm_on

_POSE = Pose(position_mm=np.array([450.0, -120.0, 320.0]), quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
             frame=Frame.BASE, label="halted")
_Q = JointPositions([0.2, -1.0, 1.1, -0.6, 1.45, 0.35])
_REASON = "the operator pressed halt now"


def _answer(verb: Any) -> Any:
    """What a verb answered: a result, a bool, or the exception it raised."""
    try:
        return verb()
    except Exception as exc:  # noqa: BLE001 (a refusal that raises is the answer under test)
        return exc


def _cancelled(answer: Any) -> bool:
    """Whether ``answer`` is a halt's refusal: CANCELLED, False, or a RobotMotionRejected carrying CANCELLED."""
    if isinstance(answer, MotionResult):
        return answer.status is MotionStatus.CANCELLED
    if isinstance(answer, RobotMotionRejected):
        return isinstance(answer.result, MotionResult) and answer.result.status is MotionStatus.CANCELLED
    return answer is False


def _always_open(_question: str) -> str:
    return "open"


class TheLatchOnAUrArmTests(unittest.TestCase):
    """The UR arm: every verb and every output refused once halted, and nothing at all reaches the controller."""

    def _verbs(self, arm: Any) -> dict[str, Any]:
        return {
            "move_to": lambda: arm.move_to(_POSE),
            "move_to linear": lambda: arm.move_to(_POSE, linear=True),
            "move": lambda: arm.move(_POSE),
            "move linear": lambda: arm.move(_POSE, linear=True),
            "move_joint": lambda: arm.move_joint(_Q),
            "move_to_joints": lambda: arm.move_to_joints(_Q),
            "move_linear": lambda: arm.move_linear(_POSE),
            "move_home": lambda: arm.move_home(),
            "move_to_home": lambda: arm.move_to_home(),
            "amove_to": lambda: asyncio.run(arm.amove_to(_POSE)),
            "amove_home": lambda: asyncio.run(arm.amove_home()),
        }

    def test_every_motion_verb_of_a_halted_arm_refuses_cancelled_and_sends_nothing(self) -> None:
        """Red before: there was no latch, so every one of these reached the controller."""
        for planner in ("ik", "curobo"):
            rtde = RecordingRtde()
            arm = ur_arm_on(rtde, planner)
            arm.halt(_REASON)
            verbs = self._verbs(arm)
            if planner == "curobo":
                verbs["move_to_joints_on_the_line"] = lambda: arm.move_to_joints_on_the_line(_Q)
            with arm.without_camera_world("the halt tests wire no camera world"):
                for name, verb in verbs.items():
                    with self.subTest(planner=planner, verb=name):
                        answer = _answer(verb)
                        self.assertTrue(_cancelled(answer), f"{name} answered {answer!r}")
                        if isinstance(answer, (MotionResult, RobotMotionRejected)):
                            said = answer.message if isinstance(answer, MotionResult) else str(answer)
                            self.assertIn(_REASON, said)
                            self.assertIn("halted", said)
            self.assertEqual([], rtde.calls, f"a halted {planner} arm reached its controller")

    def test_the_raw_transports_refuse_too(self) -> None:
        """``arm.motion`` and ``arm.connection`` are below the gates, and still send nothing while halted."""
        rtde = RecordingRtde()
        arm = ur_arm_on(rtde)
        arm.halt(_REASON)
        self.assertFalse(arm.motion.move_to(arm._pose_to_controller(_POSE)))
        self.assertIs(MotionStatus.CANCELLED, arm.motion.last_reject_status)
        self.assertFalse(arm.motion.move_home())
        self.assertFalse(arm.connection.moveJ(list(_Q.tolist())))
        self.assertIs(MoveEnd.REFUSED_HALTED, arm.connection.last_move_end)
        self.assertFalse(arm.connection.moveL([0.45, -0.12, 0.45, 0.0, 3.14, 0.0], 0.1, 0.2))
        self.assertEqual([], rtde.calls)

    def test_a_halted_arm_switches_no_output_and_still_reads_them(self) -> None:
        """Red before: the outputs had no latch to ask."""
        rtde = RecordingRtde()
        arm = ur_arm_on(rtde)
        arm.halt(_REASON)
        for name, switch in {
            "tool DO0 on": lambda: arm.set_digital_output(0, True, port=DigitalIOPort.TOOL),
            "standard DO3 off": lambda: arm.set_digital_output(3, False),
            "analog AO0": lambda: arm.set_analog_output(0, 2.5),
            "the connection's own": lambda: arm.connection.set_digital_out(0, True, "tool"),
        }.items():
            with self.subTest(output=name):
                with self.assertRaises(ArmHalted) as raised:
                    switch()
                self.assertIn(_REASON, str(raised.exception))
        self.assertEqual([], [call for call in rtde.calls if call[0] == "io"])
        self.assertFalse(arm.get_digital_output(0, port=DigitalIOPort.TOOL))
        self.assertEqual([["recv", "getDigitalOutState", [16], {}]], rtde.calls)

    def test_a_toggle_hand_on_a_halted_arm_moves_no_jaws(self) -> None:
        """The owner's Hand-E: single_toggle on tool DO0, where every change moves the jaws. Not one change goes out."""
        sim = SimulatedUr()
        arm = ur_arm_on(sim)
        jaws = JawIOGripper(arm, actuation="single_toggle", close_output_pin=0, pulse_s=0.0, close_settle_s=0.0,
                            min_width_mm=5.0, max_width_mm=49.99, ask=_always_open, sleep=lambda _s: None)
        jaws.connect()
        arm.halt(_REASON)
        sim.calls.clear()
        with self.assertRaises(Exception):
            jaws.set_closed(True)
        self.assertEqual([], [call for call in sim.calls if call[1] == "io"], "DO0 changed on a halted arm")
        self.assertFalse(sim.outputs.get(16, False))

    def test_clear_halt_gives_motion_and_outputs_back(self) -> None:
        rtde = RecordingRtde()
        arm = ur_arm_on(rtde)
        arm.halt(_REASON)
        arm.clear_halt()
        self.assertIsNone(arm.halt_state())
        self.assertIs(MotionStatus.EXECUTED, arm.move_to_joints(_Q).status)
        arm.set_digital_output(0, True, port=DigitalIOPort.TOOL)
        self.assertEqual(["moveJ", "setToolDigitalOut"], [call[1] for call in rtde.calls])

    def test_the_latch_outlives_a_disconnect_and_a_connect_of_the_same_arm(self) -> None:
        """It lives on the connection object, which the arm keeps from build to rebuild (build plan 1.3.5)."""
        rtde = RecordingRtde(overrides={"dash": {"connect": None, "getRobotModel": "UR5", "polyscopeVersion": ""}})
        arm = ur_arm_on(rtde)
        arm.halt(_REASON)
        arm.disconnect()
        self.assertIsNotNone(arm.halt_state())
        fake_modules = {
            "rtde_receive": SimpleNamespace(RTDEReceiveInterface=lambda ip: rtde.recv),
            "rtde_io": SimpleNamespace(RTDEIOInterface=lambda ip: rtde.io),
            "dashboard_client": SimpleNamespace(DashboardClient=lambda ip: rtde.dash),
            "rtde_control": SimpleNamespace(),
        }
        with mock.patch.multiple(connection_module, **fake_modules), \
                mock.patch.object(URConnection, "_open_control", return_value=rtde.ctrl):
            arm.connection.connect()
        self.assertTrue(arm.connection.is_connected)
        state = arm.halt_state()
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(_REASON, state.reason)
        rtde.calls.clear()
        self.assertTrue(_cancelled(arm.move_to_joints(_Q)))
        self.assertEqual([], rtde.calls)

    def test_halt_sends_nothing_and_says_what_it_latched(self) -> None:
        rtde = RecordingRtde()
        arm = ur_arm_on(rtde)
        before = time.time()
        state = arm.halt(_REASON)
        self.assertEqual([], rtde.calls, "halt() reached the controller from the calling thread")
        self.assertIsInstance(state, HaltState)
        self.assertEqual(_REASON, state.reason)
        self.assertGreaterEqual(state.requested_at, before)
        self.assertLessEqual(state.requested_at, time.time())
        self.assertEqual((False, False, None), (state.in_motion, state.braked, state.brake_s))
        self.assertEqual(state, arm.halt_state())
        again = arm.halt("a second press")
        self.assertEqual(_REASON, again.reason, "a second halt replaced the first")

    def test_stop_latches_and_sends_nothing(self) -> None:
        """Red before: ``stop()`` sent stopJ(2.0) and stopL(2.0) from the calling thread and latched nothing."""
        rtde = RecordingRtde()
        arm = ur_arm_on(rtde)
        arm.stop()
        self.assertEqual([], rtde.calls)
        state = arm.halt_state()
        self.assertIsNotNone(state)
        assert state is not None
        self.assertIn("stop()", state.reason)
        self.assertTrue(_cancelled(arm.move_to_joints(_Q)))

    def test_brakes_off_the_move_in_flight_ends_and_nothing_after_it_is_sent(self) -> None:
        """Q1 = A: the synchronous move runs to its end as it always did, and the next one is never sent."""
        clock = VirtualClock()
        sim = SimulatedUr(clock=clock, wait=clock.sleep)
        conn = sim.attach(URConnection("127.0.0.1", vel=0.5, acc=0.8, clock=clock, sleep=clock.sleep))
        clock.at(0.2, lambda: conn.request_halt(_REASON))
        target = [0.6, -1.0, 1.1, -0.6, 1.45, 0.35]
        self.assertTrue(conn.moveJ(target), "the move in flight is not cut short with brakes off")
        np.testing.assert_allclose(sim.q, target)
        state = conn.halt_state()
        assert state is not None
        self.assertEqual((True, False, None, "ran_out"), (state.in_motion, state.braked, state.brake_s, state.brake))
        self.assertFalse(conn.moveJ([0.0, -1.2, 1.3, -0.4, 1.5, 0.2]))
        self.assertIs(MoveEnd.REFUSED_HALTED, conn.last_move_end)
        self.assertEqual(1, len(sim.named("moveJ")), "a move was sent after the halt")
        self.assertEqual([], sim.named("stopJ") + sim.named("stopL"))
        self.assertIs(False, sim.named("moveJ")[0][3][-1], "brakes off sends the move synchronously")

    def test_brakes_off_a_move_in_flight_that_raises_is_unconfirmed_and_never_called_braked(self) -> None:
        """Brakes off, halted while a synchronous move runs, and ur_rtde then raises out of that move: nobody can say the
        arm stands, so the record is unconfirmed and says to press the emergency stop. Nothing braked it, so the record
        never says it was braked. Red before: the sentence said "the move in flight was braked"."""
        clock = VirtualClock()
        sim = SimulatedUr(clock=clock, wait=clock.sleep)
        conn = sim.attach(URConnection("127.0.0.1", vel=0.5, acc=0.8, clock=clock, sleep=clock.sleep))
        clock.at(0.2, lambda: conn.request_halt(_REASON))

        def lost_mid_move(*_args: Any) -> bool:
            clock.sleep(0.3)  # the move runs, and the halt lands inside it
            raise RuntimeError("RTDE control script is not running!")

        sim.moveJ = lost_mid_move  # type: ignore[method-assign]
        with self.assertRaises(RuntimeError):
            conn.moveJ([0.6, -1.0, 1.1, -0.6, 1.45, 0.35])
        state = conn.halt_state()
        assert state is not None
        self.assertEqual((True, False, "unconfirmed"), (state.in_motion, state.braked, state.brake))
        self.assertIn("press the emergency stop", state.render())
        self.assertNotIn("braked", state.render(), "a brake that never ran was named")
        self.assertEqual([], sim.named("stopJ") + sim.named("stopL"), "brakes off sends no stop")

    def test_what_both_arms_answer_is_supports_halt(self) -> None:
        self.assertIsInstance(ur_arm_on(RecordingRtde()), SupportsHalt)
        self.assertIsInstance(DummyRobotArm(), SupportsHalt)
        self.assertNotIsInstance(object(), SupportsHalt)
        self.assertEqual(["reason", "requested_at", "in_motion", "braked", "brake_s", "brake"],
                         [f.name for f in dataclasses.fields(HaltState)])

    def test_a_halted_arm_ends_a_pulse_it_began_and_switches_nothing_else(self) -> None:
        """A bistable valve's coil, or a blow-off, pulsed when the halt came: its output goes back to rest, so the coil is
        not left energised until the cell is cleared (what cooks it). Only that write, and only to low; a toggle hand
        never asks it, because every change of its output moves the jaws. Red before: the latch refused it as well."""
        rtde = RecordingRtde()
        arm = ur_arm_on(rtde)
        arm.halt(_REASON)
        arm.end_output_pulse(0, port=DigitalIOPort.TOOL)
        arm.connection.end_output_pulse(3, "standard")
        self.assertEqual([["io", "setToolDigitalOut", [0, False], {}], ["io", "setStandardDigitalOut", [3, False], {}]],
                         rtde.calls)
        with self.assertRaises(ArmHalted):
            arm.set_digital_output(0, False, port=DigitalIOPort.TOOL)
        self.assertEqual(2, len(rtde.calls), "a plain write went out on a halted arm")
        arm.clear_halt()
        arm.end_output_pulse(1, port=DigitalIOPort.TOOL)
        self.assertEqual(["io", "setToolDigitalOut", [1, False], {}], rtde.calls[-1])

    def test_whether_an_arm_brakes_a_move_in_flight_is_its_setting(self) -> None:
        self.assertFalse(ur_arm_on(RecordingRtde()).brakes_in_motion())
        self.assertTrue(ur_arm_on(RecordingRtde(), brake_on_halt=True).brakes_in_motion())
        self.assertFalse(DummyRobotArm().brakes_in_motion())


class TheLatchOnTheDummyArmTests(unittest.TestCase):
    """The dummy: the same latch, no brake (nothing of it moves), and ``stop()`` stays a no-op."""

    def setUp(self) -> None:
        self.arm = DummyRobotArm()
        self.arm.connect()

    def test_every_verb_of_a_halted_dummy_refuses(self) -> None:
        """Red before: the dummy had no latch, so a halted console's dummy cell kept moving."""
        joints, tcp = self.arm.get_joint_positions(), self.arm.get_tcp_pose()
        state = self.arm.halt(_REASON)
        self.assertEqual((_REASON, False, False, None), (state.reason, state.in_motion, state.braked, state.brake_s))
        for name, verb in {
            "move_to": lambda: self.arm.move_to(_POSE),
            "move_home": lambda: self.arm.move_home(),
            "move": lambda: self.arm.move(_POSE),
            "move_to_joints": lambda: self.arm.move_to_joints(_Q),
            "move_joint": lambda: self.arm.move_joint(_Q),
            "move_linear": lambda: self.arm.move_linear(_POSE),
        }.items():
            with self.subTest(verb=name):
                answer = _answer(verb)
                self.assertTrue(_cancelled(answer) or isinstance(answer, RobotMotionRejected), repr(answer))
                if isinstance(answer, MotionResult):
                    self.assertIn(_REASON, answer.message)
        self.assertIs(joints, self.arm.get_joint_positions())
        self.assertIs(tcp, self.arm.get_tcp_pose())

    def test_stop_stays_a_no_op_on_the_dummy(self) -> None:
        self.arm.stop()
        self.assertIsNone(self.arm.halt_state())
        self.assertIs(MotionStatus.EXECUTED, self.arm.move_to_joints(_Q).status)

    def test_the_latch_outlives_a_disconnect_and_clear_gives_motion_back(self) -> None:
        self.arm.halt(_REASON)
        self.arm.disconnect()
        self.arm.connect()
        self.assertIsNotNone(self.arm.halt_state())
        self.assertFalse(self.arm.move_to(_POSE))
        self.arm.clear_halt()
        self.assertIsNone(self.arm.halt_state())
        self.assertTrue(self.arm.move_to(_POSE))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
