"""A halt ends the pick where the arm stands: no motion after it, and not one change of the toggle's DO0.

The pick the console runs is the grasp policy (``GraspExecutionPolicy``) on the UR arm: a joint move to the standoff, one
line to the grasp, the close, the lifts. Here it runs on a real ``URRobotArm`` whose controller is
``tests._halt_fakes.SimulatedUr`` on a virtual clock, with the owner's hand: a ``JawIOGripper`` single_toggle on tool
DO0, where every change of the output moves the jaws and nothing reads them back.

* Brakes on: a halt during the move to the standoff brakes it; the policy ends MOTION_FAILED with CANCELLED, the arm
  stands where the brake stopped it, nothing more moves, and the jaws are never told anything.
* Brakes off, as shipped (Q1 = A): a halt during the line to the grasp lets the line run out; then the close asks the
  controller first (the gate every hand verb asks), finds the arm halted and closes nothing; no lift follows. The arm
  stands at the part, the jaws open as they were.

Either way the controller is never asked to clear anything, and a person decides what happens next.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.robot.core import MotionStatus
from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy, PolicyOutcome
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grippers.jaw_io import JawIOGripper
from tests._halt_fakes import SimulatedUr, VirtualClock, ur_arm_on

_HERE = [0.0, -1.2, 1.3, -0.4, 1.5, 0.2]
_REASON = "the operator pressed halt now"


def _always_open(_question: str) -> str:
    return "open"


def _grasp() -> GraspPoint:
    """A 40 mm part on the table, approached from above and closing along base -Y."""
    return GraspPoint(position=np.array([450.0, -120.0, 60.0]), approach=np.array([0.0, 0.0, -1.0]),
                      axis=np.array([0.0, -1.0, 0.0]), grip_width_mm=40.0, score=0.9, frame=GraspFrame.BASE,
                      label="part")


def _cell(*, brake: bool) -> tuple[Any, JawIOGripper, SimulatedUr, VirtualClock]:
    clock = VirtualClock()
    sim = SimulatedUr(_HERE, clock=clock, wait=clock.sleep)
    arm = ur_arm_on(sim, "ik", brake_on_halt=brake)
    arm._conn._clock, arm._conn._sleep = clock, clock.sleep
    jaws = JawIOGripper(arm, actuation="single_toggle", close_output_pin=0, pulse_s=0.0, close_settle_s=0.0,
                        min_width_mm=5.0, max_width_mm=49.99, ask=_always_open, sleep=lambda _s: None)
    jaws.connect()
    sim.calls.clear()
    return arm, jaws, sim, clock


def _policy(arm: Any, jaws: JawIOGripper) -> GraspExecutionPolicy:
    return GraspExecutionPolicy(arm=arm, gripper=jaws, pre_open_width_mm=49.99, require_steady_before_motion=True)


def _do0_writes(sim: SimulatedUr) -> list[Any]:
    return [call for call in sim.calls if call[1] == "io"]


class AHaltEndsThePickTests(unittest.TestCase):
    def test_brakes_on_a_halt_on_the_way_to_the_standoff_brakes_it_and_ends_the_pick(self) -> None:
        """Red before: nothing could stop the move, and the line in, the close and the lift followed."""
        arm, jaws, sim, clock = _cell(brake=True)
        clock.at(0.05, lambda: arm.halt(_REASON))
        report = _policy(arm, jaws).execute(_grasp())
        self.assertIs(PolicyOutcome.MOTION_FAILED, report.outcome)
        self.assertIs(MotionStatus.CANCELLED, report.motion_status)
        self.assertIn(_REASON, report.motion_message)
        self.assertEqual(1, len(sim.named("moveJ")))
        self.assertEqual([], sim.named("moveL"), "the line in followed the halt")
        self.assertEqual(1, len(sim.named("stopJ")))
        self.assertEqual([], _do0_writes(sim), "DO0 changed after the halt: the toggle moved the jaws")
        self.assertFalse(jaws.jaws_closed)
        stood = sim.q.copy()
        clock.sleep(2.0)
        np.testing.assert_allclose(sim.q, stood, err_msg="the arm moved on after the pick ended")

    def test_brakes_off_a_halt_on_the_line_in_lets_it_run_out_and_closes_nothing(self) -> None:
        """Red before: the halt read nowhere, so the jaws closed on the part and the lift followed."""
        arm, jaws, sim, clock = _cell(brake=False)
        line = sim.moveL

        def line_then_halt(*args: Any, **kwargs: Any) -> bool:
            clock.at(clock.t + 0.01, lambda: arm.halt(_REASON))
            return line(*args, **kwargs)

        sim.moveL = line_then_halt  # type: ignore[method-assign]
        report = _policy(arm, jaws).execute(_grasp())
        self.assertIs(PolicyOutcome.MOTION_FAILED, report.outcome)
        self.assertIs(MotionStatus.CONTROLLER_REJECTED, report.motion_status)
        self.assertEqual(1, len(sim.named("moveJ")))
        self.assertEqual(1, len(sim.named("moveL")), "a lift followed the halt")
        self.assertEqual([], sim.named("stopJ") + sim.named("stopL"))
        self.assertEqual([], _do0_writes(sim), "the jaws were closed on a halted arm")
        self.assertFalse(jaws.jaws_closed)
        self.assertEqual(2, len(report.waypoints), "the standoff and the line in ran: the arm stands at the part")

    def test_a_halted_arm_is_neither_judged_for_the_retreat_nor_sent_back_up_its_line(self) -> None:
        """The halt is asked before the retreat is judged as if carrying (2026-10-01). Red at the merge with it
        (2026-10-02): the halted arm's judgement came back refused, and the pick tried to back out up the line it came
        down, refused by the latch, CANCELLED instead of the halt's own sentence."""
        arm, jaws, sim, clock = _cell(brake=False)
        judged: list[Any] = []
        real = arm.carried_line_refusal
        arm.carried_line_refusal = lambda *args, **kwargs: (judged.append(args), real(*args, **kwargs))[1]
        line = sim.moveL

        def line_then_halt(*args: Any, **kwargs: Any) -> bool:
            clock.at(clock.t + 0.01, lambda: arm.halt(_REASON))
            return line(*args, **kwargs)

        sim.moveL = line_then_halt  # type: ignore[method-assign]
        report = _policy(arm, jaws).execute(_grasp())
        self.assertIs(MotionStatus.CONTROLLER_REJECTED, report.motion_status)
        self.assertIn(_REASON, report.motion_message or "")
        self.assertEqual([], judged, "a halted arm's retreat was judged")
        self.assertEqual(1, len(sim.named("moveL")), "the pick moved after the halt")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
