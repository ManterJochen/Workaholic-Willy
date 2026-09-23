"""A controller that cannot move is asked before the jaws are told anything, and a campaign stops on it.

Owner-cell audit, reproduced with fakes on 2026-09-23 (scratchpad ``pick_exec/pstop_campaign.py``). Example 13 on the
owner's toggle Hand-E (``jaw_io`` single_toggle on tool DO0, no feedback) protective-stops during a lift. The pick loop
names that ``CONTROLLER_NOT_OPERATIONAL`` and the service reports it CANCELLED with no fault, so ``PickRun`` started the
next run. That run's ``GraspExecutionPolicy`` detached the part and pulsed DO0 before its first motion asked the
controller anything, and a toggle's pulse opens the jaws: the part dropped from wherever the arm had stopped, likely
while a person walked to the pendant. ``Robot.pick`` had the same order.

Each red-first test below says what the code before this change did. The hand here is the real ``JawIOGripper`` on a
recording tool I/O, so "no pulse" is read off the outputs, not off a double's bookkeeping.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import (
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotCapabilities,
    RobotMode,
    RobotStatus,
    SafetyMode,
)
from src.robot.core.arm_capabilities import DigitalIOPort
from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy, PolicyOutcome
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grippers.jaw_io import JawIOGripper

_RUNNING = RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.NORMAL, protective_stopped=False,
                       emergency_stopped=False)
_PROTECTIVE = RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.PROTECTIVE_STOP,
                          protective_stopped=True, emergency_stopped=False, message="Safetystatus: PROTECTIVE_STOP")
_CAPS = RobotCapabilities(
    vendor="ur", model="ur10", dof=6, supports_joint_move=True, supports_linear_move=True,
    supports_async_move=False, has_native_fk=True, has_native_ik=True, has_force_control=False,
    is_simulated=False,
)


class _IO:
    """The tool I/O the toggle pulses, writing every output onto the log the arm shares."""

    def __init__(self, events: list[Any], inputs: "dict[int, bool] | None" = None) -> None:
        self.events = events
        self.inputs = dict(inputs or {})
        self.do: dict[int, bool] = {0: False, 1: False}

    def set_digital_output(self, pin: int, value: bool, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> None:
        self.do[pin] = bool(value)
        self.events.append(("DO", pin, bool(value)))

    def get_digital_output(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        return self.do[pin]

    def get_digital_input(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        return self.inputs.get(pin, False)

    def set_analog_output(self, pin: int, value: float, *, current: bool = False) -> None:
        return None


def _toggle(events: list[Any], *, closed: bool = False, part_pin: "int | None" = None,
            inputs: "dict[int, bool] | None" = None) -> JawIOGripper:
    """The owner's hand: a Hand-E on the I/O coupling, single_toggle on tool DO0, no feedback, 49.99 / 5.0 mm.

    ``part_pin`` wires a part-present input, which the owner's cell does not have, reading ``inputs``.
    """
    jaws = JawIOGripper(_IO(events, inputs), actuation="single_toggle", close_output_pin=0, pulse_s=0.0,
                        close_settle_s=0.0, min_width_mm=5.0, max_width_mm=49.99, closed_below_mm=49.0,
                        part_present_input_pin=part_pin, sleep=lambda _s: None)
    jaws.connect()
    if closed:
        jaws.set_closed(True)  # closed on the part the previous run lifted
    events.clear()
    return jaws


def _pulses(events: list[Any]) -> int:
    """Rising edges on DO0: each one flips a toggle hand."""
    return sum(1 for e in events if e == ("DO", 0, True))


class _Arm:
    """A UR-like arm whose controller the test scripts; every call it gets lands on the shared log.

    ``statuses`` are answered in order, the last one from then on. ``stop_at_move`` makes that motion, and every one
    after it, fail as a protective stop does, and the controller then reads stopped.
    """

    def __init__(self, events: list[Any], *, statuses: tuple[RobotStatus, ...] = (_RUNNING,),
                 stop_at_move: "int | None" = None) -> None:
        self.events = events
        self._statuses = list(statuses)
        self.stop_at_move = stop_at_move
        self.moves = 0
        self.status_reads = 0
        self._tcp = Pose(position_mm=np.array([0.0, -600.0, 400.0]), quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
                         frame=Frame.BASE, label="home")

    @property
    def capabilities(self) -> RobotCapabilities:
        return _CAPS

    def get_tcp_pose(self) -> Pose:
        return self._tcp

    def get_robot_status(self) -> RobotStatus:
        self.status_reads += 1
        self.events.append("status")
        return self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]

    def recover_from_protective_stop(self) -> bool:  # pragma: no cover - never called, by design
        raise AssertionError("nothing on the pick path may clear a stop")

    def wait_until_steady(self, timeout_s: float = 5.0, poll_interval_s: float = 0.02) -> bool:
        self.events.append("steady")
        return True

    def move(self, pose: Pose, **_: object) -> MotionResult:
        index = self.moves
        self.moves += 1
        self.events.append("move")
        if self.stop_at_move is not None and index >= self.stop_at_move:
            self._statuses = [_PROTECTIVE]
            return MotionResult.failed(MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
                                       message="Driver reported moveL() failure")
        self._tcp = pose
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)

    def detach_payload(self) -> bool:
        self.events.append("detach")
        return True

    def attach_payload(self, width_mm: float) -> bool:
        self.events.append(("attach", round(float(width_mm), 2)))
        return True


def _grasp() -> GraspPoint:
    """A 40 mm part at the owner's work area, closing along base -Y."""
    return GraspPoint(position=np.array([-130.0, -600.0, 80.0]), approach=np.array([0.0, 0.0, -1.0]),
                      axis=np.array([0.0, -1.0, 0.0]), grip_width_mm=40.0, score=0.9, frame=GraspFrame.BASE,
                      label="part")


def _policy(arm: Any, jaws: Any) -> GraspExecutionPolicy:
    """The service's policy on a real cell: jaws opened before every approach, the steady gate on."""
    return GraspExecutionPolicy(arm=arm, gripper=jaws, pre_open_width_mm=49.99, require_steady_before_motion=True)


# ---------------------------------------------------------------------------------------------------
# The grasp policy
# ---------------------------------------------------------------------------------------------------


class ThePolicyAsksTheControllerFirstTests(unittest.TestCase):
    def test_a_stopped_controller_gets_no_pulse_no_detach_and_no_motion(self) -> None:
        """Red before: detach, a pulse on DO0 (the held part dropped), then the steady gate timed out."""
        events: list[Any] = []
        jaws = _toggle(events, closed=True)
        arm = _Arm(events, statuses=(_PROTECTIVE,))

        report = _policy(arm, jaws).execute(_grasp())

        self.assertIs(PolicyOutcome.MOTION_FAILED, report.outcome)
        self.assertIs(MotionStatus.CONTROLLER_REJECTED, report.motion_status)
        self.assertIn("protective_stop=True", report.motion_message)
        self.assertIn("PROTECTIVE_STOP", report.motion_message)
        self.assertEqual(["status"], events, "something was commanded before the controller was asked")
        self.assertEqual((), report.waypoints)

    def test_a_stop_between_the_line_and_the_close_closes_nothing_and_attaches_nothing(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events)
        arm = _Arm(events, statuses=(_RUNNING, _PROTECTIVE))
        policy = _policy(arm, jaws)

        report = policy.execute(_grasp())

        self.assertIs(PolicyOutcome.MOTION_FAILED, report.outcome)
        self.assertIs(MotionStatus.CONTROLLER_REJECTED, report.motion_status)
        self.assertEqual(0, _pulses(events), "the jaws closed on an arm that cannot lift")
        self.assertFalse(any(isinstance(e, tuple) and e[0] == "attach" for e in events))
        self.assertEqual(policy.approach_steps, len(report.waypoints))
        self.assertEqual("status", events[-1])

    def test_a_running_controller_is_asked_twice_and_the_pick_runs(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events)
        arm = _Arm(events)

        report = _policy(arm, jaws).execute(_grasp())

        self.assertIs(PolicyOutcome.EXECUTED, report.outcome, report.motion_message)
        self.assertEqual(2, arm.status_reads)
        self.assertEqual(1, _pulses(events))
        self.assertLess(events.index("status"), events.index("detach"))

    def test_a_status_read_that_raises_leaves_the_pick_with_nothing_commanded(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events, closed=True)
        arm = _Arm(events)

        def _link_down() -> RobotStatus:
            raise RuntimeError("Not connected: call connect() first.")

        arm.get_robot_status = _link_down  # type: ignore[method-assign]

        with self.assertRaises(RuntimeError):
            _policy(arm, jaws).execute(_grasp())
        self.assertEqual([], events)

    def test_a_policy_with_no_gripper_does_not_ask(self) -> None:
        events: list[Any] = []
        arm = _Arm(events, statuses=(_PROTECTIVE,))

        report = GraspExecutionPolicy(arm=arm, gripper=None).execute(_grasp())

        self.assertEqual(0, arm.status_reads)
        self.assertIs(PolicyOutcome.EXECUTED, report.outcome)

    def test_an_arm_that_does_not_report_its_controller_is_not_asked(self) -> None:
        from tests.test_grasp_execution_policy import _FakeArm

        events: list[Any] = []
        jaws = _toggle(events)

        report = GraspExecutionPolicy(arm=_FakeArm(), gripper=jaws, pre_open_width_mm=49.99).execute(_grasp())

        self.assertIs(PolicyOutcome.EXECUTED, report.outcome)
        self.assertEqual(1, _pulses(events))


class ThePickLoopNamesTheStopTests(unittest.TestCase):
    def test_a_stopped_controller_is_controller_not_operational_with_the_jaws_untouched(self) -> None:
        from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator, PickOutcome
        from tests.test_pick_loop_controller_state import _Calculator, _Perception

        events: list[Any] = []
        jaws = _toggle(events, closed=True)
        arm = _Arm(events, statuses=(_PROTECTIVE,))
        loop = BinPickingOrchestrator(arm=arm, calculator=_Calculator(), perception=_Perception(),
                                      policy=_policy(arm, jaws), max_attempts=3)

        report = loop.run()

        self.assertIs(PickOutcome.CONTROLLER_NOT_OPERATIONAL, report.outcome)
        self.assertEqual(1, len(report.attempts), "a stopped cell was retried")
        self.assertEqual(0, _pulses(events))
        self.assertNotIn("detach", events)


# ---------------------------------------------------------------------------------------------------
# The campaign
# ---------------------------------------------------------------------------------------------------


class _Frames:
    """One segmented part, every time."""

    def acquire(self) -> Any:
        from types import SimpleNamespace

        from src.robot.grasping.types.perception import PerceptionFrame

        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:22, 10:22] = 1
        return PerceptionFrame(depth_map=np.full((32, 32), 500.0),
                               intrinsics=np.array([[400.0, 0.0, 16.0], [0.0, 400.0, 16.0], [0.0, 0.0, 1.0]]),
                               segmentations=(SimpleNamespace(mask=mask),))


class _Calculator:
    def compute_result(self, *_args: object, **_kwargs: object) -> Any:
        from src.robot.grasping.types.feedback import GraspResult

        return GraspResult(candidates=(_grasp(),), reasons=(), top_score=0.9)


class _Counted:
    """The service, counting its picks."""

    def __init__(self, service: Any) -> None:
        self.service = service
        self.picks = 0

    def pick(self) -> Any:
        self.picks += 1
        return self.service.pick()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.service, name)


def _service(arm: _Arm, jaws: JawIOGripper) -> Any:
    from src.robot.execution.autonomous_grasp import AutonomousGraspService, GraspMode

    return AutonomousGraspService.from_components(
        arm=arm,  # type: ignore[arg-type]
        calculator=_Calculator(),  # type: ignore[arg-type]
        perception=_Frames(),
        mode=GraspMode.EASY,
        gripper=jaws,
        policy=_policy(arm, jaws),
    )


class TheCampaignStopsOnAStoppedControllerTests(unittest.TestCase):
    def test_a_protective_stop_during_the_lift_ends_the_campaign_with_the_part_still_held(self) -> None:
        """Red before: PickRun called pick() five times, and run 2 pulsed the jaws open before any motion."""
        from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome
        from src.robot.execution.pick_run import PickOutcome, PickRun, Recording

        events: list[Any] = []
        jaws = _toggle(events)
        # Four approach waypoints, then the lift: the lift is motion 4, and it meets the protective stop.
        arm = _Arm(events, stop_at_move=4)
        service = _Counted(_service(arm, jaws))

        run = PickRun.from_service(service, runs=5, recording=Recording.off()).execute()

        self.assertEqual(1, service.picks)
        self.assertEqual([PickOutcome.RAISED] + [PickOutcome.CANCELLED] * 4, [a.outcome for a in run.attempts])
        self.assertIn("the controller cannot move", run.attempts[0].detail)
        self.assertEqual(3, run.exit_code)
        self.assertEqual(1, _pulses(events), "the jaws were pulsed after the stop: the held part dropped")
        last = run.last
        self.assertIs(AutonomousGraspOutcome.CANCELLED, last.outcome)
        self.assertTrue(last.controller_stopped)
        self.assertIsNone(last.fault)
        self.assertIn("controller cannot move", last.render())
        self.assertTrue(last.to_dict()["controller_stopped"])

    def test_a_campaign_on_a_running_controller_runs_every_pick(self) -> None:
        from src.robot.execution.pick_run import PickRun, Recording

        events: list[Any] = []
        jaws = _toggle(events)
        service = _Counted(_service(_Arm(events), jaws))

        run = PickRun.from_service(service, runs=3, recording=Recording.off()).execute()

        self.assertEqual(3, service.picks)
        self.assertEqual(3, run.succeeded)
        self.assertEqual(0, run.exit_code)

    def test_a_report_says_it_ended_on_a_stopped_controller(self) -> None:
        from dataclasses import replace
        from types import SimpleNamespace

        from src.robot.execution.autonomous_grasp.config import GraspMode, _profile_for
        from src.robot.execution.autonomous_grasp.report import AutonomousGraspOutcome, AutonomousGraspReport
        from src.robot.grasping.loop.pick_loop import PickOutcome as LoopOutcome

        base = AutonomousGraspReport(outcome=AutonomousGraspOutcome.CANCELLED, mode=GraspMode.EASY,
                                     profile=_profile_for(GraspMode.EASY))
        self.assertFalse(base.controller_stopped, "an operator's cancel is not a stopped controller")
        loop = replace(base, pick_report=SimpleNamespace(outcome=LoopOutcome.CONTROLLER_NOT_OPERATIONAL))
        self.assertTrue(loop.controller_stopped)
        two_scan = replace(base, telemetry={"low_level_outcome": str(LoopOutcome.CONTROLLER_NOT_OPERATIONAL)})
        self.assertTrue(two_scan.controller_stopped)

    def test_a_double_that_answers_every_attribute_does_not_stop_a_campaign(self) -> None:
        from unittest.mock import MagicMock

        from src.robot.execution.pick_run import PickRun, Recording

        class _Service:
            def pick(self) -> Any:
                report = MagicMock()
                report.outcome = "succeeded"
                report.fault = None
                return report

        run = PickRun.from_service(_Service(), runs=2, recording=Recording.off()).execute()
        self.assertEqual(2, run.succeeded)


class TheTwoScanPathNamesTheStopTests(unittest.TestCase):
    """The closed-loop path drives its own standoff and policy, and says a stopped controller as the loop does."""

    def _service(self, arm: _Arm, jaws: JawIOGripper) -> Any:
        return two_scan_service(arm, jaws)

    def test_a_stop_before_the_close_is_cancelled_and_says_the_controller(self) -> None:
        from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome

        events: list[Any] = []
        jaws = _toggle(events)
        arm = _Arm(events, statuses=(_PROTECTIVE,))

        report = self._service(arm, jaws).pick()

        self.assertIs(AutonomousGraspOutcome.CANCELLED, report.outcome, report.render())
        self.assertTrue(report.controller_stopped)
        self.assertIn("protective_stop=True", report.telemetry["controller"])
        self.assertEqual(0, _pulses(events))

    def test_a_standoff_move_the_controller_refused_is_cancelled(self) -> None:
        from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome

        events: list[Any] = []
        jaws = _toggle(events)
        arm = _Arm(events, stop_at_move=0)

        report = self._service(arm, jaws).pick()

        self.assertIs(AutonomousGraspOutcome.CANCELLED, report.outcome, report.render())
        self.assertEqual("standoff_move", report.telemetry["stage"])
        self.assertTrue(report.controller_stopped)

    def test_a_failed_standoff_on_a_running_controller_is_still_an_execution_failure(self) -> None:
        from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome

        events: list[Any] = []
        jaws = _toggle(events)
        arm = _Arm(events)
        arm.move = lambda pose, **_: MotionResult.failed(  # type: ignore[method-assign]
            MotionStatus.IK_FAILED, MotionCommand.MOVE_TO, target_pose=pose)

        report = self._service(arm, jaws).pick()

        self.assertIs(AutonomousGraspOutcome.EXECUTION_FAILED, report.outcome, report.render())
        self.assertFalse(report.controller_stopped)


def two_scan_service(arm: Any, jaws: Any, *, verifier: Any = None, verification_policy: Any = None) -> Any:
    """The closed-loop service of tests/test_grasp_verification.py on ``arm`` and ``jaws``: a two-scan refine, then the
    policy the service builds itself (jaws opened before the approach), then ``verifier`` (a pass by default)."""
    from src.robot.execution.autonomous_grasp import AutonomousGraspService, GraspMode
    from src.robot.grasping import (
        DefaultPreGraspRefiner,
        GraspVerificationPolicy,
        IdentityFrameResolver,
        NoOpVerifier,
        RefinementPolicy,
    )
    from tests._helpers import _FakePerception
    from tests.test_grasp_verification import (
        _grasp as _verification_grasp,
        _matching_perception_pair,
        _RefinementScriptedCalculator,
        _result,
    )

    refinement = RefinementPolicy(enabled=True, standoff_mm=80.0, max_position_correction_mm=50.0,
                                  max_grip_width_correction_mm=50.0, max_orientation_correction_deg=30.0,
                                  target_match_iou_threshold=0.3)
    initial, refined = _matching_perception_pair()
    calculator = _RefinementScriptedCalculator(initial=_result(_verification_grasp(label="initial")),
                                               refined=_result(_verification_grasp(label="refined")))
    return AutonomousGraspService.from_components(
        arm=arm,
        calculator=calculator,  # type: ignore[arg-type]
        perception=_FakePerception([initial, refined]),
        mode=GraspMode.CLOSED_LOOP,
        gripper=jaws,
        frame_resolver=IdentityFrameResolver(),
        refinement_policy=refinement,
        refiner=DefaultPreGraspRefiner(policy=refinement),
        verification_policy=verification_policy or GraspVerificationPolicy(enabled=True),
        verifier=verifier or NoOpVerifier(),
    )


# ---------------------------------------------------------------------------------------------------
# The robot's hand verbs
# ---------------------------------------------------------------------------------------------------


def _stopped_robot(events: list[Any], *, statuses: tuple[RobotStatus, ...], closed: bool = False) -> Any:
    """A desk arm whose controller the test scripts, and the owner's toggle hand on a recording tool I/O."""
    from src.robot.execution.robot import Robot
    from tests.test_robot_pick_and_place import _Log, _RecordingArm

    class _StatusArm(_RecordingArm):
        def __init__(self) -> None:
            log = _Log()
            log.entries = events  # one log for the arm and the jaws
            super().__init__(log)
            self._statuses = list(statuses)

        def get_robot_status(self) -> RobotStatus:
            self.log.entries.append(("status",))
            return self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]

        def recover_from_protective_stop(self) -> bool:  # pragma: no cover - never called, by design
            raise AssertionError("a hand verb may not clear a stop")

    arm = _StatusArm()
    arm.connect()
    jaws = _toggle(events, closed=closed)
    return Robot.from_parts(arm=arm, gripper=jaws, lock_key=None)


def _moves(events: list[Any]) -> int:
    return sum(1 for e in events if isinstance(e, tuple) and e[0] == "move")


class TheHandVerbsAskTheControllerFirstTests(unittest.TestCase):
    def test_a_pick_on_a_stopped_arm_commands_nothing(self) -> None:
        """Red before: detach, then the pre-open, then the steady gate or the first motion met the stop."""
        from src.robot.execution.handling import HandlingOutcome
        from tests.test_robot_pick_and_place import _pose

        events: list[Any] = []
        robot = _stopped_robot(events, statuses=(_PROTECTIVE,), closed=True)

        report = robot.pick(_pose(), 40.0)

        self.assertIs(HandlingOutcome.REFUSED, report.outcome, report.render())
        self.assertIn("the controller cannot move", report.message)
        self.assertEqual([("status",)], events)
        self.assertEqual((), report.poses)

    def test_a_place_on_a_stopped_arm_keeps_the_part(self) -> None:
        from src.robot.execution.handling import HandlingOutcome
        from tests.test_robot_pick_and_place import _pose

        events: list[Any] = []
        robot = _stopped_robot(events, statuses=(_PROTECTIVE,), closed=True)

        report = robot.place(_pose())

        self.assertIs(HandlingOutcome.REFUSED, report.outcome, report.render())
        self.assertEqual(0, _pulses(events))
        self.assertEqual(0, _moves(events))

    def test_grasp_and_release_on_a_stopped_arm_command_nothing(self) -> None:
        from src.robot.execution.handling import HandOutcome

        events: list[Any] = []
        robot = _stopped_robot(events, statuses=(_PROTECTIVE,), closed=True)

        for verb in (robot.release, lambda: robot.grasp(40.0)):
            report = verb()
            self.assertIs(HandOutcome.REFUSED, report.outcome, report.render())
            self.assertIn("protective_stop=True", report.error)
        self.assertEqual(0, _pulses(events))

    def test_a_stop_during_the_line_down_leaves_the_jaws_open_and_the_arm_down(self) -> None:
        from src.robot.execution.handling import HandlingOutcome, HandOutcome
        from tests.test_robot_pick_and_place import _pose

        events: list[Any] = []
        robot = _stopped_robot(events, statuses=(_RUNNING, _PROTECTIVE))

        report = robot.pick(_pose(), 40.0)

        self.assertIs(HandlingOutcome.REFUSED, report.outcome, report.render())
        assert report.hand is not None
        self.assertIs(HandOutcome.REFUSED, report.hand.outcome)
        self.assertEqual(0, _pulses(events), "the jaws closed on an arm that cannot lift")
        self.assertEqual(2, _moves(events), "the standoff and the line down, and no lift")

    def test_a_status_read_that_raises_refuses_with_nothing_commanded(self) -> None:
        from src.robot.execution.handling import HandlingOutcome
        from tests.test_robot_pick_and_place import _pose

        events: list[Any] = []
        robot = _stopped_robot(events, statuses=(_RUNNING,), closed=True)

        def _link_down() -> RobotStatus:
            raise RuntimeError("Not connected: call connect() first.")

        robot.arm.get_robot_status = _link_down

        report = robot.pick(_pose(), 40.0)

        self.assertIs(HandlingOutcome.REFUSED, report.outcome)
        self.assertIn("could not be read", report.message)
        self.assertEqual([], events)

    def test_a_running_arm_picks_as_before(self) -> None:
        from src.robot.core.gripper import HoldEvidence
        from src.robot.execution.handling import HandlingOutcome
        from tests.test_robot_pick_and_place import _pose

        events: list[Any] = []
        robot = _stopped_robot(events, statuses=(_RUNNING,))

        report = robot.pick(_pose(), 40.0)

        self.assertIs(HandlingOutcome.EXECUTED, report.outcome, report.render())
        self.assertEqual(1, _pulses(events))
        assert report.hand is not None
        self.assertIs(HoldEvidence.UNMEASURED, report.hand.hold)
        self.assertLess(events.index(("status",)), events.index(("detach",)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
