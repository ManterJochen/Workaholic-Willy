"""Watching a pick while it happens, and stopping one between attempts.

Until this seam existed, ``BinPickingOrchestrator.run()`` emitted nothing at all -- a server subscribing
to a live pick received exactly one message, after the pick was over. Two properties have to hold at
once, and they pull in opposite directions:

* **With no listener, nothing changes.** Not "changes little": the ``is None`` test comes before the
  payload is built, so the default path allocates nothing and formats nothing. The U12 soak gate is the
  real proof (it re-runs 2400 synthetic attempts and compares the rolled-up KPIs); these tests pin the
  mechanism the gate cannot see.
* **With a listener, the sequence is honest.** The stages must arrive in the order they happen, the
  terminal event must arrive even when the pick RAISES, and a listener that throws must not be able to
  abort a motion already in flight.

Honesty bucket (2): real orchestrator, scripted perception and a fake arm, no hardware.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import MotionCommand, MotionResult, RobotCapabilities
from src.robot.grasping.loop.pick_loop import (
    BinPickingOrchestrator,
    PickOutcome,
)
from src.robot.grasping.loop.progress import (
    PickProgress,
    PickStage,
    emit,
)
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.perception import PerceptionFrame

_CAPS = RobotCapabilities(
    vendor="dummy", model="dummy", dof=6,
    supports_joint_move=True, supports_linear_move=True, supports_async_move=False,
    has_native_fk=False, has_native_ik=False, has_force_control=False, is_simulated=True,
)


class _Arm:
    def __init__(self) -> None:
        self._tcp = Pose(
            position_mm=np.array([0.0, 0.0, 500.0]),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.BASE,
        )

    @property
    def capabilities(self) -> RobotCapabilities:
        return _CAPS

    def get_tcp_pose(self) -> Pose:
        return self._tcp

    def move(self, pose: Pose, **_: object) -> MotionResult:
        if pose.frame == Frame.BASE:
            self._tcp = pose
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)


class _Perception:
    """Hands back the same frame every time, and counts how often it was asked."""

    def __init__(self, *, segmentations: int = 1) -> None:
        self.calls = 0
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:22, 10:22] = 1
        self._frame = PerceptionFrame(
            depth_map=np.full((32, 32), 500.0, dtype=np.float64),
            intrinsics=np.array(
                [[400.0, 0.0, 16.0], [0.0, 400.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float64
            ),
            segmentations=tuple(SimpleNamespace(mask=mask) for _ in range(segmentations)),
        )

    def acquire(self) -> PerceptionFrame:
        self.calls += 1
        return self._frame


class _Calculator:
    """Returns a fixed result. ``succeed=False`` makes every attempt find nothing."""

    def __init__(self, *, succeed: bool = True, reasons: tuple = ()) -> None:
        self._succeed = succeed
        self._reasons = reasons
        self.calls = 0

    def compute_result(self, *_args: object, **_kwargs: object) -> GraspResult:
        self.calls += 1
        if not self._succeed:
            return GraspResult(candidates=(), reasons=self._reasons, top_score=0.0)
        return GraspResult(
            candidates=(
                GraspPoint(
                    position=np.array([100.0, 50.0, 400.0]),
                    approach=np.array([0.0, 0.0, 1.0]),
                    axis=np.array([1.0, 0.0, 0.0]),
                    grip_width_mm=40.0,
                    score=0.9,
                    frame=GraspFrame.BASE,
                    label="test",
                ),
            ),
            reasons=(),
            top_score=0.9,
        )


def _orchestrator(**kwargs: object) -> BinPickingOrchestrator:
    return BinPickingOrchestrator(
        arm=_Arm(),                    # type: ignore[arg-type]
        calculator=kwargs.pop("calculator", _Calculator()),   # type: ignore[arg-type]
        perception=kwargs.pop("perception", _Perception()),   # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


class _Recorder:
    """A listener that records what it was told."""

    def __init__(self) -> None:
        self.events: list[PickProgress] = []

    def __call__(self, event: PickProgress) -> None:
        self.events.append(event)

    @property
    def stages(self) -> list[PickStage]:
        return [e.stage for e in self.events]


class DefaultPathTests(unittest.TestCase):
    """With nothing attached, the loop is what it always was."""

    def test_a_pick_runs_with_no_listener_and_no_cancel(self) -> None:
        orchestrator = _orchestrator()
        report = orchestrator.run()
        self.assertIs(report.outcome, PickOutcome.EXECUTED)
        self.assertIsNone(orchestrator.on_progress)
        self.assertIsNone(orchestrator.should_cancel)

    def test_emit_builds_nothing_when_no_one_is_listening(self) -> None:
        """The whole default-off argument. If the payload were built and then dropped, every stage
        would cost an allocation on every attempt of every pick, forever, for nobody.

        Asserted by watching the ``PickProgress`` constructor itself. An earlier version of this test
        passed a tripwire object as a keyword ARGUMENT, which Python evaluates before ``emit`` is even
        entered -- it would have passed whatever ``emit`` did.
        """
        from unittest.mock import patch

        with patch("src.robot.grasping.loop.progress.PickProgress") as constructor:
            emit(None, PickStage.RANKED, candidate_count=3)
            constructor.assert_not_called()

            recorder = _Recorder()
            emit(recorder, PickStage.RANKED, candidate_count=3)
            constructor.assert_called_once()


class EventSequenceTests(unittest.TestCase):
    def test_a_successful_pick_reports_every_stage_in_order(self) -> None:
        recorder = _Recorder()
        report = _orchestrator(on_progress=recorder).run()

        self.assertIs(report.outcome, PickOutcome.EXECUTED)
        self.assertEqual(
            recorder.stages,
            [
                PickStage.PICK_STARTED,
                PickStage.ATTEMPT_STARTED,
                PickStage.PERCEIVED,
                PickStage.RANKED,
                PickStage.EXECUTING,
                PickStage.ATTEMPT_FINISHED,
                PickStage.PICK_FINISHED,
            ],
        )

    def test_the_executing_event_carries_where_the_arm_is_about_to_go(self) -> None:
        """The last event before motion. A console that shows this has told the operator the truth
        about what the stack decided, before the arm acts on it."""
        recorder = _Recorder()
        _orchestrator(on_progress=recorder).run()

        executing = next(e for e in recorder.events if e.stage is PickStage.EXECUTING)
        self.assertEqual(executing.position_mm, (100.0, 50.0, 400.0))
        self.assertAlmostEqual(executing.score or 0.0, 0.9)

    def test_a_pick_that_finds_nothing_reports_typed_reasons(self) -> None:
        recorder = _Recorder()
        report = _orchestrator(
            calculator=_Calculator(succeed=False), on_progress=recorder, max_attempts=2
        ).run()

        self.assertIsNot(report.outcome, PickOutcome.EXECUTED)
        self.assertIn(PickStage.NO_CANDIDATE, recorder.stages)
        no_candidate = next(e for e in recorder.events if e.stage is PickStage.NO_CANDIDATE)
        self.assertTrue(no_candidate.action, "an attempt that found nothing must say what it did next")
        self.assertIs(recorder.stages[-1], PickStage.PICK_FINISHED)

    def test_the_stream_is_closed_even_when_the_pick_raises(self) -> None:
        """A console that only ever saw PICK_STARTED shows a spinner forever, which an operator reads
        as 'the arm is still moving' -- the most dangerous thing a robot UI can imply."""
        class _Exploding:
            def acquire(self) -> PerceptionFrame:
                raise RuntimeError("the camera fell over")

        recorder = _Recorder()
        with self.assertRaises(RuntimeError):
            _orchestrator(perception=_Exploding(), on_progress=recorder).run()

        self.assertIs(recorder.stages[-1], PickStage.PICK_FINISHED)
        self.assertEqual(recorder.events[-1].outcome, "raised")

    def test_a_listener_that_throws_cannot_abort_a_pick(self) -> None:
        """It is called inline, on the thread driving the robot. A subscriber's bug is not a reason to
        stop a motion that is already in flight."""
        def _bad(event: PickProgress) -> None:
            raise ValueError("subscriber is broken")

        report = _orchestrator(on_progress=_bad).run()
        self.assertIs(report.outcome, PickOutcome.EXECUTED)

    def test_every_emitted_stage_is_in_the_canonical_set(self) -> None:
        """The contract test. A new stage without updating the enum is a name error, not a silent
        event a console's replay buffer would drop on the floor."""
        recorder = _Recorder()
        _orchestrator(on_progress=recorder).run()
        _orchestrator(calculator=_Calculator(succeed=False), on_progress=recorder).run()

        for event in recorder.events:
            self.assertIsInstance(event.stage, PickStage)
        self.assertLessEqual({e.stage for e in recorder.events}, set(PickStage))


class CancellationTests(unittest.TestCase):
    """"Stop" means: do not begin the next attempt."""

    def test_a_cancel_before_the_first_attempt_stops_the_pick(self) -> None:
        recorder = _Recorder()
        perception = _Perception()
        report = _orchestrator(
            perception=perception, on_progress=recorder, should_cancel=lambda: True
        ).run()

        self.assertIs(report.outcome, PickOutcome.CANCELLED)
        self.assertEqual(perception.calls, 0, "a cancel must land BEFORE the camera is touched")
        self.assertIn(PickStage.CANCELLED, recorder.stages)

    def test_a_cancel_mid_run_ends_after_the_attempt_that_is_running(self) -> None:
        """The difference between a stop button that works and one that looks ignored: a pick with
        five attempts ends after the current one instead of after all five."""
        from src.robot.grasping.types.feedback import GraspFailureReason

        stop = {"now": False}
        # RESCAN_RECOMMENDED, not "no reasons": `_decide_action(())` returns "exhausted", so a
        # reasonless failure ends the pick on attempt 0 and there is no second attempt to cancel.
        calculator = _Calculator(succeed=False, reasons=(GraspFailureReason.RESCAN_RECOMMENDED,))

        def _should_cancel() -> bool:
            return stop["now"]

        class _StopAfterFirst(_Perception):
            def acquire(self) -> PerceptionFrame:
                stop["now"] = True   # the operator presses stop during attempt 0
                return super().acquire()

        perception = _StopAfterFirst()
        report = _orchestrator(
            calculator=calculator, perception=perception,
            should_cancel=_should_cancel, max_attempts=5,
        ).run()

        self.assertIs(report.outcome, PickOutcome.CANCELLED)
        self.assertEqual(perception.calls, 1, "exactly the attempt already running, and no more")

    def test_a_cancelled_pick_is_not_counted_as_a_failure(self) -> None:
        """A stop is a DECISION. Mapping it onto EXECUTION_FAILED would mean every press of the button
        lowered the measured pick rate -- an operator's caution showing up as the cell's incompetence.
        """
        from src.robot.execution.autonomous_grasp.report import (
            _PICK_TO_AUTONOMOUS,
            AutonomousGraspOutcome,
        )

        mapped = _PICK_TO_AUTONOMOUS[PickOutcome.CANCELLED]
        self.assertIs(mapped, AutonomousGraspOutcome.CANCELLED)
        self.assertIsNot(mapped, AutonomousGraspOutcome.EXECUTION_FAILED)
        # ABORTED is the neighbouring outcome and DOES map to a failure -- which is exactly why a
        # cancel needed its own member rather than borrowing that one.
        self.assertIs(
            _PICK_TO_AUTONOMOUS[PickOutcome.ABORTED], AutonomousGraspOutcome.EXECUTION_FAILED
        )

    def test_the_service_refuses_to_start_a_pick_that_was_already_cancelled(self) -> None:
        """A caller running a campaign loops on ``pick()``; refusing to start is the cheapest stop."""
        from src.robot.execution.autonomous_grasp import (
            AutonomousGraspOutcome,
            AutonomousGraspService,
            GraspMode,
        )

        perception = _Perception()
        service = AutonomousGraspService.from_components(
            arm=_Arm(),                # type: ignore[arg-type]
            calculator=_Calculator(),  # type: ignore[arg-type]
            perception=perception,     # type: ignore[arg-type]
            mode=GraspMode.EASY,
        )
        service.set_cancel_check(lambda: True)
        report = service.pick()

        self.assertIs(report.outcome, AutonomousGraspOutcome.CANCELLED)
        self.assertEqual(perception.calls, 0)
        self.assertTrue(report.telemetry.get("cancelled_before_start"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
