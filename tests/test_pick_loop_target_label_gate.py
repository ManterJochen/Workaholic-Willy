"""A target label the scene does not contain is a nameable refusal, not "no valid grasp".

MEASURED 2026-08-19 through the operator console. A run prompted ``"the green cube"`` against the
rehearsal scene produced **8 valid grasp candidates**, skipped every one of them at the hard label
gate, and terminated with ``PickAttempt(reasons=())`` -- an empty tuple. The console rendered
``no_valid_grasp``, which is what it renders when depth is bad, the mask is empty, or the object is
ungraspable. An operator reading that goes to look at the lighting; the actual fix was one word in
the prompt.

The gate itself is correct and stays exactly as it was: refusing to lift an object whose label it
cannot confirm is the whole point of a hard target (see the multi-object campaign). What changes here
is only that the refusal SAYS SO.

Deliberately NOT changed, and asserted below so a later edit has to be deliberate:

* the reason is in neither ``_RESCAN_REASONS`` nor ``_RELOCATE_REASONS``, so ``_decide_action``
  still returns ``"exhausted"`` -- identical control flow, identical motion, identical outcome;
* the recovery dispatcher maps it to no action at all. Not for safety (the cell is fine) but because
  it would be POINTLESS: a detector is deterministic on a static frame, so a rescan re-derives the
  same labels, and ``NEXT_TARGET`` would hand over an object the operator did not ask for -- the
  exact substitution the gate exists to prevent.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np

from src.robot.grasping.loop.pick_loop import (
    _RELOCATE_REASONS,
    _RESCAN_REASONS,
    BinPickingOrchestrator,
)
from src.robot.grasping.loop.progress import PickProgress, PickStage
from src.robot.grasping.motion.execution_policy import PolicyOutcome, PolicyReport
from src.robot.grasping.recovery.orchestrator import RecoveryDispatcher
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.perception import PerceptionFrame


@dataclass
class _Seg:
    mask: np.ndarray
    label: str = ""


class _Perception:
    """A frame whose segmentations carry the labels the test names."""

    def __init__(self, *labels: str) -> None:
        self.labels = labels

    def acquire(self) -> PerceptionFrame:
        segs = []
        for i, label in enumerate(self.labels):
            mask = np.zeros((8, 8), dtype=bool)
            mask[2:6, 2 + i: 4 + i] = True
            segs.append(_Seg(mask=mask, label=label))
        return PerceptionFrame(
            depth_map=np.full((8, 8), 500.0),
            intrinsics=np.array([[500.0, 0, 4.0], [0, 500.0, 4.0], [0, 0, 1.0]]),
            segmentations=tuple(segs),
        )


class _Calculator:
    """Always yields one usable candidate, so a skipped segmentation is the ONLY way to fail."""

    render_debug_images = False
    calls = 0

    def compute_result(self, *_a, **_k) -> GraspResult:
        type(self).calls += 1
        gp = GraspPoint(
            position=np.array([0.0, 0.0, 500.0]), approach=np.array([0.0, 0.0, 1.0]),
            axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=40.0, score=0.9,
            frame=GraspFrame.CAMERA,
        )
        return GraspResult(candidates=(gp,), top_score=0.9)


class _Policy:
    def execute(self, _grasp) -> PolicyReport:
        return PolicyReport(outcome=PolicyOutcome.EXECUTED)


def _orchestrator(perception, label, events=None):
    orch = BinPickingOrchestrator(
        arm=MagicMock(), calculator=_Calculator(), perception=perception,
        policy=_Policy(), max_attempts=1,
    )
    orch.target_label = label
    if events is not None:
        orch.on_progress = events.append
    return orch


class LabelGateNamesItsRefusalTests(unittest.TestCase):
    def test_a_label_no_segmentation_carries_is_named(self) -> None:
        report = _orchestrator(_Perception("a screwdriver", "a bolt"), "the green cube").run()
        attempt = report.attempts[-1]
        self.assertIn(GraspFailureReason.TARGET_LABEL_NOT_FOUND, attempt.reasons)

    def test_an_unlabelled_source_is_the_measured_case(self) -> None:
        # The rehearsal scene: real segmentations, no labels at all. This is what the console hit.
        report = _orchestrator(_Perception("", ""), "the green cube").run()
        self.assertIn(
            GraspFailureReason.TARGET_LABEL_NOT_FOUND, report.attempts[-1].reasons
        )

    def test_the_event_carries_what_was_actually_seen(self) -> None:
        # The actionable half. Without it the line names a failure but not a fix.
        events: list[PickProgress] = []
        _orchestrator(_Perception("a screwdriver", "a bolt"), "the green cube", events).run()
        misses = [e for e in events if e.stage == PickStage.NO_CANDIDATE]
        self.assertEqual(len(misses), 1)
        self.assertEqual(misses[0].extra["target_label"], "the green cube")
        self.assertEqual(misses[0].extra["labels_seen"], ["a screwdriver", "a bolt"])

    def test_a_matching_label_still_picks(self) -> None:
        report = _orchestrator(
            _Perception("a screwdriver", "the green cube"), "the green cube"
        ).run()
        self.assertNotIn(
            GraspFailureReason.TARGET_LABEL_NOT_FOUND, report.attempts[-1].reasons
        )

    def test_no_label_is_untouched(self) -> None:
        # Every existing caller that sets no target label must behave exactly as before.
        report = _orchestrator(_Perception("", ""), None).run()
        self.assertNotIn(
            GraspFailureReason.TARGET_LABEL_NOT_FOUND, report.attempts[-1].reasons
        )

    def test_a_genuine_rejection_still_reports_its_own_reason(self) -> None:
        # The label matched and the CALCULATOR refused: that result's reasons must survive untouched,
        # never be overwritten by the gate's.
        class _Refusing(_Calculator):
            def compute_result(self, *_a, **_k) -> GraspResult:
                return GraspResult(reasons=(GraspFailureReason.ALL_COLLIDED,))

        orch = _orchestrator(_Perception("the green cube"), "the green cube")
        orch.calculator = _Refusing()
        reasons = orch.run().attempts[-1].reasons
        self.assertIn(GraspFailureReason.ALL_COLLIDED, reasons)
        self.assertNotIn(GraspFailureReason.TARGET_LABEL_NOT_FOUND, reasons)


class TheControlFlowIsUnchangedTests(unittest.TestCase):
    """The fix names a failure. It must not move the loop."""

    def test_the_reason_routes_to_neither_rescan_nor_relocate(self) -> None:
        self.assertNotIn(GraspFailureReason.TARGET_LABEL_NOT_FOUND, _RESCAN_REASONS)
        self.assertNotIn(GraspFailureReason.TARGET_LABEL_NOT_FOUND, _RELOCATE_REASONS)

    def test_the_action_is_still_exhausted(self) -> None:
        # What the empty reason tuple produced before, asserted directly.
        orch = _orchestrator(_Perception("a bolt"), "the green cube")
        self.assertEqual(
            orch._decide_action((GraspFailureReason.TARGET_LABEL_NOT_FOUND,)), "exhausted"
        )

    def test_no_calculator_call_is_added_or_removed(self) -> None:
        _Calculator.calls = 0
        _orchestrator(_Perception("a bolt", "a screwdriver"), "the green cube").run()
        self.assertEqual(_Calculator.calls, 0)  # every segmentation skipped, as before

    def test_the_recovery_dispatcher_offers_nothing(self) -> None:
        from src.robot.grasping.recovery.policy import SceneRecoveryAction

        actions = RecoveryDispatcher().actions_for(GraspFailureReason.TARGET_LABEL_NOT_FOUND)
        self.assertEqual(actions, ())
        for forbidden in SceneRecoveryAction:
            with self.subTest(action=forbidden):
                self.assertNotIn(forbidden, actions)


class TheOperatorSentenceTests(unittest.TestCase):
    """UI copy lives in the console, and this is the line that saves the debugging session."""

    def test_it_names_the_prompt_and_what_was_seen(self) -> None:
        from api.runs import _sentence

        _severity, sentence = _sentence(PickProgress(
            stage=PickStage.NO_CANDIDATE, attempt=0, action="exhausted",
            reasons=("target_label_not_found",),
            extra={"target_label": "the green cube", "labels_seen": ["a screwdriver", "a bolt"]},
        ))
        self.assertIn("the green cube", sentence)
        self.assertIn("a screwdriver", sentence)
        self.assertIn("a bolt", sentence)

    def test_an_unlabelled_source_says_so_rather_than_showing_blanks(self) -> None:
        from api.runs import _sentence

        _severity, sentence = _sentence(PickProgress(
            stage=PickStage.NO_CANDIDATE, attempt=0, action="exhausted",
            reasons=("target_label_not_found",),
            extra={"target_label": "the green cube", "labels_seen": ["", ""]},
        ))
        self.assertIn("emits no labels", sentence)

    def test_every_other_rejection_keeps_its_old_line(self) -> None:
        from api.runs import _sentence

        _severity, sentence = _sentence(PickProgress(
            stage=PickStage.NO_CANDIDATE, attempt=0, action="rescan",
            reasons=("all_collided",),
        ))
        self.assertIn("No usable grasp this attempt", sentence)
        self.assertIn("all_collided", sentence)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
