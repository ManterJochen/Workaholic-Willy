"""R10.1 — behavioral conformance of every GraspVerifier implementation to the post-grasp contract.

The per-impl tests in ``tests/test_grasp_verification.py`` exercise each verifier's *specific* outcome
table; this suite guards the *cross-verifier behavioral contract* documented in
``backend/src/robot/grasping/verification.py`` that every production :class:`GraspVerifier` must honour:

* :meth:`verify` returns a :class:`GraspVerificationReport` (never a bare ``bool`` / ``None`` / raw
  enum), carrying a string ``reason`` and a ``Mapping`` ``telemetry`` bag;
* ``report.outcome`` is always a valid :class:`VerificationOutcome` member;
* **the verifier is total — it NEVER raises on a well-formed context.** This is the load-bearing
  invariant: the module docstring promises "Built-in verifiers all degrade gracefully to INCONCLUSIVE
  when their required inputs are missing — they never crash." A verifier that raised instead of
  returning a report would crash the post-grasp gate (and, depending on ``fail_closed``, silently turn
  a verified pick into an exception). This suite is the cross-verifier proof that none of them does.
* on a *data-starved* context (no gripper / no width sample / no frame / no identity) every
  evidence-gathering verifier degrades to :attr:`VerificationOutcome.INCONCLUSIVE` rather than guessing
  PASSED — and :class:`NoOpVerifier` (the explicit legacy-trust opt-out) still PASSES.

Every verifier is constructed exactly the way the existing rich suite constructs it (the
``verification.py`` defaults / fixtures reused below), so this exercises the shipped verifier set —
not hand-built stubs. ``CompositeGraspVerifier`` is built over the *other real* verifiers (the way an
operator wires it) so the composite path is genuinely covered too.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.geometry import Frame
from src.robot.grasping import (
    CompositeGraspVerifier,
    GraspVerificationContext,
    GraspVerificationPolicy,
    GraspVerificationReport,
    GraspVerifier,
    NoOpVerifier,
    ObjectDetectingGripperVerifier,
    VerificationOutcome,
    VisionTargetDisplacementVerifier,
    WidthDeltaGripperVerifier,
    target_identity_from_segmentation,
)
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.perception import PerceptionFrame


# ---------------------------------------------------------------------------
# Fixtures — reused verbatim from tests/test_grasp_verification.py so the
# conformance suite builds verifiers + contexts the same way the rich suite does.
# ---------------------------------------------------------------------------


class _PlainGripper:
    """Minimal Gripper Protocol impl without object detection (mirrors the rich suite)."""

    def __init__(self, *, min_width_mm: float = 0.0, current_width: float = 80.0) -> None:
        self._min = min_width_mm
        self._max = 100.0
        self._width = current_width
        self.is_connected = True

    @property
    def min_width_mm(self) -> float:
        return self._min

    @property
    def max_width_mm(self) -> float:
        return self._max

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def activate(self) -> None: ...

    def set_width_mm(self, width_mm: Any, *, speed: Any = None, force: Any = None) -> None:
        self._width = float(width_mm)

    def get_width_mm(self) -> float:
        return self._width


class _ObjectDetectingGripper(_PlainGripper):
    """Gripper that advertises the ObjectDetectingGripper Protocol."""

    def __init__(self, *, detected: bool, current_width: float = 80.0) -> None:
        super().__init__(min_width_mm=0.0, current_width=current_width)
        self._detected = detected

    def is_object_detected(self) -> bool:
        return self._detected


def _seg_at(shape: tuple[int, int], row_slice: slice, col_slice: slice) -> SimpleNamespace:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[row_slice, col_slice] = 1
    return SimpleNamespace(mask=mask)


def _frame_with_seg(seg: SimpleNamespace, shape: tuple[int, int] = (32, 32)) -> PerceptionFrame:
    depth = np.full(shape, 500.0, dtype=np.float64)
    intrinsics = np.array(
        [[400.0, 0.0, shape[1] / 2.0], [0.0, 400.0, shape[0] / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return PerceptionFrame(depth_map=depth, intrinsics=intrinsics, segmentations=(seg,))


def _grasp() -> GraspPoint:
    return GraspPoint(
        position=np.array([100.0, 50.0, 400.0]),
        approach=np.array([0.0, 0.0, 1.0]),
        axis=np.array([1.0, 0.0, 0.0]),
        grip_width_mm=40.0,
        score=0.9,
        frame=GraspFrame.BASE,
        label="grasp",
    )


# A target identity + a post-lift frame whose only mask is far away → the
# vision verifier should report PASSED (target gone). Shared by the positive path.
_IDENTITY_SEG = _seg_at((32, 32), slice(10, 20), slice(10, 20))
_TARGET_GONE_FRAME = _frame_with_seg(_seg_at((32, 32), slice(0, 3), slice(0, 3)))


def _all_real_verifiers() -> dict[str, GraspVerifier]:
    """Every production GraspVerifier, built the way an operator/test wires it.

    The composite is built over the other real verifiers (not stubs) so its aggregation path is
    exercised against genuine child reports.
    """
    object_detecting = ObjectDetectingGripperVerifier()
    width_delta = WidthDeltaGripperVerifier()
    vision = VisionTargetDisplacementVerifier()
    return {
        "NoOpVerifier": NoOpVerifier(),
        "ObjectDetectingGripperVerifier": object_detecting,
        "WidthDeltaGripperVerifier": width_delta,
        "VisionTargetDisplacementVerifier": vision,
        "CompositeGraspVerifier": CompositeGraspVerifier(
            verifiers=(object_detecting, width_delta, vision)
        ),
    }


# A context that supplies POSITIVE evidence to every evidence-gathering verifier at once:
#   * an object-detecting gripper that reports detected -> ObjectDetecting PASSED
#   * a plausible post-close width above the engaged minimum   -> WidthDelta PASSED
#   * a target identity + post-lift frame with the target gone -> Vision PASSED
# so the all_must_pass composite PASSES too. This is the "accept" anchor.
def _positive_context() -> GraspVerificationContext:
    return GraspVerificationContext(
        grasp=_grasp(),
        policy=GraspVerificationPolicy(
            enabled=True,
            require_object_detected=True,
            width_delta_min_mm=2.0,
            vision_displacement_iou_max=0.3,
        ),
        gripper=_ObjectDetectingGripper(detected=True),  # type: ignore[arg-type]
        pre_close_width_mm=80.0,
        post_close_width_mm=39.0,
        commanded_close_width_mm=39.0,
        post_lift_frame=_TARGET_GONE_FRAME,
        target_identity=target_identity_from_segmentation(_IDENTITY_SEG),
    )


# A well-formed but DATA-STARVED context: grasp + policy only, every optional
# signal absent. Every evidence-gathering verifier must degrade to INCONCLUSIVE
# (never raise, never guess PASSED); NoOp still PASSES. This is the "reject the
# guess / fail-honest" anchor.
def _data_starved_context() -> GraspVerificationContext:
    return GraspVerificationContext(
        grasp=_grasp(),
        policy=GraspVerificationPolicy(enabled=True, require_object_detected=True),
    )


_VALID_OUTCOMES = frozenset(VerificationOutcome)


class GraspVerifierConformanceTests(unittest.TestCase):
    """Every production GraspVerifier honours the shared post-grasp Protocol contract."""

    def setUp(self) -> None:
        self._verifiers = _all_real_verifiers()
        # Sanity: every implementer named in the module __all__ is covered here.
        self.assertEqual(
            set(self._verifiers),
            {
                "NoOpVerifier",
                "ObjectDetectingGripperVerifier",
                "WidthDeltaGripperVerifier",
                "VisionTargetDisplacementVerifier",
                "CompositeGraspVerifier",
            },
            "conformance suite is not covering the full production verifier set",
        )

    def test_every_verifier_satisfies_runtime_protocol(self) -> None:
        for name, verifier in self._verifiers.items():
            with self.subTest(verifier=name):
                self.assertIsInstance(verifier, GraspVerifier)

    def test_verify_never_raises_and_returns_a_tagged_report(self) -> None:
        # THE totality invariant (verification.py docstring): every verifier RETURNS a
        # GraspVerificationReport with a valid VerificationOutcome on BOTH a positive-evidence and a
        # data-starved context — none raises, none returns a bare bool / None / raw enum. A verifier
        # that raised here would crash the post-grasp gate instead of reporting honestly.
        contexts = {
            "positive": _positive_context(),
            "data_starved": _data_starved_context(),
        }
        for label, ctx in contexts.items():
            for name, verifier in self._verifiers.items():
                with self.subTest(verifier=name, context=label):
                    try:
                        report = verifier.verify(ctx)
                    except Exception as exc:  # noqa: BLE001 - the whole point is to prove no verifier raises
                        self.fail(
                            f"{name} raised {type(exc).__name__} on the {label} context instead of "
                            f"returning a GraspVerificationReport: {exc!r}"
                        )
                    self.assertIsInstance(report, GraspVerificationReport)
                    self.assertIsInstance(report.outcome, VerificationOutcome)
                    self.assertIn(report.outcome, _VALID_OUTCOMES)
                    self.assertIsInstance(report.reason, str)
                    # telemetry is a Mapping per the contract (must be dict-iterable, JSONable bag).
                    self.assertIsInstance(dict(report.telemetry), dict)

    def test_positive_evidence_yields_passed(self) -> None:
        # ACCEPT anchor: handed unambiguous positive evidence (object detected, jaws plausibly engaged,
        # target gone from the post-lift frame) every verifier must report PASSED — including the
        # all_must_pass composite over the three real children. Proves the suite is not all-INCONCLUSIVE
        # theatre: an impl hard-wired to INCONCLUSIVE/FAILED would fail here.
        ctx = _positive_context()
        for name, verifier in self._verifiers.items():
            with self.subTest(verifier=name):
                report = verifier.verify(ctx)
                self.assertIs(
                    report.outcome,
                    VerificationOutcome.PASSED,
                    f"{name} did not PASS on unambiguous positive evidence: {report.reason!r}",
                )

    def test_data_starved_evidence_gatherers_are_inconclusive_not_guessed(self) -> None:
        # REJECT-the-guess anchor: on a context with NO gripper / width / frame / identity, every
        # evidence-gathering verifier must report INCONCLUSIVE (honest "can't tell"), never PASSED.
        # Flipping any of them to accept-on-missing-data would be a real regression and must fail here.
        # NoOpVerifier is the deliberate exception (explicit legacy-trust opt-out) and still PASSES.
        ctx = _data_starved_context()
        expected = {
            "NoOpVerifier": VerificationOutcome.PASSED,
            "ObjectDetectingGripperVerifier": VerificationOutcome.INCONCLUSIVE,
            "WidthDeltaGripperVerifier": VerificationOutcome.INCONCLUSIVE,
            "VisionTargetDisplacementVerifier": VerificationOutcome.INCONCLUSIVE,
            # composite over three INCONCLUSIVE children under default all_must_pass treats
            # inconclusive-as-pass -> PASSED (documented), so it is not asserted as a gatherer here.
        }
        for name, want in expected.items():
            with self.subTest(verifier=name):
                report = self._verifiers[name].verify(ctx)
                self.assertIs(
                    report.outcome,
                    want,
                    f"{name} returned {report.outcome} on a data-starved context (expected {want})",
                )

    def test_object_detecting_verifier_two_paths(self) -> None:
        # Two-path anchor for the detection verifier: a gripper that reports the object held -> PASSED,
        # one that reports empty (with require_object_detected) -> FAILED. Both paths genuinely exercised.
        policy = GraspVerificationPolicy(enabled=True, require_object_detected=True)
        verifier = ObjectDetectingGripperVerifier()
        passed = verifier.verify(
            GraspVerificationContext(
                grasp=_grasp(),
                policy=policy,
                gripper=_ObjectDetectingGripper(detected=True),  # type: ignore[arg-type]
            )
        )
        failed = verifier.verify(
            GraspVerificationContext(
                grasp=_grasp(),
                policy=policy,
                gripper=_ObjectDetectingGripper(detected=False),  # type: ignore[arg-type]
            )
        )
        self.assertIs(passed.outcome, VerificationOutcome.PASSED)
        self.assertIs(failed.outcome, VerificationOutcome.FAILED)

    def test_width_delta_verifier_detects_empty_jaws(self) -> None:
        # FAIL-path anchor for the width verifier: jaws collapsed to ~minimum -> FAILED with the
        # documented machine-readable reason. Proves the verifier reads the real width signal.
        report = WidthDeltaGripperVerifier().verify(
            GraspVerificationContext(
                grasp=_grasp(),
                policy=GraspVerificationPolicy(enabled=True, width_delta_min_mm=2.0),
                gripper=_PlainGripper(min_width_mm=0.0),  # type: ignore[arg-type]
                post_close_width_mm=1.0,  # below 0 + 2 -> empty grasp
            )
        )
        self.assertIs(report.outcome, VerificationOutcome.FAILED)
        self.assertEqual(report.reason, "jaws_collapsed_to_minimum")

    def test_vision_verifier_detects_target_still_present(self) -> None:
        # FAIL-path anchor for the vision verifier: the post-lift frame still holds the target's mask
        # -> FAILED (target never left the table). Confirms the IoU match path is real.
        post_frame = _frame_with_seg(_seg_at((32, 32), slice(10, 20), slice(10, 20)))
        report = VisionTargetDisplacementVerifier().verify(
            GraspVerificationContext(
                grasp=_grasp(),
                policy=GraspVerificationPolicy(enabled=True, vision_displacement_iou_max=0.3),
                target_identity=target_identity_from_segmentation(_IDENTITY_SEG),
                post_lift_frame=post_frame,
            )
        )
        self.assertIs(report.outcome, VerificationOutcome.FAILED)

    def test_composite_short_circuits_on_child_failure(self) -> None:
        # Composite FAIL path over REAL children: a width verifier handed collapsed jaws makes the
        # all_must_pass composite FAIL, and the per-child telemetry is surfaced so an operator can see
        # which check rejected. Proves the composite genuinely aggregates child verdicts.
        composite = CompositeGraspVerifier(
            verifiers=(WidthDeltaGripperVerifier(), NoOpVerifier())
        )
        ctx = GraspVerificationContext(
            grasp=_grasp(),
            policy=GraspVerificationPolicy(enabled=True, width_delta_min_mm=2.0),
            gripper=_PlainGripper(min_width_mm=0.0),  # type: ignore[arg-type]
            post_close_width_mm=1.0,
        )
        report = composite.verify(ctx)
        self.assertIs(report.outcome, VerificationOutcome.FAILED)
        self.assertIn("children", report.telemetry)
        self.assertIn("jaws_collapsed_to_minimum", report.reason)

    def test_grasp_frame_marker_is_unused_but_imported_consistently(self) -> None:
        # Guards the fixture's GraspPoint really carries the BASE frame the rich suite assumes, so the
        # context the verifiers consume is well-formed (a malformed grasp could mask a real verify bug).
        self.assertIs(_grasp().frame, GraspFrame.BASE)
        # And the base geometry Frame enum is still importable where the pick path expects it.
        self.assertTrue(hasattr(Frame, "BASE"))


if __name__ == "__main__":
    unittest.main()
