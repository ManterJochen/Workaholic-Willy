"""Tests for Phase S3 two-scan pre-grasp refinement.

Coverage layers:

* :class:`RefinementPolicy` ctor validation.
* :class:`TargetIdentity` + :func:`target_identity_from_segmentation`.
* :class:`IoUCentroidTargetTracker` match semantics (threshold, ties,
  shape mismatch, centroid cap).
* :class:`DefaultPreGraspRefiner` outcomes: ``SKIPPED``,
  ``TARGET_LOST``, ``NO_GRASP``, ``DIVERGED`` (position / orientation /
  width / frame mismatch), and ``ACCEPTED``.
* :class:`AutonomousGraspService` end-to-end: refined grasp is the one
  executed; divergence and target loss surface the right typed
  outcome and never reach final execution; without a refiner wired
  the CLOSED_LOOP refusal stays in place (legacy behaviour).
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from typing import Optional

import numpy as np

from src.geometry import Frame, Pose, Transform
from src.robot.core import (
    MotionCommand,
    MotionResult,
    RobotCapabilities,
)
from src.robot.execution.autonomous_grasp import (
    AutonomousGraspOutcome,
    AutonomousGraspService,
    GraspMode,
)
from src.robot.grasping import (
    DefaultPreGraspRefiner,
    GraspVerificationPolicy,
    IdentityFrameResolver,
    IoUCentroidTargetTracker,
    NoOpVerifier,
    RefinementOutcome,
    RefinementPolicy,
    TargetIdentity,
    WorldSpacePoseTracker,
    target_identity_from_segmentation,
)
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.perception import PerceptionFrame


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


_REAL_UR_CAPS = RobotCapabilities(
    vendor="ur",
    model="ur5e",
    dof=6,
    supports_joint_move=True,
    supports_linear_move=True,
    supports_async_move=False,
    has_native_fk=True,
    has_native_ik=True,
    has_force_control=False,
    is_simulated=False,
)


class _TypedFakeArm:
    def __init__(self) -> None:
        self._tcp = Pose(
            position_mm=np.array([0.0, 0.0, 500.0]),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.BASE,
            label="home",
        )
        self.move_calls: list[Pose] = []

    @property
    def capabilities(self) -> RobotCapabilities:
        return _REAL_UR_CAPS

    def get_tcp_pose(self) -> Pose:
        return self._tcp

    def move(self, pose: Pose, **_: object) -> MotionResult:
        self.move_calls.append(pose)
        if pose.frame == Frame.BASE:
            self._tcp = pose
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)


class _FakePerception:
    def __init__(self, frames: list[PerceptionFrame]) -> None:
        self._frames = frames
        self.calls = 0

    def acquire(self) -> PerceptionFrame:
        idx = min(self.calls, len(self._frames) - 1)
        self.calls += 1
        return self._frames[idx]


class _ScriptedCalculator:
    """Calculator that returns scripted results in order.

    The orchestrator calls it once per segmentation per frame. We
    return successive results from ``results``; if we run out we
    repeat the last one. Each call records the kwargs so tests can
    assert ``camera_to_base`` plumbing.
    """

    def __init__(self, results: list[GraspResult]) -> None:
        self._results = results
        self.calls = 0
        self.kwargs_seen: list[dict] = []

    def compute_result(self, *_args: object, **kwargs: object) -> GraspResult:
        self.kwargs_seen.append(dict(kwargs))
        idx = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[idx]


def _seg_at(
    shape: tuple[int, int],
    row_slice: slice,
    col_slice: slice,
) -> SimpleNamespace:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[row_slice, col_slice] = 1
    return SimpleNamespace(mask=mask)


def _frame_with_seg(
    seg: SimpleNamespace,
    shape: tuple[int, int] = (32, 32),
) -> PerceptionFrame:
    depth = np.full(shape, 500.0, dtype=np.float64)
    intrinsics = np.array(
        [[400.0, 0.0, shape[1] / 2.0], [0.0, 400.0, shape[0] / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return PerceptionFrame(
        depth_map=depth, intrinsics=intrinsics, segmentations=(seg,)
    )


def _grasp(
    *,
    position: tuple[float, float, float] = (100.0, 50.0, 400.0),
    approach: tuple[float, float, float] = (0.0, 0.0, 1.0),
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    grip_width_mm: float = 40.0,
    score: float = 0.9,
    frame: GraspFrame = GraspFrame.BASE,
    label: str = "grasp",
) -> GraspPoint:
    return GraspPoint(
        position=np.array(position),
        approach=np.array(approach),
        axis=np.array(axis),
        grip_width_mm=grip_width_mm,
        score=score,
        frame=frame,
        label=label,
    )


def _result(grasp: GraspPoint) -> GraspResult:
    return GraspResult(candidates=(grasp,), reasons=(), top_score=grasp.score)


def _empty_result(reasons: tuple = ()) -> GraspResult:
    return GraspResult(candidates=(), reasons=reasons, top_score=0.0)


# ---------------------------------------------------------------------------
# RefinementPolicy validation
# ---------------------------------------------------------------------------


class RefinementPolicyTests(unittest.TestCase):
    def test_defaults_are_valid(self) -> None:
        policy = RefinementPolicy()
        self.assertFalse(policy.enabled)
        self.assertGreater(policy.standoff_mm, 0.0)
        self.assertGreater(policy.max_position_correction_mm, 0.0)

    def test_rejects_negative_standoff(self) -> None:
        with self.assertRaises(ValueError):
            RefinementPolicy(standoff_mm=-1.0)

    def test_rejects_negative_position_bound(self) -> None:
        with self.assertRaises(ValueError):
            RefinementPolicy(max_position_correction_mm=-1.0)

    def test_rejects_negative_orientation_bound(self) -> None:
        with self.assertRaises(ValueError):
            RefinementPolicy(max_orientation_correction_deg=-1.0)

    def test_rejects_negative_width_bound(self) -> None:
        with self.assertRaises(ValueError):
            RefinementPolicy(max_grip_width_correction_mm=-1.0)

    def test_rejects_iou_outside_unit_interval(self) -> None:
        with self.assertRaises(ValueError):
            RefinementPolicy(target_match_iou_threshold=1.5)
        with self.assertRaises(ValueError):
            RefinementPolicy(target_match_iou_threshold=-0.1)


# ---------------------------------------------------------------------------
# TargetIdentity + factory
# ---------------------------------------------------------------------------


class TargetIdentityTests(unittest.TestCase):
    def test_factory_builds_centroid_and_area(self) -> None:
        seg = _seg_at((32, 32), slice(10, 20), slice(10, 20))
        identity = target_identity_from_segmentation(seg)
        # Centroid of a 10x10 box at rows/cols [10..19] is (14.5, 14.5).
        self.assertAlmostEqual(identity.centroid_xy[0], 14.5)
        self.assertAlmostEqual(identity.centroid_xy[1], 14.5)
        self.assertEqual(identity.area_px, 100)

    def test_factory_rejects_empty_mask(self) -> None:
        seg = SimpleNamespace(mask=np.zeros((8, 8), dtype=np.uint8))
        with self.assertRaises(ValueError):
            target_identity_from_segmentation(seg)

    def test_factory_carries_label(self) -> None:
        seg = SimpleNamespace(
            mask=np.ones((4, 4), dtype=np.uint8),
            label="cube",
        )
        identity = target_identity_from_segmentation(seg)
        self.assertEqual(identity.label, "cube")

    def test_direct_construction_rejects_non_2d_mask(self) -> None:
        with self.assertRaises(ValueError):
            TargetIdentity(
                mask=np.zeros((2, 2, 2), dtype=bool),
                centroid_xy=(0.0, 0.0),
                area_px=0,
            )


# ---------------------------------------------------------------------------
# IoUCentroidTargetTracker
# ---------------------------------------------------------------------------


class IoUCentroidTargetTrackerTests(unittest.TestCase):
    def test_returns_highest_iou_above_threshold(self) -> None:
        identity_seg = _seg_at((32, 32), slice(10, 20), slice(10, 20))
        identity = target_identity_from_segmentation(identity_seg)
        # Build a frame with two candidate masks: one matching well,
        # one only marginally overlapping.
        good = _seg_at((32, 32), slice(10, 20), slice(11, 21))  # shifted by 1 col
        bad = _seg_at((32, 32), slice(0, 5), slice(0, 5))  # tiny + far away
        frame = PerceptionFrame(
            depth_map=np.zeros((32, 32)),
            intrinsics=np.eye(3),
            segmentations=(bad, good),
        )
        tracker = IoUCentroidTargetTracker()
        match = tracker.match(identity, frame, iou_threshold=0.3)
        self.assertIsNotNone(match)
        idx, iou = match  # type: ignore[misc]
        self.assertEqual(idx, 1)  # the 'good' one
        self.assertGreater(iou, 0.3)

    def test_returns_none_when_no_candidate_clears_threshold(self) -> None:
        identity_seg = _seg_at((32, 32), slice(10, 20), slice(10, 20))
        identity = target_identity_from_segmentation(identity_seg)
        far = _seg_at((32, 32), slice(0, 3), slice(0, 3))
        frame = PerceptionFrame(
            depth_map=np.zeros((32, 32)),
            intrinsics=np.eye(3),
            segmentations=(far,),
        )
        tracker = IoUCentroidTargetTracker()
        self.assertIsNone(tracker.match(identity, frame, iou_threshold=0.5))

    def test_rejects_match_beyond_centroid_cap(self) -> None:
        # Build a fake mask that has high IoU when sizes match but whose
        # centroid is far away. To engineer this, copy the identity
        # mask but stuff it at a different location: IoU will be 0 in
        # reality. So we use a different trick: a candidate whose
        # mask completely overlaps the identity (IoU = 1.0) but whose
        # centroid is forced different. Mathematically impossible if
        # the mask is identical; so instead we craft a candidate whose
        # mask passes IoU but is offset enough to exceed the cap.
        identity_seg = _seg_at((40, 40), slice(15, 25), slice(15, 25))
        identity = target_identity_from_segmentation(identity_seg)
        cand_mask = _seg_at((40, 40), slice(16, 26), slice(16, 26))
        # Identity centroid ~ (19.5, 19.5); candidate centroid ~ (20.5, 20.5).
        # Distance ~ sqrt(2). Set max_centroid_distance to something
        # tighter than that to force rejection.
        frame = PerceptionFrame(
            depth_map=np.zeros((40, 40)),
            intrinsics=np.eye(3),
            segmentations=(cand_mask,),
        )
        tracker = IoUCentroidTargetTracker(max_centroid_distance_px=1.0)
        self.assertIsNone(tracker.match(identity, frame, iou_threshold=0.5))

    def test_shape_mismatch_is_zero_iou(self) -> None:
        identity = TargetIdentity(
            mask=np.ones((10, 10), dtype=bool),
            centroid_xy=(4.5, 4.5),
            area_px=100,
        )
        wrong_shape = SimpleNamespace(
            mask=np.ones((20, 20), dtype=bool),
        )
        frame = PerceptionFrame(
            depth_map=np.zeros((20, 20)),
            intrinsics=np.eye(3),
            segmentations=(wrong_shape,),
        )
        tracker = IoUCentroidTargetTracker()
        self.assertIsNone(tracker.match(identity, frame, iou_threshold=0.1))


# ---------------------------------------------------------------------------
# DefaultPreGraspRefiner
# ---------------------------------------------------------------------------


class DefaultPreGraspRefinerTests(unittest.TestCase):
    def _identity_for(self, seg: SimpleNamespace) -> TargetIdentity:
        return target_identity_from_segmentation(seg)

    def test_skipped_when_disabled(self) -> None:
        policy = RefinementPolicy(enabled=False)
        refiner = DefaultPreGraspRefiner(policy=policy)
        seg = _seg_at((32, 32), slice(10, 20), slice(10, 20))
        frame = _frame_with_seg(seg)
        report = refiner.refine(
            initial_grasp=_grasp(),
            initial_frame=frame,
            refined_frame=frame,
            target_identity=self._identity_for(seg),
            calculator=_ScriptedCalculator([_result(_grasp())]),
        )
        self.assertIs(report.outcome, RefinementOutcome.SKIPPED)
        self.assertIsNone(report.refined_grasp)

    def test_target_lost_when_no_match(self) -> None:
        policy = RefinementPolicy(enabled=True, target_match_iou_threshold=0.9)
        refiner = DefaultPreGraspRefiner(policy=policy)
        seg = _seg_at((32, 32), slice(10, 20), slice(10, 20))
        # Refined frame has a completely disjoint mask.
        disjoint = _seg_at((32, 32), slice(0, 3), slice(0, 3))
        refined = _frame_with_seg(disjoint)
        report = refiner.refine(
            initial_grasp=_grasp(),
            initial_frame=_frame_with_seg(seg),
            refined_frame=refined,
            target_identity=self._identity_for(seg),
            calculator=_ScriptedCalculator([_result(_grasp())]),
        )
        self.assertIs(report.outcome, RefinementOutcome.TARGET_LOST)
        self.assertIs(
            report.failure_reason, GraspFailureReason.TARGET_LOST_DURING_REFINE
        )

    def test_no_grasp_when_recompute_empty(self) -> None:
        policy = RefinementPolicy(enabled=True)
        refiner = DefaultPreGraspRefiner(policy=policy)
        seg = _seg_at((32, 32), slice(10, 20), slice(10, 20))
        frame = _frame_with_seg(seg)
        calc = _ScriptedCalculator(
            [_empty_result(reasons=(GraspFailureReason.IK_FAILED,))]
        )
        report = refiner.refine(
            initial_grasp=_grasp(),
            initial_frame=frame,
            refined_frame=frame,
            target_identity=self._identity_for(seg),
            calculator=calc,
        )
        self.assertIs(report.outcome, RefinementOutcome.NO_GRASP)

    def test_accepted_within_bounds(self) -> None:
        policy = RefinementPolicy(enabled=True)
        refiner = DefaultPreGraspRefiner(policy=policy)
        seg = _seg_at((32, 32), slice(10, 20), slice(10, 20))
        frame = _frame_with_seg(seg)
        initial = _grasp(position=(100.0, 50.0, 400.0), grip_width_mm=40.0)
        refined = _grasp(
            position=(102.0, 51.0, 400.5),
            grip_width_mm=41.0,
            label="refined",
        )
        calc = _ScriptedCalculator([_result(refined)])
        report = refiner.refine(
            initial_grasp=initial,
            initial_frame=frame,
            refined_frame=frame,
            target_identity=self._identity_for(seg),
            calculator=calc,
        )
        self.assertIs(report.outcome, RefinementOutcome.ACCEPTED)
        self.assertIsNotNone(report.refined_grasp)
        assert report.refined_grasp is not None
        self.assertEqual(report.refined_grasp.label, "refined")
        self.assertLess(report.position_delta_mm or math.inf, 5.0)
        self.assertLess(report.grip_width_delta_mm or math.inf, 5.0)

    def test_diverged_position(self) -> None:
        policy = RefinementPolicy(
            enabled=True, max_position_correction_mm=5.0
        )
        refiner = DefaultPreGraspRefiner(policy=policy)
        seg = _seg_at((32, 32), slice(10, 20), slice(10, 20))
        frame = _frame_with_seg(seg)
        initial = _grasp(position=(100.0, 50.0, 400.0))
        # 50 mm shift is well beyond the 5 mm cap.
        refined = _grasp(position=(150.0, 50.0, 400.0))
        calc = _ScriptedCalculator([_result(refined)])
        report = refiner.refine(
            initial_grasp=initial,
            initial_frame=frame,
            refined_frame=frame,
            target_identity=self._identity_for(seg),
            calculator=calc,
        )
        self.assertIs(report.outcome, RefinementOutcome.DIVERGED)
        self.assertIs(
            report.failure_reason, GraspFailureReason.REFINEMENT_DIVERGED
        )
        self.assertIn("position", report.telemetry.get("violated_bounds", ()))

    def test_diverged_orientation(self) -> None:
        policy = RefinementPolicy(
            enabled=True, max_orientation_correction_deg=5.0
        )
        refiner = DefaultPreGraspRefiner(policy=policy)
        seg = _seg_at((32, 32), slice(10, 20), slice(10, 20))
        frame = _frame_with_seg(seg)
        initial = _grasp(approach=(0.0, 0.0, 1.0), axis=(1.0, 0.0, 0.0))
        # 90 deg axis flip: well beyond a 5 deg cap.
        refined = _grasp(approach=(0.0, 0.0, 1.0), axis=(0.0, 1.0, 0.0))
        calc = _ScriptedCalculator([_result(refined)])
        report = refiner.refine(
            initial_grasp=initial,
            initial_frame=frame,
            refined_frame=frame,
            target_identity=self._identity_for(seg),
            calculator=calc,
        )
        self.assertIs(report.outcome, RefinementOutcome.DIVERGED)
        self.assertIn(
            "orientation", report.telemetry.get("violated_bounds", ())
        )

    def test_diverged_grip_width(self) -> None:
        policy = RefinementPolicy(
            enabled=True, max_grip_width_correction_mm=2.0
        )
        refiner = DefaultPreGraspRefiner(policy=policy)
        seg = _seg_at((32, 32), slice(10, 20), slice(10, 20))
        frame = _frame_with_seg(seg)
        initial = _grasp(grip_width_mm=40.0)
        refined = _grasp(grip_width_mm=55.0)  # 15 mm beyond 2 mm cap
        calc = _ScriptedCalculator([_result(refined)])
        report = refiner.refine(
            initial_grasp=initial,
            initial_frame=frame,
            refined_frame=frame,
            target_identity=self._identity_for(seg),
            calculator=calc,
        )
        self.assertIs(report.outcome, RefinementOutcome.DIVERGED)
        self.assertIn(
            "grip_width", report.telemetry.get("violated_bounds", ())
        )

    def test_seam0_cap_boundary_exact_deltas(self) -> None:
        # R4 K4 Seam-0: pin the EXACT correction deltas + the violated-bounds tuple order at the cap
        # boundary BEFORE extracting _measure_correction_deltas. The per-bound DIVERGED tests above assert
        # only outcome + membership; the exact float arithmetic and the ['position','orientation','grip_width']
        # order are golden-blind there. All three bounds diverge here with known-integer deltas.
        policy = RefinementPolicy(
            enabled=True,
            max_position_correction_mm=5.0,
            max_orientation_correction_deg=5.0,
            max_grip_width_correction_mm=2.0,
        )
        refiner = DefaultPreGraspRefiner(policy=policy)
        seg = _seg_at((32, 32), slice(10, 20), slice(10, 20))
        frame = _frame_with_seg(seg)
        initial = _grasp(
            position=(100.0, 50.0, 400.0),
            approach=(0.0, 0.0, 1.0),
            axis=(1.0, 0.0, 0.0),
            grip_width_mm=40.0,
        )
        refined = _grasp(
            position=(150.0, 50.0, 400.0),  # +50 mm
            approach=(0.0, 0.0, 1.0),
            axis=(0.0, 1.0, 0.0),  # 90 deg axis flip
            grip_width_mm=55.0,  # +15 mm
        )
        calc = _ScriptedCalculator([_result(refined)])
        report = refiner.refine(
            initial_grasp=initial,
            initial_frame=frame,
            refined_frame=frame,
            target_identity=self._identity_for(seg),
            calculator=calc,
        )
        self.assertIs(report.outcome, RefinementOutcome.DIVERGED)
        self.assertEqual(report.position_delta_mm, 50.0)
        self.assertEqual(report.orientation_delta_deg, 90.0)
        self.assertEqual(report.grip_width_delta_mm, 15.0)
        self.assertEqual(
            report.telemetry.get("violated_bounds"),
            ("position", "orientation", "grip_width"),
        )

    def test_diverged_frame_mismatch(self) -> None:
        policy = RefinementPolicy(enabled=True)
        refiner = DefaultPreGraspRefiner(policy=policy)
        seg = _seg_at((32, 32), slice(10, 20), slice(10, 20))
        frame = _frame_with_seg(seg)
        initial = _grasp(frame=GraspFrame.BASE)
        refined = _grasp(frame=GraspFrame.CAMERA)  # frame mismatch
        calc = _ScriptedCalculator([_result(refined)])
        report = refiner.refine(
            initial_grasp=initial,
            initial_frame=frame,
            refined_frame=frame,
            target_identity=self._identity_for(seg),
            calculator=calc,
        )
        self.assertIs(report.outcome, RefinementOutcome.DIVERGED)
        self.assertEqual(
            report.telemetry.get("reason"), "grasp_frame_mismatch"
        )

    def test_forwards_camera_to_base_to_calculator(self) -> None:
        from src.geometry import Transform

        policy = RefinementPolicy(enabled=True)
        refiner = DefaultPreGraspRefiner(policy=policy)
        seg = _seg_at((32, 32), slice(10, 20), slice(10, 20))
        frame = _frame_with_seg(seg)
        t = Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE)
        calc = _ScriptedCalculator([_result(_grasp())])
        refiner.refine(
            initial_grasp=_grasp(),
            initial_frame=frame,
            refined_frame=frame,
            target_identity=self._identity_for(seg),
            calculator=calc,
            camera_to_base=t,
        )
        self.assertEqual(len(calc.kwargs_seen), 1)
        self.assertIs(calc.kwargs_seen[0].get("camera_to_base"), t)


# ---------------------------------------------------------------------------
# AutonomousGraspService end-to-end refinement
# ---------------------------------------------------------------------------


class _RefinementScriptedCalculator:
    """Two-stage calculator for end-to-end tests.

    The orchestrator calls the calculator during ``_best_result_over_segmentations``
    (initial frame) and again during the refiner (refined frame). We
    return the initial grasp first and the refined grasp second.
    """

    def __init__(
        self,
        *,
        initial: GraspResult,
        refined: GraspResult,
    ) -> None:
        self._initial = initial
        self._refined = refined
        self.calls = 0
        self.kwargs_seen: list[dict] = []

    def compute_result(self, *_args: object, **kwargs: object) -> GraspResult:
        self.kwargs_seen.append(dict(kwargs))
        # First call: initial frame. Subsequent calls: refined frame.
        if self.calls == 0:
            self.calls += 1
            return self._initial
        self.calls += 1
        return self._refined


def _matching_perception_pair() -> tuple[PerceptionFrame, PerceptionFrame]:
    """Two frames whose single segmentation is the same target.

    Same shape + same mask placement (with a 1-px shift) so the IoU
    tracker re-identifies the target trivially.
    """

    initial = _frame_with_seg(_seg_at((32, 32), slice(10, 20), slice(10, 20)))
    refined = _frame_with_seg(_seg_at((32, 32), slice(10, 20), slice(11, 21)))
    return initial, refined


class AutonomousGraspServiceRefinementTests(unittest.TestCase):
    def _build_service(
        self,
        *,
        calculator,
        perception,
        policy: Optional[RefinementPolicy] = None,
    ) -> AutonomousGraspService:
        if policy is None:
            policy = RefinementPolicy(
                enabled=True,
                standoff_mm=80.0,
                max_position_correction_mm=50.0,
                max_grip_width_correction_mm=50.0,
                max_orientation_correction_deg=30.0,
                target_match_iou_threshold=0.3,
            )
        return AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=calculator,
            perception=perception,
            mode=GraspMode.CLOSED_LOOP,
            frame_resolver=IdentityFrameResolver(),
            refinement_policy=policy,
            refiner=DefaultPreGraspRefiner(policy=policy),
            verification_policy=GraspVerificationPolicy(enabled=True),
            verifier=NoOpVerifier(),
        )

    def test_refined_grasp_replaces_initial_in_execution(self) -> None:
        initial = _grasp(label="initial", position=(100.0, 50.0, 400.0))
        refined = _grasp(label="refined", position=(102.0, 51.0, 400.0))
        calc = _RefinementScriptedCalculator(
            initial=_result(initial), refined=_result(refined)
        )
        initial_frame, refined_frame = _matching_perception_pair()
        perception = _FakePerception([initial_frame, refined_frame])

        service = self._build_service(calculator=calc, perception=perception)
        arm = service.runtime.orchestrator.arm  # the _TypedFakeArm
        report = service.pick()

        self.assertIs(report.outcome, AutonomousGraspOutcome.SUCCEEDED)
        # Both perception frames were consumed.
        self.assertEqual(perception.calls, 2)
        # Calculator was called at least twice (once per frame).
        self.assertGreaterEqual(calc.calls, 2)
        # At least one motion call was made at the refined grasp's
        # position (the final approach / contact), not the initial.
        approached_refined = any(
            np.allclose(p.position_mm, refined.position, atol=1.0)
            for p in arm.move_calls  # type: ignore[attr-defined]
        )
        approached_initial_only = (
            not approached_refined
            and any(
                np.allclose(p.position_mm, initial.position, atol=1.0)
                for p in arm.move_calls  # type: ignore[attr-defined]
            )
        )
        self.assertTrue(
            approached_refined,
            f"expected motion at refined position {refined.position}; "
            f"got positions {[p.position_mm for p in arm.move_calls]}",  # type: ignore[attr-defined]
        )
        self.assertFalse(approached_initial_only)

    def test_target_lost_returns_typed_outcome(self) -> None:
        # Initial frame has the target; refined frame's mask is a
        # totally disjoint tiny mask the tracker rejects.
        initial_frame = _frame_with_seg(
            _seg_at((32, 32), slice(10, 20), slice(10, 20))
        )
        refined_frame = _frame_with_seg(
            _seg_at((32, 32), slice(0, 2), slice(0, 2))
        )
        calc = _RefinementScriptedCalculator(
            initial=_result(_grasp()), refined=_result(_grasp(label="never_used"))
        )
        perception = _FakePerception([initial_frame, refined_frame])
        service = self._build_service(calculator=calc, perception=perception)
        arm = service.runtime.orchestrator.arm

        report = service.pick()

        self.assertIs(
            report.outcome, AutonomousGraspOutcome.TARGET_LOST_DURING_REFINE
        )
        # Standoff move happened (1 call), but no final pick.
        # _TypedFakeArm.move_calls only records `arm.move(...)` calls;
        # the execution policy would have added many more on success.
        self.assertEqual(len(arm.move_calls), 1)  # type: ignore[attr-defined]
        self.assertEqual(
            arm.move_calls[0].label, "refinement_standoff"  # type: ignore[attr-defined]
        )

    def test_reperceive_false_holds_without_standoff_move(self) -> None:
        # C2: reperceive=False -> the refiner re-validates on the INITIAL frame. No standoff move, no second
        # perception scan (static eye-to-hand camera: a 2nd viewpoint is impossible + the standoff occludes /
        # times out). The full perceive->refine-HOLD->verify loop still runs.
        initial = _grasp(label="held", position=(100.0, 50.0, 400.0))
        calc = _RefinementScriptedCalculator(
            initial=_result(initial), refined=_result(initial)  # same grasp -> delta 0 -> ACCEPTED hold
        )
        initial_frame, _ = _matching_perception_pair()
        perception = _FakePerception([initial_frame])  # ONLY one frame -> proves no re-perceive
        policy = RefinementPolicy(
            enabled=True, reperceive=False, max_position_correction_mm=50.0,
            max_grip_width_correction_mm=50.0, max_orientation_correction_deg=30.0,
            target_match_iou_threshold=0.3,
        )
        service = self._build_service(calculator=calc, perception=perception, policy=policy)
        arm = service.runtime.orchestrator.arm
        report = service.pick()

        self.assertIs(report.outcome, AutonomousGraspOutcome.SUCCEEDED)
        self.assertEqual(perception.calls, 1)  # initial frame only — NO second acquire
        self.assertFalse(  # the hold skips the standoff move entirely
            any(p.label == "refinement_standoff" for p in arm.move_calls),  # type: ignore[attr-defined]
            f"reperceive=False must not drive to a standoff; got {[p.label for p in arm.move_calls]}",  # type: ignore[attr-defined]
        )

    def test_diverged_returns_typed_outcome_no_final_motion(self) -> None:
        initial = _grasp(position=(100.0, 50.0, 400.0))
        refined = _grasp(position=(500.0, 50.0, 400.0))  # 400 mm shift
        calc = _RefinementScriptedCalculator(
            initial=_result(initial), refined=_result(refined)
        )
        initial_frame, refined_frame = _matching_perception_pair()
        perception = _FakePerception([initial_frame, refined_frame])
        # Tight position cap to force divergence.
        tight_policy = RefinementPolicy(
            enabled=True,
            standoff_mm=80.0,
            max_position_correction_mm=10.0,
            target_match_iou_threshold=0.3,
        )
        service = self._build_service(
            calculator=calc, perception=perception, policy=tight_policy
        )
        arm = service.runtime.orchestrator.arm

        report = service.pick()

        self.assertIs(
            report.outcome, AutonomousGraspOutcome.REFINEMENT_DIVERGED
        )
        # Only the standoff move happened; no final approach motion.
        self.assertEqual(len(arm.move_calls), 1)  # type: ignore[attr-defined]

    def test_disabled_policy_falls_back_to_mode_not_available(self) -> None:
        # A refiner is wired but the policy is disabled. The service
        # treats this exactly like "no refiner" --- CLOSED_LOOP gets
        # MODE_NOT_AVAILABLE so the operator must consciously opt in.
        disabled = RefinementPolicy(enabled=False)
        service = AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_result(_grasp())]),  # type: ignore[arg-type]
            perception=_FakePerception([_frame_with_seg(_seg_at((32, 32), slice(10, 20), slice(10, 20)))]),
            mode=GraspMode.CLOSED_LOOP,
            refinement_policy=disabled,
            refiner=DefaultPreGraspRefiner(policy=disabled),
        )

        report = service.pick()

        self.assertIs(report.outcome, AutonomousGraspOutcome.MODE_NOT_AVAILABLE)
        self.assertEqual(
            report.telemetry.get("reason"),
            "mode_requires_refinement_but_no_refiner_wired",
        )


def _intrinsics(shape: tuple[int, int] = (32, 32), f: float = 400.0) -> np.ndarray:
    return np.array(
        [[f, 0.0, shape[1] / 2.0], [0.0, f, shape[0] / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def _cam_to_base(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> Transform:
    # camera_to_base with identity rotation + a base-frame camera position (tx,ty,tz).
    return Transform(
        translation_mm=np.array([tx, ty, tz], dtype=np.float64),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        from_frame=Frame.CAMERA,
        to_frame=Frame.BASE,
    )


class WorldSpacePoseTrackerTests(unittest.TestCase):
    """C1: the viewpoint-invariant tracker matches by 3D base-frame pose where the image-space tracker loses
    the target on a standoff/camera shift (the documented 0/3 CLOSED_LOOP failure)."""

    @staticmethod
    def _frame(segs, depth_val: float = 500.0, shape: tuple[int, int] = (32, 32)) -> PerceptionFrame:
        return PerceptionFrame(
            depth_map=np.full(shape, depth_val, dtype=np.float64),
            intrinsics=_intrinsics(shape),
            segmentations=tuple(segs),
        )

    def test_back_projection_to_camera_frame(self) -> None:
        from src.robot.grasping.closed_loop.target_tracking import _centroid_xyz_base_mm

        # mask symmetric about the principal point (cx,cy)=(16,16); depth 500 -> camera point (0,0,500)
        seg = _seg_at((32, 32), slice(14, 19), slice(14, 19))
        xyz = _centroid_xyz_base_mm(seg.mask, np.full((32, 32), 500.0), _intrinsics(), None)
        assert xyz is not None
        self.assertAlmostEqual(xyz[0], 0.0, places=6)
        self.assertAlmostEqual(xyz[1], 0.0, places=6)
        self.assertAlmostEqual(xyz[2], 500.0, places=6)

    def test_back_projection_applies_camera_to_base(self) -> None:
        from src.robot.grasping.closed_loop.target_tracking import _centroid_xyz_base_mm

        seg = _seg_at((32, 32), slice(14, 19), slice(14, 19))
        xyz = _centroid_xyz_base_mm(seg.mask, np.full((32, 32), 500.0), _intrinsics(), _cam_to_base(tx=100.0))
        assert xyz is not None
        self.assertAlmostEqual(xyz[0], 100.0, places=6)  # camera (0,0,500) + base offset (100,0,0)

    def test_factory_populates_centroid_xyz_when_given_depth(self) -> None:
        seg = _seg_at((32, 32), slice(14, 19), slice(14, 19))
        ident = target_identity_from_segmentation(
            seg, depth_map=np.full((32, 32), 500.0), intrinsics=_intrinsics(), camera_to_base=_cam_to_base()
        )
        self.assertIsNotNone(ident.centroid_xyz_mm)
        # legacy call (no depth/intrinsics) keeps it None -> byte-identical image-space identity
        self.assertIsNone(target_identity_from_segmentation(seg).centroid_xyz_mm)

    def test_world_space_matches_where_image_space_loses_target(self) -> None:
        # THE C1 PROOF: a camera shift moves the target's IMAGE location far enough that mask IoU = 0 (the
        # image-space tracker returns target_lost) — but the 3D base-frame centroid is unchanged, so the
        # world-space tracker still re-identifies it.
        initial_seg = _seg_at((32, 32), slice(14, 19), slice(14, 19))   # centred at pixel (16,16)
        identity = target_identity_from_segmentation(
            initial_seg, depth_map=np.full((32, 32), 500.0), intrinsics=_intrinsics(),
            camera_to_base=_cam_to_base(tx=0.0),  # camera 1 at base origin -> centroid (0,0,500) base
        )
        # camera 2 shifted +10mm in base-x: the SAME base point (0,0,500) now images at pixel col ~8
        refined_seg = _seg_at((32, 32), slice(14, 19), slice(6, 11))    # centred at pixel (8,16)
        refined_frame = self._frame([refined_seg])

        # image-space: masks don't overlap -> IoU 0 -> target lost
        iou = IoUCentroidTargetTracker().match(identity, refined_frame, iou_threshold=0.35)
        self.assertIsNone(iou)

        # world-space: back-project col-8 pixel with camera-2 transform -> base (0,0,500) -> matches
        ws = WorldSpacePoseTracker().match(
            identity, refined_frame, iou_threshold=0.35, camera_to_base=_cam_to_base(tx=10.0)
        )
        self.assertIsNotNone(ws)
        assert ws is not None
        self.assertEqual(ws[0], 0)
        self.assertGreater(ws[1], 0.9)  # near-zero distance -> high confidence

    def test_world_space_picks_closest_of_several(self) -> None:
        identity = target_identity_from_segmentation(
            _seg_at((32, 32), slice(14, 19), slice(14, 19)),
            depth_map=np.full((32, 32), 500.0), intrinsics=_intrinsics(), camera_to_base=_cam_to_base(),
        )
        # two candidates: one near the identity 3D point, one far (different depth -> different z)
        near = _seg_at((32, 32), slice(14, 19), slice(14, 19))
        far = _seg_at((32, 32), slice(0, 5), slice(0, 5))
        frame = PerceptionFrame(
            depth_map=np.full((32, 32), 500.0, dtype=np.float64),
            intrinsics=_intrinsics(), segmentations=(far, near),
        )
        ws = WorldSpacePoseTracker().match(identity, frame, iou_threshold=0.35, camera_to_base=_cam_to_base())
        assert ws is not None
        self.assertEqual(ws[0], 1)  # the near candidate, not the far one

    def test_falls_back_to_iou_without_3d_signal(self) -> None:
        # identity built WITHOUT depth -> no centroid_xyz_mm -> world-space tracker == image-space tracker
        seg = _seg_at((32, 32), slice(14, 19), slice(14, 19))
        identity = target_identity_from_segmentation(seg)
        self.assertIsNone(identity.centroid_xyz_mm)
        frame = self._frame([_seg_at((32, 32), slice(14, 19), slice(14, 19))])
        ws = WorldSpacePoseTracker().match(identity, frame, iou_threshold=0.35)
        iou = IoUCentroidTargetTracker().match(identity, frame, iou_threshold=0.35)
        self.assertEqual(ws, iou)  # identical behaviour when the 3D signal is absent

    def test_rejects_target_beyond_tolerance(self) -> None:
        # a candidate whose 3D pose is > max_pose_distance_mm away AND whose mask doesn't overlap -> no match
        identity = target_identity_from_segmentation(
            _seg_at((32, 32), slice(14, 19), slice(14, 19)),
            depth_map=np.full((32, 32), 500.0), intrinsics=_intrinsics(), camera_to_base=_cam_to_base(),
        )
        far = _seg_at((32, 32), slice(0, 4), slice(0, 4))  # corner, far in 3D, no mask overlap
        frame = self._frame([far])
        ws = WorldSpacePoseTracker(max_pose_distance_mm=5.0).match(
            identity, frame, iou_threshold=0.35, camera_to_base=_cam_to_base()
        )
        self.assertIsNone(ws)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
