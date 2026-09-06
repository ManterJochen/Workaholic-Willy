"""Tests for Phase S4 post-grasp verification.

Coverage layers:

* :class:`GraspVerificationPolicy` validation.
* :class:`NoOpVerifier`.
* :class:`ObjectDetectingGripperVerifier`.
* :class:`WidthDeltaGripperVerifier`.
* :class:`VisionTargetDisplacementVerifier`.
* :class:`CompositeGraspVerifier` (``all_must_pass`` and ``any_pass``).
* :class:`AutonomousGraspService` end-to-end:
  * CLOSED_LOOP without a verifier wired -> ``MODE_NOT_AVAILABLE``
    with verification reason (once a refiner *is* wired).
  * verification PASS -> SUCCEEDED.
  * verification FAIL -> ``VERIFICATION_FAILED`` (no silent success).
  * INCONCLUSIVE + ``fail_closed=True`` -> ``VERIFICATION_FAILED``.
  * INCONCLUSIVE + ``fail_closed=False`` -> SUCCEEDED.
  * EASY / AUTO are unaffected (legacy trust path keeps running).
"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np

from src.geometry import Frame, Pose
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
    CompositeGraspVerifier,
    DefaultPreGraspRefiner,
    GraspVerificationContext,
    GraspVerificationPolicy,
    GraspVerificationReport,
    IdentityFrameResolver,
    NoOpVerifier,
    ObjectDetectingGripperVerifier,
    RefinementPolicy,
    VerificationOutcome,
    VisionTargetDisplacementVerifier,
    WidthDeltaGripperVerifier,
    target_identity_from_segmentation,
)
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.perception import PerceptionFrame
from src.robot.grasping.scoring.success_probability import (
    ShadowSuccessContext,
    UncertaintyRerankConfig,
    load_success_probability_model,
)
from tests._helpers import _FakePerception, _ScriptedCalculator


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


class _PlainGripper:
    """Minimal Gripper Protocol impl without object detection."""

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

    def set_width_mm(self, width_mm, *, speed=None, force=None) -> None:
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


class _RefinementScriptedCalculator:
    def __init__(self, *, initial: GraspResult, refined: GraspResult) -> None:
        self._initial = initial
        self._refined = refined
        self.calls = 0

    def compute_result(self, *_args, **_kwargs) -> GraspResult:
        if self.calls == 0:
            self.calls += 1
            return self._initial
        self.calls += 1
        return self._refined


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _seg_at(shape, row_slice, col_slice) -> SimpleNamespace:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[row_slice, col_slice] = 1
    return SimpleNamespace(mask=mask)


def _frame_with_seg(seg, shape=(32, 32)) -> PerceptionFrame:
    depth = np.full(shape, 500.0, dtype=np.float64)
    intrinsics = np.array(
        [[400.0, 0.0, shape[1] / 2.0], [0.0, 400.0, shape[0] / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return PerceptionFrame(
        depth_map=depth, intrinsics=intrinsics, segmentations=(seg,)
    )


def _empty_frame(shape=(32, 32)) -> PerceptionFrame:
    return PerceptionFrame(
        depth_map=np.zeros(shape, dtype=np.float64),
        intrinsics=np.eye(3, dtype=np.float64),
        segmentations=(),
    )


def _grasp(
    *,
    position=(100.0, 50.0, 400.0),
    approach=(0.0, 0.0, 1.0),
    axis=(1.0, 0.0, 0.0),
    grip_width_mm=40.0,
    score=0.9,
    frame=GraspFrame.BASE,
    label="grasp",
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


def _identity_for(seg) -> "object":
    return target_identity_from_segmentation(seg)


def _matching_perception_pair():
    initial = _frame_with_seg(_seg_at((32, 32), slice(10, 20), slice(10, 20)))
    refined = _frame_with_seg(_seg_at((32, 32), slice(10, 20), slice(11, 21)))
    return initial, refined


# ---------------------------------------------------------------------------
# GraspVerificationPolicy
# ---------------------------------------------------------------------------


class GraspVerificationPolicyTests(unittest.TestCase):
    def test_defaults_are_valid(self) -> None:
        policy = GraspVerificationPolicy()
        self.assertFalse(policy.enabled)
        self.assertTrue(policy.fail_closed)
        self.assertGreaterEqual(policy.width_delta_min_mm, 0.0)

    def test_rejects_negative_width_min(self) -> None:
        with self.assertRaises(ValueError):
            GraspVerificationPolicy(width_delta_min_mm=-1.0)

    def test_rejects_negative_width_max(self) -> None:
        with self.assertRaises(ValueError):
            GraspVerificationPolicy(width_delta_max_mm=-1.0)

    def test_rejects_iou_outside_unit_interval(self) -> None:
        with self.assertRaises(ValueError):
            GraspVerificationPolicy(vision_displacement_iou_max=1.5)
        with self.assertRaises(ValueError):
            GraspVerificationPolicy(vision_displacement_iou_max=-0.1)


# ---------------------------------------------------------------------------
# NoOpVerifier
# ---------------------------------------------------------------------------


class NoOpVerifierTests(unittest.TestCase):
    def test_always_passes(self) -> None:
        ctx = GraspVerificationContext(
            grasp=_grasp(), policy=GraspVerificationPolicy(enabled=True)
        )
        report = NoOpVerifier().verify(ctx)
        self.assertIs(report.outcome, VerificationOutcome.PASSED)


# ---------------------------------------------------------------------------
# ObjectDetectingGripperVerifier
# ---------------------------------------------------------------------------


class ObjectDetectingGripperVerifierTests(unittest.TestCase):
    def test_passes_when_gripper_reports_detected(self) -> None:
        gripper = _ObjectDetectingGripper(detected=True)
        ctx = GraspVerificationContext(
            grasp=_grasp(),
            policy=GraspVerificationPolicy(
                enabled=True, require_object_detected=True
            ),
            gripper=gripper,
        )
        report = ObjectDetectingGripperVerifier().verify(ctx)
        self.assertIs(report.outcome, VerificationOutcome.PASSED)

    def test_fails_when_gripper_reports_empty_and_required(self) -> None:
        gripper = _ObjectDetectingGripper(detected=False)
        ctx = GraspVerificationContext(
            grasp=_grasp(),
            policy=GraspVerificationPolicy(
                enabled=True, require_object_detected=True
            ),
            gripper=gripper,
        )
        report = ObjectDetectingGripperVerifier().verify(ctx)
        self.assertIs(report.outcome, VerificationOutcome.FAILED)

    def test_inconclusive_when_gripper_reports_empty_but_not_required(self) -> None:
        gripper = _ObjectDetectingGripper(detected=False)
        ctx = GraspVerificationContext(
            grasp=_grasp(),
            policy=GraspVerificationPolicy(
                enabled=True, require_object_detected=False
            ),
            gripper=gripper,
        )
        report = ObjectDetectingGripperVerifier().verify(ctx)
        self.assertIs(report.outcome, VerificationOutcome.INCONCLUSIVE)

    def test_inconclusive_when_gripper_lacks_capability(self) -> None:
        gripper = _PlainGripper()
        ctx = GraspVerificationContext(
            grasp=_grasp(),
            policy=GraspVerificationPolicy(
                enabled=True, require_object_detected=True
            ),
            gripper=gripper,  # type: ignore[arg-type]
        )
        report = ObjectDetectingGripperVerifier().verify(ctx)
        self.assertIs(report.outcome, VerificationOutcome.INCONCLUSIVE)

    def test_inconclusive_when_no_gripper(self) -> None:
        ctx = GraspVerificationContext(
            grasp=_grasp(),
            policy=GraspVerificationPolicy(enabled=True),
        )
        report = ObjectDetectingGripperVerifier().verify(ctx)
        self.assertIs(report.outcome, VerificationOutcome.INCONCLUSIVE)


# ---------------------------------------------------------------------------
# WidthDeltaGripperVerifier
# ---------------------------------------------------------------------------


class WidthDeltaGripperVerifierTests(unittest.TestCase):
    def test_passes_when_post_close_width_above_min_threshold(self) -> None:
        gripper = _PlainGripper(min_width_mm=0.0)
        ctx = GraspVerificationContext(
            grasp=_grasp(grip_width_mm=40.0),
            policy=GraspVerificationPolicy(
                enabled=True, width_delta_min_mm=2.0
            ),
            gripper=gripper,  # type: ignore[arg-type]
            pre_close_width_mm=80.0,
            post_close_width_mm=39.0,
            commanded_close_width_mm=39.0,
        )
        report = WidthDeltaGripperVerifier().verify(ctx)
        self.assertIs(report.outcome, VerificationOutcome.PASSED)

    def test_fails_when_jaws_collapsed_to_minimum(self) -> None:
        gripper = _PlainGripper(min_width_mm=0.0)
        ctx = GraspVerificationContext(
            grasp=_grasp(),
            policy=GraspVerificationPolicy(
                enabled=True, width_delta_min_mm=2.0
            ),
            gripper=gripper,  # type: ignore[arg-type]
            post_close_width_mm=1.0,  # below 0 + 2
        )
        report = WidthDeltaGripperVerifier().verify(ctx)
        self.assertIs(report.outcome, VerificationOutcome.FAILED)
        self.assertEqual(report.reason, "jaws_collapsed_to_minimum")

    def test_fails_when_jaws_did_not_close_enough(self) -> None:
        gripper = _PlainGripper(min_width_mm=0.0)
        ctx = GraspVerificationContext(
            grasp=_grasp(grip_width_mm=40.0),
            policy=GraspVerificationPolicy(
                enabled=True, width_delta_min_mm=2.0, width_delta_max_mm=3.0
            ),
            gripper=gripper,  # type: ignore[arg-type]
            post_close_width_mm=70.0,  # 70 > 39 + 3
            commanded_close_width_mm=39.0,
        )
        report = WidthDeltaGripperVerifier().verify(ctx)
        self.assertIs(report.outcome, VerificationOutcome.FAILED)
        self.assertEqual(report.reason, "jaws_did_not_close_enough")

    def test_inconclusive_without_post_close_sample(self) -> None:
        gripper = _PlainGripper()
        ctx = GraspVerificationContext(
            grasp=_grasp(),
            policy=GraspVerificationPolicy(enabled=True),
            gripper=gripper,  # type: ignore[arg-type]
            post_close_width_mm=None,
        )
        report = WidthDeltaGripperVerifier().verify(ctx)
        self.assertIs(report.outcome, VerificationOutcome.INCONCLUSIVE)

    def test_inconclusive_without_gripper(self) -> None:
        ctx = GraspVerificationContext(
            grasp=_grasp(),
            policy=GraspVerificationPolicy(enabled=True),
            post_close_width_mm=50.0,
        )
        report = WidthDeltaGripperVerifier().verify(ctx)
        self.assertIs(report.outcome, VerificationOutcome.INCONCLUSIVE)


# ---------------------------------------------------------------------------
# VisionTargetDisplacementVerifier
# ---------------------------------------------------------------------------


class VisionTargetDisplacementVerifierTests(unittest.TestCase):
    def test_passes_when_target_no_longer_visible(self) -> None:
        identity_seg = _seg_at((32, 32), slice(10, 20), slice(10, 20))
        # Post-lift frame has only a small disjoint mask elsewhere.
        post_frame = _frame_with_seg(
            _seg_at((32, 32), slice(0, 3), slice(0, 3))
        )
        ctx = GraspVerificationContext(
            grasp=_grasp(),
            policy=GraspVerificationPolicy(
                enabled=True, vision_displacement_iou_max=0.3
            ),
            target_identity=_identity_for(identity_seg),
            post_lift_frame=post_frame,
        )
        report = VisionTargetDisplacementVerifier().verify(ctx)
        self.assertIs(report.outcome, VerificationOutcome.PASSED)

    def test_passes_when_frame_has_no_segmentations(self) -> None:
        identity_seg = _seg_at((32, 32), slice(10, 20), slice(10, 20))
        ctx = GraspVerificationContext(
            grasp=_grasp(),
            policy=GraspVerificationPolicy(enabled=True),
            target_identity=_identity_for(identity_seg),
            post_lift_frame=_empty_frame(),
        )
        report = VisionTargetDisplacementVerifier().verify(ctx)
        self.assertIs(report.outcome, VerificationOutcome.PASSED)

    def test_fails_when_target_still_visible(self) -> None:
        identity_seg = _seg_at((32, 32), slice(10, 20), slice(10, 20))
        # Post-lift frame still has essentially the same mask.
        post_frame = _frame_with_seg(
            _seg_at((32, 32), slice(10, 20), slice(10, 20))
        )
        ctx = GraspVerificationContext(
            grasp=_grasp(),
            policy=GraspVerificationPolicy(
                enabled=True, vision_displacement_iou_max=0.3
            ),
            target_identity=_identity_for(identity_seg),
            post_lift_frame=post_frame,
        )
        report = VisionTargetDisplacementVerifier().verify(ctx)
        self.assertIs(report.outcome, VerificationOutcome.FAILED)

    def test_inconclusive_without_identity_or_frame(self) -> None:
        ctx = GraspVerificationContext(
            grasp=_grasp(),
            policy=GraspVerificationPolicy(enabled=True),
        )
        report = VisionTargetDisplacementVerifier().verify(ctx)
        self.assertIs(report.outcome, VerificationOutcome.INCONCLUSIVE)


# ---------------------------------------------------------------------------
# CompositeGraspVerifier
# ---------------------------------------------------------------------------


class _StubVerifier:
    """Verifier that returns a pre-baked outcome."""

    def __init__(self, outcome: VerificationOutcome, reason: str = "stub") -> None:
        self._outcome = outcome
        self._reason = reason

    def verify(self, _ctx) -> GraspVerificationReport:
        return GraspVerificationReport(
            outcome=self._outcome, reason=self._reason
        )


class CompositeGraspVerifierTests(unittest.TestCase):
    def _ctx(self) -> GraspVerificationContext:
        return GraspVerificationContext(
            grasp=_grasp(), policy=GraspVerificationPolicy(enabled=True)
        )

    def test_rejects_empty_verifier_list(self) -> None:
        with self.assertRaises(ValueError):
            CompositeGraspVerifier(verifiers=())

    def test_rejects_unknown_rule(self) -> None:
        with self.assertRaises(ValueError):
            CompositeGraspVerifier(
                verifiers=(NoOpVerifier(),), rule="majority_vote"
            )

    def test_all_must_pass_passes_when_all_pass(self) -> None:
        composite = CompositeGraspVerifier(
            verifiers=(
                _StubVerifier(VerificationOutcome.PASSED),
                _StubVerifier(VerificationOutcome.PASSED),
            ),
        )
        report = composite.verify(self._ctx())
        self.assertIs(report.outcome, VerificationOutcome.PASSED)

    def test_all_must_pass_fails_on_first_failure(self) -> None:
        composite = CompositeGraspVerifier(
            verifiers=(
                _StubVerifier(VerificationOutcome.PASSED),
                _StubVerifier(VerificationOutcome.FAILED, reason="empty"),
            ),
        )
        report = composite.verify(self._ctx())
        self.assertIs(report.outcome, VerificationOutcome.FAILED)
        self.assertIn("empty", report.reason)

    def test_all_must_pass_treats_inconclusive_as_pass_by_default(self) -> None:
        composite = CompositeGraspVerifier(
            verifiers=(
                _StubVerifier(VerificationOutcome.PASSED),
                _StubVerifier(VerificationOutcome.INCONCLUSIVE),
            ),
        )
        report = composite.verify(self._ctx())
        self.assertIs(report.outcome, VerificationOutcome.PASSED)

    def test_all_must_pass_fails_inconclusive_when_required(self) -> None:
        composite = CompositeGraspVerifier(
            verifiers=(
                _StubVerifier(VerificationOutcome.INCONCLUSIVE, reason="no_data"),
            ),
            require_all_conclusive=True,
        )
        report = composite.verify(self._ctx())
        self.assertIs(report.outcome, VerificationOutcome.FAILED)

    def test_any_pass_passes_when_one_passes(self) -> None:
        composite = CompositeGraspVerifier(
            verifiers=(
                _StubVerifier(VerificationOutcome.FAILED),
                _StubVerifier(VerificationOutcome.PASSED),
            ),
            rule="any_pass",
        )
        report = composite.verify(self._ctx())
        self.assertIs(report.outcome, VerificationOutcome.PASSED)

    def test_any_pass_fails_when_none_pass(self) -> None:
        composite = CompositeGraspVerifier(
            verifiers=(
                _StubVerifier(VerificationOutcome.FAILED),
                _StubVerifier(VerificationOutcome.FAILED),
            ),
            rule="any_pass",
        )
        report = composite.verify(self._ctx())
        self.assertIs(report.outcome, VerificationOutcome.FAILED)

    def test_any_pass_inconclusive_when_all_inconclusive(self) -> None:
        composite = CompositeGraspVerifier(
            verifiers=(
                _StubVerifier(VerificationOutcome.INCONCLUSIVE),
                _StubVerifier(VerificationOutcome.INCONCLUSIVE),
            ),
            rule="any_pass",
        )
        report = composite.verify(self._ctx())
        self.assertIs(report.outcome, VerificationOutcome.INCONCLUSIVE)


# ---------------------------------------------------------------------------
# AutonomousGraspService end-to-end
# ---------------------------------------------------------------------------


class AutonomousGraspServiceVerificationTests(unittest.TestCase):
    def _refinement_policy(self) -> RefinementPolicy:
        return RefinementPolicy(
            enabled=True,
            standoff_mm=80.0,
            max_position_correction_mm=50.0,
            max_grip_width_correction_mm=50.0,
            max_orientation_correction_deg=30.0,
            target_match_iou_threshold=0.3,
        )

    def _build_service(
        self,
        *,
        verifier=None,
        verification_policy: Optional[GraspVerificationPolicy] = None,
        post_lift_seg=None,
    ) -> AutonomousGraspService:
        initial = _grasp(label="initial")
        refined = _grasp(label="refined")
        calc = _RefinementScriptedCalculator(
            initial=_result(initial), refined=_result(refined)
        )
        initial_frame, refined_frame = _matching_perception_pair()
        frames = [initial_frame, refined_frame]
        if post_lift_seg is not None:
            frames.append(_frame_with_seg(post_lift_seg))
        elif (
            verification_policy is not None
            and verification_policy.post_lift_vision_check
        ):
            # The verifier expects a third frame --- give it an empty
            # one so VisionTargetDisplacementVerifier reports PASSED
            # unless the test wires something specific.
            frames.append(_empty_frame())
        perception = _FakePerception(frames)
        ref_policy = self._refinement_policy()
        return AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
            mode=GraspMode.CLOSED_LOOP,
            frame_resolver=IdentityFrameResolver(),
            refinement_policy=ref_policy,
            refiner=DefaultPreGraspRefiner(policy=ref_policy),
            verification_policy=verification_policy,
            verifier=verifier,
        )

    def test_closed_loop_refuses_when_verifier_missing_but_refiner_wired(self) -> None:
        # Refiner is wired; verifier is not. The service must surface
        # the S4 verification gate honestly, not just defer to S3.
        ref_policy = self._refinement_policy()
        service = AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_result(_grasp())]),  # type: ignore[arg-type]
            perception=_FakePerception(
                [_frame_with_seg(_seg_at((32, 32), slice(10, 20), slice(10, 20)))]
            ),
            mode=GraspMode.CLOSED_LOOP,
            frame_resolver=IdentityFrameResolver(),
            refinement_policy=ref_policy,
            refiner=DefaultPreGraspRefiner(policy=ref_policy),
        )
        report = service.pick()
        self.assertIs(report.outcome, AutonomousGraspOutcome.MODE_NOT_AVAILABLE)
        self.assertEqual(
            report.telemetry.get("reason"),
            "mode_requires_verification_but_no_verifier_wired",
        )

    def test_closed_loop_refuses_when_verifier_policy_disabled(self) -> None:
        ref_policy = self._refinement_policy()
        service = AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_result(_grasp())]),  # type: ignore[arg-type]
            perception=_FakePerception(
                [_frame_with_seg(_seg_at((32, 32), slice(10, 20), slice(10, 20)))]
            ),
            mode=GraspMode.CLOSED_LOOP,
            frame_resolver=IdentityFrameResolver(),
            refinement_policy=ref_policy,
            refiner=DefaultPreGraspRefiner(policy=ref_policy),
            verification_policy=GraspVerificationPolicy(enabled=False),
            verifier=NoOpVerifier(),
        )
        report = service.pick()
        self.assertIs(report.outcome, AutonomousGraspOutcome.MODE_NOT_AVAILABLE)
        self.assertEqual(
            report.telemetry.get("reason"),
            "mode_requires_verification_but_no_verifier_wired",
        )

    def test_passing_verifier_yields_succeeded(self) -> None:
        service = self._build_service(
            verifier=NoOpVerifier(),
            verification_policy=GraspVerificationPolicy(enabled=True),
        )
        report = service.pick()
        self.assertIs(report.outcome, AutonomousGraspOutcome.SUCCEEDED)
        self.assertEqual(
            report.telemetry.get("verification_outcome"),
            str(VerificationOutcome.PASSED),
        )

    def test_failing_verifier_yields_verification_failed(self) -> None:
        service = self._build_service(
            verifier=_StubVerifier(
                VerificationOutcome.FAILED, reason="empty_jaws"
            ),
            verification_policy=GraspVerificationPolicy(enabled=True),
        )
        report = service.pick()
        self.assertIs(report.outcome, AutonomousGraspOutcome.VERIFICATION_FAILED)
        self.assertEqual(
            report.telemetry.get("verification_outcome"),
            str(VerificationOutcome.FAILED),
        )

    def test_inconclusive_fail_closed_yields_verification_failed(self) -> None:
        service = self._build_service(
            verifier=_StubVerifier(VerificationOutcome.INCONCLUSIVE),
            verification_policy=GraspVerificationPolicy(
                enabled=True, fail_closed=True
            ),
        )
        report = service.pick()
        self.assertIs(report.outcome, AutonomousGraspOutcome.VERIFICATION_FAILED)

    def test_inconclusive_fail_open_yields_succeeded(self) -> None:
        service = self._build_service(
            verifier=_StubVerifier(VerificationOutcome.INCONCLUSIVE),
            verification_policy=GraspVerificationPolicy(
                enabled=True, fail_closed=False
            ),
        )
        report = service.pick()
        self.assertIs(report.outcome, AutonomousGraspOutcome.SUCCEEDED)

    def test_post_lift_vision_check_acquires_third_frame(self) -> None:
        # Vision verifier sees an empty post-lift frame -> PASSED.
        service = self._build_service(
            verifier=VisionTargetDisplacementVerifier(),
            verification_policy=GraspVerificationPolicy(
                enabled=True, post_lift_vision_check=True
            ),
        )
        report = service.pick()
        self.assertIs(report.outcome, AutonomousGraspOutcome.SUCCEEDED)
        # Three perception frames consumed: initial + refined + post-lift.
        perception = service.runtime.orchestrator.perception
        self.assertEqual(perception.calls, 3)  # type: ignore[attr-defined]

    def test_post_lift_vision_check_target_still_visible_fails(self) -> None:
        # Post-lift frame still contains the target's mask -> FAILED.
        service = self._build_service(
            verifier=VisionTargetDisplacementVerifier(),
            verification_policy=GraspVerificationPolicy(
                enabled=True,
                post_lift_vision_check=True,
                vision_displacement_iou_max=0.3,
            ),
            post_lift_seg=_seg_at((32, 32), slice(10, 20), slice(10, 20)),
        )
        report = service.pick()
        self.assertIs(report.outcome, AutonomousGraspOutcome.VERIFICATION_FAILED)

    def test_easy_mode_is_not_affected_by_verification_gate(self) -> None:
        # EASY does not require verification --- a service with no
        # verifier at all must still succeed at EASY picks.
        service = AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_result(_grasp())]),  # type: ignore[arg-type]
            perception=_FakePerception(
                [_frame_with_seg(_seg_at((32, 32), slice(10, 20), slice(10, 20)))]
            ),
            mode=GraspMode.EASY,
        )
        report = service.pick()
        # EASY runs through the legacy runtime path; the underlying
        # FakePerception only has one frame so the orchestrator will
        # actually attempt a pick. The exact outcome depends on the
        # orchestrator's wiring (no gripper, no policy) --- we only
        # assert that the verification gate did NOT refuse.
        self.assertIsNot(report.outcome, AutonomousGraspOutcome.MODE_NOT_AVAILABLE)


# ---------------------------------------------------------------------------
# Wire the stub verifier helper used above
# ---------------------------------------------------------------------------


# ``_StubVerifier`` is defined inside CompositeGraspVerifierTests' scope
# but the service tests above also use it; rebind at module scope so
# both reach it (Python's class-body scoping rules would otherwise hide
# the definition from references in later classes).
_StubVerifier = _StubVerifier  # type: ignore[has-type]


class RefinePathUncertaintyRerankTests(unittest.TestCase):
    """G4 Part 2b-2: the REFINE path applies U2+U4+G4, so the per-candidate uncertainty re-rank flips the
    chosen INITIAL grasp (the one that gets refined) AND surfaces UncertaintyRerankTelemetry on the refine
    AutonomousGraspReport. Uses DENSE_AUTONOMOUS (its refine path is real + dense, so G4's dense gate
    fires) + the verification harness (DefaultPreGraspRefiner + NoOpVerifier)."""

    _ARTIFACT = Path(__file__).resolve().parents[1] / "assets/models/success_probability/v1"

    def _ctx(self) -> ShadowSuccessContext:
        return ShadowSuccessContext(
            model=load_success_probability_model(self._ARTIFACT),
            version_label="v1",
            lifecycle_phase="canary",
            mode_label="dense_autonomous",
        )

    def _initial_two(self) -> GraspResult:
        # Geometry ranks A (0.85) > B (0.70), but A's approach corridor is blocked: adjusted_A = 0.85 -
        # 0.5*0.90 = 0.40 < adjusted_B = 0.70 - 0.5*0.10 = 0.65 -> B becomes the grasp that gets refined.
        a = _grasp(label="geo_leader", score=0.85)
        b = _grasp(label="unc_safe", score=0.70)
        a.metadata["corridor_blockage_confidence"] = 0.90
        b.metadata["corridor_blockage_confidence"] = 0.10
        return GraspResult(candidates=(a, b), reasons=(), top_score=0.85)

    def _build(self, *, rerank_cfg: Optional[UncertaintyRerankConfig]) -> AutonomousGraspService:
        calc = _RefinementScriptedCalculator(
            initial=self._initial_two(), refined=_result(_grasp(label="refined"))
        )
        initial_frame, refined_frame = _matching_perception_pair()
        perception = _FakePerception([initial_frame, refined_frame])
        ref_policy = RefinementPolicy(enabled=True, target_match_iou_threshold=0.3)
        service = AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
            mode=GraspMode.DENSE_AUTONOMOUS,
            frame_resolver=IdentityFrameResolver(),
            refinement_policy=ref_policy,
            refiner=DefaultPreGraspRefiner(policy=ref_policy),
            verification_policy=GraspVerificationPolicy(enabled=True),
            verifier=NoOpVerifier(),
        )
        orch = service.runtime.orchestrator
        orch.shadow_success_context = self._ctx()
        orch.uncertainty_rerank_config = rerank_cfg
        return service

    def test_refine_path_reranks_so_runner_up_is_chosen(self) -> None:
        service = self._build(rerank_cfg=UncertaintyRerankConfig(enabled=True, weight=0.5))
        report = service.pick()
        tel = report.uncertainty_rerank_telemetry
        self.assertIsNotNone(tel)
        assert tel is not None
        self.assertTrue(tel.applied)
        self.assertTrue(tel.top1_changed)
        # B (the runner-up) is the grasp the refine path selected to refine -- a real decision change.
        self.assertEqual(tel.winner_geometric_score, 0.70)
        self.assertEqual(tel.mode_label, "dense_autonomous")

    def test_refine_path_disabled_is_byte_identical(self) -> None:
        service = self._build(rerank_cfg=None)
        report = service.pick()
        self.assertIsNone(report.uncertainty_rerank_telemetry)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
