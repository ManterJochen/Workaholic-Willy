"""Two-scan pre-grasp refinement.

This module ships the typed contract and a default implementation for
the locked "closed-loop" workflow:

    1. Acquire an initial perception frame.
    2. Compute a best grasp via :class:`GraspCalculator`.
    3. Drive the arm to a pre-grasp standoff above the chosen point.
    4. Acquire a refined perception frame at the new viewpoint.
    5. Re-identify the same target via a :class:`TargetTracker`.
    6. Recompute the grasp on the refined frame.
    7. Accept the correction only when it is bounded; otherwise
       refuse to execute and surface a typed failure.

Scope and non-goals
-------------------

* Motion is not owned by this module. The refiner is a pure
  perception/geometry computation: it takes two
  :class:`PerceptionFrame` snapshots and the calculator, and returns a
  :class:`RefinementReport`. The robot-motion side of the two-scan
  workflow lives in :class:`AutonomousGraspService`, the single place
  where motion intent is choreographed and gated by
  :class:`SafetyPreflight`.
* Verification, recovery, and learned target matchers are absent
  here. This module exposes Protocol seams so those phases slot in
  without touching the contract.
* The locked frame contract applies: when a :class:`Transform`
  ``camera_to_base`` is supplied, the refined grasp comes back in
  :attr:`GraspFrame.BASE`, because the calculator honours the kwarg.
  The refiner does not invent transforms; it forwards whatever the
  caller provides.

Public surface
--------------

* :class:`RefinementOutcome`: typed terminal status.
* :class:`RefinementPolicy`: frozen configuration dataclass.
* :class:`TargetIdentity`: frozen record describing the target
  the initial grasp was computed on.
* :class:`RefinementReport`: frozen aggregate result.
* :class:`TargetTracker`: Protocol matching a target across
  frames.
* :class:`PreGraspRefiner`: Protocol for the refiner itself.
* :class:`IoUCentroidTargetTracker`: default
  :class:`TargetTracker` implementation (mask IoU + centroid
  proximity, no learning).
* :class:`WorldSpacePoseTracker`: viewpoint-invariant
  :class:`TargetTracker` implementation (3D base-frame pose match).
* :class:`DefaultPreGraspRefiner`: default
  :class:`PreGraspRefiner` implementation.
* :func:`target_identity_from_segmentation`: factory that builds a
  :class:`TargetIdentity` from a segmentation + the grasp that was
  chosen on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple, Optional, Protocol, runtime_checkable

import numpy as np

from src.geometry import Transform
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.generation.calculator import (
    GraspCalculator,
)
from src.robot.grasping.types.grasp_point import GraspPoint
from src.robot.grasping.closed_loop.target_tracking import (  # noqa: F401 (re-exported here for backward compatibility)
    IoUCentroidTargetTracker,
    TargetIdentity,
    TargetTracker,
    WorldSpacePoseTracker,
    target_identity_from_segmentation,
)

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.grasping.types.perception import PerceptionFrame


__all__ = [
    "DefaultPreGraspRefiner",
    "IoUCentroidTargetTracker",
    "PreGraspRefiner",
    "RefinementOutcome",
    "RefinementPolicy",
    "RefinementReport",
    "TargetIdentity",
    "TargetTracker",
    "WorldSpacePoseTracker",
    "target_identity_from_segmentation",
]


@dataclass(frozen=True, slots=True)
class RefinementPolicy:
    """Bounded operator configuration for two-scan refinement.

    All fields have defaults sized for typical eye-in-hand bin-picking
    geometry. They are conservative: the goal is to refuse implausible
    corrections, not to chase the latest noisy frame at any cost.

    Attributes
    ----------
    enabled
        Master switch. When :data:`False` the refiner returns a typed
        :attr:`RefinementOutcome.SKIPPED` and the caller falls back to
        the initial grasp. Lets a service wire the refiner in without
        forcing every pick through it.
    standoff_mm
        Distance the arm should retreat above the initial grasp before
        the refined frame is acquired. Only the
        :class:`AutonomousGraspService` consumes this; the refiner
        itself does not move the robot.
    max_position_correction_mm
        Hard cap on ``||refined.position - initial.position||``. Above
        this, the refined grasp is rejected as ``DIVERGED``.
    max_orientation_correction_deg
        Hard cap on the angle between the initial and refined
        ``approach`` vectors and on the angle between the initial and
        refined ``axis`` vectors. The larger of the two is compared.
    max_grip_width_correction_mm
        Hard cap on ``|refined.grip_width_mm - initial.grip_width_mm|``.
    target_match_iou_threshold
        Minimum mask IoU the tracker must report for the refined
        segmentation to be accepted as the same target. Below this
        the refiner emits ``TARGET_LOST``.
    require_same_grasp_frame
        When :data:`True` (default) the refined grasp must report the
        same :class:`GraspFrame` as the initial grasp. Comparing
        positions across different frames is meaningless: if the
        calculator returns the refined candidate in a different frame
        the refiner refuses with ``DIVERGED``.
    """

    enabled: bool = False
    standoff_mm: float = 80.0
    max_position_correction_mm: float = 20.0
    max_orientation_correction_deg: float = 15.0
    max_grip_width_correction_mm: float = 20.0
    target_match_iou_threshold: float = 0.35
    require_same_grasp_frame: bool = True
    # Selects the viewpoint-invariant WorldSpacePoseTracker (3D base-frame pose match) over the default
    # image-space tracker. RefinementPolicy only carries the flag: whoever builds the refiner must honour
    # it by passing ``tracker=WorldSpacePoseTracker(...)``. The ``from_robot_config`` path builds
    # ``DefaultPreGraspRefiner(policy=...)`` with no tracker, so there the flag is inert and the
    # image-space tracker is used.
    use_world_space_tracker: bool = False
    # Pose distance (mm) a caller hands to ``WorldSpacePoseTracker(max_pose_distance_mm=...)`` when it builds
    # that tracker. No code here reads it; the tracker enforces its own ``max_pose_distance_mm`` (default
    # 50.0), so this field records the intent and is validated as positive.
    world_space_pose_distance_mm_threshold: float = 50.0
    # When ``False`` the refiner runs as a hold on the initial frame: no standoff move, no second
    # perception scan. For a static eye-to-hand camera (the dense overhead) a second viewpoint is impossible,
    # so the two-scan standoff is a no-op and harmful (arm occludes the camera / the move times out). Holding
    # keeps the full perceive/refine/verify/recover loop without degrading the grasp. ``True`` = two-scan.
    reperceive: bool = True

    def __post_init__(self) -> None:
        # Catch operator mistakes at construction; downstream code
        # trusts these values without re-checking them.
        if self.standoff_mm < 0.0:
            raise ValueError(
                f"standoff_mm must be non-negative; got {self.standoff_mm}"
            )
        if self.max_position_correction_mm < 0.0:
            raise ValueError(
                "max_position_correction_mm must be non-negative; "
                f"got {self.max_position_correction_mm}"
            )
        if self.max_orientation_correction_deg < 0.0:
            raise ValueError(
                "max_orientation_correction_deg must be non-negative; "
                f"got {self.max_orientation_correction_deg}"
            )
        if self.max_grip_width_correction_mm < 0.0:
            raise ValueError(
                "max_grip_width_correction_mm must be non-negative; "
                f"got {self.max_grip_width_correction_mm}"
            )
        if not 0.0 <= self.target_match_iou_threshold <= 1.0:
            raise ValueError(
                "target_match_iou_threshold must be in [0, 1]; "
                f"got {self.target_match_iou_threshold}"
            )
        if self.world_space_pose_distance_mm_threshold <= 0.0:
            raise ValueError(
                "world_space_pose_distance_mm_threshold must be positive; "
                f"got {self.world_space_pose_distance_mm_threshold}"
            )


class RefinementOutcome(StrEnum):
    """Terminal status of a single :meth:`PreGraspRefiner.refine` call."""

    # Refined grasp is within all configured bounds and replaces the
    # initial grasp. The caller should execute :attr:`refined_grasp`.
    ACCEPTED = "accepted"
    # The :class:`TargetTracker` could not re-identify the target in
    # the refined frame.
    TARGET_LOST = "target_lost"
    # Refined grasp exceeds at least one of the configured bounds.
    # The refiner refuses to execute the correction.
    DIVERGED = "diverged"
    # Target matched but the recomputed :class:`GraspResult` produced
    # no candidates (e.g. all IK-rejected, all collided). The caller
    # should treat this like a normal "no grasp" outcome.
    NO_GRASP = "no_grasp"
    # Refinement did not run because the policy is disabled. The
    # caller should fall back to the initial grasp.
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class RefinementReport:
    """Typed aggregate result of a refinement attempt.

    Always present:

    * :attr:`outcome`: one of :class:`RefinementOutcome`.
    * :attr:`initial_grasp`: the grasp the refiner started from.

    Conditionally present:

    * :attr:`refined_grasp`: the refined candidate, set iff
      :attr:`outcome` is :attr:`RefinementOutcome.ACCEPTED`.
    * :attr:`refined_result`: the full :class:`GraspResult` from the
      recompute, set whenever the calculator was actually called.
    * :attr:`matched_segmentation_index`: index into
      ``refined_frame.segmentations`` of the matched target, set
      whenever the tracker returned a match.
    * :attr:`match_iou`: the IoU score the tracker reported for the
      match, set alongside :attr:`matched_segmentation_index`.
    * :attr:`position_delta_mm`, :attr:`orientation_delta_deg`,
      :attr:`grip_width_delta_mm`: measured deltas, populated
      whenever a refined candidate exists (regardless of
      accepted/rejected).
    * :attr:`failure_reason`: the matching :class:`GraspFailureReason`
      for ``TARGET_LOST`` / ``DIVERGED`` so callers can fold it into
      higher-level reason lists without re-deriving it.
    """

    outcome: RefinementOutcome
    initial_grasp: GraspPoint
    refined_grasp: Optional[GraspPoint] = None
    refined_result: Optional[GraspResult] = None
    matched_segmentation_index: Optional[int] = None
    match_iou: Optional[float] = None
    position_delta_mm: Optional[float] = None
    orientation_delta_deg: Optional[float] = None
    grip_width_delta_mm: Optional[float] = None
    failure_reason: Optional[GraspFailureReason] = None
    telemetry: dict = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.outcome is RefinementOutcome.ACCEPTED


@runtime_checkable
class PreGraspRefiner(Protocol):
    """Compute and validate a refined grasp on a second perception frame.

    The contract is narrow: the refiner receives both frames already
    acquired plus the calculator and the camera-to-base transform; it
    does not move the robot, request perception, or plan motion.
    """

    def refine(
        self,
        *,
        initial_grasp: GraspPoint,
        initial_frame: "PerceptionFrame",
        refined_frame: "PerceptionFrame",
        target_identity: TargetIdentity,
        calculator: GraspCalculator,
        camera_to_base: Optional[Transform] = None,
    ) -> RefinementReport: ...


def _angle_between_unit_vectors_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two unit vectors.

    The dot product is clamped to ``[-1, 1]`` so numerical drift never
    leaves the domain of :func:`math.acos`.
    """

    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(dot))


class CorrectionDeltas(NamedTuple):
    """The three bounded-correction magnitudes of a refinement.

    Measured between the initial grasp and the refined grasp. Named
    fields make a field swap a name or mypy error instead of a silent
    bug.
    """

    position_mm: float
    orientation_deg: float
    width_mm: float


def _measure_correction_deltas(
    initial_grasp: GraspPoint, refined_best: GraspPoint
) -> CorrectionDeltas:
    """Position, orientation and grip-width correction magnitudes: the bounded-correction inputs."""

    position_mm = float(
        np.linalg.norm(
            np.asarray(refined_best.position, dtype=np.float64)
            - np.asarray(initial_grasp.position, dtype=np.float64)
        )
    )
    approach_angle = _angle_between_unit_vectors_deg(
        np.asarray(initial_grasp.approach, dtype=np.float64),
        np.asarray(refined_best.approach, dtype=np.float64),
    )
    axis_angle = _angle_between_unit_vectors_deg(
        np.asarray(initial_grasp.axis, dtype=np.float64),
        np.asarray(refined_best.axis, dtype=np.float64),
    )
    orientation_deg = max(approach_angle, axis_angle)
    width_mm = abs(
        float(refined_best.grip_width_mm) - float(initial_grasp.grip_width_mm)
    )
    return CorrectionDeltas(position_mm, orientation_deg, width_mm)


@dataclass(frozen=True, slots=True)
class DefaultPreGraspRefiner:
    """Default :class:`PreGraspRefiner` implementation.

    Workflow:

    1. Call :attr:`tracker` to find the refined-frame segmentation
       matching :class:`TargetIdentity`.
    2. If no match clears :attr:`policy.target_match_iou_threshold`,
       return :attr:`RefinementOutcome.TARGET_LOST`.
    3. Recompute the grasp on the matched segmentation via
       :meth:`GraspCalculator.compute_result`, forwarding
       ``camera_to_base`` so the refined candidate comes back in the
       same frame as the initial candidate (assuming the same kwarg
       was used for the initial computation).
    4. If the recompute returns no candidates, return
       :attr:`RefinementOutcome.NO_GRASP`.
    5. Validate the refined best candidate against the configured
       bounds. Any breach yields :attr:`RefinementOutcome.DIVERGED`.
    6. Otherwise the outcome is :attr:`RefinementOutcome.ACCEPTED`.

    The refiner consumes the whole refined :class:`GraspResult`
    (telemetry, reasons, etc.) and surfaces it via
    :attr:`RefinementReport.refined_result` so callers can fold it
    into their own diagnostics.
    """

    policy: RefinementPolicy
    tracker: TargetTracker = field(default_factory=IoUCentroidTargetTracker)

    def refine(
        self,
        *,
        initial_grasp: GraspPoint,
        initial_frame: "PerceptionFrame",
        refined_frame: "PerceptionFrame",
        target_identity: TargetIdentity,
        calculator: GraspCalculator,
        camera_to_base: Optional[Transform] = None,
    ) -> RefinementReport:
        # initial_frame is unused here but stays in the signature so
        # an implementation that needs it (a flow-based tracker) can
        # consult it without an API break.
        del initial_frame

        if not self.policy.enabled:
            return RefinementReport(
                outcome=RefinementOutcome.SKIPPED,
                initial_grasp=initial_grasp,
                telemetry={"reason": "policy_disabled"},
            )

        match = self.tracker.match(
            target_identity,
            refined_frame,
            iou_threshold=self.policy.target_match_iou_threshold,
            camera_to_base=camera_to_base,  # lets the WorldSpacePoseTracker match in the base frame
        )
        if match is None:
            return RefinementReport(
                outcome=RefinementOutcome.TARGET_LOST,
                initial_grasp=initial_grasp,
                failure_reason=GraspFailureReason.TARGET_LOST_DURING_REFINE,
                telemetry={
                    "iou_threshold": self.policy.target_match_iou_threshold,
                    "n_candidate_segmentations": len(refined_frame.segmentations),
                },
            )

        matched_idx, match_iou = match
        matched_seg = refined_frame.segmentations[matched_idx]
        other_masks = [
            getattr(seg, "mask", None)
            for j, seg in enumerate(refined_frame.segmentations)
            if j != matched_idx
        ]
        neighbours = [m for m in other_masks if m is not None]

        extra_kwargs: dict = {}
        if camera_to_base is not None:
            extra_kwargs["camera_to_base"] = camera_to_base

        refined_result = calculator.compute_result(
            matched_seg,
            refined_frame.depth_map,
            pixel_to_mm=None,
            other_object_masks=neighbours,
            **extra_kwargs,
        )

        refined_best = refined_result.best if refined_result is not None else None
        if refined_best is None:
            return RefinementReport(
                outcome=RefinementOutcome.NO_GRASP,
                initial_grasp=initial_grasp,
                refined_result=refined_result,
                matched_segmentation_index=matched_idx,
                match_iou=match_iou,
                telemetry={
                    "reasons": tuple(
                        str(r) for r in (refined_result.reasons or ())
                    ) if refined_result is not None else (),
                },
            )

        if (
            self.policy.require_same_grasp_frame
            and refined_best.frame is not initial_grasp.frame
        ):
            return RefinementReport(
                outcome=RefinementOutcome.DIVERGED,
                initial_grasp=initial_grasp,
                refined_grasp=refined_best,
                refined_result=refined_result,
                matched_segmentation_index=matched_idx,
                match_iou=match_iou,
                failure_reason=GraspFailureReason.REFINEMENT_DIVERGED,
                telemetry={
                    "reason": "grasp_frame_mismatch",
                    "initial_frame": str(initial_grasp.frame),
                    "refined_frame": str(refined_best.frame),
                },
            )

        deltas = _measure_correction_deltas(initial_grasp, refined_best)
        pos_delta = deltas.position_mm
        orient_delta = deltas.orientation_deg
        width_delta = deltas.width_mm

        violations: list[str] = []
        if pos_delta > self.policy.max_position_correction_mm:
            violations.append("position")
        if orient_delta > self.policy.max_orientation_correction_deg:
            violations.append("orientation")
        if width_delta > self.policy.max_grip_width_correction_mm:
            violations.append("grip_width")

        if violations:
            return RefinementReport(
                outcome=RefinementOutcome.DIVERGED,
                initial_grasp=initial_grasp,
                refined_grasp=refined_best,
                refined_result=refined_result,
                matched_segmentation_index=matched_idx,
                match_iou=match_iou,
                position_delta_mm=pos_delta,
                orientation_delta_deg=orient_delta,
                grip_width_delta_mm=width_delta,
                failure_reason=GraspFailureReason.REFINEMENT_DIVERGED,
                telemetry={"violated_bounds": tuple(violations)},
            )

        return RefinementReport(
            outcome=RefinementOutcome.ACCEPTED,
            initial_grasp=initial_grasp,
            refined_grasp=refined_best,
            refined_result=refined_result,
            matched_segmentation_index=matched_idx,
            match_iou=match_iou,
            position_delta_mm=pos_delta,
            orientation_delta_deg=orient_delta,
            grip_width_delta_mm=width_delta,
        )
