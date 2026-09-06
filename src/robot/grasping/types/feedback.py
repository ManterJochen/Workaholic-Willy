"""Structured feedback for the grasping pipeline.

:class:`GraspCalculator` deliberately does not throw for the common no-grasp cases: an
empty mask, every candidate colliding, no valid depth under the mask. A caller needs a
structured way to tell why the candidate list came back empty, so that it can retry
with a rescan or active perception, fall back to a simpler planner, surface the failure
to an operator, or keep telemetry naming the stage that rejected the candidates.

This module is a small vendor-neutral enum and a dataclass that wraps the
``list[GraspPoint]`` return without breaking the existing API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.robot.grasping.types.grasp_point import GraspPoint

__all__ = [
    "GraspFailureReason",
    "GraspResult",
]


class GraspFailureReason(StrEnum):
    """The reason codes explaining why the grasp pipeline yielded no result.

    The enum is deliberately coarse-grained. It is not a full diagnostic taxonomy, only
    enough resolution for an execution layer to choose between retry, rescan, escalation
    and abort.
    """

    EMPTY_MASK = "empty_mask"
    MASK_TOO_SMALL = "mask_too_small"
    NO_VALID_DEPTH = "no_valid_depth"
    LOW_DEPTH_CONFIDENCE = "low_depth_confidence"
    LOW_MASK_CONFIDENCE = "low_mask_confidence"
    NO_CANDIDATES_GENERATED = "no_candidates_generated"
    ALL_COLLIDED = "all_collided"
    ALL_OUT_OF_WORKSPACE = "all_out_of_workspace"
    ALL_TABLE_CONFLICT = "all_table_conflict"
    IK_FAILED = "ik_failed"
    NO_VALID_GRASP = "no_valid_grasp"
    RESCAN_RECOMMENDED = "rescan_recommended"
    TRY_NEXT_CANDIDATE = "try_next_candidate"
    ACTIVE_PERCEPTION_RECOMMENDED = "active_perception_recommended"
    # The mask topology is too risky for a stable parallel-jaw grasp: a ring, a U-shape,
    # a handle. The grasp is rejected before motion planning, and a rescan is recommended
    # unless an alternative strategy is configured upstream.
    TOPOLOGY_RISK_REJECTED = "topology_risk_rejected"
    # An operator-supplied semantic policy denied this candidate: the label is not in
    # the allow-list, is in the deny-list, or is below the confidence threshold.
    SEMANTIC_REJECTED = "semantic_rejected"
    # The target mask is heavily occluded by other detected objects: the fraction of the
    # target convex hull covered by neighbouring masks exceeded the configured
    # ``heavy_occlusion_threshold``. It pairs with ``ACTIVE_PERCEPTION_RECOMMENDED``, so
    # the caller can ask for a viewpoint change or a scene reshuffle.
    HEAVY_OCCLUSION = "heavy_occlusion"
    # The target was classified as a non-rigid object, a cable, cloth or bag, and the
    # configured ``DeformableHandlingStrategy`` refused to plan a parallel-jaw grasp. The
    # execution layer routes the request to a specialised handler, a deformable planner,
    # an operator or a different end-effector, rather than retrying the rigid pipeline.
    DEFORMABLE_ROUTING_REQUIRED = "deformable_routing_required"
    # In two-scan pre-grasp refinement, the target the initial grasp was computed on
    # could not be re-identified in the refined frame, because no matching segmentation
    # passed the tracker threshold. It is distinct from ``NO_VALID_GRASP``, so a caller
    # tells losing sight of the object from never having had a candidate.
    TARGET_LOST_DURING_REFINE = "target_lost_during_refine"
    # Refinement produced a grasp whose position, orientation or grip-width delta
    # exceeded the configured correction bounds. It is a refuse-to-execute signal: the
    # refined frame disagreed with the initial frame strongly enough that the correction
    # is not trusted.
    REFINEMENT_DIVERGED = "refinement_diverged"
    # The motion planner refused to route to an otherwise valid grasp and nothing moved,
    # which ``NO_PLAN_FAIL_SAFE_MESSAGE`` names. It is distinct from every reason above,
    # all of which describe why no grasp was found: here the grasp was found, scored and
    # accepted, and the cell then declined to drive to it. That distinction is the point
    # of the member. Measured: without it a planner refusal produces an attempt with an
    # empty reason tuple, so the recovery driver sees nothing to recover from and
    # terminates as ``escalated_no_recovery`` with zero actions.
    #
    # Its recovery mapping is deliberately the shortest in the table, ``(RESCAN,)`` and
    # nothing else. A refusal is a fail-closed decision, re-perceiving is free and
    # changes what the planner is asked next, and moving the arm after a refusal turns
    # that refusal into a motion. Widening this tuple is a safety change rather than a
    # tuning change.
    MOTION_PLAN_REFUSED = "motion_plan_refused"
    # The controller itself cannot move: a protective stop, an emergency stop, a
    # safety-mode stop, or simply powered off. It is distinct from every reason above,
    # which describe the scene or the grasp, because here nothing is wrong with either
    # and no amount of re-perceiving or re-ranking helps.
    #
    # Measured against URSim: with a real protective stop active, three commanded motions
    # came back as "RobotMotionRejected: Driver reported moveJ() failure", a message that
    # collapses to `execution_failed` and reads exactly like a bad grasp. The driver knew,
    # because get_robot_status reported stopped=True and raise_if_stopped raised, and
    # nothing in the pick path asked.
    #
    # Its recovery mapping is deliberately empty. A stopped cell needs a person rather
    # than a retry, and certainly not a motion.
    CONTROLLER_NOT_OPERATIONAL = "controller_not_operational"
    # A hard target label was requested and perception returned segmentations, and not
    # one of them carried that label. It is distinct from every reason above: nothing is
    # wrong with the scene, the grasp or the cell, and the operator asked for an object
    # this frame does not contain, or the detector named it something else.
    #
    # Measured through the operator console: a run prompted for the green cube against a
    # scene whose segmentations are unlabelled produced 8 valid candidates, skipped every
    # one of them at the label gate, and terminated with an attempt whose reason tuple
    # was empty, so the console rendered "no_valid_grasp". That sends an operator to look
    # at lighting and depth when the fix is one word in the prompt. The label gate itself
    # is correct and stays: refusing to lift an object it cannot confirm is the point of
    # a hard target.
    #
    # Its recovery mapping is deliberately empty, for a different reason from
    # CONTROLLER_NOT_OPERATIONAL above: there acting would be unsafe, and here it would
    # be pointless. The detector is deterministic on a static frame, so a rescan
    # re-derives the same labels and only burns the attempt budget. Mapping this to a
    # viewpoint relocate, since an occluded object may carry its label from another
    # angle, is defensible and is not done here: it needs a measurement first.
    #
    # ``GraspResult.telemetry['labels_seen']`` carries what perception did return, which
    # is the actionable half for whoever is reading.
    TARGET_LABEL_NOT_FOUND = "target_label_not_found"


@dataclass(frozen=True, slots=True)
class GraspResult:
    """The structured result of one ``GraspCalculator.compute`` call.

    Fields
    ------
    candidates
        The ranked grasp candidates, best first. It is empty where no valid grasp was
        found, and ``reasons`` is then non-empty.
    reasons
        The ordered, deduplicated failure and recommendation codes. It is empty for a
        successful call.
    telemetry
        The stage counters from the calculator, in the shape of
        ``GraspCalculator.last_telemetry``.
    depth_confidence
        The fraction of pixels under the mask with finite, positive depth. It is
        ``None`` where the calculator returned before this could be computed, as it does
        on an empty mask.
    mask_confidence
        The optional segmentation score, copied from the ``SegmentationResult.score``
        boundary field where one is present.
    top_score
        The score of the best candidate, or ``0.0`` where there is none.
    metadata
        Free-form extra data, such as a debug image hash or a calculator id.
    """

    candidates: tuple[GraspPoint, ...] = ()
    reasons: tuple[GraspFailureReason, ...] = ()
    telemetry: dict[str, Any] = field(default_factory=dict)
    depth_confidence: float | None = None
    mask_confidence: float | None = None
    top_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """``True`` where at least one valid candidate is available."""
        return bool(self.candidates)

    @property
    def best(self) -> GraspPoint | None:
        """The highest-ranked candidate, or ``None`` where the result is empty."""
        return self.candidates[0] if self.candidates else None

    def with_reasons(self, *reasons: GraspFailureReason) -> GraspResult:
        """Return a copy with further reasons appended, deduplicated."""
        seen: dict[GraspFailureReason, None] = dict.fromkeys(self.reasons)
        for reason in reasons:
            seen[reason] = None
        return GraspResult(
            candidates=self.candidates,
            reasons=tuple(seen.keys()),
            telemetry=self.telemetry,
            depth_confidence=self.depth_confidence,
            mask_confidence=self.mask_confidence,
            top_score=self.top_score,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "candidates": [grasp.to_dict() for grasp in self.candidates],
            "reasons": [reason.value for reason in self.reasons],
            "telemetry": dict(self.telemetry),
            "depth_confidence": self.depth_confidence,
            "mask_confidence": self.mask_confidence,
            "top_score": self.top_score,
            "metadata": dict(self.metadata),
            "is_success": self.is_success,
        }
