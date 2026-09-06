"""
Active-perception bin-picking orchestrator.

This module closes the loop that the calculator exposes via
:class:`GraspResult.reasons`. The calculator reports why a pick failed
(no candidates, all IK rejected, rescan recommended, active perception
recommended, ...); the orchestrator turns those codes into actions:

* ``RESCAN_RECOMMENDED`` or no-candidate: request a fresh perception
  frame from the perception source without moving the robot.
* ``ACTIVE_PERCEPTION_RECOMMENDED``: ask a
  :class:`ViewpointPlanner` for a new camera pose, drive the arm
  there, and re-acquire.
* ``IK_FAILED`` for the top candidate: try the next ranked candidate
  before escalating.
* Anything else: terminate with the corresponding outcome.

The orchestrator is vendor-neutral: it depends only on the
:class:`RobotArm` Protocol, the calculator, a perception protocol, and
an optional viewpoint planner. A real SAM2 perception source and a real
next-best-view solver plug in without changes to this file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

import numpy as np

from src.geometry import Frame, Pose, Transform
from src.robot.core import Gripper, MotionStatus, RobotArm, SupportsRobotStatus
from src.robot.grasping.loop._shadow_aggregator import (
    _ShadowTelemetryAggregator,
    _build_candidate_log as _build_candidate_log,  # re-exported under this module's name
)
from src.robot.grasping.motion.execution_policy import (
    GraspExecutionPolicy,
    PolicyOutcome,
    PolicyReport,
)
from src.robot.grasping.telemetry.latency_tracker import LatencyStage
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.generation.calculator import GraspCalculator
from src.robot.grasping.types.perception import (
    PerceptionFrame,
    PerceptionSource,
)
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.modes import (
    GraspSamplingMode,
    resolve_grasp_sampling_mode,
)
from src.robot.grasping.scoring.success_probability import (
    RankingBlendConfig,
    RankingBlendTelemetry,
    SHADOW_METADATA_KEY_PROBABILITY,
    ShadowSuccessContext,
    ShadowSuccessTelemetry,
    UncertaintyRerankConfig,
    UncertaintyRerankTelemetry,
    annotate_grasp_points_with_shadow_probability,
    maybe_blend_rerank_candidates,
    maybe_uncertainty_rerank_candidates,
)
from src.robot.grasping.loop.progress import (
    PickProgressListener,
    PickStage,
    ShouldCancel,
    emit,
)
from src.robot.grasping.loop.target_selector import (
    OrderingDecision,
    TargetOrderingConfig,
    select_target,
)
from src.robot.grasping.motion.trajectory_safety import (
    ApproachPathPolicy,
    validate_approach_and_retreat,
)
from src.robot.grasping.loop._pick_helpers import (
    _build_target_candidates,
    _grasp_point_to_grasp_pose,
)

if TYPE_CHECKING:  # pragma: no cover (import only for typing)
    from src.config.schema.robot.grasping_schema import (
        FusionGeometryConfig,
        GraspingSupportConfig,
    )
    from src.robot.grasping.deep.ranker.context import DeepRankerContext
    from src.robot.grasping.multiview.scene_geometry import FusedSceneGeometry
    from src.robot.grasping.types.perception import MultiCameraPerceptionSource
    from src.robot.grasping.collision import GripperGeometryStrategy
    from src.robot.grasping.scoring.corridor import CorridorAnalysisConfig
    from src.robot.grasping.telemetry.latency_tracker import LatencyTracker
    from src.robot.grasping.scoring.feasibility_score import FeasibilityScoreConfig
    from src.robot.grasping.motion.frame_resolver import FrameResolver
    from src.robot.grasping.multiview.fusion import (
        FusionTelemetry,
        SceneFusion,
    )
    from src.robot.grasping.rl.router import (
        PerceptionShadowTelemetry,
        RankingShadowTelemetry,
        RecoveryShadowTelemetry,
        SequencingAttemptState,
        SequencingShadowTelemetry,
        ShadowRouter,
    )

__all__ = [
    "BinPickingOrchestrator",
    "CommitDecision",
    "CommitPolicy",
    "LateralOffsetViewpointPlanner",
    "PerceptionFrame",
    "PerceptionSource",
    "PickAttempt",
    "PickOutcome",
    "PickReport",
    "ViewpointPlanner",
]


# ``PerceptionFrame`` and ``PerceptionSource`` live in
# ``grasping.types.perception`` (imported above) so downstream modules can name
# them without importing this orchestrator. They stay re-exported here (see
# ``__all__``) for backward compatibility. ``SegmentationLike`` lives beside
# them there and is not re-exported here; import it from
# ``grasping.types.perception``, or from ``grasping``, which does re-export it.


@runtime_checkable
class ViewpointPlanner(Protocol):
    """Compute the next camera pose when active perception is requested.

    ``current_tcp`` is the current arm TCP pose in :attr:`Frame.BASE`.
    ``history`` is the chronological list of viewpoints already tried
    this session. Return ``None`` to signal "no further viewpoint"; the
    orchestrator will then exit with :attr:`PickOutcome.RELOCATED_EXHAUSTED`.
    """

    def next_viewpoint(
        self,
        *,
        current_tcp: Pose,
        history: Sequence[Pose],
    ) -> Pose | None: ...


@dataclass(frozen=True, slots=True)
class CommitPolicy:
    """Runtime carrier for the multi-view commit gate.

    Immutable and stateless: per-pick reobserve accounting lives on the
    orchestrator instance, not here.
    """

    enabled: bool
    min_views_accepted: int
    min_corridor_hit_fraction: float
    corridor_radius_mm: float
    corridor_length_mm: float
    max_reobserve_attempts: int
    apply_modes: frozenset[str]


@dataclass(frozen=True, slots=True)
class CommitDecision:
    """Typed outcome of one commit-gate evaluation.

    Attached to :attr:`PickReport.commit_decision` on every pick with a winning
    candidate. When the gate is disabled or skipped the decision still reports
    ``allowed=True``, and ``reason`` says why the gate did not participate.
    """

    allowed: bool
    reason: str  # See module-level _COMMIT_REASON_* constants
    views_accepted: int
    min_views_required: int
    corridor_hit_fraction: float
    min_hit_fraction_required: float
    corridor_queried_voxels: int
    corridor_hit_voxels: int
    reobserve_attempts_used: int


# Stable string constants for ``CommitDecision.reason``. They are module-level
# strings rather than an Enum: the values round-trip through JSON and logs
# verbatim and are greppable as written.
_LOG = logging.getLogger(__name__)

#: How much vertical extent a single-view target cloud needs before it may be used to locate the
#: support plane, in millimetres along the support normal.
#:
#: Not a quality threshold. It separates two situations that a point cloud cannot otherwise tell
#: apart: seeing down the side of an object to what it rests on, and looking straight at one face.
#: Only the first has observed the support; the second, used as an observation, puts the plane on
#: top of the object. 25 mm is a jaw pad's own dimension: below that there is not enough geometry
#: for the lowest point to mean anything, and the declared height, which an operator can at least
#: verify, is the better answer.
_MIN_CLOUD_EXTENT_FOR_SUPPORT_MM: float = 25.0

_COMMIT_REASON_OK = "ok"
_COMMIT_REASON_VIEWS_BELOW_MIN = "views_below_min"
_COMMIT_REASON_HIT_FRACTION_BELOW_MIN = "hit_fraction_below_min"
_COMMIT_REASON_SKIPPED_DISABLED = "skipped_disabled"
_COMMIT_REASON_SKIPPED_NO_FUSION = "skipped_no_fusion"
_COMMIT_REASON_SKIPPED_NO_POLICY = "skipped_no_policy"
_COMMIT_REASON_SKIPPED_MODE = "skipped_mode"
_COMMIT_REASON_SKIPPED_CAMERA_FRAME = "skipped_camera_frame"


class PickOutcome(str, Enum):
    """Terminal status of a single :meth:`BinPickingOrchestrator.run` call."""

    EXECUTED = "executed"
    RESCANNED_EXHAUSTED = "rescanned_exhausted"
    RELOCATED_EXHAUSTED = "relocated_exhausted"
    NO_PERCEPTION = "no_perception"
    ABORTED = "aborted"
    # An operator asked the run to stop between attempts. Deliberately not ``ABORTED``: that maps to
    # EXECUTION_FAILED, which would count every press of a stop button as a failed execution and quietly
    # poison the pick-rate KPI with operator decisions.
    CANCELLED = "cancelled"
    OBJECT_NOT_DETECTED = "object_not_detected"
    EXECUTION_FAILED = "execution_failed"
    # The winning grasp candidate is a camera-frame pose, and the execution
    # policy refuses to command it, fail-closed per the frame contract. Kept
    # distinct from EXECUTION_FAILED so a caller can detect missing
    # FrameResolver wiring.
    CAMERA_FRAME_REJECTED = "camera_frame_rejected"
    # The multi-view commit gate refused the winning candidate (views below
    # ``min_views_accepted`` and/or corridor hit-fraction below
    # ``min_corridor_hit_fraction``) and the ``max_reobserve_attempts`` budget
    # was already consumed. Distinct from RESCANNED_EXHAUSTED so the operator can
    # tell apart "no valid candidates" from "had a candidate but the fused
    # evidence was insufficient to commit".
    NO_COMMIT_INSUFFICIENT_FUSION = "no_commit_insufficient_fusion"
    # Dense-mode approach validation. Every ranked candidate's straight-line
    # approach/retreat sweep collided with a neighbour in the scene cloud (the
    # calculator only checks the final pose), so the fall-back found no
    # sweep-clear candidate and refused rather than driving into an obstacle.
    APPROACH_PATH_BLOCKED = "approach_path_blocked"
    # The controller could not move: protective stop, emergency stop, a safety-mode stop, or simply
    # powered off. Deliberately not EXECUTION_FAILED, for the same reason CANCELLED is not ABORTED:
    # the grasp did not fail, the cell stopped, and folding the two together buries a machine state
    # an operator must act on inside the failure taxonomy.
    #
    # A stopped controller reports commanded motions as a driver failure ("Driver reported moveJ()
    # failure"), which reads exactly like a bad grasp unless the driver is asked: get_robot_status
    # reports the stop and raise_if_stopped raises on it.
    CONTROLLER_NOT_OPERATIONAL = "controller_not_operational"


def _motion_fields(report: "PolicyReport | None") -> dict[str, str | None]:
    """The three motion explanations a ``PolicyReport`` carries, flattened onto an attempt.

    Best-effort and never raising: a report that predates these fields, or an object that is not a
    ``PolicyReport`` at all, yields three ``None``s and the attempt is exactly as before.
    Stringified on the way out so the attempt stays JSON-safe for the telemetry tail.
    """
    if report is None:
        return {"motion_status": None, "motion_message": None, "motion_error": None}
    status = getattr(report, "motion_status", None)
    err = getattr(report, "error", None)
    return {
        "motion_status": getattr(status, "value", None) if status is not None else None,
        "motion_message": (
            str(getattr(report, "motion_message", None))
            if getattr(report, "motion_message", None) is not None
            else None
        ),
        "motion_error": f"{type(err).__name__}: {err}" if err is not None else None,
    }


@dataclass(frozen=True, slots=True)
class PickAttempt:
    """One iteration of the pick loop, surfaced for debugging."""

    attempt_index: int
    target_index: int | None
    reasons: tuple[GraspFailureReason, ...]
    score: float
    # "executed", "object_not_detected", "camera_frame_rejected",
    # "approach_path_blocked" or "execution_failed" from _map_policy_outcome;
    # "rescan", "relocate" or "exhausted" from _decide_action;
    # "commit_refused_reobserve" and "commit_refused_exhausted" from the commit
    # gate; "controller_not_operational" when the controller cannot move.
    action: str
    # When ``target_ordering`` is enabled this carries the typed
    # :class:`OrderingDecision` produced by the selector for this
    # attempt. Defaults to ``None`` so legacy consumers see no surface
    # change.
    ordering_decision: OrderingDecision | None = None
    # Why the motion failed, in the one place that survives the pick.
    #
    # ``_map_policy_outcome`` collapses a ``PolicyReport`` into the single word "execution_failed",
    # dropping the report's typed ``motion_status``, the driver's own ``motion_message`` and the
    # exception, so a cell that reports a failed motion otherwise keeps no explanation of it.
    #
    # Copied onto the attempt rather than referenced: the orchestrator keeps only the last policy
    # report (``_last_policy_report``), so on a multi-attempt pick every earlier explanation is gone
    # by the time anyone reads it. Optional and defaulted, so every existing construction site is
    # unchanged.
    motion_status: str | None = None
    motion_message: str | None = None
    motion_error: str | None = None


@dataclass(frozen=True, slots=True)
class PickReport:
    """Aggregated outcome of a :meth:`BinPickingOrchestrator.run` call."""

    outcome: PickOutcome
    attempts: tuple[PickAttempt, ...] = ()
    executed_grasp: GraspResult | None = None
    #: The telemetry of the grasp the loop chose, whatever happened next.
    #:
    #: `executed_grasp` is set only on `PickOutcome.EXECUTED`, which is right for what it names. But
    #: the calculator's telemetry, and the `deep_ranker_*` keys the shadow ranker stamps onto it, are
    #: about the candidates rather than about whether the motion succeeded. Read off `executed_grasp`
    #: they are absent from every `EXECUTION_FAILED` attempt: measured on the offline feature sweep,
    #: three of four records, each scored by the shadow whose opinion then reached no record.
    #:
    #: Empty when nothing was selected, so a run that generated no candidate is byte-identical.
    selected_telemetry: Mapping[str, Any] = field(default_factory=dict)
    viewpoints_visited: tuple[Pose, ...] = ()
    # Optional shadow-mode telemetry for the winning candidate of the
    # final attempt. ``None`` whenever the orchestrator is built without
    # a :class:`ShadowSuccessContext`, which is the case when the
    # success model is disabled or failed to load. Populated for both
    # ``EXECUTED`` outcomes and the ``EXECUTION_FAILED`` /
    # ``CAMERA_FRAME_REJECTED`` branches when the model scored the
    # chosen grasp.
    shadow_success_telemetry: ShadowSuccessTelemetry | None = None
    # Audit telemetry from the bounded probability-blend reranker.
    # ``None`` when ``ranking_blend_config`` was not provided or
    # ``ranking_blend_config.enabled is False`` (the silent
    # byte-identical no-op). When non-``None``, the ``applied`` field
    # tells whether the rerank actually fired and ``skip_reason`` names
    # the gate when it did not.
    ranking_blend_telemetry: RankingBlendTelemetry | None = None
    # Per-candidate uncertainty re-rank audit. ``None`` when the rerank config is unset/disabled
    # (the silent byte-identical path); otherwise an UncertaintyRerankTelemetry whose ``applied`` tells
    # whether the rerank fired and ``skip_reason`` names the gate when it did not.
    uncertainty_rerank_telemetry: UncertaintyRerankTelemetry | None = None
    # Shadow-only multi-view fusion telemetry. ``None`` when the
    # orchestrator was built without a ``scene_fusion`` carrier or when
    # the carrier is disabled. The contract is strict: ingest never
    # influences ranking, decision, or recovery; the substrate only
    # accumulates the bounded voxel grid and surfaces this summary for
    # replay / observability. Even when the field is populated, the rest
    # of ``PickReport`` is byte-identical to the pre-fusion path.
    fusion_telemetry: "FusionTelemetry | None" = None
    # Typed commit-gate decision for the candidate the orchestrator
    # chose for execution, or for the one it last evaluated before
    # exhausting its reobserve budget. ``None`` only when the pick
    # exits before any candidate is selected, for example on empty
    # segmentations or on recovery exhausted before a winner is found.
    # A disabled gate still populates this field, with ``allowed=True``
    # and ``reason`` naming the skip path, so downstream telemetry can
    # distinguish "no gate configured" from "gate ran and approved".
    commit_decision: "CommitDecision | None" = None


# ---------------------------------------------------------------------------
# Built-in viewpoint planner
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LateralOffsetViewpointPlanner:
    """Trivial next-best-view stub: nudge the TCP laterally in base frame.

    Alternates ``+x``, ``-x``, ``+y``, ``-y`` offsets so the camera
    captures the bin from four cardinal positions before giving up.
    A deterministic fallback until a learning-based planner lands.
    """

    offset_mm: float = 50.0
    max_viewpoints: int = 4

    def next_viewpoint(
        self,
        *,
        current_tcp: Pose,
        history: Sequence[Pose],
    ) -> Pose | None:
        if len(history) >= self.max_viewpoints:
            return None
        directions = (
            np.array([+1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, +1.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
        )
        direction = directions[len(history) % len(directions)]
        new_position = (
            np.asarray(current_tcp.position_mm, dtype=np.float64)
            + direction * float(self.offset_mm)
        )
        return Pose(
            position_mm=new_position,
            quaternion_xyzw=np.asarray(current_tcp.quaternion_xyzw, dtype=np.float64).copy(),
            frame=Frame.BASE,
            label=f"viewpoint_{len(history):02d}",
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------



def _mean_winning_association_score(fused: "FusedSceneGeometry") -> float:
    """Mean score over the (object, view) pairs that won their assignment. ``0.0`` when none did.

    Reads the associations the fusion already produced rather than recomputing anything, and counts
    only winning pairs: a losing pair's score says how strongly some other object was rejected, which
    is not evidence about the object that was kept.

    Never raises: an association shape it does not recognise contributes nothing rather than ending
    a pick, because this is telemetry.
    """
    total = 0.0
    count = 0
    for association in getattr(fused, "associations", ()) or ():
        assignment = getattr(association, "assignment", ()) or ()
        scores = getattr(association, "scores", ()) or ()
        for primary_index, matched in enumerate(assignment):
            if matched is None or primary_index >= len(scores):
                continue
            row = scores[primary_index]
            if matched < len(row):
                total += float(row[matched])
                count += 1
    return round(total / count, 6) if count else 0.0


_RESCAN_REASONS = frozenset(
    {
        GraspFailureReason.RESCAN_RECOMMENDED,
        GraspFailureReason.NO_CANDIDATES_GENERATED,
        GraspFailureReason.LOW_DEPTH_CONFIDENCE,
        GraspFailureReason.LOW_MASK_CONFIDENCE,
        GraspFailureReason.NO_VALID_DEPTH,
    }
)
_RELOCATE_REASONS = frozenset(
    {
        GraspFailureReason.ACTIVE_PERCEPTION_RECOMMENDED,
        GraspFailureReason.ALL_OUT_OF_WORKSPACE,
    }
)

@dataclass
class BinPickingOrchestrator:
    """Drive the perceive, grasp, execute loop with failure-aware retries.

    Parameters
    ----------
    arm:
        Any :class:`RobotArm` driver. Only :meth:`get_tcp_pose` and
        :meth:`move_to` are required for the default pipeline; a real
        retreat motion additionally calls :meth:`move_to` once more.
    calculator:
        Configured :class:`GraspCalculator` instance.
    perception:
        Source of :class:`PerceptionFrame` snapshots.
    viewpoint_planner:
        Optional planner for ``ACTIVE_PERCEPTION_RECOMMENDED``. If
        omitted, an active-perception recommendation falls through to a
        plain rescan.
    max_attempts:
        Hard cap on the loop iterations.
    dense_sampling:
        Legacy bool surface forwarded to :meth:`GraspCalculator.compute`.
        Recommended ``True`` for cluttered bins. Kept for backward
        compatibility; new callers should use ``grasp_sampling_mode``.
    grasp_sampling_mode:
        Typed grasp sampling mode (:class:`GraspSamplingMode`, legacy
        bool, string alias, or ``None``). When provided it overrides
        ``dense_sampling``; otherwise the legacy bool drives behavior.
        Default ``None`` resolves to ``AUTO`` so the calculator picks
        between silhouette and dense sampling heuristically.
    """

    arm: RobotArm
    calculator: GraspCalculator
    perception: PerceptionSource
    viewpoint_planner: ViewpointPlanner | None = None
    max_attempts: int = 5
    dense_sampling: bool = True
    grasp_sampling_mode: GraspSamplingMode | bool | str | None = None
    standoff_mm: float = 80.0
    retreat_mm: float = 100.0
    gripper: Gripper | None = None
    policy: GraspExecutionPolicy | None = None
    # Optional resolver for ``T_cam_to_base`` per frame. When wired, the
    # orchestrator forwards the resolved transform into
    # ``calculator.compute_result(camera_to_base=...)`` so the returned
    # grasps are in base frame. ``None``, the default, forwards nothing:
    # the calculator falls back to its constructor-time
    # ``camera_matrix`` and the candidates stay in the frame the
    # calculator chooses, which is camera unless the constructor was
    # configured with a transform.
    frame_resolver: "FrameResolver | None" = None
    # Optional clutter-aware target selector. When ``None`` or when
    # ``target_ordering.enabled`` is False the orchestrator takes the
    # plain ``argmax(local_score)`` path. When enabled the
    # per-segmentation winners are passed through :func:`select_target`
    # and the chosen candidate plus decision telemetry are returned.
    target_ordering: TargetOrderingConfig | None = None
    # Shadow-mode success-probability model carrier. When ``None`` (the
    # default) the orchestrator behaves byte-identically: no
    # ``GraspPoint`` metadata is mutated and
    # ``PickReport.shadow_success_telemetry`` stays ``None``. When set,
    # the chosen result's candidates are annotated with calibrated
    # probabilities, read-only with respect to ranking, and the winner's
    # telemetry is surfaced on the report for downstream replay tooling.
    shadow_success_context: ShadowSuccessContext | None = None
    # Bounded probability-blend reranker config. ``None`` or
    # ``enabled=False`` is the silent byte-identical no-op:
    # :func:`maybe_blend_rerank_candidates` returns the original tuple
    # and ``None`` telemetry, and the
    # ``PickReport.ranking_blend_telemetry`` field stays ``None``. When
    # enabled, the reranker fires only in dense modes
    # (``dense_clutter`` / ``dense_autonomous``) under a non-shadow
    # lifecycle phase and never mutates ``GraspPoint.score``.
    ranking_blend_config: RankingBlendConfig | None = None
    # Per-candidate uncertainty re-rank carrier. ``None`` (default) is a silent byte-identical
    # no-op. When set, the orchestrator re-sorts candidates by ``score - weight*corridor_risk`` after the
    # blend (dense-only, non-shadow, weight>0, every candidate carrying a corridor-risk signal). A
    # producer must run: either ``GraspCalculator.corridor_risk_per_candidate`` set at construction,
    # or a ``corridor_config`` forwarded to ``compute()``, which this orchestrator does whenever its
    # own ``corridor_config`` is set. With neither, no candidate carries
    # ``corridor_blockage_confidence`` and the rerank no-ops with ``skip_reason=missing_uncertainty``.
    uncertainty_rerank_config: UncertaintyRerankConfig | None = None
    # Optional multi-view fusion substrate. ``None`` is the silent
    # byte-identical default. When set, and when a ``frame_resolver`` is
    # also wired, the orchestrator calls ``scene_fusion.ingest(...)``
    # once per acquired :class:`PerceptionFrame` before grasp
    # computation. The fusion path is shadow-only: its output is
    # never read by the calculator, the execution policy, or the
    # recovery dispatcher. Ingest failures (no resolver, frame mismatch,
    # intrinsic drift, all-invalid depth) are absorbed by the
    # substrate's typed refuse path and surfaced via
    # :attr:`PickReport.fusion_telemetry`; the pick loop never raises on
    # a fusion problem and never alters its decision based on fusion
    # state.
    scene_fusion: "SceneFusion | None" = None
    # Optional commit-gate policy. ``None`` or ``commit_policy.enabled
    # is False`` is the silent byte-identical default. When enabled and
    # :attr:`mode_label` matches one of the policy's apply modes, the
    # orchestrator queries the fusion substrate for corridor evidence
    # of the winning candidate before handing off to ``_execute``.
    # When the evidence is insufficient the loop triggers up to
    # :attr:`CommitPolicy.max_reobserve_attempts` extra perception
    # captures before surfacing
    # :attr:`PickOutcome.NO_COMMIT_INSUFFICIENT_FUSION`.
    commit_policy: CommitPolicy | None = None
    # Resolved grasp-mode label (e.g. ``"auto"``, ``"dense_clutter"``,
    # ``"dense_autonomous"``). Empty string means the orchestrator was
    # constructed without mode awareness, in which case the commit gate
    # is unconditionally skipped with reason ``skipped_mode``; this
    # preserves byte-identical behavior for every caller that has not
    # threaded a mode label through, notably any caller that constructs
    # the orchestrator directly instead of via ``from_robot_config``.
    mode_label: str = ""
    # Optional hard label-target for a prompt-driven dense pick. ``None`` (default) is
    # byte-identical: every segmentation is a candidate target (legacy argmax / ordering). When set to an
    # object label, only the matching segmentation is considered as the executable target; the other
    # segmentations still feed ``other_object_masks`` (the neighbour clutter the DENSE_CLUTTER sampler +
    # corridor-risk + approach-validation consume), so the prompt picks that object and never a
    # distractor. If the target's grasp fails the pick refuses (an honest no-pick, not a wrong-object pick).
    target_label: str | None = None

    # Multi-view scene collision: an optional fused multi-view scene point cloud in the
    # robot BASE frame (mm), supplied by a multi-view runner from the fixed cameras (overhead + obliques)
    # that see the whole scene, including neighbour surfaces the single grasp-synthesis view (e.g. the
    # eye-in-hand wrist) cannot see. ``None`` (default) is byte-identical: nothing extra is fed to the
    # collision filter. When set, the orchestrator transforms it into the captured frame's CAMERA frame
    # (the frame the collision filter plans in) and passes it as ``scene_points_mm``, vstacked with any
    # same-view neighbours, so a grasp that would collide with a neighbour the synthesis view missed is
    # rejected and the neighbour pick survives. The grasp synthesis stays single-view.
    external_scene_points_base_mm: "np.ndarray | None" = None

    # Optional fused multi-view target cloud (BASE frame) fed to the calculator's geometry generator
    # so it finds real side-face antipodal contacts a single top-down view cannot (the measured
    # bottleneck). ``None`` (default) is byte-identical (key never added). Passed straight through as
    # ``geometry_points_base_mm`` (the calculator transforms BASE->CAMERA itself); only forwarded when a
    # ``target_label`` is set (so the target's fused cloud is never applied to a non-target segmentation).
    external_target_geometry_base_mm: "np.ndarray | None" = None

    # Multi-camera geometry fusion (``grasping.fusion.geometry``). The generalisation of the field
    # above: instead of one externally-supplied cloud for one labelled target, the rig is observed
    # every pick and every candidate object gets the surface every camera that can identify it sees.
    # That is what a fixed multi-camera cell can do and a single view cannot: a candidate is
    # accepted on its closing axis, the closing axis comes from the silhouette, and one view sees
    # one side while an antipodal grasp needs two.
    #
    # All three ``None`` (default) means the keys are never added and the single-view path is
    # byte-identical. ``multi_camera_perception`` supplies the other cameras' frames,
    # ``camera_frame_resolvers`` maps each camera id to its own CAMERA->BASE (built from
    # ``grasping.fusion.cameras``, one persisted calibration per camera), and
    # ``fusion_geometry_config`` carries the metric, the match threshold and the missing-camera
    # posture.
    multi_camera_perception: "MultiCameraPerceptionSource | None" = None
    camera_frame_resolvers: "dict[str, FrameResolver] | None" = None
    #: Which cameras the config asked for, which is not the same as which ones built.
    #:
    #: `on_camera_unavailable: refuse` needs an expectation that a camera failing to build cannot
    #: erase. `set(camera_frame_resolvers)` is not one: that map is `{}` whenever `fusion.enabled`
    #: is false, or the `cameras` map is empty, or `grasping_cfg` is None, and in each of those
    #: cases `expected` is empty, `missing` is empty, and the refusal cannot fire. An expectation
    #: derived from what succeeded cannot notice what did not. The schema says what that costs, in
    #: the option's own description: "a cell must never fall back to single-view silently, which is
    #: the whole failure this option exists to make visible."
    #:
    #: Empty means "nothing was configured", not "nothing arrived", and the two are different
    #: answers. A cell with no fusion block configured is single-view by design and must not be
    #: refused; a cell that named three cameras and got two must be.
    configured_camera_ids: "tuple[str, ...]" = ()
    fusion_geometry_config: "FusionGeometryConfig | None" = None

    # What the gripper is, and what the parts stand on. Both are inputs ``compute()`` accepts and
    # this loop must supply. Unsupplied, the calculator filters candidates against a default gripper
    # envelope and skips the table check entirely: ``collision_checker.py`` runs that check only when
    # ``support_plane`` is not None, and with it skipped ``rejected_table`` measured 0 in 2128 of
    # 2128 telemetry lines while an independent reference rejected 37 % of the same candidates for
    # driving a finger under the support.
    #
    # ``gripper_model`` comes from ``grasping.gripper_geometry`` via ``build_gripper_geometry``.
    # ``support_config`` comes from ``grasping.support``; the plane is resolved per pick, because the
    # declared height cannot know that a part is standing on another part and the target's own lowest
    # point can (measured: the declared height is too low on 100 % of stacked parts, the footprint on
    # 0 % of all 1073). Both ``None`` (default) means the keys are never added and the path is
    # byte-identical.
    gripper_model: "GripperGeometryStrategy | None" = None
    support_config: "GraspingSupportConfig | None" = None
    # ``deep_ranker_context`` comes from ``grasping.deep_ranker`` via ``apply_orchestrator_overlays``.
    # None (default) means nothing is scored and the path is byte-identical. Set means every
    # attempt's candidates are scored by the learned ranker and the result is stamped into
    # telemetry, changing no ordering and no decision.
    #
    # Shadow, and for a measured reason rather than caution: the failure a ranker has to predict
    # here is `finger_collision`, and features that cannot see the surroundings cannot predict it.
    # A ranker scored against an analytic verdict rather than against physics earns the right to be
    # watched on real picks and nothing more.
    deep_ranker_context: "DeepRankerContext | None" = None
    # ``corridor_config`` comes from ``grasping.occlusion`` via ``apply_orchestrator_overlays``.
    # None (default) is not forwarded, so the calculator's per-candidate corridor producer keeps its
    # own geometry. Set forwards the operator's corridor geometry into every compute() call this
    # orchestrator makes, which is the point: unforwarded, the block's values reach the telemetry
    # snapshot and stop there.
    corridor_config: "CorridorAnalysisConfig | None" = None
    # Optional latency tracker, set per pick by the service. The single ranking recording site.
    #
    # The span belongs on the one call every path shares: the default open-loop pick, the decision
    # path and the closed-loop path all reach it, while a wrap further out covers only some of them
    # and leaves the open-loop pick with no ranking span at all. Timing the open-loop path from
    # outside is not an option either: that path calls `runtime.run_attempt()`, whose elapsed time
    # includes perception and arm motion, and reporting that against an 80 ms ranking SLO would
    # breach instantly and mean nothing. A second wrap around a caller nests a span inside a span
    # and double-counts.
    latency_tracker: "LatencyTracker | None" = field(default=None, repr=False)
    # ``feasibility_*`` come from ``grasping.feasibility`` via ``apply_orchestrator_overlays``.
    # None (default) is not forwarded, so the calculator keeps its constructor values, i.e. off.
    # Enabling this is expected to cost picks. calculator.py carries an on-box measurement: the
    # ik_quality sub-signal took the UR5e overhead pick 10/10 -> 0/10 at every weight, because the
    # good top-down grasp is itself near-singular, so "demote high condition number" demotes
    # precisely the grasp the pick needs. Wired so the claim can be re-measured through config; it
    # stays default-off regardless of how the number comes out.
    feasibility_config: "FeasibilityScoreConfig | None" = None
    feasibility_weight: float | None = None

    # Optional swept-volume approach/retreat validator. ``None`` (default) is a strict no-op
    # (byte-identical for single-object and for every direct construction). When set and the
    # resolved mode is in ``_approach_path_modes`` (dense-only, set by builders), the orchestrator
    # validates each ranked candidate's approach/retreat sweep against the scene neighbour cloud and
    # executes the first sweep-clear candidate (fall-back), refusing with APPROACH_PATH_BLOCKED if all
    # are blocked. The calculator only checks the final grasp pose; this adds the intermediate sweep.
    approach_path_policy: "ApproachPathPolicy | None" = None
    _approach_path_modes: frozenset[str] = field(default_factory=frozenset, repr=False)

    # Optional :class:`ShadowRouter` carrying a ranking-shadow policy.
    # ``None`` is the silent byte-identical default. When set and the
    # router has a ``ranking_policy`` attached, the orchestrator runs the
    # ranker after the blend block and stashes the typed telemetry on
    # :attr:`_ranking_telemetry`. The ranker never mutates
    # ``best_result``, ``GraspPoint.score``, or any other field that
    # influences execution.
    shadow_router: "ShadowRouter | None" = None

    _viewpoints_visited: list[Pose] = field(default_factory=list, init=False, repr=False)
    _last_policy_report: PolicyReport | None = field(default=None, init=False, repr=False)
    _resolved_sampling_mode: GraspSamplingMode = field(
        default=GraspSamplingMode.AUTO, init=False, repr=False
    )
    # Per-pick reobserve counter. Reset to 0 at the top of :meth:`run`
    # and incremented every time the commit gate refuses and the budget
    # allows another perception capture.
    _commit_reobserve_count: int = field(default=0, init=False, repr=False)
    # ---- live progress + cancellation (both default-off, both byte-identical when unset) ----
    # A caller that wants to watch a pick attaches ``on_progress``; with it unset every emit is one
    # ``is None`` test and no payload is built. Without it a server subscribing to a live pick
    # receives exactly one message, after the pick is over.
    on_progress: "PickProgressListener | None" = None
    # Asked once at the top of each attempt. ``True`` means "do not begin another attempt"; it never
    # interrupts a motion already in flight, because there is no safe way to do that from here and the
    # physical stop is the one that can. Unset leaves the loop a bare ``for``.
    should_cancel: "ShouldCancel | None" = None
    # Per-pick CAMERA->BASE transform for the winning frame, stashed by
    # _best_result_over_segmentations so _execute can map the camera-frame neighbour cloud into the
    # candidate (BASE) frame before the swept-volume check. Reset per pick.
    _pending_camera_to_base: "Transform | None" = field(default=None, init=False, repr=False)
    # What the multi-camera geometry fusion actually did this frame (cameras that contributed,
    # objects fused). Empty while fusion is off, so the telemetry stays byte-identical.
    _fusion_geometry_telemetry: dict = field(default_factory=dict, init=False, repr=False)
    # (config key, BASE points). The walls are fixed geometry; rebuilding them every attempt would
    # be waste in the one path that runs on every pick.
    _container_wall_cache: "tuple[tuple, np.ndarray] | None" = field(
        default=None, init=False, repr=False
    )
    # Per-pick ranking-shadow telemetry. Reset to ``None`` at the top of
    # :meth:`run`; populated at most once (on the attempt that reaches
    # the blend seam). Read by ``AutonomousGraspService`` post-pick and
    # never fed back into the deterministic decision path.
    _ranking_telemetry: "RankingShadowTelemetry | None" = field(
        default=None, init=False, repr=False
    )
    # Per-pick per-candidate log. ``_candidate_log`` = the top-N candidates' 15-key feature
    # rows + a tail aggregate; ``_behavior_candidate_id`` = the executed (rank-0) candidate id. Persisted into
    # ``record.extra`` (rl_candidate_features / rl_behavior_candidate_id) by ``attach_shadow_router`` so the
    # offline RL + OPE/promotion can train + recover the behavior action from real logged picks. Reset
    # per pick; populated at most once in :meth:`_maybe_run_ranking_shadow`, which is shadow-gated, so
    # the default is off and byte-identical. Never fed back into the deterministic decision path.
    _candidate_log: "list[dict[str, object]] | None" = field(
        default=None, init=False, repr=False
    )
    _behavior_candidate_id: "str | None" = field(default=None, init=False, repr=False)
    # Per-pick sequencing-shadow accumulators. ``_sequencing_current_attempts``
    # is a stable reference to the live attempts list; ``_sequencing_states``
    # collects the per-failure :class:`SequencingAttemptState` rows; and
    # ``_sequencing_telemetry`` carries the finalised typed telemetry for the
    # post-pick reader. All three are reset/populated inside the pick loop.
    _sequencing_current_attempts: list[PickAttempt] = field(
        default_factory=list, init=False, repr=False
    )
    _sequencing_states: "list[SequencingAttemptState]" = field(
        default_factory=list, init=False, repr=False
    )
    _sequencing_telemetry: "SequencingShadowTelemetry | None" = field(
        default=None, init=False, repr=False
    )
    # Per-pick perception-budget + recovery shadow telemetry. Reset at the top of the
    # pick and populated at most once by _finalize_perception/recovery (logging-only carriers; never fed back into the
    # deterministic perceive, score, decide, commit, execute path). Read post-pick by attach_shadow_router.
    _perception_telemetry: "PerceptionShadowTelemetry | None" = field(
        default=None, init=False, repr=False
    )
    _recovery_telemetry: "RecoveryShadowTelemetry | None" = field(
        default=None, init=False, repr=False
    )
    # The stateless shadow-telemetry producer + finalizer logic. Owns no
    # per-pick state: the slots above stay here; the aggregator reads loop state via getter-closures
    # and returns telemetry, and the orchestrator's thin delegators assign their own slots
    # (return-and-assign). Built in __post_init__.
    _shadow: "_ShadowTelemetryAggregator" = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # ---- resolve grasp sampling mode once per orchestrator ----
        # Precedence: explicit ``grasp_sampling_mode`` wins. When only
        # ``dense_sampling`` is provided the legacy bool meaning holds
        # (``True`` selects DENSE_CLUTTER, ``False`` SINGLE_OBJECT) so old
        # code does not change behavior. The resolved enum is what reaches
        # the calculator at every attempt.
        if self.grasp_sampling_mode is not None:
            self._resolved_sampling_mode = resolve_grasp_sampling_mode(
                self.grasp_sampling_mode
            )
        else:
            self._resolved_sampling_mode = resolve_grasp_sampling_mode(
                self.dense_sampling
            )

        # Build a default policy when none is provided. ``standoff_mm`` and
        # ``retreat_mm`` are forwarded so a caller that sets them on the
        # orchestrator gets them on the policy.
        if self.policy is None:
            self.policy = GraspExecutionPolicy(
                arm=self.arm,
                gripper=self.gripper,
                standoff_mm=self.standoff_mm,
                retreat_mm=self.retreat_mm,
            )

        # The shadow-telemetry aggregator (logic-only; reads loop state via getter-closures-over-self
        # so it always sees end-of-pick values: _commit_reobserve_count is incremented mid-loop and
        # _sequencing_current_attempts is rebound per pick).
        self._shadow = _ShadowTelemetryAggregator(
            max_attempts_getter=lambda: self.max_attempts,
            shadow_router_getter=lambda: self.shadow_router,
            viewpoints_getter=lambda: self._viewpoints_visited,
            commit_reobserve_getter=lambda: self._commit_reobserve_count,
            resolved_mode_getter=lambda: self._resolved_sampling_mode,
            commit_policy_getter=lambda: self.commit_policy,
        )

    def _resolve_support(self, seg, frame, camera_to_base):  # noqa: ANN001, ANN202
        """The support plane for this pick: the declared height, raised by what the cameras saw.

        The observation is the fused target cloud when one is available, because a single view of a bin
        sees the part over the wall and reports its rim as the base: measured p90 error 50.6 mm in the
        bin family against 13.3 mm fused. When no fused cloud exists the current view's own target cloud
        is used, which is still correct and only noisier. ``support_config`` unset returns ``None``, the
        key is never added, and the path is byte-identical.
        """
        if self.support_config is None:
            return None
        from src.robot.grasping.collision import resolve_support_plane  # noqa: PLC0415

        cfg = self.support_config
        clouds: list = []
        if cfg.refine_from_target:
            if self.external_target_geometry_base_mm is not None and self.target_label is not None:
                clouds.append(self.external_target_geometry_base_mm)
            elif camera_to_base is not None:
                cloud = self._target_cloud_base_mm(seg, frame, camera_to_base)
                if cloud is not None:
                    clouds.append(cloud)
        return resolve_support_plane(
            declared_height_mm=cfg.height_mm,
            container_floor_mm=cfg.container.floor_height_mm,
            normal=cfg.normal,
            target_clouds_base_mm=clouds or None,
            refine_from_target=cfg.refine_from_target,
        )

    def _target_cloud_base_mm(self, seg, frame, camera_to_base):  # noqa: ANN001, ANN202
        """This view's target surface in BASE mm, or None when it cannot be built.

        The frame is the whole point of this function, and two conversions can lose it silently:

        * ``camera_to_base`` arrives as a :class:`Transform` from the frame resolver, and
          ``np.asarray(Transform, dtype=float)`` raises rather than yielding a matrix, so a guard on
          ``transform.shape`` alone never sees a Transform: the call dies before it, outside the
          ``try``.
        * ``masked_point_cloud`` returns a :class:`MaskedPointCloud`, not an array. Treating it as
          one is how a camera-frame value reaches a BASE-tagged consumer.

        Both matter here more than anywhere else: the caller feeds this cloud's lowest point to
        ``resolve_support_plane`` under ``refine_from_target``, so a camera-frame slip does not
        produce a wrong number, it produces a depth where a height belongs: measured on the
        multiview gate as a support plane at -311.11 mm (the wrist-camera standoff) against a table
        at 0, which rejected 12 of 12 candidates. So: accept either shape explicitly, and say so
        when it cannot be built.
        """
        import numpy as _np  # noqa: PLC0415

        from src.robot.grasping.geometry.pointcloud import (  # noqa: PLC0415
            masked_point_cloud,
        )

        mask = getattr(seg, "mask", None)
        depth = getattr(frame, "depth_map", None)
        intrinsics = getattr(self.calculator, "camera_matrix", None)
        if mask is None or depth is None or intrinsics is None:
            return None
        try:
            cloud = masked_point_cloud(mask, _np.asarray(depth), _np.asarray(intrinsics))
        except (ValueError, TypeError):
            return None
        points_cam = getattr(cloud, "points_mm", cloud)
        points = _np.asarray(points_cam, dtype=float).reshape(-1, 3)
        if points.size == 0:
            return None
        # The cloud must have seen the support, or it must not be used to locate it.
        #
        # `resolve_support_plane` takes this cloud's lowest point as the plane. That is right when
        # the camera can see down the side of the object to whatever it rests on, and catastrophic
        # when it cannot: a top-down view returns the top face, the plane lands on top of the
        # object, and every candidate below it is rejected as under-the-table. A cloud whose z_min
        # and z_max are both 50.0 mm takes `real_cell --rehearse` from 3/3 to exit 2.
        #
        # Vertical extent is the cheap, honest test for whether the support was observed. It is
        # measured along the support normal rather than along z, so a tilted tray works the same
        # way.
        if self.support_config is not None:
            normal = _np.asarray(self.support_config.normal, dtype=float).reshape(3)
            length = float(_np.linalg.norm(normal))
            if length > 1e-12:
                along = points @ (normal / length)
                if float(along.max() - along.min()) < _MIN_CLOUD_EXTENT_FOR_SUPPORT_MM:
                    _LOG.debug(
                        "single-view target cloud spans %.1f mm along the support normal (< %.1f); "
                        "it has not observed the support, so the declared height stands",
                        float(along.max() - along.min()), _MIN_CLOUD_EXTENT_FOR_SUPPORT_MM)
                    return None
        matrix = getattr(camera_to_base, "to_matrix", None)
        transform = _np.asarray(matrix() if callable(matrix) else camera_to_base, dtype=float)
        if transform.shape != (4, 4):
            _LOG.warning(
                "support refinement got a CAMERA->BASE of shape %s, not (4, 4); the target cloud is "
                "NOT refining the support plane this attempt", transform.shape)
            return None
        return points @ transform[:3, :3].T + transform[:3, 3]

    def run(self) -> PickReport:
        """Execute the loop and return a :class:`PickReport`.

        Thin wrapper around :meth:`_execute_pick` that finalises the
        sequencing / perception / recovery shadow telemetry in a
        ``try/finally`` block so every terminal branch surfaces the
        typed carriers for the post-pick reader.
        """

        # Per-pick reset of the multi-view fusion grid: voxel evidence accumulates within a pick
        # (across its rescan/relocate views) but must not leak across independent picks; stale
        # cross-pick evidence starves candidates. Within-pick accumulation is preserved: the clear
        # happens before the loop, not during it. No-op when fusion is unwired/shadow-off (the
        # default).
        if self.scene_fusion is not None:
            self.scene_fusion.clear()
        emit(self.on_progress, PickStage.PICK_STARTED, attempt_total=self.max_attempts)
        report: PickReport | None = None
        try:
            report = self._execute_pick()
            return report
        finally:
            self._finalize_sequencing_shadow()
            self._finalize_perception_shadow()
            self._finalize_recovery_shadow()
            # In the `finally` so a raising pick still closes its event stream: a console that only
            # ever saw PICK_STARTED would show a spinner forever, which reads as "the arm is still
            # moving", the single most misleading thing a robot UI can display.
            emit(
                self.on_progress, PickStage.PICK_FINISHED,
                outcome=str(report.outcome) if report is not None else "raised",
            )

    def _execute_pick(self) -> PickReport:
        """Run the deterministic pick loop."""
        attempts: list[PickAttempt] = []
        # The shadow finaliser reads the same list via this stable reference.
        self._sequencing_current_attempts = attempts
        self._viewpoints_visited = []
        # Reset the per-pick commit / shadow accumulators (see the field
        # declarations for what each slot carries).
        self._commit_reobserve_count = 0
        self._pending_camera_to_base = None
        self._ranking_telemetry = None
        self._candidate_log = None
        self._behavior_candidate_id = None
        self._sequencing_states = []
        self._sequencing_telemetry = None
        self._perception_telemetry = None
        self._recovery_telemetry = None
        # ``last_commit_decision`` carries the most recent gate
        # evaluation so terminal branches that did not reach the commit
        # gate still report ``None`` (e.g. NO_PERCEPTION) while branches
        # that did reach it surface the decision.
        last_commit_decision: CommitDecision | None = None
        for attempt_index in range(self.max_attempts):
            # "Stop" means do not begin another attempt. Asked here, before perception, so a cancel
            # costs at most the attempt already running; it never interrupts a motion in flight,
            # which this loop has no safe way to do and the physical stop button does.
            if self.should_cancel is not None and self.should_cancel():
                emit(self.on_progress, PickStage.CANCELLED, attempt=attempt_index)
                return PickReport(
                    outcome=PickOutcome.CANCELLED,
                    attempts=tuple(attempts),
                    viewpoints_visited=tuple(self._viewpoints_visited),
                    fusion_telemetry=self._fusion_telemetry_or_none(),
                    commit_decision=last_commit_decision,
                )
            emit(
                self.on_progress, PickStage.ATTEMPT_STARTED,
                attempt=attempt_index, attempt_total=self.max_attempts,
            )
            frame = self.perception.acquire()
            # Read after acquire, so it describes the frame that was just grounded rather than the
            # previous one. Duck-typed and optional: a source with no routed backend returns None and
            # the event is exactly the one this loop has always emitted.
            _route = getattr(self.perception, "last_route", None)
            emit(
                self.on_progress, PickStage.PERCEIVED,
                attempt=attempt_index, segmentation_count=len(frame.segmentations),
                route=_route[0] if _route else None,
                route_reason=_route[1] if _route else None,
            )
            # ---- shadow-only multi-view fusion ingest ----
            # No-op when the orchestrator was built without
            # ``scene_fusion``. When wired the substrate enforces its
            # own frame / intrinsic contract and never raises; failures
            # are absorbed and surfaced via fusion telemetry only. This
            # call must come before any decision logic so the substrate
            # sees every acquired frame even when segmentations are
            # empty or downstream steps decide to rescan / relocate.
            self._maybe_ingest_scene_fusion(frame, attempt_index)
            if not frame.segmentations:
                attempts.append(
                    PickAttempt(
                        attempt_index=attempt_index,
                        target_index=None,
                        reasons=(GraspFailureReason.EMPTY_MASK,),
                        score=0.0,
                        action="exhausted",
                    )
                )
                return PickReport(
                    outcome=PickOutcome.NO_PERCEPTION,
                    attempts=tuple(attempts),
                    viewpoints_visited=tuple(self._viewpoints_visited),
                    fusion_telemetry=self._fusion_telemetry_or_none(),
                    commit_decision=last_commit_decision,
                )

            best_result, target_index, ordering_decision = (
                self._best_result_over_segmentations(frame)
            )
            if best_result is None or not best_result.is_success:
                action = self._decide_action(
                    best_result.reasons if best_result is not None else ()
                )
                reasons = best_result.reasons if best_result is not None else ()
                emit(
                    self.on_progress, PickStage.NO_CANDIDATE,
                    attempt=attempt_index, target_index=target_index,
                    reasons=tuple(str(r) for r in reasons), action=action,
                    # A refusal an operator can act on carries what was seen, not just what failed.
                    # Only the label gate populates these keys, so every other rejection path emits
                    # exactly the event it emitted before.
                    **(
                        {"extra": {
                            "target_label": best_result.telemetry["target_label"],
                            "labels_seen": list(best_result.telemetry["labels_seen"]),
                        }}
                        if best_result is not None
                        and GraspFailureReason.TARGET_LABEL_NOT_FOUND in reasons
                        else {}
                    ),
                )
                attempts.append(
                    PickAttempt(
                        attempt_index=attempt_index,
                        target_index=target_index,
                        reasons=reasons,
                        score=0.0,
                        action=action,
                        ordering_decision=ordering_decision,
                    )
                )
                if action == "rescan":
                    continue
                if action == "relocate" and self.viewpoint_planner is not None:
                    if self._relocate_camera():
                        continue
                    return PickReport(
                        outcome=PickOutcome.RELOCATED_EXHAUSTED,
                        attempts=tuple(attempts),
                        viewpoints_visited=tuple(self._viewpoints_visited),
                        fusion_telemetry=self._fusion_telemetry_or_none(),
                        commit_decision=last_commit_decision,
                    )
                # No further recovery available.
                return PickReport(
                    outcome=(
                        PickOutcome.RELOCATED_EXHAUSTED
                        if action == "relocate"
                        else PickOutcome.RESCANNED_EXHAUSTED
                    ),
                    attempts=tuple(attempts),
                    viewpoints_visited=tuple(self._viewpoints_visited),
                    fusion_telemetry=self._fusion_telemetry_or_none(),
                    commit_decision=last_commit_decision,
                )

            emit(
                self.on_progress, PickStage.RANKED,
                attempt=attempt_index, target_index=target_index,
                candidate_count=len(best_result.candidates),
                score=float(best_result.top_score),
            )
            # Success path: execute approach, grasp, retreat via the policy.
            # Annotate the chosen result's candidates with shadow
            # success probabilities before execution so the telemetry
            # is available regardless of whether the motion succeeds.
            # ``annotate_grasp_points_with_shadow_probability`` is a
            # no-op when ``shadow_success_context is None``, preserving
            # the legacy path byte-for-byte for callers that did not
            # opt in.
            shadow_telemetry = annotate_grasp_points_with_shadow_probability(
                best_result.candidates,
                ctx=self.shadow_success_context,
            )
            # ---- capture pre-blend snapshot ----
            # The ranker needs per-candidate features in their pre-blend
            # state (the annotator has run but the reranker has not).
            # The snapshot taken here reconstructs candidate identity
            # post-blend. Each candidate is keyed by its pre-blend index:
            # the same ``GraspPoint`` object survives the blend, so
            # post-blend ids map back via ``id(...)``.
            pre_blend_candidates = tuple(best_result.candidates)
            # Bounded probability-blend reranker. No-op when:
            #   * ``ranking_blend_config`` is None / disabled (silent;
            #     returns ``blend_telemetry=None``, legacy parity).
            #   * The orchestrator has no shadow context.
            #   * The current grasp-mode is not in the dense-only
            #     allow-list (schema-locked).
            #   * The lifecycle phase is ``"shadow"``.
            #   * Any candidate is missing a shadow probability (the
            #     annotator must have populated each candidate first).
            # This never mutates ``GraspPoint.score``: only metadata and
            # the order of the returned tuple change. When the rerank
            # fires the top-1 candidate may change, and
            # ``shadow_telemetry`` is then recomputed from the new winner
            # so :attr:`PickReport.shadow_success_telemetry` always
            # describes the candidate that will be executed.
            blend_candidates, blend_telemetry = maybe_blend_rerank_candidates(
                best_result.candidates,
                ctx=self.shadow_success_context,
                config=self.ranking_blend_config,
            )
            # Apply the blend result (swap candidates + re-derive shadow telemetry on a top-1
            # change) via the shared helper.
            best_result, shadow_telemetry = self._apply_rerank_result(
                best_result, shadow_telemetry, blend_candidates, blend_telemetry
            )
            # ---- per-candidate uncertainty re-rank ----
            # Runs after the blend (re-sorts the blended order) and before the ranking shadow (so
            # the shadow observes the final executed order) and the commit gate (so the gate vets the
            # pose that actually executes). Subtractive: ``adjusted = score - weight*corridor_risk``.
            # Silent no-op unless wired + dense + non-shadow + weight>0 + every candidate carries a
            # corridor-risk signal. On a top-1 change ``shadow_telemetry`` is re-derived from the new
            # winner, as the blend block above.
            rerank_candidates, uncertainty_rerank_telemetry = (
                maybe_uncertainty_rerank_candidates(
                    best_result.candidates,
                    ctx=self.shadow_success_context,
                    config=self.uncertainty_rerank_config,
                )
            )
            # Apply the uncertainty rerank result via the same shared helper.
            best_result, shadow_telemetry = self._apply_rerank_result(
                best_result,
                shadow_telemetry,
                rerank_candidates,
                uncertainty_rerank_telemetry,
            )
            # ---- ranking-shadow producer ----
            # Silent no-op unless ``shadow_router`` and its
            # ``ranking_policy`` are both wired. Telemetry stashed on
            # ``self._ranking_telemetry`` for the caller
            # (``AutonomousGraspService``) to fold into the
            # post-pick shadow-router telemetry block. Never mutates
            # ``best_result``, ``shadow_telemetry``, or scores.
            self._maybe_run_ranking_shadow(
                pre_blend_candidates=pre_blend_candidates,
                post_blend_candidates=best_result.candidates,
                attempt_index=attempt_index,
            )
            # ---- mandatory commit gate ----
            # Evaluate the multi-view commit gate against the winner
            # after any rerank and before :meth:`_execute`. The
            # helper short-circuits to ``allowed=True`` whenever any
            # of the byte-identical-skip conditions hold (no policy,
            # disabled, no fusion, mode not in apply set, no winning
            # candidate, candidate not in BASE frame), so legacy
            # callers see no behavior change. A refusal with budget
            # left records the refused decision, logs a typed attempt,
            # and continues the outer loop into another perception
            # capture. A refusal with the budget exhausted surfaces
            # :attr:`PickOutcome.NO_COMMIT_INSUFFICIENT_FUSION`.
            commit_decision = self._evaluate_commit_gate(best_result)
            last_commit_decision = commit_decision
            if not commit_decision.allowed:
                if (
                    self.commit_policy is not None
                    and self._commit_reobserve_count
                    < self.commit_policy.max_reobserve_attempts
                ):
                    self._commit_reobserve_count += 1
                    attempts.append(
                        PickAttempt(
                            attempt_index=attempt_index,
                            target_index=target_index,
                            reasons=best_result.reasons,
                            score=best_result.top_score,
                            action="commit_refused_reobserve",
                            ordering_decision=ordering_decision,
                        )
                    )
                    # Relocate to a diverse viewpoint before the next capture so the fused
                    # corridor evidence grows across views instead of re-ingesting the same frame (a
                    # static re-acquire only inflates views_accepted without improving corridor
                    # coverage). Reuses the recovery _relocate_camera, which advances the planner
                    # history and moves the arm; for the sim arm move_to routes through move(),
                    # which runs SafetyPreflight. No-op (byte-identical) when no viewpoint_planner
                    # is wired or the planner is exhausted: the loop re-acquires at the same pose.
                    if self.viewpoint_planner is not None:
                        self._relocate_camera()
                    continue
                # Budget exhausted (or no policy with allowed=False,
                # which the helper's skip rules make impossible; the
                # branch is defensive).
                attempts.append(
                    PickAttempt(
                        attempt_index=attempt_index,
                        target_index=target_index,
                        reasons=best_result.reasons,
                        score=best_result.top_score,
                        action="commit_refused_exhausted",
                        ordering_decision=ordering_decision,
                    )
                )
                return PickReport(
                    outcome=PickOutcome.NO_COMMIT_INSUFFICIENT_FUSION,
                    attempts=tuple(attempts),
                    viewpoints_visited=tuple(self._viewpoints_visited),
                    shadow_success_telemetry=shadow_telemetry,
                    ranking_blend_telemetry=blend_telemetry,
                    uncertainty_rerank_telemetry=uncertainty_rerank_telemetry,
                    fusion_telemetry=self._fusion_telemetry_or_none(),
                    commit_decision=commit_decision,
                )
            # The last event before the arm moves. Everything after this is motion, so a console that
            # stops updating here is telling an operator the truth: the stack is done deciding.
            emit(
                self.on_progress, PickStage.EXECUTING,
                attempt=attempt_index, target_index=target_index,
                score=float(best_result.top_score),
                position_mm=_grasp_position_mm(best_result),
            )
            policy_report = self._execute(best_result)
            self._last_policy_report = policy_report
            # Map the PolicyOutcome to its action label and PickOutcome via _map_policy_outcome.
            attempt_action, pick_outcome = self._map_policy_outcome(
                policy_report.outcome
            )
            # Why the motion failed can be the cell, not the grasp. Asked only here, on a path that
            # has already failed, so a healthy pick pays nothing. When the controller itself cannot
            # move, say so instead of reporting a failed execution: the reason is typed, the recovery
            # dispatcher maps it to no action, and the outcome stays out of the failure taxonomy.
            controller_reason = (
                self._controller_cannot_move()
                if pick_outcome is not PickOutcome.EXECUTED
                else None
            )
            if controller_reason is not None:
                _LOG.error("pick aborted: %s", controller_reason)
                attempt_action = "controller_not_operational"
                pick_outcome = PickOutcome.CONTROLLER_NOT_OPERATIONAL
            emit(
                self.on_progress, PickStage.ATTEMPT_FINISHED,
                attempt=attempt_index, action=attempt_action, outcome=str(pick_outcome),
                # Why the motion did not happen, on the wire.
                #
                # The progress event is the only thing a console sees while a run is happening, so
                # it carries the same three motion fields ``PickAttempt`` carries. Without them a
                # refusal as precise as ``workspace_rejected`` on an approach pose 33.6 mm below the
                # configured workspace floor reaches the operator as a bare ``execution_failed``,
                # and a fail-closed guard that cannot explain itself to the person at the cell is a
                # guard that gets switched off.
                #
                # Additive and empty on the healthy path: `_motion_fields` yields three ``None``s
                # for a report that carries none, and the keys are dropped below, so every existing
                # consumer sees the event it saw before.
                **{k: v for k, v in _motion_fields(policy_report).items() if v is not None},
            )
            attempts.append(
                PickAttempt(
                    attempt_index=attempt_index,
                    target_index=target_index,
                    reasons=(
                        (GraspFailureReason.CONTROLLER_NOT_OPERATIONAL,)
                        if controller_reason is not None
                        else best_result.reasons
                    ),
                    score=best_result.top_score,
                    action=attempt_action,
                    ordering_decision=ordering_decision,
                    **_motion_fields(policy_report),
                )
            )
            return PickReport(
                outcome=pick_outcome,
                attempts=tuple(attempts),
                executed_grasp=(
                    best_result if pick_outcome is PickOutcome.EXECUTED else None
                ),
                # The selected grasp's telemetry, not the executed one's. See the field's comment.
                selected_telemetry=dict(getattr(best_result, "telemetry", None) or {}),
                viewpoints_visited=tuple(self._viewpoints_visited),
                shadow_success_telemetry=shadow_telemetry,
                ranking_blend_telemetry=blend_telemetry,
                uncertainty_rerank_telemetry=uncertainty_rerank_telemetry,
                fusion_telemetry=self._fusion_telemetry_or_none(),
                commit_decision=last_commit_decision,
            )

        return PickReport(
            outcome=PickOutcome.RESCANNED_EXHAUSTED,
            attempts=tuple(attempts),
            viewpoints_visited=tuple(self._viewpoints_visited),
            fusion_telemetry=self._fusion_telemetry_or_none(),
            commit_decision=last_commit_decision,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _shadow_telemetry_from_winner(
        self, candidates: tuple[Any, ...]
    ) -> ShadowSuccessTelemetry:
        """Re-derive the shadow-success telemetry from the top-1 candidate's metadata.

        Shared by the blend and uncertainty reranks, which both re-point the telemetry at the
        executed winner when a rerank changes top-1.
        """
        new_winner_md = getattr(candidates[0], "metadata", None) or {}
        new_prob = new_winner_md.get(SHADOW_METADATA_KEY_PROBABILITY)
        ctx = self.shadow_success_context
        return ShadowSuccessTelemetry(
            predicted_success_probability=(
                float(new_prob)
                if isinstance(new_prob, (int, float)) and not isinstance(new_prob, bool)
                else None
            ),
            success_probability_model_version=(
                ctx.version_label if ctx is not None else None
            ),
            model_lifecycle_phase=(ctx.lifecycle_phase if ctx is not None else None),
        )

    def _apply_rerank_result(
        self,
        best_result: GraspResult,
        shadow_telemetry: ShadowSuccessTelemetry | None,
        candidates: tuple[Any, ...],
        telemetry: RankingBlendTelemetry | UncertaintyRerankTelemetry | None,
    ) -> tuple[GraspResult, ShadowSuccessTelemetry | None]:
        """Apply a rerank result to ``best_result``; shared by the blend and uncertainty reranks.
        When the rerank fired (``telemetry.applied``) the reranked candidates and the new top score
        are swapped in; when it also changed top-1, ``shadow_telemetry`` is re-pointed at the new
        winner so the executed candidate's probability is reported. Otherwise the inputs come back
        unchanged."""
        if telemetry is not None and telemetry.applied:
            new_top_score = float(candidates[0].score) if candidates else 0.0
            best_result = replace(
                best_result, candidates=candidates, top_score=new_top_score
            )
            if telemetry.top1_changed:
                shadow_telemetry = self._shadow_telemetry_from_winner(candidates)
        return best_result, shadow_telemetry

    def _stamp_deep_ranker(  # noqa: ANN001, ANN202
        self, result, extra_kwargs: dict, fused_scene, index: int,
    ) -> None:
        """Score this attempt's candidates with the learned ranker and stamp the result. Never acts.

        Off by default (``deep_ranker_context is None``) and then this is one attribute read, so the
        default path is byte-identical.

        Two frame checks guard two hazards that sit one line apart:

        * ``support_plane`` defaults to ``Frame.CAMERA``. Its ``offset_mm`` is a BASE height only when
          the plane says so; a camera-frame plane read as a height puts 577 mm where -7 mm belongs.
        * ``scene_points_mm`` in ``extra_kwargs`` is CAMERA-frame, so the obstacle cloud is built
          from ``fused_scene`` instead, which is BASE, because the features are BASE. The
          comprehension below reads ``fused_scene.indices``, which ``FusedSceneGeometry`` does not
          define, so ``obstacle_points_mm`` is always ``None`` and the ranker scores every candidate
          without neighbour geometry.

        Nothing here may raise: the scorer's opinion is optional, the pick is not.
        """
        context = self.deep_ranker_context
        if context is None:
            return
        try:
            from src.robot.grasping.deep.ranker.shadow import score_candidates  # noqa: PLC0415

            target = extra_kwargs.get("geometry_points_base_mm")
            support_z = 0.0
            plane = extra_kwargs.get("support_plane")
            if plane is not None:
                frame = getattr(plane, "frame", None)
                if getattr(frame, "value", frame) != "base":
                    telemetry = {"deep_ranker_scored": False,
                                 "deep_ranker_reason": f"support plane is in {frame}, not BASE"}
                    _stamp(result, telemetry)
                    return
                support_z = float(plane.offset_mm)

            obstacles = None
            if fused_scene is not None:
                others = [
                    cloud for other in getattr(fused_scene, "indices", ())
                    if other != index and (cloud := fused_scene.cloud_for(other)) is not None
                    and cloud.size
                ]
                if others:
                    import numpy as _np  # noqa: PLC0415

                    obstacles = _np.vstack(others)

            outcome = score_candidates(
                context.ranker, context.spec, list(getattr(result, "candidates", ()) or ()),
                target_points_mm=target, obstacle_points_mm=obstacles, support_z_mm=support_z,
            )
            _stamp(result, outcome.as_dict())
        except Exception:  # noqa: BLE001 (an optional observer may never cost an attempt)
            _LOG.exception("deep_ranker shadow scoring failed; the pick is unaffected")

    def _controller_cannot_move(self) -> str | None:
        """A one-line reason when the controller is unable to move, else ``None``.

        Asked only after a motion has already failed: a healthy run pays nothing, because a healthy
        run never calls this. ``get_robot_status()`` is four cheap enum reads plus one dashboard
        round trip, which is fine on a failure path and would not be fine per attempt.

        The criterion is ``not is_operational`` rather than ``is_stopped``: a controller powered off
        mid-session reports ``protective_stopped=False``, ``emergency_stopped=False`` and
        ``is_stopped=False``, correctly, since none of those flags describes a power-off, while
        being exactly as unable to move. ``is_operational`` catches it.

        Inert by construction for every arm that is not a real UR: only ``URRobotArm`` implements
        ``SupportsRobotStatus``, so the sim arm, the dummy arm and every other arm take the ``None``
        path and behave byte-identically. No flag needed.
        """
        # No lazy import here: this module imports `src.robot.core` at module level, so deferring
        # `SupportsRobotStatus` defers 0 modules and 0.0 ms. The module is loaded long before this
        # function can be called.
        if not isinstance(self.arm, SupportsRobotStatus):
            return None
        try:
            status = self.arm.get_robot_status()
        except Exception:  # noqa: BLE001 (a diagnosis must never replace the fault it explains)
            return None
        if status.is_operational:
            return None
        detail = f": {status.message}" if status.message else ""
        return (
            f"the controller cannot move: robot_mode={status.robot_mode.value}, "
            f"safety_mode={status.safety_mode.value}, protective_stop={status.protective_stopped}, "
            f"emergency_stop={status.emergency_stopped}{detail}. Nothing is wrong with the scene or "
            "the grasp, and no retry or re-perceive will change it; the cell needs a human. Clear "
            "the stop where the arm is visible, or power it back on, then run again."
        )

    def _map_policy_outcome(
        self, outcome: PolicyOutcome
    ) -> tuple[str, PickOutcome]:
        """Map a :class:`PolicyOutcome` to the pair the success path records.

        The pair is an attempt-action label and a :class:`PickOutcome`. The camera-frame and
        approach-blocked outcomes keep their own labels; any outcome not enumerated here,
        MOTION_FAILED among them, falls through to EXECUTION_FAILED.
        """
        if outcome is PolicyOutcome.EXECUTED:
            return "executed", PickOutcome.EXECUTED
        if outcome is PolicyOutcome.OBJECT_NOT_DETECTED:
            return "object_not_detected", PickOutcome.OBJECT_NOT_DETECTED
        if outcome is PolicyOutcome.CAMERA_FRAME_REJECTED:
            # The policy refused a camera-frame grasp; surface distinctly.
            return "camera_frame_rejected", PickOutcome.CAMERA_FRAME_REJECTED
        if outcome is PolicyOutcome.APPROACH_PATH_BLOCKED:
            # Every candidate's approach/retreat sweep hit a neighbour: refused, no motion.
            return "approach_path_blocked", PickOutcome.APPROACH_PATH_BLOCKED
        return "execution_failed", PickOutcome.EXECUTION_FAILED

    def _maybe_run_ranking_shadow(
        self,
        *,
        pre_blend_candidates: tuple,
        post_blend_candidates: tuple,
        attempt_index: int,
    ) -> None:
        """Invoke the ranking shadow router.

        Strict no-op when:
          * :attr:`shadow_router` is :data:`None`.
          * The router has no :attr:`ranking_policy`.
          * The pre-blend candidate set is empty.

        Builds 15-key feature rows per candidate from the ``shadow`` +
        ``rl_state_features`` metadata blocks plus the three
        ``geometric_score`` / ``feasibility_score`` /
        ``shadow_predicted_success_probability`` signals, and stashes the
        typed :class:`RankingShadowTelemetry` on
        :attr:`_ranking_telemetry`. Any exception is converted to a
        fallback row by the router itself, and anything raised while
        the rows are built or the router runs is caught inside
        :meth:`_ShadowTelemetryAggregator.run_ranking_shadow`, which
        returns ``(None, None, None)`` and leaves the three slots
        unset. This helper adds no handler of its own.
        """

        router = self.shadow_router
        if router is None or getattr(router, "ranking_policy", None) is None:
            return
        if not pre_blend_candidates:
            return
        # The wired path delegates to the aggregator; assign all 3 slots from the return. The
        # aggregator returns (None, None, None) on its broad except. The no-router / no-candidates
        # guards above return early without writing any slot, so a direct call leaves all three at
        # their construction-default None.
        (
            self._ranking_telemetry,
            self._candidate_log,
            self._behavior_candidate_id,
        ) = self._shadow.run_ranking_shadow(
            pre_blend_candidates=pre_blend_candidates,
            post_blend_candidates=post_blend_candidates,
            attempt_index=attempt_index,
        )

    # ------------------------------------------------------------------
    # Sequencing-shadow helpers
    # ------------------------------------------------------------------

    def _finalize_sequencing_shadow(self) -> None:
        """Delegate to the aggregator's sequencing shadow (called from :meth:`run`'s ``finally``).
        Assigns both _sequencing_telemetry and _sequencing_states from the return; on the unwired
        path the aggregator returns (None, None), so _sequencing_states keeps its per-pick reset
        value."""
        telemetry, states = self._shadow.finalize_sequencing(attempts=self._sequencing_current_attempts)
        self._sequencing_telemetry = telemetry
        if states is not None:
            self._sequencing_states = states

    def _finalize_perception_shadow(self) -> None:
        """Delegate to the aggregator's perception-budget shadow.

        Called from :meth:`run`'s ``finally``. Assigns the slot from the return; the aggregator
        returns None on the unwired and except paths.
        """
        self._perception_telemetry = self._shadow.finalize_perception(
            attempts=self._sequencing_current_attempts
        )

    def _finalize_recovery_shadow(self) -> None:
        """Delegate to the aggregator's recovery shadow.

        Called from :meth:`run`'s ``finally``. Assigns the slot from the return; the aggregator
        returns None on the unwired, success and except paths.
        """
        self._recovery_telemetry = self._shadow.finalize_recovery(
            attempts=self._sequencing_current_attempts
        )

    # ------------------------------------------------------------------

    def _maybe_ingest_scene_fusion(
        self, frame: PerceptionFrame, attempt_index: int
    ) -> None:
        """Shadow-only multi-view fusion ingest seam.

        Strict no-op when ``scene_fusion`` is unset or the substrate's
        config is disabled. When wired, resolves ``T_cam_to_base`` via
        the typed :func:`resolve_or_none` helper so resolver failures
        (no resolver, raising resolver, wrong-frame transform) cannot
        propagate into the pick loop. Successful resolution is handed
        to :meth:`SceneFusion.ingest`, which enforces its own frame /
        intrinsic contract. Both layers absorb their failure modes
        into the substrate's typed refuse path; this method never
        raises and never alters the pick loop's control flow.
        """

        fusion = self.scene_fusion
        if fusion is None or not fusion.config.enabled:
            return
        # Lazy import to avoid pulling the fusion surface into modules
        # that touch ``pick_loop`` only for the legacy API.
        from src.robot.grasping.motion.frame_resolver import (
            resolve_or_none,
        )

        resolved = resolve_or_none(
            self.frame_resolver, frame, arm=self.arm
        )
        # ``resolve_or_none`` returns either a ``Transform`` (success)
        # or a ``FrameResolutionFailure`` carrier (refuse). A refuse
        # still counts an attempted ingest: ``ingest`` is called with a
        # deliberately bad frame so the substrate's own refuse counter
        # records the failure, which keeps the telemetry consistent
        # with "ingest was attempted but rejected".
        if not isinstance(resolved, Transform):
            # Synthesize an attempt-refuse so telemetry reflects it.
            # The transform passed is an invalid TOOL->BASE, which the
            # substrate refuses with REFUSE_BAD_FRAME, keeping the
            # counter convention uniform.
            invalid = Transform.identity(
                from_frame=Frame.TOOL, to_frame=Frame.BASE
            )
            fusion.ingest(
                depth_map=frame.depth_map,
                intrinsics=frame.intrinsics,
                t_cam_to_base=invalid,  # type: ignore[arg-type]
                timestamp_ns=self._fusion_timestamp_ns(
                    frame, attempt_index
                ),
            )
            return
        fusion.ingest(
            depth_map=frame.depth_map,
            intrinsics=frame.intrinsics,
            t_cam_to_base=resolved,
            timestamp_ns=self._fusion_timestamp_ns(frame, attempt_index),
        )

    @staticmethod
    def _fusion_timestamp_ns(
        frame: PerceptionFrame, attempt_index: int
    ) -> int:
        """Deterministic timestamp source for fusion ingest.

        Prefers ``frame.timestamp`` (interpreted as seconds) so the
        age-based eviction is anchored to real perception time. An
        unstamped frame falls back to ``attempt_index * 1_000_000_000``,
        one synthetic second per attempt, which is monotonic,
        byte-deterministic across runs, and keeps the default 20 s age
        window comfortably wide for typical loop lengths.
        """

        ts = frame.timestamp
        if ts is None or not isinstance(ts, (int, float)) or isinstance(ts, bool):
            return int(attempt_index) * 1_000_000_000
        try:
            return int(float(ts) * 1e9)
        except (TypeError, ValueError, OverflowError):
            return int(attempt_index) * 1_000_000_000

    def _fusion_telemetry_or_none(self) -> "FusionTelemetry | None":
        """Return the substrate's current telemetry or ``None``."""

        fusion = self.scene_fusion
        if fusion is None:
            return None
        return fusion.telemetry()

    def _evaluate_commit_gate(
        self, best_result: GraspResult
    ) -> CommitDecision:
        """Commit-gate evaluation.

        Returns a :class:`CommitDecision` for the winning candidate of
        ``best_result``. The decision is ``allowed=True`` whenever
        any skip condition holds (no policy, disabled policy, no
        fusion substrate, empty or non-matching mode label, no
        winning candidate, candidate not in BASE frame) so legacy
        callers see byte-identical behavior. The no-candidate branch
        reports the reason string ``skipped_no_policy``.

        When the gate runs, it queries
        :meth:`SceneFusion.corridor_evidence` for a capsule of
        :attr:`CommitPolicy.corridor_length_mm` /
        :attr:`CommitPolicy.corridor_radius_mm` centered on the
        winning candidate's position along its approach. The gate
        requires both:

        * ``views_accepted >= min_views_accepted``
        * ``hit_fraction >= min_corridor_hit_fraction``

        Strictly read-only with respect to fusion state; never
        mutates any substrate accumulators or any candidate field.
        """

        policy = self.commit_policy
        fusion = self.scene_fusion
        candidate = best_result.candidates[0] if best_result.candidates else None
        # Compose a zero-evidence template for skip paths so callers
        # can read every field unconditionally.
        def zero(reason):
            return CommitDecision(
                allowed=True,
                reason=reason,
                views_accepted=(
                    fusion.telemetry().views_accepted if fusion is not None else 0
                ),
                min_views_required=(
                    int(policy.min_views_accepted) if policy is not None else 0
                ),
                corridor_hit_fraction=0.0,
                min_hit_fraction_required=(
                    float(policy.min_corridor_hit_fraction)
                    if policy is not None
                    else 0.0
                ),
                corridor_queried_voxels=0,
                corridor_hit_voxels=0,
                reobserve_attempts_used=self._commit_reobserve_count,
            )

        if policy is None:
            return zero(_COMMIT_REASON_SKIPPED_NO_POLICY)
        if not policy.enabled:
            return zero(_COMMIT_REASON_SKIPPED_DISABLED)
        if fusion is None or not fusion.config.enabled:
            return zero(_COMMIT_REASON_SKIPPED_NO_FUSION)
        if not self.mode_label or self.mode_label not in policy.apply_modes:
            return zero(_COMMIT_REASON_SKIPPED_MODE)
        if candidate is None:
            # Defensive: a success path with no candidates should never
            # reach the gate; if it does, the gate is skipped.
            return zero(_COMMIT_REASON_SKIPPED_NO_POLICY)
        # The corridor query is BASE-frame; refuse to gate a
        # camera-frame candidate (would silently corrupt the math).
        # In real configurations this cannot happen because enabling
        # multi-view fusion requires a wired ``frame_resolver`` which
        # in turn delivers BASE-frame candidates from the calculator.
        if candidate.frame is not GraspFrame.BASE:
            return zero(_COMMIT_REASON_SKIPPED_CAMERA_FRAME)

        evidence = fusion.corridor_evidence(
            position_mm=candidate.position,
            approach=candidate.approach,
            length_mm=float(policy.corridor_length_mm),
            radius_mm=float(policy.corridor_radius_mm),
        )

        min_views = int(policy.min_views_accepted)
        min_hit = float(policy.min_corridor_hit_fraction)
        views_ok = evidence.views_accepted >= min_views
        hit_ok = evidence.hit_fraction >= min_hit

        if views_ok and hit_ok:
            return CommitDecision(
                allowed=True,
                reason=_COMMIT_REASON_OK,
                views_accepted=evidence.views_accepted,
                min_views_required=min_views,
                corridor_hit_fraction=evidence.hit_fraction,
                min_hit_fraction_required=min_hit,
                corridor_queried_voxels=evidence.queried_voxels,
                corridor_hit_voxels=evidence.hit_voxels,
                reobserve_attempts_used=self._commit_reobserve_count,
            )

        # When both fail the views_below_min reason is surfaced first
        # (the more fundamental signal: not enough viewpoints means
        # the corridor evidence is unreliable in the first place).
        refuse_reason = (
            _COMMIT_REASON_VIEWS_BELOW_MIN
            if not views_ok
            else _COMMIT_REASON_HIT_FRACTION_BELOW_MIN
        )
        return CommitDecision(
            allowed=False,
            reason=refuse_reason,
            views_accepted=evidence.views_accepted,
            min_views_required=min_views,
            corridor_hit_fraction=evidence.hit_fraction,
            min_hit_fraction_required=min_hit,
            corridor_queried_voxels=evidence.queried_voxels,
            corridor_hit_voxels=evidence.hit_voxels,
            reobserve_attempts_used=self._commit_reobserve_count,
        )

    def _external_scene_points_camera_frame(
        self, camera_to_base: "Transform | None"
    ) -> "np.ndarray | None":
        """The fused multi-view scene cloud expressed in the captured frame's CAMERA frame.

        The cloud is :attr:`external_scene_points_base_mm`, which is BASE, and the CAMERA frame is
        the frame the collision filter plans in. Returns ``None`` when no cloud is set or no
        resolver gave a transform. ``base->camera = inv(camera_to_base)``.
        """
        pts = self.external_scene_points_base_mm
        if pts is None or camera_to_base is None:
            return None
        return self._base_points_to_camera(pts, camera_to_base)

    @staticmethod
    def _base_points_to_camera(
        points_base_mm: "np.ndarray", camera_to_base: "Transform"
    ) -> "np.ndarray | None":
        """BASE mm to the captured frame's CAMERA mm, the frame the collision filter plans in."""

        base = np.asarray(points_base_mm, dtype=np.float64).reshape(-1, 3)
        if base.size == 0:
            return None
        m = np.asarray(camera_to_base.to_matrix(), dtype=np.float64)  # CAMERA -> BASE
        inv = np.linalg.inv(m)  # BASE -> CAMERA
        return (inv[:3, :3] @ base.T).T + inv[:3, 3]

    def _container_wall_points_base_mm(self) -> "np.ndarray | None":
        """The declared container walls as BASE points, built once and reused.

        Without them the candidate filter has no wall geometry at all: a target cloud, the
        neighbours' clouds and a support plane, and a plane is a floor. Replaying the reference
        verdict on the rank-0 winners that lose the ranking gap on ``finger_collision`` puts 60.9 %
        of them against a bin wall rather than an object (in the ``bin`` family, 14 of 14).
        Perception cannot supply it: no detector segments a wall as an instance. So the wall is
        declared, and this turns the declaration into the points every existing check already reads.

        Cached because it depends only on config: rebuilding a fixed bin every pick would be pure
        waste in the one path that runs on every attempt.
        """

        support = self.support_config
        container = getattr(support, "container", None) if support is not None else None
        if container is None or not bool(getattr(container, "wall_collision_enabled", False)):
            return None
        low = getattr(container, "interior_min_mm", None)
        high = getattr(container, "interior_max_mm", None)
        if low is None or high is None:
            # The schema refuses this combination at load time; a hand-built config can still reach
            # here, and a silently wall-free cell is exactly what this seam exists to prevent.
            _LOG.warning(
                "grasping.support.container.wall_collision_enabled is set but the interior box is "
                "not declared; the walls are not collision geometry for this attempt")
            return None
        key = (tuple(low), tuple(high), float(container.wall_thickness_mm),
               float(container.wall_sample_mm))
        if self._container_wall_cache is None or self._container_wall_cache[0] != key:
            from src.robot.grasping.collision import (  # noqa: PLC0415
                container_wall_points_base_mm,
            )

            points = container_wall_points_base_mm(
                low, high,
                thickness_mm=float(container.wall_thickness_mm),
                sample_mm=float(container.wall_sample_mm),
            )
            self._container_wall_cache = (key, points)
        return self._container_wall_cache[1]

    def _fused_scene(
        self,
        frame: PerceptionFrame,
        camera_to_base: "Transform | None",
    ) -> "FusedSceneGeometry | None":
        """What the whole rig sees, per segmentation of ``frame``, or ``None`` to stay single-view.

        The one place the multi-camera policy lives, because it is the one place that holds both the
        config and the rig. Everything geometric is delegated to
        :mod:`~src.robot.grasping.multiview.scene_geometry`.

        A cell configured for three cameras that quietly runs on one (a dropped trigger, a dirty
        lens, a calibration artifact that stopped loading) looks exactly like a healthy cell from
        the outside, just worse at grasping. So every stand-down reason is either a warning or a
        refusal, never silence, and the telemetry records which cameras actually contributed rather
        than which were configured.
        """

        config = self.fusion_geometry_config
        if config is None or not bool(config.enabled):
            return None
        if self.multi_camera_perception is None or camera_to_base is None:
            # Enabled but unbuildable. This is a deployment error, not a runtime hiccup, so it is
            # said plainly every pick rather than folded into the missing-camera policy below.
            _LOG.warning(
                "grasping.fusion.geometry is enabled but %s; the cell is running single-view",
                "no multi-camera perception source is wired"
                if self.multi_camera_perception is None
                else "no CAMERA->BASE transform is available",
            )
            return None

        from src.robot.grasping.multiview.association import AssociationMetric
        from src.robot.grasping.multiview.scene_geometry import (
            ObservedView,
            fuse_scene_geometry,
        )

        resolvers = self.camera_frame_resolvers or {}
        observations = tuple(self.multi_camera_perception.acquire_all())
        delivered = {observation.camera_id for observation in observations}
        # The configured list first, the resolver map only as a fallback for a caller that wired
        # this by hand. An expectation derived from the resolvers that built cannot notice a camera
        # whose resolver never built, so that camera is one nobody waits for.
        expected = set(self.configured_camera_ids) or set(resolvers)
        missing = sorted(expected - delivered)
        if missing:
            if str(config.on_camera_unavailable) == "refuse":
                raise RuntimeError(
                    f"grasping.fusion.geometry: configured camera(s) {missing} delivered no frame "
                    "and on_camera_unavailable='refuse'"
                )
            _LOG.warning(
                "grasping.fusion.geometry: camera(s) %s delivered no frame; continuing with %s",
                missing,
                sorted(delivered & expected) or "no other camera (single-view)",
            )

        views: list[ObservedView] = []
        for observation in observations:
            resolver = resolvers.get(observation.camera_id)
            if resolver is None:
                # An id with no calibration entry cannot be placed in BASE. Guessing a default
                # extrinsic would fuse a real surface into the wrong place, which is worse than
                # not fusing it at all.
                _LOG.warning(
                    "grasping.fusion.geometry: camera %r has no entry in fusion.cameras; its view "
                    "is dropped (no CAMERA->BASE calibration)",
                    observation.camera_id,
                )
                continue
            other_to_base = resolver.camera_to_base_for_frame(observation.frame, arm=self.arm)
            if other_to_base is None:
                _LOG.warning(
                    "grasping.fusion.geometry: camera %r resolved no CAMERA->BASE this frame; its "
                    "view is dropped",
                    observation.camera_id,
                )
                continue
            masks = tuple(
                mask
                for mask in (getattr(seg, "mask", None) for seg in observation.frame.segmentations)
                if mask is not None
            )
            if not masks:
                continue
            views.append(
                ObservedView(
                    name=observation.camera_id,
                    masks=masks,
                    depth_map=observation.frame.depth_map,
                    intrinsics=observation.frame.intrinsics,
                    camera_to_base=np.asarray(other_to_base.to_matrix(), dtype=np.float64),
                )
            )

        primary_masks = tuple(
            mask
            for mask in (getattr(seg, "mask", None) for seg in frame.segmentations)
            if mask is not None
        )
        if not views or not primary_masks:
            self._fusion_geometry_telemetry = {"fused_views_used": 0, "fused_objects": 0}
            return None

        with_neighbours = bool(getattr(config, "neighbour_scene_enabled", False))
        fused = fuse_scene_geometry(
            primary_masks,
            frame.depth_map,
            frame.intrinsics,
            np.asarray(camera_to_base.to_matrix(), dtype=np.float64),
            views,
            metric=AssociationMetric(str(config.metric)),
            min_score=float(config.min_score),
            neighbour_mm=float(config.neighbour_mm),
            max_centroid_mm=float(config.max_centroid_mm),
            with_neighbours=with_neighbours,
            neighbour_voxel_mm=float(getattr(config, "neighbour_voxel_mm", 8.0)),
            # Scoring-only decimation. `getattr` keeps a config object without the field on the
            # untouched path, and 0.0 is off, so this line changes nothing anywhere it is not set.
            score_voxel_mm=float(getattr(config, "score_voxel_mm", 0.0)),
        )
        # Stamp what ran, not what was configured: the same rule the geometry stage follows.
        self._fusion_geometry_telemetry = {
            "fused_views_used": len(fused.views_used),
            "fused_views": list(fused.views_used),
            "fused_objects": fused.objects_fused,
            "fused_objects_total": len(fused.clouds_base_mm),
            # How well the cameras agreed, not just how many answered: the mean association score
            # over the pairs that won their assignment. The telemetry catalog declares
            # `fusion_evidence_quality` as a unit interval without saying what it means; this is the
            # association's own confidence, the only quantity in the fusion that is one. Bounded
            # [0, 1] for every metric (overlap is a fraction, centroid is clipped, box-IoU is a
            # ratio), so the catalog's validator holds whichever is configured.
            "fused_evidence_quality": _mean_winning_association_score(fused),
        }
        if with_neighbours:
            self._fusion_geometry_telemetry["fused_neighbour_points"] = fused.neighbour_points
        return fused

    def _best_result_over_segmentations(
        self, frame: PerceptionFrame
    ) -> "tuple[GraspResult | None, int | None, OrderingDecision | None]":
        """Thin wrapper: records the ranking latency span, then delegates.

        A wrapper rather than a span inside the body because this is the one call every pick path
        makes, open-loop, decision and closed-loop alike, so this single site covers all three.
        """
        tracker = self.latency_tracker
        if tracker is None:
            return self._best_result_over_segmentations_inner(frame)
        with tracker.span(LatencyStage.RANKING):
            return self._best_result_over_segmentations_inner(frame)

    def _best_result_over_segmentations_inner(
        self,
        frame: PerceptionFrame,
    ) -> tuple[GraspResult | None, int | None, OrderingDecision | None]:
        """Compute a grasp for each segmentation; return the best success.

        Falls back to the last non-success result (with its reasons)
        when no segmentation yields candidates, so the caller can route
        on those reasons.

        When :attr:`frame_resolver` is wired, the resolved
        ``T_cam_to_base`` for the captured frame is forwarded to the
        calculator so the candidates come back in base frame. The
        resolver is invoked once per ``run`` iteration, not once per
        segmentation, because all segmentations in a single
        :class:`PerceptionFrame` were captured at the same TCP pose.

        When :attr:`target_ordering` is set and enabled the
        per-segmentation winners are passed through
        :func:`select_target` and the chosen candidate plus the
        :class:`OrderingDecision` are returned. Otherwise the legacy
        ``argmax(top_score)`` path runs verbatim and the returned
        decision is ``None``.
        """
        camera_to_base: Transform | None = None
        if self.frame_resolver is not None:
            camera_to_base = self.frame_resolver.camera_to_base_for_frame(
                frame, arm=self.arm
            )
        # Stash this frame's CAMERA->BASE so _execute can map the camera-frame neighbour cloud
        # (on result.metadata) into the candidate (BASE) frame before the swept-volume validation.
        self._pending_camera_to_base = camera_to_base
        # Transform the fused multi-view scene cloud (BASE) into this frame's CAMERA frame once
        # (the same camera_to_base for every segmentation of the frame). None when not supplied.
        external_scene_cam = self._external_scene_points_camera_frame(camera_to_base)
        # Observe the rest of the rig once per frame and fuse every candidate object's surface.
        # ``None`` (default, or any stand-down) leaves the single-view path below untouched.
        fused_scene = self._fused_scene(frame, camera_to_base)
        # The declared container walls, in this frame's CAMERA frame. Frame-level like the cloud
        # above (the bin does not move between segmentations), and additive below rather than an
        # alternative: walls and neighbours are different obstacles and a cell that has both
        # declared must get both.
        wall_points_base = self._container_wall_points_base_mm()
        wall_cam = (
            self._base_points_to_camera(wall_points_base, camera_to_base)
            if wall_points_base is not None and camera_to_base is not None
            else None
        )
        if wall_points_base is not None and camera_to_base is None:
            _LOG.warning(
                "container walls are configured but no CAMERA->BASE transform is available; the "
                "wall-collision check is not running for this attempt")
        other_masks = [getattr(seg, "mask", None) for seg in frame.segmentations]
        best_success: GraspResult | None = None
        best_index: int | None = None
        last_failure: GraspResult | None = None
        last_failure_index: int | None = None
        # Collect successful per-segmentation results so the target
        # selector can rank them after all segmentations have been
        # scored. Both arms of the conditional below allocate the
        # list; what ``ordering_active`` gates is the append, so with
        # ordering off the list stays empty and the selector below
        # short-circuits.
        ordering_active = (
            self.target_ordering is not None and self.target_ordering.enabled
        )
        successes: list[tuple[int, GraspResult]] = [] if ordering_active else []  # type: ignore[assignment]
        # How many segmentations the hard label gate below let through. Zero of them, with a label
        # set, is a nameable failure rather than an empty reason tuple.
        label_matches = 0
        for idx, seg in enumerate(frame.segmentations):
            # When a hard label-target is set, only the matching segmentation is an executable
            # target. The non-target segs are skipped here but remain in ``other_masks`` above, so they
            # still feed ``neighbours`` of the chosen target (the dense sampler + corridor-risk +
            # approach-validation see the clutter).
            if self.target_label is not None and (
                (getattr(seg, "label", "") or getattr(seg, "prim_path", "")) != self.target_label
            ):
                continue
            label_matches += 1
            neighbours = [m for j, m in enumerate(other_masks) if j != idx and m is not None]
            extra_kwargs: dict = {}
            if camera_to_base is not None:
                extra_kwargs["camera_to_base"] = camera_to_base
            # Forward the frame rgb only when the calculator opted into rendering the grasp-point
            # overlay through its ``render_debug_images`` switch. Default off means
            # ``rgb_image=None``, no render, byte-identical and zero cost. ``getattr`` keeps a
            # calculator that does not declare the switch on the off path.
            if getattr(self.calculator, "render_debug_images", False):
                extra_kwargs["rgb_image"] = getattr(frame, "rgb", None)
            # Feed the fused multi-view scene cloud (camera frame) so the collision filter
            # sees neighbour geometry this single synthesis view cannot. The calculator vstacks it
            # with the same-view ``other_object_masks`` neighbours. None keeps the key absent and
            # the path byte-identical.
            if external_scene_cam is not None:
                extra_kwargs["scene_points_mm"] = external_scene_cam
            # The multi-camera generalisation of the line above: after target fusion, 60.0 % of
            # every remaining mis-rank is a finger collision, and the neighbour that the fingers
            # hit is one the synthesis view never saw. This hands this object the rest of the rig's
            # view of everything that is not it. Same precedence rule as the target cloud: an
            # explicit external cloud still wins, so enabling fusion cannot change a runner that
            # supplies one.
            elif fused_scene is not None and camera_to_base is not None:
                neighbour_base = fused_scene.neighbour_for(idx)
                if neighbour_base is not None and neighbour_base.size:
                    neighbour_cam = self._base_points_to_camera(neighbour_base, camera_to_base)
                    if neighbour_cam is not None:
                        extra_kwargs["scene_points_mm"] = neighbour_cam
            # Walls travel on their own kwarg, alongside whichever cloud won above rather than
            # instead of it: they are a different obstacle from a neighbouring part and answer a
            # different share of the ranking gap, so a cell that declared both must get both.
            # ``rigid_obstacle_points_mm`` rather than ``scene_points_mm`` because the generator
            # dilates observed points by 12 mm to cover the gaps between depth samples, and a
            # declared wall has no gaps: dilating it forbids a band of the bin a gripper can reach.
            if wall_cam is not None:
                extra_kwargs["rigid_obstacle_points_mm"] = wall_cam
            # Forward the fused multi-view target cloud (BASE) so the generator searches side-face
            # contacts. Only for the target segmentation (target_label set), so the target's cloud
            # is never applied to a neighbour seg. None keeps the key absent and byte-identical.
            if self.external_target_geometry_base_mm is not None and self.target_label is not None:
                extra_kwargs["geometry_points_base_mm"] = self.external_target_geometry_base_mm
            # The multi-camera generalisation of the line above: this object's own fused cloud,
            # matched across the rig by surface overlap. Indexed by segmentation, so every candidate
            # object gets its own, not just a labelled target. An explicit external cloud (the sim
            # runner's ground-truth path) still wins, so enabling fusion cannot change a runner that
            # already supplies one.
            elif fused_scene is not None:
                object_cloud = fused_scene.cloud_for(idx)
                if object_cloud is not None and object_cloud.size:
                    extra_kwargs["geometry_points_base_mm"] = object_cloud
            if self.gripper_model is not None:
                extra_kwargs["gripper_model"] = self.gripper_model
            if self.corridor_config is not None:
                extra_kwargs["corridor_config"] = self.corridor_config
            if self.feasibility_config is not None:
                extra_kwargs["feasibility_config"] = self.feasibility_config
                extra_kwargs["feasibility_weight"] = self.feasibility_weight
            # The plane is BASE-framed, and the calculator's collision filter runs on CAMERA-frame
            # poses, so it needs the transform to place the table at all: without one it refuses
            # rather than silently skipping the check. The calculator is never handed an
            # unanswerable question: a cell with no CAMERA->BASE transform cannot drive to a
            # base-frame grasp either, so the plane is withheld and the warning says so instead of
            # the check being dropped quietly (a silently-skipped table check is the exact defect
            # this whole seam exists to close).
            support_cfg = self.support_config
            support = self._resolve_support(seg, frame, camera_to_base)
            if support is not None and support_cfg is not None and camera_to_base is not None:
                extra_kwargs["support_plane"] = support.plane
                extra_kwargs["min_table_clearance_mm"] = support_cfg.min_clearance_mm
            elif support is not None:
                _LOG.warning(
                    "support plane configured but no CAMERA->BASE transform is available; "
                    "the table-clearance check is not running for this attempt")
            result = self.calculator.compute_result(
                seg,
                frame.depth_map,
                pixel_to_mm=None,
                dense_sampling=self.dense_sampling,
                grasp_sampling_mode=self._resolved_sampling_mode,
                other_object_masks=neighbours,
                **extra_kwargs,
            )
            self._stamp_deep_ranker(result, extra_kwargs, fused_scene, idx)
            if result.is_success:
                if ordering_active:
                    successes.append((idx, result))
                if best_success is None or result.top_score > best_success.top_score:
                    best_success = result
                    best_index = idx
            else:
                last_failure = result
                last_failure_index = idx
        if best_success is None:
            if last_failure is None and self.target_label is not None and label_matches == 0:
                # Every segmentation was skipped at the label gate, so no calculator ran and there
                # is no result to inherit reasons from. Without a named reason the caller gets
                # ``None`` and the attempt records ``reasons=()``, so an operator prompt that
                # matched nothing surfaces as a bare "no valid grasp", indistinguishable from a
                # scene the cell genuinely could not grasp.
                #
                # This names the failure and changes nothing else: the reason is absent from
                # ``_RESCAN_REASONS`` and ``_RELOCATE_REASONS``, so ``_decide_action`` still returns
                # "exhausted", as it does for the empty tuple. ``labels_seen`` is the actionable
                # half: what the operator asked for versus what perception actually returned.
                labels_seen = tuple(
                    str(getattr(seg, "label", "") or getattr(seg, "prim_path", "") or "")
                    for seg in frame.segmentations
                )
                return (
                    GraspResult(
                        reasons=(GraspFailureReason.TARGET_LABEL_NOT_FOUND,),
                        telemetry={
                            "target_label": self.target_label,
                            "labels_seen": labels_seen,
                            "segmentations": len(frame.segmentations),
                        },
                    ),
                    None,
                    None,
                )
            return last_failure, last_failure_index, None

        if not ordering_active or len(successes) <= 1:
            return best_success, best_index, None

        # ---- clutter-aware ordering ----
        candidates = _build_target_candidates(
            successes, frame.depth_map, frame.segmentations
        )
        assert self.target_ordering is not None  # narrow for the checker
        decision = select_target(
            candidates=candidates, config=self.target_ordering
        )
        # Map the selector's choice (index into ``candidates``) back
        # to the originating segmentation index + cached GraspResult.
        if decision.chosen_index is None:
            return best_success, best_index, decision
        chosen_seg_idx, chosen_result = successes[decision.chosen_index]
        return chosen_result, chosen_seg_idx, decision

    def _decide_action(self, reasons: Iterable[GraspFailureReason]) -> str:
        reason_set = set(reasons)
        if reason_set & _RELOCATE_REASONS:
            return "relocate" if self.viewpoint_planner is not None else "rescan"
        if reason_set & _RESCAN_REASONS:
            return "rescan"
        return "exhausted"

    def _relocate_camera(self) -> bool:
        """Drive the arm to the next viewpoint.

        Returns ``False`` when no planner is wired, when the planner has no further viewpoint, and
        when the typed ``move`` came back with a status other than ``MotionStatus.EXECUTED``. The
        caller maps all three to :attr:`PickOutcome.RELOCATED_EXHAUSTED`, so a refused motion is
        not distinguishable from exhaustion in the report.
        """
        if self.viewpoint_planner is None:
            return False
        current = self.arm.get_tcp_pose()
        next_pose = self.viewpoint_planner.next_viewpoint(
            current_tcp=current,
            history=tuple(self._viewpoints_visited),
        )
        if next_pose is None:
            return False
        # The typed move when the arm has one, like the execution policy does. `move_to` returns a
        # bool and, on the drivers that carry one, is the surface with the thinner history; `move` is
        # the one that plans when a planner is configured. A relocation is a real commanded motion
        # near the cell, so it deserves the same path a grasp gets.
        typed_move = getattr(self.arm, "move", None)
        if callable(typed_move):
            result = typed_move(next_pose)
            status = getattr(result, "status", None)
            if status is not None and status is not MotionStatus.EXECUTED:
                _LOG.warning(
                    "viewpoint relocation refused (%s): %s", status, getattr(result, "message", ""),
                )
                return False
        else:
            self.arm.move_to(next_pose)
        self._viewpoints_visited.append(next_pose)
        return True

    def _execute(self, result: GraspResult) -> PolicyReport:
        """Delegate motion / gripper choreography to the policy.

        When approach validation is wired (``approach_path_policy`` set and this orchestrator's
        ``mode_label`` is in ``_approach_path_modes``, the config's
        ``approach_validation.apply_modes``, the two dense modes by default), try the ranked
        candidates in order, validating each candidate's approach/retreat sweep against the scene
        neighbour cloud, and execute the first sweep-clear one (fall-back). The check reads the
        mode label, so an orchestrator constructed with a dense sampling mode but no label gets no
        validation. If every candidate's sweep is blocked, refuse with ``APPROACH_PATH_BLOCKED``
        (no motion). Disabled (default) executes ``result.best`` verbatim, byte-identical.
        """
        grasp = result.best
        if grasp is None:
            # Defensive: callers should only invoke _execute on successful
            # results, but guard rather than IndexError on the policy.
            return PolicyReport(outcome=PolicyOutcome.EXECUTED)
        assert self.policy is not None  # ensured by __post_init__
        if self.approach_path_policy is not None and self.mode_label in self._approach_path_modes:
            cloud = self._obstacle_cloud_in_candidate_frame(result)
            for candidate in result.candidates:
                if self._approach_sweep_is_clear(candidate, cloud):
                    return self.policy.execute(candidate)
            # Every ranked candidate's sweep is blocked: refuse rather than drive into an obstacle.
            return PolicyReport(outcome=PolicyOutcome.APPROACH_PATH_BLOCKED)
        return self.policy.execute(grasp)

    def _obstacle_cloud_in_candidate_frame(self, result: GraspResult) -> "np.ndarray | None":
        """The scene neighbour cloud in the winning candidate's frame, or ``None`` to skip the check.

        The calculator surfaces a CAMERA-frame cloud on ``result.metadata``. Candidates are
        BASE-frame when a frame_resolver is wired (the production path), so transform the cloud with
        the stashed CAMERA->BASE. Fail-safe: a BASE candidate with no transform available returns
        ``None``, skipping the check and proceeding, rather than validate a base grasp against a
        camera cloud (a mis-framed verdict).
        """
        meta = result.metadata or {}
        cloud = meta.get("scene_points_mm")
        if cloud is None:
            return None
        cloud = np.asarray(cloud, dtype=np.float64)
        if cloud.size == 0:
            return None
        best = result.best
        if best is not None and best.frame is GraspFrame.BASE:
            transform = self._pending_camera_to_base
            if transform is None:
                return None  # skip: do not validate a base grasp against a camera-frame cloud
            mat = np.asarray(transform.to_matrix(), dtype=np.float64)
            return (cloud @ mat[:3, :3].T) + mat[:3, 3]
        return cloud  # candidate is camera-frame, so the cloud is already camera-frame

    def _approach_sweep_is_clear(
        self, candidate: GraspPoint, cloud: "np.ndarray | None"
    ) -> bool:
        """True iff the candidate's approach and retreat sweeps are clear of the obstacle cloud."""
        try:
            grasp_pose = _grasp_point_to_grasp_pose(candidate)
        except ValueError:
            # Degenerate axes (approach ~parallel to closing) cannot build a sweep; the calculator
            # already accepted this candidate's final pose, so proceed rather than spuriously refuse.
            return True
        approach_report, retreat_report = validate_approach_and_retreat(
            grasp_pose, obstacle_points_mm=cloud, policy=self.approach_path_policy,
        )
        return approach_report.is_clear and retreat_report.is_clear


def _stamp(result: "GraspResult", payload: dict) -> None:
    """Merge shadow telemetry into a result's own dict, without disturbing what is already there.

    `GraspResult.telemetry` is a plain mutable dict carrying `GraspCalculator.last_telemetry`, so
    this adds keys beside it rather than replacing anything. Every key is `deep_ranker_`-prefixed,
    so a bare name such as `top1` cannot collide with another subsystem's telemetry.
    """
    existing = getattr(result, "telemetry", None)
    if isinstance(existing, dict):
        existing.update(payload)


def _grasp_position_mm(result: "GraspResult") -> tuple[float, float, float] | None:
    """The winning candidate's BASE-frame position, for a progress event. ``None`` if unavailable.

    Best-effort by design: this feeds a display, and a malformed candidate must not be able to raise
    on the line before the arm moves. Everything that decides anything reads the candidate itself.
    """
    candidates = getattr(result, "candidates", ())
    if not candidates:
        return None
    position = getattr(candidates[0], "position", None)
    if position is None:
        return None
    try:
        return (float(position[0]), float(position[1]), float(position[2]))
    except (TypeError, ValueError, IndexError):  # pragma: no cover (defensive, display-only)
        return None
