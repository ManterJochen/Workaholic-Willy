"""Operator-facing grasp-mode enum, per-mode behavior profiles, and the
effective-config snapshot.

The configuration value objects live apart from the service that consumes them.
Leaf of the package dependency graph: depends only on stdlib + ``grasping.types.modes``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Optional

from src.robot.grasping.types.modes import GraspSamplingMode


class GraspMode(StrEnum):
    """High-level grasp behavior profile selected by the operator.

    Values are stable strings so they can flow through YAML config, JSON
    telemetry, and REST payloads without translation. Each value maps to a fixed
    :class:`GraspSamplingMode` plus a fixed set of behavior toggles via
    :class:`GraspBehaviorProfile`. ``_PROFILES`` below is the authority on what each
    value means.

    EASY-mode verification defaults to :class:`NoOpVerifier`; a width-delta verifier is an
    opt-in extra. EASY behaves exactly as the plain ``RuntimePickService`` path does unless
    an operator opts in, so adopting this service is not itself a behaviour change.

    Robotiq object detection is not wired automatically: the driver implements no
    ``is_object_detected``, so it never claims the ``ObjectDetectingGripper`` Protocol on its own.
    A jaw-width reading reaches the stack only through the opt-in verification stack, where
    ``WidthDeltaGripperVerifier`` is built beside ``ObjectDetectingGripperVerifier`` when
    ``verification_policy.enabled``, and that second verifier reports inconclusive for any gripper
    without the capability. No downstream verifier trusts a width reading that was not validated
    against that operator's parts.
    """

    EASY = "easy"
    AUTO = "auto"
    DENSE_CLUTTER = "dense_clutter"
    CLOSED_LOOP = "closed_loop"
    DENSE_AUTONOMOUS = "dense_autonomous"


GraspModeInput = "GraspMode | str | None"

_GRASP_MODE_ALIASES: dict[str, GraspMode] = {
    "easy": GraspMode.EASY,
    "single": GraspMode.EASY,
    "single_object": GraspMode.EASY,
    "auto": GraspMode.AUTO,
    "dense": GraspMode.DENSE_CLUTTER,
    "dense_clutter": GraspMode.DENSE_CLUTTER,
    "closed_loop": GraspMode.CLOSED_LOOP,
    "closedloop": GraspMode.CLOSED_LOOP,
    "dense_autonomous": GraspMode.DENSE_AUTONOMOUS,
    "autonomous": GraspMode.DENSE_AUTONOMOUS,
}


def resolve_grasp_mode(value: GraspMode | str | None) -> GraspMode:
    """Coerce any accepted input form into a :class:`GraspMode`.

    Accepted inputs:

    * an existing :class:`GraspMode` (returned as-is),
    * :data:`None`, which maps to :attr:`GraspMode.AUTO`,
    * a string from the case-insensitive alias table above.

    Anything else raises :class:`ValueError` listing the allowed forms.
    """

    if isinstance(value, GraspMode):
        return value
    if value is None:
        return GraspMode.AUTO
    if isinstance(value, str):
        key = value.strip().lower()
        mapped = _GRASP_MODE_ALIASES.get(key)
        if mapped is not None:
            return mapped
    allowed = "GraspMode enum, None, or one of " + ", ".join(
        sorted(set(_GRASP_MODE_ALIASES))
    )
    raise ValueError(
        f"Invalid grasp_mode value {value!r}; expected {allowed}."
    )


@dataclass(frozen=True, slots=True)
class GraspBehaviorProfile:
    """Locked behavior knobs for a given :class:`GraspMode`.

    All fields are read-only. Profiles are snapshot into
    :class:`AutonomousGraspReport` so a caller can always tell which
    flags were in effect for a specific pick, regardless of any later
    config changes.

    Attributes
    ----------
    mode
        The :class:`GraspMode` this profile represents.
    sampling_mode
        The :class:`GraspSamplingMode` value the calculator should use
        for this profile.
    refinement_enabled
        :data:`True` iff this mode is expected to perform a second-scan
        pre-grasp refinement. When the refiner is unwired the service
        refuses to execute modes that require it with a typed
        :attr:`AutonomousGraspOutcome.MODE_NOT_AVAILABLE` outcome.
    verification_enabled
        :data:`True` iff this mode expects a post-grasp verification
        policy beyond the existing :class:`ObjectDetectingGripper`
        trust-path. When the verifier is unwired the same
        ``MODE_NOT_AVAILABLE`` rule applies.
    recovery_allowed_actions
        Allow-list of :class:`SceneRecoveryAction` names, held as
        strings so this module does not couple to a Protocol that does
        not exist yet. Empty for :attr:`GraspMode.EASY`: EASY must
        never produce recovery motion.
    """

    mode: GraspMode
    sampling_mode: GraspSamplingMode
    refinement_enabled: bool = False
    verification_enabled: bool = False
    recovery_allowed_actions: tuple[str, ...] = ()


# Per-mode defaults. ``EASY`` always has zero recovery motion: that is a
# safety guarantee, not a knob.
_PROFILES: Mapping[GraspMode, GraspBehaviorProfile] = {
    GraspMode.EASY: GraspBehaviorProfile(
        mode=GraspMode.EASY,
        sampling_mode=GraspSamplingMode.SINGLE_OBJECT,
        refinement_enabled=False,
        verification_enabled=False,
        recovery_allowed_actions=(),
    ),
    GraspMode.AUTO: GraspBehaviorProfile(
        mode=GraspMode.AUTO,
        sampling_mode=GraspSamplingMode.AUTO,
        refinement_enabled=False,
        verification_enabled=False,
        recovery_allowed_actions=("rescan", "next_viewpoint"),
    ),
    GraspMode.DENSE_CLUTTER: GraspBehaviorProfile(
        mode=GraspMode.DENSE_CLUTTER,
        sampling_mode=GraspSamplingMode.DENSE_CLUTTER,
        refinement_enabled=False,
        verification_enabled=False,
        recovery_allowed_actions=("rescan", "next_viewpoint"),
    ),
    GraspMode.CLOSED_LOOP: GraspBehaviorProfile(
        mode=GraspMode.CLOSED_LOOP,
        sampling_mode=GraspSamplingMode.AUTO,
        refinement_enabled=True,
        verification_enabled=True,
        recovery_allowed_actions=("rescan", "next_viewpoint"),
    ),
    GraspMode.DENSE_AUTONOMOUS: GraspBehaviorProfile(
        mode=GraspMode.DENSE_AUTONOMOUS,
        sampling_mode=GraspSamplingMode.DENSE_CLUTTER,
        refinement_enabled=True,
        verification_enabled=True,
        recovery_allowed_actions=("rescan", "next_viewpoint", "nudge_target"),
    ),
}


def _profile_for(mode: GraspMode) -> GraspBehaviorProfile:
    """Return the locked profile for ``mode`` (a seam for future operator-supplied overrides)."""

    return _PROFILES[mode]


# The resolved ``robot.grasping`` overlay is grouped into one frozen sub-config
# per phase, mirroring the schema sub-blocks: ``grasping.decision`` becomes
# :class:`EffectiveDecisionConfig`, and so on. :class:`EffectiveGraspingConfig`
# nests these; its :meth:`~EffectiveGraspingConfig.to_dict` emits flat keys
# (``decision_enabled``, ``watchdog_mode``, and the rest) so the telemetry and
# replay serialization contract is unchanged.


@dataclass(frozen=True, slots=True)
class EffectiveDecisionConfig:
    """Decision-layer overlay. Defaults match the locked disabled state so legacy
    snapshots round-trip unchanged."""

    enabled: bool = False
    auto_uncertainty_threshold: float = 0.4
    max_reobservations: int = 2


@dataclass(frozen=True, slots=True)
class EffectiveFeasibilityConfig:
    """Feasibility-aware ranking overlay, plus the ``corridor_risk_*`` 4th-signal
    weight. Defaults match the locked schema disabled state so ``total_score``
    stays byte-identical until the operator opts in. ``apply_modes`` filtering
    (EASY lock-out) is the service's responsibility, not these flags'."""

    enabled: bool = False
    score_weight: float = 0.0
    ik_quality_enabled: bool = False
    joint_margin_enabled: bool = False
    swept_approach_enabled: bool = False
    swept_approach_top_k: int = 5
    corridor_risk_enabled: bool = False
    corridor_risk_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class EffectiveCorridorConfig:
    """Occlusion / hidden-geometry analyzer knobs. All default off so the snapshot
    is byte-identical until the operator opts in. EASY is locked out via
    ``apply_modes`` (enforced by the service)."""

    directional_enabled: bool = False
    hard_reject_enabled: bool = False
    top_k: int = 5
    radius_mm: float = 20.0
    step_mm: float = 5.0
    max_distance_mm: float = 200.0
    mask_fusion_weight: float = 0.5
    hard_reject_confidence_threshold: float = 0.7


@dataclass(frozen=True, slots=True)
class EffectiveOrderingConfig:
    """Clutter-aware target-ordering overlay. All flags default off so the snapshot
    is byte-identical until the operator opts in. EASY is locked out via
    ``apply_modes``."""

    enabled: bool = False
    unlock_weight: float = 0.0
    max_local_score_drop: float = 0.1
    mask_adjacency_enabled: bool = False
    depth_only_enabled: bool = False
    corridor_overlap_enabled: bool = False
    adjacency_radius_px: int = 5
    depth_tolerance_mm: float = 10.0


@dataclass(frozen=True, slots=True)
class EffectiveRecoveryOrchestratorConfig:
    """Recovery-orchestrator overlay. All flags default off. EASY mode is
    permanently excluded from the default ``apply_modes``."""

    enabled: bool = False
    max_actions: int = 2
    apply_modes: tuple[str, ...] = (
        "auto",
        "dense_clutter",
        "dense_autonomous",
    )
    allowed_actions: tuple[str, ...] = ()
    per_action_budget: tuple[tuple[str, int], ...] = ()
    #: The operator-declared BASE-frame box a physical recovery action may act inside, as
    #: ``(centre_xyz_mm, half_extents_xyz_mm, max_nudge_mm)``. ``None`` when the config declares no
    #: fixture, which the schema only permits when no physical action is allowed.
    #:
    #: Required, not optional, whenever `allowed_actions` names ``nudge_target`` or
    #: ``container_agitate``: `SceneRecoveryPolicy` refuses to construct without an envelope for
    #: those, because a push with no declared bound is a robot shoving an unbounded workspace.
    #:
    #: Deliberately not in :meth:`EffectiveGraspingConfig.to_dict`: that is a frozen flat 82-key
    #: telemetry contract whose exact keys and order are pinned, the first 79 positions held fixed
    #: and three keys appended after them, and a nested box is not a scalar. Widening it is a
    #: separate decision with telemetry-catalog consequences.
    fixture: "tuple[tuple[float, float, float], tuple[float, float, float], float] | None" = None


@dataclass(frozen=True, slots=True)
class EffectiveUncertaintyConfig:
    """Uncertainty-fusion overlay. All flags default off. EASY mode is permanently
    excluded from the default ``apply_modes``.

    When :attr:`enabled` is :data:`False` the builder mirrors
    :attr:`fail_closed_threshold` from the legacy
    ``decision.auto_uncertainty_threshold`` (back-compat). ``1.01`` is the
    off-sentinel for the aggressive / disagreement thresholds (the fused score is
    clamped to ``[0, 1]``)."""

    enabled: bool = False
    fail_closed_threshold: float = 0.4
    apply_modes: tuple[str, ...] = (
        "auto",
        "dense_clutter",
        "dense_autonomous",
    )
    weights: tuple[tuple[str, float], ...] = (
        ("depth_confidence", 1.0),
        ("mask_confidence", 1.0),
        ("occlusion_corridor_risk", 1.0),
        ("feasibility_margin", 1.0),
        ("verification_residual", 1.0),
        ("topology_risk", 0.0),
        ("semantic_confidence", 0.0),
    )
    ranking_penalty_weight: float = 0.0
    recovery_aggressive_threshold: float = 1.01
    channel_disagreement_threshold: float = 1.01
    calibration_artifact_path: Optional[str] = None


@dataclass(frozen=True, slots=True)
class EffectiveWatchdogConfig:
    """Drift + OOD watchdog. ``mode`` defaults to ``"shadow"`` (compute + emit
    telemetry, no behavioural effect until the operator escalates to ``canary`` /
    ``active``). ``"disabled"`` emits no telemetry. Five threshold ladders are
    non-decreasing; the OOD ladder is inverted, a lower score being worse, so it
    is validated ``moderate >= high >= severe``. Both directions are enforced at
    the schema layer by ``GraspingWatchdogConfig``."""

    mode: str = "shadow"
    window_size: int = 20
    min_samples_for_trend: int = 6
    calibration_delta_moderate_mm: float = 2.0
    calibration_delta_high_mm: float = 5.0
    calibration_delta_severe_mm: float = 10.0
    depth_confidence_shift_moderate: float = 0.05
    depth_confidence_shift_high: float = 0.10
    depth_confidence_shift_severe: float = 0.20
    fail_closed_rate_moderate: float = 0.30
    fail_closed_rate_high: float = 0.50
    fail_closed_rate_severe: float = 0.75
    hand_eye_residual_trend_moderate_mm: float = 1.0
    hand_eye_residual_trend_high_mm: float = 3.0
    hand_eye_residual_trend_severe_mm: float = 7.0
    verification_residual_trend_moderate_mm: float = 0.5
    verification_residual_trend_high_mm: float = 2.0
    verification_residual_trend_severe_mm: float = 5.0
    ood_score_moderate: float = 0.5
    ood_score_high: float = 0.3
    ood_score_severe: float = 0.1
    block_modes: tuple[str, ...] = ("auto",)


@dataclass(frozen=True, slots=True)
class EffectivePerformanceConfig:
    """Runtime SLO + bounded-compute snapshot. Defaults off / no-timeout so the
    snapshot is byte-identical until opt-in."""

    enabled: bool = False
    decision_latency_slo_ms: float = 60.0
    ranking_latency_slo_ms: float = 80.0
    fusion_latency_slo_ms: float = 220.0
    decision_timeout_ms: float = 0.0
    ranking_timeout_ms: float = 0.0
    fusion_timeout_ms: float = 0.0
    model_fallback_policy: str = "legacy_score"
    fusion_fallback_policy: str = "skip"
    emit_breach_events: bool = False
    breach_window_size: int = 32


@dataclass(frozen=True, slots=True)
class EffectiveGraspingConfig:
    """Snapshot of the ``robot.grasping`` block resolved at construction.

    The snapshot captures exactly the values the runtime used to wire
    the service, after merging:

    * the operator-supplied ``RobotConfig.grasping`` sub-tree, and
    * any per-call kwargs that overrode the config (``mode``,
      ``max_attempts``).

    Stored on :class:`AutonomousGraspService` (as
    :attr:`AutonomousGraspService.effective_config`) and threaded into
    every :class:`AutonomousGraspReport` so a downstream debugger can
    correlate behaviour to configuration without re-loading YAML.
    The :class:`GraspBehaviorProfile` already captures the locked
    per-mode toggles; this snapshot adds the config-derived overlay
    on top.

    Attributes
    ----------
    default_mode
        The :class:`GraspMode` that the service will use when
        :meth:`AutonomousGraspService.pick` is called without an
        explicit ``mode`` override.
    max_attempts
        Resolved upper bound on autonomous grasp attempts per
        :meth:`AutonomousGraspService.pick` call.
    closed_loop_enabled
        Whether the operator opted into pre-grasp refinement
        via ``robot.grasping.closed_loop.enabled``.
    verification_enabled
        Whether the operator opted into post-grasp
        verification via ``robot.grasping.closed_loop.verification.enabled``.
    dense_recovery_enabled
        Whether the operator opted into dense-clutter scene
        recovery via ``robot.grasping.dense_recovery.enabled``.
    dense_recovery_allowed_actions
        The recovery action allow-list resolved from
        ``robot.grasping.dense_recovery.allowed_actions``. Stored as a
        tuple of stable strings (matching
        :class:`src.robot.grasping.recovery.SceneRecoveryAction`
        values) so the snapshot is JSON-safe.
    """

    default_mode: GraspMode
    max_attempts: int
    closed_loop_enabled: bool
    verification_enabled: bool
    dense_recovery_enabled: bool
    dense_recovery_allowed_actions: tuple[str, ...]
    # The ``success_model``, ``fusion`` and ``approach_validation`` keys. Appended, never
    # inserted, so every key above keeps its position and its value.
    #
    # Each records the effective state, matching what `feasibility_enabled` and friends do: a
    # block outside its ``apply_modes`` reads False here even when the YAML says true, because
    # the question a record has to answer is "was it acting on this attempt", not "what did the
    # file say". `fusion` has no ``apply_modes`` so it reports its configured value; the substrate
    # additionally needs a CAMERA to BASE resolver, which is a runtime fact this snapshot cannot
    # see. `approach_validation` is the sharp one: it is the block that can refuse a motion, so
    # its key is what tells a `GraspAttemptRecord` whether the swept-volume validator was on for
    # that attempt.
    success_model_enabled: bool = False
    fusion_enabled: bool = False
    approach_validation_enabled: bool = False
    # Per-phase overlays, each grouped into a frozen sub-config that mirrors its
    # ``robot.grasping`` schema sub-block. Every sub-config defaults to its
    # locked-off state so a snapshot is byte-identical until the operator opts a
    # phase in. ``to_dict()`` flattens these back to the per-field keys.
    decision: EffectiveDecisionConfig = field(
        default_factory=EffectiveDecisionConfig
    )
    feasibility: EffectiveFeasibilityConfig = field(
        default_factory=EffectiveFeasibilityConfig
    )
    corridor: EffectiveCorridorConfig = field(
        default_factory=EffectiveCorridorConfig
    )
    ordering: EffectiveOrderingConfig = field(
        default_factory=EffectiveOrderingConfig
    )
    recovery_orchestrator: EffectiveRecoveryOrchestratorConfig = field(
        default_factory=EffectiveRecoveryOrchestratorConfig
    )
    uncertainty: EffectiveUncertaintyConfig = field(
        default_factory=EffectiveUncertaintyConfig
    )
    watchdog: EffectiveWatchdogConfig = field(
        default_factory=EffectiveWatchdogConfig
    )
    performance: EffectivePerformanceConfig = field(
        default_factory=EffectivePerformanceConfig
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict copy of the snapshot."""

        return {
            "default_mode": self.default_mode.value,
            "max_attempts": int(self.max_attempts),
            "closed_loop_enabled": bool(self.closed_loop_enabled),
            "verification_enabled": bool(self.verification_enabled),
            "dense_recovery_enabled": bool(self.dense_recovery_enabled),
            "dense_recovery_allowed_actions": list(
                self.dense_recovery_allowed_actions
            ),
            "decision_enabled": bool(self.decision.enabled),
            "decision_auto_uncertainty_threshold": float(
                self.decision.auto_uncertainty_threshold
            ),
            "decision_max_reobservations": int(
                self.decision.max_reobservations
            ),
            "feasibility_enabled": bool(self.feasibility.enabled),
            "feasibility_score_weight": float(self.feasibility.score_weight),
            "feasibility_ik_quality_enabled": bool(
                self.feasibility.ik_quality_enabled
            ),
            "feasibility_joint_margin_enabled": bool(
                self.feasibility.joint_margin_enabled
            ),
            "feasibility_swept_approach_enabled": bool(
                self.feasibility.swept_approach_enabled
            ),
            "feasibility_swept_approach_top_k": int(
                self.feasibility.swept_approach_top_k
            ),
            "corridor_directional_enabled": bool(
                self.corridor.directional_enabled
            ),
            "corridor_hard_reject_enabled": bool(
                self.corridor.hard_reject_enabled
            ),
            "corridor_top_k": int(self.corridor.top_k),
            "corridor_radius_mm": float(self.corridor.radius_mm),
            "corridor_step_mm": float(self.corridor.step_mm),
            "corridor_max_distance_mm": float(self.corridor.max_distance_mm),
            "corridor_mask_fusion_weight": float(
                self.corridor.mask_fusion_weight
            ),
            "corridor_hard_reject_confidence_threshold": float(
                self.corridor.hard_reject_confidence_threshold
            ),
            "feasibility_corridor_risk_enabled": bool(
                self.feasibility.corridor_risk_enabled
            ),
            "feasibility_corridor_risk_weight": float(
                self.feasibility.corridor_risk_weight
            ),
            "ordering_enabled": bool(self.ordering.enabled),
            "ordering_unlock_weight": float(self.ordering.unlock_weight),
            "ordering_max_local_score_drop": float(
                self.ordering.max_local_score_drop
            ),
            "ordering_mask_adjacency_enabled": bool(
                self.ordering.mask_adjacency_enabled
            ),
            "ordering_depth_only_enabled": bool(
                self.ordering.depth_only_enabled
            ),
            "ordering_corridor_overlap_enabled": bool(
                self.ordering.corridor_overlap_enabled
            ),
            "ordering_adjacency_radius_px": int(
                self.ordering.adjacency_radius_px
            ),
            "ordering_depth_tolerance_mm": float(
                self.ordering.depth_tolerance_mm
            ),
            "recovery_orchestrator_enabled": bool(
                self.recovery_orchestrator.enabled
            ),
            "recovery_orchestrator_max_actions": int(
                self.recovery_orchestrator.max_actions
            ),
            "recovery_orchestrator_apply_modes": list(
                self.recovery_orchestrator.apply_modes
            ),
            "recovery_orchestrator_allowed_actions": list(
                self.recovery_orchestrator.allowed_actions
            ),
            "recovery_orchestrator_per_action_budget": [
                [str(action), int(count)]
                for action, count in self.recovery_orchestrator.per_action_budget
            ],
            "uncertainty_enabled": bool(self.uncertainty.enabled),
            "uncertainty_fail_closed_threshold": float(
                self.uncertainty.fail_closed_threshold
            ),
            "uncertainty_apply_modes": list(self.uncertainty.apply_modes),
            "uncertainty_weights": [
                [str(name), float(value)]
                for name, value in self.uncertainty.weights
            ],
            "uncertainty_ranking_penalty_weight": float(
                self.uncertainty.ranking_penalty_weight
            ),
            "uncertainty_recovery_aggressive_threshold": float(
                self.uncertainty.recovery_aggressive_threshold
            ),
            "uncertainty_channel_disagreement_threshold": float(
                self.uncertainty.channel_disagreement_threshold
            ),
            "uncertainty_calibration_artifact_path": (
                None
                if self.uncertainty.calibration_artifact_path is None
                else str(self.uncertainty.calibration_artifact_path)
            ),
            "watchdog_mode": str(self.watchdog.mode),
            "watchdog_window_size": int(self.watchdog.window_size),
            "watchdog_min_samples_for_trend": int(
                self.watchdog.min_samples_for_trend
            ),
            "watchdog_calibration_delta_moderate_mm": float(
                self.watchdog.calibration_delta_moderate_mm
            ),
            "watchdog_calibration_delta_high_mm": float(
                self.watchdog.calibration_delta_high_mm
            ),
            "watchdog_calibration_delta_severe_mm": float(
                self.watchdog.calibration_delta_severe_mm
            ),
            "watchdog_depth_confidence_shift_moderate": float(
                self.watchdog.depth_confidence_shift_moderate
            ),
            "watchdog_depth_confidence_shift_high": float(
                self.watchdog.depth_confidence_shift_high
            ),
            "watchdog_depth_confidence_shift_severe": float(
                self.watchdog.depth_confidence_shift_severe
            ),
            "watchdog_fail_closed_rate_moderate": float(
                self.watchdog.fail_closed_rate_moderate
            ),
            "watchdog_fail_closed_rate_high": float(
                self.watchdog.fail_closed_rate_high
            ),
            "watchdog_fail_closed_rate_severe": float(
                self.watchdog.fail_closed_rate_severe
            ),
            "watchdog_hand_eye_residual_trend_moderate_mm": float(
                self.watchdog.hand_eye_residual_trend_moderate_mm
            ),
            "watchdog_hand_eye_residual_trend_high_mm": float(
                self.watchdog.hand_eye_residual_trend_high_mm
            ),
            "watchdog_hand_eye_residual_trend_severe_mm": float(
                self.watchdog.hand_eye_residual_trend_severe_mm
            ),
            "watchdog_verification_residual_trend_moderate_mm": float(
                self.watchdog.verification_residual_trend_moderate_mm
            ),
            "watchdog_verification_residual_trend_high_mm": float(
                self.watchdog.verification_residual_trend_high_mm
            ),
            "watchdog_verification_residual_trend_severe_mm": float(
                self.watchdog.verification_residual_trend_severe_mm
            ),
            "watchdog_ood_score_moderate": float(
                self.watchdog.ood_score_moderate
            ),
            "watchdog_ood_score_high": float(self.watchdog.ood_score_high),
            "watchdog_ood_score_severe": float(
                self.watchdog.ood_score_severe
            ),
            "watchdog_block_modes": list(self.watchdog.block_modes),
            "performance_enabled": bool(self.performance.enabled),
            "performance_decision_latency_slo_ms": float(
                self.performance.decision_latency_slo_ms
            ),
            "performance_ranking_latency_slo_ms": float(
                self.performance.ranking_latency_slo_ms
            ),
            "performance_fusion_latency_slo_ms": float(
                self.performance.fusion_latency_slo_ms
            ),
            "performance_decision_timeout_ms": float(
                self.performance.decision_timeout_ms
            ),
            "performance_ranking_timeout_ms": float(
                self.performance.ranking_timeout_ms
            ),
            "performance_fusion_timeout_ms": float(
                self.performance.fusion_timeout_ms
            ),
            "performance_model_fallback_policy": str(
                self.performance.model_fallback_policy
            ),
            "performance_fusion_fallback_policy": str(
                self.performance.fusion_fallback_policy
            ),
            "performance_emit_breach_events": bool(
                self.performance.emit_breach_events
            ),
            "performance_breach_window_size": int(
                self.performance.breach_window_size
            ),
            # Appended, never inserted: every key above keeps its position. The field comment
            # above says what these three record.
            "success_model_enabled": bool(self.success_model_enabled),
            "fusion_enabled": bool(self.fusion_enabled),
            "approach_validation_enabled": bool(self.approach_validation_enabled),
        }
