"""Grasping (deterministic + RL-shadow) config schema, a sibling of robot_schema.py."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field, model_validator

from .._base import StrictModel


class GraspingClosedLoopConfig(StrictModel):
    """Pre-grasp refinement (closed-loop) knobs.

    Consumed by :class:`src.robot.grasping.closed_loop.refinement.RefinementPolicy`
    when the operator opts into closed-loop modes. ``enabled=False`` disables the
    second-scan refinement step even when the active mode profile permits it. The
    ``max_*_correction_*`` fields cap how far one refinement may move the grasp,
    in millimetres and degrees.
    """

    enabled: bool = Field(default=False)
    pregrasp_rescan: bool = Field(default=True)
    max_position_correction_mm: float = Field(default=20.0, gt=0.0, le=200.0)
    max_orientation_correction_deg: float = Field(
        default=15.0, gt=0.0, le=90.0
    )
    max_grip_width_correction_mm: float = Field(
        default=20.0, gt=0.0, le=200.0
    )
    target_match_iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class GraspingVerificationConfig(StrictModel):
    """Post-grasp verification knobs.

    Mirrors the operator-tunable fields on
    :class:`src.robot.grasping.closed_loop.verification.GraspVerificationPolicy`.
    Verification stays off unless ``enabled: true`` is set explicitly.
    """

    enabled: bool = Field(default=False)
    require_object_detected: bool = Field(default=False)
    width_delta_min_mm: float = Field(default=2.0, ge=0.0, le=50.0)
    #: How far above the commanded close width the jaws may sit and still count as engaged, mm.
    #: Without a ceiling a gripper whose close never executed sits at its pre-open width (~80 mm on
    #: a 2F-85) and still passes, so the cell carries air to the drop-off and logs a success. 10 mm
    #: separates cleanly: a commanded 45 mm close reads 45-ish when it holds and ~80 mm when it did
    #: not, with room for a compliant object a few mm proud of the command. Re-measure on the real
    #: jaw before trusting the exact number.
    width_delta_max_mm: float | None = Field(default=10.0, ge=0.0, le=50.0)
    post_lift_vision_check: bool = Field(default=False)
    vision_displacement_iou_max: float = Field(default=0.2, ge=0.0, le=1.0)
    fail_closed: bool = Field(default=True)
    require_all_conclusive: bool = Field(
        default=False,
        description=(
            "How to treat a verifier that could not measure anything: a gripper with no width "
            "feedback and no object-detect capability returns INCONCLUSIVE, not FAILED. False "
            "(default) counts it as a pass and logs a WARNING plus a telemetry stamp saying "
            "verification ran and learned nothing, because the alternative is a sensorless cell "
            "that fails every pick including the good ones. True demands a real measurement and "
            "is the right setting once the gripper actually reports one. Distinct from "
            "`fail_closed`, which governs what happens after a verifier says FAILED."
        ),
    )


class GraspingDenseRecoveryConfig(StrictModel):
    """Scene-recovery knobs (dense-clutter behaviour).

    ``allowed_actions`` is a free-form list of recovery action names, matched
    against :class:`src.robot.grasping.recovery.SceneRecoveryAction` at
    runtime, so unknown names produce a typed error rather than a silent allow.
    """

    enabled: bool = Field(default=False)
    #: Per-attempt recovery budget for the dense-recovery block.
    #:
    #: `build_subpolicies` copies it into a `SceneRecoveryPolicy` that is stored on the service and
    #: never consulted by any pick path, so the key is live in the type system and dead in effect.
    #: The budget that binds at runtime is `robot.grasping.recovery.max_recovery_actions`, one
    #: block over. Kept deliberately: deleting it forces unpicking the unconsumed `recovery_policy`
    #: plumbing through a fail-closed service, a refactor rather than a config tidy-up.
    max_recovery_actions: int = Field(default=2, ge=0, le=10)
    strategy: Literal["active_perception", "next_target", "none"] = Field(
        default="active_perception",
        description=(
            "Which recovery to plan when a pick fails. The policy above says what is permitted; "
            "this says what is proposed.\n\n"
            "`active_perception` (default) escalates RESCAN -> NEXT_VIEWPOINT: look again, then "
            "look from somewhere else. It never moves a part, so it cannot make the scene worse, "
            "and it addresses the failure this reference actually has: 68 % of visible objects "
            "yield no candidate at all, which is a seeing problem. `next_target` simply takes a "
            "different object: the cheapest possible recovery and the wrong answer for a "
            "prompt-driven pick, because avoiding the asked-for object is not recovering. `none` "
            "plans nothing, which is what an operator wants while measuring the policy alone."
        ),
    )
    allowed_actions: tuple[str, ...] = Field(default=("next_viewpoint",))


class GraspingDecisionConfig(StrictModel):
    """Pre-execution decision-layer knobs.

    Consumed by :class:`src.robot.grasping.decision.DecisionPolicy` when
    the operator opts the auto fail-closed decision layer on. Defaults to
    ``enabled=False``, so a ``robot.yaml`` that omits the block is unaffected.

    Attributes
    ----------
    enabled
        Master switch. When :data:`False`, the runtime builds no
        :class:`DecisionEngine` and the pick path runs without one.
    auto_uncertainty_threshold
        Maximum uncertainty (in ``[0, 1]``) accepted as a confident grasp.
        ``uncertainty = (1 - top_score)`` plus an optional ``reasons_penalty``
        when the grasp result carries failure reasons. Default ``0.4`` => require
        ``top_score >= 0.6``.
    max_reobservations
        Bounded camera re-observation budget per
        :meth:`AutonomousGraspService.pick` call. Default ``2``.
    reasons_penalty
        Added to the base uncertainty when :class:`GraspResult.reasons` is
        non-empty. Default ``0.2``.
    fail_closed_on_real_hardware
        When :data:`True` and the running arm reports
        ``capabilities.is_simulated == False``, an exhausted re-observation budget
        or absent viewpoint planner is mapped to
        :class:`DecisionAction.FAIL_CLOSED`. Simulated runs stay permissive.
    """

    enabled: bool = Field(default=False)
    auto_uncertainty_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    max_reobservations: int = Field(default=2, ge=0, le=10)
    reasons_penalty: float = Field(default=0.2, ge=0.0, le=1.0)
    fail_closed_on_real_hardware: bool = Field(default=True)


_KNOWN_GRASP_MODES: frozenset[str] = frozenset(
    {
        "easy",
        "auto",
        "closed_loop",
        "dense_clutter",
        "dense_autonomous",
    }
)


class GraspingFeasibilityConfig(StrictModel):
    """Feasibility-aware ranking knobs.

    Carries three pre-commit feasibility signals that demote (never reject)
    candidates with poor execution prospects: IK reachability quality, joint-limit
    margin, and swept approach feasibility. All three are disabled by default
    (byte-identical). A signal takes effect only when its ``*_enabled`` flag is
    set and the top-level :attr:`weight` is non-zero; one without the other
    carries no weight. ``apply_modes`` excludes ``easy`` so easy's cycle-time
    budget is preserved.

    Attributes
    ----------
    enabled
        Master switch. When :data:`False`, the calculator is built without a
        feasibility scorer and the ranking is unchanged.
    weight
        Top-level feasibility weight in the
        :class:`~src.robot.grasping.scoring.GraspScoreWeights` normalised
        average. Default ``0.0`` keeps ``total_score`` byte-identical.
    apply_modes
        Modes not listed here skip feasibility ranking regardless of any other
        flag (easy is permanently locked out).
    ik_quality_enabled, joint_margin_enabled, swept_approach_enabled
        Per-signal switches. Disabled signals contribute a neutral ``0.5`` floor
        so they never demote a candidate below an enabled-but-unknown peer.
    ik_quality_weight, joint_margin_weight, swept_approach_weight
        Internal sub-weights inside the feasibility score; only relevant when
        their corresponding signal is enabled.
    swept_approach_top_k
        Maximum number of top-ranked candidates evaluated with swept-approach
        validation. Caps O(N) cost so the affected modes still meet cycle time.
    """

    enabled: bool = Field(default=False)
    weight: float = Field(default=0.0, ge=0.0, le=1.0)
    apply_modes: tuple[str, ...] = Field(
        default=("auto", "dense_clutter", "dense_autonomous")
    )
    ik_quality_enabled: bool = Field(default=False)
    ik_quality_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    joint_margin_enabled: bool = Field(default=False)
    joint_margin_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    swept_approach_enabled: bool = Field(default=False)
    swept_approach_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    swept_approach_top_k: int = Field(default=5, ge=1, le=64)

    @model_validator(mode="after")
    def _validate_apply_modes(self) -> "GraspingFeasibilityConfig":
        unknown = [m for m in self.apply_modes if m not in _KNOWN_GRASP_MODES]
        if unknown:
            raise ValueError(
                "feasibility.apply_modes contains unknown grasp mode(s): "
                f"{unknown!r}; valid: {sorted(_KNOWN_GRASP_MODES)!r}"
            )
        return self


class GraspingOcclusionConfig(StrictModel):
    """Occlusion and hidden-geometry knobs.

    Carries two directional-occlusion behaviours on top of the feasibility
    carrier: a corridor analyzer, which marches the approach axis (and the
    retreat axis when asked) in camera frame to estimate per-direction clearance
    and a fused depth+mask blockage confidence, and a hard reject, which drops
    candidates whose corridor confidence exceeds
    :attr:`hard_reject_confidence_threshold` instead of merely demoting them.
    All flags default off (byte-identical); ``easy`` is excluded from
    :attr:`apply_modes`.

    Attributes
    ----------
    directional_enabled
        Master switch for the corridor analyzer. When False, no corridor reports
        are produced and ``hard_reject_enabled`` has no effect.
    hard_reject_enabled
        When True (and the analyzer is enabled and the mode is in
        ``apply_modes``), candidates whose blockage confidence exceeds
        :attr:`hard_reject_confidence_threshold` are rejected outright rather than
        only demoted by feasibility.
    apply_modes
        Modes not listed here skip the analyzer entirely regardless of other flags.
    corridor_radius_mm, corridor_step_mm, corridor_max_distance_mm
        Sampling geometry passed to the analyzer.
    mask_fusion_weight
        Convex-combination weight of mask blockage vs depth blockage in the
        analyzer's fused confidence.
    partial_confidence_threshold, hard_reject_confidence_threshold
        Confidence cut-offs separating the ``clear``, ``partial`` and ``blocked``
        :class:`CorridorMode` bands, and the hard-reject trigger respectively.
    top_k
        Cap on number of candidates analyzed per ``pick()`` cycle.
    """

    directional_enabled: bool = Field(default=False)
    hard_reject_enabled: bool = Field(default=False)
    apply_modes: tuple[str, ...] = Field(
        default=("auto", "dense_clutter", "dense_autonomous")
    )
    corridor_radius_mm: float = Field(default=20.0, gt=0.0)
    corridor_step_mm: float = Field(default=5.0, gt=0.0)
    corridor_max_distance_mm: float = Field(default=200.0, gt=0.0)
    mask_fusion_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    partial_confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    hard_reject_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1, le=64)

    @model_validator(mode="after")
    def _validate(self) -> "GraspingOcclusionConfig":
        unknown = [m for m in self.apply_modes if m not in _KNOWN_GRASP_MODES]
        if unknown:
            raise ValueError(
                "occlusion.apply_modes contains unknown grasp mode(s): "
                f"{unknown!r}; valid: {sorted(_KNOWN_GRASP_MODES)!r}"
            )
        if self.partial_confidence_threshold >= self.hard_reject_confidence_threshold:
            raise ValueError(
                "occlusion.partial_confidence_threshold "
                f"({self.partial_confidence_threshold}) must be < "
                "hard_reject_confidence_threshold "
                f"({self.hard_reject_confidence_threshold})"
            )
        return self


class BlockerGraphSchemaConfig(StrictModel):
    """Per-signal blocker-graph flags.

    Every signal defaults off, so :class:`GraspingOrderingConfig` is inert until an
    operator opts in. ``adjacency_radius_px`` is integer pixels in the perception mask;
    ``depth_tolerance_mm`` is the camera-space depth difference in millimetres required
    before one candidate counts as in front of another.
    """

    mask_adjacency_enabled: bool = Field(default=False)
    depth_only_enabled: bool = Field(default=False)
    #: Reserved: whether two candidates whose approach corridors overlap count as blocking each other.
    #: The flag is carried the whole way, schema -> EffectiveOrderingConfig -> the frozen 79-key
    #: telemetry dict as `ordering_corridor_overlap_enabled` -> BlockerGraphConfig, and
    #: `target_selector._blocks` does nothing with it: corridor geometry is not consumed there. The
    #: key reaches a frozen telemetry contract, so removing it changes that contract's key set.
    corridor_overlap_enabled: bool = Field(default=False)
    adjacency_radius_px: int = Field(default=5, ge=0, le=128)
    depth_tolerance_mm: float = Field(default=10.0, ge=0.0)


class GraspingOrderingConfig(StrictModel):
    """Clutter-aware target-ordering knobs.

    When :attr:`enabled` the runtime computes an unlock score per candidate via
    the configured :class:`BlockerGraphSchemaConfig` signals and a priority
    score ``local + unlock_weight * unlock``. The ``argmax(local_score)`` winner
    is only displaced when the priority winner is within
    :attr:`max_local_score_drop` of it: the operator decides how much local score
    they will give up to unblock other picks. Every flag defaults off
    (byte-identical); ``easy`` is excluded from the default apply modes.
    """

    enabled: bool = Field(default=False)
    unlock_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    max_local_score_drop: float = Field(default=0.1, ge=0.0, le=1.0)
    apply_modes: tuple[str, ...] = Field(
        default=("auto", "dense_clutter", "dense_autonomous")
    )
    blocker_graph: BlockerGraphSchemaConfig = Field(
        default_factory=BlockerGraphSchemaConfig
    )

    @model_validator(mode="after")
    def _validate_apply_modes(self) -> "GraspingOrderingConfig":
        unknown = [m for m in self.apply_modes if m not in _KNOWN_GRASP_MODES]
        if unknown:
            raise ValueError(
                "ordering.apply_modes contains unknown grasp mode(s): "
                f"{unknown!r}; valid: {sorted(_KNOWN_GRASP_MODES)!r}"
            )
        return self


class GraspingRecoveryConfig(StrictModel):
    """Recovery-orchestration knobs.

    Mirrors the surface in
    :class:`src.robot.grasping.recovery.SceneRecoveryPolicy` but holds
    string action names, so the config layer imports no runtime enum. Every flag
    defaults off (byte-identical); ``easy`` is excluded from the default apply
    modes. When :attr:`allowed_actions` contains a physical action
    (``nudge_target`` or ``container_agitate``), :attr:`fixture` must be set: the
    cross-field check refuses to arm a physical recovery without an envelope.
    """

    enabled: bool = Field(default=False)
    max_recovery_actions: int = Field(default=2, ge=0, le=20)
    allowed_actions: tuple[str, ...] = Field(default=())
    per_action_budget: tuple[tuple[str, int], ...] = Field(default=())
    apply_modes: tuple[str, ...] = Field(
        default=("auto", "dense_clutter", "dense_autonomous")
    )
    fixture: "GraspingRecoveryFixtureConfig | None" = Field(default=None)

    @model_validator(mode="after")
    def _validate(self) -> "GraspingRecoveryConfig":
        valid_actions = frozenset(
            {
                "rescan",
                "next_viewpoint",
                "next_target",
                "nudge_target",
                "container_agitate",
            }
        )
        physical = frozenset({"nudge_target", "container_agitate"})
        for action in self.allowed_actions:
            if action not in valid_actions:
                raise ValueError(
                    "recovery.allowed_actions contains unknown action: "
                    f"{action!r}; valid: {sorted(valid_actions)!r}"
                )
        for entry in self.per_action_budget:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
            ):
                raise ValueError(
                    "recovery.per_action_budget entries must be "
                    f"(action, count) pairs; got {entry!r}"
                )
            action, count = entry
            if action not in valid_actions:
                raise ValueError(
                    "recovery.per_action_budget unknown action: "
                    f"{action!r}; valid: {sorted(valid_actions)!r}"
                )
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError(
                    "recovery.per_action_budget counts must be non-negative "
                    f"ints; got {action}={count!r}"
                )
        unknown = [m for m in self.apply_modes if m not in _KNOWN_GRASP_MODES]
        if unknown:
            raise ValueError(
                "recovery.apply_modes contains unknown grasp mode(s): "
                f"{unknown!r}; valid: {sorted(_KNOWN_GRASP_MODES)!r}"
            )
        physical_in_use = [a for a in self.allowed_actions if a in physical]
        if physical_in_use and self.fixture is None:
            raise ValueError(
                "recovery physical actions require a fixture envelope; "
                f"got {physical_in_use!r} without fixture"
            )
        return self


class GraspingRecoveryFixtureConfig(StrictModel):
    """Operator-bounded envelope for physical recovery actions.

    A recovery action is the robot deliberately pushing something: nudging a part that will not
    separate, agitating a container. That is motion aimed at the scene rather than at a grasp, so
    its reach is declared by the operator rather than inferred, and the recovery planner may not
    exceed these three numbers.
    """

    #: Centre in robot BASE frame (mm) of the axis-aligned box a physical recovery action may act
    #: inside, typically the bin or tray the cell is clearing.
    center_mm: tuple[float, float, float] = Field(...)
    #: Half the side length on each axis (mm), so the box spans ``center +/- half_extents``. Half
    #: extents rather than corners because the recovery planner reasons about distance from the centre.
    half_extents_mm: tuple[float, float, float] = Field(...)
    #: Longest single push (mm) any one recovery action may command. Bounds the displacement even when
    #: the box above would permit more: a large shove can eject a part from the bin or bury the target
    #: deeper, so the ceiling is deliberately small and separate from the envelope.
    max_nudge_mm: float = Field(default=5.0, ge=0.0, le=50.0)


class UncertaintyChannelWeightsConfig(StrictModel):
    """Per-channel weights for the uncertainty fusion layer.

    The five always-produced channels default to ``1.0``; ``topology_risk`` and
    ``semantic_confidence`` default to ``0.0`` because the runtime does not always
    produce them, so an operator opts those two in.
    """

    depth_confidence: float = Field(default=1.0, ge=0.0)
    mask_confidence: float = Field(default=1.0, ge=0.0)
    occlusion_corridor_risk: float = Field(default=1.0, ge=0.0)
    feasibility_margin: float = Field(default=1.0, ge=0.0)
    verification_residual: float = Field(default=1.0, ge=0.0)
    topology_risk: float = Field(default=0.0, ge=0.0)
    semantic_confidence: float = Field(default=0.0, ge=0.0)

    def sum(self) -> float:
        return (
            self.depth_confidence
            + self.mask_confidence
            + self.occlusion_corridor_risk
            + self.feasibility_margin
            + self.verification_residual
            + self.topology_risk
            + self.semantic_confidence
        )


class GraspingUncertaintyConfig(StrictModel):
    """Uncertainty fusion + calibration layer.

    Fusion is a weighted convex combination of seven typed per-channel
    monotone-remapped values (exposed via :class:`UncertaintyChannelWeightsConfig`;
    runtime carrier in :mod:`src.robot.grasping.uncertainty`). When
    :attr:`enabled`, :attr:`fail_closed_threshold` replaces
    :attr:`GraspingDecisionConfig.auto_uncertainty_threshold`; when disabled the
    decision layer keeps its own threshold (byte-identical). The layer always emits a
    typed snapshot on every :class:`AutonomousGraspReport` (``fused=0.0,
    fused_available=False`` when disabled). ``"easy"`` is excluded from
    :attr:`apply_modes`.

    Attributes
    ----------
    enabled
        Master switch. Defaults to :data:`False` (byte-identical).
    fail_closed_threshold
        When ``fused >= fail_closed_threshold`` the decision layer fail-closes the
        attempt (replacing ``GraspingDecisionConfig.auto_uncertainty_threshold``).
        Default ``0.4`` matches that threshold's own default.
    apply_modes
        Stable mode names the layer applies to. ``"easy"`` is rejected.
    weights
        Per-channel non-negative weights. When :attr:`enabled` at least one weight
        must be strictly positive; all-zero weights are refused, because a fused
        signal pinned at zero gates nothing.
    ranking_penalty_weight
        Ranking carrier. Multiplies the fused uncertainty when subtracting from a
        candidate score. Default ``0.0`` => ranking byte-identical.
    recovery_aggressive_threshold
        Recovery carrier. When fused exceeds this threshold the orchestrator may
        prefer perception-recovery actions (``NEXT_VIEWPOINT``/``RESCAN``). Default
        ``1.01`` => never trips (fused is clamped to ``[0, 1]``).
    calibration_artifact_path
        Optional path (absolute or relative to the config root) to a JSON
        calibration artifact produced by
        :mod:`src.robot.grasping.calibration.uncertainty_calibration`.
        When :data:`None`, the identity calibration is used.
    """

    enabled: bool = Field(default=False)
    fail_closed_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    apply_modes: tuple[str, ...] = Field(
        default=("auto", "dense_clutter", "dense_autonomous")
    )
    weights: UncertaintyChannelWeightsConfig = Field(
        default_factory=UncertaintyChannelWeightsConfig
    )
    ranking_penalty_weight: float = Field(
        default=0.0,
        ge=0.0,
        json_schema_extra={
            "runtime_mutable": True,
            "min_value": 0.0,
            "max_value": 1.0,
            "max_abs_step": 0.20,
            "max_rel_step": 0.50,
        },
    )
    recovery_aggressive_threshold: float = Field(
        default=1.01,
        ge=0.0,
        json_schema_extra={
            "runtime_mutable": True,
            "min_value": 0.0,
            "max_value": 1.01,
            "max_abs_step": 0.10,
            "max_rel_step": 0.30,
        },
    )
    # 1.01 = off-sentinel: the channel spread is bounded above by 1.0, so the gate never trips by default.
    channel_disagreement_threshold: float = Field(default=1.01, ge=0.0)
    calibration_artifact_path: str | None = Field(default=None)
    # Per-candidate uncertainty re-rank (the consumer of the corridor-risk producer). rerank_modes is
    # dense-only and separate from apply_modes (which includes "auto" and would crash the dense-only
    # carrier); default off + weight 0.0 is byte-identical, so enabling is observe-only until a positive
    # weight is set. Enabling here is not sufficient: the calculator owner must also build it with
    # ``corridor_risk_per_candidate=True`` (the producer) or the re-rank silently no-ops. Costs N corridor
    # depth-marches per dense pick.
    rerank_enabled: bool = Field(default=False)
    # Deliberately not runtime-mutable: weight 0.0 is an explicit, human-logged tuning step;
    # auto-tuning it up would silently enable reordering, defeating the observe-only intent.
    rerank_weight: float = Field(default=0.0, ge=0.0, le=0.5)
    rerank_modes: tuple[str, ...] = Field(
        default=("dense_clutter", "dense_autonomous")
    )

    @model_validator(mode="after")
    def _validate(self) -> "GraspingUncertaintyConfig":
        if "easy" in self.apply_modes:
            raise ValueError(
                "uncertainty.apply_modes must not include 'easy'; "
                "easy mode is permanently excluded from T6."
            )
        unknown = [m for m in self.apply_modes if m not in _KNOWN_GRASP_MODES]
        if unknown:
            raise ValueError(
                "uncertainty.apply_modes contains unknown grasp mode(s): "
                f"{unknown!r}; valid: {sorted(_KNOWN_GRASP_MODES)!r}"
            )
        if self.enabled and self.weights.sum() <= 0.0:
            raise ValueError(
                "uncertainty.enabled=True requires at least one "
                "weight to be strictly positive; got all zeros."
            )
        _dense_rerank_modes = frozenset({"dense_clutter", "dense_autonomous"})
        bad_rerank = [m for m in self.rerank_modes if m not in _dense_rerank_modes]
        if bad_rerank:
            raise ValueError(
                "uncertainty.rerank_modes is locked to dense modes only "
                f"(mirrors the U4 ranking-blend contract); got disallowed mode(s) {bad_rerank!r}; "
                f"valid: {sorted(_dense_rerank_modes)!r}"
            )
        if len(set(self.rerank_modes)) != len(self.rerank_modes):
            raise ValueError("uncertainty.rerank_modes must be unique")
        return self


class GraspingSuccessModelConfig(StrictModel):
    """Success-probability overlay (default off).

    Points the runtime at a trained artifact and bounds what it may influence.
    ``enabled=False`` means nothing in the pipeline loads or runs the model, so
    adding this block or omitting it is byte-identical.

    Attributes
    ----------
    enabled
        Master switch. Defaults to :data:`False`, so the runtime never loads the
        artifact.
    artifact_dir
        Path (absolute or relative to the config root) of the directory containing
        ``model.json`` and ``manifest.json``. Defaults to the committed v1 artifact
        under ``assets/models/success_probability/v1``.
    apply_modes
        Stable mode names the model may score in. Defaults to the four canonical
        modes; ``"easy"`` is included because easy pick rate is the
        strict-improvement target the model is calibrated against.
    ranking_blend_enabled
        Master switch for the bounded probability-blend reranker. Defaults to
        :data:`False` (byte-identical). When :data:`True` (and ``enabled``, and a
        non-shadow lifecycle artifact is loaded), the runtime reranks
        ``best_result.candidates`` in dense modes only using the convex blend
        ``(1 - w) * geometric + w * predicted_p``.
    ranking_blend_weight
        Blend weight ``w``. Clamped to ``[0.0, 0.5]`` so the geometric score always
        retains >=50 % influence and a miscalibrated model cannot dominate ranking.
        Default ``0.20``.
    ranking_blend_modes
        Locked dense-only subset of grasp-modes the blend reranker may operate in:
        a subset of ``{"dense_clutter", "dense_autonomous"}`` (``"easy"`` and
        ``"auto"`` stay geometry-only so easy pick-rate cannot regress). Must also
        be a subset of ``apply_modes`` (else the shadow probability is never
        annotated and the blend has nothing to read).
    """

    enabled: bool = Field(default=False)
    artifact_dir: str = Field(
        default="assets/models/success_probability/v1", min_length=1
    )
    apply_modes: tuple[str, ...] = Field(
        default=("easy", "auto", "dense_clutter", "dense_autonomous")
    )
    ranking_blend_enabled: bool = Field(default=False)
    ranking_blend_weight: float = Field(
        default=0.20,
        ge=0.0,
        le=0.5,
        json_schema_extra={
            "runtime_mutable": True,
            "min_value": 0.0,
            "max_value": 0.5,
            "max_abs_step": 0.05,
            "max_rel_step": 0.25,
        },
    )
    ranking_blend_modes: tuple[str, ...] = Field(
        default=("dense_clutter", "dense_autonomous")
    )
    lifecycle_phase: Literal["shadow", "canary", "active"] = Field(
        default="shadow",
        description=(
            "U4 lifecycle gate (G1). 'shadow' (default) = the learned success predictor runs but its "
            "output is discarded (byte-identical to the no-blend path). 'canary'/'active' let the "
            "bounded blend reorder candidates in the dense modes; both require a valid promotion.json "
            "(verify_promotion: SHA-chain + verdict='pass' + thresholds not weaker than plan-locked + "
            "the plan-locked validation slice + metrics that agree with the recorded bounds) or the "
            "model load fails closed to None (no blend)."
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> "GraspingSuccessModelConfig":
        unknown = [m for m in self.apply_modes if m not in _KNOWN_GRASP_MODES]
        if unknown:
            raise ValueError(
                "success_model.apply_modes contains unknown grasp mode(s): "
                f"{unknown!r}; valid: {sorted(_KNOWN_GRASP_MODES)!r}"
            )
        if len(set(self.apply_modes)) != len(self.apply_modes):
            raise ValueError("success_model.apply_modes must be unique")
        # Ranking-blend is locked to dense-only so easy/auto/closed_loop never blend and easy pick-rate
        # cannot regress regardless of operator config.
        _DENSE_ONLY: frozenset[str] = frozenset(
            {"dense_clutter", "dense_autonomous"}
        )
        forbidden = [
            m for m in self.ranking_blend_modes if m not in _DENSE_ONLY
        ]
        if forbidden:
            raise ValueError(
                "success_model.ranking_blend_modes is locked to "
                "dense modes only per the U4 design contract; "
                f"got disallowed mode(s) {forbidden!r}; valid: "
                f"{sorted(_DENSE_ONLY)!r}"
            )
        if len(set(self.ranking_blend_modes)) != len(self.ranking_blend_modes):
            raise ValueError(
                "success_model.ranking_blend_modes must be unique"
            )
        # Blend modes must be a subset of apply_modes: otherwise the
        # shadow probability is never annotated for that mode and the
        # blend would have nothing to read.
        missing = [
            m for m in self.ranking_blend_modes if m not in self.apply_modes
        ]
        if missing:
            raise ValueError(
                "success_model.ranking_blend_modes must be a subset of "
                f"apply_modes; got {missing!r} not in {self.apply_modes!r}"
            )
        return self


class GraspingWatchdogConfig(StrictModel):
    """Drift + OOD watchdog runtime knobs.

    Evaluates five drift monitors (predicted-vs-observed calibration delta,
    depth-confidence shift, fail-closed spike, hand-eye residual trend,
    verification residual trend) plus an OOD score over a rolling window of recent
    attempts, and surfaces a typed severity ladder
    (``none``/``low``/``moderate``/``high``/``severe``). Evaluators are pure
    functions; the :class:`AutonomousGraspService` owns the rolling history. The
    default ``shadow`` mode computes and emits telemetry but never alters
    behaviour; ``high`` and ``severe`` block the ``auto`` mode on real hardware
    only and are a telemetry flag in sim. Defaults are conservative (a healthy
    scene stays ``none``); thresholds form non-decreasing ladders and a
    misconfigured YAML is rejected at construction.

    Attributes
    ----------
    mode
        One of ``"disabled"``, ``"shadow"``, ``"canary"``, ``"active"``. Defaults
        to ``"shadow"``. ``"disabled"`` omits watchdog telemetry entirely.
    window_size
        Rolling history length, in attempts. Must be ``>= 1``.
    min_samples_for_trend
        Minimum samples before any trend monitor emits anything other than
        ``none``. Must be ``>= 2``.
    block_modes
        Stable grasp-mode names (matching
        :class:`src.robot.grasping.types.modes.GraspMode`) that fall under the
        ``high``/``severe`` block on real hardware. Defaults to ``("auto",)``.
        ``"easy"`` is permanently rejected.
    """

    mode: str = Field(default="shadow", min_length=1)
    window_size: int = Field(
        default=20,
        ge=1,
        le=1024,
        json_schema_extra={
            "runtime_mutable": True,
            "min_value": 1,
            "max_value": 1024,
            "max_abs_step": 10,
            "max_rel_step": 0.50,
        },
    )
    min_samples_for_trend: int = Field(default=6, ge=2, le=1024)

    calibration_delta_moderate_mm: float = Field(default=2.0, ge=0.0)
    calibration_delta_high_mm: float = Field(default=5.0, ge=0.0)
    calibration_delta_severe_mm: float = Field(default=10.0, ge=0.0)

    depth_confidence_shift_moderate: float = Field(default=0.05, ge=0.0, le=1.0)
    depth_confidence_shift_high: float = Field(default=0.10, ge=0.0, le=1.0)
    depth_confidence_shift_severe: float = Field(default=0.20, ge=0.0, le=1.0)

    fail_closed_rate_moderate: float = Field(default=0.30, ge=0.0, le=1.0)
    fail_closed_rate_high: float = Field(default=0.50, ge=0.0, le=1.0)
    fail_closed_rate_severe: float = Field(default=0.75, ge=0.0, le=1.0)

    hand_eye_residual_trend_moderate_mm: float = Field(default=1.0, ge=0.0)
    hand_eye_residual_trend_high_mm: float = Field(default=3.0, ge=0.0)
    hand_eye_residual_trend_severe_mm: float = Field(default=7.0, ge=0.0)

    verification_residual_trend_moderate_mm: float = Field(default=0.5, ge=0.0)
    verification_residual_trend_high_mm: float = Field(default=2.0, ge=0.0)
    verification_residual_trend_severe_mm: float = Field(default=5.0, ge=0.0)

    ood_score_moderate: float = Field(default=0.5, ge=0.0, le=1.0)
    ood_score_high: float = Field(default=0.3, ge=0.0, le=1.0)
    ood_score_severe: float = Field(default=0.1, ge=0.0, le=1.0)

    block_modes: tuple[str, ...] = Field(default=("auto",))

    _ALLOWED_MODES: ClassVar[frozenset[str]] = frozenset(
        {"disabled", "shadow", "canary", "active"}
    )

    @model_validator(mode="after")
    def _validate(self) -> "GraspingWatchdogConfig":
        if self.mode not in self._ALLOWED_MODES:
            raise ValueError(
                "watchdog.mode must be one of "
                f"{sorted(self._ALLOWED_MODES)!r}; got {self.mode!r}"
            )
        ladders: tuple[tuple[str, tuple[float, float, float]], ...] = (
            (
                "calibration_delta",
                (
                    self.calibration_delta_moderate_mm,
                    self.calibration_delta_high_mm,
                    self.calibration_delta_severe_mm,
                ),
            ),
            (
                "depth_confidence_shift",
                (
                    self.depth_confidence_shift_moderate,
                    self.depth_confidence_shift_high,
                    self.depth_confidence_shift_severe,
                ),
            ),
            (
                "fail_closed_rate",
                (
                    self.fail_closed_rate_moderate,
                    self.fail_closed_rate_high,
                    self.fail_closed_rate_severe,
                ),
            ),
            (
                "hand_eye_residual_trend",
                (
                    self.hand_eye_residual_trend_moderate_mm,
                    self.hand_eye_residual_trend_high_mm,
                    self.hand_eye_residual_trend_severe_mm,
                ),
            ),
            (
                "verification_residual_trend",
                (
                    self.verification_residual_trend_moderate_mm,
                    self.verification_residual_trend_high_mm,
                    self.verification_residual_trend_severe_mm,
                ),
            ),
        )
        for name, (m, h, s) in ladders:
            if not (m <= h <= s):
                raise ValueError(
                    f"watchdog.{name}_* thresholds must satisfy "
                    f"moderate <= high <= severe; got ({m}, {h}, {s})"
                )
        # OOD ladder is inverted (smaller score => higher severity).
        if not (
            self.ood_score_moderate
            >= self.ood_score_high
            >= self.ood_score_severe
        ):
            raise ValueError(
                "watchdog.ood_score_* thresholds must satisfy "
                "moderate >= high >= severe; got "
                f"({self.ood_score_moderate}, "
                f"{self.ood_score_high}, "
                f"{self.ood_score_severe})"
            )
        if "easy" in self.block_modes:
            raise ValueError(
                "watchdog.block_modes must not include 'easy'; "
                "easy mode is permanently excluded from U+ "
                "behavioural layers."
            )
        unknown = [m for m in self.block_modes if m not in _KNOWN_GRASP_MODES]
        if unknown:
            raise ValueError(
                "watchdog.block_modes contains unknown grasp mode(s): "
                f"{unknown!r}; valid: {sorted(_KNOWN_GRASP_MODES)!r}"
            )
        if len(set(self.block_modes)) != len(self.block_modes):
            raise ValueError(
                "watchdog.block_modes must be unique; got "
                f"{self.block_modes!r}"
            )
        return self


class GraspingPerformanceConfig(StrictModel):
    """Runtime SLO budgets and bounded-compute knobs.

    Carries the per-stage p95 latency budgets the runtime measures against, plus
    timeouts and fallback policies that are plumbed end to end and not enforced:
    enforcement waits on the live success-probability model and the fusion path,
    so the timeouts are advisory. Per-stage latency is captured by a typed
    :class:`src.robot.grasping.telemetry.latency_tracker.LatencyTracker` seam owned
    by :class:`AutonomousGraspService`; a stage that does not run has an absent
    (``None``) latency and the SLO gate skips nulls. A breach emits a typed
    ``RobotWatchdogEvent.SLO_BREACH`` and lets the listener decide; the runtime
    never fail-closes on its own SLO.

    Attributes
    ----------
    enabled
        Master switch. When ``False`` the runtime still records latency telemetry
        but does not emit breach events or apply bounded-compute fallbacks. Default
        ``False`` is byte-identical.
    decision_latency_slo_ms / ranking_latency_slo_ms / fusion_latency_slo_ms
        Per-stage p95 budgets enforced by the SLO gate.
    decision_timeout_ms / ranking_timeout_ms / fusion_timeout_ms
        Hard per-stage caps. ``0.0`` means no cap and is the default; a non-zero
        value is honoured once the producing stage enforces it, and is a telemetry
        budget until then.
    model_fallback_policy
        How the runtime degrades when the success-probability model breaches its
        timeout: ``"legacy_score"`` reverts to geometry-only ranking, ``"skip"``
        omits the probability signal, ``"fail_closed"`` fail-closes with a typed
        reason.
    fusion_fallback_policy
        How the runtime degrades when fusion breaches its timeout: ``"skip"``
        proceeds with single-view evidence, ``"fail_closed"`` fail-closes.
    emit_breach_events
        When ``True`` and ``enabled``, the service emits a
        ``RobotWatchdogEvent.SLO_BREACH`` on rising-edge p95 breaches. Default
        ``False``.
    breach_window_size
        Rolling window length (in attempts) for the in-runtime p95 that drives
        ``SLO_BREACH`` emission. The replay/soak ``--slo-gate`` computes p95 across
        the full pack.
    """

    enabled: bool = Field(default=False)
    decision_latency_slo_ms: float = Field(default=60.0, gt=0.0)
    ranking_latency_slo_ms: float = Field(default=80.0, gt=0.0)
    fusion_latency_slo_ms: float = Field(default=220.0, gt=0.0)
    decision_timeout_ms: float = Field(default=0.0, ge=0.0)
    ranking_timeout_ms: float = Field(default=0.0, ge=0.0)
    fusion_timeout_ms: float = Field(default=0.0, ge=0.0)
    model_fallback_policy: str = Field(default="legacy_score", min_length=1)
    fusion_fallback_policy: str = Field(default="skip", min_length=1)
    emit_breach_events: bool = Field(default=False)
    breach_window_size: int = Field(
        default=32,
        ge=2,
        le=4096,
        json_schema_extra={
            "runtime_mutable": True,
            "min_value": 2,
            "max_value": 4096,
            "max_abs_step": 16,
            "max_rel_step": 0.50,
        },
    )

    _ALLOWED_MODEL_FALLBACKS: ClassVar[frozenset[str]] = frozenset(
        {"legacy_score", "skip", "fail_closed"}
    )
    _ALLOWED_FUSION_FALLBACKS: ClassVar[frozenset[str]] = frozenset(
        {"skip", "fail_closed"}
    )

    @model_validator(mode="after")
    def _validate(self) -> "GraspingPerformanceConfig":
        if self.model_fallback_policy not in self._ALLOWED_MODEL_FALLBACKS:
            raise ValueError(
                "performance.model_fallback_policy must be one of "
                f"{sorted(self._ALLOWED_MODEL_FALLBACKS)!r}; got "
                f"{self.model_fallback_policy!r}"
            )
        if self.fusion_fallback_policy not in self._ALLOWED_FUSION_FALLBACKS:
            raise ValueError(
                "performance.fusion_fallback_policy must be one of "
                f"{sorted(self._ALLOWED_FUSION_FALLBACKS)!r}; got "
                f"{self.fusion_fallback_policy!r}"
            )
        # When a per-stage timeout is set, it must not exceed the
        # SLO budget for that stage: a timeout looser than the SLO
        # is silently useless and almost certainly a config mistake.
        pairs = (
            (
                "decision",
                self.decision_timeout_ms,
                self.decision_latency_slo_ms,
            ),
            (
                "ranking",
                self.ranking_timeout_ms,
                self.ranking_latency_slo_ms,
            ),
            (
                "fusion",
                self.fusion_timeout_ms,
                self.fusion_latency_slo_ms,
            ),
        )
        for name, timeout, slo in pairs:
            if timeout > 0.0 and timeout > slo:
                raise ValueError(
                    f"performance.{name}_timeout_ms ({timeout}) must "
                    f"not exceed performance.{name}_latency_slo_ms "
                    f"({slo})"
                )
        return self


class RobotGraspingCommitPolicyConfig(StrictModel):
    """Mandatory dense multi-view commit gate.

    When enabled (and the resolved mode is in :attr:`apply_modes`), the pick loop
    refuses to hand a winning candidate to execution unless the fused scene memory
    satisfies both ``views_accepted >= min_views_accepted`` and the candidate's
    approach corridor (a :attr:`corridor_length_mm` x :attr:`corridor_radius_mm`
    capsule along its approach vector) has at least
    :attr:`min_corridor_hit_fraction` of its in-ROI voxels marked hit. On refusal
    the orchestrator triggers up to :attr:`max_reobserve_attempts` more captures
    before surfacing a typed
    :attr:`~src.robot.grasping.loop.pick_loop.PickOutcome.NO_COMMIT_INSUFFICIENT_FUSION`.
    Default ``enabled=False`` is byte-identical.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Top-level switch. When False the commit gate is a no-op "
            "(byte-identical)."
        ),
    )
    min_views_accepted: int = Field(
        default=2,
        ge=1,
        le=64,
        description=(
            "Minimum number of accepted views that must already be "
            "fused before the orchestrator may commit a candidate to "
            "execution."
        ),
    )
    min_corridor_hit_fraction: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum fraction of voxels inside the candidate's "
            "approach corridor that must be marked hit in the fusion "
            "grid. 0.0 disables the corridor check (views-only gate)."
        ),
    )
    corridor_radius_mm: float = Field(
        default=25.0,
        gt=0.0,
        le=200.0,
        description=(
            "Radius (mm) of the cylindrical capsule around the "
            "candidate's approach line used for corridor evidence."
        ),
    )
    corridor_length_mm: float = Field(
        default=120.0,
        gt=0.0,
        le=1_000.0,
        description=(
            "Length (mm) of the corridor capsule along the candidate's "
            "approach vector, anchored at the grasp position and "
            "extending opposite the approach (i.e. into the camera-"
            "viewing volume where evidence should accumulate)."
        ),
    )
    max_reobserve_attempts: int = Field(
        default=1,
        ge=0,
        le=10,
        description=(
            "Per-pick budget of additional perception captures "
            "triggered when the gate refuses. After exhaustion the "
            "pick surfaces NO_COMMIT_INSUFFICIENT_FUSION."
        ),
    )
    apply_modes: tuple[str, ...] = Field(
        default=("auto", "dense_clutter", "dense_autonomous"),
        description=(
            "Mode labels that activate the gate. Defaults to all modes "
            "except easy. The orchestrator skips the gate (and the "
            "reobserve loop) when its resolved mode label is not in this set."
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> "RobotGraspingCommitPolicyConfig":
        if not self.apply_modes:
            raise ValueError(
                "commit_policy.apply_modes must contain at least one "
                "mode label"
            )
        if len(set(self.apply_modes)) != len(self.apply_modes):
            raise ValueError(
                "commit_policy.apply_modes must not contain duplicates"
            )
        if "easy" in self.apply_modes:
            raise ValueError(
                "commit_policy.apply_modes must not include 'easy' "
                "(locked Q5: easy is exempt from mandatory multi-view "
                "decisioning)"
            )
        return self


class RobotGraspingApproachValidationConfig(StrictModel):
    """Dense-mode swept-volume approach/retreat validation.

    When ``enabled`` (and the resolved mode is in :attr:`apply_modes`), the orchestrator validates each
    ranked candidate's approach/retreat sweep against the scene neighbour cloud (the calculator only
    checks the final grasp pose) and executes the first sweep-clear candidate, refusing with
    ``APPROACH_PATH_BLOCKED`` if every candidate is blocked. Default ``enabled=False`` is byte-identical;
    ``easy`` and single-object picking are exempt (``apply_modes`` excludes ``'easy'``).
    """

    enabled: bool = Field(
        default=False,
        description="Top-level switch. When False the approach validator is a no-op (byte-identical).",
    )
    standoff_mm: float = Field(
        default=80.0, ge=0.0, le=500.0,
        description="Pre-grasp standoff (mm) along -approach; the sweep is sampled standoff -> grasp.",
    )
    retreat_mm: float = Field(
        default=100.0, ge=0.0, le=500.0,
        description="Retreat distance (mm) after the grasp; the retreat leg is swept along +Z.",
    )
    num_approach_samples: int = Field(
        default=6, ge=0, le=64,
        description="Swept samples on the approach leg (0 disables the approach check).",
    )
    num_retreat_samples: int = Field(
        default=4, ge=0, le=64,
        description="Swept samples on the retreat leg (0 disables the retreat check).",
    )
    collision_margin_mm: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Extra margin (mm) added to the swept gripper box when testing collisions.",
    )
    apply_modes: tuple[str, ...] = Field(
        default=("dense_clutter", "dense_autonomous"),
        description=(
            "Mode labels that activate the validator (the dense modes by default). The orchestrator "
            "skips the check when its resolved mode label is not in this set. easy is locked out."
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> "RobotGraspingApproachValidationConfig":
        if not self.apply_modes:
            raise ValueError("approach_validation.apply_modes must contain at least one mode label")
        if len(set(self.apply_modes)) != len(self.apply_modes):
            raise ValueError("approach_validation.apply_modes must not contain duplicates")
        if "easy" in self.apply_modes:
            raise ValueError("approach_validation.apply_modes must not include 'easy'")
        return self


class CameraExtrinsicsConfig(StrictModel):
    """One camera's calibration entry in the central ``fusion.cameras`` map.

    Keyed by a camera id (the same name used elsewhere for the camera, e.g. a sim
    ``robot.sim.cameras`` key or a production rig id). Each camera declares how it is
    mounted and where its persisted calibration artifact lives, so every camera in a
    multi-view rig is calibrated + loaded individually.
    """

    enabled: bool = Field(
        default=True,
        description="Whether this camera participates. Disabled cameras are skipped by the resolver map.",
    )
    mounting_mode: Literal["eye_to_hand", "eye_in_hand"] = Field(
        default="eye_to_hand",
        description=(
            "eye_to_hand = fixed camera; the artifact is a CAMERA->BASE Extrinsics "
            "(``save_extrinsics``) -> a StaticCameraToBaseResolver. eye_in_hand = wrist "
            "camera; the artifact is a CAMERA->TOOL transform (``save_cam_to_tool``) -> an "
            "EyeInHandFrameResolver that composes the live TCP each frame."
        ),
    )
    extrinsics_artifact_path: str = Field(
        description="Path to this camera's persisted calibration JSON (per mounting_mode).",
    )


class FusionGeometryConfig(StrictModel):
    """Fuse each object's surface across the fixed cameras and feed that to the grasp generator.

    A separate switch from ``fusion.enabled``: that one turns on the voxel-occupancy substrate,
    which is shadow-only, emitting telemetry no grasping path consumes. This one changes the grasp
    candidates themselves, since every camera that can identify an object contributes its view of
    that object's surface and the generator plans on the union.

    A candidate is accepted or rejected on its closing axis, the closing axis comes from the
    object's silhouette, and one depth view sees one side of an object while an antipodal grasp
    needs two. Measured on the datagen reference through the real calculator (n=354): top-1
    43.50 % single-view -> 55.93 % fused, coverage 53.67 -> 64.69 %.

    Requires ``cameras`` to be populated (each camera individually calibrated) and a multi-camera
    perception source at runtime. Without both this stands down to the single-view path and stamps
    why in the telemetry.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Top-level switch. False (default) = the generator sees one view per object, exactly as "
            "before: byte-identical. Independent of ``fusion.enabled`` (the shadow voxel substrate)."
        ),
    )
    metric: Literal["overlap", "box_iou", "centroid"] = Field(
        default="overlap",
        description=(
            "How 'the same object' is scored across cameras. Measured over 1240 answerable decisions "
            "and 488 where the target was absent from the other view: `overlap` gets 92.0 % of the "
            "answerable ones right at 0.08 % wrong and correctly refuses 98.2 % of the impossible "
            "ones. `centroid` looks better on the answerable half alone (99.5 %) and is the trap: "
            "it never abstains, so when the object is absent it takes a neighbour instead in up to "
            "73 % of cases, welding a neighbour's far surface into the cloud the generator plans on."
        ),
    )
    min_score: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description=(
            "Match score a camera must clear to contribute. Below it the camera contributes nothing "
            "for that object, exactly as an occluded camera does. Fusing the wrong object is worse "
            "than fusing one fewer view: it invents a contact face the generator cannot distinguish "
            "from a real one."
        ),
    )
    neighbour_mm: float = Field(
        default=12.0,
        gt=0.0,
        le=200.0,
        description=(
            "How near two points count as the same surface, for the `overlap` metric. Larger than a "
            "stereo camera's depth noise at working distance, smaller than the gap between two "
            "touching objects."
        ),
    )
    score_voxel_mm: float = Field(
        default=0.0,
        ge=0.0,
        le=50.0,
        description=(
            "Decimate each cloud to one point per this many mm, for the association scoring only. "
            "0.0 (default) = off, byte-identical. The fused cloud handed to the grasp generator is "
            "rebuilt from the original full-resolution clouds, so this makes the match decision "
            "cheaper without making the geometry coarser. `_overlap_fraction` states its own "
            "precondition, 'the clouds are a few thousand points at most', and a data-generation "
            "corpus violates it by an order of magnitude: one datagen object's cloud is 37,119 "
            "points and a single scene's fusion took 182.7 s. Applies to the `overlap` metric only; "
            "`centroid` and `box_iou` are already cheap and must not shift by a voxel because this "
            "is set."
        ),
    )
    max_centroid_mm: float = Field(
        default=150.0,
        gt=0.0,
        le=5000.0,
        description="Distance at which the `centroid` metric scores 0. Unused by the other metrics.",
    )
    on_camera_unavailable: Literal["degrade", "refuse"] = Field(
        default="degrade",
        description=(
            "What to do when a configured camera delivers no frame this pick (unplugged, dropped "
            "trigger, dirty lens). `degrade` (default) continues with the cameras that did deliver "
            "and logs a WARNING; `refuse` fails the pick closed. Either way the telemetry stamps "
            "which cameras actually contributed: a cell must never fall back to single-view "
            "silently, which is the whole failure this option exists to make visible."
        ),
    )
    neighbour_scene_enabled: bool = Field(
        default=False,
        description=(
            "Also give the collision filter what the other cameras see: each object gets every "
            "other camera's view of everything that is not itself, fed as `scene_points_mm`. "
            "Requires `enabled`. False (default) = byte-identical.\n\n"
            "`finger_collision` is 60.0 % of every mis-rank left after target fusion, and showing "
            "the filter the hidden neighbours does not close it. Measured over 1073 visible objects "
            "(D5 reference, GT masks), paired per object: 5 grasps rescued, 4 destroyed, net +3 "
            "picks in 1073. Candidate precision does rise 64.6 -> 70.4 % at a cost of ~1.6 ms, but "
            "the gap itself barely moves (45 -> 39 objects) and `finger_collision` is still 59.0 % "
            "of it.\n\n"
            "Replaying the reference verdict on the surviving rank-0 winners shows 60.9 % of them "
            "collide with a bin wall rather than with an object, and in the `bin` family it is 14 "
            "of 14. No camera segments a wall as an instance, so no amount of object fusion reaches "
            "one: wall geometry comes from `container.interior_min_mm` and `interior_max_mm` with "
            "`container.wall_collision_enabled` set, never from a camera. Where the blocker is an "
            "object the lever works as designed: the `pile` gap halves, 8 -> 4. So this is on for a "
            "pile or heap cell, and for a bin it needs the container walls declared as well."
        ),
    )
    neighbour_voxel_mm: float = Field(
        default=8.0,
        gt=0.0,
        le=100.0,
        description=(
            "Voxel size the fused neighbour cloud is thinned to. Defaults to the same 8.0 mm the "
            "calculator already applies to the same-view neighbours it builds itself "
            "(`geometry_voxel_size_mm`), so the two halves of the obstacle set arrive at one "
            "density rather than letting the fused half dominate purely by point count."
        ),
    )
    promote_unmatched: bool = Field(
        default=False,
        description=(
            "Let a camera introduce an object, instead of only confirming one the primary already "
            "found. Requires `enabled`. False, the default, is the behaviour every measured number "
            "in this repository was taken under.\n\n"
            "What it fixes. Detection and segmentation already run on every camera in a fused cell: "
            "each one costs a full detect and segment pass per pick. But the object list comes from "
            "the primary camera alone, and the other cameras' blobs are matched against it, so a "
            "part that only the second camera can see matches nothing, because there is nothing for "
            "it to match. It is found, segmented, back-projected, and then dropped. If it is the "
            "last part in a bin, the cell reports the bin empty while holding a positive detection "
            "of it in memory.\n\n"
            "What it costs. Grouping becomes symmetric over every camera pair rather than "
            "primary-against-each, and an object no primary segmentation corresponds to is computed "
            "in the camera that saw it, with that camera's mask, depth and intrinsics. So a cell "
            "holds one calculator per camera. The grouping uses complete linkage: a blob joins a "
            "group only when it clears `min_score` against every blob already in it, never merely "
            "against one. That splits where a chain would weld, which costs a duplicate object and "
            "one wasted attempt, rather than welding two parts into a cloud with a closing axis "
            "running through the gap between them.\n\n"
            "Not measured. How often a bin holds a part the primary cannot see is not known: no "
            "number in this repository answers it. That is why this ships off."
        ),
    )


class RobotGraspingFusionConfig(StrictModel):
    """Bounded multi-view fusion substrate config.

    Configures the deterministic voxel-occupancy grid that
    :class:`src.robot.grasping.multiview.fusion.SceneFusion` maintains
    across perception captures within a pick session. The deployment posture is
    shadow-only: even with ``enabled=True`` the runtime merely ingests perception
    frames and emits telemetry; the commit gate is what turns fused evidence into
    dense-mode decisioning.

    All limits are hard caps. Validators enforce positive/finite numerics,
    ``depth_min_mm < depth_max_mm``, per-axis ROI extent a positive multiple of
    ``voxel_size_mm``, and ROI x voxel-size never exceeding ``max_voxels`` (so a
    misconfiguration cannot allocate a gigabyte-scale grid).
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Top-level switch. When False the orchestrator behaves "
            "byte-identically: no fusion module is constructed, no "
            "telemetry emitted, no perception ingest performed."
        ),
    )
    max_views: int = Field(
        default=6,
        ge=1,
        le=64,
        description=(
            "Maximum number of accepted views retained simultaneously. "
            "Sliding FIFO: oldest accepted view evicted on overflow."
        ),
    )
    max_view_age_s: float = Field(
        default=20.0,
        gt=0.0,
        le=600.0,
        description=(
            "Accepted views older than this are dropped on the next "
            "ingest (wall-clock relative to the newest accepted view)."
        ),
    )
    voxel_size_mm: float = Field(
        default=6.0,
        gt=0.0,
        le=100.0,
        description="Cubic voxel edge length in millimetres.",
    )
    roi_extent_mm: tuple[float, float, float] = Field(
        default=(600.0, 600.0, 360.0),
        description=(
            "Total X/Y/Z extent (mm) of the fusion ROI, centered on "
            "the workspace origin. Each axis must be > 0 and an "
            "integer multiple of ``voxel_size_mm``."
        ),
    )
    max_voxels: int = Field(
        default=1_000_000,
        ge=1_000,
        le=8_000_000,
        description=(
            "Hard cap on total voxel count "
            "(``roi_x/voxel_size * roi_y/voxel_size * roi_z/voxel_size``)."
        ),
    )
    depth_min_mm: float = Field(
        default=80.0,
        gt=0.0,
        le=10_000.0,
        description="Minimum valid depth sample (mm). Below: rejected.",
    )
    depth_max_mm: float = Field(
        default=1_400.0,
        gt=0.0,
        le=10_000.0,
        description="Maximum valid depth sample (mm). Above: rejected.",
    )
    intrinsics_atol: float = Field(
        default=1e-6,
        ge=0.0,
        le=1.0,
        description=(
            "Absolute tolerance for per-element intrinsic-matrix drift "
            "between views (strict frame contract). Views whose "
            "intrinsics differ from the first accepted view by more "
            "than this are refused."
        ),
    )
    commit_policy: RobotGraspingCommitPolicyConfig = Field(
        default_factory=RobotGraspingCommitPolicyConfig,
        description=(
            "Mandatory dense multi-view commit gate. Defaults to "
            "``enabled=False`` (byte-identical)."
        ),
    )
    active_perception_use_fusion: bool = Field(
        default=False,
        description=(
            "Read-side wiring switch. When True the viewpoint planner "
            "may consume fused-state information gain when scoring next "
            "viewpoints. Default False leaves the planner path "
            "byte-identical, so the commit gate can be armed on its own."
        ),
    )
    extrinsics_artifact_path: str | None = Field(
        default=None,
        description=(
            "Path to a persisted eye-to-hand Extrinsics JSON artifact (CAMERA -> BASE), as written "
            "by ``src.calibration.serialization.save_extrinsics``. When set and ``enabled`` is "
            "True, ``AutonomousGraspService.from_robot_config`` builds a StaticCameraToBaseResolver from "
            "the loaded transform so the multi-view fusion substrate + the commit gate become "
            "reachable in production. ``None`` (default) builds no resolver (byte-identical). "
            "Fail-closed: a set-but-unloadable path raises at construction (a misconfigured fusion "
            "deployment must not silently run with an unreachable gate). Eye-in-hand cells leave this "
            "None and pass a ``frame_resolver`` kwarg in code (the live TCP-composed resolver cannot be "
            "serialized)."
        ),
    )
    cameras: dict[str, CameraExtrinsicsConfig] = Field(
        default_factory=dict,
        description=(
            "Every camera this cell fuses, keyed by the id it has in ``camera.cameras.rigs``, "
            "INCLUDING the primary. One list answers 'how many cameras does this cell have', and a "
            "simulator and a real cell mean the same thing by it. Each entry carries {enabled, "
            "mounting_mode, extrinsics_artifact_path}, so every camera is calibrated and loaded on "
            "its own. ``build_config_frame_resolvers`` turns it into a {camera_id -> FrameResolver} "
            "map (eye_to_hand -> StaticCameraToBaseResolver, eye_in_hand -> EyeInHandFrameResolver). "
            "Empty (default) is a single-camera cell using the ``extrinsics_artifact_path`` key "
            "beside this one. Fail-closed: an enabled camera with a missing or invalid artifact "
            "raises at construction. "
            "The primary used to be excluded, and the exclusion cost two things. The boot banner "
            "counts this map, so a correctly configured two-camera cell was told it had one and "
            "advised to add a second. And ``robot.sim.yaml`` listed its own primary while a real "
            "cell was documented not to, so the two profiles taught opposite shapes. The primary is "
            "still left out of the extra-camera rig, because it already streams through the cell's "
            "main perception source and fusing a second copy of it costs a full detect and segment "
            "pass for a view already present. If the primary is listed here, its artifact must be "
            "the same one ``fusion.extrinsics_artifact_path`` names; a disagreement is refused at "
            "load."
        ),
    )
    geometry: FusionGeometryConfig = Field(
        default_factory=FusionGeometryConfig,
        description=(
            "Fuse each object's surface across ``cameras`` and hand the union to the grasp "
            "generator. Independent of ``enabled`` above: that switch runs the shadow voxel "
            "substrate, this one changes the grasp candidates."
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> "RobotGraspingFusionConfig":
        if self.depth_min_mm >= self.depth_max_mm:
            raise ValueError(
                "fusion.depth_min_mm must be strictly less than "
                f"depth_max_mm; got {self.depth_min_mm} >= "
                f"{self.depth_max_mm}"
            )
        if len(self.roi_extent_mm) != 3:
            raise ValueError(
                "fusion.roi_extent_mm must have exactly 3 components "
                f"(x, y, z); got {self.roi_extent_mm!r}"
            )
        if any((not isinstance(v, (int, float))) or v <= 0.0 for v in self.roi_extent_mm):
            raise ValueError(
                "fusion.roi_extent_mm components must all be > 0; got "
                f"{self.roi_extent_mm!r}"
            )
        vs = float(self.voxel_size_mm)
        eps = 1e-9
        for axis_label, extent in zip("xyz", self.roi_extent_mm):
            ratio = float(extent) / vs
            if abs(ratio - round(ratio)) > eps:
                raise ValueError(
                    f"fusion.roi_extent_mm[{axis_label}]={extent} must "
                    f"be an integer multiple of voxel_size_mm={vs}"
                )
        total_voxels = 1
        for extent in self.roi_extent_mm:
            total_voxels *= int(round(float(extent) / vs))
        if total_voxels > self.max_voxels:
            raise ValueError(
                "fusion ROI/voxel combo would allocate "
                f"{total_voxels} voxels which exceeds max_voxels="
                f"{self.max_voxels}. Coarsen voxel_size_mm or shrink "
                "roi_extent_mm."
            )
        return self


class GraspingParallelJawGeometryConfig(StrictModel):
    """Parallel-jaw collision envelope (mm) in the local grasp frame (X closes, Y binormal, Z approach).

    Every value flows straight into
    :class:`src.robot.grasping.collision.ParallelJawGripperModel`. The finger dimensions are
    measured off Isaac's Robotiq 2F-85 collision shapes; the palm dimensions are not measured.
    """

    # Measured off the 2F-85's own collision shapes in this frame, swept across the whole aperture
    # range because the finger swings as the jaw moves: closing it pushes the tip forward, so the
    # reach toward the table is largest at a narrow aperture, exactly the case of a thin object
    # gripped close to the table. Each value below is the worst case over apertures
    # 10/25/40/55/70/82 mm, which is what a conservative envelope needs.
    #
    # These values describe the whole finger link, not the rubber pad: the pad alone is
    # 19.16 mm thick against the link's 31.35, and 22.0 mm wide against the link's 27.0.
    # ``fingertip_depth_mm`` is the one that decides table clearance, because it says how far the
    # finger reaches past the grasp point along the approach, i.e. toward the support surface.
    finger_length_mm: float = Field(default=39.98, gt=0.0, le=500.0)
    finger_thickness_mm: float = Field(default=31.35, gt=0.0, le=200.0)
    finger_width_mm: float = Field(default=27.0, gt=0.0, le=200.0)
    finger_pad_overlap_mm: float = Field(default=2.0, ge=0.0, le=100.0)
    fingertip_depth_mm: float = Field(default=28.72, gt=0.0, le=200.0)
    #: The contact patch along the approach, the part of the finger that actually touches. Measured
    #: 38.0 mm and constant across the aperture range. It is not the finger: the pad runs from 14.4 mm
    #: behind the grasp centre to 23.6 mm in front of it, so a grasp is not symmetric about its own
    #: contact. Distinct from ``fingertip_depth_mm`` (the whole link's reach) because antipodality and
    #: table clearance are decided by the patch, not by the housing around it.
    pad_length_mm: float = Field(default=38.0, gt=0.0, le=300.0)
    #: How far the patch reaches in front of the grasp centre. Measured 23.61 mm, and not derived
    #: from ``fingertip_depth_mm``: the finger housing extends 5.11 mm past the rubber, so deriving
    #: it would place the whole patch that far forward. The remainder (pad_length - pad_ahead)
    #: reaches back toward the wrist.
    pad_ahead_mm: float = Field(default=23.61, gt=0.0, le=300.0)
    # Not measured: the palm was never probed, so these are the shipped values rather than a
    # measurement.
    palm_depth_mm: float = Field(default=35.0, gt=0.0, le=500.0)
    palm_width_mm: float = Field(default=70.0, gt=0.0, le=500.0)


class GraspingSuctionCupGeometryConfig(StrictModel):
    """Suction-cup collision envelope (mm): cup, then shaft, then wrist mount back along -approach.

    Feeds :class:`src.robot.grasping.collision.SuctionCupGripperModel`; defaults model a
    ~30 mm-diameter cup on a slim shaft with a wrist coupler.
    """

    cup_radius_mm: float = Field(default=15.0, gt=0.0, le=200.0)
    cup_height_mm: float = Field(default=25.0, gt=0.0, le=200.0)
    contact_tip_depth_mm: float = Field(default=2.0, ge=0.0, le=100.0)
    shaft_radius_mm: float = Field(default=10.0, gt=0.0, le=200.0)
    shaft_length_mm: float = Field(default=40.0, gt=0.0, le=500.0)
    mount_radius_mm: float = Field(default=30.0, gt=0.0, le=300.0)
    mount_depth_mm: float = Field(default=20.0, gt=0.0, le=300.0)



class GraspingContainerConfig(StrictModel):
    """Optional bin, tray or KLT the objects sit inside. Everything here defaults to "no container".

    Give it when the parts do not rest on the table itself. A KLT standing on the workspace raises
    the surface its contents rest on by the thickness of its own floor, and a grasp planner that
    thinks the support is the table will allow a finger that many millimetres too low.

    ``floor_height_mm`` is the inside floor in BASE, the surface you would touch if you reached
    into the empty bin. It is not the height of the bin, and not the rim.

    The walls are the other half. The candidate filter checks a target cloud, the neighbours'
    clouds and a support plane, and a plane is a floor, so a wall stays invisible to it until one
    is declared here and ``wall_collision_enabled`` turns the check on. Replaying the reference
    verdict on the rank-0 winners that lose the ranking gap on ``finger_collision``: 60.9 % of them
    hit a bin wall, and in the bin family it is 14 of 14. A wall is not an instance any detector
    segments, so no camera can close that and it has to be declared, which is what
    ``interior_min_mm`` / ``interior_max_mm`` are for.
    """

    floor_height_mm: float | None = Field(
        default=None,
        description=(
            "Height in BASE mm of the surface inside the container that parts rest on. None (default) "
            "-> the workspace surface is used, i.e. no container."
        ),
    )
    interior_min_mm: tuple[float, float, float] | None = Field(
        default=None,
        description=(
            "Lower corner of the container's interior box in BASE mm (x, y, z), the empty volume "
            "you could pour parts into. z is the inside floor, the same surface as "
            "`floor_height_mm`. None (default) -> no wall geometry."
        ),
    )
    interior_max_mm: tuple[float, float, float] | None = Field(
        default=None,
        description=(
            "Upper corner of the container's interior box in BASE mm (x, y, z). z is the rim, the "
            "height a finger may pass over from outside. Must exceed `interior_min_mm` on every axis."
        ),
    )
    wall_collision_enabled: bool = Field(
        default=False,
        description=(
            "Treat the four vertical walls of that interior box as collision geometry: they are "
            "sampled into points and appended to what the candidate filter already checks against. "
            "Separate from declaring the box so a cell can describe its bin without changing "
            "grasping behaviour. Fail-closed: setting this without both interior corners raises at "
            "config load, because a cell that believes its walls are protected and is not is worse "
            "than one that knows they are not. False (default) = byte-identical.\n\n"
            "Measured (354 jaw-graspable objects, D5 reference): top-1 55.9 -> 56.2 %, which is one "
            "object on a run that is not bit-reproducible, so flat. Do not turn this on expecting a "
            "better pick rate. What it does move: candidate precision 64.6 -> 80.7 %, and the `bin` "
            "family's ranking gap from 15 objects to 1 (to 0 with the fused neighbour cloud as "
            "well). It converts 'the ranker chose a grasp that hits the wall' into 'the ranker "
            "offered nothing', which is neutral in a simulator and is the difference between a "
            "retry and a collision on real hardware."
        ),
    )
    wall_thickness_mm: float = Field(
        default=5.0,
        gt=0.0,
        le=100.0,
        description=(
            "How far outward from each interior face the wall material is sampled. Sampling only the "
            "inner face would leave a finger standing inside a thick wall undetected unless the "
            "collision margin happened to exceed half the thickness; that is a coincidence, not a "
            "check. A KLT is typically 3-8 mm."
        ),
    )
    wall_sample_mm: float = Field(
        default=8.0,
        gt=0.0,
        le=100.0,
        description=(
            "Point spacing on the sampled walls. Matches the calculator's own "
            "`geometry_voxel_size_mm` default so wall points arrive at the same density as the "
            "neighbour points they are checked alongside."
        ),
    )

    @model_validator(mode="after")
    def _walls_need_a_box(self) -> "GraspingContainerConfig":
        if not self.wall_collision_enabled:
            return self
        if self.interior_min_mm is None or self.interior_max_mm is None:
            raise ValueError(
                "container.wall_collision_enabled requires both interior_min_mm and "
                "interior_max_mm: there is no default bin to guess at"
            )
        for axis, (low, high) in enumerate(zip(self.interior_min_mm, self.interior_max_mm)):
            if high <= low:
                raise ValueError(
                    f"container.interior_max_mm[{axis}] ({high}) must exceed "
                    f"interior_min_mm[{axis}] ({low})"
                )
        return self


class GraspingDeepGeneratorConfig(StrictModel):
    """The learned 6-DoF grasp generator, what `grasping.calculator: deep` selects.

    Different from `deep_ranker` in the one way that matters: the ranker scores candidates the
    analytic stack proposed and changes no order, this one proposes them instead of the analytic
    stack. There is no shadow mode for a generator, either it is the source of candidates or it is
    not running.

    It fails closed. `calculator: deep` with no readable artifact refuses to build the cell rather
    than falling back to `geometric`: a cell that asked for the learned generator and quietly got
    the analytic one would file the analytic one's numbers under the learned one's name.

    The score it attaches is not a probability. The v1 net has no calibrated P(hold) head: the
    corpus carries no physics until it is shaken, and `held_jaw_v1` already ranks proposals at
    0.8947 AUROC. Generator proposes, scorer ranks, and each is promoted on its own evidence.
    """

    artifact_path: str | None = Field(
        default=None,
        description=(
            "Path to the generator's `.pt`, written by "
            "`python -m src.robot.grasping.deep train`. The card beside it is committed; the "
            "weights are not, exactly as the ranker's trees are not. Null (default) means the "
            "learned generator cannot be selected; `calculator: deep` then refuses. "
            "Both generator families are accepted here: a `grasp_generator` from "
            "`deep train` and a `set_grasp_generator` from `deep train-set`. The calculator "
            "branches on the artifact's own `kind`, so the file decides which decoder runs."
        ),
    )
    device: str | None = Field(
        default=None,
        description=(
            "Torch device for inference. Null picks CUDA when available and CPU otherwise. Named "
            "explicitly only by a cell that has a reason: a second GPU, or a deliberate CPU run."
        ),
    )
    #: Which hand this cell has, for a model fitted across several (`narrow_55`, `slim_pad`,
    #: `wide_140`). The choice is not cosmetic: measured on `arm_gripper`, mean proposed width
    #: is 32.69 mm for the 55 mm hand, 43.31 for 85 and 70.46 for 140.
    #:
    #: `None` keeps the artifact's stamp, so an existing cell is byte-identical. Naming a hand the
    #: artifact never trained across is refused rather than served, because a conditioning vector the
    #: model has not seen produces grasps that read as a bad model rather than as a wrong hand.
    #: Names come from `deep/net/gripper.py::JAW_GEOMETRY`.
    gripper: str | None = Field(
        default=None,
        description=(
            "Which JAW_GEOMETRY hand this cell has, for the deep generator's conditioning vector. "
            "None uses the hand the artifact was stamped for. A hand the artifact never trained "
            "across is refused."),
    )
    minimum_score: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description=(
            "Points below this graspability score are never seeded. 0.5 is the natural threshold for "
            "a calibrated head and this head is not calibrated, so it is a knob rather than a "
            "constant: the first real run is expected to move it once the score distribution has been "
            "looked at rather than assumed."
        ),
    )


class GraspingDeepRankerConfig(StrictModel):
    """The learned grasp ranker, in shadow: it computes, the telemetry records, the order never changes.

    Measured on 1,611 (scene, view, object) units of `v1_proof`, out of fold and grouped by object
    so no scored object was ever trained on:

        random (expected)   54.4 %      the calculator's own order   51.6 %
        this ranker         84.1 %      best possible               100.0 %

    "top-1 is valid": the first candidate physically closes. Not "the grasp holds": the label is
    the analytic verdict this stack computes, and the pilot's hold rate is 5.62 % where validity
    there is 48.8 %. Those are different questions and only a shake answers the second, which is
    why nothing here promotes anything.

    Shadow, for a measured reason. A ranker fitted on aletheia's shaken corpus beat the calculator
    by 17.7 pp on the same units and still lost to picking at random, because the failure it has to
    predict is `finger_collision` and its features cannot see the surroundings. What this logs is
    ``would_change_top1``: whether turning it on would do anything, answered on every real pick.

    Enabling it adds telemetry keys and changes no decision. That is what the wiring guard checks: a
    flag whose runtime is unchanged fails, and telemetry is the runtime here.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Score every candidate with the learned ranker and record what it would have chosen. "
            "Changes no ordering and no decision; adds deep_ranker_* telemetry keys."
        ),
    )
    artifact_dir: str = Field(
        default="assets/models/grasp_ranker/v1", min_length=1,
        description=(
            "Where the fitted ranker lives. The trees are gitignored and regenerated with "
            "`python -m datagen train-ranker`; the card beside them is committed."
        ),
    )
    spec: str = Field(
        default="held_jaw_v1", min_length=1,
        description=(
            "Which feature contract to score under. Must match the artifact: the scorer refuses a "
            "mismatch rather than scoring a vector the model was never fitted on. Two ship: "
            "`held_jaw_v1` predicts whether Isaac's shake could not pull the object out, and "
            "`valid_jaw_v1` predicts this stack's own analytic verdict, which makes anything "
            "measured with it self-referential. Measured on identical rows, folds and features "
            "with only the target changed, both graded on held: 0.8947 against 0.7749 AUROC, "
            "and on the `too_wide` stratum the valid-trained model scores 0.5220, a coin toss."
        ),
    )


class GraspingSupportConfig(StrictModel):
    """Where the surface is that the parts stand on, the input the table-clearance check has
    never been given.

    ``compute()`` takes a ``support_plane`` and ``collision_checker.py`` skips the whole check
    without one: with none passed, ``rejected_table`` was 0 in 2128 of 2128 telemetry lines while
    the independent reference rejected 37 % of the same candidates for driving a finger under the
    support. This block is what supplies it.

    The height is a declared number refined by what the cameras see, and the higher of the two wins.
    That combination is measured, not chosen. Over 1073 reference objects:

    ======================  =========================  ===========================
    estimator               on the table (n=1052)      standing on something (n=21)
    ======================  =========================  ===========================
    declared height alone   median +0.00, 0 % too low  median -14.08, 100 % too low
    target footprint        median +1.81, 0 % too low  median +0.89, 0 % too low
    plane fit near the      median +19.97              median -1.19, 33 % too low
    object / whole scene    median +14.75              median +9.41, 33 % too low
    ======================  =========================  ===========================

    A declared height cannot know that an object is standing on another object, and there it is too
    low every single time. Both plane fits are worse still: they go too low on a third of the
    stacked cases, which is the direction that breaks a gripper. The target's own lowest point was
    never too low in any of the 1073 cases, so taking the higher of the two can only ever be
    conservative.

    The footprint is fused across cameras, and that is also measured. A single view of a bin sees
    the object over the wall, so its lowest visible point is the rim rather than the base: p90 error
    50.6 mm in the ``bin`` family, 38 % of objects more than 10 mm too high. Taking the lowest point
    over all cameras that identified the object cuts that to 13.3 mm and 10.6 %, and stays at 0 %
    too low everywhere. What one camera cannot see behind a wall, another can.

    Known limit. Even fused, the footprint still reads high in a bin (p90 13.3 mm against 1.7 mm in
    the open). Since the higher value wins, that costs some valid low grasps in exactly the family
    that is already hardest. Declaring ``container.floor_height_mm`` does not fix it: the floor is
    the lower of the two and the fused estimate still wins. Closing it needs either a fourth camera
    low enough to see the base, or a rule that recognises an estimate as wall-occluded; the second
    has not been measured, so it is not implemented.
    """

    height_mm: float = Field(
        default=0.0,
        description=(
            "The workspace surface in BASE mm, the table. Objects in a container use "
            "container.floor_height_mm instead when it is given."
        ),
    )
    normal: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 1.0),
        description=(
            "Surface normal in BASE. Non-vertical describes a tilted tray or a slope; the closing-axis "
            "levelling and the clearance check both honour it."
        ),
    )
    refine_from_target: bool = Field(
        default=True,
        description=(
            "Raise the declared height to the target's own lowest fused point when that is higher: "
            "the only way to know an object is standing on another object. Never lowers it."
        ),
    )
    container: GraspingContainerConfig = Field(
        default_factory=GraspingContainerConfig,
        description="Optional bin/tray the parts rest in. Default: none, the workspace surface is used.",
    )
    min_clearance_mm: float = Field(
        default=5.0,
        ge=0.0,
        le=200.0,
        description=(
            "How far the gripper envelope must stay above the support surface. Measured, not chosen: "
            "swept 0 / 2 / 5 mm over the gate subset (n=354), the strictness is nearly free: "
            "precision 18.87 -> 20.00 -> 22.08 %, coverage 28.81 -> 28.53 -> 28.53 %, top-1 flat at "
            "~24.3 % throughout. 5 mm buys 3.2 pp of precision for 0.28 pp of coverage, so it stays "
            "the default and is the safer value on real hardware. Lowering it does not unlock flat "
            "parts: with the 2F-85's 28.72 mm fingertip reach an object must be >= 33.72 mm tall for a "
            "top-down grasp to clear 5 mm even when the grasp sits on its very top edge, and 34.1 % "
            "of the reference objects are shorter than that. That is gripper geometry, not a "
            "threshold: those parts need suction or a different end-effector."
        ),
    )


class GraspingGeometryStageConfig(StrictModel):
    """Which geometry stage proposes grasp candidates.

    ``support_footprint`` (the default) reconstructs the target as its visible footprint extruded
    onto the calibrated support plane and then solves for an anchor height that clears the surface.
    ``silhouette`` anchors on the visible surface and lets the collision filter throw the result
    away, which cannot work when the constraint is the support surface: a filter can refuse, it
    cannot relocate a grasp.

    Measured through the real calculator on the D5 gate subset (n=354), against ground-truth geometry:

    =====================  ===========  ==========  ==========
    stage                  precision    top-1       coverage
    =====================  ===========  ==========  ==========
    silhouette             22.08 %      24.29 %     28.53 %
    support_footprint      63.59 %      43.50 %     53.67 %
    support_footprint      64.60 %      55.93 %     64.69 %
    + fused target cloud
    =====================  ===========  ==========  ==========

    Per family, top-1 against the silhouette stage: packed 26.6 -> 44.1 %, pile 11.1 -> 35.2 %,
    sparse 30.3 -> 55.6 %. ``bin`` went 12.5 -> 8.3 % on this subset (n=24, two objects against
    three); on the full 270-scene set (n=152) the same stage measured 27.6 % against 14.5 %.

    It needs a BASE-frame support plane and a CAMERA->BASE transform. Without both it cannot place
    the table, so it does not run and the silhouette stage stands: a runner that builds the service
    through ``from_components`` without a frame resolver gets the silhouette stage, and the
    ``geometry_stage`` telemetry key says which one ran.
    """

    stage: Literal["support_footprint", "silhouette"] = Field(
        default="support_footprint",
        description=(
            "Which stage proposes candidates. 'silhouette' anchors on the visible surface instead."
        ),
    )
    inflate_mm: float = Field(
        default=0.0,
        ge=0.0,
        le=50.0,
        description=(
            "Push every reconstructed face outward by this much. A depth image loses the grazing rim "
            "of a curved surface, so the measured footprint is systematically smaller than the "
            "object and every consequence of that is one-sided (an under-estimated span, a finger "
            "that clips a flank on the way in). Non-zero biases the estimate in the safe direction. "
            "Default 0.0 = the measurement above, which was taken without it."
        ),
    )
    grasp_depth_reference: Literal["centre", "top"] = Field(
        default="centre",
        description=(
            "Where in the object's depth spread the silhouette stage anchors a grasp. 'centre' is "
            "the median of the measured surface under the mask. 'top' is a low quantile of it, the "
            "near face. Measured on the D5 gate subset (n=354, neighbours rung): centre reaches "
            "22.08 % precision and 24.29 % top-1, against 19.44 / 22.03 for top at q=25 and 18.24 / "
            "21.47 for top at q=10. 'top' does what it promises and it is not enough: rank-0 'no "
            "candidate' falls from 65.4 to 60.2 %, so more objects clear the table, while every "
            "failure reason rises with it. The 2F-85 contact patch reaches 14.4 mm behind the grasp "
            "centre and 23.6 mm in front, so anchoring on the near face puts most of the patch in "
            "the air above the object: it buys clearance by giving up grip. Keep 'centre' unless a "
            "measurement on your parts says otherwise."
        ),
    )
    grasp_top_penetration_mm: float = Field(
        default=0.0,
        ge=0.0,
        le=50.0,
        description=(
            "How far below the reference above the grasp is driven. Only read when "
            "grasp_depth_reference is 'top', where the reference is the near face and closing on "
            "the very edge of a part slips. Until now the perception producer applied a fixed 3 mm "
            "of this to the depth map itself before anything downstream saw it, so no cell could "
            "set it and every consumer that wanted the object's shape got a flat sheet instead. The "
            "descend belongs here, where it is one number applied to one candidate, rather than in "
            "an image that six other stages read for geometry."
        ),
    )
    grasp_top_quantile: float = Field(
        default=10.0,
        ge=0.0,
        le=50.0,
        description=(
            "Which quantile of the under-mask depth counts as the near face when "
            "grasp_depth_reference is 'top'. A percentile rather than the minimum, because one "
            "speckle pixel in front of the part would otherwise set the whole object's grasp plane. "
            "Only read when the reference is 'top'."
        ),
    )
    depth_band_mm: float | None = Field(
        default=None,
        gt=0.0,
        le=500.0,
        description=(
            "Keep only the depth within this much of the near face, and discard the rest before the "
            "cloud is built. A segmentation mask usually spills a few pixels past the object onto "
            "the bench, and those pixels are metres of surface behind it: they drag the median, "
            "inflate the object's apparent depth spread, and put table points into the geometry the "
            "fingers are planned against. Null, the default, keeps every pixel under the mask. This "
            "lever could not do anything at all while the producer flattened the depth, because "
            "every percentile of a constant is that constant."
        ),
    )
    depth_band_near_pct: float = Field(
        default=10.0,
        ge=0.0,
        le=50.0,
        description=(
            "Which quantile of the under-mask depth the band above is measured from. Only read when "
            "depth_band_mm is set."
        ),
    )


class GraspingGripperGeometryConfig(StrictModel):
    """Which gripper collision envelope the calculator filters grasp candidates against.

    ``kind`` picks the active model and the matching sub-block sizes it; ``outer_margin_mm`` inflates
    whichever is chosen. Defaults reproduce the built-in parallel-jaw envelope byte-for-byte, so
    leaving this unset changes nothing. Set ``kind: suction`` for a vacuum end-effector so
    suction candidates are checked against a cup envelope, not a finger envelope.
    """

    kind: Literal["parallel_jaw", "suction"] = Field(default="parallel_jaw")
    outer_margin_mm: float = Field(default=0.0, ge=0.0, le=100.0)
    parallel_jaw: GraspingParallelJawGeometryConfig = Field(
        default_factory=GraspingParallelJawGeometryConfig
    )
    suction: GraspingSuctionCupGeometryConfig = Field(
        default_factory=GraspingSuctionCupGeometryConfig
    )


class RobotGraspingConfig(StrictModel):
    """Vendor-neutral grasping-behaviour surface.

    Sits as a sibling of :class:`RobotSafetyConfig` under :class:`RobotConfig`.
    All knobs are behavioural (mode selection, closed-loop refinement,
    verification, recovery, decision); safety remains exclusively under
    ``robot.safety``.

    The runtime derives per-mode behaviour from the
    :class:`src.robot.execution.autonomous_grasp.GraspBehaviorProfile`
    table. Values here apply on top of that profile: they tighten or disable
    optional behaviour, but cannot relax safety limits or unlock recovery actions
    the mode profile forbids. ``default_mode`` is resolved against
    :class:`src.robot.grasping.types.modes.GraspMode` at use time so the config
    layer imports no runtime modules.
    """

    # A stable :class:`GraspMode` name resolved at runtime: "easy", "auto", "closed_loop",
    # "dense_clutter" or "dense_autonomous".
    default_mode: str = Field(default="auto", min_length=1)
    # Upper bound on autonomous attempts per pick() call.
    max_attempts: int = Field(default=5, ge=1, le=50)
    #: Which grasp generator ranks the candidates. See `src/robot/grasping/deep/` and
    #: `.ai-memory/dl-grasping-plan.md`.
    #:
    #: ``geometric`` is the analytic stack: silhouette or support-footprint candidate proposal,
    #: geometric scoring, the whole collision/IK/corridor tail.
    #:
    #: ``deep`` is the learned 6-DoF generator. It replaces the analytic proposal stage: the factory
    #: returns one generator or the other, and the deep decoder seeds only from its own graspability
    #: head, not from the analytic candidates. It has no rejection tail of its own; the table check,
    #: the gripper-collision envelope, the antipodal test and the corridor filter all live in the
    #: analytic generator, and extracting them into a stage both generators run is an open build.
    #: The safety preflight is unaffected: it sits at the arm and gates the joint target whatever
    #: proposed it, so a deep candidate meets exactly the same fail-closed guards.
    #:
    #: The default stays ``geometric`` until the learned one clears its pre-registered bar: beat
    #: sfe_fused top-1 55.9 % on the eval-grasps ladder, plus on-box dense-gate parity. Selecting
    #: ``deep`` without trained weights present fails closed with the artifact path in the message,
    #: so a cell can never quietly fall back to the other generator and leave "which one ran"
    #: unanswerable from the outside.
    calculator: Literal["geometric", "deep"] = Field(
        default="geometric",
        description=(
            "Which grasp generator ranks candidates. 'geometric' is the analytic stack; 'deep' is "
            "the learned 6-DoF generator, which additionally takes the geometric candidates as "
            "seeds. Fails closed when 'deep' is selected without weights."
        ),
    )
    #: When a silhouette is near-isotropic its principal axis is numerically degenerate, so aim the jaw
    #: along the radial direction (base -> grasp) instead of following the noise. This is a property of
    #: the arm, not of the scene: measured on-box, a UR3e real-vision (M2) cell goes 5/10 -> 10/10
    #: with it on, because a vision-detected grasp has a varying azimuth and a UR3e cannot plan a
    #: tangential top-down close at any azimuth (0/4 against 4/4 radial). A UR5e absorbs both, so the
    #: default stays False and only a cell that needs it declares it (robot.ur3e.yaml does).
    isotropic_radial_closing: bool = Field(default=False)
    record_log_path: str | None = Field(
        default=None,
        description=(
            "Opt-in JSONL path for production GraspAttemptRecord logging. When set, "
            "AutonomousGraspService.from_robot_config / from_components call "
            "enable_record_logging(path), so every pick() appends one record (the soak / KPI / RL "
            "data source). Default null -> off, byte-identical. The loader applies ${ENV} "
            'substitution, e.g. record_log_path: "${WILLY_RECORD_LOG:-}" -> an empty (unset) value '
            "is treated as off."
        ),
    )
    closed_loop: GraspingClosedLoopConfig = Field(
        default_factory=GraspingClosedLoopConfig
    )
    verification: GraspingVerificationConfig = Field(
        default_factory=GraspingVerificationConfig
    )
    dense_recovery: GraspingDenseRecoveryConfig = Field(
        default_factory=GraspingDenseRecoveryConfig
    )
    decision: GraspingDecisionConfig = Field(
        default_factory=GraspingDecisionConfig
    )
    feasibility: GraspingFeasibilityConfig = Field(
        default_factory=GraspingFeasibilityConfig
    )
    occlusion: GraspingOcclusionConfig = Field(
        default_factory=GraspingOcclusionConfig
    )
    ordering: GraspingOrderingConfig = Field(
        default_factory=GraspingOrderingConfig
    )
    recovery: GraspingRecoveryConfig = Field(
        default_factory=GraspingRecoveryConfig
    )
    uncertainty: GraspingUncertaintyConfig = Field(
        default_factory=GraspingUncertaintyConfig
    )
    success_model: GraspingSuccessModelConfig = Field(
        default_factory=GraspingSuccessModelConfig
    )
    watchdog: GraspingWatchdogConfig = Field(
        default_factory=GraspingWatchdogConfig
    )
    performance: GraspingPerformanceConfig = Field(
        default_factory=GraspingPerformanceConfig
    )
    fusion: RobotGraspingFusionConfig = Field(
        default_factory=RobotGraspingFusionConfig
    )
    approach_validation: RobotGraspingApproachValidationConfig = Field(
        default_factory=RobotGraspingApproachValidationConfig,
        description=(
            "Dense-mode swept-volume approach/retreat validator. Default disabled "
            "(byte-identical); independent of fusion (it needs only the per-frame neighbour masks)."
        ),
    )
    deep_generator: GraspingDeepGeneratorConfig = Field(
        default_factory=GraspingDeepGeneratorConfig,
        description=(
            "The learned 6-DoF grasp generator, selected by `calculator: deep`. Inert while "
            "`calculator` is 'geometric', which is the default, so this block changes nothing "
            "until a cell deliberately switches its generator."
        ),
    )
    deep_ranker: GraspingDeepRankerConfig = Field(
        default_factory=GraspingDeepRankerConfig,
        description=(
            "The learned grasp ranker in shadow mode. Default off; enabling it adds telemetry and "
            "changes no decision. See the class docstring for what it measured."
        ),
    )
    support: GraspingSupportConfig = Field(
        default_factory=GraspingSupportConfig,
        description=(
            "Where the parts stand. Nothing passes the calculator a support_plane today, and "
            "without one the table-clearance check is skipped; see the class docstring for "
            "the measurement behind "
            "the defaults."
        ),
    )
    geometry: GraspingGeometryStageConfig = Field(
        default_factory=GraspingGeometryStageConfig,
        description=(
            "Which stage proposes grasp candidates. Default 'support_footprint', measured 43.50 % "
            "top-1 against the silhouette stage's 24.29 % through the real calculator."
        ),
    )
    gripper_geometry: GraspingGripperGeometryConfig = Field(
        default_factory=GraspingGripperGeometryConfig,
        description=(
            "Gripper collision envelope the calculator filters grasp candidates against "
            "(parallel-jaw or suction-cup). Default parallel_jaw with the shipped dims -> byte-identical."
        ),
    )

    #: Switches whose value reaches ``EffectiveGraspingConfig``, the cell's own telemetry, and is
    #: then read back by nothing. Turning one on changes what the cell reports about itself and not
    #: one thing about what it does.
    #:
    #: Measured on-box in a 17-run campaign, four ways (see
    #: docs/grasping-config-reference.md section 6.1):
    #: ``apply_orchestrator_overlays`` installs 13 carriers and none of these blocks is among them;
    #: repo-wide nothing reads ``effective_config.feasibility`` / ``.corridor`` / ``.ordering``
    #: outside the file that writes them and the one that serialises them; feasibility's real
    #: consumer is a ``GraspCalculator`` constructor argument that no caller passes, not the sim
    #: runners and not ``real_cell/components.py``; and the positive control (``success_model``,
    #: read at ``pick_loop.py:922``) has no equivalent line here.
    #:
    #: It refuses rather than warns because an operator who sets one of these gets a cell that
    #: boots, runs, reports the flag as on, and behaves exactly as before: not a crash, but a false
    #: sense of a capability. The flags stay in the schema because the capability exists; what is
    #: missing is the wire from here to it. When one is wired, delete its entry and measure it.
    UNWIRED_SWITCHES: ClassVar[dict[str, str]] = {
        # `hard_reject_enabled` is what lets the occlusion score refuse a candidate rather than
        # only demote it, so it stays listed until the score itself is trusted: one run that moves
        # both the ranking and the rejection answers neither. `feasibility.*` and
        # `occlusion.directional_enabled` are wired instead, supplied per call from the
        # orchestrator rather than as a constructor argument; see the on-box regression warning
        # in builders.py before enabling feasibility.
        "occlusion.hard_reject_enabled": "occlusion",
        # `ordering.*` is wired by the other route: the orchestrator carries the ordering path
        # (`target_ordering` plus the selector at pick_loop.py:1801), and
        # `apply_orchestrator_overlays` maps the block onto `TargetOrderingConfig`
        # field-for-field under its `apply_modes`.
    }

    @model_validator(mode="after")
    def _refuse_unwired_switches(self) -> "RobotGraspingConfig":
        """Fail closed on a switch that would read as on and do nothing."""

        offenders: list[str] = []
        for path in self.UNWIRED_SWITCHES:
            node: object = self
            for part in path.split("."):
                node = getattr(node, part, None)
                if node is None:
                    break
            if node is True:
                offenders.append(path)
        if offenders:
            blocks = sorted({self.UNWIRED_SWITCHES[p] for p in offenders})
            raise ValueError(
                f"robot.grasping: {', '.join(offenders)} is set, but the "
                f"{'block' if len(blocks) == 1 else 'blocks'} {', '.join(blocks)} "
                "never reaches the pick path. The value lands in EffectiveGraspingConfig (the "
                "cell's telemetry) and nothing reads it back, measured on-box, see "
                "docs/grasping-config-reference.md section 6.1. Enabling it would give you a cell that "
                "reports the capability and does not have it. Set it back to false. "
                "feasibility additionally carries a measured on-box regression "
                "(calculator.py:414: ik_quality took the UR5e overhead pick 10/10 -> 0/10), so "
                "wiring it up is not a matter of connecting it and moving on."
            )
        return self
