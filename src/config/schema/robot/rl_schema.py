"""RL optimisation-layer config schema."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .._base import ConfigPath, StrictModel


# RL optimisation extension layer, the contract surface. The schema accepts five
# product modes; the runtime supports only {geometry_only, hybrid_ml} and rejects
# the rest. Every RL-active mode requires both policy_id and artifact_path, so an
# operator cannot enable a mode whose producer was never promoted.
#
# The operator authority hierarchy is a hard lock: robot.safety > runtime and
# hardware constraints > deterministic geometry > deterministic recovery > ML/RL.
# Nothing here may alter safety. Action masks are not YAML knobs but immutable
# code-side mappings per policy class, so the runtime safety boundary cannot be
# widened from config.

#: Stable string identifiers for every product mode.
RL_MODE_GEOMETRY_ONLY: str = "geometry_only"
RL_MODE_HYBRID_ML: str = "hybrid_ml"
RL_MODE_RL_SHADOW: str = "rl_shadow"
RL_MODE_RL_ACTIVE: str = "rl_active"
RL_MODE_RL_EXPERIMENTAL: str = "rl_experimental"

#: All five product modes the schema accepts. The subset the runtime supports is
#: :data:`src.robot.grasping.rl.RL_SUPPORTED_MODES_DETERMINISTIC`.
RL_MODE_VALUES: tuple[str, ...] = (
    RL_MODE_GEOMETRY_ONLY,
    RL_MODE_HYBRID_ML,
    RL_MODE_RL_SHADOW,
    RL_MODE_RL_ACTIVE,
    RL_MODE_RL_EXPERIMENTAL,
)

#: The RL-active modes: those that may influence ranking, sequencing or recovery
#: decisions once their producers land. All three require a typed artifact and
#: trigger the telemetry-completeness requirements.
RL_ACTIVE_MODES: frozenset[str] = frozenset(
    {RL_MODE_RL_SHADOW, RL_MODE_RL_ACTIVE, RL_MODE_RL_EXPERIMENTAL}
)




class RLExperimentalConfig(StrictModel):
    """``rl_experimental`` lane carrier.

    The runtime rejects this lane; the block exists so that an operator sees the
    whole surface in YAML. ``enabled`` must be ``True`` before the schema accepts
    ``mode = "rl_experimental"``.
    """

    enabled: bool = Field(default=False)


class RobotRLConfig(StrictModel):
    """RL optimisation extension layer.

    The default ``mode = "hybrid_ml"`` is the production stack: the deterministic
    pipeline plus the promoted ML scorers, no RL artifact loaded. A default
    configuration therefore behaves as the deterministic stack does.

    An RL-active mode requires ``mode`` set to ``rl_shadow``, ``rl_active`` or
    ``rl_experimental``, both ``policy_id`` and ``artifact_path``, and for
    ``rl_experimental`` also ``experimental.enabled``. The runtime rejects every
    RL-active mode with a typed :class:`RLModeNotImplementedError`; this schema
    gate is defence in depth for when the producers land.
    """

    mode: Literal[
        "geometry_only",
        "hybrid_ml",
        "rl_shadow",
        "rl_active",
        "rl_experimental",
    ] = Field(default="hybrid_ml")
    policy_id: str | None = Field(default=None, min_length=1)
    artifact_path: ConfigPath | None = Field(default=None, min_length=1)
    # Optional per-policy artifact overrides for the shadow router; without one it loads the committed
    # baseline. StrictModel forbids extras, so shadow.py's
    # `getattr(rl_cfg, "<x>_artifact_path", None)` lookups resolve only for the fields declared here.
    ranking_artifact_path: ConfigPath | None = Field(default=None, min_length=1)
    sequencing_artifact_path: ConfigPath | None = Field(default=None, min_length=1)
    perception_artifact_path: ConfigPath | None = Field(default=None, min_length=1)
    recovery_artifact_path: ConfigPath | None = Field(default=None, min_length=1)
    experimental: RLExperimentalConfig = Field(
        default_factory=RLExperimentalConfig
    )
    # Learner runtime tier. The default ``stdlib`` is the shipping path: pure stdlib,
    # byte-deterministic, portable, the safety-auditable hot path. ``numpy`` and
    # ``torch`` are opt-in, lazily imported heavier tiers; neither becomes the ship
    # default implicitly and both stay subordinate to the safety action mask.
    # Heavy-tier artifacts are experimental-stamped and never overwrite the committed
    # stdlib goldens.
    runtime_tier: Literal["stdlib", "numpy", "torch"] = Field(default="stdlib")
    # Tier-0 online learning, shadow-only and with no influence on a grasp. When true the
    # shadow router accumulates online updates into a mutable side-state and stamps the
    # artifact root ``online_state``, still shadow, while the promotion gate refuses
    # online-mutated artifacts. Requires an RL mode, so setting it alone cannot be a
    # silent no-op.
    online_update_enabled: bool = Field(default=False)

    @model_validator(mode="after")
    def _validate_rl_active_requirements(self) -> "RobotRLConfig":
        if self.online_update_enabled and self.mode not in RL_ACTIVE_MODES:
            raise ValueError(
                "robot.rl.online_update_enabled=true requires an RL mode "
                "(rl_shadow / rl_active / rl_experimental)"
            )
        if self.mode in RL_ACTIVE_MODES:
            missing: list[str] = []
            if self.policy_id is None:
                missing.append("policy_id")
            if self.artifact_path is None:
                missing.append("artifact_path")
            if missing:
                raise ValueError(
                    f"robot.rl.mode={self.mode!r} requires "
                    f"{', '.join(missing)} to be set"
                )
            if (
                self.mode == RL_MODE_RL_EXPERIMENTAL
                and not self.experimental.enabled
            ):
                raise ValueError(
                    "robot.rl.mode='rl_experimental' requires "
                    "robot.rl.experimental.enabled=true"
                )
        return self
