"""Gap G1 — the U4 blend lifecycle_phase is now config-threaded (was a hardcoded 'shadow' literal).

The full U4 learned-blend pipeline (predictor, bounded blend, promotion gate, telemetry) already
existed and is unit-tested (test_u4_probability_active_ranking: shadow skips, canary reorders), but the
builder hardcoded lifecycle_phase='shadow', so the promoted predictor could NEVER influence which grasp
executes. These tests pin the new config knob + prove a canary artifact loads past the promotion gate
(the committed v1 artifact carries a passing promotion.json), i.e. the blend is now reachable.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.schema.robot.robot_schema import GraspingSuccessModelConfig
from src.robot.grasping.scoring.success_probability import (
    try_load_shadow_success_context,
)

_ARTIFACT = "assets/models/success_probability/v1"  # committed; carries a passing promotion.json


class TestLifecyclePhaseConfig:
    def test_default_is_shadow(self) -> None:
        assert GraspingSuccessModelConfig().lifecycle_phase == "shadow"

    def test_canary_and_active_accepted(self) -> None:
        assert GraspingSuccessModelConfig(lifecycle_phase="canary").lifecycle_phase == "canary"
        assert GraspingSuccessModelConfig(lifecycle_phase="active").lifecycle_phase == "active"

    def test_invalid_phase_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GraspingSuccessModelConfig(lifecycle_phase="live")


class TestCanaryArtifactLoads:
    def test_canary_loads_past_promotion_gate(self) -> None:
        # The now-reachable canary value loads a CANARY context (the committed artifact's promotion.json
        # passes verify_promotion) -> the blend can fire. (test_u4 proves shadow skips, canary reorders.)
        ctx = try_load_shadow_success_context(
            enabled=True, artifact_dir=_ARTIFACT,
            mode_label="dense_clutter", version_label="v1", lifecycle_phase="canary",
        )
        assert ctx is not None
        assert ctx.lifecycle_phase == "canary"

    def test_shadow_loads_as_shadow(self) -> None:
        ctx = try_load_shadow_success_context(
            enabled=True, artifact_dir=_ARTIFACT,
            mode_label="dense_clutter", version_label="v1", lifecycle_phase="shadow",
        )
        assert ctx is not None
        assert ctx.lifecycle_phase == "shadow"
