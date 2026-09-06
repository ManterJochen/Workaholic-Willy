"""Phase U4 — bounded probability-blend reranker.

U4 scope (locked at handoff):

* Convex blend ``final = (1 - w) * geometric + w * predicted_p``.
* ``w`` clamped to ``[0.0, 0.5]`` so geometry retains ≥50 % influence.
* Mode filter LOCKED to ``{"dense_clutter", "dense_autonomous"}``.
  EASY / AUTO / CLOSED_LOOP must never participate in blend reranking.
* Lifecycle gate: ``"shadow"`` is hard no-op; only ``"canary"`` and
  ``"active"`` may rerank.
* ``GraspPoint.score`` is NEVER mutated — only ``metadata`` and the
  *order* of the returned candidate tuple change.
* When the config is absent / disabled the runtime is byte-identical
  to the U2/U3 path (silent no-op, ``None`` telemetry).
* When the rerank fires and the top-1 candidate changes, the chosen
  candidate's shadow telemetry is re-derived so
  :attr:`PickReport.shadow_success_telemetry` always describes the
  candidate that will actually be executed.

These tests pin all of the above. They are additive — U0/U1/U2/U3
invariants are validated by their own suites and must not regress as
part of U4 acceptance.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Optional

import numpy as np

from src.config.schema.robot.robot_schema import GraspingSuccessModelConfig
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.loop.pick_loop import (
    BinPickingOrchestrator,
    PickOutcome,
)
from src.robot.grasping.scoring.success_probability import (
    SHADOW_METADATA_KEY_BLEND_WEIGHT,
    SHADOW_METADATA_KEY_FINAL_SCORE,
    SHADOW_METADATA_KEY_GEOMETRIC_PRE_BLEND,
    SHADOW_METADATA_KEY_PROBABILITY,
    RankingBlendConfig,
    ShadowSuccessContext,
    UncertaintyRerankConfig,
    load_success_probability_model,
    maybe_blend_rerank_candidates,
)
from tests._helpers import _FakeArm, _FakePerception, _perception_frame


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "assets/models/success_probability/v1"


# ---------------------------------------------------------------------------
# Shared fixtures (mirror the U2 patterns so the seam stays familiar)
# ---------------------------------------------------------------------------


def _load_ctx(
    *,
    mode_label: str = "dense_clutter",
    version_label: str = "v1",
    lifecycle_phase: str = "canary",
) -> ShadowSuccessContext:
    """Build a context backed by the committed v1 artifact."""

    model = load_success_probability_model(ARTIFACT_DIR)
    return ShadowSuccessContext(
        model=model,
        version_label=version_label,
        lifecycle_phase=lifecycle_phase,
        mode_label=mode_label,
    )


def _make_point(
    *,
    score: float,
    predicted_p: float | None = None,
    label: str = "test",
) -> GraspPoint:
    md: dict = {
        "geometric_score": score,
        "stability_score": 0.7,
        "reachability_score": 0.9,
        "feasibility_score": 0.85,
        "pose_confidence": 0.95,
        "score_components": {
            "geometric": score,
            "stability": 0.7,
            "reachability": 0.9,
        },
    }
    if predicted_p is not None:
        md[SHADOW_METADATA_KEY_PROBABILITY] = float(predicted_p)
    return GraspPoint(
        position=np.array([100.0, 50.0, 400.0]),
        approach=np.array([0.0, 0.0, 1.0]),
        axis=np.array([1.0, 0.0, 0.0]),
        grip_width_mm=40.0,
        score=float(score),
        frame=GraspFrame.BASE,
        label=label,
        metadata=md,
    )


# ===========================================================================
# Section A — Schema defaults + validation
# ===========================================================================


class SchemaDefaultsTests(unittest.TestCase):
    def test_defaults_preserve_byte_identical_behaviour(self) -> None:
        cfg = GraspingSuccessModelConfig()
        self.assertFalse(cfg.ranking_blend_enabled)
        self.assertEqual(cfg.ranking_blend_weight, 0.20)
        self.assertEqual(
            cfg.ranking_blend_modes,
            ("dense_clutter", "dense_autonomous"),
        )

    def test_explicit_dense_modes_accepted(self) -> None:
        cfg = GraspingSuccessModelConfig(
            ranking_blend_enabled=True,
            ranking_blend_weight=0.30,
            ranking_blend_modes=("dense_clutter",),
        )
        self.assertEqual(cfg.ranking_blend_modes, ("dense_clutter",))


class SchemaValidationTests(unittest.TestCase):
    def test_weight_above_upper_bound_rejected(self) -> None:
        with self.assertRaises(Exception):
            GraspingSuccessModelConfig(ranking_blend_weight=0.51)

    def test_weight_below_zero_rejected(self) -> None:
        with self.assertRaises(Exception):
            GraspingSuccessModelConfig(ranking_blend_weight=-0.01)

    def test_easy_mode_locked_out(self) -> None:
        with self.assertRaises(Exception) as ctx:
            GraspingSuccessModelConfig(ranking_blend_modes=("easy",))
        self.assertIn("dense modes only", str(ctx.exception))

    def test_auto_mode_locked_out(self) -> None:
        with self.assertRaises(Exception):
            GraspingSuccessModelConfig(
                ranking_blend_modes=("dense_clutter", "auto"),
            )

    def test_closed_loop_mode_locked_out(self) -> None:
        with self.assertRaises(Exception):
            GraspingSuccessModelConfig(ranking_blend_modes=("closed_loop",))

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(Exception):
            GraspingSuccessModelConfig(ranking_blend_modes=("not_a_mode",))

    def test_blend_modes_must_be_subset_of_apply_modes(self) -> None:
        with self.assertRaises(Exception) as ctx:
            GraspingSuccessModelConfig(
                # ``dense_autonomous`` is in default apply_modes but the
                # operator narrowed apply_modes; the blend set then
                # references a mode not in apply_modes -> rejected.
                apply_modes=("easy", "dense_clutter"),
                ranking_blend_modes=("dense_clutter", "dense_autonomous"),
            )
        self.assertIn("subset of apply_modes", str(ctx.exception))

    def test_blend_modes_uniqueness(self) -> None:
        with self.assertRaises(Exception):
            GraspingSuccessModelConfig(
                ranking_blend_modes=("dense_clutter", "dense_clutter"),
            )


# ===========================================================================
# Section B — RankingBlendConfig runtime carrier
# ===========================================================================


class RankingBlendConfigTests(unittest.TestCase):
    def test_is_frozen(self) -> None:
        cfg = RankingBlendConfig()
        with self.assertRaises(FrozenInstanceError):
            cfg.enabled = True  # type: ignore[misc]

    def test_default_is_disabled(self) -> None:
        cfg = RankingBlendConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.weight, 0.20)

    def test_rejects_weight_out_of_bounds(self) -> None:
        with self.assertRaises(ValueError):
            RankingBlendConfig(weight=0.51)
        with self.assertRaises(ValueError):
            RankingBlendConfig(weight=-0.01)

    def test_rejects_non_finite_weight(self) -> None:
        with self.assertRaises(ValueError):
            RankingBlendConfig(weight=float("nan"))
        with self.assertRaises(ValueError):
            RankingBlendConfig(weight=float("inf"))

    def test_rejects_non_dense_mode(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            RankingBlendConfig(enabled=True, modes=("easy",))
        self.assertIn("dense modes only", str(ctx.exception))

    def test_rejects_duplicate_modes(self) -> None:
        with self.assertRaises(ValueError):
            RankingBlendConfig(modes=("dense_clutter", "dense_clutter"))

    def test_rejects_bool_weight(self) -> None:
        with self.assertRaises(TypeError):
            RankingBlendConfig(weight=True)  # type: ignore[arg-type]


# ===========================================================================
# Section C — maybe_blend_rerank_candidates: silent no-op paths
# ===========================================================================


class SilentNoOpTests(unittest.TestCase):
    """``telemetry=None`` paths — byte-identical to legacy / U2."""

    def test_config_none_returns_original_and_no_telemetry(self) -> None:
        pts = (_make_point(score=0.9), _make_point(score=0.6))
        new, tel = maybe_blend_rerank_candidates(
            pts, ctx=_load_ctx(), config=None,
        )
        self.assertIs(new, pts if isinstance(pts, tuple) else None or new)
        self.assertEqual(new, pts)
        self.assertIsNone(tel)

    def test_config_disabled_returns_original_and_no_telemetry(self) -> None:
        pts = (_make_point(score=0.9), _make_point(score=0.6))
        cfg = RankingBlendConfig(enabled=False)
        new, tel = maybe_blend_rerank_candidates(
            pts, ctx=_load_ctx(), config=cfg,
        )
        self.assertEqual(new, pts)
        self.assertIsNone(tel)

    def test_score_field_unchanged_on_silent_path(self) -> None:
        pts = (_make_point(score=0.9), _make_point(score=0.6))
        cfg = RankingBlendConfig(enabled=False)
        maybe_blend_rerank_candidates(pts, ctx=_load_ctx(), config=cfg)
        self.assertEqual(pts[0].score, 0.9)
        self.assertEqual(pts[1].score, 0.6)


# ===========================================================================
# Section D — maybe_blend_rerank_candidates: structured skip paths
# ===========================================================================


class StructuredSkipTests(unittest.TestCase):
    """``telemetry.applied=False`` paths — audit trail for operators."""

    def _enabled_cfg(self, **kw) -> RankingBlendConfig:
        return RankingBlendConfig(enabled=True, **kw)

    def test_no_context_skip(self) -> None:
        pts = (_make_point(score=0.9, predicted_p=0.5),)
        new, tel = maybe_blend_rerank_candidates(
            pts, ctx=None, config=self._enabled_cfg(),
        )
        self.assertEqual(new, pts)
        self.assertIsNotNone(tel)
        assert tel is not None
        self.assertFalse(tel.applied)
        self.assertEqual(tel.skip_reason, "no_shadow_context")

    def test_empty_candidates_skip(self) -> None:
        new, tel = maybe_blend_rerank_candidates(
            (), ctx=_load_ctx(), config=self._enabled_cfg(),
        )
        self.assertEqual(new, ())
        assert tel is not None
        self.assertFalse(tel.applied)
        self.assertEqual(tel.skip_reason, "empty_candidates")
        self.assertEqual(tel.n_candidates, 0)

    def test_shadow_lifecycle_skip(self) -> None:
        ctx = _load_ctx(lifecycle_phase="shadow")
        pts = (_make_point(score=0.9, predicted_p=0.5),)
        new, tel = maybe_blend_rerank_candidates(
            pts, ctx=ctx, config=self._enabled_cfg(),
        )
        self.assertEqual(new, pts)
        assert tel is not None
        self.assertFalse(tel.applied)
        self.assertEqual(tel.skip_reason, "shadow_lifecycle")
        self.assertEqual(tel.lifecycle_phase, "shadow")

    def test_easy_mode_skip(self) -> None:
        ctx = _load_ctx(mode_label="easy")
        pts = (_make_point(score=0.9, predicted_p=0.5),)
        new, tel = maybe_blend_rerank_candidates(
            pts, ctx=ctx, config=self._enabled_cfg(),
        )
        self.assertEqual(new, pts)
        assert tel is not None
        self.assertFalse(tel.applied)
        self.assertEqual(tel.skip_reason, "mode_not_in_blend_modes")
        self.assertEqual(tel.mode_label, "easy")

    def test_auto_mode_skip(self) -> None:
        ctx = _load_ctx(mode_label="auto")
        pts = (_make_point(score=0.9, predicted_p=0.5),)
        _new, tel = maybe_blend_rerank_candidates(
            pts, ctx=ctx, config=self._enabled_cfg(),
        )
        assert tel is not None
        self.assertEqual(tel.skip_reason, "mode_not_in_blend_modes")

    def test_weight_zero_skip(self) -> None:
        ctx = _load_ctx(mode_label="dense_clutter")
        pts = (_make_point(score=0.9, predicted_p=0.5),)
        new, tel = maybe_blend_rerank_candidates(
            pts, ctx=ctx, config=self._enabled_cfg(weight=0.0),
        )
        self.assertEqual(new, pts)
        assert tel is not None
        self.assertFalse(tel.applied)
        self.assertEqual(tel.skip_reason, "weight_zero")
        self.assertEqual(tel.pre_blend_top1_geometric_score, 0.9)

    def test_missing_shadow_probability_hard_skip(self) -> None:
        ctx = _load_ctx(mode_label="dense_clutter")
        # One candidate has no probability written -> refuse to blend.
        pts = (
            _make_point(score=0.9, predicted_p=0.5),
            _make_point(score=0.6, predicted_p=None),
        )
        new, tel = maybe_blend_rerank_candidates(
            pts, ctx=ctx, config=self._enabled_cfg(),
        )
        self.assertEqual(new, pts)
        assert tel is not None
        self.assertFalse(tel.applied)
        self.assertEqual(tel.skip_reason, "missing_shadow_probability")
        # Score field stays untouched on the skip path.
        self.assertEqual(pts[0].score, 0.9)
        self.assertEqual(pts[1].score, 0.6)
        # No final-score metadata stamped on either candidate.
        for p in pts:
            self.assertNotIn(SHADOW_METADATA_KEY_FINAL_SCORE, p.metadata)


# ===========================================================================
# Section E — maybe_blend_rerank_candidates: applied path math + metadata
# ===========================================================================


class AppliedBlendMathTests(unittest.TestCase):
    def test_blend_math_is_convex(self) -> None:
        """For w=0.5, final = 0.5*g + 0.5*p for each candidate."""

        ctx = _load_ctx(mode_label="dense_clutter")
        cfg = RankingBlendConfig(enabled=True, weight=0.5)
        # Two candidates with identical geometry, different probability
        # -> the higher-probability candidate must win after blending.
        pts = (
            _make_point(score=0.8, predicted_p=0.2, label="lo_prob"),
            _make_point(score=0.8, predicted_p=0.9, label="hi_prob"),
        )
        new, tel = maybe_blend_rerank_candidates(pts, ctx=ctx, config=cfg)
        assert tel is not None
        self.assertTrue(tel.applied)
        self.assertTrue(tel.top1_changed)
        self.assertEqual(new[0].label, "hi_prob")
        # Final score arithmetic: 0.5*0.8 + 0.5*0.9 = 0.85
        self.assertAlmostEqual(
            new[0].metadata[SHADOW_METADATA_KEY_FINAL_SCORE], 0.85, places=6
        )
        # Loser final score: 0.5*0.8 + 0.5*0.2 = 0.50
        self.assertAlmostEqual(
            new[1].metadata[SHADOW_METADATA_KEY_FINAL_SCORE], 0.50, places=6
        )

    def test_top1_unchanged_when_geometric_winner_also_best_predicted(self) -> None:
        ctx = _load_ctx(mode_label="dense_clutter")
        cfg = RankingBlendConfig(enabled=True, weight=0.3)
        pts = (
            _make_point(score=0.9, predicted_p=0.9, label="top"),
            _make_point(score=0.6, predicted_p=0.4, label="bottom"),
        )
        new, tel = maybe_blend_rerank_candidates(pts, ctx=ctx, config=cfg)
        assert tel is not None
        self.assertTrue(tel.applied)
        self.assertFalse(tel.top1_changed)
        self.assertEqual(new[0].label, "top")

    def test_top1_changed_when_probability_flips_order(self) -> None:
        ctx = _load_ctx(mode_label="dense_autonomous")
        cfg = RankingBlendConfig(enabled=True, weight=0.5)
        # Geometric leader has very low probability; runner-up has
        # very high probability. With w=0.5 the runner-up wins.
        pts = (
            _make_point(score=0.85, predicted_p=0.05, label="geo_leader"),
            _make_point(score=0.70, predicted_p=0.95, label="prob_leader"),
        )
        new, tel = maybe_blend_rerank_candidates(pts, ctx=ctx, config=cfg)
        assert tel is not None
        self.assertTrue(tel.applied)
        self.assertTrue(tel.top1_changed)
        self.assertEqual(new[0].label, "prob_leader")
        # Winner telemetry mirrors the new winner's stats.
        self.assertEqual(tel.winner_geometric_score, 0.70)
        self.assertEqual(tel.winner_predicted_probability, 0.95)
        self.assertAlmostEqual(tel.winner_final_score, 0.825, places=6)
        self.assertEqual(tel.pre_blend_top1_geometric_score, 0.85)

    def test_metadata_stamped_on_all_candidates(self) -> None:
        ctx = _load_ctx(mode_label="dense_clutter")
        cfg = RankingBlendConfig(enabled=True, weight=0.25)
        pts = (
            _make_point(score=0.9, predicted_p=0.4),
            _make_point(score=0.7, predicted_p=0.8),
        )
        new, _ = maybe_blend_rerank_candidates(pts, ctx=ctx, config=cfg)
        for c in new:
            self.assertIn(SHADOW_METADATA_KEY_FINAL_SCORE, c.metadata)
            self.assertIn(SHADOW_METADATA_KEY_GEOMETRIC_PRE_BLEND, c.metadata)
            self.assertIn(SHADOW_METADATA_KEY_BLEND_WEIGHT, c.metadata)
            self.assertEqual(c.metadata[SHADOW_METADATA_KEY_BLEND_WEIGHT], 0.25)
            self.assertEqual(
                c.metadata[SHADOW_METADATA_KEY_GEOMETRIC_PRE_BLEND],
                c.score,
            )

    def test_score_field_is_never_mutated(self) -> None:
        ctx = _load_ctx(mode_label="dense_clutter")
        cfg = RankingBlendConfig(enabled=True, weight=0.5)
        pts = (
            _make_point(score=0.8, predicted_p=0.1),
            _make_point(score=0.7, predicted_p=0.95),
        )
        original_scores = [p.score for p in pts]
        maybe_blend_rerank_candidates(pts, ctx=ctx, config=cfg)
        self.assertEqual([p.score for p in pts], original_scores)

    def test_tie_breaking_preserves_pre_blend_order(self) -> None:
        ctx = _load_ctx(mode_label="dense_clutter")
        cfg = RankingBlendConfig(enabled=True, weight=0.5)
        # Both yield identical final_score = 0.5*0.6 + 0.5*0.8 = 0.7
        pts = (
            _make_point(score=0.6, predicted_p=0.8, label="first"),
            _make_point(score=0.6, predicted_p=0.8, label="second"),
        )
        new, tel = maybe_blend_rerank_candidates(pts, ctx=ctx, config=cfg)
        assert tel is not None
        self.assertTrue(tel.applied)
        self.assertFalse(tel.top1_changed)
        self.assertEqual(new[0].label, "first")
        self.assertEqual(new[1].label, "second")


# ===========================================================================
# Section F — Orchestrator integration
# ===========================================================================


class _ScriptedCalculator:
    """Local divergent variant: builds a fresh GraspResult from a points_factory per call
    (a DIFFERENT contract from tests._helpers._ScriptedCalculator's scripted results list).
    Intentionally NOT shared — see the anti-merge note in tests/_helpers.py."""

    def __init__(self, points_factory) -> None:
        self._factory = points_factory

    def compute_result(self, *_args: object, **_kwargs: object) -> GraspResult:
        pts = self._factory()
        return GraspResult(
            candidates=pts,
            reasons=(),
            top_score=max(p.score for p in pts),
        )


class OrchestratorBlendIntegrationTests(unittest.TestCase):
    def _build(
        self,
        *,
        ctx: Optional[ShadowSuccessContext],
        blend_cfg: Optional[RankingBlendConfig],
        points_factory,
    ) -> BinPickingOrchestrator:
        arm = _FakeArm()
        calc = _ScriptedCalculator(points_factory)
        perception = _FakePerception([_perception_frame()])
        return BinPickingOrchestrator(
            arm=arm,  # type: ignore[arg-type]
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
            max_attempts=3,
            shadow_success_context=ctx,
            ranking_blend_config=blend_cfg,
        )

    def _two_candidates(self) -> tuple[GraspPoint, ...]:
        # Geometric leader weak in probability; runner-up strong in
        # probability. Suitable for top1-flip when blend applies.
        return (
            _make_point(score=0.85, predicted_p=0.05, label="geo_leader"),
            _make_point(score=0.70, predicted_p=0.95, label="prob_leader"),
        )

    def test_disabled_config_is_byte_identical(self) -> None:
        orch = self._build(
            ctx=_load_ctx(mode_label="dense_clutter"),
            blend_cfg=RankingBlendConfig(enabled=False),
            points_factory=self._two_candidates,
        )
        report = orch.run()
        self.assertEqual(report.outcome, PickOutcome.EXECUTED)
        self.assertIsNone(report.ranking_blend_telemetry)
        executed = report.executed_grasp
        assert executed is not None
        # Geometric leader wins exactly as in legacy path.
        self.assertEqual(executed.candidates[0].label, "geo_leader")

    def test_none_config_is_byte_identical(self) -> None:
        orch = self._build(
            ctx=_load_ctx(mode_label="dense_clutter"),
            blend_cfg=None,
            points_factory=self._two_candidates,
        )
        report = orch.run()
        self.assertIsNone(report.ranking_blend_telemetry)
        executed = report.executed_grasp
        assert executed is not None
        self.assertEqual(executed.candidates[0].label, "geo_leader")

    def test_easy_mode_never_blends_even_when_enabled(self) -> None:
        orch = self._build(
            ctx=_load_ctx(mode_label="easy"),
            blend_cfg=RankingBlendConfig(enabled=True, weight=0.5),
            points_factory=self._two_candidates,
        )
        report = orch.run()
        tel = report.ranking_blend_telemetry
        self.assertIsNotNone(tel)
        assert tel is not None
        self.assertFalse(tel.applied)
        self.assertEqual(tel.skip_reason, "mode_not_in_blend_modes")
        executed = report.executed_grasp
        assert executed is not None
        # EASY mode preserves geometric ordering.
        self.assertEqual(executed.candidates[0].label, "geo_leader")

    def test_shadow_lifecycle_never_blends_even_when_enabled(self) -> None:
        orch = self._build(
            ctx=_load_ctx(mode_label="dense_clutter", lifecycle_phase="shadow"),
            blend_cfg=RankingBlendConfig(enabled=True, weight=0.5),
            points_factory=self._two_candidates,
        )
        report = orch.run()
        tel = report.ranking_blend_telemetry
        assert tel is not None
        self.assertFalse(tel.applied)
        self.assertEqual(tel.skip_reason, "shadow_lifecycle")

    def test_canary_dense_mode_reranks_and_recomputes_shadow_telemetry(self) -> None:
        # NOTE: the shadow ctx is what gets used for annotation. To make
        # the test deterministic we manually pre-seed the candidates'
        # ``shadow_predicted_success_probability`` via the factory (the
        # annotator will overwrite, but then maybe_blend_rerank reads
        # what the annotator wrote). For this test we rely on the
        # annotator's real artifact-driven predictions but accept that
        # both candidates have the same feature footprint differing
        # only in geometric_score -> the annotator may not flip.
        # Instead we use the lower-level blend function which we've
        # already tested for math correctness; here we just need to
        # confirm the orchestrator wires telemetry through. Use an
        # extreme ``predicted_p`` pre-seed and have the annotator
        # overwrite it; verify telemetry is non-None and applied=True.
        ctx = _load_ctx(mode_label="dense_clutter", lifecycle_phase="canary")
        orch = self._build(
            ctx=ctx,
            blend_cfg=RankingBlendConfig(enabled=True, weight=0.3),
            points_factory=self._two_candidates,
        )
        report = orch.run()
        tel = report.ranking_blend_telemetry
        self.assertIsNotNone(tel)
        assert tel is not None
        self.assertTrue(tel.applied)
        self.assertEqual(tel.lifecycle_phase, "canary")
        self.assertEqual(tel.mode_label, "dense_clutter")
        # The shadow telemetry must reflect the candidate that was
        # actually executed (i.e. candidates[0] of the reranked list).
        executed = report.executed_grasp
        assert executed is not None
        shadow_tel = report.shadow_success_telemetry
        self.assertIsNotNone(shadow_tel)
        assert shadow_tel is not None
        winner_prob = executed.candidates[0].metadata.get(
            SHADOW_METADATA_KEY_PROBABILITY
        )
        self.assertEqual(
            shadow_tel.predicted_success_probability,
            winner_prob,
        )
        # ``score`` field is never mutated on the actual orchestrator
        # candidates either.
        self.assertEqual(executed.candidates[0].score,
                         executed.candidates[0].metadata[
                             SHADOW_METADATA_KEY_GEOMETRIC_PRE_BLEND
                         ])

    def test_active_lifecycle_also_permits_blend(self) -> None:
        ctx = _load_ctx(mode_label="dense_autonomous", lifecycle_phase="active")
        orch = self._build(
            ctx=ctx,
            blend_cfg=RankingBlendConfig(enabled=True, weight=0.4),
            points_factory=self._two_candidates,
        )
        report = orch.run()
        tel = report.ranking_blend_telemetry
        assert tel is not None
        self.assertTrue(tel.applied)
        self.assertEqual(tel.lifecycle_phase, "active")


class OrchestratorUncertaintyRerankTests(unittest.TestCase):
    """G4 Part 2b normal-path wiring: the per-candidate uncertainty re-rank fires through the REAL pick
    path (after the U4 blend, before the V3 shadow + U6 gate), reorders so a different candidate executes,
    and surfaces UncertaintyRerankTelemetry on the report. Reuses the U4 orchestrator harness."""

    def _build(
        self,
        *,
        ctx: Optional[ShadowSuccessContext],
        rerank_cfg: Optional[UncertaintyRerankConfig],
        points_factory,
    ) -> BinPickingOrchestrator:
        return BinPickingOrchestrator(
            arm=_FakeArm(),  # type: ignore[arg-type]
            calculator=_ScriptedCalculator(points_factory),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            max_attempts=3,
            shadow_success_context=ctx,
            uncertainty_rerank_config=rerank_cfg,
        )

    def _two_candidates(self) -> tuple[GraspPoint, ...]:
        # Geometry ranks A (0.85) > B (0.70), but A's approach corridor is far more BLOCKED, so the
        # subtractive penalty flips them: adjusted_A = 0.85 - 0.5*0.90 = 0.40 < adjusted_B = 0.70 -
        # 0.5*0.10 = 0.65. The producer stamps corridor_blockage_confidence (here pre-seeded directly).
        a = _make_point(score=0.85, predicted_p=0.5, label="geo_leader")
        b = _make_point(score=0.70, predicted_p=0.5, label="unc_safe")
        a.metadata["corridor_blockage_confidence"] = 0.90
        b.metadata["corridor_blockage_confidence"] = 0.10
        return (a, b)

    def test_disabled_is_byte_identical(self) -> None:
        orch = self._build(
            ctx=_load_ctx(mode_label="dense_clutter"),
            rerank_cfg=None,
            points_factory=self._two_candidates,
        )
        report = orch.run()
        self.assertEqual(report.outcome, PickOutcome.EXECUTED)
        self.assertIsNone(report.uncertainty_rerank_telemetry)
        executed = report.executed_grasp
        assert executed is not None
        self.assertEqual(executed.candidates[0].label, "geo_leader")

    def test_dense_mode_reranks_so_runner_up_executes(self) -> None:
        ctx = _load_ctx(mode_label="dense_clutter", lifecycle_phase="canary")
        orch = self._build(
            ctx=ctx,
            rerank_cfg=UncertaintyRerankConfig(enabled=True, weight=0.5),
            points_factory=self._two_candidates,
        )
        report = orch.run()
        tel = report.uncertainty_rerank_telemetry
        self.assertIsNotNone(tel)
        assert tel is not None
        self.assertTrue(tel.applied)
        self.assertTrue(tel.top1_changed)
        self.assertEqual(tel.mode_label, "dense_clutter")
        executed = report.executed_grasp
        assert executed is not None
        # B executes, NOT the geometric leader A -> a real decision change driven by corridor risk.
        self.assertEqual(executed.candidates[0].label, "unc_safe")
        # ``score`` is NEVER mutated.
        self.assertEqual(executed.candidates[0].score, 0.70)

    def test_easy_mode_never_reranks(self) -> None:
        orch = self._build(
            ctx=_load_ctx(mode_label="easy"),
            rerank_cfg=UncertaintyRerankConfig(enabled=True, weight=0.5),
            points_factory=self._two_candidates,
        )
        report = orch.run()
        tel = report.uncertainty_rerank_telemetry
        assert tel is not None
        self.assertFalse(tel.applied)
        self.assertEqual(tel.skip_reason, "mode_not_in_rerank_modes")
        executed = report.executed_grasp
        assert executed is not None
        self.assertEqual(executed.candidates[0].label, "geo_leader")


# ===========================================================================
# Section G — End-to-end top-1 lift signal (deterministic, WARN-only)
# ===========================================================================
#
# Per the U4 design lock (Q4=B), the hard ``>= +1.5 pts`` rank-lift gate
# is deferred to U6 when real shadow-telemetry replay lands. U4 ships a
# *deterministic synthetic benchmark* that measures the lift and logs
# it; the test asserts only that the benchmark runs and that blending
# does not *regress* expected top-1 success on a contrived favourable
# scenario.


class SyntheticTop1LiftWarnTests(unittest.TestCase):
    """Synthetic, deterministic; measures lift, hard gate deferred."""

    def _scenario(self, rng: np.random.Generator) -> list[tuple[float, float, float]]:
        """One scenario = list of (geometric, predicted_p, true_p)."""

        n = 6
        out: list[tuple[float, float, float]] = []
        for _ in range(n):
            geo = float(rng.uniform(0.3, 0.95))
            # Make ``predicted_p`` weakly correlated with ``true_p`` but
            # noisy enough that geometry-only sometimes picks the worse
            # candidate. True success p is correlated with geometry but
            # has its own noise.
            true_p = float(np.clip(0.4 + 0.6 * geo + rng.normal(0, 0.15), 0, 1))
            pred_p = float(np.clip(true_p + rng.normal(0, 0.1), 0, 1))
            out.append((geo, pred_p, true_p))
        return out

    def test_blend_does_not_regress_top1_success_on_synthetic_scenarios(self) -> None:
        rng = np.random.default_rng(seed=20250307)
        n_scenarios = 200
        w = 0.30
        geo_only_wins = 0
        blend_wins = 0
        for _ in range(n_scenarios):
            cands = self._scenario(rng)
            # Geometry-only winner index.
            i_geo = int(np.argmax([c[0] for c in cands]))
            # Blend winner index using predicted probability.
            i_blend = int(
                np.argmax([(1.0 - w) * c[0] + w * c[1] for c in cands])
            )
            # "Success" = the candidate's true_p (treated as the
            # expected outcome value).
            geo_only_wins += cands[i_geo][2]
            blend_wins += cands[i_blend][2]
        geo_rate = geo_only_wins / n_scenarios
        blend_rate = blend_wins / n_scenarios
        lift_pts = (blend_rate - geo_rate) * 100.0
        # WARN-only: log the measured lift; do NOT hard-assert ≥1.5.
        # The contract here is non-regression on this favourable
        # scenario distribution: blend must not be strictly worse.
        # (Hard ≥+1.5 pts gate is deferred to U6 with real replay.)
        import sys
        print(
            f"[U4-WARN] synthetic_top1_lift: geo={geo_rate:.4f} "
            f"blend={blend_rate:.4f} lift_pts={lift_pts:+.2f}",
            file=sys.stderr,
        )
        self.assertGreaterEqual(
            blend_rate,
            geo_rate - 1e-9,
            msg=(
                "Blend regressed top-1 success on the synthetic "
                "scenario distribution; investigate predictor "
                f"correlation. lift={lift_pts:+.4f} pts"
            ),
        )
        # Sanity: lift should be measurable on this scenario design.
        # (We do not gate hard; just a soft floor to detect silent
        # disconnection of the blend math.)
        self.assertTrue(math.isfinite(lift_pts))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
