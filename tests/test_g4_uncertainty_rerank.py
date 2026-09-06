"""G4 Part 2a — maybe_uncertainty_rerank_candidates: the subtractive per-candidate uncertainty re-sort.

Mirrors the U4 maybe_blend_rerank_candidates contract (silent-None / skip-ladder / ties-by-index /
score-never-mutated). The decisive proof is test_uncertainty_flips_order: geometry ranks A>B but A is far
more uncertain, so the subtractive penalty makes B the executed winner -- a real decision change, not
telemetry. The producer + reader are pinned by tests/test_g4_corridor_producer.py.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from src.robot.grasping.scoring.success_probability import (
    UncertaintyRerankConfig,
    maybe_uncertainty_rerank_candidates,
)


def _ctx(*, mode_label: str = "dense_autonomous", lifecycle_phase: str = "canary") -> SimpleNamespace:
    return SimpleNamespace(mode_label=mode_label, lifecycle_phase=lifecycle_phase)


def _pt(score: float, uncertainty: float | None, label: str) -> SimpleNamespace:
    md = {} if uncertainty is None else {"corridor_blockage_confidence": float(uncertainty)}
    return SimpleNamespace(score=float(score), metadata=md, label=label)


class TestConfig:
    def test_defaults_disabled_weight_zero_dense_only(self) -> None:
        cfg = UncertaintyRerankConfig()
        assert cfg.enabled is False
        assert cfg.weight == 0.0
        assert cfg.modes == ("dense_clutter", "dense_autonomous")

    def test_weight_out_of_bounds_rejected(self) -> None:
        with pytest.raises(ValueError):
            UncertaintyRerankConfig(enabled=True, weight=0.6)

    def test_modes_locked_dense_only(self) -> None:
        with pytest.raises(ValueError):
            UncertaintyRerankConfig(modes=("easy",))
        with pytest.raises(ValueError):
            UncertaintyRerankConfig(modes=("auto",))


class TestRerank:
    def test_disabled_is_silent_none(self) -> None:
        pts = (_pt(0.9, 0.1, "a"),)
        new, tel = maybe_uncertainty_rerank_candidates(pts, ctx=_ctx(), config=None)
        assert tel is None and new == pts
        _, tel2 = maybe_uncertainty_rerank_candidates(
            pts, ctx=_ctx(), config=UncertaintyRerankConfig(enabled=False)
        )
        assert tel2 is None

    def test_weight_zero_skip(self) -> None:
        cfg = UncertaintyRerankConfig(enabled=True, weight=0.0)
        _, tel = maybe_uncertainty_rerank_candidates((_pt(0.9, 0.1, "a"),), ctx=_ctx(), config=cfg)
        assert tel is not None and tel.applied is False and tel.skip_reason == "weight_zero"

    def test_wrong_mode_skip(self) -> None:
        cfg = UncertaintyRerankConfig(enabled=True, weight=0.3)
        _, tel = maybe_uncertainty_rerank_candidates(
            (_pt(0.9, 0.1, "a"),), ctx=_ctx(mode_label="auto"), config=cfg
        )
        assert tel is not None and tel.applied is False and tel.skip_reason == "mode_not_in_rerank_modes"

    def test_shadow_lifecycle_skip(self) -> None:
        cfg = UncertaintyRerankConfig(enabled=True, weight=0.3)
        _, tel = maybe_uncertainty_rerank_candidates(
            (_pt(0.9, 0.1, "a"),), ctx=_ctx(lifecycle_phase="shadow"), config=cfg
        )
        assert tel is not None and tel.applied is False and tel.skip_reason == "shadow_lifecycle"

    def test_missing_uncertainty_is_hard_noop(self) -> None:
        cfg = UncertaintyRerankConfig(enabled=True, weight=0.3)
        pts = (_pt(0.9, 0.1, "a"), _pt(0.8, None, "b"))  # b carries no corridor signal
        new, tel = maybe_uncertainty_rerank_candidates(pts, ctx=_ctx(), config=cfg)
        assert tel is not None and tel.applied is False and tel.skip_reason == "missing_uncertainty"
        assert new == pts  # order preserved

    def test_uncertainty_flips_order_so_runner_up_wins(self) -> None:
        cfg = UncertaintyRerankConfig(enabled=True, weight=0.5)
        # geometry: A(0.85) > B(0.70). adjusted_A = 0.85 - 0.5*0.90 = 0.40; adjusted_B = 0.70 - 0.5*0.10
        # = 0.65 => B wins, top1 changes, B is what .best would return.
        a = _pt(0.85, 0.90, "geo_leader")
        b = _pt(0.70, 0.10, "unc_safe")
        new, tel = maybe_uncertainty_rerank_candidates((a, b), ctx=_ctx(), config=cfg)
        assert tel is not None and tel.applied is True and tel.top1_changed is True
        assert new[0].label == "unc_safe"
        assert tel.winner_geometric_score == 0.70
        assert tel.winner_uncertainty == 0.10
        assert math.isclose(tel.winner_adjusted_score, 0.65, abs_tol=1e-9)
        assert tel.pre_rerank_top1_geometric_score == 0.85
        assert [p.score for p in (a, b)] == [0.85, 0.70]  # .score is NEVER mutated

    def test_tie_break_preserves_pre_rerank_order(self) -> None:
        cfg = UncertaintyRerankConfig(enabled=True, weight=0.5)
        # adjusted_A = 0.8 - 0.5*0.4 = 0.6 ; adjusted_B = 0.7 - 0.5*0.2 = 0.6 -> tie -> pre-rerank order kept
        a = _pt(0.8, 0.4, "a")
        b = _pt(0.7, 0.2, "b")
        new, tel = maybe_uncertainty_rerank_candidates((a, b), ctx=_ctx(), config=cfg)
        assert tel is not None and tel.applied is True and tel.top1_changed is False
        assert new[0].label == "a"
