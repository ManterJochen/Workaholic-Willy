"""Tests for the suction scorer seam: the Protocol, and the analytical physics that implements it."""

from __future__ import annotations

import numpy as np
import pytest

from src.robot.grasping.suction.scorer import (
    AnalyticalSuctionScorer,
    SuctionScorer,
)


def _flat_patch(z: float = 100.0, half_mm: float = 30.0, step: float = 1.5) -> np.ndarray:
    g = np.arange(-half_mm, half_mm + step, step)
    gx, gy = np.meshgrid(g, g)
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])


def test_analytical_combines_seal_and_wrench() -> None:
    scorer = AnalyticalSuctionScorer()
    pts = _flat_patch()
    q = scorer.score(
        [0, 0, 100], [0, 0, -1], pts, surface_normal=[0, 0, 1],
        payload_mass_g=200.0, com_mm=[0, 0, 100],
    )
    assert q.source == "analytical"
    assert q.seal is not None and q.wrench is not None
    assert q.quality == pytest.approx(q.seal.seal_score * q.wrench.resist_score)
    assert q.quality > 0.9  # flat + light + centred -> excellent


def test_heavy_payload_drops_quality_via_wrench() -> None:
    scorer = AnalyticalSuctionScorer()
    pts = _flat_patch()
    light = scorer.score([0, 0, 100], [0, 0, -1], pts, surface_normal=[0, 0, 1],
                         payload_mass_g=200.0, com_mm=[0, 0, 100])
    heavy = scorer.score([0, 0, 100], [0, 0, -1], pts, surface_normal=[0, 0, 1],
                         payload_mass_g=6000.0, com_mm=[0, 0, 100])
    assert heavy.quality < light.quality
    assert heavy.wrench is not None and not heavy.wrench.feasible


def test_no_payload_is_seal_only() -> None:
    scorer = AnalyticalSuctionScorer()
    q = scorer.score([0, 0, 100], [0, 0, -1], _flat_patch(), surface_normal=[0, 0, 1])
    assert q.wrench is None
    assert q.seal is not None
    assert q.quality == q.seal.seal_score


def test_scorer_satisfies_protocol() -> None:
    assert isinstance(AnalyticalSuctionScorer(), SuctionScorer)


def test_analytical_prepare_scene_is_noop() -> None:
    # The analytical backend scores from the cloud; prepare_scene must be a harmless no-op.
    assert AnalyticalSuctionScorer().prepare_scene(None, np.ones((4, 4)), np.eye(3)) is None


def test_the_protocol_still_admits_a_second_backend() -> None:
    """The seam is the Protocol, not any one class -- asserted so the removal did not close the door.

    A learned scorer was deleted from this package for licensing, and the point of doing it that way
    was that nothing structural depended on it. This builds a throwaway implementation to prove the
    Protocol is still the whole contract: implement prepare_scene + score, and synthesis will take it.
    """
    from src.robot.grasping.suction.scorer import SuctionQuality

    class _Stub:
        def prepare_scene(self, rgb, depth_m, intrinsics) -> None:  # noqa: ANN001
            self.prepared = True

        def score(self, contact_point_mm, approach, points_mm, **kwargs) -> SuctionQuality:  # noqa: ANN001
            return SuctionQuality(quality=0.5, source="stub")

    assert isinstance(_Stub(), SuctionScorer)
