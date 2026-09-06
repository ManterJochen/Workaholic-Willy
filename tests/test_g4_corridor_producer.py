"""G4 Part 1 — the per-candidate corridor-risk producer + per_candidate_uncertainty (the honest signal).

The producer runs analyze_corridor once per surviving candidate (default-off) and stamps a real,
per-candidate, NON-circular uncertainty in [0,1] on GraspPoint.metadata. per_candidate_uncertainty is the
honest reader the G4 re-ranker (Part 2) consumes. The analyzer's correctness is pinned by
tests/test_t3_corridor_analyzer.py; this pins the WIRING + the default-off byte-identical invariant.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.robot.grasping.generation.calculator import GraspCalculator
from src.robot.grasping.uncertainty import per_candidate_uncertainty


def _camera_matrix() -> np.ndarray:
    return np.array([[200.0, 0.0, 12.0], [0.0, 200.0, 12.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _convex_mask() -> np.ndarray:
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[7:17, 6:18] = 1
    return mask


def _calc(*, enabled: bool) -> GraspCalculator:
    return GraspCalculator(
        min_grip_width_mm=1.0,
        max_grip_width_mm=200.0,
        max_candidates=3,
        camera_matrix=_camera_matrix(),
        corridor_risk_per_candidate=enabled,
    )


_SEG = SimpleNamespace(mask=_convex_mask(), label="t")
_DEPTH = np.full((24, 24), 1000.0, dtype=np.float64)


class TestCorridorProducer:
    def test_off_is_byte_identical(self) -> None:
        calc = _calc(enabled=False)
        cands = calc.compute(_SEG, _DEPTH, pixel_to_mm=5.0)
        assert cands
        assert all("corridor_blockage_confidence" not in c.metadata for c in cands)
        assert "corridor_candidates_analyzed" not in calc.last_telemetry

    def test_on_stamps_per_candidate_signal(self) -> None:
        calc = _calc(enabled=True)
        cands = calc.compute(_SEG, _DEPTH, pixel_to_mm=5.0)
        assert cands
        for c in cands:
            conf = c.metadata.get("corridor_blockage_confidence")
            assert conf is not None and 0.0 <= conf <= 1.0
            assert "corridor_mode" in c.metadata
            # the honest reader returns EXACTLY the stamped value (no inversion, no re-derivation).
            assert per_candidate_uncertainty(c) == conf
        assert calc.last_telemetry["corridor_candidates_analyzed"] == len(cands)
        assert "corridor_max_blockage_confidence" in calc.last_telemetry


class TestPerCandidateUncertainty:
    def test_reads_corridor_key(self) -> None:
        point = SimpleNamespace(metadata={"corridor_blockage_confidence": 0.85}, score=0.99)
        assert per_candidate_uncertainty(point) == 0.85

    def test_none_when_absent_or_no_metadata(self) -> None:
        assert per_candidate_uncertainty(SimpleNamespace(metadata={}, score=0.9)) is None
        assert per_candidate_uncertainty(SimpleNamespace(score=0.9)) is None

    def test_clamps_to_unit_interval(self) -> None:
        assert per_candidate_uncertainty(SimpleNamespace(metadata={"corridor_blockage_confidence": 1.5})) == 1.0
        assert per_candidate_uncertainty(SimpleNamespace(metadata={"corridor_blockage_confidence": -0.2})) == 0.0

    def test_none_on_bad_values(self) -> None:
        for bad in (True, "x", float("nan"), float("inf"), None):
            point = SimpleNamespace(metadata={"corridor_blockage_confidence": bad})
            assert per_candidate_uncertainty(point) is None

    def test_non_circular_ignores_score(self) -> None:
        # A high geometric score in a blocked corridor still yields HIGH uncertainty: the reader looks
        # ONLY at the corridor key, never at .score -> non-circular by construction.
        point = SimpleNamespace(metadata={"corridor_blockage_confidence": 0.9}, score=0.99)
        assert per_candidate_uncertainty(point) == 0.9
