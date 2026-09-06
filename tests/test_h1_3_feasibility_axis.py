"""H1.3a — the feasibility ranking axis (T2) turned ON in the calculator (default-off).

The 4th ranking axis was doubly dead (weight 0 + never fed inputs). H1.3a wires it as a POST-IK re-rank:
``_apply_ik_filter`` now retains the per-candidate ``IKQualityMetrics`` (previously discarded), and when
``GraspCalculator(feasibility_weight>0, feasibility_config=...)`` the IK survivors are re-scored with the
feasibility term so hard-to-execute (near-singular) grasps DEMOTE. Default off -> no inputs fed -> the order
and the success-model ``feas_*`` features stay byte-identical (NaN).

This is the OFF-BOX "fired"=measured proof (a near-singular candidate is demoted when on; IK quality is
ignored when off). On-box (H1.3a-ii) adds the sim ≥8/10 gate -- though the sim's ±360° joint limits make the
joint_margin sub-signal rarely bind, so ik_quality (condition number) is the sim-demonstrable lever.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.robot.grasping import GraspCalculator
from src.robot.grasping.planning.reachability import IKQualityMetrics, IKResult
from src.robot.grasping.scoring import FeasibilityScoreConfig


def _camera_matrix() -> np.ndarray:
    return np.array(
        [[200.0, 0.0, 12.0], [0.0, 200.0, 12.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _scene() -> tuple[SimpleNamespace, np.ndarray]:
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[7:17, 6:18] = 1
    depth = np.zeros((24, 24), dtype=np.float64)
    for r in range(24):
        depth[r, :] = 1000.0 + max(0.0, (r - 7)) * 20.0
    return SimpleNamespace(mask=mask, label="box"), depth


class _AllReachable:
    """Every pose reachable with a healthy (low) condition number."""

    def query(self, pose: object) -> IKResult:
        return IKResult(reachable=True, quality=IKQualityMetrics(condition_number=2.0))


class _PenalizeNear:
    """Reachable everywhere, but reports a NEAR-SINGULAR condition number for poses at one target position."""

    def __init__(self, target_xyz: tuple[float, ...], tol_mm: float = 1.0) -> None:
        self._t = np.asarray(target_xyz, dtype=np.float64)
        self._tol = float(tol_mm)

    def query(self, pose: object) -> IKResult:
        pos = np.asarray(pose.position_mm, dtype=np.float64)  # type: ignore[attr-defined]
        near = bool(np.linalg.norm(pos - self._t) <= self._tol)
        return IKResult(
            reachable=True,
            quality=IKQualityMetrics(condition_number=1e6 if near else 2.0),
        )


def _positions(grasps: list) -> list[tuple[float, ...]]:
    return [tuple(round(float(x), 4) for x in g.position) for g in grasps]


class FeasibilityReorderTests(unittest.TestCase):
    def test_feasibility_demotes_the_near_singular_candidate(self) -> None:
        seg, depth = _scene()
        off = GraspCalculator(
            camera_matrix=_camera_matrix(), max_candidates=6, ik_service=_AllReachable()
        ).compute(seg, depth, pixel_to_mm=5.0, unit="mm")
        self.assertGreaterEqual(len(off), 2)
        best = _positions(off)[0]
        # Penalize EXACTLY the off-best candidate -> with feasibility ON it must lose the #1 slot.
        on = GraspCalculator(
            camera_matrix=_camera_matrix(),
            max_candidates=6,
            ik_service=_PenalizeNear(best),
            feasibility_weight=0.5,
            feasibility_config=FeasibilityScoreConfig(ik_quality_enabled=True),
        ).compute(seg, depth, pixel_to_mm=5.0, unit="mm")
        self.assertNotEqual(_positions(on)[0], best)

    def test_off_ignores_ik_quality(self) -> None:
        # With feasibility OFF (default), a differentiating IK quality must NOT reorder candidates.
        seg, depth = _scene()
        baseline = GraspCalculator(
            camera_matrix=_camera_matrix(), max_candidates=6, ik_service=_AllReachable()
        ).compute(seg, depth, pixel_to_mm=5.0, unit="mm")
        penalized = GraspCalculator(
            camera_matrix=_camera_matrix(),
            max_candidates=6,
            ik_service=_PenalizeNear(_positions(baseline)[0]),
        ).compute(seg, depth, pixel_to_mm=5.0, unit="mm")
        self.assertEqual(_positions(baseline), _positions(penalized))


class FeasibilityValidationTests(unittest.TestCase):
    def test_negative_weight_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GraspCalculator(camera_matrix=_camera_matrix(), feasibility_weight=-1.0)

    def test_weight_without_config_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GraspCalculator(camera_matrix=_camera_matrix(), feasibility_weight=0.5)

    def test_bad_config_type_rejected(self) -> None:
        with self.assertRaises(TypeError):
            GraspCalculator(
                camera_matrix=_camera_matrix(),
                feasibility_weight=0.5,
                feasibility_config="nope",  # type: ignore[arg-type]
            )


class RerankHelperTests(unittest.TestCase):
    """The pure re-rank helper's contract (the reorder math is proven end-to-end above)."""

    def test_length_mismatch_rejected(self) -> None:
        from src.robot.grasping.scoring import (
            FeasibilityInputs,
            GraspScoreWeights,
            rerank_breakdowns_with_feasibility,
        )

        with self.assertRaises(ValueError):
            rerank_breakdowns_with_feasibility(
                [],  # zero breakdowns ...
                [FeasibilityInputs()],  # ... but one input -> length mismatch
                weights=GraspScoreWeights(feasibility=0.1),
                feasibility_config=FeasibilityScoreConfig(ik_quality_enabled=True),
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
