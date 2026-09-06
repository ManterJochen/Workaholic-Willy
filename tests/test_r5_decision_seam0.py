"""R5.1 Seam-0 — pin ALL 15 DecisionReport fields across BOTH decide() paths before the dedup.

The R5.1 refactor extracts a shared report-builder + two selector helpers from
:meth:`DecisionEngine.decide`. The existing T1/U8 suites assert only ``action`` + ``reason_code``
(plus 2 string fields on a few fused cases) — the entire fused half of ``decide`` is byte-unguarded
(the R2 characterization golden never enters ``use_fused=True``). A float-op reorder or a carrier-block
swap in the shared builder could silently change ``uncertainty_score`` / ``penalised_top_score`` /
``channel_disagreement`` / ``uncertainty_source`` and still pass green.

This test pins the EXACT 15-field ``to_dict()`` for 21 scenarios spanning every terminal branch
(fused confident / move-camera / fail-closed / permissive×3 / disagreement×3 / ranking-penalty×2;
legacy clamp-min / clamp-max / fail-closed / planner-unavailable / sim-downgrade / fallback / legacy;
and no-candidates). Captured from HEAD before any move; ``assertEqual`` on the whole dict is bit-exact
(note J4/J5 ``uncertainty_score == 0.09999999999999998`` — the legacy ``1.0 - 0.9`` float, NOT 0.1).
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.decision import DecisionEngine, DecisionPolicy
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.uncertainty import (
    UncertaintyCalibration,
    UncertaintyChannelValues,
    UncertaintyWeights,
    fuse_uncertainty,
)


def _gr(score: float, *, candidates: int = 1, reasons: tuple = ()) -> GraspResult:
    cand_score = min(float(score), 1.0)  # GraspPoint.score is bounded [0,1]; top_score is not
    pts = tuple(
        GraspPoint(
            position=np.array([100.0, 50.0, 400.0]),
            approach=np.array([0.0, 0.0, 1.0]),
            axis=np.array([1.0, 0.0, 0.0]),
            grip_width_mm=40.0,
            score=cand_score,
            frame=GraspFrame.BASE,
            label="t",
        )
        for _ in range(candidates)
    )
    return GraspResult(candidates=pts, reasons=reasons, top_score=score)


def _engine(**ov: object) -> DecisionEngine:
    kw: dict[str, object] = dict(
        enabled=True,
        auto_uncertainty_threshold=0.4,
        max_reobservations=2,
        reasons_penalty=0.2,
        fail_closed_on_real_hardware=True,
    )
    kw.update(ov)
    return DecisionEngine(policy=DecisionPolicy(**kw))  # type: ignore[arg-type]


def _cal() -> UncertaintyCalibration:
    return UncertaintyCalibration(
        weights=UncertaintyWeights(
            depth_confidence=1.0,
            mask_confidence=1.0,
            occlusion_corridor_risk=1.0,
            feasibility_margin=1.0,
            verification_residual=1.0,
            topology_risk=0.0,
            semantic_confidence=0.0,
        ),
        maps={},
    )


def _snap(**channels: float):
    return fuse_uncertainty(
        UncertaintyChannelValues(**channels),
        _cal(),
        fail_closed_threshold=0.5,
        channel_disagreement_threshold=0.5,
    )


def _run(name: str):
    """Build + run the named scenario, mirroring logs/r5_capture.py exactly."""
    base = dict(
        mode="HARD",
        attempt_id=name,
        reobservation_count=0,
        is_simulated=False,
        viewpoint_planner_available=True,
    )

    def decide(gr, eng, **kw):
        merged = dict(base)
        merged.update(kw)
        return eng.decide(grasp_result=gr, **merged)

    if name == "A":
        return decide(_gr(0.9), _engine(), uncertainty_snapshot=_snap(depth_confidence=0.2),
                      uncertainty_active=True, ranking_penalty_weight=0.0)
    if name == "B":
        return decide(_gr(0.9), _engine(), uncertainty_snapshot=_snap(depth_confidence=0.9),
                      uncertainty_active=True)
    if name == "C":
        return decide(_gr(0.9), _engine(max_reobservations=0),
                      uncertainty_snapshot=_snap(depth_confidence=0.9), uncertainty_active=True)
    if name == "D1":
        return decide(_gr(0.9), _engine(max_reobservations=0),
                      uncertainty_snapshot=_snap(depth_confidence=0.9), uncertainty_active=True,
                      is_simulated=True)
    if name == "D2":
        return decide(_gr(0.9), _engine(max_reobservations=0),
                      uncertainty_snapshot=_snap(depth_confidence=0.9), uncertainty_active=True,
                      is_simulated=True, viewpoint_planner_available=False)
    if name == "D3":
        return decide(_gr(0.9), _engine(max_reobservations=0, fail_closed_on_real_hardware=False),
                      uncertainty_snapshot=_snap(depth_confidence=0.9), uncertainty_active=True)
    if name == "E":
        return decide(_gr(0.95), _engine(),
                      uncertainty_snapshot=_snap(depth_confidence=0.1, mask_confidence=0.9),
                      uncertainty_active=True)
    if name == "F":
        return decide(_gr(0.95), _engine(max_reobservations=0),
                      uncertainty_snapshot=_snap(depth_confidence=0.1, mask_confidence=0.9),
                      uncertainty_active=True)
    if name == "G":
        return decide(_gr(0.95), _engine(max_reobservations=0),
                      uncertainty_snapshot=_snap(depth_confidence=0.1, mask_confidence=0.9),
                      uncertainty_active=True, is_simulated=True)
    if name == "H1":
        return decide(_gr(0.9), _engine(), uncertainty_snapshot=_snap(depth_confidence=0.2),
                      uncertainty_active=True, ranking_penalty_weight=0.5)
    if name == "H2":
        return decide(_gr(0.9), _engine(), uncertainty_snapshot=_snap(depth_confidence=0.0),
                      uncertainty_active=True, ranking_penalty_weight=0.5)
    if name == "I1":
        return decide(_gr(0.1, reasons=(GraspFailureReason.NO_CANDIDATES_GENERATED,)), _engine())
    if name == "I2":
        return decide(_gr(1.5), _engine())
    if name == "J1":
        return decide(_gr(0.5), _engine(max_reobservations=0))
    if name == "J2":
        return decide(_gr(0.5), _engine(), viewpoint_planner_available=False)
    if name == "J3":
        return decide(_gr(0.5), _engine(max_reobservations=0), is_simulated=True)
    if name == "J4":
        return decide(_gr(0.9), _engine(), uncertainty_snapshot=_snap(), uncertainty_active=True)
    if name == "J5":
        return decide(_gr(0.9), _engine())
    if name == "K":
        return decide(None, _engine())
    raise AssertionError(f"unknown scenario {name!r}")


# Captured from HEAD (logs/r5_capture.py) BEFORE the R5.1 dedup. Bit-exact baseline.
EXPECTED: dict[str, dict[str, object]] = {
    "A": {"attempt_id": "A", "channel_disagreement": 0.0, "channel_disagreement_triggered": False, "decision_action": "grasp_now", "decision_reason_code": "confident_grasp", "mode": "HARD", "penalised_top_score": 0.9, "ranking_penalty_applied": False, "reobservation_count": 0, "threshold_used": 0.4, "top_score": 0.9, "uncertainty_fused": 0.2, "uncertainty_fused_available": True, "uncertainty_score": 0.2, "uncertainty_source": "fused"},
    "B": {"attempt_id": "B", "channel_disagreement": 0.0, "channel_disagreement_triggered": False, "decision_action": "move_camera", "decision_reason_code": "low_confidence", "mode": "HARD", "penalised_top_score": 0.9, "ranking_penalty_applied": False, "reobservation_count": 0, "threshold_used": 0.4, "top_score": 0.9, "uncertainty_fused": 0.9, "uncertainty_fused_available": True, "uncertainty_score": 0.9, "uncertainty_source": "fused"},
    "C": {"attempt_id": "C", "channel_disagreement": 0.0, "channel_disagreement_triggered": False, "decision_action": "fail_closed", "decision_reason_code": "uncertainty_fail_closed", "mode": "HARD", "penalised_top_score": 0.9, "ranking_penalty_applied": False, "reobservation_count": 0, "threshold_used": 0.4, "top_score": 0.9, "uncertainty_fused": 0.9, "uncertainty_fused_available": True, "uncertainty_score": 0.9, "uncertainty_source": "fused"},
    "D1": {"attempt_id": "D1", "channel_disagreement": 0.0, "channel_disagreement_triggered": False, "decision_action": "grasp_now", "decision_reason_code": "reobserve_budget_exhausted", "mode": "HARD", "penalised_top_score": 0.9, "ranking_penalty_applied": False, "reobservation_count": 0, "threshold_used": 0.4, "top_score": 0.9, "uncertainty_fused": 0.9, "uncertainty_fused_available": True, "uncertainty_score": 0.9, "uncertainty_source": "fused"},
    "D2": {"attempt_id": "D2", "channel_disagreement": 0.0, "channel_disagreement_triggered": False, "decision_action": "grasp_now", "decision_reason_code": "reobserve_planner_unavailable", "mode": "HARD", "penalised_top_score": 0.9, "ranking_penalty_applied": False, "reobservation_count": 0, "threshold_used": 0.4, "top_score": 0.9, "uncertainty_fused": 0.9, "uncertainty_fused_available": True, "uncertainty_score": 0.9, "uncertainty_source": "fused"},
    "D3": {"attempt_id": "D3", "channel_disagreement": 0.0, "channel_disagreement_triggered": False, "decision_action": "grasp_now", "decision_reason_code": "reobserve_budget_exhausted", "mode": "HARD", "penalised_top_score": 0.9, "ranking_penalty_applied": False, "reobservation_count": 0, "threshold_used": 0.4, "top_score": 0.9, "uncertainty_fused": 0.9, "uncertainty_fused_available": True, "uncertainty_score": 0.9, "uncertainty_source": "fused"},
    "E": {"attempt_id": "E", "channel_disagreement": 0.8, "channel_disagreement_triggered": True, "decision_action": "move_camera", "decision_reason_code": "channel_disagreement", "mode": "HARD", "penalised_top_score": 0.95, "ranking_penalty_applied": False, "reobservation_count": 0, "threshold_used": 0.4, "top_score": 0.95, "uncertainty_fused": 0.5, "uncertainty_fused_available": True, "uncertainty_score": 0.5, "uncertainty_source": "fused"},
    "F": {"attempt_id": "F", "channel_disagreement": 0.8, "channel_disagreement_triggered": True, "decision_action": "fail_closed", "decision_reason_code": "channel_disagreement", "mode": "HARD", "penalised_top_score": 0.95, "ranking_penalty_applied": False, "reobservation_count": 0, "threshold_used": 0.4, "top_score": 0.95, "uncertainty_fused": 0.5, "uncertainty_fused_available": True, "uncertainty_score": 0.5, "uncertainty_source": "fused"},
    "G": {"attempt_id": "G", "channel_disagreement": 0.8, "channel_disagreement_triggered": True, "decision_action": "grasp_now", "decision_reason_code": "channel_disagreement", "mode": "HARD", "penalised_top_score": 0.95, "ranking_penalty_applied": False, "reobservation_count": 0, "threshold_used": 0.4, "top_score": 0.95, "uncertainty_fused": 0.5, "uncertainty_fused_available": True, "uncertainty_score": 0.5, "uncertainty_source": "fused"},
    "H1": {"attempt_id": "H1", "channel_disagreement": 0.0, "channel_disagreement_triggered": False, "decision_action": "grasp_now", "decision_reason_code": "confident_grasp", "mode": "HARD", "penalised_top_score": 0.8, "ranking_penalty_applied": True, "reobservation_count": 0, "threshold_used": 0.4, "top_score": 0.9, "uncertainty_fused": 0.2, "uncertainty_fused_available": True, "uncertainty_score": 0.2, "uncertainty_source": "fused"},
    "H2": {"attempt_id": "H2", "channel_disagreement": 0.0, "channel_disagreement_triggered": False, "decision_action": "grasp_now", "decision_reason_code": "confident_grasp", "mode": "HARD", "penalised_top_score": 0.9, "ranking_penalty_applied": True, "reobservation_count": 0, "threshold_used": 0.4, "top_score": 0.9, "uncertainty_fused": 0.0, "uncertainty_fused_available": True, "uncertainty_score": 0.0, "uncertainty_source": "fused"},
    "I1": {"attempt_id": "I1", "channel_disagreement": None, "channel_disagreement_triggered": False, "decision_action": "move_camera", "decision_reason_code": "low_confidence", "mode": "HARD", "penalised_top_score": None, "ranking_penalty_applied": False, "reobservation_count": 0, "threshold_used": 0.4, "top_score": 0.1, "uncertainty_fused": None, "uncertainty_fused_available": False, "uncertainty_score": 1.0, "uncertainty_source": "legacy"},
    "I2": {"attempt_id": "I2", "channel_disagreement": None, "channel_disagreement_triggered": False, "decision_action": "grasp_now", "decision_reason_code": "confident_grasp", "mode": "HARD", "penalised_top_score": None, "ranking_penalty_applied": False, "reobservation_count": 0, "threshold_used": 0.4, "top_score": 1.5, "uncertainty_fused": None, "uncertainty_fused_available": False, "uncertainty_score": 0.0, "uncertainty_source": "legacy"},
    "J1": {"attempt_id": "J1", "channel_disagreement": None, "channel_disagreement_triggered": False, "decision_action": "fail_closed", "decision_reason_code": "reobserve_budget_exhausted", "mode": "HARD", "penalised_top_score": None, "ranking_penalty_applied": False, "reobservation_count": 0, "threshold_used": 0.4, "top_score": 0.5, "uncertainty_fused": None, "uncertainty_fused_available": False, "uncertainty_score": 0.5, "uncertainty_source": "legacy"},
    "J2": {"attempt_id": "J2", "channel_disagreement": None, "channel_disagreement_triggered": False, "decision_action": "fail_closed", "decision_reason_code": "reobserve_planner_unavailable", "mode": "HARD", "penalised_top_score": None, "ranking_penalty_applied": False, "reobservation_count": 0, "threshold_used": 0.4, "top_score": 0.5, "uncertainty_fused": None, "uncertainty_fused_available": False, "uncertainty_score": 0.5, "uncertainty_source": "legacy"},
    "J3": {"attempt_id": "J3", "channel_disagreement": None, "channel_disagreement_triggered": False, "decision_action": "grasp_now", "decision_reason_code": "reobserve_budget_exhausted", "mode": "HARD", "penalised_top_score": None, "ranking_penalty_applied": False, "reobservation_count": 0, "threshold_used": 0.4, "top_score": 0.5, "uncertainty_fused": None, "uncertainty_fused_available": False, "uncertainty_score": 0.5, "uncertainty_source": "legacy"},
    "J4": {"attempt_id": "J4", "channel_disagreement": 0.0, "channel_disagreement_triggered": False, "decision_action": "grasp_now", "decision_reason_code": "confident_grasp", "mode": "HARD", "penalised_top_score": None, "ranking_penalty_applied": False, "reobservation_count": 0, "threshold_used": 0.4, "top_score": 0.9, "uncertainty_fused": None, "uncertainty_fused_available": False, "uncertainty_score": 0.09999999999999998, "uncertainty_source": "legacy_fallback"},
    "J5": {"attempt_id": "J5", "channel_disagreement": None, "channel_disagreement_triggered": False, "decision_action": "grasp_now", "decision_reason_code": "confident_grasp", "mode": "HARD", "penalised_top_score": None, "ranking_penalty_applied": False, "reobservation_count": 0, "threshold_used": 0.4, "top_score": 0.9, "uncertainty_fused": None, "uncertainty_fused_available": False, "uncertainty_score": 0.09999999999999998, "uncertainty_source": "legacy"},
    "K": {"attempt_id": "K", "channel_disagreement": None, "channel_disagreement_triggered": False, "decision_action": "recover", "decision_reason_code": "no_candidates", "mode": "HARD", "penalised_top_score": None, "ranking_penalty_applied": False, "reobservation_count": 0, "threshold_used": 0.4, "top_score": None, "uncertainty_fused": None, "uncertainty_fused_available": False, "uncertainty_score": None, "uncertainty_source": "legacy"},
}


class DecisionSeam0Tests(unittest.TestCase):
    def test_all_branches_exact_15_fields(self) -> None:
        for name, expected in EXPECTED.items():
            with self.subTest(scenario=name):
                report = _run(name)
                self.assertEqual(report.to_dict(), expected)


if __name__ == "__main__":
    unittest.main()
