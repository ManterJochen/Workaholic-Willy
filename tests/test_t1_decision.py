"""Phase T1 — decision policy unit tests.

These tests pin the pure-function behaviour of the new
:mod:`backend.src.robot.grasping.decision` module:

* The :class:`DecisionPolicy` dataclass is frozen and validates ranges.
* The :class:`DecisionEngine` decides ``GRASP_NOW`` / ``MOVE_CAMERA`` /
  ``RECOVER`` / ``FAIL_CLOSED`` from a typed input bag.
* The mandatory plan rule #12 fields appear on every
  :class:`DecisionReport`: ``decision_action``, ``decision_reason_code``,
  ``uncertainty_score``, ``threshold_used``, ``mode``, ``attempt_id``.
* Fail-closed only triggers on real hardware (operator-locked T1
  scope) and only after the re-observation budget is exhausted or no
  planner is wired.

The locked numeric defaults follow operator confirmation:

* ``auto_uncertainty_threshold = 0.4`` (i.e. ``top_score >= 0.6`` to
  grasp).
* ``max_reobservations = 2``.
* ``reasons_penalty = 0.2``.
* ``fail_closed_on_real_hardware = True``.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.decision import (
    DecisionAction,
    DecisionEngine,
    DecisionPolicy,
    DecisionReasonCode,
    DecisionReport,
)
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grasp_result(
    score: float, *, reasons: tuple = (), candidates: int = 1
) -> GraspResult:
    points = tuple(
        GraspPoint(
            position=np.array([100.0, 50.0, 400.0]),
            approach=np.array([0.0, 0.0, 1.0]),
            axis=np.array([1.0, 0.0, 0.0]),
            grip_width_mm=40.0,
            score=score,
            frame=GraspFrame.BASE,
            label="t",
        )
        for _ in range(candidates)
    )
    return GraspResult(candidates=points, reasons=reasons, top_score=score if points else 0.0)


def _empty_result() -> GraspResult:
    return GraspResult(candidates=(), reasons=(), top_score=0.0)


def _engine(**overrides) -> DecisionEngine:
    kwargs = dict(
        enabled=True,
        auto_uncertainty_threshold=0.4,
        max_reobservations=2,
        reasons_penalty=0.2,
        fail_closed_on_real_hardware=True,
    )
    kwargs.update(overrides)
    return DecisionEngine(policy=DecisionPolicy(**kwargs))


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class DecisionPolicyTests(unittest.TestCase):
    """The policy is frozen and validates its numeric ranges."""

    def test_defaults_match_locked_values(self) -> None:
        p = DecisionPolicy()
        self.assertTrue(p.enabled)
        self.assertEqual(p.auto_uncertainty_threshold, 0.4)
        self.assertEqual(p.max_reobservations, 2)
        self.assertEqual(p.reasons_penalty, 0.2)
        self.assertTrue(p.fail_closed_on_real_hardware)

    def test_policy_is_frozen(self) -> None:
        p = DecisionPolicy()
        with self.assertRaises((AttributeError, TypeError)):
            p.auto_uncertainty_threshold = 0.9  # type: ignore[misc]

    def test_threshold_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            DecisionPolicy(auto_uncertainty_threshold=-0.1)
        with self.assertRaises(ValueError):
            DecisionPolicy(auto_uncertainty_threshold=1.5)

    def test_negative_reobservations_raises(self) -> None:
        with self.assertRaises(ValueError):
            DecisionPolicy(max_reobservations=-1)

    def test_reasons_penalty_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            DecisionPolicy(reasons_penalty=-0.1)
        with self.assertRaises(ValueError):
            DecisionPolicy(reasons_penalty=1.1)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class DecisionReportTests(unittest.TestCase):
    """The report carries every plan-mandated structured field."""

    def test_report_is_frozen(self) -> None:
        r = DecisionReport(
            action=DecisionAction.GRASP_NOW,
            reason_code=DecisionReasonCode.CONFIDENT_GRASP,
            uncertainty_score=0.2,
            threshold_used=0.4,
            mode="auto",
            attempt_id="a-1",
            reobservation_count=0,
        )
        with self.assertRaises((AttributeError, TypeError)):
            r.action = DecisionAction.MOVE_CAMERA  # type: ignore[misc]

    def test_to_dict_contains_mandatory_fields(self) -> None:
        # Plan rule #12 mandates these exact keys be machine-readable
        # on every fail-closed decision log entry.
        r = DecisionReport(
            action=DecisionAction.FAIL_CLOSED,
            reason_code=DecisionReasonCode.REOBSERVE_BUDGET_EXHAUSTED,
            uncertainty_score=0.7,
            threshold_used=0.4,
            mode="auto",
            attempt_id="a-42",
            reobservation_count=2,
            top_score=0.3,
        )
        d = r.to_dict()
        for key in (
            "decision_action",
            "decision_reason_code",
            "uncertainty_score",
            "threshold_used",
            "mode",
            "attempt_id",
            "reobservation_count",
            "top_score",
        ):
            self.assertIn(key, d)
        self.assertEqual(d["decision_action"], "fail_closed")
        self.assertEqual(d["decision_reason_code"], "reobserve_budget_exhausted")
        self.assertEqual(d["uncertainty_score"], 0.7)
        self.assertEqual(d["threshold_used"], 0.4)
        self.assertEqual(d["mode"], "auto")
        self.assertEqual(d["attempt_id"], "a-42")
        self.assertEqual(d["reobservation_count"], 2)
        self.assertEqual(d["top_score"], 0.3)


# ---------------------------------------------------------------------------
# Engine — confident grasp path
# ---------------------------------------------------------------------------


class DecisionEngineGraspNowTests(unittest.TestCase):
    """High-confidence candidates should be grasped immediately."""

    def test_high_score_grasps_now(self) -> None:
        eng = _engine()
        rep = eng.decide(
            grasp_result=_grasp_result(0.9),
            mode="auto",
            attempt_id="a-1",
            reobservation_count=0,
            is_simulated=False,
            viewpoint_planner_available=True,
        )
        self.assertIs(rep.action, DecisionAction.GRASP_NOW)
        self.assertIs(rep.reason_code, DecisionReasonCode.CONFIDENT_GRASP)
        # uncertainty = 1 - 0.9 = 0.1
        self.assertAlmostEqual(rep.uncertainty_score, 0.1, places=6)
        self.assertEqual(rep.threshold_used, 0.4)
        self.assertEqual(rep.top_score, 0.9)

    def test_at_threshold_boundary_grasps(self) -> None:
        # top_score = 0.6  → uncertainty = 0.4 == threshold → GRASP
        eng = _engine()
        rep = eng.decide(
            grasp_result=_grasp_result(0.6),
            mode="auto",
            attempt_id="a-1",
            reobservation_count=0,
            is_simulated=False,
            viewpoint_planner_available=True,
        )
        self.assertIs(rep.action, DecisionAction.GRASP_NOW)


# ---------------------------------------------------------------------------
# Engine — reobservation path
# ---------------------------------------------------------------------------


class DecisionEngineMoveCameraTests(unittest.TestCase):
    """Low-confidence candidates with budget left should re-observe."""

    def test_low_score_with_budget_moves_camera(self) -> None:
        eng = _engine()  # max_reobservations=2
        rep = eng.decide(
            grasp_result=_grasp_result(0.4),
            mode="auto",
            attempt_id="a-1",
            reobservation_count=0,
            is_simulated=False,
            viewpoint_planner_available=True,
        )
        self.assertIs(rep.action, DecisionAction.MOVE_CAMERA)
        self.assertIs(rep.reason_code, DecisionReasonCode.LOW_CONFIDENCE)
        self.assertAlmostEqual(rep.uncertainty_score, 0.6, places=6)

    def test_reasons_flag_inflates_uncertainty(self) -> None:
        # top_score=0.7 → uncertainty = 0.3; with reasons_penalty=0.2 → 0.5 > 0.4
        eng = _engine()
        rep = eng.decide(
            grasp_result=_grasp_result(
                0.7, reasons=("antipodal_pair_not_found",)
            ),
            mode="auto",
            attempt_id="a-1",
            reobservation_count=0,
            is_simulated=False,
            viewpoint_planner_available=True,
        )
        self.assertIs(rep.action, DecisionAction.MOVE_CAMERA)
        self.assertAlmostEqual(rep.uncertainty_score, 0.5, places=6)


# ---------------------------------------------------------------------------
# Engine — fail-closed paths (real hardware)
# ---------------------------------------------------------------------------


class DecisionEngineFailClosedTests(unittest.TestCase):
    """Real-hardware fail-closed when budget or planner is gone."""

    def test_budget_exhausted_on_real_hardware_fail_closes(self) -> None:
        eng = _engine()  # max_reobservations=2
        rep = eng.decide(
            grasp_result=_grasp_result(0.3),
            mode="auto",
            attempt_id="a-1",
            reobservation_count=2,  # already used all attempts
            is_simulated=False,
            viewpoint_planner_available=True,
        )
        self.assertIs(rep.action, DecisionAction.FAIL_CLOSED)
        self.assertIs(
            rep.reason_code, DecisionReasonCode.REOBSERVE_BUDGET_EXHAUSTED
        )
        # Plan rule #12 — every fail-closed decision must carry these.
        d = rep.to_dict()
        self.assertEqual(d["decision_action"], "fail_closed")
        self.assertIsNotNone(d["uncertainty_score"])
        self.assertEqual(d["threshold_used"], 0.4)

    def test_planner_unavailable_on_real_hardware_fail_closes(self) -> None:
        eng = _engine()
        rep = eng.decide(
            grasp_result=_grasp_result(0.3),
            mode="auto",
            attempt_id="a-1",
            reobservation_count=0,
            is_simulated=False,
            viewpoint_planner_available=False,
        )
        self.assertIs(rep.action, DecisionAction.FAIL_CLOSED)
        self.assertIs(
            rep.reason_code, DecisionReasonCode.REOBSERVE_PLANNER_UNAVAILABLE
        )

    def test_no_candidates_recovers_not_fail_closed(self) -> None:
        # The "no candidates" branch is a RECOVER signal because the
        # honest action is to invoke recovery (T5 will execute it).
        # T1 emits the typed decision; the service maps to a typed
        # report. uncertainty is None because we can't compute one.
        eng = _engine()
        rep = eng.decide(
            grasp_result=_empty_result(),
            mode="dense_clutter",
            attempt_id="a-1",
            reobservation_count=0,
            is_simulated=False,
            viewpoint_planner_available=True,
        )
        self.assertIs(rep.action, DecisionAction.RECOVER)
        self.assertIs(rep.reason_code, DecisionReasonCode.NO_CANDIDATES)
        self.assertIsNone(rep.uncertainty_score)

    def test_simulated_hardware_downgrades_to_grasp_now(self) -> None:
        # On simulated hardware the plan permits the decision layer to
        # stay permissive (rule #11 talks about *real hardware*).
        # When the budget is exhausted in sim, the engine must NOT
        # fail-close — it should grasp the best-available candidate.
        eng = _engine()
        rep = eng.decide(
            grasp_result=_grasp_result(0.3),
            mode="auto",
            attempt_id="a-1",
            reobservation_count=2,
            is_simulated=True,
            viewpoint_planner_available=True,
        )
        self.assertIs(rep.action, DecisionAction.GRASP_NOW)
        # And the reason code says "we'd have fail-closed on real HW".
        self.assertIs(
            rep.reason_code, DecisionReasonCode.REOBSERVE_BUDGET_EXHAUSTED
        )

    def test_fail_closed_disabled_policy_downgrades(self) -> None:
        # Operator override: fail_closed_on_real_hardware=False keeps
        # the legacy permissive behaviour.
        eng = _engine(fail_closed_on_real_hardware=False)
        rep = eng.decide(
            grasp_result=_grasp_result(0.3),
            mode="auto",
            attempt_id="a-1",
            reobservation_count=2,
            is_simulated=False,
            viewpoint_planner_available=True,
        )
        self.assertIs(rep.action, DecisionAction.GRASP_NOW)


if __name__ == "__main__":
    unittest.main()
