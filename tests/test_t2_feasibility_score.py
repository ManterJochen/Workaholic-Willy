"""Phase T2 — pure-function feasibility scoring.

The :mod:`backend.src.robot.grasping.scoring.feasibility_score`
module is a *ranking* signal. It never rejects candidates; the
calculator's existing hard-drop pipeline (``_apply_ik_filter``,
swept-approach ``BLOCKED`` outcomes) keeps that responsibility.

T2 scaffolding intent: the *carrier*, *math*, *config* and *typed
inputs* land now so the rest of the calculator wiring is reviewable
and testable. The individual signal flags
(``ik_quality_enabled`` / ``joint_margin_enabled`` /
``swept_approach_enabled``) default to :data:`False` so the score
contributes nothing to ``total_score`` until the operator opts in
per-signal.

This test module pins:

1. :class:`FeasibilityScoreConfig` defaults, validation, weight
   normalisation.
2. :func:`feasibility_grasp_score` returns 0.5 (neutral) when no
   inputs are provided — required by the locked operator answer
   (``fail_closed_stance: A``).
3. Per-signal monotonicity (lower condition number ⇒ higher score;
   wider joint margin ⇒ higher score; ``CLEAR`` approach ⇒ higher
   score than no signal which beats ``BLOCKED``).
4. :func:`feasibility_score_components` returns a stable 3-key
   dict (``ik_quality``, ``joint_margin``, ``approach_clearance``)
   so downstream telemetry layout is predictable.
5. Disabled signals contribute the neutral 0.5 floor — *not* 0.0 —
   so they never demote a candidate below an enabled-but-unknown
   peer (operator answer locked: missing-quality ⇒ neutral).
"""

from __future__ import annotations

import unittest

from src.robot.grasping.planning.reachability import IKQualityMetrics
from src.robot.grasping.scoring.feasibility_score import (
    FeasibilityScoreConfig,
    feasibility_grasp_score,
    feasibility_score_components,
)
from src.robot.grasping.motion.trajectory_safety import (
    ApproachPathOutcome,
    ApproachPathReport,
)


def _approach(outcome: ApproachPathOutcome) -> ApproachPathReport:
    return ApproachPathReport(outcome=outcome)


class FeasibilityScoreConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        c = FeasibilityScoreConfig()
        # All sub-signals off ⇒ score collapses to neutral floor.
        self.assertFalse(c.ik_quality_enabled)
        self.assertFalse(c.joint_margin_enabled)
        self.assertFalse(c.swept_approach_enabled)
        # Internal sub-weights — only relevant when their signal is on.
        self.assertAlmostEqual(c.ik_quality_weight, 0.4)
        self.assertAlmostEqual(c.joint_margin_weight, 0.3)
        self.assertAlmostEqual(c.swept_approach_weight, 0.3)
        # IK-quality reference scales.
        self.assertGreater(c.condition_number_soft_max, 0.0)
        self.assertGreater(c.min_singular_value_floor, 0.0)
        self.assertGreater(c.joint_margin_full_score_deg, 0.0)

    def test_rejects_negative_weights(self) -> None:
        with self.assertRaises(ValueError):
            FeasibilityScoreConfig(ik_quality_weight=-0.1)
        with self.assertRaises(ValueError):
            FeasibilityScoreConfig(joint_margin_weight=-0.5)
        with self.assertRaises(ValueError):
            FeasibilityScoreConfig(swept_approach_weight=-1.0)

    def test_rejects_zero_total_weight(self) -> None:
        # If the operator enables a signal but all sub-weights are
        # 0, the score becomes undefined. Reject at construction.
        # T3 added corridor_risk_weight to the total-weight check,
        # so we zero it here too to exercise the rejection path.
        with self.assertRaises(ValueError):
            FeasibilityScoreConfig(
                ik_quality_weight=0.0,
                joint_margin_weight=0.0,
                swept_approach_weight=0.0,
                corridor_risk_weight=0.0,
            )


class FeasibilityScoreNeutralTests(unittest.TestCase):
    """All signals disabled ⇒ neutral 0.5 floor (operator-locked)."""

    def test_no_inputs_no_flags_returns_neutral(self) -> None:
        score = feasibility_grasp_score(
            ik_quality=None,
            approach_path=None,
            config=FeasibilityScoreConfig(),
        )
        self.assertEqual(score, 0.5)

    def test_inputs_present_but_flags_off_returns_neutral(self) -> None:
        # Even when inputs are available, disabled flags collapse the
        # score to neutral. This is the scaffolding contract: the
        # signals are wired but inert until per-signal flags flip.
        score = feasibility_grasp_score(
            ik_quality=IKQualityMetrics(
                condition_number=10.0,
                min_singular_value=0.1,
                joint_margin_deg=20.0,
            ),
            approach_path=_approach(ApproachPathOutcome.CLEAR),
            config=FeasibilityScoreConfig(),
        )
        self.assertEqual(score, 0.5)

    def test_missing_input_under_enabled_flag_returns_neutral_component(
        self,
    ) -> None:
        # ik_quality_enabled=True but no IKQualityMetrics ⇒ that
        # sub-signal contributes neutral 0.5 (operator answer:
        # "missing quality on legacy IK ⇒ neutral 0.5, mode-agnostic").
        score = feasibility_grasp_score(
            ik_quality=None,
            approach_path=None,
            config=FeasibilityScoreConfig(ik_quality_enabled=True),
        )
        self.assertEqual(score, 0.5)


class FeasibilityScoreMonotonicityTests(unittest.TestCase):
    def test_lower_condition_number_scores_higher(self) -> None:
        cfg = FeasibilityScoreConfig(
            ik_quality_enabled=True, condition_number_soft_max=100.0
        )
        good = feasibility_grasp_score(
            ik_quality=IKQualityMetrics(condition_number=10.0),
            approach_path=None,
            config=cfg,
        )
        bad = feasibility_grasp_score(
            ik_quality=IKQualityMetrics(condition_number=80.0),
            approach_path=None,
            config=cfg,
        )
        self.assertGreater(good, bad)

    def test_wider_joint_margin_scores_higher(self) -> None:
        cfg = FeasibilityScoreConfig(
            joint_margin_enabled=True, joint_margin_full_score_deg=30.0
        )
        good = feasibility_grasp_score(
            ik_quality=IKQualityMetrics(joint_margin_deg=25.0),
            approach_path=None,
            config=cfg,
        )
        bad = feasibility_grasp_score(
            ik_quality=IKQualityMetrics(joint_margin_deg=2.0),
            approach_path=None,
            config=cfg,
        )
        self.assertGreater(good, bad)

    def test_blocked_approach_scores_lower_than_clear(self) -> None:
        cfg = FeasibilityScoreConfig(swept_approach_enabled=True)
        clear = feasibility_grasp_score(
            ik_quality=None,
            approach_path=_approach(ApproachPathOutcome.CLEAR),
            config=cfg,
        )
        blocked = feasibility_grasp_score(
            ik_quality=None,
            approach_path=_approach(ApproachPathOutcome.BLOCKED),
            config=cfg,
        )
        self.assertGreater(clear, blocked)

    def test_no_obstacles_treated_as_clear(self) -> None:
        cfg = FeasibilityScoreConfig(swept_approach_enabled=True)
        clear = feasibility_grasp_score(
            ik_quality=None,
            approach_path=_approach(ApproachPathOutcome.CLEAR),
            config=cfg,
        )
        no_obs = feasibility_grasp_score(
            ik_quality=None,
            approach_path=_approach(ApproachPathOutcome.NO_OBSTACLES),
            config=cfg,
        )
        self.assertEqual(clear, no_obs)

    def test_skipped_treated_as_neutral(self) -> None:
        # SKIPPED ⇒ adapter did not evaluate; do not punish.
        cfg = FeasibilityScoreConfig(swept_approach_enabled=True)
        score = feasibility_grasp_score(
            ik_quality=None,
            approach_path=_approach(ApproachPathOutcome.SKIPPED),
            config=cfg,
        )
        self.assertEqual(score, 0.5)


class FeasibilityScoreBoundsTests(unittest.TestCase):
    def test_score_always_in_unit_interval(self) -> None:
        cfg = FeasibilityScoreConfig(
            ik_quality_enabled=True,
            joint_margin_enabled=True,
            swept_approach_enabled=True,
        )
        cases = [
            (
                IKQualityMetrics(
                    condition_number=1.0,
                    min_singular_value=0.5,
                    joint_margin_deg=45.0,
                ),
                _approach(ApproachPathOutcome.CLEAR),
            ),
            (
                IKQualityMetrics(
                    condition_number=9_999.0,
                    min_singular_value=1e-9,
                    joint_margin_deg=0.0,
                ),
                _approach(ApproachPathOutcome.BLOCKED),
            ),
            (None, None),
        ]
        for ikq, ap in cases:
            with self.subTest(ikq=ikq, ap=ap):
                score = feasibility_grasp_score(
                    ik_quality=ikq, approach_path=ap, config=cfg
                )
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)


class FeasibilityComponentsLayoutTests(unittest.TestCase):
    """Locks the telemetry sub-dict layout for predictability."""

    def test_three_stable_keys(self) -> None:
        comps = feasibility_score_components(
            ik_quality=IKQualityMetrics(
                condition_number=20.0,
                min_singular_value=0.05,
                joint_margin_deg=10.0,
            ),
            approach_path=_approach(ApproachPathOutcome.CLEAR),
            config=FeasibilityScoreConfig(
                ik_quality_enabled=True,
                joint_margin_enabled=True,
                swept_approach_enabled=True,
            ),
        )
        self.assertTrue(
            {"ik_quality", "joint_margin", "approach_clearance"}.issubset(
                comps.keys()
            )
        )
        for value in comps.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_keys_present_even_when_signals_disabled(self) -> None:
        # Stable layout ⇒ telemetry consumers can always read these
        # keys without conditional logic.
        comps = feasibility_score_components(
            ik_quality=None,
            approach_path=None,
            config=FeasibilityScoreConfig(),
        )
        self.assertTrue(
            {"ik_quality", "joint_margin", "approach_clearance"}.issubset(
                comps.keys()
            )
        )
        for value in comps.values():
            self.assertEqual(value, 0.5)


if __name__ == "__main__":
    unittest.main()
