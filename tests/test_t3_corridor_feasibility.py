"""Phase T3 — feasibility_score gets a 4th signal: corridor_risk.

Locked design intent (operator answer Q6-A in T3 Q&A):

* corridor_risk is the *4th* signal inside
  :class:`FeasibilityScoreConfig`. The existing 3 signals
  (``ik_quality``, ``joint_margin``, ``swept_approach``) are
  preserved byte-identical.
* :class:`FeasibilityInputs` gains an optional
  ``corridor_report: CorridorReport | None`` field. Missing/None ⇒
  neutral 0.5 (same treatment as the other 3 signals).
* Disabled by default ⇒ T2 byte-identical ranking is preserved.
* Monotonicity: lower blockage_confidence ⇒ higher subscore.
* :func:`feasibility_score_components` extends its key set to
  ``{ik_quality, joint_margin, approach_clearance, corridor_risk}``
  (additive; existing 3 keys always present; new key always
  present, even when disabled — neutral 0.5).
"""

from __future__ import annotations

import unittest

from src.robot.grasping.scoring.corridor import (
    CorridorMode,
    CorridorReport,
)
from src.robot.grasping.scoring.feasibility_score import (
    FeasibilityInputs,
    FeasibilityScoreConfig,
    feasibility_grasp_score,
    feasibility_score_components,
)


def _report(
    *,
    mode: CorridorMode,
    confidence: float,
    approach_mm: float = 100.0,
    retreat_mm: float = 100.0,
) -> CorridorReport:
    return CorridorReport(
        approach_clearance_mm=approach_mm,
        retreat_clearance_mm=retreat_mm,
        blockage_confidence=confidence,
        mode=mode,
    )


class FeasibilityInputsExtensionTests(unittest.TestCase):
    def test_inputs_accept_optional_corridor_report(self) -> None:
        inp = FeasibilityInputs(
            corridor_report=_report(mode=CorridorMode.CLEAR, confidence=0.1),
        )
        self.assertIsNotNone(inp.corridor_report)
        self.assertEqual(inp.corridor_report.mode, CorridorMode.CLEAR)

    def test_inputs_default_corridor_to_none(self) -> None:
        inp = FeasibilityInputs()
        self.assertIsNone(inp.corridor_report)


class FeasibilityConfigCorridorRiskTests(unittest.TestCase):
    def test_corridor_risk_disabled_by_default(self) -> None:
        c = FeasibilityScoreConfig()
        self.assertFalse(c.corridor_risk_enabled)
        # Locked default sub-weight from T3 plan.
        self.assertAlmostEqual(c.corridor_risk_weight, 0.25)

    def test_corridor_risk_weight_must_be_non_negative(self) -> None:
        with self.assertRaises(ValueError):
            FeasibilityScoreConfig(corridor_risk_weight=-0.1)


class FeasibilityCorridorBackCompatTests(unittest.TestCase):
    """T2 byte-identical ranking with corridor_risk disabled."""

    def test_disabled_corridor_returns_neutral_when_only_signal_enabled(
        self,
    ) -> None:
        # Only corridor_risk is enabled but no report ⇒ neutral 0.5.
        cfg = FeasibilityScoreConfig(corridor_risk_enabled=True)
        score = feasibility_grasp_score(
            ik_quality=None,
            approach_path=None,
            corridor_report=None,
            config=cfg,
        )
        self.assertEqual(score, 0.5)

    def test_corridor_flag_off_ignores_input(self) -> None:
        # Flag off ⇒ even a fully BLOCKED report cannot demote.
        cfg = FeasibilityScoreConfig()  # all flags off
        score = feasibility_grasp_score(
            ik_quality=None,
            approach_path=None,
            corridor_report=_report(
                mode=CorridorMode.BLOCKED, confidence=1.0
            ),
            config=cfg,
        )
        self.assertEqual(score, 0.5)


class FeasibilityCorridorMonotonicityTests(unittest.TestCase):
    def test_clear_outranks_blocked(self) -> None:
        cfg = FeasibilityScoreConfig(corridor_risk_enabled=True)
        clear = feasibility_grasp_score(
            ik_quality=None,
            approach_path=None,
            corridor_report=_report(mode=CorridorMode.CLEAR, confidence=0.0),
            config=cfg,
        )
        blocked = feasibility_grasp_score(
            ik_quality=None,
            approach_path=None,
            corridor_report=_report(mode=CorridorMode.BLOCKED, confidence=1.0),
            config=cfg,
        )
        self.assertGreater(clear, blocked)
        self.assertAlmostEqual(clear, 1.0)
        self.assertAlmostEqual(blocked, 0.0)

    def test_partial_between_clear_and_blocked(self) -> None:
        cfg = FeasibilityScoreConfig(corridor_risk_enabled=True)
        partial = feasibility_grasp_score(
            ik_quality=None,
            approach_path=None,
            corridor_report=_report(
                mode=CorridorMode.PARTIAL, confidence=0.5
            ),
            config=cfg,
        )
        self.assertGreater(partial, 0.0)
        self.assertLess(partial, 1.0)

    def test_skipped_treated_as_neutral(self) -> None:
        cfg = FeasibilityScoreConfig(corridor_risk_enabled=True)
        score = feasibility_grasp_score(
            ik_quality=None,
            approach_path=None,
            corridor_report=_report(
                mode=CorridorMode.SKIPPED, confidence=0.5
            ),
            config=cfg,
        )
        self.assertEqual(score, 0.5)


class FeasibilityComponentsLayoutExtensionTests(unittest.TestCase):
    """T3 extends the components dict by one stable key."""

    def test_four_stable_keys_always_present(self) -> None:
        comps = feasibility_score_components(
            ik_quality=None,
            approach_path=None,
            corridor_report=None,
            config=FeasibilityScoreConfig(),
        )
        self.assertEqual(
            set(comps.keys()),
            {
                "ik_quality",
                "joint_margin",
                "approach_clearance",
                "corridor_risk",
            },
        )
        # Disabled signals all neutral.
        for v in comps.values():
            self.assertEqual(v, 0.5)

    def test_corridor_risk_subscore_responds_to_confidence(self) -> None:
        cfg = FeasibilityScoreConfig(corridor_risk_enabled=True)
        comps_clear = feasibility_score_components(
            ik_quality=None,
            approach_path=None,
            corridor_report=_report(mode=CorridorMode.CLEAR, confidence=0.0),
            config=cfg,
        )
        comps_blocked = feasibility_score_components(
            ik_quality=None,
            approach_path=None,
            corridor_report=_report(mode=CorridorMode.BLOCKED, confidence=1.0),
            config=cfg,
        )
        self.assertAlmostEqual(comps_clear["corridor_risk"], 1.0)
        self.assertAlmostEqual(comps_blocked["corridor_risk"], 0.0)


if __name__ == "__main__":
    unittest.main()
