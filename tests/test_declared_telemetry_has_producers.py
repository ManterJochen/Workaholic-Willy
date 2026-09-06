"""Five declared telemetry fields had no producer at all. This is the fix, and the guard against a sixth.

MEASURED 2026-08-19 while driving the RL stack over generated scenes: ``fused_view_count``,
``fusion_evidence_quality``, ``multi_view_occlusion_reduced``, ``uncertainty_disagreement`` and
``drift_severity_numeric`` appeared in exactly TWO places in the repo -- read in ``shadow.py``,
declared in ``telemetry_catalog.py`` -- and nothing anywhere wrote them. They were not "default off".
They did not exist.

Meanwhile the geometry fusion computed the very same facts on every pick, stamped them on
``orchestrator._fusion_geometry_telemetry`` as ``fused_views_used`` / ``fused_objects``, and NOTHING
READ THAT EITHER. Two vocabularies for one set of facts, never joined -- and `_project_feature` maps a
missing key to ``0.0`` silently, so the only symptom was a model that would not learn.

⚠ **What this does NOT do.** All five are scene-level. The pairwise ranker differences features within
one scene's candidate list, so a value identical for every candidate differences to exactly zero. These
are worth having for DIAGNOSIS and telemetry; they cannot move a ranker, and the measurement after this
change confirms it: per-candidate live stayed 7/19. That is the expected result, not a shortfall.
"""

from __future__ import annotations

import re
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.robot.execution.autonomous_grasp.shadow import (
    _drift_severity_to_unit,
    _first_number,
)
from src.robot.grasping.loop.pick_loop import _mean_winning_association_score


class FusionEvidenceQualityTests(unittest.TestCase):
    """Mean score over the pairs that WON their assignment -- the association's own confidence."""

    @staticmethod
    def _fused(associations):
        return SimpleNamespace(associations=associations)

    def test_it_averages_only_the_winning_pairs(self) -> None:
        # Object 0 matched candidate 1 (0.9); object 1 matched nothing. The 0.2 belongs to a pair that
        # LOST, and a losing score says how firmly some other object was rejected -- not evidence
        # about the object that was kept.
        fused = self._fused([SimpleNamespace(
            assignment=(1, None), scores=((0.2, 0.9), (0.1, 0.3)),
        )])
        self.assertAlmostEqual(_mean_winning_association_score(fused), 0.9)

    def test_it_averages_across_views(self) -> None:
        fused = self._fused([
            SimpleNamespace(assignment=(0,), scores=((0.8,),)),
            SimpleNamespace(assignment=(0,), scores=((0.6,),)),
        ])
        self.assertAlmostEqual(_mean_winning_association_score(fused), 0.7)

    def test_nothing_matched_is_zero_not_an_error(self) -> None:
        fused = self._fused([SimpleNamespace(assignment=(None, None), scores=((0.4, 0.3), (0.2, 0.1)))])
        self.assertEqual(_mean_winning_association_score(fused), 0.0)

    def test_it_stays_inside_the_unit_interval_the_catalog_declares(self) -> None:
        fused = self._fused([SimpleNamespace(assignment=(0, 1), scores=((1.0, 0.0), (0.0, 1.0)))])
        value = _mean_winning_association_score(fused)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_a_shape_it_does_not_recognise_never_raises(self) -> None:
        # Telemetry must not be the reason a pick fails.
        for broken in (SimpleNamespace(), SimpleNamespace(associations=None),
                       self._fused([SimpleNamespace(assignment=(5,), scores=((0.1,),))])):
            with self.subTest(broken=broken):
                self.assertEqual(_mean_winning_association_score(broken), 0.0)


class TheThreeFusionFieldsHaveAProducerTests(unittest.TestCase):
    def _service(self, telemetry):
        from src.robot.execution.autonomous_grasp.service import AutonomousGraspService

        service = MagicMock(spec=AutonomousGraspService)
        service.runtime = SimpleNamespace(
            orchestrator=SimpleNamespace(_fusion_geometry_telemetry=telemetry)
        )
        return AutonomousGraspService._fusion_geometry_extras(service)

    def test_it_maps_the_names_the_catalog_declares(self) -> None:
        extras = self._service({
            "fused_views_used": 3, "fused_objects": 5,
            "fused_objects_total": 5, "fused_evidence_quality": 0.806474,
        })
        self.assertEqual(extras["fused_view_count"], 3)
        self.assertAlmostEqual(extras["fusion_evidence_quality"], 0.806474)
        self.assertIs(extras["multi_view_occlusion_reduced"], True)

    def test_no_object_gained_anything_is_False_not_absent(self) -> None:
        extras = self._service({"fused_views_used": 2, "fused_objects": 0,
                                "fused_evidence_quality": 0.0})
        self.assertIs(extras["multi_view_occlusion_reduced"], False)

    def test_no_fusion_means_no_keys_at_all(self) -> None:
        # Byte-identical for every cell that does not run geometry fusion, which is the shipped default.
        self.assertEqual(self._service({}), {})
        self.assertEqual(self._service(None), {})


class DriftSeverityIsMappedNotInventedTests(unittest.TestCase):
    """`drift_severity_numeric` is NOT a catalog field; the catalog declares the STRING enum."""

    def test_the_watchdogs_own_order_is_used_normalised(self) -> None:
        self.assertEqual(_drift_severity_to_unit("none"), 0.0)
        self.assertEqual(_drift_severity_to_unit("moderate"), 0.5)
        self.assertEqual(_drift_severity_to_unit("severe"), 1.0)

    def test_it_is_monotone_in_the_watchdogs_order(self) -> None:
        ladder = ["none", "low", "moderate", "high", "severe"]
        values = [_drift_severity_to_unit(name) for name in ladder]
        self.assertEqual(values, sorted(values))
        self.assertEqual(len(set(values)), len(ladder))

    def test_an_unknown_severity_is_None_not_a_guess(self) -> None:
        self.assertIsNone(_drift_severity_to_unit("catastrophic"))
        self.assertIsNone(_drift_severity_to_unit(None))


class DisagreementReadsTheNameThatExistsTests(unittest.TestCase):
    def test_the_catalog_name_wins_when_a_producer_emits_it(self) -> None:
        self.assertEqual(_first_number(0.4, 0.9, 0.1), 0.4)

    def test_it_falls_back_to_channel_disagreement(self) -> None:
        # What the decision engine actually writes today.
        self.assertEqual(_first_number(None, 0.37, None), 0.37)

    def test_then_to_the_snapshot(self) -> None:
        self.assertEqual(_first_number(None, None, 0.12), 0.12)

    def test_nothing_anywhere_is_zero(self) -> None:
        self.assertEqual(_first_number(None, None, None), 0.0)

    def test_a_bool_counts_as_a_number(self) -> None:
        self.assertEqual(_first_number(True), 1.0)
        self.assertEqual(_first_number(False), 0.0)


class TheCatalogSaysWhoWritesThemTests(unittest.TestCase):
    """A declared field with no producer is invisible; the catalog now names its source."""

    def test_the_fusion_fields_are_documented(self) -> None:
        from pathlib import Path

        source = Path(
            "src/robot/grasping/replay/telemetry_catalog.py"
        ).read_text(encoding="utf-8")
        self.assertIn("_fusion_geometry_extras", source)
        # The producer sentence, read across the comment's own line wrapping. The catalog has
        # to name what writes these three and where the value it joins is stamped, or a
        # declared field with no producer is invisible again.
        flowed = re.sub(r"\s*\n\s*#\s*", " ", source)
        self.assertIn(
            "`_fusion_geometry_extras` joins the geometry fusion telemetry stamped on "
            "`orchestrator._fusion_geometry_telemetry` to these catalog names",
            flowed,
        )

    def test_the_disagreement_alias_is_documented(self) -> None:
        from pathlib import Path

        source = Path(
            "src/robot/grasping/replay/telemetry_catalog.py"
        ).read_text(encoding="utf-8")
        self.assertIn("channel_disagreement", source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
