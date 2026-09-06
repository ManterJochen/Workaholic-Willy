"""Coverage by approach tilt — and the two denominators that decide whether it means anything.

⛔ WHY THE METRIC EXISTS. The ladder reports one `top1` per configuration, and that number cannot
distinguish "found a side grasp for a short object" from "found a top grasp for a tall one". A
generator could solve the entire off-vertical population and the headline would barely move.

⛔ AND WHY IT NEARLY LIED. My first version scored only object-views that carry a jaw label. MEASURED:
that is 1,710 of 4,848 — **35.3 %**. The report covered barely a third of what a cell meets and read
as though it covered everything. `NO_LABEL` is counted now, reported and never scored, because there
is no target for a generator to hit there.
"""

from __future__ import annotations

import unittest

from datagen.eval.approach_tilt import (
    BANDS,
    NO_LABEL,
    band_of,
    format_report,
    object_reach_tilt,
    side_approach_report,
)


def _label(scene: str, instance: int, tilt: float, kind: str = "jaw") -> dict:
    return {"scene_id": scene, "instance_id": instance, "kind": kind, "approach_tilt_deg": tilt}


def _row(scene: str, instance: int, *, config: str = "x", outcome: str = "candidate",
         rank: int = 0, valid: bool = True, view: str = "v0") -> dict:
    return {"scene_id": scene, "instance_id": instance, "kind": "jaw", "config": config,
            "view": view, "outcome": outcome, "rank": rank, "valid": valid}


class TheBandsTests(unittest.TestCase):
    def test_the_cuts_are_the_corpus_landmarks(self) -> None:
        """60 is the MEDIAN admissible tilt and 90 the p95 — measured, not chosen. A threshold placed
        by intuition at 45 would put most of this corpus on one side of it and measure nothing."""
        self.assertEqual([name for name, _a, _b in BANDS],
                         ["top_down", "tilted", "side", "under"])
        self.assertEqual([b for _n, _a, b in BANDS], [30.0, 60.0, 90.0, 135.0])

    def test_each_boundary_belongs_to_the_band_above_it(self) -> None:
        self.assertEqual(band_of(0.0), "top_down")
        self.assertEqual(band_of(29.9), "top_down")
        self.assertEqual(band_of(30.0), "tilted")
        self.assertEqual(band_of(60.0), "side")
        self.assertEqual(band_of(90.0), "under")

    def test_past_the_reference_maximum_gets_its_OWN_name(self) -> None:
        """135 deg is `APPROACH_MAX_TILT_DEG`; beyond it the reference REFUSES rather than bins, and
        snapping such a tilt into `under` would hide exactly that."""
        self.assertEqual(band_of(135.0), "beyond_max")
        self.assertEqual(band_of(179.0), "beyond_max")

    def test_a_nonsense_tilt_is_not_silently_a_band(self) -> None:
        for value in (float("nan"), float("inf"), -1.0):
            with self.subTest(value=value):
                self.assertEqual(band_of(value), "unknown")


class TheObjectBandTests(unittest.TestCase):
    def test_the_band_is_the_MINIMUM_tilt_not_the_mean(self) -> None:
        """⚠ One admissible top-down label is enough to say "this can be reached from above",
        however many side labels sit beside it. A mean would file a cylinder with one top grasp and
        twenty side grasps under `side`, then credit a generator for finding the easy one."""
        reach = object_reach_tilt([_label("s", 1, 80.0), _label("s", 1, 10.0),
                                   _label("s", 1, 85.0)])
        self.assertEqual(reach[("s", 1)], 10.0)
        self.assertEqual(band_of(reach[("s", 1)]), "top_down")

    def test_suction_labels_do_not_set_a_JAW_band(self) -> None:
        reach = object_reach_tilt([_label("s", 1, 5.0, kind="suction")])
        self.assertNotIn(("s", 1), reach)

    def test_an_object_with_no_jaw_label_is_absent(self) -> None:
        self.assertEqual(object_reach_tilt([]), {})


class TheDenominatorTests(unittest.TestCase):
    """⛔ THE DEFECT THIS FILE WAS WRITTEN AFTER FINDING. 64.6 % of object-views carry no admissible
    jaw grasp; scoring only the rest is a report on a third of the population that reads like all."""

    def test_unlabelled_objects_are_COUNTED(self) -> None:
        rows = [_row("s", 1), _row("s", 2)]
        report = side_approach_report(rows, {("s", 1): 10.0})
        bands = {band.band: band for band in report["x"]}
        self.assertEqual(bands["top_down"].objects, 1)
        self.assertEqual(bands[NO_LABEL].objects, 1, "the unlabelled object vanished")

    def test_they_are_never_SCORED(self) -> None:
        """No label means no target; a coverage rate there would be a rate over nothing."""
        report = side_approach_report([_row("s", 2)], {})
        text = format_report(report)
        self.assertIn("n/a", text)

    def test_the_header_states_how_much_is_scoreable(self) -> None:
        rows = [_row("s", i) for i in range(1, 5)]
        report = side_approach_report(rows, {("s", 1): 10.0})
        header = format_report(report)
        self.assertIn("scoreable", header)
        self.assertIn("NO admissible jaw grasp", header)

    def test_their_PROPOSAL_rate_is_still_shown(self) -> None:
        """A generator proposing confidently for objects the reference calls ungraspable is telling
        you something about itself, and it is not visible anywhere else."""
        report = side_approach_report([_row("s", 2)], {})
        bands = {band.band: band for band in report["x"]}
        self.assertEqual(bands[NO_LABEL].proposal_rate, 1.0)


class TheRatesTests(unittest.TestCase):
    def test_coverage_counts_ANY_valid_candidate_top1_only_the_best_ranked(self) -> None:
        """Two different questions: "could it have picked this up" and "would it have". A generator
        that finds the grasp at rank 7 is not the same as one that leads with it."""
        rows = [_row("s", 1, rank=0, valid=False), _row("s", 1, rank=3, valid=True)]
        bands = {b.band: b for b in side_approach_report(rows, {("s", 1): 10.0})["x"]}
        self.assertEqual(bands["top_down"].coverage, 1.0)
        self.assertEqual(bands["top_down"].top1_rate, 0.0)

    def test_the_unit_is_scene_view_instance(self) -> None:
        """The same object in two views is two chances, which is the ladder's own unit — otherwise
        the two numbers would be over different populations wearing one word."""
        rows = [_row("s", 1, view="a"), _row("s", 1, view="b", valid=False)]
        bands = {b.band: b for b in side_approach_report(rows, {("s", 1): 10.0})["x"]}
        self.assertEqual(bands["top_down"].objects, 2)
        self.assertEqual(bands["top_down"].coverage, 0.5)

    def test_a_no_candidate_row_is_a_denominator_not_a_skip(self) -> None:
        """An object the generator said nothing about still counts against it."""
        rows = [_row("s", 1, outcome="no_candidate")]
        bands = {b.band: b for b in side_approach_report(rows, {("s", 1): 10.0})["x"]}
        self.assertEqual(bands["top_down"].objects, 1)
        self.assertEqual(bands["top_down"].proposal_rate, 0.0)

    def test_an_empty_band_prints_as_zero_rather_than_vanishing(self) -> None:
        """An absent row reads as "not measured"; a zero row reads as "measured, nothing there" — and
        for the side bands that difference IS the finding."""
        report = side_approach_report([_row("s", 1)], {("s", 1): 10.0})
        text = format_report(report)
        for name, _low, _high in BANDS:
            self.assertIn(name, text)

    def test_two_configurations_are_reported_separately(self) -> None:
        rows = [_row("s", 1, config="deep", valid=False), _row("s", 1, config="sfe_fused")]
        report = side_approach_report(rows, {("s", 1): 10.0})
        self.assertEqual(sorted(report), ["deep", "sfe_fused"])
        self.assertEqual({b.band: b for b in report["deep"]}["top_down"].coverage, 0.0)
        self.assertEqual({b.band: b for b in report["sfe_fused"]}["top_down"].coverage, 1.0)


if __name__ == "__main__":                                           # pragma: no cover
    unittest.main()
