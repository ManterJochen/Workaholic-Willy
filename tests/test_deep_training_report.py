"""The training report, and the two estimator traps it exists to avoid.

⭑ WHY THIS FILE MATTERS MORE THAN A FORMATTER TEST. The report's job is to answer one question --
"was it still learning when it stopped?" -- and this arc has already misread nine training arms for
want of that answer. Both of the tests below encode a mistake that was made in this very module and
then corrected: a ramp estimator reading a step as flat, and a single-epoch maximum read as a level.

⛔ THE FIXTURE IS A RUN DIRECTORY, NOT A LOG, SINCE 2026-09-04. These tests used to synthesise the
retired single-label trainer's log lines and let a regex scrape them back. That reader was removed
because it could not see a live run at all: it wanted `loss X test hit Y (floor Z) ... test width
... mm`, and a `SetLoop` line carries none of it, so `report --log` on any current run matched
nothing and printed an empty run instead of refusing. Writing `epochs.json` is also the honest
fixture, because that file is what the live trainer actually produces, after every epoch.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.robot.grasping.deep.eval.run_report import (
    Epoch,
    build_report,
    format_report,
)


def _run(lifts: list[float], *, resume_after: int | None = None) -> Path:
    """A run directory holding the `epochs.json` a set run writes, with a held-out floor of 0.40.

    ``resume_after`` starts a second FOLD at that index, which is how `parse_run_directory` detects
    a sitting boundary: it breaks on a change of `fold`, where the log reader broke on a RESUMING
    line.
    """
    rows = []
    for index, lift in enumerate(lifts, start=1):
        rows.append({
            "fold": 0 if resume_after is None or index <= resume_after else 1,
            "epoch": index - 1,
            "seconds": 300.0,
            "total": 3.0 - index * 0.001,
            "held_total": 3.0 - index * 0.001,
            "train_top1_hit": 0.5 + lift,
            "held_top1_hit": 0.4 + lift,
            "held_offset_error_mm": 20.0,
        })
    root = Path(tempfile.mkdtemp())
    (root / "epochs.json").write_text(json.dumps({
        "plan_epochs": len(lifts),
        # top_down IS the bar the reader uses: twenty-four earlier arms lost to a fixed
        # constant, so a head that does not clearly beat "point the gripper down" beat nothing.
        "held_floor": [{"fold": 0, "top_down": {"top1_hit": 0.4}},
                       {"fold": 1, "top_down": {"top1_hit": 0.4}}],
        "epochs": rows,
    }), encoding="utf-8")
    return root


class TheStepTrapTests(unittest.TestCase):
    """⛔ THE CORRECTION THIS MODULE CARRIES. A least-squares slope over the final 20 epochs of the
    real v2 run came out NEGATIVE while the run was gaining: the gain arrived as a STEP at epoch
    ~121, and a ramp estimator straddling a step reports the flat parts on either side of it."""

    def test_a_step_at_the_end_is_read_as_improvement(self) -> None:
        lifts = [0.09] * 60 + [0.13] * 20            # flat, then a clean step up
        report = build_report("step", _run(lifts))
        self.assertTrue(report.still_improving, format_report(report))

    def test_a_genuinely_flat_tail_is_read_as_flat(self) -> None:
        """The control. Without it the test above passes for a report that says "improving" always."""
        report = build_report("flat", _run([0.09] * 80))
        self.assertFalse(report.still_improving)

    def test_a_tiny_but_precise_gain_does_not_count_as_improvement(self) -> None:
        """A gain measured to many standard errors is still not a gain worth paying epochs for if it
        is under the 0.02 this arc treats as a real difference."""
        report = build_report("tiny", _run([0.0900] * 40 + [0.0905] * 40))
        self.assertFalse(report.still_improving)


class TheSingleEpochMaximumTests(unittest.TestCase):
    """⚠ On the real run the best single epoch read +0.1216 while its own 15-epoch window averaged
    +0.0810. A maximum over 145 noisy draws is a maximum over 145 noisy draws."""

    def test_the_report_names_the_best_epoch_as_a_maximum_not_a_level(self) -> None:
        lifts = [0.05] * 30 + [0.20] + [0.05] * 30   # one lucky epoch in a flat run
        text = format_report(build_report("spike", _run(lifts)))
        self.assertIn("best single epoch", text)
        self.assertIn("NOT a level", text)

    def test_one_lucky_epoch_does_not_make_the_run_improving(self) -> None:
        report = build_report("spike", _run([0.05] * 30 + [0.20] + [0.05] * 30))
        self.assertFalse(report.still_improving)


class ThePlateauTests(unittest.TestCase):
    def test_it_finds_where_the_curve_settled(self) -> None:
        """The only number here that changes what the next run costs."""
        report = build_report("settle", _run([0.00 + 0.004 * i for i in range(25)] + [0.10] * 55))
        self.assertIsNotNone(report.plateau_epoch)
        assert report.plateau_epoch is not None
        self.assertLess(report.plateau_epoch, 45)
        # ⚠ THE MARKER IS ASCII. It was a non-ASCII glyph until 2026-09-04, and `print` of it
        # raised UnicodeEncodeError on a cp1252 console, taking `deep report` down entirely.
        # run_report.py:438 emits "  [PLATEAU] at epoch {n}: ..." -- the migration lower-cased
        # the prose after the marker; the ASCII marker itself is unchanged.
        self.assertIn("[PLATEAU] at epoch", format_report(report))

    def test_a_run_that_never_settles_says_so(self) -> None:
        report = build_report("climb", _run([0.004 * i for i in range(80)]))
        self.assertIsNone(report.plateau_epoch)
        self.assertIn("never settled", format_report(report))


class TheFloorTests(unittest.TestCase):
    def test_every_metric_is_reported_as_a_lift_over_its_own_floor(self) -> None:
        report = build_report("floors", _run([0.10] * 40))
        self.assertAlmostEqual(report.final.test_lift, 0.10, places=6)
        self.assertIn("LIFT", format_report(report))

    def test_a_run_sitting_on_its_floor_is_called_collapsed(self) -> None:
        """⛔ Seven of nine earlier arms sat EXACTLY on their own floor and every conclusion drawn
        from them was void. The report must refuse to let that read as a result."""
        report = build_report("collapse", _run([0.0] * 40))
        self.assertTrue(report.collapsed)
        self.assertIn("COLLAPSED", format_report(report))


class TheResumeSeamTests(unittest.TestCase):
    def test_compute_hours_exclude_the_gap_between_sittings(self) -> None:
        """⚠ Summing last-minus-first over a resumed run counts the hours the machine was asleep."""
        report = build_report("resumed", _run([0.10] * 40, resume_after=17))
        self.assertEqual(len(report.segments), 2)
        self.assertIn("resumed 1x", format_report(report))

    def test_a_directory_with_no_epochs_refuses_rather_than_reporting_on_nothing(self) -> None:
        """⛔ ValueError, NOT SystemExit. This raised SystemExit until 2026-09-04, and SystemExit
        derives from BaseException: a caller wrapping train-plus-report in `except Exception` does
        not catch it and the process dies. `set_loop.py` already has to name SystemExit explicitly
        in an except clause so that a failed plot cannot end a six-hour training run."""
        root = Path(tempfile.mkdtemp())
        (root / "epochs.json").write_text(json.dumps({"epochs": []}), encoding="utf-8")
        with self.assertRaises(ValueError):
            build_report("bad", root)

    def test_a_missing_directory_is_also_refused(self) -> None:
        with self.assertRaises((ValueError, FileNotFoundError, OSError)):
            build_report("gone", Path(tempfile.mkdtemp()) / "nope")


# ⛔ `TheMeasuredConstantsTests` WAS DELETED HERE ON 2026-09-04, with the constants it pinned.
# `MEASURED_APPROACH_POS_WEIGHT` (13.35) and `IRREDUCIBLE_APPROACH_CROSS_ENTROPY_NATS` (1.4198)
# were properties of the SINGLE-LABEL approach head in the retired binned trainer. Both were real
# measurements and both are now unreadable in the only sense that matters: there is no model they
# describe. A constant that outlives its architecture is exactly the thing someone quotes later
# without knowing what it measured.


if __name__ == "__main__":                                           # pragma: no cover
    unittest.main()


def _epochs(count: int, lift: float = 0.05):
    from datetime import datetime, timedelta

    base = datetime(2000, 1, 1)
    return [Epoch(stamp=base + timedelta(seconds=600 * i), fold=0, epoch=i + 1, total=0,
                  loss=1.0, test_hit=0.4 + lift, test_floor=0.4,
                  train_hit=0.4 + lift, train_floor=0.4, held_offset_error_mm=20.0)
            for i in range(count)]


from src.robot.grasping.deep.eval.run_report import _build_from_epochs  # noqa: E402


def _short_report():
    return _build_from_epochs("early", _epochs(4), [(1, 4)])


def _long_report():
    return _build_from_epochs("long", _epochs(40), [(1, 40)])


class TooEarlyTests(unittest.TestCase):
    """⛔⛔ A REPORT THAT ANSWERS A QUESTION IT CANNOT ANSWER IS WORSE THAN ONE THAT DECLINES.

    MEASURED on a live run: with four epochs of a thirty-six-epoch schedule on disk, this module
    printed "it had FLATTENED before it stopped... More epochs are not the lever; resolution, the
    target and the corpus are the open ones." That verdict came from a PLACEHOLDER: below two full
    windows the comparison is the mean against itself, `still_improving` is set False, and the
    formatter read False as a finding.
    """

    def test_a_SHORT_run_gives_no_window_verdict(self) -> None:
        text = format_report(_short_report())
        self.assertIn("[TOO EARLY]", text)
        self.assertNotIn("[FLATTENED]", text)
        self.assertNotIn("[STILL IMPROVING]", text)

    def test_the_placeholder_COMPARISON_LINE_is_not_printed_either(self) -> None:
        """It read `+0.0047 -> +0.0047 (difference +0.0000 +- 0.0000)`, which looks like a converged
        run rather than like missing data."""
        self.assertNotIn("epochs vs the", format_report(_short_report()))

    def test_a_LONG_run_still_gets_its_verdict(self) -> None:
        """The control. A gate that silenced every run would pass the two tests above."""
        text = format_report(_long_report())
        self.assertNotIn("[TOO EARLY]", text)
        self.assertIn("epochs vs the", text)

    def test_an_unfinished_run_is_not_reported_as_finished(self) -> None:
        """`4 epoch(s) of 4` for a run four epochs into thirty-six is a completed run on its face."""
        self.assertIn("so far", format_report(_short_report()))


class BelowResolutionTests(unittest.TestCase):
    """⛔⛔ A VERDICT THAT IS ARITHMETIC RATHER THAN AN OBSERVATION.

    `_REAL_DIFFERENCE` is an ABSOLUTE 0.02. This arc's lifts sit around 0.006, so every epoch falls
    inside the band by construction and "[PLATEAU] AT EPOCH 1: book 1 epoch next time" is guaranteed.
    MEASURED on the live `film` arm at seven epochs, that is exactly what it printed, alongside "More
    epochs are not the lever" from a window comparison whose own standard error exceeded its
    difference.
    """

    def test_a_SMALL_curve_gets_no_plateau_verdict(self) -> None:
        report = _build_from_epochs("small", _epochs(20, lift=0.006), [(1, 20)])
        text = format_report(report)
        self.assertIn("[BELOW RESOLUTION]", text)
        self.assertNotIn("[PLATEAU] AT EPOCH", text)

    def test_a_SMALL_curve_gets_no_schedule_claim_either(self) -> None:
        """The window comparison still reports its number; what it stops doing is concluding from a
        threshold nothing in the run could reach."""
        text = format_report(_build_from_epochs("small", _epochs(20, lift=0.006), [(1, 20)]))
        self.assertIn("[NO WINDOW VERDICT]", text)
        self.assertNotIn("More epochs are not the lever", text)

    def test_a_LARGE_curve_still_gets_both(self) -> None:
        """The control. A gate that silenced every run would pass the two tests above."""
        from datetime import datetime, timedelta

        base = datetime(2000, 1, 1)
        # A curve that climbs well past the band and then settles.
        lifts = [min(0.30, 0.02 * i) for i in range(40)]
        epochs = [Epoch(stamp=base + timedelta(seconds=600 * i), fold=0, epoch=i + 1, total=0,
                        loss=1.0, test_hit=0.4 + lift, test_floor=0.4,
                        train_hit=0.4 + lift, train_floor=0.4, held_offset_error_mm=20.0)
                  for i, lift in enumerate(lifts)]
        text = format_report(_build_from_epochs("large", epochs, [(1, 40)]))
        self.assertNotIn("[BELOW RESOLUTION]", text)

    def test_the_report_names_the_window_it_ACTUALLY_USED(self) -> None:
        """⛔ Every verdict line said "the final 15 epochs" while a 7-epoch run compared 3 against 3.
        A number in a report has to be the one that was computed."""
        report = _build_from_epochs("short", _epochs(8), [(1, 8)])
        self.assertEqual(report.window, 4)
        text = format_report(report)
        self.assertIn("last 4 epochs vs the 4 before", text)
        self.assertNotIn("last 15 epochs", text)
