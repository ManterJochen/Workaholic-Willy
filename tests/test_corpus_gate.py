"""The dataset gate: four checks, each pinned against the defect that motivated it.

Every assertion here is a thing that already happened in this project, not a thing that might:

* aletheia's ``quality`` is 0.000000 over all 218,529 rows -- a dropped return value, not missing data
* our own ``contact_angle_deg`` is one value over 51,746 labels and 4,549 distinct over the SAME field
  in the calculator's candidates: constant in training, alive at serving
* aletheia's ``split`` is 'train' on every row, so its 467 objects have no held-out set at all
* the pilot's hold rate is 5.59 %, with 120 of 463 objects holding no positive
* and three separate frame defects, which is why a spec carries its frame and a mismatch is refused
  rather than compared

The last test in this file is the one that keeps the gate honest about itself: `rl/readiness.py`
computes the same statistics for `GraspAttemptRecord` logs, and this package deliberately re-implements
them in numpy rather than importing across the stack. That duplication is only safe if it is checked,
so both are run over the same rows and asserted to agree.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.deep.ranker.features import (
    BOOTSTRAP_JAW_V1,
    BOOTSTRAP_OBJECT_FRAME_V0,
    WILLY_JAW_WITH_CONTACT_ANGLE,
    FeatureSpec,
    spec_named,
)
from datagen.corpus import assess_corpus, column_stats, format_verdict

SPEC = FeatureSpec(
    name="test_spec", frame="object",
    features=("width_mm", "tilt_deg"), label="held", group="object_id",
)


def _corpus(rows: int = 40, *, width=None, tilt=None, held=None, groups=None) -> dict:
    """A small healthy corpus, with any column overridable to make it unhealthy on purpose."""
    rng = np.random.default_rng(7)
    return {
        "width_mm": np.asarray(width if width is not None else rng.uniform(10.0, 84.0, rows)),
        "tilt_deg": np.asarray(tilt if tilt is not None else rng.uniform(0.0, 90.0, rows)),
        "held": np.asarray(held if held is not None else ([1.0, 0.0] * (rows // 2))[:rows]),
        "object_id": np.asarray(
            groups if groups is not None else [f"obj{i % 4}" for i in range(rows)]),
    }


class AConstantFeatureIsRefusedTests(unittest.TestCase):
    """The check that would have caught aletheia's `quality` and our own `contact_angle_deg`."""

    def test_a_column_of_one_value_blocks(self) -> None:
        verdict = assess_corpus(_corpus(width=np.zeros(40)), SPEC)
        self.assertFalse(verdict.usable)
        self.assertTrue(any("width_mm" in b and "CONSTANT" in b for b in verdict.blocking),
                        verdict.blocking)

    def test_a_missing_feature_blocks_and_says_why(self) -> None:
        """A trainer that maps a missing key to 0.0 reports a converged fit over zeros."""
        corpus = _corpus()
        del corpus["tilt_deg"]
        verdict = assess_corpus(corpus, SPEC)
        self.assertFalse(verdict.usable)
        self.assertTrue(any("tilt_deg" in b and "0.0" in b for b in verdict.blocking),
                        verdict.blocking)

    def test_NaN_blocks_separately_from_constant(self) -> None:
        """A column can be perfectly varied and still unusable; a mean would just come back NaN."""
        width = np.linspace(10.0, 80.0, 40)
        width[3] = np.nan
        verdict = assess_corpus(_corpus(width=width), SPEC)
        self.assertTrue(any("NaN" in b for b in verdict.blocking), verdict.blocking)

    def test_a_constant_column_that_is_NOT_a_feature_is_named_not_blocked(self) -> None:
        """Our `contact_angle_deg` is exactly 0 BY CONSTRUCTION and that is correct.

        The reference enumerates exact antipodal axes, so the worst contact deviation really is zero.
        What must not happen is nobody knowing -- so it appears as a note, and the corpus stays usable.
        """
        corpus = _corpus()
        corpus["contact_angle_deg"] = np.zeros(40)
        verdict = assess_corpus(corpus, SPEC)
        self.assertTrue(verdict.usable, verdict.blocking)
        self.assertTrue(any("contact_angle_deg" in note for note in verdict.notes), verdict.notes)


class TrainServeSkewIsRefusedTests(unittest.TestCase):
    """The more dangerous half: a feature constant in training and alive at inference."""

    def test_constant_in_training_and_live_when_served(self) -> None:
        train = _corpus(tilt=np.zeros(40))
        serve = _corpus(tilt=np.linspace(0.0, 170.0, 40))
        verdict = assess_corpus(train, SPEC, serving=serve)
        self.assertFalse(verdict.usable)
        self.assertTrue(any("skew" in b for b in verdict.blocking), verdict.blocking)

    def test_live_in_training_and_constant_when_served(self) -> None:
        verdict = assess_corpus(_corpus(), SPEC, serving=_corpus(tilt=np.full(40, 3.0)))
        self.assertFalse(verdict.usable)
        self.assertTrue(any("cannot apply" in b for b in verdict.blocking), verdict.blocking)

    def test_a_feature_absent_at_serving_blocks(self) -> None:
        serve = _corpus()
        del serve["tilt_deg"]
        verdict = assess_corpus(_corpus(), SPEC, serving=serve)
        self.assertFalse(verdict.usable)
        self.assertTrue(any("ABSENT" in b for b in verdict.blocking), verdict.blocking)

    def test_serving_outside_the_trained_range_warns_but_does_not_block(self) -> None:
        """Inside the trained range is interpolation; outside it the model is guessing."""
        train = _corpus(width=np.linspace(20.0, 40.0, 40))
        serve = _corpus(width=np.linspace(20.0, 140.0, 40))
        verdict = assess_corpus(train, SPEC, serving=serve)
        self.assertTrue(verdict.usable, verdict.blocking)
        self.assertTrue(any("extrapolation" in w for w in verdict.warnings), verdict.warnings)

    def test_a_FRAME_MISMATCH_is_refused_rather_than_compared(self) -> None:
        """The check no amount of arithmetic can provide.

        Aletheia's vectors are object-frame, Willy's are BASE. The column names line up and the
        quantities do not. This repo has paid for that class of error three times.
        """
        # Two specs that differ ONLY in frame, and neither is derived -- so this test exercises
        # the frame check and nothing else. (`bootstrap_jaw_v1` is now grasp-frame AND derived; using
        # it here would fail on absent columns for a reason that has nothing to do with frames.)
        verdict = assess_corpus(
            _corpus(), BOOTSTRAP_OBJECT_FRAME_V0,
            serving=_corpus(), serving_spec=WILLY_JAW_WITH_CONTACT_ANGLE)
        self.assertFalse(verdict.usable)
        self.assertTrue(any("frame" in b for b in verdict.blocking), verdict.blocking)


class GroupsAndLeakageTests(unittest.TestCase):
    def test_a_missing_group_column_blocks(self) -> None:
        corpus = _corpus()
        del corpus["object_id"]
        verdict = assess_corpus(corpus, SPEC)
        self.assertFalse(verdict.usable)
        self.assertTrue(any("leakage" in b for b in verdict.blocking), verdict.blocking)

    def test_one_group_only_blocks(self) -> None:
        verdict = assess_corpus(_corpus(groups=["only"] * 40), SPEC)
        self.assertFalse(verdict.usable)
        self.assertTrue(any("no grouped split" in b for b in verdict.blocking), verdict.blocking)

    def test_a_group_spanning_two_splits_blocks(self) -> None:
        corpus = _corpus()
        corpus["split"] = np.asarray(["train"] * 20 + ["val"] * 20)
        verdict = assess_corpus(corpus, SPEC)
        self.assertFalse(verdict.usable)
        self.assertTrue(any("more than one split" in b for b in verdict.blocking), verdict.blocking)

    def test_a_single_valued_split_column_warns_that_there_is_no_held_out_set(self) -> None:
        """Aletheia's exact shape: a `split` column whose one value is 'train'."""
        corpus = _corpus()
        corpus["split"] = np.asarray(["train"] * 40)
        verdict = assess_corpus(corpus, SPEC)
        self.assertTrue(verdict.usable, verdict.blocking)
        self.assertTrue(any("NO held-out set" in w for w in verdict.warnings), verdict.warnings)

    def test_the_group_sizes_are_reported_so_a_row_split_is_visibly_wrong(self) -> None:
        verdict = assess_corpus(_corpus(), SPEC)
        self.assertTrue(any("ROW-level split" in note for note in verdict.notes), verdict.notes)


class ClassBalanceTests(unittest.TestCase):
    def test_no_positive_blocks(self) -> None:
        verdict = assess_corpus(_corpus(held=np.zeros(40)), SPEC)
        self.assertFalse(verdict.usable)
        self.assertTrue(any("no positive" in b for b in verdict.blocking), verdict.blocking)

    def test_all_positive_blocks(self) -> None:
        verdict = assess_corpus(_corpus(held=np.ones(40)), SPEC)
        self.assertFalse(verdict.usable)
        self.assertTrue(any("nothing to separate" in b for b in verdict.blocking), verdict.blocking)

    def test_a_heavy_imbalance_warns(self) -> None:
        """The pilot's own shape: 5.59 % positive, and a validation split may hold almost none."""
        held = np.zeros(400)
        held[:2] = 1.0
        verdict = assess_corpus(
            _corpus(400, held=held, groups=[f"obj{i % 40}" for i in range(400)]), SPEC)
        self.assertTrue(any("imbalanced" in w for w in verdict.warnings), verdict.warnings)

    def test_positives_confined_to_one_group_blocks(self) -> None:
        """343 of 463 objects hold a positive in the real pilot; one would make a split impossible."""
        held = np.zeros(40)
        held[np.asarray([0, 4, 8, 12])] = 1.0        # every positive on obj0
        verdict = assess_corpus(_corpus(held=held), SPEC)
        self.assertFalse(verdict.usable)
        self.assertTrue(any("both sides" in b for b in verdict.blocking), verdict.blocking)


class AHealthyCorpusPassesTests(unittest.TestCase):
    def test_it_is_usable_and_reports_its_numbers(self) -> None:
        verdict = assess_corpus(_corpus(), SPEC)
        self.assertTrue(verdict.usable, verdict.blocking)
        self.assertEqual(verdict.rows, 40)
        self.assertEqual(verdict.groups, 4)
        self.assertAlmostEqual(verdict.positive_rate, 0.5)
        self.assertIn("USABLE", format_verdict(verdict))


class TheFeatureSpecIsAContractTests(unittest.TestCase):
    def test_the_label_may_not_also_be_a_feature(self) -> None:
        """The oldest leak there is: a model that scores perfectly and predicts nothing."""
        with self.assertRaises(ValueError) as caught:
            FeatureSpec(name="bad", frame="base", features=("held",), label="held", group="g")
        self.assertIn("held", str(caught.exception))

    def test_an_unknown_spec_name_fails_with_the_list(self) -> None:
        with self.assertRaises(KeyError) as caught:
            spec_named("bootstrap_jaw_v2")
        self.assertIn("bootstrap_jaw_v1", str(caught.exception))

    def test_the_shipped_specs_exclude_the_dead_columns(self) -> None:
        """`quality` and `split` are in the bootstrap file and deliberately not in any spec."""
        for spec in (BOOTSTRAP_JAW_V1, BOOTSTRAP_OBJECT_FRAME_V0, WILLY_JAW_WITH_CONTACT_ANGLE):
            with self.subTest(spec=spec.name):
                self.assertNotIn("quality", spec.features)
                self.assertNotIn("split", spec.features)

    def test_the_shipped_scorer_spec_is_grasp_frame_and_derived(self) -> None:
        """The two facts a consumer must not have to guess: which frame, and computed vs read."""
        self.assertEqual(BOOTSTRAP_JAW_V1.frame, "grasp")
        self.assertTrue(BOOTSTRAP_JAW_V1.derived)
        self.assertEqual(BOOTSTRAP_OBJECT_FRAME_V0.frame, "object")


class ThisPackageAgreesWithTheRLReadinessCheckerTests(unittest.TestCase):
    """The duplication invariant, checked rather than hoped about.

    `rl/readiness.py` computes the same statistics for `GraspAttemptRecord` logs. This package
    re-implements them in numpy instead of importing across the stack -- a datagen package should not
    depend on the RL layer, and a pure-Python loop over 218,529 rows takes minutes rather than
    milliseconds. Two implementations of the same arithmetic are only safe while something checks that
    they agree.
    """

    def test_both_reach_the_same_verdict_and_the_same_numbers(self) -> None:
        from src.robot.grasping.rl.readiness import _column  # noqa: PLC0415

        for name, values in (
            ("constant", [3.0] * 12),
            ("live", [float(v) for v in range(12)]),
            ("with_zeros", [0.0, 0.0, 1.0, 2.5, 2.5, 9.0]),
            ("noise_only", [1.0, 1.0 + 1e-12, 1.0 - 1e-12]),
        ):
            with self.subTest(column=name):
                theirs = _column(name, values)
                ours = column_stats(name, np.asarray(values))
                self.assertEqual(ours.verdict, theirs.verdict)
                self.assertEqual(ours.rows, theirs.rows)
                self.assertEqual(ours.distinct, theirs.distinct)
                self.assertEqual(ours.nonzero, theirs.nonzero)
                self.assertAlmostEqual(ours.stdev, theirs.stdev, places=9)
                self.assertAlmostEqual(ours.minimum, theirs.minimum, places=9)
                self.assertAlmostEqual(ours.maximum, theirs.maximum, places=9)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
