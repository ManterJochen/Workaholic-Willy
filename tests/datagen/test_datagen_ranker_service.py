"""The ranker path, callable from Python instead of only from argparse.

⭐ **WHY THIS PATH AND NOT ANOTHER.** The ranker is the part of the stack a customer is most likely
to want to retrain, because it is the part that learns THEIR objects, and it was the part with the
least library surface: `build-ranker-corpus` composed three typed functions inside an argparse
handler and `train-ranker` had no twin at all. Everything underneath was already typed. What was
missing was the thing that joins them, so "use it from Python" meant reading two handlers and
reproducing them.

⛔ **THE DEFECT THIS FILE EXISTS TO PREVENT, AND THE FIRST DRAFT SHIPPED IT.** The service invented
`source="datagen"` and `spec="ranker_v1"` as library defaults while the CLI declared `grasp_labels`
and `bootstrap_jaw_v1`. Those were not two defaults for one thing, they were two INVALID ones:
`SOURCES` holds three names and neither of mine is among them, and `spec_named` refuses an unknown
name outright. `RankerFit.from_corpus(path).fit()` would have raised on its own defaults while the
identical CLI call worked, which is the worst kind of twin: it exists, it is documented, and it
cannot run.
"""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from unittest import mock

from src.contracts.options import UNSET, chosen
from datagen.config import DatagenConfig
from datagen.corpus.service import (
    DEFAULT_SOURCE,
    DEFAULT_SPEC,
    RankerCorpus,
    RankerCorpusReport,
    RankerFit,
    RankerFitReport,
)


class TheDefaultsAreDeclaredOnceTests(unittest.TestCase):
    """⛔ ONE DECLARATION, TWO DOORS. The parser imports these; it does not restate them."""

    @staticmethod
    def _parsed(*argv: str):
        import sys

        from datagen.__main__ import build_parser

        saved = sys.argv
        try:
            sys.argv = ["datagen"]
            return build_parser().parse_args(["train-ranker", "--corpus", "x.npz", *argv])
        finally:
            sys.argv = saved

    def test_the_cli_and_the_library_agree_on_source_and_spec(self) -> None:
        args = self._parsed()
        self.assertEqual(DEFAULT_SOURCE, args.source)
        self.assertEqual(DEFAULT_SPEC, args.spec)

    def test_both_defaults_are_values_the_callee_actually_accepts(self) -> None:
        """The check the first draft failed. A default nothing accepts is worse than none."""
        from src.robot.grasping.deep.ranker.features import spec_named
        from datagen.corpus.sources import SOURCES

        self.assertIn(DEFAULT_SOURCE, SOURCES)
        spec_named(DEFAULT_SPEC)                       # raises KeyError on an unknown name

    def test_the_four_trainer_knobs_are_not_redeclared_by_argparse(self) -> None:
        """⛔ FOUR COMPETING DEFAULTS THAT ALL AGREED. argparse declared 5 / 400 / 0.05 / 3 and
        `fit_ranker` declares the same four. Agreement today proves nothing about tomorrow: both
        sides are valid values of the right type, so nothing raises when they part.
        """
        args = self._parsed()
        for name in ("folds", "trees", "learning_rate", "tree_depth"):
            self.assertFalse(chosen(getattr(args, name)),
                             f"--{name} declares a default the trainer owns")

    def test_a_flag_the_caller_DOES_pass_still_arrives(self) -> None:
        """The control. Without it the test above passes for a parser that drops the flag."""
        self.assertEqual(3, self._parsed("--folds", "3").folds)

    def test_the_help_text_reads_the_trainers_real_signature(self) -> None:
        """So --help cannot quote a number the trainer stopped using."""
        from src.robot.grasping.deep.ranker.training import fit_ranker

        from datagen.__main__ import _fit_default

        real = inspect.signature(fit_ranker).parameters["n_estimators"].default
        self.assertIn(str(real), _fit_default("n_estimators"))


class TheCorpusFindsItsDatasetTheSameWayEverySiblingDoesTests(unittest.TestCase):
    """⛔ THE RESOLUTION THE CLI GOT WRONG. `_cmd_build_ranker_corpus` read a hard-coded
    `logs/p5/datasets` whatever the config said, so `--config x.json --name v3` built the dataset in
    one place and looked for it in another. It failed loudly only because nothing of that name
    existed there; with a same-named dataset in the old directory it would have built a corpus from
    the WRONG scenes and stamped it with the right name.
    """

    def test_the_config_output_root_decides(self) -> None:
        config = DatagenConfig(output={"root": "D:/somewhere/else"})
        corpus = RankerCorpus.from_config(config, name="v3")
        self.assertEqual(Path("D:/somewhere/else/v3"), corpus.dataset)

    def test_an_explicit_out_root_overrides_it(self) -> None:
        corpus = RankerCorpus.from_config(DatagenConfig(), name="v3", out_root="D:/override")
        self.assertEqual(Path("D:/override/v3"), corpus.dataset)

    def test_the_mask_source_reaches_the_walk(self) -> None:
        """⚠ NOT COSMETIC. `gt` reads the simulator's instance masks and measures the ranker against
        perfect segmentation, which no camera delivers."""
        from datagen.grasps.masks import PRED_SUFFIX

        seen: dict[str, object] = {}

        def capture(root, **kwargs):                  # type: ignore[no-untyped-def]
            seen.update(kwargs)
            return {"valid": _zeros(), "object_key": _zeros()}

        with mock.patch("datagen.corpus.build.build_ranker_corpus", side_effect=capture), \
                mock.patch("datagen.corpus.build.write_corpus", return_value=Path("c.npz")):
            RankerCorpus.from_dataset("d", mask_source="pred").build()
        self.assertEqual(PRED_SUFFIX, seen["mask_suffix"])


def _zeros():
    import numpy as np

    return np.zeros(0, dtype=np.int32)


class ThePhysicsJoinIsReportedNotSwallowedTests(unittest.TestCase):

    @staticmethod
    def _build(physics_report):
        def joined(table, path):                      # type: ignore[no-untyped-def]
            return table, physics_report

        with mock.patch("datagen.corpus.build.build_ranker_corpus",
                        return_value={"valid": _zeros(), "object_key": _zeros()}), \
                mock.patch("datagen.corpus.build.attach_physics_labels", side_effect=joined), \
                mock.patch("datagen.corpus.build.write_corpus", return_value=Path("c.npz")):
            return RankerCorpus.from_dataset("d", physics_path="p.jsonl").build()

    def test_a_join_that_resolved_nothing_refuses_rather_than_writing(self) -> None:
        """⛔ AN EMPTY JOIN IS NOT AN EMPTY RESULT. Writing a corpus whose physics column resolved
        zero rows produces a file that trains on this repository's own referee while claiming to
        carry Isaac's verdict."""
        with self.assertRaises(ValueError) as caught:
            self._build({"rows_out": 0, "rows_in": 900})
        self.assertIn("nothing to train on", str(caught.exception))

    def test_the_strata_survive_into_the_report(self) -> None:
        """⚠ PER SOURCE, NEVER POOLED. The draw is stratified, so a pooled rate describes the
        sampler rather than the data."""
        report = self._build({
            "rows_out": 700, "rows_in": 900, "physics_verdicts": 800,
            "by_source": {"gso": {"trials": 400, "held": 100, "refused": 3, "unmatched": 1}},
            "conflicting_keys": [], "warning": "shake is not deterministic"})
        assert report.physics is not None
        self.assertEqual(400, report.physics["by_source"]["gso"]["trials"])
        self.assertIn("gso", report.render())

    def test_without_physics_the_report_says_so_by_being_None(self) -> None:
        with mock.patch("datagen.corpus.build.build_ranker_corpus",
                        return_value={"valid": _zeros(), "object_key": _zeros()}), \
                mock.patch("datagen.corpus.build.write_corpus", return_value=Path("c.npz")):
            report = RankerCorpus.from_dataset("d").build()
        self.assertIsNone(report.physics, "None is 'not joined'; an empty dict would be 'joined and "
                                          "found nothing', and those are different facts")


class TheVerdictIsAComparisonTests(unittest.TestCase):
    """⛔ AN ABSOLUTE IS NOT A VERDICT. AUROC 0.86 sounds like a result until `width_mm` alone scores
    0.75 on the same folds."""

    @staticmethod
    def _report(auroc: float, base_auroc: float, p: float, base_p: float) -> RankerFitReport:
        return RankerFitReport(
            spec="s", artifact_path=Path("a.json"), card_path=Path("a.card.json"), sha256="0" * 8,
            rows=1, groups=1, positives=1, auroc=auroc, baseline_auroc=base_auroc,
            precision_at_250=p, baseline_precision_at_250=base_p)

    def test_winning_on_both_is_a_win(self) -> None:
        self.assertTrue(self._report(0.86, 0.75, 0.40, 0.30).beats_baseline)

    def test_winning_on_only_one_is_not(self) -> None:
        self.assertFalse(self._report(0.86, 0.75, 0.20, 0.30).beats_baseline)
        self.assertFalse(self._report(0.70, 0.75, 0.40, 0.30).beats_baseline)

    def test_a_tie_is_not_a_win(self) -> None:
        self.assertFalse(self._report(0.75, 0.75, 0.40, 0.30).beats_baseline)

    def test_the_refusal_is_visible_in_the_rendered_text(self) -> None:
        self.assertIn("REFUSED", self._report(0.70, 0.75, 0.20, 0.30).render())
        self.assertNotIn("REFUSED", self._report(0.86, 0.75, 0.40, 0.30).render())

    def test_the_rendered_text_is_ascii(self) -> None:
        """⚠ A Windows console under cp1252 cannot print anything else."""
        self._report(0.7, 0.75, 0.2, 0.3).render().encode("ascii")
        RankerCorpusReport(path=Path("c.npz"), rows=1, objects=1, valid=1).render().encode("ascii")


class TheFitForwardsOnlyWhatWasChosenTests(unittest.TestCase):

    def test_an_untouched_knob_never_reaches_the_trainer(self) -> None:
        seen: dict[str, object] = {}

        def capture(table, spec, **kwargs):            # type: ignore[no-untyped-def]
            seen.update(kwargs)
            raise _Stop

        with mock.patch("src.robot.grasping.deep.ranker.training.fit_ranker",
                        side_effect=capture), \
                mock.patch("datagen.corpus.sources.load_corpus", return_value={}), \
                self.assertRaises(_Stop):
            RankerFit.from_corpus("c.npz").fit()
        self.assertEqual({}, seen, "the wrapper forwarded a default the trainer owns")

    def test_a_chosen_knob_does_reach_it_under_the_TRAINERS_name(self) -> None:
        """⚠ THE CLI CALLS IT `--trees`; `fit_ranker` calls it `n_estimators`. A wrapper that
        forwarded the CLI's word would be silently ignored as an unexpected keyword, or worse,
        swallowed by a `**kwargs`."""
        seen: dict[str, object] = {}

        def capture(table, spec, **kwargs):            # type: ignore[no-untyped-def]
            seen.update(kwargs)
            raise _Stop

        with mock.patch("src.robot.grasping.deep.ranker.training.fit_ranker",
                        side_effect=capture), \
                mock.patch("datagen.corpus.sources.load_corpus", return_value={}), \
                self.assertRaises(_Stop):
            RankerFit.from_corpus("c.npz", trees=17, tree_depth=2).fit()
        self.assertEqual({"n_estimators": 17, "max_depth": 2}, seen)

    def test_describe_names_only_the_overrides(self) -> None:
        """`describe()` runs before sklearn is imported, so a wrong knob is visible in the first
        second rather than after the fit."""
        self.assertIn("none, the trainer decides", RankerFit.from_corpus("c.npz").describe())
        self.assertIn("trees", RankerFit.from_corpus("c.npz", trees=17).describe())

    def test_from_report_does_not_restate_the_path(self) -> None:
        report = RankerCorpusReport(path=Path("logs/x/ranker.npz"), rows=1, objects=1, valid=1)
        self.assertEqual(Path("logs/x/ranker.npz"), RankerFit.from_report(report).corpus)

    def test_UNSET_is_the_declared_absence(self) -> None:
        self.assertIs(UNSET, RankerFit.from_corpus("c.npz").folds)


class _Stop(Exception):
    """Stop the call once the arguments have been captured; the fit itself is not under test."""


if __name__ == "__main__":
    unittest.main()
