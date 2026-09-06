"""The cost calculator, held against the run it was measured from.

⭑ THE POINT OF THIS FILE is the first test: the calculator predicts 19.1 h for the v1 corpus and
that corpus took 18.7 h. A cost model nobody checks against a real run is a plausible-looking table,
and the failure mode is silent -- an operator plans an overnight build, and finds out at hour thirty
that the number was decorative. That check is cheap and it is the only one that means anything.

The rest hold the three traps the module exists for: per-shard time is not throughput, requested
scenes are not usable scenes, and rendering is not the whole bill.
"""

from __future__ import annotations

import unittest

from datagen.cost import ENGINE_COSTS, estimate, format_estimate


class ItPredictsTheRunItWasMeasuredFromTests(unittest.TestCase):
    def test_it_reproduces_the_v1_corpus_within_five_percent(self) -> None:
        """1,981 usable scenes, 3 concurrent shards, MEASURED at 18.73 h wall clock."""
        result = estimate(1981, engine="isaac", jobs=3, label=False, corpus=False)
        self.assertAlmostEqual(result.hours, 18.73, delta=18.73 * 0.05)

    def test_it_reproduces_the_v1_corpus_size(self) -> None:
        """913 MB written for 1,981 scenes."""
        result = estimate(1981, engine="isaac", jobs=3, label=False, corpus=False)
        self.assertAlmostEqual(result.gigabytes, 0.913, delta=0.05)

    def test_it_reproduces_the_v2_training_run(self) -> None:
        """150 epochs over 1,584 training scenes, MEASURED at 292.4 s/epoch = 12.2 h.

        ⚠ `folds=1, refit=False` IS THE RUN, not a convenience: v2 was a single pass. The trainer's
        DEFAULT is two (`run_folds: 1` plus `refit: True`), and this calculator used to price one --
        so the number that reproduced the measured run was also the number that understated every
        default budget by half. Both facts live here now, one per test."""
        data = estimate(1980, engine="isaac", jobs=3, label=False, corpus=False)
        full = estimate(1980, engine="isaac", jobs=3, label=False, corpus=False, epochs=150,
                        folds=1, refit=False)
        training_hours = full.hours - data.hours
        self.assertAlmostEqual(training_hours, 12.2, delta=12.2 * 0.10)


class TheThreeTrapsTests(unittest.TestCase):
    def test_parallel_is_not_serial_divided_by_jobs(self) -> None:
        """⚠ TRAP 1. Three Isaac shards do not run three times as fast; quoting per-shard time for a
        parallel run over-estimates by ~3x, and quoting the aggregate for a single process
        under-estimates by the same. Both must be reachable, and they must differ."""
        serial = estimate(1000, engine="isaac", jobs=1, label=False, corpus=False)
        parallel = estimate(1000, engine="isaac", jobs=3, label=False, corpus=False)
        self.assertGreater(serial.hours, parallel.hours * 2.5)
        # ...and NOT exactly 3x, because the measured run gave up some of that.
        self.assertLess(serial.hours, parallel.hours * 3.0)

    def test_asking_for_n_usable_scenes_requests_more_than_n(self) -> None:
        """⚠ TRAP 2. `none` refuses `pile` by name, so a quarter of every request evaporates. The
        answer is in USABLE scenes and the request is computed backwards from it."""
        result = estimate(2000, engine="none")
        self.assertGreater(result.requested_scenes, 2600)
        self.assertEqual(result.usable_scenes, 2000)
        self.assertIn("75%", format_estimate(result))

    def test_a_lossless_engine_asks_for_exactly_what_it_needs(self) -> None:
        self.assertEqual(estimate(2000, engine="mujoco").requested_scenes, 2000)

    def test_the_cheap_engines_are_dominated_by_the_stages_after_the_render(self) -> None:
        """⚠ TRAP 3. On `none` a scene renders in 1.4 s and costs 0.8 s more to label and extract.
        An estimate covering only the render is wrong by a large fraction on exactly the
        configuration a customer without a GPU will pick."""
        result = estimate(2000, engine="none")
        render = next(s for s in result.stages if s.name == "render")
        after = result.hours - render.hours
        self.assertGreater(after / result.hours, 0.25)

    def test_isaac_is_dominated_by_the_render_instead(self) -> None:
        """The same comparison the other way, so the assertion above cannot pass by accident."""
        result = estimate(2000, engine="isaac", jobs=3)
        render = next(s for s in result.stages if s.name == "render")
        self.assertGreater(render.hours / result.hours, 0.95)


class ItSaysHowFarItIsReachingTests(unittest.TestCase):
    def test_jobs_past_the_measured_point_is_flagged_as_extrapolation(self) -> None:
        result = estimate(1000, engine="isaac", jobs=8)
        self.assertTrue(any("EXTRAPOLATION" in w for w in result.warnings), result.warnings)

    def test_an_engine_with_no_parallel_measurement_refuses_to_pretend(self) -> None:
        """`none`/`mujoco` were timed single-process. Reporting a speed-up nobody measured would be
        the calculator inventing its own evidence."""
        serial = estimate(1000, engine="none", label=False, corpus=False)
        parallel = estimate(1000, engine="none", jobs=8, label=False, corpus=False)
        self.assertAlmostEqual(serial.hours, parallel.hours, places=6)
        self.assertTrue(any("NO measured parallel" in w for w in parallel.warnings))

    def test_the_training_assumption_is_declared(self) -> None:
        result = estimate(1000, epochs=10)
        self.assertTrue(any("assumption, not a measurement" in w for w in result.warnings))

    def test_every_engine_carries_its_evidence_into_the_report(self) -> None:
        for engine, cost in ENGINE_COSTS.items():
            with self.subTest(engine=engine):
                self.assertIn(cost.evidence, format_estimate(estimate(100, engine=engine)))

    def test_every_engine_declares_what_it_cannot_do(self) -> None:
        """A cheaper engine that does not say what it gave up is a trap, not an option."""
        for engine, cost in ENGINE_COSTS.items():
            with self.subTest(engine=engine):
                self.assertTrue(cost.caveat.strip(), engine)


class ItRefusesNonsenseTests(unittest.TestCase):
    def test_zero_scenes(self) -> None:
        with self.assertRaises(ValueError):
            estimate(0)

    def test_an_unknown_engine_lists_what_there_was(self) -> None:
        with self.assertRaises(ValueError) as caught:
            estimate(100, engine="pybullet")
        self.assertIn("isaac", str(caught.exception))

    def test_zero_jobs(self) -> None:
        with self.assertRaises(ValueError):
            estimate(100, jobs=0)


class TheCliTests(unittest.TestCase):
    def test_it_answers_and_exits_zero(self) -> None:
        from datagen.__main__ import main
        self.assertEqual(main(["cost", "--scenes", "500", "--engine", "mujoco"]), 0)

    def test_a_bad_request_exits_two_rather_than_answering(self) -> None:
        from datagen.__main__ import main
        self.assertEqual(main(["cost", "--scenes", "0"]), 2)





class ItPricesTheCONFIGSCorpusTests(unittest.TestCase):
    """⛔ THE WORST POSSIBLE DEFECT IN THIS PARTICULAR TOOL, and it shipped this morning.

    `cost` exists to price a corpus before somebody spends the night on it. It read a hard-coded 2000
    and ignored the config's own `scenes`, so a config asking for 8,000 produced a confident
    "2000 usable scene(s) ... TOTAL 57.24 h" with nothing anywhere saying the number was for a
    different corpus. The engine was read from the config two lines below, so the inconsistency lived
    inside one function.
    """

    def _config_file(self, scenes: int) -> str:
        import json
        import tempfile
        from pathlib import Path

        path = Path(tempfile.mkdtemp()) / "cfg.json"
        path.write_text(json.dumps({"scenes": scenes}), encoding="utf-8")
        return str(path)

    def test_the_config_scene_count_is_priced(self) -> None:
        import contextlib
        import io

        from datagen.__main__ import main

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["cost", "--config", self._config_file(8000)]), 0)
        self.assertIn("8000 usable scene(s)", out.getvalue())

    def test_an_explicit_scenes_flag_still_wins(self) -> None:
        """A what-if is the other half of what this command is for."""
        import contextlib
        import io

        from datagen.__main__ import main

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(
                main(["cost", "--config", self._config_file(8000), "--scenes", "150"]), 0)
        self.assertIn("150 usable scene(s)", out.getvalue())


class ItPricesEVERYTrainingPassTests(unittest.TestCase):
    """⛔ THE DEFAULT IS TWO PASSES AND THIS MODULE PRICED ONE -- its worst defect, because `cost` is
    the instrument a user is told to run BEFORE committing a night. `TrainingPlan` defaults are
    `run_folds: 1` AND `refit: True`, so the documented command trains a fold and then refits on
    everything. A user budgeting 12 h ran 27, with nothing in either tool's output to catch it."""

    def _train_hours(self, **kwargs: object) -> float:
        base = estimate(1980, engine="isaac", jobs=3, label=False, corpus=False)
        full = estimate(1980, engine="isaac", jobs=3, label=False, corpus=False,
                        epochs=150, **kwargs)
        return full.hours - base.hours

    def test_the_default_prices_the_refit_pass_too(self) -> None:
        from datagen.cost import _DEFAULT_TRAIN_PASSES

        one = self._train_hours(folds=1, refit=False)
        default = self._train_hours()
        self.assertAlmostEqual(default / one, float(_DEFAULT_TRAIN_PASSES), places=6)

    def test_five_folds_costs_six_passes(self) -> None:
        one = self._train_hours(folds=1, refit=False)
        five = self._train_hours(folds=5)
        self.assertAlmostEqual(five / one, 6.0, places=6)

    def test_it_matches_the_MEASURED_single_pass(self) -> None:
        """The v2 run was ONE pass of 150 epochs and took 12.2 h. `--no-refit --train-folds 1` must
        still reproduce it, or the correction broke the thing that was right."""
        self.assertAlmostEqual(self._train_hours(folds=1, refit=False), 12.2, delta=12.2 * 0.10)

    def test_the_report_says_how_many_passes(self) -> None:
        text = format_estimate(estimate(1980, engine="isaac", jobs=3, epochs=150))
        self.assertIn("2 pass(es)", text)
        self.assertIn("refit", text)

    def test_multiple_passes_are_WARNED_about(self) -> None:
        """A number that silently doubled is worse than one that says why it doubled."""
        result = estimate(1980, engine="isaac", jobs=3, epochs=150)
        self.assertTrue(any("full passes" in w for w in result.warnings), result.warnings)

    def test_a_single_pass_does_not_warn(self) -> None:
        result = estimate(1980, engine="isaac", jobs=3, epochs=150, folds=1, refit=False)
        self.assertFalse(any("full passes" in w for w in result.warnings))


if __name__ == "__main__":                                           # pragma: no cover
    unittest.main()
