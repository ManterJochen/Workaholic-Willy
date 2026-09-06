"""The learned generator gets a rung on the ladder — built the way a CELL builds it.

⭑ THE DESIGN POINT, and it is worth more than the two lines it costs: **an evaluation that constructs
its subject differently from the cell measures a different system.** The rung therefore goes through
`build_calculator`, the same factory `cells.py` calls, so what the ladder grades is what a cell would
run — the fail-closed refusal, the jaw limits, the `deep_generator` block's device and score threshold,
all of it.

⚠ WHY IT IS NOT IN `CONFIGURATIONS`. Every rung in that tuple runs on every ladder invocation, and
this one needs a machine-local artifact. Putting it there would make the ordinary ladder refuse on a
box with no trained model. It is selected by name instead — which is also the shape the question wants:
the full ladder is hours and the comparison is a PAIR.

⚠ AND THE PAIRING IS `sfe_fused`, NOT `default`. Both rungs get the target's fused three-view cloud and
the neighbours' masks, and the deep decoder genuinely reads them (`geometry_points_base_mm`,
`scene_points_mm`, `other_object_masks`). A rung that handed a generator inputs it ignored would be a
worse comparison than no rung at all, because it would look fair.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from datagen.eval.ladder import (
    CONFIGURATIONS,
    DEEP_CONFIGURATIONS,
    ENV_DEEP_ARTIFACT,
    select_configurations,
)

_ROOT = Path(__file__).resolve().parents[1]


class TheDefaultLadderIsUnchangedTests(unittest.TestCase):
    """The byte-identity half. A new rung must not alter a single existing ladder run."""

    def test_the_deep_rungs_are_NOT_in_the_default_tuple(self) -> None:
        names = {config.name for config in CONFIGURATIONS}
        for config in DEEP_CONFIGURATIONS:
            self.assertNotIn(config.name, names, config.name)

    def test_no_selection_returns_the_default_tuple_ITSELF(self) -> None:
        self.assertIs(select_configurations(None), CONFIGURATIONS)

    def test_the_default_tuple_still_holds_the_rungs_the_arc_quotes(self) -> None:
        """A rename here would silently orphan every number in the plan."""
        names = {config.name for config in CONFIGURATIONS}
        for expected in ("default", "sfe", "sfe_fused", "floor_topdown"):
            self.assertIn(expected, names, expected)


class TheSelectorRefusesRatherThanGuessesTests(unittest.TestCase):
    def test_an_unknown_rung_refuses_and_lists_what_there_was(self) -> None:
        """⚠ The alternative -- running the rungs it recognised -- is the failure `--summary-only`
        already taught this file: a run that returns fast with rungs that no longer exist reads
        exactly like a run that worked."""
        with self.assertRaises(ValueError) as caught:
            select_configurations("sfe_fused,typo")
        message = str(caught.exception)
        self.assertIn("typo", message)
        self.assertIn("sfe_fused", message, "the refusal must list the available names")

    def test_an_empty_selection_refuses(self) -> None:
        with self.assertRaises(ValueError):
            select_configurations(" , ")

    def test_a_pair_comes_back_in_the_order_asked_for(self) -> None:
        picked = select_configurations("sfe_fused,deep")
        self.assertEqual([c.name for c in picked], ["sfe_fused", "deep"])

    def test_whitespace_is_tolerated(self) -> None:
        self.assertEqual([c.name for c in select_configurations(" deep , sfe_fused ")],
                         ["deep", "sfe_fused"])


class TheDeepRungIsPairedHonestlyTests(unittest.TestCase):
    """It must receive the same observation as the rung it is compared against."""

    @staticmethod
    def _by_name(name: str):
        for config in (*CONFIGURATIONS, *DEEP_CONFIGURATIONS):
            if config.name == name:
                return config
        raise AssertionError(name)

    def test_deep_matches_sfe_fused_on_every_observation_flag(self) -> None:
        deep, analytic = self._by_name("deep"), self._by_name("sfe_fused")
        for flag in ("neighbours", "fused_geometry", "fused_neighbours", "dense_sampling"):
            self.assertEqual(getattr(deep, flag), getattr(analytic, flag), flag)

    def test_deep_scene_matches_sfe_fused_scene(self) -> None:
        deep, analytic = self._by_name("deep_scene"), self._by_name("sfe_fused_scene")
        for flag in ("neighbours", "fused_geometry", "fused_neighbours"):
            self.assertEqual(getattr(deep, flag), getattr(analytic, flag), flag)

    def test_the_deep_calculator_actually_READS_those_inputs(self) -> None:
        """⛔ THE TRAP THIS AVOIDS. A rung that passes a fused cloud to a generator that ignores it
        looks like a fair comparison and is not. These four keys are read in `_propose`."""
        source = (_ROOT / "src/robot/grasping/deep/calculator.py").read_text(encoding="utf-8")
        for key in ("geometry_points_base_mm", "scene_points_mm", "other_object_masks",
                    "camera_to_base"):
            self.assertIn(f'kwargs.get("{key}")', source, key)

    def test_the_rung_hands_over_the_JAW(self) -> None:
        """The decoder refuses a span this gripper cannot open to, so withholding the limits would
        grade a generator proposing widths no cell would execute."""
        import numpy as np

        built = self._by_name("deep").build(np.eye(3), np.eye(4))
        self.assertEqual(built["min_grip_width_mm"], 5.0)
        self.assertEqual(built["max_grip_width_mm"], 85.0)


class TheRungGoesThroughTheProductionFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        from datagen.eval import ladder as evaluate

        evaluate._DEEP_CACHE.clear()                            # noqa: SLF001 - per-process cache

    def test_it_refuses_LOUDLY_when_no_artifact_is_named(self) -> None:
        """⚠ A rung that quietly skipped itself would report a ladder with no deep row, which looks
        exactly like a ladder that ran."""
        from datagen.eval.ladder import _make_deep                # noqa: PLC0415

        with mock.patch.dict(os.environ, {ENV_DEEP_ARTIFACT: ""}, clear=False), \
             self.assertRaises(FileNotFoundError) as caught:
            _make_deep(camera_matrix=None)
        self.assertIn(ENV_DEEP_ARTIFACT, str(caught.exception))
        self.assertIn(".ckpt.pt", str(caught.exception),
                      "the refusal must warn that a CHECKPOINT is not an artifact")

    def test_it_calls_build_calculator_and_not_the_class(self) -> None:
        """The whole point: the ladder builds what a CELL builds."""
        source = (_ROOT / "datagen/eval/ladder.py").read_text(encoding="utf-8")
        window = source[source.index("def _make_deep"):source.index("def _make_deep") + 2200]
        self.assertIn("build_calculator", window)
        self.assertNotIn("DeepGraspCalculator(", window,
                         "_make_deep constructs the class directly, bypassing the factory")

    def test_one_calculator_per_process_not_one_per_view(self) -> None:
        """⭑ FIDELITY BEFORE SPEED. A cell loads its artifact once and picks many times; the ladder
        constructs per VIEW. Reloading per view would measure a warm-up this stack never pays."""
        from datagen.eval import ladder as evaluate                           # noqa: PLC0415

        sentinel = object()
        # ⛔ THE KEY IS THE ARTIFACT PATH AND NOTHING ELSE, since 2026-09-04. It used to carry one
        # entry per substitutable head, because two ablation rungs sharing an artifact needed
        # different calculators. Those rungs went with the binned decoder they substituted heads of,
        # so a compound key would now be a tuple with one moving part and would say nothing.
        evaluate._DEEP_CACHE["/some/artifact.pt"] = sentinel        # noqa: SLF001
        with mock.patch.dict(os.environ, {ENV_DEEP_ARTIFACT: "/some/artifact.pt"}, clear=False):
            self.assertIs(evaluate._make_deep(camera_matrix=None), sentinel)   # noqa: SLF001

    def test_a_DIFFERENT_artifact_is_not_served_from_the_cache(self) -> None:
        from datagen.eval import ladder as evaluate                           # noqa: PLC0415

        evaluate._DEEP_CACHE["/one.pt"] = object()                    # noqa: SLF001
        with mock.patch.dict(os.environ, {ENV_DEEP_ARTIFACT: "/two.pt"}, clear=False), \
             self.assertRaises(FileNotFoundError):
            # /two.pt does not exist, so the FACTORY refuses -- which proves the cache was bypassed.
            evaluate._make_deep(camera_matrix=None)                   # noqa: SLF001


class APartialLadderNeverTouchesTheReferenceTests(unittest.TestCase):
    """⛔ A REPAIR, NOT A PRECAUTION -- this happened. `grasp_eval.jsonl` is APPENDED across runs, so a
    two-scene exploratory run with `--rungs sfe_fused,deep` wrote 964 rows from a throwaway 2-epoch
    model into the file the arc quotes, plus a torn line when it was interrupted. It also raced the
    test suite, which reads the same dataset, and produced four failures that looked like a regression
    in the change under test. A selective run is not the reference and must not share its file.

    ⭐ THE RULE MOVED TO `datagen/eval/service.py::evaluation_output_name` ON 2026-09-04, and these
    two tests moved with it from GREPPING THE SOURCE of `__main__.py` to calling the function. A
    source grep passes while the handler writes somewhere else entirely: it asserted that a line
    exists, not that a run obeys it. The third test below is the half neither version had, the one
    that checks the computed name actually reaches `evaluate_dataset`.
    """

    def test_a_rung_selection_gets_its_own_output_file(self) -> None:
        from datagen.eval.service import REFERENCE_OUTPUT_NAME, evaluation_output_name

        name = evaluation_output_name(selective=True)
        self.assertNotEqual(REFERENCE_OUTPUT_NAME, name,
                            "a --rungs run still writes into the full ladder's reference file")
        self.assertIn("_rungs", name)

    def test_the_default_path_is_unchanged(self) -> None:
        """Byte-identity: an ordinary ladder run must still write `grasp_eval.jsonl`."""
        from datagen.eval.service import evaluation_output_name

        self.assertEqual("grasp_eval.jsonl", evaluation_output_name())

    def test_the_rule_is_WIRED_and_not_merely_correct(self) -> None:
        """⚠ THE HALF A PURE-FUNCTION TEST CANNOT SEE. A naming rule that is right and not reached is
        the inert-switch shape this repository fences everywhere else, so this asserts the name that
        actually arrives at `evaluate_dataset`."""
        from unittest import mock

        from datagen.eval.service import GraspEvaluation

        seen: dict = {}

        def capture(root, **kwargs):                   # type: ignore[no-untyped-def]
            seen.update(kwargs)
            return {"rows": 1, "by_config": {}}

        with mock.patch("datagen.eval.ladder.evaluate_dataset", side_effect=capture),                 mock.patch("datagen.eval.ladder.select_configurations", return_value=()):
            GraspEvaluation.from_dataset("d").evaluate(rungs="deep")
        self.assertEqual("grasp_eval_rungs.jsonl", seen["out_name"])


class TheCLIExposesItTests(unittest.TestCase):
    def test_the_flag_exists_and_its_help_is_ascii(self) -> None:
        """This console is cp1252 -- a non-ASCII help string crashes `--help` and nothing else."""
        source = (_ROOT / "datagen/__main__.py").read_text(encoding="utf-8")
        self.assertIn('"--rungs"', source)
        start = source.index('"--rungs"')
        window = source[start:start + 700]
        self.assertTrue(window.isascii(), "the --rungs help text is not ASCII")
        self.assertIn("WILLY_DEEP_ARTIFACT", window)


if __name__ == "__main__":
    unittest.main()


class EveryDeepRungBuildsTheDeepCalculatorTest(unittest.TestCase):
    """⛔ A DEEP RUNG WITHOUT `make=` IS BUILT AS THE ANALYTIC CALCULATOR.

    `_evaluate_view` reads `config.make or GraspCalculator`, so the omission is not a missing-argument
    error at declaration time; it silently swaps the thing under test. It surfaces later and elsewhere,
    as `TypeError: GraspCalculator.__init__() got an unexpected keyword argument 'width_source'`, which
    reads like a bug in the head substitution rather than a rung that was never deep. That is exactly
    how `deep_learned_pose` failed on its first run.
    """

    def test_no_deep_rung_falls_back_to_the_analytic_calculator(self) -> None:
        from datagen.eval.ladder import DEEP_CONFIGURATIONS

        missing = [c.name for c in DEEP_CONFIGURATIONS
                   if c.name.startswith("deep") and c.make is None]
        self.assertEqual(missing, [], f"these deep rungs would be built as GraspCalculator: {missing}")

