"""The shipped artifact was one fold's net, so it had never seen part of the customer's own corpus.

⛔ MEASURED on this project's full-corpus arm: `--folds 5 --run-folds 1` trains on 76,750 of 90,078
units, so the weights a cell would serve never saw **14.8 %** of the corpus. On a customer's corpus
that is 14.8 % of their own catalogue, chosen by a hash rather than by anything they decided, and
there was no flag to change it. `--folds 1` is not the escape hatch it looks like: `grouped_folds`
puts every group in fold 0, which leaves the training split empty and raises.

The binned trainer this loop replaced states the policy in its own docstring, "the out-of-fold pass
earns the numbers and the shipped model is refit on everything", and ships `refit=True` with a
`--no-refit` opt-out. The set loop did not inherit it.

⚠ UNMEASURED ON THIS ARCHITECTURE, and that is stated rather than dressed up. Nobody has taken a
number for refitting here. The argument for it is a PRODUCT argument: shipping a model blind to a
seventh of a customer's parts is a defect whatever the aggregate metric says. The argument against is
a second training run. So it is opt-in, and the customer recipe turns it on.

⚠ THE REFIT REPORTS NO HELD-OUT NUMBER, and cannot. It trained on every unit, so any held-out score
would be a memorisation score wearing the wrong name.
"""

from __future__ import annotations

import unittest
from dataclasses import fields

from src.robot.grasping.deep.train.trainer import SetTrainingPlan


class TheDefaultIsUnchangedTests(unittest.TestCase):

    def test_refit_is_OFF_by_default(self) -> None:
        """⚠ Every arm already measured has to reproduce byte-identically, and a silent doubling of
        training time is not something to spring on anyone."""
        self.assertFalse(SetTrainingPlan().refit)

    def test_the_plan_carries_it_so_a_card_can_quote_it(self) -> None:
        """A run whose artifact came from a refit and one whose artifact came from fold 0 are
        different runs, and the plan is what a model card quotes."""
        self.assertIn("refit", {f.name for f in fields(SetTrainingPlan)})


class TheFoldPassStillOwnsTheNumbersTests(unittest.TestCase):

    def test_the_fold_artifact_write_is_SKIPPED_when_refitting(self) -> None:
        """Otherwise the refit's artifact would be overwritten by, or would overwrite, the fold's,
        and which one a cell ended up serving would depend on ordering."""
        import ast
        from pathlib import Path

        source = Path("src/robot/grasping/deep/train/trainer.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        guards = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.If) and "not plan.refit" in ast.unparse(node.test)
        ]
        self.assertTrue(guards, "the fold pass writes its artifact regardless of refit")

    def test_the_refit_reports_no_held_out_score(self) -> None:
        """⚠ THE HONESTY THAT MAKES IT USABLE. A net fitted to every unit has no held-out set left,
        so the key exists, is None, and says why in the payload rather than in a comment nobody
        reads."""
        from pathlib import Path

        source = Path("src/robot/grasping/deep/train/trainer.py").read_text(encoding="utf-8")
        self.assertIn('"held_out": None', source)
        self.assertIn("why_no_held_out", source)

    def test_the_refit_starts_from_a_FRESH_net(self) -> None:
        """Continuing fold 0's weights would train them on the very data they were held out from,
        which is a warm start on a different split rather than a clean fit on the whole corpus."""
        from pathlib import Path

        source = Path("src/robot/grasping/deep/train/trainer.py").read_text(encoding="utf-8")
        body = source[source.index("def _refit_on_everything("):source.index("def plan_with_slots(")]
        self.assertIn("SetGenerator(plan.model)", body,
                      "the refit does not build its own net")


class ItReallyTrainsOnEverythingTests(unittest.TestCase):
    """The claim in one assertion, read off the code rather than off a six-hour run."""

    def _body(self) -> str:
        from pathlib import Path

        source = Path("src/robot/grasping/deep/train/trainer.py").read_text(encoding="utf-8")
        return source[source.index("def _refit_on_everything("):source.index("def plan_with_slots(")]

    def test_the_order_comes_from_every_unit_not_a_fold(self) -> None:
        body = self._body()
        self.assertIn("np.arange(len(index.units))", body,
                      "the refit draws from a split rather than from the whole corpus")
        self.assertNotIn("train_index", body, "the refit reaches into a fold's split")

    def test_it_uses_a_DIFFERENT_seed_stream_from_any_fold(self) -> None:
        """Reusing fold 0's stream would make the refit a replay of fold 0's unit order over a
        larger pool, which is a confound for free."""
        body = self._body()
        self.assertIn("plan.seed + 500", body)


class ItActuallyRunsTests(unittest.TestCase):
    """⭐ The structural tests above prove the code says the right thing. This one runs it."""

    def _run(self, *, refit: bool) -> dict:
        import tempfile
        from pathlib import Path

        from src.robot.grasping.deep.train.trainer import train_set_generator
        from tests.test_deep_set_loop import _corpus, _small_plan

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            files = _corpus(root, scenes=8)
            return train_set_generator(files, _small_plan(refit=refit), out_dir=root / "out",
                                       device="cpu", probe_units=0)

    def test_a_refit_run_produces_a_refit_block_and_an_artifact(self) -> None:
        report = self._run(refit=True)

        self.assertIn("refit", report)
        self.assertTrue(report["refit"]["epochs"], "the refit ran no epoch")
        self.assertTrue(report["refit"]["artifact"]["written"],
                        f"no artifact came out of the refit: {report['refit']['artifact']}")

    def test_the_refit_sees_MORE_units_than_the_fold_did(self) -> None:
        """⭐ THE CLAIM ITSELF, measured on a real run rather than read off a comment."""
        report = self._run(refit=True)

        self.assertEqual(int(report["refit"]["units"]), 8,
                         "the refit did not draw from the whole corpus")

    def test_a_plain_run_still_has_NO_refit_block(self) -> None:
        """The byte-identical control. Every arm already measured must be unaffected."""
        report = self._run(refit=False)

        self.assertNotIn("refit", report)
        self.assertTrue(report["artifact"]["written"],
                        "the fold pass stopped writing its artifact when refit is off")


class WhatMadeTheOldBehaviourInescapableTests(unittest.TestCase):

    def test_folds_1_is_degenerate_so_it_was_never_the_way_out(self) -> None:
        """⛔ `grouped_folds` assigns every group to fold 0 when there is one fold, leaving the
        TRAIN split empty. Anyone reaching for `--folds 1` to train on everything got an error about
        an epoch drawing no batch, not a model."""
        from src.robot.grasping.deep.corpus.index import grouped_folds

        groups = [f"g{i // 3}" for i in range(30)]
        train, test = grouped_folds(groups, folds=1, seed=0)[0]

        self.assertEqual(len(train), 0, "one fold no longer leaves the training split empty")
        self.assertEqual(len(test), len(groups))


if __name__ == "__main__":
    unittest.main()
