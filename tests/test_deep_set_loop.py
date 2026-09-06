"""Training the set generator over a corpus: the index, the folds, and where the gripper comes from.

⚠ THE THREE FAILURES THIS FILE IS FOR all produce a number rather than an error.

* An index that quietly holds the corpus. One cloud is 1.33 MB, so v5 is 26.0 GB, and the loop this
  one replaces reads every scene eagerly. On a 1,980-scene corpus that was 2.6 GB and fine.
* Folds that are not asset-disjoint. A 21.5M-parameter model on 962 assets is exactly the regime
  where a high-capacity model memorises, so a held-out number computed over assets it trained on
  would read as generalisation and be recall.
* A gripper taken from the caller rather than from the cloud. A corpus may mix jaws, and a wrong
  conditioning input is worse than a constant one: constant teaches the head nothing, wrong teaches
  it a relationship that is not there.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.robot.grasping.deep.corpus.sample import SampleSpec
from src.robot.grasping.deep.net.slot_head import SlotHeadConfig
from src.robot.grasping.deep.net.serialized_backbone import SerializedConfig
from src.robot.grasping.deep.net.set_generator import SetGeneratorConfig
from src.robot.grasping.deep.train.trainer import (
    SetTrainingPlan,
    corpus_index,
    plan_with_slots,
    train_set_generator,
)
from tests.test_deep_cli import _scene


def _corpus(directory: Path, scenes: int = 8, gripper: str | None = None) -> list[Path]:
    """Clouds in the corpus's own shape, each with its own asset so the folds have something to cut.

    The gripper stamp is added afterwards rather than by the writer, because the writer is shared
    with the older tests and their corpora predate the stamp: that is the same situation a real
    pre-stamp corpus is in, and it is worth exercising.
    """
    paths: list[Path] = []
    for index in range(scenes):
        path = directory / f"scene_{index:03d}.npz"
        _scene(path, asset=f"gso_asset_{index:03d}")
        if gripper is not None:
            with np.load(path, allow_pickle=False) as handle:
                arrays = {name: handle[name] for name in handle.files}
            arrays["gripper"] = np.asarray([gripper], dtype="<U32")
            np.savez_compressed(path, **arrays)
        paths.append(path)
    return paths


def _small_plan(**kwargs: object) -> SetTrainingPlan:
    base = {
        "epochs": 1, "folds": 4, "run_folds": 1, "batch": 2, "train_units": 4, "eval_units": 2,
        "sample": SampleSpec(grasp_set=True, points=256),
        "model": SetGeneratorConfig(
            backbone=SerializedConfig(width=16, depth=1, heads=4, window=64),
            head=SlotHeadConfig(slots=3, width=32, gripper_width=8, query_width=8)),
    }
    return SetTrainingPlan(**{**base, **kwargs})   # type: ignore[arg-type]


class IndexTests(unittest.TestCase):

    def test_the_index_does_not_hold_the_clouds(self) -> None:
        """⛔ THE POINT OF THE WHOLE DESIGN. If the index carried scenes, v5 would cost 26.0 GB before
        a single step ran."""
        with TemporaryDirectory() as tmp:
            index = corpus_index(_corpus(Path(tmp)))
            for name in vars(type(index)).get("__slots__", ()):
                value = getattr(index, name)
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    self.fail(f"the index is carrying scene dicts under {name!r}")
            self.assertEqual(len(index.files), 8)
            self.assertTrue(all(isinstance(f, Path) for f in index.files))

    def test_it_finds_a_unit_and_a_group_per_object(self) -> None:
        with TemporaryDirectory() as tmp:
            index = corpus_index(_corpus(Path(tmp), scenes=6))
            self.assertEqual(len(index.units), 6)
            self.assertEqual(index.asset_groups, 6)
            self.assertEqual(len(index.groups), len(index.units))

    def test_an_empty_corpus_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            corpus_index([])


class GripperStampTests(unittest.TestCase):

    def test_the_stamp_is_read_from_the_cloud(self) -> None:
        with TemporaryDirectory() as tmp:
            index = corpus_index(_corpus(Path(tmp), scenes=4, gripper="wide_140"))
            self.assertEqual(set(index.grippers), {"wide_140"})

    def test_a_corpus_written_before_the_stamp_reads_as_the_shipped_jaw(self) -> None:
        """Safe only because every corpus this repository wrote before the stamp was labelled with
        the 2F-85. That is a fact about our history, not a property of the format, and it is written
        down where the default lives."""
        with TemporaryDirectory() as tmp:
            self.assertEqual(set(corpus_index(_corpus(Path(tmp), scenes=3)).grippers), {"2f85"})

    def test_a_mixed_corpus_keeps_both_stamps(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a").mkdir()
            (root / "b").mkdir()
            files = _corpus(root / "a", scenes=4, gripper="2f85")
            files += _corpus(root / "b", scenes=4, gripper="slim_pad")
            index = corpus_index(files)
            self.assertEqual(sorted(set(index.grippers)), ["2f85", "slim_pad"])

    def test_an_unknown_stamp_stops_the_run_rather_than_guessing(self) -> None:
        """⛔ Not caught at index time but at the first batch, which is where the vector is needed.
        Either way it must RAISE: training on the wrong hand produces a curve nobody can read."""
        with TemporaryDirectory() as tmp:
            files = _corpus(Path(tmp), scenes=4, gripper="a_hand_we_never_labelled")
            with self.assertRaises(ValueError) as raised:
                train_set_generator(files, _small_plan(), device="cpu", probe_units=0)
            self.assertIn("unknown gripper stamp", str(raised.exception))


class PlanTests(unittest.TestCase):

    def test_the_plan_refuses_a_sample_without_the_label_set(self) -> None:
        """Without it the sample carries one label per point and the K-slot head has nothing to match,
        so every slot would be supervised against the same nearest grasp."""
        with self.assertRaises(ValueError):
            SetTrainingPlan(sample=SampleSpec(grasp_set=False))

    def test_the_plan_refuses_more_folds_run_than_cut(self) -> None:
        with self.assertRaises(ValueError):
            SetTrainingPlan(folds=3, run_folds=5)

    def test_plan_with_slots_changes_exactly_one_thing(self) -> None:
        """⭐ THE FIRST ARM IS `slots=4` AGAINST `slots=1`, and it is only a comparison if nothing
        else moves. A helper that also touched the width or the seed would make the result
        uninterpretable and would look identical in a log."""
        base = _small_plan()
        changed = plan_with_slots(base, 1)
        self.assertEqual(changed.model.head.slots, 1)
        self.assertEqual(base.model.head.slots, 3)
        self.assertEqual(changed.model.backbone, base.model.backbone)
        self.assertEqual(changed.model.head.width, base.model.head.width)
        self.assertEqual(changed.model.head.query_width, base.model.head.query_width)
        self.assertEqual((changed.epochs, changed.batch, changed.seed, changed.learning_rate),
                         (base.epochs, base.batch, base.seed, base.learning_rate))
        self.assertEqual(changed.sample, base.sample)
        self.assertEqual(changed.step, base.step)


class RunTests(unittest.TestCase):

    def test_a_run_reports_train_and_held_out_side_by_side(self) -> None:
        """⚠ The gap between them IS the memorisation probe. A run that reported only one of the two
        could not answer the question this architecture was scaled up to ask."""
        with TemporaryDirectory() as tmp:
            report = train_set_generator(_corpus(Path(tmp)), _small_plan(), device="cpu",
                                         probe_units=0)
            self.assertEqual(len(report["epochs"]), 1)
            row = report["epochs"][0]
            for key in ("total", "coverage", "top1_hit", "approach_error_deg", "offset_error_mm"):
                with self.subTest(key):
                    self.assertIn(f"train_{key}", row)
                    self.assertIn(f"held_{key}", row)
                    self.assertTrue(np.isfinite(row[f"train_{key}"]))
                    self.assertTrue(np.isfinite(row[f"held_{key}"]))

    def test_the_folds_are_asset_disjoint(self) -> None:
        """A held-out number over assets the model trained on is recall wearing a generalisation
        label, and at 21.5M parameters on 962 assets that is the failure mode to expect."""
        from src.robot.grasping.deep.corpus.index import grouped_folds

        with TemporaryDirectory() as tmp:
            index = corpus_index(_corpus(Path(tmp), scenes=12))
            for train, test in grouped_folds(index.groups, folds=4, seed=0):
                trained = {index.groups[i] for i in train}
                held = {index.groups[i] for i in test}
                self.assertFalse(trained & held)

    def test_the_run_writes_a_checkpoint_and_the_epoch_table(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "corpus").mkdir()
            files = _corpus(root / "corpus")
            train_set_generator(files, _small_plan(), out_dir=root / "out", device="cpu",
                                probe_units=0)
            self.assertTrue((root / "out" / "epochs.json").is_file())
            self.assertTrue((root / "out" / "fold0.pt").is_file())

    def test_one_slot_and_four_slots_both_run_on_the_same_corpus(self) -> None:
        """The floor of the sweep has to stay reachable, or there is nothing to compare against."""
        with TemporaryDirectory() as tmp:
            files = _corpus(Path(tmp))
            for slots in (1, 4):
                with self.subTest(slots=slots):
                    report = train_set_generator(files, plan_with_slots(_small_plan(), slots),
                                                 device="cpu", probe_units=0)
                    self.assertTrue(np.isfinite(report["epochs"][0]["train_total"]))

    def test_the_probe_runs_as_part_of_the_run(self) -> None:
        """⭐ AN ACCEPTANCE CRITERION SOMEBODY HAS TO REMEMBER TO INVOKE is one that gets skipped on
        the run where it matters. It runs at the end of every fold, inside the run, and lands in the
        same report as the epochs."""
        with TemporaryDirectory() as tmp:
            report = train_set_generator(_corpus(Path(tmp), scenes=12), _small_plan(),
                                         device="cpu", probe_units=2)
            self.assertIn("probe", report)
            self.assertEqual(len(report["probe"]), 1)
            row = report["probe"][0]
            for key in ("seen_coverage", "unseen_coverage", "gap_coverage",
                        "control_gap_coverage", "excess_coverage"):
                self.assertIn(key, row)

    def test_the_probe_can_be_turned_off_for_a_smoke_test_and_nothing_else(self) -> None:
        with TemporaryDirectory() as tmp:
            report = train_set_generator(_corpus(Path(tmp)), _small_plan(), device="cpu",
                                         probe_units=0)
            self.assertNotIn("probe", report)

    def test_the_held_out_set_is_a_sample_and_not_a_prefix(self) -> None:
        """⛔⛔ `grouped_folds` returns its indices in GROUP order, so a prefix of the test fold is a
        handful of asset groups rather than a cross-section of it. MEASURED on v5: the prefix's
        coverage ceiling was 0.580 against the training population's 0.385 and its labels-per-seed
        0.29 against 0.56, so every held-out number was about those few groups.

        This repository already wrote the lesson down for scene ids, which sort by family: a prefix
        is not a sample. Read off the source, because the draw happens inside a long run.
        """
        source = Path("src/robot/grasping/deep/train/trainer.py").read_text(encoding="utf-8")
        self.assertIn("permutation(test_index)[", source)
        self.assertNotIn("held = test_index[:plan.eval_units]", source)

    def test_the_held_out_set_is_stable_across_epochs_and_arms(self) -> None:
        """Drawn once per fold from its own generator. A held-out set that changed between epochs
        would make the curve a walk over populations, and one that changed between arms would make
        the comparison meaningless."""
        source = Path("src/robot/grasping/deep/train/trainer.py").read_text(encoding="utf-8")
        self.assertIn("np.random.default_rng(plan.seed + 1000 + fold).permutation(test_index)",
                      source)

    def test_the_held_out_population_gets_its_own_ceiling(self) -> None:
        """⭐ A ceiling is a property of a POPULATION. Comparing a held-out coverage against the
        training population's ceiling is how a number came to read as above its own maximum."""
        with TemporaryDirectory() as tmp:
            report = train_set_generator(_corpus(Path(tmp), scenes=12), _small_plan(),
                                         device="cpu", probe_units=2)
            self.assertIn("ceiling", report)
            self.assertIn("held_ceiling", report)
            self.assertEqual(len(report["held_ceiling"]), 1)
            self.assertIn("coverage", report["held_ceiling"][0])

    def test_the_epoch_keeps_its_length_when_the_mix_is_rebalanced(self) -> None:
        """⛔⛔ MEASURED on v5: 68.5 % of training units carry ZERO labelled points, so two thirds of
        an epoch gives the pose heads nothing. The rebalance changes the MIX and not the LENGTH, so a
        run with it on stays comparable to one without on everything else."""
        import numpy as np

        from src.robot.grasping.deep.corpus.index import rebalance_units

        with TemporaryDirectory() as tmp:
            index = corpus_index(_corpus(Path(tmp), scenes=12))
            order = np.arange(len(index.units))
            rebalanced = rebalance_units(order, index.units, 0.5, np.random.default_rng(0))
            self.assertEqual(len(rebalanced), len(order))

    def test_the_natural_mix_stays_reachable(self) -> None:
        """`None` is what every arm before this field existed ran on, so it has to stay available or
        no later run can be compared against them."""
        self.assertIsNone(SetTrainingPlan(labelled_unit_share=None).labelled_unit_share)
        self.assertEqual(SetTrainingPlan().labelled_unit_share, 0.5)

    def test_the_reported_parameter_count_is_the_whole_model(self) -> None:
        with TemporaryDirectory() as tmp:
            report = train_set_generator(_corpus(Path(tmp)), _small_plan(), device="cpu",
                                         probe_units=0)
            self.assertGreater(report["parameters"], 1000)
            self.assertEqual(report["clouds"], 8)


if __name__ == "__main__":
    unittest.main()
