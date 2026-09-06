"""Training the set generator FROM CODE, which is what the CLI could not offer.

⭑ WHAT THESE TESTS ARE FOR. The owner asked for training to be callable like `AutonomousGraspService`
rather than only through a command line. Three things blocked that, and each has a test here because
each was a real defect rather than a missing convenience:

  * the recipe merge read `sys.argv`, so from code it overrode values the caller had passed;
  * the plan was assembled in nine chained `dataclasses.replace` calls whose ORDER is load-bearing,
    and getting it wrong produced a run that REPORTED a setting it did not run;
  * the four enum checks, the backbone divisibility check and the crop pre-check lived in the CLI,
    so a code caller learned about a typo after the corpus walk and the probes.

⚠ NO TRAINING RUNS HERE. Fitting even a tiny net takes seconds and needs a corpus on disk, and none
of the above is about fitting. The trainer is patched where a test needs to see what reached it.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.robot.grasping.deep.train.api import GeneratorTraining
from src.robot.grasping.deep.train.plan import UNSET, PlanOverrides, build_plan
from src.robot.grasping.deep.train.report import (
    EpochRow,
    TrainingOutcome,
    TrainingRunReport,
)
from src.robot.grasping.deep.train.trainer import SetTrainingPlan


def _scene_dir(count: int = 3) -> Path:
    """A directory holding `count` empty `.npz` files, which is all `scene_files` looks at."""
    root = Path(tempfile.mkdtemp())
    for index in range(count):
        (root / f"packed_{index}.npz").write_bytes(b"")
    return root


class TheRecipeMergeNeedsNoArgvTests(unittest.TestCase):
    """⛔⛔ THE HARDEST BLOCKER, AND THE REASON `UNSET` EXISTS.

    A recipe applies only to settings the caller did not choose. That was decided by
    `typed = set(sys.argv)`, which is meaningful from a shell and meaningless anywhere else: under
    pytest or a service process `sys.argv` holds the runner's arguments, `--epochs` never appears,
    and the recipe silently overrode a value the caller had passed explicitly. That is exactly the
    defect the argv scan was written to prevent, reappearing one layer out.
    """

    def test_an_explicit_value_beats_the_tier(self) -> None:
        plan, notes = build_plan(tier="smoke", overrides=PlanOverrides(epochs=7))
        self.assertEqual(7, plan.epochs, "the tier overrode a value the caller chose")
        self.assertIn("epochs=7", notes["overridden"])
        self.assertNotIn("epochs", notes)

    def test_the_tier_still_applies_to_everything_else(self) -> None:
        """The control. Without it the test above passes for a merge that applies nothing at all."""
        plan, notes = build_plan(tier="smoke", overrides=PlanOverrides(epochs=7))
        self.assertEqual(400, plan.train_units)
        self.assertEqual(400, notes["train_units"])

    def test_an_explicit_FALSE_is_not_mistaken_for_absence(self) -> None:
        """⛔ WHY `UNSET` IS A CLASS AND NOT `None` OR A FALSY DEFAULT. `--refit` is a store_true, so
        "not passed" and "passed as False" are the same value. If absence were expressed as False,
        `refit=False` from a caller could never be told apart from a caller who said nothing, and
        `--tier smoke` could not turn refit off, which is the one thing it exists to do."""
        plan, _ = build_plan(recipe="v1", overrides=PlanOverrides(refit=False))
        self.assertFalse(plan.refit, "the recipe overrode an explicit refit=False")

        without, _ = build_plan(recipe="v1")
        self.assertTrue(without.refit, "the recipe stopped applying at all")

    def test_the_argv_scan_is_gone_from_the_command(self) -> None:
        """The instrument that could not be right, read off the source.

        ⚠ BY AST. A text search for `sys.argv` would match the comment that explains why it was
        removed, which is the trap this branch hit four times in one day.
        """
        import ast

        source = Path("src/robot/grasping/deep/__main__.py").read_text(encoding="utf-8")
        command = next(node for node in ast.walk(ast.parse(source))
                       if isinstance(node, ast.FunctionDef) and node.name == "_cmd_train_set")
        reads = [node for node in ast.walk(command)
                 if isinstance(node, ast.Attribute) and node.attr == "argv"]
        self.assertEqual([], reads, "_cmd_train_set reads sys.argv again")


class ThePlanOrderIsOwnedOnceTests(unittest.TestCase):
    """⛔⛔ THE ORDER IS LOAD-BEARING AND USED TO BE A COMMENT.

    `plan_with_slots` REBUILDS `plan.model` and `plan.model.head`, so a head setting applied before
    it is silently discarded. The failure mode is the worst kind this project has: the run REPORTS
    the setting it was asked for and does not run it.
    """

    def test_slot_mixing_survives_a_slot_count(self) -> None:
        plan, _ = build_plan(overrides=PlanOverrides(slots=4, slot_mixing="film"))
        self.assertEqual(4, plan.model.head.slots)
        self.assertEqual("film", plan.model.head.slot_mixing,
                         "plan_with_slots rebuilt the head and dropped the mixing")

    def test_axis_mode_survives_a_slot_count(self) -> None:
        plan, _ = build_plan(overrides=PlanOverrides(slots=2, axis_mode="vector"))
        self.assertEqual("vector", plan.model.head.axis_mode)

    def test_part_roles_survive_a_slot_count(self) -> None:
        """The affordance head is built BEFORE the slots and has to survive the rebuild too."""
        plan, _ = build_plan(overrides=PlanOverrides(slots=2, part_roles=True))
        self.assertTrue(plan.model.head.part_roles)

    def test_the_backbone_survives_a_slot_count(self) -> None:
        plan, _ = build_plan(overrides=PlanOverrides(slots=2, backbone_width=128, backbone_heads=4))
        self.assertEqual((128, 4), (plan.model.backbone.width, plan.model.backbone.heads))


class RefusalsHappenAtConstructionTests(unittest.TestCase):
    """⚠ BEFORE THE CORPUS WALK AND THE PROBES, not after them.

    Every one of these was checked by the CLI and by nothing else, so a code caller reached the
    downstream raise only after minutes of indexing and the ceiling, floor and memorisation probes.
    """

    def test_an_unknown_target(self) -> None:
        with self.assertRaises(ValueError) as caught:
            build_plan(overrides=PlanOverrides(target="telepathy"))
        self.assertIn("approach", str(caught.exception), "the refusal does not list the real ones")

    def test_an_unknown_axis_mode(self) -> None:
        with self.assertRaises(ValueError):
            build_plan(overrides=PlanOverrides(axis_mode="nonsense"))

    def test_an_unknown_slot_mixing(self) -> None:
        with self.assertRaises(ValueError):
            build_plan(overrides=PlanOverrides(slot_mixing="nonsense"))

    def test_an_unknown_control_level(self) -> None:
        with self.assertRaises(ValueError):
            build_plan(overrides=PlanOverrides(control="nonsense"))

    def test_a_backbone_width_the_heads_do_not_divide_SUGGESTS_head_counts(self) -> None:
        """`serialized_backbone.py` raises the bare form with no hint, and the architecture plan's
        own 6.5M variant (width 256) trips it."""
        with self.assertRaises(ValueError) as caught:
            build_plan(overrides=PlanOverrides(backbone_width=100, backbone_heads=7))
        self.assertIn("[1, 2, 4, 5, 10", str(caught.exception))

    def test_an_unknown_recipe_refuses_rather_than_falling_back(self) -> None:
        """A customer who typed `v2` before it exists must not get `v1` under `v2`'s name."""
        with self.assertRaises(ValueError):
            build_plan(recipe="v2")


class TheContextIsSeparateFromThePlanTests(unittest.TestCase):

    def test_a_directory_is_walked_and_a_sequence_is_taken_as_given(self) -> None:
        root = _scene_dir(3)
        walked = GeneratorTraining.from_plan(corpus=root)
        self.assertEqual(3, len(walked.context.corpus))

        given = GeneratorTraining.from_plan(corpus=[root / "packed_0.npz"])
        self.assertEqual(1, len(given.context.corpus))

    def test_an_empty_sequence_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            GeneratorTraining.from_plan(corpus=[])

    def test_a_missing_directory_is_refused(self) -> None:
        with self.assertRaises(FileNotFoundError):
            GeneratorTraining.from_plan(corpus=Path(tempfile.mkdtemp()) / "nope")


class WhatReachesTheTrainerTests(unittest.TestCase):
    """The API is a wrapper, so the thing worth testing is what it hands to the trainer."""

    def test_the_plan_and_the_context_arrive_intact(self) -> None:
        root = _scene_dir(2)
        seen: dict[str, object] = {}

        def capture(files, plan, **kwargs):        # type: ignore[no-untyped-def]
            seen["files"] = tuple(files)
            seen["plan"] = plan
            seen.update(kwargs)
            return {"epochs": [], "artifact": {}}

        run = GeneratorTraining.from_recipe(
            corpus=root, tier="smoke", out_dir=root / "out", artifact_gripper="2f85")
        with mock.patch("src.robot.grasping.deep.train.trainer.train_set_generator",
                        side_effect=capture):
            run.train()
        self.assertEqual(2, len(seen["files"]))
        self.assertEqual(2, seen["plan"].epochs)                      # type: ignore[union-attr]
        self.assertEqual("2f85", seen["artifact_gripper"])
        self.assertEqual(root / "out", seen["out_dir"])

    def test_the_output_directory_is_created_by_the_api_not_the_cli(self) -> None:
        """A run launched from code must leave the same directory a shell run leaves."""
        root = _scene_dir(1)
        target = root / "made" / "by" / "the" / "api"
        with mock.patch("src.robot.grasping.deep.train.trainer.train_set_generator",
                        return_value={"epochs": [], "artifact": {}}):
            GeneratorTraining.from_plan(corpus=root, out_dir=target).train()
        self.assertTrue(target.is_dir())

    def test_a_progress_listener_sees_every_epoch(self) -> None:
        """⚠ OFF BY DEFAULT so a run without one is byte-identical; the trainer's parameter defaults
        to None and only this method sets it."""
        root = _scene_dir(1)
        run = GeneratorTraining.from_plan(corpus=root)
        seen: list[int] = []
        run.attach_progress_listener(lambda row: seen.append(int(row["epoch"])))

        def fire(files, plan, **kwargs):           # type: ignore[no-untyped-def]
            for epoch in range(3):
                kwargs["on_epoch"]({"epoch": epoch})
            return {"epochs": [], "artifact": {}}

        with mock.patch("src.robot.grasping.deep.train.trainer.train_set_generator",
                        side_effect=fire):
            run.train()
        self.assertEqual([0, 1, 2], seen)

    def test_no_listener_means_the_trainer_gets_None(self) -> None:
        root = _scene_dir(1)
        seen: dict[str, object] = {}

        def capture(files, plan, **kwargs):        # type: ignore[no-untyped-def]
            seen.update(kwargs)
            return {"epochs": [], "artifact": {}}

        with mock.patch("src.robot.grasping.deep.train.trainer.train_set_generator",
                        side_effect=capture):
            GeneratorTraining.from_plan(corpus=root).train()
        self.assertIsNone(seen["on_epoch"])


class TheReportCarriesTheStampsTests(unittest.TestCase):
    """⛔ FIVE STAMPS WERE ADDED BY THE CLI AFTER THE TRAINER RETURNED, so no code caller saw them.

    `control_level` is the one that matters: a control run fits a synthetic target derived from an
    input channel, and losing that stamp is how a diagnostic becomes a quoted result later.
    """

    @staticmethod
    def _report(**plan_kwargs: object) -> TrainingRunReport:
        plan = dataclasses.replace(SetTrainingPlan(), **plan_kwargs)  # type: ignore[arg-type]
        raw = {"epochs": [{"fold": 0, "epoch": 0, "held_total": 1.0, "held_top1_hit": 0.5,
                           "held_coverage": 0.1, "held_offset_error_mm": 20.0, "seconds": 1.0,
                           "total": 1.0, "train_top1_hit": 0.6}],
               "artifact": {"written": True, "weights": "x.pt"}}
        return TrainingRunReport.from_trainer(raw, plan)

    def test_the_control_level_is_in_the_json(self) -> None:
        payload = self._report(control="normal").as_dict()
        self.assertEqual("normal", payload["control_level"])

    def test_the_tier_is_in_the_json(self) -> None:
        """⛔ IT WAS IN NEITHER THE PLAN NOR THE CARD until 2026-09-04, so a `--tier smoke` artifact
        (two epochs, four hundred units, refit off) was indistinguishable at load time from a run
        that took hours."""
        payload = self._report(tier="smoke").as_dict()
        self.assertEqual("smoke", payload["tier"])

    def test_a_control_run_says_so_in_the_rendered_text(self) -> None:
        rendered = self._report(control="normal").render()
        self.assertIn("CONTROL RUN", rendered)
        self.assertIn("NOT about grasping", rendered)

    def test_the_rendered_text_is_ascii(self) -> None:
        """⚠ A Windows console under cp1252 cannot print anything else, and a non-ASCII character in
        a printed report line has broken this repository's CLI before."""
        for report in (self._report(), self._report(control="normal"), self._report(tier="smoke")):
            report.render().encode("ascii")

    def test_the_describe_text_is_ascii(self) -> None:
        root = _scene_dir(1)
        run = GeneratorTraining.from_recipe(corpus=root, tier="smoke")
        run.describe().encode("ascii")

    def test_describe_and_render_do_not_repeat_each_other(self) -> None:
        """The command prints both, and the first version printed the configuration block twice."""
        root = _scene_dir(1)
        run = GeneratorTraining.from_recipe(corpus=root, tier="smoke")
        described = set(run.describe().split(chr(10)))
        rendered = set(self._report(tier="smoke").render().split(chr(10)))
        self.assertEqual(set(), described & rendered)


class TheEmptyEpochListIsGuardedTests(unittest.TestCase):

    def test_final_is_None_rather_than_an_IndexError(self) -> None:
        """⛔ THE CLI DID `report["epochs"][-1]` UNGUARDED, and an empty list is reachable: resuming
        a checkpoint already at `plan.epochs` returns `first_epoch == plan.epochs`, the epoch range
        is empty and no row is appended, so that line raised after a SUCCESSFUL trainer return."""
        report = TrainingRunReport.from_trainer({"epochs": [], "artifact": {}}, SetTrainingPlan())
        self.assertIsNone(report.final)
        self.assertIs(TrainingOutcome.NOTHING_TO_RESUME, report.outcome)
        self.assertIn("already finished", report.failure_summary())
        report.render()          # must not raise

    def test_a_refused_artifact_is_not_a_success(self) -> None:
        raw = {"epochs": [{"epoch": 0}], "artifact": {"written": False, "reason": "two grippers"}}
        report = TrainingRunReport.from_trainer(raw, SetTrainingPlan())
        self.assertIs(TrainingOutcome.ARTIFACT_REFUSED, report.outcome)
        self.assertFalse(report.succeeded)
        self.assertIn("two grippers", report.failure_summary())


class TheReportFileIsWrittenByTheApiTests(unittest.TestCase):
    """⛔ `report.json` WAS WRITTEN ONLY BY THE CLI, and `deep report --run DIR` reads it: it merges
    the file when `epochs.json` carries no `held_floor`. A run launched from code therefore produced
    a directory the project's own analysis tool degrades on."""

    def test_it_lands_beside_the_run(self) -> None:
        root = _scene_dir(1)
        run = GeneratorTraining.from_plan(corpus=root, out_dir=root / "out")
        report = TrainingRunReport.from_trainer({"epochs": [], "artifact": {}}, run.plan)
        written = run.write_report(report)
        self.assertEqual(root / "out" / "report.json", written)
        self.assertIn("control_level", json.loads(written.read_text(encoding="utf-8")))

    def test_without_an_out_dir_it_refuses_rather_than_guessing(self) -> None:
        run = GeneratorTraining.from_plan(corpus=_scene_dir(1))
        report = TrainingRunReport.from_trainer({"epochs": [], "artifact": {}}, run.plan)
        with self.assertRaises(ValueError):
            run.write_report(report)


class TheApiReachesMoreThanTheFlagsDoTests(unittest.TestCase):
    """⭐ THE POINT OF A TYPED PLAN OVER A FLAG LIST, stated as a test so it cannot quietly stop
    being true. These three have no command-line flag at all."""

    def test_learning_rate_and_weight_decay_and_eval_units_are_reachable(self) -> None:
        plan, _ = build_plan(overrides=PlanOverrides(
            learning_rate=3e-4, weight_decay=0.0, eval_units=64))
        self.assertEqual((3e-4, 0.0, 64), (plan.learning_rate, plan.weight_decay, plan.eval_units))

    def test_a_hand_built_plan_stays_first_class(self) -> None:
        """No recipe, no overrides: constructing the dataclass directly is the PyTorch arrangement
        and is what the owner asked for instead of a YAML file."""
        plan = dataclasses.replace(SetTrainingPlan(), epochs=99, learning_rate=1e-5)
        run = GeneratorTraining.from_plan(corpus=_scene_dir(1), plan=plan)
        self.assertEqual(99, run.plan.epochs)
        self.assertEqual(1e-5, run.plan.learning_rate)


class UnsetIsNotFalsyByAccidentTests(unittest.TestCase):

    def test_it_is_falsy_on_purpose_and_distinguishable_from_False(self) -> None:
        self.assertFalse(bool(UNSET))
        self.assertIsNot(UNSET, False)
        self.assertNotIsInstance(UNSET, bool)

    def test_chosen_reports_only_what_was_set(self) -> None:
        chosen = PlanOverrides(epochs=3, refit=False).forwarded()
        self.assertEqual({"epochs": 3, "refit": False}, chosen)

    def test_None_is_a_VALUE_and_not_absence(self) -> None:
        """`train_units=None` means every unit, `control=None` means the corpus labels. If absence
        were `None` a recipe would overwrite both."""
        self.assertEqual({"train_units": None}, PlanOverrides(train_units=None).forwarded())


class EpochRowKeepsTheRawColumnsTests(unittest.TestCase):

    def test_a_column_the_view_does_not_name_is_still_reachable(self) -> None:
        """⚠ The trainer's columns change as instruments are added, and a typed subset that silently
        dropped a new one would be worse than no typing."""
        row = EpochRow.from_row({"epoch": 2, "held_top1_hit": 0.5, "some_new_probe": 1.25})
        self.assertEqual(2, row.epoch)
        self.assertEqual(1.25, row.raw["some_new_probe"])

    def test_a_missing_column_is_nan_rather_than_zero(self) -> None:
        """A zero would be read as a measured value; NaN cannot be."""
        row = EpochRow.from_row({"epoch": 0})
        self.assertNotEqual(row.held_top1_hit, row.held_top1_hit)     # NaN != NaN


if __name__ == "__main__":
    unittest.main()
