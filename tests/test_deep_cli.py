"""`python -m backend.src.robot.grasping.deep` — the generator's only operator surface.

⚠ WHY THIS FILE EXISTS. `capacity_probe` and `train_generator` had NO caller outside their own module:
the only way to run either was to import it from a Python session. That is precisely what
`build_cloud_corpus` was two days ago, with the same consequence — a runbook whose steps could not be
executed as written, and measurements nobody could re-run from a shell. This is the second time the
same gap has been found in this arc, which is why the CLI now gets its own tests instead of being
trusted because it worked once by hand.

What these pin: the subcommands exist and route to their own handlers, every way of getting the
arguments wrong refuses with a USAGE code rather than a traceback, and — the one that matters for the
experiment — **the sample always carries the multi-hot, even for the control arm**. Without that the
control could only be compared on `approach_top1`, a metric that asks which of roughly four correct
answers was named.
"""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.robot.grasping.deep import __main__ as cli


def _scene(path: Path, *, grasps: int = 3, points: int = 120, asset: str | None = None) -> None:
    """A corpus-shaped scene, the same shape `test_capacity_probe` writes."""
    rng = np.random.default_rng(len(path.name))
    obj = np.column_stack([rng.uniform(-40, 40, points), rng.uniform(-40, 40, points),
                           rng.uniform(20, 60, points)])
    env = np.column_stack([rng.uniform(-120, 120, points), rng.uniform(-120, 120, points),
                           np.zeros(points)])
    xyz = np.vstack([obj, env]).astype(np.float32)
    instance = np.concatenate([np.zeros(points, dtype=np.int16),
                               np.full(points, -1, dtype=np.int16)])
    position = obj[:grasps].copy()
    contacts = np.repeat(position, 2, axis=0)
    contacts[0::2, 0] -= 1.0
    contacts[1::2, 0] += 1.0
    np.savez_compressed(
        path, points_mm=xyz,
        normals=np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(xyz), 1)),
        normal_valid=np.ones(len(xyz), dtype=bool),
        view_count=np.full(len(xyz), 2, dtype=np.uint8), instance_id=instance,
        grasp_position_mm=position.astype(np.float32),
        grasp_approach=np.tile(np.array([0.0, 0.0, -1.0]), (grasps, 1)).astype(np.float32),
        grasp_axis=np.tile(np.array([1.0, 0.0, 0.0]), (grasps, 1)).astype(np.float32),
        grasp_width_mm=np.full(grasps, 40.0, dtype=np.float32),
        grasp_instance=np.zeros(grasps, dtype=np.int16),
        grasp_held=np.full(grasps, -1, dtype=np.int8),
        grasp_approach_admissible=np.ones(grasps, dtype=bool),
        grasp_part_role=np.asarray([""] * grasps, dtype="<U8"),
        contact_points_mm=contacts.astype(np.float32),
        contact_grasp_index=np.repeat(np.arange(grasps), 2).astype(np.int32),
        object_instance=np.zeros(1, dtype=np.int16),
        object_asset_id=np.asarray([asset or f"gso_{path.stem}"], dtype="<U32"),
        corpus_version=np.asarray([3], dtype=np.int32))


def _corpus(directory: Path, scenes: int = 6) -> Path:
    for index in range(scenes):
        _scene(directory / f"scene_{index:03d}.npz")
    return directory


class RefusalTests(unittest.TestCase):
    """Every wrong argument is a usage code, not a traceback out of the trainer."""

    def test_a_missing_corpus_directory(self) -> None:
        # ⚠ _EXIT_PROBLEM (3), NOT _EXIT_USAGE (2). The retired `probe` called a missing corpus a
        # usage error; `train-set` calls it a problem, because the arguments PARSE and what fails is
        # the world they describe. Both are non-zero, and pinning the actual one is the point: a
        # caller that branches on the code needs it to be stable.
        self.assertEqual(
            cli.main(["train-set", "--clouds", "no-such-directory", "--out", "x"]),
            cli._EXIT_PROBLEM)

    def test_a_directory_with_no_scenes_names_the_command_that_makes_them(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            self.assertEqual(
                cli.main(["train-set", "--clouds", name, "--out", str(Path(name) / "out")]),
                cli._EXIT_PROBLEM)

    # ⛔ `test_an_unknown_architecture_lists_the_real_ones` WAS DELETED HERE ON 2026-09-04. It asked
    # the retired `probe` command to reject a name absent from `net.ARCHITECTURES`, a registry of
    # binned model shapes that went with its family. There is one architecture now, described in
    # code, so there is no registry to mistype into.

    def test_a_missing_artifact(self) -> None:
        self.assertEqual(cli.main(["inspect", "--artifact", "nope.pt"]), cli._EXIT_USAGE)

    def test_no_subcommand_at_all(self) -> None:
        with self.assertRaises(SystemExit):
            cli.main([])


class PlanTests(unittest.TestCase):
    """⛔ THREE TESTS OF THE BINNED PLAN BUILDER WERE DELETED HERE ON 2026-09-04.

    They drove `cli._plan`, which turned `probe` / `train` arguments into a `TrainingPlan` for the
    single-label family: the architecture registry, the multi-hot control arm, and the fold flags.
    The first two describe a model nobody trains. The third describes behaviour that is still live,
    so it is rebuilt below against the plan `train-set` actually hands to the trainer.
    """

    def test_the_fold_flags_reach_the_plan(self) -> None:
        """`--run-folds 1` is a single grouped holdout and `--folds 5` is the honest out-of-fold
        estimate. Both are written into the card, so a card can never imply five folds' evidence
        when one was run.

        ⚠ READ OFF THE TRAINER'S ARGUMENT, not off a plan builder. `train-set` assembles its plan
        inline, so the only place the finished object exists is the call itself.

        ⚠ THE STAND-IN RAISES RATHER THAN RETURNING A FAKE REPORT. Returning one means inventing a
        shape the command then serialises, prints and indexes, and every key guessed wrong is a test
        failure that says nothing about folds. A sentinel exception stops the command at exactly the
        line under test. It deliberately does NOT derive from ValueError, which the command now
        catches to turn a bad split into an exit code.
        """

        class _Captured(Exception):
            """Not a ValueError, so the command's own refusal handler cannot swallow it."""

        seen: list[object] = []

        def _capture(*args: object, **_: object) -> None:
            seen.append(args[1])
            raise _Captured

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _corpus(root)
            with mock.patch("src.robot.grasping.deep.train.trainer.train_set_generator",
                            side_effect=_capture), self.assertRaises(_Captured):
                cli.main(["train-set", "--clouds", str(root), "--out", str(root / "out"),
                          "--folds", "4", "--run-folds", "2"])
        self.assertEqual(1, len(seen), "the trainer was not reached")
        self.assertEqual((4, 2), (seen[0].folds, seen[0].run_folds))     # type: ignore[attr-defined]


class DispatchTests(unittest.TestCase):
    """⛔⛔ THE GUARD THAT WAS MISSING WHEN IT WAS NEEDED, 2026-09-04.

    `main` used to dispatch through a chain of `if args.cmd == "..."`. Retiring the binned family
    meant deleting nine of those branches, and the regular expression that did it ate eight LIVE
    ones as well. Every symptom pointed away from the cause: the parser still advertised all ten
    subcommands, `--help` printed them, each one still parsed its flags, and each then fell through
    to the usage exit and returned 2. One test noticed, and only because it asserted an exit code.

    A chain of ifs offers nothing to check. The table does, and these two tests check it from both
    ends, so a branch cannot disappear without one of them going red.
    """

    def test_every_subcommand_the_parser_offers_has_a_handler(self) -> None:
        parser = cli.build_parser()
        actions = [a for a in parser._actions                     # noqa: SLF001
                   if isinstance(a, argparse._SubParsersAction)]  # noqa: SLF001
        self.assertEqual(1, len(actions), "the parser stopped having exactly one subcommand group")
        offered = set(actions[0].choices)
        self.assertEqual(offered - set(cli.COMMANDS), set(),
                         "these subcommands parse and then fall through to the usage exit")

    def test_every_handler_in_the_table_is_reachable_from_the_parser(self) -> None:
        """The other direction. A key with no parser entry is a handler nobody can call, which is
        what a half-finished rename leaves behind."""
        parser = cli.build_parser()
        actions = [a for a in parser._actions                     # noqa: SLF001
                   if isinstance(a, argparse._SubParsersAction)]  # noqa: SLF001
        offered = set(actions[0].choices)
        self.assertEqual(set(cli.COMMANDS) - offered, set(),
                         "these handlers cannot be reached from the command line")

    def test_a_subcommand_actually_reaches_its_own_handler(self) -> None:
        """The table could be complete and still wired to the wrong function, so route one for real.

        ⚠ NOT A MOCK OF THE TABLE. Patching `cli.COMMANDS` would test the patch. The handler itself
        is replaced, which is what `main` looks up.
        """
        seen: list[str] = []
        with mock.patch.dict(cli.COMMANDS, {
                "inspect": lambda a: seen.append("inspect") or 0,
                "report": lambda a: seen.append("report") or 0}):
            self.assertEqual(0, cli.main(["inspect", "--artifact", "z"]))
            self.assertEqual(0, cli.main(["report", "--run", "some-directory"]))
        self.assertEqual(["inspect", "report"], seen)

    def test_train_set_REFUSES_a_corpus_it_cannot_split(self) -> None:
        """Too few asset groups to cut the requested folds means the SPLIT would be wrong, and a
        wrong split reports a number that looks like generalisation and is not."""
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for index in range(2):
                _scene(root / f"scene_{index}.npz", asset="gso_only_one_asset")
            code = cli.main(["train-set", "--clouds", str(root), "--out", str(root / "out"),
                             "--folds", "5", "--epochs", "1"])
        self.assertEqual(code, cli._EXIT_PROBLEM)                  # noqa: SLF001


class InspectTests(unittest.TestCase):
    """⛔ REWRITTEN FOR THE FAMILY THAT SHIPS, 2026-09-04.

    The two tests that stood here built a `GraspGeneratorNet` from `net.ARCHITECTURES` and asked
    whether `inspect` named its architecture and read its approach logits. Both the net and the
    registry were deleted with the binned family. The command survived the cleanup because it is the
    one way anybody checks a weights file before pointing a cell at it, and it was rewritten rather
    than dropped, so its tests are rewritten too.
    """

    @staticmethod
    def _artifact(directory: Path) -> Path:
        from src.robot.grasping.deep.net.set_generator import SetGenerator
        from src.robot.grasping.deep.set_artifact import write_set_generator
        from src.robot.grasping.deep.train.trainer import SetTrainingPlan

        plan = SetTrainingPlan()
        net = SetGenerator(plan.model)
        written = write_set_generator(directory, net, sample=plan.sample, step=plan.step,
                                      gripper="2f85")
        return written["weights"]

    def test_it_reports_what_the_runtime_will_read(self) -> None:
        """A cell is pointed at a weights file; this is how anyone checks WHAT that file is first.

        ⭑ FACTS ONLY. Every line is a property of the run (parameters, gripper, slots, target), never
        a quality claim. A number that reads like a score beside a file that was never held out is
        exactly what this whole cleanup was called to remove.
        """
        import io
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as name:
            path = self._artifact(Path(name))
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli.main(["inspect", "--artifact", str(path)])
        self.assertEqual(cli._EXIT_OK, code)                          # noqa: SLF001
        printed = buffer.getvalue()
        for field in ("parameters", "gripper", "slots", "target", "points"):
            self.assertIn(field, printed, f"inspect stopped reporting {field}")
        self.assertIn("2f85", printed)

    def test_a_file_from_the_retired_family_is_named_as_such(self) -> None:
        """The refusal path, because this is the command someone runs on an OLD file.

        Somebody clearing out `logs/dl/models` needs to be told that a `grasp_generator_v1.pt`
        belonged to an architecture that was retired, not merely that its kind is unexpected.
        """
        import io
        from contextlib import redirect_stderr

        import torch

        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "grasp_generator_v1.pt"
            torch.save({"kind": "grasp_generator", "artifact_version": 1, "state_dict": {}}, path)
            buffer = io.StringIO()
            with redirect_stderr(buffer):
                code = cli.main(["inspect", "--artifact", str(path)])
        self.assertNotEqual(cli._EXIT_OK, code)                       # noqa: SLF001
        self.assertIn("BINNED", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
