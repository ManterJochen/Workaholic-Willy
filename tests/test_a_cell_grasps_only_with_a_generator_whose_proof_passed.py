"""A cell grasps with a trained generator only once its proof has passed, and every evaluation still builds one.

The owner's decision of 2026-10-09: no finished models ship and every customer trains their own, so a cell must never
grasp with a trained generator that has not passed its proof. The proof leaves a promotion record beside the weights
(``deep/promotion.py``), and ``build_calculator`` and ``preflight_calculator`` refuse a cell's deep calculator without
a passed one at ``active``, in one sentence that says why and what to do. Nothing writes a record yet, so today that is
every artifact, which is the point: the best full run so far did not beat "straight down" on objects it never saw.

``purpose="evaluate"`` skips that one check and keeps every other refusal, because the ladder, ``deep propose``, the
simulation runners and the offline sweeps are how a proof gets made. ``deep inspect`` prints the verdict first, and
``train-set`` ends on it.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest import mock

import torch

from src.config.schema.robot import RobotConfig
from src.robot.grasping.deep.set_artifact import SET_ARTIFACT_KIND, SET_ARTIFACT_VERSION
from tests._deep_promotion import promote

_ROOT = Path(__file__).resolve().parents[1]
_DEEP = "src.robot.grasping.deep.calculator.DeepGraspCalculator"

#: The refusal a cell gets for an artifact with no record beside it, word for word.
_NO_PROMOTION = (
    "generator.pt carries no promotion, so this cell will not grasp with it: a trained generator drives a cell only "
    "once its proof has passed (deep judge, coming); until then set robot.grasping.calculator: geometric, or evaluate "
    "the artifact with the ladder.")


def _artifact(directory: Path, *, kind: str = SET_ARTIFACT_KIND, version: int = SET_ARTIFACT_VERSION,
              card: dict[str, Any] | None = None) -> Path:
    """The smallest file the factory's artifact checks accept, for the 2F-85, with ``card`` as its card's report."""
    path = directory / "generator.pt"
    torch.save({"kind": kind, "artifact_version": version, "gripper": "2f85", "trained_grippers": ["2f85"]}, path)
    if card is not None:
        path.with_suffix(".card.json").write_text(json.dumps({"report": card}), encoding="utf-8")
    return path


def _cell(artifact: Path, *, hand: str = "robotiq_2f85") -> RobotConfig:
    return RobotConfig.model_validate({
        "vendor": "dummy", "gripper": {"model": hand},
        "grasping": {"calculator": "deep", "deep_generator": {"artifact_path": str(artifact)}}})


class _WithAFolder(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="willy_promotion_")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _refusal(self, artifact: Path, **kwargs: Any) -> str:
        """The sentence ``build_calculator`` refuses a cell with; the net itself must never be built."""
        from src.robot.grasping.calculator_factory import build_calculator

        with mock.patch(_DEEP) as deep, self.assertRaises(ValueError) as caught:
            build_calculator(_cell(artifact), camera_matrix=None, **kwargs)
        deep.assert_not_called()
        return str(caught.exception)

    def _built(self, artifact: Path, **kwargs: Any) -> Any:
        """The mocked deep calculator class, after a build that went through."""
        from src.robot.grasping.calculator_factory import build_calculator

        with mock.patch(_DEEP) as deep:
            build_calculator(_cell(artifact, hand=kwargs.pop("hand", "robotiq_2f85")), camera_matrix=None, **kwargs)
        return deep


class ACellRefusesAnArtifactWhoseProofHasNotPassedTests(_WithAFolder):
    def test_no_record_refuses_in_one_sentence_that_names_the_way_on(self) -> None:
        self.assertEqual(_NO_PROMOTION, self._refusal(_artifact(self.dir)))

    def test_a_record_written_for_other_bytes_refuses(self) -> None:
        artifact = _artifact(self.dir)
        promote(artifact, sha256="0" * 64)
        message = self._refusal(artifact)
        self.assertIn("generator.pt carries a promotion written for other bytes", message)
        self.assertIn("this cell will not grasp with it", message)

    def test_a_file_changed_after_its_proof_refuses(self) -> None:
        """The record vouches for the bytes it was written for, not for the name."""
        artifact = _artifact(self.dir)
        promote(artifact)
        with artifact.open("ab") as handle:
            handle.write(b"retrained since")
        self.assertIn("written for other bytes", self._refusal(artifact))

    def test_a_failed_proof_refuses(self) -> None:
        artifact = _artifact(self.dir)
        promote(artifact, verdict="fail")
        self.assertIn("carries a promotion whose proof failed (verdict 'fail'", self._refusal(artifact))

    def test_a_phase_below_active_refuses(self) -> None:
        for phase in ("shadow", "ab"):
            with self.subTest(phase=phase):
                artifact = _artifact(self.dir)
                promote(artifact, phase=phase)
                message = self._refusal(artifact)
                self.assertIn(f"is promoted only to phase {phase!r}", message)
                self.assertIn("'active' alone", message)

    def test_a_smoke_card_refuses_even_beside_a_passed_record(self) -> None:
        """A smoke run proves the chain closes and nothing about grasp quality; no record makes it a cell's."""
        artifact = _artifact(self.dir, card={"tier": "smoke"})
        promote(artifact)
        message = self._refusal(artifact)
        self.assertIn("generator.pt is a smoke-tier run by its own card (generator.card.json)", message)
        self.assertIn("this cell will not grasp with it", message)

    def test_a_control_card_refuses_even_beside_a_passed_record(self) -> None:
        artifact = _artifact(self.dir, card={"tier": "full", "control": "normal"})
        promote(artifact)
        self.assertIn("is a control run by its own card (generator.card.json, control 'normal')",
                      self._refusal(artifact))

    def test_a_card_that_cannot_be_read_refuses(self) -> None:
        artifact = _artifact(self.dir)
        promote(artifact)
        artifact.with_suffix(".card.json").write_text("{not json", encoding="utf-8")
        self.assertIn("has a card that cannot be read", self._refusal(artifact))

    def test_a_record_that_is_not_json_refuses_as_no_record_does(self) -> None:
        artifact = _artifact(self.dir)
        artifact.with_suffix(".promotion.json").write_text("passed, honestly", encoding="utf-8")
        message = self._refusal(artifact)
        self.assertIn("carries a promotion record that cannot be read (generator.promotion.json is not JSON", message)

    def test_a_record_of_another_kind_refuses(self) -> None:
        artifact = _artifact(self.dir)
        promote(artifact, kind="success_model_promotion")
        self.assertIn("is a record of another kind ('success_model_promotion'", self._refusal(artifact))

    def test_a_record_missing_a_field_refuses(self) -> None:
        artifact = _artifact(self.dir)
        record = promote(artifact)
        payload = json.loads(record.read_text(encoding="utf-8"))
        del payload["phase"]
        record.write_text(json.dumps(payload), encoding="utf-8")
        self.assertIn("generator.promotion.json lacks 'phase'", self._refusal(artifact))

    def test_a_record_with_an_unknown_phase_refuses(self) -> None:
        artifact = _artifact(self.dir)
        promote(artifact, phase="canary")
        self.assertIn("gives phase as 'canary'", self._refusal(artifact))

    def test_a_check_that_cannot_run_refuses_rather_than_waving_through(self) -> None:
        artifact = _artifact(self.dir)
        promote(artifact)
        with mock.patch("src.robot.grasping.deep.promotion.why_not_deployable", side_effect=RuntimeError("disk")):
            self.assertIn("promotion could not be checked (RuntimeError: disk)", self._refusal(artifact))

    def test_the_preflight_refuses_the_same(self) -> None:
        from src.robot.grasping.calculator_factory import preflight_calculator

        with self.assertRaises(ValueError) as caught:
            preflight_calculator(_cell(_artifact(self.dir)))
        self.assertEqual(_NO_PROMOTION, str(caught.exception))

    def test_the_rehearsal_cell_refuses_the_same(self) -> None:
        """The cell builders pass no purpose, so they get the gate."""
        from src.robot.execution.autonomous_grasp.cells import build_rehearsal_components

        with self.assertRaises(ValueError) as caught:
            build_rehearsal_components(_cell(_artifact(self.dir)))
        self.assertEqual(_NO_PROMOTION, str(caught.exception))

    def test_the_proof_is_asked_before_the_hand(self) -> None:
        """A file no proof has passed is no cell's, whatever hand it was trained for."""
        from src.robot.grasping.calculator_factory import build_calculator

        with mock.patch(_DEEP), self.assertRaises(ValueError) as caught:
            build_calculator(_cell(_artifact(self.dir), hand="robotiq_hande"), camera_matrix=None)
        self.assertEqual(_NO_PROMOTION, str(caught.exception))


class APassedProofAtActiveBuildsTheCellTests(_WithAFolder):
    def test_a_passed_record_at_active_builds(self) -> None:
        artifact = _artifact(self.dir)
        promote(artifact)
        deep = self._built(artifact)
        deep.assert_called_once()
        self.assertEqual(deep.call_args.args[0].artifact_path, str(artifact))

    def test_a_full_tier_card_beside_it_changes_nothing(self) -> None:
        artifact = _artifact(self.dir, card={"tier": "full", "control": None, "weights_are": "refit_on_every_unit"})
        promote(artifact)
        self._built(artifact).assert_called_once()

    def test_the_preflight_answers_deep(self) -> None:
        from src.robot.grasping.calculator_factory import preflight_calculator

        artifact = _artifact(self.dir)
        promote(artifact)
        self.assertEqual("deep", preflight_calculator(_cell(artifact)))

    def test_the_hand_is_still_asked_after_the_proof(self) -> None:
        from src.robot.grasping.calculator_factory import build_calculator

        artifact = _artifact(self.dir)
        promote(artifact)
        with mock.patch(_DEEP), self.assertRaises(ValueError) as caught:
            build_calculator(_cell(artifact, hand="robotiq_hande"), camera_matrix=None)
        self.assertIn("robot.gripper.model", str(caught.exception))


class AnEvaluationBuildsWithoutAProofTests(_WithAFolder):
    """``purpose="evaluate"`` skips the proof, and only the proof."""

    def test_an_unpromoted_artifact_builds_for_evaluation(self) -> None:
        self._built(_artifact(self.dir), purpose="evaluate").assert_called_once()

    def test_a_smoke_artifact_builds_for_evaluation_too(self) -> None:
        """The ladder may grade a smoke run; a cell may not grasp with one."""
        self._built(_artifact(self.dir, card={"tier": "smoke"}), purpose="evaluate").assert_called_once()

    def test_the_preflight_evaluates_the_same(self) -> None:
        from src.robot.grasping.calculator_factory import preflight_calculator

        self.assertEqual("deep", preflight_calculator(_cell(_artifact(self.dir)), purpose="evaluate"))

    def test_the_retired_kind_still_refuses(self) -> None:
        from src.robot.grasping.calculator_factory import build_calculator

        with mock.patch(_DEEP), self.assertRaises(ValueError) as caught:
            build_calculator(_cell(_artifact(self.dir, kind="grasp_generator")), camera_matrix=None,
                             purpose="evaluate")
        self.assertIn("BINNED", str(caught.exception))

    def test_another_version_still_refuses(self) -> None:
        from src.robot.grasping.calculator_factory import preflight_calculator

        with self.assertRaises(ValueError) as caught:
            preflight_calculator(_cell(_artifact(self.dir, version=SET_ARTIFACT_VERSION + 1)), purpose="evaluate")
        self.assertIn("artifact_version", str(caught.exception))

    def test_a_hand_the_artifact_never_saw_still_refuses(self) -> None:
        from src.robot.grasping.calculator_factory import build_calculator

        with mock.patch(_DEEP), self.assertRaises(ValueError) as caught:
            build_calculator(_cell(_artifact(self.dir), hand="robotiq_hande"), camera_matrix=None,
                             purpose="evaluate")
        self.assertIn("robot.gripper.model", str(caught.exception))

    def test_a_missing_artifact_still_refuses_as_a_missing_file(self) -> None:
        from src.robot.grasping.calculator_factory import build_calculator, preflight_calculator

        cell = _cell(self.dir / "absent.pt")
        with self.assertRaises(FileNotFoundError):
            build_calculator(cell, camera_matrix=None, purpose="evaluate")
        with self.assertRaises(FileNotFoundError):
            preflight_calculator(cell, purpose="evaluate")

    def test_an_active_kwarg_the_deep_path_cannot_honour_still_refuses(self) -> None:
        from src.robot.grasping.calculator_factory import build_calculator

        with mock.patch(_DEEP), self.assertRaises(ValueError) as caught:
            build_calculator(_cell(_artifact(self.dir)), camera_matrix=None, purpose="evaluate", ik_service=object())
        self.assertIn("ik_service", str(caught.exception))


class APurposeIsOneOfTwoTests(_WithAFolder):
    def test_an_unknown_purpose_refuses_at_both_doors_and_for_both_generators(self) -> None:
        from src.robot.grasping.calculator_factory import build_calculator, preflight_calculator

        geometric = RobotConfig.model_validate({"grasping": {"calculator": "geometric"}})
        deep = _cell(_artifact(self.dir))
        for cell in (geometric, deep):
            for door in (lambda c: build_calculator(c, camera_matrix=None, purpose="evaluation"),  # type: ignore[arg-type]
                         lambda c: preflight_calculator(c, purpose="evaluation")):  # type: ignore[arg-type]
                with self.subTest(calculator=cell.grasping.calculator), self.assertRaises(ValueError) as caught:
                    door(cell)
                self.assertIn("unknown calculator purpose 'evaluation'", str(caught.exception))

    def test_the_analytic_generator_needs_no_proof(self) -> None:
        from src.robot.grasping.calculator_factory import build_calculator, preflight_calculator

        cell = RobotConfig.model_validate({"grasping": {"calculator": "geometric"}})
        for purpose in ("cell", "evaluate"):
            with self.subTest(purpose=purpose):
                self.assertEqual("GraspCalculator", type(build_calculator(cell, camera_matrix=None,
                                                                          purpose=purpose)).__name__)  # type: ignore[arg-type]
                self.assertEqual("geometric", preflight_calculator(cell, purpose=purpose))  # type: ignore[arg-type]


def _purposes(path: Path, names: set[str]) -> list[tuple[str, int, object]]:
    """Every call in ``path`` to one of ``names``, as ``(name, line, purpose)``; ``None`` where it passes none."""
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        if name not in names:
            continue
        given = [keyword.value for keyword in node.keywords if keyword.arg == "purpose"]
        purpose = given[0].value if given and isinstance(given[0], ast.Constant) else (None if not given else "?")
        out.append((name, node.lineno, purpose))
    return out


class EveryBuildSiteSaysWhoItBuildsForTests(unittest.TestCase):
    """The cell builders pass no purpose and get the gate; the evaluation callers pass ``evaluate`` and get none."""

    _EVALUATORS: dict[str, set[str]] = {
        "datagen/eval/ladder.py": {"build_calculator"},
        "datagen/rl/occupancy.py": {"build_calculator", "preflight_calculator"},
        "datagen/rl/collect.py": {"preflight_calculator"},
        "src/willy_sim/run_m1_pick.py": {"build_calculator"},
        "src/willy_sim/run_m2_pick.py": {"build_calculator"},
        "src/willy_sim/run_eih_pick.py": {"build_calculator"},
        "src/willy_sim/run_dense_pick.py": {"build_calculator"},
        "src/willy_sim/run_attribute_pick.py": {"build_calculator"},
        "src/willy_sim/run_multiview_pick.py": {"build_calculator"},
    }

    def test_the_cell_builders_pass_no_purpose(self) -> None:
        calls = _purposes(_ROOT / "src/robot/execution/autonomous_grasp/cells.py",
                          {"build_calculator", "preflight_calculator"})
        self.assertTrue(calls, "cells.py no longer builds a calculator; retire this test")
        self.assertEqual([call for call in calls if call[2] is not None], [])

    def test_every_evaluation_caller_builds_to_evaluate(self) -> None:
        for relative, names in self._EVALUATORS.items():
            with self.subTest(file=relative):
                calls = _purposes(_ROOT / relative, names)
                self.assertTrue(calls, f"{relative} no longer calls {names}; retire its entry")
                self.assertEqual([call for call in calls if call[2] != "evaluate"], [])

    def test_the_scan_sees_a_call_without_a_purpose(self) -> None:
        """The control: a scan that found nothing on any text would pass both tests above."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "site.py"
            path.write_text("build_calculator(cfg)\nbuild_calculator(cfg, purpose='evaluate')\n", encoding="utf-8")
            self.assertEqual(_purposes(path, {"build_calculator"}),
                             [("build_calculator", 1, None), ("build_calculator", 2, "evaluate")])


class TheRecordTests(_WithAFolder):
    """The reader itself: where the record lives, what it vouches for, and that a passed one reads as deployable."""

    def test_the_record_sits_beside_the_artifact_under_its_stem(self) -> None:
        from src.robot.grasping.deep.promotion import promotion_path

        self.assertEqual(Path("logs/m/set_grasp_generator_v1.promotion.json"),
                         promotion_path("logs/m/set_grasp_generator_v1.pt"))

    def test_a_passed_record_at_active_reads_as_deployable(self) -> None:
        from src.robot.grasping.deep.promotion import why_not_deployable

        artifact = _artifact(self.dir)
        promote(artifact)
        self.assertEqual("", why_not_deployable(artifact))

    def test_the_hash_is_over_the_artifacts_bytes(self) -> None:
        from src.robot.grasping.deep.promotion import artifact_sha256

        artifact = _artifact(self.dir)
        self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), artifact_sha256(artifact))

    def test_a_record_reads_back_as_it_was_written(self) -> None:
        from src.robot.grasping.deep.promotion import GeneratorPromotion, read_promotion

        artifact = _artifact(self.dir)
        promote(artifact)
        record = read_promotion(artifact)
        self.assertEqual(("pass", "active", "test_proof", 1), (record.verdict, record.phase, record.protocol,
                                                                record.protocol_version))
        self.assertEqual(record, GeneratorPromotion.from_dict(record.to_dict()))

    def test_no_artifact_is_a_reason_and_not_a_crash(self) -> None:
        from src.robot.grasping.deep.promotion import why_not_deployable

        self.assertIn("there is no artifact at", why_not_deployable(self.dir / "absent.pt"))

    def test_the_module_costs_no_torch(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-c", "import sys; import src.robot.grasping.deep.promotion; "
                                   "print('torch' in sys.modules)"],
            cwd=_ROOT, capture_output=True, text=True, timeout=300, check=True)
        self.assertEqual("False", result.stdout.strip())


class TheEvaluationPathsStillReadAnyArtifactTests(unittest.TestCase):
    def test_propose_reads_an_artifact_no_cell_may_use(self) -> None:
        """``deep propose`` loads the artifact itself, through the loader, which asks for no proof."""
        from src.robot.grasping.deep.eval.propose import propose_for_scenes
        from src.robot.grasping.deep.promotion import why_not_deployable
        from tests.test_deep_cli import _scene
        from tests.test_deep_set_calculator import _artifact as _real_artifact

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            artifact = _real_artifact(root)
            _scene(root / "scene_000.npz")
            self.assertIn("carries no promotion", why_not_deployable(artifact))
            proposals = propose_for_scenes(artifact, [root / "scene_000.npz"], top=2, seeds=8, device="cpu")
        self.assertTrue(proposals, "propose proposed nothing on an artifact no cell may use")


class InspectSaysTheVerdictFirstTests(unittest.TestCase):
    @staticmethod
    def _inspect(artifact: Path) -> tuple[int, list[str]]:
        from src.robot.grasping.deep import __main__ as cli

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(["inspect", "--artifact", str(artifact)])
        return code, buffer.getvalue().splitlines()

    def test_an_unpromoted_artifact_reads_not_deployable_first(self) -> None:
        from tests.test_deep_set_calculator import _artifact as _real_artifact

        with tempfile.TemporaryDirectory() as name:
            code, lines = self._inspect(_real_artifact(Path(name)))
        self.assertEqual(0, code)
        self.assertEqual("NOT DEPLOYABLE: set_grasp_generator_v1.pt carries no promotion", lines[0])
        self.assertIn("deep judge, coming", lines[1])
        self.assertIn("set_grasp_generator_v1.pt", lines[2], "the facts follow the verdict")

    def test_a_promoted_artifact_reads_deployable_with_its_phase_and_when(self) -> None:
        from tests.test_deep_set_calculator import _artifact as _real_artifact

        with tempfile.TemporaryDirectory() as name:
            artifact = _real_artifact(Path(name))
            promote(artifact)
            code, lines = self._inspect(artifact)
        self.assertEqual(0, code)
        self.assertEqual("deployable (phase active, promoted 2026-10-09T12:00:00Z)", lines[0])
        self.assertEqual("  promoted by tests under test_proof v1", lines[1])

    def test_a_smoke_artifact_says_so_first(self) -> None:
        from tests.test_deep_set_calculator import _artifact as _real_artifact

        with tempfile.TemporaryDirectory() as name:
            artifact = _real_artifact(Path(name))
            card = artifact.with_suffix(".card.json")
            payload = json.loads(card.read_text(encoding="utf-8"))
            payload["report"] = {"tier": "smoke"}
            card.write_text(json.dumps(payload), encoding="utf-8")
            promote(artifact)
            _code, lines = self._inspect(artifact)
        self.assertTrue(lines[0].startswith("NOT DEPLOYABLE: set_grasp_generator_v1.pt is a smoke-tier run"), lines[0])


class TrainSetEndsOnTheVerdictTests(unittest.TestCase):
    """``train-set`` keeps its exit code and its report, and says last that the weights it wrote are not a cell's yet."""

    @staticmethod
    def _report(raw: dict[str, Any]) -> Any:
        from src.robot.grasping.deep.train.report import TrainingRunReport
        from src.robot.grasping.deep.train.trainer import SetTrainingPlan

        return TrainingRunReport.from_trainer({"epochs": [{"fold": 0, "epoch": 0}], **raw}, SetTrainingPlan())

    def _train_set(self, report: Any) -> tuple[int, str]:
        from src.robot.grasping.deep import __main__ as cli

        run = mock.Mock(recipe_notes={})
        run.describe.return_value = "a stand-in run"
        run.train.return_value = report
        buffer = io.StringIO()
        with mock.patch("src.robot.grasping.deep.train.api.GeneratorTraining.from_recipe", return_value=run), \
                redirect_stdout(buffer):
            code = cli.main(["train-set", "--clouds", "unused", "--out", "unused"])
        return code, buffer.getvalue()

    def test_the_weights_of_a_fold_run_are_not_deployable_until_judged(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            weights = _artifact(Path(name))
            code, printed = self._train_set(self._report({"artifact": {"written": True, "weights": str(weights)}}))
        self.assertEqual(0, code)
        last = printed.strip().splitlines()[-1]
        self.assertIn("NOT DEPLOYABLE until judged: generator.pt carries no promotion", last)
        self.assertIn("deep judge, coming", last)

    def test_the_weights_of_a_refit_are_found_where_the_refit_keeps_them(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            weights = _artifact(Path(name))
            code, printed = self._train_set(self._report(
                {"refit": {"artifact": {"written": True, "weights": str(weights)}}}))
        self.assertEqual(0, code)
        self.assertIn("NOT DEPLOYABLE until judged: generator.pt", printed)

    def test_a_run_that_wrote_no_weights_says_nothing_of_a_cell(self) -> None:
        code, printed = self._train_set(self._report({"artifact": {"written": False, "reason": "no hand named"}}))
        self.assertEqual(0, code)
        self.assertNotIn("DEPLOYABLE", printed)


if __name__ == "__main__":
    unittest.main()
