"""A detector training run end to end on a CPU, with a tiny RT-DETR: what lands in out_dir, the best and the last epoch,
early stopping, resuming, divergence, and the ``DetectorTraining`` face over it."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src.models.detection.closed_set.training import trainer
from src.models.detection.closed_set.training.api import DetectorTraining
from src.models.detection.closed_set.training.datasets import load_dataset
from src.models.detection.closed_set.training.metrics import DetectionScores
from src.models.detection.closed_set.training.plan import DetectorPlan, DetectorPlanOverrides
from src.models.detection.closed_set.training.report import DetectorTrainingOutcome, DetectorTrainingReport
from tests._detector_training_fixtures import CLASSES, tiny_factory, write_coco

TINY = DetectorPlan(image_size=64, epochs=3, batch=4, multiscale=False, patience=0, ema_warmup=10, seed=1)


def _scores(value: float) -> DetectionScores:
    return DetectionScores(map=value, map50=value, map75=value, mar100=value, per_class_ap={"cube": value})


class ARunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._tmp.name)
        write_coco(cls.root / "data", 14)
        cls.dataset = load_dataset(cls.root / "data")
        cls.out = cls.root / "run"
        cls.raw = trainer.train_detector(cls.dataset, TINY, out_dir=cls.out, device="cpu", model_factory=tiny_factory)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_best_epoch_the_last_epoch_and_their_files(self) -> None:
        for name in ("config.json", "model.safetensors", "preprocessor_config.json", "checkpoint_last.pt",
                     "results.csv", "manifest.json", "last/model.safetensors", "last/config.json"):
            self.assertTrue((self.out / name).is_file(), name)
        self.assertTrue(self.raw["artifact"]["written"])
        maps = [row["map"] for row in self.raw["epochs"]]
        self.assertEqual(3, len(maps))
        self.assertEqual(1 + int(np.argmax(maps)), self.raw["best_epoch"])

    def test_results_csv_has_a_row_per_epoch_and_a_column_per_class(self) -> None:
        with (self.out / "results.csv").open(encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(["epoch", "seconds", "lr", "train_loss", "val_loss", "map", "map50", "map75", "mar100",
                          "ap_cube", "ap_disc"], rows[0])
        self.assertEqual(["1", "2", "3"], [row[0] for row in rows[1:]])

    def test_the_manifest_names_the_dataset_the_classes_and_the_plan(self) -> None:
        manifest = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual("willy.rtdetr.train_manifest/2", manifest["schema"])
        self.assertEqual({"0": "cube", "1": "disc"}, manifest["id2label"])
        self.assertEqual(self.dataset.fingerprint, manifest["dataset"]["fingerprint"])
        self.assertEqual("split", manifest["dataset"]["val_source"])
        self.assertEqual(64, manifest["plan"]["image_size"])
        self.assertEqual(64, len(manifest["weights_sha256"]))
        self.assertEqual(self.raw["best_epoch"], manifest["best_epoch"])

    def test_the_model_loads_in_the_detector_with_the_dataset_s_classes(self) -> None:
        from src.config.schema.models.models_schema import ObjectDetectorConfig
        from src.models.detection.closed_set.detector import RtDetrObjectDetector

        detector = RtDetrObjectDetector(ObjectDetectorConfig(model_path=str(self.out), threshold=0.0, local=True))
        image = np.zeros((64, 96, 3), dtype=np.uint8)
        found = detector.detect_all(image)
        self.assertTrue(found)
        self.assertTrue({d.label for d in found} <= set(CLASSES))

    def test_a_resume_continues_and_a_finished_run_has_nothing_to_resume(self) -> None:
        out = self.root / "resume"
        trainer.train_detector(self.dataset, DetectorPlan(**{**TINY.as_dict(), "epochs": 2}), out_dir=out,
                               device="cpu", model_factory=tiny_factory)
        raw = trainer.train_detector(self.dataset, TINY, out_dir=out, device="cpu", resume=True,
                                     model_factory=tiny_factory)
        self.assertEqual((2, [1, 2, 3]), (raw["resumed_from_epoch"], [row["epoch"] for row in raw["epochs"]]))
        again = trainer.train_detector(self.dataset, TINY, out_dir=out, device="cpu", resume=True,
                                       model_factory=tiny_factory)
        self.assertTrue(again["nothing_to_resume"])
        report = DetectorTrainingReport.from_trainer(again, TINY)
        self.assertEqual(DetectorTrainingOutcome.NOTHING_TO_RESUME, report.outcome)
        self.assertIn("raise epochs", report.failure_summary())

    def test_a_resume_against_another_dataset_is_refused(self) -> None:
        other = self.root / "other"
        write_coco(other, 8, start=40)
        with self.assertRaisesRegex(ValueError, "another dataset or other classes"):
            trainer.train_detector(load_dataset(other), TINY, out_dir=self.out, device="cpu", resume=True,
                                   model_factory=tiny_factory)


class StoppingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name)
        write_coco(self.root / "data", 14)
        self.dataset = load_dataset(self.root / "data")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_patience_stops_a_run_that_no_longer_improves_and_keeps_the_best(self) -> None:
        maps = iter([0.2, 0.5, 0.5, 0.4, 0.6, 0.7])
        with mock.patch.object(trainer, "_evaluate", side_effect=lambda *a, **k: (_scores(next(maps)), 1.0)):
            raw = trainer.train_detector(self.dataset, DetectorPlan(**{**TINY.as_dict(), "epochs": 6, "patience": 2}),
                                         out_dir=self.root / "out", device="cpu", model_factory=tiny_factory)
        self.assertEqual((True, 2, 4), (raw["stopped_early"], raw["best_epoch"], len(raw["epochs"])))
        report = DetectorTrainingReport.from_trainer(raw, DetectorPlan(epochs=6, patience=2))
        self.assertEqual(DetectorTrainingOutcome.STOPPED_EARLY, report.outcome)
        self.assertTrue(report.succeeded)
        self.assertIn("stopped early", report.render())

    def test_a_loss_that_stays_non_finite_is_called_diverged(self) -> None:
        def nan_factory(plan, id2label):
            model, processor = tiny_factory(plan, id2label)
            forward = model.forward

            def poisoned(*args, **kwargs):
                out = forward(*args, **kwargs)
                if out.loss is not None:
                    out.loss = out.loss * float("nan")
                return out

            model.forward = poisoned
            return model, processor

        raw = trainer.train_detector(self.dataset, DetectorPlan(**{**TINY.as_dict(), "batch": 1, "epochs": 1}),
                                     out_dir=self.root / "out", device="cpu", model_factory=nan_factory)
        self.assertTrue(raw["diverged"])
        report = DetectorTrainingReport.from_trainer(raw, TINY)
        self.assertEqual(DetectorTrainingOutcome.DIVERGED, report.outcome)
        self.assertFalse(report.succeeded)
        self.assertIn("non-finite", report.failure_summary())

    def test_without_validation_the_training_loss_picks_the_best(self) -> None:
        dataset = load_dataset(self.root / "data", val_fraction=0.0)
        raw = trainer.train_detector(dataset, DetectorPlan(**{**TINY.as_dict(), "epochs": 2}),
                                     out_dir=self.root / "out", device="cpu", model_factory=tiny_factory)
        report = DetectorTrainingReport.from_trainer(raw, TINY)
        self.assertTrue(report.succeeded)
        self.assertFalse(report.validated)
        self.assertIn("no validation box", report.render())

    def test_multiscale_and_the_strong_augmentations_run(self) -> None:
        plan = DetectorPlan(**{**TINY.as_dict(), "epochs": 2, "multiscale": True})
        raw = trainer.train_detector(self.dataset, plan, out_dir=self.root / "out", device="cpu",
                                     model_factory=tiny_factory)
        self.assertEqual([True, False], [row["strong_augment"] for row in raw["epochs"]])


class PiecesTests(unittest.TestCase):
    def test_the_backbone_learns_at_a_tenth_and_norms_and_biases_carry_no_decay(self) -> None:
        model, _ = tiny_factory(TINY, {0: "cube", 1: "disc"})
        groups = {group["name"]: group for group in trainer.param_groups(model, TINY)}
        self.assertLessEqual({"backbone", "head", "head/no_decay"}, set(groups))
        self.assertAlmostEqual(1e-5, groups["backbone"]["lr"])
        self.assertAlmostEqual(1e-4, groups["head"]["lr"])
        self.assertEqual(0.0, groups["head/no_decay"]["weight_decay"])
        self.assertEqual(1e-4, groups["head"]["weight_decay"])
        counted = sum(len(group["params"]) for group in groups.values())
        self.assertEqual(sum(1 for p in model.parameters() if p.requires_grad), counted)

    def test_warm_up_then_cosine_to_one_percent(self) -> None:
        self.assertAlmostEqual(0.1, trainer.lr_factor(0, 10, 100))
        self.assertAlmostEqual(1.0, trainer.lr_factor(9, 10, 100))
        self.assertAlmostEqual(1.0, trainer.lr_factor(10, 10, 100))
        self.assertAlmostEqual(0.505, trainer.lr_factor(55, 10, 100))
        self.assertAlmostEqual(0.01, trainer.lr_factor(100, 10, 100))
        self.assertEqual(100, trainer.warmup_steps(10, 1000))
        self.assertEqual(2000, trainer.warmup_steps(5000, 100000))
        self.assertEqual(3, trainer.warmup_steps(8, 16))

    def test_the_average_follows_the_model_with_its_warm_up(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.0)
        ema = trainer.ModelEMA(model, decay=0.9, warmup=1)
        with torch.no_grad():
            model.weight.fill_(1.0)
        ema.update(model)
        d = 0.9 * (1 - math.exp(-1))
        self.assertAlmostEqual(1 - d, float(ema.module.weight), places=6)
        self.assertFalse(ema.module.weight.requires_grad)

    def test_loader_workers_are_chosen_on_windows_as_anywhere(self) -> None:
        """They start without running the training script again (``loading.WorkerBatches``), so Windows, where a
        spawned worker used to, needs no exception any more (the owner, 2026-10-10)."""
        cuda = torch.device("cuda")
        for platform in ("win32", "linux"):
            with mock.patch.object(trainer.sys, "platform", platform),                     mock.patch.object(trainer.os, "cpu_count", return_value=16):
                self.assertEqual(4, trainer._workers(TINY, 5000, cuda))
                self.assertEqual(3, trainer._workers(DetectorPlan(workers=3), 5000, cuda))
                self.assertEqual(0, trainer._workers(TINY, 50, cuda))
                self.assertEqual(0, trainer._workers(TINY, 5000, torch.device("cpu")))


class TrainingFaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name)
        write_coco(self.root / "data", 12)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_from_a_dataset_to_a_written_report(self) -> None:
        run = DetectorTraining.from_dataset(
            dataset=self.root / "data", tier="smoke", out_dir=self.root / "model", device="cpu",
            overrides=DetectorPlanOverrides(image_size=64, batch=4))
        text = run.describe()
        self.assertIn("SMOKE TIER", text)
        text.encode("ascii")
        probe = run.probe()
        self.assertEqual(CLASSES, probe.classes)
        self.assertEqual(probe.render(), str(probe))
        rows: list[dict] = []
        run.attach_progress_listener(rows.append)
        with mock.patch.object(trainer, "pretrained_factory", tiny_factory):
            report = run.train()
        self.assertTrue(report.succeeded)
        self.assertEqual(2, len(rows))
        self.assertEqual(str(self.root / "model"), report.model_dir)
        report.render().encode("ascii")
        written = json.loads(run.write_report(report).read_text(encoding="utf-8"))
        self.assertEqual(("completed", "v1", "smoke"), (written["outcome"], written["recipe"], written["tier"]))

    def test_refusals_before_anything_runs(self) -> None:
        with self.assertRaises(FileNotFoundError):
            DetectorTraining.from_dataset(dataset=self.root / "missing")
        with self.assertRaisesRegex(ValueError, "unknown training tier"):
            DetectorTraining.from_dataset(dataset=self.root / "data", tier="huge")
        run = DetectorTraining.from_dataset(dataset=self.root / "data")
        self.assertIn("NOT SET", run.describe())
        with self.assertRaisesRegex(ValueError, "no out_dir"):
            run.train()
        report = DetectorTrainingReport.from_trainer({"epochs": []}, DetectorPlan())
        with self.assertRaisesRegex(ValueError, "no out_dir"):
            run.write_report(report)

    def test_the_probe_warns_about_thin_classes_and_missing_validation(self) -> None:
        probe = DetectorTraining.from_dataset(dataset=self.root / "data",
                                              overrides=DetectorPlanOverrides(val_fraction=0.0)).probe()
        text = " | ".join(probe.warnings)
        self.assertIn("training box(es)", text)
        self.assertIn("no validation images", text)


if __name__ == "__main__":
    unittest.main()
