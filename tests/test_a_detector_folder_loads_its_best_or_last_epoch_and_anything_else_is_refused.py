"""A detector folder loads its best epoch or its last, and a path that holds no detector is refused, saying why.

``ObjectDetector.from_weights`` takes what ``DetectorTraining`` leaves: the best epoch at the top of its out_dir and the
last in ``last/`` (``src/models/detection/closed_set/training/trainer.py``), each as ``save_pretrained`` writes it. Here
both are the real RT-DETR architecture at toy size (``tests/_detector_training_fixtures.py``), saved and loaded through
transformers on a CPU with nothing downloaded, the last epoch with a class the best one lacks, so which one loaded is
plain from its classes. Whatever holds no detector is refused before anything loads, in a sentence naming the folder
and what it lacks; an out_dir whose run kept only its last epoch names ``last/``. A Hugging Face id is read from the
folder the fetch script writes when that is there, and ``local=True`` keeps the read on this machine. ``from_config``
reads the two blocks a cell's tree names, and refuses the key that names no checkpoint.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from src.config import ConfigError, load_tree
from src.models.detection.object_detector import DEFAULT_SEGMENTER, ObjectDetector, _weights_source
from tests._detector_training_fixtures import tiny_factory
from tests._object_detector_fakes import CHESS, detector_folder, pin_cpu, rtdetr, sam2, sam2_folder

#: What undoes the CPU pin of this module, once ``setUpModule`` has set it.
_UNPIN: list = []


def setUpModule() -> None:  # noqa: N802 (unittest's name)
    _UNPIN.append(pin_cpu())


def tearDownModule() -> None:  # noqa: N802 (unittest's name)
    while _UNPIN:
        _UNPIN.pop()()


BEST = ("cube", "disc")
LAST = ("cube", "disc", "wedge")
#: One white pawn, for the stand-in RT-DETR of the config tests.
SCENE = [((10.0, 10.0, 30.0, 40.0), CHESS.index("white_pawn"), 0.95)]


def _save(folder: Path, classes: Sequence[str]) -> None:
    """The toy RT-DETR and its image processor, written as ``save_pretrained`` writes an exported epoch."""
    model, processor = tiny_factory(SimpleNamespace(image_size=64), dict(enumerate(classes)))
    model.save_pretrained(str(folder))
    processor.save_pretrained(str(folder))


class _Tmp(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


class ADetectorTrainingOutDirTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.out = Path(cls._tmp.name) / "run"
        _save(cls.out, BEST)
        _save(cls.out / "last", LAST)
        # What else a run leaves beside the best epoch; none of it is read here.
        (cls.out / "manifest.json").write_text("{}", encoding="utf-8")
        (cls.out / "checkpoint_last.pt").write_bytes(b"")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_out_dir_loads_its_best_epoch(self) -> None:
        detector = ObjectDetector.from_weights(self.out, threshold=0.0, device="cpu")
        self.assertEqual(BEST, detector.classes)
        self.assertEqual(self.out.resolve().as_posix(), detector.source)
        found = detector.detect(np.full((64, 96, 3), 120, dtype=np.uint8))
        self.assertTrue(found, "at a threshold of 0 every query of the model answers")
        self.assertTrue(set(found.labels) <= set(BEST), found.labels)
        for obj in found:
            x0, y0, x1, y1 = obj.box
            self.assertTrue(0.0 <= x0 < x1 <= 96.0 and 0.0 <= y0 < y1 <= 64.0, obj)

    def test_its_last_folder_loads_the_last_epoch(self) -> None:
        detector = ObjectDetector.from_weights(str(self.out / "last"))
        self.assertEqual(LAST, detector.classes)
        self.assertEqual(0.5, detector.threshold)


class WhatHoldsNoDetectorTests(_Tmp):

    def test_an_out_dir_that_kept_only_its_last_epoch_names_last(self) -> None:
        out = self.root / "diverged"
        detector_folder(out / "last")
        (out / "manifest.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(FileNotFoundError) as caught:
            ObjectDetector.from_weights(out)
        message = str(caught.exception)
        self.assertIn("holds no best epoch (no config.json)", message)
        self.assertIn(f"{out.resolve() / 'last'}: pass that folder to load it", message)

    def test_a_folder_with_no_detector_says_what_it_lacks(self) -> None:
        empty = self.root / "empty"
        empty.mkdir()
        cases = {
            "no config.json": empty,
            "a config.json and no weights": detector_folder(self.root / "no_weights", weights=False),
            "no preprocessor_config.json": detector_folder(self.root / "no_processor", processor=False),
        }
        for lack, folder in cases.items():
            with self.subTest(lack):
                with self.assertRaises(FileNotFoundError) as caught:
                    ObjectDetector.from_weights(folder)
                self.assertIn(f"{folder.resolve()} holds", str(caught.exception))
                self.assertIn(lack, str(caught.exception))
                self.assertIn("A DetectorTraining out_dir holds its best epoch at the top", str(caught.exception))

    def test_another_kind_of_checkpoint_is_no_detector_and_a_detector_is_no_sam2(self) -> None:
        with self.assertRaisesRegex(ValueError, "a Sam2VideoModel checkpoint, which is no object detector"):
            ObjectDetector.from_weights(sam2_folder(self.root / "sam2"))
        with self.assertRaisesRegex(ValueError, "a rt_detr checkpoint, which is no SAM2"):
            ObjectDetector.from_weights(detector_folder(self.root / "a"), segmenter=detector_folder(self.root / "b"))

    def test_a_path_that_is_not_there_is_refused_naming_where_it_looked(self) -> None:
        missing = self.root / "nothing"
        for value in (missing, str(missing), "models\\chess", "./chess", "D:/no/such/models"):
            with self.subTest(value=str(value)):
                with self.assertRaisesRegex(FileNotFoundError, "there is no folder at .* so there is no object "
                                                               "detector to load"):
                    ObjectDetector.from_weights(value)
        with self.assertRaises(FileNotFoundError) as caught:
            ObjectDetector.from_weights(missing)
        self.assertIn(str(missing.resolve()), str(caught.exception))
        file = self.root / "model.safetensors"
        file.write_bytes(b"")
        with self.assertRaisesRegex(FileNotFoundError, "is a file; name the folder"):
            ObjectDetector.from_weights(file)

    def test_no_weights_named_is_refused_naming_a_reports_model_dir(self) -> None:
        with self.assertRaisesRegex(TypeError, "model_dir is None when its run wrote no model"):
            ObjectDetector.from_weights(None)  # type: ignore[arg-type]

    def test_a_segmenter_that_is_not_there_is_refused_before_the_detector_loads(self) -> None:
        with rtdetr(SCENE) as fake:
            with self.assertRaisesRegex(FileNotFoundError, "so there is no SAM2 to load"):
                ObjectDetector.from_weights(detector_folder(self.root / "chess"), segmenter=self.root / "no_sam2")
        self.assertEqual(0, fake.loads.call_count)


class AHuggingFaceIdTests(_Tmp):

    def test_an_id_is_read_from_the_fetched_folder_when_it_is_there_and_else_from_the_cache_or_the_hub(self) -> None:
        with mock.patch.dict(os.environ, {"WILLY_PROJECT_ROOT": str(self.root)}):
            self.assertEqual(("PekingU/rtdetr_r50vd", False),
                             _weights_source("PekingU/rtdetr_r50vd", kind="detector", local=None))
            self.assertEqual(("PekingU/rtdetr_r50vd", True),
                             _weights_source("PekingU/rtdetr_r50vd", kind="detector", local=True))
            weights = self.root / "assets" / "models" / "hf"
            fetched = detector_folder(weights / "detection" / "PekingU--rtdetr_r50vd")
            source, local = _weights_source("PekingU/rtdetr_r50vd", kind="detector", local=False)
            self.assertEqual((fetched.resolve(), True), (Path(source).resolve(), local))
            sam = sam2_folder(weights / "segmentation" / DEFAULT_SEGMENTER.replace("/", "--"))
            source, local = _weights_source(DEFAULT_SEGMENTER, kind="segmenter", local=None)
            self.assertEqual((sam.resolve(), True), (Path(source).resolve(), local))

    def test_local_true_reads_this_machine_alone_and_otherwise_the_hub_may_answer(self) -> None:
        with mock.patch.dict(os.environ, {"WILLY_PROJECT_ROOT": str(self.root)}):
            for local, files_only in ((None, False), (False, False), (True, True)):
                with self.subTest(local=local), rtdetr(SCENE) as fake:
                    detector = ObjectDetector.from_weights("PekingU/rtdetr_r50vd", local=local)
                    self.assertEqual("PekingU/rtdetr_r50vd", fake.loads.call_args.args[0])
                    self.assertEqual(files_only, fake.loads.call_args.kwargs.get("local_files_only", False))
                    self.assertEqual("PekingU/rtdetr_r50vd", detector.source)
                    self.assertEqual(DEFAULT_SEGMENTER, detector.segmenter_source)

    def test_the_default_segmenter_is_the_one_the_shipped_tree_names(self) -> None:
        self.assertEqual(load_tree(None).app_config.models.segmenter.model_id, DEFAULT_SEGMENTER)


class FromConfigTests(_Tmp):
    """``backend="closed_set"`` builds RT-DETR from ``models.rtdetr`` whatever the pipeline says; a tree whose
    pipeline is the closed set builds it unnamed. The open backends are in
    ``test_an_object_detector_finds_what_a_prompt_names_on_every_backend.py``."""

    def test_it_reads_models_rtdetr_and_models_segmenter_from_a_tree_its_config_or_its_models(self) -> None:
        folder = detector_folder(self.root / "chess").as_posix()
        tree = load_tree(None).with_values({"models.rtdetr.model_path": folder, "models.rtdetr.threshold": 0.3})
        for config in (tree, tree.app_config, tree.app_config.models):
            with self.subTest(type(config).__name__), rtdetr(SCENE) as fake:
                detector = ObjectDetector.from_config(config, backend="closed_set")
                self.assertEqual(folder, fake.loads.call_args.args[0])
                self.assertTrue(fake.loads.call_args.kwargs["local_files_only"])
            self.assertEqual("closed_set", detector.backend)
            self.assertEqual(0.3, detector.threshold)
            self.assertEqual(CHESS, detector.classes)
            self.assertEqual(folder, detector.source)
            self.assertEqual(tree.app_config.models.segmenter.model_path, detector.segmenter_source)

    def test_a_tree_whose_pipeline_is_the_closed_set_builds_rtdetr_unnamed(self) -> None:
        folder = detector_folder(self.root / "chess").as_posix()
        tree = load_tree(None).with_values({"models.rtdetr.model_path": folder, "models.pipeline.kind": "closed_set"})
        self.assertTrue(tree.ok, tree.error)
        with rtdetr(SCENE) as fake:
            detector = ObjectDetector.from_config(tree)
        self.assertEqual(("closed_set", folder), (detector.backend, fake.loads.call_args.args[0]))

    def test_the_first_mask_refuses_a_segmenter_folder_that_is_not_there_naming_its_key(self) -> None:
        tree = load_tree(None).with_values({
            "models.rtdetr.model_path": detector_folder(self.root / "chess").as_posix(),
            "models.segmenter.model_path": (self.root / "no_sam2").as_posix(),
        })
        with rtdetr(SCENE):
            detector = ObjectDetector.from_config(tree, backend="closed_set")
        image = np.zeros((64, 96, 3), np.uint8)
        self.assertEqual(1, len(detector.detect(image)), "the detector works without SAM2")
        with sam2() as fake:
            with self.assertRaises(FileNotFoundError) as caught:
                detector.detect(image, segment=True)
        self.assertIn("models.segmenter.model_path is", str(caught.exception))
        self.assertIn("python scripts/model_weights/fetch.py sam2", str(caught.exception))
        self.assertEqual(0, fake.loads.call_count)

    def test_a_local_rtdetr_folder_that_is_not_there_is_refused_naming_the_key_and_the_fetch(self) -> None:
        tree = load_tree(None).with_values({"models.rtdetr.model_path": (self.root / "nothing").as_posix()})
        with rtdetr(SCENE) as fake:
            with self.assertRaises(FileNotFoundError) as caught:
                ObjectDetector.from_config(tree, backend="closed_set")
        message = str(caught.exception)
        for part in ("models.rtdetr.model_path is", "there is no folder there", "fetch.py rtdetr", "DetectorTraining"):
            self.assertIn(part, message)
        self.assertEqual(0, fake.loads.call_count)

    def test_a_tree_with_no_rtdetr_block_is_refused(self) -> None:
        tree = load_tree(None).with_values({"models.rtdetr": None})
        self.assertTrue(tree.ok, tree.error)
        with self.assertRaisesRegex(ValueError, "models.rtdetr is not set, and the closed-set detector is RT-DETR"):
            ObjectDetector.from_config(tree, backend="closed_set")

    def test_a_tree_that_did_not_load_raises_its_refusal_and_anything_else_is_no_config(self) -> None:
        with self.assertRaises(ConfigError):
            ObjectDetector.from_config(load_tree("no_such_layer"))
        with self.assertRaisesRegex(TypeError, "takes a loaded tree, its AppConfig or its models block"):
            ObjectDetector.from_config(42)


if __name__ == "__main__":
    unittest.main()
