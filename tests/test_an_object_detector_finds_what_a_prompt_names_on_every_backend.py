"""Every detection model a cell can run answers through one door: a prompt or classes in, the objects found out.

The owner, 2026-10-09: "alle unter einem Dach". ``ObjectDetector`` carries the closed-set RT-DETR and the VLM, the two
detectors a cell's ``models.pipeline`` can name since GroundingDINO and its router left (2026-10-17), and answers each
with the same ``Detections``. An open-vocabulary call names exactly one of a prompt, every box found coming back under
it, or classes, several descriptions found in one call as the cell finds them, each box under the description the
model gave it. The models here are stand-ins that answer what a test chose, so nothing loads a weight;
``from_config`` is held to the tree's blocks by replacing the builders it calls.
"""

from __future__ import annotations

import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from src.config import load_tree
from src.config.schema.models.models_schema import SegmenterConfig
from src.models import weights as weights_module
from src.models.detection import object_detector as od
from src.models.detection.object_detector import BACKENDS, RECORDED_BACKENDS, Detections, ObjectDetector
from src.models.detection.types import Detection
from src.models.vlm.parsing import AMBIGUOUS_LABEL, VLM_NOMINAL_SCORE
from tests._object_detector_fakes import detector_folder, pin_cpu, sam2, sam2_folder

#: What undoes the CPU pin of this module, once ``setUpModule`` has set it.
_UNPIN: list = []


def setUpModule() -> None:  # noqa: N802 (unittest's name)
    _UNPIN.append(pin_cpu())


def tearDownModule() -> None:  # noqa: N802 (unittest's name)
    while _UNPIN:
        _UNPIN.pop()()


HEIGHT, WIDTH = 120, 160
RED = (10.0, 10.0, 40.0, 40.0)
BLUE = (60.0, 20.0, 120.0, 90.0)
STRAY = (130.0, 95.0, 150.0, 115.0)


def _image() -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)


def _found(box: tuple[float, ...], label: str, score: float) -> Detection:
    x0, y0, x1, y1 = box
    return Detection(box=[x0, y0, x1, y1], x_center=(x0 + x1) / 2, y_center=(y0 + y1) / 2, label=label, score=score)


class Vlm:
    """The VLM's verb: boxes with no score of their own, as a VLM writes them, and every call kept."""

    def __init__(self, answer: list[tuple[tuple[float, ...], str]]) -> None:
        self.answer = answer
        self.device = "cpu"
        self.calls: list[str] = []

    def detect_all(self, image: np.ndarray, prompt: str) -> list[Detection]:
        self.calls.append(prompt)
        return [_found(box, label, VLM_NOMINAL_SCORE) for box, label in self.answer]


class APromptOnTheVlmTests(unittest.TestCase):

    def setUp(self) -> None:
        self.vlm = Vlm([(RED, "cube"), (BLUE, "red cube")])
        self.detector = ObjectDetector(self.vlm, backend="vlm", source="Qwen/Qwen3-VL-8B-Instruct")

    def test_every_box_found_for_a_prompt_comes_back_under_the_prompt(self) -> None:
        found = self.detector.detect(_image(), prompt="  the red cube ")
        self.assertEqual(("the red cube", "the red cube"), found.labels)  # whatever words the model wrote
        self.assertEqual(("the red cube",), found.classes)
        self.assertEqual({"the red cube": 2}, found.counts)
        self.assertEqual(2, len(found.of("The Red Cube")))
        self.assertEqual(("vlm", "the red cube", None), (found.backend, found.prompt, found.threshold))
        self.assertEqual(["the red cube"], self.vlm.calls)

    def test_it_is_open_vocabulary_and_knows_no_fixed_class_list(self) -> None:
        self.assertEqual(("vlm", True, ()), (self.detector.backend, self.detector.open_vocabulary,
                                             self.detector.classes))
        self.assertIn("the VLM from Qwen/Qwen3-VL-8B-Instruct", self.detector.render())


class ClassesOnTheVlmTests(unittest.TestCase):

    def setUp(self) -> None:
        self.vlm = Vlm([(RED, "red cube"), (BLUE, "blue bin"), (STRAY, AMBIGUOUS_LABEL)])
        self.detector = ObjectDetector(self.vlm, backend="vlm")

    def test_several_descriptions_are_found_in_one_call_each_box_under_its_own(self) -> None:
        found = self.detector.detect(_image(), classes=["red cube", "blue bin"])
        self.assertEqual(["red cube | blue bin"], self.vlm.calls, "the cell's class list, one call")
        self.assertEqual(("red cube", "blue bin"), found.classes)
        self.assertEqual(("red cube", "blue bin"), found.labels)
        self.assertEqual({"red cube": 1, "blue bin": 1}, found.counts)
        self.assertEqual("red cube | blue bin", found.prompt)

    def test_a_box_no_description_claims_is_neither_and_is_left_out(self) -> None:
        found = self.detector.detect(_image(), classes=["red cube", "blue bin"])
        self.assertNotIn(AMBIGUOUS_LABEL, found.labels)
        self.assertEqual(2, len(found))

    def test_one_class_is_a_prompt_and_a_prompt_holding_a_class_list_is_classes(self) -> None:
        one = self.detector.detect(_image(), classes="red cube")
        self.assertEqual(("red cube",), one.classes)
        self.assertEqual({"red cube"}, set(one.labels), "every box found for one thing is that thing")
        listed = self.detector.detect(_image(), prompt="red cube | blue bin")
        self.assertEqual(("red cube", "blue bin"), listed.classes)
        self.assertEqual(("red cube", "blue bin"), listed.labels)

    def test_exactly_one_of_a_prompt_and_classes_and_neither_empty(self) -> None:
        with self.assertRaisesRegex(ValueError, "knows no fixed class list: give a prompt"):
            self.detector.detect(_image())
        with self.assertRaisesRegex(ValueError, "a prompt or classes, not both"):
            self.detector.detect(_image(), prompt="a cube", classes=["a cube"])
        with self.assertRaisesRegex(ValueError, "names nothing"):
            self.detector.detect(_image(), prompt="   ")
        with self.assertRaisesRegex(ValueError, "separates the descriptions"):
            self.detector.detect(_image(), classes=["red | cube", "bin"])
        with self.assertRaises(ValueError):
            self.detector.detect(_image(), classes=[])
        self.assertEqual([], self.vlm.calls, "a refused call asks the model nothing")


class TheClosedSetReadsNoTextTests(unittest.TestCase):

    def test_a_prompt_is_refused_pointing_at_classes(self) -> None:
        rtdetr = SimpleNamespace(classes=("pawn", "king"), threshold=0.5, device="cpu", source="chess",
                                 detect_all=mock.Mock(return_value=[]))
        detector = ObjectDetector(rtdetr)
        with self.assertRaisesRegex(ValueError, "reads no text: name the classes you want with classes="):
            detector.detect(_image(), prompt="the king")
        rtdetr.detect_all.assert_not_called()
        self.assertEqual(("closed_set", False, ("pawn", "king")),
                         (detector.backend, detector.open_vocabulary, detector.classes))


class TheVlmGivesNoScoreTests(unittest.TestCase):

    def setUp(self) -> None:
        self.vlm = Vlm([(RED, "the largest cube")])
        self.detector = ObjectDetector(self.vlm, backend="vlm", source="Qwen/Qwen3-VL-8B-Instruct")

    def test_its_boxes_come_back_with_no_threshold_and_say_so(self) -> None:
        found = self.detector.detect(_image(), prompt="the largest cube")
        self.assertEqual(("vlm", None), (found.backend, found.threshold))
        self.assertEqual(("the largest cube",), found.labels)
        self.assertIn("(the model gives no score)", found.render())
        self.assertIsNone(self.detector.threshold)
        self.assertIn("which gives no score", self.detector.render())

    def test_a_threshold_is_refused_and_asks_nothing(self) -> None:
        with self.assertRaisesRegex(ValueError, "without a score, so a threshold has nothing to cut"):
            self.detector.detect(_image(), prompt="a cube", threshold=0.5)
        self.assertEqual([], self.vlm.calls)


class TheBackendsThatLeftTests(unittest.TestCase):
    """GroundingDINO (``grounded_sam``) and the router left on 2026-10-17: no detector answers as either."""

    def test_the_backends_are_the_closed_set_and_the_vlm(self) -> None:
        self.assertEqual(("closed_set", "vlm"), BACKENDS)
        for gone in ("grounded_sam", "router", "dino"):
            with self.subTest(gone), self.assertRaisesRegex(ValueError, "backend is one of closed_set, vlm"):
                ObjectDetector(Vlm([]), backend=gone)

    def test_a_detection_saved_before_reads_back_as_it_was_written(self) -> None:
        self.assertEqual(("grounded_sam", "router"), RECORDED_BACKENDS)
        for backend in RECORDED_BACKENDS:
            with self.subTest(backend):
                read = Detections.from_dict({"objects": [], "classes": ["a cube"], "threshold": 0.4,
                                             "image_hw": [2, 2], "backend": backend, "prompt": "a cube"})
                self.assertEqual(backend, read.backend)


class MasksOnTheVlmTests(unittest.TestCase):

    def test_segment_true_cuts_every_box_the_vlm_found_in_one_sam2_pass(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            folder = sam2_folder(Path(tmp) / "sam2").as_posix()
            config = SegmenterConfig(model_id=folder)
            detector = ObjectDetector(Vlm([(RED, "red cube"), (BLUE, "blue bin")]), backend="vlm",
                                      load_segmenter=partial(od._sam2, config, device="cpu"))
            with sam2() as fake:
                found = detector.detect(_image(), classes=["red cube", "blue bin"], segment=True)
        self.assertEqual(1, len(fake.model.calls), "one SAM2 pass for both boxes")
        self.assertEqual(2, len(fake.processor.calls[0]["input_boxes"][0]))
        for obj in found:
            self.assertIsNotNone(obj.mask)
            x0, y0, x1, y1 = (int(value) for value in obj.box)
            inside = int(obj.mask[y0:y1, x0:x1].sum())
            self.assertEqual(inside, int(obj.mask.sum()), "the mask SAM2 cut for a box lies in that box")
            self.assertGreater(inside, 0.9 * (x1 - x0) * (y1 - y0), "and fills it, but for the cleanup's corners")


class FromTheTreeTests(unittest.TestCase):
    """What ``from_config`` builds, and from which block, with the VLM's builder replaced."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "weights"
        self.built: list[tuple[Any, Any]] = []
        self.vlm = Vlm([(BLUE, "a cube")])
        patcher = mock.patch.object(od, "_vlm_grounder", side_effect=self._build)
        patcher.start()
        self.addCleanup(patcher.stop)
        # The Hugging Face cache of the machine running the tests is no part of the tree.
        cache = mock.patch.object(weights_module, "_in_the_cache", return_value=False)
        cache.start()
        self.addCleanup(cache.stop)

    def _build(self, block: Any, root: Any) -> Any:
        self.built.append((block, root))
        return self.vlm

    def tree(self, values: dict[str, Any]) -> Any:
        tree = load_tree(None).with_values({"models.weights_root": self.root.as_posix(), **values})
        self.assertTrue(tree.ok, tree.error)
        return tree

    def test_the_backend_is_what_the_trees_pipeline_builds_for_a_cell(self) -> None:
        self.assertEqual("vlm", od._tree_backend(self.tree({}).app_config.models))
        self.assertEqual("closed_set",
                         od._tree_backend(self.tree({"models.pipeline.kind": "closed_set"}).app_config.models))
        self.assertEqual("vlm", od._tree_backend(SimpleNamespace(pipeline=None)))

    def test_the_vlm_is_built_from_its_pipeline_block_under_the_trees_weights_root(self) -> None:
        tree = self.tree({})
        detector = ObjectDetector.from_config(tree)
        self.assertEqual("vlm", detector.backend)
        self.assertEqual([(tree.app_config.models.pipeline.zero_shot.vlm, self.root.as_posix())], self.built)
        self.assertIn(f"{self.root.as_posix()}/segmentation/facebook--sam2-hiera-large", detector.segmenter_source)

    def test_the_closed_set_is_built_from_models_rtdetr_read_from_the_weights_root(self) -> None:
        detector_folder(self.root / "detection" / "PekingU--rtdetr_r50vd")
        tree = self.tree({"models.pipeline.kind": "closed_set"})
        rtdetr = SimpleNamespace(classes=("person",), threshold=0.4, device="cpu")
        with mock.patch.object(od, "_rtdetr", return_value=rtdetr) as built:
            detector = ObjectDetector.from_config(tree)
        self.assertEqual("closed_set", detector.backend)
        block = built.call_args.args[0]
        self.assertEqual("PekingU/rtdetr_r50vd", block.model_id)
        self.assertEqual(self.root.as_posix(), built.call_args.kwargs["weights_root"])
        self.assertEqual(f"{self.root.as_posix()}/detection/PekingU--rtdetr_r50vd", detector.source)
        self.assertEqual([], self.built, "the VLM is not built for a closed-set cell")

    def test_a_named_backend_is_built_whatever_the_pipeline_says(self) -> None:
        tree = self.tree({"models.pipeline.kind": "closed_set"})
        self.assertEqual("vlm", ObjectDetector.from_config(tree, backend="vlm").backend)
        self.assertEqual("vlm", ObjectDetector.from_config(tree.app_config.models, backend="vlm").backend)
        with self.assertRaisesRegex(ValueError, "backend is one of"):
            ObjectDetector.from_config(tree, backend="grounded_sam")

    def test_a_missing_closed_set_detector_is_refused_naming_its_key_and_fetch(self) -> None:
        tree = self.tree({"models.pipeline.kind": "closed_set"})
        with self.assertRaises(FileNotFoundError) as caught:
            ObjectDetector.from_config(tree)
        message = str(caught.exception)
        for part in ("models.rtdetr.model_id is 'PekingU/rtdetr_r50vd'", "detection/PekingU--rtdetr_r50vd",
                     "fetch.py rtdetr", "DetectorTraining out_dir"):
            self.assertIn(part, message)
        models = tree.app_config.models.model_copy(update={"rtdetr": None})
        with self.assertRaisesRegex(ValueError, "models.rtdetr is not set"):
            ObjectDetector.from_config(models, backend="closed_set")


class WhatADetectionsSaysTests(unittest.TestCase):

    def test_the_backend_and_the_prompt_go_into_its_data_and_come_back(self) -> None:
        found = ObjectDetector(Vlm([(RED, "cube")]), backend="vlm").detect(_image(), prompt="a cube")
        data = found.to_dict()
        self.assertEqual(("vlm", "a cube"), (data["backend"], data["prompt"]))
        self.assertEqual(found, Detections.from_dict(data))
        old = {key: value for key, value in data.items() if key not in ("backend", "prompt")}
        self.assertEqual(("closed_set", None), (Detections.from_dict(old).backend, Detections.from_dict(old).prompt))
        self.assertIn("asked      vlm for 'a cube'", found.render())

    def test_a_backend_it_does_not_know_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "Detections.backend is one of"):
            Detections(objects=(), classes=(), threshold=None, image_hw=(2, 2), backend="yolo")


if __name__ == "__main__":
    unittest.main()
