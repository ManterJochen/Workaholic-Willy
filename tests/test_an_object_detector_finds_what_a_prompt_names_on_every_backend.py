"""Every detection model a cell can run answers through one door: a prompt or classes in, the objects found out.

The owner, 2026-10-09: "alle unter einem Dach". ``ObjectDetector`` carries the closed-set RT-DETR and the three
open-vocabulary backends a cell's ``models.pipeline`` can name, GroundingDINO (``grounded_sam``), the VLM (``vlm``) and
the router between those two (``router``), and answers each with the same ``Detections``. An open-vocabulary call
names exactly one of a prompt, every box found coming back under it, or classes, several descriptions found in one
call as the cell finds them, each box under the description the model gave it. The models here are stand-ins that
answer what a test chose, so nothing loads a weight; ``from_config`` is held to the tree's blocks by replacing the
three builders it calls.
"""

from __future__ import annotations

import os
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
from src.models.detection import object_detector as od
from src.models.detection.object_detector import BACKENDS, Detections, ObjectDetector
from src.models.detection.types import Detection
from src.models.routing import RuleBasedRouter
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


class Grounder:
    """GroundingDINO's verb: the boxes a test chose, each above the threshold of the call, and every call kept."""

    def __init__(self, answer: list[tuple[tuple[float, ...], str, float]], threshold: float = 0.4) -> None:
        self.answer = answer
        self.threshold = threshold
        self.device = "cpu"
        self.model_path = "IDEA-Research/grounding-dino-tiny"
        self.calls: list[tuple[str, float]] = []

    def detect_all(self, image: np.ndarray, prompt: str, *, threshold: float | None = None) -> list[Detection]:
        cut = self.threshold if threshold is None else threshold
        self.calls.append((prompt, cut))
        return [_found(box, label, score) for box, label, score in self.answer if score > cut]


class Vlm:
    """The VLM's verb: boxes with no score of their own, as a VLM writes them, and every call kept."""

    def __init__(self, answer: list[tuple[tuple[float, ...], str]]) -> None:
        self.answer = answer
        self.device = "cpu"
        self.calls: list[str] = []

    def detect_all(self, image: np.ndarray, prompt: str) -> list[Detection]:
        self.calls.append(prompt)
        return [_found(box, label, VLM_NOMINAL_SCORE) for box, label in self.answer]


class APromptOnGroundingDinoTests(unittest.TestCase):

    def setUp(self) -> None:
        self.grounder = Grounder([(RED, "cube", 0.8), (BLUE, "red cube", 0.5), (STRAY, "cube", 0.3)])
        self.detector = ObjectDetector(self.grounder, backend="grounded_sam")

    def test_every_box_found_for_a_prompt_comes_back_under_the_prompt(self) -> None:
        found = self.detector.detect(_image(), prompt="  the red cube ")
        self.assertEqual(("the red cube", "the red cube"), found.labels)  # whatever phrase the model grounded
        self.assertEqual(("the red cube",), found.classes)
        self.assertEqual({"the red cube": 2}, found.counts)
        self.assertEqual(2, len(found.of("The Red Cube")))
        self.assertEqual(("grounded_sam", "the red cube", 0.4), (found.backend, found.prompt, found.threshold))
        self.assertEqual([("the red cube", 0.4)], self.grounder.calls)

    def test_a_threshold_for_one_call_reaches_the_model(self) -> None:
        found = self.detector.detect(_image(), prompt="a cube", threshold=0.6)
        self.assertEqual(1, len(found))
        self.assertEqual(0.6, found.threshold)
        self.assertEqual([("a cube", 0.6)], self.grounder.calls)
        self.assertEqual(0.4, self.detector.threshold, "the detector's own threshold is left as it was")

    def test_it_is_open_vocabulary_and_knows_no_fixed_class_list(self) -> None:
        self.assertEqual(("grounded_sam", True, ()), (self.detector.backend, self.detector.open_vocabulary,
                                                      self.detector.classes))
        self.assertIn("GroundingDINO from IDEA-Research/grounding-dino-tiny", self.detector.render())


class ClassesOnAnOpenBackendTests(unittest.TestCase):

    def setUp(self) -> None:
        self.grounder = Grounder([(RED, "red cube", 0.8), (BLUE, "blue bin", 0.7), (STRAY, AMBIGUOUS_LABEL, 0.9)])
        self.detector = ObjectDetector(self.grounder, backend="grounded_sam")

    def test_several_descriptions_are_found_in_one_call_each_box_under_its_own(self) -> None:
        found = self.detector.detect(_image(), classes=["red cube", "blue bin"])
        self.assertEqual([("red cube | blue bin", 0.4)], self.grounder.calls, "the cell's class list, one call")
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
        self.assertEqual([], self.grounder.calls, "a refused call asks the model nothing")


class TheClosedSetReadsNoTextTests(unittest.TestCase):

    def test_a_prompt_is_refused_pointing_at_classes(self) -> None:
        rtdetr = SimpleNamespace(classes=("pawn", "king"), threshold=0.5, device="cpu", model_path="chess",
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


class TheRouterTests(unittest.TestCase):

    def setUp(self) -> None:
        self.grounder = Grounder([(RED, "cube", 0.8)])
        self.vlm = Vlm([(BLUE, "kaputter Wuerfel")])
        self.loads = mock.Mock(return_value=self.vlm)
        self.detector = ObjectDetector(self.grounder, backend="router", load_vlm=self.loads,
                                       router=RuleBasedRouter(), vlm_source="Qwen/Qwen3-VL-8B-Instruct")

    def test_a_plain_prompt_goes_to_grounding_dino_and_loads_no_vlm(self) -> None:
        found = self.detector.detect(_image(), prompt="a red cube", threshold=0.5)
        self.assertEqual(("grounded_sam", 0.5), (found.backend, found.threshold))
        self.assertEqual(1, len(found))
        self.loads.assert_not_called()
        self.assertIn("loaded when the router first sends it a prompt", self.detector.render())

    def test_a_prompt_the_router_sends_to_the_vlm_loads_it_once_and_the_threshold_does_not_apply(self) -> None:
        for _ in range(2):
            found = self.detector.detect(_image(), prompt="greif den kaputten Wuerfel", threshold=0.5)
            self.assertEqual(("vlm", None), (found.backend, found.threshold))
            self.assertEqual(("greif den kaputten Wuerfel",), found.labels)
        self.loads.assert_called_once_with()
        self.assertEqual(["greif den kaputten Wuerfel"] * 2, self.vlm.calls)
        self.assertEqual([], self.grounder.calls)
        self.assertIn("the VLM from Qwen/Qwen3-VL-8B-Instruct for the rest, loaded", self.detector.render())

    def test_a_router_without_a_way_to_the_vlm_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "needs load_vlm"):
            ObjectDetector(self.grounder, backend="router")
        with self.assertRaisesRegex(ValueError, "backend is one of closed_set, grounded_sam, vlm, router"):
            ObjectDetector(self.grounder, backend="dino")  # type: ignore[arg-type]


class MasksOnAnOpenBackendTests(unittest.TestCase):

    def test_segment_true_cuts_every_box_an_open_backend_found_in_one_sam2_pass(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            folder = sam2_folder(Path(tmp) / "sam2").as_posix()
            config = SegmenterConfig(model_path=folder, model_id=folder, local=True)
            detector = ObjectDetector(Grounder([(RED, "red cube", 0.8), (BLUE, "blue bin", 0.7)]),
                                      backend="grounded_sam", load_segmenter=partial(od._sam2, config, device="cpu"))
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
    """What ``from_config`` builds, and from which block, with the three builders replaced."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.dino = detector_folder(Path(self._tmp.name) / "dino", architecture="GroundingDinoForObjectDetection")
        self.built: list[tuple[str, Any]] = []
        self.grounder = Grounder([(RED, "cube", 0.8)])
        self.vlm = Vlm([(BLUE, "a cube")])
        for name, answer in (("_grounding_dino", self.grounder), ("_vlm_grounder", self.vlm)):
            patcher = mock.patch.object(od, name, side_effect=partial(self._build, name, answer))
            patcher.start()
            self.addCleanup(patcher.stop)

    def _build(self, name: str, answer: Any, block: Any) -> Any:
        self.built.append((name, block))
        return answer

    def tree(self, values: dict[str, Any]) -> Any:
        tree = load_tree(None).with_values({"models.objectdetector.model_path": self.dino.as_posix(), **values})
        self.assertTrue(tree.ok, tree.error)
        return tree

    def test_the_backend_is_what_the_trees_pipeline_builds_for_a_cell(self) -> None:
        cases = {
            "grounded_sam": {},
            "vlm": {"models.pipeline.zero_shot.backend": "vlm"},
            "router": {"models.pipeline.zero_shot.backend": "vlm", "models.pipeline.router.enabled": True},
            "closed_set": {"models.pipeline.kind": "closed_set"},
        }
        for backend, values in cases.items():
            with self.subTest(backend):
                self.assertEqual(backend, od._tree_backend(self.tree(values).app_config.models))
        self.assertEqual("closed_set", od._tree_backend(SimpleNamespace(pipeline=None, detector="rtdetr")))
        self.assertEqual("grounded_sam", od._tree_backend(SimpleNamespace(pipeline=None, detector="groundingdino")))
        self.assertEqual(("closed_set", "grounded_sam", "vlm", "router"), BACKENDS)

    def test_grounding_dino_is_built_from_models_objectdetector(self) -> None:
        tree = self.tree({})
        detector = ObjectDetector.from_config(tree)
        self.assertEqual("grounded_sam", detector.backend)
        self.assertEqual([("_grounding_dino", tree.app_config.models.objectdetector)], self.built)
        self.assertEqual(tree.app_config.models.segmenter.model_path, detector.segmenter_source)

    def test_the_vlm_is_built_from_its_pipeline_block_and_grounding_dino_is_not(self) -> None:
        tree = self.tree({"models.pipeline.zero_shot.backend": "vlm"})
        detector = ObjectDetector.from_config(tree)
        self.assertEqual("vlm", detector.backend)
        self.assertEqual([("_vlm_grounder", tree.app_config.models.pipeline.zero_shot.vlm)], self.built)

    def test_a_router_builds_grounding_dino_now_and_the_vlm_at_the_first_prompt_it_sends_there(self) -> None:
        tree = self.tree({"models.pipeline.zero_shot.backend": "vlm", "models.pipeline.router.enabled": True})
        detector = ObjectDetector.from_config(tree)
        self.assertEqual(["_grounding_dino"], [name for name, _ in self.built])
        detector.detect(_image(), prompt="a red cube")
        self.assertEqual(["_grounding_dino"], [name for name, _ in self.built])
        detector.detect(_image(), prompt="nicht den roten Wuerfel")
        self.assertEqual(["_grounding_dino", "_vlm_grounder"], [name for name, _ in self.built])

    def test_a_named_backend_is_built_whatever_the_pipeline_says(self) -> None:
        tree = self.tree({"models.pipeline.kind": "closed_set"})
        self.assertEqual("grounded_sam", ObjectDetector.from_config(tree, backend="grounded_sam").backend)
        self.assertEqual("vlm", ObjectDetector.from_config(tree.app_config.models, backend="vlm").backend)
        with self.assertRaisesRegex(ValueError, "backend is one of"):
            ObjectDetector.from_config(tree, backend="yolo")  # type: ignore[arg-type]

    def test_a_missing_grounding_dino_is_refused_naming_its_key_and_fetch(self) -> None:
        tree = load_tree(None).with_values({"models.objectdetector.model_path": os.fspath(Path(self._tmp.name) / "no")})
        with self.assertRaises(FileNotFoundError) as caught:
            ObjectDetector.from_config(tree)
        message = str(caught.exception)
        for part in ("models.objectdetector.model_path is", "fetch.py dino-tiny"):
            self.assertIn(part, message)
        self.assertNotIn("DetectorTraining", message, "a GroundingDINO folder is no training run's")
        self.assertEqual([], self.built)
        models = load_tree(None).app_config.models.model_copy(update={"objectdetector": None})
        with self.assertRaisesRegex(ValueError, "models.objectdetector is not set"):
            ObjectDetector.from_config(models)


class WhatADetectionsSaysTests(unittest.TestCase):

    def test_the_backend_and_the_prompt_go_into_its_data_and_come_back(self) -> None:
        found = ObjectDetector(Grounder([(RED, "cube", 0.8)]), backend="grounded_sam").detect(_image(), prompt="a cube")
        data = found.to_dict()
        self.assertEqual(("grounded_sam", "a cube"), (data["backend"], data["prompt"]))
        self.assertEqual(found, Detections.from_dict(data))
        old = {key: value for key, value in data.items() if key not in ("backend", "prompt")}
        self.assertEqual(("closed_set", None), (Detections.from_dict(old).backend, Detections.from_dict(old).prompt))
        self.assertIn("asked      grounded_sam for 'a cube'", found.render())

    def test_a_backend_it_does_not_know_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "Detections.backend is one of"):
            Detections(objects=(), classes=(), threshold=None, image_hw=(2, 2), backend="yolo")


if __name__ == "__main__":
    unittest.main()
