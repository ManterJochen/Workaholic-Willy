"""One locate of a class list finds every bin a task still looks for, and splits them by label.

``Locator.locate(class_list_prompt(["yellow bin", "blue bin"]))`` asks the detector once. The camera source maps each
box's words onto the descriptions (``object_labels``: case, plurals and gray or grey aside, the most specific
description whose every word the label holds), and ``Located.split(prompt)`` says which objects each description
located. What no description clearly claims belongs to none and stays in the frame as an obstacle: a label of another
thing ("orange part"), a label two descriptions fit equally ("green or red part"), a bare noun ("part"), and an object
the model boxed under two descriptions, which the parser made ``ambiguous``. A locate of one description reads as it
always did, every object its own. A locate judges no colour: a bin's mask holds whatever lies in it.

Off the box: a fixed camera a metre above a bench, a backend double, and the real grounder over a model double for the
one case that runs through the parser.
"""

from __future__ import annotations

import contextlib
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.calibration.rig_calibration import RigCalibration
from src.camera.setup.image_taking.frames import RGBDFrame
from src.geometry import Frame, Transform
from src.models.perception_backend import TwoStageBackend
from src.models.segmentation.types import SegmentationResult
from src.models.vlm.parsing import AMBIGUOUS_LABEL
from src.models.vlm.qwen import Qwen3VLGrounder, class_list_prompt
from src.robot.perception.locator import Located, Locator

H, W = 40, 60
_K = np.array([[500.0, 0.0, 30.0], [0.0, 500.0, 20.0], [0.0, 0.0, 1.0]])
#: A fixed camera a metre above the bench, looking straight down.
_CAMERA_TO_BASE = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1000.0], [0.0, 0.0, 0.0, 1.0]])

#: Where things stand in the image, pixel boxes (x0, y0, x1, y1): two bins, 100 mm high, and a part between them.
YELLOW_BIN = (4, 4, 20, 20)
BLUE_BIN = (40, 4, 56, 20)
PART = (26, 26, 34, 34)
YELLOW_BGR, BLUE_BGR, ORANGE_BGR = (0, 210, 240), (200, 60, 0), (0, 120, 255)

BINS = class_list_prompt(["yellow bin", "blue bin"])


def _mask(box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _frame(*, paint: "dict[tuple[int, int, int, int], tuple[int, int, int]] | None" = None) -> RGBDFrame:
    """The bench at 1000 mm, the bins' rims 100 mm up, the part 40 mm up; each painted its colour unless told."""
    depth = np.full((H, W), 1000, dtype=np.uint16)
    colour = np.full((H, W, 3), 30, dtype=np.uint8)
    painted = {YELLOW_BIN: YELLOW_BGR, BLUE_BIN: BLUE_BGR, PART: ORANGE_BGR, **(paint or {})}
    for box, height in ((YELLOW_BIN, 900), (BLUE_BIN, 900), (PART, 960)):
        x0, y0, x1, y1 = box
        depth[y0:y1, x0:x1] = height
        colour[y0:y1, x0:x1] = painted[box]
    return RGBDFrame(color=colour, depth=depth, captured_at_s=50.0)


class _Handle:
    def __init__(self, frame: RGBDFrame) -> None:
        self.rig_id = "overhead"
        self._frame = frame

    def grab(self) -> RGBDFrame:
        return self._frame

    def get_intrinsics(self) -> np.ndarray:
        return _K.copy()


class _Owner:
    """An open fixed camera: a rig id, a source, a handle and its calibration."""

    def __init__(self, frame: RGBDFrame | None = None) -> None:
        self.rig_id = "overhead"
        self.source = "rgbd"
        self._handle = _Handle(frame if frame is not None else _frame())

    def handle(self) -> _Handle:
        return self._handle

    def calibration(self) -> RigCalibration:
        return RigCalibration(rig_id="overhead", mounting_mode="eye_to_hand", artifact_path="eth.json",
                              transform=Transform.from_matrix(_CAMERA_TO_BASE, from_frame=Frame.CAMERA,
                                                              to_frame=Frame.BASE))


@dataclass(frozen=True)
class _Segmentation:
    """Shaped like ``SegmentationResult``, which the camera source rebuilds with ``dataclasses.replace``."""

    label: str
    score: float
    bbox_xyxy: tuple
    mask: np.ndarray


class _Backend:
    """A perception backend double: one object per ``(label, box)``, and every prompt it was asked kept."""

    def __init__(self, *seen: tuple[str, tuple[int, int, int, int]]) -> None:
        self.seen = seen
        self.prompts: list[str] = []

    def perceive(self, image_bgr: Any, prompt: str) -> tuple[Any, ...]:
        self.prompts.append(prompt)
        return tuple(
            SimpleNamespace(detection=SimpleNamespace(score=0.9, box=[float(v) for v in box]),
                            segmentation=_Segmentation(label=label, score=0.9, bbox_xyxy=tuple(float(v) for v in box),
                                                       mask=_mask(box)))
            for label, box in self.seen)


def _locate(prompt: str, backend: Any, frame: RGBDFrame | None = None) -> Located:
    return Locator.from_parts(camera=_Owner(frame), backend=backend).locate(prompt)


class OneLocateFindsEveryBinTests(unittest.TestCase):
    def test_both_bins_in_one_call_split_by_label(self) -> None:
        backend = _Backend(("yellow bin", YELLOW_BIN), ("Blue bins", BLUE_BIN), ("orange part", PART))
        located = _locate(BINS, backend)
        self.assertEqual([BINS], backend.prompts, "every bin in one call")
        self.assertEqual(["yellow bin", "blue bin", "orange part"], [obj.label for obj in located.objects])
        self.assertEqual({"yellow bin": (0,), "blue bin": (1,)}, located.split(BINS))
        yellow, blue = (located.objects[index].centre_mm for index in (0, 1))
        assert yellow is not None and blue is not None
        self.assertLess(yellow[0], blue[0], "each bin where its own pixels place it")
        self.assertAlmostEqual(100.0, yellow[2], delta=1.0)

    def test_a_bin_seen_nowhere_splits_to_nothing(self) -> None:
        prompt = class_list_prompt(["yellow bin", "blue bin", "red bin"])
        located = _locate(prompt, _Backend(("yellow bin", YELLOW_BIN), ("blue bin", BLUE_BIN)))
        self.assertEqual({"yellow bin": (0,), "blue bin": (1,), "red bin": ()}, located.split(prompt))

    def test_two_boxes_under_one_description_are_both_its_own(self) -> None:
        """Which of them is the bin is the caller's choice, as a locate of one description leaves it."""
        located = _locate(BINS, _Backend(("yellow bin", YELLOW_BIN), ("yellow bin", BLUE_BIN)))
        self.assertEqual({"yellow bin": (0, 1), "blue bin": ()}, located.split(BINS))


class WhatNoDescriptionClaimsTests(unittest.TestCase):
    def test_a_label_outside_the_list_is_never_mapped_onto_a_rule(self) -> None:
        prompt = class_list_prompt(["green part", "red part"])
        backend = _Backend(("orange part", PART), ("part", YELLOW_BIN), ("green or red part", BLUE_BIN),
                           ("Green Parts", (26, 4, 34, 12)))
        located = _locate(prompt, backend)
        self.assertEqual(["orange part", "part", "green or red part", "green part"],
                         [obj.label for obj in located.objects])
        self.assertEqual({"green part": (3,), "red part": ()}, located.split(prompt))

    def test_what_belongs_to_no_description_stays_an_obstacle(self) -> None:
        prompt = class_list_prompt(["green part", "red part"])
        located = _locate(prompt, _Backend(("orange part", PART), ("green part", YELLOW_BIN)))
        self.assertGreater(located.objects[0].points_base_mm.shape[0], 0, "its surface is placed like any other")
        offer = located.keep_out(1)
        self.assertEqual(["orange part", "green part"], [label for label, _ in offer.labelled_masks])
        self.assertEqual(1, len(offer.exclude_masks), "only the target is left out of the planner world")
        self.assertTrue(np.array_equal(located.objects[1].mask, offer.exclude_masks[0]))

    def test_an_object_boxed_under_two_descriptions_is_no_one_s(self) -> None:
        """Through the real grounder and parser: the model double boxed the yellow bin twice, once as the blue one."""
        answer = _grid_answer((YELLOW_BIN, "yellow bin"), ((4, 4, 20, 19), "blue bin"), (BLUE_BIN, "blue bin"))
        grounder, model = _grounder(answer)
        located = _locate(BINS, TwoStageBackend(detector=grounder, segmenter=_BoxSegmenter()))
        self.assertEqual(1, len(model.calls), "one call for both bins")
        self.assertEqual([AMBIGUOUS_LABEL, AMBIGUOUS_LABEL, "blue bin"], [obj.label for obj in located.objects])
        self.assertEqual({"yellow bin": (), "blue bin": (2,)}, located.split(BINS))


class OneDescriptionTests(unittest.TestCase):
    def test_a_locate_of_one_description_reads_as_it_always_did(self) -> None:
        backend = _Backend(("red cube", PART), ("cube", YELLOW_BIN), ("orange part", BLUE_BIN))
        located = _locate("a red cube", backend)
        self.assertEqual(["a red cube"], backend.prompts)
        self.assertEqual(["red cube", "cube", "orange part"], [obj.label for obj in located.objects],
                         "the detector's own words, mapped onto nothing")
        self.assertEqual({"a red cube": (0, 1, 2)}, located.split("a red cube"))


class ALocateJudgesNoColourTests(unittest.TestCase):
    def test_a_bin_keeps_the_description_its_box_named_whatever_its_pixels(self) -> None:
        """The pick's sources refuse a part whose pixels name another colour; a locate keeps what the detector said."""
        frame = _frame(paint={YELLOW_BIN: BLUE_BGR})
        located = _locate(BINS, _Backend(("yellow bin", YELLOW_BIN), ("blue bin", BLUE_BIN)), frame)
        self.assertEqual({"yellow bin": (0,), "blue bin": (1,)}, located.split(BINS))


# --- the real grounder over a model double ---------------------------------------------------------------------------


def _grid_answer(*boxes: tuple[tuple[int, int, int, int], str]) -> str:
    """A grounding answer in Qwen's 0-1000 grid for pixel boxes of this frame."""
    items = []
    for (x0, y0, x1, y1), label in boxes:
        grid = [round(x0 / W * 1000), round(y0 / H * 1000), round(x1 / W * 1000), round(y1 / H * 1000)]
        items.append(f'{{"bbox_2d": {grid}, "label": "{label}"}}')
    return "[" + ", ".join(items) + "]"


class _Batch(dict):  # type: ignore[type-arg]
    def to(self, device: Any) -> "_Batch":
        return self


class _Processor:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: None, eos_token_id=None)

    def apply_chat_template(self, messages: Any, **_: Any) -> _Batch:
        return _Batch(input_ids=np.arange(7).reshape(1, -1), attention_mask=np.ones((1, 7)))

    def batch_decode(self, ids: Any, skip_special_tokens: bool = False) -> list[str]:
        return [self.answer]


class _Model:
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.config = SimpleNamespace()
        self.generation_config = SimpleNamespace(eos_token_id=None)

    def generate(self, **kwargs: Any) -> np.ndarray:
        self.calls.append(kwargs)
        return np.concatenate([kwargs["input_ids"], np.full((1, 4), 99)], axis=1)


class _Torch:
    @staticmethod
    def inference_mode() -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    cuda = SimpleNamespace(empty_cache=lambda: None)


def _grounder(answer: str) -> tuple[Qwen3VLGrounder, _Model]:
    model = _Model()
    grounder = Qwen3VLGrounder(model_id="Qwen/Qwen3-VL-4B-Instruct", local=True)
    grounder._load = lambda: (_Processor(answer), model, _Torch())  # type: ignore[method-assign]
    return grounder, model


class _BoxSegmenter:
    """Cuts each detection's box whole, as a mask of the frame."""

    def segment_detection(self, image_bgr: Any, detection: Any) -> SegmentationResult:
        height, width = np.asarray(image_bgr).shape[:2]
        x0, y0, x1, y1 = (int(round(float(value))) for value in detection.box)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 1
        return SegmentationResult(
            label=detection.label, score=float(detection.score), bbox_xyxy=(x0, y0, x1, y1), mask=mask,
            mask_area_px=int(mask.sum()), centroid_xy=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
            derived_bbox_xyxy=(x0, y0, x1, y1), inference_time_s=0.0, timestamp_utc="",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
