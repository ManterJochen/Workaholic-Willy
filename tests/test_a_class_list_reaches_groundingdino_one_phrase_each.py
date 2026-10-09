"""A class list reaches GroundingDINO as one phrase per description: ``"green part | red part"`` is captioned
``"green part . red part"``.

GroundingDINO's text encoder was trained on phrases each ended by a period, and it labels a box with the phrase it
grounded. The bar of a class list is no separator to it: the whole list would read as one phrase, and every box would
carry a label of both descriptions, which the camera source maps onto neither. Every other prompt reaches it as it
always did, byte for byte. A class list reaches it wherever GroundingDINO grounds: the ``grounded_sam`` stack, the
router's simple route (whose normalising adds the closing period) and a degraded VLM's fallback, all through
``detect_all``, so the caption is made there. Its boxes of a class list fail closed as a VLM's do: one object boxed
under two phrases, or a box grounded to none, is ``ambiguous``.

No weights: the detector is built around a processor double that keeps every caption it is handed.
"""

from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from src.models.detection.zero_shot.detector import PHRASE_SEPARATOR, GroundingDinoObjectDetector, grounding_caption
from src.models.perception_backend import TwoStageBackend
from src.models.routed_backend import RoutedPerceptionBackend
from src.models.routing import Route, normalize_simple_prompt
from src.models.vlm import GuardedVlmBackend
from src.models.vlm.parsing import AMBIGUOUS_LABEL
from src.models.vlm.qwen import class_list_prompt


class _Processor:
    """GroundingDINO's processor, as far as the detector reads it: the caption it is handed is kept, and the boxes it
    was given come back for it (one "green part" unless told), as fractions of the image."""

    def __init__(self, boxes: "tuple[tuple[tuple[float, float, float, float], str | None], ...]" = (
            ((0.1, 0.1, 0.5, 0.5), "green part"),)) -> None:
        self.captions: list[str] = []
        self.boxes = boxes

    def __call__(self, *, images: Any, text: str, return_tensors: str) -> dict[str, Any]:
        self.captions.append(text)
        return {"input_ids": torch.zeros((1, 4), dtype=torch.long)}

    def post_process_grounded_object_detection(self, outputs: Any, input_ids: Any, threshold: float) -> list[Any]:
        return [{"scores": torch.tensor([0.9] * len(self.boxes)),
                 "boxes": torch.tensor([list(box) for box, _ in self.boxes]),
                 "text_labels": [label for _, label in self.boxes]}]


def _detector(*boxes: "tuple[tuple[float, float, float, float], str | None]",
              ) -> tuple[GroundingDinoObjectDetector, _Processor]:
    """The real detector class with no weights loaded: its processor and model are doubles."""
    detector = object.__new__(GroundingDinoObjectDetector)
    processor = _Processor(boxes) if boxes else _Processor()
    detector.device = torch.device("cpu")
    detector.optim = None
    detector._model_dtype = None
    detector.threshold = 0.4
    detector.debug_images = False
    detector.logger = logging.getLogger("tests.groundingdino_caption")
    detector.processor = processor
    detector.model = lambda **inputs: SimpleNamespace()
    return detector, processor


class _Segmenter:
    def segment_detection(self, image_bgr: Any, detection: Any) -> Any:
        return SimpleNamespace(label=detection.label, mask=np.ones((8, 8), np.uint8))


class _CannotLoad:
    """A VLM route whose weights are not there."""

    def perceive(self, image_bgr: Any, prompt: str) -> Any:
        raise OSError("no weights")


IMAGE = np.zeros((8, 8, 3), np.uint8)
PROMPT = class_list_prompt(["green part", "red part"])


class TheCaptionTests(unittest.TestCase):
    def test_a_class_list_is_one_phrase_per_description(self) -> None:
        self.assertEqual(" . ", PHRASE_SEPARATOR)
        self.assertEqual("green part . red part", grounding_caption(PROMPT))
        self.assertEqual("yellow bin . blue bin . red bin",
                         grounding_caption(class_list_prompt(["yellow bin", "blue bin", "red bin"])))

    def test_the_router_s_normalised_list_keeps_its_closing_period(self) -> None:
        normalised = normalize_simple_prompt(PROMPT)
        self.assertEqual("green part | red part.", normalised)
        self.assertEqual("green part . red part.", grounding_caption(normalised))

    def test_every_other_prompt_is_captioned_as_it_was(self) -> None:
        for prompt in ("grey cube", "each separate gray cube.", "a red cube", "green part | green part"):
            with self.subTest(prompt=prompt):
                self.assertIs(prompt, grounding_caption(prompt))


class TheDetectorIsHandedTheCaptionTests(unittest.TestCase):
    def test_detect_all_grounds_the_caption(self) -> None:
        detector, processor = _detector()
        found = detector.detect_all(IMAGE, PROMPT)
        detector.detect_all(IMAGE, "grey cube")
        self.assertEqual(["green part . red part", "grey cube"], processor.captions)
        self.assertEqual(["green part"], [det.label for det in found])

    def test_detect_grounds_it_too(self) -> None:
        detector, processor = _detector()
        detector.detect(IMAGE, PROMPT)
        self.assertEqual(["green part . red part"], processor.captions)

    def test_the_router_s_simple_route_hands_it_the_list_as_phrases(self) -> None:
        detector, processor = _detector()
        backend = RoutedPerceptionBackend(simple_factory=lambda: TwoStageBackend(detector=detector,
                                                                                 segmenter=_Segmenter()),
                                          vlm_factory=_CannotLoad)
        objects = backend.perceive(IMAGE, PROMPT)
        assert backend.last_decision is not None
        self.assertIs(Route.SIMPLE, backend.last_decision.route, backend.last_decision.describe())
        self.assertEqual(["green part . red part."], processor.captions)
        self.assertEqual(1, len(objects))

    def test_a_degraded_vlm_s_fallback_hands_it_the_list_as_phrases(self) -> None:
        detector, processor = _detector()
        backend = GuardedVlmBackend(vlm=_CannotLoad(), model_id="qwen", degrade=True,
                                    fallback_factory=lambda: TwoStageBackend(detector=detector, segmenter=_Segmenter()))
        backend.perceive(IMAGE, PROMPT)
        self.assertEqual(["green part . red part"], processor.captions)


class ItsBoxesOfAClassListFailClosedTests(unittest.TestCase):
    """GroundingDINO boxes one object under two phrases often enough; neither rule may take it."""

    def test_one_object_under_two_phrases_is_ambiguous(self) -> None:
        detector, _ = _detector(((0.1, 0.1, 0.5, 0.5), "green part"), ((0.11, 0.1, 0.5, 0.5), "red part"),
                                ((0.6, 0.6, 0.9, 0.9), "red part"))
        self.assertEqual([AMBIGUOUS_LABEL, AMBIGUOUS_LABEL, "red part"],
                         [det.label for det in detector.detect_all(IMAGE, PROMPT)])

    def test_a_box_grounded_to_no_phrase_is_ambiguous(self) -> None:
        detector, _ = _detector(((0.1, 0.1, 0.5, 0.5), None), ((0.6, 0.6, 0.9, 0.9), "  "))
        self.assertEqual([AMBIGUOUS_LABEL, AMBIGUOUS_LABEL], [det.label for det in detector.detect_all(IMAGE, PROMPT)])

    def test_one_phrase_reads_as_it_always_did(self) -> None:
        detector, _ = _detector(((0.1, 0.1, 0.5, 0.5), "grey cube"), ((0.11, 0.1, 0.5, 0.5), "cube"),
                                ((0.6, 0.6, 0.9, 0.9), None))
        self.assertEqual(["grey cube", "cube", "object"], [det.label for det in detector.detect_all(IMAGE, "grey cube")])


if __name__ == "__main__":
    unittest.main()
