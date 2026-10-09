"""Stand-ins for the two models ``ObjectDetector`` drives, so its tests run on a CPU with no weights.

``rtdetr(scene)`` patches the transformers loaders ``RtDetrObjectDetector`` builds from, and ``sam2()`` the ones
``Sam2Segmenter`` builds from, as ``tests/test_model_load_error_naming.py`` patches them: both wrappers run as they do
at a cell, and only ``from_pretrained`` answers, with a stand-in that keeps what it was given and counts its calls.

The RT-DETR stand-in answers with the scene a test chose, every box above the threshold it is asked for. The SAM2
stand-in answers each box of a call with three proposals at the image's own size: the box itself first, the whole
image twice after it, and a predicted IoU of ``0.9 - 0.01 * i`` for the first proposal of box ``i``. So a mask that
lies in its own box, carrying that score, is the first proposal of that box and of no other.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest import mock

import torch

#: A board's classes in id order, as a DetectorTraining dataset of chess pieces would name them.
CHESS = ("white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king",
         "black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king")

#: One object of a scene: its box in image pixels, its class id and its score.
Seen = tuple[Sequence[float], int, float]


def pin_cpu() -> Callable[[], None]:
    """Pin every model the tests build to the CPU through ``WILLY_DEVICE``, and return what undoes it.

    The stand-ins answer on any device, but a detector built on a machine with CUDA puts its tensors there, opens a
    CUDA context beside whatever else holds the card, and says ``cuda`` where a test reads ``cpu``. CI has no GPU, so
    with the CPU pinned these tests run on every machine as they run there. A module calls it in ``setUpModule`` and
    the function it returns in ``tearDownModule``.
    """
    patcher = mock.patch.dict(os.environ, {"WILLY_DEVICE": "cpu"})
    patcher.start()
    return patcher.stop


class RtDetrProcessor:
    """RT-DETR's image processor: an image in, and the scene's boxes back above the threshold each call asks for."""

    def __init__(self, scene: Sequence[Seen]) -> None:
        self.scene = list(scene)
        self.images: list[Any] = []
        self.thresholds: list[float] = []

    def __call__(self, images: Any, return_tensors: str = "pt") -> dict[str, Any]:
        self.images.append(images)
        return {"pixel_values": torch.zeros(1, 3, 8, 8)}

    def post_process_object_detection(self, outputs: Any, target_sizes: Any = None,
                                      threshold: float = 0.5) -> list[dict[str, Any]]:
        self.thresholds.append(threshold)
        kept = [(box, label, score) for box, label, score in self.scene if score > threshold]
        return [{
            "boxes": torch.tensor([[float(v) for v in box] for box, _, _ in kept], dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor([label for _, label, _ in kept], dtype=torch.int64),
            "scores": torch.tensor([score for _, _, score in kept], dtype=torch.float32),
        }]


class Model:
    """A model: ``to`` and ``eval`` keep it where it is and say so, and a call keeps its inputs and answers."""

    def __init__(self, *, config: Any = None, answer: Callable[[Mapping[str, Any]], Any] | None = None) -> None:
        self.config = config
        self.calls: list[dict[str, Any]] = []
        self.moved_to: list[Any] = []
        self._answer = answer

    def to(self, *args: Any, **kwargs: Any) -> "Model":
        self.moved_to.append(args[0] if args else kwargs)
        return self

    def eval(self) -> "Model":
        return self

    def __call__(self, **inputs: Any) -> Any:
        self.calls.append(inputs)
        return self._answer(inputs) if self._answer is not None else SimpleNamespace()


class Sam2Processor:
    """SAM2's processor: one image and its boxes in, as they were given, and masks scaled by the threshold alone."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.image_processor = SimpleNamespace(post_process_masks=self.post_process_masks)

    def __call__(self, images: Any, input_boxes: Any = None, return_tensors: str = "pt") -> dict[str, Any]:
        self.calls.append({"images": images, "input_boxes": input_boxes})
        height, width = images.shape[:2]
        return {"pixel_values": torch.zeros(1, 3, 8, 8), "original_sizes": torch.tensor([[height, width]]),
                "input_boxes": torch.tensor(input_boxes, dtype=torch.float32)}

    @staticmethod
    def post_process_masks(masks: Any, original_sizes: Any, mask_threshold: float = 0.0,
                           binarize: bool = True) -> list[Any]:
        # The stand-in model answers at the image's own size already, so scaling it is the threshold alone.
        return [masks[index] > mask_threshold for index in range(len(original_sizes))]


def sam2_answer(inputs: Mapping[str, Any]) -> SimpleNamespace:
    """Three proposals per box at the image's size: the box, then the whole image twice; and their predicted IoUs."""
    boxes = inputs["input_boxes"][0]
    height, width = (int(value) for value in inputs["original_sizes"][0])
    count = int(boxes.shape[0])
    masks = -torch.ones(1, count, 3, height, width)
    for index, (x0, y0, x1, y1) in enumerate(boxes.tolist()):
        masks[0, index, 0, int(y0):int(y1), int(x0):int(x1)] = 1.0
    masks[0, :, 1:] = 1.0
    scores = torch.tensor([[[0.9 - 0.01 * index, 0.2, 0.1] for index in range(count)]])
    return SimpleNamespace(pred_masks=masks, iou_scores=scores)


@contextlib.contextmanager
def rtdetr(scene: Sequence[Seen], classes: Mapping[int, str] | Sequence[str] = CHESS) -> Iterator[SimpleNamespace]:
    """RT-DETR's loaders answering with a stand-in that sees ``scene`` and knows ``classes`` (id order, or a map)."""
    from src.models.detection.closed_set import detector as module

    id2label = dict(classes) if isinstance(classes, Mapping) else dict(enumerate(classes))
    processor = RtDetrProcessor(scene)
    model = Model(config=SimpleNamespace(id2label=id2label))
    with mock.patch.object(module, "AutoImageProcessor") as processors, \
            mock.patch.object(module, "AutoModelForObjectDetection") as models:
        processors.from_pretrained.return_value = processor
        models.from_pretrained.return_value = model
        yield SimpleNamespace(processor=processor, model=model, loads=models.from_pretrained)


@contextlib.contextmanager
def sam2() -> Iterator[SimpleNamespace]:
    """SAM2's loaders answering with the stand-in processor and model; ``loads`` counts how often SAM2 was built."""
    from src.models.segmentation.realtime import segmenter as module

    processor = Sam2Processor()
    model = Model(answer=sam2_answer)
    with mock.patch.object(module, "Sam2Processor") as processors, mock.patch.object(module, "Sam2Model") as models:
        processors.from_pretrained.return_value = processor
        models.from_pretrained.return_value = model
        yield SimpleNamespace(processor=processor, model=model, loads=models.from_pretrained)


def detector_folder(folder: Path, *, architecture: str = "RTDetrForObjectDetection", weights: bool = True,
                    processor: bool = True) -> Path:
    """A folder shaped as ``save_pretrained`` leaves an RT-DETR: what is checked before a load, and nothing loadable."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps({"architectures": [architecture], "model_type": "rt_detr"}),
                                        encoding="utf-8")
    if weights:
        (folder / "model.safetensors").write_bytes(b"")
    if processor:
        (folder / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    return folder


def sam2_folder(folder: Path) -> Path:
    """A folder shaped as the SAM2 checkpoint the shipped tree names: a ``sam2_video`` config and its weights."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps({"architectures": ["Sam2VideoModel"], "model_type": "sam2_video"}),
                                        encoding="utf-8")
    (folder / "model.safetensors").write_bytes(b"")
    return folder
