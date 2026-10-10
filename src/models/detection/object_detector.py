"""Every detection model a cell can run behind one door: one call finds the objects, each with its mask where asked.

    from willy import DetectorTraining, ObjectDetector, load_tree

    DetectorTraining.from_dataset(dataset="data/chess", out_dir="models/chess").train()   # your pieces, your classes
    detector = ObjectDetector.from_weights("models/chess")   # the best epoch; models/chess/last is the last one
    found = detector.detect(frame_bgr, segment=True)         # every class at once, and a mask for each piece
    print(found)                                             # a line per piece, and how many of each class
    for king in found.of("white_king"):
        print(king.box, king.centre, king.mask.sum())

    detector = ObjectDetector.from_config(load_tree())       # what the cell's models.pipeline names
    found = detector.detect(frame_bgr, prompt="the red cube", segment=True)          # free text, open vocabulary
    found = detector.detect(frame_bgr, classes=["red cube", "blue bin"])             # several kinds, one call

The owner, 2026-10-09: the chess program takes the closed-set detector, always asks it for every class at once, and
wants the masks from the same call, switched on and off by one flag ("als True oder False ausgeben mit
Segmentierung"); and every detection model goes under this one roof, "alle unter einem Dach". The backends
(:data:`BACKENDS`) are the models a cell builds, the names its ``models.pipeline`` uses: ``closed_set``, RT-DETR's
trained classes; ``grounded_sam``, GroundingDINO; ``vlm``, the vision-language model; ``router``, which sends each
prompt to one of those two by its text alone, as a cell's router does. The processor and model of a ``Sam2Segmenter``
cut every box of an image in one pass, the image encoded once, where the cell's perception cuts one box per call.

Importing this module loads no torch; building a detector does.
"""

from __future__ import annotations

import math
import os
import re
import threading
import zlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, get_args, overload

import cv2
import numpy as np

from src.config.paths import path_note
from src.config.schema.models.models_schema import ObjectDetectorConfig, SegmenterConfig
from src.models.constants import MODELS_LOG_DIR, RTDETR_LOG_FILE
from src.models.detection.types import Detection
from src.utility import bgr_to_rgb, create_logger
from src.utility.paths import weights_root

if TYPE_CHECKING:  # pragma: no cover (typing only: both wrappers import torch and transformers)
    from src.models.detection.closed_set.detector import RtDetrObjectDetector
    from src.models.segmentation.realtime.segmenter import Sam2Segmenter

__all__ = [
    "BACKENDS",
    "DEFAULT_SEGMENTER",
    "DEFAULT_THRESHOLD",
    "MASK_ENCODING",
    "Backend",
    "DetectedObject",
    "Detections",
    "ObjectDetector",
    "decode_mask",
    "encode_mask",
]

#: Which model answers a detect, by the names a cell's ``models.pipeline`` gives them: ``closed_set`` is RT-DETR's
#: trained classes (``kind: closed_set``), ``grounded_sam`` GroundingDINO (``zero_shot.backend: grounded_sam``), ``vlm``
#: the vision-language model (``backend: vlm``), and ``router`` the two of those behind the prompt router
#: (``router.enabled``), which sends each prompt to one of them by its text alone.
Backend = Literal["closed_set", "grounded_sam", "vlm", "router"]
#: Every :data:`Backend`, in that order.
BACKENDS: Final[tuple[str, ...]] = get_args(Backend)
#: The score a box must clear when :meth:`ObjectDetector.from_weights` is given none.
DEFAULT_THRESHOLD: Final = 0.5
#: The SAM2 checkpoint :meth:`ObjectDetector.from_weights` cuts the masks with when it is named none: the one the
#: shipped tree names, ``models.segmenter`` in ``config/models/segmenting.yaml``. A test holds the two together.
DEFAULT_SEGMENTER: Final = "facebook/sam2-hiera-large"
#: What :meth:`Detections.to_dict` writes a mask as: COCO's uncompressed run-length encoding (:func:`encode_mask`).
MASK_ENCODING: Final = "coco_rle"

#: The files ``save_pretrained`` writes a checkpoint's weights to, whole or in shards.
_WEIGHT_FILES: Final = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin",
                        "pytorch_model.bin.index.json")
#: A Hugging Face id as the Hub spells one: a name, or a namespace and a name.
_HUB_ID: Final = re.compile(r"[A-Za-z0-9][\w.-]*(/[A-Za-z0-9][\w.-]*)?")
#: The folder ``scripts/model_weights/fetch.py`` writes each kind of checkpoint to, under the weights root.
_FETCHED: Final = {"detector": "detection", "segmenter": "segmentation"}
#: How many masks are scaled up to the image at a time: one mask of a 4K image is 33 MB of float.
_SCALED_AT_ONCE: Final = 8
#: How much of its class's colour tints a mask in a drawing, as in the live view (``src/camera/live_view.py``).
_TINT: Final = 0.45
#: One colour per class looked for, in their order, BGR; the thirteenth class takes the first again.
_PALETTE: Final = ((0, 200, 255), (255, 140, 0), (80, 220, 80), (220, 60, 220), (0, 110, 255), (255, 255, 0),
                   (60, 60, 230), (200, 200, 200), (0, 255, 160), (255, 80, 160), (40, 170, 170), (180, 120, 255))
#: How many class names a refusal or a rendering names before it counts the rest.
_NAMED: Final = 12

_LOG = create_logger("ObjectDetector", log_file=RTDETR_LOG_FILE, log_dir=MODELS_LOG_DIR)


# ---------------------------------------------------------------------------------------------------------------------
# What a detection answers with
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DetectedObject:
    """One object a detector found: its class, how sure the model is, where it is, and its mask where one was asked.

    Two objects compare by everything but their masks.

    Attributes:
        label (str): The object's class: a closed-set model's class name, or what an open-vocabulary call asked for (the
            prompt, or the description the model gave the box).
        score (float): How sure the model is, from 0 to 1. A VLM gives no score of its own, so its boxes carry 1.0.
        box (tuple[float, float, float, float]): ``(x0, y0, x1, y1)`` in pixels of the image the detector was given,
            clipped to it, so ``0 <= x0 < x1 <= width`` and ``0 <= y0 < y1 <= height``.
        mask (np.ndarray | None): The image's height x width, ``bool``, ``True`` on the object and read-only; all
            ``False`` where SAM2 cut nothing in the box; ``None`` when no mask was asked for (default: None).
        mask_score (float | None): SAM2's own prediction of the mask's IoU, from 0 to 1; ``None`` where the model gave
            none or no mask was asked for (default: None).

    Raises:
        ValueError: An empty label, a score outside 0 to 1, a box that is not four finite numbers with ``x1 > x0`` and
            ``y1 > y0``, or a mask that is not two-dimensional.
    """

    label: str
    score: float
    box: tuple[float, float, float, float]
    mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    mask_score: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError(f"DetectedObject.label must be a non-empty string, got {self.label!r}")
        if not (math.isfinite(self.score) and 0.0 <= self.score <= 1.0):
            raise ValueError(f"DetectedObject.score must be in [0, 1], got {self.score!r}")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "box", _four(self.box))
        if self.mask is not None:
            mask = self.mask
            if not (isinstance(mask, np.ndarray) and mask.dtype == np.bool_ and not mask.flags.writeable):
                mask = np.array(mask, dtype=bool)  # the object's own copy, so nothing outside it can change it
                mask.flags.writeable = False
            if mask.ndim != 2:
                raise ValueError(f"DetectedObject.mask must be height x width, got shape {mask.shape}")
            object.__setattr__(self, "mask", mask)
        if self.mask_score is not None:
            object.__setattr__(self, "mask_score", float(self.mask_score))

    @property
    def centre(self) -> tuple[float, float]:
        """The middle of the box, ``(x, y)`` in pixels."""
        x0, y0, x1, y1 = self.box
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    def to_dict(self, *, mask: bool = True) -> dict[str, Any]:
        """The object as plain data that ``json.dumps`` takes as it is.

        Args:
            mask (bool): Whether to write the mask, run-length encoded (:func:`encode_mask`); ``False`` writes ``None``
                in its place (default: True).

        Returns:
            dict[str, Any]: ``label`` (str), ``score`` (float), ``box`` (four floats), ``centre`` (two floats), ``mask``
                (``{"size": [h, w], "counts": [...]}`` or None) and ``mask_score`` (float or None).
        """
        return {
            "label": self.label,
            "score": self.score,
            "box": list(self.box),
            "centre": list(self.centre),
            "mask": encode_mask(self.mask) if mask and self.mask is not None else None,
            "mask_score": self.mask_score,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DetectedObject:
        """The object :meth:`to_dict` wrote, its mask decoded where one was written.

        Args:
            data (Mapping[str, Any]): What :meth:`to_dict` returned, or the same read back from JSON.

        Returns:
            DetectedObject: The object, with its mask where ``data`` holds one.

        Raises:
            KeyError: ``data`` lacks ``label``, ``score`` or ``box``.
            ValueError: The values break the object's contract, or the mask's counts do not fill it.
        """
        encoded = data.get("mask")
        score = data.get("mask_score")
        return cls(label=str(data["label"]), score=float(data["score"]), box=_four(data["box"]),
                   mask=decode_mask(encoded) if encoded is not None else None,
                   mask_score=None if score is None else float(score))


@dataclass(frozen=True, slots=True)
class Detections(Sequence[DetectedObject]):
    """What one :meth:`ObjectDetector.detect` found, the highest score first, as a frozen sequence of
    :class:`DetectedObject`: ``len(found)``, ``found[0]``, ``for obj in found``, and an empty one is falsy.

    It keeps the question it answers, so a result read back later still says what was asked.

    Attributes:
        objects (tuple[DetectedObject, ...]): What was found, the highest score first.
        classes (tuple[str, ...]): The classes looked for: a closed-set detector's in its id order (every class the
            model knows unless ``classes=`` narrowed them), an open-vocabulary one's as they were asked, a prompt being
            one.
        threshold (float | None): The score every object cleared; ``None`` where the model gives no score (a VLM).
        image_hw (tuple[int, int]): The image's height and width in pixels.
        segmented (bool): Whether a mask was asked for each object (default: False).
        backend (str): Which model answered, one of :data:`BACKENDS`; a router's call names the route it took
            (default: "closed_set").
        prompt (str | None): The text the model read: the prompt, or the class list (``"red cube | blue bin"``);
            ``None`` for a closed-set detector, which reads none (default: None).

    Raises:
        ValueError: A mask that is not the image's size, or a backend that is none of :data:`BACKENDS`.
    """

    objects: tuple[DetectedObject, ...]
    classes: tuple[str, ...]
    threshold: float | None
    image_hw: tuple[int, int]
    segmented: bool = False
    backend: str = "closed_set"
    prompt: str | None = None

    def __post_init__(self) -> None:
        height, width = (int(value) for value in self.image_hw)
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "classes", tuple(str(name) for name in self.classes))
        object.__setattr__(self, "threshold", None if self.threshold is None else float(self.threshold))
        object.__setattr__(self, "image_hw", (height, width))
        if self.backend not in BACKENDS:
            raise ValueError(f"Detections.backend is one of {', '.join(BACKENDS)}, got {self.backend!r}")
        for obj in self.objects:
            if obj.mask is not None and obj.mask.shape != (height, width):
                raise ValueError(f"a mask of {obj.label!r} is {obj.mask.shape[0]} x {obj.mask.shape[1]}, and the image "
                                 f"is {height} x {width}: a mask is the image's own size")

    def __len__(self) -> int:
        return len(self.objects)

    @overload
    def __getitem__(self, index: int) -> DetectedObject: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[DetectedObject, ...]: ...

    def __getitem__(self, index: int | slice) -> DetectedObject | tuple[DetectedObject, ...]:
        return self.objects[index]

    def __iter__(self) -> Iterator[DetectedObject]:
        return iter(self.objects)

    @property
    def labels(self) -> tuple[str, ...]:
        """Each object's class, in the order of the objects."""
        return tuple(obj.label for obj in self.objects)

    @property
    def counts(self) -> dict[str, int]:
        """How many objects of each class were found: every class looked for, in id order, ``0`` where none was."""
        counted = {name: 0 for name in self.classes}
        by_key = {_name_key(name): name for name in self.classes}
        for obj in self.objects:
            name = obj.label if obj.label in counted else by_key.get(_name_key(obj.label), obj.label)
            counted[name] = counted.get(name, 0) + 1
        return counted

    def of(self, label: str) -> tuple[DetectedObject, ...]:
        """The objects of one class, the highest score first.

        Args:
            label (str): A class this detection looked for, matched as ``classes=`` matches: case and blanks aside.

        Returns:
            tuple[DetectedObject, ...]: The objects of that class; empty when none was found.

        Raises:
            ValueError: ``label`` is none of the classes looked for: nothing was looked for under that name, and an
                empty answer would read as "none there".
        """
        key = _name_key(label)
        if key not in {_name_key(name) for name in self.classes}:
            raise ValueError(f"{label!r} is not a class this detection looked for; it looked for "
                             f"{_listed(self.classes)}")
        return tuple(obj for obj in self.objects if _name_key(obj.label) == key)

    def render(self) -> str:
        """What was found, as text for a person: the question, a line per object, and how many of each class.

        Returns:
            str: ASCII only, since it reaches a Windows console under cp1252, and no newline at the end.
                ``print(found)`` shows the same.
        """
        height, width = self.image_hw
        cleared = f"above {self.threshold:.2f}" if self.threshold is not None else "(the model gives no score)"
        lines = [f"  found      {len(self.objects)} object(s) {cleared} in a {width} x {height} px "
                 f"image, of {len(self.classes)} class(es) looked for"
                 + (", masks asked for" if self.segmented else "")]
        if self.prompt is not None:
            lines.append(f"  asked      {self.backend} for {_ascii(self.prompt)!r}")
        names = [_ascii(obj.label) for obj in self.objects]
        column = max((len(name) for name in names), default=0)
        for name, obj in zip(names, self.objects):
            x0, y0, x1, y1 = obj.box
            line = f"  {name:<{column}}  {obj.score:.2f}  box ({x0:.0f}, {y0:.0f}) to ({x1:.0f}, {y1:.0f})"
            if obj.mask is not None:
                line += f"  mask {int(obj.mask.sum())} px"
                if obj.mask_score is not None:
                    line += f", SAM2 IoU {obj.mask_score:.2f}"
            lines.append(line)
        counts = self.counts
        found = [f"{_ascii(name)} {count}" for name, count in counts.items() if count]
        if found:
            lines.append("  counted    " + ", ".join(found))
        missing = [name for name, count in counts.items() if not count]
        if missing:
            lines.append("  none of    " + _listed(missing))
        return "\n".join(lines)

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def draw(self, image_bgr: np.ndarray | str | os.PathLike[str]) -> np.ndarray:
        """The image with every object drawn on it: its box, its class and score, and its mask tinted.

        Each class gets a colour of its own; the image given is not changed.

        Args:
            image_bgr (np.ndarray | str | os.PathLike[str]): The image detected in, height x width x 3 ``uint8`` in BGR
                order, or its file's path.

        Returns:
            np.ndarray: A new BGR image of the same size.

        Raises:
            ValueError: The image is of another size than the one detected in, since the boxes and masks are in its
                pixels, or it is no height x width x 3 ``uint8`` image.
            FileNotFoundError: A path names no file.
        """
        image = _image(image_bgr)
        if image.shape[:2] != self.image_hw:
            raise ValueError(f"these detections are of a {self.image_hw[1]} x {self.image_hw[0]} px image, and this "
                             f"one is {image.shape[1]} x {image.shape[0]}: draw them on the image they were found in")
        out = image.copy()
        for obj in self.objects:
            colour = self._colour(obj.label)
            if obj.mask is not None and obj.mask.any():
                tinted = out[obj.mask].astype(np.float64) * (1.0 - _TINT) + np.asarray(colour, np.float64) * _TINT
                out[obj.mask] = tinted.astype(np.uint8)
                contours, _ = cv2.findContours(obj.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, contours, -1, colour, 1, cv2.LINE_AA)
            x0, y0, x1, y1 = (int(round(value)) for value in obj.box)
            cv2.rectangle(out, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), colour, 2)
            _write_label(out, f"{_ascii(obj.label)} {obj.score:.2f}", (x0, max(12, y0 - 4)), colour)
        return out

    def write_drawing(self, image_bgr: np.ndarray | str | os.PathLike[str], path: str | os.PathLike[str]) -> Path:
        """Write :meth:`draw` of an image to a file, in the format its suffix names.

        Args:
            image_bgr (np.ndarray | str | os.PathLike[str]): The image detected in, or its path (see :meth:`draw`).
            path (str | os.PathLike[str]): Where to write, ending in ``.png`` or ``.jpg``; its folder is made if
                missing.

        Returns:
            Path: Where the drawing went.

        Raises:
            ValueError: ``path`` has no suffix, OpenCV writes no image of that format, or :meth:`draw` refuses the
                image.
        """
        target = Path(path)
        if not target.suffix:
            raise ValueError(f"{target} names no image format; end it in .png or .jpg")
        try:
            written, encoded = cv2.imencode(target.suffix, self.draw(image_bgr))
        except cv2.error as exc:
            raise ValueError(f"OpenCV writes no {target.suffix} image: {exc}") from exc
        if not written:
            raise ValueError(f"OpenCV could not encode the drawing as {target.suffix}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded.tobytes())
        return target

    def to_dict(self, *, masks: bool = True) -> dict[str, Any]:
        """The detections as plain data that ``json.dumps`` takes as it is, and :meth:`from_dict` reads back.

        Args:
            masks (bool): Whether to write the masks, each in COCO's uncompressed run-length encoding
                (:func:`encode_mask`), a few counts per image column the object crosses (default: True).

        Returns:
            dict[str, Any]: ``image_hw``, ``threshold``, ``classes``, ``backend``, ``prompt``, ``segmented``,
                ``mask_encoding`` (``"coco_rle"``, or None when no mask is written), ``counts`` and ``objects``, each
                object as :meth:`DetectedObject.to_dict` writes it.
        """
        written = masks and self.segmented
        return {
            "image_hw": list(self.image_hw),
            "threshold": self.threshold,
            "classes": list(self.classes),
            "backend": self.backend,
            "prompt": self.prompt,
            "segmented": self.segmented,
            "mask_encoding": MASK_ENCODING if written else None,
            "counts": self.counts,
            "objects": [obj.to_dict(mask=written) for obj in self.objects],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Detections:
        """The detections :meth:`to_dict` wrote, with their masks where it wrote them.

        Args:
            data (Mapping[str, Any]): What :meth:`to_dict` returned, or the same read back from JSON. Data written
                before ``backend`` and ``prompt`` existed reads as ``"closed_set"`` and ``None``.

        Returns:
            Detections: The same detections.

        Raises:
            ValueError: The masks are in an encoding this does not read, or a value breaks the contract.
            KeyError: ``data`` lacks ``image_hw``, ``objects``, ``classes`` or ``threshold``.
        """
        encoding = data.get("mask_encoding")
        if encoding not in (None, MASK_ENCODING):
            raise ValueError(f"masks written as {encoding!r} are not read here; {MASK_ENCODING!r} is")
        height, width = (int(value) for value in data["image_hw"])
        threshold = data.get("threshold")
        prompt = data.get("prompt")
        return cls(objects=tuple(DetectedObject.from_dict(item) for item in data["objects"]),
                   classes=tuple(str(name) for name in data["classes"]),
                   threshold=None if threshold is None else float(threshold), image_hw=(height, width),
                   segmented=bool(data.get("segmented", False)), backend=str(data.get("backend", "closed_set")),
                   prompt=None if prompt is None else str(prompt))

    def _colour(self, label: str) -> tuple[int, int, int]:
        """The colour of ``label``'s class: its place among the classes looked for, else one its name picks."""
        key = _name_key(label)
        for index, name in enumerate(self.classes):
            if _name_key(name) == key:
                return _PALETTE[index % len(_PALETTE)]
        return _PALETTE[zlib.crc32(label.encode("utf-8")) % len(_PALETTE)]


def encode_mask(mask: np.ndarray) -> dict[str, Any]:
    """``mask`` in COCO's uncompressed run-length encoding: ``{"size": [height, width], "counts": [...]}``.

    The pixels are read column by column, as COCO reads them, and the counts alternate between background and object,
    background first, so a mask whose first pixel is the object starts with a count of 0. ``pycocotools`` takes this
    form (``frPyObjects``), and :func:`decode_mask` turns it back into the mask.
    """
    pixels = np.asarray(mask, dtype=bool)
    if pixels.ndim != 2:
        raise ValueError(f"a mask is height x width, got shape {pixels.shape}")
    flat = pixels.ravel(order="F")
    edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = [int(count) for count in np.diff(np.concatenate(([0], edges, [flat.size])))]
    if flat.size and flat[0]:
        counts.insert(0, 0)
    return {"size": [int(pixels.shape[0]), int(pixels.shape[1])], "counts": counts}


def decode_mask(encoded: Mapping[str, Any]) -> np.ndarray:
    """The mask :func:`encode_mask` encoded, ``True`` on the object.

    Raises ``ValueError`` for counts that do not fill the mask, and for COCO's compressed form (a string of counts),
    which ``pycocotools`` decodes and this does not.
    """
    height, width = (int(value) for value in encoded["size"])
    if isinstance(encoded["counts"], (str, bytes)):
        raise ValueError("the counts are COCO's compressed form; decode them with pycocotools, or write them "
                         "uncompressed")
    counts = np.asarray(encoded["counts"], dtype=np.int64)
    if counts.ndim != 1 or bool((counts < 0).any()) or int(counts.sum()) != height * width:
        raise ValueError(f"run-length counts for a {height} x {width} mask add up to {height * width} pixels, and "
                         f"these add up to {int(counts.sum()) if counts.ndim == 1 else 'nothing'}")
    values = np.zeros(counts.size, dtype=bool)
    values[1::2] = True
    return np.repeat(values, counts).reshape((height, width), order="F")


# ---------------------------------------------------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------------------------------------------------


class ObjectDetector:
    """One door to every detection model a cell can run, answering with the objects it found and, where asked, each
    object's mask: RT-DETR, GroundingDINO or the VLM finds the boxes, SAM2 cuts them.

    Build it once with :meth:`from_weights` (the closed-set detector of a ``DetectorTraining`` run, any RT-DETR folder,
    a Hugging Face id) or :meth:`from_config` (what a cell's tree names, or the ``backend`` you name), and reuse it:
    building loads the detector (the VLM loads at its first call unless its block preloads it), and the first
    ``segment=True`` with a box to cut loads SAM2, which is then kept. Calls on one instance take turns under a lock, so
    threads may share it; every instance holds its own copy of each model but the VLM, which is the process's one copy
    (``shared_vlm()``), as a cell's is. You rarely call the constructor yourself.

    Args:
        detector (Any): The model already built: an ``RtDetrObjectDetector`` for ``closed_set``, a
            ``GroundingDinoObjectDetector`` for ``grounded_sam`` and ``router``, the VLM's grounder for ``vlm``.
        backend (str): Which model ``detector`` is, one of :data:`BACKENDS` (default: "closed_set").
        load_segmenter (Callable[[], Sam2Segmenter] | None): Builds SAM2 when the first mask is asked for; without it
            ``segment=True`` is refused (default: None).
        source (str): Where the detector's weights come from, a folder or a Hugging Face id, for :meth:`render`; left
            empty, the detector's ``model_path`` (default: "").
        segmenter_source (str): Where SAM2's weights come from, for :meth:`render` (default: "").
        load_vlm (Callable[[], Any] | None): For ``router`` only, and needed there: builds the VLM's grounder when the
            router first sends it a prompt (default: None).
        router (Any): For ``router``: the prompt router, anything with ``route(text)`` answering a ``RouteDecision``;
            ``from_config`` passes the cell's ``RuleBasedRouter`` (default: None).
        vlm_source (str): Where the VLM's weights come from, for :meth:`render` (default: "").

    Raises:
        ValueError: ``backend`` is none of :data:`BACKENDS`, or a router has no ``load_vlm``.
    """

    def __init__(self, detector: Any, *, backend: str = "closed_set",
                 load_segmenter: Callable[[], Sam2Segmenter] | None = None, source: str = "",
                 segmenter_source: str = "", load_vlm: Callable[[], Any] | None = None, router: Any = None,
                 vlm_source: str = "") -> None:
        if backend not in BACKENDS:
            raise ValueError(f"backend is one of {', '.join(BACKENDS)}, got {backend!r}")
        if backend == "router" and load_vlm is None:
            raise ValueError("a router sends some prompts to the VLM, so it needs load_vlm, which builds the VLM's "
                             "grounder; from_config gives it one")
        self._backend = str(backend)
        self._detector = detector
        self._load_segmenter = load_segmenter
        self._segmenter: Sam2Segmenter | None = None
        self._load_vlm = load_vlm
        self._vlm: Any = None
        self._router = router
        self._lock = threading.Lock()
        #: Where the detector's weights come from, a folder or a Hugging Face id, and where the VLM's and SAM2's do.
        self.source = source or str(getattr(detector, "model_path", "") or "")
        self.vlm_source = vlm_source
        self.segmenter_source = segmenter_source

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_weights(cls, path_or_id: str | os.PathLike[str], *, threshold: float = DEFAULT_THRESHOLD,
                     device: str | None = None, segmenter: str | os.PathLike[str] | None = None,
                     local: bool | None = None) -> ObjectDetector:
        """The closed-set detector in a folder or on the Hub, with SAM2 for its masks.

        RT-DETR loads now; SAM2 loads when the first mask is asked for.

        Args:
            path_or_id (str | os.PathLike[str]): A ``DetectorTraining`` out_dir (its best epoch; ``<out_dir>/last``
                holds the last and loads the same way), any folder ``save_pretrained`` wrote a transformers RT-DETR and
                its image processor to, or a Hugging Face id such as ``"PekingU/rtdetr_r50vd"``.
            threshold (float): The score from 0 to 1 a box must clear; ``detect(threshold=)`` changes it for one call
                (default: 0.5).
            device (str | None): ``"cpu"``, ``"cuda"`` or ``"mps"`` for both models; ``None`` runs them where every
                model here runs, ``WILLY_DEVICE`` or else the first of CUDA, MPS and the CPU (default: None).
            segmenter (str | os.PathLike[str] | None): SAM2 as a folder or a Hugging Face id; ``None`` is
                :data:`DEFAULT_SEGMENTER`, the one the shipped tree names (default: None).
            local (bool | None): ``True`` reads this machine alone and downloads nothing; otherwise an id is read from
                the folder ``scripts/model_weights/fetch.py`` writes, else the Hugging Face cache, else the Hub
                (default: None).

        Returns:
            ObjectDetector: A ``closed_set`` detector; ``classes`` lists every class the model knows.

        Raises:
            FileNotFoundError: A folder holds no detector (it names what is missing), an out_dir kept only its last
                epoch (it names ``last/``), or a path is not there.
            ValueError: A folder holds another kind of checkpoint, or ``threshold`` is not a score from 0 to 1.
            TypeError: ``path_or_id`` is None, which is what a ``DetectorTrainingReport.model_dir`` is when its run
                wrote no model.
        """
        cut = _checked_threshold(threshold)
        source, on_disk = _weights_source(path_or_id, kind="detector", local=local)
        mask_source, mask_on_disk = _weights_source(DEFAULT_SEGMENTER if segmenter is None else segmenter,
                                                    kind="segmenter", local=local)
        detector = _rtdetr(ObjectDetectorConfig(model_path=source, model_id=source, threshold=cut, local=on_disk),
                           device=device)
        sam2 = SegmenterConfig(model_path=mask_source, model_id=mask_source, local=mask_on_disk)
        return cls(detector, load_segmenter=partial(_sam2, sam2, device=device), source=source,
                   segmenter_source=mask_source)

    @classmethod
    def from_config(cls, config: Any, *, backend: str | None = None) -> ObjectDetector:
        """The detector a cell's config tree names, or the ``backend`` named here, with the tree's SAM2 for its masks.

        Args:
            config (Any): A loaded tree (``load_tree()``), its ``AppConfig``, or its models block.
            backend (str | None): ``None`` builds what the tree's ``models.pipeline`` builds for a cell: ``closed_set``
                for ``kind: closed_set``, else ``grounded_sam``, or ``vlm`` / ``router`` where ``zero_shot.backend:
                vlm`` (``router`` when ``router.enabled``); a tree without the block reads its legacy
                ``models.detector``. A name builds that backend from its block whatever the pipeline says:
                ``"closed_set"`` from ``models.rtdetr``, ``"grounded_sam"`` from ``models.objectdetector``, ``"vlm"``
                from ``models.pipeline.zero_shot.vlm``, ``"router"`` from the last two (default: None).

        Returns:
            ObjectDetector: The detector, its masks from ``models.segmenter`` (SAM2, whatever the pipeline names as the
                segmenter, since OneFormer takes no batch of boxes). Every model runs where every model here runs.

        Raises:
            ValueError: ``backend`` is none of :data:`BACKENDS`, or the tree lacks the block the backend needs (it names
                the key).
            FileNotFoundError: A ``local`` block's folder holds no checkpoint, naming its key: the detector's at build,
                SAM2's at the first mask, the VLM's when it loads.
            ConfigError: The tree did not load.
            TypeError: ``config`` is no tree, ``AppConfig`` or models block.
        """
        models = _models_block(config)
        chosen = str(backend) if backend is not None else _tree_backend(models)
        if chosen not in BACKENDS:
            raise ValueError(f"backend is one of {', '.join(BACKENDS)}, got {chosen!r}")
        sam2 = models.segmenter
        masks: dict[str, Any] = {
            "load_segmenter": partial(_sam2_of_the_tree, sam2) if sam2 is not None else None,
            "segmenter_source": _source_of(sam2) if sam2 is not None else "",
        }
        if chosen == "closed_set":
            block = models.rtdetr
            if block is None:
                raise ValueError("models.rtdetr is not set, and the closed-set detector is RT-DETR: add the block "
                                 "(config/models/object.yaml has one), or load a folder with "
                                 "ObjectDetector.from_weights")
            _checked_block(block, key="models.rtdetr.model_path", kind="detector", fetch="rtdetr")
            return cls(_rtdetr(block, device=None), source=_source_of(block), **masks)
        if chosen == "vlm":
            vlm_block = _vlm_block(models)
            return cls(_vlm_grounder(vlm_block), backend="vlm", source=_source_of(vlm_block), **masks)
        grounder_block = models.objectdetector
        if grounder_block is None:
            raise ValueError(f"models.objectdetector is not set, and {chosen!r} grounds "
                             + ("its plain prompts " if chosen == "router" else "")
                             + "with GroundingDINO: add the block (config/models/object.yaml has one)")
        _checked_block(grounder_block, key="models.objectdetector.model_path", kind="grounder", fetch="dino-tiny")
        if chosen == "grounded_sam":
            return cls(_grounding_dino(grounder_block), backend="grounded_sam", source=_source_of(grounder_block),
                       **masks)
        vlm_block = _vlm_block(models)  # refused before GroundingDINO loads, as a cell's build refuses
        return cls(_grounding_dino(grounder_block), backend="router", source=_source_of(grounder_block),
                   load_vlm=partial(_vlm_grounder, vlm_block), vlm_source=_source_of(vlm_block),
                   router=_rule_based_router(), **masks)

    # ------------------------------------------------------------------ what it is
    @property
    def backend(self) -> str:
        """Which model answers: ``closed_set``, ``grounded_sam``, ``vlm`` or ``router`` (:data:`BACKENDS`)."""
        return self._backend

    @property
    def open_vocabulary(self) -> bool:
        """Whether the detector reads text, a prompt or classes of your own, rather than knowing a fixed class list."""
        return self._backend != "closed_set"

    @property
    def classes(self) -> tuple[str, ...]:
        """Every class name a closed-set model knows, in id order: exactly the names of its ``id2label``. An
        open-vocabulary detector knows no fixed list, and gives ``()``: it finds what the prompt names."""
        if self._backend != "closed_set":
            return ()
        return tuple(self._detector.classes)

    @property
    def threshold(self) -> float | None:
        """The score a box must clear, unless one call names another; ``None`` for the VLM, which gives no score, and a
        router's simple route's, GroundingDINO's."""
        if self._backend == "vlm":
            return None
        return float(self._detector.threshold)

    @property
    def device(self) -> str:
        """Where the detector runs, such as ``cuda`` or ``cpu``; for the VLM, where it loads."""
        if self._backend == "vlm":
            from src.utility.device import get_device  # noqa: PLC0415 (torch)

            return str(getattr(self._detector, "device", None) or get_device())
        return str(self._detector.device)

    def render(self) -> str:
        """What this detector is, as text for a person: its model and weights, its classes, its masks.

        Returns:
            str: ASCII only, and no newline at the end. ``print(detector)`` shows the same.
        """
        if self._segmenter is not None:
            masks = f"SAM2 from {_ascii(self.segmenter_source)}, loaded on {self._segmenter.device}"
        elif self._load_segmenter is not None:
            masks = f"SAM2 from {_ascii(self.segmenter_source)}, loaded by the first segment=True with a box to cut"
        else:
            masks = "none: segment=True is refused"
        if self._backend == "closed_set":
            classes = self.classes
            lines = [f"  detector   RT-DETR from {_ascii(self.source)}, on {self.device}, "
                     f"threshold {self.threshold:.2f}",
                     f"  classes    {len(classes)}: {_listed(classes)}"]
        elif self._backend == "grounded_sam":
            lines = [f"  detector   GroundingDINO from {_ascii(self.source)}, on {self.device}, threshold "
                     f"{self.threshold:.2f}",
                     "  classes    any: a prompt, or the classes a call names"]
        elif self._backend == "vlm":
            lines = [f"  detector   the VLM from {_ascii(self.source)}, which gives no score",
                     "  classes    any: a prompt, or the classes a call names"]
        else:
            vlm = "loaded" if self._vlm is not None else "loaded when the router first sends it a prompt"
            lines = [f"  detector   a router: GroundingDINO from {_ascii(self.source)}, on {self.device}, threshold "
                     f"{self.threshold:.2f}, for the plain prompts; the VLM from {_ascii(self.vlm_source)} for the "
                     f"rest, {vlm}",
                     "  classes    any: a prompt, or the classes a call names"]
        return "\n".join([*lines, f"  masks      {masks}"])

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    # ------------------------------------------------------------------ the one verb
    def detect(self, image_bgr: np.ndarray | str | os.PathLike[str], *, prompt: str | None = None,
               classes: Iterable[str] | None = None, threshold: float | None = None,
               segment: bool = False) -> Detections:
        """Find the objects in an image, the highest score first, each with its mask where asked.

        A closed-set detector answers with every class it knows; an open-vocabulary one (GroundingDINO, the VLM, the
        router) finds what the call names, and takes exactly one of ``prompt`` and ``classes``.

        Args:
            image_bgr (np.ndarray | str | os.PathLike[str]): Height x width x 3 ``uint8`` in BGR order, as OpenCV gives
                it, or an image file's path, read the same way.
            prompt (str | None): Open vocabulary only: free text such as ``"the red cube"``; every box found comes back
                under the prompt as its class. A prompt holding a class list (``"red cube | blue bin"``) is read as
                ``classes`` (default: None).
            classes (Iterable[str] | None): Closed set: narrows the answer to these class names, matched exactly but for
                case and blanks; ``None`` is every class. Open vocabulary: several descriptions found in one call, each
                box under the description the model gave it; a box it gave none, or two, is left out. One string is one
                name (default: None).
            threshold (float | None): The score from 0 to 1 a box must clear, for this call alone; ``None`` keeps the
                detector's own. The VLM gives no score, so a VLM detector refuses one and a router applies it only where
                it sends the prompt to GroundingDINO (default: None).
            segment (bool): Cut every object's mask with SAM2, all the boxes of the image in one pass; an image with no
                box asks SAM2 nothing (default: False).

        Returns:
            Detections: The objects found, the highest score first, and the question they answer: ``classes`` looked
                for, ``threshold`` (``None`` for the VLM), ``backend`` (the route a router took) and ``prompt``.

        Raises:
            ValueError: A closed-set call with a prompt, or with a class the model does not know (it names every class
                it knows); an open-vocabulary call with neither or both of ``prompt`` and ``classes``, an empty prompt,
                or a class ``class_list_prompt`` refuses; a threshold outside 0 to 1, or any threshold on the VLM;
                ``segment=True`` on a detector built without SAM2; an image that is not height x width x 3 ``uint8``.
            FileNotFoundError: An image path names no file, or SAM2's folder holds no checkpoint at the first mask.

        Each object comes back once: RT-DETR scores each class of each of its queries on its own, so one object can
        clear the threshold under two classes with the same box; it keeps its best class, and ``classes=`` picks from
        those. A box reaching past the image is clipped to it, and one wholly outside it is dropped.
        """
        image = _image(image_bgr)
        if self._backend == "closed_set":
            if prompt is not None:
                raise ValueError("RT-DETR is a closed-set detector and reads no text: name the classes you want with "
                                 f"classes=, or leave it out for all {len(self.classes)} it knows")
            asked, wanted = _asked(self.classes, classes)
            text: str | None = None
            one: str | None = None
        else:
            asked, wanted, text, one = _asked_open(prompt, classes)
        if threshold is not None:
            if self._backend == "vlm":
                raise ValueError("the VLM writes boxes without a score, so a threshold has nothing to cut; leave it "
                                 "out, or build the detector with backend='grounded_sam'")
            _checked_threshold(threshold)
        if segment and self._segmenter is None and self._load_segmenter is None:
            raise ValueError("segment=True needs SAM2, and this ObjectDetector was built without a load_segmenter; "
                             "from_weights and from_config give it one")
        height, width = image.shape[:2]
        with self._lock:
            finder, answering = self._finder_for(text)
            if answering == "vlm":
                cut: float | None = None
                found = finder.detect_all(image, text)
            else:
                cut = float(finder.threshold) if threshold is None else float(threshold)
                found = finder.detect_all(image, text, threshold=cut)
            if one is not None:
                found = [_relabelled(detection, one) for detection in found]
            objects = _objects(found, wanted, width=width, height=height)
            if segment and objects:
                cuts = self._cut(image, [obj.box for obj in objects])
                objects = [replace(obj, mask=mask, mask_score=score) for obj, (mask, score) in zip(objects, cuts)]
        _LOG.debug("%s found %d object(s) of %d class(es) %s in a %d x %d image%s", answering, len(objects), len(asked),
                   "with no score" if cut is None else f"above {cut:.2f}", width, height,
                   ", masks cut" if segment else "")
        return Detections(objects=tuple(objects), classes=asked, threshold=cut, image_hw=(height, width),
                          segmented=bool(segment), backend=answering, prompt=text)

    def _finder_for(self, text: str | None) -> tuple[Any, str]:
        """The model that answers ``text``, and its backend's name; a router's VLM is built at its first prompt."""
        if self._backend != "router":
            return self._detector, self._backend
        assert text is not None  # an open-vocabulary call always has its text
        decision = self._router.route(text)
        if str(getattr(decision.route, "value", decision.route)) != "vlm":
            return self._detector, "grounded_sam"
        if self._vlm is None:
            assert self._load_vlm is not None  # the constructor refuses a router without it
            _LOG.info("loading the VLM for the router's first prompt it sends there (%s)",
                      getattr(decision, "describe", lambda: "")())
            self._vlm = self._load_vlm()
        return self._vlm, "vlm"

    # ------------------------------------------------------------------ the masks
    def _segmenter_built(self) -> Sam2Segmenter:
        """The SAM2 segmenter, built by ``load_segmenter`` on the first call with a box to cut, then kept."""
        if self._segmenter is None:
            if self._load_segmenter is None:
                raise ValueError("segment=True needs SAM2, and this ObjectDetector was built without a load_segmenter")
            _LOG.info("loading SAM2 from %s for the first masks asked for", self.segmenter_source or "its loader")
            self._segmenter = self._load_segmenter()
        return self._segmenter

    def _cut(self, image: np.ndarray,
             boxes: Sequence[tuple[float, float, float, float]]) -> list[tuple[np.ndarray, float | None]]:
        """Each box's mask, and SAM2's predicted IoU for it, in the order of the boxes, from one SAM2 pass.

        The processor and the model are the ``Sam2Segmenter``'s, called as its ``segment`` calls them for one box, with
        every box of the image in the one call: the image is encoded once, and the decoder answers each box on its own.
        Of the masks SAM2 proposes for a box the first is kept, as ``segment`` keeps it, and cleaned as it cleans its
        one, so a mask cut here is the mask the cell's segmenter cuts for the same box.
        """
        import torch  # noqa: PLC0415 (loaded with the models, never on import)

        from src.models._inference import autocast_ctx  # noqa: PLC0415
        from src.utility.device import move_inputs_to_device  # noqa: PLC0415

        segmenter = self._segmenter_built()
        inputs = segmenter.processor(images=bgr_to_rgb(image), input_boxes=[[list(box) for box in boxes]],
                                     return_tensors="pt")
        inputs = move_inputs_to_device(dict(inputs), segmenter.device)
        optim = getattr(segmenter, "optim", None)
        if optim is not None and optim.channels_last and segmenter.device.type == "cuda":
            pixels = inputs.get("pixel_values")
            if torch.is_tensor(pixels) and pixels.dim() == 4:
                inputs["pixel_values"] = pixels.to(memory_format=torch.channels_last)
        # Under the dtype the segmenter resolved for its own pass, so the masks come out as its one-box pass cuts them.
        with torch.inference_mode(), autocast_ctx(segmenter.device, dtype=getattr(segmenter, "_model_dtype", None)):
            outputs = segmenter.model(**inputs)
        first = outputs.pred_masks[:, :, :1].float().cpu()
        scores = _first_scores(outputs, len(boxes))
        scale = segmenter.processor.image_processor.post_process_masks  # type: ignore[attr-defined]  # stub omits it
        kernel = int(getattr(segmenter, "morph_kernel_size", 5))
        largest = bool(getattr(segmenter, "keep_largest_component", True))
        cut: list[tuple[np.ndarray, float | None]] = []
        for start in range(0, len(boxes), _SCALED_AT_ONCE):
            scaled = scale(first[:, start:start + _SCALED_AT_ONCE], inputs["original_sizes"])[0]
            for offset in range(int(scaled.shape[0])):
                raw = np.asarray(scaled[offset, 0].cpu().numpy(), dtype=bool)
                cut.append((_cleaned(raw, kernel=kernel, keep_largest=largest), scores[start + offset]))
        return cut


# ---------------------------------------------------------------------------------------------------------------------
# Building the two models
# ---------------------------------------------------------------------------------------------------------------------


def _rtdetr(config: ObjectDetectorConfig, *, device: str | None) -> RtDetrObjectDetector:
    """RT-DETR from ``config``, on ``device`` or where every model here runs."""
    from src.models.detection.closed_set.detector import RtDetrObjectDetector  # noqa: PLC0415 (torch)

    return RtDetrObjectDetector(config, device=device)


def _sam2(config: SegmenterConfig, *, device: str | None) -> Sam2Segmenter:
    """SAM2 from ``config``, moved to ``device`` when one is named and it was built elsewhere.

    ``Sam2Segmenter`` places itself where every model here runs and takes no device, so a named one is reached by
    moving the built model there. ``from_weights`` builds it in full precision, so it moves as it is.
    """
    from src.models.segmentation.realtime.segmenter import Sam2Segmenter  # noqa: PLC0415 (torch)
    from src.utility.device import get_device  # noqa: PLC0415

    built = Sam2Segmenter(config)
    if device is not None:
        target = get_device(device)
        if built.device != target:
            built.model = built.model.to(target)
            built.device = target
    return built


def _sam2_of_the_tree(block: SegmenterConfig) -> Sam2Segmenter:
    """The SAM2 ``models.segmenter`` names, refused naming the key when its ``local`` folder holds none."""
    _checked_block(block, key="models.segmenter.model_path", kind="segmenter", fetch="sam2")
    return _sam2(block, device=None)


def _tree_backend(models: Any) -> Backend:
    """The backend a cell builds from ``models``, read as ``src.models.factory.build_perception`` reads it.

    ``models.pipeline`` decides where it is set: ``kind: closed_set`` is RT-DETR, ``zero_shot.backend: vlm`` the VLM,
    behind the router where ``router.enabled``, and any other zero-shot backend GroundingDINO. A tree without the block
    reads the legacy ``models.detector``: ``rtdetr`` is the closed set, anything else GroundingDINO.
    """
    pipeline = getattr(models, "pipeline", None)
    if pipeline is None:
        return "closed_set" if str(getattr(models, "detector", "")) == "rtdetr" else "grounded_sam"
    if pipeline.kind == "closed_set":
        return "closed_set"
    if pipeline.zero_shot.backend == "vlm":
        return "router" if pipeline.router.enabled else "vlm"
    return "grounded_sam"


def _vlm_block(models: Any) -> Any:
    """``models.pipeline.zero_shot.vlm``, or a refusal naming the key when the tree has no pipeline to hold one."""
    pipeline = getattr(models, "pipeline", None)
    block = getattr(getattr(pipeline, "zero_shot", None), "vlm", None)
    if block is None:
        raise ValueError("models.pipeline.zero_shot.vlm is not set, and the VLM is built from it: add a "
                         "models.pipeline block (config/models/object.yaml has one)")
    return block


def _vlm_grounder(block: Any) -> Any:
    """The process's one VLM grounder for ``block``, configured as a cell's build configures it.

    The copy the console and every cell share (``shared_vlm()``): a block naming other weights replaces the copy held,
    and the same weights are taken as they are, loaded or not. It loads at its first answer unless ``preload`` asks
    for it now; how it writes its answers is the block's, set without loading anything.
    """
    from src.models.vlm.holder import shared_vlm  # noqa: PLC0415 (torch)

    grounder = shared_vlm().grounder_for(block)
    grounder.configure(prompt_lookup_tokens=block.prompt_lookup_tokens,
                       text_prompt_lookup_tokens=block.text_prompt_lookup_tokens,
                       stop_at_answer_end=block.stop_at_answer_end, decode_graphs=block.decode_graphs)
    if block.preload:
        grounder.ensure_loaded()
    return grounder


def _grounding_dino(block: ObjectDetectorConfig) -> Any:
    """GroundingDINO from ``block``, where every model here runs."""
    from src.models.detection.zero_shot.detector import GroundingDinoObjectDetector  # noqa: PLC0415 (torch)

    return GroundingDinoObjectDetector(block)


def _rule_based_router() -> Any:
    """The prompt router a cell's routed stack uses: rule based, deterministic, loading no model."""
    from src.models.routing import RuleBasedRouter  # noqa: PLC0415

    return RuleBasedRouter()


def _models_block(config: Any) -> Any:
    """The models block of a loaded tree, of an ``AppConfig``, or the block itself."""
    if hasattr(type(config), "app_config"):  # a loaded tree; one that did not load raises its ConfigError here
        config = config.app_config
    models = getattr(config, "models", config)
    if not (hasattr(models, "rtdetr") and hasattr(models, "segmenter")):
        raise TypeError(f"from_config takes a loaded tree, its AppConfig or its models block, not "
                        f"{type(config).__name__}")
    return models


def _source_of(block: Any) -> str:
    """Where a config block's weights are read from: its folder when ``local``, else its Hub id."""
    return str(block.model_path if block.local else (block.model_id or block.model_path))


def _weights_source(value: str | os.PathLike[str] | None, *, kind: str, local: bool | None) -> tuple[str, bool]:
    """Where a checkpoint of ``kind`` is read from, and whether from this machine alone.

    A folder is read as it is, once it holds a checkpoint of that kind. A Hugging Face id is read from the folder the
    fetch script writes it to when that is there, as training reads its base model, else from the Hugging Face cache, or
    the Hub unless ``local`` is ``True``. Anything else is refused, naming the folder it looked for.
    """
    what = "object detector" if kind == "detector" else "SAM2"
    if value is None:
        raise TypeError(f"no {what} weights were named; pass a folder or a Hugging Face id"
                        + (" (a DetectorTrainingReport's model_dir is None when its run wrote no model, and printing "
                           "the report says why)" if kind == "detector" else ""))
    text = os.fspath(value)
    folder = Path(text).expanduser()
    if folder.is_dir():
        folder = folder.resolve()
        _check_folder(folder, kind=kind)
        return folder.as_posix(), True
    if folder.exists():
        raise FileNotFoundError(f"{folder.resolve()} is a file; name the folder that holds config.json and the weights")
    hub_id = text.strip()
    if isinstance(value, os.PathLike) or not _HUB_ID.fullmatch(hub_id) or ".." in hub_id:
        raise FileNotFoundError(f"there is no folder at {folder.resolve()}, so there is no {what} to load")
    fetched = weights_root() / _FETCHED[kind] / hub_id.replace("/", "--")
    if (fetched / "config.json").is_file():
        _check_folder(fetched, kind=kind)
        return fetched.as_posix(), True
    return hub_id, bool(local)


def _checked_block(block: Any, *, key: str, kind: str, fetch: str) -> None:
    """Refuse a ``local`` config block whose folder holds no checkpoint of ``kind`` (``detector``, ``grounder`` or
    ``segmenter``), naming the key and the fetch."""
    if not block.local:
        return
    folder = Path(block.model_path)
    if not folder.is_dir():
        raise FileNotFoundError(
            f"{key} is {folder}, and there is no folder there.{path_note(block.model_path)} Fetch it with "
            f"`python scripts/model_weights/fetch.py {fetch}`"
            + (", or point the key at a DetectorTraining out_dir" if kind == "detector" else "")
            + f", or set local: false to read {block.model_id or 'model_id'} from the Hugging Face Hub")
    _check_folder(folder, kind=kind)


def _check_folder(folder: Path, *, kind: str) -> None:
    """Refuse ``folder`` unless it holds a checkpoint of ``kind``, saying what it lacks or what it holds instead."""
    problem = _checkpoint_problem(folder, kind=kind)
    if problem is None:
        return
    text, wrong_kind = problem
    if wrong_kind:
        raise ValueError(f"{folder} holds {text}")
    if kind == "grounder":
        raise FileNotFoundError(f"{folder} holds {text}, so there is no GroundingDINO to load: "
                                f"`python scripts/model_weights/fetch.py dino-tiny` writes one")
    if kind == "detector":
        last = folder / "last"
        if last.is_dir() and _checkpoint_problem(last, kind=kind) is None:
            raise FileNotFoundError(f"{folder} holds no best epoch ({text}); its training run kept the last epoch in "
                                    f"{last}: pass that folder to load it")
        raise FileNotFoundError(f"{folder} holds {text}, so there is no object detector to load. A DetectorTraining "
                                f"out_dir holds its best epoch at the top and its last in last/, and any folder "
                                f"save_pretrained wrote an RT-DETR model and its image processor to loads as well")
    raise FileNotFoundError(f"{folder} holds {text}, so there is no SAM2 to load: "
                            f"`python scripts/model_weights/fetch.py sam2` writes one, and any folder save_pretrained "
                            f"wrote a SAM2 model and its processor to loads as well")


def _checkpoint_problem(folder: Path, *, kind: str) -> tuple[str, bool] | None:
    """What keeps ``folder`` from being a checkpoint of ``kind``, and whether it holds another kind; else ``None``.

    Read from the files alone, before anything loads: ``config.json``, a weights file, and for a detector the image
    processor's ``preprocessor_config.json``. ``config.json`` says what the checkpoint is: a detector's architecture
    ends in ``ForObjectDetection``, a SAM2's model type starts with ``sam2``.
    """
    config_file = folder / "config.json"
    if not config_file.is_file():
        return "no config.json", False
    if not any((folder / name).is_file() for name in _WEIGHT_FILES):
        return "a config.json and no weights (model.safetensors or pytorch_model.bin)", False
    import json  # noqa: PLC0415

    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "a config.json that is no JSON", False
    if not isinstance(config, dict):
        return "a config.json that is no JSON object", False
    architectures = [str(name) for name in config.get("architectures") or ()]
    model_type = str(config.get("model_type") or "")
    if kind in ("detector", "grounder"):
        if architectures and not any(name.endswith("ForObjectDetection") for name in architectures):
            return f"a {architectures[0]} checkpoint, which is no object detector", True
        if not (folder / "preprocessor_config.json").is_file():
            return "the model and no preprocessor_config.json, which its image processor is read from", False
    elif model_type and not model_type.startswith("sam2"):
        return f"a {model_type} checkpoint, which is no SAM2", True
    return None


# ---------------------------------------------------------------------------------------------------------------------
# The small rules
# ---------------------------------------------------------------------------------------------------------------------


def _checked_threshold(value: float) -> float:
    """``value`` as a threshold, refused unless it is a score from 0 to 1."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"a threshold is a score from 0 to 1, got {value!r}")
    return float(value)


def _asked(known: tuple[str, ...], classes: Iterable[str] | None) -> tuple[tuple[str, ...], frozenset[str] | None]:
    """The classes looked for, in id order, and the names they are matched by; ``None`` for every class.

    Refuses a name the model does not know, naming every class it knows, and an empty list, which names no class.
    """
    if classes is None:
        return known, None
    names = [classes] if isinstance(classes, str) else [str(name) for name in classes]
    if not names:
        raise ValueError("classes= names no class; leave it out to detect every class the model knows")
    keys = {_name_key(name) for name in known}
    unknown = [name for name in names if _name_key(name) not in keys]
    if unknown:
        raise ValueError(f"the detector knows no class {', '.join(repr(name) for name in unknown)}; it knows "
                         f"{len(known)}: {', '.join(known)}")
    wanted = frozenset(_name_key(name) for name in names)
    return tuple(name for name in known if _name_key(name) in wanted), wanted


def _asked_open(prompt: str | None, classes: Iterable[str] | None,
                ) -> tuple[tuple[str, ...], frozenset[str], str, str | None]:
    """What an open-vocabulary call asks: the classes looked for, the names they are matched by, the text the model
    reads, and the one class every box is found as, where the call names one thing.

    Exactly one of ``prompt`` and ``classes``. A prompt is free text, and every box found for it is that prompt's;
    a prompt holding a class list (``"red cube | blue bin"``) is read as one. ``classes`` are descriptions found in
    one call, joined as the cell joins them (``class_list_prompt``), each box under the description the model gave it;
    one description is a prompt. Refuses neither, both, an empty prompt, and what ``class_list_prompt`` refuses.
    """
    from src.models.vlm.qwen import class_list_prompt, classes_of  # noqa: PLC0415 (imports no torch)

    if prompt is not None and classes is not None:
        raise ValueError("give a prompt or classes, not both: a prompt is one thing to find, classes are several")
    if prompt is None and classes is None:
        raise ValueError("an open-vocabulary detector knows no fixed class list: give a prompt (free text, such as "
                         "'the red cube') or the classes to find (classes=['red cube', 'blue bin'])")
    if prompt is not None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"a prompt is the text to find, and {prompt!r} names nothing")
        listed = classes_of(prompt)
        text = prompt.strip() if not listed else class_list_prompt(listed)
    else:
        assert classes is not None
        text = class_list_prompt([classes] if isinstance(classes, str) else [str(name) for name in classes])
        listed = classes_of(text)
    asked = listed if listed else (text,)
    return asked, frozenset(_name_key(name) for name in asked), text, (None if listed else text)


def _relabelled(detection: Detection, label: str) -> Detection:
    """``detection`` found as ``label``: a call that names one thing finds every box as that thing."""
    return replace(detection, label=label)


def _objects(found: Iterable[Detection], wanted: frozenset[str] | None, *, width: int,
             height: int) -> list[DetectedObject]:
    """The detections of the classes looked for, one class per object, their boxes clipped to the image, the highest
    score first.

    RT-DETR scores each class of each of its queries on its own (``post_process_object_detection`` takes the best
    query-class pairs), so one query can clear the threshold under two classes, the box gathered from the same row and
    so the same to the bit. Such an object keeps its best class alone, and only then are the classes looked for picked
    out, so an object the model takes for another class never comes back as one of them. A box wholly outside the
    image keeps nothing once clipped, and is dropped.
    """
    best: dict[tuple[float, ...], Detection] = {}
    for detection in found:
        key = tuple(float(value) for value in detection.box)
        held = best.get(key)
        if held is None or float(detection.score) > float(held.score):
            best[key] = detection
    kept: list[DetectedObject] = []
    for detection in best.values():
        if wanted is not None and _name_key(detection.label) not in wanted:
            continue
        x0, y0, x1, y1 = (float(value) for value in detection.box)
        x0, x1 = min(max(x0, 0.0), float(width)), min(max(x1, 0.0), float(width))
        y0, y1 = min(max(y0, 0.0), float(height)), min(max(y1, 0.0), float(height))
        if x1 > x0 and y1 > y0:
            kept.append(DetectedObject(label=detection.label, score=float(detection.score), box=(x0, y0, x1, y1)))
    kept.sort(key=lambda obj: obj.score, reverse=True)
    return kept


def _first_scores(outputs: Any, count: int) -> list[float | None]:
    """SAM2's predicted IoU of the first mask of each box, or ``None`` for each where the output carries none."""
    import torch  # noqa: PLC0415

    scores = getattr(outputs, "iou_scores", None)
    if not torch.is_tensor(scores) or scores.dim() != 3 or scores.shape[1] != count or scores.shape[2] == 0:
        return [None] * count
    return [float(value) for value in scores[0, :, 0].detach().float().cpu().tolist()]


def _cleaned(raw: np.ndarray, *, kernel: int, keep_largest: bool) -> np.ndarray:
    """``raw`` cleaned as ``Sam2Segmenter`` cleans its one mask: opened, closed, and its largest piece kept.

    The same steps under the segmenter's own two settings, so a mask from a batch is the mask its one-box pass leaves;
    a test holds the two together.
    """
    binary: np.ndarray = np.asarray(raw, dtype=np.uint8)
    shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, kernel), max(1, kernel)))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, shape)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, shape)
    if keep_largest:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if count > 1:
            binary = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return binary.astype(bool)


def _image(value: np.ndarray | str | os.PathLike[str]) -> np.ndarray:
    """``value`` as an OpenCV BGR image, height x width x 3 ``uint8``: the array itself, or the file at a path."""
    if isinstance(value, (str, os.PathLike)):
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"there is no image at {path.resolve()}")
        # Read as bytes, since cv2.imread opens no path outside the ANSI code page on Windows.
        decoded = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"{path} is no image OpenCV can read")
        return decoded
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or image.size == 0:
        raise ValueError(f"an image is height x width x 3 uint8 in BGR order, as OpenCV gives it, or an image file's "
                         f"path; got {image.dtype} of shape {image.shape}")
    return np.ascontiguousarray(image)


def _four(values: Iterable[Any]) -> tuple[float, float, float, float]:
    """A box as four finite floats ``(x0, y0, x1, y1)`` with ``x1 > x0`` and ``y1 > y0``, or a refusal."""
    box = tuple(float(value) for value in values)
    if len(box) != 4 or not all(math.isfinite(value) for value in box) or not (box[2] > box[0] and box[3] > box[1]):
        raise ValueError(f"a box is four finite numbers (x0, y0, x1, y1) with x1 > x0 and y1 > y0, got {values!r}")
    return box[0], box[1], box[2], box[3]


def _name_key(name: str) -> str:
    """A class name as names are matched: case and blanks aside, ``" White  King "`` as ``"white king"``."""
    return " ".join(str(name).split()).casefold()


def _listed(names: Sequence[str]) -> str:
    """The first names, in ASCII, and how many more there are."""
    shown = ", ".join(_ascii(name) for name in names[:_NAMED])
    return shown + (f", ... (+{len(names) - _NAMED} more)" if len(names) > _NAMED else "")


def _ascii(text: str) -> str:
    """``text`` with every character outside ASCII written as its escape, for a Windows console under cp1252."""
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


def _write_label(image: np.ndarray, text: str, origin: tuple[int, int], colour: tuple[int, int, int]) -> None:
    """``text`` at ``origin`` in ``colour``, outlined in black so it reads on any image."""
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
