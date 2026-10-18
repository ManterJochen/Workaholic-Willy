"""Turn a vision-language model's free-form answer into validated :class:`Detection` boxes.

The half of the VLM route that runs without a GPU. A grounding VLM returns text, not a tensor, and
that text may be prose, a markdown fence, an apology, malformed JSON, or well-formed JSON that
still describes a nonsensical box.

The contract parsed here is Qwen's grounding format,

    [{"bbox_2d": [x0, y0, x1, y1], "label": "red cube"}, ...]

in whichever coordinate space the model was trained to emit. That space is an explicit parameter
(:class:`CoordinateSpace`) and never a guess, because guessing it wrong is silent: on a 1280x720
frame, 0-1000 grid values are in range, ordered, sane-sized, and pass every validation below, so
nothing fails and the gripper goes to the wrong place.

`Qwen3-VL-4B-Instruct` emits a 0-1000 normalised grid, not pixels. Asked to ground objects in a
1280x720 frame it returns coordinates of ``999`` and ``1000``, which cannot be pixels of a
720-pixel-tall image; and on a 640x480 frame with a square at ``(100, 100, 200, 200)`` it answers
``(144, 198, 312, 418)``, where that square on a 0-1000 grid is ``(156, 208, 312, 417)``.

Every rejection below is a box that would otherwise become a grasp pose, and a mangled box does not
throw; it moves the gripper somewhere wrong. So the parser drops rather than repairs wherever
repair would mean guessing:

* Coordinates outside the declared space are not reinterpreted. Under
  :attr:`CoordinateSpace.ABSOLUTE`, a value like ``0.1`` clamps to a sub-pixel box and is dropped;
  reading it as a fraction instead would be the same guess, made per box.
* Inverted corners are repaired by swapping, because ``x1 < x0`` has exactly one possible intent.
* Out-of-image boxes are clamped, then re-checked for degeneracy, so a box entirely off-frame
  collapses and is dropped.

A grounding VLM emits no calibrated score. :data:`VLM_NOMINAL_SCORE` means "the model asserted
this", not "the model is this sure", and the model's own output order is kept as its ranking.
Nothing downstream should read it as a probability.

The boxes of a class list (several descriptions, the sorting package of 2026-10-09), each description
grounded on its own since 2026-10-15, fail closed where a box is no one description's: every box of a
group that overlaps by :data:`AMBIGUOUS_IOU` or more under more than one description is labelled
:data:`AMBIGUOUS_LABEL` (:func:`claimed_once`). Such a box stays an obstacle and is never a target,
since no description is that word.

A box that repeats one written before it under the same description (:data:`REPEAT_IOU`) is the same
object written again and is dropped: a model that loops writes it again and again, and on the cell
(2026-10-12) Qwen answered 52 boxes of 4 objects for "each separate gray cube in the blue bin".

A model can loop without repeating a box too: on the cell's Thursday (2026-10-15) Qwen walked across the
picture box by box, each the size of the one before, either a fixed step further (a ladder of 40 boxes
down a diagonal) or edge to edge (a tiling of 32 boxes over a pile), until its tokens ran out. Those
boxes are dropped (:func:`without_walks`, :data:`LADDER_RUN`, :data:`TILING_BOXES`), and so is a box
that holds two others (:func:`without_groups`): a box drawn round a pile or a bin of parts, not a part.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import replace
from enum import StrEnum
from typing import Any

from src.models.constants import MODELS_LOG_DIR, VLM_PARSING_LOG_FILE
from src.models.detection.types import Detection
from src.utility.log_cfg import create_logger

#: The parser's own file, separate from the model wrapper's. ``qwen.py`` logs what the model
#: answered; these lines log which boxes this parser refused and why. Every refusal is a grasp pose
#: that will not happen, so "the model said nothing" and "the model said something unusable" must
#: not look the same afterwards. Module scope because this module is functions, not a class.
_LOG = create_logger("VLMGroundingParser", log_file=VLM_PARSING_LOG_FILE, log_dir=MODELS_LOG_DIR)

__all__ = [
    "AMBIGUOUS_IOU",
    "AMBIGUOUS_LABEL",
    "GROUP_INSIDE",
    "LADDER_RUN",
    "REPEAT_IOU",
    "SAME_SIZE_PX",
    "SAME_STEP_PX",
    "TILING_BOXES",
    "VLM_NOMINAL_SCORE",
    "CoordinateSpace",
    "GRID_SIZE",
    "box_iou",
    "claimed_once",
    "parse_grounding_response",
    "walked",
    "without_groups",
    "without_walks",
    "extract_json_payload",
]

#: The normalised grid Qwen grounds on. The model's convention, not a value to be tuned.
GRID_SIZE = 1000.0


class CoordinateSpace(StrEnum):
    """What a model's box numbers mean. Declared per backend, never inferred per box.

    Inferring would need a rule like "if any value exceeds the image width, assume a grid", which
    silently misreads every small object in a large frame and is undetectable downstream, because
    the box it produces is well-formed.
    """

    ABSOLUTE = "absolute"    #: pixels of the image as submitted
    GRID_1000 = "grid_1000"  #: Qwen's 0-1000 normalised grid (measured for Qwen3-VL, see module docs)

#: The score stamped on every VLM detection. Not a confidence; see the module docstring. It is 1.0
#: because downstream filters compare against a detector's box threshold, and a box that survived
#: this parser was asserted by the model: dropping it on a threshold meant for a scoring detector's
#: logits would discard the VLM's entire output.
VLM_NOMINAL_SCORE = 1.0

#: The label a box of a class list is given where no one description clearly claims it: the model
#: named it by none, or boxed one object under two descriptions. No description is this word (a class
#: list refuses one that holds it, ``src.models.vlm.qwen.class_list_prompt``), so no rule takes such a
#: box as its part: it stays an obstacle every grasp is planned around and is never a target. Sorting
#: places no part by guess (the owner, 2026-10-09).
AMBIGUOUS_LABEL = "ambiguous"

#: How much two boxes of a class list overlap, as intersection over union, to be read as one object.
#: Two parts side by side overlap far less, and one object boxed twice far more. A choice, not a
#: measurement.
AMBIGUOUS_IOU = 0.7

#: How much a box overlaps one written before it under the same description, as intersection over union, to be that
#: object written again. Two parts side by side, even pressed together in a crate, overlap far less; the loop the cell
#: met wrote two boxes of one cube in turn, 0.87 apart (2026-10-12), and the owner drops a box 0.6 like another
#: (2026-10-15). A choice, not a measurement.
REPEAT_IOU = 0.6
#: Boxes written one after the other under one description, each the size of the one before (:data:`SAME_SIZE_PX`) and
#: placed by the step before it (:data:`SAME_STEP_PX`), this many in a run, are a model walking across the picture: the
#: ladders of the cell's Thursday (2026-10-15). Real parts are named in no such order.
LADDER_RUN = 5
#: Boxes of one description, each the size of the next, joined edge to edge into one patch this many or more, are a
#: model tiling a pile (2026-10-15): real parts' boxes do not meet along whole edges in such numbers.
TILING_BOXES = 6
#: Two boxes are the same size where their widths and their heights each differ by at most this, pixels.
SAME_SIZE_PX = 5.0
#: Two steps are the same, and two edges meet, within this, pixels.
SAME_STEP_PX = 3.0
#: A box holding at least two other boxes, each with this share of its area inside it and at most half its size, is a
#: box round a pile or a bin of parts, not a part (the owner, 2026-10-15: Thursday's ten group boxes). One object
#: boxed twice holds nothing.
GROUP_INSIDE = 0.8

#: ```json ... ``` or bare ``` ... ``` fences, which instruct-tuned models add unprompted.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

#: The outermost JSON array or object embedded in prose, as in "Here are the objects: [...]"
#: followed by a closing remark.
_ARRAY = re.compile(r"\[.*\]", re.DOTALL)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
#: One flat JSON object, as each box of a grounding answer is: no brace inside it. What an answer cut off mid-list
#: still holds whole.
_FLAT_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)

#: Box-field key names accepted from an answer. ``bbox_2d`` is Qwen's documented name; the others
#: cost one tuple entry each and save a box that would otherwise be dropped.
_BOX_KEYS = ("bbox_2d", "bbox", "box_2d", "box")
_LABEL_KEYS = ("label", "text", "name", "category")


def extract_json_payload(text: str) -> Any | None:
    """Recover the JSON value from a model answer, or ``None`` if there is none.

    Tries progressively less strict readings: the whole string, then a fenced block, then the
    outermost bracketed span embedded in prose. Never raises; unparseable output is a legitimate
    model failure, not an exception.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    candidates: list[str] = [text.strip()]
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    for pattern in (_ARRAY, _OBJECT):
        found = pattern.search(text)
        if found:
            candidates.append(found.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (ValueError, TypeError):
            continue
    return _whole_objects_of_a_cut_list(text)


def _whole_objects_of_a_cut_list(text: str) -> list[dict[str, Any]] | None:
    """The whole objects of a list the answer opened and never closed, or ``None`` where it is no such answer.

    A model that runs out of tokens stops mid-list: ``[{...}, {...}, {"bbox_2d": [1, 2`` reads as no JSON at all, and a
    pile of parts came back as no part. Each object written whole is what the model asserted, exactly; only the cut
    one is left out, so nothing is guessed. Asked only where no reading of the answer parsed; an answer that opens no
    list is not read so.
    """
    start = text.find("[")
    if start < 0:
        return None
    objects: list[dict[str, Any]] = []
    for found in _FLAT_OBJECT.finditer(text, start):
        try:
            value = json.loads(found.group(0))
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            objects.append(value)
    if not objects:
        return None
    _LOG.warning("the grounding answer stops mid-list (out of tokens): the %d whole object(s) in it are kept, the cut "
                 "one is left out", len(objects))
    return objects


def _as_items(payload: Any) -> list[dict[str, Any]]:
    """The payload as a list of dicts; a lone object becomes a one-element list."""
    if isinstance(payload, dict):
        # Some answers wrap the list: {"objects": [...]} / {"detections": [...]}
        for key in ("objects", "detections", "results", "boxes", "items"):
            nested = payload.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _box_from(item: dict[str, Any]) -> list[float] | None:
    """The four corner values, or ``None`` if this item has no usable box."""
    for key in _BOX_KEYS:
        raw = item.get(key)
        if raw is None:
            continue
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            continue
        try:
            values = [float(v) for v in raw]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in values):
            continue
        return values
    return None


def _label_from(item: dict[str, Any], fallback: str) -> str:
    for key in _LABEL_KEYS:
        raw = item.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return fallback


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Intersection over union of two ``[x0, y0, x1, y1]`` boxes; 0 for boxes that do not overlap or have no area.

    Args:
        first (Sequence[float]): One box, ``[x0, y0, x1, y1]``.
        second (Sequence[float]): The other, in the same space.

    Returns:
        float: The shared area over the area the two cover together, from 0 to 1.
    """
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    if width <= 0.0 or height <= 0.0:
        return 0.0
    inter = width * height
    union = ((first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1])
             - inter)
    return inter / union if union > 0.0 else 0.0


def _same_size(first: Sequence[float], second: Sequence[float]) -> bool:
    return (abs((first[2] - first[0]) - (second[2] - second[0])) <= SAME_SIZE_PX
            and abs((first[3] - first[1]) - (second[3] - second[1])) <= SAME_SIZE_PX)


def _meet(first: Sequence[float], second: Sequence[float]) -> bool:
    """Whether two boxes meet along a whole edge: one's right on the other's left, or one's bottom on the other's top,
    each spanning the same rows (or columns), all within :data:`SAME_STEP_PX` and :data:`SAME_SIZE_PX`."""
    for a, b in ((first, second), (second, first)):
        if (abs(a[2] - b[0]) <= SAME_STEP_PX and abs(a[1] - b[1]) <= SAME_SIZE_PX
                and abs(a[3] - b[3]) <= SAME_SIZE_PX):
            return True
        if (abs(a[3] - b[1]) <= SAME_STEP_PX and abs(a[0] - b[0]) <= SAME_SIZE_PX
                and abs(a[2] - b[2]) <= SAME_SIZE_PX):
            return True
    return False


def _touch(first: Sequence[float], second: Sequence[float]) -> bool:
    """Whether two boxes overlap or meet, within :data:`SAME_STEP_PX`: a model's walk, where real parts in a row leave
    room between their boxes."""
    return (first[0] <= second[2] + SAME_STEP_PX and second[0] <= first[2] + SAME_STEP_PX
            and first[1] <= second[3] + SAME_STEP_PX and second[1] <= first[3] + SAME_STEP_PX)


def walked(boxes: Sequence[Sequence[float]]) -> list[bool]:
    """Which of ``boxes`` (``[x0, y0, x1, y1]``, in the order written) a model walked across the picture with: a run of
    :data:`LADDER_RUN` or more written one after the other, each touching the one before and either placed by the step
    before it (a ladder) or spanning its very columns or rows (a walk down a column or along a row), or a patch of
    :data:`TILING_BOXES` or more, each the size of one it meets edge to edge (a tiling). Parts set out in rows with room
    between them are none of these. The caller hands the boxes of one description.

    Args:
        boxes (Sequence[Sequence[float]]): The boxes, in the order the model wrote them.

    Returns:
        list[bool]: One flag per box, ``True`` where it is a step of a walk.
    """
    count = len(boxes)
    out = [False] * count
    start = 0  # the first box of the run of steps that ends at the box looked at
    for index in range(2, count):
        before, last, box = boxes[index - 2], boxes[index - 1], boxes[index]
        if not (_touch(before, last) and _touch(last, box)
                and all(abs((box[k] - last[k]) - (last[k] - before[k])) <= SAME_STEP_PX for k in range(4))):
            start = index - 1
            continue
        if index - start + 1 >= LADDER_RUN:
            out[start:index + 1] = [True] * (index + 1 - start)
    # A walk down a column or along a row: each box spanning the very columns, or the very rows, of the one before.
    start = 0
    for index in range(1, count):
        last, box = boxes[index - 1], boxes[index]
        if not _touch(last, box) or not any(abs(box[low] - last[low]) <= SAME_STEP_PX
                                            and abs(box[high] - last[high]) <= SAME_STEP_PX
                                            for low, high in ((0, 2), (1, 3))):
            start = index
            continue
        if index - start + 1 >= LADDER_RUN:
            out[start:index + 1] = [True] * (index + 1 - start)
    patch = list(range(count))

    def root(index: int) -> int:
        while patch[index] != index:
            patch[index] = patch[patch[index]]
            index = patch[index]
        return index

    for first in range(count):
        for second in range(first + 1, count):
            if _same_size(boxes[first], boxes[second]) and _meet(boxes[first], boxes[second]):
                patch[root(second)] = root(first)
    sizes: dict[int, int] = {}
    for index in range(count):
        sizes[root(index)] = sizes.get(root(index), 0) + 1
    for index in range(count):
        if sizes[root(index)] >= TILING_BOXES:
            out[index] = True
    return out


def without_walks(detections: list[Detection]) -> tuple[list[Detection], int]:
    """``detections`` in their order, every box a model walked across the picture with dropped (:func:`walked`, the
    boxes of each description apart), and how many were dropped.

    Args:
        detections (list[Detection]): One answer's boxes, in the order written.

    Returns:
        tuple[list[Detection], int]: The boxes kept, and how many went.
    """
    dropped = [False] * len(detections)
    for words in {" ".join(detection.label.casefold().split()) for detection in detections}:
        members = [index for index, detection in enumerate(detections)
                   if " ".join(detection.label.casefold().split()) == words]
        for index, flag in zip(members, walked([detections[member].box for member in members])):
            dropped[index] = flag
    return [detection for detection, flag in zip(detections, dropped) if not flag], sum(dropped)


def without_groups(detections: list[Detection]) -> tuple[list[Detection], int]:
    """``detections`` in their order, every box that holds two others at most half its size dropped
    (:data:`GROUP_INSIDE`): a box round a pile or a bin of parts, whichever description drew it; and how many were
    dropped.

    Args:
        detections (list[Detection]): The boxes of one picture, every description's.

    Returns:
        tuple[list[Detection], int]: The boxes kept, and how many went.
    """

    def area(box: Sequence[float]) -> float:
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    def holds(outer: Sequence[float], inner: Sequence[float]) -> bool:
        width = min(inner[2], outer[2]) - max(inner[0], outer[0])
        height = min(inner[3], outer[3]) - max(inner[1], outer[1])
        return (0.0 < area(inner) <= 0.5 * area(outer)
                and max(0.0, width) * max(0.0, height) >= GROUP_INSIDE * area(inner))

    groups = [sum(1 for inner in detections if holds(detection.box, inner.box)) >= 2 for detection in detections]
    if any(groups):
        _LOG.warning("%d box(es) hold two others or more and are dropped, a box round a pile or a bin of parts: %s",
                     sum(groups), "; ".join(f"{detection.label!r} at {[round(value) for value in detection.box]}"
                                            for detection, group in zip(detections, groups) if group))
    return [detection for detection, group in zip(detections, groups) if not group], sum(groups)


def _without_repeats(detections: list[Detection]) -> tuple[list[Detection], int]:
    """The boxes in their order, every one dropped that repeats a box kept before it under the same description
    (:data:`REPEAT_IOU`, descriptions compared as words), and how many were dropped. An :data:`AMBIGUOUS_LABEL` box is
    an obstacle no rule takes, and every one is kept, as :func:`claimed_once` keeps them."""
    kept: list[Detection] = []
    for detection in detections:
        words = " ".join(detection.label.casefold().split())
        if detection.label != AMBIGUOUS_LABEL and any(" ".join(other.label.casefold().split()) == words and box_iou(detection.box, other.box) >= REPEAT_IOU
               for other in kept):
            continue
        kept.append(detection)
    return kept, len(detections) - len(kept)


def claimed_once(detections: list[Detection], prompt: str) -> list[Detection]:
    """The boxes of a class list (``prompt``), every box no one description clearly claims labelled
    :data:`AMBIGUOUS_LABEL`, in their order, none dropped.

    Boxes that overlap by :data:`AMBIGUOUS_IOU` or more, one with the next, are one object. Where such a group carries
    more than one label (compared as words, case and spacing aside), two descriptions claim that object, and every box
    of the group becomes ambiguous: neither rule may take the part, and the boxes stay where they are as obstacles. A
    chain counts as one group, so no box of a doubly claimed object is left under one of the two names. One description
    boxed twice keeps its label, as an answer to one description keeps it. The VLM applies it to the boxes it grounded
    for the descriptions of a class list one at a time (``Qwen3VLGrounder.detect_all``).
    """
    count = len(detections)
    group = list(range(count))

    def root(index: int) -> int:
        while group[index] != index:
            group[index] = group[group[index]]
            index = group[index]
        return index

    for first in range(count):
        for second in range(first + 1, count):
            if box_iou(detections[first].box, detections[second].box) >= AMBIGUOUS_IOU:
                group[root(second)] = root(first)
    said: dict[int, set[str]] = {}
    for index, detection in enumerate(detections):
        said.setdefault(root(index), set()).add(" ".join(detection.label.casefold().split()))
    claimed = [len(said[root(index)]) > 1 for index in range(count)]
    if not any(claimed):
        return detections
    # Warning level: each of these is a part the model saw and no rule may take.
    _LOG.warning(
        "class list %r: %d box(es) overlap (IoU >= %.2f) under more than one description, every one now %r, an "
        "obstacle and never a target: %s", prompt, sum(claimed), AMBIGUOUS_IOU, AMBIGUOUS_LABEL,
        "; ".join(f"{detection.label!r} at {[round(value) for value in detection.box]}"
                  for detection, doubly in zip(detections, claimed, strict=True) if doubly),
    )
    return [replace(detection, label=AMBIGUOUS_LABEL) if doubly else detection
            for detection, doubly in zip(detections, claimed, strict=True)]


def parse_grounding_response(
    text: str,
    *,
    image_width: int,
    image_height: int,
    fallback_label: str,
    space: CoordinateSpace = CoordinateSpace.GRID_1000,
) -> list[Detection]:
    """Validated detections from a grounding answer. Never raises; ``[]`` when nothing survives.

    ``space`` says what the numbers mean and defaults to the space measured for Qwen3-VL. A backend
    whose model emits pixels must say so explicitly; see :class:`CoordinateSpace` for why this is
    not detected automatically.

    ``fallback_label`` is used when the model names no label, normally the caller's prompt, so
    downstream label matching has something to match on rather than an empty string, which
    :class:`Detection` rejects outright. Boxes a model walked across the picture with are dropped
    (:func:`without_walks`), and so is a box that repeats one before it (:data:`REPEAT_IOU`).
    """
    payload = extract_json_payload(text)
    if payload is None:
        # Degraded, not empty: the model answered and none of it was JSON. Downstream that is
        # indistinguishable from "no such object", so the raw answer, truncated because it can run
        # long, is the only way to tell an apology from a hallucinated prose location.
        _LOG.warning("no JSON payload in grounding answer for %r (raw: %.200s)", fallback_label, text)
        return []

    detections: list[Detection] = []
    # Drops are counted per reason and reported once below, not per item: a wrong coordinate space
    # drops every box in the answer, and one counted line carries that as well as twenty identical
    # ones, while leaving the case of a few boxes lost out of many visible.
    dropped_no_box = 0
    dropped_degenerate = 0
    for item in _as_items(payload):
        values = _box_from(item)
        if values is None:
            dropped_no_box += 1
            continue
        if space is CoordinateSpace.GRID_1000:
            # Scale before clamping: clamping first pins grid values against the image bounds and
            # silently flattens every box in the right half of a wide frame.
            values = [
                values[0] / GRID_SIZE * image_width,
                values[1] / GRID_SIZE * image_height,
                values[2] / GRID_SIZE * image_width,
                values[3] / GRID_SIZE * image_height,
            ]
        x0, y0, x1, y1 = values
        # Inverted corners have one possible intent, so repair them; anything else would be a guess.
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        # Clamp into the frame, then re-check: a box entirely outside collapses here and is dropped
        # below, which is the correct outcome for a hallucinated location.
        x0 = min(max(x0, 0.0), float(image_width))
        x1 = min(max(x1, 0.0), float(image_width))
        y0 = min(max(y0, 0.0), float(image_height))
        y1 = min(max(y1, 0.0), float(image_height))
        if x1 - x0 < 1.0 or y1 - y0 < 1.0:
            # Sub-pixel after clamping: an off-frame box, or normalised coordinates that this
            # space deliberately does not rescale. Either way there is no box to send a gripper to.
            dropped_degenerate += 1
            continue
        detections.append(Detection(
            box=[x0, y0, x1, y1],
            x_center=(x0 + x1) / 2.0,
            y_center=(y0 + y1) / 2.0,
            label=_label_from(item, fallback_label),
            score=VLM_NOMINAL_SCORE,
        ))

    detections, repeated = _without_repeats(detections)
    if repeated:
        # Warning level: the model looped, and each of these would otherwise have been a part of its own.
        _LOG.warning("%d box(es) of %r repeat an earlier box under the same description (IoU >= %.2f) and are "
                     "dropped: the model wrote the same object again", repeated, fallback_label, REPEAT_IOU)
    # Once the repeats are gone: a part written again in turn with its twin is no walk.
    detections, walks = without_walks(detections)
    if walks:
        # Warning level: the model walked across the picture, and each of these would otherwise have been a part.
        _LOG.warning("%d box(es) of %r are a model walking across the picture (a ladder of %d or more, or a tiling of "
                     "%d or more, each box the size of the one before) and are dropped", walks, fallback_label,
                     LADDER_RUN, TILING_BOXES)

    if dropped_no_box or dropped_degenerate:
        # Warning level because each of these was a box the model asserted and this parser refused.
        # A run where `degenerate` equals the item count is the signature of a wrong `space`, the
        # silent failure the module docstring describes, so the space is on the line too.
        _LOG.warning(
            "kept %d box(es), dropped %d (no usable box) + %d (degenerate after clamp) "
            "for %r in %dx%d, space=%s",
            len(detections), dropped_no_box, dropped_degenerate,
            fallback_label, image_width, image_height, space.value,
        )
    else:
        # Debug level, not info: on the clean path the caller (`Qwen3VLGrounder.detect_all`)
        # already reports the outcome, and this line only adds the detail behind it.
        _LOG.debug(
            "parsed %d box(es) for %r in %dx%d, space=%s",
            len(detections), fallback_label, image_width, image_height, space.value,
        )
    return detections
