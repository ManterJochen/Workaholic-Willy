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
"""

from __future__ import annotations

import json
import math
import re
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
    "VLM_NOMINAL_SCORE",
    "CoordinateSpace",
    "GRID_SIZE",
    "parse_grounding_response",
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
#: this parser was asserted by the model: dropping it on a threshold meant for GroundingDINO's
#: logits would discard the route's entire output.
VLM_NOMINAL_SCORE = 1.0

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
    :class:`Detection` rejects outright.
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
