"""Keep what the looks of one pick saw, for training: one ``.npz`` per pick, off unless a program asks for it.

The owner, 2026-09-29 (addendum 7.7): an opt-in lever, ``record_views``, that stores per pick the frames of its looks
(RGB and depth), the tool pose stamped at each shutter, the intrinsics, and the target cloud the looks fused. Nothing
records unless asked: ``PickRun(record_views=True)`` writes one file per pick of a wrist camera, and
``Locator.look_around(..., record_views=True)`` one per look around, through :func:`record_views` alike.

Why not the corpus ``datagen`` already reads: a datagen scene is a folder of per-view PNGs with per-instance masks, a
scene payload and a ground-truth grasp table (``datagen/corpus/clouds.py``), none of which a real pick has. Inventing
them would put a corpus on disk that reads as ground truth and is not. So this is a small documented layout of its own,
under ``logs/``, which a converter can turn into whatever a trainer reads once one is written.

Layout of a views file (``numpy.savez_compressed``, read with ``np.load(path, allow_pickle=False)``), one pick:

``format``
    int, :data:`RECORD_VIEWS_FORMAT`. Bumped whenever an array is added, renamed or given a new meaning.
``labels``
    str (N,): each look as it reads in a report (``home``, or the joints in degrees), in the order taken; ``""`` where
    the caller named none.
``views``
    str (N,): each look's view identity (``rig@look``), ``""`` where the caller named none.
``rgb_<i>``
    uint8 (H, W, 3): look ``i``'s colour image, absent where the frame carried none.
``depth_<i>``
    float32 (H, W): look ``i``'s depth in millimetres along the optical axis, 0 where nothing was measured: the depth
    the grasp path was handed (``PerceptionFrame.depth_map``); for a Locator's look, the depth it placed its objects by
    (the measured surface, where the frame carried one).
``surface_depth_<i>``
    float32 (H, W): the depth the camera measured before any grasp-referenced overwrite
    (``PerceptionFrame.surface_depth_map``), absent where the frame carried none.
``intrinsics_<i>``
    float64 (3, 3): look ``i``'s camera matrix, pixels.
``tool_to_base_<i>``
    float64 (4, 4): the tool pose stamped at look ``i``'s shutter, millimetres in BASE; all NaN where the frame was not
    stamped.
``camera_to_base_<i>``
    float64 (4, 4): CAMERA to BASE of look ``i`` as the pick placed it (the stamped tool pose and the hand-eye),
    millimetres; all NaN where the caller had none.
``target_cloud_base_mm``
    float32 (M, 3): the target's cloud the looks fused, BASE millimetres; (0, 3) where there was none.

Format 2 (the owner, 2026-10-09: the research recording, "alles in einem Schalter") is format 1 and the arrays below, in
a file whose looks a camera recorded for research (``camera.cameras.rigs[<id>].realsense.record_for_research``: the
frame carries a ``ResearchCapture``, ``src/camera/setup/image_taking/frames.py``). A file none of whose looks carries
one is format 1, byte for byte the file it was before format 2, and says ``format`` 1. Every array of format 1 keeps its
name and its meaning in format 2, and no new name begins like an old one (``rgb_``, ``depth_``, ...), so a reader of
format 1 reads format 2 as it reads format 1. Per look ``i``:

``ir_left_<i>``, ``ir_right_<i>``
    uint8 (h, w): the left and the right infrared image (librealsense's infrared 1 and 2, Y8, at the depth stream's
    resolution) of the frameset look ``i``'s colour and depth came from, with the projector as the camera runs; absent
    where the frameset carried none.
``raw_depth_<i>``
    uint16 (h, w): the depth as the sensor streamed it, in device units (``camera_depth_units_m`` metres each, 0 where
    nothing was measured), before the filters and the alignment to colour, so in the left infrared camera's frame, the
    depth's own; ``depth_<i>`` is that depth filtered and aligned.
``ir_intrinsics_<i>``, ``ir_distortion_<i>``
    float64 (2, 3, 3) and (2, 5): the left and the right infrared camera's matrix, pixels, and distortion coefficients;
    the left one's is the depth stream's lens too, the depth being measured in its frame.
``ir_extrinsics_<i>``
    float64 (2, 4, 4): where the left infrared camera sits in the colour camera's frame and in the right infrared
    camera's, the device's extrinsics, millimetres: a point in the left infrared camera's frame maps to
    ``T @ (x, y, z, 1)``.
``color_metadata_<i>``
    float64 (4,): what the colour frame's metadata says of how it was exposed, ``actual_exposure`` (microseconds),
    ``gain_level``, ``white_balance`` (Kelvin), ``auto_exposure`` (0 off) (:data:`COLOR_METADATA`), as the device
    reports them; NaN where it reports none (on Windows a RealSense reports metadata once its driver is told to).
``seg_masks_<i>``
    uint8 (n, ceil(H * W / 8)): each of look ``i``'s n segmentations' masks, packed bits (``np.packbits``, row by row,
    the first pixel in the high bit), H, W the shape of ``depth_<i>``:
    ``np.unpackbits(m, axis=1, count=H * W).reshape(n, H, W).astype(bool)``.
``seg_boxes_<i>``
    int32 (n, 4): each segmentation's integer box, ``x0, y0, x1, y1`` pixels, the box the segmenter was prompted with;
    -1 where a segmentation carries none.
``seg_labels_<i>``
    str (n,): each segmentation's label as the frame carries it, after the label mapping and the colour check.
``seg_scores_<i>``
    float32 (n,): each segmentation's score, the detector's; NaN where none.
``seg_predicted_iou_<i>``
    float32 (n, 3): SAM2's predicted IoU of each of the three masks it proposed for the box, the first the one kept; NaN
    where the segmenter gave none.
``seg_target_<i>``
    int64: which of look ``i``'s segmentations is the part the pick went for, -1 where the look did not see it or the
    pick judged no part.

Once per file, of the camera the first look carrying a ``ResearchCapture`` was taken with:

``camera``
    str: JSON, what the camera was as it opened: the device's name, serial, firmware and USB link, the stream modes and
    each lens's distortion model, the depth units, the visual preset, the post-processing the rig asked for and each
    filter's options as it runs, every option of the depth sensor, and the colour sensor's exposure, gain and white
    balance as the rig asked for them and as the sensor held them once the warm-up frames had run, with every option
    it offers.
``camera_depth_units_m``
    float64: metres per raw depth unit, as the device reads it back.
``camera_color_intrinsics``, ``camera_color_distortion``, ``camera_depth_intrinsics``, ``camera_depth_distortion``
    float64 (3, 3) and (5,): the colour and the depth stream's camera matrix, pixels, and distortion coefficients, as
    the device reports them (``intrinsics_<i>`` is the matrix the pick used, the colour one unless a program overrode
    it).
``camera_depth_to_color``
    float64 (4, 4): where the depth's camera sits in the colour camera's frame, the device's extrinsics, millimetres: a
    point in the depth's frame maps to ``T @ (x, y, z, 1)`` in the colour camera's, the one the depth is aligned to.

A file is named after the pick (``name``, the service's ``attempt_id`` when ``PickRun`` writes it, which is the key of
the pick's record in the JSONL corpus), then the UTC time and a short random suffix, so two picks never overwrite each
other.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

__all__ = [
    "COLOR_METADATA",
    "RECORD_VIEWS_DIR",
    "RECORD_VIEWS_FORMAT",
    "RECORD_VIEWS_FORMAT_WITHOUT_RESEARCH",
    "record_views",
]

#: Where a pick's views are kept when nobody says otherwise, relative to the working directory, beside the robot's logs.
RECORD_VIEWS_DIR: Final[str] = "logs/robot/views"
#: What shape a views file is (the module docstring). Bumped whenever an array is added, renamed or given a new meaning:
#: 2 added the arrays of a camera recording for research (2026-10-09).
RECORD_VIEWS_FORMAT: Final[int] = 2
#: What a file none of whose looks a camera recorded for research says: format 1, byte for byte the file it was before
#: format 2, so with the research recording off nothing a reader sees changes.
RECORD_VIEWS_FORMAT_WITHOUT_RESEARCH: Final[int] = 1
#: The colour frame's metadata ``color_metadata_<i>`` holds, in its order.
COLOR_METADATA: Final[tuple[str, ...]] = ("actual_exposure", "gain_level", "white_balance", "auto_exposure")
#: The streams whose lenses a research file keeps once (``camera_<s>_intrinsics``); the infrared ones go with each look.
_LENSES: Final[tuple[str, ...]] = ("color", "depth")
#: The infrared lenses and extrinsics each look keeps, in the order of their arrays' first axis.
_INFRARED_LENSES: Final[tuple[str, ...]] = ("ir_left", "ir_right")
_INFRARED_EXTRINSICS: Final[tuple[str, ...]] = ("ir_left_to_color", "ir_left_to_ir_right")

#: What a file name keeps of the name it is given; everything else becomes ``_``.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _matrix(value: Any) -> np.ndarray:
    """``value`` as a (4, 4) float64 matrix: a transform or pose (``to_matrix``), an array, or all NaN for none."""
    if value is None:
        return np.full((4, 4), np.nan)
    to_matrix = getattr(value, "to_matrix", None)
    matrix = np.asarray(to_matrix() if callable(to_matrix) else value, dtype=np.float64)
    return matrix if matrix.shape == (4, 4) else np.full((4, 4), np.nan)


def _research_of(frame: Any) -> Any:
    """What a camera recorded for research with ``frame`` (``ResearchCapture``), else ``None``: a double that answers
    every attribute recorded nothing more."""
    from src.camera.setup.image_taking.frames import ResearchCapture  # noqa: PLC0415 - the camera package, on demand

    research = getattr(frame, "research", None)
    return research if isinstance(research, ResearchCapture) else None


def _packed_masks(segmentations: Sequence[Any], shape: tuple[int, ...]) -> np.ndarray:
    """Every segmentation's mask, packed (``seg_masks_<i>``); a mask of another shape than its frame's is refused."""
    size = int(shape[0]) * int(shape[1])
    rows = np.zeros((len(segmentations), (size + 7) // 8), dtype=np.uint8)
    for row, seg in enumerate(segmentations):
        mask = getattr(seg, "mask", None)
        if mask is None:
            continue
        mask = np.asarray(mask)
        if mask.shape[:2] != tuple(shape):
            raise ValueError(f"segmentation {row} carries a mask of {mask.shape} pixels in a frame of {tuple(shape)}")
        rows[row] = np.packbits(mask.astype(bool).reshape(-1))
    return rows


def _box_of(seg: Any) -> list[int]:
    """A segmentation's integer box, ``x0, y0, x1, y1``, or -1 four times where it carries none."""
    box = getattr(seg, "bbox_xyxy", None)
    try:
        values = [int(round(float(value))) for value in box] if box is not None else []
    except (TypeError, ValueError):
        values = []
    return values if len(values) == 4 else [-1, -1, -1, -1]


def _lens(held: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """One of a camera's lenses or extrinsics by name, float64 of ``shape``; NaN throughout where it reported none."""
    value = held.get(name) if hasattr(held, "get") else None
    if value is None:
        return np.full(shape, np.nan)
    array = np.asarray(value, dtype=np.float64)
    if array.size != int(np.prod(shape)):
        return np.full(shape, np.nan)
    return array.reshape(shape)


def _number(value: Any) -> float:
    """``value`` as a float, NaN where it is no number (a bool included)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return float("nan")
    return float(value)


def _ious_of(seg: Any) -> list[float]:
    """SAM2's three predicted IoUs a segmentation carries (``metadata["predicted_iou"]``), NaN where it carries none."""
    metadata = getattr(seg, "metadata", None)
    said = metadata.get("predicted_iou") if isinstance(metadata, dict) else None
    values = [_number(value) for value in said][:3] if isinstance(said, (list, tuple)) else []
    return values + [float("nan")] * (3 - len(values))


def _research_arrays(
    frames: Sequence[Any], captures: Sequence[Any], targets: "Sequence[int | None]",
) -> dict[str, np.ndarray]:
    """The arrays format 2 adds (the module docstring): per look its infrared images and their lenses, its raw depth,
    colour metadata and segmentations, and once the camera the first capture was taken with."""
    arrays: dict[str, np.ndarray] = {}
    for index, (frame, capture) in enumerate(zip(frames, captures)):
        if capture is not None:
            for key, image, dtype in (("ir_left", capture.ir_left, np.uint8), ("ir_right", capture.ir_right, np.uint8),
                                      ("raw_depth", capture.raw_depth, np.uint16)):
                if image is not None:
                    arrays[f"{key}_{index}"] = np.asarray(image, dtype=dtype)
            lenses = capture.camera
            arrays[f"ir_intrinsics_{index}"] = np.stack([_lens(lenses.intrinsics, name, (3, 3))
                                                         for name in _INFRARED_LENSES])
            arrays[f"ir_distortion_{index}"] = np.stack([_lens(lenses.distortion, name, (5,))
                                                         for name in _INFRARED_LENSES])
            arrays[f"ir_extrinsics_{index}"] = np.stack([_lens(lenses.extrinsics_mm, name, (4, 4))
                                                         for name in _INFRARED_EXTRINSICS])
            said = capture.color_metadata
            arrays[f"color_metadata_{index}"] = np.asarray(
                [_number(said.get(key)) for key in COLOR_METADATA], dtype=np.float64)
        segmentations = tuple(getattr(frame, "segmentations", ()) or ())
        arrays[f"seg_masks_{index}"] = _packed_masks(segmentations, np.asarray(frame.depth_map).shape[:2])
        arrays[f"seg_boxes_{index}"] = np.asarray(
            [_box_of(seg) for seg in segmentations], dtype=np.int32).reshape(-1, 4)
        arrays[f"seg_labels_{index}"] = np.asarray(
            [str(getattr(seg, "label", "") or "") for seg in segmentations], dtype=np.str_)
        arrays[f"seg_scores_{index}"] = np.asarray(
            [_number(getattr(seg, "score", None)) for seg in segmentations], dtype=np.float32)
        arrays[f"seg_predicted_iou_{index}"] = np.asarray(
            [_ious_of(seg) for seg in segmentations], dtype=np.float32).reshape(-1, 3)
        target = targets[index] if index < len(targets) else None
        seen = -1
        if isinstance(target, (int, np.integer)) and not isinstance(target, bool) and 0 <= target < len(segmentations):
            seen = int(target)
        arrays[f"seg_target_{index}"] = np.asarray(seen, dtype=np.int64)
    camera = next(capture.camera for capture in captures if capture is not None)
    arrays["camera"] = np.asarray(json.dumps(camera.said, sort_keys=True, default=str), dtype=np.str_)
    arrays["camera_depth_units_m"] = np.asarray(float(camera.depth_units_m), dtype=np.float64)
    for stream in _LENSES:
        if stream in camera.intrinsics:
            arrays[f"camera_{stream}_intrinsics"] = np.asarray(camera.intrinsics[stream], dtype=np.float64).reshape(3, 3)
        if stream in camera.distortion:
            arrays[f"camera_{stream}_distortion"] = np.asarray(camera.distortion[stream], dtype=np.float64).reshape(-1)
    if "depth_to_color" in camera.extrinsics_mm:
        arrays["camera_depth_to_color"] = np.asarray(camera.extrinsics_mm["depth_to_color"],
                                                     dtype=np.float64).reshape(4, 4)
    return arrays


def record_views(
    views: Iterable[Any],
    *,
    target_cloud_base_mm: Any = None,
    directory: "str | Path | None" = None,
    name: str = "",
    targets: "Sequence[int | None]" = (),
) -> Path | None:
    """Write the looks of one pick to one ``.npz`` under ``directory`` (:data:`RECORD_VIEWS_DIR` when ``None``).

    ``views`` are the looks in the order they were taken. Each is a look as the pick loop keeps it (``label``,
    ``name``, ``frame``, ``camera_to_base``: ``LookedAround.views``) or a bare ``PerceptionFrame``; only ``frame`` (or
    the frame itself) is required, with ``depth_map`` and ``intrinsics``, and ``rgb``, ``surface_depth_map`` and
    ``tool_pose`` are kept where it carries them. ``target_cloud_base_mm`` is the target's fused cloud, BASE
    millimetres (N, 3). The layout is the module docstring's: format 2 where a look's frame carries what its camera
    recorded for research (``research``), and then ``targets``, per look the index of the segmentation the pick went
    for (``None`` where none), says which one was the target; format 1, the file it always was, otherwise.

    Returns the file written, or ``None`` when there was no look to keep. A write that fails raises: whoever asked for
    the views decides what a lost file costs.
    """
    looks = list(views)
    if not looks:
        return None
    frames = [getattr(look, "frame", look) for look in looks]
    captures = [_research_of(frame) for frame in frames]
    researched = any(capture is not None for capture in captures)
    arrays: dict[str, np.ndarray] = {"format": np.asarray(
        RECORD_VIEWS_FORMAT if researched else RECORD_VIEWS_FORMAT_WITHOUT_RESEARCH, dtype=np.int64)}
    labels: list[str] = []
    names: list[str] = []
    for index, (look, frame) in enumerate(zip(looks, frames)):
        labels.append(str(getattr(look, "label", "") or ""))
        names.append(str(getattr(look, "name", "") or ""))
        rgb = getattr(frame, "rgb", None)
        if rgb is not None:
            arrays[f"rgb_{index}"] = np.asarray(rgb, dtype=np.uint8)
        arrays[f"depth_{index}"] = np.asarray(frame.depth_map, dtype=np.float32)
        surface = getattr(frame, "surface_depth_map", None)
        if surface is not None:
            arrays[f"surface_depth_{index}"] = np.asarray(surface, dtype=np.float32)
        arrays[f"intrinsics_{index}"] = np.asarray(frame.intrinsics, dtype=np.float64).reshape(3, 3)
        arrays[f"tool_to_base_{index}"] = _matrix(getattr(frame, "tool_pose", None))
        arrays[f"camera_to_base_{index}"] = _matrix(getattr(look, "camera_to_base", None))
    arrays["labels"] = np.asarray(labels, dtype=np.str_)
    arrays["views"] = np.asarray(names, dtype=np.str_)
    cloud = (np.zeros((0, 3)) if target_cloud_base_mm is None
             else np.asarray(target_cloud_base_mm, dtype=np.float64).reshape(-1, 3))
    arrays["target_cloud_base_mm"] = cloud.astype(np.float32)
    if researched:
        arrays.update(_research_arrays(frames, captures, tuple(targets)))

    folder = Path(RECORD_VIEWS_DIR if directory is None else directory)
    folder.mkdir(parents=True, exist_ok=True)
    stem = _UNSAFE.sub("_", name).strip("._") or "pick"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = folder / f"{stem}-{stamp}-{uuid.uuid4().hex[:6]}.npz"
    # mypy cannot tell the unpacked names from numpy's own ``allow_pickle`` keyword; every value here is an ndarray.
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    return path
