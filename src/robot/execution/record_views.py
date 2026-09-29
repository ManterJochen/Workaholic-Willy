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

A file is named after the pick (``name``, the service's ``attempt_id`` when ``PickRun`` writes it, which is the key of
the pick's record in the JSONL corpus), then the UTC time and a short random suffix, so two picks never overwrite each
other.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

import numpy as np

__all__ = ["RECORD_VIEWS_DIR", "RECORD_VIEWS_FORMAT", "record_views"]

#: Where a pick's views are kept when nobody says otherwise, relative to the working directory, beside the robot's logs.
RECORD_VIEWS_DIR: Final[str] = "logs/robot/views"
#: What shape a views file is (the module docstring). Bumped whenever an array is added, renamed or given a new meaning.
RECORD_VIEWS_FORMAT: Final[int] = 1

#: What a file name keeps of the name it is given; everything else becomes ``_``.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _matrix(value: Any) -> np.ndarray:
    """``value`` as a (4, 4) float64 matrix: a transform or pose (``to_matrix``), an array, or all NaN for none."""
    if value is None:
        return np.full((4, 4), np.nan)
    to_matrix = getattr(value, "to_matrix", None)
    matrix = np.asarray(to_matrix() if callable(to_matrix) else value, dtype=np.float64)
    return matrix if matrix.shape == (4, 4) else np.full((4, 4), np.nan)


def record_views(
    views: Iterable[Any],
    *,
    target_cloud_base_mm: Any = None,
    directory: "str | Path | None" = None,
    name: str = "",
) -> Path | None:
    """Write the looks of one pick to one ``.npz`` under ``directory`` (:data:`RECORD_VIEWS_DIR` when ``None``).

    ``views`` are the looks in the order they were taken. Each is a look as the pick loop keeps it (``label``,
    ``name``, ``frame``, ``camera_to_base``: ``LookedAround.views``) or a bare ``PerceptionFrame``; only ``frame`` (or
    the frame itself) is required, with ``depth_map`` and ``intrinsics``, and ``rgb``, ``surface_depth_map`` and
    ``tool_pose`` are kept where it carries them. ``target_cloud_base_mm`` is the target's fused cloud, BASE
    millimetres (N, 3). The layout is the module docstring's.

    Returns the file written, or ``None`` when there was no look to keep. A write that fails raises: whoever asked for
    the views decides what a lost file costs.
    """
    looks = list(views)
    if not looks:
        return None
    arrays: dict[str, np.ndarray] = {"format": np.asarray(RECORD_VIEWS_FORMAT, dtype=np.int64)}
    labels: list[str] = []
    names: list[str] = []
    for index, look in enumerate(looks):
        frame = getattr(look, "frame", look)
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

    folder = Path(RECORD_VIEWS_DIR if directory is None else directory)
    folder.mkdir(parents=True, exist_ok=True)
    stem = _UNSAFE.sub("_", name).strip("._") or "pick"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = folder / f"{stem}-{stamp}-{uuid.uuid4().hex[:6]}.npz"
    # mypy cannot tell the unpacked names from numpy's own ``allow_pickle`` keyword; every value here is an ndarray.
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    return path
