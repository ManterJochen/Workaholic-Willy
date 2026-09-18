"""What a hand bundle has to be before anything composes it onto an arm.

numpy only, and loadable by path: the guard, the doctor, the cover fit and every writer of a hand bundle ask this one
module, so no writer can produce what a loader refuses.

Why a loader asks at all. The guard takes every array of a hand bundle that is not a record as a mesh, and it places
only the three parts it knows: one plate out and turned by the placement. A part called anything else would be
composed where the file put it, and a hand whose fingers lie along another axis than the model's +Y would be turned
the way the planner map is turned and nowhere near where the fingers are. Both assumptions hold only for a bundle this
repository baked, and a customer hand is not one.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np

__all__ = [
    "HAND_FRAME",
    "HAND_PARTS",
    "HandBundleRefused",
    "MODEL_APPROACH",
    "ORIGINS",
    "ORIGIN_KEY",
    "RECORD_PREFIX",
    "hand_bundle_refusal",
]

#: The parts of a hand, the only arrays of a hand bundle the guard places.
HAND_PARTS: Final = ("gripper", "lfinger", "rfinger")
#: The DH frame every hand part hangs from: the flange.
HAND_FRAME: Final = 6
#: The approach of every hand model, in tool0 axes, which is DH frame 6's; every placement turns from it.
MODEL_APPROACH: Final = (0.0, 1.0, 0.0)
#: Where a hand's numbers start, as a bundle stamps it; a bundle that stamps nothing starts at the flange.
ORIGIN_KEY: Final = "gripper__origin"
ORIGINS: Final = ("flange", "mounting_face")
#: The prefix of what a bundle records about itself, never a mesh.
RECORD_PREFIX: Final = "hand__"

_SUFFIXES = ("v", "f", "frame")


class HandBundleRefused(ValueError):
    """A hand bundle nothing may compose, with the sentence that says why."""


def _tolerance_deg() -> float:
    """The degree a declared frame snaps within (``_hand_placement.TOLERANCE_DEG``), so the model axis keeps one tolerance."""
    name = "willy_hand_placement_tolerance"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name("_hand_placement.py"))
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return float(module.TOLERANCE_DEG)


def _angle_deg(a: Any, b: Any) -> float:
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if norm == 0.0:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(va, vb)) / norm))))


def hand_bundle_refusal(arrays: Mapping[str, Any], *, name: str = "this hand bundle") -> str | None:
    """Why ``arrays`` is not a hand bundle anything may compose, or ``None`` where it is one.

    Exactly the three parts, each with finite ``(N, 3)`` vertices, integer ``(M, 3)`` faces indexing them and a
    one element integer frame of 6; besides them only ``gripper__origin`` (``flange`` or ``mounting_face``) and
    ``hand__`` records. The fingers' midpoint lies along the model's +Y within the placement tolerance, and the left
    finger lies on the -X side of the right one.
    """
    keys = set(arrays)
    allowed = {f"{part}__{suffix}" for part in HAND_PARTS for suffix in _SUFFIXES} | {ORIGIN_KEY}
    unknown = sorted(key for key in keys if key not in allowed and not key.startswith(RECORD_PREFIX))
    if unknown:
        return (
            f"{name} carries {', '.join(unknown)}, and a hand bundle holds exactly the parts {', '.join(HAND_PARTS)} "
            f"(vertices, faces and frame each) with its origin and hand__ records: a part the guard does not know is "
            f"neither moved by a plate nor turned by the placement, so it would be checked where nobody put it"
        )
    for part in HAND_PARTS:
        missing = [f"{part}__{suffix}" for suffix in _SUFFIXES if f"{part}__{suffix}" not in keys]
        if missing:
            return f"{name} has no {', '.join(missing)}: a hand without its {part} is not the hand the cell carries"
        vertices = np.asarray(arrays[f"{part}__v"])
        faces = np.asarray(arrays[f"{part}__f"])
        frame = np.asarray(arrays[f"{part}__frame"]).reshape(-1)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or vertices.shape[0] == 0:
            return f"{name}: {part}__v has shape {vertices.shape}, and a part's vertices are a non empty (N, 3) array"
        if not np.issubdtype(vertices.dtype, np.floating) or not bool(np.all(np.isfinite(vertices))):
            return f"{name}: {part}__v holds a value that is not a finite float, so no distance to it means anything"
        if faces.ndim != 2 or faces.shape[1:] != (3,) or not np.issubdtype(faces.dtype, np.integer):
            return f"{name}: {part}__f is {faces.dtype} of shape {faces.shape}, and faces are an integer (M, 3) array"
        if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= vertices.shape[0]):
            return (
                f"{name}: {part}__f indexes vertex {int(faces.max()) if int(faces.min()) >= 0 else int(faces.min())} "
                f"of {vertices.shape[0]}, so a face points at a vertex the part does not have"
            )
        if frame.size != 1 or not np.issubdtype(frame.dtype, np.integer) or int(frame[0]) != HAND_FRAME:
            return (
                f"{name}: {part}__frame is {frame.tolist()}, and every hand part hangs from DH frame {HAND_FRAME}, "
                f"the flange"
            )
    if ORIGIN_KEY in keys:
        origin = np.asarray(arrays[ORIGIN_KEY]).reshape(-1)
        said = str(origin[0]) if origin.size == 1 else str(origin.tolist())
        if said not in ORIGINS:
            return f"{name}: {ORIGIN_KEY} says {said!r}, and a hand's numbers start at the {' or the '.join(ORIGINS)}"
    left = np.asarray(arrays["lfinger__v"], dtype=np.float64).mean(axis=0)
    right = np.asarray(arrays["rfinger__v"], dtype=np.float64).mean(axis=0)
    midpoint = (left + right) / 2.0
    degrees = _angle_deg(midpoint, MODEL_APPROACH)
    tolerance = _tolerance_deg()
    if degrees > tolerance:
        return (
            f"{name}: the fingers' midpoint {np.round(midpoint, 2).tolist()} mm lies {degrees:.1f} degrees from the "
            f"model's +Y, and a hand model holds its fingers along +Y within {tolerance:g} degree: every placement "
            f"turns the hand from +Y, so this one would be modelled {degrees:.0f} degrees from where its fingers are. "
            f"Write the bundle with its axes stated (closing X, approach +Y, binormal Z)"
        )
    if not float(left[0]) < float(right[0]):
        return (
            f"{name}: the left finger's mean x {float(left[0]):.2f} mm is not below the right finger's "
            f"{float(right[0]):.2f} mm, so the fingers are crossed or the closing axis is reversed"
        )
    return None
