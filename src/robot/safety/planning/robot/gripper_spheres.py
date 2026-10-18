"""Where a gripper's baked bundle stands: its frame and its origin.

The planner models a robot as spheres; a planner reads cover fits only (``scripts/arms/fit.py``, its
fit in ``scripts/arms/_cover_fit.py``). This module keeps what those and the safety guard share about a
baked bundle, ``{name}_collision_meshes.npz``: the frames it is written in (``FLANGE``,
``MOUNTING_FACE``), where it says it stands (:func:`bundle_origin`) and the refusal of one that cannot be
read as described (:class:`SphereFitError`). The grid fit kept here for measuring and comparing left on
2026-10-18, as nothing called it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "FLANGE",
    "MOUNTING_FACE",
    "SphereFitError",
    "bundle_origin",
]

#: The sphere set already sits where the hand will be bolted, so nothing is added on the box.
#: True of a bundle baked out of a composed arm asset, where the arm placed the gripper.
FLANGE = "flange"

#: The sphere set starts at the hand's own mounting face, and whatever plate sits between that
#: face and the flange has to be added before the planner sees it. True of a bundle read from a
#: standalone vendor asset, which has no idea what it will be bolted to. The plate's thickness is
#: a bench measurement, the same one `robot.gripper.tool_frame.offset_mm` needs.
MOUNTING_FACE = "mounting_face"

#: The npz key carrying one of the two. A bundle without it came out of a composed arm asset, so
#: the absent case reads as :data:`FLANGE`.
_ORIGIN_KEY = "gripper__origin"

class SphereFitError(ValueError):
    """The gripper cannot be fitted as described."""


def bundle_origin(bundle: Path) -> str:
    """Whether a bundle's gripper arrays start at the flange or at the hand's mounting face.

    A bundle without the key came out of a composed arm asset, where the arm had already placed the
    hand, so the absent case is :data:`FLANGE`. That is a statement about those files rather than a
    lenient default.
    """
    if not bundle.is_file():
        raise SphereFitError(f"no collision-mesh bundle at {bundle}")
    with np.load(bundle, allow_pickle=True) as data:
        if _ORIGIN_KEY not in data:
            return FLANGE
        value = str(np.asarray(data[_ORIGIN_KEY]).reshape(-1)[0])
    if value not in (FLANGE, MOUNTING_FACE):
        raise SphereFitError(
            f"{bundle.name} declares origin {value!r}, which is neither {FLANGE!r} nor "
            f"{MOUNTING_FACE!r}. The on-box builder decides whether to add a coupling from this, "
            "so an unknown value is not something to guess past."
        )
    return value


