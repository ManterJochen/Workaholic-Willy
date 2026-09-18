"""A gripper's collision geometry out of the numbers its registry file carries, for a hand nobody baked.

The planner and the exact-mesh guard both read a collision bundle. A hand this repository has scanned has one; a hand
a customer describes in ``config/grippers/<name>.yaml`` does not, and this builds one from the jaw block: a housing box
and two finger boxes, in the same frame a baked bundle uses, approach +Y, closing X, binormal Z, millimetres, with
y = 0 at the hand's own origin.

Each number's meaning is read off ``scripts/grippers/measure_jaw_from_bundle.py``, which measures the same numbers
the other way round, so the two cannot drift apart about what a half extent means.

An envelope is not the hand, and the gap is about ten millimetres. Measured against the three shipped hands, which
each carry both a bundle and a full set of dimensions, at each hand's own declared grasp centre, a bare envelope
leaves the Hand-E 5.73 mm outside it and the EGU-50 9.50 mm, so those are the smallest inflations that enclose them.
The 2F-85 is 81.49 mm out, and it is the one hand whose palm numbers are an estimate rather than a measurement, which
the registry already records. So a hand whose palm nobody measured is refused by name, and every other hand carries
an inflation the caller states: a safety number with no declared value is not invented here.

Ten millimetres of inflation is not small. It is the price of describing a hand with five numbers instead of scanning
it, and it is why a hand that has a bundle keeps it.

A baked bundle is never overwritten. A hand that was scanned is described better by the scan than by five numbers,
and a writer that replaced one would swap the better model for the worse with nothing said.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_PLANNING = Path(__file__).resolve().parents[1]


def _declared_body() -> Any:
    """The one place a declared box becomes geometry, loaded by path.

    It is shared with the plate and tool changer bodies and with the wrist camera's housing, and it lives beside the
    planner because the sidecar derives a camera link from it at start.
    """
    name = "willy_declared_body"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _PLANNING / "_declared_body.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["DimensionsRefused", "boxes_for_hand", "refusal_for", "write_bundle"]

#: The DH frame a hand's bodies sit in: the flange, which is frame 6 of every UR arm.
HAND_FRAME = 6


class DimensionsRefused(ValueError):
    """These numbers do not describe a hand well enough to plan against, said by name rather than worked around."""


def refusal_for(jaw: Any) -> "str | None":
    """Why this hand may not be modelled from its numbers, or ``None``.

    Two reasons, both fields of the registry. An unset ``palm_thickness_mm`` leaves the housing's width along the
    closing axis to a guess. An estimated palm is not a small error: measured on the one hand that has one, the
    envelope misses the body by 81.5 mm at that hand's own declared centre, against 5.7 and 9.5 mm where the palm was
    measured. An inflation large enough to cover that would not be a model of a gripper any more.
    """
    if not bool(getattr(jaw, "palm_measured", False)):
        return (
            "palm_measured is false, so this hand's housing numbers are an estimate. Measured on the hand that "
            "carries one, an envelope built from estimated palm numbers misses the body by 81.5 mm, against "
            "5.7 and 9.5 mm where the palm was measured. Measure the housing, or bake the hand."
        )
    if getattr(jaw, "palm_thickness_mm", None) is None:
        return (
            "palm_thickness_mm is unset, so the housing's width along the closing axis would be a guess: the fingers' "
            "outer face it used to be taken as missed the EGU-50's housing by 2.25 mm. Measure it, twice the housing's "
            "furthest reach from the grasp axis, and write it into the hand's registry file."
        )
    return None


def boxes_for_hand(jaw: Any, *, inflation_mm: float) -> list:
    """The hand as three boxes, in the bundle's frame, each grown by ``inflation_mm``.

    The housing's half width along the closing axis is the larger of ``palm_thickness_mm`` over two and the fingers'
    outer face, so the box never shrinks below the open fingers. The finger boxes reach ``finger_length_mm`` behind
    the centre, the conservative envelope length, while the housing keeps its measured place behind
    ``finger_behind_mm``.
    """
    body = _declared_body()
    centre = float(jaw.grasp_centre_mm)
    half_open = float(jaw.aperture_mm) / 2.0
    thickness = float(jaw.finger_thickness_mm)
    outer = half_open + thickness

    tip = centre + float(jaw.finger_ahead_mm)
    back = centre - float(jaw.finger_behind_mm)
    finger_back = centre - max(float(jaw.finger_length_mm), float(jaw.finger_behind_mm))
    palm_back = back - float(jaw.palm_depth_mm)
    measured_half = getattr(jaw, "palm_thickness_mm", None)
    palm_half = max(outer, float(measured_half) / 2.0) if measured_half is not None else outer

    boxes = [
        body.Box(name="gripper",
                 centre_mm=(0.0, (back + palm_back) / 2.0, 0.0),
                 half_extents_mm=(palm_half, (back - palm_back) / 2.0, float(jaw.palm_width_mm) / 2.0)),
        body.Box(name="lfinger",
                 centre_mm=(-(half_open + thickness / 2.0), (finger_back + tip) / 2.0, 0.0),
                 half_extents_mm=(thickness / 2.0, (tip - finger_back) / 2.0, float(jaw.finger_width_mm) / 2.0)),
        body.Box(name="rfinger",
                 centre_mm=(half_open + thickness / 2.0, (finger_back + tip) / 2.0, 0.0),
                 half_extents_mm=(thickness / 2.0, (tip - finger_back) / 2.0, float(jaw.finger_width_mm) / 2.0)),
    ]
    return [body.inflated(box, by_mm=float(inflation_mm)) for box in boxes]


def housing_gap_mm(jaw: Any) -> float:
    """How far the housing's back lies ahead of the hand's origin along the approach, millimetres; 0 where it reaches it.

    A gap is space between the origin and the housing that no body of this envelope occupies: an adapter, a quick
    changer, or a flange boss nobody declared. The writer says it; only a plate with a cross section makes it a body.
    """
    back = float(jaw.grasp_centre_mm) - float(jaw.finger_behind_mm) - float(jaw.palm_depth_mm)
    return max(0.0, back)


def _hand_from_mesh() -> Any:
    """``hand_from_mesh.py`` beside this file: the one writer every hand route shares."""
    if __name__.startswith("src."):
        from . import hand_from_mesh

        return hand_from_mesh
    name = "willy_hand_from_mesh"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name("hand_from_mesh.py"))
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def write_bundle(hand: str, path: "Path | str", *, inflation_mm: float, origin: str) -> Path:
    """Write ``hand``'s collision bundle from its registry numbers, or refuse and say why.

    ``inflation_mm`` is stated by the caller rather than defaulted, and lands in the bundle beside a ``hand__source``
    of ``dimensions``, so every reader can tell an envelope from a scan. ``origin`` says where ``grasp_centre_mm`` was
    measured from, ``flange`` or ``mounting_face``, and is stamped like a bake's: a mounting-face hand is moved out by
    the plates a cell declares. It goes through the one writer every route shares.
    """
    from src.config.grippers import load_gripper

    body = _declared_body()
    target = Path(path)
    jaw = load_gripper(hand).jaw

    said = refusal_for(jaw)
    if said is not None:
        raise DimensionsRefused(f"{hand}: {said}")
    if target.exists():
        raise DimensionsRefused(
            f"{target.name} is already there, and a bundle that was baked describes this hand better than five "
            f"numbers do. Delete it on purpose if the scan is the thing that is wrong."
        )

    arrays = body.boxes_to_parts(boxes_for_hand(jaw, inflation_mm=inflation_mm), frame=HAND_FRAME)
    mesh = _hand_from_mesh()
    bundle = mesh.HandBundle.from_parts(
        parts={part: (arrays[f"{part}__v"], arrays[f"{part}__f"]) for part in ("gripper", "lfinger", "rfinger")},
        axes=mesh.VendorAxes.from_words(closing="+X", approach="+Y", binormal="+Z"),
        scale_to_mm=1.0, mount_face_mm=0.0, origin=origin,
    )
    bundle.write(target, records={"hand__source": "dimensions", "hand__inflation_mm": float(inflation_mm)})
    return target
