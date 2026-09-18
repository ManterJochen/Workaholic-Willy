"""Measure a parallel jaw off a committed hand bundle, in the grasp frame the gripper registry uses.

    python scripts/grippers/measure_jaw_from_bundle.py robotiq_hande --centre-mm 135.75
    python scripts/grippers/measure_jaw_from_bundle.py schunk_egu50

Read-only: numpy over one ``.npz``, no Isaac, no GPU and no USD. The numbers a hand's registry file needs and no
file held are measured here, and the suite recomputes them with this same code. It is trusted only because it
reproduces the Hand-E numbers the tree already carried, from their own measurement session, to 0.05 mm.

The frame, as ``scripts/grippers/bake_gripper_variant.py`` bakes the bundles: approach +Y, closing X, binormal Z,
millimetres. y = 0 is the bundle's origin, named by its ``gripper__origin`` key: the hand's own mounting face for a
standalone asset, the flange otherwise. A bundle without the key predates it and was read out of a composed arm or
a mounted placement, whose origin is the flange.

What each number is, on the two finger arrays and the housing array at the pose the bundle holds:

  jaw face          per finger, the innermost x within 3 mm of the fingertip
  aperture          the distance between the two jaw faces
  face              the flat triangles lying on a jaw face plane, merged into connected stretches along the
                    approach; the face is the stretch that reaches the fingertip, the same on both fingers
  grasp centre      the face midpoint, unless a centre is given. A centre off the face is refused: a contact
                    patch cannot sit behind or ahead of itself
  finger_ahead      fingertip minus grasp centre
  finger_behind     grasp centre minus the first finger material outside the housing. Material inside the
                    housing (a Hand-E carriage) belongs to the palm
  finger_thickness  jaw face to the outermost finger material outside the housing, along the closing axis
  finger_width      the finger's extent along the binormal, outside the housing
  pad               the jaw face: pad_ahead is its top minus the centre, pad_behind the centre minus its bottom
  palm_depth        from where the finger box ends (the centre minus finger_behind) to the housing's back
  palm_width        the housing's extent along the binormal
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]

#: A vertex this close to a jaw face plane is on it. The bundles hold float32 or float64 millimetres, so this is far
#: above their noise and far below any feature of a finger.
_ON_PLANE_MM = 0.05
#: How far below the fingertip the jaw face plane is read.
_TIP_BAND_MM = 3.0
#: How far the two fingers' faces may disagree along the approach before the bundle is not one symmetric jaw.
_FACE_AGREEMENT_MM = 0.5
#: The arrays of a hand, housing first.
_HAND = ("gripper", "lfinger", "rfinger")


@dataclass(frozen=True, slots=True)
class JawMeasurement:
    """One hand's jaw numbers in millimetres, measured off one bundle."""

    #: What y = 0 is: ``flange`` or ``mounting_face``.
    origin: str
    #: The grasp centre along the approach, in the bundle frame.
    centre_mm: float
    aperture_mm: float
    finger_ahead_mm: float
    finger_behind_mm: float
    finger_thickness_mm: float
    finger_width_mm: float
    #: The flat jaw face's length, whatever centre was asked for.
    face_length_mm: float
    pad_length_mm: float
    pad_ahead_mm: float
    pad_behind_mm: float
    palm_depth_mm: float
    palm_width_mm: float
    #: The housing along the closing axis: twice its furthest reach from the grasp axis.
    palm_thickness_mm: float

    def render(self) -> str:
        """One line per number, for a person. ASCII, no trailing newline."""
        lines = []
        for name, value in asdict(self).items():
            shown = f"{value:.2f}" if isinstance(value, float) else str(value)
            lines.append(f"{name:<20} {shown}")
        return "\n".join(lines)


def _load(bundle: Path) -> tuple[dict[str, np.ndarray], str]:
    with np.load(bundle) as data:
        needed = [f"{part}__{kind}" for part in _HAND for kind in ("v", "f")]
        missing = [key for key in needed if key not in data.files]
        if missing:
            raise ValueError(f"{bundle.name} has no {', '.join(missing)}, so it is not a hand bundle")
        arrays = {key: np.asarray(data[key]) for key in needed}
        origin = str(data["gripper__origin"][0]) if "gripper__origin" in data.files else "flange"
    return arrays, origin


def _jaw_face(vertices: np.ndarray, faces: np.ndarray, *, innermost: str, tip: float) -> tuple[float, float, float]:
    """(plane x, face bottom y, face top y) of one finger's jaw face."""
    near_tip = vertices[vertices[:, 1] > tip - _TIP_BAND_MM]
    plane = float(near_tip[:, 0].max() if innermost == "max" else near_tip[:, 0].min())
    on_plane = np.abs(vertices[:, 0] - plane) <= _ON_PLANE_MM
    flat = faces[np.all(on_plane[faces], axis=1)]
    if len(flat) == 0:
        raise ValueError(f"no flat triangle lies on the jaw face plane x = {plane:.3f} mm")
    spans = sorted(
        (float(vertices[triangle, 1].min()), float(vertices[triangle, 1].max())) for triangle in flat
    )
    stretches: list[list[float]] = []
    for low, high in spans:
        if stretches and low <= stretches[-1][1] + _ON_PLANE_MM:
            stretches[-1][1] = max(stretches[-1][1], high)
        else:
            stretches.append([low, high])
    bottom, top = max(stretches, key=lambda stretch: stretch[1])
    if top < tip - _ON_PLANE_MM:
        raise ValueError(
            f"the flat jaw face on x = {plane:.3f} mm ends at y {top:.2f} mm and the fingertip is at {tip:.2f} mm"
        )
    return plane, bottom, top


def measure_parallel_jaw(bundle: "str | Path", *, centre_mm: float | None = None) -> JawMeasurement:
    """Measure the jaw of the hand in ``bundle``. ``centre_mm`` is the grasp centre in the bundle frame.

    Raises ``ValueError`` for a bundle that is not a hand, a jaw whose two faces disagree, or a centre off the face.
    """
    path = Path(bundle)
    arrays, origin = _load(path)
    housing = arrays["gripper__v"].astype(np.float64)
    left = arrays["lfinger__v"].astype(np.float64)
    right = arrays["rfinger__v"].astype(np.float64)
    front, back = float(housing[:, 1].max()), float(housing[:, 1].min())
    tip = float(max(left[:, 1].max(), right[:, 1].max()))

    left_plane, left_bottom, left_top = _jaw_face(left, arrays["lfinger__f"], innermost="max", tip=tip)
    right_plane, right_bottom, right_top = _jaw_face(right, arrays["rfinger__f"], innermost="min", tip=tip)
    if left_plane >= right_plane:
        raise ValueError(
            f"the left jaw face at x = {left_plane:.3f} mm is not left of the right one at {right_plane:.3f} mm, "
            "so the fingers are crossed or the bundle is mirrored"
        )
    if abs(left_bottom - right_bottom) > _FACE_AGREEMENT_MM or abs(left_top - right_top) > _FACE_AGREEMENT_MM:
        raise ValueError(
            f"the two jaw faces run y [{left_bottom:.2f}, {left_top:.2f}] and [{right_bottom:.2f}, {right_top:.2f}] mm, "
            "which is not one jaw"
        )
    face_bottom, face_top = max(left_bottom, right_bottom), min(left_top, right_top)
    centre = (face_bottom + face_top) / 2.0 if centre_mm is None else float(centre_mm)
    if not face_bottom - _ON_PLANE_MM <= centre <= face_top + _ON_PLANE_MM:
        raise ValueError(
            f"a grasp centre at {centre:.2f} mm is off the jaw face, which runs {face_bottom:.2f} to {face_top:.2f} mm "
            f"from the bundle's {origin}: the contact patch would sit behind or ahead of itself"
        )

    outside_left = left[left[:, 1] > front]
    outside_right = right[right[:, 1] > front]
    finger_behind = centre - float(min(outside_left[:, 1].min(), outside_right[:, 1].min()))
    return JawMeasurement(
        origin=origin,
        centre_mm=centre,
        aperture_mm=right_plane - left_plane,
        finger_ahead_mm=tip - centre,
        finger_behind_mm=finger_behind,
        finger_thickness_mm=max(
            left_plane - float(outside_left[:, 0].min()), float(outside_right[:, 0].max()) - right_plane
        ),
        finger_width_mm=max(float(np.ptp(outside_left[:, 2])), float(np.ptp(outside_right[:, 2]))),
        face_length_mm=face_top - face_bottom,
        pad_length_mm=face_top - face_bottom,
        pad_ahead_mm=face_top - centre,
        pad_behind_mm=centre - face_bottom,
        palm_depth_mm=(centre - finger_behind) - back,
        palm_width_mm=float(np.ptp(housing[:, 2])),
        palm_thickness_mm=2.0 * float(np.abs(housing[:, 0]).max()),
    )


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(_REPO))
    from src.robot.safety.planning.environment import hand_mesh_bundle

    parser = argparse.ArgumentParser(
        prog="python scripts/grippers/measure_jaw_from_bundle.py",
        description="Measure a parallel jaw off a committed hand bundle, in the registry's grasp frame.",
    )
    parser.add_argument("hand", help="the hand's registry name, for example robotiq_hande")
    parser.add_argument("--centre-mm", type=float, default=None,
                        help="the grasp centre along the approach in the bundle frame (default: the face midpoint)")
    args = parser.parse_args(argv)

    # A hand is its own bundle, the same on every arm, so no arm is asked for.
    bundle = hand_mesh_bundle(args.hand)
    if not bundle.is_file():
        print(f"no bundle for {args.hand} at {bundle}", file=sys.stderr)
        return 2
    try:
        measured = measure_parallel_jaw(bundle, centre_mm=args.centre_mm)
    except ValueError as exc:
        print(f"{bundle.name}: {exc}", file=sys.stderr)
        return 1
    print(f"{bundle.name}")
    print(measured.render())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
