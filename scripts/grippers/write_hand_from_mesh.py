"""Write a gripper's collision bundle from vendor mesh files, for a hand nobody baked.

    .venv/Scripts/python.exe scripts/grippers/write_hand_from_mesh.py acme_2f \\
        --gripper-mesh vendor/housing.stl --lfinger-mesh vendor/left.stl --rfinger-mesh vendor/right.stl \\
        --scale-to-mm 1000 --closing +Y --approach +Z --binormal +X --mount-face-mm 0 --origin flange
    ... the same line with --write

Run it with the project venv. No simulator, no GPU, no USD: trimesh reads STL, OBJ and the other formats it knows.

Nothing the caller states has a default. A vendor file is in the vendor's frame and units. The caller says how many
millimetres one unit is (``--scale-to-mm``: 1000 for metres, 1 for millimetres, 25.4 for inches), which vendor axis
the hand closes along, approaches along and has as its binormal (each one of ``+X -X +Y -Y +Z -Z``), where the
mounting face sits along the approach axis in millimetres, and whether the numbers then start at the ``flange`` or at
the hand's own ``mounting_face``. A spelling that is a mirror is refused, and a spelling that is a rotation but puts
the fingers off the model's +Y is refused by the check every hand bundle meets. The body is three files, the housing
and the two fingers, because the guard places exactly those three parts.

It proves its own reader first. Every run exports the committed Hand-E bundle into its vendor frame as OBJ files,
reads it back through the same code, and refuses to write anything if the arrays do not come back.

Without ``--write`` it prints the extents and the jaw numbers the registry file will need, and writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATA = REPO / "src/robot/safety/data"
#: The committed bundle every run reads back through its own reader before it writes.
SELF_CHECK_BUNDLE = DATA / "robotiq_hande_hand_meshes.npz"
_SELF_CHECK_WORDS = {"closing": "+Z", "approach": "+Y", "binormal": "-X"}
_SELF_CHECK_MOUNT_FACE_MM = -86.10
#: How far the self check may move a vertex: OBJ text round trips a float to far below this.
_SELF_CHECK_LIMIT_MM = 1e-6
#: A hand smaller or larger than this is a unit error, not a hand.
_EXTENT_MM = (10.0, 2000.0)
_PARTS = ("gripper", "lfinger", "rfinger")


def _load(name: str, rel: str) -> Any:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / rel)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _mesh_lib() -> Any:
    return _load("willy_hand_from_mesh", "src/robot/safety/planning/robot/hand_from_mesh.py")


def _self_check() -> float:
    """Export the reference bundle into its vendor frame as OBJ, read it back, and return the worst vertex move in mm."""
    lib = _mesh_lib()
    axes = lib.VendorAxes.from_words(**_SELF_CHECK_WORDS)
    rotation = np.asarray(axes.rotation)
    with np.load(SELF_CHECK_BUNDLE, allow_pickle=True) as held:
        reference = {key: np.array(held[key]) for key in held.files}
    with tempfile.TemporaryDirectory() as folder:
        files = {}
        for part in _PARTS:
            vendor = np.asarray(reference[f"{part}__v"], dtype=np.float64) @ rotation
            vendor[:, axes.approach_index] += _SELF_CHECK_MOUNT_FACE_MM
            vendor /= 1000.0
            path = Path(folder) / f"{part}.obj"
            with path.open("w", encoding="ascii") as handle:
                for x, y, z in vendor:
                    handle.write(f"v {float(x)!r} {float(y)!r} {float(z)!r}\n")
                for a, b, c in np.asarray(reference[f"{part}__f"]):
                    handle.write(f"f {a + 1} {b + 1} {c + 1}\n")
            files[part] = path
        read = lib.HandBundle.from_mesh_files(
            gripper=files["gripper"], lfinger=files["lfinger"], rfinger=files["rfinger"], axes=axes,
            scale_to_mm=1000.0, mount_face_mm=_SELF_CHECK_MOUNT_FACE_MM, origin="mounting_face",
        ).arrays()
    worst = 0.0
    for part in _PARTS:
        got, want = read[f"{part}__v"], np.asarray(reference[f"{part}__v"], dtype=np.float64)
        if got.shape != want.shape or not np.array_equal(read[f"{part}__f"], np.asarray(reference[f"{part}__f"])):
            return float("inf")
        worst = max(worst, float(np.abs(got - want).max()))
    return worst


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: "list[str] | None" = None) -> int:
    lib = _mesh_lib()
    parser = argparse.ArgumentParser(
        prog="python scripts/grippers/write_hand_from_mesh.py",
        description="Write a gripper's collision bundle from vendor mesh files.",
    )
    parser.add_argument("hand", help="the registry name the hand will have, robot.gripper.model")
    for part, what in (("gripper", "the housing"), ("lfinger", "the left finger"), ("rfinger", "the right finger")):
        parser.add_argument(f"--{part}-mesh", required=True, help=f"{what}: an STL, OBJ or other mesh file")
    parser.add_argument("--scale-to-mm", type=float, required=True, help="millimetres per vendor unit, 1000 for metres")
    for word in ("closing", "approach", "binormal"):
        parser.add_argument(f"--{word}", required=True, choices=lib.AXIS_WORDS, help=f"the vendor axis of the {word}")
    parser.add_argument("--mount-face-mm", type=float, required=True,
                        help="the mounting face along the approach axis, millimetres in the vendor frame")
    parser.add_argument("--origin", required=True, choices=("flange", "mounting_face"),
                        help="where the numbers start once the mounting face is at zero")
    parser.add_argument("--out", default=None, help="where to write; the committed bundle path by default")
    parser.add_argument("--write", action="store_true", help="write the bundle; without it nothing is written")
    args = parser.parse_args(lib.joined_axis_words(list(sys.argv[1:] if argv is None else argv)))

    from src.config.schema.grippers.gripper_schema import MODEL_NAME_PATTERN

    if not re.match(MODEL_NAME_PATTERN, args.hand):
        raise SystemExit(f"{args.hand!r} is not a registry name: lower case letters, digits and underscores ({MODEL_NAME_PATTERN})")

    worst = _self_check()
    if not worst <= _SELF_CHECK_LIMIT_MM:
        print(f"the self check failed: the reference bundle read back through this reader moved by {worst} mm "
              f"(limit {_SELF_CHECK_LIMIT_MM} mm), so no hand it reads can be trusted and nothing is written",
              file=sys.stderr)
        return 1

    files = {part: Path(getattr(args, f"{part}_mesh")) for part in _PARTS}
    try:
        axes = lib.VendorAxes.from_words(closing=args.closing, approach=args.approach, binormal=args.binormal)
        bundle = lib.HandBundle.from_mesh_files(
            gripper=files["gripper"], lfinger=files["lfinger"], rfinger=files["rfinger"], axes=axes,
            scale_to_mm=args.scale_to_mm, mount_face_mm=args.mount_face_mm, origin=args.origin,
        )
    except ValueError as exc:
        raise SystemExit(f"{args.hand}: {exc}") from None

    arrays = bundle.arrays()
    stacked = np.concatenate([arrays[f"{part}__v"] for part in _PARTS])
    extent = float(np.ptp(stacked, axis=0).max())
    if not _EXTENT_MM[0] <= extent <= _EXTENT_MM[1]:
        raise SystemExit(
            f"{args.hand}: the hand spans {extent:.3f} mm at --scale-to-mm {args.scale_to_mm:g}, and a gripper spans "
            f"{_EXTENT_MM[0]:g} to {_EXTENT_MM[1]:g} mm: check --scale-to-mm (metres read as millimetres is a hand a "
            f"thousand times too small)"
        )
    refusal = lib.hand_bundle_rules().hand_bundle_refusal(arrays, name=f"{args.hand}_hand_meshes.npz")
    if refusal is not None:
        raise SystemExit(f"{args.hand}: {refusal}")

    print(f"{args.hand}: {axes.render()}, scale {args.scale_to_mm:g} mm per unit, mounting face "
          f"{args.mount_face_mm:g} mm, numbers starting at the {args.origin}; self check moved nothing past {worst:.1e} mm")
    for part in _PARTS:
        vertices = arrays[f"{part}__v"]
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        print(f"  {part:<9} {len(vertices):>6} verts  x [{lo[0]:8.2f},{hi[0]:8.2f}]  y [{lo[1]:8.2f},{hi[1]:8.2f}]"
              f"  z [{lo[2]:8.2f},{hi[2]:8.2f}]")
    measure = _load("willy_measure_jaw", "scripts/grippers/measure_jaw_from_bundle.py")
    with tempfile.TemporaryDirectory() as folder:
        scratch = Path(folder) / f"{args.hand}_hand_meshes.npz"
        np.savez(scratch, **arrays)
        try:
            print("  the jaw numbers its registry file needs, measured off this bundle at the jaw face midpoint:")
            for line in measure.measure_parallel_jaw(scratch).render().splitlines():
                print(f"    {line}")
        except ValueError as exc:
            print(f"  the jaw could not be measured off this bundle: {exc}")

    if not args.write:
        print("nothing written; pass --write")
        return 0

    out = Path(args.out) if args.out else DATA / f"{args.hand}_hand_meshes.npz"
    recipe = {
        "writer": "scripts/grippers/write_hand_from_mesh.py",
        "files": {part: {"name": files[part].name, "sha256": _sha256(files[part])} for part in _PARTS},
        "scale_to_mm": args.scale_to_mm, "axes": axes.to_dict(), "mount_face_mm": args.mount_face_mm,
        "origin": args.origin, "self_check_worst_mm": worst,
    }
    try:
        report = bundle.write(out, records={"hand__recipe": json.dumps(recipe, sort_keys=True)})
    except ValueError as exc:
        raise SystemExit(f"{args.hand}: {exc}") from None
    print(report.render())

    from src.robot.safety.planning.bundle_index import INDEX_NAME, record_hand_bundle

    if (out.parent / INDEX_NAME).is_file():
        names = ", ".join(files[part].name for part in _PARTS)
        row = record_hand_bundle(
            out, source="vendor_mesh",
            note=(f"vendor mesh files {names} at {args.scale_to_mm:g} mm per unit, {axes.render()}, mounting face "
                  f"{args.mount_face_mm:g} mm, starting at the {args.origin}"),
        )
        print(f"  {row.render()}")
    else:
        print(f"  no {INDEX_NAME} beside {out.name}, so no row was recorded")
    print("next:")
    print(f"  measure the jaw into config/grippers/{args.hand}.yaml: "
          f"python scripts/grippers/measure_jaw_from_bundle.py {args.hand}")
    print(f"  fit its spheres for the planner: python scripts/curobo/fit_cover_spheres.py --hand {args.hand} --write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
