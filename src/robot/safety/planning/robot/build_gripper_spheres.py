"""Grid fit a gripper's collision-sphere map for the planner, from a baked bundle or from a mesh file.

The planner models the robot as spheres. A gripper mesh is never used directly: it is what a sphere
set is fitted from, once, and the spheres are all the planner ever sees. So this is where "the
planner knows your gripper" is decided, and a cell whose sphere map describes a different hand plans
against a different robot without anything saying so.

The command line writes no map. It exits 2 and names what replaces it: the hand's body from
`scripts/grippers/write_hand_from_mesh.py`, `write_hand_from_dimensions.py` or
`bake_gripper_variant.py --usd`, then its map from the cover fit. `build()` stays a library.

The grid fit does not write the committed maps. It grid fits a bundle: a voxel grid over the vertices, one
sphere per occupied cell, capped radius. Measured against the meshes they stand for, the three maps
it produced left a hole of 16.5 to 19.0 mm and reached 22.9 to 24.7 mm past the hand. A hole is a
false clear: the planner, and the perception self filter that reads the same map, do not see the
hand there.

The committed maps are cover fits, written by `scripts/curobo/fit_cover_spheres.py` with
`--hand <name> --write`, which covers every surface sample by construction or writes nothing at all.
`planner_hand` refuses a map that is not one, so a grid fit map under a committed map's name never
reaches a planner.

Scope, stated honestly: this covers the `tool0` gripper spheres, which are frame-correct and
directly usable by cuRobo. The arm-link spheres and the schema-complete robot descriptor are
produced on the target box by `scripts/curobo/build_ur_config.py`, which reads whatever this wrote.
See `PROVENANCE.md`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.robot.safety.planning.robot.gripper_spheres import (
    GripperSpheres,
    SphereFitError,
    fit_gripper_spheres,
    fit_spheres_from_mesh,
)

_HERE = Path(__file__).resolve().parent
_BUNDLE_DIR = _HERE.parents[1] / "data"
_DEFAULT_VARIANT = "ur5e"


def build(spheres: GripperSpheres, *, note: str = "") -> dict:
    """The sphere map with its provenance, in the shape cuRobo reads.

    The provenance block is the point of writing a file at all rather than fitting on the fly: a
    sphere map is geometry a planner trusts absolutely and looks like any other list of numbers, and
    the question a year from now is which gripper, from which file.
    """
    return {
        "_provenance": {
            "gripper": spheres.gripper,
            "frame": "tool0 (Y=approach, X=closing, Z=depth), metres",
            # Where these numbers start. `flange` means they already sit where the hand is
            # bolted; `mounting_face` means a coupling plate still has to be added, and the
            # on-box builder refuses rather than assuming zero. It is here because it is the one
            # fact about a sphere set a reader cannot recover by looking at it.
            "origin": spheres.origin,
            "source": spheres.source,
            "generated_by": (
                "src/robot/safety/planning/robot/build_gripper_spheres.py"
            ),
            "note": note or (
                "tool0 gripper spheres only, frame-correct and directly cuRobo-usable. The arm-link "
                "spheres and the complete robot descriptor are produced on-box by "
                "scripts/curobo/build_ur_config.py, which reads this file. See PROVENANCE.md."
            ),
        },
        "collision_spheres": spheres.to_dict(),
    }


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_gripper_spheres",
        description="Fit a gripper's cuRobo collision spheres from a baked bundle or a mesh file.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--variant",
        default=None,
        help=f"baked bundle to fit, as in {{variant}}_collision_meshes.npz (default {_DEFAULT_VARIANT})",
    )
    source.add_argument("--mesh", default=None, help="mesh file to fit instead of a baked bundle")
    parser.add_argument(
        "--gripper", default="", help="what to call this gripper in the provenance block"
    )
    parser.add_argument(
        "--scale-to-mm",
        type=float,
        default=1.0,
        help="multiply mesh coordinates by this to get millimetres (1000 for a mesh in metres)",
    )
    # No defaults. The pairing of 34.0 with 24.0 is the 2F-85 finger cell size against its palm
    # radius cap, a combination describing no part of any gripper. A default that looks calibrated
    # is worse than one that looks arbitrary.
    parser.add_argument("--cell-mm", type=float, default=None,
                        help="voxel size of the fit, millimetres, mesh only (required with --mesh)")
    parser.add_argument("--rmax-mm", type=float, default=None,
                        help="largest sphere radius, millimetres, mesh only (required with --mesh)")
    parser.add_argument("--origin", default=None, choices=("flange", "mounting_face"),
                        help="where the mesh starts, mesh only (required with --mesh)")
    parser.add_argument(
        "--out", default=None, help="output file (default {variant}_gripper_spheres.yml beside this)"
    )
    args = parser.parse_args(argv)

    # Retired as a writer. The grid fit leaves every shipped hand a hole of 16.5 to 19.0 mm, and a
    # bare --out lands on the name of a committed map, which planner_hand refuses. build() stays a
    # library.
    print(
        "build_gripper_spheres writes no map any more: its grid fit leaves holes the planner calls clear. Write the "
        "hand's body with scripts/grippers/write_hand_from_mesh.py (or write_hand_from_dimensions.py, or "
        "bake_gripper_variant.py --usd), then fit its map with scripts/curobo/fit_cover_spheres.py --hand <hand> --write",
        file=sys.stderr,
    )
    return 2

    try:
        if args.mesh:
            if not args.gripper:
                parser.error("--mesh needs --gripper: the provenance has to say what this is")
            missing = [
                name for name, value in (
                    ("--cell-mm", args.cell_mm),
                    ("--rmax-mm", args.rmax_mm),
                    ("--origin", args.origin),
                ) if value is None
            ]
            if missing:
                parser.error(
                    f"--mesh needs {', '.join(missing)}. None of them can be read off a mesh "
                    "file, and a wrong answer produces a fit that is plausible rather than one "
                    "that fails. For a two-finger hand the shipped bundles use 44 mm cells "
                    "capped at 24 mm for the palm and 34 mm capped at 17 mm for a finger blade."
                )
            fitted = fit_spheres_from_mesh(
                Path(args.mesh),
                gripper=args.gripper,
                cell_mm=args.cell_mm,
                rmax_mm=args.rmax_mm,
                scale_to_mm=args.scale_to_mm,
                origin=args.origin,
            )
        else:
            variant = args.variant or _DEFAULT_VARIANT
            # A hand other than the 2F-85 the arm bundles carry is its own bundle.
            hand_bundle = _BUNDLE_DIR / f"{variant}_hand_meshes.npz"
            fitted = fit_gripper_spheres(
                hand_bundle if hand_bundle.is_file() else _BUNDLE_DIR / f"{variant}_collision_meshes.npz",
                gripper=args.gripper or ("Robotiq 2F-85" if variant == _DEFAULT_VARIANT else variant),
            )
    except SphereFitError as exc:
        print(f"cannot fit: {exc}", file=sys.stderr)
        return 2

    name = args.out or f"{args.variant or _DEFAULT_VARIANT}_gripper_spheres.yml"
    out = Path(name) if Path(name).is_absolute() or "/" in name or "\\" in name else _HERE / name
    out.write_text(
        yaml.safe_dump(build(fitted), default_flow_style=False, sort_keys=False), encoding="utf-8"
    )
    print(f"wrote {out}")
    print(f"  {fitted.render()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
