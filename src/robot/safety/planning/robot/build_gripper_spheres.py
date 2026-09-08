"""Write a gripper's collision-sphere map for the planner, from a baked bundle or from a mesh file.

The planner models the robot as spheres. A gripper mesh is never used directly: it is what a sphere
set is fitted from, once, and the spheres are all the planner ever sees. So this is where "the
planner knows your gripper" is decided, and a cell whose sphere map describes a different hand plans
against a different robot without anything saying so.

Run it with the project venv. No simulator, no GPU:

    .venv/Scripts/python.exe -m src.robot.safety.planning.robot.build_gripper_spheres
    .venv/Scripts/python.exe -m src.robot.safety.planning.robot.build_gripper_spheres \\
        --variant schunk_egu50
    .venv/Scripts/python.exe -m src.robot.safety.planning.robot.build_gripper_spheres \\
        --mesh vendor/eoat.stl --gripper eoat --scale-to-mm 1000 --out eoat_gripper_spheres.yml

With no arguments it regenerates the committed Robotiq 2F-85 map, which is what the reference cell
plans with. A test compares the committed file against this fit, so the two cannot drift.

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
    parser.add_argument("--cell-mm", type=float, default=34.0, help="voxel size of the fit, mesh only")
    parser.add_argument("--rmax-mm", type=float, default=24.0, help="largest sphere radius, mesh only")
    parser.add_argument(
        "--out", default=None, help="output file (default {variant}_gripper_spheres.yml beside this)"
    )
    args = parser.parse_args(argv)

    try:
        if args.mesh:
            if not args.gripper:
                parser.error("--mesh needs --gripper: the provenance has to say what this is")
            fitted = fit_spheres_from_mesh(
                Path(args.mesh),
                gripper=args.gripper,
                cell_mm=args.cell_mm,
                rmax_mm=args.rmax_mm,
                scale_to_mm=args.scale_to_mm,
            )
        else:
            variant = args.variant or _DEFAULT_VARIANT
            fitted = fit_gripper_spheres(
                _BUNDLE_DIR / f"{variant}_collision_meshes.npz",
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
