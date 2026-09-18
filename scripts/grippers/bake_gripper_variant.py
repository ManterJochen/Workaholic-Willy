"""Bake a hand's own collision-mesh bundle out of one standalone hand USD.

    python scripts/grippers/bake_gripper_variant.py --list
    python scripts/grippers/bake_gripper_variant.py --check
    python scripts/grippers/bake_gripper_variant.py robotiq_hande --out <scratch>/robotiq_hande_hand_meshes.npz --write
    python scripts/grippers/bake_gripper_variant.py --usd vendor/acme.usd --hand acme_2f --bodies housing,left,right \\
        --closing +Z --approach +Y --binormal -X --mount-face-mm -86.10 --origin mounting_face --write

Any vendor USD bakes from the command line. The caller states the file, the three bodies (the housing, the left
finger and the right finger, as prim names or absolute prim paths), which vendor axis the hand closes along,
approaches along and has as its binormal (each one of ``+X -X +Y -Y +Z -Z``), where its mounting face sits along the
approach, and whether its numbers then start at the ``flange`` or at its own ``mounting_face``. The catalogue below
holds presets that resolve to exactly those arguments, and ``--list`` prints each as its command line.

A hand bundle is a hand and nothing else. This writes ``{hand}_hand_meshes.npz``, which the guard composes onto
whichever arm the cell has at load, and records its row in ``bundles.json`` beside it. A freshly baked hand carries no
list of arms: a planner starts on an arm and hand only with a committed evidence file for the combination
(``scripts/curobo/matrix_gate.py``), and an ik cell runs no planner, so there the exact mesh guard alone decides.

No Isaac and no GPU, because opening a stage needs neither. ``pxr`` comes from ``pip install usd-core`` (pinned in
``requirements.txt``) or from Isaac's own interpreter.

It proves itself before it is trusted. Before every bake it reads Isaac's standalone 2F-85 asset, places it by the
same reasoning, and diffs it against the committed bundle, and a failure writes nothing. A box without Isaac's assets
can skip that control only by saying why (``--skip-control REASON``), and the reason lands in the bundle's recipe,
because a wrong frame here silently corrupts a safety guard and a mirrored gripper has identical extents.

The frame change and the write are ``src/robot/safety/planning/robot/hand_from_mesh.py``, the one every hand route
shares. Units are demanded from the stage: a stage in anything but metres is baked only when ``--scale-to-mm``
states the same scale the stage declares.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from src.robot.safety.planning.robot.gripper_spheres import (  # noqa: E402
    FLANGE,
    MOUNTING_FACE,
)

_DATA = _REPO / "src" / "robot" / "safety" / "data"

#: The three arrays that describe the end effector, housing first.
_HAND_KEYS = ("gripper", "lfinger", "rfinger")


def _mesh_lib() -> Any:
    name = "willy_hand_from_mesh"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _REPO / "src/robot/safety/planning/robot/hand_from_mesh.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@dataclass(frozen=True, slots=True)
class GripperAsset:
    """A preset: where a vendor gripper lives and how its own frame relates to the bundle's, as the words a CLI takes."""

    key: str
    #: Relative to Isaac's robot asset root.
    usd: str
    #: The bodies to read, in the bundle's order: the housing first, then the left and the right finger.
    bodies: tuple[str, str, str]
    #: Asset-frame coordinate of the mounting face along the asset's approach axis, millimetres.
    mount_face_mm: float
    closing: str
    approach: str
    binormal: str
    #: Where the placed arrays start once the mounting face is at zero. A hand read out of a composed arm asset was
    #: already placed by the arm, so its origin is the flange; a standalone vendor asset's is its mounting face. The
    #: difference is one coupling plate, and a sphere set one plate too close to the flange looks reasonable.
    origin: str
    note: str

    @property
    def rotation(self) -> tuple[tuple[float, float, float], ...]:
        return _mesh_lib().VendorAxes.from_words(closing=self.closing, approach=self.approach, binormal=self.binormal).rotation

    @property
    def approach_axis(self) -> int:
        return "XYZ".index(self.approach[1])

    def cli_line(self) -> str:
        return (f"--usd <Isaac Robots>/{self.usd} --hand {self.key} --bodies {','.join(self.bodies)} "
                f"--closing {self.closing} --approach {self.approach} --binormal {self.binormal} "
                f"--mount-face-mm {self.mount_face_mm:g} --origin {self.origin}")


#: The presets. The two do not share a frame, and that is a fact about the assets rather than an oversight: Isaac
#: authors the 2F-85 with the approach on +Z and the closing on Y, and the Hand-E with the approach on +Y and the
#: closing on Z. The 2F-85 is here because it is the control.
CATALOGUE: tuple[GripperAsset, ...] = (
    GripperAsset(
        key="robotiq_2f85",
        usd="Robotiq/2F-85/Robotiq_2F_85_edit.usd",
        bodies=("base_link", "left_inner_finger", "right_inner_finger"),
        # Reading this asset in its own root frame reproduces the committed bundle to 0.00 mm on all six corners of
        # the palm, so this asset root is the UR flange. Its body starts 3.36 mm behind that root, which is the
        # mounting boss sitting inside the flange.
        mount_face_mm=0.0,
        closing="+Y", approach="+Z", binormal="+X",
        origin=FLANGE,
        note="the control: this must reproduce the committed ur5e bundle",
    ),
    GripperAsset(
        key="robotiq_hande",
        usd="Robotiq/Hand-E/Robotiq_Hand_E_edit.usd",
        bodies=("base_link", "left_gripper", "right_gripper"),
        # The housing spans y [-86.10, 0.00]; the coupling bolts reach 4.90 mm further back and are kept, because they
        # are real geometry and a collision model that drops real geometry is optimistic in the one direction that
        # matters. The binormal's sign puts `left_gripper` at negative x, where the committed 2F-85 puts `lfinger__v`.
        # The asset holds no coupling part at all, so these numbers start at the gripper's own mounting face.
        mount_face_mm=-86.10,
        closing="+Z", approach="+Y", binormal="-X",
        origin=MOUNTING_FACE,
        note="Robotiq Hand-E, 50 mm stroke, two prismatic fingers of 25 mm each",
    ),
)

_BY_KEY = {a.key: a for a in CATALOGUE}

#: Isaac's asset root. The one path this repository cannot relocate, so it is read from the environment first and
#: only then guessed, as `scripts/curobo/build_ur_config.py` does.
_ASSET_HINTS = (
    "D:/isaacsim_assets/Assets/Isaac/5.1/Isaac/Robots",
    "C:/isaacsim_assets/Assets/Isaac/5.1/Isaac/Robots",
)


def _asset_root() -> Path:
    import os

    override = os.environ.get("WILLY_ISAAC_ASSETS")
    candidates = [Path(override)] if override else []
    candidates += [Path(h) for h in _ASSET_HINTS]
    for path in candidates:
        if path.is_dir():
            return path
    raise SystemExit(
        "could not locate the Isaac robot assets, which the 2F-85 control reads before any bake.\n"
        f"  searched: {', '.join(str(c) for c in candidates)}\n"
        "  set WILLY_ISAAC_ASSETS=<...>/Assets/Isaac/<version>/Isaac/Robots, or pass --skip-control with the reason "
        "this box cannot run it"
    )


def _read_usd_parts(
    usd: Path, bodies: tuple[str, str, str], *, scale_to_mm: float | None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The three bodies of ``usd`` in the stage's own axes, in millimetres, relative to the default prim.

    A body is a prim name or an absolute prim path, and a name that two prims carry is refused naming both. The
    stage's unit is demanded: anything but metres is read only when ``scale_to_mm`` states the same millimetres per
    unit. Each body comes with its faces, because the Coal backend builds a BVH from both and the sphere fit reads the
    vertices.
    """
    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover (the message is the point)
        raise SystemExit(
            f"pxr will not load ({exc}). Either `pip install usd-core` in the project venv, or run this with Isaac's "
            "own interpreter. No simulator is needed either way."
        ) from None

    if not usd.is_file():
        raise SystemExit(f"no USD at {usd}")
    stage = Usd.Stage.Open(str(usd), Usd.Stage.LoadAll)
    if stage is None:
        raise SystemExit(f"could not open {usd}")
    metres_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    declared = metres_per_unit * 1000.0
    if not 0.999 < metres_per_unit < 1.001:
        if scale_to_mm is None or abs(float(scale_to_mm) - declared) > 1e-6 * declared:
            raise SystemExit(
                f"{usd.name} declares metersPerUnit={metres_per_unit}, which is {declared:g} mm per unit and not the metres "
                f"every vendor asset measured so far uses. Bake it with --scale-to-mm {declared:g} if that is right: a "
                f"scale error here is a hand a thousand times the wrong size."
            )
    elif scale_to_mm is not None and abs(float(scale_to_mm) - 1000.0) > 1e-6:
        raise SystemExit(f"{usd.name} is in metres, and --scale-to-mm {scale_to_mm:g} says otherwise")

    root = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    by_name: dict[str, list[Any]] = {}
    for prim in stage.TraverseAll():
        by_name.setdefault(prim.GetName(), []).append(prim)

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    reference = None
    for key, selector in zip(_HAND_KEYS, bodies, strict=True):
        if selector.startswith("/"):
            prim = stage.GetPrimAtPath(selector)
            found = [prim] if prim and prim.IsValid() else []
        else:
            found = by_name.get(selector, [])
        if len(found) != 1:
            if not found:
                raise SystemExit(f"no prim {selector!r} in {usd.name}; it holds {sorted(n for n in by_name if n)[:40]}")
            raise SystemExit(f"{selector!r} names {len(found)} prims in {usd.name}: "
                             f"{', '.join(str(p.GetPath()) for p in found)}. Name the body by its absolute path")
        prim = found[0]
        if reference is None:
            reference = cache.GetLocalToWorldTransform(root if root and root.IsValid() else prim).GetInverse()

        verts: list[np.ndarray] = []
        faces: list[list[int]] = []
        base = 0
        for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if child.GetTypeName() != "Mesh":
                continue
            mesh = UsdGeom.Mesh(child)
            points = mesh.GetPointsAttr().Get()
            if not points:
                continue
            to_root = np.array(cache.GetLocalToWorldTransform(child) * reference, dtype=np.float64)
            raw = np.array([[p[0], p[1], p[2]] for p in points], dtype=np.float64)
            local = (np.hstack([raw, np.ones((len(raw), 1))]) @ to_root)[:, :3] * declared
            counts = mesh.GetFaceVertexCountsAttr().Get() or []
            indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
            cursor = 0
            for count in counts:
                face = indices[cursor:cursor + count]
                cursor += count
                for i in range(1, count - 1):
                    faces.append([base + face[0], base + face[i], base + face[i + 1]])
            base += len(local)
            verts.append(local)
        if not verts:
            raise SystemExit(f"{selector!r} in {usd.name} carries no mesh points")
        out[key] = (np.vstack(verts), np.array(faces, dtype=np.int32))
    return out


def _bundle(asset: GripperAsset, usd: Path, *, scale_to_mm: float | None) -> Any:
    lib = _mesh_lib()
    return lib.HandBundle.from_parts(
        parts=_read_usd_parts(usd, asset.bodies, scale_to_mm=scale_to_mm),
        axes=lib.VendorAxes.from_words(closing=asset.closing, approach=asset.approach, binormal=asset.binormal),
        scale_to_mm=1.0, mount_face_mm=asset.mount_face_mm, origin=asset.origin,
    )


def _check_against_2f85() -> tuple[int, float]:
    """Read Isaac's standalone 2F-85 and diff it against the committed bundle: (exit code, worst corner deviation)."""
    preset = _BY_KEY["robotiq_2f85"]
    hand = _bundle(preset, _asset_root() / preset.usd, scale_to_mm=None).arrays()
    committed = _DATA / "ur5e_collision_meshes.npz"
    if not committed.is_file():
        print(f"no committed bundle at {committed}", file=sys.stderr)
        return 1, float("inf")
    worst = 0.0
    with np.load(committed) as reference:
        for key in _HAND_KEYS:
            got, want = hand[f"{key}__v"], reference[f"{key}__v"]
            lo_got, hi_got = got.min(axis=0), got.max(axis=0)
            lo_want, hi_want = want.min(axis=0), want.max(axis=0)
            deviation = float(max(np.abs(lo_got - lo_want).max(), np.abs(hi_got - hi_want).max()))
            worst = max(worst, deviation)
            print(f"  {key:<9} read {len(got):>6} verts, committed {len(want):>6}, worst corner deviation {deviation:.2f} mm")
    # The committed 2F-85 comes from the UR5e asset's baked Gripper variant, a simplified three-body model, while the
    # standalone asset is the nine-body articulated one. The shapes have to agree; the vertex counts do not.
    limit = 1.0
    print(f"  worst deviation over all three arrays: {worst:.2f} mm (limit {limit:.2f})")
    if worst > limit:
        print("  FAILED: the reader does not reproduce the committed 2F-85, so no hand it reads should be trusted.",
              file=sys.stderr)
        return 1, worst
    print("  the reader reproduces the committed 2F-85. A hand it reads can be trusted this far.")
    return 0, worst


def main(argv: list[str] | None = None) -> int:
    lib = _mesh_lib()
    parser = argparse.ArgumentParser(
        prog="python scripts/grippers/bake_gripper_variant.py",
        description="Bake a hand bundle, the hand alone, out of one standalone hand USD.",
    )
    parser.add_argument("preset", nargs="?", help=f"a preset, one of: {', '.join(_BY_KEY)}")
    parser.add_argument("--list", action="store_true", help="print every preset as its command line and exit")
    parser.add_argument("--check", action="store_true", help="read the 2F-85 and diff it against the committed bundle")
    parser.add_argument("--usd", default=None, help="the vendor USD to bake")
    parser.add_argument("--hand", default=None, help="the registry name the hand will have")
    parser.add_argument("--bodies", default=None, help="housing,left finger,right finger: prim names or absolute paths")
    for word in ("closing", "approach", "binormal"):
        parser.add_argument(f"--{word}", default=None, choices=lib.AXIS_WORDS, help=f"the vendor axis of the {word}")
    parser.add_argument("--mount-face-mm", type=float, default=None, help="the mounting face along the approach, mm")
    parser.add_argument("--origin", default=None, choices=(FLANGE, MOUNTING_FACE))
    parser.add_argument("--scale-to-mm", type=float, default=None,
                        help="millimetres per stage unit, required for a stage that is not in metres")
    parser.add_argument("--out", default=None, help="where to write; the committed bundle path by default")
    parser.add_argument("--skip-control", default=None, metavar="REASON",
                        help="skip the 2F-85 control, saying why; the reason is recorded in the bundle")
    parser.add_argument("--write", action="store_true", help="write the hand bundle, {hand}_hand_meshes.npz")
    args = parser.parse_args(lib.joined_axis_words(list(sys.argv[1:] if argv is None else argv)))

    if args.list or not (args.preset or args.check or args.usd):
        for asset in CATALOGUE:
            print(f"{asset.key}: {asset.cli_line()}")
            print(f"    {asset.note}")
        return 0

    if args.check:
        print("checking the reader against the committed Robotiq 2F-85")
        return _check_against_2f85()[0]

    if args.preset:
        preset = _BY_KEY.get(args.preset)
        if preset is None:
            print(f"unknown preset {args.preset!r}; one of: {', '.join(_BY_KEY)}", file=sys.stderr)
            return 2
        asset = preset
        usd = Path(args.usd) if args.usd else _asset_root() / preset.usd
    else:
        missing = [flag for flag, value in (("--hand", args.hand), ("--bodies", args.bodies), ("--closing", args.closing),
                                            ("--approach", args.approach), ("--binormal", args.binormal),
                                            ("--mount-face-mm", args.mount_face_mm), ("--origin", args.origin))
                   if value is None]
        if missing:
            print(f"a bake from --usd states every number that places the hand; missing {', '.join(missing)}",
                  file=sys.stderr)
            return 2
        bodies = tuple(body.strip() for body in args.bodies.split(","))
        if len(bodies) != 3 or not all(bodies):
            print(f"--bodies names exactly three bodies, the housing and the two fingers; got {args.bodies!r}",
                  file=sys.stderr)
            return 2
        asset = GripperAsset(key=args.hand, usd=str(args.usd), bodies=bodies, mount_face_mm=float(args.mount_face_mm),
                             closing=args.closing, approach=args.approach, binormal=args.binormal, origin=args.origin,
                             note="baked from the command line")
        usd = Path(args.usd)

    # The control runs first unless a reason is given, and a failure stops the write.
    if args.skip_control is not None:
        if not args.skip_control.strip():
            print("--skip-control needs a reason, and a blank one is not a reason: the control is what makes a baked "
                  "hand trustworthy", file=sys.stderr)
            return 2
        print(f"the 2F-85 control is SKIPPED: {args.skip_control.strip()}")
        control: dict[str, Any] = {"skipped": args.skip_control.strip()}
    else:
        print("checking the reader against the committed Robotiq 2F-85 first")
        code, worst = _check_against_2f85()
        if code:
            return 1
        control = {"worst_corner_deviation_mm": worst}

    print(f"\nreading {asset.key} from {usd.name}")
    try:
        bundle = _bundle(asset, usd, scale_to_mm=args.scale_to_mm)
        arrays = bundle.arrays()
    except ValueError as exc:
        raise SystemExit(f"{asset.key}: {exc}") from None
    refusal = lib.hand_bundle_rules().hand_bundle_refusal(arrays, name=f"{asset.key}_hand_meshes.npz")
    if refusal is not None:
        raise SystemExit(f"{asset.key}: {refusal}")
    for key in _HAND_KEYS:
        v = arrays[f"{key}__v"]
        lo, hi = v.min(axis=0), v.max(axis=0)
        print(f"  {key:<9} {len(v):>6} verts  x [{lo[0]:8.2f},{hi[0]:8.2f}]"
              f"  y [{lo[1]:8.2f},{hi[1]:8.2f}]  z [{lo[2]:8.2f},{hi[2]:8.2f}]")
    if not args.write:
        print("\nnothing written; pass --write")
        return 0

    out = Path(args.out) if args.out else _DATA / f"{asset.key}_hand_meshes.npz"
    recipe = {"writer": "scripts/grippers/bake_gripper_variant.py", "usd": usd.name, "bodies": list(asset.bodies),
              "closing": asset.closing, "approach": asset.approach, "binormal": asset.binormal,
              "mount_face_mm": asset.mount_face_mm, "origin": asset.origin, "control": control}
    try:
        report = bundle.write(out, records={"hand__recipe": json.dumps(recipe, sort_keys=True)})
    except ValueError as exc:
        raise SystemExit(f"{asset.key}: {exc}") from None
    print(report.render())

    from src.robot.safety.planning.bundle_index import INDEX_NAME, record_hand_bundle

    if (out.parent / INDEX_NAME).is_file():
        row = record_hand_bundle(
            out, source="standalone_usd",
            note=(f"the standalone USD {usd.name}, bodies {', '.join(asset.bodies)}, closing {asset.closing}, approach "
                  f"{asset.approach}, binormal {asset.binormal}, mounting face {asset.mount_face_mm:g} mm, starting at "
                  f"the {asset.origin}"),
        )
        print(f"  {row.render()}")
    else:
        print(f"  no {INDEX_NAME} beside {out.name}, so no row was recorded")
    print("\nnext, in the project venv:")
    print(f"  python scripts/grippers/measure_jaw_from_bundle.py {asset.key}")
    print(f"  python scripts/curobo/fit_cover_spheres.py --hand {asset.key} --write")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
