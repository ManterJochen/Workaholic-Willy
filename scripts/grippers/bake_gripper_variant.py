"""Bake a collision-mesh bundle for a cell running a gripper other than the baked Robotiq 2F-85.

    python scripts/grippers/bake_gripper_variant.py --list
    python scripts/grippers/bake_gripper_variant.py robotiq_hande --arm ur5e --check
    python scripts/grippers/bake_gripper_variant.py robotiq_hande --arm ur5e --write

⛔ **A VARIANT BUNDLE IS AN ARM PLUS A HAND, NOT A HAND.** ``schunk_egu50_collision_meshes.npz``
carries the UR5e's own arm meshes, and ``_variant_is_for_another_model`` compares a probe link
against the model's own bundle, so naming that variant on a ur3e cell trips ``variant_model_mismatch``
and drops the WHOLE cell to the capsule proxy. A gripper therefore needs one bundle per arm it is
mounted on, and ``--arm`` is not a convenience.

**NO ISAAC AND NO GPU.** The arm meshes are already committed and correct, so this script never
re-derives them: it copies the arm arrays from ``{arm}_collision_meshes.npz`` verbatim and replaces
only ``gripper__v`` / ``lfinger__v`` / ``rfinger__v``. That is exactly what a variant is. The only
thing it computes is the new hand, read out of a vendor USD. ``scripts/isaac/bake_ur_collision_meshes.py``
needs Isaac because it reads a COMPOSED articulation; opening a stage needs neither.

**IT PROVES ITSELF BEFORE IT IS TRUSTED.** ``--check`` reads the standalone 2F-85 asset, places it
by the same reasoning, and diffs it against the committed ``ur5e_collision_meshes.npz``. Only trust a
newly baked hand if that reproduces, because a wrong frame here silently corrupts a safety guard and
a mirrored gripper has identical extents. That is the gate ``bake_ur_collision_meshes.py`` sets for
arms, applied to hands.

THE FRAME, read out of the committed bundle rather than assumed::

    approach   +Y     palm y [-3.36, 90.00], fingers y [91.45, 148.47]
    closing     X     finger centroids 114.96 mm apart along x, nothing along y or z
    binormal    Z     finger z size 27.00, which is `finger_width_mm: 27.0` exactly

Units are millimetres in the bundle and metres in every vendor USD measured so far, so the scale is
demanded from the stage rather than assumed: metres read as millimetres is a hand a thousand times
too small that plans happily through everything it should have hit.

``pxr`` comes from ``pip install usd-core`` (pinned in ``requirements/dev.txt``) or from Isaac's own
interpreter. Neither a simulator nor a GPU is involved either way.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.robot.safety.planning.robot.gripper_spheres import (  # noqa: E402
    FLANGE,
    MOUNTING_FACE,
    _ORIGIN_KEY,
)

_REPO = Path(__file__).resolve().parents[2]
_DATA = _REPO / "src" / "robot" / "safety" / "data"

#: The three arrays that describe the end effector. Everything else in a bundle is the arm.
_HAND_KEYS = ("gripper", "lfinger", "rfinger")


@dataclass(frozen=True, slots=True)
class GripperAsset:
    """Where a vendor gripper lives and how its own frame relates to the bundle's.

    ``rotation`` maps the asset's axes onto the bundle's, as ``new = old @ rotation.T``. It is a
    matrix rather than an axis permutation because a permutation can be a REFLECTION: swapping two
    axes has determinant -1 and mirrors the hand, and a mirrored symmetric gripper has identical
    extents, so no assertion about sizes would catch it. The determinant check in
    :func:`_read_gripper` is the only thing that does.
    """

    key: str
    usd: str
    #: The rigid bodies to read, in the bundle's order: the body first, then the two fingers.
    bodies: tuple[str, str, str]
    #: Asset-frame coordinate of the MOUNTING FACE along the asset's approach axis, millimetres.
    #: Subtracted before the rotation, so the bundle's y = 0 is the flange.
    mount_face_mm: float
    #: Which asset axis carries the approach, as an index into (x, y, z). Used only to apply
    #: ``mount_face_mm``, and named so the two cannot silently disagree.
    approach_axis: int
    rotation: tuple[tuple[float, float, float], ...]
    #: Where the placed arrays START, once the mounting face has been moved to zero. Two hands,
    #: two answers: a bundle read out of a COMPOSED arm asset was already placed by the arm, so
    #: its origin is the flange, while a standalone vendor asset has no idea what it will be
    #: bolted to and its origin is the mounting face. The difference is one coupling plate, it is
    #: a bench measurement, and a sphere set one plate too close to the flange looks reasonable.
    origin: str
    note: str


#: Every gripper this repository can bake a bundle for.
#:
#: The two entries do NOT share a frame, and that is a fact about the assets rather than an oversight:
#: Isaac authors the 2F-85 with the approach on +Z and the closing on Y, and the Hand-E with the
#: approach on +Y and the closing on Z. A single "Robotiq convention" does not exist, which is why
#: each entry states its own and why the 2F-85 is baked here at all: it is the control.
CATALOGUE: tuple[GripperAsset, ...] = (
    GripperAsset(
        key="robotiq_2f85",
        usd="Robotiq/2F-85/Robotiq_2F_85_edit.usd",
        bodies=("base_link", "left_inner_finger", "right_inner_finger"),
        mount_face_mm=0.0,
        approach_axis=2,
        # (x, y, z)_asset -> (y, z, x)_bundle. A cyclic permutation, determinant +1.
        rotation=((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        # MEASURED: reading this asset in its own root frame reproduces the committed bundle to
        # 0.00 mm on all six corners of the palm, so this asset root IS the UR flange. Its body
        # starts 3.36 mm behind that root, which is the mounting boss sitting inside the flange.
        origin=FLANGE,
        note="the control: this must reproduce the committed ur5e bundle",
    ),
    GripperAsset(
        key="robotiq_hande",
        usd="Robotiq/Hand-E/Robotiq_Hand_E_edit.usd",
        bodies=("base_link", "left_gripper", "right_gripper"),
        # Group1 is the housing and spans y [-86.10, 0.00]; the coupling bolts reach 4.90 mm further
        # back and are KEPT, because they are real geometry and a collision model that drops real
        # geometry is optimistic in the one direction that matters.
        mount_face_mm=-86.10,
        approach_axis=1,
        # (x, y, z)_asset -> (z, y, -x)_bundle. Determinant +1. The sign puts the body the asset calls
        # `left_gripper` at NEGATIVE x, which is where the committed 2F-85 puts `lfinger__v`.
        rotation=((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),
        # MEASURED: the housing spans 99.20 mm, which is the published body length, and the asset
        # holds no coupling part at all. So these numbers start at the gripper own mounting face
        # and the plate between it and the flange is not in them.
        origin=MOUNTING_FACE,
        note="Robotiq Hand-E, 50 mm stroke, two prismatic fingers of 25 mm each",
    ),
)

_BY_KEY = {a.key: a for a in CATALOGUE}

#: Isaac's asset root. The one path this repository cannot relocate, so it is read from the
#: environment first and only then guessed, exactly as `scripts/curobo/build_ur_config.py` does.
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
        "could not locate the Isaac robot assets.\n"
        f"  searched: {', '.join(str(c) for c in candidates)}\n"
        "  set WILLY_ISAAC_ASSETS=<...>/Assets/Isaac/<version>/Isaac/Robots"
    )


def _read_gripper(asset: GripperAsset) -> dict[str, np.ndarray]:
    """The three hand arrays in the bundle frame, millimetres.

    Vertices only, and faces beside them: the Coal backend builds a BVH from both and the sphere fit
    reads the vertices, so a bundle without faces is half a bundle.
    """
    try:
        from pxr import Usd, UsdGeom
    except ImportError:  # pragma: no cover - the message is the point
        raise SystemExit(
            "pxr is not importable. Either `pip install usd-core` in the project venv, or run this "
            "with Isaac's own interpreter. No simulator is needed either way."
        ) from None

    rotation = np.array(asset.rotation, dtype=np.float64)
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > 1e-9:
        raise SystemExit(
            f"{asset.key}: the frame change has determinant {determinant:+.6f}, so it is a reflection "
            "rather than a rotation and would mirror the gripper. A mirrored symmetric hand has "
            "identical extents, so nothing downstream would notice."
        )

    path = _asset_root() / asset.usd
    if not path.is_file():
        raise SystemExit(f"{asset.key}: no asset at {path}")
    stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    if stage is None:
        raise SystemExit(f"{asset.key}: could not open {path}")

    metres_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    scale = metres_per_unit * 1000.0
    if not 0.999 < metres_per_unit < 1.001:
        raise SystemExit(
            f"{asset.key}: the stage declares metersPerUnit={metres_per_unit}, which is not the "
            "metres every vendor asset measured so far uses. Check it by hand before trusting a "
            "bundle from it: a scale error here is a hand a thousand times the wrong size."
        )

    root = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    bodies = {}
    for prim in stage.TraverseAll():
        bodies[prim.GetName()] = prim

    out: dict[str, np.ndarray] = {}
    reference = None
    for key, body_name in zip(_HAND_KEYS, asset.bodies, strict=True):
        prim = bodies.get(body_name)
        if prim is None or not prim.IsValid():
            raise SystemExit(
                f"{asset.key}: no prim named {body_name!r} in {path.name}. The bodies are "
                f"{sorted(n for n in bodies if n)}"
            )
        if reference is None:
            reference = cache.GetLocalToWorldTransform(
                root if root and root.IsValid() else prim
            ).GetInverse()

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
            to_root = np.array(
                cache.GetLocalToWorldTransform(child) * reference, dtype=np.float64
            )
            raw = np.array([[p[0], p[1], p[2]] for p in points], dtype=np.float64)
            local = (np.hstack([raw, np.ones((len(raw), 1))]) @ to_root)[:, :3] * scale
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
            raise SystemExit(f"{asset.key}: {body_name!r} carries no mesh points")

        stacked = np.vstack(verts)
        stacked[:, asset.approach_axis] -= asset.mount_face_mm
        out[f"{key}__v"] = stacked @ rotation.T
        out[f"{key}__f"] = np.array(faces, dtype=np.int32)
    return out


def _check_against_2f85(hand: dict[str, np.ndarray]) -> int:
    """Diff a freshly read 2F-85 against the committed bundle. Returns an exit code."""
    committed = _DATA / "ur5e_collision_meshes.npz"
    if not committed.is_file():
        print(f"no committed bundle at {committed}", file=sys.stderr)
        return 1
    with np.load(committed) as reference:
        worst = 0.0
        for key in _HAND_KEYS:
            got, want = hand[f"{key}__v"], reference[f"{key}__v"]
            lo_got, hi_got = got.min(axis=0), got.max(axis=0)
            lo_want, hi_want = want.min(axis=0), want.max(axis=0)
            deviation = float(max(np.abs(lo_got - lo_want).max(), np.abs(hi_got - hi_want).max()))
            worst = max(worst, deviation)
            print(f"  {key:<9} read {len(got):>6} verts, committed {len(want):>6}")
            print(f"    read      x [{lo_got[0]:8.2f},{hi_got[0]:8.2f}]"
                  f"  y [{lo_got[1]:8.2f},{hi_got[1]:8.2f}]  z [{lo_got[2]:8.2f},{hi_got[2]:8.2f}]")
            print(f"    committed x [{lo_want[0]:8.2f},{hi_want[0]:8.2f}]"
                  f"  y [{lo_want[1]:8.2f},{hi_want[1]:8.2f}]  z [{lo_want[2]:8.2f},{hi_want[2]:8.2f}]")
            print(f"    worst corner deviation {deviation:.2f} mm")
    # The committed 2F-85 comes from the UR5e asset's baked Gripper variant, which is a simplified
    # three-body model, while the standalone asset is the nine-body articulated one at its own
    # aperture. The SHAPES have to agree; the vertex counts do not, and demanding they did would be
    # comparing two different meshes of one gripper.
    limit = 1.0
    print(f"\n  worst deviation over all three arrays: {worst:.2f} mm (limit {limit:.2f})")
    if worst > limit:
        print("  FAILED: the reader does not reproduce the committed 2F-85, so no hand it reads "
              "should be trusted. The difference above says which axis is wrong.", file=sys.stderr)
        return 1
    print("  the reader reproduces the committed 2F-85. A hand it reads can be trusted this far.")
    return 0


def _write_variant(gripper: GripperAsset, arm: str, hand: dict[str, np.ndarray]) -> int:
    arm_bundle = _DATA / f"{arm}_collision_meshes.npz"
    if not arm_bundle.is_file():
        print(f"no arm bundle at {arm_bundle}; the arms present are "
              f"{sorted(p.name for p in _DATA.glob('*_collision_meshes.npz'))}", file=sys.stderr)
        return 2
    out = _DATA / f"{gripper.key}_{arm}_collision_meshes.npz"

    with np.load(arm_bundle) as source:
        payload = {name: source[name] for name in source.files}
    replaced = []
    for key in _HAND_KEYS:
        for suffix in ("__v", "__f"):
            name = f"{key}{suffix}"
            if name not in payload:
                print(f"{arm_bundle.name} has no {name}; it is not an arm-plus-hand bundle",
                      file=sys.stderr)
                return 2
            payload[name] = hand[name]
            replaced.append(name)
    # The frame marker travels with the arrays it labels. `_fcl_self_collision` reads `{part}__frame`
    # to decide which DH frame a mesh is placed from, and the hand sits at frame 6 on every UR flange
    # whatever hand it is, so the arm bundle's own value is carried through untouched.
    # The origin travels WITH the arrays, because it is the one fact about them a reader cannot
    # recover by looking: a sphere set placed one coupling plate too close to the flange has
    # entirely reasonable numbers.
    payload[_ORIGIN_KEY] = np.array([gripper.origin])
    np.savez_compressed(out, **payload)
    print(f"wrote {out.name}")
    print(f"  arm links copied verbatim from {arm_bundle.name}")
    print(f"  origin: {gripper.origin}")
    print(f"  hand arrays replaced: {', '.join(replaced)}")
    for key in _HAND_KEYS:
        v = payload[f"{key}__v"]
        lo, hi = v.min(axis=0), v.max(axis=0)
        print(f"  {key:<9} {len(v):>6} verts  x [{lo[0]:8.2f},{hi[0]:8.2f}]"
              f"  y [{lo[1]:8.2f},{hi[1]:8.2f}]  z [{lo[2]:8.2f},{hi[2]:8.2f}]")
    print("\nnext, in the project venv:")
    print(f"  python -m src.robot.safety.planning.robot.build_gripper_spheres "
          f"--variant {gripper.key}_{arm}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/grippers/bake_gripper_variant.py",
        description="Bake an arm-plus-hand collision bundle for a non-default gripper.",
    )
    parser.add_argument("gripper", nargs="?", help=f"one of: {', '.join(_BY_KEY)}")
    parser.add_argument("--arm", default="ur5e", help="the arm whose links the bundle carries")
    parser.add_argument("--list", action="store_true", help="print the catalogue and exit")
    parser.add_argument(
        "--check", action="store_true",
        help="read the 2F-85 and diff it against the committed bundle, then exit",
    )
    parser.add_argument("--write", action="store_true", help="write the variant bundle")
    args = parser.parse_args(argv)

    if args.list or (not args.gripper and not args.check):
        print(f"{'key':<16} {'arm bodies':<52} note")
        for asset in CATALOGUE:
            print(f"{asset.key:<16} {', '.join(asset.bodies):<52} {asset.note}")
        print(f"\narm bundles present: "
              f"{', '.join(sorted(p.name.split('_collision')[0] for p in _DATA.glob('*_collision_meshes.npz')))}")
        return 0

    if args.check:
        print("checking the reader against the committed Robotiq 2F-85")
        return _check_against_2f85(_read_gripper(_BY_KEY["robotiq_2f85"]))

    gripper = _BY_KEY.get(args.gripper or "")
    if gripper is None:
        print(f"unknown gripper {args.gripper!r}; one of: {', '.join(_BY_KEY)}", file=sys.stderr)
        return 2

    # The control runs first, every time, and a failure stops the write. A bundle nobody checked is
    # exactly the artifact this script exists to stop being produced by hand.
    print("checking the reader against the committed Robotiq 2F-85 first")
    if _check_against_2f85(_read_gripper(_BY_KEY["robotiq_2f85"])):
        return 1

    print(f"\nreading {gripper.key}")
    hand = _read_gripper(gripper)
    if not args.write:
        for key in _HAND_KEYS:
            v = hand[f"{key}__v"]
            lo, hi = v.min(axis=0), v.max(axis=0)
            print(f"  {key:<9} {len(v):>6} verts  x [{lo[0]:8.2f},{hi[0]:8.2f}]"
                  f"  y [{lo[1]:8.2f},{hi[1]:8.2f}]  z [{lo[2]:8.2f},{hi[2]:8.2f}]")
        print("\nnothing written; pass --write")
        return 0
    return _write_variant(gripper, args.arm.lower(), hand)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
