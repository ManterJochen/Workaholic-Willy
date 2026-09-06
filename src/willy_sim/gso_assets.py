"""Google-Scanned-Objects (GSO) loader: convert curated real-object meshes to USD and drop them into
the sim for realistic industrial-scene variation.

The catalog holds scanned real objects as single ``.obj`` meshes (geometry only, metres). This module
converts a curated subset to USD once, on-box, and exposes each as a
:class:`~src.config.schema.robot.SimObjectConfig` the scene builder loads as a graspable rigid body,
the same external-mesh path YCB uses.

Conversion authors a clean USD that mirrors a pre-rigged YCB asset: a root ``Xform`` (the default
prim) with a child ``UsdGeom.Mesh`` holding the geometry. It does not use
``omni.kit.asset_converter``, whose nested ``/World/<name>/mesh`` Xform tree the runtime
``SingleRigidPrim`` dynamic body cannot manage: it invalidates the physics tensor view with "prim
deleted while used by a shape".

The root-Xform-with-child-Mesh structure is load-bearing. ``add_reference_to_stage`` (scene builder)
defines the referencing prim (``/World/Object``) as an ``Xform`` and then references this USD. A local
``typeName`` opinion (``Xform``) is stronger than a referenced one, so a USD whose default prim is the
``Mesh`` itself composes ``/World/Object`` to type ``Xform`` carrying mesh attributes rather than to a
``Mesh`` gprim. Hydra renders by gprim type, so such a prim renders nothing: no depth and no
instance-id mask, while the collider still cooks from the composed points, which makes the object
physics clutter that is invisible to the camera. With the ``Mesh`` as a child, the child keeps its
``Mesh`` type through the reference, renders, and carries the instance-id ``/World/Object/mesh`` that
the ground-truth perception matches by subtree prefix exactly as it matches YCB. Physics is authored
at runtime by the scene builder (``_author_mesh_collider``: RigidBody on the root, Collision on the
child mesh), so this USD is pure geometry. See :func:`convert_gso_to_usd`.

Curation is measured: only objects whose smallest dimension fits the 2F-85 jaw span (<= ~60 mm) are
marked ``graspable``. The larger objects (mugs, cans) are kept as realistic clutter and distractors,
so the jaw picks the prompted target among them and a mug's own pick is the suction cup's job.

Paths are env-overridable: ``WILLY_GSO_DIR`` is the source ``.obj`` catalog, ``WILLY_GSO_USD_DIR`` is
where the converted USDs are written (default a sibling of the source, kept out of the repo).
Import-safe off-box: Isaac (``pxr``) is imported lazily inside :func:`convert_gso_to_usd`.

    # one-time on-box conversion of the curated subset:
    <isaac-sim>\\python.bat -m src.willy_sim.gso_assets --convert
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config.schema.robot import SimObjectConfig
from src.utility.log_cfg import create_logger
from src.willy_sim.constants import GSO_ASSETS_LOG_FILE, WILLY_SIM_LOG_DIR

#: A one-time, on-box asset conversion whose output every later scene depends on. What is worth
#: keeping is the provenance of each USD: which .obj it came from, how much geometry survived the
#: parse, and how big the result is. A mesh that parsed to a handful of faces still converts and
#: still loads, and is then invisible or uncollidable in a scene much later.
_LOG = create_logger("GsoAssets", log_file=GSO_ASSETS_LOG_FILE, log_dir=WILLY_SIM_LOG_DIR)

ENV_GSO_DIR = "WILLY_GSO_DIR"           #: the source .obj catalog of GSO meshes
ENV_GSO_USD_DIR = "WILLY_GSO_USD_DIR"   #: where converted USDs are written (default: a sibling of the source)
#: The default source catalog: this repository's own mesh library, never another project's
#: directory. The meshes live here with their sha256 index and their licences
#: (`datagen/assets/library.py`). A tool that reaches into another repository to do its job is a
#: dependency nobody declared, so this default stays a repo-relative path.
_DEFAULT_GSO_DIR = "assets/meshes/gso"


@dataclass(frozen=True, slots=True)
class GsoPart:
    """A curated GSO object and how the sim should treat it.

    ``min_dim_mm`` is the measured smallest bounding-box extent, the width the jaw must span.
    ``graspable`` is ``min_dim_mm <= _JAW_GRASPABLE_MAX_MM``; objects above that are realistic
    clutter and distractors.
    """

    obj_name: str          # the .obj base name (no extension) in the GSO catalog
    label: str             # the prompt / detector label
    min_dim_mm: float      # measured smallest bounding-box extent (mm)
    # UsdPhysics collision approximation. boundingCube (a box collider) is the robust default: a dense
    # scan mesh's convexHull or convexDecomposition can fail to cook at physics-init, which drops the
    # body and invalidates the tensor view. A box always cooks and is fine for settling and a flat-top
    # suction seal.
    collision: str = "boundingCube"
    # Approximate real-object display colour (linear RGB 0..1). The flat GSO .obj dump ships geometry
    # and UVs but no texture images, so a per-object colour is the local stand-in for the real texture,
    # and a coloured mesh gives GroundingDINO an appearance cue a uniformly grey one does not. ``None``
    # selects the neutral grey default.
    display_color: "tuple[float, float, float] | None" = None

    @property
    def graspable(self) -> bool:
        return self.min_dim_mm <= _JAW_GRASPABLE_MAX_MM


_JAW_GRASPABLE_MAX_MM = 60.0  # a 2F-85 (85 mm span) grips a min-dimension up to ~60 mm with margin

# The curated subset (measured bounding boxes; see the module docstring). Small toys and thin objects
# are jaw-graspable; the mugs are too wide for the jaw and stay as suction or demo distractors.
GSO_PARTS: list[GsoPart] = [
    GsoPart("Toysmith_Windem_Up_Flippin_Animals_Dog", "toy dog", 35.0),
    GsoPart("Schleich_African_Black_Rhino", "toy rhino", 41.0),
    GsoPart("Great_Dinos_Triceratops_Toy", "toy dinosaur", 46.0),
    GsoPart("OXO_Soft_Works_Can_Opener_SnapLock", "can opener", 56.0),
    GsoPart("ACE_Coffee_Mug_Kristen_16_oz_cup", "coffee mug", 90.0),
    GsoPart("Cole_Hardware_Mug_Classic_Blue", "blue mug", 95.0),
    # Retail packages for the suction story and scene variety, classified by what seals and lifts with
    # the 30 mm cup (the seal model reads the settled top from rendered depth; see SUCTION_TARGET_LABELS):
    #  * ink cartridge box: a uniform flat top (138x131 mm, 0.37 mm RMS). It seals and lifts, and is the
    #    suction target.
    #  * game box (RockSmith cable): settles narrow-ridge-up (a tapered 82x23 top on a 171x140 base), and
    #    its 44 mm min-dim is jaw-graspable anyway, so it is jaw-graspable clutter, not a suction target.
    #  * chocolate box: a domed lid whose only flat patches are recessed ~23 mm below the surrounding
    #    rim, so a 30 mm cup seals but cannot reach them because the rim blocks the descent. Clutter.
    GsoPart("Brother_LC_1053PKS_Ink_Cartridge_CyanMagentaYellow_1pack", "ink cartridge box", 55.0),
    GsoPart("Ubisoft_RockSmith_Real_Tone_Cable_Xbox_360", "game box", 44.0),
    GsoPart("KS_Chocolate_Cube_Box_Assortment_By_Neuhaus_2010_Ounces", "chocolate box", 170.0),
    # --- Diverse bin-clearing catalogue (measured .obj bboxes). The runtime seal and footprint model
    # decides the tool per object; min_dim_mm is the measured smallest extent. Suction = flat or boxy
    # with min-horiz > jaw span; jaw = small and compact; reject = round or hollow. All fit a
    # ~184x286 mm KLT.
    GsoPart("Perricone_MD_Nutritive_Cleanser", "skincare box tall", 60.0),          # suction
    GsoPart("Perricone_MD_Cold_Plasma", "skincare box cube", 66.0),                 # suction
    GsoPart("Wooden_ABC_123_Blocks_50_pack", "wooden blocks box", 63.0),            # suction
    GsoPart("Big_O_Sponges_Assorted_Cellulose_12_pack", "sponge pack", 19.0),       # suction (thin flat)
    GsoPart("Schleich_Lion_Action_Figure", "toy lion", 42.0),                       # jaw
    GsoPart("Android_Figure_Panda", "panda figure", 46.0),                          # jaw
    GsoPart("Kong_Puppy_Teething_Rubber_Small_Pink", "chew toy", 45.0),             # jaw
    GsoPart("Craftsman_Grip_Screwdriver_Phillips_Cushion", "screwdriver", 36.0),    # jaw
    GsoPart("OXO_Cookie_Spatula", "spatula", 25.0),                                 # jaw
    GsoPart("Room_Essentials_Mug_White_Yellow", "white mug", 102.0),                # reject (round)
    GsoPart("Now_Designs_Bowl_Akita_Black", "cereal bowl", 75.0),                   # reject (hollow)
    GsoPart("Toys_R_Us_Treat_Dispenser_Smart_Puzzle_Foobler", "treat ball", 154.0),  # reject (sphere)
    # --- Expose-flagship roster (measured .obj bboxes). A visibly diverse cast so the occlusion demo
    # does not repeat the rhino, ink-box and skincare trio. Two tall-standing Schleich animals as jaw
    # targets: they settle tall, so the low top-down jaw clears self-collision. A colourful kids'
    # shape-sorter box as the flat, low, pushable blocker; it needs no seal because a swept contact push
    # redistributes it rather than suction lifting it. A round mug and a metal C-clamp as distractors.
    GsoPart("Schleich_S_Bayala_Unicorn_70432", "toy unicorn", 55.0),                 # jaw target (tall-standing)
    GsoPart("Schleich_Hereford_Bull", "toy bull", 44.0),                             # jaw target (alt, safest grip)
    GsoPart("GEOMETRIC_SORTING_BOARD", "sorting board", 61.0),                        # blocker (175x174x61 flat box)
    GsoPart("Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White", "porcelain mug", 97.0),  # reject (round)
    GsoPart("Pony_C_Clamp_1440", "metal clamp", 27.0),                               # jaw (graspable distractor)
]

# The GSO package that seals and lifts with the 30 mm suction cup: a uniform flat top whose wide face
# the jaw cannot span. The suction demo target. The game and chocolate boxes seal but do not lift
# (see GSO_PARTS).
SUCTION_TARGET_LABELS = ("ink cartridge box",)

GSO_BY_LABEL: dict[str, GsoPart] = {p.label: p for p in GSO_PARTS}


def gso_source_dir() -> Path:
    """The source ``.obj`` catalog directory (``WILLY_GSO_DIR``, else :data:`_DEFAULT_GSO_DIR`)."""
    return Path(os.environ.get(ENV_GSO_DIR, _DEFAULT_GSO_DIR))


def gso_usd_dir() -> Path:
    """Where converted USDs live (``WILLY_GSO_USD_DIR`` or a ``gso_usd`` sibling of the source catalog)."""
    override = os.environ.get(ENV_GSO_USD_DIR)
    return Path(override) if override else gso_source_dir().parent / "gso_usd"


def gso_usd_path(part: GsoPart) -> Path:
    """The converted-USD path for ``part`` (under :func:`gso_usd_dir`)."""
    return gso_usd_dir() / f"{part.obj_name}.usd"


def gso_object_spec(
    part: GsoPart, *, position_mm: tuple[float, float, float], mass_kg: float = 0.15,
    orientation_wxyz: tuple[float, float, float, float] | None = None,
) -> SimObjectConfig:
    """A :class:`SimObjectConfig` referencing ``part``'s converted USD as a graspable rigid body.

    Requires the USD to exist (run ``--convert`` first). The scene builder loads it via the external-mesh
    path and authors the collision approximation + rigid body (``usd_collision_approximation``).
    """
    usd = gso_usd_path(part)
    if not usd.is_file():
        raise FileNotFoundError(
            f"converted GSO USD missing for {part.label!r}: {usd}. Run "
            f"'python -m src.willy_sim.gso_assets --convert' on-box first."
        )
    return SimObjectConfig(
        name=part.label, usd_asset_path=str(usd), usd_collision_approximation=part.collision,
        position_mm=position_mm, mass_kg=mass_kg, orientation_wxyz=orientation_wxyz,
    )


def _load_obj(path: Path) -> "tuple[Any, Any]":
    """Parse an ``.obj`` into ``(vertices Nx3 float32, faces Mx3 int32)``: 0-based and triangulated.

    Reads ``v x y z`` and ``f`` lines, handles ``v/vt/vn`` index triples and negative (relative)
    indices, and fan-triangulates n-gon faces. No trimesh dependency: this is enough to author a
    clean collision and render mesh, and convexHull cooks without a watertight fill.
    """
    verts: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    v = int(tok.split("/")[0])
                    idx.append(v - 1 if v > 0 else len(verts) + v)  # 1-based, or negative = relative to current
                for k in range(1, len(idx) - 1):  # fan-triangulate
                    faces.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def convert_gso_to_usd(part: GsoPart, *, force: bool = False) -> Path:
    """Author ``part``'s ``.obj`` as a clean root-Xform and child-Mesh USD: pure geometry, on-box.

    A root ``Xform`` (the default prim) with a child ``UsdGeom.Mesh`` holding the geometry. That
    structure is load-bearing: a Mesh as the default prim composes to an ``Xform`` and renders
    nothing, while the collider still cooks. The module docstring gives the composition rule. Physics
    is authored at runtime by the scene builder (``_author_mesh_collider``), so this stays pure
    geometry. Requires a booted ``SimulationApp``, which is what puts ``pxr`` on the path. Skips when
    the USD exists and ``force`` is False.
    """
    src = gso_source_dir() / f"{part.obj_name}.obj"
    dst = gso_usd_path(part)
    if dst.is_file() and not force:
        _LOG.debug("%s: USD already present, skipping conversion (%s)", part.label, dst)
        return dst
    if not src.is_file():
        raise FileNotFoundError(f"GSO source .obj not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    from pxr import Usd, UsdGeom, Vt  # type: ignore[import-not-found]

    verts, faces = _load_obj(src)
    if verts.size == 0 or faces.size == 0:
        raise RuntimeError(f"GSO .obj has no usable geometry: {src}")
    # Centre the mesh at its placement: subtract the XY bbox-centre so the object sits at the nominal
    # x, y under the overhead camera rather than wherever the raw scan's origin fell, and drop the
    # bbox-min Z to 0 so it rests on the table when spawned instead of half-buried or floating. A raw
    # GSO .obj origin is arbitrary.
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    verts = verts - np.array([(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, lo[2]], dtype=np.float32)
    lo, hi = verts.min(axis=0), verts.max(axis=0)

    prim_name = "".join(c if c.isalnum() else "_" for c in part.obj_name)  # a valid USD prim name
    root_path = f"/{prim_name}"
    stage = Usd.Stage.CreateNew(str(dst))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)  # willy_sim scenes are Z-up
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)        # GSO .obj vertices are metres
    UsdGeom.Xform.Define(stage, root_path)           # root Xform = the default (referenced) prim
    geom = UsdGeom.Mesh.Define(stage, f"{root_path}/mesh")  # child Mesh keeps its gprim type through the ref
    geom.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts))
    geom.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
    geom.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    # A raw scan mesh has no vertex normals and inconsistent face winding, so single-sided rendering
    # backface-culls the top face and the depth camera sees through it; the suction seal then reads no
    # flat surface and scores 0.0. Double-sided renders both faces, so the true top depth is captured.
    geom.CreateDoubleSidedAttr(True)
    # An explicit extent (local bbox) keeps bounds-based frustum culling from ever dropping the prim.
    geom.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([lo, hi], dtype=np.float32)))
    # Display colour: the part's approximate real colour when given, else a neutral grey so the object
    # reads as a solid body (see ``GsoPart.display_color``).
    _col = part.display_color if part.display_color is not None else (0.55, 0.55, 0.58)
    geom.CreateDisplayColorAttr(Vt.Vec3fArray.FromNumpy(np.array([list(_col)], dtype=np.float32)))
    stage.SetDefaultPrim(stage.GetPrimAtPath(root_path))
    stage.GetRootLayer().Save()
    _LOG.info(
        "converted %s -> %s (%d verts, %d faces, extent %s..%s m, %.1f KiB, collision=%s, %s)",
        src.name, dst, len(verts), len(faces), np.round(lo, 3).tolist(), np.round(hi, 3).tolist(),
        dst.stat().st_size / 1024.0, part.collision,
        "graspable" if part.graspable else "clutter",
    )
    return dst


def convert_curated(parts: list[GsoPart] | None = None, *, force: bool = False) -> list[Path]:
    """Convert every curated part (or a given subset), returning the USD paths."""
    out: list[Path] = []
    for p in parts or GSO_PARTS:
        usd = convert_gso_to_usd(p, force=force)
        print(f"[gso] {p.label:14s} ({'graspable' if p.graspable else 'clutter  '}) -> {usd}", flush=True)
        out.append(usd)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert the curated Aletheia GSO subset to USD (on-box).")
    ap.add_argument("--convert", action="store_true", help="boot Isaac + convert the curated .obj set to USD")
    ap.add_argument("--force", action="store_true", help="re-convert even if the USD already exists")
    args = ap.parse_args()
    if not args.convert:
        # Off-box: report the curated catalog and which entries are already converted; no Isaac needed.
        for p in GSO_PARTS:
            usd = gso_usd_path(p)
            print(f"  {p.label:14s} min_dim={p.min_dim_mm:5.0f}mm  {'graspable' if p.graspable else 'clutter  '}  "
                  f"{'converted' if usd.is_file() else 'NOT converted'}  ({p.obj_name})")
        print(f"\nsource: {gso_source_dir()}\noutput: {gso_usd_dir()}\nRun with --convert (on-box) to build the USDs.")
        return 0

    from isaacsim import SimulationApp  # type: ignore[import-not-found]

    app = SimulationApp({"headless": True})
    try:
        convert_curated(force=args.force)  # app boot put pxr on the path; convert() authors directly
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
