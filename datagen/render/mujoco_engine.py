"""`engine: mujoco`: the settle, and only the settle.

It does not render, and that is the design. Rendering needs no engine at all: the corpus carries no
RGB, and a numpy z-buffer produces depth and silhouettes exactly with nothing to install
(`render/raster.py`). MuJoCo's own headless path defaults to glfw, i.e. hardware OpenGL, with
`osmesa` on Linux only, so pairing engine and renderer would drag a driver requirement back in
through the door the second backend exists to close. This backend inherits `NoEngineRenderer`'s
rendering and replaces exactly one thing: how the objects come to rest.

What it buys. `engine: none` refuses `pile` by name, because `pile` is the one family that drops
objects (`scenes/layout.py::layout_scene_with_assets`) and its geometry is whatever a solver settles
it to. This backend settles, so it produces `pile`, and it also replaces the analytic support-polygon
test with a real one: a body that rocks into a neighbour, a stack that shears or a wall interaction
are dynamics, and geometry cannot decide them.

MuJoCo works in metres and this corpus in millimetres. Every crossing is at the boundary of this
module and nowhere else. It is not a hypothetical: a silent factor of 1000 in this repo's asset path
put a part into a physics engine at 1.75e9 kg, and the run stayed green.

Mesh contact is the convex hull. That is MuJoCo's default and a real limit. It does not bite on
procedural assets: all 16 kinds author as box, cylinder or sphere (`PRIMITIVE_FOR_KIND`), so for them
the hull is the shape and the contact is exact. It bites hard on scanned and composite assets, which
are genuinely concave.

Those are decomposed rather than assumed convex. A concave mesh is split into convex parts
(`render/convex_decomposition.py`), each of which MuJoCo collides exactly, and the body carries one
geom per part. The refusal remains for the one case that still deserves it: the decomposition library
is not installed. Quietly falling back to the hull would answer a question nobody asked, for the life
of the corpus.

The render still uses the whole mesh. Only contact sees the parts. Rendering the decomposition would
put its approximation into the depth image, and the depth image is the corpus.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from src.utility.log_cfg import create_logger
from datagen.constants import DATAGEN_LOG_DIR
from datagen.render.convex_decomposition import (
    DEFAULT_DECOMPOSITION,
    DecompositionUnavailable,
    convex_parts,
)
from datagen.render.noengine import (
    NoEngineRenderer,
    _mesh_for,
    composite_part_meshes,
    environment_geometry,
)
from datagen.render.result import SceneRenderResult

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from datagen.assets.manifest import AssetManifest
    from datagen.scenes.spec import SceneSpec

__all__ = ["MujocoRenderer"]

logger = create_logger("MujocoRenderer", "render_mujoco.log", log_dir=DATAGEN_LOG_DIR)

_MM_TO_M = 1e-3
_M_TO_MM = 1e3

#: Simulated seconds of settling before the check window opens. A `pile` object is released
#: `families.placement.pile_base_clearance_mm` above whatever supports it, so even a drop from the
#: top of a pile is a fraction of a second of free fall, an order of magnitude less than this.
_SETTLE_SECONDS = 3.0

#: And the window over which rest is judged.
_CHECK_SECONDS = 0.25

#: Asset sources whose meshes may be genuinely concave, and which therefore need decomposing before
#: they are settled. Without it their contact resolves against a shape the scene does not contain:
#: a mug that cannot be filled, a bucket with no inside.
#:
#: `custom` is on this list and that is the point. A customer's own CAD part, a bracket, a housing,
#: anything with a pocket, is a weaker case for assuming convexity than a research source whose
#: shapes are known, not a stronger one. Absence from a hard-coded tuple is not evidence of
#: convexity.
_CONCAVE_SOURCES = ("gso", "ycb", "objaverse", "custom")

#: Asset kinds that are concave regardless of source. A composite carries `source="procedural"`
#: because it is authored rather than scanned, so a source-only check would wave through every mug,
#: bucket, jug, pan and hammer a run draws.
_CONCAVE_KINDS = ("composite",)


def _quat_wxyz(xyzw) -> list[float]:
    """Repo convention (XYZW) to MuJoCo's (WXYZ), at the boundary and nowhere else."""
    x, y, z, w = (float(v) for v in xyzw)
    return [w, x, y, z]


def _mesh_asset(name: str, vertices_mm: np.ndarray, faces: np.ndarray) -> str:
    vertices = np.asarray(vertices_mm, dtype=np.float64) * _MM_TO_M
    flat_v = " ".join(f"{v:.6f}" for v in vertices.reshape(-1))
    flat_f = " ".join(str(int(i)) for i in np.asarray(faces).reshape(-1))
    return f'<mesh name="{name}" vertex="{flat_v}" face="{flat_f}"/>'


def _needs_decomposition(asset: Any) -> bool:
    """Is this asset's convex hull a lie about its shape?

    Absence from the lists is not evidence of convexity, and `custom` is on them for that reason. A
    customer's own CAD part, a bracket, a housing, anything with a pocket, is a weaker case for
    assuming convexity than a research source whose shapes are known, not a stronger one.
    """
    return (str(getattr(asset, "source", "")) in _CONCAVE_SOURCES
            or str(getattr(asset, "kind", "")) in _CONCAVE_KINDS)


def _collision_parts(asset: Any, vertices_mm: np.ndarray, faces: np.ndarray,
                     scale: float) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """The convex pieces MuJoCo should collide for this asset.

    A convex asset is its own single part: procedural kinds author as box, cylinder or sphere, for
    which the hull is the shape, so decomposing them would spend seconds to reproduce the input.

    The decomposition is computed at scale 1 and then scaled. Convexity survives a uniform scale, so
    the two are equivalent, and the cache is keyed on geometry: decomposing the already-scaled mesh
    would earn a fresh cache miss for every distinct scale a run happens to draw.
    """
    # A composite is already decomposed, by construction. Its parts are authored boxes, cylinders
    # and spheres, so a solver collides them exactly; running CoACD over their union would spend
    # seconds to approximate shapes that are already exact, and would lose the pocket a handle arch
    # encloses in the process.
    authored = composite_part_meshes(asset, scale)
    if authored:
        return authored
    if not _needs_decomposition(asset):
        return ((vertices_mm, faces),)
    factor = scale if abs(scale) > 1e-9 else 1.0
    canonical = vertices_mm if factor == 1.0 else vertices_mm / factor
    parts = convex_parts(canonical, faces, settings=DEFAULT_DECOMPOSITION)
    if factor == 1.0:
        return parts
    return tuple((part_vertices * factor, part_faces) for part_vertices, part_faces in parts)


def _mass_shares(parts: tuple[tuple[np.ndarray, np.ndarray], ...]) -> tuple[float, ...]:
    """How the body's mass divides over its parts, by volume.

    Falls back to an even split when the volumes cannot be computed: a mass that is merely misplaced
    still settles, whereas a zero-mass geom does not exist to MuJoCo at all.
    """
    if len(parts) == 1:
        return (1.0,)
    import trimesh  # noqa: PLC0415 (a base dependency, kept out of import time)

    volumes = []
    for part_vertices, part_faces in parts:
        try:
            volumes.append(abs(float(trimesh.Trimesh(
                vertices=part_vertices, faces=part_faces, process=False).volume)))
        except Exception:  # noqa: BLE001 (one unmeasurable part must not cost the scene)
            volumes.append(0.0)
    total = float(sum(volumes))
    if total <= 0.0:
        return tuple(1.0 / len(parts) for _ in parts)
    return tuple(volume / total for volume in volumes)


class MujocoRenderer(NoEngineRenderer):
    """Settle with MuJoCo, render with the z-buffer. Satisfies `render.engine.SceneEngine`."""

    def render(self, spec: "SceneSpec", manifest: "AssetManifest",
               rng: np.random.Generator) -> SceneRenderResult:
        del rng                                   # the settle is deterministic; nothing is drawn here
        started = time.perf_counter()

        if self._config.render.arm.mode == "driven":
            note = ("arm.mode 'driven' needs a controller for the arm; this backend settles the "
                    "objects and poses the arm. Set arm.mode to 'posed' or 'absent'.")
            return SceneRenderResult(scene_id=spec.scene_id, status="refused_arm_mode",
                                     seconds=time.perf_counter() - started, note=note)

        try:
            settled, bodies, records, dropped, note = self._settle(spec, manifest)
        except DecompositionUnavailable as exc:
            # Still fail-closed, and under the same status the by-name refusal used. The scene holds
            # a concave mesh and this machine cannot split it, so the only two answers left are the
            # convex hull, wrong and silent for the life of the corpus, and this one.
            logger.warning("%s REFUSED: %s", spec.scene_id, exc)
            return SceneRenderResult(scene_id=spec.scene_id, status="refused_concave_mesh",
                                     seconds=time.perf_counter() - started, note=str(exc))
        except Exception as exc:  # noqa: BLE001 (a scene must not take the run down)
            logger.exception("%s: settle failed", spec.scene_id)
            return SceneRenderResult(scene_id=spec.scene_id, status="render_error",
                                     seconds=time.perf_counter() - started,
                                     note=f"{type(exc).__name__}: {exc}")
        if not bodies:
            return SceneRenderResult(scene_id=spec.scene_id, status="unstable",
                                     seconds=time.perf_counter() - started,
                                     dropped=tuple(dropped), note=note)

        environment = environment_geometry(spec, self._config)
        arm, arm_pose = self._arm(spec, manifest, settled)
        views = tuple(self._view(placement, bodies, settled, records, environment, arm)
                      for placement in spec.cameras
                      if str(getattr(placement.mount, "value", placement.mount)) != "wrist")
        return SceneRenderResult(scene_id=spec.scene_id, status="ok", views=views,
                                 settled_poses=settled, seconds=time.perf_counter() - started,
                                 arm=arm_pose, dropped=tuple(dropped), note=note)

    # ---------------------------------------------------------------- the one thing it adds

    def _settle(self, spec: "SceneSpec", manifest: "AssetManifest"):
        """Drop the scene and step it until it stops moving. Returns what the raster needs."""
        import mujoco  # noqa: PLC0415 (optional dependency, imported only on this path)

        support_z = float(self._config.workspace.table_height_mm)
        meshes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        records: dict[int, Any] = {}
        assets, worldbody = [], []

        for wall_index, wall in enumerate(getattr(spec, "bin_walls", ()) or ()):
            centre = np.asarray(wall.center_mm, dtype=np.float64) * _MM_TO_M
            half = np.asarray(wall.half_extents_mm, dtype=np.float64) * _MM_TO_M
            worldbody.append(
                f'<geom name="wall{wall_index}" type="box" pos="{centre[0]:.6f} {centre[1]:.6f} '
                f'{centre[2]:.6f}" size="{half[0]:.6f} {half[1]:.6f} {half[2]:.6f}"/>')

        # Instance ids are 0-based, the base every downstream reader assumes: `grasps/labels.py`
        # enumerates spec objects and looks each up by its own 0-based index, `corpus/clouds.py`
        # reads `instances == i + 1`, and `verify` reads `dropped_objects` as instance ids.
        # Enumerating from 1 here pairs the geometry of object k with the pose of object k-1 and
        # loses object 0 entirely. The raster still needs ids above zero, because 0 means nothing
        # hit, and `pixel_id` supplies that where the map is built.
        for index, placement in enumerate(spec.objects):
            asset = manifest.get(placement.asset_id)
            vertices, faces = _mesh_for(asset, placement.scale)
            # The render keeps the whole mesh. Only the collision geometry below is decomposed;
            # putting the decomposition's approximation into the depth image would put it into the
            # corpus, and the depth image is the corpus.
            meshes[index] = (vertices, faces)
            records[index] = asset
            parts = _collision_parts(asset, vertices, faces, float(placement.scale))
            # Mass is split across the parts by volume rather than evenly. MuJoCo sums each geom's
            # own inertia, so an even split would place the centre of mass by part count: a handle
            # would weigh as much as the body it hangs off, and the object would settle on its side.
            for part_index, (part_vertices, part_faces) in enumerate(parts):
                assets.append(_mesh_asset(f"m{index}_{part_index}", part_vertices, part_faces))
            shares = _mass_shares(parts)
            total_mass = max(float(placement.mass_kg), 1e-3)
            geoms = "".join(
                f'<geom type="mesh" mesh="m{index}_{part_index}" '
                f'mass="{max(total_mass * share, 1e-6):.8f}"/>'
                for part_index, share in enumerate(shares))
            position = np.asarray(placement.position_mm, dtype=np.float64) * _MM_TO_M
            quaternion = " ".join(f"{v:.8f}" for v in _quat_wxyz(placement.orientation_xyzw))
            worldbody.append(
                f'<body name="b{index}" pos="{position[0]:.6f} {position[1]:.6f} '
                f'{position[2]:.6f}" quat="{quaternion}">'
                f'<freejoint/>{geoms}</body>')

        xml = (f'<mujoco><option timestep="0.002" integrator="implicitfast"/>'
               f'<asset>{"".join(assets)}</asset><worldbody>'
               f'<geom name="table" type="plane" pos="0 0 {support_z * _MM_TO_M:.6f}" '
               f'size="2 2 0.1"/>{"".join(worldbody)}</worldbody></mujoco>')

        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        step = float(model.opt.timestep)
        for _ in range(int(_SETTLE_SECONDS / step)):
            mujoco.mj_step(model, data)

        # Rest is a displacement, not a velocity. A velocity threshold rejects bodies that have come
        # to rest and never left the table. Isaac judges the same question by how far a body moves
        # over a check window (`render.stability_tolerance_mm`, 0.5 mm), so using that quantity here
        # makes the two engines' `unstable` rates comparable rather than merely both numbers.
        before = np.array([data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"b{i}")]
                           for i in sorted(meshes)], dtype=np.float64) * _M_TO_MM
        for _ in range(int(_CHECK_SECONDS / step)):
            mujoco.mj_step(model, data)
        after = np.array([data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"b{i}")]
                          for i in sorted(meshes)], dtype=np.float64) * _M_TO_MM
        travelled = np.linalg.norm(after - before, axis=1)
        tolerance = float(self._config.render.stability_tolerance_mm)

        settled: dict[int, tuple[tuple[float, float, float], tuple[float, ...]]] = {}
        bodies: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        kept_records: dict[int, Any] = {}
        dropped: list[int] = []
        moving: list[str] = []

        for order, index in enumerate(sorted(meshes)):
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"b{index}")
            position = np.asarray(data.xpos[body], dtype=np.float64) * _M_TO_MM
            # Escaped, in the same sense Isaac means it: a body that left the table is not a body
            # whose pose can be labelled, however still it now is.
            if position[2] < support_z - 50.0:
                dropped.append(index)
                continue
            if float(travelled[order]) > tolerance:
                moving.append(f"b{index} moved {float(travelled[order]):.2f} mm")
                continue
            rotation = np.asarray(data.xmat[body], dtype=np.float64).reshape(3, 3)
            vertices, faces = meshes[index]
            bodies[index] = (vertices @ rotation.T + position, faces)
            orientation = _xyzw_from_matrix(rotation)
            settled[index] = ((float(position[0]), float(position[1]), float(position[2])),
                              (float(orientation[0]), float(orientation[1]),
                               float(orientation[2]), float(orientation[3])))
            kept_records[index] = records[index]

        note = ("" if not moving else
                f"{len(moving)} body/bodies still moving after {_SETTLE_SECONDS:.1f} s "
                f"(over {tolerance:.2f} mm in {_CHECK_SECONDS:.2f} s): {'; '.join(moving[:3])}")
        return settled, bodies, kept_records, dropped, note


def _xyzw_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """A 3x3 rotation to the repo's XYZW quaternion. Shepperd's method, for numerical stability."""
    m = np.asarray(rotation, dtype=np.float64)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)
