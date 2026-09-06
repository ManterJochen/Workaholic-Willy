"""`engine: none`: a corpus with no physics engine, no GPU, no driver and nothing to install.

Why this is possible at all. `scenes/layout.py::layout_scene_with_assets` leaves ``drop_height`` at
0.0 for `sparse`, `packed` and `bin`, which are placed flat, and only `pile` actually drops objects.
Three of four families therefore need no dynamics, and rendering needs no engine either, because the
corpus carries no RGB and a z-buffer over triangles gives depth and silhouettes exactly
(`render/raster.py`).

It refuses `pile` by name rather than faking a settle. A faked settle renders a perfectly plausible
depth image over interpenetrating objects, and nothing downstream can tell. That is the same class as
a scene whose colour frames are all black passing every automated check as `ok`.

And it is not isaac. Three differences are known, deliberate, and stamped into the corpus so no
number is ever quoted across them by accident:

1. Seating is analytic, not simulated. Each body is placed so its rotated bounding box rests exactly
   on the support. `_place_flat` spawns 2 mm high on purpose, so that a settle seats the object
   instead of resolving a penetration, and there is no settle here to do it. Seating by the body's
   own aabb is exact and handles a mesh whose origin is not its centroid, which is a known trap with
   scanned assets, but it is systematically different by a fraction of a millimetre.
2. Stability is analytic, not simulated. Isaac rejects a share of scenes as `unstable` that this
   backend cannot: it applies a support-polygon test instead, which catches the topples and does not
   catch everything a solver does. The residual difference is for the equivalence work to measure,
   not for this module to claim away.
3. The arm is optional and rendered from collision meshes. `arm.mode` decides, as it does for Isaac.
   When posed, the links come from `ur5e_collision_meshes.npz` rather than Isaac's visual meshes, so
   the silhouette differs slightly. That matters, because the arm appears in nearly every scene and
   is the dominant occluder even though it is a small share of the points. The engine and the arm's
   model both travel in `provenance.json`, so the difference is machine-readable, not invisible.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from src.utility.log_cfg import create_logger
from datagen.constants import DATAGEN_LOG_DIR
from datagen.render.camera import intrinsics_matrix, look_at_camera_to_base
from datagen.render.labels import build_object_labels, pixel_id
from datagen.render.raster import rasterise
from datagen.render.result import SceneRenderResult, ViewRender
from datagen.render.views import ViewOutcome

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from datagen.assets.manifest import AssetManifest, AssetRecord
    from datagen.config import DatagenConfig
    from datagen.scenes.spec import ObjectPlacement, SceneSpec

__all__ = ["NoEngineRenderer"]

logger = create_logger("NoEngineRenderer", "render_noengine.log", log_dir=DATAGEN_LOG_DIR)

#: Families this backend will not produce, and the reason each is refused. `pile` is the only one that
#: actually drops objects, so it is the only one whose geometry a solver decides.
REFUSED_FAMILIES: dict[str, str] = {
    "pile": ("`pile` is the one family that DROPS objects (drop_height > 0), so its geometry is "
             "whatever a solver settles it to. Faking that would render a plausible depth image over "
             "interpenetrating bodies, which nothing downstream can detect. Use engine: isaac for "
             "pile, or drop the family from the mix."),
}

#: The environment's instance id, and it is not a choice: `corpus/clouds.py` classifies background as
#: ``instances == 0``, so the table and the walls have to land in the raster with that id or their
#: pixels carry no depth at all and the corpus loses them silently.
ENVIRONMENT_ID = 0

# The table is `workspace.table_size_mm`, one declaration for every engine. A second copy of the
# size here would let two engines render one scene onto two different tables, and every "did the
# object leave the table" answer would then fork by engine.


def _box_mesh(centre_mm, half_extents_mm) -> tuple[np.ndarray, np.ndarray]:
    """An axis-aligned box as vertices + faces, without importing trimesh for six planes."""
    centre = np.asarray(centre_mm, dtype=np.float64)
    half = np.asarray(half_extents_mm, dtype=np.float64)
    signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float64)
    vertices = centre + signs * half
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                      [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]], dtype=np.int64)
    return vertices, faces


def environment_geometry(spec: "SceneSpec", config: "DatagenConfig",
                         ) -> list[tuple[np.ndarray, np.ndarray]]:
    """The table surface and every bin wall, as triangles.

    Without this the corpus is half empty: roughly half of what a net trains on is environment
    points, and a backend that renders only the bodies it was handed produces none of them. The
    absence shows up in the corpus summary, not in any single rendered view.
    """
    centre = np.asarray(config.workspace.center_mm, dtype=np.float64)
    table_mm = config.workspace.table_size_mm
    half = np.array([table_mm / 2.0, table_mm / 2.0], dtype=np.float64)
    z = float(config.workspace.table_height_mm)
    table = (np.array([[centre[0] - half[0], centre[1] - half[1], z],
                       [centre[0] + half[0], centre[1] - half[1], z],
                       [centre[0] + half[0], centre[1] + half[1], z],
                       [centre[0] - half[0], centre[1] + half[1], z]], dtype=np.float64),
             np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64))
    out = [table]
    for wall in getattr(spec, "bin_walls", ()) or ():
        out.append(_box_mesh(wall.center_mm, wall.half_extents_mm))
    return out


#: The id the arm's triangles carry inside the raster. Not `corpus/clouds.py`'s `ARM_INSTANCE = -2`:
#: that is the id the corpus files arm points under, and it is derived downstream from `arm_mask`.
#: Here the arm needs an id the z-buffer can hold so that it occludes objects and is occluded by them,
#: after which its pixels are lifted out into `arm_mask` and the instance map is cleared of them.
#: A large positive number, because 0 already means "nothing hit" and an object id could collide.
_ARM_RASTER_ID = 1_000_000

#: The bundle the default cell's arm is drawn from. `arm_geometry` loads whichever bundle
#: `collision_mesh_bundle(render.arm.robot_model)` resolves to, so a cell on another model draws
#: from another file.
ARM_MESH_SOURCE = "src/robot/safety/data/ur5e_collision_meshes.npz"


def _self_clearance_mm(model: str, joints_rad) -> float:
    """The shared one in `render/kinematics.py`.

    Both engines call it, so `self_collision_margin_mm` means one thing. Point-to-point between
    joint origins and segment-to-segment between the links give different clearances for the same
    configuration, and two definitions under one name make the margin depend on which engine
    rendered the scene.
    """
    from datagen.render.kinematics import self_clearance_mm  # noqa: PLC0415

    return self_clearance_mm(model, joints_rad)


def arm_geometry(model: str, joints_rad, base_yaw_deg: float = 0.0,
                 ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every arm link, posed in BASE, as triangles.

    These are collision meshes, not visual ones, and that is the honest difference from Isaac: the
    silhouette will not match to the pixel. It matters more than the point count suggests, because
    the arm appears in nearly every scene and is the dominant occluder, which is the reason
    multi-view exists at all. The engine and the arm's model both travel in `provenance.json`, so
    the two engines' arm pixels can be told apart instead of being quietly pooled.

    The placement formula is the safety layer's own, in `_fcl_self_collision.py`:
    ``R_base(yaw) @ T_dh[frame](q)``. Taken rather than re-derived, because a second derivation of a
    link transform is a second chance to get a frame wrong.
    """
    from src.robot.safety._ur_kinematics import ur_link_transforms_mm  # noqa: PLC0415
    from src.robot.safety.planning.environment import (  # noqa: PLC0415
        collision_mesh_bundle,
    )

    joints = np.asarray(joints_rad, dtype=np.float64)
    transforms = ur_link_transforms_mm(model, joints)
    data = np.load(collision_mesh_bundle(model))
    names = sorted({key.split("__")[0] for key in data.files})

    angle = np.radians(float(base_yaw_deg))
    base = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                     [np.sin(angle), np.cos(angle), 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for name in names:
        vertices = np.asarray(data[f"{name}__v"], dtype=np.float64)
        faces = np.asarray(data[f"{name}__f"], dtype=np.int64)
        assert transforms is not None
        transform = transforms[int(data[f"{name}__frame"][0])]
        rotation = base @ transform[:3, :3]
        translation = base @ transform[:3, 3]
        out.append((vertices @ rotation.T + translation, faces))
    return out


def _primitive_mesh(primitive: str, extents_mm: np.ndarray) -> Any:
    """One authored solid, through the same three-shape vocabulary every engine here switches on.

    The capsule branch is unreachable today and is kept rather than deleted. `PRIMITIVE_FOR_KIND`
    maps the `capsule` kind to `cylinder`, and a `CompositePart` may only be box, cylinder or sphere,
    so neither caller can ask for one. It stays because the alternative to a dead branch here is a
    mapping change that silently authors a box.
    """
    import trimesh  # noqa: PLC0415 (already a base dependency; kept out of import time)

    if primitive == "sphere":
        return trimesh.creation.icosphere(subdivisions=2, radius=float(extents_mm.max()) / 2.0)
    if primitive == "cylinder":
        return trimesh.creation.cylinder(radius=float(extents_mm[0]) / 2.0,
                                         height=float(extents_mm[2]), sections=24)
    if primitive == "capsule":
        return trimesh.creation.capsule(radius=float(extents_mm[0]) / 2.0,
                                        height=max(float(extents_mm[2]) - float(extents_mm[0]), 1e-3),
                                        count=[12, 12])
    return trimesh.creation.box(extents=extents_mm)


def composite_part_meshes(
    asset: "AssetRecord", scale: float,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Each part of a composite as its own mesh, in the object's frame. Empty when it has no parts.

    These are already convex. A `CompositePart` is a box, a cylinder or a sphere by construction, so
    a physics engine collides them exactly and no decomposition is needed or wanted: running one
    would spend seconds to approximate shapes that are already exact.
    """
    parts = getattr(asset, "parts", None) or ()
    meshes: list[tuple[np.ndarray, np.ndarray]] = []
    for part in parts:
        extents = np.asarray(part.extent_mm, dtype=np.float64) * float(scale)
        mesh = _primitive_mesh(str(part.primitive), extents)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        yaw = np.radians(float(getattr(part, "yaw_deg", 0.0)))
        if yaw:
            cos, sin = float(np.cos(yaw)), float(np.sin(yaw))
            rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
            vertices = vertices @ rotation.T
        offset = np.asarray(part.offset_mm, dtype=np.float64) * float(scale)
        meshes.append((vertices + offset, np.asarray(mesh.faces)))
    return tuple(meshes)


def _mesh_for(asset: "AssetRecord", scale: float) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices_mm, faces)`` in the body's own frame, from the manifest record.

    A scanned asset comes from its file; a procedural one is authored from `kind` and `extent_mm`
    through the same vocabulary the Isaac path uses, so the two backends agree about what a `carton`
    is rather than each inventing one.

    A composite is its parts. Without that branch `primitive_for_kind` maps every unlisted kind to
    `box`, so a mug with a handle renders here as a cuboid while `render/isaac.py` authors the handle
    and the grasp labels describe it. Nothing catches that while `composite_weight` is 0.0.
    """
    import trimesh  # noqa: PLC0415 (already a base dependency; kept out of import time)

    extents = np.asarray(asset.extent_mm, dtype=np.float64) * float(scale)
    part_meshes = composite_part_meshes(asset, scale)
    if part_meshes:
        offset = 0
        vertices_out, faces_out = [], []
        for vertices, faces in part_meshes:
            vertices_out.append(vertices)
            faces_out.append(np.asarray(faces) + offset)
            offset += len(vertices)
        return np.vstack(vertices_out), np.vstack(faces_out)
    if getattr(asset, "mesh_path", None):
        # `force="mesh"` guarantees a `Trimesh`; the stubs only promise `Geometry`.
        loaded = cast("Any", trimesh.load(str(asset.mesh_path), force="mesh", process=False))
        vertices = np.asarray(loaded.vertices, dtype=np.float64)
        # Scanned meshes are authored in metres; the corpus is millimetres everywhere.
        span = float(np.max(vertices.max(axis=0) - vertices.min(axis=0)))
        if span > 0.0 and span < 10.0:
            vertices = vertices * 1000.0
        return vertices * float(scale), np.asarray(loaded.faces)

    from datagen.assets.procedural import primitive_for_kind  # noqa: PLC0415

    mesh = _primitive_mesh(primitive_for_kind(asset.kind), extents)
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces)


def _rotation(quaternion_xyzw) -> np.ndarray:
    """XYZW to a 3x3 rotation. The repo's convention, converted here and nowhere else in this module."""
    x, y, z, w = (float(v) for v in quaternion_xyzw)
    norm = float(np.sqrt(x * x + y * y + z * z + w * w)) or 1.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def seat_on_support(vertices_mm: np.ndarray, placement: "ObjectPlacement",
                    support_z_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Place the body so its rotated lowest point rests exactly on the support.

    Exact, and that is the point. `_place_flat` spawns 2 mm high because Isaac's settle drops the
    body onto the table; with the mesh in hand the resting height is arithmetic, and it stays right
    for a mesh whose origin is not its centroid, a known trap with scanned assets, where the reported
    pose is the origin and not the centre.

    Returns ``(vertices in BASE, the settled position)``.
    """
    rotation = _rotation(placement.orientation_xyzw)
    rotated = vertices_mm @ rotation.T
    position = np.asarray(placement.position_mm, dtype=np.float64).copy()
    position[2] = float(support_z_mm) - float(rotated[:, 2].min())
    return rotated + position, position


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """The 2-D convex hull, counter-clockwise, by monotone chain.

    Written here rather than taken from trimesh, whose hull is three-dimensional while a set of
    contact points is planar by construction: trimesh returns a zero-volume solid and a
    `RuntimeWarning: invalid value encountered in divide` from its centre-of-mass integral.
    """
    unique = np.unique(np.round(np.asarray(points, dtype=np.float64), 6), axis=0)
    if len(unique) < 3:
        return unique
    order = np.lexsort((unique[:, 1], unique[:, 0]))
    ordered = unique[order]

    def _half(sequence: np.ndarray) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for point in sequence:
            while len(out) >= 2:
                first, second = out[-2], out[-1]
                cross = ((second[0] - first[0]) * (point[1] - first[1])
                         - (second[1] - first[1]) * (point[0] - first[0]))
                if cross > 1e-12:
                    break
                out.pop()
            out.append(point)
        return out

    lower = _half(ordered)
    upper = _half(ordered[::-1])
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def is_stable(vertices_mm: np.ndarray, support_z_mm: float,
              *, contact_tolerance_mm: float = 1.0,
              point_tolerance_mm: float = 2.0) -> tuple[bool, str]:
    """Would this body stand, judged by geometry alone?

    The support-polygon test: a rigid body at rest topples unless its centre of mass projects inside
    the convex hull of its contact points. The centre of mass is the vertex centroid, which assumes
    the uniform density the corpus assumes anyway; the contacts are the vertices within
    ``contact_tolerance_mm`` of the support.

    Degenerate contact sets are not failures. A sphere touches at one point and a cylinder on its
    side along a line, and both rest perfectly well in neutral equilibrium, and the corpus contains
    both (`FAMILY_KINDS["primitive"]` lists sphere and capsule). Rejecting them would quietly delete
    a whole shape family from every corpus this backend produces. So a point or line contact is
    stable exactly when the centre of mass projects onto it, which is the same physical criterion
    the polygon expresses.

    What it does not catch is anything needing dynamics: a body that rocks into a neighbour, a stack
    that shears, a wall interaction, or a sphere that simply rolls away. Isaac rejects scenes as
    `unstable` that this cannot; how much of that this recovers is for the equivalence work to
    measure and is deliberately not claimed here.
    """
    vertices = np.asarray(vertices_mm, dtype=np.float64)
    contacts = vertices[vertices[:, 2] <= float(support_z_mm) + float(contact_tolerance_mm)]
    if len(contacts) == 0:
        return False, "no vertex touches the support; the body floats"

    centre = vertices.mean(axis=0)[:2]
    hull = _convex_hull_2d(contacts[:, :2])

    if len(hull) == 1:                                  # a sphere: neutral equilibrium on one point
        offset = float(np.linalg.norm(centre - hull[0]))
        return (offset <= point_tolerance_mm,
                "" if offset <= point_tolerance_mm else
                f"single contact point and the centre of mass is {offset:.1f} mm off it")
    if len(hull) == 2:                                  # a cylinder on its side: a contact line
        start, finish = hull
        span = finish - start
        length = float(np.linalg.norm(span))
        if length < 1e-9:                               # pragma: no cover (deduplicated above)
            return False, "the contact line has no length"
        along = float(np.clip(np.dot(centre - start, span) / (length * length), 0.0, 1.0))
        offset = float(np.linalg.norm(centre - (start + along * span)))
        return (offset <= point_tolerance_mm,
                "" if offset <= point_tolerance_mm else
                f"line contact and the centre of mass is {offset:.1f} mm off it")

    edges = np.roll(hull, -1, axis=0) - hull
    to_centre = centre[None, :] - hull
    cross = edges[:, 0] * to_centre[:, 1] - edges[:, 1] * to_centre[:, 0]
    if bool(np.all(cross >= -1e-9)) or bool(np.all(cross <= 1e-9)):
        return True, ""
    radius = float(np.max(np.linalg.norm(contacts[:, :2] - centre, axis=1)))
    return False, (f"the centre of mass falls outside the {len(hull)}-sided support polygon "
                   f"(footprint radius {radius:.1f} mm); this body would topple")


class NoEngineRenderer:
    """The `engine: none` backend. Satisfies `render.engine.SceneEngine`."""

    def __init__(self, config: "DatagenConfig", *, headless: bool = True, **_: Any) -> None:
        del headless                              # there is no window to open and none to suppress
        self._config = config

    def __enter__(self) -> "NoEngineRenderer":
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None

    # ---------------------------------------------------------------- the contract

    def render(self, spec: "SceneSpec", manifest: "AssetManifest",
               rng: np.random.Generator) -> SceneRenderResult:
        """Seat, check, rasterise, label. Never raises for a rejected scene."""
        del rng                                   # nothing here is stochastic, and that is a feature
        started = time.perf_counter()
        family = str(getattr(spec.family, "value", spec.family))
        if family in REFUSED_FAMILIES:
            logger.warning("%s REFUSED: %s", spec.scene_id, REFUSED_FAMILIES[family])
            return SceneRenderResult(scene_id=spec.scene_id, status="refused_family",
                                     seconds=time.perf_counter() - started,
                                     note=REFUSED_FAMILIES[family])
        if self._config.render.arm.mode == "driven":
            note = ("arm.mode 'driven' needs a simulator to drive the arm; this backend poses it or "
                    "omits it. Set arm.mode to 'posed' or 'absent'.")
            logger.warning("%s REFUSED: %s", spec.scene_id, note)
            return SceneRenderResult(scene_id=spec.scene_id, status="refused_arm_mode",
                                     seconds=time.perf_counter() - started, note=note)

        support_z = float(self._config.workspace.table_height_mm)
        bodies: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        settled: dict[int, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {}
        records: dict[int, Any] = {}
        dropped: list[int] = []

        # Instance ids are 0-based, the base every downstream reader assumes: `grasps/labels.py`
        # enumerates spec objects and looks each up by its own 0-based index, `corpus/clouds.py`
        # reads `instances == i + 1`, and `verify` reads `dropped_objects` as instance ids.
        # Enumerating from 1 here pairs the geometry of object k with the pose of object k-1 and
        # loses object 0 entirely. The raster still needs ids above zero, because 0 means nothing
        # hit, and `pixel_id` supplies that where the map is built.
        for index, placement in enumerate(spec.objects):
            asset = manifest.get(placement.asset_id)
            vertices, faces = _mesh_for(asset, placement.scale)
            posed, position = seat_on_support(vertices, placement, support_z)
            stable, why = is_stable(posed, support_z)
            if not stable:
                logger.info("%s: %s dropped; %s", spec.scene_id, placement.asset_id, why)
                dropped.append(index)
                continue
            bodies[index] = (posed, faces)
            records[index] = asset
            quaternion = placement.orientation_xyzw
            settled[index] = ((float(position[0]), float(position[1]), float(position[2])),
                              (float(quaternion[0]), float(quaternion[1]),
                               float(quaternion[2]), float(quaternion[3])))

        if not bodies:
            return SceneRenderResult(scene_id=spec.scene_id, status="unstable",
                                     seconds=time.perf_counter() - started,
                                     dropped=tuple(dropped),
                                     note="every body failed the support-polygon test")

        environment = environment_geometry(spec, self._config)
        arm, arm_pose = self._arm(spec, manifest, settled)
        views = tuple(self._view(placement, bodies, settled, records, environment, arm)
                      for placement in spec.cameras
                      if str(getattr(placement.mount, "value", placement.mount)) != "wrist")
        return SceneRenderResult(scene_id=spec.scene_id, status="ok", views=views,
                                 settled_poses=settled, seconds=time.perf_counter() - started,
                                 arm=arm_pose, dropped=tuple(dropped))

    def _arm(self, spec: "SceneSpec", manifest: "AssetManifest", settled: dict[int, Any]):
        """``(link triangles, the ArmPose)``, or ``([], None)`` when no arm was asked for.

        The same pose isaac resolves, through the same function. Nothing here needs a simulator: the
        wrist-camera mount is not measured at run time but resolved by `datagen/robots.py` from
        constants in `willy_sim.scene.cameras`, and every other input, the IK, the joint sampler
        and the self-clearance, is the safety layer's rather than Isaac's.

        Parking the arm instead is not an option. Park faces away from the table, so `arm.mode:
        posed` would be a silent no-op: the config says posed, the corpus contains no arm, and
        nothing says so.

        What does still differ from isaac is the meshes. These are the safety layer's collision
        meshes, not Isaac's visual ones, so the silhouette will not match to the pixel. Stamped, not
        hidden.
        """
        if self._config.render.arm.mode == "absent":
            return [], None

        from datagen.render.arm import resolve_arm_pose  # noqa: PLC0415
        from datagen.render.kinematics import (  # noqa: PLC0415
            sample_joint_configuration,
            solve_joints_for_camera,
        )
        from datagen.robots import resolve_robot  # noqa: PLC0415
        from datagen.scenes.spec import CameraMount  # noqa: PLC0415

        model = str(self._config.render.arm.robot_model)
        robot = resolve_robot(model)
        mount = robot.wrist_camera_mount
        wrist = next((camera for camera in spec.cameras
                      if camera.mount is CameraMount.WRIST), None)
        if mount is None or wrist is None:
            # Refused, not parked. Isaac parks here because it still renders the wrist view it
            # planned; this backend renders no wrist view, so a parked arm would be invisible in every
            # camera and the corpus would quietly contain no arm at all while the config said `posed`.
            note = (f"arm.mode is 'posed' but this scene has "
                    f"{'no measured wrist-camera mount for ' + model if mount is None else 'no wrist camera planned'}"
                    f". Without one the arm can only be parked, and a parked arm is measured to be "
                    f"OUTSIDE both oblique views; the corpus would contain no arm while the config "
                    f"claimed one. Set arm.mode: absent, or plan a wrist camera.")
            raise ValueError(note)

        centres, radii = [], []
        for index, (position, _orientation) in settled.items():
            asset = manifest.get(spec.objects[index].asset_id)
            centres.append(position)
            radii.append(float(np.linalg.norm(np.asarray(asset.extent_mm, dtype=np.float64))) / 2.0)

        camera_to_link = mount.camera_to_link()
        by_joints = self._config.render.arm.viewpoint_source == "joint_fk"
        planned = look_at_camera_to_base(np.asarray(wrist.position_mm, dtype=np.float64),
                                         np.asarray(wrist.look_at_mm, dtype=np.float64))
        centre = solve_joints_for_camera(model, planned, camera_to_link) if by_joints else None

        pose, _camera_to_base = resolve_arm_pose(
            model, camera_to_link, lambda attempt: planned,
            np.asarray(centres, dtype=np.float64).reshape(-1, 3),
            np.asarray(radii, dtype=np.float64),
            max_attempts=self._config.render.arm.max_viewpoint_attempts,
            self_collision_margin_mm=self._config.render.arm.self_collision_margin_mm,
            self_clearance=lambda joints: _self_clearance_mm(model, joints),
            propose_joints=(lambda attempt: sample_joint_configuration(attempt, centre_rad=centre))
            if by_joints and centre is not None else None,
        )
        return arm_geometry(model, pose.joints_rad), pose

    # ---------------------------------------------------------------- one view

    def _view(self, placement: Any, bodies: dict[int, tuple[np.ndarray, np.ndarray]],
              settled: dict[int, Any], records: dict[int, Any],
              environment: list[tuple[np.ndarray, np.ndarray]],
              arm: list[tuple[np.ndarray, np.ndarray]]) -> ViewRender:
        # Per camera, not per render config: the resolution and the lens belong to the rig, and
        # `isaac.py::_camera` reads them from the same place. Reading a global here would give the
        # two backends different cameras for the same scene spec.
        resolution = (int(placement.resolution[0]), int(placement.resolution[1]))
        k = intrinsics_matrix(resolution, float(placement.horizontal_fov_deg))
        camera_to_base = look_at_camera_to_base(
            np.asarray(placement.position_mm, dtype=np.float64),
            np.asarray(placement.look_at_mm, dtype=np.float64))

        # The environment goes in under id 0 as one entry carrying every environment triangle, so it
        # occludes and is occluded like everything else. Rendering it separately and pasting it under
        # the objects would put a body behind a wall it is in front of.
        #
        # The instance map is in the pixel-id space, not the body-index space. `rasterise` writes ids
        # as they are, so keying it by the body index puts object k at value k while
        # `build_object_labels`, `verify` and `corpus/clouds.py` all read k+1: every label then gets
        # its neighbour's pixels and the last object gets none.
        #
        # The environment stays at `ENVIRONMENT_ID`, which is `BACKGROUND_ID`: it must occlude and be
        # occluded through the same z-buffer, and it must read as background, which is exactly what
        # value 0 means on disk.
        merged = {pixel_id(index): body for index, body in bodies.items()}
        if environment:
            vertices = np.vstack([piece[0] for piece in environment])
            offsets = np.cumsum([0] + [len(piece[0]) for piece in environment[:-1]])
            faces = np.vstack([piece[1] + offset for piece, offset in zip(environment, offsets)])
            merged[ENVIRONMENT_ID] = (vertices, faces)
        # The arm goes through the same z-buffer, and it has to: it is the dominant occluder, so
        # rendering it separately and compositing would put a body in front of a link that is in front
        # of it. Its pixels are lifted out afterwards.
        if arm:
            vertices = np.vstack([piece[0] for piece in arm])
            offsets = np.cumsum([0] + [len(piece[0]) for piece in arm[:-1]])
            faces = np.vstack([piece[1] + offset for piece, offset in zip(arm, offsets)])
            merged[_ARM_RASTER_ID] = (vertices, faces)
        scene = rasterise(merged, camera_to_base, k, resolution)
        arm_mask = scene.instance_map == _ARM_RASTER_ID
        # Cleared from the instance map: the arm is not an object to be grasped and must not take an
        # instance id. It keeps its depth, so the objects it hides stay hidden.
        instance_map = np.where(arm_mask, 0, scene.instance_map).astype(scene.instance_map.dtype)
        # Unoccluded masks come from rendering each body alone, the same definition the Isaac path
        # uses, so `visibility` means one thing across both backends rather than two.
        unoccluded = {index: rasterise({index: body}, camera_to_base, k, resolution).hit
                      for index, body in bodies.items()}
        labels = build_object_labels(
            instance_map, unoccluded,
            # `.asset_id`, not the record: `build_object_labels` takes a string. `records` is
            # `dict[int, Any]`, so a type checker cannot catch a whole `SceneAsset` passed here, and
            # only the check inside `build_object_labels` can.
            {index: (records[index].asset_id, settled[index][0], settled[index][1])
             for index in bodies},
        )
        return ViewRender(
            name=placement.name, outcome=ViewOutcome.RENDERED,
            rgb=None, depth_mm=scene.depth_mm, depth_noisy_mm=None,
            instance_map=instance_map,
            arm_mask=arm_mask,
            labels=labels,
            camera_position_mm=(float(placement.position_mm[0]), float(placement.position_mm[1]),
                                float(placement.position_mm[2])),
            camera_look_at_mm=(float(placement.look_at_mm[0]), float(placement.look_at_mm[1]),
                               float(placement.look_at_mm[2])),
            camera_to_base=camera_to_base, intrinsics=k,
        )
