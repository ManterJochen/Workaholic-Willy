"""The on-box half: author, settle, verify, path-trace, read the annotators.

This is the render path's Isaac session; `datagen.grasps.physics_isaac` boots one of its own for the
physics screen. Every Isaac import is inside a function, because ``SimulationApp`` must be
constructed before any ``isaacsim`` module is imported, which makes a module-scope ``isaacsim``
import here an error rather than a style choice.

The order is the physics. Objects are authored at their spawn poses, physics settles them, and only
then is anything rendered or labelled: the layout describes where things start, the image shows
where they ended up, and every 3-D label comes from the settled state. A scene still moving after
the second settle is rejected with a reason rather than recorded, because its boxes would describe
a pose the scene is no longer in.

Two renderers, on purpose. The beauty pass is path-traced; the per-object solo passes that measure
occlusion run on the raster path. Silhouettes do not depend on light transport, so this is the same
geometry at about two orders of magnitude less cost, and without it the visibility label would
multiply the price of the dataset by the object count. The mode is switched twice per scene, not
twice per object.

The arm stands before it is posed. ``absent`` and ``posed`` both render; ``driven`` is named and
refuses. The arm is referenced before the settle, because PhysX builds its articulation views when
the world resets and one added afterwards has nothing to pose. It therefore holds park through the
drop, where it is clear of the drop column and so catches nothing and scatters nothing, and it is
teleported to its scene configuration only once everything has come to rest. Which configuration
that is, and why a viewpoint gets refused, live in :mod:`datagen.render.arm`.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.utility.log_cfg import create_logger

from datagen.assets.manifest import AssetManifest
from datagen.assets.procedural import primitive_for_kind
from datagen.config import DatagenConfig
from datagen.constants import DATAGEN_LOG_DIR, RENDER_ISAAC_LOG_FILE
from datagen.render.arm import ArmPose, resolve_arm_pose
from datagen.render.camera import intrinsics_matrix, look_at_camera_to_base, project_to_pixels
from datagen.render.depth_noise import SENSOR_MODELS, inject_depth_noise
from datagen.render.result import SceneRenderResult, ViewRender
from datagen.render.kinematics import (
    PARK_JOINTS_RAD,
    sample_joint_configuration,
    solve_joints_for_camera,
)
from datagen.render.labels import ObjectLabel, build_object_labels, instance_map_from_masks
from datagen.render.materials import MaterialParams, sample_material
from datagen.render.quality import describe_rgb
from datagen.render.views import ViewOutcome
from datagen.robots import declared_mount, resolve_robot
from datagen.scenes.spec import CameraMount, CameraPlacement, SceneFamily, SceneSpec

__all__ = ["ColourBufferNeverArrived", "IsaacRenderer", "SceneRenderResult", "ViewRender"]


logger = create_logger("datagen.render.isaac", RENDER_ISAAC_LOG_FILE, log_dir=DATAGEN_LOG_DIR)


class ColourBufferNeverArrived(RuntimeError):
    """The geometry was ready and the colour image never was.

    Its own type because the recovery is specific: the depth and segmentation annotators on this
    camera are producing correct frames, so the scene, the lighting and the physics are all fine and
    re-rendering them would waste the work. What is wedged is the path tracer's own accumulation,
    which the recovery restarts by toggling the render mode.
    """

_MM_TO_M = 0.001
#: Not a warmup count: the fix for a stale annotator is not a bigger number. The instance-id
#: annotator hands back a dict whose ``data`` (the id image) can be fresh while its
#: ``info.idToLabels`` (the id to prim map) is stale, and applying an old map to a new image is
#: silently catastrophic: an object is handed another object's mask and the solo masks come back
#: shifted by one object. Raising the warmup does not help, because ticks are not what is wrong.
#:
#: So nothing here counts ticks and hopes. Every read pumps a few steps and then checks the content
#: against what the scene is supposed to contain, retrying until it agrees. The solo pass has a
#: particularly strong check available: with one object visible, no other object may appear, and a
#: stale frame fails that by construction.
#: A visibility change is fully present in the depth after 2 steps and identical through 12, so 4
#: steps is that number with a margin; the loop below still checks the content it got.
_READ_STEPS = 4
_READ_ATTEMPTS = 12
#: How much two "identical" reads may disagree, as a fraction of the image. Two reads of a
#: motionless scene differ by float noise of about 0.01 mm plus a silhouette pixel or two, while a
#: frame belonging to a different scene state differs by more than 2 mm across thousands of pixels,
#: so the budget only has to sit between the two. 0.1 % is about 307 px at 640x480. Edge pixels are
#: where the float noise becomes a 2 mm disagreement, so a pile with many silhouette edges needs the
#: looser budget and a tighter one rejects correctly rendered scenes. The 8 px floor keeps a small
#: image from getting a budget of nothing.
_READ_AGREE_FRACTION = 0.001
_READ_AGREE_MIN_PX = 8
#: Two depth samples within this are the same surface. Objects are >= 25 mm and sit on the table, so
#: a few millimetres separates "this object" from "the table behind it" without splitting a curved
#: face.
_DEPTH_MATCH_TOLERANCE_MM = 2.0
#: Solver quality for the dropped objects. These are not scene parameters but how carefully the
#: physics is solved, so they are constants here rather than knobs in the config, and each is
#: explained at its use site in ``_author_damping``. PhysX's defaults (4/1 iterations, effectively
#: unbounded depenetration speed) are tuned for games, where a stack that drifts is invisible; here
#: it walks an object off the table and costs the scene.
_SOLVER_POSITION_ITERATIONS = 32
_SOLVER_VELOCITY_ITERATIONS = 4
#: m/s. The speed cap on separating two overlapping bodies: an ejection becomes a nudge.
_MAX_DEPENETRATION_VELOCITY_MS = 0.5
#: m^2/s^2. Below this energy a body is a candidate for stabilization, so a near-resting heap rests.
_STABILIZATION_THRESHOLD = 0.001
#: Subdivision for analytic surfaces. 2 is the usual "smooth enough" level; at the renderer default
#: a sphere renders visibly faceted.
_REFINEMENT_LEVEL = 2
#: How many extra settle rounds a scene may have while it is still converging. An unsettled scene
#: settles longer and is then discarded and logged; this is the "longer". It is a budget rather than
#: a fixed number of rounds because the loop stops the moment progress stalls, which is also what
#: separates a heap that is nearly at rest from an object in free fall, whose reported distance is
#: the same clamped value every round.
_MAX_EXTRA_SETTLE_ROUNDS = 3
#: Below this many objects a scene has stopped being the scene the layout described, so a run of
#: escapes is refused instead of shipped as a thinner one. Two is the smallest count at which
#: occlusion between objects is still possible at all.
_MIN_OBJECTS_AFTER_DROP = 2
# The table's edge length is `workspace.table_size_mm`, one declaration for every engine.
# `_author_scene` builds the table and `_drop_escaped` decides what has fallen off it, and
# `render/noengine.py` and `verify_robot.py` read the same key. A second copy anywhere, or one in
# metres, lets two engines build different tables and the reachability check plan through a third,
# and nothing says so out loud.
#: Below the table top by more than this and the object is under the table, not on it.
_BELOW_TABLE_MM = -50.0
#: Prim paths. One place, because a typo here is a mask that silently matches nothing.
_WORLD = "/World"
_TABLE = f"{_WORLD}/Table"
_OBJECT_ROOT = f"{_WORLD}/Object"
_WALL_ROOT = f"{_WORLD}/BinWall"
_CAMERA_ROOT = f"{_WORLD}/Cameras"
ROBOT_ROOT = f"{_WORLD}/Robot"
#: Seed stream for resampling a wrist viewpoint. Fixed, so two runs of one seed agree about which
#: scenes have a wrist view.
_WRIST_RESAMPLE_STREAM = 20260812
#: The six arm DoFs, shared by every UR e-series articulation (only the link lengths differ).
_ARM_JOINT_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)


def _placement_at(placement: CameraPlacement, camera_to_base: np.ndarray) -> CameraPlacement:
    """The same camera, moved to a resolved pose, expressed the way the rest of the pipeline expects.

    Position and look-at rather than the matrix, because every other camera in this generator is
    described that way and the recorded extrinsic is rebuilt from them. The view direction survives
    that rebuild exactly; the roll about it does not, because the roll is re-derived from
    ``look_at_camera_to_base``'s up-hint and a position and a target do not carry it. Under
    ``target_ik`` the pose is the IK solution for a sampled look-at, so the rebuild reproduces that
    look-at to the IK tolerance. Under ``joint_fk`` the pose is a forward-kinematics frame whose roll
    is whatever the joints give it, and the rebuilt extrinsic differs from it by that roll.
    """
    from dataclasses import replace  # noqa: PLC0415 (only this function needs it)

    position = camera_to_base[:3, 3]
    forward = camera_to_base[:3, 2]
    distance = float(np.linalg.norm(np.asarray(placement.position_mm) - np.asarray(placement.look_at_mm)))
    target = position + forward * distance
    return replace(
        placement,
        position_mm=(float(position[0]), float(position[1]), float(position[2])),
        look_at_mm=(float(target[0]), float(target[1]), float(target[2])),
    )


def _quat_xyzw_from_wxyz(q: "Sequence[float] | np.ndarray") -> tuple[float, float, float, float]:
    return (float(q[1]), float(q[2]), float(q[3]), float(q[0]))


#: Voxels along the longest axis of an SDF collider. 256 is a choice, not a measurement: nothing
#: here has swept it, and the number that would justify it is the resolution at which a mug handle
#: stops being separable from the body, which is the geometry the mesh branch exists to preserve.
#: That sweep is the one to run if `sdf` ever becomes the default. The value is stated rather than
#: left at the renderer default, because an SDF collider's fidelity is its resolution and a
#: reference that leaves it implicit cannot say what shape it collided.
_SDF_RESOLUTION = 256


def _apply_mesh_collision(prim: Any, approximation: str, *, UsdPhysics: Any) -> None:  # noqa: N803
    """Give a mesh prim a collider PhysX will actually use for a dynamic body.

    Setting ``physics:approximation = "sdf"`` through `UsdPhysics.MeshCollisionAPI` alone is not
    enough: PhysX reads it back as ``None/MeshSimplification`` and prints, per object,

        triangle mesh collision (approximation None/MeshSimplification) cannot be a part of a
        dynamic body, falling back to convexHull approximation

    A convex hull fills every concavity, so a mug meant to be grasped by its handle is a lump to
    physics while the labels describe the true surface: the two disagree silently, which is the one
    failure mode this generator cannot tolerate. ``sdf`` is a PhysX extension, so the SDF schema has
    to be applied alongside the token for the value to mean anything.

    The other approximations are core UsdPhysics and need no extension.
    """
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
    if approximation != "sdf":
        return
    from pxr import PhysxSchema  # type: ignore[import-not-found]  # noqa: PLC0415

    sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
    # Stated rather than left at the renderer default; the reason is at `_SDF_RESOLUTION`.
    sdf_api.CreateSdfResolutionAttr(_SDF_RESOLUTION)


class IsaacRenderer:
    """Owns one Isaac session across a whole dataset; booting it per scene would dominate the cost."""

    def __init__(self, config: DatagenConfig, *, headless: bool = True) -> None:
        if config.render.arm.mode == "driven":
            raise NotImplementedError(
                "render.arm.mode='driven' is named but not implemented. It means booting the full "
                "cell and actually DRIVING the arm to the viewpoint through planning and the safety "
                "pipeline; maximum fidelity, and a much larger step whose benefit for still images "
                "would have to be shown first. Use 'posed': the arm is placed by real inverse "
                "kinematics at the configuration that puts the wrist camera at the sampled viewpoint, "
                "and it is checked before it is used. It does not move; it stands where it would stand."
            )
        self._config = config
        #: Resolved once per dataset, not per scene: the refusal for an incompletely declared robot
        #: must happen before Isaac boots, and it is the same answer for every scene anyway.
        self._robot = (
            resolve_robot(config.render.arm.robot_model,
                          declared_mount(config.render.arm.wrist_camera_mount))
            if config.render.arm.mode != "absent" else None
        )
        self._articulation: Any = None
        self._arm_joints: Any = None
        from isaacsim import SimulationApp  # noqa: PLC0415 (must precede every other isaacsim import)

        self._app = SimulationApp({"headless": headless})
        from isaacsim.core.api import World  # type: ignore[import-not-found]  # noqa: PLC0415

        self._world = World(stage_units_in_meters=1.0)
        self._cameras: dict[str, Any] = {}
        #: Prims the scene registry created and will delete itself.
        self._authored: list[str] = []
        #: Prims nothing else owns (lights); removed by hand, after the registry has finished.
        self._extra_prims: list[str] = []
        self._reset_stage()
        # The session is the expensive thing in this package: one boot per dataset, not per scene,
        # so when it started and with which arm is the first line of any render post-mortem.
        logger.info("Isaac session up (headless=%s, render=%s, arm=%s, robot=%s)",
                    headless, config.render.mode, config.render.arm.mode,
                    self._robot.key if self._robot is not None else "none")

    # --- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        logger.info("closing the Isaac session")
        try:
            self._app.close()
        except Exception as exc:  # noqa: BLE001 (a headless SimulationApp can segfault on teardown)
            # Swallowed, because the run is over, but recorded: a teardown that throws every time
            # is worth knowing about before it is blamed for something upstream.
            logger.warning("Isaac teardown raised and was ignored (%s: %s)", type(exc).__name__, exc)

    def __enter__(self) -> "IsaacRenderer":
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> None:
        """Report a pending exception before closing, because closing ends the process.

        Isaac's ``SimulationApp.close()`` ends in ``shutdown_and_release_framework()``, whose own
        source says "the python process is about to be terminated". An exception still travelling up
        the stack when ``__exit__`` runs is therefore never printed: the run looks like a silent
        native exit with no traceback and no error line. Printing here costs nothing.
        """
        if exc is not None:
            import traceback  # noqa: PLC0415 (only needed on the failing path)

            print("[datagen] renderer failed; printing before Isaac's teardown ends the process:",
                  flush=True)
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            sys.stderr.flush()
        self.close()

    def _step(self, count: int = 1, *, render: bool = True) -> None:
        """Advance the simulation. ``render=False`` means physics only, and means it.

        Calling ``app.update()`` after every physics step makes the settle phase render, at frame
        cost per step instead of physics-tick cost. The settle produces no image anybody reads; only
        the passes that follow do.
        """
        for _ in range(max(0, count)):
            self._world.step(render=render)
            if render:
                self._app.update()

    def _reset_stage(self) -> None:
        """Tear the previous scene down in the order PhysX requires, then remove what it did not own.

        Removing a rigid body's prim while the simulation is running invalidates PhysX's tensor view
        ("prim '/World/Object_0' was deleted while being used by a shape in a tensor view class"),
        and every later scene then fails on an invalidated view. So: stop the simulation, let the
        scene registry delete the prims it created, and only then remove the hand-authored ones in
        ``_extra_prims`` (the lights, the objects and the arm). The cameras go with the stage too,
        and `_camera` rebuilds them for the next scene.
        """
        import omni.usd  # type: ignore[import-not-found]  # noqa: PLC0415

        try:
            self._world.stop()
        except Exception:  # noqa: BLE001 (nothing to stop before the first scene)
            pass
        self._world.clear()          # Isaac's own teardown: registry, physics views and prims together
        stage = omni.usd.get_context().get_stage()
        for path in self._extra_prims:
            if stage.GetPrimAtPath(path).IsValid():
                stage.RemovePrim(path)
        self._extra_prims.clear()
        self._authored.clear()
        # The articulation handles index prims that have just been removed; keeping them would mean
        # posing whatever the next scene happens to author at the same path.
        self._articulation = None
        self._arm_joints = None
        # Cameras are dropped with the stage: an annotator that outlives the prims it indexed holds
        # dangling ids, and re-registering it is cheaper than debugging what a stale one reports.
        for path in list(self._cameras):
            if stage.GetPrimAtPath(f"{_CAMERA_ROOT}/{path}").IsValid():
                stage.RemovePrim(f"{_CAMERA_ROOT}/{path}")
        self._cameras.clear()

    # --- authoring ---------------------------------------------------------
    def _set_render_mode(self, mode: str) -> None:
        """Path tracing for the beauty pass, raster for the solo masks. Twice per scene, not per object."""
        import carb  # type: ignore[import-not-found]  # noqa: PLC0415

        settings = carb.settings.get_settings()
        if mode == "pathtrace":
            settings.set("/rtx/rendermode", "PathTracing")
            settings.set("/rtx/pathtracing/spp", int(self._config.render.samples_per_pixel))
            settings.set("/rtx/pathtracing/totalSpp", int(self._config.render.samples_per_pixel))
        else:
            settings.set("/rtx/rendermode", "RaytracedLighting")

    def _author_material(self, prim_path: str, material: MaterialParams) -> Any:
        """This generator's PBR parameters on the renderer's principled shader, not a library one."""
        try:
            from isaacsim.core.api.materials import OmniPBR  # type: ignore[import-not-found]  # noqa: PLC0415

            return OmniPBR(
                prim_path=f"{prim_path}/Material",
                color=np.asarray(material.base_color_rgb, dtype=np.float64),
                roughness=material.roughness,
                metallic=material.metallic,
            )
        except Exception:  # noqa: BLE001 (older/newer Isaac: fall back to the flat colour)
            return None

    @staticmethod
    def _bind_material(prim: Any, material_path: str, *, purpose: str = "") -> None:
        """Bind a material to a prim. Creating one is not using one.

        Without this call ``_author_material`` builds an OmniPBR per object and binds it to nothing,
        so every randomised roughness and metallic value is decorative and the image is rendered
        from the flat ``displayColor`` instead.
        """
        from pxr import UsdShade  # type: ignore[import-not-found]  # noqa: PLC0415

        import omni.usd  # type: ignore[import-not-found]  # noqa: PLC0415

        stage = omni.usd.get_context().get_stage()
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        if not material:
            return
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        if purpose:
            binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, purpose)
        else:
            binding.Bind(material)

    def _author_object(  # noqa: PLR0913 (each argument is a different source of truth)
        self, index: int, placement: Any, asset: Any, material: MaterialParams,
        physics_material: Any = None,
    ) -> str:
        """Author one object as two prims: a rigid body, and a scaled collider beneath it.

        With body and geometry on one prim, that prim's transform is ``T*R*S`` and PhysX has to
        recover a rigid pose from a matrix carrying a non-uniform scale. The leftover shows up as
        ``ScaleOrientation is not supported for rigid bodies`` and, worse, as a collider that is not
        the box being rendered and labelled, which is the one disagreement this generator cannot
        have.

        Split in two, the body carries ``T*R`` (rigid, exactly representable) and the collider
        carries a pure diagonal scale in the body's own frame, which is the case PhysX supports.
        Cylinders and spheres need no scale at all, because their dimensions are attributes, so only
        boxes ever carry one.
        """
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # type: ignore[import-not-found]  # noqa: PLC0415

        import omni.usd  # type: ignore[import-not-found]  # noqa: PLC0415

        stage = omni.usd.get_context().get_stage()
        path = f"{_OBJECT_ROOT}_{index}"
        position = np.asarray(placement.position_mm, dtype=np.float64) * _MM_TO_M
        quaternion = placement.orientation_xyzw

        body = UsdGeom.Xform.Define(stage, Sdf.Path(path))
        xform = UsdGeom.Xformable(body.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in position)))
        xform.AddOrientOp().Set(Gf.Quatf(
            float(quaternion[3]),
            Gf.Vec3f(float(quaternion[0]), float(quaternion[1]), float(quaternion[2])),
        ))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(float(placement.mass_kg))

        parts = getattr(asset, "parts", None)
        if parts:
            # A composite: several colliders under the one rigid body above, which is the shape
            # PhysX wants. Each part carries its own local offset and yaw, so the body still carries
            # only T*R.
            for part_index, part in enumerate(parts):
                self._author_part(
                    stage,
                    f"{path}/Part{part_index}",
                    primitive=part.primitive,
                    extent_m=np.asarray(part.extent_mm, dtype=np.float64) * _MM_TO_M,
                    material=material,
                    physics_material=physics_material,
                    offset_m=np.asarray(part.offset_mm, dtype=np.float64) * _MM_TO_M,
                    yaw_deg=float(part.yaw_deg),
                )
        elif getattr(asset, "mesh_path", None):
            # A scanned object, and it needs this branch: `primitive_for_kind("mesh")` returns
            # "box", so without one the object is drawn as a cuboid and the labeller, deriving its
            # shape the same way, agrees. Two components consistent with each other and both wrong
            # about the object.
            #
            # The mesh authored is the closed one from `datagen.assets.mesh_geometry`, the very same
            # object the labeller intersects. Not the raw scan: many scans have open boundaries, and
            # an open surface has no inside, so physics and the labels would each have to invent
            # one. The fabricated surface is the underside the scanner never saw.
            self._author_mesh(
                stage,
                f"{path}/Geom",
                mesh_path=str(asset.mesh_path),
                material=material,
                physics_material=physics_material,
            )
        else:
            # The kind to solid mapping lives in assets/procedural.py because datagen.grasps
            # computes exact contacts from the same mapping; a local switch here would let the two
            # drift apart.
            self._author_part(
                stage,
                f"{path}/Geom",
                primitive=primitive_for_kind(asset.kind),
                extent_m=np.asarray(asset.extent_mm, dtype=np.float64) * _MM_TO_M,
                material=material,
                physics_material=physics_material,
            )
        self._author_damping(path)
        # Hand-authored, so the scene registry does not own it: this list is what removes it later.
        self._extra_prims.append(path)
        return path

    def _author_part(
        self,
        stage: Any,
        geom_path: str,
        *,
        primitive: str,
        extent_m: Any,
        material: Any,
        physics_material: Any,
        offset_m: Any = None,
        yaw_deg: float = 0.0,
    ) -> Any:
        """One collider and its visual, at an optional local pose inside the rigid body.

        `_author_object` calls it once for a single-primitive object and once per part for a
        composite. The local pose is what lets several parts share one body without the body itself
        carrying anything but T*R.
        """
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # type: ignore[import-not-found]  # noqa: PLC0415

        extent = np.asarray(extent_m, dtype=np.float64)
        if primitive == "sphere":
            radius = float(extent[0] / 2.0)
            geom = UsdGeom.Sphere.Define(stage, Sdf.Path(geom_path))
            geom.CreateRadiusAttr(radius)
            geom.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -radius),
                                   Gf.Vec3f(radius, radius, radius)])
            self._set_refinement(geom.GetPrim())
        elif primitive == "cylinder":
            radius, height = float(extent[0] / 2.0), float(extent[2])
            geom = UsdGeom.Cylinder.Define(stage, Sdf.Path(geom_path))
            geom.CreateRadiusAttr(radius)
            geom.CreateHeightAttr(height)
            geom.CreateAxisAttr(UsdGeom.Tokens.z)
            geom.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2.0),
                                   Gf.Vec3f(radius, radius, height / 2.0)])
            self._set_refinement(geom.GetPrim())
        else:
            geom = UsdGeom.Cube.Define(stage, Sdf.Path(geom_path))
            geom.CreateSizeAttr(1.0)
            geom.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])

        # Order matters: translate, then rotate, then scale, so the scale stays a pure diagonal in
        # the part's own frame, which is the case PhysX supports. Authored on the geom itself for a
        # single primitive and for a part alike.
        xformable = UsdGeom.Xformable(geom.GetPrim())
        if offset_m is not None:
            offset = np.asarray(offset_m, dtype=np.float64)
            if float(np.abs(offset).max()) > 0.0:
                xformable.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in offset)))
        if abs(yaw_deg) > 1e-9:
            xformable.AddRotateZOp().Set(float(yaw_deg))
        if primitive not in ("sphere", "cylinder"):
            # The only scale in the object, and it lives below the rigid body, not on it.
            xformable.AddScaleOp().Set(Gf.Vec3f(*(float(v) for v in extent)))

        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        geom.CreateDisplayColorAttr([Gf.Vec3f(*(float(c) for c in material.base_color_rgb))])

        visual = self._author_material(geom_path, material)
        if visual is not None:
            self._bind_material(geom.GetPrim(), f"{geom_path}/Material")
        if physics_material is not None:
            self._bind_material(geom.GetPrim(), physics_material.prim_path, purpose="physics")
        return geom

    def _author_mesh(
        self,
        stage: Any,
        geom_path: str,
        *,
        mesh_path: str,
        material: Any,
        physics_material: Any,
    ) -> Any:
        """One scanned object as a ``UsdGeom.Mesh`` under the rigid body, visual and collider alike.

        One mesh for both. Authoring the raw scan for looks and something else for physics is the
        standard sim arrangement and it is exactly the drift this generator cannot afford: the
        labels would then describe a third shape. Here the pixels, the contacts and the labels are
        the same triangles.

        The vertices arrive already centred on the object's bounds, so this prim carries no
        transform at all and the rigid body above keeps carrying only T*R, the invariant
        `_author_object` exists to protect. `origin_offset_mm` on the `MeshShape` records where the
        file's own origin went, because anything reading the mesh back has to apply the same shift
        or the object stands elsewhere.

        The collision approximation is a config decision, not a default buried here: PhysX cannot
        collide a dynamic body against raw triangles, and the choice between an SDF and a convex
        decomposition is the choice of whether physics agrees with the labels. See
        `AssetSourcesConfig.mesh_collision`.
        """
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt  # type: ignore[import-not-found]  # noqa: PLC0415

        from datagen.assets.mesh_geometry import load_mesh_shape  # noqa: PLC0415

        shape = load_mesh_shape(mesh_path)
        if shape is None:
            # `load_mesh_shape` logs which refusal this is. Raising rather than drawing a stand-in:
            # the bank is supposed to have filtered these out already, so reaching here means the
            # filter and the renderer disagree, and that is worth stopping for.
            raise ValueError(
                f"mesh {mesh_path!r} cannot be closed into a solid, so it cannot be authored. "
                f"The asset bank should have excluded it; see MeshAssetBank.load."
            )

        points, counts, indices, lower, upper = shape.usd_arrays()
        mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(geom_path))
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*(float(v) for v in p)) for p in points]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
        mesh.CreateExtentAttr([Gf.Vec3f(*(float(v) for v in lower)),
                               Gf.Vec3f(*(float(v) for v in upper))])
        # Flat shading: the labeller reads a triangle's own normal at a contact, so smoothing the
        # rendered surface would make the picture disagree with the geometry the labels came from.
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        _apply_mesh_collision(mesh.GetPrim(), self._config.assets.mesh_collision,
                              UsdPhysics=UsdPhysics)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*(float(c) for c in material.base_color_rgb))])

        visual = self._author_material(geom_path, material)
        if visual is not None:
            self._bind_material(mesh.GetPrim(), f"{geom_path}/Material")
        if physics_material is not None:
            self._bind_material(mesh.GetPrim(), physics_material.prim_path, purpose="physics")
        return mesh

    @staticmethod
    def _set_refinement(prim: Any) -> None:
        """Tessellate curved surfaces finely enough that a sphere is not visibly a polygon.

        At the default refinement a procedural sphere renders faceted, and a photorealistic dataset
        cannot ship visible polygons on a ball.

        Curved surfaces only. Applied to a cube this rounds its rendered edges while its collider
        stays a sharp box, and a rendered shape that differs from the labelled and simulated one is
        the same defect as the ScaleOrientation collider mismatch, from the other direction.
        """
        from pxr import Sdf  # type: ignore[import-not-found]  # noqa: PLC0415

        prim.CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int).Set(_REFINEMENT_LEVEL)
        prim.CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool).Set(True)

    def _physics_material(self) -> Any:
        """One shared contact material for the table, the walls and every object.

        Built once per scene and reused, because a material per prim is a material per prim to keep
        in sync. Without it the scene runs on PhysX's default friction and restitution rather than
        the configured ones.
        """
        from isaacsim.core.api.materials import PhysicsMaterial  # type: ignore[import-not-found]  # noqa: PLC0415

        render = self._config.render
        return PhysicsMaterial(
            prim_path=f"{_WORLD}/ContactMaterial",
            static_friction=float(render.friction_static),
            dynamic_friction=float(render.friction_dynamic),
            restitution=float(render.restitution),
        )

    def _author_damping(self, prim_path: str) -> None:
        """Everything that decides whether a heap settles or slowly walks off the table.

        Each setting answers one symptom of a pile that will not come to rest:

        * damping: PhysX gives a rigid body no rolling resistance, so a cylinder that lands on a
          flat table rolls until the table runs out.
        * max depenetration velocity: the direct cause of a launch. When bodies overlap, the solver
          separates them, and by default it may do so arbitrarily fast. Capping it turns an
          ejection into a nudge.
        * solver iterations: an under-solved stack never fully resolves its contacts, so it is
          pushed apart a little on every step and creeps across the table. No amount of friction
          fixes it, because the push is the solver's.
        * stabilization: lets a nearly-resting heap come to actual rest instead of shivering.
        """
        from pxr import PhysxSchema  # type: ignore[import-not-found]  # noqa: PLC0415

        import omni.usd  # type: ignore[import-not-found]  # noqa: PLC0415

        prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        api.CreateLinearDampingAttr(float(self._config.render.linear_damping))
        api.CreateAngularDampingAttr(float(self._config.render.angular_damping))
        api.CreateMaxDepenetrationVelocityAttr(_MAX_DEPENETRATION_VELOCITY_MS)
        api.CreateSolverPositionIterationCountAttr(_SOLVER_POSITION_ITERATIONS)
        api.CreateSolverVelocityIterationCountAttr(_SOLVER_VELOCITY_ITERATIONS)
        api.CreateStabilizationThresholdAttr(_STABILIZATION_THRESHOLD)

    def _author_scene(self, spec: SceneSpec, manifest: AssetManifest, rng: np.random.Generator):  # noqa: ANN202
        from isaacsim.core.api.objects import FixedCuboid  # type: ignore[import-not-found]  # noqa: PLC0415
        from pxr import Sdf, UsdLux  # type: ignore[import-not-found]  # noqa: PLC0415

        import omni.usd  # type: ignore[import-not-found]  # noqa: PLC0415

        self._reset_stage()
        stage = omni.usd.get_context().get_stage()

        contact = self._physics_material()

        # table
        table_mm = self._config.workspace.table_size_mm
        table_size = np.array([table_mm * _MM_TO_M, table_mm * _MM_TO_M, 0.4])
        table = FixedCuboid(
            prim_path=_TABLE, name="table",
            position=np.array([
                spec.cameras[0].look_at_mm[0] * _MM_TO_M, spec.cameras[0].look_at_mm[1] * _MM_TO_M,
                -table_size[2] / 2.0,
            ]),
            scale=table_size,
            color=np.asarray(spec.randomization.table_color_rgb, dtype=np.float64),
        )
        self._world.scene.add(table)
        table.apply_physics_material(contact)
        self._authored.append(_TABLE)

        for i, wall in enumerate(spec.bin_walls):
            path = f"{_WALL_ROOT}_{i}"
            wall_prim = FixedCuboid(
                prim_path=path, name=f"bin_wall_{i}",
                position=np.asarray(wall.center_mm, dtype=np.float64) * _MM_TO_M,
                scale=np.asarray(wall.half_extents_mm, dtype=np.float64) * 2.0 * _MM_TO_M,
                color=np.array([0.30, 0.32, 0.38]),
            )
            self._world.scene.add(wall_prim)
            wall_prim.apply_physics_material(contact)
            self._authored.append(path)

        # lighting: one distant key at the randomised elevation/azimuth plus a dome fill
        key_path = f"{_WORLD}/KeyLight"
        key = UsdLux.DistantLight.Define(stage, Sdf.Path(key_path))
        key.CreateIntensityAttr(float(spec.randomization.light_intensity))
        key.CreateAngleAttr(1.5)
        from pxr import Gf, UsdGeom  # type: ignore[import-not-found]  # noqa: PLC0415

        xform = UsdGeom.Xformable(key.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddRotateXYZOp().Set(Gf.Vec3f(
            -float(spec.randomization.light_elevation_deg), 0.0,
            float(spec.randomization.light_azimuth_deg),
        ))
        dome_path = f"{_WORLD}/DomeLight"
        dome = UsdLux.DomeLight.Define(stage, Sdf.Path(dome_path))
        dome.CreateIntensityAttr(float(spec.randomization.dome_intensity))
        self._extra_prims.extend([key_path, dome_path])

        # objects
        instances: dict[int, tuple[str, str]] = {}
        materials: dict[int, MaterialParams] = {}
        self._author_arm()

        for i, placement in enumerate(spec.objects):
            asset = manifest.get(placement.asset_id)
            family = asset.tags[0] if asset.tags else "primitive"
            material = sample_material(rng, family, placement.color_rgb)
            path = self._author_object(i, placement, asset, material, contact)
            instances[i] = (path, placement.asset_id)
            materials[i] = material
        return instances, materials

    # --- the arm -----------------------------------------------------------
    def _author_arm(self) -> None:
        """Reference the robot and hold it at park, before the settle.

        PhysX builds its articulation views when the world resets, so an arm added afterwards has no
        articulation to pose. The arm therefore stands through the settle, at the park configuration
        in `render/kinematics.py`, which is clear of the drop column: it neither catches falling
        objects nor scatters a pile it would never have touched. It is teleported to its scene
        configuration once everything has come to rest and nothing can be disturbed any more.
        """
        if self._robot is None:
            return
        from isaacsim.core.utils.stage import add_reference_to_stage  # type: ignore[import-not-found]  # noqa: PLC0415
        from isaacsim.storage.native import get_assets_root_path  # type: ignore[import-not-found]  # noqa: PLC0415

        from pxr import UsdGeom  # type: ignore[import-not-found]  # noqa: PLC0415

        import omni.usd  # type: ignore[import-not-found]  # noqa: PLC0415

        root = (get_assets_root_path() or "").rstrip("/")
        add_reference_to_stage(root + self._robot.usd_relpath, ROBOT_ROOT)

        # Isaac's UR asset and the DH table behind `ur_link_origins_mm` disagree by 180 deg about
        # the base Z axis: for every link, the USD link origin has x and y negated against the FK
        # result and z identical. willy_sim does not have to notice, because there the DH chain only
        # feeds the self-collision guard, link-to-link distances are invariant under a global
        # rotation, and the IK comes from Lula reading the USD. The moment a DH chain places a
        # camera in the world the two conventions have to be reconciled, or the arm reaches away
        # from the workspace and never appears in frame. Rotating the asset, rather than correcting
        # every FK result afterwards, leaves exactly one kinematic truth inside this generator.
        stage = omni.usd.get_context().get_stage()
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT_ROOT))
        xform.ClearXformOpOrder()
        xform.AddRotateZOp().Set(180.0)
        self._extra_prims.append(ROBOT_ROOT)
        self._articulation = None
        self._arm_joints = None

    def _bind_articulation(self) -> None:
        """Attach to the arm's articulation once physics is live. Idempotent across scenes."""
        if self._robot is None or self._arm_joints is not None:
            return
        from isaacsim.core.api.articulations import ArticulationSubset  # type: ignore[import-not-found]  # noqa: PLC0415
        from isaacsim.core.prims import SingleArticulation  # type: ignore[import-not-found]  # noqa: PLC0415

        articulation = SingleArticulation(prim_path=ROBOT_ROOT, name="datagen_arm")
        articulation.initialize()
        names = list(articulation.dof_names)
        missing = [joint for joint in _ARM_JOINT_NAMES if joint not in names]
        if missing:
            raise RuntimeError(
                f"the {self._robot.key} articulation is missing arm joints {missing}; it exposes "
                f"{names}. Without them the arm cannot be posed, and a dataset that claims a posed arm "
                f"must not be written from one standing wherever the asset happened to open."
            )
        self._articulation = articulation
        self._arm_joints = ArticulationSubset(articulation, list(_ARM_JOINT_NAMES))

    def _hold_joints(self, joints_rad: "Sequence[float] | np.ndarray") -> None:
        """Teleport the arm and hold it there. Both, because either alone drifts or snaps back.

        ``set_joint_positions`` moves the state; ``apply_action`` makes it the drive target. Without
        the second call the arm creeps back towards its old target over the following steps, and the
        rendered pose then differs from the labelled one by however far it got.
        """
        if self._arm_joints is None:
            return
        target = np.asarray(joints_rad, dtype=np.float64)
        self._arm_joints.set_joint_positions(target)
        self._arm_joints.apply_action(joint_positions=target)
        self._step(2, render=False)

    def _set_arm_visibility(self, visible: bool) -> None:
        from pxr import UsdGeom  # type: ignore[import-not-found]  # noqa: PLC0415

        import omni.usd  # type: ignore[import-not-found]  # noqa: PLC0415

        prim = omni.usd.get_context().get_stage().GetPrimAtPath(ROBOT_ROOT)
        if not prim.IsValid():
            return
        imageable = UsdGeom.Imageable(prim)
        imageable.MakeVisible() if visible else imageable.MakeInvisible()

    def _self_clearance_mm(self, joints_rad: np.ndarray) -> float:
        """The smallest non-adjacent link gap, from `render/kinematics.py`.

        One implementation for every engine: a private copy here and a second in `noengine.py`
        would compute different quantities under the same name.
        """
        from datagen.render.kinematics import self_clearance_mm  # noqa: PLC0415

        return self_clearance_mm(self._config.render.arm.robot_model, joints_rad)

    # --- physics -----------------------------------------------------------
    def _object_poses(self, instances: Mapping[int, tuple[str, str]]) -> dict[int, np.ndarray]:
        from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]  # noqa: PLC0415

        return {
            index: np.asarray(SingleRigidPrim(path).get_world_pose()[0], dtype=np.float64)
            for index, (path, _asset) in instances.items()
        }

    def _settle_and_verify(
        self, instances: dict[int, tuple[str, str]],
    ) -> tuple[bool, int, str, str, tuple[int, ...]]:
        """Settle until it stops helping. ``(stable, steps, note, status, dropped)``.

        More than one round because piles legitimately need it: rejecting on the first check throws
        away the most interesting family. An unsettled scene settles longer, and is then discarded
        and logged; this is the "longer".

        An object that leaves the table is removed and the scene continues. Refusing the whole scene
        instead would throw away every correctly settled object to punish one runaway, and the
        runaway is already outside the placement area the cameras frame, so it was never going to be
        in a picture. It is dropped from ``instances`` and hidden rather than left lying in the
        scene, which is what makes that true rather than merely likely.

        The scene is still refused when the escape is not incidental; see ``_drop_escaped``. A heap
        that mostly falls off the table is a broken scene, not a smaller one.
        """
        render = self._config.render
        self._step(render.settle_steps, render=False)
        total = render.settle_steps + render.stability_check_steps
        moved, worst = self._residual_motion(instances)
        dropped: list[int] = []
        rounds = 0
        while rounds < _MAX_EXTRA_SETTLE_ROUNDS:
            refusal = self._drop_escaped(instances, dropped)
            if refusal:
                return False, total, refusal, "escaped", tuple(dropped)
            if dropped and dropped[-1] not in instances:
                # The scene changed under the measurement: what was moving may have been the object
                # that just left. Re-measure before spending another settle round on a stale number.
                moved, worst = self._residual_motion(instances)
                total += render.stability_check_steps
            if moved <= render.stability_tolerance_mm:
                return True, total, worst, "ok", tuple(dropped)
            self._step(render.extra_settle_steps, render=False)
            moved, worst = self._residual_motion(instances)
            total += render.extra_settle_steps + render.stability_check_steps
            rounds += 1
        refusal = self._drop_escaped(instances, dropped)
        if refusal:
            return False, total, refusal, "escaped", tuple(dropped)
        if moved > render.stability_tolerance_mm:
            return False, total, worst, "unstable", tuple(dropped)
        return True, total, worst, "ok", tuple(dropped)

    def _drop_escaped(self, instances: dict[int, tuple[str, str]], dropped: list[int]) -> str:
        """Remove whatever has left the table. Returns a refusal reason, or ``""`` to continue.

        Removal is by hiding the prim and forgetting it, not by deleting it. Deleting a rigid body
        while the simulation runs invalidates PhysX's tensor views and takes every later scene down
        with it, which is also why ``_reset_stage`` is ordered the way it is. Hidden and forgotten
        is enough: it cannot reach an image, a mask, or a label, because every one of those is built
        from ``instances``.
        """
        centre = self._config.workspace.center_mm
        gone: list[tuple[int, str]] = []
        for index, position in sorted(self._object_poses(instances).items()):
            x, y, z = (float(v) * 1000.0 for v in position)
            half_mm = self._config.workspace.table_size_mm / 2.0
            if (abs(x - centre[0]) > half_mm or abs(y - centre[1]) > half_mm
                    or z < _BELOW_TABLE_MM):
                gone.append((index, f"object {index} left the table at ({x:.0f}, {y:.0f}, {z:.0f}) mm"))
        if not gone:
            return ""

        remaining = len(instances) - len(gone)
        started_with = len(instances) + len(dropped)
        if remaining < _MIN_OBJECTS_AFTER_DROP or len(dropped) + len(gone) > started_with / 2:
            # Not an incidental runaway. A scene that mostly ended up on the floor did not happen the
            # way the layout described, and shipping the remnant would quietly relabel it as an easier
            # scene than the one that was asked for.
            return ("; ".join(note for _index, note in gone)
                    + f"; {len(dropped) + len(gone)} of {started_with} objects left, which is the "
                      f"scene falling apart rather than one runaway")

        self._set_object_visibility(instances, set(instances) - {index for index, _ in gone})
        for index, _note in gone:
            instances.pop(index, None)
            dropped.append(index)
        return ""

    def _residual_motion(self, instances: Mapping[int, tuple[str, str]]) -> tuple[float, str]:
        """How far the busiest object still moves over the check window, in mm, and which one.

        Returns the number rather than a verdict: "the scene is still moving" is not something
        anyone can act on, while "object 4 moved 37.20 mm" says whether to settle longer or to stop
        dropping that object from that height.
        """
        before = self._object_poses(instances)
        self._step(self._config.render.stability_check_steps, render=False)
        after = self._object_poses(instances)
        worst_index, worst_mm = -1, 0.0
        for index in before:
            moved_mm = float(np.linalg.norm(after[index] - before[index])) / _MM_TO_M
            if moved_mm > worst_mm:
                worst_index, worst_mm = index, moved_mm
        return worst_mm, f"object {worst_index} moved {worst_mm:.2f} mm"

    # --- cameras + rendering ----------------------------------------------
    def _camera(self, placement: CameraPlacement) -> Any:
        from isaacsim.core.utils.viewports import set_camera_view  # type: ignore[import-not-found]  # noqa: PLC0415
        from isaacsim.sensors.camera import Camera  # type: ignore[import-not-found]  # noqa: PLC0415

        path = f"{_CAMERA_ROOT}/{placement.name}"
        camera = self._cameras.get(placement.name)
        if camera is None:
            camera = Camera(
                prim_path=path,
                position=np.asarray(placement.position_mm, dtype=np.float64) * _MM_TO_M,
                resolution=tuple(placement.resolution),
            )
            camera.initialize()
            camera.add_distance_to_image_plane_to_frame()
            camera.add_instance_id_segmentation_to_frame()
            try:
                camera.set_clipping_range(0.05, 1.0e6)
            except Exception:  # noqa: BLE001 (near-clip API varies by Isaac version)
                pass
            # Author the lens so the configured FOV is the real one. Without this the field is
            # decorative and every dataset silently carries Isaac's default aperture instead.
            from src.willy_sim.scene.cameras import set_horizontal_fov

            measured = set_horizontal_fov(camera, placement.horizontal_fov_deg, placement.resolution)
            if abs(measured - placement.horizontal_fov_deg) > 0.5:
                raise RuntimeError(
                    f"camera {placement.name}: asked for {placement.horizontal_fov_deg} deg HFOV, the lens "
                    f"reads back {measured:.2f} deg; the intrinsics in the labels would not match the image"
                )
            self._cameras[placement.name] = camera
        set_camera_view(
            eye=list(np.asarray(placement.position_mm) * _MM_TO_M),
            target=list(np.asarray(placement.look_at_mm) * _MM_TO_M),
            camera_prim_path=path,
        )
        return camera

    def _light_census(self) -> str:
        """Every light on the stage and its intensity: the one thing that kills colour, not depth.

        Depth and segmentation are geometry passes and do not care whether the scene is lit; the
        colour pass is the only output that goes black when the lights are gone. ``_reset_stage``
        removes the light prims between scenes and ``_author_scene`` re-creates them, so "the new
        lights did not take" is a failure this pipeline can have, and the census rules it in or out.
        """
        from pxr import UsdLux  # type: ignore[import-not-found]  # noqa: PLC0415

        import omni.usd  # type: ignore[import-not-found]  # noqa: PLC0415

        stage = omni.usd.get_context().get_stage()
        found = []
        for prim in stage.Traverse():
            light = UsdLux.LightAPI(prim)
            if not light:
                continue
            attribute = light.GetIntensityAttr()
            intensity = float(attribute.Get()) if attribute and attribute.Get() is not None else 0.0
            from pxr import UsdGeom  # type: ignore[import-not-found]  # noqa: PLC0415

            token = UsdGeom.Imageable(prim).ComputeVisibility()
            hidden = "" if token != UsdGeom.Tokens.invisible else " (INVISIBLE)"
            found.append(f"{prim.GetName()}={intensity:.0f}{hidden}")
        return ", ".join(found) if found else "NONE; the stage has no lights at all"

    def _read_beauty(
        self, camera: Any, placement: CameraPlacement,
    ) -> tuple[np.ndarray | None, np.ndarray, Any]:
        """The beauty pass, with one recovery attempt if the colour buffer is dead.

        A wedged colour buffer stays at exactly zero for every step the read loop allows, while
        depth and segmentation on the same camera stay correct throughout, and rebuilding the camera
        does not clear it. The wedge is not in this camera's annotators but in the path tracer's own
        accumulation, so the recovery below toggles the render mode: leaving and re-entering
        ``PathTracing`` restarts it.

        The raster read taken on the way through is not a fallback, it is the diagnostic. A lit
        raster frame at the same pose says the camera, the geometry and the lighting are all fine
        and only light transport is failing; a black one says the camera is looking at nothing,
        which is a different fault with a different fix. Reporting which of the two happened is the
        point.

        The scene is refused if the toggle does not help. Writing the frame anyway is the defect
        this path exists to prevent, and silently substituting the raster image for the path-traced
        one would be worse: the dataset would record a render mode it did not use.

        The live camera is returned rather than only stored, because the mask passes that follow
        read from the same object.
        """
        try:
            rgb, depth_mm = self._read_stable(camera, require_rgb=True)
            return rgb, depth_mm, camera
        except ColourBufferNeverArrived as first:
            self._set_render_mode("raster")
            raster_rgb, _raster_depth, _frame = self._read_view(camera)
            self._step(_READ_STEPS)
            raster_rgb, _raster_depth, _frame = self._read_view(camera)
            raster = describe_rgb(raster_rgb)
            self._set_render_mode(self._config.render.mode)
            print(f"[datagen] {placement.name}: colour buffer silent under "
                  f"{self._config.render.mode}; the same pose on the raster path is "
                  f"{'LIT' if raster.renderable else 'BLACK'} "
                  f"(lit {raster.lit_fraction * 100:.1f}%, {raster.distinct_levels} levels); "
                  f"lights on stage: {self._light_census()}", flush=True)
            # The print above is the operator's live diagnostic; this is the line that survives the
            # run. It is the only retry in the render path.
            logger.warning("%s: colour buffer silent under %s, retrying after a render-mode toggle "
                           "(raster at the same pose is %s)",
                           placement.name, self._config.render.mode,
                           "LIT" if raster.renderable else "BLACK")
            try:
                rgb, depth_mm = self._read_stable(camera, require_rgb=True)
                return rgb, depth_mm, camera
            except ColourBufferNeverArrived as second:
                raise ColourBufferNeverArrived(
                    f"{second}\n  The render-mode toggle did not recover it. At the same camera pose "
                    f"the raster path is {'LIT' if raster.renderable else 'BLACK'} "
                    f"({raster.lit_fraction * 100:.1f}% lit, {raster.distinct_levels} distinct "
                    f"levels, peak {raster.peak}); so "
                    + ("the geometry and lighting are fine and the path tracer is what is failing."
                       if raster.renderable else
                       "the camera is looking at nothing renderable, which is a placement problem "
                       "rather than a renderer one.")
                ) from first

    def _read_view(self, camera: Any) -> tuple[np.ndarray | None, np.ndarray, dict]:
        frame = camera.get_current_frame()
        rgba = frame.get("rgb") if isinstance(frame, Mapping) else None
        rgb = np.asarray(rgba)[..., :3].copy() if rgba is not None else None
        depth_m = np.asarray(camera.get_depth(), dtype=np.float64)
        depth_mm = np.where(np.isfinite(depth_m), depth_m * 1000.0, 0.0)
        return rgb, depth_mm, dict(frame) if isinstance(frame, Mapping) else {}

    def _read_stable(
        self, camera: Any, *, require_rgb: bool = False,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Pump until two consecutive reads agree on content, then return the second.

        Not a tick count, and not bit-equality either. Two reads of a motionless scene differ by
        float noise of about 0.01 mm on a few per cent of pixels, plus a silhouette pixel or two
        flickering between surfaces, so demanding identical depth images never succeeds. A frame
        belonging to a different scene state differs by more than 2 mm across thousands of pixels,
        so a small pixel budget keeps the check sharp while letting float noise through.

        A visibility change is present in the depth after 2 steps and does not change through 12, so
        this loop normally costs exactly two reads. The staleness this guards against is the
        instance-id annotator's; depth does not have it.

        ``require_rgb`` extends the agreement to the colour buffer, because depth alone is not
        enough: the depth, the masks and the labels of a scene can all be correct while every one of
        its RGB frames is entirely zero, and without the flag that frame goes to disk unexamined.
        Only the beauty pass passes it. The solo passes below discard their RGB, and demanding a lit
        frame from a scene rendered with every object hidden would be wrong.

        The colour check is content, not stability, and that asymmetry is deliberate: under path
        tracing two consecutive frames differ by sampling noise everywhere, so "two RGB reads agree"
        has no threshold that means anything, while "this frame is a render at all" does.
        """
        previous: np.ndarray | None = None
        disagreeing = -1
        content = describe_rgb(None)
        for _ in range(_READ_ATTEMPTS):
            self._step(_READ_STEPS)
            rgb, depth_mm, _frame = self._read_view(camera)
            budget = max(_READ_AGREE_MIN_PX, int(depth_mm.size * _READ_AGREE_FRACTION))
            if previous is not None:
                disagreeing = int(
                    np.count_nonzero(np.abs(previous - depth_mm) > _DEPTH_MATCH_TOLERANCE_MM)
                )
                if disagreeing <= budget:
                    if not require_rgb:
                        return rgb, depth_mm
                    content = describe_rgb(rgb)
                    if content.renderable:
                        return rgb, depth_mm
                    # Depth is ready and colour is not, so keep pumping rather than return. Falling
                    # through resets nothing: the next iteration re-checks both, and the loop's own
                    # attempt budget bounds the wait.
            previous = depth_mm
        if require_rgb and not content.renderable and disagreeing >= 0:
            # Distinguish the two failures. A depth timeout and a colour timeout have different
            # causes and different fixes, and reporting "the depth never settled" for a black frame
            # sends the next reader looking in the wrong buffer. A distinct type, not a flag,
            # because exactly one caller knows how to recover from it.
            raise ColourBufferNeverArrived(
                f"the depth settled but the colour buffer never did, after "
                f"{_READ_ATTEMPTS * _READ_STEPS} steps: {content.why_not()}. Refusing to write a "
                f"frame that is not an image; the labels for it would be perfect and the picture "
                f"blank, which is exactly how this went unnoticed once already"
            )
        raise RuntimeError(
            f"the depth buffer never settled after {_READ_ATTEMPTS * _READ_STEPS} steps; two "
            f"consecutive reads disagreed on {disagreeing} px by > {_DEPTH_MATCH_TOLERANCE_MM} mm, "
            f"over a budget of {max(_READ_AGREE_MIN_PX, int(np.prod(previous.shape) * _READ_AGREE_FRACTION)) if previous is not None else 0}. "
            f"The count is reported because it is the only thing that says whether the budget is too "
            f"tight or the frame is genuinely wrong"
        )

    def _set_object_visibility(self, instances: Mapping[int, tuple[str, str]], visible: set[int]) -> None:
        from pxr import UsdGeom  # type: ignore[import-not-found]  # noqa: PLC0415

        import omni.usd  # type: ignore[import-not-found]  # noqa: PLC0415

        stage = omni.usd.get_context().get_stage()
        for index, (path, _asset) in instances.items():
            imageable = UsdGeom.Imageable(stage.GetPrimAtPath(path))
            if index in visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()

    def _set_arm_visible(self, visible: bool) -> None:
        """Show or hide the whole robot. The same USD mechanism the object solo passes use."""
        if self._robot is None:
            return
        from pxr import UsdGeom  # type: ignore[import-not-found]  # noqa: PLC0415

        import omni.usd  # type: ignore[import-not-found]  # noqa: PLC0415

        stage = omni.usd.get_context().get_stage()
        imageable = UsdGeom.Imageable(stage.GetPrimAtPath(ROBOT_ROOT))
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def _geometric_masks(
        self, camera: Any, instances: Mapping[int, tuple[str, str]], scene_depth_mm: np.ndarray,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray]:
        """``(unoccluded, visible, arm)`` from depth alone. No annotator involved.

        Isaac's instance-id map is untrustworthy here: its ``idToLabels`` can belong to a different
        frame than its id image, which permutes objects in a way no set comparison can detect, and
        an object is then handed another object's mask. Depth has no such indirection.

        * unoccluded: render the object alone; it is wherever the depth differs from a background
          pass of the empty scene. Geometry, not bookkeeping.
        * visible: a pixel in the full scene belongs to the object whose solo depth matches the
          scene depth there. Nothing else can be in front of it, because the scene depth is the
          nearest surface by definition.

        Two properties then hold by construction rather than by assertion: visible is a subset of
        unoccluded, and no object can claim another's pixels.
        """
        tolerance = _DEPTH_MATCH_TOLERANCE_MM
        self._set_object_visibility(instances, set())
        _rgb, background = self._read_stable(camera)

        # The arm, for one extra pass. `background` already holds the table, the walls and the arm,
        # so hiding the robot and reading again isolates it exactly: the same depth-difference trick
        # the object silhouettes use, and for the same reason, that nothing here trusts an
        # annotator.
        #
        # A capsule chain is not good enough as a substitute. Built from `ur_link_origins_mm` it
        # misses part of the arm whenever the arm is folded, because the DH joint origins do not
        # follow the upper arm's housing, which sits some 180 mm off the line between two of them.
        # No radius fixes it: a small one leaves the gap, and one large enough to close it claims
        # table and object points that are not the arm.
        arm = np.zeros(scene_depth_mm.shape, dtype=bool)
        if self._robot is not None:
            self._set_arm_visible(False)
            _rgb, without_arm = self._read_stable(camera)
            self._set_arm_visible(True)
            silhouette = (np.abs(background - without_arm) > tolerance) & (background > 0.0)
            # Visible in the full scene, not merely present in the empty one: an object standing in
            # front of the arm owns those pixels, exactly as `visible` is defined for objects.
            arm = (silhouette
                   & (np.abs(scene_depth_mm - background) <= tolerance)
                   & (scene_depth_mm > 0.0))

        unoccluded: dict[int, np.ndarray] = {}
        solo_depths: dict[int, np.ndarray] = {}
        for index in sorted(instances):
            self._set_object_visibility(instances, {index})
            _rgb, solo = self._read_stable(camera)
            # "different from the empty scene" = the object is there. An invalid (0) background pixel
            # that becomes valid also counts: the object filled a hole in the sky.
            differs = (np.abs(solo - background) > tolerance) & (solo > 0.0)
            unoccluded[index] = differs
            solo_depths[index] = np.where(differs, solo, np.inf)

        self._set_object_visibility(instances, set(instances))
        visible: dict[int, np.ndarray] = {}
        for index in sorted(instances):
            visible[index] = (
                (np.abs(scene_depth_mm - solo_depths[index]) <= tolerance)
                & (scene_depth_mm > 0.0)
                & unoccluded[index]
            )
        return unoccluded, visible, arm

    def _pose_the_arm(
        self, spec: SceneSpec, settled: Mapping[int, Any], manifest: AssetManifest,
    ) -> tuple[Any, np.ndarray | None]:
        """Put the arm where it would stand for this scene. ``(ArmPose, wrist camera pose | None)``.

        The objects are already at rest, so the clearance check measures the scene as it will be
        photographed rather than as it was dropped. ``None`` for the camera means every attempt was
        refused and the arm parked: the wrist view is then skipped and the fixed cameras still
        render, which is the rule for an unreachable viewpoint and holds unchanged when the reason
        is collision rather than reach.
        """
        if self._robot is None:
            return None, None
        self._bind_articulation()
        mount = self._robot.wrist_camera_mount
        wrist = next((camera for camera in spec.cameras if camera.mount is CameraMount.WRIST), None)
        if mount is None or wrist is None:
            # No measured mount (or no wrist view asked for): park, and never guess an extrinsic.
            self._hold_joints(PARK_JOINTS_RAD)
            return ArmPose(PARK_JOINTS_RAD, "park", 0, ("no measured wrist-camera mount",)), None

        centres, radii = [], []
        for index, (position, _orientation) in settled.items():
            asset = manifest.get(spec.objects[index].asset_id)
            centres.append(position)
            radii.append(float(np.linalg.norm(np.asarray(asset.extent_mm, dtype=np.float64))) / 2.0)

        camera_to_link = mount.camera_to_link()
        by_joints = self._config.render.arm.viewpoint_source == "joint_fk"
        centre = None
        if by_joints:
            # The one IK solve joint_fk does: where the arm stands to look at this scene. Everything
            # after it is a joint-space draw around that, which is what makes the mode incapable of
            # an unreachable viewpoint. If this solve fails, `centre` stays None and the call below
            # falls back to the target_ik path.
            centre = solve_joints_for_camera(
                self._robot.key, self._sample_wrist_pose(wrist, 0), camera_to_link,
            )
        pose, camera_to_base = resolve_arm_pose(
            self._robot.key, camera_to_link,
            lambda attempt: self._sample_wrist_pose(wrist, attempt),
            np.asarray(centres, dtype=np.float64).reshape(-1, 3),
            np.asarray(radii, dtype=np.float64),
            max_attempts=self._config.render.arm.max_viewpoint_attempts,
            self_collision_margin_mm=self._config.render.arm.self_collision_margin_mm,
            self_clearance=self._self_clearance_mm,
            # Only joint_fk needs these: it proposes configurations instead of camera poses, and so
            # is the only source that can aim the camera at nothing. Under target_ik the camera is
            # aimed by construction, and re-testing it there would reject viewpoints the layout
            # asked for.
            propose_joints=(lambda attempt: sample_joint_configuration(attempt, centre_rad=centre))
            if by_joints and centre is not None else None,
            accept_camera=(lambda pose_matrix: self._camera_sees_scene(wrist, pose_matrix))
            if by_joints and centre is not None else None,
        )
        self._hold_joints(pose.joints_rad)
        return pose, camera_to_base

    def _camera_sees_scene(self, wrist: CameraPlacement, camera_to_base: np.ndarray) -> str | None:
        """Does this camera pose actually contain the scene? A rejection reason, or ``None``.

        Only ``joint_fk`` asks. A drawn configuration is reachable by construction and can still be
        useless: the arm holds a perfectly comfortable pose while the wrist camera faces the far
        wall, and the resulting view is a correctly labelled picture of nothing. The check is the
        weakest one that rules that out, namely that the point the layout wanted photographed lands
        inside the frame. A quality bar instead would re-impose the viewpoint bias joint_fk exists
        to expose.
        """
        target = np.asarray(wrist.look_at_mm, dtype=np.float64).reshape(1, 3)
        k = intrinsics_matrix(wrist.resolution, wrist.horizontal_fov_deg)
        uv = project_to_pixels(target, camera_to_base, k)[0]
        if not np.all(np.isfinite(uv)):
            return "the camera faces away from the scene"
        width, height = wrist.resolution
        if not (0.0 <= uv[0] < width and 0.0 <= uv[1] < height):
            return (f"the scene projects to ({uv[0]:.0f}, {uv[1]:.0f}) px, outside the "
                    f"{width}x{height} frame")
        return None

    def _sample_wrist_pose(self, wrist: CameraPlacement, attempt: int) -> np.ndarray:
        """Candidate wrist-camera poses. Attempt 0 is the viewpoint the layout planned.

        Later attempts keep the radius and redraw the direction between 45 and 80 degrees of
        elevation, deterministically from the attempt index. A resample driven by a live rng would
        make two runs of one seed disagree about which scenes have wrist views.
        """
        if attempt == 0:
            return look_at_camera_to_base(wrist.position_mm, wrist.look_at_mm)
        generator = np.random.default_rng([_WRIST_RESAMPLE_STREAM, attempt])
        target = np.asarray(wrist.look_at_mm, dtype=np.float64)
        offset = np.asarray(wrist.position_mm, dtype=np.float64) - target
        radius = float(np.linalg.norm(offset))
        azimuth = float(generator.uniform(0.0, 2.0 * np.pi))
        elevation = float(generator.uniform(np.radians(45.0), np.radians(80.0)))
        position = target + radius * np.array([
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ])
        return look_at_camera_to_base(position, target)

    # --- the scene ---------------------------------------------------------
    def render(
        self, spec: SceneSpec, manifest: AssetManifest, rng: np.random.Generator,
    ) -> SceneRenderResult:
        """Author, settle, verify, render every view, and label. Never raises for a rejected scene."""
        started = time.perf_counter()
        instances, materials = self._author_scene(spec, manifest, rng)
        self._world.reset()
        stable, steps, note, status, dropped = self._settle_and_verify(instances)
        if not stable:
            prefix = ("an object left the table after" if status == "escaped"
                      else "still moving after")
            logger.warning("%s REJECTED (%s): %s %d settle steps; %s",
                           spec.scene_id, status, prefix, steps, note)
            return SceneRenderResult(
                scene_id=spec.scene_id, status=status, materials=materials,
                seconds=time.perf_counter() - started, dropped=dropped,
                note=f"{prefix} {steps} settle steps "
                     f"(tolerance {self._config.render.stability_tolerance_mm} mm): {note}",
            )

        from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]  # noqa: PLC0415

        settled: dict[int, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {}
        for index, (path, _asset) in instances.items():
            position, orientation = SingleRigidPrim(path).get_world_pose()
            xyz = np.asarray(position, dtype=np.float64) * 1000.0
            settled[index] = (
                (float(xyz[0]), float(xyz[1]), float(xyz[2])),
                _quat_xyzw_from_wxyz(np.asarray(orientation)),
            )

        arm_pose, wrist_camera = self._pose_the_arm(spec, settled, manifest)
        if arm_pose is not None and wrist_camera is None:
            # Every viewpoint was refused, so the wrist view is skipped and the fixed cameras still
            # render. The arm's own reasons are the difference between "out of reach" and "it would
            # have stood in the scene". The line below reads them from ``reasons``, which ``ArmPose``
            # does not define (its field is ``rejections``), so it always reports "no reason
            # recorded" and nothing downstream records them either.
            logger.warning("%s: no wrist viewpoint after %d attempt(s); %s",
                           spec.scene_id, getattr(arm_pose, "attempts", 0),
                           "; ".join(getattr(arm_pose, "reasons", ()) or ("no reason recorded",)))

        views: list[ViewRender] = []
        for placement in spec.cameras:
            if placement.mount is CameraMount.WRIST:
                if self._config.render.arm.mode == "absent":
                    # No arm: the wrist viewpoint is still a valid camera pose, and rendering it
                    # without the arm is honest as long as nothing claims the arm was there. The
                    # dataset records the arm mode, so a consumer can tell these apart.
                    pass
                elif wrist_camera is None:
                    # No configuration put the camera here without the arm reaching past its
                    # limits, folding into itself, or standing in the scene, so the view is skipped
                    # and says so, rather than being rendered from a camera floating where no arm
                    # could hold it.
                    views.append(ViewRender(
                        name=placement.name, outcome=ViewOutcome.SKIPPED_UNREACHABLE,
                        camera_position_mm=placement.position_mm,
                        camera_look_at_mm=placement.look_at_mm,
                    ))
                    continue
            if placement.mount is CameraMount.WRIST and wrist_camera is not None:
                # The resolved viewpoint, which is not always the planned one: a later attempt lands
                # somewhere else on the hemisphere. Substituting it here keeps the camera, the
                # recorded extrinsic and the arm describing the same pose. The alternative is an
                # image taken from where the arm is and labelled with where the layout wanted it.
                placement = _placement_at(placement, wrist_camera)
            if self._robot is not None:
                self._set_arm_visibility(
                    self._config.render.arm.visible_in_views == "all"
                    or placement.mount is CameraMount.WRIST,
                )
            camera = self._camera(placement)
            self._set_object_visibility(instances, set(instances))
            if self._config.render.depth_only:
                # No beauty pass and no colour-buffer retry: there is no colour buffer to be silent.
                # Depth still goes through `_read_stable`, which is where the two-agreeing-reads
                # rule lives. That rule is about the annotator being one frame behind, not about
                # colour, and dropping it here would trade a picture for a wrong depth.
                self._set_render_mode("raster")
                rgb, depth_mm = None, self._read_stable(camera, require_rgb=False)[1]
            else:
                self._set_render_mode(self._config.render.mode)
                rgb, depth_mm, camera = self._read_beauty(camera, placement)

            # Silhouettes do not depend on light transport, so the solo passes run on the raster
            # renderer while the beauty pass above was path-traced. The same geometry, far cheaper.
            if self._config.render.measure_visibility:
                self._set_render_mode("raster")
                unoccluded, visible_masks, arm_mask = self._geometric_masks(
                    camera, instances, depth_mm)
            else:
                empty = np.zeros(depth_mm.shape, dtype=bool)
                unoccluded = {index: empty for index in instances}
                visible_masks = {index: empty for index in instances}
                # No solo passes means no arm pass either. Empty rather than absent, so a consumer
                # reads "no arm pixels here" instead of guessing whether the channel exists.
                arm_mask = empty
            instance_map = instance_map_from_masks(visible_masks, depth_mm.shape)

            labels = build_object_labels(
                instance_map, unoccluded,
                {index: (instances[index][1], settled[index][0], settled[index][1])
                 for index in instances},
            )
            broken = _inconsistent_labels(labels)
            if broken:
                # visible is a subset of unoccluded by construction, so a violation here means the
                # two depth passes disagree about geometry; worth refusing loudly rather than
                # writing.
                logger.error("%s/%s REJECTED (label_inconsistent): visible > unoccluded for %s; "
                             "the two depth passes disagree about geometry",
                             spec.scene_id, placement.name, broken)
                return SceneRenderResult(
                    scene_id=spec.scene_id, status="label_inconsistent", materials=materials,
                    seconds=time.perf_counter() - started,
                    note=f"visible > unoccluded for {broken}; the depth passes disagree",
                )
            noisy = None
            if self._config.render.depth_noise.enabled:
                sensor = SENSOR_MODELS[self._config.render.depth_noise.sensor]
                noisy = inject_depth_noise(depth_mm, sensor, rng)
            views.append(ViewRender(
                name=placement.name, outcome=ViewOutcome.RENDERED, rgb=rgb, depth_mm=depth_mm,
                depth_noisy_mm=noisy, instance_map=instance_map, arm_mask=arm_mask, labels=labels,
                camera_position_mm=placement.position_mm, camera_look_at_mm=placement.look_at_mm,
                camera_to_base=look_at_camera_to_base(placement.position_mm, placement.look_at_mm),
                intrinsics=intrinsics_matrix(placement.resolution, placement.horizontal_fov_deg),
            ))

        # Debug, not info: the writer logs one info per scene when the result reaches the index, and
        # two identical progress lines per scene is how a log stops being read. What is here and not
        # there is the timing a slow render gets diagnosed from.
        logger.debug("%s rendered %d view(s) in %.1f s (%d object(s), %d dropped)",
                     spec.scene_id, len(views), time.perf_counter() - started, len(settled),
                     len(dropped))
        return SceneRenderResult(
            scene_id=spec.scene_id, status="ok" if views else "no_view_rendered",
            views=tuple(views), settled_poses=settled, materials=materials,
            seconds=time.perf_counter() - started, arm=arm_pose, dropped=dropped,
        )


def _inconsistent_labels(labels: "tuple[ObjectLabel, ...]") -> str:
    """Objects showing more visible pixels than their own unoccluded silhouette, as a readable list.

    The cheapest cross-check between the two render passes, and the one that catches a frame-lagged
    annotator. The slack is 5 % plus 8 px for anti-aliasing; anything beyond that means the passes
    are not describing the same frame.
    """
    return ", ".join(
        f"{label.asset_id}({label.visible_px}>{label.unoccluded_px})"
        for label in labels
        if label.visible_px > label.unoccluded_px * 1.05 + 8
    )


def family_of(spec: SceneSpec) -> str:
    """The family name as a plain string, for the index."""
    return spec.family.value if isinstance(spec.family, SceneFamily) else str(spec.family)
