"""Isaac scene builder: robot, table, object(s), bin walls, lights and overhead camera.

Authors everything a pick needs onto a live Isaac stage and computes the eye-to-hand ``CAMERA -> BASE``
transform the grasp calculator consumes. The robot is the USD of the configured UR model, by default
``ur5e.usd`` with its baked ``Gripper="Robotiq_2f_85"`` variant selected: one 12-DoF articulation
rooted at ``/World/UR5e``, so there is no manual flange joint.

All ``isaacsim.*`` imports are lazy, inside :func:`build_combined_scene`, so this module imports on a
machine with no Isaac Sim installed. Coordinate frames: the Isaac world is metres and Z-up, and the UR
base sits at the world origin, so the base frame coincides with the world frame. Everything on the
Willy side is millimetres in ``Frame.BASE``.
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.geometry import Frame, Transform
from src.utility.log_cfg import create_logger
from src.willy_sim.constants import SCENE_BUILD_LOG_FILE, WILLY_SIM_LOG_DIR

from .cameras import (
    camera_elevation_deg,
    camera_to_base_ground_truth,
    look_at_quat_in_link_xyzw,
    set_horizontal_fov,
    top_down_camera_to_base_matrix,
)
from .constants import (
    ARM_PRIM,
    ARUCO_DICT_NAME,
    ARUCO_LENGTH_MM,
    BIN_WALL_PRIM,
    CAMERA_POS_M,
    CAMERA_PRIM,
    CAMERA_RESOLUTION,
    KLT_BIN_PRIM,
    MARKER_PRIM,
    OBJECT_PRIM,
    TABLE_PRIM,
)
from .markers import add_aruco_marker

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot.sim_schema import SimConfig
    from src.robot.drivers.sim._isaac_protocols import IsaacCamera
    from src.willy_sim.grippers import MountedGripperSpec

#: What is on the stage. Kept apart from the boot log because "the cell started" and "the stage holds
#: these objects, this camera and this extrinsic" fail independently: a scene that renders nothing
#: boots perfectly well, and the first symptom is a pick that finds no candidates.
_LOG = create_logger("SceneBuilder", log_file=SCENE_BUILD_LOG_FILE, log_dir=WILLY_SIM_LOG_DIR)




@dataclass(frozen=True)
class SceneHandles:
    """What the runner needs after the scene is built."""

    camera: IsaacCamera         # the live isaacsim.sensors.camera.Camera
    object_specs: tuple[tuple[str, str], ...]  # (prim_path, label) per authored object (>=1)
    camera_to_base: Transform   # Frame.CAMERA -> Frame.BASE, for the grasp calculator / resolver
    marker_prim_path: str | None = None  # dedicated calibration marker, if requested

    @property
    def object_prim_path(self) -> str:
        """The first authored object's prim path, which the single-object runners read."""
        return self.object_specs[0][0]



@dataclass(frozen=True, slots=True)
class SceneAppearance:
    """Appearance overrides for :func:`build_combined_scene`, varied together.

    Groups the table colour and the dome and key light intensities, which are the values a scene that
    must not look like the built-in one changes as a set. Each ``None`` keeps the built-in default: a
    gray table, dome intensity 150 and key intensity 300. A default ``SceneAppearance()`` therefore
    leaves the scene exactly as it would be with none.
    """

    table_color: tuple[float, float, float] | None = None
    dome_intensity: float | None = None
    key_intensity: float | None = None



def _mm_to_m(mm: object) -> np.ndarray:
    """Convert a tuple or list of millimetres to a float64 array of metres, the Isaac world unit."""
    return np.asarray(mm, dtype=np.float64) / 1000.0



def _vec3(values: Iterable[float]) -> tuple[float, float, float]:
    """Pin a config tuple / numpy row to a 3-vector for the geometry helpers."""
    x, y, z = (float(v) for v in values)
    return (x, y, z)



def _author_mesh_collider(
    stage: object, prim_path: str, *, approximation: str, mass_kg: float
) -> None:
    """Make a referenced non-physics YCB mesh subtree a graspable rigid body.

    The ``/Isaac/Props/YCB/Axis_Aligned/`` variants ship with geometry only, no collider and no rigid
    body, unlike the pre-rigged ``Axis_Aligned_Physics/`` ones: referenced as they are, they free-fall
    through the table. This authors UsdPhysics at runtime, following the layout of
    ``Axis_Aligned_Physics/003_cracker_box.usd``: ``RigidBodyAPI`` and ``MassAPI`` on the body root,
    ``CollisionAPI`` and ``MeshCollisionAPI`` (``physics:approximation``) on each descendant ``Mesh``.
    It is idempotent: an asset whose referenced root already carries a ``RigidBodyAPI``, as a
    pre-rigged physics variant does, is left untouched. Must run after ``add_reference_to_stage`` and
    before ``SingleRigidPrim``, which reads the USD physics state at construction. The ``pxr`` import
    is lazy, so this module imports without Isaac Sim installed.
    """
    from pxr import Usd, UsdGeom, UsdPhysics  # type: ignore[import-not-found]

    root = stage.GetPrimAtPath(prim_path)  # type: ignore[attr-defined]
    if root.HasAPI(UsdPhysics.RigidBodyAPI):
        return  # already a physics asset (Axis_Aligned_Physics): nothing to author
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass_kg))
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)



def _mount_standalone_gripper(
    stage: object, arm_prim_path: str, spec: MountedGripperSpec, root: str
) -> str:
    """Mount a standalone vendor gripper USD on the UR wrist instead of the baked Gripper variant.

    The steps are: reference the gripper USD under the arm, drop its own articulation root and
    world-fix so PhysX merges its bodies and joints into the arm articulation, then add a fixed joint
    from ``wrist_3_link`` to the gripper base, oriented by ``spec.mount_rotation_matrix`` so the
    gripper approach lands on wrist_3 +Y and translated by ``spec.flange_offset_mm``. The gripper's
    driven joint then appears in the arm articulation DOFs and the IsaacGripper drives it via
    ``spec.profile``. The caller must have selected the ``"None"`` Gripper variant first. The ``pxr``
    import is lazy, so this module imports without Isaac Sim installed.

    Dropping the nested articulation root takes two mechanisms, because vendor assets are authored
    differently:

    * a ``root_joint`` prim, which the Schunk assets carry: deactivate it;
    * ``ArticulationRootAPI`` applied directly to the gripper's body root with no ``root_joint`` at
      all, as in Isaac's ``Robotiq_2F_85_edit.usd``: set
      ``physxArticulation:articulationEnabled = False``.

    Both are needed. Handling only the first leaves the 2F-85's root live, PhysX sees two articulation
    roots in one subtree and builds neither, and ``SingleArticulation`` then fails with ``Failed to
    find articulation at <arm>/root_joint`` and ``'NoneType' has no attribute 'is_homogeneous'``: a
    failure that names the arm, not the gripper that caused it.

    Disable the root, never ``RemoveAPI`` it: removing the API breaks the 2F-85's mimic joints.
    """
    from isaacsim.core.utils.stage import add_reference_to_stage  # type: ignore[import-not-found]
    from pxr import Gf, PhysxSchema, Usd, UsdPhysics  # type: ignore[import-not-found]

    from src.geometry.quaternion import from_rotation_matrix

    mount_path = f"{arm_prim_path}/mounted_gripper"
    add_reference_to_stage(root + spec.usd_asset_path, mount_path)
    for prim in Usd.PrimRange(stage.GetPrimAtPath(mount_path), Usd.TraverseInstanceProxies()):  # type: ignore[attr-defined]
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI) or prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateArticulationEnabledAttr(False)
    root_joint = stage.GetPrimAtPath(f"{mount_path}/root_joint")  # type: ignore[attr-defined]
    if root_joint and root_joint.IsValid():
        root_joint.SetActive(False)  # drop the root and world-fix so PhysX merges the gripper
    wrist = f"{arm_prim_path}/wrist_3_link"
    base = f"{mount_path}/{spec.base_prim_name}"
    fj = UsdPhysics.FixedJoint.Define(stage, f"{wrist}/mounted_gripper_joint")
    fj.CreateBody0Rel().SetTargets([wrist])
    fj.CreateBody1Rel().SetTargets([base])
    q = from_rotation_matrix(np.asarray(spec.mount_rotation_matrix, dtype=np.float64))  # XYZW
    fj.CreateLocalRot0Attr().Set(Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2])))
    fj.CreateLocalPos0Attr().Set(Gf.Vec3f(*(float(c) / 1000.0 for c in spec.flange_offset_mm)))
    # Firm the driven joint's PD drive so the grip holds the object through the lift: the vendor
    # default is soft enough to drop it. Search the mount subtree for the driven joint by name, which
    # is robust across vendors, and override the DriveAPI gains when the spec sets them.
    if spec.drive_stiffness is not None or spec.drive_damping is not None:
        for jp in Usd.PrimRange(stage.GetPrimAtPath(mount_path)):  # type: ignore[attr-defined]
            if jp.GetName() != spec.profile.driven_joint:
                continue
            for axis in ("linear", "angular"):
                drv = UsdPhysics.DriveAPI.Get(jp, axis)
                if not drv:
                    continue
                if spec.drive_stiffness is not None and drv.GetStiffnessAttr():
                    drv.GetStiffnessAttr().Set(float(spec.drive_stiffness))
                if spec.drive_damping is not None and drv.GetDampingAttr():
                    drv.GetDampingAttr().Set(float(spec.drive_damping))
            break
    # Override the mounted-gripper body masses: a heavy end-effector stalls the RMPflow move.
    if spec.body_mass_kg is not None:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(mount_path)):  # type: ignore[attr-defined]
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(float(spec.body_mass_kg))
    return mount_path



def build_combined_scene(
    session: object,
    sim_cfg: SimConfig,
    *,
    marker_kind: str | None = None,
    aruco_length_mm: float | None = None,
    aruco_dict_name: str | None = None,
    objects_override: list | None = None,
    camera_position_mm: tuple[float, float, float] | None = None,
    appearance: SceneAppearance | None = None,
    bin_walls: list | None = None,
    enable_self_collisions: bool = False,
    real_klt_bin: tuple[float, ...] | None = None,
) -> SceneHandles:
    """Author robot, table, object(s), overhead camera and lighting from the sim config.

    ``sim_cfg`` is ``cfg.robot.sim``, a :class:`SimConfig`: scene geometry comes from
    ``sim_cfg.scene_setup`` (object, table and marker), the overhead camera from
    ``sim_cfg.cameras['overhead']``, the asset root from ``sim_cfg.assets_root`` and the gripper
    variant from ``sim_cfg.gripper_variant``. Returns the overhead camera handle, one (prim path,
    label) pair per authored object, the calibrated ``CAMERA -> BASE`` transform, and the marker prim
    path when a marker was authored.

    ``marker_kind``, one of ``"none"``, ``"flat"`` or ``"aruco"``, overrides
    ``sim_cfg.scene_setup.marker.kind``. The marker is the dedicated hand-eye target: opt-in, and
    never on the pick path. For ``"aruco"`` pass ``aruco_length_mm`` and ``aruco_dict_name``; the
    calibration runner reads them from ``cfg.camera.hand_eye.eye_in_hand``, which is the single
    source of truth shared with the calibrator.
    """
    import carb  # type: ignore[import-not-found]  # noqa: F401
    import omni.usd  # type: ignore[import-not-found]
    from isaacsim.core.api.materials import PhysicsMaterial  # type: ignore[import-not-found]
    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid  # type: ignore[import-not-found]
    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]
    from isaacsim.core.utils.rotations import euler_angles_to_quat  # type: ignore[import-not-found]
    from isaacsim.core.utils.stage import add_reference_to_stage  # type: ignore[import-not-found]
    from isaacsim.sensors.camera import Camera  # type: ignore[import-not-found]
    from isaacsim.storage.native import get_assets_root_path  # type: ignore[import-not-found]
    from pxr import Sdf, UsdLux  # type: ignore[import-not-found]

    _app = appearance if appearance is not None else SceneAppearance()
    world = session.world  # type: ignore[attr-defined]
    scene = sim_cfg.scene_setup
    arm_prim_path = sim_cfg.robot_prim_path or ARM_PRIM
    marker_kind = marker_kind if marker_kind is not None else scene.marker.kind
    root = (sim_cfg.assets_root or get_assets_root_path()).rstrip("/")

    # 1) Combined robot: reference the configured UR model's USD, where robot_model selects the Isaac
    # asset relpath, then select the gripper. The default is the baked Gripper USD variant, a Robotiq
    # 2F-85. When sim_cfg.gripper_mount names a spec, select the "None" variant instead and mount a
    # standalone vendor gripper on the wrist, which merges into the arm articulation.
    from src.robot.drivers.sim.robot_models import ur_model_spec

    _model = ur_model_spec(sim_cfg.robot_model)
    # A model whose USD bakes no gripper, such as ur3e, must name a gripper_mount. Without this check
    # the variant selection below silently no-ops, because SetVariantSelection returns False on a
    # missing variant set, the cell comes up as a bare 6-DoF arm, and the first symptom is an
    # IsaacGripper.connect() failure ("driven joint 'finger_joint' not in gripper dof_names") long
    # after the scene was built.
    if _model.baked_gripper_variant is None and not sim_cfg.gripper_mount:
        raise ValueError(
            f"robot_model={_model.key!r} ships NO baked gripper variant, so sim.gripper_mount must name a "
            f"standalone gripper (e.g. 'robotiq_2f85'; the same 2F-85 the ur5e asset bakes in). "
            f"Set robot.sim.gripper_mount in the config."
        )
    add_reference_to_stage(root + _model.usd_relpath, arm_prim_path)
    stage = omni.usd.get_context().get_stage()
    arm_prim = stage.GetPrimAtPath(arm_prim_path)
    if sim_cfg.gripper_mount:
        from src.willy_sim.grippers import resolve_mounted_gripper

        arm_prim.GetVariantSets().GetVariantSet("Gripper").SetVariantSelection("None")
        stage.Load(Sdf.Path(arm_prim_path))
        _mount_standalone_gripper(stage, arm_prim_path, resolve_mounted_gripper(sim_cfg.gripper_mount), root)
    else:
        arm_prim.GetVariantSets().GetVariantSet("Gripper").SetVariantSelection(sim_cfg.gripper_variant)
        stage.Load(Sdf.Path(arm_prim_path))

    # 2) Table (static) + object (dynamic, graspable).
    world.scene.add(
        FixedCuboid(
            prim_path=TABLE_PRIM, name="table",
            position=_mm_to_m(scene.table.position_mm), scale=_mm_to_m(scene.table.size_mm),
            # table_color overrides the default gray, for a scene that must not look like the default.
            color=np.array(_app.table_color if _app.table_color is not None else (0.4, 0.4, 0.45), dtype=np.float64),
        )
    )
    # Default-off: static bin/tray walls. Each ``bin_walls`` entry is a FixtureBoxConfig-shaped object
    # (``center_mm`` and ``half_extents_mm`` in the robot BASE frame, which is the world frame here).
    # It is the single source for both the physical FixedCuboid prim, so the gripper cannot pass and
    # the depth sees it, and the SafetyPreflight FixtureBoxConfig, so an approach into a wall is
    # rejected as a self-collision. The runner sets the same objects into the guard, so the prim and
    # the guard fixture agree by construction. None means no walls.
    for _wi, _wall in enumerate(bin_walls or []):
        _he = np.asarray(_wall.half_extents_mm, dtype=np.float64)
        world.scene.add(
            FixedCuboid(
                prim_path=f"{BIN_WALL_PRIM}_{_wi}",
                name=f"bin_wall_{_wi}",
                position=_mm_to_m(tuple(float(c) for c in _wall.center_mm)),
                scale=_mm_to_m(tuple(float(2.0 * h) for h in _he)),
                color=np.array((0.30, 0.32, 0.38), dtype=np.float64),
            )
        )
    # Opt-in: reference the real small_KLT.usd as the deep bin, so the demo shows a recognisable
    # crate. Placed static at (cx, cy) with a z offset so its inner floor sits at about the table top
    # (z=0). The prim referenced below is the visual variant and carries no usable collider, so the
    # physics of the bin stays with the fixture-box walls. None means no KLT bin.
    if real_klt_bin is not None:
        from isaacsim.core.utils.stage import add_reference_to_stage  # type: ignore[import-not-found]
        from pxr import Gf, UsdGeom  # type: ignore[import-not-found]
        _rk = tuple(float(v) for v in real_klt_bin)
        _kx, _ky, _kz = _rk[0], _rk[1], _rk[2]
        _kzs = _rk[3] if len(_rk) > 3 else 1.0  # optional 4th = Z-scale (a squashed KLT for a shallow tray look)
        # Use the visual variant, a hollow mesh, for the look. The _visual_collision variant's
        # collider is a convex-hull solid block, so an object spawned inside it is ejected downward;
        # the hollow physics therefore stays with the runner's fixture-box walls, which carry the
        # real KLT dimensions.
        add_reference_to_stage(root + "/Isaac/Props/KLT_Bin/small_KLT_visual.usd", KLT_BIN_PRIM)
        _klt_xf = UsdGeom.Xformable(stage.GetPrimAtPath(KLT_BIN_PRIM))
        _klt_xf.ClearXformOpOrder()
        # opOrder [translate, scale] means translate(scale(p)): scale about the prim origin, then
        # place. The KLT origin is its bounding-box centre, so a Z-scale shrinks toward that centre
        # and the translate then puts the scaled floor at kz.
        _klt_xf.AddTranslateOp().Set(Gf.Vec3d(_kx / 1000.0, _ky / 1000.0, _kz / 1000.0))
        if _kzs != 1.0:
            _klt_xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, float(_kzs)))
        # Only one bin visible: hide the fixture-box wall prims, keeping their colliders, so physics
        # is unaffected and the real KLT is the only bin on screen with no grey walls poking through.
        for _wi in range(len(bin_walls or [])):
            _wp = stage.GetPrimAtPath(f"{BIN_WALL_PRIM}_{_wi}")
            if _wp and _wp.IsValid():
                UsdGeom.Imageable(_wp).MakeInvisible()
    # Effective object list, in order of precedence: an explicit override from a clutter runner, then
    # the config `objects` list, then the single `object` entry. A single object keeps the unsuffixed
    # prim OBJECT_PRIM and the name "object"; more than one gets an index suffix.
    effective_objects = objects_override or (list(scene.objects) if scene.objects else [scene.object])
    single = len(effective_objects) == 1
    grip_material = PhysicsMaterial(
        prim_path="/World/Physics/GripMaterial",
        static_friction=effective_objects[0].static_friction,
        dynamic_friction=effective_objects[0].dynamic_friction, restitution=0.0,
    )
    object_specs: list[tuple[str, str]] = []
    # Counted over the object loop and reported once after it: a dense scene is a dozen-plus objects
    # and a line each would bury the one summary line that matters.
    _material_failures: list[str] = []
    for i, spec in enumerate(effective_objects):
        prim_path = OBJECT_PRIM if single else f"{OBJECT_PRIM}_{i}"
        name = "object" if single else f"object_{i}"
        if spec.usd_asset_path:
            # Mesh branch: reference a YCB USD, where the pre-rigged Axis_Aligned_Physics variants
            # carry collision and rigid-body, and register it with world.scene as a SingleRigidPrim
            # so world.reset() initialises its physics. The cubes get that through DynamicCuboid; a
            # bare add_reference_to_stage leaves the body unmanaged and it free-falls through the
            # table. The object is spawned high and settled further below. YCB assets are
            # root-relative (``/Isaac/...``); an external converted mesh, such as a GSO USD converted
            # outside the Isaac tree, is already a full path on disk and is referenced as it is. The
            # ``isfile`` check that tells them apart cannot match a root-relative ``/Isaac/...``.
            _asset = spec.usd_asset_path if os.path.isfile(spec.usd_asset_path) else root + spec.usd_asset_path
            add_reference_to_stage(_asset, prim_path)
            # A non-physics YCB mesh (Axis_Aligned/) carries no collider and no body, so UsdPhysics is
            # authored here and SingleRigidPrim below finds a rigid body. The helper is idempotent: a
            # no-op for the pre-rigged Axis_Aligned_Physics variants. Runs only when the spec opts in.
            if spec.usd_collision_approximation:
                _author_mesh_collider(
                    stage, prim_path,
                    approximation=spec.usd_collision_approximation, mass_kg=spec.mass_kg,
                )
            _orient = (
                np.asarray(spec.orientation_wxyz, dtype=np.float64)
                if spec.orientation_wxyz is not None else None
            )
            _ycb_prim = SingleRigidPrim(
                prim_path=prim_path, name=name,
                position=_mm_to_m(spec.position_mm), orientation=_orient, mass=spec.mass_kg,
            )
            world.scene.add(_ycb_prim)
            # Give the referenced YCB the same high-friction grip material as the cubes: its authored
            # USD material can be low-friction, and the parallel close then slips on a smooth body.
            try:
                _ycb_prim.apply_physics_material(grip_material)
            except Exception:  # noqa: BLE001 (material binding is best-effort; never fail scene build)
                _material_failures.append(spec.name)
        elif spec.shape == "cylinder":
            # A round body for the 3-finger centric gripper (EZU-35): the 3 fingers wrap it firmly,
            # where a flat cube gives only point and edge contacts and is over-squeezed or slips.
            # size_mm is (diameter, diameter, height); DynamicCylinder takes radius and height in
            # metres.
            from isaacsim.core.api.objects import DynamicCylinder  # type: ignore[import-not-found]
            from pxr import PhysxSchema  # type: ignore[import-not-found]
            world.scene.add(
                DynamicCylinder(
                    prim_path=prim_path, name=name,
                    position=_mm_to_m(spec.position_mm),
                    radius=float(spec.size_mm[0]) / 2000.0, height=float(spec.size_mm[2]) / 1000.0,
                    color=np.array(spec.color, dtype=np.float64),
                    physics_material=grip_material,
                    mass=spec.mass_kg,
                )
            )
            # PhysX cylinder primitives tunnel on fast contact, so CCD keeps the body on the table.
            PhysxSchema.PhysxRigidBodyAPI.Apply(
                stage.GetPrimAtPath(prim_path)  # type: ignore[attr-defined]
            ).CreateEnableCCDAttr(True)
        else:
            world.scene.add(
                DynamicCuboid(
                    prim_path=prim_path, name=name,
                    position=_mm_to_m(spec.position_mm), scale=_mm_to_m(spec.size_mm),
                    color=np.array(spec.color, dtype=np.float64),
                    physics_material=grip_material,
                    mass=spec.mass_kg,
                )
            )
        object_specs.append((prim_path, spec.name))

    if _material_failures:
        # Swallowed so the scene still builds, but not harmless: without the high-friction grip
        # material the parallel close slips on a smooth referenced mesh, and that reads as a grasp
        # quality problem rather than a scene-authoring one.
        _LOG.warning(
            "grip material not applied to %d of %d object(s) (%s); their close may slip",
            len(_material_failures), len(effective_objects), _material_failures,
        )

    # 2b) Optional dedicated calibration marker (clear of the object).
    if marker_kind == "flat":
        world.scene.add(
            FixedCuboid(
                prim_path=MARKER_PRIM, name="calibration_marker",
                position=_mm_to_m(scene.marker.position_mm),
                scale=_mm_to_m(scene.marker.flat_size_mm),
                color=np.array([0.95, 0.95, 0.95]),
            )
        )
    elif marker_kind == "aruco":
        add_aruco_marker(
            stage, pos_m=tuple(_mm_to_m(scene.marker.position_mm)),
            plate_size_m=scene.marker.aruco_plate_size_mm / 1000.0,
            marker_id=scene.marker.aruco_marker_id,
            length_mm=aruco_length_mm if aruco_length_mm is not None else ARUCO_LENGTH_MM,
            dict_name=aruco_dict_name if aruco_dict_name is not None else ARUCO_DICT_NAME,
        )
    elif marker_kind != "none":
        raise ValueError(f"build_combined_scene: unknown marker_kind {marker_kind!r}")

    # 3) Deterministic lighting: a black render starves depth and segmentation. The intensities are
    # overridable through SceneAppearance; None on either keeps the default 150 dome and 300 key.
    UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Lighting/Dome")).CreateIntensityAttr(
        float(_app.dome_intensity) if _app.dome_intensity is not None else 150.0)
    UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Lighting/Key")).CreateIntensityAttr(
        float(_app.key_intensity) if _app.key_intensity is not None else 300.0)

    # 4) Overhead camera looking straight down (USD camera looks down its local -Z).
    overhead = sim_cfg.cameras.get("overhead")
    cam_prim = overhead.prim_path if overhead is not None else CAMERA_PRIM
    # camera_position_mm override: place the camera's nadir over a tall target so the single top-down
    # view sees its top, centred, rather than a side wall; that keeps the view parallax-free and the
    # grasp reachable. None uses the configured position.
    if camera_position_mm is not None:
        cam_pos = _mm_to_m(camera_position_mm)
    elif overhead is not None and overhead.position_mm:
        cam_pos = _mm_to_m(overhead.position_mm)
    else:
        cam_pos = np.array(CAMERA_POS_M)
    cam_res: tuple[int, int] = (
        (int(overhead.resolution[0]), int(overhead.resolution[1]))
        if (overhead is not None and overhead.resolution) else CAMERA_RESOLUTION
    )
    # A tilted eye-to-hand camera, of the kind the real cell has with its D435s off to the side about
    # 70 deg up from the table rather than overhead, is opted into by giving the camera an aim point.
    # The orientation then comes from a geometric look-at instead of the fixed nadir quaternion, the
    # same way author_fixed_camera builds the oblique rigs. No aim point means the straight-down
    # authoring below.
    cam_aim_mm = _vec3(overhead.mount_aim_mm) if (overhead is not None and overhead.mount_aim_mm) else None
    if cam_aim_mm is not None and overhead is not None:
        cam_up = _vec3(overhead.mount_up_hint) if overhead.mount_up_hint else (0.0, 0.0, 1.0)
        qx, qy, qz, qw = (
            float(v) for v in look_at_quat_in_link_xyzw(_vec3(cam_pos * 1000.0), cam_aim_mm, cam_up)
        )
        cam_orientation = np.array([qw, qx, qy, qz], dtype=np.float64)  # Isaac Camera takes WXYZ
    else:
        cam_orientation = euler_angles_to_quat(np.array([0.0, 90.0, 0.0]), degrees=True)
    camera = Camera(
        prim_path=cam_prim, position=cam_pos, orientation=cam_orientation, resolution=cam_res,
    )

    if enable_self_collisions:
        # Opt-in PhysX arm-vs-self collision, so the sim physics refuses what the real arm would.
        # The shipped UR5e asset sets enableSelfCollisions=false, so a self-colliding IK branch
        # silently "lifts" on a physically impossible configuration; turning the attribute on makes
        # such a configuration fail in physics too. Must be set before world.reset(), which is where
        # PhysX picks it up. Left off, the attribute keeps the asset's false.
        from pxr import PhysxSchema  # type: ignore[import-not-found]
        for _cand in (f"{arm_prim_path}/root_joint", arm_prim_path):
            _pr = stage.GetPrimAtPath(_cand)
            if _pr and _pr.IsValid() and _pr.HasAPI(PhysxSchema.PhysxArticulationAPI):
                PhysxSchema.PhysxArticulationAPI(_pr).CreateEnabledSelfCollisionsAttr(True)
                break

    world.reset()  # register the new prims with physics + the articulation
    camera.initialize()
    # The Isaac Camera defaults to a 1.0 m near clip. Only the real-vision path needs a small near
    # clip, for close-range RGB; it sets one itself in IsaacVisionPerceptionSource and parks the arm
    # out of view. The ground-truth path keeps the 1.0 m default on purpose: the clip hides the
    # home-pose arm link that overlaps the object, so the instance-id mask stays clean.

    # The YCB objects were spawned high, registered through SingleRigidPrim above and initialised by
    # world.reset. Step the sim so they drop and settle on the table under gravity, leaving the
    # table-resting pose the runner reads. A scene of cubes has no mesh asset, so this is a no-op.
    if any(s.usd_asset_path for s in effective_objects):
        for _ in range(120):
            world.step(render=False)  # settle the dropped YCB objects on the table under gravity

    # Author the lens from the configured sensor FOV, for instance a D435's 69.4 deg RGB stream, then
    # read the FOV back out of Isaac's own intrinsics: setting aperture attributes is not proof that
    # the renderer agreed. Unset, the camera keeps Isaac's default lens.
    if overhead is not None and overhead.hfov_deg is not None:
        measured_hfov = set_horizontal_fov(camera, overhead.hfov_deg, cam_res)
        print(
            f"[scene] overhead lens: requested HFOV {overhead.hfov_deg:.1f} deg -> "
            f"measured {measured_hfov:.1f} deg",
            flush=True,
        )

    cam_pos_m, _ = camera.get_world_pose()
    if cam_aim_mm is None:
        mat = top_down_camera_to_base_matrix(np.asarray(cam_pos_m) * 1000.0)
        cam_to_base = Transform.from_matrix(mat, from_frame=Frame.CAMERA, to_frame=Frame.BASE)
    else:
        # A tilted camera has no nadir shortcut: top_down_camera_to_base_matrix hardcodes the rotation
        # of a straight-down view, so reusing it here would hand the grasp calculator a confidently
        # wrong extrinsic, with a position that still looks right, which is what makes it dangerous.
        # Fit the true CV-optical extrinsic from Isaac's own projection instead, the same oracle the
        # oblique rigs built by author_fixed_camera use.
        #
        # A close-range near clip is mandatory first: the Isaac default is 1.0 m, and a side camera
        # sits well inside that, so every pixel would come back empty and the fit would raise.
        # Re-author the world xform before anything reads this camera. The Camera ctor was given
        # `orientation=` above, and its orientation convention is not the one a look-at quaternion
        # assumes: a tilted camera authored through the ctor aims into empty space and returns a
        # correctly shaped depth buffer in which no ray hits geometry. author_fixed_camera sets the
        # xform ops directly for exactly this reason. The nadir branch is unaffected because its euler
        # quaternion matches the ctor's convention. Reached only when a camera declares mount_aim_mm.
        import omni.usd  # type: ignore[import-not-found]
        from pxr import Gf, UsdGeom  # type: ignore[import-not-found]

        _cam_prim = omni.usd.get_context().get_stage().GetPrimAtPath(cam_prim)
        _xform = UsdGeom.Xformable(_cam_prim)
        _xform.ClearXformOpOrder()
        for _attr in ("xformOp:translate", "xformOp:orient", "xformOp:scale",
                      "xformOp:rotateXYZ", "xformOp:transform"):
            if _cam_prim.HasAttribute(_attr):
                _cam_prim.RemoveProperty(_attr)
        _pos_mm = _vec3(np.asarray(cam_pos, dtype=np.float64) * 1000.0)
        _q = look_at_quat_in_link_xyzw(_pos_mm, cam_aim_mm, cam_up)
        _xform.AddTranslateOp().Set(Gf.Vec3d(*(float(v) / 1000.0 for v in _pos_mm)))
        _qx, _qy, _qz, _qw = (float(v) for v in _q)
        _xform.AddOrientOp().Set(Gf.Quatf(_qw, Gf.Vec3f(_qx, _qy, _qz)))

        camera.add_distance_to_image_plane_to_frame()
        _near = float(overhead.near_clip_m) if (overhead is not None and overhead.near_clip_m) else 0.05
        # Not swallowed. This clip is load-bearing, not cosmetic: a tilted rig stands about 900 mm
        # from its aim point, inside Isaac's 1.0 m default near plane, so if this call does not take,
        # every pixel comes back empty and the fit raises "camera sees nothing", with no hint that the
        # clip was the cause. A nadir cell does not hit this: its camera is 1200 mm from the table,
        # outside the default plane. Report what the camera actually ended up with.
        try:
            camera.set_clipping_range(_near, 1.0e6)
        except Exception as _clip_exc:  # noqa: BLE001 (report, do not hide)
            print(f"[scene] WARNING set_clipping_range({_near}, 1e6) failed: {_clip_exc}", flush=True)
            # On file too, because the failure this clip causes, "camera sees nothing", surfaces
            # further down with no hint that the clip was the cause.
            _LOG.warning("set_clipping_range(%s, 1e6) failed: %r", _near, _clip_exc)
        try:
            print(f"[scene] tilted camera clipping range now {camera.get_clipping_range()} "
                  f"(requested near {_near} m)", flush=True)
        except Exception:  # noqa: BLE001 (getter is informational only)
            pass
        _app_handle = getattr(session, "app", None)

        def _pump(n: int) -> None:
            for _ in range(n):
                session.step(render=True)  # type: ignore[attr-defined]
                if _app_handle is not None:
                    _app_handle.update()

        # Pump and retry, not a single warmup burst. On a tilted rig such as
        # `WILLY_PROFILE=sim,ur3e,tiltcam` the depth annotator is not populated after one burst, and a
        # single attempt dies with "<3 valid depth pixels (camera sees nothing)". run_eth_calibrate
        # retries the same call the same way. A nadir cell does not reach this: its first burst
        # succeeds.
        _fitted: "Transform | None" = None
        _fit_rmse = 0.0
        for _attempt in range(6):
            _pump(max(1, scene.render_warmup_steps) if _attempt == 0 else 12)
            try:
                _fitted, _fit_rmse = camera_to_base_ground_truth(camera)
                break
            except Exception as _exc:  # noqa: BLE001 (depth not ready yet; pump + retry)
                if _attempt == 5:
                    raise  # no log line of its own: the traceback is the record
                print(f"[scene] tilted-camera oracle retry {_attempt + 1}/5 ({_exc})", flush=True)
                _LOG.warning(
                    "tilted-camera CAMERA->BASE oracle retry %d/5: %r", _attempt + 1, _exc,
                )
        if _fitted is None:  # the loop either sets it or re-raises; this narrows the type
            raise RuntimeError("tilted-camera CAMERA->BASE fit did not converge")
        cam_to_base = _fitted
        print(
            f"[scene] overhead camera TILTED: elevation "
            f"{camera_elevation_deg(_vec3(np.asarray(cam_pos_m) * 1000.0), cam_aim_mm):.1f} deg above the "
            f"table, aim {cam_aim_mm} -> CAMERA->BASE fit from Isaac projection (rmse {_fit_rmse:.2f} mm)",
            flush=True,
        )

    # The one line that records which scene a later result belongs to. camera_to_base is called out
    # explicitly because a scene whose extrinsic is missing still picks, in the CAMERA frame and with
    # no table check, and nothing downstream announces that.
    _LOG.info(
        "scene built: robot=%s(%s) gripper=%s objects=%d %s walls=%d klt=%s marker=%s "
        "camera=%s res=%dx%d pos=%s mm camera_to_base=%s",
        sim_cfg.robot_model, arm_prim_path,
        sim_cfg.gripper_mount or f"baked:{sim_cfg.gripper_variant}",
        len(object_specs), [label for _, label in object_specs], len(bin_walls or []),
        real_klt_bin is not None, marker_kind, cam_prim, cam_res[0], cam_res[1],
        np.round(np.asarray(cam_pos, dtype=np.float64) * 1000.0, 1).tolist(),
        "SET" if cam_to_base is not None else "NONE",
    )
    return SceneHandles(
        camera=camera, object_specs=tuple(object_specs), camera_to_base=cam_to_base,
        marker_prim_path=MARKER_PRIM if marker_kind != "none" else None,
    )
