"""The Isaac half of the physics sample: a world-fixed 2F-85, a rebuilt scene, and a grip that holds.

On box only. Everything Isaac-shaped is imported inside methods, so importing this module on a
laptop costs nothing and fails nothing.

The harness passes all four of its controls on a lone object. Teleporting a gripper into a full bin
creates overlaps that destabilise the articulation, so a large share of cluttered-scene trials is
refused rather than scored. Refusing is the point, but a run of refusals is not a sample.

Everything about the gripper is measured at start-up rather than written down, because a constant
recorded here that is wrong invalidates every trial silently instead of failing:

* Which way the jaw runs. 0 rad is 85 mm open and 0.8 rad is shut. The gap between the two
  ``inner_finger`` body origins moves opposite to the pads, so a jaw belief taken from it is
  inverted and every trial approaches too narrow and then opens to "close".
* How wide the jaw is at a given angle. Read off the collision shapes themselves with PhysX overlap
  queries (:meth:`PhysicsCell._calibrate_aperture`), because the body origins sit ~6 mm inboard of
  the pad faces.
* Where the grasp centre is. From the pads' bounds in the base frame, not from the link origins,
  which all sit near the base.

The cell itself is arranged around three facts about Isaac 5.1: a floating-base articulation has no
collision with dynamic bodies (hence the world fixed joint), ``World.reset()`` restores every body
to its state at the last play (hence exactly one reset per session), and a teleported body
transmits no tangential force (hence the hold test removes the table instead of lifting the
gripper).

The gripper is teleported to the grasp pose rather than carried there by an arm, so no outcome
depends on whether a UR5e could have reached it. That is deliberate, it is also the source of the
remaining instability, and reachability is a separate question asked separately.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from src.utility.log_cfg import create_logger

from datagen.constants import DATAGEN_LOG_DIR, PHYSICS_CELL_LOG_FILE
from datagen.grasps.physics import (
    _MM_TO_M,
    _STANDOFF_MM,
    _STEPS_APPROACH,
    _STEPS_CLOSE,
    PhysicsTrial,
    TrialOutcome,
    _grasp_frame,
)
from datagen.grasps.shapes import Solid, solid_from_scene

__all__ = ["PhysicsCell"]

#: Its own file, separate from the sampler's. When a hold rate looks wrong the first question is
#: whether the cell was healthy, meaning the calibration it measured and the scenes it built, and
#: that answer should not have to be picked out from between the trials it is used to judge.
logger = create_logger("datagen.grasps.physics_cell", PHYSICS_CELL_LOG_FILE,
                       log_dir=DATAGEN_LOG_DIR)

_GRIPPER_USD = "/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd"
_GRIPPER_ROOT = "/World/FloatingGripper"
_OBJECT_ROOT = "/World/PhysicsObjects"
#: The 2F-85's driven joint and the range it is driven over. Which end is open is not written down
#: here: the separation between the two ``inner_finger`` body origins runs the opposite way to the
#: jaw, so a constant taken from it inverts the sense. Measured against the collision shapes
#: themselves (see :meth:`_calibrate_aperture`), 0 rad is ~85 mm open and 0.8 rad is ~2 mm shut.
#: Open and shut are read off that curve at start-up; only the driven interval is a constant.
_DRIVEN_JOINT = "finger_joint"
_JOINT_LIMIT_RAD = (0.05, 0.80)
#: Jaw drive gains, firmer than the asset's vendor default. `willy_sim`'s `grippers.py` firms the
#: EGU-50 the same way, at 12000/120; see the note in _build_stage for what the soft default does.
_DRIVE_STIFFNESS = 2000.0
_DRIVE_DAMPING = 60.0
#: Contact friction for every collider in the cell. Stated, not inherited: see the note in _build_stage.
_FRICTION_MATERIAL = "/World/GraspFriction"
_FRICTION_MU = 0.5
#: Where the gripper waits while a scene is built. Two metres up is clear of any cell this generates.
_PARK_POSITION_MM = (0.0, 0.0, 2000.0)
#: How far inside the object's own width the jaw is commanded, so the drive presses rather than crushes.
_SQUEEZE_MM = 6.0
#: Where a finished scene's bodies go: a wide grid far from the cell, one slot each, parked at
#: z = 200 mm. Every one of these is load-bearing. Far from the cell, so nothing retired can touch
#: the scene under test. Not all at one point, because coincident bodies are a depenetration
#: explosion. They must be moved, never deleted or deactivated, because either invalidates the PhysX
#: tensor view. And there is no ground plane under the cell (see _build_stage), so nothing catches
#: them: `_retire_previous` switches their gravity off as well, because a body in permanent free
#: fall accumulates velocity for the rest of the session and eventually returns NaN.
_RETIRE_ORIGIN_MM = (6000.0, 6000.0)
_RETIRE_PITCH_MM = 400.0
_RETIRE_COLUMNS = 40
#: How far the target may have drifted from its labelled pose before a trial stops being a measurement
#: of the grasp and becomes a measurement of the harness. Refused, not scored.
_DRIFT_REFUSE_MM = 5.0
#: Steps allowed for a restored scene to come to rest before it is checked.
_STEPS_RESTORE = 12
#: How far the gripper may end up from the pose it was commanded to hold. It is teleported every
#: physics step, so any offset at all means physics won the argument, and a gripper that was pushed
#: out of the way did not measure the grasp.
_GRIPPER_OFFSET_REFUSE_MM = 5.0
#: The table the scenes rest on: a kinematic slab whose top is the labels' z = 0. It is taken away to
#: test the grip (see run_trial), which is why there is no ground plane under it.
_TABLE_PATH = "/World/Table"
_TABLE_EXTENT_MM = (4000.0, 4000.0, 400.0)
_TABLE_RETIRE_Z_MM = -8000.0
#: The hold test. With the table gone the object hangs from the jaw or it does not, and the two cases
#: are not close: a held object cannot fall further than the pads are long, while an unheld one is
#: gone by two orders of magnitude. Every drop is written to the row so that separation stays
#: checkable against data instead of resting on this comment.
_HELD_DROP_MM = 100.0
_STEPS_HANG = 120
#: Depenetration speed limit, in m/s, on every body in the cell. The gripper is teleported to its
#: grasp pose through whatever is in the way, so deep overlaps with a neighbour are routine rather
#: than exceptional. Unlimited depenetration turns one of them into an exploding articulation whose
#: driven joint leaves its range and then reads NaN, after which every later trial reports "did not
#: hold" from a simulation that has already died.
_MAX_DEPENETRATION_MPS = 1.0
_SOLVER_POSITION_ITERATIONS = 32
_SOLVER_VELOCITY_ITERATIONS = 4


@dataclass
class _GripperFrame:
    """What the gripper actually is, measured once."""

    closing_axis_local: np.ndarray      # unit, in base_link frame
    approach_local: np.ndarray          # unit, in base_link frame
    tcp_offset_mm: float                # base_link origin -> grasp centre, along approach_local
    opening_at_open_mm: float
    #: (joint angle, aperture mm) measured this session against the pad faces themselves. This is
    #: the curve the trials use. See _calibrate_aperture for why it is not derived from geometry.
    travel: tuple[tuple[float, float], ...] = ()
    #: (joint angle, body-origin separation mm). Kept for provenance only: it is not the aperture and
    #: reads ~6 mm narrow, because the pad face is outboard of its body origin.
    origin_travel: tuple[tuple[float, float], ...] = ()
    #: Read off the measured curve, never assumed. For this asset angle_open is ~0.05 rad and
    #: angle_shut is ~0.80.
    angle_open: float = 0.0
    angle_shut: float = 0.0
    finger_paths: tuple[str, str] = ("", "")


class PhysicsCell:
    """One Isaac session: build scenes, replay grasps, report whether they held."""

    def __init__(self, *, headless: bool = True, mesh_collision: str = "sdf") -> None:
        self._headless = headless
        #: How PhysX may approximate a scanned mesh. Carried here rather than defaulted at the
        #: authoring site because the shake must collide the same approximation the render run used:
        #: a label produced against an SDF and shaken against a convex hull says nothing about that
        #: label. See `AssetSourcesConfig.mesh_collision`.

        self._mesh_collision = str(mesh_collision)
        self._app: Any = None
        self._world: Any = None
        self._gripper: Any = None
        self._frame: _GripperFrame | None = None
        self._joint_index: int = 0
        self._rest_dofs: np.ndarray | None = None
        self._objects: Any = None
        self._object_ids: list[int] = []
        self._build_index: int = 0
        self._settle_drift_mm: float = 0.0
        #: Every physics view of the current build, objects and walls alike. Retiring walks this list;
        #: anything not in it cannot be moved out of the way and would haunt every later scene.
        self._live: list[Any] = []
        self._retire_slot: int = 0
        #: The pose the gripper has been told to be at, re-applied on every physics step. See _step.
        self._commanded: tuple[np.ndarray, np.ndarray] | None = None
        self._gbase: Any = None
        self._table: Any = None

    # ------------------------------------------------------------ lifecycle

    def __enter__(self) -> PhysicsCell:
        from isaacsim import SimulationApp  # type: ignore[import-not-found]  # noqa: PLC0415

        logger.info("booting Isaac for the physics cell (headless=%s)", self._headless)
        self._app = SimulationApp({"headless": self._headless})
        self._build_stage()
        logger.info("physics cell ready")
        return self

    def __exit__(self, exc_type, exc, tb) -> Literal[False]:
        # Isaac's close() ends in shutdown_and_release_framework(), which terminates the process, so
        # an exception still unwinding here would never be printed. Print it first.
        if exc is not None:
            traceback.print_exception(exc_type, exc, tb)
            # The same reason the print exists: close() terminates the process, so this is the last
            # moment anything can be recorded at all. The traceback goes to the console, the one-line
            # cause goes to the file that outlives the run.
            logger.error("physics cell failed: %s: %s", exc_type.__name__ if exc_type else "?", exc)
        if self._app is not None:
            logger.info("closing the physics cell")
            self._app.close()
        return False

    def _build_stage(self) -> None:
        from isaacsim.core.api import World  # type: ignore[import-not-found]  # noqa: PLC0415
        from isaacsim.core.utils.stage import (  # type: ignore[import-not-found]  # noqa: PLC0415
            add_reference_to_stage,
        )
        from isaacsim.storage.native import get_assets_root_path  # type: ignore[import-not-found]  # noqa: PLC0415, E501
        from pxr import (  # type: ignore[import-not-found]  # noqa: PLC0415
            Gf,
            PhysxSchema,
            Sdf,
            Usd,
            UsdGeom,
            UsdPhysics,
            UsdShade,
        )

        self._world = World(stage_units_in_meters=1.0)
        stage = self._world.stage
        UsdGeom.Xform.Define(stage, _OBJECT_ROOT)
        # No ground plane. The scenes rest on a kinematic table slab whose top is the labels' z = 0,
        # and the space below it is empty on purpose: the hold test works by taking the table away (see
        # run_trial). An infinite ground plane would catch everything it drops and there would be
        # nothing to measure.

        add_reference_to_stage(get_assets_root_path() + _GRIPPER_USD, _GRIPPER_ROOT)
        # Un-instance the gripper before anything plays. The 2F-85 asset marks each link's
        # ``visuals`` scope instanceable, so every one of its collision meshes lives inside an
        # instance proxy and PhysX creates no collision shapes for them: the gripper then has no
        # collision with dynamic bodies at all, and a block dropped onto it falls straight through.
        # Every collider still reports ``collisionEnabled=True``, so reading the stage agrees with
        # itself while nothing can touch anything.
        #
        # It has to happen here, before the first ``World.reset()``: PhysX parses colliders when the
        # scene starts, and un-instancing afterwards does not create the shapes retroactively.
        instanceable = [prim for prim in Usd.PrimRange(stage.GetPrimAtPath(_GRIPPER_ROOT))
                        if prim.IsInstanceable()]
        for prim in instanceable:
            prim.SetInstanceable(False)
        # A floating base would fall. Gravity off for the gripper only: it is teleported, so its own
        # dynamics are never the thing under test, the grip is.
        #
        # The bodies are found through ``UsdPhysics.RigidBodyAPI``. The shipped 2F-85 asset does not
        # carry the PhysX extension schema ``PhysxSchema.PhysxRigidBodyAPI``, so a loop keyed on that
        # matches nothing, gravity is never disabled, and the gripper falls out of its 2000 mm park
        # and ends up on the floor.
        bodies = [prim for prim in Usd.PrimRange(stage.GetPrimAtPath(_GRIPPER_ROOT),
                                                 Usd.TraverseInstanceProxies())
                  if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        if not bodies:
            raise RuntimeError(
                "no rigid bodies found under the gripper, so gravity cannot be disabled on it; "
                "a falling gripper measures nothing"
            )
        for prim in bodies:
            physx = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            physx.CreateDisableGravityAttr(True)
            physx.CreateMaxDepenetrationVelocityAttr(_MAX_DEPENETRATION_MPS)
            physx.CreateSolverPositionIterationCountAttr(_SOLVER_POSITION_ITERATIONS)
            physx.CreateSolverVelocityIterationCountAttr(_SOLVER_VELOCITY_ITERATIONS)

        # Pin the base to the world. Without this the harness measures nothing at all: a
        # floating-base 2F-85 articulation generates no contacts with dynamic bodies in Isaac 5.1,
        # so a cube released above the parked gripper falls straight through it to the floor.
        # `willy_sim` mounts the same gripper on the arm, whose articulation is world-fixed, which is
        # why its picks grip. ``set_world_pose`` still moves a fixed-base articulation, so nothing
        # about teleporting the gripper changes.
        fixed = UsdPhysics.FixedJoint.Define(stage, f"{_GRIPPER_ROOT}/world_fix")
        fixed.CreateBody1Rel().SetTargets([f"{_GRIPPER_ROOT}/Robotiq_2F_85/base_link"])

        # Firm the jaw drive. The vendor default is too soft to carry a 100 g object: the jaw does
        # stall on the object, and then the load pushes it back open and the object slides out while
        # the pads are still touching it. This is a property of the asset, not a tuning knob for the
        # result, since a jaw that cannot hold what it has gripped reports every good grasp as a
        # failure.
        drives = 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath(_GRIPPER_ROOT)):
            if prim.GetName() != _DRIVEN_JOINT:
                continue
            for axis in ("linear", "angular"):
                drive = UsdPhysics.DriveAPI.Get(prim, axis)
                if drive and drive.GetStiffnessAttr():
                    drive.GetStiffnessAttr().Set(_DRIVE_STIFFNESS)
                    drive.GetDampingAttr().Set(_DRIVE_DAMPING)
                    drives += 1
        if not drives:
            raise RuntimeError(
                f"no drive found on {_DRIVEN_JOINT!r}; the jaw cannot be firmed and the vendor default "
                "was measured to let a gripped block slide back out during the lift"
            )

        # A friction coefficient, chosen once and stated. Neither the vendor gripper's colliders nor
        # the cubes this module authors carry a physics material, so without this the contact runs on
        # whatever PhysX defaults to, an unstated number every hold rate would depend on. The value
        # is the same mu the analytic verdict uses for its friction cone
        # (`JawModel.friction_coefficient` = 0.5), because the physics sample exists to judge that
        # verdict: giving the simulation a grippier world than the reference assumes would flatter it.
        material = UsdPhysics.MaterialAPI.Apply(
            UsdShade.Material.Define(stage, _FRICTION_MATERIAL).GetPrim())
        material.CreateStaticFrictionAttr(_FRICTION_MU)
        material.CreateDynamicFrictionAttr(_FRICTION_MU)
        material.CreateRestitutionAttr(0.0)
        for prim in Usd.PrimRange(stage.GetPrimAtPath(_GRIPPER_ROOT), Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                _bind_friction(stage, prim, UsdShade=UsdShade)

        from isaacsim.core.prims import SingleArticulation  # type: ignore[import-not-found]  # noqa: PLC0415, E501

        self._gripper = SingleArticulation(prim_path=_GRIPPER_ROOT, name="floating_gripper")
        self._world.reset()
        self._gripper.initialize()
        names = list(self._gripper.dof_names or ())
        if _DRIVEN_JOINT not in names:
            raise RuntimeError(f"{_DRIVEN_JOINT!r} not among the gripper's DOFs: {names}")
        self._joint_index = names.index(_DRIVEN_JOINT)
        # The articulation's rest state, captured once while it is known good. `restore()` writes it
        # back between trials, and it is captured rather than constructed because this gripper's other
        # DOFs are mimic joints: a hand-written "open" array would be a guess about linkage the USD
        # already answers.
        self._rest_dofs = np.asarray(self._gripper.get_joint_positions(), dtype=np.float64).copy()
        # The park has to be the default state, not just a pose that was set: ``World.reset()``
        # restores every registered object to its default, so parking before a reset is silently
        # undone by it and the gripper materialises on top of the scene at every build.

        self._gripper.set_default_state(position=np.array(_PARK_POSITION_MM) * _MM_TO_M,
                                        orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        self._park()
        table = solid_from_scene("box", _TABLE_EXTENT_MM,
                                 (0.0, 0.0, -_TABLE_EXTENT_MM[2] / 2.0), (0.0, 0.0, 0.0, 1.0),
                                 instance_id=0, asset_id="table", mass_kg=1000.0)
        _author_solid(stage, _TABLE_PATH, table, kinematic=True, Gf=Gf, Sdf=Sdf, UsdGeom=UsdGeom,
                      UsdPhysics=UsdPhysics, UsdShade=UsdShade, PhysxSchema=PhysxSchema)
        from isaacsim.core.prims import RigidPrim  # noqa: PLC0415

        self._table = RigidPrim(prim_paths_expr=_TABLE_PATH, name="table")
        self._table.initialize()
        self._frame = self._calibrate()

    # ---------------------------------------------------------- calibration

    def _finger_body_paths(self) -> tuple[str, str]:
        """The two pad bodies, found by name rather than by position in a list.

        Isaac's ``Robotiq_2F_85_edit.usd`` names its bodies ``base_link``,
        ``{left,right}_outer_knuckle``, ``_outer_finger``, ``_inner_finger``, ``_inner_knuckle``. The
        rubber pad rides on ``inner_finger``, so that is the body whose motion defines the closing
        axis. The other spellings stay as fallbacks for a differently-named asset; the search tries
        them in order and a failure names every body it saw.
        """
        from pxr import Usd, UsdPhysics  # type: ignore[import-not-found]  # noqa: PLC0415

        stage = self._world.stage
        bodies = [
            (prim.GetName().lower(), str(prim.GetPath()))
            for prim in Usd.PrimRange(stage.GetPrimAtPath(_GRIPPER_ROOT),
                                      Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        for token in ("inner_finger", "pad", "finger_tip", "fingertip"):
            left = next((path for name, path in bodies if token in name and "left" in name), None)
            right = next((path for name, path in bodies if token in name and "right" in name), None)
            if left and right:
                return (left, right)
        raise RuntimeError(
            "could not find two pad bodies on the 2F-85; the grasp frame cannot be measured and "
            f"guessing it would rotate every trial. Bodies seen: {[n for n, _ in bodies]}"
        )

    def _calibrate(self) -> _GripperFrame:
        """Drive the joint, watch the pads, and read the finger geometry. Nothing here is assumed.

        Three properties of this asset, each of which silently invalidates every trial when it is
        assumed instead:

        * USD xforms of articulation links lag. Read through ``get_world_pose`` right after a drive,
          every link of the 2F-85 reports the same transform, which measures a pad separation of
          exactly 0 mm and makes the harness refuse itself. The physics view (``RigidPrim``) reports
          the truth; USD does not, until a render tick writes it back.
        * The body-origin separation runs the opposite way to the jaw: it grows with the joint angle
          while the aperture closes. It fixes the axes and the TCP, not the opening, which
          :meth:`_calibrate_aperture` measures by contact.
        * Link origins are not pads. The 2F-85's link frames all sit near the base, so their midpoint
          reads about 13 mm from base_link where the real grasp centre is ~133 mm. The pads'
          geometric bounds give it: the inner fingers span z = 102.3 to 164.6 mm in the base frame,
          whose centre corroborates the 132.0 mm grasp-centre offset of the baked mount.
        """
        from isaacsim.core.prims import RigidPrim  # type: ignore[import-not-found]  # noqa: PLC0415

        left, right = self._finger_body_paths()
        pads = RigidPrim(prim_paths_expr=f"{_GRIPPER_ROOT}/Robotiq_2F_85/*_inner_finger", name="pads")
        base = RigidPrim(prim_paths_expr=f"{_GRIPPER_ROOT}/Robotiq_2F_85/base_link", name="gbase")
        pads.initialize()
        base.initialize()
        # Kept: every step re-commands the pose and zeroes this body's velocity, and the guard in
        # gripper_offset_mm reads it back. Both need the view to outlive calibration.
        self._gbase = base
        self._park()
        self._step(10)
        residual = self.gripper_offset_mm()
        if residual > 1.0:
            raise RuntimeError(
                f"base_link sits {residual:.1f} mm from the articulation root this class commands, so "
                "the gripper-position guard would measure that offset instead of a fault"
            )

        travel: list[tuple[float, float]] = []
        closing_world = np.array([0.0, 1.0, 0.0])
        rotation = np.eye(3)
        base_position = np.zeros(3)
        for angle in (0.2, 0.4, 0.6, _JOINT_LIMIT_RAD[1]):
            self._drive_joint(angle, steps=60)
            positions, _ = pads.get_world_poses()
            p = np.asarray(positions, dtype=np.float64).reshape(2, 3) / _MM_TO_M
            delta = p[0] - p[1]
            separation = float(np.linalg.norm(delta))
            travel.append((angle, separation))
            if angle == _JOINT_LIMIT_RAD[1]:
                position, quat = base.get_world_poses()
                base_position = np.asarray(position, dtype=np.float64).reshape(3) / _MM_TO_M
                rotation = _quat_wxyz_to_matrix(np.asarray(quat).reshape(4))
                closing_world = delta / max(separation, 1e-9)

        closing_local = rotation.T @ closing_world
        low, high = self._pad_bounds_in_base(rotation, base_position)
        # The approach axis is the one the pads are farthest along from the base, not the one they
        # are widest on. The widest axis is the closing direction, because the two pads together span
        # the full 85 mm opening there, and taking it returns a TCP offset of 0, which places the
        # gripper body inside the object.
        centre = (high + low) / 2.0
        index = int(np.argmax(np.abs(centre) * (1.0 - np.abs(closing_local))))
        approach_local = np.zeros(3)
        approach_local[index] = float(np.sign(centre[index])) or 1.0

        frame = _GripperFrame(
            closing_axis_local=closing_local,
            approach_local=approach_local,
            tcp_offset_mm=abs(float(centre[index])),
            opening_at_open_mm=travel[-1][1],
            origin_travel=tuple(travel),
            finger_paths=(left, right),
        )
        # The body-origin sweep above fixes the axes and the TCP correctly. It does not give the
        # aperture: see _calibrate_aperture, which measures that by contact. _place needs the frame
        # to exist before it can be called, hence the two stages.
        self._frame = frame
        frame.travel = self._calibrate_aperture()
        frame.opening_at_open_mm = frame.travel[-1][1]      # aperture-ascending: last is widest
        frame.angle_open = frame.travel[-1][0]
        frame.angle_shut = frame.travel[0][0]
        # The measurement everything downstream is expressed in. Nothing here is assumed, so the
        # values this session actually read belong in the record beside the hold rates they produced.
        logger.info("gripper calibrated: TCP %.1f mm from base, aperture %.1f mm at %.3f rad open / "
                    "%.3f rad shut, closing axis %s, approach %s",
                    frame.tcp_offset_mm, frame.opening_at_open_mm, frame.angle_open,
                    frame.angle_shut,
                    np.round(frame.closing_axis_local, 3).tolist(),
                    np.round(frame.approach_local, 3).tolist())
        return frame

    def _pad_bounds_in_base(self, rotation: np.ndarray, base_position: np.ndarray):
        """The inner fingers' extent expressed in the base frame: where the pads actually are."""
        from pxr import Usd, UsdGeom  # type: ignore[import-not-found]  # noqa: PLC0415

        self._world.step(render=True)      # force the write-back the bounds cache reads
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                  [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        corners: list[np.ndarray] = []
        for path in self._finger_body_paths():
            prim = self._world.stage.GetPrimAtPath(path)
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            lo = np.asarray([box.GetMin()[i] for i in range(3)]) / _MM_TO_M
            hi = np.asarray([box.GetMax()[i] for i in range(3)]) / _MM_TO_M
            corners.extend(np.array([a, b, c])
                           for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2]))
        local = (np.asarray(corners) - base_position) @ rotation
        return local.min(axis=0), local.max(axis=0)

    # ------------------------------------------------------------- driving

    def _drive_joint(self, radians: float, *, steps: int) -> None:
        from isaacsim.core.utils.types import ArticulationAction  # type: ignore[import-not-found]  # noqa: PLC0415, E501

        target = np.full(len(self._gripper.dof_names), np.nan)
        target[self._joint_index] = float(radians)
        self._gripper.apply_action(ArticulationAction(joint_positions=target))
        self._step(steps)

    def _step(self, count: int) -> None:
        """Advance physics while holding the gripper on its commanded pose.

        Every stepping loop in this class goes through here. The gripper is a free-floating
        articulation, so closing its jaw on an object produces a reaction force on the base: a close
        that steps without re-commanding the pose walks the gripper away and shoots the block out
        instead of gripping it. Re-teleporting every step is what makes "the gripper is teleported,
        not carried" true for the whole trial rather than for one loop.
        """
        for _ in range(count):
            self._hold_pose()
            self._world.step(render=False)

    def _hold_pose(self) -> None:
        """Re-apply the last commanded pose and kill any momentum the contact gave the gripper."""
        if self._commanded is None or self._gripper is None:
            return
        position, orientation = self._commanded
        self._gripper.set_world_pose(position=position, orientation=orientation)
        if self._gbase is not None:
            self._gbase.set_velocities(np.zeros((1, 6)))

    def _joint_for_opening(self, opening_mm: float) -> float:
        """Interpolate the angle from the aperture curve measured this session.

        The curve is stored aperture-ascending, which for this joint means angle-descending: 0.05 rad
        is 82.5 mm open and 0.80 rad is 2.0 mm shut. Interpolating rather than fitting a line matters
        because the curve is not quite straight, and a jaw that arrives a few millimetres too narrow
        pushes the object away instead of surrounding it.
        """
        if not self._frame or not self._frame.travel:
            return _JOINT_LIMIT_RAD[0]
        widths = np.array([w for _, w in self._frame.travel])
        angles = np.array([a for a, _ in self._frame.travel])
        low, high = min(_JOINT_LIMIT_RAD), max(_JOINT_LIMIT_RAD)
        return float(np.clip(np.interp(opening_mm, widths, angles), low, high))

    def _opening_for_joint(self, angle: float) -> float:
        """The inverse of _joint_for_opening, for reporting what the jaw actually reached."""
        if not self._frame or not self._frame.travel:
            return float("nan")
        pairs = sorted(self._frame.travel)                       # angle-ascending
        return float(np.interp(angle, [a for a, _ in pairs], [w for _, w in pairs]))

    def _park(self) -> None:
        """Move the gripper somewhere it cannot touch anything. Called before every scene change."""
        self._commanded = (np.array(_PARK_POSITION_MM) * _MM_TO_M,
                           np.array([1.0, 0.0, 0.0, 0.0]))
        self._hold_pose()

    def _reset_jaw(self) -> None:
        """Teleport the jaw back to its rest DOFs. The cure for a blown-up articulation.

        A trial that explodes leaves the driven joint far outside its range, and `_park()` moves only
        the base, so every later trial inherits it and the sample is poisoned from there on.

        It has to be a teleport, not a command. A position drive told to travel back from 1e9 rad
        does not converge, it generates force: the joint decays slowly while the drive pulls at it
        and never gets back into range.
        """
        if self._gripper is None or self._rest_dofs is None:
            return
        self._gripper.set_joint_velocities(np.zeros_like(self._rest_dofs))
        self._gripper.set_joint_positions(self._rest_dofs.copy())
        # Without this the next `apply_action` re-asserts the pre-explosion target the moment physics
        # steps, and the teleport is undone before anything can observe it.
        self._drive_joint(float(self._rest_dofs[self._joint_index]), steps=2)

    def jaw_fault(self) -> str:
        """The reason the jaw is unusable, or an empty string when there is none.

        The return is a reason, not a predicate: `""` is the value meaning everything is fine and it
        is falsy, so a call site reads `problem = cell.jaw_fault() or ...` rather than branching on
        truth.

        Read before a trial as well as after. Checked only after the close, an inherited explosion is
        indistinguishable from a grasp that failed on its own merits.
        """
        if self._gripper is None:
            return ""
        reached = float(self._gripper.get_joint_positions()[self._joint_index])
        if not np.isfinite(reached) or not -1.0 <= reached <= 2.0:
            return f"the jaw joint blew up ({reached:.3g} rad)"
        return ""

    def _place(self, grasp_centre_mm: np.ndarray, approach: np.ndarray, closing: np.ndarray) -> None:
        """Put the gripper so that its measured grasp centre lands on ``grasp_centre_mm``."""
        assert self._frame is not None
        target = _grasp_frame(approach, closing)                       # columns: closing, binormal, approach
        local = _grasp_frame(self._frame.approach_local, self._frame.closing_axis_local)
        rotation = target @ local.T
        base = grasp_centre_mm - rotation @ (self._frame.approach_local * self._frame.tcp_offset_mm)
        self._commanded = (np.asarray(base, dtype=np.float64) * _MM_TO_M,
                           _matrix_to_quat_wxyz(rotation))
        self._hold_pose()

    def gripper_offset_mm(self) -> float:
        """How far the gripper actually is from where it was told to be. The guard for a floating base.

        A gripper that has been shoved out of position by the object it was supposed to grip will
        happily report "nothing rose". This turns that into a refusal instead of a data point.
        """
        if self._commanded is None or self._gbase is None:
            return 0.0
        position, _ = self._gbase.get_world_poses()
        actual = np.asarray(position, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(actual)):
            return float("inf")
        return float(np.linalg.norm(actual - self._commanded[0])) / _MM_TO_M

    # -------------------------------------------------------------- scenes

    #: Where the aperture is measured: high above the cell, so nothing else is in the probe's way.
    _APERTURE_PROBE_MM = (0.0, 0.0, 1500.0)

    def _calibrate_aperture(self) -> tuple[tuple[float, float], ...]:
        """The curve from joint angle to aperture, found by asking PhysX where the pad faces are.

        The distance between the two ``inner_finger`` body origins is not the aperture: the pad's
        gripping face sits outboard of its body origin, so the origins read about 6 mm narrow while
        agreeing with the 2F-85's nominal 85 mm at full open, which makes the wrong number look like
        a confirmation. A jaw calibrated from it sits outside the object it is supposed to squeeze,
        reaches its commanded angle without resistance, and lifts nothing.

        Two other routes are rejected, each for a measured reason. Closing on a kinematic block of
        known width blows the solver up, because a position drive pressing into infinite mass
        returns joint angles of NaN and -7e34. Reading the pads' USD bounds does not track the joint
        at all, the same lag this module documents for link transforms. Scanning a thin probe box
        outward from the grasp centre and asking the physics engine which shape it first touches uses
        the simulation's own geometry, needs no dynamics, and is exact to the scan step.
        """
        import carb  # noqa: PLC0415
        from omni.physx import get_physx_scene_query_interface  # type: ignore[import-not-found]  # noqa: PLC0415, E501

        query = get_physx_scene_query_interface()
        centre = np.array(self._APERTURE_PROBE_MM, dtype=np.float64)
        approach, closing = np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0])
        hit_pad = False

        def report(hit) -> bool:  # noqa: ANN001
            nonlocal hit_pad
            hit_pad = hit_pad or "inner_finger" in str(hit.collision)
            return True

        curve: list[tuple[float, float]] = []
        for angle in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, _JOINT_LIMIT_RAD[1]):
            self._place(centre, approach, closing)
            self._drive_joint(angle, steps=60)
            face = None
            for step in range(1, 241):                       # 0.25 mm steps out to 60 mm
                offset = step * 0.25
                hit_pad = False
                query.overlap_box(
                    carb.Float3(0.000125, 0.005, 0.005),
                    carb.Float3(*((centre + closing * offset) * _MM_TO_M)),
                    carb.Float4(0.0, 0.0, 0.0, 1.0), report, False)
                if hit_pad:
                    face = offset
                    break
            if face is None:
                raise RuntimeError(
                    f"no pad face found within 60 mm of the grasp centre at joint angle {angle}; the "
                    "aperture cannot be calibrated and every commanded grip width would be a guess"
                )
            curve.append((angle, 2.0 * face))
        # Stored aperture-ascending, which for this joint is angle-descending. The direction is taken
        # from the measurement rather than asserted, because hard-coding which end of the driven
        # range is "open" is the mistake this calibration exists to avoid.
        curve.sort(key=lambda pair: pair[1])
        angles = [a for a, _ in curve]
        if angles != sorted(angles, reverse=True) and angles != sorted(angles):
            raise RuntimeError(
                f"the aperture curve is not monotonic in the joint angle ({curve}); a jaw whose "
                "opening does not move consistently with its drive cannot be commanded to a width"
            )
        return tuple(curve)

    def build_scene(self, geometry, *, kinematic_objects: bool = False) -> None:  # noqa: ANN001
        """Spawn the settled objects (and bin walls) exactly where the labels say they are.

        The drop is not replayed: the poses are already the settled ones, and re-dropping would
        reintroduce the PhysX non-determinism this measurement is trying to hold still.

        There is exactly one ``World.reset()`` per session and it happens in ``_build_stage``. Three
        facts decide that. Writing a USD translate op on a live rigid body does move it, so the
        retire works; a physics-view teleport moves it too; and ``reset()`` snaps every body back to
        the pose it had when the scene was last started, undoing both. A build sequence of retire,
        author, ``reset()`` therefore cancels its own retire: the previous scene comes back, the new
        one is authored inside it, and PhysX separates them by firing them apart, after which every
        trial grades a grasp against a scene that has been blown apart.
        """
        from pxr import (  # type: ignore[import-not-found]  # noqa: PLC0415
            Gf,
            PhysxSchema,
            Sdf,
            UsdGeom,
            UsdPhysics,
            UsdShade,
        )
        from isaacsim.core.prims import RigidPrim  # type: ignore[import-not-found]  # noqa: PLC0415

        stage = self._world.stage
        # Park the gripper first. It loads at the origin and its envelope spans x -42 to 40, y -76 to
        # 76, z -12 to 169 mm, so anything authored near the origin is spawned inside it and PhysX
        # resolves the overlap by firing it away, which knocks the scene out of its labelled poses.
        self._park()
        self._retire_previous()
        self._build_index += 1
        scene_root = f"{_OBJECT_ROOT}/build_{self._build_index}"
        UsdGeom.Xform.Define(stage, scene_root)

        for instance_id, solid in sorted(geometry.objects.items()):
            path = f"{scene_root}/obj_{instance_id}"
            _author_solid(stage, path, solid, kinematic=kinematic_objects,
                          mesh_collision=self._mesh_collision, Gf=Gf, Sdf=Sdf,
                          UsdGeom=UsdGeom, UsdPhysics=UsdPhysics, UsdShade=UsdShade,
                          PhysxSchema=PhysxSchema)
        for index, wall in enumerate(geometry.walls):
            path = f"{scene_root}/wall_{index}"
            # Kinematic, not static. A static collider has no rigid body and therefore no entry in
            # the physics view, which leaves it the one thing in the scene that cannot be retired,
            # and a previous bin's walls standing around the current scene corrupt it silently.
            # Kinematic is immovable by contact, like static, and teleportable through the same API
            # as everything else.
            _author_solid(stage, path, wall, kinematic=True, Gf=Gf, Sdf=Sdf, UsdGeom=UsdGeom,
                          UsdPhysics=UsdPhysics, UsdShade=UsdShade, PhysxSchema=PhysxSchema)
        # New prims become live rigid bodies immediately, so the views are built here and no reset
        # follows.
        # Debug level: one line per scene build, and a sample builds one scene per labelled scene
        # plus four for the controls: the jaw-solidity probe, the control block, the decoy, and the
        # control block rebuilt for the repeat. It is the detail a "refused: sits N mm from its
        # labelled pose" is read against.
        logger.debug("build %d: %s; %d object(s), %d wall(s), kinematic=%s",
                     self._build_index, getattr(geometry, "scene_id", "?"),
                     len(geometry.objects), len(geometry.walls), kinematic_objects)
        self._object_ids = sorted(geometry.objects)
        self._objects = None
        if self._object_ids:
            self._objects = RigidPrim(
                prim_paths_expr=f"{scene_root}/obj_*", name=f"objs_{self._build_index}")
            self._objects.initialize()
            self._object_ids = [int(str(path).rsplit("_", 1)[-1]) for path in self._objects.prim_paths]
            self._live.append(self._objects)
        if geometry.walls:
            walls = RigidPrim(prim_paths_expr=f"{scene_root}/wall_*", name=f"walls_{self._build_index}")
            walls.initialize()
            self._live.append(walls)
        self.restore(geometry)

    def _retire_previous(self) -> None:
        """Move every body of every earlier build onto its own slot of a far-away grid.

        Not deleted and not deactivated: removing a prim invalidates the PhysX tensor view, after
        which every pose read is quietly wrong, while moving a live body is something PhysX is happy
        about. There is no ground plane to catch them, so their gravity goes off as well: a body in
        permanent free fall accumulates velocity for the rest of the session. Each gets its own slot
        because coincident bodies depenetrate explosively.
        """
        for view in self._live:
            count = int(len(view.prim_paths))
            positions = np.zeros((count, 3))
            for row in range(count):
                slot = self._retire_slot
                self._retire_slot += 1
                positions[row] = [
                    _RETIRE_ORIGIN_MM[0] + (slot % _RETIRE_COLUMNS) * _RETIRE_PITCH_MM,
                    _RETIRE_ORIGIN_MM[1] + (slot // _RETIRE_COLUMNS) * _RETIRE_PITCH_MM,
                    200.0,
                ]
            orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (count, 1))
            view.set_world_poses(positions * _MM_TO_M, orientations)
            view.set_velocities(np.zeros((count, 6)))
            # And turn their gravity off. There is no ground plane under the cell by design (see
            # _build_stage), so a retired body falls for the rest of the session: the accumulated
            # velocities produce "Illegal BroadPhaseUpdateData", the articulation's joint angle goes
            # to NaN, and every trial after that is silently garbage. Retired means retired: parked
            # and weightless.
            self._set_gravity(view.prim_paths, disabled=True)
        self._live.clear()
        self._step(5)

    def _other_object_paths(self, keep_instance: int) -> list[str]:
        if self._objects is None:
            return []
        return [str(path) for row, path in enumerate(self._objects.prim_paths)
                if self._object_ids[row] != keep_instance]

    def _set_kinematic(self, paths, *, kinematic: bool) -> None:  # noqa: ANN001
        """Make bodies immovable-but-solid (or dynamic again), through USD."""
        from pxr import UsdPhysics  # type: ignore[import-not-found]  # noqa: PLC0415

        for path in paths:
            prim = self._world.stage.GetPrimAtPath(str(path))
            if prim and prim.IsValid():
                UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(kinematic)

    def _set_gravity(self, paths, *, disabled: bool) -> None:  # noqa: ANN001
        """Turn gravity on or off for a set of bodies, through USD (which PhysX picks up live)."""
        from pxr import PhysxSchema  # type: ignore[import-not-found]  # noqa: PLC0415

        for path in paths:
            prim = self._world.stage.GetPrimAtPath(str(path))
            if prim and prim.IsValid():
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(disabled)

    def _object_position(self, instance_id: int) -> np.ndarray:
        """Where an object is, from the physics view. Never from USD: see the note in build_scene.

        An object this cell does not have returns NaN, not the origin. Zeros would be a plausible
        pose that silently reads as "sitting at the base", and a missing measurement has to announce
        itself.
        """
        if self._objects is None or instance_id not in self._object_ids:
            return np.full(3, np.nan)
        positions, _ = self._objects.get_world_poses()
        row = np.asarray(positions, dtype=np.float64).reshape(-1, 3)[self._object_ids.index(instance_id)]
        return row / _MM_TO_M

    def _max_drift(self, geometry) -> float:  # noqa: ANN001
        """How far physics moved the scene from the renderer's settled poses.

        Reported, not refused: two different solvers resting the same body will not agree to the
        micrometre, and pretending otherwise would either fail every run or hide a real divergence
        behind a loose tolerance.
        """
        worst = 0.0
        for instance_id, solid in sorted(geometry.objects.items()):
            delta = self._object_position(instance_id) - np.asarray(solid.centre_mm, dtype=np.float64)
            worst = max(worst, float(np.linalg.norm(delta)) if np.all(np.isfinite(delta)) else np.inf)
        return worst

    # -------------------------------------------------------------- trials

    def restore(self, geometry) -> None:  # noqa: ANN001
        """Put every object back on its labelled pose, at rest, and let it settle.

        This stands in for ``reset()``, which the session calls exactly once. It is a physics-view
        teleport, and the gripper is parked first because teleporting an object into the gripper's
        envelope throws it metres away.
        """
        self._park()
        self._reset_jaw()
        # The table went away during the last trial's hold test; put it back before anything is asked
        # to rest on it.
        self._set_table_z(-_TABLE_EXTENT_MM[2] / 2.0)
        if self._objects is None or not self._object_ids:
            return
        positions = np.zeros((len(self._object_ids), 3))
        orientations = np.zeros((len(self._object_ids), 4))
        for row, instance_id in enumerate(self._object_ids):
            solid = geometry.objects[instance_id]
            positions[row] = np.asarray(solid.centre_mm, dtype=np.float64) * _MM_TO_M
            orientations[row] = _matrix_to_quat_wxyz(solid.rotation)
        self._objects.set_world_poses(positions, orientations)
        # Un-freeze before zeroing. PhysX refuses `setLinearVelocity` on a kinematic body: silently
        # as far as this code is concerned, loudly in its own log. Zeroing first is therefore a no-op
        # on every object the previous trial froze as an obstacle, and each one comes back dynamic
        # still carrying the velocity that trial left it with, which for an exploded trial is
        # enormous.
        self._set_gravity(self._objects.prim_paths, disabled=False)   # undo _drop_support
        self._set_kinematic(self._objects.prim_paths, kinematic=False)   # undo the obstacle freeze
        self._objects.set_velocities(np.zeros((len(self._object_ids), 6)))
        self._step(_STEPS_RESTORE)
        self._settle_drift_mm = self._max_drift(geometry)

    def check_ready(self, geometry, instance_id: int) -> str:  # noqa: ANN001
        """Is the scene in front of the gripper the scene the labels describe? Empty string if yes.

        Read through the physics view, against the labelled poses. The authored USD transform is
        correct by construction, so a check against it agrees with itself while the actual scene lies
        in pieces. A trial that fails here is refused rather than scored: a grasp graded against
        wreckage is not a measurement of the grasp, and calling it "did not hold" folds a harness
        fault into the result.
        """
        position = self._object_position(instance_id)
        if not np.all(np.isfinite(position)):
            return f"target {instance_id} has a non-finite pose {position.tolist()}"
        target = geometry.objects.get(instance_id)
        if target is None:
            return f"target {instance_id} is not in this scene"
        drift = float(np.linalg.norm(position - np.asarray(target.centre_mm, dtype=np.float64)))
        if drift > _DRIFT_REFUSE_MM:
            return f"target {instance_id} sits {drift:.1f} mm from its labelled pose"
        return ""

    def target_drift_mm(self, geometry, instance_id: int) -> float:  # noqa: ANN001
        target = geometry.objects.get(instance_id)
        if target is None:
            return float("nan")
        position = self._object_position(instance_id)
        return float(np.linalg.norm(position - np.asarray(target.centre_mm, dtype=np.float64)))

    def run_trial(self, trial: PhysicsTrial, geometry) -> TrialOutcome:  # noqa: ANN001
        """Approach, close, take the support away. ``held`` is whether the object stayed up.

        The caller restores the scene before each trial (:meth:`restore`) and the trial refuses itself
        if the restored scene is not the labelled one (:meth:`check_ready`). A refusal is reported as
        such, with ``refused`` in the note, and the caller keeps it out of every rate, because a grasp
        graded against a scene that drifted is not evidence about the grasp.
        """
        position = np.asarray(trial.position_mm, dtype=np.float64)
        approach = np.asarray(trial.approach, dtype=np.float64)
        closing = np.asarray(trial.closing_axis, dtype=np.float64)
        if float(np.linalg.norm(closing)) < 1e-6:
            return TrialOutcome(trial, False, 0.0, "refused: no closing axis recorded")
        if not (np.all(np.isfinite(position)) and np.all(np.isfinite(approach))):
            return TrialOutcome(trial, False, 0.0, "refused: the candidate pose is not finite")
        problem = self.jaw_fault() or self.check_ready(geometry, trial.instance_id)
        if problem:
            return TrialOutcome(trial, False, 0.0, f"refused: {problem}")
        # Measured before the approach. Afterwards the target has moved under the close and the
        # dropped support, and a "drift" read at the end would report that as a fault.
        drift_before = self.target_drift_mm(geometry, trial.instance_id)
        # The neighbours become kinematic for the duration of the trial. They are here to be
        # obstacles, the thing the verdict's finger_collision check models, not to be simulated.
        #
        # It does not do what it was introduced to do: freezing them makes blow-ups slightly more
        # common, not less. It is kept because it removes a different failure outright. Without it a
        # trial that explodes leaves the scene displaced, and the trials after it are refused for
        # finding their scene out of place.
        self._set_kinematic(self._other_object_paths(trial.instance_id), kinematic=True)

        width = trial.width_mm if trial.width_mm > 1.0 else 85.0
        self._drive_joint(self._joint_for_opening(min(85.0, width + 12.0)), steps=20)
        self._place(position - approach * _STANDOFF_MM, approach, closing)
        self._step(5)
        for step in range(_STEPS_APPROACH):
            fraction = (step + 1) / _STEPS_APPROACH
            self._place(position - approach * _STANDOFF_MM * (1.0 - fraction), approach, closing)
            self._step(1)

        # Squeeze, do not crush. Commanding full travel against a 40 mm object is 40 mm of
        # over-travel into a stiff position drive, which fires the object sideways out from between
        # the pads instead of gripping it. A real gripper closes until force, so the command is a few
        # millimetres inside the object's own width.
        commanded = self._joint_for_opening(max(4.0, width - _SQUEEZE_MM))
        self._drive_joint(commanded, steps=_STEPS_CLOSE)
        # Whether the jaw stalled is the difference between "this grasp failed" and "this jaw closed on
        # nothing", and the two are indistinguishable in a hold rate. Recorded per trial.
        reached = float(self._gripper.get_joint_positions()[self._joint_index])
        blown = self.jaw_fault()
        if blown:
            # The articulation left physics behind. Refused, never scored: a dead simulation reports
            # "did not hold" for every grasp, which is the most convincing wrong answer available.
            # Reaching here means this trial broke it, since an inherited explosion is refused
            # before the approach, so the next `restore()` teleports it back instead of passing it
            # on.
            return TrialOutcome(trial, False, 0.0, f"refused: {blown}")
        gap = self._opening_for_joint(reached)
        start_z = float(self._object_position(trial.instance_id)[2])

        # The hold test is gravity, not a lift. A world-fixed articulation moved by
        # ``set_world_pose`` transmits no tangential force: the teleport re-places the pads every step
        # with no relative velocity, so PhysX's friction solver has nothing to act on and a lifted
        # object is left behind however good the grip is.
        #
        # Taking the table away instead asks the question the lift was a proxy for, whether the jaw
        # can carry this object's weight, with no gripper motion at all, so no artefact of how the
        # gripper is moved can enter the answer. The neighbours go with the table: whether the target
        # is propped up by the pile is not what a grasp is being credited for.
        self._drop_support(trial.instance_id)
        self._step(_STEPS_HANG)

        end_z = float(self._object_position(trial.instance_id)[2])
        offset = self.gripper_offset_mm()
        if not (np.isfinite(start_z) and np.isfinite(end_z)):
            return TrialOutcome(trial, False, 0.0,
                                "refused: the object's pose went non-finite during the trial")
        if offset > _GRIPPER_OFFSET_REFUSE_MM:
            # The gripper is teleported, so it can only be out of position if physics moved it
            # against the command, which means the trial measured the harness, not the grasp.
            return TrialOutcome(trial, False, 0.0,
                                f"refused: the gripper ended {offset:.1f} mm off its commanded pose")
        drop = start_z - end_z
        return TrialOutcome(
            trial, drop < _HELD_DROP_MM, -drop,
            f"drift {drift_before:.2f} mm; jaw asked {max(4.0, width - _SQUEEZE_MM):.1f} mm "
            f"({commanded:.3f} rad), stalled at {gap:.1f} mm ({reached:.3f} rad)")

    def _set_table_z(self, z_mm: float) -> None:
        """Move the table, through USD rather than through the physics view.

        A kinematic body does not follow ``set_world_poses``: PhysX keeps driving it to its kinematic
        target, so the write is silently ignored, the table stays retired, and every restored scene
        free-falls through its settle steps. Writing the USD translate op does move it, and it
        sticks, because this session never calls ``reset()`` again after start-up.
        """
        from pxr import Gf, UsdGeom  # type: ignore[import-not-found]  # noqa: PLC0415

        prim = self._world.stage.GetPrimAtPath(_TABLE_PATH)
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(0.0, 0.0, float(z_mm) * _MM_TO_M))

    def _drop_support(self, keep_instance: int) -> None:
        """Take the table (and every object but the target) out from under the grasp."""
        self._set_table_z(_TABLE_RETIRE_Z_MM)
        if self._objects is None or not self._object_ids:
            return
        positions, orientations = self._objects.get_world_poses()
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3).copy()
        aside = [str(p) for row, p in enumerate(self._objects.prim_paths)
                 if self._object_ids[row] != keep_instance]
        for row, instance_id in enumerate(self._object_ids):
            if instance_id != keep_instance:
                positions[row] = [
                    (_RETIRE_ORIGIN_MM[0] + (row % _RETIRE_COLUMNS) * _RETIRE_PITCH_MM) * _MM_TO_M,
                    (_RETIRE_ORIGIN_MM[1] - 2000.0) * _MM_TO_M, 0.0]
        self._objects.set_world_poses(positions, orientations)
        # PhysX logs `setLinearVelocity: Body must be non-kinematic!` for every object the running
        # trial froze as an obstacle. It is a no-op on a body that cannot move anyway, and
        # un-freezing them here to silence it would make them fall through a cell that has no ground
        # plane, so the ordering is left as it is. In `restore()` the same ordering is a real fault
        # and is corrected there, because those bodies go dynamic again straight afterwards.
        self._objects.set_velocities(np.zeros((len(self._object_ids), 6)))
        self._set_gravity(aside, disabled=True)      # they are out of the way, not falling forever

    def _check_jaw_is_solid(self) -> dict:
        """Drop a block on the shut jaw. If it passes through, every later number is decoration.

        A gripper with no contact against dynamic bodies at all produces grasp results that look
        exactly like a jaw that grips badly, and nothing else in the run says otherwise. One block,
        released 90 mm above the closed pads, separates "gripped nothing" from "cannot grip
        anything".
        """
        from datagen.grasps.labels import SceneGeometry  # noqa: PLC0415

        top = _PARK_POSITION_MM[2] + 165.0          # the fingertips, measured: 164.6 mm from base
        probe = solid_from_scene("box", (30.0, 30.0, 30.0), (0.0, 0.0, top + 90.0),
                                 (0.0, 0.0, 0.0, 1.0), instance_id=0, asset_id="solidity", mass_kg=0.1)
        self.build_scene(SceneGeometry("jaw_solidity", "control", {0: probe}, ()))
        self._drive_joint(self._frame.angle_shut if self._frame else _JOINT_LIMIT_RAD[1], steps=40)
        self._step(200)
        rest = float(self._object_position(0)[2])
        # The bar is "it stopped at the gripper rather than on the floor", not "it stopped on the
        # fingertips". A shut 2F-85 swings its tips inward and down, so a block that lands on the jaw
        # settles onto the knuckles a few centimetres lower, and insisting on the fingertip height
        # fails a gripper that works.
        return {"solid": bool(np.isfinite(rest) and rest > _PARK_POSITION_MM[2]),
                "rest_mm": round(rest, 1), "with_no_jaw_it_falls_forever": True}

    def run_controls(self) -> dict:
        """The trials that decide whether anything else here is worth reading.

        The control object is a 90 mm tall block, not a 40 mm cube: the 2F-85's inner finger spans
        z = 102.3 to 164.6 mm from its base, so a pad centred on a 40 mm cube's mid-height puts the
        fingertip 11 mm under the table, which is not a grasp a real gripper can perform. The
        negative control sits 150 mm to the side rather than 60 mm, because at 60 mm the near pad is
        still inside the cube and catapults it, and "gripping air" has to actually grip air.

        The repeat is the third control and the one that matters. A positive and a negative can both
        pass while the harness holds nothing, when the fault is in what happens to a scene as the
        next one is built. So the positive is repeated after an unrelated scene has been built and
        retired in between: if the lifecycle leaks, the repeat fails and the whole sample is refused
        before it costs an hour. A control that only ever runs on a fresh session cannot see a fault
        that appears on the second one.
        """
        from datagen.grasps.labels import SceneGeometry  # noqa: PLC0415

        block = solid_from_scene("box", (40.0, 40.0, 90.0), (0.0, 0.0, 45.0), (0.0, 0.0, 0.0, 1.0),
                                 instance_id=0, asset_id="control_block", mass_kg=0.1)
        geometry = SceneGeometry("control", "control", {0: block}, ())
        centred = PhysicsTrial("control", 0, "control_positive", (0.0, 0.0, 70.0),
                               (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), 40.0)
        beside = PhysicsTrial("control", 0, "control_negative", (150.0, 0.0, 70.0),
                              (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), 40.0)

        solid = self._check_jaw_is_solid()

        self.build_scene(geometry)
        authored_check = [round(float(v), 2) for v in self._object_position(0)]
        positive = self.run_trial(centred, geometry)
        self.restore(geometry)
        negative = self.run_trial(beside, geometry)

        # An unrelated scene, built and retired, standing between the first positive and the repeat.
        decoy = solid_from_scene("box", (60.0, 60.0, 60.0), (300.0, 0.0, 30.0), (0.0, 0.0, 0.0, 1.0),
                                 instance_id=0, asset_id="decoy", mass_kg=0.2)
        self.build_scene(SceneGeometry("decoy", "control", {0: decoy}, ()))
        self.build_scene(geometry)
        repeat_check = [round(float(v), 2) for v in self._object_position(0)]
        repeat = self.run_trial(centred, geometry)

        assert self._frame is not None
        return {
            "jaw_is_solid": solid["solid"], "jaw_probe_rest_mm": solid["rest_mm"],
            "positive_held": positive.held, "positive_drop_mm": round(-positive.rise_mm, 2),
            "negative_held": negative.held, "negative_drop_mm": round(-negative.rise_mm, 2),
            "repeat_held": repeat.held, "repeat_drop_mm": round(-repeat.rise_mm, 2),
            "repeat_note": repeat.note,
            "measured_tcp_offset_mm": round(self._frame.tcp_offset_mm, 2),
            "measured_open_width_mm": round(self._frame.opening_at_open_mm, 2),
            "settle_drift_mm": round(self._settle_drift_mm, 3),
            "control_object_first_build_mm": authored_check,
            "control_object_repeat_build_mm": repeat_check,
            "control_object_authored_mm": [0.0, 0.0, 45.0],
            "closing_axis_local": [round(float(v), 3) for v in self._frame.closing_axis_local],
            "approach_local": [round(float(v), 3) for v in self._frame.approach_local],
        }


# ------------------------------------------------------------------ helpers


def _author_solid(stage, path: str, solid: Solid, *, kinematic: bool = False,
                  mesh_collision: str = "sdf", **usd) -> None:
    """Two prims: a rigid body carrying the translation and rotation, a child collider the scale.

    The same authoring the renderer uses, and for the same measured reason: a non-uniform scale on a
    rotated rigid body is not representable in PhysX, so the collider stops being the shape that was
    rendered and labelled.
    """
    Gf, Sdf, UsdGeom, UsdPhysics = usd["Gf"], usd["Sdf"], usd["UsdGeom"], usd["UsdPhysics"]
    UsdShade = usd["UsdShade"]
    quat = _matrix_to_quat_wxyz(solid.rotation)
    body = UsdGeom.Xform.Define(stage, Sdf.Path(path))
    xform = UsdGeom.Xformable(body.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*(float(v) * _MM_TO_M for v in solid.centre_mm)))
    xform.AddOrientOp().Set(Gf.Quatf(float(quat[0]),
                                     Gf.Vec3f(float(quat[1]), float(quat[2]), float(quat[3]))))
    rigid = UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(max(0.005, float(solid.mass_kg)))
    physx = usd["PhysxSchema"].PhysxRigidBodyAPI.Apply(body.GetPrim())
    physx.CreateMaxDepenetrationVelocityAttr(_MAX_DEPENETRATION_MPS)
    physx.CreateSolverPositionIterationCountAttr(_SOLVER_POSITION_ITERATIONS)
    physx.CreateSolverVelocityIterationCountAttr(_SOLVER_VELOCITY_ITERATIONS)
    if kinematic:
        # A wall is a rigid body so it has a physics-view entry and can be retired; kinematic so no
        # contact can push it. A static collider is the one thing in a scene that cannot be moved
        # out of the way afterwards.
        rigid.CreateKinematicEnabledAttr(True)

    geom_path = f"{path}/Geom"
    extent = np.asarray(solid.half_extent_mm, dtype=np.float64) * 2.0 * _MM_TO_M
    if solid.kind == "sphere":
        radius = float(extent[0] / 2.0)
        geom = UsdGeom.Sphere.Define(stage, Sdf.Path(geom_path))
        geom.CreateRadiusAttr(radius)
        geom.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -radius),
                               Gf.Vec3f(radius, radius, radius)])
    elif solid.kind == "cylinder":
        radius, height = float(extent[0] / 2.0), float(extent[2])
        geom = UsdGeom.Cylinder.Define(stage, Sdf.Path(geom_path))
        geom.CreateRadiusAttr(radius)
        geom.CreateHeightAttr(height)
        geom.CreateAxisAttr(UsdGeom.Tokens.z)
        geom.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2.0),
                               Gf.Vec3f(radius, radius, height / 2.0)])
    elif solid.mesh is not None:
        # A scanned object. Without this branch it falls through to the cube below and the shake
        # collides a cuboid while the label describes a mug. It is the least visible version of that
        # failure, because a physics outcome carries no shape in it to check.
        from pxr import Vt  # type: ignore[import-not-found]  # noqa: PLC0415

        points, counts, indices, lower, upper = solid.mesh.usd_arrays()
        geom = UsdGeom.Mesh.Define(stage, Sdf.Path(geom_path))
        geom.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*(float(v) for v in p)) for p in points]))
        geom.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
        geom.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
        geom.CreateExtentAttr([Gf.Vec3f(*(float(v) for v in lower)),
                               Gf.Vec3f(*(float(v) for v in upper))])
        geom.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr(mesh_collision)
    else:
        geom = UsdGeom.Cube.Define(stage, Sdf.Path(geom_path))
        geom.CreateSizeAttr(1.0)
        geom.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
        UsdGeom.Xformable(geom.GetPrim()).AddScaleOp().Set(
            Gf.Vec3f(*(float(v) for v in extent)))
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    _bind_friction(stage, geom.GetPrim(), UsdShade=UsdShade)


def _bind_friction(stage, prim, *, UsdShade) -> None:  # noqa: ANN001, N803
    """Bind the cell's one stated friction material to a collider. See the note in _build_stage."""
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        UsdShade.Material(stage.GetPrimAtPath(_FRICTION_MATERIAL)),
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics")


def _quat_wxyz_to_matrix(quat_wxyz) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.asarray(quat_wxyz, dtype=np.float64))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Isaac wants WXYZ; the repo speaks XYZW. Converted here, at the boundary, like a driver."""
    from src.geometry.quaternion import from_rotation_matrix  # noqa: PLC0415

    q = np.asarray(from_rotation_matrix(np.asarray(rotation, dtype=np.float64)), dtype=np.float64)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
