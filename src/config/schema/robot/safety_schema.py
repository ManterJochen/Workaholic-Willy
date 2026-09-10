"""Vendor-neutral safety-guard config schema."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .._base import StrictModel


class LimitsSafetyConfig(StrictModel):
    """Workspace-margin safety config.

    ``workspace_margin_mm`` is consumed by :class:`SafetyPreflight.from_safety_config` to shrink the
    ``workspace_limits`` box on every face before the workspace guard runs.

    No velocity or acceleration ceiling lives here. Per-move vel/acc come from
    :class:`MotionLimitsConfig` (``config.motion_limits``), which is also where hardware
    speed-limiting belongs, as a ``min(motion_limits, safety)`` clamp at the UR/KUKA boundary.
    """

    enforce: bool = Field(default=True)

    workspace_margin_mm: float = Field(default=20.0, ge=0.0, le=500.0)


class JointLimitSafetyConfig(StrictModel):
    """Per-axis joint-limit guard configuration.

    The guard prefers driver-side telemetry where it exists (UR's RTDE
    ``query_joint_limits``) and otherwise falls back to the static ``min_deg`` /
    ``max_deg`` lists below. With neither source it returns
    :attr:`SafetyReason.UNAVAILABLE` and fails closed while ``enforce`` is ``True``.

    ``margin_deg`` is the buffer in degrees the guard keeps between each commanded
    joint angle and the axis limit itself.
    """

    enforce: bool = Field(default=True)
    margin_deg: float = Field(default=5.0, ge=0.0, le=45.0)
    # Optional per-axis static fallback in degrees. When set, both lists must be
    # as long as the arm's DoF, with ``min_deg[i] < max_deg[i]`` on every axis i.
    min_deg: list[float] | None = Field(default=None)
    max_deg: list[float] | None = Field(default=None)

    @model_validator(mode="after")
    def _check_axis_ordering(self) -> JointLimitSafetyConfig:
        if (self.min_deg is None) != (self.max_deg is None):
            raise ValueError(
                "joint_limit.min_deg and joint_limit.max_deg must be set "
                "together (both or neither)."
            )
        if self.min_deg is not None and self.max_deg is not None:
            if len(self.min_deg) != len(self.max_deg):
                raise ValueError(
                    "joint_limit.min_deg and joint_limit.max_deg must have "
                    "the same length."
                )
            for i, (lo, hi) in enumerate(zip(self.min_deg, self.max_deg)):
                if lo >= hi:
                    raise ValueError(
                        f"joint_limit axis {i}: min_deg ({lo}) must be < "
                        f"max_deg ({hi})."
                    )
        return self


class IkQualitySafetyConfig(StrictModel):
    """IK-solution-quality guard configuration.

    The guard runs once the driver has produced a joint solution, and rejects:

    * NaN or wrong-DoF solutions outright,
    * solutions differing from the current pose by more than ``max_jump_rad`` on
      any axis, since a jump that large signals a wrap-around ambiguity,
    * solutions whose Jacobian condition number exceeds ``max_condition_number``
      or whose smallest singular value falls below ``min_singular_value``, both
      near-singular configurations,
    * solutions inside ``limit_proximity_deg`` of the configured joint limits,
      which the joint-limit guard also covers when enabled and this one still
      catches when it is not.
    """

    enforce: bool = Field(default=True)
    max_jump_rad: float = Field(default=1.0, gt=0.0, le=6.283185307179586)
    min_singular_value: float = Field(default=0.005, gt=0.0, le=1.0)
    max_condition_number: float = Field(default=250.0, gt=0.0, le=1.0e6)
    limit_proximity_deg: float = Field(default=2.0, ge=0.0, le=45.0)


class FixtureBoxConfig(StrictModel):
    """Static fixture obstacle used by the self-collision guard.

    Boxes are axis-aligned in the robot base frame, given as ``center_mm``, the
    geometric centre, and ``half_extents_mm``, half the side length on each axis.
    The capsule backend checks fixtures against capsule-approximated arm links,
    the ``fcl`` backend against the user-provided mesh files.
    """

    name: str = Field(default="fixture", min_length=1)
    center_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    half_extents_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @model_validator(mode="after")
    def _check_positive_extents(self) -> FixtureBoxConfig:
        for axis, extent in zip(("x", "y", "z"), self.half_extents_mm):
            if extent < 0.0:
                raise ValueError(
                    f"fixture {self.name!r}: half_extents_mm.{axis} must be "
                    f">= 0; got {extent}."
                )
        return self


class SelfCollisionSafetyConfig(StrictModel):
    """Self-collision guard configuration.

    Two ``backend`` values are supported, the default being ``fcl``:

    * ``capsule`` is always available. It approximates each arm link as a capsule
      and runs a closed-form distance check over every link pair and every
      link/fixture pair.
    * ``fcl`` uses the optional ``python-fcl`` package against the meshes in
      ``mesh_dir``. Selected without a populated ``mesh_dir``, the guard reports
      :attr:`SafetyReason.UNAVAILABLE` and, while ``enforce`` is ``True``, rejects
      the motion: it fails closed.

    ``min_distance_mm`` is the minimum signed distance between any two monitored
    shapes; closer than that counts as a collision.

    ``link_radii_mm`` is the per-link capsule radius in mm; when omitted the guard
    uses a single default radius supplied at runtime. Its length is not validated
    here because DoF is a driver-side property: the guard checks it at evaluation
    time and reports :attr:`SafetyReason.UNAVAILABLE` on mismatch.
    """

    enforce: bool = Field(default=True)
    backend: Literal["capsule", "fcl"] = Field(default="fcl")
    min_distance_mm: float = Field(default=10.0, ge=0.0, le=500.0)
    #: Clearance (mm) handed to the trajectory planner so it stops proposing configurations this guard
    #: will reject: cuRobo plans against a sphere model, the guard re-checks the exact meshes, and the
    #: two disagree. cuRobo returns UR5e plans at 9.44-9.47 mm against this guard's 10.000 mm, losing
    #: 3 of 10 picks to what reads as bad grasping. Not derived from ``min_distance_mm``: the margin a
    #: planner can absorb depends on how tightly its spheres fit that robot. A UR5e plans fine at
    #: 10 mm; a UR3e plans fine to 6 mm and finds no plan at all at 10 mm, its thinner links reading as
    #: permanent self-collision, which takes a UR3e cell from 10/10 to 0/10. The default 0.0 leaves the
    #: planner's own config untouched and is byte-identical.
    planner_margin_mm: float = Field(default=0.0, ge=0.0, le=100.0)
    link_radii_mm: list[float] | None = Field(default=None)
    fixtures: list[FixtureBoxConfig] = Field(default_factory=list)
    mesh_dir: str | None = Field(default=None)

    # Which bundled arm kinematics (DH) table supplies the per-link arm-vs-arm capsules. Under the
    # default ``None`` only a real ``vendor == "ur"`` arm gets arm-vs-arm capsules, keyed by its own
    # model. Setting e.g. ``"ur5e"`` lets a non-UR vendor, the Isaac sim being physically a UR5e, opt
    # into the UR DH and so get real arm-vs-arm self-collision rather than base, tool and fixture
    # checks alone.
    kinematics_model: str | None = Field(default=None)

    # Names the per-gripper fcl/Coal mesh bundle the backend loads (``{variant}_collision_meshes.npz``)
    # in place of ``{kinematics_model}_...``, for a mounted gripper whose collision geometry differs
    # from the baked Robotiq 2F-85. The arm kinematics stay ``kinematics_model``: the variant bundle
    # copies the arm-link meshes and swaps only the gripper meshes. The default ``None`` is the
    # kinematics_model bundle, the 2F-85. The sim threads this from ``robot.sim.gripper_mount``.
    collision_mesh_variant: str | None = Field(default=None)

    # The coupling plate between the ARM FLANGE and the gripper's own MOUNTING FACE, millimetres.
    #
    # ⛔ ONLY A VARIANT BUNDLE NEEDS IT, AND ONLY THE GUARD WAS NEVER TOLD. A bundle baked from a
    # composed arm asset already sits where the hand is bolted (`gripper__origin` absent, or
    # "flange"). A bundle read from a standalone vendor asset starts at the hand's own mounting face
    # and stamps `gripper__origin = "mounting_face"`, which the bake module documents as "whatever
    # plate sits between that face and the flange has to be added before the planner sees it". The
    # sphere fit reads that stamp and the on-box cuRobo builder adds the plate via `--coupling-mm`;
    # the exact-mesh guard read neither and had no parameter that could carry the number, so a
    # Hand-E cell ran two collision models of the same hand that differed by one plate.
    #
    # ⚠ Default 0.0 is exactly the previous behaviour, byte-identical for every existing cell, and
    # it is NOT a safe guess: it is the absence of a measurement. A cell that leaves it at 0.0 with a
    # mounting-face bundle is told so once, loudly, at guard construction.
    #
    # It is the SAME bench measurement as the plate term in `robot.gripper.tool_frame.offset_mm` and
    # `build_ur_config.py --coupling-mm`. Measure it once and write it in all three, or the three
    # descriptions of one hand disagree, which is the family of defect this cell has already had.
    coupling_mm: float = Field(default=0.0, ge=0.0)

    # Yaw (degrees) of the bundled-DH base frame relative to the robot/system base frame that poses and
    # fixtures are expressed in. The official UR DH (``_ur_kinematics.py``) base is rotated 180 deg
    # about Z from the Isaac UR5e USD ``base_link``: ``ur_link_origins_mm`` == ``[-x, -y, z]`` against
    # the Lula FK ground truth at every pose. The guard rotates the DH-derived arm-link origins by this
    # yaw so that the arm capsules line up with the base, tool and fixture capsules, which are already
    # in the system frame. Only the arm-link capsules move, and arm-vs-arm distance is
    # rotation-invariant, so this knob cannot change it. The default 0.0 is no rotation and
    # byte-identical for every existing cell. The arm self-collision path has never run on real UR, so
    # a real ``vendor == "ur"`` cell must have its base yaw validated on hardware before
    # arm-vs-fixture capsules are relied on.
    kinematics_base_yaw_deg: float = Field(default=0.0, ge=-360.0, le=360.0)

    # Tool and base capsule geometry (mm); the defaults match the module constants, so a cell that
    # sets none of them keeps the built-in envelope. Override them to declare the real mounted tool,
    # e.g. the Robotiq 2F-85, so tool-vs-link and tool-vs-fixture distances describe the tool that is
    # actually there.
    tool_length_mm: float = Field(default=150.0, gt=0.0, le=1000.0)
    tool_radius_mm: float = Field(default=70.0, gt=0.0, le=500.0)
    base_radius_mm: float = Field(default=80.0, gt=0.0, le=1000.0)
    base_height_mm: float = Field(default=150.0, gt=0.0, le=2000.0)

    # The tool collision model. The default ``capsule`` is one capsule along the tool approach axis
    # (R[:, 2]) with radius ``tool_radius_mm``, a rotation-invariant bounding cylinder. For a
    # parallel-jaw gripper that is over-conservative against a bin wall: a 2F-85 reaching into a KLT is
    # ~27 mm wide perpendicular to its closing axis (from the Robotiq_2f_85 USD) while the r=70 cylinder
    # claims 140 mm in every direction, so an off-center target is falsely wall-rejected. ``finger``
    # models the descending fingers instead, as a thin capsule along the grasp's closing axis (R[:, 0];
    # the [closing, binormal, approach] convention of execution_policy._quaternion_from_axes), of length
    # ``tool_finger_span_mm`` (fingertip to fingertip at full open) and radius ``tool_finger_radius_mm``
    # (the perpendicular half-width). Being rotation-aware, it clears the narrow wall the thin side
    # faces and still rejects when the open span faces it. Keeping the default is byte-identical to
    # every existing cell.
    tool_model: Literal["capsule", "finger"] = Field(default="capsule")
    tool_finger_radius_mm: float = Field(default=16.0, gt=0.0, le=200.0)
    tool_finger_span_mm: float = Field(default=150.0, gt=0.0, le=500.0)


class PayloadSafetyConfig(StrictModel):
    """Tool and payload envelope.

    Validated when the operator updates the payload, by calling
    ``URRobotArm.set_payload`` or by editing this YAML and reloading. Per-move
    evaluation only verifies that the currently configured payload still fits the
    envelope.

    ``cog_mm`` is the centre-of-gravity offset from the flange in mm.
    ``inertia_kgm2`` is the diagonal inertia tensor (Ixx, Iyy, Izz) in kg*m^2.
    ``max_mass_kg`` is the absolute upper bound the guard refuses to set,
    regardless of the configured ``mass_kg``.
    """

    enforce: bool = Field(default=True)
    mass_kg: float = Field(default=0.0, ge=0.0, le=50.0)
    max_mass_kg: float = Field(default=5.0, gt=0.0, le=50.0)
    cog_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    inertia_kgm2: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @model_validator(mode="after")
    def _check_envelope(self) -> PayloadSafetyConfig:
        if self.mass_kg > self.max_mass_kg:
            raise ValueError(
                f"payload.mass_kg ({self.mass_kg}) must be <= max_mass_kg "
                f"({self.max_mass_kg})."
            )
        for axis, val in zip(("Ixx", "Iyy", "Izz"), self.inertia_kgm2):
            if val < 0.0:
                raise ValueError(
                    f"payload.inertia_kgm2.{axis} must be >= 0; got {val}."
                )
        return self


class MotionContinuitySafetyConfig(StrictModel):
    """Step-size guard between consecutive commanded targets.

    Catches an accidental ``move(other_pose)`` that would slam the arm across the
    workspace. The guard caches the last accepted target inside
    :class:`SafetyPreflight`, so the first command after a
    :meth:`SafetyPreflight.reset` is always accepted: there is no previous target
    to diff against.
    """

    enforce: bool = Field(default=True)
    max_joint_step_deg: float = Field(default=45.0, gt=0.0, le=180.0)
    max_orientation_step_deg: float = Field(default=30.0, gt=0.0, le=180.0)
    max_tcp_step_mm: float = Field(default=250.0, gt=0.0, le=2000.0)


class DwellSafetyConfig(StrictModel):
    """Post-Stop dwell and steady-state gating.

    A temporal check, does the controller report steady state, rather than the per-target spatial
    check the other guards make, so it sits outside the per-move :class:`SafetyPreflight` pipeline.

    ``require_steady_before_motion`` and ``steady_timeout_s`` are consumed by
    :class:`GraspExecutionPolicy`: before every commanded approach, grasp and retreat move the policy
    blocks on ``arm.wait_until_steady(steady_timeout_s)`` and fails closed on timeout.
    ``dwell_after_stop_s`` is not enforced; there is no exercised Stop or E-stop path in the runtime.
    """

    require_steady_before_motion: bool = Field(default=True)
    steady_timeout_s: float = Field(default=5.0, gt=0.0, le=60.0)


class SupportPlaneConfig(StrictModel):
    """The bench, table or floor the cell stands on, as one axis-aligned slab in the base frame.

    ``height_mm`` is the top surface, the number an operator can measure: put a rule on the bench and
    read the height above the robot's base plate. The slab is built downwards from there by
    ``thickness_mm``, so raising the thickness never moves the surface the arm must stay above.

    Sizing it is a real decision: a slab at the bench surface makes the planner refuse low top-down
    reaches, which is why the sim runners closest to a production pick sink their floor to a top of
    ``-50 mm`` and carry no walls at all.

    ``center_mm`` is where the slab sits in the base plane, and it exists because a robot is not
    always bolted to the middle of its table. A cell with its base at the head of a 550 mm bench that
    runs from x=250 to x=800 has no bench at all under the half it works over, unless the slab is
    either moved there or made wide enough to reach it from the centre. Both work; only one of them
    stops short of covering ground the arm never visits. Oversizing remains the answer for a cell
    that will not touch its config, and it is harmless where the workspace limits already forbid the
    space behind the robot.
    """

    height_mm: float = Field(default=0.0)
    extent_mm: tuple[float, float] = (2000.0, 2000.0)
    thickness_mm: float = Field(default=50.0, gt=0.0, le=2000.0)
    #: Where the slab is centred in the base plane, millimetres. The default is the base itself,
    #: which is what every configuration written before this field existed meant.
    #:
    #: Note the asymmetry with ``height_mm``, which is the slab's top rather than its centre: a
    #: height is measured against a surface an operator can put a rule on, and a position is not.
    center_mm: tuple[float, float] = (0.0, 0.0)

    @model_validator(mode="after")
    def _check_extent(self) -> SupportPlaneConfig:
        for axis, extent in zip(("x", "y"), self.extent_mm):
            if extent <= 0.0:
                raise ValueError(
                    f"support_plane.extent_mm.{axis} must be > 0; got {extent}. A zero-extent slab is "
                    "not a table, it is a plane the planner cannot collide with."
                )
        return self


class TrajectoryCheckConfig(StrictModel):
    """Whether a planned path is checked configuration by configuration before any of it is commanded.

    The one-shot guards judge where a move ends, so a plan that grazes a fixture in the middle and
    lands clear passes them. While this is disabled nothing checks the middle: the sim applies each
    waypoint straight to the articulation and the real UR moveJ's them in turn, so the planner is
    asked for a collision-free path and then trusted to have produced one.

    The exact-mesh backend costs about 9.6 ms per configuration, so a 30 to 100 waypoint plan adds
    roughly 0.3 to 1.0 s before the arm starts moving. That is paid once per move, ahead of motion,
    not interleaved with control.
    """

    enabled: bool = Field(default=False)
    stride: int = Field(default=1, ge=1, le=64)


class AttachedPayloadConfig(StrictModel):
    """Whether the planner is told that the gripper is carrying something.

    The planner's collision model ends at the gripper, so every transit, lift, place and retreat after
    a successful close is planned as if the hand were empty. On a cell carrying a part out of a bin
    that part is the geometry most likely to meet a wall.

    It blocks on geometry rather than on principle: against the real planner (ur5e, a wall at
    x = 250 mm) a 300 x 300 x 50 mm plate turns a 61-waypoint plan into no plan, while a 20 mm cube
    still plans.

    The box is not the part. Its lateral extents come from the jaw opening at the grasp, which
    measures the part at the grasp line and says nothing about the rest of it, and ``length_mm`` is a
    declared worst case rather than a measurement.
    """

    enabled: bool = Field(default=False)
    sphere_slots: int = Field(default=16, ge=4, le=128)
    length_mm: float = Field(default=120.0, gt=0.0, le=2000.0)
    lateral_margin_mm: float = Field(default=10.0, ge=0.0, le=500.0)


class PlannerMeshConfig(StrictModel):
    """A piece of the cell the planner routes around as the shape it is, rather than as a box.

    The one case that pays for itself is a container. As a box a tote is solid, so a cell that
    declares one can never reach into it; as a mesh it keeps its hollow and the planner takes the arm
    down between the walls. The same goes for a machine with an opening, a fixture with a slot, or
    anything else whose useful part is the space inside it.

    ⛔ THE GUARD CANNOT READ A MESH. The path guard and the one-shot guards work on boxes, so
    geometry declared here is known to the planner and to nothing else. That is not a gap to paper
    over silently: if this shape must also be enforced, declare the parts of it that matter as
    ``self_collision.fixtures`` boxes as well, and accept that a box around a hollow is solid.

    ``path`` is read by the planning sidecar, in its own process, so it has to be a path that process
    can open. Anything trimesh loads works: STL, OBJ, PLY, GLB.
    """

    #: Unique among every obstacle. The planner keys its world by name and a duplicate silently
    #: collapses two obstacles into one.
    name: str = Field(min_length=1)

    #: Mesh file the planning sidecar reads. Absolute, or relative to the working directory it runs
    #: in, which is the repository root.
    path: str = Field(min_length=1)

    #: Where the mesh origin sits in the base frame, millimetres.
    center_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)

    #: Rotation about base Z, degrees. A tote stands on a floor; if yours needs three angles, rotate
    #: the mesh once in a modelling tool and ship it that way.
    yaw_deg: float = Field(default=0.0, ge=-360.0, le=360.0)

    #: Uniform scale. 0.001 turns a mesh authored in millimetres into the metres the planner uses.
    scale: float = Field(default=1.0, gt=0.0, le=1000.0)


class PerceivedWorldConfig(StrictModel):
    """Turning what the cameras see into obstacles the planner routes around.

    This has no opt-in of its own. A cell that declares ``planning_world`` has said the planner
    should know its surroundings, and a photograph taken once at startup is not that: everything that
    arrived afterwards is invisible to it while looking exactly like empty space in every log there
    is. So it follows ``planning_world.enabled``, and ``enabled`` here is the way out rather than the
    way in, for a cell that wants its bench declared and nothing camera driven.

    What it costs, stated before the knobs. Every plan waits for a depth frame, converts it and
    registers a world, and a frame older than ``max_age_ms`` refuses the motion rather than planning
    against a cell that had time to change. That refusal is the point of the feature and it is also
    the way it will first annoy somebody: a camera that stalls stops the arm.
    """

    #: The way out, not the way in. ``planning_world.enabled`` is what turns this on.
    enabled: bool = Field(default=True)

    #: A frame older than this is not planned against, milliseconds.
    #:
    #: Measured from the shutter rather than from the conversion, because that is the time in which
    #: the cell could have changed. A frame with no capture time counts as unknown age and therefore
    #: as too old: a producer that does not stamp its frames has to be fixed, not trusted.
    max_age_ms: float = Field(default=500.0, gt=0.0, le=60_000.0)

    #: Read every nth pixel of every frame. 1 reads all of them.
    #:
    #: The cheapest lever there is, and not an approximation while it stays under the voxel size in
    #: image terms: at a metre a D435 pixel is about 1.6 mm, so a 10 mm voxel spans six pixels and
    #: every second pixel fills the same voxels the whole frame would. Measured on an 848 x 480
    #: frame: 33 ms at 1, 9 ms at 2, with the resulting box within a millimetre.
    pixel_stride: int = Field(default=2, ge=1, le=16)

    #: Cloud thinning before anything else, millimetres. Smaller keeps more shape and costs more time.
    voxel_size_mm: float = Field(default=10.0, gt=0.0, le=200.0)

    #: Edge length of a voxel in the distance field handed to the planner, millimetres. 0 sends none.
    #:
    #: This is the difference between the planner seeing the nearest few obstacles and seeing the
    #: whole cell. The box channel is bounded by the planner collision slots, so something always has
    #: to be left out and the report has to say what; a distance field carries everything the cameras
    #: saw at the resolution it is cut to.
    #:
    #: ⛔ The planner allocates the grid when it STARTS. The volume comes from the workspace limits
    #: and the resolution from this key, and both have to be settled before the sidecar spawns: a
    #: planner started without a voxel reservation refuses the field, and refusing the field refuses
    #: the motion rather than planning against half a cell.
    #:
    #: Measured on this repository: a 30 mm field over a two metre cell is 13.7 ms to build, 1.9 ms
    #: to register, and it blocked a path a plan otherwise took.
    voxel_field_mm: float = Field(default=0.0, ge=0.0, le=200.0)

    #: The grid the clustering runs on, millimetres. Two points in touching cells are one object, so
    #: this is also the gap at which two parts stop being one obstacle.
    cluster_voxel_mm: float = Field(default=25.0, gt=0.0, le=500.0)

    #: Below this a cluster is sensor noise rather than a thing.
    min_points: int = Field(default=12, ge=1, le=100_000)

    #: Grown on every side of every box, millimetres. A box that is exactly the measured hull is a
    #: box the planner will graze, and depth noise at an edge is one-sided.
    margin_mm: float = Field(default=15.0, ge=0.0, le=500.0)

    #: How many perceived boxes there is room for beside the declared ones.
    #:
    #: The planner reserves a fixed number of collision slots when it starts, so this is a real
    #: ceiling and not a preference. The nearest survive and the rest are counted and named in the
    #: report, because an obstacle the planner never received is one it will route straight through.
    max_boxes: int = Field(default=8, ge=0, le=64)

    #: Carry each box down to the support plane instead of stopping at the surface that was seen.
    #:
    #: A depth camera measures surfaces, not bodies: a part on a bench comes back as its top face and
    #: nothing else, and a planner routing around that sheet will drive a link through the part under
    #: it. Carrying the box down is the conservative reading. It has one cost worth knowing: a
    #: container whose rim the camera sees and that nobody declared becomes a solid block from rim to
    #: bench, and the cell can then never reach into it. Declaring the container as a mesh is the
    #: fix, and turning this off is the other one.
    floor_to_plane: bool = Field(default=True)

    #: How far above the declared plane a point still counts as the plane, millimetres.
    plane_clearance_mm: float = Field(default=5.0, ge=0.0, le=200.0)

    #: Radius of the capsules that stand for the arm's own links, millimetres.
    #:
    #: A fixed camera sees the robot, and a robot registered as an obstacle cannot move at all. The
    #: capsules are deliberately generous: a point wrongly kept is an obstacle that is not there, and
    #: a point wrongly dropped is a hole exactly where the arm is.
    self_radius_mm: float = Field(default=90.0, ge=0.0, le=1000.0)

    #: Radius of the last capsule, which covers the gripper and whatever it is holding, millimetres.
    tool_radius_mm: float = Field(default=150.0, ge=0.0, le=1000.0)

    @model_validator(mode="after")
    def _check_grids(self) -> PerceivedWorldConfig:
        if self.cluster_voxel_mm < self.voxel_size_mm:
            raise ValueError(
                f"perceived.cluster_voxel_mm ({self.cluster_voxel_mm}) is finer than "
                f"perceived.voxel_size_mm ({self.voxel_size_mm}). Clustering on a grid finer than the "
                "cloud splits one object into one cluster per point, and nothing downstream would "
                "say so."
            )
        return self


class PlanningWorldConfig(StrictModel):
    """What the trajectory planner is told about the cell, as axis-aligned boxes in the base frame.

    Separate from the guard's own fixture list only in what consumes it: the guard checks a single
    commanded configuration, the planner shapes the whole path. ``include_fixtures`` keeps the two one
    declaration, so a bin wall added for the guard is a bin wall the planner routes around.

    The planner takes three kinds of geometry and the choice between them is a real one. Boxes are
    what an operator writes down and what a refusal can name. A mesh is how a container keeps its
    hollow, so a cell can reach into a tote instead of treating it as a solid block. A distance field
    over a grid is how the whole scene arrives at once, with nothing left out for want of a collision
    slot: see :class:`PerceivedWorldConfig`, which turns what the cameras see into boxes, a field, or
    both, and is on whenever this block is.
    """

    enabled: bool = Field(default=False)
    support_plane: SupportPlaneConfig | None = Field(default=None)
    payload: AttachedPayloadConfig = Field(default_factory=AttachedPayloadConfig)
    include_fixtures: bool = Field(default=True)
    require_registration: bool = Field(default=True)
    perceived: PerceivedWorldConfig = Field(default_factory=PerceivedWorldConfig)
    meshes: list[PlannerMeshConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_mesh_names(self) -> PlanningWorldConfig:
        names = [m.name for m in self.meshes]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"planning_world.meshes declares {duplicates} more than once. The planner keys its "
                "world by name, so duplicates collapse into one obstacle and the others are silently "
                "absent."
            )
        return self

    @model_validator(mode="after")
    def _check_support_declared(self) -> PlanningWorldConfig:
        if self.enabled and self.support_plane is None:
            raise ValueError(
                "safety.planning_world.enabled is true but no support_plane is declared. Registering a "
                "world replaces the planner's own, boot table included, so enabling this without a "
                "bench would remove the only surface the planner currently knows about. Declare the "
                "bench you measured, or leave the block disabled."
            )
        return self


class RobotSafetyConfig(StrictModel):
    """Vendor-neutral safety surface.

    Each guard is a dedicated sub-block carrying its own gate: ``enforce`` on most, ``enabled`` on
    :attr:`planning_world` and :attr:`trajectory_check`, ``require_steady_before_motion`` on
    :attr:`dwell`. An operator can disable one guard without touching the others.

    Sub-blocks
    ----------
    * :attr:`limits`: the ``workspace_margin_mm`` consumed by the workspace guard.
    * :attr:`joint_limits`: per-axis joint hard limits and margin.
    * :attr:`ik_quality`: IK-solution quality checks.
    * :attr:`motion_continuity`: step size between consecutive commanded targets.
    * :attr:`payload`: mass, CoG and inertia envelope.
    * :attr:`self_collision`: link-link and link-fixture collision.
    * :attr:`dwell`: post-Stop dwell and steady-state gating.
    * :attr:`planning_world`: the boxes the trajectory planner routes around.
    * :attr:`trajectory_check`: gate every configuration of a planned path, not only its end.

    Not here: the emergency stop. It is a hardware and controller function, and nothing in this
    package can enable, disable, route or observe it.
    """

    limits: LimitsSafetyConfig = Field(default_factory=LimitsSafetyConfig)
    joint_limits: JointLimitSafetyConfig = Field(default_factory=JointLimitSafetyConfig)
    ik_quality: IkQualitySafetyConfig = Field(default_factory=IkQualitySafetyConfig)
    self_collision: SelfCollisionSafetyConfig = Field(
        default_factory=SelfCollisionSafetyConfig
    )
    payload: PayloadSafetyConfig = Field(default_factory=PayloadSafetyConfig)
    motion_continuity: MotionContinuitySafetyConfig = Field(
        default_factory=MotionContinuitySafetyConfig
    )
    dwell: DwellSafetyConfig = Field(default_factory=DwellSafetyConfig)
    planning_world: PlanningWorldConfig = Field(default_factory=PlanningWorldConfig)
    trajectory_check: TrajectoryCheckConfig = Field(default_factory=TrajectoryCheckConfig)

