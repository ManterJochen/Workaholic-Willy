"""Vendor-neutral safety-guard config schema."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .._base import ConfigPath, StrictModel

#: What the camera world grows every box it builds by, millimetres: ``perceived.CAMERA_BOX_MARGIN_MM``, the owner's
#: cell's 6, fixed with the rest of the camera world on 2026-10-17 (``RobotSafetyConfig._check_seen_box_clearance``).
#: Its own number here, because the config layer imports no runtime module; a test pins the two equal.
_SEEN_BOX_MARGIN_MM = 6.0

#: The least distance the exact guard keeps from a box a camera saw, millimetres (the owner: 5 on 2026-10-01, 3 on
#: 2026-10-05, 1 on 2026-10-12). Wherever only those boxes refuse the planner's world, the planner is asked again with
#: them set aside and the exact guard alone decides them, so a config asking less is refused at load
#: (``SelfCollisionSafetyConfig.perceived_min_distance_mm``). The box itself stands :data:`_SEEN_BOX_MARGIN_MM` past the
#: points it holds, and at the camera world's 10 mm voxel no pixel the camera measured lay less than 4.07 mm inside its
#: box over 3,400 noise-free synthetic scenes (review of 2026-09-30).
_SEEN_BOX_LEAST_MM = 1.0

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
      :attr:`SafetyReason.UNAVAILABLE` and rejects the motion: it fails closed.

    The guard runs on every cell since 2026-10-17 (the owner's cleanup took ``enforce``).

    ``min_distance_mm`` is the minimum signed distance between any two monitored
    shapes; closer than that counts as a collision.

    The capsule proxy's own geometry is no key since 2026-10-17 (the owner's cleanup): its fingers come from the
    hand's registry file, its base from the arm's description, every link is 60 mm round
    (``SelfCollisionGuard._tool_capsule``), and every path is judged whole first
    (``SafetyPreflight.gate_joint_path``); ``schema/_removed.py`` refuses a tree that still writes one of those keys.
    """

    backend: Literal["capsule", "fcl"] = Field(default="fcl")
    min_distance_mm: float = Field(default=3.0, ge=0.0, le=500.0)
    #: Clearance (mm) handed to the trajectory planner so it stops proposing configurations this guard
    #: will reject: cuRobo plans against a sphere model, the guard re-checks the exact meshes, and the
    #: two disagree. cuRobo returns UR5e plans at 9.44-9.47 mm against this guard's 10.000 mm, losing
    #: 3 of 10 picks to what reads as bad grasping. Not derived from ``min_distance_mm``: the margin a
    #: planner can absorb depends on how tightly its spheres fit that robot. On the refitted sphere maps
    #: no UR arm has a retract at all at 8 or 10 mm, the UR5 finds one at 4 and at 6 mm, and the UR3e
    #: measured about 6 mm as its ceiling, so 4.0 is the family's value (``config/robot/robot.sim.yaml``).
    #: It pads the planner's model only: where the padded spheres alone refuse a pose the exact guard
    #: accepts, on a pair that guard judges, the guard decides (``safety/planning/band.py``).
    #:
    #: The default ``None`` means undeclared, and a UR cell that plans with cuRobo then refuses to
    #: start a planner, naming this key (``safety/planning/margin.py``). It is not 0.0: 0.0 is a cell
    #: that measured this pair and chose no margin, and the two must not look the same. A consumer on
    #: any other path reads ``None`` as 0.0.
    planner_margin_mm: float | None = Field(default=None, ge=0.0, le=100.0)
    fixtures: list[FixtureBoxConfig] = Field(default_factory=list)
    #: The distance the exact mesh guard keeps from a box a camera saw, in mm. Such a box holds the
    #: camera world's thinned cloud, one point per 10 mm camera voxel, grown by the 6 mm every cell's
    #: camera world grows its boxes by (``perceived.CAMERA_BOX_MARGIN_MM``), so 2 keeps the arm 8 mm from
    #: every point of that cloud; a declared fixture is measured and keeps ``min_distance_mm``, and so does
    #: the arm against itself. This distance plus the margin has to reach ``min_distance_mm``, the step
    #: every path is sampled at, or the arm could reach a point of the cloud between two samples that both
    #: passed: a load refuses a config where it does not.
    #:
    #: A pixel the thinning dropped can lie nearer a box face than the margin: over 3,400 noise-free
    #: synthetic scenes, down to 10.9 mm past the points kept, at an oblique face or an edge (4.07 mm
    #: inside a box grown by 15, review of 2026-09-30). A measurement, not a bound: the voxel's diagonal,
    #: 17.3 mm, is longer than any margin. A flat face the box is square to loses nothing, since every
    #: pixel of it stands where its kept points stand. The camera world thins at 10 mm, where this was
    #: measured: at a 20 mm voxel a measured pixel lay 5.6 mm outside every box.
    #:
    #: 2 by default (the owner's 1 to 2 mm, 2026-10-12: over the replay bank's 116 recorded picks, 75 had a
    #: grasp the guard admits at 2 and at 1, 71 at 3, none fewer); at least 1, refused below at load:
    #: wherever only these boxes refuse the planner's world, the planner is asked again with them set
    #: aside, and this guard alone decides them.
    perceived_min_distance_mm: float = Field(default=2.0, ge=_SEEN_BOX_LEAST_MM, le=500.0)
    mesh_dir: ConfigPath | None = Field(default=None)

    # Which arm the guard judges is not a key here either. It is the model the arm reports, which is the model the
    # cell's selected block names (robot.ur.model, robot.sim.robot_model), and the guard gives it a UR chain where the
    # arm registry describes one (P0d D5); ``schema/_removed.py`` refuses a tree that still writes ``kinematics_model``.

    # The hand's mesh bundle and its coupling plate are not keys here. The guard derives both from
    # the hand the cell names, robot.gripper.model and robot.gripper.coupling_plates, through
    # ``safety.planning.hand.planner_hand``; ``schema/_removed.py`` refuses a tree that still writes
    # ``collision_mesh_variant`` or ``coupling_mm``.

    @field_validator("perceived_min_distance_mm", mode="before")
    @classmethod
    def _keep_the_least_distance_from_what_a_camera_saw(cls, value: Any) -> Any:
        """Refuse a number below :data:`_SEEN_BOX_LEAST_MM` with a sentence naming the key; anything else goes on to
        the field's own checks."""
        if isinstance(value, bool):
            return value
        try:
            number = float(value)
        except (TypeError, ValueError):
            return value
        if number < _SEEN_BOX_LEAST_MM:
            raise ValueError(
                f"robot.safety.self_collision.perceived_min_distance_mm ({number:g}) is below {_SEEN_BOX_LEAST_MM:g} mm. "
                "Wherever only the boxes a camera saw refuse the planner's world, the planner is asked again with them "
                "set aside, and the exact guard alone decides them at this distance, so it keeps at least "
                f"{_SEEN_BOX_LEAST_MM:g} mm from them (the owner, 2026-10-12). Set it to {_SEEN_BOX_LEAST_MM:g} or more."
            )
        return value


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

    The guard runs, and a real arm pushes ``mass_kg`` and ``cog_mm`` to its controller at connect, on every cell since
    2026-10-17 (the owner's cleanup took ``enforce``): a mass of 0, the shipped "not weighed yet" marker, is refused at
    connect, so a cell weighs what its flange carries.
    """

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


class PlannerMeshConfig(StrictModel):
    """A piece of the cell the planner routes around as the shape it is, rather than as a box.

    The one case that pays for itself is a container. As a box a tote is solid, so a cell that
    declares one can never reach into it; as a mesh it keeps its hollow and the planner takes the arm
    down between the walls. The same goes for a machine with an opening, a fixture with a slot, or
    anything else whose useful part is the space inside it.

    The guard cannot read a mesh. The path guard and the one-shot guards work on boxes, so
    geometry declared here is known to the planner and to nothing else. That is not a gap to paper
    over silently: if this shape must also be enforced, declare the parts of it that matter as
    ``self_collision.fixtures`` boxes as well, and accept that a box around a hollow is solid.

    ``path`` is read by the planning sidecar, in its own process, so it has to be a path that process
    can open. Anything trimesh loads works: STL, OBJ, PLY, GLB.
    """

    #: Unique among every obstacle. The planner keys its world by name and a duplicate silently
    #: collapses two obstacles into one.
    name: str = Field(min_length=1)

    #: Mesh file the planning sidecar reads. A relative path is read against the config folder, as every
    #: path in a tree is (:mod:`src.config.paths`), and reaches the sidecar absolute.
    path: ConfigPath = Field(min_length=1)

    #: Where the mesh origin sits in the base frame, millimetres.
    center_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)

    #: Rotation about base Z, degrees. A tote stands on a floor; if yours needs three angles, rotate
    #: the mesh once in a modelling tool and ship it that way.
    yaw_deg: float = Field(default=0.0, ge=-360.0, le=360.0)

    #: Uniform scale. 0.001 turns a mesh authored in millimetres into the metres the planner uses.
    scale: float = Field(default=1.0, gt=0.0, le=1000.0)


class PlanningWorldConfig(StrictModel):
    """What the trajectory planner is told about the cell, in the base frame: the bench, the meshes, and what the
    cameras see.

    The world is on wherever a ``support_plane`` is declared. Registering a world replaces the planner's own, boot table
    included, so a cell that declares no bench registers none and its planner keeps the world it booted with. Nothing
    else switches it since the owner's cleanup of 2026-10-17, every switch fixed at the owner's cell's: the cameras'
    world is built wherever it is on (``safety.planning.perceived``, its numbers every cell's), a pick and a place judge
    across their standoff, their carry and their drop in the world they hold rather than in a new frame, a world the
    planner confirms only in part refuses the motion, the guard's own fixtures stay the guard's (the cameras find what
    the parts stand on), and no part in the jaws is modelled: a closed hand is judged as an empty one, by the exact
    guard at full stroke.

    The planner takes three kinds of geometry and the choice between them is a real one. Boxes are what an operator
    writes down and what a refusal can name. A mesh is how a container keeps its hollow, so a cell can reach into a tote
    instead of treating it as a solid block. A distance field over a grid is how the whole scene arrives at once, with
    nothing left out for want of a collision slot: every cell's camera world sends the boxes it fits and a field of
    every column its cameras measured. Every plan waits for a depth frame, converts it and registers a world, and a frame
    too old refuses the motion rather than planning against a cell that had time to change, so a camera that stalls
    stops the arm.
    """

    support_plane: SupportPlaneConfig | None = Field(default=None)
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


class RobotSafetyConfig(StrictModel):
    """Vendor-neutral safety surface.

    Every guard family runs on every cell since 2026-10-17 (the owner's cleanup took the ``enforce`` switches), and
    the workspace margin, the joint limits, the IK quality and the motion continuity run at the numbers of the arm
    (``src/robot/safety/guard_numbers.py``: every real arm the owner's cell's), so their blocks left too. What a cell
    still says is what differs from cell to cell. The steady gate before every motion is the arm's own and fixed
    (``robot_arm.STEADY_TIMEOUT_S``; ``safety.dwell`` until 2026-10-17).

    Sub-blocks
    ----------
    * :attr:`payload`: mass, CoG and inertia envelope.
    * :attr:`self_collision`: link-link and link-fixture collision.
    * :attr:`planning_world`: the boxes the trajectory planner routes around.

    Not here: the emergency stop. It is a hardware and controller function, and nothing in this
    package can enable, disable, route or observe it.
    """

    self_collision: SelfCollisionSafetyConfig = Field(
        default_factory=SelfCollisionSafetyConfig
    )
    payload: PayloadSafetyConfig = Field(default_factory=PayloadSafetyConfig)
    planning_world: PlanningWorldConfig = Field(default_factory=PlanningWorldConfig)

    @model_validator(mode="after")
    def _check_seen_box_clearance(self) -> RobotSafetyConfig:
        """What the camera world keeps stays clear between two path samples: the guard's distance to a seen box,
        plus the margin the box is grown by (:data:`_SEEN_BOX_MARGIN_MM`), is at least the step every path is sampled
        at, ``self_collision.min_distance_mm``.

        What it keeps is the thinned cloud, one point per 10 mm camera voxel, and a pixel the thinning dropped can lie
        nearer a box face than the margin, which this sum does not see: at that voxel, where a seen box held nearer than
        the step was measured, no pixel lay less than 4.07 mm inside its box (review of 2026-09-30).

        Checked whether or not a camera world is on, so turning one on later cannot bring the gap in with it.
        """
        seen = float(self.self_collision.perceived_min_distance_mm)
        step = float(self.self_collision.min_distance_mm)
        if seen + _SEEN_BOX_MARGIN_MM < step:
            raise ValueError(
                f"safety.self_collision.perceived_min_distance_mm ({seen:g}) plus the {_SEEN_BOX_MARGIN_MM:g} mm every "
                f"camera world grows its boxes by is {seen + _SEEN_BOX_MARGIN_MM:g} mm, less than "
                f"safety.self_collision.min_distance_mm ({step:g}), the step every path is sampled at: between two "
                "samples that both passed, the arm could reach a surface the camera saw. Raise perceived_min_distance_mm "
                f"to {step - _SEEN_BOX_MARGIN_MM:g} or lower min_distance_mm to {seen + _SEEN_BOX_MARGIN_MM:g}."
            )
        return self
