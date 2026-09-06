"""The generator's own config tree, on the robot's conventions and deliberately not in its tree.

`config/` is the robot's configuration: a frozen, strict, profile-layered tree, because changing
the robot's configuration is a safety-relevant act. This generator is a tool. Giving it a block in
there would put every new scene knob into `AppConfig` and re-bless the robot's pinned configuration,
a cost with no benefit, so it gets its own tree under the same rules: Pydantic v2,
``extra='forbid'``, frozen, explicit units in the field names.

The defaults describe the cell this repository drives, so the in-domain case is the cheap one: a
UR5e workspace around x = 450 mm, the wrist camera plus the two raised oblique cameras that see
past the arm from either side, path tracing on, and 300 scenes.
"""

from __future__ import annotations

import warnings
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ArmConfig",
    "AssetSourcesConfig",
    "CameraRigConfig",
    "DatagenConfig",
    "DepthNoiseConfig",
    "FamilyMixConfig",
    "OutputConfig",
    "RandomizationConfig",
    "RenderConfig",
    "StrictModel",
    "WorkspaceConfig",
]


class BinExceedsWorkspace(UserWarning):
    """The default bin reaches past the workspace this cell declares it can reach.

    Its own category so a caller can silence exactly this. Raised rather than logged for the same
    reason :class:`StoredConfigDrift` is: this module is imported by `ProcessPoolExecutor` workers,
    and `create_logger` opens its file at import time.
    """


class StoredConfigDrift(UserWarning):
    """A stored provenance config carries a key this build no longer has.

    Its own category so a caller can silence exactly this and nothing else, and so the message is
    distinguishable from a typo in a live config, which still refuses.
    """


class StrictModel(BaseModel):
    """Frozen, no unknown keys: the same contract the robot config uses, for the same reasons."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class BinConfig(StrictModel):
    """The container the `bin` family places into, in millimetres.

    A cell has the bin it has. These proportions are a small KLT. A dataset built for a cell whose
    bin is 600 x 400 teaches a model the walls of a bin that cell does not own, and bin walls are
    load-bearing: a large part of the finger collisions in a bin scene are against a wall.

    The walls are rendered and collided from one description. A wall the camera sees but the planner
    does not is worse than no wall, so changing a number here moves both together or neither.
    """

    #: Inner footprint (x, y). Objects are placed inside this, walls just outside it.
    inner_mm: tuple[float, float] = (300.0, 200.0)
    wall_height_mm: float = Field(default=147.0, gt=0.0)
    wall_thickness_mm: float = Field(default=8.0, gt=0.0)
    #: How far inside the walls objects may be placed. It is a clearance for the jaw, so a cell with
    #: a wider gripper needs it wider.
    object_inset_mm: float = Field(default=30.0, ge=0.0)


class PlacementConfig(StrictModel):
    """How much room the families leave between objects, and how hard they try.

    These are the difficulty knobs of the corpus. The margins decide whether a scene is reachable
    clutter or a jam; the try budgets decide whether a dense scene is honestly reported as too dense
    or silently looped over. Changing one changes what a model sees, so they are recorded in the
    provenance stamp.
    """

    #: Clear space between neighbours in the `sparse` family. Wide enough that an 80 mm pre-open jaw
    #: reaches past a target's surface without touching its neighbour.
    sparse_margin_mm: float = Field(default=60.0, ge=0.0)
    #: `packed` objects may touch but not interpenetrate; a small positive margin keeps the settle
    #: stable. Also used inside the bin, where the walls already supply the crowding.
    packed_margin_mm: float = Field(default=2.0, ge=0.0)
    #: Placements to try before accepting a tighter fit than asked for. A budget, not a promise: a
    #: dense family in a small workspace genuinely runs out of room, and saying so beats looping.
    max_place_tries: int = Field(default=240, ge=1)
    #: Pile objects are released in a column; the first hovers this far above the table, the rest are
    #: spaced by their own size rather than by a constant.
    pile_base_clearance_mm: float = Field(default=6.0, ge=0.0)
    #: Clear air between one dropped object and the next.
    pile_gap_mm: float = Field(default=8.0, ge=0.0)
    #: Candidate (x, y) per object; the one giving the lowest rest height wins. This is what makes a
    #: heap compact. A single random draw is legal but stacks into a tower instead of a heap.
    pile_drop_tries: int = Field(default=24, ge=1)


class WorkspaceConfig(StrictModel):
    """Where objects may be placed, in BASE-frame millimetres.

    The cell defaults are the reachable, camera-covered patch the cell picks in, not the arm's full
    envelope. A scene outside it renders fine and then fails every reach, which reads as a grasping
    problem and is a geometry one.
    """

    center_mm: tuple[float, float] = (450.0, 0.0)
    #: Half-extents of the placement area. 150 x 150 mm covers the validated pick patch with margin.
    half_extents_mm: tuple[float, float] = (150.0, 150.0)
    table_height_mm: float = 0.0
    #: The table's edge length, and the only place it is declared. `render/isaac.py`,
    #: `render/noengine.py` and `verify_robot.py` all read it from here. Two engines that disagree
    #: here render the same scene onto two different tables, and the check that an object is on the
    #: table then answers differently per engine.
    table_size_mm: float = Field(default=1200.0, gt=0.0)
    #: The bin the `bin` family places into. See :class:`BinConfig`.
    bin: BinConfig = Field(default_factory=BinConfig)


class FamilyMixConfig(StrictModel):
    """How many scenes of each family, as relative weights, and how many objects each holds.

    Counts are ranges because a family with a fixed object count stops being a family and becomes a
    template. The count is one of the few knobs that changes difficulty without changing anything
    else, so it is worth varying inside a family rather than across datasets.
    """

    sparse_weight: float = Field(default=1.0, ge=0.0)
    packed_weight: float = Field(default=1.0, ge=0.0)
    pile_weight: float = Field(default=1.0, ge=0.0)
    bin_weight: float = Field(default=1.0, ge=0.0)

    #: How an object is oriented when it is placed flat. `upright` spawns every object on its base
    #: with a random yaw, which is what `sparse`, `packed` and `bin` do. `random` draws a full
    #: orientation.
    #:
    #: An object placed upright on its own base sits in a stable equilibrium, so a settle tends to
    #: leave it there. That costs grasp yield, because a tipped object is far more likely to admit a
    #: jaw grasp than an upright one, and a corpus of standing objects does not look like a
    #: bin-picking cell, where parts are dumped.
    #:
    #: The orientation belongs in the spec rather than in an engine. Reaching for a drop height
    #: instead would make the variety a property of the physics, so `engine: none` would produce a
    #: different distribution from `mujoco` for the same config. An orientation in the spec is read
    #: identically by every backend: MuJoCo settles it, and `none` seats it and refuses what will
    #: not stand.
    #:
    #: Default `upright`, which is what an existing config already generates.
    flat_orientation: Literal["upright", "random"] = "upright"

    sparse_objects: tuple[int, int] = (2, 6)
    packed_objects: tuple[int, int] = (4, 9)
    pile_objects: tuple[int, int] = (5, 13)
    bin_objects: tuple[int, int] = (5, 12)

    #: How much of the workspace the pile's drop covers, as a fraction of its smallest half-extent.
    #:
    #: A tighter footprint does not pack the heap. It raises the drop column instead, and a taller
    #: tower scatters, so tightening costs accepted scenes and buys no extra occlusion. The default
    #: 0.75 is the top of the useful range and still keeps the drop clear of the workspace edge,
    #: rather than spawning objects on the boundary.
    #:
    #: The occlusion this family exists for comes from the arm, not from the pile. Reach for
    #: ``arm.mode`` to make a dataset harder, not for this.
    pile_footprint_fraction: float = Field(default=0.75, gt=0.0, le=1.0)
    #: How the families space objects and how hard they try to place them. See
    #: :class:`PlacementConfig`.
    placement: PlacementConfig = Field(default_factory=PlacementConfig)

    @model_validator(mode="after")
    def _at_least_one_family(self) -> "FamilyMixConfig":
        total = self.sparse_weight + self.packed_weight + self.pile_weight + self.bin_weight
        if total <= 0.0:
            raise ValueError(
                "every scene-family weight is zero, so no scene could be generated. Set at least one "
                "of sparse_weight / packed_weight / pile_weight / bin_weight above zero."
            )
        return self


class CameraSpec(StrictModel):
    """One camera of a customer's own cell, declared rather than chosen from a fixed list.

    The mount is declared here, never inferred. `CameraRigConfig.views` is four literal names, and a
    name is a label: a label should not decide whether the robot has to drive somewhere. Declaring
    the mount is what lets a cell carry more than three fixed cameras, more than one wrist camera,
    and names of its own.

    `CameraRigConfig.cameras` is `None` unless a caller sets it, and then the four-name rig runs
    unchanged.
    """

    #: Any name. It becomes the image filename prefix and the key in the view record.
    name: str

    #: How the camera gets to its pose. Fixed is authored directly; wrist must be reached by the arm.
    mount: Literal["fixed", "wrist"] = "fixed"

    #: Where it sits, in BASE millimetres. Required for a fixed camera. For a wrist camera it may be
    #: omitted, and then the viewpoint is sampled on the hemisphere.
    position_mm: tuple[float, float, float] | None = None

    #: What it looks at, in BASE millimetres. Defaults to the workspace centre at table height.
    look_at_mm: tuple[float, float, float] | None = None

    #: Roll, as an up direction. The look-at convention derives roll from world-up when this is None,
    #: which is right for a rig nobody rolled and wrong for a camera mounted on its side.
    up_hint: tuple[float, float, float] | None = None

    #: Per-camera overrides. `None` takes the rig's value, so a mixed rig is expressible: a wide
    #: wrist camera beside narrower fixed ones fits in one dataset.
    resolution: tuple[int, int] | None = None
    horizontal_fov_deg: float | None = Field(default=None, gt=0.0, lt=180.0)

    @model_validator(mode="after")
    def _a_fixed_camera_needs_a_position(self) -> "CameraSpec":
        if self.mount == "fixed" and self.position_mm is None:
            raise ValueError(
                f"camera {self.name!r} is fixed and has no position_mm. A fixed camera is authored "
                f"at its pose, so there is nothing to derive it from. A wrist camera may omit it: "
                f"its viewpoint is sampled on a hemisphere the arm then has to reach."
            )
        return self


class CameraRigConfig(StrictModel):
    """Which views to render. Each is a direct multiplier on path-tracing cost, so the list is explicit.

    ``oblique_left`` and ``oblique_right`` are raised and tilted to either side, aimed at the
    workspace centre, so the arm, which is the dominant occluder in this cell, never hides the scene
    from both at once. ``wrist`` is the eye-in-hand view and the one the renderer may fail to reach,
    because it needs the arm to get there.
    """

    #: Any subset, in render order. Empty is refused: a scene with no view is not a datapoint.
    views: tuple[Literal["wrist", "oblique_left", "oblique_right", "overhead"], ...] = (
        "wrist", "oblique_left", "oblique_right",
    )
    resolution: tuple[int, int] = (640, 480)
    horizontal_fov_deg: float = Field(default=47.0, gt=0.0, lt=180.0)

    #: The oblique pair: lateral offset from the workspace centre, height above the table, and how far
    #: back along +x they sit. Defaults look down at roughly 50 deg: an overlook, not a top view.
    oblique_lateral_mm: float = 450.0
    oblique_height_mm: float = 600.0
    oblique_setback_mm: float = 0.0

    #: The wrist viewpoint is sampled on a hemisphere around the workspace centre.
    wrist_radius_mm: tuple[float, float] = (260.0, 340.0)
    wrist_elevation_deg: tuple[float, float] = (45.0, 80.0)

    overhead_height_mm: float = 1000.0

    #: A customer's own cell, declared camera by camera. `None` keeps the four-name rig above; a
    #: non-empty tuple replaces it entirely and `views` is then ignored.
    #:
    #: The rig above describes one physical setup: two obliques at a mirrored offset, an overhead
    #: and one wrist view, all sharing one resolution and one field of view, all aimed at the same
    #: point. Every one of those is a property of that cell. A cell with four fixed cameras, or with
    #: two cameras that are not mirror images of each other, or with a calibrated extrinsic that was
    #: measured rather than derived, needs this field to express itself.
    #:
    #: The mount comes with the camera. See `CameraSpec`: a name is a label, and a label may not
    #: decide whether the arm has to drive somewhere.
    cameras: tuple[CameraSpec, ...] | None = None

    #: An eye-in-hand camera belongs on the arm.
    #:
    #: `IsaacRenderer.render` places a wrist viewpoint with no arm in the scene when
    #: `render.arm.mode` is `absent`: the pose is a valid camera pose and the dataset records the arm
    #: mode, so a consumer can tell. It is still almost never what the caller meant, because an
    #: eye-in-hand dataset whose images contain no hand is a different dataset wearing the same name.
    #:
    #: With this False, that combination is refused at config time rather than silently rendered.
    #: Setting it True is how a caller says they meant it.
    allow_unmounted_wrist: bool = False

    @model_validator(mode="after")
    def _views_are_unique_and_present(self) -> "CameraRigConfig":
        if self.cameras is not None:
            if not self.cameras:
                raise ValueError(
                    "camera_rig.cameras is an empty tuple. Leave it None to use the built-in rig, "
                    "or declare at least one camera: a scene with no view is not a datapoint."
                )
            names = [camera.name for camera in self.cameras]
            if len(set(names)) != len(names):
                raise ValueError(f"camera_rig.cameras has duplicate names: {sorted(names)}")
            return self
        if not self.views:
            raise ValueError(
                "camera_rig.views is empty; a scene with no view produces no image and no label. "
                "Pick at least one of: wrist, oblique_left, oblique_right, overhead."
            )
        if len(set(self.views)) != len(self.views):
            raise ValueError(f"camera_rig.views contains duplicates: {self.views}")
        return self


class AssetSourcesConfig(StrictModel):
    """Which asset sources may be drawn from, as weights. Every source passes the licence audit.

    Procedural leads by default because it is the only source with no third-party licence in the
    chain at all, with exact known geometry and unlimited variation. The others buy realism the
    procedural library cannot fake.
    """

    procedural_weight: float = Field(default=1.0, ge=0.0)
    gso_weight: float = Field(default=0.0, ge=0.0)
    objaverse_weight: float = Field(default=0.0, ge=0.0)
    ycb_weight: float = Field(default=0.0, ge=0.0)
    #: The weights that let `scenes/layout.py` draw the two smaller collections. `thingi10k` and
    #: `asos` are in `SUPPORTED_SOURCES`, `datagen.assets.fetch` downloads them and `screen-meshes`
    #: grades them; without a weight here they are fetched, normalised, screened and never placed,
    #: and nothing says so. Default 0.0 like every other mesh source.
    thingi10k_weight: float = Field(default=0.0, ge=0.0)
    asos_weight: float = Field(default=0.0, ge=0.0)
    #: The customer's own parts. `gso` and `ycb` are household objects from two research datasets;
    #: a bin-picking cell runs on the parts that cell handles, and no public dataset contains them.
    #: This is the weight that puts them in the scenes. Fetch them with
    #: `python -m datagen.assets --fetch --source custom --from <dir> --license own`, then raise this.
    #:
    #: Default 0.0 like every other mesh source, so a machine with no custom meshes is unaffected.
    custom_weight: float = Field(default=0.0, ge=0.0)
    #: Weight for composite objects: a mug with a handle, a hammer with a grip. These are the only
    #: source with per-part grasp labels, which is what a part-aware model trains on. Default 0.0,
    #: so a run that wants part supervision asks for it.
    composite_weight: float = Field(default=0.0, ge=0.0)

    #: Longest-axis limit, in millimetres, for a real mesh to be placeable. A mesh bank holds objects
    #: far larger than the 300 x 300 mm placement area, so an unfiltered draw can put an object
    #: larger than the whole workspace into a scene.
    #:
    #: The default 180 mm still fits two or three objects side by side and keeps most of them
    #: jaw-graspable. A cell with a bigger table has a different right answer, which is why this is a
    #: knob.
    max_mesh_extent_mm: float = Field(default=180.0, ge=20.0, le=1000.0)

    #: The axis a jaw actually closes on. `max_mesh_extent_mm` above bounds the longest axis, which
    #: is about fitting the workspace. This bounds the shortest, which is about fitting the gripper,
    #: and they are not substitutes: a 250 mm broom handle is easy to grasp and a 90 mm cube is
    #: impossible at an 85 mm aperture.
    #:
    #: The share of objects carrying a jaw label falls off a cliff exactly at the aperture, so
    #: dropping the meshes past it lifts the yield of a corpus without costing the variety that a
    #: tighter length limit costs. `None`, the default, applies no such bound.
    max_jaw_span_mm: float | None = Field(default=None, ge=10.0, le=1000.0)

    #: Restrict the mesh draw to exactly these asset ids. Empty, the default, draws the whole bank.
    #:
    #: An evaluation dataset drawn from the same assets the model trained on can never report a
    #: held-out number, whatever the row says. This field is how a dataset is built from meshes the
    #: model never saw. For a customer, the same key reads "generate scenes from exactly these three
    #: parts of mine".
    mesh_asset_ids: tuple[str, ...] = Field(default=())

    #: Refuse to fall back to procedural when a weighted mesh source has nothing placeable.
    #:
    #: With the default False, an empty bank logs one warning and the draws become procedural, so a
    #: scene still happens rather than the generator looking broken. That is right for an ordinary
    #: run and wrong for a held-out one: the procedural families are in the training corpus, so the
    #: fallback silently refills a held-out dataset with objects the model has seen, at one warning
    #: for the whole render.
    #:
    #: Set it true whenever the point of the dataset is which assets are in it.
    refuse_procedural_fallback: bool = False

    #: The same restriction, read from a JSON file: a list of ids, or `{source: [ids]}`. Both forms
    #: exist because they answer different questions. An inline list is right for three custom parts
    #: and unusable for a held-out set of hundreds. When both are given the union is used.
    mesh_asset_ids_path: str = Field(default="")
    #: The labeller's verdict, replacing the bounding-box jaw filter where it is given. Written by
    #: `datagen screen-meshes`: one row per mesh with a `jaw` and a `suction` count, taken alone and
    #: upright at the density the corpus will be labelled with.
    #:
    #: `max_jaw_span_mm` asks whether the object's hull fits between the fingers; a jaw closes on a
    #: line through the object. The box proxy therefore both admits meshes that carry no grasp and
    #: refuses meshes that do, sometimes by a single millimetre of span.
    #:
    #: Empty keeps the proxy.
    jaw_screen_path: str = Field(default="")

    #: How PhysX may approximate a scanned mesh when it collides. Not a fidelity preference: it
    #: decides whether the physics and the grasp labels describe the same object. PhysX cannot
    #: simulate a dynamic rigid body against raw triangles, so one of these is unavoidable, and the
    #: labeller intersects the exact closed surface either way.
    #:
    #: A scene that will not settle is refused, so an unstable collider costs throughput rather than
    #: producing wrong data. `sdf` costs both: it settles far fewer scenes, it is the slowest, and
    #: its failures are tens of millimetres of residual motion rather than a near miss of the 0.5 mm
    #: stability tolerance. Its `PhysxSchema.PhysxSDFMeshCollisionAPI` is applied correctly; that is
    #: what an SDF collider on scanned meshes does here.
    #:
    #: Default ``convexDecomposition``: the same yield as a convex hull for a little more wall
    #: clock, and the only one of the two that can represent a concavity at all. A hull fills a
    #: mug's handle. ``convexHull`` stays reachable because a bin of boxes has no concavity to lose
    #: and may as well take the cheapest collider.
    mesh_collision: Literal["sdf", "convexDecomposition", "convexHull"] = "convexDecomposition"

    #: Procedural families to draw from. All four by default.
    procedural_families: tuple[Literal["primitive", "industrial", "packaging", "vessel"], ...] = (
        "primitive", "industrial", "packaging", "vessel",
    )

    #: Composite families to draw from. Listed explicitly rather than defaulting to "all" so the
    #: provenance stamp records which families a corpus could contain, not only which it happened to.
    #:
    #: These are the only objects in the corpus with named parts, so they are the only source of a
    #: handle-grasp measurement. Two held-out questions live here and they need different setups: new
    #: instances of a known family (a mug of unseen dimensions; a disjoint seed is enough and the
    #: family list stays full) against a family whose part layout was never trained on (drop
    #: "hammer" here, render an evaluation corpus of nothing but "hammer"). The second is the
    #: stronger claim and the only one that needs this field.
    #:
    #: The order is load-bearing. The sampler draws with `rng.choice(list(available))`, which selects
    #: by index, so a different order draws a different family at every composite slot, and a
    #: provenance stamp that carries no such key falls back to this default. It must therefore stay
    #: `sorted(COMPOSITE_KINDS)`, so a stamp written without the key rebuilds what it recorded.
    composite_kinds: tuple[
        Literal["bucket", "hammer", "jug", "mug", "pan"], ...
    ] = ("bucket", "hammer", "jug", "mug", "pan")

    #: Whether a composite's asset id names its shape or its slot in the scene. Default "slot".
    #:
    #: Under "slot" many differently shaped composites share one id, and the manifest keeps one of
    #: them, so the renderer settles the scene against a different object than the layout drew and
    #: the placements are spaced for the wrong bounding box. The slot index also caps diversity: a
    #: corpus holds at most (objects per scene) x (kinds) distinct composites, however many scenes
    #: are rendered.
    #:
    #: "unique" changes every composite id and therefore the manifest hash, so a corpus rendered
    #: under one setting cannot be rebuilt from its own provenance under the other, which the
    #: evaluation refuses outright. Choose it for a new render; never flip it under an existing
    #: corpus.
    composite_id: Literal["slot", "unique"] = "slot"

    @model_validator(mode="after")
    def _something_to_draw_from(self) -> "AssetSourcesConfig":
        """Refuse a config with nothing to draw from.

        The total is derived from every field whose name ends in `_weight`, never enumerated by
        hand: a hand-written sum is a second declaration of what the sources are and drifts the
        moment one is added. Adding a weight field is therefore the whole change.
        """
        total = sum(float(getattr(self, name)) for name in type(self).model_fields
                    if name.endswith("_weight"))
        if total <= 0.0:
            raise ValueError("every asset-source weight is zero; there would be nothing to place.")
        if self.procedural_weight > 0.0 and not self.procedural_families:
            raise ValueError(
                "procedural_weight > 0 but procedural_families is empty; nothing to generate."
            )
        if self.composite_weight > 0.0 and not self.composite_kinds:
            raise ValueError(
                "composite_weight > 0 but composite_kinds is empty; nothing to generate."
            )
        return self


class RandomizationConfig(StrictModel):
    """Domain-randomisation ranges, resolved at layout time so a seed fully determines a scene."""

    light_elevation_deg: tuple[float, float] = (30.0, 90.0)
    light_intensity: tuple[float, float] = (800.0, 3000.0)
    dome_intensity: tuple[float, float] = (100.0, 400.0)
    table_color_min: tuple[float, float, float] = (0.15, 0.15, 0.18)
    table_color_max: tuple[float, float, float] = (0.55, 0.55, 0.60)
    object_color_min: tuple[float, float, float] = (0.05, 0.05, 0.05)
    object_color_max: tuple[float, float, float] = (0.95, 0.95, 0.95)


class WristMountSpec(StrictModel):
    """A customer's own measured eye-in-hand mount, in the wrist link's frame, millimetres.

    A declaration, not a lookup. `robots.py::_measured_mount` returns a mount for `ur5e` alone and
    refuses every other model, and that gate stays: its constants are the mount geometry declared in
    `src/willy_sim/scene/cameras.py`, chosen so the gripper does not occlude the view and validated
    on that arm, and the UR e-series sharing a wrist-3 flange is not the same as a mount having been
    measured. Falling back to another robot's numbers would put the camera in the wrong place and
    record it as if it belonged there. This type is how a cell that has run its own calibration says
    so, without editing that file, which datagen may only read from.

    `source` is required and is carried into the dataset. A mount is a measurement, and a
    measurement without a provenance is a number, so the stamp of every scene keeps a declared mount
    distinguishable from the one this repository measured.
    """

    #: Where the camera sits on the wrist link, in that link's own frame.
    offset_mm: tuple[float, float, float]

    #: What it aims at, as a point rather than a direction, because that is the form the measurement
    #: is taken and validated in.
    aim_target_mm: tuple[float, float, float]

    #: Roll. Without it the look-at derives roll from world-up, which is wrong for a camera mounted
    #: on its side.
    up_hint: tuple[float, float, float] = (0.0, -1.0, 0.0)

    #: Metres. A wrist camera sits close to what it looks at, so this is not the rig's near clip.
    near_clip_m: float = Field(default=0.01, gt=0.0)

    #: How this was obtained, in words. Written into the dataset.
    source: str


class ArmConfig(StrictModel):
    """Whether the robot is in the picture, and how it got there.

    The arm is this cell's dominant occluder, so a dataset rendered without it does not contain the
    failure the two oblique cameras exist to survive. An arm placed at an arbitrary pose is worse
    than none: it would occlude in ways the real cell never does.

    * ``absent``: no robot. Correct for ``domain: tabletop``, and for general-purpose data.
    * ``posed``: the arm is placed by real inverse kinematics at the configuration that puts the
      wrist camera at the sampled viewpoint, and that configuration is checked by the same
      self-collision guard a real pick uses. It does not move; it stands where it would stand.
    * ``driven``: the full cell boots and the arm is driven there through planning and the safety
      pipeline. Not implemented, and named here rather than silently unavailable.
    """

    mode: Literal["absent", "posed", "driven"] = "posed"
    #: Which robot stands there. The key is the stack's canonical model string, the same one that
    #: selects the DH table, the self-collision model and the cuRobo config, so a dataset naming
    #: ``ur3e`` was generated by the kinematics of a UR3e and not merely by its picture.
    robot_model: str = "ur5e"
    #: Where the wrist camera pose comes from. It is a trade rather than a preference:
    #:
    #: * ``target_ik``: draw the viewpoint, solve IK for it. The viewpoint distribution stays the one
    #:   that was designed; unreachable draws are handled by the rule in :mod:`datagen.render.views`.
    #: * ``joint_fk``: draw a joint configuration around park, put the camera wherever forward
    #:   kinematics lands. No IK step, so no viewpoint is ever unreachable; it can still be refused
    #:   for self-collision, for standing inside the objects, or for pointing away from the scene,
    #:   and a scene whose every draw is refused parks exactly as ``target_ik`` does. The viewpoints
    #:   are then whatever the arm finds comfortable, which is the bias that makes a model believe
    #:   awkward angles do not happen.
    #:
    #: Both exist so that bias is measurable rather than assumed.
    viewpoint_source: Literal["target_ik", "joint_fk"] = "target_ik"
    #: Which views the arm appears in. It physically occludes all of them, which is the default and
    #: the honest case. A different cell, with the arm out of frame or a fixed overhead rig, is a
    #: real configuration, so it is selectable rather than assumed away.
    visible_in_views: Literal["all", "wrist_only"] = "all"
    #: How many wrist viewpoints to try before giving up, consulted only when the policy says
    #: resample; see :mod:`datagen.render.views`. A viewpoint counts as unusable when IK fails, when
    #: the configuration self-collides, or when the posed arm would sit inside a scene object.
    max_viewpoint_attempts: int = Field(default=12, ge=1)
    #: Reject a posed configuration whose closest non-adjacent link pair is below this. The same
    #: margin the continuous guard in `src/robot/safety/continuous_monitor.py` uses, so "the arm
    #: could hold this pose" means the same thing in a dataset as it does in a pick.
    self_collision_margin_mm: float = Field(default=8.0, ge=0.0)

    #: The customer's own measured eye-in-hand mount. `None` uses the one this repository measured,
    #: which exists for `ur5e` alone; see `WristMountSpec` for why that gate stays.
    wrist_camera_mount: WristMountSpec | None = None


class DepthNoiseConfig(StrictModel):
    """Isaac's depth is exact; a real one is not. Both are written, so noise stays a variable."""

    enabled: bool = True
    sensor: Literal["realsense_d435"] = "realsense_d435"


class RenderConfig(StrictModel):
    """Path tracing is the default; the raster path stays available for one specific job.

    Silhouettes do not depend on light transport, so the per-object solo passes that yield the
    occlusion ratio run on the cheap renderer while the beauty pass is path-traced. It is the same
    geometry, and it keeps the visibility label from multiplying the dataset's cost by the object
    count.
    """

    #: Which engine renders. Not a fidelity knob but an installability one: scene generation is the
    #: user's job, and Isaac Sim is a multi-GB NVIDIA install that in many companies needs an
    #: approval process. See `datagen/render/engine.py` for the contract a backend owes: depth in
    #: mm, per-object silhouettes, settled poses, and the camera pose and K it used.
    #:
    #: It is stamped into `provenance.json` and must stay there. Two engines never produce
    #: byte-identical settles, so a corpus that cannot say which one made it silently confounds
    #: engine with scene family in every number drawn from it.
    engine: Literal["isaac", "mujoco", "none"] = "isaac"

    mode: Literal["pathtrace", "raster"] = "pathtrace"
    samples_per_pixel: int = Field(default=64, ge=1)

    #: Render depth and masks only, and no colour image at all. Off by default.
    #:
    #: It is the throughput lever for a training corpus, and a lever rather than the default because
    #: the two consumers want different things. The learned scorer is depth and point-cloud native,
    #: with no RGB entering the net, while the VLM and auto-annotation passes need pictures, and so
    #: does anyone opening the folder to look.
    #:
    #: What it skips: the path-traced beauty pass and, with it, the colour-buffer retry that exists
    #: because Isaac occasionally returns an entirely black frame. Depth and segmentation come off
    #: the raster path, because silhouettes and ranges do not depend on light transport, which is
    #: the argument `measure_visibility` already runs on.
    #:
    #: `datagen.verify` reads this back from the provenance stamp and drops its "the colour image is
    #: an image" check for such a dataset. That coupling is deliberate: without it a depth-only run
    #: would fail the gate that catches a scene whose colour frames are entirely black, and the
    #: tempting fix would be to weaken the gate for everyone.
    depth_only: bool = False
    #: Physics settle before anything is rendered, then a stability check; an unstable scene is dropped
    #: rather than recorded, because its labels would describe a pose it is no longer in.
    settle_steps: int = Field(default=180, ge=1)
    stability_check_steps: int = Field(default=30, ge=1)
    stability_tolerance_mm: float = Field(default=0.5, gt=0.0)
    #: Visibility (unoccluded silhouette) is measured on the raster path. Off means no visibility
    #: label.
    measure_visibility: bool = True
    #: A second, longer settle for scenes that are still moving, before they are rejected. Piles
    #: legitimately need it; rejecting on the first check would throw away the interesting family.
    extra_settle_steps: int = Field(default=240, ge=0)
    #: Rolling resistance and air drag, which PhysX does not model for rigid bodies and reality does.
    #: Without them the pile family rarely settles: the objects creep rather than explode, and a
    #: round object with no rolling resistance rolls until the table runs out, so the scene is
    #: rejected for the free fall that follows. These two numbers are an approximation of a real
    #: effect, exposed here so the approximation is visible and tunable.
    linear_damping: float = Field(default=0.05, ge=0.0)
    angular_damping: float = Field(default=1.0, ge=0.0)
    #: Contact between objects and the table. Without a physics material PhysX's default lets objects
    #: slide off the table after a drop. These are ordinary values for cardboard and plastic on a
    #: work surface, and restitution 0 because the parts this generator describes do not bounce.
    friction_static: float = Field(default=0.7, ge=0.0)
    friction_dynamic: float = Field(default=0.6, ge=0.0)
    restitution: float = Field(default=0.0, ge=0.0, le=1.0)
    arm: ArmConfig = Field(default_factory=ArmConfig)
    depth_noise: DepthNoiseConfig = Field(default_factory=DepthNoiseConfig)


class OutputConfig(StrictModel):
    """Where a dataset lands.

    ``root`` defaults inside the repository and is gitignored, so a run with no arguments still
    works and lands somewhere obvious; point it at a real data drive for anything large.

    There is no format key, because there is no format choice. `DatasetWriter` writes `index.jsonl`
    unconditionally, and datagen has no COCO exporter at all. `StrictModel` forbids extra keys, so a
    config file that still sets `jsonl` or `coco` is refused rather than silently ignored: the
    refusal names the key, and the answer is to delete the line.
    """

    root: str = "data/datagen"
    #: Written next to every dataset: every asset that reached a pixel, with licence and author.
    #: CC-BY obliges attribution for derivative works, and a render is one.
    attribution_file: bool = True


class DatagenConfig(StrictModel):
    """One dataset build, fully described."""

    seed: int = 0
    scenes: int = Field(default=300, ge=1)
    domain: Literal["cell", "tabletop"] = "cell"
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    families: FamilyMixConfig = Field(default_factory=FamilyMixConfig)
    camera_rig: CameraRigConfig = Field(default_factory=CameraRigConfig)
    assets: AssetSourcesConfig = Field(default_factory=AssetSourcesConfig)
    randomization: RandomizationConfig = Field(default_factory=RandomizationConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _the_bin_fits_where_the_arm_can_reach(self) -> "DatagenConfig":
        """Refuse or report a bin that places objects outside the declared reach.

        The bin family places inside the bin, not inside the workspace, so the two are related only
        if something relates them. A scene outside the reachable patch renders fine and then fails
        every reach.

        Declared is refused, inherited is reported, and the difference is not a softening. A caller
        who writes `workspace.bin` is making a claim about their cell, and a claim that contradicts
        the reach they also declared is an error worth stopping for. A caller who never mentioned
        the bin inherited it, and refusing them would be refusing a decision they did not make.

        The distinction is load-bearing for the small arms. A UR3e declares a 100 x 100 mm
        half-extent patch, the default bin places out to 120 x 70, and that pairing is the one the
        sibling reach refusal in this class recommends by name, so the 20 mm overhang is reported
        rather than refused.
        """
        box = self.workspace.bin
        half_x = box.inner_mm[0] / 2.0 - box.object_inset_mm
        half_y = box.inner_mm[1] / 2.0 - box.object_inset_mm
        reach_x, reach_y = self.workspace.half_extents_mm
        if half_x <= reach_x and half_y <= reach_y:
            return self
        complaint = (
            f"the bin places objects up to ({half_x:.0f}, {half_y:.0f}) mm from the workspace "
            f"centre, but workspace.half_extents_mm reaches only ({reach_x:.0f}, {reach_y:.0f}), so "
            f"objects in the bin family will be outside the patch this cell declares it can reach. "
            f"Widen workspace.half_extents_mm, or shrink workspace.bin.inner_mm, or raise "
            f"workspace.bin.object_inset_mm.")
        if "bin" in self.workspace.model_fields_set:
            raise ValueError(complaint)
        warnings.warn(complaint, BinExceedsWorkspace, stacklevel=2)
        return self

    @model_validator(mode="after")
    def _an_eye_in_hand_camera_has_an_arm(self) -> "DatagenConfig":
        """Refuse an eye-in-hand view with no arm in the scene, before anything renders.

        `IsaacRenderer.render` places a wrist viewpoint with no arm when `render.arm.mode` is
        `absent`: the pose is a valid camera pose and the dataset records the arm mode, so a
        consumer can tell the two apart. That is almost never what the caller meant. An eye-in-hand
        dataset whose images contain no hand is a different dataset wearing the same name, and the
        arm is this cell's dominant occluder, so its absence changes every image rather than
        decorating it.

        A refusal, not a removal: `camera_rig.allow_unmounted_wrist = True` is how a caller says
        they meant it, and then the combination renders. Firing at config time is the point, because
        the alternative is a full dataset that records the disagreement in a field somebody has to
        think to read.
        """
        if self.camera_rig.allow_unmounted_wrist or self.render.arm.mode != "absent":
            return self
        rig = self.camera_rig
        if rig.cameras is not None:
            # A declared camera is always a request: nobody writes one out by accident.
            wrist = [c.name for c in rig.cameras if c.mount == "wrist"]
        elif "views" in rig.model_fields_set:
            wrist = [name for name in rig.views if name == "wrist"]
        else:
            # The default rig is not a request, and this boundary is deliberate. `views` defaults to
            # a three-view rig that has a wrist camera because the cell it describes has an arm. A
            # caller who writes `arm.mode = "absent"` and leaves `views` alone is saying "no robot",
            # not "give me an eye-in-hand view without one", and refusing them would make the
            # natural way to ask for armless tabletop data an error.
            #
            # `model_fields_set` is what separates the two, and it is the same distinction as the
            # UNSET sentinel in `src/contracts`: chosen and defaulted are different facts, and a
            # check that cannot tell them apart has to guess which one it is looking at.
            #
            # So the defaulted case still renders a free-floating wrist view. This validator narrows
            # the failure to the case somebody meant; it does not remove the behaviour.
            return self
        if wrist:
            raise ValueError(
                f"camera_rig asks for the eye-in-hand view(s) {sorted(wrist)} while "
                f"render.arm.mode is 'absent', so the wrist camera would be rendered free-floating "
                f"with no arm in the picture. Either put an arm in the scene, or drop the wrist "
                f"view, or set camera_rig.allow_unmounted_wrist = True to say you meant it."
            )
        return self

    @model_validator(mode="after")
    def _a_posed_arm_has_something_to_reach_for(self) -> "DatagenConfig":
        """Refuse a posed arm with no wrist camera to place it by, before anything renders.

        `NoEngineRenderer._arm` refuses a scene whose arm mode is `posed` when the model has no
        measured wrist-camera mount, or when no wrist camera is planned: without one the arm can
        only be parked, and a parked arm sits outside both oblique views, so the corpus would
        contain no arm while the config claimed one. That refusal is right, and it fires per scene,
        inside the renderer, after a build has already started.

        This validator moves the second of those two reasons to config time; the first still fires
        per scene, so a model with no measured mount and a planned wrist camera passes here and is
        refused by the renderer.

        Same boundary as the rule above. The built-in rig plans a wrist camera, so a caller who
        leaves the rig alone is unaffected; this fires when the planned cameras are the caller's own
        and none of them is on the wrist.
        """
        if self.render.arm.mode == "absent":
            return self
        rig = self.camera_rig
        has_wrist = (any(c.mount == "wrist" for c in rig.cameras) if rig.cameras is not None
                     else "wrist" in rig.views)
        if not has_wrist:
            raise ValueError(
                f"render.arm.mode is {self.render.arm.mode!r} and no wrist camera is planned. "
                f"Without one the arm can only be parked, and a parked arm is measured to sit "
                f"OUTSIDE the oblique views, so the dataset would contain no arm while the config "
                f"claims one. Either plan a wrist camera, or set render.arm.mode to 'absent'."
            )
        return self

    @model_validator(mode="after")
    def _the_arm_can_reach_the_workspace(self) -> "DatagenConfig":
        """Refuse a workspace the configured robot cannot reach, before anything else runs.

        The earliest possible point on purpose. A dataset whose far corner is outside the arm's
        reach renders perfectly and is worthless for grasp labels, and nothing in the images says
        which objects were never pickable. Swapping a UR5e cell to a UR3e shrinks the reach from
        850 to 500 mm, which puts the UR5e patch 177 mm outside the sphere, and it is a change of
        one config string.

        Skipped when no arm is placed: a ``tabletop`` dataset with ``arm.mode="absent"`` is a
        legitimate thing to generate and has no reach to satisfy.
        """
        if self.render.arm.mode == "absent":
            return self
        # Deferred for cost, not for a cycle. `datagen.robots` imports only `datagen.constants`, so
        # a top-level import here would close no loop. It would instead pull a few hundred modules
        # into `datagen.config`, which is imported by nearly everything in the package, so the
        # import stays here. Promoting it makes the package slower and fixes nothing.
        from datagen.robots import resolve_robot, workspace_overshoot_mm  # noqa: PLC0415

        from datagen.robots import declared_mount  # noqa: PLC0415

        definition = resolve_robot(self.render.arm.robot_model,
                                   declared_mount(self.render.arm.wrist_camera_mount))
        overshoot = workspace_overshoot_mm(
            definition, self.workspace.center_mm, self.workspace.half_extents_mm,
            self.workspace.table_height_mm,
        )
        if overshoot > 0.0:
            raise ValueError(
                # ASCII on purpose: this message is printed to a Windows console (cp1252), where a
                # non-ASCII separator arrives as a replacement character in the middle of the number.
                f"workspace {self.workspace.center_mm} +/- {self.workspace.half_extents_mm} mm reaches "
                f"{overshoot:.0f} mm beyond what a {definition.key} can touch "
                f"(reach {definition.max_reach_mm:.0f} mm about a "
                f"{definition.shoulder_height_mm:.0f} mm shoulder). Objects out there would render "
                f"fine and never be pickable.\n"
                f"  This robot's reachable patch is center_mm={definition.workspace_center_mm}, "
                f"half_extents_mm={definition.workspace_half_extents_mm}.\n"
                f"  Set those, or set render.arm.mode='absent' if the dataset does not need the arm."
            )
        return self

    @classmethod
    def from_stamp(cls, stored: dict) -> "DatagenConfig":
        """Rebuild the config a dataset records, tolerating keys this build no longer has.

        A stored config is a historical record, not a configuration to validate. Reading it with
        `extra='forbid'` makes every schema change retroactively invalidate every dataset already
        built, because the provenance stamp embeds the whole config the dataset was built with.
        Three callers rebuild it, `eval/ladder.py`, `grasps/labels.py` and `prompts/build.py`, all
        to re-derive the asset manifest and check the assets have not drifted.

        Dropped keys are reported, never silently ignored. `extra='ignore'` would make a typo in a
        live config and a retired key look identical; what is dropped is warned about, so drift
        stays visible and a reader can tell which of the two they are looking at.

        It does not tolerate a key whose type changed, or a missing required field. Those are real
        incompatibilities between the record and this build, and they still refuse.
        """
        dropped: list[str] = []

        def prune(payload: dict, model: type, path: str = "") -> dict:
            fields = getattr(model, "model_fields", {})
            out: dict = {}
            for key, value in payload.items():
                if key not in fields:
                    dropped.append(f"{path}{key}")
                    continue
                annotation = fields[key].annotation
                if isinstance(value, dict) and hasattr(annotation, "model_fields"):
                    out[key] = prune(value, annotation, f"{path}{key}.")
                else:
                    out[key] = value
            return out

        pruned = prune(dict(stored), cls)
        if dropped:
            # A warning, not a logger. This module is imported by everything in the package,
            # including `ProcessPoolExecutor` workers, and `create_logger` opens its file at import.
            # `src/utility/log_cfg.py` records what that costs: each worker holds its own log
            # handles, the parent can then rotate none of them, and the resulting
            # `PermissionError: WinError 32` is silent because logging swallows handler errors. A
            # warning needs no handle.
            warnings.warn(
                f"provenance config carries {len(dropped)} key(s) this build no longer has, "
                f"ignored while rebuilding the manifest: {', '.join(sorted(dropped))}",
                StoredConfigDrift, stacklevel=2)
        return cls(**pruned)

