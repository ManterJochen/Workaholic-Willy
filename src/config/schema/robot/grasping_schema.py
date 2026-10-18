"""Grasping config schema, a sibling of robot_schema.py."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from .._base import ConfigPath, StrictModel
from .._removed import REMOVED_GRASP_MODES


class GraspingContainerConfig(StrictModel):
    """Optional bin, tray or KLT the objects sit inside. Everything here defaults to "no container".

    Give it when the parts do not rest on the table itself. A KLT standing on the workspace raises
    the surface its contents rest on by the thickness of its own floor, and a grasp planner that
    thinks the support is the table will allow a finger that many millimetres too low.

    ``floor_height_mm`` is the inside floor in BASE, the surface you would touch if you reached
    into the empty bin. It is not the height of the bin, and not the rim.

    The walls are the other half. The candidate filter checks a target cloud, the neighbours'
    clouds and a support plane, and a plane is a floor, so a wall stays invisible to it until one
    is declared here and ``wall_collision_enabled`` turns the check on. Replaying the reference
    verdict on the rank-0 winners that lose the ranking gap on ``finger_collision``: 60.9 % of them
    hit a bin wall, and in the bin family it is 14 of 14. A wall is not an instance any detector
    segments, so no camera can close that and it has to be declared, which is what
    ``interior_min_mm`` / ``interior_max_mm`` are for.
    """

    floor_height_mm: float | None = Field(
        default=None,
        description=(
            "Height in BASE mm of the surface inside the container that parts rest on. None (default) "
            "-> the workspace surface is used, i.e. no container."
        ),
    )
    interior_min_mm: tuple[float, float, float] | None = Field(
        default=None,
        description=(
            "Lower corner of the container's interior box in BASE mm (x, y, z), the empty volume "
            "you could pour parts into. z is the inside floor, the same surface as "
            "`floor_height_mm`. None (default) -> no wall geometry."
        ),
    )
    interior_max_mm: tuple[float, float, float] | None = Field(
        default=None,
        description=(
            "Upper corner of the container's interior box in BASE mm (x, y, z). z is the rim, the "
            "height a finger may pass over from outside. Must exceed `interior_min_mm` on every axis."
        ),
    )
    wall_collision_enabled: bool = Field(
        default=False,
        description=(
            "Treat the four vertical walls of that interior box as collision geometry: they are "
            "sampled into points and appended to what the candidate filter already checks against. "
            "Separate from declaring the box so a cell can describe its bin without changing "
            "grasping behaviour. Fail-closed: setting this without both interior corners raises at "
            "config load, because a cell that believes its walls are protected and is not is worse "
            "than one that knows they are not. False (default) = byte-identical.\n\n"
            "Measured (354 jaw-graspable objects, D5 reference): top-1 55.9 -> 56.2 %, which is one "
            "object on a run that is not bit-reproducible, so flat. Do not turn this on expecting a "
            "better pick rate. What it does move: candidate precision 64.6 -> 80.7 %, and the `bin` "
            "family's ranking gap from 15 objects to 1 (to 0 with the fused neighbour cloud as "
            "well). It converts 'the ranker chose a grasp that hits the wall' into 'the ranker "
            "offered nothing', which is neutral in a simulator and is the difference between a "
            "retry and a collision on real hardware."
        ),
    )
    wall_thickness_mm: float = Field(
        default=5.0,
        gt=0.0,
        le=100.0,
        description=(
            "How far outward from each interior face the wall material is sampled. Sampling only the "
            "inner face would leave a finger standing inside a thick wall undetected unless the "
            "collision margin happened to exceed half the thickness; that is a coincidence, not a "
            "check. A KLT is typically 3-8 mm."
        ),
    )
    wall_sample_mm: float = Field(
        default=8.0,
        gt=0.0,
        le=100.0,
        description=(
            "Point spacing on the sampled walls. Matches the calculator's cloud voxel "
            "(`calculator.CLOUD_VOXEL_MM`, 8 mm) so wall points arrive at the same density as the "
            "neighbour points they are checked alongside."
        ),
    )

    @property
    def interior_declared(self) -> bool:
        """Whether this container is declared as a box: both interior corners given, each axis a real span.

        ``floor_height_mm`` alone raises the surface the parts rest on, but it declares no interior to act
        inside. So a push box inside a container reads this.
        """

        low, high = self.interior_min_mm, self.interior_max_mm
        if low is None or high is None:
            return False
        return all(float(h) > float(lo) for lo, h in zip(low, high))

    @model_validator(mode="after")
    def _walls_need_a_box(self) -> "GraspingContainerConfig":
        if not self.wall_collision_enabled:
            return self
        if self.interior_min_mm is None or self.interior_max_mm is None:
            raise ValueError(
                "container.wall_collision_enabled requires both interior_min_mm and "
                "interior_max_mm: there is no default bin to guess at"
            )
        for axis, (low, high) in enumerate(zip(self.interior_min_mm, self.interior_max_mm)):
            if high <= low:
                raise ValueError(
                    f"container.interior_max_mm[{axis}] ({high}) must exceed "
                    f"interior_min_mm[{axis}] ({low})"
                )
        return self


class GraspingDeepGeneratorConfig(StrictModel):
    """The learned 6-DoF grasp generator, what `grasping.calculator: deep` selects.

    Different from `deep_ranker` in the one way that matters: the ranker scores candidates the
    analytic stack proposed and changes no order, this one proposes them instead of the analytic
    stack. This build has no shadow mode for a generator: either it is the source of candidates or
    it is not running. A promotion record names `shadow` and `ab` for a proof that has not reached
    `active`, and a cell builds a generator at `active` alone.

    It fails closed. `calculator: deep` with no readable artifact refuses to build the cell rather
    than falling back to `geometric`: a cell that asked for the learned generator and quietly got
    the analytic one would file the analytic one's numbers under the learned one's name. A cell
    also refuses an artifact whose proof has not passed (the owner's decision of 2026-10-09): no
    finished models ship, every customer trains their own, and a trained generator drives a cell
    only once the promotion record beside it says its proof passed at `active`
    (`deep/promotion.py`). Nothing writes one yet, so today every artifact refuses a cell, and the
    ladder and the simulation runners still build it to measure it.

    The score it attaches is not a probability. The v1 net has no calibrated P(hold) head: the
    corpus carries no physics until it is shaken. A scorer fitted on shaken holds, `held_jaw_v1`,
    reached 0.8947 AUROC on analytic candidates with folds that share meshes and 0.8758 with no test
    mesh trained on (the 2026-10-09 review), and it was never scored on this generator's proposals.
    Generator proposes, scorer ranks, and each is promoted on its own evidence.
    """

    artifact_path: ConfigPath | None = Field(
        default=None,
        description=(
            "Path to the generator's `.pt`, written by "
            "`python -m src.robot.grasping.deep train-set`. The card beside it is committed; the "
            "weights are not, exactly as the ranker's trees are not. Null (default) means the "
            "learned generator cannot be selected; `calculator: deep` then refuses. "
            "One family is accepted here, a `set_grasp_generator`; a file of the binned family, "
            "retired on 2026-09-04, is refused by name. A cell also refuses an artifact whose "
            "promotion record beside it (`<stem>.promotion.json`) does not say its proof passed at "
            "`active` (2026-10-09); nothing writes one yet."
        ),
    )
    device: str | None = Field(
        default=None,
        description=(
            "Torch device for inference. Null picks CUDA when available and CPU otherwise. Named "
            "explicitly only by a cell that has a reason: a second GPU, or a deliberate CPU run."
        ),
    )
    #: No field here names the hand. The generator conditions on the cell's hand,
    #: `robot.gripper.model`, the one name a hand has, which the calculator factory checks against the
    #: artifact's trained hands at build. `schema/_removed.py` refuses a tree that still writes
    #: `gripper` here.
    minimum_score: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description=(
            "Points below this graspability score are never seeded. 0.5 is the natural threshold for "
            "a calibrated head and this head is not calibrated, so it is a knob rather than a "
            "constant: the first real run is expected to move it once the score distribution has been "
            "looked at rather than assumed."
        ),
    )


class GraspingDeepRankerConfig(StrictModel):
    """The learned grasp ranker, in shadow: it computes, the telemetry records, the order never changes.

    Measured on 1,611 (scene, view, object) units of `v1_proof`, out of fold, with the folds cut by
    placement (a scene and an instance) and not by mesh: the corpus places each mesh many times over,
    so a scored mesh can sit in the training folds in another placement (the 2026-10-09 review):

        random (expected)   54.4 %      the calculator's own order   51.6 %
        this ranker         84.1 %      best possible               100.0 %

    Pooled over the candidates of every evaluation configuration, not the support-footprint stage's
    list alone. Cut by mesh, its AUROC goes from 0.9188 to 0.9054; and the same recipe fitted on
    shaken holds, with no test mesh trained on, chose among SFE's own candidates as well as SFE's
    order did and no better, +0.000 (95 % CI -0.027 to +0.027).

    "top-1 is valid": the first candidate physically closes. Not "the grasp holds": the label is
    the analytic verdict this stack computes, and the pilot's hold rate is 5.62 % where validity
    there is 48.8 %. Those are different questions and only a shake answers the second, which is
    why nothing here promotes anything.

    Shadow, for a measured reason. A ranker fitted on aletheia's shaken corpus beat the calculator
    by 17.7 pp on the same units and still lost to picking at random, because the failure it has to
    predict is `finger_collision` and its features cannot see the surroundings. What this logs is
    ``would_change_top1``: whether turning it on would do anything, answered on every real pick.

    Enabling it adds telemetry keys and changes no decision. That is what the wiring guard checks: a
    flag whose runtime is unchanged fails, and telemetry is the runtime here.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Score every candidate with the learned ranker and record what it would have chosen. "
            "Changes no ordering and no decision; adds deep_ranker_* telemetry keys."
        ),
    )
    artifact_dir: ConfigPath = Field(
        default="${WILLY_PROJECT_ROOT}/assets/models/grasp_ranker/v1", min_length=1, validate_default=True,
        description=(
            "Where the fitted ranker lives. The trees are gitignored and regenerated with "
            "`python -m datagen train-ranker`; the card beside them is committed."
        ),
    )
    spec: str = Field(
        default="held_jaw_v1", min_length=1,
        description=(
            "Which feature contract to score under. Must match the artifact: the scorer refuses a "
            "mismatch rather than scoring a vector the model was never fitted on. Two ship: "
            "`held_jaw_v1` predicts whether Isaac's shake could not pull the object out, and "
            "`valid_jaw_v1` predicts this stack's own analytic verdict, which makes anything "
            "measured with it self-referential. Measured on identical rows, folds and features "
            "with only the target changed, both graded on held: 0.8947 against 0.7749 AUROC, "
            "and on the `too_wide` stratum the valid-trained model scores 0.5220, a coin toss."
        ),
    )


class GraspingSupportConfig(StrictModel):
    """What the parts stand on besides the bench: the bin they rest in, where the cell has one.

    The table is the planner world's bench, the top of ``robot.safety.planning_world.support_plane`` (0 mm where the
    tree declares none). It is level, a part's own lowest point raises it wherever the part's cloud saw down its side,
    and the gripper keeps 1 mm over it, the guard's finger floor (``src/robot/grasping/collision/support_resolver.py``;
    the owner's cleanup of 2026-10-17 took the four keys that restated these).

    Known limit. Even fused over the cameras, a part's lowest point still reads high in a bin (p90 13.3 mm against
    1.7 mm in the open). Since the higher of the two wins, that costs some valid low grasps in exactly the family that
    is already hardest. Declaring ``container.floor_height_mm`` does not fix it: the floor is the lower of the two and
    the fused estimate still wins.
    """

    container: GraspingContainerConfig = Field(
        default_factory=GraspingContainerConfig,
        description="Optional bin/tray the parts rest in. Default: none, the bench is what they stand on.",
    )


class RobotGraspingConfig(StrictModel):
    """Vendor-neutral grasping-behaviour surface.

    Sits as a sibling of :class:`RobotSafetyConfig` under :class:`RobotConfig`.
    All knobs are behavioural (mode selection, recovery, decision); safety
    remains exclusively under ``robot.safety``.

    ``verification`` and ``dense_recovery`` were removed on 2026-09-29: no pick path ran the
    post-grasp verification stage, the gripper's own hold evidence decides every close, and the
    dense-recovery policy was consulted by no pick. A tree that still writes either block is refused
    at load with the sentence that says what to do instead (``src/config/schema/_removed.py``).

    The runtime derives per-mode behaviour from the
    :class:`src.robot.execution.autonomous_grasp.GraspBehaviorProfile`
    table. Values here apply on top of that profile: they tighten or disable
    optional behaviour, but cannot relax safety limits or unlock recovery actions
    the mode profile forbids. ``default_mode`` is resolved against
    :class:`src.robot.grasping.types.modes.GraspMode` at use time so the config
    layer imports no runtime modules.

    Twelve switches left on 2026-10-17 at the values the owner's cell ran (the owner's cleanup): the attempts a pick
    may make (5), the parts beside the one grasped seen as obstacles, tilted approaches, grasps from inside a bore, the
    first good part taken at once (0.6), the colour check (clipped pixels left out), the parts beside named, the weak
    look, the task's own places hidden from the detector and a task following its parts. Each now says what it does
    where it is done, and a tree that still writes one is refused at load (``schema/_removed.py``).
    """

    # A stable :class:`GraspMode` name resolved at runtime: "easy", "auto" or "dense_clutter", the owner's cell's and
    # the default since 2026-10-17. "closed_loop" and "dense_autonomous" were removed on 2026-09-29 and are refused at
    # load with the mode to name instead (`_refuse_removed_default_mode`).
    default_mode: str = Field(default="dense_clutter", min_length=1)
    #: Which grasp generator ranks the candidates. See `src/robot/grasping/deep/` and
    #: `.ai-memory/dl-grasping-plan.md`.
    #:
    #: ``geometric`` is the analytic stack: the part's own body over its support proposes and scores the
    #: candidates, then the whole collision/IK/corridor tail.
    #:
    #: ``deep`` is the learned 6-DoF generator. It replaces the analytic proposal stage: the factory
    #: returns one generator or the other, and the deep decoder seeds only from its own graspability
    #: head, not from the analytic candidates. It has no rejection tail of its own; the table check,
    #: the gripper-collision envelope, the antipodal test and the corridor filter all live in the
    #: analytic generator, and extracting them into a stage both generators run is an open build.
    #: The safety preflight is unaffected: it sits at the arm and gates the joint target whatever
    #: proposed it, so a deep candidate meets exactly the same fail-closed guards.
    #:
    #: The default stays ``geometric`` until the learned one clears its pre-registered bar: beat
    #: sfe_fused top-1 55.9 % on the eval-grasps ladder, plus on-box dense-gate parity. Selecting
    #: ``deep`` without trained weights present fails closed with the artifact path in the message,
    #: so a cell can never quietly fall back to the other generator and leave "which one ran"
    #: unanswerable from the outside.
    calculator: Literal["geometric", "deep"] = Field(
        default="geometric",
        description=(
            "Which grasp generator ranks candidates. 'geometric' is the analytic stack; 'deep' is "
            "the learned 6-DoF generator, which additionally takes the geometric candidates as "
            "seeds. Fails closed when 'deep' is selected without weights."
        ),
    )
    record_log_path: ConfigPath | None = Field(
        default=None,
        description=(
            "The JSONL file every pick() appends its GraspAttemptRecord to (what "
            "`python -m src.robot.grasping.replay` reads). Null, the default, or an environment reference "
            'that expands to nothing (record_log_path: "${WILLY_RECORD_LOG:-}") appends to '
            "grasp_records.jsonl in the log folder (WILLY_LOG_DIR, else logs/), since 2026-10-17: a cell "
            "keeps its records whether or not its tree names a file."
        ),
    )
    deep_generator: GraspingDeepGeneratorConfig = Field(
        default_factory=GraspingDeepGeneratorConfig,
        description=(
            "The learned 6-DoF grasp generator, selected by `calculator: deep`. Inert while "
            "`calculator` is 'geometric', which is the default, so this block changes nothing "
            "until a cell deliberately switches its generator."
        ),
    )
    deep_ranker: GraspingDeepRankerConfig = Field(
        default_factory=GraspingDeepRankerConfig,
        description=(
            "The learned grasp ranker in shadow mode. Default off; enabling it adds telemetry and "
            "changes no decision. See the class docstring for what it measured."
        ),
    )
    support: GraspingSupportConfig = Field(
        default_factory=GraspingSupportConfig,
        description=(
            "What the parts stand on besides the bench (robot.safety.planning_world.support_plane): the bin they "
            "rest in, where the cell has one."
        ),
    )

    @field_validator("default_mode")
    @classmethod
    def _refuse_removed_default_mode(cls, value: str) -> str:
        """A mode removed on purpose is refused here, at load, rather than at the first pick, with the sentence that
        names the mode to use instead: ``closed_loop`` and ``dense_autonomous`` left on 2026-09-29 with the two-scan
        refinement they ran."""
        said = REMOVED_GRASP_MODES.get(str(value).strip().lower())
        if said is not None:
            raise ValueError(f"default_mode names the grasp mode {value!r}, removed on purpose: {said}")
        return value

