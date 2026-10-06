"""Wire types for the console.

These serialise what the library already decided; they add no judgement of their own. Every verdict,
every provenance record and every validation error comes from ``src``. A field here that has to
compute something is a sign the logic belongs in the library instead.
"""

from __future__ import annotations

from typing import Annotated, Any, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from api.codes import (
    BlockerCodeName,
    CommandNoteName,
    EventTypeName,
    JawsChoiceName,
    JawsStageName,
    LightCodeName,
    LightIdName,
    LightStateName,
    RefusalCodeName,
    RunKindName,
    StopClassName,
    StopCodeName,
)

#: The four verdicts a check can carry. Mirrors ``CheckStatus``; pinned as a literal so a new status in
#: the library fails this contract loudly instead of arriving in the UI as an unstyled unknown string.
Status: TypeAlias = Literal["ok", "block", "warn", "bench"]


class PreflightCheckOut(BaseModel):
    """One row of the cell checklist, exactly as ``run_config_preflight`` produced it."""

    name: str
    status: Status
    detail: str
    #: The concrete fix. Empty for ``ok``. A checklist that says "wrong" without saying "do this" just
    #: moves the guessing, so the UI renders this row-by-row rather than linking to a manual.
    fix: str = ""


class PreflightOut(BaseModel):
    """The whole checklist plus the one-line verdict the operator acts on."""

    checks: list[PreflightCheckOut]
    ok: bool
    n_blocking: int
    n_warn: int
    n_bench: int
    #: The profile chain this was evaluated under (e.g. ``"ur3e,tiltcam"``). A verdict without its chain
    #: is unreadable: the same tree can block as a UR cell and pass as a sim cell.
    profile: str | None = None
    vendor: str


class LayerOut(BaseModel):
    """One file that sets a key: where, what it wrote, and whether it won."""

    location: str
    raw: str
    winner: bool


class ConfigValueOut(BaseModel):
    """A config key with the provenance the CLI's ``explain`` prints.

    ``source`` and ``layers`` are what make this worth an endpoint: with layered profiles, "what is
    the value" is only half a question. The other half is which layer won, and the UI must be able to
    say so without the operator reconstructing a four-file merge in their head.
    """

    key: str
    value: Any
    type: str | None = None
    default: Any = None
    doc: str | None = None
    #: ``"means"`` when the prose describes this key, ``"block"`` when it describes its neighbourhood.
    doc_scope: str | None = None
    source: str | None = None
    tier: str | None = None
    layers: list[LayerOut] = Field(default_factory=list)
    #: The YAML comment above the winning line, usually the justification for the number itself.
    why: str | None = None
    writable: bool = False
    #: The CLI's own rendering, verbatim. Carried so the UI can show exactly what the terminal shows
    #: rather than re-typesetting it, and so the two can never disagree about the same key.
    text: str = ""


class WritableOut(BaseModel):
    """One key the console offers as a form field, with the instruction for obtaining its value."""

    key: str
    label: str
    #: What the operator physically does. A form that says "float, kg" has not told them anything.
    measure: str
    unit: str = ""


class ConfigPatchOut(BaseModel):
    """The result of a guided write. All the values landed, or the request failed and none did."""

    applied: bool
    #: What the tree holds after the write, read back rather than echoed: the value that survives
    #: coercion and the validators is the one the cell will run with.
    values: dict[str, Any]
    files: list[str]
    #: Re-rendered because a single write routinely clears or reveals several checklist rows.
    preflight: PreflightOut


class MotionWarningOut(BaseModel):
    """One thing that physically moves when connect is pressed.

    Derived from what was built, not from what config asked for: warning about a finger sweep that
    cannot happen (because the gripper fell back to a substitute) teaches an operator to skip the
    warning that can.
    """

    subject: str
    what: str
    precaution: str


#: What kind of hand this is, as the console tells hands apart: a toggle on one output with no sensor (the owner's
#: Hand-E on DO0), a two-state jaw on solenoids, a gripper commanded by width, a suction cup, or none.
HandKind: TypeAlias = Literal["toggle", "jaw", "width", "suction", "none"]
#: Where the program believes a toggle's jaws stand: its own count of its changes, never a measurement. ``unknown``
#: where nobody can vouch for the count (a change that failed, the output switched at the pendant); ``not_counted``
#: for a hand that keeps no count, and for a toggle that is not connected (its count ends at a disconnect).
JawsBelief: TypeAlias = Literal["open", "closed", "unknown", "not_counted"]


class HandOut(BaseModel):
    """The hand, as the hand chip and the stop card's gates read it (``api.jaws.hand_of``). Reads, never commands."""

    kind: HandKind = "none"
    #: The driver's class, e.g. ``JawIOGripper``.
    driver: str = ""
    #: The bank and pin the jaws are driven on, as a person reads it on the pendant (``tool output 0``); ``""`` where
    #: the hand is not driven on an output.
    where: str = ""
    connected: bool = False
    jaws: JawsBelief = "not_counted"
    #: Why nobody can say where the jaws stand, in the driver's words; ``""`` where the count stands.
    why_unknown: str = ""
    #: True where nothing reads the hand back: a hold is never measured, a close counts as a grasp.
    no_sensor: bool = False
    #: Commands sent to the jaws since the driver was built, a connect's included; null for a hand that does not count.
    commands_sent: int | None = None


#: What a halt did to the move in flight, in the library's words (``HALT_BRAKE_OUTCOMES``, ``HaltState.brake``):
#: ``none`` no move was in flight; ``pending`` one was, and it has not ended yet; ``braked`` braked under control, and the
#: arm stood still; ``unconfirmed`` the arm was not seen to stand still (if it still moves, the emergency stop is the
#: answer); ``ran_out`` it ended without a brake (brakes off, already finished, or the controller stopped it).
HaltBrake: TypeAlias = Literal["none", "pending", "braked", "unconfirmed", "ran_out"]


class HaltStateOut(BaseModel):
    """The arm's halt latch ("halt now", ``POST /v1/cell/brake``), set until "the cell is clear" clears it."""

    reason: str
    requested_at: float
    #: A motion was in flight when it was pressed. It stays true after that move ended: read ``brake`` for what became
    #: of it.
    in_motion: bool = False
    #: The move in flight was braked (``robot.ur.brake_on_halt``); false where it ran to its end first.
    braked: bool = False
    brake_s: float | None = None
    #: What became of the move in flight (:data:`HaltBrake`), as the arm's own record says it (``HaltState.brake``);
    #: ``None`` where the record does not say, never a word guessed from ``in_motion`` and ``braked``: a move that ran out
    #: would read as one still braking.
    brake: HaltBrake | None = None


#: The arm's planner: ``not_used`` where it plans nothing (ik, the dummy, a KUKA), else cuRobo's state.
PlannerState: TypeAlias = Literal["not_used", "off", "starting", "ready"]


class PlannerOut(BaseModel):
    """cuRobo's state on this arm. Starting it moves nothing and takes about a minute."""

    state: PlannerState = "not_used"


#: What models the part the gripper carries now: ``none``, ``filter_only`` or ``planner_and_filter`` as the arm says it
#: (``CarriesPayload.payload_model``); ``not_applicable`` for an arm that models no part at all (the dummy, sim);
#: ``unknown`` for an arm that models parts whose answer could not be read now, or is one this console does not know.
PayloadModelName: TypeAlias = Literal["none", "filter_only", "planner_and_filter", "not_applicable", "unknown"]

#: The ``payload_model`` values that model no part now: one read of the server's ``part_still_held`` rule
#: (``api.readiness.part_held``), beside a toggle's count that says closed, a hand that measures a part, and, while a
#: stop record stands, the record's own belief that a part is in a hand that can say nothing (until "Backen leer").
#: The stop card's "no part held, now" gate follows that whole rule through the ready bar's ``part_still_held``
#: blocker. ``not_applicable`` is in it: an arm that models no part carries none the planner knows of, so the gate
#: opens on the rehearsal cell as on a UR. ``unknown`` is not: a part nobody can rule out keeps the gate shut.
#: ``frontend/src/api/codes.ts`` lists the same set (``NO_PART_MODELLED``).
NO_PART_MODELLED: Final[frozenset[PayloadModelName]] = frozenset({"none", "not_applicable"})


class RecoveryOut(BaseModel):
    """The console's recovery record: a moving run ended on a problem code, and the arm stands where it stopped.

    While it stands and ``cleared_at`` is not after ``at``, every moving route refuses ``cell_not_cleared``; while it
    stands at all, a new task or pick refuses ``restart_required``. Only a Restart's or a Home run's arrival at its
    return pose ends it. It lives on the console, so it survives Disconnect, a rebuild and a page reload; ``python -m
    api`` also keeps it in a file (``Console.stop_file``), so it comes back, uncleared, after a restart of the server.
    """

    run_id: str
    kind: RunKindName
    stop_code: StopCodeName
    #: When the run ended, Unix seconds.
    at: float
    #: Whether the program believed a part was in the jaws. Every gate reads the live hand first; for a hand that can
    #: say nothing itself (no toggle count, no measurement) this belief keeps ``part_still_held`` standing until a
    #: person says "Backen leer" (``POST /v1/cell/acknowledge`` with ``jaws_empty``).
    holding: bool = False
    #: When a person confirmed "the cell is clear" since the stop; null until then.
    cleared_at: float | None = None


class CellOut(BaseModel):
    """Where the cell is, what it is made of, and who owns it."""

    state: str
    arm: str | None = None
    gripper: str | None = None
    vendor: str
    profile: str | None = None
    #: Non-null when a real end-effector could not be built. Connect is refused while it is set: the
    #: cell would come up, every pick would report success, and the gripper would close on nothing.
    gripper_substitution: dict[str, Any] | None = None
    lock_key: str | None = None
    #: Who holds the cross-process lock, if anyone. A snapshot: it never gates anything.
    lock_holder: str | None = None
    active_run_id: str | None = None
    #: Which perception stack this cell will ground with. Here as well as in ``/v1/diagnostics``
    #: because "what is this cell" and "which models will it use" are the same question to an
    #: operator, and the answer decides whether their prompt will work at all.
    perception: "PerceptionStackOut | None" = None
    #: The hand, read now (the hand chip, the stop card's gates).
    hand: HandOut = Field(default_factory=HandOut)
    #: Why a pick stopped where the arm stands, a person to decide (the service's latch); ``""`` while nothing waits.
    needs_person: str = ""
    #: The arm's halt latch, where "halt now" set it; null where it is not set or the arm has none.
    halted: HaltStateOut | None = None
    planner: PlannerOut = Field(default_factory=PlannerOut)
    #: What models the carried part now; no part is modelled while it is one of ``NO_PART_MODELLED``.
    payload_model: PayloadModelName = "not_applicable"
    recovery: RecoveryOut | None = None
    #: When the jaws were last answered open in the browser (a toggle), Unix seconds.
    jaws_confirmed_at: float | None = None
    #: The next moving run counts down 3 s first: a person's hands were at the arm since its last motion.
    countdown_due: bool = False
    #: A jaws question waits for its answer (``GET /v1/cell/jaws``); every moving route refuses meanwhile.
    jaws_question: bool = False


class ConnectPreviewOut(BaseModel):
    """What connecting will do, and the token that records having read it."""

    token: str
    expires_at: str
    arm: str
    gripper: str
    warnings: list[MotionWarningOut] = Field(default_factory=list)
    #: Non-empty means connect will be refused no matter what token is presented.
    blocking: list[str] = Field(default_factory=list)


class ConnectIn(BaseModel):
    """The acknowledgement. A token, and nothing else: there is no force flag."""

    token: str = Field(min_length=1)


class SdkOut(BaseModel):
    """One vendor SDK module, as ``doctor`` found it."""

    module: str
    #: A real import, not a `find_spec`, which says yes for a package the OS refuses to load.
    #: Still not a version match against the pin. `VendorReadinessOut.note` carries the reason
    #: it failed.
    importable: bool
    version: str | None = None


class VendorReadinessOut(BaseModel):
    vendor: str
    kind: str
    registered: bool
    ready: bool
    note: str
    sdks: list[SdkOut] = Field(default_factory=list)


class MotionStackOut(BaseModel):
    """Can this cell plan? Reported for the model the cell is configured to drive, not a default."""

    model: str
    #: Which config key supplied the model. A reading whose provenance is "default" is about a robot
    #: nobody configured, and the UI says so rather than showing a confident answer.
    model_source: str
    fully_anchored: bool
    curobo_available: bool
    curobo_python: str
    curobo_robot_config: str | None = None
    #: Carried in the payload so a UI cannot render "available" as more than it means.
    curobo_caveat: str = ""
    collision_engine: str | None = None
    mesh_bundle_present: bool = False
    coal_prefix: str | None = None


class ReachabilityOut(BaseModel):
    """Is anything listening at the controller's address?"""

    checked: bool
    address: str | None = None
    port: int | None = None
    reachable: bool | None = None
    latency_ms: float | None = None
    #: States the limit of the claim in the payload itself.
    detail: str = ""


class PerceptionStackOut(BaseModel):
    """Which perception stack this cell is configured for, and whether it can actually run.

    Answers the question an operator asks before a pick: which models will this prompt touch? The VLM
    route can be perfectly configured and still unrunnable because the weights are not on this box,
    and nothing says so until the first complex prompt arrives. This checks presence without loading
    anything.
    """

    pipeline_configured: bool = Field(
        description="False = the legacy keys decide everything (the byte-identical default)."
    )
    kind: str = Field(description="zero_shot | closed_set")
    backend: str = Field(description="grounded_sam | vlm | rtdetr")
    segmenter: str
    router_enabled: bool = Field(
        description="True = each prompt is routed between the phrase grounder and the VLM."
    )
    vlm_model_id: str | None = None
    vlm_weights_present: bool | None = Field(
        default=None,
        description="Whether the VLM checkpoint is in the local cache. None when no VLM is configured.",
    )
    vlm_on_unavailable: str | None = Field(
        default=None, description="refuse = reject the pick | degrade = fall back, loudly."
    )
    detail: str = Field(description="One sentence an operator can act on.")


class RoutePreviewOut(BaseModel):
    """What route a prompt would take, decided without a GPU, a model, or an image.

    The router is pure text analysis, so this is free and instant. It exists because "why was that
    pick slow?" and "why did it refuse?" are usually answered by the route, and an operator should be
    able to find that out by typing the prompt rather than by running it.
    """

    prompt: str
    route: str = Field(description="simple | vlm")
    reason: str = Field(description="the rule that decided it, e.g. non_english, negation")
    description: str = Field(description="human-readable, e.g. 'vlm (non_english)'")
    normalized_prompt: str | None = Field(
        default=None,
        description="What the phrase grounder would actually receive. None on the VLM route, which "
                    "gets the operator's words untouched.",
    )
    signals: dict[str, object] = Field(default_factory=dict)
    runnable: bool = Field(
        description="False when this route is configured but cannot run on this box right now."
    )
    blocked_reason: str | None = None


class DiagnosticsOut(BaseModel):
    """The whole bench panel: software, planner, perception, network. Nothing here moves anything."""

    vendors: list[VendorReadinessOut] = Field(default_factory=list)
    motion_stack: MotionStackOut
    perception: PerceptionStackOut
    reachability: ReachabilityOut


class ErrorOut(BaseModel):
    """One error envelope for every failure, so the client has exactly one shape to handle."""

    code: str
    message: str
    #: Machine payload: which key, which validator, which run held the lock.
    detail: dict[str, Any] = Field(default_factory=dict)


class TelemetryOut(BaseModel):
    """One live snapshot of the cell.

    Every measurement is optional because every one comes from an optional capability: sim, dummy and
    KUKA advertise no force/torque or controller-status Protocol at all. A missing field means "this
    driver does not offer it", which is not an error and does not appear in ``unavailable``.
    """

    state: str
    connected: bool
    #: On the panel deliberately: a simulated arm produces numbers that look exactly like a real one's,
    #: and this says which one produced them. Not the only such field, and not the whole answer: it is
    #: about the driver, so it is correctly False for URSim; ``controller_is_simulator`` below is the
    #: other half.
    simulated: bool = False
    vendor: str = ""
    model: str = ""

    tcp_position_mm: list[float] | None = None
    tcp_quaternion_xyzw: list[float] | None = None
    joint_positions: list[float] | None = None
    tcp_force_n: list[float] | None = None
    tcp_torque_nm: list[float] | None = None

    robot_mode: str | None = None
    safety_mode: str | None = None
    protective_stopped: bool | None = None
    emergency_stopped: bool | None = None
    controller_message: str | None = None

    #: Three-valued and never False: True = provably a simulator, None = this console cannot tell.
    #: `simulated` above is about the driver and is correctly False for URSim, which is real UR
    #: controller software driving a robot that does not exist. A UI must not render None as
    #: "physical": it means unknown.
    controller_is_simulator: bool | None = None
    controller_serial: str | None = None
    controller_host: str | None = None
    #: Echoes the ``include_controller_state`` query flag: False on a default tick, True when the
    #: caller asked for the controller block. What was asked for, not what was paid: only a driver
    #: with ``get_robot_status`` (UR today) then costs the extra dashboard read, and a default tick
    #: on a UR arm already spends one on the provenance evidence below.
    controller_state_included: bool = False

    #: Reads that should have worked and did not; on a live cell, usually a dropped connection.
    unavailable: dict[str, str] = Field(default_factory=dict)


class PickIn(BaseModel):
    """An operator instruction: what to pick, and how many times."""

    #: What to pick, typed or spoken: the phrase the detector grounds for this run and the label a
    #: target must carry. Empty keeps the phrase the cell was built with and filters no label.
    prompt: str = ""
    picks: int = Field(default=1, ge=1, le=100)
    #: How far a push of a failed part moves it, in mm, on a cell whose recovery pushes (``nudge_target``,
    #: dense_clutter). Null: the cell's ``recovery.fixture.push_distance_mm`` (30 mm by default, and wherever
    #: no fixture is declared). A request up to the cell's ``max_nudge_mm`` (50 mm without a fixture) is
    #: taken as asked; one above it, above 50 mm or under 10 mm is refused (422 ``push_distance_refused``,
    #: with the sentence), never shortened.
    push_mm: float | None = None


# --- a task: pick, then place, then return (POST /v1/task) --------------------------------------------------------

#: One part, or until nothing matching is left (owner decision 7).
TaskScope: TypeAlias = Literal["once", "until_empty"]
#: The directions a grasp may be asked to close along, as ``Pose.tool_down`` names them.
ClosingAxisName: TypeAlias = Literal["x", "-x", "y", "-y", "radial", "-radial", "tangential", "-tangential"]
#: A task's first motion: its first look, or, for a Restart, the planned move to the return pose.
FirstMotion: TypeAlias = Literal["look", "return"]
#: Where a command came from.
CommandSource: TypeAlias = Literal["typed", "spoken"]


class PosePlaceIn(BaseModel):
    """Place at a taught pose. The pose says where the part's BOTTOM is let go: the tool goes there raised by the part's
    hang, so every error goes toward more air."""

    kind: Literal["pose"]
    #: A name in ``robot.named_poses``; null means ``robot.default_place_pose``.
    pose: str | None = None


class CameraPlaceIn(BaseModel):
    """Place into a target the camera finds, such as a bin: opened just above its rim."""

    kind: Literal["camera"]
    #: The English phrase the camera looks for, e.g. ``blue bin``.
    phrase: str = Field(min_length=1, max_length=80)
    #: The operator's own words for it, for display.
    said: str | None = None


#: Where a task puts the part, told apart by ``kind``.
PlaceIn: TypeAlias = Annotated[PosePlaceIn | CameraPlaceIn, Field(discriminator="kind")]


class TaskOptionsIn(BaseModel):
    """The Advanced drawer: per task, never written to the cell's config."""

    #: Wrist looks as configured. False: the first configured look only, and no generated view.
    multi_view: bool = True
    #: Both jaw faces must be seen before a grasp is taken.
    both_faces: bool = False
    #: Keep only grasps that close along this direction (one of :data:`ClosingAxisName`, a leading ``+`` allowed);
    #: one the library cannot read is refused ``422 closing_axis_refused``.
    closing_axis: str | None = Field(default=None, max_length=16)
    #: How far a push of a failed part moves it, in mm; refused as ``POST /v1/pick`` refuses it. Null: the cell's, and a
    #: push that opens too little room there may go as far as the cell allows.
    push_mm: float | None = None
    #: The owner's switch for this task (2026-10-03): true, the parts are critical, nothing is pushed and a blocker is
    #: cleared instead; false, a push may rearrange the scene, and a blocker is cleared where none plans. Null: the
    #: cell's ``robot.grasping.recovery.critical_parts``.
    critical_parts: bool | None = None
    #: The run's own word on recovery (2026-10-05), overriding the cell's ``recovery.allowed_actions`` for this task:
    #: look again, push a boxed-in part, clear a blocker. Null: the cell's.
    rescan: bool | None = None
    push: bool | None = None
    clear: bool | None = None
    #: The owner's switch for this task (2026-10-06): true, a task that names no object takes a blocker as the part its
    #: pick takes and sets it down where the parts go, then picks the part it blocked; false, a blocker is set aside on
    #: the support. Null: the cell's ``robot.grasping.recovery.blocker_into_the_place``.
    blocker_into_the_place: bool | None = None
    #: Keep each pick's looks on disk.
    record_views: bool = False
    #: A camera place only: the air over the rim when the jaws open, 10-50 mm; null means 20.
    rim_air_mm: float | None = Field(default=None, ge=10.0, le=50.0)
    #: The operator confirmed that an empty ``object`` means anything the camera sees, bin walls included. A real cell
    #: refuses an empty object without it (``422 object_required``); the rehearsal cell does not.
    pick_anything: bool = False
    #: Render the grasp overlay during the task (costs pick time; a measuring aid in the tech view).
    overlay: bool = True


class CommandProvenanceIn(BaseModel):
    """What the operator said or typed, for the record only: nothing is read from it to decide what moves."""

    text: str = Field(default="", max_length=500)
    source: CommandSource = "typed"
    language: str | None = None
    #: The command reader produced the card's fields.
    parsed: bool = False
    #: The card's fields the operator changed by hand before Start.
    edited: list[str] = Field(default_factory=list)


class TaskIn(BaseModel):
    """One task: what to pick, where to put it, where to go after, once or until empty. Start is the confirmation."""

    #: The English phrase the detector grounds. ``""`` means anything; a real cell takes it only with
    #: ``options.pick_anything``.
    object: str = Field(default="", max_length=80)
    #: The operator's own words for it, for display.
    object_said: str | None = None
    place: PlaceIn
    #: ``home`` or a name in ``robot.named_poses``.
    return_to: str = "home"
    scope: TaskScope = "once"
    options: TaskOptionsIn = Field(default_factory=TaskOptionsIn)
    command: CommandProvenanceIn | None = None


class PlanPlaceOut(BaseModel):
    """Where the resolved plan puts the part."""

    kind: Literal["pose", "camera"]
    pose: str | None = None
    pose_label: str | None = None
    pose_joints_deg: list[float] | None = None
    phrase: str | None = None
    said: str | None = None


class TaskOptionsOut(BaseModel):
    """The options the task runs with, ``push_mm`` and ``rim_air_mm`` resolved."""

    multi_view: bool = True
    both_faces: bool = False
    closing_axis: ClosingAxisName | None = None
    push_mm: float | None = None
    #: Whether the operator set ``push_mm``: a distance asked for is taken as asked, the cell's may go longer.
    push_asked: bool = False
    #: As asked (``TaskOptionsIn.critical_parts``); null, the cell's.
    critical_parts: bool | None = None
    #: As asked (``TaskOptionsIn``); null, the cell's.
    rescan: bool | None = None
    push: bool | None = None
    clear: bool | None = None
    #: As asked (``TaskOptionsIn.blocker_into_the_place``); null, the cell's.
    blocker_into_the_place: bool | None = None
    record_views: bool = False
    rim_air_mm: float | None = None
    pick_anything: bool = False
    overlay: bool = True


class CommandProvenanceOut(BaseModel):
    """The command a task came from, as the request gave it."""

    text: str = ""
    source: CommandSource = "typed"
    language: str | None = None
    parsed: bool = False
    edited: list[str] = Field(default_factory=list)


class TaskPlanOut(BaseModel):
    """The resolved plan: echoed in ``run_started`` and in ``RunOut.plan``, and what a Restart runs again."""

    object: str
    object_said: str | None = None
    place: PlanPlaceOut
    return_to: str = "home"
    return_label: str | None = None
    #: The return pose's joints; null for home.
    return_joints_deg: list[float] | None = None
    scope: TaskScope = "once"
    options: TaskOptionsOut = Field(default_factory=TaskOptionsOut)
    command: CommandProvenanceOut | None = None
    #: ``look`` for a new task, ``return`` for a Restart: its first motion is the planned move to the return pose.
    first_motion: FirstMotion = "look"
    #: The run counts down 3 s before its first motion.
    countdown: bool = False


class RunRefusalOut(BaseModel):
    """A refusal the library met inside a run, before anything moved: the cell changed between the request and the run,
    and the run refused what the route would have refused."""

    code: RefusalCodeName
    #: The status the route answers this code with (``REFUSAL_STATUS``).
    status: int


class RunOut(BaseModel):
    """One run. The counts are the honest summary; the event stream is the detail."""

    id: str
    prompt: str
    requested_picks: int
    state: str
    started_at: float
    finished_at: float | None = None
    succeeded: int = 0
    attempted: int = 0
    #: Typed outcome strings, in order. Never free text.
    outcomes: list[str] = Field(default_factory=list)
    error: str = ""
    stop_requested: bool = False
    #: Which body ran.
    kind: RunKindName = "pick"
    #: Why it ended, typed; ``""`` while it runs. ``error`` carries the backend's sentence beside it.
    stop_code: Literal[StopCodeName, ""] = ""
    #: The class of ``stop_code``: what the frontend shows, and whether a recovery record was written.
    stop_class: Literal[StopClassName, ""] = ""
    #: The resolved task plan (a task, a Restart); null for the other kinds.
    plan: TaskPlanOut | None = None
    #: A task: the parts released at the place.
    parts_placed: int = 0
    #: The program believed a part was in the jaws at the end. Every gate reads the live hand first; the stop record
    #: carries this belief, which keeps ``part_still_held`` standing for a hand that can say nothing itself.
    holding: bool = False
    #: The run a Restart continues.
    restart_of: str | None = None
    #: "Halt now" was pressed during the run.
    halt_requested: bool = False
    #: The last timeline step (``countdown``, ``survey``, ``look``, ``detect``, ``grasp``, ``place``, ``return``).
    step: str = ""
    #: A task the library refused inside its run before anything moved (``stop_code`` ``cancelled``, no recovery
    #: record): the refusal a route would have answered, so the browser says it in its own words; null otherwise.
    refusal: RunRefusalOut | None = None


class TranscriptOut(BaseModel):
    """What Whisper heard, as the speech engine's `Transcript` reports it; `Transcript.to_dict()` whole."""

    #: The words as decoded: stripped, never lower-cased, never translated.
    text: str
    #: The Whisper language code the decoder prompt carried (``de``, ``en``), or null when it carried none.
    language: str | None
    #: ``detected`` when the engine picked German or English, ``configured`` when `models.stt.language`
    #: forced it.
    language_source: Literal["detected", "configured"]
    #: Length of the recording in seconds.
    duration_s: float
    #: Which engine decoded it, e.g. ``whisper-transformers``.
    engine: str
    #: The weights it ran: a local directory or a Hub id.
    model: str
    #: The torch device type the decode ran on.
    device: str
    #: Milliseconds from the audio to the text; the one-time weight load is not part of it.
    latency_ms: float


class SpeechCheckOut(BaseModel):
    """What the voice detector found before Whisper was asked; `SpeechCheck.to_dict()` whole."""

    #: An utterance closed: a window at or above `onset` and at least `min_speech_s` of speech.
    heard_speech: bool
    #: Length of the recording in seconds.
    duration_s: float
    #: Seconds of audio scored before the detector answered; it stops at the first utterance.
    checked_s: float
    #: The highest speech probability of any window scored.
    peak_probability: float
    #: The probability a window needs to open an utterance.
    onset: float
    #: The speech an utterance needs to be kept, in seconds.
    min_speech_s: float
    #: Which detector scored it, e.g. ``silero-vad``.
    detector: str
    #: Milliseconds the check took; the one-time model load is not part of it.
    latency_ms: float


class ProposalOut(BaseModel):
    """What one recording proposes for the prompt box, as the speech library's `Proposal` reports it.

    A proposal only: transcription never starts anything, and a human confirms the text before it becomes
    a prompt. The fields are `Proposal.to_dict()`, whole, so the console and a library caller read the
    same answer.
    """

    #: The words Whisper decoded, or an empty string when there are none to propose.
    text: str
    #: Why `text` is empty, as a sentence; null when it holds words.
    reason: str | None
    #: What the voice detector found before Whisper was asked.
    speech: SpeechCheckOut
    #: Whisper's report; null when the detector heard no speech and Whisper was not asked.
    transcript: TranscriptOut | None


class ListenIn(BaseModel):
    """How long a push to talk listen at the cell PC waits for the talk switch."""

    #: Seconds to wait for the switch to go down. The turn itself ends at the release, and at Whisper's
    #: 30 s window at the latest.
    timeout_s: float = Field(default=10.0, gt=0.0, le=60.0)


class TalkIn(BaseModel):
    """One edge of the talk switch: down when a person starts talking, up when they are done."""

    pressed: bool


class TalkOut(BaseModel):
    """The talk switch after the edge."""

    pressed: bool
    #: Presses counted since the console process started. A press while the switch is already down is
    #: not counted, so a key that repeats while it is held is one press.
    presses: int


class ViewfinderOut(BaseModel):
    """What the cell is looking at, and, always, which kind of picture this is.

    The console can show three things that look alike and mean different things: a colour frame taken
    from a device now, a synthetic scene this process drew for a rehearsal, and the grasp overlay the
    stack rendered during a pick. Rendering them identically is how a minutes-old segmentation, or a
    picture of a room that does not exist, ends up on a screen an operator reads as live. So
    ``source`` and ``age_s`` travel with every frame and ``human`` says it in words.

    ``image_base64`` is ``None`` for the four "no picture" cases, and ``reason`` names which. That is
    an answer, not an error: a cell with no camera is a legitimate cell, and a console that returned
    404 for it would make an ordinary state look like a fault.
    """

    #: ``"camera"`` | ``"synthetic"`` | ``"overlay"`` | ``"none"``. ``synthetic`` is a real picture
    #: this process drew: the rehearsal cell has no camera, and the client must not call it live.
    source: str
    #: ``not_built`` | ``pick_owns_camera`` | ``no_colour_source`` | ``encode_failed``; empty when
    #: there is a picture.
    reason: str
    human: str
    image_base64: str | None = None
    #: ``"image/jpeg"`` for a camera frame, ``"image/png"`` for an overlay. The client builds its own
    #: ``data:`` URI from this rather than assuming one.
    media_type: str = ""
    width: int = 0
    height: int = 0
    #: ``0.0`` for a picture taken now; for an overlay, seconds since this server first saw those
    #: bytes; ``None`` when there is no picture.
    age_s: float | None = None
    #: Whether ``age_s`` is the real age or only a floor. Nothing records when an overlay was
    #: rendered, so the first one a server sees can only be reported as "at least this old".
    age_is_exact: bool = True
    overlay_enabled: bool = False


class RollupOut(BaseModel):
    """KPIs over the logged records, with the unmeasurable ones named rather than zeroed.

    ``unmeasurable`` is the load-bearing field. A KPI computed from a value nothing ever writes
    returns a confident 0.0, not a blank, and a 0.0% false-positive-grasp rate on a demo screen is a
    claim this stack cannot make.
    """

    total_attempts: int
    kpis: dict[str, Any] = Field(default_factory=dict)
    #: Maps a KPI name to why no number can be given. Rendered where the number would have been.
    unmeasurable: dict[str, str] = Field(default_factory=dict)
    #: Maps a typed outcome to its count. The reason surface, tallied.
    outcomes: dict[str, int] = Field(default_factory=dict)
    source: str = ""
    record_log_path: str = ""
    #: False means this cell has logged nothing yet: a real state on a fresh bench, not a fault.
    record_log_exists: bool = False


class RecordOut(BaseModel):
    """One logged grasp attempt, as the frozen telemetry contract stores it."""

    #: Unix seconds, as the frozen record stores it, not a formatted string. Formatting belongs to
    #: whoever displays it, and a server that picked a format would pick the wrong timezone.
    timestamp: float
    attempt_id: str
    mode: str
    final_outcome: str
    #: Carried verbatim: it holds the "safety refused to move" signal and the vendor/model provenance,
    #: and re-shaping it here would be a second contract to keep in step with the frozen one.
    extra: dict[str, Any] = Field(default_factory=dict)


# --- the console of commit 2: the contract of build plan section 1 ------------------------------------------------


class CodesOut(BaseModel):
    """Every typed code the console answers with (``api/codes.py``), each list in declaration order.

    The frontend derives its unions from these fields (``Schemas['CodesOut']['stop_codes'][number]``), so every
    translation table over them is exhaustive at compile time.
    """

    run_kinds: list[RunKindName]
    #: The kinds that move the arm by themselves.
    moving_kinds: list[RunKindName]
    stop_codes: list[StopCodeName]
    stop_classes: list[StopClassName]
    stop_class_of: dict[StopCodeName, StopClassName]
    event_types: list[EventTypeName]
    refusal_codes: list[RefusalCodeName]
    light_ids: list[LightIdName]
    light_states: list[LightStateName]
    light_codes: list[LightCodeName]
    #: The codes each light may carry.
    light_codes_of: dict[LightIdName, list[LightCodeName]]
    blocker_codes: list[BlockerCodeName]
    jaws_stages: list[JawsStageName]
    jaws_choices: list[JawsChoiceName]
    command_notes: list[CommandNoteName]


class RestartIn(BaseModel):
    """Restart the console's recovery record as a new run whose first motion is the planned move home."""

    run_id: str = Field(min_length=1)


class HomeIn(BaseModel):
    """Where the Home button drives: ``home`` or a name in ``robot.named_poses``."""

    to: str = Field(default="home", min_length=1)


class AcknowledgeIn(BaseModel):
    """A person's word that the cell is clear, and, where they emptied the hand, that the jaws hold nothing."""

    #: Must be true: "the cell is clear" is said, never defaulted.
    cell_clear: Literal[True]
    #: The hand holds nothing (a hand that is not a toggle; a toggle answers the jaws question instead).
    jaws_empty: bool = False


class BrakeOut(BaseModel):
    """What "halt now" did. The run stops commanding; where the arm latches, nothing more is sent to it."""

    #: A run was active and was told to halt.
    run_halted: bool
    #: The arm's halt latch is set: every next motion and output switch is refused before anything is sent.
    latched: bool
    #: The move in flight is being braked (``robot.ur.brake_on_halt``); false where it runs to its end first.
    braking: bool
    #: A motion was in flight.
    in_motion: bool
    run_id: str | None = None
    message: str = ""


class TargetOut(BaseModel):
    """A place target the camera found and the task keeps (a bin), in the base frame, millimetres."""

    label: str
    score: float
    centre_mm: list[float]
    #: The rim's height: the 95th-percentile top of what was seen.
    rim_mm: float
    #: Width and length of the footprint.
    footprint_mm: list[float]
    #: Width and length of the opening, where the height map found one.
    opening_mm: list[float] | None = None
    #: The look it was seen from; null for a fixed camera.
    look: str | None = None
    seen_at: float
    #: The URL of the target overlay, where one was rendered.
    overlay: str | None = None


# --- cell facts (GET /v1/cell/facts), read once after build and once after connect --------------------------------

#: How a rig is mounted.
Mounting: TypeAlias = Literal["wrist", "fixed", "unknown"]


class RigOut(BaseModel):
    """One camera rig the cell's picks see through."""

    rig_id: str
    mounting: Mounting = "unknown"
    primary: bool = False


class PushFactsOut(BaseModel):
    can_push: bool = False
    #: Why this cell does not push; ``""`` where it does.
    why_not: str = ""
    default_mm: float | None = None
    ceiling_mm: float | None = None
    #: The cell's ``robot.grasping.recovery.critical_parts``: true, nothing is pushed and a blocker is cleared instead.
    critical_parts: bool = False
    #: The cell's ``robot.grasping.recovery.blocker_into_the_place``: true, a task that names no object sets a blocker
    #: down where the parts go; false, it is set aside on the support.
    blocker_into_the_place: bool = True
    #: Whether the cell's ``recovery.allowed_actions`` names ``rescan`` and ``nudge_target`` (with recovery on): what
    #: the console's ticks start from; a task may override them for itself.
    rescan_allowed: bool = False
    push_allowed: bool = False


class DetectorFactsOut(BaseModel):
    backend: str = ""
    router_enabled: bool = False
    vlm_model_id: str | None = None
    precision: str | None = None


class BrakeFactsOut(BaseModel):
    """What "halt now" can do on this arm."""

    #: The arm refuses every next motion once halted (the UR and the dummy).
    latches: bool = False
    #: The arm also brakes a move in flight (``robot.ur.brake_on_halt: true``).
    brakes_in_motion: bool = False


class PayloadFactsOut(BaseModel):
    """Whether the planner models a carried part; a task refuses to run unless it does (a real cuRobo arm)."""

    modelled: bool = False
    declined_reason: str | None = None
    length_mm: float | None = None


class RouteFactsOut(BaseModel):
    """How the arm's moves are planned: through a planner ``Robot.place`` and ``Robot.home`` accept, or not."""

    route: Literal["planned", "unplanned", "refused"]
    sentence: str = ""


class CellFactsOut(BaseModel):
    """What this cell is, read once: the cockpit's Advanced drawer, the stop buttons and the ready bar need it."""

    wrist_camera: bool = False
    cameras: list[RigOut] = Field(default_factory=list)
    #: The configured looks, by name.
    looks: list[str] = Field(default_factory=list)
    natural_closing_axis: str | None = None
    push: PushFactsOut = Field(default_factory=PushFactsOut)
    detector: DetectorFactsOut = Field(default_factory=DetectorFactsOut)
    #: Above this gap the hand-eye check says the calibration may have drifted. Nothing recalibrates.
    hand_eye_warn_mm: float = 6.0
    brake: BrakeFactsOut = Field(default_factory=BrakeFactsOut)
    payload: PayloadFactsOut = Field(default_factory=PayloadFactsOut)
    route: RouteFactsOut
    #: The rehearsal scene, not a camera ("Probe").
    rehearsal: bool = False


# --- readiness (GET /v1/cell/readiness): the ready bar before the first task --------------------------------------


class LightOut(BaseModel):
    """One light of the ready bar."""

    id: LightIdName
    state: LightStateName
    code: LightCodeName
    #: The backend's sentence, for the tech view.
    message: str = ""
    #: Takes part in ``ready``. The commands light never does.
    blocks: bool = True


class BlockerOut(BaseModel):
    """Why Start is disabled although the lights may be green."""

    code: BlockerCodeName
    message: str = ""
    run_id: str | None = None


class ReadinessOut(BaseModel):
    """``ready`` means every light that blocks reads ok and nothing blocks. The server enforces its gates itself."""

    ready: bool
    lights: list[LightOut] = Field(default_factory=list)
    blockers: list[BlockerOut] = Field(default_factory=list)


# --- the jaws question, answered in the browser (a toggle hand) ---------------------------------------------------


class JawQuestionOut(BaseModel):
    """The question a toggle hand asks, waiting for its answer. It has no default: unanswered, it is refused."""

    question_id: str
    stage: JawsStageName
    #: Asked while connecting, or by ``POST /v1/cell/jaws/check``.
    at: Literal["connect", "check"]
    #: The bank and pin, as the pendant shows it (``tool output 0``).
    where: str = ""
    reason: str = ""
    choices: list[JawsChoiceName]
    attempt: int = 1
    of: int = 3
    #: Why it is asked again, where it is.
    why_again: str = ""
    #: Unix seconds; after it the question counts as unanswered, never as "open".
    expires_at: float
    #: The driver's own question, for the tech view.
    text: str = ""


class JawsOut(BaseModel):
    hand: HandOut
    question: JawQuestionOut | None = None


class JawAnswerIn(BaseModel):
    question_id: str = Field(min_length=1)
    choice: JawsChoiceName


# --- the live image (GET /v1/camera/live) -------------------------------------------------------------------------

#: A frame from a camera now, a picture this process drew (the rehearsal cell), or none.
LiveSource: TypeAlias = Literal["camera", "synthetic", "none"]
#: Why there is no picture; ``measuring`` means a pick is grabbing, and the client keeps its last frame.
LiveReason: TypeAlias = Literal["", "not_built", "no_camera", "measuring", "encode_failed", "no_rig"]


class LiveFrameOut(BaseModel):
    """One display frame, never a measurement: read through ``Camera.peek``, which never waits."""

    rig_id: str | None = None
    rigs: list[RigOut] = Field(default_factory=list)
    source: LiveSource = "none"
    reason: LiveReason = ""
    #: JPEG, downscaled.
    image_base64: str | None = None
    width: int = 0
    height: int = 0
    captured_at: float | None = None
    age_s: float | None = None


# --- poses (GET /v1/poses) and teaching one by hand (POST /v1/teach) ----------------------------------------------

#: A taught pose's screen verdict; an unscreened or refused pose is never written.
PoseScreenName: TypeAlias = Literal["clear", "band"]


class PoseOut(BaseModel):
    """A named pose: Home from the config (read-only), or one taught in the console."""

    #: The YAML key: an ASCII identifier.
    name: str
    #: What the chat and the cards show, e.g. "Ablage links".
    label: str = ""
    joints_deg: list[float]
    source: Literal["config", "taught"] = "config"
    screen: PoseScreenName | None = None
    note: str = ""
    taught_at: str | None = None


class PosesOut(BaseModel):
    home: PoseOut
    poses: list[PoseOut] = Field(default_factory=list)
    default_place: str | None = None
    #: A pose can be taught now; ``why_not`` says why not.
    teachable: bool = False
    why_not: str = ""
    #: ``why_not`` as its code, for the browser's own words (``planner_not_ready``: the planner is still starting;
    #: ``no_layer``: the chain has no layer of the cell's own); ``""`` where a pose can be taught.
    why_not_code: RefusalCodeName | Literal[""] = ""
    #: The file a taught pose is written to, shown before the arm is freed.
    target_file: str | None = None


class DefaultPlaceIn(BaseModel):
    """The default place pose, or null to have none."""

    name: str | None


class PayloadOut(BaseModel):
    """The payload the controller compensates for: it decides whether a freed arm floats, sinks or rises."""

    mass_kg: float | None = None
    cog_mm: list[float] | None = None
    readable: bool = False
    source: str = ""


class PayloadSeenIn(BaseModel):
    """The payload the person saw and confirmed before the arm is freed. A value that is no number (JSON's ``NaN`` or
    ``Infinity``) is a malformed request (422), never a payload anybody saw."""

    mass_kg: float | None = Field(default=None, allow_inf_nan=False)
    cog_mm: list[Annotated[float, Field(allow_inf_nan=False)]] | None = None


class TeachIn(BaseModel):
    """Teach one pose by hand. The names are checked by the route (``invalid_name``, ``invalid_label``)."""

    #: The YAML key: an ASCII identifier of at most 32 characters, not a keyword and not ``home``.
    name: str
    #: Free text of at most 40 characters, no newline, no " #".
    label: str
    #: ``place``: the fingertips go where the part's bottom is let go.
    role: Literal["place", "other"] = "other"
    replace: bool = False
    make_default_place: bool = False
    payload_seen: PayloadSeenIn | None


class TeachStartOut(BaseModel):
    """The teach run, and the token its polls, capture and cancel carry."""

    run: RunOut
    token: str


#: Where a teach session stands.
TeachState: TypeAlias = Literal[
    "freeing", "free", "holding_when_still", "holding", "screening", "saved", "refused", "not_saved", "ended",
]
#: What the exact guard and the planner say about a pose (the library's ``PoseVerdict``).
ScreenVerdict: TypeAlias = Literal["clear", "band", "guard_refused", "planner_refused", "unscreened"]


class TeachStateOut(BaseModel):
    """A teach session, polled: every poll is the browser's heartbeat."""

    state: TeachState
    #: The arm is outside the cell's limits where it is held now.
    outside: bool = False
    lines: list[str] = Field(default_factory=list)
    joints_deg: list[float] | None = None
    tcp_mm: list[float] | None = None
    verdict: ScreenVerdict | None = None
    #: A nearby configuration both authorities clear, where the pose was refused: one to teach instead.
    nearby_deg: list[float] | None = None
    message: str = ""
    time_left_s: float | None = None


# --- commands (POST /v1/commands/parse): the VLM reads the sentence, a person confirms --------------------------


class CommandIn(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    source: CommandSource = "typed"
    language: str | None = None


class CommandPhraseOut(BaseModel):
    """One field the reader filled: the English phrase, the operator's words, and how the detector would route it."""

    phrase: str
    said: str | None = None
    #: The phrase was found in the sentence; false shows "please check".
    verified: bool = False
    route: RoutePreviewOut | None = None


class CommandModelOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    latency_ms: float
    attempts: int = 1
    loaded_now: bool = False


#: How the console answers a greeting, ``runtime.greeting.wave`` of the app config
#: (``src.config.schema.runtime.GreetingWave``, which ``tests/test_a_greeting_waves.py`` pins against this).
GreetingAnswer: TypeAlias = Literal["direct", "confirm", "off"]


class CommandOut(BaseModel):
    """What the reader understood. It creates no run and touches no cell; Start does, after a person looked."""

    understood: bool
    intent: Literal["task", "stop", "none"]
    object: CommandPhraseOut | None = None
    place: CommandPhraseOut | None = None
    #: A pose NAME: the reader is handed label -> name pairs of the taught poses.
    place_pose: str | None = None
    scope: TaskScope | None = None
    count: int | None = None
    return_to: str | None = None
    notes: list[CommandNoteName] = Field(default_factory=list)
    reason: str = ""
    model: CommandModelOut | None = None
    #: The model's raw answer, at most 500 characters.
    raw: str = ""
    #: The sentence greets Willy, bids it goodbye or asks it to wave, and how the console answers it, as the app config
    #: says (``runtime.greeting.wave``): ``direct`` waves at once (``POST /v1/cell/wave``), ``confirm`` asks first in a
    #: dialog like Home's, ``off`` greets back only. ``None`` for any other sentence. The reading moves nothing.
    greeting: GreetingAnswer | None = None


#: The command reader's state (the commands light carries the same codes).
CommandState: TypeAlias = Literal["ready", "idle", "loading", "missing", "failed", "not_configured"]


class CommandStatusOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    state: CommandState
    model_id: str | None = None
    weights_present: bool | None = None
    cause: str = ""
    last_latency_ms: float | None = None
    #: One VLM copy reads commands and detects.
    shared_with_detection: bool = False
