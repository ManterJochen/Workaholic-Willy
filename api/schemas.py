"""Wire types for the console.

These serialise what the library already decided; they add no judgement of their own. Every verdict,
every provenance record and every validation error comes from ``src``. A field here that has to
compute something is a sign the logic belongs in the library instead.
"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field

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

    #: The text prompt. Empty means "whatever the perception source already targets".
    prompt: str = ""
    picks: int = Field(default=1, ge=1, le=100)


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


class TranscriptOut(BaseModel):
    """What was heard. Text only: transcription never starts anything on its own."""

    text: str


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
