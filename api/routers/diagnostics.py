"""The buttons that move nothing: is the software there, can it plan, is the controller reachable?

Everything here is read-only in the strongest sense: no motion, no control script, no write to the
controller. It exists because a cell that will not come up should tell an operator which thing is
wrong instead of "it does not work", and because at a bench the cheapest checks should come first.

Four states, not two. The preflight already speaks in ``ok / block / warn / bench``, and collapsing
those into a green light would turn two deliberately-named honesty gaps into false confidence:
``doctor`` reports whether a module can be found, not whether it imports or matches its pin; the
cuRobo reading means the sidecar's Python exists on disk and explicitly does not verify the robot
descriptor inside that separate environment. Both say so in their own payload.
"""

from __future__ import annotations

import socket
import time
from typing import Annotated

from fastapi import APIRouter, Depends

from api.cell import Console, console
from api.constants import API_LOG_DIR, ROUTER_DIAGNOSTICS_LOG_FILE
from api.schemas import (
    DiagnosticsOut,
    MotionStackOut,
    PerceptionStackOut,
    ReachabilityOut,
    RoutePreviewOut,
    VendorReadinessOut,
)
from src.models.perception_spec import PerceptionSpec
from src.utility.log_cfg import create_logger

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])

#: Only the probes that an operator asked for are logged. ``_perception`` and ``/route`` are excluded
#: on purpose: the first is re-read by ``GET /v1/cell`` every 3-5 seconds (``App.tsx``, ``Demo.tsx``)
#: and the second runs on a 350 ms debounce while a prompt is being typed, so a line each would be a
#: poll trace rather than a record of anything.
logger = create_logger("DiagnosticsRouter", ROUTER_DIAGNOSTICS_LOG_FILE, log_dir=API_LOG_DIR)

#: The RTDE port a UR controller listens on. Used only for a connect/close reachability probe: no
#: RTDE session is opened and no control script is uploaded, so this cannot disturb a running program.
_UR_RTDE_PORT = 30004
_REACH_TIMEOUT_S = 1.5


def _sdks(cell: Console) -> list[VendorReadinessOut]:
    from src.robot.drivers.doctor import arm_vendor_readiness, gripper_vendor_readiness

    # Constructed from `to_dict()`, not transcribed field by field: the wire model is handed the whole
    # mapping, so a field added to `VendorReadiness` either arrives or is rejected by name rather than
    # being silently absent from the console.
    return [VendorReadinessOut(**entry.to_dict())
            for entry in (*arm_vendor_readiness(), *gripper_vendor_readiness())]


def _motion_stack(cell: Console) -> MotionStackOut:
    """The cuRobo + exact-mesh reading, for the model this cell is configured to drive.

    Imported rather than shelled out, so the console cannot drift from the CLI by forgetting a flag.
    A UR3e cell being told about UR5e descriptors is the failure this reading exists to prevent.
    """
    from src.robot.safety.planning.stack import MotionStack

    # One ladder, not two: the three steps live in `safety/planning/__main__.py`, and the console reads
    # that answer through the report's provenance field instead of deriving its own. Which key decided
    # which robot the plan targets then has exactly one answer.
    #
    # `model_source` is a human-readable provenance string, not an identifier. Where nothing declares a
    # model it takes the longer form that names the missing key.
    reading = MotionStack.from_robot_config(cell.robot()).probe()
    model, source, env = reading.model, reading.model_source, reading.environment
    # `model` and `model_source` together are the whole point of the reading: the model says what the
    # plan was built for and the source says which key decided it.
    logger.info(
        "Motion stack for %s (from %s): fully_anchored=%s, cuRobo=%s, collision engine=%s, "
        "mesh bundle=%s.",
        model, source, env.fully_anchored, env.curobo.available, env.collision.engine,
        env.collision.mesh_bundle_present,
    )
    return MotionStackOut(
        model=model,
        model_source=source,
        fully_anchored=env.fully_anchored,
        curobo_available=env.curobo.available,
        curobo_python=env.curobo.python_path,
        curobo_robot_config=env.curobo.robot_config,
        curobo_caveat=(
            "'available' means the cuRobo environment's Python exists on disk. The robot descriptor "
            "named above lives INSIDE that separate environment and is not verified from here; a "
            "cell configured for a model with no descriptor passes this and fails later, in the "
            "sidecar."
        ),
        collision_engine=env.collision.engine,
        mesh_bundle_present=env.collision.mesh_bundle_present,
        coal_prefix=env.collision.coal_prefix,
    )


def _reachability(cell: Console) -> ReachabilityOut:
    """Is there anything at the controller's address? Opens a TCP socket and closes it.

    Deliberately the weakest useful probe. It does not open an RTDE session, upload a control script, or
    read a single register, so it cannot disturb a program someone else is running, and correspondingly
    it proves only that something is listening. It does not prove the controller is powered past boot,
    in Remote Control, or free. This has never been run against a physical controller.
    """
    robot = cell.robot()
    vendor = str(robot.vendor)
    address = None
    if vendor == "ur":
        address = getattr(getattr(robot, "ur", None), "ip", None)
    elif vendor == "kuka":
        address = getattr(getattr(robot, "kuka", None), "controller_ip", None)

    if not address:
        return ReachabilityOut(
            checked=False, address=None, port=None, reachable=None, latency_ms=None,
            detail=f"vendor {vendor!r} configures no controller address; nothing to reach.",
        )

    started = time.perf_counter()
    try:
        with socket.create_connection((address, _UR_RTDE_PORT), timeout=_REACH_TIMEOUT_S):
            pass
        elapsed = (time.perf_counter() - started) * 1000.0
        logger.info(
            "Something is listening on %s:%d (%.1f ms). That is all this proves.",
            address, _UR_RTDE_PORT, elapsed,
        )
        return ReachabilityOut(
            checked=True, address=address, port=_UR_RTDE_PORT, reachable=True,
            latency_ms=round(elapsed, 1),
            detail="something is listening. That is ALL this proves: not powered, not in Remote "
                   "Control, not free; those are the [bench] rows on the preflight.",
        )
    except OSError as exc:
        elapsed = (time.perf_counter() - started) * 1000.0
        logger.warning(
            "Nothing answered on %s:%d within %.1f s (%s: %s).",
            address, _UR_RTDE_PORT, _REACH_TIMEOUT_S, type(exc).__name__, exc,
        )
        return ReachabilityOut(
            checked=True, address=address, port=_UR_RTDE_PORT, reachable=False,
            latency_ms=round(elapsed, 1),
            detail=f"nothing answered on {address}:{_UR_RTDE_PORT} within {_REACH_TIMEOUT_S:g}s "
                   f"({type(exc).__name__}). Check the address, the cable, and that the controller is "
                   f"powered.",
        )


def _vlm_weights_present(model_id: str, model_path: str | None) -> bool:
    """Is the checkpoint on this box? Never downloads, never loads, never raises."""
    if model_path:
        from pathlib import Path

        return Path(model_path).exists()
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=model_id, local_files_only=True)
    except Exception:  # noqa: BLE001 (any failure means "not usable here", which is the answer)
        return False
    return True


def _perception(cell: Console) -> PerceptionStackOut:
    """Read the configured stack. Loads no model: an operator asks this before spending VRAM."""
    # `Console.config` is a method, not an attribute. Read without the call it yields a bound method
    # object, which has no `models` and is truthy, so the guard below would not fire and the endpoint
    # would report the legacy zero_shot stack whatever the tree says.
    try:
        app_cfg = cell.config()
    except Exception as exc:  # noqa: BLE001 (a config that will not load is "cannot answer", not 500)
        # This fallback answers with the legacy zero_shot stack, which is a confident answer derived
        # from a failure. The warning is what separates the two: a tree that stopped loading otherwise
        # looks exactly like a tree that configures no pipeline.
        logger.warning(
            "Cannot read the config while reporting the perception stack (%s: %s); "
            "falling back to the legacy answer.", type(exc).__name__, exc,
        )
        app_cfg = None
    models = getattr(app_cfg, "models", None) if app_cfg is not None else None
    if models is None:
        return PerceptionStackOut(
            pipeline_configured=False, kind="zero_shot", backend="grounded_sam",
            segmenter="sam2", router_enabled=False,
            detail="No models.pipeline block: the legacy keys decide everything (GroundingDINO + SAM2). "
                   "This is the default and is byte-identical to every cell that predates the pipeline.",
        )

    # The branch logic is not re-derived here. This reads the same resolution the builder acts on, so
    # the panel cannot describe a VLM stack for a cell that will run GroundingDINO: there is nothing
    # left in this file for the builder to disagree with.
    resolution = PerceptionSpec.from_config(models).resolve()
    #: The wire vocabulary is not the resolution's. `PerceptionStackOut.backend` says
    #: "grounded_sam | vlm | rtdetr" and a browser reads it, so the map is here rather than in the
    #: value: a routed stack is a VLM stack whose `router_enabled` is true.
    backend = {
        "groundingdino": "grounded_sam", "rtdetr": "rtdetr", "vlm": "vlm", "routed": "vlm",
    }[resolution.detector]
    configured = resolution.source == "models.pipeline"

    if resolution.refusal:
        # A stack that cannot be built is the most useful thing this endpoint can say, and it says it
        # here rather than leaving the refusal to arrive as a traceback at cell construction. The words
        # are the builder's own, not a paraphrase.
        return PerceptionStackOut(
            pipeline_configured=configured, kind=resolution.kind, backend=backend,
            segmenter=resolution.segmenter, router_enabled=resolution.router_enabled,
            detail=f"This cell CANNOT build a perception stack: {resolution.refusal}",
        )

    if resolution.kind == "closed_set":
        return PerceptionStackOut(
            pipeline_configured=configured, kind="closed_set", backend="rtdetr",
            segmenter=resolution.segmenter, router_enabled=False,
            detail="Closed-set stack: RT-DETR answers with its trained class list and ignores the "
                   "prompt text. Free-text prompts are refused rather than silently mismatched.",
        )

    if backend != "vlm":
        legacy = "" if configured else (
            " No models.pipeline block, so the legacy keys decide everything; the default, and "
            "byte-identical to every cell that predates the pipeline."
        )
        return PerceptionStackOut(
            pipeline_configured=configured, kind="zero_shot", backend=backend,
            segmenter=resolution.segmenter, router_enabled=False,
            detail=f"Phrase grounding only (GroundingDINO + {resolution.segmenter}). Complex "
                   f"prompts are NOT routed anywhere better; they are grounded by a model that fails "
                   f"confidently on them.{legacy}",
        )

    model_id = resolution.vlm_model_id or ""
    vlm = models.pipeline.zero_shot.vlm if models.pipeline is not None else None
    present = _vlm_weights_present(model_id, vlm.model_path if vlm is not None else None)
    routed = resolution.router_enabled
    if present:
        detail = (f"VLM route ready ({model_id}). "
                  + ("Each prompt is routed; simple ones still go to the phrase grounder."
                     if routed else "EVERY prompt goes to the VLM; the router is off."))
    else:
        consequence = ("complex prompts will be refused"
                       if resolution.vlm_on_unavailable == "refuse"
                       else "complex prompts will fall back to the phrase grounder, which returns a "
                            "confident box for the WRONG object")
        detail = (f"VLM weights for {model_id} are NOT on this box, so {consequence}. "
                  f"Fetch them: python scripts/model_weights/fetch.py")
    return PerceptionStackOut(
        pipeline_configured=configured, kind="zero_shot", backend="vlm",
        # Read from the resolution, not pinned to "sam2": the builder reads the same config key.
        segmenter=resolution.segmenter,
        router_enabled=routed, vlm_model_id=model_id, vlm_weights_present=present,
        vlm_on_unavailable=resolution.vlm_on_unavailable, detail=detail,
    )


@router.get("/route", response_model=RoutePreviewOut,
            summary="Which route a prompt would take; no GPU, no model, no image")
def preview_route(prompt: str, cell: Annotated[Console, Depends(console)]) -> RoutePreviewOut:
    """Answer 'why was that pick slow / why was it refused' by typing the prompt instead of running it.

    Pure text analysis, so it costs nothing and works on a cell with no weights at all. It also reports
    whether the chosen route could actually run here, which is the difference between "the router would
    send this to the VLM" and "this pick will work".
    """
    from src.models.routing import Route, normalize_simple_prompt, route

    decision = route(prompt)
    stack = _perception(cell)
    raw_signals = decision.to_dict().get("signals")
    signals: dict[str, object] = raw_signals if isinstance(raw_signals, dict) else {}

    runnable, blocked = True, None
    if decision.route is Route.VLM:
        if stack.backend != "vlm":
            runnable = False
            blocked = ("this prompt needs the VLM route, but this cell configures none; it will be "
                       "ground by the phrase detector, which fails confidently on prompts like this.")
        elif stack.vlm_weights_present is False:
            runnable = False
            blocked = f"the VLM route is configured but {stack.vlm_model_id} is not on this box."
        elif not stack.router_enabled:
            blocked = None  # every prompt goes to the VLM anyway; still runnable

    return RoutePreviewOut(
        prompt=prompt,
        route=str(decision.route),
        reason=str(decision.reason),
        description=decision.describe(),
        normalized_prompt=(normalize_simple_prompt(prompt) if decision.route is Route.SIMPLE else None),
        signals=signals,
        runnable=runnable,
        blocked_reason=blocked,
    )


@router.get("", response_model=DiagnosticsOut, summary="Everything that can be checked without moving")
def get_diagnostics(cell: Annotated[Console, Depends(console)]) -> DiagnosticsOut:
    """One call, because an operator at a bench wants one panel, not four requests.

    Ordered the way the runbook is: software first (free), then the planner (free), then perception
    (free), then the network (cheap). Each stage's failure explains the next stage's failure, which
    is the reason the runbook has an order.
    """
    return DiagnosticsOut(
        vendors=_sdks(cell),
        motion_stack=_motion_stack(cell),
        perception=_perception(cell),
        reachability=_reachability(cell),
    )
