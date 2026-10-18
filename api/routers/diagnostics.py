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
    VendorReadinessOut,
)
from src.models.perception_spec import PerceptionSpec
from src.utility.log_cfg import create_logger

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])

#: Only the probes that an operator asked for are logged. ``_perception`` is excluded on purpose: it is re-read by
#: ``GET /v1/cell`` every 2 seconds (the console's shared poll, ``useCell``) and every second in the audience window,
#: so a line each would be a poll trace rather than a record of anything.
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
    selected = robot.arm
    address = selected.controller if vendor == "ur" else None

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


def _vlm_weights_present(model_id: str, weights_root: str | None = None) -> bool:
    """Is the checkpoint on this box, in the weights folder or the Hugging Face cache? Never downloads, never loads,
    never raises (``src.models.weights``, the resolver the model reads its weights through)."""
    from src.models.weights import weights_present

    return weights_present(model_id, kind="vlm", root=weights_root)


def _perception(cell: Console) -> PerceptionStackOut:
    """Read the configured stack. Loads no model: an operator asks this before spending VRAM."""
    # `Console.config` is a method, not an attribute. Read without the call it yields a bound method
    # object, which has no `models` and is truthy, so the guard below would not fire and the endpoint
    # would report a stack whatever the tree says.
    try:
        app_cfg = cell.config()
    except Exception as exc:  # noqa: BLE001 (a config that will not load is "cannot answer", not 500)
        logger.warning(
            "Cannot read the config while reporting the perception stack (%s: %s).", type(exc).__name__, exc,
        )
        app_cfg = None
    models = getattr(app_cfg, "models", None) if app_cfg is not None else None
    if models is None:
        return PerceptionStackOut(
            kind="zero_shot", backend="vlm", segmenter="sam2",
            detail="The config does not load, so which stack this cell builds cannot be read; "
                   "`python -m src.config` names what is wrong.",
        )

    # The branch logic is not re-derived here. This reads the same resolution the builder acts on, so
    # the panel cannot describe another stack than the one the cell builds.
    resolution = PerceptionSpec.from_config(models).resolve()
    #: The wire vocabulary is not the resolution's: `PerceptionStackOut.backend` says "vlm | rtdetr", and a browser
    #: reads it.
    backend = "rtdetr" if resolution.detector == "rtdetr" else "vlm"

    if resolution.refusal:
        # A stack that cannot be built is the most useful thing this endpoint can say, and it says it
        # here rather than leaving the refusal to arrive as a traceback at cell construction. The words
        # are the builder's own, not a paraphrase.
        return PerceptionStackOut(
            kind=resolution.kind, backend=backend, segmenter=resolution.segmenter,
            detail=f"This cell CANNOT build a perception stack: {resolution.refusal}",
        )

    if resolution.kind == "closed_set":
        return PerceptionStackOut(
            kind="closed_set", backend="rtdetr", segmenter=resolution.segmenter,
            detail="Closed-set stack: RT-DETR answers with its trained class list and reads a prompt only as a "
                   "class-name filter. Free text is refused rather than silently mismatched.",
        )

    model_id = resolution.vlm_model_id or ""
    present = _vlm_weights_present(model_id, models.weights_root)
    if present:
        detail = f"VLM stack ready ({model_id}): it grounds every prompt, and SAM2 cuts each box."
    else:
        from src.models.weights import fetch_hint

        detail = (f"VLM weights for {model_id} are NOT on this box, so every pick is refused until they are: "
                  f"{fetch_hint(model_id).replace('`', '')}.")
    return PerceptionStackOut(
        kind="zero_shot", backend="vlm", segmenter=resolution.segmenter,
        vlm_model_id=model_id, vlm_weights_present=present, detail=detail,
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
