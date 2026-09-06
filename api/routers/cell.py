"""Bringing the cell up and down.

Four endpoints and one shape: look, then acknowledge, then act. ``GET /v1/cell`` says where the cell
is; ``POST /v1/cell/build`` assembles it without touching a robot; ``GET /v1/cell/connect-preview``
names every motion connecting will cause and hands back a token; ``POST /v1/cell/connect`` accepts only
that token.

The token is not ceremony. On this cell "connect" is motion: a Robotiq activates by sweeping its full
finger travel, and a vacuum cup asserts its ejector and drops whatever it holds. A browser button is
reachable by a stray curl, a replayed request and a reloaded tab, so requiring a token that a preview
issued for this configuration means a connect cannot happen without something having read what it
would do.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from api.cell import Console, console
from api.constants import API_LOG_DIR, ROUTER_CELL_LOG_FILE
from api.lifecycle import CellState, CellTransitionError, ConnectRefused
from api.schemas import (
    CellOut,
    ConnectIn,
    ConnectPreviewOut,
    MotionWarningOut,
    TelemetryOut,
)
from api.telemetry import read_telemetry
from src.utility.log_cfg import create_logger

router = APIRouter(prefix="/cell", tags=["cell"])

#: The typed refusals themselves are logged where they are decided (``api.lifecycle``), and every
#: failure envelope is logged once more as it leaves the server (``api.app``). What is left for this
#: file is what only it knows: the cross-process lock, and why a build refused.
logger = create_logger("CellRouter", ROUTER_CELL_LOG_FILE, log_dir=API_LOG_DIR)

#: Each typed refusal maps to one HTTP status. Each is a different thing for the operator to do, which
#: is the whole reason the reasons are typed: 409 means "fix the cell", 403 means "fix the config",
#: 428 means "read the preview first".
_STATUS: dict[ConnectRefused, int] = {
    ConnectRefused.NOT_BUILT: 409,
    ConnectRefused.WRONG_STATE: 409,
    ConnectRefused.CELL_BUSY: 409,
    ConnectRefused.NOT_ACKNOWLEDGED: 428,   # Precondition Required
    ConnectRefused.STALE_TOKEN: 428,
    ConnectRefused.NO_REAL_GRIPPER: 403,
    ConnectRefused.DRIVER_REFUSED: 502,     # the refusal came from the controller, not from this server
}


def _refuse(error: CellTransitionError) -> HTTPException:
    return HTTPException(
        status_code=_STATUS.get(error.reason, 409),
        detail={"code": str(error.reason), "message": str(error), "detail": {}},
    )


def _render(cell: Console) -> CellOut:
    from src.robot.execution.cell_lock import peek

    # Reused, not re-derived: one reading of the perception stack, so /v1/cell and /v1/diagnostics can
    # never disagree about which models this cell would use.
    from api.routers.diagnostics import _perception

    session = cell.session
    substitution = session.substitution
    key = cell.lock_key()
    holder = peek(key) if key else None
    return CellOut(
        state=str(session.state),
        arm=type(session.arm).__name__ if session.arm is not None else None,
        gripper=type(session.gripper).__name__ if session.gripper is not None else None,
        vendor=str(cell.robot().vendor),
        profile=cell.profile,
        perception=_perception(cell),
        # A substituted gripper is reported whether or not anything is connected: it is a property of
        # what was built, and the operator needs it before they reach for the connect button.
        gripper_substitution=(
            {
                "reason": str(substitution.reason),
                "requested": substitution.requested,
                "detail": substitution.detail,
                "fix": substitution.fix,
            }
            if substitution is not None else None
        ),
        lock_key=key,
        # Who holds the cell right now: this console, or the CLI runner in another terminal.
        # A snapshot by nature: it can be stale the instant it is read, which is why it never gates
        # anything. The OS arbitrates the actual acquisition.
        lock_holder=holder.describe() if holder is not None else None,
        active_run_id=cell.active_run_id,
    )


@router.get("", response_model=CellOut, summary="Where is the cell?")
def get_cell(cell: Annotated[Console, Depends(console)]) -> CellOut:
    return _render(cell)


@router.post("/build", response_model=CellOut, summary="Assemble the cell (no robot is touched)")
def post_build(
    cell: Annotated[Console, Depends(console)],
    rehearse: bool = False,
) -> CellOut:
    """Build through the same functions the CLI runner uses.

    ``rehearse=true`` builds the desk scene instead of opening a camera and loading models, which is how
    the whole console stays playable with no hardware attached. Not cheap otherwise: a real build takes
    tens of seconds while the detector and segmenter load onto the GPU.
    """
    try:
        cell.build(rehearse=rehearse)
    except CellTransitionError as error:
        raise _refuse(error) from error
    except Exception as exc:  # noqa: BLE001 (a build refusal is a designed outcome, not a crash)
        # The message travels to the browser and nowhere else. On a real cell this is the camera that
        # would not open or the weights that are not on the box, and it is the first thing anyone
        # asks about afterwards.
        logger.error("Build refused: %s: %s", type(exc).__name__, exc)
        raise HTTPException(
            status_code=422,
            detail={
                "code": "build_refused",
                "message": f"{type(exc).__name__}: {exc}",
                "detail": {},
            },
        ) from exc
    return _render(cell)


@router.get(
    "/connect-preview",
    response_model=ConnectPreviewOut,
    summary="What will move, and the token that acknowledges it",
)
def get_connect_preview(cell: Annotated[Console, Depends(console)]) -> ConnectPreviewOut:
    try:
        preview = cell.session.preview(cell.robot(), cell.fingerprint())
    except CellTransitionError as error:
        raise _refuse(error) from error
    return ConnectPreviewOut(
        token=preview.token,
        expires_at=preview.expires_at,
        arm=preview.arm,
        gripper=preview.gripper,
        warnings=[
            MotionWarningOut(subject=w.subject, what=w.what, precaution=w.precaution)
            for w in preview.warnings
        ],
        blocking=list(preview.blocking),
    )


@router.post("/connect", response_model=CellOut, summary="Bring the cell up (THIS MOVES)")
def post_connect(cell: Annotated[Console, Depends(console)], body: ConnectIn) -> CellOut:
    """Arm first, then gripper, and roll the arm back if the gripper refuses.

    The cross-process lock is taken before the arm, so a cell already owned by the CLI runner is
    refused without a single command reaching the controller.
    """
    from src.robot.execution.cell_lock import CellBusy, CellLock

    key = cell.lock_key()
    lock = CellLock(key, owner="operator console") if key else None
    if lock is not None:
        try:
            lock.acquire()
            logger.info("Cell lock %s acquired for the operator console.", key)
        except CellBusy as busy:
            # Which other process owns the cell, usually the CLI runner in another terminal. The
            # HTTP envelope carries the holder to the browser; this keeps it after the tab is closed.
            logger.warning(
                "Connect refused: cell %s is held by %s.",
                key, busy.holder.describe() if busy.holder else "another process",
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "code": str(ConnectRefused.CELL_BUSY),
                    "message": str(busy),
                    "detail": {"holder": busy.holder.describe() if busy.holder else None},
                },
            ) from busy

    try:
        cell.session.connect(body.token, cell.fingerprint(), lock)
    except CellTransitionError as error:
        # connect() already released the lock on the paths it took; releasing again is a no-op, and
        # doing it here covers the refusals raised before it ever got that far.
        if lock is not None:
            lock.release()
        raise _refuse(error) from error
    return _render(cell)


@router.get("/status", response_model=TelemetryOut, summary="Live telemetry (receive stream only)")
def get_status(
    cell: Annotated[Console, Depends(console)],
    include_controller_state: bool = False,
) -> TelemetryOut:
    """A snapshot cheap enough to poll.

    The default reading is the RTDE output stream: pose, joints, TCP wrench, which the controller is
    already broadcasting, so polling it costs a running motion nothing. ``get_joint_torques`` is
    deliberately absent: it goes through the control interface a pick is using, and it is never
    offered here.

    ``include_controller_state=true`` adds robot mode, safety mode and the human-readable safety text,
    and costs a dashboard socket round trip per call, which is why it is opt-in rather than part of
    the default tick. Poll it at seconds, not at frames.

    Read-only in the strong sense: there is no counterpart that clears a protective stop. That is done
    where the arm is visible, at the pendant, for the same reason there is no E-stop button here.
    """
    telemetry = read_telemetry(cell.session.arm, include_controller_state=include_controller_state)
    return TelemetryOut(
        state=str(cell.session.state),
        connected=telemetry.connected,
        simulated=telemetry.simulated,
        vendor=telemetry.vendor,
        model=telemetry.model,
        tcp_position_mm=list(telemetry.tcp_position_mm) if telemetry.tcp_position_mm else None,
        tcp_quaternion_xyzw=(
            list(telemetry.tcp_quaternion_xyzw) if telemetry.tcp_quaternion_xyzw else None
        ),
        joint_positions=list(telemetry.joint_positions) if telemetry.joint_positions else None,
        tcp_force_n=list(telemetry.tcp_force_n) if telemetry.tcp_force_n else None,
        tcp_torque_nm=list(telemetry.tcp_torque_nm) if telemetry.tcp_torque_nm else None,
        robot_mode=telemetry.robot_mode,
        safety_mode=telemetry.safety_mode,
        protective_stopped=telemetry.protective_stopped,
        emergency_stopped=telemetry.emergency_stopped,
        controller_message=telemetry.controller_message,
        controller_is_simulator=telemetry.controller_is_simulator,
        controller_serial=telemetry.controller_serial,
        controller_host=telemetry.controller_host,
        unavailable=telemetry.unavailable,
        controller_state_included=include_controller_state,
    )


@router.post("/disconnect", response_model=CellOut, summary="Take the cell down")
def post_disconnect(cell: Annotated[Console, Depends(console)]) -> CellOut:
    """Gripper first, then arm: the reverse of connect, so a cup's release still reaches the I/O.

    Idempotent: disconnecting an already-disconnected cell is a no-op rather than an error, because the
    one thing an operator must always be able to do is put the cell down.
    """
    cell.session.disconnect()
    return _render(cell)


__all__ = ["CellState", "router"]
