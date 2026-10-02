"""The live image: a display frame of a camera, during a run too, never a measurement.

It reads through ``Camera.peek``, which never waits and never grabs, so it cannot take a frame from a pick: during a
measuring grab it answers ``measuring`` and the browser keeps its last frame. ``GET /v1/camera`` stays as it is.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from api.cell import Console, console
from api.live import LIVE_MAX_WIDTH, LIVE_QUALITY
from api.schemas import LiveFrameOut

router = APIRouter(prefix="/camera", tags=["media"])


@router.get("/live", response_model=LiveFrameOut, summary="A live display frame of one camera")
def get_live(
    cell: Annotated[Console, Depends(console)],
    rig: str | None = None,
    max_width: Annotated[int, Query(ge=32, le=4096)] = LIVE_MAX_WIDTH,
) -> LiveFrameOut:
    """Downscaled to ``max_width`` and JPEG-encoded; ``rig`` picks the camera (the primary where omitted).

    Declared ``def``: a peek on a device runs on the thread pool, never on the event loop that streams a run's events.
    No picture is an answer, not an error: ``not_built``, ``no_camera``, ``no_rig``, ``measuring`` (a pick is grabbing;
    keep the last frame), ``encode_failed``. The rehearsal cell's picture is ``synthetic``, never ``camera``.
    """
    try:
        quality = int(cell.config().runtime.image_encoding.frame_quality)
    except Exception:  # noqa: BLE001 (a tree that does not load still shows the picture, at the default quality)
        quality = LIVE_QUALITY
    cell.live.quality = quality
    return cell.live.frame(cell.session.service, rig=rig, max_width=max_width)
