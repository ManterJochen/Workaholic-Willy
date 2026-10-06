"""The live image: a display frame of a camera, during a run too, never a measurement.

It reads through ``Camera.peek``, which never waits and never grabs, so it cannot take a frame from a pick: during a
measuring grab it answers ``measuring`` and the browser keeps its last frame. ``GET /v1/camera`` stays as it is.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

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


#: The stream's pace: about the camera's own 30 frames a second, never faster than a new frame arrives.
STREAM_FPS = 30.0
#: How long the stream waits before it says "no picture" again while the camera gives none.
_IDLE_S = 0.2


@router.get("/live.mjpeg", summary="The live image as a continuous MJPEG stream")
def get_live_stream(
    cell: Annotated[Console, Depends(console)],
    rig: str | None = None,
    max_width: Annotated[int, Query(ge=32, le=4096)] = LIVE_MAX_WIDTH,
    fps: Annotated[float, Query(gt=0.0, le=60.0)] = STREAM_FPS,
) -> StreamingResponse:
    """The same display frames as ``/live``, pushed as ``multipart/x-mixed-replace`` so an ``<img>`` shows them as
    video. A frame is sent only when the camera has a new one; while a pick grabs (``measuring``) or the camera gives
    none, the browser keeps the last picture. Read through ``Camera.peek`` as ``/live`` is: it never takes a frame from
    a pick. The stream ends when the browser closes it."""

    def frames() -> Iterator[bytes]:
        last_taken: float | None = None
        period = 1.0 / float(fps)
        while True:
            started = time.monotonic()
            try:
                quality = int(cell.config().runtime.image_encoding.frame_quality)
            except Exception:  # noqa: BLE001 (as /live: the default quality)
                quality = LIVE_QUALITY
            cell.live.quality = quality
            frame = cell.live.frame(cell.session.service, rig=rig, max_width=max_width)
            if frame.image_base64 and frame.captured_at != last_taken:
                last_taken = frame.captured_at
                jpeg = base64.b64decode(frame.image_base64)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode("ascii")
                       + b"\r\n\r\n" + jpeg + b"\r\n")
                time.sleep(max(0.0, period - (time.monotonic() - started)))
            else:
                time.sleep(max(period, _IDLE_S) if not frame.image_base64 else period)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})
