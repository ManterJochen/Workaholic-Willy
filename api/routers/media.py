"""The viewfinder, the overlay stream, and turning speech into a prompt.

All three are optional in the strict sense: the console works without any of them, and none is on the
path of anything that moves.

The overlay is an honest still, not a video. The library renders one PNG per segmentation and
overwrites it, so what exists is the last segmentation the calculator processed, not necessarily the
target it grasped, unless a label narrowed the loop to one. Calling that a video feed would be a lie an
operator could act on ("the arm is where the picture shows"), so every frame carries its own age and the
stream says plainly what the picture is.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from api.audio import (
    AudioDecodeError,
    AudioFormatUnsupported,
    decode_audio,
    describe_upload,
)
from api.cell import Console, console
from api.constants import API_LOG_DIR, ROUTER_MEDIA_LOG_FILE
from api.schemas import TranscriptOut, ViewfinderOut
from api.viewfinder import ViewfinderFrame, read_viewfinder
from src.utility.log_cfg import create_logger

router = APIRouter(tags=["media"])

#: ``GET /v1/camera`` is deliberately absent from this log: the browser polls it every 160 ms while a
#: camera frame is coming back (``Viewfinder.tsx``), so a line per request would be a frame counter.
#: What is logged is per-event: a socket opening and closing, the overlay being switched on, and a
#: transcription with what it cost.
logger = create_logger("MediaRouter", ROUTER_MEDIA_LOG_FILE, log_dir=API_LOG_DIR)

#: How often the overlay socket looks for a new render. The library produces roughly one image per
#: segmentation per attempt, so polling faster than this only re-sends bytes the client already has;
#: the hash check below turns those into a cheap "still the same" tick instead of a frame.
_OVERLAY_POLL_S = 0.25
#: A quiet stream still says something at this cadence, so a UI can distinguish "no new render" from
#: "the socket died".
_OVERLAY_HEARTBEAT_S = 5.0


@router.get("/camera", response_model=ViewfinderOut, summary="What the cell is looking at")
def get_camera(cell: Annotated[Console, Depends(console)]) -> ViewfinderOut:
    """One picture and one sentence saying what the picture is.

    Polled rather than streamed, on purpose. A socket would have to decide what "no picture" looks like
    between frames, and every one of the four no-picture states here is a state an operator needs to
    read, not to infer from a stalled image. A poll makes each tick a complete, self-describing answer:
    a cell that has not been built, a pick that owns the camera, a simulated source that cannot be
    peeked at, or a frame.

    Declared ``def``, not ``async def``, and that is load-bearing. A real ``grab()`` blocks on
    ``wait_for_frames``; FastAPI runs a synchronous route in its threadpool, so the block lands there
    instead of on the event loop. An ``async def`` here would stall the run event stream every tick,
    the one thing on this server that must not pause while a robot is moving.

    The JPEG quality comes from ``runtime.image_encoding.frame_quality``.
    """
    quality = int(cell.config().runtime.image_encoding.frame_quality)
    return _render_viewfinder(read_viewfinder(cell, jpeg_quality=quality))


def _render_viewfinder(frame: ViewfinderFrame) -> ViewfinderOut:
    """``ViewfinderFrame`` as the wire shape. Base64 only when there is something to encode."""
    return ViewfinderOut(
        source=frame.source,
        reason=frame.reason,
        human=frame.human,
        image_base64=None if frame.image is None else base64.b64encode(frame.image).decode("ascii"),
        media_type=frame.media_type,
        width=frame.width,
        height=frame.height,
        age_s=frame.age_s,
        age_is_exact=frame.age_is_exact,
        overlay_enabled=frame.overlay_enabled,
    )


@router.post("/overlay/enable", summary="Start rendering the debug overlay")
def post_overlay_enable(
    cell: Annotated[Console, Depends(console)], enabled: bool = True
) -> dict[str, bool]:
    """Opt into the overlay render. Off by default, in the library, and it costs real time per pick.

    The calculator only forwards the camera's RGB into the renderer when this is on, which is what keeps
    the default pick path free of it. Turning it on is therefore a deliberate trade: a picture per
    segmentation, paid for out of the pick's own latency budget.
    """
    if cell.session.service is None:
        raise HTTPException(
            status_code=409,
            detail={"code": "not_built", "message": "build the cell first.", "detail": {}},
        )
    cell.session.service.enable_debug_image_rendering(enabled)
    # A state change that costs pick latency for as long as it is on, so the log carries it: an
    # overlay switched on for a demo is otherwise invisible in every later timing report.
    logger.info("Debug overlay rendering %s.", "ENABLED" if enabled else "disabled")
    return {"enabled": enabled}


@router.websocket("/overlay")
async def overlay_socket(socket: WebSocket) -> None:
    """Send each new overlay render, with its age, and say what the picture actually is.

    A frame is sent only when the bytes change: the library overwrites one buffer rather than producing
    a stream, so re-sending an unchanged image at poll rate would fake a video that does not exist.
    Between changes the socket sends a tick carrying the current frame's age, which is the honest signal
    a UI needs: an overlay that is 40 seconds old should not be rendered as a live view.
    """
    from api.cell import console as _console

    cell = _console()
    await socket.accept()
    await socket.send_json({
        "type": "overlay_info",
        "human": (
            "This is the grasp-point overlay the stack rendered, not a camera feed. The library "
            "produces one image per segmentation and overwrites it, so between picks it holds the LAST "
            "segmentation processed; which is not necessarily the object that was grasped."
        ),
    })

    logger.info("Overlay socket opened.")
    last_digest = ""
    last_change = time.monotonic()
    last_tick = 0.0
    sent = 0
    try:
        while True:
            service = cell.session.service
            png = getattr(service, "last_debug_image_png", None) if service is not None else None
            now = time.monotonic()
            if png:
                digest = hashlib.sha256(png).hexdigest()[:16]
                if digest != last_digest:
                    last_digest, last_change = digest, now
                    sent += 1
                    await socket.send_json({
                        "type": "overlay_frame",
                        "digest": digest,
                        "age_s": 0.0,
                        "bytes": len(png),
                        "png_base64": base64.b64encode(png).decode("ascii"),
                    })
                    last_tick = now
                elif now - last_tick >= _OVERLAY_HEARTBEAT_S:
                    # Unchanged: say how old it is rather than re-sending it.
                    last_tick = now
                    await socket.send_json({
                        "type": "overlay_age", "digest": last_digest,
                        "age_s": round(now - last_change, 1),
                    })
            elif now - last_tick >= _OVERLAY_HEARTBEAT_S:
                last_tick = now
                await socket.send_json({
                    "type": "overlay_none",
                    "human": (
                        "No overlay yet. Either rendering is off (POST /v1/overlay/enable), or no pick "
                        "has run with a perception source that provides colour."
                    ),
                })
            await asyncio.sleep(_OVERLAY_POLL_S)
    except WebSocketDisconnect:
        # Counted rather than logged per frame: the socket ticks four times a second and sends only on
        # a new render, so the total is the number that means something afterwards.
        logger.info("Overlay socket closed after %d frame(s).", sent)
        return


@router.post("/voice/transcribe", response_model=TranscriptOut, summary="Speech to a prompt")
async def post_transcribe(
    cell: Annotated[Console, Depends(console)],
    audio: Annotated[UploadFile, File(description=describe_upload)],
) -> TranscriptOut:
    """Turn a recording into text. It returns the text; it does not start anything.

    Deliberately not a shortcut to a pick. A spoken command that went straight to motion would mean a
    misheard word moves an arm, so speech lands in the prompt box and a human presses the button: the
    same button, with the same acknowledgement, as a typed prompt.

    Whisper is loaded on first use, not at import: a console on a machine without it must still start,
    and must say what is missing rather than fail to boot.
    """
    raw = await audio.read()
    if not raw:
        raise HTTPException(
            status_code=400,
            detail={"code": "empty_audio", "message": "no audio was uploaded.", "detail": {}},
        )
    started = time.perf_counter()
    try:
        text = await asyncio.to_thread(_transcribe, cell, raw)
    except AudioFormatUnsupported as exc:
        # 501 and not 422, and the difference is the whole point of the separate exception. The
        # operator did nothing wrong: their browser recorded webm because that is what browsers do,
        # and an "unprocessable" answer would send them hunting through microphone settings for
        # something that is a `pip install`. Same shape as the missing-Whisper answer below.
        logger.warning("A recording arrived in a format this host cannot decode: %s", exc)
        raise HTTPException(
            status_code=501,
            detail={"code": "audio_format_unsupported", "message": str(exc), "detail": {}},
        ) from exc
    except AudioDecodeError as exc:
        logger.warning("A recording could not be decoded: %s", exc)
        raise HTTPException(
            status_code=422,
            detail={"code": "audio_undecodable", "message": str(exc), "detail": {}},
        ) from exc
    except ImportError as exc:
        logger.warning(
            "Speech-to-text is not installed on this machine (%s); the prompt must be typed.", exc
        )
        raise HTTPException(
            status_code=501,
            detail={
                "code": "speech_unavailable",
                "message": (
                    f"speech-to-text is not installed on this machine ({exc}). "
                    f"pip install -r requirements.txt, or type the prompt instead."
                ),
                "detail": {},
            },
        ) from exc
    except Exception as exc:  # noqa: BLE001 (a failed transcription is an answer, not a crash)
        logger.error(
            "Transcription of %d byte(s) failed after %.1f s: %s: %s",
            len(raw), time.perf_counter() - started, type(exc).__name__, exc,
        )
        raise HTTPException(
            status_code=422,
            detail={
                "code": "transcription_failed",
                "message": f"{type(exc).__name__}: {exc}",
                "detail": {},
            },
        ) from exc
    # The text as well as the timing: it becomes the prompt of a run that will move an arm, and
    # "the operator asked for the wrong thing" and "Whisper heard the wrong thing" are different
    # faults that look identical in a run log alone.
    logger.info(
        "Transcribed %d byte(s) of audio in %.1f s -> %r.",
        len(raw), time.perf_counter() - started, text.strip(),
    )
    return TranscriptOut(text=text.strip())


def _transcribe(cell: Console, raw: bytes) -> str:
    """Decode the upload and run it through Whisper. Blocking: the caller runs it off the loop."""
    from src.models.speech.speech_to_text import WhisperSpeechToText

    samples, rate = decode_audio(raw)
    config = cell.config().models.stt
    return WhisperSpeechToText(config).transcribe_array(samples, rate)
