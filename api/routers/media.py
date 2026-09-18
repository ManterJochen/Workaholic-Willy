"""The viewfinder, the overlay stream, and turning speech into a prompt.

All three are optional in the strict sense: the console works without any of them, and none is on the
path of anything that moves.

The overlay is an honest still, not a video. The library renders one PNG per segmentation and
overwrites it, so what exists is the last segmentation the calculator processed, not necessarily the
target it grasped, unless a label narrowed the loop to one. Calling that a video feed would be a lie an
operator could act on ("the arm is where the picture shows"), so every frame carries its own age and the
stream says plainly what the picture is.

Speech arrives two ways and answers one way. A recording made in the browser is uploaded to
``/v1/voice/transcribe``; a push to talk turn is caught by the cell PC's own microphone through
``/v1/voice/listen``, while ``/v1/voice/talk`` presses and releases the talk switch. Both go through the
process's one voice gate and engine and answer with the same `Proposal`, and neither starts anything.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import threading
import time
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from api.audio import (
    AudioDecodeError,
    AudioFormatUnsupported,
    decode_audio,
    describe_upload,
)
from api.cell import Console, console
from api.constants import API_LOG_DIR, ROUTER_MEDIA_LOG_FILE
from api.schemas import ListenIn, ProposalOut, TalkIn, TalkOut, ViewfinderOut
from api.viewfinder import ViewfinderFrame, read_viewfinder
from src.utility.log_cfg import create_logger

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.models.models_schema import SpeechToTextConfig
    from src.models.speech.holder import HeldSpeech
    from src.models.speech.push_to_talk import PushToTalkSource, TalkRecording
    from src.models.speech.transcript import Proposal

router = APIRouter(tags=["media"])

#: ``GET /v1/camera`` is deliberately absent from this log: the browser polls it every 160 ms while a
#: camera frame is coming back (``Viewfinder.tsx``), so a line per request would be a frame counter.
#: What is logged is per-event: a socket opening and closing, the overlay being switched on, a
#: transcription or a push to talk turn with what it cost, and each edge of the talk switch.
logger = create_logger("MediaRouter", ROUTER_MEDIA_LOG_FILE, log_dir=API_LOG_DIR)

#: How often the overlay socket looks for a new render. The library produces roughly one image per
#: segmentation per attempt, so polling faster than this only re-sends bytes the client already has;
#: the hash check below turns those into a cheap "still the same" tick instead of a frame.
_OVERLAY_POLL_S = 0.25
#: A quiet stream still says something at this cadence, so a UI can distinguish "no new render" from
#: "the socket died".
_OVERLAY_HEARTBEAT_S = 5.0
#: Held while a push to talk listen runs. The cell PC has one microphone and the process one talk switch,
#: so a second listen would catch the same turn twice; it is refused instead.
_LISTENING = threading.Lock()


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


@router.post("/voice/transcribe", response_model=ProposalOut, summary="Speech to a prompt")
async def post_transcribe(
    cell: Annotated[Console, Depends(console)],
    audio: Annotated[UploadFile, File(description=describe_upload)],
) -> ProposalOut:
    """Turn a recording into a proposal for the prompt box. It does not start anything.

    Deliberately not a shortcut to a pick. A spoken command that went straight to motion would mean a
    misheard word moves an arm, so speech lands in the prompt box and a human presses the button: the
    same button, with the same acknowledgement, as a typed prompt. "Stopp" is no exception.

    The answer is the speech library's `Proposal`: the words, or an empty text and the sentence saying
    why there are none; what the voice detector found; and Whisper's `Transcript` when Whisper was
    asked. A recording in which the detector hears no speech is never handed to Whisper, which answers
    silence with a word. The text stays in the language it was spoken in. The process's one engine and
    gate live in `src.models.speech.holder` and load on the first recording, never at import and never
    per request: a console on a machine without the speech stack still starts, and answers 501 naming
    what is missing.
    """
    raw = await audio.read()
    if not raw:
        raise HTTPException(
            status_code=400,
            detail={"code": "empty_audio", "message": "no audio was uploaded.", "detail": {}},
        )
    started = time.perf_counter()
    try:
        proposal = await asyncio.to_thread(_transcribe, cell, raw)
    except Exception as exc:  # noqa: BLE001 (every failure becomes a typed answer, never a crash)
        raise _refusal(exc, size=len(raw), declared=audio.content_type, started=started) from exc
    # The text as well as the timing: it becomes the prompt of a run that will move an arm, and
    # "the operator asked for the wrong thing" and "Whisper heard the wrong thing" are different
    # faults that look identical in a run log alone. The language goes with it, because a command
    # decoded as the wrong language reads as nonsense rather than as a mishearing.
    transcript = proposal.transcript
    logger.info(
        "Proposed %r from %d byte(s) in %.1f s (speech %s, peak %.2f; %s).",
        proposal.text, len(raw), time.perf_counter() - started,
        "heard" if proposal.speech.heard_speech else "not heard", proposal.speech.peak_probability,
        "Whisper not asked" if transcript is None else (
            f"{transcript.language}, {transcript.language_source.value}, decode "
            f"{transcript.latency_ms:.0f} ms on {transcript.device}"
        ),
    )
    return ProposalOut(**proposal.to_dict())


def _refusal(exc: Exception, *, size: int, declared: str | None, started: float) -> HTTPException:
    """The typed answer to a recording that produced no proposal.

    The speech library's refusals are imported here, on the error path, so they are the classes of the
    modules that raised them, even in a process that imported the speech package afresh. Each answer
    says whether the request was wrong (415, 422) or this host lacks something (501).
    """
    from src.models.speech.engine import (
        RecordingTooLong,
        SpeechModelMissing,
        SpeechStackUnavailable,
        requirements_for,
    )

    def answer(status: int, code: str, message: str) -> HTTPException:
        return HTTPException(status_code=status, detail={"code": code, "message": message, "detail": {}})

    if isinstance(exc, AudioFormatUnsupported):
        # 415: a format this server never decodes. The console records WAV, and the sentence says so,
        # names what arrived and what the upload claimed to be.
        logger.warning("A recording arrived in a format the console does not decode: %s", exc)
        return answer(415, "audio_format_unsupported", f"{exc} It was declared as {declared or 'no type'}.")
    if isinstance(exc, AudioDecodeError):
        logger.warning("A recording could not be decoded: %s", exc)
        return answer(422, "audio_undecodable", str(exc))
    if isinstance(exc, RecordingTooLong):
        logger.warning("A recording of %.1f s was refused as longer than Whisper's window.", exc.duration_s)
        return answer(422, "audio_too_long", str(exc))
    if isinstance(exc, SpeechModelMissing):
        logger.warning("A speech model is not on this machine: %s", exc)
        return answer(501, "speech_model_missing", f"{exc} Type the prompt instead.")
    if isinstance(exc, ImportError):
        # A package this machine cannot import: a capability this host lacks (501), never a bad request.
        # `SpeechStackUnavailable` already says what to do, including when Windows refused a DLL.
        if isinstance(exc, SpeechStackUnavailable):
            hint = str(exc)
        else:
            hint = (
                f"speech-to-text is not installed on this machine ({exc}). It is installed by "
                f"{requirements_for(exc.name)}."
            )
        logger.warning("Speech-to-text cannot run on this machine (%s); the prompt must be typed.", exc)
        return answer(501, "speech_unavailable", f"{hint} Type the prompt instead.")
    logger.error(
        "Transcription of %d byte(s) failed after %.1f s: %s: %s",
        size, time.perf_counter() - started, type(exc).__name__, exc,
    )
    return answer(422, "transcription_failed", f"{type(exc).__name__}: {exc}")


def _transcribe(cell: Console, raw: bytes) -> Proposal:
    """Decode the upload and hand it to the process's speech holder. Blocking; the caller runs it off the loop."""
    samples, rate = decode_audio(raw)
    return _held_speech(cell).propose(samples, samplerate=rate)


def _held_speech(cell: Console) -> HeldSpeech:
    """The process's voice gate and engine for the console's `models.stt`.

    The config is `models.stt` alone, read from the console's own tree under its own profile chain:
    speech needs no camera, robot or detector, so a fault in one of those must not refuse a recording.
    Both speech routes ask here, so an upload and a push to talk turn go through one gate and one engine.
    """
    from src.models.speech.holder import shared_speech

    return shared_speech().for_tree(cell.root, profile=cell.profile)


@router.post("/voice/talk", response_model=TalkOut, summary="Press or release the talk switch")
def post_talk(body: TalkIn) -> TalkOut:
    """Press or release the process's talk switch, the event a push to talk listen waits for.

    A client sends ``pressed: true`` on the way down and ``pressed: false`` on the way up, and a switch
    reader at the cell PC presses the same switch (`shared_talk_button()`). A press while the switch is
    already down is not counted, so a key that repeats while it is held is one press. It moves nothing
    and starts nothing. No console screen sends it yet: the console's talk button records in the browser
    and uploads to ``/v1/voice/transcribe``.
    """
    from src.models.speech.push_to_talk import shared_talk_button

    button = shared_talk_button()
    if body.pressed:
        button.press()
    else:
        button.release()
    logger.info("Talk switch %s, %d press(es) so far.", "down" if button.pressed else "up", button.presses)
    return TalkOut(pressed=button.pressed, presses=button.presses)


@router.post("/voice/listen", response_model=ProposalOut, summary="Push to talk at the cell PC to a prompt")
async def post_listen(
    cell: Annotated[Console, Depends(console)], body: ListenIn | None = None
) -> ProposalOut:
    """Listen at the cell PC's own microphone for one push to talk turn, and propose what was said.

    The turn is what the microphone catches while the talk switch is held (``/v1/voice/talk``). The route
    opens the microphone, waits up to ``timeout_s`` for the press, drops the audio from before it, and at
    the release hands everything held to the same voice gate and engine as an upload. The answer is the
    same `Proposal`, and like an upload it starts nothing: the text lands in the prompt box and a person
    confirms it, "Stopp" included.

    A turn that proposes nothing is an answer, not a crash: 409 when the switch was not pressed in time or
    the microphone ended, 422 when the switch came up before any audio or stayed down past Whisper's 30 s
    window, each with the library's sentence and `TalkRecording.to_dict()`. A microphone this host cannot
    open is 501, like a speech stack it cannot import. One turn at a time: a second listen is 409.
    """
    timeout_s = (body or ListenIn()).timeout_s
    if not _LISTENING.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "listen_busy",
                "message": "a push to talk listen is already running on this cell PC; it serves one turn "
                           "at a time.",
                "detail": {},
            },
        )
    started = time.perf_counter()
    try:
        recording, proposal = await asyncio.to_thread(_listen, cell, timeout_s)
    except Exception as exc:  # noqa: BLE001 (every failure becomes a typed answer, never a crash)
        raise _listen_refusal(exc, started=started) from exc
    finally:
        _LISTENING.release()
    if proposal is None:
        raise _turn_refusal(recording)
    logger.info(
        "Proposed %r from a %.2f s push to talk turn, %.1f s after the listen began (speech %s, peak %.2f).",
        proposal.text, recording.duration_s, time.perf_counter() - started,
        "heard" if proposal.speech.heard_speech else "not heard", proposal.speech.peak_probability,
    )
    return ProposalOut(**proposal.to_dict())


def _listen(cell: Console, timeout_s: float) -> tuple[TalkRecording, Proposal | None]:
    """One push to talk turn at the cell PC's microphone, proposed like an upload when the switch came up
    with audio. Blocking; the caller runs it off the loop. The microphone is closed after the turn."""
    held = _held_speech(cell)
    source = _talk_source(held.config)
    with source:
        recording = source.record(timeout_s=timeout_s)
    if not recording.ok:
        return recording, None
    return recording, held.propose(recording.samples, samplerate=recording.samplerate)


def _talk_source(config: SpeechToTextConfig) -> PushToTalkSource:
    """The cell PC's microphone from `models.stt`, behind the process's talk switch. Opens nothing. A
    function of its own, so the route can be driven by a source that needs no microphone."""
    from src.models.speech.push_to_talk import PushToTalkSource

    return PushToTalkSource.from_config(config=config)


def _turn_refusal(recording: TalkRecording) -> HTTPException:
    """The typed answer to a turn that proposes nothing: the switch or the microphone (409), or the turn
    itself (422). The message is the library's own sentence, and the detail its `to_dict()`."""
    from src.models.speech.push_to_talk import TalkOutcome

    answers: dict[TalkOutcome, tuple[int, str]] = {
        TalkOutcome.NOT_PRESSED: (409, "talk_not_pressed"),
        TalkOutcome.SOURCE_ENDED: (409, "microphone_ended"),
        TalkOutcome.NOTHING_CAPTURED: (422, "nothing_recorded"),
        TalkOutcome.HELD_TOO_LONG: (422, "audio_too_long"),
    }
    status, code = answers[recording.outcome]
    message = recording.render()
    logger.warning("A push to talk listen proposed nothing: %s", message.replace("\n", " "))
    return HTTPException(
        status_code=status, detail={"code": code, "message": message, "detail": recording.to_dict()}
    )


def _listen_refusal(exc: Exception, *, started: float) -> HTTPException:
    """The typed answer to a push to talk listen that failed.

    A microphone this host cannot open is a capability it lacks (501), like a speech stack it cannot
    import. The refusals an upload shares (a package or a model missing, a recording too long) get the
    upload's answers from `_refusal`.
    """
    from src.models.speech.capture import MicrophoneUnavailable
    from src.models.speech.engine import RecordingTooLong, SpeechModelMissing

    if isinstance(exc, MicrophoneUnavailable):
        logger.warning("The cell PC's microphone did not open: %s", exc)
        return HTTPException(
            status_code=501,
            detail={
                "code": "microphone_unavailable",
                "message": f"{exc} Type the prompt instead.",
                "detail": {},
            },
        )
    if isinstance(exc, (ImportError, SpeechModelMissing, RecordingTooLong)):
        return _refusal(exc, size=0, declared=None, started=started)
    logger.error(
        "A push to talk listen failed after %.1f s: %s: %s",
        time.perf_counter() - started, type(exc).__name__, exc,
    )
    return HTTPException(
        status_code=422,
        detail={"code": "listen_failed", "message": f"{type(exc).__name__}: {exc}", "detail": {}},
    )
