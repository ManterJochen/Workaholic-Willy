"""What the robot is looking at, as one JPEG and one honest sentence about what the picture is.

Three pictures, never blurred into one word. A console that shows "the camera" can be showing any of
three very different things, and an operator who cannot tell them apart is being misled:

    source="camera"    a colour frame taken from the device just now. There is no arm in it, no
                       segmentation, no decision: it is the scene, and it is current.
    source="synthetic" a scene this process drew for a rehearsal cell, which has no camera at all.
    source="overlay"   the grasp-point render the stack produced during a pick: the segmentation mask,
                       the projected gripper, the chosen contact points. It is what the robot saw and
                       decided, and it is as old as the last segmentation, which may be minutes.

So every answer carries its `source`, its `age_s`, and a sentence saying which of the three it is. The
UI is not trusted to remember; the payload says it.

Why the pick wins the camera. The camera package holds no lock (measured: ``grep -ri thread
src/camera/`` is empty), and a pick runs on its own daemon thread. Two threads calling ``grab()`` on
one ``rs.pipeline`` split the frame stream between them. So while a run owns the cell, this module
does not touch the camera at all; it falls back to the overlay, which is both safe (it is a bytes
attribute) and better, because during a pick what the robot decided is more informative than the raw
scene. The exclusion is by run state rather than by mutex because a mutex would have to live inside
the camera package, on the pick's own hot path, to be correct.

There is one narrow race left and it is named rather than hidden: a run can start between the state
check and the grab. Its cost is that one frameset lands in the viewer instead of the pick, and the
pick's next ``acquire()`` opens by discarding ``warmup_grabs`` frames, five by default, so nothing it
relies on changes.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Any

from api.constants import API_LOG_DIR, VIEWFINDER_LOG_FILE
from src.utility.log_cfg import create_logger

__all__ = ["ViewfinderFrame", "read_viewfinder"]

#: One line per new overlay, and nothing per request. ``read_viewfinder`` answers a poll that runs at
#: 160 ms while a camera frame is coming back, so anything logged on the request path, including the
#: no-picture states, which are steady states rather than events, would be six lines a second saying
#: the same thing. The overlay digest below is already change-detected, so it is the one place in this
#: module where something actually happens.
logger = create_logger("Viewfinder", VIEWFINDER_LOG_FILE, log_dir=API_LOG_DIR)

#: Serialises browser tabs against each other. Two consoles open on one cell would otherwise both call
#: ``grab()``; this makes them queue instead of interleave. It does not protect against the pick; see
#: the module docstring for why that exclusion is by run state.
_PEEK_LOCK = threading.Lock()

#: ``(digest, first-seen monotonic, exact)`` for the overlay.
#:
#: The age of an overlay is only ever a lower bound, and the payload says which kind. The renderer
#: stamps no timestamp on the PNG it writes, so nothing anywhere records when an overlay was made:
#: all this process can measure is when it first saw those bytes. Two cases follow, and conflating
#: them is the mistake ``/v1/overlay`` makes, since it seeds its clock at socket-accept and reports
#: ``age_s: 0.0`` for a frame rendered minutes earlier:
#:
#:   * the first overlay this process ever sees may have been rendered before the server started, so
#:     the age is a floor and ``age_is_exact`` is False;
#:   * on every later change this process watched the digest flip, so the age is real to within the
#:     poll interval, and ``age_is_exact`` is True.
#:
#: Reporting the first case as exact is what puts a stale picture on a demo screen labelled "now".
_OVERLAY_SEEN: tuple[str, float, bool] = ("", 0.0, False)
_OVERLAY_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class ViewfinderFrame:
    """One answer from the viewfinder. ``image is None`` is an answer, not a failure."""

    #: ``"camera"`` | ``"synthetic"`` | ``"overlay"`` | ``"none"``.
    source: str
    #: A machine code for the "no picture" cases: ``not_built`` | ``pick_owns_camera`` |
    #: ``no_colour_source`` | ``encode_failed``. Empty when there is a picture.
    reason: str
    #: A sentence for a person. Always populated, including when there is a picture.
    human: str
    #: The encoded picture, or ``None``. JPEG for a camera frame, PNG for an overlay: the overlay is
    #: already encoded by the renderer, and re-encoding a lossless render as lossy JPEG to make one
    #: field name true would be a worse trade than carrying the type.
    image: bytes | None = None
    #: ``"image/jpeg"`` | ``"image/png"`` | ``""``.
    media_type: str = ""
    width: int = 0
    height: int = 0
    #: Seconds since the picture was produced. ``0.0`` for a picture taken now; for an overlay, seconds
    #: since this process first saw those bytes; ``None`` when there is no picture.
    age_s: float | None = None
    #: Whether ``age_s`` is the real age or only a floor. False for the first overlay this process
    #: sees, which may predate the server, and True once it has watched the image change. Nothing
    #: records when an overlay was rendered, so this is the most that can honestly be said.
    age_is_exact: bool = True
    #: Whether overlay rendering is currently switched on, so a UI can offer the switch rather than
    #: leaving an operator to guess why a pick shows nothing.
    overlay_enabled: bool = False


def read_viewfinder(cell: Any, *, jpeg_quality: int = 60) -> ViewfinderFrame:
    """Resolve the best honest picture of what this cell is looking at, right now.

    ``cell`` is the :class:`api.cell.Console`. Typed loosely so a caller holding a stand-in session
    can use this module without building a whole console.
    """
    service = getattr(getattr(cell, "session", None), "service", None)
    if service is None:
        return ViewfinderFrame(
            source="none",
            reason="not_built",
            human="No cell is assembled, so there is nothing looking at anything yet.",
        )

    overlay_on = bool(getattr(service, "debug_image_rendering_enabled", False))

    # --- a run owns the camera ---------------------------------------------------------------------
    if getattr(cell, "active_run_id", None) is not None:
        overlay = _overlay_frame(service, overlay_on)
        if overlay is not None:
            return overlay
        return ViewfinderFrame(
            source="none",
            reason="pick_owns_camera",
            overlay_enabled=overlay_on,
            human=(
                "A pick is running and it owns the camera, so the live view stands down rather than "
                "take frames out of the stream the robot is grasping from. "
                + (
                    "The grasp overlay is on but nothing has been rendered yet: the first one "
                    "appears when the stack segments a scene."
                    if overlay_on
                    else "Switch the grasp overlay on to watch what the robot decides while it works."
                )
            ),
        )

    # --- idle: the camera is free ------------------------------------------------------------------
    colour, kind = _peek(service)
    if colour is not None:
        jpeg = _encode(colour, jpeg_quality)
        if jpeg is None:
            return ViewfinderFrame(
                source="none",
                reason="encode_failed",
                overlay_enabled=overlay_on,
                human="A colour frame arrived but could not be encoded as JPEG.",
            )
        height, width = int(colour.shape[0]), int(colour.shape[1])
        return ViewfinderFrame(
            source="camera" if kind == "camera" else "synthetic",
            reason="",
            human=_LIVE_SAYS.get(kind, _LIVE_SAYS["unknown"]),
            image=jpeg,
            media_type="image/jpeg",
            width=width,
            height=height,
            age_s=0.0,
            overlay_enabled=overlay_on,
        )

    # --- no live colour: an old overlay is still more use than nothing, if it says it is old --------
    overlay = _overlay_frame(service, overlay_on)
    if overlay is not None:
        return overlay

    return ViewfinderFrame(
        source="none",
        reason="no_colour_source",
        overlay_enabled=overlay_on,
        human=(
            "This cell's perception source cannot hand over a picture without doing perception, so "
            "there is no live view. A simulated source is the usual reason: reading it would step the "
            "simulator this console is only supposed to be watching."
        ),
    )


#: What to say about a live picture, per what the source declared it is. The rehearsal wording is the
#: point of the whole distinction: it must be impossible to read that line and think a camera exists.
_LIVE_SAYS = {
    "camera": "Live from the cell's camera. No segmentation, no decision: just the scene.",
    "synthetic": (
        "This is not a camera. The cell was assembled in rehearsal, so its perception source draws a "
        "synthetic scene (one box on a plane) and that is what you are looking at."
    ),
    "unknown": (
        "A colour image from this cell's perception source. The source does not say whether it came "
        "from a device, so this console will not call it a camera."
    ),
}


def _peek(service: Any) -> tuple[Any, str]:
    """``(colour image | None, what kind of picture it is)`` from the built perception source."""
    from src.robot.perception.viewfinder import colour_source_kind, peek_color_of

    perception = getattr(getattr(service, "runtime", None), "orchestrator", None)
    perception = getattr(perception, "perception", None)
    if perception is None:
        return None, "unknown"
    with _PEEK_LOCK:
        return peek_color_of(perception), colour_source_kind(perception)


def _overlay_frame(service: Any, overlay_on: bool) -> ViewfinderFrame | None:
    """The last grasp-point render, with a truthful age, or ``None`` if there is none."""
    png = getattr(service, "last_debug_image_png", None)
    if not png:
        return None

    global _OVERLAY_SEEN
    digest = hashlib.sha256(png).hexdigest()[:16]
    now = time.monotonic()
    with _OVERLAY_LOCK:
        changed = _OVERLAY_SEEN[0] != digest
        if changed:
            # Exact only if this process watched the previous image turn into this one. The very
            # first overlay it sees may have been rendered before it started, and there is no
            # timestamp anywhere to appeal to.
            _OVERLAY_SEEN = (digest, now, _OVERLAY_SEEN[0] != "")
        _, first_seen, exact = _OVERLAY_SEEN

    age = round(now - first_seen, 1)
    width, height = _png_size(png)
    if changed:
        # `age_is_exact` is on the line because it is the difference between "the stack decided this
        # just now" and "this picture is at least that old, possibly much older", and a demo screen
        # renders both the same way.
        logger.info(
            "New grasp overlay: %dx%d, %d bytes, digest %s (age exact: %s).",
            width, height, len(png), digest, exact,
        )
    return ViewfinderFrame(
        source="overlay",
        reason="",
        human=(
            "The grasp overlay: the segmentation and the grasp the stack chose, drawn on the frame it "
            "chose them from. This is what the robot decided, not what the camera shows now."
            + (
                ""
                if exact
                else " Nothing records when an overlay was rendered, and this is the first one this "
                "server has seen, so it is at least this old, possibly much older."
            )
        ),
        image=png,
        media_type="image/png",
        width=width,
        height=height,
        age_s=age,
        age_is_exact=exact,
        overlay_enabled=overlay_on,
    )


def _encode(bgr: Any, quality: int) -> bytes | None:
    """JPEG bytes for a BGR array at the configured quality, or ``None`` if OpenCV refuses."""
    import cv2

    ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return bytes(buffer.tobytes()) if ok else None


def _png_size(png: bytes) -> tuple[int, int]:
    """Width/height straight out of the PNG IHDR, so the overlay does not have to be decoded."""
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n":
        return 0, 0
    return int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
