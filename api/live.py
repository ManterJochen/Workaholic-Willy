"""The live image: a display frame of each camera the cell's picks see through, never a measurement.

``GET /v1/camera/live`` (build plan 1.7, OD 11) reads through ``Camera.peek`` of the service's camera owners
(``service_cameras``): the primary rig first, then the fused ones, then those the planner world opened for itself.
``peek`` never waits and never takes a frame a measurement relies on: while a measuring grab holds the rig, and for a
moment after it, it answers nothing, and the answer says ``measuring`` so the browser keeps its last frame. So the live
image runs during a pick too, which ``GET /v1/camera`` does not (it stands down and shows the overlay instead, and it
stays as it is).

A frame is downscaled to the width asked for, its aspect kept, and JPEG-encoded at the runtime's frame quality
(``runtime.image_encoding.frame_quality``, which the route hands over as :attr:`LiveFrames.quality`); it carries when it
was taken and how old it is. The rehearsal cell has no camera: its source draws its own picture, which is shown as
``synthetic``, never as a camera. Every "no picture" is an answer, not an error: ``not_built``, ``no_camera`` (the cell
has none, or the rig gives no frame), ``no_rig`` (a rig the cell does not hold), ``measuring``, ``encode_failed``.

:class:`LiveFrames` also keeps each rig's last frame time, for the ready bar's camera light: :meth:`LiveFrames.ages`
says how old each is, and :meth:`LiveFrames.refresh_stale` peeks a rig whose frame is older than the bound, so a rig
nobody shows on the stage never reads as stale.
"""

from __future__ import annotations

import base64
import threading
import time
from typing import Any, Final, Literal

from api.schemas import LiveFrameOut, RigOut

__all__ = ["LIVE_MAX_WIDTH", "LIVE_QUALITY", "STALE_AFTER_S", "LiveFrames"]

#: The default width a live frame is downscaled to, in pixels.
LIVE_MAX_WIDTH = 960
#: A rig's frame older than this reads as stale to the ready bar's camera light.
STALE_AFTER_S = 2.0
#: The JPEG quality a frame is written at until the route hands over the runtime's (``frame_quality``).
LIVE_QUALITY: Final[int] = 60


class LiveFrames:
    """Each rig's last display frame time, and the frames themselves on request. One per console (``Console.live``).

    Thread-safe: the route answers on the server's thread pool, and the ready bar reads the ages from another request.
    It holds no camera and opens nothing; every frame is a ``peek`` through the service's own owners.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        #: When each rig's last frame was taken, Unix seconds (``PeekFrame.captured_at_s``).
        self._taken: dict[str, float] = {}
        #: The JPEG quality, 1-100; the route sets it from the runtime config on every request.
        self.quality: int = LIVE_QUALITY

    def frame(self, service: Any, *, rig: str | None = None, max_width: int = LIVE_MAX_WIDTH) -> LiveFrameOut:
        """The newest display frame of ``rig`` (the primary rig where ``None``), at most ``max_width`` pixels wide.

        Never raises for a camera: a rig that gives no frame is ``no_camera``, a peek that answers nothing is
        ``measuring`` (with the last frame's time, which the browser keeps showing), a frame that will not encode is
        ``encode_failed``.
        """
        if service is None:
            return LiveFrameOut(rig_id=rig, source="none", reason="not_built")
        owners = _owners_of(service)
        rigs = [RigOut(rig_id=str(owner.rig_id), mounting=_mounting_of(owner), primary=index == 0)
                for index, owner in enumerate(owners)]
        if not owners:
            return _synthetic(service, max_width=max_width, quality=self.quality) or LiveFrameOut(
                rig_id=rig, source="none", reason="no_camera")
        chosen = owners[0] if rig is None else next((o for o in owners if str(o.rig_id) == rig), None)
        if chosen is None:
            return LiveFrameOut(rig_id=rig, rigs=rigs, source="none", reason="no_rig")
        rig_id = str(chosen.rig_id)
        try:
            peeked = chosen.peek()
        except Exception:  # noqa: BLE001 (a rig that is not open, or that refuses a look, shows no picture)
            return LiveFrameOut(rig_id=rig_id, rigs=rigs, source="none", reason="no_camera")
        if peeked is None:
            taken = self._last(rig_id)
            return LiveFrameOut(rig_id=rig_id, rigs=rigs, source="camera", reason="measuring", captured_at=taken,
                                age_s=None if taken is None else max(0.0, time.time() - taken))
        taken = float(getattr(peeked, "captured_at_s", time.time()))
        self._keep(rig_id, taken)
        encoded = _encode(getattr(peeked, "color", None), max_width=max_width, quality=self.quality)
        if encoded is None:
            return LiveFrameOut(rig_id=rig_id, rigs=rigs, source="none", reason="encode_failed")
        jpeg, width, height = encoded
        return LiveFrameOut(
            rig_id=rig_id, rigs=rigs, source="camera", reason="", image_base64=base64.b64encode(jpeg).decode("ascii"),
            width=width, height=height, captured_at=taken, age_s=max(0.0, time.time() - taken),
        )

    def ages(self) -> dict[str, float]:
        """Seconds since each rig's last frame, by rig id; a rig never seen has none."""
        now = time.time()
        with self._lock:
            return {rig_id: max(0.0, now - taken) for rig_id, taken in self._taken.items()}

    def refresh_stale(self, service: Any, *, max_age_s: float = STALE_AFTER_S) -> dict[str, float]:
        """Peek every rig of ``service`` whose frame is older than ``max_age_s``, or has none, and answer :meth:`ages`.

        A peek that answers nothing (the rig measures) leaves the rig's age as it was; one that raises too. Nothing is
        encoded: only the time is kept.
        """
        if service is None:
            return self.ages()
        ages = self.ages()
        for owner in _owners_of(service):
            rig_id = str(owner.rig_id)
            age = ages.get(rig_id)
            if age is not None and age <= max_age_s:
                continue
            try:
                peeked = owner.peek()
            except Exception:  # noqa: BLE001 (a rig that gives no frame keeps no new age)
                continue
            if peeked is not None:
                self._keep(rig_id, float(getattr(peeked, "captured_at_s", time.time())))
        return self.ages()

    def _keep(self, rig_id: str, taken: float) -> None:
        with self._lock:
            if taken >= self._taken.get(rig_id, float("-inf")):
                self._taken[rig_id] = taken

    def _last(self, rig_id: str) -> float | None:
        with self._lock:
            return self._taken.get(rig_id)


def _owners_of(service: Any) -> tuple[Any, ...]:
    """The camera owners the built service holds (``service_cameras``), the primary first; none on a rehearsal cell."""
    from src.robot.execution.lifecycle import service_cameras  # noqa: PLC0415 (the console imports this to start)

    try:
        return service_cameras(service)
    except Exception:  # noqa: BLE001 (a service whose cameras cannot be named shows none)
        return ()


def _mounting_of(owner: Any) -> Literal["wrist", "fixed", "unknown"]:
    """Where a rig rides, read off its configuration: a declared body or an eye-in-hand calibration is the wrist, an
    eye-to-hand calibration is fixed, and a rig that declares neither says so."""
    rig = getattr(owner, "rig", None)
    mode = getattr(getattr(rig, "extrinsics", None), "mounting_mode", None)
    if getattr(rig, "body", None) is not None or mode == "eye_in_hand":
        return "wrist"
    if mode == "eye_to_hand":
        return "fixed"
    return "unknown"


def _synthetic(service: Any, *, max_width: int, quality: int) -> LiveFrameOut | None:
    """The rehearsal source's own picture, labelled ``synthetic``; ``None`` for a source that draws none.

    Only a source that declares it drew its pixels is shown here: a source of another kind that can hand a picture
    without a camera owner is no camera this console may call live.
    """
    from src.robot.perception.viewfinder import colour_source_kind, peek_color_of  # noqa: PLC0415

    perception = getattr(getattr(getattr(service, "runtime", None), "orchestrator", None), "perception", None)
    if perception is None or colour_source_kind(perception) != "synthetic":
        return None
    taken = time.time()
    try:
        colour = peek_color_of(perception)
    except Exception:  # noqa: BLE001 (a drawing that failed is no picture)
        colour = None
    if colour is None:
        return None
    encoded = _encode(colour, max_width=max_width, quality=quality)
    if encoded is None:
        return LiveFrameOut(source="none", reason="encode_failed")
    jpeg, width, height = encoded
    return LiveFrameOut(source="synthetic", reason="", image_base64=base64.b64encode(jpeg).decode("ascii"),
                        width=width, height=height, captured_at=taken, age_s=0.0)


def _encode(colour: Any, *, max_width: int, quality: int) -> tuple[bytes, int, int] | None:
    """JPEG bytes and the size of ``colour`` (BGR), downscaled to at most ``max_width`` with its aspect kept; ``None``
    where it is no image or OpenCV refuses it."""
    import cv2  # noqa: PLC0415 (OpenCV loads on the first frame, not at the console's start)
    import numpy as np  # noqa: PLC0415

    try:
        image = np.asarray(colour)
        if image.ndim < 2 or image.shape[0] == 0 or image.shape[1] == 0:
            return None
        height, width = int(image.shape[0]), int(image.shape[1])
        bound = max(1, int(max_width))
        if width > bound:
            height = max(1, round(height * bound / width))
            width = bound
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    except Exception:  # noqa: BLE001 (a frame that will not encode is said, never raised)
        return None
    if not ok or buffer is None:
        return None
    return bytes(buffer.tobytes()), width, height
