"""One colour image, on demand, from whatever perception source a cell was built with.

An operator console wants to show what the robot is looking at, and ``acquire()`` is the wrong way
to get it twice over. ``acquire()`` on a real cell throws away five warm-up frames and then runs
GroundingDINO and SAM2 on the GPU, so a viewer at 8 Hz would run a detector 8 times a second beside
the pick that needs it. ``acquire()`` on the Isaac source steps the simulator (``vision.py`` warms
up 20 steps), so a browser tab left open would advance the physics clock of the cell it is watching.

So peeking is a separate, deliberately narrow capability:

    peek_color() -> HxWx3 BGR uint8, or None

and the contract on it is what makes it safe to call from a web request:

* No models. It must not ground, segment, or score anything.
* No simulation. It must not step, render-on-demand, or otherwise advance a simulator.
* No perception state. It must not change what the next ``acquire()`` would decide: not the prompt,
  not the intrinsics, not a cached frame, not a counter another component reads.

A source that cannot honour all three does not implement it, which is why it is optional:
:func:`peek_color_of` returns ``None`` for such a source, and the caller says that this cell cannot
show a live view and why, instead of showing a stale or fabricated picture.

The third rule is worded about decisions rather than about touching hardware at all, because taking
a frame from a live device is unavoidable and is dealt with below. It is not a loophole: the
rehearsal source's ``peek_color`` builds its image inline instead of calling ``acquire``, precisely
so that ``frames_served``, which the pick path reads, does not move.

The one side effect that is allowed, named explicitly: on a real camera, peeking consumes one
frameset from the device. There is no way to look at a stream without taking a frame from it, and it
is harmless here for a measured reason: the real source discards ``warmup_grabs`` (default 5) frames
at the top of every ``acquire()``, because the first frames after an idle period are not to be
trusted, so a frame taken by a viewer lands in that discarded prefix. What is not safe is peeking
while a pick is mid-acquire, because both would then call ``grab()`` on one unsynchronised
``rs.pipeline``; the camera package holds no lock (measured: ``grep -ri thread src/camera/`` is
empty), so the exclusion has to be enforced by the caller. :mod:`api.viewfinder` is that caller, and
it excludes on run state.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = ["COLOUR_SOURCE_KINDS", "ColourPeekable", "colour_source_kind", "peek_color_of"]

#: What a peekable source is a picture of. A synthetic scene rendered by the rehearsal source and a
#: frame off a D435 are both HxWx3 BGR arrays, and a console that labels the first one "live from
#: the camera" is making a claim about a room that has no camera in it. The source says which;
#: nothing downstream has to guess from the shape of the array.
#:
#:   "camera"     pixels that came off a physical device
#:   "synthetic"  pixels this process drew
#:   "unknown"    a source that can peek but does not say; described neutrally, never as a camera
COLOUR_SOURCE_KINDS = ("camera", "synthetic", "unknown")


@runtime_checkable
class ColourPeekable(Protocol):
    """A perception source that can hand over a colour image without doing any perception."""

    #: One of :data:`COLOUR_SOURCE_KINDS`. Optional; a source that omits it is described as
    #: "unknown", never as a camera.
    colour_source_kind: str

    def peek_color(self) -> np.ndarray | None:
        """The scene as colour, right now: ``HxWx3`` BGR ``uint8``, or ``None`` if unavailable.

        ``None`` is a legitimate answer (a camera that has not produced a frame yet, a source whose
        colour channel is disabled) and it means "no picture", never "black picture".
        """
        ...


def colour_source_kind(source: Any) -> str:
    """What ``source``'s picture is of: ``"camera"``, ``"synthetic"`` or ``"unknown"``.

    Unset means unknown, never camera. "camera" is the one answer that would put a claim on screen,
    so a source that does not declare its kind does not get it.
    """
    kind = str(getattr(source, "colour_source_kind", "") or "unknown")
    return kind if kind in COLOUR_SOURCE_KINDS else "unknown"


def peek_color_of(source: Any) -> np.ndarray | None:
    """The colour image a perception source can show, or ``None`` when it cannot show one.

    Duck-typed rather than isinstance-checked against the Protocol: the sim sources are constructed
    behind a lazy Isaac import and the rehearsal source is a plain dataclass, and requiring every
    one of them to inherit a marker would be a coupling that buys nothing here. What matters is
    whether the method exists.

    A source that raises is treated as one that cannot show a picture. A camera unplugged
    mid-session degrades the view, not the console, and never the pick.
    """
    peek = getattr(source, "peek_color", None)
    if peek is None or not callable(peek):
        return None
    try:
        image = peek()
    except Exception:  # noqa: BLE001 (a broken viewfinder must never take down the caller)
        return None
    if image is None:
        return None
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.size == 0:
        return None
    return np.ascontiguousarray(array.astype(np.uint8, copy=False))
