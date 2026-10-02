"""The grasp overlays a task run captured: "what the robot decided", kept as its own image.

A task's progress listener (``api.task_run``) stores the service's overlay at every ``pick.executing`` (the grasp as
decided, rendered when it was computed, before the arm moved) and after each pick, and only where it is a new image:
the identity rule of ``PickRun`` (an image that already stood before the pick is no picture of this pick's grasp, so a
stale one is never claimed for it). The target a camera place found is kept beside them under ``"target"``.
``GET /v1/runs/{run_id}/overlays/{n}`` and ``GET /v1/runs/{run_id}/target/overlay`` serve them.

The store is bounded (:data:`OVERLAY_STORE_BYTES`) and lets the oldest overlay go first; an overlay larger than the
whole store is refused rather than evicting everything it could never replace. A run the registry forgets takes its
overlays with it (``Console`` wires :meth:`OverlayStore.forget` to the registry).

An overlay is never drawn over the moving live picture: the browser pins it over the stage for 5 s as its own image,
with the band "as decided at ...; the arm has moved since" on a wrist camera.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

__all__ = ["OVERLAY_STORE_BYTES", "OverlayStore", "overlay_url"]

#: The most PNG bytes the store keeps across all runs; the oldest overlay goes first beyond it.
OVERLAY_STORE_BYTES = 64 * 1024 * 1024


def overlay_url(run_id: str, key: int | str) -> str:
    """Where an overlay is served: ``/v1/runs/<id>/overlays/<n>``, or ``/v1/runs/<id>/target/overlay``."""
    if key == "target":
        return f"/v1/runs/{run_id}/target/overlay"
    return f"/v1/runs/{run_id}/overlays/{key}"


class OverlayStore:
    """PNG overlays keyed by (run, n) and (run, ``"target"``). One per console (``Console.overlays``); thread-safe,
    since a run's thread puts while a request reads."""

    def __init__(self, capacity_bytes: int = OVERLAY_STORE_BYTES) -> None:
        self.capacity_bytes = capacity_bytes
        self._lock = threading.Lock()
        self._items: OrderedDict[tuple[str, str], bytes] = OrderedDict()
        self._bytes = 0

    @property
    def size_bytes(self) -> int:
        """How many PNG bytes the store keeps now."""
        with self._lock:
            return self._bytes

    def put(self, run_id: str, key: int | str, png: bytes) -> str | None:
        """Keep ``png`` as the run's overlay ``key`` and answer its URL; ``None`` where nothing was kept (no image, or
        one larger than the whole store). The oldest overlays go first to make room."""
        if not isinstance(png, (bytes, bytearray)) or not png or len(png) > self.capacity_bytes:
            return None
        slot = (str(run_id), str(key))
        data = bytes(png)
        with self._lock:
            replaced = self._items.pop(slot, None)
            if replaced is not None:
                self._bytes -= len(replaced)
            while self._items and self._bytes + len(data) > self.capacity_bytes:
                _oldest, dropped = self._items.popitem(last=False)
                self._bytes -= len(dropped)
            self._items[slot] = data
            self._bytes += len(data)
        return overlay_url(run_id, key)

    def get(self, run_id: str, key: int | str) -> bytes | None:
        """The overlay's bytes, or ``None`` (``404 no_overlay``)."""
        with self._lock:
            return self._items.get((str(run_id), str(key)))

    def count(self, run_id: str) -> int:
        """How many numbered overlays (the target's aside) the store keeps of one run."""
        with self._lock:
            return sum(1 for (run, key) in self._items if run == run_id and key != "target")

    def forget(self, run_id: str) -> None:
        """Drop every overlay of one run. Idempotent, and safe for a run that kept none."""
        with self._lock:
            for slot in [slot for slot in self._items if slot[0] == str(run_id)]:
                self._bytes -= len(self._items.pop(slot))
