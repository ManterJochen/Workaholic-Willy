"""The camera world an Isaac cell's motions stand on: a decline held for its life, or a wired world.

Every Isaac boot on the cuRobo planner states its camera world, because a cuRobo motion with neither
a world nor a decline is refused. A runner that plans without a camera passes a
:class:`~src.robot.core.camera_world.CameraWorldDecline` naming itself, held on the cell until it
closes; the reference runner passes :class:`SimCameraWorld` and gets a live world from the Isaac
overhead camera. An ik or RMPflow boot needs neither, because nothing plans against a world there.

Importable off the box: nothing here imports Isaac.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

from src.robot.core.camera_world import CameraWorldDecline

__all__ = ["CellCameraWorld", "SimCameraWorld", "hold_camera_world_decline"]


@dataclass(frozen=True)
class SimCameraWorld:
    """A request for a live camera world from the Isaac camera named ``rig_id``.

    Only ``overhead`` is authored by every boot.
    """

    rig_id: str = "overhead"


@dataclass(frozen=True)
class CellCameraWorld:
    """The camera world an Isaac cell's arm stands on: its held decline, or the wiring it was handed."""

    #: The decline held for the cell's life, or ``None`` for a wired cell.
    decline: CameraWorldDecline | None
    #: The ``CameraWorldWiring`` a wired cell was handed, or ``None`` for a declined cell.
    wiring: Any = None
    #: What keeps the decline in scope; closing it ends the decline.
    hold: ExitStack | None = None
    #: For a wired cell, how far the fitted CAMERA to BASE turns from the scene's hand-built one, in
    #: degrees.
    fit_angle_deg: float | None = None

    def render(self) -> str:
        """Describe this cell's camera world to a person as ASCII text, with no trailing newline."""
        if self.decline is not None:
            return f"declined({self.decline.reason})"
        cameras = ", ".join(repr(camera) for camera in getattr(self.wiring, "cameras", ()) or ())
        angle = "" if self.fit_angle_deg is None else f" fit_vs_scene_deg={self.fit_angle_deg:.2f}"
        return f"wired({cameras}){angle}"

    def close(self) -> None:
        """End the held decline. Idempotent."""
        if self.hold is not None:
            self.hold.close()


def hold_camera_world_decline(arm: Any, decline: CameraWorldDecline, stack: ExitStack) -> CellCameraWorld:
    """Enter ``arm.without_camera_world(reason)`` on ``stack`` and return what the cell holds.

    The decline stays in scope for every motion the cell's thread commands until ``stack`` closes.
    It does not follow a thread started later, which no Isaac runner starts.
    """
    if not isinstance(decline, CameraWorldDecline):
        raise TypeError(f"camera_world is a CameraWorldDecline or a SimCameraWorld, not {decline!r}")
    stack.enter_context(arm.without_camera_world(decline.reason))
    return CellCameraWorld(decline=decline, hold=stack)
