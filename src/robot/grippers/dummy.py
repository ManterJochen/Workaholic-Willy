""":class:`DummyGripper`, a pure-Python implementation of the
:class:`~src.robot.core.Gripper` Protocol for offline development.

It has no SDK dependency. It tracks an internal width and clamps every command into
``[min_width_mm, max_width_mm]``.
"""

from __future__ import annotations

from src.robot.core import Gripper

__all__ = ["DummyGripper"]


class DummyGripper:
    """Sim gripper. Implements the :class:`Gripper` Protocol structurally."""

    def __init__(
        self,
        *,
        min_width_mm: float = 0.0,
        max_width_mm: float = 150.0,
        initial_width_mm: float | None = None,
    ) -> None:
        if min_width_mm < 0:
            raise ValueError("min_width_mm must be >= 0")
        if max_width_mm <= min_width_mm:
            raise ValueError("max_width_mm must be > min_width_mm")
        self._min = float(min_width_mm)
        self._max = float(max_width_mm)
        self._connected = False
        self._activated = False
        self._width = (
            float(initial_width_mm) if initial_width_mm is not None else self._max
        )

    # ------- introspection -------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def min_width_mm(self) -> float:
        return self._min

    @property
    def max_width_mm(self) -> float:
        return self._max

    # ------- lifecycle -----------------------------------------------------

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False
        self._activated = False

    def activate(self) -> None:
        if not self._connected:
            raise RuntimeError("DummyGripper.activate() requires connect() first.")
        self._activated = True

    # ------- commands ------------------------------------------------------

    def set_width_mm(
        self,
        width_mm: float,
        *,
        speed: float | None = None,
        force: float | None = None,
    ) -> None:
        if not self._connected:
            raise RuntimeError("DummyGripper not connected.")
        if not self._activated:
            # Auto-activation here is silent, where the production GripperController
            # logs a warning and then calls activate(). The difference is deliberate:
            # nothing consumes a warning from this double.
            self._activated = True
        clamped = max(self._min, min(self._max, float(width_mm)))
        # speed and force are accepted for Protocol compatibility and do nothing here.
        del speed, force
        self._width = clamped

    def get_width_mm(self) -> float:
        return self._width


# Structural check at import time, so Protocol drift surfaces early.
assert isinstance(DummyGripper(), Gripper), (
    "DummyGripper does not satisfy the Gripper Protocol"
)
