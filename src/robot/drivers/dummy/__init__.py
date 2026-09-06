"""Pure-Python sim drivers: :class:`DummyRobotArm` and its companions.

They are for offline development and need no external SDK.
"""

from __future__ import annotations

from .arm import DUMMY_CAPABILITIES, DummyRobotArm

__all__ = ["DUMMY_CAPABILITIES", "DummyRobotArm"]
