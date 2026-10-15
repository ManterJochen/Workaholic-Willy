"""The UR transport over a test's own fake connection, as the UR arm builds it (P0e E7).

The cuRobo planner is handed the arm's transport since E7, and reads the joints it plans from and sends a plan through
it. Since P0g slice 13 a planner test builds the planner over the fake transport (``tests/_fake_transport.py``); what
wraps a fake connection here is a test of the UR itself, which reads the `get_joint_positions` and `moveJ` calls the
real translation sent.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from src.robot.drivers.ur.transport import URTransport


def over(connection: Any) -> URTransport:
    """``connection`` behind the UR transport; the planner asks no motion controller, so that half is a stand-in."""
    return URTransport(connection, MagicMock(name="motion controller the planner never asks"))
