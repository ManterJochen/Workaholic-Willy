"""Sim-only grippers, backed by Isaac Sim.

These :class:`~src.robot.core.Gripper` implementations drive gripper articulations
inside Isaac Sim. They are kept out of the :mod:`src.robot.grippers` vendor registry
deliberately: there is no :class:`GripperVendor` member for them and
:func:`~src.robot.grippers.create_gripper` cannot reach them, so
``from_robot_config`` can never select one for a real robot. A real cell uses a real
vendor driver, meaning ``robotiq``, ``dummy`` or ``none``. The sim runners hand-build
these with a live ``IsaacSimSession``, and every ``isaacsim.*`` import stays lazy so
this package still imports on a host without Isaac.

* :class:`IsaacGripper` is parallel-jaw, driven by a swappable
  :class:`GripperProfile`.
* :class:`IsaacSuctionGripper` is the Isaac surface-gripper, vacuum on or off, driven
  by a swappable :class:`SuctionCupProfile`. A finer cup is a new profile rather than
  a new driver.
"""

from __future__ import annotations

from .gripper import (
    ROBOTIQ_2F85_PROFILE,
    SCHUNK_EGU50_PROFILE,
    SCHUNK_EZU35_PROFILE,
    GripperProfile,
    IsaacGripper,
)
from .suction_gripper import (
    SLIM_SUCTION_CUP,
    STANDARD_SUCTION_CUP,
    IsaacSuctionGripper,
    SuctionCupProfile,
)

__all__ = [
    "ROBOTIQ_2F85_PROFILE",
    "SCHUNK_EGU50_PROFILE",
    "SCHUNK_EZU35_PROFILE",
    "GripperProfile",
    "IsaacGripper",
    "STANDARD_SUCTION_CUP",
    "SLIM_SUCTION_CUP",
    "SuctionCupProfile",
    "IsaacSuctionGripper",
]
