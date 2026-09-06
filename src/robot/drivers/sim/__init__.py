"""The Isaac-backed simulation :class:`RobotArm` driver package.

Importing it is always safe, including on a host without the Isaac SDK, because:

* :mod:`.config` and :mod:`.adapter` are pure Python;
* :mod:`.session` and :mod:`.arm` touch Isaac SDK symbols only inside
  :meth:`IsaacSimSession.start`, which :meth:`IsaacRobotArm.connect` invokes.

Calling :meth:`IsaacRobotArm.connect` on a host without Isaac raises
:class:`~src.robot.core.IsaacNotAvailableError`, re-exported here for convenience. The
:class:`RobotVendor.SIM` factory in :mod:`src.robot.drivers` therefore stays
import-safe while still constructing the driver eagerly, so its protocol surface can be
inspected. The Isaac-backed grippers live in :mod:`src.robot.grippers.sim`, which is
sim-only and deliberately unregistered.
"""

from __future__ import annotations

from .adapter import (
    willy_joints_to_isaac,
    willy_pose_to_isaac,
    isaac_joints_to_willy,
    isaac_pose_to_willy,
    isaac_rotmat_to_wxyz,
    isaac_wxyz_to_xyzw,
    metres_to_millimetres,
    millimetres_to_metres,
    xyzw_to_isaac_wxyz,
)
from ...core import IsaacNotAvailableError
from .arm import ISAAC_CAPABILITIES, IsaacRobotArm
from .config import SimCameraConfig, SimRobotConfig
from .session import IsaacSimSession

__all__ = [
    "ISAAC_CAPABILITIES",
    "IsaacNotAvailableError",
    "IsaacRobotArm",
    "IsaacSimSession",
    "SimCameraConfig",
    "SimRobotConfig",
    "willy_joints_to_isaac",
    "willy_pose_to_isaac",
    "isaac_joints_to_willy",
    "isaac_pose_to_willy",
    "isaac_rotmat_to_wxyz",
    "isaac_wxyz_to_xyzw",
    "metres_to_millimetres",
    "millimetres_to_metres",
    "xyzw_to_isaac_wxyz",
]
