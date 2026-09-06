"""The KUKA arm driver, an EthernetKRL implementation.

Public surface
--------------
* :class:`KukaRobotArm` is the :class:`src.robot.core.RobotArm` implementation that
  talks to the KUKA controller over TCP with XML.
* :class:`EkiClient` is the transport underneath, exposed so a caller can drive it
  directly.
* :class:`KukaCartesian` is the value object equivalent to a KUKA e6pos.

The KRL and EKI XML templates for the controller side live under
``config/robot/templates/kuka/``.
"""

from __future__ import annotations

from .arm import KUKA_CAPABILITIES, KukaRobotArm
from .eki_client import EkiClient
from .pose_convert import (
    KukaCartesian,
    joints_deg_to_rad,
    joints_rad_to_deg,
    kuka_cartesian_to_pose,
    pose_to_kuka_cartesian,
)

__all__ = [
    "KUKA_CAPABILITIES",
    "EkiClient",
    "KukaCartesian",
    "KukaRobotArm",
    "joints_deg_to_rad",
    "joints_rad_to_deg",
    "kuka_cartesian_to_pose",
    "pose_to_kuka_cartesian",
]
