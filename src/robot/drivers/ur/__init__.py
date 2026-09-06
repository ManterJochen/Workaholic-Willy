"""The Universal Robots driver package.

:class:`URConnection`, :class:`URPose` and the pose adapters are eager, because they
carry no vendor SDK. :class:`URRobotArm`, ``UR_CAPABILITIES`` and
:class:`MotionController` resolve on first access through ``__getattr__``, which keeps
the driver chain out of an import that only needs the pose types.
"""

from __future__ import annotations

from importlib import import_module

from .connection import URConnection
from .pose import URPose
from .pose_adapter import pose_to_urpose, urpose_to_pose

_LAZY: dict[str, tuple[str, str]] = {
    "URRobotArm": ("src.robot.drivers.ur.arm", "URRobotArm"),
    "UR_CAPABILITIES": ("src.robot.drivers.ur.arm", "UR_CAPABILITIES"),
    "MotionController": ("src.robot.drivers.ur.motion", "MotionController"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'src.robot.drivers.ur' has no attribute {name!r}")
    module_name, attr = target
    value = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "UR_CAPABILITIES",
    "URConnection",
    "URPose",
    "URRobotArm",
    "pose_to_urpose",
    "urpose_to_pose",
]
