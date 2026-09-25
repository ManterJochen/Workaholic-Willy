"""Live-camera perception adapters for the real-hardware pick path.

Lives under ``robot/`` because the dependency edge only ever runs ``robot -> camera``: an adapter
that needs both a camera streamer and ``robot.grasping``'s ``PerceptionFrame`` cannot sit in
``camera/`` without inverting it. See :mod:`src.robot.perception.realsense_source`.

``Locator`` places what one open camera grounds in BASE, and ``Located.scene(i, robot_config)``
hands object i to the grasp planner, whose candidates give the pose and the width ``Robot.pick``
takes. The locator's names are exported here and resolve on first use: the locator brings the
multi-view association and with it scipy, 527 modules and about 280 ms, which a console peek through
``viewfinder`` does not need.

Nothing here is imported at package load beyond the pure adapter; ``pyrealsense2`` and torch are
built only by the ``python -m src.robot.perception`` exerciser.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .realsense_source import RealSenseVisionPerceptionSource

if TYPE_CHECKING:  # pragma: no cover (typing only; at run time __getattr__ resolves them)
    from .locator import Located, LocatedObject, LocatedOrientation, Locator, LocatorRefused, SetDown

#: Public name -> the module that defines it, imported on first use.
_LAZY: dict[str, str] = {
    "Located": "src.robot.perception.locator",
    "LocatedObject": "src.robot.perception.locator",
    "LocatedOrientation": "src.robot.perception.locator",
    "Locator": "src.robot.perception.locator",
    "LocatorRefused": "src.robot.perception.locator",
    "SetDown": "src.robot.perception.locator",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        value = getattr(import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(["RealSenseVisionPerceptionSource", *_LAZY])
