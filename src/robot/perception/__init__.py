"""Live-camera perception adapters for the real-hardware pick path.

Lives under ``robot/`` because the dependency edge only ever runs ``robot -> camera``: an adapter
that needs both a camera streamer and ``robot.grasping``'s ``PerceptionFrame`` cannot sit in
``camera/`` without inverting it. See :mod:`src.robot.perception.realsense_source`.

Nothing here is imported at package load beyond the pure adapter; ``pyrealsense2`` and torch are
built only by the ``python -m src.robot.perception`` exerciser.
"""

from __future__ import annotations

from .realsense_source import RealSenseVisionPerceptionSource

__all__ = ["RealSenseVisionPerceptionSource"]
