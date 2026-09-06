"""``willy_sim``: the Isaac Sim composition and runner layer for Workaholic-Willy.

The composition root of the simulated cell. It ties the vendor-neutral grasping stack
(``src.robot.grasping`` and ``execution``) to the Isaac driver (``src.robot.drivers.sim``):
it builds the perception source and the scene, wires the arm and gripper, and drives
``AutonomousGraspService.pick()`` in the simulator.

It sits above both ``drivers`` and ``grasping`` so it can import from each. The perception
source produces a ``grasping.PerceptionFrame``, which ``drivers`` is forbidden to import under
the downward-stack rule. Modules that touch Isaac keep their ``isaacsim.*`` imports lazy, so
this package stays importable on macOS and CI.
"""

from __future__ import annotations

from .perception import (
    GroundTruthPerceptionSource,
    GroundTruthSegmentation,
    object_mask_from_frame,
)

__all__ = [
    "GroundTruthPerceptionSource",
    "GroundTruthSegmentation",
    "object_mask_from_frame",
]
