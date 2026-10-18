"""A person acknowledging a fault at the cell: the one door to ``SupportsRobotStatus.acknowledge_fault`` (P0i i4).

Recovery is never automatic (owner, 2026-09-19): nothing in the execution, grasping or safety layers and nothing in the
console clears a fault, and ``tests/robot/drivers/test_robot_boundaries.py`` holds them to it. This package is where a person does,
standing at the cell and seeing the arm (owner, 2026-09-26: a CLI verb, outside those layers):

    python -m src.robot.faults acknowledge --profile ursim

``FaultAcknowledgement`` is its library twin (``from_tree(loaded, confirm=...)``, ``run()``, ``AcknowledgementReport``),
and ``acknowledge_command`` the command the console shows beside a fault, which it never clears itself.
"""

from __future__ import annotations

from .acknowledge import AcknowledgementOutcome, AcknowledgementReport, FaultAcknowledgement, acknowledge_command

__all__ = ["AcknowledgementOutcome", "AcknowledgementReport", "FaultAcknowledgement", "acknowledge_command"]
