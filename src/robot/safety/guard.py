"""SafetyGuard, the vendor-neutral safety-guard contract.

Each guard in the safety-preflight pipeline implements the :class:`SafetyGuard`
:class:`~typing.Protocol`. The orchestrator ``SafetyPreflight`` feeds every guard a
:class:`SafetyContext` describing the commanded motion and any current state, and
consumes the guard :class:`~src.robot.safety.decision.SafetyDecision`.

Design notes
------------
* A guard is stateless across calls apart from an explicit documented memo, such as
  the motion-continuity guard caching its last accepted target. Cached state belongs
  on the guard instance and not on module state, which keeps several arms and
  preflights isolated.

* A guard does not raise on an ordinary rejection. Returning
  :meth:`SafetyDecision.reject` is the only sanctioned channel.

* A guard may raise on a programming-error input, such as a wrong-frame pose reaching
  a guard that should only see :attr:`Frame.BASE`. The orchestrator does not catch
  those, because they are a bug at the call site.

* A guard short-circuits when its ``enforce`` flag is off, returning
  :meth:`SafetyDecision.accept` immediately, which keeps the pipeline predictable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.geometry import Pose
    from src.robot.core import (
        JointPositions,
        MotionCommand,
        RobotArm,
    )

    from .decision import SafetyDecision

__all__ = [
    "SafetyContext",
    "SafetyGuard",
]


@dataclass(frozen=True)
class SafetyContext:
    """Input bundle passed to every guard in the preflight pipeline.

    Parameters
    ----------
    command
        Which :class:`MotionCommand` is being evaluated, which lets a guard
        specialise: ``MOVE_HOME`` may skip continuity.
    target_pose
        The commanded TCP pose. ``None`` for a pure joint move.
    target_joints
        The commanded joint positions. ``None`` while the driver has not resolved IK.
    current_pose
        The arm current TCP pose at evaluation time. ``None`` where it is
        unavailable, for instance with the driver disconnected.
    current_joints
        The arm current joint positions. ``None`` where unavailable.
    last_target_pose
        Previous accepted TCP target, used by a continuity guard. The preflight
        orchestrator may populate it from its memo cache.
    last_target_joints
        Previous accepted joint target, used by a continuity guard.
    arm
        Optional reference to the :class:`RobotArm` being commanded. A guard that
        needs driver-side telemetry, such as the UR joint-limit guard pulling
        ``query_joint_limits()``, consults it when present. A guard tolerates ``None``
        and either falls back to configured static data or returns
        :meth:`SafetyDecision.unavailable`.
    """

    command: "MotionCommand"
    target_pose: "Pose | None" = None
    target_joints: "JointPositions | None" = None
    current_pose: "Pose | None" = None
    current_joints: "JointPositions | None" = None
    last_target_pose: "Pose | None" = None
    last_target_joints: "JointPositions | None" = None
    arm: "RobotArm | None" = None


@runtime_checkable
class SafetyGuard(Protocol):
    """Vendor-neutral safety-guard interface.

    Implementations live under :mod:`src.robot.safety` and are wired into
    :class:`SafetyPreflight` in a deterministic order that the orchestrator defines.
    """

    @property
    def name(self) -> str:
        """Short, stable identifier such as ``"workspace"`` or ``"joint_limit"``.

        It becomes the ``guard`` field on every :class:`SafetyDecision` this guard
        produces.
        """

    def evaluate(self, ctx: SafetyContext) -> "SafetyDecision":
        """Evaluate ``ctx`` and return a :class:`SafetyDecision`.

        A guard does not raise on an ordinary rejection, which is
        :meth:`SafetyDecision.reject`. It may raise on a programming-error input.
        """
