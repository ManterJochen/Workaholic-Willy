"""A robot as one noun: the arm, the gripper on it, and the one way to bring them up and down.

``Cell`` is the whole pick: it builds perception and the grasp stack before it can connect anything.
Calibration, a bench check and a jog from Python need only the arm and the gripper, and ``Robot``
hands over those two with no pick service built around them.

``Robot`` builds the same two handles through :mod:`src.robot.execution.robot_parts`, the builder the
pick service calls, and connects them through
:class:`~src.robot.execution.lifecycle.ConnectedRobot`, which runs the enter and the exit
``ConnectedCell`` runs. The lock, the arm before the gripper, the refusal of a substituted gripper and
the teardown order are one implementation for both nouns.

``Cell`` and ``Robot`` are siblings. ``Cell.build`` does not build a ``Robot``: it reaches the builder
through the pick service, after the grasping-block refusals and after the camera opens.

Importing this module loads no grasping stack, no perception, no model and no camera package.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from src.contracts import UNSET, Maybe, chosen
from src.robot.core import (
    CameraWorldDecline,
    Gripper,
    RobotArm,
    RobotCapabilities,
    RobotVendor,
)
from src.robot.core.camera_world import without_camera_world as _without_camera_world
from src.robot.execution.cell_lock import CellLock, cell_lock_key
from src.robot.execution.lifecycle import ConnectedRobot, ConnectStage
from src.robot.execution.robot_parts import build_gripper, resolve_arm
from src.robot.safety import SafetyAttestation

if TYPE_CHECKING:
    from src.config.schema.robot import RobotConfig

__all__ = ["LockKeyRequired", "Robot"]

#: Vendors whose arms drive no controller of their own, so there is nothing to lock for them.
_OWNS_NO_CONTROLLER = frozenset({RobotVendor.SIM.value, RobotVendor.DUMMY.value})


class LockKeyRequired(ValueError):
    """An arm that drives a controller was handed in, and no lock key derives from it.

    The key is the controller address in the tree the arm was built with (``arm.config``). An adapter
    that keeps no tree, or a UR driver built from a tree naming another vendor, yields none, and a
    robot that took no lock on such an arm would let a second process compete for its controller.
    Stating ``lock_key`` resolves it, and ``lock_key=None`` states that no lock is taken.
    """


@dataclass(frozen=True)
class Robot:
    """An arm and the gripper on it, built and not yet connected.

        from src.config import load_robot_section
        from src.robot.execution.robot import Robot

        robot = Robot.from_config(load_robot_section())
        print(robot.safety().render())          # what this arm refuses, before it moves
        with robot.connected() as live:         # lock, arm, then gripper
            live.gripper.set_width_mm(40.0)
        print(live.teardown.render())

    ``gripper`` is ``None`` for an arm-only robot. ``lock_key`` names the controller the cross-process
    lock is taken on, and is ``None`` for a robot that takes no lock.
    """

    arm: RobotArm
    gripper: Gripper | None
    lock_key: str | None

    @classmethod
    def from_config(
        cls, robot_config: "RobotConfig", *, gripper: "Maybe[None]" = UNSET,
    ) -> "Robot":
        """The arm and the gripper ``robot_config`` describes. Connects nothing.

        The arm-vendor readiness gate runs first, as it does for a pick service built from the same
        tree. ``gripper`` left unset builds the configured end-effector, and one that had to be
        substituted is refused at connect. ``gripper=None`` builds the arm alone: no activation
        sweep, and a gripper that could not be built does not stand in the way.
        """
        arm = resolve_arm(robot_config, arm=None)
        hand = None if chosen(gripper) else build_gripper(robot_config, arm=arm)
        return cls.from_parts(arm=arm, gripper=hand, lock_key=cell_lock_key(robot_config))

    @classmethod
    def from_parts(
        cls, *, arm: RobotArm, gripper: Gripper | None, lock_key: "Maybe[str | None]" = UNSET,
    ) -> "Robot":
        """A robot around handles that are already built. No construction, no gate, no substitution.

        ``lock_key`` left unset is derived from the tree the arm keeps (``arm.config``): ``ur@<ip>``
        for a UR driver, ``kuka@<ip>`` for a KUKA driver. An arm that reports a vendor with a
        controller of its own (anything but sim and dummy) and yields no key is refused with
        :class:`LockKeyRequired`. An object that reports no capabilities at all is a stand-in, as
        ``resolve_arm`` counts it, and takes no lock.
        """
        if chosen(lock_key):
            return cls(arm=arm, gripper=gripper, lock_key=lock_key)
        derived = cell_lock_key(getattr(arm, "config", None))
        if derived is None:
            capabilities = getattr(arm, "capabilities", None)
            vendor = capabilities.vendor if isinstance(capabilities, RobotCapabilities) else None
            if vendor is not None and vendor not in _OWNS_NO_CONTROLLER:
                raise LockKeyRequired(
                    f"the arm in hand ({type(arm).__name__}) reports vendor {vendor!r}, which drives "
                    f"a controller of its own, and no lock key derives from the tree it keeps: "
                    f"arm.config is missing, names another vendor, or names no controller address. "
                    f"State lock_key as the controller it drives (for a UR, 'ur@<ip>'), or pass "
                    f"lock_key=None to take no lock on purpose."
                )
        return cls(arm=arm, gripper=gripper, lock_key=derived)

    def connected(
        self, *, announce: "Callable[[ConnectStage], None] | None" = None,
    ) -> ConnectedRobot:
        """The robot, connected, for the duration of a ``with`` block.

        It takes the cross-process lock first when ``lock_key`` is set. That is the lock ``Cell``
        takes for the same controller, so the two refuse each other. Then the arm, then the gripper,
        and on the way out the gripper, the arm and the lock. A substituted gripper is refused before
        the arm is commanded.
        """
        lock = CellLock(self.lock_key, owner="Robot") if self.lock_key else None
        return ConnectedRobot(self.arm, self.gripper, lock=lock, announce=announce)

    def safety(self) -> SafetyAttestation:
        """What this robot's arm will refuse. Commands nothing.

        Asked of the built arm, not of the config that asked for it, as ``Cell.safety`` asks.
        """
        return SafetyAttestation.of(self.arm)

    def without_camera_world(self, reason: str) -> AbstractContextManager[CameraWorldDecline]:
        """Decline the camera world for every motion of this robot's arm inside the ``with`` block.

            with robot.without_camera_world("bench check, no cameras mounted"):
                robot.arm.move(pose)

        Bound to this robot's arm, so another robot in the same process plans as it would without it,
        and a blank reason is refused here. A motion no planner plans still says UNPLANNED, and on an arm
        whose live camera world is wired a declined planned motion is refused. An arm that does not stamp
        its motions ignores the block. The block does not follow into a thread started inside it.
        """
        return _without_camera_world(self.arm, reason)
