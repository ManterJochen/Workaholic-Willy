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

from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from src.contracts import UNSET, Maybe, chosen
from src.robot.core import (
    CameraWorldDecline,
    Gripper,
    RobotArm,
    RobotCapabilities,
    RobotVendor,
)
from src.robot.core.camera_world import without_camera_world as _without_camera_world
from src.robot.core.gripper import HoldEvidence
from src.robot.execution import handling as _handling
from src.robot.execution.cell_lock import CellLock, cell_lock_key
from src.robot.execution.lifecycle import ConnectedRobot, ConnectStage
from src.robot.execution.robot_parts import build_gripper, resolve_arm
from src.robot.safety import SafetyAttestation

if TYPE_CHECKING:
    from src.config.schema.robot import RobotConfig
    from src.geometry import Pose
    from src.robot.core.keep_out import SegmentationOffer
    from src.robot.execution.camera_world_wiring import CameraWorldWiring
    from src.robot.execution.wrist_bodies import WristBodies

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
            print(robot.grasp(40.0).render())   # close, read the hold, model the part
            print(robot.release().render())
        print(live.teardown.render())

    ``gripper`` is ``None`` for an arm-only robot. ``lock_key`` names the controller the cross-process
    lock is taken on, and is ``None`` for a robot that takes no lock. ``camera_world`` is what the
    cameras handed in built, ``None`` for a robot handed no camera.
    """

    arm: RobotArm
    gripper: Gripper | None
    lock_key: str | None
    #: The live planner world the robot's cameras built and its arm was handed, or why there is none.
    camera_world: "CameraWorldWiring | None" = None
    #: The wrist cameras among those cameras that the arm carries, handed to it before the world.
    wrist_bodies: "WristBodies | None" = None

    @classmethod
    def from_config(
        cls, robot_config: "RobotConfig", *, gripper: "Maybe[None]" = UNSET,
        cameras: "Maybe[Sequence[Any]]" = UNSET,
    ) -> "Robot":
        """The arm and the gripper ``robot_config`` describes. Connects nothing.

        The arm-vendor readiness gate runs first, as it does for a pick service built from the same
        tree. ``gripper`` left unset builds the configured end-effector, and one that had to be
        substituted is refused at connect. ``gripper=None`` builds the arm alone: no activation
        sweep, and a gripper that could not be built does not stand in the way. ``cameras`` are passed
        through to :meth:`from_parts` with this tree, which the world's planning block is read from.
        """
        arm = resolve_arm(robot_config, arm=None)
        hand = None if chosen(gripper) else build_gripper(robot_config, arm=arm)
        return cls.from_parts(arm=arm, gripper=hand, lock_key=cell_lock_key(robot_config), cameras=cameras,
                              robot_config=robot_config)

    @classmethod
    def from_parts(
        cls, *, arm: RobotArm, gripper: Gripper | None, lock_key: "Maybe[str | None]" = UNSET,
        cameras: "Maybe[Sequence[Any]]" = UNSET, robot_config: "Maybe[RobotConfig]" = UNSET,
    ) -> "Robot":
        """A robot around handles that are already built. No construction, no gate, no substitution.

        ``lock_key`` left unset is derived from the tree the arm keeps (``arm.config``): ``ur@<ip>``
        for a UR driver, ``kuka@<ip>`` for a KUKA driver. An arm that reports a vendor with a
        controller of its own (anything but sim and dummy) and yields no key is refused with
        :class:`LockKeyRequired`. An object that reports no capabilities at all is a stand-in, as
        ``resolve_arm`` counts it, and takes no lock.

        ``cameras`` are open camera owners, the primary first, each answering ``rig_id``, ``rig``,
        ``handle()`` and ``calibration()`` as :class:`~src.camera.orchestration.camera.Camera`
        does. The caller opens and releases them; the robot only reads them. The live planner world they
        build, under the planning block of ``robot_config`` (``arm.config`` when unset), is handed to the
        arm through its ``set_live_planner_world`` where it has one, and kept as ``camera_world`` either way.
        Unset builds no world. ``None`` is refused, because unset already says no camera was chosen.

        On an arm whose every motion needs a world or a decline (``camera_world_required``), a
        calibrated camera that yields no world is refused with
        :class:`~src.robot.execution.camera_world_wiring.CameraWorldRequired`, before the wrist bodies
        are handed over and again after the wiring.
        """
        _refuse_a_calibrated_robot_without_a_world(arm, cameras, robot_config)
        wrist = _wrist_bodies(arm, cameras, robot_config)
        wiring = _camera_world(arm, cameras, robot_config)
        if chosen(lock_key):
            return cls(arm=arm, gripper=gripper, lock_key=lock_key, camera_world=wiring, wrist_bodies=wrist)
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
        return cls(arm=arm, gripper=gripper, lock_key=derived, camera_world=wiring, wrist_bodies=wrist)

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

    def wrist_body_line(self) -> str:
        """One ASCII line saying which wrist cameras this robot's arm carries."""
        if self.wrist_bodies is None:
            return "wrist cameras  not handed any camera"
        return self.wrist_bodies.line()

    def camera_world_line(self) -> str:
        """One ASCII line saying what world this robot's cameras give its arm.

        On an arm whose every motion needs a world or a decline and that holds none, the line says
        so: a robot before its first calibration moves only under a decline.
        """
        needs = (": every planned motion needs a decline"
                 if bool(getattr(self.arm, "camera_world_required", False)) else "")
        if self.camera_world is None:
            return f"camera world  not handed any camera{needs}"
        if self.camera_world.world is None:
            return f"camera world  none: {self.camera_world.reason}{needs}"
        return "camera world  wired: " + ", ".join(repr(camera) for camera in self.camera_world.cameras)

    def safety(self) -> SafetyAttestation:
        """What this robot's arm will refuse. Commands nothing.

        Asked of the built arm, not of the config that asked for it, as ``Cell.safety`` asks.
        """
        return SafetyAttestation.of(self.arm)

    def grasp(self, width_mm: float) -> "_handling.HandReport":
        """Close the gripper to ``width_mm``, read what it measured, and hand a held part to the arm's model.

        Commands nothing on a robot with no gripper, a gripper that holds nothing, or a link that is not
        open. A hold the gripper measured, or that nothing could measure, is modelled as a carried part
        where the arm models one; a measured empty close is not. No force is commanded. A gripper that
        raises is stopped once where it can be.
        """
        return _handling.grasp(self, width_mm)

    def release(self) -> "_handling.HandReport":
        """Open the gripper to the hand's width and forget the carried part, unless the gripper still measures one."""
        return _handling.release(self)

    def pick(
        self, pose: Pose, width_mm: float, *, standoff_mm: float = 80.0, squeeze_mm: float = 1.0,
        pre_open_mm: "Maybe[float | None]" = UNSET, camera_world: "Maybe[CameraWorldDecline]" = UNSET,
        keep_out: "Maybe[SegmentationOffer]" = UNSET,
    ) -> "_handling.HandlingReport":
        """Pick the part at ``pose`` (BASE, approach along its +Z), ``width_mm`` across.

        Refuses before any command a robot that cannot hold, a closed link, a pose not in BASE, a camera
        world this arm's motion would be refused for, and an arm that keeps no straight line. Then it
        detaches, opens to ``pre_open_mm`` (the hand's width when unset, not at all when ``None``), makes
        a planned move to the standoff ``standoff_mm`` back along the approach, drives a line to ``pose``,
        runs :meth:`grasp` at ``width_mm`` minus ``squeeze_mm``, and drives a line back to the standoff.
        A close that measures nothing opens again and backs out. ``keep_out`` is held in the arm's live
        world through every motion. A refused motion ends the pick with nothing commanded after it.
        """
        return _handling.pick(self, pose, width_mm, standoff_mm=standoff_mm, squeeze_mm=squeeze_mm,
                              pre_open_mm=pre_open_mm, camera_world=camera_world, keep_out=keep_out)

    def place(
        self, pose: Pose, *, standoff_mm: float = 80.0, camera_world: "Maybe[CameraWorldDecline]" = UNSET,
    ) -> "_handling.HandlingReport":
        """Place the held part at ``pose``: a planned move to the standoff, a line in, :meth:`release`, a line out.

        The same refusals as :meth:`pick`. A release the gripper does not confirm leaves the arm where it
        stands.
        """
        return _handling.place(self, pose, standoff_mm=standoff_mm, camera_world=camera_world)

    def is_holding(self) -> HoldEvidence:
        """What the gripper measured about a hold, commanding nothing; UNMEASURED where nothing can say."""
        return _handling.is_holding(self)

    def without_camera_world(self, reason: str) -> AbstractContextManager[CameraWorldDecline]:
        """Decline the camera world for every motion of this robot's arm inside the ``with`` block.

            with robot.without_camera_world("bench check, no cameras mounted"):
                robot.arm.move(pose)

        Bound to this robot's arm, so another robot in the same process plans as it would without it,
        and a blank reason is refused here. A motion no planner plans or checks still says UNPLANNED, and on
        an arm whose live camera world is wired a declined planned or checked motion is refused. An arm that
        does not stamp its motions ignores the block. The block does not follow into a thread started inside it.
        """
        return _without_camera_world(self.arm, reason)


def _refuse_a_calibrated_robot_without_a_world(
    arm: RobotArm, cameras: "Maybe[Sequence[Any]]", robot_config: "Maybe[RobotConfig]",
) -> None:
    """Raise ``CameraWorldRequired`` where the arm needs a world, a handed camera is calibrated, and none comes of it.

    Reads config only, before anything is handed to the arm. The wiring half, cameras the plan names
    that build no world, is refused in :func:`_camera_world`.
    """
    if cameras is None or not chosen(cameras) or not bool(getattr(arm, "camera_world_required", False)):
        return
    tree = robot_config if chosen(robot_config) else getattr(arm, "config", None)
    owners = list(cameras)
    if tree is None or not owners:
        return  # _camera_world refuses both, saying why
    from src.robot.execution.camera_world_wiring import CameraWorldPlan, CameraWorldRequired

    plan = CameraWorldPlan.from_config(tree, [owner.rig for owner in owners], primary_rig_id=str(owners[0].rig_id))
    refusal = plan.refusal()
    if refusal is not None:
        raise CameraWorldRequired(refusal)


def _wrist_bodies(
    arm: RobotArm, cameras: "Maybe[Sequence[Any]]", robot_config: "Maybe[RobotConfig]",
) -> "WristBodies | None":
    """The wrist cameras among ``cameras`` the arm carries, handed to it; ``None`` when no camera was chosen.

    Raises :class:`~src.robot.execution.wrist_bodies.WristBodyRequired` for a body that cannot be placed,
    for an enabled wrist camera without one on a tree that reads geometry, and for an arm that cannot
    hold the bodies.
    """
    if cameras is None or not chosen(cameras):
        return None
    tree = robot_config if chosen(robot_config) else getattr(arm, "config", None)
    owners = list(cameras)
    if tree is None or not owners:
        return None  # _camera_world refuses both, saying why
    from src.robot.execution.wrist_bodies import WristBodies

    wrist = WristBodies.from_owners(tree, owners)
    wrist.hand_to(arm)
    return wrist


def _camera_world(
    arm: RobotArm, cameras: "Maybe[Sequence[Any]]", robot_config: "Maybe[RobotConfig]",
) -> "CameraWorldWiring | None":
    """The world ``cameras`` build for ``arm``, handed to it where it takes one; ``None`` when no camera was chosen."""
    if cameras is None:
        raise TypeError(
            "cameras=None: leave cameras UNSET for a robot handed no camera; None would say a camera was chosen "
            "and none was given"
        )
    if not chosen(cameras):
        return None
    tree = robot_config if chosen(robot_config) else getattr(arm, "config", None)
    if tree is None:
        raise ValueError(
            f"cameras were handed to a robot around {type(arm).__name__}, which keeps no tree (arm.config), and no "
            f"robot_config was passed: the live world's planning block is read from one of them"
        )
    # Imported here, so a robot handed no camera loads nothing of the camera world.
    from src.robot.execution.camera_world_wiring import CameraWorldPlan, CameraWorldWiring

    owners = list(cameras)
    if not owners:
        raise ValueError("cameras is empty: leave it unset for a robot handed no camera")
    plan = CameraWorldPlan.from_config(tree, [owner.rig for owner in owners], primary_rig_id=str(owners[0].rig_id))
    get_tcp_pose = getattr(arm, "get_tcp_pose", None)
    wiring = CameraWorldWiring.from_cameras(
        tree, plan=plan, cameras={str(owner.rig_id): owner for owner in owners},
        tool_pose=get_tcp_pose if callable(get_tcp_pose) else UNSET,
    )
    if wiring.world is None and plan.rig_ids and bool(getattr(arm, "camera_world_required", False)):
        from src.robot.execution.camera_world_wiring import CameraWorldRequired

        raise CameraWorldRequired(
            f"this arm plans with cuRobo and its calibrated cameras built no live world: {wiring.reason}")
    setter = getattr(arm, "set_live_planner_world", None)
    if wiring.world is not None and callable(setter):
        setter(wiring.world)
    return wiring
