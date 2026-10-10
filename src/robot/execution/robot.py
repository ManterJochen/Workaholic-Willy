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

The verbs that move the arm (``move``, ``move_joints``, ``home``, ``pick``, ``place``) go through the
arm's own checked verbs and read, before any command, whether its motions go through cuRobo and the
exact mesh guard with the camera world (:mod:`src.robot.execution.motion`). Each returns a frozen
report and raises only for a programmer's error. ``tool_down`` builds the pose they take through the
cell: ``Pose.tool_down``, closing the way the cell's hand naturally stands (``robot.natural_closing_axis``)
where the program names no axis.

Importing this module loads no grasping stack, no perception, no model and no camera package.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable

from src.contracts import UNSET, Maybe, chosen
from src.robot.core import (
    CameraWorldDecline,
    Gripper,
    JointPositions,
    RobotArm,
    RobotCapabilities,
    RobotVendor,
)
from src.robot.core.camera_world import without_camera_world as _without_camera_world
from src.robot.core.gripper import HoldEvidence
from src.robot.execution import handling as _handling
from src.robot.execution import motion as _motion
from src.robot.execution.cell_lock import CellLock, cell_lock_key
from src.robot.execution.lifecycle import ConnectedRobot, ConnectStage, let_go_of_held_views
from src.robot.execution.robot_parts import build_gripper, resolve_arm
from src.robot.safety import SafetyAttestation

if TYPE_CHECKING:
    from src.config.schema.robot import RobotConfig
    from src.config.tree import LoadedTree
    from src.geometry import Pose
    from src.geometry.closing_axis import ClosingAxisLike
    from src.robot.core.keep_out import SegmentationOffer
    from src.robot.execution.camera_world_wiring import CameraWorldWiring
    from src.robot.execution.real_cell.preflight import PreflightReport
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
    """An arm and the hand on it, built and not yet connected: the object a program moves.

    Build it with :meth:`from_tree` (the usual way), :meth:`from_config` or :meth:`from_parts`, then connect it in a
    ``with`` block and call its verbs; every verb returns a report that prints as itself.

    ```python
    robot = Robot.from_tree(load_tree("console_dummy"))
    with robot.connected():                    # lock, arm, then hand
        print(robot.home())
        print(robot.move(robot.tool_down(450.0, 100.0, 300.0)))
        print(robot.grasp(40.0))
    ```

    Attributes:
        arm (RobotArm): The arm driver, built and not connected.
        gripper (Gripper | None): The hand on it; ``None`` for an arm-only robot.
        lock_key (str | None): The controller the cross-process lock is taken on (``ur@<ip>``); ``None`` for a robot
            that takes no lock (a desk arm).
        camera_world (CameraWorldWiring | None): What the cameras handed in built for the arm's planner; ``None`` for a
            robot handed no camera.
        wrist_bodies (WristBodies | None): The camera bodies the arm carries: every one its tree declares for a robot
            :meth:`from_tree` built, camera open or not, else those of the cameras handed in.
        robot_config (RobotConfig | None): The robot section a factory was handed; ``None`` for a robot built around
            handles alone.
        tree (LoadedTree | None): The loaded tree :meth:`from_tree` read; ``None`` otherwise.
    """

    arm: RobotArm
    gripper: Gripper | None
    lock_key: str | None
    #: The live planner world the robot's cameras built and its arm was handed, or why there is none.
    camera_world: "CameraWorldWiring | None" = None
    #: The wrist cameras the arm carries, handed to it before the world: every body the tree declares for a
    #: robot ``from_tree`` built, the bodies of the cameras handed in otherwise, ``None`` when neither was read.
    wrist_bodies: "WristBodies | None" = None
    #: The robot section this robot was built from, where a factory was handed one.
    robot_config: "RobotConfig | None" = field(default=None, repr=False, compare=False)
    #: The loaded tree ``from_tree`` read the robot section from, for the camera half and the root.
    tree: "LoadedTree | None" = field(default=None, repr=False, compare=False)

    @classmethod
    def from_tree(
        cls, tree: "LoadedTree", *, gripper: "Maybe[None]" = UNSET, cameras: "Maybe[Sequence[Any]]" = UNSET,
        unmodelled_wrist_body: "Maybe[str]" = UNSET,
    ) -> "Robot":
        """The robot a loaded tree describes, built from its robot section. Connects nothing.

        Args:
            tree (LoadedTree): What ``load_tree()`` returned; ``load_tree()`` reads the cell ``WILLY_PROFILE`` names.
            gripper (Maybe[None]): Leave unset to build the hand the tree names; ``None`` builds the arm alone, with no
                activation sweep (default: UNSET).
            cameras (Maybe[Sequence[Any]]): Open camera owners (:class:`Camera`), the primary first, whose live world
                the arm plans in; unset builds no world. ``None`` is refused, since unset already says no camera
                (default: UNSET).
            unmodelled_wrist_body (Maybe[str]): Why this robot may move without a wrist camera body that cannot be
                placed yet because its camera is not calibrated; read only for such a body (default: UNSET).

        Returns:
            Robot: The robot, keeping the tree, so :meth:`preflight` reads the camera half from the same load. Every
                wrist camera body the tree declares hangs on the arm, camera open or not.

        Raises:
            ConfigError: The tree did not load (its own refusal).
            TypeError: ``tree`` is not a ``LoadedTree``, or ``cameras=None``.
            RobotConnectionError: This machine is not ready for the arm's vendor (its SDK is missing).
            WristBodyUnplaced: A wrist camera's body cannot be placed (its camera is not calibrated) and no
                ``unmodelled_wrist_body`` says why the robot may move without it.
            CameraWorldRequired: The arm needs a camera world for every motion and a calibrated camera builds none.
        """
        from src.config.loader import ConfigError
        from src.config.tree import LoadedTree

        if not isinstance(tree, LoadedTree):
            raise TypeError(
                f"Robot.from_tree takes a loaded tree, not {type(tree).__name__}: pass "
                f"ConfigTree.from_directory(...).load()")
        if not tree.ok:
            raise ConfigError(f"the tree did not load, so it describes no robot:\n{tree.error}")
        robot_config = tree.robot
        arm = resolve_arm(robot_config, arm=None)
        wrist = _tree_wrist_bodies(tree, robot_config, cameras, unmodelled_wrist_body)
        hand = None if chosen(gripper) else build_gripper(robot_config, arm=arm)
        robot = cls._assemble(arm=arm, gripper=hand, lock_key=cell_lock_key(robot_config), cameras=cameras,
                              robot_config=robot_config, wrist=wrist)
        return replace(robot, tree=tree)

    @classmethod
    def from_config(
        cls, robot_config: "RobotConfig", *, gripper: "Maybe[None]" = UNSET,
        cameras: "Maybe[Sequence[Any]]" = UNSET,
    ) -> "Robot":
        """The arm and the hand a robot section describes. Connects nothing.

        The arm-vendor readiness gate runs first, as for a pick service built from the same tree. A robot section holds
        no camera section, so the arm carries only the bodies of the cameras handed in; a robot beside a wrist camera it
        was not handed is built with :meth:`from_tree`, which reads them all.

        Args:
            robot_config (RobotConfig): The robot section, ``tree.robot``.
            gripper (Maybe[None]): Leave unset to build the configured hand (a substituted one is refused at connect);
                ``None`` builds the arm alone (default: UNSET).
            cameras (Maybe[Sequence[Any]]): Open camera owners, passed to :meth:`from_parts` with this section, whose
                planning block the world is built under (default: UNSET).

        Returns:
            Robot: The robot, keeping ``robot_config``.

        Raises:
            RobotConnectionError: This machine is not ready for the arm's vendor (its SDK is missing).
            TypeError: ``cameras=None``.
            CameraWorldRequired: The arm needs a camera world for every motion and a calibrated camera builds none.
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
        """A robot around handles that are already built: no construction, no gate, no substitution.

        Args:
            arm (RobotArm): The arm driver.
            gripper (Gripper | None): The hand, or ``None`` for an arm-only robot.
            lock_key (Maybe[str | None]): The controller the cross-process lock is taken on. Unset derives it from the
                tree the arm keeps (``ur@<ip>``, ``kuka@<ip>``); ``None`` states that no lock is taken (default: UNSET).
            cameras (Maybe[Sequence[Any]]): Open camera owners, the primary first, each answering ``rig_id``, ``rig``,
                ``handle()`` and ``calibration()`` as :class:`Camera` does; the caller opens and releases them, the
                robot only reads them. Unset builds no world; ``None`` is refused (default: UNSET).
            robot_config (Maybe[RobotConfig]): The section whose planning block the camera world is built under; unset
                reads the arm's ``arm.config`` (default: UNSET).

        Returns:
            Robot: The robot; ``camera_world`` holds what the cameras built, handed to the arm through its
                ``set_live_planner_world`` where it has one.

        Raises:
            LockKeyRequired: An arm with a controller of its own (anything but sim and dummy) yields no lock key and
                none was stated.
            CameraWorldRequired: The arm needs a camera world for every motion and a calibrated camera builds none.
            TypeError: ``cameras=None``.
            ValueError: ``cameras`` is empty, or was handed to an arm that keeps no tree with no ``robot_config``.
        """
        return cls._assemble(arm=arm, gripper=gripper, lock_key=lock_key, cameras=cameras,
                             robot_config=robot_config, wrist=UNSET)

    @classmethod
    def _assemble(
        cls, *, arm: RobotArm, gripper: Gripper | None, lock_key: "Maybe[str | None]",
        cameras: "Maybe[Sequence[Any]]", robot_config: "Maybe[RobotConfig]", wrist: "Maybe[WristBodies]",
    ) -> "Robot":
        """:meth:`from_parts` with the wrist bodies already resolved and handed over, or resolved from ``cameras``."""
        _refuse_a_calibrated_robot_without_a_world(arm, cameras, robot_config)
        if chosen(wrist):
            # Before the world, because the world's self filter reads the bodies the arm holds.
            wrist.hand_to(arm)
            held: "WristBodies | None" = wrist
        else:
            held = _wrist_bodies(arm, cameras, robot_config)
        wiring = _camera_world(arm, cameras, robot_config)
        kept = robot_config if chosen(robot_config) else None
        if chosen(lock_key):
            return cls(arm=arm, gripper=gripper, lock_key=lock_key, camera_world=wiring, wrist_bodies=held,
                       robot_config=kept)
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
        return cls(arm=arm, gripper=gripper, lock_key=derived, camera_world=wiring, wrist_bodies=held,
                   robot_config=kept)

    def connected(
        self, *, announce: "Callable[[ConnectStage], None] | None" = None,
    ) -> ConnectedRobot:
        """The robot, connected, for the duration of a ``with`` block.

        The cross-process lock comes first when ``lock_key`` is set (the lock ``Cell`` takes for the same controller, so
        the two refuse each other), then the arm, then the hand; on the way out the hand, the arm and the lock, after
        the frames a wrist camera's ``Locator.look_around`` held in the arm's live world are let go.

        Args:
            announce (Callable[[ConnectStage], None] | None): Called with each stage as the connect reaches it, for a
                console that shows progress; ``None`` says nothing (default: None).

        Returns:
            ConnectedRobot: The context manager; ``with robot.connected() as live:`` gives the robot, and
                ``live.teardown`` says how it came down.

        Raises:
            CellBusy: On entering the block: another process holds the controller's lock; the message names it.
            NoRealGripper: On entering the block: the tree names a hand this arm cannot drive, so a stand-in was built.
        """
        lock = CellLock(self.lock_key, owner="Robot") if self.lock_key else None
        return ConnectedRobot(self.arm, self.gripper, lock=lock, announce=announce)

    def wrist_body_line(self) -> str:
        """One line saying which wrist cameras this robot's arm carries.

        Returns:
            str: ASCII, no newline.
        """
        if self.wrist_bodies is None:
            return "wrist cameras  not handed any camera"
        return self.wrist_bodies.line()

    def camera_world_line(self) -> str:
        """One line saying what world this robot's cameras give its arm.

        On an arm whose every motion needs a world or a decline and that holds none, the line says so: a robot before
        its first calibration moves only under a decline.

        Returns:
            str: ASCII, no newline.
        """
        needs = (": every planned motion needs a decline"
                 if bool(getattr(self.arm, "camera_world_required", False)) else "")
        if self.camera_world is None:
            return f"camera world  not handed any camera{needs}"
        if self.camera_world.world is None:
            return f"camera world  none: {self.camera_world.reason}{needs}"
        return "camera world  wired: " + ", ".join(repr(camera) for camera in self.camera_world.cameras)

    def safety(self) -> SafetyAttestation:
        """What this robot's arm will refuse, asked of the built arm. Commands nothing.

        Returns:
            SafetyAttestation: The guards the arm runs before every motion, and what each refuses; prints as itself.
        """
        return SafetyAttestation.of(self.arm)

    def route(self) -> "_motion.RouteReading":
        """How this robot's motions reach its arm, read off the built arm. Commands nothing, starts no planner.

        Returns:
            RouteReading: ``PLANNED`` (cuRobo plans, the exact mesh guard judges), ``UNPLANNED`` (a desk arm) or
                ``REFUSED`` (an arm whose motions would not pass both), and why.
        """
        return _motion.route_of(self.arm)

    def preflight(self) -> "PreflightReport":
        """The real cell's desk checklist for the tree this robot was built from. Touches no hardware.

        The robot half comes from the tree the robot keeps, else from the arm's ``arm.config``; the camera half and the
        root only from a tree :meth:`from_tree` read. A robot that keeps no tree gets one BLOCK row that says so.

        Returns:
            PreflightReport: One row per check, ``OK``, ``WARN`` or ``BLOCK``, and an ``exit_code``; prints as itself.
        """
        from src.config.schema.robot import RobotConfig
        from src.robot.execution.real_cell.preflight import (
            CheckStatus,
            PreflightCheck,
            PreflightReport,
            run_config_preflight,
        )

        config = self.robot_config if self.robot_config is not None else getattr(self.arm, "config", None)
        if not isinstance(config, RobotConfig):
            return PreflightReport(checks=(PreflightCheck(
                "config", CheckStatus.BLOCK,
                f"this robot keeps no config tree: its arm ({type(self.arm).__name__}) was handed in without one",
                fix="build it with Robot.from_tree(...) or Robot.from_config(...), or pass robot_config= to "
                    "Robot.from_parts",
            ),))
        if self.tree is None:
            return run_config_preflight(config)
        return run_config_preflight(config, camera=self.tree.app_config.camera, data_dir=self.tree.root)

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """Describe this robot to a person: arm, hand, lock, safety, planner route, camera world, wrist cameras and
        tree, one line each, read off the built arm as it is now. Nothing connects or moves.

        Returns:
            str: ASCII, no trailing newline. ``print(robot)`` shows the same.
        """
        facts = self._facts()
        lines = [
            f"robot  {facts['arm']} ({facts['vendor']}, {facts['model']})",
            f"gripper  {facts['gripper'] or 'none: an arm-only robot'}",
            f"lock  {facts['lock_key'] or 'none: this robot takes no lock'}",
            f"safety  {facts['safety'].upper()}",
            f"planner  {self.route().render()}",
            self.camera_world_line(),
            self.wrist_body_line(),
        ]
        tree = facts["tree"]
        if tree is None:
            lines.append("tree  none kept: " + ("built from a robot section" if self.robot_config is not None
                                               else "built around handles"))
        else:
            lines.append(f"tree  {tree['chain']} under {tree['root']}")
        return "\n".join(line.encode("ascii", "backslashreplace").decode("ascii") for line in lines)

    def to_dict(self) -> dict[str, Any]:
        """The readings :meth:`render` prints, as plain data.

        Returns:
            dict[str, Any]: ``json.dumps`` safe.
        """
        facts = self._facts()
        facts["route"] = self.route().to_dict()
        facts["camera_world"] = {
            "needs_decline": bool(getattr(self.arm, "camera_world_required", False)),
            "wiring": None if self.camera_world is None else self.camera_world.to_dict(),
        }
        facts["wrist_bodies"] = None if self.wrist_bodies is None else self.wrist_bodies.to_dict()
        return facts

    def _facts(self) -> dict[str, Any]:
        """The plain readings both halves share: names, lock, safety posture and the tree."""
        capabilities = getattr(self.arm, "capabilities", None)
        if isinstance(capabilities, RobotCapabilities):
            vendor, model = str(capabilities.vendor), str(capabilities.model)
        else:
            vendor = model = "no capabilities"
        return {
            "arm": type(self.arm).__name__,
            "vendor": vendor,
            "model": model,
            "gripper": None if self.gripper is None else type(self.gripper).__name__,
            "lock_key": self.lock_key,
            "safety": self.safety().posture.value,
            "tree": None if self.tree is None else {"chain": self.tree.chain, "root": str(self.tree.root)},
        }

    # ---- poses through the cell --------------------------------------------------------------

    def tool_down(
        self, x_mm: float, y_mm: float, z_mm: float, *, yaw_deg: float = 0.0,
        closing_axis: "Maybe[ClosingAxisLike]" = UNSET,
    ) -> "Pose":
        """A BASE pose with the tool pointing straight down, its jaws closing the way this cell's hand and camera
        naturally stand. Commands nothing.

        ```python
        above = robot.tool_down(450.0, 100.0, 300.0)              # along robot.natural_closing_axis
        part = robot.tool_down(450.0, 100.0, 120.0, yaw_deg=90.0)  # a quarter turn further about the vertical
        ```

        Args:
            x_mm (float): The tool's x in the robot's base frame, in millimetres.
            y_mm (float): The tool's y in the base frame, in millimetres.
            z_mm (float): The tool's z in the base frame, in millimetres.
            yaw_deg (float): A further turn of the closing axis about the vertical, in degrees, counted from that axis;
                on a cell whose hand stands along ``-y``, ``90.0`` closes along base x (default: 0.0).
            closing_axis (Maybe[ClosingAxisLike]): Which way the jaws close: a name ``Pose.tool_down`` takes (``"x"``,
                ``"-y"``, ...), a quaternion ``(x, y, z, w)`` or a BASE ``Pose`` whose tool +X laid onto the base XY
                plane is the direction. Unset is the cell's ``robot.natural_closing_axis``, and ``"x"`` where it names
                none (default: UNSET).

        Returns:
            Pose: The pose in ``Frame.BASE``, for :meth:`move`, :meth:`pick` or :meth:`place`.

        Raises:
            ValueError: ``closing_axis`` names no axis.
            TypeError: ``closing_axis`` is of another type, ``None`` included.
        """
        from src.geometry import Pose  # noqa: PLC0415
        from src.geometry.closing_axis import closing_axis_of, natural_closing_axis_of  # noqa: PLC0415

        config = self.robot_config if self.robot_config is not None else getattr(self.arm, "config", None)
        axis = closing_axis_of(closing_axis) if chosen(closing_axis) else natural_closing_axis_of(config)
        if axis is not None and axis.heading_deg is not None:
            return Pose.tool_down(x_mm, y_mm, z_mm, yaw_deg=axis.heading_deg + float(yaw_deg))
        return Pose.tool_down(x_mm, y_mm, z_mm, yaw_deg=yaw_deg, closing_axis=axis.name if axis is not None else "x")

    # ---- the verbs that move the arm ----------------------------------------------------------

    def move(
        self, pose: "Pose", *, decline: "Maybe[str]" = UNSET, linear: bool = False,
        vel: "Maybe[float]" = UNSET, acc: "Maybe[float]" = UNSET,
    ) -> "_motion.MotionReport":
        """Move the arm to a pose, planned around what the cameras see, or as a straight line.

        Args:
            pose (Pose): The target, in ``Frame.BASE`` millimetres; :meth:`tool_down` builds one.
            decline (Maybe[str]): Why this motion needs no camera world, such as ``"bench, no camera mounted"``; unset
                asks for the world the cameras built (default: UNSET).
            linear (bool): Drive a straight line in Cartesian space instead of a planned move (default: False).
            vel (Maybe[float]): The speed handed to the arm driver's move, in the driver's own units; unset keeps the
                driver's default speed (default: UNSET).
            acc (Maybe[float]): The acceleration, likewise (default: UNSET).

        Returns:
            MotionReport: What was asked, how the arm's motions reach it, and what the arm said; ``outcome`` is
                ``EXECUTED``, ``MOTION_REFUSED``, ``CAMERA_WORLD_UNAVAILABLE`` or ``REFUSED``. Prints as itself.

        The move is refused before any command for a closed link, a pose not in BASE, a camera world the arm would
        refuse, and an arm whose motions do not go through cuRobo and the exact mesh guard (a UR on the ik planner, a
        KUKA); a desk arm runs and says UNPLANNED. A refusal is an outcome, not an exception.
        """
        return _motion.move(self.arm, pose, decline=_motion.decline_of(decline), linear=linear, vel=vel, acc=acc)

    def move_joints(
        self, joints: "JointPositions | Sequence[float]", *, decline: "Maybe[str]" = UNSET,
    ) -> "_motion.MotionReport":
        """Move the arm to a joint configuration, refused as :meth:`move` is.

        Args:
            joints (JointPositions | Sequence[float]): The target joints in radians, base to wrist;
                ``JointPositions.deg(...)`` takes degrees as the pendant shows them.
            decline (Maybe[str]): Why this motion needs no camera world (default: UNSET).

        Returns:
            MotionReport: What was asked and what the arm said; prints as itself.
        """
        target = joints if isinstance(joints, JointPositions) else JointPositions(joints)
        return _motion.move_joints(self.arm, target, decline=_motion.decline_of(decline))

    def home(self, *, decline: "Maybe[str]" = UNSET) -> "_motion.MotionReport":
        """Move the arm to its configured home through its own gated home verb, refused as :meth:`move` is.

        Args:
            decline (Maybe[str]): Why this motion needs no camera world (default: UNSET).

        Returns:
            MotionReport: What the arm said. An arm that goes home as a typed verb (the UR driver) reports the status
                and the sentence of a gate that refused it; any other arm answers with a bool, so a refused home reads
                UNKNOWN and its log names the gate.
        """
        return _motion.home(self.arm, decline=_motion.decline_of(decline))

    def grasp(self, width_mm: float) -> "_handling.HandReport":
        """Close the hand to a width, read what it measured, and hand a held part to the arm's model.

        Args:
            width_mm (float): How far the jaws close, in millimetres. No force is commanded.

        Returns:
            HandReport: What was commanded and measured (``HoldEvidence``), and what it did to the carried part model;
                ``outcome`` is ``GRASPED``, ``NOTHING_HELD``, ... Prints as itself.

        Commands nothing on a robot with no hand, a hand that holds nothing, or a link that is not open. A hold the hand
        measured, or that nothing could measure, is modelled as a carried part where the arm models one; a measured
        empty close is not. A hand that raises is stopped once where it can be.
        """
        return _handling.grasp(self, width_mm)

    def release(self) -> "_handling.HandReport":
        """Open the hand to its width and forget the carried part, unless the hand still measures one.

        Returns:
            HandReport: What was commanded and measured; ``RELEASE_NOT_CONFIRMED`` where the hand still holds.
        """
        return _handling.release(self)

    def pick(
        self, pose: Pose, width_mm: float, *, standoff_mm: float = 80.0, squeeze_mm: float = 1.0,
        pre_open_mm: "Maybe[float | None]" = UNSET, decline: "Maybe[str]" = UNSET,
        camera_world: "Maybe[CameraWorldDecline]" = UNSET, keep_out: "Maybe[SegmentationOffer]" = UNSET,
    ) -> "_handling.HandlingReport":
        """Pick the part at a known pose: open, a planned move to the standoff, a line in, close, a line out.

        Args:
            pose (Pose): Where the tool closes, in ``Frame.BASE``, the approach along its +Z (straight down from
                :meth:`tool_down`).
            width_mm (float): The part's width across the jaws, in millimetres.
            standoff_mm (float): How far back along the approach the planned move stops before the line in, in
                millimetres (default: 80.0).
            squeeze_mm (float): How much narrower than ``width_mm`` the jaws close, in millimetres (default: 1.0).
            pre_open_mm (Maybe[float | None]): How wide to open before the approach; unset opens to the hand's width,
                ``None`` does not open (default: UNSET).
            decline (Maybe[str]): Why these motions need no camera world (default: UNSET).
            camera_world (Maybe[CameraWorldDecline]): The same decline as an object, kept as an alias; passing both is a
                ``TypeError`` (default: UNSET).
            keep_out (Maybe[SegmentationOffer]): What the part is, held out of the arm's live world through every
                motion, so the line in may come close to it: ``located.keep_out(i)`` (default: UNSET).

        Returns:
            HandlingReport: What was commanded, what stood behind each motion, and what the hand did; ``outcome`` is
                ``SUCCEEDED``, ``NOTHING_HELD`` (the jaws closed on nothing, the arm opened and backed out), a refusal,
                ... Prints as itself.

        Raises:
            TypeError: Both ``decline`` and ``camera_world`` are given.

        Refused before any command, as an outcome: a robot that cannot hold, a closed link, a pose not in BASE, a camera
        world the motion would be refused for, and an arm whose motions do not go through cuRobo and the exact mesh
        guard. A hand that toggles with no sensor (a ``jaw_io`` single_toggle) is never switched before the arm moves:
        it is asked whether its jaws stand open, asks a person where it believes them closed, and closes with one change
        of its output at the part. The frames a wrist camera's ``Locator.look_around`` held stay in the arm's world
        through the pick and are let go when it ends, however it ends.
        """
        try:
            return _handling.pick(self, pose, width_mm, standoff_mm=standoff_mm, squeeze_mm=squeeze_mm,
                                  pre_open_mm=pre_open_mm, camera_world=_motion.decline_of(decline, camera_world),
                                  keep_out=keep_out)
        finally:
            let_go_of_held_views(self.arm)

    def place(
        self, pose: Pose, *, standoff_mm: float = 80.0, decline: "Maybe[str]" = UNSET,
        camera_world: "Maybe[CameraWorldDecline]" = UNSET, keep_out: "Maybe[SegmentationOffer]" = UNSET,
    ) -> "_handling.HandlingReport":
        """Place the held part at a pose: a planned move to the standoff, a line in, release, a line out.

        ```python
        set_down = located.set_down(0, grasp=best.pose(), part_bottom_mm=scene.part_bottom_mm)
        if set_down.pose is not None:
            robot.place(set_down.pose, keep_out=located.keep_out(0))
        ```

        Args:
            pose (Pose): Where the tool releases, in ``Frame.BASE``, the approach along its +Z.
            standoff_mm (float): How far back along the approach the planned move stops, in millimetres (default: 80.0).
            decline (Maybe[str]): Why these motions need no camera world (default: UNSET).
            camera_world (Maybe[CameraWorldDecline]): The same decline as an object; passing both is a ``TypeError``
                (default: UNSET).
            keep_out (Maybe[SegmentationOffer]): What the part is set down on, held out of the arm's live world from
                before the first motion to after the last: a part set down onto something a camera located comes within
                the line clearance of it, and the world would refuse the line in (default: UNSET).

        Returns:
            HandlingReport: What was commanded and what the hand did. A release the hand does not confirm leaves the arm
                where it stands.

        Raises:
            TypeError: Both ``decline`` and ``camera_world`` are given.
        """
        return _handling.place(self, pose, standoff_mm=standoff_mm,
                               camera_world=_motion.decline_of(decline, camera_world), keep_out=keep_out)

    def is_holding(self) -> HoldEvidence:
        """What the hand measured about a hold, commanding nothing.

        Returns:
            HoldEvidence: ``HELD``, ``EMPTY``, or ``UNMEASURED`` where nothing can say.
        """
        return _handling.is_holding(self)

    def without_camera_world(self, reason: str) -> AbstractContextManager[CameraWorldDecline]:
        """Decline the camera world for every motion of this robot's arm inside a ``with`` block.

        ```python
        with robot.without_camera_world("bench check, no cameras mounted"):
            robot.move(pose)
            robot.home()
        ```

        Args:
            reason (str): Why these motions need no camera world; it goes into every report and log line.

        Returns:
            AbstractContextManager[CameraWorldDecline]: The block. A ``decline=`` on a verb inside it beats the block;
                it does not follow into a thread started inside it.

        Raises:
            ValueError: ``reason`` is blank.

        Bound to this robot's arm, so another robot in the same process plans as it would without it. A motion no
        planner plans or checks still says UNPLANNED, and on an arm whose live camera world is wired a declined planned
        or checked motion is refused.
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


def _tree_wrist_bodies(
    tree: "LoadedTree", robot_config: "RobotConfig", cameras: "Maybe[Sequence[Any]]", reason: "Maybe[str]",
) -> "WristBodies":
    """Every wrist body ``tree`` declares, and those of handed cameras whose rig it does not hold, not yet handed over.

    Resolved from the camera section, so a body is carried whether its camera is open or not. A body that cannot be
    placed yet is refused naming ``unmodelled_wrist_body``, unless ``reason`` says why the arm may move without it.
    """
    from src.robot.execution.wrist_bodies import WristBodies, WristBodyUnplaced

    camera = getattr(tree.app_config, "camera", None)
    try:
        wrist = WristBodies.from_config(robot_config, camera, data_dir=tree.root, unmodelled_reason=reason)
    except WristBodyUnplaced as exc:
        raise WristBodyUnplaced(
            f"{str(exc).rstrip('.')}. To build this robot before the body can be placed, say why it may move "
            'without it: Robot.from_tree(tree, unmodelled_wrist_body="<reason>"); it then moves with no camera '
            "body in the planner, the exact guard or the self filter") from exc
    if not chosen(cameras) or cameras is None:
        return wrist
    declared = {str(rig.rig_id) for rig in getattr(getattr(camera, "cameras", None), "rigs", None) or ()}
    others = [owner for owner in cameras if str(owner.rig_id) not in declared]
    if not others:
        return wrist
    handed = WristBodies.from_owners(robot_config, others)
    return replace(wrist, bodies=wrist.bodies + handed.bodies)


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
