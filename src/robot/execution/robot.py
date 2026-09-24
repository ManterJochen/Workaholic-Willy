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
report and raises only for a programmer's error.

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
from src.robot.execution.lifecycle import ConnectedRobot, ConnectStage
from src.robot.execution.robot_parts import build_gripper, resolve_arm
from src.robot.safety import SafetyAttestation

if TYPE_CHECKING:
    from src.config.schema.robot import RobotConfig
    from src.config.tree import LoadedTree
    from src.geometry import Pose
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
    """An arm and the gripper on it, built and not yet connected.

        from src.config import ConfigTree
        from src.robot.execution import Robot

        robot = Robot.from_tree(ConfigTree.from_directory(profile="console_dummy").load())
        print(robot.render())                   # arm, gripper, lock, planner route, camera world
        print(robot.preflight().render())       # the desk checklist for the tree it came from
        with robot.connected() as live:         # lock, arm, then gripper
            print(robot.home().render())
            print(robot.move(pose, decline="bench, no camera mounted").render())
            print(robot.grasp(40.0).render())   # close, read the hold, model the part
            print(robot.release().render())
        print(live.teardown.render())

    ``gripper`` is ``None`` for an arm-only robot. ``lock_key`` names the controller the cross-process
    lock is taken on, and is ``None`` for a robot that takes no lock. ``camera_world`` is what the
    cameras handed in built, ``None`` for a robot handed no camera. ``wrist_bodies`` are the camera
    bodies the arm carries: every one its tree declares for a robot ``from_tree`` built, camera open or
    not, and those of the cameras handed in otherwise. ``robot_config`` is the tree a factory was handed
    and ``tree`` the loaded tree ``from_tree`` read; both are ``None`` where the robot was built around
    handles alone.
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

            robot = Robot.from_tree(ConfigTree.from_directory(profile="console_dummy").load())

        The robot section is built as :meth:`from_config` builds it, and the robot keeps the tree, so
        :meth:`preflight` reads the camera half and the root from the same load. A tree that did not load
        is refused with its own refusal (``ConfigError``); anything but a ``LoadedTree`` is a ``TypeError``.

        Every wrist camera body the tree's camera section declares is handed to the arm, whether or not
        its camera is handed in or even enabled: a camera that is not open still hangs on the arm, and the
        planner, the exact guard and the self filter carry its housing (``execution.wrist_bodies``). A body
        that cannot be placed yet, because its camera is not calibrated, is refused with
        :class:`~src.robot.execution.wrist_bodies.WristBodyUnplaced`, unless ``unmodelled_wrist_body`` says
        why this robot may move without it; the robot then names the camera it moves without. The reason
        is read only for such a body.
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
        """The arm and the gripper ``robot_config`` describes. Connects nothing.

        The arm-vendor readiness gate runs first, as it does for a pick service built from the same
        tree. ``gripper`` left unset builds the configured end-effector, and one that had to be
        substituted is refused at connect. ``gripper=None`` builds the arm alone: no activation
        sweep, and a gripper that could not be built does not stand in the way. ``cameras`` are passed
        through to :meth:`from_parts` with this tree, which the world's planning block is read from.

        A robot section holds no camera section, so the arm carries only the bodies of the cameras handed
        in. A robot that moves beside a wrist camera it was not handed is built with :meth:`from_tree`,
        which reads them all; the calibration sweep builds its arm here and hands it the body itself.
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

    def route(self) -> "_motion.RouteReading":
        """How this robot's motions reach its arm, read off the built arm: PLANNED, UNPLANNED or REFUSED.

        UNPLANNED is a desk arm. Commands nothing and starts no planner. Every verb that moves the arm reads
        it before its first command.
        """
        return _motion.route_of(self.arm)

    def preflight(self) -> "PreflightReport":
        """The real cell's desk checklist for the tree this robot was built from. Touches no hardware.

        The robot half comes from the tree this robot keeps (``from_tree``, ``from_config``, ``robot_config`` on
        ``from_parts``), else from the arm's own ``arm.config``. The camera half and the root come only from a tree
        ``from_tree`` read, so a robot built from a robot section alone says its camera row was handed nothing rather
        than reading another tree. A robot that keeps no tree at all gets one BLOCK row that says so.
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
        """Describe this robot to a person, as text.

        Arm, gripper, lock, safety, planner route, camera world, wrist cameras and tree, one line each.
        ASCII, no trailing newline. Every line is read off the built arm as it is now; nothing connects or moves.
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
        """Plain data, ``json.dumps`` safe: the same readings ``render()`` prints."""
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

    # ---- the verbs that move the arm ----------------------------------------------------------

    def move(
        self, pose: "Pose", *, decline: "Maybe[str]" = UNSET, linear: bool = False,
        vel: "Maybe[float]" = UNSET, acc: "Maybe[float]" = UNSET,
    ) -> "_motion.MotionReport":
        """Move the arm to ``pose`` (BASE), as a straight line when ``linear``, through the arm's own typed ``move``.

        Refuses before any command a closed link, a pose not in BASE, a camera world the arm would refuse, and an arm
        whose motions do not go through cuRobo and the exact mesh guard (a UR on the ik planner, a KUKA). A desk arm
        runs and its report says UNPLANNED. ``decline`` is the reason this motion needs no camera world. ``vel`` and
        ``acc`` reach the arm only when chosen. The report carries the arm's result and, for a line, what the arm keeps
        of it.
        """
        return _motion.move(self.arm, pose, decline=_motion.decline_of(decline), linear=linear, vel=vel, acc=acc)

    def move_joints(
        self, joints: "JointPositions | Sequence[float]", *, decline: "Maybe[str]" = UNSET,
    ) -> "_motion.MotionReport":
        """Move the arm to ``joints`` (radians) through its own typed ``move_to_joints``.

        The same refusals as :meth:`move`.
        """
        target = joints if isinstance(joints, JointPositions) else JointPositions(joints)
        return _motion.move_joints(self.arm, target, decline=_motion.decline_of(decline))

    def home(self, *, decline: "Maybe[str]" = UNSET) -> "_motion.MotionReport":
        """Move the arm to its configured home through its own gated home verb, refused as :meth:`move` is.

        An arm that goes home as a typed verb (the UR driver's ``move_to_home``) reports the status and the sentence of
        the gate that refused it. Any other arm answers ``move_home`` with a bool, so a home it refuses reads UNKNOWN
        and its log names the gate.
        """
        return _motion.home(self.arm, decline=_motion.decline_of(decline))

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
        pre_open_mm: "Maybe[float | None]" = UNSET, decline: "Maybe[str]" = UNSET,
        camera_world: "Maybe[CameraWorldDecline]" = UNSET, keep_out: "Maybe[SegmentationOffer]" = UNSET,
    ) -> "_handling.HandlingReport":
        """Pick the part at ``pose`` (BASE, approach along its +Z), ``width_mm`` across.

        Refuses before any command a robot that cannot hold, a closed link, a pose not in BASE, a camera
        world this arm's motion would be refused for, and an arm whose motions do not go through cuRobo
        and the exact mesh guard (a desk arm runs). Then it detaches, opens to ``pre_open_mm`` (the
        hand's width when unset, not at all when ``None``), makes a planned move to the standoff
        ``standoff_mm`` back along the approach, drives a line to ``pose``, runs :meth:`grasp` at
        ``width_mm`` minus ``squeeze_mm``, and drives a line back to the standoff. A close that measures
        nothing opens again and backs out. ``keep_out`` is held in the arm's live world through every
        motion. A refused motion, and a camera that could not vouch for the cell, end the pick with
        nothing commanded after it, as an outcome. A hand that toggles with no sensor (a ``jaw_io``
        single_toggle) is never pulsed before the arm moves, whatever ``pre_open_mm`` says: it is asked
        whether its jaws stand open, asks a person where it believes them closed, and is closed with
        one pulse at the part.

        ``decline`` is the reason these motions need no camera world. ``camera_world=CameraWorldDecline(...)``
        is the same decline as an object, kept as an alias; passing both is a ``TypeError``.
        """
        return _handling.pick(self, pose, width_mm, standoff_mm=standoff_mm, squeeze_mm=squeeze_mm,
                              pre_open_mm=pre_open_mm, camera_world=_motion.decline_of(decline, camera_world),
                              keep_out=keep_out)

    def place(
        self, pose: Pose, *, standoff_mm: float = 80.0, decline: "Maybe[str]" = UNSET,
        camera_world: "Maybe[CameraWorldDecline]" = UNSET,
    ) -> "_handling.HandlingReport":
        """Place the held part at ``pose``: a planned move to the standoff, a line in, :meth:`release`, a line out.

        The same refusals, the same outcomes and the same ``decline`` as :meth:`pick`. A release the
        gripper does not confirm leaves the arm where it stands.
        """
        return _handling.place(self, pose, standoff_mm=standoff_mm,
                               camera_world=_motion.decline_of(decline, camera_world))

    def is_holding(self) -> HoldEvidence:
        """What the gripper measured about a hold, commanding nothing; UNMEASURED where nothing can say."""
        return _handling.is_holding(self)

    def without_camera_world(self, reason: str) -> AbstractContextManager[CameraWorldDecline]:
        """Decline the camera world for every motion of this robot's arm inside the ``with`` block.

            with robot.without_camera_world("bench check, no cameras mounted"):
                robot.move(pose)
                robot.home()

        Bound to this robot's arm, so another robot in the same process plans as it would without it,
        and a blank reason is refused here. A ``decline=`` on a verb inside the block beats the block. A
        motion no planner plans or checks still says UNPLANNED, and on an arm whose live camera world is
        wired a declined planned or checked motion is refused. An arm that does not stamp its motions
        ignores the block. The block does not follow into a thread started inside it.
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
