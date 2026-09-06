"""The KUKA :class:`RobotArm`, the production driver over EthernetKRL.

It bridges the vendor-neutral :class:`src.robot.core.RobotArm` Protocol to the KUKA
controller through :class:`EkiClient`, over TCP with XML. Nothing outside
:mod:`src.robot.drivers.kuka` knows about KRL conventions, and a pipeline or the API
layer talks to ``RobotArm`` alone.

Numerics
--------
* Every :class:`Pose` argument and return is millimetres with an XYZW quaternion in
  ``frame=Frame.BASE``.
* Joint vectors on the wire are degrees, which is the KUKA convention, and the driver
  converts at the boundary so a vendor-neutral :class:`JointPositions` always carries
  radians.

Capabilities
------------
* Joint moves, Cartesian LIN and PTP moves, and the native FK and IK the controller KRL
  utilities provide, round-tripped through EKI.
* There is no native asynchronous API, so ``supports_async_move=False``.
* There is no native force control.

The controller side is a small KRL program plus an ``EkiHwInterface`` XML config, whose
templates live under ``config/robot/templates/kuka/``.
"""

from __future__ import annotations

from src.config.schema.robot import RobotConfig
from src.geometry import Frame, FrameMismatchError, Pose
from src.robot.constants import HOME_JOINTS_DEFAULT, KUKA_ARM_LOG_FILE, create_robot_logger
from src.robot.core import (
    JointPositions,
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotArm,
    RobotCapabilities,
    RobotConnectionError,
    RobotKinematicsError,
    RobotMotionRejected,
)
from src.robot.safety.workspace import WorkspaceGuard
from src.robot.safety import SafetyPreflight

from .eki_client import EkiClient
from .pose_convert import (
    joints_deg_to_rad,
    joints_rad_to_deg,
    kuka_cartesian_to_pose,
    pose_to_kuka_cartesian,
)

__all__ = ["KUKA_CAPABILITIES", "KukaRobotArm"]


KUKA_CAPABILITIES = RobotCapabilities(
    vendor="kuka",
    model="kr6-r900",
    dof=6,
    supports_joint_move=True,
    supports_linear_move=True,
    supports_async_move=False,
    has_native_fk=True,
    has_native_ik=True,
    has_force_control=False,
    is_simulated=False,
)


class KukaRobotArm(RobotArm):
    """The production KUKA driver, speaking EKI and KRL over TCP with XML.

    Parameters
    ----------
    config
        The full :class:`RobotConfig` tree. ``config.kuka.eki`` selects the wire
        transport and ``config.workspace_limits`` drives the internal
        :class:`WorkspaceGuard`.
    home_joints
        An optional override for the home joint configuration, in radians. It defaults
        to :data:`src.robot.constants.HOME_JOINTS_DEFAULT`.
    eki_client
        An optional pre-constructed :class:`EkiClient`, which is the injection seam for
        the transport. With ``None`` the driver builds one from ``config.kuka.eki``.
    """

    def __init__(
        self,
        config: RobotConfig,
        *,
        home_joints: list[float] | None = None,
        eki_client: EkiClient | None = None,
    ) -> None:
        self.config = config
        self.logger = create_robot_logger("KukaRobotArm", KUKA_ARM_LOG_FILE)

        kuka_cfg = config.kuka
        eki_cfg = kuka_cfg.eki

        if eki_client is None:
            eki_client = EkiClient(
                role=eki_cfg.role,
                host=eki_cfg.host if eki_cfg.role == "server" else kuka_cfg.controller_ip,
                port=eki_cfg.port,
                timeout_s=eki_cfg.timeout_s,
                heartbeat_s=eki_cfg.heartbeat_s,
                buffer_size=eki_cfg.buffer_size,
            )
        self._eki = eki_client

        self._guard = WorkspaceGuard(config.workspace_limits)
        self._home_joints = list(home_joints or HOME_JOINTS_DEFAULT)
        # The vendor-neutral safety preflight pipeline. KUKA EKI exposes neither
        # runtime joint-limit telemetry nor runtime payload writes, so the joint-limit
        # and payload guards rely on the static fields in ``config.safety``, which
        # ``config/robot/robot.yaml`` carries.
        self._preflight = SafetyPreflight.from_safety_config(
            config.safety, config.workspace_limits,
        )

        # Resolve the capability descriptor from the configured model and DoF.
        self._capabilities = (
            KUKA_CAPABILITIES
            if (kuka_cfg.model == KUKA_CAPABILITIES.model and kuka_cfg.dof == 6)
            else RobotCapabilities(
                vendor="kuka",
                model=kuka_cfg.model,
                dof=kuka_cfg.dof,
                supports_joint_move=True,
                supports_linear_move=True,
                supports_async_move=False,
                has_native_fk=True,
                has_native_ik=True,
                has_force_control=False,
                is_simulated=False,
            )
        )

        # Surface the underlying transport so a caller can inspect it. There is
        # deliberately no `controller` or `transport` field mirrored from config: the
        # EKI framing, the timing and the KRL templates are generation-agnostic and the
        # `eki:` block is itself the transport declaration, so both would be write-only.
        # A real difference between a KRC4 and a KRC5 belongs in a capability descriptor
        # beside `model` rather than in a bare string.
        self.eki = eki_cfg

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def safety_preflight(self) -> "SafetyPreflight | None":
        """The guard pipeline every motion of this arm passes through.

        It implements :class:`~src.robot.safety.attestation.SafetyGated`, so a caller
        asks what this arm will refuse without reaching into `_preflight`. That matters
        for the library API: a caller supplies their own arm, which enumerating the
        driver registry cannot see, so the answer travels with the object.
        """
        return self._preflight

    @property
    def capabilities(self) -> RobotCapabilities:
        return self._capabilities

    @property
    def is_connected(self) -> bool:
        return self._eki.is_connected

    @property
    def guard(self) -> WorkspaceGuard:
        return self._guard

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the EKI link and wait for the first telemetry frame."""
        self._eki.connect()
        # Block briefly for the first ``<State>``, so a later ``get_tcp_pose`` or
        # ``get_joint_positions`` returns real data.
        if not self._eki.wait_for_first_state():
            self.logger.warning(
                "KUKA EKI: no telemetry within timeout; the link is up "
                "but the first <State> frame has not arrived yet.",
            )

    def disconnect(self) -> None:
        self._eki.disconnect()

    def __enter__(self) -> KukaRobotArm:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_tcp_pose(self) -> Pose:
        if not self._eki.is_connected:
            raise RobotConnectionError("KUKA driver: get_tcp_pose() requires an open EKI link.")
        snapshot = self._eki.get_state()
        if snapshot.pose is None:
            raise RobotConnectionError(
                "KUKA driver: no <State> telemetry has arrived yet."
            )
        return kuka_cartesian_to_pose(snapshot.pose, label="current")

    def get_joint_positions(self) -> JointPositions:
        if not self._eki.is_connected:
            raise RobotConnectionError(
                "KUKA driver: get_joint_positions() requires an open EKI link."
            )
        snapshot = self._eki.get_state()
        if snapshot.joints_deg is None:
            raise RobotConnectionError(
                "KUKA driver: no <State> telemetry has arrived yet."
            )
        return JointPositions(joints_deg_to_rad(snapshot.joints_deg))

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def move_to_joints(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> MotionResult:
        """The typed joint move: gate the destination through the preflight, then drive."""
        if self._preflight is not None:
            rejected = self._preflight.gate_joint_target(joints, arm=self)
            if rejected is not None:
                return rejected
        self.move_joint(joints, velocity=velocity, acceleration=acceleration)
        return MotionResult.executed(
            MotionCommand.MOVE_JOINTS, target_joints=joints, message="move_to_joints",
        )

    def move_joint(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        """A gated joint-space move over EKI.

        A KUKA cell has no built-in joint-limit table, so without a static ``min_deg``
        and ``max_deg`` in config the guard returns unavailable and this refuses. That
        is the documented fail-closed behaviour, and it is what :meth:`move_to_joints`
        does too.
        """
        if self._preflight is not None:
            rejected = self._preflight.gate_joint_target(joints, arm=self)
            if rejected is not None:
                raise RobotMotionRejected(
                    f"move_joint refused by the safety preflight: {rejected.message}"
                )
        self._drive_joints(joints, velocity=velocity, acceleration=acceleration)

    def _drive_joints(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        """Command the joints with no safety gate. Callers must have gated the destination first."""
        if not self._eki.is_connected:
            raise RobotConnectionError("KUKA driver: move_joint() requires an open EKI link.")
        if joints.dof != self._capabilities.dof:
            raise RobotMotionRejected(
                f"KUKA move_joint() expected {self._capabilities.dof} DoF, got {joints.dof}."
            )
        try:
            self._eki.send_movej(
                joints_rad_to_deg(joints.tolist()),
                vel=velocity,
                acc=acceleration,
            )
        except RobotConnectionError as exc:
            raise RobotMotionRejected(f"KUKA move_joint failed: {exc}") from exc

    def move_linear(
        self,
        pose: Pose,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"KUKA move_linear() requires Frame.BASE; got {pose.frame!r}."
            )
        if not self._eki.is_connected:
            raise RobotConnectionError(
                "KUKA driver: move_linear() requires an open EKI link."
            )
        try:
            self._eki.send_move_cartesian(
                pose_to_kuka_cartesian(pose),
                mode="LIN",
                vel=velocity,
                acc=acceleration,
            )
        except RobotConnectionError as exc:
            raise RobotMotionRejected(f"KUKA move_linear failed: {exc}") from exc

    def stop(self) -> None:
        """Send an immediate stop to the controller. Always safe."""
        if not self._eki.is_connected:
            return
        try:
            self._eki.send_stop()
        except RobotConnectionError as exc:
            self.logger.warning("KUKA stop() send failed: %s", exc)

    def wait_until_steady(
        self,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.02,
    ) -> bool:
        """Block until the cached telemetry reports ``Steady=1``, or until the timeout.

        ``poll_interval_s`` is accepted for Protocol compatibility and ignored, because
        the EKI client wakes the waiter on every incoming ``<State>`` frame.
        """
        return self._eki.wait_until_steady(timeout_s)

    # ------------------------------------------------------------------
    # Kinematics, round-tripped through the KRL forward and inverse solvers
    # ------------------------------------------------------------------

    def fk(self, joints: JointPositions) -> Pose:
        if joints.dof != self._capabilities.dof:
            raise RobotKinematicsError(
                f"KUKA fk() expected {self._capabilities.dof} DoF, got {joints.dof}."
            )
        if not self._eki.is_connected:
            raise RobotConnectionError("KUKA fk() requires an open EKI link.")
        try:
            cart = self._eki.request_fk(joints_rad_to_deg(joints.tolist()))
        except RobotConnectionError as exc:
            raise RobotKinematicsError(f"KUKA FK failed: {exc}") from exc
        return kuka_cartesian_to_pose(cart, label="fk")

    def ik(
        self,
        pose: Pose,
        *,
        seed: JointPositions | None = None,
    ) -> JointPositions:
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"KUKA ik() requires Frame.BASE; got {pose.frame!r}."
            )
        if not self._eki.is_connected:
            raise RobotConnectionError("KUKA ik() requires an open EKI link.")
        if seed is None:
            # Use the cached current joints as the IK seed where the caller gave none,
            # which is the best chance of nearest-solution selection on a multi-branch
            # IK such as the kr6-r900.
            try:
                seed_joints = self.get_joint_positions().tolist()
            except RobotConnectionError:
                seed_joints = list(self._home_joints)
        else:
            seed_joints = seed.tolist()
        seed_deg = joints_rad_to_deg(seed_joints)
        try:
            joints_deg = self._eki.request_ik(pose_to_kuka_cartesian(pose), seed_deg)
        except RobotConnectionError as exc:
            raise RobotKinematicsError(f"KUKA IK failed: {exc}") from exc
        joints_rad = joints_deg_to_rad(joints_deg)
        if len(joints_rad) != self._capabilities.dof:
            raise RobotKinematicsError(
                f"KUKA IK returned {len(joints_rad)} joints; expected {self._capabilities.dof}."
            )
        return JointPositions(joints_rad)

    # ------------------------------------------------------------------
    # High-level pipeline surface
    # ------------------------------------------------------------------

    def is_inside_workspace(self, pose: Pose) -> bool:
        if not isinstance(pose, Pose):
            raise TypeError(
                f"KUKA is_inside_workspace expects Pose; got {type(pose).__name__}."
            )
        return self._guard.is_inside_workspace(pose)

    def move_to(
        self,
        pose: Pose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> bool:
        """The vendor-neutral go-there, gated through the full preflight pipeline.

        It returns ``True`` on a controller ack and ``False`` where a guard refused or
        the driver could not reach the controller. It runs the whole pipeline and not
        the workspace box alone, because a surface that ran the box alone would command
        motion no joint-limit, IK-quality, self-collision, fixture, payload or
        continuity guard ever saw, on a cell where :meth:`move` runs all six.
        """
        if not self._eki.is_connected:
            self.logger.error("KUKA move_to: link not connected.")
            return False
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"KUKA move_to() requires Frame.BASE; got {pose.frame!r}."
            )
        rejected = self._gate_pose(pose)
        if rejected is not None:
            self.logger.error(
                "KUKA move_to REFUSED by %s: %s", rejected.status, rejected.message,
            )
            return False
        return self._drive_pose(pose, linear=linear, vel=vel, acc=acc, register=register)

    def _gate_pose(self, pose: Pose) -> "MotionResult | None":
        """The full Cartesian pipeline for one commanded pose. ``None`` means every guard passed.

        IK is pre-resolved for the reason :meth:`move` pre-resolves it: without
        ``target_joints`` the joint-limit, IK-quality and arm-against-arm guards have
        nothing to read and fail closed on every Cartesian move.
        """
        if self._preflight is None:
            return None
        target_joints: JointPositions | None = None
        current_joints: JointPositions | None = None
        try:
            target_joints = self.ik(pose)
        except (RobotKinematicsError, RobotConnectionError) as exc:
            status = (MotionStatus.IK_FAILED if isinstance(exc, RobotKinematicsError)
                      else MotionStatus.CONNECTION_ERROR)
            return MotionResult.failed(
                status, MotionCommand.MOVE_TO, target_pose=pose,
                message=str(exc), exception=exc,
            )
        try:
            current_joints = self.get_joint_positions()
        except RobotConnectionError:
            current_joints = None
        ctx = self._preflight.context_for_pose(
            pose, command=MotionCommand.MOVE_TO, target_joints=target_joints,
            current_joints=current_joints, arm=self,
        )
        return SafetyPreflight.as_motion_result(
            self._preflight.evaluate(ctx), MotionCommand.MOVE_TO, target_pose=pose,
        )

    def _drive_pose(
        self,
        pose: Pose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> bool:
        """Command a Cartesian pose with no safety gate. Callers must have gated it first."""
        if not self._eki.is_connected:
            self.logger.error("KUKA move_to: link not connected.")
            return False
        # The workspace box alone, deliberately not `validate()`, which also runs the
        # pose-diversity check. That check, refusing a pose too similar to one already
        # accepted, belongs to calibration pose sampling, and that is where it lives:
        # `execution/pose_provider.py` builds its own local guard for it. Wired into the
        # general motion path it rejects the ordinary shape of a pick. Measured: the
        # standoff is accepted, the descend is accepted, and the retreat is rejected as
        # too similar to the standoff at 0.0 mm and 0.0 deg. The arm would grasp and
        # refuse to lift on every pick, and the symptom reads as a gripper, planner or
        # calibration fault rather than a sampling rule.
        if not self._guard.is_inside_workspace(pose):
            self.logger.warning(
                "KUKA pose '%s' rejected by workspace guard.",
                pose.label or "<unlabeled>",
            )
            return False

        target = pose_to_kuka_cartesian(pose)
        try:
            self._eki.send_move_cartesian(
                target,
                mode="LIN" if linear else "PTP",
                vel=vel,
                acc=acc,
            )
        except RobotConnectionError as exc:
            self.logger.error("KUKA move_to '%s' failed: %s", pose.label, exc)
            return False
        if register:
            self._guard.accept(pose)
        return True

    def move_home(self) -> bool:
        """Move to the home joint configuration, gated like every other joint move.

        The home configuration is a place the arm will sit, so it gets at least the
        destination guards a commanded joint move gets rather than going straight down
        the EKI link unchecked.
        """
        if not self._eki.is_connected:
            self.logger.error("KUKA move_home: link not connected.")
            return False
        if self._preflight is not None:
            home = JointPositions(tuple(float(v) for v in self._home_joints))
            rejected = self._preflight.gate_joint_target(home, arm=self)
            if rejected is not None:
                self.logger.error(
                    "KUKA move_home REFUSED by %s: %s", rejected.status, rejected.message,
                )
                return False
        try:
            self._eki.send_movej(joints_rad_to_deg(self._home_joints))
        except RobotConnectionError as exc:
            self.logger.error("KUKA move_home failed: %s", exc)
            return False
        return True

    def move(
        self,
        pose: Pose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> MotionResult:
        """The typed counterpart of :meth:`move_to`.

        It routes every pose through the vendor-neutral :class:`SafetyPreflight`
        pipeline, so a rejection surfaces the precise :class:`MotionStatus` rather than
        a blanket ``WORKSPACE_REJECTED``. A connection fault comes back as
        :attr:`MotionStatus.CONNECTION_ERROR` rather than ``False``.
        """
        if pose.frame is not Frame.BASE:
            return MotionResult.failed(
                MotionStatus.INVALID_TARGET,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=f"KUKA move requires Frame.BASE; got {pose.frame!r}",
            )
        if not self._eki.is_connected:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message="KUKA EKI link not connected",
            )
        # IK is pre-resolved on the driver side, so the preflight pipeline sees
        # ``target_joints`` for every Cartesian command. The round-trip goes through
        # EKI, and a failure is reported as IK_FAILED rather than routed through the
        # preflight.
        target_joints: JointPositions | None = None
        current_joints: JointPositions | None = None
        try:
            target_joints = self.ik(pose)
        except RobotKinematicsError as exc:
            return MotionResult.failed(
                MotionStatus.IK_FAILED,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=str(exc),
                exception=exc,
            )
        except RobotConnectionError as exc:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=str(exc),
                exception=exc,
            )
        try:
            current_joints = self.get_joint_positions()
        except RobotConnectionError:
            current_joints = None
        ctx = self._preflight.context_for_pose(
            pose,
            command=MotionCommand.MOVE_TO,
            target_joints=target_joints,
            current_joints=current_joints,
            arm=self,
        )
        decision = self._preflight.evaluate(ctx)
        rejected = SafetyPreflight.as_motion_result(
            decision, MotionCommand.MOVE_TO, target_pose=pose,
        )
        if rejected is not None:
            return rejected
        try:
            # The ungated primitive. The pipeline above has already judged this pose.
            ok = self._drive_pose(
                pose, linear=linear, vel=vel, acc=acc, register=register,
            )
        except RobotConnectionError as exc:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=str(exc),
                exception=exc,
            )
        return MotionResult.from_bool(
            ok, MotionCommand.MOVE_TO, target_pose=pose,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def eki_client(self) -> EkiClient:
        """Underlying transport (mostly useful in tests)."""
        return self._eki

    @property
    def home_joints(self) -> list[float]:
        return list(self._home_joints)
