""":class:`RobotArm` is the vendor-neutral abstract surface for a manipulator.

Every driver in :mod:`src.robot.drivers` (UR, KUKA, Franka, ROS 2, sim, dummy)
implements this Protocol. Pipelines and planners depend on this interface alone and
never import a driver class directly.

Numerics contract
-----------------
* Every :class:`Pose` argument and return is in millimetres with a canonical XYZW
  quaternion, frame-tagged. ``get_tcp_pose()`` returns a Pose in :attr:`Frame.BASE`.
* Joint positions are :class:`JointPositions`, in radians and ``float64``.
* Velocity and acceleration follow the driver's own convention and pass through
  unchanged. UR uses rad/s and rad/s^2 for a joint move, m/s and m/s^2 for a
  Cartesian one. A planner that needs the units reads
  :attr:`RobotArm.capabilities`.

The Protocol is :func:`runtime_checkable`, so a candidate driver is verified with
``isinstance(driver, RobotArm)`` without nominal subclassing. A real driver still
inherits from the Protocol explicitly, which is what gives it static-type help.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.geometry import Pose

from .capabilities import RobotCapabilities
from .joint_positions import JointPositions
from .motion_result import MotionResult

__all__ = ["RobotArm"]


@runtime_checkable
class RobotArm(Protocol):
    """Vendor-neutral arm interface."""

    # ---- introspection --------------------------------------------------

    @property
    def capabilities(self) -> RobotCapabilities:
        """Static feature flags for this driver."""
        ...

    @property
    def is_connected(self) -> bool:
        """``True`` while a live link to the controller is open."""
        ...

    # ---- lifecycle ------------------------------------------------------

    def connect(self) -> None:
        """Open the link to the controller. Idempotent."""
        ...

    def disconnect(self) -> None:
        """Close the link. Idempotent, and safe to call repeatedly."""
        ...

    # ---- state ----------------------------------------------------------

    def get_tcp_pose(self) -> Pose:
        """Current TCP pose, tagged ``Frame.BASE``."""
        ...

    def get_joint_positions(self) -> JointPositions:
        """Current joint configuration."""
        ...

    # ---- motion ---------------------------------------------------------

    def move_joint(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        """Joint-space move to ``joints``.

        Raises
        ------
        RobotMotionRejected
            If a workspace or safety pre-check denies the move.
        RobotConnectionError
            If the link is not open.
        """
        ...

    def move_to_joints(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> MotionResult:
        """Typed joint-space move, the fail-closed counterpart of :meth:`move_joint`.

        The commanded joints go through the ``SafetyPreflight`` destination guards
        (joint limit, self-collision, payload, and static IK quality including
        singularity) and are then driven. A commanded joint move to a park, home or
        scan pose is a deliberate trajectory restart, so the step-size checks
        (motion continuity and the IK-jump check) and the pose-only workspace guard
        are exempted, and the continuity reference is reset around it. A guard
        rejection comes back as a typed :class:`MotionResult` carrying the matching
        :class:`MotionStatus`, where the void :meth:`move_joint` raises. A driver
        with no preflight wired drives and returns ``EXECUTED``.
        """
        ...

    def move_linear(
        self,
        pose: Pose,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        """Cartesian move to ``pose``, which must be in :attr:`Frame.BASE`.

        Raises
        ------
        FrameMismatchError
            If ``pose.frame`` is not :attr:`Frame.BASE`.
        RobotMotionRejected
            If a workspace or safety pre-check denies the move.
        RobotKinematicsError
            If no IK solution exists.
        """
        ...

    def stop(self) -> None:
        """Emergency stop. Always safe to call."""
        ...

    # ---- kinematics -----------------------------------------------------

    def fk(self, joints: JointPositions) -> Pose:
        """Forward kinematics, returning a TCP :class:`Pose` in :attr:`Frame.BASE`."""
        ...

    def ik(
        self,
        pose: Pose,
        *,
        seed: JointPositions | None = None,
    ) -> JointPositions:
        """Inverse kinematics.

        Parameters
        ----------
        pose
            Target TCP pose, which must be in :attr:`Frame.BASE`.
        seed
            Optional joint seed for nearest-solution selection.

        Raises
        ------
        FrameMismatchError
            If ``pose.frame`` is not :attr:`Frame.BASE`.
        RobotKinematicsError
            If no IK solution exists.
        """
        ...

    # ---- high-level pipeline surface (vendor-neutral) -------------------
    #
    # These helpers keep pipelines and the API layer away from vendor-specific
    # symbols such as URPose or the workspace ``guard`` attribute. Every driver
    # implements them with bool semantics: pre-check first, then return ``False``
    # rather than raise, so a pipeline keeps its try-and-retry control flow.

    def is_inside_workspace(self, pose: Pose) -> bool:
        """Pre-flight check: would the driver's workspace policy accept ``pose``?

        ``pose.frame`` must be :attr:`Frame.BASE`. A driver with no configured
        workspace returns ``True``.
        """
        ...

    def move_to(
        self,
        pose: Pose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> bool:
        """High-level go-there command for pipelines.

        Returns ``True`` on success and ``False`` where a workspace or safety check
        or the controller rejected the move. A driver does not raise for the ordinary
        out-of-workspace and IK-failed cases, which the bool return signals, but may
        raise for a connection error.

        ``linear=True`` asks for a straight-line Cartesian move where the driver
        supports one, and the driver otherwise falls back to a joint-space move.

        ``register=False`` tells the driver not to add ``pose`` to any internal
        diversity or sampling history. A calibration routine or another per-pose
        orchestrator that keeps its own bookkeeping passes ``register=False``, so the
        driver's long-running guard does not double-count.
        """
        ...

    def move_home(self) -> bool:
        """Move to the driver's configured home configuration.

        Returns ``True`` on success and ``False`` on rejection.
        """
        ...

    def wait_until_steady(
        self,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.02,
    ) -> bool:
        """Block until the controller reports the arm has come to rest.

        Returns ``True`` where the arm settled within ``timeout_s`` and ``False`` on
        timeout. A driver with no explicit steady signal may implement this as
        ``time.sleep(timeout_s); return True``, so the exact semantics are
        vendor-specific. A calibration routine calls this between a commanded move
        and the next capture.
        """
        ...

    # ---- typed motion surface -------------------------------------------
    #
    # The bool-returning :meth:`move_to` and :meth:`move_home` collapse every failure
    # into a single ``False``, so an orchestrator or a calibration routine cannot
    # tell a workspace rejection from an IK failure, a controller refusal or a
    # connection fault. :meth:`move` is the typed replacement, and the bool methods
    # remain as compatibility shims.

    def move(
        self,
        pose: Pose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> MotionResult:
        """Typed go-there command for runtime orchestration.

        Returns a :class:`MotionResult` that classifies the outcome into one of
        :class:`MotionStatus`, so a caller reacts without scraping a log. A driver
        does not raise for the ordinary workspace, IK and controller-refusal cases,
        which :attr:`MotionResult.status` signals. It may raise on a connection or
        transport fault, but returning :attr:`MotionStatus.CONNECTION_ERROR` keeps
        the caller on the typed code path.

        ``register=False`` tells the driver not to add ``pose`` to any internal
        diversity or sampling history, with the semantics of :meth:`move_to`.
        """
        ...
