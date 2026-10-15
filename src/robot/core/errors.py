"""Vendor-neutral error hierarchy for the Workaholic-Willy robot subsystem.

Every driver translates its vendor-specific failures (``ur_rtde`` exceptions, ROS 2
service errors, ``franky`` faults) into one of these types before re-raising, so
application and pipeline code catches only the abstract types defined here.

Hierarchy::

    RobotError
        +-- RobotConnectionError    (cannot reach controller, dropped link)
        +-- RobotKinematicsError    (FK / IK failure, unreachable target)
        +-- RobotMotionRejected     (workspace guard / safety pre-check denied)
        |       `-- RobotSingularityRisk   (target too close to a singularity)
        +-- ArmFault                (the cell cannot move until a person acts; ArmFaultClass says which fault)
        +-- CameraWorldUnavailable  (a camera could not vouch for the cell after its attempts)
        +-- PerceptionFrameMoved    (a wrist camera took no still pick frame after its attempts)
        `-- IsaacNotAvailableError  (sim/SDK backend not installed on this host)
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "ArmFault",
    "ArmFaultClass",
    "CameraWorldUnavailable",
    "IsaacNotAvailableError",
    "RobotConnectionError",
    "RobotError",
    "RobotKinematicsError",
    "RobotMotionRejected",
    "RobotSingularityRisk",
]


class RobotError(Exception):
    """Base class for all vendor-neutral robot failures."""


class RobotConnectionError(RobotError):
    """Raised when the controller link cannot be opened or has dropped."""


class RobotKinematicsError(RobotError):
    """Raised when forward or inverse kinematics fail, or a target is unreachable."""


class RobotMotionRejected(RobotError):
    """Raised when a motion request is rejected before it reaches the driver.

    The causes are a workspace-guard rejection, a joint-limit pre-check failure, a
    soft-limit violation, or an input-validation error.

    ``result`` carries the typed :class:`~src.robot.core.motion_result.MotionResult` the
    refusal came from, where one exists. The bool-and-raise verbs and their typed twins
    refuse for the same reasons and through the same gates, so the raise carries the
    status, the target and the camera-world stamp instead of one string. It is ``None``
    for a refusal raised before any gate produced a result, such as a wrong number of
    joints.
    """

    def __init__(self, *args: object, result: object | None = None) -> None:
        super().__init__(*args)
        #: The refusal as the typed verbs report it, or ``None``. Typed as ``object`` so
        #: the error module stays importable from anywhere: ``motion_result`` imports the
        #: camera world, which imports back into core, and a cycle here would be paid by
        #: every module in the package.
        self.result = result


class RobotSingularityRisk(RobotMotionRejected):
    """Raised when a target is rejected because singularity risk is too high."""


class ArmFaultClass(StrEnum):
    """What kind of fault a cell is in, in words every vendor maps its controller into.

    The vocabulary is closed on purpose: a cell reports the class and what to do about it, and a class nobody can name
    is a class nobody can act on. What produces each one today is stated, because a member nothing produces is a
    promise, not a feature.
    """

    #: A safety stop a person can clear at the cell. Produced by :meth:`ArmFault.from_status`.
    PROTECTIVE_STOP = "protective_stop"
    #: The emergency button. Produced by :meth:`ArmFault.from_status`; never cleared from software.
    EMERGENCY_STOP = "emergency_stop"
    #: Powered off, in a fault mode, or otherwise not able to run. Produced by :meth:`ArmFault.from_status`.
    CONTROLLER_FAULT = "controller_fault"
    #: The pendant holds control, so the cell may not command it. Nothing produces it yet: the UR connect says local
    #: mode as a refused connection.
    NO_REMOTE_CONTROL = "no_remote_control"
    #: The link to the controller is gone. Nothing produces it yet.
    LINK_LOST = "link_lost"
    #: The controller latched a limit (joint, speed, force) and will not move until it is released. Nothing produces it
    #: yet; a vendor whose controller reports one maps it here rather than inventing a seventh word.
    LIMIT_REACHED = "limit_reached"
    #: The controller is ready, and the program on it that takes the cell's commands does not run (on a UR the RTDE
    #: control script, which a singular IK request and a protective stop both end, measured on URSim 2026-09-22).
    #: Produced by :meth:`ArmFault.from_status` from ``RobotStatus.program_running`` (P0e E13).
    PROGRAM_STOPPED = "program_stopped"


#: Which classes a person can acknowledge from software at all. An emergency stop is a physical button and a lost link
#: is not a state to acknowledge, so a cell that offered to clear them would be wrong about what it can do.
_ACKNOWLEDGEABLE: dict[ArmFaultClass, bool] = {
    ArmFaultClass.PROTECTIVE_STOP: True,
    ArmFaultClass.EMERGENCY_STOP: False,
    ArmFaultClass.CONTROLLER_FAULT: False,
    ArmFaultClass.NO_REMOTE_CONTROL: False,
    ArmFaultClass.LINK_LOST: False,
    ArmFaultClass.LIMIT_REACHED: True,
    ArmFaultClass.PROGRAM_STOPPED: False,
}

#: What a person does about each class, because the answer is never "retry".
_WHAT_TO_DO: dict[ArmFaultClass, str] = {
    ArmFaultClass.PROTECTIVE_STOP: "clear the stop where the arm is visible, then start the run again",
    ArmFaultClass.EMERGENCY_STOP: "release the emergency button at the cell, then start the run again",
    ArmFaultClass.CONTROLLER_FAULT: "power the controller back on and bring it to RUNNING, then start the run again",
    ArmFaultClass.NO_REMOTE_CONTROL: "give the controller back to remote control at the pendant",
    ArmFaultClass.LINK_LOST: "check the cable and the address, then start the run again",
    ArmFaultClass.LIMIT_REACHED: "release the latched limit at the controller, then start the run again",
    ArmFaultClass.PROGRAM_STOPPED: "connect the arm again, which starts the controller program anew, then start the "
                                   "run again",
}


class ArmFault(RobotError):
    """A fault of the cell, not of the grasp: this arm cannot move until a person acts.

    Nothing is wrong with the scene, the target or the plan, so no retry and no new look changes it. A campaign stops on
    it: as the report's ``fault`` where it was raised, and as the report's ``controller_stopped`` where a pick ended its
    attempt on it (owner-cell audit, 2026-09-23).

    Recovery is never automatic (the owner, 2026-09-19). Only a person acknowledges a fault, through
    ``SupportsRobotStatus.acknowledge_fault`` from a script run at the cell, and :attr:`acknowledgeable` says whether
    that is possible at all: an emergency button is not released by software.

    A halt is not one of these. It is the arm's own latch (``SupportsHalt``), set and cleared at the console by a person
    who confirms the cell is clear, and :meth:`from_status` leaves it aside.

    It replaced ``RobotEmergencyStop`` on 2026-10-13, which named one class and carried no evidence.
    """

    def __init__(self, kind: ArmFaultClass, message: str = "", *, status: object = None,
                 acknowledgeable: bool | None = None) -> None:
        #: Which fault this is, from the closed vocabulary.
        self.kind = kind
        #: The controller's own text, where it gave one.
        self.message = str(message or "")
        #: The reading this was derived from, or ``None``. Typed as ``object`` for the import rule of this module.
        self.status = status
        #: Whether a person can acknowledge this from software at all.
        self.acknowledgeable = _ACKNOWLEDGEABLE[kind] if acknowledgeable is None else bool(acknowledgeable)
        super().__init__(self.render())

    @classmethod
    def from_status(cls, status: object) -> "ArmFault | None":
        """The fault a ``RobotStatus`` describes, or ``None`` where the controller can move.

        The criterion is the controller's own answer (``controller_operational``), not ``is_stopped``, and that was
        measured: a controller powered off mid-session reported every stop flag as ``False``, correctly, while being
        exactly as unable to move. The halt latch is left aside: a halted arm on a healthy controller is no fault of the
        cell. A stopped program is named only on a controller that is otherwise ready, so a stop that ended the program
        as well is still named as the stop, which is what a person clears first.

        Args:
            status (object): A ``RobotStatus``, or anything with its fields.

        Returns:
            ArmFault | None: The classified fault, or ``None`` where the controller says it can move.
        """
        if getattr(status, "controller_operational", False):
            return None
        message = str(getattr(status, "message", "") or "")
        if getattr(status, "controller_ready", False) and getattr(status, "program_running", True) is False:
            return cls(ArmFaultClass.PROGRAM_STOPPED, message, status=status)
        if getattr(status, "emergency_stopped", False):
            return cls(ArmFaultClass.EMERGENCY_STOP, message, status=status)
        if getattr(status, "protective_stopped", False):
            return cls(ArmFaultClass.PROTECTIVE_STOP, message, status=status)
        if getattr(getattr(status, "safety_mode", None), "is_stopped", False):
            return cls(ArmFaultClass.PROTECTIVE_STOP, message, status=status)
        return cls(ArmFaultClass.CONTROLLER_FAULT, message, status=status)

    def what_to_do(self) -> str:
        """What a person does about this fault. Never "retry".

        Returns:
            str: One clause, ASCII.
        """
        return _WHAT_TO_DO[self.kind]

    def render(self) -> str:
        """Describe this to a person.

        Returns:
            str: One line, ASCII, no trailing newline: the class, the controller's text, the reading, what to do.
        """
        detail = f": {self.message}" if self.message else ""
        evidence = ""
        status = self.status
        if status is not None:
            evidence = (f" (robot_mode={getattr(getattr(status, 'robot_mode', None), 'value', '?')}, "
                        f"safety_mode={getattr(getattr(status, 'safety_mode', None), 'value', '?')}, "
                        f"protective_stop={getattr(status, 'protective_stopped', '?')}, "
                        f"emergency_stop={getattr(status, 'emergency_stopped', '?')}"
                        f"{', program_running=False' if getattr(status, 'program_running', True) is False else ''})")
        return (f"the cell is in {self.kind.value}{detail}{evidence}. Nothing is wrong with the scene or the grasp, and "
                f"no retry changes it: {self.what_to_do()}")


class CameraWorldUnavailable(RobotError):
    """Raised when a camera could not vouch for the cell after its fresh-frame attempts.

    This is a fault of the cell rather than the refusal of one motion: the camera answered nothing,
    a blind frame, or only frames older than the cell allows, and asking it again did not change
    that. A campaign stops on a dead camera instead of retrying it as though a grasp had missed:
    ``AutonomousGraspService.pick()`` reports it as its report's ``fault`` rather than raising it,
    and ``PickRun`` stops the campaign on that fault.

    It is neither a ``CuroboUnavailableError``, which driver sites turn into CONTROLLER_REJECTED,
    nor a :class:`RobotMotionRejected`, which is the ordinary refusal.
    """

    def __init__(
        self,
        *,
        camera: str,
        verdict: object,
        attempts: int,
        reason: str,
        refresh: object | None = None,
    ) -> None:
        #: The camera that could not vouch, as the cell names it.
        self.camera = str(camera)
        #: What it answered on the last reading: no frame, blind or stale.
        self.verdict = verdict
        #: How many readings were asked of it, the first included.
        self.attempts = int(attempts)
        #: The sentence the world source gave for the last reading.
        self.reason = str(reason)
        #: The refused refresh as a report carries it, or ``None``. Typed as ``object`` for the
        #: same import cycle rule as :attr:`RobotMotionRejected.result`.
        self.refresh = refresh
        super().__init__(
            f"camera {self.camera!r} could not vouch for the cell after {self.attempts} reading(s) "
            f"({verdict}): {self.reason}"
        )


class PerceptionFrameMoved(RobotError):
    """Raised when a wrist camera could not take a pick frame with the tool held still, after its attempts.

    The frame is placed by where the tool stood when the shutter opened, and a frame taken while
    the tool moved beyond the rig's shutter tolerance is placed by no single pose. The pick stops
    rather than grasping at a place the camera never saw from: an arm that moves while a camera
    takes the frame a pick is placed by is a fault of the cell.
    """

    def __init__(
        self, *, camera: str, attempts: int, moved_mm: float, turned_deg: float, tolerance_mm: float,
        tolerance_deg: float,
    ) -> None:
        #: The camera that could not take a still frame, as the cell names it.
        self.camera = str(camera)
        #: How many grabs were made, the first included.
        self.attempts = int(attempts)
        #: How far the tool moved and turned across the last grab.
        self.moved_mm = float(moved_mm)
        self.turned_deg = float(turned_deg)
        super().__init__(
            f"camera {self.camera!r} took {self.attempts} frame(s) and the tool moved across each grab, last by "
            f"{self.moved_mm:.3f} mm and {self.turned_deg:.3f} deg against the rig's shutter tolerance of "
            f"{float(tolerance_mm):g} mm and {float(tolerance_deg):g} deg, so no pose places a frame of it: the arm "
            "has to hold still while this camera takes a pick frame"
        )


class IsaacNotAvailableError(RobotError, RuntimeError):
    """Raised when a simulator or SDK feature is requested on a host without it.

    This is the canonical "the sim backend is not installed here" error, for instance
    Isaac Sim on a macOS host. It subclasses :class:`RobotError` so it belongs to the
    vendor-neutral taxonomy, and :class:`RuntimeError` so an existing
    ``except RuntimeError`` guard keeps catching it. A caller catches this type
    directly and surfaces a :class:`~src.robot.core.MotionResult` carrying
    :attr:`~src.robot.core.MotionStatus.UNSUPPORTED`.
    """
