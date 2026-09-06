"""Robot calibration event protocol.

:class:`CalibrationRoutine` emits progress events during a hand-eye calibration
run. The surface is small and stable so loggers and tooling can subscribe to it
with confidence.

The event names below are canonical, so changing one is a breaking change for
every consumer. There is no web or UI bridge in this repo, and consumers are
in-process listeners.
"""

from __future__ import annotations

from typing import Any, Final, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Canonical event-type names
# ---------------------------------------------------------------------------

class RobotCalibrationEvent:
    """Namespace of canonical event-type strings for hand-eye calibration.

    Callers use ``RobotCalibrationEvent.MOVING_TO_POSE`` rather than repeating the
    literal, so a typo at a callsite raises a name error instead of silently
    dropping the event.
    """

    MOVING_TO_POSE: Final[str] = "robot_calibration_moving_to_pose"
    POSE_REJECTED: Final[str] = "robot_calibration_pose_rejected"
    MARKER_DETECTED: Final[str] = "robot_calibration_marker_detected"
    POSE_ACCEPTED: Final[str] = "robot_calibration_pose_accepted"

    #: Every canonical event name.
    ALL: Final[tuple[str, ...]] = (
        MOVING_TO_POSE,
        POSE_REJECTED,
        MARKER_DETECTED,
        POSE_ACCEPTED,
    )


# ---------------------------------------------------------------------------
# Listener protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class RobotCalibrationEventListener(Protocol):
    """Callable signature accepted by :class:`CalibrationRoutine`.

    An implementation must be fast and must not throw. The routine swallows
    listener exceptions so a buggy listener cannot abort a calibration run, but a
    slow listener still blocks the main loop.
    """

    def __call__(self, event_type: str, data: dict[str, Any]) -> None: ...


# ---------------------------------------------------------------------------
# Drift and out-of-distribution watchdog event surface
# ---------------------------------------------------------------------------

class RobotWatchdogEvent:
    """Namespace of canonical event-type strings for the watchdog.

    :class:`AutonomousGraspService` emits these when the watchdog's per-tick state
    crosses one of the following edges:

    * ``DRIFT_DETECTED``, drift_severity escalated from none or low to moderate
      or higher.
    * ``OOD_DETECTED``, ood_flagged went from False to True.
    * ``DEGRADED_MODE_ENGAGED``, degraded_mode_active went from False to True.
    * ``BLOCK_AUTO_TRIGGERED``, an enforced BLOCK_AUTO action fired and the
      service short-circuited the decision loop.
    * ``SLO_BREACH``, the runtime's rolling p95 for one of the locked latency
      stages (decision, ranking, fusion) crossed its configured budget. Emission
      is opt-in through ``robot.grasping.performance.emit_breach_events`` behind
      the ``performance.enabled`` master switch. The listener decides what to do:
      the runtime never fail-closes on its own SLO.

    Emission is best-effort and never blocks the grasp loop. Listener exceptions
    are swallowed so a buggy listener cannot abort a pick.
    """

    DRIFT_DETECTED: Final[str] = "robot_watchdog_drift_detected"
    OOD_DETECTED: Final[str] = "robot_watchdog_ood_detected"
    DEGRADED_MODE_ENGAGED: Final[str] = "robot_watchdog_degraded_mode_engaged"
    BLOCK_AUTO_TRIGGERED: Final[str] = "robot_watchdog_block_auto_triggered"
    SLO_BREACH: Final[str] = "robot_watchdog_slo_breach"

    #: Every canonical event name.
    ALL: Final[tuple[str, ...]] = (
        DRIFT_DETECTED,
        OOD_DETECTED,
        DEGRADED_MODE_ENGAGED,
        BLOCK_AUTO_TRIGGERED,
        SLO_BREACH,
    )


@runtime_checkable
class RobotWatchdogEventListener(Protocol):
    """Callable signature accepted by :class:`AutonomousGraspService`.

    An implementation must be fast and must not throw. The service swallows
    listener exceptions so a buggy listener cannot abort a grasp dispatch, but a
    slow listener still blocks the main loop.
    """

    def __call__(self, event_type: str, data: dict[str, Any]) -> None: ...
