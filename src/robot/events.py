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
