"""Whether a camera world stood behind a motion, stamped on the motion's own result.

The rule this contract serves: on a cell with cuRobo every motion plans against a world built from a
current camera image, unless the caller explicitly declines, and the decline is visible. Visible means
it travels with the motion rather than living in a log line or a config key:
:class:`~src.robot.core.motion_result.MotionResult` carries the stamp, and anything that aggregates
motions reads it from there.

There are four answers, and the default promises nothing. A result built without a stamp is UNSTATED:
nobody said whether the planner knew what the cameras saw, so a reader must not assume it did. Only
PLANNED vouches. Nothing enforces the rule and no driver sets the stamp, so every result a driver
builds says UNSTATED.

``captured_at_s`` is seconds on the wall clock, ``time.time()``, which is the clock the live planner
world stamps its frames with in ``src/robot/execution/autonomous_grasp/live_world.py``. A world built
from several cameras carries the capture time of its oldest image, because a world is as stale as the
stalest image in it.

``__post_init__`` checks the invariants, not only the factories, so a stamp built with the bare
constructor cannot claim a planned world without naming a camera or a decline without a reason, and a
plain string given as the use meets the same checks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = ["CameraWorldDecline", "CameraWorldStamp", "CameraWorldUse"]


class CameraWorldUse(StrEnum):
    """What a motion's planner knew about the cell as the cameras saw it."""

    #: Nothing was said. The default of every result built without a stamp, read as not vouched for.
    UNSTATED = "unstated"
    #: A world built from a current camera image was registered before this motion was planned.
    PLANNED = "planned"
    #: The caller declined the camera world for this motion or this block, with a reason.
    DECLINED = "declined"
    #: No planner runs on this cell, so no world could be consulted: a cell-wide decline, with a reason.
    UNPLANNED = "unplanned"


def _one_line_ascii(text: str) -> str:
    """``render()`` is one ASCII line, and a reason or a camera name is free text.

    ``unicode_escape`` writes a line break, a tab or a non-ASCII character as its escape sequence, so a
    caller's text can neither break the line nor leave ASCII.
    """
    return text.encode("unicode_escape").decode("ascii")


def _reason_or_refuse(reason: object, what: str) -> str:
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(
            f"{what} needs a reason: a decline nobody can read back afterwards is not a decision"
        )
    return reason


def _cameras_or_refuse(cameras: object) -> tuple[str, ...]:
    if not isinstance(cameras, tuple):
        raise TypeError(f"cameras is a tuple of camera names, not a {type(cameras).__name__}")
    for name in cameras:
        if not isinstance(name, str):
            raise TypeError(f"a camera name is text, not {name!r}")
        if not name.strip():
            raise ValueError("a camera name cannot be blank")
    if len(set(cameras)) != len(cameras):
        raise ValueError(f"a camera is named twice in {list(cameras)}")
    return cameras


def _capture_time_or_refuse(moment: object) -> float:
    if isinstance(moment, bool) or not isinstance(moment, (int, float)):
        raise TypeError(f"captured_at_s is a number of seconds, not {moment!r}")
    seconds = float(moment)
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError(f"captured_at_s is a finite, non-negative time.time() reading, not {moment!r}")
    return seconds


@dataclass(frozen=True, slots=True)
class CameraWorldDecline:
    """An explicit decision to move without a camera world, and why."""

    reason: str

    def __post_init__(self) -> None:
        _reason_or_refuse(self.reason, "a camera-world decline")


@dataclass(frozen=True, slots=True)
class CameraWorldStamp:
    """The camera-world answer for one motion. Build it with a factory."""

    use: CameraWorldUse
    reason: str = ""
    cameras: tuple[str, ...] = ()
    captured_at_s: float | None = None

    def __post_init__(self) -> None:
        use = self.use
        if not isinstance(use, CameraWorldUse):
            try:
                use = CameraWorldUse(use)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{use!r} is not a camera-world use; the uses are "
                    f"{', '.join(item.value for item in CameraWorldUse)}"
                ) from None
            object.__setattr__(self, "use", use)
        # The fields that must stay empty are compared with the empty value of their own type, not
        # tested for truth: None, [] or "" in their place is refused too, so a stamp that is accepted
        # equals the one its factory builds, hashes, and survives to_dict().
        if use is CameraWorldUse.UNSTATED:
            if self.reason != "" or self.cameras != () or self.captured_at_s is not None:
                raise ValueError("an unstated camera world carries nothing, because nothing was said")
            return
        if use is CameraWorldUse.PLANNED:
            if self.reason != "":
                raise ValueError("a planned camera world carries no reason, because nothing was declined")
            if not _cameras_or_refuse(self.cameras):
                raise ValueError("a planned camera world names at least one camera")
            if self.captured_at_s is None:
                raise ValueError("a planned camera world carries the time its image was captured")
            object.__setattr__(self, "captured_at_s", _capture_time_or_refuse(self.captured_at_s))
            return
        _reason_or_refuse(self.reason, f"a {use.value} camera world")
        if self.cameras != () or self.captured_at_s is not None:
            raise ValueError(
                f"a {use.value} camera world names no camera and no capture time: no image was "
                f"planned against"
            )

    # ------------------------------------------------------------------ factories

    @classmethod
    def unstated(cls) -> "CameraWorldStamp":
        return cls(use=CameraWorldUse.UNSTATED)

    @classmethod
    def planned(cls, *, cameras: Sequence[str], captured_at_s: float) -> "CameraWorldStamp":
        if isinstance(cameras, str):
            raise TypeError(
                f"cameras is a sequence of camera names, and {cameras!r} is one name: pass "
                f"({cameras!r},)"
            )
        return cls(use=CameraWorldUse.PLANNED, cameras=tuple(cameras), captured_at_s=captured_at_s)

    @classmethod
    def declined(cls, decline: CameraWorldDecline) -> "CameraWorldStamp":
        return cls(use=CameraWorldUse.DECLINED, reason=decline.reason)

    @classmethod
    def unplanned(cls, reason: str) -> "CameraWorldStamp":
        return cls(use=CameraWorldUse.UNPLANNED, reason=reason)

    # ------------------------------------------------------------------ reading

    @property
    def vouched(self) -> bool:
        """Did a current camera world stand behind the motion? True only for PLANNED."""
        return self.use is CameraWorldUse.PLANNED

    def render(self) -> str:
        """One line for a person. ASCII, no trailing newline."""
        head = f"camera world  {self.use.value.upper()}"
        if self.use is CameraWorldUse.PLANNED:
            cameras = ", ".join(_one_line_ascii(name) for name in self.cameras)
            return f"{head}  cameras {cameras}, image captured at {self.captured_at_s:.3f} s"
        if self.use is CameraWorldUse.UNSTATED:
            return f"{head}  nothing said whether the planner knew what the cameras saw"
        return f"{head}  {_one_line_ascii(self.reason)}"

    def to_dict(self) -> dict[str, Any]:
        """Plain data; survives ``json.dumps`` with no custom encoder."""
        return {
            "use": self.use.value,
            "vouched": self.vouched,
            "reason": self.reason,
            "cameras": list(self.cameras),
            "captured_at_s": self.captured_at_s,
        }
