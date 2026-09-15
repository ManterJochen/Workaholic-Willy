"""Whether a camera world stood behind a motion, stamped on the motion's own result.

The rule this contract serves: on a cell with cuRobo every motion plans against a world built from a
current camera image, unless the caller explicitly declines, and the decline is visible. Visible means
it travels with the motion rather than living in a log line or a config key:
:class:`~src.robot.core.motion_result.MotionResult` carries the stamp, and anything that aggregates
motions reads it from there.

There are five answers, and the default promises nothing. A result built without a stamp is UNSTATED:
nobody said whether the planner knew what the cameras saw, so a reader must not assume it did. Only
PLANNED vouches. DECLINED is a caller's decision and MISSING is the absence of one: a planner planned
the motion with no camera world, and nobody declined. Nothing enforces the rule, so such a motion is
stamped MISSING rather than refused.

Every driver in this repository stamps its two typed verbs, ``move`` and ``move_to_joints``, from what
the arm knows as it moves: UNPLANNED and why where no planner planned the motion, MISSING where a
planner planned it with no camera world, DECLINED where the caller declined one. Where a live camera
world is wired and nothing declined it, a ``move`` says PLANNED when the refresh made for that motion
vouched for the cell, naming the cameras and the capture time of the oldest image, and UNSTATED when
no refresh made for it did: the motion was refused before one ran, or its planner reports none. An
arm that stamps nothing, such as a caller's own, keeps saying UNSTATED, which is what the default is
for.

A decline is a keyword on the verb or a block around several motions (:func:`without_camera_world`),
with a reason either way. The block is bound to one arm, because two arms can run in one process, and
it is held in a :class:`~contextvars.ContextVar`, so it does not follow into a thread started inside
it.

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
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from src.contracts import Maybe, chosen

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from .motion_result import MotionResult

__all__ = [
    "DECLINE_ON_A_LIVE_WORLD_MESSAGE",
    "CameraWorldDecline",
    "CameraWorldStamp",
    "CameraWorldUse",
    "DeclinesCameraWorld",
    "active_decline",
    "resolve_camera_world",
    "stamp_result",
    "without_camera_world",
]


class CameraWorldUse(StrEnum):
    """What a motion's planner knew about the cell as the cameras saw it."""

    #: Nothing was said. The default of every result built without a stamp, read as not vouched for.
    UNSTATED = "unstated"
    #: A world built from a current camera image was registered before this motion was planned, or
    #: before its path was checked against it (a cuRobo joint move is checked rather than planned).
    PLANNED = "planned"
    #: The caller declined the camera world for this motion or this block, with a reason.
    DECLINED = "declined"
    #: No planner planned this motion or checked its path against a world, so none was consulted, with
    #: a reason: a cell with no planner, or a verb that neither plans nor checks on a cell that has one.
    UNPLANNED = "unplanned"
    #: A planner planned or checked this motion with no camera world, and nobody declined one, with a
    #: reason.
    MISSING = "missing"


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

    @classmethod
    def missing(cls, reason: str) -> "CameraWorldStamp":
        return cls(use=CameraWorldUse.MISSING, reason=reason)

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


# ------------------------------------------------------------ declining, and reading a decline

#: The declines in scope, innermost last, each bound to the id of the arm it was entered for.
_DECLINES: ContextVar[tuple[tuple[int, CameraWorldDecline], ...]] = ContextVar(
    "camera_world_declines", default=()
)

DECLINE_ON_A_LIVE_WORLD_MESSAGE = (
    "Refused before planning: this motion declines the camera world, and the planner cannot set aside "
    "the live camera world wired to this arm for one motion. Nothing moved."
)
"""The message of the result a driver returns for a declined motion on an arm whose live world is wired.

The planner is handed that world before every plan, and nothing registers the declared world in its
place for one motion, so planning anyway would plan against the world that was declined. The status is
``UNSUPPORTED``: the cell cannot do what the caller asked, and the controller is never reached.
"""


def without_camera_world(arm: object, reason: str) -> AbstractContextManager[CameraWorldDecline]:
    """Decline the camera world for every motion ``arm`` commands inside the ``with`` block.

    The reason is checked when the block is asked for, so a blank one is refused before anything moves.
    The block is bound to ``arm`` alone: another arm in the same process plans as it would without it.
    A keyword decline on a verb inside the block beats the block, and an inner block beats an outer one.

    The block follows the code in the ``with`` body and does not follow a thread started inside it, so a
    pick that runs on a thread of its own is not declined by a block entered around starting it.
    """
    return _declining(arm, CameraWorldDecline(reason))


@contextmanager
def _declining(arm: object, decline: CameraWorldDecline) -> Iterator[CameraWorldDecline]:
    # The generator frame holds ``arm`` for the life of the block, so its id cannot pass to another
    # object while the decline is in scope.
    token = _DECLINES.set((*_DECLINES.get(), (id(arm), decline)))
    try:
        yield decline
    finally:
        _DECLINES.reset(token)


def active_decline(arm: object) -> CameraWorldDecline | None:
    """The innermost decline in scope for ``arm``, or ``None`` when no block for it is open."""
    wanted = id(arm)
    for owner, decline in reversed(_DECLINES.get()):
        if owner == wanted:
            return decline
    return None


def resolve_camera_world(
    *,
    unplanned: str | None,
    missing: str | None,
    keyword: Maybe[CameraWorldDecline],
    block: CameraWorldDecline | None,
    planned: CameraWorldStamp | None = None,
) -> CameraWorldStamp:
    """The camera-world answer for one motion, from what its arm knows before it moves. Pure.

    ``unplanned`` says why no planner plans or checks this motion, and is ``None`` when one does.
    ``missing`` says why such a motion has no camera world, and is ``None`` when a live one is wired.
    ``keyword`` is the verb's own decline and ``block`` the innermost one in scope
    (:func:`active_decline`). ``planned`` is the PLANNED stamp of the refresh made for this motion, and
    ``None`` when none vouched.

    The first rule that matches answers:

    1. UNPLANNED when no planner plans or checks the motion. A decline changes nothing, because there
       was no world to set aside.
    2. DECLINED when a decline is in scope, the keyword before the block.
    3. MISSING when the planned or checked motion has no camera world.
    4. PLANNED when the refresh made for this motion vouched for the cell.
    5. UNSTATED otherwise: a live world is wired, nothing declined it, and no refresh made for this
       motion vouched for it.

    A keyword that is not a :class:`CameraWorldDecline` is refused whatever the arm, so a reason passed as
    text fails where it was written rather than only on the cell that plans.
    """
    if chosen(keyword) and not isinstance(keyword, CameraWorldDecline):
        raise TypeError(
            f"camera_world is a CameraWorldDecline, not {keyword!r}; pass "
            f"CameraWorldDecline('why this motion needs no camera world')"
        )
    if planned is not None and planned.use is not CameraWorldUse.PLANNED:
        raise ValueError(
            f"planned is the stamp a refresh vouched with, and {planned.render()!r} vouches for nothing"
        )
    if unplanned is not None:
        return CameraWorldStamp.unplanned(unplanned)
    decline = keyword if chosen(keyword) else block
    if decline is not None:
        return CameraWorldStamp.declined(decline)
    if missing is not None:
        return CameraWorldStamp.missing(missing)
    if planned is not None:
        return planned
    return CameraWorldStamp.unstated()


def stamp_result(result: "MotionResult", stamp: CameraWorldStamp) -> "MotionResult":
    """``result`` carrying ``stamp`` when it says nothing yet, else ``result`` itself.

    Only a :class:`~src.robot.core.motion_result.MotionResult` whose stamp is UNSTATED is replaced. A
    result that already carries a stamp keeps it, and anything that is not a result, such as what a
    stand-in planner returns, is handed back as it came.
    """
    # Imported here because the result module imports this one at its top.
    from .motion_result import MotionResult as _Result

    if isinstance(result, _Result) and result.camera_world.use is CameraWorldUse.UNSTATED:
        return replace(result, camera_world=stamp)
    return result


@runtime_checkable
class DeclinesCameraWorld(Protocol):
    """Capability extension: this arm reads a camera-world decline and stamps its typed motions.

    The shape of ``SafetyGated`` (``src/robot/safety/attestation.py``): some arms can do this, and an
    arm that cannot does not implement it. It is not a member of ``RobotArm``, so a caller's own arm
    keeps satisfying that Protocol, and the results it returns keep saying UNSTATED.
    """

    def without_camera_world(self, reason: str) -> AbstractContextManager[CameraWorldDecline]:
        """Decline the camera world for every motion this arm commands inside the ``with`` block."""
        ...
