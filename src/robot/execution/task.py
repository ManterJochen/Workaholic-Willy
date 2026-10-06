"""A task: pick a part, place it, return, and look again: once, or until nothing is left (the owner's OD 7 and OD 9).

The console's task as a library verb, which a program calls as the console does::

    from willy import PlaceAt, TaskPlan, run_task

    report = run_task(service, TaskPlan(object="green cube", place=PlaceAt(pose="drop_left"), scope="until_empty"),
                      hooks=my_hooks, poses={"drop_left": JointPositions.deg(-60, -95, -120, -55, 90, 0)})
    print(report)

``service`` is a connected cell's pick service (``AutonomousGraspService``), ``hooks`` what the task says and reads
between its motions (:class:`TaskHooks`), and ``poses`` the taught poses it may use, by name. Each part is one turn of
the loop: a pick that looks fresh, a place, the return to the return pose. Its rules:

* **The order** (build plan 1.3.3). The service's latch first: a recovery of an earlier pick that stopped where the arm
  stands ends the task ``recovery_needs_person`` with nothing read, asked or commanded, and the latch kept: the task
  never calls ``start_campaign``, which would acknowledge it, before it read it, and never clears it. Then the screen
  of every taught pose it will use, before any motion (``task.pose_screened``; an ERROR is ``pose_refused``). Then the
  setup, every switch put back as found when the task ends, whatever ended it: the campaign (its push distance), the
  prompt, the closing axis, the overlay, the cancel check (halt and a cell taken down, never "stop after this part"),
  and the regions it keeps out. A Restart moves to its return pose first. A camera place then finds its bin
  (``place_target.survey``) before the first pick.
* **The scope.** ``once`` ends when the part is placed and the arm is back; ``until_empty`` when
  ``empty_looks_to_end`` looks in a row saw nothing to pick; a once task that sees nothing looks once more too. A look
  that saw only parts the task keeps out (its regions, never a part ``next_target`` skips after a failed pick) is an
  empty one. ``max_failed_in_a_row`` failed picks in a row end it where the arm stands, and ``max_parts`` placed end it
  (``part_limit``). The two rows count apart, as the build plan writes them (1.3.3): a failed pick does not start the
  empty row again, nor an empty look the failed row; only a part placed starts both. A detector that failed where
  nothing was found is ``detector_failed``, never "nothing left".
* **Benign ends return, problems stop.** A done, operator or ask end moves the arm to its return pose first (where the
  task moved it at all and it is not standing there); a problem stop leaves it where it stands and commands nothing
  more, an output included. Before every motion the task reads whether it may still move (the operator's halt, the
  arm's own halt latch, a cell being taken down); a halt is never read as a refused motion or a stopped controller.
  After a motion refused once the part was let go (a line out), the controller is asked too before the return: a
  stopped one gets nothing more. "Stop after this part" is read at the top of every part, and before the survey and
  each of its looks, where no part is in hand yet.
* **The carried part.** A task refuses to start on an arm that cannot model the part it carries
  (``payload_declined_reason``), or whose place and return no planner judges (``route_of``): between the close and
  the release only the planner holds it. After every pick it reads what the pick's attach left in force
  (``payload_model``): a part the planner declined there is carried nowhere, the task ends where the arm stands.
* **The hand.** Before the task's first motion of any kind and before every pick, the library's copy of the console's
  check (:func:`jaws_before_a_pick`): a toggle the program believes closed, or whose count nobody can vouch for, ends
  the task with nobody asked; then a hand that measures a part, or a planner that still models one, ends it
  ``part_still_held`` with nothing moved. While the task runs nobody is asked on its thread either
  (``asking_nobody``), so a question the hand would ask is a refusal. A toggle hand changes DO0 exactly twice per part,
  the close and the place's release, and the place leaves its count OPEN for the next pick; twice more for each
  blocker the pick sets aside first (its close and its release, the pick loop's "clear the blocker"). A pick that
  failed with a part possibly in the jaws ends the task holding it.
* **The place.** A taught pose says where the part's bottom is let go (``place_target.pose_drop``); a camera place goes
  into the bin over its rim (``place_target.drop_plan``), carried to the look it was kept from, every frame held, the
  bin checked again first (against where the survey found it), and placed without a keep-out. Either drop keeps the
  tilt the tool gripped with, turned about the vertical only, so the hang it is raised by still bounds the part. A
  drop refused, a bin lost or a part that does not fit puts the part back where it was gripped (``service.put_back``),
  returns, and asks. A place counts its part placed once the jaws let go (:func:`place_released`), its line out
  refused or not.
* **What it keeps out of its picks** (Q4), for the whole task: a camera place's bin, its footprint grown by 10 mm, for
  both scopes; a pose place's drop, a circle of ``pose_keep_out_mm`` about it, for "until empty", where the arm's
  forward kinematics say where the taught pose puts the tool (an arm whose ``fk`` is no kinematic model, the
  rehearsal's dummy, gets none, and the log says so). No pick takes the bin or a part already placed; a look that sees
  only those is an empty one.

It never asks a person, and every end of a run it started is a :class:`TaskReport`. A request this cell cannot do (an
arm whose place cannot run, an unknown pose, a carried part nobody models, an empty object without ``pick_anything``,
both jaw faces on a cell that turns every grasp off them) raises :class:`TaskRefused` before anything is commanded, and
a programmer's error raises.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, Protocol, Sequence

from src.contracts import UNSET, Maybe, chosen
from src.robot.core.errors import RobotError
from src.robot.execution.pick_run import configured_looks_of, keep_pick_views
from src.robot.execution.place_target import (
    RIM_AIR_MAX_MM,
    RIM_AIR_MIN_MM,
    DropPlan,
    KeptTarget,
    TargetCheck,
    drop_plan,
    locators_for_service,
    nominal_drop,
    pose_drop,
    recheck,
    rim_clearance_mm,
    screen_joints,
    screen_pose,
    survey,
)

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.geometry import Pose
    from src.robot.core import JointPositions
    from src.robot.execution.autonomous_grasp.report import AutonomousGraspReport
    from src.robot.execution.handling import HandlingReport
    from src.robot.execution.motion import MotionReport

__all__ = [
    "PlaceAt",
    "TaskEvent",
    "TaskHooks",
    "TaskOptions",
    "TaskPlan",
    "TaskRefused",
    "TaskReport",
    "TaskStop",
    "jaws_before_a_pick",
    "place_released",
    "run_task",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------------------------------------
# The words of a task
# ---------------------------------------------------------------------------------------------------------------------


class TaskStop(StrEnum):
    """Why a task ended. Each is a stop code of the console (``api.codes.StopCode``), of the class :attr:`stop_class`."""

    FINISHED = "finished"
    NOTHING_LEFT = "nothing_left"
    PART_LIMIT = "part_limit"
    STOPPED_AFTER_PART = "stopped_after_part"
    TARGET_NOT_FOUND = "target_not_found"
    TARGET_LOST = "target_lost"
    TARGET_UNREACHABLE = "target_unreachable"
    PART_DOES_NOT_FIT = "part_does_not_fit"
    POSE_REFUSED = "pose_refused"
    HALTED = "halted"
    CONTROLLER_STOPPED = "controller_stopped"
    HAND_NEEDS_PERSON = "hand_needs_person"
    RECOVERY_NEEDS_PERSON = "recovery_needs_person"
    PART_STILL_HELD = "part_still_held"
    RETURN_FAILED = "return_failed"
    FAILED_IN_A_ROW = "failed_in_a_row"
    DETECTOR_FAILED = "detector_failed"
    CELL_FAULT = "cell_fault"
    DISCONNECTED = "disconnected"

    @property
    def stop_class(self) -> str:
        """``done``, ``operator``, ``ask`` or ``problem``: what the operator sees, the console's classes."""
        return _CLASS_OF.get(self, "problem")

    @property
    def returns(self) -> bool:
        """Whether a task that ends so moves the arm to its return pose first: every class but a problem."""
        return self.stop_class != "problem"


_CLASS_OF: Mapping[TaskStop, str] = {
    TaskStop.FINISHED: "done",
    TaskStop.NOTHING_LEFT: "done",
    TaskStop.PART_LIMIT: "done",
    TaskStop.STOPPED_AFTER_PART: "operator",
    TaskStop.TARGET_NOT_FOUND: "ask",
    TaskStop.TARGET_LOST: "ask",
    TaskStop.TARGET_UNREACHABLE: "ask",
    TaskStop.PART_DOES_NOT_FIT: "ask",
    TaskStop.POSE_REFUSED: "ask",
}


class TaskEvent(StrEnum):
    """What a task says between its motions, each an event type of the console (build plan 1.4)."""

    POSE_SCREENED = "task.pose_screened"
    SURVEY_STARTED = "task.survey_started"
    TARGET_FOUND = "task.target_found"
    TARGET_MISSING = "task.target_missing"
    PART_STARTED = "task.part_started"
    NOTHING_FOUND = "task.nothing_found"
    CARRY_STARTED = "task.carry_started"
    TARGET_CHECKED = "task.target_checked"
    TARGET_LOST = "task.target_lost"
    DROP_PLANNED = "task.drop_planned"
    PLACE_STARTED = "task.place_started"
    PLACED = "task.placed"
    PLACE_FAILED = "task.place_failed"
    PUT_BACK = "task.put_back"
    RETURN_STARTED = "task.return_started"
    RETURNED = "task.returned"
    RETURN_FAILED = "task.return_failed"
    PART_FINISHED = "task.part_finished"


#: What a task grounds in front of the object it names, so every such part comes back in a box of its own. On a mat of
#: 20 parts Qwen3-VL-4B grounded one box for "object", "objects", "all objects" or "every object" and all 20 for "each
#: separate object", one of the two green parts for "green part" and both for "each separate green part", both red
#: ones and no orange one for "each separate red part" (2026-10-06). A part the detector does not name keeps the whole
#: rule beside the fingers, the push and the blocker.
EACH_SEPARATE = "each separate"
#: What a task that names no object grounds: every part in a box of its own.
EVERY_PART_PHRASE = f"{EACH_SEPARATE} object"


def _grounds_a_phrase(service: Any) -> bool:
    """Whether ``service``'s cell grounds a phrase (``grounds_a_phrase``); ``False`` for one that does not say."""
    asked = getattr(service, "grounds_a_phrase", None)
    return bool(asked()) if callable(asked) else False

class TaskRefused(ValueError):
    """A task asked to do what this cell cannot, refused before anything was commanded.

    ``code`` is the console's refusal code for it (``api.codes.RefusalCode``): ``unknown_pose``, ``route_refused``,
    ``carried_part_not_modelled``, ``object_required``, ``closing_axis_refused``, ``bad_request`` (both jaw faces asked
    of a cell whose motion turns every grasp off the faces judged), ``push_distance_refused`` or
    ``camera_target_unavailable``. The console refuses each before it starts a run; this is the library's own backstop,
    and a run that meets it moved nothing.
    """

    def __init__(self, code: str, sentence: str) -> None:
        super().__init__(sentence)
        self.code = code


# ---------------------------------------------------------------------------------------------------------------------
# What a task is asked to do
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PlaceAt:
    """Where a task puts each part: a taught pose (``pose``, its name in the task's ``poses``), or a target the camera
    finds (``camera``, an English phrase such as ``"blue bin"``), one of the two.

    ``air_mm`` is the air over a camera target's rim when the jaws open, 10 to 50 mm (Q3); ``None`` is the cell's own
    (``place_target.rim_clearance_mm``, 20 mm as shipped). A taught pose takes none: it says where the part's bottom is
    let go.
    """

    pose: str | None = None
    camera: str | None = None
    air_mm: float | None = None

    def __post_init__(self) -> None:
        named = [value for value in (self.pose, self.camera) if value is not None]
        if len(named) != 1:
            raise ValueError("a place names a taught pose (pose=) or a target the camera finds (camera=), one of the two")
        if not str(named[0]).strip():
            raise ValueError("a place's pose name or camera phrase names nothing")
        if self.air_mm is not None:
            if self.camera is None:
                raise ValueError("air_mm is the air over a camera target's rim; a taught pose says where the part's "
                                 "bottom is let go and takes none")
            air = float(self.air_mm)
            if not (math.isfinite(air) and RIM_AIR_MIN_MM <= air <= RIM_AIR_MAX_MM):
                raise ValueError(f"the air over a bin's rim is {RIM_AIR_MIN_MM:g} to {RIM_AIR_MAX_MM:g} mm (Q3), not "
                                 f"{self.air_mm!r}")


@dataclass(frozen=True)
class TaskOptions:
    """The Advanced drawer's switches, for one task: never the cell's.

    ``multi_view`` the looks as configured, ``False`` the first look only and no generated view (Q11); ``both_faces``
    both jaw contact faces seen before a grip; ``closing_axis`` only grasps that close along it (a name
    ``Pose.tool_down`` takes); ``push_mm`` how far a push of a failed part moves it (``None``: the cell's, which may go
    longer where it opens too little room); ``critical_parts`` the owner's switch for this task (``None``: the cell's
    ``recovery.critical_parts``; true clears a blocker and pushes nothing); ``record_views`` each pick's looks
    kept for training; ``overlay`` the grasp overlay rendered during the task; ``pick_anything`` the operator's word
    that an empty object means anything the camera sees, bin walls included (Q7 A+). ``rescan``, ``push`` and ``clear``
    are the run's own word on recovery (the owner, 2026-10-05: the console's ticks override the config for one run):
    ``None`` keeps the cell's ``recovery.allowed_actions``; ``push`` or ``clear`` true allows ``nudge_target``, both
    false takes it out; ``clear`` without ``push`` clears blockers and pushes nothing (critical parts).
    ``blocker_into_the_place`` the owner's switch for where a blocker goes (``None``: the cell's
    ``recovery.blocker_into_the_place``): true, a task that names no object takes it as the part its pick takes and
    sets it down where the parts go, then picks the part it blocked; false, it is set aside on the support. A task that
    names an object sets every blocker aside (2026-10-06).
    """

    multi_view: bool = True
    both_faces: bool = False
    closing_axis: str | None = None
    push_mm: float | None = None
    critical_parts: bool | None = None
    record_views: bool = False
    overlay: bool = True
    pick_anything: bool = False
    rescan: bool | None = None
    push: bool | None = None
    clear: bool | None = None
    blocker_into_the_place: bool | None = None


@dataclass(frozen=True)
class TaskPlan:
    """One task: what to pick (``object``, the English phrase the detector grounds; ``""`` anything), where to put it,
    where to go after (``home`` or a taught pose's name), and how long (``once`` or ``until_empty``).

    ``first_motion`` is ``look`` for a new task and ``return`` for a Restart, whose first motion is the planned move to
    the return pose. The limits are the owner's: ``max_failed_in_a_row`` (3), ``empty_looks_to_end`` (2), ``max_parts``
    (100) and ``pose_keep_out_mm`` (150 mm about a pose place's drop for "until empty", Q4).
    """

    object: str
    place: PlaceAt
    return_to: str = "home"
    scope: Literal["once", "until_empty"] = "once"
    options: TaskOptions = field(default_factory=TaskOptions)
    first_motion: Literal["look", "return"] = "look"
    max_failed_in_a_row: int = 3
    empty_looks_to_end: int = 2
    max_parts: int = 100
    pose_keep_out_mm: float = 150.0

    def __post_init__(self) -> None:
        if self.scope not in ("once", "until_empty"):
            raise ValueError(f"a task's scope is 'once' or 'until_empty', not {self.scope!r}")
        if self.first_motion not in ("look", "return"):
            raise ValueError(f"a task's first motion is 'look' or 'return', not {self.first_motion!r}")
        for name in ("max_failed_in_a_row", "empty_looks_to_end", "max_parts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} is a whole number of at least 1, not {value!r}")
        keep_out = float(self.pose_keep_out_mm)
        if not (math.isfinite(keep_out) and keep_out > 0.0):
            raise ValueError(f"pose_keep_out_mm is a positive number of mm, not {self.pose_keep_out_mm!r}")
        if not isinstance(self.place, PlaceAt):
            raise TypeError(f"a task's place is a PlaceAt, not {type(self.place).__name__}")
        if not str(self.return_to).strip():
            raise ValueError("return_to names home or a taught pose; it names nothing")


class TaskHooks(Protocol):
    """What a task calls on its caller between its motions: the console's hooks (``api.task_run.ConsoleTaskHooks``),
    or a program's own.

    ``event`` is one of the task's events (:class:`TaskEvent`) with its data as keywords, among them ``said``, the
    library's English sentence for it (the console's ``human``, kept out of ``data``); a camera target's events carry
    its overlay as ``image_png`` (bytes, or ``None``), which is no data to publish. ``pick_done`` is one pick of the task
    ended, with its report (its looks' file in ``telemetry['views_file']`` where they were kept). ``stop_after_part`` is
    whether the operator asked the task to stop once the part in hand is placed; ``halted`` whether "halt now" was
    pressed; ``abandoned`` why the cell is being taken down under the task, ``""`` while it is not.
    """

    def event(self, name: TaskEvent, /, **data: Any) -> None:
        ...

    def pick_done(self, part: int, pick: int, report: "AutonomousGraspReport") -> None:
        ...

    def stop_after_part(self) -> bool:
        ...

    def halted(self) -> bool:
        ...

    def abandoned(self) -> str:
        ...


@dataclass(frozen=True)
class TaskReport:
    """What a task did: why it ended (``stop``, said in ``sentence``), the parts it released at the place, the picks it
    made and how many gripped, whether the program believed a part in the jaws at the end (a record only: a gate reads
    the live hand), whether the arm stands at the task's return pose because the task's last motion was the return there
    (``at_return``; ``False`` where the task moved nothing, or its last motion went anywhere else), and the last pick's
    own report."""

    stop: TaskStop
    sentence: str
    parts_placed: int = 0
    picks: int = 0
    succeeded: int = 0
    holding: bool = False
    last_report: "AutonomousGraspReport | None" = None
    at_return: bool = False

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """The task, for a person. ASCII, no trailing newline, no arguments."""
        lines = [f"task  {self.stop.value.upper()}  {self.parts_placed} part(s) placed in {self.picks} pick(s), "
                 f"{self.succeeded} gripped", f"  {self.sentence}"]
        if not self.stop.returns:
            lines.append("  the arm stays where it stands: nothing more was commanded, and a person decides what "
                         "happens next")
        elif self.at_return:
            lines.append("  the arm is back at its return pose")
        else:
            lines.append("  the task moved nothing: the arm stands where it stood")
        if self.holding:
            lines.append("  the program believes a part is in the jaws")
        return "\n".join(line.encode("ascii", "backslashreplace").decode("ascii") for line in lines)

    def to_dict(self) -> dict[str, Any]:
        """Plain data, ``json.dumps`` safe."""
        last = self.last_report
        return {
            "stop": self.stop.value,
            "stop_class": self.stop.stop_class,
            "sentence": self.sentence,
            "parts_placed": self.parts_placed,
            "picks": self.picks,
            "succeeded": self.succeeded,
            "holding": self.holding,
            "at_return": self.at_return,
            "last_report": None if last is None else last.to_dict(),
        }


# ---------------------------------------------------------------------------------------------------------------------
# Two rules a task shares with the console
# ---------------------------------------------------------------------------------------------------------------------


def jaws_before_a_pick(gripper: Any) -> str:
    """Why the next pick of a task must not start on ``gripper``, or ``""`` where it may: the console's rule
    (``api/runs.py``, ``_why_no_pick_starts``) in the library's own words.

    Only a hand that toggles with no sensor is read, and nothing is asked or sent. A count nobody can vouch for (a
    change that failed, an output switched by hand at the pendant, a read that fails) and a count that says CLOSED (the
    last part was never set down) end the task: ``pick()`` would ask a person where the jaws stand, and a task asks
    nobody. Fail closed: anything but a plain ``False`` from ``edge_unknown`` and ``jaws_closed`` stops it.
    """
    from src.robot.core.gripper import toggle_without_sensor_of  # noqa: PLC0415

    toggle = toggle_without_sensor_of(gripper)
    if toggle is None:
        return ""
    try:
        reader = getattr(toggle, "why_jaws_unknown", None)
        why = reader() if callable(reader) else ""
        unknown = getattr(toggle, "edge_unknown", True)
        closed = toggle.jaws_closed
    except Exception as exc:  # noqa: BLE001 (a hand that cannot be read vouches for nothing)
        return (f"the gripper needs a person: where its jaws stand could not be read ({type(exc).__name__}: {exc}), "
                "so the task stops and asks nobody")
    why = why if isinstance(why, str) else ""
    if why or unknown is not False:
        return ("the gripper needs a person: nobody can say where the jaws of this toggle hand stand "
                f"({why or 'its last change failed'}), so the task stops before the pick and asks nobody. Look at the "
                "jaws, empty them, and answer where they stand in the console")
    if closed is not False:
        return ("the gripper needs a person: the program's count says the jaws of this toggle hand stand CLOSED (the "
                "last part was never set down), so the task stops before the pick and asks nobody. Take the part, and "
                "answer where the jaws stand in the console")
    return ""


def place_released(report: "HandlingReport") -> bool:
    """Whether a place (or a put back) let its part go: it ran whole, or a motion after the release was refused.

    ``Robot.place`` commands the standoff, the line in, the release, then the line out, and stops at the first motion
    refused; so a report that commanded three motions released the part before the third was refused (its line out,
    refused as a motion or for a camera that could not vouch). Anything else, refused before the release or a release
    the gripper did not confirm, still holds it.
    """
    from src.robot.execution.handling import HandlingOutcome  # noqa: PLC0415

    if report.outcome is HandlingOutcome.EXECUTED:
        return True
    after_release = (HandlingOutcome.MOTION_REFUSED, HandlingOutcome.CAMERA_WORLD_UNAVAILABLE)
    return report.outcome in after_release and len(report.poses) >= 3


# ---------------------------------------------------------------------------------------------------------------------
# The verb
# ---------------------------------------------------------------------------------------------------------------------


def run_task(service: Any, plan: TaskPlan, *, hooks: TaskHooks, poses: "Mapping[str, JointPositions]",
             locators: "Maybe[Sequence[Any]]" = UNSET) -> TaskReport:
    """Run ``plan`` on the connected cell ``service`` holds, saying each step to ``hooks``; what it came to.

    ``poses`` are the taught poses the plan may name (place and return), joints by name. ``locators`` find a camera
    place's target: unset, the service's own (``place_target.locators_for_service``). See the module docstring for the
    order and the rules. Raises :class:`TaskRefused` before anything is commanded for a request this cell cannot do,
    and a programmer's error; every other end is the report's.
    """
    return _Task(service, plan, hooks, poses, locators).run()


class _Ended(Exception):
    """How a step ends the task: the report travels up to :meth:`_Task.run`."""

    def __init__(self, report: TaskReport) -> None:
        super().__init__(report.sentence)
        self.report = report


class _Task:
    """One run of :func:`run_task`: the cell, the plan, and what happened so far."""

    def __init__(self, service: Any, plan: TaskPlan, hooks: TaskHooks, poses: "Mapping[str, JointPositions]",
                 locators: "Maybe[Sequence[Any]]") -> None:
        self.service = service
        self.plan = plan
        self.hooks = hooks
        self.poses = dict(poses)
        self.chosen_locators = locators
        orchestrator = service.runtime.orchestrator
        self.orchestrator = orchestrator
        self.arm = orchestrator.arm
        self.gripper = getattr(orchestrator, "gripper", None)
        self.standoff_mm = float(getattr(getattr(orchestrator, "policy", None), "standoff_mm", 80.0) or 80.0)
        self.natural = _natural_axis(getattr(orchestrator, "natural_closing_axis", None))
        tree = getattr(self.arm, "config", None)
        self.tree = tree
        self.part_bottom_mm = _declared_support_mm(tree)
        self.length_mm = _declared_length_mm(tree)
        place = plan.place
        self.air_mm = float(place.air_mm) if place.air_mm is not None else rim_clearance_mm(tree)
        self.locators: list[Any] = []
        # The bin the survey found, which every check measures against, and the sighting of it the drops go to now.
        self.surveyed: KeptTarget | None = None
        self.kept: KeptTarget | None = None
        self.place_tcp: "Pose | None" = None
        self.robot: Any = None
        self.moved = False
        self.at_return = False
        self.parts_placed = 0
        self.picks = 0
        self.succeeded = 0
        self.holding = False
        self.last_report: Any = None
        self.part_number = 0
        self.part_since = time.monotonic()

    # ---- the whole task ---------------------------------------------------------------------------------------

    def run(self) -> TaskReport:
        self._refuse_the_plan()
        waiting = str(getattr(self.service, "stopped_where_the_arm_stands", "") or "")
        if waiting:
            return self._report(TaskStop.RECOVERY_NEEDS_PERSON, (
                "a recovery of an earlier pick stopped where the arm stands, and a person decides first; the task "
                f"started nothing and commanded nothing: {waiting}"))
        stopped = self._may_not_move()
        if stopped is not None:
            return self._report(*stopped)
        self._refuse_what_this_cell_cannot_do()
        self.robot = self._robot()
        try:
            self._screen()
            with ExitStack() as undo:
                # Nobody is asked where a toggle's jaws stand on this thread while the task runs, in a pick, a place,
                # a put back or a return alike: a question the hand would ask is a refusal instead.
                undo.enter_context(self._asking_nobody())
                self._set_up(undo)
                return self._drive()
        except _Ended as ended:
            return ended.report

    def _drive(self) -> TaskReport:
        # The hand before the task's first motion of any kind: a Restart's move home and a wrist survey carry whatever
        # the jaws hold as a pick's motions do. Read again before every pick below.
        self._hand_before("before its first motion")
        if self.plan.first_motion == "return":
            self._return()
        if self.plan.place.camera is not None:
            if self.hooks.stop_after_part():
                # No part is in hand yet: the task ends here, back at its return pose where it moved at all.
                return self._finish(TaskStop.STOPPED_AFTER_PART, "the task stopped before its first pick, as the "
                                                                 "operator asked; nothing was picked")
            self._survey()
        once = self.plan.scope == "once"
        empty = failed = 0
        while True:
            if self.hooks.stop_after_part():
                return self._finish(TaskStop.STOPPED_AFTER_PART, "the task stopped after the part, as the operator "
                                                                 "asked")
            self._go_on_or_end()
            self._hand_before("before the pick")
            if self.kept is not None and not self.kept.on_the_wrist:
                self._check_a_fixed_cameras_target()
            if self.part_number != self.parts_placed + 1:
                self.part_number, self.part_since = self.parts_placed + 1, time.monotonic()
            self.picks += 1
            self._say(TaskEvent.PART_STARTED, f"Looking for part {self.part_number}"
                                              + (" of 1." if once else "."),
                      part=self.part_number, of=1 if once else None, pick=self.picks)
            report, failures_before = self._pick()
            self._stop_on_what_the_pick_said(report)
            kept_out = _only_kept_out(report)
            if _found_nothing(report) or kept_out:
                if self._detector_failures() > failures_before:
                    return self._problem(TaskStop.DETECTOR_FAILED, (
                        "the detector failed while the pick looked, so 'nothing found' is no answer: "
                        + _summary(report)))
                empty += 1
                self._say(TaskEvent.NOTHING_FOUND,
                          ("Only parts this task keeps out were seen" if kept_out else "Nothing matching was seen")
                          + f" ({empty} empty look(s) in a row).",
                          part=self.part_number, empty_in_a_row=empty, only_excluded=kept_out)
                if empty >= self.plan.empty_looks_to_end:
                    return self._finish(TaskStop.NOTHING_LEFT, (
                        f"nothing to pick is left: {empty} looks in a row saw {self._what()} nowhere, or only where "
                        f"the task keeps out; {self.parts_placed} part(s) placed"))
                continue
            # The two rows count apart (build plan 1.3.3): a failed pick leaves the row of empty looks standing, as an
            # empty look leaves the row of failures; only a part placed starts both again. Alternating the two still
            # ends, at the empty row's limit.
            if not report.succeeded:
                held = _part_may_be_held(self.arm, self.gripper)
                if held:
                    self.holding = True
                    return self._problem(TaskStop.PART_STILL_HELD, (
                        f"the pick failed and {held}, so no further pick starts with it in the jaws: "
                        + _summary(report)))
                failed += 1
                if failed >= self.plan.max_failed_in_a_row:
                    return self._problem(TaskStop.FAILED_IN_A_ROW, (
                        f"{failed} picks in a row failed, so the task stops where the arm stands: " + _summary(report)))
                continue
            self.succeeded += 1
            empty = failed = 0
            unmodelled = _carried_part_unmodelled(self.arm)
            if unmodelled:
                # Between the close and the release only the planner holds the part: a part it declined would be
                # carried to the drop, or back, with every motion planned as if the hand were empty.
                self.holding = True
                return self._problem(TaskStop.PART_STILL_HELD, (
                    f"the pick gripped the part and {unmodelled}, so every carry, the place and a put back would be "
                    "planned as if the hand were empty; the task stops where the arm stands with the part in the "
                    "jaws, and a person takes it"))
            self._place(report)
            if once:
                return self._finish(TaskStop.FINISHED, "the part is placed and the arm is back "
                                    + ("home" if self.plan.return_to == "home" else f"at {self._return_said()}"))
            if self.parts_placed >= self.plan.max_parts:
                return self._finish(TaskStop.PART_LIMIT, (
                    f"{self.parts_placed} parts placed, the most one task places; start another to go on"))

    # ---- refusals before anything moves -------------------------------------------------------------------

    def _refuse_the_plan(self) -> None:
        place = self.plan.place
        if place.pose is not None and place.pose not in self.poses:
            raise TaskRefused("unknown_pose", f"the place names the pose {place.pose!r}, and no pose of that name was "
                                              f"taught ({', '.join(sorted(self.poses)) or 'none is'})")
        if self.plan.return_to != "home" and self.plan.return_to not in self.poses:
            raise TaskRefused("unknown_pose", f"the task returns to the pose {self.plan.return_to!r}, and no pose of "
                                              "that name was taught")
        axis = self.plan.options.closing_axis
        if axis is not None:
            from src.geometry.closing_axis import closing_axis_of  # noqa: PLC0415

            try:
                closing_axis_of(axis)
            except (TypeError, ValueError) as exc:
                raise TaskRefused("closing_axis_refused", str(exc)) from None

    def _refuse_what_this_cell_cannot_do(self) -> None:
        from src.robot.core.arm_capabilities import CarriesPayload  # noqa: PLC0415
        from src.robot.execution.motion import route_of  # noqa: PLC0415

        route = route_of(self.arm)
        if not route.runs:
            # Robot.place and Robot.home refuse this arm before any command: a part picked here could not be set down.
            raise TaskRefused("route_refused", (
                "a task sets down and returns from every part it picks, and the place and the return refuse this arm "
                f"before any command, so a part picked here would stay in the jaws: {route.reason}"))
        if isinstance(self.arm, CarriesPayload):
            declined = self.arm.payload_declined_reason()
            if declined is not None:
                raise TaskRefused("carried_part_not_modelled", (
                    f"a task carries every part it picks, and this arm models no carried part ({declined}): between "
                    "the close and the release only the planner holds it, so nothing would keep it clear; declare "
                    "safety.planning_world.payload.length_mm in the cell profile"))
        if not self.plan.object.strip() and not self.plan.options.pick_anything:
            grounds = getattr(self.service, "grounds_a_phrase", None)
            if not callable(grounds) or grounds():
                raise TaskRefused("object_required", (
                    "the task names no object, and this cell's detector grounds a phrase: an empty one picks anything "
                    "the camera sees, bin walls included; name the object, or say pick_anything"))
        axis = self.plan.options.closing_axis
        refusal_of = getattr(self.service, "closing_axis_refusal", None)
        if axis is not None and callable(refusal_of):
            why = refusal_of(axis)
            if isinstance(why, str) and why:
                raise TaskRefused("closing_axis_refused", why)
        if self.plan.options.both_faces:
            from src.robot.grasping.loop.pick_loop import judged_faces_turned_away  # noqa: PLC0415

            # The pick refuses the same, but only once it runs: after a Restart's move home or a survey's looks.
            turned = judged_faces_turned_away(self.orchestrator)
            if turned:
                raise TaskRefused("bad_request", turned)
        push = self.plan.options.push_mm
        distance = getattr(self.service, "push_distance", None)
        if push is not None and callable(distance):
            try:
                distance(push)
            except ValueError as exc:
                raise TaskRefused("push_distance_refused", str(exc)) from None
        if self.plan.place.camera is not None:
            self.locators = self._locators()

    def _locators(self) -> list[Any]:
        if chosen(self.chosen_locators):
            locators = list(self.chosen_locators)
        else:
            try:
                locators = list(locators_for_service(self.service))
            except ValueError as exc:
                raise TaskRefused("camera_target_unavailable", (
                    f"a camera place finds its target with the cell's own camera, and this cell has none to lend: "
                    f"{exc}")) from None
        if not locators:
            raise TaskRefused("camera_target_unavailable", "a camera place was handed no camera to find its target with")
        if any(bool(getattr(locator, "on_the_wrist", False)) for locator in locators):
            world = getattr(self.arm, "live_planner_world", None)
            if not (callable(getattr(world, "hold_pick_views", None))
                    and callable(getattr(world, "forget_pick_views", None))):
                raise TaskRefused("camera_target_unavailable", (
                    "a wrist camera's place holds every frame of the bin's look while the part goes in, and this arm "
                    "has no live planner world to hold them; nothing would keep the bin's walls in the world"))
        return locators

    # ---- the screen of the taught poses (build plan 1.3.3 [screen]) ----------------------------------------------

    def _screen(self) -> None:
        refused: list[str] = []
        place = self.plan.place.pose
        if place is not None:
            screen = self._screen_place(place)
            if screen.is_error:
                refused.append(f"the place pose {place!r}: {screen.detail}")
        if self.plan.return_to != "home":
            name = self.plan.return_to
            screen = screen_joints(self.arm, self.poses[name])
            self._say(TaskEvent.POSE_SCREENED, _screened_said("return", name, screen.verdict), pose=name,
                      role="return", verdict=screen.verdict, detail=screen.detail, nearby_deg=screen.nearby_deg)
            if screen.is_error:
                refused.append(f"the return pose {name!r}: {screen.detail}")
        if refused:
            self._end(TaskStop.POSE_REFUSED, "no move goes to a pose the task uses, so it moved nothing: "
                      + "; ".join(refused))

    def _screen_place(self, name: str) -> Any:
        import numpy as np  # noqa: PLC0415

        from src.geometry import Frame, Pose  # noqa: PLC0415
        from src.robot.execution.place_target import Screen  # noqa: PLC0415

        try:
            at = self.arm.fk(self.poses[name])
        except (RobotError, RuntimeError, OSError, ValueError) as exc:
            screen = Screen(None, "guard_refused", f"the arm cannot say where the pose puts the tool: "
                                                   f"{type(exc).__name__}: {exc}")
        else:
            self.place_tcp = at
            # Where the tool stands for the longest part the cell declares: the taught pose raised by its hang, and the
            # configuration the arm would choose for it (the place goes to a pose, not to the taught joints).
            rise = self.length_mm or 0.0
            position = np.asarray(at.position_mm, dtype=np.float64) + np.array([0.0, 0.0, rise])
            raised = Pose(position_mm=position, quaternion_xyzw=np.asarray(at.quaternion_xyzw, dtype=np.float64),
                          frame=Frame.BASE, label=f"pose {name}, raised")
            screen = screen_pose(self.arm, raised)
            if rise <= 0.0:
                screen = dataclasses.replace(screen, detail=f"{screen.detail} (screened at the taught pose itself: the "
                                                            "cell declares no carried part's hang)")
        self._say(TaskEvent.POSE_SCREENED, _screened_said("place", name, screen.verdict), pose=name,
                  role="place", verdict=screen.verdict, detail=screen.detail, nearby_deg=screen.nearby_deg)
        return screen

    # ---- the setup, put back as found when the task ends -------------------------------------------------------

    def _set_up(self, undo: ExitStack) -> None:
        options = self.plan.options
        started: dict[str, Any] = {}
        if options.push_mm is not None:
            started["push_mm"] = options.push_mm
        if options.critical_parts is not None:
            started["critical_parts"] = bool(options.critical_parts)
        actions = run_recovery_actions(self.service, rescan=options.rescan, push=options.push, clear=options.clear)
        if actions is not None:
            started["recovery_actions"] = actions
            if options.push is False and options.clear is True and options.critical_parts is None:
                started["critical_parts"] = True
            elif options.push is True and options.critical_parts is None:
                started["critical_parts"] = False
        # A blocker goes where the parts go only where every part goes there: a task that names an object sets one
        # aside, as it is not the part it asked for (the owner, 2026-10-06).
        into = options.blocker_into_the_place
        if into is None:
            into = bool(getattr(getattr(getattr(self.service, "effective_config", None), "recovery_orchestrator", None),
                                "blocker_into_the_place", True))
        if into and not self.plan.object.strip():
            started["blocker_is_the_pick"] = True
        try:
            self.service.start_campaign(**started)
        except ValueError as exc:
            raise TaskRefused("push_distance_refused", str(exc)) from None
        zones = self.service.campaign.zones
        undo.callback(_quietly, "forgetting the regions the task kept out", zones.forget_regions)
        named = self.plan.object.strip()
        if named and not _grounds_a_phrase(self.service):
            previous = self.service.set_prompt(self.plan.object)
            undo.callback(_quietly, "putting the cell's prompt back", self.service.set_prompt, previous)
        elif _grounds_a_phrase(self.service):
            # Every part the task asks for in a box of its own: "each separate green part" grounds every green part, the
            # phrase alone one of them. The detector echoes the phrase, which maps onto the object the task named; a
            # task that names none takes every part it grounds, called "object" where the detector's words allow.
            from src.robot.execution.autonomous_grasp.prompt import PickPrompt  # noqa: PLC0415

            prompt = (PickPrompt(phrase=f"{EACH_SEPARATE} {named}", target_label=named, object_labels=(named,))
                      if named else PickPrompt(phrase=EVERY_PART_PHRASE, target_label=None, object_labels=("object",)))
            previous = self.service.set_prompt(prompt)
            undo.callback(_quietly, "putting the cell's prompt back", self.service.set_prompt, previous)
        if options.closing_axis is not None:
            try:
                axis = self.service.set_closing_axis(options.closing_axis)
            except (TypeError, ValueError) as exc:  # the cell's own rule beside it (align_closing_to_base_x)
                raise TaskRefused("closing_axis_refused", str(exc)) from None
            undo.callback(_quietly, "putting the closing axis back", self.service.set_closing_axis, axis)
        if options.overlay:
            was = getattr(self.service, "debug_image_rendering_enabled", None)
            self.service.enable_debug_image_rendering(True)
            if isinstance(was, bool):
                undo.callback(_quietly, "putting the overlay switch back", self.service.enable_debug_image_rendering,
                              was)
        self.service.set_cancel_check(self._cancelled)
        undo.callback(_quietly, "taking the cancel check back", self.service.set_cancel_check, None)
        if self.plan.place.pose is not None and self.plan.scope == "until_empty" and self.place_tcp is not None:
            from src.robot.grasping.recovery.exclusion_zones import ExclusionRegion  # noqa: PLC0415

            if _kinematic(self.arm):
                drop_xy = (float(self.place_tcp.position_mm[0]), float(self.place_tcp.position_mm[1]))
                zones.keep_out_region(ExclusionRegion.circle(
                    drop_xy, self.plan.pose_keep_out_mm, reason=f"the drop at pose {self.plan.place.pose!r}"))
            else:
                # An fk that answers where the arm stands says nothing of where the pose puts the tool: a circle about
                # it would keep out whatever lies under the arm now (the rehearsal's one part), not the drop.
                logger.warning(
                    "task: no keep-out circle about the drop at pose %r: %s has no forward kinematics of its own "
                    "(has_native_fk is False), so where the pose puts the tool is not known, and a part set down "
                    "there may be picked again", self.plan.place.pose, type(self.arm).__name__)

    def _cancelled(self) -> bool:
        """What the service and its pick loop ask before every attempt and look: halt, or a cell taken down; never the
        operator's "stop after this part", which lets the pick in flight finish its attempts."""
        return self._may_not_move() is not None

    # ---- before and after every motion ------------------------------------------------------------------------

    def _halted_why(self) -> str:
        if self.hooks.halted():
            return "halt now was pressed; nothing more is commanded, the jaws included"
        state_of = getattr(self.arm, "halt_state", None)
        if not callable(state_of):
            return ""
        try:
            state = state_of()
        except Exception as exc:  # noqa: BLE001 (a latch that cannot be read is no latch that is open)
            return f"the arm's halt latch could not be read ({type(exc).__name__}: {exc})"
        if state is None:
            return ""
        reason = getattr(state, "reason", "")
        return f"the arm is halted ({reason or 'its halt latch is set'}); nothing more is commanded, the jaws included"

    def _may_not_move(self) -> "tuple[TaskStop, str] | None":
        gone = self.hooks.abandoned()
        if gone:
            return TaskStop.DISCONNECTED, str(gone)
        halted = self._halted_why()
        if halted:
            return TaskStop.HALTED, halted
        return None

    def _go_on_or_end(self) -> None:
        stopped = self._may_not_move()
        if stopped is not None:
            self._problem(*stopped)

    def _go_on_after_the_release(self, sentence: str) -> None:
        """Before the return that follows a motion refused once the part was let go (a place's or a put back's line
        out): end the task where the arm stands if it may not move any more or the controller is stopped, so no return
        is sent to a controller a protective stop took. A refusal of the planner or the guard alone still tries the
        return (build plan 1.3.3)."""
        self._go_on_or_end()
        controller = getattr(self.service, "controller_refusal", None)
        said = controller() if callable(controller) else ""
        if isinstance(said, str) and said:
            self._problem(TaskStop.CONTROLLER_STOPPED, f"{sentence}; {said}")

    def _hand_before(self, when: str) -> None:
        """End the task with nothing moved where the hand may not take its next step: a toggle's count nobody can
        vouch for, or that says closed (``jaws_before_a_pick``, ``hand_needs_person``); then a part the hand may still
        hold, a hand that measures one or a planner that still models one (``part_still_held``). Reads only."""
        hand = jaws_before_a_pick(self.gripper)
        if hand:
            self._problem(TaskStop.HAND_NEEDS_PERSON, hand.replace("before the pick", when))
        held = _part_may_be_held(self.arm, self.gripper)
        if held:
            self.holding = True
            self._problem(TaskStop.PART_STILL_HELD, (
                f"{held}; the task stops {when} and moves nothing more, as a part may be in the jaws: a person takes "
                "it out of the hand and says so, then starts the task again"))

    def _after_a_failed_motion(self, default: TaskStop, sentence: str) -> TaskReport:
        """End the task where the arm stands after a motion that failed: a cell taken down, a halt, a stopped controller
        first, each said as itself, else ``default``."""
        stopped = self._may_not_move()
        if stopped is not None:
            return self._problem(*stopped)
        controller = getattr(self.service, "controller_refusal", None)
        said = controller() if callable(controller) else ""
        if isinstance(said, str) and said:
            return self._problem(TaskStop.CONTROLLER_STOPPED, f"{sentence}; {said}")
        return self._problem(default, sentence)

    # ---- the pick ---------------------------------------------------------------------------------------------

    def _looks(self) -> Any:
        from src.robot.execution.looks import HOME  # noqa: PLC0415

        configured = configured_looks_of(self.service)
        if configured:
            return configured
        return HOME if getattr(self.service, "perceives_from_the_wrist", False) is True else None

    def _pick(self) -> "tuple[Any, int]":
        options = self.plan.options
        keywords: dict[str, Any] = {}
        look = self._looks()
        if look is not None:
            keywords["look"] = look
        if not options.multi_view:
            keywords["multi_view"] = False
        if options.both_faces:
            keywords["both_faces"] = True
        before = self._detector_failures()
        report = self.service.pick(**keywords)  # inside the task's asking_nobody (run)
        self.moved, self.at_return = True, False
        self.last_report = report
        if options.record_views:
            report = self._with_views(report)
            self.last_report = report
        if bool(getattr(report, "succeeded", False)):
            self.holding = True
        self.hooks.pick_done(self.part_number, self.picks, report)
        return report, before

    def _asking_nobody(self) -> Any:
        from src.robot.core.gripper import toggle_without_sensor_of  # noqa: PLC0415

        toggle = toggle_without_sensor_of(self.gripper)
        asking = getattr(toggle, "asking_nobody", None) if toggle is not None else None
        return asking("a task never asks a person where the jaws stand; the step stops here, and the task with it") \
            if callable(asking) else nullcontext()

    def _with_views(self, report: Any) -> Any:
        """``report`` with where its looks were kept in ``telemetry['views_file']``, where they were."""
        from src.robot.execution.autonomous_grasp.report import AutonomousGraspReport  # noqa: PLC0415

        where = keep_pick_views(self.service, report, name=f"task-part{self.part_number}-pick{self.picks}")
        if not where or not isinstance(report, AutonomousGraspReport):
            return report
        return dataclasses.replace(report, telemetry={**dict(report.telemetry), "views_file": where})

    def _detector_failures(self) -> int:
        count = getattr(self.service, "detector_failures", None)
        value = count() if callable(count) else 0
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0

    def _stop_on_what_the_pick_said(self, report: Any) -> None:
        """End the task where the arm stands on what a pick reports rather than raises, read in this order: a cell
        taken down, a halt (a pick a halt cut short reports what the halt did to it, never why it ended), a fault, a
        stopped controller, a hand or a recovery that needs a person."""
        gone = self.hooks.abandoned()
        if gone:
            self._problem(TaskStop.DISCONNECTED, f"{gone} The last pick said: {_summary(report)}")
        halted = self._halted_why()
        if halted:
            self._problem(TaskStop.HALTED, f"{halted}. The last pick said: {_summary(report)}")
        fault = getattr(report, "fault", None)
        if fault is not None:
            self._problem(TaskStop.CELL_FAULT, f"a fault of the cell ended the pick ({type(fault).__name__}: {fault})")
        if getattr(report, "controller_stopped", False) is True:
            self._problem(TaskStop.CONTROLLER_STOPPED, (
                "the controller cannot move (a protective or emergency stop, or a power-off), so the task stops; a "
                "person clears the stop at the pendant: " + _summary(report)))
        hand = getattr(report, "gripper_fault", "")
        if isinstance(hand, str) and hand:
            self._problem(TaskStop.HAND_NEEDS_PERSON, f"the gripper needs a person, so the task stops: {hand}")
        if getattr(report, "needs_person", False) is True:
            self._problem(TaskStop.RECOVERY_NEEDS_PERSON, (
                "a recovery stopped where the arm stands, so the task stops; nothing more is commanded and a person "
                "decides what happens next: " + _summary(report)))

    # ---- the camera's target --------------------------------------------------------------------------------

    def _survey(self) -> None:
        from src.robot.execution.looks import HOME, look_label, looks_of  # noqa: PLC0415

        phrase = str(self.plan.place.camera)
        configured = self._looks()
        looks = tuple(looks_of(configured)) if configured is not None else (HOME,)
        if not self.plan.options.multi_view:
            looks = looks[:1]
        wrist = any(bool(getattr(locator, "on_the_wrist", False)) for locator in self.locators)
        self._say(TaskEvent.SURVEY_STARTED, f"Looking for the {phrase} "
                  + (f"from {len(looks)} look(s) " if wrist else "with the fixed camera ") + "before the first pick.",
                  phrase=phrase, looks=[look_label(look) for look in looks] if wrist else [])
        before = self._detector_failures()
        self._go_on_or_end()
        stop_asked: list[bool] = []

        def may_move() -> str:
            # Asked before every look: halt and a cell taken down first, then "stop after this part", which has no part
            # in hand to finish here and so ends the survey before its next look.
            stopped = self._may_not_move()
            if stopped is not None:
                return stopped[1]
            if self.hooks.stop_after_part():
                stop_asked.append(True)
                return "the operator asked the task to stop after this part"
            return ""

        try:
            found = survey(self.arm, self.locators, phrase, object_phrase=self.plan.object, looks=looks,
                           may_move=may_move)
        except (RobotError, RuntimeError, OSError, ValueError) as exc:
            self.moved = self.moved or wrist
            self._problem(TaskStop.CELL_FAULT, f"the camera could not look for the {phrase} ({type(exc).__name__}: "
                                               f"{exc})")
        if wrist and (found.looks_tried or found.refused or found.stopped is not None):
            self.moved, self.at_return = True, False
        if found.halted:
            # Ended before a motion: said as what stopped it, a halt or a Disconnect before the operator's stop.
            stopped = self._may_not_move()
            if stopped is None and stop_asked:
                self._finish(TaskStop.STOPPED_AFTER_PART, f"the task stopped while it looked for the {phrase}, as the "
                                                          "operator asked; nothing was picked")
            self._problem(*(stopped or (TaskStop.HALTED, found.halted)))
        if found.stopped is not None:
            self._after_a_failed_motion(TaskStop.CELL_FAULT, (
                f"the arm did not reach look {found.stopped_look} while it looked for the {phrase}, and nothing else "
                f"was commanded after it: {found.stopped.status.value}: {found.stopped.message}"))
        if found.kept is None:
            if self._detector_failures() > before:
                self._problem(TaskStop.DETECTOR_FAILED, f"the detector failed while the task looked for the {phrase}, "
                                                        "so 'not found' is no answer")
            self._say(TaskEvent.TARGET_MISSING, f"No {phrase} was seen from "
                      + (f"{len(found.looks_tried)} look(s)" if wrist else "the fixed camera")
                      + ": the task picks nothing.", phrase=phrase, looks_tried=list(found.looks_tried))
            self._finish(TaskStop.TARGET_NOT_FOUND, (
                f"no {phrase} was seen from {', '.join(found.looks_tried) or 'where the camera stands'}; nothing was "
                "picked"))
        kept = found.kept
        assert kept is not None  # narrowed: a survey with no target ended the task above
        said = f"Found the {kept.label}" + ("" if kept.score is None else f" ({kept.score:.2f})")
        said += f" at look {kept.look_label}" if kept.look_label else " with the fixed camera"
        if found.parts_seen is not None:
            said += f"; {found.parts_seen} {self.plan.object}(s) seen outside it"
        self._say(TaskEvent.TARGET_FOUND, said + ".", target=kept.to_dict(), look=kept.look_label,
                  parts_seen=found.parts_seen, image_png=kept.image_png)
        nominal = nominal_drop(self.arm, kept, hang_mm=self.length_mm or 0.0, air_mm=self.air_mm,
                               standoff_mm=self.standoff_mm, natural_axis=self.natural)
        if not nominal.ok:
            self._finish(TaskStop.TARGET_UNREACHABLE, f"{nominal.reason}; nothing was picked")
        self.kept = self.surveyed = kept
        self.service.campaign.zones.keep_out_region(kept.keep_out_region())

    def _check(self) -> TargetCheck:
        # Against the bin the survey found, never a later sighting: a bin that creeps is followed only within the bound
        # of where it was found, and its size and rim are that bin's.
        assert self.kept is not None and self.surveyed is not None  # only a camera place checks its target
        before = self._detector_failures()
        try:
            check = recheck(self.locators, self.surveyed)
        except (RobotError, RuntimeError, OSError, ValueError) as exc:
            self._problem(TaskStop.CELL_FAULT, f"the camera could not look at the {self.kept.label} again "
                                               f"({type(exc).__name__}: {exc})")
        if check.seen is None and self._detector_failures() > before:
            self._problem(TaskStop.DETECTOR_FAILED, f"the detector failed while the task looked at the "
                                                    f"{self.kept.label} again, so 'not seen' is no answer")
        image = check.seen.image_png if check.seen is not None else None
        self._say(TaskEvent.TARGET_CHECKED, _checked_said(self.kept.label, check), moved_mm=check.moved_mm,
                  followed=check.followed, target=None if check.seen is None else check.seen.to_dict(),
                  image_png=image)
        if check.kept is not None:
            zones = self.service.campaign.zones
            zones.forget_regions()
            zones.keep_out_region(check.kept.keep_out_region())
            self.kept = check.kept
        return check

    def _check_a_fixed_cameras_target(self) -> None:
        """A fixed camera's target is looked at again before each pick, while the arm stands clear of its view."""
        assert self.kept is not None
        check = self._check()
        if check.kept is None:
            self._say(TaskEvent.TARGET_LOST, f"{_lost_said(self.kept.label, check)}: the task ends before the next "
                                             "pick.", look=None, why=check.why)
            self._finish(TaskStop.TARGET_LOST, f"the {self.kept.label} was lost before the pick: {check.render()}")

    # ---- the place ------------------------------------------------------------------------------------------

    def _place(self, report: Any) -> None:
        if self.plan.place.pose is not None:
            name = self.plan.place.pose
            drop = pose_drop(self.arm, self.poses[name], report.grasp_pose, part_bottom_mm=self.part_bottom_mm,
                             length_mm=self.length_mm, standoff_mm=self.standoff_mm)
            self._said_drop(drop)
            if not drop.ok:
                self._put_back_and_ask(report, TaskStop.POSE_REFUSED, f"the part was not set down at pose {name!r}: "
                                                                      f"{drop.reason}")
            self._place_at(drop, f"pose:{name}")
            return
        kept = self.kept
        assert kept is not None  # a camera place found its target before its first pick
        if kept.look is None:  # a fixed camera's: checked before the pick, while the arm stood out of its view
            self._drop_into(report)
            return
        from src.robot.core.keep_out import holding_views  # noqa: PLC0415
        from src.robot.execution.generated_view import refused_before_sending  # noqa: PLC0415
        from src.robot.execution.looks import move_to_look  # noqa: PLC0415
        from src.robot.execution.motion import MotionOutcome  # noqa: PLC0415

        self._say(TaskEvent.CARRY_STARTED, f"Carrying the part to look {kept.look_label} to check the "
                                           f"{kept.label} before the drop.", to_look=kept.look_label)
        self._go_on_or_end()
        moved = move_to_look(self.arm, kept.look)
        self.at_return = False
        if not moved.ok:
            if moved.outcome is MotionOutcome.CAMERA_WORLD_UNAVAILABLE:
                self._problem(TaskStop.CELL_FAULT, f"a camera could not vouch for the cell on the way to look "
                                                   f"{kept.look_label}: {moved.message}")
            if refused_before_sending(moved):
                self._go_on_or_end()
                self._put_back_and_ask(report, TaskStop.TARGET_UNREACHABLE, (
                    f"the arm could not carry the part to look {kept.look_label}, where the {kept.label} is checked: "
                    f"{moved.status.value}: {moved.message}"))
            self._after_a_failed_motion(TaskStop.PART_STILL_HELD, (
                f"the carry to look {kept.look_label} failed with the part in the jaws, and the arm may have moved "
                f"part of the way: {moved.status.value}: {moved.message}"))
        with holding_views(self.arm) as held:
            if not held:
                # Not seen: the bin cannot be checked against frames the world does not hold. The console's three
                # words for a lost target (not_seen, moved_too_far, footprint_changed); the sentence says the frames.
                self._say(TaskEvent.TARGET_LOST, f"The frames of look {kept.look_label} could not be held in the "
                                                 "arm's world, so the bin is not checked there: the part goes back "
                                                 "where it was grasped.", look=kept.look_label, why="not_seen")
                self._put_back_and_ask(report, TaskStop.TARGET_LOST, (
                    f"the arm's world could not hold the frames of look {kept.look_label}, so the {kept.label}'s walls "
                    "would not be in the world the part goes in against"))
            check = self._check()
            if check.kept is None:
                self._say(TaskEvent.TARGET_LOST, f"{_lost_said(kept.label, check)}: the part goes back where it was "
                                                 "grasped.", look=kept.look_label, why=check.why)
                self._put_back_and_ask(report, TaskStop.TARGET_LOST, f"the {kept.label} was lost at the drop: "
                                                                     f"{check.render()}")
            self._drop_into(report)

    def _drop_into(self, report: Any) -> None:
        kept = self.kept
        assert kept is not None
        grasp = report.grasp_pose
        if grasp is not None:
            cloud = getattr(getattr(getattr(self.service, "looked_around", None), "judged", None),
                            "target_cloud_base_mm", None)
            drop = drop_plan(self.arm, kept, grasp, part_bottom_mm=self.part_bottom_mm, air_mm=self.air_mm,
                             standoff_mm=self.standoff_mm, natural_axis=self.natural, part_cloud_mm=cloud)
        elif self.length_mm is not None:
            drop = nominal_drop(self.arm, kept, hang_mm=self.length_mm, air_mm=self.air_mm,
                                standoff_mm=self.standoff_mm, natural_axis=self.natural)
        else:
            drop = DropPlan(kind="camera", pose=None, standoff_mm=self.standoff_mm, rim_mm=kept.rim_mm,
                            air_mm=self.air_mm, refusal="unreachable", reason=(
                                "the pick reported no grasp pose and safety.planning_world.payload.length_mm is "
                                "undeclared, so how far the part hangs is unknown and nothing is set down blind"))
        self._said_drop(drop)
        if not drop.ok:
            stop = TaskStop.PART_DOES_NOT_FIT if drop.refusal == "does_not_fit" else TaskStop.TARGET_UNREACHABLE
            self._put_back_and_ask(report, stop, f"the part was not set down in the {kept.label}: {drop.reason}")
        self._place_at(drop, f"target:{self.plan.place.camera}")

    def _said_drop(self, drop: DropPlan) -> None:
        self._say(TaskEvent.DROP_PLANNED, _drop_said(drop, self.kept.label if self.kept is not None else ""),
                  **drop.to_event())

    def _place_at(self, drop: DropPlan, where: str) -> None:
        from src.robot.execution.handling import HandlingOutcome  # noqa: PLC0415

        assert drop.pose is not None
        self._go_on_or_end()
        kind, _, name = where.partition(":")
        self._say(TaskEvent.PLACE_STARTED, f"Placing the part at {name!r}." if kind == "pose"
                  else f"Placing the part in the {self.kept.label if self.kept is not None else name}.", place=where)
        placed = self.robot.place(drop.pose, standoff_mm=self.standoff_mm)
        self.at_return = False
        if not place_released(placed):
            self._say(TaskEvent.PLACE_FAILED, f"The place did not let the part go: {placed.outcome.value}: "
                                              f"{placed.message}", outcome=placed.outcome.value,
                      message=placed.message)
            if placed.outcome in (HandlingOutcome.GRIPPER_FAULT, HandlingOutcome.RELEASE_NOT_CONFIRMED):
                stopped = self._may_not_move()
                if stopped is not None:
                    self._problem(*stopped)
                self._problem(TaskStop.HAND_NEEDS_PERSON, (
                    f"the place's release needs a person ({placed.outcome.value}: {placed.message}); the part may "
                    "still be in the jaws, and the arm stays where it stands"))
            self._after_a_failed_motion(TaskStop.PART_STILL_HELD, (
                f"the place was refused before its release ({placed.outcome.value}: {placed.message}), so the part is "
                "still in the jaws and the arm stays where it stands"))
        self.parts_placed += 1
        self.holding = False
        line_out = placed.outcome is not HandlingOutcome.EXECUTED
        no_sensor = self._no_sensor(placed)
        self._say(TaskEvent.PLACED, "Placed: the jaws opened at the drop"
                  + (" (no sensor: the release is not measured)" if no_sensor else "")
                  + ("; the line out after it was refused, and the arm stands where the release left it" if line_out
                     else "") + ".",
                  outcome=placed.outcome.value, no_sensor=no_sensor, line_out_refused=line_out)
        if placed.outcome is HandlingOutcome.CAMERA_WORLD_UNAVAILABLE:
            self._problem(TaskStop.CELL_FAULT, f"a camera could not vouch for the cell on the place's line out, after "
                                               f"the part was let go: {placed.message}")
        if line_out:
            self._go_on_after_the_release(f"the part was let go at the drop and the line out after it was refused "
                                          f"({placed.message}); the arm stays where the release left it")
        self._return()
        self._said_part_finished(placed=True)

    def _put_back_and_ask(self, report: Any, stop: TaskStop, sentence: str) -> None:
        """Put the part back where it was gripped, return, and end the task asking (``stop``); a put back that does not
        let the part go ends it where the arm stands."""
        from src.robot.execution.handling import HandlingOutcome  # noqa: PLC0415

        self._go_on_or_end()
        back = self.service.put_back(report)
        self.at_return = False
        self._say(TaskEvent.PUT_BACK, "Put the part back where it was grasped." if place_released(back)
                  else f"The part could not be put back: {back.outcome.value}: {back.message}",
                  outcome=back.outcome.value)
        if not place_released(back):
            if back.outcome in (HandlingOutcome.GRIPPER_FAULT, HandlingOutcome.RELEASE_NOT_CONFIRMED):
                stopped = self._may_not_move()
                if stopped is not None:
                    self._problem(*stopped)
                self._problem(TaskStop.HAND_NEEDS_PERSON, (
                    f"{sentence}; putting the part back needs a person ({back.outcome.value}: {back.message})"))
            self._after_a_failed_motion(TaskStop.PART_STILL_HELD, (
                f"{sentence}; the part could not be put back ({back.outcome.value}: {back.message}), so it is still "
                "in the jaws and the arm stays where it stands"))
        self.holding = False
        if back.outcome is HandlingOutcome.CAMERA_WORLD_UNAVAILABLE:
            self._problem(TaskStop.CELL_FAULT, f"{sentence}; a camera could not vouch for the cell on the put back's "
                                               f"line out: {back.message}")
        if back.outcome is not HandlingOutcome.EXECUTED:
            self._go_on_after_the_release(f"{sentence}; the part was put back where it was gripped and the line out "
                                          f"after it was refused ({back.message}); the arm stays where the release "
                                          "left it")
        self._return()
        self._said_part_finished(placed=False)
        self._end(stop, f"{sentence}; the part was put back where it was gripped")

    def _said_part_finished(self, *, placed: bool) -> None:
        self._say(TaskEvent.PART_FINISHED, f"Part {self.part_number} " + ("is placed." if placed else "was not placed."),
                  part=self.part_number, placed=placed, duration_s=round(time.monotonic() - self.part_since, 3))

    def _no_sensor(self, placed: Any) -> bool:
        from src.robot.core.gripper import toggle_without_sensor_of  # noqa: PLC0415

        hand = getattr(placed, "hand", None)
        if hand is not None:
            return bool(getattr(hand, "no_sensor", False))
        return toggle_without_sensor_of(self.gripper) is not None

    # ---- the return ------------------------------------------------------------------------------------------

    def _return(self) -> None:
        from src.robot.execution.motion import MotionOutcome  # noqa: PLC0415

        self._go_on_or_end()
        to = self.plan.return_to
        self._say(TaskEvent.RETURN_STARTED, "Returning home." if to == "home" else f"Returning to {to!r}.", to=to,
                  note="")
        moved: "MotionReport" = self.robot.home() if to == "home" else self.robot.move_joints(self.poses[to])
        self.moved = True
        if moved.ok:
            self.at_return = True
            self._say(TaskEvent.RETURNED, "Back home." if to == "home" else f"Back at {to!r}.", to=to,
                      note=_motion_note(moved))
            return
        self.at_return = False
        self._say(TaskEvent.RETURN_FAILED, f"The return to {self._return_said()} did not run: {moved.status.value}: "
                                           f"{moved.message}", status=moved.status.value, message=moved.message)
        if moved.outcome is MotionOutcome.CAMERA_WORLD_UNAVAILABLE:
            self._problem(TaskStop.CELL_FAULT, f"a camera could not vouch for the cell on the return: {moved.message}")
        self._after_a_failed_motion(TaskStop.RETURN_FAILED, (
            f"the return to {self._return_said()} was refused ({moved.status.value}: {moved.message}), and the arm "
            "stays where it stands"))

    def _return_said(self) -> str:
        return "home" if self.plan.return_to == "home" else f"pose {self.plan.return_to!r}"

    # ---- ending ----------------------------------------------------------------------------------------------

    def _finish(self, stop: TaskStop, sentence: str) -> TaskReport:
        """End the task on a benign ``stop``: the arm moves to its return pose first where the task moved it and it is
        not standing there."""
        if self.moved and not self.at_return:
            self._return()
        return self._end(stop, sentence)

    def _problem(self, stop: TaskStop, sentence: str) -> TaskReport:
        """End the task where the arm stands, with nothing more commanded."""
        if not self.holding and _part_may_be_held(self.arm, self.gripper, read_hand=False):
            self.holding = True
        return self._end(stop, sentence)

    def _end(self, stop: TaskStop, sentence: str) -> TaskReport:
        raise _Ended(self._report(stop, sentence))

    def _report(self, stop: TaskStop, sentence: str) -> TaskReport:
        (logger.info if stop.returns else logger.error)("task %s: %s", stop.value, sentence)
        return TaskReport(stop=stop, sentence=sentence, parts_placed=self.parts_placed, picks=self.picks,
                          succeeded=self.succeeded, holding=self.holding, last_report=self.last_report,
                          at_return=self.at_return)

    # ---- helpers ---------------------------------------------------------------------------------------------

    def _say(self, name: TaskEvent, said: str, **data: Any) -> None:
        self.hooks.event(name, said=said.encode("ascii", "backslashreplace").decode("ascii"), **data)

    def _what(self) -> str:
        return repr(self.plan.object) if self.plan.object.strip() else "anything to pick"

    def _robot(self) -> Any:
        """The arm and the hand as the place and the return command them, as ``put_back`` builds its own: no lock of
        its own (the connect that brought the service up holds the cell's), the tree the arm keeps."""
        from src.robot.execution.robot import Robot  # noqa: PLC0415

        return Robot.from_parts(arm=self.arm, gripper=self.gripper, lock_key=None)


# ---------------------------------------------------------------------------------------------------------------------
# What a task reads off the cell
# ---------------------------------------------------------------------------------------------------------------------


def _quietly(what: str, call: Callable[..., Any], *args: Any) -> None:
    """A put back of the task's teardown: one that raises is logged, and the rest still run."""
    try:
        call(*args)
    except Exception as exc:  # noqa: BLE001 (teardown reports, it does not propagate)
        logger.warning("task: %s failed: %s: %s", what, type(exc).__name__, exc)


def _natural_axis(value: Any) -> Any:
    """The cell's natural closing axis as the pick loop holds it (a ``ClosingAxis``), read where it is a name."""
    if value is None:
        return None
    from src.geometry.closing_axis import ClosingAxis, closing_axis_of  # noqa: PLC0415

    if isinstance(value, ClosingAxis):
        return value
    try:
        return closing_axis_of(value)
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> "float | None":
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def _declared_support_mm(tree: Any) -> float:
    """The height no part stood below, a lower bound on a part's bottom: the lower of the container floor and the table
    the cell declares (a part stands on either), the table alone where no floor is declared, 0 where the tree says
    nothing. A bound too low only adds air under the part when it is let go."""
    support = getattr(getattr(tree, "grasping", None), "support", None)
    table = _number(getattr(support, "height_mm", None))
    floor = _number(getattr(getattr(support, "container", None), "floor_height_mm", None))
    declared = [value for value in (table, floor) if value is not None]
    return min(declared) if declared else 0.0


def _declared_length_mm(tree: Any) -> "float | None":
    """How far the cell declares its longest part hangs past the fingertips (``payload.length_mm``), or ``None``."""
    length = _number(getattr(getattr(getattr(getattr(tree, "safety", None), "planning_world", None), "payload", None),
                             "length_mm", None))
    return length if length is not None and length > 0.0 else None


def _found_nothing(report: Any) -> bool:
    from src.robot.execution.autonomous_grasp.service import found_nothing  # noqa: PLC0415

    return bool(found_nothing(report))


def _only_kept_out(report: Any) -> bool:
    """Whether the pick saw only parts this task keeps out (its regions): an empty look. A part a zone skips after a
    failed pick still stands there, and the look that saw it is a failed pick."""
    from src.robot.execution.autonomous_grasp.service import only_kept_out  # noqa: PLC0415

    return bool(only_kept_out(report))


def _kinematic(arm: Any) -> bool:
    """Whether ``arm.fk`` is a kinematic model, so it says where a pose puts the tool: every arm but one whose
    capabilities say it has none (``has_native_fk`` False: the dummy, whose fk answers where it stands)."""
    return getattr(getattr(arm, "capabilities", None), "has_native_fk", True) is not False


def _carried_part_unmodelled(arm: Any) -> str:
    """Why the part a pick just gripped is not carried by the planner, or ``""`` where it is: what the pick's attach
    left in force (``payload_model``), on an arm that models a carried part at all. Reads only."""
    from src.robot.core.arm_capabilities import CarriesPayload, PayloadModel  # noqa: PLC0415

    if not isinstance(arm, CarriesPayload):
        return ""
    try:
        model = arm.payload_model()
    except Exception as exc:  # noqa: BLE001 (a model nobody can read carries nothing anyone can vouch for)
        return f"what models it could not be read ({type(exc).__name__}: {exc})"
    if model is PayloadModel.PLANNER_AND_FILTER:
        return ""
    if model is PayloadModel.FILTER_ONLY:
        return ("the planner declined to carry it (payload_model filter_only: only the self filter takes it out of what "
                "the cameras see)")
    return f"nothing models it: the planner carries no part (payload_model {getattr(model, 'value', model)})"


def _summary(report: Any) -> str:
    summary = getattr(report, "failure_summary", None)
    said = summary() if callable(summary) else ""
    return str(said or getattr(report, "outcome", "") or "no reason given")


def _part_may_be_held(arm: Any, gripper: Any, *, read_hand: bool = True) -> str:
    """Why a part may still be in the jaws, or ``""``: a toggle's count says CLOSED, the planner still models a carried
    part, or (``read_hand``) the gripper measures a hold. Reads only; nothing is asked or sent."""
    from src.robot.core.arm_capabilities import CarriesPayload, PayloadModel  # noqa: PLC0415
    from src.robot.core.gripper import HoldEvidence, hold_evidence_of, toggle_without_sensor_of  # noqa: PLC0415

    toggle = toggle_without_sensor_of(gripper)
    try:
        if toggle is not None and toggle.jaws_closed is not False:
            return "the program's count says the toggle's jaws stand CLOSED"
        if isinstance(arm, CarriesPayload) and arm.payload_model() is not PayloadModel.NONE:
            return "the planner still models a part in the hand"
        if read_hand and gripper is not None and bool(getattr(gripper, "is_connected", False)) \
                and hold_evidence_of(gripper) is HoldEvidence.HELD:
            return "the gripper still measures a part"
    except Exception as exc:  # noqa: BLE001 (a hand nobody can read may hold anything)
        return f"the hand could not be read ({type(exc).__name__}: {exc})"
    return ""


_SCREEN_HEADS: Mapping[str, str] = {
    "clear": "is clear",
    "band": "is in the planner's cushion band: a straight leg takes the arm in and out",
    "guard_refused": "is refused by the exact guard",
    "planner_refused": "is refused by the planner",
    "unscreened": "is not screened: the move judges it when it runs",
}


def _screened_said(role: str, name: str, verdict: str) -> str:
    """``task.pose_screened`` for a person: ``The place pose 'drop_left' is clear.``"""
    return f"The {role} pose {name!r} {_SCREEN_HEADS.get(verdict, f'screened {verdict}')}."


def _checked_said(label: str, check: TargetCheck) -> str:
    """``task.target_checked`` for a person."""
    if check.seen is None:
        return f"The {label} was not seen where it was kept."
    moved = 0.0 if check.moved_mm is None else check.moved_mm
    if check.followed:
        return f"The {label} moved {moved:.0f} mm since it was kept, within {check.bound_mm:.0f} mm: followed."
    if check.why == "moved_too_far":
        return f"The {label} moved {moved:.0f} mm since it was kept; the bound is {check.bound_mm:.0f} mm."
    return f"Another {label} stands {moved:.0f} mm from where it was kept, of another size."


def _lost_said(label: str, check: TargetCheck) -> str:
    """The head of ``task.target_lost`` for a person: why the kept target is no longer followed."""
    if check.seen is None:
        return f"The {label} is not where it was kept (not seen)"
    moved = 0.0 if check.moved_mm is None else check.moved_mm
    if check.why == "moved_too_far":
        return f"The {label} is not where it was kept (moved {moved:.0f} mm)"
    return f"The {label} is not where it was kept (another size, {moved:.0f} mm away)"


def _drop_said(drop: DropPlan, label: str) -> str:
    """``task.drop_planned`` for a person."""
    if not drop.ok:
        return f"No drop: {drop.reason}."
    hang = 0.0 if drop.hang_mm is None else drop.hang_mm
    if drop.kind == "pose":
        return f"The drop stands {hang:.0f} mm above the taught pose: the part's hang."
    air = 0.0 if drop.air_mm is None else drop.air_mm
    rim = 0.0 if drop.rim_mm is None else drop.rim_mm
    return (f"The drop is over the middle of the {label or 'target'}: its rim at {rim:.0f} mm, the part hanging "
            f"{hang:.0f} mm, {air:.0f} mm of air" + (", turned along the cell's natural closing axis" if drop.turned
                                                       else "") + ".")


def _motion_note(moved: Any) -> str:
    """The motion's own sentence where it says more than its verb's name (a planned move that took a straight band leg
    says it there); ``""`` otherwise."""
    message = str(getattr(moved, "message", "") or "")
    return "" if message in ("", "move_home", "move_to_home", "move_to_joints", "move_joint", "home", "joints") \
        else message

def run_recovery_actions(service: Any, *, rescan: "bool | None", push: "bool | None",
                         clear: "bool | None") -> "tuple[str, ...] | None":
    """The recovery actions a run allows, the cell's ``recovery.allowed_actions`` changed by the run's ticks; ``None``
    where no tick was given (the cell's list stands). ``push`` or ``clear`` allows ``nudge_target`` (the push gate, which
    clearing a blocker runs under too); both false take it out."""
    if rescan is None and push is None and clear is None:
        return None
    cfg = getattr(service, "effective_config", None)
    recovery = getattr(cfg, "recovery_orchestrator", None)
    base = list(getattr(recovery, "allowed_actions", ()) or ()) if getattr(recovery, "enabled", False) else []
    actions = [str(a) for a in base]
    if rescan is not None:
        actions = [a for a in actions if a != "rescan"] + (["rescan"] if rescan else [])
    if push is not None or clear is not None:
        wants = bool(push) or bool(clear)
        actions = [a for a in actions if a != "nudge_target"] + (["nudge_target"] if wants else [])
    return tuple(dict.fromkeys(actions))
