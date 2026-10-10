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
  (``place_target.survey``) before the first pick: the bin an earlier task kept (``known_target``, the console's for
  the same phrase; ``known_targets`` by phrase for every place of a sort) is looked at again first, from its look, and
  a bin that stood still is kept with one quick look at its rim; any other is surveyed from every look, as without one.
  The survey grounds the bin alone: the parts are the pick's to count (2026-10-08).
* **The scope.** ``once`` ends when the part is placed and the arm is back; ``until_empty`` when
  ``empty_looks_to_end`` looks in a row saw nothing to pick. An empty pick counts every look it perceived from (its
  report's ``looks``), so one empty pass over a wrist camera's four looks ends either scope, and a pick of one look or
  none counts one: a fixed camera still looks twice (the owner, 2026-10-08). A look that saw only parts the task keeps
  out (its regions, never a part ``next_target`` skips after a failed pick) is an empty one. ``max_failed_in_a_row``
  failed picks in a row end it where the arm stands, and ``max_parts`` placed end it (``part_limit``). The two rows
  count apart, as the build plan writes them (1.3.3): a failed pick does not start the empty row again, nor an empty
  look the failed row; only a part placed starts both. A detector that failed where nothing was found is
  ``detector_failed``, never "nothing left".
* **The check look** (``TaskOptions.check_look``, on; the owner's "Check-Look von Home", 2026-10-08). Where a placed
  part was the only target its pick's first look counted (its report's ``targets_by_look``), the next pick is the
  check: the first look alone, home, where the return left the arm, and no generated view, as with multi-view off. It
  seeing nothing ends the task ``nothing_left`` at once; a part it sees is picked as any other; a part it sees and
  fails on counts no failure, and the pick after it looks from every look again. A pick that does not count its
  targets (a fixed camera's) is never followed by one.
* **Following its parts** (``robot.grasping.follow_parts``, off unless the cell turns it on; the owner, 2026-10-09).
  The first pick grounds every part; the next pick's first look finds the parts the last one kept (its first look,
  less the part it gripped) again with SAM2 on their boxes and no detector, where nothing changed in depth and every
  part passes every check, and is grounded as before where anything does not (``src/robot/perception/kept_scene.py``),
  and the later looks of every pick find its first look's parts by their projected boxes. The pick after a push, a
  blocker cleared, a recovery, a try that sent motion and failed, or the grip of a part that was not kept grounds
  again, so does one ``refresh_every_picks`` after the last grounding, and so does the check look: the end of a task
  is always asked of the detector. A pick that failed with nothing sent keeps what its first look saw. The memory is
  this run's alone: a Restart starts with nothing kept.
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
  bin checked again first (against where the survey found it: its rim's depth and colour where it was last seen, the
  detector wherever that is unsure), and placed without a keep-out. Either drop keeps the tilt the tool gripped with,
  turned about the vertical only, so the hang it is raised by still bounds the part. A bin the check lost is looked
  for again first (``robot.place.relocate``, on; the owner, 2026-10-09: "wenn sie dies nicht mehr tut, dann kann er
  seine Ablage nochmal neu errechnen"): the hold of the look ends, the task drives its looks once more with the part in
  the jaws (``place_target.relocate``), and a bin of the size the survey found and the colour it followed, standing in
  no other place, is kept in place of the old one (``task.target_relocated``); the part is carried to the look it was
  found from and checked there as before, the drop planned anew over it. Once per part: lost again, or found nowhere,
  it is as before. A drop refused, a bin lost or a part that does not fit puts the part back where it was gripped
  (``service.put_back``), returns, and asks. A place counts its part placed once the jaws let go
  (:func:`place_released`), its line out refused or not.
* **Set down, not dropped, never piled** (the cell's ``robot.place``, each off as shipped; the owner, 2026-10-08
  night). Below a box's rim, from what the check read inside it, over the rim where the box reads full, the part or
  the open hand would not stand clear of its walls, or the guard refuses the line in (``release_in_a_box``); the hang
  from what the pick's looks read under the part (``part_bottom``); side by side on a flat place, a spot the camera
  reads free first, else the next of the grid the task has not filled, none left putting the part back and asking
  ``part_does_not_fit`` (``side_by_side``), each part laid at a taught pose kept out of an "until empty" task's picks;
  and straight over the rim to a bin one of the pick's looks saw, checked on that look's frame and judged against it
  (``carry``, or the task's own ``TaskOptions.carry``), via the look wherever that cannot be.
* **What it keeps out of its picks** (Q4), for the whole task: a camera place's bin, its footprint grown by 10 mm, for
  both scopes; a pose place's drop, a circle of ``pose_keep_out_mm`` about it, for "until empty", where the arm's
  forward kinematics say where the taught pose puts the tool (an arm whose ``fk`` is no kinematic model, the
  rehearsal's dummy, gets none, and the log says so). No pick takes the bin or a part already placed; a look that sees
  only those is an empty one.
* **A sort** (the owner, 2026-10-09: "Gruene Teile in die gelbe Kiste, rote in die blaue"). A plan of several rules
  (``TaskPlan.more_rules``, up to :data:`MAX_FURTHER_RULES` :class:`SortRule` beside its own) puts each kind of part
  where its rule says; two rules may share a place. Every place a camera finds is found before the first pick, one
  locate of a class list per look (``place_target.survey_places``), or the task stops before its first pick naming
  every place it is missing; each is kept out of the picks, and a check of one replaces its own region alone. Where
  the detector looks at one bin of several, at a check or a search, it grounds them all (``together``), so a second
  bin in view is never taken for it. Each pick grounds every rule's kind in one call (``class_list_prompt``), and the
  part it gripped goes by the rule of the kind it went for (``task.rule``); a gripped part no rule names goes back where
  it was gripped, and the task asks (``target_not_found``). A part no rule clearly claims stays where it lies, and the
  end names them (``task.unsorted``): what the last empty pick's first look turned away (``ambiguous``, a word no rule
  names), or what one locate of every part from where the arm stands sees outside every place of the task, a part of
  no rule's kind included, whichever counts more; nothing moves for it. A sort sets every blocker aside, follows no
  parts from pick to pick (the kept scene follows one label), judges no carry ahead of its pick (its place is known
  only once the pick says which kind it gripped), and runs only on a cell whose perception grounds a phrase.

It never asks a person, and every end of a run it started is a :class:`TaskReport`. A request this cell cannot do (an
arm whose place cannot run, an unknown pose, a carried part nobody models, an empty object without ``pick_anything``,
both jaw faces on a cell that turns every grasp off them, a sort on a cell that grounds no phrase) raises
:class:`TaskRefused` before anything is commanded, and a programmer's error raises.
"""

from __future__ import annotations

import dataclasses
import math
import re
import time
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, Protocol, Sequence

from src.contracts import UNSET, Maybe, chosen
from src.robot.constants import create_robot_logger
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
    relocate,
    rim_clearance_mm,
    screen_joints,
    screen_pose,
    survey,
    survey_places,
)

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.geometry import Pose
    from src.robot.core import JointPositions
    from src.robot.execution.autonomous_grasp.report import AutonomousGraspReport
    from src.robot.execution.handling import HandlingReport
    from src.robot.execution.motion import MotionReport
    from src.robot.execution.place_target import Relocated
    from src.robot.grasping.recovery.exclusion_zones import ExclusionRegion
    from src.robot.perception.kept_scene import KeptScene

__all__ = [
    "MAX_FURTHER_RULES",
    "PlaceAt",
    "SortRule",
    "TaskEvent",
    "TaskHooks",
    "TaskOptions",
    "TaskPlan",
    "TaskRefused",
    "TaskReport",
    "TaskStop",
    "jaws_before_a_pick",
    "pick_phrase",
    "place_released",
    "run_task",
]

logger = create_robot_logger(__name__, "task.log")


# ---------------------------------------------------------------------------------------------------------------------
# The words of a task
# ---------------------------------------------------------------------------------------------------------------------


class TaskStop(StrEnum):
    """Why a task ended: each a stop code of the console (``api.codes.StopCode``), in one of four classes
    (:attr:`stop_class`). A task that ends ``done``, ``operator`` or ``ask`` returns to its return pose first; a
    ``problem`` leaves the arm where it stopped.

    Attributes:
        FINISHED: The task is done; the arm stands at its return pose. Class ``done``.
        NOTHING_LEFT: Nothing matching was seen twice in a row ("until empty"). Class ``done``.
        PART_LIMIT: The part limit (``TaskPlan.max_parts``) is reached. Class ``done``.
        STOPPED_AFTER_PART: Stopped as asked once the part in hand was placed. Class ``operator``.
        TARGET_NOT_FOUND: The place's target was seen from no look; nothing was picked. Class ``ask``.
        TARGET_LOST: The target was no longer where it was found, and not found again. Class ``ask``.
        TARGET_UNREACHABLE: The arm cannot reach the target without a collision. Class ``ask``.
        PART_DOES_NOT_FIT: The part does not fit the opening; it is back where it was. Class ``ask``.
        POSE_REFUSED: The screen refused one of the task's poses. Class ``ask``.
        HALTED: Halt now: nothing was commanded after it; the arm moves again only after a person confirms the cell is
            clear. Class ``problem``.
        CONTROLLER_STOPPED: The controller cannot move (protective stop, emergency stop, powered off). Class
            ``problem``.
        HAND_NEEDS_PERSON: The hand reports a fault or an unknown state. Class ``problem``.
        RECOVERY_NEEDS_PERSON: A push or a recovery stopped where the arm stands; a person decides. Class ``problem``.
        PART_STILL_HELD: The place was refused before the release; the part is still in the hand. Class ``problem``.
        RETURN_FAILED: The move to the return pose was refused. Class ``problem``.
        FAILED_IN_A_ROW: Several picks in a row failed (``TaskPlan.max_failed_in_a_row``). Class ``problem``.
        DETECTOR_FAILED: The detector failed, which is not "nothing found". Class ``problem``.
        CELL_FAULT: The camera or the cell reported a fault. Class ``problem``.
        DISCONNECTED: The cell was disconnected during the run. Class ``problem``.
    """

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
    """What a task says between its motions, each an event type of the console; ``TaskHooks.event`` gets them with their
    data as keywords.

    Attributes:
        POSE_SCREENED: A taught pose of the task was screened before anything moved.
        SURVEY_STARTED: The task began looking for its camera places.
        TARGET_FOUND: A camera place's target was found, with its overlay.
        TARGET_MISSING: A camera place's target was seen from no look.
        PART_STARTED: The picks for the next part began.
        NOTHING_FOUND: A pick found nothing to grip.
        CARRY_STARTED: The gripped part is on its way to the place.
        TARGET_CHECKED: The bin was looked at again before the drop, and stood where it was.
        TARGET_LOST: The bin was not where it was found.
        DROP_PLANNED: The drop over the target was planned.
        PLACE_STARTED: The place began.
        PLACED: The part was released at its place.
        PLACE_FAILED: The place was refused or failed.
        PUT_BACK: A part was put back where it was gripped.
        RETURN_STARTED: The move to the return pose began.
        RETURNED: The arm reached the return pose.
        RETURN_FAILED: The move to the return pose was refused.
        PART_FINISHED: One part's work ended.
        RULE: A sort: the rule a gripped part goes by.
        TARGET_RELOCATED: A bin that moved was found again, and the drop computed anew.
        UNSORTED: A sort's end: how many parts no rule took, left where they lie.
    """

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
    # A sort (the owner, 2026-10-09): the rule a gripped part goes by, a place found again, the parts no rule took.
    RULE = "task.rule"
    TARGET_RELOCATED = "task.target_relocated"
    UNSORTED = "task.unsorted"


#: What a task grounds in front of the object it names, so every such part comes back in a box of its own. On a mat of
#: 20 parts Qwen3-VL-4B grounded one box for "object", "objects", "all objects" or "every object" and all 20 for "each
#: separate object", one of the two green parts for "green part" and both for "each separate green part", both red
#: ones and no orange one for "each separate red part" (2026-10-06). A part the detector does not name keeps the whole
#: rule beside the fingers, the push and the blocker.
EACH_SEPARATE = "each separate"
#: What a task that names no object grounds: every part in a box of its own.
EVERY_PART_PHRASE = f"{EACH_SEPARATE} object"
#: How wide a thing the end of a sort may see and still count as a part left where it lies, mm, on either side: one
#: wider is what the parts stand on (a mat, a tray), which a locate of every part boxes too.
LEFT_OVER_MAX_MM = 250.0


def pick_phrase(named: str, *, which: str = "", source: str = "") -> str:
    """What a task's picks ground on a cell that grounds a phrase (G1, the owner's speed round of 2026-10-08): the one
    part the operator singled out, as said (``which``, "the gray cube on top of the other one": no "each separate",
    which asks for every part of the kind and grounded both cubes of a stack); else every part of the kind named, each
    in a box of its own, where the parts lie after it (``source``, "each separate gray cube on the black mat"); else
    every part there ("each separate object on the black mat")."""
    named, which, source = named.strip(), which.strip(), source.strip()
    where = f" {source}" if source else ""
    if which:
        return which
    if named:
        return f"{EACH_SEPARATE} {named}{where}"
    return f"{EVERY_PART_PHRASE}{where}"


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
    """Where a task puts each part: a taught pose, or a target the camera finds; exactly one of the two.

    ```python
    PlaceAt(pose="Ablage links")                 # a taught pose, by its name in the task's poses
    PlaceAt(camera="blue bin", air_mm=20.0)      # a bin the camera finds
    ```

    Attributes:
        pose (str | None): The name of a taught pose in the task's ``poses``: where the part's bottom is let go
            (default: None).
        camera (str | None): An English phrase the camera grounds, such as ``"blue bin"`` (default: None).
        air_mm (float | None): For a camera target: the air over its rim when the jaws open, 10 to 50 mm; ``None`` is
            the cell's own (``place_target.rim_clearance_mm``, 20 mm as shipped). A taught pose takes none (default:
            None).

    Raises:
        ValueError: Neither or both of ``pose`` and ``camera``, an empty one, or ``air_mm`` outside 10 to 50 mm, or
            given for a pose.
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


#: How many rules a sort holds beside the task's own first one: four in all (the owner, 2026-10-09). The command
#: reader's own bound (``src.models.vlm.command.MAX_FURTHER_RULES``), kept here so a plan is checked without loading it.
MAX_FURTHER_RULES = 3


@dataclass(frozen=True)
class SortRule:
    """One rule of a sort ("Gruene Teile in die gelbe Kiste, rote in die blaue"): a kind of part and where it goes. A
    plan's first rule is its own ``object`` and ``place``; its further rules are these.

    Attributes:
        object (str): The kind of part, the English phrase the detector grounds, such as ``"red cube"``.
        place (PlaceAt): Where that kind goes.
        which (str): The one part the operator singled out, grounded alone, as a plan's ``which`` (default: "").
        source (str): Where the parts lie, such as ``"on the black mat"`` (default: "").

    Raises:
        TypeError: A field that is not a phrase, or a place that is not a ``PlaceAt``. The plan refuses an empty kind
            and a phrase holding ``|`` or the word ``ambiguous`` (the detector's class list).
    """

    object: str
    place: PlaceAt
    which: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        for name in ("object", "which", "source"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"a rule's {name} is a phrase, not {type(getattr(self, name)).__name__}")
        if not isinstance(self.place, PlaceAt):
            raise TypeError(f"a rule's place is a PlaceAt, not {type(self.place).__name__}")


#: The detector's label of a box no description of a class list clearly claims, and the bar between the descriptions
#: (``src.models.vlm.parsing.AMBIGUOUS_LABEL``, ``CLASS_LIST_SEPARATOR``): a sort's phrase may hold neither.
_AMBIGUOUS = "ambiguous"
_BAR = "|"


def _words(text: Any) -> str:
    """How a task compares two kinds of part, or two places a camera finds: trimmed, each run of blanks one space, case
    folded (``place_target``'s own compare)."""
    return " ".join(str(text or "").split()).casefold()


def _class_list_refusal(text: str, what: str) -> str:
    """Why ``text`` cannot be one description of a class list, or ``""``: ``class_list_prompt``'s own two refusals, a
    bar, which would split it in two, and the word ``ambiguous``, which would take a box no rule claims for its own.
    Kept here, so a plan is checked without loading a model package."""
    if _BAR in text:
        return (f"{what} {text!r} holds {_BAR!r}, which separates the phrases the detector is asked for in one call; a "
                "sort grounds every rule's phrase so")
    if _AMBIGUOUS in re.findall(r"[a-z0-9]+", text.casefold()):
        return (f"{what} {text!r} holds the word {_AMBIGUOUS!r}, the detector's label of a part no rule clearly "
                "claims")
    return ""


def _refuse_a_sort_of(rules: "Sequence[SortRule]") -> None:
    """Raise ``ValueError`` where ``rules``, a sort's (its own first), cannot be sorted: more than
    :data:`MAX_FURTHER_RULES` beside the first, a rule that names no kind, one kind in two rules (case and blanks aside:
    each kind goes to one place), two rules that ground one phrase, and a phrase the detector's class list cannot hold
    (a part's, as the picks ground it, or a place's the camera finds)."""
    if len(rules) - 1 > MAX_FURTHER_RULES:
        raise ValueError(f"a sort holds at most {MAX_FURTHER_RULES} rules beside its own ({MAX_FURTHER_RULES + 1} in "
                         f"all), not {len(rules) - 1}")
    kinds: dict[str, int] = {}
    grounded: dict[str, int] = {}
    for number, rule in enumerate(rules, start=1):
        if not rule.object.strip():
            raise ValueError(f"a sort names the kind of part of every rule, and rule {number} names none")
        earlier = kinds.setdefault(_words(rule.object), number)
        if earlier != number:
            raise ValueError(f"a sort puts each kind of part in one place, and rules {earlier} and {number} both name "
                             f"{rule.object.strip()!r}")
        phrase = pick_phrase(rule.object, which=rule.which, source=rule.source)
        earlier = grounded.setdefault(_words(phrase), number)
        if earlier != number:
            raise ValueError(f"rules {earlier} and {number} of the sort ground one phrase, {phrase!r}, so no part could "
                             "be told whose it is")
        why = _class_list_refusal(phrase, f"rule {number}'s part")
        if not why and rule.place.camera is not None:
            why = _class_list_refusal(rule.place.camera.strip(), f"rule {number}'s place")
        if why:
            raise ValueError(why)


@dataclass(frozen=True)
class TaskOptions:
    """The Advanced drawer's switches, for one task: never the cell's.

    Attributes:
        multi_view (bool): The looks as configured; ``False`` is the first look only and no generated view
            (default: True).
        both_faces (bool): Grip only once both jaw contact faces of the grasp were seen (default: False).
        closing_axis (str | None): Grip only grasps closing along this axis, a name ``Pose.tool_down`` takes (``"-y"``);
            ``None`` any (default: None).
        push_mm (float | None): How far a push of a failed part moves it, millimetres; ``None`` the cell's, which may go
            longer where it opens too little room (default: None).
        critical_parts (bool | None): The owner's switch for this task: ``True`` clears a blocker and pushes nothing;
            ``None`` the cell's ``recovery.critical_parts`` (default: None).
        record_views (bool): Keep each pick's looks for training (default: False).
        overlay (bool): Render the grasp overlay during the task (default: True).
        pick_anything (bool): An empty object means anything the camera sees, bin walls included (default: False).
        rescan (bool | None): The run's word on the ``rescan`` recovery; ``None`` keeps the cell's (default: None).
        push (bool | None): The run's word on pushing (``nudge_target``); ``None`` keeps the cell's (default: None).
        clear (bool | None): The run's word on clearing a blocker: ``True`` without ``push`` clears blockers and pushes
            nothing; ``None`` keeps the cell's (default: None).
        blocker_into_the_place (bool | None): Where a blocker goes: ``True``, a task that names no object takes it as
            its part and sets it down where the parts go; ``False``, set aside on the support; ``None`` the cell's
            (default: None). A task that names an object sets every blocker aside.
        check_look (bool): Where the part placed was the last target its pick's first look counted, the next pick looks
            from that first look alone, and seeing nothing there ends an "until empty" task (default: True).
        carry (Literal["via_the_look", "over_the_rim"] | None): How a part is carried to a bin a wrist camera
            found: ``via_the_look`` to the look it was found from, looked at again there; ``over_the_rim`` straight
            over; ``None`` the cell's ``robot.place.carry`` (default: None).
        every_look (bool): "Alle Posen": each pick visits every look, however safe an earlier grasp is (default: False).

    Raises:
        ValueError: ``every_look`` with ``multi_view`` off, or a ``carry`` that is neither of its two.
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
    check_look: bool = True
    carry: Literal["via_the_look", "over_the_rim"] | None = None
    every_look: bool = False

    def __post_init__(self) -> None:
        if self.carry not in (None, "via_the_look", "over_the_rim"):
            raise ValueError(f"a task's carry is 'via_the_look' or 'over_the_rim' (None: the cell's), not {self.carry!r}")
        if self.every_look and not self.multi_view:
            raise ValueError("a task asks for every look or for its first look only (multi-view off), not for both")


@dataclass(frozen=True)
class TaskPlan:
    """One task: what to pick, where to put it, where to go after, and how long.

    ```python
    plan = TaskPlan(object="gray cube", place=PlaceAt(camera="yellow bin"), scope="until_empty")
    report = run_task(cell.service, plan, hooks=my_hooks, poses={})
    ```

    Attributes:
        object (str): What to pick, the English phrase the detector grounds; ``""`` anything.
        place (PlaceAt): Where each part goes.
        return_to (str): Where the arm goes after each part: ``"home"`` or a taught pose's name (default: "home").
        scope (Literal["once", "until_empty"]): One part, or until nothing is left (default: "once").
        options (TaskOptions): This task's switches (default: TaskOptions()).
        first_motion (Literal["look", "return"]): ``look`` for a new task, ``return`` for a Restart, whose first motion
            is the planned move to the return pose (default: "look").
        max_failed_in_a_row (int): Picks in a row that may fail before the task stops (default: 3).
        empty_looks_to_end (int): Empty looks in a row that end an "until empty" task (default: 2).
        max_parts (int): The most parts one task places (default: 100).
        pose_keep_out_mm (float): How far about a pose place's drop an "until empty" task picks nothing, millimetres
            (default: 150.0).
        which (str): The one part the operator singled out ("the gray cube on top of the other one"), grounded alone
            (default: "").
        source (str): Where the parts lie ("on the black mat") (default: "").
        more_rules (tuple[SortRule, ...]): Makes the task a sort: up to :data:`MAX_FURTHER_RULES` further rules, each
            kind of part to its rule's place; ``()`` a task of one kind (default: ()).

    Raises:
        ValueError: A sort whose rules name no kind, a kind in two rules, a phrase holding ``|`` or ``ambiguous``, more
            than :data:`MAX_FURTHER_RULES` further rules; a scope, first motion or limit out of its range; a
            ``return_to`` that names nothing.
        TypeError: A place that is not a ``PlaceAt``, a further rule that is not a ``SortRule``, a phrase that is not a
            string.
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
    which: str = ""
    source: str = ""
    more_rules: "tuple[SortRule, ...]" = ()

    @property
    def rules(self) -> "tuple[SortRule, ...]":
        """Every rule of the task, its own first (its ``object``, ``place``, ``which`` and ``source``), then its
        ``more_rules``: one for a task of one kind."""
        return (SortRule(self.object, self.place, self.which, self.source), *self.more_rules)

    def __post_init__(self) -> None:
        if isinstance(self.more_rules, list):
            object.__setattr__(self, "more_rules", tuple(self.more_rules))
        if not isinstance(self.more_rules, tuple):
            raise TypeError(f"a task's further rules are SortRules in a tuple, not {type(self.more_rules).__name__}")
        for rule in self.more_rules:
            if not isinstance(rule, SortRule):
                raise TypeError(f"a further rule of a task is a SortRule, not {type(rule).__name__}")
        for name in ("which", "source"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"a task's {name} is a phrase, \"\" for none, not {type(getattr(self, name)).__name__}")
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
        if self.more_rules:
            _refuse_a_sort_of(self.rules)


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
        """One of the task's events, said between its motions.

        Args:
            name (TaskEvent): Which event.
            **data (Any): Its data: among them ``said``, the library's English sentence for it; a camera target's events
                carry its overlay as ``image_png`` (bytes or ``None``).
        """
        ...

    def pick_done(self, part: int, pick: int, report: "AutonomousGraspReport") -> None:
        """One pick of the task ended.

        Args:
            part (int): The part's number in the task, from 1.
            pick (int): The pick's number for that part, from 1.
            report (AutonomousGraspReport): The pick's report; its looks' file in ``telemetry["views_file"]`` where they
                were kept.
        """
        ...

    def stop_after_part(self) -> bool:
        """Whether the operator asked the task to stop once the part in hand is placed.

        Returns:
            bool: ``True`` stops after the part.
        """
        ...

    def halted(self) -> bool:
        """Whether "halt now" was pressed.

        Returns:
            bool: ``True`` ends the task with nothing more commanded.
        """
        ...

    def abandoned(self) -> str:
        """Why the cell is being taken down under the task.

        Returns:
            str: The reason, or ``""`` while it is not.
        """
        ...


@dataclass(frozen=True)
class TaskReport:
    """What a task did, and why it ended.

    Attributes:
        stop (TaskStop): Why it ended.
        sentence (str): The same, as a sentence for a person.
        parts_placed (int): Parts released at the place.
        picks (int): Picks made.
        succeeded (int): Picks that gripped.
        holding (bool): Whether the program believed a part in the jaws at the end: a record only; a gate reads the live
            hand.
        last_report (AutonomousGraspReport | None): The last pick's own report; ``None`` if none ran.
        at_return (bool): Whether the arm stands at the return pose because the task's last motion went there.
        kept_target (KeptTarget | None): The bin a camera place followed last, which the next task into the same phrase
            may take as ``known_target``; ``None`` for a pose place, or where it ended ``target_lost`` or
            ``target_not_found`` (default: None).
        kept_targets (Mapping[str, KeptTarget]): Every bin a camera place followed last, by its phrase, for the next
            task's ``known_targets`` (default: {}).
        unsorted (int): How many parts no rule of a sort claimed, left where they lie; 0 otherwise (default: 0).
    """

    stop: TaskStop
    sentence: str
    parts_placed: int = 0
    picks: int = 0
    succeeded: int = 0
    holding: bool = False
    last_report: "AutonomousGraspReport | None" = None
    at_return: bool = False
    kept_target: "KeptTarget | None" = None
    kept_targets: "Mapping[str, KeptTarget]" = field(default_factory=dict)
    unsorted: int = 0

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """The task as a person reads it.

        Returns:
            str: ASCII, no trailing newline. ``print(report)`` shows the same.
        """
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
        """The task as plain data.

        Returns:
            dict[str, Any]: ``json.dumps`` safe.
        """
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
            "kept_target": None if self.kept_target is None else self.kept_target.to_dict(),
            "kept_targets": {str(phrase): kept.to_dict() for phrase, kept in self.kept_targets.items()},
            "unsorted": self.unsorted,
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
             locators: "Maybe[Sequence[Any]]" = UNSET, known_target: "KeptTarget | None" = None,
             known_targets: "Mapping[str, KeptTarget | None] | None" = None) -> TaskReport:
    """Run a task on a connected cell: pick a part, set it down, return, and look again, once or until nothing is left.
    The operator console's Start runs exactly this.

    ```python
    with cell.connected():
        report = run_task(cell.service, TaskPlan(object="gray cube", place=PlaceAt(camera="yellow bin")),
                          hooks=my_hooks, poses={})
    print(report)
    ```

    Args:
        service (Any): The connected pick service, ``cell.service``.
        plan (TaskPlan): What to do.
        hooks (TaskHooks): What the task says between its motions, and asks: stop after the part, halted, abandoned.
        poses (Mapping[str, JointPositions]): The taught poses the plan may name (place and return), joints by name.
        locators (Maybe[Sequence[Any]]): What finds a camera place's target; unset is the service's own cameras
            (default: UNSET).
        known_target (KeptTarget | None): The bin an earlier task kept (its report's ``kept_target``): a camera place
            into the same phrase looks at it again first, from its look; in a sort, the first camera place's (default:
            None).
        known_targets (Mapping[str, KeptTarget | None] | None): The bins earlier tasks kept, by phrase (a report's
            ``kept_targets``), each looked at again first by the place of its phrase (default: None).

    Returns:
        TaskReport: Why it ended, the parts placed, the picks, whether the arm is back. Every end but a refusal is the
            report's.

    Raises:
        TaskRefused: A request this cell cannot do (an unknown pose, a refused route, ...), raised before anything is
            commanded; ``code`` is the console's refusal code.
    """
    return _Task(service, plan, hooks, poses, locators, known_target, known_targets).run()


class _Ended(Exception):
    """How a step ends the task: the report travels up to :meth:`_Task.run`."""

    def __init__(self, report: TaskReport) -> None:
        super().__init__(report.sentence)
        self.report = report


@dataclass(eq=False)
class _Place:
    """One place of a task's rules, as the task follows it: where it is (``at``, the first rule's word for it) and the
    rules that put their parts there (``rules``, indices into ``TaskPlan.rules``; two rules may share one, the owner,
    2026-10-09).

    A camera place keeps the bin its survey found (``surveyed``, what every check measures against, a bin found again
    in its stead), the sighting the drops go to now (``kept``), the region of it kept out of the picks (``region``, as
    ``keep_out_region`` returned it: a check of this place replaces it alone), what its last check read where the part
    goes (``inside``), the air over its rim (``air_mm``), and whether its bin was lost and found nowhere (``lost``). A
    pose place keeps where its taught pose puts the tool (``tcp``) and the circle about its drop ("until empty",
    ``region``). Either keeps the spots of a flat place this task laid a part on, each its middle and how far the part
    reached from it (``spots``)."""

    at: PlaceAt
    rules: "list[int]"
    air_mm: float
    surveyed: "KeptTarget | None" = None
    kept: "KeptTarget | None" = None
    region: "ExclusionRegion | None" = None
    inside: Any = None
    spots: "list[tuple[tuple[float, float], float]]" = field(default_factory=list)
    tcp: "Pose | None" = None
    lost: bool = False

    @property
    def where(self) -> str:
        """The place as the task's events name it: ``pose:<name>`` or ``target:<phrase>``."""
        return f"pose:{self.at.pose}" if self.at.pose is not None else f"target:{self.at.camera}"

    @property
    def named(self) -> str:
        """What names it: the taught pose's name, or the phrase the camera finds it by."""
        return str(self.at.pose if self.at.pose is not None else self.at.camera).strip()


def _places_of(plan: TaskPlan, tree: Any) -> "list[_Place]":
    """The places of ``plan``'s rules, each once, in the order the rules first name them: one for a task of one kind. A
    camera place is one phrase, case and blanks aside; a pose place one name."""
    places: list[_Place] = []
    for index, rule in enumerate(plan.rules):
        at = rule.place
        same = next((place for place in places if place.at.pose == at.pose and (
            at.pose is not None or _words(place.at.camera) == _words(at.camera))), None)
        if same is not None:
            same.rules.append(index)
            continue
        air = float(at.air_mm) if at.air_mm is not None else rim_clearance_mm(tree)
        places.append(_Place(at=at, rules=[index], air_mm=air))
    return places


def _known_targets_of(known: Any) -> "dict[str, KeptTarget]":
    """The bins earlier tasks kept, by the camera phrase each was found for, as handed in (``None`` entries left out);
    anything that is no such map, and an entry that is no ``KeptTarget``, is a programmer's error: raised."""
    if known is None:
        return {}
    if not isinstance(known, Mapping):
        raise TypeError(f"the known targets are the bins earlier tasks kept by their phrases, not {type(known).__name__}")
    bins: dict[str, KeptTarget] = {}
    for phrase, kept in known.items():
        if kept is None:
            continue
        if not isinstance(kept, KeptTarget):
            raise TypeError(f"a known target is the KeptTarget an earlier task kept, not {type(kept).__name__}")
        bins[str(phrase)] = kept
    return bins


class _Task:
    """One run of :func:`run_task`: the cell, the plan, and what happened so far."""

    def __init__(self, service: Any, plan: TaskPlan, hooks: TaskHooks, poses: "Mapping[str, JointPositions]",
                 locators: "Maybe[Sequence[Any]]", known_target: "KeptTarget | None" = None,
                 known_targets: "Mapping[str, KeptTarget | None] | None" = None) -> None:
        self.service = service
        self.plan = plan
        self.hooks = hooks
        self.poses = dict(poses)
        self.chosen_locators = locators
        if known_target is not None and not isinstance(known_target, KeptTarget):
            raise TypeError(f"a known target is the KeptTarget an earlier task kept, not {type(known_target).__name__}")
        self.known_target = known_target
        self.known_targets = _known_targets_of(known_targets)
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
        # How far below the tool a part's bottom hangs at least (the fingertips' reach past the TCP), and at most where
        # the cell declares its longest part's length past the fingertips.
        self.fingertips_mm = _fingertips_past_the_tcp_mm(tree)
        self.hang_mm = self.fingertips_mm + (self.length_mm or 0.0)
        self.locators: list[Any] = []
        # The places of the task's rules, each once (one for a task of one kind), and what the task keeps of each: the
        # bin a camera place's survey found and the sighting of it the drops go to now, where a taught pose puts the
        # tool, the spots a part was laid on. Whether the task sorts (several rules, the owner, 2026-10-09).
        self.places = _places_of(plan, tree)
        self.sorts = bool(plan.more_rules)
        # How the parts are set down (robot.place, the owner, 2026-10-08 night): the cell's rules and the open hand.
        self.rules = _place_rules(tree)
        self.open_hand = _open_hand_of(tree)
        # How many parts no rule of a sort clearly claimed, as its last look saw them (task.unsorted).
        self.unsorted = 0
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
        # The parts this task follows from pick to pick (robot.grasping.follow_parts, the owner, 2026-10-09): the cell's
        # block where it follows, and what the last pick kept. This run's alone: a Restart is a new run, and grounds.
        self.follow = _follow_parts_of(tree, service)
        self.memory: "KeptScene | None" = None
        if self.follow is not None and self.sorts:
            # The kept scene follows one label (the target's), and a sort's picks ground every rule's kind in one call.
            logger.info("task: a sort follows no parts from pick to pick (robot.grasping.follow_parts is on): what a pick "
                        "keeps follows one label, and a sort's picks ground every rule's kind; every pick grounds them")
            self.follow = None
        if self.follow is not None:
            logger.info("task: follows its parts from pick to pick (robot.grasping.follow_parts: refresh_every_picks "
                        "%d, max_shift_mm %g, max_creep_mm %g); the first pick and every trigger ground them",
                        int(self.follow.refresh_every_picks), float(self.follow.max_shift_mm),
                        float(self.follow.max_creep_mm))

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
        if self._camera_places():
            if self.hooks.stop_after_part():
                # No part is in hand yet: the task ends here, back at its return pose where it moved at all.
                return self._finish(TaskStop.STOPPED_AFTER_PART, "the task stopped before its first pick, as the "
                                                                 "operator asked; nothing was picked")
            self._survey()
        once = self.plan.scope == "once"
        empty = failed = 0
        # Whether the next pick is the check look (the owner, 2026-10-08): set where the part placed last was the only
        # target its pick's first look counted, taken back by the pick it sets.
        check = False
        while True:
            if self.hooks.stop_after_part():
                return self._finish(TaskStop.STOPPED_AFTER_PART, "the task stopped after the part, as the operator "
                                                                 "asked")
            self._go_on_or_end()
            self._hand_before("before the pick")
            self._check_a_fixed_cameras_target()
            if self.part_number != self.parts_placed + 1:
                self.part_number, self.part_since = self.parts_placed + 1, time.monotonic()
            self.picks += 1
            self._say(TaskEvent.PART_STARTED, f"Looking for part {self.part_number}"
                                              + (" of 1." if once else "."),
                      part=self.part_number, of=1 if once else None, pick=self.picks)
            from src.robot.execution.motion import expecting_next  # noqa: PLC0415

            # The carry after the pick, judged while the jaws close where the arm does (robot.motion.judge_next_leg). A
            # sort's place is known only once the pick says which kind it gripped: nothing is judged ahead.
            first = self.places[0]
            carry = first.kept.look if not self.sorts and first.at.pose is None and first.kept is not None else None
            with expecting_next(self.arm, carry):
                report, failures_before = self._pick(check=check)
            checked, check = check, False
            self._stop_on_what_the_pick_said(report)
            kept_out = _only_kept_out(report)
            if _found_nothing(report) or kept_out:
                if self._detector_failures() > failures_before:
                    return self._problem(TaskStop.DETECTOR_FAILED, (
                        "the detector failed while the pick looked, so 'nothing found' is no answer: "
                        + _summary(report)))
                # Every look the empty pick perceived from is an empty look: one empty pass over the looks ends it.
                looks = _looks_that_saw_nothing(report)
                empty += looks
                self._say(TaskEvent.NOTHING_FOUND,
                          ("Only parts this task keeps out were seen" if kept_out else "Nothing matching was seen")
                          + (" from the check look" if checked else f" from {looks} look(s)")
                          + f" ({empty} empty look(s) in a row).",
                          part=self.part_number, empty_in_a_row=empty, only_excluded=kept_out, looks=looks,
                          check_look=checked)
                if checked:
                    return self._finish(TaskStop.NOTHING_LEFT, (
                        f"nothing to pick is left: the part placed last was the only {self._what()} its pick's first "
                        f"look counted, and the check look from there saw none, or only where the task keeps out; "
                        f"{self.parts_placed} part(s) placed") + self._unsorted_after(report))
                if empty >= self.plan.empty_looks_to_end:
                    return self._finish(TaskStop.NOTHING_LEFT, (
                        f"nothing to pick is left: {empty} looks in a row saw {self._what()} nowhere, or only where "
                        f"the task keeps out; {self.parts_placed} part(s) placed") + self._unsorted_after(report))
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
                # The pick's move back to its look did not run, and the arm stands where its last grasp left it (S2,
                # 2026-10-08): no pick starts from there, whose looks would be judged on that frame alone.
                stands = _stands_at(report)
                if stands:
                    return self._problem(TaskStop.RECOVERY_NEEDS_PERSON, (
                        f"the pick's move back to its look did not run, and the arm stands where the pick left it "
                        f"({stands}); no pick starts from there, nothing more is commanded and a person decides what "
                        "happens next: " + _summary(report)))
                if checked:
                    # The check look saw a part it did not pick: no failure, and the next pick looks from every look.
                    continue
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
            self._place(report, self._place_of_the_part(report))
            if once:
                return self._finish(TaskStop.FINISHED, "the part is placed and the arm is back "
                                    + ("home" if self.plan.return_to == "home" else f"at {self._return_said()}"))
            if self.parts_placed >= self.plan.max_parts:
                return self._finish(TaskStop.PART_LIMIT, (
                    f"{self.parts_placed} parts placed, the most one task places; start another to go on"))
            # The part placed was the only target its pick's first look counted: the next pick checks from there alone.
            check = self.plan.options.check_look and _targets_at_the_first_look(report) == 1

    # ---- refusals before anything moves -------------------------------------------------------------------

    def _refuse_the_plan(self) -> None:
        for place in self.places:
            name = place.at.pose
            if name is not None and name not in self.poses:
                raise TaskRefused("unknown_pose", f"the place names the pose {name!r}, and no pose of that name was "
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
        from src.robot.core.arm_capabilities import CarriesPayload, chose_no_carried_part  # noqa: PLC0415
        from src.robot.execution.motion import route_of  # noqa: PLC0415

        route = route_of(self.arm)
        if not route.runs:
            # Robot.place and Robot.home refuse this arm before any command: a part picked here could not be set down.
            raise TaskRefused("route_refused", (
                "a task sets down and returns from every part it picks, and the place and the return refuse this arm "
                f"before any command, so a part picked here would stay in the jaws: {route.reason}"))
        # A cell that chose to model no carried part judges a closed hand as an empty one (the owner, 2026-10-05), and
        # carries its parts so (the owner, 2026-10-07): only a model left on that cannot model the part is refused.
        if isinstance(self.arm, CarriesPayload) and not chose_no_carried_part(self.arm):
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
        if self.sorts and not _grounds_a_phrase(self.service):
            # The rule of a part is the kind its pick went for, as the detector called it: words only a cell that
            # grounds a phrase has.
            raise TaskRefused("bad_request", (
                "a sort tells its kinds apart by the detector's words, and this cell's perception grounds no phrase "
                "(the rehearsal scene, or a detector of fixed classes), so no part could be told whose rule it goes "
                "by; name one kind of part, or sort on a cell that grounds a phrase"))
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
        if self._camera_places():
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
        for place in self.places:
            if place.at.pose is None:
                continue
            screen = self._screen_place(place)
            if screen.is_error:
                refused.append(f"the place pose {place.at.pose!r}: {screen.detail}")
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

    def _screen_place(self, place: _Place) -> Any:
        import numpy as np  # noqa: PLC0415

        from src.geometry import Frame, Pose  # noqa: PLC0415
        from src.robot.execution.place_target import Screen  # noqa: PLC0415

        name = str(place.at.pose)
        try:
            at = self.arm.fk(self.poses[name])
        except (RobotError, RuntimeError, OSError, ValueError) as exc:
            screen = Screen(None, "guard_refused", f"the arm cannot say where the pose puts the tool: "
                                                   f"{type(exc).__name__}: {exc}")
        else:
            place.tcp = at
            # Where the tool stands for the longest part the cell declares: the taught pose raised by its hang below the
            # tool (the fingertips' reach and the declared length past them), and the configuration the arm would choose
            # for it (the place goes to a pose, not to the taught joints).
            rise = self.hang_mm
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
            # phrase alone one of them; where they lie after it, and the one part singled out alone (pick_phrase). The
            # detector echoes the phrase, which maps onto the object the task named (a which names its noun); a task
            # that names none takes every part it grounds, called "object" where the detector's words allow.
            from src.robot.execution.autonomous_grasp.prompt import PickPrompt  # noqa: PLC0415

            if self.sorts:
                prompt = self._sort_prompt(PickPrompt)
            else:
                phrase = pick_phrase(named, which=self.plan.which, source=self.plan.source)
                prompt = (PickPrompt(phrase=phrase, target_label=named, object_labels=(named,))
                          if named else PickPrompt(phrase=phrase, target_label=None, object_labels=("object",)))
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
        from src.robot.grasping.recovery.exclusion_zones import ExclusionRegion  # noqa: PLC0415

        for place in self.places:
            if place.at.pose is None or self.plan.scope != "until_empty" or place.tcp is None:
                continue
            if _kinematic(self.arm):
                drop_xy = (float(place.tcp.position_mm[0]), float(place.tcp.position_mm[1]))
                place.region = zones.keep_out_region(ExclusionRegion.circle(
                    drop_xy, self.plan.pose_keep_out_mm, reason=f"the drop at pose {place.at.pose!r}"))
            else:
                # An fk that answers where the arm stands says nothing of where the pose puts the tool: a circle about
                # it would keep out whatever lies under the arm now (the rehearsal's one part), not the drop.
                logger.warning(
                    "task: no keep-out circle about the drop at pose %r: %s has no forward kinematics of its own "
                    "(has_native_fk is False), so where the pose puts the tool is not known, and a part set down "
                    "there may be picked again", place.at.pose, type(self.arm).__name__)

    def _sort_prompt(self, prompt_type: Any) -> Any:
        """What a sort's picks ground (the owner, 2026-10-09): every rule's kind in one call, each a phrase of one class
        list (``class_list_prompt``: "each separate green part | each separate red part"), the detector's words mapped
        onto the kinds (``object_labels``), and a target any of them (``target_labels``, where the cell's prompt takes
        it): a part no rule clearly claims (the detector's ``ambiguous``, or a word no rule names) is no target."""
        # Here, not at the module's top: a task of one kind loads no model package.
        from src.models.vlm.qwen import class_list_prompt  # noqa: PLC0415

        rules = self.plan.rules
        phrase = class_list_prompt([pick_phrase(rule.object, which=rule.which, source=rule.source) for rule in rules])
        kinds = tuple(rule.object.strip() for rule in rules)
        keywords: dict[str, Any] = {"phrase": phrase, "target_label": None, "object_labels": kinds}
        if "target_labels" in {item.name for item in dataclasses.fields(prompt_type)}:
            keywords["target_labels"] = kinds
        else:
            logger.warning("task: this cell's pick prompt takes no target_labels, so the pick's gate passes every part "
                           "it grounds; a part no rule names is put back after its pick")
        return prompt_type(**keywords)

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

    def _pick(self, *, check: bool = False) -> "tuple[Any, int]":
        options = self.plan.options
        keywords: dict[str, Any] = {}
        look = self._looks()
        if look is not None:
            keywords["look"] = look
        if not options.multi_view or check:
            # The check look is a pick with multi-view off (Q11): its first look alone, where the return left the arm,
            # and no generated view.
            keywords["multi_view"] = False
        elif options.every_look:
            # "Alle Posen": every look of the pick, however safe an earlier one's grasp; never the check look's.
            keywords["every_look"] = True
        if options.both_faces:
            keywords["both_faces"] = True
        if self.follow is not None:
            # The cell follows its parts: the later looks find the first look's parts by their projected boxes, and the
            # first look follows what the last pick kept, where it hands any on.
            keywords["follow_looks"] = True
            memory = self._handed(check)
            if memory is not None:
                keywords["follow"] = memory
        before = self._detector_failures()
        report = self.service.pick(**keywords)  # inside the task's asking_nobody (run)
        self.moved, self.at_return = True, False
        self.last_report = report
        # What the next pick follows: this pick's first look, less the part it gripped, or nothing on a trigger.
        self.memory = self._kept_after(report)
        if options.record_views:
            report = self._with_views(report)
            self.last_report = report
        if bool(getattr(report, "succeeded", False)):
            self.holding = True
        self.hooks.pick_done(self.part_number, self.picks, report)
        return report, before

    def _handed(self, check: bool) -> "KeptScene | None":
        """What this pick's first look follows (``robot.grasping.follow_parts``): the parts the last pick kept, or
        ``None`` where it grounds them again. The check look always grounds: the end of a task is always asked of the
        detector (the owner, 2026-10-09). So does a pick ``refresh_every_picks`` picks after the last grounding."""
        memory = self.memory
        if memory is None or check:
            return None
        every = int(getattr(self.follow, "refresh_every_picks", 0) or 0)
        if every > 0 and self.picks - memory.grounded_at_pick >= every:
            logger.info("task: pick %d grounds its parts again: %d pick(s) since the grounding at pick %d "
                        "(refresh_every_picks %d)", self.picks, self.picks - memory.grounded_at_pick,
                        memory.grounded_at_pick, every)
            return None
        return memory

    def _kept_after(self, report: Any) -> "KeptScene | None":
        """What the pick after ``report``'s follows: the parts its first look grounded or followed, less the part it
        gripped (``KeptScene.of_look``, ``without``); ``None`` where the next pick grounds them again.

        The owner's triggers (2026-10-09): a push, a blocker cleared, a recovery, a try that sent motion and failed (a
        grip that closed on nothing among them), the arm left where a try left it, a gripped part that was not among
        the kept parts, and no part left to keep. A pick that failed with nothing sent keeps the scene as its first look
        saw it. A halt, a stop or a disconnect ends the task, and the memory with it.
        """
        if self.follow is None:
            return None
        why = _why_grounded_again(report)
        looked = getattr(self.service, "looked_around", None)
        if not why and looked is None:
            why = "the pick kept no looks"
        if why:
            logger.info("task: the pick after pick %d grounds its parts again: %s", self.picks, why)
            return None
        from src.robot.perception.kept_scene import KeptScene  # noqa: PLC0415

        zones = getattr(getattr(self.service, "campaign", None), "zones", None)
        regions = tuple(zones.regions()) if zones is not None and callable(getattr(zones, "regions", None)) else ()
        judged = getattr(looked, "judged", None)
        try:
            kept = KeptScene.of_look(
                looked, label=getattr(self.orchestrator, "target_label", None), regions=regions,
                prompt=str(getattr(getattr(self.orchestrator, "perception", None), "prompt", "") or ""),
                pick=self.picks, support_mm=getattr(judged, "support_height_mm", None),
                workspace=getattr(self.tree, "workspace_limits", None), max_shift_mm=float(self.follow.max_shift_mm),
                max_creep_mm=float(self.follow.max_creep_mm), before=self.memory)
            if kept is not None and bool(getattr(report, "succeeded", False)):
                kept = kept.without(getattr(judged, "target_cloud_base_mm", None))
                if kept is None:
                    why = "the part it gripped is not among the parts its first look kept"
        except Exception as exc:  # noqa: BLE001 (a memory that cannot be built grounds a pick, never ends a task)
            logger.exception("task: what pick %d kept could not be read", self.picks)
            kept, why = None, f"what it kept could not be read ({type(exc).__name__}: {exc})"
        if kept is not None and not kept.parts:
            kept, why = None, "every part its first look kept is taken: the end is asked of the detector"
        if kept is None:
            logger.info("task: the pick after pick %d grounds its parts again: %s", self.picks,
                        why or "its first look kept no part to follow")
        else:
            logger.info("task: the pick after pick %d follows %d part(s) kept from look %s (grounded at pick %d)",
                        self.picks, len(kept.parts), kept.look, kept.grounded_at_pick)
        return kept

    def _asking_nobody(self) -> Any:
        from src.robot.core.gripper import toggle_without_sensor_of  # noqa: PLC0415

        toggle = toggle_without_sensor_of(self.gripper)
        asking = getattr(toggle, "asking_nobody", None) if toggle is not None else None
        return asking("a task never asks a person where the jaws stand; the step stops here, and the task with it") \
            if callable(asking) else nullcontext()

    def _with_views(self, report: Any) -> Any:
        """``report`` with where its looks were kept in ``telemetry['views_file']``, where they were.

        On a cell that writes in the background (``writes_in_the_background``, a real cell's), the looks are taken now,
        before the next pick lets them go, and written by the process's background writer while the task goes on; the
        file said is where they will be."""
        from src.robot.execution.autonomous_grasp.record_logging import background_writer  # noqa: PLC0415
        from src.robot.execution.autonomous_grasp.report import AutonomousGraspReport  # noqa: PLC0415

        writer = background_writer() if getattr(self.service, "writes_in_the_background", False) is True else None
        where = keep_pick_views(self.service, report, name=f"task-part{self.part_number}-pick{self.picks}",
                                writer=writer)
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
        # A pick that says where its last grasp left the arm needs a person too (``report.needs_person``), but the task
        # names that place itself, after it asked whether the hand still holds a part (``_drive``).
        if getattr(report, "needs_person", False) is True and not _stands_at(report):
            self._problem(TaskStop.RECOVERY_NEEDS_PERSON, (
                "a recovery stopped where the arm stands, so the task stops; nothing more is commanded and a person "
                "decides what happens next: " + _summary(report)))

    # ---- the rule of each part (a sort, the owner, 2026-10-09) ---------------------------------------------------

    def _place_of_the_part(self, report: Any) -> _Place:
        """Where the part a pick gripped goes: a task of one kind's one place; in a sort, the place of the rule whose kind
        the pick went for (the label of the part it gripped, ``PickReport.target_label``, case and blanks aside), said
        (``task.rule``). A part whose kind no rule names, or a pick that does not say, goes back where it was gripped
        and the task asks (``target_not_found``): the pick's gate lets no such part through, and nobody can say where
        one goes."""
        if not self.sorts:
            return self.places[0]
        label = getattr(getattr(report, "pick_report", None), "target_label", "")
        label = label.strip() if isinstance(label, str) else ""
        rules = self.plan.rules
        index = next((number for number, rule in enumerate(rules) if label and _words(rule.object) == _words(label)),
                     None)
        if index is None:
            said = f"the pick gripped a part the detector called {label!r}" if label else \
                "the pick gripped a part and did not say which kind it went for"
            self._put_back_and_ask(report, TaskStop.TARGET_NOT_FOUND, (
                f"{said}, and no rule of the sort names its kind ({self._what()}), so nobody can say where it goes"))
        assert index is not None  # narrowed: a part no rule names went back above
        rule = rules[index]
        place = next(place for place in self.places if index in place.rules)
        kind = rule.object.strip()
        goes = f"to pose {place.at.pose!r}" if place.at.pose is not None else f"into the {place.named}"
        self._say(TaskEvent.RULE, f"Part {self.part_number} is {_a(kind)} {kind}: it goes {goes}.",
                  part=self.part_number, rule=index, object=kind, place=place.where, place_label=place.named)
        return place

    def _unsorted_after(self, report: Any) -> str:
        """In a sort that ends with nothing left: the parts left where they lie that no rule clearly claims, said
        (``task.unsorted``) and counted on the report; what the finishing sentence adds, ``""`` for none and for a task
        of one kind (the owner, 2026-10-09: "liegen lassen, am Ende nennen").

        Two counts, the larger said: what the last empty pick's first look saw that no rule clearly claims
        (``PickReport.unclaimed_labels``: the detector's ``ambiguous``, a word no rule names), whose words the sentence
        names, and what one locate of every part from where the arm stands sees outside every place of the task
        (:meth:`_left_where_they_lie`), which counts a part of no rule's kind too: the sort's class list never boxes
        it."""
        if not self.sorts:
            return ""
        labels = getattr(getattr(report, "pick_report", None), "unclaimed_labels", ())
        labels = tuple(str(label) for label in labels) if isinstance(labels, (tuple, list)) else ()
        left = self._left_where_they_lie()
        count = max(len(labels), left or 0)
        logger.info("task: the sort left %d part(s) where they lie: %d label(s) no rule claims at the last pick's first "
                    "look, %s by one locate of every part", count, len(labels),
                    "none counted" if left is None else f"{left} seen")
        if count <= 0:
            return ""
        self.unsorted = count
        named = sorted(set(labels))
        said = f"{count} part(s) no rule clearly claims stay where they lie" + (f" ({', '.join(named)})" if named
                                                                              else "")
        self._say(TaskEvent.UNSORTED, f"{said}.", count=count, labels=named)
        return f"; {said}"

    def _left_where_they_lie(self) -> "int | None":
        """How many parts the camera sees from where the arm stands at the end of a sort, outside every place the task
        keeps out (its bins, the circles about its drops, the parts it laid there): one locate of every part, where the
        parts lie as the first rule says (:func:`pick_phrase` with no object), and nothing moved. A thing wider than
        :data:`LEFT_OVER_MAX_MM` on a side is what the parts stand on, and one outside the cell's workspace
        (``robot.workspace_limits``) no part of the task's, a marker or a tool beside it: neither is a part left. ``None``
        where nothing could be counted: no camera to look with, or a locate that raised, said in the log and never a
        reason to fail a task that finished."""
        import numpy as np  # noqa: PLC0415

        locators = list(self.locators)
        if not locators:
            try:
                locators = list(locators_for_service(self.service))
            except Exception as exc:  # noqa: BLE001 (a count of what is left never fails a task that finished)
                logger.info("task: what the sort left is not counted: no camera to look with (%s: %s)",
                            type(exc).__name__, exc)
                return None
        wrist = [locator for locator in locators if bool(getattr(locator, "on_the_wrist", False))]
        if not (wrist or locators):
            return None
        locator = (wrist or locators)[0]
        phrase = pick_phrase("", source=self.plan.source)
        zones = getattr(getattr(self.service, "campaign", None), "zones", None)
        regions = tuple(zones.regions()) if zones is not None and callable(getattr(zones, "regions", None)) else ()
        try:
            located = locator.locate(phrase)
        except Exception as exc:  # noqa: BLE001 (a count of what is left never fails a task that finished)
            logger.warning("task: what the sort left is not counted: the locate of %r raised %s: %s", phrase,
                           type(exc).__name__, exc)
            return None
        # Only what stands in the cell's workspace is a part left (robot.workspace_limits): a marker, a tool or a cable
        # beside it is no part of the task's (the review of 2026-10-09).
        limits = getattr(self.tree, "workspace_limits", None)
        try:
            box = None if limits is None else tuple(float(getattr(limits, name)) for name in (
                "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"))
        except (AttributeError, TypeError, ValueError):
            box = None
        count = 0
        for obj in tuple(getattr(located, "objects", ()) or ()):
            centre = getattr(obj, "centre_mm", None)
            points = np.asarray(getattr(obj, "points_base_mm", np.empty((0, 3))), dtype=np.float64).reshape(-1, 3)
            if centre is None or not points.shape[0]:
                continue
            if any(region.contains(centre) for region in regions):
                continue
            if float(np.max(np.ptp(points[:, :2], axis=0))) > LEFT_OVER_MAX_MM:
                continue
            if box is not None and not (box[0] <= float(centre[0]) <= box[1] and box[2] <= float(centre[1]) <= box[3]
                                        and box[4] <= float(centre[2]) <= box[5]):
                continue
            count += 1
        return count

    # ---- the camera's target --------------------------------------------------------------------------------

    def _camera_places(self) -> "list[_Place]":
        """The task's places a camera finds, in the order its rules first name them."""
        return [place for place in self.places if place.at.camera is not None]

    def _others(self, place: _Place) -> "tuple[str, ...]":
        """The phrases of every camera place but ``place``: where the detector looks at one bin of several, it grounds
        them all in one locate (``together``), so a second bin in view is never taken for this one (the owner,
        2026-10-09). None for a task of one camera place, whose locates are as they always were."""
        return tuple(str(other.at.camera) for other in self._camera_places() if other is not place)

    def _task_looks(self) -> "tuple[Any, ...]":
        """The looks a camera place is looked for from: the cell's own (home, on a wrist that names none), the first
        alone with multi-view off (Q11)."""
        from src.robot.execution.looks import HOME, looks_of  # noqa: PLC0415

        configured = self._looks()
        looks = tuple(looks_of(configured)) if configured is not None else (HOME,)
        return looks if self.plan.options.multi_view else looks[:1]

    def _known_for(self, place: _Place, *, first: bool) -> "KeptTarget | None":
        """The bin an earlier task kept for ``place``: the caller's ``known_target`` for the first camera place, else
        the one ``known_targets`` names by its phrase; ``None`` where none was handed in."""
        if first and self.known_target is not None:
            return self.known_target
        return next((kept for phrase, kept in self.known_targets.items() if _words(phrase) == _words(place.at.camera)),
                    None)

    def _survey(self) -> None:
        from src.robot.execution.looks import look_label  # noqa: PLC0415

        places = self._camera_places()
        phrases = [str(place.at.camera) for place in places]
        phrase = phrases[0]
        named = _listed([f"the {each}" for each in phrases], "and")
        looks = self._task_looks()
        wrist = any(bool(getattr(locator, "on_the_wrist", False)) for locator in self.locators)
        self._say(TaskEvent.SURVEY_STARTED, f"Looking for {named} "
                  + (f"from {len(looks)} look(s) " if wrist else "with the fixed camera ") + "before the first pick.",
                  phrase=phrase, phrases=phrases, looks=[look_label(look) for look in looks] if wrist else [])
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
            # The bins alone: the pick grounds every part with its own phrase and counts them itself, so a count here
            # was one more grounding for a chat line (2026-10-08). A bin an earlier task kept is looked at first. The
            # places of a sort are found together, one locate of all those still missing per look (2026-10-09).
            if len(places) == 1:
                found: Any = survey(self.arm, self.locators, phrase, looks=looks, may_move=may_move,
                                    known=self._known_for(places[0], first=True))
            else:
                found = survey_places(self.arm, self.locators, phrases, looks=looks, may_move=may_move, known={
                    str(place.at.camera): self._known_for(place, first=index == 0)
                    for index, place in enumerate(places)})
        except (RobotError, RuntimeError, OSError, ValueError) as exc:
            self.moved = self.moved or wrist
            self._problem(TaskStop.CELL_FAULT, f"the camera could not look for {named} ({type(exc).__name__}: {exc})")
        if wrist and (found.looks_tried or found.refused or found.stopped is not None):
            self.moved, self.at_return = True, False
        if found.halted:
            # Ended before a motion: said as what stopped it, a halt or a Disconnect before the operator's stop.
            stopped = self._may_not_move()
            if stopped is None and stop_asked:
                self._finish(TaskStop.STOPPED_AFTER_PART, f"the task stopped while it looked for {named}, as the "
                                                          "operator asked; nothing was picked")
            self._problem(*(stopped or (TaskStop.HALTED, found.halted)))
        if found.stopped is not None:
            self._after_a_failed_motion(TaskStop.CELL_FAULT, (
                f"the arm did not reach look {found.stopped_look} while it looked for {named}, and nothing else "
                f"was commanded after it: {found.stopped.status.value}: {found.stopped.message}"))
        # Every place as a survey of its own: a task of one place's is the survey itself. Every place a camera finds is
        # found before the first pick, or the task picks nothing and names each place it is missing (2026-10-09).
        each = [(place, found if len(places) == 1 else found.survey_of(str(place.at.camera))) for place in places]
        missing = [(place, one) for place, one in each if one.kept is None]
        if missing:
            if self._detector_failures() > before:
                self._problem(TaskStop.DETECTOR_FAILED, f"the detector failed while the task looked for {named}, so "
                                                        "'not found' is no answer")
            for place, one in missing:
                self._say(TaskEvent.TARGET_MISSING, f"No {place.at.camera} was seen from "
                          + (f"{len(one.looks_tried)} look(s)" if wrist else "the fixed camera")
                          + ": the task picks nothing.", phrase=str(place.at.camera), looks_tried=list(one.looks_tried))
            self._finish(TaskStop.TARGET_NOT_FOUND, (
                f"no {' and no '.join(str(place.at.camera) for place, _one in missing)} was seen from "
                f"{', '.join(found.looks_tried) or 'where the camera stands'}; nothing was picked"))
        zones = self.service.campaign.zones
        for place, one in each:
            kept = one.kept
            assert kept is not None  # narrowed: a survey that missed a place ended the task above
            known = one.known if one.known is not None and one.known.followed else None
            said = f"Found the {kept.label}" + ("" if kept.score is None else f" ({kept.score:.2f})")
            said += f" at look {kept.look_label}" if kept.look_label else " with the fixed camera"
            if known is not None and known.by == "depth":
                said += ", where the last task left it: its rim reads in depth and colour as it did"
            elif known is not None:
                said += f", {0.0 if known.moved_mm is None else known.moved_mm:.0f} mm from where the last task left it"
            self._say(TaskEvent.TARGET_FOUND, said + ".", target=kept.to_dict(), look=kept.look_label,
                      parts_seen=one.parts_seen, known=known is not None, by="" if known is None else known.by,
                      image_png=kept.image_png, phrase=str(place.at.camera))
            nominal = nominal_drop(self.arm, kept, hang_mm=self.hang_mm, air_mm=place.air_mm,
                                   standoff_mm=self.standoff_mm, natural_axis=self.natural)
            if not nominal.ok:
                self._finish(TaskStop.TARGET_UNREACHABLE, f"{nominal.reason}; nothing was picked")
            place.kept = place.surveyed = kept
            place.region = zones.keep_out_region(kept.keep_out_region())

    def _check(self, place: _Place) -> TargetCheck:
        # Against the bin the survey found (or found again in its stead), never a later sighting: a bin that creeps is
        # followed only within the bound of where it was found, and its size and rim are that bin's. Its rim is read
        # first where it was last seen; where the detector looks, it grounds every place of a sort (together).
        assert place.kept is not None and place.surveyed is not None  # only a camera place checks its target
        before = self._detector_failures()
        try:
            check = recheck(self.locators, place.surveyed, last=place.kept, inside=self._reads_where_the_part_goes(),
                            together=self._others(place))
        except (RobotError, RuntimeError, OSError, ValueError) as exc:
            self._problem(TaskStop.CELL_FAULT, f"the camera could not look at the {place.kept.label} again "
                                               f"({type(exc).__name__}: {exc})")
        if check.seen is None and self._detector_failures() > before:
            self._problem(TaskStop.DETECTOR_FAILED, f"the detector failed while the task looked at the "
                                                    f"{place.kept.label} again, so 'not seen' is no answer")
        self._checked(place, check)
        return check

    def _checked(self, place: _Place, check: TargetCheck) -> None:
        """Say a check of ``place``'s target (``task.target_checked``), and follow what it followed: its region kept out
        in place of the place's last, its sighting the one the drops go to, and what it read where the part goes."""
        assert place.kept is not None
        image = check.seen.image_png if check.seen is not None else None
        self._say(TaskEvent.TARGET_CHECKED, _checked_said(place.kept.label, check), moved_mm=check.moved_mm,
                  followed=check.followed, by=check.by, target=None if check.seen is None else check.seen.to_dict(),
                  phrase=str(place.at.camera or ""), image_png=image)
        if check.kept is not None:
            self._keep(place, check.kept)
        place.inside = check.inside
        if check.inside is not None:
            logger.info("task: the %s's check read %s", place.kept.label, check.inside.said)

    def _keep(self, place: _Place, kept: KeptTarget) -> None:
        """Follow ``kept`` as ``place``'s bin: its region kept out of the picks in place of the place's own last one,
        every other place's region kept (the sorting map's F4, 2026-10-09: no pick takes a sorted part back out of
        another rule's bin), and the sighting the drops go to."""
        zones = self.service.campaign.zones
        if place.region is not None:
            zones.forget_region(place.region)
        place.region = zones.keep_out_region(kept.keep_out_region())
        place.kept = kept

    def _reads_where_the_part_goes(self) -> bool:
        """Whether a check of the camera place's target reads where the part goes on the frame it checks on: a box's
        inside for a part set down below its rim (``robot.place.release_in_a_box``), a flat top's spots for parts laid
        side by side (``robot.place.side_by_side``)."""
        return self.rules.release_in_a_box == "below_the_rim" or bool(self.rules.side_by_side)

    def _check_a_fixed_cameras_target(self) -> None:
        """A fixed camera's targets are looked at again before each pick, while the arm stands clear of its view: every
        camera place it keeps. A bin lost there is looked for again where the camera stands (:meth:`_found_again`), and
        only one found nowhere ends the task."""
        for place in self._camera_places():
            kept = place.kept
            if kept is None or kept.on_the_wrist:
                continue
            check = self._check(place)
            if check.kept is not None:
                continue
            found = self._found_again(place, check)
            if found is not None and found.kept is not None:
                continue
            place.lost = True
            nowhere = "" if found is None else ", and was found nowhere"
            self._say(TaskEvent.TARGET_LOST, f"{_lost_said(kept.label, check)}{nowhere}: the task ends before the next "
                                             "pick.", look=None, why=check.why,
                      phrase=str(place.at.camera or ""))
            self._finish(TaskStop.TARGET_LOST, f"the {kept.label} was lost before the pick{nowhere}: {check.render()}")

    def _found_again(self, place: _Place, check: TargetCheck) -> "Relocated | None":
        """Look for ``place``'s bin again where ``check`` lost it (``place_target.relocate``; the owner, 2026-10-09:
        "wenn sie dies nicht mehr tut, dann kann er seine Ablage nochmal neu errechnen"), say what that came to
        (``task.target_relocated``), and keep a bin found again in the old one's stead: its region in place of the old,
        the bin every later check measures against, the sighting the drop goes to. ``None`` where the cell keeps the
        old rule (``robot.place.relocate`` off) and nothing was looked for.

        On a wrist the task drives the search's looks, the part in its jaws, each through the arm's judged verb with a
        halt or a cell taken down read first: a look refused before anything was sent is skipped and said, one that may
        have moved the arm ends the task where it stands as a carry that failed does. A fixed camera locates where it
        stands, and nothing moves. A bin standing in another place's region (every other place's, the circles about
        taught drops among them; never its own) is not taken for it, nor one of another size or colour."""
        if not self.rules.relocate:
            return None
        from src.robot.execution.generated_view import refused_before_sending  # noqa: PLC0415
        from src.robot.execution.looks import look_label, move_to_look  # noqa: PLC0415
        from src.robot.execution.motion import MotionOutcome  # noqa: PLC0415

        surveyed, last = place.surveyed, place.kept
        assert surveyed is not None and last is not None  # a camera place found its bin before its first pick
        label = last.label
        avoid = [other.region for other in self.places if other is not place and other.region is not None]
        search = relocate(self.locators, surveyed, looks=self._task_looks(), check=check, last=last, avoid=avoid,
                          together=self._others(place))
        before = self._detector_failures()
        for look in search:
            if look is not None:
                self._go_on_or_end()
                moved = move_to_look(self.arm, look)
                self.at_return = False
                if not moved.ok:
                    if moved.outcome is MotionOutcome.CAMERA_WORLD_UNAVAILABLE:
                        self._problem(TaskStop.CELL_FAULT, f"a camera could not vouch for the cell on the way to look "
                                                           f"{look_label(look)}, where the {label} was looked for "
                                                           f"again: {moved.message}")
                    if refused_before_sending(moved):
                        search.skip(look, f"{moved.status.value}: {moved.message}")
                        continue
                    self._after_a_failed_motion(TaskStop.PART_STILL_HELD, (
                        f"the carry to look {look_label(look)}, where the {label} was looked for again, failed with "
                        f"the part in the jaws, and the arm may have moved part of the way: {moved.status.value}: "
                        f"{moved.message}"))
            try:
                search.at(look)
            except (RobotError, RuntimeError, OSError, ValueError) as exc:
                self._problem(TaskStop.CELL_FAULT, f"the camera could not look for the {label} again "
                                                   f"({type(exc).__name__}: {exc})")
        found = search.result()
        if found.kept is None and self._detector_failures() > before:
            self._problem(TaskStop.DETECTOR_FAILED, f"the detector failed while the task looked for the {label} again, "
                                                    "so 'found nowhere' is no answer")
        if found.kept is not None:
            self._keep(place, found.kept)
            place.surveyed = found.kept
        self._say(TaskEvent.TARGET_RELOCATED, _relocated_said(label, found), **found.to_dict(),
                  image_png=None if found.kept is None else found.kept.image_png)
        return found

    # ---- the place ------------------------------------------------------------------------------------------

    def _place(self, report: Any, place: _Place) -> None:
        if place.at.pose is not None:
            name = place.at.pose
            drop = self._pose_drop(report, place)
            self._said_drop(drop, place)
            if not drop.ok:
                stop = TaskStop.PART_DOES_NOT_FIT if drop.refusal == "does_not_fit" else TaskStop.POSE_REFUSED
                self._put_back_and_ask(report, stop, f"the part was not set down at pose {name!r}: {drop.reason}")
            self._place_at(drop, place)
            return
        kept = place.kept
        assert kept is not None  # a camera place found its target before its first pick
        if kept.look is None:  # a fixed camera's: checked before the pick, while the arm stood out of its view
            self._drop_into(report, place)
            return
        if self._carry() == "over_the_rim" and self._over_the_rim(report, place):
            return
        self._via_the_look(report, place)

    def _via_the_look(self, report: Any, place: _Place) -> None:
        """Carry the part to the look ``place``'s bin was kept from, hold every frame taken there, check the bin, and
        drop the part into it. A bin the check lost is looked for again (:meth:`_found_again`), the hold of this look
        ended first, so the frames held next are those of the look the arm stands at: found, the part is carried to the
        look it was found from and the bin is checked there as before, the drop planned anew over it. Once per part:
        lost again or found nowhere, or with ``robot.place.relocate`` off, the part goes back where it was gripped and
        the task asks."""
        from src.robot.core.keep_out import holding_views  # noqa: PLC0415
        from src.robot.execution.generated_view import refused_before_sending  # noqa: PLC0415
        from src.robot.execution.looks import move_to_look  # noqa: PLC0415
        from src.robot.execution.motion import MotionOutcome  # noqa: PLC0415

        searched = False
        while True:
            kept = place.kept
            assert kept is not None and kept.look is not None  # a bin a wrist camera keeps, from one of its looks
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
                        f"the arm could not carry the part to look {kept.look_label}, where the {kept.label} is "
                        f"checked: {moved.status.value}: {moved.message}"))
                self._after_a_failed_motion(TaskStop.PART_STILL_HELD, (
                    f"the carry to look {kept.look_label} failed with the part in the jaws, and the arm may have moved "
                    f"part of the way: {moved.status.value}: {moved.message}"))
            with holding_views(self.arm) as held:
                if not held:
                    # Not seen: the bin cannot be checked against frames the world does not hold. The console's three
                    # words for a lost target (not_seen, moved_too_far, footprint_changed); the sentence says the frames.
                    place.lost = True
                    self._say(TaskEvent.TARGET_LOST, f"The frames of look {kept.look_label} could not be held in the "
                                                     "arm's world, so the bin is not checked there: the part goes back "
                                                     "where it was grasped.", look=kept.look_label, why="not_seen",
                              phrase=str(place.at.camera or ""))
                    self._put_back_and_ask(report, TaskStop.TARGET_LOST, (
                        f"the arm's world could not hold the frames of look {kept.look_label}, so the {kept.label}'s "
                        "walls would not be in the world the part goes in against"))
                check = self._check(place)
                if check.kept is not None:
                    self._drop_into(report, place)
                    return
                if searched or not self.rules.relocate:
                    self._lost_at_the_drop(report, place, kept, check)
            # The bin was lost, and the hold of this look has ended: the search moves the arm to its looks.
            searched = True
            found = self._found_again(place, check)
            if found is None or found.kept is None:
                self._lost_at_the_drop(report, place, kept, check, nowhere=True)

    def _lost_at_the_drop(self, report: Any, place: _Place, kept: KeptTarget, check: TargetCheck, *,
                          nowhere: bool = False) -> None:
        """End the task asking (``target_lost``) for a bin the check before the drop lost (``kept``, the sighting the
        task followed; ``nowhere``: looked for again and found nowhere): the part goes back where it was gripped."""
        place.lost = True
        again = ", and it was found nowhere" if nowhere else ""
        self._say(TaskEvent.TARGET_LOST, f"{_lost_said(kept.label, check)}{again}: the part goes back where it was "
                                         "grasped.", look=kept.look_label, why=check.why,
                  phrase=str(place.at.camera or ""))
        self._put_back_and_ask(report, TaskStop.TARGET_LOST, (
            f"the {kept.label} was lost at the drop{' and found nowhere' if nowhere else ''}: {check.render()}"))

    def _drop_into(self, report: Any, place: _Place) -> None:
        kept = place.kept
        assert kept is not None
        drop = self._camera_drop(report, place)
        self._said_drop(drop, place)
        if not drop.ok:
            stop = TaskStop.PART_DOES_NOT_FIT if drop.refusal == "does_not_fit" else TaskStop.TARGET_UNREACHABLE
            self._put_back_and_ask(report, stop, f"the part was not set down in the {kept.label}: {drop.reason}")
        self._place_at(drop, place)

    def _camera_drop(self, report: Any, place: _Place, *, inside: Any = UNSET) -> DropPlan:
        """The drop into ``place``'s target (``place_target.drop_plan``), the hang from what the part stood on
        (:meth:`_part_bottom`): below a box's rim (``robot.place.release_in_a_box``) from what the last check read inside
        it (``inside``, the check's where given), at the next free spot of a flat top (``robot.place.side_by_side``);
        straight down with the declared length where the pick reported no grasp; nothing set down blind with neither."""
        kept = place.kept
        assert kept is not None
        grasp = report.grasp_pose
        if grasp is None:
            if self.length_mm is not None:
                return nominal_drop(self.arm, kept, hang_mm=self.hang_mm, air_mm=place.air_mm,
                                    standoff_mm=self.standoff_mm, natural_axis=self.natural)
            return DropPlan(kind="camera", pose=None, standoff_mm=self.standoff_mm, rim_mm=kept.rim_mm,
                            air_mm=place.air_mm, refusal="unreachable", reason=(
                                "the pick reported no grasp pose and safety.planning_world.payload.length_mm is "
                                "undeclared, so how far the part hangs is unknown and nothing is set down blind"))
        cloud = _judged_cloud(self.service)
        keywords: dict[str, Any] = {
            "part_bottom_mm": self._part_bottom(grasp).mm, "air_mm": place.air_mm, "standoff_mm": self.standoff_mm,
            "natural_axis": self.natural, "part_cloud_mm": cloud}
        if self.rules.release_in_a_box == "below_the_rim":
            keywords.update(below_rim_mm=float(self.rules.below_the_rim_mm),
                            inside=place.inside if not chosen(inside) else inside,
                            opening_margin_mm=float(self.rules.opening_margin_mm), hand=self.open_hand)
        if not (self.rules.side_by_side and kept.opening_mm is None):
            return drop_plan(self.arm, kept, grasp, **keywords)
        from src.robot.execution.place_target import top_spot_reader, top_spots  # noqa: PLC0415

        part, reach = self._reach(grasp, cloud)
        if reach is None:
            logger.warning("task: nothing says how far the part reaches, so it is set down over the middle of the %s, "
                           "as before, not beside the parts there", kept.label)
            return drop_plan(self.arm, kept, grasp, **keywords)
        pitch = self._pitch(reach)
        candidates = top_spots(kept, pitch_mm=pitch, reach_mm=part if part is not None else reach)
        read = top_spot_reader(place.inside if not chosen(inside) else inside)
        return self._at_a_spot(candidates, reach, part, read, "camera", f"the {kept.label}",
                               lambda spot: drop_plan(self.arm, kept, grasp, spot=spot, **keywords), place.spots)

    def _pose_drop(self, report: Any, place: _Place) -> DropPlan:
        """The drop at ``place``'s taught pose (``place_target.pose_drop``), the hang from what the part stood on
        (:meth:`_part_bottom`, which keeps the set-down air over the taught point where it was measured): with
        ``robot.place.side_by_side`` on, at the first spot of the place about the taught pose the camera reads free, or
        where it reads none, that this task has not filled (:meth:`_at_a_spot`)."""
        from src.robot.perception.locator import SET_DOWN_AIR_MM  # noqa: PLC0415

        name = str(place.at.pose)
        grasp = report.grasp_pose
        bottom = self._part_bottom(grasp)
        keywords: dict[str, Any] = {
            "part_bottom_mm": bottom.mm, "length_mm": self.length_mm, "standoff_mm": self.standoff_mm,
            "fingertips_mm": self.fingertips_mm, "air_mm": float(SET_DOWN_AIR_MM) if bottom.measured else 0.0}
        taught = self.poses[name]
        if not self.rules.side_by_side:
            return pose_drop(self.arm, taught, grasp, **keywords)
        from src.robot.execution.place_target import (  # noqa: PLC0415
            _heading_of,
            grid_spots,
            look_points,
            views_spot_reader,
        )

        at = place.tcp
        part, reach = self._reach(grasp, _judged_cloud(self.service)) if grasp is not None else (None, None)
        if at is None or not _kinematic(self.arm) or reach is None:
            logger.warning("task: the part is let go at pose %r itself, as before, not beside the parts there: %s", name,
                           "nothing says how far the part reaches" if reach is None
                           else "the arm does not say where the taught pose puts the tool")
            return pose_drop(self.arm, taught, grasp, **keywords)
        grid = self.rules.grid
        xy = (float(at.position_mm[0]), float(at.position_mm[1]))
        candidates = grid_spots(xy, _heading_of(at) or 0.0, rows=int(grid.rows), columns=int(grid.columns),
                                pitch_mm=self._pitch(reach))
        looked = getattr(self.service, "looked_around", None)
        judged = getattr(looked, "judged", None)
        read = views_spot_reader(look_points(getattr(looked, "views", ())), getattr(judged, "support_model", None), xy)
        return self._at_a_spot(candidates, reach, part, read, "pose", f"the place about pose {name!r}",
                               lambda spot: pose_drop(self.arm, taught, grasp, spot=spot, **keywords), place.spots)

    def _at_a_spot(self, candidates: "Sequence[tuple[float, float]]", reach: float, part: "float | None", read: Any,
                   kind: str, where: str, drop_at: "Callable[[Any], DropPlan]",
                   laid: "Sequence[tuple[tuple[float, float], float]]") -> DropPlan:
        """The drop at the first of ``candidates`` a part reaching ``reach`` lies on clear of the parts this task laid
        there (``laid``, the place's spots) and of what the camera reads (``read``, ``None`` where no camera reads the
        place), ``drop_at`` a spot: a spot whose drop no configuration reaches is passed over for the next. None free is
        a drop of ``kind`` that ``does_not_fit`` (``where`` says the place); free ones none reaches, the last
        ``unreachable`` one."""
        from src.robot.execution.place_target import choose_spot  # noqa: PLC0415

        margin = float(self.rules.spacing_margin_mm)
        remaining = list(candidates)
        refused: "DropPlan | None" = None
        why = "the place has no spot"
        while remaining:
            spot, why = choose_spot(remaining, reach_mm=reach, margin_mm=margin, placed=laid, read=read)
            if spot is None:
                break
            drop = drop_at(spot)
            if drop.ok:
                logger.info("task: the part is laid at (%.0f, %.0f) mm on %s, the spot the %s chose", spot.xy[0],
                            spot.xy[1], where, spot.by)
                return dataclasses.replace(drop, spot_reach_mm=part if part is not None else reach)
            refused = drop
            remaining = [xy for xy in remaining if (float(xy[0]), float(xy[1])) != spot.xy]
        if refused is not None:
            return refused
        return DropPlan(kind=kind, pose=None, standoff_mm=self.standoff_mm, refusal="does_not_fit",
                        reason=f"{where} has no spot left the part lies on beside the others: {why}")

    def _part_bottom(self, grasp: "Pose | None") -> Any:
        """What the part's hang below the tool is measured from (``robot.place.part_bottom``): the declared support, or
        what the pick's own looks read under the part (``place_target.part_bottom``), where that stands below the
        grasp."""
        from src.robot.execution.place_target import PartBottom, part_bottom  # noqa: PLC0415

        declared = PartBottom(self.part_bottom_mm, False, f"the declared support at {self.part_bottom_mm:.1f} mm")
        if self.rules.part_bottom != "measured" or grasp is None:
            return declared
        judged = getattr(getattr(self.service, "looked_around", None), "judged", None)
        bottom = part_bottom(getattr(judged, "target_cloud_base_mm", None), getattr(judged, "support_model", None),
                             declared_mm=self.part_bottom_mm)
        if bottom.measured and float(grasp.position_mm[2]) - bottom.mm <= 0.0:
            logger.warning("task: the part's bottom read at %.1f mm does not stand below the grasp at %.1f mm, so its "
                           "hang is measured from %s", bottom.mm, float(grasp.position_mm[2]), declared.said)
            return declared
        logger.info("task: the part's hang is measured from %s", bottom.said)
        return bottom

    def _reach(self, grasp: "Pose", cloud: Any) -> "tuple[float | None, float | None]":
        """How far the part reaches from the tool (``place_target.part_reach_mm``) and how far it or the open hand
        reaches, whichever is more, mm; ``None`` for what nobody can say."""
        from src.robot.execution.place_target import part_reach_mm  # noqa: PLC0415

        part = part_reach_mm(cloud, grasp)
        hand = self.open_hand.reach_mm if self.open_hand is not None else None
        known = [value for value in (part, hand) if value is not None]
        return part, (max(known) if known else None)

    def _pitch(self, reach: float) -> float:
        """How far apart two spots of a flat place stand: the cell's grid spacing, else twice what a part reaches and
        the spacing margin."""
        spacing = self.rules.grid.spacing_mm
        return float(spacing) if spacing is not None else 2.0 * float(reach) + float(self.rules.spacing_margin_mm)

    def _carry(self) -> str:
        """How the part is carried to a bin a wrist camera found: the task's ``carry``, else the cell's."""
        return str(self.plan.options.carry or self.rules.carry)

    def _over_the_rim(self, report: Any, place: _Place) -> bool:
        """Carry the part straight over the rim to ``place``'s bin (``carry`` ``over_the_rim``, the motion map's C8, the
        owner's switch of 2026-10-08 night), and place it; ``True`` once the part was placed, put back, or the task ended.
        ``False`` where it goes via the bin's look instead, as before, the part still in the jaws and the arm where it
        stood or straight above it, said in the log.

        Where one of the pick's looks was the bin's (``place_target.the_bins_look``), the bin is checked on that look's
        frame, its rim's depth and colour as a check at the look reads them first, with no detector and no new frame
        (``recheck_on_look``), and the drop planned from it. The arm's world holds the pick's frames again for the place
        (``hold_pick_views_again``), so the line into the bin is judged against the walls that look saw. The part goes up,
        then straight over to the drop's standoff, each a line the guard judges sample by sample, at one height: its
        bottom the rim air and ``robot.place.rim_floor_margin_mm`` over the rim and over everything the pick's looks saw
        on the way (``highest_on_the_way``, the part's own place left out), never lower than the standoff or where it
        stands. A bin unsure on that frame, a way the looks did not see, a world that cannot hold the frames again, a
        drop refused there, or a line refused before anything was sent carry the part via the look instead; a line that
        failed once sent ends the task where the arm stands, as a carry to the look does."""
        import numpy as np  # noqa: PLC0415

        from src.geometry import Frame, Pose  # noqa: PLC0415
        from src.robot.execution.generated_view import refused_before_sending  # noqa: PLC0415
        from src.robot.execution.motion import MotionOutcome  # noqa: PLC0415
        from src.robot.execution.place_target import (  # noqa: PLC0415
            WAY_SEEN_SHARE,
            highest_on_the_way,
            look_points,
            recheck_on_look,
            the_bins_look,
        )

        kept, surveyed = place.kept, place.surveyed
        assert kept is not None and surveyed is not None

        def via_the_look(why: str) -> bool:
            logger.info("task: the part is carried to look %s, not straight over the rim: %s", kept.look_label, why)
            return False

        grasp = report.grasp_pose
        world = getattr(self.arm, "live_planner_world", None)
        again, forget = getattr(world, "hold_pick_views_again", None), getattr(world, "forget_pick_views", None)
        if grasp is None:
            return via_the_look("the pick reported no grasp pose, so how far the part hangs is not known")
        if not (callable(again) and callable(forget)):
            return via_the_look("the arm's world cannot hold the frames of the pick's looks again for the place")
        looked = getattr(self.service, "looked_around", None)
        view = the_bins_look(looked, kept)
        if view is None:
            return via_the_look(f"none of the pick's looks was look {kept.look_label}, where the {kept.label} was found")
        try:
            check, unsure = recheck_on_look(view, surveyed, last=kept, inside=self._reads_where_the_part_goes())
        except (RobotError, RuntimeError, OSError, ValueError) as exc:
            return via_the_look(f"the pick's look could not be read again ({type(exc).__name__}: {exc})")
        if check is None:
            return via_the_look(f"the {kept.label} is not sure on the pick's look {kept.look_label}: {unsure}")
        drop = self._camera_drop(report, place, inside=check.inside)
        if not drop.ok or drop.pose is None or drop.hang_mm is None:
            return via_the_look(f"the drop planned on the pick's look was not one to take: {drop.reason}")
        here = self.arm.get_tcp_pose()
        cloud = _judged_cloud(self.service)
        _part, reach = self._reach(grasp, cloud)
        matrix = np.asarray(drop.pose.to_matrix(), dtype=np.float64)
        standoff = matrix[:3, 3] - matrix[:3, 2] * float(drop.standoff_mm)
        part_xy = None if cloud is None else np.asarray(cloud, dtype=np.float64).reshape(-1, 3)[:, :2]
        top, seen = highest_on_the_way(look_points(getattr(looked, "views", ())), here.position_mm[:2], standoff[:2],
                                       reach_mm=(reach or 0.0) + float(self.rules.spacing_margin_mm),
                                       leave_out_xy=part_xy, leave_out_mm=10.0)
        if seen < WAY_SEEN_SHARE:
            return via_the_look(f"the pick's looks read {seen:.0%} of the way to the {kept.label}, and at least "
                                f"{WAY_SEEN_SHARE:.0%} must be read")
        floor = max(float(kept.rim_mm), float("-inf") if top is None else top) + place.air_mm \
            + float(self.rules.rim_floor_margin_mm) + float(drop.hang_mm)
        height = max(floor, float(standoff[2]), float(here.position_mm[2]))
        up = Pose(position_mm=np.array([float(here.position_mm[0]), float(here.position_mm[1]), height]),
                  quaternion_xyzw=np.asarray(here.quaternion_xyzw, dtype=np.float64), frame=Frame.BASE,
                  label="up over the rim floor")
        over = Pose(position_mm=np.array([float(standoff[0]), float(standoff[1]), height]),
                    quaternion_xyzw=np.asarray(drop.pose.quaternion_xyzw, dtype=np.float64), frame=Frame.BASE,
                    label=f"over the {kept.label}")
        if not bool(again()):
            forget()
            return via_the_look("the arm's world held none of the pick's frames to hold again")
        try:
            self._checked(place, check)
            self._say(TaskEvent.CARRY_STARTED, f"Carrying the part straight over to the {kept.label}, its bottom "
                                               f"{height - float(drop.hang_mm) - float(kept.rim_mm):.0f} mm over the rim.",
                      to_look=None, over_the_rim=True)
            lines = ([up] if height > float(here.position_mm[2]) + 0.5 else []) + [over]
            for line in lines:
                self._go_on_or_end()
                moved = self.robot.move(line, linear=True)
                self.at_return = False
                if moved.ok:
                    continue
                if moved.outcome is MotionOutcome.CAMERA_WORLD_UNAVAILABLE:
                    self._problem(TaskStop.CELL_FAULT, f"a camera could not vouch for the cell on the carry over the "
                                                       f"rim: {moved.message}")
                if refused_before_sending(moved):
                    return via_the_look(f"the line {line.label} was refused before anything was sent "
                                        f"({moved.status.value}: {moved.message})")
                self._after_a_failed_motion(TaskStop.PART_STILL_HELD, (
                    f"the carry over the rim failed with the part in the jaws, and the arm may have moved part of the "
                    f"way: {moved.status.value}: {moved.message}"))
            self._said_drop(drop, place)
            self._place_at(drop, place)
            return True
        finally:
            forget()

    def _said_drop(self, drop: DropPlan, place: _Place) -> None:
        self._say(TaskEvent.DROP_PLANNED, _drop_said(drop, place.kept.label if place.kept is not None else ""),
                  **drop.to_event())

    def _laid(self, drop: DropPlan, where: str, place: _Place) -> None:
        """Keep the spot a part was let go at on ``place`` (``robot.place.side_by_side``): the next part lies clear of
        it, and in an "until empty" task at a taught pose no pick takes it back (a circle about it kept out, Q4, as the
        drop's own)."""
        assert drop.spot_xy is not None
        reach = float(drop.spot_reach_mm or 0.0)
        place.spots.append((drop.spot_xy, reach))
        if drop.kind == "pose" and self.plan.scope == "until_empty":
            from src.robot.grasping.recovery.exclusion_zones import ExclusionRegion  # noqa: PLC0415

            self.service.campaign.zones.keep_out_region(ExclusionRegion.circle(
                drop.spot_xy, reach + float(self.rules.spacing_margin_mm) / 2.0,
                reason=f"the part laid at ({drop.spot_xy[0]:.0f}, {drop.spot_xy[1]:.0f}) mm by {where}"))

    def _place_at(self, drop: DropPlan, place: _Place) -> None:
        from src.robot.execution.handling import HandlingOutcome  # noqa: PLC0415

        assert drop.pose is not None
        self._go_on_or_end()
        where = place.where
        kind, _, name = where.partition(":")
        self._say(TaskEvent.PLACE_STARTED, f"Placing the part at {name!r}." if kind == "pose"
                  else f"Placing the part in the {place.kept.label if place.kept is not None else name}.", place=where)
        from src.robot.execution.motion import expecting_next  # noqa: PLC0415

        # The way back after the place, judged while the jaws open where the arm does (robot.motion.judge_next_leg).
        to = self.plan.return_to
        with expecting_next(self.arm, "home" if to == "home" else self.poses[to]):
            if drop.instead is None:
                placed = self.robot.place(drop.pose, standoff_mm=self.standoff_mm)
            else:
                from src.robot.execution import handling  # noqa: PLC0415

                # A drop below a box's rim: where the guard refuses the line into the box before anything is sent, the
                # part is let go over the rim instead, from the same standoff (Robot.place's own verb, with the drop over
                # it).
                placed = handling.place(self.robot, drop.pose, standoff_mm=drop.standoff_mm, instead=drop.instead)
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
        below: dict[str, Any] = {}
        if drop.instead is not None:
            # Which line in ran: the one into the box, or the one to the drop over its rim.
            below = {"below_the_rim": len(placed.poses) >= 2 and placed.poses[1] is drop.pose}
        self._say(TaskEvent.PLACED, "Placed: the jaws opened at the drop"
                  + ("" if not below or below["below_the_rim"]
                     else " over the rim: the line into the box was refused before anything was sent")
                  + (" (no sensor: the release is not measured)" if no_sensor else "")
                  + ("; the line out after it was refused, and the arm stands where the release left it" if line_out
                     else "") + ".",
                  outcome=placed.outcome.value, no_sensor=no_sensor, line_out_refused=line_out, **below)
        if drop.spot_xy is not None:
            self._laid(drop, where, place)
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
        # The bin followed last, for the next task into the same phrase; none where it was lost or never found: the first
        # camera place's, none where the task ended so, and every camera place's by its phrase, each of its own.
        cameras = self._camera_places()
        first = cameras[0].kept if cameras else None
        kept = None if stop in (TaskStop.TARGET_LOST, TaskStop.TARGET_NOT_FOUND) else first
        kept_targets = {str(place.at.camera).strip(): place.kept for place in cameras
                        if place.kept is not None and not place.lost}
        return TaskReport(stop=stop, sentence=sentence, parts_placed=self.parts_placed, picks=self.picks,
                          succeeded=self.succeeded, holding=self.holding, last_report=self.last_report,
                          at_return=self.at_return, kept_target=kept, kept_targets=kept_targets,
                          unsorted=self.unsorted)

    # ---- helpers ---------------------------------------------------------------------------------------------

    def _say(self, name: TaskEvent, said: str, **data: Any) -> None:
        self.hooks.event(name, said=said.encode("ascii", "backslashreplace").decode("ascii"), **data)

    def _what(self) -> str:
        if self.sorts:
            return _listed([repr(rule.object.strip()) for rule in self.plan.rules], "or")
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


def _fingertips_past_the_tcp_mm(tree: Any) -> float:
    """How far the fingertips reach below the tool, past the TCP along the approach, millimetres: the named hand's
    ``finger_ahead_mm`` past its grasp centre and that centre past the TCP (``hand_past_the_tcp_mm``). A part's bottom
    hangs at least this far below the tool, and ``payload.length_mm`` is measured past it. 0 where no hand resolves."""
    from src.config.schema.robot import RobotConfig  # noqa: PLC0415
    from src.robot.safety.planning.hand import hand_past_the_tcp_mm, planner_hand  # noqa: PLC0415

    if not isinstance(tree, RobotConfig):
        return 0.0
    try:
        hand = planner_hand(tree)
    except Exception:  # noqa: BLE001 (a hand nobody can resolve hangs nothing past the TCP; its refusal is said elsewhere)
        return 0.0
    if not chosen(hand):
        return 0.0
    return float(hand.jaw.finger_ahead_mm) + float(hand_past_the_tcp_mm(tree))


def _declared_length_mm(tree: Any) -> "float | None":
    """How far the cell declares its longest part hangs past the fingertips (``payload.length_mm``), or ``None``."""
    length = _number(getattr(getattr(getattr(getattr(tree, "safety", None), "planning_world", None), "payload", None),
                             "length_mm", None))
    return length if length is not None and length > 0.0 else None


def _place_rules(tree: Any) -> Any:
    """How the cell sets its parts down (``robot.place``), every default where its tree carries none: what a task did
    before the block existed."""
    from src.config.schema.robot.place_schema import RobotPlaceConfig  # noqa: PLC0415

    rules = getattr(tree, "place", None)
    return rules if isinstance(rules, RobotPlaceConfig) else RobotPlaceConfig()


def _open_hand_of(tree: Any) -> Any:
    """The open hand the cell names (``place_target.OpenHand``), ``None`` where none resolves."""
    from src.robot.execution.place_target import OpenHand  # noqa: PLC0415

    return OpenHand.of(tree)


def _judged_cloud(service: Any) -> Any:
    """The cloud of the part the last pick gripped, fused over its looks (``looked_around.judged``), BASE mm; ``None``
    where the pick kept none."""
    return getattr(getattr(getattr(service, "looked_around", None), "judged", None), "target_cloud_base_mm", None)


def _found_nothing(report: Any) -> bool:
    from src.robot.execution.autonomous_grasp.service import found_nothing  # noqa: PLC0415

    return bool(found_nothing(report))


def _only_kept_out(report: Any) -> bool:
    """Whether the pick saw only parts this task keeps out (its regions): an empty look. A part a zone skips after a
    failed pick still stands there, and the look that saw it is a failed pick."""
    from src.robot.execution.autonomous_grasp.service import only_kept_out  # noqa: PLC0415

    return bool(only_kept_out(report))


def _looks_that_saw_nothing(report: Any) -> int:
    """The looks an empty pick perceived from (``report.looks``: the visited ones; a refused look is not among them). A
    pick handed no look, or a report that names none, is one look."""
    looks = getattr(report, "looks", ()) or ()
    return max(1, len(looks)) if isinstance(looks, tuple) else 1


def _targets_at_the_first_look(report: Any) -> "int | None":
    """How many targets the first look of ``report``'s pick saw (``telemetry['targets_by_look']``, the pick loop's count:
    past the label gate, no surface, none kept out), or ``None`` where the pick does not say: a fixed camera's."""
    telemetry = getattr(report, "telemetry", None)
    counts = telemetry.get("targets_by_look") if isinstance(telemetry, Mapping) else None
    if not isinstance(counts, (list, tuple)) or not counts:
        return None
    first = counts[0]
    return first if isinstance(first, int) and not isinstance(first, bool) else None


def _follow_parts_of(tree: Any, service: Any = None) -> Any:
    """The cell's ``robot.grasping.follow_parts`` block where it follows its parts (``enabled``), else ``None``: the one
    the built cell carries (``BinPickingOrchestrator.follow_parts``, set at the build), else the tree's, for a service
    built without the overlays; a cell that says nothing grounds every pick, as before."""
    orchestrator = getattr(getattr(service, "runtime", None), "orchestrator", None)
    built = getattr(orchestrator, "follow_parts", UNSET) if orchestrator is not None else UNSET
    if "follow_parts" in getattr(type(orchestrator), "__dataclass_fields__", {}) and built is not UNSET:
        block = built
    else:
        block = getattr(getattr(tree, "grasping", None), "follow_parts", None)
    return block if getattr(block, "enabled", False) is True else None


def _why_grounded_again(report: Any) -> str:
    """Why the pick after ``report``'s grounds its parts again rather than follow them, or ``""``: the pick pushed a
    part, cleared a blocker, ran a recovery, sent a try that failed (a grip that closed on nothing among them) or left
    the arm where a try left it, each a scene that may have changed where the camera has not looked."""
    attempts = tuple(getattr(getattr(report, "pick_report", None), "attempts", ()) or ())
    telemetry = getattr(report, "telemetry", None)
    telemetry = telemetry if isinstance(telemetry, Mapping) else {}
    if any(getattr(attempt, "push", None) for attempt in attempts) or telemetry.get("pushes"):
        return "it pushed a part"
    if any(getattr(attempt, "blocker", None) for attempt in attempts):
        return "it cleared a blocker"
    if getattr(report, "recovery_actions", ()) or telemetry.get("recovery_trail_actions"):
        return "a recovery ran"
    tries = [row for attempt in attempts for row in (getattr(attempt, "tries", ()) or ())]
    sent = [bool(row.get("sent")) if isinstance(row, Mapping) else True for row in tries]
    done = bool(getattr(report, "succeeded", False))
    if (any(sent) and not done) or (done and any(sent[:-1])):
        return "a try sent motion and failed"
    stands = getattr(getattr(report, "pick_report", None), "stands_at", "")
    if isinstance(stands, str) and stands:
        return f"the arm stands where a try left it ({stands})"
    return ""


def _kinematic(arm: Any) -> bool:
    """Whether ``arm.fk`` is a kinematic model, so it says where a pose puts the tool: every arm but one whose
    capabilities say it has none (``has_native_fk`` False: the dummy, whose fk answers where it stands)."""
    return getattr(getattr(arm, "capabilities", None), "has_native_fk", True) is not False


def _carried_part_unmodelled(arm: Any) -> str:
    """Why the part a pick just gripped is not carried by the planner, or ``""`` where it is: what the pick's attach
    left in force (``payload_model``), on an arm that models a carried part at all. Reads only. A cell that chose to
    model no carried part (``chose_no_carried_part``) carries it as it chose, judged as an empty hand: the owner fixed
    this on the cell, 2026-10-08, where every pick stopped ``part_still_held`` after its grip."""
    from src.robot.core.arm_capabilities import CarriesPayload, PayloadModel, chose_no_carried_part  # noqa: PLC0415

    if not isinstance(arm, CarriesPayload):
        return ""
    if chose_no_carried_part(arm):
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


def _stands_at(report: Any) -> str:
    """Where a pick's last grasp left the arm when its move back to the look did not run (``PickReport.stands_at``,
    e.g. "standoff of grasp 2"); ``""`` for every other pick."""
    stands = getattr(getattr(report, "pick_report", None), "stands_at", "")
    return stands if isinstance(stands, str) else ""


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
    if check.followed and check.by == "depth":
        where = "where it was kept" if moved < 0.5 else f"where it was last seen, {moved:.0f} mm from where it was kept"
        return f"The {label} stands {where}: its rim reads in depth and colour as it did, so the detector was not asked."
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


def _relocated_said(label: str, found: "Relocated") -> str:
    """``task.target_relocated`` for a person: where the lost target was found again, or that it was found nowhere."""
    if found.kept is None:
        if not found.looks_tried:
            return f"The {label} was found nowhere: the camera saw none, and no other look was left to look from."
        return (f"The {label} was found nowhere: none of its size and colour was seen from "
                f"{', '.join(found.looks_tried)}.")
    moved = 0.0 if found.moved_mm is None else found.moved_mm
    where = ("where the check saw it" if found.by == "check"
             else f"at look {found.kept.look_label}" if found.kept.look_label else "with the fixed camera")
    return f"Found the {label} again {where}, {moved:.0f} mm from where it was kept: the drop is planned anew over it."


def _listed(items: "Sequence[str]", last: str) -> str:
    """``items`` as a person lists them, ``last`` the word before the last one: "a", "a and b", "a, b and c"."""
    items = list(items)
    if len(items) < 2:
        return "".join(items)
    return f"{', '.join(items[:-1])} {last} {items[-1]}"


def _a(word: str) -> str:
    """The article before ``word``: "an orange part", "a green part"."""
    return "an" if word[:1].casefold() in ("a", "e", "i", "o", "u") else "a"


def _drop_said(drop: DropPlan, label: str) -> str:
    """``task.drop_planned`` for a person."""
    if not drop.ok:
        return f"No drop: {drop.reason}."
    hang = 0.0 if drop.hang_mm is None else drop.hang_mm
    spot = "" if drop.spot_xy is None else (f", at the spot ({drop.spot_xy[0]:.0f}, {drop.spot_xy[1]:.0f}) mm the "
                                            f"{drop.spot_by} chose beside the parts there")
    if drop.kind == "pose":
        kept_air = "" if not drop.air_mm else f" and {drop.air_mm:.0f} mm of air"
        return f"The drop stands {hang:.0f} mm above the taught pose: the part's hang{kept_air}{spot}."
    air = 0.0 if drop.air_mm is None else drop.air_mm
    rim = 0.0 if drop.rim_mm is None else drop.rim_mm
    turned = ", turned along the cell's natural closing axis" if drop.turned else ""
    if drop.below_rim_mm is not None:
        return (f"The part is set down in the {label or 'target'}: its bottom {drop.below_rim_mm:.0f} mm under the rim at "
                f"{rim:.0f} mm, the part hanging {hang:.0f} mm{turned}; over the rim, should the line in be refused.")
    where = "on" if drop.spot_xy is not None else "over the middle of"
    over = f", over the rim and not below it: {drop.over_the_rim_why}" if drop.over_the_rim_why else ""
    return (f"The drop is {where} the {label or 'target'}: its rim at {rim:.0f} mm, the part hanging {hang:.0f} mm, "
            f"{air:.0f} mm of air{turned}{spot}{over}.")


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
