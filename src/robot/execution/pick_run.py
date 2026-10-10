"""A campaign of N picks: one connect around them, one verdict over them, one report about them.

The loop itself is six lines of ``for _ in range(n): service.pick()``. The knowledge is in the four
things around it.

1. The verdict rule. `PassRule` defaults to unanimity: every pick must succeed. The sim gate
   configures 80 % instead (`sim_schema.py` `pass_fraction: 0.8`) and does not take the service's
   own ``SUCCEEDED`` as sufficient evidence: it additionally requires a lift measured from the
   physics to clear `sim_schema.py` `lift_threshold_mm`, which defaults to 50.0. A cell with a
   `NullGripper` reports SUCCEEDED on every run, so the rule is an explicit argument here rather
   than a constant. A success the gripper did not measure (a jaw with no feedback wired) is counted
   as a success and said to be unmeasured, on its run line and under the result.

   A pick that ended on a controller that cannot move (a protective or emergency stop) stops the
   campaign as a fault of the cell does: the next run would otherwise start on an arm a person has
   to walk up to. So does a pick that ended on a hand that needs a person (a gripper that raised, a
   toggle that would not start on jaws it believes closed with nobody at a terminal to say otherwise),
   and one whose recovery stopped where the arm stands (a push of a failed part that stopped once
   something may have moved: ``needs_person``), since the next pick would drive the arm back to its look.

2. The connect belongs outside the loop. Connecting is motion: Robotiq activation is a calibration
   sweep of the full finger travel, the cross-process `CellLock` is taken and released once, and on
   a vacuum cell the ejector drops whatever is held. Ten campaigns of one are not one campaign of
   ten.

3. The teardown report is `None` while the block runs. ``ConnectedCell.teardown`` is assigned in
   ``__exit__``, so a caller who returns from inside the ``with`` never sees it, and it is the only
   place "the gripper did not release" is reported. It is on the report here.

4. Record logging is off by default. The shipped tree has ``record_log_path: null``, so
   `from_robot_config` wires nothing and a caller that wants a JSONL corpus asks for one. `Recording`
   is an argument with no default, and the report says which way it went.

Two more are the owner's, 2026-09-24. Where each pick looks from (``look``): a wrist camera sees what
the arm points it at, and a pick ends above its own grasp, so every pick of a wrist cell first moves to
the looks the program declares, else the ones its cell profile configures
(``robot.look_joint_positions_deg``), else home on a wrist camera (`src.robot.execution.looks`); a
fixed camera's arm moves only to looks the program or the profile names. A wrist camera's looks are
fused, each with the ones before, until a grasp is safe (2026-09-28). And ``put_back``: a part lifted is put back where it was grasped, so one part
serves a whole campaign; a part that could not be put back stops the campaign, because the next pick
would start with it in the hand.

And one more, the same day: ``view``, a `LiveView` (``src/camera/live_view.py``), shows every camera the
cell's build opened, one window each, while the campaign runs. The campaign renders the grasp overlay on
every pick (the service's debug image, put back as it was afterwards), pins each attempt's overlay on the
primary camera's window with the attempt's outcome, and says that outcome on every window. It is display
only: a window closed, or a view that raises, changes nothing the campaign does, and the view reads the
cameras through their display path, which changes nothing the picks measure.

Two switches of 2026-09-29, both off by default. ``both_faces`` asks every pick to see both jaw contact
faces of its chosen grasp before it grips, for safety-critical processes: a wrist camera looks on for them, and
a pick no view showed both of ends ``no_valid_grasp`` with nothing gripped, on a fixed camera as on the wrist.
``record_views`` keeps each pick's looks for training, one file per pick
(`src.robot.execution.record_views`), named after the pick's record.

And one of 2026-09-30: a campaign is what a cell's recovery counts across (``service.start_campaign``): the push
budgets of ``nudge_target`` (1 per part, 2 per pick, 5 per campaign) and the parts ``next_target`` skips.
``push_mm`` is how far a push moves a part, the config's 30 mm when unset, refused with a sentence above the cell's
ceiling or the hard cap of 50 mm and under 10 mm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np

from src.contracts import UNSET, Maybe, chosen
from src.robot.constants import create_robot_logger

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.geometry import Pose
    from src.robot.execution.cell import Cell
    from src.robot.execution.handling import HandlingReport
    from src.robot.execution.lifecycle import ConnectStage, TeardownReport
    from src.robot.execution.looks import Look, LookPose

__all__ = [
    "PassRule",
    "PickAttempt",
    "PickOutcome",
    "PickRun",
    "PickRunReport",
    "Recording",
    "configured_looks_of",
    "keep_pick_views",
]

#: PickRun's own lines, to the robot log and ``pick_run.log`` (RC6 of the cell-fix plan: they reached no file).
logger: logging.Logger = create_robot_logger(__name__, "pick_run.log")

#: The outcome string a service reports for a pick that worked. Compared as a string:
#: `AutonomousGraspOutcome` lives one layer up, and importing it here to compare an enum member
#: against a string that arrives off a report would buy nothing but an import.
_SUCCEEDED = "succeeded"


class PickOutcome(StrEnum):
    """How one attempt in a campaign ended, from this campaign's point of view.

    Not a copy of `AutonomousGraspOutcome`. That enum says what the pick did (no target, execution
    failed, verification failed); this says what the campaign learned, which has one extra state the
    other cannot have: an attempt that never ran because the campaign stopped.
    """

    SUCCEEDED = "succeeded"
    #: Ran and did not succeed. The service's own outcome is on the report beside it.
    FAILED = "failed"
    #: A fault of the cell ended the pick: `pick()` reported one (a camera that could not vouch, a
    #: controller link that dropped), the pick ended on a controller that cannot move (a protective
    #: or emergency stop, a power-off) or on a hand that needs a person, or an exception escaped it.
    #: The campaign stops; the cell still comes down.
    RAISED = "raised"
    #: The campaign was asked to stop before this attempt started.
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class PassRule:
    """When a campaign of N picks counts as a pass.

    A rule object rather than a bare number: the sim gate also refuses to take the service's own word for a success, and
    ``confirm`` is where that refusal fits. It has no default that accepts.

    Attributes:
        fraction (float): The share of attempts that must succeed, from 0 to 1; 1.0 is every one (default: 1.0).
        confirm (Callable[[Any], bool] | None): A second opinion per attempt: takes the service's report and answers
            whether it really happened. ``None`` takes the service's own outcome as the only evidence; the sim harness
            supplies one, since a cell with a ``NullGripper`` reports success on every run (default: None).
    """

    #: Fraction of attempts that must succeed, 1.0 for unanimity.
    fraction: float = 1.0
    #: A second opinion per attempt, taking the service's report and answering "did this really
    #: happen". `None` means the service's own outcome is the only evidence; the sim harness
    #: supplies one, because a cell with a `NullGripper` reports success on every run.
    confirm: "Callable[[Any], bool] | None" = None

    def accepts(self, attempts: "Sequence[PickAttempt]") -> bool:
        """Whether these attempts pass the rule.

        Args:
            attempts (Sequence[PickAttempt]): The campaign's attempts, cancelled ones included.

        Returns:
            bool: ``True`` where at least ``fraction`` of them passed; a campaign of zero passes.
        """
        if not attempts:
            # A campaign of zero is a pass: `--runs 0` connects, moves the gripper on activation,
            # picks nothing and exits 0.
            return True
        good = sum(1 for a in attempts if a.passed)
        return good >= self.fraction * len(attempts)

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """The rule as a person reads it.

        Returns:
            str: Such as ``"every attempt must succeed, independently confirmed"``.
        """
        confirmed = ", independently confirmed" if self.confirm is not None else ""
        if self.fraction >= 1.0:
            return f"every attempt must succeed{confirmed}"
        return f"at least {self.fraction:.0%} of attempts must succeed{confirmed}"


@dataclass(frozen=True, slots=True)
class Recording:
    """Where a campaign appends its attempt records, one JSON line each, or that it appends none.

    A noun rather than an optional path, so that "this run produced no corpus" is something the report states.

    Attributes:
        path (str): The JSONL file the records are appended to; ``""`` records nothing (default: "").
        provenance (Mapping[str, Any]): Stamped into every record: who ran it, which build, why (default: {}).
    """

    path: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        """Whether anything is recorded: ``True`` when a path is set."""
        return bool(self.path)

    @classmethod
    def off(cls) -> "Recording":
        """Record nothing: the default of a config-driven cell, whose shipped ``robot.yaml`` sets ``record_log_path:
        null``.

        Returns:
            Recording: One with no path.
        """
        return cls()

    @classmethod
    def to_file(cls, path: str, *, provenance: "Mapping[str, Any] | None" = None) -> "Recording":
        """Append every attempt's record to a file.

        Args:
            path (str): The JSONL file; it is created if missing and appended to otherwise.
            provenance (Mapping[str, Any] | None): Stamped into every record (default: None, none).

        Returns:
            Recording: One that records to ``path``.
        """
        return cls(path=str(path), provenance=dict(provenance or {}))

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """Where the records go, as a person reads it.

        Returns:
            str: ``"recording to <path>"`` or ``"recording nothing"``.
        """
        return f"recording to {self.path}" if self.enabled else "recording nothing"


@dataclass(frozen=True, slots=True)
class PickAttempt:
    """One attempt of a campaign, and what the campaign made of it, as ``on_attempt`` gets it.

    Attributes:
        index (int): Its number in the campaign, from 1.
        outcome (PickOutcome): ``SUCCEEDED``, ``FAILED``, ``RAISED`` (a fault of the cell stopped it) or ``CANCELLED``
            (the campaign stopped before it started).
        reported (str): The service's own outcome, verbatim; empty when it never ran (default: "").
        detail (str): The service's one-line reason for a failure; empty otherwise (default: "").
        hold_measured (bool | None): For a success: whether the hand measured the hold (``False`` rests on the close
            command alone); ``None`` otherwise (default: None).
        object_mm (tuple[float, float, float] | None): Where the object it went for was seen, BASE millimetres, the
            median of its mask's surface; ``None`` where no candidate reached the arm (default: None).
        grasp_pose (Pose | None): Where the tool closed, BASE, on a success; ``None`` otherwise (default: None).
        looks (tuple[str, ...]): The looks it moved to before it perceived, as they read (``home``, or joints in
            degrees); empty where it perceived from where the arm stood (default: ()).
        put_back (HandlingReport | None): How the part went back, on a campaign that puts it back; ``None`` otherwise
            (default: None).
        fused_views (tuple[str, ...]): The cameras (rig ids) whose views of the object were fused into the cloud its
            grasp was planned on, the grasp's camera first; empty for one view (default: ()).
        fused_objects (int): How many objects of that frame gained a second camera's surface (default: 0).
        looks_fused (tuple[str, ...]): The wrist camera's looks fused into the cloud its grasp was ranked on, the
            ranking look first; empty on a fixed camera (default: ()).
        jaw_faces_seen (tuple[bool, bool] | None): Whether each jaw's contact face of the chosen grasp was seen, (jaw 1,
            jaw 2); ``None`` where no grasp was judged (default: None).
        hand_eye_gap_mm (float | None): The hand-eye check of a wrist pick: how far apart two looks measure the part, in
            millimetres; ``None`` where fewer than two looks shared enough of it (default: None).
        views_file (str): Where this pick's looks were kept, with ``record_views``; empty otherwise (default: "").
        generated_view_deg (float | None): How far the one view a wrist pick generated turned about the part, in
            degrees; ``None`` where none was generated (default: None).
    """

    index: int
    outcome: PickOutcome
    #: The service's own outcome string, verbatim. Empty when the attempt never ran.
    reported: str = ""
    #: The service's own one-line reason for a failure. Empty otherwise.
    detail: str = ""
    #: For a success with a gripper, whether the gripper measured the hold (the service report's
    #: `hold_measured`). `False` is a success that rests on the close command alone; `None` where
    #: the attempt did not succeed or the service does not say.
    hold_measured: bool | None = None
    #: Where the object this attempt went for was seen, BASE mm (the service report's `object_centre_mm`): the
    #: median of its mask's surface. `None` where no candidate reached the arm or the service does not say.
    object_mm: "tuple[float, float, float] | None" = None
    #: The pose the tool closed at, in BASE, on an attempt that succeeded (the service report's `grasp_pose`): the
    #: grasp waypoint that was judged and ran. `None` otherwise.
    grasp_pose: "Pose | None" = None
    #: The looks the attempt moved to before it perceived, as they read (`home`, or joints in degrees). Empty where
    #: it perceived from where the arm stood.
    looks: tuple[str, ...] = ()
    #: How the part went back where it was grasped, on a campaign that puts it back (`PickRun.put_back`). `None`
    #: where nothing was put back.
    put_back: "HandlingReport | None" = None
    #: The cameras whose views of the object this attempt went for were fused into the one cloud its grasp was planned
    #: on, the camera the grasp is synthesised in first, as rig ids (the service report's `fused_views`). Empty where
    #: that cloud came from one camera: fusion off, no second camera delivered, or none identified that object; never
    #: one camera alone.
    fused_views: tuple[str, ...] = ()
    #: How many objects of the frame that grasp was planned in gained a second camera's surface, the object it went for
    #: or not (the service report's `fused_objects`); 0 where none did.
    fused_objects: int = 0
    #: The looks of a wrist camera whose views of the part were fused into the cloud its grasp was ranked on, the look
    #: it was ranked on first (the service report's `looks_fused`). Empty on a fixed camera.
    looks_fused: tuple[str, ...] = ()
    #: Whether the contact face each jaw closes on, of the chosen grasp, was seen: (jaw 1, jaw 2), the service report's
    #: `jaw_faces_seen`. `None` where no grasp was judged or the service does not say.
    jaw_faces_seen: "tuple[bool, bool] | None" = None
    #: The hand-eye check of a wrist pick, mm (the service report's `hand_eye_gap_mm`). `None` where fewer than two
    #: looks shared enough of the part.
    hand_eye_gap_mm: float | None = None
    #: Where this pick's looks were kept, on a campaign that keeps them (`PickRun.record_views`). Empty otherwise.
    views_file: str = ""
    #: How far the one view a wrist pick generated, once its declared looks were used up, turned about the part,
    #: degrees (the service report's `generated_view_deg`); its look is among `looks`. `None` where none was generated.
    generated_view_deg: float | None = None

    @property
    def passed(self) -> bool:
        """Whether it succeeded: ``outcome`` is ``SUCCEEDED``."""
        return self.outcome is PickOutcome.SUCCEEDED

    @property
    def unmeasured(self) -> bool:
        """A success whose hold the gripper did not measure."""
        return self.passed and self.hold_measured is False

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """The attempt as a person reads it: its outcome, then a line for each thing it measured.

        Returns:
            str: ASCII, indented to sit under a campaign's heading, no trailing newline.
        """
        lines = [f"  run {self.index}: {self.reported or self.outcome.value}" + (
            f"  {self.detail}" if self.detail else ""
        ) + ("  hold not measured" if self.unmeasured else "")]
        if self.looks:
            fused = f"; fused {' + '.join(self.looks_fused)}" if len(self.looks_fused) > 1 else ""
            lines.append(f"    looked from {'; '.join(self.looks)}{fused}")
        if self.jaw_faces_seen is not None:
            said = ", ".join(f"jaw {jaw} {'seen' if seen else 'not seen'}"
                             for jaw, seen in zip((1, 2), self.jaw_faces_seen))
            lines.append(f"    contact faces of the chosen grasp: {said}")
        if self.generated_view_deg is not None:
            lines.append(f"    generated one view, turned {self.generated_view_deg:+.0f} deg about the part")
        if self.hand_eye_gap_mm is not None:
            from src.robot.grasping.multiview.association import HAND_EYE_DRIFT_WARN_MM  # noqa: PLC0415

            drifted = self.hand_eye_gap_mm > HAND_EYE_DRIFT_WARN_MM
            lines.append(f"    hand-eye: the looks measure the part {self.hand_eye_gap_mm:.1f} mm apart"
                         + (f", more than {HAND_EYE_DRIFT_WARN_MM:.1f} mm: the calibration may have drifted, check it"
                            if drifted else ""))
        if self.object_mm is not None:
            lines.append("    object seen at ({:.1f}, {:.1f}, {:.1f}) mm BASE".format(*self.object_mm))
        if self.grasp_pose is not None:
            lines.append("    grasp closed at ({:.1f}, {:.1f}, {:.1f}) mm BASE".format(*self.grasp_pose.position_mm))
        # Only where a second camera added surface to something, so a single-view campaign prints as it always did.
        # Rig ids are the tree's words, escaped to keep the campaign's text ASCII.
        if self.fused_views:
            views = " + ".join(self.fused_views).encode("ascii", "backslashreplace").decode("ascii")
            lines.append(f"    fused views: {views} ({self.fused_objects} object(s) fused in that frame)")
        elif self.fused_objects:
            lines.append(f"    planned from one view ({self.fused_objects} other object(s) fused in that frame)")
        if self.put_back is not None:
            lines.append(f"    put back: {self.put_back.outcome.value}"
                         + ("" if self.put_back.ok else f", {self.put_back.message or 'see the place report'}"))
        if self.views_file:
            lines.append(f"    views kept in {self.views_file}".encode("ascii", "backslashreplace").decode("ascii"))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """The attempt as plain data.

        Returns:
            dict[str, Any]: Every attribute, ``json.dumps`` safe: the outcome as its value, the grasp pose as
                ``position_mm`` and ``quaternion_xyzw``, ``put_back`` as its outcome.
        """
        return {
            "index": self.index,
            "outcome": self.outcome.value,
            "reported": self.reported,
            "detail": self.detail,
            "hold_measured": self.hold_measured,
            "object_mm": None if self.object_mm is None else list(self.object_mm),
            "grasp_pose": None if self.grasp_pose is None else {
                "position_mm": [float(v) for v in self.grasp_pose.position_mm],
                "quaternion_xyzw": [float(v) for v in self.grasp_pose.quaternion_xyzw]},
            "looks": list(self.looks),
            "put_back": None if self.put_back is None else self.put_back.outcome.value,
            "fused_views": list(self.fused_views),
            "fused_objects": self.fused_objects,
            "looks_fused": list(self.looks_fused),
            "jaw_faces_seen": None if self.jaw_faces_seen is None else list(self.jaw_faces_seen),
            "hand_eye_gap_mm": self.hand_eye_gap_mm,
            "views_file": self.views_file,
            "generated_view_deg": self.generated_view_deg,
        }


@dataclass(frozen=True, slots=True)
class PickRunReport:
    """What a campaign did, whether it passed, and how the cell came down.

    Attributes:
        requested (int): How many picks were asked for.
        attempts (tuple[PickAttempt, ...]): Every attempt, cancelled ones included.
        rule (PassRule): The rule it was judged by.
        recording (Recording): Where its records went.
        teardown (TeardownReport | None): How the cell came down; ``None`` where the campaign did not own the connect
            (``from_service``) (default: None).
        last (Any): The last report the service produced, for ``layers_that_ran()``; ``None`` if nothing ran
            (default: None).
        error (str): A refusal that stopped the campaign before or during the picks; empty otherwise (default: "").
    """

    requested: int
    attempts: tuple[PickAttempt, ...]
    rule: PassRule
    recording: Recording
    #: How the cell came down. On the report, because it is `None` inside the `with` block and the
    #: only place "the gripper did not release" is ever reported.
    teardown: "TeardownReport | None" = None
    #: The last report the service produced, for `layers_that_ran()`. `None` if nothing ran.
    last: Any = None
    #: A refusal that stopped the campaign before or during the picks.
    error: str = ""

    @property
    def succeeded(self) -> int:
        """How many attempts succeeded."""
        return sum(1 for a in self.attempts if a.passed)

    @property
    def unmeasured(self) -> int:
        """Successes whose hold the gripper did not measure: each is the close command's word.

        On a jaw with no feedback wired, the owner's toggle Hand-E (2026-09-23), that is every success, and a
        count of successes alone would read as parts held.
        """
        return sum(1 for a in self.attempts if a.unmeasured)

    @property
    def attempted(self) -> int:
        """Attempts that actually ran, cancelled ones excluded.

        Separate from `succeeded` and `requested`: a campaign that raises on attempt 3 of 10 still
        has two results worth reporting.
        """
        return sum(1 for a in self.attempts if a.outcome is not PickOutcome.CANCELLED)

    @property
    def cancelled(self) -> int:
        """How many attempts never started, because the campaign was asked to stop."""
        return sum(1 for a in self.attempts if a.outcome is PickOutcome.CANCELLED)

    @property
    def raised(self) -> bool:
        """Whether a fault of the cell stopped the campaign (an attempt ``RAISED``)."""
        return any(a.outcome is PickOutcome.RAISED for a in self.attempts)

    @property
    def passed(self) -> bool:
        """Whether the campaign passed: no refusal, no fault, and its rule accepts the attempts."""
        return not self.error and not self.raised and self.rule.accepts(self.attempts)

    @property
    def clean_teardown(self) -> bool:
        """`True` when the cell never connected: nothing was left asserted because nothing was
        ever energised. `False` means a teardown ran and part of it failed.
        """
        return self.teardown is None or bool(getattr(self.teardown, "clean", True))

    @property
    def exit_code(self) -> int:
        """0 pass, 1 refused (before picking, or to go on without a part put back), 2 picked and did not pass, 3 a
        fault of the cell stopped it.

        The same four codes the command-line runner returns, derived from the report rather than
        branched at four `return` statements.
        """
        if self.error:
            return 1
        if self.raised:
            return 3
        return 0 if self.passed else 2

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """The whole campaign as a person reads it: each attempt, the verdict, the teardown.

        Returns:
            str: ASCII, no trailing newline. ``print(report)`` shows the same.
        """
        lines = [a.render() for a in self.attempts if a.outcome is not PickOutcome.CANCELLED]
        if self.error:
            lines.append(f"  REFUSED: {self.error}")
        lines.append("")
        lines.append(self.summary())
        return "\n".join(lines)

    def summary(self) -> str:
        """The verdict block alone: the count, the rule it was judged by, and what was recorded.

        The command-line runner prints its own banners between the picks and the teardown and needs only this tail.

        Returns:
            str: The tail of :meth:`render`, ASCII.
        """
        lines = [f"RESULT: {self.succeeded}/{self.requested} succeeded"]
        # What the count rests on, where the gripper measured nothing: a success count that reads as
        # parts held when no hold was ever measured is the report this line exists to prevent.
        if self.unmeasured:
            judged = ("confirm= judged each one" if self.rule.confirm is not None
                      else "each is the close command's word; confirm= is where a check of your own goes")
            lines.append(f"  {self.unmeasured} of {self.succeeded} success(es) with no hold measured by the "
                         f"gripper: {judged}")
        # The rule is printed so an operator reading the verdict can see which rule produced it:
        # unanimity and 80 % answer differently on the same nine successes out of ten.
        lines.append(f"  rule: {self.rule.render()}  ->  {'PASS' if self.passed else 'FAIL'}")
        if self.cancelled:
            lines.append(f"  {self.cancelled} attempt(s) never ran")
        # And whether a corpus exists: record logging is off by default, so "run ten picks and
        # measure the rate offline" produces nothing unless a caller asked for it.
        lines.append(f"  {self.recording.render()}")
        if self.teardown is not None and not self.clean_teardown:
            lines.append("  TEARDOWN NOT CLEAN: on a real cell an output may still be asserted")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """The campaign as plain data.

        Returns:
            dict[str, Any]: ``json.dumps`` safe with no custom encoder: the counts, the verdict, ``exit_code`` and each
                attempt as :meth:`PickAttempt.to_dict` writes it.
        """
        return {
            "requested": self.requested,
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "unmeasured": self.unmeasured,
            "cancelled": self.cancelled,
            "passed": self.passed,
            "exit_code": self.exit_code,
            "rule": {"fraction": self.rule.fraction, "confirmed": self.rule.confirm is not None},
            "recording": {"enabled": self.recording.enabled, "path": self.recording.path},
            "clean_teardown": self.clean_teardown,
            "error": self.error,
            "attempts": [a.to_dict() for a in self.attempts],
        }


def _checked_prompt(prompt: "Maybe[str]") -> "Maybe[str]":
    """A campaign's prompt, refused at the factory when it names nothing, before a cell is built.

    The detector refuses an empty phrase on every frame, so a campaign carrying one would build and
    connect a cell for picks that cannot ground anything.
    """
    if chosen(prompt) and not str(prompt).strip():
        raise ValueError("PickRun(prompt=...) names what to pick; an empty prompt grounds nothing")
    return prompt


@dataclass(frozen=True, slots=True)
class PickRun:
    """N picks against one cell, under one connect.

    ```python
    cell = Cell.from_tree(load_tree(), prompt="a red cube")
    report = PickRun.from_cell(cell, runs=10, recording=Recording.off()).execute()
    print(report)
    raise SystemExit(report.exit_code)
    ```

    Build it with :meth:`from_cell` (the campaign owns the connect) or :meth:`from_service` (you do), then
    :meth:`execute`. Exactly one of ``cell`` and ``service`` is set.

    Attributes:
        cell (Cell | None): The cell the campaign builds, connects, drives and takes down (default: None).
        service (Any): An already connected pick service, the caller owning its connect (default: None).
        runs (int): How many picks (default: 1).
        rule (PassRule): When the campaign passes (default: every attempt).
        recording (Recording): Where the records go (default: off).
        target_label (Maybe[str | None]): The label a pick target must carry (default: UNSET).
        prompt (Maybe[str]): What the campaign picks (default: UNSET).
        should_cancel (Callable[[], bool] | None): Asked before every attempt (default: None).
        announce (Callable[[ConnectStage], None] | None): Called with each connect stage (default: None).
        on_attempt (Callable[[PickAttempt], None] | None): Called as each attempt finishes (default: None).
        look (tuple[LookPose, ...]): Where each pick looks from (default: (), the configured looks).
        put_back (bool): Put each lifted part back (default: False).
        view (Any): A ``LiveView``, or ``None`` (default: None).
        both_faces (bool): Grip only once both jaw faces were seen (default: False).
        record_views (bool): Keep each pick's looks (default: False).
        push_mm (Maybe[float]): How far a push moves a part, in millimetres (default: UNSET).
    """

    #: Exactly one of these two is set. `cell` means "own the connect"; `service` means the caller
    #: has already connected and is responsible for taking the cell down.
    cell: "Cell | None" = None
    service: Any = None
    runs: int = 1
    rule: PassRule = field(default_factory=PassRule)
    #: No default at the factories: whether a campaign writes a corpus is stated, not inherited.
    recording: Recording = field(default_factory=Recording.off)
    #: The label a pick target must carry, for this campaign only. Set and cleared around the run.
    target_label: "Maybe[str | None]" = UNSET
    #: What this campaign picks: the phrase every camera grounds, the labels the detector's words map
    #: onto, and the label filter, all three set before the first pick and put back after the last,
    #: with no camera reopened and no model reloaded. `target_label` alone sets the filter and
    #: nothing the detector reads.
    prompt: "Maybe[str]" = UNSET
    #: Asked before every attempt. Returning True stops the campaign; the remaining attempts are
    #: reported CANCELLED rather than silently missing.
    should_cancel: "Callable[[], bool] | None" = None
    announce: "Callable[[ConnectStage], None] | None" = None
    #: Called as each attempt finishes, before the next one starts.
    #:
    #: `Cell` has no such hook and needs none: its four steps are separate methods, so narration is
    #: what a caller writes between two calls. This loop runs inside one verb, and an operator at a
    #: bench must see run 3 before run 4 starts rather than all ten at the end.
    on_attempt: "Callable[[PickAttempt], None] | None" = None
    #: Where each pick looks from before it perceives (`src.robot.execution.looks`). A wrist camera's
    #: looks go to the pick loop, which fuses each with the ones before and stops at the first safe
    #: grasp; a fixed camera tries them in turn until one finds something. Unset: the looks the cell
    #: profile configures (`service.configured_looks`), else the arm's home on a wrist camera, else,
    #: on a fixed camera, where it is mounted, with no motion. Nothing is said to the hand before a look.
    look: "tuple[LookPose, ...]" = ()
    #: Put each lifted part back where it was grasped (`service.put_back`), so one part serves the
    #: whole campaign. A part that does not go back stops the campaign.
    put_back: bool = False
    #: The cameras' windows (a `LiveView`), or None. Given one, the campaign hands it every camera the
    #: cell's build opened (``watch``) and starts it, renders the grasp overlay on every pick, pins each
    #: attempt's new overlay on the primary camera's window with the attempt's outcome (``show_image``)
    #: and says that outcome on every window (``note``). The caller closes it, after the campaign:
    #: ``with LiveView(show=...) as view:``. Display only: a view that raises changes nothing here.
    view: Any = None
    #: Ask every pick to see both jaw contact faces of its chosen grasp before it grips (`service.pick`'s `both_faces`,
    #: which says how a pick whose looks show both nowhere ends): the owner's switch for safety-critical processes.
    #: Off, the default and the fast one. A wrist camera looks on for both; a fixed camera grips only a grasp its
    #: cameras showed both faces of. The cell's natural orientation (`robot.natural_closing_axis`) and a motion naming a
    #: closing axis (`GraspMotion(closing_axis=...)`) choose the way round before the faces are judged, so both work
    #: with it. A cell whose motion turns every grasp about base Z before it closes
    #: (`GraspMotion(align_closing_to_base_x=True)`) would grip faces nobody judged: its first pick raises before anything
    #: moves, and the campaign stops on it.
    both_faces: bool = False
    #: Keep each pick's looks for training (`src.robot.execution.record_views`): the frames of its looks (RGB and
    #: depth), the tool pose stamped at each, the intrinsics and the target cloud the looks fused, one file per pick
    #: under `RECORD_VIEWS_DIR`, named after the pick's record (`attempt_id`). Off by default. A fixed camera has no looks
    #: to keep; a file that cannot be written is said and the campaign goes on.
    record_views: bool = False
    #: How far a push of a failed part moves it, in mm, on a cell whose recovery pushes (`nudge_target`, dense_clutter):
    #: unset, the config's `recovery.fixture.push_distance_mm` (30 mm, also where no fixture is declared). A request up
    #: to the cell's `recovery.fixture.max_nudge_mm` (the hard cap without a fixture) is taken as asked; above it,
    #: above the hard cap of 50 mm, or under 10 mm it is refused with a sentence, never shortened: the last two at the
    #: factory, the cell's ceiling as the campaign starts, before anything moves. The campaign's push budgets (1 per
    #: part, 2 per pick, 5 per campaign) start with it.
    push_mm: "Maybe[float]" = UNSET

    # --- two doors -----------------------------------------------------------------------------

    @classmethod
    def from_cell(
        cls,
        cell: "Cell",
        *,
        runs: int,
        recording: Recording,
        rule: "Maybe[PassRule]" = UNSET,
        target_label: "Maybe[str | None]" = UNSET,
        prompt: "Maybe[str]" = UNSET,
        should_cancel: "Callable[[], bool] | None" = None,
        announce: "Callable[[ConnectStage], None] | None" = None,
        on_attempt: "Callable[[PickAttempt], None] | None" = None,
        look: "Maybe[Look]" = UNSET,
        put_back: bool = False,
        view: Any = None,
        both_faces: bool = False,
        record_views: bool = False,
        push_mm: "Maybe[float]" = UNSET,
    ) -> "PickRun":
        """A campaign that builds, connects, drives and takes down a cell.

        The connect happens once, outside the loop: connecting is motion (a Robotiq activation sweeps the full finger
        travel, a vacuum cup asserts its ejector at once), so ten campaigns of one are not one campaign of ten.

        Args:
            cell (Cell): The cell, built or not; ``Cell.from_tree(tree, prompt=...)``.
            runs (int): How many picks, under one connect; 0 connects, picks nothing and passes.
            recording (Recording): Where each attempt's record line goes: ``Recording.to_file(path)`` or
                ``Recording.off()``. Stated, never inherited.
            rule (Maybe[PassRule]): When the campaign passes; unset is every attempt (default: UNSET).
            target_label (Maybe[str | None]): The label a pick target must carry, for this campaign only; it sets the
                filter alone, nothing the detector reads (default: UNSET).
            prompt (Maybe[str]): What this campaign picks: the phrase every camera grounds, the labels the detector's
                words map onto and the label filter, all set before the first pick and put back after the last (default:
                UNSET).
            should_cancel (Callable[[], bool] | None): Asked before every attempt; ``True`` stops the campaign, and the
                remaining attempts are reported ``CANCELLED`` (default: None).
            announce (Callable[[ConnectStage], None] | None): Called with each connect stage, for a console that shows
                progress (default: None).
            on_attempt (Callable[[PickAttempt], None] | None): Called as each attempt finishes, before the next starts
                (default: None).
            look (Maybe[Look]): Where each pick looks from before it perceives: joint poses (``JointPositions``, or
                ``"home"``). A wrist camera's looks are fused one after another until the first safe grasp; a fixed
                camera tries them in turn until one finds something. Unset: the looks the cell profile configures, else
                the arm's home on a wrist camera, else, on a fixed camera, where it is mounted, with no motion (default:
                UNSET).
            put_back (bool): Put each lifted part back where it was grasped, so one part serves the whole campaign; a
                part that does not go back stops it (default: False).
            view (Any): A ``LiveView`` showing every camera the cell opened and each attempt's grasp overlay, or
                ``None``. Display only; the caller closes it (default: None).
            both_faces (bool): Grip only once both jaw contact faces of the chosen grasp were seen: the switch for
                safety-critical processes; a wrist camera looks on for both (default: False).
            record_views (bool): Keep each pick's looks for training (frames, tool poses, intrinsics, the fused cloud),
                one file per pick under ``RECORD_VIEWS_DIR`` (default: False).
            push_mm (Maybe[float]): How far a push of a failed part moves it, in millimetres, on a cell whose recovery
                pushes; unset is the config's ``recovery.fixture.push_distance_mm`` (default: UNSET).

        Returns:
            PickRun: The campaign; nothing has moved yet. :meth:`execute` runs it.

        Raises:
            ValueError: A look list that names nothing, or ``push_mm`` above 50 mm or under 10 mm.
        """
        return cls(
            cell=cell,
            runs=runs,
            recording=recording,
            rule=rule if chosen(rule) else PassRule(),
            target_label=target_label,
            prompt=_checked_prompt(prompt),
            should_cancel=should_cancel,
            announce=announce,
            on_attempt=on_attempt,
            look=_checked_looks(look),
            put_back=bool(put_back),
            view=view,
            both_faces=bool(both_faces),
            record_views=bool(record_views),
            push_mm=_checked_push_mm(push_mm),
        )

    @classmethod
    def from_service(
        cls,
        service: Any,
        *,
        runs: int,
        recording: Recording,
        rule: "Maybe[PassRule]" = UNSET,
        target_label: "Maybe[str | None]" = UNSET,
        prompt: "Maybe[str]" = UNSET,
        should_cancel: "Callable[[], bool] | None" = None,
        on_attempt: "Callable[[PickAttempt], None] | None" = None,
        look: "Maybe[Look]" = UNSET,
        put_back: bool = False,
        view: Any = None,
        both_faces: bool = False,
        record_views: bool = False,
        push_mm: "Maybe[float]" = UNSET,
    ) -> "PickRun":
        """A campaign on a service that is already connected; the caller owns the connect and the teardown, so the
        report's ``teardown`` stays ``None``.

        Args:
            service (Any): The connected pick service, ``cell.service`` inside ``with cell.connected():``.
            runs (int): How many picks, under one connect; 0 connects, picks nothing and passes.
            recording (Recording): Where each attempt's record line goes: ``Recording.to_file(path)`` or
                ``Recording.off()``. Stated, never inherited.
            rule (Maybe[PassRule]): When the campaign passes; unset is every attempt (default: UNSET).
            target_label (Maybe[str | None]): The label a pick target must carry, for this campaign only; it sets the
                filter alone, nothing the detector reads (default: UNSET).
            prompt (Maybe[str]): What this campaign picks: the phrase every camera grounds, the labels the detector's
                words map onto and the label filter, all set before the first pick and put back after the last (default:
                UNSET).
            should_cancel (Callable[[], bool] | None): Asked before every attempt; ``True`` stops the campaign, and the
                remaining attempts are reported ``CANCELLED`` (default: None).
            on_attempt (Callable[[PickAttempt], None] | None): Called as each attempt finishes, before the next starts
                (default: None).
            look (Maybe[Look]): Where each pick looks from before it perceives: joint poses (``JointPositions``, or
                ``"home"``). A wrist camera's looks are fused one after another until the first safe grasp; a fixed
                camera tries them in turn until one finds something. Unset: the looks the cell profile configures, else
                the arm's home on a wrist camera, else, on a fixed camera, where it is mounted, with no motion (default:
                UNSET).
            put_back (bool): Put each lifted part back where it was grasped, so one part serves the whole campaign; a
                part that does not go back stops it (default: False).
            view (Any): A ``LiveView`` showing every camera the cell opened and each attempt's grasp overlay, or
                ``None``. Display only; the caller closes it (default: None).
            both_faces (bool): Grip only once both jaw contact faces of the chosen grasp were seen: the switch for
                safety-critical processes; a wrist camera looks on for both (default: False).
            record_views (bool): Keep each pick's looks for training (frames, tool poses, intrinsics, the fused cloud),
                one file per pick under ``RECORD_VIEWS_DIR`` (default: False).
            push_mm (Maybe[float]): How far a push of a failed part moves it, in millimetres, on a cell whose recovery
                pushes; unset is the config's ``recovery.fixture.push_distance_mm`` (default: UNSET).

        Returns:
            PickRun: The campaign; :meth:`execute` runs it.

        Raises:
            ValueError: A look list that names nothing, or ``push_mm`` above 50 mm or under 10 mm.
        """
        return cls(
            service=service,
            runs=runs,
            recording=recording,
            rule=rule if chosen(rule) else PassRule(),
            target_label=target_label,
            prompt=_checked_prompt(prompt),
            should_cancel=should_cancel,
            on_attempt=on_attempt,
            look=_checked_looks(look),
            put_back=bool(put_back),
            view=view,
            both_faces=bool(both_faces),
            record_views=bool(record_views),
            push_mm=_checked_push_mm(push_mm),
        )

    # --- the verb ------------------------------------------------------------------------------

    def execute(self) -> PickRunReport:
        """Run the campaign: build and connect the cell when it owns it, pick ``runs`` times, and always take the cell
        down again.

        Returns:
            PickRunReport: Every attempt, the verdict and the teardown; ``exit_code`` is 0 passed, 1 refused, 2 picked
                and did not pass, 3 a fault of the cell stopped it.

        A refusal or a fault is part of the report, never an exception: the cell still comes down. Nothing moves on its
        own after a stop.
        """
        if self.service is not None:
            return self._drive(self.service, teardown=None)
        if self.cell is None:  # pragma: no cover (neither factory can produce this)
            return PickRunReport(
                requested=self.runs, attempts=(), rule=self.rule, recording=self.recording,
                error="no cell and no service",
            )

        try:
            self.cell.build()
        except Exception as exc:  # noqa: BLE001 (a refusal to build is a verdict, not a crash)
            return PickRunReport(
                requested=self.runs, attempts=(), rule=self.rule, recording=self.recording,
                error=f"{type(exc).__name__}: {exc}",
            )

        session = self.cell.connected(announce=self.announce)
        try:
            session.__enter__()
        except Exception as exc:  # noqa: BLE001 (connect is a transaction and has rolled itself back)
            return PickRunReport(
                requested=self.runs, attempts=(), rule=self.rule, recording=self.recording,
                error=f"{type(exc).__name__}: {exc}",
            )
        try:
            report = self._drive(session.service, teardown=None)
        finally:
            # The exit is in a `finally` and the teardown is read after it. `session.teardown` is
            # assigned by `__exit__`, so reading it inside the block would always read `None`, which
            # is exactly the mistake a caller writing `with cell.connected() as live:` makes.
            session.__exit__(None, None, None)
        return PickRunReport(
            requested=report.requested, attempts=report.attempts, rule=report.rule,
            recording=report.recording, teardown=session.teardown, last=report.last,
            error=report.error,
        )

    def _announce(self, attempt: PickAttempt) -> None:
        if self.on_attempt is not None:
            self.on_attempt(attempt)

    def _start_campaign(self, service: Any) -> str:
        """Start the campaign on ``service`` (``start_campaign``: fresh push budgets, no part skipped, the push
        distance); why it was refused, or ``""``. A service that starts none is driven as it always was, unless a push
        distance was asked for, which it could not honour."""
        start = getattr(service, "start_campaign", None)
        if not callable(start):
            return ("this service takes no push distance (it has no start_campaign), so push_mm cannot be honoured"
                    if chosen(self.push_mm) else "")
        try:
            start(**({"push_mm": self.push_mm} if chosen(self.push_mm) else {}))
        except ValueError as refused:
            return f"push_mm refused: {refused}"
        return ""

    def _looks_for(self, service: Any) -> "Look | None":
        """What every pick of this campaign looks from: the program's looks, else the ones the cell profile configures,
        else home on a wrist camera, else none.

        The configured looks only as :func:`configured_looks_of` reads them, and `is True`, so a double that answers
        every attribute is read as neither; a service that does not say perceives from where the arm stands, as it
        always did.
        """
        from src.robot.execution.looks import HOME  # noqa: PLC0415

        if self.look:
            return self.look
        configured = configured_looks_of(service)
        if configured:
            return configured
        return HOME if getattr(service, "perceives_from_the_wrist", False) is True else None

    def _pick(self, service: Any, look: "Look | None") -> Any:
        """One pick, handed the campaign's looks and switch; a service handed neither is called as it always was."""
        keywords: dict[str, Any] = {} if look is None else {"look": look}
        if self.both_faces:
            keywords["both_faces"] = True
        return service.pick(**keywords)

    def _keep_views(self, service: Any, report: Any, index: int) -> str:
        """Keep the looks of the pick that just ran, on a campaign that asked (`record_views`); where, or `""`.

        :func:`keep_pick_views`, named `run<index>` where the pick's record names it nothing.
        """
        if not self.record_views:
            return ""
        return keep_pick_views(service, report, name=f"run{index:03d}")

    def _put_back(self, service: Any, report: Any) -> "tuple[HandlingReport | None, str]":
        """The put back of a part a pick lifted, and why it did not go back: `""` when it did or none was asked for.

        A put back that raised is a part not put back as well, said in the same words: the campaign stops on it with
        its report, and the cell still comes down.
        """
        if not self.put_back:
            return None, ""
        try:
            placed = service.put_back(report)
        except Exception as exc:  # noqa: BLE001 (the campaign stops, the cell still comes down)
            return None, f"raised {type(exc).__name__}: {exc}"
        if placed.ok:
            return placed, ""
        return placed, placed.outcome.value + (f": {placed.message}" if placed.message else "")

    def _drive(self, service: Any, *, teardown: "TeardownReport | None") -> PickRunReport:
        """The loop, and the three per-campaign settings that must be put back afterwards."""
        # The campaign first: its push distance is the first thing that can be refused against the cell's own ceiling,
        # before anything else is set. It starts the campaign's push budgets and the parts next_target skips.
        refused = self._start_campaign(service)
        if refused:
            return PickRunReport(requested=self.runs, attempts=(), rule=self.rule, recording=self.recording,
                                 teardown=teardown, error=refused)
        # The prompt next: it is the other setting that can refuse, and a refusal here leaves the other
        # two untouched. `set_prompt` returns what it replaced, and that is what goes back.
        previous_prompt: Any = None
        if chosen(self.prompt):
            previous_prompt = service.set_prompt(self.prompt)
        if self.recording.enabled:
            service.enable_record_logging(
                self.recording.path, provenance=dict(self.recording.provenance) or None
            )
        if chosen(self.target_label):
            service.set_target_label(self.target_label)

        attempts: list[PickAttempt] = []
        last: Any = None
        error = ""
        # Asked once: whether a camera sits on the wrist does not change between two picks. A service
        # handed no look is called as it always was, so a service that takes none still runs.
        look = self._looks_for(service)
        _screen_looks(service, look)
        # The cameras' windows, where the caller handed a view: every camera of the cell, the overlay rendered.
        shown = None if self.view is None else _OnTheView(self.view, service)
        try:
            for index in range(self.runs):
                if self.should_cancel is not None and self.should_cancel():
                    attempts.extend(
                        PickAttempt(index=i, outcome=PickOutcome.CANCELLED)
                        for i in range(index, self.runs)
                    )
                    break
                before = None if shown is None else shown.overlay(service)
                views_file = ""
                try:
                    report_i = self._pick(service, look)
                except Exception as exc:  # noqa: BLE001 (the campaign stops, the cell still comes down)
                    stopped_by = f"{type(exc).__name__}: {exc}"
                else:
                    last = report_i
                    views_file = self._keep_views(service, report_i, index)
                    # A fault of the cell stops the campaign as its raise did. `pick()` reports it on
                    # the report instead of raising it, so the campaign reads it there and says it in
                    # the same words.
                    fault = getattr(report_i, "fault", None)
                    stopped_by = "" if fault is None else f"{type(fault).__name__}: {fault}"
                    # So does a controller that cannot move. The pick ends through its own path on it,
                    # CANCELLED with no fault, and the campaign went on: after a protective stop mid-pick
                    # the next run perceived and pulsed the jaws of an arm a person had to walk up to
                    # (owner-cell audit, reproduced with fakes, 2026-09-23). `is True`, so a double that
                    # answers every attribute is not read as a stop.
                    if not stopped_by and getattr(report_i, "controller_stopped", False) is True:
                        stopped_by = (
                            "the controller cannot move (a protective or emergency stop, or a power-off), "
                            "so the campaign stops; a person clears the stop where the arm is visible: "
                            + str(report_i.failure_summary())
                        )
                    # And so does a hand that needs a person: a gripper that raised, or a toggle that would not
                    # start the pick on jaws it believes closed with nobody at a terminal to say otherwise, or a hand
                    # nobody can vouch for mid-pick (a toggle's count, read by a push or before next_target drives
                    # the looks again; a gripper that measures its width, found not connected or unreadable by a
                    # push). The next run would perceive again for a hand that still cannot start (owner's decisions,
                    # 2026-09-24 and 2026-09-30). A string, so a double that answers every attribute is not read as one.
                    hand = getattr(report_i, "gripper_fault", "")
                    if not stopped_by and isinstance(hand, str) and hand:
                        stopped_by = f"the gripper needs a person, so the campaign stops: {hand}"
                    # And so does a recovery that stopped where the arm stands: a push of a failed part that stopped
                    # once something may have moved, or whose move back to the look failed. The next pick would drive
                    # the arm back to its look, the escape the owner ruled out (2026-09-29). `is True`, as above.
                    if not stopped_by and getattr(report_i, "needs_person", False) is True:
                        stopped_by = (
                            "a recovery stopped where the arm stands, so the campaign stops; nothing more is "
                            "commanded and a person decides what happens next: " + str(report_i.failure_summary())
                        )
                if stopped_by:
                    attempts.append(
                        PickAttempt(index=index, outcome=PickOutcome.RAISED, detail=stopped_by,
                                    views_file=views_file)
                    )
                    self._announce(attempts[-1])
                    if shown is not None:
                        shown.attempt(service, attempts[-1], before)
                    attempts.extend(
                        PickAttempt(index=i, outcome=PickOutcome.CANCELLED)
                        for i in range(index + 1, self.runs)
                    )
                    break
                raw = getattr(report_i, "outcome", None)
                reported = str(getattr(raw, "value", raw))
                ok = reported == _SUCCEEDED
                if ok and self.rule.confirm is not None:
                    # The second opinion. The sim gate refuses to take `SUCCEEDED` as evidence and
                    # measures the lift independently; the real cell path supplies none. A
                    # `NullGripper` cell reports SUCCEEDED on every run.
                    ok = bool(self.rule.confirm(report_i))
                # Whether the gripper measured the hold, as the service says it: only a bool counts,
                # so a service that does not say leaves it None.
                hold = getattr(report_i, "hold_measured", None)
                # Whenever the service says it lifted a part, whatever a second opinion made of it: the
                # jaws closed on something, and the next pick must not start with it in the hand.
                put_back, not_back = self._put_back(service, report_i) if reported == _SUCCEEDED else (None, "")
                attempts.append(
                    PickAttempt(
                        index=index,
                        outcome=PickOutcome.SUCCEEDED if ok else PickOutcome.FAILED,
                        reported=reported,
                        detail="" if ok else str(report_i.failure_summary()),
                        hold_measured=hold if ok and isinstance(hold, bool) else None,
                        **_what_the_attempt_saw(report_i),
                        put_back=put_back,
                        views_file=views_file,
                    )
                )
                self._announce(attempts[-1])
                if shown is not None:
                    shown.attempt(service, attempts[-1], before)
                if not_back:
                    error = (f"the part of run {index} was not put back ({not_back}), so no further pick starts "
                             "with it in the hand")
                    attempts.extend(
                        PickAttempt(index=i, outcome=PickOutcome.CANCELLED)
                        for i in range(index + 1, self.runs)
                    )
                    break
        finally:
            # Put back in a `finally`. A per-campaign setting that outlives its campaign is
            # indistinguishable from a configured one: on a shared service every later run would
            # keep hunting the object this one was told to find.
            if chosen(self.target_label):
                service.set_target_label(None)
            if chosen(self.prompt):
                # After the label, so a campaign that set both ends on the filter the prompt replaced.
                service.set_prompt(previous_prompt)
            if self.recording.enabled:
                service.enable_record_logging(None)
            if shown is not None:
                shown.done(service)

        return PickRunReport(
            requested=self.runs, attempts=tuple(attempts), rule=self.rule,
            recording=self.recording, teardown=teardown, last=last, error=error,
        )


class _OnTheView:
    """What a campaign shows on its view: the cell's cameras, and each attempt's grasp overlay on the primary's window.

    Built at the start of the loop: every camera the service holds (the primary first, as `Cell.cameras` names them)
    is handed to the view and the view started, and the service's grasp overlay is switched on. After each attempt the
    overlay it rendered, if it rendered a new one, is pinned on the primary camera's window, the camera every grasp
    is synthesised in, with the attempt's outcome as its caption, and the outcome is said on every window. At the end
    the overlay switch is put back as the campaign found it. Every call on the view is guarded: it is display only,
    and one that raises changes nothing the campaign does.
    """

    __slots__ = ("_primary", "_rendering", "_view")

    def __init__(self, view: Any, service: Any) -> None:
        from src.robot.execution.lifecycle import service_cameras  # noqa: PLC0415

        self._view = view
        cameras = service_cameras(service)
        self._primary: str | None = str(cameras[0].rig_id) if cameras else None
        was = getattr(service, "debug_image_rendering_enabled", None)
        self._rendering: bool | None = was if isinstance(was, bool) else None
        if cameras:
            _quietly(view, "watch", *cameras)
        _quietly(view, "start")
        _quietly(service, "enable_debug_image_rendering", True)

    @staticmethod
    def overlay(service: Any) -> Any:
        """The grasp overlay the service holds now, to tell the one an attempt renders from it."""
        return getattr(service, "last_debug_image_png", None)

    def attempt(self, service: Any, attempt: PickAttempt, before: Any) -> None:
        """Pin the attempt's new overlay on the primary camera's window, and say the attempt's outcome on every one."""
        caption = f"run {attempt.index}: {attempt.reported or attempt.outcome.value}" + (
            f", {attempt.detail}" if attempt.detail else "")
        overlay = self.overlay(service)
        # A render is a new object. One the attempt did not replace (it never reached the grasp calculator) is the
        # attempt's before, and pinned under this attempt's outcome it would show a grasp this attempt never chose.
        if isinstance(overlay, (bytes, bytearray)) and overlay and overlay is not before:
            _quietly(self._view, "show_image", self._primary, overlay, caption, ok=attempt.passed)
        _quietly(self._view, "note", caption)

    def done(self, service: Any) -> None:
        """Put the service's overlay switch back as the campaign found it, where the service said how that was."""
        if self._rendering is not None:
            _quietly(service, "enable_debug_image_rendering", self._rendering)


def _quietly(target: Any, method: str, *args: Any, **keywords: Any) -> None:
    """``target.<method>(...)``, for display only: a target that lacks it, or raises, changes nothing; it is logged."""
    call = getattr(target, method, None)
    if not callable(call):
        return
    try:
        call(*args, **keywords)
    except Exception as exc:  # noqa: BLE001 (a window is never a reason to stop a campaign)
        logger.debug("pick run: %s.%s raised %s: %s", type(target).__name__, method, type(exc).__name__, exc)


def keep_pick_views(service: Any, report: Any, *, name: str, writer: Any = None) -> str:
    """Keep the looks of the pick that just ran on ``service`` for training (`src.robot.execution.record_views`); where
    they went, or `""`.

    Only a pick that says it looked (`report.looks`), from what the service's pick loop kept of those looks, so a pick
    that ended before its first look, or a fixed camera's, writes nothing and never another pick's looks. The file is
    named after the pick's record (`attempt_id`), else `name`. One that cannot be written is said, and the caller goes
    on: the views are for training, and the pick they record is over. What a campaign (`PickRun(record_views=True)`)
    and a task (`TaskOptions(record_views=True)`) keep alike.

    With ``writer`` (the process's background writer, `record_logging.background_writer`, a task's on a cell that writes
    in the background) the looks are taken now, before the next pick lets them go, and written by it once the pick has
    gone on: the file returned is where they will be, under the name the writer gives it, and one that cannot be written
    is said in the log while the task goes on.
    """
    looks = getattr(report, "looks", ())
    looked = getattr(service, "looked_around", None)
    views = getattr(looked, "views", None)
    if not (isinstance(looks, tuple) and looks) or not (isinstance(views, tuple) and views):
        return ""
    from src.robot.execution.record_views import record_views  # noqa: PLC0415

    telemetry = getattr(report, "telemetry", None)
    recorded = str(telemetry.get("attempt_id") or "") if isinstance(telemetry, Mapping) else ""
    if writer is not None:
        judged = getattr(looked, "judged", None)
        ranked = getattr(getattr(service, "ranked_looked", None), "judged", None)
        where = _views_file(recorded or name)
        writer.submit(f"the looks of {recorded or name}",
                      lambda: _write_views_to(where, views, judged, ranked, name=recorded or name))
        return str(where)
    try:
        written = record_views(views, target_cloud_base_mm=getattr(getattr(looked, "judged", None),
                                                                    "target_cloud_base_mm", None),
                               name=recorded or name, targets=_targets_of(views, getattr(looked, "judged", None)))
    except Exception as exc:  # noqa: BLE001 (a lost training file never stops a campaign or a task)
        logger.warning("pick run: the looks of %s were not kept: %s: %s", recorded or name, type(exc).__name__, exc)
        return ""
    if written is not None:
        try:
            _keep_debug_images(views, getattr(looked, "judged", None), Path(written),
                               ranked=getattr(getattr(service, "ranked_looked", None), "judged", None))
        except Exception as exc:  # noqa: BLE001 (a debug picture never stops a campaign or a task)
            logger.warning("pick run: the debug pictures of %s were not drawn: %s: %s", recorded or name,
                           type(exc).__name__, exc)
    return "" if written is None else str(written)


def _views_file(name: str) -> Path:
    """Where the looks of the pick ``name`` will be kept: the file name `record_views` gives, decided now."""
    import time  # noqa: PLC0415
    import uuid  # noqa: PLC0415

    from src.robot.execution.record_views import RECORD_VIEWS_DIR, _UNSAFE  # noqa: PLC0415

    stem = _UNSAFE.sub("_", name).strip("._") or "pick"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return Path(RECORD_VIEWS_DIR) / f"{stem}-{stamp}-{uuid.uuid4().hex[:6]}.npz"


def _write_views_to(where: Path, views: Any, judged: Any, ranked: Any, *, name: str) -> None:
    """The background half of :func:`keep_pick_views`: the looks taken at the pick written to ``where``, and their debug
    pictures beside it. A file that cannot be written raises, which the writer says; a picture that cannot be drawn is
    said here."""
    from src.robot.execution.record_views import record_views  # noqa: PLC0415

    written = record_views(views, target_cloud_base_mm=getattr(judged, "target_cloud_base_mm", None), name=name,
                           targets=_targets_of(views, judged))
    if written is None:
        return
    Path(written).replace(where)
    try:
        _keep_debug_images(views, judged, where, ranked=ranked)
    except Exception as exc:  # noqa: BLE001 (a debug picture never stops a campaign or a task)
        logger.warning("pick run: the debug pictures of %s were not drawn: %s: %s", name, type(exc).__name__, exc)


def _targets_of(views: Any, judged: Any) -> "tuple[int | None, ...]":
    """Per look of ``views``, which of its segmentations is the part the pick went for, as the look the pick judged its
    grasp on says it: that look's own ``target_index``, and for every other look the segmentation behind the blob its
    view of the part was associated with (``members``, read as the pick loop reads it). ``None`` where a look did not
    see the part or nothing was judged. What a views file of a camera recording for research keeps as each look's
    target (``record_views``'s ``targets``, the owner, 2026-10-09)."""
    def index(value: Any) -> int | None:
        """``value`` as an index, None where it is no integer (a bool included)."""
        return int(value) if isinstance(value, (int, np.integer)) and not isinstance(value, bool) else None

    members = getattr(judged, "members", None)
    members = members if isinstance(members, Mapping) else {}
    targets: list[int | None] = []
    for view in views:
        if judged is not None and view is getattr(judged, "look", None):
            targets.append(index(getattr(judged, "target_index", None)))
            continue
        blob = index(members.get(getattr(view, "name", None)))
        blobs = getattr(view, "blob_objects", ())
        if blob is not None and isinstance(blobs, tuple) and 0 <= blob < len(blobs):
            targets.append(index(blobs[blob]))
        else:
            targets.append(None)
    return tuple(targets)


#: How many of the look's ranked grasps a debug picture draws, the chosen one first and thicker.
DEBUG_GRASPS_DRAWN = 5


def _keep_debug_images(views: Any, judged: Any, written: Path, *, ranked: Any = None) -> list[Path]:
    """One PNG per look beside ``written`` (``<stem>_look<i>.png``): every detection's mask, box, label and score, and
    the ranked grasps of the judged look as grasp rectangles, the chosen one thicker (the owner, 2026-10-05). A look
    with no colour image is skipped.

    Where the judged look ranked no grasp, as after a blocker was set aside or the attempt looked again, the grasps are
    those of ``ranked``, the pick's last look that ranked any (``ranked_looked``): what the pick tried and why it failed
    is still in the picture."""
    from src.robot.grasping.visualization.scene_debug import (  # noqa: PLC0415
        draw_scene_debug,
        grasp_rectangle,
        save_scene_debug,
    )

    result = getattr(judged, "result", None)
    candidates = list(getattr(result, "candidates", ()) or ())[:DEBUG_GRASPS_DRAWN]
    if not candidates and ranked is not None:
        candidates = list(getattr(getattr(ranked, "result", None), "candidates", ()) or ())[:DEBUG_GRASPS_DRAWN]
    out: list[Path] = []
    for index, view in enumerate(views):
        frame = getattr(view, "frame", view)
        rgb = getattr(frame, "rgb", None)
        if rgb is None:
            continue
        camera_to_base = getattr(view, "camera_to_base", None)
        base_to_camera = None
        if camera_to_base is not None and np.all(np.isfinite(np.asarray(camera_to_base, dtype=np.float64))):
            base_to_camera = np.linalg.inv(np.asarray(camera_to_base, dtype=np.float64))
        rects = []
        for grasp in candidates:
            in_base = str(getattr(getattr(grasp, "frame", None), "value", getattr(grasp, "frame", ""))).lower() == "base"
            if in_base and base_to_camera is None:
                continue
            rect = grasp_rectangle(grasp.position, grasp.axis, grasp.approach, float(grasp.grip_width_mm),
                                   np.asarray(frame.intrinsics), base_to_camera=base_to_camera if in_base else None)
            if rect is not None:
                rects.append(rect)
        title = f"{getattr(view, 'name', '')} {getattr(view, 'label', '')}".strip()
        image = draw_scene_debug(rgb, tuple(getattr(frame, "segmentations", ()) or ()), grasps=rects, chosen=0,
                                 title=title)
        out.append(save_scene_debug(written.with_name(f"{written.stem}_look{index}.png"), image))
    return out


def configured_looks_of(service: Any) -> "tuple[LookPose, ...]":
    """The looks ``service``'s cell profile configures (``configured_looks``), or none.

    Only a tuple of looks counts, each a :class:`~src.robot.core.JointPositions` or ``"home"``, as the service reads
    them from ``robot.look_joint_positions_deg``: a double that answers every attribute, a list, or a raw row of
    numbers is read as none, so it never becomes a place the arm is sent. The rule a campaign and the console share.
    """
    from src.robot.core.joint_positions import JointPositions  # noqa: PLC0415
    from src.robot.execution.looks import HOME  # noqa: PLC0415

    configured = getattr(service, "configured_looks", ())
    if isinstance(configured, tuple) and configured and all(
            isinstance(look, JointPositions) or (isinstance(look, str) and look == HOME) for look in configured):
        return configured
    return ()


def _checked_push_mm(push_mm: "Maybe[float]") -> "Maybe[float]":
    """A campaign's push distance, refused at the factory above the hard cap of 50 mm or under 10 mm, before a cell is
    built (``push_planner.resolve_push_distance`` against the cap alone; the cell's own ceiling is the campaign's)."""
    if not chosen(push_mm):
        return push_mm
    from src.robot.grasping.recovery.push_planner import (  # noqa: PLC0415
        DEFAULT_PUSH_DISTANCE_MM,
        PUSH_DISTANCE_CAP_MM,
        PushRefusal,
        resolve_push_distance,
    )

    try:
        requested = float(push_mm)
    except (TypeError, ValueError):
        raise ValueError(f"PickRun(push_mm=...) is a number of mm, not {push_mm!r}") from None
    resolved = resolve_push_distance(requested, default_mm=DEFAULT_PUSH_DISTANCE_MM, ceiling_mm=PUSH_DISTANCE_CAP_MM)
    if isinstance(resolved, PushRefusal):
        raise ValueError(f"PickRun(push_mm={push_mm!r}) refused: {resolved.sentence}")
    return requested


def _checked_looks(look: "Maybe[Look]") -> "tuple[LookPose, ...]":
    """A campaign's looks, in order, refused at the factory when they name nothing, before a cell is built."""
    from src.robot.execution.looks import looks_of  # noqa: PLC0415

    return looks_of(look) if chosen(look) else ()


def _screen_looks(service: Any, look: "Look | None") -> None:
    """Screen every look of a starting campaign with the exact guard and the planner, and say each verdict; nothing moves.

    The owner, 2026-09-30 (``planning.band``): a look the planner's padded spheres alone refuse runs on straight lines,
    and a planned move out of it takes a short escape leg; a look the exact guard refuses is never reached. Said at
    the start of the campaign, in the log, one line a look (an ERROR where no move goes there), with a pose nearby both
    clear. Only an arm whose class screens (``URRobotArm.screen_configuration``) is asked; the campaign runs the same
    either way, and a screen that raises is said and passed over.
    """
    arm = getattr(getattr(getattr(service, "runtime", None), "orchestrator", None), "arm", None)
    if arm is None or look is None or not callable(getattr(type(arm), "screen_configuration", None)):
        return
    from src.robot.core.joint_positions import JointPositions  # noqa: PLC0415
    from src.robot.execution.looks import look_label, looks_of  # noqa: PLC0415
    from src.robot.safety.planning.band import PoseScreen  # noqa: PLC0415

    ask_planner = True
    for pose in looks_of(look):
        joints = pose if isinstance(pose, JointPositions) else JointPositions(tuple(arm.home_joint_positions))
        label = f"look {look_label(pose)}"
        try:
            screen = arm.screen_configuration(joints, ask_planner=ask_planner)
        except Exception as exc:  # noqa: BLE001 (a screen never stops a campaign; the move judges the look again)
            logger.warning("pick run: %s was not screened: %s: %s", label, type(exc).__name__, exc)
            continue
        if not isinstance(screen, PoseScreen):
            continue
        (logger.error if screen.is_error else logger.info)("pick run: %s", screen.line(label))
        ask_planner = ask_planner and not screen.planner_unavailable


def _what_the_attempt_saw(report: Any) -> dict[str, Any]:
    """Where the service says the object was, where the tool closed, where it looked from, which cameras and looks its
    grasp was planned from, which contact faces of that grasp were seen, the hand-eye check and the view the looks
    generated, as `PickAttempt` keeps them.

    Each only where the service says it in the type it promises, so a service that does not say, or a double that
    answers every attribute, leaves the attempt as it was. Fused views are two or more names or none: one camera alone
    is not a fusion.
    """
    from src.geometry import Pose  # noqa: PLC0415

    centre = getattr(report, "object_centre_mm", None)
    grasp = getattr(report, "grasp_pose", None)
    looks = getattr(report, "looks", ())
    fused = getattr(report, "fused_views", ())
    count = getattr(report, "fused_objects", 0)
    looks_fused = getattr(report, "looks_fused", ())
    faces = getattr(report, "jaw_faces_seen", None)
    gap = getattr(report, "hand_eye_gap_mm", None)
    turn = getattr(report, "generated_view_deg", None)
    return {
        "object_mm": centre if isinstance(centre, tuple) and len(centre) == 3 else None,
        "grasp_pose": grasp if isinstance(grasp, Pose) else None,
        "looks": looks if isinstance(looks, tuple) and all(isinstance(v, str) for v in looks) else (),
        "fused_views": (fused if isinstance(fused, tuple) and len(fused) > 1 and all(isinstance(v, str) for v in fused)
                        else ()),
        "fused_objects": count if isinstance(count, int) and not isinstance(count, bool) and count > 0 else 0,
        "looks_fused": (looks_fused if isinstance(looks_fused, tuple) and all(isinstance(v, str) for v in looks_fused)
                        else ()),
        "jaw_faces_seen": (faces if isinstance(faces, tuple) and len(faces) == 2
                           and all(isinstance(v, bool) for v in faces) else None),
        "hand_eye_gap_mm": float(gap) if isinstance(gap, (int, float)) and not isinstance(gap, bool) else None,
        "generated_view_deg": (float(turn) if isinstance(turn, (int, float)) and not isinstance(turn, bool)
                               else None),
    }
