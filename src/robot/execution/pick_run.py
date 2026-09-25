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
   toggle that would not start on jaws it believes closed with nobody at a terminal to say otherwise).

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
a look the program declares, or home when it declares none (`src.robot.execution.looks`); a fixed
camera does not move. And ``put_back``: a part lifted is put back where it was grasped, so one part
serves a whole campaign; a part that could not be put back stops the campaign, because the next pick
would start with it in the hand.

And one more, the same day: ``view``, a `LiveView` (``src/camera/live_view.py``), shows every camera the
cell's build opened, one window each, while the campaign runs. The campaign renders the grasp overlay on
every pick (the service's debug image, put back as it was afterwards), pins each attempt's overlay on the
primary camera's window with the attempt's outcome, and says that outcome on every window. It is display
only: a window closed, or a view that raises, changes nothing the campaign does, and the view reads the
cameras through their display path, which changes nothing the picks measure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from src.contracts import UNSET, Maybe, chosen

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
]

logger = logging.getLogger(__name__)

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

    A rule object rather than a bare number. A bare `int` threshold would let the sim's 80 % and
    this runner's unanimity look like the same kind of thing, and they are not: the sim gate also
    refuses to take the service's own word for a success. `confirm` is where that refusal fits, and
    it has no default that accepts.
    """

    #: Fraction of attempts that must succeed, 1.0 for unanimity.
    fraction: float = 1.0
    #: A second opinion per attempt, taking the service's report and answering "did this really
    #: happen". `None` means the service's own outcome is the only evidence; the sim harness
    #: supplies one, because a cell with a `NullGripper` reports success on every run.
    confirm: "Callable[[Any], bool] | None" = None

    def accepts(self, attempts: "Sequence[PickAttempt]") -> bool:
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
        confirmed = ", independently confirmed" if self.confirm is not None else ""
        if self.fraction >= 1.0:
            return f"every attempt must succeed{confirmed}"
        return f"at least {self.fraction:.0%} of attempts must succeed{confirmed}"


@dataclass(frozen=True, slots=True)
class Recording:
    """Where a campaign appends its `GraspAttemptRecord` lines, or that it appends none.

    A noun rather than an optional path, so that "this run produced no corpus" is something the
    report states rather than a `None` a reader has to notice.
    """

    path: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return bool(self.path)

    @classmethod
    def off(cls) -> "Recording":
        """Log nothing. The shipped `robot.yaml` has `record_log_path: null`, so this is the
        default state of a config-driven cell."""
        return cls()

    @classmethod
    def to_file(cls, path: str, *, provenance: "Mapping[str, Any] | None" = None) -> "Recording":
        return cls(path=str(path), provenance=dict(provenance or {}))

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        return f"recording to {self.path}" if self.enabled else "recording nothing"


@dataclass(frozen=True, slots=True)
class PickAttempt:
    """One attempt, and what the campaign made of it."""

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

    @property
    def passed(self) -> bool:
        return self.outcome is PickOutcome.SUCCEEDED

    @property
    def unmeasured(self) -> bool:
        """A success whose hold the gripper did not measure."""
        return self.passed and self.hold_measured is False

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        lines = [f"  run {self.index}: {self.reported or self.outcome.value}" + (
            f"  {self.detail}" if self.detail else ""
        ) + ("  hold not measured" if self.unmeasured else "")]
        if self.looks:
            lines.append(f"    looked from {'; '.join(self.looks)}")
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
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
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
        }


@dataclass(frozen=True, slots=True)
class PickRunReport:
    """What a campaign did, whether it passed, and how the cell came down."""

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
        return sum(1 for a in self.attempts if a.outcome is PickOutcome.CANCELLED)

    @property
    def raised(self) -> bool:
        return any(a.outcome is PickOutcome.RAISED for a in self.attempts)

    @property
    def passed(self) -> bool:
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
        """The whole campaign. ASCII, no trailing newline, no arguments."""
        lines = [a.render() for a in self.attempts if a.outcome is not PickOutcome.CANCELLED]
        if self.error:
            lines.append(f"  REFUSED: {self.error}")
        lines.append("")
        lines.append(self.summary())
        return "\n".join(lines)

    def summary(self) -> str:
        """The verdict block alone: the count, the rule it was judged by, and what was recorded.

        A fragment of `render()`, which returns the whole campaign. The command-line runner prints
        its own staged banners between the picks and the teardown and needs only this tail.
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
        """Plain data, `json.dumps`-safe with no custom encoder."""
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

        from src.config import load_robot_config
        from src.robot.execution.cell import Cell
        from src.robot.execution.pick_run import PickRun, Recording

        cell = Cell.from_robot_config(load_robot_config())
        report = PickRun.from_cell(cell, runs=10, recording=Recording.off()).execute()
        print(report.render())
        raise SystemExit(report.exit_code)

    The two factories are `from_cell` and `from_service`, following the `from_<python-input>`
    convention. A campaign takes the cell and owns the connect, because owning the connect is half
    of what it knows.
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
    #: Where each pick looks from before it perceives, tried in order until one finds something
    #: (`src.robot.execution.looks`). Unset: a wrist camera looks from the arm's home, a fixed camera
    #: from where it is mounted, with no motion. Nothing is said to the hand before a look.
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
    ) -> "PickRun":
        """A cell this campaign will build, connect, drive and take down.

        The connect happens once, outside the loop. Connecting is motion: a Robotiq activation
        sweeps the full finger travel and a vacuum cup asserts its ejector immediately. Ten
        campaigns of one are not one campaign of ten. A look list that names nothing raises here,
        before any cell is built. ``view`` (a `LiveView`) shows every camera the build opened and
        each attempt's grasp overlay; see the field.
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
    ) -> "PickRun":
        """An already-connected service. The caller owns the connect and the teardown.

        `teardown` stays `None` on the report: this factory did not bring the cell up and must not
        claim to know how it came down. ``view`` shows the service's cameras and each attempt's
        grasp overlay, as `from_cell`'s does.
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
        )

    # --- the verb ------------------------------------------------------------------------------

    def execute(self) -> PickRunReport:
        """Run the campaign. Builds and connects when it owns the cell; always takes it down again."""
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

    def _looks_for(self, service: Any) -> "Look | None":
        """What every pick of this campaign looks from: the program's looks, else home on a wrist camera, else none.

        `is True`, so a double that answers every attribute is not read as a wrist camera; a service that
        does not say perceives from where the arm stands, as it always did.
        """
        from src.robot.execution.looks import HOME  # noqa: PLC0415

        if self.look:
            return self.look
        return HOME if getattr(service, "perceives_from_the_wrist", False) is True else None

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
        # The prompt first: it is the one setting that can refuse, and a refusal here leaves the other
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
                try:
                    report_i = service.pick() if look is None else service.pick(look=look)
                except Exception as exc:  # noqa: BLE001 (the campaign stops, the cell still comes down)
                    stopped_by = f"{type(exc).__name__}: {exc}"
                else:
                    last = report_i
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
                    # start the pick on jaws it believes closed with nobody at a terminal to say otherwise. The
                    # next run would perceive again for a hand that still cannot start (owner's decision,
                    # 2026-09-24). A string, so a double that answers every attribute is not read as one.
                    hand = getattr(report_i, "gripper_fault", "")
                    if not stopped_by and isinstance(hand, str) and hand:
                        stopped_by = f"the gripper needs a person, so the campaign stops: {hand}"
                if stopped_by:
                    attempts.append(
                        PickAttempt(index=index, outcome=PickOutcome.RAISED, detail=stopped_by)
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


def _checked_looks(look: "Maybe[Look]") -> "tuple[LookPose, ...]":
    """A campaign's looks, in order, refused at the factory when they name nothing, before a cell is built."""
    from src.robot.execution.looks import looks_of  # noqa: PLC0415

    return looks_of(look) if chosen(look) else ()


def _what_the_attempt_saw(report: Any) -> dict[str, Any]:
    """Where the service says the object was, where the tool closed, where it looked from and which cameras its grasp
    was planned from, as `PickAttempt` keeps them.

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
    return {
        "object_mm": centre if isinstance(centre, tuple) and len(centre) == 3 else None,
        "grasp_pose": grasp if isinstance(grasp, Pose) else None,
        "looks": looks if isinstance(looks, tuple) and all(isinstance(v, str) for v in looks) else (),
        "fused_views": (fused if isinstance(fused, tuple) and len(fused) > 1 and all(isinstance(v, str) for v in fused)
                        else ()),
        "fused_objects": count if isinstance(count, int) and not isinstance(count, bool) and count > 0 else 0,
    }
