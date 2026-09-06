"""A campaign of N picks: one connect around them, one verdict over them, one report about them.

The loop itself is six lines of ``for _ in range(n): service.pick()``. The knowledge is in the four
things around it.

1. The verdict rule. `PassRule` defaults to unanimity: every pick must succeed. The sim gate
   configures 80 % instead (`sim_schema.py` `pass_fraction: 0.8`) and does not take the service's
   own ``SUCCEEDED`` as sufficient evidence: it additionally requires a lift measured from the
   physics to clear `sim_schema.py` `lift_threshold_mm`, which defaults to 50.0. A cell with a
   `NullGripper` reports SUCCEEDED on every run, so the rule is an explicit argument here rather
   than a constant.

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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from src.contracts import UNSET, Maybe, chosen

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.execution.cell import Cell
    from src.robot.execution.lifecycle import ConnectStage, TeardownReport

__all__ = [
    "PassRule",
    "PickAttempt",
    "PickOutcome",
    "PickRun",
    "PickRunReport",
    "Recording",
]

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
    #: An exception escaped `pick()`. The campaign stops; the cell still comes down.
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

    @property
    def passed(self) -> bool:
        return self.outcome is PickOutcome.SUCCEEDED

    def render(self) -> str:
        return f"  run {self.index}: {self.reported or self.outcome.value}" + (
            f"  {self.detail}" if self.detail else ""
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "outcome": self.outcome.value,
            "reported": self.reported,
            "detail": self.detail,
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
        """0 pass, 1 refused before picking, 2 picked and did not pass, 3 an exception escaped.

        The same four codes the command-line runner returns, derived from the report rather than
        branched at four `return` statements.
        """
        if self.error:
            return 1
        if self.raised:
            return 3
        return 0 if self.passed else 2

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
            "cancelled": self.cancelled,
            "passed": self.passed,
            "exit_code": self.exit_code,
            "rule": {"fraction": self.rule.fraction, "confirmed": self.rule.confirm is not None},
            "recording": {"enabled": self.recording.enabled, "path": self.recording.path},
            "clean_teardown": self.clean_teardown,
            "error": self.error,
            "attempts": [a.to_dict() for a in self.attempts],
        }


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
    #: What the vision front end looks for, for this campaign only. Set and cleared around the run.
    target_label: "Maybe[str | None]" = UNSET
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
        should_cancel: "Callable[[], bool] | None" = None,
        announce: "Callable[[ConnectStage], None] | None" = None,
        on_attempt: "Callable[[PickAttempt], None] | None" = None,
    ) -> "PickRun":
        """A cell this campaign will build, connect, drive and take down.

        The connect happens once, outside the loop. Connecting is motion: a Robotiq activation
        sweeps the full finger travel and a vacuum cup asserts its ejector immediately. Ten
        campaigns of one are not one campaign of ten.
        """
        return cls(
            cell=cell,
            runs=runs,
            recording=recording,
            rule=rule if chosen(rule) else PassRule(),
            target_label=target_label,
            should_cancel=should_cancel,
            announce=announce,
            on_attempt=on_attempt,
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
        should_cancel: "Callable[[], bool] | None" = None,
        on_attempt: "Callable[[PickAttempt], None] | None" = None,
    ) -> "PickRun":
        """An already-connected service. The caller owns the connect and the teardown.

        `teardown` stays `None` on the report: this factory did not bring the cell up and must not
        claim to know how it came down.
        """
        return cls(
            service=service,
            runs=runs,
            recording=recording,
            rule=rule if chosen(rule) else PassRule(),
            target_label=target_label,
            should_cancel=should_cancel,
            on_attempt=on_attempt,
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

    def _drive(self, service: Any, *, teardown: "TeardownReport | None") -> PickRunReport:
        """The loop, and the two per-campaign settings that must be put back afterwards."""
        if self.recording.enabled:
            service.enable_record_logging(
                self.recording.path, provenance=dict(self.recording.provenance) or None
            )
        if chosen(self.target_label):
            service.set_target_label(self.target_label)

        attempts: list[PickAttempt] = []
        last: Any = None
        try:
            for index in range(self.runs):
                if self.should_cancel is not None and self.should_cancel():
                    attempts.extend(
                        PickAttempt(index=i, outcome=PickOutcome.CANCELLED)
                        for i in range(index, self.runs)
                    )
                    break
                try:
                    report_i = service.pick()
                except Exception as exc:  # noqa: BLE001 (the campaign stops, the cell still comes down)
                    attempts.append(
                        PickAttempt(
                            index=index, outcome=PickOutcome.RAISED,
                            detail=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    self._announce(attempts[-1])
                    attempts.extend(
                        PickAttempt(index=i, outcome=PickOutcome.CANCELLED)
                        for i in range(index + 1, self.runs)
                    )
                    break
                last = report_i
                raw = getattr(report_i, "outcome", None)
                reported = str(getattr(raw, "value", raw))
                ok = reported == _SUCCEEDED
                if ok and self.rule.confirm is not None:
                    # The second opinion. The sim gate refuses to take `SUCCEEDED` as evidence and
                    # measures the lift independently; the real cell path supplies none. A
                    # `NullGripper` cell reports SUCCEEDED on every run.
                    ok = bool(self.rule.confirm(report_i))
                attempts.append(
                    PickAttempt(
                        index=index,
                        outcome=PickOutcome.SUCCEEDED if ok else PickOutcome.FAILED,
                        reported=reported,
                        detail="" if ok else str(report_i.failure_summary()),
                    )
                )
                self._announce(attempts[-1])
        finally:
            # Put back in a `finally`. A per-campaign setting that outlives its campaign is
            # indistinguishable from a configured one: on a shared service every later run would
            # keep hunting the object this one was told to find.
            if chosen(self.target_label):
                service.set_target_label(None)
            if self.recording.enabled:
                service.enable_record_logging(None)

        return PickRunReport(
            requested=self.runs, attempts=tuple(attempts), rule=self.rule,
            recording=self.recording, teardown=teardown, last=last,
        )
