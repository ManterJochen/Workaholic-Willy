"""The spine every example shares, so the examples read as one product rather than as a directory.

An example is read far more often than it is run, and usually by somebody deciding whether this
stack can do their job. That makes the output as much of the deliverable as the code: a reader
should be able to follow what happened without knowing the repository.

Three things this module enforces, each because of a specific way an example goes wrong:

  1. One shape of output. A step announces itself, then reports a verdict. Examples that each
     invent their own print style read as a pile of prototypes; the same frame around every step
     makes them read as one tool.
  2. A refusal is a teaching moment. An example that dies with `ConnectionError` has wasted the
     reader's time. `not_ready()` names what was missing and the command that fixes it, and it
     returns a different exit code from a real failure: "your cell is not connected" and "the grasp
     failed" are different answers, and a script chaining examples has to tell them apart.
  3. Real hardware never moves by accident. Several of these examples drive an arm, and a file that
     commands a robot the moment somebody types `python run.py` is a hazard however good its
     docstring is: the reader who most needs the example is the least likely to have read it to the
     end first. The default is a rehearsal, and motion needs `--live`. That is the same decision
     `real_cell --rehearse` makes.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

# Run as a script, `python scripts/examples/cell/01_robot_setup.py` puts only the directory holding
# that file on sys.path, and `import src...` then fails. Every example needs the repository root, so
# it is added once here rather than once per example: importing this module is what every example
# does first anyway. The tests import an example as a module and an operator runs the file, and those
# are two different environments. This file sits at `<repo>/scripts/examples/`, so the root is two
# parents up; the repository is not pip installable, so there is no import path without this. An
# example in a topic folder reaches this module by putting its own parent on sys.path first, which
# is the one line of boilerplate the folders cost.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

__all__ = ["EXIT_FAILED", "EXIT_NOT_READY", "EXIT_OK", "Example", "Outcome",
           "hardware_parser", "not_ready"]

#: Exit codes, shared so one script can chain examples and branch on why the last one stopped.
EXIT_OK, EXIT_FAILED, EXIT_NOT_READY = 0, 1, 2


@dataclass(frozen=True, slots=True)
class Outcome:
    """One step's result. `ok=False` with `blocking=False` is a finding, not a stop."""

    step: str
    detail: str
    ok: bool = True
    blocking: bool = True


@dataclass
class Example:
    """A named run made of steps, with one output shape and one summary.

    Used as a context manager, so the summary is printed even when a step raises::

        with Example("pick", "one grasp, end to end", live=args.live) as run:
            with run.step("connect the arm") as report:
                arm.connect()
                report(f"{type(arm).__name__} on {config.ur.ip}")
    """

    name: str
    what: str
    #: False means rehearse: the example builds and checks, and commands nothing.
    live: bool = False
    #: Whether this example can touch a robot at all.
    #:
    #: False for the examples that do their real work on files with no cell attached. For those,
    #: announcing "rehearsal, nothing is commanded" would be a lie in the other direction: it
    #: implies something was held back that was not. An example must not claim to be less than it
    #: is, any more than it may claim to be more.
    hardware: bool = True
    #: `WILLY_PROFILE` to apply for the duration, then put back.
    #:
    #: Restored, not merely set. As a script, setting the variable and letting the process exit is
    #: harmless. But these examples are also called: by `06_full_pipeline`, and by the tests that
    #: keep them from decaying into prose. A permanent mutation of `os.environ` leaks into
    #: everything that runs afterwards, so a later reader of the variable loads a cell it never
    #: asked for and the symptom appears far from the cause. `Example` already brackets the whole
    #: body, so the restoration belongs here rather than in a `finally` block per example.
    profile: str | None = None
    steps: list[Outcome] = field(default_factory=list)
    _started: float = field(default_factory=time.perf_counter)
    _previous_profile: tuple[bool, str] = (False, "")

    def __enter__(self) -> "Example":
        if self.profile:
            import os

            had = "WILLY_PROFILE" in os.environ
            self._previous_profile = (had, os.environ.get("WILLY_PROFILE", ""))
            os.environ["WILLY_PROFILE"] = self.profile
        print(f"\n=== {self.name}: {self.what} ===")
        # Said at the top, not only in the summary. A reader who learns it was a rehearsal only
        # afterwards has already read everything above it as if it were real.
        if not self.hardware:
            print("    No robot involved. This example does its real work on files.\n")
        else:
            print("    REHEARSAL: nothing is commanded. Pass --live to drive the cell.\n"
                  if not self.live else "    LIVE: this moves the robot. Clear the cell.\n")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.profile:
            import os

            had, previous = self._previous_profile
            if had:
                os.environ["WILLY_PROFILE"] = previous
            else:
                os.environ.pop("WILLY_PROFILE", None)
        # Returns None, not False. Both mean "do not swallow", but a `bool` return type tells a type
        # checker this manager might swallow, and an example that quietly eats an exception is the
        # worst possible teaching tool.
        self.summary()

    @contextmanager
    def step(self, title: str) -> Iterator[Callable[[str], None]]:
        """Announce a step, then let the body report one line of detail.

        The body calls the yielded function with that line. It returns nothing on purpose: a step
        that reports no detail has told the reader nothing, and making that awkward is the point.
        """
        # The progress line only works on a terminal. Redirected to a file, which is what an
        # operator does with a long hardware run and what CI always does, the carriage return leaves
        # both the announcement and the verdict on one unreadable line. An example whose log cannot
        # be read afterwards has failed at the thing examples are for, so off a TTY only the verdict
        # is written. On a TTY the announcement is an open line: anything the body prints itself
        # lands on it, which is why a step whose body narrates its own stages does that narration
        # before the step rather than inside it.
        spinner = sys.stdout.isatty()
        rewind = "\r" if spinner else ""
        if spinner:
            print(f"  ..   {title} ...", end="", flush=True)
        captured: list[str] = []
        try:
            yield captured.append
        except Exception as error:                                   # noqa: BLE001  (examples report)
            print(f"{rewind}  FAIL {title:<46} {type(error).__name__}: {error}")
            self.steps.append(Outcome(title, f"{type(error).__name__}: {error}", ok=False))
            raise
        if not captured:
            # No detail is a failed step, not a tidy one. Two bodies reach here without reporting,
            # and a tick would hide both: one that forgot to call `report`, and one that returned
            # early out of the `with` block. An early return leaves the context manager normally, so
            # the step would otherwise be recorded as passed while the example was in fact refusing,
            # and a reader would see a tick next to a step that did not happen.
            print(f"{rewind}  FAIL {title:<46} did not complete (no detail reported)")
            self.steps.append(Outcome(title, "did not complete", ok=False))
            return
        print(f"{rewind}  OK   {title:<46} {captured[-1]}")
        self.steps.append(Outcome(title, captured[-1]))

    def note(self, text: str) -> None:
        """A line that is not a step: context, a caveat, a measured number."""
        print(f"    {text}")

    def finding(self, title: str, detail: str) -> None:
        """Something came out wrong, and the example continues anyway.

        Deliberately distinct from a failed step. "The calibration RMSE is 4.2 mm, which is
        marginal" is a result the reader needs, and treating it as a crash would hide the rest of
        the run.

        A finding never changes `exit_code`, which counts blocking steps only. An example that
        reports a finding and still owes the caller a non-zero status returns that status itself,
        usually the one the library report it is quoting already computed.
        """
        print(f"  WARN {title:<46} {detail}")
        self.steps.append(Outcome(title, detail, ok=False, blocking=False))

    def summary(self) -> None:
        bad = [step for step in self.steps if not step.ok]
        elapsed = time.perf_counter() - self._started
        print(f"\n  {len(self.steps) - len(bad)}/{len(self.steps)} step(s) OK in {elapsed:.1f} s")
        for step in bad:
            print(f"    {'FAILED ' if step.blocking else 'finding'} {step.step}: {step.detail}")
        if self.hardware and not self.live:
            print("    WARN REHEARSAL: the call path ran, the cell did not move. Nothing here is "
                  "evidence about your hardware.")

    @property
    def exit_code(self) -> int:
        """`EXIT_FAILED` when a blocking step failed, `EXIT_OK` otherwise.

        Findings are not blocking, so an example that reports one and returns this property returns
        zero. That is the trap `finding()` documents: quote the report's own exit code instead.
        """
        return EXIT_FAILED if any(s.blocking for s in self.steps if not s.ok) else EXIT_OK


def hardware_parser(description: str, *, epilog: str = "") -> argparse.ArgumentParser:
    """The argument parser every hardware example shares.

    `--live` is opt-in and stays that way. Without it the example builds everything, checks
    everything checkable without motion, and reports: the mode a reader wants the first time, and
    the only mode CI can run. Defaulting the other way would make `python run.py` a command that
    moves a robot, and the person most likely to type it blind has not read this far.
    """
    parser = argparse.ArgumentParser(
        description=description, epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--live", action="store_true",
        help="actually drive the cell. Without it the example rehearses: it builds and checks "
             "everything that needs no motion, and commands nothing.")
    parser.add_argument(
        "--profile", default=None,
        help="WILLY_PROFILE to load (e.g. `ursim`, `ur3e`, `sim`). Default: the environment's.")
    return parser


def not_ready(what: str, fix: str) -> int:
    """Report a missing precondition and return `EXIT_NOT_READY`.

    Never `EXIT_FAILED` for this. "Your cell is not connected" and "the grasp came out wrong" are
    different answers, and a script chaining examples has to tell them apart.
    """
    print(f"\n  STOP not ready: {what}\n     fix: {fix}", file=sys.stderr)
    return EXIT_NOT_READY
