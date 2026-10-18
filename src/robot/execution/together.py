"""Several arms at once: :func:`together` runs a program's steps on their own threads and waits for every one.

A cell of several arms (the owner, 2026-10-15) is programmed as code, not from the console: one step after another is
one line after another with each arm's own verbs (``cell.arms["werker"].robot.move(...)``), and steps that run at the
same time go into :func:`together` (the owner, 2026-10-16: "Python + together()"). Nothing here decides whether two
motions may run at once: the cell's reservations do (``execution.cell_motion``). A joint motion reserves the space it
sweeps and runs beside the other arm where the two stay apart; a line holds the whole cell; a motion that would meet
the other arm waits for it, and is refused once it has waited long enough.

Where one step fails, every arm halts (the owner, 2026-10-16: "Alle Arme halten an"): an exception, or a report whose
``ok`` is false (a motion refused, a hand that closed on nothing). Each arm latches (``SupportsHalt``), brakes what it
has in flight where it watches its sends, and moves again only once a person clears the latch. :func:`together` waits
for every step to end, the others included, and raises :class:`TogetherFailed` with what each step did.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

__all__ = ["StepEnd", "TogetherFailed", "halt_arms", "live_cell", "together"]

#: The connected cell a ``with cell.connected():`` block is in, outermost first: what :func:`together` halts.
_LIVE: "ContextVar[Any]" = ContextVar("willy_live_cell", default=None)


def halt_arms(arms: "Iterable[Any]", reason: str) -> None:
    """Halt each of ``arms`` that keeps a halt latch (``SupportsHalt``); an arm that keeps none is left as it is.

    Args:
        arms (Iterable[Any]): The arms.
        reason (str): Why, as each latch keeps it.
    """
    from src.robot.core.arm_capabilities import SupportsHalt  # noqa: PLC0415

    for arm in arms:
        if isinstance(arm, SupportsHalt):
            arm.halt(reason)


@contextmanager
def live_cell(session: Any) -> Iterator[None]:
    """Name ``session`` the connected cell :func:`together` halts, for the block, where no block outside named one.

    ``ConnectedArms`` and ``ConnectedCell`` enter it as they come up, so a program never calls it.

    Args:
        session (Any): The connected cell: what its ``halt_every_arm(reason)`` halts.

    Yields:
        None: The block runs with ``session`` named, unless an outer block named its own.
    """
    if _LIVE.get() is not None:
        yield
        return
    token = _LIVE.set(session)
    try:
        yield
    finally:
        _LIVE.reset(token)


@dataclass(frozen=True)
class StepEnd:
    """How one step of :func:`together` ended.

    Attributes:
        index (int): The step's place among the steps, from 0.
        result (Any): What the step returned; ``None`` where it raised.
        error (BaseException | None): What the step raised; ``None`` where it returned.
    """

    index: int
    result: Any = None
    error: BaseException | None = None

    @property
    def failed(self) -> bool:
        """Whether the step raised or returned a report whose ``ok`` is false."""
        return self.error is not None or getattr(self.result, "ok", True) is False


class TogetherFailed(RuntimeError):
    """A step of :func:`together` failed, and every arm of the cell was halted.

    Attributes:
        steps (tuple[StepEnd, ...]): How each step ended, in the order they were given.
        reason (str): The halt's reason, as each arm's latch keeps it.
    """

    def __init__(self, steps: Sequence[StepEnd], reason: str) -> None:
        self.steps = tuple(steps)
        self.reason = reason
        super().__init__(reason)

    @property
    def failed(self) -> tuple[StepEnd, ...]:
        """The steps that failed, in their order."""
        return tuple(step for step in self.steps if step.failed)


def _why(step: StepEnd) -> str:
    if step.error is not None:
        return f"step {step.index} raised {type(step.error).__name__}: {step.error}"
    render = getattr(step.result, "render", None)
    said = render() if callable(render) else str(step.result)
    return f"step {step.index} did not succeed: {' '.join(str(said).split())[:200]}"


def together(*steps: Callable[[], Any]) -> tuple[Any, ...]:
    """Run ``steps`` at the same time, each on its own thread, and wait for every one.

    Call it inside ``with cell.connected():``. The cell's reservations keep the arms apart: two joint motions run at
    once where the space each sweeps stays clear of the other arm, and one waits for the other where it does not. A step
    fails where it raises, or returns a report whose ``ok`` is false; then every arm of the cell halts at once, the
    other steps end as their arms refuse, and :class:`TogetherFailed` says what each did. A Ctrl+C while the steps run
    halts every arm as well, and is raised again.

    ```python
    from willy import Cell, together

    cell = Cell.from_cell("clip")
    cell.build()
    with cell.connected() as arms:
        halter, werker = arms["halter"].robot, arms["werker"].robot
        halter.move_joints(halten)
        together(
            lambda: halter.move_joints(nachfassen),
            lambda: werker.move_joints(anfahrt),
        )
    ```

    Args:
        *steps (Callable[[], Any]): The steps, each a call that takes no argument, such as a ``lambda``.

    Returns:
        tuple[Any, ...]: What each step returned, in the order given.

    Raises:
        RuntimeError: Called outside ``with cell.connected():``; nothing ran.
        TypeError: A step is not callable; nothing ran.
        TogetherFailed: A step failed; every arm of the cell was halted.
    """
    session = _LIVE.get()
    if session is None:
        raise RuntimeError("together() runs inside `with cell.connected():`, the cell whose arms it halts when a step "
                           "fails; nothing ran")
    for index, step in enumerate(steps):
        if not callable(step):
            raise TypeError(f"together(): step {index} is a {type(step).__name__}, not a call; nothing ran")
    ends: list[StepEnd | None] = [None] * len(steps)
    halted = threading.Event()
    reason_holder: list[str] = []

    def halt(reason: str) -> None:
        if not halted.is_set():
            halted.set()
            reason_holder.append(reason)
            session.halt_every_arm(reason)

    def run(index: int, step: Callable[[], Any]) -> None:
        try:
            end = StepEnd(index, result=step())
        except BaseException as exc:  # noqa: BLE001 (every way a step ends is reported, and halts the cell)
            end = StepEnd(index, error=exc)
        ends[index] = end
        if end.failed:
            halt(f"together(): {_why(end)}")

    threads = [threading.Thread(target=run, args=(index, step), name=f"together-{index}", daemon=True)
               for index, step in enumerate(steps)]
    for thread in threads:
        thread.start()
    try:
        for thread in threads:
            thread.join()
    except BaseException:
        halt("together(): interrupted while its steps ran")
        for thread in threads:
            thread.join()
        raise
    done = [end for end in ends if end is not None]
    if halted.is_set():
        raise TogetherFailed(done, reason_holder[0])
    return tuple(end.result for end in done)
