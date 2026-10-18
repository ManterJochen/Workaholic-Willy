"""A cell of several arms before every send of one arm: the gate its transport carries (owner, 2026-10-15).

:func:`gate_sends` puts a cell's gate before the motion verbs of one transport object, the one the arm, its planner and
a hand guiding session all send through. ``run_trajectory`` reserves the joints it runs through, from where the arm
stands; ``move_pose`` and hand guiding reserve the whole cell, hand guiding until it ends. :func:`holding` reserves a
span of several sends at once, inside which each send passes: the planner's waypoints as one route, a line from its
judgement to its end. The gate goes on the object itself, not on a wrapper about it, so every reference to it is gated
and every check of what it can do (``isinstance`` against the transport Protocols) answers as before.

A send the cell does not grant raises :class:`CellOccupied` before anything is sent. A transport no cell gates is
untouched, and :func:`holding` holds nothing on it: a cell of one arm runs as it always did.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Protocol, Sequence

from .errors import RobotMotionRejected
from .motion_result import MotionCommand, MotionResult, MotionSent, MotionStatus

__all__ = ["CellGate", "CellOccupied", "gate_sends", "holding", "ungate_sends"]

#: Where the gate's state hangs on a gated transport.
_SENDS = "_willy_cell_sends"


class CellOccupied(RobotMotionRejected):
    """A send the cell did not grant: another arm stands or moves where it would go, or holds the whole cell, and that did
    not clear within the cell's wait. Nothing was sent.

    Args:
        message (str): Why, naming the arm in the way.
        status (MotionStatus): How a typed verb reports it (default: SELF_COLLISION_REJECTED).
    """

    def __init__(self, message: str, *, status: MotionStatus = MotionStatus.SELF_COLLISION_REJECTED) -> None:
        super().__init__(message)
        #: How a typed verb reports it.
        self.status = status

    def refusal(self, command: MotionCommand, **targets: Any) -> MotionResult:
        """The typed refusal of ``command``: this status and sentence, nothing sent.

        Args:
            command (MotionCommand): The command refused.
            **targets (Any): ``target_pose`` or ``target_joints``, as the verb names its target.

        Returns:
            MotionResult: The refusal.
        """
        return MotionResult.failed(self.status, command, **targets, message=str(self), sent=MotionSent.NOTHING)


class CellGate(Protocol):
    """One arm's door to its cell: what :func:`gate_sends` asks before a span and tells after it."""

    def reserve(self, route: "Sequence[Sequence[float]] | None") -> None:
        """Hold ``route`` (the whole cell for ``None``) for this arm, or raise :class:`CellOccupied`."""

    def release(self) -> None:
        """End what this arm holds."""


class _Sends:
    """One gated transport's spans: how deep the sends in flight nest, and whether a person guides the arm."""

    def __init__(self, gate: CellGate) -> None:
        self.gate = gate
        self.depth = 0
        self.guided = False
        self.lock = threading.Lock()
        #: The verbs the gate stands before, given back by :func:`ungate_sends`.
        self.gated: tuple[str, ...] = ()

    @contextmanager
    def span(self, route: "Callable[[], Sequence[Sequence[float]]] | None") -> Iterator[None]:
        """Reserve ``route()`` (the whole cell for ``None``) unless a span is held already; release at its end."""
        with self.lock:
            outer = self.depth == 0
            self.depth += 1
        try:
            if outer:
                self.gate.reserve(None if route is None else route())
        except BaseException:
            with self.lock:
                self.depth -= 1
            raise
        try:
            yield
        finally:
            with self.lock:
                self.depth -= 1
            if outer:
                self.gate.release()

    def guide(self) -> None:
        self.gate.reserve(None)
        self.guided = True

    def unguide(self) -> None:
        if self.guided:
            self.guided = False
            self.gate.release()


def gate_sends(transport: Any, gate: CellGate) -> None:
    """Put ``gate`` before every motion verb ``transport`` has, once.

    Args:
        transport (Any): The arm's transport object.
        gate (CellGate): The arm's door to its cell.

    Raises:
        RuntimeError: ``transport`` is gated already.
    """
    if getattr(transport, _SENDS, None) is not None:
        raise RuntimeError("this transport's sends are behind a cell's gate already")
    sends = _Sends(gate)
    verbs: dict[str, Any] = {name: getattr(transport, name, None) for name in
                             ("run_trajectory", "move_pose", "begin_hand_guiding", "end_hand_guiding", "disconnect")}

    def run_trajectory(waypoints: "Sequence[Sequence[float]]", *args: Any, **kwargs: Any) -> Any:
        with sends.span(lambda: [transport.read_joints(), *waypoints]):
            return verbs["run_trajectory"](waypoints, *args, **kwargs)

    def move_pose(*args: Any, **kwargs: Any) -> Any:
        with sends.span(None):
            return verbs["move_pose"](*args, **kwargs)

    def begin_hand_guiding() -> Any:
        sends.guide()
        began = False
        try:
            began = verbs["begin_hand_guiding"]()
        finally:
            if not began:
                sends.unguide()
        return began

    def ending(name: str) -> Callable[[], Any]:
        def end() -> Any:
            try:
                return verbs[name]()
            finally:
                sends.unguide()
        return end

    setattr(transport, _SENDS, sends)
    gated = {"run_trajectory": run_trajectory, "move_pose": move_pose, "begin_hand_guiding": begin_hand_guiding,
             "end_hand_guiding": ending("end_hand_guiding"), "disconnect": ending("disconnect")}
    for name, verb in verbs.items():
        if callable(verb):
            setattr(transport, name, gated[name])
    sends.gated = tuple(name for name, verb in verbs.items() if callable(verb))


def ungate_sends(transport: Any) -> None:
    """Take the gate off ``transport`` again: its own verbs answer once more. Nothing on a transport no cell gates.

    Args:
        transport (Any): The arm's transport object.
    """
    sends = getattr(transport, _SENDS, None)
    if sends is None:
        return
    for name in sends.gated:
        delattr(transport, name)
    delattr(transport, _SENDS)


@contextmanager
def holding(transport: Any, route: "Callable[[], Sequence[Sequence[float]]] | None") -> Iterator[None]:
    """Hold ``route()`` in the cell, or the whole cell for ``None``, over every send inside the block; nothing on a
    transport no cell gates.

    Args:
        transport (Any): The arm's transport object.
        route (Callable[[], Sequence[Sequence[float]]] | None): The joints the block's sends run through, the arm's own
            first, read when the span is reserved; ``None`` for the whole cell.

    Yields:
        None: Inside the span.

    Raises:
        CellOccupied: The cell did not grant the span; nothing inside the block ran.
    """
    sends = getattr(transport, _SENDS, None)
    if sends is None:
        yield
        return
    with sends.span(route):
        yield
