"""The event stream, and the reason it has sequence numbers.

A run must survive the browser. An operator starts a pick, the laptop sleeps, the tab is closed, the
wifi drops. The arm does not care: it is holding a part. So the run finishes regardless, every event
it produced is kept, and a client that comes back asks for everything after seq 41 and receives 42
onwards before the live feed resumes. Without a monotonic ``seq`` there is no way to ask that
question, and a UI that reconnects can only guess whether it missed anything.

The buffer is bounded and says when it dropped. A ring buffer per run, and a reconnect asking for a
``since_seq`` older than the buffer gets a truthful gap marker instead of a silently short replay. A
console that renders "events 12-40 are gone" is annoying; one that renders 41 onwards as if nothing
were missing is wrong.

Two audiences, one envelope. ``human`` is a plain sentence for the operator; ``data`` is the machine
payload. Both, always: the person watching a demo reads one, the person diagnosing a failure reads the
other, and an envelope that carries only one of them forces the other to guess.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = ["EventEnvelope", "EventHub", "Severity"]

#: Events kept per run. At five stage events per attempt and five attempts per pick, this holds dozens
#: of picks: more than a browser needs to catch up after a sleep, and small enough that a forgotten
#: long-running server does not grow without bound.
_RING = 2048


class Severity(StrEnum):
    """How loudly to render. Not a log level: it is what the operator's eye should go to first."""

    INFO = "info"
    #: Something the operator should notice but that did not stop anything.
    WARN = "warn"
    #: The run stopped, or the cell refused.
    ERROR = "error"
    #: A milestone worth celebrating on a demo screen: a successful pick.
    SUCCESS = "success"


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """One thing that happened, addressed to both a person and a program."""

    type: str
    run_id: str
    #: Monotonic within a run, starting at 1. A reconnecting client resumes from the last one it saw.
    seq: int
    #: Unix seconds. Wall clock on purpose: it is compared against the operator's own clock.
    ts: float
    severity: Severity = Severity.INFO
    #: A plain sentence. "Ranked 3 candidates, best score 0.85."
    human: str = ""
    #: Where in the run this sits, when that is meaningful.
    step: str = ""
    step_index: int | None = None
    step_total: int | None = None
    #: The machine payload. Never a stringified version of ``human``.
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "run_id": self.run_id,
            "seq": self.seq,
            "ts": self.ts,
            "severity": str(self.severity),
            "human": self.human,
            "step": self.step,
            "step_index": self.step_index,
            "step_total": self.step_total,
            "data": self.data,
        }


class EventHub:
    """Per-run sequence numbers, a bounded history, and live subscribers.

    Publishing is safe from the pick thread; subscribing is safe from a request thread. A run
    publishes from a robot-driving thread while a browser subscribes from an event loop, so every
    mutation is under one lock: an interleaving that gives two events the same ``seq`` leaves a
    reconnecting client unable to tell what it missed.
    """

    def __init__(self, *, capacity: int = _RING) -> None:
        self._capacity = capacity
        self._lock = threading.RLock()
        self._history: dict[str, deque[EventEnvelope]] = {}
        self._next_seq: dict[str, int] = {}
        #: One condition per hub; waiters re-check their own run's tail on every publish. Simpler than
        #: per-run conditions and correct: a wake-up meant for another run costs one comparison.
        self._published = threading.Condition(self._lock)

    def publish(
        self,
        run_id: str,
        event_type: str,
        *,
        human: str = "",
        severity: Severity = Severity.INFO,
        step: str = "",
        step_index: int | None = None,
        step_total: int | None = None,
        **data: Any,
    ) -> EventEnvelope:
        """Append one event and wake anything waiting. Never raises for an unknown run."""
        with self._lock:
            seq = self._next_seq.get(run_id, 1)
            self._next_seq[run_id] = seq + 1
            envelope = EventEnvelope(
                type=event_type, run_id=run_id, seq=seq, ts=time.time(),
                severity=severity, human=human, step=step,
                step_index=step_index, step_total=step_total, data=dict(data),
            )
            self._history.setdefault(run_id, deque(maxlen=self._capacity)).append(envelope)
            self._published.notify_all()
            return envelope

    def since(self, run_id: str, since_seq: int) -> tuple[list[EventEnvelope], int]:
        """``(events after since_seq, number dropped)``.

        A client that slept longer than the buffer is told how many events it will never see, rather
        than receiving a short replay that looks complete.
        """
        with self._lock:
            history = self._history.get(run_id)
            if not history:
                return [], 0
            oldest = history[0].seq
            dropped = max(0, oldest - (since_seq + 1))
            return [e for e in history if e.seq > since_seq], dropped

    def latest_seq(self, run_id: str) -> int:
        with self._lock:
            return self._next_seq.get(run_id, 1) - 1

    def wait_for(self, run_id: str, after_seq: int, *, timeout: float) -> list[EventEnvelope]:
        """Block until an event after ``after_seq`` exists, or the timeout expires.

        Returns an empty list on timeout, which is what lets a WebSocket loop send a keep-alive and
        then ask again: a socket that is merely quiet is indistinguishable from a dead one otherwise.
        """
        deadline = time.monotonic() + timeout
        with self._published:
            while True:
                pending = [
                    e for e in self._history.get(run_id, ()) if e.seq > after_seq
                ]
                if pending:
                    return pending
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._published.wait(remaining)

    def forget(self, run_id: str) -> None:
        """Drop a run's history. Sequence numbers are not reset: a re-used id would replay stale seqs."""
        with self._lock:
            self._history.pop(run_id, None)
