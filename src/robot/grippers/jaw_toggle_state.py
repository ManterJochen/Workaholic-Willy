"""Where a single-toggle gripper's jaws stand, kept on disk between programs.

A toggle has one pin, and every pulse flips the jaws: the device keeps its own state, the pin says nothing about it,
and with no open switch wired nothing on the controller does either. The driver can only count its own pulses. Held
in memory that count died with every program, and the next program took the jaws to stand open. On the owner's cell
(2026-09-23) that was wrong whenever a program ended with the jaws closed, a pick without a place, a stop, a Ctrl-C,
and every command after it was inverted: the pick's pre-open closed the jaws before the part, the close at the part
opened them, and a release closed them.

So the count is written down, one small JSON file per controller, bank and pin, after every pulse, and the next
program starts from it. Before a pulse the file says the jaws are on their way; after it, where they stand. A program
that died between the two leaves ``in_flight``, and the driver then refuses to pulse until a person has looked and
said where the jaws stand (:func:`declare`, or ``python -m src.robot.drivers.ur --jaws-stand open``).

What this cannot see is a pulse nobody counted: the teach pendant's I/O tab driving the pin, a power cut in the
middle of a stroke, a device reset. After one of those, declare where the jaws stand before running.

The directory is ``logs/robot/state`` under the repository, or ``WILLY_JAW_STATE_DIR``.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ToggleRecord", "declare", "load", "record_pulse_started", "record_stands", "state_dir", "state_path"]

#: The repository root: src/robot/grippers/ is three levels below it.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def state_dir() -> Path:
    """Where the records live: ``WILLY_JAW_STATE_DIR`` when set, else ``logs/robot/state`` in the repository."""
    override = os.environ.get("WILLY_JAW_STATE_DIR")
    return Path(override) if override else _REPO_ROOT / "logs" / "robot" / "state"


def state_path(controller: str, port: str, pin: int) -> Path:
    """The record of one toggle: the controller it is wired to, the bank and the pin."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(controller)).strip("_") or "controller"
    return state_dir() / f"jaw_toggle_{safe}_{port}_{int(pin)}.json"


@dataclass(frozen=True)
class ToggleRecord:
    """What the file says: the jaws stand closed or open, or a pulse was started and not seen to finish."""

    closed: bool
    in_flight: bool
    at: str
    by: str

    def describe(self) -> str:
        if self.in_flight:
            return f"a pulse toward {'CLOSED' if self.closed else 'OPEN'} was started at {self.at} ({self.by}) and not seen to finish"
        return f"the jaws were left {'CLOSED' if self.closed else 'OPEN'} at {self.at} ({self.by})"


def load(path: Path) -> ToggleRecord | None:
    """The record at ``path``, or ``None`` when there is none. A file that does not parse is a pulse in flight.

    Unreadable is not "open": a record that cannot be read says nothing about the jaws, and the one safe reading of
    nothing is that a person has to look.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return ToggleRecord(closed=False, in_flight=True, at="?", by=f"{path.name} could not be read")
    try:
        data = json.loads(raw)
        return ToggleRecord(closed=bool(data["closed"]), in_flight=bool(data.get("in_flight", False)),
                            at=str(data.get("at", "?")), by=str(data.get("by", "?")))
    except (ValueError, KeyError, TypeError):
        return ToggleRecord(closed=False, in_flight=True, at="?", by=f"{path.name} does not parse")


def _write(path: Path, *, closed: bool, in_flight: bool, by: str) -> None:
    """Replace the record in one step, so a reader sees the old record or the new one and never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps({"closed": bool(closed), "in_flight": bool(in_flight),
                       "at": time.strftime("%Y-%m-%d %H:%M:%S"), "by": by}, indent=2)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(body + "\n", encoding="utf-8")
    os.replace(tmp, path)


def record_pulse_started(path: Path, *, toward_closed: bool, by: str) -> None:
    """Before a pulse: the jaws are on their way to ``toward_closed``, not there yet."""
    _write(path, closed=toward_closed, in_flight=True, by=by)


def record_stands(path: Path, *, closed: bool, by: str) -> None:
    """After a pulse, or at a connect that knows: where the jaws stand."""
    _write(path, closed=closed, in_flight=False, by=by)


def declare(path: Path, *, closed: bool) -> None:
    """A person looked at the jaws and says where they stand. The only way out of a pulse in flight."""
    _write(path, closed=closed, in_flight=False, by="declared by a person who looked at the jaws")
