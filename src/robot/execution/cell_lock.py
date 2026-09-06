"""One owner per cell, enforced by the operating system rather than by a promise.

A UR controller accepts exactly one RTDE control script. Two processes that both call ``connect()``
do not get an error and a queue; they compete for the control script, and the arm moves in ways
nobody commanded.

The lock is a byte range held by an open file handle, not a PID file. A PID file records an
intention and holds nothing: a killed process leaves one behind for the next operator to delete by
hand. A byte lock is released by the OS when the process dies for any reason, including ``kill -9``
and a blue screen, so there is no stale state to clean up.

Release is not instantaneous. After ``Popen.kill()`` and ``wait()`` return, the byte lock stays held
for a short interval while the OS tears the handle down, so a caller that kills a holder and
immediately acquires can see one spurious :class:`CellBusy`. :meth:`CellLock.acquire` does not
retry: a retry loop would delay every genuinely refused acquisition and would make "never blocks"
untrue. A caller that restarts itself under a supervisor retries deliberately.

What it does not protect against: the lock lives on this machine, and the controller it stands for
lives on the network. A laptop across the room can open its own RTDE connection without ever seeing
this file. Only asking the controller can catch that, which is the job of the live probe and not of
this module. On Windows the lock file lives in the per-user temp directory, so two user accounts on
one PC do not see each other's lock.

The key is the controller's address, because that is the resource. Two arms behind one controller
share a lock, which over-locks; the gripper's URCap socket and the cuRobo sidecar's GPU are separate
resources that ride along on the same key. One coarse lock that is always right about the arm is
preferred to four fine ones.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

from src.robot.constants import CELL_LOCK_LOG_FILE, create_robot_logger

__all__ = ["CellBusy", "CellLock", "LockHolder", "cell_lock_key", "lock_path_for", "peek"]

# The holder record lives in a file the OS reclaims when the process dies, so it leaves no history.
# This log is the history: who took the cell, when, and when they gave it back. A refusal is not
# logged here; :class:`CellBusy` carries it to the caller already.
logger = create_robot_logger("CellLock", CELL_LOCK_LOG_FILE)

#: Byte 0 is the locked range and never carries data; the holder record starts at byte 1. Keeping them
#: in one file means there is no second file to go stale, and no window where one exists without the
#: other.
_LOCK_BYTE = b"#"
_INFO_OFFSET = 1

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def lock_path_for(key: str) -> Path:
    """Where the lock for ``key`` lives.

    The system temp directory, not the repo: two checkouts of this repository on one machine are two
    processes driving the same physical arm, and a repo-relative lock would let both think they held
    it. Temp is also self-cleaning across reboots, and a lock cannot outlive the process that held it.
    """
    return Path(tempfile.gettempdir()) / f"willy-cell-{_SAFE.sub('_', key) or 'default'}.lock"


def cell_lock_key(robot_config: object) -> str | None:
    """The lockable resource this config drives, or ``None`` when there is nothing to contend for.

    A simulated or dummy cell owns no controller, so it yields ``None`` and two rehearsals can run
    at once. Only a vendor with a real controller address produces a key.
    """
    vendor = str(getattr(robot_config, "vendor", "") or "")
    if vendor == "ur":
        ip = getattr(getattr(robot_config, "ur", None), "ip", None)
        return f"ur@{ip}" if ip else None
    if vendor == "kuka":
        ip = getattr(getattr(robot_config, "kuka", None), "controller_ip", None)
        return f"kuka@{ip}" if ip else None
    # sim / dummy / anything unrecognised: no exclusive physical resource is being claimed. An
    # unknown vendor returns None rather than a guess; a key invented for a driver this function
    # does not describe would lock something arbitrary and protect nothing.
    return None


@dataclass(frozen=True, slots=True)
class LockHolder:
    """Who holds the cell, as recorded by the holder itself while it held the lock."""

    #: Free text naming the tool, e.g. ``"real_cell CLI"`` or ``"operator console"``.
    owner: str
    pid: int
    #: ISO-8601 UTC. Written at acquisition.
    since: str
    host: str = ""

    def describe(self) -> str:
        where = f" on {self.host}" if self.host else ""
        return f"{self.owner} (pid {self.pid}{where}, since {self.since})"


class CellBusy(RuntimeError):
    """Raised when another process already owns the cell. Carries who, so the message can say so."""

    def __init__(self, key: str, holder: LockHolder | None, path: Path) -> None:
        who = holder.describe() if holder is not None else "another process"
        super().__init__(
            f"the cell at {key} is already owned by {who}. Only one process may hold a UR control "
            f"script; a second connection does not queue, it competes. Stop the other one, or wait for "
            f"it to finish. (lock: {path})"
        )
        self.key = key
        self.holder = holder
        self.path = path


class CellLock:
    """Exclusive ownership of one cell, released by the OS when this process ends.

    Not reentrant and not blocking: a contended acquire raises :class:`CellBusy` naming the holder
    rather than waiting an unbounded time.

        with CellLock(config.ur.ip, owner="operator console"):
            arm.connect()
            ...
    """

    def __init__(self, key: str, *, owner: str) -> None:
        self.key = key
        self.owner = owner
        self.path = lock_path_for(key)
        self._fd: int | None = None

    # --- acquisition ------------------------------------------------------------------------------

    def acquire(self) -> None:
        """Take the lock, or raise :class:`CellBusy` naming the holder. Never blocks."""
        if self._fd is not None:
            raise RuntimeError("this CellLock is already held by this object")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Opened without truncation: the previous holder's record must survive long enough to be
        # read back in the failure path below, and truncating before the lock is taken destroys it.
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            os.write(fd, _LOCK_BYTE) if os.fstat(fd).st_size == 0 else None
            _lock_byte0(fd)
        except OSError as exc:
            holder = _read_holder(fd)
            os.close(fd)
            raise CellBusy(self.key, holder, self.path) from exc
        except BaseException:
            os.close(fd)
            raise

        self._fd = fd
        _write_holder(
            fd,
            LockHolder(
                owner=self.owner,
                pid=os.getpid(),
                since=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                host=_hostname(),
            ),
        )
        logger.info("acquired the cell %s for %r (pid %d, lock %s)", self.key, self.owner, os.getpid(), self.path)

    def release(self) -> None:
        """Give it up. Idempotent, and never raises: a teardown that can fail is not a teardown."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        logger.info("released the cell %s held by %r (pid %d)", self.key, self.owner, os.getpid())
        try:
            _unlock_byte0(fd)
        except OSError:
            # The handle is about to close, which releases the lock regardless. Nothing to recover.
            pass
        try:
            os.close(fd)
        except OSError:
            pass

    @property
    def held(self) -> bool:
        return self._fd is not None

    def __enter__(self) -> CellLock:
        self.acquire()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.release()


def peek(key: str) -> LockHolder | None:
    """Who holds ``key`` right now. ``None`` if nobody does.

    The answer comes from taking the lock and handing it straight back, which is the only way to
    tell held from free: for that instant ``peek`` is the holder, and a concurrent
    :meth:`CellLock.acquire` can be refused with a :class:`CellBusy` naming whatever record was
    last written. Best-effort and a snapshot: the answer can be stale the instant it is returned.
    Use it to render a status line, never to decide whether it is safe to connect. That decision
    belongs to :meth:`CellLock.acquire`, which cannot race because the OS arbitrates it.
    """
    path = lock_path_for(key)
    if not path.exists():
        return None
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return None
    try:
        try:
            _lock_byte0(fd)
        except OSError:
            return _read_holder(fd)  # held by someone else, so their record is current
        _unlock_byte0(fd)
        return None  # the lock was available, so nobody holds it
    finally:
        os.close(fd)


# --------------------------------------------------------------------------------------------------
# Platform primitives. Both give the same guarantee through entirely different APIs: the lock is a
# property of the open handle, so it disappears when the process does.
# --------------------------------------------------------------------------------------------------

if sys.platform == "win32":  # pragma: no cover (the other branch is unreachable here)
    import msvcrt

    def _lock_byte0(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        # LK_NBLCK = non-blocking exclusive; raises OSError immediately if another handle holds it.
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _unlock_byte0(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:  # pragma: no cover (not reachable on Windows)
    import fcntl

    def _lock_byte0(fd: int) -> None:
        # flock is whole-file rather than byte-range, which is the same guarantee for this use: it is
        # advisory between flock callers, and every caller here is this module.
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock_byte0(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _write_holder(fd: int, holder: LockHolder) -> None:
    """Record the current holder past the locked byte. Called only while holding the lock."""
    payload = json.dumps(
        {"owner": holder.owner, "pid": holder.pid, "since": holder.since, "host": holder.host}
    ).encode("utf-8")
    os.lseek(fd, _INFO_OFFSET, os.SEEK_SET)
    os.write(fd, payload)
    os.truncate(fd, _INFO_OFFSET + len(payload))
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover (fsync is unavailable on some filesystems)
        pass


def _read_holder(fd: int) -> LockHolder | None:
    """The holder's record, or ``None`` if it is missing or unreadable.

    Unreadable is not an error worth propagating: the lock itself already answered the only question
    that matters, and a garbled record must not turn "someone else has the cell" into a traceback.
    """
    try:
        os.lseek(fd, _INFO_OFFSET, os.SEEK_SET)
        raw = os.read(fd, 4096)
    except OSError:
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
        return LockHolder(
            owner=str(data["owner"]), pid=int(data["pid"]),
            since=str(data["since"]), host=str(data.get("host", "")),
        )
    except (ValueError, KeyError, TypeError):
        return None


def _hostname() -> str:
    try:
        import socket

        return socket.gethostname()
    except Exception:  # pragma: no cover (a name is a nicety, never a reason to fail)
        return ""
