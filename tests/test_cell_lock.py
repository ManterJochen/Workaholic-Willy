"""One owner per cell: does the lock actually hold, and does it actually let go?

The load-bearing test is ``test_a_hard_killed_holder_releases_the_cell``. It spawns a real process, has
it take the lock, then ``kill()``s it -- no ``finally``, no ``atexit``, no signal handler gets to run --
and asserts the next process can take the cell immediately. That is the whole reason this is an OS byte
lock and not a PID file: a lock that survives its holder's death gets deleted by hand at 2am during a
bring-up, and the habit of deleting it is what eventually deletes a live one.

Everything here is honesty bucket (2): the lock is exercised for real, on real processes, on this
platform. What it cannot be tested for is the case it does not cover -- a second machine on the network
opening its own connection to the same controller. No local lock can see that, and this suite does not
pretend otherwise.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
import unittest
from pathlib import Path

from src.robot.execution.cell_lock import (
    CellBusy,
    CellLock,
    LockHolder,
    lock_path_for,
    peek,
)

_ROOT = Path(__file__).resolve().parents[1]


class LockPathTests(unittest.TestCase):
    def test_the_key_becomes_a_safe_filename(self) -> None:
        """An IP is the key, and an IP is not a filename until the dots and colons are dealt with."""
        self.assertEqual(lock_path_for("192.168.1.100").name, "willy-cell-192.168.1.100.lock")
        self.assertEqual(lock_path_for("10.0.0.5:30004").name, "willy-cell-10.0.0.5_30004.lock")
        self.assertEqual(lock_path_for("../../etc/passwd").name, "willy-cell-.._.._etc_passwd.lock")
        self.assertEqual(lock_path_for("").name, "willy-cell-default.lock")

    def test_it_lives_outside_the_repository(self) -> None:
        """Two checkouts on one machine are two processes driving ONE arm.

        A repo-relative lock would let each of them believe it held the cell, which is exactly the
        collision the lock exists to prevent.
        """
        self.assertNotIn(_ROOT.resolve(), lock_path_for("192.168.1.100").resolve().parents)


class CellLockTests(unittest.TestCase):
    KEY = "test-cell-198.51.100.7"

    def tearDown(self) -> None:
        lock_path_for(self.KEY).unlink(missing_ok=True)

    def test_a_second_holder_is_refused_and_told_who_has_it(self) -> None:
        """"Busy" without a name sends an operator hunting; the message must end the hunt."""
        with CellLock(self.KEY, owner="real_cell CLI") as first:
            self.assertTrue(first.held)
            with self.assertRaises(CellBusy) as ctx:
                CellLock(self.KEY, owner="operator console").acquire()

        self.assertEqual(ctx.exception.key, self.KEY)
        assert ctx.exception.holder is not None
        self.assertEqual(ctx.exception.holder.owner, "real_cell CLI")
        self.assertEqual(ctx.exception.holder.pid, os.getpid())
        self.assertIn("real_cell CLI", str(ctx.exception))

    def test_it_never_blocks(self) -> None:
        """A blocking lock would hold a browser request open for an unbounded time."""
        with CellLock(self.KEY, owner="first"):
            started = time.monotonic()
            with self.assertRaises(CellBusy):
                CellLock(self.KEY, owner="second").acquire()
            self.assertLess(time.monotonic() - started, 1.0)

    def test_releasing_hands_the_cell_on(self) -> None:
        lock = CellLock(self.KEY, owner="first")
        lock.acquire()
        lock.release()
        self.assertFalse(lock.held)
        with CellLock(self.KEY, owner="second"):
            assert (holder := peek(self.KEY)) is not None
            self.assertEqual(holder.owner, "second")

    def test_release_is_idempotent_and_never_raises(self) -> None:
        """A teardown that can fail is not a teardown -- it is a second thing that can go wrong."""
        lock = CellLock(self.KEY, owner="first")
        lock.release()          # never acquired
        lock.acquire()
        lock.release()
        lock.release()          # already released
        self.assertFalse(lock.held)

    def test_the_same_object_refuses_to_take_the_lock_twice(self) -> None:
        """Not reentrant, and it says so rather than leaking the first handle."""
        with CellLock(self.KEY, owner="first") as lock:
            with self.assertRaises(RuntimeError):
                lock.acquire()

    def test_different_cells_do_not_collide(self) -> None:
        other = "test-cell-198.51.100.8"
        self.addCleanup(lock_path_for(other).unlink, True)
        with CellLock(self.KEY, owner="ur3e"), CellLock(other, owner="ur5e"):
            pass  # both held at once: two arms, two locks

    def test_the_exception_survives_a_context_manager_body(self) -> None:
        """``with`` must release on the way out of an exception, or one failed pick locks the cell."""
        with self.assertRaises(ValueError):
            with CellLock(self.KEY, owner="first"):
                raise ValueError("the pick failed")
        self.assertIsNone(peek(self.KEY))


class PeekTests(unittest.TestCase):
    KEY = "test-cell-198.51.100.9"

    def tearDown(self) -> None:
        lock_path_for(self.KEY).unlink(missing_ok=True)

    def test_peek_reports_nothing_for_a_free_cell(self) -> None:
        self.assertIsNone(peek(self.KEY))
        with CellLock(self.KEY, owner="someone"):
            pass
        # The file survives release; "the file exists" must not read as "the cell is held".
        self.assertTrue(lock_path_for(self.KEY).exists())
        self.assertIsNone(peek(self.KEY))

    def test_a_garbled_record_does_not_become_a_traceback(self) -> None:
        """The lock already answered the question that matters; the name is a nicety.

        Reachable for real: a holder killed between taking the lock and finishing its write leaves a
        truncated record behind, and the next process to ask must still be told the cell is busy.
        """
        with CellLock(self.KEY, owner="first"):
            path = lock_path_for(self.KEY)
            with open(path, "r+b") as handle:
                handle.seek(1)
                handle.write(b'{"owner": "trunca')
            self.assertIsNone(peek(self.KEY), "an unreadable record reads as no record")

            # ...and acquiring still fails, which is the part that must never degrade.
            with self.assertRaises(CellBusy) as ctx:
                CellLock(self.KEY, owner="second").acquire()
            self.assertIsNone(ctx.exception.holder)
            self.assertIn("another process", str(ctx.exception))


class HardKillTests(unittest.TestCase):
    """The reason this is an OS lock and not a PID file."""

    KEY = "test-cell-198.51.100.10"

    def tearDown(self) -> None:
        lock_path_for(self.KEY).unlink(missing_ok=True)

    def test_a_hard_killed_holder_releases_the_cell(self) -> None:
        script = textwrap.dedent(
            f"""
            import sys, time
            sys.path.insert(0, {str(_ROOT)!r})
            from src.robot.execution.cell_lock import CellLock
            CellLock({self.KEY!r}, owner="doomed holder").acquire()
            print("ACQUIRED", flush=True)
            time.sleep(120)
            """
        )
        holder = subprocess.Popen(
            [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True,
        )
        try:
            assert holder.stdout is not None
            self.assertEqual(holder.stdout.readline().strip(), "ACQUIRED")

            assert (who := peek(self.KEY)) is not None
            self.assertEqual(who.owner, "doomed holder")
            with self.assertRaises(CellBusy):
                CellLock(self.KEY, owner="console").acquire()

            # No finally, no atexit, no signal handler: the OS is the only thing releasing this.
            holder.kill()
            holder.wait(timeout=30)
        finally:
            if holder.poll() is None:  # pragma: no cover - only on an assertion failure above
                holder.kill()
                holder.wait(timeout=30)

        # Bounded, not instantaneous -- and the bound is a measurement, not a guess. On this Windows
        # workstation the byte lock survives `kill()`+`wait()` by 3.1-4.6 ms (8 runs, median 4.2) while
        # the OS tears the handle down. One second is ~200x that: generous enough not to flake, tight
        # enough that a lock which genuinely leaked would still fail this test.
        deadline = time.monotonic() + 1.0
        while peek(self.KEY) is not None and time.monotonic() < deadline:
            time.sleep(0.002)

        self.assertIsNone(peek(self.KEY), "a dead holder must not still own the cell")
        with CellLock(self.KEY, owner="console"):
            assert (now := peek(self.KEY)) is not None
            self.assertEqual(now.owner, "console")

    def test_a_holder_record_left_by_a_dead_process_is_never_believed(self) -> None:
        """The file can outlive its author; the LOCK cannot. Only the lock is consulted."""
        path = lock_path_for(self.KEY)
        record = LockHolder(owner="ghost", pid=999_999, since="2026-01-01T00:00:00+00:00")
        path.write_bytes(
            b"#" + json.dumps(
                {"owner": record.owner, "pid": record.pid, "since": record.since, "host": ""}
            ).encode("utf-8")
        )

        self.assertIsNone(peek(self.KEY), "a record with no lock behind it means the cell is free")
        with CellLock(self.KEY, owner="console"):
            assert (now := peek(self.KEY)) is not None
            self.assertEqual(now.owner, "console")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
