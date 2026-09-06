"""One pipe, many requests: a late reply must never be executed as the answer to the next one.

⛔⛔ **THIS IS A MOTION PATH ON A ROBOT ARM, AND THE FAILURE WAS SILENT.** `CuroboPlanClient` is
request/response over a single queue. A call that times out stops waiting but leaves its request
outstanding, so the sidecar's late answer lands in the queue and used to be handed to whatever asked
next. Three callers in `drivers/ur/curobo_motion.py` swallow the timeout exception and return
`False`, so the poisoned queue survived with nothing said.

Reproduced end to end against a stub sidecar that answers the first plan too late and the second on
time. With the old `_recv`, the second motion received the FIRST goal's trajectory. The arm would
have driven a path planned for a goal nobody asked for.

⛔ **AND A DEAD SIDECAR LEFT THE TYPED MOTION PATH.** `self._proc` was never cleared and `poll()`
never consulted, so a call after the child exited wrote into a dead pipe and raised a raw `OSError`
out of a path whose whole contract is typed, fail-closed refusals.

⚠ **THESE TESTS SPAWN A REAL SUBPROCESS**, this interpreter running a small stub script, because the
defect lives in the interaction between a pipe, a reader thread and a queue. A mocked queue would
prove the code I wrote and not the behaviour I fixed.
"""

from __future__ import annotations

import queue
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

import src.robot.safety.planning.curobo_client as client_module
from src.robot.safety.planning.curobo_client import (
    CuroboPlanClient,
    CuroboUnavailableError,
)

#: Short enough to keep the suite quick, long enough that a healthy stub answers inside it.
_TIMEOUT_S = 0.8

_READY = ('{"status": "ready", "joint_names": ["j0","j1","j2","j3","j4","j5"], "dt": 0.02}')

#: Answers the FIRST request after the client has given up, the rest immediately. The stale answer
#: carries a waypoint value nothing else uses, so a test can say which goal a trajectory belongs to.
_SLOW_FIRST = textwrap.dedent(f"""
    import json, sys, time
    def emit(obj, rid):
        if rid is not None:
            obj = dict(obj, id=rid)
        sys.stdout.write(json.dumps(obj) + chr(10)); sys.stdout.flush()
    sys.stdout.write({_READY!r} + chr(10)); sys.stdout.flush()
    n = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("cmd") == "shutdown":
            break
        n += 1
        if n == 1:
            time.sleep({_TIMEOUT_S * 2:.2f})
            emit({{"success": True, "trajectory": [[111.0] * 6]}}, req.get("id"))
        else:
            emit({{"success": True, "trajectory": [[222.0] * 6]}}, req.get("id"))
""")

#: Reports ready, then exits the moment anything is asked of it.
_DIES = textwrap.dedent(f"""
    import json, sys
    sys.stdout.write({_READY!r} + chr(10)); sys.stdout.flush()
    for line in sys.stdin:
        sys.exit(1)
""")

#: Answers correctly but stamps no id, like a sidecar older than this protocol.
_UNSTAMPED = textwrap.dedent(f"""
    import json, sys
    sys.stdout.write({_READY!r} + chr(10)); sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if json.loads(line).get("cmd") == "shutdown":
            break
        sys.stdout.write(json.dumps({{"success": True, "trajectory": [[333.0] * 6]}}) + chr(10))
        sys.stdout.flush()
""")


def _stub(source: str) -> str:
    path = Path(tempfile.mkdtemp()) / "sidecar.py"
    path.write_text(source, encoding="utf-8")
    return str(path)


class _StubbedClient:
    """A client wired to a stub sidecar, with the plan timeout shortened for the duration."""

    def __init__(self, source: str) -> None:
        self._source = source
        self._saved = client_module._PLAN_TIMEOUT_S

    def __enter__(self) -> CuroboPlanClient:
        client_module._PLAN_TIMEOUT_S = _TIMEOUT_S
        self._client = CuroboPlanClient(
            python_path=sys.executable, server_script=_stub(self._source),
            robot_config="unused-by-the-stub", scene_config=None)
        self._client.start()
        return self._client

    def __exit__(self, *exc: object) -> None:
        client_module._PLAN_TIMEOUT_S = self._saved
        try:
            self._client.close()
        except Exception:                                      # noqa: BLE001 - teardown is best-effort
            pass


def _plan(client: CuroboPlanClient, x: float) -> "list[list[float]] | None":
    return client.plan([0.0] * 6, [x, x, x], [1.0, 0.0, 0.0, 0.0])


class ALateReplyIsDroppedRatherThanExecutedTests(unittest.TestCase):

    def test_the_second_motion_gets_its_OWN_trajectory(self) -> None:
        """⛔ THE DEFECT, AS A ROBOT WOULD HAVE MET IT. 111.0 is the first goal's answer, arriving
        after its caller gave up; 222.0 belongs to the second."""
        with _StubbedClient(_SLOW_FIRST) as client:
            with self.assertRaises(CuroboUnavailableError):
                _plan(client, 0.1)
            time.sleep(_TIMEOUT_S)                             # let the stale answer land
            trajectory = _plan(client, 0.9)
        assert trajectory is not None
        self.assertEqual(222.0, trajectory[0][0],
                         "the previous goal's trajectory was returned for this motion")

    def test_the_OLD_recv_returned_the_previous_goal(self) -> None:
        """⚠ THE CONTROL, AND WITHOUT IT THE TEST ABOVE PROVES NOTHING. A stub that simply answered
        every call correctly would pass it too. This pins that the defect is real on this exact
        stub by restoring the pre-fix read and watching it fail."""
        def take_whatever_is_next(self, timeout_s, *, want=None):  # type: ignore[no-untyped-def]
            try:
                return self._q.get(timeout=timeout_s)
            except queue.Empty:
                return None

        saved = CuroboPlanClient._recv
        CuroboPlanClient._recv = take_whatever_is_next          # type: ignore[method-assign]
        try:
            with _StubbedClient(_SLOW_FIRST) as client:
                with self.assertRaises(CuroboUnavailableError):
                    _plan(client, 0.1)
                time.sleep(_TIMEOUT_S)
                trajectory = _plan(client, 0.9)
        finally:
            CuroboPlanClient._recv = saved                      # type: ignore[method-assign]
        assert trajectory is not None
        self.assertEqual(111.0, trajectory[0][0],
                         "the pre-fix client no longer reproduces the defect, so the test above is "
                         "measuring something else")

    def test_a_sidecar_that_stamps_no_id_is_still_answered(self) -> None:
        """⚠ REFUSING EVERY UNSTAMPED REPLY WOULD BE A WORSE FAILURE THAN THE ONE THIS GUARDS. A
        sidecar older than this protocol answers correctly; it simply cannot be told apart from a
        late one. It is accepted, and the client says so once."""
        with _StubbedClient(_UNSTAMPED) as client:
            trajectory = _plan(client, 0.4)
            self.assertTrue(client._warned_unstamped, "the missing id passed unnoticed")
        assert trajectory is not None
        self.assertEqual(333.0, trajectory[0][0])

    def test_the_timeout_budget_is_not_reset_by_a_dropped_message(self) -> None:
        """⚠ OTHERWISE A SIDECAR EMITTING STALE LINES HOLDS A MOTION OPEN FOREVER. The drop loop
        must spend the caller's budget, not restart it."""
        client = CuroboPlanClient.__new__(CuroboPlanClient)
        client._q = queue.Queue()
        client._alive = True
        client._warned_unstamped = False
        for stale in range(6):
            client._q.put({"id": stale, "success": True})
        started = time.monotonic()
        self.assertIsNone(client._recv(0.3, want=999))
        self.assertLess(time.monotonic() - started, 1.0,
                        "six stale messages extended a 0.3 s budget")


class ADeadSidecarFailsTypedTests(unittest.TestCase):
    """⛔ A RAW `OSError` OUT OF THE FAIL-CLOSED MOTION PATH is a failure the callers cannot classify.
    They turn `CuroboUnavailableError` into a typed `CONTROLLER_REJECTED`; an `OSError` from a pipe
    goes past them."""

    def _outcomes(self, delay_s: float) -> list[str]:
        seen: list[str] = []
        with _StubbedClient(_DIES) as client:
            for call in (1, 2):
                try:
                    _plan(client, 0.1)
                    seen.append("returned")
                except CuroboUnavailableError:
                    seen.append("typed")
                except OSError as exc:
                    seen.append(f"raw {type(exc).__name__}")
                if call == 1:
                    time.sleep(delay_s)
        return seen

    def test_every_call_after_the_sidecar_dies_is_typed(self) -> None:
        """MEASURED at three delays because the pre-fix behaviour depended on the race: an immediate
        second call could still see the EOF sentinel, a realistic one wrote into a dead pipe."""
        for delay in (0.0, 0.3, 1.0):
            with self.subTest(delay=delay):
                self.assertEqual(["typed", "typed"], self._outcomes(delay))

    def test_the_client_knows_its_sidecar_is_gone(self) -> None:
        """`self._proc` alone could not say this: it is cleared only by `close()`, so a client whose
        child had exited still believed it was running."""
        with _StubbedClient(_DIES) as client:
            with self.assertRaises(CuroboUnavailableError):
                _plan(client, 0.1)
            time.sleep(0.3)
            self.assertFalse(client._alive)

    def test_the_refusal_names_the_exit_code(self) -> None:
        """An operator reading one line at 2 a.m. needs to know the child exited, not that a write
        failed."""
        with _StubbedClient(_DIES) as client:
            with self.assertRaises(CuroboUnavailableError):
                _plan(client, 0.1)
            time.sleep(0.3)
            with self.assertRaises(CuroboUnavailableError) as caught:
                _plan(client, 0.2)
        self.assertIn("exited", str(caught.exception))


class TheProtocolCarriesTheIdOnBothSidesTests(unittest.TestCase):
    """The server is run by a DIFFERENT interpreter, so its half is checked as source rather than
    imported: this repository cannot import cuRobo's python from its own."""

    def test_the_server_stamps_every_reply_from_one_place(self) -> None:
        source = (Path(client_module.__file__).with_name("curobo_planner_server.py")
                  .read_text(encoding="utf-8"))
        self.assertIn('obj = {**obj, "id": _REQUEST_ID}', source,
                      "the server no longer stamps its replies")
        self.assertIn('_REQUEST_ID = req.get("id")', source,
                      "the server no longer reads the request id")

    def test_every_request_method_demands_its_own_reply(self) -> None:
        """⚠ ONE FORGOTTEN CALL SITE WOULD BE THE WHOLE DEFECT AGAIN, on whichever verb it was."""
        source = Path(client_module.__file__).read_text(encoding="utf-8")
        unpaired = [line.strip() for line in source.splitlines()
                    if "self._recv(_PLAN_TIMEOUT_S)" in line]
        self.assertEqual([], unpaired,
                         "a request method reads the queue without naming which reply it wants")


if __name__ == "__main__":
    unittest.main()
