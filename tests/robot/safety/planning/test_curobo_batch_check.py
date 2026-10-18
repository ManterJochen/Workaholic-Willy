"""A joint path is judged in one request, and no way that request can fail reaches a caller as a pass.

The sidecar's ``check_js`` request judges every configuration of a joint path against the joint
limits, the robot itself and the world the planner holds, and answers with the first sample it
refuses. ``CuroboPlanClient.check_joints`` sends it and reads the answer into a ``JointCheckVerdict``.

What these tests hold the client to:

* the verdict and the first refused index are the sidecar's, for the samples in the order sent;
* no configuration at all, a value that is not a finite number, or a configuration of the wrong
  length is refused before anything is sent, while a path over the cap of 1000 configurations is
  split across requests;
* a sidecar older than ``check_js`` answers it through its plan branch, and the client says the
  sidecar must be restarted instead of reporting a planner failure;
* a failed call, a reply that is not a whole verdict, no reply and a dead sidecar all raise
  ``CuroboUnavailableError`` and never return a verdict;
* a late reply to an earlier check is dropped rather than read as the answer to the next one.

The stubs are real subprocesses run by this interpreter, as in test_curobo_client_correlation.py,
because the client's half lives in a pipe, a reader thread and a queue. The sidecar's own half needs
the cuRobo environment and a GPU, so it is pinned by source scans in
test_curobo_client_error_vs_verdict.py instead.
"""

from __future__ import annotations

import dataclasses
import json
import math
import queue
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from typing import Any

import src.robot.safety.planning.curobo_client as client_module
from src.robot.safety.planning.curobo_client import (
    CuroboPlanClient,
    CuroboUnavailableError,
)

#: Short enough to keep the suite quick, long enough that a healthy stub answers inside it.
_TIMEOUT_S = 0.8

_READY = '{"status": "ready", "joint_names": ["j0","j1","j2","j3","j4","j5"], "dt": 0.02}'

#: Every stub opens with the handshake and a reply function that stamps the request id.
_PREAMBLE = textwrap.dedent(f"""
    import json, sys, time
    def emit(obj, rid):
        if rid is not None:
            obj = dict(obj, id=rid)
        sys.stdout.write(json.dumps(obj) + chr(10)); sys.stdout.flush()
    sys.stdout.write({_READY!r} + chr(10)); sys.stdout.flush()
""")

#: Judges like a sidecar that knows check_js: a sample whose first joint reaches 1.0 rad is inside a
#: wall. The index it answers is computed from the samples it received, so reading index 3 back also
#: shows that the samples arrived whole and in order.
_JUDGES = _PREAMBLE + textwrap.dedent("""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("cmd") == "shutdown":
            break
        if req.get("cmd") != "check_js":
            emit({"success": False, "planner_error": True, "reason": "not modelled"}, req.get("id"))
            continue
        flags = [row[0] < 1.0 for row in req["joints"]]
        first = next((i for i, ok in enumerate(flags) if not ok), None)
        emit({"success": True, "valid": first is None, "first_invalid": first,
              "checked": len(flags)}, req.get("id"))
""")

#: A sidecar from before check_js, dispatching as that sidecar does: a command it does not know falls
#: through into the plan branch, which reads start_joints first and fails on it.
_OLD = _PREAMBLE + textwrap.dedent("""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("cmd") == "shutdown":
            break
        try:
            start = req["start_joints"]
            emit({"success": True, "trajectory": [start], "dt": 0.02}, req.get("id"))
        except Exception as exc:
            emit({"success": False, "planner_error": True,
                  "reason": f"{type(exc).__name__}: {exc}"}, req.get("id"))
""")

#: Reads every request and answers none of them.
_SILENT = _PREAMBLE + textwrap.dedent("""
    for line in sys.stdin:
        line = line.strip()
        if line and json.loads(line).get("cmd") == "shutdown":
            break
""")

#: Reports ready, then exits the moment anything is asked of it.
_DIES = _PREAMBLE + textwrap.dedent("""
    for line in sys.stdin:
        sys.exit(1)
""")

#: Answers the first check after the client has given up, with a refusal at index 1, and every later
#: check at once as valid. A client that read the late refusal as the next answer shows it.
_SLOW_FIRST = _PREAMBLE + textwrap.dedent(f"""
    n = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("cmd") == "shutdown":
            break
        n += 1
        count = len(req["joints"])
        if n == 1:
            time.sleep({_TIMEOUT_S * 2:.2f})
            emit({{"success": True, "valid": False, "first_invalid": 1, "checked": count}},
                 req.get("id"))
        else:
            emit({{"success": True, "valid": True, "first_invalid": None, "checked": count}},
                 req.get("id"))
""")


def _answering(reply: dict[str, Any]) -> str:
    """A stub that answers every request with ``reply``, stamped with the request's id."""
    return _PREAMBLE + textwrap.dedent(f"""
        REPLY = json.loads({json.dumps(reply)!r})
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            req = json.loads(line)
            if req.get("cmd") == "shutdown":
                break
            emit(dict(REPLY), req.get("id"))
    """)


def _stub(source: str) -> str:
    path = Path(tempfile.mkdtemp()) / "sidecar.py"
    path.write_text(source, encoding="utf-8")
    return str(path)


class _StubbedClient:
    """A client wired to a stub sidecar, with the reply timeout shortened for the duration."""

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
        except Exception:  # teardown is best effort
            pass


def _still(count: int) -> list[list[float]]:
    """``count`` samples clear of the judging stub's wall."""
    return [[0.0] * 6 for _ in range(count)]


class TheVerdictIsTheSidecarsTests(unittest.TestCase):

    def test_check_joints_reads_the_verdict(self) -> None:
        """Samples 3 and 4 sit in the stub's wall and 5 is clear again; the verdict names 3."""
        path = _still(6)
        path[3][0] = path[4][0] = 1.5
        with _StubbedClient(_JUDGES) as client:
            verdict = client.check_joints(path)
        self.assertFalse(verdict.valid)
        self.assertEqual(3, verdict.first_invalid)
        self.assertEqual(6, verdict.checked)
        self.assertIn("sample 3 of 6", verdict.reason)

    def test_a_clear_path_is_valid_with_no_index(self) -> None:
        with _StubbedClient(_JUDGES) as client:
            verdict = client.check_joints(_still(6))
        self.assertEqual((True, None, 6), (verdict.valid, verdict.first_invalid, verdict.checked))
        self.assertTrue(verdict.reason)

    def test_the_verdict_is_frozen(self) -> None:
        verdict = client_module.JointCheckVerdict(
            valid=True, first_invalid=None, checked=1, reason="all 1 samples pass")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            verdict.valid = False  # type: ignore[misc]


class RefusedBeforeAnythingIsSentTests(unittest.TestCase):
    """A request the protocol does not carry is refused in the client; the sidecar never hears of it."""

    def test_the_cap_is_1000_configurations_to_a_REQUEST(self) -> None:
        """⭐ THE PREMISE CHANGED DELIBERATELY, 2026-09-12, and it is worth reading twice.

        This test used to pin that 1001 configurations were REFUSED, because the cap was read as a
        rule about how long a move may be. It is not: it is how many configurations fit in one
        `check_js` request. The M2 Isaac gate measured the difference, refusing about one real cuRobo
        plan in ten. A longer path is split across requests, which thins nothing and leaves no gap.
        """
        self.assertEqual(1000, client_module.MAX_CHECK_CONFIGURATIONS)
        with _StubbedClient(_JUDGES) as client:
            verdict = client.check_joints(_still(1001))
            self.assertEqual(2, client._next_id, "1001 configurations did not take two requests")
            # The control: exactly the cap is still one request.
            client._next_id = 0
            exact = client.check_joints(_still(1000))
            self.assertEqual(1, client._next_id)
        self.assertEqual((True, 1001), (verdict.valid, verdict.checked))
        self.assertEqual((True, 1000), (exact.valid, exact.checked))

    def test_a_request_the_protocol_cannot_carry_is_refused_before_the_sidecar_starts(self) -> None:
        """An empty path is refused as empty, not as a sidecar that would not start.

        This used to be about an oversized path, which is no longer a refusal at all: 1001
        configurations are two requests now. What is still refused before anything is spawned is a
        request that could not be sent whatever the sidecar did.
        """
        client = CuroboPlanClient(python_path=str(Path(tempfile.mkdtemp()) / "absent_python.exe"))
        with self.assertRaises(ValueError):
            client.check_joints([])
        # The control: this client really cannot start, so the refusal above came first.
        with self.assertRaises(CuroboUnavailableError):
            client.check_joints(_still(1))

    def test_an_empty_path_a_non_finite_value_and_a_wrong_length_are_refused(self) -> None:
        cases: dict[str, list[list[float]]] = {
            "no configuration": [],
            "not a number": [[0.0] * 6, [0.0, math.nan, 0.0, 0.0, 0.0, 0.0]],
            "infinite": [[math.inf] + [0.0] * 5],
            "wrong length": [[0.0] * 6, [0.0] * 5],
        }
        with _StubbedClient(_JUDGES) as client:
            for label, path in cases.items():
                with self.subTest(label):
                    with self.assertRaises(ValueError):
                        client.check_joints(path)
            self.assertEqual(0, client._next_id, "a refused request reached the sidecar")


class AnOldSidecarIsNamedTests(unittest.TestCase):

    def test_an_old_sidecar_refuses_typed(self) -> None:
        with _StubbedClient(_OLD) as client:
            with self.assertRaises(CuroboUnavailableError) as caught:
                client.check_joints(_still(4))
            # The control: the same stub plans, so the refusal is about check_js, not a broken stub.
            trajectory = client.plan([0.0] * 6, [0.3, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0])
        self.assertIn("the sidecar does not know check_js; restart it from this tree",
                      str(caught.exception))
        self.assertEqual([[0.0] * 6], trajectory)


class NothingButAWholeVerdictPassesTests(unittest.TestCase):

    def test_a_failed_call_raises_with_its_reason_and_is_not_called_old(self) -> None:
        reply = {"success": False, "planner_error": True, "reason": "RuntimeError: CUDA out of memory"}
        with _StubbedClient(_answering(reply)) as client:
            with self.assertRaises(CuroboUnavailableError) as caught:
                client.check_joints(_still(4))
        message = str(caught.exception)
        self.assertIn("CUDA out of memory", message)
        self.assertNotIn("does not know check_js", message)

    def test_a_reply_that_is_not_a_whole_verdict_raises(self) -> None:
        sent = 4
        # The control: a whole refusal through the same stub mechanism is read as a verdict.
        whole = {"success": True, "valid": False, "first_invalid": 2, "checked": sent}
        with _StubbedClient(_answering(whole)) as client:
            self.assertEqual(2, client.check_joints(_still(sent)).first_invalid)
        replies: dict[str, dict[str, Any]] = {
            "unlabelled failure": {"success": False, "reason": "something"},
            "no valid key": {"success": True, "first_invalid": None, "checked": sent},
            "fewer judged than sent": {"success": True, "valid": True, "first_invalid": None,
                                       "checked": sent - 1},
            "refused with no index": {"success": True, "valid": False, "first_invalid": None,
                                      "checked": sent},
            "an index past the end": {"success": True, "valid": False, "first_invalid": sent,
                                      "checked": sent},
            "valid with an index": {"success": True, "valid": True, "first_invalid": 2,
                                    "checked": sent},
        }
        for label, reply in replies.items():
            with self.subTest(label):
                with _StubbedClient(_answering(reply)) as client:
                    with self.assertRaises(CuroboUnavailableError):
                        client.check_joints(_still(sent))


class NoReplyIsNeverAPassTests(unittest.TestCase):

    def test_a_sidecar_that_does_not_answer_raises(self) -> None:
        with _StubbedClient(_SILENT) as client:
            started = time.monotonic()
            with self.assertRaises(CuroboUnavailableError):
                client.check_joints(_still(4))
            self.assertLess(time.monotonic() - started, _TIMEOUT_S * 4)

    def test_a_dead_sidecar_raises_typed_on_every_call(self) -> None:
        outcomes: list[str] = []
        with _StubbedClient(_DIES) as client:
            for _ in range(2):
                try:
                    client.check_joints(_still(4))
                    outcomes.append("returned")
                except CuroboUnavailableError:
                    outcomes.append("typed")
                except OSError as exc:
                    outcomes.append(f"raw {type(exc).__name__}")
                time.sleep(0.3)
        self.assertEqual(["typed", "typed"], outcomes)


class ALateCheckReplyIsDroppedTests(unittest.TestCase):

    def _second_verdict(self) -> Any:
        with _StubbedClient(_SLOW_FIRST) as client:
            with self.assertRaises(CuroboUnavailableError):
                client.check_joints(_still(4))
            time.sleep(_TIMEOUT_S)  # let the late answer land
            return client.check_joints(_still(4))

    def test_a_late_check_reply_is_dropped(self) -> None:
        verdict = self._second_verdict()
        self.assertTrue(verdict.valid, "the previous request's refusal was returned for this path")

    def test_a_read_that_takes_whatever_is_next_returns_the_late_refusal(self) -> None:
        """The control: on this stub, a read that ignores the id does hand back the stale verdict."""
        def take_whatever_is_next(
            self: CuroboPlanClient, timeout_s: float, *, want: int | None = None,
        ) -> dict[str, Any] | None:
            try:
                return self._q.get(timeout=timeout_s)
            except queue.Empty:
                return None

        saved = CuroboPlanClient._recv
        CuroboPlanClient._recv = take_whatever_is_next  # type: ignore[method-assign]
        try:
            verdict = self._second_verdict()
        finally:
            CuroboPlanClient._recv = saved  # type: ignore[method-assign]
        self.assertEqual((False, 1), (verdict.valid, verdict.first_invalid))


class ALongPathIsSentInBatchesTests(unittest.TestCase):
    """A path longer than one request is split across requests, never thinned and never refused.

    ⛔ IT USED TO BE REFUSED, AND AN ISAAC RUN SHOWED WHAT THAT COSTS. The cap of 1000 is how many
    configurations fit in ONE `check_js` request; it was applied to the whole move as though it were a
    rule about motions. Measured 2026-09-12 on the M2 gate: a real cuRobo plan of 121 waypoints needed
    more than 1000 samples and was refused, about one pick in ten. The owner chose to split the
    request instead (2026-09-12), so nothing is thinned, no gap appears, and the only cost is time.

    The index a caller gets back is the index in the WHOLE path. A verdict that named sample 3 of the
    fourth batch would point at a configuration nobody can find.
    """

    def test_a_path_over_the_cap_is_sent_in_batches(self) -> None:
        with _StubbedClient(_JUDGES) as client:
            verdict = client.check_joints(_still(2500))
        self.assertTrue(verdict.valid)
        self.assertEqual(2500, verdict.checked)
        self.assertEqual(3, client._next_id, "2500 samples did not take three requests of 1000")

    @staticmethod
    def _bad_at(index: int, count: int = 2500) -> list[list[float]]:
        """A long path with exactly one configuration the judging stub refuses."""
        path = _still(count)
        path[index] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return path

    def test_the_refused_index_is_the_index_in_the_whole_path(self) -> None:
        """The one bad sample sits in the SECOND batch, at 200 inside it and 1200 in the path."""
        with _StubbedClient(_JUDGES) as client:
            verdict = client.check_joints(self._bad_at(1200))
        self.assertFalse(verdict.valid)
        self.assertEqual(1200, verdict.first_invalid)
        self.assertIn("1200", verdict.reason)

    def test_the_control_the_same_sample_in_the_first_batch(self) -> None:
        """Without this, an offset that was never added would still pass the test above by luck."""
        with _StubbedClient(_JUDGES) as client:
            verdict = client.check_joints(self._bad_at(200))
        self.assertEqual(200, verdict.first_invalid)

    def test_a_refusal_stops_the_remaining_batches(self) -> None:
        """No point asking about the rest of a path the arm is not going to take."""
        with _StubbedClient(_JUDGES) as client:
            client.check_joints(self._bad_at(1200))
            self.assertEqual(2, client._next_id, "the third batch was sent after a refusal")

    def test_exactly_the_cap_is_still_one_request(self) -> None:
        with _StubbedClient(_JUDGES) as client:
            verdict = client.check_joints(_still(1000))
        self.assertEqual(1, client._next_id)
        self.assertEqual((True, 1000), (verdict.valid, verdict.checked))


class ThePackageNamesWhatItOffersTests(unittest.TestCase):
    """The batch check is part of the planning package, not only of the module that implements it.

    Every other name a caller needs from `curobo_client` is re-exported by the package: the client
    itself, its unavailability error and the availability probe. A name reachable only through the
    module is one a reader has to know the file layout to find.
    """

    def test_the_verdict_and_the_cap_are_reachable_from_the_package(self) -> None:
        import src.robot.safety.planning as planning

        for name in ("JointCheckVerdict", "MAX_CHECK_CONFIGURATIONS"):
            with self.subTest(name=name):
                self.assertIs(
                    getattr(planning, name, None),
                    getattr(client_module, name),
                    "the package does not offer what the module defines",
                )
                self.assertIn(name, planning.__all__, "the name is not in __all__")

    def test_the_control_the_names_that_were_always_offered(self) -> None:
        """Without this the test above could pass on a package that re-exports nothing at all."""
        import src.robot.safety.planning as planning

        self.assertIs(planning.CuroboPlanClient, client_module.CuroboPlanClient)
        self.assertIn("CuroboPlanClient", planning.__all__)


if __name__ == "__main__":
    unittest.main()
