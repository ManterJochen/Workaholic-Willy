"""A cuRobo refusal reaches the Willy side as a typed reason, and never as a changed verdict (B1, S15).

Today a refused plan is a warning line in a log and a ``None`` the caller turns into "no motion". The pair of links
that touched, the depth, and whether it was the START of the move or its GOAL are all inside the sidecar. These tests
hold the client to carrying that out:

* a sidecar that refuses its own ``default_q`` raises ``CuroboNotReady``, which carries the refusal;
* a refused start or goal sets ``last_refusal`` and still returns ``None``, so no driver status changes;
* a refusal that cannot name a pair is a refusal all the same. A stock descriptor resolves its spheres from a file and
  has no per link ownership, so an unnamed pair must cost the operator a NAME and never a planner;
* a block this client cannot parse is not silently half read: the parser raises, and the plan path keeps its ``None``;
* the measure only sidecar, which reports a refusal instead of exiting, is refused by every client that did not ask
  for it. No driver asks.

The stubs are real subprocesses, as in test_curobo_batch_check.py: the client's half of this lives in a pipe, a reader
thread and a queue. The sidecar's half needs cuRobo and a GPU and is pinned by source scans in S16.
"""

from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any

import src.robot.safety.planning.curobo_client as client_module
from src.contracts import UNSET, chosen
from src.robot.safety.planning._curobo_protocol import ENV_MEASURE_ONLY
from src.robot.safety.planning.curobo_client import (
    CuroboNotReady,
    CuroboPlanClient,
    CuroboUnavailableError,
    StateRefusal,
    StateRefusalKind,
    StateWhere,
)

_TIMEOUT_S = 0.8

_ROOT = Path(__file__).resolve().parents[1]

_READY: dict[str, Any] = {"status": "ready", "joint_names": ["j0", "j1", "j2", "j3", "j4", "j5"], "dt": 0.02}

#: A refusal as the sidecar writes it: the pair sorted, the depth in millimetres, the configuration it judged.
_TOUCHING: dict[str, Any] = {
    "where": "default_q", "kind": "self_collision", "pair": ["wrist_1_link", "hand"], "depth_mm": 12.2,
    "joints": [0.0, -1.57, 1.57, 0.0, 0.0, 0.0],
}

_TRAJECTORY: dict[str, Any] = {"success": True, "trajectory": [[0.0] * 6, [0.1] * 6], "dt": 0.02}


def _refused(where: str, **changes: Any) -> dict[str, Any]:
    """A plan reply that is a VERDICT, not a failure: no plan, and the reason typed."""
    refusal = dict(_TOUCHING, where=where, **changes)
    return {"success": False, "planner_error": False, "reason": "refused before planning", "refusal": refusal}


def _sidecar(first: dict[str, Any], replies: "list[dict[str, Any]] | None" = None) -> str:
    """A stub that opens with ``first`` and answers requests from ``replies``, repeating the last one."""
    source = textwrap.dedent(f"""
        import json, sys
        FIRST = json.loads({json.dumps(first)!r})
        REPLIES = json.loads({json.dumps(replies or [{"success": False, "reason": "not modelled"}])!r})
        sys.stdout.write(json.dumps(FIRST) + chr(10)); sys.stdout.flush()
        n = 0
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            req = json.loads(line)
            if req.get("cmd") == "shutdown":
                break
            reply = dict(REPLIES[min(n, len(REPLIES) - 1)])
            n += 1
            if req.get("id") is not None:
                reply["id"] = req["id"]
            sys.stdout.write(json.dumps(reply) + chr(10)); sys.stdout.flush()
    """)
    path = Path(tempfile.mkdtemp()) / "sidecar.py"
    path.write_text(source, encoding="utf-8")
    return str(path)


class _Stubbed:
    """A client wired to a stub sidecar, with the reply timeout shortened for the duration."""

    def __init__(self, first: dict[str, Any], replies: "list[dict[str, Any]] | None" = None, **kwargs: Any) -> None:
        self._script, self._kwargs = _sidecar(first, replies), kwargs
        self._saved = client_module._PLAN_TIMEOUT_S  # noqa: SLF001

    def __enter__(self) -> CuroboPlanClient:
        client_module._PLAN_TIMEOUT_S = _TIMEOUT_S  # noqa: SLF001
        self._client = CuroboPlanClient(
            python_path=sys.executable, server_script=self._script,
            robot_config="unused-by-the-stub", scene_config=None, **self._kwargs)
        return self._client

    def __exit__(self, *exc: object) -> None:
        client_module._PLAN_TIMEOUT_S = self._saved  # noqa: SLF001
        try:
            self._client.close()
        except Exception:  # teardown is best effort
            pass


class TheSidecarRefusesItsOwnRetractTests(unittest.TestCase):
    def test_a_ready_refusal_names_the_pair_and_the_depth(self) -> None:
        error = {"status": "error", "reason": "the retract is in self collision", "refusal": _TOUCHING}
        with _Stubbed(error) as client, self.assertRaises(CuroboNotReady) as caught:
            client.start()

        refusal = caught.exception.refusal
        self.assertEqual((refusal.link_a, refusal.link_b), ("wrist_1_link", "hand"))
        self.assertEqual(refusal.where, StateWhere.DEFAULT_Q)
        self.assertEqual(refusal.kind, StateRefusalKind.SELF_COLLISION)
        self.assertAlmostEqual(refusal.depth_mm, 12.2, places=6)
        for expected in ("wrist_1_link", "hand", "12.2 mm"):
            self.assertIn(expected, refusal.render())
        self.assertIn("12.2 mm", str(caught.exception))

    def test_an_error_with_no_refusal_stays_the_plain_unavailable_error(self) -> None:
        """⭐ THE CONTROL. Most ways a sidecar fails carry no refusal, and a typed one must not be invented."""
        with _Stubbed({"status": "error", "reason": "CUDA is not available"}) as client:
            with self.assertRaises(CuroboUnavailableError) as caught:
                client.start()
            self.assertNotIsInstance(caught.exception, CuroboNotReady)
            self.assertIn("CUDA", str(caught.exception))

    def test_a_refusal_that_cannot_name_a_pair_is_a_refusal_all_the_same(self) -> None:
        """A stock descriptor resolves its spheres from a file, so nothing can say which link owns which sphere.

        That costs the operator the NAME of the pair. It must not cost them the refusal, and it must not cost them a
        planner either: the same descriptor plans as it always did.
        """
        unnamed = {key: value for key, value in _TOUCHING.items() if key != "pair"}
        error = {"status": "error", "reason": "the retract is in self collision", "refusal": unnamed}
        with _Stubbed(error) as client, self.assertRaises(CuroboNotReady) as caught:
            client.start()

        refusal = caught.exception.refusal
        self.assertFalse(chosen(refusal.link_a))
        self.assertFalse(chosen(refusal.link_b))
        self.assertEqual(refusal.kind, StateRefusalKind.SELF_COLLISION)
        self.assertIn("unnamed", refusal.render())
        self.assertIn("12.2 mm", refusal.render())
        self.assertIsNone(refusal.to_dict()["link_a"])


class ThePlanCarriesTheReasonTests(unittest.TestCase):
    def test_a_refused_start_is_readable_and_the_next_plan_clears_it(self) -> None:
        with _Stubbed(_READY, [_refused("start"), _TRAJECTORY]) as client:
            self.assertIsNone(client.plan([0.0] * 6, [0.4, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0]))
            refusal = client.last_refusal
            assert refusal is not None
            self.assertEqual(refusal.where, StateWhere.START)
            self.assertEqual(refusal.joints, (0.0, -1.57, 1.57, 0.0, 0.0, 0.0))

            self.assertIsNotNone(client.plan([0.0] * 6, [0.4, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0]))
            self.assertIsNone(client.last_refusal)

    def test_a_refused_goal_reaches_the_joint_plan_too(self) -> None:
        with _Stubbed(_READY, [_refused("goal")]) as client:
            self.assertIsNone(client.plan_joint([0.0] * 6, [0.1] * 6))
            assert client.last_refusal is not None
            self.assertEqual(client.last_refusal.where, StateWhere.GOAL)

    def test_a_block_this_client_cannot_read_leaves_the_refused_plan_exactly_as_it_was(self) -> None:
        """⭐ THE CONTROL that the reason is a DIAGNOSTIC. The parser is strict, so a word it does not know raises.

        At a plan, though, the verdict is already 'no collision free plan': raising there would turn a UR TIMEOUT
        into a CONTROLLER_REJECTED over the spelling of a field nobody moves on.
        """
        with self.assertRaises(CuroboUnavailableError):
            StateRefusal.from_reply(dict(_TOUCHING, where="elbow"))

        with _Stubbed(_READY, [_refused("elbow")]) as client, self.assertLogs("CuroboPlanClient", "WARNING") as logs:
            self.assertIsNone(client.plan_joint([0.0] * 6, [0.1] * 6))
        self.assertIsNone(client.last_refusal)
        self.assertTrue(any("elbow" in line for line in logs.output), logs.output)

    def test_a_checked_path_carries_the_refusal_of_the_sample_it_names(self) -> None:
        judged = {"success": True, "valid": False, "first_invalid": 1, "checked": 2,
                  "refusal": dict(_TOUCHING, where="path")}
        with _Stubbed(_READY, [judged]) as client:
            client.start()
            verdict = client.check_joints([[0.0] * 6, [0.1] * 6])
        self.assertEqual(verdict.first_invalid, 1)
        assert verdict.refusal is not None
        self.assertEqual(verdict.refusal.link_b, "hand")
        # Found porting to dev, 2026-09-25: a refused sample read "the start of this move" wherever it sat on the path.
        self.assertIs(StateWhere.PATH, verdict.refusal.where)
        self.assertIn("cuRobo refuses a configuration of the path it was asked to judge", verdict.refusal.render())
        self.assertNotIn("start", verdict.refusal.render())

        passing = {"success": True, "valid": True, "first_invalid": None, "checked": 2}
        with _Stubbed(_READY, [passing]) as client:
            client.start()
            self.assertIsNone(client.check_joints([[0.0] * 6, [0.1] * 6]).refusal)


class TheMeasureOnlySidecarIsRefusedTests(unittest.TestCase):
    def test_a_sidecar_that_cannot_plan_is_refused_by_a_client_that_wants_to_plan(self) -> None:
        ready = dict(_READY, measure_only=True, refusal=_TOUCHING)
        with _Stubbed(ready) as client:
            with self.assertRaises(CuroboUnavailableError) as caught:
                client.start()
            self.assertNotIsInstance(caught.exception, CuroboNotReady)
            self.assertIn(ENV_MEASURE_ONLY, str(caught.exception))

    def test_the_gate_asks_for_it_and_then_reads_the_refusal_that_would_have_stopped_a_driver(self) -> None:
        ready = dict(_READY, measure_only=True, refusal=_TOUCHING)
        with _Stubbed(ready, measure_only=True) as client:
            client.start()
            self.assertEqual(client.joint_names, _READY["joint_names"])
            assert client.last_refusal is not None
            self.assertEqual(client.last_refusal.where, StateWhere.DEFAULT_Q)

    def test_the_flag_is_written_into_the_sidecar_environment_and_a_stale_one_is_not_inherited(self) -> None:
        asked = CuroboPlanClient(measure_only=True)._sidecar_env()  # noqa: SLF001
        self.assertEqual(asked[ENV_MEASURE_ONLY], "1")

        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {ENV_MEASURE_ONLY: "1"}):
            self.assertNotIn(ENV_MEASURE_ONLY, CuroboPlanClient()._sidecar_env())  # noqa: SLF001


#: Answers explain_js the way the sidecar will: one row per configuration it was sent, a collision on every row whose
#: first joint reaches 1.0 rad. Answering from the rows received is what lets a batched call be counted here.
_EXPLAINS = textwrap.dedent("""
    import json, sys
    sys.stdout.write(json.dumps({"status": "ready", "joint_names": ["j%d" % i for i in range(6)], "dt": 0.02}) + chr(10))
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("cmd") == "shutdown":
            break
        rows = req["joints"]
        hits = [row[0] >= 1.0 for row in rows]
        named = req.get("name_pairs", True)
        reply = {
            "success": True,
            "self_collides": hits,
            "bound_ok": [True] * len(rows),
            "pairs": ([["hand", "wrist_1_link"] if hit else None for hit in hits] if named else None),
            "depths_mm": ([12.2 if hit else None for hit in hits] if named else None),
            "asked_for_pairs": bool(named),
        }
        if req.get("id") is not None:
            reply["id"] = req["id"]
        sys.stdout.write(json.dumps(reply) + chr(10)); sys.stdout.flush()
""")


class TheGateAsksWhyTests(unittest.TestCase):
    """``explain_joints`` is the matrix gate's question: judge these poses and say what touched, sample by sample.

    It exists so the gate measures through the SIDECAR, loading the config exactly as a cell's planner loads it,
    instead of reimplementing cuRobo's own arithmetic beside it.
    """

    def _client(self) -> CuroboPlanClient:
        path = Path(tempfile.mkdtemp()) / "sidecar.py"
        path.write_text(_EXPLAINS, encoding="utf-8")
        client = CuroboPlanClient(python_path=sys.executable, server_script=str(path),
                                  robot_config="unused-by-the-stub", scene_config=None, measure_only=True)
        self.addCleanup(client.close)
        return client

    def test_every_sample_is_judged_and_the_colliding_ones_are_named(self) -> None:
        client = self._client()
        explained = client.explain_joints([[0.0] * 6, [1.0] + [0.0] * 5])

        self.assertEqual(explained.self_collides, (False, True))
        self.assertEqual(explained.bound_ok, (True, True))
        self.assertEqual(explained.pairs, (None, ("hand", "wrist_1_link")))
        self.assertEqual(explained.depths_mm, (None, 12.2))
        self.assertEqual(len(explained), 2)
        self.assertIn("1 of 2", explained.render())
        self.assertEqual(explained.to_dict()["self_collides"], [False, True])

    def test_more_samples_than_one_request_holds_come_back_in_order(self) -> None:
        """⭐ The gate sends 1,482 poses. Batched, the answer is only usable if the batches are joined in order."""
        rows = [[0.0] * 6 for _ in range(1200)]
        rows[0] = [1.0] + [0.0] * 5
        rows[1100] = [1.0] + [0.0] * 5

        explained = self._client().explain_joints(rows)

        self.assertEqual(len(explained), 1200)
        self.assertEqual([i for i, hit in enumerate(explained.self_collides) if hit], [0, 1100])

    def test_a_caller_that_only_wants_the_verdict_does_not_pay_for_the_names(self) -> None:
        """⭐ THE ONE THAT COST AN AFTERNOON. Naming the deepest pair is pairwise arithmetic over every sphere
        of the robot, in numpy on the CPU, per configuration. Measured 2026-09-16 on a ur5 with the EGU-50,
        590 spheres: 173,755 pairs and 8.29 ms PER POSE, which is 51 minutes over one candidate family of the
        retract rule. That rule reads `self_collides` and `bound_ok` and throws the names away.

        So the question carries whether the names are wanted, and the default stays yes, because the gate
        that asked first wants them.
        """
        explained = self._client().explain_joints([[0.0] * 6, [1.0] + [0.0] * 5], name_pairs=False)

        self.assertEqual(explained.self_collides, (False, True))
        self.assertEqual(explained.bound_ok, (True, True))
        self.assertFalse(chosen(explained.pairs), explained.render())
        self.assertFalse(chosen(explained.depths_mm))

    def test_not_asked_never_reads_as_nothing_overlapped(self) -> None:
        """⛔ THE TRAP THIS AVOIDS. A per-entry None is a real answer here: "this pose collides and no link
        owns the sphere", or "this pose is clear". If not-asked were also None, an evidence file would record
        a whole family of poses as overlapping NOTHING, which is a sentence about the robot that nobody
        measured. So it is UNSET, and everything that reports says which of the two it is.
        """
        asked = self._client().explain_joints([[1.0] + [0.0] * 5])
        not_asked = self._client().explain_joints([[1.0] + [0.0] * 5], name_pairs=False)

        self.assertEqual(asked.pairs, (("hand", "wrist_1_link"),))
        self.assertIs(not_asked.pairs, UNSET)
        self.assertIn("not asked", not_asked.render())
        self.assertIs(not_asked.to_dict()["pairs"], None)
        self.assertFalse(not_asked.to_dict()["pairs_named"])
        self.assertTrue(asked.to_dict()["pairs_named"])

    def test_two_answers_that_were_not_asked_the_same_question_are_not_joined(self) -> None:
        """A batch is joined in order, and joining a named half onto an unnamed one would invent names for the
        half that has none, or drop the half that has them."""
        asked = self._client().explain_joints([[1.0] + [0.0] * 5])
        not_asked = self._client().explain_joints([[1.0] + [0.0] * 5], name_pairs=False)

        with self.assertRaises(ValueError):
            asked.joined(not_asked)
        with self.assertRaises(ValueError):
            not_asked.joined(asked)

    def test_a_long_unnamed_question_still_comes_back_in_order(self) -> None:
        """The batching has to hold either way: the retract rule asks about 4,096 poses at a time."""
        rows = [[0.0] * 6 for _ in range(1200)]
        rows[0] = [1.0] + [0.0] * 5
        rows[1100] = [1.0] + [0.0] * 5

        explained = self._client().explain_joints(rows, name_pairs=False)

        self.assertEqual(len(explained), 1200)
        self.assertEqual([i for i, hit in enumerate(explained.self_collides) if hit], [0, 1100])
        self.assertIs(explained.pairs, UNSET)

    def test_an_answer_that_does_not_account_for_every_sample_is_not_an_answer(self) -> None:
        """⭐ THE CONTROL, the rule check_js already follows: a partial judgement is never returned as a judgement."""
        short = {"success": True, "self_collides": [False], "bound_ok": [True, True],
                 "pairs": [None, None], "depths_mm": [None, None]}
        with _Stubbed(_READY, [short], measure_only=True) as client, self.assertRaises(CuroboUnavailableError):
            client.explain_joints([[0.0] * 6, [0.1] * 6])

        failed = {"success": False, "planner_error": True, "reason": "no checker was built"}
        with _Stubbed(_READY, [failed], measure_only=True) as client, self.assertRaises(CuroboUnavailableError):
            client.explain_joints([[0.0] * 6])


def _reports_the_pair(source: str) -> bool:
    """True when the sim's no-plan branch reads the client's refusal before it returns the fail safe result."""
    opening = source.find('"cuRobo found NO plan:')
    closing = source.find("NO_PLAN_FAIL_SAFE_MESSAGE", opening)
    if opening < 0 or closing < 0:
        return False
    branch = source[opening:closing]
    return "last_refusal" in branch and ".render()" in branch


class TheSimReportsItTooTests(unittest.TestCase):
    """The sim arm's own log, which needs Isaac to reach, so this is a scan with a control that can fail.

    The client logs the refusal into ITS log file. An operator debugging a sim pick reads the sim's, which is why the
    branch that reports "no plan" reads it too.
    """

    def test_the_sim_no_plan_branch_reads_the_refusal(self) -> None:
        source = (_ROOT / "src" / "robot" / "drivers" / "sim" / "arm.py").read_text(encoding="utf-8")
        self.assertTrue(_reports_the_pair(source), "the sim reports no plan without reading the refusal")

    def test_the_scan_can_fail(self) -> None:
        """⭐ THE CONTROL, in the shape test_the_order_scan_can_fail uses: the tree before this step must read False."""
        before = '_LOGGER.warning("cuRobo found NO plan: %s", x)' + chr(10) + "NO_PLAN_FAIL_SAFE_MESSAGE"
        self.assertFalse(_reports_the_pair(before))
        self.assertFalse(_reports_the_pair("nothing of the kind here"))


class TheMeasureOnlyFlagReachesNoDriverTests(unittest.TestCase):
    def test_no_driver_asks_to_measure(self) -> None:
        """⭐ THE CONTROL. A planner that refuses to plan is not a planner: only the gate may ask for one."""
        for name in ("drivers/ur/arm.py", "drivers/ur/curobo_motion.py", "drivers/sim/arm.py"):
            source = (_ROOT / "src" / "robot" / name).read_text(encoding="utf-8")
            self.assertNotIn("measure_only", source, name)


if __name__ == "__main__":
    unittest.main()
