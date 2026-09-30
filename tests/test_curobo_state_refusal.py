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


#: Answers check_js the way the sidecar does, from the rows it receives: a row whose first joint reaches 1.0 rad is a self
#: collision of forearm_link and wrist_2_link (1.2 mm padded, 2.8 mm apart unpadded), one whose second joint reaches
#: 1.0 rad is outside the bounds, and one whose third reaches 1.0 rad is in the world. With report_refused it lists every
#: refused row; the reply echoes the keys it was sent, so a test reads exactly what was asked.
_REPORTS = textwrap.dedent("""
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
        terms = [(row[1] < 1.0, row[0] < 1.0, row[2] < 1.0) for row in rows]
        bad = [i for i, (bound, self_ok, world) in enumerate(terms) if not (bound and self_ok and world)]
        reply = {"success": True, "valid": not bad, "first_invalid": bad[0] if bad else None, "checked": len(rows),
                 "clearance_m": float(req.get("clearance_m", 0.0)), "asked": sorted(req)}
        if req.get("report_refused"):
            named = bool(req.get("name_pairs", True))
            reply["refused"] = [{
                "index": i, "bound_ok": terms[i][0], "self_ok": terms[i][1], "world_ok": terms[i][2],
                "pairs": ([["forearm_link", "wrist_2_link", 1.21, -2.79]] if not terms[i][1] else []) if named else None,
            } for i in bad]
            reply["pairs_named"] = named
        if req.get("id") is not None:
            reply["id"] = req["id"]
        sys.stdout.write(json.dumps(reply) + chr(10)); sys.stdout.flush()
""")

_IN_BAND = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_OUT_OF_BOUNDS = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
_IN_THE_WORLD = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
_CLEAR = [0.0] * 6


class TheCheckReportsEveryRefusedSampleTests(unittest.TestCase):
    """F1 (the owner, 2026-09-30): the exact guard decides the planner's self pairs it judges, where it accepted the
    configuration. The driver can only do that knowing every configuration the planner refused and every pair of links
    in each, with the three terms apart, so ``judge_joints`` asks check_js for all of them.

    Nothing about ``check_joints`` changes: a request without the key is the request it always was.
    """

    def _client(self) -> CuroboPlanClient:
        path = Path(tempfile.mkdtemp()) / "sidecar.py"
        path.write_text(_REPORTS, encoding="utf-8")
        client = CuroboPlanClient(python_path=sys.executable, server_script=str(path),
                                  robot_config="unused-by-the-stub", scene_config=None)
        self.addCleanup(client.close)
        return client

    def test_a_plain_check_is_the_request_it_always_was(self) -> None:
        """⭐ THE CONTROL: check_joints sends no report key, so every caller that did not ask is answered as before."""
        client = self._client()
        verdict = client.check_joints([_CLEAR, _IN_BAND])
        self.assertFalse(verdict.valid)
        self.assertEqual(verdict.first_invalid, 1)
        want = self._asked(client, [_CLEAR])
        self.assertEqual(want, ["cmd", "id", "joints"])

    def _asked(self, client: CuroboPlanClient, rows: "list[list[float]]") -> "list[str]":
        want = client._send({"cmd": "check_js", "joints": rows})  # noqa: SLF001 (reads the stub's echo)
        reply = client._recv(_TIMEOUT_S * 5, want=want)  # noqa: SLF001
        assert reply is not None
        return reply["asked"]

    def test_every_refused_sample_comes_back_with_its_terms_and_every_pair(self) -> None:
        judged = self._client().judge_joints([_CLEAR, _IN_BAND, _OUT_OF_BOUNDS, _CLEAR, _IN_THE_WORLD])
        self.assertFalse(judged.valid)
        self.assertEqual(judged.checked, 5)
        self.assertTrue(judged.pairs_named)
        self.assertEqual([row.index for row in judged.refused], [1, 2, 4])
        band, bound, world = judged.refused
        self.assertEqual((band.bound_ok, band.self_ok, band.world_ok), (True, False, True))
        assert chosen(band.pairs)
        (pair,) = band.pairs
        self.assertEqual((pair.link_a, pair.link_b), ("forearm_link", "wrist_2_link"))
        self.assertAlmostEqual(pair.depth_mm, 1.21, places=6)
        self.assertAlmostEqual(pair.unpadded_depth_mm, -2.79, places=6)
        self.assertEqual((bound.bound_ok, bound.self_ok), (False, True))
        self.assertEqual(bound.pairs, ())
        self.assertFalse(world.world_ok)
        self.assertIn("forearm_link", judged.render())

    def test_the_request_names_the_report_and_the_clearance(self) -> None:
        client = self._client()
        judged = client.judge_joints([_CLEAR], clearance_mm=15.0, name_pairs=False)
        self.assertTrue(judged.valid)
        self.assertEqual(judged.clearance_mm, 15.0)
        self.assertFalse(judged.pairs_named)

    def test_a_sample_whose_names_were_not_asked_carries_none(self) -> None:
        judged = self._client().judge_joints([_IN_BAND], name_pairs=False)
        (row,) = judged.refused
        self.assertFalse(row.self_ok)
        self.assertIs(row.pairs, UNSET)
        self.assertFalse(judged.pairs_named)

    def test_a_long_path_is_judged_whole_and_every_index_is_the_whole_paths(self) -> None:
        """⭐ Batched at 1000 like every check, and not stopped at the first refused batch: the driver needs them all."""
        rows = [list(_CLEAR) for _ in range(1500)]
        rows[3] = list(_IN_BAND)
        rows[1200] = list(_IN_BAND)
        rows[1499] = list(_IN_THE_WORLD)
        judged = self._client().judge_joints(rows)
        self.assertEqual(judged.checked, 1500)
        self.assertEqual([row.index for row in judged.refused], [3, 1200, 1499])

    def test_an_answer_that_is_not_a_whole_report_is_not_an_answer(self) -> None:
        """⭐ Fail closed: anything the client cannot read whole raises, and the driver's refusal then stands."""
        base = {"success": True, "valid": False, "first_invalid": 0, "checked": 1, "clearance_m": 0.0,
                "pairs_named": True}
        row = {"index": 0, "bound_ok": True, "self_ok": False, "world_ok": True,
               "pairs": [["forearm_link", "wrist_2_link", 1.21, -2.79]]}
        broken = {
            "an old sidecar": {k: v for k, v in base.items() if k != "pairs_named"},
            "no rows for a refused path": dict(base, refused=[]),
            "a row past the path": dict(base, refused=[dict(row, index=1)]),
            "a row the verdict does not open with": dict(base, first_invalid=0, checked=2,
                                                          refused=[dict(row, index=1)]),
            "the same row twice": dict(base, checked=2, refused=[row, row]),
            "a pair that is not four fields": dict(base, refused=[dict(row, pairs=[["forearm_link", 1.21]])]),
            "a depth that is no number": dict(base, refused=[dict(row, pairs=[["a", "b", "deep", -1.0]])]),
            "a term that is no bool": dict(base, refused=[dict(row, world_ok="yes")]),
            "names missing where they were named": dict(base, refused=[dict(row, pairs=None)]),
            "rows for a valid path": dict(base, valid=True, first_invalid=None, refused=[row]),
            "a row that is every term clear": dict(base, refused=[dict(row, self_ok=True, pairs=[])]),
        }
        for what, reply in broken.items():
            with self.subTest(what=what):
                sent = int(reply["checked"])
                with _Stubbed(_READY, [reply]) as client, self.assertRaises(CuroboUnavailableError):
                    client.judge_joints([_CLEAR] * sent)

    def test_names_nobody_asked_for_are_not_read(self) -> None:
        """⭐ Fail closed on anything unexpected: pairs where the question asked for none, or a row carrying pairs its
        reply says were not named, is not a report."""
        row = {"index": 0, "bound_ok": True, "self_ok": False, "world_ok": True,
               "pairs": [["forearm_link", "wrist_2_link", 1.21, -2.79]]}
        base = {"success": True, "valid": False, "first_invalid": 0, "checked": 1, "clearance_m": 0.0}
        with _Stubbed(_READY, [dict(base, pairs_named=True, refused=[row])]) as client, \
                self.assertRaises(CuroboUnavailableError):
            client.judge_joints([_IN_BAND], name_pairs=False)
        with _Stubbed(_READY, [dict(base, pairs_named=False, refused=[row])]) as client, \
                self.assertRaises(CuroboUnavailableError):
            client.judge_joints([_IN_BAND])

    def test_a_sidecar_older_than_the_report_is_refused_by_name(self) -> None:
        old = {"success": True, "valid": False, "first_invalid": 0, "checked": 1, "clearance_m": 0.0}
        with _Stubbed(_READY, [old]) as client, self.assertRaises(CuroboUnavailableError) as caught:
            client.judge_joints([_IN_BAND])
        self.assertIn("restart", str(caught.exception))

    def test_a_failed_call_is_never_a_report(self) -> None:
        failed = {"success": False, "planner_error": True, "reason": "no checker was built"}
        with _Stubbed(_READY, [failed]) as client, self.assertRaises(CuroboUnavailableError):
            client.judge_joints([_CLEAR])


def _check_block(source: str) -> str:
    """The sidecar's check_js branch, as text: from its dispatch to the next command's."""
    start = source.find('if cmd == "check_js":')
    end = source.find('if cmd == "explain_js":', start)
    return source[start:end] if 0 <= start < end else ""


def _reports_only_when_asked(block: str) -> bool:
    """True when every write of the report keys sits after the one test of the request key, each written once."""
    ask = block.find("if req.get(REPORT_REFUSED_KEY):")
    if ask < 0:
        return False
    writes = ("_reply[REFUSED_KEY]", "_reply[PAIRS_NAMED_KEY]")
    return all(block.count(key) == 1 and block.find(key) > ask for key in writes)


class TheSidecarReportsOnlyWhenAskedTests(unittest.TestCase):
    """The sidecar's half runs only in the cuRobo environment, so it is read as source (the GPU half is
    ``scripts/curobo/probe_band_admission.py``): the report joins a check_js reply only under its request key, so every
    other check is answered byte for byte as before, and its rows are ``refused_rows``, the CPU suite's own
    arithmetic (``tests/test_curobo_pair_depth.py``), whose pairs are ``overlapping_pairs``, every one."""

    def setUp(self) -> None:
        self.block = _check_block((_ROOT / "src" / "robot" / "safety" / "planning" / "curobo_planner_server.py")
                                  .read_text(encoding="utf-8"))

    def test_the_report_is_written_only_under_its_key(self) -> None:
        self.assertTrue(self.block, "the sidecar has no check_js branch")
        self.assertTrue(_reports_only_when_asked(self.block))
        self.assertIn("refused_rows(", self.block)
        self.assertIn("NAME_PAIRS_KEY", self.block)

    def test_the_scan_can_fail(self) -> None:
        """⭐ THE CONTROL: a report written whatever was asked, or twice, reads False."""
        always = '_reply[REFUSED_KEY] = []\nif req.get(REPORT_REFUSED_KEY):\n    _reply[PAIRS_NAMED_KEY] = True\n'
        self.assertFalse(_reports_only_when_asked(always))
        twice = ('if req.get(REPORT_REFUSED_KEY):\n    _reply[REFUSED_KEY] = []\n    _reply[PAIRS_NAMED_KEY] = True\n'
                 '_reply[REFUSED_KEY] = []\n')
        self.assertFalse(_reports_only_when_asked(twice))
        self.assertFalse(_reports_only_when_asked("_emit(_reply)\n"))


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
