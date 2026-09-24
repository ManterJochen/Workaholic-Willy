"""The sidecar plans the same request the same way every time, never through the retract, and judges a line with room.

Three rules of how the cuRobo sidecar plans and judges (``_curobo_plan_policy``), each pinned without a GPU:

* cuRobo's shipped graph planner (``graph_planner/exact_graph_planner.yml``) sets ``use_default_position_heuristic``,
  which links the start and the goal of every roadmap it seeds through the descriptor's retract. From the second
  attempt of a plan on, every plan was offered the route start, retract, goal; a UR10's retract is a quarter turn of
  the base from most of the bench. The sidecar builds its planner on a copy of that config with the link switched off,
  and refuses to start on a config that no longer has the key rather than guess.
* The seed is reset before every plan, so the same request in the same world gets the same plan: a move a person
  watched once is the move it makes again. It was reset once, in warm up.
* ``check_js`` passes a configuration whose spheres do not penetrate the world, which is right for a path cuRobo shaped.
  A straight joint line nobody shaped is judged at a clearance the request names (``clearance_m``), the reply says the
  clearance it judged at, and the client refuses a reply that does not, because an older sidecar judges at none and
  would pass the line that grazes an obstacle only the camera saw.

The sidecar runs under the cuRobo interpreter with a GPU, so its half is read as source here, scoped to the block each
rule lives in; the policy module is imported on both sides and tested as it is; the client meets stub sidecars. The GPU
half (that the planner accepts the dict and that a clearance query refuses a 5 mm graze) is for the bench.
"""

from __future__ import annotations

import copy
import json
import math
import textwrap
import unittest
from pathlib import Path
from typing import Any

import yaml

from src.robot.safety.planning import CuroboUnavailableError, StateRefusal
from src.robot.safety.planning._curobo_plan_policy import (
    CLEARANCE_KEY,
    GRAPH_PLANNER_CONFIG,
    MAX_CLEARANCE_M,
    graph_planner_without_retract,
    requested_clearance_m,
)
from tests.test_curobo_batch_check import _PREAMBLE, _StubbedClient, _still
from tests.test_curobo_client_error_vs_verdict import _block

_ROOT = Path(__file__).resolve().parents[1]
_SERVER = _ROOT / "src" / "robot" / "safety" / "planning" / "curobo_planner_server.py"
#: cuRobo's own graph planner config in the pinned checkout, where a tree carries one (the owner's does).
_PINNED = _ROOT / "ext_deps" / "curobo" / "curobo" / "content" / "configs" / "task" / GRAPH_PLANNER_CONFIG

#: The shipped config as the pinned cuRobo carries it, restated so the rule is tested where no checkout is.
_SHIPPED: dict[str, Any] = {"graph_planner": {
    "max_nodes": 20000, "steer_buffer_size": 5000, "cspace_similarity_threshold": 0.005,
    "sample_rejection_ratio": 10, "neighbors_per_node": 10, "feasibility_buffer_size": 2000,
    "new_nodes_per_iteration": 20, "max_path_finding_iterations": 10, "min_finetune_iterations": 2,
    "use_default_position_heuristic": True, "exploration_radius": 1.05, "exploration_radius_growth_factor": 1.05,
    "sampler_seed": 0, "sampler_buffer_size": 2000, "connect_terminal_nodes_with_nearest": False,
    "ellipsoid_projection_method": "householder", "neighbors_per_node_growth_factor": 1.05,
    "new_nodes_per_iteration_growth_factor": 1.05,
}}


class TheGraphPlannerTests(unittest.TestCase):
    def test_the_retract_link_is_switched_off_and_nothing_else_changes(self) -> None:
        shipped = copy.deepcopy(_SHIPPED)
        turned = graph_planner_without_retract(shipped)
        self.assertIs(False, turned["graph_planner"]["use_default_position_heuristic"])
        expected = copy.deepcopy(_SHIPPED)
        expected["graph_planner"]["use_default_position_heuristic"] = False
        self.assertEqual(expected, turned)
        self.assertEqual(_SHIPPED, shipped, "the config it was handed was changed")

    def test_a_config_without_the_key_is_refused_rather_than_guessed(self) -> None:
        for config in ({}, {"graph_planner": {}}, {"graph_planner": None}, [], {"planner": _SHIPPED["graph_planner"]}):
            with self.subTest(config=config), self.assertRaises(ValueError) as caught:
                graph_planner_without_retract(config)  # type: ignore[arg-type]
            self.assertIn("use_default_position_heuristic", str(caught.exception))

    def test_the_pinned_cuRobo_still_carries_the_key_and_the_default_this_turns_off(self) -> None:
        if not _PINNED.is_file():
            self.skipTest(f"no pinned cuRobo checkout at {_PINNED}")
        shipped = yaml.safe_load(_PINNED.read_text(encoding="utf-8"))
        self.assertIs(True, shipped["graph_planner"]["use_default_position_heuristic"])
        self.assertEqual(_SHIPPED, shipped, "the restated config drifted from the pinned one")


class TheClearanceRequestTests(unittest.TestCase):
    def test_no_clearance_named_is_todays_check(self) -> None:
        self.assertEqual(0.0, requested_clearance_m({"cmd": "check_js", "joints": []}))
        self.assertEqual(0.0, requested_clearance_m({CLEARANCE_KEY: None}))

    def test_a_clearance_in_range_is_read_as_metres(self) -> None:
        self.assertEqual(0.01, requested_clearance_m({CLEARANCE_KEY: 0.01}))
        self.assertEqual(MAX_CLEARANCE_M, requested_clearance_m({CLEARANCE_KEY: MAX_CLEARANCE_M}))

    def test_anything_else_fails_the_call_rather_than_judging_at_another_clearance(self) -> None:
        for value in (True, "0.01", -0.001, MAX_CLEARANCE_M + 1e-6, math.nan, math.inf, [0.01]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                requested_clearance_m({CLEARANCE_KEY: value})


class TheSidecarSourceTests(unittest.TestCase):
    """Scoped scans of the sidecar, each with the block it reads; the control below shows a scan can fail."""

    def setUp(self) -> None:
        self.source = _SERVER.read_text(encoding="utf-8")
        self.plan_js = "\n".join(_block(self.source, 'if cmd == "plan_js":'))
        self.check_js = "\n".join(_block(self.source, 'if cmd == "check_js":'))

    def test_the_planner_is_built_on_the_graph_config_without_the_retract(self) -> None:
        self.assertIn("from _curobo_plan_policy import", self.source)
        built = self.source[self.source.index("_graph = graph_planner_without_retract("):
                            self.source.index("_planner.warmup(")]
        self.assertIn("GRAPH_PLANNER_CONFIG", built)
        self.assertIn("MotionPlannerCfg.create(", built)
        self.assertIn("graph_planner_config=_graph", built)

    def test_the_seed_is_reset_before_every_joint_plan(self) -> None:
        self.assertTrue(self.plan_js, "the plan_js branch is gone")
        reset, plan = self.plan_js.find("_planner.reset_seed()"), self.plan_js.find("_planner.plan_cspace(")
        self.assertTrue(0 <= reset < plan, "plan_js plans without resetting the seed first")

    def test_check_js_reads_the_clearance_and_says_the_one_it_judged_at(self) -> None:
        self.assertTrue(self.check_js, "the check_js branch is gone")
        self.assertIn("requested_clearance_m(req)", self.check_js)
        self.assertIn("CLEARANCE_KEY: _clearance", self.check_js)
        self.assertIn("clearance_m=_clearance", self.check_js, "the named refusal is judged at another clearance")

    def test_the_scoped_scans_can_fail(self) -> None:
        """The control: the names present in a source, and absent from the block they are scanned in."""
        synthetic = textwrap.dedent('''
            _planner.reset_seed()
            requested_clearance_m(req)
            for line in stdin:
                if cmd == "plan_js":
                    result = _planner.plan_cspace(goal, start)
                    continue
        ''')
        block = "\\n".join(_block(synthetic, 'if cmd == "plan_js":'))
        self.assertTrue(block)
        self.assertIn("_planner.reset_seed()", synthetic)
        self.assertNotIn("_planner.reset_seed()", block)


#: Judges like a sidecar from this tree: echoes the clearance it was asked, refuses a sample whose first joint reaches
#: 1.0 rad less the clearance, and names that refusal as a clearance missed.
_WITH_CLEARANCE = _PREAMBLE + textwrap.dedent("""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("cmd") == "shutdown":
            break
        clearance = req.get("clearance_m", 0.0)
        with open(ASKED, "a") as seen:
            seen.write(json.dumps(sorted(req)) + chr(10))
        flags = [row[0] < 1.0 - clearance for row in req["joints"]]
        first = next((i for i, ok in enumerate(flags) if not ok), None)
        reply = {"success": True, "valid": first is None, "first_invalid": first, "checked": len(flags),
                 "clearance_m": clearance}
        if first is not None and clearance > 0.0:
            reply["refusal"] = {"where": "start", "kind": "world", "joints": req["joints"][first],
                                "clearance_mm": clearance * 1000.0}
        emit(reply, req.get("id"))
""")

#: A sidecar from before the clearance: judges at no penetration and says nothing about a clearance.
_BEFORE_CLEARANCE = _PREAMBLE + textwrap.dedent("""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("cmd") == "shutdown":
            break
        emit({"success": True, "valid": True, "first_invalid": None, "checked": len(req["joints"])}, req.get("id"))
""")


class TheClientAsksForTheClearanceTests(unittest.TestCase):
    def _recording(self) -> "tuple[str, Path]":
        asked = Path(self.enterContext(__import__("tempfile").TemporaryDirectory())) / "asked.jsonl"
        return _WITH_CLEARANCE.replace("ASKED", json.dumps(str(asked))), asked

    def test_a_clearance_is_sent_in_metres_and_judged_at(self) -> None:
        source, asked = self._recording()
        path = _still(4)
        path[2][0] = 0.995  # 5 mm short of the stub's wall in its own units: clear at none, too close at 10 mm
        with _StubbedClient(source) as client:
            at_none = client.check_joints(path)
            at_ten = client.check_joints(path, clearance_mm=10.0)
        self.assertTrue(at_none.valid)
        self.assertFalse(at_ten.valid)
        self.assertEqual(2, at_ten.first_invalid)
        refusal = at_ten.refusal
        assert isinstance(refusal, StateRefusal)
        self.assertEqual(10.0, refusal.clearance_mm)
        self.assertIn("comes closer than 10.0 mm to the world", refusal.render())
        keys = [json.loads(row) for row in asked.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(["cmd", "id", "joints"], keys[0], "a check at no clearance was not sent as it always was")
        self.assertEqual(["clearance_m", "cmd", "id", "joints"], keys[1])

    def test_a_sidecar_that_does_not_say_it_judged_at_the_clearance_is_refused(self) -> None:
        with _StubbedClient(_BEFORE_CLEARANCE) as client:
            self.assertTrue(client.check_joints(_still(3)).valid, "the control: at none it is answered as before")
            with self.assertRaises(CuroboUnavailableError) as caught:
                client.check_joints(_still(3), clearance_mm=10.0)
        self.assertIn("restart it", str(caught.exception))

    def test_a_clearance_the_sidecar_cannot_ask_is_refused_before_anything_is_sent(self) -> None:
        source, _ = self._recording()
        with _StubbedClient(source) as client:
            for clearance in (-1.0, MAX_CLEARANCE_M * 1000.0 + 1.0, math.nan):
                with self.subTest(clearance_mm=clearance), self.assertRaises(ValueError):
                    client.check_joints(_still(2), clearance_mm=clearance)
            self.assertEqual(0, client._next_id, "a refused request reached the sidecar")


if __name__ == "__main__":
    unittest.main()
