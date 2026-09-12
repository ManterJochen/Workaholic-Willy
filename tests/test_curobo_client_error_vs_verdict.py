"""A failed planning CALL must never reach a caller looking like a planning VERDICT.

This test exists because the confusion cost three commits and two wrong conclusions in writing.

``plan_js`` is not a method on this cuRobo build's planner — it is ``plan_cspace``. Every joint-space
request therefore raised ``AttributeError`` inside the sidecar, the sidecar's broad ``except`` reported
it as a failure, and the client mapped any failure to ``None``. ``None`` is the client's documented way
of saying *"the planner searched and found no collision-free path"*, so the arm refused to move and the
refusal read as a statement about the robot. It was investigated as one: the scan pose was called
unsafe (twice, in commits), then the planner's collision world was blamed, then the robot descriptor —
and a search began for inflated collision spheres in ``ur5e.yml``, a file that had nothing to do with
it. The tell was there from the first run and went unnoticed: the planner also refused to plan from a
configuration **to itself**.

So the two outcomes are now distinguishable by construction — the sidecar labels every failure
``planner_error: true|false`` — and this asserts the client acts on that label. It needs no GPU, no
cuRobo and no sidecar: the whole point is that the distinction lives in the protocol, not in the
planner.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.robot.safety.planning.curobo_client import (
    CuroboUnavailableError,
    _raise_if_call_failed,
)


class ErrorVersusVerdictTests(unittest.TestCase):
    def test_a_failed_call_raises(self) -> None:
        with self.assertRaises(CuroboUnavailableError) as ctx:
            _raise_if_call_failed({
                "success": False,
                "planner_error": True,
                "reason": "AttributeError: 'MotionPlanner' object has no attribute 'plan_js'",
            })
        message = str(ctx.exception)
        self.assertIn("CALL failed", message)
        self.assertIn("AttributeError", message, "the raised error must carry the actual cause")

    def test_a_genuine_no_solution_does_not_raise(self) -> None:
        """The planner searched and found nothing: a verdict the caller must be able to act on."""
        _raise_if_call_failed({
            "success": False,
            "planner_error": False,
            "reason": "no collision-free joint-space plan",
        })

    def test_an_unlabelled_failure_does_not_raise(self) -> None:
        """Back-compat: an older sidecar sends no label, and its failures stay verdicts as before."""
        _raise_if_call_failed({"success": False, "reason": "something"})

    def test_the_sidecar_labels_both_outcomes(self) -> None:
        """Read the sidecar source: every failure emission must carry the label, or the split is fiction.

        A source check rather than a live one, because running the sidecar needs a GPU and the cuRobo
        env — but a missing label silently collapses the two outcomes back into one, which is exactly
        the bug this file is about. Cheap to assert, expensive to rediscover.
        """
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "src/robot/safety/planning/curobo_planner_server.py"
        ).read_text(encoding="utf-8")
        emissions = [
            line for line in source.splitlines()
            if '"success": False' in line
        ]
        self.assertGreaterEqual(len(emissions), 2, "expected both plan paths to emit failures")
        for line in emissions:
            with self.subTest(line=line.strip()):
                self.assertIn(
                    "planner_error", line,
                    "a failure emission without planner_error makes a broken call indistinguishable "
                    "from a planning verdict -- the exact bug this test exists for",
                )

    def test_the_sidecar_calls_a_method_that_exists(self) -> None:
        """``plan_js`` was invented. Assert the name in the source is the real one.

        cuRobo is not importable in CI, so this cannot introspect the class; it pins the name that was
        verified on-box against ``MotionPlanner`` (2026-08-11, curobo 0.8.0.post1.dev42). If a future
        cuRobo renames it, this fails and points at the probe rather than at a phantom robot descriptor.
        """
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "src/robot/safety/planning/curobo_planner_server.py"
        ).read_text(encoding="utf-8")
        self.assertIn("_planner.plan_cspace(", source)
        self.assertNotIn("_planner.plan_js(", source)


_SERVER = (
    Path(__file__).resolve().parents[1]
    / "src/robot/safety/planning/curobo_planner_server.py"
)
_CHECK_HEAD = 'if cmd == "check_js":'


def _block(source: str, head: str) -> list[str]:
    """The lines of the block that opens with the line ``head``, by indentation; empty when absent."""
    lines = source.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == head), None)
    if start is None:
        return []
    indent = len(lines[start]) - len(lines[start].lstrip())
    block = [lines[start]]
    for line in lines[start + 1:]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        block.append(line)
    return block


def _check_branch_precedes_plan(source: str) -> bool:
    """True when the check_js dispatch sits before the plan call an unknown command falls into."""
    branch = source.find(_CHECK_HEAD)
    plan = source.find("_planner.plan_pose(")
    return 0 <= branch < plan


def _ends_with_continue(block: list[str]) -> bool:
    code = [line.strip() for line in block if line.strip() and not line.strip().startswith("#")]
    return bool(code) and code[-1] == "continue"


class TheBatchCheckBranchTests(unittest.TestCase):
    """The sidecar's check_js branch, read as source: it runs only in the cuRobo environment.

    The names below are the ones the on-box probe called and got answers from (probe_batch_check.py,
    cuRobo 0.8.0.post1.dev42). The probe also measured why two other ways of asking are not the
    verdict: a separately built checker's own validate never saw a payload attached to the planner,
    and the distance call is banded by its activation distance. Those two stay out of the branch.
    """

    def setUp(self) -> None:
        self.source = _SERVER.read_text(encoding="utf-8")
        self.block = _block(self.source, _CHECK_HEAD)

    def test_the_branch_is_dispatched_before_the_plan_it_would_fall_into(self) -> None:
        self.assertTrue(
            _check_branch_precedes_plan(self.source),
            "check_js is not dispatched before the plan branch, so it would be read as a pose goal")
        self.assertTrue(
            _ends_with_continue(self.block),
            "the check_js branch does not end with continue, so it would also run the plan branch")

    def test_the_order_scan_can_fail(self) -> None:
        """Self-failing controls: a branch after the plan call, no branch, a branch that falls through."""
        after = '_planner.plan_pose(goal)\nif cmd == "check_js":\n    continue\n'
        self.assertFalse(_check_branch_precedes_plan(after))
        self.assertFalse(_check_branch_precedes_plan("_planner.plan_pose(goal)\n"))
        falls = ('for x in y:\n    if cmd == "check_js":\n        _emit({})\n'
                 '    _planner.plan_pose(goal)\n')
        self.assertTrue(_check_branch_precedes_plan(falls))
        self.assertFalse(_ends_with_continue(_block(falls, _CHECK_HEAD)))

    def test_the_failure_path_is_one_line_with_planner_error(self) -> None:
        failures = [line.strip() for line in self.block if '"success": False' in line]
        self.assertTrue(failures, "the check_js branch has no failure emission")
        for line in failures:
            with self.subTest(line=line):
                self.assertIn('"planner_error": True', line)
                self.assertIn("_emit(", line)

    def test_every_reply_goes_through_emit(self) -> None:
        text = "\n".join(self.block)
        self.assertIn("_emit(", text)
        self.assertNotIn("sys.stdout", text)
        for line in self.block:
            if "print(" in line:
                with self.subTest(line=line.strip()):
                    self.assertIn("file=sys.stderr", line)

    def test_the_branch_uses_the_variant_the_probe_selected(self) -> None:
        text = "\n".join(self.block)
        for name in (
            "RobotCollisionCheckerCfg.load_from_config(",
            "robot_config=_ROBOT_IN_USE",
            "scene_collision_checker=_planner.scene_collision_checker",
            "collision_activation_distance=0.0",
            "_planner.compute_kinematics(",
            ".get_bound(",
            ".get_self_collision(",
            ".get_collision_constraint(",
            "== 0.0",
        ):
            with self.subTest(present=name):
                self.assertIn(name, text)
        for name in (".validate(", "get_scene_self_collision_distance_from_joints", "_kin."):
            with self.subTest(absent=name):
                self.assertNotIn(name, text)


if __name__ == "__main__":
    unittest.main()
