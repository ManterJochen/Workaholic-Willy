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


def _def_block(source: str, name: str) -> list[str]:
    """The lines of the function ``name`` at whatever depth it is defined; empty when absent."""
    lines = source.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.strip().startswith(f"def {name}(")), None
    )
    if start is None:
        return []
    indent = len(lines[start]) - len(lines[start].lstrip())
    block = [lines[start]]
    for line in lines[start + 1:]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        block.append(line)
    return block


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

    def test_the_judgement_uses_the_variant_the_probe_selected(self) -> None:
        """The names moved into ``_terms`` (S16), which is the point: one judgement, asked by three callers.

        So the scan follows them there rather than being deleted. The checker itself is built once at start,
        before the judgement that uses it, because the ready gate cannot judge the retract without one. The world term
        moved one step further, into ``_world_term``, which a straight line asks at a clearance; ``_terms`` calls it,
        so the scan reads the two together.
        """
        terms = NL.join(_def_block(self.source, "_terms"))
        self.assertTrue(terms, "the sidecar has no _terms: nothing here judges a configuration")
        self.assertIn("_world_term(", terms)
        terms += NL + NL.join(_def_block(self.source, "_world_term"))
        for name in (
            "_planner.compute_kinematics(",
            ".get_bound(",
            ".get_self_collision(",
            ".get_collision_constraint(",
        ):
            with self.subTest(present=name):
                self.assertIn(name, terms)
        for name in (".validate(", "get_scene_self_collision_distance_from_joints", "_kin."):
            with self.subTest(absent=name):
                self.assertNotIn(name, terms)

        build = self.source[: self.source.find('def _terms(')]
        for name in (
            "RobotCollisionCheckerCfg.load_from_config(",
            "robot_config=copy.deepcopy(_COMPOSED)",
            "scene_collision_checker=_planner.scene_collision_checker",
            "collision_activation_distance=0.0",
        ):
            with self.subTest(built=name):
                self.assertIn(name, build)

        # The pass rule stays where the verdict is formed.
        self.assertIn("== 0.0", NL.join(self.block))

    def test_the_scoped_judgement_scan_can_fail(self) -> None:
        """The CONTROL: a source that calls the checker somewhere else leaves _terms empty of the names."""
        elsewhere = NL.join((
            "def _terms(rows):",
            "    return 0, 0, 0, None",
            "def other(rows):",
            "    return _CHECKER.get_bound(rows)",
        ))
        self.assertIn(".get_bound(", elsewhere)
        self.assertNotIn(".get_bound(", NL.join(_def_block(elsewhere, "_terms")))


_SET_WORLD_HEAD = 'if cmd == "set_world":'
_SET_VOXELS_HEAD = 'if cmd == "set_voxels":'


class TheSceneBranchesTests(unittest.TestCase):
    """The sidecar's world and field branches, read as source, each scan scoped to its own block.

    Scoped because a whole-file scan is green before any of this exists: ``"mesh"`` is already in the
    file twice (the slot reservation and the set_world branch) and ``_collision_cache["voxel"]`` once
    (the grid reservation). What was missing is each name in the branch whose behaviour it stands for:
    the field branch rebuilt the scene from the cuboids alone, set_world remembered no meshes, and
    nothing compared a field's grid with the grid the planner reserved.
    """

    def setUp(self) -> None:
        self.source = _SERVER.read_text(encoding="utf-8")
        self.set_world = "\n".join(_block(self.source, _SET_WORLD_HEAD))
        self.set_voxels = "\n".join(_block(self.source, _SET_VOXELS_HEAD))
        self.grid_check = "\n".join(_def_block(self.source, "_grid_refusal"))

    def test_the_field_branch_keeps_the_declared_meshes(self) -> None:
        self.assertTrue(self.set_voxels, "the sidecar has no set_voxels branch")
        self.assertIn("_LAST_MESHES", self.set_voxels)
        self.assertIn('"mesh"', self.set_voxels)

    def test_set_world_remembers_its_meshes_and_takes_a_field_in_the_same_request(self) -> None:
        self.assertTrue(self.set_world, "the sidecar has no set_world branch")
        self.assertIn("_LAST_MESHES =", self.set_world)
        self.assertIn('req.get("voxels")', self.set_world)
        self.assertIn('"voxels_set"', self.set_world)

    def test_a_field_on_a_grid_that_was_not_reserved_is_refused(self) -> None:
        self.assertIn('_collision_cache["voxel"]', self.grid_check)
        for label, branch in (("set_world", self.set_world), ("set_voxels", self.set_voxels)):
            with self.subTest(branch=label):
                self.assertIn("_grid_refusal(", branch)

    def test_the_scoped_scans_can_fail(self) -> None:
        """The control: every name present in a source, and absent from the block it is scanned in."""
        synthetic = (
            'scene["mesh"] = meshes\n'
            "_LAST_MESHES = {}\n"
            '_cache = _collision_cache["voxel"]\n'
            "for line in stdin:\n"
            '    if cmd == "set_voxels":\n'
            '        _planner.update_world(SceneCfg.create({"cuboid": cuboids}))\n'
            "        continue\n"
        )
        block = "\n".join(_block(synthetic, _SET_VOXELS_HEAD))
        self.assertTrue(block)
        for name in ('"mesh"', "_LAST_MESHES", '_collision_cache["voxel"]'):
            with self.subTest(name=name):
                self.assertIn(name, synthetic)
                self.assertNotIn(name, block)
        self.assertEqual(_def_block(synthetic, "_grid_refusal"), [])
        found = _def_block("try:\n    def _grid_refusal(req):\n        return ''\n    x = 1\n", "_grid_refusal")
        self.assertEqual(len(found), 2, "a helper defined inside the sidecar's try block must be found")


_PLAN_JS_HEAD = 'if cmd == "plan_js":'
_EXPLAIN_HEAD = 'if cmd == "explain_js":'
_READY_EMIT = '_emit({"status": "ready"'
NL = chr(10)


def _precedes(text: str, first: str, second: str) -> bool:
    """True when ``first`` appears in ``text`` and appears before ``second``."""
    a, b = text.find(first), text.find(second)
    return 0 <= a < b


class TheSidecarJudgesBeforeItPlansTests(unittest.TestCase):
    """Every configuration this sidecar accepts is one it judged first (B1, S16), read as source.

    The whole point of B1 is that a cell is never told "no collision free plan" when the truth is "your arm starts
    inside its own hand". The judgement has to happen BEFORE the planner is asked, and before the sidecar calls itself
    ready, or the typed reason has nothing to describe.

    Scoped scans with controls, because this file runs only in the cuRobo environment. What no scan can show is that
    the judgement agrees with cuRobo: only the box gates can.
    """

    def setUp(self) -> None:
        self.source = _SERVER.read_text(encoding="utf-8")

    def test_the_retract_is_judged_before_the_sidecar_calls_itself_ready(self) -> None:
        """The CALL on default_q, not the def: a definition alone sits before the ready emit by coincidence."""
        self.assertTrue(
            _precedes(self.source, "_judge_states([_default_q]", _READY_EMIT),
            "the sidecar reports ready without judging its own retract",
        )
        # And it exits rather than carrying on: a judgement whose refusal nothing acts on is not a gate.
        self.assertIn("sys.exit(1)", self.source[: self.source.find(_READY_EMIT)])

    def test_the_start_and_the_goal_are_judged_before_each_kind_of_plan(self) -> None:
        joint_plan = NL.join(_block(self.source, _PLAN_JS_HEAD))
        self.assertTrue(joint_plan, "the plan_js branch is gone")
        self.assertTrue(
            _precedes(joint_plan, "_judge_states(", "_planner.plan_cspace("),
            "plan_js plans without judging its start and goal",
        )
        tail = self.source[self.source.find(_EXPLAIN_HEAD):]
        self.assertTrue(
            _precedes(tail, "_judge_states(", "_planner.plan_pose("),
            "the pose plan plans without judging its start",
        )

    def test_the_sidecar_names_the_pair_from_the_config_it_loaded(self) -> None:
        self.assertIn("from _curobo_pairs import", self.source)
        self.assertIn("SphereLayout.from_robot_config(_COMPOSED)", self.source)
        self.assertIn("deepest_pairs(", self.source)

    def test_the_explain_branch_fails_in_one_line_that_is_never_a_judgement(self) -> None:
        block = _block(self.source, _EXPLAIN_HEAD)
        self.assertTrue(block, "there is no explain_js branch")
        failures = [line.strip() for line in block if '"success": False' in line]
        self.assertTrue(failures, "the explain_js branch has no failure emission")
        for line in failures:
            with self.subTest(line=line):
                self.assertIn('"planner_error": True', line)
        self.assertTrue(_ends_with_continue(block))

    def test_a_measuring_sidecar_says_so_and_refuses_to_plan(self) -> None:
        self.assertIn("ENV_MEASURE_ONLY", self.source)
        self.assertTrue(_precedes(self.source, "_MEASURE_ONLY", _READY_EMIT))
        for head in (_PLAN_JS_HEAD, 'if cmd == "attach":'):
            with self.subTest(branch=head):
                self.assertIn("_MEASURE_ONLY", NL.join(_block(self.source, head)))

    def test_these_order_scans_can_fail(self) -> None:
        """⭐ THE CONTROLS, in the shape test_the_order_scan_can_fail uses: each scan read against a source that lies."""
        defined_only = "def _judge_states(rows):" + NL + "    pass" + NL + _READY_EMIT
        self.assertFalse(_precedes(defined_only,
                                   "_judge_states([_default_q]", _READY_EMIT))
        too_late = _READY_EMIT + NL + "_judge_states([_default_q], world=False)"
        self.assertFalse(_precedes(too_late,
                                   "_judge_states([_default_q]", _READY_EMIT))
        after = NL.join((
            'if cmd == "plan_js":',
            "    _planner.plan_cspace(g, s)",
            "    _judge_states([s])",
            "    continue",
        ))
        self.assertFalse(_precedes(NL.join(_block(after, _PLAN_JS_HEAD)), "_judge_states(", "_planner.plan_cspace("))


if __name__ == "__main__":
    unittest.main()
