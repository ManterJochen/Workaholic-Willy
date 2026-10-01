"""A push needs no fixture box (the owner, 2026-10-01: "es werden ja eh nur Teile geschoben ... das soll optional werden").

Until then four places refused or disarmed ``nudge_target`` without a declared ``robot.grasping.recovery.fixture``: the
load, ``SceneRecoveryPolicy``, the gate the service hands a pick (``push_permitted``) and the recovery loop. The push
never needed that box to know where a part may land: the **automatic push box** bounds it, the workspace box
intersected with the table the looks saw, shrunk by the push plus 30 mm, or a declared container's interior. A declared
fixture box only narrows it. ``container_agitate`` keeps both of its needs: a fixture, the box every waypoint of the
agitation stays inside, and a declared container's interior.

Without a fixture a push is the documented default, 30 mm, and at most 50 mm: a request above 50 mm or under 10 mm is
refused with a sentence, never shortened, by the service and by ``PickRun``, and by the console
(``tests/test_api_pick_push_distance.py``, since only a ``test_api_*`` file may import the web framework).

Nothing else of the push moved. The end-to-end cases run on the owner's cell in miniature from
``tests/test_a_boxed_in_part_is_pushed_inside_the_pick.py`` with no fixture declared: ``dense_clutter`` only, a wrist
camera's pick (the same cell on a fixed camera never pushes), ``all_collided``, or a blocked approach where approach
validation runs, with a neighbour within 25 mm, the jaws counted open and never written, an attempt left to pick the
part from after the push, the budgets of 1 per part, 2 per pick and 5 per campaign, a landing over table no look saw
refused before anything moves, and a stop after contact that ends the campaign. The service a loaded config builds
arms the gate itself: a cell that names its hand reads its push inputs with no fixture declared, and hands its picks a
gate with no operator box.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

import numpy as np
from pydantic import ValidationError

from src.config.schema.robot import RobotConfig
from src.geometry import Frame, Transform
from src.robot.core import MotionStatus, RobotError
from src.robot.execution.autonomous_grasp import (
    AutonomousGraspOutcome,
    AutonomousGraspService,
    GraspMode,
    _profile_for,
)
from src.robot.execution.pick_run import PickOutcome, PickRun, Recording
from src.robot.grasping.motion.execution_policy import PolicyOutcome, PolicyReport
from src.robot.grasping.motion.frame_resolver import StaticCameraToBaseResolver
from src.robot.grasping.recovery import push_planner
from src.robot.grasping.recovery.policy import SceneRecoveryAction as A
from src.robot.grasping.recovery.policy import SceneRecoveryPolicy, push_permitted
from src.robot.grasping.recovery.push_gate import PushCell, PushGate
from src.robot.grasping.recovery.push_planner import AUTO_BOX_EXTRA_SHRINK_MM, LANDING_MARGIN_MM, PushPlan
from tests._wrist_views import CUBE, Box, mount
from tests.test_a_boxed_in_part_is_pushed_inside_the_pick import (
    CONTAINER,
    FIXTURE,
    LOOKS,
    PUSH_LABELS,
    WORKSPACE,
    _PushCell,
    _refused,
)
from tests.test_a_wrist_pick_looks_until_its_grasp_is_safe import LOOK_PLUS_X, LOOK_PLUS_Y, _keys
from tests.test_grasping_config_wiring import _calc_and_perception

#: A container's interior around the bench, which ``container_agitate`` needs besides a fixture.
_INTERIOR = {"interior_min_mm": [-200.0, -900.0, 20.0], "interior_max_mm": [200.0, -500.0, 180.0]}
#: The owner's push config with no fixture declared.
_PUSH_ACTIONS = ["rescan", "next_target", "nudge_target"]
#: The owner's hand named on a cell with no gripper driver: what the push's hand is read from (the gripper registry).
_HAND_E = {"vendor": "none", "model": "robotiq_hande"}
_EPS_MM = 1e-6


def _config(recovery: dict[str, Any], container: dict[str, Any] | None = None, *,
            gripper: dict[str, Any] | None = None) -> RobotConfig:
    grasping: dict[str, Any] = {"default_mode": "dense_clutter", "recovery": recovery}
    if container is not None:
        grasping["support"] = {"container": container}
    return RobotConfig(vendor="dummy", gripper=dict(gripper or {"vendor": "none"}), grasping=grasping)


def _service(*, gripper: dict[str, Any] | None = None, fixture: dict[str, Any] | None = None) -> AutonomousGraspService:
    """The service a config that arms the push builds: no fixture declared unless ``fixture`` is, and no hand named
    unless ``gripper`` names one."""
    recovery: dict[str, Any] = {"enabled": True, "allowed_actions": list(_PUSH_ACTIONS)}
    if fixture is not None:
        recovery["fixture"] = fixture
    calculator, perception = _calc_and_perception()
    return AutonomousGraspService.from_robot_config(
        _config(recovery, gripper=gripper), calculator=calculator, perception=perception)


def _gate(service: AutonomousGraspService) -> PushGate | None:
    """The gate the service hands a dense_clutter pick of a campaign it starts now, or ``None``: nothing pushes."""
    return service._push_gate(_profile_for(GraspMode.DENSE_CLUTTER), service._build_recovery_orchestrator_policy(),
                              service.start_campaign())


def _assert_the_campaign_ended(test: unittest.TestCase, cell: _PushCell, run: Any, *, motions: list[str],
                               joints: int) -> None:
    """A push stopped where the arm stands: nothing more moved, the campaign's next pick never started, nothing was
    gripped, and the jaws were never written."""
    test.assertEqual(motions, cell.arm.push_motions(), "a motion followed the stop")
    test.assertEqual(joints, len(cell.joint_moves()), "the arm was sent back to a look after the stop")
    test.assertEqual([PickOutcome.RAISED, PickOutcome.CANCELLED], [a.outcome for a in run.attempts], run.render())
    test.assertEqual(3, len(cell.camera.taken), "a further pick looked again")
    test.assertEqual([], cell.policy.executed)
    cell.assert_the_jaws_untouched(test)


def _planned(cell: _PushCell, **campaign: Any) -> tuple[Any, list[tuple[dict[str, Any], Any]]]:
    """A campaign of ``cell``, and every answer of the push planner with what it was handed. Nobody may be asked."""
    real = push_planner.plan_push
    plans: list[tuple[dict[str, Any], Any]] = []

    def plan_push(**handed: Any) -> Any:
        answer = real(**handed)
        plans.append((handed, answer))
        return answer

    with mock.patch("src.robot.grasping.recovery.push_planner.plan_push", side_effect=plan_push), \
            mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
        run = cell.campaign(**campaign)
    return run, plans


def _blocked_once(cell: _PushCell) -> list[str]:
    """Script the approach check's answer: every approach sweep of the first grasp blocked (nothing moved), the next
    grasp executed. What each call came to, in order."""
    executed = cell.orchestrator._execute  # noqa: SLF001
    answers: list[str] = []

    def execute(result: Any) -> Any:
        if not answers:
            answers.append("blocked")
            return PolicyReport(outcome=PolicyOutcome.APPROACH_PATH_BLOCKED)
        answers.append("executed")
        return executed(result)

    cell.orchestrator._execute = execute  # type: ignore[method-assign]
    return answers


def _inside(xy: Any, box: tuple[tuple[float, float], tuple[float, float]], margin_mm: float = 0.0) -> bool:
    (low_x, low_y), (high_x, high_y) = box
    x, y = float(xy[0]), float(xy[1])
    return (low_x + margin_mm - _EPS_MM <= x <= high_x - margin_mm + _EPS_MM
            and low_y + margin_mm - _EPS_MM <= y <= high_y - margin_mm + _EPS_MM)


class TheConfigLoadsAndArmsThePushWithoutAFixtureTests(unittest.TestCase):
    def test_nudge_target_loads_without_a_fixture(self) -> None:
        cfg = _config({"enabled": True, "allowed_actions": list(_PUSH_ACTIONS)})

        self.assertIsNone(cfg.grasping.recovery.fixture)
        self.assertEqual(tuple(_PUSH_ACTIONS), cfg.grasping.recovery.allowed_actions)

    def test_the_policy_builds_without_a_fixture_and_permits_the_push_in_dense_clutter_only(self) -> None:
        bare = SceneRecoveryPolicy(enabled=True, allowed_actions=(A.RESCAN, A.NUDGE_TARGET))
        self.assertIsNone(bare.fixture)
        self.assertTrue(push_permitted(_profile_for(GraspMode.DENSE_CLUTTER), bare))

        policy = _service()._build_recovery_orchestrator_policy()

        self.assertIsNone(policy.fixture)
        self.assertIn(A.NUDGE_TARGET, policy.allowed_actions)
        self.assertTrue(push_permitted(_profile_for(GraspMode.DENSE_CLUTTER), policy))
        for mode in (GraspMode.AUTO, GraspMode.EASY):
            with self.subTest(mode=mode):
                self.assertFalse(push_permitted(_profile_for(mode), policy))

    def test_every_other_gate_still_closes_it(self) -> None:
        dense = _profile_for(GraspMode.DENSE_CLUTTER)
        closed = {
            "disabled": SceneRecoveryPolicy(enabled=False, allowed_actions=(A.NUDGE_TARGET, A.RESCAN)),
            "not allowed": SceneRecoveryPolicy(enabled=True, allowed_actions=(A.RESCAN,)),
            "mode not applied": SceneRecoveryPolicy(enabled=True, allowed_actions=(A.NUDGE_TARGET,),
                                                    apply_modes=("auto",)),
            "budget zero": SceneRecoveryPolicy(enabled=True, allowed_actions=(A.NUDGE_TARGET,),
                                               per_action_budget={A.NUDGE_TARGET: 0}),
            "max actions zero": SceneRecoveryPolicy(enabled=True, allowed_actions=(A.NUDGE_TARGET,),
                                                    max_recovery_actions=0),
        }
        for name, policy in closed.items():
            with self.subTest(case=name):
                self.assertIsNone(policy.fixture)
                self.assertFalse(push_permitted(dense, policy))

    def test_a_loaded_cell_that_names_its_hand_hands_its_picks_a_gate_with_no_operator_box(self) -> None:
        """From the loaded config to the gate a pick is handed, with no fixture declared: the cell's push inputs are
        read (the Hand-E from the gripper registry, the workspace, the clearance), and the gate carries no operator
        box, so the automatic push box alone bounds the push, at the default 30 mm. The hand is what the gate needs,
        not a box: a cell that names none pushes nothing. A declared fixture's box is the operator box, the one thing
        that narrows the push box."""
        service = _service(gripper=_HAND_E)

        self.assertIsInstance(service.push_cell, PushCell, "the cell's push inputs were not read without a fixture")
        gate = _gate(service)
        self.assertIsInstance(gate, PushGate, "the push was not armed without a fixture")
        assert isinstance(gate, PushGate)
        self.assertIs(service.push_cell, gate.cell)
        self.assertIsNone(gate.operator_box, "something narrowed the push box with no fixture declared")
        self.assertEqual(30.0, gate.distance_mm)

        self.assertNotIsInstance(_service().push_cell, PushCell)
        self.assertIsNone(_gate(_service()), "a cell that names no hand was armed to push")

        boxed = _gate(_service(gripper=_HAND_E, fixture={
            "center_mm": [0.0, -700.0, 100.0], "half_extents_mm": [300.0, 300.0, 300.0]}))
        assert isinstance(boxed, PushGate) and boxed.operator_box is not None
        self.assertEqual(((-300.0, -1000.0), (300.0, -400.0)), boxed.operator_box.xy)
        self.assertEqual(30.0, boxed.distance_mm)


class AnAgitationStillNeedsItsFixtureTests(unittest.TestCase):
    def test_container_agitate_without_a_fixture_is_refused_at_load_with_a_container_declared(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            _config({"enabled": True, "allowed_actions": ["container_agitate"]}, dict(_INTERIOR))
        text = str(caught.exception)
        self.assertIn("container_agitate", text)
        self.assertIn("recovery.fixture", text)

    def test_container_agitate_without_either_is_told_both_at_once(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            _config({"enabled": True, "allowed_actions": ["container_agitate", "nudge_target"]})
        (error,) = caught.exception.errors()
        said = str(error["msg"])
        self.assertIn("recovery.fixture", said)
        self.assertIn("robot.grasping.support.container", said)
        self.assertIn("nudge_target, the push, needs neither", said)
        self.assertNotIn("'nudge_target'", said, "the push was named among what needs a fixture")

    def test_the_policy_still_refuses_an_agitation_without_a_fixture(self) -> None:
        with self.assertRaises(ValueError) as caught:
            SceneRecoveryPolicy(enabled=True, allowed_actions=(A.NUDGE_TARGET, A.CONTAINER_AGITATE))
        self.assertIn("FixtureEnvelope", str(caught.exception))
        self.assertIn("container_agitate", str(caught.exception))


class ABoxedInPartIsPushedWithoutAFixtureTests(unittest.TestCase):
    def test_a_wrist_dense_clutter_pick_pushes_it_inside_the_automatic_box_and_picks_it(self) -> None:
        cell = _PushCell(self, fixture=None)

        run, plans = _planned(cell)

        (attempt,) = run.attempts
        self.assertTrue(attempt.passed, run.render())
        self.assertEqual(PUSH_LABELS, cell.arm.push_motions())
        ((handed, plan),) = plans
        self.assertIsNone(handed["operator_box"], "something narrowed the push box with no fixture declared")
        self.assertIsInstance(plan, PushPlan)
        assert isinstance(plan, PushPlan)
        self.assertEqual(30.0, plan.push_distance_mm)
        # The automatic box: the workspace box intersected with the table the looks saw, shrunk by the push plus
        # 30 mm. The looks saw the table past the workspace's -y face, so that face bounds the box there.
        shrink = plan.push_distance_mm + AUTO_BOX_EXTRA_SHRINK_MM
        (low, high) = plan.landing_box_xy_mm
        self.assertAlmostEqual(WORKSPACE.min_mm[1] + shrink, low[1], places=6)
        self.assertTrue(_inside(low, ((WORKSPACE.min_mm[0], WORKSPACE.min_mm[1]),
                                      (WORKSPACE.max_mm[0], WORKSPACE.max_mm[1])), shrink))
        self.assertTrue(_inside(high, ((WORKSPACE.min_mm[0], WORKSPACE.min_mm[1]),
                                       (WORKSPACE.max_mm[0], WORKSPACE.max_mm[1])), shrink))
        # The landing the plan predicted keeps its margin inside it, and the part landed inside it.
        self.assertTrue(_inside(plan.predicted_landing_centre_mm, plan.landing_box_xy_mm, LANDING_MARGIN_MM))
        landed = cell.camera.centre_of("part")
        self.assertTrue(_inside(landed, plan.landing_box_xy_mm, LANDING_MARGIN_MM), landed)
        self.assertGreater(float(np.linalg.norm(landed[:2] - np.array([0.0, -700.0]))), 25.0, "the part did not move")
        # Picked where it landed, the jaws untouched and nobody asked.
        (grasp,) = cell.policy.executed
        np.testing.assert_allclose(grasp.position, landed)
        cell.assert_the_jaws_untouched(self)
        report = run.last
        self.assertIs(AutonomousGraspOutcome.SUCCEEDED, report.outcome)
        (push,) = report.telemetry["pushes"]
        self.assertEqual(("all_collided", "pushed", 30.0), (push["trigger"], push["code"], push["distance_mm"]))
        self.assertEqual([("nudge_target", "recovered_success", True)],
                         [(row["action"], row["outcome"], row["executed"]) for row in report.recovery_actions])

    def test_a_declared_fixture_box_still_narrows_where_the_part_may_land(self) -> None:
        free, free_plans = _planned(_PushCell(self, fixture=None))
        boxed, boxed_plans = _planned(_PushCell(self, fixture=FIXTURE))

        self.assertTrue(free.attempts[0].passed, free.render())
        self.assertTrue(boxed.attempts[0].passed, boxed.render())
        ((_, free_plan),) = free_plans
        ((handed, boxed_plan),) = boxed_plans
        assert isinstance(free_plan, PushPlan) and isinstance(boxed_plan, PushPlan)
        (cx, cy, _cz), (hx, hy, _hz), _ceiling = FIXTURE
        operator = handed["operator_box"]
        self.assertEqual(((cx - hx, cy - hy), (cx + hx, cy + hy)),
                         ((operator.min_mm[0], operator.min_mm[1]), (operator.max_mm[0], operator.max_mm[1])))
        shrink = boxed_plan.push_distance_mm + AUTO_BOX_EXTRA_SHRINK_MM
        fixture_xy = ((cx - hx, cy - hy), (cx + hx, cy + hy))
        (low, high) = boxed_plan.landing_box_xy_mm
        self.assertTrue(_inside(low, fixture_xy, shrink) and _inside(high, fixture_xy, shrink),
                        f"{boxed_plan.landing_box_xy_mm} reaches past the fixture box shrunk by {shrink:g} mm")
        # Narrowed, never widened: inside the automatic box, and smaller than it.
        self.assertTrue(_inside(low, free_plan.landing_box_xy_mm) and _inside(high, free_plan.landing_box_xy_mm))
        (free_low, free_high) = free_plan.landing_box_xy_mm
        self.assertGreater((free_high[0] - free_low[0]) * (free_high[1] - free_low[1]),
                           (high[0] - low[0]) * (high[1] - low[1]))
        self.assertFalse(_inside(free_high, fixture_xy, shrink), "the fixture box did not narrow anything here")

    def test_a_fixture_box_that_leaves_no_room_still_refuses_the_push(self) -> None:
        cell = _PushCell(self, fixture=((0.0, -700.0, 100.0), (40.0, 40.0, 300.0), 50.0),
                         allowed=("rescan", "nudge_target"))

        report = cell.service.pick(look=list(LOOKS))

        self.assertEqual([], cell.arm.pushes)
        (push,) = cell.orchestrator.pushes
        self.assertEqual("push_box_empty", push.code)
        self.assertFalse(push.motion_started)
        self.assertIs(AutonomousGraspOutcome.NO_VALID_GRASP, report.outcome)
        cell.assert_the_jaws_untouched(self)


class ThePushDistanceWithoutAFixtureTests(unittest.TestCase):
    def test_30_mm_unless_asked_and_50_mm_at_most(self) -> None:
        service = _service()

        self.assertIsNone(service.effective_config.recovery_orchestrator.fixture)
        self.assertEqual(30.0, service.effective_config.recovery_orchestrator.push_distance_mm)
        self.assertEqual(30.0, service.push_distance())
        self.assertEqual(30.0, service.start_campaign().distance_mm)
        self.assertEqual(50.0, service.push_distance(50.0), "the ceiling without a fixture is 50 mm")
        self.assertEqual(10.0, service.push_distance(10.0))
        for asked, said in ((51.0, "the hard cap is 50 mm"), (9.0, "cannot open the 10 mm")):
            with self.subTest(push_mm=asked):
                with self.assertRaises(ValueError) as caught:
                    service.push_distance(asked)
                self.assertIn(said, str(caught.exception))
                with self.assertRaises(ValueError):
                    service.start_campaign(push_mm=asked)
                with self.assertRaises(ValueError):
                    PickRun.from_service(service, runs=1, recording=Recording.off(), push_mm=asked)

    def test_a_campaign_asking_50_mm_pushes_50_mm(self) -> None:
        cell = _PushCell(self, fixture=None)

        run, plans = _planned(cell, push_mm=50.0)

        self.assertTrue(run.attempts[0].passed, run.render())
        self.assertEqual(50.0, cell.service.campaign.distance_mm)
        ((handed, plan),) = plans
        self.assertEqual(50.0, handed["push_distance_mm"])
        assert isinstance(plan, PushPlan)
        (push,) = run.last.telemetry["pushes"]
        self.assertEqual(("pushed", 50.0), (push["code"], push["distance_mm"]))


class NothingElseOfThePushMovedTests(unittest.TestCase):
    """The same cell with no fixture declared: every other precondition of the push holds as it did."""

    def test_auto_never_pushes(self) -> None:
        cell = _PushCell(self, mode=GraspMode.AUTO, allowed=("rescan", "nudge_target"), fixture=None)

        run = cell.campaign()

        self.assertFalse(run.attempts[0].passed)
        self.assertEqual(_keys(*LOOKS), cell.joint_moves(), "the pick did not run its looks")
        self.assertEqual([], cell.arm.pushes, "auto pushed")
        self.assertEqual((), cell.orchestrator.pushes)
        cell.assert_the_jaws_untouched(self)

    def test_no_neighbour_within_25_mm_is_no_push_and_the_planner_is_not_asked(self) -> None:
        far = Box((-80.0, -722.0, 0.0), (-50.0, -714.0, 30.0), "post")
        cell = _PushCell(self, boxes=(CUBE, far), allowed=("rescan", "nudge_target"), fixture=None)

        with mock.patch("src.robot.grasping.recovery.push_planner.plan_push",
                        side_effect=AssertionError("the planner was asked")):
            cell.service.pick(look=list(LOOKS))

        self.assertEqual([], cell.arm.pushes)
        (push,) = cell.orchestrator.pushes
        self.assertEqual("no_blocking_neighbour", push.code)

    def test_a_blocked_approach_pushes_only_where_approach_validation_runs(self) -> None:
        """Every approach sweep of the part's grasp blocked, nothing moved: a trigger of the push where approach
        validation runs, under the same neighbour evidence, and never where it does not."""
        for validation in (True, False):
            with self.subTest(approach_validation=validation):
                # The grasp is uncertain at the first two looks, so all three are fused, and safe at the third.
                cell = _PushCell(self, approach_validation=validation, uncertain_on=(0, 1), freed_on=(2,),
                                 fixture=None)
                answers = _blocked_once(cell)

                run = cell.campaign()

                if validation:
                    self.assertTrue(run.attempts[0].passed, run.render())
                    self.assertEqual(["blocked", "executed"], answers)
                    self.assertEqual(PUSH_LABELS, cell.arm.push_motions())
                    (push,) = run.last.telemetry["pushes"]
                    self.assertEqual(("approach_path_blocked", "pushed"), (push["trigger"], push["code"]))
                else:
                    self.assertEqual(["blocked"], answers)
                    self.assertEqual([], cell.arm.pushes, "a blocked approach pushed where validation does not run")
                    self.assertIn("approach_path_blocked", str(run.last.telemetry["low_level_outcome"]).lower())
                cell.assert_the_jaws_untouched(self)

    def test_a_pick_with_no_attempt_left_after_the_push_is_no_push(self) -> None:
        cell = _PushCell(self, allowed=("rescan", "nudge_target"), max_attempts=1, fixture=None)

        report = cell.service.pick(look=list(LOOKS))

        self.assertEqual([], cell.arm.pushes)
        (push,) = cell.orchestrator.pushes
        self.assertEqual("refused_no_attempt_left", push.code)
        self.assertIs(AutonomousGraspOutcome.NO_VALID_GRASP, report.outcome)

    def test_a_closed_jaw_count_never_pushes_and_never_asks(self) -> None:
        holder: dict[str, Any] = {}

        def spoil(number: int) -> None:
            if number != 2 or holder.get("done"):
                return
            holder["done"] = True
            holder["cell"].jaws.set_closed(True)
            holder["writes"] = len(holder["cell"].events)

        cell = _PushCell(self, allowed=("rescan", "nudge_target"), on_frame=spoil, fixture=None)
        holder["cell"] = cell

        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            cell.service.pick(look=list(LOOKS))

        self.assertEqual([], cell.arm.pushes, "the arm moved for a push on jaws the count says closed")
        self.assertEqual([], cell.nobody.asked, "a person was asked")
        self.assertEqual(holder["writes"], len(cell.events), "the push wrote DO0")
        (push,) = cell.orchestrator.pushes
        self.assertEqual("refused_jaws_closed", push.code)

    def test_one_push_per_part_across_the_picks_of_a_campaign(self) -> None:
        cell = _PushCell(self, allowed=("rescan", "nudge_target"), stuck=True, fixture=None)
        cell.camera.stuck = True

        run = cell.campaign(runs=2)

        self.assertEqual(PUSH_LABELS, cell.arm.push_motions(), "the part was pushed twice")
        self.assertEqual([PickOutcome.FAILED, PickOutcome.FAILED], [a.outcome for a in run.attempts], run.render())
        self.assertEqual(["part_already_pushed"], [push["code"] for push in run.last.telemetry["pushes"]])
        self.assertEqual(1, cell.service.campaign.budgets.pushes_this_campaign)

    def test_two_pushes_per_pick(self) -> None:
        holder: dict[str, Any] = {}

        def two_already(number: int) -> None:
            # Two other parts of this pick were pushed before this one, far from it.
            if number == 0:
                budgets = holder["cell"].service.campaign.budgets
                for x in (400.0, -400.0):
                    budgets.record(centre_mm=(x, -300.0, 20.0), landing_mm=(x, -270.0, 20.0))

        cell = _PushCell(self, allowed=("rescan", "nudge_target"), on_frame=two_already, fixture=None)
        holder["cell"] = cell

        report = cell.service.pick(look=list(LOOKS))

        self.assertEqual([], cell.arm.pushes)
        self.assertEqual("pick_budget_spent", cell.orchestrator.pushes[0].code)
        self.assertFalse(report.needs_person)
        # The next pick has its own two again.
        cell.camera.taken.clear()
        cell.calculator.on_frame = None
        report = cell.service.pick(look=list(LOOKS))
        self.assertEqual(PUSH_LABELS, cell.arm.push_motions())
        self.assertIs(AutonomousGraspOutcome.SUCCEEDED, report.outcome)

    def test_five_pushes_per_campaign_and_a_new_campaign_starts_afresh(self) -> None:
        cell = _PushCell(self, allowed=("rescan", "nudge_target"), fixture=None)
        campaign = cell.service.start_campaign()
        for number in range(5):
            campaign.budgets.record(centre_mm=(400.0 - 150.0 * number, -300.0, 20.0))

        report = cell.service.pick(look=list(LOOKS))

        self.assertEqual([], cell.arm.pushes)
        self.assertEqual("campaign_budget_spent", cell.orchestrator.pushes[0].code)
        self.assertFalse(report.needs_person)
        self.assertIs(AutonomousGraspOutcome.NO_VALID_GRASP, report.outcome)
        # PickRun starts a campaign of its own: its pushes are its own.
        cell.camera.taken.clear()
        run = cell.campaign()
        self.assertTrue(run.attempts[0].passed, run.render())
        self.assertEqual(PUSH_LABELS, cell.arm.push_motions())

    def test_a_fixed_camera_never_pushes_where_the_wrist_camera_does(self) -> None:
        """A pick handed no look, from where the arm stands, a declared container answering where the part may land:
        on the wrist camera the part is pushed and picked. The same cell with its camera bolted where the wrist camera
        stood (its frames placed by that fixed transform) never considers a push, and nothing moves."""
        for camera in ("wrist", "fixed"):
            with self.subTest(camera=camera):
                cell = _PushCell(self, allowed=("rescan", "nudge_target"), container=CONTAINER, fixture=None)
                cell.arm.move_to_joints(LOOK_PLUS_Y)  # where the arm stands; the pick is handed no look
                cell.arm.motions.clear()
                if camera == "fixed":
                    standing = cell.arm.get_tcp_pose().to_matrix() @ mount()
                    cell.orchestrator.frame_resolver = StaticCameraToBaseResolver(
                        transform=Transform.from_matrix(standing, from_frame=Frame.CAMERA, to_frame=Frame.BASE))

                report = cell.service.pick()

                if camera == "wrist":
                    self.assertEqual(PUSH_LABELS, cell.arm.push_motions())
                    self.assertIs(AutonomousGraspOutcome.SUCCEEDED, report.outcome, report.render())
                else:
                    self.assertEqual([], cell.arm.motions, "the arm moved on a fixed camera's pick")
                    self.assertEqual((), cell.orchestrator.pushes, "a fixed camera's pick considered a push")
                    self.assertNotIn("pushes", report.telemetry)
                    self.assertEqual([], cell.policy.executed)
                cell.assert_the_jaws_untouched(self)

    def test_a_landing_over_table_no_look_saw_is_refused_before_anything_moves(self) -> None:
        """With no fixture the automatic push box is all that bounds the landing. A pick handed no look rescans where
        it stands; from its one 45 degree view the part's shadow hides the table where it would land, so the planner
        refuses, nothing moves, and the pick goes on to that rescan, with the same refusal."""
        cell = _PushCell(self, allowed=("rescan", "nudge_target"), fixture=None)
        cell.arm.move_to_joints(LOOK_PLUS_X)  # where the arm stands; the pick is handed no look
        cell.arm.motions.clear()
        where = cell.arm.get_tcp_pose()

        with mock.patch("src.robot.grasping.recovery.push_planner.plan_push",
                        wraps=push_planner.plan_push) as planned:
            report = cell.service.pick()

        self.assertTrue(planned.call_args_list, "the planner was not asked")
        self.assertEqual([None] * len(planned.call_args_list),
                         [call.kwargs["operator_box"] for call in planned.call_args_list],
                         "something narrowed the push box with no fixture declared")
        self.assertEqual([], cell.arm.pushes, "the arm was asked to move for a push the planner refused")
        self.assertIs(where, cell.arm.get_tcp_pose(), "the arm moved")
        self.assertEqual([], cell.joint_moves(), "the arm was sent somewhere")
        self.assertEqual(["rescan", "rescan", "rescan"], [attempt.action for attempt in report.pick_report.attempts])
        pushes = cell.orchestrator.pushes
        self.assertEqual("no_free_direction", pushes[0].code)
        self.assertIn("landing_over_unseen_table", pushes[0].sentence)
        self.assertTrue(all(not push.motion_started for push in pushes))
        self.assertFalse(report.needs_person)
        self.assertEqual([], cell.policy.executed)
        cell.assert_the_jaws_untouched(self)


class AStopAfterContactStillEndsTheCampaignTests(unittest.TestCase):
    """With no fixture declared, a failure of P0 once it may have been sent, or of any leg once the down leg was sent,
    still ends the push where the arm is: nothing more is commanded, nothing clears a stop, and the campaign ends."""

    def test_a_failure_of_p0_once_sent_or_of_any_leg_after_the_down_leg_was_sent_stops_there(self) -> None:
        cases = {
            "push_p0": _refused(MotionStatus.CONTROLLER_REJECTED, "Driver reported moveJ() failure"),
            "push_down": RobotError("the line raised"),
            "push_push": _refused(MotionStatus.SELF_COLLISION_REJECTED, "rfinger 4 mm off the post"),
            "push_back": _refused(MotionStatus.WORKSPACE_REJECTED, "sample 3 leaves the box"),
            "push_up": _refused(MotionStatus.CONTROLLER_REJECTED, "Driver reported moveL() failure"),
        }
        for leg, answer in cases.items():
            with self.subTest(leg=leg):
                cell = _PushCell(self, answers={leg: answer}, fixture=None)

                run = cell.campaign(runs=2)

                _assert_the_campaign_ended(self, cell, run, motions=PUSH_LABELS[:PUSH_LABELS.index(leg) + 1], joints=3)
                report = run.last
                self.assertIs(AutonomousGraspOutcome.UNSAFE_RECOVERY_REFUSED, report.outcome)
                self.assertTrue(report.needs_person)
                self.assertIn("a recovery stopped where the arm stands", run.attempts[0].detail)
                (push,) = report.telemetry["pushes"]
                self.assertEqual((True, True), (push["stopped"], push["motion_started"]))
                self.assertEqual(1, len(cell.service.campaign.budgets.records), "a push that moved was not counted")
                # The service starts no further pick until a person decides.
                again = cell.service.pick(look=list(LOOKS))
                self.assertTrue(again.needs_person)
                self.assertEqual(PUSH_LABELS[:PUSH_LABELS.index(leg) + 1], cell.arm.push_motions())
                self.assertEqual(3, len(cell.joint_moves()), "the arm left where the push stopped it")
                self.assertEqual(3, len(cell.camera.taken), "a further pick looked")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
