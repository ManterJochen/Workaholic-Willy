"""A push needs no fixture box (the owner, 2026-10-01: "es werden ja eh nur Teile geschoben ... das soll optional werden").

Until then four places refused or disarmed ``nudge_target`` without a declared ``robot.grasping.recovery.fixture``. The
push never needed that box to know where a part may land: the **automatic push box** bounds it, the workspace box
intersected with the table the looks saw, shrunk by 30 mm, or a declared container's interior. The fixture box that
could narrow it left on 2026-10-16 with ``container_agitate``, which needed it, and the whole ``recovery`` block with
them (the owner's cleanup): a cell built from config runs the owner's fixed recovery.

A push is 30 mm unless a run asks for another distance, and at most 50 mm: a request above 50 mm or under 10 mm is
refused with a sentence, never shortened, by the service and by ``PickRun``, and by the console
(``tests/test_api_pick_push_distance.py``, since only a ``test_api_*`` file may import the web framework).

Pinned here: a loaded cell that names its hand hands its dense_clutter picks a push gate, and one that names none
pushes nothing; the part lands inside the automatic box and is picked where it landed; a campaign asking 50 mm pushes
50 mm. The rest of the push is pinned in ``tests/test_a_boxed_in_part_is_pushed_inside_the_pick.py``.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

import numpy as np

from src.config.schema.robot import RobotConfig
from src.robot.execution.autonomous_grasp import (
    AutonomousGraspOutcome,
    AutonomousGraspService,
    GraspMode,
    _profile_for,
)
from src.robot.execution.pick_run import PickRun, Recording
from src.robot.grasping.recovery import push_planner
from src.robot.grasping.recovery.policy import SceneRecoveryAction as A
from src.robot.grasping.recovery.policy import push_permitted
from src.robot.grasping.recovery.push_gate import PushCell, PushGate
from src.robot.grasping.recovery.push_planner import AUTO_BOX_EXTRA_SHRINK_MM, LANDING_MARGIN_MM, PushPlan
from tests.test_a_boxed_in_part_is_pushed_inside_the_pick import PUSH_LABELS, WORKSPACE, _PushCell
from tests.test_grasping_config_wiring import _calc_and_perception

#: The owner's hand named on a cell with no gripper driver: what the push's hand is read from (the gripper registry).
_HAND_E = {"vendor": "none", "model": "robotiq_hande"}
_EPS_MM = 1e-6


def _service(*, gripper: dict[str, Any] | None = None) -> AutonomousGraspService:
    """The service a dense_clutter config builds, no hand named unless ``gripper`` names one."""
    calculator, perception = _calc_and_perception()
    cfg = RobotConfig(vendor="dummy", gripper=dict(gripper or {"vendor": "none"}),
                      grasping={"default_mode": "dense_clutter"})
    return AutonomousGraspService.from_robot_config(cfg, calculator=calculator, perception=perception)


def _gate(service: AutonomousGraspService) -> PushGate | None:
    """The gate the service hands a dense_clutter pick of a campaign it starts now, or ``None``: nothing pushes."""
    return service._push_gate(_profile_for(GraspMode.DENSE_CLUTTER), service._build_recovery_orchestrator_policy(),
                              service.start_campaign())


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


def _inside(xy: Any, box: tuple[tuple[float, float], tuple[float, float]], margin_mm: float = 0.0) -> bool:
    (low_x, low_y), (high_x, high_y) = box
    x, y = float(xy[0]), float(xy[1])
    return (low_x + margin_mm - _EPS_MM <= x <= high_x - margin_mm + _EPS_MM
            and low_y + margin_mm - _EPS_MM <= y <= high_y - margin_mm + _EPS_MM)


class ALoadedCellArmsThePushTests(unittest.TestCase):
    def test_the_policy_permits_the_push_in_dense_clutter_only(self) -> None:
        policy = _service()._build_recovery_orchestrator_policy()

        self.assertIn(A.NUDGE_TARGET, policy.allowed_actions)
        self.assertTrue(push_permitted(_profile_for(GraspMode.DENSE_CLUTTER), policy))
        for mode in (GraspMode.AUTO, GraspMode.EASY):
            with self.subTest(mode=mode):
                self.assertFalse(push_permitted(_profile_for(mode), policy))

    def test_a_loaded_cell_that_names_its_hand_hands_its_picks_a_gate(self) -> None:
        """From the loaded config to the gate a pick is handed: the cell's push inputs are read (the Hand-E from the
        gripper registry, the workspace, the clearance), at the default 30 mm. The hand is what the gate needs: a cell
        that names none pushes nothing."""
        service = _service(gripper=_HAND_E)

        self.assertIsInstance(service.push_cell, PushCell, "the cell's push inputs were not read")
        gate = _gate(service)
        self.assertIsInstance(gate, PushGate, "the push was not armed")
        assert isinstance(gate, PushGate)
        self.assertIs(service.push_cell, gate.cell)
        self.assertEqual(30.0, gate.distance_mm)

        self.assertNotIsInstance(_service().push_cell, PushCell)
        self.assertIsNone(_gate(_service()), "a cell that names no hand was armed to push")


class ABoxedInPartIsPushedInsideTheAutomaticBoxTests(unittest.TestCase):
    def test_a_wrist_dense_clutter_pick_pushes_it_inside_the_automatic_box_and_picks_it(self) -> None:
        cell = _PushCell(self)

        run, plans = _planned(cell)

        (attempt,) = run.attempts
        self.assertTrue(attempt.passed, run.render())
        self.assertEqual(PUSH_LABELS, cell.arm.push_motions())
        ((handed, plan),) = plans
        self.assertNotIn("operator_box", handed)
        self.assertIsInstance(plan, PushPlan)
        assert isinstance(plan, PushPlan)
        self.assertEqual(30.0, plan.push_distance_mm)
        # The automatic box: the workspace box intersected with the table the looks saw, shrunk by 30 mm, however far
        # the push goes (the owner, 2026-10-03). The looks saw the table past the workspace's -y face, so that face
        # bounds the box there.
        shrink = AUTO_BOX_EXTRA_SHRINK_MM
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


class ThePushDistanceTests(unittest.TestCase):
    def test_30_mm_unless_asked_and_50_mm_at_most(self) -> None:
        service = _service()

        self.assertEqual(30.0, service.push_distance())
        self.assertEqual(30.0, service.start_campaign().distance_mm)
        self.assertEqual(50.0, service.push_distance(50.0), "the hard cap is 50 mm")
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
        cell = _PushCell(self)

        run, plans = _planned(cell, push_mm=50.0)

        self.assertTrue(run.attempts[0].passed, run.render())
        self.assertEqual(50.0, cell.service.campaign.distance_mm)
        ((handed, plan),) = plans
        self.assertEqual(50.0, handed["push_distance_mm"])
        assert isinstance(plan, PushPlan)
        (push,) = run.last.telemetry["pushes"]
        self.assertEqual(("pushed", 50.0), (push["code"], push["distance_mm"]))


if __name__ == "__main__":
    unittest.main()
