"""Clearing a blocker where a look from straight above hides its foot, or where no grasp of the part can be reached
(URSim, 2026-10-03).

The owner's LOOK[0] looks at the mat almost straight down: a 30 mm block's sides are seen at a grazing angle and left
out of what the calculator saw beside the part, so its lowest point seen stood 22 mm over the mat, and the clearing took
it for something resting on something else. And once one block of an L was set aside, both grasps the part had left
lifted wrist 1 out of the cable window the owner's Monday tree keeps (``within_half_turn_of_home``), each refused only
once the hand stood at the part. Pinned:

* a neighbour whose foot no look saw stands on the support where the support's own pixels fill a quarter of the ring
  within 15 mm round it, the push's rule (the owner: "Allgemein ... sofern nicht ein Abgrund"); where the camera saw the
  support under it, it floats (a finger, a cable) and is never gripped;
* every grasp of the part is judged from the look before the arm leaves for it, as a blocker's are; a grasp refused
  there moves nothing and the next follows;
* where every grasp of the part was refused that way, nothing moved, the scene is changed as for a part every grasp of
  which collided: for critical parts a blocker is cleared, and the part is picked after it.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Any

import numpy as np

from src.robot.grasping.recovery.blocker import (
    STANDS_ON_THE_SUPPORT_MM,
    Cluster,
    not_a_blocker,
    support_seen_round,
)
from src.robot.grasping.types.feedback import GraspResult


def _square(cx: float, cy: float, half: float, step: float = 2.0) -> np.ndarray:
    xs = np.arange(cx - half, cx + half + 1e-9, step)
    ys = np.arange(cy - half, cy + half + 1e-9, step)
    return np.array([(x, y) for x in xs for y in ys])


#: A 30 mm block's footprint, and a mat seen round it to 60 mm, none of it under the block.
FOOT = _square(0.0, 0.0, 15.0)
MAT = np.array([p for p in _square(0.0, 0.0, 60.0, step=2.5) if max(abs(p[0]), abs(p[1])) > 15.5])


class AFootNoLookSawTests(unittest.TestCase):

    def test_the_support_seen_round_it_says_it_stands(self) -> None:
        self.assertTrue(support_seen_round(FOOT, MAT))

    def test_the_support_seen_under_it_says_it_floats(self) -> None:
        under = np.vstack([MAT, _square(0.0, 0.0, 12.0, step=2.5)])
        self.assertFalse(support_seen_round(FOOT, under))

    def test_neighbours_alone_round_it_say_nothing(self) -> None:
        far = np.array([p for p in MAT if max(abs(p[0]), abs(p[1])) > 40.0])
        self.assertFalse(support_seen_round(FOOT, far))
        self.assertFalse(support_seen_round(FOOT, np.zeros((0, 2))))

    def test_the_blocker_rule_takes_the_inferred_foot(self) -> None:
        # Its sides seen from straight above at a grazing angle and left out: its top alone, 30 mm over the mat.
        top = np.column_stack([FOOT, np.full(len(FOOT), 85.0)])
        cluster = Cluster(points_base_mm=top)
        mat_top = 55.0
        self.assertGreater(cluster.low_mm - mat_top, STANDS_ON_THE_SUPPORT_MM)
        said = not_a_blocker(cluster, support_mm=mat_top, target_points_mm=np.zeros((0, 3)), base_radius_mm=None,
                             open_width_mm=50.0)
        self.assertIn("does not stand on what the part stands on", said)
        self.assertEqual("", not_a_blocker(cluster, support_mm=mat_top, target_points_mm=np.zeros((0, 3)),
                                           base_radius_mm=None, open_width_mm=50.0, foot_seen_round=True))


class NoGraspThePartCanReachChangesTheSceneTests(unittest.TestCase):
    """On the clearing's scene of ``test_a_blocker_is_set_aside_before_the_part.py``: with the post beside it, the
    calculator offers the part a grasp closing along x that the arm, judged ahead, would refuse at the lift; with the
    post gone, one closing along y that it admits."""

    def _cell(self) -> Any:
        from tests.test_a_blocker_is_set_aside_before_the_part import _Cell, _top_down

        cell = _Cell()
        assert cell.orchestrator.push_gate is not None
        cell.orchestrator.push_gate = replace(cell.orchestrator.push_gate, critical_parts=True)
        calculator = cell.calculator
        real = calculator.compute_result

        def compute(seg: Any, depth: Any, *args: Any, **kwargs: Any) -> GraspResult:
            answer = real(seg, depth, *args, **kwargs)
            if str(getattr(seg, "label", "")) != "part":
                return answer
            part = _top_down(cell.scene.centre("part"), width_mm=40.0, label="part")
            if answer.is_success:
                # The post gone: a grasp along y, which the arm reaches.
                return replace(answer, candidates=(replace(part, axis=np.array([0.0, 1.0, 0.0])),))
            # The post beside the part: one grasp, along x, which no arm reaches with the part in its jaws.
            return GraspResult(candidates=(replace(part, axis=np.array([1.0, 0.0, 0.0])),), top_score=0.9,
                               metadata=answer.metadata, telemetry=answer.telemetry)

        calculator.compute_result = compute  # type: ignore[method-assign]
        judged: list[tuple[float, ...]] = []

        def ahead(grasp: Any) -> str:
            judged.append(tuple(round(float(v), 3) for v in grasp.axis))
            if str(getattr(grasp, "label", "")) == "part" and abs(float(grasp.axis[0])) > 0.9:
                return "the lift, as if the jaws held a part 40 mm across, would be refused (joint_limit_rejected)"
            return ""

        cell.policy.refusal_ahead = ahead  # type: ignore[attr-defined]
        return cell, judged

    def test_a_grasp_judged_ahead_as_refused_moves_nothing_and_the_blocker_is_cleared(self) -> None:
        cell, judged = self._cell()

        report = cell.run()

        self.assertIn((1.0, 0.0, 0.0), judged, "the part's grasp was not judged ahead")
        self.assertEqual("set_aside", cell.orchestrator.blockers[0].code, cell.orchestrator.blockers[0].sentence)
        self.assertEqual(["clear_blocker", "executed"], [a.action for a in report.attempts])
        self.assertEqual(("motion_plan_refused",), tuple(str(r.value) for r in report.attempts[0].reasons))
        # The grasp refused ahead was never driven to: the hand closed on the post, opened over its spot, and closed on
        # the part, one command each.
        self.assertEqual(["close", "open", "close"], cell.hand.commands)
        self.assertEqual((), cell.orchestrator.pushes, "a critical part was pushed")


if __name__ == "__main__":
    unittest.main()
