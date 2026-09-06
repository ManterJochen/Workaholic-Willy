"""Tests for Phase S6 active-perception planner.

Coverage:

* :class:`ViewpointSignals` validation.
* :class:`ViewpointHistory` record / best_observation behaviour.
* :class:`AcceptAllViewpointSafetyCheck` and
  :class:`WorkspaceBoxSafetyCheck`.
* :class:`ViewScoringPolicy` validation.
* :class:`ScoringViewpointPlanner` determinism, history-driven
  diversity, zero-candidate penalty, occlusion preference,
  exhaustion, safety-check filtering, and Protocol compatibility
  with :class:`BinPickingOrchestrator`'s :class:`ViewpointPlanner`
  surface.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.geometry import Frame, Pose
from src.robot.grasping import (
    AcceptAllViewpointSafetyCheck,
    ScoringViewpointPlanner,
    ViewScoringPolicy,
    ViewpointHistory,
    ViewpointObservation,
    ViewpointSignals,
    WorkspaceBoxSafetyCheck,
)
from src.robot.grasping.loop.pick_loop import ViewpointPlanner


def _pose(xyz=(0.0, 0.0, 200.0), label: str = "tcp") -> Pose:
    return Pose(
        position_mm=np.asarray(xyz, dtype=np.float64),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        frame=Frame.BASE,
        label=label,
    )


class ViewpointSignalsTests(unittest.TestCase):
    def test_accepts_valid_signals(self) -> None:
        s = ViewpointSignals(
            occlusion_ratio=0.4,
            approach_clearance_mm=12.5,
            reachability_score=0.8,
            candidate_count=3,
        )
        self.assertAlmostEqual(s.occlusion_ratio or 0.0, 0.4)

    def test_rejects_out_of_range_occlusion(self) -> None:
        with self.assertRaises(ValueError):
            ViewpointSignals(occlusion_ratio=-0.1)
        with self.assertRaises(ValueError):
            ViewpointSignals(occlusion_ratio=1.5)

    def test_rejects_negative_clearance(self) -> None:
        with self.assertRaises(ValueError):
            ViewpointSignals(approach_clearance_mm=-1.0)

    def test_rejects_negative_count(self) -> None:
        with self.assertRaises(ValueError):
            ViewpointSignals(candidate_count=-1)

    def test_rejects_out_of_range_reachability(self) -> None:
        with self.assertRaises(ValueError):
            ViewpointSignals(reachability_score=1.5)


class ViewpointHistoryTests(unittest.TestCase):
    def test_records_observations_in_order(self) -> None:
        hist = ViewpointHistory()
        a = ViewpointObservation(pose=_pose((10.0, 0.0, 200.0)))
        b = ViewpointObservation(pose=_pose((20.0, 0.0, 200.0)))
        hist.record(a)
        hist.record(b)
        self.assertEqual(hist.observations, [a, b])
        self.assertEqual(hist.poses, (a.pose, b.pose))

    def test_best_observation_prefers_low_occlusion_with_candidates(self) -> None:
        hist = ViewpointHistory()
        hist.record(
            ViewpointObservation(
                pose=_pose((10.0, 0.0, 200.0)),
                signals=ViewpointSignals(occlusion_ratio=0.9, candidate_count=0),
            )
        )
        best_obs = ViewpointObservation(
            pose=_pose((20.0, 0.0, 200.0)),
            signals=ViewpointSignals(occlusion_ratio=0.1, candidate_count=5),
        )
        hist.record(best_obs)
        self.assertIs(hist.best_observation, best_obs)

    def test_best_observation_is_none_when_empty(self) -> None:
        self.assertIsNone(ViewpointHistory().best_observation)


class WorkspaceBoxSafetyCheckTests(unittest.TestCase):
    def test_inside_box_is_safe(self) -> None:
        check = WorkspaceBoxSafetyCheck(
            center_mm=(0.0, 0.0, 200.0),
            half_extents_mm=(100.0, 100.0, 100.0),
        )
        self.assertTrue(check.is_safe(_pose((0.0, 0.0, 200.0))))
        self.assertTrue(check.is_safe(_pose((100.0, 0.0, 200.0))))

    def test_outside_box_is_unsafe(self) -> None:
        check = WorkspaceBoxSafetyCheck(
            center_mm=(0.0, 0.0, 200.0),
            half_extents_mm=(50.0, 50.0, 50.0),
        )
        self.assertFalse(check.is_safe(_pose((60.0, 0.0, 200.0))))

    def test_rejects_invalid_dims(self) -> None:
        with self.assertRaises(ValueError):
            WorkspaceBoxSafetyCheck(
                center_mm=(0.0, 0.0),  # type: ignore[arg-type]
                half_extents_mm=(1.0, 1.0, 1.0),
            )
        with self.assertRaises(ValueError):
            WorkspaceBoxSafetyCheck(
                center_mm=(0.0, 0.0, 0.0),
                half_extents_mm=(-1.0, 1.0, 1.0),
            )


class ViewScoringPolicyTests(unittest.TestCase):
    def test_defaults_are_safe(self) -> None:
        p = ViewScoringPolicy()
        self.assertGreater(p.max_viewpoints, 0)
        self.assertGreater(p.lateral_offset_mm, 0.0)

    def test_rejects_negative_budget(self) -> None:
        with self.assertRaises(ValueError):
            ViewScoringPolicy(max_viewpoints=-1)

    def test_rejects_negative_offsets(self) -> None:
        with self.assertRaises(ValueError):
            ViewScoringPolicy(lateral_offset_mm=-1.0)
        with self.assertRaises(ValueError):
            ViewScoringPolicy(vertical_offset_mm=-1.0)

    def test_rejects_negative_weights(self) -> None:
        with self.assertRaises(ValueError):
            ViewScoringPolicy(diversity_weight=-1.0)
        with self.assertRaises(ValueError):
            ViewScoringPolicy(occlusion_weight=-1.0)


class ScoringViewpointPlannerTests(unittest.TestCase):
    def test_implements_viewpoint_planner_protocol(self) -> None:
        planner = ScoringViewpointPlanner()
        self.assertIsInstance(planner, ViewpointPlanner)

    def test_returns_pose_in_base_frame(self) -> None:
        planner = ScoringViewpointPlanner()
        pose = planner.next_viewpoint(current_tcp=_pose(), history=())
        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertIs(pose.frame, Frame.BASE)

    def test_exhausts_after_max_viewpoints(self) -> None:
        planner = ScoringViewpointPlanner(
            policy=ViewScoringPolicy(max_viewpoints=2)
        )
        history: list[Pose] = []
        for _ in range(2):
            p = planner.next_viewpoint(current_tcp=_pose(), history=tuple(history))
            assert p is not None
            history.append(p)
        self.assertIsNone(
            planner.next_viewpoint(current_tcp=_pose(), history=tuple(history))
        )

    def test_deterministic_for_same_input(self) -> None:
        p1 = ScoringViewpointPlanner()
        p2 = ScoringViewpointPlanner()
        a = p1.next_viewpoint(current_tcp=_pose(), history=())
        b = p2.next_viewpoint(current_tcp=_pose(), history=())
        assert a is not None and b is not None
        self.assertTrue(np.allclose(a.position_mm, b.position_mm))
        self.assertEqual(a.label, b.label)

    def test_safety_check_filters_candidates(self) -> None:
        # Workspace box tight around origin; the lateral offset
        # candidates all land at |x|=50, outside a |x|<=10 box -> no
        # safe candidate, planner returns None.
        planner = ScoringViewpointPlanner(
            policy=ViewScoringPolicy(
                lateral_offset_mm=50.0, vertical_offset_mm=0.0
            ),
            safety_check=WorkspaceBoxSafetyCheck(
                center_mm=(0.0, 0.0, 200.0),
                half_extents_mm=(10.0, 10.0, 10.0),
            ),
        )
        self.assertIsNone(
            planner.next_viewpoint(current_tcp=_pose(), history=())
        )

    def test_zero_candidate_observation_pushes_planner_away(self) -> None:
        # Record a "+x viewpoint produced zero candidates" history.
        # The planner should prefer a different candidate next.
        planner = ScoringViewpointPlanner(
            policy=ViewScoringPolicy(
                lateral_offset_mm=50.0,
                vertical_offset_mm=0.0,
                # Disable scene-signal weights other than the implicit
                # zero-candidate penalty inside _score_one.
                occlusion_weight=0.0,
                clearance_weight=0.0,
                reachability_weight=0.0,
            )
        )
        bad_pose = _pose((50.0, 0.0, 200.0), label="viewpoint_+x")
        planner.record_observation(
            ViewpointObservation(
                pose=bad_pose,
                signals=ViewpointSignals(candidate_count=0),
            )
        )
        # First proposal: must not be the +x direction (which is
        # right next to the bad pose).
        chosen = planner.next_viewpoint(current_tcp=_pose(), history=())
        assert chosen is not None
        self.assertFalse(
            np.allclose(chosen.position_mm, bad_pose.position_mm)
        )

    def test_diversity_drives_away_from_visited_history(self) -> None:
        # Caller has already visited +x. Planner should pick a
        # different cardinal.
        planner = ScoringViewpointPlanner(
            policy=ViewScoringPolicy(
                lateral_offset_mm=50.0,
                vertical_offset_mm=0.0,
                occlusion_weight=0.0,
                clearance_weight=0.0,
                reachability_weight=0.0,
            )
        )
        visited = _pose((50.0, 0.0, 200.0))
        next_pose = planner.next_viewpoint(
            current_tcp=_pose(), history=(visited,)
        )
        assert next_pose is not None
        self.assertFalse(
            np.allclose(next_pose.position_mm, visited.position_mm)
        )

    def test_candidates_includes_lift_when_vertical_offset_positive(self) -> None:
        planner = ScoringViewpointPlanner(
            policy=ViewScoringPolicy(vertical_offset_mm=25.0)
        )
        cands = planner.candidates(_pose())
        labels = {c.reason for c in cands}
        self.assertIn("+z", labels)

    def test_candidates_skip_lift_when_vertical_offset_zero(self) -> None:
        planner = ScoringViewpointPlanner(
            policy=ViewScoringPolicy(vertical_offset_mm=0.0)
        )
        cands = planner.candidates(_pose())
        labels = {c.reason for c in cands}
        self.assertNotIn("+z", labels)

    def test_record_observation_is_tracked(self) -> None:
        planner = ScoringViewpointPlanner()
        obs = ViewpointObservation(
            pose=_pose((10.0, 0.0, 200.0)),
            signals=ViewpointSignals(occlusion_ratio=0.2, candidate_count=1),
            reason="initial_capture",
        )
        planner.record_observation(obs)
        self.assertEqual(planner.last_selection_reason, "initial_capture")
        self.assertIn(obs, planner.history.observations)

    def test_min_score_to_act_can_block_selection(self) -> None:
        planner = ScoringViewpointPlanner(
            policy=ViewScoringPolicy(
                lateral_offset_mm=50.0,
                vertical_offset_mm=0.0,
                diversity_weight=0.0,
                occlusion_weight=0.0,
                clearance_weight=0.0,
                reachability_weight=0.0,
                min_score_to_act=10.0,
            )
        )
        self.assertIsNone(
            planner.next_viewpoint(current_tcp=_pose(), history=())
        )

    def test_accept_all_safety_check_default(self) -> None:
        check = AcceptAllViewpointSafetyCheck()
        self.assertTrue(check.is_safe(_pose((1e6, 1e6, 1e6))))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
