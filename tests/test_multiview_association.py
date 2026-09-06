"""Cross-view target association: the decisions must be right, and the refusals must be refusals.

The expensive failure this module can produce is not a miss — it is picking the WRONG object, because
the fused cloud then carries a contact face belonging to a neighbour and the grasp generator has no way
to tell it from a real one. So the tests are built around that asymmetry.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.multiview.association import (
    AssociationMetric,
    ViewCandidates,
    associate_target,
    fuse_target_cloud,
)


def _slab(centre: tuple[float, float, float], size: float = 40.0, n: int = 8) -> np.ndarray:
    """A filled cube of points around a centre, in BASE mm."""
    axis = np.linspace(-size / 2.0, size / 2.0, n)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid + np.asarray(centre, dtype=np.float64)


class AssociationTests(unittest.TestCase):
    def test_the_same_object_seen_from_another_angle_is_matched(self) -> None:
        target = _slab((100.0, 0.0, 30.0))
        others = (ViewCandidates("oblique", (_slab((400.0, 0.0, 30.0)), _slab((100.0, 2.0, 30.0)))),)
        for metric in AssociationMetric:
            with self.subTest(metric=metric):
                result = associate_target(target, others, metric=metric)[0]
                self.assertEqual(result.index, 1, f"{metric} picked the wrong object")
                self.assertTrue(result.matched)

    def test_a_view_that_contains_only_other_objects_abstains(self) -> None:
        """Costing one view is the cheap failure; fabricating a contact face is the expensive one."""
        target = _slab((100.0, 0.0, 30.0))
        others = (ViewCandidates("oblique", (_slab((600.0, 0.0, 30.0)), _slab((600.0, 200.0, 30.0)))),)
        for metric in AssociationMetric:
            with self.subTest(metric=metric):
                result = associate_target(target, others, metric=metric)[0]
                self.assertIsNone(result.index)
                self.assertFalse(result.matched)

    def test_a_view_with_no_detections_is_reported_not_skipped(self) -> None:
        """One result per view, matched or not, so a caller can log WHY a view was dropped."""
        results = associate_target(_slab((0.0, 0.0, 0.0)), (ViewCandidates("blind", ()),))
        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0].index)
        self.assertEqual(results[0].scores, ())

    def test_the_evidence_survives_a_refusal(self) -> None:
        target = _slab((100.0, 0.0, 30.0))
        far = ViewCandidates("oblique", (_slab((260.0, 0.0, 30.0)),))
        result = associate_target(target, (far,), metric=AssociationMetric.OVERLAP,
                                  min_score=0.99)[0]
        self.assertIsNone(result.index)
        self.assertEqual(len(result.scores), 1, "the score is kept so a refusal is diagnosable")

    def test_ties_go_to_the_first_detection_so_the_result_is_deterministic(self) -> None:
        target = _slab((100.0, 0.0, 30.0))
        twin = ViewCandidates("oblique", (_slab((100.0, 0.0, 30.0)), _slab((100.0, 0.0, 30.0))))
        first = associate_target(target, (twin,), metric=AssociationMetric.BOX_IOU)[0]
        second = associate_target(target, (twin,), metric=AssociationMetric.BOX_IOU)[0]
        self.assertEqual(first.index, 0)
        self.assertEqual(first.index, second.index)

    def test_margin_reports_how_close_the_call_was(self) -> None:
        target = _slab((100.0, 0.0, 30.0))
        clear = ViewCandidates("v", (_slab((100.0, 1.0, 30.0)), _slab((900.0, 0.0, 30.0))))
        close = ViewCandidates("v", (_slab((100.0, 1.0, 30.0)), _slab((100.0, 3.0, 30.0))))
        wide = associate_target(target, (clear,), metric=AssociationMetric.OVERLAP)[0]
        tight = associate_target(target, (close,), metric=AssociationMetric.OVERLAP)[0]
        self.assertGreater(wide.margin, tight.margin,
                           "an ambiguous decision must not look as confident as an obvious one")


class FusionTests(unittest.TestCase):
    def test_a_matched_view_adds_its_points(self) -> None:
        target = _slab((100.0, 0.0, 30.0))
        other = _slab((100.0, 2.0, 30.0))
        fused, results = fuse_target_cloud(target, (ViewCandidates("oblique", (other,)),))
        self.assertEqual(fused.shape[0], target.shape[0] + other.shape[0])
        self.assertTrue(results[0].matched)

    def test_no_match_leaves_the_single_view_cloud_untouched(self) -> None:
        """The default path has to stay byte-identical when no other camera can confirm the target."""
        target = _slab((100.0, 0.0, 30.0))
        elsewhere = ViewCandidates("oblique", (_slab((900.0, 0.0, 30.0)),))
        fused, results = fuse_target_cloud(target, (elsewhere,))
        np.testing.assert_array_equal(fused, target)
        self.assertFalse(results[0].matched)

    def test_the_target_points_come_first_and_are_unmodified(self) -> None:
        target = _slab((100.0, 0.0, 30.0))
        fused, _ = fuse_target_cloud(
            target, (ViewCandidates("a", (_slab((100.0, 2.0, 30.0)),)),))
        np.testing.assert_array_equal(fused[:target.shape[0]], target)


if __name__ == "__main__":
    unittest.main()
