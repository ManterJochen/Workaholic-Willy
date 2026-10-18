"""Cross-view target association: the decisions must be right, and the refusals must be refusals.

The expensive failure this module can produce is not a miss — it is picking the WRONG object, because
the fused cloud then carries a contact face belonging to a neighbour and the grasp generator has no way
to tell it from a real one. So the tests are built around that asymmetry.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.multiview.association import (
    ViewCandidates,
    associate_target,
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
        result = associate_target(target, others)[0]
        self.assertEqual(result.index, 1, "the overlap picked the wrong object")
        self.assertTrue(result.matched)

    def test_a_view_that_contains_only_other_objects_abstains(self) -> None:
        """Costing one view is the cheap failure; fabricating a contact face is the expensive one."""
        target = _slab((100.0, 0.0, 30.0))
        others = (ViewCandidates("oblique", (_slab((600.0, 0.0, 30.0)), _slab((600.0, 200.0, 30.0)))),)
        result = associate_target(target, others)[0]
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
        result = associate_target(target, (far,), min_score=0.99)[0]
        self.assertIsNone(result.index)
        self.assertEqual(len(result.scores), 1, "the score is kept so a refusal is diagnosable")

    def test_ties_go_to_the_first_detection_so_the_result_is_deterministic(self) -> None:
        target = _slab((100.0, 0.0, 30.0))
        twin = ViewCandidates("oblique", (_slab((100.0, 0.0, 30.0)), _slab((100.0, 0.0, 30.0))))
        first = associate_target(target, (twin,))[0]
        second = associate_target(target, (twin,))[0]
        self.assertEqual(first.index, 0)
        self.assertEqual(first.index, second.index)

    def test_margin_reports_how_close_the_call_was(self) -> None:
        target = _slab((100.0, 0.0, 30.0))
        clear = ViewCandidates("v", (_slab((100.0, 1.0, 30.0)), _slab((900.0, 0.0, 30.0))))
        close = ViewCandidates("v", (_slab((100.0, 1.0, 30.0)), _slab((100.0, 3.0, 30.0))))
        wide = associate_target(target, (clear,))[0]
        tight = associate_target(target, (close,))[0]
        self.assertGreater(wide.margin, tight.margin,
                           "an ambiguous decision must not look as confident as an obvious one")


if __name__ == "__main__":
    unittest.main()
