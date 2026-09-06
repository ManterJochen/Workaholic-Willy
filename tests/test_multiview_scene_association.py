"""Whole-scene multi-view association: every object, and no candidate used twice.

``associate_target`` answers "which blob in the other camera is MY target". A cell that fuses for
every candidate object asks it once per object, and the moment it does, a new failure appears that
the single-target function cannot have: two objects independently choosing the SAME partner. The
loser gets a neighbour's far surface welded into its cloud, and the grasp generator has no way to
tell a fabricated contact face from a real one.

These tests pin that the one-to-one constraint holds, that a refusal is still a refusal under it,
and that the single-view path is untouched when no other camera confirms anything.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.multiview.association import (
    AssociationMetric,
    ViewCandidates,
    assign_view,
    fuse_scene_clouds,
)


def _box(centre: tuple[float, float, float], *, half: float = 10.0, n: int = 8) -> np.ndarray:
    """A small filled cube of points around ``centre``, in BASE mm."""

    axis = np.linspace(-half, half, n)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid + np.asarray(centre, dtype=np.float64)


class OneToOneTests(unittest.TestCase):
    """No candidate may be handed to two primaries."""

    def test_two_primaries_cannot_both_take_the_same_candidate(self) -> None:
        """The whole reason this function exists rather than a loop over associate_target.

        Two primaries sit near each other; the other camera sees only ONE of them plus a far-away
        third object. Scored independently both primaries would claim the single nearby candidate.
        """

        near_a = _box((0.0, 0.0, 0.0))
        near_b = _box((14.0, 0.0, 0.0))  # overlaps A's neighbourhood: both score against candidate 0
        view = ViewCandidates(
            name="left",
            clouds_base_mm=(_box((0.0, 0.0, 0.0)), _box((500.0, 0.0, 0.0))),
        )

        association = assign_view([near_a, near_b], view)

        # The scene must actually BE a contest, or this test proves nothing. Measured 1.000 / 0.875:
        # scored independently, both primaries clear the threshold against candidate 0 and take it.
        self.assertGreaterEqual(association.scores[0][0], 0.30)
        self.assertGreaterEqual(association.scores[1][0], 0.30, "no contest -- test would be vacuous")

        assigned = [i for i in association.assignment if i is not None]
        self.assertEqual(
            len(assigned),
            len(set(assigned)),
            f"candidate handed to more than one primary: {association.assignment}",
        )

    def test_the_better_claim_wins_the_contested_candidate(self) -> None:
        """One-to-one is only useful if the pair it keeps is the well-supported one."""

        exact = _box((0.0, 0.0, 0.0))
        offset = _box((16.0, 0.0, 0.0))
        view = ViewCandidates(name="left", clouds_base_mm=(_box((0.0, 0.0, 0.0)),))

        association = assign_view([offset, exact], view)

        # Measured 0.750 vs 1.000 -- the loser is admissible on its own, so the constraint is what
        # decides, not the threshold.
        self.assertGreaterEqual(association.scores[0][0], 0.30, "no contest -- test would be vacuous")
        self.assertIsNone(association.assignment[0], "the weaker claim kept the contested candidate")
        self.assertEqual(association.assignment[1], 0)

    def test_disjoint_objects_each_keep_their_own_partner(self) -> None:
        """The constraint must not cost anything when there is no contest."""

        primaries = [_box((0.0, 0.0, 0.0)), _box((400.0, 0.0, 0.0))]
        # Deliberately in the OPPOSITE order, so a positional shortcut would fail this.
        view = ViewCandidates(
            name="right",
            clouds_base_mm=(_box((400.0, 0.0, 0.0)), _box((0.0, 0.0, 0.0))),
        )

        association = assign_view(primaries, view)

        self.assertEqual(association.assignment, (1, 0))
        self.assertEqual(association.matched_count, 2)


class RefusalTests(unittest.TestCase):
    """A view that cannot identify an object must contribute nothing for it."""

    def test_an_object_no_camera_sees_stays_unmatched(self) -> None:
        primaries = [_box((0.0, 0.0, 0.0))]
        view = ViewCandidates(name="left", clouds_base_mm=(_box((900.0, 0.0, 0.0)),))

        association = assign_view(primaries, view)

        self.assertEqual(association.assignment, (None,))

    def test_a_sub_threshold_pair_is_never_taken_to_fill_a_slot(self) -> None:
        """Zeroing before the optimisation, not filtering after it.

        With one primary and one candidate the assignment problem has exactly one solution, so a
        solver run on the raw scores would return it and a later filter would have to undo it. The
        pair must be refused by the optimisation itself.
        """

        primaries = [_box((0.0, 0.0, 0.0))]
        view = ViewCandidates(name="left", clouds_base_mm=(_box((60.0, 0.0, 0.0)),))

        association = assign_view(primaries, view, min_score=0.30)

        self.assertLess(association.scores[0][0], 0.30, "test scene no longer sub-threshold")
        self.assertEqual(association.assignment, (None,))

    def test_scores_survive_a_refusal(self) -> None:
        """A bare None is not diagnosable; the runner-up evidence has to stay."""

        primaries = [_box((0.0, 0.0, 0.0))]
        view = ViewCandidates(name="left", clouds_base_mm=(_box((900.0, 0.0, 0.0)), _box((0.0, 0.0, 0.0))))

        association = assign_view(primaries, view)

        self.assertEqual(len(association.scores), 1)
        self.assertEqual(len(association.scores[0]), 2)


class FuseSceneCloudsTests(unittest.TestCase):
    """The fused output contract."""

    def test_unconfirmed_object_comes_back_exactly_as_it_went_in(self) -> None:
        """The single-view path is unchanged, not degraded, when nothing confirms an object."""

        primary = _box((0.0, 0.0, 0.0))
        view = ViewCandidates(name="left", clouds_base_mm=(_box((900.0, 0.0, 0.0)),))

        fused, _ = fuse_scene_clouds([primary], [view])

        np.testing.assert_array_equal(fused[0], primary)

    def test_a_confirmed_object_gains_the_other_view(self) -> None:
        primary = _box((0.0, 0.0, 0.0))
        partner = _box((0.0, 0.0, 0.0)) + np.array([1.0, 0.0, 0.0])
        view = ViewCandidates(name="left", clouds_base_mm=(partner,))

        fused, associations = fuse_scene_clouds([primary], [view])

        self.assertEqual(associations[0].assignment, (0,))
        self.assertEqual(len(fused[0]), len(primary) + len(partner))
        # The object's own cloud stays first, so a caller can still find its primary evidence.
        np.testing.assert_array_equal(fused[0][: len(primary)], primary)

    def test_output_is_one_cloud_per_primary_in_input_order(self) -> None:
        primaries = [_box((0.0, 0.0, 0.0)), _box((400.0, 0.0, 0.0)), _box((800.0, 0.0, 0.0))]
        view = ViewCandidates(name="left", clouds_base_mm=(_box((400.0, 0.0, 0.0)),))

        fused, _ = fuse_scene_clouds(primaries, [view])

        self.assertEqual(len(fused), 3)
        np.testing.assert_array_equal(fused[0], primaries[0])
        np.testing.assert_array_equal(fused[2], primaries[2])

    def test_no_views_is_a_no_op(self) -> None:
        primaries = [_box((0.0, 0.0, 0.0))]

        fused, associations = fuse_scene_clouds(primaries, [])

        self.assertEqual(associations, ())
        np.testing.assert_array_equal(fused[0], primaries[0])

    def test_empty_scene_is_handled(self) -> None:
        fused, associations = fuse_scene_clouds([], [ViewCandidates("left", (_box((0.0, 0.0, 0.0)),))])

        self.assertEqual(fused, ())
        self.assertEqual(associations[0].assignment, ())


class DeterminismTests(unittest.TestCase):
    """Same input, same answer -- the repo's standing requirement for this layer."""

    def test_repeated_runs_agree(self) -> None:
        primaries = [_box((0.0, 0.0, 0.0)), _box((30.0, 0.0, 0.0))]
        view = ViewCandidates(
            name="left", clouds_base_mm=(_box((30.0, 0.0, 0.0)), _box((0.0, 0.0, 0.0)))
        )

        first = assign_view(primaries, view, metric=AssociationMetric.OVERLAP)
        for _ in range(5):
            self.assertEqual(assign_view(primaries, view).assignment, first.assignment)


if __name__ == "__main__":
    unittest.main()
