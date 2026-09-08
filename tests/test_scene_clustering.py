"""Grouping every camera's detections into objects, with no camera privileged.

Today a pick's object list is the primary camera's list, and every other camera is matched against it.
An object only a secondary camera can see is dropped: it matches nothing, because there is nothing for
it to match. `cluster_views` is the piece that removes the asymmetry, and the whole of it is one rule.

⛔⛔ **COMPLETE LINKAGE, NOT UNION-FIND, AND THE DIFFERENCE IS A GRASP ON A SHAPE THAT DOES NOT
EXIST.** A blob joins a group only if it clears the threshold against EVERY blob already in it. The
case that separates the two rules is three blobs where A matches B and B matches C and A does not
match C, which a threshold produces routinely on parts that touch:

* single linkage, which is what a union-find over the same edges computes, puts all three in one
  group. One false edge welds three objects, the fused cloud spans the gap between them, and the
  closing axis is derived from a silhouette nothing has.
* complete linkage refuses. B goes with whichever of A or C it agrees with more, the third stays its
  own object, and the cost is a duplicate.

The two failures are not equally priced. A duplicate costs one attempt that finds nothing. A phantom
drives a jaw into the space between two parts. That is why the owner chose splitting over welding,
and every test below is about holding that choice in place, because the merging version is the one
somebody will reach for when a duplicate shows up in a log.

⚠ NEVER RUN ON HARDWARE. Point clouds here are constructed so the metric answers what the test needs;
no physical multi-camera cell exists.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.multiview.association import (
    AssociationMetric,
    ViewCandidates,
    cluster_views,
)


def _blob(centre: tuple[float, float, float], *, spread: float = 4.0, n: int = 60) -> np.ndarray:
    """A small cloud around a point. Deterministic, so a failure is reproducible."""
    rng = np.random.default_rng(abs(hash(centre)) % (2**32))
    return np.asarray(centre, dtype=np.float64) + rng.uniform(-spread, spread, size=(n, 3))


def _view(name: str, *centres: tuple[float, float, float]) -> ViewCandidates:
    return ViewCandidates(name=name, clouds_base_mm=tuple(_blob(c) for c in centres))


def _by_view(cluster) -> dict[int, list[int]]:  # noqa: ANN001
    out: dict[int, list[int]] = {}
    for view, blob in cluster.members:
        out.setdefault(view, []).append(blob)
    return out


class OneObjectSeenByEveryCameraIsOneObjectTests(unittest.TestCase):
    def test_three_cameras_looking_at_one_part_produce_one_cluster(self) -> None:
        """The control that fails if the grouping is inert: three views, one object, one entry.

        Without this, a scene list built from every camera's detections would carry an object once
        per camera that saw it, which is worse than the asymmetry it replaces: the same part would be
        picked, then picked again at a place where it no longer is.
        """
        views = [_view("left", (0.0, 0.0, 0.0)),
                 _view("right", (2.0, 0.0, 0.0)),
                 _view("top", (0.0, 2.0, 0.0))]

        clusters = cluster_views(views, metric=AssociationMetric.CENTROID, max_centroid_mm=150.0)

        self.assertEqual(1, len(clusters))
        self.assertEqual((0, 1, 2), clusters[0].views)

    def test_two_parts_far_apart_stay_two_objects(self) -> None:
        views = [_view("left", (0.0, 0.0, 0.0), (500.0, 0.0, 0.0)),
                 _view("right", (2.0, 0.0, 0.0), (502.0, 0.0, 0.0))]

        clusters = cluster_views(views, metric=AssociationMetric.CENTROID, max_centroid_mm=150.0)

        self.assertEqual(2, len(clusters))
        for cluster in clusters:
            self.assertEqual((0, 1), cluster.views, "each part should be seen by both cameras")


class AnObjectOnlyOneCameraSeesSurvivesTests(unittest.TestCase):
    """The reason the whole symmetric pass exists. Today this object is dropped."""

    def test_a_part_only_the_second_camera_saw_is_its_own_cluster(self) -> None:
        views = [_view("left", (0.0, 0.0, 0.0)),
                 _view("right", (2.0, 0.0, 0.0), (900.0, 0.0, 0.0))]

        clusters = cluster_views(views, metric=AssociationMetric.CENTROID, max_centroid_mm=150.0)

        self.assertEqual(2, len(clusters))
        alone = [c for c in clusters if len(c.members) == 1]
        self.assertEqual(1, len(alone), "the unmatched blob must survive as an object")
        self.assertEqual((1, 1), alone[0].members[0], "and it must be the second camera's second blob")

    def test_a_cluster_of_one_reports_full_agreement_rather_than_none(self) -> None:
        """There is nothing to disagree with. Reporting 0.0 would read as a weak match."""
        clusters = cluster_views([_view("only", (0.0, 0.0, 0.0))],
                                 metric=AssociationMetric.CENTROID)

        self.assertEqual(1, len(clusters))
        self.assertEqual(1.0, clusters[0].agreement)


class TheThreeWayCaseTests(unittest.TestCase):
    """⛔ THE TEST THIS MODULE EXISTS FOR. A matches B, B matches C, A does not match C.

    Constructed rather than hoped for: the three centres are placed so that the two neighbouring
    gaps clear the threshold and the outer gap does not. Union-find would return one cluster of
    three. Complete linkage must return two objects.
    """

    def _abc(self) -> list[ViewCandidates]:
        # 80 mm apart pairwise, 160 mm across, against a 150 mm centroid scale: the outer pair scores
        # zero and the two inner pairs do not.
        return [_view("a", (0.0, 0.0, 0.0)),
                _view("b", (80.0, 0.0, 0.0)),
                _view("c", (160.0, 0.0, 0.0))]

    def test_the_premise_holds_before_the_conclusion_is_asserted(self) -> None:
        """The control. If the outer pair also cleared, the test below would pass for the wrong
        reason and would keep passing under a rule it is supposed to catch."""
        pairs = cluster_views([_view("a", (0.0, 0.0, 0.0)), _view("c", (160.0, 0.0, 0.0))],
                              metric=AssociationMetric.CENTROID, max_centroid_mm=150.0)
        neighbours = cluster_views([_view("a", (0.0, 0.0, 0.0)), _view("b", (80.0, 0.0, 0.0))],
                                   metric=AssociationMetric.CENTROID, max_centroid_mm=150.0)

        self.assertEqual(2, len(pairs), "A and C must NOT match, or the case is not the case")
        self.assertEqual(1, len(neighbours), "A and B must match, or there is no chain to break")

    def test_a_chain_is_broken_rather_than_welded(self) -> None:
        clusters = cluster_views(self._abc(), metric=AssociationMetric.CENTROID,
                                 max_centroid_mm=150.0)

        self.assertEqual(2, len(clusters),
                         "single linkage would return 1: one false edge welding three objects")
        sizes = sorted(len(c.members) for c in clusters)
        self.assertEqual([1, 2], sizes)

    def test_the_odd_one_out_is_a_whole_object_and_not_a_fragment(self) -> None:
        """A split leaves a duplicate, which costs an attempt. It must not leave a partial object,
        which would cost a grasp planned on half a part."""
        clusters = cluster_views(self._abc(), metric=AssociationMetric.CENTROID,
                                 max_centroid_mm=150.0)

        alone = next(c for c in clusters if len(c.members) == 1)
        self.assertEqual(1, len(_by_view(alone)), "it belongs to exactly one camera")


class OneCameraNeverMatchesItselfTests(unittest.TestCase):
    def test_two_detections_in_one_view_are_two_objects(self) -> None:
        """Even when they overlap. Whether one camera's two detections are one object is the
        segmenter's decision, made with the image; this function has only points and would be
        second-guessing it with less information."""
        views = [ViewCandidates(name="only", clouds_base_mm=(
            _blob((0.0, 0.0, 0.0)), _blob((1.0, 0.0, 0.0))))]

        clusters = cluster_views(views, metric=AssociationMetric.CENTROID, max_centroid_mm=150.0)

        self.assertEqual(2, len(clusters))

    def test_a_camera_that_segmented_nothing_contributes_nothing(self) -> None:
        views = [_view("left", (0.0, 0.0, 0.0)), ViewCandidates(name="blind", clouds_base_mm=())]

        clusters = cluster_views(views, metric=AssociationMetric.CENTROID)

        self.assertEqual(1, len(clusters))
        self.assertEqual((0,), clusters[0].views)


class ItIsDeterministicTests(unittest.TestCase):
    def test_the_same_rig_gives_the_same_clusters_every_time(self) -> None:
        views = [_view("a", (0.0, 0.0, 0.0), (400.0, 0.0, 0.0)),
                 _view("b", (3.0, 0.0, 0.0), (403.0, 0.0, 0.0)),
                 _view("c", (6.0, 0.0, 0.0))]

        runs = [cluster_views(views, metric=AssociationMetric.CENTROID, max_centroid_mm=150.0)
                for _ in range(5)]

        for later in runs[1:]:
            self.assertEqual(runs[0], later)

    def test_members_are_ordered_so_a_diff_of_two_runs_is_readable(self) -> None:
        views = [_view("a", (0.0, 0.0, 0.0)), _view("b", (2.0, 0.0, 0.0)),
                 _view("c", (4.0, 0.0, 0.0))]

        clusters = cluster_views(views, metric=AssociationMetric.CENTROID, max_centroid_mm=150.0)

        for cluster in clusters:
            self.assertEqual(sorted(cluster.members), list(cluster.members))


if __name__ == "__main__":
    unittest.main()
