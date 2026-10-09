"""A followed target is kept out of the planner world by its new mask, never by the kept cloud (fail-closed rule 1).

What a pick leaves out of the planner world is its target: the target's mask in the frame it ranked, and its surface in
BASE, whose box no frame the world holds keeps (``segmentation_offer_from_frame``). On a followed frame both come from
this frame's mask, cut by SAM2 on the part's box and clipped to the kept footprint, shifted by the part's move, and
15 mm: never from the cloud the memory kept, and never past the part. The parts named beside it are exactly the parts
followed, so a finger may come to their measured surface and to nothing else the memory saw.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from src.robot.grasping.loop.pick_loop import segmentation_offer_from_frame
from tests.test_a_kept_scene_follows_only_what_stands_where_it_stood import CAMERA, kept_of, shoot, two_grey_cubes
from tests.test_a_source_grounds_the_frame_whenever_following_fails import _follow, _source


def _offered(frame: Any, target: int = 0) -> Any:
    return segmentation_offer_from_frame(frame, target, camera="wrist", camera_to_base_mm=CAMERA)


class AFollowedTargetIsKeptOutByItsNewMaskTests(unittest.TestCase):
    def setUp(self) -> None:
        scene = two_grey_cubes()
        self.kept = kept_of(shoot(scene))
        # The memory's cloud of part 0 stands 9 mm from where the part stands now (a finger nudged it).
        self.now = shoot([scene[0].moved(dx=9.0), scene[1]])
        source, _, self.grounder, _ = _source(self.now)
        self.frame = source.acquire(follow=_follow(self.kept))
        self.route = source.last_route

    def test_the_frame_was_followed(self) -> None:
        self.assertEqual("followed", self.route[0])
        self.assertEqual([], self.grounder.asked)

    def test_the_target_s_mask_is_this_frame_s(self) -> None:
        offer = _offered(self.frame)

        self.assertEqual(1, len(offer.exclude_masks))
        self.assertTrue(np.array_equal(np.asarray(offer.exclude_masks[0]).astype(bool), self.now.hit == 0))

    def test_the_target_s_points_stand_where_the_part_stands_now(self) -> None:
        offer = _offered(self.frame)
        points = np.asarray(offer.target_points_base_mm)
        kept_centre = self.kept.parts[0].centre_xy_mm

        moved = np.median(points[:, :2], axis=0) - kept_centre
        self.assertAlmostEqual(9.0, float(moved[0]), delta=1.5)
        self.assertAlmostEqual(0.0, float(moved[1]), delta=1.5)

    def test_no_target_point_stands_past_the_shifted_footprint_and_15_mm(self) -> None:
        offer = _offered(self.frame)
        points = np.asarray(offer.target_points_base_mm)
        footprint = self.kept.parts[0].cloud_base_mm[:, :2] + np.array([9.0, 0.0])

        distance, _ = cKDTree(footprint).query(points[:, :2])
        self.assertLessEqual(float(distance.max()), 15.0)

    def test_the_parts_named_beside_the_target_are_exactly_the_parts_followed(self) -> None:
        offer = _offered(self.frame)

        self.assertEqual(["grey cube"], [label for label, _ in offer.named_points_base_mm])
        named = np.asarray(offer.named_points_base_mm[0][1])
        # Every point named is the other cube's: its 40 mm box about (60, -560), a millimetre round it.
        inside = (np.abs(named[:, 0] - 60.0) <= 21.0) & (np.abs(named[:, 1] + 560.0) <= 21.0) & (named[:, 2] <= 41.0)
        self.assertTrue(bool(np.all(inside)))
        self.assertEqual(2, len(self.frame.segmentations), "nothing else the frame shows is named")


class AStrayPixelOfAFollowedMaskIsNeverLeftOutTests(unittest.TestCase):
    def test_a_mask_with_a_stray_patch_far_from_the_part_offers_none_of_it(self) -> None:
        scene = two_grey_cubes()
        now = shoot(scene)
        source, _, _, segmenter = _source(now)
        cut = segmenter.segment_detection

        def with_a_stray_patch(bgr: Any, det: Any) -> Any:
            seg = cut(bgr, det)
            mask = np.asarray(seg.mask).copy()
            mask[10:13, 10:13] = 1  # nine pixels of bench, far from every part: under the leak bound
            return type(seg)(mask=mask, label=seg.label)

        segmenter.segment_detection = with_a_stray_patch  # type: ignore[method-assign]
        frame = source.acquire(follow=_follow(kept_of(shoot(scene))))

        self.assertEqual("followed", source.last_route[0])
        offer = _offered(frame)
        self.assertFalse(np.asarray(offer.exclude_masks[0]).astype(bool)[10:13, 10:13].any())
        labelled = dict((label, mask) for label, mask in offer.labelled_masks[:1])
        self.assertFalse(np.asarray(labelled["grey cube"]).astype(bool)[10:13, 10:13].any())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
