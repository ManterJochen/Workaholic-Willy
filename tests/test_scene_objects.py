"""The list a pick iterates: one entry per object, and the camera each one is grasped from.

The pick loop used to iterate the primary camera's segmentation list, and every consumer downstream
was an integer index into it. So an object existed only if the primary had a segmentation for it, and
"which object" and "which position in one camera's list" were the same fact.

`build_scene_objects` separates them. Two properties carry the whole change and both are asserted
here rather than argued:

* **Order is a contract.** Every object the primary saw comes first, in the primary's own order, so a
  cell whose other cameras add nothing produces exactly the list it produced before, position for
  position. That is what makes the default path unchanged rather than merely similar.
* **The camera travels with the object.** `camera_id`, `segmentation`, `depth_map` and `intrinsics`
  come from ONE camera, because the calculator is called with all four and mixing them across
  cameras is a grasp synthesised in a frame nothing was measured in.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.robot.grasping.multiview.association import SceneCluster
from src.robot.grasping.multiview.scene_geometry import ObservedView, build_scene_objects

_K = np.array([[600.0, 0.0, 32.0], [0.0, 600.0, 32.0], [0.0, 0.0, 1.0]])


def _view(name: str, count: int, *, with_segmentations: bool = True) -> ObservedView:
    masks = tuple(np.zeros((8, 8), dtype=bool) for _ in range(count))
    segs = tuple(SimpleNamespace(label=f"{name}-{i}", mask=m) for i, m in enumerate(masks))
    return ObservedView(
        name=name,
        masks=masks,
        depth_map=np.full((8, 8), 500.0 + 10.0 * len(name)),
        intrinsics=_K,
        camera_to_base=np.eye(4),
        segmentations=segs if with_segmentations else (),
    )


class ThePrimaryComesFirstAndInItsOwnOrderTests(unittest.TestCase):
    def test_a_cell_whose_other_cameras_add_nothing_gets_the_list_it_had(self) -> None:
        """The property that makes this change default-off rather than merely default-quiet: same
        length, same order, same masks, nothing promoted."""
        primary = _view("cam_left", 3)
        clusters = tuple(SceneCluster(members=((0, i),), agreement=1.0) for i in range(3))

        objects = build_scene_objects([primary], clusters)

        self.assertEqual(3, len(objects))
        for index, obj in enumerate(objects):
            self.assertIs(primary.segmentations[index], obj.segmentation, "order and identity")
            self.assertEqual("cam_left", obj.camera_id)
            self.assertFalse(obj.promoted)

    def test_a_primary_object_stays_at_its_own_index_even_when_a_promoted_one_exists(self) -> None:
        """The index is a position now, but it is still the position the rest of the loop uses, so a
        promoted object must never push a primary one along."""
        primary, other = _view("cam_left", 2), _view("cam_right", 3)
        clusters = (
            SceneCluster(members=((0, 0), (1, 0)), agreement=0.8),
            SceneCluster(members=((0, 1), (1, 1)), agreement=0.8),
            SceneCluster(members=((1, 2),), agreement=1.0),
        )

        objects = build_scene_objects([primary, other], clusters)

        self.assertEqual(3, len(objects), "two the primary saw, one only the other camera did")
        self.assertIs(primary.segmentations[0], objects[0].segmentation)
        self.assertIs(primary.segmentations[1], objects[1].segmentation)
        self.assertTrue(objects[2].promoted)
        self.assertFalse(any(o.promoted for o in objects[:2]))


class ThePromotedObjectCarriesItsOwnCameraTests(unittest.TestCase):
    """Everything the calculator is called with has to come from the camera that saw the object."""

    def test_it_carries_that_camera_s_segmentation_depth_and_intrinsics(self) -> None:
        primary, other = _view("cam_left", 1), _view("cam_right", 2)
        clusters = (
            SceneCluster(members=((0, 0), (1, 0)), agreement=0.9),
            SceneCluster(members=((1, 1),), agreement=1.0),
        )

        objects = build_scene_objects([primary, other], clusters)

        promoted = objects[-1]
        self.assertTrue(promoted.promoted)
        self.assertEqual("cam_right", promoted.camera_id)
        self.assertIs(other.segmentations[1], promoted.segmentation)
        np.testing.assert_array_equal(other.depth_map, promoted.depth_map)
        self.assertIsNot(primary.depth_map, promoted.depth_map,
                         "the primary's depth would place this object in a frame it was not seen in")

    def test_an_object_both_cameras_saw_is_grasped_from_the_PRIMARY(self) -> None:
        """Not from whichever camera scored higher. Every measured number in this repository was
        taken with the grasp synthesised from the primary's reading, so an object the primary saw
        must keep being read from the primary or the numbers stop comparing."""
        primary, other = _view("cam_left", 1), _view("cam_right", 1)
        clusters = (SceneCluster(members=((0, 0), (1, 0)), agreement=0.9),)

        objects = build_scene_objects([primary, other], clusters)

        self.assertEqual("cam_left", objects[0].camera_id)
        self.assertEqual(("cam_left", "cam_right"), objects[0].views_used)


class ViewsUsedIsTheHonestRecordTests(unittest.TestCase):
    def test_an_object_one_camera_saw_reports_only_that_camera(self) -> None:
        primary = _view("cam_left", 1)
        objects = build_scene_objects([primary], (SceneCluster(members=((0, 0),), agreement=1.0),))

        self.assertEqual(("cam_left",), objects[0].views_used)

    def test_a_primary_object_no_cluster_mentions_still_reports_the_primary(self) -> None:
        """A caller may skip clustering entirely, which is what the default path does. The object is
        still an object and it was still seen by exactly one camera."""
        primary = _view("cam_left", 2)

        objects = build_scene_objects([primary], ())

        self.assertEqual(2, len(objects))
        for obj in objects:
            self.assertEqual(("cam_left",), obj.views_used)
            self.assertFalse(obj.promoted)


class APromotedObjectIsFUSED_TooTests(unittest.TestCase):
    """The deficit fusion exists to close, left open longest for the objects that need it most.

    An object the primary camera cannot see is by construction one that something is in front of. So
    it is exactly the case where a single view is worst, and it was the case that got a single view:
    the fusion path was built around the primary, so a promoted object carried no fused surface even
    when two other cameras had seen it.
    """

    @staticmethod
    def _clouds(counts: dict[tuple[int, int], int]) -> list[list[np.ndarray]]:
        """Per-view blobs, each an (n, 3) array whose row count identifies it."""
        out: list[list[np.ndarray]] = [[], [], []]
        for (view, blob), rows in counts.items():
            while len(out[view]) <= blob:
                out[view].append(np.zeros((0, 3)))
            out[view][blob] = np.full((rows, 3), float(view * 10 + blob))
        return out

    def test_two_cameras_that_saw_it_are_both_in_its_cloud(self) -> None:
        primary, second, third = _view("a", 1), _view("b", 1), _view("c", 1)
        clusters = (
            SceneCluster(members=((0, 0),), agreement=1.0),
            SceneCluster(members=((1, 0), (2, 0)), agreement=0.8),
        )
        clouds = self._clouds({(0, 0): 5, (1, 0): 7, (2, 0): 11})

        objects = build_scene_objects([primary, second, third], clusters, view_clouds=clouds)

        promoted = objects[-1]
        self.assertTrue(promoted.promoted)
        assert promoted.fused_cloud_base_mm is not None
        self.assertEqual(18, len(promoted.fused_cloud_base_mm), "7 from b and 11 from c")

    def test_the_owning_camera_s_points_come_first(self) -> None:
        """The fused cloud starts where the grasp is being synthesised, which is the same order
        `fuse_scene_clouds` gives a primary object and for the same reason."""
        primary, second, third = _view("a", 1), _view("b", 1), _view("c", 1)
        clusters = (
            SceneCluster(members=((0, 0),), agreement=1.0),
            SceneCluster(members=((1, 0), (2, 0)), agreement=0.8),
        )
        clouds = self._clouds({(0, 0): 5, (1, 0): 7, (2, 0): 11})

        objects = build_scene_objects([primary, second, third], clusters, view_clouds=clouds)

        cloud = objects[-1].fused_cloud_base_mm
        assert cloud is not None
        self.assertEqual(10.0, float(cloud[0][0]), "camera b owns it, so b's points lead")

    def test_an_object_only_ONE_camera_saw_gets_no_fused_cloud(self) -> None:
        """`None` rather than its own points, the same rule the primary objects follow: the caller
        omits the generator kwarg and the calculator derives the cloud from the mask and depth it was
        already given, instead of being handed back what it would have derived."""
        primary, second = _view("a", 1), _view("b", 1)
        clusters = (
            SceneCluster(members=((0, 0),), agreement=1.0),
            SceneCluster(members=((1, 0),), agreement=1.0),
        )
        clouds = self._clouds({(0, 0): 5, (1, 0): 7})

        objects = build_scene_objects([primary, second], clusters, view_clouds=clouds)

        self.assertTrue(objects[-1].promoted)
        self.assertIsNone(objects[-1].fused_cloud_base_mm)

    def test_without_the_clouds_nothing_is_invented(self) -> None:
        """A caller that does not supply them gets no fused surface rather than a fabricated one."""
        primary, second, third = _view("a", 1), _view("b", 1), _view("c", 1)
        clusters = (
            SceneCluster(members=((0, 0),), agreement=1.0),
            SceneCluster(members=((1, 0), (2, 0)), agreement=0.8),
        )

        objects = build_scene_objects([primary, second, third], clusters)

        self.assertIsNone(objects[-1].fused_cloud_base_mm)


class OrderIsDeterministicTests(unittest.TestCase):
    def test_promoted_objects_are_ordered_by_camera_then_by_detection(self) -> None:
        primary, second, third = _view("a", 1), _view("b", 2), _view("c", 1)
        clusters = (
            SceneCluster(members=((0, 0),), agreement=1.0),
            SceneCluster(members=((2, 0),), agreement=1.0),
            SceneCluster(members=((1, 1),), agreement=1.0),
            SceneCluster(members=((1, 0),), agreement=1.0),
        )

        objects = build_scene_objects([primary, second, third], clusters)

        self.assertEqual(["a", "b", "b", "c"], [o.camera_id for o in objects])
        self.assertEqual(["b-0", "b-1"], [o.segmentation.label for o in objects[1:3]])


class AViewWithoutSegmentationsIsStillUsableTests(unittest.TestCase):
    def test_a_masks_only_view_yields_objects_with_no_segmentation(self) -> None:
        """The old shape: a caller that only fuses surfaces onto objects the primary already found
        passes masks and nothing else. Such an object cannot be grasped from that camera, and the
        `None` is what says so rather than a stand-in that would fail later and further away."""
        primary = _view("cam_left", 1, with_segmentations=False)

        objects = build_scene_objects([primary], (SceneCluster(members=((0, 0),), agreement=1.0),))

        self.assertIsNone(objects[0].segmentation)


if __name__ == "__main__":
    unittest.main()
