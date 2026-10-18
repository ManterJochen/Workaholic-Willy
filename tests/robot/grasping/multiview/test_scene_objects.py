"""The list a pick iterates: one entry per object the primary camera segmented, in its own order.

The pick loop used to iterate the primary camera's segmentation list, and every consumer downstream
was an integer index into it. `build_scene_objects` keeps that list position for position and gives each
entry what a grasp is synthesised from, so two properties carry it and both are asserted here:

* **Order is a contract.** Every object comes in the primary's own order, so a cell whose other cameras
  add nothing produces exactly the list it produced before, position for position.
* **The camera travels with the object.** `camera_id`, `segmentation`, `depth_map` and `intrinsics`
  come from the primary, because the calculator is called with all four and mixing them across cameras
  is a grasp synthesised in a frame nothing was measured in. Only the fused cloud is the other cameras'.

An object only another camera saw is not one of them: letting a camera introduce one left on
2026-10-16 with its code (``promote_unmatched``, never switched on in a shipped tree).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

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
        """Same length, same order, same segmentations, each read from the primary."""
        primary = _view("cam_left", 3)

        objects = build_scene_objects(primary)

        self.assertEqual(3, len(objects))
        for index, obj in enumerate(objects):
            self.assertIs(primary.segmentations[index], obj.segmentation, "order and identity")
            self.assertEqual("cam_left", obj.camera_id)
            self.assertIs(primary.depth_map, obj.depth_map)
            self.assertIs(primary.intrinsics, obj.intrinsics)
            self.assertIsNone(obj.fused_cloud_base_mm)


class EachObjectCarriesItsOwnFusedCloudTests(unittest.TestCase):
    def test_the_fused_clouds_land_on_their_objects_and_none_where_nothing_was_fused(self) -> None:
        """`None` rather than an object's own points where no other camera saw it: the caller then omits the
        generator kwarg and the calculator derives the cloud from the mask and depth it was already given."""
        fused = np.full((7, 3), 1.0)

        objects = build_scene_objects(_view("cam_left", 3), fused_clouds=(None, fused))

        self.assertIsNone(objects[0].fused_cloud_base_mm)
        self.assertIs(fused, objects[1].fused_cloud_base_mm)
        self.assertIsNone(objects[2].fused_cloud_base_mm, "an object past the fused list gained nothing")


class AViewWithoutSegmentationsIsStillUsableTests(unittest.TestCase):
    def test_a_masks_only_view_yields_objects_with_no_segmentation(self) -> None:
        """Such an object cannot be grasped from that camera, and the `None` is what says so rather than a
        stand-in that would fail later and further away."""
        objects = build_scene_objects(_view("cam_left", 1, with_segmentations=False))

        self.assertIsNone(objects[0].segmentation)


if __name__ == "__main__":
    unittest.main()
