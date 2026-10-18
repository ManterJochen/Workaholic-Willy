"""Multi-camera geometry fusion in the pick path: what runs, and what it says when it cannot.

The measured lever is large (top-1 43.50 % single-view against 55.93 % fused on the datagen
reference), which is exactly why the dangerous state is not "the cell fuses no camera" but "the
operator believes the cell fuses and it is quietly running on one camera". A dropped trigger, a dirty
lens or a calibration artifact that stopped loading all produce a cell that looks healthy from the
outside and is simply worse at grasping.

So these tests pin three things: a cell that fuses no camera is untouched, a working rig really does
hand every candidate object its own fused cloud, and every way the rig can fall short is loud --
never silent. Which cameras a cell fuses is its camera section's (every enabled RGB-D rig that
declares its calibration, where two or more do), so here it is the resolver map the overlay builds
for them.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from src.geometry import Frame, Transform
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator
from src.robot.grasping.multiview.scene_geometry import FusedSceneGeometry, ObservedView, fuse_scene_geometry
from src.robot.grasping.types.perception import CameraObservation, PerceptionFrame

_LOGGER = "src.robot.grasping.loop.pick_loop"


def _frame(*, boxes: tuple[tuple[int, int], ...] = ((10, 22),)) -> PerceptionFrame:
    """A depth frame with one square mask per entry in ``boxes``."""

    segmentations = []
    for lo, hi in boxes:
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[lo:hi, lo:hi] = 1
        segmentations.append(SimpleNamespace(mask=mask))
    return PerceptionFrame(
        depth_map=np.full((32, 32), 500.0, dtype=np.float64),
        intrinsics=np.array(
            [[400.0, 0.0, 16.0], [0.0, 400.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float64
        ),
        segmentations=tuple(segmentations),
    )


class _Resolver:
    """A fixed camera whose CAMERA->BASE is the identity."""

    def __init__(self, *, resolves: bool = True) -> None:
        self._resolves = resolves

    def camera_to_base_for_frame(self, _frame_in, *, arm=None):  # noqa: ANN001, ANN202
        if not self._resolves:
            return None
        return Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE)


class _Rig:
    def __init__(self, observations: tuple[CameraObservation, ...]) -> None:
        self._observations = observations
        self.calls = 0

    def acquire_all(self) -> tuple[CameraObservation, ...]:
        self.calls += 1
        return self._observations


def _orchestrator(**kwargs) -> BinPickingOrchestrator:
    return BinPickingOrchestrator(
        arm=SimpleNamespace(),  # type: ignore[arg-type]
        calculator=SimpleNamespace(),  # type: ignore[arg-type]
        perception=SimpleNamespace(acquire=lambda: _frame()),  # type: ignore[arg-type]
        **kwargs,
    )


def _fused(primary_boxes: tuple[tuple[int, int], ...], other_boxes: tuple[tuple[int, int], ...],
           **kwargs) -> FusedSceneGeometry:
    """The fusion itself, of a primary frame and one other camera at the same placement."""
    primary, other = _frame(boxes=primary_boxes), _frame(boxes=other_boxes)
    view = ObservedView(name="left", masks=tuple(seg.mask for seg in other.segmentations),
                        depth_map=other.depth_map, intrinsics=other.intrinsics, camera_to_base=np.eye(4))
    return fuse_scene_geometry(tuple(seg.mask for seg in primary.segmentations), primary.depth_map,
                               primary.intrinsics, np.eye(4), [view], **kwargs)


_IDENTITY = Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE)


class DefaultPathTests(unittest.TestCase):
    """A cell that fuses no camera: no resolver map, nothing observed."""

    def test_a_cell_that_fuses_no_camera_never_touches_the_rig(self) -> None:
        for resolvers in (None, {}):
            with self.subTest(resolvers=resolvers):
                rig = _Rig(())
                orch = _orchestrator(multi_camera_perception=rig, camera_frame_resolvers=resolvers)

                self.assertIsNone(orch._fused_scene(_frame(), _IDENTITY))
                self.assertEqual(rig.calls, 0, "the rig was observed although the cell fuses no camera")


class WorkingRigTests(unittest.TestCase):
    """A rig that delivers really does enlarge every object's cloud."""

    def _two_camera_orchestrator(self, *, boxes=((10, 22),)) -> BinPickingOrchestrator:
        rig = _Rig((CameraObservation(camera_id="left", frame=_frame(boxes=boxes)),))
        return _orchestrator(multi_camera_perception=rig, camera_frame_resolvers={"left": _Resolver()})

    def test_every_object_gets_its_own_fused_cloud(self) -> None:
        """Not just a labelled target -- the whole reason this is not the old single-target seam."""

        orch = self._two_camera_orchestrator(boxes=((10, 22), (24, 30)))

        fused = orch._fused_scene(_frame(boxes=((10, 22), (24, 30))), _IDENTITY)

        assert fused is not None
        clouds = fused.clouds_base_mm
        self.assertEqual(len(clouds), 2, "one fused cloud per segmentation of the primary frame")
        for index, cloud in enumerate(clouds):
            self.assertIsNotNone(cloud, f"object {index} gained nothing from the second camera")

    def test_the_fused_cloud_is_larger_than_the_single_view_one(self) -> None:
        orch = self._two_camera_orchestrator()
        primary = _frame()

        fused = orch._fused_scene(primary, _IDENTITY)

        assert fused is not None
        cloud = fused.cloud_for(0)
        assert cloud is not None
        single_view_points = int(np.count_nonzero(primary.segmentations[0].mask))
        self.assertGreater(len(cloud), single_view_points)

    def test_telemetry_stamps_what_ran(self) -> None:
        orch = self._two_camera_orchestrator()

        orch._fused_scene(_frame(), _IDENTITY)

        stamp = orch._fusion_geometry_telemetry
        self.assertEqual(stamp["fused_views_used"], 1)
        self.assertEqual(stamp["fused_views"], ["left"])
        self.assertEqual(stamp["fused_objects"], 1)

    def test_an_object_no_other_camera_sees_is_reported_as_none(self) -> None:
        """None, not a copy of its own cloud -- so the caller omits the kwarg entirely."""

        rig = _Rig((CameraObservation(camera_id="left", frame=_frame(boxes=((24, 30),))),))
        orch = _orchestrator(multi_camera_perception=rig, camera_frame_resolvers={"left": _Resolver()})

        fused = orch._fused_scene(_frame(boxes=((2, 6),)), _IDENTITY)

        assert fused is not None
        self.assertIsNone(fused.cloud_for(0))


class TheIndexIsTheSegmentationIndexTests(unittest.TestCase):
    """The one that would have returned the wrong object's surface, in silence.

    `cloud_for(idx)` is addressed by segmentation index at the candidate loop, and the primary mask
    list used to be compacted: a segmentation with no mask was dropped, so every later object shifted
    down one slot. The lookup is bounds-safe, so nothing raises; object two would simply be planned
    against object three's surface. Latent today, because `SegmentationResult.mask` is not optional,
    which is why it is worth a test rather than a comment: the day someone makes it optional, this is
    what tells them.
    """

    def test_a_mask_less_segmentation_does_not_shift_the_ones_after_it(self) -> None:
        rig = _Rig((CameraObservation(camera_id="left", frame=_frame(boxes=((10, 22), (24, 30)))),))
        orch = _orchestrator(multi_camera_perception=rig, camera_frame_resolvers={"left": _Resolver()})
        primary = _frame(boxes=((10, 22), (24, 30)))
        # A segmentation with no mask in the MIDDLE, which is the position that shifts the rest.
        holed = replace(primary, segmentations=(
            primary.segmentations[0], SimpleNamespace(mask=None), primary.segmentations[1]))

        fused = orch._fused_scene(holed, _IDENTITY)

        assert fused is not None
        self.assertEqual(3, len(fused.clouds_base_mm), "one slot per segmentation, in order")
        self.assertIsNone(fused.cloud_for(1), "the mask-less one owns its slot and fuses nothing")
        self.assertIsNotNone(fused.cloud_for(2), "the object after it kept its own index")


class MappedCameraRigTests(unittest.TestCase):
    """The adapter the sim runners and the first real rigs reach for."""

    def test_it_satisfies_the_protocol_and_names_every_camera(self) -> None:
        from src.robot.grasping.types.perception import (
            MappedCameraRig,
            MultiCameraPerceptionSource,
        )

        rig = MappedCameraRig(
            {"overhead": SimpleNamespace(acquire=_frame), "left": SimpleNamespace(acquire=_frame)}
        )

        self.assertIsInstance(rig, MultiCameraPerceptionSource)
        observations = rig.acquire_all()
        self.assertEqual([o.camera_id for o in observations], ["overhead", "left"])

    def test_order_follows_the_mapping_so_it_is_deterministic(self) -> None:
        from src.robot.grasping.types.perception import MappedCameraRig

        rig = MappedCameraRig(
            {"b": SimpleNamespace(acquire=_frame), "a": SimpleNamespace(acquire=_frame)}
        )

        self.assertEqual([o.camera_id for o in rig.acquire_all()], ["b", "a"])

    def test_it_drives_the_fusion_end_to_end(self) -> None:
        """The adapter is only worth having if the orchestrator accepts it as the rig."""

        from src.robot.grasping.types.perception import MappedCameraRig

        orch = _orchestrator(
            multi_camera_perception=MappedCameraRig({"left": SimpleNamespace(acquire=_frame)}),
            camera_frame_resolvers={"left": _Resolver()},
        )

        fused = orch._fused_scene(_frame(), _IDENTITY)

        assert fused is not None
        self.assertIsNotNone(fused.cloud_for(0))
        self.assertEqual(orch._fusion_geometry_telemetry["fused_views"], ["left"])


class TheNeighbourCloudsTests(unittest.TestCase):
    """The obstacle half of the same observation, which a wrist pick's earlier looks hand the collision filter
    (``with_neighbours``).

    Fusing the target surface alone left ``finger_collision`` at 60.0 % of every remaining mis-rank
    while the neighbour-free ``sparse`` family ranked perfectly, so what is missing is not a better
    score -- it is the neighbour the synthesis view never saw. These pin that the fusion delivers
    that neighbour, and, more importantly, that it never delivers the TARGET as its own obstacle. The fixed
    cameras hand the filter none: on the reference they rescued 5 grasps and destroyed 4 over 1073 objects, and
    the switch that let them (``neighbour_scene_enabled``) left with the fusion keys on 2026-10-16.
    """

    def test_the_fixed_cameras_hand_the_collision_filter_no_neighbour(self) -> None:
        rig = _Rig((CameraObservation(camera_id="left", frame=_frame(boxes=((10, 22), (24, 30)))),))
        orch = _orchestrator(multi_camera_perception=rig, camera_frame_resolvers={"left": _Resolver()})

        fused = orch._fused_scene(_frame(boxes=((10, 22),)), _IDENTITY)

        assert fused is not None
        self.assertEqual(fused.neighbour_clouds_base_mm, ())
        self.assertIsNone(fused.neighbour_for(0))
        self.assertNotIn("fused_neighbour_points", orch._fusion_geometry_telemetry)

    def test_the_other_camera_s_extra_object_becomes_an_obstacle(self) -> None:
        """The whole point: a collider the synthesis view never saw."""

        fused = _fused(((10, 22),), ((10, 22), (24, 30)), with_neighbours=True)

        neighbour = fused.neighbour_for(0)
        assert neighbour is not None
        self.assertGreater(len(neighbour), 0)
        self.assertGreater(fused.neighbour_points, 0)

    def test_the_target_is_never_its_own_obstacle(self) -> None:
        """Points where the fingers must close would refuse nearly every candidate."""

        fused = _fused(((10, 22),), ((10, 22), (24, 30)), with_neighbours=True)

        neighbour = fused.neighbour_for(0)
        target = fused.cloud_for(0)
        assert neighbour is not None and target is not None
        # The target's own square projects to x/y within the first box; the obstacle cloud must
        # come entirely from the second one.
        overlapping = np.array(
            [np.any(np.all(np.isclose(target, point, atol=1e-6), axis=1)) for point in neighbour]
        )
        self.assertFalse(
            bool(overlapping.any()),
            "the target's own surface leaked into the cloud it is collision-checked against",
        )

    def test_an_unconfirmed_target_is_still_excluded(self) -> None:
        """The asymmetric rule, and the reason it exists.

        The measured association abstains on 8.5 % of (object, view) pairs. The other camera here
        sees a partial view of the same object (0.69 overlap under a 1 mm tolerance), so a
        ``min_score`` of 1.0 makes the assignment give up -- but the blob is still the target, so
        it must still be kept OUT of the obstacle cloud rather than becoming the one thing
        guaranteed to sit between the fingers.
        """

        fused = _fused(((10, 22),), ((10, 20),), with_neighbours=True, min_score=1.0, neighbour_mm=1.0)

        self.assertIsNone(fused.cloud_for(0), "min_score=1.0 should refuse to FUSE this pair")
        self.assertIsNone(
            fused.neighbour_for(0),
            "the refused match is still the target and must not become its own obstacle",
        )


class TheOtherCamerasVouchForTheWayInTests(unittest.TestCase):
    """The owner, 2026-10-16: several cameras joined. The other cameras the pick fused reach the calculator's scene views
    beside the pick's looks, so a point any of them saw counts as seen (``SeenInAnyFrame``): a fixed camera at a slant
    vouches for the way under a tilted part."""

    def test_the_fused_cameras_view_reaches_the_calculator(self) -> None:
        from src.robot.grasping.types.feedback import GraspResult

        class _Recording:
            render_debug_images = False

            def __init__(self) -> None:
                self.kwargs: list[dict] = []

            def compute_result(self, *_args: object, **kwargs: object) -> GraspResult:
                self.kwargs.append(dict(kwargs))
                return GraspResult(candidates=(), reasons=(), telemetry={})

        calculator = _Recording()
        rig = _Rig((CameraObservation(camera_id="left", frame=_frame(boxes=((10, 22),))),))
        orch = BinPickingOrchestrator(
            arm=SimpleNamespace(),  # type: ignore[arg-type]
            calculator=calculator,  # type: ignore[arg-type]
            perception=SimpleNamespace(acquire=lambda: _frame()),  # type: ignore[arg-type]
            frame_resolver=SimpleNamespace(  # type: ignore[arg-type]
                camera_to_base_for_frame=lambda *_a, **_k: _IDENTITY
            ),
            multi_camera_perception=rig,
            camera_frame_resolvers={"left": _Resolver()},
        )

        orch._best_result_over_segmentations(_frame(boxes=((10, 22),)))

        views = calculator.kwargs[0].get("scene_views")
        self.assertEqual(1, len(views or ()), "the fused camera's view did not reach the calculator")
        np.testing.assert_allclose(_frame().depth_map, views[0].depth_mm)


class StandDownIsNeverSilentTests(unittest.TestCase):
    """Every way the rig can fall short has to be loud."""

    def test_a_fused_cell_without_a_rig_warns(self) -> None:
        orch = _orchestrator(camera_frame_resolvers={"left": _Resolver()})

        with self.assertLogs(_LOGGER, level="WARNING") as captured:
            self.assertIsNone(orch._fused_scene(_frame(), _IDENTITY))

        # The reason half distinguishes this branch from the next test.
        self.assertTrue(any("it is running single-view" in line
                            and "no multi-camera perception source is wired" in line
                            for line in captured.output))

    def test_a_fused_cell_without_a_transform_warns(self) -> None:
        orch = _orchestrator(multi_camera_perception=_Rig(()), camera_frame_resolvers={"left": _Resolver()})

        with self.assertLogs(_LOGGER, level="WARNING") as captured:
            self.assertIsNone(orch._fused_scene(_frame(), None))

        self.assertTrue(any("it is running single-view" in line
                            and "no CAMERA->BASE transform is available" in line
                            for line in captured.output))

    def test_a_missing_camera_warns_and_the_pick_goes_on(self) -> None:
        rig = _Rig((CameraObservation(camera_id="left", frame=_frame()),))
        orch = _orchestrator(
            multi_camera_perception=rig,
            camera_frame_resolvers={"left": _Resolver(), "right": _Resolver()},
        )

        with self.assertLogs(_LOGGER, level="WARNING") as captured:
            fused = orch._fused_scene(_frame(), _IDENTITY)

        self.assertTrue(any("right" in line for line in captured.output))
        self.assertIsNotNone(fused, "the cameras that did deliver are still fused")

    def test_a_camera_with_no_calibration_entry_is_dropped_loudly(self) -> None:
        """Guessing a default extrinsic would fuse a real surface into the wrong place."""

        rig = _Rig((CameraObservation(camera_id="unknown_cam", frame=_frame()),))
        orch = _orchestrator(multi_camera_perception=rig, camera_frame_resolvers={"left": _Resolver()})

        with self.assertLogs(_LOGGER, level="WARNING") as captured:
            self.assertIsNone(orch._fused_scene(_frame(), _IDENTITY))

        self.assertTrue(any("unknown_cam" in line for line in captured.output))

    def test_a_camera_that_resolves_no_transform_is_dropped_loudly(self) -> None:
        rig = _Rig((CameraObservation(camera_id="left", frame=_frame()),))
        orch = _orchestrator(multi_camera_perception=rig, camera_frame_resolvers={"left": _Resolver(resolves=False)})

        with self.assertLogs(_LOGGER, level="WARNING") as captured:
            self.assertIsNone(orch._fused_scene(_frame(), _IDENTITY))

        self.assertTrue(any("left" in line for line in captured.output))


if __name__ == "__main__":
    unittest.main()
