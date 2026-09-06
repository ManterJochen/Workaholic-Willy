"""Multi-camera geometry fusion in the pick path: what runs, and what it says when it cannot.

The measured lever is large (top-1 43.50 % single-view against 55.93 % fused on the datagen
reference), which is exactly why the dangerous state is not "fusion is off" but "the operator
believes fusion is on and the cell is quietly running on one camera". A dropped trigger, a dirty
lens or a calibration artifact that stopped loading all produce a cell that looks healthy from the
outside and is simply worse at grasping.

So these tests pin three things: the default path is untouched, a working rig really does hand every
candidate object its own fused cloud, and every way the rig can fall short is either loud or refused
-- never silent.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.config.schema.robot.grasping_schema import FusionGeometryConfig
from src.geometry import Frame, Transform
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator
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


def _enabled(**overrides) -> FusionGeometryConfig:
    return FusionGeometryConfig(enabled=True, **overrides)


_IDENTITY = Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE)


class DefaultPathTests(unittest.TestCase):
    """Off is off: no config, no rig, nothing observed."""

    def test_no_config_returns_none_without_touching_the_rig(self) -> None:
        rig = _Rig(())
        orch = _orchestrator(multi_camera_perception=rig)

        self.assertIsNone(orch._fused_scene(_frame(), _IDENTITY))
        self.assertEqual(rig.calls, 0, "the rig was observed although fusion is off")

    def test_disabled_config_returns_none(self) -> None:
        rig = _Rig(())
        orch = _orchestrator(
            multi_camera_perception=rig, fusion_geometry_config=FusionGeometryConfig()
        )

        self.assertIsNone(orch._fused_scene(_frame(), _IDENTITY))
        self.assertEqual(rig.calls, 0)


class WorkingRigTests(unittest.TestCase):
    """A rig that delivers really does enlarge every object's cloud."""

    def _two_camera_orchestrator(self, *, boxes=((10, 22),)) -> BinPickingOrchestrator:
        rig = _Rig((CameraObservation(camera_id="left", frame=_frame(boxes=boxes)),))
        return _orchestrator(
            multi_camera_perception=rig,
            camera_frame_resolvers={"left": _Resolver()},
            fusion_geometry_config=_enabled(),
        )

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

    def test_telemetry_stamps_what_ran_not_what_was_configured(self) -> None:
        orch = self._two_camera_orchestrator()

        orch._fused_scene(_frame(), _IDENTITY)

        stamp = orch._fusion_geometry_telemetry
        self.assertEqual(stamp["fused_views_used"], 1)
        self.assertEqual(stamp["fused_views"], ["left"])
        self.assertEqual(stamp["fused_objects"], 1)

    def test_an_object_no_other_camera_sees_is_reported_as_none(self) -> None:
        """None, not a copy of its own cloud -- so the caller omits the kwarg entirely."""

        rig = _Rig((CameraObservation(camera_id="left", frame=_frame(boxes=((24, 30),))),))
        orch = _orchestrator(
            multi_camera_perception=rig,
            camera_frame_resolvers={"left": _Resolver()},
            fusion_geometry_config=_enabled(),
        )

        fused = orch._fused_scene(_frame(boxes=((2, 6),)), _IDENTITY)

        assert fused is not None
        self.assertIsNone(fused.cloud_for(0))


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
            fusion_geometry_config=_enabled(),
        )

        fused = orch._fused_scene(_frame(), _IDENTITY)

        assert fused is not None
        self.assertIsNotNone(fused.cloud_for(0))
        self.assertEqual(orch._fusion_geometry_telemetry["fused_views"], ["left"])


class NeighbourSceneTests(unittest.TestCase):
    """The obstacle half of the same observation.

    Fusing the target surface alone left ``finger_collision`` at 60.0 % of every remaining mis-rank
    while the neighbour-free ``sparse`` family ranked perfectly, so what is missing is not a better
    score -- it is the neighbour the synthesis view never saw. These pin that the switch delivers
    that neighbour, and, more importantly, that it never delivers the TARGET as its own obstacle.
    """

    def _rig_orchestrator(self, *, neighbours: bool, other_boxes) -> BinPickingOrchestrator:
        rig = _Rig((CameraObservation(camera_id="left", frame=_frame(boxes=other_boxes)),))
        return _orchestrator(
            multi_camera_perception=rig,
            camera_frame_resolvers={"left": _Resolver()},
            fusion_geometry_config=_enabled(neighbour_scene_enabled=neighbours),
        )

    def test_off_by_default_costs_nothing(self) -> None:
        orch = self._rig_orchestrator(neighbours=False, other_boxes=((10, 22), (24, 30)))

        fused = orch._fused_scene(_frame(boxes=((10, 22),)), _IDENTITY)

        assert fused is not None
        self.assertEqual(fused.neighbour_clouds_base_mm, ())
        self.assertIsNone(fused.neighbour_for(0))
        self.assertNotIn("fused_neighbour_points", orch._fusion_geometry_telemetry)

    def test_the_other_camera_s_extra_object_becomes_an_obstacle(self) -> None:
        """The whole point: a collider the synthesis view never saw."""

        orch = self._rig_orchestrator(neighbours=True, other_boxes=((10, 22), (24, 30)))

        fused = orch._fused_scene(_frame(boxes=((10, 22),)), _IDENTITY)

        assert fused is not None
        neighbour = fused.neighbour_for(0)
        assert neighbour is not None
        self.assertGreater(len(neighbour), 0)
        self.assertGreater(orch._fusion_geometry_telemetry["fused_neighbour_points"], 0)

    def test_the_target_is_never_its_own_obstacle(self) -> None:
        """Points where the fingers must close would refuse nearly every candidate."""

        orch = self._rig_orchestrator(neighbours=True, other_boxes=((10, 22), (24, 30)))
        primary = _frame(boxes=((10, 22),))

        fused = orch._fused_scene(primary, _IDENTITY)

        assert fused is not None
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

        rig = _Rig((CameraObservation(camera_id="left", frame=_frame(boxes=((10, 20),))),))
        orch = _orchestrator(
            multi_camera_perception=rig,
            camera_frame_resolvers={"left": _Resolver()},
            fusion_geometry_config=_enabled(
                neighbour_scene_enabled=True, min_score=1.0, neighbour_mm=1.0
            ),
        )

        fused = orch._fused_scene(_frame(boxes=((10, 22),)), _IDENTITY)

        assert fused is not None
        self.assertIsNone(fused.cloud_for(0), "min_score=1.0 should refuse to FUSE this pair")
        self.assertIsNone(
            fused.neighbour_for(0),
            "the refused match is still the target and must not become its own obstacle",
        )

    def test_it_reaches_the_calculator_as_a_camera_frame_cloud(self) -> None:
        """Wiring, not intent: the collision filter plans in CAMERA mm, so the cloud must be too."""

        from src.robot.grasping.types.feedback import GraspResult

        class _Recording:
            render_debug_images = False

            def __init__(self) -> None:
                self.kwargs: list[dict] = []

            def compute_result(self, *_args: object, **kwargs: object) -> GraspResult:
                self.kwargs.append(dict(kwargs))
                return GraspResult(candidates=(), reasons=(), telemetry={})

        calculator = _Recording()
        rig = _Rig((CameraObservation(camera_id="left", frame=_frame(boxes=((10, 22), (24, 30)))),))
        # CAMERA->BASE is a pure 100 mm translation in x, so a BASE cloud that was NOT mapped back
        # would land 100 mm away from the camera-frame points the filter compares it with.
        shifted = np.eye(4, dtype=np.float64)
        shifted[0, 3] = 100.0
        camera_to_base = Transform.from_matrix(
            shifted, from_frame=Frame.CAMERA, to_frame=Frame.BASE
        )
        orch = BinPickingOrchestrator(
            arm=SimpleNamespace(),  # type: ignore[arg-type]
            calculator=calculator,  # type: ignore[arg-type]
            perception=SimpleNamespace(acquire=lambda: _frame()),  # type: ignore[arg-type]
            frame_resolver=SimpleNamespace(  # type: ignore[arg-type]
                camera_to_base_for_frame=lambda *_a, **_k: camera_to_base
            ),
            multi_camera_perception=rig,
            camera_frame_resolvers={"left": _Resolver()},
            fusion_geometry_config=_enabled(neighbour_scene_enabled=True),
        )

        orch._best_result_over_segmentations(_frame(boxes=((10, 22),)))

        self.assertEqual(len(calculator.kwargs), 1)
        scene = calculator.kwargs[0].get("scene_points_mm")
        self.assertIsNotNone(scene, "the fused neighbour cloud never reached the collision filter")
        assert scene is not None
        # Identity resolver for "left" => its points are BASE == its own camera frame; the primary
        # camera sits 100 mm along +x, so mapping back must subtract that.
        self.assertLess(float(np.max(scene[:, 0])), 0.0)


class StandDownIsNeverSilentTests(unittest.TestCase):
    """Every way the rig can fall short has to be loud or refused."""

    def test_enabled_without_a_rig_warns(self) -> None:
        orch = _orchestrator(fusion_geometry_config=_enabled())

        with self.assertLogs(_LOGGER, level="WARNING") as captured:
            self.assertIsNone(orch._fused_scene(_frame(), _IDENTITY))

        # pick_loop.py:1869 now warns "... the cell is running single-view"; the migration
        # lower-cased the shout. The reason half distinguishes this branch from the next test.
        self.assertTrue(any("the cell is running single-view" in line
                            and "no multi-camera perception source is wired" in line
                            for line in captured.output))

    def test_enabled_without_a_transform_warns(self) -> None:
        orch = _orchestrator(
            multi_camera_perception=_Rig(()), fusion_geometry_config=_enabled()
        )

        with self.assertLogs(_LOGGER, level="WARNING") as captured:
            self.assertIsNone(orch._fused_scene(_frame(), None))

        self.assertTrue(any("the cell is running single-view" in line
                            and "no CAMERA->BASE transform is available" in line
                            for line in captured.output))

    def test_a_missing_camera_warns_under_degrade(self) -> None:
        rig = _Rig((CameraObservation(camera_id="left", frame=_frame()),))
        orch = _orchestrator(
            multi_camera_perception=rig,
            camera_frame_resolvers={"left": _Resolver(), "right": _Resolver()},
            fusion_geometry_config=_enabled(on_camera_unavailable="degrade"),
        )

        with self.assertLogs(_LOGGER, level="WARNING") as captured:
            fused = orch._fused_scene(_frame(), _IDENTITY)

        self.assertTrue(any("right" in line for line in captured.output))
        self.assertIsNotNone(fused, "degrade must still fuse with the cameras that did deliver")

    def test_a_missing_camera_refuses_under_refuse(self) -> None:
        rig = _Rig((CameraObservation(camera_id="left", frame=_frame()),))
        orch = _orchestrator(
            multi_camera_perception=rig,
            camera_frame_resolvers={"left": _Resolver(), "right": _Resolver()},
            fusion_geometry_config=_enabled(on_camera_unavailable="refuse"),
        )

        with self.assertRaises(RuntimeError) as ctx:
            orch._fused_scene(_frame(), _IDENTITY)

        self.assertIn("right", str(ctx.exception))

    def test_a_camera_with_no_calibration_entry_is_dropped_loudly(self) -> None:
        """Guessing a default extrinsic would fuse a real surface into the wrong place."""

        rig = _Rig((CameraObservation(camera_id="unknown_cam", frame=_frame()),))
        orch = _orchestrator(
            multi_camera_perception=rig,
            camera_frame_resolvers={},
            fusion_geometry_config=_enabled(),
        )

        with self.assertLogs(_LOGGER, level="WARNING") as captured:
            self.assertIsNone(orch._fused_scene(_frame(), _IDENTITY))

        self.assertTrue(any("unknown_cam" in line for line in captured.output))

    def test_a_camera_that_resolves_no_transform_is_dropped_loudly(self) -> None:
        rig = _Rig((CameraObservation(camera_id="left", frame=_frame()),))
        orch = _orchestrator(
            multi_camera_perception=rig,
            camera_frame_resolvers={"left": _Resolver(resolves=False)},
            fusion_geometry_config=_enabled(),
        )

        with self.assertLogs(_LOGGER, level="WARNING") as captured:
            self.assertIsNone(orch._fused_scene(_frame(), _IDENTITY))

        self.assertTrue(any("left" in line for line in captured.output))


if __name__ == "__main__":
    unittest.main()
