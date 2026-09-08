"""The thing that answers "what does the cell look like right now", and when it refuses to.

The failure this guards is not a wrong box. It is a world that looks fine and is out of date: a
planner handed the cell as it was ten seconds ago plans through whatever arrived since, and the
resulting plan is indistinguishable from a plan through empty space at every layer above.

So most of what is asserted here is refusal. No frame, an unstamped frame, a frame past its age, an
arm that cannot say where its own links are: each one has to come back as a snapshot that says it
cannot be planned against, while still carrying the declared world, because the declared world is
the one thing that is true whatever the camera did.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.safety.planning.live_world import (
    CameraView,
    DepthSnapshot,
    LivePlannerWorld,
    WorldVerdict,
    refresh_planner_world,
)
from src.robot.safety.planning.perceived import WorldBuildLimits, WorldBuildTuning
from src.robot.safety.planning.world import planner_cuboid

_FX = _FY = 500.0
_CX = _CY = 100.0
_SHAPE = (200, 200)
_CAMERA_HEIGHT_MM = 1000.0
_CAMERA_TO_BASE = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, _CAMERA_HEIGHT_MM],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_INTRINSICS = np.array([[_FX, 0.0, _CX], [0.0, _FY, _CY], [0.0, 0.0, 1.0]], dtype=np.float64)
_LIMITS = WorldBuildLimits(
    x_mm=(-400.0, 400.0), y_mm=(-400.0, 400.0), z_mm=(-50.0, 900.0), support_plane_top_mm=0.0
)
#: A declared bench, exactly as `build_planner_cuboids` would emit it.
_DECLARED = (planner_cuboid("support_plane", (0.0, 0.0, -25.0), (1600.0, 1600.0, 50.0)),)
#: The arm, standing out of the way of the scene these tests build.
_LINKS = [(0.0, 0.0, 0.0), (0.0, -350.0, 300.0)]


def _scene_with_a_block(*, at_mm: tuple[float, float] = (150.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """A bench with one block on it, and the block's mask."""
    depth = np.full(_SHAPE, _CAMERA_HEIGHT_MM, dtype=np.float64)
    top_depth = _CAMERA_HEIGHT_MM - 80.0
    scale = top_depth / _FX
    rows, cols = np.mgrid[0 : _SHAPE[0], 0 : _SHAPE[1]]
    mask = (
        (np.abs(cols - (_CX + at_mm[0] / scale)) <= 30.0 / scale)
        & (np.abs(rows - (_CY - at_mm[1] / scale)) <= 30.0 / scale)
    )
    depth[mask] = top_depth
    return depth, mask


class _Camera:
    """A depth source under the test's control, counting how often it was asked."""

    def __init__(self, snapshot: DepthSnapshot | None) -> None:
        self.snapshot = snapshot
        self.grabs = 0

    def grab_surface_depth(self) -> DepthSnapshot | None:
        self.grabs += 1
        return self.snapshot


def _world(camera: _Camera, **kwargs: object) -> LivePlannerWorld:
    settings: dict[str, object] = {
        "cameras": (
            CameraView(name="overhead", depth_source=camera, camera_to_base=_CAMERA_TO_BASE),
        ),
        "declared": _DECLARED,
        "limits": _LIMITS,
        "tuning": WorldBuildTuning(max_boxes=4),
        "max_age_ms": 500.0,
    }
    settings.update(kwargs)
    return LivePlannerWorld(**settings)  # type: ignore[arg-type]


class FreshWorldTests(unittest.TestCase):
    def test_a_fresh_frame_yields_the_declared_world_plus_what_was_seen(self) -> None:
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))

        snapshot = _world(camera).world_for(link_origins_mm=_LINKS, now=100.1)

        self.assertIs(snapshot.verdict, WorldVerdict.FRESH)
        self.assertTrue(snapshot.usable)
        self.assertEqual(snapshot.declared_count, 1)
        self.assertEqual(snapshot.perceived_count, 1)
        names = [box["name"] for box in snapshot.cuboids]
        self.assertEqual(names[0], "support_plane", "the declared world must come first and survive")
        self.assertTrue(names[1].startswith("seen_"))
        self.assertAlmostEqual(snapshot.age_ms or 0.0, 100.0, delta=1.0)

    def test_a_second_ask_inside_the_age_reuses_the_reading(self) -> None:
        """A grab is I/O and a conversion is arithmetic. Two plans in one cycle share the reading."""
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))
        world = _world(camera)

        world.world_for(link_origins_mm=_LINKS, now=100.1)
        world.world_for(link_origins_mm=_LINKS, now=100.2)
        self.assertEqual(camera.grabs, 1)

        world.world_for(link_origins_mm=_LINKS, now=101.0)
        self.assertEqual(camera.grabs, 2, "past the age limit the camera has to be asked again")

    def test_the_arm_is_taken_out_of_what_the_camera_saw(self) -> None:
        """One scene, two arm poses. What changes is which points are the robot itself."""
        depth, _ = _scene_with_a_block(at_mm=(150.0, 0.0))
        frame = DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0)

        over_the_block = _world(_Camera(frame)).world_for(
            link_origins_mm=[(150.0, 0.0, 0.0), (150.0, 0.0, 400.0)], now=100.1
        )
        self.assertEqual(over_the_block.perceived_count, 0, over_the_block.render())

        elsewhere = _world(_Camera(frame)).world_for(link_origins_mm=_LINKS, now=100.1)
        self.assertEqual(elsewhere.perceived_count, 1, elsewhere.render())


class RefusalTests(unittest.TestCase):
    def test_no_frame_is_not_planned_against_and_keeps_the_declared_world(self) -> None:
        snapshot = _world(_Camera(None)).world_for(link_origins_mm=_LINKS, now=100.0)

        self.assertIs(snapshot.verdict, WorldVerdict.NO_FRAME)
        self.assertFalse(snapshot.usable)
        self.assertEqual([box["name"] for box in snapshot.cuboids], ["support_plane"])
        self.assertIn("no depth", snapshot.reason)
        self.assertIn("no_frame", snapshot.render())

    def test_an_unstamped_frame_counts_as_unknown_age_rather_than_as_fresh(self) -> None:
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=None))

        snapshot = _world(camera).world_for(link_origins_mm=_LINKS, now=100.0)

        self.assertIs(snapshot.verdict, WorldVerdict.STALE)
        self.assertIn("no capture time", snapshot.reason)

    def test_a_frame_past_the_age_limit_is_refused_and_says_by_how_much(self) -> None:
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))

        snapshot = _world(camera).world_for(link_origins_mm=_LINKS, now=102.0)

        self.assertIs(snapshot.verdict, WorldVerdict.STALE)
        self.assertIn("2000 ms old", snapshot.reason)
        self.assertIn("500 ms", snapshot.reason)

    def test_an_arm_that_cannot_place_its_own_links_gets_no_perceived_world(self) -> None:
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))

        snapshot = _world(camera).world_for(link_origins_mm=None, now=100.1)

        self.assertIs(snapshot.verdict, WorldVerdict.UNUSABLE)
        self.assertIn("its own links", snapshot.reason)
        self.assertEqual([box["name"] for box in snapshot.cuboids], ["support_plane"])

    def test_geometry_that_cannot_be_built_is_reported_rather_than_raised(self) -> None:
        """A plan is about to happen. The caller needs a verdict it can refuse on, not a traceback."""
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))
        limits = WorldBuildLimits(
            x_mm=(-400.0, 400.0), y_mm=(-400.0, 400.0), z_mm=(-50.0, 900.0),
            support_plane_top_mm=None,
        )

        snapshot = _world(camera, limits=limits).world_for(link_origins_mm=_LINKS, now=100.1)

        self.assertIs(snapshot.verdict, WorldVerdict.UNUSABLE)
        self.assertIn("support_plane", snapshot.reason)


class SegmentationTests(unittest.TestCase):
    def test_offered_masks_name_the_boxes_and_leave_the_target_out(self) -> None:
        depth, mask = _scene_with_a_block(at_mm=(150.0, 0.0))
        far_depth, far_mask = _scene_with_a_block(at_mm=(-150.0, 0.0))
        depth = np.minimum(depth, far_depth)
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))
        world = _world(camera)

        both = world.world_for(link_origins_mm=_LINKS, now=100.1)
        self.assertEqual(both.perceived_count, 2, both.render())

        world.offer_segmentation(
            labelled_masks=(("red cube", far_mask),), exclude_masks=(mask,), timestamp=100.0
        )
        one = world.world_for(link_origins_mm=_LINKS, now=100.1)
        self.assertEqual(one.perceived_count, 1, one.render())
        self.assertEqual(one.cuboids[1]["name"], "seen_00_red_cube")

    def test_masks_older_than_the_age_limit_stop_being_used(self) -> None:
        """A target mask from half a second ago is a hole where the object is no longer standing."""
        depth, mask = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))
        world = _world(camera)
        world.offer_segmentation(exclude_masks=(mask,), timestamp=100.0)

        self.assertEqual(world.world_for(link_origins_mm=_LINKS, now=100.1).perceived_count, 0)

        camera.snapshot = DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=102.0)
        self.assertEqual(world.world_for(link_origins_mm=_LINKS, now=102.1).perceived_count, 1)

    def test_forgetting_the_segmentation_puts_the_object_back_in_the_world(self) -> None:
        depth, mask = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))
        world = _world(camera)
        world.offer_segmentation(exclude_masks=(mask,), timestamp=100.0)
        self.assertEqual(world.world_for(link_origins_mm=_LINKS, now=100.1).perceived_count, 0)

        world.forget_segmentation()
        self.assertEqual(world.world_for(link_origins_mm=_LINKS, now=100.1).perceived_count, 1)


class _Planner:
    """A planner client under the test's control, recording what it was handed."""

    def __init__(self, *, confirm_all: bool = True, take_voxels: bool = True) -> None:
        self.worlds: list[list[dict]] = []
        self.meshes: list[list[dict]] = []
        self.voxel_calls: list[dict] = []
        self.confirm_all = confirm_all
        self.take_voxels = take_voxels

    def set_world(self, cuboids: list[dict], meshes: "list[dict] | None" = None) -> int:
        self.worlds.append(list(cuboids))
        self.meshes.append(list(meshes or ()))
        total = len(cuboids) + len(meshes or ())
        return total if self.confirm_all else max(0, total - 1)

    def set_voxels(self, path, *, dims_m, voxel_size_m, pose):  # noqa: ANN001, ANN201
        self.voxel_calls.append(
            {"path": path, "dims_m": list(dims_m), "voxel_size_m": voxel_size_m, "pose": list(pose)}
        )
        if not self.take_voxels:
            return None
        return int(np.load(path).shape[0])


class _OldPlanner:
    """A planner client from before the live scene existed, which must keep working."""

    def __init__(self) -> None:
        self.worlds: list[list[dict]] = []

    def set_world(self, cuboids: list[dict]) -> int:
        self.worlds.append(list(cuboids))
        return len(cuboids)


class RefreshTests(unittest.TestCase):
    """What a driver actually calls: build the world, register it, or refuse the motion."""

    def _source(self, **kwargs: object) -> LivePlannerWorld:
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))
        return _world(camera, **kwargs)

    def test_a_good_refresh_registers_the_world_and_reports_its_price(self) -> None:
        planner = _Planner()
        refresh = refresh_planner_world(
            source=self._source(), client=planner, link_origins_mm=_LINKS, now=100.1
        )

        self.assertTrue(refresh.ok, refresh.render())
        self.assertEqual(len(planner.worlds), 1)
        self.assertEqual([box["name"] for box in planner.worlds[0]][0], "support_plane")
        self.assertGreater(refresh.build_ms, 0.0)
        self.assertEqual(refresh.registered, refresh.sent)
        self.assertIn("planner world refreshed", refresh.render())

    def test_the_live_scene_goes_as_a_file_and_the_declared_world_goes_with_it(self) -> None:
        planner = _Planner()
        source = self._source(tuning=WorldBuildTuning(max_boxes=4, voxel_field_mm=30.0))

        refresh = refresh_planner_world(
            source=source, client=planner, link_origins_mm=_LINKS, now=100.1
        )

        self.assertTrue(refresh.ok, refresh.render())
        self.assertEqual(len(planner.voxel_calls), 1)
        call = planner.voxel_calls[0]
        self.assertTrue(str(call["path"]).endswith(".npy"))
        self.assertAlmostEqual(call["voxel_size_m"], 0.03)
        self.assertEqual(len(call["pose"]), 7, "a pose is position plus a WXYZ quaternion")
        self.assertGreater(refresh.voxels_registered or 0, 0)
        self.assertIn("live scene", refresh.render())
        # The boxes still went, because the guard cannot read a distance field and an operator
        # cannot read one either.
        self.assertGreaterEqual(len(planner.worlds[0]), 2)
        self.assertGreater(len(refresh.guard_boxes), 0)

    def test_a_planner_with_no_live_scene_channel_refuses_rather_than_pretending(self) -> None:
        """A cell that asked for a live scene and cannot have one must not plan as if it had one."""
        planner = _OldPlanner()
        source = self._source(tuning=WorldBuildTuning(max_boxes=4, voxel_field_mm=30.0))

        refresh = refresh_planner_world(
            source=source, client=planner, link_origins_mm=_LINKS, now=100.1
        )

        self.assertFalse(refresh.ok)
        self.assertIn("no live-scene channel", refresh.reason)
        self.assertIn("voxel reservation", refresh.reason)

    def test_a_cell_that_asked_for_no_live_scene_is_untouched_by_any_of_this(self) -> None:
        """The byte-identical path: no field configured, so nothing is sent and nothing refuses."""
        planner = _OldPlanner()

        refresh = refresh_planner_world(
            source=self._source(), client=planner, link_origins_mm=_LINKS, now=100.1
        )

        self.assertTrue(refresh.ok, refresh.render())
        self.assertIsNone(refresh.voxels_registered)
        self.assertEqual(len(planner.worlds), 1)

    def test_a_refused_live_scene_is_a_refused_motion(self) -> None:
        """Asking for a live scene and silently not getting one is the defect this exists to end."""
        planner = _Planner(take_voxels=False)
        source = self._source(tuning=WorldBuildTuning(max_boxes=4, voxel_field_mm=30.0))

        refresh = refresh_planner_world(
            source=source, client=planner, link_origins_mm=_LINKS, now=100.1
        )

        self.assertFalse(refresh.ok)
        self.assertIn("the planner refused it", refresh.reason)
        self.assertEqual(refresh.guard_boxes, (), "a refused refresh must not leave the guard armed")

    def test_a_declared_mesh_reaches_the_planner_beside_the_boxes(self) -> None:
        """A container is the case: as a box it is solid, as a mesh it keeps the hollow."""
        planner = _Planner()
        source = self._source(
            declared_meshes=(
                {
                    "name": "tote",
                    "file_path": "assets/tote.obj",
                    "pose": [0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    "scale": [0.001, 0.001, 0.001],
                },
            )
        )

        refresh = refresh_planner_world(
            source=source, client=planner, link_origins_mm=_LINKS, now=100.1
        )

        self.assertTrue(refresh.ok, refresh.render())
        self.assertEqual([m["name"] for m in planner.meshes[0]], ["tote"])
        self.assertEqual(
            refresh.sent, len(planner.worlds[0]) + 1, "a mesh counts as an obstacle that was sent"
        )

    def test_a_cell_with_no_mesh_never_names_the_parameter(self) -> None:
        """A planner client from before meshes existed takes one argument, and must keep working."""
        planner = _OldPlanner()

        refresh = refresh_planner_world(
            source=self._source(), client=planner, link_origins_mm=_LINKS, now=100.1
        )

        self.assertTrue(refresh.ok, refresh.render())

    def test_a_partial_registration_refuses_and_says_the_two_counts(self) -> None:
        planner = _Planner(confirm_all=False)
        refresh = refresh_planner_world(
            source=self._source(), client=planner, link_origins_mm=_LINKS, now=100.1
        )

        self.assertFalse(refresh.ok)
        self.assertIn("confirmed", refresh.reason)
        self.assertEqual(refresh.guard_boxes, ())

    def test_a_stale_frame_never_reaches_the_planner_at_all(self) -> None:
        planner = _Planner()
        refresh = refresh_planner_world(
            source=self._source(), client=planner, link_origins_mm=_LINKS, now=102.0
        )

        self.assertFalse(refresh.ok)
        self.assertEqual(planner.worlds, [], "nothing is registered from a world nobody can vouch for")
        self.assertEqual(refresh.registered, 0)
        self.assertIn("NOT refreshed", refresh.render())


class LoopHandsOverItsMasksTests(unittest.TestCase):
    """The pick loop is the only thing that knows what the attempt is reaching for.

    Without this the object being grasped is registered as an obstacle like everything else, and the
    planner refuses the one approach the attempt exists for: the goal sits inside a solid.
    """

    class _World:
        def __init__(self) -> None:
            self.offers: list[dict] = []

        def offer_segmentation(self, **kwargs: object) -> None:
            self.offers.append(dict(kwargs))

    class _Arm:
        def __init__(self, world: object) -> None:
            self.live_planner_world = world

    @staticmethod
    def _frame(labels: "tuple[str, ...]") -> object:
        from types import SimpleNamespace

        return SimpleNamespace(
            segmentations=tuple(
                SimpleNamespace(label=label, mask=np.full((4, 4), index, dtype=np.uint8))
                for index, label in enumerate(labels)
            ),
            timestamp=100.0,
        )

    def _offer(self, arm: object, frame: object, target: "int | None") -> None:
        from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator
        from types import SimpleNamespace

        BinPickingOrchestrator._offer_masks_to_planner_world(  # noqa: SLF001
            SimpleNamespace(arm=arm), frame, target
        )

    def test_the_target_is_left_out_and_the_others_are_named(self) -> None:
        world = self._World()
        self._offer(self._Arm(world), self._frame(("red cube", "blue box")), 1)

        (offer,) = world.offers
        self.assertEqual([name for name, _ in offer["labelled_masks"]], ["red cube", "blue box"])
        self.assertEqual(len(offer["exclude_masks"]), 1)
        np.testing.assert_array_equal(
            offer["exclude_masks"][0], np.full((4, 4), 1, dtype=np.uint8)
        )
        self.assertEqual(offer["timestamp"], 100.0, "masks age with the frame, not with this call")

    def test_no_winner_excludes_nothing_and_still_names_what_was_seen(self) -> None:
        world = self._World()
        self._offer(self._Arm(world), self._frame(("red cube",)), None)

        (offer,) = world.offers
        self.assertEqual(offer["exclude_masks"], [])
        self.assertEqual(len(offer["labelled_masks"]), 1)

    def test_an_unnamed_segmentation_still_gets_a_name(self) -> None:
        """A refusal that says `object_2` beats one that says nothing at all."""
        from types import SimpleNamespace

        world = self._World()
        frame = SimpleNamespace(
            segmentations=(SimpleNamespace(label="", mask=np.zeros((4, 4), dtype=np.uint8)),),
            timestamp=100.0,
        )
        self._offer(self._Arm(world), frame, 0)

        self.assertEqual(world.offers[0]["labelled_masks"][0][0], "object_0")

    def test_an_arm_with_no_live_world_costs_one_lookup(self) -> None:
        """The byte-identical path, which is every cell that did not ask for a live world."""
        from types import SimpleNamespace

        self._offer(SimpleNamespace(live_planner_world=None), self._frame(("x",)), 0)
        self._offer(SimpleNamespace(), self._frame(("x",)), 0)


if __name__ == "__main__":
    unittest.main()
