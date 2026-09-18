"""A pick's target leaves the planner world it plans against: the keep-out primitives, and the pick loop.

The world fits the target's box with the plane, limits, clustering, margin and floor it fits every obstacle with, so a
target leaves exactly the space it would have filled, and the box leaves it out of every camera, fixed or on the wrist.
An offer that cannot be dated or placed is refused before it reaches a world.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.contracts import UNSET
from src.robot.core.keep_out import KeepOutBox, SegmentationOffer
from src.robot.safety.planning.perceived import WorldBuildLimits, WorldBuildTuning, target_keep_out_box

_LIMITS = WorldBuildLimits(x_mm=(-800.0, 800.0), y_mm=(-800.0, 800.0), z_mm=(-50.0, 900.0), support_plane_top_mm=0.0)


def _block_cloud() -> np.ndarray:
    """The top face of a 60 by 40 mm block standing 50 mm high at (430, 0), a bench row bleeding into it, a stray."""
    xs, ys = np.meshgrid(np.arange(400.0, 460.1, 5.0), np.arange(-20.0, 20.1, 5.0))
    top = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 50.0)])
    bench_row = np.column_stack([np.arange(380.0, 480.1, 5.0), np.full(21, -30.0), np.full(21, 2.0)])
    flying = np.array([[600.0, 300.0, 400.0]])
    return np.vstack([top, bench_row, flying])


class TheTargetBoxTests(unittest.TestCase):
    def test_a_target_box_is_the_box_the_world_would_fit(self) -> None:
        box = target_keep_out_box(_block_cloud(), name="target", limits=_LIMITS, tuning=WorldBuildTuning())
        assert box is not None
        matrix = box.matrix()
        # 60 by 40 mm grown by the 15 mm margin on each side; from the bench top (the floor) to 50 mm, grown by 15.
        np.testing.assert_allclose(matrix[:3, 3], (430.0, 0.0, 25.0), atol=1e-9)
        np.testing.assert_allclose(box.half_extents_mm, (45.0, 35.0, 40.0), atol=1e-9)
        self.assertAlmostEqual(abs(matrix[0, 0]), 1.0, places=9, msg="the long side lies along BASE X")
        self.assertAlmostEqual(matrix[2, 2], 1.0, places=9, msg="the box stands upright")
        self.assertEqual("target", box.name)

    def test_no_floor_when_floor_to_plane_is_off(self) -> None:
        box = target_keep_out_box(_block_cloud(), name="target", limits=_LIMITS,
                                  tuning=WorldBuildTuning(floor_to_plane=False))
        assert box is not None
        self.assertAlmostEqual(box.matrix()[2, 3], 50.0)
        self.assertAlmostEqual(box.half_extents_mm[2], 15.0)

    def test_nothing_above_the_plane_gives_no_box(self) -> None:
        bench = np.column_stack([np.arange(0.0, 50.0, 5.0), np.zeros(10), np.full(10, 3.0)])
        self.assertIsNone(target_keep_out_box(bench, name="target", limits=_LIMITS, tuning=WorldBuildTuning()))

    def test_a_box_contains_its_faces_and_turns_with_its_yaw(self) -> None:
        turn = np.eye(4)
        c, s = math.cos(math.radians(90.0)), math.sin(math.radians(90.0))
        turn[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        turn[:3, 3] = (100.0, 0.0, 0.0)
        box = KeepOutBox.from_matrix("turned", turn, (30.0, 10.0, 5.0))
        inside = box.contains([[100.0, 30.0, 0.0], [100.0, 31.0, 0.0], [130.0, 0.0, 0.0], [110.0, 0.0, 5.0]])
        self.assertEqual([True, False, False, True], inside.tolist())


class AnOfferThatCannotBeDatedOrPlacedIsRefusedTests(unittest.TestCase):
    def test_an_offer_that_cannot_be_dated_or_placed_is_refused(self) -> None:
        mask = np.zeros((4, 4), dtype=bool)
        rows = (
            ("no capture time", dict(captured_at_s=float("nan"))),
            ("a negative capture time", dict(captured_at_s=-1.0)),
            ("a blank camera", dict(captured_at_s=1.0, camera=" ")),
            ("masks with no camera", dict(captured_at_s=1.0, exclude_masks=(mask,))),
            ("labels with no camera", dict(captured_at_s=1.0, labelled_masks=(("cube", mask),))),
            ("points that are not (N, 3)", dict(captured_at_s=1.0, target_points_base_mm=np.zeros((4, 2)))),
            ("points that are not finite", dict(captured_at_s=1.0, target_points_base_mm=np.array([[0.0, np.inf, 0.0]]))),
        )
        for label, kwargs in rows:
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    SegmentationOffer(**kwargs)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            SegmentationOffer()  # type: ignore[call-arg]
        box_only = SegmentationOffer(captured_at_s=1.0, target_points_base_mm=np.zeros((3, 3)), target_label="cube")
        self.assertIs(UNSET, box_only.camera)
        self.assertEqual({"camera": None, "labels": [], "exclude_masks": 0, "target_label": "cube", "target_points": 3,
                          "captured_at_s": 1.0}, box_only.to_dict())

    def test_a_box_that_is_not_rigid_or_has_no_size_is_refused(self) -> None:
        sheared = np.eye(4)
        sheared[0, 1] = 0.5
        for label, args in (("sheared", ("b", sheared, (1.0, 1.0, 1.0))), ("negative size", ("b", np.eye(4), (1.0, -1.0, 1.0))),
                            ("blank name", (" ", np.eye(4), (1.0, 1.0, 1.0)))):
            with self.subTest(label), self.assertRaises(ValueError):
                KeepOutBox.from_matrix(*args)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------------
# One keep-out path for the pick loop
# ---------------------------------------------------------------------------------------------------


class _HoldingWorld:
    """A live world double that records every offer and forget, and whether an offer is held right now."""

    def __init__(self) -> None:
        self.offers: list[dict] = []
        self.forgets = 0
        self.holding = False

    def offer_segmentation(self, **kwargs: object) -> None:
        self.offers.append(dict(kwargs))
        self.holding = bool(kwargs.get("hold"))

    def forget_segmentation(self) -> None:
        self.forgets += 1
        self.holding = False


class _WorldArm:
    """The shared fake arm, carrying a live world and recording whether it held an offer at each motion."""

    def __init__(self, world: _HoldingWorld, *, raise_on_move: BaseException | None = None) -> None:
        from tests._helpers import _FakeArm

        self._inner = _FakeArm()
        self.live_planner_world = world
        self.held_at_motion: list[bool] = []
        self.forgets_at_motion: list[int] = []
        self._raise = raise_on_move

    def get_tcp_pose(self):  # noqa: ANN201
        return self._inner.get_tcp_pose()

    def move_to(self, pose, **kwargs):  # noqa: ANN001, ANN201
        self.held_at_motion.append(self.live_planner_world.holding)
        self.forgets_at_motion.append(self.live_planner_world.forgets)
        if self._raise is not None:
            raise self._raise
        return self._inner.move_to(pose, **kwargs)


def _stamped_frame(stamp: float = 100.0):  # noqa: ANN202
    from dataclasses import replace

    from tests._helpers import _perception_frame

    return replace(_perception_frame(), timestamp=stamp)


class _Resolver:
    """A fixed CAMERA to BASE, counting how often a frame asked for it."""

    def __init__(self, matrix: np.ndarray | None = None) -> None:
        from src.geometry import Frame, Transform

        self.calls = 0
        self._transform = Transform.from_matrix(np.eye(4) if matrix is None else matrix,
                                                from_frame=Frame.CAMERA, to_frame=Frame.BASE)

    def camera_to_base_for_frame(self, frame, *, arm=None):  # noqa: ANN001, ANN201
        self.calls += 1
        return self._transform


def _orchestrator(arm: object, *, results: list, perception: object | None = None, **kwargs: object):  # noqa: ANN202
    from types import SimpleNamespace

    from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator
    from tests._helpers import _FakePerception, _ScriptedCalculator

    source = perception if perception is not None else _FakePerception([_stamped_frame()])
    if perception is None:
        source.streamer = SimpleNamespace(rig_id="overhead")  # type: ignore[attr-defined]
    return BinPickingOrchestrator(
        arm=arm,  # type: ignore[arg-type]
        calculator=_ScriptedCalculator(results),  # type: ignore[arg-type]
        perception=source,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


class ThePickLoopHoldsItsTargetTests(unittest.TestCase):
    def test_the_pick_loop_holds_its_target_through_every_motion(self) -> None:
        from tests.test_pick_loop import _success_result

        world = _HoldingWorld()
        arm = _WorldArm(world)
        _orchestrator(arm, results=[_success_result()], max_attempts=3).run()

        self.assertTrue(arm.held_at_motion and all(arm.held_at_motion), arm.held_at_motion)
        self.assertEqual(1, len(world.offers))
        self.assertIs(True, world.offers[0]["hold"])
        self.assertEqual(1, world.forgets, "the pick ended and the target stayed out of the world")

    def test_a_pick_that_raises_still_forgets(self) -> None:
        from src.robot.core.errors import CameraWorldUnavailable
        from tests.test_pick_loop import _success_result

        world = _HoldingWorld()
        arm = _WorldArm(world, raise_on_move=CameraWorldUnavailable(
            camera="overhead", verdict="no_frame", attempts=1, reason="the camera answered nothing"))
        with self.assertRaises(CameraWorldUnavailable):
            _orchestrator(arm, results=[_success_result()], max_attempts=3).run()
        self.assertEqual(1, world.forgets)

    def test_a_relocation_after_no_candidate_plans_with_the_target_back(self) -> None:
        from src.robot.grasping.loop.pick_loop import LateralOffsetViewpointPlanner
        from tests.test_pick_loop import _active_perception_result, _success_result

        world = _HoldingWorld()
        arm = _WorldArm(world)
        _orchestrator(arm, results=[_active_perception_result(), _success_result()],
                      viewpoint_planner=LateralOffsetViewpointPlanner(offset_mm=50.0), max_attempts=3).run()

        self.assertFalse(arm.held_at_motion[0], "the viewpoint move planned with the last target left out")
        self.assertTrue(all(arm.held_at_motion[1:]))

    def test_a_commit_refused_relocation_plans_with_the_target_back(self) -> None:
        from src.robot.grasping.loop.pick_loop import LateralOffsetViewpointPlanner
        from tests.test_pick_loop import CommitGateReobserveRelocateTests, _success_result

        world = _HoldingWorld()
        arm = _WorldArm(world)
        _orchestrator(arm, results=[_success_result()], viewpoint_planner=LateralOffsetViewpointPlanner(offset_mm=50.0),
                      max_attempts=3, scene_fusion=CommitGateReobserveRelocateTests._fusion(),
                      commit_policy=CommitGateReobserveRelocateTests._refusing_policy(), mode_label="auto").run()

        self.assertTrue(arm.held_at_motion, "no viewpoint move happened")
        self.assertTrue(world.offers, "the refused commit's attempt offered no target")
        self.assertGreaterEqual(arm.forgets_at_motion[0], 1, "the held target was not given back before the move")
        self.assertFalse(arm.held_at_motion[0], "the refused commit relocated with the target still left out")

    def test_the_offer_names_the_perception_camera(self) -> None:
        from tests.test_pick_loop import _success_result

        world = _HoldingWorld()
        _orchestrator(_WorldArm(world), results=[_success_result()], max_attempts=1).run()
        self.assertEqual("overhead", world.offers[0]["camera"])
        self.assertEqual(100.0, world.offers[0]["timestamp"])

    def test_a_source_that_names_no_camera_offers_only_the_target_box(self) -> None:
        from tests._helpers import _FakePerception
        from tests.test_pick_loop import _success_result

        world = _HoldingWorld()
        _orchestrator(_WorldArm(world), results=[_success_result()], perception=_FakePerception([_stamped_frame()]),
                      frame_resolver=_Resolver(), max_attempts=1).run()

        (offer,) = world.offers
        self.assertIs(UNSET, offer["camera"])
        self.assertEqual((), tuple(offer["labelled_masks"]))
        self.assertEqual((), tuple(offer["exclude_masks"]))
        self.assertIsNotNone(offer["target_points_base_mm"])

    def test_the_offer_uses_the_ranked_frames_transform_without_a_second_resolve(self) -> None:
        from src.robot.grasping.loop.pick_loop import segmentation_offer_from_frame
        from tests.test_pick_loop import _success_result

        world = _HoldingWorld()
        resolver = _Resolver()
        _orchestrator(_WorldArm(world), results=[_success_result()], frame_resolver=resolver, max_attempts=1).run()

        self.assertEqual(1, resolver.calls, "the offer resolved CAMERA to BASE a second time")
        expected = segmentation_offer_from_frame(_stamped_frame(), 0, camera="overhead", camera_to_base_mm=np.eye(4))
        assert expected is not None and expected.target_points_base_mm is not None
        np.testing.assert_allclose(world.offers[0]["target_points_base_mm"], expected.target_points_base_mm)

    def test_no_transform_and_no_camera_leaves_the_target_in_the_world_and_says_so(self) -> None:
        from tests._helpers import _FakePerception
        from tests.test_pick_loop import _rescan_result, _success_result

        world = _HoldingWorld()
        with self.assertLogs("src.robot.grasping.loop.pick_loop", level="WARNING") as logs:
            _orchestrator(_WorldArm(world), results=[_rescan_result(), _success_result()],
                          perception=_FakePerception([_stamped_frame()]), max_attempts=3).run()
        self.assertEqual([], world.offers)
        said = [line for line in logs.output if "stays an obstacle" in line]
        self.assertEqual(1, len(said), logs.output)

    def test_a_source_name_that_contradicts_the_primary_camera_id_is_refused(self) -> None:
        from tests.test_pick_loop import _success_result

        with self.assertRaises(ValueError) as caught:
            _orchestrator(_WorldArm(_HoldingWorld()), results=[_success_result()], primary_camera_id="side",
                          max_attempts=1).run()
        self.assertIn("'overhead'", str(caught.exception))
        self.assertIn("'side'", str(caught.exception))

    def test_an_arm_without_a_world_opens_an_empty_scope(self) -> None:
        from types import SimpleNamespace

        from src.robot.core.keep_out import keeping_out

        offer = SegmentationOffer(captured_at_s=1.0, target_points_base_mm=np.zeros((2, 3)))
        for arm in (SimpleNamespace(live_planner_world=None), SimpleNamespace()):
            with keeping_out(arm, offer) as scope:
                self.assertFalse(scope.world_wired)


class TheBuilderTests(unittest.TestCase):
    def test_object_indices_name_unnamed_masks_and_the_target_is_excluded(self) -> None:
        from types import SimpleNamespace

        from src.robot.grasping.loop.pick_loop import segmentation_offer_from_frame

        frame = SimpleNamespace(
            segmentations=(SimpleNamespace(label="", mask=np.zeros((4, 4), dtype=np.uint8)),
                           SimpleNamespace(label="red cube", mask=np.ones((4, 4), dtype=np.uint8))),
            depth_map=np.full((4, 4), 500.0), surface_depth_map=None, intrinsics=np.eye(3), timestamp=100.0,
        )
        offer = segmentation_offer_from_frame(frame, 1, camera="overhead", camera_to_base_mm=None)
        assert offer is not None
        self.assertEqual(["object_0", "red cube"], [name for name, _ in offer.labelled_masks])
        self.assertEqual(1, len(offer.exclude_masks))
        self.assertEqual(100.0, offer.captured_at_s)
        self.assertIsNone(offer.target_points_base_mm)
        self.assertIsNone(segmentation_offer_from_frame(frame, 1, camera=UNSET, camera_to_base_mm=None))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
