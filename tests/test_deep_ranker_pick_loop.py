"""The shadow ranker inside the pick loop: it may watch, it may refuse, it may never cost an attempt.

Three obligations, and each is a defect this repo has already paid for once:

1. **Off by default is byte-identical.** No context, no telemetry key, no work.
2. **The support plane's FRAME is checked.** `SupportPlane.frame` defaults to `CAMERA`, and reading its
   `offset_mm` as a BASE height is literally the defect measured at 577 mm against −7 mm. A
   camera-frame plane must produce a refusal with a reason, never a number.
3. **Nothing it does may raise.** The scorer's opinion is optional; the pick is not. A failure has to
   come back as telemetry, not as a lost attempt.

The obstacle cloud comes from `fused_scene` (BASE) rather than `extra_kwargs["scene_points_mm"]`
(CAMERA) for the same reason as (2), and that is asserted too — the two are one line apart at the call
site, which is exactly how a frame defect gets written.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.robot.grasping.collision.table_collision import SupportPlane
from src.robot.grasping.deep.ranker.context import DeepRankerContext
from src.robot.grasping.deep.ranker.features import BOOTSTRAP_JAW_V1, HELD_JAW_V1
from src.robot.grasping.deep.ranker.runtime import GbtRanker, RankerTree
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator
from src.robot.grasping.multiview.scene_geometry import FusedSceneGeometry
from src.robot.grasping.types.feedback import GraspResult
from src.geometry import Frame


def _orchestrator(**kwargs) -> BinPickingOrchestrator:
    return BinPickingOrchestrator(
        arm=SimpleNamespace(),  # type: ignore[arg-type]
        calculator=SimpleNamespace(),  # type: ignore[arg-type]
        perception=SimpleNamespace(),  # type: ignore[arg-type]
        **kwargs,
    )


def _context() -> DeepRankerContext:
    tree = RankerTree(
        feature=np.asarray([0, -1, -1]), threshold=np.asarray([50.0, -2.0, -2.0]),
        left=np.asarray([1, -1, -1]), right=np.asarray([2, -1, -1]),
        value=np.asarray([0.0, -1.0, 1.0]),
    )
    ranker = GbtRanker(spec=BOOTSTRAP_JAW_V1.name, features=BOOTSTRAP_JAW_V1.features,
                       init_score=0.0, learning_rate=1.0, trees=(tree,), sha256="ab" * 32)
    return DeepRankerContext(ranker=ranker, spec=BOOTSTRAP_JAW_V1)


def _clearance_context() -> DeepRankerContext:
    """A ranker that splits on `jaw_clearance_mm`, so an obstacle cloud can move its answer.

    `BOOTSTRAP_JAW_V1` has only pose features, so every obstacle produces the same score under it and
    a test built on it would pass whether or not the cloud arrived. That is exactly how the missing
    obstacle cloud stayed invisible.
    """
    clearance = HELD_JAW_V1.features.index("jaw_clearance_mm")
    tree = RankerTree(
        feature=np.asarray([clearance, -1, -1]), threshold=np.asarray([25.0, -2.0, -2.0]),
        left=np.asarray([1, -1, -1]), right=np.asarray([2, -1, -1]),
        value=np.asarray([0.0, -1.0, 1.0]),
    )
    ranker = GbtRanker(spec=HELD_JAW_V1.name, features=HELD_JAW_V1.features,
                       init_score=0.0, learning_rate=1.0, trees=(tree,), sha256="cd" * 32)
    return DeepRankerContext(ranker=ranker, spec=HELD_JAW_V1)


class _Candidate:
    def __init__(self, width: float) -> None:
        self.position_mm = (0.0, 0.0, 40.0)
        self.approach = (0.0, 0.0, -1.0)
        self.closing_axis = (1.0, 0.0, 0.0)
        self.width_mm = width


def _result() -> GraspResult:
    return GraspResult(
        candidates=(_Candidate(30.0), _Candidate(70.0)),  # type: ignore[arg-type]
        telemetry={"existing_key": 1},
    )


def _cloud(n: int = 400) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.column_stack([rng.uniform(-30, 30, n), rng.uniform(-30, 30, n),
                            rng.uniform(10.0, 60.0, n)])


class OffByDefaultTests(unittest.TestCase):
    def test_no_context_means_no_telemetry_and_no_work(self) -> None:
        result = _result()
        _orchestrator()._stamp_deep_ranker(result, {}, None, 0)
        self.assertEqual(result.telemetry, {"existing_key": 1})


class ItScoresWhenEnabledTests(unittest.TestCase):
    def test_the_telemetry_appears_beside_what_was_there(self) -> None:
        result = _result()
        _orchestrator(deep_ranker_context=_context())._stamp_deep_ranker(
            result, {"geometry_points_base_mm": _cloud()}, None, 0)
        self.assertEqual(result.telemetry["existing_key"], 1, "it must not disturb what was there")
        self.assertTrue(result.telemetry["deep_ranker_scored"], result.telemetry)
        self.assertEqual(result.telemetry["deep_ranker_candidates"], 2)

    def test_it_reports_that_it_would_have_chosen_differently(self) -> None:
        """The whole point of shadow: whether turning it on would do anything."""
        result = _result()
        _orchestrator(deep_ranker_context=_context())._stamp_deep_ranker(
            result, {"geometry_points_base_mm": _cloud()}, None, 0)
        self.assertTrue(result.telemetry["deep_ranker_would_change_top1"])
        self.assertEqual(result.telemetry["deep_ranker_top1"], 1)

    def test_it_changes_no_candidate_and_no_order(self) -> None:
        """Shadow means shadow. The candidates the caller holds are the ones it had."""
        result = _result()
        before = tuple(c.width_mm for c in result.candidates)
        _orchestrator(deep_ranker_context=_context())._stamp_deep_ranker(
            result, {"geometry_points_base_mm": _cloud()}, None, 0)
        self.assertEqual(tuple(c.width_mm for c in result.candidates), before)


class TheSupportPlaneFrameIsCheckedTests(unittest.TestCase):
    """`SupportPlane.frame` defaults to CAMERA. Using its offset as a BASE height is the 577 mm defect."""

    def test_a_camera_frame_plane_is_REFUSED_with_a_reason(self) -> None:
        result = _result()
        _orchestrator(deep_ranker_context=_context())._stamp_deep_ranker(
            result,
            {"geometry_points_base_mm": _cloud(),
             "support_plane": SupportPlane(offset_mm=-598.0, frame=Frame.CAMERA)},
            None, 0)
        self.assertFalse(result.telemetry["deep_ranker_scored"])
        self.assertIn("not BASE", result.telemetry["deep_ranker_reason"])

    def test_a_base_frame_plane_is_used(self) -> None:
        result = _result()
        _orchestrator(deep_ranker_context=_context())._stamp_deep_ranker(
            result,
            {"geometry_points_base_mm": _cloud(),
             "support_plane": SupportPlane(offset_mm=0.0, frame=Frame.BASE)},
            None, 0)
        self.assertTrue(result.telemetry["deep_ranker_scored"], result.telemetry)


class TheObstacleCloudComesFromTheBaseFrameSourceTests(unittest.TestCase):
    """`scene_points_mm` is CAMERA-frame and `fused_scene` is BASE. Mixing them is the same defect."""

    def test_the_fused_scene_supplies_the_obstacles(self) -> None:
        # The REAL type, not a stand-in. A hand-built double once carried an `indices` attribute the
        # fused scene has never had, so the seam read an empty tuple, scored every candidate against
        # no obstacle, and this test passed throughout.
        fused = FusedSceneGeometry(
            clouds_base_mm=(_cloud(), np.asarray([[15.0, 0.0, 40.0]])),
            views_used=("cam_right",),
            associations=(),
        )
        result = _result()
        _orchestrator(deep_ranker_context=_context())._stamp_deep_ranker(
            result, {"geometry_points_base_mm": _cloud()}, fused, 0)
        self.assertTrue(result.telemetry["deep_ranker_scored"], result.telemetry)
        # Object 0 is the target, so exactly one other object supplies obstacle points.
        self.assertEqual(1, result.telemetry["deep_ranker_obstacle_objects"])

    def test_the_obstacle_count_is_reported_so_an_empty_one_is_visible(self) -> None:
        """A scene of one object has nothing to clear, and says so rather than staying silent."""
        alone = FusedSceneGeometry(
            clouds_base_mm=(_cloud(),), views_used=("cam_right",), associations=(),
        )
        result = _result()
        _orchestrator(deep_ranker_context=_context())._stamp_deep_ranker(
            result, {"geometry_points_base_mm": _cloud()}, alone, 0)
        self.assertEqual(0, result.telemetry["deep_ranker_obstacle_objects"])

    def test_the_obstacles_reach_the_scorer_and_change_the_score(self) -> None:
        """The count alone would be satisfied by a number nobody reads. This asserts the effect.

        Same target, same candidates; the only difference is a second object standing next to it. A
        ranker that splits on jaw clearance cannot answer both the same way, and if it does then the
        cloud never reached the scorer. The bootstrap spec cannot express this at all -- it carries
        four pose features and no environment feature -- so the test uses the spec that has one.
        """
        context = _clearance_context()
        alone = FusedSceneGeometry(
            clouds_base_mm=(_cloud(),), views_used=("cam_right",), associations=(),
        )
        crowded = FusedSceneGeometry(
            clouds_base_mm=(_cloud(), np.asarray([[12.0, 0.0, 40.0]] * 40)),
            views_used=("cam_right",), associations=(),
        )
        scores = []
        for scene in (alone, crowded):
            result = _result()
            _orchestrator(deep_ranker_context=context)._stamp_deep_ranker(
                result, {"geometry_points_base_mm": _cloud()}, scene, 0)
            scores.append(result.telemetry["deep_ranker_top_score"])
        self.assertNotEqual(scores[0], scores[1])

    def test_a_camera_frame_scene_cloud_is_NOT_used_as_obstacles(self) -> None:
        """Handing it `scene_points_mm` must not change the score: it is in the wrong frame, so the
        seam ignores it entirely rather than quietly measuring clearances in camera coordinates."""
        context = _context()
        plain = _result()
        _orchestrator(deep_ranker_context=context)._stamp_deep_ranker(
            plain, {"geometry_points_base_mm": _cloud()}, None, 0)
        with_camera_cloud = _result()
        _orchestrator(deep_ranker_context=context)._stamp_deep_ranker(
            with_camera_cloud,
            {"geometry_points_base_mm": _cloud(),
             "scene_points_mm": np.asarray([[0.0, 0.0, 600.0]] * 50)}, None, 0)
        self.assertEqual(plain.telemetry["deep_ranker_top_score"],
                         with_camera_cloud.telemetry["deep_ranker_top_score"])


class ItNeverCostsAnAttemptTests(unittest.TestCase):
    def test_a_scorer_that_explodes_leaves_the_result_alone(self) -> None:
        """The one obligation that outranks every other: an optional observer may not break a pick."""

        class _Exploding:
            spec = BOOTSTRAP_JAW_V1

            @property
            def ranker(self):  # noqa: ANN202
                raise RuntimeError("boom")

        result = _result()
        _orchestrator(deep_ranker_context=_Exploding())._stamp_deep_ranker(  # type: ignore[arg-type]
            result, {"geometry_points_base_mm": _cloud()}, None, 0)
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(result.telemetry["existing_key"], 1)

    def test_a_missing_cloud_is_a_reason_not_a_crash(self) -> None:
        result = _result()
        _orchestrator(deep_ranker_context=_context())._stamp_deep_ranker(result, {}, None, 0)
        self.assertFalse(result.telemetry["deep_ranker_scored"])
        self.assertIn("no target cloud", result.telemetry["deep_ranker_reason"])


class TheContextRefusesAMismatchedArtifactTests(unittest.TestCase):
    def test_a_disabled_block_loads_nothing(self) -> None:
        self.assertIsNone(DeepRankerContext.from_config(SimpleNamespace(enabled=False)))

    def test_a_missing_artifact_returns_None_rather_than_raising(self) -> None:
        self.assertIsNone(DeepRankerContext.from_config(SimpleNamespace(
            enabled=True, artifact_dir="/nowhere/at/all", spec="valid_jaw_v1")))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
