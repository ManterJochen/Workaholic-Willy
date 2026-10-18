"""The shadow seam: it may be wrong, it may be silent, it may never raise inside a pick.

A learned ranker in shadow mode has exactly one hard obligation — **it must not cost an attempt.** Its
opinion is optional; the pick is not. So every refusal path here is asserted to come back as a
telemetry object carrying a reason, and never as an exception. That is the same trade
`success_probability` makes and the reason its rerankers ship at weight zero.

The second thing asserted is `would_change_top1`, because while the ranker is in shadow that number IS
the measurement: whether turning it on would do anything, answered on every real pick instead of argued
about.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from src.robot.grasping.deep.ranker.features import VALID_JAW_V1, BOOTSTRAP_JAW_V1
from src.robot.grasping.deep.ranker.runtime import GbtRanker, RankerTree
from src.robot.grasping.deep.ranker.shadow import score_candidates


@dataclass(frozen=True)
class _Candidate:
    """The TRAINING-side row's field names -- `datagen.grasps.labels.GraspLabel`.

    ⛔ THIS CLASS IS WHY THE SEAM WAS INERT FOR ITS WHOLE LIFE. Its docstring used to read
    "Duck-typed like a `GraspPoint`", and it is not: a `GraspPoint` carries `position` / `axis` /
    `grip_width_mm`. Every test here passed against a type NO RUNTIME PATH PRODUCES, while
    `_candidate_vectors` returned None for every real candidate on every pick. A duck-type test that
    invents its own duck tests nothing.

    Kept -- renamed and re-documented -- because the seam genuinely must accept both families: these
    are the names the ranker was FIT under. The real-`GraspPoint` tests are below.
    """

    position_mm: tuple[float, float, float]
    approach: tuple[float, float, float]
    closing_axis: tuple[float, float, float]
    width_mm: float


def _grasp_point(*, position=(10.0, 20.0, 30.0), approach=(0.0, 0.0, -1.0),
                 axis=(1.0, 0.0, 0.0), width_mm: float = 42.0, score: float = 0.5):
    """A REAL `GraspPoint`, the type the pick loop actually hands the seam."""
    import numpy as np

    from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint

    return GraspPoint(position=np.asarray(position, dtype=float),
                      approach=np.asarray(approach, dtype=float),
                      axis=np.asarray(axis, dtype=float),
                      grip_width_mm=float(width_mm), score=float(score), frame=GraspFrame.BASE)


def _ranker(features: tuple[str, ...], *, on: int = 0) -> GbtRanker:
    """A ranker that reads exactly one feature and prefers a LARGER value of it.

    Deliberately trivial: these tests are about the seam, and a real model would make every assertion
    depend on what it happened to learn.
    """
    tree = RankerTree(
        feature=np.asarray([on, -1, -1]), threshold=np.asarray([50.0, -2.0, -2.0]),
        left=np.asarray([1, -1, -1]), right=np.asarray([2, -1, -1]),
        value=np.asarray([0.0, -1.0, 1.0]),
    )
    return GbtRanker(spec="t", features=features, init_score=0.0, learning_rate=1.0,
                     trees=(tree,), sha256="deadbeef" * 8)


def _cloud(n: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(-30, 30, n), rng.uniform(-30, 30, n),
                            rng.uniform(10.0, 60.0, n)])


def _candidates() -> list[_Candidate]:
    return [
        _Candidate((0.0, 0.0, 35.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), 30.0),
        _Candidate((0.0, 0.0, 35.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), 70.0),
    ]


class ItScoresAndReportsWhatItWouldChangeTests(unittest.TestCase):
    def test_it_scores_every_candidate(self) -> None:
        result = score_candidates(
            _ranker(BOOTSTRAP_JAW_V1.features), BOOTSTRAP_JAW_V1, _candidates(),
            target_points_mm=_cloud())
        self.assertTrue(result.scored, result.reason)
        self.assertEqual(result.candidates, 2)
        self.assertEqual(len(result.scores), 2)

    def test_it_says_when_it_would_pick_a_different_candidate(self) -> None:
        """The ranker prefers width > 50 mm, so it wants candidate 1 while the robot takes 0."""
        result = score_candidates(
            _ranker(BOOTSTRAP_JAW_V1.features), BOOTSTRAP_JAW_V1, _candidates(),
            target_points_mm=_cloud(), executed_index=0)
        self.assertEqual(result.ranker_top1, 1)
        self.assertTrue(result.would_change_top1)

    def test_it_says_when_it_agrees(self) -> None:
        result = score_candidates(
            _ranker(BOOTSTRAP_JAW_V1.features), BOOTSTRAP_JAW_V1, _candidates(),
            target_points_mm=_cloud(), executed_index=1)
        self.assertFalse(result.would_change_top1)

    def test_the_centre_it_used_is_reported(self) -> None:
        """A bad score and a bad centre look identical in the score alone."""
        result = score_candidates(
            _ranker(BOOTSTRAP_JAW_V1.features), BOOTSTRAP_JAW_V1, _candidates(),
            target_points_mm=_cloud(), support_z_mm=0.0)
        assert result.object_centre_mm is not None
        # z is (support + top) / 2, not the cloud's own mean -- the measured correction.
        self.assertAlmostEqual(result.object_centre_mm[2], _cloud()[:, 2].max() / 2.0, places=6)


class ItNeverRaisesInsideAPickTests(unittest.TestCase):
    """Every one of these is a real state a cell reaches, and none of them may cost an attempt."""

    def _score(self, **kwargs):
        return score_candidates(
            _ranker(BOOTSTRAP_JAW_V1.features), BOOTSTRAP_JAW_V1,
            kwargs.pop("candidates", _candidates()), **kwargs)

    def test_no_candidates(self) -> None:
        result = self._score(candidates=[], target_points_mm=_cloud())
        self.assertFalse(result.scored)
        self.assertIn("no candidates", result.reason)

    def test_no_cloud_at_all(self) -> None:
        result = self._score(target_points_mm=None)
        self.assertFalse(result.scored)
        self.assertIn("no target cloud", result.reason)

    def test_a_cloud_too_sparse_to_support_a_centre(self) -> None:
        """Quoting a centre from 12 points would be quoting outside the range it was measured on."""
        result = self._score(target_points_mm=_cloud(12))
        self.assertFalse(result.scored)
        self.assertIn("under 200", result.reason)

    def test_a_candidate_without_a_pose(self) -> None:
        result = self._score(candidates=[object()], target_points_mm=_cloud())
        self.assertFalse(result.scored)
        self.assertIn("missing its pose", result.reason)

    def test_a_candidate_whose_axis_is_parallel_to_its_approach(self) -> None:
        """Refused for the WHOLE set, not skipped: a ranking over a different candidate list than the
        robot has is a ranking of something else."""
        broken = [_Candidate((0.0, 0.0, 35.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 30.0)]
        result = self._score(candidates=broken, target_points_mm=_cloud())
        self.assertFalse(result.scored)
        self.assertIn("perpendicular", result.reason)

    def test_a_spec_asking_for_a_feature_the_seam_cannot_compute(self) -> None:
        spec = type(BOOTSTRAP_JAW_V1)(
            name="imaginary", frame="grasp", features=("width_mm", "colour"),
            label="held", group="object_id", derived=True)
        result = score_candidates(
            _ranker(("width_mm", "colour")), spec, _candidates(), target_points_mm=_cloud())
        self.assertFalse(result.scored)
        self.assertIn("does not compute", result.reason)

    def test_a_ranker_expecting_a_different_feature_count(self) -> None:
        """A model and a spec that disagree is a misconfiguration, and it must still not raise."""
        result = score_candidates(
            _ranker(("width_mm",)), BOOTSTRAP_JAW_V1, _candidates(), target_points_mm=_cloud())
        self.assertFalse(result.scored)
        self.assertIn("scoring", result.reason)


class TheEnvironmentFeaturesSurviveTheSeamTests(unittest.TestCase):
    def test_an_infinite_clearance_is_clipped_rather_than_refused(self) -> None:
        """`jaw_clearance_mm` is +inf when nothing is known to be in the way. That is honest as a value
        and unusable as a model input -- and it is the COMMON case in a sparse scene, not an error."""
        result = score_candidates(
            _ranker(VALID_JAW_V1.features), VALID_JAW_V1, _candidates(),
            target_points_mm=_cloud(), obstacle_points_mm=None)
        self.assertTrue(result.scored, result.reason)
        self.assertTrue(all(np.isfinite(v) for row in result.features for v in row))

    def test_an_obstacle_cloud_shrinks_the_clearance(self) -> None:
        index = VALID_JAW_V1.features.index("jaw_clearance_mm")
        far = score_candidates(
            _ranker(VALID_JAW_V1.features), VALID_JAW_V1, _candidates(),
            target_points_mm=_cloud(), obstacle_points_mm=np.asarray([[500.0, 500.0, 500.0]]))
        near = score_candidates(
            _ranker(VALID_JAW_V1.features), VALID_JAW_V1, _candidates(),
            target_points_mm=_cloud(), obstacle_points_mm=np.asarray([[16.0, 0.0, 35.0]]))
        self.assertLess(near.features[0][index], far.features[0][index])

    def test_the_telemetry_dict_is_flat_and_prefixed(self) -> None:
        payload = score_candidates(
            _ranker(BOOTSTRAP_JAW_V1.features), BOOTSTRAP_JAW_V1, _candidates(),
            target_points_mm=_cloud()).as_dict()
        self.assertTrue(all(key.startswith("deep_ranker_") for key in payload), sorted(payload))
        self.assertTrue(payload["deep_ranker_scored"])
        self.assertIn("deep_ranker_would_change_top1", payload)

    def test_a_refusal_reports_its_reason_in_the_telemetry(self) -> None:
        payload = score_candidates(
            _ranker(BOOTSTRAP_JAW_V1.features), BOOTSTRAP_JAW_V1, [],
            target_points_mm=_cloud()).as_dict()
        self.assertFalse(payload["deep_ranker_scored"])
        self.assertIn("deep_ranker_reason", payload)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class TheRuntimeTypeTests(unittest.TestCase):
    """⛔ THE TESTS THAT WOULD HAVE CAUGHT IT. Everything above this class runs against `_Candidate`,
    whose field names come from the TRAINING row. The pick loop hands `score_candidates` a
    `tuple[GraspPoint, ...]` (feedback.py `result.candidates`), and against that type the seam
    returned `deep_ranker_reason: 'a candidate is missing its pose'` on every pick, for its whole
    life -- so the learned ranker the schema quotes at 0.8947 AUROC had never produced one number.
    """

    def test_a_real_GraspPoint_yields_its_vectors(self) -> None:
        from src.robot.grasping.deep.ranker.shadow import _candidate_vectors
        vectors = _candidate_vectors(_grasp_point())
        self.assertIsNotNone(vectors, "the seam cannot read the type the pick loop hands it")
        assert vectors is not None
        position, approach, axis, width = vectors
        self.assertEqual(list(position), [10.0, 20.0, 30.0])
        self.assertEqual(list(axis), [1.0, 0.0, 0.0])
        self.assertAlmostEqual(width, 42.0)

    def test_the_training_row_still_works(self) -> None:
        """Both families, not one instead of the other: these are the names the ranker was fit under."""
        from src.robot.grasping.deep.ranker.shadow import _candidate_vectors
        self.assertIsNotNone(_candidate_vectors(_Candidate(
            position_mm=(1.0, 2.0, 3.0), approach=(0.0, 0.0, -1.0),
            closing_axis=(1.0, 0.0, 0.0), width_mm=30.0)))

    def test_something_that_is_not_a_candidate_is_still_refused(self) -> None:
        """The control. Accepting two name families must not turn the guard into "accept anything"."""
        from src.robot.grasping.deep.ranker.shadow import _candidate_vectors
        self.assertIsNone(_candidate_vectors(object()))
        self.assertIsNone(_candidate_vectors(
            _Candidate(position_mm=(1.0, 2.0), approach=(0.0, 0.0, -1.0),  # type: ignore[arg-type]
                       closing_axis=(1.0, 0.0, 0.0), width_mm=30.0)))

    def test_the_shipped_ranker_scores_real_GraspPoints_end_to_end(self) -> None:
        """⭑ THE ONE THAT MATTERS: the artifact that ships, the type that runs, a real number out."""
        from pathlib import Path
        from types import SimpleNamespace

        import numpy as np

        from src.robot.grasping.deep.ranker.context import DeepRankerContext
        from src.robot.grasping.deep.ranker.shadow import score_candidates

        artifact_dir = Path("assets/models/grasp_ranker/v1")
        if not (artifact_dir / "held_jaw_v1.json").is_file():
            self.skipTest("the shipped ranker artifact is not present")
        context = DeepRankerContext.from_config(SimpleNamespace(
            enabled=True, artifact_dir=str(artifact_dir), spec="held_jaw_v1"))
        self.assertIsNotNone(context)
        assert context is not None

        candidates = [_grasp_point(position=(10.0, 20.0, 30.0 + 10.0 * i), width_mm=42.0 + i,
                                   score=0.5 - 0.1 * i) for i in range(3)]
        cloud = np.random.default_rng(0).normal(size=(400, 3)) * 20.0 + np.array([10.0, 20.0, 30.0])
        telemetry = score_candidates(context.ranker, context.spec, candidates,
                                     target_points_mm=cloud).as_dict()
        self.assertTrue(telemetry["deep_ranker_scored"], telemetry)
        self.assertEqual(telemetry["deep_ranker_candidates"], 3)
        self.assertNotIn("deep_ranker_reason", telemetry)
