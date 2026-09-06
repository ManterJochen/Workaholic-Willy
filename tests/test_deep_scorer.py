"""The re-implemented tree walk, checked three ways — because a nice reader can be confidently wrong.

The owner's decision was to build the ranker fresh in `deep/` rather than inherit
`scoring/success_probability`'s reader, and to test anything risky against that existing one so no error
creeps in. The risk is specific and it is not hypothetical: a re-implemented tree walk agrees with
itself, and the only thing that can catch it disagreeing with the MODEL THAT WAS FITTED is an
independent evaluator.

So three referees, each able to catch something the others cannot:

1. **sklearn's own `decision_function`**, on the very model that was exported. Catches an export bug --
   a swapped child, an off-by-one in the node arrays, a wrong `init_score`, a forgotten learning rate.
2. **`success_probability`'s reader**, fed the same ensemble. Written months earlier, by a different
   route, for a different feature schema. Catches a walk that matches sklearn only because the export
   and the reader share a mistake.
3. **A tree built by hand**, whose answers are known by inspection. Catches both of the above agreeing
   about a structure neither of them understands.

`init_score` deserves the mention: it is RECOVERED as `decision_function - lr * sum(leaf)` on a probe
row rather than read off a private sklearn attribute. That is exact by the model's definition, and
referee 1 is what turns it from a plausible identity into a verified one.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.deep.ranker.features import (
    BOOTSTRAP_JAW_V1,
    GRASP_FRAME_FEATURES,
    derive_grasp_frame_columns,
    grasp_frame_offsets,
    object_centre_mm,
)
from src.robot.grasping.deep.ranker.runtime import (
    ARTIFACT_KIND,
    ARTIFACT_VERSION,
    GbtRanker,
    RankerTree,
    ranker_from_dict,
)
from src.robot.grasping.deep.ranker.training import export_ranker, fit_ranker

ROWS = 900


def _synthetic_corpus(rows: int = ROWS, seed: int = 7) -> dict:
    """A corpus with a REAL relationship in it, so a fit has something to find.

    Deliberately not noise: a model fitted on noise scores 0.5 and every comparison below would pass
    while proving nothing about the tree walk. The rule here is arbitrary but learnable.
    """
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=(rows, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    approach = np.cross(axis, rng.normal(size=(rows, 3)))
    approach /= np.linalg.norm(approach, axis=1, keepdims=True)
    centre = rng.normal(scale=20.0, size=(rows, 3))
    width = rng.uniform(10.0, 85.0, size=rows)
    held = ((width > 45.0) & (np.linalg.norm(centre, axis=1) < 25.0)).astype(float)
    return {
        "width_mm": width,
        "centre_x_mm": centre[:, 0], "centre_y_mm": centre[:, 1], "centre_z_mm": centre[:, 2],
        "axis_x": axis[:, 0], "axis_y": axis[:, 1], "axis_z": axis[:, 2],
        "approach_x": approach[:, 0], "approach_y": approach[:, 1], "approach_z": approach[:, 2],
        "held": held,
        "object_id": np.asarray([f"obj{i % 30}" for i in range(rows)]),
    }


class TheFreshReaderAgreesWithSklearnTests(unittest.TestCase):
    """Referee 1: the model that was exported, evaluated by the library that fitted it."""

    @classmethod
    def setUpClass(cls) -> None:
        table = _synthetic_corpus()
        cls.model, cls.report = fit_ranker(
            table, BOOTSTRAP_JAW_V1, folds=3, n_estimators=40, max_depth=3)
        cls.payload = export_ranker(cls.model, BOOTSTRAP_JAW_V1, cls.report)
        cls.ranker = ranker_from_dict(cls.payload)
        derived = derive_grasp_frame_columns(table)
        cls.matrix = np.column_stack(
            [np.asarray(derived[f], dtype=np.float64) for f in GRASP_FRAME_FEATURES])

    def test_scores_match_decision_function_to_machine_precision(self) -> None:
        """Not "close": the same arithmetic in a different order should land within float noise."""
        np.testing.assert_allclose(
            self.ranker.score(self.matrix), self.model.decision_function(self.matrix), atol=1e-9)

    def test_the_recovered_init_score_is_right(self) -> None:
        """The one value not copied but derived. A wrong one shifts every score by a constant --
        invisible to a ranking metric, and wrong in any artifact that ever gets calibrated."""
        leaf_total = sum(
            float(stage[0].predict(np.zeros((1, len(GRASP_FRAME_FEATURES))))[0])
            for stage in self.model.estimators_)
        expected = float(self.model.decision_function(np.zeros((1, len(GRASP_FRAME_FEATURES))))[0])
        self.assertAlmostEqual(
            self.ranker.init_score + self.ranker.learning_rate * leaf_total, expected, places=9)

    def test_the_fit_actually_learnt_the_rule(self) -> None:
        """Guards the guard: on noise every comparison above passes and proves nothing."""
        self.assertGreater(self.report.auroc, 0.9, self.report.summary())


class TheFreshReaderAgreesWithTheEXISTINGReaderTests(unittest.TestCase):
    """Referee 2: `success_probability`'s reader, written earlier and independently.

    Its ensemble has the same mathematical shape -- ``init + lr * sum(leaf)`` over
    feature/threshold/left/right/value trees -- so the exported payload can be handed to both. If the
    export and this package's walk shared a mistake, referee 1 would still pass and this would not.
    """

    def test_both_readers_produce_the_same_scores(self) -> None:
        from src.robot.grasping.scoring.success_probability._model import (  # noqa: PLC0415
            GbtEnsemble,
            GbtTree,
            _gbt_raw_predict,
        )

        table = _synthetic_corpus(rows=400, seed=3)
        model, report = fit_ranker(table, BOOTSTRAP_JAW_V1, folds=3, n_estimators=30, max_depth=3)
        payload = export_ranker(model, BOOTSTRAP_JAW_V1, report)
        mine = ranker_from_dict(payload)

        theirs = GbtEnsemble(
            init_score=float(payload["init_score"]),
            learning_rate=float(payload["learning_rate"]),
            trees=tuple(
                GbtTree(
                    feature=np.asarray(t["feature"], dtype=np.int64),
                    threshold=np.asarray(t["threshold"], dtype=np.float64),
                    left=np.asarray(t["left"], dtype=np.int64),
                    right=np.asarray(t["right"], dtype=np.int64),
                    value=np.asarray(t["value"], dtype=np.float64),
                )
                for t in payload["trees"]
            ),
        )
        derived = derive_grasp_frame_columns(table)
        matrix = np.column_stack(
            [np.asarray(derived[f], dtype=np.float64) for f in GRASP_FRAME_FEATURES])
        np.testing.assert_allclose(mine.score(matrix), _gbt_raw_predict(theirs, matrix), atol=1e-12)


class ATreeWhoseAnswersAreKnownByInspectionTests(unittest.TestCase):
    """Referee 3: both other referees could agree about a structure neither of them understands."""

    def _ranker(self) -> GbtRanker:
        # node 0: feature 0 <= 5 ? node 1 : node 2   |  leaves: -1.0 and +2.0
        tree = RankerTree(
            feature=np.asarray([0, -1, -1]), threshold=np.asarray([5.0, -2.0, -2.0]),
            left=np.asarray([1, -1, -1]), right=np.asarray([2, -1, -1]),
            value=np.asarray([0.0, -1.0, 2.0]),
        )
        return GbtRanker(spec="hand", features=("f0",), init_score=0.5, learning_rate=0.1,
                         trees=(tree,))

    def test_each_side_of_the_split_lands_where_it_should(self) -> None:
        ranker = self._ranker()
        np.testing.assert_allclose(
            ranker.score(np.asarray([[0.0], [5.0], [5.0001], [99.0]])),
            [0.5 - 0.1, 0.5 - 0.1, 0.5 + 0.2, 0.5 + 0.2],  # `<=` goes LEFT, so 5.0 is left
            atol=1e-12,
        )

    def test_a_wrong_feature_count_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._ranker().score(np.asarray([[1.0, 2.0]]))
        self.assertIn("expects 1 feature", str(caught.exception))

    def test_NaN_is_refused_rather_than_scored(self) -> None:
        """A NaN fails every `<=` and so walks consistently right -- a confident number from a
        missing measurement, which is worse than an error."""
        with self.assertRaises(ValueError) as caught:
            self._ranker().score(np.asarray([[np.nan]]))
        self.assertIn("NaN", str(caught.exception))


class TheArtifactRefusesWhatItShouldTests(unittest.TestCase):
    def _payload(self) -> dict:
        table = _synthetic_corpus(rows=300, seed=11)
        model, report = fit_ranker(table, BOOTSTRAP_JAW_V1, folds=3, n_estimators=10, max_depth=2)
        return export_ranker(model, BOOTSTRAP_JAW_V1, report)

    def test_a_tampered_artifact_fails_its_own_sha(self) -> None:
        """The one failure that would otherwise score silently: a file edited after it was fitted."""
        payload = self._payload()
        payload["learning_rate"] = float(payload["learning_rate"]) * 2.0
        with self.assertRaises(ValueError) as caught:
            ranker_from_dict(payload)
        self.assertIn("SHA mismatch", str(caught.exception))

    def test_the_wrong_kind_of_artifact_is_refused(self) -> None:
        payload = self._payload()
        payload["kind"] = "success_probability"
        with self.assertRaises(ValueError):
            ranker_from_dict(payload, verify_sha=False)

    def test_a_future_artifact_version_is_refused(self) -> None:
        payload = self._payload()
        payload["artifact_version"] = ARTIFACT_VERSION + 1
        with self.assertRaises(ValueError) as caught:
            ranker_from_dict(payload, verify_sha=False)
        self.assertIn("serialised shape", str(caught.exception))

    def test_the_card_says_it_is_not_a_probability(self) -> None:
        """Ranking-only is an owner decision, and it has to travel WITH the model or it becomes a
        probability by assumption the first time somebody reads the number."""
        card = self._payload()["card"]
        self.assertIn("NOT a calibrated probability", card["claims"])
        self.assertIn("GroupKFold", card["split"])
        self.assertEqual(self._payload()["kind"], ARTIFACT_KIND)


class TheFeatureDerivationAgreesWithItselfTests(unittest.TestCase):
    """The vectorised path and the single-grasp path must give the same numbers.

    They exist for different callers -- a corpus of 20,000 rows and one live grasp -- and a fast path
    that disagrees with the reference path is worse than no fast path.
    """

    def test_row_by_row_matches_the_vectorised_derivation(self) -> None:
        table = _synthetic_corpus(rows=200, seed=5)
        derived = derive_grasp_frame_columns(table)
        for row in range(200):
            with self.subTest(row=row):
                centre = np.asarray([table["centre_x_mm"][row], table["centre_y_mm"][row],
                                     table["centre_z_mm"][row]])
                axis = np.asarray([table["axis_x"][row], table["axis_y"][row], table["axis_z"][row]])
                approach = np.asarray([table["approach_x"][row], table["approach_y"][row],
                                       table["approach_z"][row]])
                # The corpus stores the centre RELATIVE to the grasp, so the grasp sits at the origin.
                got = grasp_frame_offsets(centre, np.zeros(3), approach, axis)
                np.testing.assert_allclose(
                    got,
                    [derived["centre_along_axis_mm"][row],
                     derived["centre_along_approach_mm"][row],
                     derived["centre_along_binormal_mm"][row]],
                    atol=1e-9,
                )

    def test_a_parallel_axis_and_approach_are_refused(self) -> None:
        """The GraspPoint contract promises perpendicular. A zero binormal would silently make the
        most informative feature 0 for that row rather than complain."""
        with self.assertRaises(ValueError) as caught:
            grasp_frame_offsets(np.ones(3), np.zeros(3), np.asarray([1.0, 0.0, 0.0]),
                                np.asarray([1.0, 0.0, 0.0]))
        self.assertIn("perpendicular", str(caught.exception))


class TheObjectCentreUsesTheTableRatherThanTheCloudTests(unittest.TestCase):
    """MEASURED: a fused cloud's own centroid sits 12.58 mm too high on 97.9 % of 435 objects."""

    def test_z_comes_from_the_support_plane_and_the_top(self) -> None:
        # A box from 0..40 mm in z, but with the underside missing -- the case a camera actually sees.
        rng = np.random.default_rng(0)
        points = np.column_stack([
            rng.uniform(-30, 30, 500), rng.uniform(-20, 20, 500), rng.uniform(20.0, 40.0, 500)])
        centre = object_centre_mm(points, support_z_mm=0.0)
        self.assertAlmostEqual(centre[2], 20.0, delta=0.5)      # (0 + 40) / 2, not the ~30 mean
        self.assertGreater(points[:, 2].mean(), 28.0, "the fixture must actually be top-heavy")

    def test_an_empty_cloud_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            object_centre_mm(np.zeros((0, 3)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
