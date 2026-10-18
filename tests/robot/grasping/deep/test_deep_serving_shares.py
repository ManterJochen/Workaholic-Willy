"""The seed mixture a trained head is SERVED with, derived from the one it was TRAINED with.

⛔⛔ **THE MEASUREMENT.** The first referee success rate this project computed, 0 held of 307, was
taken on proposals whose seeds were drawn entirely from the model's own graspability field. MEASURED
on that same artifact:

    draw        cosine between two seed features    spatial spread
    predicted   1.0000                              16 to 68 mm
    random      0.70                                163 to 269 mm

Same model, same cloud, same encoder. The field is nearly flat (p99 minus p50 is 0.08 logits), so its
top-k is one small blob, every point in it carries the same feature direction to four decimal places,
and a head handed one input returns one answer.

⭐ AND THE REPAIR IS MEASURED TOO. The same artifact over the same 40 scenes, changing only where the
seeds come from:

    distinct at 20 mm / 20 deg   25.6 %  ->  71.2 %
    distinct approach vectors    11      ->  20
    on-object share              70.9 %  ->  61.3 %

⚠ SO IT WAS A MEASUREMENT DEFECT AND NOT ONLY A MODEL DEFECT, and both halves are true: the referee
would have judged one grasp four times, AND 20 distinct vectors across 320 proposals is still a
near-constant head. Fixing the seeding does not make a three-epoch model good; it makes its number
mean what it says.
"""

from __future__ import annotations

import unittest

from src.robot.grasping.deep.net.set_targets import SeedShares
from src.robot.grasping.deep.eval.propose import serving_shares


class RedistributionTests(unittest.TestCase):

    def test_the_labelled_share_is_GONE_at_serve_time(self) -> None:
        """There are no labels in a cell. Seeding from them would grade the head on poses it was
        handed, which is the mimicry problem one level up from filtering the proposals."""
        self.assertEqual(serving_shares(SeedShares(0.5, 0.25, 0.25)).labelled, 0.0)

    def test_the_trained_RATIO_survives(self) -> None:
        """⭐ The point. A head fitted at predicted:random of 1:1 is served at 1:1, so serve-time
        seeding stays as close to train-time seeding as the missing labels allow."""
        out = serving_shares(SeedShares(0.5, 0.25, 0.25))
        self.assertAlmostEqual(out.predicted, 0.5)
        self.assertAlmostEqual(out.random, 0.5)

    def test_an_UNEVEN_trained_ratio_survives_too(self) -> None:
        """The control for the test above: 1:1 in and 1:1 out could be a hardcoded even split."""
        out = serving_shares(SeedShares(0.4, 0.45, 0.15))
        self.assertAlmostEqual(out.predicted, 0.75)
        self.assertAlmostEqual(out.random, 0.25)

    def test_the_shares_still_sum_to_one(self) -> None:
        for trained in (SeedShares(0.5, 0.25, 0.25), SeedShares(0.0, 0.9, 0.1),
                        SeedShares(0.8, 0.1, 0.1)):
            with self.subTest(str(trained)):
                out = serving_shares(trained)
                self.assertAlmostEqual(out.labelled + out.predicted + out.random, 1.0)

    def test_ALL_PREDICTED_is_no_longer_what_a_default_artifact_gets(self) -> None:
        """⛔ THE DEFECT ITSELF, stated as the assertion that would have caught it. The shipped
        training default is labelled 0.5 / predicted 0.25 / random 0.25, and `propose` served it
        predicted 1.0."""
        out = serving_shares(SeedShares(0.5, 0.25, 0.25))
        self.assertLess(out.predicted, 1.0)
        self.assertGreater(out.random, 0.0,
                           "an all-predicted draw collapses onto one blob of identical features")

    def test_a_head_trained_on_LABELS_ALONE_falls_back_evenly(self) -> None:
        """⚠ No mixture is faithful there, because the head only ever saw labelled seeds. Refusing
        would strand an artifact over a choice that has no right answer, so it splits and says so in
        the docstring rather than pretending to be faithful."""
        out = serving_shares(SeedShares(1.0, 0.0, 0.0))
        self.assertAlmostEqual(out.predicted, 0.5)
        self.assertAlmostEqual(out.random, 0.5)

    def test_a_head_trained_with_NO_predicted_share_keeps_none(self) -> None:
        """Faithfulness cuts both ways: a head that never saw its own field at training time is not
        served from it either."""
        out = serving_shares(SeedShares(0.5, 0.0, 0.5))
        self.assertAlmostEqual(out.predicted, 0.0)
        self.assertAlmostEqual(out.random, 1.0)


class WiringTests(unittest.TestCase):

    def test_propose_uses_it_rather_than_a_literal(self) -> None:
        from pathlib import Path

        source = Path("src/robot/grasping/deep/eval/propose.py").read_text(encoding="utf-8")
        self.assertIn("shares=serving_shares(loaded.step.shares)", source)
        self.assertNotIn("SeedShares(labelled=0.0, predicted=1.0, random=0.0)", source)

    def test_the_run_PRINTS_the_mixture_it_drew_with(self) -> None:
        """A mixture nobody can see is a mixture nobody checks, and this one silently decided the
        first referee number the project ever published."""
        from pathlib import Path

        source = Path("src/robot/grasping/deep/eval/propose.py").read_text(encoding="utf-8")
        self.assertIn("seeds drawn at predicted", source)


if __name__ == "__main__":
    unittest.main()
