"""Stage 3: the ball crop, the seed-local frame, and a kappa that was measured rather than copied.

⛔⛔ **THE GAP THIS CLOSES WAS REAL FOR THE WHOLE RESTART.** `SetGenerator.propose` was
`self.head(encoded[batch_index, point_index], gripper)`: one gathered per-point feature and the
gripper vector, while the architecture plan's §3.3 describes a crop, a local frame and a scale
normalisation. A reader of the plan and a reader of the code were describing different models.

⚠ **WHETHER IT HELPS IS AN ARM, NOT AN ASSUMPTION.** It ships off. What these tests pin is that off is
byte-identical, that on genuinely changes what the head reads, and that kappa comes from the
measurement rather than from the paper's household constant.
"""

from __future__ import annotations

import unittest

import torch

from src.robot.grasping.deep.net.local_crop import (
    MEASURED_RANGE_M,
    LocalCrop,
    LocalCropConfig,
    kappa_for_radius,
)
from src.robot.grasping.deep.net.set_generator import SetGenerator, SetGeneratorConfig


def _cloud(batch: int = 2, points: int = 256, seed: int = 0):
    torch.manual_seed(seed)
    return torch.randn(batch, points, 3) * 0.08, torch.randn(batch, points, 16)


class KappaTests(unittest.TestCase):

    def test_it_matches_the_MEASURED_table_at_the_measured_radii(self) -> None:
        for radius, mean_range in MEASURED_RANGE_M.items():
            with self.subTest(radius):
                self.assertAlmostEqual(kappa_for_radius(radius), 1.0 / mean_range, places=6)

    def test_it_is_NOT_the_paper_constant(self) -> None:
        """⛔ Outside work derives 3.27 for a household distribution. At the reference jaw's 85 mm
        aperture our corpus gives 8.4, so copying it would have been wrong by 2.6x."""
        self.assertGreater(kappa_for_radius(85.0), 7.0)
        self.assertLess(kappa_for_radius(85.0), 10.0)

    def test_it_REFUSES_outside_the_measured_band(self) -> None:
        """⚠ The relation looks linear over 40 to 120 mm, which is exactly the observation that
        invites extrapolating to 300 mm and getting a number nobody measured."""
        for radius in (10.0, 300.0):
            with self.subTest(radius), self.assertRaises(ValueError) as caught:
                kappa_for_radius(radius)
            self.assertIn("measured", str(caught.exception))

    def test_a_LARGER_radius_means_a_smaller_kappa(self) -> None:
        """The direction is the whole finding: kappa depends on the RADIUS, not on the object. A ball
        of radius r holds at most 2r of span, so the crop stops growing once the object exceeds it."""
        values = [kappa_for_radius(r) for r in sorted(MEASURED_RANGE_M)]
        self.assertEqual(values, sorted(values, reverse=True))


class CropTests(unittest.TestCase):

    def test_it_starts_at_ZERO_so_an_arm_starts_identical(self) -> None:
        """⛔ An arm whose two halves start from different initialisations measures the
        initialisation. The output projection is zero-initialised for the same reason `film` mixing
        is."""
        crop = LocalCrop(16, LocalCropConfig(neighbours=8, width=32))
        points, encoded = _cloud()
        out = crop(points, encoded, torch.tensor([0, 1]), torch.tensor([3, 5]))
        self.assertTrue(bool((out == 0).all()))

    def test_a_TRAINED_crop_is_not_zero_and_depends_on_the_neighbourhood(self) -> None:
        """The control for the test above: a module that could only ever return zero would pass it."""
        torch.manual_seed(1)
        crop = LocalCrop(16, LocalCropConfig(neighbours=8, width=32))
        torch.nn.init.normal_(crop.out.weight, std=0.2)
        points, encoded = _cloud()
        first = crop(points, encoded, torch.tensor([0]), torch.tensor([3]))
        # Move the neighbourhood, keep the seed's own feature: only the crop can see the difference.
        moved = points.clone()
        moved[0, 10:] += 0.5
        second = crop(moved, encoded, torch.tensor([0]), torch.tensor([3]))
        self.assertGreater(float((first - second).abs().max()), 1e-6)

    def test_a_seed_with_an_EMPTY_neighbourhood_reads_as_absent_not_as_minus_infinity(self) -> None:
        """⚠ The pool fills masked neighbours with the dtype's minimum so a real negative feature is
        not outvoted by an empty slot. A seed with NOTHING inside the radius would then pool that
        fill, which is not a feature; it becomes zero."""
        torch.manual_seed(2)
        crop = LocalCrop(16, LocalCropConfig(radius_mm=40.0, neighbours=8, width=32))
        torch.nn.init.normal_(crop.out.weight, std=0.2)
        far = torch.full((1, 64, 3), 10.0)      # every point 10 m away
        far[0, 0] = 0.0                          # except the seed itself
        encoded = torch.randn(1, 64, 16)
        out = crop(far, encoded, torch.tensor([0]), torch.tensor([0]))
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_points_OUTSIDE_the_radius_are_masked_rather_than_dropped(self) -> None:
        """⛔ Taking the nearest `neighbours` regardless would reach across a gap and describe another
        object's surface as this one's. A sparse neighbourhood has to read as sparse."""
        torch.manual_seed(3)
        crop = LocalCrop(16, LocalCropConfig(radius_mm=40.0, neighbours=16, width=32))
        torch.nn.init.normal_(crop.out.weight, std=0.2)
        points = torch.full((1, 40, 3), 0.5)
        points[0, :4] = torch.randn(4, 3) * 0.005      # four genuinely near points
        encoded = torch.randn(1, 40, 16)
        near_only = crop(points, encoded, torch.tensor([0]), torch.tensor([0]))
        # Move the FAR points further still. If they were being used, this would change the answer.
        farther = points.clone()
        farther[0, 4:] = 3.0
        self.assertTrue(torch.allclose(
            near_only, crop(farther, encoded, torch.tensor([0]), torch.tensor([0])), atol=1e-6))


class WiringTests(unittest.TestCase):

    def _net(self, crop: LocalCropConfig | None) -> SetGenerator:
        torch.manual_seed(0)
        base = SetGeneratorConfig()
        return SetGenerator(type(base)(
            backbone=type(base.backbone)(width=32, depth=2, heads=2),
            head=type(base.head)(slots=2, width=32), crop=crop))

    @staticmethod
    def _inputs(points: int = 128):
        """⚠ THE BACKBONE'S OWN `in_features`, not a round number. It refuses a mismatch, which is
        correct, and a test that hardcoded 16 would be testing the refusal instead of the crop."""
        from src.robot.grasping.deep.net.serialized_backbone import SerializedConfig

        torch.manual_seed(0)
        width = SerializedConfig().in_features
        return torch.randn(2, points, 3) * 0.08, torch.randn(2, points, width)

    def test_the_default_is_OFF(self) -> None:
        self.assertIsNone(SetGeneratorConfig().crop)
        self.assertIsNone(self._net(None).crop)

    def test_OFF_is_byte_identical_to_a_net_that_never_heard_of_the_crop(self) -> None:
        """⛔ DEFAULT-OFF BYTE-IDENTICAL is the rule every seam here ships under, and a new tensor in
        the graph is the easiest way to break it without noticing."""
        points, features = self._inputs()
        seeds = (torch.tensor([0, 1]), torch.tensor([4, 9]))
        gripper = torch.zeros(2, 14)
        a = self._net(None)
        b = self._net(None)
        with torch.no_grad():
            first = a.propose(a.encode(points, features), *seeds, gripper, points)
            second = b.propose(b.encode(points, features), *seeds, gripper)
        self.assertTrue(torch.equal(first.approach, second.approach))

    def test_ON_changes_the_head_INPUT_WIDTH_so_a_wrong_artifact_cannot_load(self) -> None:
        """⚠ An artifact trained with a crop must not load into a net built without one. The state
        dict simply will not fit, which is the loudest possible failure and the right one."""
        without = self._net(None)
        with_crop = self._net(LocalCropConfig(neighbours=8, width=16))
        self.assertNotEqual(without.head.in_features, with_crop.head.in_features)
        with self.assertRaises(RuntimeError):
            without.load_state_dict(with_crop.state_dict())

    def test_ON_still_starts_identical_because_the_crop_starts_at_zero(self) -> None:
        """⭐ The arm's two halves begin at the same numbers, so any difference is the crop learning
        rather than the crop existing."""
        points, features = self._inputs()
        seeds = (torch.tensor([0, 1]), torch.tensor([4, 9]))
        net = self._net(LocalCropConfig(neighbours=8, width=16))
        with torch.no_grad():
            encoded = net.encode(points, features)
            crop_feature = net.crop(points, encoded, *seeds)
        self.assertTrue(bool((crop_feature == 0).all()))

    def test_a_crop_net_REFUSES_propose_without_the_cloud(self) -> None:
        """It cannot crop what it is not given, and returning a silently uncropped answer would make
        the arm measure nothing."""
        net = self._net(LocalCropConfig(neighbours=8, width=16))
        points, features = self._inputs()
        with torch.no_grad(), self.assertRaises(ValueError) as caught:
            net.propose(net.encode(points, features), torch.tensor([0]), torch.tensor([4]),
                        torch.zeros(1, 14))
        self.assertIn("points_m", str(caught.exception))

    def test_forward_passes_the_cloud_through(self) -> None:
        """The path a training step actually takes."""
        net = self._net(LocalCropConfig(neighbours=8, width=16))
        points, features = self._inputs()
        with torch.no_grad():
            field, prediction = net(points, features, torch.tensor([0, 1]), torch.tensor([4, 9]),
                                    torch.zeros(2, 14))
        self.assertEqual(tuple(field.shape), (2, 128))
        self.assertEqual(prediction.approach.shape[0], 2)


if __name__ == "__main__":
    unittest.main()


class ArtifactTests(unittest.TestCase):
    """⛔⛔ TRAINABLE BUT UNLOADABLE IS THE DEFECT THIS PACKAGE SPENT LAST NIGHT FIXING ONE LEVEL UP.

    The first version of stage 3 wrote six extra tensors and a wider head trunk into a card whose
    reader rebuilt neither, so a crop-trained artifact refused to load with a size mismatch. Loud, and
    still a model nobody could deploy. Found by round-tripping the smoke run rather than by reading.
    """

    def test_a_crop_artifact_ROUND_TRIPS(self) -> None:
        import tempfile

        from src.robot.grasping.deep.set_artifact import (
            load_set_generator,
            write_set_generator,
        )
        from src.robot.grasping.deep.train.trainer import SetTrainingPlan

        torch.manual_seed(0)
        base = SetGeneratorConfig()
        crop = LocalCropConfig(radius_mm=60.0, neighbours=12, width=16)
        net = SetGenerator(type(base)(
            backbone=type(base.backbone)(width=32, depth=2, heads=2),
            head=type(base.head)(slots=2, width=32), crop=crop))
        plan = SetTrainingPlan()
        with tempfile.TemporaryDirectory() as name:
            from pathlib import Path as _Path

            paths = write_set_generator(_Path(name), net, sample=plan.sample, step=plan.step,
                                        gripper="2f85")
            loaded = load_set_generator(paths["weights"], device="cpu")
        self.assertEqual(loaded.net.config.crop, crop)
        self.assertIsNotNone(loaded.net.crop)
        self.assertEqual(sum(p.numel() for p in loaded.net.parameters()),
                         sum(p.numel() for p in net.parameters()))

    def test_a_NO_CROP_artifact_still_round_trips_with_crop_None(self) -> None:
        """The control, and the compatibility guarantee: every artifact written before stage 3
        existed has no `crop` key at all, and must load exactly as it always did."""
        import tempfile
        from pathlib import Path as _Path

        from src.robot.grasping.deep.set_artifact import (
            load_set_generator,
            write_set_generator,
        )
        from src.robot.grasping.deep.train.trainer import SetTrainingPlan

        torch.manual_seed(0)
        base = SetGeneratorConfig()
        net = SetGenerator(type(base)(
            backbone=type(base.backbone)(width=32, depth=2, heads=2),
            head=type(base.head)(slots=2, width=32)))
        plan = SetTrainingPlan()
        with tempfile.TemporaryDirectory() as name:
            paths = write_set_generator(_Path(name), net, sample=plan.sample, step=plan.step,
                                        gripper="2f85")
            loaded = load_set_generator(paths["weights"], device="cpu")
        self.assertIsNone(loaded.net.config.crop)
        self.assertIsNone(loaded.net.crop)


class LiveSeamTests(unittest.TestCase):

    def test_every_propose_CALL_SITE_passes_the_cloud(self) -> None:
        """⛔ THE FLAG IS ONLY LIVE WHERE THE CLOUD ARRIVES. `propose` refuses a crop-enabled net
        without it, so a missed call site is a loud failure rather than a silent one; this checks all
        three anyway, because "loud at runtime" still means the arm dies six hours in."""
        import ast
        from pathlib import Path as _Path

        # ⚠ AST, NOT STRING SLICING. My first version cut at the first ")" and landed inside
        # `torch.zeros_like(picks)`, so it failed on all three call sites that were in fact correct.
        # A guard that misreads the code it guards is worse than none.
        for name in ("src/robot/grasping/deep/train/step.py",
                     "src/robot/grasping/deep/calculator.py",
                     "src/robot/grasping/deep/eval/propose.py"):
            with self.subTest(name):
                tree = ast.parse(_Path(name).read_text(encoding="utf-8"))
                calls = [n for n in ast.walk(tree)
                         if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                         and n.func.attr == "propose"]
                self.assertTrue(calls, f"{name} no longer calls propose; retire this entry")
                for call in calls:
                    self.assertGreaterEqual(
                        len(call.args), 5,
                        f"{name} calls propose without the cloud, so --crop-mm is inert there")

    def test_the_flag_reaches_the_plan(self) -> None:
        from src.robot.grasping.deep.__main__ import build_parser

        args = build_parser().parse_args(
            ["train-set", "--clouds", "c", "--out", "o", "--crop-mm", "85"])
        self.assertEqual(args.crop_mm, 85.0)
        self.assertEqual(build_parser().parse_args(
            ["train-set", "--clouds", "c", "--out", "o"]).crop_mm, 0.0)
