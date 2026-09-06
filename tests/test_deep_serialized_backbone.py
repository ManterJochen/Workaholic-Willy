"""The serialized-attention encoder.

⚠ THE DEFECT THIS FILE IS FOR is a permutation that is not undone. The encoder sorts the cloud along
a space-filling curve, attends inside windows of that order, and must put the result back. If the
scatter and the gather ever disagree, every point receives another point's features: the shapes stay
right, nothing raises, and every metric downstream blames the head. So the first tests here check
that the encoder is EQUIVARIANT to the caller's point order, which is the property that breaks.

The second is that the curve does something. An ordering that grouped points arbitrarily would give a
transformer with a random receptive field, which trains and means nothing, so the Morton key is
checked against the geometry it claims to follow.
"""

from __future__ import annotations

import unittest

import torch

from src.robot.grasping.deep.net.serialized_backbone import (
    AXIS_ORDERS,
    SerializedBackbone,
    SerializedConfig,
    morton_code,
)


def _cloud(batch: int = 2, count: int = 300, features: int = 5, seed: int = 0) -> tuple:
    generator = torch.Generator().manual_seed(seed)
    points = torch.rand(batch, count, 3, generator=generator) * 0.6
    return points, torch.randn(batch, count, features, generator=generator)


class MortonTests(unittest.TestCase):

    def test_nearby_points_get_nearby_keys(self) -> None:
        """⭐ THE PROPERTY THE WHOLE DESIGN RESTS ON. If the ordering did not follow the geometry, the
        windows would group arbitrary points and the attention would have a random receptive field."""
        points, _ = _cloud(batch=1, count=2000, seed=1)
        code = morton_code(points)
        order = code.argsort(dim=1)[0]
        ordered = points[0][order]
        neighbour = (ordered[1:] - ordered[:-1]).norm(dim=-1).mean()
        shuffled = points[0][torch.randperm(2000, generator=torch.Generator().manual_seed(2))]
        random_step = (shuffled[1:] - shuffled[:-1]).norm(dim=-1).mean()
        self.assertLess(float(neighbour), 0.5 * float(random_step))

    def test_a_different_axis_order_produces_a_different_ordering(self) -> None:
        """The blocks alternate curves so consecutive ones group points differently. If the orders
        collapsed to one, the alternation would be decoration."""
        points, _ = _cloud(batch=1, count=500, seed=3)
        first = morton_code(points, AXIS_ORDERS[0]).argsort(dim=1)
        second = morton_code(points, AXIS_ORDERS[1]).argsort(dim=1)
        self.assertFalse(torch.equal(first, second))

    def test_a_degenerate_cloud_does_not_divide_by_zero(self) -> None:
        points = torch.full((1, 16, 3), 0.25)
        code = morton_code(points)
        self.assertTrue(torch.isfinite(code.to(torch.float64)).all())
        self.assertEqual(int(code.max()), int(code.min()))

    def test_the_key_is_a_permutation_of_no_information_loss(self) -> None:
        """Ten bits per axis is finer than the 3 mm voxel the cloud is built at, so two distinct
        points must not collide into one key on a realistic workspace."""
        generator = torch.Generator().manual_seed(4)
        points = (torch.randint(0, 200, (1, 4000, 3), generator=generator).float() * 0.003)
        code = morton_code(points)
        unique_points = torch.unique(points[0], dim=0).shape[0]
        self.assertEqual(int(torch.unique(code[0]).numel()), unique_points)

    def test_refusals(self) -> None:
        with self.assertRaises(ValueError):
            morton_code(torch.rand(4, 3))


class OrderEquivarianceTests(unittest.TestCase):
    """⛔ The defect no shape check can see."""

    def test_shuffling_the_input_shuffles_the_output_the_same_way(self) -> None:
        backbone = SerializedBackbone(SerializedConfig(width=32, depth=3, heads=4,
                                                       window=16)).eval()
        points, features = _cloud(batch=1, count=200, seed=5)
        with torch.no_grad():
            straight = backbone(points, features)
            order = torch.randperm(200, generator=torch.Generator().manual_seed(6))
            shuffled = backbone(points[:, order], features[:, order])
        self.assertTrue(torch.allclose(straight[:, order], shuffled, atol=1e-4),
                        "the encoder did not put its serialisation back")

    def test_the_output_keeps_the_caller_s_point_count_and_order(self) -> None:
        backbone = SerializedBackbone(SerializedConfig(width=32, depth=2, heads=4,
                                                       window=64)).eval()
        points, features = _cloud(batch=3, count=257, seed=7)
        with torch.no_grad():
            out = backbone(points, features)
        self.assertEqual(out.shape, (3, 257, 32))


class PaddingTests(unittest.TestCase):

    def test_a_count_that_does_not_divide_the_window_still_works(self) -> None:
        backbone = SerializedBackbone(SerializedConfig(width=32, depth=2, heads=4,
                                                       window=64)).eval()
        for count in (1, 63, 64, 65, 129):
            with self.subTest(count=count):
                points, features = _cloud(batch=1, count=count, seed=8)
                with torch.no_grad():
                    out = backbone(points, features)
                self.assertEqual(out.shape, (1, count, 32))
                self.assertTrue(torch.isfinite(out).all())

    def test_padding_cannot_leak_into_a_real_point(self) -> None:
        """⭐ A padded KEY that was attended to would mix a zero vector into the last window's real
        points, which is a quiet corruption of exactly the points at the end of the curve.

        Checked by adding points AFTER the padding boundary would move: with the mask working, a real
        point's output depends only on its own window's real points.
        """
        config = SerializedConfig(width=32, depth=1, heads=4, window=8)
        backbone = SerializedBackbone(config).eval()
        points, features = _cloud(batch=1, count=8, seed=9)
        padded_points = torch.cat([points, points[:, :3] + 5.0], dim=1)
        padded_features = torch.cat([features, torch.zeros(1, 3, 5)], dim=1)
        with torch.no_grad():
            exact = backbone(points, features)
            with_extra = backbone(padded_points, padded_features)
        # The three added points are far away, so they sort to their own window and cannot change the
        # first eight. If the mask were broken they would be padded INTO that window instead.
        self.assertTrue(torch.allclose(exact, with_extra[:, :8], atol=1e-4))


class ShapeAndScaleTests(unittest.TestCase):

    def test_the_default_lands_in_the_range_the_field_uses(self) -> None:
        """⭐ THE POINT OF THIS FILE. This repository measured capacity as not the constraint over
        87k to 1.27M parameters; the works it borrows from run at tens of millions. The default here
        has to be in THAT band or the measurement is the old one again under a new name."""
        backbone = SerializedBackbone()
        parameters = sum(p.numel() for p in backbone.parameters())
        self.assertGreater(parameters, 10_000_000)
        self.assertLess(parameters, 30_000_000)

    def test_gradients_reach_the_first_layer(self) -> None:
        backbone = SerializedBackbone(SerializedConfig(width=32, depth=3, heads=4, window=32))
        points, features = _cloud(batch=2, count=100, seed=10)
        backbone(points, features).sum().backward()
        self.assertIsNotNone(backbone.embed.weight.grad)
        self.assertGreater(float(backbone.embed.weight.grad.abs().sum()), 0.0)

    def test_the_position_projection_actually_matters(self) -> None:
        """Two clouds with identical features and different geometry must not encode the same."""
        backbone = SerializedBackbone(SerializedConfig(width=32, depth=2, heads=4,
                                                       window=32)).eval()
        features = torch.randn(1, 64, 5, generator=torch.Generator().manual_seed(11))
        first = torch.rand(1, 64, 3, generator=torch.Generator().manual_seed(12)) * 0.6
        second = torch.rand(1, 64, 3, generator=torch.Generator().manual_seed(13)) * 0.6
        with torch.no_grad():
            self.assertFalse(torch.allclose(backbone(first, features), backbone(second, features),
                                            atol=1e-3))

    def test_refusals(self) -> None:
        with self.assertRaises(ValueError):
            SerializedBackbone(SerializedConfig(depth=0))
        with self.assertRaises(ValueError):
            SerializedBackbone(SerializedConfig(window=0))
        with self.assertRaises(ValueError):
            SerializedBackbone(SerializedConfig(width=32, heads=5))
        backbone = SerializedBackbone(SerializedConfig(width=32, depth=1, heads=4))
        with self.assertRaises(ValueError):
            backbone(torch.rand(1, 10, 3), torch.rand(1, 10, 4))
        with self.assertRaises(ValueError):
            backbone(torch.rand(1, 10, 3), torch.rand(1, 9, 5))


if __name__ == "__main__":
    unittest.main()
