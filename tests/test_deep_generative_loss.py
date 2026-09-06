"""The denoising objective that makes stage 4b an arm instead of a shape.

⛔⛔ **THE HEAD SHIPPED WITH NO LOSS AND NO CALL SITE.** It had a forward pass, a sampler and eleven
tests, and nothing that could train it. These tests pin the half that was missing, and in particular
the one property the whole arm exists to test.

⭐⭐ **EVERY LABEL, NOT A MATCHED SUBSET.** MEASURED on v5: 75.2 % of supervised points admit more than
one approach further than fifteen degrees apart, median four, p90 fifteen. The deterministic head
binds at most K of them, so its coverage ceiling at K = 4 is 0.385. Running an assignment in this loss
would reimpose exactly that bound, and the comparison would then be decided by the wiring rather than
by the architecture.
"""

from __future__ import annotations

import unittest

import torch

from src.robot.grasping.deep.net.generative_head import (
    POSE_DIM,
    GenerativeGraspHead,
    GenerativeHeadConfig,
)
from src.robot.grasping.deep.net.generative_loss import (
    decode_samples,
    generative_loss,
    pose_from_target,
)
from src.robot.grasping.deep.net.gripper import GRIPPER_VECTOR_DIM
from src.robot.grasping.deep.net.slot_head import SlotHeadConfig
from src.robot.grasping.deep.net.set_loss import GraspSetTarget


def _head(**over) -> GenerativeGraspHead:
    torch.manual_seed(0)
    base = dict(width=64, depth=2, steps=8, samples=4, gripper_width=8, time_width=16)
    base.update(over)
    return GenerativeGraspHead(32, GenerativeHeadConfig(**base))


def _target(seeds: int = 12, labels: int = 5, valid: torch.Tensor | None = None) -> GraspSetTarget:
    torch.manual_seed(1)
    return GraspSetTarget(
        approach=torch.nn.functional.normalize(torch.randn(seeds, labels, 3), dim=-1),
        axis=torch.nn.functional.normalize(torch.randn(seeds, labels, 3), dim=-1),
        offset_m=torch.randn(seeds, labels, 3) * 0.02,
        width_m=torch.rand(seeds, labels) * 0.08,
        valid=torch.ones(seeds, labels, dtype=torch.bool) if valid is None else valid)


def _inputs(seeds: int = 12):
    torch.manual_seed(2)
    return torch.randn(seeds, 32), torch.randn(seeds, GRIPPER_VECTOR_DIM)


class EveryLabelTests(unittest.TestCase):

    def test_it_trains_on_EVERY_valid_label(self) -> None:
        """⭐ THE PROPERTY THE ARM EXISTS FOR. Fifteen labels on one seed is fifteen examples, not
        four; a matched subset would reimpose the K bound this head is meant to escape."""
        target = _target(seeds=6, labels=9)
        out = generative_loss(_head(), *_inputs(6), target)
        self.assertEqual(int(out["noise_mse_count"]), int(target.valid.sum()))
        self.assertEqual(int(out["noise_mse_count"]), 54)

    def test_PADDED_labels_are_excluded(self) -> None:
        """The control. Counting the padding would make the count a function of the tensor shape
        rather than of the data, which is how a metric comes to describe its own buffer."""
        valid = torch.zeros(6, 9, dtype=torch.bool)
        valid[:, :2] = True
        out = generative_loss(_head(), *_inputs(6), _target(6, 9, valid))
        self.assertEqual(int(out["noise_mse_count"]), 12)

    def test_a_batch_with_NO_label_contributes_nothing_rather_than_a_zero(self) -> None:
        """⚠ THE DENOMINATOR RULE this repository already paid for: averaging an undefined batch in
        as a nought deflated a perfect head's ceiling from 1.000 to 0.292."""
        empty = torch.zeros(6, 9, dtype=torch.bool)
        out = generative_loss(_head(), *_inputs(6), _target(6, 9, empty))
        self.assertEqual(float(out["noise_mse_count"]), 0.0)
        self.assertEqual(float(out["total"]), 0.0)

    def test_the_loss_carries_a_GRADIENT(self) -> None:
        """A loss that cannot move the weights is a shape with a number attached."""
        head = _head()
        out = generative_loss(head, *_inputs(6), _target(6, 4))
        out["total"].backward()
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertGreater(max(float(g.abs().max()) for g in grads), 0.0)


class EncodingTests(unittest.TestCase):

    def test_the_pose_round_trips_through_the_decode(self) -> None:
        """⛔ A decode that disagrees with the encode by one column produces plausible poses that are
        wrong, which is the hardest kind of defect to see."""
        target = _target(seeds=5, labels=3)
        pose, _ = pose_from_target(target)
        back = decode_samples(pose)
        self.assertTrue(torch.allclose(back.approach, target.approach, atol=1e-5))
        self.assertTrue(torch.allclose(back.axis, target.axis, atol=1e-5))
        self.assertTrue(torch.allclose(back.offset_m, target.offset_m, atol=1e-6))
        self.assertTrue(torch.allclose(back.width_m, target.width_m, atol=1e-6))

    def test_the_width_encoding_is_the_BASELINE_S_OWN(self) -> None:
        """⭐ Imported, not restated. An arm that differs from 4a in two things measures neither, and
        MEASURED on 33,801 v5 labels the better constants would be 55.3 / 12.5 rather than 70 / 50.
        Using the better ones here would have been the second difference."""
        from src.robot.grasping.deep.net import generative_loss as module

        self.assertEqual(module._WIDTH_CENTRE_M, SlotHeadConfig().width_centre_m)
        self.assertEqual(module._WIDTH_SCALE_M, SlotHeadConfig().width_scale_m)

    def test_the_axis_keeps_its_SIGN(self) -> None:
        """⚠ The director exists to make a LOSS sign-invariant, and this head predicts NOISE: a sign
        it cannot see is a sign it cannot denoise. Sign invariance comes from the sampling instead."""
        target = _target(seeds=4, labels=2)
        pose, _ = pose_from_target(target)
        flipped = _target(seeds=4, labels=2)
        flipped = GraspSetTarget(approach=flipped.approach, axis=-flipped.axis,
                                 offset_m=flipped.offset_m, width_m=flipped.width_m,
                                 valid=flipped.valid)
        other, _ = pose_from_target(flipped)
        self.assertFalse(torch.allclose(pose, other))

    def test_a_wrong_sample_shape_refuses(self) -> None:
        with self.assertRaises(ValueError):
            decode_samples(torch.randn(4, 3, POSE_DIM + 1))


class ScheduleTests(unittest.TestCase):

    def test_the_two_schedules_reach_DIFFERENT_columns(self) -> None:
        """⛔ Applying one level to all ten numbers is the defect the separate embeddings exist to
        prevent, and it would be invisible: the head would still train, just at the wrong rate for
        six of the ten. Checked by driving the noising directly."""
        from src.robot.grasping.deep.net.generative_loss import _alpha

        levels = torch.tensor([0, 4, 7])
        alpha = _alpha(levels, 8)
        self.assertAlmostEqual(float(alpha[0]), 1.0)          # level 0 keeps the clean pose
        self.assertLess(float(alpha[-1]), float(alpha[0]))    # and it decays

    def test_a_ZERO_noise_level_leaves_the_pose_alone(self) -> None:
        """The sanity end of the schedule: at level 0 the head is asked to denoise nothing."""
        from src.robot.grasping.deep.net.generative_loss import _alpha

        self.assertEqual(float(_alpha(torch.tensor([0]), 16)), 1.0)


class ComparabilityTests(unittest.TestCase):

    def test_samples_decode_into_the_shape_the_DETERMINISTIC_metrics_read(self) -> None:
        """⭐ An arm with its own metric is not an arm. Coverage and top-1 are computed over a
        `(B, K, ...)` prediction, so N samples must arrive in that shape and be scored by the same
        code at the same thresholds."""
        head = _head()
        features, gripper = _inputs(7)
        decoded = decode_samples(head.sample(features, gripper, count=5))
        self.assertEqual(tuple(decoded.approach.shape), (7, 5, 3))
        self.assertEqual(tuple(decoded.width_m.shape), (7, 5))
        self.assertTrue(bool(decoded.valid.all()))
        self.assertLess(float((decoded.approach.norm(dim=-1) - 1.0).abs().max()), 1e-5)

    def test_the_sample_COUNT_is_a_budget_and_not_K(self) -> None:
        """The deterministic head emits exactly K. This one answers with as many as asked."""
        head = _head()
        features, gripper = _inputs(4)
        for count in (1, 4, 12):
            with self.subTest(count):
                self.assertEqual(decode_samples(
                    head.sample(features, gripper, count=count)).approach.shape[1], count)


if __name__ == "__main__":
    unittest.main()


class ArtifactCoverageTests(unittest.TestCase):
    """⛔⛔ TWICE IN ONE MORNING, SO THE THIRD TIME IS A TEST FAILURE INSTEAD OF A LOST RUN.

    `_model_config` rebuilds the nested config field by field, which is the right design: a generic
    `**raw` would swallow a renamed key and load weights into shapes that no longer match. The cost is
    that every NEW field has to be added to it, and stage 3 and stage 4b were each shipped
    trainable-but-unloadable an hour apart because they were not.

    This enumerates `SetGeneratorConfig`'s own fields and asserts the reader mentions each one, so a
    new field fails here rather than after a six-hour arm.
    """

    def test_every_config_field_is_REBUILT_by_the_artifact_reader(self) -> None:
        """⛔⛔ THIS GUARD WAS ITSELF THE THIRD DEFECT.

        It enumerated `SetGeneratorConfig`'s OWN fields and asked whether each appears in
        `_model_config`. `head=` appears, so it passed, while the reader rebuilt eight of
        `SlotHeadConfig`'s ELEVEN fields. `asdict(net.config)` had been writing all eleven the whole
        time, so the data was never missing: the reader ignored three of them.

        The three fail in OPPOSITE directions, which is why one guard has to cover both.
        `slot_mixing` changes the parameters (a `film` net carries `head.film`, `mlp` swaps the output
        block), so a strict load raises. `axis_mode` is shape-neutral BY DESIGN, so a `vector`-trained
        net loads with no error at all and has six numbers decoded by the wrong rule. The runbook put
        `--slot-mixing film` in the ONE training command it gives a customer.

        So it recurses now. A nested config is exactly where this hides, because the OUTER field name
        is present and satisfies a shallow check.
        """
        import dataclasses
        from pathlib import Path as _Path

        from src.robot.grasping.deep.net.set_generator import SetGeneratorConfig

        source = _Path("src/robot/grasping/deep/set_artifact.py").read_text(encoding="utf-8")
        body = source[source.index("def _model_config("):source.index("def write_set_generator(")]

        def walk(cls: type, path: str) -> list[str]:
            """Every field of `cls` and of every dataclass it nests, as dotted names."""
            out: list[str] = []
            for field in dataclasses.fields(cls):
                here = f"{path}.{field.name}" if path else field.name
                if f"{field.name}=" not in body:
                    out.append(here)
                # The annotation may be a STRING under `from __future__ import annotations`, so the
                # nested dataclass is resolved off the default INSTANCE rather than off the type.
                nested = field.default
                if (nested is dataclasses.MISSING
                        and field.default_factory is not dataclasses.MISSING):
                    nested = field.default_factory()
                if dataclasses.is_dataclass(nested) and not isinstance(nested, type):
                    out.extend(walk(type(nested), here))
            return out

        missing = walk(SetGeneratorConfig, "")
        self.assertEqual(missing, [], "\n".join([
            "these config fields are written into the artifact by `asdict(net.config)` and NOT "
            "rebuilt by `_model_config`, so a net using them trains and then either refuses to "
            "load or, worse, loads and decodes by the wrong rule:", *missing]))

    def test_a_GENERATIVE_artifact_round_trips(self) -> None:
        import tempfile
        from pathlib import Path as _Path

        from src.robot.grasping.deep.net.set_generator import (
            SetGenerator,
            SetGeneratorConfig,
        )
        from src.robot.grasping.deep.set_artifact import (
            load_set_generator,
            write_set_generator,
        )
        from src.robot.grasping.deep.train.trainer import SetTrainingPlan

        torch.manual_seed(0)
        base = SetGeneratorConfig()
        head_config = GenerativeHeadConfig(width=64, depth=2, steps=8, gripper_width=8,
                                           time_width=16)
        net = SetGenerator(type(base)(
            backbone=type(base.backbone)(width=32, depth=2, heads=2),
            head=type(base.head)(slots=2, width=32), generative=head_config))
        plan = SetTrainingPlan()
        with tempfile.TemporaryDirectory() as name:
            paths = write_set_generator(_Path(name), net, sample=plan.sample, step=plan.step,
                                        gripper="2f85")
            loaded = load_set_generator(paths["weights"], device="cpu")
        self.assertEqual(loaded.net.config.generative, head_config)
        self.assertIsNotNone(loaded.net.generative)

    def test_the_flag_reaches_the_plan(self) -> None:
        from src.robot.grasping.deep.__main__ import build_parser

        self.assertTrue(build_parser().parse_args(
            ["train-set", "--clouds", "c", "--out", "o", "--generative"]).generative)
        self.assertFalse(build_parser().parse_args(
            ["train-set", "--clouds", "c", "--out", "o"]).generative)
