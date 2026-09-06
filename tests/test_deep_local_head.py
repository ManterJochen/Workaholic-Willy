"""The head that emits K grasps for one seed.

⚠ THE FAILURE THAT LOOKS LIKE SUCCESS is slot collapse: every slot emits the same grasp, the loss
falls because one mode is always covered, and the coverage metric reports slots rather than modes. A
shape test cannot see it, so the tests below read whether the slots actually DIFFER, at
initialisation and after a fit.

The second thing pinned is that the width starts where the data is. MEASURED on v5 over 1,998,974
pairs, the median labelled opening is 70.3 mm against an 85 mm aperture: most grasps are nearly wide
open. A head whose output starts at zero starts 70 mm wrong on every seed.
"""

from __future__ import annotations

import unittest

import torch

from src.robot.grasping.deep.net.gripper import GRIPPER_VECTOR_DIM, swept_volume_vector
from src.robot.grasping.deep.net.slot_head import (
    SLOT_OUTPUTS,
    SlotGraspHead,
    SlotHeadConfig,
)
from src.robot.grasping.deep.net.rotation import director_decode
from src.robot.grasping.deep.net.assignment import optimal_one_to_one
from src.robot.grasping.deep.net.set_loss import (
    GraspSetTarget,
    grasp_set_cost,
    grasp_set_loss,
)
from src.robot.grasping.deep.train.step import SetStepConfig, _metrics


def _coverage(prediction, target) -> float:
    """⛔ COVERAGE MOVED TO `_metrics`, where the hit thresholds are, and this file kept reading it
    off the loss. It used to count every matched pair, and the matching is one-to-one and
    unconstrained, so it bound `min(K, L)` pairs however badly they fit: three unrelated random
    models scored 0.6906 each. It now counts only labels whose matched slot LANDED on them, which is
    what these tests meant all along."""
    matching = optimal_one_to_one(grasp_set_cost(prediction, target), target.valid)
    row = _metrics(prediction, target, matching, SetStepConfig())
    count = float(row["coverage_count"])
    return float(row["coverage_sum"]) / count if count else float("nan")

_JAW = {
    "aperture_mm": 85.0, "min_width_mm": 5.0, "finger_ahead_mm": 28.72,
    "finger_behind_mm": 33.37, "finger_thickness_mm": 31.35, "finger_width_mm": 27.0,
    "pad_span_mm": 38.0, "friction_coefficient": 0.5,
}


def _gripper(seeds: int) -> torch.Tensor:
    return swept_volume_vector(**_JAW).expand(seeds, GRIPPER_VECTOR_DIM)


class ShapeTests(unittest.TestCase):

    def test_it_emits_one_grasp_per_slot_per_seed(self) -> None:
        head = SlotGraspHead(64, SlotHeadConfig(slots=5))
        out = head(torch.randn(7, 64), _gripper(7))
        self.assertEqual(out.approach.shape, (7, 5, 3))
        self.assertEqual(out.axis_params.shape, (7, 5, 6))
        self.assertEqual(out.offset_m.shape, (7, 5, 3))
        self.assertEqual(out.width_m.shape, (7, 5))
        self.assertEqual(out.confidence.shape, (7, 5))

    def test_one_slot_is_the_old_single_output_shape(self) -> None:
        out = SlotGraspHead(32, SlotHeadConfig(slots=1))(torch.randn(4, 32), _gripper(4))
        self.assertEqual(out.approach.shape, (4, 1, 3))

    def test_the_slot_outputs_constant_matches_what_forward_slices(self) -> None:
        """Two spellings of one layout is how a head starts reading its own director parameters as
        an offset without anything failing."""
        head = SlotGraspHead(16)
        self.assertEqual(head.output.out_features, SLOT_OUTPUTS)
        self.assertEqual(3 + 6 + 3 + 1 + 1, SLOT_OUTPUTS)

    def test_refusals(self) -> None:
        with self.assertRaises(ValueError):
            SlotGraspHead(16, SlotHeadConfig(slots=0))
        with self.assertRaises(ValueError):
            SlotGraspHead(0)
        head = SlotGraspHead(16)
        with self.assertRaises(ValueError):
            head(torch.randn(4, 15), _gripper(4))
        with self.assertRaises(ValueError):
            head(torch.randn(4, 16), _gripper(3))

    def test_a_bare_gripper_vector_is_refused_rather_than_broadcast(self) -> None:
        """⛔ A silent broadcast is how an input that should vary per sample becomes one that never
        does, which is the exact failure the conditioning seam exists to expose."""
        head = SlotGraspHead(16)
        with self.assertRaises(ValueError):
            head(torch.randn(4, 16), swept_volume_vector(**_JAW))


class SlotDistinctnessTests(unittest.TestCase):
    """⭐ The failure a shape test cannot see."""

    def test_the_slots_differ_at_initialisation(self) -> None:
        head = SlotGraspHead(64, SlotHeadConfig(slots=6)).eval()
        with torch.no_grad():
            out = head(torch.randn(32, 64), _gripper(32))
        spread = out.approach.std(dim=1).max()
        self.assertGreater(float(spread), 1e-4,
                           "every slot emitted the same approach before training even started")

    def test_the_queries_are_what_makes_them_differ(self) -> None:
        """The mitigation is load-bearing, not decoration: zero the per-slot embeddings and every
        slot becomes the same function of the same shared feature."""
        head = SlotGraspHead(64, SlotHeadConfig(slots=6)).eval()
        with torch.no_grad():
            head.queries.zero_()
            out = head(torch.randn(32, 64), _gripper(32))
        self.assertLess(float(out.approach.std(dim=1).max()), 1e-6)

    def test_two_slots_learn_two_modes_rather_than_one_average(self) -> None:
        """⭐⭐ THE END-TO-END CLAIM, fitted rather than asserted.

        One seed, two labelled grasps 90 degrees apart in approach and at the same place. A
        single-output head can only land between them. Two slots plus one-to-one matching must end up
        covering both, and the coverage term is what says so.
        """
        head = SlotGraspHead(8, SlotHeadConfig(slots=2, width=64))
        features = torch.zeros(1, 8)          # one seed, no information: the slots are on their own
        gripper = _gripper(1)
        approach = torch.tensor([[[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]])
        axis = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
        target = GraspSetTarget(
            approach=approach, axis=axis, offset_m=torch.zeros(1, 2, 3),
            width_m=torch.tensor([[0.05, 0.05]]),
            valid=torch.ones(1, 2, dtype=torch.bool))

        optimiser = torch.optim.Adam(head.parameters(), lr=0.02)
        for _ in range(400):
            optimiser.zero_grad()
            grasp_set_loss(head(features, gripper), target)["total"].backward()
            optimiser.step()

        head.eval()
        with torch.no_grad():
            out = head(features, gripper)
        predicted = out.approach[0] / out.approach[0].norm(dim=-1, keepdim=True)
        for label in approach[0]:
            best = float((predicted @ label).max())
            self.assertGreater(best, 0.95,
                               "no slot came close to one of the two modes; the head averaged them")
        self.assertEqual(_coverage(out, target), 1.0)

    def test_a_single_slot_cannot_do_it(self) -> None:
        """The control. Without it the test above shows only that the head can fit something."""
        head = SlotGraspHead(8, SlotHeadConfig(slots=1, width=64))
        features, gripper = torch.zeros(1, 8), _gripper(1)
        approach = torch.tensor([[[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]])
        axis = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
        target = GraspSetTarget(
            approach=approach, axis=axis, offset_m=torch.zeros(1, 2, 3),
            width_m=torch.tensor([[0.05, 0.05]]), valid=torch.ones(1, 2, dtype=torch.bool))
        optimiser = torch.optim.Adam(head.parameters(), lr=0.02)
        for _ in range(400):
            optimiser.zero_grad()
            grasp_set_loss(head(features, gripper), target)["total"].backward()
            optimiser.step()
        head.eval()
        with torch.no_grad():
            coverage = _coverage(head(features, gripper), target)
        self.assertEqual(coverage, 0.5, "one slot covered more than one of two modes")


class OffsetParameterisationTests(unittest.TestCase):
    """⛔⛔⛔ THE DEFECT THAT SURVIVED A TWENTY-FOLD GRADIENT INCREASE.

    The seed is one of TWO contacts and nothing tells the head which. MEASURED over 14,358 pairs,
    `offset . axis / (width/2)` is within 0.2 of +-1 for 41.7 % of them and its sign is positive
    44.3 % of the time: a coin flip. The L2-optimal answer to a target that is `+v` or `-v` with equal
    probability is their MEAN, which is zero, so the free-vector head sat at 35 mm of error and did
    not move when its share of the loss went from 1.0 % to 17 %. It was already at its optimum.

    A unit vector cannot shrink to a mean, and the length is not in question: MEASURED, an oracle
    direction with half the opening as the magnitude reaches 5.7 mm, against 30.8 for predicting zero
    and 22.6 for half the opening along the closing axis.
    """

    def test_the_offset_length_is_half_the_predicted_width(self) -> None:
        head = SlotGraspHead(64, SlotHeadConfig(slots=3)).eval()
        with torch.no_grad():
            out = head(torch.randn(16, 64), _gripper(16))
        self.assertTrue(torch.allclose(out.offset_m.norm(dim=-1), out.width_m * 0.5, atol=1e-5))

    def test_the_offset_cannot_collapse_to_zero(self) -> None:
        """⭐ THE PROPERTY THE REPAIR RESTS ON. However the direction weights move, the length is the
        width head's, so the head cannot answer an ambiguous target by predicting nothing."""
        head = SlotGraspHead(64, SlotHeadConfig(slots=3))
        with torch.no_grad():
            for parameter in head.output.parameters():
                parameter.zero_()
            out = head(torch.randn(16, 64), _gripper(16))
        self.assertTrue(torch.all(out.offset_m.norm(dim=-1) > 0.01),
                        "a head with zeroed outputs still emitted a zero-length offset")

    def test_the_free_vector_behaviour_stays_reachable(self) -> None:
        """Every arm before the measurement ran on a free 3-vector, so it has to stay available or
        the two cannot be compared."""
        head = SlotGraspHead(64, SlotHeadConfig(slots=3, offset_from_width=False)).eval()
        with torch.no_grad():
            out = head(torch.randn(16, 64), _gripper(16))
        self.assertFalse(torch.allclose(out.offset_m.norm(dim=-1), out.width_m * 0.5, atol=1e-3))

    def test_a_near_zero_direction_does_not_produce_a_nan(self) -> None:
        """The only division in the head, and at initialisation the raw direction is near zero."""
        head = SlotGraspHead(8, SlotHeadConfig(slots=2))
        with torch.no_grad():
            head.output.weight.zero_()
            head.output.bias.zero_()
            out = head(torch.zeros(4, 8), _gripper(4))
        self.assertFalse(bool(torch.isnan(out.offset_m).any()))
        self.assertFalse(bool(torch.isinf(out.offset_m).any()))

    def test_a_wider_grasp_puts_the_centre_further_from_the_seed(self) -> None:
        """The geometry the coupling encodes: the centre is half an opening from the contact, so a
        wider grasp is a longer offset. A free vector had no reason to obey that."""
        head = SlotGraspHead(32, SlotHeadConfig(slots=1, width_centre_m=0.02)).eval()
        wide = SlotGraspHead(32, SlotHeadConfig(slots=1, width_centre_m=0.08)).eval()
        wide.load_state_dict(head.state_dict())
        with torch.no_grad():
            features = torch.randn(8, 32)
            narrow_offset = head(features, _gripper(8)).offset_m.norm(dim=-1).median()
            wide_offset = wide(features, _gripper(8)).offset_m.norm(dim=-1).median()
        self.assertGreater(float(wide_offset), float(narrow_offset) * 2.0)


class OutputScaleTests(unittest.TestCase):

    def test_an_untrained_head_proposes_a_plausible_width(self) -> None:
        """MEASURED median 70.3 mm. Starting at zero would put every seed 70 mm wrong before the
        first step, on a quantity the loss measures in millimetres."""
        head = SlotGraspHead(64).eval()
        with torch.no_grad():
            width = head(torch.randn(64, 64), _gripper(64)).width_m
        self.assertGreater(float(width.median()), 0.03)
        self.assertLess(float(width.median()), 0.11)

    def test_an_untrained_head_proposes_a_plausible_offset(self) -> None:
        """MEASURED median 30.6 mm, and half the median opening is 29.6. With the length taken from
        the width head an untrained offset is already the right SIZE and only the direction is
        unlearned, which is the whole point of the coupling."""
        head = SlotGraspHead(64).eval()
        with torch.no_grad():
            offset = head(torch.randn(64, 64), _gripper(64)).offset_m
        self.assertLess(float(offset.norm(dim=-1).max()), 0.5)
        self.assertGreater(float(offset.norm(dim=-1).median()), 0.01)

    def test_the_scales_are_configurable_rather_than_baked_in(self) -> None:
        """They are corpus statistics. A different corpus has different ones, and a head that hid
        them in its arithmetic could not say what it assumed."""
        head = SlotGraspHead(16, SlotHeadConfig(width_centre_m=0.02, width_scale_m=0.001)).eval()
        with torch.no_grad():
            width = head(torch.randn(32, 16), _gripper(32)).width_m
        self.assertAlmostEqual(float(width.median()), 0.02, delta=0.005)


class DecodeTests(unittest.TestCase):

    def test_the_axis_parameters_decode_to_a_unit_axis(self) -> None:
        head = SlotGraspHead(64, SlotHeadConfig(slots=3)).eval()
        with torch.no_grad():
            axis = director_decode(head(torch.randn(9, 64), _gripper(9)).axis_params)
        self.assertEqual(axis.shape, (9, 3, 3))
        self.assertTrue(torch.allclose(axis.norm(dim=-1), torch.ones(9, 3), atol=1e-4))


if __name__ == "__main__":
    unittest.main()
