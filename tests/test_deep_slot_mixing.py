"""Whether a slot may differ from its neighbour in a way the INPUT can change.

⛔⛔ THE DEFECT THIS PINS. The head concatenates a per-slot query onto the shared feature and applies
one `nn.Linear`. Linear is affine, so the output is `W_s @ shared(x) + (W_q @ q_k + b)`: the
input-dependent term is IDENTICAL for every slot, and the slot-dependent term cannot see the input.
The K slots are a rigid constellation that translates with the shared prediction and can never change
shape, whatever the seed looks like.

Two consequences, both measured before these tests were written:

    `argmax` over confidence picks the SAME slot for every seed, so every `top1` figure reported
    through it has been reading one fixed slot rather than the head's answer

    `coverage` is capped by a fan that cannot spread or close, which is why it sat at 0.007 against a
    held-out ceiling of 0.453

The module docstring said a per-slot embedding "lets a slot specialise and keep its specialisation".
The intent was right and the arithmetic did not deliver it. These tests hold the defect in place under
its own name, so the default stays byte-identical for the arms already measured, and they hold the two
repairs to what they claim.
"""

from __future__ import annotations

import unittest

import torch

from src.robot.grasping.deep.net.slot_head import (
    SLOT_MIXING, SlotGraspHead, SlotHeadConfig)


def _head(mixing: str, *, slots: int = 4, seed: int = 0) -> SlotGraspHead:
    torch.manual_seed(seed)
    return SlotGraspHead(16, SlotHeadConfig(slots=slots, slot_mixing=mixing))


def _inputs(count: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    return torch.randn(count, 16), torch.randn(count, 14)


class DefectTests(unittest.TestCase):
    """`affine` is the default and it is the shape with the defect. Pinned, not fixed in place."""

    def test_the_slot_to_slot_difference_is_a_frozen_constant(self) -> None:
        """⭐ The confidence channel is emitted raw, so this reads the affine structure directly."""
        head = _head("affine")
        with torch.no_grad():
            confidence = head(*_inputs()).confidence
        gap = confidence - confidence[:, :1]
        self.assertLess(float(gap.std(dim=0).max()), 1e-6,
                        "an affine head's slot difference must not vary with the input")

    def test_the_confidence_argmax_is_the_same_slot_for_every_seed(self) -> None:
        """⛔ THE INSTRUMENT DEFECT. `top1` is scored on the argmax slot, so under `affine` it has
        been measuring one fixed slot rather than the head's answer."""
        head = _head("affine")
        with torch.no_grad():
            winners = head(*_inputs()).confidence.argmax(dim=1)
        self.assertEqual(len(torch.unique(winners)), 1)


class RepairTests(unittest.TestCase):

    def test_film_starts_numerically_identical_to_affine(self) -> None:
        """⭐ WHY THE SCALE AND SHIFT ARE ZEROS AND BUILT LAST. Zeros mean scale 1 and shift 0, and
        building the parameter after every random one leaves the RNG stream untouched. So a `film`
        arm and an `affine` arm begin as the same model, and any difference in their curves is the
        repair rather than a different initialisation."""
        plain, repaired = _head("affine"), _head("film")
        with torch.no_grad():
            first, second = plain(*_inputs()), repaired(*_inputs())
        for name in ("approach", "axis_params", "offset_m", "width_m", "confidence"):
            with self.subTest(name):
                self.assertTrue(torch.equal(getattr(first, name), getattr(second, name)))

    def test_film_learns_a_slot_difference_the_input_can_change(self) -> None:
        """The claim the repair has to earn. A target needing a different direction per (seed, slot)
        is unreachable for `affine` by construction, and `film` should reach it."""
        head = _head("film")
        features, gripper = _inputs()
        torch.manual_seed(2)
        want = torch.randn(len(features), head.config.slots, 3)
        before = float((head(features, gripper).approach
                        - head(features, gripper).approach[:, :1]).std(dim=0).max())
        optimiser = torch.optim.Adam(head.parameters(), lr=0.05)
        for _ in range(200):
            optimiser.zero_grad()
            torch.nn.functional.mse_loss(head(features, gripper).approach, want).backward()
            optimiser.step()
        with torch.no_grad():
            spread = head(features, gripper).approach
            after = float((spread - spread[:, :1]).std(dim=0).max())
        self.assertGreater(after, before * 50.0,
                           f"film did not learn an input-dependent slot difference "
                           f"({before:.4f} -> {after:.4f})")

    def test_mlp_separates_its_slots_from_the_start(self) -> None:
        """A nonlinearity between the concatenation and the outputs means the query and the feature
        interact at initialisation, so the confidence argmax is not a constant."""
        head = _head("mlp")
        with torch.no_grad():
            winners = head(*_inputs()).confidence.argmax(dim=1)
        self.assertGreater(len(torch.unique(winners)), 1)

    def test_every_repair_keeps_the_output_contract(self) -> None:
        for mixing in SLOT_MIXING:
            with self.subTest(mixing):
                head = _head(mixing, slots=3)
                out = head(*_inputs(32))
                self.assertEqual(out.approach.shape, (32, 3, 3))
                self.assertEqual(out.axis_params.shape, (32, 3, 6))
                self.assertEqual(out.offset_m.shape, (32, 3, 3))
                self.assertEqual(out.width_m.shape, (32, 3))
                self.assertEqual(out.confidence.shape, (32, 3))


class RefusalTests(unittest.TestCase):

    def test_an_unknown_mixing_refuses_by_name(self) -> None:
        with self.assertRaises(ValueError) as caught:
            SlotGraspHead(16, SlotHeadConfig(slot_mixing="whatever_sounds_plausible"))
        self.assertIn("slot_mixing", str(caught.exception))

    def test_the_default_is_still_the_measured_one(self) -> None:
        """⚠ Twenty-five arms were measured on `affine`. Changing this default silently would make
        every one of their numbers a comparison against a different model."""
        self.assertEqual(SlotHeadConfig().slot_mixing, "affine")
        self.assertEqual(SLOT_MIXING[0], "affine")


if __name__ == "__main__":
    unittest.main()
