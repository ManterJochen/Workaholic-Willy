"""Turning K slots into grasps a cell can execute.

⛔ THE FRAME ROUND TRIP IS THE FOURTH ONE THIS ARC HAS HAD TO GET RIGHT, and none of the first three
raised anything:

* the offset target in a frame that flips under the gripper's own symmetry,
* the grasp table emitted in the raw scene frame while the cloud was local,
* the closing axis whose sign the encoding discarded.

`build_sample` hands the network a LOCAL cloud, xy with the scene mean removed and z with the support
height subtracted, and carries `centre_xy_mm` / `support_height_mm` so it can be undone. A decoder
that forgot would emit every grasp offset by a per-scene constant: invisible in a loss curve, a
gripper in the wrong place in a cell.

So the round trip is checked against a HAND-COMPUTED position rather than against the decoder's own
arithmetic.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.robot.grasping.deep.net.rotation import director_matrix
from src.robot.grasping.deep.net.set_loss import GraspSetPrediction
from src.robot.grasping.deep.set_decode import as_metadata, decode_set_prediction


def _params_of(axis: torch.Tensor) -> torch.Tensor:
    matrix = director_matrix(axis)
    return torch.stack([matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2],
                        matrix[..., 1, 1], matrix[..., 1, 2], matrix[..., 2, 2]], dim=-1)


def _prediction(seeds: int = 3, slots: int = 2, seed: int = 0) -> GraspSetPrediction:
    generator = torch.Generator().manual_seed(seed)
    axis = torch.randn(seeds, slots, 3, generator=generator)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    return GraspSetPrediction(
        approach=torch.randn(seeds, slots, 3, generator=generator),
        axis_params=_params_of(axis),
        offset_m=torch.randn(seeds, slots, 3, generator=generator) * 0.02,
        width_m=torch.rand(seeds, slots, generator=generator) * 0.08,
        confidence=torch.randn(seeds, slots, generator=generator))


def _decode(prediction: GraspSetPrediction, points_m: torch.Tensor, seed_index: torch.Tensor,
            **kwargs: object) -> list:
    base = {"centre_xy_mm": np.array([120.0, -45.0]), "support_height_mm": 12.0,
            "min_width_mm": 5.0, "max_width_mm": 85.0}
    return decode_set_prediction(prediction, seed_index, points_m,
                                 **{**base, **kwargs})  # type: ignore[arg-type]


class FrameTests(unittest.TestCase):

    def test_the_position_is_the_hand_computed_one(self) -> None:
        """⛔ Checked against arithmetic done here, not against the decoder's own. A round trip that
        verified itself would agree with any consistent mistake."""
        points_m = torch.tensor([[0.100, 0.200, 0.050]])
        prediction = GraspSetPrediction(
            approach=torch.tensor([[[0.0, 0.0, -1.0]]]),
            axis_params=_params_of(torch.tensor([[[1.0, 0.0, 0.0]]])),
            offset_m=torch.tensor([[[0.010, -0.020, 0.005]]]),
            width_m=torch.tensor([[0.040]]),
            confidence=torch.tensor([[1.0]]))
        grasp = _decode(prediction, points_m, torch.tensor([0]))[0]
        # local metres -> local millimetres -> undo the sample's own centring
        expected = np.array([(0.100 + 0.010) * 1000.0 + 120.0,
                             (0.200 - 0.020) * 1000.0 - 45.0,
                             (0.050 + 0.005) * 1000.0 + 12.0])
        self.assertTrue(np.allclose(grasp.position_mm, expected, atol=1e-6),
                        f"{grasp.position_mm} != {expected}")

    def test_the_seed_position_gets_the_same_transform(self) -> None:
        """The seed is a diagnostic and it travels beside the grasp. Two transforms where there
        should be one is how a proposal comes to look like it was seeded off the object."""
        points_m = torch.tensor([[0.100, 0.200, 0.050]])
        prediction = GraspSetPrediction(
            approach=torch.tensor([[[0.0, 0.0, -1.0]]]),
            axis_params=_params_of(torch.tensor([[[1.0, 0.0, 0.0]]])),
            offset_m=torch.zeros(1, 1, 3), width_m=torch.tensor([[0.04]]),
            confidence=torch.tensor([[1.0]]))
        grasp = _decode(prediction, points_m, torch.tensor([0]))[0]
        self.assertTrue(np.allclose(grasp.seed_position_mm, grasp.position_mm, atol=1e-6))
        self.assertTrue(np.allclose(grasp.seed_position_mm,
                                    np.array([220.0, 155.0, 62.0]), atol=1e-6))

    def test_a_zero_transform_is_the_identity(self) -> None:
        points_m = torch.tensor([[0.0, 0.0, 0.0]])
        prediction = GraspSetPrediction(
            approach=torch.tensor([[[0.0, 0.0, -1.0]]]),
            axis_params=_params_of(torch.tensor([[[1.0, 0.0, 0.0]]])),
            offset_m=torch.tensor([[[0.001, 0.002, 0.003]]]),
            width_m=torch.tensor([[0.04]]), confidence=torch.tensor([[0.0]]))
        grasp = _decode(prediction, points_m, torch.tensor([0]),
                        centre_xy_mm=np.zeros(2), support_height_mm=0.0)[0]
        self.assertTrue(np.allclose(grasp.position_mm, np.array([1.0, 2.0, 3.0]), atol=1e-9))


class GeometryTests(unittest.TestCase):

    def test_the_frame_is_orthonormal(self) -> None:
        points_m = torch.rand(4, 3) * 0.3
        for grasp in _decode(_prediction(seeds=4, slots=3), points_m, torch.arange(4)):
            self.assertAlmostEqual(float(np.linalg.norm(grasp.approach)), 1.0, places=6)
            self.assertAlmostEqual(float(np.linalg.norm(grasp.axis)), 1.0, places=6)
            self.assertLess(abs(float(grasp.approach @ grasp.axis)), 1e-6)

    def test_the_approach_keeps_its_sign(self) -> None:
        """The approach points INTO the object; flipping it is a different grasp, not the same one."""
        points_m = torch.zeros(1, 3)
        prediction = GraspSetPrediction(
            approach=torch.tensor([[[0.0, 0.0, -2.0]]]),
            axis_params=_params_of(torch.tensor([[[1.0, 0.0, 0.0]]])),
            offset_m=torch.zeros(1, 1, 3), width_m=torch.tensor([[0.04]]),
            confidence=torch.tensor([[0.0]]))
        grasp = _decode(prediction, points_m, torch.tensor([0]))[0]
        self.assertTrue(np.allclose(grasp.approach, np.array([0.0, 0.0, -1.0]), atol=1e-6))

    def test_the_width_is_clamped_to_what_the_gripper_has(self) -> None:
        """⭐ CLAMPED HERE AND DELIBERATELY NOT IN THE HEAD. The head regresses freely so the gripper
        conditioning can be MEASURED rather than hard-wired; a cell cannot be handed an opening its
        gripper does not have. Two jobs, two places."""
        points_m = torch.zeros(2, 3)
        prediction = GraspSetPrediction(
            approach=torch.tensor([[[0.0, 0.0, -1.0]], [[0.0, 0.0, -1.0]]]),
            axis_params=_params_of(torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])),
            offset_m=torch.zeros(2, 1, 3),
            width_m=torch.tensor([[0.500], [0.0001]]),
            confidence=torch.zeros(2, 1))
        widths = sorted(g.width_mm for g in _decode(prediction, points_m, torch.tensor([0, 1])))
        self.assertEqual(widths, [5.0, 85.0])


class OrderingTests(unittest.TestCase):

    def test_the_list_is_ordered_by_confidence(self) -> None:
        """⚠ THE ORDER IS THE PRODUCT. A cell takes the first candidate it can reach, so emission
        order would hand it slot zero of seed zero whatever the head thought."""
        grasps = _decode(_prediction(seeds=5, slots=4), torch.rand(5, 3) * 0.3, torch.arange(5))
        scores = [g.score for g in grasps]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(len(grasps), 20)

    def test_a_confidence_floor_drops_the_slots_the_head_calls_empty(self) -> None:
        grasps = _decode(_prediction(seeds=5, slots=4), torch.rand(5, 3) * 0.3, torch.arange(5),
                         min_confidence=0.5)
        self.assertLess(len(grasps), 20)
        self.assertTrue(all(g.score >= 0.5 for g in grasps))

    def test_the_score_is_a_probability(self) -> None:
        for grasp in _decode(_prediction(seeds=4, slots=3), torch.rand(4, 3) * 0.3,
                             torch.arange(4)):
            self.assertGreaterEqual(grasp.score, 0.0)
            self.assertLessEqual(grasp.score, 1.0)


class MetadataTests(unittest.TestCase):

    def test_the_metadata_says_the_confidence_is_not_calibrated(self) -> None:
        """It is a BCE output over "this slot answers a real grasp", never a probability that the
        grasp holds, and the key says so wherever it travels."""
        grasp = _decode(_prediction(), torch.rand(3, 3) * 0.3, torch.arange(3))[0]
        row = as_metadata(grasp)
        self.assertIs(row["confidence_is_calibrated"], False)
        self.assertEqual(row["generator"], "deep_set")

    def test_the_metadata_traces_a_grasp_back_to_its_slot(self) -> None:
        """What makes a set head's output readable: which of the K answers did the cell take."""
        grasp = _decode(_prediction(), torch.rand(3, 3) * 0.3, torch.arange(3))[0]
        row = as_metadata(grasp)
        self.assertIn("slot", row)
        self.assertIn("seed_index", row)
        self.assertEqual(len(row["seed_position_mm"]), 3)


class RefusalTests(unittest.TestCase):

    def test_refusals(self) -> None:
        points_m = torch.rand(3, 3)
        with self.assertRaises(ValueError):
            _decode(_prediction(seeds=3), points_m, torch.arange(2))
        with self.assertRaises(ValueError):
            _decode(_prediction(seeds=3), points_m, torch.arange(3),
                    min_width_mm=90.0, max_width_mm=85.0)
        flat = GraspSetPrediction(approach=torch.rand(3, 3), axis_params=torch.rand(3, 6),
                                  offset_m=torch.rand(3, 3), width_m=torch.rand(3),
                                  confidence=torch.rand(3))
        with self.assertRaises(ValueError):
            _decode(flat, points_m, torch.arange(3))


if __name__ == "__main__":
    unittest.main()
