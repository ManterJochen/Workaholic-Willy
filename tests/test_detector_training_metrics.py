"""COCO-style mAP, checked against values worked out by hand from COCOeval's rules."""

from __future__ import annotations

import unittest

import numpy as np

from src.models.detection.closed_set.training.metrics import box_iou, evaluate_detections

GT = (0.0, 0.0, 10.0, 10.0)


class MapTests(unittest.TestCase):
    def test_a_perfect_detection_scores_one_everywhere(self) -> None:
        scores = evaluate_detections([[(0, GT)]], [[(0, 0.9, GT)]], ["a"])
        self.assertEqual((1.0, 1.0, 1.0, 1.0), (scores.map, scores.map50, scores.map75, scores.mar100))
        self.assertEqual({"a": 1.0}, scores.per_class_ap)

    def test_an_iou_of_0_62_counts_at_three_of_the_ten_thresholds(self) -> None:
        detection = (0.0, 0.0, 10.0, 6.2)
        self.assertAlmostEqual(0.62, float(box_iou(np.array([detection]), np.array([GT]))[0, 0]))
        scores = evaluate_detections([[(0, GT)]], [[(0, 0.9, detection)]], ["a"])
        self.assertAlmostEqual(0.3, scores.map)
        self.assertEqual((1.0, 0.0), (scores.map50, scores.map75))

    def test_half_the_boxes_found_is_51_of_101_recall_points(self) -> None:
        other = (20.0, 20.0, 30.0, 30.0)
        scores = evaluate_detections([[(0, GT), (0, other)]], [[(0, 0.9, GT)]], ["a"])
        self.assertAlmostEqual(51 / 101, scores.map)
        self.assertAlmostEqual(0.5, scores.mar100)

    def test_a_false_positive_ranked_first_halves_the_precision(self) -> None:
        miss = (50.0, 50.0, 60.0, 60.0)
        scores = evaluate_detections([[(0, GT)]], [[(0, 0.9, miss), (0, 0.8, GT)]], ["a"])
        self.assertAlmostEqual(0.5, scores.map)

    def test_the_wrong_class_does_not_match(self) -> None:
        scores = evaluate_detections([[(0, GT)]], [[(1, 0.9, GT)]], ["a", "b"])
        self.assertEqual(0.0, scores.map)
        self.assertEqual({"a": 0.0}, scores.per_class_ap)  # b has no ground truth and is not averaged

    def test_one_detection_takes_one_box(self) -> None:
        scores = evaluate_detections([[(0, GT)]], [[(0, 0.9, GT), (0, 0.8, GT)]], ["a"])
        self.assertEqual(1.0, scores.mar100)
        self.assertAlmostEqual(1.0, scores.map)  # the duplicate ranks below every true positive

    def test_images_are_scored_together(self) -> None:
        miss = (50.0, 50.0, 60.0, 60.0)
        scores = evaluate_detections([[(0, GT)], [(0, GT)]], [[(0, 0.95, miss)], [(0, 0.5, GT)]], ["a"])
        # The miss outranks the hit across images: recall 0.5 at precision 0.5 after two detections.
        self.assertAlmostEqual(0.5 * 51 / 101, scores.map50)

    def test_only_a_hundred_detections_per_image_and_class_count(self) -> None:
        boxes = [(0, (float(i), 0.0, float(i) + 1.0, 1.0)) for i in range(0, 300, 2)]
        detections = [(0, 1.0 - i / 1000.0, box) for i, (_, box) in enumerate(boxes)]
        scores = evaluate_detections([boxes], [detections], ["a"])
        self.assertAlmostEqual(100 / 150, scores.mar100)

    def test_no_ground_truth_at_all_is_not_a_number(self) -> None:
        scores = evaluate_detections([[]], [[(0, 0.9, GT)]], ["a"])
        self.assertTrue(np.isnan(scores.map))
        self.assertEqual({}, scores.per_class_ap)

    def test_lengths_must_agree(self) -> None:
        with self.assertRaisesRegex(ValueError, "1 image"):
            evaluate_detections([[]], [[], []], ["a"])


if __name__ == "__main__":
    unittest.main()
