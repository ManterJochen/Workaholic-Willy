"""Reading a detection dataset: the COCO and YOLO layouts labelling tools export, the fixed validation split, and
every box or image left out with its reason."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.models.detection.closed_set.training.datasets import (
    LabelledImage,
    dataset_stats,
    load_dataset,
    split_off,
    write_shapes_dataset,
)
from tests._detector_training_fixtures import CLASSES, write_coco, write_yolo_split


class CocoLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_train_and_val_folders_are_read_with_the_dataset_s_own_validation(self) -> None:
        write_coco(self.root / "train", 6)
        write_coco(self.root / "val", 2, start=6)
        dataset = load_dataset(self.root)
        self.assertEqual("coco", dataset.format)
        self.assertEqual(CLASSES, dataset.classes)
        self.assertEqual((6, 2, "declared"), (len(dataset.train), len(dataset.val), dataset.val_source))
        first = dataset.train[0]
        self.assertEqual((96, 64), (first.width, first.height))
        self.assertEqual("train/images/img_000.png", first.key)
        # COCO's x y w h becomes the corners.
        x1, y1, x2, y2 = first.boxes[0].xyxy
        self.assertEqual((4.0, 4.0, 20.0, 20.0), (x1, y1, x2, y2))

    def test_a_single_folder_gives_fifteen_percent_to_validation_the_same_way_every_time(self) -> None:
        write_coco(self.root, 20)
        first = load_dataset(self.root)
        second = load_dataset(self.root)
        self.assertEqual(("split", 3, 17), (first.val_source, len(first.val), len(first.train)))
        self.assertEqual([image.key for image in first.val], [image.key for image in second.val])
        self.assertEqual(0.15, first.val_fraction)

    def test_a_declared_validation_split_wins_over_the_fraction(self) -> None:
        write_coco(self.root / "train", 6)
        write_coco(self.root / "valid", 2, start=6)
        dataset = load_dataset(self.root, val_fraction=0.5)
        self.assertEqual(("declared", 2, 0.0), (dataset.val_source, len(dataset.val), dataset.val_fraction))

    def test_roboflow_layout_and_its_empty_super_category(self) -> None:
        categories = [{"id": 0, "name": "parts", "supercategory": "none"}, {"id": 1, "name": "cube"},
                      {"id": 2, "name": "disc"}]
        write_coco(self.root / "train", 5, images_dir="", annotations="_annotations.coco.json",
                   categories=categories)
        write_coco(self.root / "valid", 2, start=5, images_dir="", annotations="_annotations.coco.json",
                   categories=categories)
        dataset = load_dataset(self.root)
        self.assertEqual(CLASSES, dataset.classes)
        self.assertEqual("declared", dataset.val_source)
        self.assertTrue(any("'parts'" in line and "no box" in line for line in dataset.skipped), dataset.skipped)

    def test_cvat_layout_with_annotations_beside_the_images(self) -> None:
        write_coco(self.root, 4, annotations="annotations/instances_default.json")
        dataset = load_dataset(self.root, val_fraction=0.0)
        self.assertEqual((4, "none"), (len(dataset.train), dataset.val_source))
        self.assertTrue(all(image.boxes for image in dataset.train))

    def test_crowd_unknown_degenerate_and_outside_boxes_are_named(self) -> None:
        extra = [
            {"id": 900, "image_id": 1, "category_id": 1, "bbox": [1, 1, 10, 10], "iscrowd": 1},
            {"id": 901, "image_id": 1, "category_id": 7, "bbox": [1, 1, 10, 10]},
            {"id": 902, "image_id": 1, "category_id": 1, "bbox": [200, 200, 10, 10]},
            {"id": 903, "image_id": 2, "category_id": 2, "bbox": [80, 50, 40, 40]},
            {"id": 904, "image_id": 99, "category_id": 2, "bbox": [1, 1, 5, 5]},
        ]
        write_coco(self.root, 3, extra=extra)
        dataset = load_dataset(self.root, val_fraction=0.0)
        text = " | ".join(dataset.skipped)
        for phrase in ("crowd region", "a class the dataset does not declare", "under 1 px", "cut to it",
                       "name an image the file does not declare"):
            self.assertIn(phrase, text)
        clipped = [box for box in dataset.train[1].boxes if box.xyxy[2] == 96.0 and box.xyxy[3] == 64.0]
        self.assertEqual(1, len(clipped))

    def test_a_missing_image_is_named_and_left_out(self) -> None:
        write_coco(self.root, 3)
        (self.root / "images" / "img_001.png").unlink()
        dataset = load_dataset(self.root, val_fraction=0.0)
        self.assertEqual(2, len(dataset.train))
        self.assertTrue(any("not found" in line and "img_001.png" in line for line in dataset.skipped))

    def test_one_category_list_per_export(self) -> None:
        write_coco(self.root / "train", 2)
        write_coco(self.root / "val", 1, start=2, categories=[{"id": 1, "name": "disc"}, {"id": 2, "name": "cube"}])
        with self.assertRaisesRegex(ValueError, "one export must use one category list"):
            load_dataset(self.root)

    def test_a_dataset_without_a_box_is_refused_unless_inspected(self) -> None:
        (self.root / "images").mkdir()
        (self.root / "annotations.json").write_text(json.dumps(
            {"images": [], "annotations": [], "categories": [{"id": 1, "name": "cube"}]}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "holds no box"):
            load_dataset(self.root)
        self.assertEqual(0, load_dataset(self.root, require_boxes=False).images)

    def test_neither_format_names_what_was_looked_for(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "neither a COCO nor a YOLO dataset"):
            load_dataset(self.root)

    def test_the_fingerprint_follows_the_annotations(self) -> None:
        path = write_coco(self.root, 4)
        before = load_dataset(self.root).fingerprint
        self.assertEqual(before, load_dataset(self.root).fingerprint)
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["annotations"][0]["bbox"][0] += 1
        path.write_text(json.dumps(doc), encoding="utf-8")
        self.assertNotEqual(before, load_dataset(self.root).fingerprint)


class YoloLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_an_ultralytics_data_yaml(self) -> None:
        write_yolo_split(self.root / "images" / "train", self.root / "labels" / "train", 5)
        write_yolo_split(self.root / "images" / "val", self.root / "labels" / "val", 2, start=5)
        (self.root / "data.yaml").write_text(
            "path: .\ntrain: images/train\nval: images/val\nnames:\n  - cube\n  - disc\n", encoding="utf-8")
        dataset = load_dataset(self.root)
        self.assertEqual(("yolo", CLASSES, 5, 2), (dataset.format, dataset.classes, len(dataset.train),
                                                   len(dataset.val)))
        x1, y1, x2, y2 = dataset.train[0].boxes[0].xyxy
        self.assertAlmostEqual(4.0, x1, places=3)
        self.assertAlmostEqual(20.0, y2, places=3)

    def test_a_roboflow_data_yaml_with_its_parent_paths_polygons_and_background_images(self) -> None:
        write_yolo_split(self.root / "train" / "images", self.root / "train" / "labels", 4)
        write_yolo_split(self.root / "valid" / "images", self.root / "valid" / "labels", 2, start=4)
        (self.root / "train" / "labels" / "img_000.txt").write_text(
            "1 0.1 0.1 0.4 0.1 0.4 0.5 0.1 0.5\n5 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (self.root / "train" / "labels" / "img_001.txt").unlink()
        (self.root / "data.yaml").write_text(
            "train: ../train/images\nval: ../valid/images\nnc: 2\nnames: {0: cube, 1: disc}\n", encoding="utf-8")
        dataset = load_dataset(self.root)
        self.assertEqual((4, 2), (len(dataset.train), len(dataset.val)))
        polygon = dataset.train[0].boxes[0]
        self.assertEqual(1, polygon.label)
        self.assertAlmostEqual(0.4 * 96, polygon.xyxy[2], places=3)
        self.assertEqual((), dataset.train[1].boxes)
        self.assertTrue(any("a class the dataset does not declare" in line for line in dataset.skipped))

    def test_a_classes_list_with_flat_folders_is_split_here(self) -> None:
        write_yolo_split(self.root / "images", self.root / "labels", 10)
        (self.root / "classes.txt").write_text("cube\ndisc\n", encoding="utf-8")
        dataset = load_dataset(self.root)
        self.assertEqual(("yolo", "split", 8, 2), (dataset.format, dataset.val_source, len(dataset.train),
                                                   len(dataset.val)))

    def test_nc_must_agree_with_the_names(self) -> None:
        write_yolo_split(self.root / "images", self.root / "labels", 2)
        (self.root / "data.yaml").write_text("train: images\nnc: 3\nnames: [cube, disc]\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "nc: 3"):
            load_dataset(self.root)


class SplitAndStatsTests(unittest.TestCase):
    def _images(self, count: int) -> list[LabelledImage]:
        return [LabelledImage(path=Path(f"{i}.png"), width=10, height=10, boxes=(), key=f"{i}.png")
                for i in range(count)]

    def test_split_bounds(self) -> None:
        self.assertEqual((1, 0), tuple(map(len, split_off(self._images(1), 0.15, 0))))
        self.assertEqual((1, 1), tuple(map(len, split_off(self._images(2), 0.15, 0))))
        self.assertEqual((85, 15), tuple(map(len, split_off(self._images(100), 0.15, 0))))
        self.assertEqual((3, 0), tuple(map(len, split_off(self._images(3), 0.0, 0))))

    def test_an_added_image_does_not_move_the_others(self) -> None:
        _, before = split_off(self._images(40), 0.15, 0)
        _, after = split_off(self._images(41), 0.15, 0)
        moved = {image.key for image in before} - {image.key for image in after}
        self.assertLessEqual(len(moved), 1)

    def test_stats_count_per_class_and_size_band(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            write_coco(Path(tmp), 6)
            stats = dataset_stats(load_dataset(tmp, val_fraction=0.0))
        self.assertEqual(6, stats["train"]["images"])
        self.assertEqual(sum(stats["train"]["boxes_per_class"].values()), stats["train"]["boxes"])
        self.assertEqual(stats["train"]["boxes"], stats["train"]["small"] + stats["train"]["medium"]
                         + stats["train"]["large"])

    def test_the_shapes_dataset_is_the_same_every_time(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as a, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as b:
            write_shapes_dataset(a, images=4)
            write_shapes_dataset(b, images=4)
            self.assertEqual((Path(a) / "annotations.json").read_text(encoding="utf-8"),
                             (Path(b) / "annotations.json").read_text(encoding="utf-8"))
            self.assertEqual((Path(a) / "images" / "shapes_0003.png").read_bytes(),
                             (Path(b) / "images" / "shapes_0003.png").read_bytes())
            dataset = load_dataset(a)
            self.assertEqual(("red_block", "green_disc", "blue_wedge"), dataset.classes)


if __name__ == "__main__":
    unittest.main()
