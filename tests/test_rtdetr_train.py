"""RT-DETR training-scaffolding tests: COCO parsing, class-map remap, manifest, CLI.

Covers the parts that import WITHOUT the training stack (torch / transformers / accelerate) --
the actual fine-tune loop needs those + labelled data (later). No heavy deps are imported here.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.models.detection.closed_set.train import (
    TrainConfig,
    _category_remap,
    build_id2label,
    build_manifest,
    load_coco_index,
    main,
    write_manifest,
)


def _coco_doc() -> dict:
    # Two images; NON-contiguous category ids (5, 2) to exercise the remap.
    return {
        "images": [
            {"id": 1, "file_name": "a.png", "width": 64, "height": 64},
            {"id": 2, "file_name": "b.png", "width": 64, "height": 64},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 5, "bbox": [1, 2, 10, 12], "area": 120, "iscrowd": 0},
            {"id": 11, "image_id": 1, "category_id": 2, "bbox": [5, 5, 8, 8]},
            {"id": 12, "image_id": 2, "category_id": 5, "bbox": [0, 0, 20, 20]},
            {"id": 13, "image_id": 99, "category_id": 2, "bbox": [0, 0, 1, 1]},  # dangling -> skipped
        ],
        "categories": [{"id": 5, "name": "rhino"}, {"id": 2, "name": "box"}],
    }


def _write_split(root: Path, doc: dict) -> Path:
    (root / "images").mkdir(parents=True, exist_ok=True)
    ann = root / "annotations.json"
    ann.write_text(json.dumps(doc), encoding="utf-8")
    return ann


class CocoParsingTests(unittest.TestCase):
    def test_load_groups_by_image_and_skips_dangling(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            ann = _write_split(Path(d), _coco_doc())
            idx = load_coco_index(ann)
        self.assertEqual(idx.categories, {2: "box", 5: "rhino"})  # sorted by id
        self.assertEqual(len(idx.records), 2)
        self.assertEqual(idx.num_annotations, 4)  # total incl. dangling (raw doc count)
        rec1 = next(r for r in idx.records if r.image_id == 1)
        self.assertEqual(len(rec1.coco_annotations), 2)
        # normalised: area filled from bbox when absent (bbox 8x8 -> 64)
        by_cat = {a["category_id"]: a for a in rec1.coco_annotations}
        self.assertAlmostEqual(by_cat[2]["area"], 64.0)

    def test_id2label_is_contiguous(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            idx = load_coco_index(_write_split(Path(d), _coco_doc()))
        # sorted category ids (2, 5) -> contiguous 0..1
        self.assertEqual(build_id2label(idx), {0: "box", 1: "rhino"})
        self.assertEqual(_category_remap(idx), {2: 0, 5: 1})

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_coco_index("does/not/exist.json")

    def test_malformed_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "annotations.json"
            bad.write_text(json.dumps({"images": [], "annotations": []}), encoding="utf-8")  # no categories key
            with self.assertRaises(ValueError):
                load_coco_index(bad)

    def test_no_categories_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "annotations.json"
            bad.write_text(json.dumps({"images": [], "annotations": [], "categories": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_coco_index(bad)


class ManifestTests(unittest.TestCase):
    def test_manifest_fields_and_sha(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            idx = load_coco_index(_write_split(Path(d), _coco_doc()))
            man = build_manifest(config=TrainConfig(epochs=3), index=idx, output_dir="out/dir",
                                 metrics={"final_train_loss": 1.5}, trained_at="2026-07-16T00:00:00+00:00")
            self.assertEqual(man["schema"], "willy.rtdetr.train_manifest/1")
            self.assertEqual(man["dataset"]["num_images"], 2)
            self.assertEqual(man["dataset"]["id2label"], {"0": "box", "1": "rhino"})
            self.assertEqual(len(man["dataset"]["annotations_sha256"]), 64)
            self.assertEqual(man["train_config"]["epochs"], 3)
            self.assertIn("python", man["env"])
            # canonical write is sorted-key + trailing newline
            path = write_manifest(man, Path(d) / "out")
            raw = path.read_bytes()
            self.assertTrue(raw.endswith(b"\n"))
            self.assertEqual(json.loads(raw)["schema"], "willy.rtdetr.train_manifest/1")


class CliTests(unittest.TestCase):
    def test_inspect_prints_class_map(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write_split(Path(d) / "train", _coco_doc())
            rc = main(["inspect", "--data-dir", d, "--split", "train"])
        self.assertEqual(rc, 0)

    def test_inspect_missing_split_exit_2(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            rc = main(["inspect", "--data-dir", d, "--split", "val"])
        self.assertEqual(rc, 2)

    def test_train_missing_data_exit_2_without_training_stack(self) -> None:
        # The data-existence check runs BEFORE the heavy imports, so a missing dataset returns 2
        # even on a machine without accelerate/transformers installed.
        with tempfile.TemporaryDirectory() as d:
            rc = main(["train", "--data-dir", d, "--output-dir", str(Path(d) / "out")])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
