"""The RT-DETR command line and the small COCO reader it keeps for older callers.

Covers what runs WITHOUT the training stack (torch / transformers): COCO parsing, the class-map remap, ``inspect`` and
the exit codes. The training run itself is ``tests/test_detector_training_run.py``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.models.detection.closed_set.train import (
    _category_remap,
    build_id2label,
    load_coco_index,
    main,
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

    def test_inspect_prints_json_on_request(self) -> None:
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as d:
            _write_split(Path(d) / "train", _coco_doc())
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = main(["inspect", "--data-dir", d, "--json"])
        self.assertEqual(0, rc)
        self.assertEqual(["box", "rhino"], json.loads(out.getvalue())["classes"])

    def test_eval_without_a_model_folder_exit_2(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write_split(Path(d), _coco_doc())
            rc = main(["eval", "--data-dir", d, "--model-dir", str(Path(d) / "no_model")])
        self.assertEqual(2, rc)

    def test_a_setting_no_run_can_use_is_exit_2(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            rc = main(["train", "--data-dir", d, "--epochs", "0"])
        self.assertEqual(2, rc)


if __name__ == "__main__":
    unittest.main()
