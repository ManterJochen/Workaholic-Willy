"""A physics verdict joins the label it was measured on, per label file, and every jaw keeps its own files.

Lane (i) D3. Each label file numbers its rows from zero and the shake writes `(origin, row_index)`, so a key that
names no file joins a `grasps.jsonl` verdict onto whichever row of `grasps_jaw_<model>.jsonl` has the same index.
`scene_grasp_table` keyed every row `("grasps", row)`, so every per-jaw extraction carried the 2F-85's verdicts on
another jaw's rows, and its contacts were the 2F-85's pads whatever jaw the labels were for. The per-jaw corpora
extracted that way are discarded, not rebuilt (owner, Q-H).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from datagen.corpus import clouds
from datagen.grasps.verdict import PROCEDURAL_JAWS, JawGrasp, jaw_contact_patch

_JAW = "narrow_55"
_LABELS = f"grasps_jaw_{_JAW}.jsonl"


def _label(index: int = 0) -> dict[str, Any]:
    return {"scene_id": "bin_000001", "instance_id": 0, "kind": "jaw", "position_mm": [float(index), 0.0, 50.0],
            "approach": [0.0, 0.0, -1.0], "closing_axis": [1.0, 0.0, 0.0], "width_mm": 30.0}


def _row(index: int = 0) -> dict[str, Any]:
    """A label as `build_cloud_corpus` hands it to the table: with its line number."""
    return {**_label(index), "_row_index": index}


class _Slab:
    """A target every line through the grasp crosses, 10 mm either side of the line's origin."""

    def line_span(self, origin: Any, direction: Any, *, margin_mm: float = 0.0) -> tuple[float, float]:
        return (-10.0, 10.0)


class TheJoinKeyNamesTheLabelFileTests(unittest.TestCase):
    def test_a_2f85_verdict_never_lands_on_another_jaws_row(self) -> None:
        table = clouds.scene_grasp_table([_row(0)], held={("grasps", 0): True}, origin=f"grasps_jaw_{_JAW}")
        self.assertEqual(table["grasp_held"].tolist(), [-1])

    def test_a_verdict_from_its_own_label_file_joins(self) -> None:
        table = clouds.scene_grasp_table([_row(0)], held={(f"grasps_jaw_{_JAW}", 0): True},
                                         origin=f"grasps_jaw_{_JAW}")
        self.assertEqual(table["grasp_held"].tolist(), [1])

    def test_the_default_origin_is_still_the_corpus_file(self) -> None:
        """The control: every v5 extraction reads `grasps.jsonl` and keeps its key."""
        self.assertEqual(clouds.scene_grasp_table([_row(0)], held={("grasps", 0): False})["grasp_held"].tolist(), [0])


class TheContactsComeFromTheirOwnJawTests(unittest.TestCase):
    def test_the_table_takes_the_contacts_of_the_jaw_it_is_given(self) -> None:
        narrow = PROCEDURAL_JAWS[_JAW]
        grasp = JawGrasp(np.array([0.0, 0.0, 50.0]), np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0]), 30.0)
        own = jaw_contact_patch(grasp, _Slab(), model=narrow)  # type: ignore[arg-type]
        self.assertFalse(np.allclose(own, jaw_contact_patch(grasp, _Slab())),  # type: ignore[arg-type]
                         "the control: the two jaws' pads have to touch different points")
        table = clouds.scene_grasp_table([_row(0)], SimpleNamespace(objects={0: _Slab()}), model=narrow)
        np.testing.assert_allclose(table["contact_points_mm"], own.astype(np.float32))


class TheExtractionPassesItsLabelFileTests(unittest.TestCase):
    """`build_cloud_corpus` over a one-scene dataset, with the rendering and the assets stubbed out."""

    def _extract(self, tmp: Path, verdicts: list[dict[str, Any]], *, labels: str = _LABELS) -> tuple[list[int], Any]:
        root = tmp / "v5_s0"
        scene = root / "scenes" / "bin_000001"
        scene.mkdir(parents=True)
        payload = {"spec": {"objects": [{"asset_id": "a"}], "family": "bin"}, "views": []}
        (scene / "scene.json").write_text(json.dumps(payload), encoding="utf-8")
        (root / labels).write_text(json.dumps(_label(0)) + "\n", encoding="utf-8")
        physics = tmp / "grasp_physics.jsonl"
        physics.write_text("\n".join(json.dumps(v) for v in verdicts), encoding="utf-8")
        cloud = {"points_mm": np.zeros((1, 3), dtype=np.float32), "instance_id": np.zeros(1, dtype=np.int16)}
        assets = SimpleNamespace(geometry=lambda payload, name: SimpleNamespace(objects={}))
        out = tmp / "clouds"
        with mock.patch("datagen.grasps.labels.scene_assets", return_value=assets), \
             mock.patch.object(clouds, "scene_cloud", return_value=cloud), \
             mock.patch.object(clouds, "_engine_of", return_value="mujoco"), \
             mock.patch.object(clouds, "scene_grasp_table", wraps=clouds.scene_grasp_table) as table:
            clouds.build_cloud_corpus(root, out, physics=physics, labels=labels)
        with np.load(out / "bin_000001.npz", allow_pickle=False) as written:
            held = [int(v) for v in written["grasp_held"]]
        return held, table.call_args

    def test_a_2f85_verdict_does_not_reach_a_per_jaw_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            held, _ = self._extract(Path(name), [{"origin": "grasps", "row_index": 0, "held": True, "note": ""}])
        self.assertEqual(held, [-1])

    def test_the_jaws_own_verdict_reaches_it(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            held, _ = self._extract(Path(name), [{"origin": f"grasps_jaw_{_JAW}", "row_index": 0, "held": True,
                                                  "note": ""}])
        self.assertEqual(held, [1])

    def test_the_extraction_hands_the_table_its_jaw(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            _, call = self._extract(Path(name), [])
        self.assertIs(call.kwargs["model"], PROCEDURAL_JAWS[_JAW])
        self.assertEqual(call.kwargs["origin"], f"grasps_jaw_{_JAW}")

    def test_the_corpus_file_keeps_its_key_and_the_2f85(self) -> None:
        """The control: a `grasps.jsonl` extraction joins its own verdicts and takes the default jaw."""
        with tempfile.TemporaryDirectory() as name:
            held, call = self._extract(Path(name), [{"origin": "grasps", "row_index": 0, "held": True, "note": ""}],
                                       labels="grasps.jsonl")
        self.assertEqual(held, [1])
        self.assertIsNone(call.kwargs.get("model"))


class TheShakeDrawsFromTheNamedFileTests(unittest.TestCase):
    @staticmethod
    def _dataset(root: Path, labels: str) -> None:
        rows = [{**_label(i), "scene_id": f"bin_{i:03d}", "instance_id": i} for i in range(12)]
        (root / labels).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    def test_a_per_jaw_draw_reads_that_file_and_names_it_as_the_origin(self) -> None:
        from datagen.grasps.physics import sample_trials

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self._dataset(root, _LABELS)
            trials = sample_trials(root, per_class=100, labels=_LABELS)
            lines = (root / _LABELS).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(trials), 12)
        self.assertEqual({t.origin for t in trials}, {f"grasps_jaw_{_JAW}"})
        for trial in trials:
            self.assertEqual(json.loads(lines[trial.row_index])["scene_id"], trial.scene_id)

    def test_the_default_draw_still_reads_the_corpus_file(self) -> None:
        from datagen.grasps.physics import sample_trials

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self._dataset(root, "grasps.jsonl")
            trials = sample_trials(root, per_class=100)
        self.assertEqual({t.origin for t in trials}, {"grasps"})


class EachJawShakesIntoItsOwnFileTests(unittest.TestCase):
    def test_a_per_jaw_label_run_writes_its_own_file(self) -> None:
        from datagen.grasps.service import physics_output_name

        self.assertEqual(physics_output_name(jaw=_JAW), f"grasp_physics_jaw_{_JAW}.jsonl")
        self.assertNotEqual(physics_output_name(jaw=_JAW), physics_output_name())
        self.assertNotEqual(physics_output_name(jaw=_JAW), physics_output_name(jaw="wide_140"))

    def test_a_jaw_on_a_run_that_is_not_a_label_run_refuses(self) -> None:
        from datagen.grasps.service import physics_output_name

        with self.assertRaises(ValueError):
            physics_output_name(proposals=True, jaw=_JAW)
        with self.assertRaises(ValueError):
            physics_output_name(paired_arms=["sfe", "deep"], jaw=_JAW)


class SplitViewsCarryPerJawFilesTests(unittest.TestCase):
    def test_per_jaw_labels_and_reports_reach_every_part(self) -> None:
        """A part's scenes carry row indices into the whole label file, so every jaw's file goes with them."""
        from datagen.corpus.split import split_dataset
        from tests.test_datagen_split_dataset import _dataset

        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset(Path(tmp), "v_s0", 4)
            (dataset / _LABELS).write_text("{}\n", encoding="utf-8")
            (dataset / f"grasp_label_report_jaw_{_JAW}.json").write_text("{}", encoding="utf-8")
            for view in split_dataset(dataset, 2)["views"]:
                self.assertTrue((Path(view["path"]) / _LABELS).is_file())
                self.assertTrue((Path(view["path"]) / f"grasp_label_report_jaw_{_JAW}.json").is_file())
                self.assertTrue((Path(view["path"]) / "grasps.jsonl").is_file(), "the control")


if __name__ == "__main__":
    unittest.main()
