"""Another hand's grasp table, written beside the clouds a corpus already has.

Lane (i) D5, owner Q-E: one sidecar per scene per hand. A cloud keeps the table its extraction joined, the 2F-85's
for every corpus this repository has built; a table for another hand is its own file, `<scene>.<model>.grasps`, so
labelling the same scenes for a new hand costs a table rather than a re-extraction, and no scene walk reads a table
as a scene. Contacts, the approach corridor and the physics join are the hand's own, through `scene_grasp_table`.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from datagen.grasps.shapes import Solid

_HANDE = "robotiq_hande"
_LABELS = f"grasps_jaw_{_HANDE}.jsonl"
_SCENE = "bin_000001"


def _box() -> Solid:
    return Solid("box", np.array([15.0, 15.0, 20.0]), np.eye(3), np.array([0.0, 0.0, 20.0]),
                 instance_id=0, asset_id="a")


def _label(z: float = 20.0, approach: tuple[float, float, float] = (0.0, 0.0, -1.0)) -> dict[str, Any]:
    return {"scene_id": _SCENE, "instance_id": 0, "kind": "jaw", "position_mm": [0.0, 0.0, z],
            "approach": list(approach), "closing_axis": [1.0, 0.0, 0.0], "width_mm": 30.0}


class _Dataset(unittest.TestCase):
    """A one-scene dataset with the Hand-E's labels and one extracted cloud; the mesh bank is a box."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.root = self.tmp / "v5_s0"
        (self.root / "scenes" / _SCENE).mkdir(parents=True)
        (self.root / "scenes" / _SCENE / "scene.json").write_text(
            json.dumps({"spec": {"objects": [{"asset_id": "a"}]}}), encoding="utf-8")
        self.clouds = self.tmp / "clouds"
        self.cloud = self._cloud("v5_s0")
        # The second label approaches from below at 115 mm: inside the 2F-85's 113.37 mm corridor, outside the
        # Hand-E's 116.53 mm.
        self._labels([_label(), _label(115.0, (0.0, 0.0, 1.0))])
        self.geometry = SimpleNamespace(objects={0: _box()})
        assets = SimpleNamespace(geometry=lambda payload, scene_id: self.geometry)
        patcher = mock.patch("datagen.grasps.labels.scene_assets", return_value=assets)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _cloud(self, identity: str, name: str = _SCENE, directory: Path | None = None) -> Path:
        path = (directory or self.clouds) / f"{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, points_mm=np.zeros((7, 3), dtype=np.float32),
                            source_dataset=np.asarray([identity], dtype="<U64"))
        return path

    def _labels(self, rows: list[dict[str, Any]], name: str = _LABELS) -> None:
        (self.root / name).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    def _build(self, **kwargs: Any) -> dict[str, Any]:
        from datagen.corpus.tables import build_grasp_tables  # type: ignore[import-not-found]

        return build_grasp_tables(self.root, self.clouds, **{"labels": _LABELS, **kwargs})

    def _table(self, cloud: Path | None = None) -> dict[str, np.ndarray]:
        path = (cloud or self.cloud).with_name(f"{(cloud or self.cloud).stem}.{_HANDE}.grasps")
        with np.load(path, allow_pickle=False) as handle:
            return {key: handle[key] for key in handle.files}


class ATableIsWrittenBesideTheCloudTests(_Dataset):
    def test_the_table_lands_beside_the_cloud_under_its_own_suffix(self) -> None:
        from src.robot.grasping.deep.corpus.discovery import scene_files

        self._build()
        self.assertTrue((self.clouds / f"{_SCENE}.{_HANDE}.grasps").is_file())
        self.assertFalse((self.clouds / f"{_SCENE}.{_HANDE}.grasps.npz").exists(), "numpy appended .npz")
        self.assertEqual(scene_files(self.clouds), [self.cloud], "a table was read as a scene")

    def test_the_table_says_whose_it_is_and_which_cloud_it_belongs_to(self) -> None:
        self._build()
        table = self._table()
        self.assertEqual(int(table["table_version"][0]), 1)
        self.assertEqual(str(table["gripper"][0]), _HANDE)
        self.assertEqual(str(table["label_file"][0]), _LABELS)
        self.assertEqual(str(table["source_dataset"][0]), "v5_s0")
        self.assertEqual(int(table["cloud_points"][0]), 7)
        self.assertAlmostEqual(float(table["gripper_aperture_mm"][0]), 49.99, places=4)

    def test_the_table_is_the_hands_table(self) -> None:
        from datagen.corpus.clouds import scene_grasp_table
        from datagen.grasps.labels import jaw_model_for

        self._build()
        rows = [{**_label(), "_row_index": 0}, {**_label(115.0, (0.0, 0.0, 1.0)), "_row_index": 1}]
        expected = scene_grasp_table(rows, self.geometry, {}, {0: "a"}, origin=f"grasps_jaw_{_HANDE}",
                                     model=jaw_model_for(_HANDE))
        table = self._table()
        for key, value in expected.items():
            if key.startswith(("grasp_", "contact_")):
                with self.subTest(key=key):
                    np.testing.assert_array_equal(table[key], value)
        self.assertEqual(table["grasp_approach_admissible"].tolist(), [True, False])
        self.assertGreater(len(table["contact_points_mm"]), 0, "the control: the pads touch the box")

    def test_the_hands_own_verdicts_join_and_the_2f85s_do_not(self) -> None:
        physics = self.tmp / "grasp_physics_jaw.jsonl"
        physics.write_text("\n".join(json.dumps(v) for v in (
            {"origin": f"grasps_jaw_{_HANDE}", "row_index": 0, "held": True, "note": ""},
            {"origin": "grasps", "row_index": 1, "held": True, "note": ""})), encoding="utf-8")
        self._build(physics=physics)
        self.assertEqual(self._table()["grasp_held"].tolist(), [1, -1])

    def test_a_cloud_of_a_split_view_gets_its_table(self) -> None:
        """`split-dataset` extracts through `<dataset>_p<n>` views, and their clouds say so."""
        self.cloud.unlink()
        cloud = self._cloud("v5_s0_p3", directory=self.clouds / "s0_p3")
        self._build()
        self.assertEqual(str(self._table(cloud)["source_dataset"][0]), "v5_s0_p3")


class TheWriterRefusesTests(_Dataset):
    def test_an_unknown_jaw_refuses(self) -> None:
        self._labels([_label()], name="grasps_jaw_no_such_hand.jsonl")
        with self.assertRaises(ValueError) as caught:
            self._build(labels="grasps_jaw_no_such_hand.jsonl")
        self.assertIn("unknown jaw", str(caught.exception))

    def test_a_report_stamping_another_jaw_refuses(self) -> None:
        (self.root / f"grasp_label_report_jaw_{_HANDE}.json").write_text(
            json.dumps({"jaw_model": "wide_140"}), encoding="utf-8")
        with self.assertRaises(ValueError):
            self._build()

    def test_the_corpus_label_file_needs_no_table(self) -> None:
        with self.assertRaises(ValueError):
            self._build(labels="grasps.jsonl")

    def test_clouds_of_another_dataset_get_no_table_and_none_of_this_one_refuses(self) -> None:
        self.cloud.unlink()
        foreign = self._cloud("v5_s9")
        with self.assertRaises(FileNotFoundError):
            self._build()
        self.assertFalse(foreign.with_name(f"{foreign.stem}.{_HANDE}.grasps").exists())

    def test_a_cloud_whose_scene_is_missing_refuses(self) -> None:
        self._cloud("v5_s0", name="bin_000404")
        with self.assertRaises(FileNotFoundError) as caught:
            self._build()
        self.assertIn("bin_000404", str(caught.exception))


class TheCommandTests(unittest.TestCase):
    def test_the_command_is_registered_and_routed(self) -> None:
        from datagen import __main__ as cli

        with mock.patch.object(cli, "_cmd_build_grasp_tables", return_value=0) as handler:
            self.assertEqual(cli.main(["build-grasp-tables", "--name", "v5_s0", "--clouds", "x",
                                       "--labels", _LABELS]), 0)
        handler.assert_called_once()

    def test_the_command_needs_its_clouds(self) -> None:
        from datagen import __main__ as cli

        args = argparse.Namespace(command="build-grasp-tables", name="v5_s0", out=None, clouds=None,
                                  labels=_LABELS, physics=None)
        config = SimpleNamespace(output=SimpleNamespace(root="."))
        self.assertEqual(cli._cmd_build_grasp_tables(args, config), cli._EXIT_USAGE)  # noqa: SLF001


class TheCorpusVersionTests(unittest.TestCase):
    def test_the_version_is_6_and_the_reader_floor_stays_3(self) -> None:
        """The tables are new files beside the clouds; no array a sample is built from changed, so the floor stays."""
        from src.robot.grasping.deep.corpus.sample import _MINIMUM_CORPUS_VERSION
        from datagen.corpus.clouds import CORPUS_VERSION

        self.assertEqual(CORPUS_VERSION, 6)
        self.assertEqual(_MINIMUM_CORPUS_VERSION, 3)

    def test_v3_to_v5_clouds_still_read_without_a_hand(self) -> None:
        """The control: every cloud on disk keeps reading as it did."""
        from src.robot.grasping.deep.corpus.sample import load_scene
        from tests.test_deep_cli import _scene

        for version in (3, 4, 5):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as name:
                path = Path(name) / "scene_000.npz"
                _scene(path)
                with np.load(path, allow_pickle=False) as handle:
                    arrays = {key: handle[key] for key in handle.files}
                arrays["corpus_version"] = np.asarray([version], dtype=np.int32)
                np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
                self.assertEqual(int(load_scene(path)["corpus_version"][0]), version)


if __name__ == "__main__":
    unittest.main()
