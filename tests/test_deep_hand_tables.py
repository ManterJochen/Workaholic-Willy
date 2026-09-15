"""A cloud read for another hand: its grasp table overlays the cloud's, and one run trains across hands.

Lane (i) D5. `build-grasp-tables` writes `<scene>.<model>.grasps` beside a cloud. The reader takes a hand
(`load_scene(path, gripper=)`); the index takes several (`corpus_index(files, hands=)`), one entry per cloud and hand,
with every hand's entries of one asset in that asset's fold; and scale jitter is capped by the sample's own hand
rather than by the 2F-85's 85 mm.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from src.robot.grasping.deep.corpus.sample import SampleSpec, build_sample, load_scene
from tests.test_deep_cli import _scene

_HANDE = "robotiq_hande"
_BOTH = ["robotiq_2f85", _HANDE]


def _cloud(directory: Path, index: int = 0, *, source: str = "v5_s0") -> Path:
    path = directory / f"scene_{index:03d}.npz"
    _scene(path, asset=f"gso_asset_{index:03d}")
    with np.load(path, allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    arrays["source_dataset"] = np.asarray([source], dtype="<U64")
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    return path


def _table(cloud: Path, *, gripper: str = _HANDE, source: str = "v5_s0", points: int | None = None,
           width_mm: float = 30.0, aperture_mm: float = 49.99) -> Path:
    """The hand's table beside ``cloud`` as the writer stamps it: the cloud's own grasps at another width."""
    with np.load(cloud, allow_pickle=False) as handle:
        scene = {key: handle[key] for key in handle.files}
    arrays = {key: value for key, value in scene.items() if key.startswith(("grasp_", "contact_"))}
    arrays["grasp_width_mm"] = np.full(len(arrays["grasp_width_mm"]), width_mm, dtype=np.float32)
    arrays.update({
        "table_version": np.asarray([1], dtype=np.int32),
        "gripper": np.asarray([gripper], dtype="<U32"),
        "label_file": np.asarray([f"grasps_jaw_{gripper}.jsonl"], dtype="<U64"),
        "source_dataset": np.asarray([source], dtype="<U64"),
        "cloud_points": np.asarray([len(scene["points_mm"]) if points is None else points], dtype=np.int64),
        "gripper_aperture_mm": np.asarray([aperture_mm], dtype=np.float32),
    })
    path = cloud.with_name(f"{cloud.stem}.{gripper}.grasps")
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
    return path


class _Clouds(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.cloud = _cloud(self.dir)


class TheReaderTakesAHandTests(_Clouds):
    def test_no_hand_reads_the_cloud_as_it_is(self) -> None:
        """The control."""
        with np.load(self.cloud, allow_pickle=False) as handle:
            raw = {key: handle[key] for key in handle.files}
        scene = load_scene(self.cloud)
        self.assertEqual(set(scene), set(raw))
        np.testing.assert_array_equal(scene["grasp_width_mm"], raw["grasp_width_mm"])

    def test_the_clouds_own_hand_reads_the_cloud_as_it_is(self) -> None:
        """A cloud extracted before the stamp is the 2F-85's, by its alias or by its model name."""
        _table(self.cloud)
        for name in ("2f85", "robotiq_2f85"):
            with self.subTest(hand=name):
                scene = load_scene(self.cloud, gripper=name)  # type: ignore[call-arg]
                np.testing.assert_array_equal(scene["grasp_width_mm"], np.full(3, 40.0, dtype=np.float32))
                self.assertNotIn("gripper_aperture_mm", scene)

    def test_another_hand_overlays_its_table(self) -> None:
        _table(self.cloud)
        points = load_scene(self.cloud)["points_mm"]
        scene = load_scene(self.cloud, gripper=_HANDE)  # type: ignore[call-arg]
        np.testing.assert_array_equal(scene["grasp_width_mm"], np.full(3, 30.0, dtype=np.float32))
        np.testing.assert_array_equal(scene["points_mm"], points)
        self.assertEqual(str(scene["gripper"][0]), _HANDE)
        self.assertAlmostEqual(float(scene["gripper_aperture_mm"][0]), 49.99, places=4)

    def test_a_missing_table_refuses_naming_it(self) -> None:
        with self.assertRaises(FileNotFoundError) as caught:
            load_scene(self.cloud, gripper=_HANDE)  # type: ignore[call-arg]
        self.assertIn(f"scene_000.{_HANDE}.grasps", str(caught.exception))

    def test_a_table_for_another_extraction_refuses(self) -> None:
        _table(self.cloud, source="v5_s9")
        with self.assertRaises(ValueError):
            load_scene(self.cloud, gripper=_HANDE)  # type: ignore[call-arg]

    def test_a_table_beside_a_cloud_extracted_again_refuses(self) -> None:
        _table(self.cloud, points=1)
        with self.assertRaises(ValueError):
            load_scene(self.cloud, gripper=_HANDE)  # type: ignore[call-arg]

    def test_the_reader_finds_the_table_where_the_writer_puts_it(self) -> None:
        """`src` may not import `datagen`, so the reader mirrors the writer's name; this holds them equal."""
        from src.robot.grasping.deep.corpus import sample
        from datagen.corpus.tables import TABLE_SUFFIX, table_path

        self.assertEqual(sample.GRASP_TABLE_SUFFIX, TABLE_SUFFIX)  # type: ignore[attr-defined]
        self.assertEqual(sample.grasp_table_path(self.cloud, _HANDE), table_path(self.cloud, _HANDE))  # type: ignore[attr-defined]


class TheJitterIsCappedByTheHandTests(_Clouds):
    def test_the_draw_is_capped_at_the_hands_aperture(self) -> None:
        from src.robot.grasping.deep.corpus.sample import _draw_scale

        factor = _draw_scale(np.random.default_rng(0), SampleSpec(scale_jitter=(2.0, 2.0)),  # type: ignore[call-arg]
                             np.full(4, 40.0), aperture_mm=49.99)
        self.assertAlmostEqual(factor, 49.99 / 40.0, places=6)

    def test_a_sample_read_for_the_hande_never_outgrows_it(self) -> None:
        """30 mm grasps doubled would be 60 mm, inside the 2F-85's 85 and outside the Hand-E's 49.99."""
        from tests.test_scale_jitter import _widths

        _table(self.cloud)
        scene = load_scene(self.cloud, gripper=_HANDE)  # type: ignore[call-arg]
        spec = SampleSpec(scale_jitter=(2.0, 2.0), points=256)
        sample = build_sample(scene, np.random.default_rng(0), spec, target_instance=0)
        self.assertLessEqual(float(np.max(_widths(sample))), 49.99 + 1e-3)


class TheIndexTakesHandsTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.files = [_cloud(self.dir, index) for index in range(4)]

    def test_without_hands_the_index_is_todays(self) -> None:
        """The control."""
        from src.robot.grasping.deep.train.trainer import corpus_index

        index = corpus_index(self.files)
        self.assertEqual(index.files, self.files)
        self.assertEqual(index.grippers, ["2f85"] * 4)

    def test_each_hand_is_an_entry_per_cloud_and_an_asset_stays_in_one_fold(self) -> None:
        from src.robot.grasping.deep.corpus.index import grouped_folds
        from src.robot.grasping.deep.train.trainer import corpus_index

        for cloud in self.files:
            _table(cloud)
        index = corpus_index(self.files, hands=_BOTH)  # type: ignore[call-arg]
        self.assertEqual(len(index.files), 8)
        self.assertEqual(index.grippers, _BOTH * 4)
        self.assertEqual(index.asset_groups, 4)
        for train, test in grouped_folds(index.groups, folds=2, seed=0):
            self.assertFalse({index.groups[i] for i in train} & {index.groups[i] for i in test})

    def test_a_hand_without_its_table_refuses(self) -> None:
        from src.robot.grasping.deep.train.trainer import corpus_index

        for cloud in self.files[:3]:
            _table(cloud)
        with self.assertRaises(FileNotFoundError):
            corpus_index(self.files, hands=_BOTH)  # type: ignore[call-arg]

    def test_a_mixed_run_records_both_hands(self) -> None:
        from src.robot.grasping.deep.train.trainer import train_set_generator
        from tests.test_deep_set_loop import _small_plan

        for cloud in self.files:
            _table(cloud)
        report = train_set_generator(self.files, _small_plan(), device="cpu", probe_units=0,
                                     hands=_BOTH)  # type: ignore[call-arg]
        self.assertEqual(report["grippers"], sorted(_BOTH))
        self.assertEqual(report["clouds"], 4)


class TheRunCarriesItsHandsTests(unittest.TestCase):
    def test_the_context_carries_the_hands(self) -> None:
        from src.robot.grasping.deep.train.api import GeneratorTraining

        with tempfile.TemporaryDirectory() as name:
            files = [_cloud(Path(name), index) for index in range(2)]
            run = GeneratorTraining.from_plan(corpus=files, hands=_BOTH)  # type: ignore[call-arg]
        self.assertEqual(run.context.hands, tuple(_BOTH))  # type: ignore[attr-defined]

    def test_the_command_passes_the_hands(self) -> None:
        from src.robot.grasping.deep import __main__ as cli

        class _Captured(Exception):
            """Not a ValueError, so the command's own refusal handler cannot swallow it."""

        seen: dict[str, Any] = {}

        def _capture(*args: object, **kwargs: Any) -> None:
            seen.update(kwargs)
            raise _Captured

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for index in range(4):
                _cloud(root, index)
            with mock.patch("src.robot.grasping.deep.train.trainer.train_set_generator",
                            side_effect=_capture), self.assertRaises(_Captured):
                cli.main(["train-set", "--clouds", str(root), "--out", str(root / "out"),
                          "--hands", f"robotiq_2f85, {_HANDE}", "--artifact-gripper", "robotiq_2f85"])
        self.assertEqual(seen["hands"], tuple(_BOTH))


if __name__ == "__main__":
    unittest.main()
