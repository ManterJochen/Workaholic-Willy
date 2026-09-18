"""A camera the arm carries is described once, as data, and the repository's registry is the one authority for it.

Owner decisions: one box per camera model, written down the way the grippers are, first for the RealSense D435i,
D435, D405 and D415, every number from the vendor drawing and its revision; and the repository's registry is the one
authority, as it is for grippers, so a deployment tree may repeat a file and nothing else. The registry refuses what
it cannot resolve instead of guessing, and a tree's own validator opens it.
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from src.config import ConfigError
from src.config.cameras import available_cameras, load_camera, tree_camera_refusal
from src.config.schema.cameras import CameraBodySpec

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "config"
_SHIPPED = ("realsense_d405", "realsense_d415", "realsense_d435", "realsense_d435i")


def _spec(**changes: object) -> dict:
    spec: dict = {"model": "acme_cam", "source": "Acme drawing 7, revision B", "optical_frame": "color",
                  "housing": {"size_mm": [90.0, 25.0, 25.0], "centre_mm": [30.0, 0.0, -8.0]}}
    spec.update(changes)
    return spec


class TheShippedCamerasTests(unittest.TestCase):
    def test_the_four_cameras_the_owner_named_are_shipped(self) -> None:
        self.assertEqual(tuple(available_cameras()), _SHIPPED)

    def test_every_shipped_file_names_a_drawing_and_its_revision(self) -> None:
        for model in _SHIPPED:
            with self.subTest(model=model):
                source = load_camera(model).source
                self.assertIn("337029-017", source)
                self.assertIn("revision 019", source)
                self.assertIn("97c15bdb1abd3ae2e833518ce502de960748cdd6", source)

    def test_the_housings_are_the_datasheets_maximum_extents(self) -> None:
        # Tables 3-51, 3-52 and 3-54 of 337029-017: nominal plus the 0.15 mm tolerance on every axis.
        expected = {"realsense_d405": (42.15, 42.15, 23.15), "realsense_d415": (99.15, 23.15, 20.15),
                    "realsense_d435": (90.15, 25.15, 25.15), "realsense_d435i": (90.15, 25.15, 25.15)}
        for model, size in expected.items():
            with self.subTest(model=model):
                self.assertEqual(load_camera(model).housing.size_mm, size)

    def test_each_housing_is_placed_as_its_file_derives_it(self) -> None:
        # x: tripod centreline to left imager (Tables 4-19, 4-20) plus the colour camera's offset (xacro);
        # z: the front face (Table 4-16 plus the glass inset) minus half the maximum depth.
        expected = {"realsense_d405": (9.0, 0.0, (3.7 + 0.1) - 23.15 / 2),
                    "realsense_d415": (20.0 + 15.0, 0.0, 1.1 - 20.15 / 2),
                    "realsense_d435": (17.5 + 15.0, 0.0, (4.2 + 0.1) - 25.15 / 2),
                    "realsense_d435i": (17.5 + 15.0, 0.0, (4.2 + 0.1) - 25.15 / 2)}
        for model, centre in expected.items():
            with self.subTest(model=model):
                for got, want in zip(load_camera(model).housing.centre_mm, centre):
                    self.assertAlmostEqual(got, want, places=9)

    def test_the_d435i_is_the_d435_housing(self) -> None:
        self.assertEqual(load_camera("realsense_d435i").housing, load_camera("realsense_d435").housing)


class TheSchemaTests(unittest.TestCase):
    def test_a_description_loads(self) -> None:
        spec = CameraBodySpec.model_validate(_spec())
        self.assertEqual((spec.model, spec.optical_frame), ("acme_cam", "color"))

    def test_a_housing_written_in_another_frame_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            CameraBodySpec.model_validate(_spec(optical_frame="depth"))

    def test_a_box_without_extent_or_beyond_a_metre_is_refused(self) -> None:
        for size in ([0.0, 25.0, 25.0], [90.0, -1.0, 25.0], [1000.5, 25.0, 25.0]):
            with self.subTest(size=size), self.assertRaises(ValidationError) as caught:
                CameraBodySpec.model_validate(_spec(housing={"size_mm": size, "centre_mm": [0.0, 0.0, 0.0]}))
            self.assertIn("each above 0 and at most 1000 mm", str(caught.exception))
        with self.assertRaises(ValidationError):
            CameraBodySpec.model_validate(_spec(housing={"size_mm": [9.0, 9.0, 9.0], "centre_mm": [0.0, 1200.0, 0.0]}))

    def test_a_number_that_is_not_finite_is_refused(self) -> None:
        for bad in (float("inf"), float("nan")):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                CameraBodySpec.model_validate(_spec(housing={"size_mm": [9.0, 9.0, 9.0], "centre_mm": [bad, 0.0, 0.0]}))

    def test_a_description_without_a_source_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            CameraBodySpec.model_validate(_spec(source=""))


class TheLoaderRefusesTests(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name) / "data"
        shutil.copytree(_DATA / "cameras", self.root / "cameras")
        self.shipped = (self.root / "cameras" / "realsense_d435.yaml").read_text(encoding="utf-8")

    def test_a_camera_loads_by_its_file_name(self) -> None:
        self.assertEqual(load_camera("realsense_d435", data_dir=self.root), load_camera("realsense_d435"))

    def test_a_file_named_after_another_camera_is_refused(self) -> None:
        (self.root / "cameras" / "other_cam.yaml").write_text(self.shipped, encoding="utf-8")
        with self.assertRaises(ConfigError) as caught:
            load_camera("realsense_d435", data_dir=self.root)
        self.assertIn("describes the camera 'realsense_d435' but is named 'other_cam'", str(caught.exception))

    def test_an_unknown_name_lists_the_known_cameras(self) -> None:
        with self.assertRaises(ConfigError) as caught:
            load_camera("realsense_d999", data_dir=self.root)
        self.assertIn("Known cameras: " + ", ".join(_SHIPPED), str(caught.exception))

    def test_a_name_that_is_not_registry_shaped_is_refused_before_a_file_is_read(self) -> None:
        with self.assertRaises(ConfigError) as caught:
            load_camera("../robot/robot", data_dir=self.root)
        self.assertIn("is not a registry name", str(caught.exception))

    def test_one_broken_file_refuses_every_lookup(self) -> None:
        (self.root / "cameras" / "realsense_d405.yaml").write_text("camera: [never closed\n", encoding="utf-8")
        with self.assertRaises(ConfigError) as caught:
            load_camera("realsense_d435", data_dir=self.root)
        self.assertIn("realsense_d405.yaml", str(caught.exception))

    def test_a_file_without_the_top_level_key_is_refused(self) -> None:
        (self.root / "cameras" / "realsense_d405.yaml").write_text("gripper: {}\n", encoding="utf-8")
        with self.assertRaises(ConfigError) as caught:
            available_cameras(self.root)
        self.assertIn("must contain a top-level 'camera:' key", str(caught.exception))

    def test_a_file_the_registry_cannot_read_as_a_camera_is_refused_by_name(self) -> None:
        for name in ("acme.yml", "ACME.yaml", "acme.YAML", "realsense_d435.sim.yaml"):
            with self.subTest(name=name):
                path = self.root / "cameras" / name
                path.write_text(self.shipped, encoding="utf-8")
                try:
                    with self.assertRaises(ConfigError) as caught:
                        available_cameras(self.root)
                    self.assertIn("is not a camera the registry can read: a camera is one file named <model>.yaml",
                                  str(caught.exception))
                finally:
                    path.unlink()

    def test_a_file_that_is_not_yaml_is_not_a_camera(self) -> None:
        (self.root / "cameras" / "README.md").write_text("drawings\n", encoding="utf-8")
        self.assertEqual(tuple(available_cameras(self.root)), _SHIPPED)

    def test_a_tree_without_a_registry_is_refused(self) -> None:
        with self.assertRaises(ConfigError) as caught:
            available_cameras(Path(self.root).parent / "nowhere")
        self.assertIn("no camera registry at", str(caught.exception))


class TheRepositoryIsTheAuthorityTests(unittest.TestCase):
    """A tree may repeat the repository's file for a camera its rigs name, and nothing else (owner decision)."""

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name) / "data"
        shutil.copytree(_DATA / "cameras", self.root / "cameras")

    def test_no_tree_and_the_repository_tree_are_the_authority(self) -> None:
        self.assertIsNone(tree_camera_refusal("realsense_d435", data_dir=None))
        self.assertIsNone(tree_camera_refusal("realsense_d435", data_dir=_DATA))

    def test_an_identical_copy_stands_for_the_repository(self) -> None:
        self.assertIsNone(tree_camera_refusal("realsense_d435", data_dir=self.root))

    def test_a_tree_without_a_camera_registry_reads_the_repository(self) -> None:
        shutil.rmtree(self.root / "cameras")
        self.assertIsNone(tree_camera_refusal("realsense_d435", data_dir=self.root))

    def test_a_copy_that_differs_is_refused_naming_both_files_and_the_key(self) -> None:
        path = self.root / "cameras" / "realsense_d435.yaml"
        path.write_text(path.read_text(encoding="utf-8").replace("centre_mm: [32.5,", "centre_mm: [30.0,"),
                        encoding="utf-8")
        refusal = tree_camera_refusal("realsense_d435", data_dir=self.root)
        assert refusal is not None
        self.assertIn(f"{path.resolve()} describes the camera realsense_d435 differently from the repository registry "
                      "config/cameras/realsense_d435.yaml", refusal)
        self.assertIn("housing.centre_mm [30.0, 0.0, -8.275] there and [32.5, 0.0, -8.275] in the repository", refusal)
        self.assertIn("copy config/cameras/realsense_d435.yaml over the tree's file, or add the camera to "
                      "the repository registry", refusal)

    def test_a_camera_only_the_tree_describes_is_refused(self) -> None:
        text = (self.root / "cameras" / "realsense_d435.yaml").read_text(encoding="utf-8")
        (self.root / "cameras" / "acme_cam.yaml").write_text(text.replace("model: realsense_d435", "model: acme_cam"),
                                                             encoding="utf-8")
        refusal = tree_camera_refusal("acme_cam", data_dir=self.root)
        assert refusal is not None
        self.assertIn("describes the camera acme_cam, and the repository registry config/cameras does "
                      "not", refusal)
        self.assertIn("describe the camera in config/cameras/acme_cam.yaml", refusal)

    def test_a_broken_tree_registry_answers_with_its_own_refusal(self) -> None:
        (self.root / "cameras" / "realsense_d405.yaml").write_text("camera: [never closed\n", encoding="utf-8")
        refusal = tree_camera_refusal("realsense_d435", data_dir=self.root)
        assert refusal is not None
        self.assertIn("realsense_d405.yaml", refusal)


class TheValidatorReadsTheCameraRegistryTests(unittest.TestCase):
    """``python -m src.config`` opens the camera registry and compares the cameras the rigs name."""

    def setUp(self) -> None:
        self._environment = mock.patch.dict(os.environ)
        self._environment.start()
        os.environ.pop("WILLY_PROFILE", None)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name) / "data"
        shutil.copytree(_DATA, self.root)

    def tearDown(self) -> None:
        self._environment.stop()

    def _validate(self, *extra: str) -> tuple[int, str]:
        from src.config.__main__ import main

        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["--data", str(self.root), *extra])
        return code, out.getvalue() + err.getvalue()

    def _declare_a_body(self, model: str) -> tuple[int, str]:
        """A camera layer whose shipped RGB-D rig declares a body naming ``model``."""
        from src.config import load_config

        rigs = [rig.model_dump(mode="json") for rig in load_config(self.root).camera.cameras.rigs]
        rgbd = next(rig for rig in rigs if rig["source"] == "rgbd")
        rgbd["body"] = {"model": model, "margin_mm": 5.0, "bracket": None}
        import yaml

        (self.root / "camera" / "cam.bodycheck.yaml").write_text(
            yaml.safe_dump({"cameras": {"rigs": rigs}}, sort_keys=False), encoding="utf-8")
        return self._validate("--profile", "bodycheck")

    def test_the_shipped_tree_validates(self) -> None:
        code, printed = self._validate()
        self.assertEqual(code, 0, printed)

    def test_a_rig_naming_a_shipped_camera_validates(self) -> None:
        code, printed = self._declare_a_body("realsense_d435")
        self.assertEqual(code, 0, printed)

    def test_a_rig_naming_a_camera_no_file_describes_fails_validation(self) -> None:
        code, printed = self._declare_a_body("realsense_d999")
        self.assertEqual(code, 1, printed)
        self.assertIn("no camera 'realsense_d999'", printed)

    def test_a_rig_naming_a_camera_the_tree_describes_differently_fails_validation(self) -> None:
        path = self.root / "cameras" / "realsense_d435.yaml"
        path.write_text(path.read_text(encoding="utf-8").replace("[90.15, 25.15, 25.15]", "[90.0, 25.0, 25.0]"),
                        encoding="utf-8")
        code, printed = self._declare_a_body("realsense_d435")
        self.assertEqual(code, 1, printed)
        self.assertIn("describes the camera realsense_d435 differently from the repository registry", printed)

    def test_a_broken_camera_file_fails_validation_even_when_no_rig_names_it(self) -> None:
        (self.root / "cameras" / "realsense_d405.yaml").write_text("camera: [never closed\n", encoding="utf-8")
        code, printed = self._validate()
        self.assertEqual(code, 1, printed)
        self.assertIn("realsense_d405.yaml", printed)

    def test_a_tree_whose_rigs_declare_no_body_validates_without_a_camera_registry(self) -> None:
        shutil.rmtree(self.root / "cameras")
        code, printed = self._validate()
        self.assertEqual(code, 0, printed)


if __name__ == "__main__":
    unittest.main()
