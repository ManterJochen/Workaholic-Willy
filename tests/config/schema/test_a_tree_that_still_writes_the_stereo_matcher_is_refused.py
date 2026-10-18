"""``camera/stereomatcher.yaml`` left the tree on 2026-10-17, and a tree that still carries it is refused, saying why.

The owner: the disparity range is derived from a pair's calibration and from where the work stands before it, and every
other setting of the matcher is fixed at values measured on the D415's infrared pair, with no WLS fill and no temporal
smoothing (``src/calibration/stereo/sub_modules/disparity.py``). The block that set them goes with its schema, and a
tree written before is refused at load, naming the file, the line and what replaced it, as every removed key is
(``src/config/schema/_removed.py``), rather than loading as if it said nothing.
"""

from __future__ import annotations

import importlib
import shutil
import tempfile
import unittest
from pathlib import Path

from src.config import ConfigError, load_camera_section, load_config, reload_config
from src.config.schema._removed import removed_key_sentence
from src.config.schema.app import CameraConfig

_ROOT = Path(__file__).resolve().parents[3]

#: The block as the shipped tree wrote it before 2026-10-17, its first lines.
_STALE = (
    "stereomatcher:\n"
    "  minDisparity: 0\n"
    "  numDisparities: 320\n"
    "  blockSize: 7\n"
    "  wls:\n"
    "    enabled: true\n"
)


class TheMatcherBlockIsGoneTests(unittest.TestCase):

    def test_the_shipped_tree_has_neither_the_file_nor_the_block(self) -> None:
        self.assertFalse((_ROOT / "config" / "camera" / "stereomatcher.yaml").exists())
        self.assertFalse((_ROOT / "config" / "all_keys" / "camera" / "stereomatcher.yaml").exists())
        self.assertNotIn("stereomatcher", CameraConfig.model_fields)
        reload_config()
        self.assertFalse(hasattr(load_config().camera, "stereomatcher"))

    def test_no_schema_of_the_matcher_is_left_to_import(self) -> None:
        schema = importlib.import_module("src.config.schema")
        camera = importlib.import_module("src.config.schema.camera")
        for module in (schema, camera):
            for name in ("StereoMatcherConfig", "WlsFilterConfig"):
                with self.subTest(module=module.__name__, name=name):
                    self.assertFalse(hasattr(module, name))

    def test_every_key_of_the_block_answers_with_what_replaced_it(self) -> None:
        for key in ("camera.stereomatcher", "camera.stereomatcher.numDisparities", "camera.stereomatcher.wls.lambda"):
            with self.subTest(key):
                sentence = removed_key_sentence(key)
                self.assertIsNotNone(sentence)
                assert sentence is not None
                self.assertIn("derived from its own stereo calibration", sentence)
                self.assertIn("no WLS fill", sentence)


class AStaleTreeIsRefusedTests(unittest.TestCase):

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name) / "config"
        shutil.copytree(_ROOT / "config", self.root)
        (self.root / "camera" / "stereomatcher.yaml").write_text(_STALE, encoding="utf-8")
        reload_config()

    def tearDown(self) -> None:
        reload_config()
        self._folder.cleanup()

    def test_the_whole_tree_is_refused_naming_the_file_and_the_reason(self) -> None:
        with self.assertRaises(ConfigError) as caught:
            load_config(self.root, profile=None)
        said = str(caught.exception)
        self.assertRegex(said, r"stereomatcher\.yaml:\d")  # the file and the line the block is written on
        self.assertIn("removed on purpose", said)
        self.assertIn("delete the file", said)

    def test_the_camera_section_alone_is_refused_too(self) -> None:
        with self.assertRaisesRegex(ConfigError, "removed on purpose"):
            load_camera_section(self.root, profile=None)

    def test_without_the_file_the_same_tree_loads(self) -> None:
        (self.root / "camera" / "stereomatcher.yaml").unlink()
        self.assertEqual("webcam_main", load_config(self.root, profile=None).camera.cameras.primary_rig_id)


if __name__ == "__main__":
    unittest.main()
