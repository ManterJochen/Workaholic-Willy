"""The wrist camera inspection saves where the repository keeps debug images and says what it wrote (P0h h8).

``willy_sim/inspect_wrist_cam.py`` wrote its two PNGs to ``D:/dev/.cache``, a folder of the owner's box, and printed
"saved" without reading ``cv2.imwrite``'s answer, which is ``False`` rather than an exception when the folder is not
there. So every run off that box said "saved" and saved nothing. The images now go to ``logs/debug/eih_wrist``
(``utility.paths.debug_dir``, which ``WILLY_DEBUG_DIR`` moves), and each line says whether its file was written.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

_MODULE = Path(__file__).resolve().parents[1] / "src" / "willy_sim" / "inspect_wrist_cam.py"


class WhatTheInspectionSavesTests(unittest.TestCase):
    def _views(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        depth = np.linspace(300.0, 600.0, 24).reshape(4, 6)
        return rgb, depth, depth[np.isfinite(depth)]

    def test_each_image_is_written_and_named(self) -> None:
        import cv2

        from src.willy_sim.inspect_wrist_cam import save_views

        rgb, depth, finite = self._views()
        with tempfile.TemporaryDirectory() as folder:
            lines = save_views(rgb, depth, finite, where=Path(folder), imwrite=cv2.imwrite)
            written = sorted(p.name for p in Path(folder).iterdir())
        self.assertEqual(written, ["_eih_wrist_depth.png", "_eih_wrist_rgb.png"])
        self.assertEqual(len(lines), 2)
        for line in lines:
            with self.subTest(line):
                self.assertTrue(line.startswith("  saved "), line)

    def test_a_write_that_did_not_happen_is_not_called_saved(self) -> None:
        from src.willy_sim.inspect_wrist_cam import save_views

        rgb, depth, finite = self._views()
        lines = save_views(rgb, depth, finite, where=Path("nowhere"), imwrite=lambda path, image: False)
        self.assertEqual(len(lines), 2)
        for line in lines:
            with self.subTest(line):
                self.assertIn("NOT saved", line)
                self.assertNotIn("  saved ", line)

    def test_the_runner_names_no_folder_of_one_box(self) -> None:
        source = _MODULE.read_text(encoding="utf-8")
        for code_line in (line for line in source.splitlines() if not line.lstrip().startswith(("#", "D:\\\\isaacsim"))):
            with self.subTest(code_line):
                self.assertNotIn("D:/dev", code_line)
        self.assertIn('debug_dir("eih_wrist")', source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
