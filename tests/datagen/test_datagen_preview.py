"""The previews — the only artefact in this package aimed at a human eye rather than a model.

Worth testing for one reason: they exist because the owner opened the dataset, saw black, and
concluded the depth was broken. The depth was fine. If the preview ever goes black too, the next
person draws the same wrong conclusion and this time nothing contradicts it.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from datagen.preview import preview_dataset, preview_scene
from datagen.render.quality import describe_rgb

_RESOLUTION = (64, 48)


def _scene(
    root: Path, *, with_depth: bool = True, with_masks: bool = True, with_noisy: bool = False,
) -> Path:
    import cv2

    scene = root / "scenes" / "scene_000000"
    scene.mkdir(parents=True)
    ramp = np.linspace(30, 220, _RESOLUTION[0], dtype=np.uint8)
    rgb = np.repeat(np.tile(ramp, (_RESOLUTION[1], 1))[:, :, None], 3, axis=2)
    cv2.imwrite(str(scene / "oblique_left_rgb.png"), rgb)
    if with_depth:
        depth = np.full(_RESOLUTION[::-1], 400, np.uint16)
        depth[:, :20] = 250          # something nearer, so the colouring has a range to work with
        depth[:, -10:] = 0           # invalid
        cv2.imwrite(str(scene / "oblique_left_depth.png"), depth)
        if with_noisy:
            # The sensor model's signature: some pixels shifted, some dropped entirely.
            noisy = depth.copy()
            noisy[:, :20] = 262
            noisy[5:9, 25:35] = 0
            cv2.imwrite(str(scene / "oblique_left_depth_noisy.png"), noisy)
    if with_masks:
        instances = np.zeros(_RESOLUTION[::-1], np.uint16)
        instances[10:30, 10:30] = 1
        instances[10:30, 35:50] = 2
        cv2.imwrite(str(scene / "oblique_left_instances.png"), instances)
    (scene / "scene.json").write_text(json.dumps({
        "views": [{
            "name": "oblique_left", "outcome": "rendered",
            "objects": [
                {"instance_id": 0, "bbox_xyxy": [10, 10, 30, 30], "visibility": 1.0},
                {"instance_id": 1, "bbox_xyxy": [35, 10, 50, 30], "visibility": 0.4},
            ],
        }],
    }), encoding="utf-8")
    return scene


class PreviewTests(unittest.TestCase):
    def _read(self, path: Path) -> np.ndarray:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(image, f"{path} did not decode")
        return np.asarray(image)

    def test_a_preview_is_written_and_is_actually_visible(self) -> None:
        with TemporaryDirectory() as tmp:
            scene = _scene(Path(tmp))
            written, skipped = preview_scene(scene)
            self.assertEqual((written, skipped), (1, []))
            preview = self._read(scene / "oblique_left_preview.png")
            self.assertTrue(describe_rgb(preview).renderable, "the preview itself must not be black")

    def test_it_is_three_panels_wide(self) -> None:
        """rgb | depth | instances, side by side — the whole point is seeing them together."""
        with TemporaryDirectory() as tmp:
            scene = _scene(Path(tmp))
            preview_scene(scene)
            preview = self._read(scene / "oblique_left_preview.png")
            self.assertGreater(preview.shape[1], 3 * _RESOLUTION[0])
            self.assertGreater(preview.shape[0], _RESOLUTION[1], "captions add a strip under each")

    def test_every_unviewable_file_gets_a_viewable_twin(self) -> None:
        """The owner's complaint, as a test: no file in the folder may be a black mystery."""
        with TemporaryDirectory() as tmp:
            scene = _scene(Path(tmp), with_noisy=True)
            preview_scene(scene)
            for stem in ("depth", "depth_noisy", "instances"):
                raw = self._read(scene / f"oblique_left_{stem}.png")
                self.assertLess(int(np.asarray(raw).max()), 1000,
                                f"{stem}: the ORIGINAL must stay uint16 data, not be recoloured")
                twin = scene / f"oblique_left_{stem}_view.png"
                self.assertTrue(twin.exists(), f"{stem} has no viewable twin")
                self.assertTrue(describe_rgb(self._read(twin)).renderable, f"{twin.name} is black")

    def test_the_noisy_depth_is_drawn_on_the_clean_depth_scale(self) -> None:
        """Otherwise dropped pixels renormalise the image and the noise reads as a new scene."""
        with TemporaryDirectory() as tmp:
            scene = _scene(Path(tmp), with_noisy=True)
            preview_scene(scene)
            clean = self._read(scene / "oblique_left_depth_view.png")
            noisy = self._read(scene / "oblique_left_depth_noisy_view.png")
            # The 400 mm background is identical in both files, so it must be the same colour.
            np.testing.assert_array_equal(clean[24, 30], noisy[24, 30])

    def test_depth_becomes_visible_rather_than_staying_dark(self) -> None:
        """The defect that started this: 250-400 mm in a uint16 file is black to a human.

        Asserted as separation, not brightness. A first version of this test demanded a bright peak
        and failed at 122 -- because TURBO's near end is dark red, so "bright" was never the property.
        What the preview owes the reader is that 250 mm and 400 mm do not look the same.
        """
        with TemporaryDirectory() as tmp:
            scene = _scene(Path(tmp))
            preview_scene(scene)
            preview = self._read(scene / "oblique_left_preview.png")
            depth_panel = preview[:_RESOLUTION[1], _RESOLUTION[0] + 6:2 * _RESOLUTION[0] + 6]
            near = depth_panel[24, 5]          # the 250 mm band
            far = depth_panel[24, 30]          # the 400 mm background
            invalid = depth_panel[24, _RESOLUTION[0] - 3]
            self.assertGreater(int(np.abs(near.astype(int) - far.astype(int)).max()), 60,
                               f"near {near} and far {far} must not look alike")
            self.assertGreater(int(depth_panel.max()), 60, "the panel must not be near-black")
            np.testing.assert_array_equal(invalid, [0, 0, 0], "invalid depth stays black")

    def test_two_objects_get_two_different_colours(self) -> None:
        with TemporaryDirectory() as tmp:
            scene = _scene(Path(tmp))
            preview_scene(scene)
            preview = self._read(scene / "oblique_left_preview.png")
            panel = preview[:_RESOLUTION[1], 2 * (_RESOLUTION[0] + 6):]
            first = panel[20, 20 - 0]
            colours = {tuple(int(c) for c in panel[y, x]) for y in (20,) for x in (15, 42)}
            self.assertEqual(len(colours), 2, f"ids must not share a colour (got {colours}, {first})")

    def test_a_missing_depth_or_mask_still_produces_a_preview(self) -> None:
        with TemporaryDirectory() as tmp:
            scene = _scene(Path(tmp), with_depth=False, with_masks=False)
            written, _skipped = preview_scene(scene)
            self.assertEqual(written, 1, "the rgb alone is still worth looking at")

    def test_a_view_with_no_rgb_is_skipped_with_a_reason(self) -> None:
        with TemporaryDirectory() as tmp:
            scene = _scene(Path(tmp))
            (scene / "oblique_left_rgb.png").unlink()
            written, skipped = preview_scene(scene)
            self.assertEqual(written, 0)
            self.assertIn("no rgb", skipped[0])

    def test_a_dataset_rolls_up(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ds"
            _scene(root)
            report = preview_dataset(root)
            self.assertEqual((report.scenes, report.written), (1, 1))
            self.assertIn("OK", report.summary())

    def test_a_dataset_collects_its_previews_in_one_flat_folder(self) -> None:
        """The owner's layout: 900 images should be one scroll, not 300 folders."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ds"
            _scene(root, with_noisy=True)
            preview_dataset(root)
            preview_dir = root / "preview"
            names = sorted(path.name for path in preview_dir.iterdir())
            self.assertEqual(names, [
                "scene_000000_oblique_left_depth_noisy_view.png",
                "scene_000000_oblique_left_depth_view.png",
                "scene_000000_oblique_left_instances_view.png",
                "scene_000000_oblique_left_preview.png",
            ], "every preview carries its scene id and lives in one directory")
            self.assertFalse(
                list((root / "scenes" / "scene_000000").glob("*_view.png")),
                "the scene directory must be left holding only what a consumer reads",
            )
            self.assertTrue((root / "HOW_TO_READ.txt").exists())

    def test_an_empty_root_is_reported_not_crashed(self) -> None:
        with TemporaryDirectory() as tmp:
            report = preview_dataset(Path(tmp))
            self.assertEqual(report.written, 0)
            self.assertTrue(report.skipped)


if __name__ == "__main__":
    unittest.main()
