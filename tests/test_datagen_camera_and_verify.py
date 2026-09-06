"""The camera convention and the dataset verifier — the two things that decide if a label is true.

These are the checks that would have caught the 2026-08-11 pair of defects, so they are written to
fail loudly rather than to pass quietly. Everything here is numpy and JSON: no Isaac, no GPU, so the
gate runs in CI on every commit rather than only on the box with the RTX card.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from datagen.render.camera import intrinsics_matrix, look_at_camera_to_base, project_to_pixels
from datagen.render.quality import describe_rgb, rgb_is_renderable
from datagen.verify import verify_dataset, verify_scene

_RESOLUTION = (640, 480)
_FOV = 60.0


class CameraConventionTests(unittest.TestCase):
    """+Z forward, +X image-right, +Y image-down — asserted, because assuming it cost a cycle."""

    def test_the_target_lands_in_the_middle(self) -> None:
        eye, target = (450.0, -450.0, 600.0), (450.0, 0.0, 0.0)
        uv = project_to_pixels(
            np.asarray([target]), look_at_camera_to_base(eye, target),
            intrinsics_matrix(_RESOLUTION, _FOV),
        )[0]
        self.assertAlmostEqual(uv[0], 320.0, places=6)
        self.assertAlmostEqual(uv[1], 240.0, places=6)

    def test_higher_in_the_world_is_higher_in_the_image(self) -> None:
        """The sign that was wrong. A 180° roll passes every centred test and fails this one."""
        eye, target = (450.0, -450.0, 600.0), (450.0, 0.0, 0.0)
        model = (look_at_camera_to_base(eye, target), intrinsics_matrix(_RESOLUTION, _FOV))
        low = project_to_pixels(np.asarray([[450.0, 0.0, 0.0]]), *model)[0]
        high = project_to_pixels(np.asarray([[450.0, 0.0, 200.0]]), *model)[0]
        self.assertLess(high[1], low[1], "a point 200 mm higher must appear ABOVE, i.e. smaller v")

    def test_further_left_in_the_world_is_further_left_in_the_image(self) -> None:
        """The other half of the roll. Camera on -Y looking at +Y: world +X is image-right."""
        eye, target = (450.0, -450.0, 600.0), (450.0, 0.0, 0.0)
        model = (look_at_camera_to_base(eye, target), intrinsics_matrix(_RESOLUTION, _FOV))
        centre = project_to_pixels(np.asarray([[450.0, 0.0, 0.0]]), *model)[0]
        plus_x = project_to_pixels(np.asarray([[550.0, 0.0, 0.0]]), *model)[0]
        self.assertGreater(plus_x[0], centre[0])

    def test_the_frame_is_right_handed_and_orthonormal(self) -> None:
        matrix = look_at_camera_to_base((100.0, 200.0, 300.0), (0.0, 0.0, 0.0))
        rotation = matrix[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=9)

    def test_a_straight_down_camera_has_a_defined_answer(self) -> None:
        """The overhead view is this generator's normal case, not an edge case."""
        matrix = look_at_camera_to_base((450.0, 0.0, 800.0), (450.0, 0.0, 0.0))
        self.assertTrue(np.all(np.isfinite(matrix)))
        uv = project_to_pixels(
            np.asarray([[450.0, 0.0, 0.0]]), matrix, intrinsics_matrix(_RESOLUTION, _FOV),
        )[0]
        np.testing.assert_allclose(uv, [320.0, 240.0], atol=1e-6)

    def test_a_point_behind_the_camera_is_nan_not_a_mirrored_pixel(self) -> None:
        eye, target = (450.0, -450.0, 600.0), (450.0, 0.0, 0.0)
        uv = project_to_pixels(
            np.asarray([[450.0, -900.0, 600.0]]), look_at_camera_to_base(eye, target),
            intrinsics_matrix(_RESOLUTION, _FOV),
        )[0]
        self.assertTrue(bool(np.all(np.isnan(uv))))

    def test_a_camera_on_its_target_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            look_at_camera_to_base((1.0, 2.0, 3.0), (1.0, 2.0, 3.0))

    def test_a_wider_lens_sees_a_point_closer_to_the_centre(self) -> None:
        eye, target = (450.0, -450.0, 600.0), (450.0, 0.0, 0.0)
        pose = look_at_camera_to_base(eye, target)
        point = np.asarray([[550.0, 0.0, 0.0]])
        narrow = project_to_pixels(point, pose, intrinsics_matrix(_RESOLUTION, 40.0))[0]
        wide = project_to_pixels(point, pose, intrinsics_matrix(_RESOLUTION, 90.0))[0]
        self.assertLess(abs(wide[0] - 320.0), abs(narrow[0] - 320.0))

    def test_an_impossible_field_of_view_is_refused(self) -> None:
        for bad in (0.0, 180.0, -10.0, 200.0):
            with self.assertRaises(ValueError):
                intrinsics_matrix(_RESOLUTION, bad)


def _scene_payload(swap: bool = False, break_visibility: bool = False) -> dict:
    """A two-object scene whose labels are correct — or deliberately not."""
    eye, target = (450.0, -450.0, 600.0), (450.0, 0.0, 0.0)
    pose = look_at_camera_to_base(eye, target)
    k = intrinsics_matrix(_RESOLUTION, _FOV)
    positions = [(380.0, 0.0, 25.0), (520.0, 0.0, 25.0)]
    pixels = project_to_pixels(np.asarray(positions), pose, k)

    boxes = [
        [int(u) - 30, int(v) - 30, int(u) + 30, int(v) + 30] for u, v in pixels
    ]
    if swap:
        boxes = [boxes[1], boxes[0]]
    objects = [
        {
            "asset_id": f"proc_0000{index}_box", "instance_id": index, "bbox_xyxy": boxes[index],
            "visible_px": 3600, "unoccluded_px": 900 if break_visibility and index == 0 else 3600,
            "visibility": 1.0,
            "position_mm": list(positions[index]), "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        for index in (0, 1)
    ]
    return {
        "spec": {"cameras": [{"name": "oblique_left", "resolution": list(_RESOLUTION),
                              "horizontal_fov_deg": _FOV}]},
        "status": "ok", "arm_mode": "absent", "render_mode": "pathtrace",
        "settled_poses_mm_xyzw": {
            str(index): [list(positions[index]), [0.0, 0.0, 0.0, 1.0]] for index in (0, 1)
        },
        "materials": {},
        "views": [{
            # The literal the renderer writes. An earlier version of this fixture said
            # "ViewOutcome.RENDERED", which no written scene has ever contained -- so any check keyed
            # on the outcome silently did nothing here while passing in CI.
            "name": "oblique_left", "outcome": "rendered",
            "camera_position_mm": list(eye), "camera_look_at_mm": list(target),
            "camera_to_base_mm": pose.tolist(), "intrinsics": k.tolist(),
            "objects": objects,
        }],
    }


def _lit_frame() -> np.ndarray:
    """A frame with the properties a real render has: everything lit, many distinct levels."""
    ramp = np.linspace(40, 220, _RESOLUTION[0], dtype=np.uint8)
    return np.repeat(np.tile(ramp, (_RESOLUTION[1], 1))[:, :, None], 3, axis=2)


class VerifierTests(unittest.TestCase):
    def _write(self, payload: dict, root: Path, rgb: str = "lit") -> Path:
        """One scene on disk. ``rgb`` is what the colour frame is: lit, blank, or absent."""
        import cv2  # noqa: PLC0415 - only the file-level tests need it

        scene = root / "scenes" / "scene_000000"
        scene.mkdir(parents=True)
        (scene / "scene.json").write_text(json.dumps(payload), encoding="utf-8")
        if rgb != "absent":
            frame = _lit_frame() if rgb == "lit" else np.zeros((*_RESOLUTION[::-1], 3), np.uint8)
            cv2.imwrite(str(scene / "oblique_left_rgb.png"), frame)
        return scene

    def test_correct_labels_pass(self) -> None:
        with TemporaryDirectory() as tmp:
            scene = self._write(_scene_payload(), Path(tmp))
            report = verify_scene(scene)
            self.assertTrue(report.ok, report.summary())
            self.assertEqual(report.projection_checked, 2)
            self.assertEqual((report.images_checked, report.images_blank), (1, 0))

    def test_an_all_black_frame_is_caught(self) -> None:
        """The 2026-08-12 defect: correct depth, correct masks, correct boxes, no picture.

        Every other check in the verifier passes on this scene -- that is the entire point of it.
        """
        with TemporaryDirectory() as tmp:
            scene = self._write(_scene_payload(), Path(tmp), rgb="blank")
            report = verify_scene(scene)
            self.assertFalse(report.ok, "a scene with no picture must not verify")
            self.assertEqual(report.images_blank, 1)
            self.assertIn("entirely zero", report.summary())
            self.assertEqual(report.projection_failed, 0, "the labels were never the problem")

    def test_a_view_that_claims_an_image_it_never_wrote_is_caught(self) -> None:
        with TemporaryDirectory() as tmp:
            scene = self._write(_scene_payload(), Path(tmp), rgb="absent")
            report = verify_scene(scene)
            self.assertFalse(report.ok)
            self.assertIn("never wrote", report.summary())

    def test_a_dropped_object_may_not_appear_anywhere(self) -> None:
        """Removing a runaway is only honest if it is really gone from every downstream artefact."""
        payload = _scene_payload()
        payload["dropped_objects"] = [1]
        with TemporaryDirectory() as tmp:
            scene = self._write(payload, Path(tmp))
            report = verify_scene(scene)
            self.assertFalse(report.ok, "a dropped object still carrying a label must fail")
            self.assertIn("recorded as dropped", report.summary())

    def test_a_scene_with_a_genuinely_removed_object_passes(self) -> None:
        payload = _scene_payload()
        payload["dropped_objects"] = [1]
        payload["views"][0]["objects"] = [payload["views"][0]["objects"][0]]
        payload["settled_poses_mm_xyzw"].pop("1")
        with TemporaryDirectory() as tmp:
            scene = self._write(payload, Path(tmp))
            report = verify_scene(scene)
            self.assertTrue(report.ok, report.summary())

    def test_a_skipped_view_is_not_asked_for_an_image(self) -> None:
        """A wrist view the arm could not reach has no picture and owes none."""
        payload = _scene_payload()
        payload["views"][0]["outcome"] = "skipped_unreachable"
        with TemporaryDirectory() as tmp:
            scene = self._write(payload, Path(tmp), rgb="absent")
            report = verify_scene(scene)
            self.assertTrue(report.ok, report.summary())
            self.assertEqual(report.images_checked, 0)

    def test_swapped_boxes_are_caught(self) -> None:
        """The defect that shipped twice. Pixel counts stay perfect; only the geometry disagrees."""
        with TemporaryDirectory() as tmp:
            scene = self._write(_scene_payload(swap=True), Path(tmp))
            report = verify_scene(scene)
            self.assertFalse(report.ok)
            self.assertEqual(report.projection_failed, 2)
            self.assertIn("swapped", report.summary())

    def test_visible_beyond_unoccluded_is_caught(self) -> None:
        with TemporaryDirectory() as tmp:
            scene = self._write(_scene_payload(break_visibility=True), Path(tmp))
            report = verify_scene(scene)
            self.assertFalse(report.ok)
            self.assertIn("impossible by construction", report.summary())

    def test_an_empty_dataset_is_not_silently_ok(self) -> None:
        with TemporaryDirectory() as tmp:
            report = verify_dataset(Path(tmp))
            self.assertFalse(report.ok)

    def test_a_dataset_rolls_up_its_scenes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ds"
            self._write(_scene_payload(), root)
            second = root / "scenes" / "scene_000001"
            second.mkdir(parents=True)
            (second / "scene.json").write_text(json.dumps(_scene_payload(swap=True)), encoding="utf-8")
            report = verify_dataset(root)
            self.assertEqual(report.scenes, 2)
            self.assertFalse(report.ok)


class ADepthOnlyDatasetIsVerifiedWithoutItsPictureTests(unittest.TestCase):
    """A depth-only run has no colour image and must still pass — without weakening the gate.

    "There is no colour image" and "the colour image is black" are the SAME thing on disk and different
    facts about the run. Check 4 exists because three all-black frames were once recorded as `ok`, so
    it may not decide which case it is by looking for the file: the answer comes from the dataset's own
    provenance stamp, and an unstamped dataset keeps the strict treatment.
    """

    def _dataset(self, root: Path, *, depth_only: bool | None, rgb: str) -> Path:
        import cv2  # noqa: PLC0415

        scene = root / "scenes" / "scene_000000"
        scene.mkdir(parents=True)
        (scene / "scene.json").write_text(json.dumps(_scene_payload()), encoding="utf-8")
        if rgb != "absent":
            frame = _lit_frame() if rgb == "lit" else np.zeros((*_RESOLUTION[::-1], 3), np.uint8)
            cv2.imwrite(str(scene / "oblique_left_rgb.png"), frame)
        if depth_only is not None:
            (root / "provenance.json").write_text(
                json.dumps({"config": {"render": {"depth_only": depth_only}}}), encoding="utf-8")
        return scene

    def test_a_stamped_depth_only_dataset_with_no_picture_passes(self) -> None:
        with TemporaryDirectory() as tmp:
            self._dataset(Path(tmp), depth_only=True, rgb="absent")
            report = verify_dataset(Path(tmp))
            self.assertTrue(report.ok, report.summary())
            self.assertEqual(report.images_checked, 0)

    def test_the_SAME_dataset_without_the_stamp_is_REFUSED(self) -> None:
        """The whole point: only the stamp may excuse a missing picture."""
        with TemporaryDirectory() as tmp:
            self._dataset(Path(tmp), depth_only=None, rgb="absent")
            report = verify_dataset(Path(tmp))
            self.assertFalse(report.ok)
            self.assertTrue(any("never wrote" in p for p in report.problems), report.problems)

    def test_a_stamp_that_says_colour_still_refuses_a_missing_picture(self) -> None:
        with TemporaryDirectory() as tmp:
            self._dataset(Path(tmp), depth_only=False, rgb="absent")
            self.assertFalse(verify_dataset(Path(tmp)).ok)

    def test_an_unreadable_stamp_keeps_the_check_ON(self) -> None:
        """Fail-closed in the direction that matters: a broken stamp gets the stricter treatment."""
        with TemporaryDirectory() as tmp:
            self._dataset(Path(tmp), depth_only=None, rgb="absent")
            (Path(tmp) / "provenance.json").write_text("{not json", encoding="utf-8")
            self.assertFalse(verify_dataset(Path(tmp)).ok)

    def test_depth_only_does_not_stop_the_OTHER_checks(self) -> None:
        """Skipping the picture must not skip the labels -- a permuted mask set still has to fail."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = root / "scenes" / "scene_000000"
            scene.mkdir(parents=True)
            (scene / "scene.json").write_text(json.dumps(_scene_payload(swap=True)), encoding="utf-8")
            (root / "provenance.json").write_text(
                json.dumps({"config": {"render": {"depth_only": True}}}), encoding="utf-8")
            report = verify_dataset(root)
            self.assertFalse(report.ok, report.summary())
            self.assertGreater(report.projection_failed, 0)


class ImageContentTests(unittest.TestCase):
    """The predicate the renderer and the verifier share. Thresholds are measured; these fix them."""

    def test_a_real_frame_is_renderable(self) -> None:
        self.assertTrue(rgb_is_renderable(_lit_frame()))

    def test_an_all_zero_frame_is_not(self) -> None:
        content = describe_rgb(np.zeros((480, 640, 3), np.uint8))
        self.assertFalse(content.renderable)
        self.assertEqual(content.peak, 0)
        self.assertIn("entirely zero", content.why_not())

    def test_a_uniform_fill_is_not_an_image(self) -> None:
        """Bright but flat: a lens or exposure fault, which the lit-fraction test alone would pass."""
        content = describe_rgb(np.full((480, 640, 3), 200, np.uint8))
        self.assertFalse(content.renderable)
        self.assertIn("distinct level", content.why_not())

    def test_a_mostly_black_frame_is_not(self) -> None:
        frame = np.zeros((480, 640, 3), np.uint8)
        frame[:48] = _lit_frame()[:48]  # 10 % of the rows lit
        self.assertFalse(rgb_is_renderable(frame))

    def test_a_missing_frame_describes_rather_than_crashes(self) -> None:
        for absent in (None, np.zeros((0, 0, 3), np.uint8)):
            self.assertFalse(describe_rgb(absent).renderable)

    def test_a_frame_darker_than_any_real_one_still_passes(self) -> None:
        """Headroom, stated as a test. The gate must catch blank frames, not merely dim ones.

        MEASURED 2026-08-12 over 45 real renders: the darkest had mean 94.5 and 99.81 % of pixels
        above black. This frame is far darker than that -- mean ~30, peak 95 -- and must still be
        accepted, because the failure being caught is "no buffer arrived", not "the scene is dim".
        """
        ramp = np.linspace(9, 95, _RESOLUTION[0], dtype=np.uint8)
        dim = np.repeat(np.tile(ramp, (_RESOLUTION[1], 1))[:, :, None], 3, axis=2)
        content = describe_rgb(dim)
        self.assertTrue(content.renderable, content.why_not())
        self.assertGreater(content.lit_fraction, 0.99)


if __name__ == "__main__":
    unittest.main()
