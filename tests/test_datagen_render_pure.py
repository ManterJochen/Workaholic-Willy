"""The renderer's numerics, tested without a GPU — because these modules decide what the labels MEAN.

A labelling bug that can only be caught on the Isaac box is a labelling bug that ships: it survives
every laptop test, appears in 20 000 scenes, and is discovered by a model that trained on it. So
everything except the actual Isaac call is numpy, and everything numpy is asserted here.
"""

from __future__ import annotations

import unittest

import numpy as np

from datagen.render.depth_noise import REALSENSE_D435, inject_depth_noise
from datagen.render.labels import (
    ObjectLabel,
    bbox_from_mask,
    build_object_labels,
    instance_map_from_masks,
    mask_for_instance,
    pixel_id,
    visibility,
)
from datagen.render.materials import sample_material
from datagen.render.views import (
    UnreachablePolicy,
    ViewOutcome,
    resolve_wrist_viewpoint,
    unreachable_policy,
)
from datagen.render.writer import decode_depth_png, encode_depth_png


class DepthNoiseTests(unittest.TestCase):
    def _flat(self, value_mm: float = 800.0, shape: tuple[int, int] = (64, 64)) -> np.ndarray:
        return np.full(shape, value_mm, dtype=np.float64)

    def test_the_input_is_never_modified(self) -> None:
        depth = self._flat()
        before = depth.copy()
        inject_depth_noise(depth, REALSENSE_D435, np.random.default_rng(0))
        np.testing.assert_array_equal(depth, before)

    def test_noise_grows_with_the_square_of_distance(self) -> None:
        """The defining property of stereo depth: 2 m is far worse than 400 mm, not slightly worse."""
        rng = np.random.default_rng(1)
        near = inject_depth_noise(self._flat(400.0, (256, 256)), REALSENSE_D435, rng)
        far = inject_depth_noise(self._flat(2000.0, (256, 256)), REALSENSE_D435, rng)
        near_error = np.std(near[near > 0] - 400.0)
        far_error = np.std(far[far > 0] - 2000.0)
        self.assertGreater(far_error, near_error * 4.0, f"near {near_error:.3f} far {far_error:.3f}")

    def test_invalid_pixels_stay_invalid(self) -> None:
        depth = self._flat()
        depth[10:20, 10:20] = 0.0
        noised = inject_depth_noise(depth, REALSENSE_D435, np.random.default_rng(2))
        self.assertTrue(np.all(noised[10:20, 10:20] == 0.0))

    def test_out_of_range_becomes_invalid_rather_than_clipped(self) -> None:
        depth = self._flat(50.0)  # closer than the sensor's minimum range
        noised = inject_depth_noise(depth, REALSENSE_D435, np.random.default_rng(3))
        self.assertEqual(float(noised.max()), 0.0)

    def test_depth_edges_lose_pixels(self) -> None:
        """A real stereo camera fails at discontinuities; a flat wall is not where it struggles."""
        stepped = self._flat(800.0, (128, 128))
        stepped[:, 64:] = 1400.0
        noised = inject_depth_noise(stepped, REALSENSE_D435, np.random.default_rng(4))
        near_edge = noised[:, 60:68]
        far_from_edge = noised[:, :40]
        self.assertGreater(
            float((near_edge == 0).mean()), float((far_from_edge == 0).mean()) + 0.2,
            "the edge should drop out far more than flat regions do",
        )

    def test_it_is_reproducible(self) -> None:
        a = inject_depth_noise(self._flat(), REALSENSE_D435, np.random.default_rng(9))
        b = inject_depth_noise(self._flat(), REALSENSE_D435, np.random.default_rng(9))
        np.testing.assert_array_equal(a, b)

    def test_an_all_invalid_frame_survives(self) -> None:
        out = inject_depth_noise(np.zeros((8, 8)), REALSENSE_D435, np.random.default_rng(0))
        self.assertEqual(float(out.sum()), 0.0)


class LabelTests(unittest.TestCase):
    def _instance_map(self) -> np.ndarray:
        """Built with the REAL painter, so the id convention is exercised rather than hand-written.

        Hand-writing ids here is what let an off-by-one through: the map said object k was k+1 while
        the reader looked up k, and a test that also hand-writes the ids agrees with whichever it
        happens to copy. Objects 1 and 2 are the interesting pair; 3 occludes half of 2.
        """
        masks = {index: np.zeros((40, 40), dtype=bool) for index in (1, 2, 3)}
        masks[1][5:15, 5:15] = True        # fully visible
        masks[2][20:30, 20:25] = True      # half of it hidden by object 3
        masks[3][20:30, 25:30] = True
        return instance_map_from_masks(masks, (40, 40))

    def test_bbox_is_tight_and_exclusive(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:5, 3:8] = True
        self.assertEqual(bbox_from_mask(mask), (3, 2, 8, 5))

    def test_an_empty_mask_has_no_box(self) -> None:
        self.assertIsNone(bbox_from_mask(np.zeros((10, 10), dtype=bool)))

    def test_visibility_is_the_measured_ratio(self) -> None:
        instances = self._instance_map()
        solo = {1: mask_for_instance(instances, pixel_id(1)), 2: np.zeros((40, 40), dtype=bool)}
        solo[2][20:30, 20:30] = True       # unoccluded, object 2 would cover twice what is visible
        labels = build_object_labels(
            instances, solo,
            {1: ("a", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
             2: ("b", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))},
        )
        by_id = {label.instance_id: label for label in labels}
        self.assertAlmostEqual(by_id[1].visibility, 1.0)
        self.assertAlmostEqual(by_id[2].visibility, 0.5)

    def test_an_object_outside_the_frame_is_labelled_zero_not_dropped(self) -> None:
        instances = np.zeros((10, 10), dtype=np.int32)
        labels = build_object_labels(
            instances, {7: np.zeros((10, 10), dtype=bool)},
            {7: ("gone", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))},
        )
        self.assertEqual(len(labels), 1)
        self.assertFalse(labels[0].in_frame)
        self.assertEqual(labels[0].visibility, 0.0)

    def test_a_missing_solo_mask_is_refused_rather_than_guessed(self) -> None:
        """Inventing visibility 1.0 would be indistinguishable from a genuinely unoccluded object."""
        with self.assertRaises(KeyError) as ctx:
            build_object_labels(
                self._instance_map(), {},
                {1: ("a", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))},
            )
        self.assertIn("solo pass", str(ctx.exception))

    def test_object_zero_does_not_read_the_background(self) -> None:
        """The exact defect that refused both smoke scenes on 2026-08-11, in one assertion.

        The painter wrote object k as k+1 while the reader looked up k, so object 0 was handed pixel
        value 0 -- the BACKGROUND, ~90 % of the image. It shows up as visible > unoccluded, which is
        impossible by construction, which is why the scene was refused rather than written. Here
        object 0 is a small square in a large empty frame: reading the background instead would give
        it 1591 of 1600 pixels rather than 9.
        """
        masks = {0: np.zeros((40, 40), dtype=bool), 1: np.zeros((40, 40), dtype=bool)}
        masks[0][1:4, 1:4] = True          # 9 px
        masks[1][20:24, 20:24] = True      # 16 px
        instances = instance_map_from_masks(masks, (40, 40))
        labels = build_object_labels(
            instances, masks,
            {0: ("small", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
             1: ("other", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))},
        )
        by_id = {label.instance_id: label for label in labels}
        self.assertEqual(by_id[0].visible_px, 9)
        self.assertEqual(by_id[1].visible_px, 16)
        for label in labels:
            self.assertLessEqual(label.visible_px, label.unoccluded_px)

    def test_the_painter_and_the_reader_are_inverses(self) -> None:
        """Round-trip over a range of indices: whatever the offset is, both sides must apply the same."""
        rng = np.random.default_rng(0)
        masks = {}
        for index in range(6):
            mask = np.zeros((24, 24), dtype=bool)
            row, col = int(rng.integers(0, 20)), int(rng.integers(0, 20))
            mask[row:row + 3, col:col + 3] = True
            masks[index] = mask
        instances = instance_map_from_masks(masks, (24, 24))
        for index, mask in masks.items():
            recovered = mask_for_instance(instances, pixel_id(index))
            # Later objects win overlaps, so recovered <= mask; nothing outside the mask may appear.
            self.assertTrue(bool((recovered & ~mask).sum() == 0))
        self.assertEqual(int((instances == 0).sum()), int((~np.any(list(masks.values()), axis=0)).sum()))

    def test_visibility_never_divides_by_zero(self) -> None:
        self.assertEqual(visibility(0, 0), 0.0)
        self.assertEqual(visibility(10, 0), 0.0)

    def test_visibility_is_clamped(self) -> None:
        """Anti-aliasing can put a pixel or two of an object outside its own solo mask."""
        self.assertEqual(visibility(105, 100), 1.0)

    def test_labels_are_frozen(self) -> None:
        label = ObjectLabel("a", 1, (0, 0, 1, 1), 1, 1, 1.0, (0.0,) * 3, (0.0, 0.0, 0.0, 1.0))
        with self.assertRaises(Exception):
            label.visibility = 0.5  # type: ignore[misc]


class ViewPolicyTests(unittest.TestCase):
    def test_a_wrist_only_rig_resamples(self) -> None:
        self.assertIs(unreachable_policy(["wrist"]), UnreachablePolicy.RESAMPLE)

    def test_a_rig_with_other_views_skips(self) -> None:
        self.assertIs(
            unreachable_policy(["wrist", "oblique_left"]), UnreachablePolicy.SKIP_VIEW,
        )

    def test_a_reachable_first_try_is_not_a_resample(self) -> None:
        point, outcome, attempts = resolve_wrist_viewpoint(
            ["wrist"], lambda i: i, lambda _p: True,
        )
        self.assertEqual((point, outcome, attempts), (0, ViewOutcome.RENDERED, 1))

    def test_resampling_finds_a_later_candidate_and_says_so(self) -> None:
        point, outcome, attempts = resolve_wrist_viewpoint(
            ["wrist"], lambda i: i, lambda p: p >= 3,
        )
        self.assertEqual(point, 3)
        self.assertIs(outcome, ViewOutcome.RENDERED_AFTER_RESAMPLE)
        self.assertEqual(attempts, 4)

    def test_resampling_gives_up_after_the_budget(self) -> None:
        point, outcome, attempts = resolve_wrist_viewpoint(
            ["wrist"], lambda i: i, lambda _p: False, max_attempts=5,
        )
        self.assertIsNone(point)
        self.assertIs(outcome, ViewOutcome.SKIPPED_UNREACHABLE)
        self.assertEqual(attempts, 5)

    def test_a_multi_view_rig_never_resamples(self) -> None:
        """The bias this avoids: wrist views that are only ever the angles the arm finds easy."""
        calls: list[int] = []

        def sample(attempt: int) -> int:
            calls.append(attempt)
            return attempt

        point, outcome, _ = resolve_wrist_viewpoint(
            ["wrist", "oblique_left"], sample, lambda _p: False,
        )
        self.assertIsNone(point)
        self.assertIs(outcome, ViewOutcome.SKIPPED_UNREACHABLE)
        self.assertEqual(calls, [0], "it must not have kept drawing candidates")


class MaterialTests(unittest.TestCase):
    def test_the_assigned_colour_is_not_re_rolled_for_dielectrics(self) -> None:
        """A referring expression may name the colour; a material sampler must not change it."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            material = sample_material(rng, "packaging", (0.9, 0.1, 0.1))
            if material.metallic == 0.0:
                self.assertEqual(material.base_color_rgb, (0.9, 0.1, 0.1))

    def test_industrial_parts_are_usually_metal_and_packaging_never_is(self) -> None:
        rng = np.random.default_rng(1)
        industrial = [sample_material(rng, "industrial", (0.5,) * 3).metallic for _ in range(200)]
        packaging = [sample_material(rng, "packaging", (0.5,) * 3).metallic for _ in range(200)]
        self.assertGreater(sum(industrial) / len(industrial), 0.5)
        self.assertLess(sum(packaging) / len(packaging), 0.1)

    def test_parameters_stay_in_range(self) -> None:
        rng = np.random.default_rng(2)
        for family in ("primitive", "industrial", "packaging", "vessel"):
            for _ in range(100):
                material = sample_material(rng, family, (0.5, 0.5, 0.5))
                with self.subTest(family=family):
                    self.assertTrue(0.0 <= material.roughness <= 1.0)
                    self.assertIn(material.metallic, (0.0, 1.0))
                    self.assertTrue(0.0 <= material.clearcoat <= 1.0)
                    self.assertTrue(all(0.0 <= c <= 1.0 for c in material.base_color_rgb))

    def test_an_unknown_family_falls_back_rather_than_crashing_a_run(self) -> None:
        material = sample_material(np.random.default_rng(0), "meteorite", (0.5,) * 3)
        self.assertTrue(0.0 <= material.roughness <= 1.0)


class DepthEncodingTests(unittest.TestCase):
    def test_millimetres_round_trip(self) -> None:
        depth = np.array([[0.0, 1.4, 999.6, 65535.0]])
        np.testing.assert_allclose(decode_depth_png(encode_depth_png(depth)), [[0.0, 1.0, 1000.0, 65535.0]])

    def test_out_of_range_becomes_invalid_not_wrapped(self) -> None:
        """A wrapped 4 m reading would be wrong AND plausible, which is worse than being marked bad."""
        encoded = encode_depth_png(np.array([[70000.0, -5.0, np.nan, np.inf]]))
        self.assertEqual(encoded.tolist(), [[0, 0, 0, 0]])

    def test_the_dtype_is_uint16(self) -> None:
        self.assertEqual(encode_depth_png(np.array([[123.0]])).dtype, np.uint16)


if __name__ == "__main__":
    unittest.main()
