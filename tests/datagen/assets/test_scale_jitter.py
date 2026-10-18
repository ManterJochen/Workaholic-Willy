"""Resizing the scene — the free lever against the gap that `base_control` measured.

⚠ WHY SIZE AND NOT SOMETHING ELSE. MEASURED on `base_control`, a net over 168 asset groups: within one
object the labelled grasp width varies by 2.32 mm and across the corpus by 23.6 mm, so width is very
nearly an OBJECT PROPERTY -- the kind of thing a net can memorise per asset instead of reading off the
geometry. It does exactly that: 14.31 mm MAE on assets it has seen, 26.57 mm on assets it has not,
against 23.27 mm for a constant median. Resizing makes shape and answer independent.

The two things that could go wrong, and are pinned here: an augmentation that is "off" but still
draws (every later sample in the epoch shifts, and no comparison to an earlier run means anything),
and an augmentation that mints a grasp WIDER THAN THE GRIPPER (the net would learn to propose it).
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.deep.corpus.sample import (
    _MAX_JAW_APERTURE_MM,
    SampleSpec,
    _draw_scale,
    build_sample,
)


def _scene(width_mm: float = 40.0, grasps: int = 4, points: int = 160) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    obj = np.column_stack([rng.uniform(-40, 40, points), rng.uniform(-40, 40, points),
                           rng.uniform(20, 60, points)])
    env = np.column_stack([rng.uniform(-120, 120, points), rng.uniform(-120, 120, points),
                           np.zeros(points)])
    xyz = np.vstack([obj, env]).astype(np.float32)
    instance = np.concatenate([np.zeros(points, dtype=np.int16),
                               np.full(points, -1, dtype=np.int16)])
    position = obj[:grasps].copy()
    contacts = np.repeat(position, 2, axis=0)
    contacts[0::2, 0] -= 1.0
    contacts[1::2, 0] += 1.0
    return {
        "points_mm": xyz,
        "normals": np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(xyz), 1)),
        "normal_valid": np.ones(len(xyz), dtype=bool),
        "view_count": np.full(len(xyz), 2, dtype=np.uint8),
        "instance_id": instance,
        "grasp_position_mm": position.astype(np.float32),
        "grasp_approach": np.tile(np.array([0.0, 0.0, -1.0]), (grasps, 1)).astype(np.float32),
        "grasp_axis": np.tile(np.array([1.0, 0.0, 0.0]), (grasps, 1)).astype(np.float32),
        "grasp_width_mm": np.full(grasps, width_mm, dtype=np.float32),
        "grasp_instance": np.zeros(grasps, dtype=np.int16),
        "grasp_held": np.full(grasps, -1, dtype=np.int8),
        "grasp_approach_admissible": np.ones(grasps, dtype=bool),
        "grasp_part_role": np.asarray([""] * grasps, dtype="<U8"),
        "contact_points_mm": contacts.astype(np.float32),
        "contact_grasp_index": np.repeat(np.arange(grasps), 2).astype(np.int32),
        "object_instance": np.zeros(1, dtype=np.int16),
        "object_asset_id": np.asarray(["gso_test"], dtype="<U32"),
    }


def _widths(sample: dict) -> np.ndarray:
    width = np.asarray(sample["width_m"])
    return width[np.isfinite(width)] * 1000.0


class OffMeansUntouchedTests(unittest.TestCase):
    def test_the_default_takes_NO_DRAW_so_the_stream_is_not_advanced(self) -> None:
        """A draw taken and discarded still moves the generator, and every sample after it in the
        epoch would differ -- byte-identity lost for an augmentation that is switched OFF."""
        rng = np.random.default_rng(0)
        before = rng.bit_generator.state
        self.assertEqual(_draw_scale(rng, SampleSpec(), np.full(3, 40.0)), 1.0)
        self.assertEqual(rng.bit_generator.state, before)

    def test_the_test_above_is_not_vacuous(self) -> None:
        rng = np.random.default_rng(0)
        before = rng.bit_generator.state
        _draw_scale(rng, SampleSpec(scale_jitter=(0.8, 1.25)), np.full(3, 40.0))
        self.assertNotEqual(rng.bit_generator.state, before)

    def test_a_sample_with_the_jitter_off_matches_one_built_without_the_field(self) -> None:
        scene = _scene()
        a = build_sample(scene, np.random.default_rng(3), SampleSpec(points=256))
        b = build_sample(scene, np.random.default_rng(3),
                         SampleSpec(points=256, scale_jitter=(1.0, 1.0)))
        np.testing.assert_array_equal(a["points_m"], b["points_m"])
        np.testing.assert_array_equal(a["width_m"], b["width_m"])


class ItActuallyResizesTests(unittest.TestCase):
    def test_a_fixed_factor_scales_the_cloud_the_width_AND_the_depth(self) -> None:
        scene = _scene(width_mm=40.0)
        plain = build_sample(scene, np.random.default_rng(3), SampleSpec(points=256))
        big = build_sample(scene, np.random.default_rng(3),
                           SampleSpec(points=256, scale_jitter=(2.0, 2.0)))
        self.assertAlmostEqual(float(np.max(np.abs(big["points_m"]))),
                               2.0 * float(np.max(np.abs(plain["points_m"]))), places=5)
        self.assertAlmostEqual(float(np.mean(_widths(big))),
                               2.0 * float(np.mean(_widths(plain))), places=3)

    def test_the_DIRECTIONS_do_not_move_where_both_carry_a_label(self) -> None:
        """A uniform resize leaves every direction where it was, so a bin that CHANGED would mean the
        augmentation had rotated something.

        ⚠ ONLY WHERE BOTH CARRY ONE, and the first version of this test asserted otherwise and failed.
        It was the test that was wrong: the 10 mm assignment radius is absolute and deliberately does
        not scale, so spreading the cloud apart leaves fewer points inside it. Which points are
        labelled is EXPECTED to change; what each label SAYS is not."""
        scene = _scene()
        plain = build_sample(scene, np.random.default_rng(3),
                             SampleSpec(points=256, rotate_z=False))
        big = build_sample(scene, np.random.default_rng(3),
                           SampleSpec(points=256, rotate_z=False, scale_jitter=(2.0, 2.0)))
        both = (np.asarray(plain["approach_bin"]) >= 0) & (np.asarray(big["approach_bin"]) >= 0)
        self.assertTrue(both.any(), "nothing was labelled on both sides -- the test proves nothing")
        np.testing.assert_array_equal(np.asarray(plain["approach_bin"])[both],
                                      np.asarray(big["approach_bin"])[both])
        np.testing.assert_array_equal(np.asarray(plain["rotation_bin"])[both],
                                      np.asarray(big["rotation_bin"])[both])

    def test_spreading_the_cloud_apart_leaves_FEWER_points_inside_the_pad_radius(self) -> None:
        """The consequence the test above had to be corrected for, asserted directly so that it is a
        stated property of the augmentation rather than a surprise the next reader rediscovers."""
        scene = _scene()
        labelled = {}
        for name, jitter in (("plain", (1.0, 1.0)), ("big", (2.0, 2.0))):
            sample = build_sample(scene, np.random.default_rng(3),
                                  SampleSpec(points=256, rotate_z=False, scale_jitter=jitter))
            labelled[name] = int((np.asarray(sample["approach_bin"]) >= 0).sum())
        self.assertLess(labelled["big"], labelled["plain"])


class TheJawIsNotNegotiableTests(unittest.TestCase):
    def test_scaling_up_STOPS_at_the_aperture_rather_than_minting_an_impossible_grasp(self) -> None:
        widths = np.full(4, 60.0)
        factor = _draw_scale(np.random.default_rng(0), SampleSpec(scale_jitter=(2.0, 2.0)), widths)
        self.assertAlmostEqual(factor, _MAX_JAW_APERTURE_MM / 60.0, places=6)
        self.assertLessEqual(float(np.max(widths * factor)), _MAX_JAW_APERTURE_MM + 1e-9)

    def test_no_drawn_factor_EVER_takes_a_label_past_the_jaw(self) -> None:
        rng = np.random.default_rng(0)
        for width in (10.0, 40.0, 60.0, 84.0):
            widths = np.full(4, width)
            for _ in range(50):
                factor = _draw_scale(rng, SampleSpec(scale_jitter=(0.8, 1.6)), widths)
                self.assertLessEqual(width * factor, _MAX_JAW_APERTURE_MM + 1e-9,
                                     f"{width} mm x {factor} outgrew the gripper")

    def test_a_scene_with_no_labelled_grasp_still_gets_a_factor(self) -> None:
        factor = _draw_scale(np.random.default_rng(0), SampleSpec(scale_jitter=(0.8, 1.25)),
                             np.zeros(0))
        self.assertTrue(0.8 <= factor <= 1.25)

    def test_the_copied_aperture_agrees_with_the_one_datagen_labels_by(self) -> None:
        """⚠ `backend` may never import `datagen`, so the constant is a COPY. This test importing
        both is the only thing standing between a copy and a silent divergence -- the same fence
        `test_deep_calculator` keeps around the voxel sizes."""
        from datagen.assets.procedural import MAX_JAW_APERTURE_MM

        self.assertEqual(_MAX_JAW_APERTURE_MM, MAX_JAW_APERTURE_MM)


class LogUniformTests(unittest.TestCase):
    def test_it_shrinks_about_as_often_as_it_grows(self) -> None:
        """A UNIFORM draw over (0.8, 1.25) grows 69 % of the time; the geometric midpoint is what
        makes a resize symmetric."""
        rng = np.random.default_rng(0)
        spec = SampleSpec(scale_jitter=(0.8, 1.25))
        drawn = [_draw_scale(rng, spec, np.zeros(0)) for _ in range(4000)]
        grew = float(np.mean([f > 1.0 for f in drawn]))
        self.assertTrue(0.45 <= grew <= 0.55, f"grew {grew:.3f} of the time")


if __name__ == "__main__":
    unittest.main()
