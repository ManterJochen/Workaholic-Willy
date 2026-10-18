"""The layout half, asserted where it is cheap — which is the entire reason it is Isaac-free.

Every claim in these tests is one that would otherwise be discovered after a night of path tracing:
a seed that does not reproduce, a family mix that came out lopsided, objects overlapping before physics
even runs, a camera rig aimed at nothing. None of them need a GPU to catch, and all of them are
expensive to catch late.
"""

from __future__ import annotations

import math
import unittest
from collections import Counter
from dataclasses import asdict

from datagen.config import CameraRigConfig, DatagenConfig, FamilyMixConfig
from datagen.scenes import CameraMount, SceneFamily, layout_scene, plan_families, scene_seed
from datagen.scenes.layout import _oriented_half_extents, layout_scene_with_assets


def _config(**overrides: object) -> DatagenConfig:
    return DatagenConfig(**{"scenes": 40, **overrides})  # type: ignore[arg-type]


class DeterminismTests(unittest.TestCase):
    def test_the_same_seed_gives_a_byte_identical_scene(self) -> None:
        a = layout_scene(_config(seed=7), 3)
        b = layout_scene(_config(seed=7), 3)
        self.assertEqual(asdict(a), asdict(b))

    def test_a_different_seed_gives_a_different_scene(self) -> None:
        a = layout_scene(_config(seed=7), 3)
        b = layout_scene(_config(seed=8), 3)
        self.assertNotEqual(asdict(a), asdict(b))

    def test_neighbouring_datasets_do_not_share_scenes(self) -> None:
        """The trap sequential seeding walks into: dataset N+1 reusing dataset N's scenes, shifted.

        With ``seed + index`` as the per-scene seed, scene 1 of dataset 0 and scene 0 of dataset 1 are
        the SAME scene. Two datasets then look independent and are not, and every number measured
        across them is correlated in a way nobody wrote down.
        """
        first = {scene_seed(0, i) for i in range(50)}
        second = {scene_seed(1, i) for i in range(50)}
        self.assertEqual(first & second, set())

    def test_the_seed_stream_refuses_negative_indices(self) -> None:
        with self.assertRaises(ValueError):
            scene_seed(0, -1)

    def test_streams_are_independent(self) -> None:
        self.assertNotEqual(scene_seed(0, 0, stream=0), scene_seed(0, 0, stream=1))


class FamilyMixTests(unittest.TestCase):
    def test_equal_weights_give_an_exact_split(self) -> None:
        counts = Counter(f.value for f in plan_families(_config(scenes=400)))
        self.assertEqual(set(counts.values()), {100}, counts)

    def test_weights_are_honoured(self) -> None:
        config = DatagenConfig(scenes=100, families=FamilyMixConfig(
            sparse_weight=3.0, packed_weight=1.0, pile_weight=0.0, bin_weight=0.0,
        ))
        counts = Counter(f.value for f in plan_families(config))
        self.assertEqual(counts["sparse"], 75)
        self.assertEqual(counts["packed"], 25)
        self.assertNotIn("pile", counts)

    def test_the_families_are_interleaved(self) -> None:
        """A truncated or interrupted run must still have seen every family, not 100 sparse scenes."""
        families = plan_families(_config(scenes=400))
        self.assertEqual(len({f.value for f in families[:40]}), 4)

    def test_every_scene_index_is_assigned(self) -> None:
        config = _config(scenes=37)
        self.assertEqual(len(plan_families(config)), 37)

    def test_a_mix_with_no_weight_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            FamilyMixConfig(sparse_weight=0.0, packed_weight=0.0, pile_weight=0.0, bin_weight=0.0)


class PlacementTests(unittest.TestCase):
    def _scene(self, family: SceneFamily):  # noqa: ANN202 - a fixture, typed by use
        return layout_scene(_config(seed=11), 0, family=family)

    def test_sparse_objects_do_not_touch(self) -> None:
        """The baseline family's whole job: nothing overlaps, so a failure there is not a layout bug."""
        scene = self._scene(SceneFamily.SPARSE)
        for i, a in enumerate(scene.objects):
            for b in scene.objects[i + 1:]:
                distance = math.dist(a.position_mm[:2], b.position_mm[:2])
                self.assertGreater(distance, 25.0, f"{a.asset_id} and {b.asset_id} are on top of each other")

    def test_packed_objects_are_closer_than_sparse_ones(self) -> None:
        def closest(scene) -> float:  # noqa: ANN001
            return min(
                math.dist(a.position_mm[:2], b.position_mm[:2])
                for i, a in enumerate(scene.objects) for b in scene.objects[i + 1:]
            )

        self.assertLess(closest(self._scene(SceneFamily.PACKED)), closest(self._scene(SceneFamily.SPARSE)))

    def test_pile_objects_never_spawn_inside_one_another(self) -> None:
        """The defect that made a pile LAUNCH instead of settle, and the test that missed it.

        The previous version of this test asserted that the spawn heights were distinct and called
        that "would spawn interpenetrating". Distinct heights prevent nothing: the pitch was a fixed
        22 mm while assets run to 118 mm, so objects spawned deep inside each other, PhysX applied a
        separating impulse, and one object was still travelling 1267 mm per half-second after 1080
        settle steps. Orientation is random, so the honest test is on bounding spheres.
        """
        for seed in (11, 12, 13, 14, 15):
            spec, assets = layout_scene_with_assets(_config(seed=seed), 0, family=SceneFamily.PILE)
            extents = {asset.asset_id: asset.extent_mm for asset in assets}
            for i, a in enumerate(spec.objects):
                half_a = _oriented_half_extents(extents[a.asset_id], a.orientation_xyzw)
                for b in spec.objects[i + 1:]:
                    half_b = _oriented_half_extents(extents[b.asset_id], b.orientation_xyzw)
                    # Disjoint on ANY axis is enough: two separated AABBs cannot contain overlapping
                    # shapes. Asserted per axis so a failure says which direction is too tight.
                    separated = [
                        abs(a.position_mm[axis] - b.position_mm[axis]) >= half_a[axis] + half_b[axis]
                        for axis in range(3)
                    ]
                    self.assertTrue(
                        any(separated),
                        f"seed {seed}: {a.asset_id} and {b.asset_id} overlap on every axis -- "
                        f"they start inside each other",
                    )

    def test_pile_objects_start_above_the_table_and_stay_low(self) -> None:
        """Low matters as much as non-overlapping, and for a measured reason.

        A column that merely stacks each object above the last is non-overlapping and useless: it
        reached 1134 mm, objects hit at ~4.7 m/s, and two of twelve left the table altogether. The
        packing drops each object into the lowest free spot instead, which measured 310 mm for the
        same scene. Anything approaching a metre is the tower coming back.
        """
        scene = self._scene(SceneFamily.PILE)
        heights = [o.position_mm[2] for o in scene.objects]
        self.assertTrue(all(z > 0.0 for z in heights), heights)
        self.assertLess(max(heights), 600.0, f"the drop column is a tower again: {max(heights):.0f} mm")
        self.assertGreater(scene.drop_height_mm, 0.0)

    def test_pile_objects_stay_inside_the_workspace(self) -> None:
        """Objects released outside the workspace cannot land in it, and cannot be picked from it."""
        for seed in (11, 12, 13):
            scene = layout_scene(_config(seed=seed), 0, family=SceneFamily.PILE)
            for placement in scene.objects:
                distance = math.dist(placement.position_mm[:2], (450.0, 0.0))
                self.assertLess(distance, 150.0, f"{placement.asset_id} spawns {distance:.0f} mm out")

    def test_flat_families_spawn_upright_on_the_table(self) -> None:
        for family in (SceneFamily.SPARSE, SceneFamily.PACKED, SceneFamily.BIN):
            with self.subTest(family=family):
                for placement in self._scene(family).objects:
                    # yaw-only quaternion: x and y components are zero
                    self.assertAlmostEqual(placement.orientation_xyzw[0], 0.0)
                    self.assertAlmostEqual(placement.orientation_xyzw[1], 0.0)

    def test_orientations_are_unit_quaternions(self) -> None:
        for family in SceneFamily:
            for placement in self._scene(family).objects:
                norm = math.sqrt(sum(component ** 2 for component in placement.orientation_xyzw))
                self.assertAlmostEqual(norm, 1.0, places=6)

    def test_the_bin_family_emits_walls_and_keeps_objects_inside(self) -> None:
        scene = self._scene(SceneFamily.BIN)
        self.assertEqual(len(scene.bin_walls), 4)
        inner_x = max(abs(w.center_mm[0] - scene.objects[0].position_mm[0]) for w in scene.bin_walls)
        self.assertGreater(inner_x, 0.0)
        for placement in scene.objects:
            self.assertLess(abs(placement.position_mm[0] - 450.0), 150.0)
            self.assertLess(abs(placement.position_mm[1]), 100.0)

    def test_only_the_bin_family_has_walls(self) -> None:
        for family in (SceneFamily.SPARSE, SceneFamily.PACKED, SceneFamily.PILE):
            with self.subTest(family=family):
                self.assertEqual(self._scene(family).bin_walls, ())

    def test_object_counts_stay_in_their_configured_band(self) -> None:
        config = _config(seed=3)
        for index in range(config.scenes):
            scene = layout_scene(config, index)
            low, high = {
                SceneFamily.SPARSE: config.families.sparse_objects,
                SceneFamily.PACKED: config.families.packed_objects,
                SceneFamily.PILE: config.families.pile_objects,
                SceneFamily.BIN: config.families.bin_objects,
            }[scene.family]
            self.assertGreaterEqual(len(scene.objects), low)
            self.assertLess(len(scene.objects), high)

    def test_an_out_of_range_index_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            layout_scene(_config(scenes=5), 5)


class CameraRigTests(unittest.TestCase):
    def test_the_default_rig_is_the_owners(self) -> None:
        """Wrist plus the raised oblique pair — not the overhead camera, by decision."""
        scene = layout_scene(_config(), 0)
        self.assertEqual([c.name for c in scene.cameras], ["wrist", "oblique_left", "oblique_right"])

    def test_the_obliques_flank_the_workspace_and_look_down_at_it(self) -> None:
        scene = layout_scene(_config(), 0)
        left = next(c for c in scene.cameras if c.name == "oblique_left")
        right = next(c for c in scene.cameras if c.name == "oblique_right")
        self.assertLess(left.position_mm[1], scene.cameras[0].look_at_mm[1])
        self.assertGreater(right.position_mm[1], scene.cameras[0].look_at_mm[1])
        for camera in (left, right):
            self.assertGreater(camera.position_mm[2], 300.0, "an oblique view has to be raised")
            self.assertEqual(camera.look_at_mm[:2], (450.0, 0.0))
            self.assertIs(camera.mount, CameraMount.FIXED)

    def test_the_wrist_view_is_marked_as_arm_carried(self) -> None:
        """It is the one view the renderer may fail to reach, so it must be distinguishable."""
        wrist = next(c for c in layout_scene(_config(), 0).cameras if c.name == "wrist")
        self.assertIs(wrist.mount, CameraMount.WRIST)

    def test_the_rig_is_selectable(self) -> None:
        only_wrist = layout_scene(_config(camera_rig=CameraRigConfig(views=("wrist",))), 0)
        self.assertEqual([c.name for c in only_wrist.cameras], ["wrist"])
        # ⚠ NO ARM WITH NO WRIST VIEW. Since 2026-09-04 a posed arm with no wrist camera is refused
        # at config time: without one the arm can only be parked, and a parked arm is measured to sit
        # outside the oblique views, so the dataset would contain no arm while the config claims one.
        # The renderer already refused this per SCENE; the rule moved the refusal to where it costs
        # nothing.
        no_wrist = layout_scene(
            _config(camera_rig=CameraRigConfig(views=("oblique_left", "oblique_right")),
                    render={"arm": {"mode": "absent"}}), 0,
        )
        self.assertEqual([c.mount for c in no_wrist.cameras], [CameraMount.FIXED] * 2)

    def test_an_empty_rig_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            CameraRigConfig(views=())

    def test_a_duplicated_view_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            CameraRigConfig(views=("wrist", "wrist"))


class SpecTests(unittest.TestCase):
    def test_asset_ids_are_deduplicated_and_ordered(self) -> None:
        scene = layout_scene(_config(), 0)
        ids = scene.asset_ids()
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(set(ids) <= {o.asset_id for o in scene.objects})

    def test_the_spec_is_frozen(self) -> None:
        scene = layout_scene(_config(), 0)
        with self.assertRaises(Exception):
            scene.seed = 5  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
