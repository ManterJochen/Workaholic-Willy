"""`engine: none` — a corpus with no physics engine, and the refusals that keep it honest.

⭑ WHY IT IS POSSIBLE, and it was measured rather than hoped: `scenes/layout.py:556` sets
``drop_height = 0.0``, so `sparse`, `packed` and `bin` are placed FLAT and only `pile` drops. Three of
four families at default weights need no dynamics — and rendering needs no engine either, because the
corpus carries no RGB.

⛔ THE TWO REFUSALS ARE THE FEATURE. `pile` is refused BY NAME, and `arm.mode: driven` likewise. A
faked settle renders a perfectly plausible depth image over interpenetrating bodies and nothing
downstream can tell — the same class as the blank-RGB defect, which passed every automated check as
`ok` and was found only by a human opening the folder.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import trimesh

from datagen.config import DatagenConfig
from datagen.render.engine import build_engine
from datagen.render.equivalence import (
    check_depth_reconstructs_geometry,
    check_points_match_analytic_rays,
    oblique_camera,
)
from datagen.render.noengine import NoEngineRenderer, REFUSED_FAMILIES, is_stable, seat_on_support
from datagen.scenes.spec import ObjectPlacement


def _cfg(**render) -> DatagenConfig:
    return DatagenConfig.model_validate({"render": {"engine": "none", **render}})


def _placed(mesh: trimesh.Trimesh, *, quat=(0.0, 0.0, 0.0, 1.0), z_mm: float = 100.0):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    placement = ObjectPlacement(asset_id="x", position_mm=(0.0, 0.0, z_mm), orientation_xyzw=quat)
    return seat_on_support(vertices, placement, 0.0)


class SeatingIsExactTests(unittest.TestCase):
    """⭑ `_place_flat` spawns 2 mm high ON PURPOSE, because Isaac's settle drops the body onto the
    table. There is no settle here, so the height is arithmetic instead -- and arithmetic is exact."""

    def test_a_body_rests_EXACTLY_on_the_support(self) -> None:
        posed, position = _placed(trimesh.creation.box(extents=(60.0, 60.0, 60.0)), z_mm=32.0)
        self.assertAlmostEqual(float(posed[:, 2].min()), 0.0, places=9)
        self.assertAlmostEqual(float(position[2]), 30.0, places=9,
                               msg="the 2 mm spawn clearance was not removed by seating")

    def test_it_is_right_for_a_mesh_whose_ORIGIN_IS_NOT_ITS_CENTRE(self) -> None:
        """⚠ THE TRAP THIS PROJECT HAS ALREADY PAID FOR with scanned assets: the reported pose is the
        ORIGIN, not the centre. A settle hides it; arithmetic on the body's own AABB does not need to."""
        mesh = trimesh.creation.box(extents=(40.0, 40.0, 40.0))
        mesh.apply_translation([0.0, 0.0, 137.0])          # origin nowhere near the centroid
        posed, _position = _placed(mesh)
        self.assertAlmostEqual(float(posed[:, 2].min()), 0.0, places=9)

    def test_a_ROTATED_body_rests_on_its_rotated_lowest_point(self) -> None:
        angle = math.radians(30.0)
        posed, _ = _placed(trimesh.creation.box(extents=(80.0, 40.0, 40.0)),
                           quat=(0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)))
        self.assertAlmostEqual(float(posed[:, 2].min()), 0.0, places=9)


class StabilityIsJudgedGeometricallyTests(unittest.TestCase):
    """⚠ Isaac rejects ~13 % of scenes as `unstable` (one shard: ok 658 / unstable 105 /
    render_error 62). This cannot reject what a solver would; it rejects what GEOMETRY can."""

    def test_an_upright_box_stands(self) -> None:
        posed, _ = _placed(trimesh.creation.box(extents=(60.0, 60.0, 60.0)))
        self.assertTrue(is_stable(posed, 0.0)[0])

    def test_a_SPHERE_stands_and_this_is_the_defect_that_was_caught(self) -> None:
        """⛔ THE FIRST VERSION REJECTED EVERY SPHERE. A sphere touches at ONE point, and a
        contact-count rule called that unsupportable -- which would have quietly deleted a whole shape
        family from every corpus this backend produced. `FAMILY_KINDS["primitive"]` lists sphere and
        capsule; the corpus contains both. A point contact is stable when the centre of mass is above
        it, which is the same physical criterion the polygon expresses."""
        posed, _ = _placed(trimesh.creation.icosphere(subdivisions=3, radius=30.0))
        stable, why = is_stable(posed, 0.0)
        self.assertTrue(stable, why)

    def test_a_cylinder_ON_ITS_SIDE_stands(self) -> None:
        """A LINE contact -- the other degenerate case, and equally real."""
        angle = math.radians(90.0)
        posed, _ = _placed(trimesh.creation.cylinder(radius=25.0, height=200.0, sections=24),
                           quat=(math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)))
        stable, why = is_stable(posed, 0.0)
        self.assertTrue(stable, why)

    def test_a_TILTED_column_is_rejected_and_the_reason_is_readable(self) -> None:
        angle = math.radians(40.0)
        posed, _ = _placed(trimesh.creation.box(extents=(20.0, 20.0, 200.0)),
                           quat=(math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)))
        stable, why = is_stable(posed, 0.0)
        self.assertFalse(stable)
        self.assertIn("centre of mass", why)

    def test_a_FLOATING_body_is_rejected(self) -> None:
        posed, _ = _placed(trimesh.creation.box(extents=(40.0, 40.0, 40.0)))
        stable, why = is_stable(posed + np.array([0.0, 0.0, 50.0]), 0.0)
        self.assertFalse(stable)
        self.assertIn("floats", why)

    def test_it_emits_no_numpy_warning(self) -> None:
        """⚠ The first version asked trimesh for a 3-D hull of a PLANAR contact set and got
        `RuntimeWarning: invalid value encountered in divide` from its centre-of-mass integral. A
        warning is a defect that has not been read yet."""
        import warnings

        posed, _ = _placed(trimesh.creation.icosphere(subdivisions=2, radius=25.0))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            self.assertTrue(is_stable(posed, 0.0)[0])


class TheRefusalsTests(unittest.TestCase):
    def test_pile_is_refused_BY_NAME_and_the_reason_names_the_way_out(self) -> None:
        self.assertIn("pile", REFUSED_FAMILIES)
        reason = REFUSED_FAMILIES["pile"]
        self.assertIn("engine: isaac", reason, "the refusal must name the alternative")
        self.assertIn("interpenetrating", reason, "it must say WHY faking it is worse than refusing")

    def test_only_pile_is_refused(self) -> None:
        """`sparse`, `packed` and `bin` are placed flat -- refusing them would throw away three
        quarters of the families for no measured reason."""
        self.assertEqual(set(REFUSED_FAMILIES), {"pile"})

    def test_the_engine_is_selectable_and_needs_nothing_installed(self) -> None:
        engine = build_engine(_cfg())
        self.assertIsInstance(engine, NoEngineRenderer)

    def test_the_arm_is_posed_from_a_mount_that_needs_NO_simulator(self) -> None:
        """⛔ THE CLAIM I GOT WRONG, pinned so it cannot come back. I read `isaac.py::_pose_the_arm`
        reaching for `self._robot.wrist_camera_mount`, concluded the mount was measured on the live
        articulation, and parked the arm instead. It is not measured at run time: `robots.py:117`
        resolves it from constants taken on-box in 2026-06-07. MEASURED consequence of the wrong
        version -- **0 arm pixels in both oblique cameras**, because park faces away from the table, so
        `arm.mode: posed` was a silent no-op."""
        from datagen.robots import resolve_robot

        mount = resolve_robot("ur5e").wrist_camera_mount
        self.assertIsNotNone(mount, "the measured mount no longer resolves without a simulator")
        self.assertEqual(np.asarray(mount.camera_to_link()).shape, (4, 4))

    def test_the_arm_geometry_loads_and_stands_where_an_arm_stands(self) -> None:
        from datagen.render.kinematics import PARK_JOINTS_RAD
        from datagen.render.noengine import arm_geometry

        links = arm_geometry("ur5e", PARK_JOINTS_RAD)
        self.assertGreaterEqual(len(links), 6, "fewer link meshes than a UR has links")
        vertices = np.vstack([piece[0] for piece in links])
        self.assertGreater(float(vertices[:, 2].max()), 300.0, "the arm is flat on the floor")

    def test_a_posed_arm_WITHOUT_a_wrist_camera_is_refused_rather_than_parked(self) -> None:
        """⚠ Isaac parks here because it still renders the wrist view it planned. This backend renders
        no wrist view, so a parked arm is invisible in every camera -- the corpus would contain no arm
        while the config claimed one. Measured: 0 pixels in both oblique views at park."""
        from datagen.render.noengine import NoEngineRenderer

        engine = NoEngineRenderer(DatagenConfig.model_validate(
            {"render": {"engine": "none", "arm": {"mode": "posed"}}}))
        spec = type("Spec", (), {"cameras": (), "objects": (), "scene_id": "x"})()
        with self.assertRaises(ValueError) as caught:
            engine._arm(spec, None, {})                          # noqa: SLF001 - the unit under test
        message = str(caught.exception)
        self.assertIn("arm.mode: absent", message, "the refusal must name the way out")
        self.assertIn("OUTSIDE both oblique views", message, "it must say WHY parking is not an option")

    def test_the_arm_is_SETTABLE_and_driven_is_refused_by_name(self) -> None:
        """The owner asked for the arm to be settable rather than absent. `absent` and `posed` are the
        backend's own business; `driven` needs a simulator to drive it, so it is refused with a
        sentence rather than silently treated as `posed`."""
        from datagen.config import DatagenConfig as Cfg

        for mode in ("absent", "posed", "driven"):
            with self.subTest(mode=mode):
                Cfg.model_validate({"render": {"engine": "none", "arm": {"mode": mode}}})


class ItPassesTheSameGateTests(unittest.TestCase):
    """The backend's renderer is the one already graded by Tier 0 -- asserted here on a body seated by
    THIS module, so the seating and the rendering are checked together rather than separately."""

    def test_a_seated_body_reconstructs_to_where_it_was_seated(self) -> None:
        from datagen.render.raster import rasterise

        camera_to_base, k = oblique_camera(resolution=(320, 240))
        quad = (np.array([[-700.0, -700.0, 0.0], [700.0, -700.0, 0.0],
                          [700.0, 700.0, 0.0], [-700.0, 700.0, 0.0]]),
                np.array([[0, 1, 2], [0, 2, 3]]))
        posed, _ = _placed(trimesh.creation.box(extents=(80.0, 80.0, 50.0)), z_mm=99.0)
        bodies = {1: quad, 2: (posed, np.asarray(trimesh.creation.box(
            extents=(80.0, 80.0, 50.0)).faces))}
        view = rasterise(bodies, camera_to_base, k, (320, 240))

        table = view.instance_map == 1
        for check in (check_depth_reconstructs_geometry, check_points_match_analytic_rays):
            finding = check(view.depth_mm, table, camera_to_base, k, plane_z_mm=0.0)
            self.assertTrue(finding.passed, f"{finding.check}: {finding.measured:.4f} mm")

        from datagen.render.camera import unproject_to_base

        body = unproject_to_base(view.depth_mm, view.instance_map == 2, camera_to_base, k)
        self.assertGreater(len(body), 200)
        self.assertAlmostEqual(float(body[:, 2].max()), 50.0, delta=0.6,
                               msg="the seated body's top face is not at its seated height")


class InstanceIdSpaceTests(unittest.TestCase):
    """⛔ THE INSTANCE MAP IS IN THE PIXEL-ID SPACE, and this path wrote it in the body-index space.

    `pixel_id`'s own docstring records this defect from 2026-08-11, in two different id spaces: the
    painter wrote object k as k+1 while the reader looked up k, so every label got its neighbour's
    pixels. It came back here, because `rasterise` writes ids "as they are" and `_view` keyed it by
    the body index while `build_object_labels`, `verify` and `corpus/clouds.py:166` all read k+1.

    MEASURED before the fix, on three raster scenes: 7 of 18 pose-to-box checks failed and the
    verifier reported `visible > unoccluded -- impossible by construction`, against 1 in 1503 on the
    Isaac corpus. After: 34 of 34 pass. It was caught before any corpus rested on it -- only three
    non-Isaac datasets existed, all probes -- and that was luck, not process, which is why this is
    here.
    """

    def _render_two_objects(self):
        from datagen.config import DatagenConfig
        from datagen.render.engine import build_engine
        from datagen.scenes.layout import layout_scene_with_assets
        from datagen.scenes.spec import SceneFamily
        from datagen.assets.manifest import AssetManifest

        config = DatagenConfig(
            scenes=1, seed=5,
            assets={"procedural_weight": 1.0},
            families={"sparse_weight": 1.0, "packed_weight": 0.0,
                      "pile_weight": 0.0, "bin_weight": 0.0},
            camera_rig={"views": ("overhead", "oblique_left")},
            render={"engine": "none", "mode": "raster", "depth_only": True,
                    "arm": {"mode": "absent"}},
        )
        spec, assets = layout_scene_with_assets(config, 0, SceneFamily.SPARSE)
        manifest = AssetManifest()
        for asset in assets:
            manifest.add(asset.as_record())
        with build_engine(config) as engine:
            return engine.render(spec, manifest, np.random.default_rng(0))

    def test_a_labels_pixels_are_its_own_and_never_its_neighbours(self) -> None:
        """The invariant the verifier calls impossible by construction: an object cannot show MORE
        pixels in a scene that contains occluders than it shows rendered alone."""
        from datagen.render.labels import pixel_id

        result = self._render_two_objects()
        self.assertEqual(result.status, "ok", result.note)
        self.assertTrue(result.views, "no view rendered")
        for view in result.views:
            for label in view.labels:
                self.assertLessEqual(
                    label.visible_px, label.unoccluded_px,
                    f"{view.name}/object {label.instance_id}: {label.visible_px} visible against "
                    f"{label.unoccluded_px} alone -- the mask is reading another object's pixels")
                painted = int(np.count_nonzero(view.instance_map == pixel_id(label.instance_id)))
                self.assertEqual(
                    painted, label.visible_px,
                    f"{view.name}/object {label.instance_id}: the map holds {painted} pixels at "
                    f"pixel_id but the label counted {label.visible_px} -- two id spaces again")

    def test_the_map_holds_EXACTLY_the_pixel_ids_of_its_labels(self) -> None:
        """The direct statement of the invariant, so a revert fails HERE rather than three stages later
        in a cloud corpus where every object carries its neighbour's points.

        ⚠ AN EARLIER VERSION OF THIS TEST CHECKED THAT NO BODY WAS PAINTED AT ITS RAW INDEX, and that
        witness stopped existing the moment instances became 0-based like Isaac's: instance 0's raw
        value IS the background value, so the check fired on every scene. Comparing the SETS is the
        statement that survives a renumbering, because it names both halves of the mapping.
        """
        from datagen.render.labels import pixel_id

        result = self._render_two_objects()
        self.assertEqual(result.status, "ok", result.note)
        for view in result.views:
            painted = {int(v) for v in np.unique(view.instance_map)} - {0}
            expected = {pixel_id(label.instance_id) for label in view.labels
                        if label.visible_px > 0}
            self.assertEqual(
                painted, expected,
                f"{view.name}: the map paints {sorted(painted)} but the labels claim "
                f"{sorted(expected)} -- the two are in different id spaces")


if __name__ == "__main__":
    unittest.main()
