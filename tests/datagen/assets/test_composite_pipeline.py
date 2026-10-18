"""Parts survive the whole way: config -> layout -> manifest -> scene geometry -> labels.

Each piece of this was testable on its own and all of them passed while the chain was still broken.
`scene_geometry` derives an object's shape from the tail of its asset id (`comp_00042_mug` -> "mug")
and `primitive_for_kind` maps anything unlisted to "box" -- so before the manifest carried parts, a
mug reached the labeller as a cuboid. Every unit test stayed green; the data was silently wrong.

So this test walks the real chain and asserts the thing that actually matters at the end of it: that
grasps come back tagged with the part they touch.
"""

from __future__ import annotations

import collections
import unittest

from datagen.assets.library import MeshLibrary
from datagen.build import build_manifest
from datagen.config import AssetSourcesConfig, DatagenConfig
from datagen.grasps.labels import SceneAssets, label_jaw_grasps, scene_geometry
from datagen.scenes.layout import layout_scene_with_assets, reset_mesh_bank


def _settled_payload(spec, assets) -> dict:
    """A scene as the labeller sees it, with objects resting where layout placed them."""
    return {
        "spec": {
            "objects": [{"asset_id": a.asset_id, "mass_kg": a.mass_kg} for a in assets],
            "family": "sparse",
            "bin_walls": [],
        },
        "settled_poses_mm_xyzw": {
            str(i): [list(p.position_mm), list(p.orientation_xyzw)]
            for i, p in enumerate(spec.objects)
        },
        "dropped_objects": [],
    }


def _roles_over(scenes: int, **weights: float) -> collections.Counter[str]:
    reset_mesh_bank()
    config = DatagenConfig(scenes=scenes, assets=AssetSourcesConfig(**weights))
    manifest = build_manifest(config)
    extents = {r.asset_id: tuple(r.extent_mm) for r in manifest.records}
    parts = {r.asset_id: r.parts for r in manifest.records if r.parts}

    roles: collections.Counter[str] = collections.Counter()
    for index in range(scenes):
        spec, assets = layout_scene_with_assets(config, index, None)
        geometry = scene_geometry(
            _settled_payload(spec, assets), extents, f"s{index}", parts=parts
        )
        for instance in geometry.objects:
            labels, _ = label_jaw_grasps(geometry, instance)
            roles.update(label.part_role for label in labels)
    return roles


class PartsSurviveTheWholeChainTests(unittest.TestCase):
    def test_composite_scenes_produce_feature_labels(self) -> None:
        """The end-to-end claim: a dataset run yields grasps tagged with a non-body part."""
        roles = _roles_over(6, procedural_weight=1.0, composite_weight=2.0)
        feature = {role: n for role, n in roles.items() if role not in ("", "body")}
        self.assertTrue(
            feature,
            "no feature-tagged grasps survived the chain -- parts are not reaching the labeller",
        )

    def test_single_primitive_objects_still_report_no_part(self) -> None:
        """The other half of the same claim: nothing that was working changed."""
        roles = _roles_over(4, procedural_weight=1.0)
        self.assertEqual(set(roles), {""})

    def test_the_manifest_carries_the_part_structure(self) -> None:
        """The link that was missing. Without it every composite is labelled as a cuboid."""
        reset_mesh_bank()
        config = DatagenConfig(
            scenes=4, assets=AssetSourcesConfig(procedural_weight=1.0, composite_weight=2.0)
        )
        manifest = build_manifest(config)
        with_parts = [r for r in manifest.records if r.parts]
        self.assertTrue(with_parts, "no manifest record carries parts")
        for record in with_parts:
            with self.subTest(asset=record.asset_id):
                self.assertEqual(record.kind, "composite")
                self.assertTrue(all(p.primitive in ("box", "cylinder", "sphere") for p in record.parts))

    def test_composites_audit_clean(self) -> None:
        """They are our own geometry -- `own`, no attribution owed -- and the gate must agree.

        `audit()` RAISES rather than returning problems (`problems()` is the list form), so the
        assertion is that it does not raise.
        """
        reset_mesh_bank()
        config = DatagenConfig(
            scenes=4, assets=AssetSourcesConfig(procedural_weight=0.0, composite_weight=1.0)
        )
        manifest = build_manifest(config)
        self.assertEqual(manifest.problems(), [])
        manifest.audit()  # must not raise

    def test_the_default_dataset_contains_no_composites(self) -> None:
        """Default-off byte-identical: an existing run must generate exactly what it did before."""
        reset_mesh_bank()
        manifest = build_manifest(DatagenConfig(scenes=4))
        self.assertEqual([r for r in manifest.records if r.parts], [])


class TheGeometryMatchesWhatWouldBeRenderedTests(unittest.TestCase):
    """The coupling the whole design rests on, checked at the level where it can break silently."""

    def test_a_composite_becomes_several_solids_not_one_box(self) -> None:
        reset_mesh_bank()
        config = DatagenConfig(
            scenes=2, assets=AssetSourcesConfig(procedural_weight=0.0, composite_weight=1.0)
        )
        manifest = build_manifest(config)
        extents = {r.asset_id: tuple(r.extent_mm) for r in manifest.records}
        parts = {r.asset_id: r.parts for r in manifest.records if r.parts}

        spec, assets = layout_scene_with_assets(config, 0, None)
        geometry = scene_geometry(_settled_payload(spec, assets), extents, "s0", parts=parts)

        self.assertTrue(geometry.parts, "a composite scene produced no extra part solids")
        for instance in geometry.objects:
            with self.subTest(instance=instance):
                self.assertGreater(
                    len(geometry.parts_of(instance)), 1,
                    "a composite collapsed to a single solid -- it would be labelled as a cuboid",
                )

    def test_part_mass_is_split_not_repeated(self) -> None:
        """Each solid carries its share; giving every part the whole mass makes each too heavy."""
        from datagen.assets.composite import build_composite
        from datagen.grasps.shapes import solids_from_parts
        import numpy as np

        asset = build_composite("hammer", np.random.default_rng(3), index=0)
        solids = solids_from_parts(
            asset.parts, (450.0, 0.0, 100.0), (0.0, 0.0, 0.0, 1.0), mass_kg=1.0,
        )
        self.assertAlmostEqual(sum(s.mass_kg for s in solids), 1.0, places=6)
        for solid in solids:
            self.assertLess(solid.mass_kg, 1.0)



class TheManifestCarriesEveryFieldAnAssetHasTests(unittest.TestCase):
    """The defect that assembling a record field-by-field kept producing, now closed by construction.

    `build_manifest` used to build each `AssetRecord` by listing its fields at the call site, and it
    silently dropped whatever the last change forgot. THREE times, all found rather than guessed:

      * ``parts``       -- composites reached the labeller shapeless and were labelled as cuboids.
      * ``mesh_path``   -- the labeller could not find a scanned object's geometry.
      * ``attribution`` -- the licence audit REFUSED all 24 rows, so a mesh dataset could not be
                           generated at all. That one is not a data-quality issue, it is the gate
                           whose whole reason for existing is legal.

    Each was a line someone had to remember. `SceneAsset.as_record` makes the asset build its own
    record, so the next field arrives without anyone remembering, and mypy fails a type that forgets.
    """

    def test_a_composite_reaches_the_manifest_with_its_parts(self) -> None:
        reset_mesh_bank()
        manifest = build_manifest(DatagenConfig(
            scenes=4, assets=AssetSourcesConfig(procedural_weight=0.0, composite_weight=1.0)))
        self.assertTrue(manifest.records)
        for record in manifest.records:
            with self.subTest(asset=record.asset_id):
                self.assertTrue(record.parts, "a composite without parts is labelled as a cuboid")

    @unittest.skipUnless(MeshLibrary().is_ready("gso"), "no mesh library on this machine")
    def test_a_scanned_asset_reaches_the_manifest_with_its_file_and_its_attribution(self) -> None:
        reset_mesh_bank()
        manifest = build_manifest(DatagenConfig(
            scenes=3, assets=AssetSourcesConfig(procedural_weight=0.0, gso_weight=1.0)))
        self.assertTrue(manifest.records)
        for record in manifest.records:
            with self.subTest(asset=record.asset_id):
                self.assertTrue(record.mesh_path, "without the file the labeller cannot see the shape")
                self.assertTrue(record.attribution, "CC-BY without attribution fails the licence gate")

    @unittest.skipUnless(MeshLibrary().is_ready("gso"), "no mesh library on this machine")
    def test_a_scanned_manifest_passes_the_licence_audit(self) -> None:
        """The consequence, asserted as behaviour: `audit()` raises rather than returning problems."""
        reset_mesh_bank()
        manifest = build_manifest(DatagenConfig(
            scenes=3, assets=AssetSourcesConfig(procedural_weight=0.0, gso_weight=1.0)))
        self.assertEqual(manifest.problems(), [])
        manifest.audit()  # must not raise


class TheAssetBundleCannotBeHalfPassedTests(unittest.TestCase):
    """`SceneAssets` exists because three separate channels were three separate things to forget.

    `scene_geometry` grew a ``parts`` argument and NOT ONE production caller passed it -- the chain
    was assembled by hand in a test, which is why the test above stayed green while every composite in
    a real dataset would have been labelled as a cuboid. The bundle can be forgotten wholesale, loudly,
    but it cannot be half-passed.
    """

    def test_the_bundle_carries_parts_into_the_labels(self) -> None:
        reset_mesh_bank()
        config = DatagenConfig(
            scenes=6, assets=AssetSourcesConfig(procedural_weight=1.0, composite_weight=2.0))
        manifest = build_manifest(config)
        assets = SceneAssets(
            extents={r.asset_id: tuple(r.extent_mm) for r in manifest.records},
            parts={r.asset_id: tuple(r.parts) for r in manifest.records if r.parts},
            mesh_paths={r.asset_id: r.mesh_path for r in manifest.records if r.mesh_path},
        )

        roles: collections.Counter[str] = collections.Counter()
        for index in range(6):
            spec, drawn = layout_scene_with_assets(config, index, None)
            geometry = assets.geometry(_settled_payload(spec, drawn), f"s{index}")
            for instance in geometry.objects:
                labels, _ = label_jaw_grasps(geometry, instance)
                roles.update(label.part_role for label in labels)
        self.assertTrue({role for role in roles if role not in ("", "body")},
                        "the bundle did not carry parts through to the labels")

    def test_a_sourced_asset_without_its_file_is_REFUSED(self) -> None:
        """The remaining fail-closed guard, and it is the one that caught the missing `mesh_path`.

        `primitive_for_kind("mesh")` returns "box", so a scanned object with no file would be labelled
        as a cuboid it is not -- and the renderer, deriving its shape the same way, would AGREE.
        """
        payload = {
            "spec": {
                "objects": [{"asset_id": "gso_30_CONSTRUCTION_SET", "mass_kg": 0.2}],
                "family": "sparse", "bin_walls": [],
            },
            "settled_poses_mm_xyzw": {"0": [[450.0, 0.0, 40.0], [0.0, 0.0, 0.0, 1.0]]},
            "dropped_objects": [],
        }
        with self.assertRaises(NotImplementedError) as caught:
            scene_geometry(payload, {"gso_30_CONSTRUCTION_SET": (80.0, 80.0, 80.0)}, "s")
        message = str(caught.exception)
        self.assertIn("gso_30_CONSTRUCTION_SET", message)
        self.assertIn("box", message, "the refusal must say WHAT would go wrong, not just that it does")

    def test_a_procedural_asset_is_unaffected(self) -> None:
        """The refusal keys on the source prefix, so nothing that worked stops working."""
        payload = {
            "spec": {
                "objects": [{"asset_id": "proc_00000_cylinder", "mass_kg": 0.2}],
                "family": "sparse", "bin_walls": [],
            },
            "settled_poses_mm_xyzw": {"0": [[450.0, 0.0, 40.0], [0.0, 0.0, 0.0, 1.0]]},
            "dropped_objects": [],
        }
        geometry = scene_geometry(payload, {"proc_00000_cylinder": (60.0, 60.0, 80.0)}, "s")
        self.assertEqual(len(geometry.objects), 1)
if __name__ == "__main__":  # pragma: no cover
    unittest.main()
