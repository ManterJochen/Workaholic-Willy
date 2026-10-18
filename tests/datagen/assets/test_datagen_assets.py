"""The asset side: shapes that are recognisably what they are called, and a licence gate that refuses.

Two properties, both easy to lose and expensive to discover late.

**Shape words have to mean something.** A referring expression says "the red cylinder", and the label
is only true if the mesh reads as a cylinder. A 'tube' as wide as it is long is a puck; a 'plate' as
thick as it is wide is a block. So proportions are constrained per kind, and asserted here — otherwise
the dataset teaches a model that shape words are noise.

**A licence problem must stop a run before it renders.** Every rendered image is a derivative work of
the meshes in it, so the audit is fail-closed and runs on the manifest, not on the output.
"""

from __future__ import annotations

import unittest

import numpy as np

from datagen.assets.licensing import audit_asset_rows, is_noncommercial
from datagen.assets.manifest import AssetManifest, AssetRecord, ManifestAuditError
from datagen.assets.procedural import (
    FAMILY_KINDS,
    MAX_JAW_APERTURE_MM,
    sample_procedural_asset,
)


def _asset(rng: np.random.Generator, families=("primitive",), index: int = 0):  # noqa: ANN001, ANN202
    return sample_procedural_asset(rng, families, index=index)


class ProceduralLibraryTests(unittest.TestCase):
    def test_the_same_generator_state_gives_the_same_asset(self) -> None:
        a = _asset(np.random.default_rng(4))
        b = _asset(np.random.default_rng(4))
        self.assertEqual(a, b)

    def test_every_family_is_drawable(self) -> None:
        rng = np.random.default_rng(0)
        seen = {
            _asset(rng, tuple(FAMILY_KINDS), index=i).family
            for i in range(400)
        }
        self.assertEqual(seen, set(FAMILY_KINDS))

    def test_an_unknown_family_is_refused_by_name(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _asset(np.random.default_rng(0), ("gadgets",))
        self.assertIn("gadgets", str(ctx.exception))

    def test_oversized_objects_exist_and_are_labelled_rather_than_filtered(self) -> None:
        """A carton too big for the jaw is a real object; calling it jaw-pickable would be the bug.

        So the library produces them, labels them, and this asserts BOTH halves: the label agrees with
        the geometry, and the majority of objects are still jaw-graspable (a dataset dominated by
        un-pickable objects would be a different dataset than the one anyone asked for).
        """
        rng = np.random.default_rng(1)
        assets = [_asset(rng, tuple(FAMILY_KINDS), index=i) for i in range(300)]
        for asset in assets:
            with self.subTest(kind=asset.kind, extent=asset.extent_mm):
                self.assertGreater(min(asset.extent_mm), 2.0)
                self.assertLessEqual(max(asset.extent_mm), 130.0)
                self.assertEqual(
                    asset.jaw_graspable, min(asset.extent_mm) <= MAX_JAW_APERTURE_MM,
                    "the jaw_graspable label must follow the geometry, not the other way round",
                )
        graspable = sum(1 for a in assets if a.jaw_graspable)
        self.assertGreater(graspable / len(assets), 0.6, "most objects should still be jaw-pickable")
        self.assertLess(graspable, len(assets), "the oversized case must actually occur")

    def test_shape_words_are_earned(self) -> None:
        """Elongated kinds are elongated; flat kinds are flat. Otherwise the label is a lie."""
        rng = np.random.default_rng(2)
        checked = set()
        for i in range(600):
            asset = _asset(rng, tuple(FAMILY_KINDS), index=i)
            x, y, z = asset.extent_mm
            if asset.kind in ("tube", "bottle", "capsule"):
                self.assertGreater(z, max(x, y) * 1.5, f"{asset.kind} is not elongated: {asset.extent_mm}")
            if asset.kind in ("plate", "blister"):
                self.assertLess(z, min(x, y) * 0.5, f"{asset.kind} is not flat: {asset.extent_mm}")
            if asset.kind == "sphere":
                self.assertEqual(len(set(asset.extent_mm)), 1, "a sphere is not axis-dependent")
            checked.add(asset.kind)
        self.assertGreaterEqual(len(checked), 12, "the sample missed most of the vocabulary")

    def test_mass_is_plausible_and_positive(self) -> None:
        rng = np.random.default_rng(3)
        for i in range(200):
            asset = _asset(rng, tuple(FAMILY_KINDS), index=i)
            with self.subTest(kind=asset.kind):
                self.assertGreater(asset.mass_kg, 0.0)
                self.assertLess(asset.mass_kg, 8.0, "a 120 mm part should not weigh more than the payload")

    def test_procedural_assets_carry_our_own_licence(self) -> None:
        asset = _asset(np.random.default_rng(0))
        self.assertEqual(asset.license, "own")
        self.assertEqual(audit_asset_rows([asset.as_manifest_row()]), [])


class LicenceGateTests(unittest.TestCase):
    def test_a_clean_manifest_audits(self) -> None:
        manifest = AssetManifest()
        manifest.add(AssetRecord("a", "procedural", "own", "box", (30.0, 30.0, 50.0), 0.1))
        manifest.add(AssetRecord(
            "b", "gso", "CC-BY-4.0", "mug", (80.0, 80.0, 90.0), 0.2,
            attribution="Google Scanned Objects — CC-BY 4.0",
        ))
        manifest.audit()

    def test_a_cc_by_asset_without_attribution_is_refused(self) -> None:
        """The rule most easily forgotten: CC-BY obliges naming the author in derivative works too."""
        manifest = AssetManifest()
        manifest.add(AssetRecord("b", "gso", "CC-BY-4.0", "mug", (80.0, 80.0, 90.0), 0.2))
        with self.assertRaises(ManifestAuditError) as ctx:
            manifest.audit()
        self.assertIn("requires attribution", str(ctx.exception))

    def test_a_non_commercial_asset_is_refused(self) -> None:
        manifest = AssetManifest()
        manifest.add(AssetRecord("c", "somewhere", "CC BY-NC-SA 4.0", "part", (30.0,) * 3, 0.1))
        with self.assertRaises(ManifestAuditError):
            manifest.audit()

    def test_an_unknown_licence_is_refused_rather_than_waved_through(self) -> None:
        manifest = AssetManifest()
        manifest.add(AssetRecord("d", "somewhere", "GPL-3.0", "part", (30.0,) * 3, 0.1))
        problems = manifest.problems()
        self.assertTrue(any("not CC0 / CC-BY / own" in p for p in problems), problems)

    def test_the_audit_reports_every_problem_at_once(self) -> None:
        """One run of the gate, one fix list -- not five runs discovering one bad row each."""
        manifest = AssetManifest()
        manifest.add(AssetRecord("a", "somewhere", "", "part", (30.0,) * 3, 0.1))
        manifest.add(AssetRecord("b", "somewhere", "CC BY-NC 4.0", "part", (30.0,) * 3, 0.1))
        manifest.add(AssetRecord("c", "somewhere", "GPL-3.0", "part", (30.0,) * 3, 0.1))
        self.assertEqual(len(manifest.problems()), 3)

    def test_noncommercial_detection(self) -> None:
        for value in ("CC BY-NC-SA 4.0", "cc-by-nc", "Non-Commercial"):
            self.assertTrue(is_noncommercial(value), value)
        for value in ("CC0-1.0", "CC-BY-4.0", "own", "Apache-2.0"):
            self.assertFalse(is_noncommercial(value), value)

    def test_a_missing_asset_says_what_is_missing(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            AssetManifest().get("nope")
        self.assertIn("nope", str(ctx.exception))


class AttributionFileTests(unittest.TestCase):
    def test_the_attribution_file_names_every_asset_and_its_licence(self) -> None:
        manifest = AssetManifest()
        manifest.add(AssetRecord("a", "procedural", "own", "box", (30.0,) * 3, 0.1))
        manifest.add(AssetRecord(
            "b", "gso", "CC-BY-4.0", "mug", (80.0,) * 3, 0.2, attribution="Google — CC-BY 4.0",
        ))
        text = manifest.attribution_text(dataset_name="v1_probe")
        self.assertIn("v1_probe", text)
        self.assertIn("CC-BY-4.0", text)
        self.assertIn("Google — CC-BY 4.0", text)
        self.assertIn("own", text)
        self.assertIn("derivative work", text, "the file must say WHY it exists")


if __name__ == "__main__":
    unittest.main()
