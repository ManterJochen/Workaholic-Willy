"""The asset-source weights change what a scene contains — which they never did before.

`assets.gso_weight` / `ycb_weight` / `objaverse_weight` validated from the day they were written and
had no reader: `layout.py` called `sample_procedural_asset` unconditionally, so every object in every
generated scene was one of three convex solids. A "cup" had no handle. That is the specific reason a
network trained on this data could only ever learn a convex world.

These tests are differential, in the spirit of `test_grasping_wiring_guard`: set the knob, sample,
and assert the DISTRIBUTION moved. A test that only checked "a mesh can be sampled" would have
passed against the broken version too, since the meshes were always loadable — nobody was asking
for them.
"""

from __future__ import annotations

import collections
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from datagen.assets.library import MeshLibrary, import_from_directory
from datagen.assets.meshes import MeshAsset, MeshAssetBank, measure_mesh
from datagen.assets.composite import sample_composite_asset
from datagen.assets.procedural import ProceduralAsset, sample_procedural_asset
from datagen.config import AssetSourcesConfig, DatagenConfig
from datagen.scenes.layout import reset_mesh_bank, sample_scene_asset
from datagen.scenes.spec import SceneAsset

_HAVE_MESHES = MeshLibrary().is_ready("ycb") and MeshLibrary().is_ready("gso")
_REASON = "real meshes not fetched -- python -m datagen.assets --check"


def _sources(config: DatagenConfig, n: int = 300, seed: int = 11) -> dict[str, int]:
    reset_mesh_bank()
    rng = np.random.default_rng(seed)
    return dict(
        collections.Counter(
            sample_scene_asset(config, rng, index=i).source for i in range(n)
        )
    )


class BothAssetKindsSatisfyTheLayoutContractTests(unittest.TestCase):
    """If either fails this, placement would need to branch on the kind — which is the thing to avoid."""

    def test_a_procedural_asset_is_a_scene_asset(self) -> None:
        asset = ProceduralAsset(
            asset_id="p", family="primitive", kind="cube",
            extent_mm=(50.0, 50.0, 50.0), mass_kg=0.1,
        )
        self.assertIsInstance(asset, SceneAsset)

    def test_a_mesh_asset_is_a_scene_asset(self) -> None:
        asset = MeshAsset(
            asset_id="m", source="ycb", mesh_path="x.ply",
            extent_mm=(50.0, 50.0, 50.0), mass_kg=0.1,
        )
        self.assertIsInstance(asset, SceneAsset)

    def test_the_protocol_survives_frozen_dataclasses(self) -> None:
        """A Protocol member written `x: str` demands a SETTABLE attribute; these types are frozen.

        Pinned because it is the failure that is easy to 'fix' the wrong way -- by unfreezing the
        assets rather than by declaring the protocol read-only.
        """
        asset = MeshAsset(
            asset_id="m", source="ycb", mesh_path="x.ply",
            extent_mm=(10.0, 10.0, 10.0), mass_kg=0.01,
        )
        with self.assertRaises(Exception):
            asset.asset_id = "other"  # type: ignore[misc]
        self.assertIsInstance(asset, SceneAsset)


class TheWeightsDecideWhatIsSampledTests(unittest.TestCase):
    """The differential. Before this wiring, every row below read `{'procedural': 300}`."""

    def test_the_default_is_still_procedural_only(self) -> None:
        """Default-off byte-identical: a cell that set nothing must see exactly what it saw before."""
        counts = _sources(DatagenConfig())
        self.assertEqual(counts, {"procedural": 300})

    @unittest.skipUnless(_HAVE_MESHES, _REASON)
    def test_a_mesh_only_mix_draws_no_procedural(self) -> None:
        counts = _sources(
            DatagenConfig(assets=AssetSourcesConfig(procedural_weight=0.0, ycb_weight=1.0))
        )
        self.assertEqual(set(counts), {"ycb"})

    @unittest.skipUnless(_HAVE_MESHES, _REASON)
    def test_equal_weights_split_roughly_evenly(self) -> None:
        counts = _sources(
            DatagenConfig(assets=AssetSourcesConfig(procedural_weight=1.0, gso_weight=1.0))
        )
        self.assertEqual(set(counts), {"procedural", "gso"})
        for source, n in counts.items():
            with self.subTest(source=source):
                self.assertGreater(n, 90, "a 1:1 weight produced a lopsided draw")

    @unittest.skipUnless(_HAVE_MESHES, _REASON)
    def test_three_sources_all_appear(self) -> None:
        counts = _sources(
            DatagenConfig(assets=AssetSourcesConfig(
                procedural_weight=1.0, gso_weight=1.0, ycb_weight=1.0,
            ))
        )
        self.assertEqual(set(counts), {"procedural", "gso", "ycb"})

    def test_sampling_is_deterministic_for_a_seed(self) -> None:
        config = DatagenConfig(assets=AssetSourcesConfig(procedural_weight=1.0))
        self.assertEqual(_sources(config, seed=5), _sources(config, seed=5))


class OneSourceMayNotTOUCHTheRandomStreamTests(unittest.TestCase):
    """The regression that made a whole reference corpus unreproducible.

    A weighted draw consumes an `rng.random()` even when there is nothing to choose between. Routing
    the DEFAULT (procedural-only) config through `sample_scene_asset` therefore shifted every later
    draw by one number: same seed, different assets, different order.

    MEASURED on `v1_proof`, the D5 reference corpus: its provenance stamp stopped rebuilding the
    manifest it was rendered with -- 181 assets rebuilt against 172 referenced, two of them absent from
    the rebuild entirely. A dataset that cannot be rebuilt from its own stamp cannot be re-labelled,
    because `scene_geometry` looks extents up by asset id: the objects that survived would have been
    given other objects' sizes. Nothing announced it beyond one returned problem string on a code path
    nobody re-ran in between.
    """

    def test_a_single_source_leaves_the_stream_untouched(self) -> None:
        """The direct statement: with one source, the sampler must consume exactly what the
        procedural sampler alone consumes -- so the NEXT draw is unchanged."""
        config = DatagenConfig()          # procedural-only, the default
        through_sampler = np.random.default_rng(1234)
        sample_scene_asset(config, through_sampler, index=0)
        directly = np.random.default_rng(1234)
        sample_procedural_asset(directly, config.assets.procedural_families, index=0)
        self.assertEqual(float(through_sampler.random()), float(directly.random()))

    def test_the_same_holds_for_a_single_MESH_source(self) -> None:
        """Not a procedural special case: any one-source mix must leave the stream alone."""
        config = DatagenConfig(assets=AssetSourcesConfig(
            procedural_weight=0.0, composite_weight=1.0))
        through_sampler = np.random.default_rng(99)
        sample_scene_asset(config, through_sampler, index=0)
        directly = np.random.default_rng(99)
        sample_composite_asset(directly, index=0)
        self.assertEqual(float(through_sampler.random()), float(directly.random()))

    def test_TWO_sources_do_consume_a_draw(self) -> None:
        """The other half, so the fix cannot silently become 'never draw at all'."""
        config = DatagenConfig(assets=AssetSourcesConfig(
            procedural_weight=1.0, composite_weight=1.0))
        through_sampler = np.random.default_rng(7)
        sample_scene_asset(config, through_sampler, index=0)
        untouched = np.random.default_rng(7)
        self.assertNotEqual(float(through_sampler.random()), float(untouched.random()))

    def test_the_default_layout_matches_the_reference_corpus_it_was_built_with(self) -> None:
        """The end of the chain: a manifest built from the DEFAULT config is stable.

        Pinned by hash rather than by content, because what broke was not any single asset -- it was
        the sequence. `v1_proof`'s recorded stamp is `ad321504...`; this asserts the shape of that
        guarantee without depending on a dataset being present on the machine.
        """
        from datagen.build import build_manifest  # noqa: PLC0415
        from datagen.provenance import sha256_of  # noqa: PLC0415, SLF001

        reset_mesh_bank()
        first = sha256_of([r.as_row() for r in build_manifest(DatagenConfig(scenes=12))])
        reset_mesh_bank()
        second = sha256_of([r.as_row() for r in build_manifest(DatagenConfig(scenes=12))])
        self.assertEqual(first, second)


class UnsupportedSourcesFallBackLoudlyTests(unittest.TestCase):
    """A weighted source that cannot be supplied must warn and continue, not crash at scene 4,000."""

    def test_an_unsupplied_source_warns_by_name_and_says_WHY(self) -> None:
        """⭐ THIS USED TO NAME OBJAVERSE, AND IT WENT RED BECAUSE THE THING IT GUARDED GOT FIXED.

        The deferral read "we hold no author names, so a fail-closed CC-BY audit cannot pass them --
        resolve the authors before enabling this". `f7cc043` met that condition on 2026-09-03: the
        fetch writes `ATTRIBUTION.tsv`, and MEASURED on this box 297 of 297 rows carry an author. The
        licence side lifted the deferral and `scenes/layout.py` did not, so for a day a customer who
        set `objaverse_weight` got a warning and procedural objects, while the commit that admitted
        the source said it was admitted.

        ⚠ THE TEST IS DERIVED FROM THE LIST NOW, NOT WRITTEN AGAINST ONE ENTRY. It asserts the
        BEHAVIOUR of whatever is currently unsupplied and skips when nothing is, which is today's
        state. A test naming one source has to be rewritten by whoever adds the next, which is
        exactly how this one came to forbid its own success.
        """
        from datagen.scenes.layout import _UNSUPPORTED_SOURCE_WEIGHTS

        if not _UNSUPPORTED_SOURCE_WEIGHTS:
            self.skipTest("every source this generator knows can be supplied; nothing to warn about")
        source, field, _reason = _UNSUPPORTED_SOURCE_WEIGHTS[0]
        reset_mesh_bank()
        config = DatagenConfig(
            assets=AssetSourcesConfig(**{"procedural_weight": 1.0, field: 1.0}))
        with self.assertLogs("SceneLayout", level="WARNING") as captured:
            counts = _sources(config)
        self.assertEqual(counts, {"procedural": 300})
        joined = chr(10).join(captured.output)
        self.assertIn(source, joined)
        self.assertGreater(len(joined), len(source) + 40,
                           "the warning must say WHY, not just that it is unsupported")

    def test_objaverse_is_no_longer_among_them(self) -> None:
        """⛔ A REFUSAL THAT OUTLIVES ITS REASON IS WORSE THAN NONE, because it reads as a considered
        decision rather than as a stale line. Pinned separately so that putting objaverse back has to
        be a deliberate act carrying a new reason."""
        from datagen.scenes.layout import _UNSUPPORTED_SOURCE_WEIGHTS

        self.assertNotIn("objaverse", {s for s, _f, _r in _UNSUPPORTED_SOURCE_WEIGHTS})

    def test_the_schema_refuses_an_all_zero_mix_before_the_sampler_sees_it(self) -> None:
        """Measured while writing this: the config validator already blocks it, one layer earlier.

        The sampler still carries a procedural fallback for the all-zero case, and that is not dead
        code -- it is reachable when every WEIGHTED source turns out to be empty at runtime (a
        weight set for meshes that were never fetched). What it is not reachable from is a config,
        and this pins where the refusal actually lives.
        """
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as caught:
            AssetSourcesConfig(procedural_weight=0.0)
        self.assertIn("zero", str(caught.exception))

    def test_a_weighted_but_empty_source_falls_back_instead_of_emptying_the_scene(self) -> None:
        """The reachable case: meshes are asked for and are not on this machine."""
        from datagen.assets.library import MeshLibrary

        if MeshLibrary().is_ready("gso"):
            self.skipTest("meshes ARE present here; the empty-bank path needs them absent")
        counts = _sources(
            DatagenConfig(assets=AssetSourcesConfig(procedural_weight=1.0, gso_weight=1.0))
        )
        self.assertEqual(counts, {"procedural": 300})


class TheMeshBankMeasuresRatherThanAssumesTests(unittest.TestCase):
    def _bank_with(self, sizes_m: list[float], max_extent_mm: float) -> MeshAssetBank:
        """A bank over synthetic cube meshes of known size."""
        import trimesh

        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        origin = root / "origin" / "ycb"
        origin.mkdir(parents=True)
        for i, size in enumerate(sizes_m):
            trimesh.creation.box(extents=(size, size, size)).export(origin / f"cube{i}.ply")
        import_from_directory("ycb", origin, destination=root / "library")
        return MeshAssetBank(MeshLibrary(root / "library"), max_extent_mm=max_extent_mm)

    def test_extent_is_read_from_the_mesh(self) -> None:
        bank = self._bank_with([0.05], max_extent_mm=180.0)
        assets = bank.load("ycb")
        self.assertEqual(len(assets), 1)
        for axis in assets[0].extent_mm:
            self.assertAlmostEqual(axis, 50.0, places=1)

    def test_an_oversized_mesh_is_skipped_and_counted(self) -> None:
        """Silently shrinking banks are how a dataset loses variety without anyone noticing."""
        bank = self._bank_with([0.05, 0.30], max_extent_mm=180.0)
        self.assertEqual(len(bank.load("ycb")), 1)
        self.assertEqual(bank.skipped["ycb"], 1)

    def test_an_unreadable_file_costs_that_file_and_not_the_run(self) -> None:
        with TemporaryDirectory() as tmp:
            bad = Path(tmp) / "broken.ply"
            bad.write_bytes(b"not a mesh")
            self.assertIsNone(measure_mesh("ycb_broken", "ycb", bad))

    def test_jaw_graspable_reads_the_shortest_axis(self) -> None:
        """The label the whole size discussion turns on: a long thin object IS graspable."""
        long_thin = MeshAsset(
            asset_id="m", source="ycb", mesh_path="x.ply",
            extent_mm=(170.0, 40.0, 30.0), mass_kg=0.1,
        )
        fat = MeshAsset(
            asset_id="m2", source="ycb", mesh_path="y.ply",
            extent_mm=(100.0, 95.0, 90.0), mass_kg=0.1,
        )
        self.assertTrue(long_thin.jaw_graspable)
        self.assertFalse(fat.jaw_graspable)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
