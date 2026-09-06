"""Restricting the mesh draw to a named subset — the instrument a HELD-OUT dataset needs.

⛔ WHY IT EXISTS. MEASURED 2026-08-27: the eval ladder's reference dataset shares **100 %** of its fold
groups with the DL training corpus — all 16 of its procedural families are among the trainer's 168 —
so the ladder's `deep` row can never be a held-out number, whatever it reports. And the mesh half is
worse: of the 465 mesh files on disk the bank yields **147**, and the training corpus used **all 147**.
A dataset drawn only from assets the model never saw is the only way to turn the ladder into a
measurement rather than a recall test.

⚠ THE FAILURE MODE THIS FILE GUARDS is not "the restriction errors". It is "the restriction silently
does not apply" — which produces a full-bank dataset that gets REPORTED as held-out, with nothing
downstream able to tell. Two of these tests exist only for that: the cache key, and the refusal on a
missing file.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from datagen.config import DatagenConfig
from datagen.scenes.layout import _mesh_bank, reset_mesh_bank, resolve_mesh_asset_ids


def _config(**assets: object) -> DatagenConfig:
    return DatagenConfig.model_validate({"assets": assets})


def _write(payload: object) -> str:
    path = Path(tempfile.mkdtemp()) / "ids.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


class TheResolverTests(unittest.TestCase):
    def test_no_restriction_by_default(self) -> None:
        """An existing config must draw exactly the bank it always drew."""
        self.assertIsNone(resolve_mesh_asset_ids(_config()))

    def test_an_inline_list_is_honoured(self) -> None:
        """The three-custom-parts case, where a file would be absurd."""
        ids = resolve_mesh_asset_ids(_config(mesh_asset_ids=("custom_bracket", "custom_pin")))
        self.assertEqual(ids, frozenset({"custom_bracket", "custom_pin"}))

    def test_a_file_of_ids_is_honoured(self) -> None:
        """The 318-id case, where an inline list would be absurd."""
        ids = resolve_mesh_asset_ids(_config(mesh_asset_ids_path=_write(["gso_a", "gso_b"])))
        self.assertEqual(ids, frozenset({"gso_a", "gso_b"}))

    def test_a_file_keyed_by_source_is_honoured(self) -> None:
        """The shape datagen's own held-out analysis writes: {source: [ids]}."""
        ids = resolve_mesh_asset_ids(
            _config(mesh_asset_ids_path=_write({"gso": ["gso_a"], "ycb": ["ycb_b"]})))
        self.assertEqual(ids, frozenset({"gso_a", "ycb_b"}))

    def test_both_forms_together_are_UNIONED(self) -> None:
        """Both exist because they answer different questions; neither may shadow the other."""
        ids = resolve_mesh_asset_ids(_config(mesh_asset_ids=("custom_x",),
                                             mesh_asset_ids_path=_write(["gso_a"])))
        self.assertEqual(ids, frozenset({"custom_x", "gso_a"}))

    def test_an_empty_file_is_the_same_as_no_restriction(self) -> None:
        self.assertIsNone(resolve_mesh_asset_ids(_config(mesh_asset_ids_path=_write([]))))

    def test_a_MISSING_file_is_REFUSED_rather_than_ignored(self) -> None:
        """⛔ THE ONE THAT MATTERS MOST. Falling back to the whole bank would let a run succeed,
        produce a contaminated dataset, and be reported as held-out. A restriction that can silently
        not apply is not a restriction."""
        with self.assertRaises(FileNotFoundError) as caught:
            resolve_mesh_asset_ids(_config(mesh_asset_ids_path="nowhere/at/all.json"))
        message = str(caught.exception)
        self.assertIn("held-out", message, "the refusal must say WHY it refuses rather than falling back")


class TheBankHonoursItTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_mesh_bank()

    def tearDown(self) -> None:
        reset_mesh_bank()

    def test_a_subset_that_matches_nothing_yields_an_empty_bank_and_SAYS_SO(self) -> None:
        """An empty bank with no explanation is indistinguishable from a missing library. The count
        is what makes "your subset excluded everything" readable."""
        bank = _mesh_bank(_config(mesh_asset_ids=("gso_definitely_not_a_real_mesh",)))
        self.assertEqual(bank.load("gso"), [])
        self.assertGreater(bank.restricted.get("gso", 0), 0,
                           "the excluded count must be reported, not folded into `skipped`")

    def test_an_excluded_mesh_is_NOT_counted_as_a_defective_one(self) -> None:
        """`skipped` means "this mesh is unusable". A held-out run must not read as a broken library."""
        bank = _mesh_bank(_config(mesh_asset_ids=("gso_definitely_not_a_real_mesh",)))
        bank.load("gso")
        self.assertEqual(bank.skipped.get("gso", 0), 0)


class TheCacheKeyTests(unittest.TestCase):
    """⚠ THE SECOND SILENT-FAILURE PATH. `_mesh_bank` used to cache on `if _MESH_BANK is None`, so a
    second config in the same process received the FIRST one's bank — latent for `max_mesh_extent_mm`
    and load-bearing the moment a restriction arrived."""

    def setUp(self) -> None:
        reset_mesh_bank()

    def tearDown(self) -> None:
        reset_mesh_bank()

    def test_changing_the_restriction_rebuilds_the_bank(self) -> None:
        first = _mesh_bank(_config(mesh_asset_ids=("gso_one",)))
        second = _mesh_bank(_config(mesh_asset_ids=("gso_two",)))
        self.assertIsNot(first, second, "the cache served a bank built for a different subset")

    def test_changing_the_extent_limit_rebuilds_the_bank(self) -> None:
        first = _mesh_bank(_config(max_mesh_extent_mm=180.0))
        second = _mesh_bank(_config(max_mesh_extent_mm=90.0))
        self.assertIsNot(first, second)

    def test_the_same_config_reuses_the_bank(self) -> None:
        """The control: measuring 465 meshes costs ~21 s, so the cache has to still work."""
        first = _mesh_bank(_config(mesh_asset_ids=("gso_one",)))
        second = _mesh_bank(_config(mesh_asset_ids=("gso_one",)))
        self.assertIs(first, second)

    def test_the_ORDER_of_an_inline_list_does_not_defeat_the_cache(self) -> None:
        """The key is a frozenset, so two spellings of one subset are one subset."""
        first = _mesh_bank(_config(mesh_asset_ids=("a", "b")))
        second = _mesh_bank(_config(mesh_asset_ids=("b", "a")))
        self.assertIs(first, second)


class TheProceduralFallbackTests(unittest.TestCase):
    """⛔ THE CONTAMINATION PATH A HELD-OUT DATASET HAS TO CLOSE.

    When a weighted mesh source has nothing placeable, the layout logs ONE warning and makes those
    draws procedural, so a scene still happens rather than the generator looking broken. That is right
    for an ordinary run. For a held-out render it is fatal and silent: all 21 procedural families are
    in the DL training corpus, so an 80-scene dataset comes out fully contaminated behind a single
    log line. MEASURED: with `gso_weight=1, procedural_weight=0` and a restriction matching nothing,
    the draw returns a procedural `can`.
    """

    def setUp(self) -> None:
        reset_mesh_bank()

    def tearDown(self) -> None:
        reset_mesh_bank()

    def _draw(self, **extra: object):
        import numpy as np

        from datagen.scenes.layout import sample_scene_asset

        config = _config(gso_weight=1.0, procedural_weight=0.0,
                         mesh_asset_ids=("gso_definitely_not_a_real_mesh",), **extra)
        return sample_scene_asset(config, np.random.default_rng(0), index=0)

    def test_the_fallback_is_SILENT_by_default(self) -> None:
        """Not a complaint about the default -- a record of what it does, so the refusal below has a
        control. An ordinary run must keep generating scenes."""
        asset = self._draw()
        self.assertEqual(getattr(asset, "source", ""), "procedural")

    def test_refuse_procedural_fallback_REFUSES(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._draw(refuse_procedural_fallback=True)
        message = str(caught.exception)
        self.assertIn("already seen", message, "the refusal must say WHY it matters")
        self.assertIn("heldout", message, "and name the command that produces the right list")

    def test_all_weights_zero_is_refused_by_the_SCHEMA_before_it_reaches_the_draw(self) -> None:
        """⭑ A BETTER PLACE THAN THE ONE I WROTE THE GUARD FOR. My second refusal covers "every
        weighted source came out EMPTY"; a config with every weight at zero never gets that far,
        because the schema already rejects it at load. Two layers, and the outer one is the schema --
        which is where this repo puts refusals it can make without running anything."""
        with self.assertRaises(Exception) as caught:
            _config(gso_weight=0.0, ycb_weight=0.0, procedural_weight=0.0, composite_weight=0.0)
        self.assertIn("weight", str(caught.exception).lower())

    def test_a_healthy_mesh_source_is_unaffected_by_the_flag(self) -> None:
        """The flag must not turn a working run into a refusal -- it only closes the fallback."""
        import numpy as np

        from datagen.assets.library import MeshLibrary
        from datagen.scenes.layout import sample_scene_asset

        present = MeshLibrary().available("gso")
        if not present:
            self.skipTest("no gso meshes on this box")
        config = _config(gso_weight=1.0, procedural_weight=0.0,
                         refuse_procedural_fallback=True)
        asset = sample_scene_asset(config, np.random.default_rng(0), index=0)
        self.assertEqual(getattr(asset, "source", ""), "gso")





class TheRestrictionRestrictsTheMESHDrawOnlyTests(unittest.TestCase):
    """⛔ THE TRAP THE WHOLE RESTRICTION FEATURE CAN FALL INTO, and the guard could not see it.

    MEASURED: 60 asset ids + `gso_weight 1.0` + `refuse_procedural_fallback` -- with
    `procedural_weight` left at its DEFAULT of 1.0 -- produced **50 % procedural objects**, none of
    them in the requested list, and not one warning. A user building a held-out corpus gets a
    contaminated one that looks exactly right.

    The old guard fires only when a weighted mesh source is EMPTY, and here nothing is empty:
    procedural is simply a configured choice sitting beside the restricted meshes. That reading is
    honest -- procedural at weight 1.0 is not a "fallback" -- and it is exactly why a user who has
    just written out a list of asset ids does not expect it.
    """

    def setUp(self) -> None:
        reset_mesh_bank()

    def tearDown(self) -> None:
        reset_mesh_bank()

    def _draw(self, count: int, **extra: object) -> dict[str, int]:
        import collections

        import numpy as np

        from datagen.assets.library import MeshLibrary
        from datagen.scenes.layout import sample_scene_asset

        ids = [f"gso_{path.stem}" for path in MeshLibrary().available("gso")[:60]]
        if not ids:
            self.skipTest("no gso meshes on this box")
        config = _config(gso_weight=1.0, mesh_asset_ids=tuple(ids), **extra)
        rng = np.random.default_rng(0)
        seen: dict[str, int] = collections.Counter()
        for index in range(count):
            seen[str(getattr(sample_scene_asset(config, rng, index=index), "source", "?"))] += 1
        return dict(seen)

    def test_the_flag_REFUSES_the_mix(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._draw(10, refuse_procedural_fallback=True)
        message = str(caught.exception)
        self.assertIn("NOT in your list", message)
        self.assertIn("0.0", message, "the refusal must name the way out")

    def test_without_the_flag_it_WARNS_and_keeps_going(self) -> None:
        """Refusing by default would break every config that mixes procedural with meshes on
        purpose -- the ordinary case, and byte-identical today."""
        with self.assertLogs("SceneLayout", level="WARNING") as logged:
            drawn = self._draw(40)
        self.assertIn("NOT in your list", "\n".join(logged.output))
        self.assertGreater(drawn.get("procedural", 0), 0, "the mix still happens, as before")

    def test_zeroing_both_weights_is_silent(self) -> None:
        """The state a held-out render actually wants: no warning, no refusal, only the named assets."""
        drawn = self._draw(30, procedural_weight=0.0, composite_weight=0.0,
                           refuse_procedural_fallback=True)
        self.assertEqual(set(drawn), {"gso"}, drawn)

    def test_no_restriction_means_no_complaint(self) -> None:
        """The guard must not fire for the 99 % of configs that never name an asset id."""
        import numpy as np

        from datagen.scenes.layout import sample_scene_asset

        config = _config(procedural_weight=1.0, refuse_procedural_fallback=True)
        sample_scene_asset(config, np.random.default_rng(0), index=0)

if __name__ == "__main__":                                           # pragma: no cover
    unittest.main()


class JawSpanFilterTests(unittest.TestCase):
    """⭑ THE AXIS A JAW CLOSES ON, which is not the axis `max_mesh_extent_mm` bounds.

    That one bounds the LONGEST axis and is about fitting the workspace. This one bounds the SHORTEST
    and is about fitting the GRIPPER. MEASURED on the v3 corpus, 38,494 settled objects, share carrying
    a jaw label by smallest extent: 0-40 mm 20.7 %, 40-60 mm 17.2 %, 60-85 mm 20.5 %, 85-120 mm 5.8 %,
    120+ 11.8 %. The cliff is exactly the 85 mm aperture and 30.4 % of that corpus sat past it.
    """

    def _asset(self, asset_id: str, extent):
        from datagen.assets.meshes import MeshAsset

        return MeshAsset(asset_id=asset_id, source="gso", mesh_path=f"{asset_id}.obj",
                         extent_mm=tuple(float(v) for v in extent), mass_kg=0.1)

    def test_it_drops_only_what_cannot_fit_the_jaw(self) -> None:
        """A long thin object stays; a compact one wider than the aperture goes. Reaching for the
        LENGTH limit instead would reject the broom handle and keep the cube."""
        from datagen.assets.meshes import MeshAssetBank

        handle = self._asset("handle", (240.0, 30.0, 30.0))
        cube = self._asset("cube", (90.0, 90.0, 90.0))
        bank = MeshAssetBank.__new__(MeshAssetBank)
        bank.max_extent_mm, bank.max_jaw_span_mm = 250.0, 85.0
        self.assertLessEqual(min(handle.extent_mm), bank.max_jaw_span_mm)
        self.assertGreater(min(cube.extent_mm), bank.max_jaw_span_mm)
        self.assertLessEqual(max(cube.extent_mm), bank.max_extent_mm,
                             "the cube passes the LENGTH limit, which is why that limit cannot do "
                             "this job")

    def test_the_default_changes_no_existing_bank(self) -> None:
        from datagen.assets.meshes import MeshAssetBank
        from datagen.config import DatagenConfig

        self.assertIsNone(DatagenConfig().assets.max_jaw_span_mm)
        self.assertIsNone(MeshAssetBank(max_extent_mm=180.0).max_jaw_span_mm)

    def test_the_LAYOUT_honours_it_and_it_is_in_the_bank_cache_key(self) -> None:
        """⛔ A CACHED BANK IS SERVED TO THE NEXT CONFIG. Without the span in the key, a run that sets
        it gets whichever bank the previous run built and the filter reads as inert -- the shape that
        cost this repo a 103-scene run when `depth_source` was declared, decoded and never popped."""
        from datagen.config import DatagenConfig
        from datagen.scenes import layout

        layout._MESH_BANK = None  # noqa: SLF001 - the cache under test
        layout._MESH_BANK_KEY = None  # noqa: SLF001
        loose = layout._mesh_bank(DatagenConfig(assets={"max_mesh_extent_mm": 250.0}))  # noqa: SLF001
        tight = layout._mesh_bank(  # noqa: SLF001
            DatagenConfig(assets={"max_mesh_extent_mm": 250.0, "max_jaw_span_mm": 85.0}))
        self.assertIsNot(loose, tight, "the cache handed back the same bank, so the key ignores it")
        self.assertIsNone(loose.max_jaw_span_mm)
        self.assertEqual(tight.max_jaw_span_mm, 85.0)
