"""Restricting the COMPOSITE draw to a named subset of families.

⭑ WHY IT EXISTS. Composites are the only objects in this corpus with named parts, so they are the
only source a handle-grasp number can come from. MEASURED 2026-08-31 on a 25-scene slice: the
analytic stack's matched candidates are **27 body, 0 handle, 0 grip**, against labels on those same
scenes carrying 36 handle and 49 grip -- so "does the learned head grasp a handle" is a question with
a zero baseline and no held-out instrument yet. `logs/dl/heldout2` is GSO-only and contains no
composite at all.

Two DIFFERENT held-out questions live here, and only one of them needs this field:

  * new INSTANCES of a trained family -- a mug whose dimensions were never seen. A disjoint seed
    gives that, and the family list stays full. Weaker claim, available on a model already trained.
  * a family whose PART LAYOUT was never trained on -- drop "hammer" from a corpus, render an eval
    corpus of only "hammer". Stronger claim, and it has to be decided BEFORE the training render.

⚠ THE FAILURE MODE GUARDED HERE is not "the restriction errors". It is the shape this repo has been
bitten by twice: a knob that is read, stamped into provenance, and then not applied -- `calculator:
deep` was inert for an entire arc because every construction site named the analytic class directly,
and the wiring guard could not see it because it enumerates BOOLEAN flags. A reserved family that
still appears in training turns the eval corpus into a recall test that gets REPORTED as held-out.
"""

from __future__ import annotations

import unittest

import numpy as np

from datagen.assets.composite import COMPOSITE_KINDS
from datagen.config import DatagenConfig
from datagen.scenes.layout import sample_scene_asset


def _composites_only(**assets: object) -> DatagenConfig:
    """Weights that make every draw a composite, so the kind is the only free variable."""
    return DatagenConfig.model_validate(
        {"assets": {"procedural_weight": 0.0, "gso_weight": 0.0, "objaverse_weight": 0.0,
                    "ycb_weight": 0.0, "composite_weight": 1.0, **assets}})


def _drawn_sequence(config: DatagenConfig, *, n: int = 60, seed: int = 0) -> list[str]:
    """The ORDER of the draw, not the set. A set cannot see the defect this field shipped with."""
    rng = np.random.default_rng(seed)
    return [str(sample_scene_asset(config, rng, index=i).composite_kind) for i in range(n)]


def _drawn_kinds(config: DatagenConfig, *, n: int = 200, seed: int = 0) -> set[str]:
    rng = np.random.default_rng(seed)
    return {str(sample_scene_asset(config, rng, index=i).composite_kind) for i in range(n)}


class TheDefaultTests(unittest.TestCase):
    def test_every_family_is_drawable_by_default(self) -> None:
        self.assertEqual(set(DatagenConfig().assets.composite_kinds), set(COMPOSITE_KINDS))

    def test_the_default_ORDER_is_the_one_the_sampler_used_before_this_field(self) -> None:
        """⛔ THE DEFECT THIS FIELD SHIPPED WITH, and the test above is why it got through.

        `sample_composite_asset` draws with `rng.choice(list(available))`, which selects by INDEX. So
        the ORDER decides which family lands in each composite slot, not just which families are
        allowed. Before this field existed the sampler used `tuple(sorted(_BUILDERS))`; the field
        first shipped in the dict's insertion order.

        A provenance stamp written before the field has no such key, so `DatagenConfig(**stamp)` takes
        this default -- and rebuilt every composite in the existing corpora as a different object of a
        different family. MEASURED: the two orders produce different manifest hashes for the same
        recorded config, and shared composite ids come back with a median 18.4 % worst-axis extent
        error.

        ⚠ THE TEST ABOVE COMPARES SETS AND PASSED THE WHOLE TIME. A set comparison cannot see an
        ordering defect, which is exactly the shape that got a whole corpus re-rendered earlier in
        this arc when a raster numbered instances from 1 and Isaac from 0.
        """
        self.assertEqual(DatagenConfig().assets.composite_kinds, tuple(sorted(COMPOSITE_KINDS)))

    def test_a_config_WITHOUT_the_key_draws_what_it_always_drew(self) -> None:
        """The property that makes an old provenance stamp still mean what it says."""
        without = _drawn_sequence(DatagenConfig.model_validate(
            {"assets": {"procedural_weight": 0.0, "gso_weight": 0.0, "objaverse_weight": 0.0,
                        "ycb_weight": 0.0, "composite_weight": 1.0}}))
        explicit = _drawn_sequence(_composites_only(composite_kinds=tuple(sorted(COMPOSITE_KINDS))))
        self.assertEqual(without, explicit)

    def test_the_default_really_draws_all_five(self) -> None:
        """A default that LISTS five while the sampler yields four would be the same silent lie."""
        self.assertEqual(_drawn_kinds(_composites_only()), set(COMPOSITE_KINDS))


class TheRestrictionTests(unittest.TestCase):
    def test_a_reserved_family_never_appears(self) -> None:
        """⛔ THE ONE THAT MATTERS: 200 draws, and the reserved family is absent from all of them."""
        drawn = _drawn_kinds(_composites_only(composite_kinds=("mug", "jug", "pan", "bucket")))
        self.assertNotIn("hammer", drawn)
        self.assertEqual(drawn, {"mug", "jug", "pan", "bucket"})

    def test_the_complement_yields_only_the_reserved_family(self) -> None:
        """The eval half of the same split -- both halves have to be constructible."""
        self.assertEqual(_drawn_kinds(_composites_only(composite_kinds=("hammer",))), {"hammer"})

    def test_the_two_halves_do_not_overlap(self) -> None:
        """Stated as a property rather than by eye, because THAT is what held-out means."""
        train = _drawn_kinds(_composites_only(composite_kinds=("mug", "jug", "pan", "bucket")))
        evaluation = _drawn_kinds(_composites_only(composite_kinds=("hammer",)))
        self.assertEqual(train & evaluation, set())

    def test_a_reserved_family_still_carries_its_parts(self) -> None:
        """A split is worthless if the eval half lost the part roles it was reserved FOR."""
        rng = np.random.default_rng(3)
        asset = sample_scene_asset(_composites_only(composite_kinds=("hammer",)), rng, index=0)
        self.assertEqual({part.role for part in asset.parts}, set(COMPOSITE_KINDS["hammer"]))


class TheRefusalTests(unittest.TestCase):
    def test_it_refuses_an_empty_list_it_would_have_to_draw_from(self) -> None:
        """Mirrors `procedural_families`: asking for composites with no family is unanswerable."""
        with self.assertRaises(ValueError) as caught:
            _composites_only(composite_kinds=())
        self.assertIn("composite_kinds", str(caught.exception))

    def test_an_empty_list_is_fine_when_nothing_draws_composites(self) -> None:
        DatagenConfig.model_validate({"assets": {"composite_weight": 0.0, "composite_kinds": ()}})

    def test_an_unknown_family_is_refused_by_the_schema(self) -> None:
        with self.assertRaises(ValueError):
            _composites_only(composite_kinds=("teapot",))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class TheEvaluationRefusesADriftedManifestTest(unittest.TestCase):
    """⛔ NOTHING CHECKED, AND AN 832-SCENE RUN REPORTED NUMBERS JUDGED AGAINST THE WRONG SOLIDS.

    An evaluation rebuilds each object's SOLID from the manifest and grades every candidate against
    it, so a manifest that drifted grades proposals against geometry the images do not show. The label
    path already rebuilds and compares (`prompts/build.py:70`) but returns the problem as text and
    writes anyway; the evaluation path never rebuilt at all.

    MEASURED 2026-08-31: `composite_kinds` shipping in the dict's insertion order rather than sorted
    made `v3_s0`'s rebuilt manifest hash to 49bf2ba8 over 404 rows against the recorded 9b27dbab over
    402. The sorted order reproduces the stamp bit-exactly. Neither the run nor the report noticed.
    """

    def _dataset(self, mutate):
        import json
        import tempfile
        from pathlib import Path

        root = Path(tempfile.mkdtemp())
        (root / "scenes").mkdir()
        stamp = {"asset_manifest_sha256": "0" * 64,
                 "config": DatagenConfig().model_dump(mode="json")}
        mutate(stamp)
        (root / "provenance.json").write_text(json.dumps(stamp), encoding="utf-8")
        return root

    def test_a_mismatched_hash_REFUSES_and_the_message_names_both_causes(self) -> None:
        from datagen.eval.ladder import _refuse_a_drifted_manifest

        root = self._dataset(lambda s: None)          # a stamp whose hash cannot possibly match
        with self.assertRaises(ValueError) as caught:
            _refuse_a_drifted_manifest(root)
        message = str(caught.exception)
        self.assertIn("Refusing", message)
        self.assertIn("re-stamp", message)
        self.assertIn("re-render", message)

    def test_a_matching_hash_passes(self) -> None:
        """It must not refuse a dataset that is fine, or it is a blocker rather than a guard."""
        from datagen.build import build_manifest
        from datagen.eval.ladder import _refuse_a_drifted_manifest
        from datagen.provenance import sha256_of

        def stamp_the_truth(stamp: dict) -> None:
            rows = [r.as_row() for r in build_manifest(DatagenConfig(**stamp["config"]))]
            stamp["asset_manifest_sha256"] = sha256_of(rows)

        _refuse_a_drifted_manifest(self._dataset(stamp_the_truth))

    def test_a_directory_with_no_provenance_is_left_alone(self) -> None:
        """Hand-assembled slices exist and have nothing to check against."""
        import tempfile
        from pathlib import Path

        from datagen.eval.ladder import _refuse_a_drifted_manifest

        _refuse_a_drifted_manifest(Path(tempfile.mkdtemp()))


class TheCompositeIdNamesASlotTests(unittest.TestCase):
    """A composite's id names its POSITION in the scene, not its shape, and that has a measured cost.

    ⛔ MEASURED on `logs/dl/v3_s0`: 406 composite placements resolve to **37 distinct asset ids**, and
    `comp_00000_mug` alone appears 23 times with **23 DISTINCT masses**. Mass is a pure function of the
    parts, so those are 23 different shapes wearing one name. `build_manifest` dedups first-wins and
    both the renderer and the labeller resolve geometry through that single record, so 369 of the 406
    drawn composites are discarded: the scene was settled against a different object than the layout
    drew (spec mass over rendered mass p95 1.709, max 2.429) and 78.8 % of placements were spaced for
    an AABB more than 10 % off.

    ⭐ AND IT CAPS DIVERSITY, which explains something the arc could not: the slot index runs 0..8, so
    a corpus holds at most (objects per scene) x (kinds) distinct composites HOWEVER MANY SCENES ARE
    RENDERED. Ten thousand fresh scenes bought no new composite shapes.

    ⚠ THE REPAIR IS DEFAULT OFF ON PURPOSE. It changes every composite id and therefore the manifest
    hash, so a corpus rendered under one setting cannot be rebuilt from its own provenance under the
    other, and `evaluate_dataset` now refuses exactly that. It is a choice for a NEW render.
    """

    def test_the_default_is_the_slot_and_collides(self) -> None:
        """Pins the behaviour every existing corpus was built under, collision included."""
        import numpy as np

        from datagen.assets.composite import build_composite

        ids = {build_composite("mug", np.random.default_rng(seed), index=0).asset_id
               for seed in range(20)}
        self.assertEqual(len(ids), 1, "20 different mugs in slot 0 must collide under the default")

    def test_unique_ids_separate_different_shapes(self) -> None:
        import numpy as np

        from datagen.assets.composite import build_composite

        ids = {build_composite("mug", np.random.default_rng(seed), index=0, unique_id=True).asset_id
               for seed in range(20)}
        self.assertEqual(len(ids), 20)

    def test_the_same_shape_keeps_the_same_id(self) -> None:
        """Content-based, not a counter: draw order must not rename an object, or a fold split keyed
        on the id would separate an object from itself across two shards."""
        import numpy as np

        from datagen.assets.composite import build_composite

        first = build_composite("pan", np.random.default_rng(7), index=3, unique_id=True)
        again = build_composite("pan", np.random.default_rng(7), index=3, unique_id=True)
        self.assertEqual(first.asset_id, again.asset_id)

    def test_a_unique_id_still_groups_by_FAMILY_for_folds(self) -> None:
        r"""⛔ THE ONE THAT WOULD BREAK SILENTLY. `asset_group` maps a generated id to its family with
        `^(proc|comp)_\d+_(?P<family>.+)$`. An id shape that stops matching would make every composite
        its own fold group, and a held-out split would then leak."""
        import numpy as np

        from src.robot.grasping.deep.corpus.index import asset_group
        from datagen.assets.composite import build_composite

        for kind in ("mug", "hammer"):
            asset = build_composite(kind, np.random.default_rng(1), index=2, unique_id=True)
            self.assertEqual(asset_group(asset.asset_id), f"family:{kind}",
                             f"{asset.asset_id} no longer groups by family")

    def test_the_config_default_keeps_every_existing_corpus_rebuildable(self) -> None:
        self.assertEqual(DatagenConfig().assets.composite_id, "slot")
