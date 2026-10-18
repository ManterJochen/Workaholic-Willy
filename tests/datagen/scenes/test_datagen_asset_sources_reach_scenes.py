"""A source you can download, normalise and screen, and never place, is a feature that is dead.

⛔⛔ **THREE SOURCES COULD NOT REACH A SCENE, EACH FOR ITS OWN REASON, AND NONE OF THEM SAID SO
CLEARLY.** `thingi10k` and `asos` are in `SUPPORTED_SOURCES`, `datagen.assets.fetch` downloads them,
the mesh library reads them and `screen-meshes` grades them, and **no weight field existed**, so
`scenes/layout.py` had nothing to draw them with. `objaverse` had a weight and was refused here by
name, under a reason that had been met a day earlier.

⛔⛔ **AND THE VALIDATOR REFUSED THE CONFIGURATION THIS PACKAGE EXISTS FOR.**
`AssetSourcesConfig._something_to_draw_from` summed the weights BY HAND and left out
`custom_weight`. A customer whose only source is their own parts was refused with the sentence
"every asset-source weight is zero" while one of them was 1.0. That refusal is not merely wrong, it
is unarguable: it states a fact about the config that the config contradicts, so the only way past it
is to read the validator.
"""

from __future__ import annotations

import unittest

from datagen.config import AssetSourcesConfig, DatagenConfig


class EverySourceWithAWeightCanCarryAConfigAloneTests(unittest.TestCase):
    """The property, derived rather than listed, so a source added tomorrow is covered too."""

    @staticmethod
    def _weights() -> list[str]:
        return [n for n in AssetSourcesConfig.model_fields if n.endswith("_weight")]

    def test_there_is_more_than_one_weight_to_check(self) -> None:
        """The control on the loop below: over an empty list it would pass trivially."""
        self.assertGreater(len(self._weights()), 4, self._weights())

    def test_each_one_alone_is_a_legal_config(self) -> None:
        refused = []
        for name in self._weights():
            try:
                DatagenConfig(assets={"procedural_weight": 0.0, name: 1.0})
            except Exception:                                  # noqa: BLE001 - any refusal counts
                refused.append(name)
        self.assertEqual([], refused,
                         "a source with a weight of its own cannot carry a config alone")

    def test_custom_alone_is_legal_and_it_is_the_point(self) -> None:
        """⭑ NAMED SEPARATELY BECAUSE IT IS THE CONFIGURATION THIS PACKAGE EXISTS FOR. A bin-picking
        cell runs on the parts THAT CELL handles, and no public dataset contains them."""
        config = DatagenConfig(assets={"procedural_weight": 0.0, "custom_weight": 1.0})
        self.assertEqual(1.0, config.assets.custom_weight)

    def test_all_zero_is_still_refused(self) -> None:
        """The control. Without it the tests above pass for a validator that accepts anything."""
        with self.assertRaises(Exception):
            DatagenConfig(assets={"procedural_weight": 0.0})

    def test_the_sum_is_DERIVED_rather_than_written_out(self) -> None:
        """⚠ A HAND-WRITTEN SUM IS A SECOND DECLARATION OF WHAT THE SOURCES ARE, and it drifted the
        moment one was added. This checks the mechanism, not one instance of it: a weight field
        invented here must be seen by the validator without the validator being edited."""
        import pydantic

        class Extra(AssetSourcesConfig):
            invented_weight: float = pydantic.Field(default=0.0, ge=0.0)

        Extra(procedural_weight=0.0, invented_weight=1.0)


class EverySupportedMeshSourceCanBeDrawnTests(unittest.TestCase):
    """⛔ THE OTHER HALF. A weight the schema accepts and the layout ignores is an inert switch, and
    this repository fences that shape everywhere else."""

    def test_no_supported_mesh_source_lacks_a_weight(self) -> None:
        from datagen.assets.library import SUPPORTED_SOURCES
        from datagen.scenes.layout import _MESH_SOURCE_WEIGHTS

        drawable = {source for source, _field in _MESH_SOURCE_WEIGHTS}
        # `custom` is drawable and is in both; nothing else may be missing.
        missing = set(SUPPORTED_SOURCES) - drawable
        self.assertEqual(set(), missing,
                         f"{sorted(missing)} can be fetched and screened but never placed")

    def test_every_drawable_source_names_a_field_that_exists(self) -> None:
        """The mirror. A mapping entry pointing at a field nobody declared would read as zero
        forever, via `getattr(..., field, 0.0)`."""
        from datagen.scenes.layout import _MESH_SOURCE_WEIGHTS

        for source, field in _MESH_SOURCE_WEIGHTS:
            self.assertIn(field, AssetSourcesConfig.model_fields,
                          f"{source} draws on {field}, which no schema field declares")

    def test_a_refusal_that_outlived_its_reason_is_gone(self) -> None:
        """⛔ `f7cc043` ADMITTED OBJAVERSE BECAUSE ITS AUTHORS ARE WRITTEN DOWN NOW, measured at 297
        of 297 rows carrying one, and this list still refused it by name with "we hold no author
        names". For a day the commit said admitted and half the package said no. A refusal that
        outlives its reason is worse than none, because it reads as a considered decision."""
        from datagen.scenes.layout import _UNSUPPORTED_SOURCE_WEIGHTS

        refused = {source for source, _field, _reason in _UNSUPPORTED_SOURCE_WEIGHTS}
        self.assertNotIn("objaverse", refused)

    def test_the_refusal_mechanism_still_exists(self) -> None:
        """⚠ EMPTY IS NOT DELETED. The next source may genuinely be unsupplied, and the loud
        one-time warning is the right answer then."""
        from datagen.scenes import layout

        self.assertTrue(hasattr(layout, "_UNSUPPORTED_SOURCE_WEIGHTS"))


if __name__ == "__main__":
    unittest.main()
