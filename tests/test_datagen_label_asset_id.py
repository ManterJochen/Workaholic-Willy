"""What a rendered scene puts in `views[i].objects[j].asset_id`.

⛔ WHY IT EXISTS. `build_object_labels` documents its `placements` argument as carrying an asset_id
STRING. The Isaac path passes `instances[index][1]`, a string. The raster path passed
`records[index]` -- the whole `SceneAsset` -- for the entire v3 and v4 corpus, so every raster scene
holds a record where an id belongs.

⚠ AND THE FIRST DRAFT OF THIS FILE READ `view.objects`, which does not exist -- the field is
`labels`, and `getattr(v, "objects", ())` returned empty for every view. The test would have passed
over a corpus of records without ever looking at one. It failed loudly only because the first
assertion refuses an EMPTY set of views rather than iterating over nothing; write that guard.

⚠ THREE THINGS HAD TO LINE UP FOR THAT TO SURVIVE A WHOLE CORPUS, and each is a lesson on its own:

  * the caller's dict is `dict[int, Any]`, so the annotation was decoration and mypy had nothing to
    check. An `Any` at the boundary turns a type error into a runtime shape nobody sees.
  * it was harmless BY LUCK. Every consumer that needs an id reads `spec.objects[i].asset_id`
    instead, so the corpus and the fold grouping were unaffected. Harmless-by-luck is not a reason
    to leave it: the next consumer to reach for the label's own field would have got a record.
  * NO TEST RENDERED A SCENE AND LOOKED AT A LABEL. It was found by opening a finished scene by
    hand, which is the same way this arc found the blank-RGB defect and the instance-base defect.

So this file renders one, end to end, and reads the labels the way a consumer would.
"""

from __future__ import annotations

import unittest

import numpy as np

from datagen.build import build_manifest
from datagen.config import DatagenConfig
from datagen.render.labels import build_object_labels
from datagen.render.noengine import NoEngineRenderer
from datagen.scenes.layout import layout_scene


def _config() -> DatagenConfig:
    """Procedural only: no mesh bank to load, so the test costs a rasterisation and nothing else."""
    return DatagenConfig.model_validate({
        "scenes": 1, "seed": 7,
        "assets": {"procedural_weight": 1.0, "gso_weight": 0.0, "ycb_weight": 0.0,
                   "objaverse_weight": 0.0, "composite_weight": 0.0},
        "families": {"sparse_weight": 1.0, "packed_weight": 0.0, "pile_weight": 0.0,
                     "bin_weight": 0.0, "sparse_objects": [2, 3]},
        "render": {"arm": {"mode": "absent"}},
        "camera_rig": {"views": ["overhead"], "resolution": [96, 72]},
    })


def _render():
    config = _config()
    spec = layout_scene(config, 0)
    result = NoEngineRenderer(config).render(spec, build_manifest(config),
                                             np.random.default_rng(0))
    return spec, result


class TheRenderedLabelTests(unittest.TestCase):
    def test_a_label_carries_an_ID_and_not_a_RECORD(self) -> None:
        """⛔ THE ONE. Read the field a consumer would read, on a scene that was actually rendered."""
        _spec, result = _render()
        views = [v for v in result.views if v.labels]
        self.assertTrue(views, "the render produced no labelled view, so nothing was checked")
        for view in views:
            for label in view.labels:
                self.assertIsInstance(
                    label.asset_id, str,
                    f"asset_id is a {type(label.asset_id).__name__}, not an id")

    def test_the_id_is_the_one_the_SPEC_placed(self) -> None:
        """A string is not enough -- `str(record)` is a string too. It has to be the same id."""
        spec, result = _render()
        placed = {obj.asset_id for obj in spec.objects}
        for view in result.views:
            for label in view.labels:
                self.assertIn(label.asset_id, placed)


class TheRefusalTests(unittest.TestCase):
    """The check that keeps it fixed where the annotation could not."""

    def _placements(self, asset_id: object) -> dict:
        return {0: (asset_id, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))}

    def test_a_non_string_is_refused_and_the_message_says_what_to_pass(self) -> None:
        mask = np.zeros((4, 4), dtype=bool)
        with self.assertRaises(TypeError) as caught:
            build_object_labels(np.zeros((4, 4), dtype=np.int32), {0: mask},
                                self._placements(object()))
        self.assertIn("record.asset_id", str(caught.exception))

    def test_a_string_still_passes(self) -> None:
        mask = np.zeros((4, 4), dtype=bool)
        labels = build_object_labels(np.zeros((4, 4), dtype=np.int32), {0: mask},
                                     self._placements("gso/mug_01"))
        self.assertEqual(labels[0].asset_id, "gso/mug_01")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
