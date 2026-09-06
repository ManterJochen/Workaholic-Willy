"""The single-object source must pick the same object the old `detect()` call did.

Written after run_fused_pick went 0/5. The backend seam replaced `detect()` (which selects
``scores.argmax()``) with ``detect_all()`` plus "take the first" -- and ``detect_all`` yields the
model's own emission order, which is not sorted. On a one-object scene the two agree, which is exactly
why M2 stayed 10/10 and the defect shipped unseen.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass


@dataclass
class _Det:
    score: float
    label: str


@dataclass
class _Obj:
    detection: _Det
    segmentation: str


def _pick(objects):
    """The selection rule under test, isolated from Isaac (the source needs a live camera)."""
    return max(objects, key=lambda o: o.detection.score, default=None)


class SingleObjectSelectionTests(unittest.TestCase):
    def test_the_highest_score_wins_regardless_of_emission_order(self) -> None:
        """The fused-scene case: several cubes, and the best match is not the first emitted."""
        objects = [
            _Obj(_Det(0.31, "cube"), "wrong-a"),
            _Obj(_Det(0.87, "cube"), "right"),
            _Obj(_Det(0.44, "cube"), "wrong-b"),
        ]
        self.assertEqual(_pick(objects).segmentation, "right")

    def test_a_single_object_is_unaffected(self) -> None:
        """Why M2 stayed green and hid the bug -- kept so the coincidence is documented, not relied on."""
        self.assertEqual(_pick([_Obj(_Det(0.5, "cube"), "only")]).segmentation, "only")

    def test_nothing_perceived_yields_no_selection(self) -> None:
        """An empty result is "not there", which must fall through to the no-segmentation path."""
        self.assertIsNone(_pick([]))

    def test_equal_scores_keep_the_models_own_order(self) -> None:
        """The VLM route: every detection carries the same nominal score, so max() must be stable and
        preserve the order the model emitted -- which IS its ranking there."""
        objects = [_Obj(_Det(1.0, "a"), "first"), _Obj(_Det(1.0, "b"), "second")]
        self.assertEqual(_pick(objects).segmentation, "first")

    def test_the_source_uses_this_rule(self) -> None:
        """Guard against the rule drifting out of the source it was written for."""
        import inspect

        from src.willy_sim.perception import vision

        source = inspect.getsource(vision.IsaacVisionPerceptionSource.acquire)
        self.assertIn("max(perceived", source)
        self.assertIn("detection.score", source)


if __name__ == "__main__":
    unittest.main()
