"""R4 K5 SEAM 2 Seam-0 — pin the corridor + gain query LITERAL values before extracting _fusion_queries.

R-6 (the adversarial design's refuted hazard): the existing U6 tests assert only INEQUALITIES for corridor_evidence
+ a determinism check for the gain — the EXACT values (especially seen_voxels / seen_fraction) are golden-blind, so
a float-op reorder in the extracted free functions could flip a value and pass. This pins the literal values
(captured from HEAD before the move) for the reduction-heavy on-axis corridor + the empty-fusion gain.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_u6_multiview_decision_integration import (  # noqa: E402
    _enabled_fusion,
    _identity_cam_to_base,
    _pinhole_intrinsics,
)

from src.robot.grasping.multiview.fusion import SceneFusion  # noqa: E402


class FusionQueriesSeam0Tests(unittest.TestCase):
    def test_corridor_on_axis_exact_values(self) -> None:
        fusion = SceneFusion(config=_enabled_fusion())
        fusion.ingest(
            depth_map=np.full((16, 16), 150.0, dtype=np.float64),
            intrinsics=_pinhole_intrinsics(),
            t_cam_to_base=_identity_cam_to_base(),
            timestamp_ns=0,
        )
        ev = fusion.corridor_evidence(
            position_mm=np.array([0.0, 0.0, 150.0], dtype=np.float64),
            approach=np.array([0.0, 0.0, -1.0], dtype=np.float64),
            length_mm=120.0,
            radius_mm=25.0,
        )
        self.assertEqual(ev.queried_voxels, 260)
        self.assertEqual(ev.hit_voxels, 4)
        self.assertEqual(ev.seen_voxels, 4)
        self.assertEqual(ev.hit_fraction, 0.015384615384615385)
        self.assertEqual(ev.seen_fraction, 0.015384615384615385)
        self.assertEqual(ev.views_accepted, 1)

    def test_gain_empty_exact_values(self) -> None:
        fusion = SceneFusion(config=_enabled_fusion())
        gain = fusion.viewpoint_information_gain(
            t_cam_to_base=_identity_cam_to_base(),
            intrinsics=_pinhole_intrinsics(),
            depth_shape=(16, 16),
        )
        self.assertEqual(gain.unseen_voxels, 600000)
        self.assertEqual(gain.predicted_visible_unseen, 68)
        self.assertEqual(gain.image_shape, (16, 16))


if __name__ == "__main__":
    unittest.main()
