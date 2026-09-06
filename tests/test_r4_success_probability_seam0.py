"""R4 K2 Seam-0 — pin the ABSOLUTE predict_proba values before the success_probability package conversion.

R-2 (the adversarial design's refuted hazard): NO absolute predict_proba golden existed. The committed model.json
sha compares the sklearn params, never a predict VALUE, so a float-perturbing reorder of the load-bearing scoring
core (which reranks candidates via the default-off U4 blend / G4 rerank) would land green. This pins predict_proba
on fixed non-imputed 23-d float64 vectors to the exact HEAD values, captured before the single-file → 6-leaf
package move. assertEqual = bit-exact (no tolerance).
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.robot.grasping.scoring import success_probability as sp

_ART = (
    Path(__file__).resolve().parent.parent
    / "assets" / "models" / "success_probability" / "v1"
)

# Captured from HEAD (uniform-fill 1x23 float64 vectors). Non-degenerate places-12 values.
_BASELINE = {
    0.3: 0.2,
    0.5: 0.8170974155069582,
    0.6: 0.914572864321608,
    0.7: 0.9407407407407408,
}


class SuccessProbabilitySeam0Tests(unittest.TestCase):
    def test_predict_proba_absolute_values(self) -> None:
        model = sp.load_success_probability_model(_ART)
        for fill, expected in _BASELINE.items():
            with self.subTest(fill=fill):
                x = np.full((1, 23), fill, dtype=np.float64)
                got = float(np.asarray(sp.predict_proba(model, x)).ravel()[0])
                self.assertEqual(got, expected)

    def test_default_logger_name_unchanged(self) -> None:
        # R-1: try_load's default warning (no logger passed) must emit under the ORIGINAL module path, not
        # the _shadow leaf — the assertLogs suites inject their own logger so the default name is golden-blind.
        with self.assertLogs(
            "src.robot.grasping.scoring.success_probability", level="WARNING"
        ) as cm:
            result = sp.try_load_shadow_success_context(
                enabled=True,
                artifact_dir=_ART,
                mode_label="easy",
                lifecycle_phase="bogus_phase",
            )
        self.assertIsNone(result)
        self.assertTrue(any("lifecycle_phase" in m for m in cm.output))


if __name__ == "__main__":
    unittest.main()
