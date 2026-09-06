"""Tests for the production uncertainty calibration tool.

The tool lives at
``backend/src/robot/grasping/calibration/uncertainty_calibration.py``
and exposes:

* :func:`fit_uncertainty_calibration` — pure function over a list of
  replay records that produces a :class:`UncertaintyCalibration`
  artifact. The fit is **deterministic** (no PRNG, no thread-local
  state) and **monotone-preserving** (the per-channel map fitted
  against ``label`` is non-decreasing).
* A ``__main__`` CLI entry-point:
  ``python -m backend.src.robot.grasping.calibration.uncertainty_calibration \
      --replay PATH --out PATH``.

These assert determinism, monotonicity, and YAML artifact round-trip ---
not isotonic regression accuracy itself.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.robot.grasping.calibration.uncertainty_calibration import (
    fit_uncertainty_calibration,
)
from src.robot.grasping.uncertainty import (
    UncertaintyCalibration,
    UncertaintyChannel,
)

FIXTURE = (
    Path(__file__).resolve().parent / "data" / "uncertainty_replay.jsonl"
)


def _load_records() -> list[dict]:
    with FIXTURE.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


class FitDeterminismTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = _load_records()
        # Attach synthetic binary labels; the fit must learn a
        # non-decreasing map from each channel value to label.
        for rec in self.records:
            # A deterministic label rule that is monotone in
            # ``feasibility_margin`` so the fit has signal.
            fm = rec.get("feasibility_margin") or 0.0
            rec["label"] = 1.0 if fm > 0.5 else 0.0

    def test_fit_is_bit_identical_across_two_runs(self) -> None:
        cal1 = fit_uncertainty_calibration(self.records)
        cal2 = fit_uncertainty_calibration(self.records)
        a1 = cal1.to_artifact()
        a2 = cal2.to_artifact()
        self.assertEqual(
            json.dumps(a1, sort_keys=True),
            json.dumps(a2, sort_keys=True),
        )

    def test_fit_maps_are_monotone(self) -> None:
        cal = fit_uncertainty_calibration(self.records)
        for ch in UncertaintyChannel:
            m = cal.maps[ch]
            for i in range(1, len(m.values)):
                self.assertGreaterEqual(
                    m.values[i], m.values[i - 1] - 1e-12,
                    f"channel {ch.value} map non-monotone at idx {i}",
                )

    def test_artifact_round_trip(self) -> None:
        cal = fit_uncertainty_calibration(self.records)
        artifact = cal.to_artifact()
        cal2 = UncertaintyCalibration.from_artifact(artifact)
        # Sample a few points and verify equivalence.
        for ch in UncertaintyChannel:
            for x in (0.0, 0.25, 0.5, 0.75, 1.0):
                self.assertAlmostEqual(
                    cal.maps[ch].apply(x),
                    cal2.maps[ch].apply(x),
                    places=9,
                )


class CLITests(unittest.TestCase):
    def test_cli_produces_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "artifact.json"
            res = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.robot.grasping.calibration.uncertainty_calibration",
                    "--replay",
                    str(FIXTURE),
                    "--out",
                    str(out),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                res.returncode, 0,
                msg=f"stderr={res.stderr!r} stdout={res.stdout!r}",
            )
            self.assertTrue(out.exists())
            data = json.loads(out.read_text(encoding="utf-8"))
            # Artifact carries weights + per-channel maps.
            self.assertIn("weights", data)
            self.assertIn("maps", data)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
