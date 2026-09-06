"""Phase T6 — RED tests for monotone determinism on the replay fixture.

Two guarantees are asserted:

1. **Monotone-per-channel**: increasing exactly one channel's value while
   holding all others fixed never decreases the fused uncertainty.
   This holds with any non-decreasing calibration map and non-negative
   weights — the locked Q2 fusion contract.
2. **Bit-identical replay**: re-running ``fuse_uncertainty`` over the
   shipped ``tests/data/uncertainty_replay.jsonl`` fixture twice in
   the same process produces byte-identical floats, and the recorded
   ``expected_fused`` field matches.

The fixture is committed and seeded; agents must not regenerate it
without an explicit reviewer-approved change.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.robot.grasping.uncertainty import (
    UncertaintyCalibration,
    UncertaintyChannelValues,
    UncertaintyWeights,
    fuse_uncertainty,
)

FIXTURE = (
    Path(__file__).resolve().parent / "data" / "uncertainty_replay.jsonl"
)


def _channels_from_record(rec: dict) -> UncertaintyChannelValues:
    return UncertaintyChannelValues(
        depth_confidence=rec.get("depth_confidence"),
        mask_confidence=rec.get("mask_confidence"),
        occlusion_corridor_risk=rec.get("occlusion_corridor_risk"),
        feasibility_margin=rec.get("feasibility_margin"),
        verification_residual=rec.get("verification_residual"),
        topology_risk=rec.get("topology_risk"),
        semantic_confidence=rec.get("semantic_confidence"),
    )


class MonotoneTests(unittest.TestCase):
    def test_monotone_in_each_channel(self) -> None:
        cal = UncertaintyCalibration.identity()
        # Use uniform non-zero weights so every channel contributes.
        cal_uniform = UncertaintyCalibration(
            weights=UncertaintyWeights(
                depth_confidence=1.0,
                mask_confidence=1.0,
                occlusion_corridor_risk=1.0,
                feasibility_margin=1.0,
                verification_residual=1.0,
                topology_risk=1.0,
                semantic_confidence=1.0,
            ),
            maps=dict(cal.maps),
        )
        base_kwargs = dict(
            depth_confidence=0.2,
            mask_confidence=0.3,
            occlusion_corridor_risk=0.4,
            feasibility_margin=0.5,
            verification_residual=0.6,
            topology_risk=0.5,
            semantic_confidence=0.4,
        )
        base = fuse_uncertainty(
            values=UncertaintyChannelValues(**base_kwargs),
            calibration=cal_uniform,
            fail_closed_threshold=0.5,
        ).fused
        for ch_name in base_kwargs:
            bumped = dict(base_kwargs)
            bumped[ch_name] = min(1.0, base_kwargs[ch_name] + 0.2)
            bumped_fused = fuse_uncertainty(
                values=UncertaintyChannelValues(**bumped),
                calibration=cal_uniform,
                fail_closed_threshold=0.5,
            ).fused
            self.assertGreaterEqual(
                bumped_fused, base - 1e-12,
                f"bumping {ch_name} decreased fused",
            )


class ReplayDeterminismTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(
            FIXTURE.exists(),
            f"replay fixture missing at {FIXTURE}",
        )
        with FIXTURE.open() as fh:
            self.records = [json.loads(line) for line in fh if line.strip()]
        self.assertGreaterEqual(len(self.records), 20)

    def test_replay_bit_identical_across_two_runs(self) -> None:
        cal = UncertaintyCalibration.identity()
        run1 = [
            fuse_uncertainty(
                _channels_from_record(rec),
                calibration=cal,
                fail_closed_threshold=0.5,
            ).fused
            for rec in self.records
        ]
        run2 = [
            fuse_uncertainty(
                _channels_from_record(rec),
                calibration=cal,
                fail_closed_threshold=0.5,
            ).fused
            for rec in self.records
        ]
        for a, b in zip(run1, run2):
            self.assertEqual(a.hex(), b.hex())

    def test_replay_expected_fused_matches(self) -> None:
        cal = UncertaintyCalibration.identity()
        for rec in self.records:
            expected = rec["expected_fused_identity"]
            got = fuse_uncertainty(
                _channels_from_record(rec),
                calibration=cal,
                fail_closed_threshold=0.5,
            ).fused
            self.assertAlmostEqual(
                got, expected, places=9,
                msg=f"mismatch on record {rec.get('id')}",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
