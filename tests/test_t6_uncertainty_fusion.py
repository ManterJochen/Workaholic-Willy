"""Phase T6 — RED tests for the uncertainty fusion carrier.

T6 fuses 7 typed signal channels into a single calibrated
:class:`UncertaintySnapshot`:

* depth_confidence, mask_confidence, occlusion_corridor_risk,
  feasibility_margin, verification_residual, topology_risk,
  semantic_confidence.

The fusion contract (locked design decisions Q1=B, Q2 hybrid):

1. Each channel value lives in ``[0, 1]``. ``None`` means "not
   measured" — the channel contributes zero weight.
2. The calibration ships a per-channel monotone non-decreasing
   piecewise-linear remap. The default ``identity`` calibration is a
   no-op.
3. The fused value is a convex combination of remapped channels
   weighted by ``UncertaintyWeights`` (per-channel, ≥ 0). When no
   channels are available the fused value is ``0.0`` and the snapshot
   records ``fused_available=False``.
4. ``UncertaintySnapshot`` is JSON-safe via ``to_dict()``.

Each test asserts a specific clause of the contract.
"""

from __future__ import annotations

import unittest

from src.robot.grasping.uncertainty import (
    UncertaintyCalibration,
    UncertaintyChannel,
    UncertaintyChannelValues,
    UncertaintyMonotoneMap,
    UncertaintyWeights,
    fuse_uncertainty,
)


class UncertaintyChannelValuesTests(unittest.TestCase):
    def test_default_all_none(self) -> None:
        v = UncertaintyChannelValues()
        for name in (
            "depth_confidence",
            "mask_confidence",
            "occlusion_corridor_risk",
            "feasibility_margin",
            "verification_residual",
            "topology_risk",
            "semantic_confidence",
        ):
            self.assertIsNone(getattr(v, name), name)

    def test_values_outside_unit_interval_rejected(self) -> None:
        with self.assertRaises(ValueError):
            UncertaintyChannelValues(depth_confidence=-0.1)
        with self.assertRaises(ValueError):
            UncertaintyChannelValues(mask_confidence=1.1)

    def test_nan_rejected(self) -> None:
        with self.assertRaises(ValueError):
            UncertaintyChannelValues(topology_risk=float("nan"))


class UncertaintyWeightsTests(unittest.TestCase):
    def test_defaults_match_locked_q1_priorities(self) -> None:
        w = UncertaintyWeights()
        # 5 core channels start at 1.0; topology + semantic default 0
        # because those channels are not always produced.
        self.assertEqual(w.depth_confidence, 1.0)
        self.assertEqual(w.mask_confidence, 1.0)
        self.assertEqual(w.occlusion_corridor_risk, 1.0)
        self.assertEqual(w.feasibility_margin, 1.0)
        self.assertEqual(w.verification_residual, 1.0)
        self.assertEqual(w.topology_risk, 0.0)
        self.assertEqual(w.semantic_confidence, 0.0)

    def test_negative_weights_rejected(self) -> None:
        with self.assertRaises(ValueError):
            UncertaintyWeights(depth_confidence=-0.5)


class UncertaintyMonotoneMapTests(unittest.TestCase):
    def test_identity_passes_through(self) -> None:
        m = UncertaintyMonotoneMap.identity()
        for x in (0.0, 0.25, 0.5, 0.75, 1.0):
            self.assertAlmostEqual(m.apply(x), x, places=9)

    def test_monotone_violation_rejected(self) -> None:
        with self.assertRaises(ValueError):
            UncertaintyMonotoneMap(
                breakpoints=(0.0, 0.5, 1.0),
                values=(0.0, 0.7, 0.3),  # decreases
            )

    def test_breakpoint_length_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            UncertaintyMonotoneMap(
                breakpoints=(0.0, 1.0),
                values=(0.0, 0.5, 1.0),
            )

    def test_piecewise_linear_interpolation(self) -> None:
        m = UncertaintyMonotoneMap(
            breakpoints=(0.0, 0.5, 1.0),
            values=(0.0, 0.25, 1.0),
        )
        # At a breakpoint
        self.assertAlmostEqual(m.apply(0.5), 0.25, places=9)
        # Linear interpolation halfway between 0.5 and 1.0 -> 0.625
        self.assertAlmostEqual(m.apply(0.75), 0.625, places=9)

    def test_clamps_outside_breakpoints(self) -> None:
        m = UncertaintyMonotoneMap(
            breakpoints=(0.0, 1.0),
            values=(0.1, 0.9),
        )
        self.assertAlmostEqual(m.apply(-0.5), 0.1, places=9)
        self.assertAlmostEqual(m.apply(2.0), 0.9, places=9)


class UncertaintyCalibrationTests(unittest.TestCase):
    def test_identity_is_noop(self) -> None:
        cal = UncertaintyCalibration.identity()
        for ch in UncertaintyChannel:
            self.assertAlmostEqual(cal.maps[ch].apply(0.3), 0.3, places=9)

    def test_artifact_round_trip(self) -> None:
        cal = UncertaintyCalibration(
            weights=UncertaintyWeights(),
            maps={
                ch: UncertaintyMonotoneMap.identity()
                for ch in UncertaintyChannel
            },
            calibration_id="test-cal-0001",
        )
        artifact = cal.to_artifact()
        cal2 = UncertaintyCalibration.from_artifact(artifact)
        self.assertEqual(cal2.calibration_id, "test-cal-0001")
        for ch in UncertaintyChannel:
            self.assertAlmostEqual(
                cal2.maps[ch].apply(0.42),
                cal.maps[ch].apply(0.42),
                places=9,
            )


class FuseUncertaintyTests(unittest.TestCase):
    def test_all_none_yields_zero_unavailable(self) -> None:
        snap = fuse_uncertainty(
            values=UncertaintyChannelValues(),
            calibration=UncertaintyCalibration.identity(),
            fail_closed_threshold=0.5,
        )
        self.assertEqual(snap.fused, 0.0)
        self.assertFalse(snap.fused_available)
        self.assertFalse(snap.fail_closed)

    def test_uniform_channels_uniform_weights_yields_average(self) -> None:
        snap = fuse_uncertainty(
            values=UncertaintyChannelValues(
                depth_confidence=0.4,
                mask_confidence=0.6,
                occlusion_corridor_risk=0.2,
                feasibility_margin=0.8,
                verification_residual=0.5,
            ),
            calibration=UncertaintyCalibration.identity(),
            fail_closed_threshold=0.5,
        )
        # Default weights: 5 active channels each weight 1.0,
        # topology+semantic weight 0. Expected fused = mean(0.4, 0.6,
        # 0.2, 0.8, 0.5) = 0.5
        self.assertAlmostEqual(snap.fused, 0.5, places=9)
        self.assertTrue(snap.fused_available)

    def test_missing_channel_skipped(self) -> None:
        snap = fuse_uncertainty(
            values=UncertaintyChannelValues(
                depth_confidence=0.5,
                mask_confidence=None,  # absent
            ),
            calibration=UncertaintyCalibration.identity(),
            fail_closed_threshold=0.5,
        )
        # Only depth_confidence contributes; fused = 0.5.
        self.assertAlmostEqual(snap.fused, 0.5, places=9)
        self.assertTrue(snap.fused_available)

    def test_zero_weighted_channel_skipped(self) -> None:
        snap = fuse_uncertainty(
            values=UncertaintyChannelValues(
                depth_confidence=0.5,
                topology_risk=1.0,  # weight 0 by default
            ),
            calibration=UncertaintyCalibration.identity(),
            fail_closed_threshold=0.5,
        )
        # topology_risk weight is 0 -> does not raise the fused.
        self.assertAlmostEqual(snap.fused, 0.5, places=9)

    def test_fail_closed_threshold_evaluated(self) -> None:
        snap_below = fuse_uncertainty(
            values=UncertaintyChannelValues(depth_confidence=0.3),
            calibration=UncertaintyCalibration.identity(),
            fail_closed_threshold=0.5,
        )
        snap_above = fuse_uncertainty(
            values=UncertaintyChannelValues(depth_confidence=0.7),
            calibration=UncertaintyCalibration.identity(),
            fail_closed_threshold=0.5,
        )
        self.assertFalse(snap_below.fail_closed)
        self.assertTrue(snap_above.fail_closed)

    def test_weights_normalisation(self) -> None:
        snap = fuse_uncertainty(
            values=UncertaintyChannelValues(
                depth_confidence=0.2,
                mask_confidence=0.8,
            ),
            calibration=UncertaintyCalibration(
                weights=UncertaintyWeights(
                    depth_confidence=3.0,
                    mask_confidence=1.0,
                ),
                maps={ch: UncertaintyMonotoneMap.identity() for ch in UncertaintyChannel},
            ),
            fail_closed_threshold=0.5,
        )
        # fused = (3*0.2 + 1*0.8) / (3+1) = 1.4/4 = 0.35
        self.assertAlmostEqual(snap.fused, 0.35, places=9)

    def test_monotone_remap_applied(self) -> None:
        # Square-the-input style monotone map only on depth_confidence.
        maps = {ch: UncertaintyMonotoneMap.identity() for ch in UncertaintyChannel}
        maps[UncertaintyChannel.DEPTH_CONFIDENCE] = UncertaintyMonotoneMap(
            breakpoints=(0.0, 1.0),
            values=(0.0, 0.5),
        )
        cal = UncertaintyCalibration(
            weights=UncertaintyWeights(
                depth_confidence=1.0,
                mask_confidence=0.0,
                occlusion_corridor_risk=0.0,
                feasibility_margin=0.0,
                verification_residual=0.0,
            ),
            maps=maps,
        )
        snap = fuse_uncertainty(
            values=UncertaintyChannelValues(depth_confidence=1.0),
            calibration=cal,
            fail_closed_threshold=0.5,
        )
        # depth remapped 1.0 -> 0.5 (per the identity-on-[0,1] -> 0.5 map)
        self.assertAlmostEqual(snap.fused, 0.5, places=9)


class UncertaintySnapshotTests(unittest.TestCase):
    def test_to_dict_jsonsafe(self) -> None:
        snap = fuse_uncertainty(
            values=UncertaintyChannelValues(
                depth_confidence=0.3, mask_confidence=0.7
            ),
            calibration=UncertaintyCalibration.identity(),
            fail_closed_threshold=0.6,
        )
        d = snap.to_dict()
        self.assertEqual(set(d.keys()), {
            "fused", "fused_available", "fail_closed",
            "fail_closed_threshold", "channels", "calibration_id",
            # U8 additive disagreement fields.
            "disagreement", "disagreement_threshold",
            "disagreement_triggered",
        })
        self.assertIsInstance(d["fused"], float)
        self.assertIsInstance(d["fused_available"], bool)
        self.assertIsInstance(d["fail_closed"], bool)
        self.assertIsInstance(d["channels"], dict)
        # Channels JSON-safe: ints/floats/None only.
        for v in d["channels"].values():
            self.assertTrue(v is None or isinstance(v, (int, float)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
