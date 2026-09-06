"""Phase T6 — RED tests for ``EffectiveGraspingConfig`` integration.

Two things matter here:

* **Snapshot extension**: ``EffectiveGraspingConfig`` grows seven new
  ``uncertainty_*`` fields and ``to_dict()`` surfaces them. Defaults
  match Q5=B legacy mapping (when ``uncertainty.enabled=False`` the
  effective ``uncertainty_fail_closed_threshold`` mirrors
  ``decision.auto_uncertainty_threshold``).
* **Always-emit (Q7=B)**: every :class:`AutonomousGraspReport`
  produced by :meth:`AutonomousGraspService.pick` carries a
  populated :class:`UncertaintySnapshot`. When the layer is off the
  snapshot reports ``fused=0.0`` and ``fused_available=False``.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.config.schema.robot import RobotConfig
from src.geometry import Frame, Pose
from src.robot.core import MotionCommand, MotionResult, RobotCapabilities
from src.robot.execution.autonomous_grasp import (
    AutonomousGraspService,
)
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.perception import PerceptionFrame
from src.robot.grasping.uncertainty import UncertaintySnapshot


_SIM_CAPS = RobotCapabilities(
    vendor="dummy", model="sim", dof=6,
    supports_joint_move=True, supports_linear_move=True,
    supports_async_move=False, has_native_fk=False,
    has_native_ik=False, has_force_control=False, is_simulated=True,
)


class _FakeArm:
    def __init__(self) -> None:
        self._caps = _SIM_CAPS
        self._tcp = Pose(
            position_mm=np.array([0.0, 0.0, 500.0]),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.BASE, label="home",
        )

    @property
    def capabilities(self) -> RobotCapabilities:
        return self._caps

    def get_tcp_pose(self) -> Pose:
        return self._tcp

    def move(self, pose: Pose, **_: object) -> MotionResult:
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)


class _FakePerception:
    def __init__(self) -> None:
        depth = np.full((32, 32), 500.0, dtype=np.float64)
        intr = np.array([[400.0, 0.0, 16.0], [0.0, 400.0, 16.0], [0.0, 0.0, 1.0]])
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:22, 10:22] = 1
        self._frame = PerceptionFrame(
            depth_map=depth, intrinsics=intr,
            segmentations=(SimpleNamespace(mask=mask),),
        )

    def acquire(self) -> PerceptionFrame:
        return self._frame


class _Calc:
    def compute_result(self, *_a: object, **_k: object) -> GraspResult:
        gp = GraspPoint(
            position=np.array([100.0, 50.0, 400.0]),
            approach=np.array([0.0, 0.0, 1.0]),
            axis=np.array([1.0, 0.0, 0.0]),
            grip_width_mm=40.0, score=0.9, frame=GraspFrame.BASE, label="t",
        )
        return GraspResult(candidates=(gp,), reasons=(), top_score=0.9)


def _cfg(**uncertainty_overrides: object) -> RobotConfig:
    return RobotConfig(
        vendor="dummy",
        gripper={"vendor": "none"},
        grasping={
            "default_mode": "auto",
            "max_attempts": 5,
            "uncertainty": uncertainty_overrides,
        },
    )


class EffectiveConfigSnapshotTests(unittest.TestCase):
    def test_defaults_when_block_absent(self) -> None:
        cfg = RobotConfig(
            vendor="dummy", gripper={"vendor": "none"},
            grasping={"default_mode": "auto"},
        )
        svc = AutonomousGraspService.from_robot_config(
            cfg, calculator=_Calc(), perception=_FakePerception(),
        )
        eff = svc.effective_config
        self.assertIsNotNone(eff)
        self.assertFalse(eff.uncertainty.enabled)
        # Q5=B: when off, the effective threshold mirrors the legacy
        # decision.auto_uncertainty_threshold default (0.4).
        self.assertEqual(eff.uncertainty.fail_closed_threshold, 0.4)
        self.assertEqual(
            eff.uncertainty.apply_modes,
            ("auto", "dense_clutter", "dense_autonomous"),
        )

    def test_enabled_block_wins_over_legacy(self) -> None:
        cfg = RobotConfig(
            vendor="dummy", gripper={"vendor": "none"},
            grasping={
                "default_mode": "auto",
                "decision": {"auto_uncertainty_threshold": 0.25},
                "uncertainty": {"enabled": True, "fail_closed_threshold": 0.55},
            },
        )
        svc = AutonomousGraspService.from_robot_config(
            cfg, calculator=_Calc(), perception=_FakePerception(),
        )
        eff = svc.effective_config
        self.assertTrue(eff.uncertainty.enabled)
        # Layer enabled → new key wins, not 0.25.
        self.assertAlmostEqual(eff.uncertainty.fail_closed_threshold, 0.55)

    def test_to_dict_includes_uncertainty_keys(self) -> None:
        cfg = _cfg()
        svc = AutonomousGraspService.from_robot_config(
            cfg, calculator=_Calc(), perception=_FakePerception(),
        )
        d = svc.effective_config.to_dict()
        for k in (
            "uncertainty_enabled",
            "uncertainty_fail_closed_threshold",
            "uncertainty_apply_modes",
            "uncertainty_weights",
            "uncertainty_calibration_artifact_path",
        ):
            self.assertIn(k, d)


class AlwaysEmitSnapshotTests(unittest.TestCase):
    def test_report_carries_uncertainty_snapshot_when_disabled(self) -> None:
        cfg = _cfg()  # disabled by default
        svc = AutonomousGraspService.from_robot_config(
            cfg, calculator=_Calc(), perception=_FakePerception(),
        )
        report = svc.pick()
        # Q7=B is independent of outcome: the snapshot must be
        # present on every report regardless of whether execution
        # succeeded.
        self.assertIsInstance(report.uncertainty, UncertaintySnapshot)
        self.assertEqual(report.uncertainty.fused, 0.0)
        self.assertFalse(report.uncertainty.fused_available)
        self.assertFalse(report.uncertainty.fail_closed)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
