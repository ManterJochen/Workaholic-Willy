from __future__ import annotations

import unittest
from pathlib import Path

from pydantic import ValidationError

from src.config.schema.camera import (
    CameraSystemConfig,
    EyeInHandWorkflowConfig,
    EyeToHandWorkflowConfig,
    HandEyeConfig,
    StereoMatcherConfig,
    WebcamPairRigConfig,
)
from src.config.schema.robot import (
    GraspingClosedLoopConfig,
    GraspingDecisionConfig,
    GraspingDenseRecoveryConfig,
    GraspingVerificationConfig,
    GripperConfig,
    RobotConfig,
    RobotGraspingConfig,
    WorkspaceLimitsConfig,
    RobotCalibrationConfig,
)


def _rig(
    rig_id: str = "rig",
    enabled: bool = True,
    **overrides: object,
) -> WebcamPairRigConfig:
    data = dict(
        source="webcam_pair",
        rig_id=rig_id,
        enabled=enabled,
        fps=30,
        backend=0,
        frame_size=(640, 480),
        max_cam_scan=2,
        cam_left_id=0,
        cam_right_id=1,
        min_pairs=4,
        max_pairs=10,
        calibration_paths={"base_dir": f"calibration/{rig_id}"},
    )
    data.update(overrides)
    return WebcamPairRigConfig(**data)


class ConfigSchemaTests(unittest.TestCase):
    def test_hand_eye_workflows_are_independent(self) -> None:
        hand_eye = HandEyeConfig(
            eye_to_hand={"enabled": False, "min_angle_deg": 11.0},
            eye_in_hand={"enabled": True, "min_angle_deg": 12.0},
        )

        self.assertEqual(hand_eye.eye_to_hand.mode, "eye_to_hand")
        self.assertEqual(hand_eye.eye_to_hand.min_angle, 11.0)
        self.assertEqual(hand_eye.eye_in_hand.mode, "eye_in_hand")
        self.assertEqual(hand_eye.eye_in_hand.min_angle_deg, 12.0)

        with self.assertRaises(ValidationError):
            EyeInHandWorkflowConfig(mode="eye_to_hand")

    def test_the_live_workflow_config_accepts_the_min_angle_deg_alias(self) -> None:
        """Coverage inherited from the deleted legacy block: the `min_angle_deg` alias is what existing
        YAML writes, and it is a property of the LIVE EyeHandRoutineConfig, so it keeps a test."""
        routine = EyeToHandWorkflowConfig(min_angle_deg=10.0)
        self.assertEqual(routine.min_angle, 10.0)
        self.assertEqual(routine.min_angle_deg, 10.0)

    def test_camera_system_validates_active_rig_contract(self) -> None:
        CameraSystemConfig(active_mode="auto", rigs=[_rig("a")])
        CameraSystemConfig(active_mode="rig", active_rig_id="a", rigs=[_rig("a")])

        with self.assertRaises(ValidationError):
            CameraSystemConfig(active_mode="rig", rigs=[_rig("a")])
        with self.assertRaises(ValidationError):
            CameraSystemConfig(active_mode="rig", active_rig_id="missing", rigs=[_rig("a")])
        with self.assertRaises(ValidationError):
            CameraSystemConfig(active_mode="rig", active_rig_id="a", rigs=[_rig("a", enabled=False)])
        with self.assertRaises(ValidationError):
            CameraSystemConfig(active_mode="auto", rigs=[_rig("a"), _rig("a")])

    def test_camera_rig_and_matcher_numeric_validation(self) -> None:
        with self.assertRaises(ValidationError):
            _rig(frame_size=(0, 480))
        with self.assertRaises(ValidationError):
            WebcamPairRigConfig(
                source="webcam_pair",
                rig_id="bad",
                enabled=True,
                fps=30,
                frame_size=(640, 480),
                max_cam_scan=1,
                cam_left_id=0,
                cam_right_id=0,
                calibration_paths={"base_dir": "calibration/bad"},
            )
        with self.assertRaises(ValidationError):
            StereoMatcherConfig(
                minDisparity=0,
                numDisparities=30,
                blockSize=7,
                uniquenessRatio=10,
                speckleWindowSize=50,
                speckleRange=1,
                disp12MaxDiff=1,
            )
        with self.assertRaises(ValidationError):
            StereoMatcherConfig(
                minDisparity=0,
                numDisparities=32,
                blockSize=7,
                p1=100,
                p2=50,
                uniquenessRatio=10,
                speckleWindowSize=50,
                speckleRange=1,
                disp12MaxDiff=1,
            )

    def test_robot_schema_is_runtime_safe_and_still_validates_bounds(self) -> None:
        # L0.2: vendor is validated against the RobotVendor/GripperVendor enums at load (the enums
        # are pure StrEnums, lazily imported, so config-edit hosts stay free of robot-runtime deps).
        cfg = RobotConfig(vendor="SIM", gripper={"vendor": "Robotiq"})  # case-insensitive coercion
        self.assertEqual(cfg.vendor, "sim")
        self.assertEqual(cfg.gripper.vendor, "robotiq")
        self.assertIsInstance(cfg.gripper, GripperConfig)

        # Unknown vendors (typos like 'ur5'/'future_vendor') are rejected EARLY, not late at create_arm.
        with self.assertRaises(ValidationError):
            RobotConfig(vendor="future_vendor")
        with self.assertRaises(ValidationError):
            GripperConfig(vendor="future_gripper")

        with self.assertRaises(ValidationError):
            RobotConfig(vendor="")
        with self.assertRaises(ValidationError):
            WorkspaceLimitsConfig(x_min=1.0, x_max=1.0)
        with self.assertRaises(ValidationError):
            GripperConfig(max_width_mm=5.0, min_width_mm=5.0)


class GraspingConfigTests(unittest.TestCase):
    def test_defaults_are_safe_and_frozen(self) -> None:
        cfg = RobotConfig()
        self.assertIsInstance(cfg.grasping, RobotGraspingConfig)
        self.assertEqual(cfg.grasping.default_mode, "auto")
        self.assertEqual(cfg.grasping.max_attempts, 5)
        self.assertFalse(cfg.grasping.closed_loop.enabled)
        self.assertFalse(cfg.grasping.verification.enabled)
        self.assertFalse(cfg.grasping.dense_recovery.enabled)
        # Frozen — direct mutation is rejected.
        with self.assertRaises(ValidationError):
            cfg.grasping.max_attempts = 7  # type: ignore[misc]

    def test_extra_fields_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            RobotGraspingConfig(typo_field=True)  # type: ignore[call-arg]
        with self.assertRaises(ValidationError):
            GraspingClosedLoopConfig(unknown_knob=1.0)  # type: ignore[call-arg]

    def test_yaml_shape_round_trips(self) -> None:
        cfg = RobotGraspingConfig(
            default_mode="closed_loop",
            max_attempts=3,
            closed_loop={
                "enabled": True,
                "pregrasp_rescan": False,
                "max_position_correction_mm": 10.0,
                "max_orientation_correction_deg": 5.0,
            },
            verification={
                "enabled": True,
                "require_object_detected": True,
                "post_lift_vision_check": True,
            },
            dense_recovery={
                "enabled": True,
                "allowed_actions": ["next_viewpoint", "next_target"],
                "max_recovery_actions": 1,
            },
        )
        self.assertEqual(cfg.default_mode, "closed_loop")
        self.assertTrue(cfg.closed_loop.enabled)
        self.assertFalse(cfg.closed_loop.pregrasp_rescan)
        self.assertEqual(
            cfg.dense_recovery.allowed_actions,
            ("next_viewpoint", "next_target"),
        )

    def test_validation_bounds(self) -> None:
        with self.assertRaises(ValidationError):
            RobotGraspingConfig(default_mode="")
        with self.assertRaises(ValidationError):
            RobotGraspingConfig(max_attempts=0)
        with self.assertRaises(ValidationError):
            RobotGraspingConfig(max_attempts=51)
        with self.assertRaises(ValidationError):
            GraspingClosedLoopConfig(max_position_correction_mm=0.0)
        with self.assertRaises(ValidationError):
            GraspingClosedLoopConfig(max_orientation_correction_deg=91.0)
        with self.assertRaises(ValidationError):
            GraspingClosedLoopConfig(target_match_iou_threshold=1.5)
        with self.assertRaises(ValidationError):
            GraspingVerificationConfig(width_delta_min_mm=-1.0)
        with self.assertRaises(ValidationError):
            GraspingVerificationConfig(vision_displacement_iou_max=1.5)
        with self.assertRaises(ValidationError):
            GraspingDenseRecoveryConfig(max_recovery_actions=-1)
        with self.assertRaises(ValidationError):
            GraspingDenseRecoveryConfig(nudge_max_offset_mm=0.0)

    def test_verification_max_width_delta_defaults_to_a_real_ceiling(self) -> None:
        """It used to default to None, and MEASURED 2026-08-09 that made the verifier blind to the case
        it exists for: a close that never executed leaves the jaws at their pre-open width (~80 mm on a
        2F-85) and, with no ceiling, that was reported PASSED -- the cell carries air to the drop-off
        and logs a success. The type stays Optional so a cell can still switch the bound off
        deliberately."""
        cfg = GraspingVerificationConfig()
        self.assertEqual(cfg.width_delta_max_mm, 10.0)
        self.assertEqual(GraspingVerificationConfig(width_delta_max_mm=5.0).width_delta_max_mm, 5.0)
        self.assertIsNone(GraspingVerificationConfig(width_delta_max_mm=None).width_delta_max_mm)

    # ------------------------------------------------------------------
    # Phase T1 — decision sub-block. Defaults preserve T0 behaviour
    # (enabled=False); enabling the block must round-trip cleanly.
    # ------------------------------------------------------------------

    def test_decision_defaults_preserve_t0_behaviour(self) -> None:
        cfg = RobotGraspingConfig()
        self.assertIsInstance(cfg.decision, GraspingDecisionConfig)
        # Disabled by default — a vanilla robot.yaml without a
        # ``decision:`` stanza must NOT activate the new decision
        # layer. This is critical for back-compat with the
        # T0-shipped configs.
        self.assertFalse(cfg.decision.enabled)
        self.assertEqual(cfg.decision.auto_uncertainty_threshold, 0.4)
        self.assertEqual(cfg.decision.max_reobservations, 2)
        self.assertEqual(cfg.decision.reasons_penalty, 0.2)
        self.assertTrue(cfg.decision.fail_closed_on_real_hardware)

    def test_decision_extra_fields_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            GraspingDecisionConfig(typo=True)  # type: ignore[call-arg]

    def test_decision_validation_bounds(self) -> None:
        with self.assertRaises(ValidationError):
            GraspingDecisionConfig(auto_uncertainty_threshold=-0.1)
        with self.assertRaises(ValidationError):
            GraspingDecisionConfig(auto_uncertainty_threshold=1.5)
        with self.assertRaises(ValidationError):
            GraspingDecisionConfig(max_reobservations=-1)
        with self.assertRaises(ValidationError):
            GraspingDecisionConfig(max_reobservations=11)
        with self.assertRaises(ValidationError):
            GraspingDecisionConfig(reasons_penalty=-0.1)
        with self.assertRaises(ValidationError):
            GraspingDecisionConfig(reasons_penalty=1.1)

    def test_decision_yaml_round_trip(self) -> None:
        cfg = RobotGraspingConfig(
            decision={
                "enabled": True,
                "auto_uncertainty_threshold": 0.25,
                "max_reobservations": 1,
                "reasons_penalty": 0.1,
                "fail_closed_on_real_hardware": False,
            }
        )
        self.assertTrue(cfg.decision.enabled)
        self.assertEqual(cfg.decision.auto_uncertainty_threshold, 0.25)
        self.assertEqual(cfg.decision.max_reobservations, 1)
        self.assertEqual(cfg.decision.reasons_penalty, 0.1)
        self.assertFalse(cfg.decision.fail_closed_on_real_hardware)


if __name__ == "__main__":
    unittest.main()

class PoseBoxZFracValidationTests(unittest.TestCase):
    """`calibration.pose_box_z_frac` must refuse the three ways it was silently wrong.

    MEASURED 2026-08-09, the day after the field shipped: a typo'd key, an inverted band and
    out-of-range fractions were ALL accepted. The typo is the worst -- `{"typo_source": [0.5, 0.9]}`
    validates, the runner's `.get(marker)` returns None, the shipped default applies, and the operator's
    setting has vanished without a word. That is the shape this repo already removed once as
    worse-than-nothing (`quality_threshold_mm`, a key documenting a gate that did not exist), and it was
    reintroduced in the very same file.
    """

    def test_the_valid_shipped_values_still_load(self) -> None:
        for band in ({"ground_truth": (0.5, 0.9)}, {"aruco": (0.55, 0.75)},
                     {"ground_truth": (0.5, 0.9), "aruco": (0.55, 0.75)}):
            with self.subTest(band=band):
                cfg = RobotCalibrationConfig(pose_box_z_frac=band)
                self.assertEqual(cfg.pose_box_z_frac, band)

    def test_an_unknown_marker_source_is_refused(self) -> None:
        """The silent case: not looked up, so the default applies and the setting disappears."""
        with self.assertRaises(ValueError) as ctx:
            RobotCalibrationConfig(pose_box_z_frac={"typo_source": (0.5, 0.9)})
        msg = str(ctx.exception)
        self.assertIn("unknown marker source", msg)
        self.assertIn("ground_truth", msg)      # the message must name what IS valid

    def test_a_band_that_is_not_a_band_is_refused(self) -> None:
        """Inverted or degenerate would give z_min >= z_max and the generator would sample nothing."""
        for band in ((0.9, 0.5), (0.5, 0.5)):
            with self.subTest(band=band), self.assertRaises(ValueError):
                RobotCalibrationConfig(pose_box_z_frac={"ground_truth": band})

    def test_values_outside_zero_to_one_are_refused(self) -> None:
        """They are FRACTIONS of workspace_limits.z_max, not millimetres."""
        for band in ((-1.0, 2.0), (0.0, 1.5), (-0.1, 0.9)):
            with self.subTest(band=band), self.assertRaises(ValueError):
                RobotCalibrationConfig(pose_box_z_frac={"ground_truth": band})

    def test_the_valid_key_set_matches_the_runner_flag(self) -> None:
        """A closed key set drifts the moment the runner grows a third --marker choice, and the drift
        would be silent again. Pin them together."""
        import src.willy_sim.run_eth_calibrate as runner

        source = Path(runner.__file__).read_text(encoding="utf-8")
        for marker in RobotCalibrationConfig.MARKER_SOURCES:
            self.assertIn(f'"{marker}"', source, f"{marker!r} is not a --marker choice any more")
