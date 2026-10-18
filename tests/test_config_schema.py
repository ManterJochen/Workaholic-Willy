from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.config.schema.camera import (
    CameraSystemConfig,
    EyeInHandWorkflowConfig,
    EyeToHandWorkflowConfig,
    HandEyeConfig,
    WebcamPairRigConfig,
)
from src.config.schema.robot import (
    GripperConfig,
    RobotConfig,
    RobotGraspingConfig,
    WorkspaceLimitsConfig,
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

    def test_camera_system_validates_the_primary_rig_contract(self) -> None:
        """`primary_rig_id` is required and must resolve. It replaced `active_mode` + `active_rig_id`,
        an optional pair only the stereo calibration tool read, while the cell took the first RGB-D
        rig by list position."""
        CameraSystemConfig(primary_rig_id="a", rigs=[_rig("a")])

        with self.assertRaises(ValidationError):
            CameraSystemConfig(rigs=[_rig("a")])  # naming it is not optional
        with self.assertRaises(ValidationError):
            CameraSystemConfig(primary_rig_id="missing", rigs=[_rig("a")])
        with self.assertRaises(ValidationError):
            CameraSystemConfig(primary_rig_id="a", rigs=[_rig("a"), _rig("a")])

    def test_a_disabled_primary_LOADS_and_is_refused_where_the_camera_is_opened(self) -> None:
        """⚠ DELIBERATELY NOT A LOAD ERROR. `cam.tiltcam.yaml` ships both of its rigs off until an
        operator fills in their serial numbers, and a profile that is not ready to RUN is not a file
        that is malformed. `build_real_components` refuses it where the message can say what to
        switch on."""
        CameraSystemConfig(primary_rig_id="a", rigs=[_rig("a", enabled=False)])

    def test_camera_rig_numeric_validation(self) -> None:
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
        self.assertEqual(cfg.grasping.default_mode, "dense_clutter")
        # `verification` and `dense_recovery` left the schema on 2026-09-29, `recovery` on 2026-10-16: every cell
        # recovers as the owner's cell did.
        for gone in ("verification", "dense_recovery", "recovery"):
            with self.subTest(block=gone):
                self.assertNotIn(gone, RobotGraspingConfig.model_fields)
        # Frozen — direct mutation is rejected.
        with self.assertRaises(ValidationError):
            cfg.grasping.default_mode = "easy"  # type: ignore[misc]

    def test_extra_fields_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            RobotGraspingConfig(typo_field=True)  # type: ignore[call-arg]

    def test_yaml_shape_round_trips(self) -> None:
        cfg = RobotGraspingConfig(default_mode="easy")
        self.assertEqual(cfg.default_mode, "easy")
        # The two blocks removed on 2026-09-29, and the ten of 2026-10-16, are unknown keys now, whatever they hold.
        for block in ("verification", "dense_recovery", "decision", "feasibility", "occlusion", "ordering",
                      "uncertainty", "success_model", "watchdog", "performance", "approach_validation", "recovery"):
            with self.subTest(block=block):
                with self.assertRaises(ValidationError):
                    RobotGraspingConfig(**{block: {"enabled": False}})

    def test_validation_bounds(self) -> None:
        with self.assertRaises(ValidationError):
            RobotGraspingConfig(default_mode="")
        # The attempts a pick makes left on 2026-10-17 at the cell's 5: the key is refused whatever it holds.
        for attempts in (5, 0, 51):
            with self.subTest(attempts=attempts), self.assertRaises(ValidationError):
                RobotGraspingConfig(max_attempts=attempts)  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
