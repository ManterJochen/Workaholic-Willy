"""
Tests for the :class:`RobotSafetyConfig` sub-block schema.

Locks the new YAML shape:

* Defaults exist for every sub-block.
* Each sub-block honours ``StrictModel`` (``extra="forbid"``).
* Legacy flat fields (``max_tcp_velocity_mm_s``, ``workspace_margin_mm``,
  ``joint_margin_rad``, ``dwell_after_stop_s``,
  ``require_steady_before_motion``, ``steady_timeout_s``) are no longer
  accepted at the top level -- the flat-to-nested refactor retired
  them.
* Cross-field validators in :class:`JointLimitSafetyConfig`,
  :class:`PayloadSafetyConfig`, :class:`FixtureBoxConfig` fire.
* :class:`RobotConfig` loads the full new shape end-to-end via Pydantic.
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.config.schema.robot import (
    DwellSafetyConfig,
    FixtureBoxConfig,
    IkQualitySafetyConfig,
    JointLimitSafetyConfig,
    LimitsSafetyConfig,
    MotionContinuitySafetyConfig,
    PayloadSafetyConfig,
    RobotConfig,
    RobotSafetyConfig,
    SelfCollisionSafetyConfig,
)


class DefaultsTests(unittest.TestCase):
    def test_top_level_defaults(self) -> None:
        cfg = RobotSafetyConfig()
        self.assertIsInstance(cfg.limits, LimitsSafetyConfig)
        self.assertIsInstance(cfg.joint_limits, JointLimitSafetyConfig)
        self.assertIsInstance(cfg.ik_quality, IkQualitySafetyConfig)
        self.assertIsInstance(cfg.self_collision, SelfCollisionSafetyConfig)
        self.assertIsInstance(cfg.payload, PayloadSafetyConfig)
        self.assertIsInstance(cfg.motion_continuity, MotionContinuitySafetyConfig)
        self.assertIsInstance(cfg.dwell, DwellSafetyConfig)

    def test_limits_defaults(self) -> None:
        c = LimitsSafetyConfig()
        self.assertTrue(c.enforce)
        self.assertEqual(c.workspace_margin_mm, 20.0)

    def test_limits_rejects_removed_velocity_caps(self) -> None:
        # The dead vel/accel caps were removed; they must now be rejected (extra='forbid').
        for removed in (
            "max_tcp_velocity_mm_s",
            "max_tcp_acceleration_mm_s2",
            "max_joint_velocity_rad_s",
            "max_joint_acceleration_rad_s2",
        ):
            with self.assertRaises(ValidationError):
                LimitsSafetyConfig(**{removed: 100.0})

    def test_joint_limits_default_unspecified(self) -> None:
        c = JointLimitSafetyConfig()
        self.assertTrue(c.enforce)
        self.assertEqual(c.margin_deg, 5.0)
        self.assertIsNone(c.min_deg)
        self.assertIsNone(c.max_deg)

    def test_self_collision_defaults(self) -> None:
        c = SelfCollisionSafetyConfig()
        self.assertEqual(c.backend, "fcl")
        self.assertEqual(c.min_distance_mm, 10.0)
        self.assertEqual(c.fixtures, [])
        self.assertIsNone(c.mesh_dir)


class StrictExtraKeyTests(unittest.TestCase):
    """Every sub-block forbids unknown keys (StrictModel)."""

    def test_limits_rejects_extra_key(self) -> None:
        with self.assertRaises(ValidationError):
            LimitsSafetyConfig.model_validate({"unknown_field": 1.0})

    def test_top_level_rejects_legacy_flat_fields(self) -> None:
        # The legacy flat shape was retired in the nested-schema rewrite; the loader
        # MUST reject it so YAML left over from an older deployment
        # never silently behaves like an empty safety config.
        for legacy in (
            "max_tcp_velocity_mm_s",
            "max_tcp_acceleration_mm_s2",
            "max_joint_velocity_rad_s",
            "max_joint_acceleration_rad_s2",
            "workspace_margin_mm",
            "joint_margin_rad",
            "dwell_after_stop_s",
            "require_steady_before_motion",
            "steady_timeout_s",
        ):
            with self.subTest(field=legacy):
                with self.assertRaises(ValidationError):
                    RobotSafetyConfig.model_validate({legacy: 1.0})


class CrossFieldValidatorTests(unittest.TestCase):
    def test_joint_limits_min_max_must_be_paired(self) -> None:
        with self.assertRaises(ValidationError):
            JointLimitSafetyConfig.model_validate(
                {"min_deg": [-180.0, -180.0]}  # max_deg missing
            )

    def test_joint_limits_min_max_length_must_match(self) -> None:
        with self.assertRaises(ValidationError):
            JointLimitSafetyConfig.model_validate(
                {"min_deg": [-180.0, -180.0], "max_deg": [180.0]}
            )

    def test_joint_limits_min_must_be_less_than_max(self) -> None:
        with self.assertRaises(ValidationError):
            JointLimitSafetyConfig.model_validate(
                {"min_deg": [180.0], "max_deg": [-180.0]}
            )

    def test_payload_mass_must_not_exceed_max(self) -> None:
        with self.assertRaises(ValidationError):
            PayloadSafetyConfig.model_validate(
                {"mass_kg": 10.0, "max_mass_kg": 5.0}
            )

    def test_payload_inertia_must_be_non_negative(self) -> None:
        with self.assertRaises(ValidationError):
            PayloadSafetyConfig.model_validate(
                {"inertia_kgm2": (-0.1, 0.0, 0.0)}
            )

    def test_fixture_half_extents_must_be_non_negative(self) -> None:
        with self.assertRaises(ValidationError):
            FixtureBoxConfig.model_validate(
                {"name": "shelf", "half_extents_mm": (1.0, -1.0, 1.0)}
            )


class YamlShapeIntegrationTests(unittest.TestCase):
    """End-to-end load of the new shape through :class:`RobotConfig`."""

    def test_full_shape_loads(self) -> None:
        payload = {
            "vendor": "ur",
            "safety": {
                "limits": {
                    "enforce": True,
                    "workspace_margin_mm": 30.0,
                },
                "joint_limits": {
                    "enforce": True,
                    "margin_deg": 3.0,
                    "min_deg": [-180.0] * 6,
                    "max_deg": [180.0] * 6,
                },
                "ik_quality": {
                    "enforce": True,
                    "max_jump_rad": 0.5,
                },
                "self_collision": {
                    "enforce": True,
                    "backend": "capsule",
                    "min_distance_mm": 15.0,
                    "fixtures": [
                        {
                            "name": "table",
                            "center_mm": (0.0, 0.0, -100.0),
                            "half_extents_mm": (500.0, 500.0, 10.0),
                        }
                    ],
                },
                "payload": {
                    "enforce": True,
                    "mass_kg": 1.2,
                    "max_mass_kg": 5.0,
                },
                "motion_continuity": {
                    "enforce": True,
                    "max_joint_step_deg": 30.0,
                },
                "dwell": {
                    "require_steady_before_motion": True,
                },
            },
        }
        cfg = RobotConfig.model_validate(payload)
        self.assertEqual(cfg.safety.limits.workspace_margin_mm, 30.0)
        self.assertEqual(cfg.safety.joint_limits.min_deg, [-180.0] * 6)
        self.assertEqual(
            cfg.safety.self_collision.fixtures[0].name, "table",
        )
        self.assertEqual(cfg.safety.payload.mass_kg, 1.2)


class DefaultYamlLoadsTests(unittest.TestCase):
    """The shipped robot.yaml + robot.web.yaml must load against the new
    schema. This is the regression test for the nested-schema YAML rewrite."""

    def test_default_robot_yaml(self) -> None:
        from src.config.loader import load_config
        cfg = load_config()
        # All sub-blocks present and typed.
        self.assertIsInstance(cfg.robot.safety, RobotSafetyConfig)
        self.assertGreaterEqual(cfg.robot.safety.limits.workspace_margin_mm, 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
