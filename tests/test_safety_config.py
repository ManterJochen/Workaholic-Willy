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


class UncheckedPathWarningPremiseTests(unittest.TestCase):
    """The shipped-tree facts that `SafetyPreflight.gate_planned_path` cites to warn, not refuse.

    ⛔ THE COMMENT AT `preflight.py` SAID "Every profile this repository ships has
    `motion_planner: curobo`" AND ONE DOES NOT. Measured 2026-09-10 over the base tree plus all 26
    overlay layers: `ursim` ships `ik`, deliberately, because the URSim container has no GPU and the
    cuRobo sidecar cannot start there, so the fail-closed planner path would refuse every move.

    ⚠ AND THE WRONG HALF WAS THE DECORATIVE ONE. The planner value is not why the guard warns: what
    justifies a warning over a config-time refusal is that NO shipped profile turns the trajectory
    check on or declares a world, so a refusal would refuse the shipped tree. That premise is what is
    pinned here, alongside the one exception, so a profile that quietly changes either fact turns the
    comment red instead of leaving it to be read as true.
    """

    #: Profiles whose `robot.ur.motion_planner` is NOT `curobo`, and why. `ursim` is a container with
    #: no GPU; adding a row here is a decision, not a fix.
    NON_CUROBO_PROFILES = {"ursim": "ik"}

    #: Profiles that declare a planning world or a fixture; adding a name here is a decision, not a fix.
    DECLARED_WORLD_PROFILES = {
        # The Isaac reference cell's live camera world, as the owner decided: the scene's table as its support
        # plane, layered on `sim`.
        "sim_camera_world",
        # The physical cell this tree was written for, bench included.
        "ur5e",
    }

    @staticmethod
    def _profiles() -> list[str | None]:
        from src.config.tree import default_data_dir

        root = default_data_dir()
        return [None] + sorted({p.name.split(".")[-2] for p in root.rglob("*.*.yaml")})

    def _robot_configs(self) -> list[tuple[str, "RobotConfig"]]:
        from src.config.loader import load_config

        rows: list[tuple[str, RobotConfig]] = []
        for profile in self._profiles():
            robot = load_config(profile=profile).robot
            if robot is not None:
                rows.append((profile or "(base)", robot))
        return rows

    def test_the_planner_exceptions_are_exactly_the_declared_ones(self) -> None:
        """Every declared exception still is one, and a profile that does not plan says so in its own file.

        A customer's ik twin of a cell (docs/runbooks/your_own_gripper.md) is a profile this repository does not ship
        and cannot list here; MEASURED 2026-09-17 in the customer chain trial, it turned this red. What stays pinned
        is the class of defect: a profile that stopped planning by inheritance, which nobody decided.
        """
        import yaml

        from src.config.tree import default_data_dir

        measured = {
            name: robot.ur.motion_planner
            for name, robot in self._robot_configs()
            if robot.ur.motion_planner != "curobo"
        }
        for name, planner in self.NON_CUROBO_PROFILES.items():
            with self.subTest(declared=name):
                self.assertEqual(measured.get(name), planner, "a declared exception changed its motion_planner")
        for name, planner in measured.items():
            layer = default_data_dir() / "robot" / f"robot.{name}.yaml"
            raw = (yaml.safe_load(layer.read_text(encoding="utf-8")) if layer.is_file() else None) or {}
            stated = ((raw.get("robot") or {}).get("ur") or {}).get("motion_planner")
            with self.subTest(profile=name):
                self.assertEqual(stated, planner,
                                 f"{name} runs {planner!r} and its own layer does not say so: it stopped planning "
                                 f"by inheritance, which nobody decided")

    def test_no_shipped_profile_can_turn_a_planned_path_check_off(self) -> None:
        """⭐ THE PREMISE IS GONE, AND THAT IS THE POINT.

        This test used to check that no shipped profile ARMED the trajectory check, because the whole
        tree shipped with it off and a config-time refusal would have refused the tree itself. There
        is no key to arm any more: a planned path is judged on every cell, so what is left to check is
        that nothing in the tree can stand it down again.

        What every profile but the ones named in DECLARED_WORLD_PROFILES still ships without is a declared
        world and declared fixtures, which is a different and smaller claim: the guards run, and on a cell
        that declared no bench they have no bench to find.
        """
        for name, robot in self._robot_configs():
            with self.subTest(profile=name):
                self.assertFalse(
                    hasattr(robot.safety, "trajectory_check"),
                    "a profile carries a trajectory_check block again",
                )
        declared = {
            name
            for name, robot in self._robot_configs()
            if robot.safety.planning_world.enabled
            or (robot.safety.self_collision.fixtures or ())
        }
        self.assertEqual(
            declared,
            self.DECLARED_WORLD_PROFILES,
            "the profiles that declare a world or a fixture changed; say so in DECLARED_WORLD_PROFILES",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
