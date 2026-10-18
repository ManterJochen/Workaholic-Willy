"""UR model registry + the config validators that gate it (UR3e bring-up).

The registry (``src/robot/drivers/sim/robot_models.py``) de-locks the historically UR5e-hardcoded sim
driver: ``robot_model`` selects the Lula solver config, the Isaac USD and the cuRobo ``{key}.yml``. It shipped
without tests; these cover the contract plus the field validators on ``sim.robot_model`` and ``ur.model`` that
make a typo'd model fail at config-load instead of ~60 s into an Isaac boot or at the first motion.

Two validators once also held ``safety.self_collision.kinematics_model`` equal to those two fields. Both went
with the key in P0d D5, because the arm now reports its own model (tests/robot/test_the_arm_says_what_it_is.py).
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.config.schema.robot.sim_schema import SimConfig
from src.robot.drivers.sim.robot_models import (
    URModelSpec,
    ur_model_spec,
)


class URModelRegistryTests(unittest.TestCase):
    def test_known_models_resolve(self) -> None:
        for key in ("ur3e", "ur5e", "ur10e"):
            spec = ur_model_spec(key)
            self.assertIsInstance(spec, URModelSpec)
            self.assertEqual(spec.key, key)
            # the Lula supported-config name is the capitalised UR spelling Isaac ships
            self.assertEqual(spec.lula_key.lower(), key)
            # the USD relpath points at that model's Isaac asset dir
            self.assertIn(f"/{key}/", spec.usd_relpath)
            self.assertTrue(spec.usd_relpath.endswith(f"{key}.usd"))

    def test_case_insensitive(self) -> None:
        self.assertEqual(ur_model_spec("UR3e").key, "ur3e")
        self.assertEqual(ur_model_spec("UR5E").key, "ur5e")

    def test_unknown_model_raises_with_known_keys(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            # A real UR model, described in the arm registry with committed geometry, `admitted: false`
            # and no sim block, so this stays a negative control. It said "ur3" until 2026-09-09, when
            # the CB-series arms were added and it silently became a supported model.
            ur_model_spec("ur16e")
        self.assertIn("ur16e", str(ctx.exception))
        self.assertIn("ur5e", str(ctx.exception))  # the message lists what IS supported

    def test_curobo_descriptor_name_round_trip(self) -> None:
        """One descriptor per arm from UM lane S11; the hand is a body link the planner adds when it starts.

        Named by the arm registry since P0d D3a, which refuses an arm nobody describes with a ConfigError.
        """
        from src.config import ConfigError
        from src.config.arms import curobo_descriptor_name

        self.assertEqual(curobo_descriptor_name("ur3e"), "willy_ur3e.yml")
        self.assertEqual(curobo_descriptor_name("UR5e"), "willy_ur5e.yml")
        with self.assertRaises(ConfigError):
            curobo_descriptor_name("nope")


class RobotModelConfigValidatorTests(unittest.TestCase):
    def test_default_is_ur5e(self) -> None:
        """Every existing cell stays byte-identical: the field defaults to the historical robot."""
        self.assertEqual(SimConfig().robot_model, "ur5e")

    def test_valid_model_accepted(self) -> None:
        self.assertEqual(SimConfig(robot_model="ur3e").robot_model, "ur3e")

    def test_typo_rejected_at_config_load(self) -> None:
        for bad in ("ur16e", "UR3E ", "ur-3e", ""):
            with self.assertRaises(ValidationError, msg=f"{bad!r} should be rejected"):
                SimConfig(robot_model=bad)


class URConfigModelAdmissionTests(unittest.TestCase):
    """``ur.model`` admits only an arm the registry admits, and its default is the historical robot."""

    def test_an_unknown_ur_model_is_rejected_at_load(self) -> None:
        from src.config.schema.robot.ur_schema import URConfig

        with self.assertRaises(ValidationError) as ctx:
            URConfig.model_validate({"model": "ur16e"})
        self.assertIn("unknown UR model", str(ctx.exception))

    def test_the_default_keeps_existing_configs_unchanged(self) -> None:
        from src.config.schema.robot.ur_schema import URConfig

        self.assertEqual(URConfig().model, "ur5e")


class BakedGripperVariantTests(unittest.TestCase):
    """A model whose USD ships bare must be distinguishable BEFORE the scene is built."""

    def test_ur3e_ships_bare(self) -> None:
        # verified at byte level on-box: ur3e.usd carries neither a Gripper variant set nor a Robotiq token
        self.assertIsNone(ur_model_spec("ur3e").baked_gripper_variant)

    def test_ur5e_bakes_the_2f85(self) -> None:
        self.assertEqual(ur_model_spec("ur5e").baked_gripper_variant, "Robotiq_2f_85")


class MountedGripperRegistryTests(unittest.TestCase):
    """MOUNTED_GRIPPERS had no test at all; a bad key only surfaced as a bare KeyError at bootstrap."""

    def test_registry_keys(self) -> None:
        from src.willy_sim.grippers import MOUNTED_GRIPPERS

        self.assertEqual(set(MOUNTED_GRIPPERS), {"schunk_egu50", "schunk_ezu35", "robotiq_2f85"})

    def test_every_spec_is_self_consistent(self) -> None:
        from src.willy_sim.grippers import MOUNTED_GRIPPERS

        for key, spec in MOUNTED_GRIPPERS.items():
            self.assertEqual(spec.name, key, "registry key must equal spec.name")
            self.assertTrue(spec.usd_asset_path.startswith("/Isaac/"), spec.usd_asset_path)
            self.assertTrue(spec.base_prim_name, "a mount needs a base body path")
            self.assertEqual(len(spec.mount_rotation_matrix), 3)
            self.assertEqual(len(spec.flange_offset_mm), 3)
            self.assertGreater(spec.tcp_offset_mm, 0.0)

    def test_robotiq_2f85_mount_reproduces_the_baked_joint_frame(self) -> None:
        """MEASURED from the ur5e asset's authored `robot_gripper_joint`:
        localPos0 == (0,0,0) and localRot0 == (w=-0.5, x=0.5, y=0.5, z=0.5).
        The scene builder writes `from_rotation_matrix(mount_rotation_matrix)` as localRot0, so the spec's
        matrix must map to that quaternion (up to sign -- q and -q are the same rotation)."""
        import numpy as np

        from src.geometry.quaternion import from_rotation_matrix
        from src.willy_sim.grippers import ROBOTIQ_2F85_MOUNT

        spec = ROBOTIQ_2F85_MOUNT
        self.assertEqual(spec.flange_offset_mm, (0.0, 0.0, 0.0))  # ISO-9409-1-50 flange: zero offset
        self.assertEqual(spec.profile.driven_joint, "finger_joint")
        q = np.asarray(from_rotation_matrix(np.asarray(spec.mount_rotation_matrix, dtype=float)), dtype=float)
        baked_xyzw = np.array([0.5, 0.5, 0.5, -0.5])  # (x, y, z, w) of the measured localRot0
        same = np.allclose(q, baked_xyzw, atol=1e-6) or np.allclose(q, -baked_xyzw, atol=1e-6)
        self.assertTrue(same, f"derived {q.tolist()} is not the baked rotation {baked_xyzw.tolist()} (or its negation)")

if __name__ == "__main__":
    unittest.main()
