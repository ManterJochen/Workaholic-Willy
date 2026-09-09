"""UR model registry + the config validators that gate it (UR3e bring-up).

The registry (``src/robot/drivers/sim/robot_models.py``) de-locks the historically UR5e-hardcoded sim
driver: ``robot_model`` selects the Lula solver config, the Isaac USD and the cuRobo ``{key}.yml``. It shipped
without tests; these cover the contract plus the two config validators that make a typo'd / half-migrated cell
fail at config-load instead of ~60 s into an Isaac boot (or, worse, silently against the wrong DH chain).
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.config.schema.robot.sim_schema import UR_MODEL_KEYS, SimConfig
from src.robot.drivers.sim.robot_models import (
    URModelSpec,
    curobo_robot_yml,
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
            # A real UR model with a bundled DH row that UR_MODEL_KEYS deliberately does not
            # admit, so this stays a negative control. It said "ur3" until 2026-09-09, when
            # the CB-series arms were added and it silently became a supported model.
            ur_model_spec("ur16e")
        self.assertIn("ur16e", str(ctx.exception))
        self.assertIn("ur5e", str(ctx.exception))  # the message lists what IS supported

    def test_curobo_robot_yml_round_trip(self) -> None:
        self.assertEqual(curobo_robot_yml("ur3e"), "ur3e.yml")
        self.assertEqual(curobo_robot_yml("ur5e"), "ur5e.yml")
        self.assertEqual(curobo_robot_yml("UR3e"), "ur3e.yml")
        with self.assertRaises(ValueError):
            curobo_robot_yml("nope")

    def test_registry_and_schema_key_sets_match(self) -> None:
        """Drift guard: the config layer cannot import the driver package (it is the bottom of the dependency
        stack), so the allowed-model tuple is duplicated in ``sim_schema``. Keep the two in lockstep."""
        from src.robot.drivers.sim.robot_models import _UR_MODELS

        self.assertEqual(
            set(_UR_MODELS), set(UR_MODEL_KEYS),
            "sim_schema.UR_MODEL_KEYS drifted from robot_models._UR_MODELS -- update both.",
        )


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


class SelfCollisionModelCouplingTests(unittest.TestCase):
    """safety.self_collision.kinematics_model must match sim.robot_model on an ENABLED sim cell."""

    def _robot_config(self, *, robot_model: str, kinematics_model: str, sim_enabled: bool = True):
        from src.config.schema.robot.robot_schema import RobotConfig

        # vendor="sim" is load-bearing here, not decoration: the REAL-UR twin of this validator
        # couples kinematics_model to ur.model, and a fixture that leaves vendor at its default would be
        # judged as a UR cell instead of the sim cell these tests are about.
        return RobotConfig.model_validate({
            "vendor": "sim",
            "sim": {"enabled": sim_enabled, "robot_model": robot_model},
            "safety": {"self_collision": {"kinematics_model": kinematics_model}},
        })

    def test_matching_pair_is_accepted(self) -> None:
        cfg = self._robot_config(robot_model="ur3e", kinematics_model="ur3e")
        self.assertEqual(cfg.sim.robot_model, "ur3e")
        self.assertEqual(cfg.safety.self_collision.kinematics_model, "ur3e")

    def test_mismatch_raises_naming_both_keys(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            self._robot_config(robot_model="ur3e", kinematics_model="ur5e")
        msg = str(ctx.exception)
        self.assertIn("kinematics_model", msg)
        self.assertIn("robot_model", msg)

    def test_mismatch_inert_when_sim_disabled(self) -> None:
        """A real-robot config leaves the sim block inert -- the coupling must not fire there."""
        cfg = self._robot_config(robot_model="ur5e", kinematics_model="ur3e", sim_enabled=False)
        self.assertEqual(cfg.safety.self_collision.kinematics_model, "ur3e")



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


class RealUrSelfCollisionModelCouplingTests(unittest.TestCase):
    """The same coupling for a REAL UR cell -- with a physical arm on the other end.

    ``ur.model`` and ``safety.self_collision.kinematics_model`` are independent hand-edited keys, and the
    guard feeds the latter into the UR DH table. A half-migrated cell would evaluate every real motion
    against another robot's link lengths (ur5e a2/a3 -425/-392.2 mm vs ur3e -243.55/-213.2) and return
    wrong self-collision verdicts.
    """

    @staticmethod
    def _ur_config(*, model: str, kinematics_model: str, vendor: str = "ur"):
        from src.config.schema.robot.robot_schema import RobotConfig

        return RobotConfig.model_validate({
            "vendor": vendor,
            "ur": {"model": model},
            "safety": {"self_collision": {"kinematics_model": kinematics_model}},
        })

    def test_matching_pair_is_accepted(self) -> None:
        cfg = self._ur_config(model="ur3e", kinematics_model="ur3e")
        self.assertEqual(cfg.ur.model, "ur3e")

    def test_mismatch_raises_naming_both_keys(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            self._ur_config(model="ur3e", kinematics_model="ur5e")
        msg = str(ctx.exception)
        self.assertIn("kinematics_model", msg)
        self.assertIn("ur.model", msg)

    def test_the_ur_block_is_inert_boilerplate_on_a_non_ur_cell(self) -> None:
        """A sim or dummy cell's ``ur`` block is not a claim about its arm, so this rule must not fire on
        it -- holding an inert block to the arm's model would reject perfectly valid configs."""
        cfg = self._ur_config(model="ur5e", kinematics_model=None, vendor="dummy")
        self.assertEqual(cfg.ur.model, "ur5e")

    def test_a_ur_dh_table_is_still_rejected_on_that_cell(self) -> None:
        """SUPERSEDED PREMISE, recorded rather than deleted: this case used to be ACCEPTED, because the
        only rules coupled kinematics_model to sim.robot_model and to ur.model and neither fires on a
        dummy cell. A third, vendor-agnostic arm now catches it -- `kinematics_model` selects a UR DH
        table, and there is no correct value for a non-UR arm. The gap was reachable in practice:
        WILLY_PROFILE=sim,web composed vendor='kuka' with kinematics_model='ur5e' and loaded cleanly."""
        with self.assertRaises(ValidationError) as ctx:
            self._ur_config(model="ur5e", kinematics_model="ur3e", vendor="dummy")
        self.assertIn("Universal Robots DH table", str(ctx.exception))

    def test_an_unknown_ur_model_is_rejected_at_load(self) -> None:
        from src.config.schema.robot.ur_schema import URConfig

        with self.assertRaises(ValidationError) as ctx:
            URConfig.model_validate({"model": "ur16e"})
        self.assertIn("unknown UR model", str(ctx.exception))

    def test_the_default_keeps_existing_configs_unchanged(self) -> None:
        from src.config.schema.robot.ur_schema import URConfig

        self.assertEqual(URConfig().model, "ur5e")
