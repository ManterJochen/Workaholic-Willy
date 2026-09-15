"""The sim mount is a function of the hand a cell names and the arm asset it runs, not a second hand name.

``robot.sim.gripper_mount`` said which gripper Isaac mounts, beside ``robot.gripper.model`` saying which hand the
cell carries, and nothing tied the two together. The owner decided (Step 4 Q5, lane (i)) that the model is the one
name and the mount is derived from it and from the arm's USD: an asset that bakes the hand selects its variant, a
bare asset mounts the hand's standalone spec, and a hand the sim cannot mount is refused by name.

The derivation was pinned before anything read it (i3); since i4 the bootstrap, the scene and the record stamp read
it, and both keys are refused at load. The migration proof is the second test: every shipped sim chain derives
exactly the mount it wrote before i4, so switching the readers changed no cell. No test here starts Isaac.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.config import ConfigError, reload_config
from src.config.grippers import available_grippers
from src.robot.drivers.sim.robot_models import ur_model_spec

_DATA = Path(__file__).resolve().parents[1] / "config"
_ARMS = ("ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e")
_BARE = ("ur3", "ur3e", "ur5", "ur10")
_BAKED = ("ur5e", "ur10e")


class TheMountFollowsTheHandAndTheAssetTests(unittest.TestCase):
    def test_the_mount_follows_the_hand_and_the_arm_asset(self) -> None:
        from src.willy_sim.grippers import ROBOTIQ_2F85_MOUNT, SCHUNK_EGU50_MOUNT, sim_mount_for

        for arm in _BARE:
            with self.subTest(arm=arm, hand="robotiq_2f85"):
                self.assertIs(sim_mount_for(arm, "robotiq_2f85"), ROBOTIQ_2F85_MOUNT)
        for arm in _BAKED:
            with self.subTest(arm=arm, hand="robotiq_2f85"):
                self.assertIsNone(sim_mount_for(arm, "robotiq_2f85"), "a baked hand selects its variant")
        for arm in _ARMS:
            with self.subTest(arm=arm, hand="schunk_egu50"):
                self.assertIs(sim_mount_for(arm, "schunk_egu50"), SCHUNK_EGU50_MOUNT)

    def test_every_shipped_sim_chain_derives_the_mount_it_wrote_before_i4(self) -> None:
        """The migration proof. Until i4 each chain wrote its mount; these are the values it wrote, and each chain
        now derives exactly that from the hand the sim profile names."""
        from src.willy_sim.config import load_sim_config, require_robot
        from src.willy_sim.grippers import sim_mount_for

        written = {None: None, "ur3": "robotiq_2f85", "ur3e": "robotiq_2f85", "ur5": "robotiq_2f85",
                   "ur10": "robotiq_2f85", "ur10e": None}
        for arm, mount in written.items():
            with self.subTest(chain=f"sim,{arm}" if arm else "sim"):
                reload_config()
                robot = require_robot(load_sim_config(None, robot_model=arm))
                derived = sim_mount_for(robot.sim.robot_model, robot.gripper.model)
                self.assertEqual(None if derived is None else derived.name, mount)


class AHandTheSimCannotMountIsRefusedTests(unittest.TestCase):
    def test_a_hand_with_no_sim_mount_refuses_and_names_what_is_missing(self) -> None:
        from src.willy_sim.grippers import sim_mount_for

        with self.assertRaises(ConfigError) as caught:
            sim_mount_for("ur5e", "robotiq_hande")
        message = str(caught.exception)
        self.assertIn("robotiq_hande", message)
        self.assertIn("tcp_offset_mm", message)

    def test_short_and_unknown_names_refuse_through_the_registry(self) -> None:
        from src.willy_sim.grippers import sim_mount_for

        with self.assertRaises(ConfigError) as short:
            sim_mount_for("ur3e", "2f85")
        self.assertIn("robotiq_2f85", str(short.exception))
        with self.assertRaises(ConfigError) as unknown:
            sim_mount_for("ur3e", "no_such_hand")
        self.assertIn("no_such_hand", str(unknown.exception))

    def test_an_unset_hand_refuses_naming_the_key(self) -> None:
        from src.willy_sim.grippers import sim_mount_for

        with self.assertRaises(ConfigError) as caught:
            sim_mount_for("ur3e", None)
        self.assertIn("robot.gripper.model", str(caught.exception))

    def test_the_registry_is_read_from_the_cells_tree(self) -> None:
        """A cell run from its own tree must not borrow the repository's hands."""
        from src.willy_sim.grippers import sim_mount_for

        with tempfile.TemporaryDirectory(prefix="willy_mount_tree_") as tmp:
            root = Path(tmp) / "data"
            shutil.copytree(_DATA / "grippers", root / "grippers")
            (root / "grippers" / "schunk_egu50.yaml").unlink()
            with self.assertRaises(ConfigError) as caught:
                sim_mount_for("ur5e", "schunk_egu50", data_dir=root)
        self.assertIn("schunk_egu50", str(caught.exception))
        self.assertIn("willy_mount_tree_", str(caught.exception))


class TheAssetsNameTheHandTheyBakeTests(unittest.TestCase):
    def test_the_baked_hand_is_a_registry_name_exactly_where_a_variant_is_baked(self) -> None:
        hands = available_grippers()
        for arm in _ARMS:
            spec = ur_model_spec(arm)
            with self.subTest(arm=arm):
                self.assertEqual(spec.baked_hand is None, spec.baked_gripper_variant is None)
                if spec.baked_hand is not None:
                    self.assertIn(spec.baked_hand, hands)

    def test_every_mount_but_the_three_finger_one_is_a_registry_hand(self) -> None:
        """The registry describes parallel jaws only, so the EZU-35 has no name a cell can write."""
        from src.willy_sim.grippers import MOUNTED_GRIPPERS

        hands = set(available_grippers())
        self.assertLessEqual(set(MOUNTED_GRIPPERS) - {"schunk_ezu35"}, hands)
        self.assertNotIn("schunk_ezu35", hands)


# ---------------------------------------------------------------------------------------------------
# i4: the readers take the derived mount, and the two keys that named the hand a second time leave
# ---------------------------------------------------------------------------------------------------


class _ScratchTree(unittest.TestCase):
    """A copy of the shipped tree with one extra profile layer, so a removed key can be written back."""

    def setUp(self) -> None:
        from src.config.loader import set_active_profile

        self._tmp = tempfile.TemporaryDirectory(prefix="willy_i4_")
        self.root = Path(self._tmp.name) / "data"
        shutil.copytree(_DATA, self.root)
        set_active_profile(None)
        reload_config()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(reload_config)
        self.addCleanup(set_active_profile, None)

    def _layer(self, name: str, text: str) -> None:
        (self.root / "robot" / f"robot.{name}.yaml").write_text(text, encoding="utf-8")

    def _load_error(self, profile: str) -> str:
        from src.config import load_config
        from src.config.loader import set_active_profile

        set_active_profile(profile)
        reload_config()
        with self.assertRaises(ConfigError) as caught:
            load_config(self.root)
        return str(caught.exception)


class TheRemovedKeysAreRefusedTests(_ScratchTree):
    def test_a_tree_that_still_writes_the_mount_is_refused_naming_the_hand_key(self) -> None:
        self._layer("mountcheck", 'robot:\n  sim:\n    gripper_mount: "robotiq_2f85"\n')
        message = self._load_error("sim,mountcheck")
        self.assertIn("robot.sim.gripper_mount", message)
        self.assertIn("robot.gripper.model", message)
        self.assertIn("robot.mountcheck.yaml", message)
        self.assertIn("[layer: mountcheck]", message)

    def test_a_tree_that_still_writes_the_variant_is_refused_naming_the_hand_key(self) -> None:
        self._layer("variantcheck", 'robot:\n  sim:\n    gripper_variant: "Robotiq_2f_85"\n')
        message = self._load_error("sim,variantcheck")
        self.assertIn("robot.sim.gripper_variant", message)
        self.assertIn("robot.gripper.model", message)

    def test_the_schema_refuses_both_keys_at_every_door(self) -> None:
        from pydantic import ValidationError

        from src.config.schema.robot import RobotConfig

        for key, value in (("gripper_mount", None), ("gripper_variant", "Robotiq_2f_85")):
            with self.subTest(key=key), self.assertRaises(ValidationError):
                RobotConfig.model_validate({"sim": {key: value}})


class TheCellsNameTheirHandsTests(unittest.TestCase):
    def test_the_sim_cell_names_its_hand(self) -> None:
        from src.willy_sim.config import load_sim_config, require_robot

        reload_config()
        self.assertEqual(require_robot(load_sim_config(None)).gripper.model, "robotiq_2f85")

    def test_the_hand_layer_names_its_hand_over_any_chain(self) -> None:
        from src.config import load_config
        from src.willy_sim.config import load_sim_config, require_robot

        reload_config()
        self.assertEqual(load_config(profile="ur3e,hande").robot.gripper.model, "robotiq_hande")
        reload_config()
        self.assertEqual(
            require_robot(load_sim_config(None, extra_profiles=["hande"])).gripper.model, "robotiq_hande",
        )

    def test_the_real_arm_layer_names_no_hand(self) -> None:
        """The control, green before and after: robot.ur3e.yaml is also a real UR overlay and inherits no hand."""
        from src.config import load_config

        reload_config()
        self.assertIsNone(load_config(profile="ur3e").robot.gripper.model)


class _StopBoot(RuntimeError):
    """Raised by a patched step so the bootstrap never reaches Isaac."""


class TheBootstrapDerivesTheMountTests(unittest.TestCase):
    def _boot_until_the_reach_check(self, **kwargs: Any) -> object:
        """Run the bootstrap to its reach check and return the robot it would boot, or raise what it raised."""
        import unittest.mock as mock

        import src.willy_sim.harness.bootstrap as boot

        seen: list[object] = []

        def stop(robot: object) -> None:
            seen.append(robot)
            raise _StopBoot

        reload_config()
        with mock.patch.object(boot, "warn_if_unreachable", stop), self.assertRaises(_StopBoot):
            boot.bootstrap_sim_cell(kwargs.pop("data_dir", None), headless=True, **kwargs)
        return seen[0]

    def test_the_bootstrap_derives_the_mount_before_the_boot(self) -> None:
        import unittest.mock as mock

        import src.willy_sim.harness.bootstrap as boot
        from src.willy_sim import grippers

        with mock.patch.object(boot, "sim_mount_for", wraps=grippers.sim_mount_for) as derive:
            self._boot_until_the_reach_check(robot_model="ur3e")
        self.assertEqual(derive.call_args.args[:2], ("ur3e", "robotiq_2f85"))

    def test_every_shipped_arm_chain_passes_the_frame_check(self) -> None:
        """The migration proof for the refusal: no shipped sim chain declares a frame its hand's mount disagrees with,
        baked or mounted, so the check refuses no cell that booted before i4."""
        for arm in _ARMS:
            with self.subTest(arm=arm):
                robot = self._boot_until_the_reach_check(robot_model=arm)
                self.assertEqual(tuple(robot.gripper.tool_frame.offset_mm), (0.0, 132.0, 0.0))  # type: ignore[attr-defined]

    def test_a_hand_override_is_validated_before_the_boot(self) -> None:
        import src.willy_sim.harness.bootstrap as boot

        reload_config()
        with self.assertRaises(ConfigError) as caught:
            boot.bootstrap_sim_cell(None, headless=True, robot_model="ur3e", hand="2f85")
        self.assertIn("robotiq_2f85", str(caught.exception))

    def test_a_hand_with_no_sim_mount_is_refused_before_the_boot(self) -> None:
        import src.willy_sim.harness.bootstrap as boot

        reload_config()
        with self.assertRaises(ConfigError) as caught:
            boot.bootstrap_sim_cell(None, headless=True, hand="robotiq_hande")
        self.assertIn("tcp_offset_mm", str(caught.exception))

    def test_the_override_writes_the_tool_frame_the_mount_composes(self) -> None:
        robot = self._boot_until_the_reach_check(hand="schunk_egu50")
        self.assertEqual(robot.gripper.model, "schunk_egu50")  # type: ignore[attr-defined]
        self.assertEqual(tuple(robot.gripper.tool_frame.offset_mm), (0.0, 159.1, 0.0))  # type: ignore[attr-defined]

    def test_a_declared_tool_frame_that_disagrees_with_the_mount_is_refused(self) -> None:
        """Owner, lane (i) Q-D. A profile naming the EGU-50 keeps the sim's declared 2F-85 frame, 132 mm.

        The reach check is patched to stop the boot, so a bootstrap that does not refuse ends there, never in
        Isaac, and the refusal has to come before it.
        """
        import unittest.mock as mock

        import src.willy_sim.harness.bootstrap as boot

        def stop(robot: object) -> None:
            raise _StopBoot

        with tempfile.TemporaryDirectory(prefix="willy_i4_frame_") as tmp:
            root = Path(tmp) / "data"
            shutil.copytree(_DATA, root)
            (root / "robot" / "robot.egucheck.yaml").write_text(
                "robot:\n  gripper:\n    model: schunk_egu50\n", encoding="utf-8",
            )
            reload_config()
            try:
                with mock.patch.object(boot, "warn_if_unreachable", stop), self.assertRaises(ConfigError) as caught:
                    boot.bootstrap_sim_cell(str(root), headless=True, extra_profiles=["egucheck"])
            finally:
                reload_config()
        message = str(caught.exception)
        self.assertIn("159.1", message)
        self.assertIn("132", message)


class TheRecordStampFollowsTheMountTests(unittest.TestCase):
    def test_the_record_stamp_follows_the_derived_mount(self) -> None:
        from types import SimpleNamespace

        from src.willy_sim.grippers import ROBOTIQ_2F85_MOUNT
        from src.willy_sim.harness.instrumentation import cell_identity

        mounted = SimpleNamespace(sim=SimpleNamespace(robot_model="ur3e"), mount=ROBOTIQ_2F85_MOUNT,
                                  degraded_engines=())
        baked = SimpleNamespace(sim=SimpleNamespace(robot_model="ur5e"), mount=None, degraded_engines=())
        self.assertEqual(cell_identity(mounted)["gripper_mount"], "robotiq_2f85")
        self.assertEqual(cell_identity(baked)["gripper_mount"], "baked")


class TheIsaacOnlyCodeNamesNoMountKeyTests(unittest.TestCase):
    """Isaac-only modules no unit test can run; their source is held to the derived mount instead."""

    def test_the_scene_and_the_bootstrap_read_no_removed_key(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "willy_sim"
        for relative in ("scene/build.py", "harness/bootstrap.py", "harness/instrumentation.py"):
            text = (root / relative).read_text(encoding="utf-8")
            with self.subTest(file=relative):
                self.assertNotIn("gripper_mount=", text)
                self.assertNotIn(".gripper_mount", text)
                self.assertNotIn(".gripper_variant", text)

    def test_the_dense_runner_takes_a_hand(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "src" / "willy_sim" / "run_dense_pick.py"
                ).read_text(encoding="utf-8")
        self.assertNotIn("gripper_mount", text)
        self.assertIn("hand: \"str | None\" = None", text)
        # From the command line to the bootstrap: the flag, the gate's pass to build_service, build_service's to the cell.
        self.assertIn("hand=args.hand", text)
        self.assertIn("planner_owns_approach=planner_owns_approach, hand=hand,", text)
        self.assertIn("data_dir, headless=headless, hand=hand,", text)


class EveryRunnerTakesAHandTests(unittest.TestCase):
    """The runners share one cell parser; ``--hand`` reaches the bootstrap through it, and its absence changes nothing."""

    def test_the_hand_flag_reaches_the_bootstrap_kwargs(self) -> None:
        import argparse
        import inspect

        from src.willy_sim.harness.bootstrap import bootstrap_sim_cell
        from src.willy_sim.harness.cli import add_cell_arguments, cell_profile_kwargs

        parser = add_cell_arguments(argparse.ArgumentParser())
        self.assertEqual(cell_profile_kwargs(parser.parse_args(["--hand", "schunk_egu50"]))["hand"], "schunk_egu50")
        self.assertIsNone(cell_profile_kwargs(parser.parse_args([]))["hand"])
        accepted = set(inspect.signature(bootstrap_sim_cell).parameters)
        self.assertLessEqual(set(cell_profile_kwargs(parser.parse_args([]))), accepted)


if __name__ == "__main__":
    unittest.main()
