"""The arm says which robot it is, and every safety reader takes it from there through one rule (P0d D5).

Until D5 the Isaac arm answered ``isaac-sim``, which no registry describes, and a cell stated its arm a second time
in ``robot.safety.self_collision.kinematics_model`` so the guard had a chain to judge. Two validators held the two
names equal and a third refused the key on any other arm, which is the whole reason a second name could be deleted:
it never said anything the first did not.

What replaces it is one function, ``_ur_kinematics.described_ur_model``: a model is a UR where the arm registry
describes one by that name. The guard asks it of the model the arm reports, the build-time hand refusal and the
desk checklist ask it of the model the cell names. These tests pin that the three agree, that the Isaac arm names
its UR in both modes, that each sim builder still hands the guard its arm, and that the continuous monitor watches
the arm the guard judges rather than a UR5e by default. The last class pins that a tree still writing the second
name is refused at load with the sentence that says where the arm is named now, and that no removed key is still
a field of the schema.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pydantic import ValidationError

from src.config import load_robot_config
from src.config._schema_index import SchemaField, schema_index
from src.config.arms import available_arms
from src.config.schema._removed import REMOVED_KEYS
from src.config.schema.robot import RobotConfig, SelfCollisionSafetyConfig
from src.robot.drivers.sim.arm import ISAAC_CAPABILITIES, IsaacRobotArm
from src.robot.drivers.sim.config import SimRobotConfig
from src.robot.execution.real_cell import CheckStatus, run_config_preflight
from src.robot.safety._ur_kinematics import described_ur_model
from tests.willy_sim.test_sim_mount_is_derived import _ScratchTree

_ROOT = Path(__file__).resolve().parents[2]


def _isaac(model: str = "ur5e", *, mock_mode: bool = False) -> IsaacRobotArm:
    """An Isaac arm built as the sim builds one, which constructs without Isaac: nothing here connects."""
    return IsaacRobotArm(SimRobotConfig(enabled=True, scene="placeholder.usd", robot_prim_path="/World/Robot",
                                        robot_model=model, mock_mode=mock_mode))


class TheOneRuleTests(unittest.TestCase):
    def test_every_described_arm_is_a_ur_by_its_own_name(self) -> None:
        """Described, not admitted: ur16e and ur20 have link lengths, and the guard judges what has link lengths."""
        for model in available_arms():
            with self.subTest(model=model):
                self.assertEqual(model, described_ur_model(model))

    def test_what_nobody_describes_is_no_ur(self) -> None:
        for model in ("isaac-sim", "dummy-6dof", "kr6-r900", "", None):
            with self.subTest(model=model):
                self.assertIsNone(described_ur_model(model))


class TheIsaacArmNamesItsUrTests(unittest.TestCase):
    def test_it_reports_the_ur_it_drives(self) -> None:
        for model in ("ur5e", "ur3e", "ur10"):
            with self.subTest(model=model):
                capabilities = _isaac(model).capabilities
                self.assertEqual(model, capabilities.model)
                self.assertEqual("sim", capabilities.vendor, "it is still the simulator, and says so")

    def test_the_mock_names_it_too_and_drops_only_the_kinematics(self) -> None:
        """The mock arm is what CI judges: an arm that named its UR only when Isaac ran would pass here and differ there."""
        capabilities = _isaac("ur3e", mock_mode=True).capabilities
        self.assertEqual("ur3e", capabilities.model)
        self.assertFalse(capabilities.has_native_fk)

    def test_the_default_arm_is_the_constant(self) -> None:
        self.assertEqual(ISAAC_CAPABILITIES, _isaac().capabilities)

    def test_the_guard_judges_the_sim_profile_s_arm_as_the_ur_it_is_and_turns_it_as_before(self) -> None:
        """The byte identity of the shipped sim cell, at the level the guard sees it. The key named ur5e and
        the guard turned its chain by 180 degrees; the arm names ur5e and the guard turns it by the same 180.
        """
        from src.willy_sim.config import load_sim_config, sim_driver_config, sim_safety_preflight

        for model in ("ur5e", "ur3e"):
            with self.subTest(model=model):
                cfg = load_sim_config(robot_model=model)
                # The arm is built from the tree, as the sim builds it: an arm handed the expected model directly
                # would pass whatever the profile's sim.robot_model said.
                arm = IsaacRobotArm(sim_driver_config(cfg))
                self.assertEqual((model, 180.0), sim_safety_preflight(cfg).self_kinematics(arm))


class EverySimBuilderHandsTheGuardItsArmTests(unittest.TestCase):
    """The refusal that would have gone silently. With the key gone, the only build-time source of the arm the
    guard reads hand geometry on is ``arm_model``. A builder that forgets it builds a guard with no hand refusal:
    no "robot.gripper.model is unset", no placement refusal, and the arm bundle's baked 2F-85 judged instead.
    """

    def test_every_call_outside_the_preflight_passes_arm_model(self) -> None:
        missing: list[str] = []
        seen = 0
        for base in ("src", "api", "willy", "datagen", "examples", "scripts"):
            for path in sorted((_ROOT / base).rglob("*.py")):
                rel = path.relative_to(_ROOT).as_posix()
                if "__pycache__" in rel or rel == "src/robot/safety/preflight.py":
                    continue
                text = path.read_text(encoding="utf-8")
                if "from_safety_config(" not in text:
                    continue
                for node in ast.walk(ast.parse(text)):
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "from_safety_config"):
                        seen += 1
                        # The keyword alone is not enough: `arm_model=UNSET` is present and hands the guard nothing.
                        # The value must read an arm's model off the config the builder holds.
                        passed = [ast.unparse(k.value) for k in node.keywords if k.arg == "arm_model"]
                        if not passed or not any(".model" in value and "UNSET" != value for value in passed):
                            missing.append(f"{rel}:{node.lineno} arm_model={passed[0] if passed else '(absent)'}")
        self.assertGreaterEqual(seen, 4, "the walk stopped finding the builders it exists for")
        self.assertEqual([], missing)


class TheContinuousMonitorWatchesTheJudgedArmTests(unittest.TestCase):
    def _wire(self, arm: object) -> mock.MagicMock:
        from src.robot.safety import continuous_monitor
        from src.willy_sim.run_dense_pick import wire_safety_guards

        with mock.patch.object(continuous_monitor.ContinuousCollisionMonitor, "from_model", return_value=None) as built:
            wire_safety_guards(arm, natural_aim=False, continuous_guard=True)
        return built

    def test_it_takes_the_model_and_the_yaw_from_the_arm_s_own_preflight(self) -> None:
        """Until D5 these were parameters defaulting to ur5e and 180: a ur3e cell was watched with UR5e links."""
        preflight = SimpleNamespace(self_kinematics=lambda arm: ("ur3e", 180.0))
        built = self._wire(SimpleNamespace(safety_preflight=preflight))
        self.assertEqual(("ur3e", 180.0), built.call_args.args[:2])

    def test_an_arm_with_nothing_to_watch_gets_no_monitor(self) -> None:
        built = self._wire(SimpleNamespace(safety_preflight=None))
        built.assert_not_called()


class TheDeskNamesTheArmTheGuardJudgesTests(unittest.TestCase):
    """The self-collision row said "set kinematics_model explicitly" until D5, and WARNed on every tree without it."""

    @staticmethod
    def _row(robot: RobotConfig):
        report = run_config_preflight(robot, curobo_available=True, collision_engine="coal")
        (row,) = [check for check in report.checks if check.name == "self-collision"]
        return row

    def test_a_ur_cell_is_judged_as_its_ur(self) -> None:
        row = self._row(load_robot_config(profile="ur3e"))
        self.assertIs(CheckStatus.OK, row.status, row.detail)
        self.assertIn("ur3e", row.detail)
        self.assertIn("robot.ur block", row.detail)
        self.assertNotIn("kinematics_model", row.detail + row.fix)

    def test_a_sim_cell_is_judged_as_the_ur_it_drives(self) -> None:
        row = self._row(load_robot_config(profile="sim"))
        self.assertIs(CheckStatus.OK, row.status, row.detail)
        self.assertIn("ur5e", row.detail)
        self.assertIn("robot.sim block", row.detail)

    def test_the_dummy_is_told_it_carries_no_guard(self) -> None:
        row = self._row(RobotConfig.model_validate({"vendor": "dummy"}))
        self.assertIs(CheckStatus.WARN, row.status)
        self.assertIn("no safety preflight", row.detail)


_THE_SECOND_NAME = "robot.safety.self_collision.kinematics_model"


def _same_key(removed: str, indexed: str) -> bool:
    """One dotted key against one schema index path. A ``*`` on either side stands for one map key, and the index's
    ``[]`` marks a list item on the segment before it rather than being a segment of its own."""
    a, b = removed.split("."), indexed.replace("[]", "").split(".")
    return len(a) == len(b) and all(x == y or "*" in (x, y) for x, y in zip(a, b))


def _removed_keys_still_in(index: "dict[str, SchemaField]") -> list[str]:
    """One sentence per REMOVED_KEYS entry that the schema still accepts, which is an entry refusing nothing."""
    return [f"{removed} is still a field of the schema, as {path}"
            for removed in sorted(REMOVED_KEYS) for path in sorted(index) if _same_key(removed, path)]


def _inside_a_removed_block(removed: str) -> bool:
    """Whether ``removed`` sits in a block that left whole, with an entry of its own (Willy's grasp verification and
    dense recovery): that block's entry is what the index is held to, and its keys have no block left to sit in."""
    parts = removed.split(".")
    return any(".".join(parts[:end]) in REMOVED_KEYS for end in range(1, len(parts)))


def _blocks_the_index_never_reached(index: "dict[str, SchemaField]") -> list[str]:
    """Every removed key whose block has no field in the index: its absence there would say nothing."""
    parents = [path.rsplit(".", 1)[0] for path in index if "." in path]
    return [removed for removed in sorted(REMOVED_KEYS)
            if not _inside_a_removed_block(removed)
            and not any(_same_key(removed.rsplit(".", 1)[0], parent) for parent in parents)]


class TheSecondNameIsRefusedAtLoadTests(_ScratchTree):
    """D5b. The key left the schema, and a tree written before D5 still writes it: refused, never ignored.

    Refused rather than ignored because an ignored key reads as a setting that holds. A cell that kept
    ``kinematics_model: "ur3e"`` on a ur5e would believe it chose its guard's arm, and the guard would judge the ur5e.
    """

    def test_a_tree_that_still_writes_it_is_refused_naming_both_vendor_blocks(self) -> None:
        self._layer("armkey", 'robot:\n  safety:\n    self_collision:\n      kinematics_model: "ur5e"\n')
        message = self._load_error("sim,armkey")
        self.assertIn(_THE_SECOND_NAME, message)
        self.assertIn("removed on purpose", message)
        self.assertIn(REMOVED_KEYS[_THE_SECOND_NAME], message)
        self.assertIn("robot.ur.model", message)
        self.assertIn("robot.sim.robot_model", message)
        self.assertIn("robot.armkey.yaml", message)
        self.assertIn("[layer: armkey]", message)

    def test_a_tree_that_writes_it_as_null_is_refused_the_same_way(self) -> None:
        """Every customer tree generated from data_example before D5 carried ``kinematics_model: null``, and a null
        is still the key: a tree that set it to nothing is refused with the same sentence."""
        self._layer("nullkey", "robot:\n  safety:\n    self_collision:\n      kinematics_model: null\n")
        message = self._load_error("nullkey")
        self.assertIn(_THE_SECOND_NAME, message)
        self.assertIn("removed on purpose", message)
        self.assertIn("robot.ur.model", message)
        self.assertIn("robot.sim.robot_model", message)

    def test_the_schema_refuses_the_key(self) -> None:
        with self.assertRaises(ValidationError):
            SelfCollisionSafetyConfig.model_validate({"kinematics_model": "ur5e"})


class NoRemovedKeyIsStillAFieldTests(unittest.TestCase):
    """An entry whose field stayed is A Sentence nobody reads. REMOVED_KEYS only speaks on ``extra_forbidden``, so
    an entry for a key the schema still accepts never fires, and the key keeps loading with the meaning it had.
    Nothing checked that until D5b; this walks every entry against the schema index.
    """

    def test_every_removed_key_sits_in_a_block_the_index_reaches(self) -> None:
        """The control on the walk: the index is depth capped, and a block it never reached would pass below."""
        self.assertEqual([], _blocks_the_index_never_reached(schema_index()))

    def test_no_removed_key_is_still_a_field_of_the_schema(self) -> None:
        self.assertEqual([], _removed_keys_still_in(schema_index()))

    def test_a_removed_key_left_in_the_schema_is_caught(self) -> None:
        """The negative control: the same check, over an index with the D5 field put back where it was."""
        index = dict(schema_index())
        index[_THE_SECOND_NAME] = index["robot.safety.self_collision.backend"]
        self.assertEqual([f"{_THE_SECOND_NAME} is still a field of the schema, as {_THE_SECOND_NAME}"],
                         _removed_keys_still_in(index))

    def test_a_star_entry_is_caught_under_the_index_s_star_and_under_a_named_key(self) -> None:
        index = dict(schema_index())
        field = index["robot.safety.self_collision.backend"]  # any field, planted where the removed star entry was
        for path in ("robot.grasping.fusion.cameras.*.mounting_mode", "robot.grasping.fusion.cameras.left.mounting_mode"):
            with self.subTest(path=path):
                found = _removed_keys_still_in({**index, path: field})
                self.assertEqual([f"robot.grasping.fusion.cameras.*.mounting_mode is still a field of the schema, "
                                  f"as {path}"], found)


if __name__ == "__main__":
    unittest.main()
