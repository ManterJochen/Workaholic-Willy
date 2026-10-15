"""A tree names ONE arm, and every reader asks that name rather than a vendor block that is always there.

Every vendor block is a ``default_factory``, so a tree materialises all of them whatever its ``vendor`` says: a
``vendor: sim`` tree carries ``ur.model = "ur5e"``, and until 2026-10-13 it carried ``ur.ip = "192.168.1.100"`` too.
Readers outside the UR driver acted on that and answered plausibly rather than saying they did not know (aurora,
which this arc is ported from, found ten; the desk planner start stamped ``arm='ur5e'`` on the report refusing a
non-UR tree).

``RobotConfig.arm`` is the one answer. These tests hold three things: it reads only the selected vendor's block, the
blocks stay loadable so nothing else moves, and it tells "this vendor drives no controller" from "this vendor drives
one and nobody said where it is", which is the difference between a cell that may run without a lock and a cell
that must not be built at all.
"""

from __future__ import annotations

import ast
import unittest
import unittest.mock
from pathlib import Path

from src.config.schema.robot import ArmSelection, RobotConfig


def _tree(**overrides: object) -> RobotConfig:
    return RobotConfig.model_validate(overrides)


class TheSelectionReadsTheSelectedBlockTests(unittest.TestCase):
    def test_a_ur_tree_names_its_model_its_address_and_its_planner(self) -> None:
        arm = _tree(vendor="ur", ur={"model": "ur3e", "ip": "10.0.0.5"}, motion={"planner": "ik"}).arm
        self.assertEqual("ur", arm.vendor)
        self.assertEqual("ur3e", arm.model)
        self.assertEqual("10.0.0.5", arm.controller)
        self.assertEqual("ik", arm.planner)
        self.assertTrue(arm.owns_a_controller)
        self.assertIn("10.0.0.5", arm.render())

    def test_a_sim_tree_carries_no_ur_address_although_the_block_is_there(self) -> None:
        """The whole point: the UR block still loads, and stops being reachable through the arm."""
        cfg = _tree(vendor="sim", ur={"ip": "10.0.0.7"})
        self.assertEqual("ur5e", cfg.ur.model, "the block itself is unchanged and still loadable")
        self.assertEqual("10.0.0.7", cfg.ur.ip)

        arm = cfg.arm
        self.assertEqual("sim", arm.vendor)
        self.assertIsNone(arm.controller, "a simulated cell reported a UR controller address")
        self.assertFalse(arm.owns_a_controller)
        self.assertNotIn("10.0.0.7", arm.render())

    def test_no_tree_names_a_controller_nobody_typed_in(self) -> None:
        """The placeholder is gone: a UR tree that never had an address typed into it names none."""
        cfg = _tree(vendor="ur")
        self.assertIsNone(cfg.ur.ip)
        self.assertIsNone(cfg.arm.controller)
        self.assertIn("NO ADDRESS DECLARED", cfg.arm.render())

    def test_the_owners_cell_keeps_its_address(self) -> None:
        """The base tree carries the owner's UR10 cell's controller, so the cell's chain names it (the owner, 2026-10-13)."""
        from src.config import load_robot_config

        selected = load_robot_config(profile="ur10,hande").arm
        self.assertEqual("ur", selected.vendor)
        self.assertEqual("192.168.1.100", selected.controller)

    def test_a_dummy_tree_owns_no_controller_and_names_no_model(self) -> None:
        arm = _tree(vendor="dummy").arm
        self.assertEqual("dummy", arm.vendor)
        self.assertEqual("", arm.model)
        self.assertIsNone(arm.controller)
        self.assertFalse(arm.owns_a_controller)

    def test_the_sim_names_the_model_it_simulates(self) -> None:
        """A sim cell is a UR in every way the planner and the guards care about, and says so by its own key."""
        arm = _tree(vendor="sim", sim={"robot_model": "ur3e"}).arm
        self.assertEqual("ur3e", arm.model)

    def test_a_vendor_with_no_block_owns_a_controller_and_declares_no_address(self) -> None:
        """The case the cell lock turns into a refusal: a real robot whose block this layer does not describe yet.

        ``franka``, ``kuka`` and ``ros2`` are reserved names that pass config validation (``core/vendor.py``). Until one
        has a block, the honest answer is "it drives a controller and nobody said where", not "it drives none", because
        the second reads as a cell that may run without a lock.
        """
        for vendor in ("franka", "kuka", "ros2"):
            with self.subTest(vendor=vendor):
                arm = _tree(vendor=vendor).arm
                self.assertTrue(arm.owns_a_controller)
                self.assertIsNone(arm.controller)
                self.assertIn("NO ADDRESS DECLARED", arm.render())


class TheLockIsKeyedOnTheSelectionTests(unittest.TestCase):
    def test_a_cell_with_a_controller_is_locked_by_vendor_and_address(self) -> None:
        from src.robot.execution.cell_lock import cell_lock_key

        self.assertEqual("ur@10.0.0.5", cell_lock_key(_tree(vendor="ur", ur={"ip": "10.0.0.5"})))

    def test_a_cell_that_drives_no_controller_takes_no_lock(self) -> None:
        from src.robot.execution.cell_lock import cell_lock_key

        for vendor in ("sim", "dummy"):
            with self.subTest(vendor=vendor):
                self.assertIsNone(cell_lock_key(_tree(vendor=vendor)))

    def test_a_cell_that_drives_a_controller_and_names_none_is_refused(self) -> None:
        """Not read as the harmless case: a robot that took no lock would let a second process command its arm."""
        from src.robot.execution.cell_lock import LockKeyRequired, cell_lock_key

        for vendor in ("ur", "franka"):
            with self.subTest(vendor=vendor), self.assertRaises(LockKeyRequired) as caught:
                cell_lock_key(_tree(vendor=vendor))
            self.assertIn("no address is declared", str(caught.exception))

    def test_the_ur_driver_refuses_to_connect_with_no_address(self) -> None:
        from src.robot.core import RobotConnectionError
        from src.robot.drivers.ur.arm import URRobotArm

        arm = URRobotArm(_tree(vendor="ur", gripper={"model": "robotiq_2f85"}))
        with self.assertRaises(RobotConnectionError) as caught:
            arm.connect()
        self.assertIn("robot.ur.ip declares no address", str(caught.exception))


class NoReaderReportsAnotherVendorsDefaultsTests(unittest.TestCase):
    """The behaviour half. A tree that names a vendor this repository has no block for reports nothing of UR's.

    ``franka`` passes config validation because it is a reserved vendor name, so the tree is built in memory.
    """

    @staticmethod
    def _foreign() -> RobotConfig:
        return RobotConfig.model_validate({"vendor": "franka"})

    def test_the_motion_stack_says_nobody_declared_a_model_rather_than_ur5e(self) -> None:
        from src.robot.safety.planning.stack import ModelSource, MotionStack

        reading = MotionStack.from_robot_config(self._foreign())

        self.assertIs(ModelSource.FALLBACK, reading.source, reading.model_source)
        self.assertIn("no model declared", reading.model_source)

    def test_the_camera_world_wiring_names_no_planner(self) -> None:
        selected = self._foreign().arm
        self.assertEqual("", selected.planner)

    def test_the_desk_planner_start_refuses_without_stamping_a_ur_model(self) -> None:
        from src.robot.execution.planner_start import PlannerStart

        report = PlannerStart(robot_config=self._foreign()).run()

        self.assertFalse(report.started)
        self.assertEqual("", report.arm, "the refusal named a UR model on a cell that has no UR in it")
        self.assertIn("franka", report.refusal)

    def test_the_exact_mesh_row_does_not_judge_a_foreign_cell_against_a_ur5e(self) -> None:
        """The guard's mesh bundle is keyed on the arm model, and the row fell back to ``robot.ur.model`` whichever
        vendor the tree named: a confident OK about the wrong robot's geometry (measured in aurora, 2026-09-20)."""
        from src.robot.execution.real_cell.preflight import CheckStatus, run_config_preflight

        report = run_config_preflight(self._foreign(), curobo_available=True, collision_engine="coal")
        row = next(c for c in report.checks if c.name == "exact mesh engine")

        self.assertIsNot(CheckStatus.OK, row.status)
        self.assertNotIn("ur5e", row.detail, "a cell with no UR in it was judged against a UR5e bundle")
        self.assertIn("names no arm model", row.detail)

    def test_the_desk_names_the_missing_controller_address(self) -> None:
        from src.robot.execution.real_cell.preflight import CheckStatus, run_config_preflight

        report = run_config_preflight(self._foreign(), curobo_available=True, collision_engine="coal")
        row = next(c for c in report.checks if c.name == "controller address")

        self.assertIs(CheckStatus.BLOCK, row.status)
        self.assertIn("franka", row.detail)

    def test_the_safety_preflight_is_handed_no_arm_model(self) -> None:
        """The guard keys its exact-mesh bundle on this: a wrong model is a wrong mesh, silently."""
        from types import SimpleNamespace

        from src.robot.safety.preflight import SafetyPreflight

        tree = SimpleNamespace(robot=self._foreign(), root=None)
        with unittest.mock.patch.object(SafetyPreflight, "from_safety_config") as built:
            SafetyPreflight.from_tree(tree)
        handed = built.call_args.kwargs.get("arm_model")
        # Named as no UR since P0e (arm_model is required), where it was left unset before.
        self.assertIsNone(handed, f"the preflight was handed {handed!r} on a cell with no UR in it")


#: Every place that still reads a field off the UR BLOCK rather than off the selection, with why it may. The list may
#: only shrink: each reader of this shape that went wrong answered confidently and wrongly on a tree with no UR in it,
#: and nothing stops the next one from being written except a list that makes it visible.
_UR_BLOCK_READS: tuple[tuple[str, int, str], ...] = (
    ("src/robot/grippers/robotiq.py", 527,
     "deliberate: the refusal contrasts the arm's own address with what this tree's UR block says, and says so; it "
     "moved with the Robotiq's refusal into its driver's spec (P0g slice 8)"),
    ("scripts/ursim/probe_halt.py", 279,
     "the halt probe records what its layer wrote beside what reached the arm: `brake_on_halt` is the UR block's own "
     "setting, which no selection answers"),
)


class NoNewReaderTakesTheUrBlockTests(unittest.TestCase):
    """The rule, by AST: a field read off ``.ur`` outside ``drivers/ur`` is listed here or it is a defect.

    The heuristic is blunt on purpose (any ``X.ur.<field>``), because the readers that went wrong were blunt: each
    reached for the block every tree materialises. A gated read is still listed, so the list says how many remain.
    """

    def test_every_direct_read_of_the_ur_block_is_one_this_list_knows(self) -> None:
        root = Path(__file__).resolve().parents[1]
        allowed = {(path, line) for path, line, _why in _UR_BLOCK_READS}
        offenders: list[str] = []
        scanned = 0
        # scripts/ and examples/ since P0e: `probe_curobo_bed.py` read a removed `.ur.` field from P0c on and nothing
        # noticed, because the served config types as Any to mypy and no rule looked there.
        for folder in ("src", "api", "willy", "datagen", "scripts", "examples"):
            for path in (root / folder).rglob("*.py"):
                rel = path.relative_to(root).as_posix()
                if rel.startswith("src/robot/drivers/ur/") or "__pycache__" in rel:
                    continue
                scanned += 1
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
                            and node.value.attr == "ur" and (rel, node.lineno) not in allowed):
                        offenders.append(f"{rel}:{node.lineno}: .ur.{node.attr} (ask RobotConfig.arm instead)")
        self.assertGreater(scanned, 200, f"only {scanned} modules scanned: the walk stopped reaching the tree")
        self.assertEqual(offenders, [])

    def test_the_list_only_shrinks(self) -> None:
        # 2 since the scan reached scripts/ (P0e): the halt probe's `brake_on_halt` was there before, not a new read.
        self.assertLessEqual(len(_UR_BLOCK_READS), 2, "a new direct read of the UR block was added to the list")

    def test_every_listed_read_is_still_there(self) -> None:
        """A read that has gone, left on the list, is a permission for the next one at that line.

        The list is keyed by line, so a stale entry is worse than a stale import debt: any new ``.ur.`` read that
        lands on that exact line later is admitted in silence (aurora found one, 2026-09-22).
        """
        root = Path(__file__).resolve().parents[1]
        stale: list[str] = []
        for path, line, _why in _UR_BLOCK_READS:
            tree = ast.parse((root / path).read_text(encoding="utf-8"))
            if not any(isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
                       and node.value.attr == "ur" and node.lineno == line for node in ast.walk(tree)):
                stale.append(f"{path}:{line} reads nothing off the UR block any more: remove it from _UR_BLOCK_READS")
        self.assertEqual(stale, [])

    def test_the_heuristic_finds_a_read_that_is_there(self) -> None:
        """The positive control: a walk that matches nothing passes the rule above without proving it."""
        node = ast.parse("model = robot_cfg.ur.model").body[0]
        assert isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute)
        self.assertTrue(isinstance(node.value.value, ast.Attribute) and node.value.value.attr == "ur")


class TheSelectionIsAReadingNotAKeyTests(unittest.TestCase):
    def test_it_does_not_appear_in_the_schema_index(self) -> None:
        """So it reaches no example tree, no ``explain`` output and no edit guard, like ``gripper.coupling_mm``.

        A ``computed_field`` would appear there and the example generator would emit it as a writable key, which is
        how a reading becomes a second place to write the same fact.
        """
        from src.config._schema_index import schema_index

        keys = schema_index()
        self.assertGreater(len(keys), 500, "the index is empty, so this proves nothing")
        self.assertEqual([key for key in keys if key.startswith("robot.arm")], [])

    def test_it_is_a_frozen_value_a_person_can_read(self) -> None:
        arm = _tree(vendor="ur").arm
        self.assertIsInstance(arm, ArmSelection)
        with self.assertRaises(Exception):
            arm.vendor = "franka"  # type: ignore[misc]
        self.assertEqual(arm.render(), arm.render(), "render() is a description, not a computation with state")


if __name__ == "__main__":
    unittest.main()
