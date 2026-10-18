"""The test files that build a ``URRobotArm`` by hand, counted, so the count only falls (P0g slice 3).

P0g moves the core tests onto ``PlannedArm`` over a fake transport (slice 13): a test that checks the planning half
should not need a UR, and every one that builds a UR to check a core rule is a test that cannot run for another
vendor. Measured in Willy before slice 4 (aurora's ``ccda597d``): 55 files call ``URRobotArm(`` at 125 sites, two
test helpers among them. Slice 13 lowered it step by step: 53 files and 123 sites once the shared helpers moved to
``tests/_arm_kit.py``, 46 and 110 once the path, line and goal judges moved, 45 and 108 after the gate fence, 37 and
90 after the planner binding, 36 and 74 after the preflight build (which added the one file that pins the kit's
recipe to the UR driver's, ``test_the_kit_builds_what_the_ur_driver_builds.py``, 3 sites), and 35 and 59 after the
UR's own files (``test_ur_arm.py`` one build more, for the UR's home on a fresh control program, whose tests moved
there from the refused-home file because the restart is the UR's own), and 32 and 50 once connect's neutral half
became the planned arm's (slice 15; ``test_ur_tool_frame.py`` rose by one site in the same slice, a builder its new
tests of the UR's answers to connect's hooks share), and 31 and 49 once the joint route's twin and plan-around tests
stood on the kit as aurora's do (port S2), and 30 and 47 once hand guiding was a mixin of the planned arm (port S4).
Each file is pinned at its count. A file that
builds more than it did, or a file not listed that starts building one, fails here; a file that builds fewer must
lower its number in the same commit, so the list stays the measure and never becomes a ceiling nobody reads.

A test that checks the UR itself (its wire, its size classes, its tool register) keeps its UR: the list only asks that
a new one is a decision, not a habit.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

_TESTS = Path(__file__).resolve().parent

#: File -> ``URRobotArm(`` call sites, measured before slice 4.
#: The list may only shrink: lower a number when a file builds fewer, remove a file that builds none.
_BUILD_A_UR_ARM: dict[str, int] = {
    "_cell_2026_10_01.py": 1,
    "_halt_fakes.py": 1,
    "test_a_bin_the_camera_saw_beside_the_base_is_the_exact_guards.py": 1,
    "test_a_cable_is_never_wound_past_half_a_turn_of_home.py": 1,
    "test_a_generated_view_drives_its_straight_joint_line_or_nothing.py": 1,
    "test_a_grasp_judges_its_lift_carrying_before_it_closes.py": 1,
    "test_a_grasp_with_no_goal_on_the_arm_s_branch_takes_its_half_turned_twin.py": 1,
    "test_a_joint_path_is_judged_whole_and_sent_without_a_pause.py": 1,
    "test_a_motion_waits_for_steady_right_before_it_is_sent.py": 1,
    "test_a_ur_arm_names_the_configuration_a_pose_goes_to.py": 1,
    "test_a_whole_path_is_judged_as_every_sample_is.py": 2,
    "test_every_verb_meets_the_camera_world.py": 1,
    "test_robot.py": 2,
    "test_the_arm_says_who_it_is.py": 1,
    "test_the_band_admission_judges_with_the_boxes_the_camera_saw.py": 1,
    "test_the_cell_names_its_planner.py": 1,
    "test_the_desk_names_the_gripper_driver.py": 1,
    "test_the_exact_guard_decides_the_planners_self_pairs.py": 1,
    "test_the_gripper_dispatch_holds_its_golden.py": 1,
    "test_the_guard_judges_the_arm_it_was_built_for.py": 1,
    "test_the_guide_describes_the_registry_as_it_is.py": 1,
    "test_the_hand_is_judged_where_the_controller_puts_the_flange.py": 1,
    "test_the_kit_builds_what_the_ur_driver_builds.py": 3,
    "test_the_planner_turns_as_the_description_says.py": 1,
    "test_the_tree_names_one_arm.py": 1,
    "test_ur_arm.py": 4,
    "test_ur_arm_builds_no_gripper.py": 1,
    "test_ur_io_force_status.py": 1,
    "test_ur_tool_frame.py": 10,
    "test_what_a_test_injects_reaches_the_arm.py": 1,
}


def _sites(source: str) -> int:
    """``URRobotArm(...)`` calls, by name or by attribute (``ur_arm.URRobotArm(...)``)."""
    return sum(1 for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call) and (
        (isinstance(node.func, ast.Name) and node.func.id == "URRobotArm")
        or (isinstance(node.func, ast.Attribute) and node.func.attr == "URRobotArm")))


def _measured() -> dict[str, int]:
    found: dict[str, int] = {}
    for path in sorted(_TESTS.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        count = _sites(path.read_text(encoding="utf-8"))
        if count:
            found[path.relative_to(_TESTS).as_posix()] = count
    return found


class TheTestsThatBuildAUrArmOnlyShrinkTests(unittest.TestCase):
    def test_no_file_builds_more_than_it_did(self) -> None:
        measured = _measured()
        grown = {name: (count, _BUILD_A_UR_ARM.get(name, 0)) for name, count in measured.items()
                 if count > _BUILD_A_UR_ARM.get(name, 0)}
        self.assertEqual({}, grown, "(now, listed): build a PlannedArm over the fake transport instead, or, for a "
                                    "test of the UR itself, raise the number with the reason in the commit")

    def test_the_list_is_the_measure(self) -> None:
        """A number above what the file builds would admit a new UR build later without anyone deciding it."""
        measured = _measured()
        stale = {name: (measured.get(name, 0), listed) for name, listed in _BUILD_A_UR_ARM.items()
                 if measured.get(name, 0) != listed}
        self.assertEqual({}, stale, "(now, listed): lower the number, or remove the file")

    def test_the_scan_counts_a_build(self) -> None:
        """The positive control: both spellings count, and a mention in a string or a comment does not."""
        self.assertEqual(2, _sites("arm = URRobotArm(cfg)\nother = ur.URRobotArm(cfg)\n"))
        self.assertEqual(0, _sites("# URRobotArm(cfg)\ntext = 'URRobotArm(cfg)'\nkind = URRobotArm\n"))
        self.assertIn("test_ur_arm.py", _measured(), "a scan that finds no build in the UR arm's own tests is blind")


if __name__ == "__main__":
    unittest.main()
