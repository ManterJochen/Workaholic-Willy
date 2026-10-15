"""The Isaac arm keeps a copy of the UR arm's planning half, and the copy may only shrink (plan section 4.3).

`drivers/sim/arm.py` re-implements, method by method, the judging and planning half of `URRobotArm`: the path gates, the
camera world, the planner client and its refresh, the chain helpers, the tool-frame composition. P0e extracts that half
into `PlannedArm`; the sim moves onto it in PY-b, where its M1, M2 and EIH re-runs happen anyway. Until then this file
counts the copy, so it cannot grow unnoticed, and it is deleted with the copy.

Measured in aurora on 2026-09-22 (the P0e map), and the same in Willy on 2026-10-13: of the Isaac arm's 61 methods, 49
are the copy. 31 share a name with the UR arm, 18 do not (renamed twins, and planning pieces only the sim has); the
remaining 12 are the transport (6) and the contract verbs both arms implement with their own vendor's body (6). A same-named check alone would miss the renamed twins, so
the list names each one's counterpart, and a counterpart that is renamed or deleted fails here too. Main's bf55789
(2026-10-13) gave both arms ``set_finger_opening``, a capability verb, so the contract verbs are seven.
"""

from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path

from src.robot.drivers.sim.arm import IsaacRobotArm
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.drivers.planned.curobo_motion import CuroboPlanner

_ROOT = Path(__file__).resolve().parents[1]

#: (the sim method, the counterpart it duplicates or "" where only the sim has it, what PY-b does with it).
#: The list may only shrink. A method that leaves the sim for `PlannedArm` leaves the list in the same commit.
_SIM_PLANNING_HALF: tuple[tuple[str, str, str], ...] = (
    # --- same name on both arms -----------------------------------------------------------------------------------------
    ("__init__", "URRobotArm.__init__", "PlannedArm takes typed values; the runners' knobs become arguments"),
    ("safety_preflight", "URRobotArm.safety_preflight", "PlannedArm takes the preflight as an argument"),
    ("plans_paths", "URRobotArm.plans_paths", "identical expression"),
    ("disconnect", "URRobotArm.disconnect", "split: planner close in PlannedArm, the rest in the transport"),
    ("get_tcp_pose", "URRobotArm.get_tcp_pose", "the transport reports the flange, PlannedArm composes the TCP"),
    ("fk", "URRobotArm.fk", "an optional transport fk, composition in PlannedArm"),
    ("ik", "URRobotArm.ik", "an optional transport solve_ik, composition in PlannedArm"),
    ("camera_world_required", "URRobotArm.camera_world_required", "plans_paths AND the transport moves physically"),
    ("line_motion", "URRobotArm.line_motion", "PlannedArm asks the transport's route for its line reading"),
    ("camera_world_for_move", "URRobotArm.camera_world_for_move", "identical body"),
    ("live_camera_world_wired", "URRobotArm.live_camera_world_wired", "identical"),
    ("without_camera_world", "URRobotArm.without_camera_world", "identical body"),
    ("_move_camera_world", "URRobotArm._move_camera_world", "the UNPLANNED reason comes from the route"),
    ("_refused_for_camera_world", "URRobotArm._refused_for_camera_world", "identical body"),
    ("move_to_joints", "URRobotArm.move_to_joints", "the stamp helper differs"),
    ("_move_to_joints_unstamped", "URRobotArm._move_to_joints_unstamped", "PlannedArm types what the sim lets escape"),
    ("_judge_joint_move", "URRobotArm._judge_joint_move", "same order: refresh, path gate, planner"),
    ("_drive_joints", "URRobotArm._drive_joints", "the sim plans and monitors inside a transport verb"),
    ("move_linear", "URRobotArm.move_linear", "the UR judges the line; the sim drops the flag off cuRobo"),
    ("stop", "URRobotArm.stop", "the sim resets the continuity memo, the UR does not"),
    ("move_home", "URRobotArm.move_home", "the UR boxes the home grasp centre, the sim does not"),
    ("live_planner_world", "URRobotArm.live_planner_world", "identical"),
    ("set_live_planner_world", "URRobotArm.set_live_planner_world", "the UR also hands it to a built planner"),
    ("_self_envelope", "URRobotArm._self_envelope", "the UR passes the attached payload"),
    ("_goal_keep_out", "URRobotArm._goal_keep_out", "different return shapes"),
    ("tcp_at_joints_mm", "URRobotArm.tcp_at_joints_mm", "the sim yaws the chain 180 degrees"),
    ("_flange_mm", "URRobotArm._flange_mm", "the sim reads the yawed transforms"),
    ("_drive_curobo", "URRobotArm._drive_curobo", "the UR checks the plan end before moving, the sim after"),
    ("move", "URRobotArm.move", "same stamping shape"),
    ("_move_unstamped", "URRobotArm._move_unstamped", "the sim pre-resolves IK on cuRobo"),
    ("_drive_checked_line", "URRobotArm._drive_checked_line", "no chord check, no per-sample box on the sim"),
    # --- renamed twins ---------------------------------------------------------------------------------------------------
    ("_planner_judges_joints", "URRobotArm._judge_legs", "the UR judges the line as legs, at the line clearance"),
    ("_to_client_joint_order", "CuroboPlanner._to_client_order", "one remap from the description's joint names"),
    ("_resolve_ik", "URRobotArm.ik", "the sim's multi-seed Lula search stays in the Isaac transport"),
    ("_get_curobo_client", "URRobotArm._curobo_planner", "one planner builder, the UR semantics"),
    ("_resolve_motion_planner", "URRobotArm.start_planner", "start_planner and stop_planner for both"),
    ("set_planner_base_world", "URRobotArm._curobo_planner", "the declared world as a constructor value"),
    ("set_curobo_world", "CuroboPlanner.set_world", "one runtime registration, the sim semantics"),
    ("last_world_refresh", "URRobotArm._planner_refresh", "the refresh identity"),
    ("_refresh_planner_world", "URRobotArm._refresh_before_the_path_gate", "one refresh"),
    ("_drive_to_target", "URRobotArm._drive_pose", "the Isaac transport's drive_pose, typed"),
    ("_preflight_reject", "URRobotArm._gate_pose", "the same context and evaluate"),
    ("_guard_self_collision_margin_mm", "URRobotArm._default_curobo_client_factory",
     "PlannedArm takes the planner margin as a Maybe, and UNSET refuses"),
    # --- planning pieces only the sim has ------------------------------------------------------------------------------
    ("reset_safety_continuity", "", "read by calibration through getattr; decided in PY-b"),
    ("_joint_move_camera_world", "", "joint moves get their own route reason from the transport"),
    ("_move_joint_gated", "", "the raising joint move; PlannedArm has only the typed verbs"),
    ("_into_joint_limits", "", "a PlannedArm option over the description's limits"),
    ("_plan_joint_path", "", "PlannedArm, behind plan_joint_moves"),
    ("_drive_rmpflow", "", "the Isaac transport's drive_pose for the rmpflow route"),
)

#: Methods both arms define by name because the contract names them, each with its own vendor's body. Not the copy.
_CONTRACT_VERBS = frozenset({
    "capabilities", "connect", "get_joint_positions", "is_connected", "is_inside_workspace", "wait_until_steady",
    # HoldsItsFingersAsOpened (arm_capabilities): each arm tells its own guard how wide the fingers stand (lever G).
    "set_finger_opening",
})


def _own_methods(path: Path, class_name: str) -> set[str]:
    """The methods ``class_name`` defines in its own body, read from its source."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {item.name for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}
    raise AssertionError(f"{class_name} is not defined in {path}")


def _ur_side_methods() -> set[str]:
    """Every method the UR arm defines, in its own body or in any driver-layer base, so the extraction moves nothing out
    of this file's sight: after P0e the planning half lives on `PlannedArm`, which is one of those bases."""
    names: set[str] = set()
    for cls in inspect.getmro(URRobotArm):
        if cls.__module__.startswith("src.robot.drivers."):
            names.update(name for name, value in vars(cls).items()
                         if callable(value) or isinstance(value, (property, staticmethod, classmethod)))
    return names


class TheSimArmKeepsAShrinkingCopyTests(unittest.TestCase):
    _SIM = _ROOT / "src" / "robot" / "drivers" / "sim" / "arm.py"

    def test_the_count_only_falls(self) -> None:
        self.assertEqual(len(_SIM_PLANNING_HALF), 49, "the copy may only shrink, and its count says so")

    def test_every_listed_method_is_still_on_the_sim_arm(self) -> None:
        """A method that left the sim, left on the list, would make the count lie."""
        sim = _own_methods(self._SIM, "IsaacRobotArm")
        stale = [name for name, _, _ in _SIM_PLANNING_HALF if name not in sim]
        self.assertEqual([], stale, "remove these from _SIM_PLANNING_HALF and lower the count")

    def test_every_counterpart_it_names_still_exists(self) -> None:
        """A renamed UR method would leave a twin nobody can find again."""
        owners = {"URRobotArm": _ur_side_methods(), "CuroboPlanner": set(vars(CuroboPlanner))}
        missing = []
        for name, counterpart, _ in _SIM_PLANNING_HALF:
            if counterpart:
                owner, _, method = counterpart.partition(".")
                if method not in owners[owner]:
                    missing.append(f"{name} -> {counterpart}")
        self.assertEqual([], missing)

    def test_a_method_both_arms_define_is_on_the_list_or_a_contract_verb(self) -> None:
        """The detection half: a new same-named method on both arms is a new duplicate until someone says otherwise."""
        shared = _own_methods(self._SIM, "IsaacRobotArm") & _ur_side_methods()
        listed = {name for name, _, _ in _SIM_PLANNING_HALF}
        unexplained = sorted(shared - listed - _CONTRACT_VERBS)
        self.assertEqual([], unexplained, "list it with its counterpart, or name it a contract verb with a reason")

    def test_the_list_holds_no_name_twice_and_no_contract_verb(self) -> None:
        names = [name for name, _, _ in _SIM_PLANNING_HALF]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(), set(names) & _CONTRACT_VERBS)

    def test_the_sim_arm_imports_as_this_test_needs(self) -> None:
        """The control: the counts above describe the class that exists, not a stub of it."""
        self.assertEqual(IsaacRobotArm.__name__, "IsaacRobotArm")
        self.assertGreater(len(_own_methods(self._SIM, "IsaacRobotArm")), len(_SIM_PLANNING_HALF))


if __name__ == "__main__":
    unittest.main()
