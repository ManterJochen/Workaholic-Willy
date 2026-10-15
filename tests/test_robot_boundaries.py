from __future__ import annotations

import ast
import importlib
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.config.schema.robot import WorkspaceLimitsConfig
from src.calibration.eye_hand import MountingMode
from src.geometry import Frame, Pose, Transform, rotation_vector_to_matrix
from src.robot.core import JointPositions, RobotArm, RobotConnectionError, RobotVendor
from src.robot.core.motion_result import MotionCommand, MotionResult
from src.robot.drivers import available_vendors, create_arm
from src.robot.execution.calibration import CalibrationMode, CalibrationRoutine
from src.robot.grippers import create_gripper
from src.robot.grasping import GraspCalculator, GraspFrame


def _transform_matrix(rvec: np.ndarray, translation_mm: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_vector_to_matrix(rvec)
    matrix[:3, 3] = np.asarray(translation_mm, dtype=np.float64)
    return matrix


def _inverse(matrix: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return inverse


def _sample_tool_pose(index: int) -> np.ndarray:
    rvecs = [
        np.array([0.25, 0.07, -0.04], dtype=np.float64),
        np.array([-0.05, 0.32, 0.09], dtype=np.float64),
        np.array([0.11, -0.08, 0.29], dtype=np.float64),
    ]
    return _transform_matrix(
        rvecs[index % len(rvecs)] * (1.0 + 0.2 * index),
        np.array(
            [120.0 * index, (-1) ** index * 40.0 * index, 100.0 + 30.0 * index],
            dtype=np.float64,
        ),
    )


def _synthetic_eye_to_hand_data() -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    T_cam_to_base = _transform_matrix(
        np.array([0.18, -0.21, 0.34], dtype=np.float64),
        np.array([320.0, -140.0, 760.0], dtype=np.float64),
    )
    T_tool_to_marker = _transform_matrix(
        np.array([-0.08, 0.13, 0.05], dtype=np.float64),
        np.array([35.0, 20.0, 90.0], dtype=np.float64),
    )
    tool_poses = [_sample_tool_pose(index) for index in range(7)]
    return T_cam_to_base, T_tool_to_marker, tool_poses


def _synthetic_eye_in_hand_data() -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Synthesize a consistent eye-in-hand dataset.

    Mirrors :func:`_synthetic_eye_to_hand_data` but with the mounting
    geometry flipped:

    * The camera is bolted to the tool. The calibrator's returned
      ``T_cam_to_tool`` is the unknown we want to recover.
    * A calibration marker is fixed in the world (``T_base_to_marker``
      constant) and observed from each pose.

    The calibrator follows the same convention as
    :class:`EyeInHandCalibrator`'s direct unit tests in
    ``tests/test_eye_hand_workflows.py``; the camera observation from
    tool pose ``i`` is::

        T_cam_to_marker_i = inv(T_cam_to_tool) @ inv(T_base_to_tool_i) @ T_base_to_marker

    Returns ``(T_cam_to_tool_truth, T_base_to_marker, [T_base_to_tool_i])``.
    """

    # Truth value chosen directly in the calibrator's output convention.
    T_cam_to_tool = _transform_matrix(
        np.array([0.11, -0.07, 0.22], dtype=np.float64),
        np.array([25.0, -10.0, 80.0], dtype=np.float64),
    )
    T_base_to_marker = _transform_matrix(
        np.array([0.05, 0.18, -0.04], dtype=np.float64),
        np.array([400.0, -50.0, 250.0], dtype=np.float64),
    )
    tool_poses = [_sample_tool_pose(index) for index in range(7)]
    return T_cam_to_tool, T_base_to_marker, tool_poses


class FakeArm:
    def __init__(self) -> None:
        self.current_pose: Pose | None = None
        self.settle_calls: list[float] = []

    def move_to(self, pose: Pose, **_: object) -> bool:
        self.current_pose = pose
        return True

    def move(self, pose: Pose, **_: object) -> MotionResult:
        self.current_pose = pose
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)

    def get_tcp_pose(self) -> Pose:
        if self.current_pose is None:
            raise RuntimeError("FakeArm has not moved yet")
        return self.current_pose

    def wait_until_steady(self, timeout_s: float) -> None:
        self.settle_calls.append(float(timeout_s))


class FakeProvider:
    def get_stereo_rig_index(self, rig_id: str) -> int:
        if rig_id != "rig_0":
            raise KeyError(rig_id)
        return 0

    def grab_rectified(self, rig_id: str) -> SimpleNamespace:
        del rig_id
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        return SimpleNamespace(left=image, right=image.copy())


class RobotImportBoundaryTests(unittest.TestCase):
    def test_robot_import_surfaces_do_not_require_removed_packages(self) -> None:
        modules = [
            "src.robot",
            "src.robot.execution",
            "src.robot.execution.calibration",
            "src.robot.grasping",
            "src.robot.grasping.generation.calculator",
        ]

        for module_name in modules:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_the_package_root_hands_out_no_driver(self) -> None:
        """`src.robot` is the vendor-neutral door: a driver reaches a caller as a `RobotArm`.

        Until 2026-09-20 the root re-exported `URRobotArm`, `URConnection`, `URPose` and `MotionController`
        through a lazy map of string targets, so one vendor's classes were importable from the neutral package
        and no import scan could see it: the map holds `".drivers.ur.arm"` as text, not as an import statement.
        This test resolves every exported name and asks the object which module it came from, so a
        string-literal re-export is caught the same as an import.
        """
        package = importlib.import_module("src.robot")

        exported = list(getattr(package, "__all__", ()))
        self.assertGreater(len(exported), 30, "nothing was exported, so this test proves nothing")
        from_a_driver = sorted(
            name for name in exported
            if ".drivers." in str(getattr(getattr(package, name), "__module__", ""))
        )
        self.assertEqual(from_a_driver, [])

    def test_stale_robot_dependency_paths_are_not_reintroduced(self) -> None:
        root = Path(__file__).resolve().parents[1]
        checked = [
            root / "src" / "robot" / "execution" / "calibration.py",
            root / "src" / "robot" / "grasping" / "generation" / "calculator.py",
        ]
        forbidden = [
            "src.camera.calibration",
            "src.pipelines.camera.orchestration",
            "src.models.segmentation",
        ]

        for path in checked:
            text = path.read_text(encoding="utf-8")
            for phrase in forbidden:
                with self.subTest(path=path.name, phrase=phrase):
                    self.assertNotIn(phrase, text)

    def test_no_layer_below_a_person_acknowledges_a_fault(self) -> None:
        """Recovery is NEVER automatic (owner, 2026-09-19), and this is the rule rather than the habit.

        `acknowledge_fault` clears a protective stop at the controller. It is reached from a script run at the cell,
        where whoever clears a stop can see the arm. Nothing in the execution, grasping or safety layers
        may call it, and nothing in the console may either: a cell that clears its own stop and carries on is the
        one failure mode a stop exists to prevent. MEASURED before the rule: no such call existed, so this pins
        the present rather than describing a repair.
        """
        root = Path(__file__).resolve().parents[1]
        watched = [root / "src" / "robot" / "execution",
                   root / "src" / "robot" / "grasping",
                   root / "src" / "robot" / "safety",
                   root / "api"]
        callers: list[str] = []
        files = 0
        for base in watched:
            for path in base.rglob("*.py"):
                files += 1
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "acknowledge_fault":
                        callers.append(f"{path.relative_to(root).as_posix()}:{node.lineno}")
                    if (isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("getattr", "hasattr")
                            and len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
                            and node.args[1].value == "acknowledge_fault"):
                        callers.append(f"{path.relative_to(root).as_posix()}:{node.lineno} (by name)")

        self.assertGreater(files, 100, "the walk found almost nothing, so it proves nothing")
        self.assertEqual(callers, [])

    def test_vendor_sdk_imports_stay_inside_vendor_packages(self) -> None:
        root = Path(__file__).resolve().parents[1]
        patterns = [
            re.compile(r"^\s*(?:import|from)\s+rtde_control\b", re.MULTILINE),
            re.compile(r"^\s*(?:import|from)\s+rtde_receive\b", re.MULTILINE),
            re.compile(r"^\s*from\s+robotiq_gripper\b", re.MULTILINE),
            re.compile(r"^\s*(?:import|from)\s+rclpy\b", re.MULTILINE),
            # Isaac SDK (isaacsim/omni) + Franka (franky) — the most import-safety-sensitive drivers
            # (CLAUDE.md priority 1: Isaac imports stay lazy, inside the sim/willy_sim boundary).
            re.compile(r"^\s*(?:import|from)\s+isaacsim\b", re.MULTILINE),
            re.compile(r"^\s*(?:import|from)\s+omni\b", re.MULTILINE),
            re.compile(r"^\s*(?:import|from)\s+franky\b", re.MULTILINE),
        ]
        allowed = {
            "src/robot/drivers/ur/connection.py",
            "src/robot/grippers/robotiq.py",
        }
        # Vendor SDKs that legitimately live across a whole package: Isaac inside the sim driver, the
        # sim-only grippers, and the willy_sim integration layer; rclpy inside ros2/; franky inside franka/.
        allowed_prefixes = (
            "src/robot/drivers/sim/",
            "src/robot/grippers/sim/",
            "src/willy_sim/",
            "src/robot/drivers/ros2/",
            "src/robot/drivers/franka/",
        )
        offenders: list[str] = []

        for path in (root / "src").rglob("*.py"):
            rel = path.relative_to(root).as_posix()
            text = path.read_text(encoding="utf-8")
            if (any(pattern.search(text) for pattern in patterns)
                    and rel not in allowed and not rel.startswith(allowed_prefixes)):
                offenders.append(rel)

        self.assertEqual(offenders, [])


# ---------------------------------------------------------------------------------------------------
# The arm contract: one door, and what is not through it yet
# ---------------------------------------------------------------------------------------------------
#
# More vendors are to join UR, each as a transport under one planning core. That only stays one core if
# the stack above `robot/drivers` talks to an arm through `RobotArm` and the capability Protocols and
# NOTHING else: not a driver import, not a private attribute, and not a member read by name that no
# Protocol declares. The last one is the quiet one. A `getattr(arm, "x", None)` on a name the contract
# does not know does not fail on an arm without it; the feature it guards simply switches off, and
# nothing says so.
#
# Each list below is COUNTED and may only shrink. An entry names the step that retires it, so the list
# is a schedule rather than an excuse, and `assertEqual` on the count means removing an entry without
# lowering the number fails as loudly as adding one. Ported from aurora (69b7ba7c, 2026-10-13); the
# members Willy's own work reads (the cell's speed, the console, the wrist camera) were declared first.

#: Modules that may import a driver package. Everything else reaches a driver through `create_arm`.
_DRIVER_IMPORTERS = ("src/robot/drivers/", "src/robot/grippers/", "src/willy_sim/")

#: (path, imported module, the step that retires it). Rule I.
_DRIVER_IMPORT_DEBTS: tuple[tuple[str, str, str], ...] = (
    ("src/robot/execution/robot_parts.py", "src.robot.drivers.sim.config",
     "after P0 (aurora's PY-b): one `create_arm(vendor, robot_cfg)` signature once the sim arm is a PlannedArm "
     "and builds its own config"),
    ("datagen/robots.py", "src.robot.drivers.sim.robot_models",
     "unscheduled since P0d left it: datagen reads `load_arm` once the UR half of the sim model table goes"),
)

#: (path, member, the step that retires it). Rule III.b: a member read off an arm BY NAME that no
#: Protocol declares.
_UNDECLARED_MEMBER_DEBTS: tuple[tuple[str, str, str], ...] = (
    ("src/robot/core/arm_capabilities.py", "config",
     "after P0 (Willy's own, W2): the tree is handed to it as aurora's E5 hands it to the Robot's readers"),
    ("src/robot/execution/motion.py", "config",
     "after P0 (Willy's own, W2): steady_timeout_of, the rest the push waits for before each of its legs"),
    ("src/robot/execution/place_target.py", "config", "after P0 (Willy's own, W2), as arm_capabilities.py"),
    ("src/robot/execution/task.py", "config", "after P0 (Willy's own, W2), as arm_capabilities.py"),
    ("src/robot/grasping/loop/pick_loop.py", "config", "after P0 (Willy's own, W2), as arm_capabilities.py"),
    ("api/readiness.py", "config", "after P0 (Willy's own, W2), as arm_capabilities.py"),
    ("api/teach.py", "config", "after P0 (Willy's own, W2), as arm_capabilities.py"),
    ("api/routers/task.py", "config", "after P0 (Willy's own, W2), as arm_capabilities.py"),
    ("src/robot/execution/calibration.py", "reset_safety_continuity",
     "PY-b: only the Isaac arm has it, and the ETH and EIH calibrations change if it goes"),
)

#: Names that are not arms at all: a variable called `arm` holding a step outcome, a dict or a string.
_NOT_AN_ARM: tuple[tuple[str, int, str], ...] = (
    ("src/robot/execution/lifecycle.py", 126, "value"),
    ("src/robot/grasping/deep/__main__.py", 190, "get"),
    ("src/robot/safety/planning/hand.py", 191, "lower"),
    ("src/robot/safety/planning/planning_model.py", 67, "lower"),
)

_ARM_ROOTS = frozenset({"arm", "_arm", "the_arm"})


def _is_arm_root(node: ast.AST) -> bool:
    """Whether ``node`` is an expression this repository spells an arm with."""
    if isinstance(node, ast.Name):
        return node.id in _ARM_ROOTS
    if isinstance(node, ast.Attribute):
        return node.attr in _ARM_ROOTS
    return False


def _declared_members() -> set[str]:
    """Every member `RobotArm` or a runtime-checkable capability Protocol declares, by introspection.

    Collected rather than listed, so the contract growing a member grows this set with it and a new
    capability needs no edit here. Hand guiding's capability is named rather than its module walked: that
    module also holds the session's Protocol, whose verbs are no arm's.
    """
    from src.robot.core import arm_capabilities, camera_world
    from src.robot.core.freedrive import SupportsFreedrive
    from src.robot.core.robot_arm import RobotArm
    from src.robot.safety import attestation

    names = {name for name in dir(RobotArm) if not name.startswith("_")}
    for module in (arm_capabilities, camera_world, attestation):
        for value in vars(module).values():
            if isinstance(value, type) and getattr(value, "_is_protocol", False):
                names |= {name for name in dir(value) if not name.startswith("_")}
    names |= {name for name in dir(SupportsFreedrive) if not name.startswith("_")}
    return names


def _imported_modules(node: ast.AST) -> list[str]:
    """The modules an import statement names, or none for any other node."""
    if isinstance(node, ast.ImportFrom) and node.module:
        return [node.module]
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    return []


def _arm_member(node: ast.AST) -> str:
    """The member ``node`` reads off an arm, or ``""`` where it reads none. One definition for the rule and its audit."""
    if isinstance(node, ast.Attribute) and _is_arm_root(node.value):
        return node.attr
    if (isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("getattr", "hasattr")
            and node.args and _is_arm_root(node.args[0]) and len(node.args) > 1
            and isinstance(node.args[1], ast.Constant)):
        return str(node.args[1].value)
    return ""


class ArmContractBoundaryTests(unittest.TestCase):
    """Above `robot/drivers`, an arm is a `RobotArm` plus the capability Protocols it satisfies."""

    root: Path
    scope: tuple[Path, ...]
    declared: set[str]

    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.root = root
        cls.scope = tuple(
            base for base in (
                root / "src" / "robot",
                root / "src" / "calibration",
                root / "api",
                root / "willy",
                root / "datagen",
                root / "examples",
            ) if base.exists()
        )
        cls.declared = _declared_members()

    def _modules(self, *, drivers: bool, arms_only: bool = False) -> list[tuple[str, ast.AST]]:
        """Every module in scope.

        ``drivers`` says whether the driver packages come along. ``arms_only`` drops the two trees whose
        ``arm`` names are not arms: datagen builds config specs called ``arm``, and the examples read them
        back. They stay in rule I, which is about imports and means the same there.
        """
        out: list[tuple[str, ast.AST]] = []
        for base in self.scope:
            for path in base.rglob("*.py"):
                rel = path.relative_to(self.root).as_posix()
                if "__pycache__" in rel:
                    continue
                if not drivers and rel.startswith(_DRIVER_IMPORTERS):
                    continue
                if arms_only and rel.startswith(("datagen/", "examples/")):
                    continue
                out.append((rel, ast.parse(path.read_text(encoding="utf-8"))))
        return out

    def test_the_walk_sees_the_tree_it_claims_to_judge(self) -> None:
        """The control every rule below rests on: a walk that found nothing would pass all of them.

        Floors, not counts, so ordinary growth never touches this test.
        """
        modules = self._modules(drivers=False, arms_only=True)
        self.assertGreater(len(modules), 300, "the walk collected almost nothing")
        self.assertGreater(len(self.declared), 30, "no contract members were introspected")
        accesses = sum(
            1 for _rel, tree in modules for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and _is_arm_root(node.value)
        )
        self.assertGreater(accesses, 40, f"only {accesses} arm accesses found: the root heuristic stopped matching")

    def test_every_listed_debt_is_still_owed(self) -> None:
        """A paid debt left on a list is a standing permission to incur it again.

        The rules below only ask whether each finding is on its list; nothing asked whether each entry still has a
        finding. In aurora, D3a moved four modules off the sim driver's model table and the file stayed green with all
        nine import debts listed, because an entry nobody matches is never consulted, and the count held because the
        list had not changed.
        """
        imports = {(rel, module) for rel, tree in self._modules(drivers=True)
                   for node in ast.walk(tree) for module in _imported_modules(node)}
        stale = [f"{path} no longer imports {module}: remove it from _DRIVER_IMPORT_DEBTS and lower the count"
                 for path, module, _why in _DRIVER_IMPORT_DEBTS if (path, module) not in imports]
        reads = {(rel, getattr(node, "lineno", 0), _arm_member(node))
                 for rel, tree in self._modules(drivers=False, arms_only=True)
                 for node in ast.walk(tree) if _arm_member(node)}
        members = {(rel, member) for rel, _line, member in reads}
        stale += [f"{path} no longer reads `{member}` off an arm: remove it from _UNDECLARED_MEMBER_DEBTS"
                  for path, member, _why in _UNDECLARED_MEMBER_DEBTS if (path, member) not in members]
        stale += [f"{path}:{line} no longer reads `{member}`: remove it from _NOT_AN_ARM"
                  for path, line, member in _NOT_AN_ARM if (path, line, member) not in reads]
        self.assertEqual(stale, [])

    def test_nothing_outside_the_drivers_imports_a_driver(self) -> None:
        """Rule I. A vendor reaches the stack as a `RobotArm` from `create_arm`, and nothing else.

        The factory root (`src.robot.drivers`) is the door. The robot parts and the console's diagnostics may
        read the doctor, which exists to know about vendors. Everything else is a debt with a date.
        """
        allowed_paths = {
            "src/robot/execution/robot_parts.py": {"src.robot.drivers.doctor"},
            "api/routers/diagnostics.py": {"src.robot.drivers.doctor"},
        }
        debts = {(path, module) for path, module, _why in _DRIVER_IMPORT_DEBTS}
        offenders: list[str] = []
        for rel, tree in self._modules(drivers=False):
            for node in ast.walk(tree):
                for module in _imported_modules(node):
                    if not module.startswith("src.robot.drivers."):
                        continue
                    if module in allowed_paths.get(rel, set()) or (rel, module) in debts:
                        continue
                    offenders.append(f"{rel}:{getattr(node, 'lineno', '?')}: {module}")
        self.assertEqual(offenders, [])
        self.assertEqual(len(_DRIVER_IMPORT_DEBTS), 2, "the debt list may only shrink, and its count says so")

    def test_no_private_member_of_an_arm_is_read_above_the_drivers(self) -> None:
        """Rule III.a, and it is at zero: a private attribute is a driver's own business.

        The console read `arm._conn` to tell URSim from a real controller, which is why
        `ReportsControllerProvenance` exists, and the calibration read `arm._tcp_at_joints_mm`, which is
        `tcp_at_joints_mm` now (2026-10-13).
        """
        offenders: list[str] = []
        for rel, tree in self._modules(drivers=False, arms_only=True):
            for node in ast.walk(tree):
                if (isinstance(node, ast.Attribute) and _is_arm_root(node.value)
                        and node.attr.startswith("_")):
                    offenders.append(f"{rel}:{node.lineno}: {node.attr}")
                if (isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("getattr", "hasattr")
                        and node.args and _is_arm_root(node.args[0]) and len(node.args) > 1
                        and isinstance(node.args[1], ast.Constant)
                        and str(node.args[1].value).startswith("_")):
                    offenders.append(f"{rel}:{node.lineno}: {node.args[1].value}")
        self.assertEqual(offenders, [])

    def test_no_member_is_read_off_an_arm_that_no_protocol_declares(self) -> None:
        """Rule III.b. The quiet one: a name the contract does not know switches a feature off in silence.

        A member the contract does declare may still be read defensively, because a caller may be handed an
        arm-like object, and several of those reads check the value they get, which is a stronger check than
        asking the arm. What may not happen is a name no Protocol knows.
        """
        debts = {(path, member) for path, member, _why in _UNDECLARED_MEMBER_DEBTS}
        not_an_arm = {(path, line, member) for path, line, member in _NOT_AN_ARM}
        offenders: list[str] = []
        for rel, tree in self._modules(drivers=False, arms_only=True):
            for node in ast.walk(tree):
                member = _arm_member(node)
                if not member or member.startswith("_") or member in self.declared:
                    continue
                if (rel, member) in debts or (rel, getattr(node, "lineno", 0), member) in not_an_arm:
                    continue
                offenders.append(f"{rel}:{getattr(node, 'lineno', '?')}: {member}")
        self.assertEqual(offenders, [])
        self.assertEqual(len(_UNDECLARED_MEMBER_DEBTS), 9, "the debt list may only shrink, and its count says so")

    def test_a_driver_never_imports_the_stack_above_it(self) -> None:
        """The other direction of the same arrow.

        A driver may import `robot/core`, `geometry`, `safety` and `config`: the contract it implements, the
        types it speaks and the guards it is judged by. It may not import `execution`, `grasping` or the
        console, because a vendor that reaches up into the pick path is a vendor the pick path cannot be
        tested without. The one that did, the UR bench's command line building the cell's gripper under the cell
        lock, left the driver for ``execution/io_bench`` in P0g slice 9: composition lives above the drivers.
        """
        forbidden = ("src.robot.execution", "src.robot.grasping", "api.", "willy.")
        offenders: list[str] = []
        drivers = 0
        for rel, tree in self._modules(drivers=True):
            if not rel.startswith(("src/robot/drivers/", "src/robot/grippers/")):
                continue
            drivers += 1
            for node in ast.walk(tree):
                for module in _imported_modules(node):
                    if module.startswith(forbidden) or module in ("api", "willy"):
                        offenders.append(f"{rel}:{getattr(node, 'lineno', '?')}: {module}")
        self.assertGreater(drivers, 20, "the driver walk found almost nothing")
        self.assertEqual(offenders, [])

    def test_each_rule_finds_its_own_violation(self) -> None:
        """The positive controls. A rule that matches nothing passes every tree, including a broken one."""
        source = (
            "from src.robot.drivers.ur.arm import URRobotArm\n"
            "def f(arm):\n"
            "    a = arm._conn\n"
            "    b = getattr(arm, 'twist_the_wrist', None)\n"
            "    c = arm.capabilities\n"
            "    return a, b, c\n"
        )
        tree = ast.parse(source)

        imports = [module for node in ast.walk(tree) for module in _imported_modules(node)
                   if module.startswith("src.robot.drivers.")]
        self.assertEqual(imports, ["src.robot.drivers.ur.arm"], "rule I matched nothing")

        private = [node.attr for node in ast.walk(tree)
                   if isinstance(node, ast.Attribute) and _is_arm_root(node.value) and node.attr.startswith("_")]
        self.assertEqual(private, ["_conn"], "rule III.a matched nothing")

        undeclared = [
            str(node.args[1].value) for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "getattr"
            and node.args and _is_arm_root(node.args[0]) and len(node.args) > 1
            and isinstance(node.args[1], ast.Constant) and str(node.args[1].value) not in self.declared
        ]
        self.assertEqual(undeclared, ["twist_the_wrist"], "rule III.b matched nothing")
        self.assertIn("capabilities", self.declared, "a declared member was read as undeclared")


class RobotRegistryBoundaryTests(unittest.TestCase):
    def test_dummy_arm_factory_returns_robot_arm_without_config(self) -> None:
        arm = create_arm(RobotVendor.DUMMY, dof=7, config=object())

        self.assertIsInstance(arm, RobotArm)
        self.assertIn(RobotVendor.DUMMY.value, available_vendors())
        arm.connect()
        target = Pose.identity(Frame.BASE, label="target")
        arm.move_linear(target)
        self.assertEqual(arm.get_tcp_pose(), target)
        joints = arm.get_joint_positions()
        self.assertIsInstance(joints, JointPositions)
        self.assertEqual(len(joints), 7)
        arm.disconnect()

    def test_reserved_unregistered_vendor_fails_clearly(self) -> None:
        with self.assertRaises(RobotConnectionError):
            create_arm(RobotVendor.FRANKA)

    def test_dummy_and_null_grippers_are_pure_python_protocols(self) -> None:
        dummy = create_gripper("dummy", min_width_mm=1.0, max_width_mm=20.0)
        dummy.connect()
        dummy.activate()
        dummy.set_width_mm(12.5)
        self.assertAlmostEqual(dummy.get_width_mm(), 12.5)
        dummy.disconnect()

        null = create_gripper("none")
        null.connect()
        null.set_width_mm(99.0)
        self.assertEqual(null.get_width_mm(), 0.0)


class GraspCalculatorBoundaryTests(unittest.TestCase):
    def _segmentation_fixture(self) -> tuple[SimpleNamespace, np.ndarray]:
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[7:17, 6:18] = 1
        depth = np.full((24, 24), 1000.0, dtype=np.float64)
        return SimpleNamespace(mask=mask, label="box"), depth

    def _calculator(self) -> GraspCalculator:
        return GraspCalculator(
            min_grip_width_mm=1.0,
            max_grip_width_mm=200.0,
            max_candidates=3,
            camera_matrix=np.array(
                [[200.0, 0.0, 12.0], [0.0, 200.0, 12.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
        )

    def test_compute_accepts_segmentation_like_object_and_typed_transform(self) -> None:
        segmentation, depth = self._segmentation_fixture()
        T_cam_to_base = np.eye(4, dtype=np.float64)
        T_cam_to_base[:3, 3] = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        transform = Transform.from_matrix(
            T_cam_to_base,
            from_frame=Frame.CAMERA,
            to_frame=Frame.BASE,
        )

        grasps = self._calculator().compute(
            segmentation,
            depth,
            camera_to_base=transform,
            pixel_to_mm=5.0,
            unit="mm",
        )

        self.assertTrue(grasps)
        self.assertTrue(all(grasp.frame is GraspFrame.BASE for grasp in grasps))

    def test_compute_without_transform_returns_camera_frame_only(self) -> None:
        segmentation, depth = self._segmentation_fixture()

        grasps = self._calculator().compute(
            segmentation,
            depth,
            pixel_to_mm=5.0,
            unit="mm",
        )

        self.assertTrue(grasps)
        self.assertTrue(all(grasp.frame is GraspFrame.CAMERA for grasp in grasps))

    def test_compute_rejects_bad_transform_frame_or_matrix(self) -> None:
        segmentation, depth = self._segmentation_fixture()
        bad_frame = Transform.identity(from_frame=Frame.BASE, to_frame=Frame.CAMERA)
        with self.assertRaises(ValueError):
            self._calculator().compute(
                segmentation,
                depth,
                camera_to_base=bad_frame,
                pixel_to_mm=5.0,
            )

        bad_matrix = np.eye(4, dtype=np.float64)
        bad_matrix[3, 3] = 2.0
        with self.assertRaises(ValueError):
            self._calculator().compute(
                segmentation,
                depth,
                T_cam_to_base=bad_matrix,
                pixel_to_mm=5.0,
            )

    def test_compute_rejects_ambiguous_transform_arguments(self) -> None:
        segmentation, depth = self._segmentation_fixture()
        transform = Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE)

        with self.assertRaises(ValueError):
            self._calculator().compute(
                segmentation,
                depth,
                T_cam_to_base=transform,
                camera_to_base=transform,
                pixel_to_mm=5.0,
            )


class CalibrationRoutineBoundaryTests(unittest.TestCase):
    def test_calibration_mode_alias_maps_to_mounting_mode(self) -> None:
        self.assertIs(CalibrationMode.EYE_TO_HAND, MountingMode.EYE_TO_HAND)
        self.assertIs(CalibrationMode.EYE_IN_HAND, MountingMode.EYE_IN_HAND)

    def test_routine_uses_robot_arm_and_modern_eye_to_hand_calibrator(self) -> None:
        T_cam_to_base, T_tool_to_marker, T_base_to_tool_list = _synthetic_eye_to_hand_data()
        poses = [
            Pose.from_matrix(T_base_to_tool, frame=Frame.BASE, label=f"pose_{index}")
            for index, T_base_to_tool in enumerate(T_base_to_tool_list)
        ]
        marker_poses = iter(
            _inverse(T_cam_to_base) @ T_base_to_tool @ T_tool_to_marker
            for T_base_to_tool in T_base_to_tool_list
        )
        settings = SimpleNamespace(
            mode="eye_to_hand",
            min_samples=4,
            min_distance_mm=0.0,
            min_angle=0.0,
            min_angle_deg=0.0,
        )
        routine = CalibrationRoutine(
            arm=FakeArm(),
            provider=FakeProvider(),
            stereo=SimpleNamespace(),
            workspace_limits=WorkspaceLimitsConfig(
                x_min=-2000.0,
                x_max=2000.0,
                y_min=-2000.0,
                y_max=2000.0,
                z_min=-2000.0,
                z_max=2000.0,
            ),
            eth_settings=settings,
            calibration_mode="eye_to_hand",
            settle_time_s=0.0,
        )
        routine._capture_marker_pose = lambda: next(marker_poses)  # type: ignore[method-assign]

        result = routine._execute(poses, T_tool_to_marker=None, dataset_save_path=None)

        self.assertEqual(result.mode, MountingMode.EYE_TO_HAND)
        self.assertIsNotNone(result.extrinsics)
        self.assertIsNone(result.T_cam_to_tool)
        self.assertIsNotNone(result.T_cam_to_base)
        self.assertTrue(np.allclose(result.T_cam_to_base, T_cam_to_base, atol=1e-7))
        self.assertEqual(result.num_samples, len(poses))

    def test_routine_uses_robot_arm_and_modern_eye_in_hand_calibrator(self) -> None:
        """Phase Q.A: Eye-in-Hand mode runs end-to-end through the routine.

        Complements the existing eye-to-hand test by proving the Phase N
        UR-decoupling holds for both mounting modes. The routine must
        accept a vendor-neutral :class:`FakeArm`, drive every pose via
        the typed ``move()`` surface, read TCP through ``get_tcp_pose``,
        and surface ``T_cam_to_tool`` on the result --- with no UR
        symbols imported anywhere along the path.
        """

        T_cam_to_tool_truth, T_base_to_marker, T_base_to_tool_list = (
            _synthetic_eye_in_hand_data()
        )
        poses = [
            Pose.from_matrix(T_base_to_tool, frame=Frame.BASE, label=f"pose_{index}")
            for index, T_base_to_tool in enumerate(T_base_to_tool_list)
        ]
        # The camera observes the world-fixed marker from each tool pose.
        # See :func:`_synthetic_eye_in_hand_data` for the geometry.
        marker_poses = iter(
            _inverse(T_cam_to_tool_truth) @ _inverse(T_base_to_tool) @ T_base_to_marker
            for T_base_to_tool in T_base_to_tool_list
        )
        settings = SimpleNamespace(
            mode="eye_in_hand",
            min_samples=4,
            min_distance_mm=0.0,
            min_angle=0.0,
            min_angle_deg=0.0,
        )
        routine = CalibrationRoutine(
            arm=FakeArm(),
            provider=FakeProvider(),
            stereo=SimpleNamespace(),
            workspace_limits=WorkspaceLimitsConfig(
                x_min=-2000.0,
                x_max=2000.0,
                y_min=-2000.0,
                y_max=2000.0,
                z_min=-2000.0,
                z_max=2000.0,
            ),
            eth_settings=settings,
            calibration_mode="eye_in_hand",
            settle_time_s=0.0,
        )
        routine._capture_marker_pose = lambda: next(marker_poses)  # type: ignore[method-assign]

        result = routine._execute(poses, T_tool_to_marker=None, dataset_save_path=None)

        self.assertEqual(result.mode, MountingMode.EYE_IN_HAND)
        # EIH solves for T_cam_to_tool, not T_cam_to_base.
        self.assertIsNotNone(result.T_cam_to_tool)
        self.assertIsNone(result.T_cam_to_base)
        # EIH does not produce camera extrinsics in the world frame ---
        # those only make sense for a fixed camera (ETH).
        self.assertIsNone(result.extrinsics)
        # Reconstruction must match the synthesised truth to high precision
        # (numerical only --- no measurement noise in the synthetic fixture).
        self.assertTrue(
            np.allclose(result.T_cam_to_tool, T_cam_to_tool_truth, atol=1e-6),
            msg=(
                "EIH solver did not recover the synthesised T_cam_to_tool.\n"
                f"Expected:\n{T_cam_to_tool_truth}\nGot:\n{result.T_cam_to_tool}"
            ),
        )
        self.assertEqual(result.num_samples, len(poses))


if __name__ == "__main__":
    unittest.main()