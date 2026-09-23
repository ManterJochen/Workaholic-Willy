"""A cable along the arm is never wound more than half a turn from home, and nobody types a joint range.

``safety.joint_limits.within_half_turn_of_home`` narrows every axis whose range is a full turn or more to half a turn
either side of the home the arm drives to, before the margin comes off; a UR elbow, which Universal Robots plans within
+-180 degrees and so never turns a full turn, is left alone. The guard refuses outside it, the UR driver
chooses a Cartesian goal's configuration inside it, a plan cuRobo was left to choose the goal of is refused by the
same gates when it ends or passes outside, and the desk check prints the controller limits that hold it on the pendant.

What it costs, measured here with the closed-form inverse kinematics over random homes and poses: nothing at a margin
of zero, because every UR joint is 2*pi-periodic and each angle has a twin inside any full-turn window; at the
shipped 5 degree margin, the poses whose every configuration needs a joint within 5 degrees of the seam behind home
(73 of 20000 random poses on a UR10, 2026-09-23), and those alone.
"""

from __future__ import annotations

import math
import unittest
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config.schema.robot import JointLimitSafetyConfig, RobotConfig
from src.robot.constants import HOME_JOINTS_DEFAULT
from src.robot.core import JointPositions, MotionCommand, MotionStatus, RobotCapabilities
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.drivers.ur.curobo_motion import PLANNER_JOINT_ENVELOPE_RAD
from src.robot.safety import JointLimitGuard, SafetyContext, SafetyPreflight, SafetyReason
from src.robot.safety._fcl_self_collision import mesh_backend_status
from src.robot.safety._ur_ik import nearest_goals, ur_flange_ik
from src.robot.safety._ur_kinematics import ur_link_transforms_mm
from src.robot.safety.joint_limits import (
    UR_ELBOW_AXIS,
    JointLimitTableMissing,
    assert_joint_limit_table_available,
    half_turn_about_home_deg,
    half_turn_axes,
    stated_home_rad,
)
from tests._plan_end import OPEN_WORKSPACE, pose_where_it_ends
from tests.test_the_nearest_goal_is_planned import _Planner

_DECLINED = "unit double: this file reads which configurations a planner double is asked for, and no camera world"
_HOME = (0.2, -1.4, 1.3, -1.5, -1.57, 0.1)
_WINDOW = {"joint_limits": {"within_half_turn_of_home": True}}


class _Arm:
    """What the guard reads off an arm: its capabilities and the home it states."""

    def __init__(self, home: Any = None, *, model: str = "ur10", vendor: str = "ur") -> None:
        self.capabilities = RobotCapabilities(
            vendor=vendor, model=model, dof=6, supports_joint_move=True, supports_linear_move=True,
            supports_async_move=False, has_native_fk=False, has_native_ik=False, has_force_control=False,
            is_simulated=False,
        )
        if home is not None:
            self.home_joint_positions = home


def _judged(guard: JointLimitGuard, joints_deg: "list[float]", arm: Any) -> Any:
    ctx = SafetyContext(command=MotionCommand.MOVE_JOINTS, target_joints=JointPositions(np.radians(joints_deg)),
                        arm=arm)
    return guard.evaluate(ctx)


def _ur(*, home: "tuple[float, ...] | None" = _HOME, safety: "dict[str, Any] | None" = None,
        model: str = "ur10") -> URRobotArm:
    return URRobotArm(RobotConfig.model_validate({
        "vendor": "ur", "ur": {"model": model, "motion_planner": "curobo"},
        "safety": {"payload": {"enforce": False}, **(safety if safety is not None else _WINDOW)},
        "gripper": {"model": "robotiq_2f85"}, "workspace_limits": OPEN_WORKSPACE,
        **({"home_joint_positions": home} if home is not None else {}),
    }))


def _driven(planner: _Planner, *, judged: bool = True, **kwargs: Any) -> URRobotArm:
    """A cuRobo UR on a planner double; ``judged=False`` stubs the end and path gates, for a test of the goal asked."""
    arm = _ur(**kwargs)
    arm._conn = MagicMock()
    arm._conn.is_connected = True
    arm._conn.get_joint_positions.return_value = list(planner.here)
    arm._curobo_ur = planner  # type: ignore[assignment]
    if not judged:
        arm._preflight.gate_planned_path = lambda waypoints, *, arm=None, command=None: None  # type: ignore
        arm._gate_planned_config = lambda pose, joints: None  # type: ignore[method-assign]
    return arm


def _asked(planner: _Planner) -> "list[tuple[float, ...]]":
    """Every configuration the arm put to the planner as a goal: screened alone, or planned to."""
    goals = [tuple(call[1]) for call in planner.named("plan_joint")]
    goals += [call[1][0] for call in planner.named("check") if len(call[1]) == 1]
    return goals


# ------------------------------------------------------------------------------------------------------------------
# The window
# ------------------------------------------------------------------------------------------------------------------


class TheWindowTests(unittest.TestCase):
    def test_it_is_off_unless_a_cell_asks_and_the_ur10_profile_asks(self) -> None:
        from src.config import load_config, reload_config
        from src.config.loader import set_active_profile

        self.assertFalse(JointLimitSafetyConfig().within_half_turn_of_home)
        try:
            for profile, expected in ((None, False), ("sim", False), ("ur10", True)):
                reload_config()
                set_active_profile(profile)
                with self.subTest(profile=profile):
                    self.assertIs(expected, load_config(None).robot.safety.joint_limits.within_half_turn_of_home)
        finally:
            set_active_profile(None)
            reload_config()

    def test_every_full_turn_axis_is_narrowed_to_half_a_turn_either_side_of_home(self) -> None:
        home_deg = [10.0, -90.0, 45.0, -90.0, 90.0, 200.0]
        lower, upper = half_turn_about_home_deg([-360.0] * 6, [360.0] * 6, np.radians(home_deg))
        np.testing.assert_allclose([h - 180.0 for h in home_deg[:5]] + [20.0], lower, atol=1e-9)
        # Wrist 3's home lies past a half turn, so the table's own +360 cuts its window short.
        np.testing.assert_allclose([h + 180.0 for h in home_deg[:5]] + [360.0], upper, atol=1e-9)

    def test_a_ur_elbow_is_left_as_its_table_has_it(self) -> None:
        """Universal Robots plans the elbow within +-180 deg: narrowed about a home with the elbow at -90, it refused the
        elbow bent the other way, which has no twin to fall back on (measured on the real sidecar: example 10's radial
        ring, none of six stations ran)."""
        self.assertEqual(2, UR_ELBOW_AXIS)
        self.assertEqual((True, True, False, True, True, True), half_turn_axes([-360.0] * 6, [360.0] * 6, vendor="ur"))
        self.assertEqual((True,) * 6, half_turn_axes([-360.0] * 6, [360.0] * 6, vendor="sim"))
        lower, upper = half_turn_about_home_deg([-360.0] * 6, [360.0] * 6, np.radians([0, -90, -90, -90, 90, 0]),
                                                vendor="ur")
        self.assertEqual((-360.0, 360.0), (lower[2], upper[2]))
        self.assertEqual((-270.0, 90.0), (lower[3], upper[3]))

    def test_an_axis_of_less_than_a_full_turn_is_left_as_it_is(self) -> None:
        lower, upper = half_turn_about_home_deg([-360.0] * 5 + [-100.0], [360.0] * 5 + [100.0], [1.0] * 6)
        self.assertEqual((-100.0, 100.0), (lower[5], upper[5]))

    def test_one_bound_per_axis_or_nothing(self) -> None:
        with self.assertRaises(ValueError):
            half_turn_about_home_deg([-360.0] * 6, [360.0] * 6, [0.0] * 5)


# ------------------------------------------------------------------------------------------------------------------
# The guard
# ------------------------------------------------------------------------------------------------------------------


class TheGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = JointLimitGuard(JointLimitSafetyConfig(within_half_turn_of_home=True))
        self.home = _Arm((0.0, -math.pi / 2, 0.0, -math.pi / 2, 0.0, 0.0))

    def test_inside_the_window_is_admitted_and_past_it_is_refused_naming_the_twin(self) -> None:
        self.assertTrue(_judged(self.guard, [170.0, -90.0, 0.0, -90.0, 0.0, 0.0], self.home).accepted)
        refused = _judged(self.guard, [200.0, -90.0, 0.0, -90.0, 0.0, 0.0], self.home)
        self.assertIs(SafetyReason.JOINT_LIMIT, refused.reason)
        self.assertIn("half a turn either side of home", refused.message)
        self.assertIn("within_half_turn_of_home", refused.message)
        self.assertIn("-160.000 deg, is inside it", refused.message)
        self.assertEqual("0.000000", refused.detail["home_deg"])

    def test_the_same_configuration_passes_the_guard_without_the_window(self) -> None:
        """The control: the table alone admits it, so the refusal above is the window's."""
        plain = JointLimitGuard(JointLimitSafetyConfig())
        self.assertTrue(_judged(plain, [200.0, -90.0, 0.0, -90.0, 0.0, 0.0], self.home).accepted)

    def test_a_ur_elbow_bent_the_other_way_from_home_is_admitted(self) -> None:
        elbow_down = _Arm((0.0, -math.pi / 2, -math.pi / 2, -math.pi / 2, math.pi / 2, 0.0))
        self.assertTrue(_judged(self.guard, [0.0, -90.0, 120.0, -90.0, 90.0, 0.0], elbow_down).accepted)
        # The control: an arm of another vendor with the same table has its axis 2 narrowed like any other.
        other = JointLimitGuard(JointLimitSafetyConfig(
            within_half_turn_of_home=True, min_deg=[-360.0] * 6, max_deg=[360.0] * 6))
        sim = _Arm(elbow_down.home_joint_positions, vendor="sim")
        self.assertIs(SafetyReason.JOINT_LIMIT, _judged(other, [0.0, -90.0, 120.0, -90.0, 90.0, 0.0], sim).reason)

    def test_an_angle_at_the_seam_has_no_twin_inside_and_is_told_so(self) -> None:
        refused = _judged(self.guard, [0.0, -90.0, 0.0, -90.0, 0.0, 178.0], self.home)
        self.assertIs(SafetyReason.JOINT_LIMIT, refused.reason)
        self.assertIn("No full turn of this angle is inside it", refused.message)

    def test_the_window_follows_the_home_the_arm_states(self) -> None:
        wound = _Arm((math.radians(170.0), 0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertTrue(_judged(self.guard, [300.0, 0.0, 0.0, 0.0, 0.0, 0.0], wound).accepted)
        self.assertFalse(_judged(self.guard, [-20.0, 0.0, 0.0, 0.0, 0.0, 0.0], wound).accepted)

    def test_an_arm_that_states_no_home_is_refused_every_move_naming_the_key(self) -> None:
        for arm in (_Arm(), _Arm(MagicMock()), None):
            with self.subTest(arm=arm):
                decision = _judged(self.guard, [0.0] * 6, arm)
                self.assertIs(SafetyReason.UNAVAILABLE, decision.reason)
        decision = _judged(self.guard, [0.0] * 6, _Arm())
        self.assertIn("home_joint_positions", decision.message)
        self.assertIsNone(self.guard.limits_for(vendor="ur", model="ur10"))
        self.assertIsNone(self.guard.limits_for_arm(_Arm()))
        self.assertIsNone(stated_home_rad(_Arm(MagicMock())), "a mock's attribute read as a home")

    def test_a_home_written_in_degrees_leaves_no_range_and_says_radians(self) -> None:
        degrees = _Arm((0.0, -90.0, 90.0, -90.0, -90.0, 0.0))
        decision = _judged(self.guard, [0.0] * 6, degrees)
        self.assertIs(SafetyReason.UNAVAILABLE, decision.reason)
        self.assertIn("read in radians", decision.message)


class TheBuildTests(unittest.TestCase):
    """An arm the window cannot be centred on is refused where it is built, not at every move."""

    def _arm(self, home: Any) -> _Arm:
        arm = _Arm(home)
        arm.safety_preflight = SafetyPreflight([  # type: ignore[attr-defined]
            JointLimitGuard(JointLimitSafetyConfig(within_half_turn_of_home=True))])
        return arm

    def test_no_home_is_refused_at_build_naming_the_key(self) -> None:
        with self.assertRaises(JointLimitTableMissing) as caught:
            assert_joint_limit_table_available(self._arm(None))
        self.assertIn("robot.home_joint_positions", str(caught.exception))

    def test_a_home_in_degrees_is_refused_at_build(self) -> None:
        with self.assertRaises(JointLimitTableMissing) as caught:
            assert_joint_limit_table_available(self._arm((0.0, -90.0, 90.0, -90.0, -90.0, 0.0)))
        self.assertIn("read in radians", str(caught.exception))

    def test_a_home_in_radians_builds(self) -> None:
        assert_joint_limit_table_available(self._arm(_HOME))

    def test_the_ur_driver_states_the_home_it_drives_to(self) -> None:
        self.assertEqual(_HOME, _ur().home_joint_positions)
        self.assertEqual(HOME_JOINTS_DEFAULT, _ur(home=None).home_joint_positions)
        explicit = URRobotArm(_ur().config, home_joints=[0.5] * 6)
        self.assertEqual((0.5,) * 6, explicit.home_joint_positions)
        assert_joint_limit_table_available(_ur())


# ------------------------------------------------------------------------------------------------------------------
# The driver chooses inside it, and refuses a plan that leaves it
# ------------------------------------------------------------------------------------------------------------------


class TheDriverTests(unittest.TestCase):
    def test_the_goal_window_is_the_half_turn_about_home_less_the_margin(self) -> None:
        lower, upper = _ur()._goal_joint_window()
        for axis, home in enumerate(_HOME):
            with self.subTest(axis=axis):
                reach = math.radians(355.0 if axis == UR_ELBOW_AXIS else 175.0)
                self.assertAlmostEqual(max(PLANNER_JOINT_ENVELOPE_RAD[0][axis], home - reach), lower[axis])
                self.assertAlmostEqual(min(PLANNER_JOINT_ENVELOPE_RAD[1][axis], home + reach), upper[axis])

    def test_a_wrist_near_the_seam_is_turned_back_rather_than_across_it(self) -> None:
        """Wrist 3 at 170 deg, home at 0.1 rad; the goal's -170 deg is 20 deg away across the seam, at 190."""
        here = [*_HOME[:5], math.radians(170.0)]
        end = [*_HOME[:5], math.radians(-170.0)]
        without = _Planner(here=here)
        with without_camera_world_of(_driven(without, judged=False, safety={})) as arm:
            arm.move(pose_where_it_ends(arm, end))
        (asked,) = without.named("plan_joint")
        self.assertAlmostEqual(190.0, math.degrees(asked[1][5]), places=4)

        within = _Planner(here=here)
        with without_camera_world_of(_driven(within, judged=False)) as arm:
            result = arm.move(pose_where_it_ends(arm, end))
            lower, upper = arm._goal_joint_window()
        self.assertTrue(result.ok, result.message)
        for goal in _asked(within):
            for axis, value in enumerate(goal):
                self.assertTrue(lower[axis] <= value <= upper[axis], f"axis {axis} asked at {math.degrees(value)}")
        self.assertEqual([], within.named("plan"), "cuRobo was left to choose although a goal lies in the window")

    def test_no_goal_outside_the_window_is_ever_asked_for(self) -> None:
        rng = np.random.default_rng(7)
        lower, upper = _ur()._goal_joint_window()
        for trial in range(12):
            here = [rng.uniform(low + 0.2, high - 0.2) for low, high in zip(lower, upper)]
            end = [q + rng.normal(0.0, 1.0) for q in here]
            planner = _Planner(here=here)
            with without_camera_world_of(_driven(planner, judged=False)) as arm:
                arm.move(pose_where_it_ends(arm, end))
            for goal in _asked(planner):
                for axis, value in enumerate(goal):
                    with self.subTest(trial=trial, axis=axis):
                        self.assertTrue(lower[axis] <= value <= upper[axis])

    def test_cuRobos_own_goal_that_ends_outside_is_refused_and_named(self) -> None:
        """Every goal in the window refused by the planner; cuRobo's own plan ends a full turn round on the base."""
        here = list(_HOME)
        end = [_HOME[0] + 0.4, *_HOME[1:]]
        wound = [end[0] + 2.0 * math.pi, *end[1:]]
        planner = _Planner(here=here, cartesian=[list(here), wound])
        planner.screen = lambda config: False
        with without_camera_world_of(_driven(planner)) as arm:
            result = arm.move(pose_where_it_ends(arm, end))
        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status, result.message)
        self.assertIn("cuRobo chose this goal's configuration itself (plan_pose)", result.message)
        self.assertIn("ends outside the half turn either side of home", result.message)
        self.assertIn("within_half_turn_of_home", result.message)
        self.assertEqual([], planner.named("execute"))

    def test_without_the_window_a_refused_end_says_nothing_about_one(self) -> None:
        here = list(_HOME)
        wound = [_HOME[0] + 0.4, *_HOME[1:]]
        planner = _Planner(here=here, cartesian=[list(here), wound])
        planner.screen = lambda config: False
        limits = {"joint_limits": {"min_deg": [-360.0] * 6, "max_deg": [360.0] * 5 + [1.0]}}
        with without_camera_world_of(_driven(planner, safety=limits)) as arm:
            result = arm.move(pose_where_it_ends(arm, wound))
        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status, result.message)
        self.assertNotIn("plan_pose", result.message)


@pytest.mark.skipif(mesh_backend_status("ur10") != "ok", reason="no exact mesh backend on this box")
def test_cuRobos_own_plan_that_passes_outside_is_refused_by_the_path_gate() -> None:
    """Both ends inside the window, the base swinging past the seam between them: the path gate refuses it."""
    here = list(_HOME)
    end = [_HOME[0] + 0.3, *_HOME[1:]]
    out = [_HOME[0] + math.radians(200.0), *_HOME[1:]]
    dense = [list(here)] + [[here[0] + (out[0] - here[0]) * i / 10, *here[1:]] for i in range(1, 11)]
    dense += [[out[0] + (end[0] - out[0]) * i / 10, *here[1:]] for i in range(1, 11)]
    planner = _Planner(here=here, cartesian=dense)
    planner.screen = lambda config: False
    with without_camera_world_of(_driven(planner)) as arm:
        result = arm.move(pose_where_it_ends(arm, end))
    assert result.status is MotionStatus.JOINT_LIMIT_REJECTED, result.message
    assert "passes outside the half turn either side of home" in result.message
    assert "sample" in result.message
    assert planner.named("execute") == []


class without_camera_world_of:  # noqa: N801 (a context manager read as a phrase)
    """The arm, inside its own ``without_camera_world``, for a planner double that holds no world."""

    def __init__(self, arm: URRobotArm) -> None:
        self.arm = arm
        self._inner: Any = None

    def __enter__(self) -> URRobotArm:
        self._inner = self.arm.without_camera_world(_DECLINED)
        self._inner.__enter__()
        return self.arm

    def __exit__(self, *exc: Any) -> None:
        self._inner.__exit__(*exc)


# ------------------------------------------------------------------------------------------------------------------
# What it costs: nothing but the margin's band at the seam
# ------------------------------------------------------------------------------------------------------------------


def _reachable(arm: URRobotArm, solutions: Any) -> bool:
    lower, upper = arm._goal_joint_window()
    return bool(nearest_goals(solutions, current=list(arm.home_joint_positions), lower=lower, upper=upper,
                              velocity=1.0))


def _wrapped(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


@pytest.mark.parametrize("model", ["ur10", "ur5e", "ur3e"])
def test_the_window_makes_no_reachable_pose_unreachable(model: str) -> None:
    """Random homes and random configurations in the planner's window: every pose reachable without the window,
    through any closed-form solution, is reachable inside it at a margin of zero.

    Homes stay within 174 deg of zero on every joint, where the planner's own +-(2 pi - 0.1) cannot cut the window.
    """
    rng = np.random.default_rng(2026)
    before = _ur(safety={"joint_limits": {"margin_deg": 0.0}}, model=model)
    within = _ur(safety={"joint_limits": {"margin_deg": 0.0, "within_half_turn_of_home": True}}, model=model)
    low, high = PLANNER_JOINT_ENVELOPE_RAD
    checked = 0
    for _ in range(1500):
        home = tuple(rng.uniform(-math.radians(174.0), math.radians(174.0)) for _ in range(6))
        before._home_joints = list(home)
        within._home_joints = list(home)
        joints = [rng.uniform(a, b) for a, b in zip(low, high)]
        frames = ur_link_transforms_mm(model, np.asarray(joints))
        solutions = ur_flange_ik(model, frames[-1], q6_if_singular=joints[5])
        assert solutions is not None
        if not _reachable(before, solutions):
            continue
        checked += 1
        assert _reachable(within, solutions), f"{model}: a pose reachable from {joints} lost to the window at {home}"
    assert checked > 1400


def test_what_the_margin_costs_is_the_band_at_the_seam_behind_home() -> None:
    """At the shipped 5 deg a pose is lost only where every configuration the window holds needs a joint within
    5 deg of the seam, and that is rare: a few in a thousand random poses on a UR10."""
    rng = np.random.default_rng(11)
    open_window = _ur(safety={"joint_limits": {"margin_deg": 0.0, "within_half_turn_of_home": True}})
    shipped = _ur()
    guard = next(g for g in shipped._preflight.guards if g.name == "joint_limit")
    self_margin = math.radians(guard.margin_deg)  # type: ignore[attr-defined]
    assert self_margin == pytest.approx(math.radians(5.0))
    low, high = PLANNER_JOINT_ENVELOPE_RAD
    lost = checked = 0
    for _ in range(4000):
        home = tuple(rng.uniform(-math.radians(174.0), math.radians(174.0)) for _ in range(6))
        open_window._home_joints = list(home)
        shipped._home_joints = list(home)
        joints = [rng.uniform(a, b) for a, b in zip(low, high)]
        frames = ur_link_transforms_mm("ur10", np.asarray(joints))
        solutions = ur_flange_ik("ur10", frames[-1], q6_if_singular=joints[5])
        assert solutions is not None
        if not _reachable(open_window, solutions):
            continue
        checked += 1
        if _reachable(shipped, solutions):
            continue
        lost += 1
        for solution in solutions:
            if not _reachable(open_window, [solution]):
                continue
            assert any(abs(_wrapped(q - h)) > math.pi - self_margin - 1e-9
                       for axis, (q, h) in enumerate(zip(solution, home)) if axis != UR_ELBOW_AXIS), (
                f"{solution} lost at the margin without a joint at the seam behind home {home}")
    assert 0 < lost < 0.01 * checked, (lost, checked)


# ------------------------------------------------------------------------------------------------------------------
# The desk prints the controller's own limits
# ------------------------------------------------------------------------------------------------------------------


def _desk_row(config: RobotConfig) -> Any:
    from src.robot.execution.real_cell.preflight import run_config_preflight

    report = run_config_preflight(config, curobo_available=True, collision_engine="coal")
    rows = [check for check in report.checks if check.name == "joint window"]
    return rows[0] if rows else None


class TheDeskTests(unittest.TestCase):
    def test_the_desk_prints_the_polyscope_ranges_of_the_window_in_degrees(self) -> None:
        from src.robot.execution.real_cell.preflight import CheckStatus

        home_deg = (90.0, -90.0, 90.0, -90.0, -90.0, 0.0)
        row = _desk_row(_ur(home=tuple(math.radians(v) for v in home_deg)).config)
        self.assertIs(CheckStatus.BENCH, row.status, row.detail)
        self.assertIn("PolyScope Installation > Safety > Joint Limits > Position Range", row.detail)
        for name, centre in zip(("Base", "Shoulder", "Elbow", "Wrist 1", "Wrist 2", "Wrist 3"), home_deg):
            if name == "Elbow":
                self.assertIn("Elbow as it is", row.detail)
                continue
            self.assertIn(f"{name} {centre - 180.0:.1f} to {centre + 180.0:.1f}", row.detail)
        self.assertIn("5 deg inside these limits", row.detail)

    def test_an_unset_home_warns_that_the_seam_follows_the_default(self) -> None:
        from src.robot.execution.real_cell.preflight import CheckStatus

        row = _desk_row(_ur(home=None).config)
        self.assertIs(CheckStatus.WARN, row.status)
        self.assertIn("Base -180.0 to 180.0", row.detail)
        self.assertIn("robot.home_joint_positions", row.fix)

    def test_a_home_in_degrees_blocks(self) -> None:
        from src.robot.execution.real_cell.preflight import CheckStatus

        row = _desk_row(_ur(home=(0.0, -90.0, 90.0, -90.0, -90.0, 0.0)).config)
        self.assertIs(CheckStatus.BLOCK, row.status)
        self.assertIn("radians", row.fix)

    def test_no_window_no_row(self) -> None:
        self.assertIsNone(_desk_row(_ur(safety={}).config))


if __name__ == "__main__":
    unittest.main()
