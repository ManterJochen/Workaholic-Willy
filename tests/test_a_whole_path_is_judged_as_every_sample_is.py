"""A path judged whole is judged as every sample of it is (``safety.self_collision.whole_path_judge``).

On the owner's cell (2026-10-08) the exact guard took 2.3 to 2.6 s for one route of 905 samples, 2.9 s for the way home
and 0.2 to 0.6 s for a line, judging one sample at a time; three quarters of it measured the arm against itself on pairs
that never came within 18 mm of the guard's 3. The owner, 2026-10-09: speed only "solange wir keine Qualität verlieren".

With the switch on, every guard of a joint move first looks at the whole path at once and names the first sample it might
refuse, never later than the first it does: the joint limits on every sample at once, the payload once, the exact meshes
with every part placed at every sample in one pass, a sample passed only on a proof that every distance the guard would
measure there keeps its limit (a bounding sphere, a part's convex hull, a distance measured earlier less how far the pair
can have moved since). From that sample on the gate judges every sample as it does with the switch off.

What this file pins, switch on against switch off, the same answer (``None``, or the status, the whole message and the
joints a refusal names):

* the owner's guard (``tests/_owner_guard.py``) on the three cell frames of 2026-10-07, cases A and D: the route from the
  look to 100 mm over the target, the line down to the recorded grasp, 204 lines down to grasps moved up to 12 mm and
  turned up to 0.6 rad, and 120 seeded joint paths among the parts;
* the borderline, on a slab under the Hand-E's fingers: one sample 3 mm from it give or take 1e-4 and 1e-7 mm, seen or
  declared, a refusal at the first sample and at the last, two fingers refusing at one sample (the first in the guard's
  order named), two boxes refusing at one sample (the first box named), the arm folded into itself (the first pair
  named), and a declared box under the hand of an arm whose base the guard turns half a turn (the Isaac cell's);
* the guard order: a joint limit refused before, at and after the first collision names the guard and the sample the loop
  names; a payload refused at the first sample; a guard that offers no whole-path pass, and the self-collision guard
  without an arm (the capsule proxy), leave every sample to the loop;
* the key: it loads, it is off by default, and off, every guard sees every sample; on, the gate reads it;
* mesh first (``URRobotArm._mesh_first_refusal``): the same sentence over the declared support plane and beside the
  robot's base, the same ``None`` where neither is near.

The exact mesh backend needs Coal or python-fcl and the baked bundles; these tests skip without them.
"""

from __future__ import annotations

import math
import unittest
from functools import lru_cache
from typing import Any
from unittest import mock

import numpy as np

from src.config.schema.robot import JointLimitSafetyConfig, PayloadSafetyConfig, RobotConfig, SelfCollisionSafetyConfig
from src.robot.core import MotionCommand
from src.robot.safety import SafetyPreflight
from src.robot.safety._capsule import AxisAlignedBox
from src.robot.safety._ur_kinematics import ur_link_transforms_mm
from src.robot.safety.decision import SafetyDecision
from src.robot.safety.joint_limits import JointLimitGuard
from src.robot.safety.path_samples import PathSamples, waypoint_path_samples
from src.robot.safety.payload import PayloadGuard
from tests import _owner_guard as owner
from tests.test_a_faster_camera_world_builds_the_same_world import frame

_FRAMES = ("F1", "F2", "F3")
_CASES = ("A", "D")
#: Lines down to a grasp moved off the recorded one, and joint paths among the parts, a world: 204 and 120 over the six.
_MOVED_LINES_A_WORLD = 34
_PATHS_A_WORLD = 20
#: The slab the borderline puts under the fingers: wide and thin, its top the nearest of it to the hand, millimetres.
_SLAB_HALF_MM = (300.0, 300.0, 5.0)


@lru_cache(maxsize=1)
def _pair() -> "tuple[Any, Any, Any]":
    """The owner's guard twice, the switch off and on, and the arm both judge."""
    off, arm = owner.owner_cell(whole=False)
    on, _ = owner.owner_cell(whole=True)
    return off, on, arm


def _gate(preflight: SafetyPreflight, arm: Any, kind: str, payload: Any) -> Any:
    if kind == "path":
        return preflight.gate_planned_path(payload, arm=arm, command=MotionCommand.MOVE_TO)
    return preflight.gate_joint_path(payload, arm=arm, command=MotionCommand.MOVE_TO)


def _both(off: SafetyPreflight, on: SafetyPreflight, arm: Any, kind: str, payload: Any) -> "tuple[Any, Any]":
    return owner.verdict(_gate(off, arm, kind, payload)), owner.verdict(_gate(on, arm, kind, payload))


def _sample_named(said: Any) -> int:
    """The sample a refusal names, counted from 1."""
    assert said is not None
    return int(str(said[1]).split("sample ", 1)[1].split(" ", 1)[0])


def _turned(about_z: float, turn: np.ndarray) -> np.ndarray:
    c, s = math.cos(about_z), math.sin(about_z)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ turn


class TheCellsWorldsAreJudgedAsEverySampleIsTests(unittest.TestCase):
    """The owner's guard over the cell's own worlds of 2026-10-07, each answer the same with the switch on."""

    def setUp(self) -> None:
        owner.needs_the_engine()

    def test_the_route_the_lines_and_the_paths_among_the_parts_get_the_same_answer(self) -> None:
        off, on, arm = _pair()
        reach = off.joint_radii_mm(arm)
        assert reach is not None
        rng = np.random.default_rng(20261007)
        judged = {"route": 0, "line": 0, "moved line": 0, "path": 0}
        refused = 0
        for name in _FRAMES:
            recorded = frame(name)
            look, down = np.asarray(recorded["look_rad"]), np.asarray(recorded["down_rad"])
            goal, grasp = np.asarray(recorded["goal"]), np.asarray(recorded["grasp"])
            for case in _CASES:
                boxes = owner.seen_world(name, case)
                off.set_perceived_obstacles(boxes)
                on.set_perceived_obstacles(boxes)
                jobs: list[tuple[str, str, Any]] = []
                standoff = owner.nearest_solution(goal, down)
                self.assertIsNotNone(standoff, f"{name}: the goal 100 mm over the target is out of reach")
                assert standoff is not None
                jobs.append(("route", "path", [look.tolist(), standoff.tolist()]))
                line = owner.line_samples(goal, grasp, standoff, reach)
                if line is not None:
                    jobs.append(("line", "samples", line))
                for _ in range(_MOVED_LINES_A_WORLD):
                    moved = grasp.copy()
                    moved[:3, :3] = _turned(rng.uniform(-0.6, 0.6), grasp[:3, :3])
                    moved[:3, 3] = grasp[:3, 3] + np.r_[rng.uniform(-12.0, 12.0, 2), rng.uniform(-8.0, 8.0)]
                    top = moved.copy()
                    top[:3, 3] = moved[:3, 3] + (goal[:3, 3] - grasp[:3, 3])
                    line = owner.line_samples(top, moved, standoff, reach)
                    if line is not None:
                        jobs.append(("moved line", "samples", line))
                for _ in range(_PATHS_A_WORLD):
                    start, end = down + rng.uniform(-0.12, 0.12, 6), down + rng.uniform(-0.12, 0.12, 6)
                    jobs.append(("path", "path", [start.tolist(), end.tolist()]))
                for what, kind, payload in jobs:
                    was, now = _both(off, on, arm, kind, payload)
                    self.assertEqual(was, now, f"{name} case {case}, {what}")
                    judged[what] += 1
                    refused += was is not None
        self.assertEqual(6, judged["route"])
        self.assertGreaterEqual(judged["moved line"], 200, "the lines moved off the grasp: too few were in reach")
        self.assertEqual(120, judged["path"])
        total = sum(judged.values())
        self.assertGreater(refused, total // 5, "the control: the worlds refuse a fair share of what they are asked")
        self.assertLess(refused, total - total // 5, "the control: the worlds accept a fair share of what they are asked")


def _down_from_the_goal(millimetres: int = 30) -> PathSamples:
    """The TCP straight down from F1's goal, 100 mm over its target, one solution a millimetre: the hand only translates."""
    recorded = frame("F1")
    goal = np.asarray(recorded["goal"])
    previous = owner.nearest_solution(goal, recorded["down_rad"])
    assert previous is not None
    configs = [tuple(float(v) for v in previous)]
    for k in range(1, millimetres + 1):
        lower = goal.copy()
        lower[2, 3] -= float(k)
        following = owner.nearest_solution(lower, previous)
        assert following is not None
        configs.append(tuple(float(v) for v in following))
        previous = following
    return PathSamples(configs=tuple(configs), step_bound_mm=1.0)


def _parts_over(preflight: SafetyPreflight, arm: Any, joints: Any, centre: Any) -> "dict[str, float]":
    """Every part's exact distance at ``joints`` from a slab centred at ``centre``, millimetres, by name."""
    guard = preflight._path_authority(arm)
    assert guard is not None
    backend: Any = guard._exact_mesh_backend("ur10")
    slab = backend._a.box_object(np.asarray(_SLAB_HALF_MM), np.asarray(centre, dtype=np.float64))
    frames = ur_link_transforms_mm("ur10", np.asarray(joints, dtype=np.float64))
    assert frames is not None
    out: dict[str, float] = {}
    for name in backend._names:
        placed = frames[backend._frame[name]]
        backend._a.set_transform(backend._models[name], placed[:3, :3], placed[:3, 3])
        out[name] = backend._a.distance(backend._models[name], slab)
    return out


def _slab(preflight: SafetyPreflight, arm: Any, path: PathSamples, sample: int, distance_mm: float,
          name: str = "slab") -> "tuple[AxisAlignedBox, dict[str, float]]":
    """A slab under the hand whose top lies ``distance_mm`` from the nearest part at ``sample`` (from 0), and what every
    part keeps from it there."""
    tcp = np.asarray(frame("F1")["goal"])[:3, 3]
    below = np.array([tcp[0], tcp[1], tcp[2] - 200.0])
    nearest = min(_parts_over(preflight, arm, path.configs[sample], below).values())
    centre = below + np.array([0.0, 0.0, nearest - distance_mm])
    return (AxisAlignedBox(center_mm=centre, half_extents_mm=np.asarray(_SLAB_HALF_MM), name=name),
            _parts_over(preflight, arm, path.configs[sample], centre))


class TheBorderlineIsJudgedAsEverySampleIsTests(unittest.TestCase):
    """A slab under the Hand-E's fingers, the TCP going straight down a millimetre a sample."""

    def setUp(self) -> None:
        owner.needs_the_engine()

    def _judge(self, *boxes: AxisAlignedBox, path: "PathSamples | None" = None) -> Any:
        off, on, arm = _pair()
        for preflight in (off, on):
            preflight.set_perceived_obstacles(boxes)
        was, now = _both(off, on, arm, "samples", path if path is not None else _down_from_the_goal())
        self.assertEqual(was, now)
        return was

    def test_a_sample_3_mm_from_the_slab_give_or_take_a_tenth_of_a_micron_is_judged_alike(self) -> None:
        off, _, arm = _pair()
        path = _down_from_the_goal()
        for delta in (1e-4, -1e-4, 1e-7, -1e-7):
            with self.subTest(delta=delta):
                slab, kept = _slab(off, arm, path, 10, owner.GUARD_MM + delta)
                self.assertAlmostEqual(owner.GUARD_MM + delta, min(kept.values()), delta=1e-9, msg="the premise")
                said = self._judge(slab)
                if abs(delta) > 1e-6:
                    self.assertEqual(11 if delta < 0 else 12, _sample_named(said), said)

    def test_a_declared_fixture_3_mm_from_a_sample_give_or_take_a_tenth_of_a_micron_is_judged_alike(self) -> None:
        off, _, arm = _pair()
        path = _down_from_the_goal()
        for delta in (1e-4, -1e-4):
            with self.subTest(delta=delta):
                slab, _ = _slab(off, arm, path, 10, owner.GUARD_MM + delta, name="bench")
                fixture = {"name": "bench", "center_mm": [float(v) for v in slab.center_mm],
                           "half_extents_mm": [float(v) for v in slab.half_extents_mm]}
                said = []
                for whole in (False, True):
                    preflight, declared_arm = owner.owner_cell(whole=whole, fixtures=[fixture])
                    said.append(owner.verdict(preflight.gate_joint_path(path, arm=declared_arm,
                                                                        command=MotionCommand.MOVE_TO)))
                self.assertEqual(said[0], said[1])
                self.assertEqual(11 if delta < 0 else 12, _sample_named(said[0]), said[0])
                self.assertIn("|fixture:bench", said[0][1])

    def test_a_refusal_at_the_first_sample_and_at_the_last_is_judged_alike(self) -> None:
        off, _, arm = _pair()
        path = _down_from_the_goal()
        first, _ = _slab(off, arm, path, 0, owner.GUARD_MM - 0.5)
        self.assertEqual(1, _sample_named(self._judge(first)))
        last, _ = _slab(off, arm, path, len(path.configs) - 1, owner.GUARD_MM - 0.5)
        self.assertEqual(len(path.configs), _sample_named(self._judge(last)))
        clear, _ = _slab(off, arm, path, len(path.configs) - 1, owner.GUARD_MM + 0.5)
        self.assertIsNone(self._judge(clear), "the control: a slab 3.5 mm under the lowest sample")

    def test_two_fingers_refusing_at_one_sample_name_the_first_in_the_guards_order(self) -> None:
        off, _, arm = _pair()
        path = _down_from_the_goal()
        slab, kept = _slab(off, arm, path, 10, owner.GUARD_MM - 0.4)
        self.assertLess(kept["lfinger"], owner.GUARD_MM, "the premise: both fingers inside the guard's distance")
        self.assertLess(kept["rfinger"], owner.GUARD_MM, "the premise: both fingers inside the guard's distance")
        said = self._judge(slab)
        self.assertEqual(11, _sample_named(said))
        self.assertIn("lfinger|fixture:slab", said[1])

    def test_two_boxes_refusing_at_one_sample_name_the_first(self) -> None:
        off, _, arm = _pair()
        path = _down_from_the_goal()
        first, _ = _slab(off, arm, path, 10, owner.GUARD_MM - 0.5, name="first")
        second = AxisAlignedBox(center_mm=first.center_mm, half_extents_mm=first.half_extents_mm, name="second")
        said = self._judge(first, second)
        self.assertIn("|fixture:first", said[1])
        said = self._judge(second, first)
        self.assertIn("|fixture:second", said[1])

    def test_a_turned_base_and_a_declared_fixture_are_judged_alike(self) -> None:
        """The Isaac cell's 180 degree base and a declared box under the hand (``test_joint_path_gate.py``'s UR5e and
        2F-85, 10 mm): 30 seeded joint paths about the clear configuration, about half of them into the box."""
        from tests.test_joint_path_gate import _CLEAR, _Arm, _preflight

        flange = ur_link_transforms_mm("ur5e", np.asarray(_CLEAR))[-1][:3, 3]  # type: ignore[index]
        turned = np.array([-flange[0], -flange[1], flange[2]])  # where the 180 degree base puts it
        box = {"name": "box", "center_mm": [float(turned[0]), float(turned[1]), float(turned[2] - 120.0)],
               "half_extents_mm": [60.0, 60.0, 40.0]}
        off, on = (_preflight(kinematics_base_yaw_deg=180.0, min_distance_mm=10.0, fixtures=[box],
                              whole_path_judge=whole) for whole in (False, True))
        rng = np.random.default_rng(180)
        refused = 0
        for _ in range(30):
            start, end = (np.asarray(_CLEAR) + rng.uniform(-0.4, 0.4, 6) for _ in range(2))
            was, now = (owner.verdict(preflight.gate_planned_path([start.tolist(), end.tolist()], arm=_Arm(),
                                                                  command=MotionCommand.MOVE_TO))
                        for preflight in (off, on))
            self.assertEqual(was, now)
            refused += was is not None
        self.assertTrue(6 <= refused <= 24, f"the control: {refused} of 30 refused")

    def test_the_arm_folded_into_itself_names_the_same_first_pair(self) -> None:
        from tests.test_joint_path_gate import _CLEAR, _FOLDED, _Arm, _preflight

        off, on = _preflight(), _preflight(whole_path_judge=True)
        # The last fifth of the way from the clear configuration into the folded one, which test_joint_path_gate.py
        # refuses as forearm|lfinger 0.367 mm apart.
        near = [f + 0.2 * (c - f) for c, f in zip(_CLEAR, _FOLDED)]
        for waypoints in ([near, _FOLDED], [_FOLDED, near], [near, _FOLDED, near]):
            with self.subTest(waypoints=len(waypoints)):
                was, now = (owner.verdict(preflight.gate_planned_path([list(w) for w in waypoints], arm=_Arm(),
                                                                      command=MotionCommand.MOVE_TO))
                            for preflight in (off, on))
                self.assertIsNotNone(was, "the control: the folded configuration is refused")
                self.assertEqual(was, now)
                self.assertIn("forearm|", was[1])


class _SiteLocal:
    """A guard of a cell's own, run last, that offers no whole-path pass: it counts what it is asked and accepts."""

    name = "site_local"

    def __init__(self) -> None:
        self.asked = 0

    def evaluate(self, ctx: Any) -> SafetyDecision:
        self.asked += 1
        return SafetyDecision.accept(self.name)


def _limits_crossed_at(path: PathSamples, sample: int) -> JointLimitSafetyConfig:
    """Static joint limits, no margin, that the joint turning furthest along ``path`` leaves at ``sample`` (from 0)."""
    degrees = np.degrees(np.asarray(path.configs, dtype=np.float64))
    axis = int(np.argmax(np.abs(degrees[-1] - degrees[0])))
    steps = np.diff(degrees[:, axis])
    assert np.all(steps > 0.0) or np.all(steps < 0.0), "the premise: the joint turns one way along the line"
    lower, upper = [-360.0] * 6, [360.0] * 6
    crossing = (degrees[sample - 1, axis] + degrees[sample, axis]) / 2.0
    if steps[0] > 0.0:
        upper[axis] = float(crossing)
    else:
        lower[axis] = float(crossing)
    return JointLimitSafetyConfig(min_deg=lower, max_deg=upper, margin_deg=0.0)


class TheGuardOrderIsTheLoopsTests(unittest.TestCase):
    """The first refusal a guard of the loop names, named the same where the guards judge the path whole first."""

    def setUp(self) -> None:
        owner.needs_the_engine()

    def _guarded(self, *guards: Any) -> "tuple[SafetyPreflight, SafetyPreflight]":
        return SafetyPreflight(guards), SafetyPreflight(guards, whole_path_judge=True)

    def test_a_joint_limit_before_at_and_after_the_first_collision_is_named_as_the_loop_names_it(self) -> None:
        off, _, arm = _pair()
        collision = off._path_authority(arm)
        path = _down_from_the_goal()
        slab, _ = _slab(off, arm, path, 15, owner.GUARD_MM - 0.5)
        collision.set_perceived_fixtures((slab,))
        for crossing, guard, sample in ((8, "joint_limit", 9), (15, "joint_limit", 16), (22, "self_collision", 16)):
            with self.subTest(crossing=crossing):
                was_gate, now_gate = self._guarded(JointLimitGuard(_limits_crossed_at(path, crossing)), collision)
                was, now = (owner.verdict(gate.gate_joint_path(path, arm=arm, command=MotionCommand.MOVE_TO))
                            for gate in (was_gate, now_gate))
                self.assertEqual(was, now)
                self.assertIn(f"[safety:{guard}/", was[1])
                self.assertEqual(sample, _sample_named(was))

    def test_a_payload_the_guard_refuses_is_refused_at_the_first_sample(self) -> None:
        off, _, arm = _pair()
        collision = off._path_authority(arm)
        collision.set_perceived_fixtures(())
        heavy = PayloadGuard(PayloadSafetyConfig.model_construct(enforce=True, mass_kg=12.0, max_mass_kg=10.0,
                                                                 cog_mm=(0.0, 0.0, 65.0), inertia_kgm2=(0.0, 0.0, 0.0)))
        was_gate, now_gate = self._guarded(collision, heavy)
        was, now = (owner.verdict(gate.gate_joint_path(_down_from_the_goal(), arm=arm, command=MotionCommand.MOVE_TO))
                    for gate in (was_gate, now_gate))
        self.assertEqual(was, now)
        self.assertIn("[safety:payload/", was[1])
        self.assertEqual(1, _sample_named(was))

    def test_a_guard_without_a_whole_path_pass_leaves_every_sample_to_the_loop(self) -> None:
        off, on, arm = _pair()
        on.set_perceived_obstacles(())
        collision = on._path_authority(arm)
        path = _down_from_the_goal()
        with mock.patch.object(collision, "evaluate", wraps=collision.evaluate) as asked:
            self.assertIsNone(on.gate_joint_path(path, arm=arm, command=MotionCommand.MOVE_TO))
        self.assertEqual(0, asked.call_count, "the control: the whole path passed on its proofs")
        local = _SiteLocal()
        _, judged = self._guarded(*on.guards, local)
        with mock.patch.object(collision, "evaluate", wraps=collision.evaluate) as asked:
            self.assertIsNone(judged.gate_joint_path(path, arm=arm, command=MotionCommand.MOVE_TO))
        self.assertEqual(len(path.configs), local.asked)
        self.assertEqual(len(path.configs), asked.call_count)

    def test_without_an_arm_the_capsule_proxy_judges_every_sample(self) -> None:
        off, _, arm = _pair()
        collision = off._path_authority(arm)
        slab, _ = _slab(off, arm, _down_from_the_goal(), 10, owner.GUARD_MM - 0.5)
        collision.set_perceived_fixtures((slab,))
        self.assertIsNone(collision.first_suspect(None, np.asarray(_down_from_the_goal().configs), 31))
        was_gate, now_gate = self._guarded(collision)
        with mock.patch.object(collision, "evaluate", wraps=collision.evaluate) as asked:
            was, now = (owner.verdict(gate.gate_joint_path(_down_from_the_goal(), arm=None,
                                                           command=MotionCommand.MOVE_TO))
                        for gate in (was_gate, now_gate))
            self.assertEqual(was, now)
        self.assertEqual(2 * (len(_down_from_the_goal().configs) if was is None else _sample_named(was)),
                         asked.call_count, "every sample asked of the guard, both times")


class TheKeyTests(unittest.TestCase):
    def test_the_key_loads_and_is_off_by_default(self) -> None:
        self.assertFalse(SelfCollisionSafetyConfig().whole_path_judge)
        self.assertFalse(RobotConfig.model_validate({"vendor": "ur"}).safety.self_collision.whole_path_judge)
        self.assertTrue(SelfCollisionSafetyConfig.model_validate({"whole_path_judge": True}).whole_path_judge)

    def test_off_every_guard_sees_every_sample_and_on_the_gate_reads_it(self) -> None:
        owner.needs_the_engine()
        off, on, arm = _pair()
        path = _down_from_the_goal()
        for preflight, every in ((off, True), (on, False)):
            preflight.set_perceived_obstacles(())
            guards = [guard for guard in preflight.guards if guard.name in ("joint_limit", "self_collision", "payload")]
            self.assertEqual(3, len(guards), "the premise: the owner's cell asks all three of a joint move")
            patches = [mock.patch.object(guard, "evaluate", wraps=guard.evaluate) for guard in guards]
            asked = [patch.start() for patch in patches]
            try:
                self.assertIsNone(preflight.gate_joint_path(path, arm=arm, command=MotionCommand.MOVE_TO))
            finally:
                for patch in patches:
                    patch.stop()
            for guard, counted in zip(guards, asked):
                # On, the payload guard's whole-path pass is its own verdict at the first sample, asked once.
                expected = len(path.configs) if every else (1 if guard.name == "payload" else 0)
                self.assertEqual(expected, counted.call_count, guard.name)


class MeshFirstSaysTheSameTests(unittest.TestCase):
    """``URRobotArm._mesh_first_refusal`` with the switch on: the plane and the base asked of the whole line at once,
    and the same sentence, from the configuration the switch off names."""

    def setUp(self) -> None:
        owner.needs_the_engine()

    def _arm(self, *, whole: bool, plane_mm: "float | None") -> Any:
        """The owner's arm, mesh first on, the support plane declared at ``plane_mm`` (``None``: a metre down, out of
        every reach)."""
        from src.robot.drivers.ur.arm import URRobotArm

        robot = owner.owner_robot(whole=whole)
        robot["safety"]["planning_world"]["support_plane"] = {
            "height_mm": -1000.0 if plane_mm is None else plane_mm, "extent_mm": [900.0, 700.0], "thickness_mm": 50.0,
            "center_mm": [-10.0, -585.0]}
        robot["safety"]["planned_motion"] = {"mesh_first": True}
        return URRobotArm(RobotConfig.model_validate(robot))

    def _base_arm(self, *, whole: bool) -> Any:
        """The UR10 and the Hand-E of the base review (``HowNearTheBaseTests`` in
        ``test_a_joint_path_is_judged_whole_and_sent_without_a_pause.py``): 10 mm from the base, no planning world."""
        from src.robot.drivers.ur.arm import URRobotArm
        from tests._plan_end import OPEN_WORKSPACE

        return URRobotArm(RobotConfig.model_validate({
            "vendor": "ur", "ur": {"model": "ur10", "motion_planner": "curobo"}, "workspace_limits": OPEN_WORKSPACE,
            "gripper": {"model": "robotiq_hande", "coupling_plates": [{"name": "adapter", "thickness_mm": 20.0}],
                        "tool_frame": {"source": "willy", "offset_mm": [0.0, 0.0, 155.75],
                                       "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}},
            "safety": {"payload": {"enforce": False}, "planned_motion": {"mesh_first": True},
                       "self_collision": {"backend": "fcl", "min_distance_mm": 10.0, "perceived_min_distance_mm": 5.0,
                                          "kinematics_model": "ur10", "planner_margin_mm": 4.0,
                                          "whole_path_judge": whole}},
        }))

    def _sentences(self, configs: "list[list[float]]", *, plane_mm: "float | None" = None,
                   base: bool = False) -> "tuple[Any, Any]":
        from tests._route_planner import RoutePlanner

        class _Glue(RoutePlanner):
            carries_part = False

        said = []
        for whole in (False, True):
            arm = self._base_arm(whole=whole) if base else self._arm(whole=whole, plane_mm=plane_mm)
            pairs = arm._preflight.exact_pairs(arm)
            self.assertEqual(whole, pairs.lows is not None and pairs.near_base_from is not None)
            said.append(arm._mesh_first_refusal(_Glue(here=configs[0]), configs))
        return said[0], said[1]

    def test_the_support_plane_is_named_from_the_same_configuration(self) -> None:
        line = [list(q) for q in _down_from_the_goal().configs]
        arm = self._arm(whole=False, plane_mm=None)
        lowest = arm._preflight.exact_pairs(arm).lowest
        heights = [float(lowest(q)) for q in line]
        plane = min(heights) - 1.5
        first = next(index for index, height in enumerate(heights) if height < plane + owner.GUARD_MM)
        was, now = self._sentences(line, plane_mm=plane)
        self.assertIn(f"comes {heights[first] - plane:.1f} mm over the declared support plane", was or "")
        self.assertEqual(was, now)
        was, now = self._sentences(line, plane_mm=min(heights) - 3.5)
        self.assertIsNone(was, "the control: the whole line keeps 3.5 mm over the plane")
        self.assertEqual(was, now)

    def test_the_robots_base_is_named_from_the_same_configuration(self) -> None:
        from tests.test_a_pose_only_the_planners_spheres_refuse_is_the_exact_guards import BASE_NEAR, BASE_START

        line = [list(q) for q in waypoint_path_samples([list(BASE_START), list(BASE_NEAR)],
                                                       reach_mm=(2000.0,) * 6, max_step_mm=3.0).configs]
        was, now = self._sentences(line, base=True)
        self.assertIn("from the robot's base, within the guard's 10 mm", was or "")
        self.assertEqual(was, now)
        away = [[0.0, -1.5, 1.5, 0.0, 0.0, 0.0], [0.1, -1.5, 1.5, 0.0, 0.0, 0.0]]
        was, now = self._sentences(away, base=True)
        self.assertIsNone(was, "the control: a line far from the base")
        self.assertEqual(was, now)


if __name__ == "__main__":
    unittest.main()
