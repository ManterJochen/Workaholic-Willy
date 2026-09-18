"""The space between the jaws at a motion's goal is not an obstacle, and every motion says what it left out.

A pick drives the open jaws around its part, so the goal of the approach has the part between the fingers, and a
planner that registered that part as an obstacle has no plan by construction. The pick loop holds its target out, but
user code that calls ``move`` has no target to hold. So every refresh a motion makes leaves out the space between the
pads at its goal, laid out by the hand's registry jaw on the declared TCP, with no padding: a finger closing on a wall
is still checked, and a part wider than the fingers keeps its box.

The region is computed only where a live world is wired, so every shipped profile, which wires none, is unchanged, and a
stamp built with no region is today's stamp.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.config.schema.robot import RobotConfig
from src.robot.core import JointPositions, MotionCommand
from src.robot.core.camera_world import CameraWorldStamp, CameraWorldUse
from src.robot.drivers.sim import arm as sim_arm_module
from src.robot.drivers.sim.arm import IsaacRobotArm
from src.robot.drivers.sim.config import SimRobotConfig
from src.robot.drivers.ur import curobo_motion
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.drivers.ur.curobo_motion import UR_ARM_JOINT_NAMES
from src.robot.drivers.ur.tool_frame import tool_frame_matrix
from src.robot.safety._ur_kinematics import ur_link_transforms_mm
from src.robot.safety.planning.hand import planner_hand
from src.robot.safety.planning.live_world import DepthSnapshot, refresh_planner_world
from src.robot.safety.planning.perceived import LinkCapsule, SelfEnvelope, WorldBuildTuning
from src.robot.safety.planning.self_envelope import yawed_link_transforms_mm
from tests.test_a_refused_placement_never_reaches_the_guard import _CLOCKED_45, _cell
from tests.test_every_verb_meets_the_camera_world import (
    _HERE,
    _THERE,
    _Camera as _BenchCamera,
    _Client,
    _mesh_backend_available,
    _pose,
    _ur,
)
from tests.test_every_verb_meets_the_camera_world import _world as _bench_world
from tests.test_live_planner_world import (
    _CAMERA_HEIGHT_MM,
    _CX,
    _CY,
    _FX,
    _INTRINSICS,
    _SELF,
    _SHAPE,
    _block_top_points,
    _Camera,
    _Planner,
    _scene_with_a_block,
    _world,
)

#: The 2F-85's registry jaw (config/grippers/robotiq_2f85.yaml).
_JAW_2F85 = {"aperture_mm": 85.0, "finger_width_mm": 27.0, "pad_ahead_mm": 23.61, "pad_behind_mm": 14.39}
#: The Isaac frame: approach along flange +Y (tests/test_the_hand_is_placed_by_the_declared_frame.py).
_SIM_ROTATION = (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)


def _goal_keep_out(*args: object):  # noqa: ANN202
    from src.robot.safety.planning.self_envelope import goal_keep_out

    return goal_keep_out(*args)  # type: ignore[arg-type]


def _types():  # noqa: ANN202
    from src.robot.core.keep_out import GoalKeepOut, KeepOutBox

    return GoalKeepOut, KeepOutBox


def _tcp(x: float, y: float, z: float, *, closing_along_y: bool = False) -> np.ndarray:
    """A TCP approaching straight down, closing along BASE X, or along BASE Y when turned a quarter about its approach."""
    tcp = np.eye(4)
    tcp[:3, :3] = ([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]] if closing_along_y
                   else [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    tcp[:3, 3] = (x, y, z)
    return tcp


def _jaw_region(tcp: np.ndarray):  # noqa: ANN202
    _, KeepOutBox = _types()
    return KeepOutBox.from_jaw(tcp_to_base_mm=tcp, **_JAW_2F85)


def _goal(region: object):  # noqa: ANN202
    GoalKeepOut, _ = _types()
    return GoalKeepOut(region=region)  # type: ignore[arg-type]


def _small_block(at_mm: tuple[float, float], *, half_mm: float = 10.0, height_mm: float = 80.0) -> np.ndarray:
    """A bench with one block of ``2 * half_mm`` on it, ``height_mm`` tall."""
    depth = np.full(_SHAPE, _CAMERA_HEIGHT_MM, dtype=np.float64)
    top = _CAMERA_HEIGHT_MM - height_mm
    scale = top / _FX
    rows, cols = np.mgrid[0 : _SHAPE[0], 0 : _SHAPE[1]]
    mask = (np.abs(cols - (_CX + at_mm[0] / scale)) <= half_mm / scale) & (
        np.abs(rows - (_CY - at_mm[1] / scale)) <= half_mm / scale
    )
    depth[mask] = top
    return depth


#: The block narrower than the fingers is 20 mm across, a few voxels, fewer than the default 12 points a box needs.
_SMALL = WorldBuildTuning(max_boxes=4, min_points=3)


def _world_over(depth: np.ndarray, **kwargs: object):  # noqa: ANN202
    settings: dict[str, object] = {"tuning": _SMALL}
    settings.update(kwargs)
    return _world(_Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0)), **settings)


class _PlacedHand:
    """A preflight that answers one real hand, the way ``SafetyPreflight.planner_hand`` does."""

    def __init__(self, hand: object) -> None:
        self.hand = hand

    def planner_hand(self, arm: object = None) -> object:
        return self.hand


def _declared(model: str, offset_mm: float, *, bolted_to_the_flange: bool = False) -> RobotConfig:
    """A UR cell whose hand ``model`` has its grasp centre declared ``offset_mm`` out along flange +Y.

    A hand whose map starts at its own mounting face needs its plates stated; ``bolted_to_the_flange`` states none.
    """
    gripper: dict[str, object] = {
        "model": model,
        "tool_frame": {"source": "willy", "offset_mm": [0.0, offset_mm, 0.0], "rotation_quat_xyzw": list(_SIM_ROTATION)},
    }
    if bolted_to_the_flange:
        gripper["coupling_plates"] = []
    return RobotConfig.model_validate({"vendor": "ur", "ur": {"model": "ur5e"}, "gripper": gripper})


# ---------------------------------------------------------------------------------------------------
# The region in the world
# ---------------------------------------------------------------------------------------------------


class TheRegionInTheWorldTests(unittest.TestCase):
    def test_points_between_the_jaws_at_the_goal_are_not_obstacles(self) -> None:
        world = _world_over(_small_block((150.0, 0.0)))
        self.assertEqual(1, world.world_for(self_envelope=_SELF, now=100.1).perceived_count)

        snapshot = world.world_for(self_envelope=_SELF, now=100.1, goal_keep_out=_goal(_jaw_region(_tcp(150.0, 0.0, 70.0))))

        self.assertEqual(0, snapshot.perceived_count, snapshot.render())
        assert snapshot.perceived is not None
        self.assertGreater(snapshot.perceived.dropped_points.get("keep_out", 0), 0)

    def test_a_finger_is_still_checked(self) -> None:
        """A block 60 mm out along the closing axis sits where a finger closes: outside the region, so it stays."""
        world = _world_over(_small_block((150.0, 0.0)))

        snapshot = world.world_for(self_envelope=_SELF, now=100.1, goal_keep_out=_goal(_jaw_region(_tcp(90.0, 0.0, 70.0))))

        self.assertEqual(1, snapshot.perceived_count, snapshot.render())

    def test_the_region_follows_the_tcp_axes_not_tool0(self) -> None:
        """30 mm across the fingers is outside the 27 mm finger width; turned a quarter it is inside the aperture."""
        across = _world_over(_small_block((150.0, 0.0)))
        self.assertEqual(1, across.world_for(
            self_envelope=_SELF, now=100.1, goal_keep_out=_goal(_jaw_region(_tcp(150.0, 30.0, 70.0))),
        ).perceived_count)

        turned = _world_over(_small_block((150.0, 0.0)))
        self.assertEqual(0, turned.world_for(
            self_envelope=_SELF, now=100.1,
            goal_keep_out=_goal(_jaw_region(_tcp(150.0, 30.0, 70.0, closing_along_y=True))),
        ).perceived_count)

    def test_the_voxel_field_loses_the_goal_region_too(self) -> None:
        world = _world_over(_small_block((150.0, 0.0)),
                            tuning=WorldBuildTuning(max_boxes=4, min_points=3, voxel_field_mm=30.0))
        before = world.world_for(self_envelope=_SELF, now=100.1)
        assert before.perceived is not None and before.perceived.voxels is not None
        self.assertGreater(before.perceived.voxels.occupied, 0)

        after = world.world_for(self_envelope=_SELF, now=100.1, goal_keep_out=_goal(_jaw_region(_tcp(150.0, 0.0, 70.0))))

        assert after.perceived is not None and after.perceived.voxels is not None
        self.assertEqual(0, after.perceived.voxels.occupied)

    def test_a_part_wider_than_the_fingers_keeps_its_box(self) -> None:
        """The region is not padded, so the 60 mm block's sides beyond the 27 mm fingers stay and its box grows back
        over the region. A bare ``move`` meets that box; a pick holds its target out."""
        depth, _ = _scene_with_a_block(at_mm=(150.0, 0.0))
        world = _world_over(depth)

        snapshot = world.world_for(self_envelope=_SELF, now=100.1, goal_keep_out=_goal(_jaw_region(_tcp(150.0, 0.0, 70.0))))

        self.assertEqual(1, snapshot.perceived_count, snapshot.render())


# ---------------------------------------------------------------------------------------------------
# The region a hand builds
# ---------------------------------------------------------------------------------------------------


class TheRegionAHandBuildsTests(unittest.TestCase):
    def test_the_hande_region_is_the_hande_jaw(self) -> None:
        hand = planner_hand(_declared("robotiq_hande", 136.2, bolted_to_the_flange=True))
        tcp = _tcp(0.0, 0.0, 300.0)

        goal = _goal_keep_out(_PlacedHand(hand), None, tcp)

        assert goal.region is not None, goal.reason
        np.testing.assert_allclose(goal.region.half_extents_mm, (49.99 / 2.0, 29.24 / 2.0, (10.45 + 10.46) / 2.0))
        np.testing.assert_allclose(goal.region.matrix()[:3, :3], tcp[:3, :3])
        # The pad runs 10.45 mm ahead and 10.46 behind, so the box's centre sits 0.005 mm back along the approach.
        np.testing.assert_allclose(goal.region.matrix()[:3, 3], (0.0, 0.0, 300.005), atol=1e-9)

    def test_an_arm_that_names_no_hand_builds_no_region_and_says_why(self) -> None:
        tcp = _tcp(0.0, 0.0, 300.0)
        for label, preflight in (("no preflight", None), ("a guard double", MagicMock())):
            with self.subTest(label):
                goal = _goal_keep_out(preflight, None, tcp)
                self.assertIsNone(goal.region)
                self.assertIn("reads no hand", goal.reason)

        refused = planner_hand(_cell(_CLOCKED_45))
        goal = _goal_keep_out(_PlacedHand(refused), None, tcp)
        self.assertIsNone(goal.region)
        self.assertIn(str(refused.placement_refusal), goal.reason)

        hande = planner_hand(_declared("robotiq_hande", 136.2, bolted_to_the_flange=True))
        goal = _goal_keep_out(_PlacedHand(hande), None, None)
        self.assertIsNone(goal.region)
        self.assertIn("TCP at this goal is not known", goal.reason)

    def test_the_region_sits_on_the_declared_tcp(self) -> None:
        """The region sits on the declared TCP, the frame the verb, the planner and the grasp calculator use.

        A cell declared at its registry grasp centre gets its region there. The shipped 2F-85 frames declare 132.0 mm
        against the registry's 146.5 mm; the region stays on the declared 132.0, 14.5 mm short of the registry, and
        the desk's grasp centre row warns about exactly that.
        """
        cases = (
            ("robotiq_hande", 136.2, 136.2, True),
            ("schunk_egu50", 159.1, 159.1, False),
            ("robotiq_2f85", 132.0, 146.5, False),
        )
        for model, declared, registry, bolted in cases:
            with self.subTest(model):
                config = _declared(model, declared, bolted_to_the_flange=bolted)
                hand = planner_hand(config)
                self.assertAlmostEqual(registry, float(hand.jaw.grasp_centre_mm) + float(hand.coupling_mm))
                tcp = tool_frame_matrix(config.gripper.tool_frame.offset_mm,
                                        config.gripper.tool_frame.rotation_quat_xyzw)

                goal = _goal_keep_out(_PlacedHand(hand), None, tcp)

                assert goal.region is not None, goal.reason
                jaw = hand.jaw
                shift = (float(jaw.pad_ahead_mm) - float(jaw.pad_behind_mm)) / 2.0
                box = goal.region.matrix()
                centre_on_the_approach = box[:3, 3] - box[:3, 2] * shift
                # The flange stands at the origin, and the Isaac frame approaches along its +Y.
                along = float(centre_on_the_approach[1])
                np.testing.assert_allclose(box[:3, 2], (0.0, 1.0, 0.0), atol=1e-9)
                self.assertAlmostEqual(declared, along, places=6)
                self.assertAlmostEqual(registry - declared, registry - along, places=6)
                if model != "robotiq_2f85":
                    self.assertLessEqual(abs(registry - along), 1.0)
                else:
                    self.assertAlmostEqual(14.5, registry - along, places=6)


# ---------------------------------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------------------------------


class TheRecordTests(unittest.TestCase):
    def test_the_refresh_and_the_stamp_record_the_keep_out(self) -> None:
        from src.robot.core.keep_out import GoalKeepOut, KeepOutSummary

        world = _world_over(_small_block((150.0, 0.0)))
        refresh = refresh_planner_world(source=world, client=_Planner(), self_envelope=_SELF, now=100.1,
                                        goal_keep_out=_goal(_jaw_region(_tcp(150.0, 0.0, 70.0))))

        self.assertTrue(refresh.ok, refresh.render())
        assert refresh.keep_out is not None
        self.assertGreater(refresh.keep_out.goal_points or 0, 0)
        self.assertIn("kept out: goal region left out", refresh.render())
        self.assertEqual(refresh.keep_out.goal_points, json.loads(json.dumps(refresh.to_dict()))["keep_out"]["goal_points"])
        stamp = refresh.camera_world()
        assert stamp is not None
        self.assertIs(CameraWorldUse.PLANNED, stamp.use)
        self.assertEqual(refresh.keep_out, stamp.keep_out)
        self.assertIn("kept out: goal region left out", stamp.render())
        self.assertEqual(refresh.keep_out.goal_points, stamp.to_dict()["keep_out"]["goal_points"])

        # No region and nothing held: the refresh keeps the reason, the stamp carries nothing.
        why = GoalKeepOut(region=None, reason="the TCP at this goal is not known, so no goal region is placed")
        bare = refresh_planner_world(source=_world_over(_small_block((150.0, 0.0))), client=_Planner(),
                                     self_envelope=_SELF, now=100.1, goal_keep_out=why)
        assert bare.keep_out is not None
        self.assertIsNone(bare.keep_out.goal_points)
        self.assertEqual(why.reason, bare.keep_out.goal_reason)
        stamp = bare.camera_world()
        assert stamp is not None
        self.assertIsNone(stamp.keep_out)

        # Only a planned motion can have left anything out.
        with self.assertRaises(ValueError):
            CameraWorldStamp(use=CameraWorldUse.MISSING, reason="no world", keep_out=KeepOutSummary(goal_points=3))


    def test_a_held_box_is_named_in_the_record_by_its_camera_and_its_shutter(self) -> None:
        depth, _ = _scene_with_a_block(at_mm=(150.0, 0.0))
        world = _world_over(depth)
        world.offer_segmentation(camera="overhead", target_points_base_mm=_block_top_points(), timestamp=100.0,
                                 hold=True)

        refresh = refresh_planner_world(source=world, client=_Planner(), self_envelope=_SELF, now=100.1)

        assert refresh.keep_out is not None
        self.assertIsNone(refresh.keep_out.goal_points)
        ((camera, stamp, points),) = refresh.keep_out.held
        self.assertEqual(("overhead", 100.0), (camera, stamp))
        self.assertGreater(points, 0)
        stamped = refresh.camera_world()
        assert stamped is not None
        self.assertEqual(refresh.keep_out, stamped.keep_out, "a held box was in force, so the stamp says so")


# ---------------------------------------------------------------------------------------------------
# Every verb hands its goal to the world
# ---------------------------------------------------------------------------------------------------


def _regions(refresh: MagicMock) -> list[object]:
    """The goal region each refresh a motion made was handed; every call must have been handed one."""
    GoalKeepOut, _ = _types()
    found = []
    for call in refresh.call_args_list:
        goal = call.kwargs.get("goal_keep_out")
        assert isinstance(goal, GoalKeepOut), f"a refresh was made with no goal: {call.kwargs}"
        found.append(goal)
    return found


def _expected_at(tcp: np.ndarray) -> np.ndarray:
    return _jaw_region(tcp).matrix()


class EveryUrVerbHandsTheWorldItsGoalTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def _arm(self) -> URRobotArm:
        return _ur(_bench_world(_BenchCamera(silent=False)), _Client(joint_names=UR_ARM_JOINT_NAMES), [])

    def test_every_ur_verb_hands_the_world_its_goal(self) -> None:
        pose_goal = _pose()
        verbs = {
            "move": (lambda arm: arm.move(pose_goal), "pose"),
            "move_to": (lambda arm: arm.move_to(pose_goal), "pose"),
            "amove_to": (lambda arm: asyncio.run(arm.amove_to(pose_goal)), "pose"),
            "move, linear": (lambda arm: arm.move(pose_goal, linear=True), "pose"),
            "move_linear": (lambda arm: arm.move_linear(pose_goal), "pose"),
            "move_to_joints": (lambda arm: arm.move_to_joints(JointPositions(_THERE)), "joints"),
            "move_joint": (lambda arm: arm.move_joint(JointPositions(_THERE)), "joints"),
            "move_home": (lambda arm: arm.move_home(), "joints"),
            "amove_home": (lambda arm: asyncio.run(arm.amove_home()), "joints"),
        }
        for label, (verb, kind) in verbs.items():
            with self.subTest(label):
                arm = self._arm()
                with patch.object(curobo_motion, "refresh_planner_world",
                                  wraps=curobo_motion.refresh_planner_world) as refresh:
                    verb(arm)
                goals = _regions(refresh)
                self.assertGreaterEqual(len(goals), 1, "the verb refreshed nothing")
                if kind == "pose":
                    expected = _expected_at(pose_goal.to_matrix())
                else:
                    expected = _expected_at(self._joint_tcp(arm, _THERE))
                for goal in goals:
                    assert goal.region is not None, goal.reason  # type: ignore[attr-defined]
                    np.testing.assert_allclose(goal.region.matrix(), expected, atol=1e-6)  # type: ignore[attr-defined]

    @staticmethod
    def _joint_tcp(arm: URRobotArm, joints: tuple[float, ...]) -> np.ndarray:
        frames = ur_link_transforms_mm(str(arm.config.ur.model), np.asarray(joints, dtype=np.float64))
        assert frames is not None
        tf = arm.config.gripper.tool_frame
        return frames[-1] @ tool_frame_matrix(tf.offset_mm, tf.rotation_quat_xyzw)

    def test_a_ur_joint_goal_region_sits_at_the_flange_times_the_declared_frame(self) -> None:
        arm = self._arm()

        with patch.object(curobo_motion, "refresh_planner_world", wraps=curobo_motion.refresh_planner_world) as refresh:
            arm.move_to_joints(JointPositions(_THERE))

        (goal,) = _regions(refresh)
        assert goal.region is not None  # type: ignore[attr-defined]
        box = goal.region.matrix()  # type: ignore[attr-defined]
        flange = ur_link_transforms_mm(str(arm.config.ur.model), np.asarray(_THERE, dtype=np.float64))
        assert flange is not None
        # The Isaac frame of the fixture: the grasp centre 132 mm out along flange +Y, and the region's approach with it.
        np.testing.assert_allclose(box[:3, 2], flange[-1][:3, 1], atol=1e-9)
        shift = (_JAW_2F85["pad_ahead_mm"] - _JAW_2F85["pad_behind_mm"]) / 2.0
        np.testing.assert_allclose(box[:3, 3], flange[-1][:3, 3] + flange[-1][:3, 1] * (132.0 + shift), atol=1e-6)
        record = arm._curobo_ur_planner().last_world_refresh
        assert record is not None and record.keep_out is not None
        self.assertIsNotNone(record.keep_out.goal_points)

    def test_an_undeclared_ur_tool_frame_records_why_no_region(self) -> None:
        arm = self._arm()
        arm._declared_tool_matrix = lambda: None  # type: ignore[method-assign]

        refused = arm._refresh_before_the_path_gate(
            near_point_mm=None, command=MotionCommand.MOVE_JOINTS, target_joints=JointPositions(_THERE),
            goal_tcp_mm=arm._tcp_at_joints_mm(JointPositions(_THERE)),
        )

        self.assertIsNone(refused)
        record = arm._curobo_ur_planner().last_world_refresh
        assert record is not None and record.keep_out is not None
        self.assertIsNone(record.keep_out.goal_points)
        self.assertIn("robot.gripper.tool_frame is undeclared", record.keep_out.goal_reason)


def _sim_arm(client: _Client) -> IsaacRobotArm:
    """The every-verb sim double, with a real 2F-85 on its guard and the Isaac tool frame on its config."""
    config = SimRobotConfig(enabled=True, scene="placeholder.usd", robot_prim_path="/World/Robot",
                            tool_offset_mm=(0.0, 132.0, 0.0), tool_rotation_quat_xyzw=_SIM_ROTATION)
    arm = IsaacRobotArm(config)
    arm._connected = True
    arm._articulation = object()  # type: ignore[assignment]
    arm._kin_solver = object()  # type: ignore[assignment]
    arm._rmpflow = object()  # type: ignore[assignment]
    arm._arm_subset = MagicMock()
    arm._arm_subset.get_joint_positions.return_value = np.asarray(_HERE, dtype=np.float64)
    arm._resolve_ik = lambda pose, **_: JointPositions(_HERE)  # type: ignore[method-assign]
    arm.get_joint_positions = lambda: JointPositions(_HERE)  # type: ignore[method-assign]
    arm.get_tcp_pose = lambda: _pose(x=300.0)  # type: ignore[method-assign]
    arm._get_curobo_client = lambda: client  # type: ignore[method-assign, assignment, return-value]
    arm._self_envelope = lambda: SelfEnvelope(  # type: ignore[method-assign]
        frames_mm=(np.eye(4),),
        capsules=(LinkCapsule(frame=0, start_mm=(0.0, 0.0, 0.0), end_mm=(0.0, 0.0, 300.0), radius_mm=135.0),),
    )
    arm._drive_joints = MagicMock()  # type: ignore[method-assign]
    arm._home_joints = JointPositions(_THERE)
    guard = MagicMock()
    guard.path_step_mm = 10.0
    guard.joint_radii_mm.return_value = (500.0,) * 6
    for name in ("gate_planned_path", "gate_joint_path", "gate_joint_target"):
        getattr(guard, name).return_value = None
    # The Isaac cell's guard: the ur5e turned 180 degrees about the base, and a real hand.
    guard.self_kinematics.return_value = ("ur5e", 180.0)
    guard.planner_hand.return_value = planner_hand(_declared("robotiq_2f85", 132.0))
    arm._preflight = guard
    arm.set_live_planner_world(_bench_world(_BenchCamera(silent=False)))
    return arm


class EverySimVerbHandsTheWorldItsGoalTests(unittest.TestCase):
    @staticmethod
    def _joint_tcp(joints: tuple[float, ...]) -> np.ndarray:
        frames = yawed_link_transforms_mm("ur5e", np.asarray(joints, dtype=np.float64), 180.0)
        assert frames is not None
        return frames[-1] @ tool_frame_matrix((0.0, 132.0, 0.0), _SIM_ROTATION)

    def test_every_sim_verb_hands_the_world_its_goal(self) -> None:
        pose_goal = _pose(x=400.0)
        verbs = {
            "move": (lambda arm: arm.move(pose_goal), "pose"),
            "move, linear": (lambda arm: arm.move(pose_goal, linear=True), "pose"),
            "move_to_joints": (lambda arm: arm.move_to_joints(JointPositions(_THERE)), "joints"),
            "move_joint": (lambda arm: arm.move_joint(JointPositions(_THERE)), "joints"),
            "move_home": (lambda arm: arm.move_home(), "joints"),
            "the planned joint path": (
                lambda arm: arm._plan_joint_path(np.asarray(_HERE), np.asarray(_THERE)), "joints"),
        }
        for label, (verb, kind) in verbs.items():
            with self.subTest(label):
                client = _Client(joint_names=sim_arm_module._ARM_JOINT_NAMES)
                client.plan_joint = lambda start, target: [list(start), list(target)]  # type: ignore[attr-defined]
                client.dt = 0.01  # type: ignore[attr-defined]
                arm = _sim_arm(client)
                # What runs after the plan is not this test's: the refresh before it is.
                arm._execute_curobo_trajectory = MagicMock()  # type: ignore[method-assign]
                with patch.object(sim_arm_module, "refresh_planner_world",
                                  wraps=sim_arm_module.refresh_planner_world) as refresh:
                    verb(arm)
                goals = _regions(refresh)
                self.assertGreaterEqual(len(goals), 1, "the verb refreshed nothing")
                expected = _expected_at(pose_goal.to_matrix() if kind == "pose" else self._joint_tcp(_THERE))
                for goal in goals:
                    assert goal.region is not None, goal.reason  # type: ignore[attr-defined]
                    np.testing.assert_allclose(goal.region.matrix(), expected, atol=1e-6)  # type: ignore[attr-defined]

    def test_a_sim_joint_move_stamps_what_it_left_out(self) -> None:
        arm = _sim_arm(_Client())

        result = arm.move_to_joints(JointPositions(_THERE))

        self.assertIs(CameraWorldUse.PLANNED, result.camera_world.use, result.camera_world.render())
        refreshed = arm._last_world_refresh
        assert refreshed is not None and refreshed.keep_out is not None
        # An empty bench: the region was in force and took nothing out, and the stamp says so rather than nothing.
        self.assertEqual(0, refreshed.keep_out.goal_points)
        self.assertEqual(refreshed.keep_out, result.camera_world.keep_out)


if __name__ == "__main__":
    unittest.main()
