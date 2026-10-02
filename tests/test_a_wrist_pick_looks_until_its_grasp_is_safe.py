"""A wrist pick looks from its looks, fused, until its grasp is safe; a fixed camera picks as it did (plan steps 4a-4c).

The owner's decisions of 2026-09-28 and 2026-09-29: a pick on the wrist D415 looks from the looks its program declares,
in order, and every look is fused with the ones before it. The looks exist to find a safe grasp point, never to scan the
part whole: the looking stops at the first look whose grasp is valid and carries none of the rescan reasons, and an
uncertain grasp or no candidate at all goes on to the next look. Both jaw contact faces seen is a switch
(``both_faces``), off by default, and with it on a grasp no view showed both faces of is never gripped. A look a guard or
the planner refused before anything was sent is skipped and said; a pick that reaches none ends with nothing else
commanded, as a refused look always ended it, and so does a look motion that may have moved the arm. Once every
declared look was visited and the grasp is still not safe enough, one view is generated, aimed at the contact face of
the chosen grasp that no view showed, on the straight joint line only; the arm moves back to the look that saw the part
on that line, or approaches from where it stands; neither runs unless the planner world holds the frames of the pick.
The frames of the pick are held in the planner world until it ends, and the world is offered the fused target. A fixed
camera asked for both faces grips only a grasp its cameras showed both faces of. A look around reaches a later run only
when it is handed on. And the looks
add what one view cannot: the support found from every view, the neighbours every look saw, the labels the looks agree
on, grazing points thinned, and a check that the looks still agree on where the part is, the hand-eye.

The scene is ``tests/_wrist_views.py``: a 40 mm cube on the bench, ray-cast through the wrist D415 from the tool pose each
look puts the arm at, and stamped with it. The pick loop is the real one; the calculator grasps the cube on the frames a
test names, straight down and closing along base x, so jaw 1 closes on the cube's -x face and jaw 2 on its +x face; the
policy only records what it was asked to execute.
"""

from __future__ import annotations

import logging
import math
import time
import unittest
from typing import Any
from unittest import mock

import numpy as np

from src.config.schema.robot.grasping_schema import FusionGeometryConfig, GraspingSupportConfig
from src.contracts import UNSET
from src.geometry import Frame, Pose, Transform
from src.robot.core import (
    NO_PLAN_FAIL_SAFE_MESSAGE,
    JointPositions,
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotMode,
    RobotStatus,
    SafetyMode,
)
from src.robot.core.errors import CameraWorldUnavailable, RobotConnectionError, RobotKinematicsError
from src.robot.execution import motion
from src.robot.grasping.collision.gripper_model import ParallelJawGripperModel
from src.robot.grasping.generation.depth_steps import pixels_behind_depth_steps
from src.robot.grasping.loop.pick_loop import (
    BinPickingOrchestrator,
    LookedAround,
    PickOutcome,
    _LookView,
    segmentation_offer_from_frame,
)
from src.robot.grasping.loop.progress import PickProgress, PickStage
from src.robot.grasping.motion.execution_policy import PolicyOutcome, PolicyReport
from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver, StaticCameraToBaseResolver
from src.robot.grasping.multiview import scene_geometry
from src.robot.grasping.multiview.association import HAND_EYE_DRIFT_WARN_MM, nearest_surface_distances_mm
from src.robot.grasping.multiview.scene_geometry import grazing_pixels, to_base_mm
from src.robot.grasping.telemetry.latency_tracker import LatencyStage, LatencyTracker
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.perception import CameraObservation, PerceptionFrame
from tests._wrist_views import (
    CUBE,
    CUBE_CENTRE,
    K,
    Box,
    LookCalculator,
    LookingArm,
    WristCamera,
    base_points,
    camera_looking_at,
    camera_to_tool,
    joints_key,
    render,
    tool_for,
)

#: The looks a program declares: the cube from its +x side, its -x side, its +y side, 45 degrees up.
LOOK_PLUS_X = JointPositions.deg(0.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LOOK_MINUS_X = JointPositions.deg(180.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LOOK_PLUS_Y = JointPositions.deg(90.0, -90.0, -110.0, -60.0, 90.0, 0.0)
#: A look from the +x side at a spot of bench 400 mm along +y of the cube, wide of the cube in the image.
LOOK_ELSEWHERE = JointPositions.deg(20.0, -90.0, -110.0, -60.0, 90.0, 0.0)
#: The cube seen from its +x side 14 degrees up and 250 mm off: its top at 76 degrees of incidence, past the grazing
#: limit and short of what the depth-step rule takes out at that range, its +x face almost square.
LOOK_LOW = JointPositions.deg(5.0, -90.0, -110.0, -60.0, 90.0, 0.0)
#: The look from elsewhere with the base turned 130 and 170 degrees round: the same view of the bench, a long way back.
LOOK_ELSEWHERE_WIDE = JointPositions.deg(130.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LOOK_ELSEWHERE_FAR = JointPositions.deg(170.0, -90.0, -110.0, -60.0, 90.0, 0.0)
#: The Hand-E's contact patch (config/grippers/robotiq_hande.yaml), the owner's hand.
HAND_E = ParallelJawGripperModel(finger_width_mm=29.24, pad_length_mm=20.91, pad_ahead_mm=10.45)


def _label(joints: JointPositions) -> str:
    return "(" + ", ".join(f"{v:.1f}" for v in joints.degrees()) + ") deg"


def _poses() -> dict[JointPositions, Any]:
    elsewhere = tool_for(camera_looking_at((0.0, -300.0, 0.0), bearing_deg=0.0))
    return {
        LOOK_PLUS_X: tool_for(camera_looking_at(bearing_deg=0.0)),
        LOOK_MINUS_X: tool_for(camera_looking_at(bearing_deg=180.0)),
        LOOK_PLUS_Y: tool_for(camera_looking_at(bearing_deg=90.0)),
        LOOK_ELSEWHERE: elsewhere,
        LOOK_LOW: tool_for(camera_looking_at(bearing_deg=0.0, elevation_deg=14.0, range_mm=250.0)),
        LOOK_ELSEWHERE_WIDE: elsewhere,
        LOOK_ELSEWHERE_FAR: elsewhere,
    }


class _Policy:
    """Executes nothing and says it executed: what the loop asked of the arm is what these tests read.

    ``joints_at`` is where the arm stood as each grasp was handed over, the start of its approach, and ``held_at``
    whether the arm's planner world held the frames of the pick then (``None`` for an arm with no world)."""

    def __init__(self, arm: Any = None) -> None:
        self.arm = arm
        self.executed: list[Any] = []
        self.joints_at: list[JointPositions] = []
        self.held_at: list[bool | None] = []

    def execute(self, grasp: Any) -> PolicyReport:
        self.executed.append(grasp)
        if self.arm is not None:
            self.joints_at.append(self.arm.get_joint_positions())
            world = getattr(self.arm, "live_planner_world", None)
            self.held_at.append(world.holding if world is not None else None)
        return PolicyReport(outcome=PolicyOutcome.EXECUTED)


class _Cell:
    """A wrist cell on the looking arm, its camera, the calculator and the recording policy."""

    def __init__(self, *, looks: tuple[JointPositions, ...], grasps_on: tuple[int, ...] | None = None,
                 boxes: tuple[Box, ...] = (CUBE,), refuse: tuple[JointPositions, ...] = (),
                 unvouched: tuple[JointPositions, ...] = (), arm: LookingArm | None = None,
                 camera_matrix: np.ndarray | None = None, uncertain_on: tuple[int, ...] = (),
                 calculator: Any = None, **wiring: Any) -> None:
        self.arm = arm or LookingArm(_poses(), refuse=refuse, unvouched=unvouched)
        self.camera = WristCamera(self.arm, boxes=boxes, labels=wiring.pop("labels", {}),
                                  stamp_error_mm=wiring.pop("stamp_error_mm", {}))
        self.calculator = (calculator(self.camera) if calculator is not None else LookCalculator(
            self.camera, grasps_on=grasps_on, uncertain_on=uncertain_on, camera_matrix=camera_matrix))
        self.policy = _Policy(self.arm)
        self.events: list[PickProgress] = []
        self.orchestrator = BinPickingOrchestrator(
            arm=self.arm, calculator=self.calculator, perception=self.camera,  # type: ignore[arg-type]
            frame_resolver=wiring.pop("frame_resolver", EyeInHandFrameResolver(t_cam_to_tool=camera_to_tool())),
            policy=self.policy,  # type: ignore[arg-type]
            max_attempts=wiring.pop("max_attempts", 3), primary_camera_id="wrist", gripper_model=HAND_E,
            looks=looks, on_progress=self.events.append, **wiring,
        )

    def run(self) -> Any:
        return self.orchestrator.run()

    @property
    def looked(self) -> Any:
        return self.orchestrator.looked_around

    def joints_moved_to(self) -> list[tuple[float, ...]]:
        return [key for kind, key in self.arm.motions if kind == "joints"]


def _keys(*looks: JointPositions) -> list[tuple[float, ...]]:
    return [tuple(round(v, 1) for v in look.degrees()) for look in looks]


def _on_the_plus_x_face(points: np.ndarray) -> int:
    """How many of ``points`` lie on the cube's +x face, below the crease with its top."""
    return int(np.sum((points[:, 0] > 19.9) & (points[:, 2] < 39.0)))


class _FailingLookArm(LookingArm):
    """The looking arm, whose motion to one look ends as a real drive's can once it was asked for: ``answer`` is the
    status and message of the result it returns, or the exception it raises."""

    def __init__(self, *, fail_at: JointPositions, answer: Any) -> None:
        super().__init__(_poses())
        self.fail_key = joints_key(fail_at)
        self.answer = answer

    def move_to_joints(self, joints: JointPositions, **keywords: Any) -> MotionResult:
        if joints_key(joints) != self.fail_key:
            return super().move_to_joints(joints, **keywords)
        self.motions.append(("joints", self.fail_key))
        if isinstance(self.answer, BaseException):
            raise self.answer
        status, message = self.answer
        return MotionResult.failed(status, MotionCommand.MOVE_JOINTS, target_joints=joints, message=message)


def _top_camera() -> dict[str, Any]:
    """A fixed camera over the bench, 80 degrees up, as the wiring of a wrist cell that fuses it: its rig lists the
    wrist camera too, whose copy of the first look is not a view of its own."""
    above = camera_looking_at(bearing_deg=-90.0, elevation_deg=80.0)
    depth, hit = render(above, (CUBE,))
    seen = PerceptionFrame(depth_map=depth, intrinsics=K.copy(),
                           segmentations=(type("S", (), {"mask": hit == 0, "label": "part"})(),))
    placed = StaticCameraToBaseResolver(transform=Transform.from_matrix(
        above, from_frame=Frame.CAMERA, to_frame=Frame.BASE))
    rig = _Rig(CameraObservation(camera_id="top", frame=seen), CameraObservation(camera_id="wrist", frame=seen))
    return {"multi_camera_perception": rig, "camera_frame_resolvers": {"top": placed, "wrist": placed},
            "configured_camera_ids": ("top",), "fusion_geometry_config": FusionGeometryConfig(enabled=True)}


class _ByPart(LookCalculator):
    """Grasps each part at its own centre, straight down and closing along base x, on the frames ``table`` names:
    ``(frame number, label) -> reasons``, ``()`` for a grasp the calculator is sure of. Anything else gets no
    candidate with a rescan reason."""

    def __init__(self, camera: WristCamera, *, table: dict[tuple[int, str], tuple[GraspFailureReason, ...]],
                 centres: dict[str, tuple[float, float, float]]) -> None:
        super().__init__(camera)
        self.table = table
        self.centres = centres

    def compute_result(self, seg: Any, *_args: Any, **kwargs: Any) -> GraspResult:
        number = len(self.camera.taken) - 1
        label = str(getattr(seg, "label", ""))
        self.calls.append((number, label, dict(kwargs)))
        if (number, label) not in self.table:
            return GraspResult(reasons=(GraspFailureReason.NO_CANDIDATES_GENERATED,
                                        GraspFailureReason.RESCAN_RECOMMENDED))
        grasp = GraspPoint(position=np.array(self.centres[label]), approach=np.array([0.0, 0.0, -1.0]),
                           axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=40.0, score=0.9, frame=GraspFrame.BASE,
                           label=label)
        return GraspResult(candidates=(grasp,), top_score=0.9, reasons=self.table[(number, label)])


class _World:
    """A live planner world as the pick loop meets it: it holds the frames of a pick, and is offered its target.

    ``holding`` is whether a pick holds its frames now; ``calls`` every hold and let-go in order; ``offers`` every
    target the pick offered it. ``holds`` false is a world that cannot hold a pick's frames (no camera on the wrist)."""

    def __init__(self, *, holds: bool = True) -> None:
        self.holds = holds
        self.holding = False
        self.calls: list[str] = []
        self.offers: list[dict[str, Any]] = []

    def hold_pick_views(self) -> bool:
        self.holding = self.holds
        self.calls.append("hold")
        return self.holds

    def forget_pick_views(self) -> None:
        self.holding = False
        self.calls.append("forget")

    def offer_segmentation(self, **offered: Any) -> None:
        self.offers.append(offered)

    def forget_segmentation(self) -> None:
        pass


def _turned(pose: Pose) -> float:
    """How far round the cube's vertical a tool pose stands from the +x look's, degrees in (-180, 180]: what the base
    joint of an arm standing over the cube turns by, 0 at the +x look, 90 at the +y look, 180 at the -x look."""
    def bearing(of: Pose) -> float:
        x, y, _ = (float(v) for v in of.position_mm)
        return math.degrees(math.atan2(y - CUBE_CENTRE[1], x - CUBE_CENTRE[0]))

    turned = (bearing(pose) - bearing(_poses()[LOOK_PLUS_X])) % 360.0
    return round(turned - 360.0 if turned > 180.0 else turned, 3)


class _LineArm(LookingArm):
    """The looking arm, which also drives the straight joint line to a configuration, and nothing else, when asked to.

    Every such line is written down, from where the arm stood to where it went, in degrees (``lines``). ``blocked``
    holds targets whose line the planner world finds not clear, and ``answers`` a result a drive gave once it was
    asked. The planner fails the test when asked: ``move_to_joints`` is the verb that plans around a line, and it may
    run only to a declared look, once each, never to a generated view and never back to a look. Its live planner world
    is ``world``, a fresh :class:`_World` that holds the frames of a pick by default; ``bare`` is an arm with none."""

    def __init__(self, test: unittest.TestCase, *, world: _World | None = None, bare: bool = False,
                 **keywords: Any) -> None:
        super().__init__(_poses(), **keywords)
        self.test = test
        self.declared = set(self.table)
        self.asked: set[tuple[float, ...]] = set()
        self.home_joint_positions = tuple(LOOK_PLUS_X.tolist())
        self.lines: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
        self.blocked: set[tuple[float, ...]] = set()
        self.answers: dict[tuple[float, ...], tuple[MotionStatus, str]] = {}
        if not bare:
            self.live_planner_world = world if world is not None else _World()

    def move_to_joints_on_the_line(self, joints: JointPositions) -> MotionResult:
        key = joints_key(joints)
        self.lines.append((joints_key(self._joints), key))
        self.motions.append(("line", key))
        if key in self.answers:
            status, message = self.answers[key]
            return MotionResult.failed(status, MotionCommand.MOVE_JOINTS, target_joints=joints, message=message)
        if key in self.blocked:
            return MotionResult.failed(
                MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_JOINTS, target_joints=joints,
                message="the planner refused this straight joint line: it passes 4 mm from a held frame's box; nothing "
                        "is planned around it")
        self._joints = joints
        self._tcp = self.table[key]
        return MotionResult.executed(MotionCommand.MOVE_JOINTS, target_joints=joints,
                                     message="move_to_joints_on_the_line")

    def move_to_joints(self, joints: JointPositions, **keywords: Any) -> MotionResult:
        key = joints_key(joints)
        if key not in self.declared or key in self.asked:
            self.test.fail(f"the planner was asked for {key}: the generated view and the move back never plan")
        self.asked.add(key)
        return super().move_to_joints(joints, **keywords)


class _OrbitingArm(_LineArm):
    """The line arm, which also names the configuration of a turn of a look about the part (``nearest_configuration``):
    the base joint turned as far round the cube as the tool stands from the +x look, the rest of the +x look. ``screened``
    is every turn it was asked about, ``unreachable`` the turns it refuses as out of reach."""

    def __init__(self, test: unittest.TestCase, **keywords: Any) -> None:
        super().__init__(test, **keywords)
        self.screened: list[float] = []
        self.unreachable: set[float] = set()

    def nearest_configuration(self, pose: Pose) -> JointPositions:
        turned = _turned(pose)
        self.screened.append(turned)
        if turned in self.unreachable:
            raise RobotKinematicsError(f"turn {turned:+g}: refused by the inverse kinematics: out of reach")
        joints = JointPositions.deg(turned, *LOOK_PLUS_X.degrees()[1:])
        self.table[joints_key(joints)] = pose
        return joints

    def turned_to(self, degrees: float) -> tuple[float, ...]:
        """The key of the configuration this arm names for a turn of ``degrees`` from the +x look."""
        return joints_key(JointPositions.deg(degrees, *LOOK_PLUS_X.degrees()[1:]))


# ---------------------------------------------------------------------------------------------------------------------
# The early stop: a valid grasp without a rescan reason ends the looking; anything less goes on
# ---------------------------------------------------------------------------------------------------------------------


class TheLookingStopsAtTheFirstSafeGraspTests(unittest.TestCase):
    def test_a_valid_grasp_at_the_first_look_ends_the_looking_there(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X))

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(_keys(LOOK_PLUS_X), cell.joints_moved_to(), "the arm was sent on after a safe grasp")
        self.assertEqual(1, len(cell.camera.taken))
        self.assertEqual((_label(LOOK_PLUS_X),), cell.looked.visited)
        self.assertEqual(1, len(cell.policy.executed))
        self.assertTrue(cell.looked.judged.good)

    def test_no_candidate_at_the_first_look_goes_on_to_the_next_and_fuses_them(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), grasps_on=(1,))

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(_keys(LOOK_PLUS_X, LOOK_MINUS_X), cell.joints_moved_to())
        self.assertEqual((_label(LOOK_MINUS_X), _label(LOOK_PLUS_X)), cell.looked.looks_fused,
                         "the look the grasp was ranked on comes first")
        # What the generator was handed at the second look holds the face only the first look saw.
        (at_second,) = cell.calculator.calls_on(1)
        cloud = base_points(at_second, "geometry_points_base_mm")
        self.assertGreater(int(np.sum(cloud[:, 0] > 19.0)), 500, "the +x face the first look saw is missing")
        self.assertGreater(int(np.sum(cloud[:, 0] < -19.0)), 500, "the second look's own -x face is missing")
        self.assertEqual(1, len(report.attempts), "a wrist pick never rescans where it stands")
        self.assertEqual(
            (f"wrist@{_label(LOOK_MINUS_X)}", f"wrist@{_label(LOOK_PLUS_X)}"), report.attempts[0].fused_views)

    def test_an_uncertain_grasp_goes_on_to_the_next_look(self) -> None:
        """The second look calls the part a bolt, the first a part: uncertain, so the third look is visited."""
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X, LOOK_PLUS_Y), grasps_on=(1, 2), labels={1: {"part": "bolt"}})

        with self.assertLogs("src.robot.grasping.loop.pick_loop", level="WARNING") as said:
            report = cell.run()

        self.assertEqual(_keys(LOOK_PLUS_X, LOOK_MINUS_X, LOOK_PLUS_Y), cell.joints_moved_to())
        self.assertIn(GraspFailureReason.RESCAN_RECOMMENDED, cell.looked.judged.reasons)
        self.assertTrue(any("calls its target" in line for line in said.output), said.output)
        # Every look was visited, so the pick goes on with the views fused so far.
        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual((), report.attempts[0].reasons, "the executed attempt's reasons are the calculator's")

    def test_the_looks_that_agree_on_the_label_count_as_surer(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X, LOOK_PLUS_Y), grasps_on=(1, 2))

        cell.run()

        self.assertEqual(_keys(LOOK_PLUS_X, LOOK_MINUS_X), cell.joints_moved_to())
        self.assertEqual(1, cell.looked.judged.labels_agreed)
        self.assertNotIn(GraspFailureReason.RESCAN_RECOMMENDED, cell.looked.judged.reasons)

    def test_no_look_with_a_candidate_ends_the_attempt_without_a_grasp(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), grasps_on=())

        with self.assertLogs("src.robot.grasping.loop.pick_loop", level="INFO") as said:
            report = cell.run()

        self.assertIs(PickOutcome.RESCANNED_EXHAUSTED, report.outcome)
        self.assertEqual(["exhausted"], [attempt.action for attempt in report.attempts])
        self.assertEqual(2, len(cell.camera.taken), "a frame was taken somewhere other than at a look")
        self.assertEqual([], cell.policy.executed)
        self.assertTrue(any("no look ranked a grasp" in line and "the attempt ends without one" in line
                            for line in said.output), said.output)
        self.assertFalse(any("goes on with" in line for line in said.output), said.output)

    def test_the_looks_reached_are_counted_in_what_the_pick_says(self) -> None:
        """A look the arm was refused was not visited: the account says how many of the looks were reached."""
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_PLUS_Y, LOOK_MINUS_X), grasps_on=(), uncertain_on=(0, 1),
                     refuse=(LOOK_PLUS_Y,))

        with self.assertLogs("src.robot.grasping.loop.pick_loop", level="INFO") as said:
            cell.run()

        self.assertTrue(any("the looks reached (2 of 3)" in line for line in said.output), said.output)
        self.assertFalse(any("every look was visited" in line for line in said.output), said.output)

    def test_every_look_is_on_the_wire_and_the_ranking_names_the_looks(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), grasps_on=(1,))

        cell.run()

        perceived = [event.extra.get("look") for event in cell.events if event.stage is PickStage.PERCEIVED]
        self.assertEqual([_label(LOOK_PLUS_X), _label(LOOK_MINUS_X)], perceived)
        (ranked,) = [event for event in cell.events if event.stage is PickStage.RANKED]
        self.assertEqual([_label(LOOK_PLUS_X), _label(LOOK_MINUS_X)], ranked.extra["looks"])
        self.assertEqual([_label(LOOK_MINUS_X), _label(LOOK_PLUS_X)], ranked.extra["looks_fused"])
        self.assertEqual([True, True], ranked.extra["jaw_faces_seen"])

    def test_every_look_keeps_what_a_record_of_the_pick_needs(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), grasps_on=(1,))

        cell.run()

        views = cell.looked.views
        self.assertEqual(2, len(views))
        for view, taken in zip(views, cell.camera.taken, strict=True):
            self.assertIsNotNone(view.frame.rgb)
            self.assertEqual(K.tolist(), np.asarray(view.frame.intrinsics).tolist())
            np.testing.assert_allclose(taken.to_matrix(), view.frame.tool_pose.to_matrix())
            self.assertEqual(np.asarray(view.frame.depth_map).shape, (360, 640))
        self.assertGreater(len(cell.looked.judged.target_cloud_base_mm), 1000)


# ---------------------------------------------------------------------------------------------------------------------
# Both jaw contact faces: a switch, off by default
# ---------------------------------------------------------------------------------------------------------------------


class BothJawFacesTests(unittest.TestCase):
    def test_by_default_one_face_seen_is_enough_to_stop(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X))

        cell.run()

        self.assertEqual(_keys(LOOK_PLUS_X), cell.joints_moved_to())
        self.assertEqual((False, True), cell.looked.jaw_faces_seen, "from +x only jaw 2's face is seen")

    def test_with_both_faces_asked_for_the_pick_looks_until_both_were_seen(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X, LOOK_PLUS_Y), both_faces=True)

        with self.assertLogs("src.robot.grasping.loop.pick_loop", level="INFO") as said:
            report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(_keys(LOOK_PLUS_X, LOOK_MINUS_X), cell.joints_moved_to())
        self.assertEqual((True, True), cell.looked.jaw_faces_seen)
        self.assertTrue(
            any("the contact face at jaw 1 of the chosen grasp was not seen" in line for line in said.output),
            said.output)


# ---------------------------------------------------------------------------------------------------------------------
# A look the arm is refused, a look that misses the part, a stop on the way
# ---------------------------------------------------------------------------------------------------------------------


class ARefusedLookTests(unittest.TestCase):
    def test_a_look_refused_after_a_valid_grasp_is_skipped_and_said(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_PLUS_Y, LOOK_MINUS_X), both_faces=True, refuse=(LOOK_PLUS_Y,))

        with self.assertLogs("src.robot.grasping.loop.pick_loop", level="WARNING") as said:
            report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual((_label(LOOK_PLUS_X), _label(LOOK_MINUS_X)), cell.looked.visited)
        self.assertEqual(_label(LOOK_PLUS_Y), cell.looked.refused[0][0])
        warning = next(line for line in said.output if "was refused" in line)
        self.assertIn(_label(LOOK_PLUS_Y), warning)
        self.assertIn("outside the cable window", warning)
        self.assertIn("with the views fused so far", warning)

    def test_a_look_refused_before_any_grasp_goes_on_to_the_next(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_Y, LOOK_MINUS_X), refuse=(LOOK_PLUS_Y,))

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual((_label(LOOK_MINUS_X),), cell.looked.visited)
        self.assertEqual(_keys(LOOK_PLUS_Y, LOOK_MINUS_X), cell.joints_moved_to())

    def test_a_pick_that_reaches_no_look_ends_with_nothing_else_commanded(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), refuse=(LOOK_PLUS_X, LOOK_MINUS_X))

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTION_FAILED, report.outcome)
        (attempt,) = report.attempts
        self.assertEqual("look_refused", attempt.action)
        self.assertEqual("workspace_rejected", attempt.motion_status)
        self.assertIn("cable window", attempt.motion_message)
        self.assertEqual([], cell.camera.taken, "a frame was taken although no look was reached")
        self.assertEqual([], cell.policy.executed)
        self.assertEqual(_keys(LOOK_PLUS_X, LOOK_MINUS_X), cell.joints_moved_to())

    def test_a_stopped_controller_ends_the_looking_where_it_is(self) -> None:
        class _StoppedArm(LookingArm):
            def get_robot_status(self) -> RobotStatus:
                return RobotStatus(RobotMode.RUNNING, SafetyMode.PROTECTIVE_STOP, protective_stopped=True,
                                   emergency_stopped=False)

            def recover_from_protective_stop(self) -> bool:
                return False

        arm = _StoppedArm(_poses(), refuse=(LOOK_PLUS_X,))
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), arm=arm)

        report = cell.run()

        self.assertIs(PickOutcome.CONTROLLER_NOT_OPERATIONAL, report.outcome)
        self.assertEqual((GraspFailureReason.CONTROLLER_NOT_OPERATIONAL,), report.attempts[0].reasons)
        self.assertEqual(_keys(LOOK_PLUS_X), cell.joints_moved_to(), "the arm was sent on after the stop")
        self.assertEqual([], cell.camera.taken)

    def test_a_camera_that_cannot_vouch_on_the_way_is_a_fault(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), unvouched=(LOOK_PLUS_X,))

        with self.assertRaises(CameraWorldUnavailable):
            cell.run()
        self.assertEqual([], cell.camera.taken)

    def test_a_stop_asked_for_between_looks_sends_the_arm_nowhere_else(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), grasps_on=())
        cell.orchestrator.should_cancel = lambda: len(cell.camera.taken) > 0

        report = cell.run()

        self.assertIs(PickOutcome.CANCELLED, report.outcome)
        self.assertEqual(_keys(LOOK_PLUS_X), cell.joints_moved_to())

    def test_a_motion_refused_before_any_command_is_not_said_to_have_moved(self) -> None:
        """A motion the verb refused before commanding anything (a camera world the arm would refuse, a link not open)
        still ends the attempt, and the log says what happened: it was refused, and nothing was sent."""
        cell = _Cell(looks=(LOOK_PLUS_X,))
        stopped = motion.MotionReport(
            motion.MotionVerb.MOVE_JOINTS, motion.MotionOutcome.REFUSED,
            motion.RouteReading(motion.MotionRoute.PLANNED, "test"),
            MotionResult.failed(MotionStatus.UNSUPPORTED, MotionCommand.MOVE_JOINTS,
                                message="Refused before planning: no live camera world is wired. Nothing moved."))
        looked = LookedAround(stopped=stopped, stopped_look="orbit +140 deg (140.0, -90.0, -110.0, -60.0, 90.0, 0.0) deg")

        with self.assertLogs("src.robot.grasping.loop.pick_loop", level="ERROR") as said:
            report = cell.orchestrator._look_refused(0, looked, [])

        self.assertIs(PickOutcome.EXECUTION_FAILED, report.outcome)
        self.assertEqual("look_refused", report.attempts[0].action)
        (line,) = said.output
        self.assertIn("refused before any command", line)
        self.assertNotIn("may have moved", line)


class ALookMotionThatMayHaveMovedTests(unittest.TestCase):
    """Only a look a guard or the planner refused before anything was sent is skipped (the owner, 2026-09-29). A look
    motion that failed once it may have been commanded ends the attempt there, as every failed motion of a pick does:
    the arm may have moved part of the way, or may still be moving."""

    #: What a drive answers once moveJ may have been sent, each in the UR driver's own words.
    MAY_HAVE_MOVED: dict[str, Any] = {
        "controller_rejected": (MotionStatus.CONTROLLER_REJECTED,
                                "the drive refused after the move was judged, and moveJ may have been sent"),
        "timeout": (MotionStatus.TIMEOUT, "moveJ did not complete within its budget"),
        "connection_error": (MotionStatus.CONNECTION_ERROR,
                             "UR moveJ raised executing waypoint 2 of 5 of the judged path. moveJ was sent"),
        "unknown": (MotionStatus.UNKNOWN, "the driver could not classify the failure"),
        "raised": RobotConnectionError("the RTDE link dropped while moveJ was running"),
    }
    #: What a guard or the planner answers before anything is sent.
    REFUSED_BEFORE_SENDING: dict[str, Any] = {
        "no plan": (MotionStatus.TIMEOUT, NO_PLAN_FAIL_SAFE_MESSAGE),
        "ik": (MotionStatus.IK_FAILED, "no configuration reaches it"),
        "line": (MotionStatus.SELF_COLLISION_REJECTED, "the straight line to it collides"),
        "window": (MotionStatus.JOINT_LIMIT_REJECTED, "outside the cable window"),
    }

    def test_a_look_motion_that_may_have_moved_ends_the_attempt_with_nothing_else_commanded(self) -> None:
        for name, answer in self.MAY_HAVE_MOVED.items():
            with self.subTest(name):
                # The first look's grasp is valid and uncertain, so the looking goes on; the third look's is safe.
                cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X, LOOK_PLUS_Y), grasps_on=(1,), uncertain_on=(0,),
                             arm=_FailingLookArm(fail_at=LOOK_MINUS_X, answer=answer))

                report = cell.run()

                self.assertIs(PickOutcome.EXECUTION_FAILED, report.outcome)
                (attempt,) = report.attempts
                self.assertEqual("look_refused", attempt.action)
                self.assertEqual(_keys(LOOK_PLUS_X, LOOK_MINUS_X), cell.joints_moved_to(), "the arm was sent on")
                self.assertEqual([], cell.policy.executed, "the approach ran after a look motion that may have moved")
                self.assertEqual(1, len(cell.camera.taken))
                self.assertEqual((), cell.looked.refused, "a motion that may have moved was taken for a refusal")
                self.assertEqual(_label(LOOK_MINUS_X), cell.looked.stopped_look)

    def test_a_look_refused_before_anything_was_sent_is_skipped(self) -> None:
        for name, answer in self.REFUSED_BEFORE_SENDING.items():
            with self.subTest(name):
                cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X, LOOK_PLUS_Y), grasps_on=(1,), uncertain_on=(0,),
                             arm=_FailingLookArm(fail_at=LOOK_MINUS_X, answer=answer))

                report = cell.run()

                self.assertIs(PickOutcome.EXECUTED, report.outcome)
                self.assertEqual(_keys(LOOK_PLUS_X, LOOK_MINUS_X, LOOK_PLUS_Y), cell.joints_moved_to())
                self.assertEqual(_label(LOOK_MINUS_X), cell.looked.refused[0][0])
                self.assertEqual((_label(LOOK_PLUS_X), _label(LOOK_PLUS_Y)), cell.looked.visited)


class TheLooksAreHandedOnOnlyOnPurposeTests(unittest.TestCase):
    """A look around is a pick's own. Its grasp reaches the next run only when the caller hands it on
    (``go_on_with``), and a fixed camera is never sent round the looks."""

    def test_a_look_around_nobody_handed_on_is_never_picked(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X,))
        cell.orchestrator.look_around()
        # The part is taken away, and the next pick is handed no looks.
        cell.camera.boxes = ()
        cell.orchestrator.looks = ()

        report = cell.run()

        self.assertEqual(2, len(cell.camera.taken), "the later pick did not perceive for itself")
        self.assertIs(PickOutcome.NO_PERCEPTION, report.outcome)
        self.assertEqual([], cell.policy.executed, "the grasp of an earlier look around was picked")

    def test_a_look_around_handed_on_is_picked_without_looking_again(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X))
        looked = cell.orchestrator.look_around()
        cell.orchestrator.go_on_with(looked)

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(1, len(cell.camera.taken))
        self.assertEqual(_keys(LOOK_PLUS_X), cell.joints_moved_to())
        self.assertIs(looked, cell.looked)

    def test_only_the_looks_still_open_are_handed_on(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X,))
        looked = cell.orchestrator.look_around()

        with self.assertRaises(ValueError):
            cell.orchestrator.go_on_with(LookedAround())
        cell.orchestrator.close_looks()
        with self.assertRaises(ValueError):
            cell.orchestrator.go_on_with(looked)

    def test_a_fixed_camera_is_never_sent_round_its_looks(self) -> None:
        fixed = StaticCameraToBaseResolver(transform=Transform.from_matrix(
            camera_looking_at(bearing_deg=0.0), from_frame=Frame.CAMERA, to_frame=Frame.BASE))
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), frame_resolver=fixed)

        with self.assertRaises(ValueError):
            cell.orchestrator.look_around()
        self.assertEqual([], cell.arm.motions)
        self.assertEqual([], cell.camera.taken)


class ALookThatMissesThePartTests(unittest.TestCase):
    #: A part like the cube 400 mm along +y of it, which the look from elsewhere sees and the first look does not.
    OTHER = Box((-20.0, -320.0, 0.0), (20.0, -280.0, 30.0), "part")

    def test_a_look_that_does_not_see_the_part_is_left_out_and_said(self) -> None:
        """The grasp of the first look is valid and uncertain, so the pick looks on, and the second look's camera does
        not see the part: it is left out, the grasp stands on the first look, and the arm moves back there, once, on the
        straight joint line, before the approach runs from the look that saw the part."""
        arm = _LineArm(self)
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_ELSEWHERE), boxes=(CUBE, self.OTHER), grasps_on=(), uncertain_on=(0,),
                     arm=arm)

        with self.assertLogs("src.robot.grasping.loop.pick_loop", level="INFO") as said:
            report = cell.run()

        self.assertEqual((_label(LOOK_ELSEWHERE),), cell.looked.dropped)
        self.assertEqual(_label(LOOK_PLUS_X), cell.looked.judged.look.label, "the first look's grasp stands")
        self.assertEqual((_label(LOOK_PLUS_X),), cell.looked.looks_fused)
        self.assertTrue(any("did not see the part" in line for line in said.output), said.output)
        self.assertEqual(_keys(LOOK_PLUS_X, LOOK_ELSEWHERE), cell.joints_moved_to())
        self.assertEqual([(_keys(LOOK_ELSEWHERE)[0], _keys(LOOK_PLUS_X)[0])], arm.lines, "not exactly one move back")
        self.assertEqual(_label(LOOK_PLUS_X), cell.looked.moved_back)
        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        (executed,) = cell.policy.executed
        np.testing.assert_allclose(executed.position, CUBE_CENTRE)
        self.assertEqual(_keys(LOOK_PLUS_X)[0], joints_key(cell.policy.joints_at[0]),
                         "the approach did not start from the look that saw the part")

    def test_the_grasp_of_an_earlier_look_is_gone_on_with_as_that_look_saw_it(self) -> None:
        """Every look ranked after the judged one overwrote what the rest of the attempt reads: the CAMERA to BASE the
        planner world places the target by, and the objects the target's centre is read from. Going on with the judged
        look's grasp puts back the state its own ranking left."""
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_ELSEWHERE), boxes=(CUBE, self.OTHER), both_faces=True)
        orchestrator = cell.orchestrator
        looked = orchestrator.look_around()
        first, second = looked.views
        np.testing.assert_allclose(second.camera_to_base, orchestrator._pending_camera_to_base.to_matrix())

        frame, result, index, _ = orchestrator._adopt(looked.judged)

        self.assertIs(first.frame, frame)
        self.assertIs(looked.judged.result, result)
        np.testing.assert_allclose(first.camera_to_base, orchestrator._pending_camera_to_base.to_matrix())
        centre = orchestrator._target_centre_base_mm(frame, index)
        self.assertIsNotNone(centre)
        self.assertLess(abs(centre[1] + 700.0), 15.0, f"the target's centre is read off another part: {centre}")
        self.assertEqual(looked.judged.fused_views_by_object, orchestrator._fused_views_by_object)


# ---------------------------------------------------------------------------------------------------------------------
# A fixed camera picks as it did, and a wrist pick never rescans where it stands
# ---------------------------------------------------------------------------------------------------------------------


class AFixedCameraPicksAsItDidTests(unittest.TestCase):
    def test_a_fixed_camera_ignores_the_looks_and_rescans_where_it_stands(self) -> None:
        fixed = StaticCameraToBaseResolver(transform=Transform.from_matrix(
            camera_looking_at(bearing_deg=0.0), from_frame=Frame.CAMERA, to_frame=Frame.BASE))
        # The arm stands where the camera would be at the +x look: the camera is fixed there.
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), grasps_on=(1,), frame_resolver=fixed)

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual([], cell.joints_moved_to(), "a fixed camera was moved to look")
        self.assertEqual(["rescan", "executed"], [attempt.action for attempt in report.attempts])
        self.assertIsNone(cell.looked)
        self.assertFalse(any(event.extra for event in cell.events if event.stage is PickStage.PERCEIVED))

    def test_a_wrist_pick_handed_no_look_rescans_where_it_stands_as_it_did(self) -> None:
        """A pick handed no looks has no next look to go on to: a reason worth another look gets a fresh frame where
        the arm stands, as a wrist pick always got, and every event is the one it always was."""
        cell = _Cell(looks=(), grasps_on=(1,))

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(["rescan", "executed"], [attempt.action for attempt in report.attempts])
        self.assertEqual(2, len(cell.camera.taken))
        self.assertEqual([], cell.joints_moved_to())
        self.assertEqual(("here",), cell.looked.visited)
        perceived = [event for event in cell.events if event.stage is PickStage.PERCEIVED]
        self.assertEqual([0, 1], [event.attempt for event in perceived])
        said = [event for event in cell.events
                if event.stage in (PickStage.PERCEIVED, PickStage.NO_CANDIDATE, PickStage.RANKED)]
        self.assertEqual(4, len(said))
        self.assertFalse(any(event.extra for event in said), [event.extra for event in said])

    def test_a_wrist_pick_handed_no_look_names_its_view_by_the_rig(self) -> None:
        """Only a look is ``rig@look``: the view of a pick handed none is its rig, beside the fixed camera fused."""
        cell = _Cell(looks=(), **_top_camera())

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(("wrist", "top"), report.attempts[0].fused_views)

    def test_the_fusion_clock_of_a_fixed_camera_starts_once_its_frames_are_in_hand(self) -> None:
        """As it always started: placing the other cameras' frames is part of the fusion span, and their capture is
        not."""
        from tests.test_pick_loop_fusion_geometry import _IDENTITY, _enabled, _frame, _orchestrator
        from tests.test_pick_loop_fusion_geometry import _Rig as _FixedRig

        class _SlowPlacement:
            def camera_to_base_for_frame(self, _frame_in: Any, *, arm: Any = None) -> Transform:
                time.sleep(0.2)
                return _IDENTITY

        tracker = LatencyTracker()
        orchestrator = _orchestrator(
            multi_camera_perception=_FixedRig((CameraObservation(camera_id="left", frame=_frame()),)),
            camera_frame_resolvers={"left": _SlowPlacement()}, fusion_geometry_config=_enabled(),
            latency_tracker=tracker)

        orchestrator._fused_scene(_frame(), _IDENTITY)

        # Well under the 200 ms the placement sleeps: a Windows monotonic clock ticks every 15.6 ms.
        self.assertGreaterEqual(tracker.get(LatencyStage.FUSION), 150.0)


def _behind_camera() -> dict[str, Any]:
    """A second fixed camera, behind the part from the first (the cube's -x side, 45 degrees up), fused."""
    behind = camera_looking_at(bearing_deg=180.0)
    depth, hit = render(behind, (CUBE,))
    seen = PerceptionFrame(depth_map=depth, intrinsics=K.copy(),
                           segmentations=(type("S", (), {"mask": hit == 0, "label": "part"})(),))
    placed = StaticCameraToBaseResolver(transform=Transform.from_matrix(
        behind, from_frame=Frame.CAMERA, to_frame=Frame.BASE))
    return {"multi_camera_perception": _Rig(CameraObservation(camera_id="behind", frame=seen)),
            "camera_frame_resolvers": {"behind": placed}, "configured_camera_ids": ("behind",),
            "fusion_geometry_config": FusionGeometryConfig(enabled=True)}


class _Rig:
    """The fixed cameras of a wrist cell, counting how often they are asked."""

    def __init__(self, *observations: CameraObservation) -> None:
        self.observations = observations
        self.calls = 0
        self.last_failures: dict[str, str] = {}

    def acquire_all(self) -> tuple[CameraObservation, ...]:
        self.calls += 1
        return self.observations


class TheFixedCamerasOfAWristCellTests(unittest.TestCase):
    def test_they_are_acquired_once_per_pick_and_fused_into_every_look(self) -> None:
        # The rig lists the wrist camera too; its copy of the first look is not a view of its own.
        wiring = _top_camera()
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), grasps_on=(1,), **wiring)

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(1, wiring["multi_camera_perception"].calls,
                         "the fixed cameras were acquired again for a later look")
        self.assertEqual(
            (f"wrist@{_label(LOOK_MINUS_X)}", f"wrist@{_label(LOOK_PLUS_X)}", "top"), report.attempts[0].fused_views)


# ---------------------------------------------------------------------------------------------------------------------
# What the looks add (the owner, 2026-09-29): use all the data, simply
# ---------------------------------------------------------------------------------------------------------------------


class WhatTheLooksAddTests(unittest.TestCase):
    def test_the_support_is_found_from_the_cloud_of_every_look(self) -> None:
        """A low block hides the foot of the cube's -x face from the second look, which alone cannot find the bench;
        the first look saw the foot of the +x face, and the two fused put the bench where it is."""
        block = Box((-45.0, -730.0, 0.0), (-28.0, -670.0, 30.0), "block")
        support = GraspingSupportConfig(height_mm=-50.0, refine_from_target=True)
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), grasps_on=(1,), boxes=(CUBE, block), support_config=support,
                     camera_matrix=K)

        cell.run()

        telemetry = cell.orchestrator._support_telemetry
        self.assertEqual("observed", telemetry["support_source"], telemetry)
        self.assertLess(abs(telemetry["support_height_mm"]), 2.0, telemetry)
        (at_second,) = cell.calculator.calls_on(1)
        self.assertLess(abs(float(at_second["support_plane"].offset_mm)), 2.0)

    def test_a_part_only_an_earlier_look_saw_stays_an_obstacle(self) -> None:
        """A low block behind the cube, seen from the first look and hidden by the cube from the second."""
        block = Box((25.0, -715.0, 0.0), (40.0, -685.0, 15.0), "block")
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), grasps_on=(1,), boxes=(CUBE, block))

        cell.run()

        self.assertEqual({"part"}, {label for number, label, _ in cell.calculator.calls if number == 1},
                         "the second look was meant not to see the block")
        (at_second,) = cell.calculator.calls_on(1)
        obstacles = base_points(at_second, "scene_points_mm")
        # The neighbour cloud is thinned to one point per 8 mm cube (`neighbour_voxel_mm`): a dozen points is a block.
        self.assertGreater(int(block.holds(obstacles).sum()), 10, "the block the first look saw is not an obstacle")
        self.assertEqual(0, int(CUBE.holds(obstacles).sum()), "the part is its own obstacle")

    def test_grazing_points_are_thinned_before_the_looks_are_fused(self) -> None:
        cell = _Cell(looks=(LOOK_LOW,))

        cell.run()

        (view,) = cell.looked.views
        depth, hit = render(view.camera_to_base, (CUBE,))
        mask = hit == 0
        stepped = mask & ~pixels_behind_depth_steps(mask, depth)
        grazing = grazing_pixels(stepped, depth, K)
        np.testing.assert_array_equal(stepped & ~grazing, view.surfaces[0])
        on_top = to_base_mm(stepped & grazing, depth, K, view.camera_to_base)
        self.assertGreater(int(np.sum(on_top[:, 2] > 39.9)), 500, "the scene no longer shows a grazing top")
        seen, kept = to_base_mm(mask, depth, K, view.camera_to_base), view.clouds[0]
        self.assertEqual(_on_the_plus_x_face(seen), _on_the_plus_x_face(kept),
                         "the +x face, seen almost square, lost points")

    def test_a_surface_seen_edge_on_is_grazing_and_one_seen_square_is_not(self) -> None:
        for elevation, grazing in ((10.0, True), (45.0, False), (70.0, False)):
            camera = camera_looking_at(bearing_deg=0.0, elevation_deg=elevation)
            depth, hit = render(camera, (CUBE,))
            flagged = to_base_mm(grazing_pixels(hit == 0, depth, K), depth, K, camera)
            on_top = int(np.sum(flagged[:, 2] > 39.9))
            with self.subTest(elevation=elevation):
                self.assertEqual(grazing, on_top > 0, on_top)
                self.assertEqual(0, int(np.sum((flagged[:, 0] > 19.9) & (flagged[:, 2] < 39.0))))

    def test_looks_that_agree_on_the_part_say_the_hand_eye_holds(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), grasps_on=(1,))

        with self.assertNoLogs("src.robot.grasping.loop.pick_loop", level="WARNING"):
            cell.run()

        self.assertLess(cell.looked.hand_eye_gap_mm, 2.0)

    def test_looks_that_disagree_on_the_part_say_the_hand_eye_may_have_drifted(self) -> None:
        """The second look's frame is placed 9 mm above where the camera took it: a hand-eye that moved."""
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), grasps_on=(1,), stamp_error_mm={1: (0.0, 0.0, 9.0)})

        with self.assertLogs("src.robot.grasping.loop.pick_loop", level="WARNING") as said:
            cell.run()

        gap = cell.looked.hand_eye_gap_mm
        self.assertGreater(gap, HAND_EYE_DRIFT_WARN_MM)
        self.assertLess(gap, 12.0)
        self.assertTrue(any("hand-eye calibration may have drifted" in line for line in said.output), said.output)

    def test_the_faces_of_a_thin_part_are_not_read_as_drift(self) -> None:
        """On a calibration that is exact, a 12 mm pin seen from both sides shows two faces 12 mm apart that face away
        from each other, and a post seen from two sides shows two faces that meet at an edge: neither pair is surface
        both looks saw, and nothing is said about the hand-eye."""
        shapes = {
            "pin from +x and -x": (Box((-6.0, -706.0, 0.0), (6.0, -694.0, 60.0), "part"), (LOOK_PLUS_X, LOOK_MINUS_X)),
            "post from +x and +y": (Box((-10.0, -710.0, 0.0), (10.0, -690.0, 80.0), "part"),
                                    (LOOK_PLUS_X, LOOK_PLUS_Y)),
        }
        for name, (part, looks) in shapes.items():
            with self.subTest(name):
                cell = _Cell(looks=looks, grasps_on=(1,), boxes=(part,))

                with self.assertNoLogs("src.robot.grasping.loop.pick_loop", level="WARNING"):
                    cell.run()

                self.assertEqual(2, len(cell.looked.looks_fused))
                gap = cell.looked.hand_eye_gap_mm
                self.assertIsNotNone(gap, "the top both looks saw was not measured")
                self.assertLess(gap, 2.0)

    def test_two_looks_that_share_only_a_few_points_say_nothing_of_the_hand_eye(self) -> None:
        """Fewer than twenty shared points is an edge both looks caught, not a face: no gap is read off it."""
        seen_a, seen_b = np.eye(4), np.eye(4)
        seen_a[:3, 3] = (300.0, -700.0, 400.0)
        seen_b[:3, 3] = (-300.0, -700.0, 400.0)
        for side, speaks in ((4, False), (8, True)):
            xs, ys = np.meshgrid(np.arange(side, dtype=np.float64), np.arange(side, dtype=np.float64))
            patch = np.column_stack([xs.ravel(), ys.ravel() - 700.0, np.full(xs.size, 40.0)])
            earlier = _LookView(label="a", pose=None, name="wrist@a", frame=None,  # type: ignore[arg-type]
                                camera_to_base=seen_a, clouds=(patch,), blob_objects=(0,))
            this = _LookView(label="b", pose=None, name="wrist@b", frame=None,  # type: ignore[arg-type]
                             camera_to_base=seen_b, clouds=(patch + (0.0, 0.0, 1.0),), blob_objects=(0,))

            gap = BinPickingOrchestrator._hand_eye_gap_mm(this, 0, [earlier], {"wrist@a": 0, "wrist@b": 0})

            with self.subTest(points=side * side):
                if speaks:
                    self.assertAlmostEqual(1.0, gap, delta=0.2)
                else:
                    self.assertIsNone(gap)

    def test_a_fused_cloud_that_never_reached_the_support_leaves_the_declared_plane(self) -> None:
        """An 8 mm plate spans less than the 25 mm that says a cloud reached what it stands on, however many looks saw
        it: the declared support stands, the same gate in BASE as for one view."""
        plate = Box((-40.0, -740.0, 0.0), (40.0, -660.0, 8.0), "part")
        support = GraspingSupportConfig(height_mm=-50.0, refine_from_target=True)
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), grasps_on=(1,), boxes=(plate,), support_config=support,
                     camera_matrix=K)

        cell.run()

        (at_second,) = cell.calculator.calls_on(1)
        self.assertIn("geometry_points_base_mm", at_second, "the looks were not fused")
        self.assertAlmostEqual(-50.0, float(at_second["support_plane"].offset_mm), places=3)
        self.assertEqual("declared", cell.orchestrator._support_telemetry["support_source"])

    def test_a_kept_part_whose_grasp_is_lost_is_let_go_and_the_next_look_judges_its_own_choice(self) -> None:
        """The first look's grasp of the cube is valid and uncertain, so the cube is kept. The second look finds the
        cube and no grasp on it, so the cube is let go, and the third look judges what its own ranking chose: the bolt
        behind the cube."""
        bolt = Box((-20.0, -790.0, 0.0), (20.0, -750.0, 40.0), "bolt")
        centres = {"part": (0.0, -700.0, 20.0), "bolt": (0.0, -770.0, 20.0)}
        table = {(0, "part"): (GraspFailureReason.RESCAN_RECOMMENDED,), (2, "bolt"): ()}
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X, LOOK_PLUS_Y), boxes=(CUBE, bolt),
                     calculator=lambda camera: _ByPart(camera, table=table, centres=centres))

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(_label(LOOK_PLUS_Y), cell.looked.judged.look.label)
        np.testing.assert_allclose(cell.policy.executed[0].position, centres["bolt"])

    def test_two_looks_are_associated_at_the_wrist_tolerances(self) -> None:
        """A score of 0.30 over points 15 mm apart, scored on 3 mm voxels (``association.WRIST_VIEW_*``), and a wider
        radius the fusion config asks for kept."""
        cases: dict[str, tuple[dict[str, Any], float]] = {
            "no fusion config": ({}, 15.0),
            "a wider radius configured": (
                {"fusion_geometry_config": FusionGeometryConfig(enabled=False, neighbour_mm=20.0)}, 20.0),
        }
        real = scene_geometry.fuse_scene_geometry
        for name, (wiring, radius) in cases.items():
            with self.subTest(name):
                asked: list[dict[str, Any]] = []

                def spy(*args: Any, asked: list[dict[str, Any]] = asked, **kwargs: Any) -> Any:
                    asked.append(kwargs)
                    return real(*args, **kwargs)

                cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), grasps_on=(1,), **wiring)
                with mock.patch.object(scene_geometry, "fuse_scene_geometry", spy):
                    cell.run()

                (second,) = asked  # the first look had nothing to be fused with
                self.assertEqual((0.30, radius, 3.0),
                                 (second["min_score"], second["neighbour_mm"], second["score_voxel_mm"]))
                self.assertEqual("overlap", str(second["metric"]))

    def test_the_fusion_of_the_looks_is_timed(self) -> None:
        for looks, timed in (((LOOK_PLUS_X, LOOK_MINUS_X), True), ((LOOK_PLUS_X,), False)):
            with self.subTest(looks=len(looks)):
                tracker = LatencyTracker()
                cell = _Cell(looks=looks, grasps_on=(len(looks) - 1,), latency_tracker=tracker)

                cell.run()

                self.assertEqual(timed, tracker.get(LatencyStage.FUSION) is not None)

    def test_a_pick_lets_go_of_its_looks_when_it_ends(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), grasps_on=(1,))

        cell.run()

        self.assertEqual([], cell.orchestrator._look_views, "the looks' frames outlived the pick")
        self.assertEqual(2, len(cell.looked.views), "what the looks saw is no longer readable for the record")


class TheSurfaceTwoLooksShareTests(unittest.TestCase):
    """What the hand-eye check measures: each point's distance to the nearest point of the other look, where that lies
    within reach on a surface that faces the same way, each surface turned toward the camera that saw it."""

    @staticmethod
    def _face(axis: int, at: float, size: int = 30) -> np.ndarray:
        """A square face of points 1 mm apart, square to base ``axis`` at ``at``."""
        u, v = np.meshgrid(np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64))
        points = np.zeros((u.size, 3))
        across = [index for index in range(3) if index != axis]
        points[:, across[0]], points[:, across[1]], points[:, axis] = u.ravel(), v.ravel(), at
        return points

    def test_a_surface_both_looks_saw_is_measured_where_it_lies(self) -> None:
        top = self._face(2, 40.0)

        distances = nearest_surface_distances_mm(top + (0.0, 0.0, 5.0), top, seen_from_mm=(15.0, 15.0, 400.0),
                                                 other_seen_from_mm=(-300.0, 15.0, 400.0))

        self.assertGreater(distances.size, 500)
        self.assertAlmostEqual(5.0, float(np.median(distances)), delta=0.2)

    def test_two_faces_that_face_away_from_each_other_are_not_one_surface(self) -> None:
        near, far = self._face(0, 6.0), self._face(0, -6.0)

        distances = nearest_surface_distances_mm(far, near, seen_from_mm=(-400.0, 15.0, 15.0),
                                                 other_seen_from_mm=(400.0, 15.0, 15.0))

        self.assertEqual(0, distances.size, "the far face of a 12 mm part was measured against its near face")

    def test_nothing_out_of_reach_is_measured(self) -> None:
        top = self._face(2, 40.0)

        distances = nearest_surface_distances_mm(top + (0.0, 0.0, 16.0), top, seen_from_mm=(15.0, 15.0, 400.0),
                                                 other_seen_from_mm=(15.0, 15.0, 400.0))

        self.assertEqual(0, distances.size)


# ---------------------------------------------------------------------------------------------------------------------
# The one generated view (the owner, 2026-09-29): the last resort, on the straight joint line only
# ---------------------------------------------------------------------------------------------------------------------


def _lines_to(arm: _LineArm) -> list[tuple[float, ...]]:
    return [key for kind, key in arm.motions if kind == "line"]


class TheGeneratedViewTests(unittest.TestCase):
    """From the +x look, 45 degrees up, the cube's -x face (jaw 1's) comes within 60 degrees of incidence 140 degrees
    round either way (135 about the cube's centre in closed form; the face's own centre lies 20 mm further round, and
    the camera beside the tool): past the old 120 degree bound, inside the half turn, and said as a WARNING."""

    def test_the_view_turns_to_the_contact_face_no_look_showed_and_the_pick_goes_on_from_there(self) -> None:
        arm = _OrbitingArm(self)
        cell = _Cell(looks=(LOOK_PLUS_X,), arm=arm, both_faces=True)

        with self.assertLogs("src.robot.execution.generated_view", level="WARNING") as said:
            report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(140.0, cell.looked.generated_view_deg)
        # The executed joint path: the straight joint line from the look the arm stood at to the screened configuration.
        self.assertEqual([(_keys(LOOK_PLUS_X)[0], arm.turned_to(140.0))], arm.lines)
        self.assertEqual(_keys(LOOK_PLUS_X), cell.joints_moved_to(), "a declared-look verb ran for the generated view")
        generated = cell.looked.generated
        self.assertTrue(generated.startswith("orbit +140 deg ("), generated)
        self.assertEqual((_label(LOOK_PLUS_X), generated), cell.looked.visited)
        self.assertEqual((True, True), cell.looked.jaw_faces_seen, "the view did not show jaw 1's face")
        self.assertEqual(generated, cell.looked.judged.look.label, "the grasp is not ranked on the generated view")
        self.assertEqual("", cell.looked.moved_back, "the arm moved back from the look its grasp was ranked on")
        self.assertIn(generated, cell.looked.looks_fused)
        self.assertTrue(any("orbit +140 deg" in line for line in said.output), said.output)
        self.assertEqual(2, len(cell.camera.taken))

    def test_it_comes_only_after_every_declared_look_was_visited(self) -> None:
        """The +y look shows neither jaw face: after it, and only after it, the view turns from it to jaw 1's face,
        50 degrees on round the part, where the base joint stands at 140 degrees from the +x look."""
        arm = _OrbitingArm(self)
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_PLUS_Y), arm=arm, both_faces=True)

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(
            [("joints", _keys(LOOK_PLUS_X)[0]), ("joints", _keys(LOOK_PLUS_Y)[0]), ("line", arm.turned_to(140.0))],
            arm.motions)
        self.assertEqual(50.0, cell.looked.generated_view_deg, "the turn is from the look the grasp was judged on")

    def test_a_declared_look_refused_is_used_up_and_the_view_still_comes_last(self) -> None:
        """The declared looks are used up once each was reached or refused before anything was sent (addendum 1: 'once
        the declared looks are used up'). The -x look is refused after the +x look's uncertain grasp: it is skipped and
        said, and the one view is still generated after it; the warning says so rather than promising the pick goes on
        without one."""
        arm = _OrbitingArm(self, refuse=(LOOK_MINUS_X,))
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), arm=arm, uncertain_on=(0,), grasps_on=(1,))

        with self.assertLogs("src.robot.grasping.loop.pick_loop", level="WARNING") as said:
            report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(_keys(LOOK_PLUS_X, LOOK_MINUS_X), cell.joints_moved_to())
        self.assertEqual((_label(LOOK_MINUS_X),), tuple(look for look, _ in cell.looked.refused))
        self.assertEqual([(_keys(LOOK_PLUS_X)[0], arm.turned_to(140.0))], arm.lines)
        self.assertEqual(140.0, cell.looked.generated_view_deg)
        warning = next(line for line in said.output if "was refused" in line)
        self.assertIn(_label(LOOK_MINUS_X), warning)
        self.assertIn("with the views fused so far", warning)
        self.assertIn("once none is left and the grasp is still not safe enough, to the one generated view", warning)

    def test_a_safe_grasp_at_a_declared_look_generates_nothing(self) -> None:
        arm = _OrbitingArm(self)
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), arm=arm, both_faces=True)

        cell.run()

        self.assertEqual([], arm.screened)
        self.assertEqual([], arm.lines)
        self.assertIsNone(cell.looked.generated_view_deg)
        self.assertEqual("", cell.looked.generated_skipped)

    def test_a_turn_whose_straight_line_is_not_clear_is_skipped_and_the_next_tried(self) -> None:
        arm = _OrbitingArm(self)
        arm.blocked.add(arm.turned_to(140.0))
        cell = _Cell(looks=(LOOK_PLUS_X,), arm=arm, both_faces=True)

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual([arm.turned_to(140.0), arm.turned_to(-140.0)], _lines_to(arm))
        self.assertEqual(-140.0, cell.looked.generated_view_deg)

    def test_the_view_never_asks_the_planner_and_tries_at_most_three_motions(self) -> None:
        """Every straight line round the part is refused: nothing is planned around one (the arm's planner fails the
        test when asked), at most three motions are tried, and with both faces asked for nothing is gripped."""
        arm = _OrbitingArm(self)
        arm.blocked = {arm.turned_to(float(turn)) for turn in range(-180, 181, 5)}
        cell = _Cell(looks=(LOOK_PLUS_X,), arm=arm, both_faces=True)

        report = cell.run()

        self.assertEqual(3, len(arm.lines), "more generated motions were tried than three")
        self.assertIsNone(cell.looked.generated_view_deg)
        self.assertIn("3 motion(s) tried", cell.looked.generated_skipped)
        self.assertIs(PickOutcome.RESCANNED_EXHAUSTED, report.outcome)
        self.assertEqual([], cell.policy.executed)

    def test_an_arm_that_names_no_configuration_skips_the_view_and_says_why(self) -> None:
        """The simulator's arm, as the desk arm here: no ``nearest_configuration``, so no angle is screened, nothing
        moves, and the reason is said. An uncertain grasp still goes on, with the views fused."""
        cell = _Cell(looks=(LOOK_PLUS_X,), uncertain_on=(0,))

        with self.assertLogs("src.robot.execution.generated_view", level="INFO") as said:
            report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertIsNone(cell.looked.generated_view_deg)
        self.assertIn("ChoosesConfigurations", cell.looked.generated_skipped)
        self.assertTrue(any("ChoosesConfigurations" in line for line in said.output), said.output)
        self.assertEqual(_keys(LOOK_PLUS_X), cell.joints_moved_to())

    def test_a_generated_motion_that_may_have_moved_ends_the_attempt(self) -> None:
        arm = _OrbitingArm(self)
        arm.answers[arm.turned_to(140.0)] = (MotionStatus.CONTROLLER_REJECTED,
                                             "the drive refused after the move was judged, and moveJ may have been sent")
        cell = _Cell(looks=(LOOK_PLUS_X,), arm=arm, uncertain_on=(0,))

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTION_FAILED, report.outcome)
        (attempt,) = report.attempts
        self.assertEqual("look_refused", attempt.action)
        self.assertEqual([arm.turned_to(140.0)], _lines_to(arm), "the arm was sent on after a motion that may have moved")
        self.assertEqual([], cell.policy.executed)

    def test_a_pick_handed_no_look_generates_nothing(self) -> None:
        arm = _OrbitingArm(self)
        cell = _Cell(looks=(), arm=arm, uncertain_on=(0,))

        cell.run()

        self.assertEqual([], arm.screened)
        self.assertEqual([], arm.lines)

    def test_the_travel_cap_is_read_from_where_the_arm_stands_not_from_the_judged_look(self) -> None:
        """The grasp is judged on the +x look and the look after it misses the part, so the arm stands 20 degrees round
        on the base when the view turns from the +x look. The line at +140 is not clear; -140 lies 160 degrees from
        where the arm stands, past the cap, though only 140 from the judged look, so it is screened out, and the next
        turn runs, from where the arm stands."""
        arm = _OrbitingArm(self)
        arm.blocked.add(arm.turned_to(140.0))
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_ELSEWHERE), boxes=(CUBE, ALookThatMissesThePartTests.OTHER),
                     grasps_on=(2,), uncertain_on=(0,), arm=arm)

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertIn(-140.0, arm.screened)
        elsewhere = _keys(LOOK_ELSEWHERE)[0]
        self.assertEqual([(elsewhere, arm.turned_to(140.0)), (elsewhere, arm.turned_to(145.0))], arm.lines)
        self.assertEqual(145.0, cell.looked.generated_view_deg)

    def test_the_cable_window_is_held_about_the_arms_home(self) -> None:
        """Home 60 degrees the other way on the base: +140 stands 200 degrees from it, past half a turn, so -140 runs."""
        arm = _OrbitingArm(self)
        arm.home_joint_positions = tuple(JointPositions.deg(-60.0, *LOOK_PLUS_X.degrees()[1:]).tolist())
        cell = _Cell(looks=(LOOK_PLUS_X,), arm=arm, uncertain_on=(0,), grasps_on=(1,))

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual([(_keys(LOOK_PLUS_X)[0], arm.turned_to(-140.0))], arm.lines)
        self.assertEqual(-140.0, cell.looked.generated_view_deg)

    def test_a_world_that_does_not_hold_the_frames_moves_nothing_the_pick_made_up(self) -> None:
        """Neither the generated view nor the move back runs unless the arm's world holds every frame of the pick: the
        approach starts from where the arm stands, planned and judged as every approach is."""
        arms = {
            "a world that holds no frame": lambda: _OrbitingArm(self, world=_World(holds=False)),
            "no live world": lambda: _OrbitingArm(self, bare=True),
        }
        for name, make in arms.items():
            with self.subTest(name):
                arm = make()
                cell = _Cell(looks=(LOOK_PLUS_X, LOOK_ELSEWHERE), boxes=(CUBE, ALookThatMissesThePartTests.OTHER),
                             grasps_on=(), uncertain_on=(0,), arm=arm)

                report = cell.run()

                self.assertIs(PickOutcome.EXECUTED, report.outcome)
                self.assertEqual([], arm.screened)
                self.assertEqual([], arm.lines)
                self.assertIn("does not hold the frames of this pick", cell.looked.generated_skipped)
                self.assertIn("does not hold the frames of this pick", cell.looked.move_back_refused)
                self.assertEqual(_keys(LOOK_ELSEWHERE), [joints_key(joints) for joints in cell.policy.joints_at])

    def test_with_no_candidate_the_view_turns_to_the_side_no_look_faced(self) -> None:
        arm = _OrbitingArm(self)
        cell = _Cell(looks=(LOOK_PLUS_X,), grasps_on=(), arm=arm)

        with self.assertLogs("src.robot.execution.generated_view", level="INFO") as said:
            cell.run()

        turn = cell.looked.generated_view_deg
        self.assertIsNotNone(turn)
        self.assertGreater(abs(turn), 90.0, "the view did not turn to the side the +x look did not face")
        ((start, _),) = arm.lines
        self.assertEqual(_keys(LOOK_PLUS_X)[0], start)
        self.assertTrue(any("no look faced" in line for line in said.output), said.output)
        self.assertEqual(2, len(cell.camera.taken))

    def test_a_stop_asked_for_after_the_last_look_generates_nothing(self) -> None:
        arm = _OrbitingArm(self)
        cell = _Cell(looks=(LOOK_PLUS_X,), arm=arm, uncertain_on=(0,))
        cell.orchestrator.should_cancel = lambda: len(cell.camera.taken) > 0

        report = cell.run()

        self.assertIs(PickOutcome.CANCELLED, report.outcome)
        self.assertEqual([], arm.screened)
        self.assertEqual([], arm.lines)
        self.assertEqual([], cell.policy.executed)


# ---------------------------------------------------------------------------------------------------------------------
# Both jaw contact faces asked for: no view showed both, nothing is gripped
# ---------------------------------------------------------------------------------------------------------------------


class BothFacesFailClosedTests(unittest.TestCase):
    def test_no_view_showing_both_contact_faces_grips_nothing_and_names_the_face(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X,), both_faces=True)

        with self.assertLogs("src.robot.grasping.loop.pick_loop", level="ERROR") as said:
            report = cell.run()

        self.assertIs(PickOutcome.RESCANNED_EXHAUSTED, report.outcome)
        (attempt,) = report.attempts
        self.assertEqual("faces_unseen", attempt.action)
        self.assertEqual((GraspFailureReason.NO_VALID_GRASP,), attempt.reasons)
        self.assertEqual([], cell.policy.executed)
        self.assertIn("the contact face at jaw 1 of the chosen grasp was not seen", cell.looked.withheld)
        self.assertTrue(any("jaw 1" in line and "nothing is gripped" in line for line in said.output), said.output)
        (finished,) = [event for event in cell.events if event.stage is PickStage.ATTEMPT_FINISHED]
        self.assertEqual("faces_unseen", finished.action)
        self.assertIn("jaw 1", finished.extra["withheld"])

    def test_off_by_default_the_same_grasp_is_gripped(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X,), uncertain_on=(0,))

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual("", cell.looked.withheld)

    def test_faces_that_could_not_be_judged_grip_nothing(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X,), both_faces=True)

        with mock.patch.object(BinPickingOrchestrator, "_jaw_faces", return_value=None):
            report = cell.run()

        self.assertIs(PickOutcome.RESCANNED_EXHAUSTED, report.outcome)
        (attempt,) = report.attempts
        self.assertEqual("faces_unseen", attempt.action)
        self.assertEqual([], cell.policy.executed)
        self.assertIn("could not be judged", cell.looked.withheld)

    def test_a_fixed_camera_grips_only_a_grasp_its_cameras_showed_both_faces_of(self) -> None:
        """A fixed camera cannot look around: from the +x side alone jaw 1's face is not seen, and nothing is gripped;
        with a second fixed camera behind the part fused in, both faces are seen, and the part is gripped."""
        fixed = StaticCameraToBaseResolver(transform=Transform.from_matrix(
            camera_looking_at(bearing_deg=0.0), from_frame=Frame.CAMERA, to_frame=Frame.BASE))
        cases: dict[str, tuple[dict[str, Any], bool]] = {
            "one camera": ({}, False),
            "a second camera behind the part": (_behind_camera(), True),
        }
        for name, (wiring, gripped) in cases.items():
            with self.subTest(name):
                cell = _Cell(looks=(), frame_resolver=fixed, both_faces=True, **wiring)

                report = cell.run()

                self.assertIsNone(cell.looked, "a fixed camera looked around")
                if gripped:
                    self.assertIs(PickOutcome.EXECUTED, report.outcome)
                    self.assertEqual(1, len(cell.policy.executed))
                    continue
                self.assertIs(PickOutcome.RESCANNED_EXHAUSTED, report.outcome)
                (attempt,) = report.attempts
                self.assertEqual("faces_unseen", attempt.action)
                self.assertEqual((GraspFailureReason.NO_VALID_GRASP,), attempt.reasons)
                self.assertEqual([], cell.policy.executed)
                (finished,) = [event for event in cell.events if event.stage is PickStage.ATTEMPT_FINISHED]
                self.assertIn("the contact face at jaw 1 of the chosen grasp was not seen", finished.extra["withheld"])


class _TwoGrasps(LookCalculator):
    """The scene's calculator, with a second candidate behind every grasp it gives: the same grasp 5 mm along base y,
    scored lower. What a rerank, or the approach check's fall-back, could put in the first one's place."""

    def compute_result(self, seg: Any, *args: Any, **kwargs: Any) -> GraspResult:
        result = super().compute_result(seg, *args, **kwargs)
        if not result.is_success:
            return result
        best = result.candidates[0]
        other = GraspPoint(position=np.asarray(best.position) + (0.0, 5.0, 0.0), approach=best.approach, axis=best.axis,
                           grip_width_mm=40.0, score=0.8, frame=GraspFrame.BASE, label=best.label)
        return GraspResult(candidates=(best, other), top_score=result.top_score, reasons=result.reasons)


class TheGraspGrippedIsTheOneWhoseFacesWereJudgedTests(unittest.TestCase):
    """Integration of 2026-09-29: ``both_faces`` is judged on the calculator's best grasp at ranking time, so the grasp
    gripped must be that one. The reranks stand down, and the approach check falls back to no other candidate, whose
    contact faces nobody judged. Off, both run as they always did."""

    def test_with_both_faces_a_blocked_approach_falls_back_to_no_grasp_whose_faces_nobody_judged(self) -> None:
        for asked, gripped in ((False, [0.8]), (True, [])):
            with self.subTest(both_faces=asked):
                cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), calculator=_TwoGrasps, both_faces=asked,
                             approach_path_policy=object(), mode_label="dense_clutter",
                             _approach_path_modes=frozenset({"dense_clutter"}))
                # The best grasp's approach is blocked, the second's is clear.
                with mock.patch.object(BinPickingOrchestrator, "_approach_sweep_is_clear",
                                       side_effect=lambda candidate, _cloud: float(candidate.score) < 0.85):
                    report = cell.run()

                self.assertEqual(gripped, [float(grasp.score) for grasp in cell.policy.executed])
                self.assertIs(PickOutcome.APPROACH_PATH_BLOCKED if asked else PickOutcome.EXECUTED, report.outcome)

    def test_with_both_faces_the_reranks_stand_down(self) -> None:
        from src.robot.grasping.loop import pick_loop

        blend, uncertainty = object(), object()
        for asked, handed in ((False, (blend, uncertainty)), (True, (None, None))):
            with self.subTest(both_faces=asked):
                cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), both_faces=asked, ranking_blend_config=blend,
                             uncertainty_rerank_config=uncertainty)
                with mock.patch.object(pick_loop, "maybe_blend_rerank_candidates",
                                       side_effect=lambda candidates, **_: (candidates, None)) as blended, \
                        mock.patch.object(pick_loop, "maybe_uncertainty_rerank_candidates",
                                          side_effect=lambda candidates, **_: (candidates, None)) as reranked:
                    report = cell.run()

                self.assertIs(PickOutcome.EXECUTED, report.outcome)
                self.assertEqual(handed, (blended.call_args.kwargs["config"], reranked.call_args.kwargs["config"]))

    def test_with_both_faces_a_motion_that_turns_the_grasp_is_refused_before_anything_moves(self) -> None:
        """A policy that yaws every grasp about base Z onto base X (``GraspMotion(align_closing_to_base_x=True)``)
        closes the jaws on another pair of faces than the one judged. With both faces asked for, the pick and its look
        around raise before the arm moves or the camera takes a frame. Off, the same motion runs as it always did."""
        from src.robot.grasping.motion.grasp_motion import GraspMotion, build_execution_policy

        for entry in ("run", "look_around"):
            with self.subTest(entry):
                cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), both_faces=True)
                cell.orchestrator.policy = build_execution_policy(
                    GraspMotion(align_closing_to_base_x=True), arm=cell.arm, gripper=None, standoff_mm=80.0,
                    retreat_mm=100.0, base_frame_required=True)

                with self.assertRaises(ValueError) as refused:
                    getattr(cell.orchestrator, entry)()

                self.assertIn("align_closing_to_base_x", str(refused.exception))
                self.assertEqual([], cell.arm.motions)
                self.assertEqual([], cell.camera.taken)
                self.assertEqual([], [event for event in cell.events if event.stage is PickStage.PICK_STARTED])
        cell = _Cell(looks=(LOOK_PLUS_X,))
        cell.policy.align_closing_to_base_x = True  # type: ignore[attr-defined]
        self.assertIs(PickOutcome.EXECUTED, cell.run().outcome, "the control: without both faces the motion runs")


# ---------------------------------------------------------------------------------------------------------------------
# The move back to the look that saw the part: the straight joint line, or the approach starts where the arm stands
# ---------------------------------------------------------------------------------------------------------------------


class TheMoveBackTests(unittest.TestCase):
    OTHER = ALookThatMissesThePartTests.OTHER

    def _cell(self, arm: _LineArm) -> _Cell:
        return _Cell(looks=(LOOK_PLUS_X, LOOK_ELSEWHERE), boxes=(CUBE, self.OTHER), grasps_on=(), uncertain_on=(0,),
                     arm=arm)

    def test_a_move_back_whose_line_is_not_clear_approaches_from_where_the_arm_stands(self) -> None:
        """Nothing is planned around the line back: the arm's planner fails the test when it is asked for the look a
        second time, the way a move back through the verb that plans around a line would ask it."""
        arm = _LineArm(self)
        arm.blocked.add(_keys(LOOK_PLUS_X)[0])
        cell = self._cell(arm)

        with self.assertLogs("src.robot.grasping.loop.pick_loop", level="WARNING") as said:
            report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual([_keys(LOOK_PLUS_X)[0]], _lines_to(arm), "the move back was tried other than once")
        self.assertEqual(_keys(LOOK_ELSEWHERE), [joints_key(joints) for joints in cell.policy.joints_at])
        self.assertEqual("", cell.looked.moved_back)
        self.assertIn("self_collision_rejected", cell.looked.move_back_refused)
        warning = next(line for line in said.output if "move back" in line)
        self.assertIn("the approach starts from where the arm stands", warning)

    def test_a_move_back_that_may_have_moved_ends_the_attempt_with_nothing_else_commanded(self) -> None:
        arm = _LineArm(self)
        arm.answers[_keys(LOOK_PLUS_X)[0]] = (MotionStatus.TIMEOUT, "moveJ did not complete within its budget")
        cell = self._cell(arm)

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTION_FAILED, report.outcome)
        (attempt,) = report.attempts
        self.assertEqual("look_refused", attempt.action)
        self.assertEqual("timeout", attempt.motion_status)
        self.assertEqual([], cell.policy.executed)
        self.assertEqual(_label(LOOK_PLUS_X), cell.looked.stopped_look)

    def test_an_arm_that_cannot_drive_the_line_alone_approaches_from_where_it_stands(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_ELSEWHERE), boxes=(CUBE, self.OTHER), grasps_on=(), uncertain_on=(0,))

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(_keys(LOOK_PLUS_X, LOOK_ELSEWHERE), cell.joints_moved_to(), "the arm was moved back")
        self.assertIn("DrivesJointLines", cell.looked.move_back_refused)

    def test_the_move_back_goes_to_the_look_that_saw_the_part_not_to_the_first(self) -> None:
        """The first look ranks no grasp, the second an uncertain one, the third misses the part: back to the second."""
        arm = _LineArm(self)
        cell = _Cell(looks=(LOOK_PLUS_Y, LOOK_PLUS_X, LOOK_ELSEWHERE), boxes=(CUBE, self.OTHER), grasps_on=(),
                     uncertain_on=(1,), arm=arm)

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(_label(LOOK_PLUS_X), cell.looked.judged.look.label)
        self.assertEqual([(_keys(LOOK_ELSEWHERE)[0], _keys(LOOK_PLUS_X)[0])], arm.lines)
        self.assertEqual(_keys(LOOK_PLUS_X), [joints_key(joints) for joints in cell.policy.joints_at])

    def test_a_long_move_back_is_said_and_one_past_the_travel_cap_is_refused(self) -> None:
        """The look that misses the part stands 130, or 170, degrees round on the base from the one that saw it: the
        first line back runs with a WARNING, the second is past the 150 degree cap, refused, and the approach starts
        from where the arm stands."""
        cases = {"130 degrees back": (LOOK_ELSEWHERE_WIDE, True), "170 degrees back": (LOOK_ELSEWHERE_FAR, False)}
        for name, (far, driven) in cases.items():
            with self.subTest(name):
                arm = _LineArm(self)
                cell = _Cell(looks=(LOOK_PLUS_X, far), boxes=(CUBE, self.OTHER), grasps_on=(), uncertain_on=(0,),
                             arm=arm)

                if driven:
                    # The generated view's own logger, by name: it is a robot logger and does not propagate to the root.
                    with self.assertLogs("src.robot.execution.generated_view", level="WARNING") as said:
                        report = cell.run()
                else:
                    report = cell.run()

                self.assertIs(PickOutcome.EXECUTED, report.outcome)
                if driven:
                    self.assertEqual([(_keys(far)[0], _keys(LOOK_PLUS_X)[0])], arm.lines)
                    self.assertTrue(any("joint 1 travelling 130.0 deg" in line for line in said.output), said.output)
                    self.assertEqual(_keys(LOOK_PLUS_X), [joints_key(joints) for joints in cell.policy.joints_at])
                else:
                    self.assertEqual([], arm.lines)
                    self.assertIn("joint 1 would travel 170.0 deg", cell.looked.move_back_refused)
                    self.assertEqual(_keys(far), [joints_key(joints) for joints in cell.policy.joints_at])

    def test_a_stop_asked_for_before_the_move_back_sends_the_arm_nowhere_else(self) -> None:
        arm = _LineArm(self)
        cell = self._cell(arm)
        dealt: list[bool] = []
        real = cell.orchestrator._generated_view

        def generated(*args: Any, **kwargs: Any) -> Any:
            dealt.append(True)
            return real(*args, **kwargs)

        cell.orchestrator.should_cancel = lambda: bool(dealt)
        with mock.patch.object(cell.orchestrator, "_generated_view", side_effect=generated):
            report = cell.run()

        self.assertTrue(dealt, "no generated view was due, so the stop was asked for elsewhere")
        self.assertIs(PickOutcome.CANCELLED, report.outcome)
        self.assertEqual([], arm.lines)
        self.assertEqual([], cell.policy.executed)


# ---------------------------------------------------------------------------------------------------------------------
# The planner world of a pick (plan step 4c): every frame held until it ends, and the fused target left out
# ---------------------------------------------------------------------------------------------------------------------


class ThePlannerWorldOfAPickTests(unittest.TestCase):
    def test_the_frames_are_held_from_the_first_look_and_let_go_when_the_pick_ends(self) -> None:
        cases: dict[str, dict[str, Any]] = {
            "a pick that grips": {"grasps_on": (1,)},
            "a pick that finds no grasp": {"grasps_on": ()},
            "a pick whose look motion may have moved the arm": {"answers": True},
        }
        for name, case in cases.items():
            with self.subTest(name):
                world = _World()
                arm = _LineArm(self, world=world)
                if case.pop("answers", False):
                    arm = _FailingLookArm(fail_at=LOOK_MINUS_X, answer=(MotionStatus.TIMEOUT, "moveJ ran out of time"))
                    arm.live_planner_world = world  # type: ignore[attr-defined]
                    case["uncertain_on"] = (0,)
                cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), arm=arm, **case)

                cell.run()

                self.assertEqual(["hold", "forget"], world.calls)
                self.assertFalse(world.holding, "the frames of the pick outlived it")
                if cell.policy.executed:
                    self.assertEqual([True], cell.policy.held_at, "the approach was judged without the pick's frames")

    def test_a_pick_that_raises_lets_its_frames_go(self) -> None:
        world = _World()
        arm = LookingArm(_poses(), unvouched=(LOOK_MINUS_X,))
        arm.live_planner_world = world  # type: ignore[attr-defined]
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), arm=arm, uncertain_on=(0,))

        with self.assertRaises(CameraWorldUnavailable):
            cell.run()

        self.assertEqual(["hold", "forget"], world.calls)

    def test_a_look_around_that_raises_lets_its_frames_go(self) -> None:
        """The public look around, as the service's decision path calls it: its frames go when it raises."""
        world = _World()
        arm = LookingArm(_poses(), unvouched=(LOOK_MINUS_X,))
        arm.live_planner_world = world  # type: ignore[attr-defined]
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), arm=arm, uncertain_on=(0,))

        with self.assertRaises(CameraWorldUnavailable):
            cell.orchestrator.look_around()

        self.assertEqual(["hold", "forget"], world.calls)
        self.assertFalse(world.holding)

    def test_a_wrist_pick_handed_no_look_offers_and_places_its_part_by_its_own_frame(self) -> None:
        """Until a pick is handed looks, the planner world is offered the part as its own frame saw it, and the part's
        centre is read off that frame, even where a fixed camera is fused into its grasp: the fused target belongs to
        a pick that looked around."""
        world = _World()
        cell = _Cell(looks=(), arm=_LineArm(self, world=world), **_top_camera())

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(2, len(cell.looked.judged.members), "the fixed camera was not fused into the grasp")
        (view,) = cell.looked.views
        own = segmentation_offer_from_frame(view.frame, 0, camera=UNSET, camera_to_base_mm=view.camera_to_base)
        np.testing.assert_array_equal(own.target_points_base_mm, world.offers[-1]["target_points_base_mm"])
        self.assertEqual(cell.orchestrator._target_centre_base_mm(view.frame, 0), report.target_centre_mm)

    def test_a_pick_handed_no_look_holds_nothing(self) -> None:
        world = _World()
        arm = _LineArm(self, world=world)
        cell = _Cell(looks=(), arm=arm)

        cell.run()

        self.assertEqual([], world.calls)

    def test_the_planner_world_is_offered_the_fused_target(self) -> None:
        """The grasp is ranked on the -x look, whose own surface holds no point of the +x face: the fused one does."""
        world = _World()
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), grasps_on=(1,), arm=_LineArm(self, world=world))

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        offered = np.asarray(world.offers[-1]["target_points_base_mm"])
        self.assertGreater(int(np.sum(offered[:, 0] > 19.0)), 500, "the +x face the first look saw was not offered")
        self.assertGreater(int(np.sum(offered[:, 0] < -19.0)), 500, "the -x face was not offered")

    def test_the_target_centre_is_the_middle_of_the_fused_cloud(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), grasps_on=(1,))

        report = cell.run()

        centre = report.target_centre_mm
        self.assertIsNotNone(centre)
        self.assertLess(abs(centre[0] - CUBE_CENTRE[0]), 3.0, f"the centre is one look's side of the part: {centre}")
        self.assertLess(abs(centre[1] - CUBE_CENTRE[1]), 3.0, centre)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
