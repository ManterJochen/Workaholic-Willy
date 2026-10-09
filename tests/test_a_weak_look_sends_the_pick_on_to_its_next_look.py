"""A weak look sends a wrist pick on to its next look, where the cell turns the trigger on (the owner, 2026-10-08 night).

A valid grasp at the first look ended the looking on the owner's cell every time: a stacked pair or a shiny part was
gripped on what one look showed of it. The automatic trigger (``robot.grasping.weak_look_trigger``, the map's change C)
judges a look whose grasp is valid weak where its view of the part is: depth measured on less than 85 % of its own mask;
a part standing more than one and a half times its footprint's short side over its support with no side of it seen; or
two height plateaus in its cloud at least 15 mm apart. A weak look carries ``ACTIVE_PERCEPTION_RECOMMENDED``, so it is
not safe to stop at, and the pick goes on to its next look. It is judged again on the views fused so far at every look,
so it clears once a side or more depth was seen, and it only adds looks: the grasp the looks end on is gripped as before.
A wrist pick handed looks only; a fixed camera, and a pick handed none, are never judged. Off by default; the cell's
tree turns it on, and the service sets it on the pick loop where the cell is built.

The scene is ``tests/_wrist_views.py``: parts on the bench, ray cast through the wrist D415 from the tool pose each look
puts the arm at. The pick loop is the real one; the calculator grasps the part on every frame.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from src.config.schema.robot import RobotConfig
from src.config.schema.robot.grasping_schema import GraspingSupportConfig, RobotGraspingConfig
from src.geometry import Frame, Transform
from src.robot.core import JointPositions
from src.robot.execution.autonomous_grasp import AutonomousGraspService
from src.robot.grasping.collision.gripper_model import ParallelJawGripperModel
from src.robot.grasping.loop.pick_loop import (
    WEAK_LOOK_DEPTH_SHARE,
    BinPickingOrchestrator,
    PickOutcome,
    _footprint_short_side_mm,
    _two_plateaus,
)
from src.robot.grasping.motion.execution_policy import PolicyOutcome, PolicyReport
from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver, StaticCameraToBaseResolver
from src.robot.grasping.types.feedback import GraspFailureReason
from src.robot.grasping.types.perception import PerceptionFrame
from tests._helpers import _FakePerception, _perception_frame, _ScriptedCalculator
from tests._wrist_views import (
    CUBE,
    Box,
    LookCalculator,
    LookingArm,
    WristCamera,
    camera_looking_at,
    camera_to_tool,
    tool_for,
)

#: A 20 mm square part standing 80 mm tall: two such cubes stacked, as a look from straight above sees them.
TALL = Box((-10.0, -710.0, 0.0), (10.0, -690.0, 80.0), "part")
TALL_CENTRE = (0.0, -700.0, 40.0)
#: The cube from its +x side and its -x side, 45 degrees up; the tall part from straight above and from its +x side.
LOOK_PLUS_X = JointPositions.deg(0.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LOOK_MINUS_X = JointPositions.deg(180.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LOOK_ABOVE = JointPositions.deg(30.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LOOK_SIDE = JointPositions.deg(60.0, -90.0, -110.0, -60.0, 90.0, 0.0)
#: The Hand-E's contact patch (config/grippers/robotiq_hande.yaml), the owner's hand.
HAND_E = ParallelJawGripperModel(finger_width_mm=29.24, pad_length_mm=20.91, pad_ahead_mm=10.45)


def _label(joints: JointPositions) -> str:
    return "(" + ", ".join(f"{v:.1f}" for v in joints.degrees()) + ") deg"


def _poses() -> dict[JointPositions, Any]:
    return {
        LOOK_PLUS_X: tool_for(camera_looking_at(bearing_deg=0.0)),
        LOOK_MINUS_X: tool_for(camera_looking_at(bearing_deg=180.0)),
        LOOK_ABOVE: tool_for(camera_looking_at(TALL_CENTRE, bearing_deg=0.0, elevation_deg=89.5)),
        LOOK_SIDE: tool_for(camera_looking_at(TALL_CENTRE, bearing_deg=0.0, elevation_deg=45.0)),
    }


@dataclass(eq=False)
class _HolesIn(WristCamera):
    """The wrist D415, whose depth on the frames in ``holes`` is missing on every third and fourth of ten pixels of the
    parts it sees: 30 % of each part's mask, as a shiny or clear part reads."""

    holes: tuple[int, ...] = field(default_factory=tuple)

    def acquire(self) -> PerceptionFrame:
        number = len(self.taken)
        frame = super().acquire()
        if number not in self.holes:
            return frame
        rows, cols = np.indices(frame.depth_map.shape)
        parts = np.zeros(frame.depth_map.shape, dtype=bool)
        for seg in frame.segmentations:
            parts |= np.asarray(seg.mask, dtype=bool)
        holed = np.where(parts & ((rows + cols) % 10 >= 7), 0.0, frame.depth_map)
        return replace(frame, depth_map=holed)


class _Policy:
    def __init__(self) -> None:
        self.executed: list[Any] = []

    def execute(self, grasp: Any) -> PolicyReport:
        self.executed.append(grasp)
        return PolicyReport(outcome=PolicyOutcome.EXECUTED)


def _loop(*, looks: tuple[JointPositions, ...], boxes: tuple[Box, ...] = (CUBE,), holes: tuple[int, ...] = (),
          weak_look: bool = True, fixed: bool = False) -> tuple[BinPickingOrchestrator, _HolesIn, _Policy]:
    """The real pick loop on the looking arm and the wrist camera (or a fixed one), the parts standing on a declared
    support at the bench, the cell's trigger as ``weak_look`` says."""
    arm = LookingArm(_poses())
    camera = _HolesIn(arm, boxes=boxes, holes=holes)
    policy = _Policy()
    resolver: Any = (StaticCameraToBaseResolver(transform=Transform.from_matrix(
        camera_looking_at(bearing_deg=0.0), from_frame=Frame.CAMERA, to_frame=Frame.BASE)) if fixed
        else EyeInHandFrameResolver(t_cam_to_tool=camera_to_tool()))
    orchestrator = BinPickingOrchestrator(
        arm=arm, calculator=LookCalculator(camera), perception=camera,  # type: ignore[arg-type]
        frame_resolver=resolver, policy=policy,  # type: ignore[arg-type]
        max_attempts=1, primary_camera_id="wrist", gripper_model=HAND_E, looks=looks,
        support_config=GraspingSupportConfig(), weak_look=weak_look,
    )
    return orchestrator, camera, policy


class AWeakLookGoesOnTests(unittest.TestCase):
    def test_a_part_with_little_depth_goes_on_to_the_next_look(self) -> None:
        orchestrator, camera, policy = _loop(looks=(LOOK_PLUS_X, LOOK_MINUS_X), holes=(0,))

        with self.assertLogs("src.robot.grasping.loop.pick_loop", level="INFO") as said:
            report = orchestrator.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        looked = orchestrator.looked_around
        self.assertEqual((_label(LOOK_PLUS_X), _label(LOOK_MINUS_X)), looked.visited)
        [(look, why)] = looked.weak
        self.assertEqual(_label(LOOK_PLUS_X), look)
        self.assertIn("depth on 70% of its mask", why)
        self.assertTrue(looked.judged.good, "the look that saw the part whole is not safe to stop at")
        self.assertEqual(_label(LOOK_MINUS_X), looked.judged.look.label)
        self.assertEqual(1, len(policy.executed))
        self.assertTrue(any("its view of the part is weak" in line for line in said.output), said.output)

    def test_a_tall_part_seen_only_from_above_goes_on_and_stops_once_its_side_was_seen(self) -> None:
        orchestrator, camera, policy = _loop(looks=(LOOK_ABOVE, LOOK_SIDE, LOOK_PLUS_X), boxes=(TALL,))

        report = orchestrator.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        looked = orchestrator.looked_around
        self.assertEqual((_label(LOOK_ABOVE), _label(LOOK_SIDE)), looked.visited, "the looking did not stop at the side")
        [(look, why)] = looked.weak
        self.assertEqual(_label(LOOK_ABOVE), look)
        self.assertIn("no side of it seen", why)
        self.assertTrue(looked.judged.good)
        self.assertNotIn(GraspFailureReason.ACTIVE_PERCEPTION_RECOMMENDED, looked.judged.reasons)

    def test_a_weak_look_carries_the_reason_that_drives_the_next_look(self) -> None:
        orchestrator, _, _ = _loop(looks=(LOOK_ABOVE,), boxes=(TALL,))

        report = orchestrator.run()

        judged = orchestrator.looked_around.judged
        self.assertIn(GraspFailureReason.ACTIVE_PERCEPTION_RECOMMENDED, judged.reasons)
        self.assertFalse(judged.good)
        self.assertTrue(judged.weak)
        self.assertIs(PickOutcome.EXECUTED, report.outcome, "a weak look's grasp was not gripped once the looks ran out")
        self.assertEqual((), report.attempts[-1].reasons, "the executed attempt's reasons are the calculator's")

    def test_a_matte_cube_seen_from_its_side_stops_at_the_first_look(self) -> None:
        orchestrator, camera, _ = _loop(looks=(LOOK_PLUS_X, LOOK_MINUS_X))

        orchestrator.run()

        looked = orchestrator.looked_around
        self.assertEqual((_label(LOOK_PLUS_X),), looked.visited)
        self.assertEqual((), looked.weak)
        self.assertEqual(1, len(camera.taken))

    def test_off_by_default_a_part_with_little_depth_stops_at_the_first_look_as_before(self) -> None:
        orchestrator, camera, _ = _loop(looks=(LOOK_PLUS_X, LOOK_MINUS_X), holes=(0,), weak_look=False)

        orchestrator.run()

        self.assertEqual((_label(LOOK_PLUS_X),), orchestrator.looked_around.visited)
        self.assertEqual((), orchestrator.looked_around.weak)
        self.assertIs(False, BinPickingOrchestrator.__dataclass_fields__["weak_look"].default)


class NoLookIsJudgedWhereNoneIsHandedTests(unittest.TestCase):
    def test_a_fixed_camera_never_judges_a_look_weak(self) -> None:
        orchestrator, camera, policy = _loop(looks=(LOOK_PLUS_X, LOOK_MINUS_X), holes=(0,), fixed=True)

        report = orchestrator.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(1, len(camera.taken))
        self.assertIsNone(orchestrator.looked_around)

    def test_a_wrist_pick_handed_no_looks_is_never_judged_weak(self) -> None:
        orchestrator, camera, policy = _loop(looks=(), holes=(0,))

        report = orchestrator.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(1, len(camera.taken))
        self.assertEqual((), orchestrator.looked_around.weak)
        self.assertTrue(orchestrator.looked_around.judged.good)

    def test_a_pick_with_multi_view_off_grips_what_its_first_look_saw_unjudged(self) -> None:
        orchestrator, camera, policy = _loop(looks=(LOOK_ABOVE,), boxes=(TALL,))
        orchestrator.generated_view = False  # multi-view off (Q11): the first look alone, no view generated after it

        report = orchestrator.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(1, len(camera.taken))
        self.assertEqual((), orchestrator.looked_around.weak, "a look no other look may follow was judged weak")
        self.assertTrue(orchestrator.looked_around.judged.good)


class TheShapeTriggersTests(unittest.TestCase):
    """The two shape triggers on clouds of known heights: what a stack seen from above makes, and what a side does not."""

    def test_two_plateaus_far_enough_apart_are_two_parts_stacked(self) -> None:
        rng = np.random.default_rng(3)
        heights = np.concatenate([rng.normal(40.0, 0.7, 600), rng.normal(80.0, 0.7, 400)])
        found = _two_plateaus(heights)
        self.assertIsNotNone(found)
        assert found is not None
        self.assertAlmostEqual(40.0, found[0], delta=2.0)
        self.assertAlmostEqual(80.0, found[1], delta=2.0)

    def test_a_side_seen_at_an_angle_makes_no_second_plateau(self) -> None:
        rng = np.random.default_rng(4)
        top = rng.normal(40.0, 0.7, 500)
        for side in (rng.uniform(0.0, 40.0, 500), rng.uniform(0.0, 40.0, 1500), rng.uniform(20.0, 40.0, 300)):
            with self.subTest(points=side.size):
                self.assertIsNone(_two_plateaus(np.concatenate([top, side])))

    def test_plateaus_too_near_or_too_small_are_no_stack(self) -> None:
        rng = np.random.default_rng(5)
        near = np.concatenate([rng.normal(40.0, 0.5, 500), rng.normal(50.0, 0.5, 500)])
        small = np.concatenate([rng.normal(40.0, 0.5, 950), rng.normal(80.0, 0.5, 50)])
        self.assertIsNone(_two_plateaus(near), "10 mm apart is not 15")
        self.assertIsNone(_two_plateaus(small), "5 % of the points is not a plateau")
        self.assertIsNone(_two_plateaus(np.array([40.0, 80.0] * 10)), "too few points to say")

    def test_the_footprint_is_read_along_the_parts_own_directions(self) -> None:
        rng = np.random.default_rng(6)
        along = rng.uniform(-30.0, 30.0, 2000)
        across = rng.uniform(-10.0, 10.0, 2000)
        turn = np.radians(35.0)
        cloud = np.column_stack([along * np.cos(turn) - across * np.sin(turn),
                                 along * np.sin(turn) + across * np.cos(turn), np.full(2000, 40.0)])
        short = _footprint_short_side_mm(cloud, np.array([0.0, 0.0, 1.0]))
        self.assertIsNotNone(short)
        self.assertAlmostEqual(19.2, float(short), delta=1.0)
        self.assertIsNone(_footprint_short_side_mm(cloud[:10], np.array([0.0, 0.0, 1.0])))

    def test_the_depth_share_is_the_owners(self) -> None:
        self.assertEqual(0.85, WEAK_LOOK_DEPTH_SHARE)


class TheCellsKeyTests(unittest.TestCase):
    def test_the_key_is_off_by_default_and_can_be_switched_on(self) -> None:
        self.assertIs(False, RobotGraspingConfig().weak_look_trigger)
        self.assertIs(True, RobotGraspingConfig.model_validate({"weak_look_trigger": True}).weak_look_trigger)

    def test_a_cell_built_from_its_tree_judges_weak_looks_as_its_key_says(self) -> None:
        def built(**grasping: object) -> Any:
            config = RobotConfig(vendor="dummy", gripper={"vendor": "none"}, grasping=grasping)
            return AutonomousGraspService.from_robot_config(
                config, calculator=_ScriptedCalculator([]),  # type: ignore[arg-type]
                perception=_FakePerception([_perception_frame()])).runtime.orchestrator

        self.assertIs(False, built().weak_look)
        self.assertIs(True, built(weak_look_trigger=True).weak_look)


if __name__ == "__main__":
    unittest.main()
