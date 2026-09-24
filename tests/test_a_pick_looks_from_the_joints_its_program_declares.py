"""A pick looks from the joint positions its program declares, in order, until one finds something.

The owner, 2026-09-24: the look poses of a wrist camera are written in the Python program as joints in degrees
(``JointPositions.deg``), handed to each pick, one or a list tried in order until something is found, home from the
config when none is given; a fixed camera does not move. They replace the viewing pose the software generated, and
nothing is said to the hand before a look: no release, no pulse.

The service here is the real one (``AutonomousGraspService.from_components``) on a desk arm that records every call,
a hand that records every command, and a camera scripted per look, so the order is read off one shared log.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.geometry import Frame, Transform
from src.robot.core import JointPositions, MotionCommand, MotionResult, MotionStatus
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome, AutonomousGraspService, GraspMode
from src.robot.execution.pick_run import PickRun, Recording
from src.robot.grasping import DefaultPreGraspRefiner, GraspVerificationPolicy, NoOpVerifier, RefinementPolicy
from src.robot.grasping.decision import DecisionEngine, DecisionPolicy
from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver, StaticCameraToBaseResolver
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.perception import PerceptionFrame

#: Two looks the way a program writes them, in degrees.
LOOK_A = JointPositions.deg(-90.0, -100.0, -110.0, -60.0, 90.0, 0.0)
LOOK_B = JointPositions.deg(-70.0, -100.0, -110.0, -60.0, 90.0, 0.0)
#: Where the scripted part is grasped, BASE mm, approached straight down and closed along base -Y.
GRASP_MM = (-130.0, -600.0, 80.0)


class _Arm(DummyRobotArm):
    """A desk arm (the route runs, nothing real moves) that writes every motion onto the shared log."""

    def __init__(self, events: list[Any], *, refuse_joints: bool = False) -> None:
        super().__init__()
        self.events = events
        self.refuse_joints = refuse_joints
        self.connect()

    def move_to_joints(self, joints: JointPositions, **keywords: Any) -> MotionResult:
        self.events.append(("joints", joints.degrees()))
        if self.refuse_joints:
            return MotionResult.failed(MotionStatus.WORKSPACE_REJECTED, MotionCommand.MOVE_JOINTS,
                                       target_joints=joints, message="outside the cable window")
        return super().move_to_joints(joints, **keywords)

    def move_home(self) -> bool:
        self.events.append("home")
        return super().move_home()

    def move(self, pose: Any, **keywords: Any) -> MotionResult:
        self.events.append(("move", tuple(round(float(v), 1) for v in pose.position_mm), bool(keywords.get("linear"))))
        return super().move(pose, **keywords)


class _Hand:
    """A jaw that measures nothing and writes every command onto the shared log."""

    min_width_mm = 5.0
    max_width_mm = 50.0
    is_connected = True

    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self.width = self.max_width_mm

    def set_width_mm(self, width_mm: float, speed: Any = None, force: Any = None) -> None:
        self.events.append(("hand", round(float(width_mm), 1)))
        self.width = float(width_mm)

    def get_width_mm(self) -> float:
        return self.width


class _Camera:
    """One object in view or none, scripted per frame; every frame taken is written onto the shared log.

    ``True`` is the part (label ``part``), a text is an object of that label, ``False`` is an empty view.
    """

    def __init__(self, events: list[Any], sees: list[bool | str]) -> None:
        self.events = events
        self.sees = list(sees)

    def acquire(self) -> PerceptionFrame:
        seen = self.sees.pop(0) if len(self.sees) > 1 else self.sees[0]
        self.events.append("perceive")
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[11:22, 11:22] = 1  # 11 x 11 pixels round the principal point
        label = "part" if seen is True else str(seen)
        return PerceptionFrame(depth_map=np.full((32, 32), 500.0),
                               intrinsics=np.array([[400.0, 0.0, 16.0], [0.0, 400.0, 16.0], [0.0, 0.0, 1.0]]),
                               segmentations=(_Segment(mask, label),) if seen else ())


class _Segment:
    def __init__(self, mask: np.ndarray, label: str = "part") -> None:
        self.mask = mask
        self.label = label


class _Calculator:
    def compute_result(self, *_args: object, **_kwargs: object) -> GraspResult:
        grasp = GraspPoint(position=np.array(GRASP_MM), approach=np.array([0.0, 0.0, -1.0]),
                           axis=np.array([0.0, -1.0, 0.0]), grip_width_mm=40.0, score=0.9, frame=GraspFrame.BASE,
                           label="part")
        return GraspResult(candidates=(grasp,), reasons=(), top_score=0.9)


class _CalculatorThatFindsNoGrasp:
    """Sees the object and grasps none of it: every candidate collided."""

    def compute_result(self, *_args: object, **_kwargs: object) -> GraspResult:
        return GraspResult(candidates=(), reasons=(GraspFailureReason.ALL_COLLIDED,), top_score=0.0)


def camera_looking_down() -> Transform:
    """A fixed camera 600 mm above base (100, -500): its +Z points down, so a part 500 mm away lies at z = 100."""
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag([1.0, -1.0, -1.0])
    matrix[:3, 3] = (100.0, -500.0, 600.0)
    return Transform.from_matrix(matrix, from_frame=Frame.CAMERA, to_frame=Frame.BASE)


def cell(events: list[Any], sees: list[bool | str], *, wrist: bool, refuse_joints: bool = False,
         mode: GraspMode = GraspMode.EASY, calculator: Any = None, **wiring: Any) -> AutonomousGraspService:
    """The pick service on the recording arm, hand and camera; a wrist camera's frames are placed by the tool.

    ``wiring`` goes to ``from_components`` as it is: a decision layer, a refiner, a verifier.
    """
    resolver: Any = (EyeInHandFrameResolver(t_cam_to_tool=Transform.from_matrix(
        np.eye(4), from_frame=Frame.CAMERA, to_frame=Frame.TOOL)) if wrist
        else StaticCameraToBaseResolver(transform=camera_looking_down()))
    return AutonomousGraspService.from_components(
        arm=_Arm(events, refuse_joints=refuse_joints),
        calculator=calculator or _Calculator(),  # type: ignore[arg-type]
        perception=_Camera(events, sees),
        mode=mode,
        gripper=_Hand(events),  # type: ignore[arg-type]
        frame_resolver=resolver,
        max_attempts=1,
        **wiring,
    )


def deciding(policy: DecisionPolicy | None = None) -> dict[str, Any]:
    """The decision layer switched on: it ranks the frame and decides before the pick loop runs."""
    policy = policy or DecisionPolicy(enabled=True)
    return {"decision_policy": policy, "decision_engine": DecisionEngine(policy=policy)}


def closed_loop() -> dict[str, Any]:
    """The two-scan refine and a verifier (one that passes), what ``GraspMode.CLOSED_LOOP`` asks for."""
    refinement = RefinementPolicy(enabled=True)
    return {"refinement_policy": refinement, "refiner": DefaultPreGraspRefiner(policy=refinement),
            "verification_policy": GraspVerificationPolicy(enabled=True), "verifier": NoOpVerifier()}


def _looks(events: list[Any]) -> list[Any]:
    return [e for e in events if e == "home" or (isinstance(e, tuple) and e[0] == "joints")]


def _first(events: list[Any], kind: str) -> int:
    return next(i for i, e in enumerate(events) if e == kind or (isinstance(e, tuple) and e[0] == kind))


class TheLooksAreTriedInOrderTests(unittest.TestCase):
    def test_a_look_that_finds_nothing_is_followed_by_the_next_and_the_part_is_picked_there(self) -> None:
        events: list[Any] = []
        report = cell(events, [False, True], wrist=True).pick(look=[LOOK_A, LOOK_B])

        self.assertIs(AutonomousGraspOutcome.SUCCEEDED, report.outcome, report.failure_summary())
        self.assertEqual([("joints", LOOK_A.degrees()), "perceive", ("joints", LOOK_B.degrees()), "perceive"],
                         events[:4])
        self.assertEqual(("(-90.0, -100.0, -110.0, -60.0, 90.0, 0.0) deg",
                          "(-70.0, -100.0, -110.0, -60.0, 90.0, 0.0) deg"), report.looks)
        self.assertIn("looks", report.render())

    def test_the_first_look_that_finds_something_ends_the_looking(self) -> None:
        events: list[Any] = []
        report = cell(events, [True], wrist=True).pick(look=[LOOK_A, LOOK_B])

        self.assertTrue(report.succeeded)
        self.assertEqual([("joints", LOOK_A.degrees())], _looks(events))
        self.assertEqual(1, len(report.looks))

    def test_nothing_found_from_every_look_is_reported_with_the_looks_tried(self) -> None:
        events: list[Any] = []
        report = cell(events, [False], wrist=True).pick(look=(LOOK_A, "home", LOOK_B))

        self.assertIs(AutonomousGraspOutcome.NO_TARGET, report.outcome)
        self.assertEqual([("joints", LOOK_A.degrees()), "home", ("joints", LOOK_B.degrees())], _looks(events))
        self.assertEqual(3, events.count("perceive"))
        self.assertEqual("home", report.looks[1])
        self.assertIn("looked from 3", report.failure_summary())
        self.assertEqual(3, len(report.to_dict()["looks"]))
        self.assertFalse(any(isinstance(e, tuple) and e[0] == "hand" for e in events), "the hand was told something")

    def test_nothing_is_said_to_the_hand_before_or_between_the_looks(self) -> None:
        """No release before a look, no pulse: the first hand command is the pick's own, after the last look."""
        events: list[Any] = []
        cell(events, [False, False, True], wrist=True).pick(look=[LOOK_A, LOOK_B, "home"])

        last_look = max(i for i, e in enumerate(events) if e in _looks(events))
        self.assertGreater(_first(events, "hand"), last_look)
        self.assertGreater(_first(events, "hand"), max(i for i, e in enumerate(events) if e == "perceive"))

    def test_a_look_the_arm_does_not_reach_ends_the_attempt_with_nothing_perceived(self) -> None:
        events: list[Any] = []
        report = cell(events, [True], wrist=True, refuse_joints=True).pick(look=[LOOK_A, LOOK_B])

        self.assertIs(AutonomousGraspOutcome.EXECUTION_FAILED, report.outcome)
        self.assertEqual([("joints", LOOK_A.degrees())], events, "anything after the refused look was commanded")
        self.assertIn("outside the cable window", report.failure_summary())
        self.assertIn("not reached", report.failure_summary())
        self.assertEqual(1, len(report.looks))

    def test_a_pick_handed_no_look_perceives_where_the_arm_stands(self) -> None:
        events: list[Any] = []
        report = cell(events, [True], wrist=True).pick()

        self.assertEqual("perceive", events[0])
        self.assertEqual((), report.looks)


class EveryPickPathSendsAnEmptyLookOnTests(unittest.TestCase):
    """The decision layer and the two-scan refine answer an empty view in their own words, not the pick loop's.

    An empty frame reaches the decision layer as ``RECOVER`` for want of candidates, and a prompted label the
    two-scan path did not find as ``NO_VALID_GRASP`` before anything moved; neither carries the pick loop's trail.
    Both still found nothing where they looked, and are sent on to the next look. An object seen and not grasped is
    not: it is answered from where it was seen, as on the pick loop's path.
    """

    def test_the_decision_layer_sends_an_empty_look_on_to_the_next(self) -> None:
        events: list[Any] = []
        report = cell(events, [False, True], wrist=True, mode=GraspMode.AUTO, **deciding()).pick(look=[LOOK_A, LOOK_B])

        self.assertEqual([("joints", LOOK_A.degrees()), "perceive", ("joints", LOOK_B.degrees()), "perceive"],
                         events[:4])
        self.assertIs(AutonomousGraspOutcome.SUCCEEDED, report.outcome, report.failure_summary())
        self.assertEqual(2, len(report.looks))

    def test_the_decision_layer_sends_a_look_without_the_prompted_label_on_to_the_next(self) -> None:
        events: list[Any] = []
        service = cell(events, ["bolt", True], wrist=True, mode=GraspMode.AUTO, **deciding())
        service.set_target_label("part")

        report = service.pick(look=[LOOK_A, LOOK_B])

        self.assertEqual([("joints", LOOK_A.degrees()), ("joints", LOOK_B.degrees())], _looks(events))
        self.assertIs(AutonomousGraspOutcome.SUCCEEDED, report.outcome, report.failure_summary())

    def test_the_two_scan_refine_sends_a_look_without_the_prompted_label_on_to_the_next(self) -> None:
        events: list[Any] = []
        service = cell(events, ["bolt", True], wrist=True, mode=GraspMode.CLOSED_LOOP, **closed_loop())
        service.set_target_label("part")

        report = service.pick(look=[LOOK_A, LOOK_B])

        self.assertEqual([("joints", LOOK_A.degrees()), "perceive", ("joints", LOOK_B.degrees()), "perceive"],
                         events[:4])
        self.assertEqual([("joints", LOOK_A.degrees()), ("joints", LOOK_B.degrees())], _looks(events))
        self.assertIs(AutonomousGraspOutcome.SUCCEEDED, report.outcome, report.failure_summary())

    def test_what_the_decision_layer_saw_is_in_its_report(self) -> None:
        """The frame's facts the next look is decided on: how many objects it held, and why none was grasped."""
        events: list[Any] = []
        report = cell(events, [True], wrist=True, mode=GraspMode.AUTO, calculator=_CalculatorThatFindsNoGrasp(),
                      **deciding()).pick()

        self.assertIs(AutonomousGraspOutcome.DECISION_RECOVER_PENDING, report.outcome)
        self.assertEqual(1, report.telemetry["n_segmentations"])
        self.assertEqual(("all_collided",), report.telemetry["reasons"])

    def test_an_object_seen_and_not_grasped_by_the_decision_layer_ends_the_looking_where_it_was_seen(self) -> None:
        events: list[Any] = []
        report = cell(events, [True], wrist=True, mode=GraspMode.AUTO, calculator=_CalculatorThatFindsNoGrasp(),
                      **deciding()).pick(look=[LOOK_A, LOOK_B])

        self.assertIs(AutonomousGraspOutcome.DECISION_RECOVER_PENDING, report.outcome)
        self.assertEqual([("joints", LOOK_A.degrees())], _looks(events))
        self.assertEqual(1, len(report.looks))

    def test_an_object_seen_and_not_grasped_by_the_two_scan_refine_ends_the_looking_where_it_was_seen(self) -> None:
        events: list[Any] = []
        service = cell(events, [True], wrist=True, mode=GraspMode.CLOSED_LOOP,
                       calculator=_CalculatorThatFindsNoGrasp(), **closed_loop())
        service.set_target_label("part")

        report = service.pick(look=[LOOK_A, LOOK_B])

        self.assertIs(AutonomousGraspOutcome.NO_VALID_GRASP, report.outcome)
        self.assertEqual([("joints", LOOK_A.degrees())], _looks(events))
        self.assertFalse(any(isinstance(e, tuple) and e[0] == "move" for e in events), "the arm left its look")


class AStopIsHonouredBeforeEveryLookTests(unittest.TestCase):
    """A stop asked for while a look perceives ends the pick there: the arm is sent to no further look."""

    def test_a_stop_asked_for_while_the_first_look_perceives_sends_the_arm_to_no_other_look(self) -> None:
        events: list[Any] = []
        service = cell(events, [False, True], wrist=True)
        service.set_cancel_check(lambda: "perceive" in events)

        report = service.pick(look=[LOOK_A, LOOK_B, "home"])

        self.assertIs(AutonomousGraspOutcome.CANCELLED, report.outcome)
        self.assertEqual([("joints", LOOK_A.degrees()), "perceive"], events)
        self.assertEqual(("(-90.0, -100.0, -110.0, -60.0, 90.0, 0.0) deg",), report.looks)
        self.assertFalse(report.telemetry["cancelled_before_start"])

    def test_a_stop_asked_for_after_the_pick_began_and_before_the_first_look_moves_nothing(self) -> None:
        events: list[Any] = []
        asked: list[bool] = []
        service = cell(events, [True], wrist=True)
        # Not yet at the check that opens the pick, then from the next question on.
        service.set_cancel_check(lambda: bool(asked.append(True)) or len(asked) > 1)

        report = service.pick(look=[LOOK_A, LOOK_B])

        self.assertIs(AutonomousGraspOutcome.CANCELLED, report.outcome)
        self.assertEqual([], events)
        self.assertEqual((), report.looks)
        self.assertTrue(report.telemetry["cancelled_before_start"])


class ACampaignLooksWhereItsCameraNeedsTests(unittest.TestCase):
    def test_a_wrist_camera_given_no_look_looks_from_home_before_every_pick(self) -> None:
        events: list[Any] = []
        service = cell(events, [True], wrist=True)
        self.assertTrue(service.perceives_from_the_wrist)

        run = PickRun.from_service(service, runs=2, recording=Recording.off()).execute()

        self.assertEqual(2, run.succeeded)
        self.assertEqual(["home", "home"], _looks(events))
        self.assertEqual("home", events[0])
        self.assertEqual(("home",), run.attempts[0].looks)

    def test_a_fixed_camera_does_not_move_to_look(self) -> None:
        events: list[Any] = []
        service = cell(events, [True], wrist=False)
        self.assertFalse(service.perceives_from_the_wrist)

        run = PickRun.from_service(service, runs=2, recording=Recording.off()).execute()

        self.assertEqual(2, run.succeeded)
        self.assertEqual([], _looks(events))
        self.assertEqual("perceive", events[0])
        self.assertEqual((), run.attempts[0].looks)

    def test_the_programs_looks_are_handed_to_every_pick(self) -> None:
        events: list[Any] = []
        run = PickRun.from_service(cell(events, [False, True], wrist=True), runs=1, recording=Recording.off(),
                                   look=[LOOK_A, LOOK_B]).execute()

        self.assertEqual(1, run.succeeded)
        self.assertEqual(2, len(run.attempts[0].looks))
        self.assertIn("looked from", run.attempts[0].render())

    def test_a_look_list_that_names_nothing_or_no_unit_is_refused_before_any_cell(self) -> None:
        with self.assertRaises(ValueError):
            PickRun.from_service(object(), runs=1, recording=Recording.off(), look=[])
        with self.assertRaises(TypeError):  # radians or degrees? a raw list does not say
            PickRun.from_service(object(), runs=1, recording=Recording.off(), look=[[0.0, -1.57, 1.57, 0, 0, 0]])
        with self.assertRaises(ValueError):
            PickRun.from_service(object(), runs=1, recording=Recording.off(), look="house")

    def test_a_service_that_takes_no_look_is_called_as_it_always_was(self) -> None:
        class _Plain:
            def pick(self) -> Any:
                return type("R", (), {"outcome": "succeeded", "fault": None})()

        self.assertEqual(1, PickRun.from_service(_Plain(), runs=1, recording=Recording.off()).execute().succeeded)


class JointsInDegreesTests(unittest.TestCase):
    def test_deg_takes_the_pendants_degrees_and_stores_radians(self) -> None:
        joints = JointPositions.deg(0.0, -90.0, 180.0)

        np.testing.assert_allclose(joints.values, [0.0, -np.pi / 2, np.pi])
        self.assertEqual((0.0, -90.0, 180.0), joints.degrees())
        self.assertEqual(JointPositions([0.0, -np.pi / 2, np.pi]), joints)

    def test_deg_refuses_what_a_joint_vector_refuses(self) -> None:
        with self.assertRaises(ValueError):
            JointPositions.deg()
        with self.assertRaises(ValueError):
            JointPositions.deg(0.0, float("nan"))


if __name__ == "__main__":
    unittest.main()
