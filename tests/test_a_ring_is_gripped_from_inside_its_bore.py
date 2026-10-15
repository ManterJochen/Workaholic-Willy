"""A ring is gripped from inside its bore where no grasp holds it from outside (the owner, 2026-10-14).

The owner's "Nur wo außen nichts geht", "Wie beim Schließen" and "Deine Ringe mit großer Bohrung": a Robotiq hand comes
down narrow into a bore the camera saw free, opens at the close's force until the wall stops it, and closes to let the
part go. Pinned on the owner's Hand-E (the shipped ``hande`` profile, 49.99 mm stroke, 10.8 mm fingers 29.24 mm wide), on
synthetic parts read as a camera straight above reads them:

* the bores: a ring 90 mm across with a 60 mm bore holds one, about its centre and read up to 8 mm narrow; a cube none;
* the grasps: through the bore's centre, coming down 7 mm inside the wall each side, opening onto it within the stroke;
  none where the bore is narrower than the closed fingers and the room, none where no ray saw the fingers' way, none
  where a neighbour lies in the bore;
* the stage and the ranking: every grasp from inside after every outer grasp, its margins saying so; off, nothing of it;
* the switch: on as shipped, for a Robotiq hand alone;
* the execution: narrow on the way in, the guard told the wall, the open at the close's force and speed, the part
  attached at its own width; a wall nobody met is no hold; a hand that cannot say how it holds takes none;
* the release: a hand that gripped from inside closes, and one whose release opened onto something opens again.
"""

from __future__ import annotations

import unittest
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.robot.grasping.generation.body_grasps import body_least_height_mm, body_map
from src.robot.grasping.generation.inner_grasps import bores, inner_grasps
from src.robot.grasping.generation.support_footprint import NARROW_ROOM_MM
from src.robot.grasping.types.grasp_point import (
    INNER_GRASP_KEY,
    GraspFrame,
    GraspPoint,
    fingers_held_at,
    grip_from_inside_of,
    outer_first,
)
from tests.test_the_hand_grasps_the_parts_own_body import _cube, _jaw, _ring, _surface

_STROKE_MM = 49.99


def _seen(points: np.ndarray) -> np.ndarray:
    return np.ones(points.shape[0], dtype=bool)


@lru_cache(maxsize=None)
def _ring_cloud(outer: float, bore: float) -> np.ndarray:
    cloud = _surface(_ring(outer, bore), outer / 2.0 + 5.0, n=25000)
    cloud.flags.writeable = False
    return cloud


class TheBoresOfAPart(unittest.TestCase):
    def test_a_ring_holds_one_bore_about_its_centre_and_a_cube_none(self) -> None:
        jaw = _jaw()
        body = body_map(_ring_cloud(90.0, 60.0), 0.0, floor_margin_mm=2.0)
        assert body is not None
        found = bores(body, deeper_than_mm=body_least_height_mm(jaw))
        self.assertEqual(1, len(found))
        self.assertLess(float(np.linalg.norm(np.asarray(found[0].centre_mm)[:2])), 1.5)
        self.assertGreaterEqual(found[0].diameter_mm, 52.0, "read at most 8 mm narrow")
        self.assertLessEqual(found[0].diameter_mm, 60.0, "never wider than it is")
        cube = body_map(_surface(_cube, 20.0), 0.0, floor_margin_mm=2.0)
        assert cube is not None
        self.assertEqual([], bores(cube, deeper_than_mm=body_least_height_mm(jaw)))


class TheGraspsFromInside(unittest.TestCase):
    def test_through_the_centre_narrow_on_the_way_in_onto_the_wall_within_the_stroke(self) -> None:
        jaw = _jaw()
        for outer, bore in ((70.0, 50.0), (90.0, 60.0)):
            with self.subTest(ring=(outer, bore)):
                found = inner_grasps(_ring_cloud(outer, bore), support_height_mm=0.0, jaw=jaw, corridor_seen=_seen)
                self.assertTrue(found)
                for grasp in found:
                    candidate = grasp.candidate
                    np.testing.assert_allclose(candidate.position_mm[:2], [0.0, 0.0], atol=2.0)
                    np.testing.assert_allclose(candidate.approach, [0.0, 0.0, -1.0])
                    self.assertEqual(grasp.line_in_width_mm, candidate.open_width_mm)
                    self.assertAlmostEqual(candidate.grip_width_mm - grasp.line_in_width_mm, 2.0 * NARROW_ROOM_MM,
                                           places=6, msg="7 mm of room beside each jaw on the way in")
                    self.assertLessEqual(candidate.grip_width_mm, jaw.aperture_mm - jaw.width_safety_mm)
                    self.assertAlmostEqual(grasp.bore_mm - 2.0 * jaw.finger_thickness_mm, candidate.grip_width_mm,
                                           places=6)
                    self.assertGreater(grasp.part_width_mm, outer - 5.0, "carried as wide as the ring")
                    self.assertGreaterEqual(grasp.depth_mm, 10.0, "the pads 10 mm under the top or more")
                    self.assertGreaterEqual(candidate.clearance_mm, jaw.table_clearance_mm - 1e-6)

    def test_none_where_the_bore_is_too_narrow_unseen_or_holds_a_neighbour(self) -> None:
        jaw = _jaw()
        ring = _ring_cloud(90.0, 60.0)
        rng = np.random.default_rng(3)
        pebble = np.column_stack([rng.uniform(-6, 6, 300), rng.uniform(15, 25, 300), rng.uniform(0, 8, 300)])
        cases: tuple[tuple[str, dict[str, Any], str], ...] = (
            ("a 34 mm bore", {"cloud_base_mm": _ring_cloud(60.0, 34.0), "corridor_seen": _seen}, "aperture"),
            ("no seen test", {"cloud_base_mm": ring, "corridor_seen": None}, "unseen_corridor"),
            ("the way seen only above the ring", {"cloud_base_mm": ring, "corridor_seen": lambda p: p[:, 2] > 20.0},
             "unseen_corridor"),
            ("a neighbour in the bore", {"cloud_base_mm": ring, "corridor_seen": _seen,
                                         "obstacle_points_base_mm": pebble}, "seen_corridor"),
        )
        for label, kwargs, cause in cases:
            with self.subTest(label):
                refusals: dict[str, int] = {}
                self.assertEqual([], inner_grasps(support_height_mm=0.0, jaw=jaw, refusals=refusals, **kwargs))
                self.assertGreater(refusals.get(cause, 0), 0, refusals)
        self.assertEqual([], inner_grasps(_surface(_cube, 20.0), support_height_mm=0.0, jaw=jaw, corridor_seen=_seen))


class EveryGraspFromInsideRanksLast(unittest.TestCase):
    def _stage(self, inner: bool) -> tuple[list[Any], dict]:
        from src.robot.grasping.generation._support_footprint_stage import support_footprint_breakdowns

        return support_footprint_breakdowns(
            _ring_cloud(70.0, 50.0), camera_to_base=np.eye(4), support_height_mm=0.0, jaw=_jaw(),
            obstacle_points_base_mm=None, max_candidates=12, corridor_seen=_seen, opens_as_needed=True,
            inner_grasps=inner)

    def test_the_stage_offers_them_after_the_body_and_off_offers_nothing_of_them(self) -> None:
        breakdowns, telemetry = self._stage(True)
        inside = [b.components["support_footprint"].get(INNER_GRASP_KEY) is True for b in breakdowns]
        self.assertIn(True, inside)
        self.assertIn(False, inside)
        self.assertEqual(sorted(inside), inside, "every grasp from inside after every outer one")
        self.assertEqual(telemetry["inner_grasps"], inside.count(True))
        for breakdown in breakdowns[inside.index(True):]:
            margins = breakdown.components["support_footprint"]
            self.assertEqual("inner", margins["stage"])
            self.assertEqual(margins["open_width_mm"], margins["line_in_width_mm"])
            self.assertEqual(fingers_held_at(breakdown), margins["grip_width_mm"])
            self.assertIsNotNone(grip_from_inside_of(breakdown))
        off, off_telemetry = self._stage(False)
        self.assertEqual(inside.count(False), len(off))
        for kept, alone in zip(breakdowns, off):
            np.testing.assert_array_equal(kept.pose.position_mm, alone.pose.position_mm)
        self.assertFalse([key for key in off_telemetry if key.startswith("inner")])

    def test_the_calculator_ranks_an_outer_grasp_before_a_better_one_from_inside(self) -> None:
        def grasp(score: float, inside: bool) -> GraspPoint:
            margins = {INNER_GRASP_KEY: True} if inside else {"stage": "body"}
            return GraspPoint(position=np.zeros(3), approach=np.array([0.0, 0.0, -1.0]), axis=np.array([1.0, 0.0, 0.0]),
                              grip_width_mm=20.0, score=score, frame=GraspFrame.BASE, label=f"{score}{inside}",
                              metadata={"score_components": {"support_footprint": margins}})

        ranked = sorted([grasp(0.9, True), grasp(0.5, False), grasp(0.95, True), grasp(0.7, False)], key=outer_first)
        self.assertEqual(["0.7False", "0.5False", "0.95True", "0.9True"], [g.label for g in ranked])


class TheSwitch(unittest.TestCase):
    def test_on_as_shipped_for_a_robotiq_hand_alone(self) -> None:
        from src.config.schema.robot.grasping_schema import RobotGraspingConfig
        from src.robot.grasping.calculator_factory import inner_grasps as searched

        self.assertTrue(RobotGraspingConfig().inner_grasps)

        def cell(on: bool, vendor: str) -> Any:
            return SimpleNamespace(grasping=SimpleNamespace(inner_grasps=on), gripper=SimpleNamespace(vendor=vendor))

        self.assertTrue(searched(cell(True, "robotiq")))
        self.assertFalse(searched(cell(False, "robotiq")))
        for vendor in ("jaw_io", "onrobot", "vacuum", "dummy", "none", ""):
            self.assertFalse(searched(cell(True, vendor)), vendor)


class _BoreDriver:
    """A Robotiq seam whose fingers stop on a bore's wall while they open past it (gOBJ 1), reach every other target
    (gOBJ 3), and can be made to fail its status read."""

    def __init__(self, wall_count: int | None, *, start: int = 255) -> None:
        self.wall = wall_count
        self.position = start
        self.obj = 3
        self.moves: list[tuple[int, int, int]] = []
        self.status_raises = False

    def connect(self, ip: str, port: int) -> None:
        return None

    def activate_if_needed(self) -> None:
        return None

    def move(self, count: int, speed: int, force: int) -> None:
        self.moves.append((int(count), int(speed), int(force)))
        if self.wall is not None and count < self.wall <= self.position:
            self.position, self.obj = self.wall, 1
        else:
            self.position, self.obj = int(count), 3

    def wait_for_motion(self) -> int:
        return self.obj

    def object_status(self) -> int:
        if self.status_raises:
            raise OSError("the gripper's socket closed")
        return self.obj

    def get_current_position(self) -> int:
        return self.position

    def disconnect(self) -> None:
        return None


def _hand(wall_mm: float | None) -> tuple[Any, _BoreDriver]:
    from src.config.schema.robot import GripperConfig
    from src.robot.grippers.robotiq import GripperController

    holder: list[_BoreDriver] = []
    hand = GripperController(GripperConfig(min_width_mm=0.0, max_width_mm=_STROKE_MM, closed_width_mm=0.0),
                             ip="127.0.0.1", driver_factory=lambda: holder[0])
    holder.append(_BoreDriver(None if wall_mm is None else hand._mm_to_count(wall_mm)))  # noqa: SLF001
    hand.connect()
    return hand, holder[0]


class _Arm:
    """An arm that keeps no line and records what the policy tells it."""

    def __init__(self) -> None:
        self.moves: list[Any] = []
        self.told: list[float | None] = []
        self.attached: list[float] = []

    def move(self, pose: Any, *, linear: bool = False) -> Any:
        from src.robot.core import MotionCommand, MotionResult

        self.moves.append(pose)
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)

    def set_finger_opening(self, width: float | None) -> None:
        self.told.append(width)

    def attach_payload(self, grip_width_mm: float) -> bool:
        self.attached.append(float(grip_width_mm))
        return True

    def detach_payload(self) -> bool:
        return True

    def payload_declined_reason(self) -> str | None:
        return None

    def payload_model(self) -> Any:
        from src.robot.core.arm_capabilities import PayloadModel

        return PayloadModel.PLANNER_AND_FILTER


_LINE_IN, _CONTACT, _PART = 16.9, 30.9, 92.0


def _inner_grasp(**dropped: Any) -> GraspPoint:
    margins = {"stage": "inner", INNER_GRASP_KEY: True, "line_in_width_mm": _LINE_IN, "open_width_mm": _LINE_IN,
               "grip_width_mm": _CONTACT, "part_width_mm": _PART, "depth_mm": 20.0, "bore_mm": 52.5}
    for name in dropped:
        margins.pop(name)
    return GraspPoint(position=np.array([0.0, -650.0, 12.0]), approach=np.array([0.0, 0.0, -1.0]),
                      axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=_CONTACT, score=0.8, frame=GraspFrame.BASE,
                      label="inner", metadata={"score_components": {"support_footprint": margins}})


def _policy(arm: Any, hand: Any) -> Any:
    from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy

    return GraspExecutionPolicy(arm=arm, gripper=hand, pre_open_width_mm=_STROKE_MM, close_speed=0.2,
                                close_force_n=0.4)


class TheExecutionGripsFromInside(unittest.TestCase):
    def test_narrow_in_the_guard_at_the_wall_the_open_at_the_close_s_force_the_part_at_its_width(self) -> None:
        from src.robot.core.gripper import FINGER_OPENING_SLACK_MM
        from src.robot.grasping.motion.execution_policy import PolicyOutcome

        hand, driver = _hand(wall_mm=36.0)
        arm = _Arm()
        report = _policy(arm, hand).execute(_inner_grasp())

        self.assertIs(PolicyOutcome.EXECUTED, report.outcome, report.motion_message)
        self.assertIs(True, report.object_detected)
        self.assertEqual(hand._mm_to_count(_LINE_IN), driver.moves[0][0], "narrow on the way in")  # noqa: SLF001
        opened = driver.moves[-1]
        self.assertEqual(hand._mm_to_count(_STROKE_MM), opened[0], "opened as far as the hand goes")  # noqa: SLF001
        self.assertEqual((hand._normalise_to_count(0.2), hand._normalise_to_count(0.4)), opened[1:],  # noqa: SLF001
                         "at the close's speed and force")
        self.assertAlmostEqual(_CONTACT + FINGER_OPENING_SLACK_MM, arm.told[0], msg="the guard holds the wall")
        self.assertAlmostEqual(hand.get_width_mm() + FINGER_OPENING_SLACK_MM, arm.told[1],
                               msg="and where the hand reads them at the wall, wider than planned")
        self.assertGreater(hand.get_width_mm(), _CONTACT)
        self.assertIsNone(arm.told[-1], "fully open again once the pick's motions end")
        self.assertEqual([_PART], arm.attached)
        self.assertTrue(hand.holds_from_inside())

    def test_a_wall_nobody_met_is_no_hold(self) -> None:
        from src.robot.grasping.motion.execution_policy import PolicyOutcome

        hand, _driver = _hand(wall_mm=None)
        arm = _Arm()
        report = _policy(arm, hand).execute(_inner_grasp())
        self.assertIs(PolicyOutcome.OBJECT_NOT_DETECTED, report.outcome)
        self.assertEqual([], arm.attached)
        self.assertFalse(hand.holds_from_inside())

    def test_a_hand_that_cannot_say_how_it_holds_takes_none(self) -> None:
        from src.robot.grasping.motion.execution_policy import PolicyOutcome
        from tests.test_grasp_execution_policy import _DetectingGripper

        plain = _DetectingGripper(detects=True)
        robotiq, driver = _hand(wall_mm=36.0)
        for label, hand, grasp, sent in (
                ("no word of how it holds", plain, _inner_grasp(), lambda: plain.width_calls),
                ("a margin missing", robotiq, _inner_grasp(part_width_mm=None), lambda: driver.moves)):
            with self.subTest(label):
                arm = _Arm()
                policy = _policy(arm, hand)
                said = policy.refusal_ahead(grasp)
                self.assertIn("from inside a bore", said)
                report = policy.execute(grasp)
                self.assertIs(PolicyOutcome.GRIPPER_FAULT, report.outcome)
                self.assertEqual(said, report.motion_message)
                self.assertEqual(([], [], [], []), (arm.moves, arm.told, arm.attached, sent()),
                                 "nothing was commanded")


class TheReleaseLetsGo(unittest.TestCase):
    def _robot(self, hand: Any) -> Any:
        from src.robot.drivers.dummy.arm import DummyRobotArm
        from src.robot.execution.robot import Robot

        arm = DummyRobotArm()
        arm.connect()
        return Robot.from_parts(arm=arm, gripper=hand, lock_key=None)

    def test_a_hand_that_gripped_from_inside_closes(self) -> None:
        from src.robot.core.gripper import HoldEvidence
        from src.robot.execution.handling import HandOutcome

        hand, driver = _hand(wall_mm=36.0)
        hand.grip_from_inside(speed=0.2, force=0.4)
        report = self._robot(hand).release()
        self.assertIs(HandOutcome.RELEASED, report.outcome, report.render())
        self.assertEqual(hand._mm_to_count(0.0), driver.moves[-1][0], "closed as far as it goes")  # noqa: SLF001
        self.assertEqual(0.0, report.commanded_width_mm)
        self.assertIs(HoldEvidence.EMPTY, report.hold)
        self.assertFalse(hand.holds_from_inside())

    def test_a_release_that_opened_onto_something_opens_again(self) -> None:
        from src.robot.execution.handling import HandOutcome

        hand, driver = _hand(wall_mm=36.0)
        hand.grip_from_inside()
        hand.set_width_mm(0.0)
        hand.set_width_mm(_STROKE_MM)
        self.assertEqual(1, driver.obj, "the fingers stopped on something while opening")
        self.assertFalse(hand.holds_from_inside(), "a command since the grip from inside")
        report = self._robot(hand).release()
        self.assertIs(HandOutcome.RELEASE_NOT_CONFIRMED, report.outcome, report.render())
        self.assertEqual(hand._mm_to_count(_STROKE_MM), driver.moves[-1][0], "opened, not closed")  # noqa: SLF001

    def test_a_hand_that_cannot_say_how_it_holds_is_a_fault_with_nothing_commanded(self) -> None:
        from src.robot.execution.handling import HandOutcome

        hand, driver = _hand(wall_mm=36.0)
        hand.grip_from_inside()
        sent = len(driver.moves)
        driver.status_raises = True
        report = self._robot(hand).release()
        self.assertIs(HandOutcome.GRIPPER_FAULT, report.outcome, report.render())
        self.assertIsNone(report.commanded_width_mm)
        self.assertEqual(sent, len(driver.moves))


if __name__ == "__main__":
    unittest.main()
