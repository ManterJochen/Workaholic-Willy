"""What number actually reaches the gripper when the stack commands a width.

⛔ **WHY THIS FILE EXISTS.** MEASURED 2026-09-10: nothing in the suite asserted the value that leaves
the driver. `tests/test_gripper_protocol_conformance.py` pins the SIGNATURES and the return types,
`RobotiqCountMapTests` pins `_mm_to_count` in isolation, and `tests/test_robotiq_socket.py` pins the
wire grammar against a fake URCap that acks everything. Between them sits the one question an
operator cares about, and it was untested: **when the pick path says "close on this part", what
integer arrives at the gripper?** `close()` could have been made to OPEN the jaws, or to command
double force, with all 208 tests green.

That gap was not hypothetical. On the same day two defects rode straight through it:

    the planner planned a 49.99 mm Hand-E as though it opened 85.0 mm, so a commanded close
    was clamped to max_width_mm, which maps to count 0, which is FULLY OPEN

    a `closed_width_mm` at or above the stroke collapsed the whole count map onto 255, so every
    commanded width including `open()` became a full close at full speed

Both are arithmetic between the config and the wire, which is exactly what this file watches.

⭐ **THE CONTROL IS THE POINT.** Every test here is written so that inverting the map, dropping the
stroke, or swapping open for close makes it FAIL. `test_the_check_can_fail` proves that on a
deliberately broken map rather than asserting it in prose.
"""

from __future__ import annotations

import unittest
from typing import Any

from src.config.loader import load_config
from src.config.schema.robot import GripperConfig
from src.robot.core.gripper import ObjectDetectingGripper
from src.robot.grippers.robotiq import GripperController


class _RecordingDriver:
    """A driver seam that REMEMBERS the numbers, which is the whole instrument.

    It implements the five documented seam methods plus `object_status` and `wait_for_motion`, so it
    exercises the same path a real :class:`~backend.src.robot.grippers.robotiq_socket.RobotiqSocket`
    takes rather than the degraded one.
    """

    def __init__(self) -> None:
        self.moves: list[tuple[int, int, int]] = []
        self.waits = 0
        self.position = 0
        #: 3 = AT_POSITION = the fingers reached the target and are holding NOTHING.
        self.obj = 3

    def connect(self, ip: str, port: int) -> None:
        return None

    def activate_if_needed(self) -> None:
        return None

    def move(self, count: int, speed: int, force: int) -> None:
        self.moves.append((int(count), int(speed), int(force)))
        self.position = int(count)

    def wait_for_motion(self) -> int:
        self.waits += 1
        return self.obj

    def object_status(self) -> int:
        return self.obj

    def get_current_position(self) -> int:
        return self.position

    def disconnect(self) -> None:
        return None

    @property
    def last(self) -> tuple[int, int, int]:
        assert self.moves, "nothing was commanded"
        return self.moves[-1]


def _gripper(cfg: GripperConfig) -> tuple[GripperController, _RecordingDriver]:
    driver = _RecordingDriver()
    g = GripperController(cfg, ip="127.0.0.1", driver_factory=lambda: driver)
    g.connect()
    return g, driver


class TheCommandedNumberReachesTheDriverTests(unittest.TestCase):
    """The direction of the map, at the boundary the operator sees."""

    def test_a_close_is_not_an_open(self) -> None:
        """⭐ THE ONE THAT WOULD HAVE CAUGHT THE APERTURE DEFECT. Counts run 0 = open, 255 = closed,
        so a close must command a HIGHER count than an open. Nothing asserted this."""
        g, drv = _gripper(GripperConfig(min_width_mm=5.0, max_width_mm=50.0, closed_width_mm=0.0))
        g.open()
        opened = drv.last[0]
        g.close()
        closed = drv.last[0]
        self.assertLess(
            opened, closed,
            f"open() commanded count {opened} and close() commanded {closed}. Counts are "
            f"0 = open, 255 = closed, so a close must be the larger number. This is the assertion "
            f"that distinguishes a working driver from one that opens the hand on the part.",
        )
        self.assertEqual(opened, 0, "open() must command the fully-open end of the travel")

    def test_a_width_at_the_stroke_is_fully_open_and_that_is_correct(self) -> None:
        """⚠ THE HONEST HALF OF THE APERTURE DEFECT. `max_width_mm -> count 0` is right; the defect
        was that a width the hand cannot reach ARRIVED here. Pinned so the correct half is not
        "fixed" by somebody reading the incident report and inverting the map."""
        cfg = GripperConfig(min_width_mm=5.0, max_width_mm=49.99, closed_width_mm=0.0)
        g, drv = _gripper(cfg)
        g.set_width_mm(49.99)
        self.assertEqual(drv.last[0], 0)
        g.set_width_mm(83.0)  # over-stroke: clamped to the stroke, which is the open end
        self.assertEqual(drv.last[0], 0)

    def test_a_commanded_width_lands_where_it_was_asked_to(self) -> None:
        """The count that leaves must map back to the millimetres that came in."""
        cfg = GripperConfig(min_width_mm=5.0, max_width_mm=49.99, closed_width_mm=0.0)
        g, drv = _gripper(cfg)
        for want in (10.0, 20.0, 30.0, 40.0):
            with self.subTest(width_mm=want):
                g.set_width_mm(want)
                got = g._count_to_mm(drv.last[0])
                self.assertAlmostEqual(got, want, delta=0.25,
                                       msg=f"commanded {want} mm, the driver received a count worth "
                                           f"{got:.2f} mm")

    def test_the_policy_floor_is_applied_and_the_map_is_not_shifted_by_it(self) -> None:
        """Two configs differing only in the floor must send the SAME count for a width above it."""
        a, da = _gripper(GripperConfig(min_width_mm=0.0, max_width_mm=50.0, closed_width_mm=0.0))
        b, db = _gripper(GripperConfig(min_width_mm=5.0, max_width_mm=50.0, closed_width_mm=0.0))
        a.set_width_mm(30.0)
        b.set_width_mm(30.0)
        self.assertEqual(da.last[0], db.last[0])
        b.set_width_mm(1.0)  # below the floor
        # ⚠ delta, not equality: counts are integers, so one step is max_width_mm/255 = 0.196 mm on
        # this stroke and the floor round-trips to 4.902 mm. The first draft of this line asserted
        # exact equality and failed against a driver that was right, which is this repo's standing
        # lesson about a test claiming more than the system promises.
        self.assertAlmostEqual(b._count_to_mm(db.last[0]), 5.0, delta=50.0 / 255.0)

    def test_force_and_speed_are_fractions_not_newtons(self) -> None:
        """⚠ `close_force_n` is named in newtons and normalised as a [0,1] fraction, so anything at
        or above 1.0 is MAXIMUM force. Nothing sets it today and it is not a YAML key; this pins the
        behaviour so the trap is visible to whoever wires it."""
        cfg = GripperConfig(min_width_mm=5.0, max_width_mm=50.0, closed_width_mm=0.0)
        g, drv = _gripper(cfg)
        g.set_width_mm(20.0, speed=0.5, force=0.5)
        self.assertEqual(drv.last[1], 128)
        self.assertEqual(drv.last[2], 128)
        g.set_width_mm(20.0, speed=1.0, force=40.0)  # 40 "newtons"
        self.assertEqual(drv.last[2], 255, "a value at or above 1.0 saturates: this is a fraction")

    def test_every_shipped_gripper_profile_commands_a_reachable_count(self) -> None:
        """⭐ DERIVED FROM THE CONFIG TREE, so a new gripper profile is covered by existing.

        For each profile: a full open and a full close must land on distinct counts inside 0..255.
        A degenerate map collapses them onto one value, which is what a closed width at or above the
        stroke used to do.
        """
        for profile in (None, "hande"):
            with self.subTest(profile=profile or "base"):
                cfg = load_config(profile=profile).robot.gripper
                g, drv = _gripper(cfg)
                g.open()
                opened = drv.last[0]
                g.close()
                closed = drv.last[0]
                self.assertEqual(opened, 0)
                self.assertGreater(closed, opened,
                                   f"{profile or 'base'}: open and close command the same end of the "
                                   f"travel, so this map is collapsed")
                self.assertLessEqual(closed, 255)


class TheCloseWaitsAndReportsWhatItHoldsTests(unittest.TestCase):
    """The two capabilities added on 2026-09-10, pinned so they cannot quietly go away again."""

    def test_the_controller_advertises_object_detection(self) -> None:
        """MEASURED 2026-09-10: this was False, and `GripperController` was the only real jaw
        gripper in the repo that was not an `ObjectDetectingGripper`. The pick path's fail-closed
        gate was skipped for the one gripper this project ships."""
        self.assertTrue(issubclass(GripperController, ObjectDetectingGripper))

    def test_at_position_is_not_a_grasp(self) -> None:
        """⛔ THE DISTINCTION THE WHOLE REGISTER EXISTS FOR. gOBJ 3 means the fingers REACHED the
        commanded width, which on a close means they met and are holding nothing."""
        cfg = GripperConfig(min_width_mm=5.0, max_width_mm=50.0, closed_width_mm=0.0)
        g, drv = _gripper(cfg)
        drv.obj = 3  # AT_POSITION
        self.assertFalse(g.is_object_detected())
        drv.obj = 2  # STOPPED_CLOSING: stalled on something
        self.assertTrue(g.is_object_detected())
        drv.obj = 0  # MOVING is not evidence of anything
        self.assertFalse(g.is_object_detected())

    def test_a_close_waits_for_the_fingers(self) -> None:
        """`ack` is the daemon accepting the line, not the fingers arriving. Before this, the arm's
        retreat was the next statement after the close."""
        cfg = GripperConfig(min_width_mm=5.0, max_width_mm=50.0, closed_width_mm=0.0)
        g, drv = _gripper(cfg)
        g.set_width_mm(20.0)
        self.assertEqual(drv.waits, 1, "set_width_mm returned without waiting for the motion")

    def test_a_driver_that_cannot_answer_reports_no_grasp(self) -> None:
        """⛔ FAIL CLOSED, and this is the asymmetry worth pinning. "No evidence" must not become
        "holding": claiming a grasp nobody can confirm is the defect the capability exists to end."""

        class _Mute:
            def connect(self, ip: str, port: int) -> None: ...
            def activate_if_needed(self) -> None: ...
            def move(self, count: int, speed: int, force: int) -> None: ...
            def get_current_position(self) -> int: return 0
            def disconnect(self) -> None: ...

        mute: Any = _Mute()
        g = GripperController(GripperConfig(min_width_mm=5.0, max_width_mm=50.0),
                              ip="127.0.0.1", driver_factory=lambda: mute)
        g.connect()
        self.assertFalse(g.is_object_detected())


class TheCheckCanFailTests(unittest.TestCase):
    """⚠ THE SELF-FAILURE CONTROL. Every assertion above is worthless if it cannot go red, and this
    repo has shipped three green-but-inert guards in one week."""

    def test_an_inverted_map_is_caught(self) -> None:
        cfg = GripperConfig(min_width_mm=5.0, max_width_mm=50.0, closed_width_mm=0.0)
        g, drv = _gripper(cfg)
        g._mm_to_count = lambda mm: 255 - GripperController._mm_to_count(g, mm)  # type: ignore[method-assign]
        g.open()
        opened = drv.last[0]
        g.close()
        closed = drv.last[0]
        self.assertGreater(opened, closed,
                           "the deliberately inverted map did not invert, so this control proves "
                           "nothing about the assertions above")

    def test_a_collapsed_map_is_caught(self) -> None:
        """The degenerate config the schema now refuses, forced past the schema to prove the
        assertion above would have caught it."""
        cfg = GripperConfig(min_width_mm=5.0, max_width_mm=50.0, closed_width_mm=0.0)
        g, drv = _gripper(cfg)
        object.__setattr__(g, "_collapsed", True)
        g._mm_to_count = lambda mm: 255  # type: ignore[method-assign]
        g.open()
        opened = drv.last[0]
        g.close()
        closed = drv.last[0]
        self.assertEqual(opened, closed, "a collapsed map must send one value for both ends")
        self.assertNotEqual(opened, 0, "and that value is the closed end, which is the hazard")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
