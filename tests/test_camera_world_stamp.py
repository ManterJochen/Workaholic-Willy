"""Every motion result says whether a camera world stood behind the motion.

The owner decided on 2026-09-11 that with cuRobo every motion plans against a current camera image
unless the caller declines, and that a decline is explicit and visible. This file pins the carrier; what
each driver stamps is pinned in ``tests/test_camera_world_on_every_driver.py``. A result built without a
stamp says it is NOT vouched for; a decline cannot be written without a reason; and the stamp is part
of what makes two results the same result, because two motions that differ in what the planner knew
about the cell did not do the same thing.
"""

from __future__ import annotations

import json
import unittest

from src.robot.core import MotionCommand, MotionResult, MotionStatus
from src.robot.core.camera_world import CameraWorldDecline, CameraWorldStamp, CameraWorldUse


def _every_kind() -> tuple[CameraWorldStamp, ...]:
    return (
        CameraWorldStamp.unstated(),
        CameraWorldStamp.planned(cameras=("realsense_d435", "oblique_left"), captured_at_s=12.5),
        CameraWorldStamp.declined(CameraWorldDecline("calibration sweep, no CAMERA->BASE yet")),
        CameraWorldStamp.unplanned("robot.ur.motion_planner is 'ik'"),
        CameraWorldStamp.missing("cuRobo plans this motion and no live camera world is wired"),
    )


class TheUsesAreFiveTests(unittest.TestCase):
    def test_the_uses_and_their_wire_values(self) -> None:
        """MISSING is its own answer: a planner planned with no camera world and nobody declined,
        which is what a later step refuses. DECLINED stays a caller's decision."""
        self.assertEqual([use.value for use in CameraWorldUse],
                         ["unstated", "planned", "declined", "unplanned", "missing"])

    def test_a_missing_world_renders_its_reason(self) -> None:
        self.assertEqual(
            CameraWorldStamp.missing("no live camera world is wired").render(),
            "camera world  MISSING  no live camera world is wired",
        )


class TheDefaultPromisesNothingTests(unittest.TestCase):
    def test_a_result_built_without_a_stamp_is_unstated(self) -> None:
        results = (
            MotionResult.executed(MotionCommand.MOVE_TO),
            MotionResult.failed(MotionStatus.IK_FAILED, MotionCommand.MOVE_TO),
            MotionResult.from_bool(True, MotionCommand.MOVE_JOINTS),
            MotionResult.from_bool(False, MotionCommand.MOVE_HOME),
            MotionResult(status=MotionStatus.EXECUTED, command=MotionCommand.OTHER),
        )
        for result in results:
            with self.subTest(status=result.status.value, command=result.command.value):
                self.assertIs(result.camera_world.use, CameraWorldUse.UNSTATED)
                self.assertFalse(result.camera_world.vouched)

    def test_only_a_planned_stamp_vouches(self) -> None:
        vouching = [stamp.use for stamp in _every_kind() if stamp.vouched]
        self.assertEqual(vouching, [CameraWorldUse.PLANNED])


class ADeclineNeedsAReasonTests(unittest.TestCase):
    def test_an_empty_or_blank_reason_is_refused(self) -> None:
        for reason in ("", "   ", "\t\n"):
            with self.subTest(reason=repr(reason)), self.assertRaises(ValueError):
                CameraWorldDecline(reason)

    def test_an_unplanned_cell_must_say_why_too(self) -> None:
        with self.assertRaises(ValueError):
            CameraWorldStamp.unplanned("  ")

    def test_a_missing_world_must_say_why_too(self) -> None:
        for reason in ("", "   "):
            with self.subTest(reason=repr(reason)), self.assertRaises(ValueError):
                CameraWorldStamp.missing(reason)

    def test_the_reason_reaches_the_stamp_and_its_text(self) -> None:
        reason = "calibration sweep, no CAMERA->BASE yet"
        stamp = CameraWorldStamp.declined(CameraWorldDecline(reason))
        self.assertIs(stamp.use, CameraWorldUse.DECLINED)
        self.assertEqual(stamp.reason, reason)
        self.assertIn(reason, stamp.render())


class APlannedStampNamesWhatItSawTests(unittest.TestCase):
    def test_planned_needs_a_camera(self) -> None:
        with self.assertRaises(ValueError):
            CameraWorldStamp.planned(cameras=(), captured_at_s=12.5)

    def test_planned_carries_the_cameras_and_the_capture_time(self) -> None:
        stamp = CameraWorldStamp.planned(cameras=("realsense_d435",), captured_at_s=12.5)
        self.assertEqual(stamp.cameras, ("realsense_d435",))
        self.assertEqual(stamp.captured_at_s, 12.5)
        self.assertIn("realsense_d435", stamp.render())


class TheStampDescribesItselfTests(unittest.TestCase):
    def test_render_names_the_use_in_ascii_without_a_trailing_newline(self) -> None:
        for stamp in _every_kind():
            with self.subTest(use=stamp.use.value):
                text = stamp.render()
                self.assertIn(stamp.use.value.upper(), text)
                self.assertTrue(text.isascii())
                self.assertFalse(text.endswith("\n"))

    def test_to_dict_is_plain_data_and_agrees_with_the_object(self) -> None:
        for stamp in _every_kind():
            with self.subTest(use=stamp.use.value):
                data = json.loads(json.dumps(stamp.to_dict()))
                self.assertEqual(data["use"], stamp.use.value)
                self.assertEqual(data["vouched"], stamp.vouched)
                self.assertEqual(data["reason"], stamp.reason)
                self.assertEqual(data["cameras"], list(stamp.cameras))
                self.assertEqual(data["captured_at_s"], stamp.captured_at_s)


class AStampWithNothingKeptOutTests(unittest.TestCase):
    def test_a_stamp_with_no_keep_out_equals_todays_planned_stamp(self) -> None:
        """The control, green before and after: a planned stamp built with no region says what it said before."""
        stamp = CameraWorldStamp.planned(cameras=("overhead",), captured_at_s=100.0)

        self.assertEqual("camera world  PLANNED  cameras overhead, image captured at 100.000 s", stamp.render())
        data = stamp.to_dict()
        self.assertEqual(
            {"use": "planned", "vouched": True, "reason": "", "cameras": ["overhead"], "captured_at_s": 100.0},
            {key: data[key] for key in ("use", "vouched", "reason", "cameras", "captured_at_s")},
        )
        self.assertEqual(stamp, CameraWorldStamp.planned(cameras=("overhead",), captured_at_s=100.0))
        self.assertIsNone(getattr(stamp, "keep_out", None))


class TheStampIsPartOfTheResultTests(unittest.TestCase):
    def test_results_that_differ_only_in_the_stamp_are_different_results(self) -> None:
        plain = MotionResult.executed(MotionCommand.MOVE_TO)
        declined = MotionResult.executed(
            MotionCommand.MOVE_TO,
            camera_world=CameraWorldStamp.declined(CameraWorldDecline("bench test")),
        )
        self.assertNotEqual(plain, declined)

    def test_every_constructor_carries_a_supplied_stamp(self) -> None:
        stamp = CameraWorldStamp.unplanned("robot.ur.motion_planner is 'ik'")
        built = (
            MotionResult.executed(MotionCommand.MOVE_TO, camera_world=stamp),
            MotionResult.failed(
                MotionStatus.WORKSPACE_REJECTED, MotionCommand.MOVE_TO, camera_world=stamp,
            ),
            MotionResult.from_bool(True, MotionCommand.MOVE_TO, camera_world=stamp),
            MotionResult.from_bool(False, MotionCommand.MOVE_TO, camera_world=stamp),
        )
        for result in built:
            with self.subTest(status=result.status.value):
                self.assertEqual(result.camera_world, stamp)

    def test_the_exception_is_still_not_part_of_equality(self) -> None:
        stamp = CameraWorldStamp.unplanned("no planner on this cell")
        a = MotionResult.failed(MotionStatus.CONNECTION_ERROR, MotionCommand.MOVE_TO,
                                exception=ConnectionError("a"), camera_world=stamp)
        b = MotionResult.failed(MotionStatus.CONNECTION_ERROR, MotionCommand.MOVE_TO,
                                exception=ConnectionError("b"), camera_world=stamp)
        self.assertEqual(a, b)


class AStampHoldsOnlyWhatItsUseCanMeanTests(unittest.TestCase):
    """The invariants hold for the bare constructor as well as for the factories, and against the values
    a caller could plausibly pass by mistake."""

    def test_planned_refuses_a_single_name_where_a_sequence_of_names_belongs(self) -> None:
        """A str is a sequence of one-letter strings, so without this it became cameras c, a, m, 1. The
        name repeats no letter, so the duplicate rule cannot be what refuses it."""
        with self.assertRaises(TypeError):
            CameraWorldStamp.planned(cameras="cam1", captured_at_s=12.5)

    def test_the_bare_constructor_refuses_cameras_that_are_not_a_tuple(self) -> None:
        """planned() always hands over a tuple, so this check is the only one the bare constructor meets. A
        list would also make the stamp, and every result carrying it, unhashable."""
        for cameras in ("cam1", ["realsense_d435"]):
            with self.subTest(cameras=repr(cameras)), self.assertRaises(TypeError):
                CameraWorldStamp(use=CameraWorldUse.PLANNED, cameras=cameras, captured_at_s=1.0)

    def test_each_empty_field_is_the_empty_value_of_its_own_type(self) -> None:
        """None, [] or "" where a field must stay empty is refused too, or a stamp rebuilt from to_dict()
        would compare unequal to the one it came from, and a None cameras breaks to_dict()."""
        cases = (
            {"use": CameraWorldUse.UNSTATED, "reason": "bench"},
            {"use": CameraWorldUse.UNSTATED, "cameras": ("realsense_d435",)},
            {"use": CameraWorldUse.UNSTATED, "captured_at_s": 1.0},
            {"use": CameraWorldUse.UNSTATED, "reason": None},
            {"use": CameraWorldUse.UNSTATED, "cameras": None},
            {"use": CameraWorldUse.UNSTATED, "cameras": []},
            {"use": CameraWorldUse.DECLINED, "reason": "bench", "cameras": None},
            {"use": CameraWorldUse.DECLINED, "reason": "bench", "cameras": []},
            {"use": CameraWorldUse.UNPLANNED, "reason": "ik cell", "cameras": ""},
            {"use": CameraWorldUse.PLANNED, "cameras": ("realsense_d435",), "captured_at_s": 1.0,
             "reason": None},
        )
        for fields in cases:
            with self.subTest(**{key: repr(value) for key, value in fields.items()}):
                with self.assertRaises((TypeError, ValueError)):
                    CameraWorldStamp(**fields)

    def test_a_result_refuses_a_camera_world_that_is_not_a_stamp(self) -> None:
        """The result's repr reads the stamp's use, so a None or a string in its place would make every log
        line printing that result raise instead of print."""
        for value in (None, "declined", CameraWorldUse.DECLINED):
            with self.subTest(camera_world=repr(value)):
                with self.assertRaises(TypeError):
                    MotionResult(status=MotionStatus.EXECUTED, command=MotionCommand.MOVE_TO,
                                 camera_world=value)
                with self.assertRaises(TypeError):
                    MotionResult.executed(MotionCommand.MOVE_TO, camera_world=value)

    def test_planned_refuses_blank_repeated_or_non_text_camera_names(self) -> None:
        for cameras in (("",), ("   ",), ("realsense_d435", "realsense_d435"), (3,)):
            with self.subTest(cameras=cameras), self.assertRaises((TypeError, ValueError)):
                CameraWorldStamp.planned(cameras=cameras, captured_at_s=12.5)

    def test_planned_refuses_a_capture_time_that_is_no_time(self) -> None:
        for moment in (float("nan"), float("inf"), float("-inf"), -1.0, "12.5", True, None):
            with self.subTest(captured_at_s=moment), self.assertRaises((TypeError, ValueError)):
                CameraWorldStamp.planned(cameras=("realsense_d435",), captured_at_s=moment)

    def test_a_whole_number_of_seconds_is_a_time(self) -> None:
        stamp = CameraWorldStamp.planned(cameras=("realsense_d435",), captured_at_s=12)
        self.assertEqual(stamp.captured_at_s, 12.0)
        self.assertIsInstance(stamp.captured_at_s, float)

    def test_the_use_is_checked_and_a_plain_string_cannot_skip_the_rules(self) -> None:
        for use in ("planned", "declined", "unplanned", "missing", "not_a_use", 3):
            with self.subTest(use=use), self.assertRaises(ValueError):
                CameraWorldStamp(use=use)
        self.assertIs(CameraWorldStamp(use="unstated").use, CameraWorldUse.UNSTATED)

    def test_each_use_carries_only_its_own_fields(self) -> None:
        cases = (
            {"use": CameraWorldUse.DECLINED, "reason": "bench", "cameras": ("realsense_d435",)},
            {"use": CameraWorldUse.DECLINED, "reason": "bench", "captured_at_s": 1.0},
            {"use": CameraWorldUse.UNPLANNED, "reason": "ik cell", "cameras": ("realsense_d435",)},
            {"use": CameraWorldUse.UNPLANNED, "reason": "ik cell", "captured_at_s": 1.0},
            {"use": CameraWorldUse.MISSING, "reason": "no world", "cameras": ("realsense_d435",)},
            {"use": CameraWorldUse.MISSING, "reason": "no world", "captured_at_s": 1.0},
            {"use": CameraWorldUse.PLANNED, "cameras": ("realsense_d435",), "captured_at_s": 1.0,
             "reason": "a planned world has no reason to give"},
        )
        for fields in cases:
            with self.subTest(**{key: str(value) for key, value in fields.items()}):
                with self.assertRaises(ValueError):
                    CameraWorldStamp(**fields)

    def test_render_stays_one_ascii_line_whatever_a_caller_wrote(self) -> None:
        reason = "Kalibrierfahrt ohne Kamera\nzweite Zeile\tmit Tab, Grüße\r"
        stamps = (
            CameraWorldStamp.declined(CameraWorldDecline(reason)),
            CameraWorldStamp.unplanned(reason),
            CameraWorldStamp.planned(cameras=("kamera_süd\nzwei",), captured_at_s=1.0),
        )
        for stamp in stamps:
            with self.subTest(use=stamp.use.value):
                text = stamp.render()
                self.assertTrue(text.isascii(), text)
                for control in ("\n", "\r", "\t"):
                    self.assertNotIn(control, text)
                self.assertIn("Kalibrierfahrt" if stamp.use is not CameraWorldUse.PLANNED else "kamera_s",
                              text)


class TheStampIsVisibleWhereResultsAreLoggedTests(unittest.TestCase):
    """A decline is visible in the text a log line prints for a result, and silence adds nothing to it."""

    def test_an_unstated_stamp_leaves_the_repr_exactly_as_it_was(self) -> None:
        """`tests/test_r6_motion_enum_seam0.py` pins this text, and every `%r` of a result prints it."""
        self.assertEqual(
            repr(MotionResult.executed(MotionCommand.MOVE_TO, message="hi")),
            "MotionResult(status=<MotionStatus.EXECUTED: 'executed'>, "
            "command=<MotionCommand.MOVE_TO: 'move_to'>, target_pose=None, "
            "target_joints=None, message='hi')",
        )

    def test_any_other_stamp_is_shown_with_what_it_says(self) -> None:
        reason = "bench test, no cameras mounted"
        stamps = (
            CameraWorldStamp.declined(CameraWorldDecline(reason)),
            CameraWorldStamp.unplanned(reason),
            CameraWorldStamp.missing(reason),
            CameraWorldStamp.planned(cameras=(reason,), captured_at_s=1.0),
        )
        for stamp in stamps:
            with self.subTest(use=stamp.use.value):
                text = repr(MotionResult.executed(MotionCommand.MOVE_TO, camera_world=stamp))
                self.assertIn("camera_world=", text)
                self.assertIn(reason, text)


if __name__ == "__main__":
    unittest.main()
