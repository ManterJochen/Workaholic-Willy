"""Where a second arm stands in the cell is measured by touching, never written (owner, 2026-10-15).

A person guides each arm's TCP onto the same marked points, in order (Freedrive, Enter at each, as a pose is taught by
hand), and the rigid fit of the two point sets is the arm's base in the cell's frame, the first arm's base. No
automatic calibration (owner, 2026-09-24): nothing moves by itself, and a touch that disagrees is refused, not kept.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.config import default_data_dir
from src.config.cell import load_cell
from src.robot.execution.arm_registration import (
    MAX_RMSE_MM,
    ArmRegistration,
    RegistrationRefused,
    base_in_cell,
    registration_file,
    solve_registration,
    touch_points,
)
from src.robot.execution.hand_guiding import HandGuide, HandGuidingRefused
from tests.test_a_pose_is_taught_by_hand import _answers, _Clock, _Console, _GuidedArm, _sample

#: The marked points as the frame arm's TCP reads them, in its base: spread over the table, not on one line.
POINTS = ((400.0, -300.0, 20.0), (700.0, -250.0, 22.0), (650.0, 200.0, 18.0), (350.0, 150.0, 25.0))
#: Where the second arm truly stands: 1.4 m along x, turned half a turn and 7 degrees, 5 mm higher.
YAW = math.radians(187.0)
TRUTH = np.array([[math.cos(YAW), -math.sin(YAW), 0.0, 1400.0],
                  [math.sin(YAW), math.cos(YAW), 0.0, -60.0],
                  [0.0, 0.0, 1.0, -5.0],
                  [0.0, 0.0, 0.0, 1.0]])


def _seen_by_the_other_arm(points: tuple[tuple[float, float, float], ...]) -> list[tuple[float, float, float]]:
    """The same points as the second arm's TCP reads them, in its own base."""
    inverse = np.linalg.inv(TRUTH)
    return [tuple(float(v) for v in (inverse @ np.array([*p, 1.0]))[:3]) for p in points]  # type: ignore[misc]


class TheFitTests(unittest.TestCase):
    def test_the_same_points_touched_by_both_arms_give_the_second_arms_base(self) -> None:
        matrix, rmse, residuals = solve_registration(POINTS, _seen_by_the_other_arm(POINTS))
        np.testing.assert_allclose(matrix, TRUTH, atol=1e-9)
        self.assertLess(rmse, 1e-9)
        self.assertEqual(len(residuals), 4)

    def test_a_touch_a_millimetre_off_is_kept_and_said(self) -> None:
        other = np.asarray(_seen_by_the_other_arm(POINTS))
        other += np.random.default_rng(7).normal(0.0, 0.5, other.shape)
        matrix, rmse, _ = solve_registration(POINTS, other)
        self.assertLess(rmse, 1.5)
        self.assertLess(float(np.linalg.norm(matrix[:3, 3] - TRUTH[:3, 3])), 2.0)

    def test_what_the_fit_refuses(self) -> None:
        swapped = _seen_by_the_other_arm(POINTS)
        swapped[1], swapped[2] = swapped[2], swapped[1]
        line = ((400.0, 0.0, 20.0), (500.0, 0.0, 20.0), (600.0, 0.0, 20.0), (700.0, 0.0, 20.0))
        for frame, other, said in (
            (POINTS[:2], _seen_by_the_other_arm(POINTS[:2]), "at least 3"),
            (POINTS, _seen_by_the_other_arm(POINTS[:3]), "touch the same points with both"),
            (line, _seen_by_the_other_arm(line), "nearly on one line"),
            (POINTS, swapped, f"at most {MAX_RMSE_MM:.1f} mm kept"),
        ):
            with self.subTest(said=said), self.assertRaisesRegex(RegistrationRefused, said):
                solve_registration(frame, other)


class TouchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = _Clock()
        stands = [[_sample((0.0,) * 6, xyz=p, speed=0.3)] * 3 + [_sample((0.0,) * 6, xyz=p)] for p in POINTS]
        self.arm = _GuidedArm(stands, self.clock)

    def touch(self, *typed: str) -> tuple[tuple[float, float, float], ...]:
        self.console = _Console(_answers(*typed), self.arm.log)
        guide = HandGuide(self.console, None, clock=self.clock, sleep=self.clock.sleep)
        return touch_points(self.arm, name="werker", guide=guide)

    def test_enter_at_each_point_reads_the_held_tcp_and_the_arm_never_moves_by_itself(self) -> None:
        """The points lie on the table, under any workspace box's floor: touching them is no motion."""
        # The payload, then Enter at each of the four points.
        points = self.touch("", "", "", "", "")
        self.assertEqual(points, POINTS)
        steps = [line.split(" at ")[0] for line in self.arm.log if not line.startswith("say")]
        self.assertEqual(steps, ["payload read", "session"] + ["free", "hold"] * 4 + ["left, held"])
        self.assertIn("werker: point 4 at (350.0, 150.0, 25.0) mm; the arm holds", self.console.said)

    def test_finishing_early_keeps_nothing_and_leaves_the_arm_held(self) -> None:
        with self.assertRaisesRegex(HandGuidingRefused, "finished after 1 of 4 points"):
            self.touch("", "", "q")
        self.assertEqual(self.arm.log[-1], "left, held")


class _CellCase(unittest.TestCase):
    """A cell file of two desk arms in a fresh folder: halter, the frame, and werker."""

    def setUp(self) -> None:
        folder = Path(tempfile.mkdtemp())
        (folder / "cells").mkdir()
        (folder / "cells" / "clip.yaml").write_text("arms:\n  halter: console_dummy\n  werker: console_dummy\n",
                                                    encoding="utf-8")
        self.cell = load_cell(folder / "cells" / "clip.yaml", root=default_data_dir())
        self.assertTrue(self.cell.ok, self.cell.error)


class TheFileTests(_CellCase):
    def test_the_registration_lives_beside_the_cell_file_and_reads_back(self) -> None:
        matrix, rmse, residuals = solve_registration(POINTS, _seen_by_the_other_arm(POINTS))
        registration = ArmRegistration(arm="werker", frame_arm="halter", base_in_cell=matrix, rmse_mm=rmse,
                                       residuals_mm=residuals, frame_points_mm=POINTS,
                                       arm_points_mm=tuple(_seen_by_the_other_arm(POINTS)), captured_at="now")
        path = registration_file(self.cell, "WERKER")
        self.assertEqual(path, self.cell.path.parent / "clip" / "werker.json")
        self.assertIsNone(base_in_cell(self.cell, "werker"), "not registered yet")
        registration.write(path)
        np.testing.assert_allclose(base_in_cell(self.cell, "werker"), TRUTH, atol=1e-9)
        read = ArmRegistration.read(path)
        self.assertEqual(read.residuals_mm, residuals)
        self.assertIn("werker stands at (1400.0, -60.0, -5.0) mm, turned -173.00 deg, in halter's base", str(read))

    def test_the_first_arm_stands_at_the_frame(self) -> None:
        np.testing.assert_array_equal(base_in_cell(self.cell, "halter"), np.eye(4))

    def test_a_file_of_another_kind_is_refused(self) -> None:
        path = registration_file(self.cell, "werker")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"schema": "something.else"}', encoding="utf-8")
        with self.assertRaisesRegex(RegistrationRefused, "register the arm again"):
            base_in_cell(self.cell, "werker")


if __name__ == "__main__":
    unittest.main()


class _Robot:
    """A robot built alone from a tree: the scripted guided arm, connected for a ``with`` block."""

    def __init__(self, arm: _GuidedArm, log: list[str], name: str) -> None:
        self.arm, self.log, self.name = arm, log, name

    def connected(self) -> "_Robot":
        return self

    def __enter__(self) -> "_Robot":
        self.log.append(f"{self.name} connected")
        return self

    def __exit__(self, *exc: object) -> None:
        self.log.append(f"{self.name} disconnected")


class RegisteringAnArmTests(_CellCase):
    def test_both_arms_touch_the_same_marks_one_after_the_other_and_the_file_is_written(self) -> None:
        clock, log = _Clock(), []
        stands = {"halter": POINTS, "werker": tuple(_seen_by_the_other_arm(POINTS))}
        arms = {name: _GuidedArm([[_sample((0.0,) * 6, xyz=p)] for p in points], clock)
                for name, points in stands.items()}
        trees = {id(arm.tree): arm.name for arm in self.cell.arms}

        def robot_of(tree: object) -> _Robot:
            name = trees[id(tree)]
            return _Robot(arms[name], log, name)

        def guide_for(name: str) -> HandGuide:
            return HandGuide(_Console(_answers("", "", "", "", ""), log), None, clock=clock, sleep=clock.sleep)

        from src.robot.execution.arm_registration import register_arm

        registration = register_arm(self.cell, "werker", robot_of=robot_of, guide_for=guide_for)
        np.testing.assert_allclose(registration.base_in_cell, TRUTH, atol=1e-9)
        order = [line for line in log if line.endswith(("connected", "disconnected"))]
        self.assertEqual(order, ["halter connected", "halter disconnected", "werker connected", "werker disconnected"])
        np.testing.assert_allclose(base_in_cell(self.cell, "werker"), TRUTH, atol=1e-9)

    def test_the_frame_arm_is_not_registered(self) -> None:
        from src.robot.execution.arm_registration import register_arm

        with self.assertRaisesRegex(RegistrationRefused, "halter is the cell's frame"):
            register_arm(self.cell, "halter")

    def test_the_command_refuses_a_cell_that_does_not_load_and_too_few_marks(self) -> None:
        from src.robot.execution.real_cell.register import main

        self.assertEqual(main(["--cell", str(self.cell.path.parent / "nowhere.yaml"), "--arm", "werker"]), 1)
        with self.assertRaises(SystemExit) as usage:
            main(["--cell", str(self.cell.path), "--arm", "werker", "--points", "2"])
        self.assertEqual(usage.exception.code, 2)
