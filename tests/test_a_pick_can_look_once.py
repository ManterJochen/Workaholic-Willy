"""A pick can look from one place only, and the service says what a task needs to read off it (the owner's Q11, Q13).

"Multi-View aus" (Q11 A, 2026-09-30): a task the operator switched multi-view off for looks from the first configured
look only, and no view is generated after it. ``AutonomousGraspService.pick(multi_view=False)`` hands the pick loop the
first look alone and switches its generated view off for that pick (``BinPickingOrchestrator.generated_view``), both
taken back when the pick ends, whatever ended it, so the next pick looks as configured again. A fixed camera handed
looks tries the first one only.

And what a task reads off the service between its picks, each a small public door onto what was private:

* ``controller_refusal()``: why the controller cannot start a pick, in the words ``Robot.pick`` uses; ``""`` where it
  can, and on a service with no gripper, which the controller is not asked about;
* ``set_closing_axis(axis)``: the closing axis every grasp of the next picks closes along (the opt-in filter a task's
  Advanced drawer sets), refused before anything changes for a value that names no axis, and returning the one it
  replaced so a task puts it back;
* ``detector_failures()``: how many times the cell's grounding detectors failed and answered nothing, each backend
  counted once, so a task tells "the detector failed" from "nothing is there" (``detector_failed``);
* the module functions ``found_nothing(report)`` and ``only_excluded(report)``: a pick that saw nothing of what it was
  asked for, and one that saw only parts its campaign keeps out.

The wrist scene is ``tests/_wrist_views.py`` through the cell of ``tests/test_a_wrist_pick_hands_its_looks_to_the_pick_
loop.py``: a 40 mm cube on the bench, ray cast through the wrist D415 from the tool pose each look puts the arm at.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome
from tests.test_a_wrist_pick_hands_its_looks_to_the_pick_loop import (
    LOOK_MINUS_X,
    LOOK_PLUS_X,
    LOOK_PLUS_Y,
    _Cell,
    _key,
    _label,
    _looks_handed_back,
)


class APickLooksOnceTests(unittest.TestCase):
    def test_multi_view_off_looks_from_the_first_look_only_and_generates_no_view(self) -> None:
        """Red before: ``pick`` took no ``multi_view``, and a pick that found no safe grasp went on to every look and
        then to the generated view."""
        cell = _Cell(grasps_on=())
        dealt: list[bool] = []
        real = cell.orchestrator._generated_view  # noqa: SLF001

        def generated(*args: Any, **kwargs: Any) -> Any:
            dealt.append(True)
            return real(*args, **kwargs)

        with mock.patch.object(cell.orchestrator, "_generated_view", side_effect=generated):
            report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X, LOOK_PLUS_Y], multi_view=False)

        self.assertIs(AutonomousGraspOutcome.NO_VALID_GRASP, report.outcome)
        self.assertEqual([_key(LOOK_PLUS_X)], cell.joints_moved_to(), "the arm was sent on past the first look")
        self.assertEqual((_label(LOOK_PLUS_X),), report.looks)
        self.assertEqual([], dealt, "a view was generated although multi-view is off")
        self.assertIsNone(report.generated_view_deg)
        looked = cell.service.looked_around
        assert looked is not None
        self.assertIn("multi-view", looked.generated_skipped)
        _looks_handed_back(self, cell.orchestrator)
        self.assertIs(True, cell.orchestrator.generated_view, "the next pick would generate no view either")

    def test_multi_view_on_is_the_pick_it_was_every_look_and_the_view_after_them(self) -> None:
        cell = _Cell(grasps_on=())
        dealt: list[bool] = []
        real = cell.orchestrator._generated_view  # noqa: SLF001

        def generated(*args: Any, **kwargs: Any) -> Any:
            dealt.append(True)
            return real(*args, **kwargs)

        with mock.patch.object(cell.orchestrator, "_generated_view", side_effect=generated):
            report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X])

        self.assertEqual([_key(LOOK_PLUS_X), _key(LOOK_MINUS_X)], cell.joints_moved_to())
        self.assertEqual((_label(LOOK_PLUS_X), _label(LOOK_MINUS_X)), report.looks)
        self.assertEqual([True], dealt, "the one generated view was not even tried")

    def test_a_safe_grasp_at_the_first_look_is_picked_with_multi_view_off(self) -> None:
        cell = _Cell()

        report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X], multi_view=False)

        self.assertTrue(report.succeeded, report.failure_summary())
        self.assertEqual([_key(LOOK_PLUS_X)], cell.joints_moved_to())
        self.assertEqual(1, len(cell.policy.executed))

    def test_the_switch_is_taken_back_when_the_pick_raises(self) -> None:
        cell = _Cell(grasps_on=(0,))

        with mock.patch.object(cell.camera, "acquire", side_effect=TypeError("a programmer's error")), \
                self.assertRaises(TypeError):
            cell.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X], multi_view=False)

        self.assertIs(True, cell.orchestrator.generated_view)
        _looks_handed_back(self, cell.orchestrator)

    def test_a_fixed_camera_handed_looks_tries_the_first_only(self) -> None:
        cell = _Cell(grasps_on=(), fixed=True)

        report = cell.service.pick(look=[LOOK_PLUS_X, LOOK_MINUS_X], multi_view=False)

        self.assertEqual([_key(LOOK_PLUS_X)], cell.joints_moved_to())
        self.assertEqual((_label(LOOK_PLUS_X),), report.looks)


class TheServiceSaysWhatATaskReadsTests(unittest.TestCase):
    def test_the_controller_is_asked_in_the_words_robot_pick_uses(self) -> None:
        from src.robot.execution.autonomous_grasp import AutonomousGraspService, GraspMode
        from tests._task_fakes import PROTECTIVE, RUNNING, TaskArm, toggle
        from tests.test_a_stopped_controller_moves_no_jaws import _Calculator, _Frames, _policy

        log: list[Any] = []
        arm = TaskArm(log, statuses=(RUNNING, PROTECTIVE))
        jaws = toggle(log)
        service = AutonomousGraspService.from_components(
            arm=arm, calculator=_Calculator(), perception=_Frames(), mode=GraspMode.EASY,  # type: ignore[arg-type]
            gripper=jaws, policy=_policy(arm, jaws))

        self.assertEqual("", service.controller_refusal())
        said = service.controller_refusal()
        self.assertIn("the controller cannot move", said)
        self.assertIn("protective_stop=True", said)

    def test_a_closing_axis_is_set_on_the_policy_refused_when_it_names_none_and_the_old_one_returned(self) -> None:
        from src.geometry.closing_axis import closing_axis_of

        cell = _Cell()
        policy = cell.orchestrator.policy
        policy.closing_axis = None
        policy.align_closing_to_base_x = False

        self.assertIsNone(cell.service.set_closing_axis("-y"))
        self.assertEqual(closing_axis_of("-y"), policy.closing_axis)
        with self.assertRaises(ValueError):
            cell.service.set_closing_axis("sideways")
        self.assertEqual(closing_axis_of("-y"), policy.closing_axis, "a refused axis changed the policy")
        self.assertEqual(closing_axis_of("-y"), cell.service.set_closing_axis(None))
        self.assertIsNone(policy.closing_axis)

    def test_a_closing_axis_beside_a_twisting_motion_is_refused_with_nothing_changed(self) -> None:
        cell = _Cell()
        policy = cell.orchestrator.policy
        policy.closing_axis = None
        policy.align_closing_to_base_x = True

        with self.assertRaises(ValueError):
            cell.service.set_closing_axis("x")
        self.assertIsNone(policy.closing_axis)

    def test_the_refusal_of_a_closing_axis_is_read_without_setting_it(self) -> None:
        """What a task asks before its first event: whether ``set_closing_axis`` would refuse, read, nothing changed."""
        cell = _Cell()
        policy = cell.orchestrator.policy
        policy.closing_axis = None
        policy.align_closing_to_base_x = False

        self.assertEqual("", cell.service.closing_axis_refusal("-y"))
        self.assertEqual("", cell.service.closing_axis_refusal(None))
        self.assertTrue(cell.service.closing_axis_refusal("sideways"))
        policy.align_closing_to_base_x = True
        twisted = cell.service.closing_axis_refusal("x")
        self.assertTrue(twisted)
        self.assertEqual("", cell.service.closing_axis_refusal(None), "no axis is never refused")
        self.assertIsNone(policy.closing_axis, "reading the refusal set the axis")
        with self.assertRaises(ValueError) as raised:
            cell.service.set_closing_axis("x")
        self.assertEqual(twisted, str(raised.exception), "the read and the setter refuse in different words")

    def test_detector_failures_are_summed_over_the_cells_backends_each_counted_once(self) -> None:
        cell = _Cell()
        shared = SimpleNamespace(failures=2)
        other = SimpleNamespace(failures=1)
        sources = [SimpleNamespace(set_prompt=lambda *_a, **_k: None, backend=shared),
                   SimpleNamespace(set_prompt=lambda *_a, **_k: None, backend=shared),
                   SimpleNamespace(set_prompt=lambda *_a, **_k: None, backend=other)]
        cell.orchestrator.perception = sources[0]
        cell.orchestrator.multi_camera_perception = SimpleNamespace(sources={"a": sources[1], "b": sources[2]})

        self.assertEqual(3, cell.service.detector_failures())
        shared.failures = 5
        self.assertEqual(6, cell.service.detector_failures())

    def test_a_cell_whose_backends_count_nothing_reports_none(self) -> None:
        cell = _Cell()
        self.assertEqual(0, cell.service.detector_failures(), "the scene's camera grounds no phrase and counts nothing")
        cell.orchestrator.perception = SimpleNamespace(set_prompt=lambda *_a, **_k: None,
                                                       backend=SimpleNamespace(failures=True))
        self.assertEqual(0, cell.service.detector_failures(), "a flag is no count")

    def test_found_nothing_and_only_excluded_read_what_the_pick_said(self) -> None:
        from src.robot.execution.autonomous_grasp.service import found_nothing, only_excluded, only_kept_out

        cell = _Cell(grasps_on=())
        nothing = _Cell(grasps_on=())
        nothing.camera.boxes = ()

        failed = cell.service.pick(look=[LOOK_PLUS_X])
        empty = nothing.service.pick(look=[LOOK_PLUS_X])

        self.assertFalse(found_nothing(failed), "a part seen with no grasp was read as nothing there")
        self.assertTrue(found_nothing(empty))
        self.assertFalse(only_excluded(failed))
        self.assertFalse(only_excluded(empty))
        self.assertFalse(only_kept_out(failed))
        self.assertFalse(only_kept_out(empty))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
