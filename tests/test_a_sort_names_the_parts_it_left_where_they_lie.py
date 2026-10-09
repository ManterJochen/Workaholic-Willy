"""A sort that ends with nothing left names the parts it left where they lie (the owner, 2026-10-09: "liegen lassen,
am Ende nennen").

A part of no rule's kind is never boxed by the sort's class list ("each separate green part | each separate red
part" boxes no grey cube), so the last empty pick cannot say it saw one. At the end, one locate of every part ("each
separate object") from where the arm stands counts what lies outside every place of the task: a part placed into a bin
stands in that bin's region and is not counted, and the mat the parts stand on is wider than any part. The larger of
that count and what the last pick's first look turned away (``PickReport.unclaimed_labels``) is said
(``task.unsorted``), and a locate that cannot run leaves the labels' count. Nothing moves for it, and a task of one kind
asks no such locate.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.geometry import Pose
from src.robot.core import JointPositions
from src.robot.perception.locator import Located, LocatedObject
from tests._task_fakes import (
    BIN_CENTRE,
    BLUE,
    BinScene,
    ClassListLocator,
    SeenBin,
    SortPick,
    TaskArm,
    motions,
    run_sort,
)

LY = JointPositions.deg(-20.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LB = JointPositions.deg(40.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LE = JointPositions.deg(70.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LOOKS = (LY, LB, LE)
AT_LY = Pose.tool_down(BIN_CENTRE[0], BIN_CENTRE[1], 450.0, label="LY")
AT_LB = Pose.tool_down(-300.0, -500.0, 450.0, label="LB")
AT_LE = Pose.tool_down(-300.0, -50.0, 450.0, label="LE")
BLUE_CENTRE = (-300.0, -500.0)
EVERY_PART = "each separate object"


def _key(joints: JointPositions) -> tuple[float, ...]:
    return tuple(round(v, 1) for v in joints.degrees())


def _thing(x: float, y: float, side_mm: float, label: str = EVERY_PART) -> LocatedObject:
    """What a locate of every part boxes: a square of ``side_mm`` about (x, y) on the bench."""
    xs, ys = np.meshgrid(np.linspace(x - side_mm / 2.0, x + side_mm / 2.0, 6),
                         np.linspace(y - side_mm / 2.0, y + side_mm / 2.0, 6))
    points = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 30.0)])
    return LocatedObject(label=label, score=0.8, box_px=None, mask=np.zeros((2, 2), dtype=bool),
                         points_base_mm=points, centre_mm=(x, y, 30.0))


class LeftOverLocator(ClassListLocator):
    """A :class:`ClassListLocator` that, asked for every part, boxes ``left``: what still lies on the bench. ``raises``
    makes that locate raise, as a detector that failed does."""

    def __init__(self, scene: BinScene, *, arm: Any, log: "list[Any] | None" = None,
                 left: "tuple[LocatedObject, ...]" = (), raises: "BaseException | None" = None) -> None:
        super().__init__(scene, arm=arm, log=log)
        self.left = tuple(left)
        self.raises = raises

    def locate(self, prompt: str) -> Located:
        if prompt.strip() != EVERY_PART:
            return super().locate(prompt)
        self.asked.append(prompt)
        if self.raises is not None:
            raise self.raises
        frame, _image, _shows = self._taken()
        return Located(camera=self.rig_id, captured_at_s=float(self.count), mounting="eye_in_hand",
                       tool_to_base_mm=None, objects=self.left, _frame=frame)


def _plan(*, sort: bool = True) -> Any:
    from src.robot.execution.task import PlaceAt, SortRule, TaskPlan

    more = (SortRule(object="red part", place=PlaceAt(camera="blue bin")),) if sort else ()
    return TaskPlan(object="green part", place=PlaceAt(camera="yellow bin"), scope="until_empty", more_rules=more)


def _sort(picks: Any, *, left: "tuple[LocatedObject, ...]" = (), raises: "BaseException | None" = None,
          sort: bool = True, workspace: Any = None) -> Any:
    log: list[Any] = []
    arm = TaskArm(log, fk_table={_key(LY): AT_LY, _key(LB): AT_LB, _key(LE): AT_LE})
    if workspace is not None:
        arm.config = SimpleNamespace(workspace_limits=workspace)  # type: ignore[attr-defined]
    bins = [SeenBin()] + ([SeenBin(label="blue bin", centre_xy=BLUE_CENTRE, colour_bgr=BLUE)] if sort else [])
    locator = LeftOverLocator(BinScene(bins=bins), arm=arm, log=log, left=left, raises=raises)
    ran = run_sort(_plan(sort=sort), picks, arm=arm, locators=[locator], wrist=True, looks=LOOKS)
    ran.locator = locator  # type: ignore[attr-defined]
    return ran


#: A grey cube no rule names, on the bench away from both bins.
STRAY = _thing(-100.0, -300.0, 40.0)
#: A green part the sort placed: it stands in the yellow bin, inside the region the task keeps out.
PLACED = _thing(BIN_CENTRE[0], BIN_CENTRE[1], 40.0)
#: The mat the parts stand on: wider than any part.
MAT = _thing(-200.0, -250.0, 600.0)


class TheEndCountsWhatTheSortLeftTests(unittest.TestCase):
    def test_a_part_of_no_rules_kind_is_named_and_the_bins_and_the_mat_are_not(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = _sort([SortPick("part", label="green part"), "empty", "empty"], left=(STRAY, PLACED, MAT))

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual([{"count": 1, "labels": []}], ran.hooks.of("task.unsorted"))
        self.assertIn("; 1 part(s) no rule clearly claims stay where they lie", ran.report.sentence)
        self.assertEqual(1, ran.report.unsorted)
        self.assertEqual(1, ran.locator.asked.count(EVERY_PART), "one locate of every part, at the end")

    def test_the_larger_count_is_said_with_the_words_the_detector_gave(self) -> None:
        seen = ("ambiguous",)
        ran = _sort([SortPick("part", label="green part"), SortPick("empty", unclaimed=seen),
                     SortPick("empty", unclaimed=seen)], left=(STRAY, _thing(-150.0, -200.0, 40.0), PLACED))

        self.assertEqual([{"count": 2, "labels": ["ambiguous"]}], ran.hooks.of("task.unsorted"))
        self.assertEqual(2, ran.report.unsorted)

    def test_the_labels_count_where_the_locate_sees_fewer(self) -> None:
        seen = ("ambiguous", "orange part")
        ran = _sort([SortPick("part", label="green part"), SortPick("empty", unclaimed=seen),
                     SortPick("empty", unclaimed=seen)], left=())

        self.assertEqual([{"count": 2, "labels": ["ambiguous", "orange part"]}], ran.hooks.of("task.unsorted"))

    def test_a_locate_that_raises_leaves_the_labels_count_and_the_task_still_finishes(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = _sort([SortPick("part", label="green part"), SortPick("empty", unclaimed=("ambiguous",)),
                     SortPick("empty", unclaimed=("ambiguous",))], raises=RuntimeError("the detector failed"))

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual([{"count": 1, "labels": ["ambiguous"]}], ran.hooks.of("task.unsorted"))

    def test_a_thing_outside_the_cells_workspace_is_no_part_left(self) -> None:
        """A marker, a tool or a cable beside the work area is boxed by a locate of every part too: only what stands in
        the cell's ``robot.workspace_limits`` is counted."""
        workspace = SimpleNamespace(x_min=-500.0, x_max=0.0, y_min=-600.0, y_max=-200.0, z_min=0.0, z_max=400.0)
        beside = _thing(200.0, -300.0, 40.0)
        ran = _sort([SortPick("part", label="green part"), "empty", "empty"], left=(STRAY, beside),
                    workspace=workspace)

        self.assertEqual([{"count": 1, "labels": []}], ran.hooks.of("task.unsorted"))

    def test_nothing_left_anywhere_names_nothing(self) -> None:
        ran = _sort([SortPick("part", label="green part"), "empty", "empty"], left=(PLACED, MAT))

        self.assertNotIn("task.unsorted", ran.names())
        self.assertEqual(0, ran.report.unsorted)

    def test_the_count_moves_nothing(self) -> None:
        moved = _sort([SortPick("part", label="green part"), "empty", "empty"], left=(STRAY,))
        still = _sort([SortPick("part", label="green part"), "empty", "empty"], left=())

        self.assertEqual(motions(still.log), motions(moved.log), "the end's locate sends no motion")

    def test_a_task_of_one_kind_asks_no_such_locate(self) -> None:
        ran = _sort([SortPick("part", label="green part"), "empty", "empty"], left=(STRAY,), sort=False)

        self.assertNotIn(EVERY_PART, ran.locator.asked)
        self.assertNotIn("task.unsorted", ran.names())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
