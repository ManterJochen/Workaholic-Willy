"""The console remembers the bin its last task kept, and hands it to the next task into the same phrase (2026-10-08).

A task reports the bin it followed last (``TaskReport.kept_target``); the console keeps it by the phrase the bin was
found for (``api.task_run.remembered_bin``) and hands it to the next task into that phrase, a Restart included, which
looks at it first instead of surveying (``run_task(..., known_target=)``). Only while the cell stands as it stood: a
rebuild, a disconnect of a cell that claims a controller (its cell lock is taken anew at every connect) and a changed
configuration forget it, and so does a task that lost its bin. Where the bin was followed by its rim's depth and colour
the check brings no new picture, and the target's overlay stays the one kept.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

from src.geometry import Pose
from tests._console_task_fakes import (
    LOOK_1,
    LOOK_2,
    ConsoleCase,
    Scripted,
    SightedLocator,
    wait_for_event,
)
from tests._task_fakes import BLUE, BinScene, MeasuringLocator, SeenBin, bin_object

_CAMERA_PLACE = {"kind": "camera", "phrase": "blue bin"}


class _Lock:
    """A cell lock as a connect of a cell that claims a controller holds it: a new one at every connect."""

    def release(self) -> None:
        return None


def _key(joints: Any) -> tuple[float, ...]:
    return tuple(round(v, 1) for v in joints.degrees())


class TheRememberedBinTests(ConsoleCase):
    def _cell(self, picks: list[Any], *, sees: Any = None) -> tuple[Any, Any]:
        """A scripted wrist cell whose camera place finds the blue bin ``sees`` answers, wherever the arm stands."""
        cell = self.scripted(picks, wrist=True, looks=(LOOK_1, LOOK_2))
        answer = sees if sees is not None else (lambda _tcp: (bin_object(),))
        locator = SightedLocator({"blue bin": answer}, arm=cell.arm, wrist=True)
        for where in ("src.robot.execution.place_target.locators_for_service",
                      "src.robot.execution.task.locators_for_service"):
            patcher = patch(where, return_value=[locator])
            patcher.start()
            self.addCleanup(patcher.stop)
        return cell, locator

    def _found(self, *, expect: str = "finished") -> dict[str, Any]:
        """Run one task into the blue bin; what its ``task.target_found`` said."""
        run = self.task(object="red cube", place=_CAMERA_PLACE)
        body = self.finished(run["id"])
        self.assertEqual(expect, body["stop_code"], body)
        return dict(wait_for_event(self.cell, run["id"], "task.target_found").data)

    def test_the_next_task_into_the_same_bin_is_handed_the_bin_the_last_kept(self) -> None:
        from api.task_run import remembered_bin

        self._cell([Scripted("part"), Scripted("part")])

        first = self._found()
        kept = remembered_bin(self.cell, "blue bin")
        second = self._found()

        self.assertIs(False, first["known"])
        assert kept is not None
        self.assertEqual("blue bin", kept.label)
        self.assertEqual((True, "detector"), (second["known"], second["by"]))

    def test_a_rebuild_forgets_the_bin(self) -> None:
        from api.task_run import remembered_bin

        self._cell([Scripted("part")])
        self._found()
        self.assertIsNotNone(remembered_bin(self.cell, "blue bin"))

        self.assertEqual(200, self.client.post("/v1/cell/disconnect").status_code)
        self.scripted([Scripted("part")], wrist=True, looks=(LOOK_1, LOOK_2))

        self.assertIsNone(remembered_bin(self.cell, "blue bin"))
        self.assertIs(False, self._found()["known"])

    def test_a_disconnect_of_a_cell_that_claims_a_controller_forgets_the_bin(self) -> None:
        from api.task_run import remembered_bin

        self._cell([Scripted("part")])
        self.cell.session.cell_lock = _Lock()
        self._found()
        self.assertIsNotNone(remembered_bin(self.cell, "blue bin"))

        self.assertEqual(200, self.client.post("/v1/cell/disconnect").status_code)
        self.assertIsNone(remembered_bin(self.cell, "blue bin"), "remembered across a disconnect")
        preview = self.cell.session.preview(self.cell.robot(), self.cell.fingerprint())
        self.cell.session.connect(preview.token, self.cell.fingerprint(), _Lock())
        self.assertIsNone(remembered_bin(self.cell, "blue bin"), "remembered across a connect")

    def test_a_disconnect_of_a_cell_that_claims_no_controller_forgets_the_bin_too(self) -> None:
        from api.task_run import remembered_bin

        self._cell([Scripted("part")])
        self._found()
        self.assertIsNotNone(remembered_bin(self.cell, "blue bin"))

        self.assertEqual(200, self.client.post("/v1/cell/disconnect").status_code)
        preview = self.cell.session.preview(self.cell.robot(), self.cell.fingerprint())
        self.cell.session.connect(preview.token, self.cell.fingerprint(), None)

        self.assertIsNone(remembered_bin(self.cell, "blue bin"), "remembered across a disconnect")

    def test_a_changed_configuration_forgets_the_bin(self) -> None:
        from api.task_run import remembered_bin

        self._cell([Scripted("part")])
        self._found()

        with patch.object(self.cell, "fingerprint", return_value="another tree"):
            self.assertIsNone(remembered_bin(self.cell, "blue bin"))
        self.assertIsNone(remembered_bin(self.cell, "blue bin"), "a bin forgotten came back")

    def test_a_task_that_lost_its_bin_leaves_none_remembered(self) -> None:
        """The bin moved too far and another of another size stands there: a bin lost is looked for again (the owner,
        2026-10-09), and none of its size is found anywhere."""
        from api.task_run import remembered_bin

        from tests._task_fakes import BIN_CENTRE, BIN_SIZE, bin_points

        moved: dict[str, Any] = {"mm": 0.0, "size": BIN_SIZE}

        def sees(_tcp: Any) -> Any:
            return (bin_object(bin_points((BIN_CENTRE[0] + moved["mm"], BIN_CENTRE[1]), moved["size"])),)

        self._cell([Scripted("part"), Scripted("part", then=lambda: moved.update(mm=180.0, size=(200.0, 120.0)))],
                   sees=sees)
        self._found()
        self.assertIsNotNone(remembered_bin(self.cell, "blue bin"))

        self._found(expect="target_lost")

        self.assertIsNone(remembered_bin(self.cell, "blue bin"))

    def test_forget_bins_forgets_every_bin(self) -> None:
        from api.task_run import forget_bins, remembered_bin

        self._cell([Scripted("part")])
        self._found()

        forget_bins(self.cell, "the operator's test")

        self.assertIsNone(remembered_bin(self.cell, "blue bin"))


class AFollowedRimKeepsItsPictureTests(ConsoleCase):
    def test_a_bin_followed_by_its_rim_keeps_the_targets_overlay(self) -> None:
        at_the_bin = Pose.tool_down(400.0, -300.0, 450.0, label="look_1")
        cell = self.scripted([Scripted("part")], wrist=True, looks=(LOOK_1, LOOK_2),
                             arm_keywords={"fk_table": {_key(LOOK_1): at_the_bin}})
        locator = MeasuringLocator(BinScene(bins=[SeenBin(label="blue bin", colour_bgr=BLUE)]), arm=cell.arm)
        for where in ("src.robot.execution.place_target.locators_for_service",
                      "src.robot.execution.task.locators_for_service"):
            patcher = patch(where, return_value=[locator])
            patcher.start()
            self.addCleanup(patcher.stop)

        run = self.task(object="red cube", place=_CAMERA_PLACE)
        body = self.finished(run["id"])

        self.assertEqual("finished", body["stop_code"], body)
        checked = wait_for_event(self.cell, run["id"], "task.target_checked").data
        self.assertEqual((True, "depth"), (checked["followed"], checked["by"]))
        self.assertEqual(["blue bin"], locator.asked)
        self.assertEqual(f"/v1/runs/{run['id']}/target/overlay", checked["target"]["overlay"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
