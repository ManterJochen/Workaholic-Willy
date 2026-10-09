"""``POST /v1/task``: the console's one choice of looks and its carry switch reach every pick of the task.

The owner, 2026-10-08 night: "Multi-View" and "Alle Posen" became one choice (``options.multi_view`` and
``options.every_look``), and "Direkt über die Kante tragen" overrides the cell's carry to a bin for one task
(``options.carry``, null the cell's ``robot.place.carry``). The route echoes both in the resolved plan, the run carries
them to the library (``api.task_run.library_plan``), and each pick of the task is handed ``every_look``; asking for every
look with multi-view off is refused before anything moves.

These need the route's plan to echo the two options (``api/routers/task.py`` ``_plan``); the library's side is pinned in
``tests/test_a_wrist_pick_drives_every_look_where_asked.py``.
"""

from __future__ import annotations

import unittest

from tests._console_task_fakes import ConsoleCase, Scripted


class TheConsolesLooksAndCarryReachTheTaskTests(ConsoleCase):
    def test_a_task_asked_for_every_look_echoes_it_and_hands_it_to_each_pick(self) -> None:
        cell = self.scripted([Scripted("part")])

        run = self.task(object="green cube", options={"every_look": True, "carry": "over_the_rim"})

        options = run["plan"]["options"]
        self.assertEqual((True, True, "over_the_rim"), (options["multi_view"], options["every_look"], options["carry"]))
        self.assertEqual("finished", self.finished(run["id"])["stop_code"])
        self.assertIs(True, cell.service.calls[0].get("every_look"), cell.service.calls[0])

    def test_a_task_that_asks_for_neither_looks_and_carries_as_the_cell_does(self) -> None:
        cell = self.scripted([Scripted("part")])

        run = self.task(object="green cube")

        options = run["plan"]["options"]
        self.assertEqual((False, None), (options["every_look"], options["carry"]))
        self.finished(run["id"])
        self.assertNotIn("every_look", cell.service.calls[0])

    def test_every_look_with_multi_view_off_is_refused_before_anything_moves(self) -> None:
        cell = self.scripted([Scripted("part")])

        refused = self.refused(self.client.post("/v1/task", json={
            "object": "green cube", "place": {"kind": "pose", "pose": None},
            "options": {"multi_view": False, "every_look": True}}), 422, "bad_request")

        self.assertIn("every look", refused["message"])
        self.assertEqual([], cell.motions())
        self.assertIsNone(self.cell.registry.active())

    def test_a_carry_the_cell_does_not_know_is_refused(self) -> None:
        self.scripted([Scripted("part")])
        self.refused(self.client.post("/v1/task", json={
            "object": "green cube", "place": {"kind": "pose", "pose": None}, "options": {"carry": "sideways"}}),
            422, "bad_request")


if __name__ == "__main__":
    unittest.main()
