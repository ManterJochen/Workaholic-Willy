"""A console task says whether its parts are critical, and whether its push distance was asked for (the owner,
2026-10-03).

The Advanced drawer's switch ``options.critical_parts``: true, the task's campaign pushes nothing and a blocker is
cleared instead; false, a push may rearrange the scene; null, the cell's ``robot.grasping.recovery.critical_parts``. The
resolved plan echoes it, and says whether the operator set the push distance (``push_asked``): a distance asked for is
taken as asked, the cell's may go longer where it opens too little room, so the plan's resolved number reaches the
library only where it was asked for. The cell's facts say the cell's own switch, which the drawer shows.
"""

from __future__ import annotations

from types import SimpleNamespace

from tests._console_task_fakes import ConsoleCase, Scripted


class TheTaskSaysItsSwitchTests(ConsoleCase):

    def test_critical_parts_reach_the_campaign_and_the_distance_is_the_cells(self) -> None:
        cell = self.scripted([Scripted("part")])
        run = self.task(object="green cube", options={"critical_parts": True})
        options = run["plan"]["options"]
        self.assertEqual((True, False, 30.0), (options["critical_parts"], options["push_asked"], options["push_mm"]))
        self.finished(run["id"])
        self.assertEqual({"push_mm": None, "critical_parts": True}, cell.service.campaigns[-1])

    def test_a_distance_asked_for_is_taken_as_asked_and_the_cells_switch_stands(self) -> None:
        cell = self.scripted([Scripted("part")])
        run = self.task(object="green cube", options={"push_mm": 40})
        options = run["plan"]["options"]
        self.assertEqual((None, True, 40.0), (options["critical_parts"], options["push_asked"], options["push_mm"]))
        self.finished(run["id"])
        self.assertEqual({"push_mm": 40.0}, cell.service.campaigns[-1])


class TheFactsSayTheCellsSwitchTests(ConsoleCase):

    def test_the_cells_own_switch(self) -> None:
        from api.readiness import _push_facts

        for said in (False, True):
            with self.subTest(critical_parts=said):
                service = SimpleNamespace(push_cell=None, push_distance=lambda: 30.0, effective_config=SimpleNamespace(
                    recovery_orchestrator=SimpleNamespace(fixture=None, critical_parts=said)))
                self.assertIs(said, _push_facts(service).critical_parts)
        self.assertFalse(_push_facts(SimpleNamespace()).critical_parts)
