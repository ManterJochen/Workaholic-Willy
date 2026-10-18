"""The jaws wait for "Zelle ist frei": after a problem stop nothing moves before a person says the cell is clear, the one
change of DO0 that "Jetzt öffnen" sends included (review of 2026-10-02).

A halt already refused the jaws check (``halted``: a latched arm switches no output). A problem stop that is no halt
(three failed picks in a row, a part still held, a refused return, a controller stop already released at the pendant)
left the check and its one change of DO0 open before anybody said the cell is clear, against the stop card's own order
and ``api/README``'s "every moving route answers ``cell_not_cleared``". What is held here:

* ``POST /v1/cell/jaws/check`` answers ``409 cell_not_cleared`` while the stop record stands uncleared, asking nobody
  and switching nothing; once a person said the cell is clear it asks as before;
* the gate the hand asks right before its one change (``api.jaws.before_change``, the connect's question included)
  refuses while the record stands uncleared, and lets the change go once it is cleared;
* a hand that is not a toggle is asked nothing either way, as before.

Honesty bucket (2): the real console, routes and ``run_task`` on the scripted owner's cell (the real ``JawIOGripper``
single_toggle on a recording tool I/O).
"""

from __future__ import annotations

import threading
import time
import unittest
from typing import Any

from tests._console_task_fakes import ConsoleCase, Scripted, event_types


class AProblemStopThatIsNoHaltTests(ConsoleCase):
    def _stopped(self) -> Any:
        """Three failed picks in a row: a problem stop, no halt; the browser's question seam on the hand."""
        from api import jaws as browser_jaws

        cell = self.scripted([Scripted("failed")] * 3)
        run = self.task(object="green cube")
        body = self.finished(run["id"])
        self.assertEqual(("failed_in_a_row", "problem"), (body["stop_code"], body["stop_class"]), body)
        now = self.client.get("/v1/cell").json()
        self.assertIsNone(now["halted"])
        self.assertIsNone(now["recovery"]["cleared_at"])
        cell.jaws.answer_questions_with(self.cell.jaws, before_change=browser_jaws.before_change(self.cell))
        return cell

    def test_the_check_is_refused_until_a_person_said_the_cell_is_clear(self) -> None:
        cell = self._stopped()
        before = cell.do0_changes()

        refused = self.refused(self.client.post("/v1/cell/jaws/check"), 409, "cell_not_cleared")

        self.assertIn("the cell is clear", refused["message"])
        self.assertNotIn("cell.jaws_question", event_types(self.cell, "cell"), "the hand was asked before the clear")
        self.assertEqual(0, cell.do0_changes() - before)
        self.assertFalse(self.client.get("/v1/cell").json()["jaws_question"], "a refused check stayed pending")

        self.assertEqual(200, self.client.post("/v1/cell/acknowledge", json={"cell_clear": True}).status_code)
        answered: dict[str, Any] = {}

        def answer_open() -> None:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                waiting = self.client.get("/v1/cell/jaws").json().get("question")
                if waiting:
                    answered["status"] = self.client.post("/v1/cell/jaws/answer", json={
                        "question_id": waiting["question_id"], "choice": "open"}).status_code
                    return
                time.sleep(0.02)

        helper = threading.Thread(target=answer_open, daemon=True)
        helper.start()
        checked = self.client.post("/v1/cell/jaws/check")
        helper.join(timeout=15.0)
        self.assertEqual(200, checked.status_code, checked.text)
        self.assertEqual(200, answered.get("status"))
        self.assertEqual(0, cell.do0_changes() - before)

    def test_the_gate_before_the_one_change_waits_for_the_cell_to_be_clear(self) -> None:
        from api import jaws as browser_jaws

        self._stopped()
        gate = browser_jaws.before_change(self.cell)

        said = gate()

        self.assertIn("the cell is clear", said)
        self.assertIn("not changed", said)
        self.assertEqual(200, self.client.post("/v1/cell/acknowledge", json={"cell_clear": True}).status_code)
        self.assertEqual("", gate(), "a cleared stop still kept the change back")

    def test_a_hand_that_is_not_a_toggle_is_asked_nothing_either_way(self) -> None:
        from api.codes import RunKind

        self.build_dummy()
        run = self.cell.registry.start_kind(self.cell, RunKind.TASK, lambda _c, _r: "failed_in_a_row")
        self.finished(run.id)
        self.assertIsNotNone(self.cell.recovery)
        checked = self.client.post("/v1/cell/jaws/check")
        self.assertEqual(200, checked.status_code, checked.text)
        self.assertIsNone(checked.json()["question"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
