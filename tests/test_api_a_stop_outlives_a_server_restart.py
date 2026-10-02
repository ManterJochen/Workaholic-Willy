"""A stop outlives a restart of the console's server, and a rebuild of the cell (review of 2026-10-02).

The stop record, a halt a person pressed, and the service's "a person decides" latch lived only in the server's memory.
After a restart the cell built and connected clean, the ready bar said ALLES BEREIT, and a new task started from where
the problem stop had left the arm, with no "Zelle ist frei", Restart or Home. A halt pressed with no run went the same
way at Disconnect, Build and Connect: the rebuilt arm carried no latch. What is held here:

* the server's console keeps its stop in a small file (``Console.stop_file``; ``python -m api`` names one beside the
  console's grasp records): the record as it is written, cleared and ended, the stopped run it names, and a halt nobody
  has said the cell is clear of. Once nothing stands, the file is gone;
* a console started again reads it back. The record stands, uncleared (a restart is no "Zelle ist frei", and nobody
  knows what happened in the cell meanwhile), so every moving route answers ``cell_not_cleared`` and the ready bar is
  not ready; the stopped run is known again, so Restart replays its plan; a halt latches the arm of the next build, so
  Connect answers ``halted`` until "Zelle ist frei";
* a halt with no run is carried to the arm of every later build as well: a rebuild ends no halt, only a person does;
* a file nobody can read gates as a stop of unknown origin: "Zelle ist frei", then Home;
* a console nobody named a file for (a program's own, a test's) keeps nothing on disk.

Honesty bucket (2): the real console, routes and ``run_task`` on console_dummy; a second console on the same tree and
file stands for the server started again.
"""

from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tests._console_task_fakes import ConsoleCase, events_of

_TASK = {"object": "", "place": {"kind": "pose", "pose": None}}


class _Restartable(ConsoleCase):
    def setUp(self) -> None:
        super().setUp()
        self.stop_file = self.tmp.parent / "logs" / "console" / "stop.json"
        self.cell.stop_file = self.stop_file

    def restarted(self) -> None:
        """The server starts again: a new console on the same tree, profile and stop file, the old one let go of."""
        from api.cell import Console, set_console

        old = self.cell
        old.session.release()
        fresh = Console(root=old.root, profile=old.profile, stop_file=self.stop_file)
        fresh.record_log_path = old.record_log_path
        fresh.registry.countdown_step_s = 0.05
        set_console(fresh)
        self.cell = fresh

    def built(self) -> None:
        built = self.client.post("/v1/cell/build", params={"rehearse": True})
        self.assertEqual(200, built.status_code, built.text)

    def connect(self) -> Any:
        token = self.client.get("/v1/cell/connect-preview").json()["token"]
        return self.client.post("/v1/cell/connect", json={"token": token})

    def halted_after_its_pick(self) -> dict[str, Any]:
        """A task on console_dummy halted right after its pick: it ends ``halted``, believing a part is in the jaws."""
        from api import task_run

        original = task_run.ConsoleTaskHooks.pick_done

        def pick_done(hooks: Any, part: int, pick: int, report: Any) -> None:
            original(hooks, part, pick, report)
            self.client.post("/v1/cell/brake")

        with patch.object(task_run.ConsoleTaskHooks, "pick_done", pick_done):
            run = self.task(object="")
            stopped = self.finished(run["id"])
        self.assertEqual(("halted", True), (stopped["stop_code"], stopped["holding"]), stopped)
        return stopped

    def kept(self) -> dict[str, Any]:
        return dict(json.loads(self.stop_file.read_text(encoding="utf-8")))


class AStopRecordOutlivesARestartTests(_Restartable):
    def test_a_halted_task_gates_the_restarted_console_until_the_cell_is_clear_and_restart_replays_it(self) -> None:
        self.build_dummy()
        stopped = self.halted_after_its_pick()
        kept = self.kept()
        self.assertEqual((stopped["id"], "halted", None), (kept["recovery"]["run_id"], kept["recovery"]["stop_code"],
                                                            kept["recovery"]["cleared_at"]))
        self.assertEqual(stopped["id"], kept["run"]["id"])
        self.assertTrue(kept["halt"]["reason"])

        self.restarted()

        cell = self.client.get("/v1/cell").json()
        self.assertEqual((stopped["id"], "halted", True, None),
                         (cell["recovery"]["run_id"], cell["recovery"]["stop_code"], cell["recovery"]["holding"],
                          cell["recovery"]["cleared_at"]))
        self.assertIn("cell.recovery", [event.type for event in events_of(self.cell, "cell")])
        record = self.client.get(f"/v1/runs/{stopped['id']}")
        self.assertEqual(200, record.status_code, record.text)
        self.assertEqual(("halted", stopped["plan"]), (record.json()["stop_code"], record.json()["plan"]))
        self.built()
        self.assertIsNotNone(self.client.get("/v1/cell").json()["halted"], "the built arm carries no halt")
        self.refused(self.connect(), 409, "halted")
        self.assertEqual(200, self.client.post("/v1/cell/acknowledge",
                                               json={"cell_clear": True, "jaws_empty": True}).status_code)
        self.assertEqual(200, self.connect().status_code)
        readiness = self.client.get("/v1/cell/readiness").json()
        self.assertFalse(readiness["ready"])
        self.assertIn("restart_required", [blocker["code"] for blocker in readiness["blockers"]])
        self.refused(self.client.post("/v1/task", json=_TASK), 409, "restart_required")

        restart = self.client.post("/v1/task/restart", json={"run_id": stopped["id"]})

        self.assertEqual(202, restart.status_code, restart.text)
        self.assertEqual("finished", self.finished(restart.json()["id"])["stop_code"])
        self.assertIsNone(self.client.get("/v1/cell").json()["recovery"])
        self.assertFalse(self.stop_file.exists(), "nothing stands, and the file still says a stop does")

    def test_a_problem_stop_with_no_halt_gates_until_the_cell_is_clear_and_home(self) -> None:
        from api.codes import RunKind

        self.build_dummy()
        run = self.cell.registry.start_kind(self.cell, RunKind.TASK, lambda _c, _r: "return_failed")
        self.finished(run.id)
        self.assertTrue(self.stop_file.exists())

        self.restarted()
        self.build_dummy()

        self.refused(self.client.post("/v1/task", json=_TASK), 409, "cell_not_cleared")
        self.refused(self.client.post("/v1/pick", json={"prompt": "", "picks": 1}), 409, "cell_not_cleared")
        self.refused(self.client.post("/v1/cell/home", json={"to": "home"}), 409, "cell_not_cleared")
        self.assertFalse(self.client.get("/v1/cell/readiness").json()["ready"])
        self.assertEqual(200, self.client.post("/v1/cell/acknowledge", json={"cell_clear": True}).status_code)
        self.refused(self.client.post("/v1/task", json=_TASK), 409, "restart_required")
        home = self.client.post("/v1/cell/home", json={"to": "home"})
        self.assertEqual(202, home.status_code, home.text)
        self.assertEqual("finished", self.finished(home.json()["id"])["stop_code"])
        self.assertIsNone(self.client.get("/v1/cell").json()["recovery"])
        self.assertFalse(self.stop_file.exists())

    def test_a_restart_is_no_clear_a_cleared_record_asks_again(self) -> None:
        from api.codes import RunKind

        self.build_dummy()
        run = self.cell.registry.start_kind(self.cell, RunKind.TASK, lambda _c, _r: "cell_fault")
        self.finished(run.id)
        self.assertEqual(200, self.client.post("/v1/cell/acknowledge", json={"cell_clear": True}).status_code)
        self.assertIsNotNone(self.kept()["recovery"]["cleared_at"])

        self.restarted()
        self.build_dummy()

        self.assertIsNone(self.client.get("/v1/cell").json()["recovery"]["cleared_at"])
        self.refused(self.client.post("/v1/cell/home", json={"to": "home"}), 409, "cell_not_cleared")

    def test_a_file_nobody_can_read_gates_as_a_stop_of_unknown_origin(self) -> None:
        self.stop_file.parent.mkdir(parents=True, exist_ok=True)
        self.stop_file.write_text("{ this is no stop record", encoding="utf-8")

        self.restarted()
        self.build_dummy()

        record = self.client.get("/v1/cell").json()["recovery"]
        self.assertIsNotNone(record, "an unreadable stop file was taken for no stop")
        self.refused(self.client.post("/v1/task", json=_TASK), 409, "cell_not_cleared")
        self.assertEqual(200, self.client.post("/v1/cell/acknowledge",
                                               json={"cell_clear": True, "jaws_empty": True}).status_code)
        home = self.client.post("/v1/cell/home", json={"to": "home"})
        self.assertEqual(202, home.status_code, home.text)
        self.finished(home.json()["id"])
        self.assertIsNone(self.client.get("/v1/cell").json()["recovery"])
        self.assertFalse(self.stop_file.exists())


class AHaltIsCarriedToEveryLaterBuildTests(_Restartable):
    def test_a_halt_with_no_run_survives_disconnect_build_and_connect(self) -> None:
        """Finding P8: a halt with no run (a latch, no record) was ended by Disconnect, Build and Connect."""
        self.build_dummy()
        self.assertEqual(200, self.client.post("/v1/cell/brake").status_code)
        self.assertEqual(200, self.client.post("/v1/cell/disconnect").status_code)

        self.built()

        self.assertIsNotNone(self.client.get("/v1/cell").json()["halted"], "the rebuilt arm carries no halt")
        self.refused(self.connect(), 409, "halted")
        self.assertEqual(200, self.client.post("/v1/cell/acknowledge", json={"cell_clear": True}).status_code)
        self.assertEqual(200, self.connect().status_code)
        run = self.task(object="")
        self.assertEqual("finished", self.finished(run["id"])["stop_code"])
        self.assertFalse(self.stop_file.exists())

    def test_a_halt_with_no_run_survives_a_restart(self) -> None:
        self.build_dummy()
        self.assertEqual(200, self.client.post("/v1/cell/brake").status_code)
        self.assertIsNone(self.kept()["recovery"])
        self.assertTrue(self.kept()["halt"]["reason"])

        self.restarted()
        self.built()

        self.refused(self.connect(), 409, "halted")
        self.assertEqual(200, self.client.post("/v1/cell/acknowledge", json={"cell_clear": True}).status_code)
        self.assertEqual(200, self.connect().status_code)
        self.assertFalse(self.stop_file.exists())


class AStopTheShutdownCutShortIsKeptTests(_Restartable):
    def test_a_run_the_shutdown_abandons_writes_its_record_before_the_process_goes(self) -> None:
        """The server stops while a task drives the arm: the shutdown abandons the run, and the run's thread writes its
        stop record as it ends, a moment later. The shutdown waits for it, so the next start reads the stop back."""
        import threading
        import time

        from fastapi.testclient import TestClient

        from api import cell as cell_module
        from api.app import create_app
        from tests._console_task_fakes import Scripted, wait_for_event

        release = threading.Event()
        self.addCleanup(release.set)
        self.scripted([Scripted("part", on_executing=lambda: (release.wait(timeout=20), time.sleep(0.5)))])
        run = self.task(object="green cube")
        wait_for_event(self.cell, run["id"], "pick.executing")
        real = cell_module.take_down

        def take_down_then_let_the_run_end(*args: Any, **keywords: Any) -> None:
            real(*args, **keywords)
            release.set()

        with patch("api.cell.take_down", side_effect=take_down_then_let_the_run_end):
            with TestClient(create_app()):
                pass

        self.assertTrue(self.stop_file.exists(), "the shutdown went before the run it cut short kept its stop")
        self.assertEqual((run["id"], "disconnected"), (self.kept()["recovery"]["run_id"],
                                                       self.kept()["recovery"]["stop_code"]))
        self.restarted()
        self.assertEqual(run["id"], self.client.get("/v1/cell").json()["recovery"]["run_id"])


class WhereNothingIsKeptTests(unittest.TestCase):
    def test_a_console_nobody_named_a_file_for_keeps_nothing(self) -> None:
        from api.cell import STOP_FILE_ENV, Console

        with patch.dict(os.environ, {STOP_FILE_ENV: ""}):
            self.assertIsNone(Console().stop_file)
        named = Path(__file__).resolve().parent / "no_such_dir" / "stop.json"
        with patch.dict(os.environ, {STOP_FILE_ENV: str(named)}):
            # The variable carries the server's file to ``--reload``'s worker, whose console is the module's own.
            self.assertEqual(named, Console().stop_file)
            self.assertIsNone(Console(stop_file=None).stop_file, "a console told to keep nothing kept a file")
        self.assertFalse(named.parent.exists(), "a console wrote a stop that never stood")

    def test_a_desk_rehearsal_and_the_cell_keep_their_stops_apart(self) -> None:
        from api.cell import default_stop_file

        self.assertEqual("stop.json", default_stop_file(None).name)
        self.assertEqual("stop.console_dummy.json", default_stop_file("console_dummy").name)
        self.assertEqual("stop.ur10+tiltcam.json", default_stop_file("ur10,tiltcam").name)
        self.assertEqual(("logs", "console"), default_stop_file(None).parent.parts[-2:])


class TheServerKeepsItsStopTests(unittest.TestCase):
    """``python -m api``: its console keeps its stop beside its grasp records, on the shipped tree; nothing on a tree
    named with ``--data`` unless ``WILLY_CONSOLE_STOP_FILE`` names a file."""

    def setUp(self) -> None:
        import tempfile

        import uvicorn

        from api.cell import console, set_console

        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self._real_run, uvicorn.run = uvicorn.run, lambda *_a, **_k: None
        self.addCleanup(setattr, uvicorn, "run", self._real_run)
        self.addCleanup(set_console, console())

    def _main(self, argv: list[str]) -> Any:
        import contextlib
        import io

        from api.__main__ import main
        from api.cell import console

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(0, main(argv))
        return console()

    def _free_port(self) -> str:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            return str(probe.getsockname()[1])

    def test_on_the_shipped_tree_beside_its_records(self) -> None:
        kept = self.tmp / "stop.json"
        with patch.dict(os.environ, {"WILLY_CONSOLE_STOP_FILE": ""}), \
                patch("api.__main__.default_stop_file", return_value=kept) as named:
            cell = self._main(["--port", self._free_port()])
        self.assertEqual(kept, cell.stop_file)
        named.assert_called_once_with(cell.profile)

    def test_a_tree_named_with_data_keeps_nothing_unless_a_file_is_named(self) -> None:
        from api.cell import _REPO_ROOT

        with patch.dict(os.environ, {"WILLY_CONSOLE_STOP_FILE": ""}):
            cell = self._main(["--port", self._free_port(), "--data", str(_REPO_ROOT / "config")])
        self.assertIsNone(cell.stop_file)
        named = self.tmp / "named.json"
        with patch.dict(os.environ, {"WILLY_CONSOLE_STOP_FILE": str(named)}):
            cell = self._main(["--port", self._free_port(), "--data", str(_REPO_ROOT / "config")])
        self.assertEqual(named, cell.stop_file)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
