"""A pick returns before its record is written, and a task's views are kept after the pick has gone on.

The owner's cell, 2026-10-08: after every retreat the arm waited 1.5 to 2.3 s while the pick's record (a 5 MB line) and
its views were written. A cell that writes in the background (``build_real_cell`` turns it on:
``AutonomousGraspService.write_records_in_the_background``) hands both to the process's one background writer
(``record_logging.background_writer``): the pick returns at once, the records are written in the order of their picks
and stamped when each pick ended, whatever is still waiting is written before the process ends, and a write that fails
is logged and stops nothing. A task's views are taken when the pick ends, before the next pick lets them go, and the
file the task reports is the one they are kept in. Off, the default of every cell built by hand, records are written
before the pick returns, as before.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from src.robot.execution.autonomous_grasp import AutonomousGraspService
from src.robot.execution.autonomous_grasp import record_logging
from src.robot.execution.autonomous_grasp.record_logging import BackgroundWriter, background_writer
from src.robot.grasping.telemetry.outcome_logging import iter_jsonl
from src.robot.grasping.types.perception import PerceptionFrame
from tests.test_autonomous_grasp_service import (
    _FakePerception,
    _perception_frame,
    _ScriptedCalculator,
    _success_result,
    _TypedFakeArm,
)

ROOT = Path(__file__).resolve().parents[1]


def _service() -> AutonomousGraspService:
    return AutonomousGraspService.from_components(
        arm=_TypedFakeArm(),  # type: ignore[arg-type]
        calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
        perception=_FakePerception([_perception_frame()]),
    )


class _Gate:
    """``log_record`` held at a gate until the test opens it; every record it was asked for, in order."""

    def __init__(self) -> None:
        self.open = threading.Event()
        self.asked: list[str] = []
        self.real = record_logging.log_record

    def __call__(self, report: Any, **keywords: Any) -> Any:
        self.asked.append(str(keywords["attempt_id"]))
        assert self.open.wait(10.0), "the gate was never opened"
        return self.real(report, **keywords)


class ThePickReturnsFirstTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp())
        self.path = self.folder / "records.jsonl"
        self.gate = _Gate()
        patcher = mock.patch.object(record_logging, "log_record", self.gate)
        patcher.start()
        self.addCleanup(patcher.stop)
        # Whatever a failed test left behind is written before the next starts.
        self.addCleanup(lambda: (self.gate.open.set(), background_writer().flush(10.0)))

    def test_the_pick_returns_while_its_record_waits_and_the_record_is_written_after(self) -> None:
        service = _service()
        service.enable_record_logging(self.path)
        service.write_records_in_the_background()
        self.assertTrue(service.writes_in_the_background)

        before = time.time()
        report = service.pick()
        ended = time.time()
        self.assertTrue(report.succeeded)
        self.assertFalse(self.path.exists(), "the record was written before the pick returned")

        self.gate.open.set()
        self.assertTrue(background_writer().flush(10.0))
        (record,) = tuple(iter_jsonl(self.path))
        self.assertEqual(report.telemetry["attempt_id"], record.attempt_id)
        self.assertEqual("succeeded", record.final_outcome)
        self.assertLessEqual(before, record.timestamp)
        self.assertLessEqual(record.timestamp, ended, "the record is stamped when the pick ended")

    def test_the_records_are_written_in_the_order_of_their_picks(self) -> None:
        service = _service()
        service.enable_record_logging(self.path)
        service.write_records_in_the_background()
        ids = [str(service.pick().telemetry["attempt_id"]) for _ in range(4)]

        self.gate.open.set()
        self.assertTrue(background_writer().flush(10.0))
        records = tuple(iter_jsonl(self.path))
        self.assertEqual(ids, [record.attempt_id for record in records])
        self.assertEqual(sorted(record.timestamp for record in records), [record.timestamp for record in records])

    def test_off_the_record_is_written_before_the_pick_returns_as_before(self) -> None:
        self.gate.open.set()
        service = _service()
        service.enable_record_logging(self.path)
        self.assertFalse(service.writes_in_the_background)
        service.pick()
        self.assertEqual(1, len(tuple(iter_jsonl(self.path))))


class TheWriterTests(unittest.TestCase):
    def test_a_write_that_fails_is_logged_and_the_next_one_is_written(self) -> None:
        writer = BackgroundWriter(name="test-writer")
        written: list[str] = []

        def fails() -> None:
            raise OSError("the disk is full")

        with self.assertLogs("GraspRecordLogging", "WARNING") as said:
            writer.submit("the record of pick-1", fails)
            writer.submit("the record of pick-2", lambda: written.append("pick-2"))
            self.assertTrue(writer.flush(10.0))
        self.assertEqual(["pick-2"], written)
        self.assertEqual(1, len(said.records))
        self.assertIn("the record of pick-1 was not written: OSError: the disk is full", said.output[0])

    def test_a_writer_never_handed_anything_has_nothing_to_flush(self) -> None:
        self.assertTrue(BackgroundWriter().flush(0.0))

    def test_what_still_waits_when_the_process_ends_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "records.jsonl"
            done = subprocess.run([sys.executable, "-c", textwrap.dedent(f"""
                import time
                from src.robot.execution.autonomous_grasp import record_logging
                from tests.test_a_pick_returns_before_its_record_is_written import _service

                real = record_logging.log_record

                def slow(report, **keywords):
                    time.sleep(1.5)
                    return real(report, **keywords)

                record_logging.log_record = slow
                service = _service()
                service.enable_record_logging({str(path)!r})
                service.write_records_in_the_background()
                service.pick()
                service.pick()
                print("picked")
            """)], cwd=str(ROOT), capture_output=True, text=True, timeout=120, env=dict(os.environ))
            self.assertEqual(0, done.returncode, done.stderr)
            self.assertIn("picked", done.stdout)
            self.assertEqual(2, len(tuple(iter_jsonl(path))), "a record still waiting at the end was lost")


# ---------------------------------------------------------------------------------------------------------------------
# A task's views
# ---------------------------------------------------------------------------------------------------------------------


def _view(value: int) -> Any:
    frame = PerceptionFrame(depth_map=np.full((24, 32), 400.0 + value), intrinsics=np.array(
        [[300.0, 0.0, 16.0], [0.0, 300.0, 12.0], [0.0, 0.0, 1.0]]), segmentations=(),
        rgb=np.full((24, 32, 3), value, dtype=np.uint8))
    return SimpleNamespace(frame=frame, camera_to_base=np.eye(4), name=f"wrist@{value}", label=f"look {value}")


class ATasksViewsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.mkdtemp()
        patcher = mock.patch("src.robot.execution.record_views.RECORD_VIEWS_DIR", self.folder)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_views_are_taken_at_the_pick_and_written_where_the_task_said_after_it(self) -> None:
        from src.robot.execution.pick_run import keep_pick_views

        judged = SimpleNamespace(target_cloud_base_mm=np.zeros((5, 3)), result=SimpleNamespace(candidates=()))
        service = SimpleNamespace(looked_around=SimpleNamespace(views=(_view(10), _view(20)), judged=judged))
        report = SimpleNamespace(looks=("home",), telemetry={"attempt_id": "pick-0123"})
        writer = BackgroundWriter(name="test-views")
        gate = threading.Event()
        writer.submit("the gate", lambda: gate.wait(10.0))

        where = keep_pick_views(service, report, name="task-part1-pick1", writer=writer)
        # The next pick replaces the looks before the writer reaches them.
        service.looked_around = SimpleNamespace(views=(_view(99),), judged=judged)
        self.assertTrue(where.startswith(str(Path(self.folder) / "pick-0123-")), where)
        self.assertFalse(Path(where).exists(), "the views were written before the task went on")

        gate.set()
        self.assertTrue(writer.flush(10.0))
        kept = np.load(where, allow_pickle=False)
        self.assertEqual(["look 10", "look 20"], list(kept["labels"]))
        self.assertEqual(410.0, float(kept["depth_0"][0, 0]))
        self.assertTrue(Path(where).with_name(f"{Path(where).stem}_look0.png").exists(), "the debug pictures")
        self.assertEqual([Path(where).name], [p.name for p in Path(self.folder).glob("*.npz")])

    def test_views_that_cannot_be_written_are_logged_and_the_task_goes_on(self) -> None:
        from src.robot.execution.pick_run import keep_pick_views

        service = SimpleNamespace(looked_around=SimpleNamespace(views=(_view(10),), judged=None))
        report = SimpleNamespace(looks=("home",), telemetry={"attempt_id": "pick-0456"})
        writer = BackgroundWriter(name="test-views")
        with mock.patch("src.robot.execution.record_views.record_views", side_effect=OSError("disk full")), \
                self.assertLogs("GraspRecordLogging", "WARNING") as said:
            where = keep_pick_views(service, report, name="x", writer=writer)
            self.assertTrue(writer.flush(10.0))
        self.assertTrue(where)
        self.assertIn("the looks of pick-0456 was not written: OSError: disk full", said.output[0])

    def test_a_task_hands_its_views_to_the_writer_on_a_cell_that_writes_in_the_background(self) -> None:
        from src.robot.execution.task import TaskOptions
        from tests._task_fakes import TaskService, run

        for background in (True, False):
            with self.subTest(background=background), \
                    mock.patch.object(TaskService, "writes_in_the_background", background, create=True), \
                    mock.patch("src.robot.execution.task.keep_pick_views", return_value="D:/views/a.npz") as kept:
                run(["part"], options=TaskOptions(record_views=True))
            writer = kept.call_args.kwargs["writer"]
            self.assertIs(background_writer() if background else None, writer)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
