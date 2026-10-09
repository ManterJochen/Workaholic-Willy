"""``robot.grasping.workers`` starts SFE's pool where a real cell is built; the pick loop hands it to the calculator.

How many worker processes search a part's grasps beside the cell's own: ``0`` (the repository's default) none, a number
that many, ``auto`` every physical core but one (``psutil``, else half the logical cores less one, never fewer than
one), said in the log. A yes or a no, a negative number or another word is refused at load. ``build_real_cell`` starts
the pool once per process (``workers.shared_workers``: a rebuild keeps it, another size replaces it), hands it to the
pick loop (``BinPickingOrchestrator.sfe_workers``) and turns the background records on; the pick loop hands the pool to
the calculator with every ranking, and hands nothing where there is none, so a calculator is asked as before.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from pydantic import ValidationError

from src.config.schema.robot.grasping_schema import RobotGraspingConfig
from src.robot.grasping import workers as workers_module
from src.robot.grasping.workers import SfeWorkers, shared_workers, worker_count

ROOT = Path(__file__).resolve().parents[1]


class TheKeyTests(unittest.TestCase):
    def test_none_is_the_default_of_the_schema_and_of_the_repositorys_tree(self) -> None:
        from src.config.loader import load_robot_section

        self.assertEqual(0, RobotGraspingConfig().workers)
        self.assertEqual(0, load_robot_section().grasping.workers)

    def test_auto_and_a_number_load_and_a_yes_a_negative_or_another_word_is_refused(self) -> None:
        self.assertEqual("auto", RobotGraspingConfig(workers="auto").workers)
        self.assertEqual(6, RobotGraspingConfig(workers=6).workers)
        for wrong in (True, False, -1, "many", 2.5):
            with self.subTest(wrong=wrong), self.assertRaises(ValidationError):
                RobotGraspingConfig(workers=wrong)

    def test_the_reference_tree_writes_it_down_with_its_default(self) -> None:
        text = (ROOT / "config" / "all_keys" / "robot" / "robot.yaml").read_text(encoding="utf-8")
        before, _, after = text.partition("    workers: 0\n")
        self.assertTrue(after, "robot.grasping.workers is not in config/all_keys/robot/robot.yaml")
        self.assertIn("# [default: 0]", before.splitlines()[-1])


class HowManyWorkersTests(unittest.TestCase):
    def test_a_number_is_that_many_and_none_is_none(self) -> None:
        self.assertEqual(0, worker_count(0)[0])
        self.assertIn("no workers", worker_count(0)[1])
        self.assertEqual((3, "3 worker(s), as configured"), worker_count(3))

    def test_auto_is_every_physical_core_but_one_and_says_so(self) -> None:
        for physical, expected in ((16, 15), (4, 3), (1, 1)):
            with self.subTest(physical=physical), \
                    mock.patch("psutil.cpu_count", return_value=physical):
                count, said = worker_count("auto")
            self.assertEqual(expected, count)
            self.assertIn(f"the {physical} physical cores but one", said)

    def test_without_psutil_auto_is_half_the_logical_cores_less_one_never_fewer_than_one(self) -> None:
        for logical, expected in ((32, 15), (8, 3), (2, 1), (None, 1)):
            with self.subTest(logical=logical), \
                    mock.patch.dict("sys.modules", {"psutil": None}), \
                    mock.patch("os.cpu_count", return_value=logical):
                count, said = worker_count("auto")
            self.assertEqual(expected, count)
            self.assertIn("psutil did not say", said)

    def test_never_more_than_the_most_a_pool_runs_and_it_says_so(self) -> None:
        count, said = worker_count(64)
        self.assertEqual(workers_module.MOST_WORKERS, count)
        self.assertIn(f"{workers_module.MOST_WORKERS} at most", said)
        with mock.patch("psutil.cpu_count", return_value=96):
            self.assertEqual(workers_module.MOST_WORKERS, worker_count("auto")[0])
        with self.assertRaises(ValueError):
            SfeWorkers(workers_module.MOST_WORKERS + 1)

    def test_anything_else_is_refused(self) -> None:
        for wrong in (True, -2, "several", 1.5):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                worker_count(wrong)


class _Orchestrator(SimpleNamespace):
    pass


class TheBuildStartsThePoolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.addCleanup(self._no_shared_pool)

    @staticmethod
    def _no_shared_pool() -> None:
        workers_module._close_the_shared_pool()
        workers_module._SHARED = None

    def _built(self, setting: Any) -> Any:
        from src.robot.execution.autonomous_grasp.cells import _start_the_pick_writers

        service = SimpleNamespace(runtime=SimpleNamespace(orchestrator=_Orchestrator(sfe_workers=None)),
                                  background=None)
        service.write_records_in_the_background = lambda on=True: setattr(service, "background", on)
        _start_the_pick_writers(SimpleNamespace(grasping=SimpleNamespace(workers=setting)), service)
        return service

    def test_no_workers_start_no_pool_and_the_records_go_to_the_background(self) -> None:
        service = self._built(0)
        self.assertIsNone(service.runtime.orchestrator.sfe_workers)
        self.assertIs(True, service.background)

    def test_workers_start_once_per_process_and_are_handed_to_the_pick_loop(self) -> None:
        with self.assertLogs("src.robot.execution.autonomous_grasp.cells", "INFO") as said:
            first = self._built(2)
        pool = first.runtime.orchestrator.sfe_workers
        self.assertIsInstance(pool, SfeWorkers)
        self.assertTrue(pool.running)
        self.assertEqual(2, len(pool.pids()))
        self.assertIn("2 worker(s), as configured", said.output[0])

        again = self._built(2)
        self.assertIs(pool, again.runtime.orchestrator.sfe_workers, "a rebuild keeps the running pool")
        other = self._built(1)
        self.assertIsNot(pool, other.runtime.orchestrator.sfe_workers, "another size replaces it")
        self.assertFalse(pool.running)

    def test_a_pool_that_cannot_start_is_said_and_every_part_is_searched_in_the_cells_process(self) -> None:
        with mock.patch.object(SfeWorkers, "_start", side_effect=workers_module._Broken("spawning was blocked")), \
                self.assertLogs("src.robot.grasping.workers", "WARNING") as said:
            pool = shared_workers(2)
        assert pool is not None
        self.assertIn("spawning was blocked", pool.given_up)
        self.assertIn("every part is searched in the cell's process", said.output[0])


class ThePickLoopHandsThePoolOnTests(unittest.TestCase):
    def _kwargs_seen(self, pool: Any) -> dict[str, Any]:
        from src.robot.execution.autonomous_grasp import AutonomousGraspService
        from tests.test_autonomous_grasp_service import (
            _FakePerception,
            _perception_frame,
            _ScriptedCalculator,
            _success_result,
            _TypedFakeArm,
        )

        calculator = _ScriptedCalculator([_success_result()])
        service = AutonomousGraspService.from_components(
            arm=_TypedFakeArm(), calculator=calculator,  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]))
        service.runtime.orchestrator.sfe_workers = pool
        service.pick()
        return calculator.kwargs_seen[0]

    def test_the_calculator_is_handed_the_pool_with_every_ranking(self) -> None:
        pool = object()
        self.assertIs(pool, self._kwargs_seen(pool)["sfe_workers"])

    def test_without_a_pool_the_calculator_is_asked_as_before(self) -> None:
        self.assertNotIn("sfe_workers", self._kwargs_seen(None))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
