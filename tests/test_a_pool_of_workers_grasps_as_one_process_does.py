"""A pool of worker processes grasps as one process does, and a pool that breaks costs a part nothing but time.

The owner's cell, 2026-10-08: one boxed-in cube took 24 of a look's 30 s. SFE's units now run on worker processes the
cell starts once (``src/robot/grasping/workers.py``, ``robot.grasping.workers``), handed to the calculator by the pick
loop with every ranking. The answer is the one process's, to the bit: the recorded Zollstock looks of 2026-10-01 get
the same candidates, reasons, refusal counts and telemetry from a pool of two as from the cell's own process, and a
frame of several parts ranked by the pick loop chooses the same part with the same support telemetry and the same
overlay. A pool whose worker raises, dies or does not answer has that part searched in the cell's process, from the
start, said in one log line; one that breaks again is given up. A worker writes no log line, and one BLAS thread or
four give the same grasps.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import os
import subprocess
import sys
import textwrap
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.geometry import Frame, Transform
from src.robot.grasping.generation.support_footprint import generate_support_footprint_grasps
from src.robot.grasping.workers import SfeWorkers

ROOT = Path(__file__).resolve().parents[1]
_POOL: "SfeWorkers | None" = None


def setUpModule() -> None:  # noqa: N802 (unittest's name)
    global _POOL  # noqa: PLW0603
    _POOL = SfeWorkers(2)
    _POOL.start()


def tearDownModule() -> None:  # noqa: N802 (unittest's name)
    if _POOL is not None:
        _POOL.close()


def _pool() -> SfeWorkers:
    assert _POOL is not None
    return _POOL


def said(value: Any) -> Any:
    """``value`` as plain data, every array by its type, shape and bytes and every dict in its order: what equal means
    here is the same bits in the same order."""
    if isinstance(value, np.ndarray):
        return ["array", value.dtype.str, list(value.shape), value.tobytes().hex()]
    if isinstance(value, (np.floating, float)):
        return ["float", np.float64(value).tobytes().hex()]
    if isinstance(value, (np.integer, int, bool, str)) or value is None:
        return value if not isinstance(value, np.integer) else int(value)
    if isinstance(value, enum.Enum):
        return ["enum", type(value).__name__, said(value.value)]
    if isinstance(value, dict):
        return ["dict", [[said(key), said(item)] for key, item in value.items()]]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [said(item) for item in value]]
    if dataclasses.is_dataclass(value):
        return [type(value).__name__, [[f.name, said(getattr(value, f.name))] for f in dataclasses.fields(value)]]
    return ["repr", repr(value)]


def _result(result: Any) -> str:
    return json.dumps([said(result.candidates), said(result.reasons), said(result.telemetry), said(result.top_score)])


# ---------------------------------------------------------------------------------------------------------------------
# The recorded looks, through the cell's calculator
# ---------------------------------------------------------------------------------------------------------------------


def _zollstock(name: str, *, workers: "SfeWorkers | None") -> Any:
    """``test_a_recorded_zollstock_look_is_boxed_in``'s replay of look ``name`` with the scene, the support model of the
    look and side approaches, its SFE handed ``workers``."""
    from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry
    from src.robot.execution.autonomous_grasp.config import _profile_for, resolve_grasp_mode
    from src.robot.grasping.calculator_factory import build_calculator
    from tests._cell_2026_10_01 import look
    from tests.test_a_recorded_zollstock_look_is_boxed_in import _cell, _model, _rules, _support

    recorded = look(name)
    cfg = _cell()
    calc = build_calculator(cfg, camera_matrix=recorded.intrinsics, max_grip_width_mm=cfg.gripper.max_width_mm,
                            min_grip_width_mm=cfg.gripper.min_width_mm, support_footprint_geometry=True,
                            support_footprint_inflate_mm=0.0, side_approaches=True, scene_obstacles=_rules())
    return calc.compute_result(
        SimpleNamespace(mask=recorded.target_mask, label="Einen Zollstock", score=1.0), recorded.depth_mm,
        pixel_to_mm=None, dense_sampling=True,
        grasp_sampling_mode=_profile_for(resolve_grasp_mode(cfg.grasping.default_mode)).sampling_mode,
        other_object_masks=[],
        camera_to_base=Transform.from_matrix(recorded.camera_to_base, from_frame=Frame.CAMERA, to_frame=Frame.BASE),
        support_plane=_support(recorded).plane, min_table_clearance_mm=cfg.grasping.support.min_clearance_mm,
        gripper_model=build_gripper_geometry(cfg.grasping.gripper_geometry), support_model=_model(recorded),
        sfe_workers=workers)


class TheRecordedLooksTests(unittest.TestCase):
    def test_the_zollstock_looks_get_the_one_processs_answer_from_a_pool(self) -> None:
        for name in ("P1", "P5"):
            with self.subTest(look=name):
                here = _zollstock(name, workers=None)
                pooled = _zollstock(name, workers=_pool())
                self.assertIn("support_footprint_refused", here.telemetry)
                self.assertEqual(_result(here), _result(pooled))


# ---------------------------------------------------------------------------------------------------------------------
# A frame of several parts, through the pick loop
# ---------------------------------------------------------------------------------------------------------------------


def _ranked(*, workers: "SfeWorkers | None") -> tuple[Any, ...]:
    """A ray-cast look 45 degrees over four parts (a 58 mm cube the Hand-E cannot close across and three 40 mm cubes)
    ranked by the pick loop on the cell's calculator, its overlay drawn: what it chose and everything it kept."""
    from src.config.schema.robot.grasping_schema import GraspingSupportConfig
    from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry
    from src.robot.grasping.calculator_factory import build_calculator
    from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator
    from src.robot.grasping.motion.frame_resolver import StaticCameraToBaseResolver
    from src.robot.grasping.types.perception import PerceptionFrame
    from tests._helpers import _FakeArm
    from tests._wrist_views import K, Box, camera_looking_at, render
    from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import hande_cell, owner_rules

    cfg = hande_cell()
    looking = camera_looking_at((0.0, -700.0, 20.0), bearing_deg=90.0, elevation_deg=45.0, range_mm=520.0)
    boxes = (Box((-110.0, -735.0, 0.0), (-52.0, -677.0, 58.0), "part"),
             Box((-20.0, -720.0, 0.0), (20.0, -680.0, 40.0), "part"),
             Box((50.0, -700.0, 0.0), (90.0, -660.0, 40.0), "part"),
             Box((20.0, -790.0, 0.0), (60.0, -750.0, 40.0), "part"))
    depth, hit = render(looking, boxes)
    rgb = np.full(depth.shape + (3,), 90, dtype=np.uint8)
    segmentations = tuple(SimpleNamespace(mask=hit == i, label="part", score=0.9, name=f"part {i}")
                          for i in range(len(boxes)) if np.any(hit == i))
    frame = PerceptionFrame(depth_map=depth, intrinsics=K.copy(), timestamp=1.0, segmentations=segmentations, rgb=rgb)
    calc = build_calculator(cfg, camera_matrix=K, max_grip_width_mm=cfg.gripper.max_width_mm,
                            min_grip_width_mm=cfg.gripper.min_width_mm, support_footprint_geometry=True,
                            support_footprint_inflate_mm=0.0, side_approaches=True, scene_obstacles=owner_rules())
    calc.render_debug_images = True
    orch = BinPickingOrchestrator(
        arm=_FakeArm(), calculator=calc, perception=None, max_attempts=1,  # type: ignore[arg-type]
        gripper_model=build_gripper_geometry(cfg.grasping.gripper_geometry), sfe_workers=workers,
        frame_resolver=StaticCameraToBaseResolver(transform=Transform.from_matrix(
            looking, from_frame=Frame.CAMERA, to_frame=Frame.BASE)),
        support_config=GraspingSupportConfig(height_mm=0.0, refine_from_target=False,
                                             min_clearance_mm=float(cfg.grasping.support.min_clearance_mm)))
    result, index, _decision = orch._best_result_over_segmentations(frame)
    every = {idx: _result(ranked.result) for idx, ranked in sorted(orch._results_by_object.items())}
    return (index, _result(result), every, json.dumps(said(orch._support_telemetry)), calc.last_debug_image_png,
            json.dumps(said(calc.last_telemetry)))


class AFrameOfSeveralPartsTests(unittest.TestCase):
    def test_the_pick_loop_chooses_the_same_part_with_the_same_telemetry_and_overlay(self) -> None:
        here = _ranked(workers=None)
        pooled = _ranked(workers=_pool())
        self.assertIsNotNone(here[0], "the look's best part")
        self.assertGreater(len(here[2]), 1, "every part of the look was computed")
        self.assertIsNotNone(here[4], "the overlay was drawn")
        self.assertEqual(here[0], pooled[0], "the part chosen")
        self.assertEqual(here[1], pooled[1], "its result")
        self.assertEqual(here[2], pooled[2], "every part's result")
        self.assertEqual(here[3], pooled[3], "the support telemetry")
        self.assertEqual(here[4], pooled[4], "the overlay")
        self.assertEqual(here[5], pooled[5], "the last telemetry")


# ---------------------------------------------------------------------------------------------------------------------
# A pool that breaks
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Trap:
    """A seen test that answers "seen" in the cell's process and, in a worker, does ``what``: ``raise``, ``die`` or
    ``hang``. ``talk`` logs, warns and prints from a worker instead, and answers "seen" there too."""

    what: str

    def __call__(self, points: np.ndarray) -> np.ndarray:
        from src.robot.grasping import workers

        if workers._IN_A_WORKER:
            if self.what == "raise":
                raise RuntimeError("a seen test broke in a worker")
            if self.what == "die":
                os._exit(3)
            if self.what == "hang":
                time.sleep(30.0)
            if self.what == "talk":
                import logging
                import warnings

                logging.getLogger("src.robot.grasping.workers").error("a worker spoke")
                logging.getLogger().critical("a worker spoke at the root")
                warnings.warn("a worker warned")
        return np.ones(np.asarray(points).reshape(-1, 3).shape[0], dtype=bool)


def _beside_a_wall(seen: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """The cylinder beside its wall with side approaches, whose tilted builds ask ``seen``."""
    from tests.test_sfe_says_why_it_refused import cylinder_beside_a_wall, hande_jaw

    cylinder, wall = cylinder_beside_a_wall(12.0)
    return cylinder, dict(support_height_mm=0.0, jaw=hande_jaw(), obstacle_points_base_mm=wall,
                          side_approaches=True, corridor_seen=seen)


def _grasps(cloud: np.ndarray, keywords: dict[str, Any], runner: Any = None) -> str:
    counts: dict[str, int] = {}
    found = generate_support_footprint_grasps(cloud, refusals=counts, runner=runner, **keywords)
    return json.dumps([said(found), said(counts)])


class APoolThatBreaksTests(unittest.TestCase):
    def _falls_back_once(self, what: str, **pool: Any) -> SfeWorkers:
        cloud, keywords = _beside_a_wall(_Trap(what))
        workers = SfeWorkers(1, **pool)
        self.addCleanup(workers.close)
        workers.start()
        with self.assertLogs("src.robot.grasping.workers", "WARNING") as told:
            got = _grasps(cloud, keywords, workers)
        self.assertEqual(_grasps(cloud, keywords), got, "the part searched in the cell's process")
        self.assertEqual(1, len(told.records), told.output)
        self.assertIn("this part is searched in the cell's process", told.output[0])
        return workers

    def test_a_worker_that_raises_has_the_part_searched_here_said_once(self) -> None:
        workers = self._falls_back_once("raise")
        self.assertFalse(workers.running, "a pool that broke is not used again as it was")

    def test_a_worker_that_dies_has_the_part_searched_here_and_the_next_part_starts_the_pool_again(self) -> None:
        workers = self._falls_back_once("die")
        cloud, keywords = _beside_a_wall(_Trap("seen"))
        with self.assertNoLogs("src.robot.grasping.workers", "WARNING"):
            self.assertEqual(_grasps(cloud, keywords), _grasps(cloud, keywords, workers))
        self.assertTrue(workers.running)
        self.assertEqual("", workers.given_up)

    def test_a_worker_that_does_not_answer_has_the_part_searched_here_after_the_timeout(self) -> None:
        started = time.monotonic()
        workers = self._falls_back_once("hang", timeout_s=1.0)
        self.assertLess(time.monotonic() - started, 25.0, "the hung worker was waited for")
        self.assertFalse(workers.running)

    def test_a_pool_that_breaks_twice_in_a_row_is_given_up_and_every_part_is_searched_here(self) -> None:
        cloud, keywords = _beside_a_wall(_Trap("raise"))
        workers = SfeWorkers(1)
        self.addCleanup(workers.close)
        with self.assertLogs("src.robot.grasping.workers", "WARNING") as told:
            _grasps(cloud, keywords, workers)
            _grasps(cloud, keywords, workers)
            seen = _beside_a_wall(_Trap("seen"))
            self.assertEqual(_grasps(*seen), _grasps(seen[0], seen[1], workers))
        self.assertEqual(3, len(told.records), told.output)
        self.assertIn("given up", told.output[1])
        self.assertIn("given up", told.output[2])
        self.assertTrue(workers.given_up)
        self.assertFalse(workers.running)

    def test_inputs_that_do_not_pickle_are_searched_here_said_once(self) -> None:
        cloud, keywords = _beside_a_wall(lambda points: np.ones(np.asarray(points).reshape(-1, 3).shape[0], bool))
        with self.assertLogs("src.robot.grasping.workers", "WARNING") as told:
            got = _grasps(cloud, keywords, _pool())
        self.assertEqual(_grasps(cloud, keywords), got)
        self.assertEqual(1, len(told.records))
        self.assertIn("could not be handed over", told.output[0])
        self.assertTrue(_pool().running, "inputs that do not pickle break no worker")


# ---------------------------------------------------------------------------------------------------------------------
# What a worker says, and its threads
# ---------------------------------------------------------------------------------------------------------------------


def _run_python(code: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=str(ROOT), capture_output=True,
                          text=True, timeout=300, env={**os.environ, **env})


def talking_worker() -> str:
    """A pool of one searches a part whose seen test logs, warns and prints in its worker: the grasps, as text."""
    workers = SfeWorkers(1)
    try:
        workers.start()
        cloud, keywords = _beside_a_wall(_Trap("talk"))
        return _grasps(cloud, keywords, workers)
    finally:
        workers.close()


def grasps_here() -> str:
    """A boxed-in bar's grasps and counts, searched in this process, as text."""
    from tests.test_sfe_says_why_it_refused import boxed_in_bar, hande_jaw, MAT_MM

    bar, pile = boxed_in_bar()
    return _grasps(bar, dict(support_height_mm=MAT_MM, jaw=hande_jaw(), obstacle_points_base_mm=pile))


class AWorkerSaysNothingTests(unittest.TestCase):
    def test_a_worker_writes_no_log_line_and_warns_of_nothing(self) -> None:
        done = _run_python("""
            import logging
            from tests.test_a_pool_of_workers_grasps_as_one_process_does import talking_worker
            logging.basicConfig(level=logging.DEBUG)
            print(len(talking_worker()))
        """)
        self.assertEqual(0, done.returncode, done.stderr)
        self.assertNotIn("a worker spoke", done.stderr + done.stdout)
        self.assertNotIn("a worker warned", done.stderr + done.stdout)
        self.assertIn("SFE workers: 1 ready", done.stderr, "the cell's process says it, the worker never")


class AWorkerNeverRunsTheProgramTests(unittest.TestCase):
    def test_a_program_without_a_main_guard_runs_once_and_its_workers_only_search(self) -> None:
        """A spawned process runs its parent's main module again; a worker runs the worker loop alone. A program that
        built a cell and moved the arm at its top level would otherwise build it and move it in every worker."""
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            program = Path(folder) / "unguarded_program.py"
            program.write_text(textwrap.dedent(f"""
                import sys
                sys.path.insert(0, {str(ROOT)!r})
                print("the program ran", flush=True)
                from tests.test_a_pool_of_workers_grasps_as_one_process_does import _Trap, _beside_a_wall, _grasps
                from src.robot.grasping.workers import SfeWorkers
                workers = SfeWorkers(2)
                workers.start()
                cloud, keywords = _beside_a_wall(_Trap("seen"))
                print("same" if _grasps(cloud, keywords, workers) == _grasps(cloud, keywords) else "different")
                workers.close()
            """), encoding="utf-8")
            done = subprocess.run([sys.executable, str(program)], cwd=folder, capture_output=True, text=True,
                                  timeout=300)
        self.assertEqual(0, done.returncode, done.stderr)
        self.assertEqual(1, done.stdout.count("the program ran"), done.stdout)
        self.assertIn("same", done.stdout)


class TheBlasThreadsChangeNoGraspTests(unittest.TestCase):
    def test_one_blas_thread_or_four_give_the_same_grasps(self) -> None:
        code = """
            from tests.test_a_pool_of_workers_grasps_as_one_process_does import grasps_here
            print(grasps_here())
        """
        answers = []
        for threads in ("1", "4"):
            done = _run_python(code, OPENBLAS_NUM_THREADS=threads, OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
            self.assertEqual(0, done.returncode, done.stderr)
            answers.append(done.stdout.strip().splitlines()[-1])
        self.assertTrue(answers[0])
        self.assertEqual(answers[0], answers[1])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
