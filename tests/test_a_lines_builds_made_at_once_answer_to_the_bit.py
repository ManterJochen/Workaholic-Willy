"""A closing line's builds made at once answer what they answer one at a time, to the bit.

The owner's cell, 2026-10-08: one boxed-in cube took 24 of a look's 30 s, the support-footprint stage making every grasp
it tried one after another, each in numpy's overhead on 3-vectors. ``batched`` (``robot.grasping.batched_builds``) makes
each closing line's builds at once (``support_footprint._build_many``): ``_build``'s checks in its order over every
build still standing, every number a candidate carries made one build at a time through a stacked ``np.matmul``, every
number only a verdict reads held 1e-6 off its threshold, and a build nearer one made by ``_build`` alone.

Here the five scenes of the units' test, the seventeen golden scenes and the owner's two recorded Zollstock looks get
the same candidates (every field, a signed zero included), the same refusal counts in the same order and the same stages
with it as without, under every switch the search has. A seen test, an envelope or a floor of another kind is asked one
build at a time and answers the same; without every tilt the ladder makes exactly the builds the loop makes; one BLAS
thread or eight give the same grasps; and a pool's workers make their units' builds at once too.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from src.robot.grasping.generation import support_footprint as sf
from src.robot.grasping.generation.support_footprint import generate_support_footprint_grasps
from tests.test_a_pool_of_workers_grasps_as_one_process_does import said
from tests.test_sfe_says_why_it_refused import box_cloud
from tests.test_sfe_units_in_any_order_give_the_search_its_answer import scenes

ROOT = Path(__file__).resolve().parents[1]


def _grasps(cloud: np.ndarray, keywords: dict[str, Any], *, batched: bool, counting: bool = True,
            **more: Any) -> tuple[list[Any], "dict[str, int] | None", dict[str, str]]:
    counts: "dict[str, int] | None" = {} if counting else None
    stages: dict[str, str] = {}
    found = generate_support_footprint_grasps(cloud, refusals=counts, stages=stages, batched=batched,
                                              **{**keywords, **more})
    return found, counts, stages


def _same(test: unittest.TestCase, expected: tuple[Any, ...], got: tuple[Any, ...], what: str) -> None:
    """Every field of every candidate bit for bit, the counts with their order, and the stages."""
    test.assertEqual(json.dumps(said(expected[0])), json.dumps(said(got[0])), f"{what}: the candidates")
    test.assertEqual(expected[1], got[1], f"{what}: the counts")
    test.assertEqual(list(expected[1] or {}), list(got[1] or {}), f"{what}: the counts' order")
    test.assertEqual(expected[2], got[2], f"{what}: the stages")


def _a_wall_beside(cloud: np.ndarray) -> np.ndarray:
    """A declared wall 18 mm off the part's +y side, as a container's wall is handed to SFE."""
    lo, hi = cloud.min(axis=0), cloud.max(axis=0)
    xs, zs = np.meshgrid(np.arange(lo[0] - 30.0, hi[0] + 30.0, 3.0), np.arange(lo[2], hi[2] + 40.0, 3.0))
    return np.column_stack([xs.ravel(), np.full(xs.size, hi[1] + 18.0), zs.ravel()])


#: Every switch the search has, as keywords over a scene's own: ``None`` where it does not apply to the scene.
def _variants(keywords: dict[str, Any], cloud: np.ndarray) -> dict[str, "dict[str, Any] | None"]:
    return {
        "as it is": {},
        "side approaches on": None if keywords.get("side_approaches") else {"side_approaches": True},
        "side approaches off": {"side_approaches": False} if keywords.get("side_approaches") else None,
        "no seen test": {"corridor_seen": None} if keywords.get("corridor_seen") is not None else None,
        "no camera boxes": {"seen_envelope": None} if keywords.get("seen_envelope") is not None else None,
        "no floor": {"hand_floor": None} if keywords.get("hand_floor") is not None else None,
        "palm aware": {"palm_aware": True},
        "inflated 2 mm": {"inflate_mm": 2.0},
        "its own weights": {"score_weights": (0.7, 0.1, 0.1, 0.05, 0.05)},
        "a declared wall": (None if keywords.get("rigid_obstacle_points_base_mm") is not None
                            else {"rigid_obstacle_points_base_mm": _a_wall_beside(cloud)}),
        "the fine pass left for later": {"fine_pass": False},
    }


class EveryScenesAnswerIsTheSameTests(unittest.TestCase):
    def test_every_scene_under_every_switch_gets_the_same_grasps_counts_and_stages(self) -> None:
        for name, (cloud, keywords) in scenes().items():
            for variant, more in _variants(keywords, cloud).items():
                if more is None:
                    continue
                with self.subTest(scene=name, switch=variant):
                    _same(self, _grasps(cloud, keywords, batched=False, **more),
                          _grasps(cloud, keywords, batched=True, **more), f"{name}, {variant}")

    def test_uncounted_the_grasps_are_the_same_too(self) -> None:
        for name, (cloud, keywords) in scenes().items():
            with self.subTest(scene=name):
                _same(self, _grasps(cloud, keywords, batched=False, counting=False),
                      _grasps(cloud, keywords, batched=True, counting=False), name)

    def test_the_seventeen_golden_scenes_keep_their_digests(self) -> None:
        import tests.test_sfe_says_why_it_refused as golden

        real = golden.generate_support_footprint_grasps
        for side in (False, True):
            runs: dict[bool, dict[str, list[Any]]] = {}
            for batched in (False, True):
                def run(*args: Any, _batched: bool = batched, **keywords: Any) -> list[Any]:
                    return real(*args, **keywords, side_approaches=side, batched=_batched)

                with mock.patch.object(golden, "generate_support_footprint_grasps", side_effect=run):
                    runs[batched] = golden.golden_runs()
            for name, expected in runs[False].items():
                with self.subTest(scene=name, side_approaches=side):
                    self.assertEqual(json.dumps(said(expected)), json.dumps(said(runs[True][name])))
                    if not side:
                        self.assertEqual(golden.HEAD_DIGESTS[name], golden.candidate_digest(runs[True][name]))

    def test_the_owners_recorded_zollstock_looks_get_the_same_result_from_the_cells_calculator(self) -> None:
        """``test_a_recorded_zollstock_look_is_boxed_in``'s replay with the scene, the look's support model and side
        approaches, on a calculator the cell's tree builds with the key and without it: the candidates, reasons, counts
        and every telemetry key, bit for bit."""
        from tests.test_a_pool_of_workers_grasps_as_one_process_does import _result

        for name in ("P1", "P5"):
            with self.subTest(look=name):
                here, calculator = zollstock(name, batched=False)
                self.assertFalse(calculator.sfe_batched)
                at_once, calculator = zollstock(name, batched=True)
                self.assertTrue(calculator.sfe_batched, "the key reached the calculator")
                self.assertIn("support_footprint_refused", here.telemetry)
                self.assertEqual(_result(here), _result(at_once))


class AFrameOfSeveralPartsTests(unittest.TestCase):
    def test_the_pick_loop_chooses_the_same_part_with_the_same_telemetry_and_overlay(self) -> None:
        """``test_a_pool_of_workers_grasps_as_one_process_does``'s ray-cast look over four parts (a 58 mm cube the
        Hand-E cannot close across and three 40 mm cubes), ranked by the pick loop on the cell's calculator, its
        support-footprint stage made to build at once."""
        from src.robot.grasping.generation import calculator as calculator_module
        from tests.test_a_pool_of_workers_grasps_as_one_process_does import _ranked

        stage = calculator_module.support_footprint_breakdowns

        def at_once(*args: Any, **keywords: Any) -> Any:
            return stage(*args, **{**keywords, "batched": True})

        here = _ranked(workers=None)
        with mock.patch.object(calculator_module, "support_footprint_breakdowns", side_effect=at_once) as asked:
            batched = _ranked(workers=None)
        self.assertGreater(asked.call_count, 1, "every part of the look was ranked at once")
        self.assertIsNotNone(here[0], "the look's best part")
        for k, what in enumerate(("the part chosen", "its result", "every part's result", "the support telemetry",
                                  "the overlay", "the last telemetry")):
            self.assertEqual(here[k], batched[k], what)


def zollstock(name: str, *, batched: bool) -> tuple[Any, Any]:
    """The recorded look ``name`` through the calculator the ``hande`` tree builds with ``batched_builds`` as asked,
    as ``test_a_pool_of_workers_grasps_as_one_process_does`` replays it: the result, and the calculator."""
    from types import SimpleNamespace

    from src.geometry import Frame, Transform
    from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry
    from src.robot.execution.autonomous_grasp.config import _profile_for, resolve_grasp_mode
    from src.robot.grasping.calculator_factory import build_calculator
    from tests._cell_2026_10_01 import look
    from tests.test_a_recorded_zollstock_look_is_boxed_in import _cell, _model, _rules, _support

    recorded = look(name)
    cfg = _cell()
    cfg = cfg.model_copy(update={"grasping": cfg.grasping.model_copy(update={"batched_builds": batched})})
    calc = build_calculator(cfg, camera_matrix=recorded.intrinsics, max_grip_width_mm=cfg.gripper.max_width_mm,
                            min_grip_width_mm=cfg.gripper.min_width_mm, support_footprint_geometry=True,
                            support_footprint_inflate_mm=0.0, side_approaches=True, scene_obstacles=_rules())
    result = calc.compute_result(
        SimpleNamespace(mask=recorded.target_mask, label="Einen Zollstock", score=1.0), recorded.depth_mm,
        pixel_to_mm=None, dense_sampling=True,
        grasp_sampling_mode=_profile_for(resolve_grasp_mode(cfg.grasping.default_mode)).sampling_mode,
        other_object_masks=[],
        camera_to_base=Transform.from_matrix(recorded.camera_to_base, from_frame=Frame.CAMERA, to_frame=Frame.BASE),
        support_plane=_support(recorded).plane, min_table_clearance_mm=cfg.grasping.support.min_clearance_mm,
        gripper_model=build_gripper_geometry(cfg.grasping.gripper_geometry), support_model=_model(recorded))
    return result, calc


# ---------------------------------------------------------------------------------------------------------------------
# What SFE does not know is asked one build at a time
# ---------------------------------------------------------------------------------------------------------------------


class _Asked:
    """Counts how often it is asked, and answers as what it wraps answers."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.asked = 0


class _AnEnvelope(_Asked):
    def refuses(self, position: np.ndarray, rotation: np.ndarray) -> bool:
        self.asked += 1
        return bool(self.inner.refuses(position, rotation))


class _AFloor(_Asked):
    def under(self, points: np.ndarray, *, fingers: bool = False) -> bool:
        self.asked += 1
        return bool(self.inner.under(points, fingers=fingers))

    def at(self, xy: np.ndarray, *, fingers: bool = False) -> np.ndarray:
        return self.inner.at(xy, fingers=fingers)

    @property
    def highest_mm(self) -> float:
        return float(self.inner.highest_mm)


class _ASeenTest(_Asked):
    def __call__(self, points: np.ndarray) -> np.ndarray:
        self.asked += 1
        return np.asarray(self.inner(points))


class WhatSfeDoesNotKnowTests(unittest.TestCase):
    def _asked(self, scene: str, key: str, kind: type, **more: Any) -> None:
        cloud, keywords = scenes()[scene]
        one, many = kind(keywords[key]), kind(keywords[key])
        expected = _grasps(cloud, keywords, batched=False, **{key: one}, **more)
        made_alone = mock.patch.object(sf, "_build", wraps=sf._build)
        with made_alone as alone:
            got = _grasps(cloud, keywords, batched=True, **{key: many}, **more)
        _same(self, expected, got, f"{scene}, {kind.__name__}")
        self.assertGreater(one.asked, 0, "the scene asks it")
        # Asked once per build as ``_build`` asks it; a build the band leaves to ``_build`` may be asked again there.
        self.assertGreaterEqual(many.asked, one.asked)
        self.assertLessEqual(many.asked, one.asked + 2 * alone.call_count)

    def test_an_envelope_of_another_kind_is_asked_one_build_at_a_time_and_answers_the_same(self) -> None:
        # Without the pile's points in the grid, the builds come as far as the camera's boxes.
        self._asked("a bar boxed in", "seen_envelope", _AnEnvelope, obstacle_points_base_mm=None)

    def test_a_floor_of_another_kind_is_asked_one_build_at_a_time_and_answers_the_same(self) -> None:
        self._asked("a bar boxed in", "hand_floor", _AFloor)

    def test_a_seen_test_of_another_kind_is_asked_one_build_at_a_time_and_answers_the_same(self) -> None:
        self._asked("a cylinder beside a wall", "corridor_seen", _ASeenTest)

    def test_a_seen_test_that_answers_for_the_wrong_number_of_points_is_still_refused(self) -> None:
        cloud, keywords = scenes()["a cylinder beside a wall"]
        with self.assertRaises(ValueError):
            _grasps(cloud, keywords, batched=True, corridor_seen=lambda points: np.ones(len(points) + 1, dtype=bool))


# ---------------------------------------------------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------------------------------------------------


class TheLadderMakesTheLoopsBuildsTests(unittest.TestCase):
    def test_the_rungs_make_exactly_the_builds_the_loop_makes(self) -> None:
        """Without every tilt the loop ends a height's ladder at the first tilt that found a grasp; the rungs are made
        one tilt at a time over the heights still open, so the builds made are the loop's, no more."""
        for name, (cloud, keywords) in scenes().items():
            for more in ({}, {"side_approaches": False}, {"corridor_seen": None}):
                with self.subTest(scene=name, switch=more):
                    with mock.patch.object(sf, "_build", wraps=sf._build) as alone:
                        _grasps(cloud, keywords, batched=False, **more)
                    made = []
                    real = sf._build_many

                    def counted(plan: Any, anchors: np.ndarray, ladder: Any, rows: np.ndarray, **kw: Any) -> Any:
                        made.append(int(rows.size))
                        return real(plan, anchors, ladder, rows, **kw)

                    with mock.patch.object(sf, "_build_many", side_effect=counted):
                        _grasps(cloud, keywords, batched=True, **more)
                    self.assertEqual(alone.call_count, sum(made))


# ---------------------------------------------------------------------------------------------------------------------
# Threads and workers
# ---------------------------------------------------------------------------------------------------------------------


def batched_answers() -> str:
    """The boxed-in bar's and the cylinder's grasps and counts, built at once in this process, as text."""
    out = []
    for name in ("a bar boxed in", "a cylinder beside a wall"):
        cloud, keywords = scenes()[name]
        found, counts, stages = _grasps(cloud, keywords, batched=True)
        out.append([said(found), said(counts), said(stages)])
    return json.dumps(out)


class OneThreadOrEightTests(unittest.TestCase):
    def test_one_blas_thread_or_eight_give_the_same_grasps(self) -> None:
        answers = []
        for threads in ("1", "8"):
            done = subprocess.run(
                [sys.executable, "-c", textwrap.dedent("""
                    from tests.test_a_lines_builds_made_at_once_answer_to_the_bit import batched_answers
                    print(batched_answers())
                """)], cwd=str(ROOT), capture_output=True, text=True, timeout=600,
                env={**os.environ, "OPENBLAS_NUM_THREADS": threads, "OMP_NUM_THREADS": threads,
                     "MKL_NUM_THREADS": threads})
            self.assertEqual(0, done.returncode, done.stderr)
            answers.append(done.stdout.strip().splitlines()[-1])
        self.assertEqual(answers[0], answers[1])
        self.assertEqual(batched_answers(), answers[0], "and the same as this process's")


class APoolsWorkersBuildAtOnceTests(unittest.TestCase):
    def test_a_worker_makes_its_units_builds_at_once(self) -> None:
        from src.robot.grasping.workers import SfeWorkers, _WorkerPlans

        cloud, keywords = scenes()["a bar boxed in"]
        inputs = sf.SfeInputs(cloud, counting=True, batched=True, **keywords)
        plan = sf.plan_support_footprint(inputs)
        assert plan is not None
        pool = SfeWorkers(1)   # never started: only its shared memory is used, as a worker reads it
        self.addCleanup(pool.close)
        key, payload = pool._shared(plan)
        units = plan.units(sf.COARSE)
        with mock.patch.object(sf, "_run_units_many", wraps=sf._run_units_many) as many:
            found, counted = _WorkerPlans().run(key, payload, units)
        self.assertEqual(1, many.call_count, "the worker's units were built at once")
        expected: dict[str, int] = {}
        planned = sf.plan_support_footprint(sf.SfeInputs(cloud, counting=True, **keywords))
        assert planned is not None
        alone = sf.run_units(planned, units, expected)
        self.assertEqual(json.dumps(said(alone)), json.dumps(said(found)))
        self.assertEqual(expected, counted)

    def test_a_line_is_never_cut_between_two_chunks_of_work(self) -> None:
        from src.robot.grasping.workers import _chunks

        cloud, keywords = scenes()["a bar boxed in"]
        plan = sf.plan_support_footprint(sf.SfeInputs(cloud, **keywords))
        assert plan is not None
        for units in (plan.units(sf.COARSE), plan.units(sf.FINE) + plan.units(sf.ROLLED)):
            for workers in (1, 2, 7, 8, 30):
                with self.subTest(units=len(units), workers=workers):
                    chunks = _chunks(units, workers)
                    self.assertEqual(list(units), [unit for chunk in chunks for unit in chunk], "in order, each once")
                    for before, after in zip(chunks, chunks[1:]):
                        self.assertNotEqual(before[-1][:3], after[0][:3], "a line cut between two chunks")
                    self.assertLessEqual(len(chunks), workers * 4)

    def test_a_pool_of_two_gets_the_one_processs_answer(self) -> None:
        from src.robot.grasping.workers import SfeWorkers

        pool = SfeWorkers(2)
        self.addCleanup(pool.close)
        pool.start()
        for name in ("a bar boxed in", "a cylinder beside a wall", "an open cube"):
            cloud, keywords = scenes()[name]
            with self.subTest(scene=name):
                _same(self, _grasps(cloud, keywords, batched=False),
                      _grasps(cloud, keywords, batched=True, runner=pool), name)


class AnOpenCubeIsFasterAndTheSameTests(unittest.TestCase):
    def test_a_cube_alone_on_the_mat_gets_its_grasps(self) -> None:
        """The plainest scene: every line closes at the first tilt."""
        cloud = box_cloud((-20.0, -20.0, 0.0), (20.0, 20.0, 40.0))
        keywords: dict[str, Any] = {"support_height_mm": 0.0}
        expected = _grasps(cloud, keywords, batched=False)
        self.assertTrue(expected[0])
        _same(self, expected, _grasps(cloud, keywords, batched=True), "an open cube")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
