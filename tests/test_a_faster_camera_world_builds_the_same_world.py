"""The camera world builds today's cell exactly as it built it before its build was made fast.

On the cell PC every judged motion refreshed the planner world, and the log of 2026-10-07 said what each one cost:
``planner world refreshed: 65 box(es), frame 80 ms old, 1097.0 ms to build, 154.7 ms to register; 109 box(es) merged``,
1.0 to 1.6 s and more with a frame of an earlier pose held. The build was then made faster without changing what it
builds. This holds it to that: the same boxes, each where it was, as large and as turned, with the same label, the same
soft fingers and the same finger top; the same supports and their solids; the same counts of everything left out.

Three of the cell's own frames of 2026-10-07 (``tests/data/cell_2026_10_07``, depth only, as the looks of 2026-10-01
are kept): the wrist D415 over the owner's pile of parts on the mat, at 1280 x 720, at the look its pick took it from.
F1 (``pick-3a861468af1b``) merged 118 boxes to fit 128 slots, F2 (``pick-7a14ddb4b5ee``) stands on six supports and
merged 74, F3 (``pick-526dd4c2aa70``) fitted its slots without a merge. Each frame keeps the depth at every second
pixel, which is all a world built at ``pixel_stride`` 2 reads, in whole millimetres as the camera gave it; the target's
cloud the pick recorded; and stand-ins for the other parts its detector named: the frame's surface over the mat, the
target left out, in 60 mm tiles thinned to 5 mm, as the pick loop offers a named part.

Five cases a frame, built the way the cell builds its world (``camera_world_wiring._live_planner_world`` with the cell's
``planning_world.perceived`` of that day, ``LivePlannerWorld.world_for`` with the arm's own body):

* ``A`` the cell's 128 slots, the target held, the named parts offered, the arm where it took the frame and the motion
  going 100 mm over the target;
* ``B`` the same at 192 slots, which the cell ran for an hour that morning;
* ``C`` 64 slots and nothing held or named: the merging at its heaviest;
* ``D`` as ``A`` with the hand down at the target, its fingers about the part: the self filter takes the hand out of
  the frame, and what the hand hid is filled as high as what was seen beside it;
* ``E`` two views: the frame held by the pick at its look, and the same camera's frame with the hand down at the target,
  each without the robot where it stood.

``worlds_golden.json.gz`` is what the build of commit f771b9c, before any of the speed-ups, made of every case. Floats
are held to 1e-9 mm (of a rotation, 1e-9), every count, name and label exactly. A change that is MEANT to change the
world writes it again with ``python -m tests.test_a_faster_camera_world_builds_the_same_world --write-golden`` and says
why in its commit; a speed-up never does.

Beside the golden, each step that was made faster against the way it was done before, on clouds of every shape: the
clustering against the walk over the cells it replaced, the voxel thinning's first keys against ``np.unique``, and the
named parts against asking every part's tree for every box. And the same question asked twice of a real frame is
answered with the golden world, built once.
"""

from __future__ import annotations

import gzip
import json
import math
import sys
import time
import unittest
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

_DATA = Path(__file__).resolve().parent / "data" / "cell_2026_10_07"
_GOLDEN = _DATA / "worlds_golden.json.gz"
_FRAMES = {
    "F1": "f1_pick-3a861468af1b.npz",
    "F2": "f2_pick-7a14ddb4b5ee.npz",
    "F3": "f3_pick-526dd4c2aa70.npz",
}
#: ``(slots, the target held and the parts named, the hand down at the target, two views)`` per case.
_CASES = {
    "A": (128, True, False, False),
    "B": (192, True, False, False),
    "C": (64, False, False, False),
    "D": (128, True, True, False),
    "E": (128, True, True, True),
}
#: How far a float of the world may lie from the golden one: millimetres, radians and shares alike.
_TOLERANCE = 1e-9
#: When the frames were taken, on the world's clock, and when the world is asked: 50 ms later.
_SHUTTER_S = 1000.0
_ASKED_S = 1000.05


def _robot(slots: int) -> dict[str, Any]:
    """The owner's ``robot`` block as the cell ran it on 2026-10-07 (``robot.yaml`` from 10:27 on), as far as the
    camera world reads it: the UR10 and the Hand-E of ``_cell_2026_10_01``, today's workspace box and 3 mm distances,
    the bench slab about the base, and every key of ``planning_world.perceived`` written out, so no schema default the
    world reads can move under the golden."""
    from tests import _cell_2026_10_01 as cell

    robot = cell.owner_robot(
        support_plane={"height_mm": 0.0, "extent_mm": [900.0, 700.0], "thickness_mm": 50.0, "center_mm": [0.0, 0.0]},
        perceived={
            "enabled": True, "max_age_ms": 2000.0, "fresh_frame_attempts": 3, "pixel_stride": 2,
            "voxel_size_mm": 10.0, "voxel_field_mm": 0.0, "cluster_voxel_mm": 25.0, "min_points": 12,
            "margin_mm": 8.0, "fingers_touch_parts": True, "finger_contact_mm": 3.0,
            "fingers_to_the_support_reading": True, "finger_floor_mm": 1.0, "max_boxes": int(slots),
            "floor_to_plane": True, "plane_clearance_mm": 5.0, "support_surfaces": True, "support_allowance_mm": 2.0,
        },
    )
    robot["workspace_limits"] = {"x_min": -470.0, "x_max": 400.0, "y_min": -900.0, "y_max": -200.0, "z_min": 16.0,
                                 "z_max": 800.0}
    robot["safety"]["self_collision"].update(min_distance_mm=3.0, perceived_min_distance_mm=3.0, planner_margin_mm=3.0)
    return robot


@lru_cache(maxsize=None)
def _config(slots: int) -> Any:
    from src.config.schema.robot import RobotConfig

    return RobotConfig.model_validate(_robot(slots))


@lru_cache(maxsize=None)
def frame(name: str) -> dict[str, Any]:
    """One recorded frame: its depth at the frame's size, every second pixel the camera's and the three beside it
    repeating it, and what the pick knew at it."""
    with np.load(_DATA / _FRAMES[name]) as data:
        stride = int(data["stride"])
        shape = tuple(int(v) for v in data["frame_shape"])
        strided = data["surface_depth_strided_mm"].astype(np.float64)
        depth = np.repeat(np.repeat(strided, stride, axis=0), stride, axis=1)[: shape[0], : shape[1]]
        counts = data["named_counts"].astype(np.int64)
        named_points = data["named_points_base_mm"].astype(np.float64)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        named = tuple((str(label), named_points[start:start + count])
                      for label, start, count in zip(data["named_labels"], starts, counts))
        out = {
            "depth": depth, "intrinsics": data["intrinsics"].astype(np.float64),
            "camera_to_base": data["camera_to_base"].astype(np.float64),
            "tool_to_base": data["tool_to_base"].astype(np.float64),
            "look_rad": np.radians(data["joints_deg"].astype(np.float64)),
            "down_rad": data["grasp_joints_rad"].astype(np.float64),
            "target": data["target_points_base_mm"].astype(np.float64), "named": named,
            "goal": data["goal_tcp_mm"].astype(np.float64), "grasp": data["grasp_tcp_mm"].astype(np.float64),
        }
    for value in out.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return out


def _live_world(recorded: dict[str, Any], slots: int) -> Any:
    """The cell's live world over its wrist camera, which answers with the recorded frame stamped at its shutter."""
    from src.robot.execution.camera_world_wiring import _live_planner_world
    from src.robot.safety.planning.live_world import CameraView, DepthSnapshot

    class _Wrist:
        def grab_surface_depth(self) -> DepthSnapshot:
            return DepthSnapshot(depth_mm=recorded["depth"], intrinsics=recorded["intrinsics"], timestamp=_SHUTTER_S,
                                 tool_to_base_mm=recorded["tool_to_base"])

    camera_to_tool = np.linalg.inv(recorded["tool_to_base"]) @ recorded["camera_to_base"]
    return _live_planner_world(_config(slots), [CameraView(name="EIH_Cam", depth_source=_Wrist(),
                                                           camera_to_tool=camera_to_tool)])


def build(name: str, case: str) -> Any:
    """The snapshot the cell's live world answers in ``case`` of frame ``name``."""
    from tests import _cell_2026_10_01 as cell

    slots, held, down, two = _CASES[case]
    recorded = frame(name)
    owner = cell.owner_cell()
    world = _live_world(recorded, slots)
    if held:
        world.offer_segmentation(target_points_base_mm=recorded["target"], target_label="target", hold=True,
                                 timestamp=_SHUTTER_S, named_points_base_mm=recorded["named"])
    look, hand_down = owner.envelope(recorded["look_rad"]), owner.envelope(recorded["down_rad"])
    goal, grasp = recorded["goal"], recorded["grasp"]
    if two:
        world.hold_pick_views()
        world.world_for(self_envelope=look, near_point_mm=goal[:3, 3], now=_ASKED_S,
                        goal_keep_out=owner.goal_keep_out(goal))
        return world.world_for(self_envelope=hand_down, near_point_mm=grasp[:3, 3], now=_ASKED_S + 0.05,
                               goal_keep_out=owner.goal_keep_out(grasp))
    return world.world_for(self_envelope=hand_down if down else look, near_point_mm=goal[:3, 3], now=_ASKED_S,
                           goal_keep_out=owner.goal_keep_out(goal))


def _plain(value: Any) -> Any:
    """``value`` as JSON holds it: lists for tuples and arrays, Python numbers for numpy's."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value


def canonical(snapshot: Any) -> dict[str, Any]:
    """Everything the world built, every float as it came out: each box whole, the supports' surfaces and solids as
    they were fitted (not as their ``to_dict`` rounds them), and every count the report carries."""
    perceived = snapshot.perceived
    assert perceived is not None, snapshot.reason
    boxes = [
        {"name": box.name, "center_mm": box.center_mm, "dims_mm": box.dims_mm, "yaw_rad": box.yaw_rad,
         "rotation": box.rotation, "points": box.points, "distance_mm": box.distance_mm, "label": box.label,
         "hidden_cells": box.hidden_cells, "robot_gap_mm": box.robot_gap_mm, "kind": box.kind, "detail": box.detail,
         "soft_mm": box.soft_mm, "finger_top_mm": box.finger_top_mm}
        for box in perceived.boxes
    ]
    report = perceived.to_dict()
    report.pop("boxes")
    report.pop("supports", None)
    supports = perceived.supports
    model = None if supports is None else {
        "surfaces": [{"index": s.index, "name": s.name, "coef": s.coef, "tilt_deg": s.tilt_deg, "rms_mm": s.rms_mm,
                      "area_mm2": s.area_mm2, "width_mm": s.width_mm, "z_range_mm": s.z_range_mm,
                      "band_mm": s.band_mm, "solids": s.solids, "own_pixels": s.own_pixels}
                     for s in supports.surfaces],
        "solids": [{"name": s.name, "kind": s.kind, "surface": s.surface, "centre_mm": s.centre_mm,
                    "half_extents_mm": s.half_extents_mm, "rotation": s.rotation, "local_plane": s.local_plane,
                    "excess_mm": s.excess_mm, "band_mm": s.band_mm, "allowance_mm": s.allowance_mm, "cells": s.cells}
                   for s in supports.solids],
        "bench_top_mm": supports.bench_top_mm, "bench_band_mm": supports.bench_band_mm,
        "allowance_mm": supports.allowance_mm,
        "bench_reading": None if supports.bench_reading is None else supports.bench_reading.to_dict(),
        "left_out": supports.left_out, "stats": supports.stats,
    }
    return _plain({
        "verdict": str(snapshot.verdict), "boxes": boxes, "report": report, "supports": model,
        "goal_seen_by": snapshot.goal_seen_by, "held_views": snapshot.held_views,
        "keep_out": None if snapshot.keep_out is None else snapshot.keep_out.to_dict(),
    })


def differences(golden: Any, built: Any, where: str = "") -> list[str]:
    """Where ``built`` is not ``golden``: a float further than :data:`_TOLERANCE`, anything else not equal."""
    if isinstance(golden, float) and isinstance(built, (int, float)) and not isinstance(built, bool):
        if math.isnan(golden) and math.isnan(float(built)):
            return []
        return [] if abs(float(built) - golden) <= _TOLERANCE else [f"{where}: {golden!r} built {built!r}"]
    if isinstance(golden, dict) and isinstance(built, dict):
        if set(golden) != set(built):
            return [f"{where}: keys {sorted(golden)} built {sorted(built)}"]
        return [line for key in golden for line in differences(golden[key], built[key], f"{where}.{key}")]
    if isinstance(golden, list) and isinstance(built, list):
        if len(golden) != len(built):
            return [f"{where}: {len(golden)} item(s) built {len(built)}"]
        return [line for index, (a, b) in enumerate(zip(golden, built))
                for line in differences(a, b, f"{where}[{index}]")]
    return [] if golden == built else [f"{where}: {golden!r} built {built!r}"]


def write_golden() -> None:
    """Write the golden from the build of this tree (``--write-golden``)."""
    worlds: dict[str, Any] = {}
    for name in _FRAMES:
        for case in _CASES:
            started = time.perf_counter()
            worlds[f"{name}/{case}"] = canonical(build(name, case))
            print(f"{name}/{case}: {1000.0 * (time.perf_counter() - started):.0f} ms")
    with gzip.open(_GOLDEN, "wt", encoding="utf-8") as handle:
        json.dump(worlds, handle, sort_keys=True, separators=(",", ":"))


@lru_cache(maxsize=1)
def _golden() -> dict[str, Any]:
    with gzip.open(_GOLDEN, "rt", encoding="utf-8") as handle:
        return json.load(handle)


class AFasterCameraWorldBuildsTheSameWorldTests(unittest.TestCase):
    def test_every_case_of_every_frame_builds_the_golden_world(self) -> None:
        golden = _golden()
        self.assertEqual(sorted(golden), sorted(f"{name}/{case}" for name in _FRAMES for case in _CASES))
        for key in sorted(golden):
            name, case = key.split("/")
            with self.subTest(world=key):
                found = differences(golden[key], canonical(build(name, case)))
                self.assertEqual(found, [], f"{len(found)} difference(s), the first: {found[:5]}")

    def test_the_cases_build_what_they_are_meant_to(self) -> None:
        """The golden is worth holding to only where the cases reach the code they are named for."""
        golden = _golden()
        merged = {key: world["report"]["merged_to_fit"] for key, world in golden.items()}
        self.assertGreater(merged["F1/A"], 100)
        self.assertEqual(merged["F3/A"], 0)
        self.assertGreater(min(merged[f"{name}/C"] for name in _FRAMES), merged["F1/A"] // 2)
        for name in _FRAMES:
            down, two = golden[f"{name}/D"], golden[f"{name}/E"]
            self.assertGreater(down["report"]["dropped_points"]["self"], 0)
            self.assertEqual(two["held_views"], ["EIH_Cam (held 1)"])
            self.assertTrue(any(box["label"] for box in golden[f"{name}/A"]["boxes"]))
            self.assertTrue(any(box["soft_mm"] > 0.0 for box in golden[f"{name}/A"]["boxes"]))
            self.assertTrue(golden[f"{name}/A"]["supports"]["surfaces"])
        self.assertGreater(sum(golden[f"{name}/{case}"]["report"]["hidden_cells"] for name in _FRAMES
                               for case in "DE"), 0)

    def test_the_same_question_asked_again_is_answered_with_the_golden_world(self) -> None:
        """Case ``A`` of F1 asked twice, the arm's body placed afresh for the second question as the arm places it: the
        live world answers it with the world it built (``live_world._BuiltWorld``), which is the golden one."""
        from tests import _cell_2026_10_01 as cell

        recorded, owner = frame("F1"), cell.owner_cell()
        world = _live_world(recorded, 128)
        world.offer_segmentation(target_points_base_mm=recorded["target"], target_label="target", hold=True,
                                 timestamp=_SHUTTER_S, named_points_base_mm=recorded["named"])
        goal = recorded["goal"]
        first = world.world_for(self_envelope=owner.envelope(recorded["look_rad"]), near_point_mm=goal[:3, 3],
                                now=_ASKED_S, goal_keep_out=owner.goal_keep_out(goal))
        again = world.world_for(self_envelope=owner.envelope(recorded["look_rad"]), near_point_mm=goal[:3, 3],
                                now=_ASKED_S + 0.05, goal_keep_out=owner.goal_keep_out(goal))
        self.assertEqual((world.worlds_built, world.worlds_served), (1, 1))
        self.assertIs(again.perceived, first.perceived)
        self.assertEqual(differences(_golden()["F1/A"], canonical(again)), [])


def _clouds(seed: int, count: int) -> Any:
    """Clouds of every shape a step meets: uniform, piles about a few centres, points on a few grid cells, one point
    repeated, and spans from a millimetre to a hundred metres."""
    rng = np.random.default_rng(seed)
    for trial in range(count):
        n = int(rng.integers(1, 700))
        kind = trial % 5
        if kind == 0:
            points = rng.uniform(-500.0, 500.0, (n, 3))
        elif kind == 1:
            centres = rng.uniform(-300.0, 300.0, (int(rng.integers(1, 8)), 3))
            points = centres[rng.integers(0, centres.shape[0], n)] + rng.normal(0.0, 30.0, (n, 3))
        elif kind == 2:
            points = rng.integers(-6, 6, (n, 3)).astype(np.float64) * 25.0 + rng.uniform(0.0, 25.0, (n, 3))
        elif kind == 3:
            points = np.repeat(rng.uniform(-100.0, 100.0, (1, 3)), n, axis=0)
        else:
            points = rng.uniform(-1e5, 1e5, (n, 3)) * rng.choice([1.0, 1e-3], (n, 1))
        yield trial, points


class EachFasterStepDoesWhatItDidTests(unittest.TestCase):
    def test_the_clustering_numbers_every_cluster_as_the_walk_over_the_cells_did(self) -> None:
        from src.robot.safety.planning.perceived import _cluster, _cluster_walk

        for trial, points in _clouds(3, 500):
            for cell in (10.0, 25.0, 37.5):
                with self.subTest(trial=trial, cell=cell):
                    walked = _cluster_walk(np.floor(points / cell).astype(np.int64))
                    self.assertTrue(np.array_equal(_cluster(points, cell), walked))
        # A grid too wide for one integer key is walked.
        far = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [3e17, -3e17, 3e17], [3e17 + 10.0, -3e17, 3e17]])
        self.assertEqual(_cluster(far, 25.0).tolist(), [0, 0, 1, 1])

    def test_the_voxel_thinning_finds_each_first_point_as_unique_did(self) -> None:
        from src.robot.safety.planning.perceived import _first_and_place

        rng = np.random.default_rng(11)
        for trial in range(400):
            n = int(rng.integers(1, 5000))
            # Runs of equal keys, as the pixels of a row fall into one voxel, and keys met again further on.
            keys = np.repeat(rng.integers(-50, 50, n), rng.integers(1, 12, n))[:n].astype(np.int64)
            _, first, place = np.unique(keys, return_index=True, return_inverse=True)
            mine_first, mine_place = _first_and_place(keys)
            with self.subTest(trial=trial):
                self.assertTrue(np.array_equal(mine_first, first))
                self.assertTrue(np.array_equal(mine_place, np.asarray(place).reshape(-1)))

    def test_a_box_is_named_by_its_points_as_every_parts_tree_named_it(self) -> None:
        """The reference is the per-box loop of commit f771b9c: every part's tree asked for the box's own points."""
        from scipy.spatial import cKDTree

        from src.robot.safety.planning import perceived

        def named_before(trees: Any, points_mm: np.ndarray) -> "str | None":
            if not trees or points_mm.shape[0] == 0:
                return None
            best, share = None, 0.0
            for label, tree in trees:
                near, _ = tree.query(points_mm, k=1, distance_upper_bound=perceived._ON_A_NAMED_PART_MM)
                on = float(np.mean(np.isfinite(near)))
                if on > share:
                    best, share = label, on
            return best if share >= perceived._NAMED_SHARE else None

        rng = np.random.default_rng(5)
        for trial in range(60):
            cloud = rng.uniform(-200.0, 200.0, (int(rng.integers(50, 3000)), 3))
            parts = []
            for number in range(int(rng.integers(0, 12))):
                centre = rng.uniform(-200.0, 200.0, 3)
                parts.append((f"part_{number % 4}", centre + rng.normal(0.0, float(rng.uniform(3.0, 40.0)),
                                                                        (int(rng.integers(1, 400)), 3))))
            trees = perceived._named_trees(parts)
            labels = [label for label, _ in trees]
            on = perceived._on_named_parts(trees, cloud)
            self.assertEqual(on.shape, (cloud.shape[0], len(trees)))
            reference = [(label, cKDTree(points)) for label, points in parts]
            for box in range(40):
                held = np.flatnonzero(rng.random(cloud.shape[0]) < float(rng.uniform(0.0, 0.2)))
                with self.subTest(trial=trial, box=box):
                    self.assertEqual(perceived._named_by_points(labels, on[held]),
                                     named_before(reference, cloud[held]))


if __name__ == "__main__":
    if "--write-golden" in sys.argv:
        write_golden()
    else:
        unittest.main()
