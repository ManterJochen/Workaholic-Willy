"""A part's convex hull never clears what its mesh refuses, and without a hull the mesh is measured.

The Hand-E housing is 48,198 triangles, and one exact query against it cost 0.9 to 2.4 ms on the desk; its convex hull
(Coal's ``Convex``, 4,631 vertices) answers in 0.008 to 0.021 ms. The whole-path judge
(``MeshSelfCollisionBackend.first_suspect``) asks the hulls first: a hull holds every triangle of its part, so its
distance is never more than the mesh's, and a hull that keeps the limit plus ``_HULL_SLACK_MM`` proves the mesh does.
Where it does not, the mesh is measured, as the guard measures it.

What this file pins, on the owner's guard (``tests/_owner_guard.py``):

* every part, the Hand-E housing among them, against boxes about it and against every part the guard checks it against:
  the hull's distance less ``_HULL_SLACK_MM`` is never more than the mesh's (Coal only: python-fcl builds no hull);
* boxes 3.5 to 12 mm under the housing and under a finger, at the cell's 3 mm: the hull alone clears every one;
* without a hull, on python-fcl or for a part the engine cannot hull, the mesh is measured and the answer is the same.
"""

from __future__ import annotations

import math
import unittest
from collections import Counter
from typing import Any

import numpy as np

from src.robot.safety._fcl_self_collision import _HULL_SLACK_MM, _EngineAdapter
from src.robot.safety._ur_kinematics import ur_link_transforms_mm_many
from src.robot.safety.path_samples import waypoint_path_samples
from tests import _owner_guard as owner
from tests.test_a_faster_camera_world_builds_the_same_world import frame


def _backend() -> "tuple[Any, Any, Any]":
    preflight, arm = owner.owner_cell(whole=True)
    guard = preflight._path_authority(arm)
    assert guard is not None
    return guard._exact_mesh_backend("ur10"), preflight, arm


def _needs_hulls(backend: Any) -> "dict[str, Any]":
    hulls = backend._hull_objects()
    if not hulls:
        raise unittest.SkipTest(f"the {backend.engine} engine builds no convex hull (Coal does)")
    return hulls


def _place(backend: Any, hulls: "dict[str, Any]", frames: np.ndarray, part: str) -> None:
    placed = frames[backend._frame[part]]
    backend._a.set_transform(backend._models[part], placed[:3, :3], placed[:3, 3])
    backend._a.set_transform(hulls[part], placed[:3, :3], placed[:3, 3])


class AHullIsNeverFartherThanItsMeshTests(unittest.TestCase):
    def setUp(self) -> None:
        owner.needs_the_engine()

    def test_no_hull_keeps_more_than_its_mesh_from_a_box(self) -> None:
        backend, _, _ = _backend()
        hulls = _needs_hulls(backend)
        self.assertEqual(set(backend._names), set(hulls), "every part has its hull, the Hand-E housing among them")
        rng = np.random.default_rng(7)
        look = np.radians(owner.HOME_DEG)
        for _ in range(25):
            frames = ur_link_transforms_mm_many("ur10", (look + rng.uniform(-0.5, 0.5, 6))[None])
            assert frames is not None
            for part in backend._names:
                _place(backend, hulls, frames[0], part)
                placed = frames[0][backend._frame[part]]
                centre = placed[:3, :3] @ backend._sph_c[part] + placed[:3, 3] + rng.uniform(-120.0, 120.0, 3)
                box = backend._a.box_object(rng.uniform(5.0, 80.0, 3), centre, float(rng.uniform(-1.0, 1.0)))
                mesh = backend._a.distance(backend._models[part], box)
                hull = backend._a.distance(hulls[part], box)
                self.assertLessEqual(hull - _HULL_SLACK_MM, max(mesh, 0.0), part)
                if mesh > 0.0:
                    self.assertLessEqual(hull, mesh + 1e-6 * max(1.0, mesh), f"{part}: a hull farther than its mesh")

    def test_no_pair_of_hulls_keeps_more_than_its_meshes(self) -> None:
        backend, _, _ = _backend()
        hulls = _needs_hulls(backend)
        rng = np.random.default_rng(8)
        for _ in range(40):
            frames = ur_link_transforms_mm_many("ur10", rng.uniform(-math.pi, math.pi, (1, 6)))
            assert frames is not None
            for part in backend._names:
                _place(backend, hulls, frames[0], part)
            for i, j in backend._pairs:
                part_i, part_j = backend._names[i], backend._names[j]
                mesh = backend._a.distance(backend._models[part_i], backend._models[part_j])
                hull = backend._a.distance(hulls[part_i], hulls[part_j])
                self.assertLessEqual(hull - _HULL_SLACK_MM, max(mesh, 0.0), f"{part_i}|{part_j}")

    def test_the_hull_alone_clears_a_box_3_5_to_12_mm_under_the_housing_and_a_finger(self) -> None:
        backend, _, _ = _backend()
        hulls = _needs_hulls(backend)
        frames = ur_link_transforms_mm_many("ur10", np.radians(owner.HOME_DEG)[None])
        assert frames is not None
        rng = np.random.default_rng(5)
        cleared = 0
        for part in ("gripper", "lfinger"):
            _place(backend, hulls, frames[0], part)
            placed = frames[0][backend._frame[part]]
            points = backend._hull[part] @ placed[:3, :3].T + placed[:3, 3]
            lowest = points[np.argmin(points[:, 2])]
            row = backend._names.index(part)
            for gap in (3.5, 6.0, 12.0):
                for _ in range(20):
                    centre = lowest + np.r_[rng.uniform(-25.0, 25.0, 2), -gap - 20.0]
                    box = backend._a.box_object(np.array([20.0, 20.0, 20.0]), centre, float(rng.uniform(-0.6, 0.6)))
                    bound = backend._bound_mm(row, box, box, owner.GUARD_MM, hulls)
                    self.assertGreaterEqual(bound, owner.GUARD_MM, f"{part}, {gap} mm under it")
                    self.assertLessEqual(bound, backend._a.distance(backend._models[part], box), part)
                    cleared += bound == backend._a.distance(hulls[part], box) - _HULL_SLACK_MM
        self.assertEqual(120, cleared, "the hull alone answered every one")


class WithoutAHullTheMeshIsMeasuredTests(unittest.TestCase):
    def setUp(self) -> None:
        owner.needs_the_engine()

    def test_python_fcl_and_an_engine_without_hulls_build_none(self) -> None:
        self.assertIsNone(_EngineAdapter(object(), "fcl").convex_object(np.eye(3)))

        class _NoHulls:
            kind = "stub"

            def build_object(self, vertices: Any, faces: Any) -> object:
                return object()

        from src.robot.safety._fcl_self_collision import MeshSelfCollisionBackend

        cube = (np.asarray([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]),
                np.zeros((0, 3), dtype=np.int64))
        backend = MeshSelfCollisionBackend(_NoHulls(), {"upper_arm": (*cube, 2), "wrist_3": (*cube, 6)})  # type: ignore[arg-type]
        self.assertEqual({}, backend._hull_objects())

    def test_without_hulls_the_meshes_answer_the_same_route_and_line(self) -> None:
        backend, preflight, arm = _backend()
        guard = preflight._path_authority(arm)
        assert guard is not None
        reach = preflight.joint_radii_mm(arm)
        assert reach is not None
        recorded = frame("F1")
        preflight.set_perceived_obstacles(owner.seen_world("F1", "A"))
        standoff = owner.nearest_solution(np.asarray(recorded["goal"]), recorded["down_rad"])
        assert standoff is not None
        route = waypoint_path_samples([list(recorded["look_rad"]), standoff.tolist()], reach_mm=reach,
                                      max_step_mm=owner.GUARD_MM)
        line = owner.line_samples(np.asarray(recorded["goal"]), np.asarray(recorded["grasp"]), standoff, reach)
        assert line is not None
        paths = [np.asarray(route.configs, dtype=np.float64), np.asarray(line.configs, dtype=np.float64)]
        with_hulls = [guard.first_suspect(arm, configs, len(configs)) for configs in paths]
        held = backend._hulls
        hulls = set(map(id, (held or {}).values()))
        asked: Counter[str] = Counter()
        measure = backend._a.distance

        def counting(a: Any, b: Any) -> float:
            asked["hull" if id(a) in hulls else "mesh"] += 1
            return float(measure(a, b))

        backend._hulls = {}
        backend._a.distance = counting
        try:
            without = [guard.first_suspect(arm, configs, len(configs)) for configs in paths]
        finally:
            backend._hulls = held
            del backend._a.distance
        self.assertEqual(with_hulls, without)
        self.assertEqual(0, asked["hull"])
        self.assertGreater(asked["mesh"], 0, "the meshes were measured where the spheres and the bound left it open")


if __name__ == "__main__":
    unittest.main()
