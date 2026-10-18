"""A perceived box never covers the target a pick keeps out, and a declared tote is not perceived again as a block.

The audit of the owner's cell, 2026-09-23 (lens "world"), reproduced both:

  * the keep-out took only the target's own points out of the world. The neighbours around it clustered on a 25 mm
    grid and one box was fitted around the cluster, hole included: a 3 x 3 grid of 40 mm cubes with the centre one
    kept out came back as one 207 mm box over the target, and parts 30 mm apart, which the Hand-E fits between,
    were blocked;
  * nothing subtracted declared geometry from the camera points, so a declared tote's rim came back as a
    290 x 200 x 150 mm box over its inside, floored to the bench, and the descent to every part in it was refused.

A cluster that spans a keep-out box is now cut around it, and a point within the bench band of a declared fixture or
mesh is that fixture. Every scene is ray cast through the pinhole the converter inverts, straight down, so the
geometry is known in millimetres. Names that are new with this change are imported inside the tests, so the file
loads against the tree before it and each test fails there on its own assertion or its own missing name.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.robot.safety.planning.perceived import (
    DepthView,
    PerceivedBox,
    WorldBuildLimits,
    WorldBuildTuning,
    build_perceived_boxes,
    target_keep_out_box,
)

_W, _H, _F = 320, 240, 280.0
_K = np.array([[_F, 0.0, _W / 2.0], [0.0, _F, _H / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
_LIMITS = WorldBuildLimits(x_mm=(-760.7, 760.7), y_mm=(-760.7, 760.7), z_mm=(-100.0, 748.4), support_plane_top_mm=0.0)
_TUNING = WorldBuildTuning(voxel_size_mm=10.0, cluster_voxel_mm=25.0, min_points=12, margin_mm=15.0, max_boxes=8)
#: Where every pick below closes: the target's centre on the bench.
_TARGET_XY = (400.0, -300.0)


def _down(x_mm: float, y_mm: float, height_mm: float) -> np.ndarray:
    """CAMERA to BASE for a camera at ``height_mm`` over (x, y) looking straight down."""
    transform = np.eye(4)
    transform[:3, :3] = np.diag([1.0, -1.0, -1.0])
    transform[:3, 3] = (x_mm, y_mm, height_mm)
    return transform


def _render(camera_to_base: np.ndarray, boxes: "list[tuple[tuple, tuple]]") -> np.ndarray:
    """Depth along each ray to the nearest of ``boxes`` (low, high corners) or the bench at z 0 (slab method)."""
    cols, rows = np.meshgrid(np.arange(_W, dtype=np.float64), np.arange(_H, dtype=np.float64))
    rays = np.stack([(cols - _K[0, 2]) / _F, (rows - _K[1, 2]) / _F, np.ones_like(cols)], axis=-1) @ camera_to_base[:3, :3].T
    origin = camera_to_base[:3, 3]
    depth = np.where(rays[..., 2] < 0.0, -origin[2] / np.where(rays[..., 2] < 0.0, rays[..., 2], -1.0), np.inf)
    for low, high in boxes:
        with np.errstate(divide="ignore", invalid="ignore"):
            t_low = (np.asarray(low, dtype=np.float64) - origin) / rays
            t_high = (np.asarray(high, dtype=np.float64) - origin) / rays
        near = np.nanmax(np.minimum(t_low, t_high), axis=-1)
        far = np.nanmin(np.maximum(t_low, t_high), axis=-1)
        depth = np.where((far >= near) & (near > 0.0) & (near < depth), near, depth)
    return np.where(np.isfinite(depth), depth, 0.0)


def _cube(x_mm: float, y_mm: float, side_mm: float = 40.0, base_mm: float = 0.0) -> "tuple[tuple, tuple]":
    half = side_mm / 2.0
    return (x_mm - half, y_mm - half, base_mm), (x_mm + half, y_mm + half, base_mm + side_mm)


def _target_points(x_mm: float, y_mm: float, top_mm: float, side_mm: float = 40.0) -> np.ndarray:
    """The target's top face in BASE, as a located part hands it to the world."""
    half = side_mm / 2.0
    xs, ys = np.meshgrid(np.linspace(x_mm - half, x_mm + half, 9), np.linspace(y_mm - half, y_mm + half, 9))
    return np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, top_mm)])


def _inside(box: PerceivedBox, points_mm: np.ndarray) -> np.ndarray:
    """Which BASE points lie inside the turned box the planner receives."""
    local = np.asarray(points_mm, dtype=np.float64) - np.asarray(box.center_mm)
    cos_yaw, sin_yaw = math.cos(box.yaw_rad), math.sin(box.yaw_rad)
    x = cos_yaw * local[:, 0] + sin_yaw * local[:, 1]
    y = -sin_yaw * local[:, 0] + cos_yaw * local[:, 1]
    half = np.asarray(box.dims_mm) / 2.0
    return (np.abs(x) < half[0]) & (np.abs(y) < half[1]) & (np.abs(local[:, 2]) < half[2])


def _covered(boxes: "tuple[PerceivedBox, ...]", points_mm: np.ndarray) -> "list[str]":
    return [box.name for box in boxes if bool(_inside(box, points_mm).any())]


def _hull(x_mm: float, y_mm: float, top_mm: float, side_mm: float = 40.0, *, inset_mm: float = 1.0) -> np.ndarray:
    """Points throughout the target's own body, ``inset_mm`` inside its faces."""
    half = side_mm / 2.0 - inset_mm
    axis = np.linspace(-half, half, 5)
    xs, ys, zs = np.meshgrid(axis, axis, np.linspace(inset_mm, top_mm - inset_mm, 5))
    return np.column_stack([xs.ravel() + x_mm, ys.ravel() + y_mm, zs.ravel()])


def _pile(gap_mm: float) -> "list[tuple[tuple, tuple]]":
    pitch = 40.0 + gap_mm
    return [_cube(_TARGET_XY[0] + dx, _TARGET_XY[1] + dy) for dx in (-pitch, 0.0, pitch) for dy in (-pitch, 0.0, pitch)]


def _pick(scene: "list[tuple[tuple, tuple]]", target_top_mm: float = 40.0, **build: object) -> object:
    camera = _down(*_TARGET_XY, 600.0)
    keep = target_keep_out_box(_target_points(*_TARGET_XY, target_top_mm), name="target", limits=_LIMITS,
                               tuning=_TUNING)
    assert keep is not None
    view = DepthView(surface_depth_mm=_render(camera, scene), intrinsics=_K, camera_to_base=camera, name="wrist")
    return build_perceived_boxes(views=[view], limits=_LIMITS, tuning=_TUNING, keep_out=(keep,), **build)


class ANeighbourClusterIsCutAroundTheTargetTests(unittest.TestCase):

    def test_no_box_covers_the_target_in_a_pile_at_any_gap(self) -> None:
        """⛔ The finding: at 10 mm one 207 mm box covered the kept-out cube."""
        for gap in (10.0, 20.0, 30.0):
            with self.subTest(gap_mm=gap):
                world = _pick(_pile(gap))
                self.assertEqual(_covered(world.boxes, _hull(*_TARGET_XY, 40.0)), [])  # type: ignore[attr-defined]
                self.assertEqual(world.keep_out_cuts, {"target": 1})  # type: ignore[attr-defined]

    def test_every_neighbour_is_still_an_obstacle(self) -> None:
        """The cut takes nothing out of the world but the hole: each neighbour's own top stays inside a box."""
        pitch = 50.0
        world = _pick(_pile(10.0))
        for dx in (-pitch, 0.0, pitch):
            for dy in (-pitch, 0.0, pitch):
                if dx == dy == 0.0:
                    continue
                with self.subTest(dx=dx, dy=dy):
                    self.assertTrue(_covered(world.boxes, _hull(_TARGET_XY[0] + dx, _TARGET_XY[1] + dy, 40.0,  # type: ignore[attr-defined]
                                                                inset_mm=5.0)))

    def test_the_hand_e_fits_between_parts_30_mm_apart(self) -> None:
        """⛔ Measured by the audit's verifier: at 10, 20 and 30 mm one ring box covered both pads (x +/- 28)."""
        pads = np.array([[_TARGET_XY[0] - 28.0, _TARGET_XY[1], 20.0], [_TARGET_XY[0] + 28.0, _TARGET_XY[1], 20.0]])
        self.assertEqual(_covered(_pick(_pile(30.0)).boxes, pads), [])  # type: ignore[attr-defined]
        # At 10 mm the neighbour's own face is 2 mm past the pad, and its margin keeps it: that refusal is right.
        self.assertTrue(_covered(_pick(_pile(10.0)).boxes, pads))  # type: ignore[attr-defined]

    def test_a_cut_box_reaches_into_the_keep_out_by_no_more_than_the_margin(self) -> None:
        world = _pick(_pile(10.0))
        # The keep-out is the target grown by the 15 mm margin, so a cut box grown by the same margin stops at the
        # target's own faces: half a millimetre inside them nothing is covered.
        self.assertEqual(_covered(world.boxes, _hull(*_TARGET_XY, 40.0, inset_mm=0.5)), [])  # type: ignore[attr-defined]

    def test_something_resting_over_the_target_stands_on_its_top_and_not_on_the_bench(self) -> None:
        """A bar across the target on a post beside it: before, the one box floored through the target."""
        bar = ((_TARGET_XY[0] - 120.0, _TARGET_XY[1] - 15.0, 150.0), (_TARGET_XY[0] + 40.0, _TARGET_XY[1] + 15.0, 165.0))
        post = _cube(_TARGET_XY[0] - 100.0, _TARGET_XY[1], side_mm=30.0)
        post = ((post[0][0], post[0][1], 0.0), (post[1][0], post[1][1], 150.0))
        world = _pick([_cube(*_TARGET_XY), bar, post])
        self.assertEqual(_covered(world.boxes, _hull(*_TARGET_XY, 40.0)), [])  # type: ignore[attr-defined]
        over_the_target = np.array([[_TARGET_XY[0], _TARGET_XY[1], 157.0]])
        self.assertTrue(_covered(world.boxes, over_the_target), "the bar over the target is still an obstacle")  # type: ignore[attr-defined]

    def test_a_cluster_beside_the_target_is_fitted_as_it_was(self) -> None:
        """The control: one neighbour that does not span the keep-out is not cut, and keeps its own turn."""
        world = _pick([_cube(*_TARGET_XY), _cube(_TARGET_XY[0] - 50.0, _TARGET_XY[1])])
        self.assertEqual(world.keep_out_cuts, {})  # type: ignore[attr-defined]
        self.assertEqual(len(world.boxes), 1)  # type: ignore[attr-defined]
        self.assertEqual(_covered(world.boxes, _hull(*_TARGET_XY, 40.0, inset_mm=16.0)), [])  # type: ignore[attr-defined]

    def test_a_turned_keep_out_leaves_the_target_free_for_the_guard_too(self) -> None:
        """The path guard holds each perceived box as the axis-aligned box enclosing it. A keep-out turned against BASE
        is cut around square with BASE first, and only what lies beside it is cut in its turn, so the guard's boxes
        leave the target and the pads free as the planner's do."""
        from dataclasses import replace

        from src.robot.core.keep_out import KeepOutBox

        camera = _down(*_TARGET_XY, 600.0)
        angle = math.radians(20.0)
        matrix = np.eye(4)
        matrix[:3, :3] = [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0],
                          [0.0, 0.0, 1.0]]
        matrix[:3, 3] = (*_TARGET_XY, 27.5)
        keep = KeepOutBox.from_matrix("target", matrix, (35.0, 35.0, 27.5))
        view = DepthView(surface_depth_mm=_render(camera, _pile(30.0)), intrinsics=_K, camera_to_base=camera,
                         name="wrist")
        world = build_perceived_boxes(views=[view], limits=_LIMITS, tuning=_TUNING, keep_out=(keep,))
        as_the_guard_holds_them = tuple(
            replace(box, dims_mm=tuple(2.0 * h for h in box.enclosing_half_extents_mm), yaw_rad=0.0)
            for box in world.boxes)
        free = np.vstack([_hull(*_TARGET_XY, 40.0),
                          [[_TARGET_XY[0] - 28.0, _TARGET_XY[1], 20.0], [_TARGET_XY[0] + 28.0, _TARGET_XY[1], 20.0]]])
        self.assertEqual(_covered(world.boxes, free), [])
        self.assertEqual(_covered(as_the_guard_holds_them, free), [])
        self.assertEqual(world.keep_out_cuts, {"target": 1})

    def test_a_part_with_no_direction_is_fitted_square_with_base(self) -> None:
        """A lone square part at 30 mm: its principal direction is noise, and a box turned by it is one the guard
        holds 1.4 times as wide, over the pads. Fitted square with BASE, the guard's box is the planner's."""
        world = _pick([_cube(*_TARGET_XY), _cube(_TARGET_XY[0] - 70.0, _TARGET_XY[1] + 70.0)])
        (box,) = world.boxes  # type: ignore[attr-defined]
        self.assertEqual(box.yaw_rad, 0.0)
        self.assertEqual(tuple(round(2.0 * h, 6) for h in box.enclosing_half_extents_mm),
                         tuple(round(d, 6) for d in box.dims_mm))

    def test_a_long_part_keeps_its_direction(self) -> None:
        """The control: a part twice as long as it is wide is still turned along its length."""
        from src.robot.safety.planning.perceived import _oriented_box

        angle = math.radians(30.0)
        along = np.linspace(-40.0, 40.0, 41)
        across = np.linspace(-20.0, 20.0, 21)
        a, c = np.meshgrid(along, across)
        points = np.column_stack([a.ravel() * math.cos(angle) - c.ravel() * math.sin(angle),
                                  a.ravel() * math.sin(angle) + c.ravel() * math.cos(angle), np.full(a.size, 30.0)])
        _, dims, yaw = _oriented_box(points, 0.0)
        self.assertAlmostEqual(math.degrees(yaw) % 180.0, 30.0, places=6)
        self.assertAlmostEqual(dims[0], 80.0, places=6)

    def test_the_cut_is_reported(self) -> None:
        world = _pick(_pile(10.0))
        self.assertIn("1 cluster(s) cut around 'target', so no box covers it", world.render())  # type: ignore[attr-defined]
        self.assertEqual(world.to_dict()["keep_out_cuts"], {"target": 1})  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------------------------------
# A declared tote
# ---------------------------------------------------------------------------------------------------

#: A 320 x 220 tote with 10 mm walls, its rim at 120 mm and its inside floor 12 mm over the bench.
_TOTE_CENTRE = (450.0, -350.0)
_TOTE = {
    "wall_left": ((290.0, -460.0, 0.0), (300.0, -240.0, 120.0)),
    "wall_right": ((600.0, -460.0, 0.0), (610.0, -240.0, 120.0)),
    "wall_front": ((290.0, -460.0, 0.0), (610.0, -450.0, 120.0)),
    "wall_back": ((290.0, -250.0, 0.0), (610.0, -240.0, 120.0)),
    "tote_floor": ((300.0, -450.0, 0.0), (600.0, -250.0, 12.0)),
}


def _tote_scene(*parts: "tuple[tuple, tuple]") -> "tuple[np.ndarray, np.ndarray]":
    camera = _down(_TOTE_CENTRE[0], _TOTE_CENTRE[1], 650.0)
    return camera, _render(camera, [*_TOTE.values(), *parts])


def _declared_boxes() -> tuple:
    from src.robot.safety.planning.perceived import DeclaredBody

    bodies = []
    for name, (low, high) in _TOTE.items():
        matrix = np.eye(4)
        matrix[:3, 3] = (np.asarray(low) + np.asarray(high)) / 2.0
        bodies.append(DeclaredBody.box(name, matrix, (np.asarray(high) - np.asarray(low)) / 2.0))
    return tuple(bodies)


def _box_mesh(low: tuple, high: tuple) -> "tuple[np.ndarray, np.ndarray]":
    """The twelve triangles of an axis-aligned box."""
    lo, hi = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    quads = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    faces = [tri for a, b, c, d in quads for tri in ((a, b, c), (a, c, d))]
    return corners, np.asarray(faces, dtype=np.int64)


def _declared_mesh() -> object:
    from src.robot.safety.planning.perceived import DeclaredBody

    vertices, faces = [], []
    for low, high in _TOTE.values():
        v, f = _box_mesh(low, high)
        faces.append(f + sum(len(x) for x in vertices))
        vertices.append(v)
    return DeclaredBody.mesh("tote", np.concatenate(vertices), np.concatenate(faces), spacing_mm=2.0)


def _tote_pick(declared: tuple, *parts: "tuple[tuple, tuple]") -> object:
    target = _cube(*_TOTE_CENTRE, base_mm=12.0)
    camera, depth = _tote_scene(target, *parts)
    keep = target_keep_out_box(_target_points(*_TOTE_CENTRE, 52.0), name="target", limits=_LIMITS, tuning=_TUNING)
    view = DepthView(surface_depth_mm=depth, intrinsics=_K, camera_to_base=camera, name="wrist")
    return build_perceived_boxes(views=[view], limits=_LIMITS, tuning=_TUNING, keep_out=(keep,), declared=declared)


class ADeclaredToteIsNotABlockTests(unittest.TestCase):
    _GRASP = np.array([[_TOTE_CENTRE[0], _TOTE_CENTRE[1], 32.0],
                       [_TOTE_CENTRE[0] - 28.0, _TOTE_CENTRE[1], 32.0], [_TOTE_CENTRE[0] + 28.0, _TOTE_CENTRE[1], 32.0]])

    def test_a_tote_declared_as_boxes_leaves_the_descent_free(self) -> None:
        """⛔ The finding: the rim came back as a 290 x 200 x 150 mm box over the grasp point."""
        from src.robot.safety.planning.perceived import DropReason

        world = _tote_pick(_declared_boxes())
        self.assertEqual(_covered(world.boxes, self._GRASP), [])  # type: ignore[attr-defined]
        self.assertGreater(world.dropped_points.get(DropReason.DECLARED, 0), 0)  # type: ignore[attr-defined]
        self.assertGreater(world.declared_points["wall_left"], 0)  # type: ignore[attr-defined]
        self.assertGreater(world.declared_points["tote_floor"], 0)  # type: ignore[attr-defined]

    def test_a_tote_declared_as_a_mesh_leaves_the_descent_free(self) -> None:
        world = _tote_pick((_declared_mesh(),))
        self.assertEqual(_covered(world.boxes, self._GRASP), [])  # type: ignore[attr-defined]
        self.assertGreater(world.declared_points["tote"], 0)  # type: ignore[attr-defined]

    def test_a_part_in_the_declared_tote_is_still_an_obstacle(self) -> None:
        """The control that matters: the tote is declared, what lies in it is not."""
        for declared in (_declared_boxes(), (_declared_mesh(),)):
            with self.subTest(declared=[body.name for body in declared]):  # type: ignore[attr-defined]
                neighbour = _cube(_TOTE_CENTRE[0] + 90.0, _TOTE_CENTRE[1], base_mm=12.0)
                world = _tote_pick(declared, neighbour)
                self.assertTrue(_covered(world.boxes, _hull(_TOTE_CENTRE[0] + 90.0, _TOTE_CENTRE[1], 52.0,  # type: ignore[attr-defined]
                                                            inset_mm=13.0)))

    def test_an_undeclared_tote_still_comes_back(self) -> None:
        """The control: nothing declared, and the walls are obstacles, cut around the target."""
        world = _tote_pick(())
        self.assertTrue(world.boxes)  # type: ignore[attr-defined]
        self.assertEqual(world.declared_points, {})  # type: ignore[attr-defined]
        self.assertEqual(_covered(world.boxes, self._GRASP[:1]), [])  # type: ignore[attr-defined]

    def test_a_declared_box_takes_only_what_lies_within_the_bench_band_of_it(self) -> None:
        """A wall declared 4 mm off where it stands is still the wall; one declared 30 mm off leaves a sliver."""
        from src.robot.safety.planning.perceived import DeclaredBody

        camera = _down(450.0, -350.0, 650.0)
        wall = ((440.0, -450.0, 0.0), (460.0, -250.0, 120.0))
        view = DepthView(surface_depth_mm=_render(camera, [wall]), intrinsics=_K, camera_to_base=camera, name="wrist")
        for off, stays in ((4.0, False), (30.0, True)):
            with self.subTest(off_mm=off):
                matrix = np.eye(4)
                matrix[:3, 3] = (450.0, -350.0, 60.0 - off)
                declared = (DeclaredBody.box("wall", matrix, (10.0, 100.0, 60.0)),)
                world = build_perceived_boxes(views=[view], limits=_LIMITS, tuning=_TUNING, declared=declared)
                self.assertEqual(bool(world.boxes), stays)

    def test_the_declared_distance_is_never_short(self) -> None:
        from src.robot.safety.planning.perceived import DeclaredBody

        rng = np.random.default_rng(7)
        points = rng.uniform(-60.0, 60.0, size=(400, 3))
        box = DeclaredBody.box("b", np.eye(4), (20.0, 10.0, 5.0))
        exact_box = np.linalg.norm(np.maximum(np.abs(points) - (20.0, 10.0, 5.0), 0.0), axis=1)
        np.testing.assert_allclose(box.distance_mm(points), exact_box, atol=1e-9)
        vertices, faces = _box_mesh((-20.0, -10.0, -5.0), (20.0, 10.0, 5.0))
        mesh = DeclaredBody.mesh("m", vertices, faces, spacing_mm=2.0)
        outside = exact_box > 0.0
        measured = mesh.distance_mm(points[outside])
        self.assertTrue(bool(np.all(measured >= exact_box[outside] - 1e-9)))
        self.assertTrue(bool(np.all(measured <= exact_box[outside] + 2.0)))

    def test_a_mesh_lays_the_same_points_twice(self) -> None:
        a, b = _declared_mesh(), _declared_mesh()
        np.testing.assert_array_equal(a.surface_mm, b.surface_mm)  # type: ignore[attr-defined]


class TheLiveWorldReadsItsDeclaredWorldTests(unittest.TestCase):
    """The live world hands the converter the declared fixtures and meshes it registers, from their wire format."""

    def _world(self, **kwargs: object) -> object:
        from src.robot.safety.planning.live_world import CameraView, DepthSnapshot, LivePlannerWorld
        from src.robot.safety.planning.perceived import LinkCapsule, SelfEnvelope
        from src.robot.safety.planning.world import planner_cuboid

        camera, depth = _tote_scene()

        class _Camera:
            def grab_surface_depth(self) -> DepthSnapshot:
                return DepthSnapshot(depth_mm=depth, intrinsics=_K, timestamp=100.0)

        declared = tuple(planner_cuboid(name, (np.asarray(low) + np.asarray(high)) / 2.0,
                                        np.asarray(high) - np.asarray(low)) for name, (low, high) in _TOTE.items())
        settings: dict[str, object] = {
            "cameras": (CameraView(name="wrist", depth_source=_Camera(), camera_to_base=camera),),
            "declared": declared, "limits": _LIMITS, "tuning": _TUNING, "max_age_ms": 500.0,
        }
        settings.update(kwargs)
        world = LivePlannerWorld(**settings)  # type: ignore[arg-type]
        body = SelfEnvelope(frames_mm=(np.eye(4),), capsules=(
            LinkCapsule(frame=0, start_mm=(0.0, 0.0, 0.0), end_mm=(0.0, 0.0, 100.0), radius_mm=50.0),))
        return world.world_for(self_envelope=body, now=100.1)

    def test_the_declared_tote_leaves_no_perceived_box(self) -> None:
        snapshot = self._world()
        self.assertTrue(snapshot.usable)  # type: ignore[attr-defined]
        self.assertEqual(snapshot.perceived_count, 0, snapshot.render())  # type: ignore[attr-defined]
        self.assertGreater(snapshot.perceived.declared_points["wall_back"], 0)  # type: ignore[attr-defined]

    def test_a_turned_declared_box_is_placed_as_the_planner_places_it(self) -> None:
        from src.robot.safety.planning.live_world import _declared_bodies
        from src.robot.safety.planning.world import planner_cuboid

        (body,), unread = _declared_bodies((planner_cuboid("turned", (100.0, 0.0, 50.0), (200.0, 20.0, 100.0),
                                                           yaw_rad=math.pi / 2.0),), ())
        self.assertEqual(unread, ())
        # Turned a quarter, its long side runs along base y: a point 80 mm along y is inside, 80 mm along x is not.
        np.testing.assert_allclose(body.distance_mm(np.array([[100.0, 80.0, 50.0], [180.0, 0.0, 50.0]])),  # type: ignore[attr-defined]
                                   [0.0, 70.0], atol=1e-6)

    def test_a_mesh_that_cannot_be_read_keeps_what_the_camera_sees_and_says_so(self) -> None:
        meshes = ({"name": "tote_mesh", "file_path": "no/such/tote.stl", "pose": [0.45, -0.35, 0.0, 1.0, 0.0, 0.0, 0.0],
                   "scale": [0.001, 0.001, 0.001]},)
        snapshot = self._world(declared=(), declared_meshes=meshes)
        self.assertTrue(snapshot.usable)  # type: ignore[attr-defined]
        self.assertGreater(snapshot.perceived_count, 0, "fail safe: the tote the camera sees stays an obstacle")  # type: ignore[attr-defined]
        self.assertIn("declared mesh 'tote_mesh' could not be read", snapshot.render())  # type: ignore[attr-defined]
        self.assertIn("comes back as obstacles", snapshot.render())  # type: ignore[attr-defined]

    def test_a_mesh_file_is_read_scaled_and_placed_as_the_sidecar_reads_it(self) -> None:
        import tempfile
        from pathlib import Path

        import trimesh

        vertices, faces = [], []
        for low, high in _TOTE.values():  # authored in millimetres about the tote's own centre
            v, f = _box_mesh(tuple(np.asarray(low) - (*_TOTE_CENTRE, 0.0)), tuple(np.asarray(high) - (*_TOTE_CENTRE, 0.0)))
            faces.append(f + sum(len(x) for x in vertices))
            vertices.append(v)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tote.stl"
            trimesh.Trimesh(np.concatenate(vertices), np.concatenate(faces), process=False).export(path)
            meshes = ({"name": "tote_mesh", "file_path": str(path),
                       "pose": [_TOTE_CENTRE[0] / 1000.0, _TOTE_CENTRE[1] / 1000.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                       "scale": [0.001, 0.001, 0.001]},)
            snapshot = self._world(declared=(), declared_meshes=meshes)
        self.assertEqual(snapshot.perceived_count, 0, snapshot.render())  # type: ignore[attr-defined]
        self.assertGreater(snapshot.perceived.declared_points["tote_mesh"], 0)  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
