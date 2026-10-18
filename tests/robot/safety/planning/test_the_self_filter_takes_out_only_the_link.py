"""The self filter takes out the robot's own links and what lies within the padding of them, nothing further away.

The filter held each arm link as one capsule fitted round its mesh. A capsule holds the link and a good deal more: the
UR10 upper arm's, 122 mm about the tube's axis, reaches 118 mm about the shoulder end at the height of a bin's rim,
where the shoulder housing hangs 52 mm over the base plate. Every point a camera saw in there was taken for the robot.
One box per cluster covered the hole such a capsule cut into a bin beside the base; a height map of what the camera saw
does not, so a rim 6 mm under the housing would have been free (2026-09-30).

Now a capsule fitted round a link carries the link's own surface, points laid over its triangles no surface point is
further than 4 mm from, and a point inside the capsule is the link only within the padding (15 mm) of that surface.
The padding is then all that is added around a link, as around the hand and a wrist camera: what the camera sees past
it is an obstacle, and the box grown round it by the same 15 mm reaches back to the link. Names new with this change
are imported inside the tests.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.robot.safety.planning.environment import collision_mesh_bundle
from src.robot.safety.planning.perceived import LinkCapsule, SelfBody
from src.robot.safety.planning.self_envelope import arm_capsules, yawed_link_transforms_mm

_Q = np.radians([0.0, -60.0, 80.0, -110.0, -90.0, 0.0])
_PADDING_MM = 15.0
_LINKS = {"shoulder": 1, "upper_arm": 2, "forearm": 3, "wrist_1": 4, "wrist_2": 5, "wrist_3": 6}


def _body(padding_mm: float = _PADDING_MM) -> SelfBody:
    frames = yawed_link_transforms_mm("ur10", _Q, 0.0)
    capsules = arm_capsules("ur10")
    assert frames is not None and capsules is not None
    return SelfBody.from_frames(frames, capsules, padding_mm=padding_mm)


def _surface_samples(name: str, count: int, *, offset_mm: float = 0.0, seed: int = 3) -> np.ndarray:
    """Random points on a link's triangles in BASE, moved ``offset_mm`` out along each triangle's normal."""
    frames = yawed_link_transforms_mm("ur10", _Q, 0.0)
    assert frames is not None
    with np.load(collision_mesh_bundle("ur10")) as data:
        vertices = np.asarray(data[f"{name}__v"], dtype=np.float64)
        faces = np.asarray(data[f"{name}__f"], dtype=np.int64)
    rng = np.random.default_rng(seed)
    corners = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    pick = rng.choice(len(faces), size=count, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(count)), rng.random(count)
    tri = corners[pick]
    points = ((1 - r1)[:, None] * tri[:, 0] + (r1 * (1 - r2))[:, None] * tri[:, 1] + (r1 * r2)[:, None] * tri[:, 2])
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    frame = frames[_LINKS[name]]
    return (points + offset_mm * normal) @ frame[:3, :3].T + frame[:3, 3]


class TheFilterTakesOutTheLinkTests(unittest.TestCase):
    def test_every_point_of_every_link_is_taken_out(self) -> None:
        body = _body()
        for name in _LINKS:
            with self.subTest(link=name):
                self.assertTrue(bool(body.contains(_surface_samples(name, 4000)).all()), name)

    def test_a_point_the_camera_placed_up_to_11_mm_off_the_link_is_taken_out_too(self) -> None:
        """The spacing of the laid points (4 mm, at most 3.2 mm to the nearest) leaves 11 mm of the padding for a
        camera's own error along the surface's normal, both ways."""
        body = _body()
        for name in _LINKS:
            for offset in (-11.0, 11.0):
                with self.subTest(link=name, offset_mm=offset):
                    self.assertTrue(bool(body.contains(_surface_samples(name, 2000, offset_mm=offset)).all()))


def _arm_triangles() -> np.ndarray:
    """Every triangle of the six arm links, placed at the test's joints, ``(T, 3, 3)`` in BASE."""
    frames = yawed_link_transforms_mm("ur10", _Q, 0.0)
    assert frames is not None
    placed = []
    with np.load(collision_mesh_bundle("ur10")) as data:
        for name, frame in _LINKS.items():
            vertices = np.asarray(data[f"{name}__v"], dtype=np.float64) @ frames[frame][:3, :3].T + frames[frame][:3, 3]
            placed.append(vertices[np.asarray(data[f"{name}__f"], dtype=np.int64)])
    return np.concatenate(placed)


def _exact_distance_mm(point: np.ndarray, triangles: np.ndarray) -> float:
    """The distance from ``point`` to the nearest of ``triangles``, exact: the closest point on each triangle by its
    Voronoi region (Ericson, Real-Time Collision Detection, 5.1.5), an oracle apart from anything the filter lays."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    d1, d2 = np.einsum("ij,ij->i", ab, point - a), np.einsum("ij,ij->i", ac, point - a)
    d3, d4 = np.einsum("ij,ij->i", ab, point - b), np.einsum("ij,ij->i", ac, point - b)
    d5, d6 = np.einsum("ij,ij->i", ab, point - c), np.einsum("ij,ij->i", ac, point - c)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = va + vb + vc
        closest = a + ab * (vb / denominator)[:, None] + ac * (vc / denominator)[:, None]
        on_bc = (va <= 0.0) & (d4 - d3 >= 0.0) & (d5 - d6 >= 0.0)
        closest[on_bc] = (b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None])[on_bc]
        on_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
        closest[on_ac] = (a + ac * (d2 / (d2 - d6))[:, None])[on_ac]
        on_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
        closest[on_ab] = (a + ab * (d1 / (d1 - d3))[:, None])[on_ab]
    at_c = (d6 >= 0.0) & (d5 <= d6)
    closest[at_c] = c[at_c]
    at_b = (d3 >= 0.0) & (d4 <= d3)
    closest[at_b] = b[at_b]
    at_a = (d1 <= 0.0) & (d2 <= 0.0)
    closest[at_a] = a[at_a]
    return float(np.nanmin(np.linalg.norm(point - closest, axis=1)))


class TheFilterKeepsWhatIsNotTheLinkTests(unittest.TestCase):
    def test_a_point_past_the_padding_is_kept_though_the_capsule_holds_it(self) -> None:
        """⛔ Before, every point inside a link's capsule was the robot. The points are laid off the upper arm and the
        forearm along their faces' normals, and judged by their exact distance to every triangle of the arm."""
        body = _body()
        capsule_only = SelfBody(segments_mm=body.segments_mm, radii_mm=body.radii_mm)
        triangles = _arm_triangles()
        rng = np.random.default_rng(9)
        for name in ("upper_arm", "forearm"):
            with self.subTest(link=name):
                away = np.concatenate([_surface_samples(name, 150, offset_mm=float(offset), seed=int(offset))
                                       for offset in rng.uniform(16.0, 45.0, 6)])
                away = away[capsule_only.contains(away)]
                exact = np.array([_exact_distance_mm(point, triangles) for point in away])
                past = away[exact > _PADDING_MM]
                self.assertGreater(len(past), 100, "the control: the capsule holds points past the padding")
                self.assertFalse(bool(body.contains(past).any()), "a point past the padding was taken out")

    def test_a_rim_under_the_shoulder_housing_is_an_obstacle(self) -> None:
        """The investigation's case: the housing's lowest point stands 53 mm over the base plate; a rim 20 mm under it
        and one 40 mm under it are what the camera saw, not the robot."""
        frames = yawed_link_transforms_mm("ur10", _Q, 0.0)
        assert frames is not None
        with np.load(collision_mesh_bundle("ur10")) as data:
            upper = np.asarray(data["upper_arm__v"], dtype=np.float64) @ frames[2][:3, :3].T + frames[2][:3, 3]
        lowest = upper[int(np.argmin(upper[:, 2]))]
        rim = np.array([[lowest[0] + dx, lowest[1], lowest[2] - drop] for dx in (-20.0, 0.0, 20.0) for drop in (20.0, 40.0)])
        body = _body()
        self.assertFalse(bool(body.contains(rim).any()))
        capsule_only = SelfBody(segments_mm=body.segments_mm, radii_mm=body.radii_mm)
        self.assertTrue(bool(capsule_only.contains(rim).all()), "the control: the capsule took all of it")

    def test_the_filter_never_takes_out_what_the_capsule_kept(self) -> None:
        body = _body()
        capsule_only = SelfBody(segments_mm=body.segments_mm, radii_mm=body.radii_mm)
        points = np.random.default_rng(5).uniform((-1200.0, -1200.0, -100.0), (400.0, 400.0, 900.0), size=(20000, 3))
        self.assertFalse(bool((body.contains(points) & ~capsule_only.contains(points)).any()))


class WhereTheFilterStaysACapsuleTests(unittest.TestCase):
    def test_a_padding_narrower_than_the_spacing_keeps_the_capsule(self) -> None:
        """Below the spacing of the laid points the link's own surface between two of them could be left in."""
        body = _body(padding_mm=0.0)
        self.assertEqual(body.surfaces, ())  # type: ignore[attr-defined]
        self.assertTrue(bool(body.contains(_surface_samples("upper_arm", 2000)).all()))

    def test_a_hand_made_capsule_is_taken_as_it_is(self) -> None:
        body = SelfBody.from_frames([np.eye(4)], (LinkCapsule(frame=0, start_mm=(0.0, 0.0, 0.0),
                                                              end_mm=(0.0, 0.0, 100.0), radius_mm=50.0),),
                                    padding_mm=_PADDING_MM)
        self.assertTrue(bool(body.contains(np.array([[0.0, 60.0, 50.0]]))[0]))

    def test_every_link_of_every_bundled_arm_carries_its_surface(self) -> None:
        for model in ("ur3e", "ur5e", "ur10", "ur10e"):
            capsules = arm_capsules(model)
            assert capsules is not None
            with self.subTest(model=model):
                self.assertEqual([capsule.surface.name for capsule in capsules],  # type: ignore[union-attr]
                                 list(_LINKS))
                self.assertTrue(all(math.isclose(capsule.surface.spacing_mm, 4.0)  # type: ignore[union-attr]
                                    for capsule in capsules))


if __name__ == "__main__":
    unittest.main()
