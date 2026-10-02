"""A side grasp never goes through space no depth ray reached.

The camera world takes an occlusion shadow for free space (``perceived.py``, the shadow stage), and the exact guard
judges an approach against that same world, so a tilted hand sent through a neighbour's shadow is judged against free
space that nobody saw. With side approaches on (the owner, 2026-10-01: "gleichwertig nach Geometrie"), SFE therefore
offers a candidate tilted 15 degrees or more only where every sample of its open corridor lies in space a depth ray
reached (``corridor_seen``, the cell-fix plan's contract 5: the measured depth along a point's pixel ray reaches at least
to it, 5 mm tolerance), counts the ones it drops as ``unseen_corridor``, and without a seen test offers no tilt past the
first that fits.

The camera here is a pinhole that ray-casts the scene's solids exactly, so the shadow is the one a D415 would leave.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.robot.grasping.generation.support_footprint import generate_support_footprint_grasps
from tests.test_sfe_says_why_it_refused import (
    box_cloud,
    cylinder_beside_a_wall,
    hande_jaw,
)

#: The seen test's tolerance, millimetres: a point up to this far behind the measured surface still counts as reached.
SEEN_TOLERANCE_MM = 5.0


@dataclass(frozen=True)
class Plane:
    """The support: the plane ``z = height_mm``."""

    height_mm: float = 0.0


@dataclass(frozen=True)
class Cylinder:
    """A standing cylinder with its top cap."""

    centre_xy: tuple[float, float]
    radius_mm: float
    z0_mm: float
    z1_mm: float


@dataclass(frozen=True)
class Box:
    """An axis-aligned box."""

    lo: tuple[float, float, float]
    hi: tuple[float, float, float]


@dataclass
class DepthCamera:
    """A pinhole looking from ``eye`` at ``target`` that ray-casts ``solids`` into a z-depth image, BASE millimetres."""

    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    solids: tuple[Any, ...]
    width: int = 320
    height: int = 240
    focal_px: float = 280.0
    rotation: np.ndarray = field(init=False)
    depth: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        forward = np.asarray(self.target, dtype=np.float64) - np.asarray(self.eye, dtype=np.float64)
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0]) if abs(forward[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        self.rotation = np.column_stack([right, down, forward])   # camera axes in BASE: x right, y down, z ahead
        self.depth = self._render()

    def _rays(self) -> np.ndarray:
        u, v = np.meshgrid(np.arange(self.width, dtype=np.float64), np.arange(self.height, dtype=np.float64))
        camera = np.stack([(u - self.width / 2.0) / self.focal_px, (v - self.height / 2.0) / self.focal_px,
                           np.ones_like(u)], axis=-1)
        return camera.reshape(-1, 3) @ self.rotation.T             # per pixel, the BASE step for one mm of depth

    def _render(self) -> np.ndarray:
        origin = np.asarray(self.eye, dtype=np.float64)
        rays = self._rays()
        best = np.full(rays.shape[0], np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            for solid in self.solids:
                best = np.minimum(best, _hit(solid, origin, rays))
        return best.reshape(self.height, self.width)

    def seen(self, points: np.ndarray) -> np.ndarray:
        """True for a BASE point whose pixel's measured depth reaches at least to it, ``SEEN_TOLERANCE_MM`` tolerance."""
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        camera = (p - np.asarray(self.eye)) @ self.rotation
        z = camera[:, 2]
        ahead = z > 1.0
        safe = np.where(ahead, z, 1.0)
        u = np.rint(camera[:, 0] / safe * self.focal_px + self.width / 2.0).astype(np.int64)
        v = np.rint(camera[:, 1] / safe * self.focal_px + self.height / 2.0).astype(np.int64)
        inside = ahead & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        measured = np.full(p.shape[0], -np.inf)
        measured[inside] = self.depth[v[inside], u[inside]]
        return inside & np.isfinite(measured) & (measured >= z - SEEN_TOLERANCE_MM)


def _hit(solid: Any, origin: np.ndarray, rays: np.ndarray) -> np.ndarray:
    """The depth at which each ray first meets ``solid``, ``inf`` where it misses."""
    miss = np.full(rays.shape[0], np.inf)
    if isinstance(solid, Plane):
        t = (solid.height_mm - origin[2]) / rays[:, 2]
        return np.where(t > 0.0, t, miss)
    if isinstance(solid, Box):
        lo = (np.asarray(solid.lo) - origin)[None, :] / rays
        hi = (np.asarray(solid.hi) - origin)[None, :] / rays
        near = np.nanmax(np.minimum(lo, hi), axis=1)
        far = np.nanmin(np.maximum(lo, hi), axis=1)
        return np.where((near <= far) & (near > 0.0), near, miss)
    if isinstance(solid, Cylinder):
        cx, cy = solid.centre_xy
        ox, oy = origin[0] - cx, origin[1] - cy
        a = rays[:, 0] ** 2 + rays[:, 1] ** 2
        b = 2.0 * (ox * rays[:, 0] + oy * rays[:, 1])
        c = ox * ox + oy * oy - solid.radius_mm ** 2
        disc = b * b - 4.0 * a * c
        t_side = (-b - np.sqrt(np.where(disc >= 0.0, disc, np.nan))) / (2.0 * a)
        z_side = origin[2] + t_side * rays[:, 2]
        side = np.where((t_side > 0.0) & (z_side >= solid.z0_mm) & (z_side <= solid.z1_mm), t_side, miss)
        t_cap = (solid.z1_mm - origin[2]) / rays[:, 2]
        x_cap = ox + t_cap * rays[:, 0]
        y_cap = oy + t_cap * rays[:, 1]
        cap = np.where((t_cap > 0.0) & (x_cap ** 2 + y_cap ** 2 <= solid.radius_mm ** 2), t_cap, miss)
        return np.minimum(side, cap)
    raise TypeError(f"no ray cast for {type(solid).__name__}")


# --------------------------------------------------------------------------------------------------------------------
# The scene: the side-approach cylinder beside its wall, and a tall neighbour on the side the hand would come from
# --------------------------------------------------------------------------------------------------------------------

#: The cylinder of ``cylinder_beside_a_wall``: 40 mm across, 60 mm tall, on the support at z 0, centred on the origin.
CYLINDER = Cylinder((0.0, 0.0), 20.0, 0.0, 60.0)
#: Its wall, 12 mm off on +y, 120 mm tall, as a thin box the camera sees.
WALL = Box((-80.0, 32.0, 0.0), (80.0, 34.0, 120.0))
#: A tall neighbour on -y, 150 mm off the cylinder and 260 mm tall: past the reach of every tilted corridor and its
#: obstacle grid's dilation (a 90 degree corridor ends 142 mm out), so it refuses nothing as an obstacle.
NEIGHBOUR = Box((-60.0, -200.0, 0.0), (60.0, -170.0, 260.0))
#: A camera low on -y, beyond the neighbour: the neighbour's shadow falls over the cylinder's whole -y side.
LOW_ON_MINUS_Y = ((0.0, -520.0, 250.0), (0.0, 0.0, 40.0))
#: A camera high over the cylinder: it sees down both sides, the shadow is gone.
HIGH_OVERHEAD = ((0.0, -60.0, 650.0), (0.0, 0.0, 30.0))


def camera(view: tuple[tuple[float, float, float], tuple[float, float, float]], *solids: Any) -> DepthCamera:
    return DepthCamera(eye=view[0], target=view[1], solids=(Plane(0.0), *solids))


def tilt_deg(candidate: Any) -> float:
    return float(np.degrees(np.arccos(np.clip(-float(candidate.approach[2]), -1.0, 1.0))))


def open_corridor(candidate: Any, *, support_mm: float = 0.0) -> np.ndarray:
    """The samples SFE judges a candidate's way in by: both open fingers, from the fingertip back along the approach."""
    from src.robot.grasping.generation.support_footprint import _finger_points

    jaw = hande_jaw()
    approach = np.asarray(candidate.approach, dtype=np.float64)
    axis = np.asarray(candidate.closing_axis, dtype=np.float64)
    binormal = np.cross(approach, axis)
    binormal /= float(np.linalg.norm(binormal))
    anchor = np.asarray(candidate.position_mm, dtype=np.float64)
    half = jaw.aperture_mm / 2.0
    reach = jaw.finger_reach_mm + 80.0
    swept = np.concatenate([_finger_points(anchor + half * axis, approach, binormal, jaw, reach, 8.0),
                            _finger_points(anchor - half * axis, approach, binormal, jaw, reach, 8.0)])
    return swept[swept[:, 2] >= support_mm]


def run(seen: Any, *, max_candidates: int = 12, side: bool = True,
        neighbour: bool = True) -> tuple[list[Any], dict[str, int]]:
    cloud, wall = cylinder_beside_a_wall(12.0)
    assert wall is not None
    obstacles = np.vstack([wall, box_cloud(NEIGHBOUR.lo, NEIGHBOUR.hi, step_mm=4.0)]) if neighbour else wall
    counts: dict[str, int] = {}
    found = generate_support_footprint_grasps(cloud, support_height_mm=0.0, jaw=hande_jaw(),
                                              obstacle_points_base_mm=obstacles, max_candidates=max_candidates,
                                              refusals=counts, side_approaches=side, corridor_seen=seen)
    return found, counts


class TheCameraTests(unittest.TestCase):
    """The ray cast is what a D415 would measure, or the tests below prove nothing."""

    def test_the_neighbours_shadow_is_unseen_and_its_light_is_seen(self) -> None:
        low = camera(LOW_ON_MINUS_Y, CYLINDER, WALL, NEIGHBOUR)
        behind_the_neighbour = np.array([[0.0, -60.0, 40.0], [10.0, -40.0, 80.0]])
        in_front_of_it = np.array([[0.0, -200.0, 10.0]])
        self.assertFalse(low.seen(behind_the_neighbour).any())
        self.assertTrue(low.seen(in_front_of_it).all())
        high = camera(HIGH_OVERHEAD, CYLINDER, WALL, NEIGHBOUR)
        self.assertTrue(high.seen(behind_the_neighbour).all())

    def test_the_cylinders_top_is_measured_where_it_stands(self) -> None:
        high = camera(HIGH_OVERHEAD, CYLINDER, WALL, NEIGHBOUR)
        self.assertTrue(high.seen(np.array([[0.0, 0.0, 60.0]])).all())
        self.assertFalse(high.seen(np.array([[0.0, 0.0, 40.0]])).any(), "inside the cylinder, under its top")


class NoSideGraspThroughTheShadowTests(unittest.TestCase):
    def test_no_tilted_candidate_has_a_corridor_sample_in_the_shadow(self) -> None:
        """⭐ Red before: there was no side-approach switch and no seen test. The hand would lean away from the wall,
        into the -y side the neighbour hides from the camera: every candidate tilted 15 degrees or more that is offered
        sweeps only space a ray reached, and the ones that would not are counted as ``unseen_corridor``."""
        low = camera(LOW_ON_MINUS_Y, CYLINDER, WALL, NEIGHBOUR)
        found, counts = run(low.seen, max_candidates=200)
        self.assertTrue(found, "the vertical grasps stay")
        self.assertGreater(counts.get("unseen_corridor", 0), 0, counts)
        for candidate in found:
            if tilt_deg(candidate) >= 15.0 - 1e-6:
                self.assertTrue(low.seen(open_corridor(candidate)).all(),
                                f"a {tilt_deg(candidate):.0f} deg candidate sweeps space no ray reached")

    def test_seen_from_overhead_the_side_grasp_comes_back(self) -> None:
        """The control: the same scene from a camera that sees past the neighbour. The side grasp away from the wall
        is offered and wins, so the shadow, not the neighbour, is what refused it above."""
        high = camera(HIGH_OVERHEAD, CYLINDER, WALL, NEIGHBOUR)
        found, counts = run(high.seen)
        self.assertTrue(found)
        self.assertGreaterEqual(tilt_deg(found[0]), 15.0 - 1e-6, [round(tilt_deg(c)) for c in found])
        self.assertGreater(float(found[0].approach[1]), 0.0, "the hand leans away from the wall, toward -y")
        low_found, _ = run(camera(LOW_ON_MINUS_Y, CYLINDER, WALL, NEIGHBOUR).seen)
        self.assertNotEqual((tilt_deg(found[0]), float(found[0].approach[1])),
                            (tilt_deg(low_found[0]), float(low_found[0].approach[1])))

    def test_in_the_shadow_the_vertical_grasp_is_rank_zero(self) -> None:
        """Every tilt on the free side sweeps the shadow and every tilt toward the wall meets it: what is left to rank
        first is the vertical grasp, as before side approaches."""
        found, _counts = run(camera(LOW_ON_MINUS_Y, CYLINDER, WALL, NEIGHBOUR).seen)
        self.assertTrue(found)
        self.assertLess(tilt_deg(found[0]), 1.0, [round(tilt_deg(c)) for c in found])


class WithoutASeenTestNoNewTiltTests(unittest.TestCase):
    def test_without_corridor_seen_the_candidates_are_the_ones_offered_with_the_switch_off(self) -> None:
        """No seen test, no new tilt: every candidate offered is one the generator offers with side approaches off.
        Only their order may differ."""
        on, _ = run(None, max_candidates=500, neighbour=False)
        off, _ = run(None, max_candidates=500, side=False, neighbour=False)

        def same(a: Any, b: Any) -> bool:
            """One grasp, as the generator's own de-duplication reads one: within 4 mm, axes and approach aligned."""
            return (float(np.linalg.norm(a.position_mm - b.position_mm)) < 4.0
                    and abs(float(a.closing_axis @ b.closing_axis)) > 0.995 and float(a.approach @ b.approach) > 0.995)

        self.assertTrue(on)
        self.assertTrue(any(tilt_deg(c) >= 15.0 - 1e-6 for c in off), "the scene offers tilts with the switch off")
        for candidate in on:
            self.assertTrue(any(same(candidate, other) for other in off),
                            f"a {tilt_deg(candidate):.0f} deg candidate the switch off never offers")
        for candidate in off:
            self.assertTrue(any(same(candidate, other) for other in on))

    def test_a_seen_test_that_answers_for_the_wrong_number_of_points_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            run(lambda points: np.ones(len(points) + 1, dtype=bool))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
