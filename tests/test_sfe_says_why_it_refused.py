"""SFE says why it refused: every candidate ``_build`` throws away is counted, by cause and by the obstacle set it met.

The owner's Zollstock picks (2026-10-01, RC4 of the cell-fix plan): the calculator only saw the part the prompt named,
so SFE's obstacle grid was empty, and where it is not empty today SFE drops every obstacle hit without counting it
(``support_footprint.py`` before this change, the finger hit and the corridor hit). A pick that got no grasp therefore
could not say whether a neighbour the camera saw boxed the part in, a declared wall did, the part's own low fragments
did, or the part is simply too short for the hand, and nothing typed ``ALL_COLLIDED`` that the push waits for.

``generate_support_footprint_grasps(..., refusals=counts)`` now fills ``counts`` with one count per refused build, under
its cause: ``seen_fingers`` / ``seen_corridor`` (points a caller handed as observed), ``declared_fingers`` /
``declared_corridor`` (points it handed as declared), ``own_fragments`` (the target's own low fragments the footprint
left out), ``table``, ``span``, ``cone``, ``prism``, ``aperture``, and with side approaches ``unseen_corridor``.
``refusals=None`` is byte-identical to the generator before the counters, pinned here against digests taken on HEAD
``1d91e3a``. The stage stamps the counts as ``support_footprint_refused``.

The scene builders here are shared with the side-approach tests.
"""

from __future__ import annotations

import functools
import hashlib
import math
import unittest
from typing import Any

import numpy as np

from src.robot.grasping.collision import ParallelJawGripperModel
from src.robot.grasping.generation._support_footprint_stage import support_footprint_breakdowns
from src.robot.grasping.generation.support_footprint import (
    SupportFootprintJaw,
    generate_support_footprint_grasps,
)

#: The mat the owner's parts stand on, as the camera reads it: about 55 mm over the declared bench.
MAT_MM = 55.0


@functools.lru_cache(maxsize=None)
def hande_jaw() -> SupportFootprintJaw:
    """The shipped ``hande`` profile's jaw, the owner's Hand-E, built the way the cell builds it."""
    from src.config.loader import load_robot_section

    return SupportFootprintJaw.from_robot_config(load_robot_section(profile="hande"))


def default_jaw() -> SupportFootprintJaw:
    """The library's own jaw, as ``test_support_footprint_stage.py`` builds it."""
    return SupportFootprintJaw.from_model(ParallelJawGripperModel(), table_clearance_mm=5.0)


def cylinder_cloud(diameter_mm: float, height_mm: float, *, centre_xy: tuple[float, float] = (0.0, 0.0),
                   support_mm: float = 0.0, step_mm: float = 1.5) -> np.ndarray:
    """A standing cylinder's top disc and side, BASE mm, sampled as ``R_SCOPE/b_wall_prototype.py`` sampled it."""
    r = diameter_mm / 2.0
    cx, cy = centre_xy
    top = [(cx + x, cy + y, support_mm + height_mm) for x in np.arange(-r, r + 0.1, step_mm)
           for y in np.arange(-r, r + 0.1, step_mm) if x * x + y * y <= r * r]
    side = [(cx + r * np.cos(a), cy + r * np.sin(a), z) for a in np.arange(0.0, 2.0 * np.pi, 1.0 / r)
            for z in np.arange(support_mm + 1.0, support_mm + height_mm, step_mm)]
    return np.asarray(top + side, dtype=np.float64)


def box_cloud(lo: tuple[float, float, float], hi: tuple[float, float, float], *, step_mm: float = 2.0) -> np.ndarray:
    """An axis-aligned box's top and four sides, BASE mm (no bottom: it stands on something)."""
    xs = np.arange(lo[0], hi[0] + 1e-9, step_mm)
    ys = np.arange(lo[1], hi[1] + 1e-9, step_mm)
    zs = np.arange(lo[2], hi[2] + 1e-9, step_mm)
    faces = [np.array([(x, y, hi[2]) for x in xs for y in ys])]
    for x in (lo[0], hi[0]):
        faces.append(np.array([(x, y, z) for y in ys for z in zs]))
    for y in (lo[1], hi[1]):
        faces.append(np.array([(x, y, z) for x in xs for z in zs]))
    return np.vstack(faces).astype(np.float64)


def wall_cloud(y_mm: float, *, x_from: float = -80.0, x_to: float = 80.0, support_mm: float = 0.0,
               height_mm: float = 120.0, step_mm: float = 2.0) -> np.ndarray:
    """A thin vertical wall in the plane ``y = y_mm``, as a camera samples its face."""
    xs = np.arange(x_from, x_to + 0.1, step_mm)
    zs = np.arange(support_mm, support_mm + height_mm + 0.1, step_mm)
    xx, zz = np.meshgrid(xs, zs)
    return np.column_stack([xx.ravel(), np.full(xx.size, y_mm), zz.ravel()]).astype(np.float64)


def top_face_cloud(height_mm: float, *, step: float = 2.0, extent: float = 15.0) -> np.ndarray:
    """``test_support_footprint_stage.py``'s box top face in BASE mm, as a top-down camera sees it."""
    xs = np.arange(450.0 - extent, 450.0 + extent + 1e-9, step)
    ys = np.arange(-extent, extent + 1e-9, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, height_mm)])


# --------------------------------------------------------------------------------------------------------------------
# The scenes
# --------------------------------------------------------------------------------------------------------------------


def boxed_in_bar() -> tuple[np.ndarray, np.ndarray]:
    """A 30 mm bar, 160 long, 30 tall, on the 55 mm mat, between six 60 mm cylinders standing 3 mm off its long sides
    (the Zollstock in the pile, ``FIT/fit_zollstock_track_a.out``): ``(target, neighbours)``."""
    bar = box_cloud((-80.0, -15.0, MAT_MM), (80.0, 15.0, MAT_MM + 30.0))
    neighbours = np.vstack([cylinder_cloud(60.0, 100.0, centre_xy=(x, side * 48.0), support_mm=MAT_MM)
                            for x in (-50.0, 0.0, 50.0) for side in (-1.0, 1.0)])
    return bar, neighbours


def thin_bar_with_a_low_fragment() -> np.ndarray:
    """A 10 mm thin bar, 60 long and 80 tall, whose mask bled onto a neighbour's foot: a block 18 mm off its +x face,
    38 mm tall, under half the bar's height, so the reconstruction leaves it out of the footprint as the part's own low
    fragment, right where the open +x finger comes down to the bar's lowest anchors."""
    bar = box_cloud((-5.0, -30.0, 0.0), (5.0, 30.0, 80.0))
    foot = box_cloud((23.0, -8.0, 3.0), (29.0, 8.0, 38.0))
    return np.vstack([bar, foot])


def cylinder_beside_a_wall(gap_mm: float | None = 12.0) -> tuple[np.ndarray, np.ndarray | None]:
    """Track B's red-first scene: a cylinder 40 mm across and 60 mm tall, a 120 mm wall ``gap_mm`` off it on +y."""
    cloud = cylinder_cloud(40.0, 60.0)
    return cloud, (None if gap_mm is None else wall_cloud(20.0 + gap_mm))


def candidate_digest(candidates: list[Any]) -> str:
    """Every number a candidate carries, rounded to 1e-6 mm or rad and in order, hashed: the pin "byte-identical"."""
    digest = hashlib.sha256()
    for c in candidates:
        values = np.concatenate([[c.score], c.position_mm, c.approach, c.closing_axis,
                                 [c.grip_width_mm, c.contact_angle_rad, c.clearance_mm]]).astype(np.float64)
        digest.update((np.round(values, 6) + 0.0).tobytes())
    return f"{len(candidates)}:{digest.hexdigest()[:16]}"


def golden_runs() -> dict[str, list[Any]]:
    """The scenes the "byte-identical" pins are taken on, each run with the library's defaults."""
    bar, neighbours = boxed_in_bar()
    cylinder, wall = cylinder_beside_a_wall(12.0)
    fragment = thin_bar_with_a_low_fragment()
    cube = box_cloud((-20.0, -20.0, 0.0), (20.0, 20.0, 40.0))
    runs: dict[str, list[Any]] = {}
    for height in (50.0, 60.0, 70.0, 90.0, 120.0):
        for palm in (False, True):
            runs[f"top_face_h{height:.0f}_palm{int(palm)}"] = generate_support_footprint_grasps(
                top_face_cloud(height), support_height_mm=0.0, jaw=default_jaw(), obstacle_points_base_mm=None,
                max_candidates=12, palm_aware=palm)
    runs["hande_cylinder_wall_12"] = generate_support_footprint_grasps(
        cylinder, support_height_mm=0.0, jaw=hande_jaw(), obstacle_points_base_mm=wall, max_candidates=12)
    runs["hande_cylinder_free"] = generate_support_footprint_grasps(
        cylinder, support_height_mm=0.0, jaw=hande_jaw(), obstacle_points_base_mm=None, max_candidates=12)
    runs["hande_bar_boxed_in_seen"] = generate_support_footprint_grasps(
        bar, support_height_mm=MAT_MM, jaw=hande_jaw(), obstacle_points_base_mm=neighbours, max_candidates=12)
    runs["hande_thin_bar_fragment"] = generate_support_footprint_grasps(
        fragment, support_height_mm=0.0, jaw=hande_jaw(), obstacle_points_base_mm=None, max_candidates=12)
    runs["hande_cube_declared_wall"] = generate_support_footprint_grasps(
        cube, support_height_mm=0.0, jaw=hande_jaw(), obstacle_points_base_mm=None,
        rigid_obstacle_points_base_mm=wall_cloud(36.0, height_mm=80.0), max_candidates=12)
    for wall_y in (None, 40.0):
        # The library's 2F-85 on a 30 mm cube 50 tall: tilted grasps, and since the fingers stand at the anchor
        # (2026-10-06) straight ones too.
        runs[f"default_cube_tilted_wall_{wall_y}"] = generate_support_footprint_grasps(
            box_cloud((-15.0, -15.0, 0.0), (15.0, 15.0, 50.0)), support_height_mm=0.0, jaw=default_jaw(),
            obstacle_points_base_mm=None if wall_y is None else wall_cloud(wall_y, height_mm=60.0), max_candidates=40)
    return runs


#: ``golden_runs()`` since the fingers stand at the anchor and the grasps sit on the part (recorded 2026-10-06, Windows,
#: numpy as pinned); the counters change none of them.
HEAD_DIGESTS: dict[str, str] = {
    "top_face_h50_palm0": "12:5c94309281c66830",
    "top_face_h50_palm1": "12:75e67c551b8e1eed",
    "top_face_h60_palm0": "12:49d0c376bcb2cbae",
    "top_face_h60_palm1": "12:2a07c674f5bc4614",
    "top_face_h70_palm0": "12:142cc6753630feeb",
    "top_face_h70_palm1": "12:d22140189c7a3594",
    "top_face_h90_palm0": "12:d0c669ed6464769f",
    "top_face_h90_palm1": "12:f669685fb103d46b",
    "top_face_h120_palm0": "12:2cb5c4c140704070",
    "top_face_h120_palm1": "12:911f06b8144ff459",
    # The Hand-E's two cylinders since its fingers come to 1 mm of the support (2026-10-06), 3 mm before; and since a
    # round footprint's fan stands in the base frame and its lines pass through its centroid (2026-10-09): the same
    # twelve grasps a side, their lines 0.00 mm off the cylinder's axis (0.02 to 0.04 before), scores within 0.02.
    "hande_cylinder_wall_12": "12:76916993ef66a89e",
    "hande_cylinder_free": "12:0f053b1d0f553055",
    "hande_bar_boxed_in_seen": "0:e3b0c44298fc1c14",
    "hande_thin_bar_fragment": "6:63a94c5ac609b13d",
    "hande_cube_declared_wall": "12:2bce7bcef31b0bc3",
    "default_cube_tilted_wall_None": "12:6b69ffe34469825e",
    "default_cube_tilted_wall_40.0": "6:04d2d04392b2cb6a",
}


def _run(cloud: np.ndarray, **kwargs: Any) -> tuple[list[Any], dict[str, int]]:
    counts: dict[str, int] = {}
    kwargs.setdefault("support_height_mm", 0.0)
    kwargs.setdefault("jaw", hande_jaw())
    kwargs.setdefault("max_candidates", 12)
    found = generate_support_footprint_grasps(cloud, refusals=counts, **kwargs)
    return found, counts


#: Every cause a refusal is counted under, in the order the contract (cell-fix plan, contract 1) names them.
CAUSES = ("seen_fingers", "seen_corridor", "declared_fingers", "declared_corridor", "own_fragments", "table", "span",
          "cone", "prism", "aperture")


class TheCountersSayWhyTests(unittest.TestCase):
    def test_a_part_boxed_in_by_seen_points_is_refused_by_what_was_seen(self) -> None:
        """⭐ Red before: no counter existed. The Zollstock's neighbours handed as seen points: every grasp refused,
        and the counts say the camera's neighbours did it, not a declared body and not the part's own fragments."""
        bar, neighbours = boxed_in_bar()
        found, counts = _run(bar, support_height_mm=MAT_MM, obstacle_points_base_mm=neighbours)
        self.assertEqual([], found)
        self.assertGreater(counts.get("seen_corridor", 0), 0, counts)
        self.assertEqual(0, counts.get("declared_fingers", -1), counts)
        self.assertEqual(0, counts.get("declared_corridor", -1), counts)
        self.assertEqual(0, counts.get("own_fragments", -1), counts)

    def test_a_part_beside_declared_points_is_refused_by_what_was_declared(self) -> None:
        """The same neighbours handed as declared geometry (a container's walls): the declared counters, no seen one.
        The coarse grid fits no grasp between them, and the counts are its own; the fine grid it then runs (2026-10-06)
        fits one closing 20 degrees off the bar's axis, inside the friction cone, where the declared grid stays
        undilated."""
        bar, neighbours = boxed_in_bar()
        found, counts = _run(bar, support_height_mm=MAT_MM, obstacle_points_base_mm=None,
                             rigid_obstacle_points_base_mm=neighbours)
        self.assertTrue(all(19.9 <= math.degrees(c.contact_angle_rad) <= 20.1 for c in found),
                        [round(math.degrees(c.contact_angle_rad), 1) for c in found])
        self.assertGreater(counts.get("declared_fingers", 0) + counts.get("declared_corridor", 0), 0, counts)
        self.assertEqual((0, 0), (counts.get("seen_fingers"), counts.get("seen_corridor")), counts)

    def test_the_parts_own_low_fragment_is_counted_as_its_own(self) -> None:
        """A neighbour's foot bled into the mask is left out of the footprint and kept off the fingers: what it refuses
        is counted as the part's own fragments, never as a neighbour the camera saw."""
        found, counts = _run(thin_bar_with_a_low_fragment())
        self.assertGreater(counts.get("own_fragments", 0), 0, counts)
        self.assertEqual((0, 0), (counts.get("seen_fingers"), counts.get("seen_corridor")), counts)
        self.assertEqual((0, 0), (counts.get("declared_fingers"), counts.get("declared_corridor")), counts)
        del found

    def test_a_part_too_short_for_the_hand_is_refused_by_the_table(self) -> None:
        """A 12 mm part: the Hand-E's fingertip cannot clear the support's 1 mm, whatever the tilt (15 mm until the
        fingers came to 1 mm of the support, 2026-10-06; its least part is 13.5 mm)."""
        found, counts = _run(box_cloud((-20.0, -20.0, 0.0), (20.0, 20.0, 12.0)))
        self.assertEqual([], found)
        self.assertGreater(counts.get("table", 0), 0, counts)

    def test_a_part_wider_than_the_hand_is_refused_by_the_aperture(self) -> None:
        """A 60 mm cube does not fit a 50 mm Hand-E across either axis."""
        found, counts = _run(box_cloud((-30.0, -30.0, 0.0), (30.0, 30.0, 60.0)))
        self.assertEqual([], found)
        self.assertGreater(counts.get("aperture", 0), 0, counts)

    def test_every_cause_is_in_the_counts_even_at_zero(self) -> None:
        """A reader keys into the counts by name: every cause is there, a count of zero included."""
        _found, counts = _run(cylinder_cloud(40.0, 60.0))
        for cause in CAUSES:
            self.assertIn(cause, counts)
            self.assertIsInstance(counts[cause], int)

    def test_counts_add_up_over_calls_into_one_dict(self) -> None:
        """The dict is filled, not replaced: two calls into one dict count both."""
        bar, neighbours = boxed_in_bar()
        _found, once = _run(bar, support_height_mm=MAT_MM, obstacle_points_base_mm=neighbours)
        twice: dict[str, int] = {}
        for _ in range(2):
            generate_support_footprint_grasps(bar, support_height_mm=MAT_MM, jaw=hande_jaw(),
                                              obstacle_points_base_mm=neighbours, max_candidates=12, refusals=twice)
        self.assertEqual({key: 2 * value for key, value in once.items()}, twice)


class NothingChangesWithoutTheCountersTests(unittest.TestCase):
    def test_counting_changes_no_candidate(self) -> None:
        """The counted run and the uncounted run return the same candidates, number for number."""
        for name, kwargs in (
            ("cylinder beside a wall", {"obstacle_points_base_mm": cylinder_beside_a_wall(12.0)[1]}),
            ("thin bar with its fragment", {}),
            ("cube beside a declared wall", {"rigid_obstacle_points_base_mm": wall_cloud(36.0, height_mm=80.0)}),
        ):
            with self.subTest(name):
                cloud = (cylinder_beside_a_wall(12.0)[0] if name.startswith("cylinder")
                         else thin_bar_with_a_low_fragment() if name.startswith("thin")
                         else box_cloud((-20.0, -20.0, 0.0), (20.0, 20.0, 40.0)))
                plain = generate_support_footprint_grasps(cloud, support_height_mm=0.0, jaw=hande_jaw(),
                                                          max_candidates=12, **kwargs)
                counted, _counts = _run(cloud, **kwargs)
                self.assertEqual(candidate_digest(plain), candidate_digest(counted))
                self.assertEqual(len(plain), len(counted))
                for a, b in zip(plain, counted):
                    self.assertEqual(a.score, b.score)
                    np.testing.assert_array_equal(a.position_mm, b.position_mm)
                    np.testing.assert_array_equal(a.approach, b.approach)
                    np.testing.assert_array_equal(a.closing_axis, b.closing_axis)

    def test_refusals_none_is_byte_identical_to_head(self) -> None:
        """Every golden scene, with the library's defaults, gives the candidates recorded when the fingers were placed
        at the anchor and the grasps seated on the part (2026-10-06): before, HEAD 1d91e3a's."""
        self.assertTrue(HEAD_DIGESTS, "the HEAD digests were never recorded")
        runs = golden_runs()
        self.assertEqual(sorted(HEAD_DIGESTS), sorted(runs))
        for name, candidates in runs.items():
            with self.subTest(name):
                self.assertEqual(HEAD_DIGESTS[name], candidate_digest(candidates))


class TheStageStampsTheCountsTests(unittest.TestCase):
    def test_the_stage_stamps_support_footprint_refused(self) -> None:
        bar, neighbours = boxed_in_bar()
        breakdowns, telemetry = support_footprint_breakdowns(
            bar, camera_to_base=np.eye(4), support_height_mm=MAT_MM, jaw=hande_jaw(),
            obstacle_points_base_mm=neighbours, max_candidates=12)
        self.assertEqual([], breakdowns)
        refused = telemetry.get("support_footprint_refused")
        self.assertIsInstance(refused, dict, telemetry)
        assert isinstance(refused, dict)
        for cause in CAUSES:
            self.assertIn(cause, refused)
        self.assertGreater(refused["seen_corridor"], 0)

    def test_an_abstention_stamps_zero_counts(self) -> None:
        """Too few points to reconstruct: nothing was built, so nothing was refused, and the counts say so."""
        _breakdowns, telemetry = support_footprint_breakdowns(
            top_face_cloud(50.0, step=12.0), camera_to_base=np.eye(4), support_height_mm=0.0, jaw=default_jaw(),
            obstacle_points_base_mm=None, max_candidates=12)
        refused = telemetry.get("support_footprint_refused")
        self.assertIsInstance(refused, dict, telemetry)
        assert isinstance(refused, dict)
        self.assertEqual(0, sum(refused.values()))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
