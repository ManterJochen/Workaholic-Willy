"""A side grasp wins where it keeps more room, and only there: the approaches are equal by geometry.

The owner, 2026-10-01: side grasps on, "gleichwertig nach Geometrie". SFE stopped at the first tilt that fit an anchor,
and its score gave being upright 0.15 of the blend, so a part standing 8-20 mm from a wall got the vertical grasp whose
fingers pass the wall closest. With ``side_approaches`` every tilt that fits is a candidate, upright weighs nothing, and
the clearance term is the room the grasp keeps: the smaller of the fingertip's height over the support and the open
corridor's least distance to an obstacle point (``SIDE_APPROACH_SCORE_WEIGHTS``). A tie goes to the vertical grasp.

Measured first on a prototype (``R_SCOPE/b_wall_prototype.out``, the cell-fix plan's evidence): a cylinder 40 mm across
and 60 mm tall, a 120 mm wall 12 mm off on +y, the owner's Hand-E: rank 0 tilted 30 degrees away from the wall, a
vertical candidate still listed; no wall, rank 0 vertical.

The library's default stays off, so a direct caller of ``support_footprint_breakdowns`` is unchanged: the stage's own
cases give HEAD 1d91e3a's breakdowns and telemetry, plus the refusal counts (``body_refused``).
"""

from __future__ import annotations

import hashlib
import unittest
from typing import Any

import numpy as np

from src.robot.grasping.collision import ParallelJawGripperModel
from src.robot.grasping.generation._support_footprint_stage import support_footprint_breakdowns
from src.robot.grasping.generation.body_grasps import body_grasps
from src.robot.grasping.generation.hand import SIDE_APPROACH_SCORE_WEIGHTS, SupportFootprintJaw
from tests._part_scenes import cylinder_beside_a_wall, hande_jaw, top_face_cloud
from tests.test_a_side_grasp_never_goes_through_unseen_space import (
    CYLINDER,
    HIGH_OVERHEAD,
    WALL,
    camera,
    open_corridor,
    tilt_deg,
)


def _side(gap_mm: float | None, *, max_candidates: int = 12) -> list[Any]:
    """The prototype's scene with side approaches on, seen by a camera high over it."""
    cloud, wall = cylinder_beside_a_wall(gap_mm)
    solids = (CYLINDER,) if gap_mm is None else (CYLINDER, WALL)
    return body_grasps(
        cloud, support_height_mm=0.0, jaw=hande_jaw(), obstacle_points_base_mm=wall, max_candidates=max_candidates,
        corridor_seen=camera(HIGH_OVERHEAD, *solids).seen, score_weights=SIDE_APPROACH_SCORE_WEIGHTS)


def _room_to_the_wall_mm(candidate: Any) -> float:
    _cloud, wall = cylinder_beside_a_wall(12.0)
    assert wall is not None
    corridor = open_corridor(candidate)
    return float(np.sqrt(((corridor[:, None, :] - wall[None, :, :]) ** 2).sum(axis=-1)).min())


class TheSideGraspWinsWhereItKeepsMoreRoomTests(unittest.TestCase):
    def test_beside_a_wall_rank_zero_leans_away_and_a_vertical_stays_listed(self) -> None:
        """⭐ Red before: there was no side-approach switch, and the vertical grasp was rank 0 (the prototype's
        "shipped" row). Now rank 0 is tilted, its hand leaning away from the wall: 30 degrees until 2026-10-05, 75 since
        the score pays for how a grasp sits on the part and the side grasp at the part's middle sits best, 30 again since
        the round footprint's fan stands in the base frame (2026-10-09): it closes exactly along the wall, 30, 45 and 60
        degrees tie and the most upright wins the tie. The vertical grasp stays listed by rule since then
        (``body_grasps._with_a_vertical``): by its score alone it ranked 13th, 0.009 behind."""
        found = _side(12.0)
        self.assertTrue(found)
        self.assertGreaterEqual(tilt_deg(found[0]), 15.0, [round(tilt_deg(c)) for c in found])
        self.assertGreater(float(found[0].approach[1]), 0.0, "the tool travels toward the wall: the hand leans away")
        self.assertTrue(any(tilt_deg(c) < 1.0 for c in found), "a vertical candidate is still listed")

    def test_without_the_wall_rank_zero_is_vertical(self) -> None:
        """No wall: a lean keeps no more room than straight down, so the hand comes straight down (the body ranks the
        tie vertical first, ``test_the_body_grasps_a_part_as_sfe_did``)."""
        found = _side(None)
        self.assertTrue(found)
        self.assertLess(tilt_deg(found[0]), 1.0, [round(tilt_deg(c)) for c in found])

    def test_the_side_grasp_keeps_more_room_than_the_best_vertical(self) -> None:
        """What makes it win: its open corridor stays further from the wall than the best vertical grasp's."""
        found = _side(12.0)
        vertical = next(c for c in found if tilt_deg(c) < 1.0)
        self.assertGreater(_room_to_the_wall_mm(found[0]), _room_to_the_wall_mm(vertical))
        self.assertGreater(found[0].score, vertical.score)

    def test_the_weights_are_the_owners_equal_geometry(self) -> None:
        """Friction cone 0.35, aperture 0.20, the room the grasp keeps 0.35, upright nothing, centred 0.10."""
        from src.robot.grasping.generation.hand import SIDE_APPROACH_SCORE_WEIGHTS

        self.assertEqual((0.35, 0.20, 0.35, 0.0, 0.10), SIDE_APPROACH_SCORE_WEIGHTS)


# --------------------------------------------------------------------------------------------------------------------
# The library caller is unchanged
# --------------------------------------------------------------------------------------------------------------------

_FLIPPED = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 400.0], [0.0, 0.0, 0.0, 1.0]])


def breakdown_digest(breakdowns: list[Any]) -> str:
    """Every number a breakdown carries, rounded to 1e-6 and in order, hashed."""
    digest = hashlib.sha256()
    for b in breakdowns:
        comp = b.components["support_footprint"]
        values = np.concatenate([b.pose.position_mm, np.asarray(b.pose.rotation_matrix).ravel(),
                                 [b.pose.grip_width_mm, b.total_score, b.geometric_score, comp["contact_angle_deg"],
                                  comp["support_clearance_mm"], comp["grip_width_mm"]]]).astype(np.float64)
        digest.update((np.round(values, 6) + 0.0).tobytes())
    return f"{len(breakdowns)}:{digest.hexdigest()[:16]}"


def stage_runs(**switches: Any) -> dict[str, tuple[list[Any], dict]]:
    """``test_support_footprint_stage.py``'s cases, and the wall scene on the Hand-E, through the stage."""
    jaw = SupportFootprintJaw.from_model(ParallelJawGripperModel(), table_clearance_mm=5.0)
    runs: dict[str, tuple[list[Any], dict]] = {}
    for name, cloud, transform in (("top90_identity", top_face_cloud(90.0), np.eye(4)),
                                   ("top90_flipped", top_face_cloud(90.0), _FLIPPED),
                                   ("sparse50", top_face_cloud(50.0, step=12.0), np.eye(4)),
                                   ("empty", np.zeros((0, 3)), np.eye(4))):
        runs[name] = support_footprint_breakdowns(cloud, camera_to_base=transform, support_height_mm=0.0, jaw=jaw,
                                                  obstacle_points_base_mm=None, max_candidates=12, **switches)
    for height in (50.0, 60.0, 70.0, 90.0, 120.0):
        runs[f"top{height:.0f}"] = support_footprint_breakdowns(
            top_face_cloud(height), camera_to_base=np.eye(4), support_height_mm=0.0, jaw=jaw,
            obstacle_points_base_mm=None, max_candidates=12, **switches)
    cloud, wall = cylinder_beside_a_wall(12.0)
    runs["hande_cylinder_wall_12"] = support_footprint_breakdowns(
        cloud, camera_to_base=_FLIPPED, support_height_mm=0.0, jaw=hande_jaw(), obstacle_points_base_mm=wall,
        max_candidates=12, **switches)
    return runs


def _telemetry(points: int, z: float | None = None, anchor: float | None = None,
               clearance: float | None = None, kept: int = 12) -> dict[str, Any]:
    # Every grasp these blocks are offered holds with the hand 4 mm off it (``support_footprint_robust``, 2026-10-09).
    out: dict[str, Any] = {"support_footprint_kept": kept, "support_footprint_points": points,
                           "support_footprint_robust": kept}
    if z is not None:
        out["support_footprint_cloud_z_min_mm"] = z
        out["support_footprint_cloud_z_max_mm"] = z
    if anchor is not None:
        out["support_footprint_top_anchor_z_mm"] = anchor
        out["support_footprint_best_clearance_mm"] = clearance
    return out


#: ``stage_runs()`` on HEAD 1d91e3a (recorded 2026-10-02): the breakdowns' digest and the telemetry the stage stamped.
#: Re-recorded 2026-10-06: SFE stands its fingers at the anchor and its score pays for how a grasp sits on the part, no
#: longer for height over the support (the owner, 2026-10-05: "wir müssen tiefer gehen"). Re-recorded 2026-10-14: the
#: stage asks the part's own body alone (``body_grasps``), SFE's prism no longer, so the digest is of the body's
#: breakdowns and the grasps sit at the blocks' middles; palm-aware runs came out alike, the body holding the palm off
#: the solids under the hand (``HandFloor``) where ``palm_aware`` was SFE's, and the switch left on 2026-10-16. The body's own counts (``body_grasps``,
#: ``body_kept``, ``body_refused``) are ``test_the_hand_grasps_the_parts_own_body``'s. Re-recorded 2026-10-14 evening:
#: the pads hold a part at least ``body_grasps.HOLD_OVERLAP_MM`` (10 mm) under its top, 3 before, so the anchors stand
#: on a grid 1 mm off the earlier one, and how they sit on the part weighs 0.6 of their third (the owner's cell:
#: "sehr knapp an der Oberseite"); every block keeps its 12 grasps, the best about its middle. Re-recorded 2026-10-15:
#: the hold read from the part's robust top (its 98th percentile), not its highest cell, and how the pads sit weighs 0.8
#: of their third (the owner: deeper first), so the twelve offered stand up to 3 mm otherwise; every block keeps its 12
#: grasps, the best about its middle.
HEAD_STAGE: dict[str, tuple[str, dict[str, Any]]] = {
    "top90_identity": ("12:e7585ee9d544c116", _telemetry(256, 90.0, 45.61, 34.89)),
    "top90_flipped": ("12:c34de951143aebdc", _telemetry(256, 90.0, 45.61, 34.89)),
    "sparse50": ("0:e3b0c44298fc1c14", _telemetry(9, 50.0, kept=0)),
    "empty": ("0:e3b0c44298fc1c14", _telemetry(0, kept=0)),
    "top50": ("12:645d70ec7288095f", _telemetry(256, 50.0, 35.61, 14.89)),
    "top60": ("12:9affb54781c67408", _telemetry(256, 60.0, 35.61, 16.89)),
    "top70": ("12:94860e3fb91a21a3", _telemetry(256, 70.0, 35.61, 26.89)),
    "top90": ("12:e7585ee9d544c116", _telemetry(256, 90.0, 45.61, 34.89)),
    "top120": ("12:9bda5d25c9ab6e84", _telemetry(256, 120.0, 61.61, 52.89)),
    # The Hand-E's fingers come to 1 mm of the support (2026-10-06).
    "hande_cylinder_wall_12": ("12:5a0da9b317bf7898", {
        "support_footprint_kept": 12, "support_footprint_points": 5593,
        "support_footprint_cloud_z_min_mm": 1.0, "support_footprint_cloud_z_max_mm": 60.0,
        "support_footprint_top_anchor_z_mm": 28.45, "support_footprint_best_clearance_mm": 28.0,
        "support_footprint_robust": 12}),
}


class TheLibraryCallerIsUnchangedTests(unittest.TestCase):
    def test_the_stage_by_default_gives_heads_breakdowns_and_telemetry_plus_the_counts(self) -> None:
        runs = stage_runs()
        self.assertEqual(sorted(HEAD_STAGE), sorted(runs))
        for name, (breakdowns, telemetry) in runs.items():
            with self.subTest(name):
                digest, head_telemetry = HEAD_STAGE[name]
                self.assertEqual(digest, breakdown_digest(breakdowns))
                stamped = dict(telemetry)
                counts = stamped.pop("body_refused")
                for key in ("body_grasps", "body_kept"):
                    stamped.pop(key)
                self.assertEqual(head_telemetry, stamped, "the stage stamps nothing else new")
                self.assertIsInstance(counts, dict)

    def test_switched_off_by_name_is_the_default(self) -> None:
        default, off = stage_runs(), stage_runs(side_approaches=False)
        for name in default:
            with self.subTest(name):
                self.assertEqual(breakdown_digest(default[name][0]), breakdown_digest(off[name][0]))
                self.assertEqual(default[name][1], off[name][1])

    def test_switched_on_the_stage_says_so_and_names_the_tilt_it_chose(self) -> None:
        cloud, wall = cylinder_beside_a_wall(12.0)
        breakdowns, telemetry = support_footprint_breakdowns(
            cloud, camera_to_base=_FLIPPED, support_height_mm=0.0, jaw=hande_jaw(), obstacle_points_base_mm=wall,
            max_candidates=12, side_approaches=True, corridor_seen=camera(HIGH_OVERHEAD, CYLINDER, WALL).seen)
        self.assertTrue(breakdowns)
        self.assertIs(True, telemetry["support_footprint_side_approaches"])
        self.assertIs(True, telemetry["support_footprint_corridor_seen"])
        # SFE leaned 30 degrees away from the wall for room. The part's own body leans every grasp on SFE's whole ladder
        # since 2026-10-16 (the owner: "alles von SFE"), a lean winning only with more room: its best grasp closes along
        # the wall, straight down its open corridor keeps 15.2 mm from it, leaning 75 degrees away 16.0.
        self.assertAlmostEqual(75.0, telemetry["support_footprint_top_tilt_deg"], places=1)
        _none, without = support_footprint_breakdowns(
            cloud, camera_to_base=_FLIPPED, support_height_mm=0.0, jaw=hande_jaw(), obstacle_points_base_mm=wall,
            max_candidates=12, side_approaches=True)
        self.assertIs(False, without["support_footprint_corridor_seen"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
