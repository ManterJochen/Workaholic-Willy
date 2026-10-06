"""The surface a part stands on is no neighbour, and a part too short for the hand says so.

Track A of the cell fixes reads the depth around a part as obstacles, by the planner world's rules. What the part stands
on is not one of them: where the camera world's support model holds a surface (``support_surfaces``), a pixel within
the band of that surface's local reading is the surface (F5), however the mat tilts and ripples. And the calculator
keeps the open hand as far from the solid the guard holds as the guard keeps it (``perceived_min_distance_mm``), so a try
is not spent on a grasp the guard refuses at the mat: an R3-type grasp 13 mm over the solid stays, 4 mm over it goes,
counted as ``rejected_support``. Since 2026-10-06 SFE plans on that solid itself (``support_footprint.HandFloor``): a
grasp the reading alone would put within the distance is planned higher, and the filter is left nothing to drop. Where
nothing is left for want of height, the reason is ``ALL_TABLE_CONFLICT`` and the sentence says the part is too short for
this hand (the Hand-E: about 28 mm, ``FIT/fit_sfe_flat_part.out``).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import (
    BENCH,
    EYE,
    MAT,
    MAT_MM,
    Frame_,
    _hande,
    calculator,
    compute,
    owner_rules,
)
from tests.test_a_side_grasp_never_goes_through_unseen_space import Cylinder


def tilted_mat_frame(part: Any, *, tilt_deg: float = 1.0, noise_mm: float = 1.3, seed: int = 11) -> Frame_:
    """The ray-cast frame of ``part`` standing on a mat that reads ``tilt_deg`` tilted about x and ripples by
    ``noise_mm``, as the owner's D415 reads it: the mat ``MAT_MM`` high where the part stands."""
    frame = Frame_((BENCH, MAT), part)
    rays = frame.camera._rays()
    eye = np.asarray(EYE, dtype=np.float64)
    slope = np.tan(np.radians(tilt_deg))
    # z = MAT_MM + slope * (y + 650): the plane through the part's foot.
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (MAT_MM + slope * (eye[1] + 650.0) - eye[2]) / (rays[:, 2] - slope * rays[:, 1])
    hit = eye + t[:, None] * rays
    inside = (t > 0) & (hit[:, 0] >= MAT.lo[0]) & (hit[:, 0] <= MAT.hi[0]) & (hit[:, 1] >= MAT.lo[1]) \
        & (hit[:, 1] <= MAT.hi[1])
    mat = np.where(inside, t, np.inf).reshape(frame.depth.shape)
    noisy = mat + np.random.default_rng(seed).normal(0.0, noise_mm, mat.shape)
    # Where the mat is seen (not the part): the tilted, rippled reading replaces the level one.
    replace = np.isfinite(mat) & ~frame.mask
    frame.depth = np.where(replace, noisy, frame.depth)
    return frame


def support_model_of(frame: Frame_) -> Any:
    """The camera world's support model of ``frame`` on the owner's cell."""
    from src.robot.safety.planning.perceived import DepthView
    from src.robot.safety.planning.self_envelope import ROBOT_BASES
    from src.robot.safety.planning.support_surfaces import find_supports
    from tests._cell_2026_10_01 import OWNER_LIMITS, owner_tuning

    return find_supports([DepthView(surface_depth_mm=frame.depth, intrinsics=frame.intrinsics,
                                    camera_to_base=frame.camera_to_base, name="EIH_Cam")],
                         limits=OWNER_LIMITS, tuning=owner_tuning(support_surfaces=True, support_allowance_mm=2.0),
                         base=ROBOT_BASES.get("ur10"))


class TheMatIsNoNeighbourTests(unittest.TestCase):
    """Mat pixels within the band of their local reading are no obstacle, however the mat tilts (F5)."""

    frame: Frame_
    model: Any

    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = tilted_mat_frame(Cylinder((0.0, -650.0), 20.0, MAT_MM - 5.0, MAT_MM + 40.0))
        cls.model = support_model_of(cls.frame)

    def test_the_model_finds_the_mat_tilted(self) -> None:
        surface = self.model.surface_under(np.array([[0.0, -650.0]]))
        self.assertIsNotNone(surface)
        self.assertAlmostEqual(1.0, surface.tilt_deg, delta=0.4)

    def test_with_the_model_no_pixel_of_the_mat_is_an_obstacle(self) -> None:
        from src.robot.grasping.generation.scene_obstacles import scene_obstacle_points

        seen = scene_obstacle_points(self.frame.depth, self.frame.mask, self.frame.intrinsics,
                                     self.frame.camera_to_base, rules=owner_rules(), support_height_mm=MAT_MM,
                                     support_model=self.model)
        self.assertEqual(0, seen.points_base_mm.shape[0])
        self.assertGreater(seen.counts["support"], 1000)

    def test_without_it_the_high_side_of_the_tilted_mat_would_be_a_neighbour(self) -> None:
        """The control: one level floor at the part's support reads the mat's high side, 2-4 mm up 150-250 mm off, as
        obstacles. The model is what keeps it a support."""
        from src.robot.grasping.generation.scene_obstacles import scene_obstacle_points

        seen = scene_obstacle_points(self.frame.depth, self.frame.mask, self.frame.intrinsics,
                                     self.frame.camera_to_base, rules=owner_rules(), support_height_mm=MAT_MM - 2.0,
                                     support_model=None)
        self.assertGreater(seen.points_base_mm.shape[0], 0)

    def test_the_calculator_keeps_the_grasps_it_offers_without_the_scene(self) -> None:
        on = compute(calculator(scene=True), self.frame, support_model=self.model)
        off = compute(calculator(scene=False), self.frame)
        self.assertEqual(off.telemetry["support_footprint_kept"], on.telemetry["support_footprint_kept"])
        self.assertEqual(0, on.telemetry["scene_obstacle_points"])


class TooShortForThisHandTests(unittest.TestCase):
    """A part lower than the Hand-E grips at any tilt: ``ALL_TABLE_CONFLICT``, "too short for this hand"."""

    def test_a_12_mm_part_is_too_short_and_says_so(self) -> None:
        """Under the Hand-E's least height: 1 mm over the support, the fingertip's 10.45 mm and 2 mm (2026-10-06, the
        owner's "bis auf 1 mm"; 15.5 mm at the 3 mm of 2026-10-05, 28 mm until the fingers stood at the anchor)."""
        from src.robot.grasping.generation.scene_obstacles import no_grasp_said
        from src.robot.grasping.types.feedback import GraspFailureReason

        frame = Frame_((BENCH, MAT), Cylinder((0.0, -650.0), 20.0, MAT_MM, MAT_MM + 12.0))
        result = compute(calculator(scene=True), frame)
        self.assertEqual((), result.candidates)
        self.assertEqual((GraspFailureReason.ALL_TABLE_CONFLICT, GraspFailureReason.RESCAN_RECOMMENDED),
                         result.reasons)
        said = no_grasp_said(result)
        self.assertIn("too short for this hand", said)
        self.assertIn("less than about 13 mm", said)

    def test_a_25_mm_part_gets_grasps(self) -> None:
        """The owner's flat parts, 24 to 26 mm tall, which the hand did not grip until 2026-10-06."""
        frame = Frame_((BENCH, MAT), Cylinder((0.0, -650.0), 20.0, MAT_MM, MAT_MM + 25.0))
        self.assertTrue(compute(calculator(scene=True), frame).candidates)

    def test_the_least_height_is_the_one_SFE_grips_at(self) -> None:
        """``least_part_height_mm`` restates SFE's ladder: just under it SFE offers nothing, just over it a grasp."""
        from src.robot.grasping.generation.scene_obstacles import least_part_height_mm
        from src.robot.grasping.generation.support_footprint import SupportFootprintJaw, generate_support_footprint_grasps
        from tests.test_sfe_says_why_it_refused import cylinder_cloud

        jaw = SupportFootprintJaw.from_robot_config(_hande())
        least = least_part_height_mm(jaw)
        self.assertAlmostEqual(13.45, least, delta=0.05)
        for height, expect in ((least - 1.0, False), (least + 1.5, True)):
            with self.subTest(height=height):
                cloud = cylinder_cloud(40.0, height, support_mm=MAT_MM)
                found = generate_support_footprint_grasps(cloud, support_height_mm=MAT_MM, jaw=jaw)
                self.assertEqual(expect, bool(found))


class TheOpenHandKeepsTheGuardsDistanceFromTheMatTests(unittest.TestCase):
    """An R3-type grasp straight down beside a part on the mat: 13 mm over the mat's solid stays, 4 mm over it goes."""

    def _solid(self, top_mm: float, tilt_deg: float = 1.0) -> Any:
        turn = Rotation.from_euler("x", tilt_deg, degrees=True).as_matrix()
        half = np.array([200.0, 150.0, 30.0])
        centre = np.array([0.0, -650.0, 0.0]) - turn @ np.array([0.0, 0.0, half[2]]) + np.array([0.0, 0.0, top_mm])
        return SimpleNamespace(name="seen_s00_support0", centre_mm=centre, rotation=turn, half_extents_mm=half)

    def _verdict(self, gap_mm: float) -> Any:
        from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry
        from src.robot.grasping.generation.scene_obstacles import envelope_verdicts

        cfg = _hande()
        hand = build_gripper_geometry(cfg.grasping.gripper_geometry)
        top = 62.0
        # Straight down: the closing axis +x, the approach -z; the fingertips stand fingertip_depth under the grasp.
        tip = float(cfg.grasping.gripper_geometry.parallel_jaw.fingertip_depth_mm)
        position = np.array([0.0, -650.0, top + gap_mm + tip])
        turn = np.column_stack([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        return envelope_verdicts([(position, turn)], gripper_model=hand, open_width_mm=cfg.gripper.max_width_mm,
                                 solids=(self._solid(top, tilt_deg=0.0),), support_distance_mm=5.0)[0]

    def test_13_mm_over_the_solid_stays(self) -> None:
        verdict = self._verdict(13.0)
        self.assertEqual("", verdict.refused)
        self.assertAlmostEqual(13.0, verdict.distance_mm, delta=1e-6)

    def test_4_mm_over_it_is_refused_at_the_support(self) -> None:
        verdict = self._verdict(4.0)
        self.assertEqual("support", verdict.refused)
        self.assertEqual("seen_s00_support0", verdict.nearest)

    def _held(self, top_mm: float) -> Any:
        """A support model holding one level solid under the part, its top at ``top_mm``: a reading of the mat the part's
        own foot does not show, so SFE plans on the slab while the guard holds the solid."""
        solid = self._solid(top_mm, tilt_deg=0.0)
        return solid, SimpleNamespace(
            solids=(solid,), height_under=lambda xy: None, local_plane_under=lambda xy: None,
            is_support_point=lambda points: np.zeros(np.asarray(points).reshape(-1, 3).shape[0], dtype=bool))

    def _offered(self) -> "tuple[Any, list[float]]":
        """A 40 mm part on the slab, what SFE offers for it with the scene and nothing held, and each grasp's open hand's
        distance to a solid whose top is the slab's."""
        from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry
        from src.robot.grasping.generation.scene_obstacles import envelope_verdicts

        frame = Frame_((BENCH, MAT), Cylinder((0.0, -650.0), 20.0, MAT_MM, MAT_MM + 40.0))
        offered = compute(calculator(scene=True), frame)
        cfg = _hande()
        hand = build_gripper_geometry(cfg.grasping.gripper_geometry)
        poses = [(g.position, np.column_stack([g.axis, np.cross(g.approach, g.axis), g.approach]))
                 for g in offered.candidates]
        reference = envelope_verdicts(poses, gripper_model=hand, open_width_mm=cfg.gripper.max_width_mm,
                                      solids=(self._solid(MAT_MM, tilt_deg=0.0),))
        return frame, [v.distance_mm for v in reference]

    def _kept_from(self, result: Any, solid: Any) -> "list[float]":
        """Each offered grasp's open hand's distance to ``solid``, as the guard's filter measures it."""
        from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry
        from src.robot.grasping.generation.scene_obstacles import envelope_verdicts

        cfg = _hande()
        hand = build_gripper_geometry(cfg.grasping.gripper_geometry)
        poses = [(g.position, np.column_stack([g.axis, np.cross(g.approach, g.axis), g.approach]))
                 for g in result.candidates]
        return [v.distance_mm for v in envelope_verdicts(poses, gripper_model=hand,
                                                         open_width_mm=cfg.gripper.max_width_mm, solids=(solid,))]

    def test_a_held_solid_over_every_grasp_planned_on_the_reading_lifts_them_all_over_it(self) -> None:
        """A held solid raised until every grasp planned on the reading alone stands 3 mm or less over it: SFE plans on
        the solid instead (``support_footprint.HandFloor``), so every grasp offered keeps the guard's 5 mm and the
        post-hoc filter has nothing left to drop. Until 2026-10-06 the filter dropped all of them, and a part the
        hand could take higher up got no grasp."""
        frame, over_slab = self._offered()
        self.assertGreater(len(over_slab), 0)
        solid, held = self._held(MAT_MM + max(over_slab) - 3.0)
        on = compute(calculator(scene=True), frame, support_model=held)
        self.assertGreater(len(on.candidates), 0)
        self.assertEqual(0, on.telemetry["rejected_support"])
        self.assertGreaterEqual(min(self._kept_from(on, solid)), 5.0 - 1e-6)

    def test_a_held_solid_the_hand_cannot_keep_its_distance_from_is_a_table_conflict_said(self) -> None:
        """Held up to 5 mm under the part's top, no finger fits over it at the guard's distance: no grasp, the support
        named, the part too short for the hand over what the guard holds."""
        from src.robot.grasping.generation.scene_obstacles import no_grasp_said
        from src.robot.grasping.types.feedback import GraspFailureReason

        frame, _ = self._offered()
        _, held = self._held(MAT_MM + 40.0 - 5.0)
        on = compute(calculator(scene=True), frame, support_model=held)
        self.assertEqual((), on.candidates)
        self.assertEqual((GraspFailureReason.ALL_TABLE_CONFLICT, GraspFailureReason.RESCAN_RECOMMENDED), on.reasons)
        self.assertIn("too short", no_grasp_said(on))

    def test_a_held_solid_6_mm_under_every_open_hand_takes_nothing(self) -> None:
        frame, over_slab = self._offered()
        solid, held = self._held(MAT_MM + min(over_slab) - 6.0)
        on = compute(calculator(scene=True), frame, support_model=held)
        # As many grasps or more: the palm, which the guard holds and the reading's check did not, is planned in SFE
        # over the held solid, so a grasp the filter dropped for the reading's table is replaced there.
        self.assertGreaterEqual(len(on.candidates), len(over_slab))
        self.assertEqual(0, on.telemetry["rejected_support"])
        self.assertGreaterEqual(on.telemetry["scene_held_least_mm"], 5.0)
        self.assertGreaterEqual(min(self._kept_from(on, solid)), 5.0 - 1e-6)

    def test_between_the_two_every_grasp_is_planned_over_the_solid(self) -> None:
        """Held between the lowest and the highest grasp planned on the reading: the low ones are planned higher, none
        comes within the guard's 5 mm, none is dropped after."""
        frame, over_slab = self._offered()
        middle = (min(over_slab) + max(over_slab)) / 2.0
        solid, held = self._held(MAT_MM + middle - 5.0)
        on = compute(calculator(scene=True), frame, support_model=held)
        self.assertGreater(len(on.candidates), 0)
        self.assertEqual(0, on.telemetry["rejected_support"])
        self.assertGreaterEqual(min(self._kept_from(on, solid)), 5.0 - 1e-6)

    def test_all_of_them_near_the_mat_is_a_table_conflict_said(self) -> None:
        from src.robot.grasping.generation.scene_obstacles import why_no_grasp
        from src.robot.grasping.types.feedback import GraspFailureReason

        reasons, said = why_no_grasp({"geometry_stage": "support_footprint", "support_footprint_kept": 3,
                                      "rejected_support": 3, "scene_support_distance_mm": 5.0})
        self.assertEqual((GraspFailureReason.ALL_TABLE_CONFLICT, GraspFailureReason.RESCAN_RECOMMENDED), reasons)
        self.assertEqual("every grasp brings the hand within 5 mm of the support surface the part stands on", said)


class TheReasonsReadTheCountsTests(unittest.TestCase):
    """``why_no_grasp`` reads B.1's counts in the plan's order: a neighbour before the support (the boxed-in bar counts
    far more ``table`` than ``seen_corridor``), a declared body, the part's own fragments last."""

    def _read(self, **refused: int) -> Any:
        from src.robot.grasping.generation.scene_obstacles import why_no_grasp

        counts = {"seen_fingers": 0, "seen_corridor": 0, "declared_fingers": 0, "declared_corridor": 0,
                  "own_fragments": 0, "table": 0, "aperture": 0}
        counts.update(refused)
        return why_no_grasp({"geometry_stage": "support_footprint", "support_footprint_kept": 0,
                             "support_footprint_refused": counts, "scene_part_height_mm": 32.0,
                             "scene_least_part_height_mm": 27.9, "scene_table_clearance_mm": 5.0})

    def test_a_neighbour_wins_over_the_table(self) -> None:
        from src.robot.grasping.types.feedback import GraspFailureReason as R

        reasons, said = self._read(seen_corridor=6, table=111, aperture=117)
        self.assertEqual((R.ALL_COLLIDED, R.RESCAN_RECOMMENDED), reasons)
        self.assertIn("neighbour", said)

    def test_a_declared_body_is_a_table_conflict(self) -> None:
        from src.robot.grasping.types.feedback import GraspFailureReason as R

        reasons, said = self._read(declared_corridor=4, table=20)
        self.assertEqual((R.ALL_TABLE_CONFLICT, R.RESCAN_RECOMMENDED), reasons)
        self.assertIn("declared", said)

    def test_a_tall_enough_part_the_table_refused_is_no_too_short(self) -> None:
        from src.robot.grasping.types.feedback import GraspFailureReason as R

        reasons, said = self._read(table=40)
        self.assertEqual((R.ALL_TABLE_CONFLICT, R.RESCAN_RECOMMENDED), reasons)
        self.assertNotIn("too short", said)
        self.assertIn("within 5 mm of the support", said)

    def test_a_list_the_table_check_emptied_says_its_clearance(self) -> None:
        """SFE offered grasps and the post-hoc table check refused them all: the clearance it was held to is said."""
        from src.robot.grasping.generation.scene_obstacles import why_no_grasp
        from src.robot.grasping.types.feedback import GraspFailureReason as R

        for stamps in ({"scene_table_clearance_mm": 5.0}, {"table_clearance_required_mm": 5.0}):
            with self.subTest(stamps):
                reasons, said = why_no_grasp({"geometry_stage": "support_footprint", "support_footprint_kept": 2,
                                              "rejected_table": 2, **stamps})
                self.assertEqual((R.ALL_TABLE_CONFLICT, R.RESCAN_RECOMMENDED), reasons)
                self.assertEqual("every grasp brings the hand within 5 mm of the support the part stands on", said)

    def test_the_parts_own_fragments_alone_are_no_valid_grasp(self) -> None:
        from src.robot.grasping.types.feedback import GraspFailureReason as R

        reasons, _ = self._read(own_fragments=9)
        self.assertEqual((R.NO_VALID_GRASP, R.RESCAN_RECOMMENDED), reasons)


class TheDistanceIsExactTests(unittest.TestCase):
    """The box distance the envelope filter reads: never more than a dense sampling of the two boxes measures, never
    less than that less its sampling step, and zero where they overlap."""

    def test_against_a_dense_sampling(self) -> None:
        from src.robot.grasping.generation.scene_obstacles import box_distances_mm

        rng = np.random.default_rng(3)

        def surface(c: np.ndarray, turn: np.ndarray, half: np.ndarray, n: int = 30) -> np.ndarray:
            g = np.linspace(-1.0, 1.0, n)
            u, v = (a.ravel() for a in np.meshgrid(g, g))
            faces = []
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    local = np.zeros((u.size, 3))
                    a, b = (i for i in range(3) if i != axis)
                    local[:, axis], local[:, a], local[:, b] = sign, u, v
                    faces.append(local)
            return c + (np.vstack(faces) * half) @ turn.T

        for k in range(60):
            ca, cb = rng.uniform(-60, 60, 3), rng.uniform(-60, 60, 3)
            ra = Rotation.random(random_state=k).as_matrix()
            rb = Rotation.random(random_state=500 + k).as_matrix()
            ha, hb = rng.uniform(3, 40, 3), rng.uniform(3, 40, 3)
            exact = float(box_distances_mm(ca[None], ra[None], ha[None], cb[None], rb[None], hb[None])[0])
            sampled = float(cKDTree(surface(cb, rb, hb)).query(surface(ca, ra, ha))[0].min())
            if exact > 0.0:
                self.assertLessEqual(exact, sampled + 1e-6)
                self.assertGreaterEqual(exact, sampled - 4.0)
            else:
                inside = (np.all(np.abs((surface(cb, rb, hb) - ca) @ ra) <= ha, axis=1).any()
                          or np.all(np.abs((surface(ca, ra, ha) - cb) @ rb) <= hb, axis=1).any())
                self.assertTrue(inside or sampled < 4.0, (k, sampled))


if __name__ == "__main__":
    unittest.main()
