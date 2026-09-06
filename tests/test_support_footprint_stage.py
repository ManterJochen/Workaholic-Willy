"""Tests for the SFE -> calculator adapter (``_support_footprint_stage``) and its telemetry.

The support-footprint stage is the DEFAULT geometry stage and had no test of its own -- it was
measured against the datagen reference instead, which catches rate regressions but says nothing about
the contract. These pin the part a live diagnosis depends on: what the stage REPORTS about what it
was handed and what it planned.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.geometry import Frame
from src.robot.grasping.collision import ParallelJawGripperModel, SupportPlane
from src.robot.grasping.collision.table_collision import gripper_table_clearance_mm
from src.robot.grasping.generation._support_footprint_stage import (
    support_footprint_breakdowns,
)
from src.robot.grasping.generation.support_footprint import (
    SupportFootprintJaw,
    generate_support_footprint_grasps,
)
from src.robot.grasping.planning import GraspPose

MIN_CLEARANCE_MM = 5.0
IDENTITY = np.eye(4, dtype=np.float64)


def top_face_cloud(height_mm: float, *, step: float = 2.0, extent: float = 15.0) -> np.ndarray:
    """A box's top face in BASE mm, as a top-down camera sees it."""
    xs = np.arange(450.0 - extent, 450.0 + extent + 1e-9, step)
    ys = np.arange(-extent, extent + 1e-9, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, height_mm)])


def run_stage(cloud: np.ndarray, *, support_height_mm: float = 0.0):  # noqa: ANN201
    return support_footprint_breakdowns(
        cloud,
        camera_to_base=IDENTITY,
        support_height_mm=support_height_mm,
        jaw=SupportFootprintJaw.from_model(
            ParallelJawGripperModel(), table_clearance_mm=MIN_CLEARANCE_MM),
        obstacle_points_base_mm=None,
        max_candidates=12,
    )


class TelemetryContractTests(unittest.TestCase):
    def test_reports_the_cloud_it_was_handed(self) -> None:
        """The height span is the check on 'was the object where I think it was'."""
        cloud = top_face_cloud(90.0)
        _breakdowns, tel = run_stage(cloud)
        self.assertEqual(tel["support_footprint_points"], cloud.shape[0])
        self.assertAlmostEqual(tel["support_footprint_cloud_z_min_mm"], 90.0, places=2)
        self.assertAlmostEqual(tel["support_footprint_cloud_z_max_mm"], 90.0, places=2)

    def test_reports_what_it_planned(self) -> None:
        """Anchor height + its own best clearance, both BASE mm, both from the score-ordered list."""
        cloud = top_face_cloud(90.0)
        breakdowns, tel = run_stage(cloud)
        self.assertTrue(breakdowns)
        self.assertGreater(tel["support_footprint_candidates"], 0)
        # The anchor sits inside the object, above the support and no higher than the observed top.
        self.assertGreater(tel["support_footprint_top_anchor_z_mm"], 0.0)
        self.assertLessEqual(tel["support_footprint_top_anchor_z_mm"], 90.0)
        # SFE only ever returns candidates it planned as clearing its own bar.
        self.assertGreaterEqual(tel["support_footprint_best_clearance_mm"], MIN_CLEARANCE_MM)

    def test_abstention_still_reports_the_cloud(self) -> None:
        """An abstention is a real answer -- but a diagnosis needs to see WHAT it abstained on.

        Below ``reconstruct_support_prism``'s 20-point floor the stage produces nothing; the cloud
        keys must survive that, because 'too few points' and 'no legal grasp' are opposite causes and
        the point count is what separates them (the voxel-cliff defect, measured 2026-08-14).
        """
        sparse = top_face_cloud(50.0, step=12.0)  # 3x3 = 9 points, under the floor
        self.assertLess(sparse.shape[0], 20)
        breakdowns, tel = run_stage(sparse)
        self.assertEqual(breakdowns, [])
        self.assertEqual(tel["support_footprint_candidates"], 0)
        self.assertIn("support_footprint_cloud_z_min_mm", tel)
        self.assertNotIn("support_footprint_top_anchor_z_mm", tel)

    def test_empty_cloud_is_not_a_crash(self) -> None:
        breakdowns, tel = run_stage(np.zeros((0, 3), dtype=np.float64))
        self.assertEqual(breakdowns, [])
        self.assertEqual(tel["support_footprint_points"], 0)
        self.assertNotIn("support_footprint_cloud_z_min_mm", tel)


class CameraFrameSeamTests(unittest.TestCase):
    def test_candidates_come_back_in_camera_frame(self) -> None:
        """SFE plans in BASE and the calculator filters in CAMERA; the rotation happens once, here."""
        cloud = top_face_cloud(90.0)
        # A camera 400 mm above the table, looking down, with the axes flipped as a real one is.
        camera_to_base = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 400.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        base_candidates = generate_support_footprint_grasps(
            cloud, support_height_mm=0.0,
            jaw=SupportFootprintJaw.from_model(
                ParallelJawGripperModel(), table_clearance_mm=MIN_CLEARANCE_MM),
            obstacle_points_base_mm=None, max_candidates=12)
        breakdowns, _tel = support_footprint_breakdowns(
            cloud, camera_to_base=camera_to_base, support_height_mm=0.0,
            jaw=SupportFootprintJaw.from_model(
                ParallelJawGripperModel(), table_clearance_mm=MIN_CLEARANCE_MM),
            obstacle_points_base_mm=None, max_candidates=12)
        self.assertTrue(breakdowns)
        self.assertEqual(len(breakdowns), len(base_candidates))
        for breakdown, candidate in zip(breakdowns, base_candidates):
            self.assertIs(breakdown.pose.frame, Frame.CAMERA)
            back = (camera_to_base[:3, :3] @ breakdown.pose.position_mm) + camera_to_base[:3, 3]
            np.testing.assert_allclose(back, candidate.position_mm, atol=1e-6)


class PlannerVersusEnvelopeTests(unittest.TestCase):
    """Two models of one gripper, and the seam between them has no other guard.

    SFE PLANS the table clearance from finger points; the calculator's collision filter RE-CHECKS it
    against the full :class:`ParallelJawGripperModel` envelope -- which also carries the PALM (70 mm
    wide, 35 mm deep). SFE has no palm, so on a strongly tilted grasp the palm can hang below
    everything SFE looked at.

    Measured over a 50-120 mm height sweep of a 30x30 top face: SFE is conservative on almost every
    candidate (by up to 23.6 mm), and OPTIMISTIC on one palm-limited case by **8.00 mm** -- it plans
    9.0 mm of clearance where the envelope measures 1.0 and the filter therefore rejects it. That
    direction is fail-safe (nothing under-clearing ever ships) but it costs candidates: SFE spends
    its budget on grasps the filter will kill, and a filter cannot move a grasp somewhere legal.

    The bound below is the tripwire. It passes today, it keeps passing if SFE is taught about the
    palm (the gap shrinks), and it fails if either model grows apart from the other.
    """

    MAX_OPTIMISM_MM = 10.0

    def _worst_optimism_mm(self, model: ParallelJawGripperModel, *, palm_aware: bool) -> float:
        jaw = SupportFootprintJaw.from_model(model, table_clearance_mm=MIN_CLEARANCE_MM)
        worst = 0.0
        for height in (50.0, 60.0, 70.0, 90.0, 120.0):
            candidates = generate_support_footprint_grasps(
                top_face_cloud(height), support_height_mm=0.0, jaw=jaw,
                obstacle_points_base_mm=None, max_candidates=12, palm_aware=palm_aware)
            self.assertTrue(candidates, f"no candidate at h={height} (palm_aware={palm_aware})")
            for candidate in candidates:
                worst = max(worst, float(candidate.clearance_mm)
                            - self._envelope_clearance_mm(candidate, model))
        return worst

    def _envelope_clearance_mm(self, candidate, model: ParallelJawGripperModel) -> float:
        approach = np.asarray(candidate.approach, dtype=np.float64)
        closing = np.asarray(candidate.closing_axis, dtype=np.float64)
        binormal = np.cross(approach, closing)
        binormal /= float(np.linalg.norm(binormal))
        position = np.asarray(candidate.position_mm, dtype=np.float64)
        half = 0.5 * float(candidate.grip_width_mm)
        pose = GraspPose(
            position_mm=position,
            rotation_matrix=np.column_stack([closing, binormal, approach]),
            grip_width_mm=float(candidate.grip_width_mm),
            score=float(np.clip(candidate.score, 0.0, 1.0)),
            confidence=float(np.clip(candidate.score, 0.0, 1.0)),
            contacts=(position - half * closing, position + half * closing),
            frame=Frame.BASE,
        )
        return gripper_table_clearance_mm(
            pose,
            SupportPlane(normal=np.array([0.0, 0.0, 1.0]), offset_mm=0.0, frame=Frame.BASE),
            gripper_model=model,
        )

    def test_planner_optimism_over_the_envelope_stays_bounded(self) -> None:
        worst = self._worst_optimism_mm(ParallelJawGripperModel(), palm_aware=False)
        self.assertLessEqual(
            worst, self.MAX_OPTIMISM_MM,
            f"SFE plans {worst:.2f} mm more table clearance than the envelope grants; the two models "
            "of one gripper have drifted further apart than measured (8.00 mm, palm-limited)")

    def test_palm_aware_planning_removes_the_optimism_entirely(self) -> None:
        """``palm_aware=True`` is the fix, and this is the assertion that it IS one.

        Measured: 8.00 mm of optimism over this sweep without it, 0.00 mm with it -- not merely
        smaller but gone, because the planner is now reading the same palm box the envelope checks
        rather than a second description of it. The candidate count is unchanged at every height, so
        on clean geometry the fix costs nothing; what it changes is which anchor the height solve
        picks when the approach tilts.
        """
        self.assertEqual(
            round(self._worst_optimism_mm(ParallelJawGripperModel(), palm_aware=True), 6), 0.0)

    def test_palm_aware_is_never_more_permissive(self) -> None:
        """The safety direction, pinned. Adding the palm can only LOWER the planned clearance.

        A candidate that survives with the palm in the reasoning would have survived without it, so
        this can refuse and can never admit -- which is what makes it safe to turn on before the
        rate has been measured.
        """
        model = ParallelJawGripperModel()
        jaw = SupportFootprintJaw.from_model(model, table_clearance_mm=MIN_CLEARANCE_MM)
        for height in (50.0, 60.0, 70.0, 90.0, 120.0):
            cloud = top_face_cloud(height)
            plain = generate_support_footprint_grasps(
                cloud, support_height_mm=0.0, jaw=jaw, obstacle_points_base_mm=None,
                max_candidates=12, palm_aware=False)
            palm = generate_support_footprint_grasps(
                cloud, support_height_mm=0.0, jaw=jaw, obstacle_points_base_mm=None,
                max_candidates=12, palm_aware=True)
            for candidate in palm:
                self.assertGreaterEqual(
                    self._envelope_clearance_mm(candidate, model), float(candidate.clearance_mm),
                    f"h={height}: a palm-aware candidate is still optimistic about the envelope")
            self.assertTrue(plain, f"no baseline candidate at h={height}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
