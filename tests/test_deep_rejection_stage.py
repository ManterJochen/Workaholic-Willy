"""The geometric rejection stage both grasp generators run, and the frame check that guards it.

⛔ WHY IT EXISTS. The DL calculator had NO geometric filter: `gripper_model`, `support_plane`,
`min_table_clearance_mm` and `workspace` all reached `_propose` and were swallowed by `**kwargs`. Its
only rejection was a width comparison. On the analytic stack the equivalent filter is worth 22 % ->
64 % precision, and `SafetyPreflight` lives in the DRIVER -- so an unfiltered bad candidate does not
fall through to the next one, the pick FAILS at the arm.

⚠ AND THE FRAME. Nothing in the collision stack compares frames: `validate_grasp_collision` reads
neither `pose.frame` nor `support_plane.frame`. A mis-framed filter would not raise, it would return a
plausible wrong number -- which is how the last six frame defects in this repo survived, most recently
a BASE plane against CAMERA poses that measured a median 577 mm "clearance" where the true clearance
was -7.01 mm, rejecting 0.4 % of candidates instead of 65.5 %.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.geometry import Frame
from src.robot.grasping.collision import (
    REJECTION_KEYS,
    filter_candidates,
    grasp_point_to_pose,
    rejection_reasons,
)
from src.robot.grasping.collision.table_collision import SupportPlane
from src.robot.grasping.types.feedback import GraspFailureReason
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint


def _point(z_mm: float, *, frame: GraspFrame = GraspFrame.BASE, width: float = 40.0) -> GraspPoint:
    return GraspPoint(position=np.array([0.0, 0.0, z_mm]), approach=np.array([0.0, 0.0, -1.0]),
                      axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=width, score=0.5, frame=frame)


class TheFrameGuardTests(unittest.TestCase):
    """⛔ THE SEVENTH FRAME DEFECT, PRE-EMPTED."""

    def test_a_base_pose_against_a_camera_plane_is_REFUSED(self) -> None:
        poses = [grasp_point_to_pose(_point(60.0, frame=GraspFrame.BASE))]
        plane = SupportPlane(normal=np.array([0.0, 0.0, 1.0]), offset_mm=0.0, frame=Frame.CAMERA)
        with self.assertRaises(ValueError) as caught:
            filter_candidates(poses, support_plane=plane)
        message = str(caught.exception)
        self.assertIn("camera", message)
        self.assertIn("base", message)
        self.assertIn("to_camera_frame", message, "the refusal must name the way out")

    def test_matching_frames_pass(self) -> None:
        for frame, plane_frame in ((GraspFrame.BASE, Frame.BASE),
                                   (GraspFrame.CAMERA, Frame.CAMERA)):
            with self.subTest(frame=frame.value):
                poses = [grasp_point_to_pose(_point(200.0, frame=frame))]
                plane = SupportPlane(normal=np.array([0.0, 0.0, 1.0]), offset_mm=0.0,
                                     frame=plane_frame)
                self.assertEqual(len(filter_candidates(poses, support_plane=plane).kept), 1)

    def test_candidates_in_two_different_frames_are_refused(self) -> None:
        """One plane cannot be correct for two frames, and quietly using it for both is the defect."""
        poses = [grasp_point_to_pose(_point(200.0, frame=GraspFrame.BASE)),
                 grasp_point_to_pose(_point(200.0, frame=GraspFrame.CAMERA))]
        plane = SupportPlane(normal=np.array([0.0, 0.0, 1.0]), offset_mm=0.0, frame=Frame.BASE)
        with self.assertRaises(ValueError):
            filter_candidates(poses, support_plane=plane)

    def test_no_plane_means_no_frame_question(self) -> None:
        """A caller with no plane has nothing to disagree about, and must not be refused."""
        poses = [grasp_point_to_pose(_point(10.0, frame=GraspFrame.BASE))]
        self.assertEqual(len(filter_candidates(poses).kept), 1)


class ItActuallyRejectsTests(unittest.TestCase):
    def test_a_grasp_below_the_table_is_rejected(self) -> None:
        """The table check, in the frame the deep path works in."""
        plane = SupportPlane(normal=np.array([0.0, 0.0, 1.0]), offset_mm=0.0, frame=Frame.BASE)
        low = grasp_point_to_pose(_point(1.0))
        high = grasp_point_to_pose(_point(300.0))
        outcome = filter_candidates([low, high], support_plane=plane, min_table_clearance_mm=20.0)
        self.assertEqual(outcome.kept, (1,), outcome.telemetry)
        self.assertEqual(int(outcome.telemetry["rejected_table"]), 1)

    def test_it_reports_how_far_the_table_check_missed_by(self) -> None:
        """A bare count cannot tell "too short to grasp at all" from "the bar is a millimetre too
        strict" -- opposite conclusions, and a sim gate once sat on exactly that fork at 12 of 12."""
        plane = SupportPlane(normal=np.array([0.0, 0.0, 1.0]), offset_mm=0.0, frame=Frame.BASE)
        outcome = filter_candidates([grasp_point_to_pose(_point(1.0))], support_plane=plane,
                                    min_table_clearance_mm=20.0)
        self.assertIn("table_clearance_best_mm", outcome.telemetry)
        self.assertEqual(outcome.telemetry["table_clearance_required_mm"], 20.0)

    def test_the_plane_offset_is_stamped_WITH_ITS_FRAME(self) -> None:
        """Stamped untagged, this offset once read as a BASE height of -311 mm against a table at 0
        and sent a session hunting a frame defect that did not exist."""
        for plane_frame, expected in ((Frame.BASE, "support_plane_offset_base_mm"),
                                      (Frame.CAMERA, "support_plane_offset_camera_mm")):
            with self.subTest(frame=plane_frame.value):
                grasp = GraspFrame.BASE if plane_frame is Frame.BASE else GraspFrame.CAMERA
                outcome = filter_candidates(
                    [grasp_point_to_pose(_point(300.0, frame=grasp))],
                    support_plane=SupportPlane(normal=np.array([0.0, 0.0, 1.0]), offset_mm=-7.0,
                                               frame=plane_frame))
                self.assertIn(expected, outcome.telemetry)

    def test_a_grasp_outside_the_workspace_is_rejected_before_the_expensive_check(self) -> None:
        class _Box:
            def contains(self, position: np.ndarray) -> bool:
                return bool(abs(float(position[2])) < 100.0)

        outcome = filter_candidates([grasp_point_to_pose(_point(500.0))], workspace=_Box())
        self.assertEqual(outcome.kept, ())
        self.assertEqual(int(outcome.telemetry["rejected_workspace"]), 1)

    def test_with_nothing_to_filter_with_everything_survives(self) -> None:
        """Byte-identical to life before this stage: no plane, no box, no cloud -> no rejections."""
        poses = [grasp_point_to_pose(_point(z)) for z in (10.0, 50.0, 300.0)]
        outcome = filter_candidates(poses)
        self.assertEqual(len(outcome.kept), 3)
        self.assertEqual(outcome.rejected, 0)


class TheReasonMappingTests(unittest.TestCase):
    """⭑ WHAT MAKES THE FILTER USEFUL BEYOND ACCURACY: the recovery orchestrator routes on these."""

    def test_a_table_conflict_routes_differently_from_a_collision(self) -> None:
        table = rejection_reasons({"rejected_table": 3})
        collided = rejection_reasons({"rejected_collision": 3})
        self.assertIn(GraspFailureReason.ALL_TABLE_CONFLICT, table)
        self.assertIn(GraspFailureReason.ALL_COLLIDED, collided)
        self.assertNotIn(GraspFailureReason.ALL_COLLIDED, table)

    def test_a_workspace_exit_is_its_own_reason(self) -> None:
        self.assertIn(GraspFailureReason.ALL_OUT_OF_WORKSPACE,
                      rejection_reasons({"rejected_workspace": 1}))

    def test_rescan_is_always_recommended_after_a_geometric_wipeout(self) -> None:
        for counters in ({"rejected_table": 1}, {"rejected_collision": 1}, {}):
            with self.subTest(counters=counters):
                self.assertIn(GraspFailureReason.RESCAN_RECOMMENDED, rejection_reasons(counters))

    def test_no_counters_still_gives_a_reason(self) -> None:
        self.assertIn(GraspFailureReason.NO_VALID_GRASP, rejection_reasons({}))

    def test_it_matches_what_the_analytic_path_builds(self) -> None:
        """One mapping, not two: a re-implementation drifts within a release."""
        reasons = rejection_reasons({"rejected_workspace": 1, "rejected_table": 2,
                                     "rejected_collision": 3})
        self.assertEqual(reasons[:3], (GraspFailureReason.ALL_OUT_OF_WORKSPACE,
                                       GraspFailureReason.ALL_TABLE_CONFLICT,
                                       GraspFailureReason.ALL_COLLIDED))


class TheSharedKeysTests(unittest.TestCase):
    def test_both_calculators_use_the_same_histogram_keys(self) -> None:
        """A rejection histogram is only worth having if the two can be put side by side."""
        for key in REJECTION_KEYS:
            self.assertTrue(key.startswith("rejected_"), key)
        self.assertEqual(set(REJECTION_KEYS),
                         {"rejected_workspace", "rejected_table", "rejected_collision"})

    def test_the_adapter_carries_the_frame_through(self) -> None:
        """Without this the adapter itself would be the frame defect it is meant to bridge."""
        for grasp_frame, expected in ((GraspFrame.BASE, Frame.BASE),
                                      (GraspFrame.CAMERA, Frame.CAMERA)):
            with self.subTest(frame=grasp_frame.value):
                self.assertIs(grasp_point_to_pose(_point(10.0, frame=grasp_frame)).frame, expected)

    def test_the_adapter_synthesises_contacts_at_the_jaw_span(self) -> None:
        pose = grasp_point_to_pose(_point(100.0, width=40.0))
        left, right = pose.contacts
        self.assertAlmostEqual(float(np.linalg.norm(np.asarray(right) - np.asarray(left))), 40.0)

    def test_there_is_only_ONE_adapter(self) -> None:
        """`_pick_helpers` delegates rather than keeping a copy -- a second contact synthesis would
        have been this repo's THIRD gripper description."""
        from src.robot.grasping.loop._pick_helpers import _grasp_point_to_grasp_pose
        point = _point(123.0)
        first, second = _grasp_point_to_grasp_pose(point), grasp_point_to_pose(point)
        self.assertTrue(np.array_equal(first.position_mm, second.position_mm))
        self.assertTrue(np.array_equal(first.rotation_matrix, second.rotation_matrix))
        self.assertIs(first.frame, second.frame)


class TheTransformReaderTests(unittest.TestCase):
    """⛔ THE DEFECT THAT MADE THE WHOLE DEEP PATH RETURN NOTHING, on every real pick.

    `_propose` read `getattr(transform, "matrix", transform)` and the pick loop hands a typed
    `Transform`, which has `to_matrix()` and no `matrix`. `np.asarray(Transform)` RAISES, the blanket
    `except Exception` turned that into NO_CANDIDATES_GENERATED, and every test in the suite passed a
    raw `np.eye(4)` -- so nothing in the repo ever handed this calculator the type its own production
    caller hands it.
    """

    def test_it_reads_a_typed_Transform(self) -> None:
        from src.geometry.transform import Transform
        from src.robot.grasping.deep.calculator import _matrix_of

        transform = Transform(translation_mm=np.array([10.0, 20.0, 30.0]),
                              quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                              from_frame=Frame.CAMERA, to_frame=Frame.BASE)
        matrix = _matrix_of(transform)
        self.assertEqual(matrix.shape, (4, 4))
        self.assertEqual(matrix[:3, 3].tolist(), [10.0, 20.0, 30.0])

    def test_a_raw_matrix_still_works(self) -> None:
        from src.robot.grasping.deep.calculator import _matrix_of
        self.assertEqual(_matrix_of(np.eye(4)).shape, (4, 4))

    def test_the_old_expression_would_have_raised(self) -> None:
        """The control: without it this file cannot claim the defect was real."""
        from src.geometry.transform import Transform

        transform = Transform(translation_mm=np.array([1.0, 2.0, 3.0]),
                              quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                              from_frame=Frame.CAMERA, to_frame=Frame.BASE)
        self.assertFalse(hasattr(transform, "matrix"))
        with self.assertRaises(TypeError):
            np.asarray(getattr(transform, "matrix", transform), dtype=np.float64)

    def test_a_wrong_shape_is_refused_by_name(self) -> None:
        from src.robot.grasping.deep.calculator import _matrix_of
        with self.assertRaises(ValueError) as caught:
            _matrix_of(np.eye(3))
        self.assertIn("4x4", str(caught.exception))


if __name__ == "__main__":                                           # pragma: no cover
    unittest.main()
