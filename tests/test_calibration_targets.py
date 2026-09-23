"""The hand-eye calibration target: one ArUco marker or a ChArUco board, named in the config and seen per frame.

Issue 4 of the calibration chain (owner, 2026-09-23): the target could only be one ArUco marker, its id had no YAML
home, and a bad dictionary, a length of -5 mm or an id the dictionary does not hold passed ``--check``. A ChArUco
board, which the stereo calibration already prints, could not be named at all, and a sweep pointed at one posed its
marker 0 at the configured edge: VERIFIED 2x to 2.5x too far away, and counted.

Held here, with no camera and no robot:

* the schema's target union (``cam_schema.py``): what it refuses, that it is additive to the old keys, and that a
  marker is stated once;
* ``parse_board_spec``, the ``--board`` spelling;
* ``CharucoPoseEstimator.observe`` on a synthetic render with a known camera matrix and board pose, as the analysis
  measured it (24 corners, 0.30 px, under 1 mm at 520 mm), and why it says no;
* ``ArucoPoseEstimator.observe``: the same pose ``estimate()`` gives, and why it says no.
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

cv = pytest.importorskip("cv2")

from src.calibration.exceptions import StereoCalibrationError  # noqa: E402
from src.calibration.stereo.sub_modules.aruco_esti import ArucoPoseEstimator, resolve_aruco_dictionary  # noqa: E402
from src.calibration.targets import (  # noqa: E402
    CharucoPoseEstimator,
    aruco_dictionary_size,
    canonical_aruco_dict_name,
    describe_target,
    estimator_for,
    parse_board_spec,
)
from src.config.schema._base import _ARUCO_DICT_NAMES  # noqa: E402
from src.config.schema.camera import EyeHandRoutineConfig, HandEyeConfig  # noqa: E402
from src.config.schema.camera.cam_schema import (  # noqa: E402
    ArucoTargetConfig,
    CalibrationTargetConfig,
    CharucoTargetConfig,
    ids_in_aruco_dictionary,
)

_K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 480.0], [0.0, 0.0, 1.0]])
_DIST = np.zeros(5)
_TARGET = TypeAdapter(CalibrationTargetConfig)


def _board_view(squares=(7, 5), square=30.0, marker=22.0, dict_name="DICT_5X5_100", legacy=False,
                rvec=(0.35, -0.25, 0.1), tvec=(-90.0, -60.0, 520.0), px_per_mm=4.0) -> np.ndarray:
    """A ChArUco board as a pinhole camera with ``_K`` sees it at a known pose (board frame, millimetres).

    The printed board image maps to the board plane by a scale; the plane maps to the image by ``K [r1 r2 t]``.
    """
    dictionary = cv.aruco.getPredefinedDictionary(getattr(cv.aruco, dict_name))
    board = cv.aruco.CharucoBoard(squares, square, marker, dictionary)
    if legacy:
        board.setLegacyPattern(True)
    printed = board.generateImage((int(squares[0] * square * px_per_mm), int(squares[1] * square * px_per_mm)),
                                  marginSize=0)
    rotation, _ = cv.Rodrigues(np.asarray(rvec, dtype=np.float64))
    scale = np.diag([1.0 / px_per_mm, 1.0 / px_per_mm, 1.0])
    homography = _K @ np.column_stack([rotation[:, 0], rotation[:, 1], np.asarray(tvec, dtype=np.float64)]) @ scale
    return cv.warpPerspective(cv.cvtColor(printed, cv.COLOR_GRAY2BGR), homography, (1280, 960),
                              borderValue=(255, 255, 255))


def _marker_view(dict_name="DICT_5X5_100", marker_id=0, marker_px=240) -> np.ndarray:
    dictionary = cv.aruco.getPredefinedDictionary(getattr(cv.aruco, dict_name))
    canvas = np.full((960, 1280), 255, dtype=np.uint8)
    y0, x0 = (960 - marker_px) // 2, (1280 - marker_px) // 2
    canvas[y0:y0 + marker_px, x0:x0 + marker_px] = cv.aruco.generateImageMarker(dictionary, marker_id, marker_px)
    return cv.cvtColor(canvas, cv.COLOR_GRAY2BGR)


def _charuco(**overrides) -> CharucoPoseEstimator:
    values = dict(squares_x=7, squares_y=5, square_length_mm=30.0, marker_length_mm=22.0)
    values.update(overrides)
    return CharucoPoseEstimator(**values)


# ---------------------------------------------------------------------------------------------------------------------
# The schema
# ---------------------------------------------------------------------------------------------------------------------


class DictionaryNamesTests(unittest.TestCase):

    def test_the_schema_knows_how_many_ids_every_dictionary_holds_as_opencv_does(self) -> None:
        """The schema checks an id without importing OpenCV, so its table is held against OpenCV here."""
        for name in sorted(_ARUCO_DICT_NAMES):
            with self.subTest(name):
                self.assertEqual(ids_in_aruco_dictionary(name), aruco_dictionary_size(name))

    def test_a_name_is_read_the_way_the_estimator_reads_it(self) -> None:
        self.assertEqual(canonical_aruco_dict_name("5x5_100"), "DICT_5X5_100")
        self.assertEqual(canonical_aruco_dict_name(" dict_4x4_50 "), "DICT_4X4_50")
        # cv2.aruco spells the AprilTag names twice; the schema accepts the mixed-case one.
        self.assertEqual(canonical_aruco_dict_name("apriltag_36h11"), "DICT_APRILTAG_36h11")
        with self.assertRaisesRegex(ValueError, "BOGUS"):
            canonical_aruco_dict_name("BOGUS")
        with self.assertRaises(StereoCalibrationError):
            resolve_aruco_dictionary("BOGUS")


class TheTargetSchemaTests(unittest.TestCase):

    def _refusal(self, data: dict) -> str:
        with self.assertRaises(ValidationError) as caught:
            _TARGET.validate_python(data)
        return str(caught.exception)

    def test_a_marker_the_dictionary_does_not_hold_is_refused(self) -> None:
        refusal = self._refusal({"kind": "aruco", "marker_id": 100, "marker_length_mm": 40.0,
                                 "aruco_dict_name": "DICT_5X5_100"})
        self.assertIn("marker_id 100 is not in DICT_5X5_100, whose ids run from 0 to 99", refusal)
        self.assertEqual(_TARGET.validate_python({"kind": "aruco", "marker_id": 99, "marker_length_mm": 40.0}).marker_id,
                         99)

    def test_a_marker_needs_a_length_above_zero_and_a_known_dictionary(self) -> None:
        self.assertIn("greater than 0", self._refusal({"kind": "aruco", "marker_length_mm": 0.0}))
        self.assertIn("marker_length_mm", self._refusal({"kind": "aruco"}))
        self.assertIn("unknown ArUco dictionary", self._refusal({"kind": "aruco", "marker_length_mm": 40.0,
                                                                  "aruco_dict_name": "BOGUS_DICT"}))

    def test_a_board_whose_marker_does_not_fit_its_square_is_refused(self) -> None:
        refusal = self._refusal({"kind": "charuco", "squares_x": 7, "squares_y": 5, "square_length_mm": 22.0,
                                 "marker_length_mm": 30.0})
        self.assertIn("must be smaller than square_length_mm", refusal)

    def test_a_board_with_more_markers_than_its_dictionary_holds_is_refused(self) -> None:
        refusal = self._refusal({"kind": "charuco", "squares_x": 12, "squares_y": 10, "square_length_mm": 30.0,
                                 "marker_length_mm": 22.0, "aruco_dict_name": "DICT_4X4_50"})
        self.assertIn("carries 60 markers and DICT_4X4_50 holds 50", refusal)

    def test_a_board_asking_for_more_corners_than_it_has_is_refused(self) -> None:
        refusal = self._refusal({"kind": "charuco", "squares_x": 3, "squares_y": 3, "square_length_mm": 30.0,
                                 "marker_length_mm": 22.0, "min_corners": 5})
        self.assertIn("more than the 4 inner corners", refusal)

    def test_an_unknown_kind_is_refused(self) -> None:
        self.assertIn("'aruco', 'charuco'", self._refusal({"kind": "chessboard", "squares_x": 7}))


class TheBlockKeepsItsOldKeysTests(unittest.TestCase):

    def test_the_old_keys_alone_are_one_marker_with_id_zero(self) -> None:
        block = EyeHandRoutineConfig.model_validate({"marker_length_mm": 48.0, "aruco_dict_name": "DICT_4X4_50"})
        self.assertIsNone(block.target)
        self.assertEqual(block.resolved_target(), ArucoTargetConfig(marker_length_mm=48.0,
                                                                    aruco_dict_name="DICT_4X4_50", marker_id=0))

    def test_the_marker_id_has_a_yaml_home_and_is_checked_against_the_dictionary(self) -> None:
        self.assertEqual(EyeHandRoutineConfig.model_validate({"marker_id": 7}).resolved_target().marker_id, 7)
        with self.assertRaisesRegex(ValidationError, "marker_id 50 is not in DICT_4X4_50"):
            EyeHandRoutineConfig.model_validate({"marker_id": 50, "aruco_dict_name": "DICT_4X4_50"})

    def test_a_board_target_takes_the_blocks_dictionary_when_it_names_none(self) -> None:
        block = EyeHandRoutineConfig.model_validate({
            "aruco_dict_name": "DICT_4X4_50",
            "target": {"kind": "charuco", "squares_x": 7, "squares_y": 5, "square_length_mm": 30.0,
                       "marker_length_mm": 22.0}})
        self.assertIsInstance(block.target, CharucoTargetConfig)
        self.assertEqual(block.resolved_target().aruco_dict_name, "DICT_4X4_50")

    def test_a_marker_stated_twice_must_agree(self) -> None:
        with self.assertRaisesRegex(ValidationError, "marker_id 2 and target.marker_id 3"):
            EyeHandRoutineConfig.model_validate({
                "marker_id": 2, "target": {"kind": "aruco", "marker_id": 3, "marker_length_mm": 40.0}})
        with self.assertRaisesRegex(ValidationError, "marker_length_mm 50.0 and target.marker_length_mm 40.0"):
            EyeHandRoutineConfig.model_validate({
                "marker_length_mm": 50.0, "target": {"kind": "aruco", "marker_length_mm": 40.0}})
        agreed = EyeHandRoutineConfig.model_validate({
            "marker_length_mm": 40.0, "target": {"kind": "aruco", "marker_length_mm": 40.0}})
        self.assertEqual(agreed.resolved_target().marker_length_mm, 40.0)

    def test_beside_a_board_only_the_dictionary_must_agree(self) -> None:
        """The single-marker keys are not read beside a board, so the shipped block's 50 mm does not refuse one."""
        board = {"kind": "charuco", "squares_x": 7, "squares_y": 5, "square_length_mm": 30.0,
                 "marker_length_mm": 22.0}
        block = EyeHandRoutineConfig.model_validate({"marker_length_mm": 50.0, "marker_id": 0, "target": board})
        self.assertEqual(block.resolved_target().marker_length_mm, 22.0)
        with self.assertRaisesRegex(ValidationError, "aruco_dict_name 'DICT_5X5_100' and target.aruco_dict_name"):
            EyeHandRoutineConfig.model_validate({"aruco_dict_name": "DICT_5X5_100",
                                                 "target": {**board, "aruco_dict_name": "DICT_4X4_50"}})

    def test_the_shipped_default_block_dumps_the_new_keys_as_their_defaults(self) -> None:
        dumped = HandEyeConfig().eye_to_hand.model_dump()
        self.assertEqual((dumped["marker_id"], dumped["target"]), (0, None))


# ---------------------------------------------------------------------------------------------------------------------
# --board
# ---------------------------------------------------------------------------------------------------------------------


class BoardSpecTests(unittest.TestCase):

    def test_the_spellings_round_trip_into_the_schema(self) -> None:
        cases = {
            "aruco:0:40": {"kind": "aruco", "marker_id": 0, "marker_length_mm": 40.0},
            "aruco:3:40:DICT_4X4_50": {"kind": "aruco", "marker_id": 3, "marker_length_mm": 40.0,
                                       "aruco_dict_name": "DICT_4X4_50"},
            "charuco:7x5:30:22:DICT_5X5_100": {"kind": "charuco", "squares_x": 7, "squares_y": 5,
                                               "square_length_mm": 30.0, "marker_length_mm": 22.0,
                                               "aruco_dict_name": "DICT_5X5_100"},
            "charuco:7x5:30:22:DICT_5X5_100:legacy": {"kind": "charuco", "squares_x": 7, "squares_y": 5,
                                                      "square_length_mm": 30.0, "marker_length_mm": 22.0,
                                                      "aruco_dict_name": "DICT_5X5_100", "legacy_pattern": True},
            "charuco:7x5:30:22:legacy": {"kind": "charuco", "squares_x": 7, "squares_y": 5, "square_length_mm": 30.0,
                                         "marker_length_mm": 22.0, "legacy_pattern": True},
        }
        for spec, expected in cases.items():
            with self.subTest(spec):
                self.assertEqual(parse_board_spec(spec), expected)
                self.assertEqual(_TARGET.validate_python(parse_board_spec(spec)).kind, expected["kind"])

    def test_a_spec_that_does_not_parse_is_one_sentence_naming_the_form(self) -> None:
        for spec, said in (("charuco:7:30:22:X", "squares are written XxY"),
                           ("chess:7x5:30", "the kind is aruco or charuco"),
                           ("aruco:zero:40", "the marker id is a whole number"),
                           ("aruco:0", "a marker is written aruco:ID:SIZE_MM[:DICT]"),
                           ("charuco:7x5:30:22:DICT_5X5_100:legacy:extra", "a board is written")):
            with self.subTest(spec), self.assertRaises(ValueError) as caught:
                parse_board_spec(spec)
            self.assertIn(said, str(caught.exception))
            self.assertEqual(1, len(str(caught.exception).splitlines()))

    def test_values_are_the_schemas_to_refuse(self) -> None:
        """The marker larger than its square parses; the schema refuses it, with its own sentence."""
        with self.assertRaisesRegex(ValidationError, "must be smaller than square_length_mm"):
            _TARGET.validate_python(parse_board_spec("charuco:7x5:22:30"))

    def test_a_target_describes_itself_in_one_line(self) -> None:
        self.assertEqual(describe_target(ArucoTargetConfig(marker_length_mm=50.0)), "50.0 mm, id 0, DICT_5X5_100")
        board = _TARGET.validate_python(parse_board_spec("charuco:7x5:30:22:DICT_5X5_100:legacy"))
        self.assertEqual(describe_target(board),
                         "charuco 7x5, square 30.0 mm, marker 22.0 mm, DICT_5X5_100, legacy pattern, at least 6 corners")


# ---------------------------------------------------------------------------------------------------------------------
# ChArUco on a synthetic render
# ---------------------------------------------------------------------------------------------------------------------


class CharucoPoseEstimatorTests(unittest.TestCase):

    def test_a_rendered_board_is_posed_where_it_was_put(self) -> None:
        """MEASURED: 24 corners, 0.30 px, within 1 mm of the truth at 520 mm; held here to 2 mm and 1 px."""
        observation = _charuco().observe(_board_view(), _K, _DIST)
        self.assertTrue(observation.found, observation.why_not)
        self.assertGreaterEqual(observation.n_points, 20)
        assert observation.reprojection_px is not None and observation.T_cam_to_target is not None
        self.assertLess(observation.reprojection_px, 1.0)
        np.testing.assert_allclose(observation.T_cam_to_target[:3, 3], [-90.0, -60.0, 520.0], atol=2.0)
        rotation, _ = cv.Rodrigues(np.array([0.35, -0.25, 0.1]))
        np.testing.assert_allclose(observation.T_cam_to_target[:3, :3], rotation, atol=5e-3)
        self.assertEqual(observation.why_not, "")
        # The distance is the board origin's, |t| = |(-90, -60, 520)| = 531 mm.
        assert observation.distance_mm is not None
        self.assertAlmostEqual(observation.distance_mm, 531.1, delta=2.0)
        self.assertEqual(observation.summary(), f"{observation.n_points} corners, "
                                                f"{observation.reprojection_px:.2f} px, "
                                                f"{observation.distance_mm:.0f} mm away")

    def test_the_estimator_the_target_names_is_a_board_estimator(self) -> None:
        board = _TARGET.validate_python(parse_board_spec("charuco:7x5:30:22"))
        estimator = estimator_for(board)
        self.assertIsInstance(estimator, CharucoPoseEstimator)
        self.assertTrue(estimator.observe(_board_view(), _K, _DIST).found)

    def test_too_few_corners_is_no_pose_and_says_how_many(self) -> None:
        view = _board_view()
        view[:, 700:] = 255  # half the board out of the frame
        observation = _charuco(min_corners=20).observe(view, _K, _DIST)
        self.assertFalse(observation.found)
        self.assertIsNone(observation.T_cam_to_target)
        self.assertRegex(observation.why_not, r"chessboard corners interpolated \(need 20\)")
        self.assertGreater(observation.n_points, 0, "the corners it did see are kept for drawing")

    def test_a_board_read_in_the_other_layout_says_legacy_pattern(self) -> None:
        """An even squares_y board generated before OpenCV 4.6 shows its markers and interpolates no corners."""
        for printed_legacy in (True, False):
            with self.subTest(printed_legacy=printed_legacy):
                view = _board_view(squares=(6, 4), legacy=printed_legacy)
                wrong = _charuco(squares_x=6, squares_y=4, legacy_pattern=not printed_legacy).observe(view, _K, _DIST)
                self.assertFalse(wrong.found)
                self.assertIn("legacy_pattern", wrong.why_not)
                right = _charuco(squares_x=6, squares_y=4, legacy_pattern=printed_legacy).observe(view, _K, _DIST)
                self.assertTrue(right.found, right.why_not)

    def test_nothing_in_view_says_so(self) -> None:
        blank = np.full((960, 1280, 3), 127, dtype=np.uint8)
        self.assertEqual(_charuco().observe(blank, _K, _DIST).why_not, "no DICT_5X5_100 marker of the board in view")

    def test_a_camera_matrix_no_pose_can_use_is_an_error_not_a_miss(self) -> None:
        with self.assertRaises(StereoCalibrationError):
            _charuco().observe(_board_view(), None, _DIST)
        with self.assertRaises(StereoCalibrationError):
            _charuco().observe(np.zeros((10, 10, 4), dtype=np.uint8), _K, _DIST)

    def test_the_estimator_refuses_a_board_opencv_cannot_hold(self) -> None:
        with self.assertRaises(StereoCalibrationError):
            _charuco(square_length_mm=20.0, marker_length_mm=22.0)
        with self.assertRaises(StereoCalibrationError):
            _charuco(squares_x=12, squares_y=10, dict_name="DICT_4X4_50")


# ---------------------------------------------------------------------------------------------------------------------
# One marker
# ---------------------------------------------------------------------------------------------------------------------


class ArucoObserveTests(unittest.TestCase):

    def test_observe_poses_the_marker_estimate_poses(self) -> None:
        estimator = ArucoPoseEstimator(marker_length_mm=50.0, dict_name="DICT_5X5_100", target_id=0)
        view = _marker_view()
        observation = estimator.observe(view, _K, _DIST)
        self.assertTrue(observation.found, observation.why_not)
        np.testing.assert_array_equal(observation.T_cam_to_target, estimator.estimate(view, _K, _DIST, target_id=0))
        self.assertEqual((observation.n_points, observation.marker_ids, observation.hint), (4, (0,), ""))
        assert observation.reprojection_px is not None
        self.assertLess(observation.reprojection_px, 0.5)

    def test_another_id_in_view_is_named(self) -> None:
        observation = ArucoPoseEstimator(50.0, "DICT_5X5_100", target_id=0).observe(_marker_view(marker_id=7), _K,
                                                                                   _DIST)
        self.assertFalse(observation.found)
        self.assertEqual(observation.why_not, "marker 0 not in view; DICT_5X5_100 ids in view: [7]")

    def test_a_marker_of_another_dictionary_is_named(self) -> None:
        view = _marker_view(dict_name="DICT_4X4_50", marker_id=3)
        observation = ArucoPoseEstimator(50.0, "DICT_5X5_100", target_id=3).observe(view, _K, _DIST)
        self.assertFalse(observation.found)
        self.assertTrue(observation.why_not.startswith("no DICT_5X5_100 marker in view"), observation.why_not)
        self.assertIn("DICT_4X4_1000 ids [3]", observation.why_not)
        self.assertIn("check aruco_dict_name", observation.why_not)

    def test_a_single_marker_target_pointed_at_a_charuco_board_asks_whether_it_is_one(self) -> None:
        """VERIFIED by the analysis: marker 0 of a board posed at the configured edge counts, 2x to 2.5x too far."""
        observation = ArucoPoseEstimator(50.0, "DICT_5X5_100", target_id=0).observe(_board_view(), _K, _DIST)
        self.assertTrue(observation.found)
        self.assertIn("markers visible: is this a ChArUco board?", observation.hint)
        assert observation.distance_mm is not None
        self.assertGreater(observation.distance_mm, 1.5 * 520.0, "the mis-scale the hint exists for")

    def test_observe_needs_a_marker_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_id"):
            ArucoPoseEstimator(50.0).observe(_marker_view(), _K, _DIST)

    def test_the_estimator_a_marker_target_names_poses_that_marker(self) -> None:
        estimator = estimator_for(ArucoTargetConfig(marker_length_mm=50.0, marker_id=7))
        self.assertIsInstance(estimator, ArucoPoseEstimator)
        self.assertTrue(estimator.observe(_marker_view(marker_id=7), _K, _DIST).found)
        with self.assertRaisesRegex(ValueError, "unknown calibration target kind"):
            estimator_for({"kind": "chessboard"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
