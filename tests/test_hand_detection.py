"""Hand detection: the geometry, the value objects, the gesture mapping, and the 3-D projection.

Everything here runs WITHOUT mediapipe and without a `.task` bundle. That is not a compromise, it is
where the interesting logic lives: the palm centre is arithmetic, the gesture mapping is a lookup,
and the 3-D projection is two matrix operations. MediaPipe's own contribution -- turning pixels into
21 landmarks -- is the one part these tests cannot and should not second-guess.

The MediaPipe-facing wiring is pinned separately, in `test_hand_detection_wiring.py`.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.models.handdetection.gestures import normalise_gesture_label, to_hand_gesture
from src.models.handdetection.hand_finder import HandFinder
from src.models.handdetection.landmarks import (
    LANDMARK_COUNT,
    PALM_LANDMARKS,
    HandLandmark,
    calculate_palm_center,
)
from src.models.handdetection.model_files import resolve_model_file
from src.models.handdetection.types import (
    GestureReading,
    Handedness,
    HandGesture,
    HandObservation,
    HandPosition3D,
    PalmDetection,
    as_landmark_tuple,
)


def _landmarks(**overrides: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """21 landmarks at the origin, with named ones moved. Keeps each test's intent visible."""
    points = [(0, 0)] * LANDMARK_COUNT
    for name, value in overrides.items():
        points[int(HandLandmark[name.upper()])] = value
    return as_landmark_tuple(points)


# ── The palm centre ─────────────────────────────────────────────────────────────────────────────


class ThePalmCentreIsTheMeanOfSixLandmarksTests(unittest.TestCase):
    def test_it_averages_exactly_the_palm_landmarks(self) -> None:
        """Every palm landmark at (100, 200) and every other at the origin -> the centre is (100, 200).

        This is the assertion that would catch a changed index set: any finger tip creeping into
        `PALM_LANDMARKS` drags the mean toward the origin.
        """
        points = [(0, 0)] * LANDMARK_COUNT
        for index in PALM_LANDMARKS:
            points[int(index)] = (100, 200)
        self.assertEqual(calculate_palm_center(as_landmark_tuple(points)), (100.0, 200.0))

    def test_fingertips_do_not_move_it(self) -> None:
        """A closing fist must not move the palm centre -- that is the whole point of using MCPs."""
        base = _landmarks(
            wrist=(10, 10), index_mcp=(20, 10), middle_mcp=(30, 10),
            ring_mcp=(40, 10), pinky_mcp=(50, 10), thumb_cmc=(10, 20),
        )
        splayed = list(base)
        for tip in (
            HandLandmark.THUMB_TIP, HandLandmark.INDEX_TIP, HandLandmark.MIDDLE_TIP,
            HandLandmark.RING_TIP, HandLandmark.PINKY_TIP,
        ):
            splayed[int(tip)] = (999, 999)
        self.assertEqual(
            calculate_palm_center(base), calculate_palm_center(as_landmark_tuple(splayed))
        )

    def test_it_is_sub_pixel(self) -> None:
        """Integer landmarks, fractional centre. Rounding here would quantise every 3-D position."""
        points = [(0, 0)] * LANDMARK_COUNT
        for index in PALM_LANDMARKS:
            points[int(index)] = (0, 0)
        points[int(HandLandmark.WRIST)] = (1, 0)
        centre = calculate_palm_center(as_landmark_tuple(points))
        self.assertAlmostEqual(centre[0], 1.0 / len(PALM_LANDMARKS))

    def test_a_short_landmark_list_is_refused(self) -> None:
        """The alternative is an IndexError from inside the averaging, one layer too late."""
        with self.assertRaises(ValueError) as caught:
            calculate_palm_center([(0, 0)] * 5)
        self.assertIn("21", str(caught.exception))


# ── Gesture mapping ─────────────────────────────────────────────────────────────────────────────


class TheGestureLabelMappingSurvivesSpellingTests(unittest.TestCase):
    """The exact spelling MediaPipe's bundled model emits is NOT verified on this machine.

    The `.task` files are an operator download and are absent here, so the mapping is deliberately
    normalised rather than matched literally -- these tests pin that the normalisation covers the
    plausible spellings, which is what makes the unverified spelling safe.
    """

    def test_the_documented_spelling_maps(self) -> None:
        self.assertIs(to_hand_gesture("Thumb_Up"), HandGesture.THUMB_UP)
        self.assertIs(to_hand_gesture("Thumb_Down"), HandGesture.THUMB_DOWN)

    def test_other_plausible_spellings_map_the_same_way(self) -> None:
        for label in ("thumb up", "THUMBUP", "Thumbs_Up", "thumb-up", " Thumb_Up "):
            with self.subTest(label=label):
                self.assertIs(to_hand_gesture(label), HandGesture.THUMB_UP)

    def test_a_recognised_shape_we_do_not_act_on_is_OTHER_not_NONE(self) -> None:
        """The distinction an operator needs: "not a thumbs-up" vs "no hand gesture at all"."""
        self.assertIs(to_hand_gesture("Open_Palm"), HandGesture.OTHER)
        self.assertIs(to_hand_gesture("Victory"), HandGesture.OTHER)

    def test_mediapipes_own_none_category_is_NONE(self) -> None:
        for label in ("None", "none", "", "   "):
            with self.subTest(label=label):
                self.assertIs(to_hand_gesture(label), HandGesture.NONE)

    def test_normalisation_strips_separators_and_case(self) -> None:
        self.assertEqual(normalise_gesture_label("Thumb_Up"), "thumbup")
        self.assertEqual(normalise_gesture_label("I Love-You"), "iloveyou")


# ── Value objects ───────────────────────────────────────────────────────────────────────────────


class TheValueObjectsRefuseNonsenseTests(unittest.TestCase):
    def test_a_palm_detection_needs_21_landmarks(self) -> None:
        with self.assertRaises(ValueError):
            PalmDetection(palm_center_xy=(1.0, 1.0), landmarks=as_landmark_tuple([(0, 0)] * 20))

    def test_a_non_finite_palm_centre_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            PalmDetection(palm_center_xy=(float("nan"), 1.0), landmarks=_landmarks())

    def test_a_position_behind_the_camera_is_refused(self) -> None:
        """Negative depth is a back-projection bug, and it would reach motion code as a pose."""
        with self.assertRaises(ValueError) as caught:
            HandPosition3D(
                position_base=np.zeros(3), position_cam=np.zeros(3),
                palm_center_xy=(0.0, 0.0), depth_mm=-5.0,
            )
        self.assertIn("behind the camera", str(caught.exception))

    def test_the_positions_are_read_only(self) -> None:
        """These are handed to motion code; an in-place edit would change a pose someone else holds."""
        position = HandPosition3D(
            position_base=np.array([1.0, 2.0, 3.0]), position_cam=np.array([1.0, 2.0, 3.0]),
            palm_center_xy=(0.0, 0.0), depth_mm=3.0,
        )
        with self.assertRaises(ValueError):
            position.position_base[0] = 99.0

    def test_a_gesture_confidence_outside_zero_one_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            GestureReading(gesture=HandGesture.THUMB_UP, confidence=1.4)


# ── The 3-D projection ──────────────────────────────────────────────────────────────────────────


class _FakeProvider:
    """Just enough `FrameProvider` for the geometry: which rigs exist and what kind each is."""

    def __init__(self, kinds: dict[str, str]) -> None:
        self._kinds = kinds

    @property
    def rig_ids(self) -> list[str]:
        return list(self._kinds)

    def is_rgbd(self, rig_id: str) -> bool:
        return self._kinds[rig_id] == "rgbd"

    def is_stereo(self, rig_id: str) -> bool:
        return self._kinds[rig_id] == "stereo"

    def get_stereo_rig_index(self, rig_id: str) -> int:
        return 0

    def grab(self, rig_id: str) -> Any:  # pragma: no cover - only find_hand() uses it
        raise NotImplementedError


def _observation(centre: tuple[float, float]) -> HandObservation:
    points = [(0, 0)] * LANDMARK_COUNT
    for index in PALM_LANDMARKS:
        points[int(index)] = (int(centre[0]), int(centre[1]))
    landmarks = as_landmark_tuple(points)
    return HandObservation(
        palm=PalmDetection(
            palm_center_xy=centre, landmarks=landmarks, handedness=Handedness.RIGHT
        ),
        gesture=GestureReading(gesture=HandGesture.THUMB_UP, confidence=0.9, raw_label="Thumb_Up"),
    )


def _rgbd_frame(depth_mm: int, size: int = 200) -> Any:
    from src.camera.setup.image_taking.frames import RGBDFrame

    return RGBDFrame(
        color=np.zeros((size, size, 3), dtype=np.uint8),
        depth=np.full((size, size), depth_mm, dtype=np.uint16),
    )


class TheRgbdPathBackProjectsThroughRealIntrinsicsTests(unittest.TestCase):
    """The RealSense path. Camera K is required -- inventing one was the previous defect."""

    def _finder(self, **kwargs: Any) -> HandFinder:
        defaults: dict[str, Any] = {
            "provider": _FakeProvider({"d435": "rgbd"}),
            "transforms": {"d435": np.eye(4)},
            "camera_matrices": {
                "d435": np.array([[600.0, 0.0, 100.0], [0.0, 600.0, 100.0], [0.0, 0.0, 1.0]])
            },
            "min_depth_samples": 1,
        }
        defaults.update(kwargs)
        return HandFinder(observer=object(), **defaults)  # type: ignore[arg-type]

    def test_a_palm_at_the_principal_point_lies_on_the_optical_axis(self) -> None:
        position = self._finder().locate(_observation((100.0, 100.0)), _rgbd_frame(500), "d435")
        assert position is not None
        np.testing.assert_allclose(position.position_cam, [0.0, 0.0, 500.0])
        self.assertEqual(position.depth_mm, 500.0)

    def test_an_offset_palm_back_projects_by_the_focal_length(self) -> None:
        """60 px off the principal point at 600 px focal length and 500 mm -> 50 mm across."""
        position = self._finder().locate(_observation((160.0, 100.0)), _rgbd_frame(500), "d435")
        assert position is not None
        np.testing.assert_allclose(position.position_cam, [50.0, 0.0, 500.0])

    def test_the_transform_is_applied_to_reach_the_base_frame(self) -> None:
        transform = np.eye(4)
        transform[:3, 3] = [1000.0, 0.0, 250.0]
        finder = self._finder(transforms={"d435": transform})
        position = finder.locate(_observation((100.0, 100.0)), _rgbd_frame(500), "d435")
        assert position is not None
        np.testing.assert_allclose(position.position_base, [1000.0, 0.0, 750.0])

    def test_missing_intrinsics_refuse_instead_of_inventing_a_camera(self) -> None:
        """THE fixed defect: `fx = fy = width / 2` produced a confident, unmeasured position."""
        finder = self._finder(camera_matrices={})
        with self.assertRaises(ValueError) as caught:
            finder.locate(_observation((100.0, 100.0)), _rgbd_frame(500), "d435")
        self.assertIn("intrinsics", str(caught.exception))

    def test_a_rig_without_a_transform_is_skipped_not_guessed(self) -> None:
        """THE other fixed defect: one global transform silently mislabelled every other rig."""
        finder = self._finder(transforms={})
        self.assertIsNone(finder.locate(_observation((100.0, 100.0)), _rgbd_frame(500), "d435"))

    def test_too_few_valid_depth_pixels_is_not_a_measurement(self) -> None:
        frame = _rgbd_frame(0)
        # Three valid pixels in an otherwise empty depth map, well under the floor.
        frame.depth.setflags(write=True)
        frame.depth[100, 100:103] = 500
        finder = self._finder(min_depth_samples=20)
        self.assertIsNone(finder.locate(_observation((100.0, 100.0)), frame, "d435"))

    def test_the_median_ignores_dropouts(self) -> None:
        """Depth maps drop out on skin; the median over the disc is why that is survivable."""
        frame = _rgbd_frame(0)
        frame.depth.setflags(write=True)
        frame.depth[90:111, 90:111] = 400
        frame.depth[95:100, 95:100] = 0          # a dropout inside the patch
        position = self._finder().locate(_observation((100.0, 100.0)), frame, "d435")
        assert position is not None
        self.assertEqual(position.depth_mm, 400.0)


class TheStereoPathUsesTheStereoEngineTests(unittest.TestCase):
    class _FakeStereo:
        def __init__(self, point: Any) -> None:
            self.point = point
            self.rectified = 0

        def rectify(self, left: Any, right: Any, rig: int = 0) -> tuple[Any, Any]:
            self.rectified += 1
            return left, right

        def compute_3D_point(self, *args: Any, **kwargs: Any) -> Any:
            return self.point

    def _stereo_frame(self, size: int = 200) -> Any:
        from src.camera.setup.image_taking.frames import StereoFrame

        image = np.zeros((size, size, 3), dtype=np.uint8)
        return StereoFrame(left=image, right=image.copy())

    def _finder(self, stereo: Any) -> HandFinder:
        return HandFinder(
            observer=object(),  # type: ignore[arg-type]
            provider=_FakeProvider({"pair": "stereo"}),
            transforms={"pair": np.eye(4)},
            stereo=stereo,
        )

    def test_the_triangulated_point_becomes_the_hand_position(self) -> None:
        stereo = self._FakeStereo(np.array([12.0, -8.0, 640.0]))
        position = self._finder(stereo).locate(
            _observation((100.0, 100.0)), self._stereo_frame(), "pair"
        )
        assert position is not None
        np.testing.assert_allclose(position.position_cam, [12.0, -8.0, 640.0])
        self.assertEqual(position.depth_mm, 640.0)
        self.assertEqual(stereo.rectified, 1, "the pair must be rectified before triangulation")

    def test_no_triangulation_means_no_position(self) -> None:
        position = self._finder(self._FakeStereo(None)).locate(
            _observation((100.0, 100.0)), self._stereo_frame(), "pair"
        )
        self.assertIsNone(position)

    def test_a_point_behind_the_camera_is_discarded_not_returned(self) -> None:
        position = self._finder(self._FakeStereo(np.array([0.0, 0.0, -300.0]))).locate(
            _observation((100.0, 100.0)), self._stereo_frame(), "pair"
        )
        self.assertIsNone(position)

    def test_a_stereo_rig_without_a_stereo_engine_refuses(self) -> None:
        finder = HandFinder(
            observer=object(),  # type: ignore[arg-type]
            provider=_FakeProvider({"pair": "stereo"}),
            transforms={"pair": np.eye(4)},
            stereo=None,
        )
        with self.assertRaises(ValueError) as caught:
            finder.locate(_observation((100.0, 100.0)), self._stereo_frame(), "pair")
        self.assertIn("StereoCam3D", str(caught.exception))


# ── Model files ─────────────────────────────────────────────────────────────────────────────────


class TheMissingModelSaysWhereToGetItTests(unittest.TestCase):
    def test_the_error_names_the_config_key_the_path_and_the_download(self) -> None:
        with self.assertRaises(FileNotFoundError) as caught:
            resolve_model_file(
                "no/such/model.task",
                config_key="models.handdetect.model_path",
                download_url="https://example.invalid/model.task",
            )
        message = str(caught.exception)
        for expected in (
            "models.handdetect.model_path", "no/such/model.task", "https://example.invalid/model.task"
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, message)

    def test_an_empty_path_is_its_own_message(self) -> None:
        with self.assertRaises(FileNotFoundError) as caught:
            resolve_model_file("", config_key="models.handdetect.model_path", download_url="u")
        self.assertIn("is empty", str(caught.exception))

    def test_an_existing_file_resolves_to_an_absolute_path(self) -> None:
        resolved = resolve_model_file(
            __file__, config_key="models.handdetect.model_path", download_url="u"
        )
        self.assertTrue(resolved.endswith("test_hand_detection.py"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
