"""S2: the live-camera PerceptionSource adapter, tested against a fake streamer + fake models.

No pyrealsense2, no torch, no hardware. These pin the adapter's acquire() LOGIC -- intrinsics source,
top-referenced depth over holes, mask-fill-to-box, label canonicalisation, and the honest error paths --
because that logic is all that can be verified off-box. Behaviour on real D435 depth stays bucket (3).
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from src.camera.setup.image_taking.frames import RGBDFrame
from src.geometry import Frame, Pose
from src.robot.perception import RealSenseVisionPerceptionSource


# --------------------------------------------------------------------------- fakes
@dataclass
class _Det:
    box: tuple[float, float, float, float]
    label: str = "thing"
    score: float = 0.9


@dataclass(frozen=True)
class _Seg:
    """A frozen seg with .mask + .label, so dataclasses.replace() works exactly as on SegmentationResult."""

    mask: np.ndarray
    label: str = "thing"


class _FakeStreamer:
    """grab() -> a fixed RGBDFrame; get_intrinsics() -> a fixed K. Counts grabs for the warmup test."""

    def __init__(self, color: np.ndarray, depth: np.ndarray, k: np.ndarray | None) -> None:
        self._frame = RGBDFrame(color=color, depth=depth)
        self._k = k
        self.grabs = 0

    def grab(self) -> RGBDFrame:
        self.grabs += 1
        return self._frame

    def get_intrinsics(self) -> np.ndarray | None:
        return None if self._k is None else self._k.copy()


class _FakeDetector:
    def __init__(self, dets: list[_Det], raise_it: bool = False) -> None:
        self._dets, self._raise = dets, raise_it

    def detect_all(self, bgr: np.ndarray, prompt: str) -> list[_Det]:
        if self._raise:
            raise RuntimeError("model boom")
        return self._dets


class _FakeSegmenter:
    """Return a seg per detection; the mask is the detection box unless overridden by `masks`."""

    def __init__(self, masks: dict[int, np.ndarray] | None = None, fail_on: set[int] | None = None) -> None:
        self._masks, self._fail = masks or {}, fail_on or set()
        self._i = -1

    def segment_detection(self, bgr: np.ndarray, det: _Det) -> _Seg:
        self._i += 1
        if self._i in self._fail:
            raise RuntimeError("segment boom")
        h, w = bgr.shape[:2]
        if self._i in self._masks:
            return _Seg(mask=self._masks[self._i].astype(np.uint8), label=det.label)
        m = np.zeros((h, w), dtype=np.uint8)
        x0, y0, x1, y1 = (int(c) for c in det.box)
        m[y0:y1, x0:x1] = 1
        return _Seg(mask=m, label=det.label)


_K = np.array([[600.0, 0.0, 32.0], [0.0, 600.0, 32.0], [0.0, 0.0, 1.0]])


def _color(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


class IntrinsicsTests(unittest.TestCase):
    def test_factory_intrinsics_used_by_default(self) -> None:
        s = RealSenseVisionPerceptionSource(
            streamer=_FakeStreamer(_color(), np.full((64, 64), 500, np.uint16), _K),
            detector=_FakeDetector([]), segmenter=_FakeSegmenter(), prompt="x", warmup_grabs=0)
        frame = s.acquire()
        self.assertEqual(s.intrinsics_source, "factory")
        np.testing.assert_array_equal(frame.intrinsics, _K)

    def test_override_intrinsics_wins_and_is_recorded(self) -> None:
        override = _K * 2.0
        s = RealSenseVisionPerceptionSource(
            streamer=_FakeStreamer(_color(), np.full((64, 64), 500, np.uint16), _K),
            detector=_FakeDetector([]), segmenter=_FakeSegmenter(), prompt="x",
            intrinsics=override, warmup_grabs=0)
        frame = s.acquire()
        self.assertEqual(s.intrinsics_source, "override")
        np.testing.assert_array_equal(frame.intrinsics, override)

    def test_missing_intrinsics_and_no_override_raises(self) -> None:
        s = RealSenseVisionPerceptionSource(
            streamer=_FakeStreamer(_color(), np.full((64, 64), 500, np.uint16), None),
            detector=_FakeDetector([]), segmenter=_FakeSegmenter(), prompt="x", warmup_grabs=0)
        with self.assertRaises(RuntimeError):
            s.acquire()


class DepthAndMaskTests(unittest.TestCase):
    """⛔ THE DEPTH THIS SOURCE PUBLISHES IS THE DEPTH IT MEASURED, under a mask as everywhere else.

    It used to overwrite every pixel under a detection with one number, the nearest real surface plus
    a penetration, because that is the plane a jaw closes at. The tests here asserted that overwrite.
    Six stages downstream read the same array for the SHAPE of an object and got a flat sheet, so the
    antipodal search found no opposing normals, the dense sampler measured no curvature, the
    support-plane refinement saw an extent of exactly zero, and the multi-camera fusion fused sheets.
    Where a grasp is anchored inside an object is now decided per candidate by `grasping.geometry`.
    """

    def test_relief_under_a_mask_SURVIVES(self) -> None:
        """The whole change in one assertion. A part 100 mm above the bench must still be 100 mm
        above the bench after the source has looked at it."""
        depth = np.full((64, 64), 500, np.uint16)
        depth[20:30, 20:30] = 400
        depth[24:26, 24:26] = 430  # relief WITHIN the object: a step down its visible face
        det = _Det(box=(20, 20, 30, 30))
        s = RealSenseVisionPerceptionSource(
            streamer=_FakeStreamer(_color(), depth, _K),
            detector=_FakeDetector([det]), segmenter=_FakeSegmenter(), prompt="x", warmup_grabs=0)

        dm = s.acquire().depth_map

        self.assertAlmostEqual(float(dm[21, 21]), 400.0)
        self.assertAlmostEqual(float(dm[24, 24]), 430.0, msg="the step was flattened away")
        self.assertAlmostEqual(float(dm[0, 0]), 500.0)

    def test_the_object_has_a_DEPTH_SPREAD_at_all(self) -> None:
        """Stated as the quantity the stages downstream actually consume. Under the overwrite this
        was exactly zero for every object, which is why every gate that measured an extent, a
        curvature or an opposition could only ever see one answer."""
        depth = np.full((64, 64), 500, np.uint16)
        depth[20:30, 20:30] = 400
        depth[24:26, 24:26] = 430
        det = _Det(box=(20, 20, 30, 30))
        s = RealSenseVisionPerceptionSource(
            streamer=_FakeStreamer(_color(), depth, _K),
            detector=_FakeDetector([det]), segmenter=_FakeSegmenter(), prompt="x", warmup_grabs=0)

        frame = s.acquire()
        under_mask = frame.depth_map[np.asarray(frame.segmentations[0].mask).astype(bool)]

        self.assertGreater(float(under_mask.max() - under_mask.min()), 0.0)

    def test_a_hole_stays_a_hole(self) -> None:
        """A pixel the sensor could not measure is 0, and it must not be filled in by anything here.
        Inventing a surface at an invented distance is a surface a grasp is then planned against."""
        depth = np.full((64, 64), 500, np.uint16)
        depth[20:30, 20:30] = 400
        depth[22, 22] = 0
        det = _Det(box=(20, 20, 30, 30))
        s = RealSenseVisionPerceptionSource(
            streamer=_FakeStreamer(_color(), depth, _K),
            detector=_FakeDetector([det]), segmenter=_FakeSegmenter(), prompt="x", warmup_grabs=0)

        dm = s.acquire().depth_map

        self.assertEqual(float(dm[22, 22]), 0.0)
        self.assertAlmostEqual(float(dm[23, 23]), 400.0)

    def test_all_holes_mask_leaves_depth_unchanged(self) -> None:
        # A mask over an all-holes region reaches the calculator as holes, so the failure is visible
        # downstream rather than answered with a fabricated plane.
        depth = np.full((64, 64), 500, np.uint16)
        depth[20:30, 20:30] = 0
        det = _Det(box=(20, 20, 30, 30))
        s = RealSenseVisionPerceptionSource(
            streamer=_FakeStreamer(_color(), depth, _K),
            detector=_FakeDetector([det]), segmenter=_FakeSegmenter(), prompt="x", warmup_grabs=0)
        dm = s.acquire().depth_map
        self.assertEqual(float(dm[25, 25]), 0.0)

    def test_the_two_published_arrays_AGREE(self) -> None:
        """`surface_depth_map` exists because `depth_map` used to be something else. It now carries
        the same measurement, and a harness that adds sensor noise to one is what separates them."""
        depth = np.full((64, 64), 500, np.uint16)
        depth[20:30, 20:30] = 400
        det = _Det(box=(20, 20, 30, 30))
        s = RealSenseVisionPerceptionSource(
            streamer=_FakeStreamer(_color(), depth, _K),
            detector=_FakeDetector([det]), segmenter=_FakeSegmenter(), prompt="x", warmup_grabs=0)

        frame = s.acquire()

        self.assertIsNotNone(frame.surface_depth_map)
        np.testing.assert_array_equal(frame.depth_map, frame.surface_depth_map)
        self.assertIsNot(frame.depth_map, frame.surface_depth_map,
                         "they must be separate arrays, or noise on one lands on both")

    def test_the_default_leaves_a_tiny_mask_ALONE(self) -> None:
        """What ships. Measured 2026-08-20 over 270 reference scenes, replacing a mask with its box
        cost 7.5 percentage points of top-1 -- it fires on any non-rectangular mask, not an incomplete
        one -- so the default is now `NONE` and SAM2's silhouette reaches the calculator untouched."""
        depth = np.full((64, 64), 500, np.uint16)
        det = _Det(box=(10, 10, 40, 40))
        tiny = np.zeros((64, 64), dtype=np.uint8)
        tiny[10:13, 10:13] = 1
        s = RealSenseVisionPerceptionSource(
            streamer=_FakeStreamer(_color(), depth, _K),
            detector=_FakeDetector([det]), segmenter=_FakeSegmenter(masks={0: tiny}),
            prompt="x", warmup_grabs=0)
        self.assertEqual(int(np.asarray(s.acquire().segmentations[0].mask).sum()), 9)

    def test_incomplete_mask_filled_to_detection_box(self) -> None:
        # SAM2 returns a tiny mask (one corner) far smaller than the box -> fill to the box.
        # Asked for BY NAME: removed as a default, kept as a capability.
        from src.robot.perception.mask_completion import MaskCompletion

        depth = np.full((64, 64), 500, np.uint16)
        det = _Det(box=(10, 10, 40, 40))               # 30x30 = 900 px box
        tiny = np.zeros((64, 64), dtype=np.uint8)
        tiny[10:13, 10:13] = 1                          # 9 px << 0.8*900
        s = RealSenseVisionPerceptionSource(
            streamer=_FakeStreamer(_color(), depth, _K),
            detector=_FakeDetector([det]), segmenter=_FakeSegmenter(masks={0: tiny}),
            prompt="x", warmup_grabs=0, mask_completion=MaskCompletion.AXIS_ALIGNED_BOX)
        seg = s.acquire().segmentations[0]
        self.assertEqual(int(np.asarray(seg.mask).sum()), 30 * 30)   # filled to the box

    def test_near_complete_mask_kept(self) -> None:
        depth = np.full((64, 64), 500, np.uint16)
        det = _Det(box=(10, 10, 40, 40))
        full = np.zeros((64, 64), dtype=np.uint8)
        full[10:40, 10:40] = 1                          # already the whole box -> keep SAM2 precision
        s = RealSenseVisionPerceptionSource(
            streamer=_FakeStreamer(_color(), depth, _K),
            detector=_FakeDetector([det]), segmenter=_FakeSegmenter(masks={0: full}),
            prompt="x", warmup_grabs=0)
        seg = s.acquire().segmentations[0]
        self.assertEqual(int(np.asarray(seg.mask).sum()), 30 * 30)


class LabelAndErrorTests(unittest.TestCase):
    def test_canonical_label_mapping(self) -> None:
        det = _Det(box=(10, 10, 20, 20), label="a red sugar box on the table")
        s = RealSenseVisionPerceptionSource(
            streamer=_FakeStreamer(_color(), np.full((64, 64), 500, np.uint16), _K),
            detector=_FakeDetector([det]), segmenter=_FakeSegmenter(),
            object_labels=("sugar box", "mug"), prompt="x", warmup_grabs=0)
        self.assertEqual(s.acquire().segmentations[0].label, "sugar box")

    def test_labels_pass_through_without_object_labels(self) -> None:
        det = _Det(box=(10, 10, 20, 20), label="free form phrase")
        s = RealSenseVisionPerceptionSource(
            streamer=_FakeStreamer(_color(), np.full((64, 64), 500, np.uint16), _K),
            detector=_FakeDetector([det]), segmenter=_FakeSegmenter(), prompt="x", warmup_grabs=0)
        self.assertEqual(s.acquire().segmentations[0].label, "free form phrase")

    def test_detector_error_yields_no_segmentation(self) -> None:
        s = RealSenseVisionPerceptionSource(
            streamer=_FakeStreamer(_color(), np.full((64, 64), 500, np.uint16), _K),
            detector=_FakeDetector([], raise_it=True), segmenter=_FakeSegmenter(),
            prompt="x", warmup_grabs=0)
        self.assertEqual(s.acquire().segmentations, ())

    def test_one_segmentation_failure_keeps_the_rest(self) -> None:
        dets = [_Det(box=(5, 5, 15, 15), label="a"), _Det(box=(20, 20, 30, 30), label="b")]
        s = RealSenseVisionPerceptionSource(
            streamer=_FakeStreamer(_color(), np.full((64, 64), 500, np.uint16), _K),
            detector=_FakeDetector(dets), segmenter=_FakeSegmenter(fail_on={0}),
            prompt="x", warmup_grabs=0)
        segs = s.acquire().segmentations
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].label, "b")


class WarmupTests(unittest.TestCase):
    def test_warmup_grabs_are_discarded(self) -> None:
        st = _FakeStreamer(_color(), np.full((64, 64), 500, np.uint16), _K)
        s = RealSenseVisionPerceptionSource(
            streamer=st, detector=_FakeDetector([]), segmenter=_FakeSegmenter(),
            prompt="x", warmup_grabs=3)
        s.acquire()
        self.assertEqual(st.grabs, 4)   # 3 warmup + 1 real


class AWristSourceStampsTheToolPoseTests(unittest.TestCase):
    """A source handed the arm's reader reads the TCP after its warm-ups, just before the real grab and again after it."""

    def test_the_pose_is_read_after_the_warm_ups_and_immediately_before_the_real_grab(self) -> None:
        events: list[str] = []
        streamer = _FakeStreamer(_color(), np.full((64, 64), 500, np.uint16), _K)
        grab = streamer.grab

        def logged_grab() -> RGBDFrame:
            events.append("grab")
            return grab()

        pose = Pose(position_mm=np.array([400.0, 0.0, 300.0]), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                    frame=Frame.BASE)

        def reader() -> Pose:
            events.append("pose")
            return pose

        streamer.grab = logged_grab  # type: ignore[method-assign]
        s = RealSenseVisionPerceptionSource(
            streamer=streamer, detector=_FakeDetector([]), segmenter=_FakeSegmenter(), prompt="x", warmup_grabs=2)
        s.stamp_tool_pose_with(reader, motion_tolerance=(1.0, 0.5), attempts=1)
        frame = s.acquire()

        self.assertEqual(events, ["grab", "grab", "pose", "grab", "pose"])
        self.assertEqual(frame.tool_pose, pose)

    def test_a_source_nobody_stamps_carries_no_pose(self) -> None:
        """The control: a fixed camera's source, handed no reader, carries no tool pose."""
        s = RealSenseVisionPerceptionSource(
            streamer=_FakeStreamer(_color(), np.full((64, 64), 500, np.uint16), _K),
            detector=_FakeDetector([]), segmenter=_FakeSegmenter(), prompt="x", warmup_grabs=0)
        self.assertIsNone(s.acquire().tool_pose)


# --------------------------------------------------------------------------- the colour a label names (F1)
#: The owner's mat, a grey cube, a part between grey and white (p80 L* 77), and a red part, BGR.
_MAT, _GREY, _PALE, _RED = (34, 40, 42), (156, 160, 162), (186, 190, 191), (40, 40, 200)
_PROMPT = "each separate grey cube"


class _NamingDetector(_FakeDetector):
    """A VLM as the detector: it boxes what it is handed and names a colour when asked, or raises."""

    def __init__(self, dets: list[_Det], *, answer: str = "", raises: bool = False) -> None:
        super().__init__(dets)
        self.answer, self.raises = answer, raises
        self.asked: list[tuple[float, ...]] = []

    def name_colour(self, image_bgr: np.ndarray, box: tuple[float, ...]) -> str:
        self.asked.append(tuple(box))
        if self.raises:
            raise RuntimeError("CUDA error: out of memory")
        return self.answer


def _two_parts(first: tuple[int, int, int], second: tuple[int, int, int]) -> tuple[np.ndarray, list[_Det]]:
    """Two 20 px parts on the mat, both boxed with the prompt's words, as Qwen boxed them on the cell."""
    color = np.empty((64, 64, 3), dtype=np.uint8)
    color[...] = _MAT
    color[8:28, 8:28] = first
    color[36:56, 36:56] = second
    return color, [_Det(box=(8, 8, 28, 28), label=_PROMPT), _Det(box=(36, 36, 56, 56), label=_PROMPT)]


def _colour_source(color: np.ndarray, detector: _FakeDetector, *, labels: tuple[str, ...] = ("grey cube",),
                   **keywords: object) -> RealSenseVisionPerceptionSource:
    return RealSenseVisionPerceptionSource(
        streamer=_FakeStreamer(color, np.full((64, 64), 500, np.uint16), _K), detector=detector,
        segmenter=_FakeSegmenter(), object_labels=labels, prompt=_PROMPT, warmup_grabs=0, **keywords)


class AColourCheckTests(unittest.TestCase):
    """The camera source judges every part mapped onto an object label whose words name a colour, before any fusion
    reads it: a part of another colour keeps its mask under a label no object carries, a neighbour and no target."""

    def test_a_red_part_boxed_as_a_grey_cube_is_no_grey_cube_and_keeps_its_mask(self) -> None:
        color, dets = _two_parts(_GREY, _RED)
        frame = _colour_source(color, _FakeDetector(dets)).acquire()

        grey, red = frame.segmentations
        self.assertEqual("grey cube", grey.label)
        self.assertEqual("red object, not grey cube", red.label)
        expected = np.zeros((64, 64), dtype=np.uint8)
        expected[36:56, 36:56] = 1
        np.testing.assert_array_equal(expected, np.asarray(red.mask), "the refused part's mask changed")

    def test_an_unsure_part_is_settled_by_the_vlm_s_one_colour_word(self) -> None:
        for answer, label in (("white", "white object, not grey cube"), ("Grey.", "grey cube"),
                              ("", "unknown colour object, not grey cube")):
            with self.subTest(answer=answer):
                color, dets = _two_parts(_GREY, _PALE)
                detector = _NamingDetector(dets, answer=answer)
                frame = _colour_source(color, detector).acquire()
                self.assertEqual(["grey cube", label], [seg.label for seg in frame.segmentations])
                self.assertEqual([(36, 36, 56, 56)], detector.asked, "asked of the unsure part alone, with its box")

    def test_a_question_that_raises_refuses_the_part_and_counts_as_the_detector_s_failure(self) -> None:
        color, dets = _two_parts(_GREY, _PALE)
        source = _colour_source(color, _NamingDetector(dets, raises=True))

        frame = source.acquire()

        self.assertEqual("unknown colour object, not grey cube", frame.segmentations[1].label)
        self.assertEqual(1, source.backend.failures, "a task would read an empty look here, not a failed detector")
        self.assertIn("naming a colour", source.backend.last_failure)

    def test_a_phrase_that_names_no_colour_checks_nothing(self) -> None:
        color, _ = _two_parts(_GREY, _RED)
        detector = _NamingDetector([_Det(box=(36, 36, 56, 56), label="cube")], answer="white")
        frame = _colour_source(color, detector, labels=("cube",)).acquire()

        self.assertEqual(["cube"], [seg.label for seg in frame.segmentations])
        self.assertEqual([], detector.asked)

    def test_a_label_mapped_onto_no_object_is_not_judged(self) -> None:
        color, dets = _two_parts(_GREY, _PALE)
        detector = _NamingDetector([_Det(box=(36, 36, 56, 56), label="screwdriver")], answer="white")
        frame = _colour_source(color, detector).acquire()

        self.assertEqual(["screwdriver"], [seg.label for seg in frame.segmentations])
        self.assertEqual([], detector.asked)

    def test_log_judges_and_changes_nothing_and_off_judges_nothing(self) -> None:
        color, dets = _two_parts(_RED, _PALE)
        logged = _NamingDetector(dets, answer="white")
        with self.assertLogs("src.robot.perception.realsense_source", level="INFO") as said:
            frame = _colour_source(color, logged, colour_check="log").acquire()
        self.assertEqual(["grey cube", "grey cube"], [seg.label for seg in frame.segmentations])
        self.assertIn("logged only", said.output[0])
        self.assertEqual(1, len(logged.asked), "log asks the VLM as on does")

        off = _NamingDetector(dets, answer="white")
        frame = _colour_source(color, off, colour_check="off").acquire()
        self.assertEqual(["grey cube", "grey cube"], [seg.label for seg in frame.segmentations])
        self.assertEqual([], off.asked)

    def test_every_verdict_is_logged_with_what_the_detector_called_the_part(self) -> None:
        color, dets = _two_parts(_GREY, _RED)
        with self.assertLogs("src.robot.perception.realsense_source", level="INFO") as said:
            _colour_source(color, _FakeDetector(dets)).acquire()

        self.assertEqual(2, len(said.output))
        self.assertIn("agrees", said.output[0])
        self.assertIn("refuses it as red", said.output[1])
        self.assertIn(f"the detector said {_PROMPT!r}", said.output[1])

    def test_a_switch_that_names_no_mode_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            _colour_source(_color(), _FakeDetector([]), colour_check="sometimes")


if __name__ == "__main__":
    unittest.main()
