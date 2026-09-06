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
    def test_top_referenced_depth_skips_holes(self) -> None:
        # Depth 500 everywhere; inside a 10x10 mask put a nearer surface (400) and a hole (0). The grasp
        # depth must reference the nearest REAL surface (400), not the hole, + penetration.
        depth = np.full((64, 64), 500, np.uint16)
        depth[20:30, 20:30] = 400
        depth[22, 22] = 0  # a hole inside the mask
        det = _Det(box=(20, 20, 30, 30))
        s = RealSenseVisionPerceptionSource(
            streamer=_FakeStreamer(_color(), depth, _K),
            detector=_FakeDetector([det]), segmenter=_FakeSegmenter(), prompt="x",
            grasp_top_penetration_mm=3.0, warmup_grabs=0)
        frame = s.acquire()
        dm = frame.depth_map
        # over the mask -> 400 + 3; outside untouched at 500
        self.assertAlmostEqual(float(dm[25, 25]), 403.0)
        self.assertAlmostEqual(float(dm[0, 0]), 500.0)

    def test_all_holes_mask_leaves_depth_unchanged(self) -> None:
        # A mask over an all-holes region must NOT fabricate a grasp plane -- leave the raw (0) depth so
        # the failure is visible downstream.
        depth = np.full((64, 64), 500, np.uint16)
        depth[20:30, 20:30] = 0
        det = _Det(box=(20, 20, 30, 30))
        s = RealSenseVisionPerceptionSource(
            streamer=_FakeStreamer(_color(), depth, _K),
            detector=_FakeDetector([det]), segmenter=_FakeSegmenter(), prompt="x", warmup_grabs=0)
        dm = s.acquire().depth_map
        self.assertEqual(float(dm[25, 25]), 0.0)

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


if __name__ == "__main__":
    unittest.main()
