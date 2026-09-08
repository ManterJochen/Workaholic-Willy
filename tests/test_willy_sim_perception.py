"""GroundTruthPerceptionSource — Isaac-free unit tests (run everywhere).

The camera is injected, so the source's logic (mask extraction, m->mm depth, PerceptionFrame
shape) is fully testable with a fake camera. The on-box behaviour is validated separately.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from src.willy_sim import GroundTruthPerceptionSource, object_mask_from_frame
from src.willy_sim.perception import (
    MultiObjectGroundTruthPerceptionSource,
    MultiObjectVisionPerceptionSource,
)
from src.robot.grasping.types.perception import PerceptionFrame, PerceptionSource


def _frame(data: np.ndarray) -> dict:
    return {
        "instance_id_segmentation": {
            "data": data,
            "info": {"idToLabels": {"1": "INVALID", "2": "/World/ground", "3": "/World/Cube"}},
        }
    }


class _FakeCamera:
    def __init__(self, depth_m, intrinsics, frame, rgb=None) -> None:
        self._depth_m = depth_m
        self._K = intrinsics
        self._frame = frame
        self._rgb = rgb
        self.added: list[str] = []

    def add_distance_to_image_plane_to_frame(self) -> None:
        self.added.append("depth")

    def add_instance_id_segmentation_to_frame(self) -> None:
        self.added.append("iid")

    def get_depth(self):
        return self._depth_m

    def get_intrinsics_matrix(self):
        return self._K

    def get_current_frame(self):
        return self._frame

    def get_rgb(self):
        return self._rgb


class ObjectMaskTests(unittest.TestCase):
    def test_extracts_target_mask(self) -> None:
        data = np.zeros((4, 4), dtype=np.uint32)
        data[1:3, 1:3] = 3
        mask = object_mask_from_frame(_frame(data), "/World/Cube")
        self.assertEqual(int(mask.sum()), 4)
        self.assertTrue(bool(mask[1, 1]))

    def test_missing_prim_returns_none(self) -> None:
        self.assertIsNone(object_mask_from_frame(_frame(np.zeros((4, 4), np.uint32)), "/World/X"))

    def test_no_segmentation_returns_none(self) -> None:
        self.assertIsNone(object_mask_from_frame({}, "/World/Cube"))

    def test_matches_child_subtree_for_referenced_usd(self) -> None:
        # P2.D: a referenced YCB USD carries the instance-id on CHILD mesh prims, not the top prim ->
        # matching the top prim must union the subtree's pixels.
        data = np.zeros((4, 4), dtype=np.uint32)
        data[0, 0] = 5  # mesh prim
        data[3, 3] = 6  # collisions prim
        frame = {"instance_id_segmentation": {"data": data, "info": {"idToLabels": {
            "1": "INVALID", "2": "/World/ground",
            "5": "/World/Object_0/_05_tomato_soup_can/mesh",
            "6": "/World/Object_0/_05_tomato_soup_can/collisions",
        }}}}
        mask = object_mask_from_frame(frame, "/World/Object_0")
        self.assertEqual(int(mask.sum()), 2)  # both child-prim pixels unioned
        self.assertTrue(bool(mask[0, 0]) and bool(mask[3, 3]))

    def test_prefix_does_not_false_match_a_sibling(self) -> None:
        # "/World/Object_1" must NOT match the target "/World/Object_0" (the prefix is target + "/").
        data = np.zeros((4, 4), dtype=np.uint32)
        data[0, 0] = 7
        frame = {"instance_id_segmentation": {"data": data, "info": {"idToLabels": {
            "7": "/World/Object_10/mesh",  # sibling-ish; must not match /World/Object_1
        }}}}
        self.assertIsNone(object_mask_from_frame(frame, "/World/Object_1"))


class GroundTruthPerceptionSourceTests(unittest.TestCase):
    def _src(self, data=None, depth=None):
        if data is None:
            data = np.zeros((4, 4), dtype=np.uint32)
            data[1:3, 1:3] = 3
        if depth is None:
            depth = np.full((4, 4), 0.5)  # 0.5 m
        cam = _FakeCamera(depth, np.eye(3), _frame(data))
        src = GroundTruthPerceptionSource(
            camera=cam, target_prim_path="/World/Cube", warmup_steps=0
        )
        return src, cam

    def test_satisfies_perception_source_protocol(self) -> None:
        src, _ = self._src()
        self.assertIsInstance(src, PerceptionSource)

    def test_adds_annotators_on_construction(self) -> None:
        _, cam = self._src()
        self.assertIn("depth", cam.added)
        self.assertIn("iid", cam.added)

    def test_acquire_builds_mm_frame_with_matching_mask(self) -> None:
        src, _ = self._src()
        frame = src.acquire()
        self.assertIsInstance(frame, PerceptionFrame)
        np.testing.assert_allclose(frame.depth_map, 500.0)  # 0.5 m -> 500 mm
        self.assertEqual(len(frame.segmentations), 1)
        mask = np.asarray(frame.segmentations[0].mask)
        self.assertEqual(mask.shape, frame.depth_map.shape)
        self.assertEqual(int(mask.sum()), 4)

    def test_invisible_object_gives_empty_matching_mask(self) -> None:
        src, _ = self._src(data=np.zeros((4, 4), dtype=np.uint32))
        frame = src.acquire()
        mask = np.asarray(frame.segmentations[0].mask)
        self.assertEqual(int(mask.sum()), 0)
        self.assertEqual(mask.shape, frame.depth_map.shape)

    def test_nonfinite_depth_is_zeroed(self) -> None:
        depth = np.full((4, 4), 0.5)
        depth[0, 0] = np.inf
        src, _ = self._src(depth=depth)
        frame = src.acquire()
        self.assertEqual(float(frame.depth_map[0, 0]), 0.0)


class MaybeTopReferenceTests(unittest.TestCase):
    """P2.D tall-object fix: the top-down grasp is referenced to the object TOP, taking the HIGHER
    point (= smaller depth, since depth and world-z are inverse). OFF (None) is byte-identical; a TALL
    object is raised toward its top, a SHORT object keeps centre+lift."""

    def _src(self, penetration):  # noqa: ANN001, ANN202
        cam = _FakeCamera(np.full((4, 4), 0.9), np.eye(3), _frame(np.zeros((4, 4), np.uint32)))
        return MultiObjectGroundTruthPerceptionSource(
            camera=cam, targets=[], warmup_steps=0,
            ground_truth_depth=True, grasp_top_penetration_mm=penetration,
        )

    @staticmethod
    def _mask() -> np.ndarray:
        m = np.zeros((4, 4), dtype=bool)
        m[1:3, 1:3] = True
        return m

    def test_off_is_noop(self) -> None:
        out = self._src(None)._maybe_top_reference(941.0, np.full((4, 4), 908.0), self._mask(), "x")
        self.assertEqual(out, 941.0)  # None => returns centre depth unchanged (byte-identical)

    def test_tall_object_is_raised_to_top(self) -> None:
        rendered = np.full((4, 4), 5000.0)  # far background
        mask = self._mask()
        rendered[mask] = 908.0  # the object's nearest (top) surface
        out = self._src(12.0)._maybe_top_reference(941.0, rendered, mask, "box")
        self.assertAlmostEqual(out, 920.0)  # top(908)+pen(12)=920 < centre(941) -> grasp raised

    def test_short_object_keeps_center(self) -> None:
        rendered = np.full((4, 4), 5000.0)
        mask = self._mask()
        rendered[mask] = 970.0  # a short object's top is DEEPER than its own centre+lift
        out = self._src(12.0)._maybe_top_reference(973.0, rendered, mask, "cube")
        self.assertEqual(out, 973.0)  # centre(973) < top+pen(982) -> unchanged

    def test_zero_background_pixels_in_mask_are_ignored(self) -> None:
        rendered = np.full((4, 4), 5000.0)
        mask = self._mask()
        rendered[mask] = 0.0  # zeroed background leaked into the mask edge
        ys, xs = np.where(mask)
        rendered[ys[0], xs[0]] = 908.0  # one real top pixel survives
        out = self._src(12.0)._maybe_top_reference(941.0, rendered, mask, "box")
        self.assertAlmostEqual(out, 920.0)  # min over POSITIVE vals = 908 -> +12

    def test_empty_mask_keeps_center(self) -> None:
        out = self._src(12.0)._maybe_top_reference(
            941.0, np.full((4, 4), 908.0), np.zeros((4, 4), dtype=bool), "x"
        )
        self.assertEqual(out, 941.0)  # no masked pixels -> centre depth unchanged


class _FakeDet:
    def __init__(self, label: str, score: float = 0.9) -> None:
        self.label = label
        self.score = score
        self.box = [0.0, 0.0, 2.0, 2.0]


class _FakeDetector:
    """detect_all returns one detection per configured label. Items may be ``str`` or ``(label, score)``."""

    def __init__(self, labels: list) -> None:
        self._labels = labels
        self.last_prompt: str | None = None

    def detect_all(self, _bgr, prompt: str):  # noqa: ANN001, ANN202
        self.last_prompt = prompt
        out = []
        for item in self._labels:
            out.append(_FakeDet(item[0], item[1]) if isinstance(item, tuple) else _FakeDet(item))
        return out


@dataclass(frozen=True, slots=True)
class _FakeSeg:  # mirrors the real (now frozen) SegmentationResult so dataclasses.replace works
    label: str
    mask: np.ndarray


class _FakeSegmenter:
    def __init__(self, mask_by_label: dict) -> None:
        self._mask_by_label = mask_by_label

    def segment_detection(self, _bgr, det):  # noqa: ANN001, ANN202
        return _FakeSeg(det.label, self._mask_by_label[det.label])


class _VisCam:
    def __init__(self, depth_m: np.ndarray, rgb: np.ndarray) -> None:
        self._depth_m = depth_m
        self._rgb = rgb
        self._K = np.eye(3)
        self.added: list[str] = []

    def add_distance_to_image_plane_to_frame(self) -> None:
        self.added.append("depth")

    def set_clipping_range(self, *_a) -> None:  # noqa: ANN002
        self.added.append("clip")

    def get_depth(self):  # noqa: ANN201
        return self._depth_m

    def get_intrinsics_matrix(self):  # noqa: ANN201
        return self._K

    def get_current_frame(self):  # noqa: ANN201
        return {"rgb": self._rgb}


class MultiObjectVisionPerceptionSourceTests(unittest.TestCase):
    """P2.D.2: detect_all -> SAM2 per box -> N labelled segs, with canonical labels + top-referenced depth.
    Detector + segmenter are injected fakes (no isaacsim / torch / GDINO / SAM2)."""

    @staticmethod
    def _mask(region) -> np.ndarray:  # noqa: ANN001
        m = np.zeros((4, 4), dtype=np.uint8)
        m[region] = 1
        return m

    def _src(self, detector, segmenter, object_labels, mask_completion=None):  # noqa: ANN001, ANN202
        from src.robot.perception.mask_completion import DEFAULT_MASK_COMPLETION

        depth_m = np.full((4, 4), 5.0)  # far background (5 m)
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        cam = _VisCam(depth_m, rgb)
        return MultiObjectVisionPerceptionSource(
            camera=cam, detector=detector, segmenter=segmenter, prompt="the sugar box",
            object_labels=object_labels, warmup_steps=0,
            mask_completion=mask_completion or DEFAULT_MASK_COMPLETION,
        ), cam

    def test_detect_prompt_is_multi_phrase_over_labels(self) -> None:
        det = _FakeDetector([])
        src, _ = self._src(det, _FakeSegmenter({}), ["sugar box", "tomato soup can"])
        self.assertEqual(src._detect_prompt(), "sugar box. tomato soup can.")

    def test_canonicalizes_gdino_labels_to_spawn_names(self) -> None:
        # GDINO returns free-form phrases; the source maps them to the canonical spawn names (exact match).
        labels = ["sugar box", "tomato soup can", "cracker box"]
        det = _FakeDetector(["a sugar box", "soup can"])
        seg = _FakeSegmenter({"a sugar box": self._mask((0, slice(None))),
                              "soup can": self._mask((1, slice(None)))})
        src, _ = self._src(det, seg, labels)
        frame = src.acquire()
        got = [s.label for s in frame.segmentations]
        self.assertEqual(got, ["sugar box", "tomato soup can"])  # canonicalized -> set_target_label matches

    def test_the_rendered_surface_reaches_the_frame_UNTOUCHED(self) -> None:
        """⛔ THIS SOURCE USED TO OVERWRITE THE DEPTH UNDER EVERY MASK with one number, the nearest
        surface plus 12 mm, and this test asserted that number. An object therefore arrived at the
        calculator as a flat sheet at its top face: no body, no sides, no curvature. Where a grasp is
        anchored inside an object is decided per candidate by `grasping.geometry` instead.
        """
        labels = ["sugar box"]
        det = _FakeDetector(["sugar box"])
        mask = self._mask((slice(1, 3), slice(1, 3)))
        seg = _FakeSegmenter({"sugar box": mask})
        src, cam = self._src(det, seg, labels)
        # A face with relief: the near corner at 0.909 m, the rest of it 30 mm further away.
        d = np.full((4, 4), 5.0)
        d[1:3, 1:3] = 0.939
        d[1, 1] = 0.909
        cam._depth_m = d

        frame = src.acquire()

        m = np.asarray(frame.segmentations[0].mask).astype(bool)
        self.assertAlmostEqual(float(frame.depth_map[1, 1]), 909.0)
        self.assertAlmostEqual(float(frame.depth_map[2, 2]), 939.0, msg="relief was flattened away")
        self.assertGreater(float(frame.depth_map[m].max() - frame.depth_map[m].min()), 0.0)
        self.assertAlmostEqual(float(frame.depth_map[0, 0]), 5000.0)

    def test_the_two_published_arrays_AGREE(self) -> None:
        """`surface_depth_map` was added because `depth_map` was something else. They now carry the
        same measurement, as separate arrays, so a noise harness can move one and not the other."""
        det = _FakeDetector(["sugar box"])
        seg = _FakeSegmenter({"sugar box": self._mask((slice(1, 3), slice(1, 3)))})
        src, cam = self._src(det, seg, ["sugar box"])
        d = np.full((4, 4), 5.0)
        d[1:3, 1:3] = 0.909
        cam._depth_m = d

        frame = src.acquire()

        np.testing.assert_array_equal(frame.depth_map, frame.surface_depth_map)
        self.assertIsNot(frame.depth_map, frame.surface_depth_map)

    def test_duplicate_label_detections_are_all_kept(self) -> None:
        # GDINO merges/duplicates: two phrases both canonicalize to "sugar box". We deliberately do NOT
        # dedup -- the orchestrator evaluates EVERY target-labelled seg and executes the best GRASP, which is
        # more reliable than a detection-score heuristic (score-dedup once discarded the real box -> 0/5).
        big = self._mask((slice(0, 3), slice(0, 3)))   # the real box
        small = self._mask((3, 3))                       # a spurious fragment
        det = _FakeDetector([("sugar box cracker box", 0.30), ("a sugar box", 0.55)])
        seg = _FakeSegmenter({"sugar box cracker box": small, "a sugar box": big})
        src, _ = self._src(det, seg, ["sugar box", "tomato soup can"])
        frame = src.acquire()
        self.assertEqual(len(frame.segmentations), 2)  # both kept (the orchestrator picks the best grasp)
        self.assertEqual([s.label for s in frame.segmentations], ["sugar box", "sugar box"])  # canonicalized

    def test_fills_incomplete_mask_to_detection_box(self) -> None:
        # P5.1: a SAM2 mask covering far less than the GDINO box (incomplete long-box mask) is filled to the
        # box -> centred, complete silhouette (robust grasp). det.box = [x0,y0,x1,y1] pixels.
        #
        # ⚠ NO LONGER THE DEFAULT. Measured 2026-08-20 over 270 reference scenes, this rule cost 7.5
        # percentage points of top-1 and lost five grasps for every one it won, because it fires on
        # any non-rectangular mask rather than an incomplete one. It is asked for BY NAME here so the
        # capability stays tested; `test_the_default_leaves_a_mask_alone` pins what now ships.
        from src.robot.perception.mask_completion import MaskCompletion

        src, _ = self._src(_FakeDetector([]), _FakeSegmenter({}), ["sugar box"],
                           mask_completion=MaskCompletion.AXIS_ALIGNED_BOX)
        m = np.zeros((10, 20), dtype=bool)
        m[4:6, 1:6] = True  # a small left-shifted blob (10 px) inside a much larger detection box
        det = _FakeDet("sugar box")
        det.box = [1.0, 2.0, 17.0, 8.0]  # box area = 16*6 = 96; blob 10 << 0.8*96 -> fill
        out = src._maybe_fill_to_detection_box(m, det)
        self.assertTrue(out[2:8, 1:17].all())          # filled to the detection box
        self.assertEqual(int(out.sum()), 16 * 6)        # exactly the box area
        # centroid is the box centre (not the left-shifted blob)
        ys, xs = np.where(out)
        self.assertAlmostEqual(float(xs.mean()), (1 + 16) / 2, delta=0.6)

    def test_the_default_leaves_a_mask_alone(self) -> None:
        """What ships now: SAM2's silhouette reaches the calculator exactly as segmented."""
        src, _ = self._src(_FakeDetector([]), _FakeSegmenter({}), ["sugar box"])
        m = np.zeros((10, 20), dtype=bool)
        m[4:6, 1:6] = True                      # the very mask the old default replaced
        det = _FakeDet("sugar box")
        det.box = [1.0, 2.0, 17.0, 8.0]
        self.assertIs(src._maybe_fill_to_detection_box(m, det), m)

    def test_keeps_near_complete_mask(self) -> None:
        # A mask already covering most of the box is kept (SAM2 precision preserved).
        from src.robot.perception.mask_completion import MaskCompletion

        src, _ = self._src(_FakeDetector([]), _FakeSegmenter({}), ["sugar box"],
                           mask_completion=MaskCompletion.AXIS_ALIGNED_BOX)
        m = np.zeros((10, 20), dtype=bool)
        m[2:8, 1:17] = True  # ~fills the box
        det = _FakeDet("sugar box")
        det.box = [1.0, 2.0, 17.0, 8.0]
        out = src._maybe_fill_to_detection_box(m, det)
        self.assertIs(out, m)  # unchanged (>= 0.8 of box)

    def test_no_rgb_yields_no_segmentations(self) -> None:
        class _NoRgbCam(_VisCam):
            def get_current_frame(self):  # noqa: ANN201
                return {}  # no "rgb" key
        cam = _NoRgbCam(np.full((4, 4), 5.0), np.zeros((4, 4, 3), np.uint8))
        src = MultiObjectVisionPerceptionSource(
            camera=cam, detector=_FakeDetector(["x"]), segmenter=_FakeSegmenter({}),
            prompt="x", object_labels=["x"], warmup_steps=0,
        )
        self.assertEqual(src.acquire().segmentations, ())


if __name__ == "__main__":
    unittest.main()
