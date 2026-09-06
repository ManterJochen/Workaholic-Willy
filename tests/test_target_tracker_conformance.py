"""R10.1 — behavioral conformance of every TargetTracker implementation to the re-identify contract.

The grasp-refinement tests pin each tracker's *specific* matching algebra (IoU ties, the centroid cap,
the 3D back-projection). This suite is the cross-tracker proof of the *shared* :class:`TargetTracker`
Protocol contract documented in ``backend/src/robot/grasping/refinement.py`` — the contract a learned
tracker would also have to honour to slot in. Every production tracker must:

* satisfy the runtime-checkable :class:`TargetTracker` Protocol;
* **NOT raise on a well-formed frame** — ``match`` returns a verdict, it never crashes the refine loop;
* return either :data:`None` (no candidate re-identified) **or** a ``(index, score)`` tuple whose
  ``index`` is a valid position in ``frame.segmentations`` and whose ``score`` is a finite float
  (never a bare ``bool`` / a raw mask / an out-of-range index);
* re-identify a co-located target (ACCEPT path) and refuse a disjoint, far-away one (REJECT/``None``
  path) — proving both verdict paths are genuinely exercised, not all-accept theatre.

The trackers are constructed exactly as the existing ``tests/test_grasp_refinement.py`` builds them
(default ctors) and exercised on the same ``PerceptionFrame`` / ``target_identity_from_segmentation``
fixtures, so this guards the real shipped tracker set — not hand-built stubs.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from src.geometry import Frame, Transform
from src.robot.grasping import (
    IoUCentroidTargetTracker,
    TargetTracker,
    WorldSpacePoseTracker,
    target_identity_from_segmentation,
)
from src.robot.grasping.types.perception import PerceptionFrame


def _intrinsics(shape: tuple[int, int] = (32, 32), f: float = 400.0) -> np.ndarray:
    return np.array(
        [[f, 0.0, shape[1] / 2.0], [0.0, f, shape[0] / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _cam_to_base(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> Transform:
    return Transform(
        translation_mm=np.array([tx, ty, tz], dtype=np.float64),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        from_frame=Frame.CAMERA,
        to_frame=Frame.BASE,
    )


def _seg_at(
    shape: tuple[int, int],
    row_slice: slice,
    col_slice: slice,
) -> SimpleNamespace:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[row_slice, col_slice] = 1
    return SimpleNamespace(mask=mask)


def _frame(segs: tuple[SimpleNamespace, ...], depth_val: float = 500.0,
           shape: tuple[int, int] = (32, 32)) -> PerceptionFrame:
    return PerceptionFrame(
        depth_map=np.full(shape, depth_val, dtype=np.float64),
        intrinsics=_intrinsics(shape),
        segmentations=segs,
    )


# Every real production TargetTracker, constructed with the same default idiom the
# grasp-refinement suite uses. A learned tracker would be appended here.
_TRACKERS: tuple[TargetTracker, ...] = (
    IoUCentroidTargetTracker(),
    WorldSpacePoseTracker(),
)


class TargetTrackerConformanceTests(unittest.TestCase):
    """Every production TargetTracker honours the shared re-identify Protocol contract."""

    def setUp(self) -> None:
        self._trackers = _TRACKERS
        self.assertGreaterEqual(
            len(self._trackers), 2, "expected at least the two shipped trackers"
        )
        # A target identity carrying BOTH the image-space mask and the back-projected 3D centroid,
        # so the world-space tracker exercises its real (non-fallback) 3D matching path while the
        # image-space tracker uses the mask. Centred so both can re-identify it.
        self._identity_seg = _seg_at((32, 32), slice(14, 19), slice(14, 19))
        self._identity = target_identity_from_segmentation(
            self._identity_seg,
            depth_map=np.full((32, 32), 500.0),
            intrinsics=_intrinsics(),
            camera_to_base=_cam_to_base(),
        )

    def test_every_tracker_satisfies_runtime_protocol(self) -> None:
        for tracker in self._trackers:
            with self.subTest(tracker=type(tracker).__name__):
                self.assertIsInstance(tracker, TargetTracker)

    def test_match_never_raises_and_returns_a_valid_verdict(self) -> None:
        # THE core invariant (refinement.py): match RETURNS a verdict on a well-formed frame — it never
        # raises (a raise would crash the refine loop instead of reporting "target lost"). For both a
        # co-located target and a disjoint one, every tracker must return either None or an in-range
        # (index, finite-score) tuple — never a bare bool / a mask / an out-of-bounds index.
        co_located = _frame((_seg_at((32, 32), slice(14, 19), slice(14, 19)),))
        disjoint = _frame((_seg_at((32, 32), slice(0, 3), slice(0, 3)),))
        cases = {"co_located": co_located, "disjoint": disjoint}
        for label, frame in cases.items():
            for tracker in self._trackers:
                with self.subTest(tracker=type(tracker).__name__, frame=label):
                    try:
                        verdict = tracker.match(
                            self._identity,
                            frame,
                            iou_threshold=0.35,
                            camera_to_base=_cam_to_base(),
                        )
                    except Exception as exc:  # noqa: BLE001 - the whole point is to prove no tracker raises
                        self.fail(
                            f"{type(tracker).__name__} raised {type(exc).__name__} on the {label} "
                            f"frame instead of returning a verdict: {exc!r}"
                        )
                    if verdict is None:
                        continue
                    self.assertNotIsInstance(
                        verdict, bool, "verdict must be None or a tuple, never a bare bool"
                    )
                    self.assertIsInstance(verdict, tuple)
                    self.assertEqual(len(verdict), 2)
                    idx, score = verdict
                    self.assertIsInstance(idx, int)
                    self.assertNotIsInstance(idx, bool)
                    self.assertGreaterEqual(idx, 0)
                    self.assertLess(
                        idx, len(frame.segmentations),
                        "matched index must point at a real segmentation in the frame",
                    )
                    self.assertIsInstance(score, float)
                    self.assertTrue(
                        math.isfinite(score), f"match score must be finite; got {score!r}"
                    )

    def test_accepts_colocated_target(self) -> None:
        # ACCEPT path: a single candidate sitting at the identity's pixel + 3D location must be
        # re-identified (index 0, a positive finite score) by EVERY tracker — proving the suite is not
        # all-reject. Co-located so the image-space (IoU) and world-space (3D) paths both fire.
        frame = _frame((_seg_at((32, 32), slice(14, 19), slice(14, 19)),))
        for tracker in self._trackers:
            with self.subTest(tracker=type(tracker).__name__):
                verdict = tracker.match(
                    self._identity, frame, iou_threshold=0.35, camera_to_base=_cam_to_base()
                )
                self.assertIsNotNone(
                    verdict, f"{type(tracker).__name__} failed to re-identify a co-located target"
                )
                assert verdict is not None
                idx, score = verdict
                self.assertEqual(idx, 0)
                self.assertGreater(score, 0.0)

    def test_rejects_disjoint_far_target(self) -> None:
        # REJECT path: a lone candidate that neither overlaps the identity mask (IoU 0) nor sits near
        # its 3D centroid must yield None from EVERY tracker — proving both verdict paths are genuinely
        # exercised, so the suite is not all-accept (or all-error-swallow) theatre.
        #
        # measure>map: a corner candidate at the SAME depth back-projects only ~26 mm from the centred
        # identity — INSIDE the WorldSpacePoseTracker's default 50 mm tolerance, so it would match. To
        # make a SHARED reject anchor (both trackers refuse it) the far region also sits at a far DEPTH
        # (2000 vs 500 mm) → ~1.5 m away in 3D (rejects world-space) while IoU stays 0 (rejects image-space).
        far = _seg_at((32, 32), slice(0, 4), slice(0, 4))  # corner: no mask overlap with the centred identity
        depth = np.full((32, 32), 500.0, dtype=np.float64)
        depth[0:4, 0:4] = 2000.0  # the far candidate sits far behind the identity plane
        frame = PerceptionFrame(
            depth_map=depth, intrinsics=_intrinsics(), segmentations=(far,)
        )
        for tracker in self._trackers:
            with self.subTest(tracker=type(tracker).__name__):
                verdict = tracker.match(
                    self._identity, frame, iou_threshold=0.35, camera_to_base=_cam_to_base()
                )
                self.assertIsNone(
                    verdict,
                    f"{type(tracker).__name__} matched a disjoint far-away candidate "
                    f"(should fail closed to None)",
                )

    def test_matched_index_points_at_the_closest_candidate(self) -> None:
        # Documented invariant: when several candidates are present, the returned index selects a
        # genuine re-identification — the co-located one — not just "the first non-empty mask". Builds a
        # frame [far, near]; every tracker must pick index 1 (near), not 0 (far).
        far = _seg_at((32, 32), slice(0, 5), slice(0, 5))
        near = _seg_at((32, 32), slice(14, 19), slice(14, 19))
        frame = _frame((far, near))
        for tracker in self._trackers:
            with self.subTest(tracker=type(tracker).__name__):
                verdict = tracker.match(
                    self._identity, frame, iou_threshold=0.35, camera_to_base=_cam_to_base()
                )
                self.assertIsNotNone(verdict)
                assert verdict is not None
                idx, _score = verdict
                self.assertEqual(
                    idx, 1, "tracker must re-identify the co-located candidate, not the far one"
                )


if __name__ == "__main__":
    unittest.main()
