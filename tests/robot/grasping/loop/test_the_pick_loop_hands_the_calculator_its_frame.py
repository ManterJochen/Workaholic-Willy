"""What the pick loop hands the calculator of a frame: its colour where an overlay is drawn, and the target a label
names with every other part beside it.

The calculator draws the grasps over the frame's colour only where its ``render_debug_images`` asks, and the loop
forwards the colour then alone. A ``target_label`` makes the matching segmentation the only one tried, every other
one handed as ``other_object_masks``. (These lived beside the grasp sampling modes, which reach the calculator no more
since 2026-10-16.)
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.geometry import Frame, Pose
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.loop.pick_loop import (
    BinPickingOrchestrator,
    PerceptionFrame,
)


class _RecordingCalculator:
    """Captures the kwargs passed to ``compute_result`` for assertions."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def compute_result(self, *args: object, **kwargs: object) -> GraspResult:
        self.calls.append(dict(kwargs))
        return GraspResult(reasons=(GraspFailureReason.NO_VALID_GRASP,))


class _FakeArm:
    def __init__(self) -> None:
        self._tcp = Pose(
            position_mm=np.zeros(3),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.BASE,
            label="home",
        )

    def get_tcp_pose(self) -> Pose:
        return self._tcp

    def move_to(self, pose: Pose, **_: object) -> None:
        self._tcp = pose


class _SinglePerception:
    def __init__(self) -> None:
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        depth = np.full((8, 8), 500.0, dtype=np.float64)
        intrinsics = np.array(
            [[200.0, 0.0, 4.0], [0.0, 200.0, 4.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self._frame = PerceptionFrame(
            depth_map=depth,
            intrinsics=intrinsics,
            segmentations=(SimpleNamespace(mask=mask),),
        )

    def acquire(self) -> PerceptionFrame:
        return self._frame


class _RgbPerception:
    """A perception frame that carries rgb (the P0 grasp-point-viewer path)."""

    def __init__(self) -> None:
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        depth = np.full((8, 8), 500.0, dtype=np.float64)
        intrinsics = np.array(
            [[200.0, 0.0, 4.0], [0.0, 200.0, 4.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        self._frame = PerceptionFrame(
            depth_map=depth,
            intrinsics=intrinsics,
            segmentations=(SimpleNamespace(mask=mask),),
            rgb=rgb,
        )

    def acquire(self) -> PerceptionFrame:
        return self._frame


class OrchestratorDebugRenderForwardingTests(unittest.TestCase):
    """P0: pick_loop forwards the frame rgb to the calculator ONLY when render_debug_images is on."""

    def _run(self, calc: _RecordingCalculator) -> _RecordingCalculator:
        orch = BinPickingOrchestrator(
            arm=_FakeArm(),
            calculator=calc,  # type: ignore[arg-type]
            perception=_RgbPerception(),
            max_attempts=1,
        )
        orch.run()
        return calc

    def test_rgb_forwarded_when_render_enabled(self) -> None:
        calc = _RecordingCalculator()
        calc.render_debug_images = True  # type: ignore[attr-defined]
        self._run(calc)
        self.assertIsNotNone(calc.calls[0].get("rgb_image"))

    def test_rgb_not_forwarded_when_render_disabled(self) -> None:
        calc = _RecordingCalculator()
        calc.render_debug_images = False  # type: ignore[attr-defined]
        self._run(calc)
        self.assertNotIn("rgb_image", calc.calls[0])

    def test_rgb_not_forwarded_when_flag_absent(self) -> None:
        # A calculator that never declares the flag (every pre-P0 fake) stays on the off path.
        calc = self._run(_RecordingCalculator())
        self.assertNotIn("rgb_image", calc.calls[0])


class _LabeledSeg:
    def __init__(self, mask: np.ndarray, label: str) -> None:
        self.mask = mask
        self.label = label
        self.prim_path = f"/World/{label}"


class _LabeledMultiPerception:
    """A frame with N labelled segmentations — the P2 dense-clutter substrate."""

    def __init__(self) -> None:
        intr = np.array([[200.0, 0.0, 4.0], [0.0, 200.0, 4.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        segs = []
        for i, label in enumerate(("red", "green", "blue")):
            m = np.zeros((8, 8), dtype=np.uint8)
            m[2:6, i * 2 : i * 2 + 2] = 1
            segs.append(_LabeledSeg(m, label))
        self._frame = PerceptionFrame(
            depth_map=np.full((8, 8), 500.0, dtype=np.float64),
            intrinsics=intr,
            segmentations=tuple(segs),
        )

    def acquire(self) -> PerceptionFrame:
        return self._frame


class _SegLabelRecordingCalculator:
    """Records the label of each segmentation handed to compute_result (i.e. tried as a target)."""

    def __init__(self) -> None:
        self.target_labels: list[str] = []
        self.neighbour_counts: list[int] = []

    def compute_result(self, seg: object, depth: object, **kwargs: object) -> GraspResult:
        self.target_labels.append(getattr(seg, "label", ""))
        masks = kwargs.get("other_object_masks") or []
        self.neighbour_counts.append(len(masks))  # type: ignore[arg-type]
        return GraspResult(reasons=(GraspFailureReason.NO_VALID_GRASP,))


class OrchestratorTargetLabelTests(unittest.TestCase):
    """P2: a HARD ``target_label`` restricts the EXECUTABLE target to the matching segmentation; the
    other segmentations stay as ``other_object_masks`` neighbours. Default ``None`` -> all tried."""

    def _run(self, target_label: str | None) -> _SegLabelRecordingCalculator:
        calc = _SegLabelRecordingCalculator()
        orch = BinPickingOrchestrator(
            arm=_FakeArm(), calculator=calc, perception=_LabeledMultiPerception(),  # type: ignore[arg-type]
            max_attempts=1, target_label=target_label,
        )
        orch.run()
        return calc

    def test_default_none_tries_all_segments(self) -> None:
        calc = self._run(None)
        self.assertEqual(calc.target_labels, ["red", "green", "blue"])

    def test_label_target_restricts_to_one_with_others_as_neighbours(self) -> None:
        calc = self._run("green")
        self.assertEqual(calc.target_labels, ["green"])  # only the prompted object is an executable target
        self.assertEqual(calc.neighbour_counts, [2])     # the 2 distractors are still neighbours (G4/G12)

    def test_unknown_label_grasps_nothing(self) -> None:
        # the prompted object is not present -> no executable target -> honest no-pick (never a distractor)
        calc = self._run("purple")
        self.assertEqual(calc.target_labels, [])


if __name__ == "__main__":
    unittest.main()
