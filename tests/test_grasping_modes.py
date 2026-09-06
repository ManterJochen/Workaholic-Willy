"""Tests for the typed grasp sampling mode contract (Phase G0)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.geometry import Frame, Pose
from src.robot.grasping import (
    GraspSamplingMode,
    mode_to_dense_sampling,
    resolve_grasp_sampling_mode,
)
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.loop.pick_loop import (
    BinPickingOrchestrator,
    PerceptionFrame,
)


# ---------------------------------------------------------------------------
# Pure mode contract
# ---------------------------------------------------------------------------


class ResolveModeTests(unittest.TestCase):
    def test_none_resolves_to_auto(self) -> None:
        self.assertIs(resolve_grasp_sampling_mode(None), GraspSamplingMode.AUTO)

    def test_bool_true_is_dense(self) -> None:
        self.assertIs(
            resolve_grasp_sampling_mode(True), GraspSamplingMode.DENSE_CLUTTER
        )

    def test_bool_false_is_single_object(self) -> None:
        self.assertIs(
            resolve_grasp_sampling_mode(False), GraspSamplingMode.SINGLE_OBJECT
        )

    def test_enum_passthrough(self) -> None:
        self.assertIs(
            resolve_grasp_sampling_mode(GraspSamplingMode.DENSE_CLUTTER),
            GraspSamplingMode.DENSE_CLUTTER,
        )

    def test_string_aliases(self) -> None:
        self.assertIs(
            resolve_grasp_sampling_mode("auto"), GraspSamplingMode.AUTO
        )
        self.assertIs(
            resolve_grasp_sampling_mode("Single"), GraspSamplingMode.SINGLE_OBJECT
        )
        self.assertIs(
            resolve_grasp_sampling_mode("single_object"),
            GraspSamplingMode.SINGLE_OBJECT,
        )
        self.assertIs(
            resolve_grasp_sampling_mode(" DENSE "), GraspSamplingMode.DENSE_CLUTTER
        )
        self.assertIs(
            resolve_grasp_sampling_mode("dense_clutter"),
            GraspSamplingMode.DENSE_CLUTTER,
        )

    def test_invalid_string_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_grasp_sampling_mode("turbo")
        self.assertIn("turbo", str(ctx.exception))
        self.assertIn("auto", str(ctx.exception))

    def test_invalid_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_grasp_sampling_mode(2)
        with self.assertRaises(ValueError):
            resolve_grasp_sampling_mode(object())

    def test_mode_to_dense_sampling_mapping(self) -> None:
        self.assertIsNone(mode_to_dense_sampling(GraspSamplingMode.AUTO))
        self.assertFalse(mode_to_dense_sampling(GraspSamplingMode.SINGLE_OBJECT))
        self.assertTrue(mode_to_dense_sampling(GraspSamplingMode.DENSE_CLUTTER))


# ---------------------------------------------------------------------------
# Orchestrator forwarding
# ---------------------------------------------------------------------------


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


class OrchestratorModeForwardingTests(unittest.TestCase):
    def _run(self, **kwargs: object) -> _RecordingCalculator:
        calc = _RecordingCalculator()
        orch = BinPickingOrchestrator(
            arm=_FakeArm(),
            calculator=calc,  # type: ignore[arg-type]
            perception=_SinglePerception(),
            max_attempts=1,
            **kwargs,  # type: ignore[arg-type]
        )
        orch.run()
        return calc

    def test_default_resolves_to_dense_clutter_legacy_bool(self) -> None:
        # Default ``dense_sampling=True`` (legacy) should resolve to
        # DENSE_CLUTTER so production telemetry is meaningful even when
        # callers never set the typed knob.
        calc = self._run()
        self.assertEqual(
            calc.calls[0]["grasp_sampling_mode"], GraspSamplingMode.DENSE_CLUTTER
        )

    def test_typed_mode_wins_over_dense_sampling(self) -> None:
        calc = self._run(
            dense_sampling=True, grasp_sampling_mode=GraspSamplingMode.SINGLE_OBJECT
        )
        self.assertEqual(
            calc.calls[0]["grasp_sampling_mode"], GraspSamplingMode.SINGLE_OBJECT
        )

    def test_legacy_bool_false_resolves_to_single_object(self) -> None:
        calc = self._run(dense_sampling=False)
        self.assertEqual(
            calc.calls[0]["grasp_sampling_mode"], GraspSamplingMode.SINGLE_OBJECT
        )

    def test_string_alias_accepted(self) -> None:
        calc = self._run(grasp_sampling_mode="auto")
        self.assertEqual(
            calc.calls[0]["grasp_sampling_mode"], GraspSamplingMode.AUTO
        )


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


# ---------------------------------------------------------------------------
# Calculator precedence + telemetry
# ---------------------------------------------------------------------------


def _camera_matrix() -> np.ndarray:
    return np.array(
        [[200.0, 0.0, 12.0], [0.0, 200.0, 12.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _convex_mask() -> np.ndarray:
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[7:17, 6:18] = 1
    return mask


def _depth(shape: tuple[int, int], value_mm: float = 1000.0) -> np.ndarray:
    return np.full(shape, value_mm, dtype=np.float64)


class CalculatorModePrecedenceTests(unittest.TestCase):
    def setUp(self) -> None:
        from src.robot.grasping.generation.calculator import GraspCalculator

        self.calc = GraspCalculator(
            min_grip_width_mm=1.0,
            max_grip_width_mm=200.0,
            max_candidates=3,
            camera_matrix=_camera_matrix(),
        )
        self.seg = SimpleNamespace(mask=_convex_mask(), label="t")
        self.depth = _depth((24, 24))

    def _telemetry(self, **kwargs: object) -> dict:
        self.calc.compute(self.seg, self.depth, pixel_to_mm=5.0, **kwargs)
        return dict(self.calc.last_telemetry)

    def test_default_legacy_bool_stamps_dense(self) -> None:
        telem = self._telemetry(dense_sampling=True)
        self.assertEqual(telem["grasp_sampling_mode"], "dense_clutter")

    def test_typed_mode_overrides_bool(self) -> None:
        telem = self._telemetry(
            dense_sampling=True,
            grasp_sampling_mode=GraspSamplingMode.SINGLE_OBJECT,
        )
        self.assertEqual(telem["grasp_sampling_mode"], "single_object")
        # SINGLE_OBJECT must run the silhouette path (no dense telemetry
        # keys appear when dense did not run).
        self.assertNotIn("dense_runtime_ms", telem)

    def test_dense_clutter_runs_dense_path(self) -> None:
        telem = self._telemetry(
            grasp_sampling_mode=GraspSamplingMode.DENSE_CLUTTER,
        )
        self.assertEqual(telem["grasp_sampling_mode"], "dense_clutter")
        self.assertTrue(telem["dense_decision"])
        self.assertIn("dense_runtime_ms", telem)
        self.assertIn("dense_timeout", telem)

    def test_auto_default_when_nothing_set(self) -> None:
        telem = self._telemetry()  # nothing supplied
        self.assertEqual(telem["grasp_sampling_mode"], "auto")


if __name__ == "__main__":
    unittest.main()
