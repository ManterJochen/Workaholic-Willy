"""The bench's camera reads depth as the cell's D415 does (``scripts/ursim/_mat_scene.py``, ``depth_model="real"``).

The owner, 2026-10-07: a fix the bench validates must hold on the cell tomorrow, so the camera the bench stands in for
hands the stack the depth the cell's camera hands it, noise and holes and all. Its numbers (``_mat_scene.D415``) were
measured on the 30 looks the cell recorded that day (four look poses, the mat 550 to 1000 mm away). Pinned here:

* the error on the bare mat (25 px and more from any part, edge or hole) at 600-700 mm and at 800-900 mm, against the
  cell's own there: what one grab differs from the next (0.64 and 0.88 mm on the cell), what a frame reads less its
  own 31 px mean (0.70 and 1.04 mm) and about one plane (1.6 to 2.0 mm by look at 600-700 mm, 1.5 to 2.4 at 800-900);
* whole millimetres: a real render is the model's depth rounded, 0 wherever it reads nothing;
* the band at the left no second imager sees (105 px at 600-700 mm on the cell), the shadow left of a nearer surface and
  never right of it, a small step eased and a large one kept;
* one seed and one pose give one frame, another seed another, and a second grab from a pose keeps that pose's error;
* the clean render is the render as it stood before (its frames pinned by hash, the same on Windows and on Linux) and
  stays every scene's default, so the URSim probe and the tests that pin a render keep theirs;
* the bench renders the real depth unless a run says ``--noise clean``;
* every grab through the bench's camera (``StandInD415``) draws the grab's own error afresh, its pose's error and holes
  kept, as two grabs on the cell differ (the review of 2026-10-07: the camera cached its render per pose and handed
  it back byte for byte, so a judgement on a re-grab could flip on the cell and never on the bench);
* the harsh depth (``--noise harsh``) is worse than the cell's: more error at every scale, more holes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
from scipy import ndimage

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT / "scripts" / "bench"), str(_ROOT / "scripts" / "ursim")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import _mat_scene as ms  # noqa: E402
import scenes  # noqa: E402

from tests._cell_2026_10_01 import look  # noqa: E402

CROP = _ROOT / "tests" / "data" / "cell_2026_10_01" / "p1_pick-63e1b3dac29f.npz"


@lru_cache(maxsize=1)
def _static() -> Any:
    static = ms.StaticScene.from_crop(CROP)
    static.whole_bench = True
    return static


def _tool(raise_mm: float = 0.0) -> np.ndarray:
    """P1's look of 2026-10-01, the tool ``raise_mm`` higher."""
    tcp = look("P1").tool_to_base.copy()
    tcp[2, 3] += raise_mm
    return tcp


def _camera(raise_mm: float = 0.0) -> np.ndarray:
    return _tool(raise_mm) @ look("P1").camera_to_tool


class TheErrorOnTheMatTests(unittest.TestCase):
    """The camera 120 mm under P1's look reads the mat at 600-700 mm, 80 mm over it at 800-900 mm."""

    def _error(self, raise_mm: float, near_mm: float, far_mm: float) -> tuple[float, float, float]:
        """Two grabs each of two seeds: the grab-to-grab sigma, the 31 px high-pass and the rms about one plane of the
        error on the bare mat between ``near_mm`` and ``far_mm``, CAMERA millimetres, averaged over the seeds."""
        static, camera = _static(), _camera(raise_mm)
        clean, surface = static.surfaces(camera)
        mat = ndimage.binary_erosion(surface == ms._MAT, iterations=30) & (clean >= near_mm) & (clean < far_mm)
        temporal, highpass, plane = [], [], []
        for seed in (1, 2):
            scene = ms.Scene(static, depth_model="real", seed=seed)
            first = scene.render(camera, None).depth_mm.astype(np.float64)
            second = scene.render(camera, None).depth_mm.astype(np.float64)
            read = mat & (first > 0) & (second > 0)
            self.assertGreater(int(read.sum()), 50_000, "too little bare mat to measure on")
            temporal.append(float(np.std((first - second)[read]) / np.sqrt(2.0)))
            error = np.where(read, first - clean, 0.0)
            share = ndimage.uniform_filter(read.astype(np.float64), 31)
            whole = read & (share > 0.999)
            mean = ndimage.uniform_filter(error, 31) / np.maximum(share, 1e-9)
            highpass.append(float(np.sqrt(np.mean((error - mean)[whole] ** 2))))
            rows, cols = np.nonzero(read)
            a = np.column_stack([np.ones(rows.size), cols, rows])
            coef, *_ = np.linalg.lstsq(a, error[read], rcond=None)
            plane.append(float(np.sqrt(np.mean((error[read] - a @ coef) ** 2))))
        return float(np.mean(temporal)), float(np.mean(highpass)), float(np.mean(plane))

    def test_at_600_to_700_mm_the_error_is_the_cells(self) -> None:
        temporal, highpass, plane = self._error(-120.0, 600.0, 700.0)
        self.assertTrue(0.55 <= temporal <= 0.75, f"grab to grab {temporal:.2f} mm; the cell 0.64 (0.67-0.75 by parts)")
        self.assertTrue(0.60 <= highpass <= 0.90, f"less its 31 px mean {highpass:.2f} mm; the cell 0.70")
        self.assertTrue(1.40 <= plane <= 2.60, f"about one plane {plane:.2f} mm; the cell 1.6-2.0 by look, P1 2.3")

    def test_at_800_to_900_mm_the_error_is_the_cells_and_larger(self) -> None:
        temporal, highpass, plane = self._error(80.0, 800.0, 900.0)
        self.assertTrue(0.75 <= temporal <= 1.10, f"grab to grab {temporal:.2f} mm; the cell 0.88")
        self.assertTrue(0.85 <= highpass <= 1.30, f"less its 31 px mean {highpass:.2f} mm; the cell 1.04")
        self.assertTrue(1.40 <= plane <= 3.40, f"about one plane {plane:.2f} mm; the cell 1.5-2.4")
        near_temporal, near_highpass, _ = self._error(-120.0, 600.0, 700.0)
        self.assertGreater(temporal, near_temporal)
        self.assertGreater(highpass, near_highpass)


class WholeMillimetresTests(unittest.TestCase):
    def test_a_real_render_is_the_models_depth_rounded_and_0_where_it_reads_nothing(self) -> None:
        static = _static()
        scene = ms.Scene(static, depth_model="real", seed=5)
        scene.set_parts(scenes.SCENES["clutter_one_10"].build(static), target="cylinder")
        read_by_the_model: list[np.ndarray] = []
        model = ms.d415_depth

        def spy(*args: Any, **kwargs: Any) -> np.ndarray:
            out = model(*args, **kwargs)
            read_by_the_model.append(out.copy())
            return out

        with mock.patch.object(ms, "d415_depth", spy):
            render = scene.render(_camera(), _tool())
        self.assertEqual(np.uint16, render.depth_mm.dtype)
        depth = read_by_the_model[0]
        read = np.isfinite(depth) & (depth >= ms.MIN_Z_MM) & (depth <= ms.MAX_Z_MM)
        np.testing.assert_array_equal(render.depth_mm[read], np.rint(depth[read]))
        self.assertLessEqual(float(np.max(np.abs(render.depth_mm[read] - depth[read]))), 0.5)
        self.assertTrue(np.all(render.depth_mm[~read] == 0))
        self.assertTrue(0.85 < float(np.mean(read)) < 0.99, "the cell reads 81 to 90 % of a frame")
        for name, mask in render.masks.items():
            self.assertFalse(np.any(mask & (render.depth_mm == 0)), f"{name}'s mask holds a pixel with no depth")


class TheStereoGeometryTests(unittest.TestCase):
    """A 120 px square ``top_mm`` away on a plane 650 mm away, every random part of the model off."""

    def _read(self, top_mm: float) -> np.ndarray:
        quiet = dataclasses.replace(ms.D415(), temporal_mm=0.0, static_mm=0.0, warp_mm=0.0, surface_lost=(0.0,) * 8,
                                    grazing_lost=(0.0,) * 7, edge_lost=0.0, edge_lost_per_px=0.0)
        depth = np.full((720, 1280), 650.0)
        depth[300:420, 600:720] = top_mm
        surface = np.full(depth.shape, ms._MAT, dtype=np.int8)
        surface[300:420, 600:720] = ms._PART
        return ms.d415_depth(depth, surface, _static().intrinsics, static_rng=np.random.default_rng(4),
                             temporal_rng=np.random.default_rng(5), model=quiet)

    def test_the_band_at_the_left_is_as_wide_as_the_cells(self) -> None:
        first = np.argmax(np.isfinite(self._read(650.0)), axis=1)
        self.assertEqual(1, len(set(first.tolist())), "the band is as wide in every row of a plane")
        self.assertTrue(100 <= int(first[0]) <= 110, f"{int(first[0])} px; the cell 105 px at 600-700 mm")

    def test_a_nearer_surface_throws_its_shadow_to_its_left_alone(self) -> None:
        read = self._read(610.0)
        left = ~np.isfinite(read[300:420, 595:600])
        right = ~np.isfinite(read[300:420, 720:725])
        self.assertTrue(0.2 <= float(left.mean()) <= 0.6, f"{left.mean():.2f}; the cell 0.33-0.40 next to a part")
        self.assertEqual(0.0, float(right.mean()))

    def test_a_small_step_is_eased_and_a_large_one_kept(self) -> None:
        small, large = self._read(610.0), self._read(500.0)
        eased = (650.0 - float(np.nanmean(small[330:390, 720]))) / 40.0
        self.assertTrue(0.2 <= eased <= 0.45, f"{eased:.2f} of the step one pixel out; the cell about a third")
        self.assertGreater(float(np.nanmean(small[330:390, 719])) - 610.0, 5.0, "the top's own edge eases too")
        self.assertLess(abs(float(np.nanmean(small[350:370, 650:670])) - 610.0), 0.5, "the top's middle does not")
        self.assertLess(650.0 - float(np.nanmean(large[330:390, 720])), 1.0, "a bin wall's 150 mm step is kept")


class OneSeedOneFrameTests(unittest.TestCase):
    def _render(self, seed: int, camera: np.ndarray, scene: Any = None) -> Any:
        static = _static()
        if scene is None:
            scene = ms.Scene(static, depth_model="real", seed=seed)
            scene.set_parts(scenes.SCENES["clutter_one_10"].build(static), target="cylinder")
        return scene, scene.render(camera, None)

    def test_one_seed_and_one_pose_give_one_frame_and_another_seed_another(self) -> None:
        _, first = self._render(5, _camera())
        _, again = self._render(5, _camera())
        _, other = self._render(6, _camera())
        np.testing.assert_array_equal(first.depth_mm, again.depth_mm)
        for name in first.masks:
            np.testing.assert_array_equal(first.masks[name], again.masks[name])
        self.assertGreater(float(np.mean(first.depth_mm != other.depth_mm)), 0.3)

    def test_a_second_grab_from_one_pose_keeps_the_poses_error_and_its_holes(self) -> None:
        scene, first = self._render(5, _camera())
        _, second = self._render(5, _camera(), scene)
        _, moved = self._render(5, _camera(1.0), scene)
        np.testing.assert_array_equal(first.depth_mm == 0, second.depth_mm == 0)
        both = (first.depth_mm > 0) & (second.depth_mm > 0) & (moved.depth_mm > 0)
        regrab = (second.depth_mm.astype(np.float64) - first.depth_mm)[both]
        elsewhere = (moved.depth_mm.astype(np.float64) - first.depth_mm)[both]
        self.assertGreater(float(np.std(regrab)), 0.3, "nothing of the error is the grab's own")
        self.assertGreater(float(np.std(elsewhere)), 2.0 * float(np.std(regrab)), "a pose 1 mm away keeps the error")


class EveryGrabDrawsItsOwnErrorTests(unittest.TestCase):
    def _camera_on(self, depth_model: str) -> Any:
        static = _static()
        scene = ms.Scene(static, depth_model=depth_model, seed=5)
        scene.set_parts(scenes.SCENES["clutter_one_10"].build(static), target="cylinder")
        tool = _tool()
        camera = ms.StandInD415(object(), scene, camera_to_tool=look("P1").camera_to_tool, tcp=lambda: tool)
        camera.open()
        return camera

    def test_two_grabs_from_one_pose_differ_as_the_cells_do_and_keep_its_holes(self) -> None:
        camera = self._camera_on("real")
        first = camera.grab().depth.astype(np.float64)
        second = camera.grab().depth.astype(np.float64)
        self.assertEqual((1, 2), (camera.renders, camera.grabs), "the pose is rendered once and grabbed twice")
        np.testing.assert_array_equal(first == 0, second == 0)
        both = (first > 0) & (second > 0)
        differ = float(np.std((second - first)[both]) / np.sqrt(2.0))
        self.assertTrue(0.3 <= differ <= 1.5, f"grab to grab {differ:.2f} mm; the cell 0.64 to 0.88")
        for name, mask in camera.last.masks.items():
            self.assertFalse(bool(np.any(mask & (camera.last.depth_mm == 0))), f"{name}'s mask covers no depth")

    def test_a_clean_camera_hands_its_render_back_as_it_was(self) -> None:
        camera = self._camera_on("clean")
        np.testing.assert_array_equal(camera.grab().depth, camera.grab().depth)


class TheHarshDepthTests(unittest.TestCase):
    def test_the_harsh_depth_is_worse_than_the_cells_at_every_scale(self) -> None:
        static, camera = _static(), _camera(-120.0)
        clean, surface = static.surfaces(camera)
        mat = ndimage.binary_erosion(surface == ms._MAT, iterations=30) & (clean >= 600.0) & (clean < 700.0)
        error, lost = {}, {}
        for model in ("real", "harsh"):
            scene = ms.Scene(static, depth_model=model, seed=3)
            first = scene.render(camera, None).depth_mm.astype(np.float64)
            read = mat & (first > 0)
            error[model] = float(np.sqrt(np.mean((first - clean)[read] ** 2)))
            lost[model] = float(np.mean(first == 0))
        self.assertGreater(error["harsh"], 1.4 * error["real"], error)
        self.assertGreater(lost["harsh"], lost["real"], lost)

    def test_the_bench_takes_harsh_too(self) -> None:
        self.assertIn("harsh", ms.DEPTH_MODELS)
        self.assertIs(ms.HARSH, ms.Scene(_static(), depth_model="harsh").d415)


class TheCleanRenderIsUnchangedTests(unittest.TestCase):
    """clutter_one_10 from P1's look twice and 300 mm lower once, and bin_wall_20 from P1's look, rendered clean
    before 2026-10-07: sha256 of each depth, and of its masks in name order (``np.packbits``)."""

    PINNED = {
        "clutter_one_10": (
            ("bbfb8635c0738dd1bc8b8a01c867f0bee1a00b0300afb36874ca69043c74e860",
             "7154f8f90d19f7bee18cecee72cd5fecdfcdd26bb5a59317e32214b594d96bbd"),
            ("e55f9b46a63550a9eda9232ba5e84c7dc881e85cdab3d70bdf346cab661b41bb",
             "927f14519038e27a3ca50e00eb0c61778597b0c1650b9eae7ef1864280b4e4e1"),
            ("6c1d6cc0bcb8582b57ebbc19bebf26f2f2357397e549957e90cd15e06195b0c9",
             "1c4106fcdabff3c7d8c10a0d3e1b2904951dd0a74936a159a51d04dec2b6ace0"),
        ),
        "bin_wall_20": (
            ("7e99b22968b0649b07a4ce0739370ef0faa4b2873f59164a799c2624de0852ad",
             "f9c954f79be861b0815cc9dbf12aee7844060b7c8443bcf41d5b4f68dbb19a34"),
        ),
    }

    def test_every_scene_still_renders_clean_unless_it_names_another_depth(self) -> None:
        self.assertEqual("clean", ms.DEPTH_MODEL)
        self.assertEqual("clean", ms.Scene(_static()).depth_model)
        with self.assertRaises(ValueError):
            ms.Scene(_static(), depth_model="drawn")

    def test_the_clean_render_is_the_render_as_it_stood(self) -> None:
        static = _static()
        poses = [(_tool(), _camera()), (_tool(), _camera()), (_tool(-300.0), _camera(-300.0))]
        for name, pinned in self.PINNED.items():
            scene = ms.Scene(static)
            scene.set_parts(scenes.SCENES[name].build(static), target=scenes.SCENES[name].target)
            for index, ((depth_hash, mask_hash), (tcp, camera)) in enumerate(zip(pinned, poses)):
                render = scene.render(camera, tcp)
                self.assertEqual(depth_hash, hashlib.sha256(np.ascontiguousarray(render.depth_mm).tobytes()).hexdigest(),
                                 f"{name}, render {index}: the clean depth changed")
                masks = b"".join(np.packbits(mask).tobytes() for _, mask in sorted(render.masks.items()))
                self.assertEqual(mask_hash, hashlib.sha256(masks).hexdigest(), f"{name}, render {index}: the masks")


class TheBenchRendersTheRealDepthTests(unittest.TestCase):
    def test_the_bench_asks_for_the_real_depth_unless_a_run_says_clean(self) -> None:
        import run_bench

        asked: list[str] = []

        class _Bench:
            def __init__(self, args: Any) -> None:
                asked.append(args.noise)

            def run(self, names: list[str]) -> dict[str, Any]:
                return {"scenes": {}}

        with tempfile.TemporaryDirectory() as work, mock.patch.object(run_bench, "Bench", _Bench), \
                redirect_stdout(io.StringIO()):
            run_bench.main(["--scenes", "clutter_one_10", "--work", work])
            run_bench.main(["--scenes", "clutter_one_10", "--work", work, "--noise", "clean"])
        self.assertEqual(["real", "clean"], asked)
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            run_bench.main(["--noise", "drawn"])


if __name__ == "__main__":
    unittest.main()
