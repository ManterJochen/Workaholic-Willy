"""Every camera goes through `FrameProvider` — the per-rig seam, and the guard that keeps it true.

⚠ WHAT THIS CHANGED. `FrameProvider` has owned rig identity and lifecycle since it was written, and
hand detection and stereo capture used it — but the ONE path that drives real hardware did not. Both
`build_real_components` (the cell the CLI runner and the operator console build) and the bench
exerciser constructed a `RealSenseRGBDStreamer` straight from config, so every rig-keyed guarantee
here was simply not on the robot path.

Two things had to exist before they could be routed through it:

* **Intrinsics.** The provider could hand out frames but not the matrix that gives them a metric
  meaning, and both consumers need K — the perception adapter to unproject, `GraspCalculator` at
  construction. Its absence is a large part of why the bypass existed.
* **A per-rig lifecycle.** `open()`/`release()` are all-or-nothing. A cell holds ONE camera, and on
  this project an opened camera nobody closes has a history — so a consumer must be able to take one
  rig and give that one back, without acquiring or releasing devices it was never handed.

`RigHandle` is the seam: it answers `grab()`, `get_intrinsics()` and `release()`, which is exactly what
`RealSenseVisionPerceptionSource` already duck-types against, so nothing under `robot/perception`
learned about camera orchestration and `datagen`'s deliberately-fake shim still works.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

from src.camera.orchestration.frame_provider import (
    FrameProvider,
    FrameProviderStateError,
    RigHandle,
    UnknownCameraRigError,
)
from tests.test_camera_boundaries import _rgbd_rig, _webcam_rig


class FakeStreamer:
    """A device that can only be held once — which is what real hardware does."""

    def __init__(self, rig, intrinsics: np.ndarray | None = None) -> None:
        self.rig = rig
        self.opens = 0
        self.releases = 0
        self.is_open = False
        self._intrinsics = intrinsics

    def open(self) -> None:
        if self.is_open:
            raise RuntimeError(f"{self.rig.rig_id} is already streaming")
        self.is_open = True
        self.opens += 1

    def release(self) -> None:
        self.is_open = False
        self.releases += 1

    def grab(self):
        if not self.is_open:
            raise RuntimeError(f"{self.rig.rig_id} is not streaming")
        return f"frame:{self.rig.rig_id}"

    def get_intrinsics(self):
        return None if self._intrinsics is None else self._intrinsics.copy()

    def get_distortion(self):
        return None if self._intrinsics is None else np.zeros(5)


def _provider(*, intrinsics: np.ndarray | None = None) -> tuple[FrameProvider, dict[str, FakeStreamer]]:
    rigs = [_rgbd_rig("overhead"), _rgbd_rig("wrist"), _webcam_rig("stereo")]
    provider = FrameProvider(rigs)
    fakes = {rig.rig_id: FakeStreamer(rig, intrinsics if rig.rig_id == "overhead" else None)
             for rig in rigs}
    provider._streamers = dict(fakes)          # noqa: SLF001 - the point is to model a real device
    return provider, fakes


class PerRigLifecycleTests(unittest.TestCase):
    def test_open_rig_touches_ONLY_that_rig(self) -> None:
        provider, fakes = _provider()
        provider.open_rig("overhead")
        self.assertTrue(fakes["overhead"].is_open)
        self.assertFalse(fakes["wrist"].is_open)
        self.assertFalse(fakes["stereo"].is_open)
        self.assertEqual(provider.open_rig_ids(), frozenset({"overhead"}))

    def test_release_rig_leaves_every_other_rig_STREAMING(self) -> None:
        """The guarantee that makes a shared provider safe to hand out."""
        provider, fakes = _provider()
        provider.open()
        provider.release_rig("overhead")
        self.assertFalse(fakes["overhead"].is_open)
        self.assertTrue(fakes["wrist"].is_open)
        self.assertTrue(fakes["stereo"].is_open)

    def test_a_released_rig_refuses_the_next_grab_instead_of_reaching_a_dead_device(self) -> None:
        provider, _fakes = _provider()
        provider.open()
        provider.release_rig("overhead")
        with self.assertRaises(FrameProviderStateError):
            provider.grab("overhead")
        self.assertEqual(provider.grab("wrist"), "frame:wrist")

    def test_is_open_is_FALSE_once_one_rig_is_handed_back(self) -> None:
        """It used to be a single flag, which would have kept saying yes."""
        provider, _fakes = _provider()
        provider.open()
        self.assertTrue(provider.is_open)
        provider.release_rig("wrist")
        self.assertFalse(provider.is_open)

    def test_open_after_a_partial_release_reopens_only_what_is_missing(self) -> None:
        """A fake that raises on a double open is what catches a re-open of a live device — the exact
        failure real hardware gives when a second `pipeline.start()` hits a streaming device."""
        provider, fakes = _provider()
        provider.open()
        provider.release_rig("wrist")
        provider.open()                                   # must not re-open overhead or stereo
        self.assertEqual(fakes["overhead"].opens, 1)
        self.assertEqual(fakes["wrist"].opens, 2)

    def test_release_rig_is_idempotent_and_never_raises(self) -> None:
        provider, fakes = _provider()
        provider.open_rig("overhead")
        provider.release_rig("overhead")
        provider.release_rig("overhead")
        self.assertEqual(fakes["overhead"].releases, 1)

    def test_a_release_that_throws_is_logged_and_swallowed(self) -> None:
        """Teardown: a camera that cannot be closed must not stop the thing that was closing it."""
        provider, fakes = _provider()
        provider.open_rig("overhead")

        def explode() -> None:
            raise RuntimeError("device is wedged")

        fakes["overhead"].release = explode              # type: ignore[method-assign]
        provider.release_rig("overhead")                 # must not raise
        self.assertNotIn("overhead", provider.open_rig_ids())

    def test_an_unknown_rig_is_refused_by_name(self) -> None:
        provider, _fakes = _provider()
        with self.assertRaises(UnknownCameraRigError):
            provider.open_rig("nope")
        with self.assertRaises(UnknownCameraRigError):
            provider.release_rig("nope")


class IntrinsicsTests(unittest.TestCase):
    def test_the_provider_can_answer_the_camera_matrix(self) -> None:
        """Its absence is a large part of why the robot path bypassed this class."""
        k = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
        provider, _fakes = _provider(intrinsics=k)
        provider.open_rig("overhead")
        np.testing.assert_allclose(provider.get_intrinsics("overhead"), k)

    def test_a_rig_with_no_pinhole_matrix_answers_None_rather_than_guessing(self) -> None:
        provider, _fakes = _provider()
        provider.open_rig("stereo")
        self.assertIsNone(provider.get_intrinsics("stereo"))

    def test_intrinsics_require_an_OPEN_rig(self) -> None:
        """A closed RealSense reports `None`, and a caller that treated that as a camera matrix would
        build a `GraspCalculator` on nonsense. Refusing is louder."""
        provider, _fakes = _provider(intrinsics=np.eye(3))
        with self.assertRaises(FrameProviderStateError):
            provider.get_intrinsics("overhead")


class RigHandleTests(unittest.TestCase):
    def test_it_answers_the_SAME_surface_the_adapter_duck_types_against(self) -> None:
        """`RealSenseVisionPerceptionSource` calls the first three; `RGBDArucoMarkerSource` -- the
        hand-eye piece that lets a fixed RGB-D camera see the board -- also calls `get_distortion`.
        A handle missing that would have made every camera addressable and none calibratable."""
        for name in ("grab", "get_intrinsics", "get_distortion", "release"):
            self.assertTrue(callable(getattr(RigHandle, name, None)), name)

    def test_the_handle_satisfies_the_MARKER_SOURCE_contract_by_construction(self) -> None:
        """Asserted against the real adapter's own signature rather than a copied list, so the two
        cannot drift apart silently."""
        import inspect

        from src.calibration.rgbd_marker_source import RGBDArucoMarkerSource

        provider, _fakes = _provider(intrinsics=np.eye(3))
        provider.open_rig("overhead")
        source = RGBDArucoMarkerSource(streamer=provider.rig("overhead"), marker_length_mm=50.0)
        self.assertIn("streamer", inspect.signature(RGBDArucoMarkerSource.__init__).parameters)
        self.assertIsNotNone(source)

    def test_grab_and_intrinsics_reach_its_own_rig(self) -> None:
        k = np.eye(3) * 2.0
        provider, _fakes = _provider(intrinsics=k)
        provider.open()
        handle = provider.rig("overhead")
        self.assertEqual(handle.grab(), "frame:overhead")
        np.testing.assert_allclose(handle.get_intrinsics(), k)
        self.assertEqual(handle.rig_id, "overhead")

    def test_release_gives_back_ONE_rig(self) -> None:
        """The whole reason the handle has a `release` at all: the console's only teardown path is
        `api/lifecycle.release_perception` -> `perception.close()` -> this. A handle that could not
        release would have turned the one call that closes a camera into a silent no-op."""
        provider, fakes = _provider()
        provider.open()
        provider.rig("overhead").release()
        self.assertFalse(fakes["overhead"].is_open)
        self.assertTrue(fakes["wrist"].is_open)

    def test_a_handle_cannot_reach_another_rig(self) -> None:
        provider, _fakes = _provider()
        provider.open()
        handle = provider.rig("overhead")
        self.assertEqual(handle.grab(), "frame:overhead")
        provider.release_rig("overhead")
        self.assertFalse(handle.is_open)
        self.assertTrue(provider.rig("wrist").is_open)

    def test_asking_for_an_unknown_rig_refuses_at_HANDOUT_not_at_first_grab(self) -> None:
        provider, _fakes = _provider()
        with self.assertRaises(UnknownCameraRigError):
            provider.rig("nope")


class NoOneBypassesTheProviderTests(unittest.TestCase):
    """⚠ THE GUARD. Routing two call sites through the provider is worth nothing if the third one
    added next month builds its own streamer -- which is exactly how the robot path came to bypass a
    class that had existed for it all along."""

    _STREAMERS = {"RealSenseRGBDStreamer", "OpenCvRGBDStreamer",
                  "WebcamPairStreamer", "SingleDeviceStreamer"}
    #: The one module allowed to construct a device streamer: the provider is what owns them.
    _ALLOWED = {Path("src/camera/orchestration/frame_provider.py")}

    def test_only_the_frame_provider_constructs_a_device_streamer(self) -> None:
        root = Path(__file__).resolve().parents[1]
        offenders: list[str] = []
        for tree_root in ("src", "api", "datagen"):
            for path in (root / tree_root).rglob("*.py"):
                relative = path.relative_to(root)
                if relative in self._ALLOWED:
                    continue
                try:
                    parsed = ast.parse(path.read_text(encoding="utf-8"))
                except SyntaxError:                        # pragma: no cover - not our file to fix
                    continue
                for node in ast.walk(parsed):
                    if not isinstance(node, ast.Call):
                        continue
                    func = node.func
                    name = (func.id if isinstance(func, ast.Name)
                            else func.attr if isinstance(func, ast.Attribute) else None)
                    if name in self._STREAMERS:
                        offenders.append(f"{relative.as_posix()}:{node.lineno} builds {name}")
        self.assertEqual(offenders, [], "\n".join(
            ["these bypass FrameProvider -- take a `provider.rig(rig_id)` handle instead:", *offenders]))

    def test_the_guard_can_actually_FAIL(self) -> None:
        """A guard nobody has seen fail is a guard nobody can trust. This asserts the detector fires on
        the exact construction it is written to catch."""
        parsed = ast.parse("streamer = RealSenseRGBDStreamer(rig)\n")
        found = [n for n in ast.walk(parsed)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id in self._STREAMERS]
        self.assertEqual(len(found), 1)

    def test_the_two_rewired_call_sites_really_take_a_handle(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for relative in ("src/robot/execution/autonomous_grasp/cells.py",
                         "src/robot/perception/__main__.py"):
            text = (root / relative).read_text(encoding="utf-8")
            self.assertIn("FrameProvider(", text, relative)
            self.assertIn(".rig(", text, relative)


if __name__ == "__main__":
    unittest.main()
