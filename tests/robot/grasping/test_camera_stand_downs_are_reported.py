"""Every way a camera can contribute nothing must leave a trace, and they are not the same way.

A fused cell has three outcomes per camera and they call for three different answers:

* it delivered a frame and surfaces were fused          -- the working case
* it delivered a frame and grounded nothing on it       -- legitimate; an empty stretch of bin
* it delivered no frame at all                          -- a fault, warned of with its reason

⛔⛔ TWO OF THE THREE WERE INVISIBLE. A camera that grounded nothing left `_fused_scene` through a
bare `continue` with no log and no counter, and a camera that FAILED could not be observed at all:
`MappedCameraRig.acquire_all` built its tuple in a generator with no exception handling, so one
camera's error propagated out of the pick. The protocol it implements promises the opposite twenty
lines above it -- "a camera that failed to produce a frame is simply ABSENT from the returned tuple"
-- and because the exception left before the expected-against-delivered comparison ran, the warning
of a missing camera was unreachable with the only rig in the tree.

A cell must never fall back to single-view silently: that is the whole failure the warning exists to
make visible. A fused camera that delivers no frame is warned of and the pick goes on with the ones
that did; the refusal a cell could be set to instead (`on_camera_unavailable: refuse`) left with the
fusion keys on 2026-10-16, set in no shipped tree (the owner's cleanup).

⚠ NEVER RUN ON HARDWARE. No physical multi-camera cell exists; this is proved against a fake rig.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.geometry import Frame, Transform
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator
from src.robot.grasping.types.perception import MappedCameraRig, PerceptionFrame

_LOGGER = "src.robot.grasping.loop.pick_loop"
_IDENTITY = Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE)


def _frame(*, grounded: bool = True) -> PerceptionFrame:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:22, 10:22] = 1
    return PerceptionFrame(
        depth_map=np.full((32, 32), 500.0, dtype=np.float64),
        intrinsics=np.array([[400.0, 0.0, 16.0], [0.0, 400.0, 16.0], [0.0, 0.0, 1.0]]),
        segmentations=(SimpleNamespace(mask=mask),) if grounded else (),
    )


class _Camera:
    def __init__(self, *, grounded: bool = True, fails: str | None = None) -> None:
        self._grounded = grounded
        self._fails = fails

    def acquire(self) -> PerceptionFrame:
        if self._fails is not None:
            raise RuntimeError(self._fails)
        return _frame(grounded=self._grounded)


class _Resolver:
    def camera_to_base_for_frame(self, _frame_in, *, arm=None):  # noqa: ANN001, ANN202, ARG002
        return _IDENTITY


def _orchestrator(rig, cameras: tuple[str, ...] = ("left", "right")) -> BinPickingOrchestrator:  # noqa: ANN001
    """A cell fusing ``cameras``: a resolver each, as the overlay builds them for the cameras a cell fuses."""
    return BinPickingOrchestrator(
        arm=SimpleNamespace(),  # type: ignore[arg-type]
        calculator=SimpleNamespace(),  # type: ignore[arg-type]
        perception=SimpleNamespace(acquire=_frame),  # type: ignore[arg-type]
        multi_camera_perception=rig,
        camera_frame_resolvers={camera: _Resolver() for camera in cameras},  # type: ignore[misc]
    )


class TheRigKeepsGoingWhenOneCameraFailsTests(unittest.TestCase):
    def test_a_camera_that_raises_is_ABSENT_rather_than_fatal(self) -> None:
        """The contract the protocol writes down, now kept by the only implementation of it."""
        rig = MappedCameraRig({"left": _Camera(), "right": _Camera(fails="pipeline not started")})

        observations = rig.acquire_all()

        self.assertEqual(["left"], [o.camera_id for o in observations])

    def test_the_reason_survives_the_absence(self) -> None:
        """Absence carries no reason on its own, and 'unplugged' and 'driver raised' are different
        problems with different remedies."""
        rig = MappedCameraRig({"left": _Camera(), "right": _Camera(fails="pipeline not started")})

        rig.acquire_all()

        self.assertIn("right", rig.last_failures)
        self.assertIn("pipeline not started", rig.last_failures["right"])
        self.assertNotIn("left", rig.last_failures)

    def test_the_record_describes_the_LAST_acquisition_only(self) -> None:
        """A stale failure would report a camera as broken on every later pick."""
        rig = MappedCameraRig({"left": _Camera(), "right": _Camera(fails="one bad frame")})
        rig.acquire_all()
        rig.sources["right"]._fails = None  # type: ignore[attr-defined]

        rig.acquire_all()

        self.assertEqual({}, rig.last_failures)


class TheFailureIsNowVisibleTests(unittest.TestCase):
    """The point of the repair above: the pick can finally see a failed camera."""

    def test_a_camera_that_raised_is_warned_of_with_its_reason_and_the_pick_goes_on(self) -> None:
        orch = _orchestrator(MappedCameraRig({"left": _Camera(), "right": _Camera(fails="pipeline not started")}))

        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            self.assertIsNotNone(orch._fused_scene(_frame(), _IDENTITY), "the camera that delivered is fused")

        joined = "\n".join(logs.output)
        self.assertIn("right", joined)
        self.assertIn("pipeline not started", joined)


class ACameraThatFindsNothingIsNotAFaultTests(unittest.TestCase):
    """It delivered. A bin can be empty where one camera happens to be looking."""

    def test_it_does_NOT_count_as_missing(self) -> None:
        orch = _orchestrator(MappedCameraRig({"left": _Camera(), "right": _Camera(grounded=False)}))

        _views, missing, grounded_nothing, _clock = orch._fixed_camera_views()

        self.assertEqual([], missing)
        self.assertEqual(["right"], grounded_nothing)

    def test_it_says_so_instead_of_dropping_out_in_silence(self) -> None:
        orch = _orchestrator(MappedCameraRig({"left": _Camera(), "right": _Camera(grounded=False)}))

        with self.assertLogs(_LOGGER, level="INFO") as logs:
            orch._fused_scene(_frame(), _IDENTITY)

        self.assertIn("right", "\n".join(logs.output))

    def test_it_is_counted_apart_from_an_absent_camera(self) -> None:
        """Two counters, because the two call for different action: one is a bin, one is a cable."""
        orch = _orchestrator(
            MappedCameraRig({"left": _Camera(), "right": _Camera(grounded=False),
                             "third": _Camera(fails="unplugged")}),
            cameras=("left", "right", "third"),
        )

        orch._fused_scene(_frame(), _IDENTITY)

        telemetry = orch._fusion_geometry_telemetry
        self.assertEqual(1, telemetry["cameras_grounded_nothing"])
        self.assertEqual(1, telemetry["cameras_absent"])

    def test_the_counters_are_stamped_even_when_NOTHING_contributed(self) -> None:
        """The early return is where a silent cell is most likely, so it stamps too. Without this,
        'no camera looked' and 'every camera looked and found nothing' produce the same telemetry."""
        orch = _orchestrator(MappedCameraRig({"left": _Camera(grounded=False), "right": _Camera(grounded=False)}))

        self.assertIsNone(orch._fused_scene(_frame(), _IDENTITY))
        self.assertEqual(2, orch._fusion_geometry_telemetry["cameras_grounded_nothing"])


class AStandDownLeavesNoCountOfTheLastPickTests(unittest.TestCase):
    """The telemetry describes THIS pick, so a pick that stood down early carries no count of the one before.

    ⛔ It did. `_fused_scene` stamped its counters only once it reached the rig, so a pick that stood down
    before it (no camera fused, no multi-camera source, no CAMERA->BASE) kept the dict of the last pick that
    fused, and the service turned it into that record's `fused_view_count` (found by the review of
    2026-09-24 that added `fused_views` to the attempt).
    """

    def _fused_once(self) -> BinPickingOrchestrator:
        orch = _orchestrator(MappedCameraRig({"left": _Camera(), "right": _Camera()}))
        orch._fused_scene(_frame(), _IDENTITY)
        self.assertTrue(orch._fusion_geometry_telemetry, "the first pick fused and stamped its counters")
        return orch

    def test_a_pick_with_no_camera_to_base_carries_no_count_of_the_last(self) -> None:
        orch = self._fused_once()

        with self.assertLogs(_LOGGER, level="WARNING"):
            self.assertIsNone(orch._fused_scene(_frame(), None))

        self.assertEqual({}, orch._fusion_geometry_telemetry)

    def test_a_pick_of_a_cell_that_fuses_no_camera_carries_no_count_of_the_last(self) -> None:
        orch = self._fused_once()
        orch.camera_frame_resolvers = None

        self.assertIsNone(orch._fused_scene(_frame(), _IDENTITY))

        self.assertEqual({}, orch._fusion_geometry_telemetry)

    def test_a_pick_that_stands_down_keeps_no_other_camera_view_of_the_last(self) -> None:
        """⛔ The other cameras' views were set only where fusion ran, so a pick that stood down kept the last fused
        pick's views (review of 2026-09-24, outside the fused-views change), and a camera of another moment would
        vouch for this pick's way in (``_scene_views``)."""
        orch = self._fused_once()
        self.assertTrue(orch._last_other_views, "the first pick fused and kept the other camera's view")

        with self.assertLogs(_LOGGER, level="WARNING"):
            self.assertIsNone(orch._fused_scene(_frame(), None))

        self.assertEqual((), orch._last_other_views)


if __name__ == "__main__":
    unittest.main()
