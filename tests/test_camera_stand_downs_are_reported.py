"""Every way a camera can contribute nothing must leave a trace, and they are not the same way.

A fused cell has three outcomes per camera and they call for three different answers:

* it delivered a frame and surfaces were fused          -- the working case
* it delivered a frame and grounded nothing on it       -- legitimate; an empty stretch of bin
* it delivered no frame at all                          -- a fault; what `on_camera_unavailable` is for

⛔⛔ TWO OF THE THREE WERE INVISIBLE. A camera that grounded nothing left `_fused_scene` through a
bare `continue` with no log and no counter, and a camera that FAILED could not be observed at all:
`MappedCameraRig.acquire_all` built its tuple in a generator with no exception handling, so one
camera's error propagated out of the pick. The protocol it implements promises the opposite twenty
lines above it -- "a camera that failed to produce a frame is simply ABSENT from the returned tuple"
-- and because the exception left before the expected-against-delivered comparison ran, BOTH branches
of `on_camera_unavailable` were unreachable with the only rig in the tree.

The schema states the cost in the option's own words: "a cell must never fall back to single-view
silently, which is the whole failure this option exists to make visible."

⚠ NEVER RUN ON HARDWARE. No physical multi-camera cell exists; this is proved against a fake rig.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.config.schema.robot.grasping_schema import FusionGeometryConfig
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


def _orchestrator(rig, **kwargs) -> BinPickingOrchestrator:  # noqa: ANN001
    return BinPickingOrchestrator(
        arm=SimpleNamespace(),  # type: ignore[arg-type]
        calculator=SimpleNamespace(),  # type: ignore[arg-type]
        perception=SimpleNamespace(acquire=_frame),  # type: ignore[arg-type]
        multi_camera_perception=rig,
        camera_frame_resolvers={"left": _Resolver(), "right": _Resolver()},  # type: ignore[dict-item]
        **kwargs,
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


class TheFailureIsNowVisibleToThePolicyTests(unittest.TestCase):
    """The point of the repair above: `on_camera_unavailable` can finally see a failed camera."""

    def test_refuse_fires_on_a_camera_that_raised(self) -> None:
        orch = _orchestrator(
            MappedCameraRig({"left": _Camera(), "right": _Camera(fails="pipeline not started")}),
            configured_camera_ids=("left", "right"),
            fusion_geometry_config=FusionGeometryConfig(
                enabled=True, on_camera_unavailable="refuse"),
        )

        with self.assertRaises(RuntimeError) as ctx:
            orch._fused_scene(_frame(), _IDENTITY)

        self.assertIn("right", str(ctx.exception))

    def test_the_refusal_says_WHY_the_camera_was_absent(self) -> None:
        """A refusal that names a camera and not its fault sends an operator to look at the wrong
        thing. The reason is the only part of the message that shortens the search.

        ⚠ IT MUST BE THE POLICY'S REFUSAL, NOT THE CAMERA'S OWN EXCEPTION, and asserting only on the
        reason text cannot tell them apart: with the old rig the camera's `RuntimeError` reached this
        line unchanged and carried the same words. So the policy is named too.
        """
        orch = _orchestrator(
            MappedCameraRig({"left": _Camera(), "right": _Camera(fails="pipeline not started")}),
            configured_camera_ids=("left", "right"),
            fusion_geometry_config=FusionGeometryConfig(
                enabled=True, on_camera_unavailable="refuse"),
        )

        with self.assertRaises(RuntimeError) as ctx:
            orch._fused_scene(_frame(), _IDENTITY)

        message = str(ctx.exception)
        self.assertIn("on_camera_unavailable='refuse'", message)
        self.assertIn("pipeline not started", message)

    def test_degrade_warns_with_the_reason_and_keeps_the_pick(self) -> None:
        orch = _orchestrator(
            MappedCameraRig({"left": _Camera(), "right": _Camera(fails="pipeline not started")}),
            configured_camera_ids=("left", "right"),
            fusion_geometry_config=FusionGeometryConfig(
                enabled=True, on_camera_unavailable="degrade"),
        )

        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            orch._fused_scene(_frame(), _IDENTITY)

        joined = "\n".join(logs.output)
        self.assertIn("right", joined)
        self.assertIn("pipeline not started", joined)


class ACameraThatFindsNothingIsNotAFaultTests(unittest.TestCase):
    """It delivered. A bin can be empty where one camera happens to be looking."""

    def test_it_does_NOT_count_as_missing(self) -> None:
        """Under `refuse` this decides whether a legitimate scene stops the cell every pick."""
        orch = _orchestrator(
            MappedCameraRig({"left": _Camera(), "right": _Camera(grounded=False)}),
            configured_camera_ids=("left", "right"),
            fusion_geometry_config=FusionGeometryConfig(
                enabled=True, on_camera_unavailable="refuse"),
        )

        orch._fused_scene(_frame(), _IDENTITY)  # must not raise

    def test_it_says_so_instead_of_dropping_out_in_silence(self) -> None:
        orch = _orchestrator(
            MappedCameraRig({"left": _Camera(), "right": _Camera(grounded=False)}),
            configured_camera_ids=("left", "right"),
            fusion_geometry_config=FusionGeometryConfig(enabled=True),
        )

        with self.assertLogs(_LOGGER, level="INFO") as logs:
            orch._fused_scene(_frame(), _IDENTITY)

        self.assertIn("right", "\n".join(logs.output))

    def test_it_is_counted_apart_from_an_absent_camera(self) -> None:
        """Two counters, because the two call for different action: one is a bin, one is a cable."""
        orch = _orchestrator(
            MappedCameraRig({"left": _Camera(), "right": _Camera(grounded=False),
                             "third": _Camera(fails="unplugged")}),
            configured_camera_ids=("left", "right", "third"),
            fusion_geometry_config=FusionGeometryConfig(enabled=True),
        )

        orch._fused_scene(_frame(), _IDENTITY)

        telemetry = orch._fusion_geometry_telemetry
        self.assertEqual(1, telemetry["cameras_grounded_nothing"])
        self.assertEqual(1, telemetry["cameras_absent"])

    def test_the_counters_are_stamped_even_when_NOTHING_contributed(self) -> None:
        """The early return is where a silent cell is most likely, so it stamps too. Without this,
        'no camera looked' and 'every camera looked and found nothing' produce the same telemetry."""
        orch = _orchestrator(
            MappedCameraRig({"left": _Camera(grounded=False), "right": _Camera(grounded=False)}),
            configured_camera_ids=("left", "right"),
            fusion_geometry_config=FusionGeometryConfig(enabled=True),
        )

        self.assertIsNone(orch._fused_scene(_frame(), _IDENTITY))
        self.assertEqual(2, orch._fusion_geometry_telemetry["cameras_grounded_nothing"])


if __name__ == "__main__":
    unittest.main()
