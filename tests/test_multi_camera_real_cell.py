"""WS3.2: a PHYSICAL cell can finally run multi-camera fusion, because something builds the rig.

⛔⛔ **THE FEATURE WAS UNREACHABLE FROM CONFIG, AND ONLY THE LAST HALF WAS MISSING.**
`apply_orchestrator_overlays` wires `fusion_geometry_config`, the per-camera CAMERA->BASE resolver
map (fail-closed at construction, so a bad artifact raises while an operator is watching) and, since
WS3.4, the list of cameras the config asked for. Then its own comment says the RIG is a live device
handle the caller must pass in. **No caller did.** The single assignment to
`orchestrator.multi_camera_perception` in the whole tree was one Isaac runner reaching into the
attribute after construction, so a physical cell could name three calibrated cameras, load three
artifacts, and run single-view. The loop logged that it was doing so, which is the difference between
a gap and a lie, but the measured lever (top-1 43.50 % single-view against 55.93 % fused on the
datagen reference) was not available to any real cell.

⚠ **NEVER RUN ON HARDWARE.** Not one physical camera has been opened by this path. The rig is built
here against a fake provider and fake models; what is proven is the WIRING, and the difference is
stated rather than blurred. Bucket 3 until a real cell runs it.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from src.config.schema.robot import RobotConfig
from src.robot.execution.autonomous_grasp.cells import CellBuildRefused
from src.robot.grasping.types.perception import MappedCameraRig

class _Handle:
    """A rig handle shaped like the one `FrameProvider.rig` hands back."""

    def __init__(self, rig_id: str) -> None:
        self.rig_id = rig_id
        self.released = 0

    def get_intrinsics(self) -> np.ndarray:
        return np.eye(3)

    def release(self) -> None:
        self.released += 1


class _Provider:
    """Records which rigs were opened, which is the whole question for a multi-camera cell."""

    def __init__(self, rigs) -> None:  # noqa: ANN001
        self.known = [r.rig_id for r in rigs]
        self.opened: list[str] = []
        self.handles: dict[str, _Handle] = {}
        self.released = 0

    def open_rig(self, rig_id: str) -> None:
        if rig_id not in self.opened:
            self.opened.append(rig_id)

    def rig(self, rig_id: str) -> _Handle:
        return self.handles.setdefault(rig_id, _Handle(rig_id))

    def release(self) -> None:
        """⛔ ADDED BECAUSE THE REAL ONE ALWAYS HAD IT AND THIS DOUBLE DID NOT.

        A build that refuses after the primary camera is open now gives the device back, and this
        fake answered that call with `AttributeError`, so the double was a narrower object than the
        class it stands in for, and every assertion made through it was made against the narrower
        one. Counting rather than passing, because the refusal tests below can then say the device
        was returned instead of only that a refusal was raised.
        """
        self.released += 1
        self.opened.clear()


def _rig(rig_id: str, source: str = "rgbd"):
    return SimpleNamespace(rig_id=rig_id, source=source)


def _app_cfg(*rig_ids: str):
    """The first id is the primary, which is now named rather than inferred from list position."""
    cfg = mock.Mock()
    cfg.camera.cameras.rigs = [_rig(name) for name in rig_ids]
    cfg.camera.cameras.primary_rig_id = rig_ids[0] if rig_ids else ""
    return cfg


def _robot_cfg(*, geometry: bool, cameras: dict[str, bool]) -> RobotConfig:
    """A real validated config, because a mock would not prove the keys exist."""
    return RobotConfig.model_validate({
        "vendor": "dummy",
        "grasping": {
            "fusion": {
                "geometry": {"enabled": geometry},
                "cameras": {
                    cam_id: {"enabled": on, "extrinsics_artifact_path": f"{cam_id}.json"}
                    for cam_id, on in cameras.items()
                },
            },
        },
    })


def _build(robot_cfg: RobotConfig, app_cfg, provider_box: list):  # noqa: ANN001
    """Walk `build_real_components` with fake devices and a fake perception stack.

    ⚠ **THE SEAM MOVED ON 2026-09-05 AND THIS HELPER MOVED WITH IT.** It used to patch
    `build_object_detector` and `build_segmenter`, which is what `cells.py` called directly. The cell
    now builds through `PerceptionSpec.build()` so that `models.pipeline` reaches hardware at all, and
    the old patches stopped covering the call: `_app_cfg` is a `mock.Mock`, so `models.pipeline` is a
    truthy Mock, so `build_perception` took its PIPELINE branch and constructed a real GroundingDINO
    on the way to a `TypeError` deep in `pathlib`. Patching the seam the cell actually uses is both
    the fix and the point -- a test that mocks a function the code no longer calls is not a test.
    """
    from src.robot.execution.autonomous_grasp import cells

    def _provider(rigs):  # noqa: ANN001, ANN202
        made = _Provider(rigs)
        provider_box.append(made)
        return made

    def _stack(_self, **_kw):  # noqa: ANN001, ANN202
        # Shaped like a `TwoStageBackend`, because the identity assertions below read `.detector`
        # and `.segmenter` off it -- a fused cell must hold ONE set of weights, not one per camera.
        return SimpleNamespace(
            detector=SimpleNamespace(kind="detector"), segmenter=SimpleNamespace(kind="segmenter"),
        )

    with mock.patch("src.config.load_config", return_value=app_cfg), \
         mock.patch("src.camera.orchestration.frame_provider.FrameProvider", _provider), \
         mock.patch("src.models.perception_spec.PerceptionSpec.build",
                    autospec=True, side_effect=_stack) as build:
        result = cells.build_real_components(robot_cfg, "a box")
    return result, build


class TheRigIsBuiltTests(unittest.TestCase):

    def test_a_configured_second_camera_IS_OPENED_and_handed_over(self) -> None:
        """⭐ THE ITEM ITSELF. Two calibrated cameras in config, two devices opened, and the cell
        carries a rig the pick loop can observe."""
        box: list = []
        (_calc, _perc, _res, multi, _lenses), _build_calls = _build(
            _robot_cfg(geometry=True, cameras={"overhead": True, "oblique": True}),
            _app_cfg("overhead", "oblique"), box)

        self.assertIsInstance(multi, MappedCameraRig)
        self.assertEqual(sorted(box[0].opened), ["oblique", "overhead"])
        self.assertEqual(sorted(multi.sources), ["oblique"])

    def test_the_PRIMARY_is_not_in_the_rig(self) -> None:
        """⚠ `fuse_scene_geometry` takes the primary as its first four arguments and the rest as
        `other_views`; a primary in both is fused with a second copy of itself, and on hardware every
        extra observation costs a full detect plus segment pass.

        The Isaac runner that measured 43.50 -> 55.93 % DOES include its primary. That is redundancy
        rather than error, and it is left alone: changing it would move a measured number without a
        measurement to justify the move.
        """
        box: list = []
        (_c, _p, _r, multi, _lenses), _build_calls = _build(
            _robot_cfg(geometry=True, cameras={"overhead": True, "oblique": True}),
            _app_cfg("overhead", "oblique"), box)

        self.assertNotIn("overhead", multi.sources, "the primary camera is observed twice per pick")

    def test_the_two_models_are_built_ONCE_for_every_camera(self) -> None:
        """The expensive part of a cell is the weights, not the passes. Four cameras must not mean
        four GroundingDINO loads on one GPU."""
        box: list = []
        _r, build = _build(
            _robot_cfg(geometry=True,
                       cameras={"overhead": True, "left": True, "right": True, "wrist": True}),
            _app_cfg("overhead", "left", "right", "wrist"), box)

        self.assertEqual(build.call_count, 1, "one stack for four cameras, not four stacks")

    def test_every_camera_shares_the_SAME_model_objects(self) -> None:
        """The control on the count above: one build call could still have been copied per camera."""
        box: list = []
        (_c, primary, _r, multi, _lenses), _build_calls = _build(
            _robot_cfg(geometry=True, cameras={"overhead": True, "oblique": True}),
            _app_cfg("overhead", "oblique"), box)

        # ⭑ THE WHOLE BACKEND IS NOW SHARED, WHICH IS THE STRONGER CLAIM. The cell used to hand
        # each camera a detector and a segmenter and let every source compose its own wrapper; it now
        # hands the same built stack to all of them, so the identity to check is the stack itself.
        # The per-model identities are asserted too, because "same wrapper" would still be true of a
        # wrapper that had been handed two fresh models.
        other = multi.sources["oblique"]
        self.assertIs(other._backend, primary._backend)
        self.assertIs(other._backend.detector, primary._backend.detector)
        self.assertIs(other._backend.segmenter, primary._backend.segmenter)


class ItStaysOffByDefaultTests(unittest.TestCase):

    def test_no_fusion_geometry_opens_ONE_camera_exactly_as_before(self) -> None:
        """⚠ THE BYTE-IDENTICAL CASE, which is every cell that exists today."""
        box: list = []
        (_c, _p, _r, multi, _lenses), _build_calls = _build(
            _robot_cfg(geometry=False, cameras={"overhead": True, "oblique": True}),
            _app_cfg("overhead", "oblique"), box)

        self.assertIsNone(multi)
        self.assertEqual(box[0].opened, ["overhead"], "a device was opened that nothing observes")

    def test_a_config_naming_only_the_primary_builds_no_rig(self) -> None:
        """One camera is not a multi-camera rig, and an empty rig would make the pick loop expect
        observations nobody produces."""
        box: list = []
        (_c, _p, _r, multi, _lenses), _build_calls = _build(
            _robot_cfg(geometry=True, cameras={"overhead": True}),
            _app_cfg("overhead"), box)

        self.assertIsNone(multi)

    def test_a_DISABLED_camera_is_not_opened(self) -> None:
        box: list = []
        (_c, _p, _r, multi, _lenses), _build_calls = _build(
            _robot_cfg(geometry=True, cameras={"overhead": True, "oblique": False}),
            _app_cfg("overhead", "oblique"), box)

        self.assertIsNone(multi)
        self.assertEqual(box[0].opened, ["overhead"])


class ItFailsClosedTests(unittest.TestCase):

    def test_a_camera_with_no_matching_rig_is_refused_AT_BUILD(self) -> None:
        """⛔ Left to runtime this is worse than a build error: the id has a calibration artifact, so
        it joins the resolver map and `configured_camera_ids`, and then delivers no frame on every
        single pick -- refusing all of them under `refuse`, warning on all of them under `degrade`.
        Say it once, here, while an operator is watching."""
        box: list = []
        with self.assertRaises(CellBuildRefused) as caught:
            _build(_robot_cfg(geometry=True, cameras={"overhead": True, "ghost": True}),
                   _app_cfg("overhead", "oblique"), box)

        message = str(caught.exception)
        self.assertIn("ghost", message)
        self.assertIn("camera.cameras.rigs", message, "the refusal must name where to fix it")


class ARefusedBuildGivesTheCameraBackTests(unittest.TestCase):
    """⛔ MEASURED 2026-09-10. The primary camera was opened and then never released on any path
    that refuses afterwards, and several such paths exist by design: the multi-camera rig refuses an
    unmatched id, and `_preload()` was added precisely so a corrupt artifact refuses AT BUILD TIME
    rather than as an empty candidate list at 3 a.m.

    The consequence is worse than a warning. A RealSense held by a process that has exited stays
    claimed, so the operator who reads the refusal, fixes the config and runs again meets "device
    busy", a second failure that names neither the cause nor the first message. The refusal that
    was added to help became the thing that hid itself.

    ⭐ THIS LIVES IN THIS FILE BECAUSE THE HARNESS IS HERE, not because the property is about
    multiple cameras. The multi-camera refusal is simply the post-open refusal that is cheapest to
    provoke without a device.
    """

    def test_a_refusal_after_the_open_returns_the_device(self) -> None:
        box: list = []
        with self.assertRaises(CellBuildRefused):
            _build(_robot_cfg(geometry=True, cameras={"overhead": True, "ghost": True}),
                   _app_cfg("overhead", "oblique"), box)

        self.assertTrue(box, "the harness never reached the provider, so this proves nothing")
        provider = box[0]
        self.assertEqual(provider.released, 1,
                         "the build refused with the primary camera still open")
        self.assertEqual(provider.opened, [],
                         "release must give back every rig this build claimed, not only count")

    def test_a_build_that_succeeds_keeps_its_camera(self) -> None:
        """The other half, and the one that makes the test above mean something.

        `release()` on every exit would satisfy the assertion above and hand back a device the cell
        is about to stream from. What is pinned is the ASYMMETRY: refused builds release, successful
        ones hold.
        """
        box: list = []
        _build(_robot_cfg(geometry=False, cameras={}), _app_cfg("overhead"), box)

        provider = box[0]
        self.assertEqual(provider.released, 0, "a cell that came up must still hold its camera")
        self.assertEqual(provider.opened, ["overhead"])


class TheRootForwardsItTests(unittest.TestCase):

    def test_from_robot_config_accepts_and_wires_the_rig(self) -> None:
        """Without this the rig is built and dropped on the floor."""
        import inspect

        from src.robot.execution.autonomous_grasp.service import AutonomousGraspService

        signature = inspect.signature(AutonomousGraspService.from_robot_config)
        self.assertIn("multi_camera_perception", signature.parameters)
        self.assertIsNone(signature.parameters["multi_camera_perception"].default,
                          "the default must be None or every existing cell changes behaviour")

    def test_build_real_cell_passes_it_through(self) -> None:
        """Read off the source: the rig is built in the component builder and there is exactly one
        place it can reach the orchestrator."""
        from pathlib import Path

        cells = Path(
            "src/robot/execution/autonomous_grasp/cells.py").read_text(encoding="utf-8")
        service = Path(
            "src/robot/execution/autonomous_grasp/service.py").read_text(encoding="utf-8")

        self.assertIn("multi_camera_perception=multi_camera", cells)
        self.assertIn(
            "runtime.orchestrator.multi_camera_perception = multi_camera_perception", service)


class TheCamerasAreGivenBackTests(unittest.TestCase):
    """⛔ Opening several devices and closing one is a defect this repo has already paid for."""

    def test_closing_the_rig_closes_EVERY_camera(self) -> None:
        closed: list[str] = []
        rig = MappedCameraRig({
            "left": SimpleNamespace(close=lambda: closed.append("left")),
            "right": SimpleNamespace(close=lambda: closed.append("right")),
        })

        rig.close()

        self.assertEqual(sorted(closed), ["left", "right"])

    def test_a_source_with_no_close_is_the_NORMAL_case(self) -> None:
        """The sim sources own a simulator this rig does not; a missing `close` is not an error."""
        MappedCameraRig({"sim": SimpleNamespace()}).close()

    def test_one_camera_that_refuses_does_not_keep_the_others_open(self) -> None:
        """⚠ THE ORDER MATTERS. Every camera is attempted, and only then does the failure surface."""
        closed: list[str] = []

        def _angry() -> None:
            raise OSError("device busy")

        rig = MappedCameraRig({
            "angry": SimpleNamespace(close=_angry),
            "willing": SimpleNamespace(close=lambda: closed.append("willing")),
        })

        with self.assertRaises(RuntimeError) as caught:
            rig.close()

        self.assertEqual(closed, ["willing"], "a healthy camera stayed open because another failed")
        self.assertIn("angry", str(caught.exception))
        self.assertIn("device busy", str(caught.exception))

    def test_the_console_teardown_reaches_the_rig_too(self) -> None:
        """⛔ `release_perception` walked `orchestrator.perception` alone. On real hardware a second
        `pipeline.start()` on a streaming device FAILS, so every camera past the primary would have
        made the console unable to rebuild a fused cell without a restart."""
        from api.lifecycle import release_perception

        closed: list[str] = []
        service = SimpleNamespace(runtime=SimpleNamespace(orchestrator=SimpleNamespace(
            perception=SimpleNamespace(close=lambda: closed.append("primary")),
            multi_camera_perception=MappedCameraRig(
                {"oblique": SimpleNamespace(close=lambda: closed.append("oblique"))}),
        )))

        release_perception(service)

        self.assertEqual(sorted(closed), ["oblique", "primary"])

    def test_teardown_still_never_raises(self) -> None:
        """A camera that cannot be closed must not stop the thing that was closing it."""
        from api.lifecycle import release_perception

        def _angry() -> None:
            raise OSError("device busy")

        service = SimpleNamespace(runtime=SimpleNamespace(orchestrator=SimpleNamespace(
            perception=SimpleNamespace(close=_angry),
            multi_camera_perception=MappedCameraRig({"x": SimpleNamespace(close=_angry)}),
        )))

        release_perception(service)


if __name__ == "__main__":
    unittest.main()
