"""The locator places what one open camera grounds in BASE, and refuses what it cannot place.

Every test runs off the box, against duck-typed camera owners and a backend double. The locator composes the pick
frame's RealSense source, so the stamp and the wrist check are the ones ``test_perception_shutter_stamp`` holds.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from src.calibration.rig_calibration import RigCalibration
from src.calibration.serialization import FlangeToTcp
from src.camera.setup.image_taking.frames import RGBDFrame, StereoFrame
from src.contracts import UNSET
from src.geometry import Frame, Pose, Transform
from src.robot.core.errors import PerceptionFrameMoved
from src.robot.perception.locator import Located, Locator, LocatorRefused

_REPO = Path(__file__).resolve().parents[1]
_FX = 500.0
_SHAPE = (40, 40)
_K = np.array([[_FX, 0.0, 20.0], [0.0, _FX, 20.0], [0.0, 0.0, 1.0]])
#: A fixed camera a metre above the bench, looking straight down.
_CAMERA_TO_BASE = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1000.0], [0.0, 0.0, 0.0, 1.0]])


def _block_frame(*, captured_at_s: float | None = 50.0) -> RGBDFrame:
    """A bench a metre down and a block, 44 mm across, whose top is 80 mm up under the centre pixels."""
    depth = np.full(_SHAPE, 1000, dtype=np.uint16)
    depth[8:32, 8:32] = 920
    return RGBDFrame(color=np.zeros((*_SHAPE, 3), dtype=np.uint8), depth=depth, captured_at_s=captured_at_s)


def _block_mask() -> np.ndarray:
    mask = np.zeros(_SHAPE, dtype=bool)
    mask[8:32, 8:32] = True
    return mask


class _Handle:
    def __init__(self, frames: "list | None" = None) -> None:
        self.rig_id = "overhead"
        self._frames = frames if frames is not None else [_block_frame()]
        self.grabs = 0

    def grab(self):  # noqa: ANN201
        frame = self._frames[min(self.grabs, len(self._frames) - 1)]
        self.grabs += 1
        return frame

    def get_intrinsics(self) -> np.ndarray:
        return _K.copy()


class _Owner:
    """An open camera owner: a rig id, a source, a handle and a calibration."""

    def __init__(self, *, source: str = "rgbd", calibration: object | None = None, handle: _Handle | None = None,
                 rig_id: str = "overhead") -> None:
        self.rig_id = rig_id
        self.source = source
        self._handle = handle if handle is not None else _Handle()
        self._calibration = calibration if calibration is not None else _fixed()

    def handle(self) -> _Handle:
        return self._handle

    def calibration(self):  # noqa: ANN201
        if isinstance(self._calibration, Exception):
            raise self._calibration
        return self._calibration


def _fixed() -> RigCalibration:
    return RigCalibration(rig_id="overhead", mounting_mode="eye_to_hand", artifact_path="eth.json",
                          transform=Transform.from_matrix(_CAMERA_TO_BASE, from_frame=Frame.CAMERA, to_frame=Frame.BASE))


def _wrist(*, record: object = "declared") -> RigCalibration:
    flange = FlangeToTcp.from_matrix("willy", np.eye(4)) if record == "declared" else (UNSET if record is None else record)
    return RigCalibration(rig_id="wrist", mounting_mode="eye_in_hand", artifact_path="eih.json",
                          transform=Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.TOOL),
                          shutter_motion_tolerance_mm=1.0, shutter_motion_tolerance_deg=0.5, flange_to_tcp=flange)


_TOOL_FRAME = SimpleNamespace(source="willy", offset_mm=(0.0, 0.0, 0.0), rotation_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                              verify_tolerance_mm=1.0)


@dataclass(frozen=True)
class _Segmentation:
    """Shaped like ``SegmentationResult``, which the RealSense source rebuilds with ``dataclasses.replace``."""

    label: str
    score: float
    bbox_xyxy: tuple
    mask: np.ndarray


def _backend(*labels: str):  # noqa: ANN202
    """A perception backend double that grounds one block per label, all on the same pixels."""
    objects = tuple(
        SimpleNamespace(detection=SimpleNamespace(score=0.9, box=[16.0, 16.0, 24.0, 24.0]),
                        segmentation=_Segmentation(label=label, score=0.9, bbox_xyxy=(16.0, 16.0, 24.0, 24.0),
                                                   mask=_block_mask()))
        for label in labels
    )
    return SimpleNamespace(perceive=lambda _bgr, _prompt: objects)


def _pose(x: float) -> Pose:
    return Pose(position_mm=np.array([x, 0.0, 1000.0]), quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]), frame=Frame.BASE)


class ALocatedCloudTests(unittest.TestCase):
    def test_a_located_cloud_is_what_scene_from_cloud_takes(self) -> None:
        from src.robot.grasping.scene import Scene

        located = Locator.from_parts(camera=_Owner(), backend=_backend("red cube")).locate("a red cube")

        (obj,) = located.objects
        assert obj.centre_mm is not None
        self.assertAlmostEqual(80.0, obj.centre_mm[2], delta=1.0)
        scene = Scene.from_cloud(obj.points_base_mm, support_height_mm=0.0)
        np.testing.assert_allclose(np.median(scene.target_points_base_mm, axis=0), obj.centre_mm, atol=1.0)

    def test_the_capture_time_is_the_owners(self) -> None:
        located = Locator.from_parts(camera=_Owner(handle=_Handle([_block_frame(captured_at_s=77.25)])),
                                     backend=_backend("red cube")).locate("a red cube")
        self.assertEqual(77.25, located.captured_at_s)
        self.assertIsInstance(located, Located)

    def test_orientation_is_marked_unmeasured(self) -> None:
        located = Locator.from_parts(camera=_Owner(), backend=_backend("red cube")).locate("a red cube")
        orientation = located.objects[0].orientation
        assert orientation is not None
        self.assertFalse(orientation.measured)
        self.assertEqual("unmeasured", orientation.quality)

    def test_an_empty_result_says_a_detector_failure_reads_the_same(self) -> None:
        located = Locator.from_parts(camera=_Owner(), backend=_backend()).locate("a red cube")
        self.assertEqual((), located.objects)
        self.assertIn("a detector that failed read the same", located.render())


class TheLocatorRefusesWhatItCannotPlaceTests(unittest.TestCase):
    def test_an_uncalibrated_rig_is_refused_naming_the_key(self) -> None:
        from src.calibration.rig_calibration import RigNotCalibrated

        try:
            RigCalibration.from_config("overhead", None)
        except RigNotCalibrated as exc:
            refusal = exc
        with self.assertRaises(RigNotCalibrated) as caught:
            Locator.from_parts(camera=_Owner(calibration=refusal), backend=_backend())
        self.assertIn("camera.cameras.rigs['overhead'].extrinsics", str(caught.exception))

    def test_a_stereo_rig_is_refused_with_a_sentence(self) -> None:
        from src.config import load_config
        from src.camera.orchestration.camera import Camera

        rigs = load_config().camera.cameras.rigs
        stereo_rig = next(rig for rig in rigs if rig.source != "rgbd")
        owner = Camera.from_rig(stereo_rig, streamer=SimpleNamespace())
        with self.assertRaises(LocatorRefused) as caught:
            Locator.from_parts(camera=owner, backend=_backend())
        self.assertIn("carries no depth of its own", str(caught.exception))

        pair = StereoFrame(left=np.zeros((*_SHAPE, 3), dtype=np.uint8), right=np.zeros((*_SHAPE, 3), dtype=np.uint8))
        locator = Locator.from_parts(camera=_Owner(handle=_Handle([pair])), backend=_backend())
        with self.assertRaises(LocatorRefused) as grabbed:
            locator.locate("a red cube")
        self.assertIn("stereo pair", str(grabbed.exception))

    def test_a_wrist_locator_without_a_tcp_reader_is_refused(self) -> None:
        from src.robot.drivers.dummy.arm import DummyRobotArm

        with self.assertRaises(LocatorRefused) as caught:
            Locator.from_parts(camera=_Owner(calibration=_wrist(), rig_id="wrist"), backend=_backend(),
                               tool_frame=_TOOL_FRAME)
        self.assertIn("no reader of the arm's TCP", str(caught.exception))

        arm = DummyRobotArm(initial_pose=_pose(0.0))
        arm.connect()
        located = Locator.from_parts(camera=_Owner(calibration=_wrist(), rig_id="wrist"), backend=_backend("cube"),
                                     tool_pose=arm.get_tcp_pose, tool_frame=_TOOL_FRAME).locate("a cube")
        self.assertEqual("eye_in_hand", located.mounting)
        self.assertIsNotNone(located.tool_to_base_mm)

    def test_a_wrist_locator_that_keeps_moving_raises_after_its_attempts(self) -> None:
        poses = iter(_pose(5.0 * n) for n in range(100))
        handle = _Handle()
        locator = Locator.from_parts(camera=_Owner(calibration=_wrist(), rig_id="wrist", handle=handle),
                                     backend=_backend("cube"), tool_pose=lambda: next(poses), attempts=2,
                                     tool_frame=_TOOL_FRAME)
        with self.assertRaises(PerceptionFrameMoved):
            locator.locate("a cube")
        self.assertGreaterEqual(handle.grabs, 3)

    def test_a_wrist_locator_refuses_a_missing_or_stale_flange_to_tcp_record(self) -> None:
        moved = np.eye(4)
        moved[:3, 3] = (0.0, 0.0, 12.0)
        for label, calibration, says in (
            ("no record", _wrist(record=None), "records none"),
            ("a stale record", _wrist(record=FlangeToTcp.from_matrix("willy", moved)), "stale"),
        ):
            with self.subTest(label), self.assertRaises(LocatorRefused) as caught:
                Locator.from_parts(camera=_Owner(calibration=calibration, rig_id="wrist"), backend=_backend(),
                                   tool_pose=lambda: _pose(0.0), tool_frame=_TOOL_FRAME)
            self.assertIn(says, str(caught.exception))


class ALocatedTargetLeavesTheWorldTests(unittest.TestCase):
    def test_a_located_target_is_left_out_of_a_world_wired_like_the_cell(self) -> None:
        from src.config.schema.robot import RobotConfig
        from src.robot.core.keep_out import keeping_out
        from src.robot.execution.camera_world_wiring import CameraWorldPlan, CameraWorldWiring
        from src.robot.safety.planning.perceived import LinkCapsule, SelfEnvelope

        cfg = RobotConfig.model_validate({
            "vendor": "ur", "safety": {"payload": {"enforce": False}, "planning_world": {
                "enabled": True,
                "support_plane": {"height_mm": 0.0, "extent_mm": [1600.0, 1600.0], "thickness_mm": 50.0},
                "perceived": {"enabled": True},
            }},
        })
        rigs = [SimpleNamespace(rig_id=name, enabled=True, source="rgbd",
                                extrinsics=SimpleNamespace(mounting_mode="eye_to_hand")) for name in ("overhead", "side")]
        owners = {name: _Owner(rig_id=name, handle=_Handle([_block_frame(captured_at_s=50.0)])) for name in ("overhead", "side")}
        plan = CameraWorldPlan.from_config(cfg, rigs, primary_rig_id="overhead")
        world = CameraWorldWiring.from_cameras(cfg, plan=plan, cameras=owners).world
        assert world is not None
        self_envelope = SelfEnvelope(frames_mm=(np.eye(4),), capsules=(
            LinkCapsule(frame=0, start_mm=(-500.0, -500.0, 600.0), end_mm=(-500.0, -500.0, 700.0), radius_mm=10.0),))

        seen = world.world_for(self_envelope=self_envelope, now=50.1)
        self.assertEqual(1, seen.perceived_count, seen.render())

        located = Locator.from_parts(camera=owners["overhead"], backend=_backend("red cube")).locate("a red cube")
        arm = SimpleNamespace(live_planner_world=world)
        with keeping_out(arm, located.keep_out(0)):
            world.drop_cached_frames()
            held = world.world_for(self_envelope=self_envelope, now=50.1)
        self.assertEqual(0, held.perceived_count, held.render())


class TheLocatorImportsLittleTests(unittest.TestCase):
    _PROBE = (
        "import sys\n"
        "import {module}\n"
        "bad = sorted(m for m in sys.modules if m.startswith(("
        "'torch', 'src.robot.drivers', 'src.robot.grasping.loop', 'src.camera', "
        "'src.models', 'pyrealsense2')))\n"
        "print(','.join(bad))\n"
    )

    def _loaded(self, module: str) -> str:
        out = subprocess.run([sys.executable, "-c", self._PROBE.format(module=module)], cwd=_REPO,
                             capture_output=True, text=True, timeout=300)
        self.assertEqual(0, out.returncode, out.stderr)
        return out.stdout.strip()

    def test_the_locator_imports_no_driver_no_pick_loop_no_camera_no_models_and_no_torch(self) -> None:
        self.assertEqual("", self._loaded("src.robot.perception.locator"))

    def test_the_control_a_pick_loop_import_is_seen(self) -> None:
        self.assertIn("src.robot.grasping.loop", self._loaded("src.robot.grasping.loop.pick_loop"))


def _cell_tree() -> SimpleNamespace:
    return SimpleNamespace(
        safety=SimpleNamespace(planning_world=SimpleNamespace(perceived=SimpleNamespace(fresh_frame_attempts=4))),
        gripper=SimpleNamespace(tool_frame=_TOOL_FRAME),
    )


class FromConfigTests(unittest.TestCase):
    def test_a_rig_it_cannot_place_is_refused_before_any_model_loads(self) -> None:
        from src.config import load_config
        from src.camera.orchestration.camera import Camera

        stereo_rig = next(rig for rig in load_config().camera.cameras.rigs if rig.source != "rgbd")
        spec = mock.MagicMock()
        with mock.patch("src.models.perception_spec.PerceptionSpec", spec):
            with self.assertRaises(LocatorRefused):
                Locator.from_config(_cell_tree(), object(),
                                    camera=Camera.from_rig(stereo_rig, streamer=SimpleNamespace()))
        spec.from_config.assert_not_called()

    def test_from_config_builds_the_backend_the_cell_builds(self) -> None:
        built = _backend("red cube")
        spec = mock.MagicMock()
        spec.from_config.return_value.build.return_value = built
        tree = _cell_tree()
        models = object()
        with mock.patch("src.models.perception_spec.PerceptionSpec", spec):
            locator = Locator.from_config(tree, models, camera=_Owner())
        spec.from_config.assert_called_once_with(models)
        self.assertIs(built, locator._backend)  # noqa: SLF001
        self.assertEqual(4, locator._attempts)  # noqa: SLF001


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
