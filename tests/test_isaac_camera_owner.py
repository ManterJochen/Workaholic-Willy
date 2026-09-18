"""The Isaac overhead camera answers as an opened rig, off the box, as the owner decided.

A grab pumps the renderer before it reads depth, because Isaac's annotators go stale without it, and reads its shutter
time before the pump, as a real owner does. Depth arrives in millimetres with no-return pixels at zero. The camera is
placed by the fitted extrinsic, and a camera whose depth never becomes readable refuses rather than placing a world by
the hand built matrix.
"""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

from src.geometry import Frame, Transform
from src.willy_sim.perception.camera_owner import IsaacCameraOwner, _angle_deg


def _transform(angle_deg: float = 0.0) -> Transform:
    a = math.radians(angle_deg)
    m = np.eye(4)
    m[:3, :3] = [[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]]
    m[:3, 3] = [450.0, 0.0, 1000.0]
    return Transform.from_matrix(m, from_frame=Frame.CAMERA, to_frame=Frame.BASE)


class _Events:
    def __init__(self) -> None:
        self.log: list[str] = []


class _App:
    def __init__(self, events: _Events) -> None:
        self._events = events

    def update(self) -> None:
        self._events.log.append("update")


class _Session:
    def __init__(self, events: _Events) -> None:
        self._events = events
        self.app = _App(events)

    def step(self, *, render: bool) -> None:
        self._events.log.append(f"step(render={render})")


class _Camera:
    def __init__(self, events: _Events, depth_m: np.ndarray, *, refuse_annotator: bool = False) -> None:
        self._events = events
        self._depth = depth_m
        self._refuse = refuse_annotator
        self.clip: tuple[float, float] | None = None

    def add_distance_to_image_plane_to_frame(self) -> None:
        if self._refuse:
            raise RuntimeError("no annotator on this camera")

    def set_clipping_range(self, near: float, far: float) -> None:
        self.clip = (near, far)

    def get_depth(self) -> np.ndarray:
        self._events.log.append("depth")
        return self._depth

    def get_intrinsics_matrix(self) -> np.ndarray:
        return np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])


class _Handles:
    def __init__(self, camera: _Camera, camera_to_base: Transform | None) -> None:
        self.camera = camera
        self.camera_to_base = camera_to_base


class TheGrabTests(unittest.TestCase):
    def test_a_grab_reads_its_shutter_time_then_pumps_then_reads_depth_in_mm(self) -> None:
        events = _Events()
        depth = np.array([[1.0, np.nan], [0.25, np.inf]])
        clock_calls: list[int] = []

        def clock() -> float:
            clock_calls.append(len(events.log))
            return 123.5

        owner = IsaacCameraOwner(_Camera(events, depth), _Session(events), rig_id="overhead",
                                 camera_to_base=_transform(), pumps=2, clock=clock)
        frame = owner.handle().grab()
        self.assertEqual([0], clock_calls, "the shutter time was read after the pump")
        self.assertEqual(["step(render=True)", "update", "step(render=True)", "update", "depth"], events.log)
        self.assertEqual(123.5, frame.captured_at_s)
        np.testing.assert_array_equal(np.array([[1000.0, 0.0], [250.0, 0.0]]), frame.depth)

    def test_the_depth_read_is_the_one_after_the_pump(self) -> None:
        events = _Events()

        class _Stale(_Camera):
            def get_depth(self) -> np.ndarray:
                pumped = "step(render=True)" in events.log
                return np.full((1, 1), 0.5 if pumped else 9.0)

        owner = IsaacCameraOwner(_Stale(events, np.zeros((1, 1))), _Session(events), rig_id="overhead",
                                 camera_to_base=_transform(), pumps=1)
        np.testing.assert_array_equal(np.full((1, 1), 500.0), owner.handle().grab().depth)
        np.testing.assert_array_equal(_Camera(events, np.zeros((1, 1))).get_intrinsics_matrix(),
                                      owner.handle().get_intrinsics())

    def test_the_rig_is_a_calibrated_eye_to_hand_rgbd_camera_with_no_body(self) -> None:
        rig = IsaacCameraOwner.rig_for("overhead")
        self.assertEqual(("overhead", True, "rgbd", "eye_to_hand", None),
                         (rig.rig_id, rig.enabled, rig.source, rig.extrinsics.mounting_mode, rig.body))
        fitted = _transform(3.0)
        owner = IsaacCameraOwner(object(), object(), rig_id="overhead", camera_to_base=fitted)
        self.assertIs(fitted, owner.calibration().camera_to_base())
        self.assertEqual("eye_to_hand", owner.calibration().mounting_mode)


class ItWiresALiveWorldOffTheBoxTests(unittest.TestCase):
    def test_the_one_builder_a_real_cell_uses_gives_a_world_led_by_the_overhead_camera(self) -> None:
        from unittest.mock import MagicMock

        from src.config.loader import load_config
        from src.robot.drivers.sim.arm import IsaacRobotArm
        from src.robot.drivers.sim.config import SimRobotConfig
        from src.robot.execution.robot import Robot

        arm = IsaacRobotArm(SimRobotConfig(enabled=True, scene="placeholder.usd", robot_prim_path="/World/Robot"))
        arm._connected = True
        arm._get_curobo_client = MagicMock(  # type: ignore[method-assign]
            side_effect=AssertionError("the planner was asked"))
        owner = IsaacCameraOwner(object(), object(), rig_id="overhead", camera_to_base=_transform())
        robot = load_config(profile="sim,sim_camera_world").robot
        wired = Robot.from_parts(arm=arm, gripper=None, lock_key=None, cameras=[owner], robot_config=robot)
        self.assertIsNotNone(wired.camera_world)
        self.assertIsNotNone(wired.camera_world.world, wired.camera_world.reason)
        self.assertEqual(("overhead",), tuple(wired.camera_world.cameras))


class FromTheSceneTests(unittest.TestCase):
    def test_it_places_the_camera_by_the_fit_after_pumping_until_depth_is_readable(self) -> None:
        events = _Events()
        camera = _Camera(events, np.ones((2, 2)))
        fitted = _transform(2.0)
        outcomes = [RuntimeError("depth not ready"), (fitted, 0.3)]

        def fit(_camera):
            result = outcomes.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        with patch("src.willy_sim.scene.camera_to_base_ground_truth", fit):
            owner = IsaacCameraOwner.from_scene(_Session(events), _Handles(camera, _transform(0.0)),
                                                rig_id="overhead", warmup_steps=3)
        self.assertIs(fitted, owner.calibration().camera_to_base())
        self.assertEqual((0.05, 1.0e6), camera.clip)
        self.assertAlmostEqual(2.0, owner.fit_angle_deg or 0.0, places=6)
        self.assertEqual(3 + 12, events.log.count("step(render=True)"), "the retry did not pump again")

    def test_a_camera_whose_depth_never_fits_is_refused(self) -> None:
        events = _Events()

        def never(_camera):
            raise RuntimeError("empty depth buffer")

        with patch("src.willy_sim.scene.camera_to_base_ground_truth", never):
            with self.assertRaises(ValueError) as caught:
                IsaacCameraOwner.from_scene(_Session(events), _Handles(_Camera(events, np.ones((2, 2))), None),
                                            rig_id="overhead", warmup_steps=1, attempts=3)
        self.assertIn("gave no depth to fit its extrinsic from", str(caught.exception))
        self.assertIn("empty depth buffer", str(caught.exception))

    def test_a_camera_that_cannot_give_depth_is_refused(self) -> None:
        events = _Events()
        with self.assertRaises(ValueError) as caught:
            IsaacCameraOwner.from_scene(_Session(events), _Handles(_Camera(events, np.ones((2, 2)),
                                                                            refuse_annotator=True), None),
                                        rig_id="overhead")
        self.assertIn("cannot give depth", str(caught.exception))
        self.assertEqual([], events.log, "it pumped a camera it had already refused")

    def test_the_angle_is_none_without_a_scene_matrix(self) -> None:
        self.assertIsNone(_angle_deg(_transform(), None))
        self.assertAlmostEqual(90.0, _angle_deg(_transform(90.0), _transform(0.0)) or 0.0, places=6)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
