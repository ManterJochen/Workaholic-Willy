"""The self filter takes the robot out of what the cameras see: all of the robot, and only the robot.

Step 4h (owner Q6 A and Q7 A; the hand as asked again before building). The filter was capsules along the DH origins,
90 mm around every link and 150 mm around the last segment. Measured on the tree before this step
(logs/step4/red_4h_polyline_measure.log), 83.4 % of a ur5e's upper arm and 66.7 % of a ur3e's lie outside it, because
the DH polyline does not follow the shoulder offset; and the sim placed the origins without the base yaw its own guard
uses, so its filter stood mirrored through the base.

Now the arm is one capsule per link, fitted once to the committed bundle in the link's own DH frame; the hand is the
spheres of its resolved sphere map, the model the planner checks against, each grown just enough to hold the hand's
committed mesh; a part in the gripper is one capsule from the
fingertips while it is attached; everything is padded by `perceived.margin_mm`; and the model and the yaw are the self
collision guard's.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import yaml
from pydantic import ValidationError

from src.config.grippers import available_grippers
from src.config.schema.robot import RobotConfig
from src.robot.core import JointPositions
from src.robot.safety._ur_kinematics import ur_link_origins_mm, ur_link_transforms_mm
from src.robot.safety.planning.environment import collision_mesh_bundle

_ARM_PARTS = {"shoulder": 1, "upper_arm": 2, "forearm": 3, "wrist_1": 4, "wrist_2": 5, "wrist_3": 6}
_HAND_PARTS = ("gripper", "lfinger", "rfinger")
_JOINTS = np.random.default_rng(7).uniform(-np.pi, np.pi, (20, 6))
_PADDING_MM = 15.0
#: A pose with the arm reaching forward over the bench.
_REACH = np.array([0.0, -1.3, 1.2, -1.5, -1.57, 0.0])


def _place(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (transform[:3, :3] @ np.asarray(points, dtype=np.float64).T).T + transform[:3, 3]


def _yawed(transforms: list[np.ndarray], yaw_deg: float) -> list[np.ndarray]:
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    rz = np.eye(4)
    rz[:2, :2] = [[c, -s], [s, c]]
    return [rz @ t for t in transforms]


def _hand(name: str = "robotiq_2f85") -> Any:
    from src.robot.safety.planning.hand import planner_hand

    text = (collision_mesh_bundle("ur5e").parent.parent / "planning" / "robot" / f"{name}_gripper_spheres.yml")
    origin = yaml.safe_load(text.read_text(encoding="utf-8"))["_provenance"]["origin"]
    gripper: dict[str, Any] = {"model": name}
    if origin == "mounting_face":
        gripper["coupling_plates"] = [{"name": "plate", "thickness_mm": 20.0}]
    return planner_hand(RobotConfig.model_validate({"vendor": "ur", "gripper": gripper}))


def _map_spheres_mm(hand: Any) -> tuple[np.ndarray, np.ndarray]:
    """The resolved map's sphere centres and radii in millimetres, on frame 6, the plates applied."""
    from src.robot.safety.planning.hand import HAND_APPROACH_IN_TOOL0

    spheres = yaml.safe_load(hand.sphere_map.read_text(encoding="utf-8"))["collision_spheres"]["tool0"]
    centres = np.asarray([s["center"] for s in spheres], dtype=np.float64) * 1000.0
    radii = np.asarray([s["radius"] for s in spheres], dtype=np.float64) * 1000.0
    if hand.origin == "mounting_face":
        centres = centres + np.asarray(HAND_APPROACH_IN_TOOL0) * hand.coupling_mm
    return centres, radii


def _tip_mm(hand: Any) -> float:
    from src.robot.safety.planning.hand import HAND_APPROACH_IN_TOOL0

    centres, radii = _map_spheres_mm(hand)
    return float(np.max(centres @ np.asarray(HAND_APPROACH_IN_TOOL0) + radii))


def _spheres(name: str = "robotiq_2f85", model: str = "ur5e") -> tuple[Any, ...]:
    from src.robot.safety.planning.self_envelope import hand_spheres

    spheres = hand_spheres(_hand(name), model)
    assert spheres is not None, (name, model)
    return spheres


class TheArmIsInsideItsEnvelopeTests(unittest.TestCase):
    def test_every_arm_vertex_is_inside_the_envelope(self) -> None:
        from src.robot.safety.planning.perceived import SelfBody
        from src.robot.safety.planning.self_envelope import arm_capsules

        for model in ("ur5e", "ur3e"):
            capsules = arm_capsules(model)
            self.assertIsNotNone(capsules, model)
            assert capsules is not None
            with np.load(collision_mesh_bundle(model)) as data:
                arrays = {part: np.asarray(data[f"{part}__v"], dtype=np.float64) for part in _ARM_PARTS}
            outside = {part: 0 for part in _ARM_PARTS}
            for joints in _JOINTS:
                frames = ur_link_transforms_mm(model, joints)
                assert frames is not None
                body = SelfBody.from_frames(frames, capsules, padding_mm=0.0)
                for part, frame in _ARM_PARTS.items():
                    outside[part] += int(np.count_nonzero(~body.contains(_place(frames[frame], arrays[part]))))
            with self.subTest(model=model):
                self.assertEqual(outside, {part: 0 for part in _ARM_PARTS})

    def test_a_point_beside_the_upper_arm_is_kept(self) -> None:
        """Bound: the padding is the only thing added around a link."""
        from src.robot.safety.planning.perceived import SelfBody
        from src.robot.safety.planning.self_envelope import arm_capsules

        capsules = arm_capsules("ur5e")
        assert capsules is not None
        (upper,) = [c for c in capsules if c.frame == 2]
        start, end = np.asarray(upper.start_mm), np.asarray(upper.end_mm)
        axis = (end - start) / np.linalg.norm(end - start)
        side = np.cross(axis, [0.0, 0.0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0.0, 0.0])
        side /= np.linalg.norm(side)
        point = (start + end) / 2 + side * (upper.radius_mm + _PADDING_MM + 30.0)
        body = SelfBody.from_frames([np.eye(4)] * 7, (upper,), padding_mm=_PADDING_MM)
        self.assertFalse(bool(body.contains(point[None, :])[0]))


class TheHandIsItsSphereMapTests(unittest.TestCase):
    def test_every_vertex_of_the_hand_is_inside_its_spheres(self) -> None:
        """As every arm vertex is inside its capsule, every hand vertex is inside its spheres before any padding.

        Measured on the map spheres as fitted (logs/step4/red_4h_hand_vertices.log): 31 of the 2F-85's 8,106 bundle
        vertices lay outside them, 1,279 of the Hand-E's 27,733 and 224 of the EGU-50's 14,332, the farthest 9.55,
        16.26 and 4.45 mm out. The Hand-E's was past the 15 mm padding too, so a camera point on its body stayed an
        obstacle touching the hand.
        """
        from src.robot.safety.planning.environment import hand_mesh_bundle
        from src.robot.safety.planning.perceived import SelfBody
        from src.robot.safety.planning.self_envelope import hand_spheres

        hands = available_grippers()
        self.assertGreaterEqual(len(hands), 3, hands)
        for name in hands:
            for model in ("ur5e", "ur3e"):
                with self.subTest(hand=name, model=model):
                    hand = _hand(name)
                    spheres = hand_spheres(hand, model)
                    assert spheres is not None
                    self.assertTrue(all(s.frame == 6 for s in spheres))
                    with np.load(hand_mesh_bundle(name)) as data:
                        vertices = np.vstack([np.asarray(data[f"{p}__v"], dtype=np.float64) for p in _HAND_PARTS])
                    if hand.origin == "mounting_face":
                        vertices = vertices + np.array([0.0, hand.coupling_mm, 0.0])
                    body = SelfBody.from_frames([np.eye(4)] * 7, spheres, padding_mm=0.0)
                    self.assertEqual(int(np.count_nonzero(~body.contains(vertices))), 0)

    def test_the_mounting_face_hand_sits_one_plate_out(self) -> None:
        hand = _hand("robotiq_hande")
        centres, _ = _map_spheres_mm(hand)
        placed = np.asarray([s.start_mm for s in _spheres("robotiq_hande")])
        np.testing.assert_allclose(placed, centres, atol=1e-9)
        self.assertEqual(hand.coupling_mm, 20.0)

    def test_an_obstacle_40_mm_past_the_fingertips_is_kept(self) -> None:
        """Bound, and the reason the hand is its spheres: one capsule reached 54 to 77 mm past the tips."""
        from src.robot.safety.planning.hand import HAND_APPROACH_IN_TOOL0
        from src.robot.safety.planning.perceived import SelfBody

        for name in available_grippers():
            with self.subTest(hand=name):
                hand = _hand(name)
                body = SelfBody.from_frames([np.eye(4)] * 7, _spheres(name), padding_mm=_PADDING_MM)
                point = np.asarray(HAND_APPROACH_IN_TOOL0) * (_tip_mm(hand) + 40.0)
                self.assertFalse(bool(body.contains(point[None, :])[0]))


class TheSimFilterUsesTheGuardYawTests(unittest.TestCase):
    """A non UR double: the Isaac arm reports no UR model, so the guard's model and yaw are the only answer."""

    def _arm(self, joints: np.ndarray) -> Any:
        from src.robot.drivers.sim.arm import IsaacRobotArm
        from src.robot.drivers.sim.config import SimRobotConfig
        from src.willy_sim.config import load_sim_config, sim_safety_preflight

        arm = IsaacRobotArm(
            SimRobotConfig(enabled=True, mock_mode=True), safety_preflight=sim_safety_preflight(load_sim_config())
        )
        arm._arm_subset = MagicMock()  # noqa: SLF001
        arm._arm_subset.get_joint_positions.return_value = joints  # noqa: SLF001
        return arm

    def test_the_sim_filter_uses_the_guard_yaw(self) -> None:
        from src.robot.safety.planning.perceived import SelfBody

        arm = self._arm(_REACH)
        envelope = arm._self_envelope()  # noqa: SLF001
        self.assertIsNotNone(envelope)
        body = SelfBody.from_frames(envelope.frames_mm, envelope.capsules, padding_mm=0.0)
        truth = ur_link_transforms_mm("ur5e", _REACH)
        assert truth is not None
        wrist = _yawed(truth, 180.0)[5][:3, 3]
        mirror = truth[5][:3, 3]
        self.assertTrue(bool(body.contains(wrist[None, :])[0]), "the wrist where the guard has it is not filtered")
        self.assertFalse(bool(body.contains(mirror[None, :])[0]), "the wrist's mirror through the base is filtered")

    def test_the_near_point_of_a_joint_move_takes_the_guard_yaw(self) -> None:
        arm = self._arm(_REACH)
        truth = ur_link_transforms_mm("ur5e", _REACH)
        assert truth is not None
        np.testing.assert_allclose(arm._flange_mm(JointPositions(_REACH)), _yawed(truth, 180.0)[6][:3, 3], atol=1e-6)  # noqa: SLF001


class APartInTheGripperIsFilteredWhileAttachedTests(unittest.TestCase):
    """Q7 A, on the UR, the one arm that attaches a payload."""

    def test_a_point_100_mm_past_the_fingertips_is_filtered_while_attached(self) -> None:
        from src.robot.drivers.ur.arm import URRobotArm
        from src.robot.safety.planning.hand import HAND_APPROACH_IN_TOOL0
        from src.robot.safety.planning.perceived import SelfBody

        config = RobotConfig.model_validate({
            "vendor": "ur",
            "ur": {"motion_planner": "curobo"},
            "gripper": {"model": "robotiq_2f85"},
            "safety": {
                "payload": {"enforce": False},
                "planning_world": {
                    "enabled": True,
                    "support_plane": {"height_mm": 0.0, "extent_mm": [1600.0, 1600.0], "thickness_mm": 50.0},
                    "payload": {"enabled": True, "length_mm": 120.0, "lateral_margin_mm": 10.0},
                },
            },
        })
        arm = URRobotArm(config)
        arm._conn = MagicMock()  # noqa: SLF001
        arm._conn.is_connected = True  # noqa: SLF001
        arm._conn.get_joint_positions.return_value = list(_REACH)  # noqa: SLF001
        planner = MagicMock()
        planner.attach_payload.return_value = True
        planner.detach_payload.return_value = True
        arm._curobo_ur = planner  # noqa: SLF001

        frames = ur_link_transforms_mm("ur5e", _REACH)
        assert frames is not None
        point = _place(frames[6], (np.asarray(HAND_APPROACH_IN_TOOL0) * (_tip_mm(_hand()) + 100.0))[None, :])

        def filtered() -> bool:
            envelope = arm._self_envelope()  # noqa: SLF001
            assert envelope is not None
            return bool(SelfBody.from_frames(envelope.frames_mm, envelope.capsules, padding_mm=_PADDING_MM)
                        .contains(point)[0])

        self.assertFalse(filtered(), "the part is filtered before anything was attached")
        arm.attach_payload(60.0)
        self.assertTrue(filtered(), "the attached part is not filtered")
        arm.detach_payload()
        self.assertFalse(filtered(), "the part is still filtered after it was detached")

    def test_the_planner_carries_the_part_the_filter_takes_out(self) -> None:
        """The planner's box and the filter's capsule are one part: along the approach, from the fingertips.

        Before, the planner hung a box 120 mm along tool0 +Z from the flange (the attach pose in curobo_motion.py), 90
        degrees off the hand Step 4g.0 measured and inside the wrist, while the filter took the part out past the tips.
        """
        from src.robot.safety.planning.self_envelope import carried_part_box, payload_capsule

        spheres = _spheres()
        capsule = payload_capsule(_hand(), spheres, length_mm=120.0, lateral_margin_mm=10.0)
        dims, centre = carried_part_box(_hand(), spheres, grip_width_mm=60.0, length_mm=120.0, lateral_margin_mm=10.0)
        start, end = np.asarray(capsule.start_mm), np.asarray(capsule.end_mm)
        np.testing.assert_allclose(centre, (start + end) / 2.0, atol=1e-9)
        axis = (end - start) / np.linalg.norm(end - start)
        self.assertAlmostEqual(float(np.abs(np.asarray(dims)) @ np.abs(axis)), 120.0, places=9)
        self.assertEqual(sorted(dims), [70.0, 70.0, 120.0])


class ACameraThatSeesTheArmRegistersNoBoxAroundItTests(unittest.TestCase):
    """The ur5e bundle arm and hand rendered into a synthetic overhead depth frame, beside one real block."""

    _FX = 120.0
    _C = 160.0
    _SHAPE = (320, 320)
    _HEIGHT = 1500.0

    def _frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
        camera_to_base = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, self._HEIGHT], [0.0, 0.0, 0.0, 1.0]]
        )
        intrinsics = np.array([[self._FX, 0.0, self._C], [0.0, self._FX, self._C], [0.0, 0.0, 1.0]])
        depth = np.full(self._SHAPE, self._HEIGHT, dtype=np.float64)
        frames = ur_link_transforms_mm("ur5e", _REACH)
        assert frames is not None
        with np.load(collision_mesh_bundle("ur5e")) as data:
            points = [_place(frames[f], np.asarray(data[f"{p}__v"])) for p, f in _ARM_PARTS.items()]
            points += [_place(frames[6], np.asarray(data[f"{p}__v"])) for p in _HAND_PARTS]
        cam = _place(np.linalg.inv(camera_to_base), np.vstack(points))
        u = np.round(self._FX * cam[:, 0] / cam[:, 2] + self._C).astype(int)
        v = np.round(self._FX * cam[:, 1] / cam[:, 2] + self._C).astype(int)
        seen = (cam[:, 2] > 0) & (u >= 0) & (u < self._SHAPE[1]) & (v >= 0) & (v < self._SHAPE[0])
        for row, col, z in zip(v[seen], u[seen], cam[seen, 2]):
            depth[row, col] = min(depth[row, col], z)
        # One block on the bench, well clear of the arm: 60 x 60 mm, 80 mm tall, at (0, 700).
        rows, cols = np.mgrid[0 : self._SHAPE[0], 0 : self._SHAPE[1]]
        top = self._HEIGHT - 80.0
        x = (cols - self._C) * top / self._FX
        y = -(rows - self._C) * top / self._FX
        block = (np.abs(x - 0.0) <= 30.0) & (np.abs(y - 700.0) <= 30.0)
        depth[block] = np.minimum(depth[block], top)
        return depth, intrinsics, camera_to_base, frames

    def _limits(self) -> Any:
        from src.robot.safety.planning.perceived import WorldBuildLimits

        return WorldBuildLimits(x_mm=(-1500.0, 1500.0), y_mm=(-1500.0, 1500.0), z_mm=(-50.0, 1400.0),
                                support_plane_top_mm=0.0)

    def test_the_polyline_filter_leaves_boxes_around_the_arm(self) -> None:
        """The control: the frame really shows an arm the old filter does not take out."""
        from src.robot.safety.planning.perceived import (
            DepthView,
            SelfBody,
            WorldBuildTuning,
            build_perceived_boxes,
        )

        depth, intrinsics, camera_to_base, _ = self._frame()
        origins = ur_link_origins_mm("ur5e", _REACH)
        assert origins is not None
        world = build_perceived_boxes(
            views=(DepthView(surface_depth_mm=depth, intrinsics=intrinsics, camera_to_base=camera_to_base),),
            limits=self._limits(), tuning=WorldBuildTuning(max_boxes=8, margin_mm=_PADDING_MM),
            self_body=SelfBody.from_polyline(
                [[float(v) for v in origin] for origin in origins], radius_mm=90.0, tool_radius_mm=150.0,
            ),
        )
        self.assertGreater(len(world.boxes), 1)

    def test_a_camera_that_sees_the_arm_registers_only_the_block(self) -> None:
        from src.robot.safety.planning.live_world import CameraView, DepthSnapshot, LivePlannerWorld
        from src.robot.safety.planning.perceived import DropReason, SelfEnvelope, WorldBuildTuning
        from src.robot.safety.planning.self_envelope import arm_capsules

        depth, intrinsics, camera_to_base, frames = self._frame()

        class _Camera:
            def grab_surface_depth(self) -> DepthSnapshot:
                return DepthSnapshot(depth_mm=depth, intrinsics=intrinsics, timestamp=100.0)

        capsules = arm_capsules("ur5e")
        assert capsules is not None
        world = LivePlannerWorld(
            cameras=(CameraView(name="overhead", depth_source=_Camera(), camera_to_base=camera_to_base),),
            declared=(), limits=self._limits(), tuning=WorldBuildTuning(max_boxes=8, margin_mm=_PADDING_MM),
            max_age_ms=500.0,
        )
        snapshot = world.world_for(
            self_envelope=SelfEnvelope(frames_mm=tuple(frames), capsules=capsules + _spheres()), now=100.1,
        )
        self.assertTrue(snapshot.usable, snapshot.reason)
        assert snapshot.perceived is not None
        self.assertGreater(snapshot.perceived.dropped_points[DropReason.SELF], 0)
        self.assertEqual(snapshot.perceived_count, 1, [b.center_mm for b in snapshot.perceived.boxes])
        (box,) = snapshot.perceived.boxes
        self.assertLess(abs(box.center_mm[1] - 700.0), 60.0)


class TheTwoRadiusKeysAreRefusedTests(unittest.TestCase):
    def test_the_schema_refuses_both(self) -> None:
        from src.config.schema.robot.safety_schema import PerceivedWorldConfig

        for key in ("self_radius_mm", "tool_radius_mm"):
            with self.subTest(key=key), self.assertRaises(ValidationError):
                PerceivedWorldConfig.model_validate({key: 90.0})

    def test_the_refusal_names_the_padding(self) -> None:
        from src.config.schema._removed import REMOVED_KEYS

        for key in ("self_radius_mm", "tool_radius_mm"):
            with self.subTest(key=key):
                self.assertIn("perceived.margin_mm", REMOVED_KEYS[f"robot.safety.planning_world.perceived.{key}"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
