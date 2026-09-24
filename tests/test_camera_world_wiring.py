"""Every calibrated enabled rig feeds the live planner world of a cell that enables it, primary first.

`CameraWorldPlan` reads config and opens nothing; `CameraWorldWiring` takes open owners and builds the world.
Both take rigs, owners and calibrations by duck typing, because the world is to be handed to `Robot`, whose
import guard forbids the camera package, and the guard is measured here. The cell half,
`_wire_live_planner_world`, runs over camera doubles, and over real UR and sim arms where a motion has to be
refused or stamped.

A documented consequence of an owner decision is pinned too: a world with two cameras stops every planned
motion when one of them cannot answer.
"""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from src.config.schema.robot import RobotConfig
from src.camera.orchestration.camera import CameraBusy
from src.geometry import Frame, Pose, Transform
from src.robot.core import JointPositions
from src.robot.core.camera_world import CameraWorldUse
from src.robot.core.errors import CameraWorldUnavailable
from src.robot.drivers.ur.curobo_motion import UR_ARM_JOINT_NAMES
from src.robot.execution.autonomous_grasp import cells
from src.robot.execution.autonomous_grasp.cells import CellBuildRefused
from src.robot.execution.autonomous_grasp.service import AutonomousGraspService
from src.robot.execution.camera_world_wiring import CameraWorldPlan, CameraWorldWiring, OpenedCameras
from src.robot.execution.lifecycle import release_perception
from src.robot.safety.planning.depth_source import RigDepthSource
from src.robot.safety.planning.live_world import refresh_planner_world
from src.robot.safety.planning.perceived import LinkCapsule, SelfEnvelope
from tests.test_every_verb_meets_the_camera_world import _THERE, _Client, _mesh_backend_available, _pose, _sim, _ur
from tests.test_robot import _ROBOT_MUST_NOT_LOAD
from tests.test_robot_parts import _loaded_after

_MODULE = "src.robot.execution.camera_world_wiring"
_K = np.array([[500.0, 0.0, 20.0], [0.0, 500.0, 20.0], [0.0, 0.0, 1.0]])
#: A fixed camera a metre above the bench, looking straight down.
_CAMERA_TO_BASE = Transform.from_matrix(
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1000.0], [0.0, 0.0, 0.0, 1.0]]),
    from_frame=Frame.CAMERA, to_frame=Frame.BASE,
)
#: A camera on the tool, looking along the approach.
_CAMERA_TO_TOOL = Transform.from_matrix(np.diag([1.0, -1.0, -1.0, 1.0]), from_frame=Frame.CAMERA, to_frame=Frame.TOOL)
_TCP = Pose(position_mm=np.array([0.0, 0.0, 1000.0]), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), frame=Frame.BASE)
_SELF = SelfEnvelope(
    frames_mm=(np.eye(4),),
    capsules=(LinkCapsule(frame=0, start_mm=(0.0, 0.0, 0.0), end_mm=(0.0, 0.0, 300.0), radius_mm=135.0),),
)


def _cell(*, world: bool = True, perceived: bool = True, planner: str = "curobo",
          fused_off: tuple[str, ...] = ()) -> RobotConfig:
    """A UR cell that enables the live planner world, or does not. Boxes only, and a generous age."""
    safety: dict[str, object] = {"payload": {"enforce": False}}
    if world:
        safety["planning_world"] = {
            "enabled": True,
            "support_plane": {"height_mm": 0.0, "extent_mm": [1600.0, 1600.0], "thickness_mm": 50.0},
            "perceived": {"enabled": perceived, "voxel_field_mm": 0.0, "max_age_ms": 5000.0},
        }
    return RobotConfig.model_validate({
        "vendor": "ur", "ur": {"motion_planner": planner}, "safety": safety,
        "grasping": {"fusion": {"cameras": {cam_id: {"enabled": False} for cam_id in fused_off}}},
    })


def _rig(rig_id: str, *, enabled: bool = True, source: str = "rgbd", calibrated: bool = True,
         wrist: bool = False) -> SimpleNamespace:
    extrinsics = None
    if calibrated:
        extrinsics = SimpleNamespace(mounting_mode="eye_in_hand" if wrist else "eye_to_hand",
                                     artifact_path=f"{rig_id}.json")
    return SimpleNamespace(rig_id=rig_id, enabled=enabled, source=source, extrinsics=extrinsics)


def _app(*rigs: SimpleNamespace) -> SimpleNamespace:
    """A camera section whose first rig is the primary."""
    return SimpleNamespace(camera=SimpleNamespace(cameras=SimpleNamespace(rigs=list(rigs),
                                                                          primary_rig_id=rigs[0].rig_id)))


class _Handle:
    """A rig handle: an empty bench a metre down on every grab, or a frame of zeros, or a stereo pair."""

    def __init__(self, rig_id: str, *, answer: str, captured_at_s: float | None) -> None:
        self.rig_id = rig_id
        self.answer = answer
        self.captured_at_s = captured_at_s

    def grab(self) -> SimpleNamespace:
        if self.answer == "stereo":
            return SimpleNamespace(left=np.zeros((4, 4, 3)), right=np.zeros((4, 4, 3)))
        depth = np.zeros((40, 40)) if self.answer == "blind" else np.full((40, 40), 1000.0)
        return SimpleNamespace(depth=depth, captured_at_s=time.time() if self.captured_at_s is None else self.captured_at_s)

    def get_intrinsics(self) -> np.ndarray:
        return _K


class _Owner:
    """A camera owner, recording what was asked of it."""

    def __init__(self, rig_id: str, *, wrist: bool = False, answer: str = "bench",
                 captured_at_s: float | None = None, busy: bool = False) -> None:
        self.rig_id = rig_id
        self.wrist = wrist
        self.busy = busy
        self.opened = 0
        self.released = 0
        self._handle = _Handle(rig_id, answer=answer, captured_at_s=captured_at_s)

    def open(self) -> None:
        if self.busy:
            raise CameraBusy(self.rig_id, "another build", ("realsense", "0001"))
        self.opened += 1

    def handle(self) -> _Handle:
        return self._handle

    def calibration(self) -> SimpleNamespace:
        if self.wrist:
            return SimpleNamespace(mounting_mode="eye_in_hand", camera_to_tool=lambda: _CAMERA_TO_TOOL,
                                   shutter_motion_tolerance_mm=1.0, shutter_motion_tolerance_deg=0.5)
        return SimpleNamespace(mounting_mode="eye_to_hand", camera_to_base=lambda: _CAMERA_TO_BASE)

    def release(self) -> None:
        self.released += 1


def _camera_class(owners: dict[str, _Owner]) -> type:
    """The camera noun, answering each rig id with its owner double."""

    class _Camera:
        @classmethod
        def from_config(cls, camera_cfg: object, *, rig_id: str) -> _Owner:  # noqa: ARG003
            return owners[rig_id]

    return _Camera


def _wire(cfg: RobotConfig, app: SimpleNamespace, owners: dict[str, _Owner], *, arm: object = None,
          extras: tuple[str, ...] = ()) -> tuple[SimpleNamespace, object]:
    """`_wire_live_planner_world` on a built service whose primary, and any fused extras, are open already."""
    if arm is None:
        arm = SimpleNamespace(set_live_planner_world=MagicMock(), get_tcp_pose=lambda: _TCP)
    sources = {rig_id: SimpleNamespace(streamer=SimpleNamespace(camera=owners[rig_id])) for rig_id in extras}
    orchestrator = SimpleNamespace(arm=arm, frame_resolver=None, planner_world_cameras=None,
                                   multi_camera_perception=SimpleNamespace(sources=sources) if extras else None)
    service = SimpleNamespace(runtime=SimpleNamespace(orchestrator=orchestrator))
    perception = SimpleNamespace(streamer=SimpleNamespace(camera=owners[app.camera.cameras.primary_rig_id]))
    with patch("src.camera.orchestration.camera.Camera", _camera_class(owners)):
        cells._wire_live_planner_world(cfg, service, perception, app_cfg=app)
    return service, arm


class TheWorldPlanTableTests(unittest.TestCase):

    def test_the_world_plan_table(self) -> None:
        rows = (
            ("a disabled block", _cell(world=False), (_rig("overhead"),), (), "safety.planning_world.enabled"),
            ("perceived switched off", _cell(perceived=False), (_rig("overhead"),), (),
             "safety.planning_world.perceived.enabled"),
            ("one calibrated RGB-D primary", _cell(), (_rig("overhead"),), ("overhead",), ""),
            ("two calibrated rigs, the primary listed second", _cell(), (_rig("side"), _rig("overhead")),
             ("overhead", "side"), ""),
            ("a disabled calibrated rig", _cell(), (_rig("overhead"), _rig("side", enabled=False)), ("overhead",), ""),
            ("an uncalibrated enabled rig", _cell(), (_rig("overhead"), _rig("side", calibrated=False)),
             ("overhead",), ""),
            ("a calibrated rig fusion switches off", _cell(fused_off=("side",)), (_rig("overhead"), _rig("side")),
             ("overhead", "side"), ""),
            ("a calibrated wrist rig", _cell(), (_rig("overhead"), _rig("wrist", wrist=True)),
             ("overhead", "wrist"), ""),
            ("an uncalibrated primary beside a calibrated rig", _cell(),
             (_rig("overhead", calibrated=False), _rig("side")), (), "masks"),
        )
        for label, cfg, rigs, ids, reason in rows:
            with self.subTest(label):
                plan = CameraWorldPlan.from_config(cfg, rigs, primary_rig_id="overhead")
                self.assertEqual(plan.rig_ids, ids, plan.render())
                self.assertIn(reason, plan.reason)

    def test_each_rig_left_out_says_why(self) -> None:
        plan = CameraWorldPlan.from_config(
            _cell(), (_rig("overhead"), _rig("off", enabled=False), _rig("bare", calibrated=False),
                      _rig("pair", source="webcam_pair", calibrated=False)),
            primary_rig_id="overhead",
        )
        why = dict(plan.left_out)
        self.assertEqual(why["off"], "enabled: false")
        self.assertIn("camera.cameras.rigs['bare'].extrinsics", why["bare"])
        self.assertIn("no depth of its own", why["pair"])
        self.assertTrue(plan.asked)

    def test_a_cell_that_did_not_ask_is_not_counted_as_asking(self) -> None:
        self.assertFalse(CameraWorldPlan.from_config(_cell(world=False), (_rig("overhead"),),
                                                     primary_rig_id="overhead").asked)


class TheWiringTableTests(unittest.TestCase):

    @staticmethod
    def _plan(*rigs: SimpleNamespace, cfg: RobotConfig | None = None) -> CameraWorldPlan:
        return CameraWorldPlan.from_config(cfg or _cell(), rigs, primary_rig_id=rigs[0].rig_id)

    def test_one_fixed_rig_gives_one_view_placed_by_its_camera_to_base(self) -> None:
        wiring = CameraWorldWiring.from_cameras(_cell(), plan=self._plan(_rig("overhead")),
                                                cameras={"overhead": _Owner("overhead")})
        assert wiring.world is not None
        (view,) = wiring.world.cameras
        self.assertEqual(view.name, "overhead")
        np.testing.assert_array_equal(view.camera_to_base, _CAMERA_TO_BASE.to_matrix())
        self.assertIsNone(view.camera_to_tool)
        self.assertIsInstance(view.depth_source, RigDepthSource)

    def test_two_fixed_rigs_give_both_in_plan_order(self) -> None:
        wiring = CameraWorldWiring.from_cameras(
            _cell(), plan=self._plan(_rig("overhead"), _rig("side")),
            cameras={"side": _Owner("side"), "overhead": _Owner("overhead")},
        )
        assert wiring.world is not None
        self.assertEqual(tuple(view.name for view in wiring.world.cameras), ("overhead", "side"))
        self.assertEqual(wiring.cameras, ("overhead", "side"))

    def test_a_wrist_rig_gives_a_camera_to_tool_view_whose_source_reads_the_arm(self) -> None:
        reads: list[int] = []

        def reader() -> Pose:
            reads.append(1)
            return _TCP

        wiring = CameraWorldWiring.from_cameras(
            _cell(), plan=self._plan(_rig("overhead"), _rig("wrist", wrist=True)),
            cameras={"overhead": _Owner("overhead"), "wrist": _Owner("wrist", wrist=True)}, tool_pose=reader,
        )
        assert wiring.world is not None
        wrist = wiring.world.cameras[1]
        np.testing.assert_array_equal(wrist.camera_to_tool, _CAMERA_TO_TOOL.to_matrix())
        reading = wrist.depth_source.grab_surface_depth()
        assert reading is not None
        np.testing.assert_array_equal(reading.tool_to_base_mm, _TCP.to_matrix())
        self.assertEqual(len(reads), 2, "a wrist frame reads the tool pose before and after its grab")

    def test_a_wrist_rig_without_the_arms_reader_gives_no_world(self) -> None:
        wiring = CameraWorldWiring.from_cameras(
            _cell(), plan=self._plan(_rig("overhead"), _rig("wrist", wrist=True)),
            cameras={"overhead": _Owner("overhead"), "wrist": _Owner("wrist", wrist=True)},
        )
        self.assertIsNone(wiring.world)
        self.assertIn("'wrist'", wiring.reason)
        self.assertIn("TCP", wiring.reason)

    def test_a_rig_answering_a_stereo_pair_gives_no_world_naming_it(self) -> None:
        wiring = CameraWorldWiring.from_cameras(_cell(), plan=self._plan(_rig("overhead")),
                                                cameras={"overhead": _Owner("overhead", answer="stereo")})
        self.assertIsNone(wiring.world)
        self.assertIn("stereo", wiring.render())
        self.assertIn("'overhead'", wiring.render())

    def test_an_ik_planner_gets_a_world_whose_render_says_nothing_reads_it(self) -> None:
        cfg = _cell(planner="ik")
        wiring = CameraWorldWiring.from_cameras(cfg, plan=self._plan(_rig("overhead"), cfg=cfg),
                                                cameras={"overhead": _Owner("overhead")})
        self.assertIsNotNone(wiring.world)
        self.assertIn("'ik'", wiring.render())
        self.assertNotIn("refreshed before every plan", wiring.render())

    def test_a_curobo_planner_is_told_the_world_is_refreshed(self) -> None:
        """The control for the row above: the sentence is true here."""
        wiring = CameraWorldWiring.from_cameras(_cell(), plan=self._plan(_rig("overhead")),
                                                cameras={"overhead": _Owner("overhead")})
        self.assertIn("refreshed before every plan", wiring.render())

    def test_opened_cameras_give_every_camera_back(self) -> None:
        first, second = _Owner("first"), _Owner("second")
        OpenedCameras((first, second)).close()
        self.assertEqual((first.released, second.released), (1, 1))


class TheWiringLoadsNothingARobotMayNotTests(unittest.TestCase):
    """The world is to be handed to `Robot`, so the wiring loads nothing its guard forbids, and no calibration."""

    _FORBIDDEN = _ROBOT_MUST_NOT_LOAD + ("src.calibration",)

    def test_the_wiring_module_loads_nothing_a_robot_may_not(self) -> None:
        self.assertEqual(_loaded_after(f"import {_MODULE}", self._FORBIDDEN), [])

    def test_the_probe_sees_the_pick_service_where_it_is(self) -> None:
        """The self-failing control: the cell builders load the pick service at module top."""
        self.assertIn("src.robot.execution.autonomous_grasp",
                      _loaded_after("import src.robot.execution.autonomous_grasp.cells", self._FORBIDDEN))


class ARealCellWiresEveryCalibratedRigTests(unittest.TestCase):

    def test_a_real_cell_wires_every_calibrated_enabled_rig(self) -> None:
        owners = {"overhead": _Owner("overhead"), "side": _Owner("side")}
        _, arm = _wire(_cell(), _app(_rig("overhead"), _rig("side")), owners)
        (world,), _ = arm.set_live_planner_world.call_args  # type: ignore[attr-defined]
        self.assertEqual(tuple(view.name for view in world.cameras), ("overhead", "side"))
        self.assertEqual((owners["overhead"].opened, owners["side"].opened), (0, 1),
                         "the primary is open already, and only the world-only rig is opened here")

    def test_a_calibrated_rig_outside_fusion_is_opened_for_the_world_and_released(self) -> None:
        owners = {"overhead": _Owner("overhead"), "side": _Owner("side")}
        service, _ = _wire(_cell(fused_off=("side",)), _app(_rig("overhead"), _rig("side")), owners)
        self.assertIsInstance(service.runtime.orchestrator.planner_world_cameras, OpenedCameras)

        release_perception(service)

        self.assertEqual(owners["side"].released, 1)

    def test_a_fused_camera_is_taken_as_it_is_open_rather_than_opened_again(self) -> None:
        owners = {"overhead": _Owner("overhead"), "side": _Owner("side")}
        service, arm = _wire(_cell(), _app(_rig("overhead"), _rig("side")), owners, extras=("side",))
        arm.set_live_planner_world.assert_called_once()  # type: ignore[attr-defined]
        self.assertEqual(owners["side"].opened, 0)
        self.assertIsNone(service.runtime.orchestrator.planner_world_cameras)

    def test_a_disabled_world_opens_no_extra_camera_and_logs_nothing(self) -> None:
        """The control for byte identity on every shipped profile, where the block is off."""
        owners = {"overhead": _Owner("overhead"), "side": _Owner("side")}
        with self.assertNoLogs(cells.__name__, level="DEBUG"):
            _, arm = _wire(_cell(world=False), _app(_rig("overhead"), _rig("side")), owners)
        arm.set_live_planner_world.assert_not_called()  # type: ignore[attr-defined]
        self.assertEqual(owners["side"].opened, 0)


class OneCameraThatCannotAnswerStopsTheWorldTests(unittest.TestCase):
    """Documented behaviour, from an owner decision: every camera in a world has to answer."""

    def test_one_blind_camera_of_two_refuses_every_refresh_naming_it(self) -> None:
        cfg = _cell()
        plan = CameraWorldPlan.from_config(cfg, (_rig("overhead"), _rig("side")), primary_rig_id="overhead")
        wiring = CameraWorldWiring.from_cameras(
            cfg, plan=plan, cameras={"overhead": _Owner("overhead"), "side": _Owner("side", answer="blind")})
        assert wiring.world is not None

        with self.assertRaises(CameraWorldUnavailable) as raised:
            refresh_planner_world(source=wiring.world, client=_Client(), self_envelope=_SELF)

        self.assertEqual(raised.exception.camera, "side")
        self.assertEqual(raised.exception.attempts, 1 + wiring.world.fresh_frame_attempts)

    def test_a_blind_second_rig_stops_every_planned_motion(self) -> None:
        arms = {"sim": lambda: _sim(client=_Client())}
        if _mesh_backend_available():
            arms["ur"] = lambda: _ur(None, _Client(joint_names=UR_ARM_JOINT_NAMES), [])  # type: ignore[arg-type]
        verbs = {
            "move": lambda arm: arm.move(_pose()),
            "move_to_joints": lambda arm: arm.move_to_joints(JointPositions(_THERE)),
        }
        for arm_name, build in arms.items():
            for verb, motion in verbs.items():
                with self.subTest(arm=arm_name, verb=verb):
                    arm = build()
                    owners = {"overhead": _Owner("overhead"), "side": _Owner("side", answer="blind")}
                    _wire(_cell(), _app(_rig("overhead"), _rig("side")), owners, arm=arm)
                    with self.assertRaises(CameraWorldUnavailable) as raised:
                        motion(arm)
                    self.assertEqual(raised.exception.camera, "side")


class APlannedStampNamesEveryWorldCameraTests(unittest.TestCase):

    def test_a_planned_stamp_names_every_world_camera(self) -> None:
        rows = {("sim", "move"): (lambda: _sim(client=_Client()), lambda arm: arm.move(_pose(), linear=True)),
                ("sim", "move_to_joints"): (lambda: _sim(client=_Client()),
                                            lambda arm: arm.move_to_joints(JointPositions(_THERE)))}
        if _mesh_backend_available():
            rows[("ur", "move")] = (lambda: _ur(None, _Client(joint_names=UR_ARM_JOINT_NAMES), []),  # type: ignore[arg-type]
                                    lambda arm: arm.move(_pose()))
            rows[("ur", "move_to_joints")] = (lambda: _ur(None, _Client(joint_names=UR_ARM_JOINT_NAMES), []),  # type: ignore[arg-type]
                                              lambda arm: arm.move_to_joints(JointPositions(_THERE)))
        for (arm_name, verb), (build, motion) in rows.items():
            with self.subTest(arm=arm_name, verb=verb):
                # Per row: a UR move judges its straight line on the exact meshes first, 2.8 s here for 648 samples,
                # and frames stamped once for all four rows went stale under the 5 s the cell allows before the last.
                now = time.time()
                arm = build()
                owners = {"overhead": _Owner("overhead", captured_at_s=now - 2.0),
                          "side": _Owner("side", captured_at_s=now - 1.0)}
                _wire(_cell(), _app(_rig("overhead"), _rig("side")), owners, arm=arm)
                stamp = motion(arm).camera_world
                self.assertIs(stamp.use, CameraWorldUse.PLANNED, stamp.render())
                self.assertEqual(stamp.cameras, ("overhead", "side"))
                self.assertEqual(stamp.captured_at_s, now - 2.0)


class AnOfferNamesItsCameraTests(unittest.TestCase):

    def test_an_offer_names_its_camera_and_an_unnamed_offer_with_masks_is_refused(self) -> None:
        from src.robot.safety.planning.perceived import PerceptionGeometryError

        cfg = _cell()
        plan = CameraWorldPlan.from_config(cfg, (_rig("side"), _rig("overhead")), primary_rig_id="overhead")
        wiring = CameraWorldWiring.from_cameras(cfg, plan=plan,
                                                cameras={"overhead": _Owner("overhead"), "side": _Owner("side")})
        assert wiring.world is not None

        wiring.world.offer_segmentation(camera="side", labelled_masks=[("red cube", np.ones((40, 40), dtype=bool))])
        self.assertEqual(set(wiring.world._labels), {"side"})  # noqa: SLF001
        with self.assertRaises(PerceptionGeometryError):
            wiring.world.offer_segmentation(labelled_masks=[("red cube", np.ones((40, 40), dtype=bool))])

    def test_the_primary_first_refusal_names_the_offer_rule(self) -> None:
        plan = CameraWorldPlan.from_config(_cell(), (_rig("overhead", calibrated=False), _rig("side")),
                                           primary_rig_id="overhead")
        self.assertIn("offers its masks under the primary's name", plan.reason)
        self.assertIn("refuses", plan.reason)


class AWorldCameraThatCannotOpenRefusesTheBuildTests(unittest.TestCase):

    def test_a_world_camera_that_cannot_open_refuses_the_build_and_releases_every_camera(self) -> None:
        owners = {"overhead": _Owner("overhead"), "fused": _Owner("fused"), "side": _Owner("side", busy=True)}
        perception = SimpleNamespace(streamer=SimpleNamespace(camera=owners["overhead"]), close=MagicMock())
        multi = SimpleNamespace(sources={"fused": SimpleNamespace(streamer=SimpleNamespace(camera=owners["fused"]))},
                                close=MagicMock())
        arm = SimpleNamespace(set_live_planner_world=MagicMock(), get_tcp_pose=lambda: _TCP)
        orchestrator = SimpleNamespace(arm=arm, frame_resolver=None, multi_camera_perception=multi,
                                       planner_world_cameras=None)
        service = SimpleNamespace(runtime=SimpleNamespace(orchestrator=orchestrator))

        with patch.object(cells, "build_real_components", return_value=(None, perception, None, multi, None)), \
             patch.object(AutonomousGraspService, "from_robot_config", return_value=service), \
             patch("src.camera.orchestration.camera.Camera", _camera_class(owners)), \
             self.assertRaises(CellBuildRefused) as refused:
            cells.build_real_cell(_cell(), app_config=_app(_rig("overhead"), _rig("fused"), _rig("side")))  # type: ignore[arg-type]

        self.assertIn("'side'", str(refused.exception))
        perception.close.assert_called_once()
        multi.close.assert_called_once()
        arm.set_live_planner_world.assert_not_called()


class ARebuiltCellOpensEveryWorldCameraAgainTests(unittest.TestCase):

    def test_a_rebuilt_cell_opens_every_world_camera_again(self) -> None:
        held: list[str] = []

        class _Exclusive(_Owner):
            """One physical device: a second open while it is held is refused, as librealsense refuses it."""

            def open(self) -> None:
                if held:
                    raise CameraBusy(self.rig_id, held[0], ("realsense", "0001"))
                held.append(self.rig_id)
                self.opened += 1

            def release(self) -> None:
                if self.rig_id in held:
                    held.remove(self.rig_id)
                self.released += 1

        for build in range(2):
            with self.subTest(build=build):
                owners = {"overhead": _Owner("overhead"), "side": _Exclusive("side")}
                service, arm = _wire(_cell(), _app(_rig("overhead"), _rig("side")), owners)
                arm.set_live_planner_world.assert_called_once()  # type: ignore[attr-defined]
                release_perception(service)
        self.assertEqual(held, [])


if __name__ == "__main__":
    unittest.main()


class _ComponentsReached(RuntimeError):
    """Raised by the patched components builder: the build got past every refusal and would open a camera."""


class ACuroboCellWhoseCamerasGiveNoWorldDoesNotBuildTests(unittest.TestCase):
    """By owner decision a calibrated rig makes the world mandatory on a cuRobo cell, refused before a camera opens."""

    def _build(self, cfg: RobotConfig, app: SimpleNamespace) -> None:
        with patch.object(cells, "build_real_components", side_effect=_ComponentsReached):
            cells.build_real_cell(cfg, app_config=app)  # type: ignore[arg-type]

    def test_a_curobo_cell_whose_calibrated_rig_yields_no_world_is_refused_before_a_camera_opens(self) -> None:
        with self.assertRaises(CellBuildRefused) as refused:
            self._build(_cell(world=False), _app(_rig("overhead")))
        self.assertIn("safety.planning_world.enabled", str(refused.exception))
        self.assertIn("'overhead'", str(refused.exception))

        with self.assertRaises(CellBuildRefused) as primary:
            self._build(_cell(), _app(_rig("overhead", calibrated=False), _rig("side")))
        self.assertIn("primary rig 'overhead'", str(primary.exception))

    def test_the_controls_an_ik_cell_and_an_uncalibrated_cell_build(self) -> None:
        with self.assertRaises(_ComponentsReached):
            self._build(_cell(world=False, planner="ik"), _app(_rig("overhead")))
        # Uncalibrated: the cell builds with no world, and every planned motion needs a decline.
        with self.assertRaises(_ComponentsReached):
            self._build(_cell(world=False), _app(_rig("overhead", calibrated=False)))

    def test_a_stereo_answer_on_a_curobo_cell_refuses_the_build(self) -> None:
        owners = {"overhead": _Owner("overhead", answer="stereo"), "side": _Owner("side")}
        with self.assertRaises(CellBuildRefused) as refused:
            _wire(_cell(), _app(_rig("overhead"), _rig("side")), owners)
        self.assertIn("built no live world", str(refused.exception))
        self.assertEqual((1, 1), (owners["side"].opened, owners["side"].released),
                         "the camera opened for the world was not given back")

    def test_the_control_a_stereo_answer_on_an_ik_cell_warns_and_builds(self) -> None:
        owners = {"overhead": _Owner("overhead", answer="stereo")}
        with self.assertLogs(cells.__name__, level="WARNING"):
            _, arm = _wire(_cell(planner="ik"), _app(_rig("overhead")), owners)
        arm.set_live_planner_world.assert_not_called()  # type: ignore[attr-defined]
