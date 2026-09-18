"""The exact mesh guard, the self filter and the UR driver carry a wrist camera's body.

As the owner decided, the planner, the exact guard and the self filter all know the body. The guard holds the camera
as a part on frame 6 and, unlike everything else on the flange, checks it against wrist_2, one frame in, because a
housing sticks out sideways. Bodies are handed in before anything is built: a camera handed to a guard that already
built its backend, or to an arm whose planner already started, would be one they say they hold and do not. A polyscope
cell compares the recorded flange to TCP with the frame it derives at connect, within the rig's record tolerances.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from src.config.cameras import load_camera
from src.calibration.serialization import FlangeToTcp
from src.robot.core import RobotConnectionError
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.safety._fcl_self_collision import MeshSelfCollisionBackend
from src.robot.safety.planning.body_link import WristBody
from tests._wrist_body import camera_to_tool, declared_frame, wrist_body, wrist_cell


def _shifted(mm: float) -> np.ndarray:
    frame = declared_frame().copy()
    frame[0, 3] += mm
    return frame


def _body(*, source: str = "willy", record: np.ndarray | None = None, tolerance: tuple = (1.0, 0.5)) -> WristBody:
    return WristBody.from_parts(
        rig_id="wrist", spec=load_camera("realsense_d435"), bracket=None, margin_mm=5.0,
        camera_to_tool=camera_to_tool(),
        flange_to_tcp=FlangeToTcp.from_matrix(source, declared_frame() if record is None else record),
        record_tolerance_mm=tolerance[0], record_tolerance_deg=tolerance[1])


class _Adapter:
    """An engine that builds nothing and records which pairs were asked for."""

    kind = "stub"

    def __init__(self) -> None:
        self.asked: set[frozenset[str]] = set()

    def build_object(self, verts: object, faces: object) -> list:
        return []

    def set_transform(self, obj: object, rotation: object, translation: object) -> None:
        return None

    def distance(self, a: list, b: list) -> float:
        self.asked.add(frozenset((a[0], b[0])))
        return 1e9


def _asked(wrist: bool) -> set[frozenset[str]]:
    cube = (np.eye(3), np.zeros((1, 3), dtype=np.int64))
    frames = {"wrist_1_link": 4, "wrist_2_link": 5, "wrist_3_link": 6, "hand": 6, "wrist_camera_wrist": 6}
    adapter = _Adapter()
    backend = MeshSelfCollisionBackend(adapter, {name: (*cube, frame) for name, frame in frames.items()},  # type: ignore[arg-type]
                                       wrist_parts=frozenset({"wrist_camera_wrist"}) if wrist else frozenset())
    for name, model in backend._models.items():
        model.append(name)
    backend.evaluate([np.eye(4)] * 7, 0.0, (), 10.0)
    return adapter.asked


class ThePairRuleTests(unittest.TestCase):
    def test_the_camera_is_checked_against_wrist_2_and_skipped_on_its_own_frame(self) -> None:
        asked = _asked(wrist=True)
        self.assertIn(frozenset(("wrist_2_link", "wrist_camera_wrist")), asked)
        self.assertIn(frozenset(("wrist_1_link", "wrist_camera_wrist")), asked)
        for skipped in (("wrist_3_link", "wrist_camera_wrist"), ("hand", "wrist_camera_wrist"),
                        ("wrist_2_link", "hand"), ("wrist_2_link", "wrist_3_link")):
            self.assertNotIn(frozenset(skipped), asked)

    def test_without_a_camera_the_rule_is_what_it_was(self) -> None:
        """The control: the same parts with nothing marked as a camera skip every pair within one frame."""
        asked = _asked(wrist=False)
        self.assertNotIn(frozenset(("wrist_2_link", "wrist_camera_wrist")), asked)
        self.assertEqual(asked, {frozenset(("wrist_1_link", name)) for name in ("wrist_3_link", "hand",
                                                                                "wrist_camera_wrist")})

    def test_a_camera_part_that_names_a_held_part_is_refused_before_an_engine_is_built(self) -> None:
        from src.robot.safety import _fcl_self_collision as fcl

        held = {"hand": (np.zeros((3, 3)), np.zeros((1, 3), dtype=np.int64), 6)}
        parts = {"hand__v": np.zeros((3, 3)), "hand__f": np.zeros((1, 3), dtype=np.int64),
                 "hand__frame": np.array([6])}
        with mock.patch.object(fcl, "mesh_backend_status", return_value="ok"), \
                mock.patch.object(fcl, "import_collision_engine", return_value=(object(), "coal")), \
                mock.patch.object(fcl, "composed_parts", return_value=held):
            with self.assertRaises(ValueError) as caught:
                fcl.make_backend("ur5e", wrist_parts=parts)
        self.assertIn("are already held by the guard's arm, hand or plates", str(caught.exception))


class BodiesAreHandedInBeforeAnythingIsBuiltTests(unittest.TestCase):
    def test_the_preflight_hands_the_guard_its_bodies_and_says_what_it_holds(self) -> None:
        arm = URRobotArm(wrist_cell())
        body = wrist_body()
        arm.set_wrist_bodies([body])
        self.assertEqual(arm._preflight.wrist_bodies(arm), (body,))

    def test_a_guard_that_built_its_backend_refuses_a_body(self) -> None:
        arm = URRobotArm(wrist_cell())
        guard = arm._preflight._path_authority(arm)
        assert guard is not None
        guard._fcl_backend_built = True
        with self.assertRaises(RuntimeError) as caught:
            arm.set_wrist_bodies([wrist_body()])
        self.assertIn("before its first judged path", str(caught.exception))

    def test_an_arm_whose_planner_was_built_refuses_a_body(self) -> None:
        arm = URRobotArm(wrist_cell())
        arm._curobo_ur = object()  # type: ignore[assignment]
        with self.assertRaises(RuntimeError) as caught:
            arm.set_wrist_bodies([wrist_body()])
        self.assertIn("before its first planned move or planner start", str(caught.exception))

    def test_the_planner_is_built_with_the_camera_beside_the_hand(self) -> None:
        arm = URRobotArm(wrist_cell())
        body = wrist_body()
        arm.set_wrist_bodies([body])
        client = mock.MagicMock()
        with mock.patch("src.robot.safety.planning.margin.planner_margin_refusal", return_value=None), \
                mock.patch("src.robot.safety.planning.robot.retract_table.read_retract",
                           return_value=[0.0] * 6), \
                mock.patch("src.robot.drivers.ur.arm.CuroboPlanClient", client):
            arm._default_curobo_client_factory()()
        sent = client.call_args.kwargs
        self.assertEqual(sent["wrist_body_links"], [body.link()])
        self.assertNotIn("wrist_camera_wrist", [link["link"] for link in sent["body_links"]])

    def test_a_camera_free_arm_builds_its_planner_with_no_camera(self) -> None:
        """The control."""
        arm = URRobotArm(wrist_cell())
        client = mock.MagicMock()
        with mock.patch("src.robot.safety.planning.margin.planner_margin_refusal", return_value=None), \
                mock.patch("src.robot.safety.planning.robot.retract_table.read_retract",
                           return_value=[0.0] * 6), \
                mock.patch("src.robot.drivers.ur.arm.CuroboPlanClient", client):
            arm._default_curobo_client_factory()()
        self.assertEqual(client.call_args.kwargs["wrist_body_links"], [])


class TheRecordMeetsTheToolFrameTests(unittest.TestCase):
    def test_a_willy_arm_refuses_a_body_recorded_against_another_frame(self) -> None:
        arm = URRobotArm(wrist_cell())
        with self.assertRaises(ValueError) as caught:
            arm.set_wrist_bodies([_body(record=_shifted(0.01))])
        self.assertIn("so its body and its pick frame are both stale", str(caught.exception))
        arm.set_wrist_bodies([_body()])

    def test_a_polyscope_arm_refuses_to_connect_beyond_the_record_tolerance(self) -> None:
        from tests.test_ur_tool_frame import _conn

        for shift, refused in ((3.0, True), (0.5, False)):
            with self.subTest(shift=shift):
                arm = URRobotArm(wrist_cell(source="polyscope"))
                arm.set_wrist_bodies([_body(source="polyscope", record=_shifted(shift))])
                arm._conn = _conn(declared_frame())
                if refused:
                    with self.assertRaises(RobotConnectionError) as caught:
                        arm.connect()
                    self.assertIn("3.000 mm", str(caught.exception))
                    self.assertIsNone(arm.active_tool_frame)
                else:
                    arm.connect()
                    self.assertIsNotNone(arm.active_tool_frame)

    def test_a_record_from_the_other_mode_is_refused(self) -> None:
        arm = URRobotArm(wrist_cell())
        with self.assertRaises(ValueError) as caught:
            arm.set_wrist_bodies([_body(source="polyscope")])
        self.assertIn("was calibrated on a polyscope cell", str(caught.exception))


class TheSelfFilterTakesTheHousingOutTests(unittest.TestCase):
    def test_the_placed_spheres_join_the_envelope_and_a_camera_free_envelope_is_unchanged(self) -> None:
        from src.robot.safety.planning.self_envelope import self_envelope

        joints = [0.0, -1.2, 1.3, -0.4, 1.5, 0.2]
        bare = URRobotArm(wrist_cell())
        before = self_envelope(bare._preflight, bare, joints)
        carrying = URRobotArm(wrist_cell())
        body = wrist_body()
        carrying.set_wrist_bodies([body])
        after = self_envelope(carrying._preflight, carrying, joints)
        if before is None or after is None:
            self.skipTest("no committed arm bundle or hand map for the self filter on this tree")
        self.assertEqual(len(after.capsules), len(before.capsules) + len(body.envelope_spheres_mm()))
        self.assertEqual(after.capsules[:len(before.capsules)], before.capsules)


class TheExactGuardHoldsTheCameraTests(unittest.TestCase):
    def test_the_backend_holds_the_camera_part_and_the_evidence_parts_are_unchanged(self) -> None:
        from src.robot.safety._fcl_self_collision import composed_parts, mesh_backend_status

        arm = URRobotArm(wrist_cell())
        guard = arm._preflight._path_authority(arm)
        assert guard is not None
        model = guard.model_for(arm)
        hand = guard.hand
        variant = hand.guard_variant
        if model is None or mesh_backend_status(model, guard._config.mesh_dir, variant) != "ok":
            self.skipTest("no exact mesh engine or bundle on this box")
        arm.set_wrist_bodies([wrist_body()])
        backend = guard._exact_mesh_backend(model)
        assert backend is not None
        self.assertIn("wrist_camera_wrist", backend._names)
        self.assertEqual(backend._wrist, frozenset({"wrist_camera_wrist"}))
        placement = hand.placement
        parts = composed_parts(model, guard._config.mesh_dir, variant, hand.coupling_mm, placement=placement,
                               coupling_boxes=hand.coupling_boxes)
        self.assertEqual(sorted(parts), sorted(name for name in backend._names if name != "wrist_camera_wrist"))


if __name__ == "__main__":
    unittest.main()
