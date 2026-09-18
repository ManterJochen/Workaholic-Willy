"""The exact mesh guard composes the arm with the hand at load and places the hand by its declared rotation (UM lane S06).

``make_backend`` used to load one arm plus hand file and shift a mounting face hand one plate along tool0 +Y. It now
composes ``{arm}_collision_meshes.npz`` with ``{hand}_hand_meshes.npz`` (S05 proved that reproduces every file that
existed) and places the hand's parts by ``HandPlacement``: the plate first, along the model's approach, then the
rotation. The identity placement does no arithmetic at all, which S02's golden holds byte for byte.

Two states the guard now names: a hand with no bundle of its own is ``no_hand_bundle``, and a hand composed onto an arm
it was never proven on keeps ``variant_model_mismatch``, so the EGU-50 does not quietly become exact meshes on an arm
where nothing has shown its placement is right.
"""

from __future__ import annotations

import pathlib
import shutil
import tempfile
import unittest
from typing import Any
from unittest import mock

import numpy as np

from src.robot.safety.planning import environment

_DATA = environment.COLLISION_MESH_DIR
_UR_Z = (0.0, 0.0, 0.0, 1.0)


def _placement(q: tuple[float, float, float, float]) -> Any:
    from src.robot.safety.planning._hand_placement import HandPlacement

    return HandPlacement.from_quaternion_xyzw(q)


class ThePlacementMovesTheHandTests(unittest.TestCase):
    def test_the_plate_goes_on_first_and_the_rotation_after(self) -> None:
        from src.robot.safety._fcl_self_collision import place_hand_vertices

        v = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
        out = place_hand_vertices("lfinger", v, origin="mounting_face", coupling_mm=20.0, placement=_placement(_UR_Z))
        np.testing.assert_allclose(out[:, 0], v[:, 0])
        np.testing.assert_allclose(out[:, 2], v[:, 1] + 20.0)
        np.testing.assert_allclose(out[:, 1], -v[:, 2])

    def test_rotating_before_the_plate_would_be_caught(self) -> None:
        """The control for the order above: turned first and then shifted along tool0 +Y, the plate lands on the
        wrong axis, and the assertion on out[:, 2] could not hold."""
        v = np.array([[1.0, 2.0, 3.0]])
        wrong = (np.asarray(_placement(_UR_Z).rotation) @ v.T).T + np.array([0.0, 20.0, 0.0])
        self.assertNotAlmostEqual(float(wrong[0, 2]), float(v[0, 1] + 20.0))

    def test_an_arm_link_is_never_placed_as_a_hand(self) -> None:
        from src.robot.safety._fcl_self_collision import place_hand_vertices

        v = np.array([[1.0, 2.0, 3.0]])
        out = place_hand_vertices("wrist_3", v, origin="mounting_face", coupling_mm=20.0, placement=_placement(_UR_Z))
        self.assertIs(out, v)

    def test_a_flange_hand_takes_no_plate(self) -> None:
        from src.robot.safety._fcl_self_collision import place_hand_vertices

        v = np.array([[1.0, 2.0, 3.0]])
        out = place_hand_vertices("gripper", v, origin="", coupling_mm=20.0, placement=_placement(_UR_Z))
        np.testing.assert_allclose(out, [[1.0, -3.0, 2.0]])

    def test_the_identity_does_no_arithmetic(self) -> None:
        from src.robot.safety._fcl_self_collision import place_hand_vertices
        from src.robot.safety.planning._hand_placement import HandPlacement

        v = np.array([[1.0, 2.0, 3.0]])
        self.assertIs(place_hand_vertices("gripper", v, origin="", coupling_mm=0.0, placement=HandPlacement.undeclared()), v)


class _Captured:
    last: dict[str, Any] | None = None

    def __init__(self, adapter: Any, meshes: dict[str, Any]) -> None:
        type(self).last = meshes


class TheGuardIsHandedThePlacedHandTests(unittest.TestCase):
    def _capture(self, **kwargs: Any) -> dict[str, Any] | None:
        from src.robot.safety import _fcl_self_collision as fcl

        _Captured.last = None
        with mock.patch.object(fcl, "import_collision_engine", return_value=(object(), "coal")), \
                mock.patch.object(fcl, "_EngineAdapter", lambda mod, kind: None), \
                mock.patch.object(fcl, "MeshSelfCollisionBackend", _Captured):
            fcl.make_backend("ur5e", None, "robotiq_hande", coupling_mm=20.0, **kwargs)
        return _Captured.last

    def test_a_z_hand_is_rotated_and_the_arm_is_untouched(self) -> None:
        identity = self._capture()
        turned = self._capture(placement=_placement(_UR_Z))
        assert identity is not None and turned is not None
        for name in ("shoulder", "forearm", "wrist_3"):
            with self.subTest(part=name):
                self.assertEqual(np.asarray(turned[name][0]).tobytes(), np.asarray(identity[name][0]).tobytes())
        v = np.asarray(identity["lfinger"][0])
        np.testing.assert_allclose(np.asarray(turned["lfinger"][0]), np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1))


class TheGuardNamesWhatItCannotComposeTests(unittest.TestCase):
    def test_a_hand_with_no_bundle_of_its_own(self) -> None:
        from src.robot.safety._fcl_self_collision import mesh_backend_status

        with tempfile.TemporaryDirectory() as tmp:
            shutil.copyfile(_DATA / "ur5e_collision_meshes.npz", pathlib.Path(tmp) / "ur5e_collision_meshes.npz")
            self.assertEqual(mesh_backend_status("ur5e", tmp, "robotiq_hande"), "no_hand_bundle")

    def test_an_old_admission_list_in_a_bundle_admits_and_refuses_nothing(self) -> None:
        """⛔ UM lane S22 retired `hand__admitted_arms`. A bundle written before that still carries one, and it
        must not come back to life: a list naming only the ur5e does not refuse the ur3e, because admission is the
        committed evidence now and a second mechanism beside it would answer for pairings nobody measured."""
        from src.robot.safety._fcl_self_collision import mesh_backend_status

        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            shutil.copyfile(_DATA / "ur3e_collision_meshes.npz", folder / "ur3e_collision_meshes.npz")
            hand = dict(np.load(_DATA / "robotiq_hande_hand_meshes.npz"))
            hand["hand__admitted_arms"] = np.array(["ur5e"])
            np.savez_compressed(folder / "robotiq_hande_hand_meshes.npz", **hand)
            self.assertIn(mesh_backend_status("ur3e", tmp, "robotiq_hande"), {"ok", "no_engine"})


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, model: str, mesh_dir: Any = None, mesh_name: Any = None, coupling_mm: float = 0.0,
                 **kwargs: Any) -> None:
        self.calls.append({"model": model, "mesh_name": mesh_name, "coupling_mm": coupling_mm, **kwargs})
        return None


class TheCallersPassThePlacementTests(unittest.TestCase):
    _UR_FRAME = {"source": "polyscope", "offset_mm": [0.0, 0.0, 155.75], "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}

    def _hand(self) -> Any:
        from src.config.schema.robot import RobotConfig
        from src.robot.safety.planning.hand import planner_hand

        return planner_hand(RobotConfig.model_validate({
            "vendor": "ur",
            "gripper": {"model": "robotiq_hande", "coupling_plates": [{"name": "plate", "thickness_mm": 20.0}], "tool_frame": self._UR_FRAME},
        }))

    def test_the_one_shot_guard(self) -> None:
        from src.config.schema.robot import SelfCollisionSafetyConfig
        from src.robot.safety.self_collision import SelfCollisionGuard

        recorder = _Recorder()
        guard = SelfCollisionGuard(SelfCollisionSafetyConfig(backend="fcl", kinematics_model="ur5e"), hand=self._hand())
        with mock.patch("src.robot.safety._fcl_self_collision.make_backend", recorder):
            guard._exact_mesh_backend("ur5e")
        self.assertEqual(len(recorder.calls), 1)
        self.assertEqual(recorder.calls[0]["placement"].approach, "+Z")

    def test_the_continuous_monitor(self) -> None:
        from src.robot.safety.continuous_monitor import ContinuousCollisionMonitor, ContinuousGuardProfile

        recorder = _Recorder()
        with mock.patch("src.robot.safety.continuous_monitor.make_backend", recorder):
            ContinuousCollisionMonitor.from_model(
                "ur5e", 180.0, (), ContinuousGuardProfile(enabled=True, margin_mm=8.0),
                variant="robotiq_hande", coupling_mm=20.0, placement=self._hand().placement,
            )
        self.assertEqual(recorder.calls[0]["placement"].approach, "+Z")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
