"""The Hand-E and the EGU-50 are described once, as data, and every number in their files is the bundle's.

Owner, lane (i) Q-A: the hand numbers no file held (the Hand-E's reach behind its grasp centre, every EGU-50 jaw
number) are measured off the committed collision bundles by ``scripts/grippers/measure_jaw_from_bundle.py``, and
this file recomputes them with that one implementation. The measurement is trusted only because it first
reproduces, to 0.05 mm, the Hand-E numbers the tree already carried from its own measurement session
(``robot.hande.yaml`` until customer chain lane C4d moved them to the registry file). A measurement that cannot
reproduce a known hand measures nothing new.

Two findings of that measurement were decided by the owner (``.commits/robot/51-the-world-and-the-hand.md``):

* The Hand-E's committed contact patch is 0.91 mm longer than its flat jaw face, down a chamfer. The committed
  patch stays, and this file pins the difference, so a re-baked bundle that changes it is noticed.
* The EGU-50's sim grasp centre sat 7.5 mm below its flat jaw face, tuned for one cube. The registry's grasp
  centre is the face midpoint, 159.1 mm from the flange, and the sim mount grasps there too.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType

from src.config import load_config
from src.config.grippers import available_grippers, load_gripper
from src.robot.grippers.sim.gripper import ROBOTIQ_HANDE_PROFILE, SCHUNK_EGU50_PROFILE
from src.robot.safety.planning.environment import hand_mesh_bundle

_ROOT = Path(__file__).resolve().parents[1]
_MEASURE = _ROOT / "scripts" / "grippers" / "measure_jaw_from_bundle.py"
#: The tolerance the 2F-85's own measured dimensions are held to (tests/test_gripper_dimensions_agree.py).
_TOLERANCE_MM = 0.05


def _measurement() -> ModuleType:
    """The measuring script, loaded by path so the script and this test run one implementation."""
    name = "_measure_jaw_from_bundle"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _MEASURE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _widths(profile) -> list[float]:  # noqa: ANN001
    return [float(width) for _, width in profile.angle_width_table]


class TheHandEFileTests(unittest.TestCase):
    def test_the_hande_profile_runs_the_hande_file(self) -> None:
        """The actuation widths and the envelope the `hande` profile runs, number for number.

        Since customer chain lane C4d the layer states none of them and the loader takes them from this file, so this
        holds the wiring, not the numbers: the numbers are held by the measurement below.
        """
        jaw = load_gripper("robotiq_hande", aliases=False).jaw
        robot = load_config(profile="hande").robot
        actuation, envelope = robot.gripper, robot.grasping.gripper_geometry.parallel_jaw
        for field, expected in {
            "aperture_mm": actuation.max_width_mm,
            "min_width_mm": actuation.min_width_mm,
            "closed_width_mm": actuation.closed_width_mm,
            "finger_length_mm": envelope.finger_length_mm,
            "finger_thickness_mm": envelope.finger_thickness_mm,
            "finger_width_mm": envelope.finger_width_mm,
            "finger_pad_overlap_mm": envelope.finger_pad_overlap_mm,
            "finger_ahead_mm": envelope.fingertip_depth_mm,
            "pad_length_mm": envelope.pad_length_mm,
            "pad_ahead_mm": envelope.pad_ahead_mm,
            "palm_depth_mm": envelope.palm_depth_mm,
            "palm_width_mm": envelope.palm_width_mm,
        }.items():
            with self.subTest(field=field):
                self.assertEqual(getattr(jaw, field), expected)

    def test_the_hande_file_is_the_sim_stroke(self) -> None:
        jaw = load_gripper("robotiq_hande", aliases=False).jaw
        self.assertEqual(jaw.aperture_mm, max(_widths(ROBOTIQ_HANDE_PROFILE)))
        self.assertEqual(jaw.closed_width_mm, min(_widths(ROBOTIQ_HANDE_PROFILE)))

    def test_what_the_hande_file_measured_and_what_it_did_not(self) -> None:
        spec = load_gripper("robotiq_hande", aliases=False)
        self.assertEqual(spec.aliases, ())
        self.assertTrue(spec.jaw.palm_measured)
        self.assertTrue(spec.jaw.friction_is_default)


class TheEgu50FileTests(unittest.TestCase):
    def test_the_egu50_file_is_the_sim_stroke(self) -> None:
        jaw = load_gripper("schunk_egu50", aliases=False).jaw
        self.assertEqual(jaw.aperture_mm, max(_widths(SCHUNK_EGU50_PROFILE)))
        self.assertEqual(jaw.closed_width_mm, min(_widths(SCHUNK_EGU50_PROFILE)))

    def test_the_egu50_housing_reaches_past_its_open_fingers_by_the_recorded_corner(self) -> None:
        """palm_thickness_mm is new; its half against the fingers' outer face is the 2.25 mm corner the file recorded
        before the field existed, so the new measurement is tied to an old one."""
        jaw = load_gripper("schunk_egu50", aliases=False).jaw
        assert jaw.palm_thickness_mm is not None
        self.assertAlmostEqual(jaw.palm_thickness_mm / 2.0 - (jaw.aperture_mm / 2.0 + jaw.finger_thickness_mm), 2.25,
                               delta=0.05)

    def test_what_the_egu50_file_measured_and_what_it_did_not(self) -> None:
        spec = load_gripper("schunk_egu50", aliases=False)
        self.assertEqual(spec.aliases, ())
        self.assertTrue(spec.jaw.palm_measured)
        self.assertTrue(spec.jaw.friction_is_default)


class TheNumbersAreTheBundlesTests(unittest.TestCase):
    _MEASURED = ("aperture_mm", "finger_ahead_mm", "finger_behind_mm", "finger_thickness_mm",
                 "finger_width_mm", "palm_depth_mm", "palm_width_mm", "palm_thickness_mm")

    def test_the_measurement_reproduces_the_committed_hande_numbers(self) -> None:
        """The control: every number the Hand-E layer already carried, at the centre it already declared."""
        measured = _measurement().measure_parallel_jaw(
            hand_mesh_bundle("robotiq_hande"), centre_mm=135.75,
        )
        jaw = load_gripper("robotiq_hande", aliases=False).jaw
        for field in self._MEASURED:
            with self.subTest(field=field):
                self.assertAlmostEqual(getattr(measured, field), getattr(jaw, field), delta=_TOLERANCE_MM)
        self.assertEqual(jaw.finger_behind_mm, jaw.finger_length_mm,
                         "the Hand-E envelope carries no margin behind the grasp centre")

    def test_the_hande_patch_is_its_flat_face_and_a_chamfer(self) -> None:
        """Pinned, not fixed: the owner kept the committed patch (lane-i-decisions, Asked while building i1)."""
        measured = _measurement().measure_parallel_jaw(
            hand_mesh_bundle("robotiq_hande"), centre_mm=135.75,
        )
        jaw = load_gripper("robotiq_hande", aliases=False).jaw
        self.assertAlmostEqual(measured.face_length_mm, 20.00, delta=_TOLERANCE_MM)
        self.assertAlmostEqual(jaw.pad_length_mm - measured.face_length_mm, 0.91, delta=_TOLERANCE_MM)

    def test_the_egu50_file_is_its_bundle_around_the_face_midpoint(self) -> None:
        measured = _measurement().measure_parallel_jaw(hand_mesh_bundle("schunk_egu50"))
        jaw = load_gripper("schunk_egu50", aliases=False).jaw
        for field in (*self._MEASURED, "pad_length_mm", "pad_ahead_mm", "pad_behind_mm"):
            with self.subTest(field=field):
                self.assertAlmostEqual(getattr(measured, field), getattr(jaw, field), delta=_TOLERANCE_MM)
        self.assertEqual(jaw.finger_behind_mm, jaw.finger_length_mm)

    def test_a_grasp_centre_off_the_jaw_face_is_refused(self) -> None:
        """The EGU-50's old sim centre, 144.1 mm from the flange, is 7.5 mm below its face."""
        module = _measurement()
        with self.assertRaises(ValueError) as caught:
            module.measure_parallel_jaw(hand_mesh_bundle("schunk_egu50"), centre_mm=144.1)
        self.assertIn("face", str(caught.exception))


class TheMountGraspsAtItsFaceTests(unittest.TestCase):
    def test_the_egu50_mount_grasps_at_its_jaw_face_midpoint(self) -> None:
        from src.willy_sim.grippers import SCHUNK_EGU50_MOUNT

        measured = _measurement().measure_parallel_jaw(hand_mesh_bundle("schunk_egu50"))
        self.assertEqual(measured.origin, "flange")
        self.assertAlmostEqual(
            SCHUNK_EGU50_MOUNT.flange_to_tcp_offset_mm[1], measured.centre_mm, delta=_TOLERANCE_MM,
        )


class TheShippedHandsTests(unittest.TestCase):
    """The hands this repository ships are in the registry, and every registry hand has a body.

    It pinned exactly three names until customer chain lane C1d, so a customer's fourth hand turned the suite red
    whether or not that hand had a body. What has to hold is that the shipped ones are there and that no registry
    hand is a name the guard cannot compose; tests/test_a_hand_writer_records_its_row.py holds both halves on a
    scratch registry with a fourth hand.
    """

    def test_the_hands_this_repository_ships_are_in_the_registry(self) -> None:
        self.assertLessEqual({"robotiq_2f85", "robotiq_hande", "schunk_egu50"}, set(available_grippers()))

    def test_every_registry_hand_has_a_body(self) -> None:
        for hand in available_grippers():
            with self.subTest(hand=hand):
                self.assertTrue(
                    hand_mesh_bundle(hand).is_file(),
                    f"{hand} is in the registry and has no {hand_mesh_bundle(hand).name}: write one with "
                    f"scripts/grippers/write_hand_from_dimensions.py, scripts/grippers/write_hand_from_mesh.py or "
                    f"scripts/grippers/bake_gripper_variant.py (docs/runbooks/your_own_gripper.md)",
                )


if __name__ == "__main__":
    unittest.main()
