"""A hand written from its registry numbers says where its numbers start, and its envelope holds the whole hand.

Customer chain lane C2e and C2f (findings 5 and 23 of the audit of 2026-09-17). A body from dimensions was always a
flange hand: a customer who measured the grasp centre from the hand's own mounting face got a body one plate too close to
the flange and a cell that refused every plate. And the envelope took the finger back from the MEASURED reach instead of
the conservative length, and the housing's width along the closing axis from the fingers, which the EGU-50's own
registry file records as a corner no envelope held.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "src" / "robot" / "safety" / "data"


def _by_path(name: str, rel: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _writer() -> Any:
    return _by_path("_dims_writer_for_origin", "src/robot/safety/planning/robot/hand_from_dimensions.py")


def _cli() -> Any:
    return _by_path("_dims_cli_for_origin", "scripts/grippers/write_hand_from_dimensions.py")


def _jaw(hand: str, **changed: Any) -> Any:
    from src.config.grippers import load_gripper

    jaw = load_gripper(hand, aliases=False).jaw
    return jaw.model_copy(update=changed) if changed else jaw


def _run(argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        try:
            code = _cli().main(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            out.write(str(exc.code))
    return int(code), out.getvalue()


class TheOriginIsStatedTests(unittest.TestCase):
    def test_the_origin_is_required_and_stamped(self) -> None:
        writer = _writer()
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with self.assertRaises(TypeError):
            writer.write_bundle("robotiq_hande", folder / "a_hand_meshes.npz", inflation_mm=10.0)  # type: ignore[call-arg]
        for origin in ("mounting_face", "flange"):
            with self.subTest(origin=origin):
                target = folder / f"{origin}_hand_meshes.npz"
                writer.write_bundle("robotiq_hande", target, inflation_mm=10.0, origin=origin)
                with np.load(target, allow_pickle=True) as data:
                    self.assertEqual(str(data["gripper__origin"][0]), origin)

    def _outside_mm(self, hand: str, origin: str, coupling_mm: float) -> float:
        """How far the scanned hand lies outside the dimensions body, both placed by the guard at a +Z flange."""
        from src.robot.safety._fcl_self_collision import composed_parts
        from src.robot.safety.planning._hand_placement import HandPlacement

        placement = HandPlacement.from_quaternion_xyzw((0.0, 0.0, 0.0, 1.0))
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        shutil.copy(_DATA / "ur5e_collision_meshes.npz", folder / "ur5e_collision_meshes.npz")
        _writer().write_bundle(hand, folder / f"{hand}_hand_meshes.npz", inflation_mm=10.0, origin=origin)
        scanned = composed_parts("ur5e", None, hand, coupling_mm, placement=placement)
        envelope = composed_parts("ur5e", str(folder), hand, coupling_mm, placement=placement)
        points = np.concatenate([scanned[part][0] for part in ("gripper", "lfinger", "rfinger")])
        worst = np.full(len(points), np.inf)
        for part in ("gripper", "lfinger", "rfinger"):
            vertices = envelope[part][0]
            lower, upper = vertices.min(axis=0), vertices.max(axis=0)
            gap = np.maximum(np.maximum(lower - points, points - upper), 0.0)
            worst = np.minimum(worst, np.linalg.norm(gap, axis=1))
        return float(worst.max())

    def test_the_envelope_holds_the_scanned_hand_where_the_guard_places_it(self) -> None:
        self.assertAlmostEqual(self._outside_mm("robotiq_hande", "mounting_face", 20.0), 0.0, places=9)
        self.assertAlmostEqual(self._outside_mm("schunk_egu50", "flange", 0.0), 0.0, places=9)

    def test_a_wrong_origin_leaves_the_hand_outside(self) -> None:
        """⭐ THE CONTROL: the Hand-E written as a flange hand misses the plate the guard adds to the scan."""
        self.assertGreater(self._outside_mm("robotiq_hande", "flange", 20.0), 5.0)

    def test_measure_only_refuses_an_origin_its_bundle_contradicts(self) -> None:
        code, said = _run(["robotiq_hande", "--inflation-mm", "10", "--origin", "flange", "--measure-only"])
        self.assertNotEqual(code, 0)
        self.assertIn("mounting_face", said)
        self.assertEqual(_run(["robotiq_hande", "--inflation-mm", "10", "--origin", "mounting_face", "--measure-only"])[0], 0)
        self.assertEqual(_run(["schunk_egu50", "--inflation-mm", "10", "--origin", "flange", "--measure-only"])[0], 0)
        self.assertNotEqual(_run(["schunk_egu50", "--inflation-mm", "10", "--measure-only"])[0], 0)

    def test_a_flange_envelope_with_plates_is_told_the_dimensions_remedy(self) -> None:
        from src.config.loader import ConfigError
        from src.config.schema.robot import RobotConfig
        from src.robot.safety.planning.hand import planner_hand

        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        _writer().write_bundle("schunk_egu50", folder / "schunk_egu50_hand_meshes.npz", inflation_mm=10.0, origin="flange")
        config = RobotConfig.model_validate({
            "vendor": "ur",
            "gripper": {"model": "schunk_egu50", "coupling_plates": [{"name": "plate", "thickness_mm": 20.0}]},
            "safety": {"self_collision": {"mesh_dir": str(folder)}},
        })
        with self.assertRaises(ConfigError) as caught:
            planner_hand(config)
        self.assertIn("write_hand_from_dimensions.py", str(caught.exception))
        self.assertIn("--origin mounting_face", str(caught.exception))


class TheEnvelopeHoldsTheWholeHandTests(unittest.TestCase):
    def test_a_finger_box_reaches_the_conservative_length(self) -> None:
        jaw = _jaw("robotiq_2f85", palm_measured=True, palm_thickness_mm=84.89)
        boxes = {box.name: box for box in _writer().boxes_for_hand(jaw, inflation_mm=0.0)}
        finger = boxes["lfinger"]
        self.assertAlmostEqual(finger.centre_mm[1] - finger.half_extents_mm[1],
                               jaw.grasp_centre_mm - jaw.finger_length_mm, places=9)
        palm = boxes["gripper"]
        self.assertAlmostEqual(palm.centre_mm[1] + palm.half_extents_mm[1], jaw.grasp_centre_mm - jaw.finger_behind_mm,
                               places=9)

    def test_a_hand_whose_length_is_its_reach_keeps_its_boxes(self) -> None:
        """⭐ THE CONTROL: where the two reaches agree, as on the Hand-E and the EGU-50, the fingers do not move."""
        for hand in ("robotiq_hande", "schunk_egu50"):
            with self.subTest(hand=hand):
                jaw = _jaw(hand)
                self.assertEqual(jaw.finger_length_mm, jaw.finger_behind_mm)

    def test_a_housing_wider_than_the_open_fingers_is_enclosed(self) -> None:
        outer = 84.1 / 2.0 + 35.0
        for thickness, half in ((200.0, 100.0), (10.0, outer)):
            with self.subTest(palm_thickness_mm=thickness):
                boxes = {box.name: box for box in _writer().boxes_for_hand(_jaw("schunk_egu50", palm_thickness_mm=thickness),
                                                                           inflation_mm=2.0)}
                self.assertAlmostEqual(boxes["gripper"].half_extents_mm[0], half + 2.0, places=9)

    def test_a_hand_with_no_housing_thickness_is_refused_by_the_writer(self) -> None:
        writer = _writer()
        said = writer.refusal_for(_jaw("robotiq_hande", palm_thickness_mm=None))
        self.assertIsNotNone(said)
        self.assertIn("palm_thickness_mm", str(said))
        for hand in ("robotiq_hande", "schunk_egu50"):
            with self.subTest(hand=hand):
                self.assertIsNone(writer.refusal_for(_jaw(hand)))

    def test_the_gap_behind_the_housing_is_said(self) -> None:
        code, said = _run(["schunk_egu50", "--inflation-mm", "10", "--origin", "flange", "--measure-only"])
        self.assertEqual(code, 0)
        self.assertIn("29.10 mm", said)
        code, said = _run(["robotiq_hande", "--inflation-mm", "10", "--origin", "mounting_face", "--measure-only"])
        self.assertEqual(code, 0)
        self.assertNotIn("in no body", said)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
