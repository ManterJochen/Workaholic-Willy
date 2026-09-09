"""Every per-arm config profile has to describe the arm it is named after, and fit inside it.

⛔ **WHY A PROFILE EXISTS AT ALL.** The base tree is a UR5e. A cell that sets only ``ur.model: ur3``
keeps a 5 kg payload cap on a 3 kg arm and a motion box whose far corner sits 707 mm from the
shoulder on an arm that reaches 500. Nothing refuses that: the poses fail IK later, one at a time,
with nothing naming the cause. The profile is what stops the inheritance.

⚠ **AND WHY IT NEEDS A TEST.** A profile is four numbers copied next to a name, which is the exact
shape that goes quietly wrong. ``robot.ur10.yaml`` with a ``ur10e`` kinematics_model would load, plan
and pick, against link lengths 53 mm off in the shoulder.

DERIVED FROM THE DIRECTORY. The profiles are discovered by globbing, so a new arm is covered by the
act of adding it rather than by remembering this file.
"""

from __future__ import annotations

import math
import pathlib
import unittest

from src.config.loader import load_config
from src.config.schema.robot._ur_models import UR_MODEL_KEYS
from src.robot.drivers.sim.robot_models import _UR_MODELS
from src.robot.safety._ur_kinematics import UR_DH_TABLES_M

_DATA = pathlib.Path(__file__).resolve().parents[1] / "config" / "robot"


def _arm_profiles() -> list[str]:
    """Every ``robot.<model>.yaml`` whose model is a UR this stack supports."""
    return sorted(p.stem.split(".", 1)[1] for p in _DATA.glob("robot.*.yaml")
                  if p.stem.split(".", 1)[1] in UR_MODEL_KEYS)


class EveryArmProfileDescribesItsOwnArmTests(unittest.TestCase):
    def test_there_are_profiles_to_check(self) -> None:
        """The control. Globbing that matched nothing would make this whole file pass in silence."""
        self.assertGreaterEqual(len(_arm_profiles()), 4, f"found {_arm_profiles()}")

    def test_the_three_model_fields_agree_with_the_filename(self) -> None:
        """⭐ THE ONE THAT MATTERS. ``ur.model`` keys the driver, ``kinematics_model`` keys the safety
        DH chain and the collision bundle, ``sim.robot_model`` keys the asset. A profile that spells
        one of them differently is a cell planning against another robot, and the only visible sign
        is that it works."""
        for model in _arm_profiles():
            with self.subTest(model=model):
                robot = load_config(profile=model).robot
                self.assertEqual(robot.ur.model, model, "ur.model")
                self.assertEqual(robot.safety.self_collision.kinematics_model, model,
                                 "safety.self_collision.kinematics_model")
                self.assertEqual(robot.sim.robot_model, model, "sim.robot_model")

    def test_the_payload_cap_is_this_arm_datasheet_figure(self) -> None:
        for model in _arm_profiles():
            with self.subTest(model=model):
                robot = load_config(profile=model).robot
                self.assertEqual(robot.safety.payload.max_mass_kg,
                                 _UR_MODELS[model].max_payload_kg)

    @staticmethod
    def _shoulder_distance(model: str, x: float, y: float, z: float) -> float:
        return math.sqrt(x * x + y * y + (z - UR_DH_TABLES_M[model][0].d_m * 1000.0) ** 2)

    def test_the_middle_of_the_box_is_reachable(self) -> None:
        """⭐ THE INHERITED-BOX DEFECT. A cell that names a smaller arm and keeps the previous box
        cannot serve the MIDDLE of its own declared workspace, and every pick there fails IK one at a
        time with nothing naming the cause. The UR5e box centre sits 561 mm from a UR3 shoulder,
        which reaches 500.

        ⚠ THIS USED TO ASSERT THE CORNER, AND THAT WAS WRONG. A measured UR5e cell declares
        x[250,800] y[+-400] z[0,700], whose far corner is 1043.5 mm from an 850 mm shoulder, and that
        cell is correct: a workspace box is a rectangular BOUND on motion, not a promise that every
        point inside it can be reached. A pose in an unreachable corner fails IK, which is what
        should happen. The strict corner rule survives below, where it belongs.
        """
        for model in _arm_profiles():
            with self.subTest(model=model):
                w = load_config(profile=model).robot.workspace_limits
                centre = self._shoulder_distance(
                    model, (w.x_min + w.x_max) / 2.0, (w.y_min + w.y_max) / 2.0,
                    (w.z_min + w.z_max) / 2.0)
                reach = _UR_MODELS[model].max_reach_mm
                self.assertLessEqual(
                    centre, reach,
                    f"{model}: the CENTRE of the declared workspace is {centre:.1f} mm from the "
                    f"shoulder and the arm reaches {reach:.1f} mm. This box belongs to a bigger arm.",
                )

    def test_a_derived_profile_keeps_its_corner_inside_the_arm(self) -> None:
        """The strict rule, for the profiles that were CONSTRUCTED to satisfy it.

        A generated profile is a reach bound: its far corner sits exactly on the fraction of reach the
        one standing cell uses. A measured profile is a room and is exempt. They are told apart by
        reading the file, because a list of which is which beside a directory that already knows is
        the failure mode this whole suite keeps finding.
        """
        derived = [m for m in _arm_profiles()
                   if "REACH BOUND, NOT A MEASUREMENT" in (_DATA / f"robot.{m}.yaml").read_text(
                       encoding="utf-8")]
        self.assertTrue(derived, "no profile declares itself derived, so this test proves nothing")
        for model in derived:
            with self.subTest(model=model):
                w = load_config(profile=model).robot.workspace_limits
                corner = max(self._shoulder_distance(model, x, y, z)
                             for x in (w.x_min, w.x_max) for y in (w.y_min, w.y_max)
                             for z in (w.z_min, w.z_max))
                reach = _UR_MODELS[model].max_reach_mm
                self.assertLessEqual(
                    corner, reach,
                    f"{model} says it is a reach bound and its corner is {corner:.1f} mm from the "
                    f"shoulder against {reach:.1f} mm of reach, so it is not one",
                )

    def test_a_bigger_arm_does_not_get_a_smaller_box(self) -> None:
        """⚠ The direction nothing else catches. A box too LARGE fails the check above; one too small
        wastes the arm and passes everything, which is what the ur10e workspace did until it was
        given its own numbers."""
        by_reach = sorted(_arm_profiles(), key=lambda m: _UR_MODELS[m].max_reach_mm)
        volumes = []
        for model in by_reach:
            w = load_config(profile=model).robot.workspace_limits
            volumes.append((w.x_max - w.x_min) * (w.y_max - w.y_min) * (w.z_max - w.z_min))
        # Ties are fine (a UR5 and a UR5e reach the same); a REVERSAL is not.
        for (a, va), (b, vb) in zip(zip(by_reach, volumes), list(zip(by_reach, volumes))[1:]):
            if _UR_MODELS[a].max_reach_mm < _UR_MODELS[b].max_reach_mm:
                self.assertLessEqual(va, vb, f"{a} reaches less than {b} and has a bigger box")


class TheMeasuredCellsAreTighterThanTheDerivedBoundsTests(unittest.TestCase):
    """⭐ THE HONESTY CHECK, and the reason the derived profiles say what they say.

    ``robot.ur3e.yaml`` came off a cell that exists: its box is a room. The generated profiles are
    REACH BOUNDS, the largest box the arm could serve. If a derived bound ever came out TIGHTER than
    a measured cell of the same size, the derivation would be wrong, because no room is bigger than
    the arm that stands in it.
    """

    def test_the_measured_ur3e_cell_fits_inside_the_derived_ur3_bound(self) -> None:
        if "ur3" not in _arm_profiles():
            self.skipTest("no derived ur3 profile to compare against")
        measured = load_config(profile="ur3e").robot.workspace_limits
        derived = load_config(profile="ur3").robot.workspace_limits
        shoulder = UR_DH_TABLES_M["ur3e"][0].d_m * 1000.0
        far = max(math.sqrt(x * x + y * y + (z - shoulder) ** 2)
                  for x in (measured.x_min, measured.x_max)
                  for y in (measured.y_min, measured.y_max)
                  for z in (measured.z_min, measured.z_max))
        bound = max(math.sqrt(x * x + y * y + (z - shoulder) ** 2)
                    for x in (derived.x_min, derived.x_max)
                    for y in (derived.y_min, derived.y_max)
                    for z in (derived.z_min, derived.z_max))
        self.assertLessEqual(
            far, bound,
            f"the standing ur3e cell reaches {far:.1f} mm from the shoulder and the derived ur3 "
            f"bound stops at {bound:.1f} mm. A real room cannot be bigger than the arm bound, so "
            f"the derivation is understating what the arm can do.",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
