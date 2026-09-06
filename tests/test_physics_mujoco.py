"""The physics referee on MuJoCo, and the controls that make anything it says mean something.

⭐⭐ WHY A SECOND REFEREE EXISTS. The shake answers the plan's PRIMARY metric, whether a grasp the
geometry calls valid actually holds, and it ran only under Isaac Sim: Windows or Linux, NVIDIA only,
about 47 GB. Everything else in this pipeline already runs without it, and our own v5 corpus of 8
million labels was rendered on MuJoCo. The referee was the one step forcing a workstation, and
therefore the one step deciding whether "generate your own data" is a claim a customer can act on.

⛔⛔ THE CONTROLS ARE THE POINT AND THEY ARE NOT A FORMALITY. A harness that cannot grip a block, that
grips thin air, or that stops gripping once a second scene has been built measures nothing, and all
three have happened in this repository. One run returned 0 of 48 with a positive and a negative both
passing, because the fault appeared only on the SECOND scene build.

⚠ AND PASSING THEM DOES NOT MAKE TWO REFEREES ONE REFEREE. These tests establish that the MuJoCo
harness works. Whether its verdicts match Isaac's is a separate measurement on the same grasps, which
nobody has taken, and until then the engine is recorded in every row.
"""

from __future__ import annotations

import unittest

import numpy as np

from datagen.grasps.labels import SceneGeometry
from datagen.grasps.shapes import Solid

mujoco = None
try:  # pragma: no cover - the import is the thing under test on a machine that has it
    import mujoco  # noqa: F811
except (ImportError, OSError):  # pragma: no cover
    # ⛔⛔ `OSError` TOO, AND CATCHING ONLY `ImportError` COST THE WHOLE SUITE. On 2026-09-05 Windows
    # Smart App Control began refusing `mujoco.dll` -- an unsigned native build whose reputation it
    # re-evaluated days after installation, the same behaviour recorded for this box before. The
    # import then raises `OSError: [WinError 4551]`, which sails past this guard, and pytest treats
    # an unimportable test module as a COLLECTION ERROR rather than a skip. Measured: exit 2,
    # `Interrupted: 1 error during collection`, ZERO of 6,300 tests run. Three sessions were reading
    # a red suite that said nothing about their own diffs.
    #
    # ⚠ THE TWO EXCEPTIONS ARE TWO DIFFERENT FACTS. `ImportError` is "not installed"; `OSError` is
    # "installed, and this machine will not load it". Both mean the same thing to a test that skips,
    # and only one of them was written down.
    pass


def _block(half_mm, centre_mm, mass_kg: float = 0.2) -> SceneGeometry:
    return SceneGeometry(
        scene_id="test", family="alone",
        objects={0: Solid(kind="box", half_extent_mm=np.asarray(half_mm, dtype=np.float64),
                          rotation=np.eye(3), centre_mm=np.asarray(centre_mm, dtype=np.float64),
                          instance_id=0, asset_id="block", mass_kg=mass_kg)},
        walls=())


@unittest.skipIf(mujoco is None, "mujoco is not installed on this machine")
class ControlTests(unittest.TestCase):
    """The four questions that make every hold rate below them a statement about the grasp."""

    @classmethod
    def setUpClass(cls) -> None:
        from datagen.grasps.physics_mujoco import PhysicsCell

        with PhysicsCell() as cell:
            cls.controls = cell.run_controls()

    def test_the_shut_jaw_is_solid_and_the_open_one_is_not(self) -> None:
        """⛔ A DIFFERENTIAL, BECAUSE THE ABSOLUTE VERSION DID NOT DISCRIMINATE. The first version
        dropped a block onto shut pads and asked whether it stayed above them. MEASURED, it stays
        above them with the jaw WIDE OPEN too, 279.9 mm against 334.8, because it falls between the
        pads and lands on the palm behind them. Correct physics, useless as a control. Worse, an
        absolute threshold also passes when nothing is simulated at all, since a block that never
        fell is still above the line."""
        self.assertTrue(self.controls["jaw_is_solid"], self.controls)
        gap = self.controls["jaw_rest_shut_mm"] - self.controls["jaw_rest_open_mm"]
        self.assertGreater(gap, 20.0, f"shut and open are indistinguishable: {self.controls}")

    def test_a_block_gripped_across_its_width_survives_the_support_going(self) -> None:
        self.assertTrue(self.controls["positive_held"], self.controls)

    def test_a_jaw_commanded_far_too_wide_does_NOT_hold(self) -> None:
        """A harness reporting a hold here is reporting something other than contact."""
        self.assertFalse(self.controls["negative_held"], self.controls)

    def test_the_positive_STILL_holds_after_an_unrelated_scene(self) -> None:
        """⭐ THE CONTROL THAT MATTERS. A run once returned 0 of 48 with the positive and the negative
        both passing, because the fault appeared only on the second scene build."""
        self.assertTrue(self.controls["repeat_held"], self.controls)

    def test_the_engine_is_recorded(self) -> None:
        """Two referees are not one referee until somebody measures that they agree."""
        self.assertEqual(self.controls["engine"], "mujoco")


@unittest.skipIf(mujoco is None, "mujoco is not installed on this machine")
class SimulationTests(unittest.TestCase):
    """⚠ The controls ran in 0.1 s, which is what made me check that anything moves at all."""

    def test_a_released_block_falls_to_the_floor(self) -> None:
        """If this fails, every number the referee produces is about a frozen world."""
        from datagen.grasps.physics_mujoco import PhysicsCell

        with PhysicsCell() as cell:
            cell.build_scene(_block((20, 20, 20), (0, 0, 400)))
            cell._step(400)                                    # noqa: SLF001 - same package
            rest = cell._object_position(0)                    # noqa: SLF001
        self.assertLess(abs(float(rest[2]) - 20.0), 3.0,
                        f"a 20 mm half-block should rest at z = 20, found {rest[2]:.1f}")

    def test_the_gripper_reaches_its_commanded_pose(self) -> None:
        from datagen.grasps.physics_mujoco import PhysicsCell

        with PhysicsCell() as cell:
            cell.build_scene(_block((20, 20, 20), (0, 0, 20)))
            cell._place(np.array([0.0, 0.0, 250.0]), np.array([0.0, 0.0, -1.0]),  # noqa: SLF001
                        np.array([1.0, 0.0, 0.0]))
            cell._step(200)                                    # noqa: SLF001
            offset = cell.gripper_offset_mm()
        self.assertLess(offset, 5.0, f"the weld did not bring the palm to its command: {offset}")

    def test_the_jaw_teleport_leaves_no_velocity_behind(self) -> None:
        """⛔ A burst trial used to poison every trial after it, 26 of 36 in one sample, because the
        jaw was ASKED to travel back from wherever the explosion left it. Writing the degrees of
        freedom directly took scored trials from 27.8 % to 91.7 %."""
        from datagen.grasps.physics_mujoco import PhysicsCell

        with PhysicsCell() as cell:
            cell.build_scene(_block((20, 20, 20), (0, 0, 20)))
            cell._drive_jaw(0.0, 50)                           # noqa: SLF001
            cell._teleport_jaw(85.0)                           # noqa: SLF001
            opening = cell._jaw_opening_mm()                   # noqa: SLF001
        self.assertAlmostEqual(opening, 85.0, delta=1.0)


@unittest.skipIf(mujoco is None, "mujoco is not installed on this machine")
class ApertureTests(unittest.TestCase):
    """⛔⛔ THE JAW MAY NOT READ WIDER THAN THE HAND THIS CELL MODELS, and for a while it could.

    MEASURED on the 19,853-trial v5_s0 shake with the old 200 mm threshold: 686 trials, 3.46 %, were
    SCORED with the jaw read past 92 mm on an 85 mm hand. They were commanded to a median of 55.8 mm
    and ended a median of 79.7 mm wider than commanded, and their hold rate falls with the opening,
    24 % at 92-100 mm down to 8 % above 160, against 23.4 % overall. Re-analysed under the corrected
    threshold the rate moves from 23.37 % to 23.62 %, so the aggregate barely moves; what moves is
    that 106 trials were counted as HELD by fingers wider apart than the hand can open.
    """

    def test_a_jaw_opened_past_the_hand_is_NOT_sane(self) -> None:
        from datagen.grasps.physics_mujoco import _JAW_APERTURE_MM, PhysicsCell

        with PhysicsCell() as cell:
            cell.build_scene(_block((20, 20, 20), (0, 0, 20)))
            cell._teleport_jaw(_JAW_APERTURE_MM)               # noqa: SLF001
            self.assertEqual(cell.jaw_fault(), "")
            cell._teleport_jaw(150.0)                          # noqa: SLF001
            self.assertIn("past the 85 mm hand", cell.jaw_fault())

    def test_the_JOINT_RANGE_still_allows_what_the_check_refuses(self) -> None:
        """⛔⛔ THE HEADROOM IS THE DETECTOR, and clamping it was my first plan. The slide joints
        allow 110 mm per finger, well past this hand. If they were tightened to the aperture a blown
        jaw would read exactly 85 mm, `jaw_fault` would return empty, and the check would go silent
        on the very trials it exists to catch. This test fails if anyone tightens them."""
        from datagen.grasps.physics_mujoco import _JAW_SANE_MM, PhysicsCell

        with PhysicsCell() as cell:
            cell.build_scene(_block((20, 20, 20), (0, 0, 20)))
            cell._teleport_jaw(200.0)                          # noqa: SLF001
            reading = cell._jaw_opening_mm()                   # noqa: SLF001
        self.assertGreater(reading, _JAW_SANE_MM,
                           f"the joints clamp at {reading:.1f} mm, so a blown jaw is now invisible")

    def test_the_threshold_is_taken_from_the_HAND_not_from_a_corpus(self) -> None:
        """⚠ 10 % over the aperture rather than the 92 mm the histogram suggested. A position servo
        IS back-driven by a stiff object, and the trials just over the aperture are real grips: 2.3 %
        read 80-85 mm and held 73 % of the time, 1.9 % read 85-88 mm and held 69 %. A number fitted to
        one corpus would not survive the next hand."""
        from datagen.grasps.physics_mujoco import _JAW_APERTURE_MM, _JAW_SANE_MM

        self.assertGreater(_JAW_SANE_MM, _JAW_APERTURE_MM)
        self.assertLess(_JAW_SANE_MM, _JAW_APERTURE_MM * 1.25)

    def test_the_jaw_is_checked_AFTER_the_hang_as_well(self) -> None:
        """⛔ The hang is 0.8 s of gravity with the support gone, which is the phase most able to
        blow the jaw open, and it used to be checked only after the close. An object could be
        reported as held by a pair of fingers 150 mm apart."""
        from pathlib import Path as _Path

        body = _Path("datagen/grasps/physics_mujoco.py").read_text(encoding="utf-8")
        after_hang = body.split("self._set_contact(support, on=True)")[-1]
        self.assertIn("jaw_fault()", after_hang)


@unittest.skipIf(mujoco is None, "mujoco is not installed on this machine")
class RefusalTests(unittest.TestCase):
    """A refusal is reported and kept out of every rate; a wrong verdict is not."""

    def _shake(self, **overrides):
        from datagen.grasps.physics import PhysicsTrial
        from datagen.grasps.physics_mujoco import PhysicsCell

        fields = dict(scene_id="test", instance_id=0, source="label",
                      position_mm=(0.0, 0.0, 20.0), approach=(0.0, 0.0, -1.0),
                      closing_axis=(1.0, 0.0, 0.0), width_mm=40.0)
        fields.update(overrides)
        geometry = _block((20, 20, 20), (0, 0, 20))
        with PhysicsCell() as cell:
            cell.restore(geometry)
            return cell.run_trial(PhysicsTrial(**{k: v for k, v in fields.items()
                                                  if k in PhysicsTrial.__dataclass_fields__}),
                                  geometry)

    def test_a_grasp_with_no_closing_axis_is_refused(self) -> None:
        outcome = self._shake(closing_axis=(0.0, 0.0, 0.0))
        self.assertIn("refused", outcome.note)
        self.assertFalse(outcome.held)

    def test_a_non_finite_pose_is_refused(self) -> None:
        outcome = self._shake(position_mm=(float("nan"), 0.0, 20.0))
        self.assertIn("refused", outcome.note)

    def test_an_instance_that_is_not_in_the_scene_is_refused(self) -> None:
        outcome = self._shake(instance_id=7)
        self.assertIn("refused", outcome.note)


class EngineChoiceTests(unittest.TestCase):
    """Reachable from the CLI, and an unknown engine refuses by name rather than importing nothing."""

    def test_an_unknown_engine_refuses(self) -> None:
        from pathlib import Path

        from datagen.grasps.physics import run_physics_sample

        with self.assertRaises(ValueError) as caught:
            run_physics_sample(Path("nowhere"), engine="pybullet")
        self.assertIn("isaac, mujoco", str(caught.exception))

    def test_the_flag_is_reachable_and_says_what_it_costs(self) -> None:
        from pathlib import Path as _Path

        text = _Path("datagen/__main__.py").read_text(encoding="utf-8")
        self.assertIn('"--physics-engine"', text)
        self.assertIn("physics_engine=args.physics_engine", text)
        self.assertIn("47 GB", text)


if __name__ == "__main__":
    unittest.main()
