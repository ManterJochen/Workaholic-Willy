"""Four registries answer "which UR can this stack drive", and until now none checked another.

    UR_MODEL_KEYS          src/config/schema/robot/_ur_models.py     the config gate
    UR_DH_TABLES_M         src/robot/safety/_ur_kinematics.py    the link lengths
    UR_JOINT_LIMITS_DEG    src/robot/safety/joint_limits.py      the axis envelope
    _UR_MODELS             src/robot/drivers/sim/robot_models.py the asset and the reach

⛔ **THE TWO DIRECTIONS FAIL DIFFERENTLY, AND ONE OF THEM IS SILENT.** A model with a DH row and no
config key is refused at load with "unknown UR model", which is loud and harmless. The reverse, a key
or a spec with no DH row, degrades without an error: `mesh_backend_status` returns `unknown_model` and
the guard drops to the capsule proxy, `_ur_arm_capsules` returns `None` so arm-against-arm checking
disappears, and `reach.shoulder_height_mm` returns 0.0, which over-states the working sphere by the
shoulder height. A cell in that state plans and picks and reports nothing.

MEASURED 2026-09-09, before this file existed: the DH table held seven models, the joint limits held
eight, and the config gate and the spec registry held three. So `ur10` had correct kinematics and
could not be configured at all, and the family had been drifting in both directions at once.

⚠ **THE CB-SERIES AND E-SERIES PAIRS ARE THE REASON THIS MATTERS.** A UR5 and a UR5e share `a2`
exactly, differ by 0.05 mm in `a3` and by 73 mm in `d1`. Nothing about a swap looks wrong; the arm is
simply somewhere else.
"""

from __future__ import annotations

import math
import unittest

from src.config.schema.robot._ur_models import UR_MODEL_KEYS
from src.robot.drivers.sim.robot_models import _UR_MODELS
from src.robot.safety._ur_kinematics import UR_DH_TABLES_M
from src.robot.safety.joint_limits import UR_JOINT_LIMITS_DEG

#: How much reach a plan is not allowed to spend, so a patch sits inside the DEXTEROUS part of the
#: sphere rather than at full extension where IK stops converging. The number the ur3e entry was
#: computed with.
_DEXTERITY_MARGIN_MM = 40.0


def _horizontal_radius_mm(model: str) -> float:
    """How far out along the table a model can reach, given its reach sphere and shoulder height."""
    reach = _UR_MODELS[model].max_reach_mm - _DEXTERITY_MARGIN_MM
    shoulder = UR_DH_TABLES_M[model][0].d_m * 1000.0
    return math.sqrt(max(reach * reach - shoulder * shoulder, 0.0))


class TheFourRegistriesAgreeTests(unittest.TestCase):
    """The gate is `UR_MODEL_KEYS`; the other three have to cover exactly it."""

    def test_there_is_more_than_one_model(self) -> None:
        """A comparison over one model, or none, passes for the wrong reason."""
        self.assertGreaterEqual(len(UR_MODEL_KEYS), 6, f"only found {UR_MODEL_KEYS}")

    def test_every_configurable_model_has_link_lengths(self) -> None:
        missing = sorted(set(UR_MODEL_KEYS) - set(UR_DH_TABLES_M))
        self.assertEqual(
            missing, [],
            "these models can be configured and have no DH row, so the exact-mesh guard reports "
            "unknown_model and drops to capsules, the arm-against-arm capsules vanish, and the reach "
            "helper reports a shoulder height of 0.0. None of that raises.",
        )

    def test_every_configurable_model_has_joint_limits(self) -> None:
        missing = sorted(set(UR_MODEL_KEYS) - set(UR_JOINT_LIMITS_DEG))
        self.assertEqual(missing, [], "resolve_joint_limits_deg returns None, which callers must "
                                      "treat as guard unavailable rather than as no limit")

    def test_every_configurable_model_has_a_spec(self) -> None:
        missing = sorted(set(UR_MODEL_KEYS) - set(_UR_MODELS))
        self.assertEqual(missing, [], "no asset path, no reach, no workspace patch")

    def test_no_spec_exists_for_a_model_nobody_can_configure(self) -> None:
        """⭐ THE OTHER DIRECTION, and the one that used to be silent. A spec for a key the config
        refuses is dead weight that reads as support."""
        extra = sorted(set(_UR_MODELS) - set(UR_MODEL_KEYS))
        self.assertEqual(extra, [])

    def test_the_dh_table_may_be_wider_but_not_narrower(self) -> None:
        """The DH table is allowed to carry kinematics for a model nobody has stood up yet, because a
        row is a physical fact rather than a claim of support. The gate is what admits it."""
        self.assertTrue(set(UR_MODEL_KEYS).issubset(set(UR_DH_TABLES_M)))


class EveryModelDescribesItselfTests(unittest.TestCase):
    def test_no_two_models_share_link_lengths(self) -> None:
        """⭐ THE DISCRIMINATING CONTROL. Six entries copied from one template would pass every
        assertion above. A UR5 and a UR5e differ by 0.05 mm in one link and 73 mm in another, which
        is exactly the size of difference a copy would erase."""
        seen: dict[tuple, str] = {}
        for model in UR_MODEL_KEYS:
            rows = tuple((r.a_m, r.d_m, r.alpha_rad) for r in UR_DH_TABLES_M[model])
            self.assertNotIn(rows, seen, f"{model} has the same link lengths as {seen.get(rows)}")
            seen[rows] = model

    def test_no_two_models_share_an_asset(self) -> None:
        seen: dict[str, str] = {}
        for model, spec in _UR_MODELS.items():
            self.assertNotIn(spec.usd_relpath, seen,
                             f"{model} points at the same USD as {seen.get(spec.usd_relpath)}")
            seen[spec.usd_relpath] = model
            self.assertIn(model, spec.usd_relpath, "the asset path should name its own model")

    def test_the_spec_key_is_its_own_key(self) -> None:
        for key, spec in _UR_MODELS.items():
            self.assertEqual(key, spec.key)

    def test_reach_grows_with_the_arm(self) -> None:
        """A sanity check on the datasheet numbers themselves: a UR10 reaches further than a UR5,
        which reaches further than a UR3, in both series."""
        for series in (("ur3", "ur5", "ur10"), ("ur3e", "ur5e", "ur10e")):
            reaches = [_UR_MODELS[m].max_reach_mm for m in series]
            self.assertEqual(reaches, sorted(reaches), f"{series} reaches are not ordered")


class EveryWorkspaceFitsItsOwnArmTests(unittest.TestCase):
    """A patch says where a scene generator should place objects. One that reaches past the arm turns
    every pick into an IK failure with nothing naming the cause."""

    def test_the_far_corner_is_inside_the_horizontal_radius(self) -> None:
        for model in UR_MODEL_KEYS:
            with self.subTest(model=model):
                spec = _UR_MODELS[model]
                cx, cy = spec.workspace_center_mm
                hx, hy = spec.workspace_half_extents_mm
                corner = math.hypot(abs(cx) + hx, abs(cy) + hy)
                limit = _horizontal_radius_mm(model)
                self.assertLessEqual(
                    corner, limit,
                    f"{model}'s patch reaches {corner:.1f} mm and the arm reaches {limit:.1f} mm at "
                    f"table height, so its far corner is outside the working sphere",
                )

    def test_a_bigger_arm_is_not_given_a_smaller_patch(self) -> None:
        """⚠ The direction nothing else catches. A patch too LARGE fails the check above; a patch too
        small wastes the arm and passes everything. `ur10e` declared 1300 mm of reach and inherited
        the UR5e patch by default until 2026-09-09, so it used half the arm it had."""
        for series in (("ur3", "ur5", "ur10"), ("ur3e", "ur5e", "ur10e")):
            areas = [
                _UR_MODELS[m].workspace_half_extents_mm[0] * _UR_MODELS[m].workspace_half_extents_mm[1]
                for m in series
            ]
            self.assertEqual(areas, sorted(areas), f"{series} patches shrink as the arm grows")


class TheAssetSaysWhetherAGripperHasToBeMountedTests(unittest.TestCase):
    """MEASURED across the six assets: four of them offer no Robotiq at all, and a cell on one of
    those without `gripper_mount` comes up as a bare arm and fails much later inside the gripper
    driver rather than at build."""

    #: The models whose Isaac asset carries no Robotiq variant. `ur10`'s Gripper set holds only
    #: suction tools, which is why it is here despite having a set at all.
    _BARE = frozenset({"ur3", "ur3e", "ur5", "ur10"})

    def test_the_bare_models_declare_no_variant(self) -> None:
        for model in sorted(self._BARE):
            with self.subTest(model=model):
                self.assertIsNone(_UR_MODELS[model].baked_gripper_variant)

    def test_the_others_declare_one(self) -> None:
        """The control. Without it `_BARE` could name every model and the test above would pass."""
        for model in sorted(set(UR_MODEL_KEYS) - self._BARE):
            with self.subTest(model=model):
                self.assertEqual(_UR_MODELS[model].baked_gripper_variant, "Robotiq_2f_85")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class ANewArmIsNotGivenAnotherArmsGeometryTests(unittest.TestCase):
    """A variant bundle carries the arm meshes of whatever robot it was baked from, and the check
    that catches a mismatch compares one arm link against the model's OWN bundle.

    ⛔ So the check is blind exactly when an arm is new. MEASURED on 2026-09-09, before the fix:
    `ur10e` with `collision_mesh_variant: schunk_egu50` reported `ok` and checked a UR10e against
    UR5e arm meshes, while the same variant on a `ur3e` correctly reported `variant_model_mismatch`.
    The only difference was that ur3e has a bundle to be compared with.

    Two individually sensible rules produced it: the resolver falls back to the flat
    `{variant}_collision_meshes.npz` when no per-arm file exists, and the mismatch check treats an
    unanswerable question as "not a mismatch" so that a working cell is not sent to the capsule proxy
    over a bundle that may be perfectly correct. The flat name is now offered only where its arm CAN
    be checked.
    """

    def _status(self, model: str, variant: str | None) -> str:
        from src.robot.safety._fcl_self_collision import mesh_backend_status

        return mesh_backend_status(model, None, variant)

    @staticmethod
    def _models_without_a_bundle() -> list[str]:
        """Every configurable model carrying no bundle of its own, whatever the reason.

        DERIVED, and widened TWICE by its own failures. It first named ur3, ur5, ur10 and
        ur10e, true for the few hours between the fallback hole being closed and those arms
        being baked. It then read UR_MODEL_KEYS, which held until ur10 was baked on
        2026-09-10 and every configurable model had one, leaving the loop empty. It reads the
        DH table now, which is wider than the config gate on purpose: ur16e has kinematics
        and no bundle, and a model in that state is what keeps this test able to fail.
        """
        from src.robot.safety._ur_kinematics import UR_DH_TABLES_M
        from src.robot.safety.planning.environment import collision_mesh_bundle

        return [m for m in sorted(UR_DH_TABLES_M) if not collision_mesh_bundle(m).is_file()]

    def test_a_model_with_no_bundle_is_told_so_rather_than_given_another_arms(self) -> None:
        without = self._models_without_a_bundle()
        self.assertTrue(without, "every model has a bundle, so this test can no longer fail. "
                                 "Delete it rather than let it pass over an empty set.")
        for model in without:
            with self.subTest(model=model):
                self.assertIn(
                    self._status(model, "schunk_egu50"),
                    {"no_bundle"},
                    f"{model} has no bundle of its own, so `ok` here would mean the guard is "
                    f"checking this arm's joint angles against ANOTHER arm's link meshes. "
                    f"`no_bundle` is the whole answer: a bundle can be baked for any arm this "
                    f"stack knows, from its USD or from its URDF package.",
                )

    def test_the_two_cases_that_were_already_right_still_are(self) -> None:
        """⭐ THE CONTROL. Closing the hole must not turn every variant into `no_bundle`: a model
        with its own bundle can still be compared, and both answers have to survive."""
        self.assertEqual(self._status("ur5e", "schunk_egu50"), "ok")
        self.assertEqual(self._status("ur3e", "schunk_egu50"), "variant_model_mismatch")

    def test_every_arm_with_a_bundle_carries_the_cell_gripper(self) -> None:
        """⭐ THE EDGE A NEW ARM NEEDS. The Hand-E is the gripper for this cell, so an arm that can
        be planned exactly has to carry it, and an arm bundle without its hand is worse than no
        arm bundle: the cell plans exactly while bare and drops to the capsule proxy the moment
        somebody names the hand actually bolted to it. Nothing else notices, because a bare arm
        plans perfectly well.

        MEASURED 2026-09-09: three arms were baked and their Hand-E variants followed as a
        separate step, which is exactly the window this closes.
        """
        from src.robot.safety.planning.environment import collision_mesh_bundle

        with_bundle = [m for m in UR_MODEL_KEYS if collision_mesh_bundle(m).is_file()]
        self.assertGreaterEqual(len(with_bundle), 2, "a check over one arm proves nothing")
        for model in with_bundle:
            with self.subTest(model=model):
                self.assertEqual(
                    self._status(model, "robotiq_hande"), "ok",
                    f"{model} has exact arm geometry and no Hand-E on it. Bake it: "
                    f"python scripts/grippers/bake_gripper_variant.py robotiq_hande "
                    f"--arm {model} --write",
                )
