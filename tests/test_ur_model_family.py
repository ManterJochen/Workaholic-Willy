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
from pathlib import Path

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
    """MEASURED across the six assets: four of them offer no Robotiq at all, so on those the sim
    mounts the hand standalone (`sim_mount_for`). A cell that got no mount there would come up as a
    bare arm and fail much later inside the gripper driver rather than at build."""

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
    `ur10e` with the `schunk_egu50` hand's bundle reported `ok` and checked a UR10e against
    UR5e arm meshes, while the same variant on a `ur3e` correctly reported `variant_model_mismatch`.
    The only difference was that ur3e has a bundle to be compared with.

    Two individually sensible rules produced it: the resolver falls back to the flat
    `{variant}_collision_meshes.npz` when no per-arm file exists, and the mismatch check treats an
    unanswerable question as "not a mismatch" so that a working cell is not sent to the capsule proxy
    over a bundle that may be perfectly correct. The flat name is now offered only where its arm CAN
    be checked.
    """

    def _status(self, model: str, variant: "str | None", folder: "Path | None" = None) -> str:
        from src.robot.safety._fcl_self_collision import mesh_backend_status

        return mesh_backend_status(model, str(folder) if folder is not None else None, variant)

    def _without_one_arm(self, missing: str) -> "Path":
        """A mesh directory holding every bundle except ``missing``, so a NEW arm can be asked about.

        This used to read the repository: an arm with kinematics and no bundle was taken from whatever the
        tree happened to be missing, first ur3 and ur5, then ur10, then ur16e. Each time those were baked the
        set emptied and the test could no longer fail, which is the failure mode the assertion below names out
        loud. A bundle removed on purpose cannot empty.
        """
        import shutil
        import tempfile

        from src.robot.safety.planning.environment import COLLISION_MESH_DIR

        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        for source in COLLISION_MESH_DIR.glob("*.npz"):
            if source.name != f"{missing}_collision_meshes.npz":
                shutil.copy2(source, folder / source.name)
        return folder

    def test_a_model_with_no_bundle_is_told_so_rather_than_given_another_arms(self) -> None:
        from src.robot.safety._ur_kinematics import UR_DH_TABLES_M
        from src.robot.safety.planning.environment import collision_mesh_bundle

        present = [m for m in sorted(UR_DH_TABLES_M) if collision_mesh_bundle(m).is_file()]
        self.assertTrue(present, "no arm has a bundle, so nothing here could be taken away")
        for model in present:
            folder = self._without_one_arm(model)
            with self.subTest(model=model):
                self.assertFalse((folder / f"{model}_collision_meshes.npz").is_file(),
                                 "the copy still holds it")
                self.assertEqual(
                    self._status(model, "schunk_egu50", folder), "no_bundle",
                    f"{model} has no bundle of its own here, so `ok` would mean the guard is checking this "
                    f"arm's joint angles against ANOTHER arm's link meshes. `no_bundle` is the whole answer: "
                    f"a bundle can be baked for any arm this stack knows, from its USD or from UR's own STLs.",
                )

    def test_taking_nothing_away_leaves_the_arm_answerable(self) -> None:
        """⭐ THE CONTROL that the case above is made by the missing file and not by the copy itself."""
        folder = self._without_one_arm("nothing_is_called_this")
        self.assertTrue((folder / "ur5e_collision_meshes.npz").is_file())
        self.assertEqual(self._status("ur5e", "schunk_egu50", folder), "ok")

    def test_the_two_cases_that_were_already_right_still_are(self) -> None:
        """⭐ THE CONTROL. Closing the hole must not turn every variant into `no_bundle`: a model
        with its own bundle can still be compared, and both answers have to survive.

        Until UM lane S22 the second answer was the guard refusing the EGU-50 on a ur3e by an inherited list.
        Admission is measured now, so the guard composes both and the EVIDENCE is what tells them apart: the
        matrix measured the EGU-50 on the ur3e, and could measure nothing for it on the ur5, which has no pose.
        """
        from src.robot.safety.planning.evidence import evidence_path

        self.assertEqual(self._status("ur5e", "schunk_egu50"), "ok")
        self.assertEqual(self._status("ur3e", "schunk_egu50"), "ok")
        common = {"hand": "schunk_egu50", "coupling_mm": 0.0, "approach": "+Y", "closing": "+X",
                  "planner_margin_mm": 4.0, "attach_spheres": 0}
        self.assertTrue(evidence_path(arm="ur3e", **common).is_file())  # type: ignore[arg-type]
        self.assertFalse(evidence_path(arm="ur5", **common).is_file())  # type: ignore[arg-type]

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
                    f"{model} has exact arm geometry and the guard cannot compose the Hand-E onto it, "
                    f"which would drop a cell carrying it to the capsule proxy",
                )
