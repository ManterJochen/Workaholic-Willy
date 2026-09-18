"""A planner starts only on a combination somebody measured, and the inherited admission list is gone (S22).

S21 committed 23 measured combinations. This step makes them load-bearing: both drivers refuse to keep a sidecar whose
combination has no file, a file below b1, or a file measured on another robot, and `hand__admitted_arms` leaves the
tree in the same step, so two admission mechanisms never stand side by side.

⭐ **THE CHECK HAS TWO HALVES, BECAUSE A DESK HAS NO SIDECAR.** Three of the hashes a file binds are what a running
sidecar reports about what it loaded; a checklist read before anything starts cannot know them. So the file half
(which file, which rung, which guard geometry, which hand map, which guard margin) runs at the desk AND at start, and
the sidecar half runs only at start. The row says which half it checked rather than pretending to both.

⛔ **RETIRING THE LIST WIDENS THE EGU-50, AND THAT IS THE MEASUREMENT TALKING.** The list admitted the EGU-50 on the
ur5e alone. The matrix measured it at b1 on the ur3, ur3e, ur5e, ur10 and ur10e with zero false clears, so those five
are admitted now and the ur5 still is not, because nothing could be measured for it.

The owner decided on 2026-09-17 that a planner margin below the guard's is OK when a file measured it there: every
shipped cell runs 4 mm against a 10 mm guard, and a warning on every cell forever teaches an operator to read past
warnings.
"""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.config.schema.robot import RobotConfig
from src.contracts import UNSET, chosen
from src.robot.execution.real_cell.preflight import CheckStatus, run_config_preflight
from src.robot.safety.planning.body_link import HandLink
from src.robot.safety.planning.curobo_client import SidecarIdentity
from src.robot.safety.planning.evidence import (
    EVIDENCE_DIR,
    CombinationEvidence,
    desk_evidence_refusal,
    planner_evidence_refusal,
)
from src.robot.safety.planning.hand import planner_hand

_TOOL = {
    "source": "willy", "offset_mm": (0.0, 132.0, 0.0),
    "rotation_quat_xyzw": (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
}
_SIM_FILE = "ur5e_robotiq_2f85_c0mm_+Y+X_m4mm_a0.json"


def _cell(**over: object) -> RobotConfig:
    """A real UR cell whose combination the matrix measured: ur5e, 2F-85, +Y+X, 4 mm against a 10 mm guard."""
    tree: dict = {
        "vendor": "ur", "ur": {"model": "ur5e", "motion_planner": "curobo"},
        "gripper": {"model": "robotiq_2f85", "tool_frame": _TOOL},
        "safety": {"self_collision": {"kinematics_model": "ur5e", "planner_margin_mm": 4.0, "backend": "fcl"}},
    }
    for key, value in over.items():
        section, _, field = key.partition("__")
        tree.setdefault(section, {})
        if "__" in field:
            inner, _, leaf = field.partition("__")
            tree[section].setdefault(inner, {})[leaf] = value
        else:
            tree[section][field] = value
    return RobotConfig.model_validate(tree)


def _committed(name: str = _SIM_FILE) -> CombinationEvidence:
    import json

    return CombinationEvidence.from_dict(json.loads((EVIDENCE_DIR / name).read_text(encoding="utf-8")))


def _identity_of(evidence: CombinationEvidence, **over: object) -> SidecarIdentity:
    """What a sidecar that loaded exactly the measured robot reports, with any field changed on purpose."""
    fields: dict = {
        "provenance": {"arm": evidence.arm, "carries_hand": False},
        "arm_descriptor_sha256": evidence.arm_descriptor_sha256,
        "urdf_sha256": evidence.urdf_sha256,
        "composed_sha256": evidence.composed_sha256,
        "bodies": ({"link": "hand", "parent": "tool0"},),
    }
    fields.update(over)
    return SidecarIdentity(**fields)


class TheFileHalfRunsAtTheDesk(unittest.TestCase):

    def test_the_measured_cell_is_admitted_and_told_what_admitted_it(self) -> None:
        said, evidence = desk_evidence_refusal(_cell())
        self.assertIsNone(said)
        assert evidence is not None
        self.assertEqual(evidence.path.name, _SIM_FILE)

    def test_a_folder_without_the_file_refuses_and_names_it(self) -> None:
        """⭐ The box control, off box: M1 with its evidence file moved away has to refuse by name."""
        with TemporaryDirectory() as empty:
            said, _ = desk_evidence_refusal(_cell(), evidence_dir=Path(empty))
        assert said is not None
        self.assertIn(_SIM_FILE, said)
        self.assertIn("matrix_gate.py", said)

    def test_a_margin_nobody_measured_refuses(self) -> None:
        said, _ = desk_evidence_refusal(_cell(safety__self_collision__planner_margin_mm=6.0))
        assert said is not None
        self.assertIn("m6mm", said)

    def test_a_guard_at_another_margin_refuses(self) -> None:
        said, _ = desk_evidence_refusal(_cell(safety__self_collision__min_distance_mm=8.0))
        assert said is not None
        self.assertIn("guard", said)

    def test_a_cell_whose_plate_became_a_body_is_another_robot(self) -> None:
        """UM8's coupling body moves the guard geometry, and that is exactly what the guard hash sees."""
        with TemporaryDirectory() as folder:
            shutil.copy(EVIDENCE_DIR / "ur5e_robotiq_hande_c20mm_+Y+X_m4mm_a0.json", folder)
            plain = _cell(gripper__model="robotiq_hande",
                          gripper__coupling_plates=[{"name": "adapter", "thickness_mm": 20.0}])
            bodied = _cell(gripper__model="robotiq_hande",
                           gripper__coupling_plates=[{"name": "adapter", "thickness_mm": 20.0,
                                                      "cross_section_mm": [31.5, 31.5]}])
            self.assertIsNone(desk_evidence_refusal(plain, evidence_dir=Path(folder))[0])
            said, _ = desk_evidence_refusal(bodied, evidence_dir=Path(folder))
        assert said is not None
        self.assertIn("guard_sha256", said)


class TheSidecarHalfRunsWhenItStarts(unittest.TestCase):

    def _asked(self, identity: SidecarIdentity, **cell: object) -> "str | None":
        """Called with the numbers a driver holds, exactly as the UR driver passes them."""
        cfg = _cell(**cell)
        hand = planner_hand(cfg)
        assert chosen(hand)
        sc = cfg.safety.self_collision
        return planner_evidence_refusal(
            identity, hand, HandLink.from_hand(hand), arm="ur5e",
            planner_margin_mm=float(sc.planner_margin_mm), guard_margin_mm=float(sc.min_distance_mm),
            attach_spheres=0, mesh_dir=sc.mesh_dir,
        )

    def test_a_sidecar_that_loaded_the_measured_robot_is_kept(self) -> None:
        self.assertIsNone(self._asked(_identity_of(_committed())))

    def test_a_sidecar_that_said_nothing_is_refused_rather_than_trusted(self) -> None:
        said = self._asked(_identity_of(_committed(), composed_sha256=UNSET))
        assert said is not None
        self.assertIn("did not say", said)

    def test_a_sidecar_that_loaded_another_config_is_refused(self) -> None:
        said = self._asked(_identity_of(_committed(), composed_sha256="d" * 64))
        assert said is not None
        self.assertIn("composed_sha256", said)

    def test_the_file_half_still_runs_at_start(self) -> None:
        """A sidecar that reports every hash correctly on a cell at an unmeasured margin is still refused."""
        said = self._asked(_identity_of(_committed()), safety__self_collision__planner_margin_mm=6.0)
        assert said is not None


class TheInheritedListIsGone(unittest.TestCase):
    """⛔ Retired in this step, so nothing can answer for a pairing the evidence did not measure."""

    def test_no_committed_hand_bundle_carries_an_admission_record(self) -> None:
        from src.robot.safety.planning.environment import COLLISION_MESH_DIR

        for path in sorted(COLLISION_MESH_DIR.glob("*_hand_meshes.npz")):
            with self.subTest(bundle=path.name):
                with np.load(path, allow_pickle=True) as held:
                    self.assertNotIn("hand__admitted_arms", held.files)

    def test_the_guard_no_longer_refuses_a_hand_by_that_list(self) -> None:
        """The EGU-50 on a ur3e was `variant_model_mismatch`; the matrix measured it at b1, so the list is not asked."""
        from src.robot.safety._fcl_self_collision import mesh_backend_status

        self.assertNotEqual(mesh_backend_status("ur3e", None, "schunk_egu50"), "variant_model_mismatch")

    def test_a_hands_provenance_no_longer_claims_arms(self) -> None:
        from src.robot.safety.planning.environment import hand_provenance

        provenance = hand_provenance("schunk_egu50")
        assert provenance is not None
        self.assertFalse(hasattr(provenance, "admitted_arms"))
        self.assertNotIn("proven on", provenance.render())


class TheChecklistSaysWhichHalfItChecked(unittest.TestCase):

    def _row(self, cfg: RobotConfig, **kw: object):
        report = run_config_preflight(cfg, curobo_available=True, collision_engine="coal", **kw)
        return next(c for c in report.checks if c.name == "planner margin")

    def test_a_measured_cell_is_ok_and_the_gap_to_the_guard_is_stated_not_warned(self) -> None:
        """Owner, 2026-09-17: 4 against 10 is measured and decided, so it is a fact in an OK row."""
        row = self._row(_cell())
        self.assertIs(row.status, CheckStatus.OK)
        self.assertIn("4 mm", row.detail)
        self.assertIn("10 mm", row.detail)
        self.assertIn(_SIM_FILE, row.detail)
        self.assertIn("when it starts", row.detail, "the sidecar half is checked at start, and the row says so")

    def test_a_cuRobo_cell_with_no_file_is_blocked_by_name(self) -> None:
        with TemporaryDirectory() as empty:
            row = self._row(_cell(), evidence_dir=Path(empty))
        self.assertIs(row.status, CheckStatus.BLOCK)
        self.assertIn(_SIM_FILE, row.detail)
        self.assertIn("matrix_gate.py", row.fix)

    def test_an_ik_cell_is_warned_that_the_exact_guard_alone_decides(self) -> None:
        """Owner, 2026-09-16: evidence refuses on cuRobo cells only; an ik cell plans nothing to compare."""
        row = self._row(_cell(ur__motion_planner="ik"))
        self.assertIs(row.status, CheckStatus.WARN)
        self.assertIn("exact", row.detail)

    def test_an_undeclared_margin_is_blocked(self) -> None:
        row = self._row(_cell(safety__self_collision__planner_margin_mm=None))
        self.assertIs(row.status, CheckStatus.BLOCK)


if __name__ == "__main__":
    unittest.main()
