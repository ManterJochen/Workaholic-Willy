"""The hand is its own link of the planner's robot, placed by the declared tool frame (UM lane S10).

A cuRobo descriptor used to carry the hand on tool0: build_ur_config.py moved the map's spheres one plate along +Y and
wrote one file per arm and hand. The hand now stays its committed sphere map, in the hand model's own frame, and becomes
a fixed link named ``hand`` under ``tool0`` whose transform is the placement: rotation R_TM, translation the plate along
the placed approach. That is where the exact mesh guard puts the hand's vertices, so the two engines hold one hand.

Nothing starts a planner with it in this step; the composition is what is tested. The pair set is compared with the
table the legacy Hand-E descriptor holds, so reading ``hand`` as ``tool0`` must check exactly what that file checked.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import unittest

import numpy as np
import yaml

from src.geometry.quaternion import to_rotation_matrix

_SIM = {"source": "willy", "offset_mm": [0.0, 132.0, 0.0], "rotation_quat_xyzw": [-0.70710678, 0.0, 0.0, 0.70710678]}
_UR = {"source": "polyscope", "offset_mm": [0.0, 0.0, 155.75], "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}
_ARM = ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]

#: The legacy table, copied from ext_deps/curobo/curobo/content/configs/robot/ur5e_robotiq_hande.yml:10-17 and
#: :847-863 as build_ur_config.py wrote it on 2026-09-15 (the content directory is not in the repository).
_LEGACY_LINKS = [*_ARM, "tool0"]
_LEGACY_IGNORE = {
    "camera_mount": ["tool0", "wrist_3_link"],
    "upper_arm_link": ["forearm_link", "shoulder_link"],
    "wrist_1_link": ["wrist_2_link", "wrist_3_link"],
    "wrist_2_link": ["wrist_3_link", "tool0"],
    "wrist_3_link": ["tool0"],
    "forearm_link": ["wrist_1_link"],
}


def _arm_only() -> dict:
    """An arm descriptor with no hand in it, as S11's builder writes one: no tool0 link, no tool0 spheres, no hand key."""
    ignore = {link: list(names) for link, names in _LEGACY_IGNORE.items() if link != "camera_mount"}
    return {
        "_provenance": {"arm": "ur5e", "carries_hand": False},
        "robot_cfg": {"kinematics": {
            "urdf_path": "robot/ur_description/ur5e.urdf",
            "base_link": "base_link",
            "tool_frames": ["tool0"],
            "collision_link_names": list(_ARM),
            "collision_spheres": {link: [{"center": [0.0, 0.0, 0.05], "radius": 0.04}] for link in _ARM},
            "collision_sphere_buffer": 0.0,
            "self_collision_buffer": {**{link: 0.0 for link in _ARM}, "shoulder_link": 0.07},
            "self_collision_ignore": ignore,
        }},
    }


def _hand(model: str, frame: dict, plates: "list[float] | None" = None):
    from src.config.schema.robot import RobotConfig
    from src.robot.safety.planning.hand import planner_hand

    gripper: dict = {"model": model, "tool_frame": frame}
    if plates is not None:
        gripper["coupling_plates"] = [{"name": f"plate_{i}", "thickness_mm": float(mm)} for i, mm in enumerate(plates)]
    return planner_hand(RobotConfig.model_validate({"vendor": "ur", "gripper": gripper}))


def _link(model: str, frame: dict, plates: "list[float] | None" = None):
    from src.robot.safety.planning.body_link import HandLink

    return HandLink.from_hand(_hand(model, frame, plates))


def _pairs(kinematics: dict, rename: "dict[str, str] | None" = None) -> set:
    """The link pairs cuRobo checks, by its rule (self_collision_params.py:170-205), renamed only afterwards.

    Renaming before the ignore lists are read would be wrong: the arm table's ``wrist_2_link: [..., tool0]`` would then
    cover a hand that no longer ignores wrist_2, and the control below could never fail.
    """
    rename = rename or {}
    links = kinematics["collision_link_names"]
    ignore = kinematics["self_collision_ignore"]
    checked = [(a, b) for a, b in itertools.combinations(links, 2)
               if b not in ignore.get(a, ()) and a not in ignore.get(b, ())]
    return {frozenset((rename.get(a, a), rename.get(b, b))) for a, b in checked}


def _compose(config: dict, bodies: list, margin_mm: float = 0.0, attach: int = 0) -> dict:
    from src.robot.safety.planning._curobo_body_links import compose_sidecar_config

    return compose_sidecar_config(config, bodies=bodies, margin_mm=margin_mm, attach_spheres=attach)


class TheHandIsAFixedLinkUnderTool0Tests(unittest.TestCase):
    def test_the_hand_e_behind_a_20_mm_plate(self) -> None:
        link = _link("robotiq_hande", _SIM, [20.0])
        source = _arm_only()
        before = copy.deepcopy(source)
        kin = _compose(source, [link.to_dict()])["robot_cfg"]["kinematics"]
        committed = yaml.safe_load(_hand("robotiq_hande", _SIM, [20.0]).sphere_map.read_text(encoding="utf-8"))
        self.assertEqual(kin["collision_link_names"][-1], "hand")
        self.assertEqual(kin["collision_spheres"]["hand"], committed["collision_spheres"]["tool0"])
        self.assertEqual(kin["extra_links"]["hand"], {
            "link_name": "hand", "parent_link_name": "tool0", "joint_name": "hand_joint", "joint_type": "FIXED",
            "fixed_transform": [0.0, 0.02, 0.0, 1.0, 0.0, 0.0, 0.0],
        })
        self.assertEqual(kin["self_collision_buffer"]["hand"], 0.0)
        self.assertEqual(source, before, "the arm descriptor is never mutated")

    def test_the_link_says_what_it_is(self) -> None:
        link = _link("robotiq_hande", _UR, [20.0])
        text = link.render()
        text.encode("ascii")
        self.assertFalse(text.endswith("\n"))
        self.assertIn("+Z", text)
        json.dumps(link.to_dict())

    def test_a_refused_placement_has_no_link(self) -> None:
        from src.robot.safety.planning._curobo_body_links import BodyLinkError

        half = math.radians(30.0) / 2.0
        oblique = dict(_UR, rotation_quat_xyzw=[math.sin(half), 0.0, 0.0, math.cos(half)])
        with self.assertRaises(BodyLinkError) as ctx:
            _link("robotiq_hande", oblique, [20.0])
        self.assertIn("30 degrees", str(ctx.exception))


class WhatABodyLinkRefusesTests(unittest.TestCase):
    """⭐ THE CONTROLS. Each case changes one thing about a composition that succeeds."""

    def setUp(self) -> None:
        from src.robot.safety.planning._curobo_body_links import BodyLinkError

        self.error = BodyLinkError
        self.body = _link("robotiq_hande", _SIM, [20.0]).to_dict()

    def test_the_composition_the_controls_change_succeeds(self) -> None:
        self.assertIn("hand", _compose(_arm_only(), [self.body])["robot_cfg"]["kinematics"]["extra_links"])

    def test_a_descriptor_that_already_carries_a_hand(self) -> None:
        spheres = _arm_only()
        spheres["robot_cfg"]["kinematics"]["collision_spheres"]["tool0"] = [{"center": [0, 0, 0], "radius": 0.04}]
        keyed = _arm_only()
        keyed["_provenance"]["gripper_key"] = "robotiq_hande"
        for label, config in (("tool0 spheres", spheres), ("gripper_key", keyed)):
            with self.subTest(label), self.assertRaises(self.error):
                _compose(config, [self.body])

    def test_a_hand_that_ignores_wrist_1(self) -> None:
        body = dict(self.body, ignore=[*self.body["ignore"], "wrist_1_link"])
        with self.assertRaises(self.error):
            _compose(_arm_only(), [body])

    def test_a_link_that_exists_even_only_as_a_stale_ignore_key(self) -> None:
        stale = _arm_only()
        stale["robot_cfg"]["kinematics"]["self_collision_ignore"]["camera_mount"] = ["tool0", "wrist_3_link"]
        camera = dict(self.body, link="camera_mount", joint="camera_joint")
        for label, config, bodies in (("twice", _arm_only(), [self.body, self.body]),
                                      ("stale ignore key", stale, [camera])):
            with self.subTest(label), self.assertRaises(self.error):
                _compose(config, bodies)

    def test_a_parent_that_is_not_the_tool_frame_or_an_earlier_body(self) -> None:
        with self.assertRaises(self.error):
            _compose(_arm_only(), [dict(self.body, parent="wrist_3_link", ignore=["wrist_3_link", "wrist_2_link"])])

    def test_an_ignore_list_without_the_parent(self) -> None:
        with self.assertRaises(self.error):
            _compose(_arm_only(), [dict(self.body, ignore=["wrist_3_link", "wrist_2_link"])])

    def test_a_body_with_no_spheres_and_no_slots(self) -> None:
        with self.assertRaises(self.error):
            _compose(_arm_only(), [dict(self.body, spheres=[], slots=0)])


def _wxyz_matrix(wxyz) -> np.ndarray:
    w, x, y, z = (float(v) for v in wxyz)
    return to_rotation_matrix(np.array([x, y, z, w], dtype=np.float64))


def _xyzw_misread(wxyz) -> np.ndarray:
    return to_rotation_matrix(np.array([float(v) for v in wxyz], dtype=np.float64))


class TheHandSitsWhereTheGuardPutsItTests(unittest.TestCase):
    """The fixed transform applied to the map equals R_TM (p + plate along model y), with R_TM and the plate taken from
    the resolved hand and the rotation oracle computed here, not by the code under test."""

    CASES = (("robotiq_2f85", None), ("schunk_egu50", None), ("robotiq_hande", []), ("robotiq_hande", [20.0]))

    def test_every_hand_under_both_frames(self) -> None:
        for (model, plates), (frame_name, frame) in itertools.product(self.CASES, (("sim", _SIM), ("ur", _UR))):
            with self.subTest(model=model, plates=plates, frame=frame_name):
                hand = _hand(model, frame, plates)
                link = _link(model, frame, plates)
                p = np.array([c for c, _ in link.spheres], dtype=np.float64)
                plate = hand.coupling_mm / 1000.0 if hand.origin == "mounting_face" else 0.0
                r_tm = np.array(hand.placement.rotation, dtype=np.float64)
                expected = (r_tm @ (p + plate * np.array([0.0, 1.0, 0.0])).T).T
                t = np.array(link.fixed_transform[:3], dtype=np.float64)
                got = (_wxyz_matrix(link.fixed_transform[3:]) @ p.T).T + t
                self.assertLessEqual(float(np.max(np.abs(got - expected))), 1e-12)
                misread = (_xyzw_misread(link.fixed_transform[3:]) @ p.T).T + t
                self.assertGreater(float(np.max(np.abs(misread - expected))), 1e-3, "the quaternion order is untested")
                if plate:
                    flipped = (r_tm @ (p - plate * np.array([0.0, 1.0, 0.0])).T).T
                    self.assertGreater(float(np.max(np.abs(got - flipped))), 1e-3, "the plate sign is untested")

    def test_the_ur_frame_turns_the_hand_onto_z(self) -> None:
        link = _link("robotiq_hande", _UR, [20.0])
        self.assertEqual(link.fixed_transform[:3], (0.0, 0.0, 0.02))


class TheOrderPutsTheHandInEveryTableTests(unittest.TestCase):
    def test_the_margin_and_the_payload_reach_the_hand(self) -> None:
        body = _link("robotiq_hande", _SIM, [20.0]).to_dict()
        kin = _compose(_arm_only(), [body], margin_mm=10.0, attach=16)["robot_cfg"]["kinematics"]
        self.assertEqual(kin["self_collision_buffer"]["hand"], 0.005)
        self.assertIn("attached_object", kin["self_collision_ignore"]["hand"])

    def test_the_margin_before_the_body_leaves_the_hand_without_it(self) -> None:
        """The control for the order."""
        from src.robot.safety.planning._curobo_body_links import apply_body_links
        from src.robot.safety.planning._curobo_margin import apply_self_collision_margin

        body = _link("robotiq_hande", _SIM, [20.0]).to_dict()
        margined, _ = apply_self_collision_margin(_arm_only(), 10.0)
        composed, _ = apply_body_links(margined, [body])
        self.assertEqual(composed["robot_cfg"]["kinematics"]["self_collision_buffer"]["hand"], 0.0)

    def test_hand_read_as_tool0_checks_what_the_legacy_descriptor_checked(self) -> None:
        body = _link("robotiq_hande", _SIM, [20.0]).to_dict()
        kin = _compose(_arm_only(), [body], margin_mm=10.0)["robot_cfg"]["kinematics"]
        legacy = _pairs({"collision_link_names": _LEGACY_LINKS, "self_collision_ignore": _LEGACY_IGNORE})
        self.assertEqual(_pairs(kin, rename={"hand": "tool0"}), legacy)
        self.assertIn(frozenset(("tool0", "wrist_1_link")), legacy, "the hand is checked against wrist_1")

    def test_a_hand_that_checks_wrist_2_changes_the_set(self) -> None:
        """The control: the comparison can fail."""
        body = _link("robotiq_hande", _SIM, [20.0]).to_dict()
        body["ignore"] = [name for name in body["ignore"] if name != "wrist_2_link"]
        kin = _compose(_arm_only(), [body])["robot_cfg"]["kinematics"]
        legacy = _pairs({"collision_link_names": _LEGACY_LINKS, "self_collision_ignore": _LEGACY_IGNORE})
        self.assertNotEqual(_pairs(kin, rename={"hand": "tool0"}), legacy)


class WhatTheSidecarSaysOfTheHandTests(unittest.TestCase):
    def test_the_report_names_what_was_loaded(self) -> None:
        """The hash in the report is the link's own, over the same spheres, and the buffer is the margined one."""
        from src.robot.safety.planning._curobo_body_links import body_report

        link = _link("robotiq_hande", _SIM, [20.0])
        composed = _compose(_arm_only(), [link.to_dict()], margin_mm=10.0, attach=16)
        (row,) = body_report(composed, ["hand"])
        self.assertEqual(row["spheres_sha256"], link.spheres_sha256)
        self.assertEqual((row["parent"], row["spheres"], row["slots"]), ("tool0", len(link.spheres), 0))
        self.assertEqual(row["buffer_m"], 0.005)
        self.assertEqual(row["fixed_transform"], list(link.fixed_transform))

    def test_a_template_that_guards_tool0_without_spheres_is_refused(self) -> None:
        """cuRobo raises on a collision link with no spheres, so the composition refuses before it would."""
        from src.robot.safety.planning._curobo_body_links import BodyLinkError

        template = _arm_only()
        template["robot_cfg"]["kinematics"]["collision_link_names"].append("tool0")
        with self.assertRaises(BodyLinkError):
            _compose(template, [_link("robotiq_hande", _SIM, [20.0]).to_dict()])


class TheScriptsSendTheSameHandTests(unittest.TestCase):
    """⭐ ONE DERIVATION, TWO CALLERS. The probes and the descriptor check run under the cuRobo interpreter, where this
    repository cannot be imported, so they read the committed map themselves (`scripts/curobo/_hand_body.py`). What they
    build must be the body a cell's planner sends, or the box would measure a hand no cell plans with. The sphere fit
    was written twice once already, one copy calling itself a mirror of the other, with nothing checking.
    """

    def setUp(self) -> None:
        import importlib.util
        import pathlib
        import sys

        path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "curobo" / "_hand_body.py"
        spec = importlib.util.spec_from_file_location("_hand_body_under_test", path)
        assert spec is not None and spec.loader is not None
        self.helper = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.helper
        spec.loader.exec_module(self.helper)

    def test_every_hand_and_frame_matches_the_cells_link(self) -> None:
        wire_keys = ("link", "parent", "joint", "fixed_transform", "spheres", "slots", "buffer_m", "ignore")
        frames = (("sim", _SIM, self.helper.SIM_TOOL_ROTATION_XYZW), ("ur", _UR, (0.0, 0.0, 0.0, 1.0)))
        for (model, plates), (name, frame, quaternion) in itertools.product(
                TheHandSitsWhereTheGuardPutsItTests.CASES, frames):
            with self.subTest(model=model, plates=plates, frame=name):
                link = _link(model, frame, plates).to_dict()
                body = self.helper.hand_body(
                    model, coupling_mm=None if plates is None else float(sum(plates)), rotation_xyzw=quaternion,
                )
                self.assertEqual(body, {key: link[key] for key in wire_keys})

    def test_the_plate_rules_are_the_cells(self) -> None:
        """The control: a mounting face map without plates and a flange map with one are refused, as a cell refuses."""
        for label, model, coupling in (("mounting face, no plate", "robotiq_hande", None),
                                       ("flange, with a plate", "robotiq_2f85", 20.0)):
            with self.subTest(label), self.assertRaises(self.helper.HandBodyError):
                self.helper.hand_body(model, coupling_mm=coupling)


if __name__ == "__main__":
    unittest.main()
