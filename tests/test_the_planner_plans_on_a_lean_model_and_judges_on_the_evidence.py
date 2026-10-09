"""cuRobo plans on a lean robot where the cell asks for one, and every judgement stays on the robot the evidence names.

The owner, 2026-10-08: "wir müssen CuRobo richtig verwenden ... die Kugeln richtig fitten, wie es bei den anderen
Modellen ist". NVIDIA's own UR10e is 20 spheres and 83 sphere pairs; the owner's cell held 845 spheres and 172,414
pairs (the UR10's cover fit, the Hand-E's, the D415's box fill), about 150 times the collision arithmetic in every
iteration of every plan. ``safety.planned_motion.planning_spheres: lean`` builds the planner, its IK and its graph
planner from a planning model made the way NVIDIA's are (``scripts/curobo/fit_planning_spheres.py``), and leaves the
check behind ``check_js``, the ready gate and every evidence hash on the evidence model. This file holds on the CPU:

* the sidecar places the planning model's arm in the URDF it resolved exactly as the descriptor builder placed the
  evidence model's (``_curobo_planning.dh_to_link``, against ``_arm_spheres.link_frames`` on every UR the repository
  renders, and against the built ``willy_ur10.yml`` where this box has one);
* the committed lean maps are NVIDIA's pattern: tens of spheres, in the bundle's DH frames, holes and reach bounded as
  recorded, and the planning model checks 112 sphere pairs where today's robot checks 172,414;
* the planning config is the evidence robot in everything but its spheres and its ignored pairs, and anything else is
  refused (the sidecar then plans on the evidence model);
* the client composes the payload from the committed maps and the cell's own bodies, and plans on the evidence model,
  saying why, where it cannot; the config key is today's by default and reaches the client through the reservation;
* the sidecar, read as source: the planner from the planning model, every judgement from the evidence model.

What the lean model costs and saves in plans is the GPU's to say: ``scripts/curobo/ab_planning_spheres.py``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import textwrap
import typing
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from src.robot.safety.planning import _curobo_planning as planning
from src.robot.safety.planning._curobo_body_links import compose_for_cell, wrist_body_link
from src.robot.safety.planning._curobo_planning import ENV_PLANNING_MODEL, PlanningModelError
from src.robot.safety.planning._declared_body import Box, inflated
from src.robot.safety.planning.planning_model import (
    LEAN,
    LEAN_BOX_REACH_MM,
    PLANNING_MODELS,
    arm_planning_spheres_path,
    hand_planning_spheres_path,
    planning_payload,
)

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts" / "curobo"
_SERVER = _ROOT / "src" / "robot" / "safety" / "planning" / "curobo_planner_server.py"
_CONTENT = _ROOT / "ext_deps" / "curobo" / "curobo" / "content"
_ARM_LINKS = ("shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link")
_GUARDED = {"upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link", "hand",
            "wrist_camera_wrist"}


def _script(name: str):  # noqa: ANN202 (a module loaded by path)
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location(f"{name}_under_lean_test", _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _owner_cell():  # noqa: ANN202
    """The owner's cell as far as the planner's robot goes: a UR10, the Hand-E on a 23 mm plate, its frame declared."""
    from src.config.schema.robot import RobotConfig

    return RobotConfig.model_validate({
        "vendor": "ur", "ur": {"model": "ur10", "motion_planner": "curobo"},
        "gripper": {"model": "robotiq_hande", "coupling_plates": [{"name": "hand_adapter", "thickness_mm": 23.0}],
                    "tool_frame": {"source": "polyscope", "offset_mm": [0.0, 0.0, 157.0],
                                   "rotation_quat_xyzw": [0.0, 0.0, -0.7071068, 0.7071068]}},
    })


def _hand_body() -> dict:
    from src.robot.safety.planning.body_link import HandLink
    from src.robot.safety.planning.hand import planner_hand

    return HandLink.from_hand(planner_hand(_owner_cell())).to_dict()


def _camera_body(width_mm: float = 131.15) -> dict:
    """The owner's D415 in its enclosure (``config/cameras/d415_enclosure.yaml``), grown by the rig's 3 mm."""
    box = inflated(Box(name="wrist_camera_wrist", centre_mm=(35.0, 0.0, -12.9),
                       half_extents_mm=(width_mm / 2.0, 35.15 / 2.0, 16.0)), by_mm=3.0)
    body = wrist_body_link(rig_id="wrist", boxes=[box], rotation=np.eye(3).tolist(), translation_m=[0.0, -0.08, 0.05])
    return {**body, "rig_id": "wrist", "model": "d415_enclosure", "margin_mm": 3.0, "boxes": [box.to_dict()]}


def _descriptor() -> dict:
    """A small arm descriptor in the shape build_ur_config.py writes: six arm links, their buffers and neighbours."""
    return {
        "_provenance": {"arm": "urX", "carries_hand": False},
        "robot_cfg": {"kinematics": {
            "urdf_path": "robot/ur_description/urX.urdf", "base_link": "base_link", "tool_frames": ["tool0"],
            "collision_link_names": list(_ARM_LINKS),
            "collision_spheres": {link: [{"center": [0.0, 0.0, 0.0], "radius": 0.05}] * 3 for link in _ARM_LINKS},
            "collision_sphere_buffer": 0.0,
            "self_collision_buffer": {link: (0.07 if link == "shoulder_link" else 0.0) for link in _ARM_LINKS},
            "self_collision_ignore": {"upper_arm_link": ["forearm_link", "shoulder_link"],
                                      "forearm_link": ["wrist_1_link"], "wrist_1_link": ["wrist_2_link", "wrist_3_link"],
                                      "wrist_2_link": ["wrist_3_link", "tool0"], "wrist_3_link": ["tool0"]},
            "cspace": {"joint_names": ["j0", "j1", "j2", "j3", "j4", "j5"], "default_joint_position": [0.0] * 6},
        }},
    }


def _payload(urdf_bodies: "list[dict] | None" = None) -> dict:
    """A planning model for the small descriptor: one sphere per arm link, a hand of one, the guard's pairs ignored."""
    hand = {"link": "hand", "parent": "tool0", "joint": "hand_joint", "fixed_transform": [0, 0, 0.023, 1, 0, 0, 0],
            "spheres": [{"center": [0.0, 0.0, 0.08], "radius": 0.05}], "slots": 0, "buffer_m": 0.0,
            "ignore": ["tool0", "wrist_3_link", "wrist_2_link"]}
    return {
        "model": LEAN, "dh_m": [[0.0, 0.1273, 1.570796327], [-0.612, 0.0, 0.0], [-0.5723, 0.0, 0.0],
                                [0.0, 0.163941, 1.570796327], [0.0, 0.1157, -1.570796327], [0.0, 0.0922, 0.0]],
        "links": {link: {"frame": index + 1, "spheres": [{"center": [0.0, 0.0, 0.0], "radius": 0.06}]}
                  for index, link in enumerate(_ARM_LINKS)},
        "bodies": urdf_bodies if urdf_bodies is not None else [hand],
        "wrist_bodies": [],
        "ignore": [["upper_arm_link", "wrist_1_link"], ["forearm_link", "hand"]],
    }


def _urdf() -> str:
    return _script("_ur_description").render_urdf("ur10")


class ThePlanningModelIsPlacedAsTheBuilderPlacesTests(unittest.TestCase):
    def test_every_link_of_every_rendered_ur_lands_where_the_builder_puts_it(self) -> None:
        """⭐ The sidecar's own placement (standard library) is the builder's (numpy), on every UR it renders."""
        description = _script("_ur_description")
        builder = _script("_arm_spheres")
        for model in ("ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e", "ur16e"):
            urdf, rows = description.render_urdf(model), description.dh_rows(model)
            theirs, ours = builder.link_frames(urdf, rows), planning.dh_to_link(urdf, rows)
            for frame, link in enumerate(_ARM_LINKS, start=1):
                with self.subTest(model=model, link=link):
                    np.testing.assert_allclose(np.asarray(ours(link, frame)), np.asarray(theirs(link, frame)),
                                               rtol=0.0, atol=1e-12)

    def test_the_committed_cover_fit_placed_so_is_the_built_descriptor_on_this_box(self) -> None:
        """Where this box built ``willy_ur10.yml`` (from Isaac's UR10, a frame family of its own), the evidence
        model's arm spheres are the committed cover fit placed exactly this way: the planning model lands in the
        frames the evidence model is in."""
        built = _CONTENT / "configs" / "robot" / "willy_ur10.yml"
        urdf = _CONTENT / "assets" / "robot" / "ur_description" / "ur10.urdf"
        if not built.is_file() or not urdf.is_file():
            self.skipTest("no built willy_ur10.yml on this box")
        descriptor = yaml.safe_load(built.read_text(encoding="utf-8"))
        fit = yaml.safe_load((_ROOT / "src/robot/safety/planning/robot/ur10_arm_spheres.yml").read_text(encoding="utf-8"))
        place = planning.dh_to_link(urdf.read_text(encoding="utf-8"), _script("_ur_description").dh_rows("ur10"))
        for body, block in fit["collision_spheres"].items():
            placed = planning._placed(block["spheres"], place(f"{body}_link", int(block["frame"])))  # noqa: SLF001
            held = descriptor["robot_cfg"]["kinematics"]["collision_spheres"][f"{body}_link"]
            with self.subTest(body=body):
                self.assertEqual(len(held), len(placed))
                np.testing.assert_allclose([s["center"] for s in placed], [s["center"] for s in held], atol=1e-9)

    def test_a_urdf_that_cannot_place_anything_is_refused_by_name(self) -> None:
        for text in ("<robot>", "<robot name='r'/>", "<robot><joint name='a'><parent link='x'/></joint></robot>"):
            with self.subTest(text=text), self.assertRaises(PlanningModelError):
                planning.dh_to_link(text, [[0.0, 0.0, 0.0]] * 6)("shoulder_link", 1)


class TheCommittedLeanMapsAreNvidiasPatternTests(unittest.TestCase):
    def test_every_arm_map_is_tens_of_spheres_in_its_bundles_dh_frames(self) -> None:
        """NVIDIA's ur10e holds 19 spheres on its arm; a hundred would no longer be its pattern. The owner's UR10 is one
        of them, and every other committed arm is held to the same."""
        from src.robot.safety._ur_kinematics import UR_DH_TABLES_M

        arms = sorted(path.name.split("_arm_")[0] for path in arm_planning_spheres_path("ur10").parent.glob(
            f"*_arm_{LEAN}_spheres.yml"))
        self.assertIn("ur10", arms)
        for arm in arms:
            document = yaml.safe_load(arm_planning_spheres_path(arm).read_text(encoding="utf-8"))
            blocks = document["collision_spheres"]
            with self.subTest(arm=arm):
                self.assertEqual(["forearm", "shoulder", "upper_arm", "wrist_1", "wrist_2", "wrist_3"], sorted(blocks))
                total = sum(len(block["spheres"]) for block in blocks.values())
                self.assertTrue(6 <= total <= 40, f"{total} arm spheres")
                self.assertEqual([1, 2, 3, 4, 5, 6], [blocks[body]["frame"] for body in (
                    "shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3")])
                table = [[row.a_m, row.d_m, row.alpha_rad] for row in UR_DH_TABLES_M[arm]]
                np.testing.assert_allclose(document["_provenance"]["dh_m"], table, atol=1e-9,
                                           err_msg="placed with other DH rows than the bundle's")
            for row in document["_provenance"]["bodies"]:
                with self.subTest(arm=arm, body=row["body"]):
                    self.assertLessEqual(row["fresh_uncovered_max_mm"], row["hole_mm"] + 2.0, "a hole past the asked")
                    self.assertLessEqual(row["fresh_reach_max_mm"], row["reach_mm"] + 0.5, "a reach past the asked")

    def test_the_ur10_is_the_count_the_fit_recorded(self) -> None:
        document = yaml.safe_load(arm_planning_spheres_path("ur10").read_text(encoding="utf-8"))
        counts = {body: len(block["spheres"]) for body, block in document["collision_spheres"].items()}
        self.assertEqual({"shoulder": 4, "upper_arm": 9, "forearm": 10, "wrist_1": 1, "wrist_2": 1, "wrist_3": 1},
                         counts)

    def test_every_hand_map_starts_where_its_cover_map_starts(self) -> None:
        """The lean spheres hang on the evidence model's fixed transform, so they share its frame and its origin."""
        hands = sorted(path.name.split("_gripper_")[0] for path in hand_planning_spheres_path("x").parent.glob(
            f"*_gripper_{LEAN}_spheres.yml"))
        self.assertIn("robotiq_hande", hands)
        for hand in hands:
            lean = yaml.safe_load(hand_planning_spheres_path(hand).read_text(encoding="utf-8"))
            cover = yaml.safe_load((_ROOT / f"src/robot/safety/planning/robot/{hand}_gripper_spheres.yml")
                                   .read_text(encoding="utf-8"))
            spheres = lean["collision_spheres"]["tool0"]
            with self.subTest(hand=hand):
                self.assertEqual(cover["_provenance"]["origin"], lean["_provenance"]["origin"])
                self.assertTrue(4 <= len(spheres) <= 32, f"{len(spheres)} hand spheres")
                self.assertTrue(all(sphere["radius"] > 0.0 for sphere in spheres))

    def test_the_planning_model_checks_a_few_hundred_pairs_where_today_checks_172414(self) -> None:
        """⭐ THE MEASURE the map named (169,642 with the 111 mm enclosure): the owner's cell, both models composed."""
        built = _CONTENT / "configs" / "robot" / "willy_ur10.yml"
        urdf = _CONTENT / "assets" / "robot" / "ur_description" / "ur10.urdf"
        if not built.is_file() or not urdf.is_file():
            self.skipTest("no built willy_ur10.yml on this box")
        raw = yaml.safe_load(built.read_text(encoding="utf-8"))
        hand, camera = _hand_body(), _camera_body()
        today, _, _, _, _ = compose_for_cell(raw, bodies=[hand], wrist_bodies=[camera])
        self.assertEqual(845, sum(planning.sphere_count(today).values()))
        self.assertEqual(172414, planning.self_pair_count(today))
        old, _, _, _, _ = compose_for_cell(raw, bodies=[hand], wrist_bodies=[_camera_body(111.15)])
        self.assertEqual(169642, planning.self_pair_count(old), "the count the map measured, 2026-10-08")
        sent = planning_payload(LEAN, arm="ur10", body_links=[hand], wrist_body_links=[camera])
        assert not isinstance(sent, str), sent
        payload = planning.read_payload(sent.to_json())
        lean = planning.planning_config(raw, payload, urdf_text=urdf.read_text(encoding="utf-8"), evidence=today)
        spheres, pairs = sum(planning.sphere_count(lean).values()), planning.self_pair_count(lean)
        self.assertLess(spheres, 80)
        self.assertLess(pairs, 1000)
        row = planning.planning_row(LEAN, lean, today)
        self.assertEqual((spheres, pairs, 845, 172414),
                         (row["spheres"], row["self_pairs"], row["evidence_spheres"], row["evidence_self_pairs"]))


class ThePlanningConfigIsTheEvidenceRobotButItsSpheresTests(unittest.TestCase):
    def _composed(self, payload: "dict | None" = None) -> "tuple[dict, dict]":
        """The planning config of ``payload`` against the evidence model of the cell ``_payload`` describes."""
        sent = payload or _payload()
        evidence, _, _, _, _ = compose_for_cell(_descriptor(), bodies=[dict(_payload()["bodies"][0], spheres=[
            {"center": [0.0, 0.0, 0.05], "radius": 0.03}] * 4)], margin_mm=4.0, default_q=[0.1] * 6)
        planned = planning.planning_config(_descriptor(), planning.read_payload(json.dumps(sent)), urdf_text=_urdf(),
                                           evidence=evidence, margin_mm=4.0, default_q=[0.1] * 6)
        return planned, evidence

    def test_only_the_spheres_and_the_ignored_pairs_differ(self) -> None:
        planned, evidence = self._composed()
        ours, theirs = planned["robot_cfg"]["kinematics"], evidence["robot_cfg"]["kinematics"]
        differ = sorted(key for key in set(ours) | set(theirs) if ours.get(key) != theirs.get(key))
        self.assertEqual(["collision_spheres", "self_collision_ignore"], differ)
        self.assertEqual(1, len(ours["collision_spheres"]["forearm_link"]))
        self.assertEqual(4, len(theirs["collision_spheres"]["hand"]))
        self.assertIn("wrist_1_link", ours["self_collision_ignore"]["upper_arm_link"])
        self.assertIn("hand", ours["self_collision_ignore"]["forearm_link"])
        self.assertEqual(planned["_provenance"], evidence["_provenance"])

    def test_an_arm_link_it_does_not_check_or_one_left_without_spheres_is_refused(self) -> None:
        stray = _payload()
        stray["links"]["elbow_link"] = {"frame": 3, "spheres": [{"center": [0.0, 0.0, 0.0], "radius": 0.05}]}
        short = _payload()
        del short["links"]["wrist_3_link"]
        for name, payload in (("a link nobody checks", stray), ("a link without spheres", short)):
            with self.subTest(case=name), self.assertRaises(PlanningModelError):
                self._composed(payload)

    def test_a_body_hung_elsewhere_than_the_evidence_hangs_it_is_refused(self) -> None:
        """⛔ The planning model's hand would plan a robot whose hand is not where the judged one's is."""
        moved = _payload()
        moved["bodies"][0]["fixed_transform"] = [0.0, 0.0, 0.040, 1.0, 0.0, 0.0, 0.0]
        with self.assertRaises(PlanningModelError) as caught:
            self._composed(moved)
        self.assertIn("hand", str(caught.exception))

    def test_a_payload_that_is_not_one_is_refused_field_by_field(self) -> None:
        good = _payload()
        broken = {
            "not json": "{",
            "no name": json.dumps(dict(good, model="")),
            "five DH rows": json.dumps(dict(good, dh_m=good["dh_m"][:5])),
            "a frame past 6": json.dumps(dict(good, links={"shoulder_link": {"frame": 7, "spheres": [
                {"center": [0, 0, 0], "radius": 0.05}]}})),
            "a radius of 0": json.dumps(dict(good, links={"shoulder_link": {"frame": 1, "spheres": [
                {"center": [0, 0, 0], "radius": 0.0}]}})),
            "a pair of one": json.dumps(dict(good, ignore=[["hand"]])),
        }
        for name, text in broken.items():
            with self.subTest(case=name), self.assertRaises(PlanningModelError):
                planning.read_payload(text)

    def test_a_pair_already_ignored_either_way_is_written_once(self) -> None:
        config = planning.with_ignored_pairs(_descriptor(), [["forearm_link", "upper_arm_link"],
                                                             ["shoulder_link", "wrist_3_link"]])
        table = config["robot_cfg"]["kinematics"]["self_collision_ignore"]
        self.assertNotIn("upper_arm_link", table["forearm_link"], "already ignored the other way round")
        self.assertEqual(["wrist_3_link"], table["shoulder_link"])

    def test_pairs_are_counted_as_curobo_lists_them(self) -> None:
        """Two links that do not ignore one another, every slot against every slot; the control is one ignore less."""
        config = _descriptor()
        counts = planning.sphere_count(config)
        self.assertEqual({link: 3 for link in _ARM_LINKS}, counts)
        ignored = 6  # the five neighbours and wrist_1|wrist_3, each 3 x 3
        self.assertEqual((15 - ignored) * 9, planning.self_pair_count(config))
        del config["robot_cfg"]["kinematics"]["self_collision_ignore"]["forearm_link"]
        self.assertEqual((15 - ignored + 1) * 9, planning.self_pair_count(config))


class TheClientComposesThePayloadTests(unittest.TestCase):
    def test_the_owner_cell_sends_its_arm_hand_and_camera_lean(self) -> None:
        sent = planning_payload(LEAN, arm="ur10", body_links=[_hand_body()], wrist_body_links=[_camera_body()])
        assert not isinstance(sent, str), sent
        payload = sent.payload
        self.assertEqual(list(_ARM_LINKS), list(payload["links"]))
        hand, camera = payload["bodies"][0], payload["wrist_bodies"][0]
        self.assertEqual(_hand_body()["fixed_transform"], hand["fixed_transform"], "the hand hangs where it hangs")
        lean_hand = yaml.safe_load(hand_planning_spheres_path("robotiq_hande").read_text(encoding="utf-8"))
        self.assertEqual(len(lean_hand["collision_spheres"]["tool0"]), len(hand["spheres"]))
        self.assertEqual(3, len(camera["spheres"]), f"the enclosure at {LEAN_BOX_REACH_MM} mm reach")
        self.assertEqual(_camera_body()["fixed_transform"], camera["fixed_transform"])
        pairs = {frozenset(pair) for pair in payload["ignore"]}
        self.assertEqual({frozenset((a, b)) for a in _GUARDED for b in _GUARDED if a < b}, pairs)
        self.assertFalse(any("shoulder_link" in pair for pair in pairs), "the shoulder stands for the base: kept")

    def test_a_sentence_where_no_model_can_be_composed(self) -> None:
        """Each of these plans on the evidence model, as today, and says why."""
        hand, camera = _hand_body(), _camera_body()
        cases = {
            "an unknown name": dict(model="nvidia", arm="ur10"),
            "no name": dict(model="", arm="ur10"),
            "a carried part": dict(model=LEAN, arm="ur10", attach_spheres=16),
            "no arm": dict(model=LEAN, arm=None),
            "an arm with no lean map": dict(model=LEAN, arm="ur99"),
        }
        for name, kwargs in cases.items():
            with self.subTest(case=name):
                said = planning_payload(kwargs.pop("model"), body_links=[hand], wrist_body_links=[camera], **kwargs)
                self.assertIsInstance(said, str)
        stranger = dict(hand, model="no_such_hand")
        self.assertIsInstance(planning_payload(LEAN, arm="ur10", body_links=[stranger], wrist_body_links=[]), str)
        flange = dict(hand, origin="flange")
        said = planning_payload(LEAN, arm="ur10", body_links=[flange], wrist_body_links=[])
        self.assertIsInstance(said, str)
        self.assertIn("plate", str(said))

    def test_the_client_sends_it_only_where_asked_and_never_inherits_one(self) -> None:
        from src.robot.safety.planning.curobo_client import CuroboPlanClient

        with mock.patch.dict(os.environ, {ENV_PLANNING_MODEL: "{\"model\": \"stale\"}"}):
            self.assertNotIn(ENV_PLANNING_MODEL, CuroboPlanClient(robot_config="willy_ur10.yml",
                                                                  body_links=[_hand_body()])._sidecar_env())  # noqa: SLF001
            asked = CuroboPlanClient(robot_config="willy_ur10.yml", body_links=[_hand_body()],
                                     wrist_body_links=[_camera_body()], planning_spheres=LEAN)
            env = asked._sidecar_env()  # noqa: SLF001
        sent = json.loads(env[ENV_PLANNING_MODEL])
        self.assertEqual(LEAN, sent["model"])
        self.assertEqual(list(_ARM_LINKS), list(sent["links"]))

    def test_a_model_the_client_cannot_compose_is_logged_and_not_sent(self) -> None:
        from src.robot.safety.planning.curobo_client import CuroboPlanClient

        client = CuroboPlanClient(robot_config="willy_ur99.yml", body_links=[_hand_body()], planning_spheres=LEAN)
        with self.assertLogs("CuroboPlanClient", level="WARNING") as logs:
            env = client._sidecar_env()  # noqa: SLF001
        self.assertNotIn(ENV_PLANNING_MODEL, env)
        self.assertTrue(any("plans on the evidence model" in line for line in logs.output), logs.output)


def _started(ready: dict, *, planning_spheres: str = ""):  # noqa: ANN202
    from src.robot.safety.planning.curobo_client import CuroboPlanClient

    script = Path(tempfile.mkdtemp()) / "sidecar.py"
    script.write_text(textwrap.dedent(f"""
        import json, sys
        sys.stdout.write(json.dumps({ready!r}) + chr(10)); sys.stdout.flush()
        for line in sys.stdin:
            if json.loads(line).get("cmd") == "shutdown":
                break
    """), encoding="utf-8")
    return CuroboPlanClient(python_path=sys.executable, server_script=str(script), robot_config="willy_ur10.yml",
                            body_links=[_hand_body()], planning_spheres=planning_spheres)


_READY = {"status": "ready", "joint_names": ["j0", "j1", "j2", "j3", "j4", "j5"], "dt": 0.025}
_ROW = {"model": "lean", "spheres": 34, "links": {}, "self_pairs": 120, "evidence_spheres": 845,
        "evidence_self_pairs": 172414, "sha256": "e" * 64}


class TheClientSaysWhichRobotPlansTests(unittest.TestCase):
    def _start(self, ready: dict, **kwargs: typing.Any):  # noqa: ANN202
        client = _started(ready, **kwargs)
        self.addCleanup(client.close)
        with self.assertLogs("CuroboPlanClient", level="INFO") as logs:
            client.start()
        return client, logs.output

    def test_a_sidecar_that_built_it_is_said_to_plan_on_it(self) -> None:
        client, said = self._start({**_READY, "planning": _ROW, "planning_sha256": "e" * 64}, planning_spheres=LEAN)
        self.assertEqual(LEAN, client.identity.plans_on)
        self.assertTrue(any("plans on the lean model" in line for line in said), said)
        self.assertIn("planning model: lean, 34 spheres and 120 self pairs", client.identity.render())

    def test_a_refusal_or_an_older_sidecar_is_said_and_plans_on_the_evidence_model(self) -> None:
        for ready, words in (({**_READY, "planning": {"model": "lean", "refused": "no URDF"}}, "no URDF"),
                             (dict(_READY), "older than this client")):
            with self.subTest(words=words):
                client, said = self._start(ready, planning_spheres=LEAN)
                self.assertEqual("", client.identity.plans_on)
                self.assertTrue(any(words in line for line in said), said)

    def test_a_client_that_asked_for_none_says_nothing_of_it(self) -> None:
        client, said = self._start(dict(_READY))
        self.assertEqual("", client.identity.plans_on)
        self.assertFalse(any("planning model" in line for line in said), said)


class TheConfigKeyTests(unittest.TestCase):
    def test_the_key_takes_the_models_planning_model_knows_and_is_todays_by_default(self) -> None:
        from src.config.schema.robot.safety_schema import PlannedMotionSafetyConfig

        self.assertEqual(set(PLANNING_MODELS),
                         set(typing.get_args(PlannedMotionSafetyConfig.model_fields["planning_spheres"].annotation)))
        self.assertEqual("", PlannedMotionSafetyConfig().planning_spheres)

    def test_it_reaches_the_client_through_the_reservation(self) -> None:
        from pydantic import ValidationError

        from src.config.schema.robot import RobotConfig
        from src.robot.safety.planning.curobo_client import CuroboPlanClient
        from src.robot.safety.planning.reservation import PlannerReservation

        self.assertEqual("", PlannerReservation.from_config(robot_cfg=RobotConfig.model_validate({"vendor": "ur"}))
                         .planning_spheres)
        with self.assertRaises(ValidationError):
            RobotConfig.model_validate({"vendor": "ur", "safety": {"planned_motion": {"planning_spheres": "nvidia"}}})
        cell = RobotConfig.model_validate({"vendor": "ur", "safety": {"planned_motion": {"planning_spheres": "lean"}}})
        client = CuroboPlanClient(robot_config="willy_ur10.yml", body_links=[_hand_body()])
        client.reserve_world(PlannerReservation.from_config(robot_cfg=cell))
        self.assertEqual(LEAN, json.loads(client._sidecar_env()[ENV_PLANNING_MODEL])["model"])  # noqa: SLF001


class TheSidecarPlansOnItAndJudgesOnTheEvidenceTests(unittest.TestCase):
    """The sidecar runs in the cuRobo environment, so its half is read as source."""

    def setUp(self) -> None:
        self.source = _SERVER.read_text(encoding="utf-8")

    def test_its_new_siblings_load_as_the_sidecar_loads_them(self) -> None:
        """⛔ A sibling the sidecar cannot import is a sidecar that never says ready: each is loaded here as it loads
        them, by plain name from its own folder, in an interpreter that cannot see this package."""
        import subprocess

        folder = _SERVER.parent
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import _curobo_world, _curobo_cspace, _curobo_planning; "
                "assert _curobo_planning.compose_for_cell is not None; print('loaded')")
        done = subprocess.run([sys.executable, "-I", "-c", code, str(folder)], capture_output=True, text=True,
                              timeout=120, cwd=tempfile.gettempdir())
        self.assertEqual(0, done.returncode, done.stderr)
        self.assertIn("loaded", done.stdout)

    def test_the_planner_is_built_from_the_planning_model_and_every_judgement_from_the_evidence(self) -> None:
        build = self.source[: self.source.find("def _terms(")]
        self.assertIn("robot=copy.deepcopy(_PLANNING)", build)
        self.assertIn("robot_config=copy.deepcopy(_COMPOSED)", build, "the checker judges on the evidence model")
        self.assertIn("_LAYOUT = SphereLayout.from_robot_config(_COMPOSED)", build, "pairs are named on the evidence")
        self.assertIn("_kin = Kinematics(KinematicsCfg.from_data_dict(copy.deepcopy(_COMPOSED)))", build)
        self.assertIn("_composed_sha256 = canonical_sha256(_EVIDENCE)", build, "the evidence hash is as it was")

    def test_a_model_it_cannot_build_plans_on_the_evidence_model_and_says_why(self) -> None:
        block = self.source[self.source.find("_planning_sent = os.environ.get(ENV_PLANNING_MODEL)"):
                            self.source.find("_TWO_MODELS = _PLANNING is not _COMPOSED")]
        # Whatever it raised: a planning model is never why a planner does not start.
        self.assertIn("except Exception as _exc:", block)
        self.assertIn("_PLANNING = _COMPOSED", block[block.find("except Exception as _exc:"):])
        self.assertIn('"refused": _said_why', block)
        self.assertIn("planning_config(_raw, _payload, urdf_text=_urdf_text, evidence=_COMPOSED", block)

    def test_the_ready_line_adds_the_planning_row_and_its_hash(self) -> None:
        self.assertIn('_PLANNING_READY = {"planning": _planning_row, "planning_sha256": _planning_row["sha256"]}',
                      self.source)
        ready = self.source[self.source.find('_emit({"status": "ready"'):]
        self.assertIn("**_PLANNING_READY", ready[:600])


if __name__ == "__main__":
    unittest.main()
