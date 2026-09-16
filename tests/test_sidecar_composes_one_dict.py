"""The sidecar loads ONE config: the descriptor, then the guard's margin, then a payload link, composed as a dict.

It used to derive a temporary yaml per step, chained by file name, and handed each cuRobo consumer a path: the planner,
``Kinematics`` and the check_js checker each read a file, and nothing said which bytes the planner actually held. Now
the descriptor is read once, composed in memory and hashed, and every consumer is built from its own deep copy of that
one dict, because cuRobo's ``LinkParams.create`` rewrites the dict it is handed (UM lane S09).

The sidecar runs in the cuRobo environment, so its half is read as source, as test_curobo_client_error_vs_verdict.py
does. The composition is a sibling module both interpreters load, and is tested here directly.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import yaml

from src.contracts import UNSET
from src.robot.safety.planning._curobo_attach import apply_attached_object_link
from src.robot.safety.planning._curobo_margin import apply_self_collision_margin

_SERVER = Path(__file__).resolve().parents[1] / "src/robot/safety/planning/curobo_planner_server.py"
_LINKS = ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link", "tool0"]


def _descriptor() -> dict:
    """Shaped like ur5e_robotiq_hande.yml as build_ur_config.py writes it, cut to what the composition reads."""
    return {
        "_provenance": {"arm": "ur5e", "gripper_key": "robotiq_hande", "coupling_mm": 20.0},
        "robot_cfg": {
            "kinematics": {
                "urdf_path": "robot/ur_description/ur5e.urdf",
                "base_link": "base_link",
                "tool_frames": ["tool0"],
                "collision_link_names": list(_LINKS),
                "collision_spheres": {
                    "forearm_link": [{"center": [0.0, 0.0, 0.1], "radius": 0.05}],
                    "tool0": [{"center": [0.0, 0.05, 0.0], "radius": 0.04}],
                },
                "collision_sphere_buffer": 0.0,
                "self_collision_buffer": {link: 0.0 for link in _LINKS},
                "self_collision_ignore": {
                    "wrist_2_link": ["wrist_3_link", "tool0"],
                    "wrist_3_link": ["tool0"],
                },
            },
        },
    }


class TheSidecarComposesOneDictTests(unittest.TestCase):
    def setUp(self) -> None:
        from src.robot.safety.planning._curobo_body_links import compose_sidecar_config

        self.compose = compose_sidecar_config

    def test_margin_then_attach_is_what_the_file_chain_derived(self) -> None:
        chained, _ = apply_self_collision_margin(_descriptor(), 10.0)
        chained, added = apply_attached_object_link(chained, spheres=16)
        self.assertTrue(added)
        self.assertEqual(self.compose(_descriptor(), bodies=(), margin_mm=10.0, attach_spheres=16), chained)

    def test_the_order_is_load_bearing(self) -> None:
        """⭐ THE CONTROL. Attach first and the margin lands on the payload link too, which the planner must not get."""
        attached, _ = apply_attached_object_link(_descriptor(), spheres=16)
        swapped, _ = apply_self_collision_margin(attached, 10.0)
        self.assertNotEqual(self.compose(_descriptor(), bodies=(), margin_mm=10.0, attach_spheres=16), swapped)

    def test_no_margin_and_no_slots_is_the_descriptor_as_written(self) -> None:
        source = _descriptor()
        composed = self.compose(source, bodies=(), margin_mm=0.0, attach_spheres=0)
        self.assertEqual(composed, source)
        self.assertIsNot(composed, source, "a consumer that rewrites the result would rewrite the descriptor")

    def test_the_descriptor_is_never_mutated(self) -> None:
        source = _descriptor()
        before = copy.deepcopy(source)
        composed = self.compose(source, bodies=(), margin_mm=10.0, attach_spheres=16)
        composed["robot_cfg"]["kinematics"]["collision_link_names"].append("mutated")
        self.assertEqual(source, before)

    def test_a_body_link_on_a_descriptor_that_carries_a_hand_is_refused(self) -> None:
        """This fixture is an {arm}_{hand} file: its tool0 spheres are a hand already, and a body would count it twice."""
        from src.robot.safety.planning._curobo_body_links import BodyLinkError

        with self.assertRaises(BodyLinkError):
            self.compose(_descriptor(), bodies=({"link_name": "hand"},), margin_mm=10.0, attach_spheres=16)


class TheConfigHashIsCanonicalTests(unittest.TestCase):
    def setUp(self) -> None:
        from src.robot.safety.planning._curobo_body_links import canonical_sha256

        self.sha = canonical_sha256

    def test_it_is_sha256_of_sorted_compact_json(self) -> None:
        """Pinned as a definition, because the py3.10 sidecar and the py3.11 evidence both compute it."""
        value = _descriptor()
        text = json.dumps(value, sort_keys=True, separators=(",", ":"))
        self.assertEqual(self.sha(value), hashlib.sha256(text.encode("utf-8")).hexdigest())

    def test_key_order_and_line_endings_do_not_change_it(self) -> None:
        text = yaml.safe_dump(_descriptor(), sort_keys=False)
        reordered = yaml.safe_dump(_descriptor(), sort_keys=True).replace("\n", "\r\n")
        self.assertEqual(self.sha(yaml.safe_load(text)), self.sha(yaml.safe_load(reordered)))

    def test_a_changed_value_or_list_order_changes_it(self) -> None:
        """The control. Link order is meaningful to cuRobo, so it has to be meaningful to the hash."""
        base = self.sha(_descriptor())
        buffer = _descriptor()
        buffer["robot_cfg"]["kinematics"]["self_collision_buffer"]["tool0"] = 0.0001
        order = _descriptor()
        order["robot_cfg"]["kinematics"]["collision_link_names"].reverse()
        self.assertNotEqual(base, self.sha(buffer))
        self.assertNotEqual(base, self.sha(order))


def _within(source: str, head: str, needle: str, chars: int) -> bool:
    """True when ``needle`` occurs within ``chars`` characters after the first ``head``."""
    start = source.find(head)
    return start >= 0 and needle in source[start:start + chars]


def _block(source: str, head: str) -> str:
    lines = source.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == head), None)
    if start is None:
        return ""
    indent = len(lines[start]) - len(lines[start].lstrip())
    block = [lines[start]]
    for line in lines[start + 1:]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        block.append(line)
    return "\n".join(block)


class TheSidecarSourceLoadsTheDictTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = _SERVER.read_text(encoding="utf-8")

    def test_no_temporary_descriptor_is_derived(self) -> None:
        for name in ("derive_margin_config_file(", "derive_attach_config_file(", "_ROBOT_IN_USE"):
            with self.subTest(absent=name):
                self.assertNotIn(name, self.source)

    def test_every_consumer_is_built_from_its_own_copy(self) -> None:
        self.assertTrue(_within(self.source, "MotionPlannerCfg.create(", "robot=copy.deepcopy(_COMPOSED)", 80))
        self.assertIn("KinematicsCfg.from_data_dict(copy.deepcopy(_COMPOSED))", self.source)
        # The checker moved out of the check_js branch in S16, because the ready gate needs it before any
        # request arrives. Its own copy is the claim, not the place: LinkParams.create rewrites what it is
        # handed, so a consumer sharing the dict would plan a robot the previous consumer edited.
        build = self.source[: self.source.find('def _terms(')]
        self.assertIn("robot_config=copy.deepcopy(_COMPOSED)", build)
        self.assertNotIn("_COMPOSED,", _block(self.source, 'if cmd == "check_js":'))

    def test_the_ready_line_says_who_the_sidecar_is(self) -> None:
        for name in ('"descriptor"', '"arm_descriptor_sha256"', '"urdf_sha256"', '"composed_sha256"', '"bodies"'):
            with self.subTest(present=name):
                self.assertTrue(_within(self.source, '_emit({"status": "ready"', name, 600))

    def test_the_scans_can_fail(self) -> None:
        """The control: a name outside the window, a name absent, and a checker built outside its branch."""
        synthetic = '_emit({"status": "ready"' + " " * 700 + '"composed_sha256"\nif cmd == "check_js":\n    continue\n'
        self.assertFalse(_within(synthetic, '_emit({"status": "ready"', '"composed_sha256"', 600))
        self.assertFalse(_within(synthetic, "MotionPlannerCfg.create(", "robot=", 80))
        self.assertNotIn("robot_config=copy.deepcopy(_COMPOSED)", synthetic[: synthetic.find("def _terms(")])
        self.assertEqual(-1, synthetic.find("def _terms("), "the control source defines no _terms")


def _client_after(ready: dict):
    from src.robot.safety.planning.curobo_client import CuroboPlanClient

    script = Path(tempfile.mkdtemp()) / "sidecar.py"
    script.write_text(textwrap.dedent(f"""
        import json, sys
        sys.stdout.write(json.dumps({ready!r}) + chr(10)); sys.stdout.flush()
        for line in sys.stdin:
            if json.loads(line).get("cmd") == "shutdown":
                break
    """), encoding="utf-8")
    return CuroboPlanClient(python_path=sys.executable, server_script=str(script), robot_config="unused",
                            scene_config=None)


_READY = {"status": "ready", "joint_names": ["j0", "j1", "j2", "j3", "j4", "j5"], "dt": 0.02}
_HASHES = {"arm_descriptor_sha256": "a" * 64, "urdf_sha256": "b" * 64, "composed_sha256": "c" * 64}


class TheClientKeepsWhoTheSidecarIsTests(unittest.TestCase):
    def _started(self, ready: dict):
        client = _client_after(ready)
        self.addCleanup(client.close)
        client.start()
        return client

    def test_a_ready_line_with_the_hashes_gives_an_identity_with_them(self) -> None:
        provenance = {"arm": "ur5e", "gripper_key": "robotiq_2f85", "coupling_mm": None}
        client = self._started({**_READY, **_HASHES, "bodies": [], "descriptor": provenance})
        identity = client.identity
        self.assertEqual(identity.arm_descriptor_sha256, "a" * 64)
        self.assertEqual(identity.urdf_sha256, "b" * 64)
        self.assertEqual(identity.composed_sha256, "c" * 64)
        self.assertEqual(identity.bodies, ())
        self.assertEqual(identity.provenance, provenance)
        self.assertEqual(client.descriptor_provenance, provenance, "the provenance stays readable as it was")

    def test_a_ready_line_that_says_nothing_gives_unset_never_an_empty_string(self) -> None:
        """⭐ THE CONTROL. An older sidecar reports none of it, and '' or None would read as a value."""
        for said in ({}, {"arm_descriptor_sha256": "", "urdf_sha256": None, "composed_sha256": 7, "bodies": None}):
            with self.subTest(said=said):
                identity = self._started({**_READY, **said}).identity
                for field in ("arm_descriptor_sha256", "urdf_sha256", "composed_sha256", "bodies"):
                    self.assertIs(getattr(identity, field), UNSET, field)
                self.assertIsNone(identity.provenance)

    def test_an_identity_renders_and_serialises(self) -> None:
        from src.robot.safety.planning.curobo_client import SidecarIdentity

        row = {"link": "hand", "parent": "tool0", "spheres": 36, "slots": 0, "spheres_sha256": "d" * 64}
        for identity in (SidecarIdentity.from_ready({**_HASHES, "bodies": [row], "descriptor": {"arm": "ur5e"}}),
                         SidecarIdentity.from_ready({})):
            with self.subTest(identity=identity):
                text = identity.render()
                text.encode("ascii")
                self.assertFalse(text.endswith("\n"))
                json.dumps(identity.to_dict())


class AnyClientHasAnIdentityTests(unittest.TestCase):
    """The drivers ask every client, and the tests inject stubs that only ever carried the provenance."""

    def test_a_stub_with_only_the_provenance_keeps_it_and_reports_no_hash(self) -> None:
        from src.robot.safety.planning.curobo_client import SidecarIdentity

        class Stub:
            descriptor_provenance = {"arm": "ur5e", "gripper_key": "robotiq_2f85", "coupling_mm": None}

        identity = SidecarIdentity.from_client(Stub())
        self.assertEqual(identity.provenance, Stub.descriptor_provenance)
        self.assertIs(identity.composed_sha256, UNSET)

    def test_a_client_that_says_nothing_has_no_provenance_so_the_check_refuses(self) -> None:
        """Fails closed: nothing to read is a missing provenance, which the descriptor refusal refuses."""
        from src.robot.safety.planning.curobo_client import SidecarIdentity

        identity = SidecarIdentity.from_client(object())
        self.assertIsNone(identity.provenance)
        self.assertIs(identity.arm_descriptor_sha256, UNSET)


if __name__ == "__main__":
    unittest.main()
