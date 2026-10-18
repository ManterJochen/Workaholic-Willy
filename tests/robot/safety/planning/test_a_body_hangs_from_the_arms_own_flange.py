"""A body on the flange hangs from the link the arm description names and ignores the arm's own last links (P0f F9b).

The hand, a coupling plate and a wrist camera are fixed links of the planner's robot. They hung from ``HAND_PARENT =
"tool0"`` and ignored ``wrist_3_link`` and ``wrist_2_link`` by name, and the sidecar refused a hand ignoring
``wrist_1_link``, so every arm was a UR: the planner start refused any description whose flange is another link. The
builders now take the arm (``ArmLinks``: the flange link and the joint links, base first) and the sidecar reads the
arm's links off the descriptor it loaded, so the rules are the same by position and the names are the arm's. On every
UR the names come out as before, so no composed config moves (the evidence recompute holds ``composed_sha256``).

The control arm is ur5e's description with a seventh joint, and the sidecar's side is a config whose links are named
nothing like a UR's.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

import numpy as np

from src.config.arms import load_arm
from src.config.schema.robot import RobotConfig
from src.robot.safety.planning import _curobo_body_links as links
from src.robot.safety.planning import body_link
from src.robot.safety.planning._curobo_body_links import ArmLinks, BodyLinkError, apply_body_links
from src.robot.safety.planning.hand import planner_hand

_IDENTITY = np.eye(3).tolist()
_SPHERE = [{"center": [0.0, 0.05, 0.0], "radius": 0.02}]


def _seven_joints() -> Any:
    arm = load_arm("ur5e")
    last = arm.joints[-1]
    extra = last.model_copy(update={"name": "wrist_4_joint", "link": "wrist_4_link", "part": "wrist_4"})
    flange = arm.flange.model_copy(update={"link": "flange_7"})
    return arm.model_copy(update={"joints": (*arm.joints, extra), "flange": flange})


def _hand_body(on: ArmLinks) -> dict[str, Any]:
    return links.hand_body_link(arm_links=on, spheres=_SPHERE, rotation=_IDENTITY, approach_in_tool0=(0.0, 1.0, 0.0),
                                origin="flange", coupling_mm=0.0)


def _config(arm_links: list[str], tool_frame: str) -> dict[str, Any]:
    """A descriptor as the sidecar holds it, reduced to what composing a body onto it reads."""
    return {"robot_cfg": {"kinematics": {
        "collision_link_names": list(arm_links), "tool_frames": [tool_frame],
        "collision_spheres": {name: _SPHERE for name in arm_links},
        "self_collision_ignore": {}, "self_collision_buffer": {name: 0.0 for name in arm_links},
    }}}


class TheBuildersNameTheArmsLinksTests(unittest.TestCase):
    def test_every_body_hangs_from_the_flange_the_description_names(self) -> None:
        on = ArmLinks.from_description(_seven_joints())
        hand = _hand_body(on)
        self.assertEqual("flange_7", hand["parent"])
        self.assertEqual(["flange_7", "wrist_4_link", "wrist_3_link"], hand["ignore"])
        from src.robot.safety.planning._declared_body import Box

        boxes = [Box(name="plate", centre_mm=(0.0, 5.0, 0.0), half_extents_mm=(30.0, 5.0, 30.0))]
        plate = links.coupling_body_link(arm_links=on, boxes=boxes, rotation=_IDENTITY)
        assert plate is not None
        self.assertEqual(("flange_7", ["flange_7", "wrist_4_link", "wrist_3_link"]), (plate["parent"], plate["ignore"]))
        camera = links.wrist_body_link(arm_links=on, rig_id="wrist", boxes=boxes, rotation=_IDENTITY,
                                       translation_m=[0.0, 0.0, 0.0])
        self.assertEqual("flange_7", camera["parent"])
        self.assertEqual(["flange_7", "wrist_4_link", links.HAND_LINK, links.COUPLING_LINK], camera["ignore"])

    def test_a_ur_keeps_the_names_it_always_had(self) -> None:
        """THE CONTROL: the rule by position gives every UR the table its descriptors were composed with."""
        for model in ("ur3", "ur5e", "ur16e", "ur20"):
            on = ArmLinks.from_description(load_arm(model))
            with self.subTest(model=model):
                self.assertEqual(("tool0", "wrist_3_link", "wrist_2_link"), on.hand_ignore)
                self.assertEqual(("tool0", "wrist_3_link", "hand", "coupling"), on.wrist_body_ignore)

    def test_an_arm_of_fewer_than_three_links_is_refused(self) -> None:
        with self.assertRaisesRegex(BodyLinkError, "2 joint links"):
            ArmLinks(flange="tool0", links=("a_link", "b_link"))


class TheSidecarReadsTheArmOffItsDescriptorTests(unittest.TestCase):
    _ARM = ["l1", "l2", "l3", "l4", "l5", "l6", "l7"]

    def _arm(self) -> ArmLinks:
        return ArmLinks(flange="flange_7", links=tuple(self._ARM))

    def test_a_hand_hung_by_the_rule_composes(self) -> None:
        _, added = apply_body_links(_config(self._ARM, "flange_7"), [_hand_body(self._arm())])
        self.assertEqual(["hand"], added)

    def test_a_hand_ignoring_the_third_link_from_the_flange_is_refused_by_its_name(self) -> None:
        body = _hand_body(self._arm())
        body["ignore"] = [*body["ignore"], "l5"]
        with self.assertRaisesRegex(BodyLinkError, "ignores l5"):
            apply_body_links(_config(self._ARM, "flange_7"), [body])

    def test_a_camera_ignoring_a_link_it_can_hit_is_refused_by_its_name(self) -> None:
        from src.robot.safety.planning._declared_body import Box

        boxes = [Box(name="housing", centre_mm=(0.0, 0.0, 30.0), half_extents_mm=(20.0, 20.0, 20.0))]
        camera = links.wrist_body_link(arm_links=self._arm(), rig_id="wrist", boxes=boxes, rotation=_IDENTITY,
                                       translation_m=[0.0, 0.0, 0.0])
        camera["ignore"] = [*camera["ignore"], "l6"]
        with self.assertRaisesRegex(BodyLinkError, r"\['l6'\]"):
            apply_body_links(_config(self._ARM, "flange_7"), [camera])

    def test_a_hand_hung_from_another_link_than_the_descriptors_tool_frame_is_refused(self) -> None:
        with self.assertRaisesRegex(BodyLinkError, "hangs from 'flange_7'.*tool frame 'tool0'"):
            apply_body_links(_config(self._ARM, "tool0"), [_hand_body(self._arm())])

    def test_a_descriptor_of_fewer_than_three_arm_links_is_refused(self) -> None:
        with self.assertRaisesRegex(BodyLinkError, "2 arm links"):
            apply_body_links(_config(["l1", "l2"], "flange_7"),
                             [_hand_body(ArmLinks(flange="flange_7", links=("l0", "l1", "l2")))])


class TheHandLinkIsTheArmsTests(unittest.TestCase):
    def _hand(self) -> Any:
        return planner_hand(RobotConfig.model_validate({"vendor": "ur", "gripper": {"model": "robotiq_2f85"}}))

    def test_the_hand_link_hangs_from_the_arm_it_is_built_for(self) -> None:
        twin = _seven_joints()
        real = load_arm

        def load(name: str, **kwargs: Any) -> Any:
            return twin if name == "ur5e" else real(name, **kwargs)

        with mock.patch.object(body_link, "load_arm", load):
            seven = body_link.HandLink.from_hand(self._hand(), arm="ur5e").to_dict()
        six = body_link.HandLink.from_hand(self._hand(), arm="ur5e").to_dict()
        self.assertEqual(("flange_7", ["flange_7", "wrist_4_link", "wrist_3_link"]), (seven["parent"], seven["ignore"]))
        self.assertEqual(("tool0", ["tool0", "wrist_3_link", "wrist_2_link"]), (six["parent"], six["ignore"]))

    def test_the_planned_arm_sends_the_flange_its_description_names(self) -> None:
        """What was a refusal at the planner start is now the body the sidecar is handed; its descriptor judges it."""
        from tests._arm_kit import planned_arm

        arm = planned_arm(RobotConfig.model_validate({
            "vendor": "ur", "ur": {"ip": "10.1.2.3"}, "motion": {"planner": "curobo"},
            "safety": {"self_collision": {"planner_margin_mm": 4.0}}, "gripper": {"model": "robotiq_2f85"},
        }))
        real = load_arm("ur5e")
        other = real.model_copy(update={"flange": real.flange.model_copy(update={"link": "link8"})})
        with mock.patch.object(body_link, "load_arm", lambda name, **_: other if name == "ur5e" else load_arm(name)):
            client = arm._default_curobo_client_factory()()
        sent = client._body_links[0]
        self.assertEqual(("link8", ["link8", "wrist_3_link", "wrist_2_link"]), (sent["parent"], sent["ignore"]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
