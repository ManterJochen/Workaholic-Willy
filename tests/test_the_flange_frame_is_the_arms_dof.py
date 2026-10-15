"""Everything bolted to the flange hangs from the arm's own flange frame, its dof, and not from a six (P0f F9a).

The hand, a coupling plate, a wrist camera and a carried part all hang from the flange, and the guard, the self filter
and the planner's evidence placed each at DH frame 6, the flange of every UR. On an arm with a seventh joint every one
of them would have hung from the sixth link, silently: a guard judging a hand one joint short of where it is. Each is
now placed at the description's ``dof``, which is 6 on every UR, so no hash moves (the evidence recompute holds all 47
files). The arm links the self filter fits come from the description's collision links the same way.

The control arm is ur5e's description with a seventh joint, handed in where the registry is asked. It is never
admitted and has no bundle row for its seventh link, which is why the self filter refuses it rather than guess.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

import numpy as np
import yaml

from src.config.arms import load_arm
from src.config.schema.robot import RobotConfig
from src.robot.safety import _fcl_self_collision
from src.robot.safety.planning import self_envelope
from src.robot.safety.planning._declared_body import Box
from src.robot.safety.planning.environment import collision_mesh_bundle
from tests._wrist_body import wrist_body


def _seven_joints(model: str = "ur5e") -> Any:
    """``model``'s description with a seventh joint copied from its sixth, as the registry would answer for it."""
    arm = load_arm(model)
    last = arm.joints[-1]
    extra = last.model_copy(update={"name": "wrist_4_joint", "link": "wrist_4_link", "part": "wrist_4"})
    return arm.model_copy(update={"joints": (*arm.joints, extra)})


def _answering(twin: Any) -> Any:
    real = load_arm

    def load(name: str, **kwargs: Any) -> Any:
        return twin if name == twin.model else real(name, **kwargs)

    return load


def _hand(name: str = "robotiq_2f85") -> Any:
    from src.robot.safety.planning.hand import planner_hand

    text = collision_mesh_bundle("ur5e").parent.parent / "planning" / "robot" / f"{name}_gripper_spheres.yml"
    origin = yaml.safe_load(text.read_text(encoding="utf-8"))["_provenance"]["origin"]
    gripper: dict[str, Any] = {"model": name}
    if origin == "mounting_face":
        gripper["coupling_plates"] = [{"name": "plate", "thickness_mm": 20.0}]
    return planner_hand(RobotConfig.model_validate({"vendor": "ur", "gripper": gripper}))


class _Twin(unittest.TestCase):
    def setUp(self) -> None:
        self.twin = _seven_joints()
        self.assertEqual(7, self.twin.dof)
        self_envelope.arm_capsules.cache_clear()
        self.addCleanup(self_envelope.arm_capsules.cache_clear)

    def _patched(self, module: Any) -> Any:
        return mock.patch.object(module, "load_arm", _answering(self.twin))


class TheSelfFilterHangsTheHandFromTheFlangeTests(_Twin):
    def test_the_hand_spheres_hang_from_the_arms_dof(self) -> None:
        hand = _hand()
        with self._patched(self_envelope):
            seven = self_envelope.hand_spheres(hand, "ur5e")
        six = self_envelope.hand_spheres(hand, "ur5e")
        assert seven is not None and six is not None
        self.assertEqual({7}, {sphere.frame for sphere in seven})
        self.assertEqual({6}, {sphere.frame for sphere in six})
        self.assertEqual({int}, {type(sphere.frame) for sphere in six})

    def test_a_carried_part_hangs_where_the_hand_does(self) -> None:
        hand = _hand()
        with self._patched(self_envelope):
            spheres = self_envelope.hand_spheres(hand, "ur5e")
        assert spheres is not None
        capsule = self_envelope.payload_capsule(hand, spheres, length_mm=30.0, lateral_margin_mm=2.0)
        self.assertEqual(7, capsule.frame)

    def test_the_arm_links_are_the_descriptions(self) -> None:
        """A link the description names and the bundle does not carry is refused, never skipped."""
        with self._patched(self_envelope):
            self.assertIsNone(self_envelope.arm_capsules("ur5e"))
        self_envelope.arm_capsules.cache_clear()
        capsules = self_envelope.arm_capsules("ur5e")
        assert capsules is not None
        self.assertEqual([1, 2, 3, 4, 5, 6], [capsule.frame for capsule in capsules])


class TheGuardHangsTheHandFromTheFlangeTests(_Twin):
    def _parts(self, **asked: Any) -> dict:
        return _fcl_self_collision.composed_parts("ur5e", None, "robotiq_2f85", 0.0, **asked)

    def test_the_hand_parts_hang_from_the_arms_dof_and_the_arm_parts_from_their_rows(self) -> None:
        with self._patched(_fcl_self_collision):
            seven = self._parts()
        six = self._parts()
        for name in ("gripper", "lfinger", "rfinger"):
            with self.subTest(part=name):
                self.assertEqual(7, seven[name][2])
                self.assertEqual(6, six[name][2])
        arm = ("shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3")
        self.assertEqual([1, 2, 3, 4, 5, 6], [seven[name][2] for name in arm])
        self.assertEqual([1, 2, 3, 4, 5, 6], [six[name][2] for name in arm])

    def test_a_coupling_plate_hangs_from_the_arms_dof(self) -> None:
        plate = (Box(name="plate", centre_mm=(0.0, 5.0, 0.0), half_extents_mm=(30.0, 5.0, 30.0)),)
        with self._patched(_fcl_self_collision):
            seven = self._parts(coupling_boxes=plate)
        six = self._parts(coupling_boxes=plate)
        added = sorted(set(six) - set(self._parts()))
        self.assertTrue(added)
        self.assertEqual({7}, {seven[name][2] for name in added})
        self.assertEqual({6}, {six[name][2] for name in added})

    def test_a_wrist_camera_hangs_from_the_frame_it_is_handed(self) -> None:
        body = wrist_body()
        parts = body.guard_parts(frame=7)
        self.assertEqual([[7]], [np.asarray(parts[key]).tolist() for key in parts if key.endswith("__frame")])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
