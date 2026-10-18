"""The fence around the planning half: what every registered arm may and may not do to it (P0g slice 10).

Three rules, each over every vendor the registry lists (``tests/_registered_arms.py``) and each with a synthetic
violation it must find, because a rule that matches nothing passes every tree, a broken one included:

- An arm that owns a controller is a planned arm. It subclasses ``PlannedArm`` and satisfies ``PlansMotions``,
  ``KeepsLines``, ``SafetyGated`` and ``DeclinesCameraWorld``, so its motions are judged, planned and stamped like
  every other. The sim and the dummy own none (``capabilities.owns_a_controller``).
- A planned arm overrides only the declared hooks (``PlannedArm._VENDOR_HOOKS``): a vendor that answered ``move``
  itself would carry a motion path the safety preflight and the camera world never see. The capability mixins it lists
  add verbs and override none.
- No public attribute hands out the controller link: the transport, what the transport holds, or the sim's
  session. A caller holding one commands the controller past every guard. The sim arm's ``session`` is the one
  counted exception, until phase PY moves the sim arm onto ``PlannedArm``.
"""

from __future__ import annotations

import inspect
import unittest
from typing import Any

from src.robot.drivers.planned import capabilities as planned_capabilities
from src.robot.drivers.planned.arm import PlannedArm
from tests import _registered_arms

#: (vendor, public name) that hands out the controller link, each with its reason. The list may only shrink.
_EXPOSES: dict[tuple[str, str], str] = {
    ("sim", "session"): "tests and tooling reach the Isaac session through it; it goes when phase PY-b moves the sim "
                        "arm onto PlannedArm over an Isaac transport",
}


def _controller_protocols() -> tuple[type, ...]:
    from src.robot.core import DeclinesCameraWorld, PlansMotions
    from src.robot.core.arm_capabilities import KeepsLines
    from src.robot.safety.attestation import SafetyGated

    return (PlansMotions, KeepsLines, SafetyGated, DeclinesCameraWorld)


def _unplanned(arm: Any) -> list[str]:
    """Why an arm that owns a controller is not a planned arm; empty when it is one or owns none."""
    if not arm.capabilities.owns_a_controller:
        return []
    found = [] if isinstance(arm, PlannedArm) else ["not a PlannedArm"]
    found += [f"not {proto.__name__}" for proto in _controller_protocols() if not isinstance(arm, proto)]
    return found


def _is_member(value: Any) -> bool:
    return inspect.isfunction(value) or isinstance(value, (property, classmethod, staticmethod))


def _overrides(cls: type) -> set[str]:
    """Every ``PlannedArm`` member a class between ``cls`` and ``PlannedArm`` defines again, dunders aside."""
    planned = {name for name, value in vars(PlannedArm).items() if _is_member(value) and not name.startswith("__")}
    found: set[str] = set()
    for klass in cls.__mro__:
        if klass is PlannedArm:
            break
        found |= {name for name, value in vars(klass).items() if name in planned and _is_member(value)}
    return found


def _exposed(arm: Any) -> list[str]:
    """Every public attribute or property of ``arm`` whose value is one of its controller links."""
    held = _registered_arms.links(arm)
    names = {name for name in dir(type(arm)) if not name.startswith("_")} | {
        name for name in vars(arm) if not name.startswith("_")}
    found: list[str] = []
    for name in sorted(names):
        static = inspect.getattr_static(arm, name)
        if inspect.isfunction(static) or isinstance(static, (classmethod, staticmethod)):
            continue
        try:
            value = getattr(arm, name)
        except Exception:  # noqa: BLE001 - a property that cannot answer unconnected hands out nothing
            continue
        if any(value is link for link in held):
            found.append(name)
    return found


class AnArmThatOwnsAControllerIsAPlannedArmTests(unittest.TestCase):
    def test_every_registered_arm_that_owns_a_controller_is_one(self) -> None:
        owners = 0
        for vendor in _registered_arms.vendors():
            arm = _registered_arms.build(vendor)
            owners += bool(arm.capabilities.owns_a_controller)
            with self.subTest(vendor=vendor):
                self.assertEqual([], _unplanned(arm))
        self.assertGreaterEqual(owners, 1, "no registered arm owns a controller: the rule checked nothing")

    def test_an_owner_that_is_not_a_planned_arm_is_found(self) -> None:
        """The positive control: a dummy that claims a controller."""
        import dataclasses

        from src.robot.drivers.dummy.arm import DummyRobotArm

        arm = DummyRobotArm()
        arm._capabilities = dataclasses.replace(arm.capabilities, owns_a_controller=True, controller="10.0.0.9")
        self.assertIn("not a PlannedArm", _unplanned(arm))


class APlannedArmOverridesOnlyTheDeclaredHooksTests(unittest.TestCase):
    def test_no_registered_planned_arm_overrides_more(self) -> None:
        planned = 0
        for vendor in _registered_arms.vendors():
            cls = type(_registered_arms.build(vendor))
            if not issubclass(cls, PlannedArm):
                continue
            planned += 1
            with self.subTest(vendor=vendor):
                self.assertLessEqual(_overrides(cls), PlannedArm._VENDOR_HOOKS)
        self.assertGreaterEqual(planned, 1, "no registered arm is a planned arm: the fence fenced nothing")

    def test_the_capability_mixins_override_nothing(self) -> None:
        for name in planned_capabilities.__all__:
            mixin = getattr(planned_capabilities, name)
            with self.subTest(mixin=name):
                self.assertEqual(set(), _overrides(type(name, (mixin, PlannedArm), {})))

    def test_an_arm_that_answers_move_itself_is_found(self) -> None:
        """The positive control, a synthetic vendor that carries its own motion path."""

        class _Rogue(PlannedArm):
            def move(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
                raise AssertionError("never called")

            def _boxed(self, pose: Any) -> Any:  # type: ignore[override]
                return pose

        self.assertEqual({"move", "_boxed"}, _overrides(_Rogue))
        self.assertEqual({"move"}, _overrides(_Rogue) - PlannedArm._VENDOR_HOOKS)


class NoPublicAttributeHandsOutTheControllerTests(unittest.TestCase):
    def test_no_registered_arm_exposes_its_link_but_the_counted_ones(self) -> None:
        exposed: set[tuple[str, str]] = set()
        for vendor in _registered_arms.vendors():
            arm = _registered_arms.build(vendor)
            exposed |= {(vendor, name) for name in _exposed(arm)}
        self.assertEqual(set(), exposed - set(_EXPOSES), "a public attribute hands out the controller link")
        self.assertEqual(set(), set(_EXPOSES) - exposed, "an exception nothing needs any more: remove it")

    def test_a_planned_arm_with_a_public_transport_is_found(self) -> None:
        """The positive control: a property that returns the transport, and one that returns what it holds."""
        from tests._fake_transport import fake_transport
        from tests.test_a_planned_arm_is_built_by_keyword import _pieces

        class _Leaky(PlannedArm):
            @property
            def transport(self) -> Any:
                return self._transport

        arm = _Leaky(**_pieces(fake_transport()))
        self.assertEqual(["transport"], _exposed(arm))


if __name__ == "__main__":
    unittest.main()
