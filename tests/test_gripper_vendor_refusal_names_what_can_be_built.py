"""The gripper-vendor refusal must offer only vendors this checkout can actually build.

MEASURED 2026-09-09, before the repair: ``GripperVendor`` carried eight members and
``available_gripper_vendors()`` six. ``GripperVendor.from_string`` built its "valid" list from the
ENUM, so a typo was answered with ``robotiq, franka_hand, schunk, vacuum, jaw_io, onrobot, dummy,
none``. ``franka_hand`` and ``schunk`` have no registered factory: a config that copies one of them
out of that message passes schema validation, then falls through to the
``SubstitutionReason.NO_DRIVER`` substitution in ``execution/runtime_pick.py`` and the cell comes up
with a ``NullGripper``. It connects, reports every pick a success, and holds nothing.

The two names are deliberate reserved slots, so the fix is not to delete them. It is the split the
arm side already makes in ``drivers/host.py`` (``Host.require``): "not a name" and "a name with no
driver here" are different mistakes with different fixes, and one message can say both.

These tests read the refusal itself rather than any helper behind it, because the message is the
product: it is what an operator copies. A name counts as OFFERED when it appears before the word
that opens the reserved clause.
"""

from __future__ import annotations

import pytest

from src.robot.core import GripperVendor

# Importing the package (not just the registry module) is what registers the shipped factories.
from src.robot.grippers import (
    available_gripper_vendors,
    is_gripper_vendor_registered,
)

#: A vendor string that is not a member and contains no member value as a substring, so the
#: offered-name scan below cannot mistake the rejected input for an offer.
_TYPO = "hand_e"

#: The word that opens the "recognised name, no driver here" half of the message.
_RESERVED_MARKER = "reserved"


def _refusal(value: str = _TYPO) -> str:
    with pytest.raises(ValueError) as excinfo:
        GripperVendor.from_string(value)
    return str(excinfo.value)


def _offered(message: str) -> set[str]:
    """Vendor names the message puts forward as a choice, i.e. outside the reserved clause."""
    head, _, _tail = message.partition(_RESERVED_MARKER)
    return {v.value for v in GripperVendor if v.value in head}


def _unregistered() -> set[str]:
    return {v.value for v in GripperVendor} - set(available_gripper_vendors())


class TestRefusalOffersOnlyBuildableVendors:
    def test_the_enum_carries_slots_the_registry_cannot_build(self) -> None:
        # The premise of every test below. If this ever goes empty the reserved-slot clause is moot,
        # and the tests that follow would be asserting nothing.
        assert _unregistered(), (
            "every GripperVendor member now has a registered driver; the reserved-slot half of the "
            "refusal has nothing left to describe."
        )

    def test_refusal_offers_exactly_the_vendors_with_a_registered_driver(self) -> None:
        message = _refusal()
        assert _offered(message) == set(available_gripper_vendors()), (
            f"the refusal offers {sorted(_offered(message))}, but only "
            f"{available_gripper_vendors()} have a registered factory. Full message: {message}"
        )

    def test_every_offered_name_can_actually_be_built(self) -> None:
        message = _refusal()
        for name in sorted(_offered(message)):
            assert is_gripper_vendor_registered(name), (
                f"the refusal offers {name!r}, which has no registered factory: a config copying it "
                f"builds a NullGripper and the cell comes up with no end-effector. "
                f"Full message: {message}"
            )

    def test_reserved_slots_are_named_as_reserved_rather_than_offered(self) -> None:
        message = _refusal()
        offered = _offered(message)
        for name in sorted(_unregistered()):
            assert name in message, (
                f"{name!r} is a real enum member and the message never mentions it, so an operator "
                f"who wrote it cannot learn from this refusal why it does not work. "
                f"Full message: {message}"
            )
            assert name not in offered, (
                f"{name!r} is offered as a choice but has no driver in this repo. "
                f"Full message: {message}"
            )

    def test_the_reserved_clause_matches_the_registry_in_both_directions(self) -> None:
        # The rot guard. A driver that lands without leaving the reserved clause, or a member added
        # without entering it, turns this red instead of shipping a message that lies.
        message = _refusal()
        named_reserved = {v.value for v in GripperVendor if v.value in message} - _offered(message)
        assert named_reserved == _unregistered(), (
            f"the message calls {sorted(named_reserved)} reserved, the registry says "
            f"{sorted(_unregistered())} have no factory. Full message: {message}"
        )


class TestConfigRefusalIsTheSameMessage:
    """The reported route: an operator copies a name out of a config-validation error."""

    def test_gripper_config_rejects_a_typo_by_naming_only_buildable_vendors(self) -> None:
        from pydantic import ValidationError

        from src.config.schema.robot import GripperConfig

        with pytest.raises(ValidationError) as excinfo:
            GripperConfig(vendor=_TYPO)
        message = str(excinfo.value)
        for name in sorted(_offered(message)):
            assert is_gripper_vendor_registered(name), (
                f"config validation offers {name!r}, which cannot be built here. "
                f"Full message: {message}"
            )
        assert set(available_gripper_vendors()) <= _offered(message), (
            f"config validation does not name every buildable vendor. Full message: {message}"
        )
