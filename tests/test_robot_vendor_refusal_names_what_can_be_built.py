"""The arm-vendor refusal must offer only vendors this checkout can actually build.

MEASURED 2026-09-09, before the repair::

    RobotVendor.from_string("abb_irb")
    ValueError: unknown robot vendor 'abb_irb'; valid: ur, kuka, franka, ros2, sim, dummy

``franka`` and ``ros2`` are offered there, and neither has a registered factory: the enum carries six
members and ``available_vendors()`` four. Both are deliberate reserved slots (``drivers/franka/`` and
``drivers/ros2/`` are package stubs whose ``__init__`` says so in as many words), so the fix is to say
which half of the list they belong to, not to delete them.

⚠ THE CONSEQUENCE IS NOT THE GRIPPER'S CONSEQUENCE, AND THAT IS WHY THIS IS A MESSAGE REPAIR RATHER
THAN A BEHAVIOUR REPAIR. A gripper name with no driver falls through to
``SubstitutionReason.NO_DRIVER`` in ``execution/runtime_pick.py`` and the cell comes up with a
``NullGripper``: it connects, reports every pick a success, and holds nothing. The arm side has no
such fallback. MEASURED, same day::

    create_arm(RobotVendor.FRANKA)
    RobotConnectionError: no driver registered for vendor 'franka'; available: dummy, kuka, sim, ur.

Nothing substitutes, nothing moves, and there is no silent cell on this side. What an operator loses
is the hour between copying ``franka`` out of a refusal that called it valid and finding out at build
time that it is not, so the repair is owed to the message and stops there.

These tests read the refusal itself rather than any helper behind it, because the message is the
product: it is what an operator copies. A name counts as OFFERED when it appears before the word that
opens the reserved clause.
"""

from __future__ import annotations

import re

import pytest

from src.robot.core import RobotVendor

# Importing the package (not just the registry module) is what registers the shipped factories.
from src.robot.drivers import available_vendors, is_vendor_registered

#: A vendor string that is not a member and contains no member value, so the offered-name scan below
#: cannot mistake the rejected input for an offer.
_TYPO = "abb_irb"

#: The word that opens the "recognised name, no driver here" half of the message.
_RESERVED_MARKER = "reserved"


def _refusal(value: str = _TYPO) -> str:
    with pytest.raises(ValueError) as excinfo:
        RobotVendor.from_string(value)
    return str(excinfo.value)


def _offered(message: str) -> set[str]:
    """Vendor names the message puts forward as a choice, i.e. outside the reserved clause.

    ⚠ WORD BOUNDARIES, NOT A BARE SUBSTRING, AND THE TWO DISAGREE HERE. ``ur`` is two letters, and
    Pydantic ends every ValidationError with "For further information visit ...": a substring scan
    reads ``ur`` out of "further" and reports it offered no matter what the refusal said. The gripper
    twin of this file gets away with a bare ``in`` because none of its names are short enough to land
    inside an English word.
    """
    head, _, _tail = message.partition(_RESERVED_MARKER)
    return {v.value for v in RobotVendor if re.search(rf"\b{re.escape(v.value)}\b", head)}


def _unregistered() -> set[str]:
    return {v.value for v in RobotVendor} - set(available_vendors())


class TestRefusalOffersOnlyBuildableVendors:
    def test_the_enum_carries_slots_the_registry_cannot_build(self) -> None:
        # The premise of every test below. If this ever goes empty the reserved-slot clause is moot,
        # and the tests that follow would be asserting nothing.
        assert _unregistered(), (
            "every RobotVendor member now has a registered driver; the reserved-slot half of the "
            "refusal has nothing left to describe."
        )

    def test_refusal_offers_exactly_the_vendors_with_a_registered_driver(self) -> None:
        message = _refusal()
        assert _offered(message) == set(available_vendors()), (
            f"the refusal offers {sorted(_offered(message))}, but only {available_vendors()} have a "
            f"registered factory. Full message: {message}"
        )

    def test_every_offered_name_can_actually_be_built(self) -> None:
        message = _refusal()
        for name in sorted(_offered(message)):
            assert is_vendor_registered(name), (
                f"the refusal offers {name!r}, which has no registered factory: a config copying it "
                f"passes schema validation and then dies at create_arm. Full message: {message}"
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
        named_reserved = {
            v.value
            for v in RobotVendor
            if re.search(rf"\b{re.escape(v.value)}\b", message)
        } - _offered(message)
        assert named_reserved == _unregistered(), (
            f"the message calls {sorted(named_reserved)} reserved, the registry says "
            f"{sorted(_unregistered())} have no factory. Full message: {message}"
        )


class TestConfigRefusalIsTheSameMessage:
    """The reported route: an operator copies a name out of a config-validation error.

    ``RobotConfig._validate_vendor`` re-raises ``str(exc)`` verbatim, so the split has to survive the
    trip through Pydantic rather than only existing at the enum.
    """

    def test_robot_config_rejects_a_typo_by_naming_only_buildable_vendors(self) -> None:
        from pydantic import ValidationError

        from src.config.schema.robot import RobotConfig

        with pytest.raises(ValidationError) as excinfo:
            RobotConfig(vendor=_TYPO)
        message = str(excinfo.value)
        for name in sorted(_offered(message)):
            assert is_vendor_registered(name), (
                f"config validation offers {name!r}, which cannot be built here. "
                f"Full message: {message}"
            )
        assert set(available_vendors()) <= _offered(message), (
            f"config validation does not name every buildable vendor. Full message: {message}"
        )
