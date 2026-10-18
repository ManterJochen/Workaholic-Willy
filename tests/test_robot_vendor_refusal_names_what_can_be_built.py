"""The arm-vendor refusal names exactly the vendors this checkout can build, and says which names left.

MEASURED 2026-09-09: ``RobotVendor.from_string("abb_irb")`` answered ``valid: ur, kuka, franka, ros2, sim, dummy``,
offering names with no factory behind them. The reserved names were split into a clause of their own then, and on
2026-10-17 they were removed (the owner's cleanup), so every member has a driver: the refusal offers the registry, and
a tree still naming ``kuka``, ``franka`` or ``ros2`` is told when that name left rather than answered as a typo.

These tests read the refusal itself rather than any helper behind it, because the message is the product: it is what
an operator copies.
"""

from __future__ import annotations

import re

import pytest

from src.robot.core import RobotVendor

# Importing the package (not just the registry module) is what registers the shipped factories.
from src.robot.drivers import available_vendors

#: A vendor string that is not a member and contains no member value, so the offered-name scan below cannot mistake
#: the rejected input for an offer.
_TYPO = "abb_irb"


def _refusal(value: str = _TYPO) -> str:
    with pytest.raises(ValueError) as excinfo:
        RobotVendor.from_string(value)
    return str(excinfo.value)


def _offered(message: str) -> set[str]:
    """Vendor names the message puts forward.

    ⚠ WORD BOUNDARIES, NOT A BARE SUBSTRING. ``ur`` is two letters, and Pydantic ends every ValidationError with
    "For further information visit ...": a substring scan reads ``ur`` out of "further" and reports it offered no matter
    what the refusal said.
    """
    return {v.value for v in RobotVendor if re.search(rf"\b{re.escape(v.value)}\b", message)}


class TestRefusalOffersOnlyBuildableVendors:
    def test_every_member_has_a_registered_driver(self) -> None:
        """The rot guard: a member added without a driver would be offered here and refused at create_arm."""
        assert sorted(v.value for v in RobotVendor) == available_vendors()

    def test_refusal_offers_exactly_the_vendors_with_a_registered_driver(self) -> None:
        message = _refusal()
        assert _offered(message) == set(available_vendors()), message

    @pytest.mark.parametrize("name", ["kuka", "franka", "ros2"])
    def test_a_removed_name_is_told_when_it_left(self, name: str) -> None:
        message = _refusal(name)
        assert "removed on 2026-10-17" in message, message
        assert _offered(message) == set(available_vendors()), message

    def test_a_typo_is_not_told_that_a_name_left(self) -> None:
        assert "removed" not in _refusal()


class TestConfigRefusalIsTheSameMessage:
    """The reported route: an operator copies a name out of a config-validation error.

    ``RobotConfig._validate_vendor`` re-raises ``str(exc)`` verbatim, so the message has to survive the trip through
    Pydantic rather than only existing at the enum.
    """

    def test_robot_config_rejects_a_typo_by_naming_only_buildable_vendors(self) -> None:
        from pydantic import ValidationError

        from src.config.schema.robot import RobotConfig

        with pytest.raises(ValidationError) as excinfo:
            RobotConfig(vendor=_TYPO)
        assert _offered(str(excinfo.value)) == set(available_vendors()), str(excinfo.value)

    def test_robot_config_tells_a_removed_name_when_it_left(self) -> None:
        from pydantic import ValidationError

        from src.config.schema.robot import RobotConfig

        with pytest.raises(ValidationError) as excinfo:
            RobotConfig(vendor="kuka")
        assert "removed on 2026-10-17" in str(excinfo.value)
