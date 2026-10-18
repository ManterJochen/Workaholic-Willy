"""Every arm model the stack can name resolves in the description registry, through its own vendor (P0g slice 11).

Two directions. Every admitted description belongs to a vendor the arm registry can build, so no description waits
for a driver nobody registered. Every registered arm, built as a test builds it (``tests/_registered_arms.py``),
reports a model its description resolves, with the dof it says; the dummy is exempt by a fact this file checks, that
its model has no description at all. And the ``ur`` block's validator admits only UR descriptions: with one vendor
today that changes nothing, and it keeps a second vendor's arm, admitted later, from being a ``robot.ur.model``.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from src.config.arms import admitted_arms, available_arms, load_arm
from src.config.loader import ConfigError
from tests import _registered_arms

#: Registered vendors whose arm names no description, each with the fact that makes it so.
#: The list may only shrink.
_NO_DESCRIPTION: dict[str, str] = {
    "dummy": "an in-memory arm whose model is its joint count ('dummy-6dof'); it has no geometry to describe",
}


class EveryDescriptionHasADriverTests(unittest.TestCase):
    def test_every_admitted_description_belongs_to_a_registered_vendor(self) -> None:
        from src.robot.drivers import available_vendors

        registered = set(available_vendors())
        for model in admitted_arms():
            with self.subTest(model=model):
                self.assertIn(load_arm(model).vendor, registered, "a description no registered driver builds")


class EveryRegisteredArmResolvesItsModelTests(unittest.TestCase):
    def test_every_registered_arm_reports_a_model_its_description_resolves(self) -> None:
        for vendor in _registered_arms.vendors():
            caps = _registered_arms.build(vendor).capabilities
            with self.subTest(vendor=vendor, model=caps.model):
                if vendor in _NO_DESCRIPTION:
                    with self.assertRaises(ConfigError, msg="the exemption's fact no longer holds: remove it"):
                        load_arm(caps.model)
                    continue
                description = load_arm(caps.model)
                self.assertEqual(description.dof, caps.dof)
                if not caps.is_simulated:
                    # A simulated arm names the arm it simulates; a real one names its own vendor's description.
                    self.assertEqual(description.vendor, caps.vendor)

    def test_every_exemption_names_a_registered_vendor(self) -> None:
        self.assertEqual(set(), set(_NO_DESCRIPTION) - set(_registered_arms.vendors()))


class TheUrModelIsAUrDescriptionTests(unittest.TestCase):
    def test_the_filter_changes_nothing_while_every_description_is_a_ur(self) -> None:
        self.assertEqual({"ur"}, {load_arm(model).vendor for model in available_arms()})
        self.assertEqual(admitted_arms(), admitted_arms(vendor="ur"))

    def test_a_second_vendors_arm_admitted_beside_the_urs_is_no_ur_model(self) -> None:
        """The positive control, on a registry with a Franka description admitted beside the URs."""
        from pydantic import ValidationError

        from src.config import arms
        from src.config.schema.robot import RobotConfig

        directory, specs = arms._registry(None)
        franka = SimpleNamespace(vendor="franka", admitted=True, sim=None)
        with mock.patch.object(arms, "_registry", return_value=(directory, {**specs, "fr3": franka})):
            self.assertIn("fr3", arms.admitted_arms())
            self.assertNotIn("fr3", arms.admitted_arms(vendor="ur"))
            self.assertEqual(admitted_arms(vendor="ur"), arms.admitted_arms(vendor="ur"))
            with self.assertRaisesRegex(ValidationError, "unknown UR model 'fr3'"):
                RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": "10.0.0.1", "model": "fr3"}})


if __name__ == "__main__":
    unittest.main()
