"""``container_agitate`` is refused at load unless a container is declared, and a push is 30 mm unless told otherwise.

Owner, 2026-09-29. This cell's parts lie on a table and it declares no container. The agitate executor either
shakes the air where the arm stands or sweeps blind along BASE +X. Until that day, a config naming the action
loaded, and the action was then never planned, because no built-in profile lists it. Now the config is refused
with one sentence. A container counts as declared when ``support.container`` gives its interior box: both
corners, each axis a real span. ``floor_height_mm`` alone raises the surface but gives no interior.

A push is ``recovery.fixture.push_distance_mm`` unless asked for another distance: 30 mm by default, at least
10 mm. The longest push the cell allows is ``recovery.fixture.max_nudge_mm``: 50 mm by default and at most, refused
above (it was 5 mm, less than the Hand-E finger's 10.8 mm thickness). A push distance above the ceiling is refused
at load; ``push_planner.resolve_push_distance`` settles a request against both.
"""

from __future__ import annotations

import unittest
from typing import Any

from pydantic import ValidationError

from src.config.schema.robot import RobotConfig

_FIXTURE = {"center_mm": [0.0, -700.0, 150.0], "half_extents_mm": [300.0, 300.0, 300.0]}
_INTERIOR = {"interior_min_mm": [-200.0, -900.0, 20.0], "interior_max_mm": [200.0, -500.0, 180.0]}


def _config(recovery: dict[str, Any], container: dict[str, Any] | None = None) -> RobotConfig:
    grasping: dict[str, Any] = {"default_mode": "dense_clutter", "recovery": recovery}
    if container is not None:
        grasping["support"] = {"container": container}
    return RobotConfig(vendor="dummy", gripper={"vendor": "none"}, grasping=grasping)


class ContainerAgitateNeedsADeclaredContainer(unittest.TestCase):
    def _refused(self, recovery: dict[str, Any], container: dict[str, Any] | None = None) -> str:
        with self.assertRaises(ValidationError) as caught:
            _config(recovery, container)
        text = str(caught.exception)
        self.assertIn("names container_agitate", text)
        self.assertIn("declares no interior box", text)
        self.assertIn("remove container_agitate or declare the container", text)
        return text

    def test_named_in_allowed_actions_without_a_container(self) -> None:
        self._refused({"enabled": True, "allowed_actions": ["container_agitate"], "fixture": dict(_FIXTURE)})

    def test_named_in_the_per_action_budget_without_a_container(self) -> None:
        self._refused({"enabled": True, "allowed_actions": ["rescan"],
                       "per_action_budget": [["container_agitate", 1]]})

    def test_refused_even_with_recovery_switched_off(self) -> None:
        self._refused({"enabled": False, "allowed_actions": ["container_agitate"], "fixture": dict(_FIXTURE)})

    def test_a_floor_height_alone_is_not_a_declared_container(self) -> None:
        self._refused({"enabled": True, "allowed_actions": ["container_agitate"], "fixture": dict(_FIXTURE)},
                      {"floor_height_mm": 20.0})

    def test_an_interior_with_no_span_is_not_a_declared_container(self) -> None:
        flat = {"interior_min_mm": [-200.0, -900.0, 20.0], "interior_max_mm": [200.0, -500.0, 20.0]}
        self._refused({"enabled": True, "allowed_actions": ["container_agitate"], "fixture": dict(_FIXTURE)}, flat)

    def test_without_a_fixture_either_the_first_refusal_already_names_the_container(self) -> None:
        # The fixture check runs first. Its sentence says the container is needed too, so the operator does
        # not add a fixture only to be refused again.
        with self.assertRaises(ValidationError) as caught:
            _config({"enabled": True, "allowed_actions": ["container_agitate"]})
        text = str(caught.exception)
        self.assertIn("fixture", text)
        self.assertIn("robot.grasping.support.container", text)
        self.assertIn("interior_min_mm and interior_max_mm", text)

    def test_accepted_where_the_container_is_declared(self) -> None:
        cfg = _config({"enabled": True, "allowed_actions": ["container_agitate"], "fixture": dict(_FIXTURE)},
                      dict(_INTERIOR))
        self.assertEqual(cfg.grasping.recovery.allowed_actions, ("container_agitate",))
        self.assertTrue(cfg.grasping.support.container.interior_declared)

    def test_the_other_actions_need_no_container(self) -> None:
        cfg = _config({"enabled": True, "allowed_actions": ["rescan", "next_target", "nudge_target"],
                       "fixture": dict(_FIXTURE)})
        self.assertFalse(cfg.grasping.support.container.interior_declared)

    def test_the_shipped_default_declares_no_container(self) -> None:
        self.assertFalse(RobotConfig(vendor="dummy", gripper={"vendor": "none"}).grasping.support.container
                         .interior_declared)


class ThePushDistance(unittest.TestCase):
    @staticmethod
    def _fixture(**keys: float) -> Any:
        cfg = _config({"enabled": True, "allowed_actions": ["nudge_target"], "fixture": {**_FIXTURE, **keys}})
        assert cfg.grasping.recovery.fixture is not None
        return cfg.grasping.recovery.fixture

    def _refused(self, **keys: float) -> str:
        with self.assertRaises(ValidationError) as caught:
            self._fixture(**keys)
        return str(caught.exception)

    def test_thirty_millimetres_by_default_under_a_fifty_millimetre_ceiling(self) -> None:
        fixture = self._fixture()
        self.assertEqual((fixture.push_distance_mm, fixture.max_nudge_mm), (30.0, 50.0))

    def test_fifty_is_the_cap_and_more_is_refused(self) -> None:
        at_cap = self._fixture(max_nudge_mm=50.0, push_distance_mm=50.0)
        self.assertEqual((at_cap.push_distance_mm, at_cap.max_nudge_mm), (50.0, 50.0))
        self.assertIn("max_nudge_mm", self._refused(max_nudge_mm=50.5))
        self.assertIn("push_distance_mm", self._refused(push_distance_mm=50.5))

    def test_a_push_distance_above_the_ceiling_is_refused_at_load(self) -> None:
        text = self._refused(max_nudge_mm=20.0)
        self.assertIn("push_distance_mm (30 mm) is longer than", text)
        self.assertIn("max_nudge_mm (20 mm)", text)
        lowered = self._fixture(max_nudge_mm=20.0, push_distance_mm=20.0)
        self.assertEqual((lowered.push_distance_mm, lowered.max_nudge_mm), (20.0, 20.0))

    def test_a_push_distance_or_a_ceiling_under_ten_millimetres_is_refused_at_load(self) -> None:
        # Under 10 mm a push cannot open room for a finger: the old 5 mm default never pushed.
        self.assertIn("push_distance_mm", self._refused(push_distance_mm=5.0))
        self.assertIn("max_nudge_mm", self._refused(max_nudge_mm=5.0, push_distance_mm=10.0))
        at_floor = self._fixture(max_nudge_mm=10.0, push_distance_mm=10.0)
        self.assertEqual((at_floor.push_distance_mm, at_floor.max_nudge_mm), (10.0, 10.0))


if __name__ == "__main__":
    unittest.main()
