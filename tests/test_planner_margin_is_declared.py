"""A UR cell that plans with cuRobo declares its planner margin, or it does not plan (B1, S17).

The planner margin is the clearance the sidecar plans WITH so that it stops proposing configurations the exact mesh
guard then refuses. It is not derived from the guard's own ``min_distance_mm``, and it cannot be: how much margin a
planner can absorb depends on how tightly its spheres fit THAT arm with THAT hand. Measured, and it cost a cell: a UR5e
plans fine at 10 mm, a UR3e finds no plan at all at 10 mm and is fine at 4 to 6 mm, and deriving one from the other
took the UR3e from 10 of 10 to 0 of 10.

So the schema's old default of ``0.0`` was an implied answer to a safety question nobody asked: a real UR cell that
says nothing got a planner with NO margin, planned to 9.44 mm against a 10 mm guard, and lost picks to what read like
bad grasping. ``None`` now means undeclared, and a UR cuRobo cell refuses to start a planner while it is.

Every case imports the module inside the test, so the schema case and the driver case are each seen failing for their
own reason rather than both dying on one missing import.
"""

from __future__ import annotations

import unittest


class TheSchemaHasNoImpliedMarginTests(unittest.TestCase):
    def test_a_cell_that_says_nothing_declares_nothing(self) -> None:
        from src.config.schema.robot import RobotConfig

        self.assertIsNone(RobotConfig.model_validate({"vendor": "ur"}).safety.self_collision.planner_margin_mm)

    def test_the_shipped_tree_declares_nothing_either(self) -> None:
        """The base tree is deliberately silent: the margin depends on the arm and the hand, so a cell declares it."""
        from src.config import load_robot_config

        self.assertIsNone(load_robot_config().safety.self_collision.planner_margin_mm)

    def test_the_profiles_that_do_declare_one_all_declare_the_same_number(self) -> None:
        """4.0 family-wide since B6 (2026-09-16), and it is a measurement rather than a tidy-up.

        The refitted sphere map is fatter where the arm is thin: the hand reaches 12 mm past its own geometry
        and wrist_1 8 mm, across a pair whose real clearance on a UR5 is 13.8 mm. At the old 10 mm the UR5 and
        the UR10 have no pose at all, over every one of 371,293 candidates, with every hand. Measured on the
        UR5: 4 and 6 find a retract, 8 and 10 refuse everything. 4 is what the UR3e bring-up had already
        measured for itself, so the two values became one. robot.sim.yaml carries the measurement.
        """
        from src.config import load_robot_config

        for profile, expected in (("sim", 4.0), ("ursim,ursim_curobo", 4.0), ("ursim,ursim_curobo,ursim_ur3", 4.0)):
            with self.subTest(profile=profile):
                self.assertEqual(
                    expected, load_robot_config(profile=profile).safety.self_collision.planner_margin_mm)

    def test_an_explicit_zero_survives_the_round_trip(self) -> None:
        """⭐ THE CONTROL that undeclared and zero are two different things, which is the whole point of None.

        ``model_fields_set`` would not survive a dump and reload; ``None`` does.
        """
        from src.config.schema.robot import RobotConfig

        cfg = RobotConfig.model_validate({"vendor": "ur", "safety": {"self_collision": {"planner_margin_mm": 0.0}}})
        self.assertEqual(0.0, cfg.safety.self_collision.planner_margin_mm)
        again = RobotConfig.model_validate(cfg.model_dump())
        self.assertEqual(0.0, again.safety.self_collision.planner_margin_mm)


class TheRefusalNamesTheKeyTests(unittest.TestCase):
    def _cfg(self, **changes):
        from src.config.schema.robot import RobotConfig

        payload: dict = {"vendor": "ur", "gripper": {"model": "robotiq_2f85"}}
        payload.update(changes)
        return RobotConfig.model_validate(payload)

    def test_a_ur_curobo_cell_with_no_margin_is_refused_by_name(self) -> None:
        from src.robot.safety.planning.margin import planner_margin_refusal

        refusal = planner_margin_refusal(self._cfg())
        assert refusal is not None
        what, fix = refusal
        self.assertIn("robot.safety.self_collision.planner_margin_mm", fix)
        self.assertIn("10", fix)
        self.assertIn("margin", what)

    def test_a_declared_margin_of_zero_is_a_declaration(self) -> None:
        """⭐ THE CONTROL. A cell may choose 0 and say so; evidence then decides whether that combination plans.

        A check written on truthiness rather than on 'was it declared' passes every other test in this file and fails
        exactly this one.
        """
        from src.robot.safety.planning.margin import planner_margin_refusal

        self.assertIsNone(planner_margin_refusal(self._cfg(safety={"self_collision": {"planner_margin_mm": 0.0}})))

    def test_nothing_else_is_refused(self) -> None:
        from src.robot.safety.planning.margin import planner_margin_refusal

        self.assertIsNone(planner_margin_refusal(self._cfg(ur={"motion_planner": "ik"})),
                          "an ik cell starts no planner, so it has no planner margin to declare")
        self.assertIsNone(planner_margin_refusal(self._cfg(vendor="sim")),
                          "the sim declares 10 mm in its own layer, and is not a UR cell")

    def test_the_reading_says_how_the_two_margins_stand(self) -> None:
        from src.robot.safety.planning.margin import MarginReading, MarginStatus

        unset = MarginReading.of(planner_mm=None, guard_mm=10.0)
        self.assertIs(MarginStatus.UNSET, unset.status)
        self.assertIn("undeclared", unset.render())
        self.assertIsNone(unset.to_dict()["planner_mm"])

        self.assertIs(MarginStatus.OK, MarginReading.of(planner_mm=10.0, guard_mm=10.0).status)
        below = MarginReading.of(planner_mm=4.0, guard_mm=10.0)
        self.assertIs(MarginStatus.BELOW_GUARD, below.status)
        for expected in ("4", "10"):
            self.assertIn(expected, below.render())
        self.assertIs(MarginStatus.ABOVE_GUARD, MarginReading.of(planner_mm=12.0, guard_mm=10.0).status)


class TheDriverRefusesToStartAPlannerTests(unittest.TestCase):
    def _arm(self, margin: "float | None"):
        from src.config.schema.robot import RobotConfig
        from src.robot.drivers.ur.arm import URRobotArm

        payload: dict = {"vendor": "ur", "gripper": {"model": "robotiq_2f85"}}
        if margin is not None:
            payload["safety"] = {"self_collision": {"planner_margin_mm": margin}}
        return URRobotArm(RobotConfig.model_validate(payload))

    def test_a_planner_is_never_started_for_a_cell_that_declared_no_margin(self) -> None:
        from src.robot.safety.planning import CuroboUnavailableError

        with self.assertRaises(CuroboUnavailableError) as caught:
            self._arm(None)._default_curobo_client_factory()()  # noqa: SLF001
        self.assertIn("robot.safety.self_collision.planner_margin_mm", str(caught.exception))

    def test_a_cell_that_declared_one_builds_the_client_it_always_did(self) -> None:
        """⭐ THE CONTROL: the refusal is about the declaration, not about building a client.

        The margin has to be one the retract table judged this pair at, because the table keys its rows by it:
        a pose the planner admits at one clearance it can refuse at another, and the driver says so. 4.0 is
        what every layer declares since B6.
        """
        client = self._arm(4.0)._default_curobo_client_factory()()  # noqa: SLF001
        self.assertEqual(4.0, client._self_collision_margin_mm)  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()
