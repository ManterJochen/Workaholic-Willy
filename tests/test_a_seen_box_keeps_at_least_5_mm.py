"""The exact guard keeps at least 3 mm from a box the camera saw: a config asking less is refused at load.

The file keeps its name from the first floor, 5 mm; the owner lowered the shipped floor to 3 mm on 2026-10-05.

The owner, 2026-10-01 ("Mindestens 5 mm erzwingen"), on verifier W's L5. Since Option 1 the exact guard alone decides
the boxes the camera saw wherever only they refuse the planner's world: cuRobo, asked again with them set aside, no
longer judges them at all there. Before, cuRobo's own world term stood behind the guard as a second check, and a cell
could ask the guard for 0 mm with ``planning_world.perceived.margin_mm`` 10, the sum the step rule asks for. Now
``robot.safety.self_collision.perceived_min_distance_mm`` below 3 is refused at load, with a sentence naming the key,
and the schema says so to anyone who reads it (``>= 3``). The step rule and the voxel rule stay as they were.
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.config.schema.robot import RobotConfig
from src.config.schema.robot.safety_schema import RobotSafetyConfig, SelfCollisionSafetyConfig

_KEY = "robot.safety.self_collision.perceived_min_distance_mm"


def _refused(value: float) -> str:
    """What a load of ``perceived_min_distance_mm`` at ``value`` is refused with."""
    try:
        RobotSafetyConfig.model_validate({"self_collision": {"perceived_min_distance_mm": value}})
    except ValidationError as exc:
        return str(exc)
    raise AssertionError(f"perceived_min_distance_mm {value:g} was meant to be refused at load, and it loaded")


class BelowThreeMillimetresIsRefusedAtLoadTests(unittest.TestCase):
    def test_every_distance_below_3_mm_is_refused_with_a_sentence_naming_the_key(self) -> None:
        """⭐ Red before: 0, 2 and 2.9 mm loaded (the schema took anything from 0)."""
        for value in (0.0, 2.0, 2.9, 2.999):
            with self.subTest(value=value):
                said = _refused(value)
                self.assertIn(_KEY, said)
                self.assertIn(f"({value:g})", said)
                for words in ("below 3 mm", "the exact guard alone"):
                    self.assertIn(words, said)

    def test_the_sentence_says_what_to_set(self) -> None:
        said = _refused(2.0)
        self.assertIn("Set it to 3 or more", said)

    def test_3_mm_and_more_load(self) -> None:
        for value in (3.0, 3.5, 5.0, 10.0, 25.0):
            with self.subTest(value=value):
                config = RobotSafetyConfig.model_validate({"self_collision": {"perceived_min_distance_mm": value}})
                self.assertEqual(config.self_collision.perceived_min_distance_mm, value)  # type: ignore[attr-defined]

    def test_the_default_is_the_floor(self) -> None:
        self.assertEqual(SelfCollisionSafetyConfig().perceived_min_distance_mm, 3.0)

    def test_the_schema_says_the_floor_to_whoever_reads_it(self) -> None:
        """⭐ Red before: the JSON schema (and so ``config describe``) said ``>= 0.0``."""
        leaf = RobotConfig.model_json_schema()["$defs"]["SelfCollisionSafetyConfig"]["properties"][
            "perceived_min_distance_mm"]
        self.assertEqual(leaf["minimum"], 3.0)
        self.assertEqual(leaf["default"], 3.0)

    def test_a_value_that_is_no_number_is_still_refused_as_one(self) -> None:
        """The sentence is for a number below the floor; anything else is refused as the schema always refused it."""
        for value in ("five", None, [5.0]):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    RobotSafetyConfig.model_validate({"self_collision": {"perceived_min_distance_mm": value}})

    def test_the_other_rules_on_the_seen_boxes_stand(self) -> None:
        """3 mm with a 4 mm box margin is still short of a 10 mm step, and the voxel rule still holds while the seen boxes are held nearer than the step."""
        with self.assertRaises(ValidationError) as caught:
            RobotSafetyConfig.model_validate({"self_collision": {"perceived_min_distance_mm": 3.0, "min_distance_mm": 10.0},
                                              "planning_world": {"perceived": {"margin_mm": 4.0}}})
        self.assertIn("the step every path is sampled at", str(caught.exception))
        with self.assertRaises(ValidationError) as caught:
            RobotSafetyConfig.model_validate({"self_collision": {"min_distance_mm": 10.0},
                                              "planning_world": {"perceived": {"voxel_size_mm": 20.0}}})
        self.assertIn("voxel_size_mm", str(caught.exception))


class EveryShippedConfigStillLoadsTests(unittest.TestCase):
    def test_every_shipped_profile_loads_and_keeps_at_least_3_mm(self) -> None:
        from src.config.loader import load_config
        from src.config.tree import default_data_dir

        root = default_data_dir()
        profiles: list[str | None] = [None, *sorted({p.name.split(".")[-2] for p in root.rglob("*.*.yaml")})]
        loaded = 0
        for profile in profiles:
            with self.subTest(profile=profile or "(base)"):
                robot = load_config(profile=profile).robot
                if robot is None:
                    continue
                loaded += 1
                self.assertGreaterEqual(float(robot.safety.self_collision.perceived_min_distance_mm), 3.0)
        self.assertGreater(loaded, 5, "the shipped profiles were not found")


if __name__ == "__main__":
    unittest.main()
