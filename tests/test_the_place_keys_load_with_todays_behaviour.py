"""The ``robot.motion.place`` keys load, are written down, and as shipped do what a task did before them (2026-10-08
night).

Every key of the block defaults to today's place: the jaws open over a box's rim, every part goes to the one spot of a
flat place, the part is carried via the bin's look, and it hangs from the TCP where the cell declares no hang. A cell
turns each on for itself; a value past what a key takes is refused at load. The place's numbers are the owner's cell's
since 2026-10-17, fixed in ``place_target`` (the owner's cleanup): 30 mm under the rim, 10 mm inside a box's opening,
20 mm of rim air (the owner's 0 to 50 of 2026-10-14 stays the task's own), 20 mm between parts side by side, 15 mm over
the rim on a carry; and a bin the check before a drop lost is always looked for again (the owner, 2026-10-09). Their
keys are refused at load with the sentence that says so.
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The keys that left on 2026-10-17, each with a value a cell could have written.
_FIXED = {"below_the_rim_mm": 45.0, "opening_margin_mm": 5.0, "rim_air_mm": 0.0, "spacing_margin_mm": 30.0,
          "rim_floor_margin_mm": 20.0, "relocate": False}


class ThePlaceBlockTests(unittest.TestCase):
    def test_as_shipped_every_key_is_todays_place(self) -> None:
        from src.config.schema.robot import RobotConfig

        place = RobotConfig().motion.place

        self.assertEqual("over_the_rim", place.release_in_a_box)
        self.assertIs(False, place.side_by_side)
        self.assertEqual((3, 3, None), (place.grid.rows, place.grid.columns, place.grid.spacing_mm))
        self.assertEqual("via_the_look", place.carry)
        self.assertIsNone(place.part_hang_mm, "the owner, 2026-10-14: no hang declared, the part hangs from the TCP")

    def test_a_cell_that_names_the_keys_loads_them(self) -> None:
        from src.config.schema.robot import RobotConfig

        robot = RobotConfig.model_validate({"motion": {"place": {
            "release_in_a_box": "below_the_rim", "side_by_side": True,
            "grid": {"rows": 2, "columns": 4, "spacing_mm": 90.0}, "carry": "over_the_rim", "part_hang_mm": 40.0}}})

        place = robot.motion.place
        self.assertEqual(("below_the_rim", True, "over_the_rim", 40.0),
                         (place.release_in_a_box, place.side_by_side, place.carry, place.part_hang_mm))
        self.assertEqual((2, 4, 90.0), (place.grid.rows, place.grid.columns, place.grid.spacing_mm))

    def test_the_numbers_are_the_owners_cells(self) -> None:
        from src.robot.execution import place_target

        self.assertEqual((30.0, 10.0, 20.0, 20.0, 15.0),
                         (place_target.BELOW_THE_RIM_MM, place_target.OPENING_MARGIN_MM,
                          place_target.RIM_AIR_DEFAULT_MM, place_target.SPACING_MARGIN_MM,
                          place_target.RIM_FLOOR_MARGIN_MM))
        self.assertEqual((0.0, 50.0), (place_target.RIM_AIR_MIN_MM, place_target.RIM_AIR_MAX_MM),
                         "the owner's 0 to 50 mm, a task's own air")

    def test_a_key_that_left_is_refused_at_load_saying_the_cells_number(self) -> None:
        from src.config.schema._removed import removed_key_sentence
        from src.config.tree import ConfigTree

        for key, value in _FIXED.items():
            with self.subTest(key=key):
                said = removed_key_sentence(f"robot.motion.place.{key}")
                self.assertIsNotNone(said)
                assert said is not None  # for mypy; assertIsNotNone above is the real check
                self.assertIn("place_target.py", said)
                refused = ConfigTree.from_directory(profile=None).load().with_values(
                    {f"robot.motion.place.{key}": value})
                self.assertFalse(refused.ok)
                self.assertIn("removed on purpose", refused.error)

    def test_values_past_what_a_key_takes_are_refused(self) -> None:
        from pydantic import ValidationError

        from src.config.schema.robot import RobotConfig

        for name, place in (("the hang's switch it no longer has", {"part_bottom": "measured"}),
                            ("a release nobody knows", {"release_in_a_box": "into_the_floor"}),
                            ("a carry nobody knows", {"carry": "sideways"}),
                            ("a grid of no rows", {"grid": {"rows": 0}}),
                            ("a hang of nothing", {"part_hang_mm": 0.0}),
                            ("a key nobody knows", {"pile": True})):
            with self.subTest(name):
                with self.assertRaises(ValidationError):
                    RobotConfig.model_validate({"motion": {"place": place}})

    def test_the_reference_writes_the_block_down_with_its_defaults(self) -> None:
        import yaml

        from src.config.schema.robot.place_schema import RobotPlaceConfig

        data = yaml.safe_load((ROOT / "config" / "all_keys" / "robot" / "robot.yaml").read_text(encoding="utf-8"))
        written = data["robot"]["motion"]["place"]

        self.assertEqual(RobotPlaceConfig().model_dump(), written)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
