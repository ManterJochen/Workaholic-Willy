"""The ``robot.place`` keys load, are written down, and as shipped do what a task did before them (2026-10-08 night).

Every key of the block defaults to today's place: the jaws open over a box's rim, the hang hangs from the declared
support, every part goes to the one spot of a flat place, the part is carried via the bin's look. A cell turns each on
for itself; the owner's ranges are refused at load where a cell writes past them. One key is on as shipped, the owner's
word of 2026-10-09: a bin the check before a drop lost is looked for again (``relocate``); off, the part goes back and
the task asks, as before.
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ThePlaceBlockTests(unittest.TestCase):
    def test_as_shipped_every_key_is_todays_place(self) -> None:
        from src.config.schema.robot import RobotConfig

        place = RobotConfig().place

        self.assertEqual("over_the_rim", place.release_in_a_box)
        self.assertEqual(30.0, place.below_the_rim_mm)
        self.assertEqual(10.0, place.opening_margin_mm)
        self.assertEqual("declared_support", place.part_bottom)
        self.assertIs(False, place.side_by_side)
        self.assertEqual(20.0, place.spacing_margin_mm)
        self.assertEqual((3, 3, None), (place.grid.rows, place.grid.columns, place.grid.spacing_mm))
        self.assertEqual("via_the_look", place.carry)
        self.assertEqual(15.0, place.rim_floor_margin_mm)
        self.assertIs(True, place.relocate, "the owner's 2026-10-09 word: a bin that moved is looked for again")

    def test_a_cell_that_names_the_keys_loads_them(self) -> None:
        from src.config.schema.robot import RobotConfig

        robot = RobotConfig.model_validate({"place": {
            "release_in_a_box": "below_the_rim", "below_the_rim_mm": 45.0, "part_bottom": "measured",
            "side_by_side": True, "grid": {"rows": 2, "columns": 4, "spacing_mm": 90.0}, "carry": "over_the_rim",
            "relocate": False}})

        self.assertEqual(("below_the_rim", 45.0, "measured", True, "over_the_rim", False),
                         (robot.place.release_in_a_box, robot.place.below_the_rim_mm, robot.place.part_bottom,
                          robot.place.side_by_side, robot.place.carry, robot.place.relocate))
        self.assertEqual((2, 4, 90.0), (robot.place.grid.rows, robot.place.grid.columns, robot.place.grid.spacing_mm))

    def test_the_owners_ranges_are_refused_past_their_ends(self) -> None:
        from pydantic import ValidationError

        from src.config.schema.robot import RobotConfig

        for name, place in (("9 mm under the rim", {"below_the_rim_mm": 9.0}),
                            ("51 mm under the rim", {"below_the_rim_mm": 51.0}),
                            ("a release nobody knows", {"release_in_a_box": "into_the_floor"}),
                            ("a carry nobody knows", {"carry": "sideways"}),
                            ("a grid of no rows", {"grid": {"rows": 0}}),
                            ("a key nobody knows", {"pile": True})):
            with self.subTest(name):
                with self.assertRaises(ValidationError):
                    RobotConfig.model_validate({"place": place})

    def test_the_reference_writes_the_block_down_with_its_defaults(self) -> None:
        import yaml

        from src.config.schema.robot.place_schema import RobotPlaceConfig

        data = yaml.safe_load((ROOT / "config" / "all_keys" / "robot" / "robot.yaml").read_text(encoding="utf-8"))
        written = data["robot"]["place"]

        self.assertEqual(RobotPlaceConfig().model_dump(), written)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
