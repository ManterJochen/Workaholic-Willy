"""The plates between the flange and the hand become collision geometry, or are named for not having any (UM8).

A plate is a TRANSLATION today: `robot.gripper.coupling_plates_mm` shifts the hand's meshes along the approach and
nothing occupies the space it left. UM8 makes that space a body, through the same declared box writer B7 uses.

⭐ **WHAT THE UNMODELLED PLATE ACTUALLY COSTS, MEASURED** (`scripts/curobo/probe_plate_body.py`). The Hand-E on its
20 mm plate, over the 1,483 judged poses, against the exact guard's 10 mm: at a UR flange radius of 31.5 mm the ur3e
turns 0 of 915 clear poses and the ur5 turns 1 of 919, and that one grazes at 9.800 mm. At 50 mm, which is wider
than any UR flange, it is 54 and 29. So the plate is COMPLETENESS and a tool changer is the safety case, and both
go through one writer because only their numbers differ.

⛔ **AND A THICKNESS IS NOT A BODY.** A plate whose cross section nobody declared gets no box at all and is named
instead, in the manner of B7's refusal of an unmeasured palm. The alternative is a body invented from one measured
number and one guessed one, which reads exactly like a measured body to every consumer downstream.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.safety.planning._declared_body import Box, box_spheres, boxes_to_parts, stacked_boxes

#: The approach axis in the hand model's own frame, which is the axis a plate stack grows along.
APPROACH = 1

#: A plate that declares a cross section and one that does not: the Hand-E's own 20 mm adapter, whose thickness is
#: an assumption and whose width nobody has written down, and a quick change coupler of a size that binds.
ADAPTER = ("hande_adapter", 20.0, None)
CHANGER = ("tool_changer", 48.0, (45.0, 45.0))


class AStackGrowsFromTheFlange(unittest.TestCase):
    """The arithmetic, against numbers that can be written down before the code runs."""

    def test_one_plate_sits_between_the_flange_and_its_own_far_face(self) -> None:
        boxes, undeclared = stacked_boxes([CHANGER], axis=APPROACH)
        self.assertEqual(undeclared, ())
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].name, "tool_changer")
        self.assertEqual(boxes[0].centre_mm, (0.0, 24.0, 0.0))
        self.assertEqual(boxes[0].half_extents_mm, (45.0, 24.0, 45.0))

    def test_the_second_plate_starts_where_the_first_one_ends(self) -> None:
        boxes, _ = stacked_boxes([("first", 10.0, (30.0, 30.0)), ("second", 6.0, (20.0, 20.0))], axis=APPROACH)
        self.assertEqual([box.centre_mm[APPROACH] for box in boxes], [5.0, 13.0])
        self.assertEqual([box.half_extents_mm[APPROACH] for box in boxes], [5.0, 3.0])

    def test_a_stack_can_start_somewhere_other_than_zero(self) -> None:
        """Step 6b hangs a camera off the same writer, and it does not start at the flange."""
        boxes, _ = stacked_boxes([CHANGER], axis=APPROACH, start_mm=12.5)
        self.assertEqual(boxes[0].centre_mm[APPROACH], 36.5)

    def test_the_axis_is_named_rather_than_assumed(self) -> None:
        boxes, _ = stacked_boxes([CHANGER], axis=0)
        self.assertEqual(boxes[0].centre_mm, (24.0, 0.0, 0.0))
        self.assertEqual(boxes[0].half_extents_mm, (24.0, 45.0, 45.0))


class AThicknessAloneIsNotABody(unittest.TestCase):
    """⭐ The half that is a decision rather than arithmetic (owner, UM8)."""

    def test_a_plate_with_no_cross_section_yields_no_box_and_is_named(self) -> None:
        boxes, undeclared = stacked_boxes([ADAPTER], axis=APPROACH)
        self.assertEqual(boxes, [])
        self.assertEqual(undeclared, ("hande_adapter",))

    def test_the_named_one_does_not_move_the_ones_that_follow_it(self) -> None:
        """It is still a translation. A plate that carries no body still holds the next plate out."""
        boxes, undeclared = stacked_boxes([ADAPTER, CHANGER], axis=APPROACH)
        self.assertEqual(undeclared, ("hande_adapter",))
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].centre_mm[APPROACH], 44.0)

    def test_a_declared_plate_is_not_named(self) -> None:
        """The control: the refusal has to be able to stay silent, or it says nothing about the other case."""
        _boxes, undeclared = stacked_boxes([CHANGER], axis=APPROACH)
        self.assertEqual(undeclared, ())

    def test_a_plate_of_no_thickness_is_refused_rather_than_stacked(self) -> None:
        with self.assertRaises(ValueError) as refused:
            stacked_boxes([("nothing", 0.0, (10.0, 10.0))], axis=APPROACH)
        self.assertIn("nothing", str(refused.exception))

    def test_an_empty_stack_is_no_boxes_and_no_names(self) -> None:
        """A hand bolted straight to the flange declares an empty list, and that is a real answer."""
        self.assertEqual(stacked_boxes([], axis=APPROACH), ([], ()))


class TheConfigCarriesOnePlateAsOneThing(unittest.TestCase):
    """A plate is one object with a thickness and, where somebody measured it, a cross section (owner, UM8).

    It replaces `coupling_plates_mm`, a bare list of thicknesses, rather than standing beside it: the thickness
    would then be written twice and checked against itself, which is the shape of defect this arc keeps finding.
    """

    def test_a_plate_declares_its_thickness_and_may_declare_a_cross_section(self) -> None:
        from src.config.schema.robot.robot_schema import GripperConfig

        gripper = GripperConfig(coupling_plates=[
            {"name": "hande_adapter", "thickness_mm": 20.0},
            {"name": "tool_changer", "thickness_mm": 48.0, "cross_section_mm": [45.0, 45.0]},
        ])
        assert gripper.coupling_plates is not None
        self.assertEqual([plate.name for plate in gripper.coupling_plates], ["hande_adapter", "tool_changer"])
        self.assertIsNone(gripper.coupling_plates[0].cross_section_mm)
        self.assertEqual(gripper.coupling_plates[1].cross_section_mm, (45.0, 45.0))

    def test_the_placement_is_the_sum_of_the_thicknesses_and_is_not_declared_twice(self) -> None:
        from src.config.schema.robot.robot_schema import GripperConfig

        self.assertIsNone(GripperConfig().coupling_mm, "undeclared is not zero: it is no measurement")
        self.assertEqual(GripperConfig(coupling_plates=[]).coupling_mm, 0.0, "bolted straight to the flange")
        gripper = GripperConfig(coupling_plates=[{"name": "a", "thickness_mm": 20.0},
                                                 {"name": "b", "thickness_mm": 48.0}])
        self.assertEqual(gripper.coupling_mm, 68.0)

    def test_the_old_key_is_refused_by_name_and_names_what_replaced_it(self) -> None:
        from src.config.schema._removed import REMOVED_KEYS

        said = REMOVED_KEYS["robot.gripper.coupling_plates_mm"]
        self.assertIn("coupling_plates", said)
        self.assertIn("thickness_mm", said)

    def test_a_plate_needs_a_name_it_can_be_said_by(self) -> None:
        from pydantic import ValidationError

        from src.config.schema.robot.robot_schema import GripperConfig

        with self.assertRaises(ValidationError):
            GripperConfig(coupling_plates=[{"name": "", "thickness_mm": 20.0}])

    def test_the_shipped_hande_profile_still_places_its_hand_where_it_did(self) -> None:
        """⭐ The control that matters: the placement number is unchanged, so no hand moved in this step."""
        from src.config.loader import load_robot_config

        self.assertEqual(load_robot_config(profile="hande").gripper.coupling_mm, 20.0)


class TheBoxesBecomeTheArraysABundleCarries(unittest.TestCase):
    """One writer for three users: what comes out here is what the guard and the fit already read."""

    def test_a_stack_becomes_parts_in_the_hand_frame(self) -> None:
        boxes, _ = stacked_boxes([CHANGER], axis=APPROACH)
        parts = boxes_to_parts(boxes, frame=6)
        self.assertEqual(sorted(parts), ["tool_changer__f", "tool_changer__frame", "tool_changer__v"])
        self.assertEqual(parts["tool_changer__v"].shape, (8, 3))
        self.assertEqual(parts["tool_changer__f"].shape, (12, 3))
        self.assertEqual(int(parts["tool_changer__frame"][0]), 6)

    def test_the_box_is_the_one_the_stack_computed(self) -> None:
        boxes, _ = stacked_boxes([CHANGER], axis=APPROACH)
        self.assertEqual(boxes[0], Box("tool_changer", (0.0, 24.0, 0.0), (45.0, 24.0, 45.0)))


class ThePlannerGetsSpheresItCanPlanOn(unittest.TestCase):
    """⭐ The same promise B6 made for the arms, but a box needs no search to keep it.

    The cover fit's safe half is that every surface sample ends inside a sphere, and its cost is reach past the
    body. Over a grid of cells whose spheres circumscribe them, both are arithmetic: the cover is complete by
    construction, and the reach follows from the cell size alone. So a customer's plate is covered exactly, at a
    reach the caller states, without the marching the arms needed.
    """

    BOX = Box("tool_changer", (0.0, 24.0, 0.0), (45.0, 24.0, 45.0))

    def _surface(self, box: Box, per_axis: int = 25) -> np.ndarray:
        """Points all over the box's surface, which is what a cover has to swallow."""
        centre = np.asarray(box.centre_mm)
        half = np.asarray(box.half_extents_mm)
        grid = np.linspace(-1.0, 1.0, per_axis)
        points = []
        for axis in (0, 1, 2):
            for sign in (-1.0, 1.0):
                a, b = np.meshgrid(grid, grid)
                local = np.zeros((a.size, 3))
                others = [i for i in (0, 1, 2) if i != axis]
                local[:, others[0]] = a.ravel()
                local[:, others[1]] = b.ravel()
                local[:, axis] = sign
                points.append(local * half + centre)
        return np.concatenate(points)

    def test_every_surface_point_ends_inside_a_sphere(self) -> None:
        spheres = box_spheres(self.BOX, reach_mm=8.0)
        self.assertTrue(spheres, "a box of real size needs at least one sphere")
        points = self._surface(self.BOX)
        centres = np.asarray([s["center"] for s in spheres]) * 1000.0
        radii = np.asarray([s["radius"] for s in spheres]) * 1000.0
        gaps = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2) - radii[None, :]
        self.assertLessEqual(float(gaps.min(axis=1).max()), 0.0, "a surface point lies outside every sphere")

    def test_no_sphere_reaches_further_past_the_box_than_it_was_asked_to(self) -> None:
        """⚠ The reach is how far a sphere POINT lies outside the body, not the sphere's radius.

        The first version of this test added the whole radius to how far the centre sat outside, which for a
        centre inside the box is the radius itself: it read 39.9 mm where the cover reaches 17.4. A sphere
        centred 24 mm inside a face and 39.9 mm across reaches 15.9 mm past it, and that is the number the arms
        are measured by too. So the sphere's surface is sampled and each point measured against the box.
        """
        centre = np.asarray(self.BOX.centre_mm)
        half = np.asarray(self.BOX.half_extents_mm)
        rng = np.random.default_rng(0)
        directions = rng.normal(size=(4000, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        for reach_mm in (4.0, 8.0, 20.0):
            with self.subTest(reach_mm=reach_mm):
                worst = 0.0
                for sphere in box_spheres(self.BOX, reach_mm=reach_mm):
                    surface = np.asarray(sphere["center"]) * 1000.0 + directions * sphere["radius"] * 1000.0
                    outside = np.linalg.norm(np.maximum(np.abs(surface - centre) - half, 0.0), axis=1)
                    worst = max(worst, float(outside.max()))
                self.assertLessEqual(worst, reach_mm + 1e-6, f"the cover reaches {worst:.3f} mm past the box")

    def test_a_smaller_reach_costs_more_spheres(self) -> None:
        """The trade is the same one the arms make, and it is the caller's to make here too."""
        self.assertGreater(len(box_spheres(self.BOX, reach_mm=4.0)), len(box_spheres(self.BOX, reach_mm=16.0)))

    def test_the_spheres_are_metres_where_the_planner_reads_them(self) -> None:
        spheres = box_spheres(self.BOX, reach_mm=8.0)
        self.assertLess(max(abs(v) for s in spheres for v in s["center"]), 1.0)
        self.assertLess(max(s["radius"] for s in spheres), 0.1)

    def test_a_reach_of_nothing_is_refused_rather_than_ground_to_a_halt(self) -> None:
        with self.assertRaises(ValueError):
            box_spheres(self.BOX, reach_mm=0.0)


class ThePlannerAndTheGuardReadTheSamePlates(unittest.TestCase):
    """Both halves take the boxes from the resolved hand, so no second derivation can disagree with the first."""

    def test_a_cell_with_no_measured_plate_adds_no_body_to_the_planner(self) -> None:
        from src.robot.safety.planning._curobo_body_links import coupling_body_link

        self.assertIsNone(coupling_body_link(boxes=[], rotation=np.eye(3).tolist()))

    def test_a_measured_plate_becomes_one_link_under_tool0(self) -> None:
        from src.robot.safety.planning._curobo_body_links import (
            COUPLING_LINK,
            HAND_IGNORE,
            HAND_PARENT,
            coupling_body_link,
        )

        boxes, _ = stacked_boxes([CHANGER], axis=APPROACH)
        link = coupling_body_link(boxes=boxes, rotation=np.eye(3).tolist())
        assert link is not None
        self.assertEqual(link["link"], COUPLING_LINK)
        self.assertEqual(link["parent"], HAND_PARENT)
        self.assertEqual(tuple(link["ignore"]), HAND_IGNORE)
        self.assertEqual(link["fixed_transform"], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                         "a plate does not move relative to the flange, so only the placement turns it")
        self.assertTrue(link["spheres"])

    def test_the_shipped_hande_cell_declares_a_plate_and_still_gets_no_body(self) -> None:
        """⭐ The whole decision in one assertion: the 20 mm holds the hand out and is named, not invented."""
        from src.config.loader import load_robot_config
        from src.contracts import chosen
        from src.robot.safety.planning.hand import planner_hand

        hand = planner_hand(load_robot_config(profile="hande"))
        assert chosen(hand)
        self.assertEqual(hand.coupling_mm, 20.0, "the plate still places the hand")
        self.assertEqual(hand.coupling_boxes, (), "and it is not a body, because nobody measured it across")
        self.assertEqual(hand.coupling_undeclared, ("hande_adapter",), "and the cell can say which one")

    def test_a_cell_that_measured_its_plate_carries_it_on_both_halves(self) -> None:
        from src.robot.safety.planning.hand import planner_hand
        from src.config.schema.robot import RobotConfig

        cfg = RobotConfig.model_validate({
            "vendor": "ur",
            "safety": {"payload": {"enforce": False}},
            "gripper": {"model": "robotiq_hande", "coupling_plates": [
                {"name": "hande_adapter", "thickness_mm": 20.0, "cross_section_mm": [31.5, 31.5]},
            ]},
        })
        hand = planner_hand(cfg)
        self.assertEqual(hand.coupling_mm, 20.0)
        self.assertEqual(hand.coupling_undeclared, ())
        self.assertEqual([box.name for box in hand.coupling_boxes], ["hande_adapter"])
        self.assertEqual(hand.coupling_boxes[0].centre_mm, (0.0, 10.0, 0.0))


class TheChecklistNamesThePlateNobodyMeasured(unittest.TestCase):
    """An operator reading the bring-up list learns which plate their cell is not modelling."""

    def test_the_shipped_hande_cell_is_told_which_plate_is_missing_a_body(self) -> None:
        from src.config.loader import load_robot_config
        from src.robot.execution.real_cell.preflight import _unmodelled_plates

        said = _unmodelled_plates(load_robot_config(profile="hande"))
        assert said is not None
        self.assertIn("hande_adapter", said)
        self.assertIn("cross_section_mm", said)

    def test_a_cell_that_measured_its_plate_is_told_nothing(self) -> None:
        """The control: the sentence has to be able to stay silent, or it says nothing about the other case."""
        from src.config.schema.robot import RobotConfig
        from src.robot.execution.real_cell.preflight import _unmodelled_plates

        cfg = RobotConfig.model_validate({"vendor": "ur", "gripper": {"model": "robotiq_hande", "coupling_plates": [
            {"name": "hande_adapter", "thickness_mm": 20.0, "cross_section_mm": [31.5, 31.5]}]}})
        self.assertIsNone(_unmodelled_plates(cfg))

    def test_a_cell_with_no_plates_at_all_is_told_nothing(self) -> None:
        from src.config.loader import load_robot_config
        from src.robot.execution.real_cell.preflight import _unmodelled_plates

        self.assertIsNone(_unmodelled_plates(load_robot_config(profile="sim")))


if __name__ == "__main__":
    unittest.main()
