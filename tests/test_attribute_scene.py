"""The P5 attribute scene, checked where it can be checked without Isaac.

The scene's whole claim is that **no single word identifies an object** — that is what makes it an
instrument for comparing perception routes rather than another cube demo. A claim like that decays
silently: someone adds a fifth object, or renames one, and the scene still runs, still prints
numbers, and the numbers no longer mean what the runner says they mean. So the claim is asserted
here, next to the arithmetic that turns a grounded box into a scored answer.

Everything tested here is pure: the prompt table, the scene geometry, the projection, and the
box→object attribution. The pick itself is on-box and measured on-box.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.willy_sim.run_attribute_pick import (
    ATTRIBUTE_OBJECTS,
    GROUNDING_TOLERANCE_PX,
    OBJECT_SIZE_MM,
    PROMPTS,
    attribute_specs,
    grounded_index,
    models_for_route,
    project_base_points_to_image,
    target_for_prompt,
)


class SceneAmbiguityTests(unittest.TestCase):
    """The property the measurement rests on: every attribute alone is ambiguous."""

    def test_no_single_attribute_identifies_an_object(self) -> None:
        for axis, values in (("colour", [o[0].split()[0] for o in ATTRIBUTE_OBJECTS]),
                             ("shape", [o[1] for o in ATTRIBUTE_OBJECTS])):
            for value in set(values):
                with self.subTest(axis=axis, value=value):
                    self.assertGreater(
                        values.count(value), 1,
                        f"{value!r} identifies exactly one object, so a prompt containing it needs no "
                        f"attribute binding -- the scene has stopped testing what it claims to test",
                    )

    def test_every_colour_shape_pair_is_unique(self) -> None:
        names = [o[0] for o in ATTRIBUTE_OBJECTS]
        self.assertEqual(len(set(names)), len(names), "two objects share a description")

    def test_objects_are_geometrically_identical(self) -> None:
        """Same grasp problem four times, so a route comparison is not a geometry comparison."""
        specs = attribute_specs()
        self.assertEqual({s.size_mm for s in specs}, {OBJECT_SIZE_MM})
        self.assertEqual({s.mass_kg for s in specs}, {specs[0].mass_kg})
        self.assertEqual({s.static_friction for s in specs}, {specs[0].static_friction})
        self.assertEqual({round(s.position_mm[0], 6) for s in specs}, {450.0})

    def test_neighbours_leave_room_for_the_open_jaw(self) -> None:
        """80 mm pre-open on a 30 mm object reaches 25 mm past its surface; neighbours are 60 mm away."""
        ys = sorted(spec.position_mm[1] for spec in attribute_specs())
        gaps = [b - a for a, b in zip(ys, ys[1:])]
        self.assertTrue(all(gap - OBJECT_SIZE_MM[1] >= 50.0 for gap in gaps), f"gaps too tight: {gaps}")


class PromptTableTests(unittest.TestCase):
    def test_every_object_is_addressed_in_both_languages(self) -> None:
        for language in ("en", "de"):
            targets = sorted(idx for _t, idx, lang in PROMPTS if lang == language)
            self.assertEqual(
                targets, list(range(len(ATTRIBUTE_OBJECTS))),
                f"the {language} half must address each object exactly once -- an unbalanced table "
                f"makes the English/German split a comparison of different questions",
            )

    def test_every_prompt_needs_both_attributes(self) -> None:
        """A prompt that names only a colour or only a shape has no unique answer to score against."""
        colours = {o[0].split()[0] for o in ATTRIBUTE_OBJECTS}
        for prompt, target, language in PROMPTS:
            with self.subTest(prompt=prompt):
                name = ATTRIBUTE_OBJECTS[target][0]
                rivals = [
                    other[0] for other in ATTRIBUTE_OBJECTS
                    if other[0] != name and (
                        other[0].split()[0] in colours and other[0].split()[0] == name.split()[0]
                        or other[1] == ATTRIBUTE_OBJECTS[target][1]
                    )
                ]
                self.assertTrue(rivals, f"{prompt!r} has no rival sharing an attribute ({language})")

    def test_the_german_half_routes_to_the_vlm(self) -> None:
        """The router's NON_ENGLISH rule is what this scene exists to test; assert it still fires."""
        from src.models.routing import Route, RuleBasedRouter

        router = RuleBasedRouter()
        for prompt, _target, language in PROMPTS:
            with self.subTest(prompt=prompt):
                decision = router.route(prompt)
                expected = Route.VLM if language == "de" else Route.SIMPLE
                self.assertIs(decision.route, expected, decision.describe())

    def test_umlauts_are_optional_when_typing_a_prompt(self) -> None:
        self.assertEqual(target_for_prompt("der rote Wuerfel"), target_for_prompt("der rote Würfel"))
        self.assertIsNotNone(target_for_prompt("DER ROTE WÜRFEL "))
        self.assertIsNone(target_for_prompt("the broken cube"))


class ProjectionTests(unittest.TestCase):
    """A nadir camera 1 m above the table: the maths that turns a GT position into a pixel."""

    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    #: CAMERA -> BASE for a camera at (450, 0, 1000) mm looking straight down, CV-optical convention
    #: (camera +Z into the scene, +Y down the image => base -Z and base -Y).
    CAM_TO_BASE = np.array([
        [1.0, 0.0, 0.0, 450.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 1000.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    def test_a_point_under_the_camera_lands_on_the_principal_point(self) -> None:
        pixel = project_base_points_to_image(np.array([[450.0, 0.0, 50.0]]), self.CAM_TO_BASE, self.K)
        np.testing.assert_allclose(pixel[0], [320.0, 240.0], atol=1e-6)

    def test_offsets_keep_the_sign_this_extrinsic_implies(self) -> None:
        """The failure this guards is a mirrored extrinsic: it lands every grasp on a distractor.

        This rig's camera +Y is base -Y, so the object at base y=-135 must land BELOW the principal
        row and the one at y=+135 above it. A 180 deg in-plane error swaps exactly those two -- and
        it is invisible at y=0, which is why M2 never caught it and the clutter runners did.
        """
        pixels = project_base_points_to_image(
            np.array([[450.0, -135.0, 50.0], [450.0, 135.0, 50.0]]), self.CAM_TO_BASE, self.K,
        )
        self.assertGreater(pixels[0][1], 240.0)
        self.assertLess(pixels[1][1], 240.0)
        self.assertAlmostEqual(pixels[0][0], pixels[1][0])  # the row moves, the column does not

    def test_a_mirrored_extrinsic_is_distinguishable(self) -> None:
        mirrored = np.diag([-1.0, -1.0, 1.0, 1.0]) @ self.CAM_TO_BASE
        point = np.array([[450.0, -135.0, 50.0]])
        self.assertFalse(np.allclose(
            project_base_points_to_image(point, self.CAM_TO_BASE, self.K),
            project_base_points_to_image(point, mirrored, self.K),
        ))

    def test_a_point_behind_the_camera_is_not_a_pixel(self) -> None:
        pixels = project_base_points_to_image(np.array([[450.0, 0.0, 2000.0]]), self.CAM_TO_BASE, self.K)
        self.assertTrue(np.isnan(pixels).all())


class AttributionTests(unittest.TestCase):
    PIXELS = np.array([[200.0, 240.0], [280.0, 240.0], [360.0, 240.0], [440.0, 240.0]])

    def test_a_box_on_an_object_attributes_to_it(self) -> None:
        self.assertEqual(grounded_index([340.0, 220.0, 380.0, 260.0], self.PIXELS), 2)

    def test_a_box_between_two_objects_attributes_to_neither(self) -> None:
        """Half the spacing, so 'somewhere in the middle' is NONE rather than a coin flip."""
        self.assertIsNone(grounded_index([300.0, 220.0, 340.0, 260.0], self.PIXELS))

    def test_a_box_around_the_whole_scene_attributes_to_neither(self) -> None:
        self.assertIsNone(grounded_index([150.0, 100.0, 500.0, 400.0], self.PIXELS))

    def test_the_tolerance_is_half_the_object_spacing(self) -> None:
        spacing = float(np.diff(self.PIXELS[:, 0]).min())
        self.assertLessEqual(GROUNDING_TOLERANCE_PX, spacing / 2.0)


class RouteOverrideTests(unittest.TestCase):
    """``--route`` is the A/B knob; it must change exactly the two fields it claims to."""

    def _models(self):  # noqa: ANN202 - a test fixture, typed by use
        """The loaded ModelsConfig -- the same fixture the pipeline tests use."""
        from src.config.loader import load_config
        from src.config.schema.models.models_schema import PipelineConfig

        return load_config().models.model_copy(update={"pipeline": PipelineConfig()})

    def test_simple_pins_the_phrase_grounder_and_silences_the_router(self) -> None:
        models = models_for_route(self._models(), "simple")
        self.assertEqual(models.pipeline.zero_shot.backend, "grounded_sam")
        self.assertFalse(models.pipeline.router.enabled)

    def test_vlm_pins_the_vlm_without_routing(self) -> None:
        models = models_for_route(self._models(), "vlm")
        self.assertEqual(models.pipeline.zero_shot.backend, "vlm")
        self.assertFalse(models.pipeline.router.enabled)

    def test_auto_is_the_shipping_configuration(self) -> None:
        models = models_for_route(self._models(), "auto")
        self.assertEqual(models.pipeline.zero_shot.backend, "vlm")
        self.assertTrue(models.pipeline.router.enabled)

    def test_the_override_leaves_everything_else_alone(self) -> None:
        before = self._models()
        after = models_for_route(before, "vlm")
        self.assertEqual(after.objectdetector, before.objectdetector)
        self.assertEqual(after.pipeline.zero_shot.segmenter, before.pipeline.zero_shot.segmenter)
        self.assertEqual(after.pipeline.zero_shot.vlm, before.pipeline.zero_shot.vlm)

    def test_an_unknown_route_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            models_for_route(self._models(), "qwen")


if __name__ == "__main__":
    unittest.main()
