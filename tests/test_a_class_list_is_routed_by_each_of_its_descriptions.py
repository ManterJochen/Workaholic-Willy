"""A class list is routed description by description, and goes to the VLM only where one of them needs it.

A sort grounds every rule's kind of part in one call, and a task every bin it looks for (the sorting package,
2026-10-09): ``"green part | red part"`` (``src.models.vlm.qwen.class_list_prompt``). Summed over the whole list, the
router read three plain colour descriptions as one phrase binding three attributes (``multi_attribute``) and three
short ones as one long instruction (``too_long``), and sent to the VLM what GroundingDINO grounds one phrase each
(``"green part . red part . blue part"``). Each description is routed on its own now: the first that needs the VLM, in
the list's order, decides, with its own reason and signals; where none does, the list routes simple. A prompt of one
description routes as it always did (the golden set, ``tests/test_prompt_router.py``).

Pure Python: no weights, no GPU.
"""

from __future__ import annotations

import unittest

from src.models.routed_backend import RoutedPerceptionBackend
from src.models.routing import MAX_SIMPLE_WORDS, Route, RouteReason, RuleBasedRouter, analyse, route
from src.models.routing.rules import MULTI_ATTRIBUTE_THRESHOLD
from src.models.vlm.qwen import class_list_prompt


class AListOfPlainDescriptionsTakesTheFastPathTests(unittest.TestCase):
    def test_three_colours_one_description_each_route_simple(self) -> None:
        prompt = class_list_prompt(["green part", "red part", "blue part"])
        self.assertGreaterEqual(analyse(prompt).attributes, MULTI_ATTRIBUTE_THRESHOLD,
                                "summed over the list, three colours read as one phrase binding three")
        decision = route(prompt)
        self.assertEqual((Route.SIMPLE, RouteReason.PLAIN_NOUN_PHRASE), (decision.route, decision.reason),
                         decision.describe())
        self.assertEqual(analyse(prompt), decision.signals, "the simple route carries the whole list's signals")

    def test_three_short_descriptions_are_no_long_instruction(self) -> None:
        prompt = class_list_prompt(["small red cube", "small green cube", "small blue cube"])
        self.assertGreater(analyse(prompt).words, MAX_SIMPLE_WORDS)
        self.assertIs(Route.SIMPLE, route(prompt).route, route(prompt).describe())

    def test_every_bin_of_a_task_routes_simple(self) -> None:
        for places in (["yellow bin", "blue bin"], ["the yellow bin", "the blue bin", "the grey tray"]):
            with self.subTest(places=places):
                self.assertIs(Route.SIMPLE, route(class_list_prompt(places)).route)


class OneDescriptionThatNeedsTheVlmDecidesTests(unittest.TestCase):
    def test_the_description_that_needs_it_gives_its_reason_and_its_signals(self) -> None:
        decision = route(class_list_prompt(["green part", "the cube that is on the box"]))
        self.assertEqual((Route.VLM, RouteReason.RELATIVE_CLAUSE), (decision.route, decision.reason))
        self.assertEqual(analyse("the cube that is on the box"), decision.signals)

    def test_the_first_in_the_lists_order_decides(self) -> None:
        decision = route(class_list_prompt(["every cube", "der rote Würfel"]))
        self.assertEqual((Route.VLM, RouteReason.QUANTIFIER), (decision.route, decision.reason))
        decision = route(class_list_prompt(["der rote Würfel", "every cube"]))
        self.assertEqual((Route.VLM, RouteReason.NON_ENGLISH), (decision.route, decision.reason))

    def test_a_sorts_pick_prompt_goes_to_the_vlm_as_one_kinds_does(self) -> None:
        """"each separate green part" grounds every green part in a box of its own: a quantifier, the VLM's."""
        one = route("each separate green part")
        both = route(class_list_prompt(["each separate green part", "each separate red part"]))
        self.assertEqual((Route.VLM, RouteReason.QUANTIFIER), (one.route, one.reason))
        self.assertEqual((one.route, one.reason, one.signals), (both.route, both.reason, both.signals))


class OneDescriptionRoutesAsItAlwaysDidTests(unittest.TestCase):
    def test_a_prompt_of_one_description_is_no_list(self) -> None:
        for prompt, expected, reason in (("a red cube", Route.SIMPLE, RouteReason.PLAIN_NOUN_PHRASE),
                                         ("the small shiny metal cube", Route.VLM, RouteReason.MULTI_ATTRIBUTE),
                                         ("the cube and the cup", Route.VLM, RouteReason.CONJUNCTION),
                                         ("der Becher", Route.VLM, RouteReason.NON_ENGLISH),
                                         ("", Route.SIMPLE, RouteReason.EMPTY_PROMPT)):
            with self.subTest(prompt=prompt):
                decision = route(prompt)
                self.assertEqual((expected, reason, analyse(prompt)), (decision.route, decision.reason,
                                                                       decision.signals))

    def test_the_seam_routes_a_list_as_the_function_does(self) -> None:
        prompt = class_list_prompt(["green part", "red part", "blue part"])
        self.assertEqual(route(prompt), RuleBasedRouter().route(prompt))


class TheRoutedBackendTests(unittest.TestCase):
    def test_a_list_of_plain_descriptions_builds_only_the_simple_route(self) -> None:
        built: dict[str, int] = {"simple": 0, "vlm": 0}
        prompts: list[str] = []

        class _Recorder:
            def perceive(self, image_bgr: object, prompt: str) -> tuple[str, ...]:
                prompts.append(prompt)
                return ()

        def factory(name: str):  # noqa: ANN202
            def build() -> _Recorder:
                built[name] += 1
                return _Recorder()
            return build

        routed = RoutedPerceptionBackend(simple_factory=factory("simple"), vlm_factory=factory("vlm"))
        routed.perceive(None, class_list_prompt(["green part", "red part", "blue part"]))
        self.assertEqual({"simple": 1, "vlm": 0}, built)
        self.assertEqual(["green part | red part | blue part."], prompts, "the simple route's caption, normalised")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
