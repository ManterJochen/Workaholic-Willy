"""The join: routing a prompt to a perception route, lazily, and recording what it chose.

Pure Python -- no weights, no GPU. The factories here are counters, which is exactly the point: the
laziness and the sharing are the properties worth asserting, and both are invisible to a test that
builds real models.
"""

from __future__ import annotations

import unittest
from unittest import mock

from src.models.routed_backend import RoutedPerceptionBackend
from src.models.routing import Route, RouteReason


class _Recorder:
    """A perception backend that records the prompt it was handed."""

    def __init__(self, name: str) -> None:
        self.name, self.prompts = name, []

    def perceive(self, image_bgr, prompt):  # noqa: ANN001
        self.prompts.append(prompt)
        return (f"{self.name}-object",)


def _backend(**kwargs):
    simple, vlm = _Recorder("simple"), _Recorder("vlm")
    built = {"simple": 0, "vlm": 0}

    def simple_factory():
        built["simple"] += 1
        return simple

    def vlm_factory():
        built["vlm"] += 1
        return vlm

    routed = RoutedPerceptionBackend(simple_factory=simple_factory, vlm_factory=vlm_factory, **kwargs)
    return routed, simple, vlm, built


class RoutingTests(unittest.TestCase):
    def test_a_plain_english_phrase_goes_to_the_simple_route(self) -> None:
        routed, simple, vlm, _ = _backend()
        self.assertEqual(routed.perceive(None, "a red cube"), ("simple-object",))
        self.assertEqual(vlm.prompts, [])
        self.assertIs(routed.last_decision.route, Route.SIMPLE)

    def test_a_german_prompt_goes_to_the_vlm(self) -> None:
        routed, simple, vlm, _ = _backend()
        self.assertEqual(routed.perceive(None, "der kaputte Würfel"), ("vlm-object",))
        self.assertEqual(simple.prompts, [])
        self.assertIs(routed.last_decision.reason, RouteReason.NON_ENGLISH)


class LazinessTests(unittest.TestCase):
    """Eager construction would load GroundingDINO AND Qwen at cell build -- ~9 GB and ~10 s."""

    def test_nothing_is_built_until_a_prompt_selects_it(self) -> None:
        routed, _, _, built = _backend()
        self.assertEqual(routed.built_routes(), ())
        self.assertEqual(built, {"simple": 0, "vlm": 0})

    def test_only_the_chosen_route_is_built(self) -> None:
        routed, _, _, built = _backend()
        routed.perceive(None, "a red cube")
        self.assertEqual(built, {"simple": 1, "vlm": 0})
        self.assertEqual(routed.built_routes(), (Route.SIMPLE,))

    def test_each_route_is_built_once_however_many_picks(self) -> None:
        routed, _, _, built = _backend()
        for prompt in ("a red cube", "the mug", "der Würfel", "die Kiste", "a screwdriver"):
            routed.perceive(None, prompt)
        self.assertEqual(built, {"simple": 1, "vlm": 1})


class PromptHandlingTests(unittest.TestCase):
    def test_the_simple_route_receives_a_normalised_caption(self) -> None:
        """GroundingDINO's text encoder was trained on lowercase, period-terminated phrases."""
        routed, simple, _, _ = _backend()
        routed.perceive(None, "A Red Cube")
        self.assertEqual(simple.prompts, ["a red cube."])

    def test_the_vlm_receives_the_operators_words_untouched(self) -> None:
        """It is being asked to reason about the phrasing -- rewriting it changes the question."""
        routed, _, vlm, _ = _backend()
        routed.perceive(None, "Der KAPUTTE Würfel")
        self.assertEqual(vlm.prompts, ["Der KAPUTTE Würfel"])

    def test_normalisation_can_be_turned_off(self) -> None:
        routed, simple, _, _ = _backend(normalize=False)
        routed.perceive(None, "A Red Cube")
        self.assertEqual(simple.prompts, ["A Red Cube"])


class DecisionVisibilityTests(unittest.TestCase):
    """What P4's console, the PERCEIVED event and GraspAttemptRecord.extra read."""

    def test_last_decision_is_none_before_any_pick(self) -> None:
        routed, _, _, _ = _backend()
        self.assertIsNone(routed.last_decision)

    def test_the_callback_sees_every_decision_in_order(self) -> None:
        seen = []
        routed, _, _, _ = _backend(on_decision=seen.append)
        routed.perceive(None, "a red cube")
        routed.perceive(None, "the broken part")
        self.assertEqual([str(d.route) for d in seen], ["simple", "vlm"])
        self.assertEqual(str(seen[1].reason), "state_word")

    def test_a_raising_callback_never_breaks_a_pick(self) -> None:
        """Telemetry is not allowed to cost a grasp."""
        def explode(_decision):
            raise RuntimeError("telemetry sink is down")

        routed, _, _, _ = _backend(on_decision=explode)
        self.assertEqual(routed.perceive(None, "a red cube"), ("simple-object",))

    def test_the_decision_serialises_for_a_record(self) -> None:
        routed, _, _, _ = _backend()
        routed.perceive(None, "the largest cube")
        payload = routed.last_decision.to_dict()
        self.assertEqual(payload["route"], "vlm")
        self.assertEqual(payload["reason"], "comparative")
        self.assertIn("attributes", payload["signals"])


class SubstitutionTests(unittest.TestCase):
    def test_a_failing_route_is_not_silently_swapped_for_the_other(self) -> None:
        """The VLM route owns refuse-vs-degrade; substituting here would reintroduce the confident
        wrong grasp that the whole arc exists to prevent."""
        routed, simple, _, _ = _backend()
        with mock.patch.object(_Recorder, "perceive", side_effect=RuntimeError("no weights")):
            with self.assertRaises(RuntimeError):
                routed.perceive(None, "der kaputte Würfel")
        self.assertEqual(simple.prompts, [], "the simple route must not have been used as a fallback")

    def test_a_custom_router_replaces_the_rules_without_touching_callers(self) -> None:
        """The PromptRouter seam -- what a learned judge will occupy."""
        from src.models.routing import PromptSignals, RouteDecision

        signals = PromptSignals(0, 0, 0, False, False, False, False, False, False, False)
        always_vlm = mock.Mock()
        always_vlm.route.return_value = RouteDecision(Route.VLM, RouteReason.TOO_LONG, signals)

        routed, simple, vlm, _ = _backend(router=always_vlm)
        routed.perceive(None, "a red cube")   # the rules would have said SIMPLE
        self.assertEqual(simple.prompts, [])
        self.assertEqual(vlm.prompts, ["a red cube"])


if __name__ == "__main__":
    unittest.main()
