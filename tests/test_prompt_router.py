"""Prompt router: unit rules, the normalizer, and the GOLDEN PROMPT SET.

The golden set below is the specification. The router's thresholds are judgement rather than
measurement (nobody has yet measured where GroundingDINO's grounding accuracy falls off on this cell),
so this table is where that judgement is written down, reviewed, and argued with. **If a line here
looks wrong, the line is the bug report** -- change the expectation first, then the rules.

Pure Python: no weights, no GPU, no network. It runs in CI on every commit.
"""

from __future__ import annotations

import unittest

from src.models.routing import (
    MAX_SIMPLE_WORDS,
    Route,
    RouteReason,
    RuleBasedRouter,
    analyse,
    normalize_simple_prompt,
    route,
)

# --------------------------------------------------------------------------------------------------
# THE GOLDEN SET -- (prompt, expected route, expected reason)
#
# Balanced DE/EN x simple/complex on purpose. The German simple-looking entries are NOT mistakes:
# German always routes to the VLM, because the simple path's text encoder is English-trained, and a
# German noun phrase grounds badly rather than failing loudly. That is the single most consequential
# rule in the router and it deserves the most examples.
# --------------------------------------------------------------------------------------------------
GOLDEN: tuple[tuple[str, Route, RouteReason], ...] = (
    # --- English, simple: the fast path's whole purpose -------------------------------------------
    ("a red cube", Route.SIMPLE, RouteReason.PLAIN_NOUN_PHRASE),
    ("red cube", Route.SIMPLE, RouteReason.PLAIN_NOUN_PHRASE),
    ("the mug", Route.SIMPLE, RouteReason.PLAIN_NOUN_PHRASE),
    ("a screwdriver", Route.SIMPLE, RouteReason.PLAIN_NOUN_PHRASE),
    ("small red cube", Route.SIMPLE, RouteReason.PLAIN_NOUN_PHRASE),
    ("the blue plastic box", Route.SIMPLE, RouteReason.PLAIN_NOUN_PHRASE),
    ("a toy rhino", Route.SIMPLE, RouteReason.PLAIN_NOUN_PHRASE),
    ("cardboard tray", Route.SIMPLE, RouteReason.PLAIN_NOUN_PHRASE),
    ("the sugar box", Route.SIMPLE, RouteReason.PLAIN_NOUN_PHRASE),
    ("a yellow banana", Route.SIMPLE, RouteReason.PLAIN_NOUN_PHRASE),

    # --- English, complex: each names the property the phrase grounder cannot represent ------------
    ("the cube that is on the box", Route.VLM, RouteReason.RELATIVE_CLAUSE),
    ("the object which is upside down", Route.VLM, RouteReason.RELATIVE_CLAUSE),
    ("not the red cube", Route.VLM, RouteReason.NEGATION),
    ("the cube without a label", Route.VLM, RouteReason.NEGATION),
    ("the broken part", Route.VLM, RouteReason.STATE_WORD),
    ("a cracked housing", Route.VLM, RouteReason.STATE_WORD),
    ("the empty tray", Route.VLM, RouteReason.STATE_WORD),
    ("the largest cube", Route.VLM, RouteReason.COMPARATIVE),
    ("the leftmost object", Route.VLM, RouteReason.COMPARATIVE),
    ("the cube behind the tray", Route.VLM, RouteReason.SPATIAL_RELATION),
    ("the mug next to the box", Route.VLM, RouteReason.SPATIAL_RELATION),
    ("every cube", Route.VLM, RouteReason.QUANTIFIER),
    ("all the parts", Route.VLM, RouteReason.QUANTIFIER),
    ("the cube and the cup", Route.VLM, RouteReason.CONJUNCTION),
    ("the small shiny metal cube", Route.VLM, RouteReason.MULTI_ATTRIBUTE),
    ("pick up the thing i pointed at before", Route.VLM, RouteReason.TOO_LONG),

    # --- German: routes to the VLM regardless of how simple it looks -------------------------------
    ("ein roter Würfel", Route.VLM, RouteReason.NON_ENGLISH),
    ("der Becher", Route.VLM, RouteReason.NON_ENGLISH),
    ("die Schachtel", Route.VLM, RouteReason.NON_ENGLISH),
    ("ein Schraubendreher", Route.VLM, RouteReason.NON_ENGLISH),
    ("das gelbe Teil", Route.VLM, RouteReason.NON_ENGLISH),
    ("greif den Würfel", Route.VLM, RouteReason.NON_ENGLISH),
    ("nimm die blaue Kiste", Route.VLM, RouteReason.NON_ENGLISH),
    ("der kaputte Würfel", Route.VLM, RouteReason.NON_ENGLISH),
    ("das beschädigte Bauteil", Route.VLM, RouteReason.NON_ENGLISH),
    ("nicht den roten Würfel", Route.VLM, RouteReason.NON_ENGLISH),
    ("alle Würfel", Route.VLM, RouteReason.NON_ENGLISH),
    ("der größte Würfel", Route.VLM, RouteReason.NON_ENGLISH),
    ("der Würfel hinter der Kiste", Route.VLM, RouteReason.NON_ENGLISH),
    ("welche Teile sind kaputt", Route.VLM, RouteReason.NON_ENGLISH),
    ("Würfel", Route.VLM, RouteReason.NON_ENGLISH),          # umlaut alone is enough
    ("rote Kiste", Route.VLM, RouteReason.NON_ENGLISH),      # no umlaut; caught by vocabulary

    # --- other languages: caught ONLY when they carry non-ASCII letters -----------------------------
    # (the accentless case is a known limitation, pinned in its own test below rather than hidden here)
    ("el cubo pequeño", Route.VLM, RouteReason.NON_ENGLISH),
    ("le cube préféré", Route.VLM, RouteReason.NON_ENGLISH),
)


class GoldenPromptTests(unittest.TestCase):
    """Every golden prompt routes exactly as the table says."""

    def test_every_golden_prompt(self) -> None:
        for prompt, expected_route, expected_reason in GOLDEN:
            with self.subTest(prompt=prompt):
                decision = route(prompt)
                self.assertEqual(
                    decision.route, expected_route,
                    f"{prompt!r} -> {decision.describe()}; signals={decision.signals.to_dict()}",
                )
                self.assertEqual(
                    decision.reason, expected_reason,
                    f"{prompt!r} -> {decision.describe()}; signals={decision.signals.to_dict()}",
                )

    def test_the_set_stays_balanced(self) -> None:
        """A golden set that drifts all one way stops testing the boundary it exists to pin."""
        simple = [g for g in GOLDEN if g[1] is Route.SIMPLE]
        self.assertGreaterEqual(len(GOLDEN), 40, "the set is meant to be ~40 prompts")
        self.assertGreaterEqual(len(simple), 8, "too few SIMPLE cases -- the fast path is untested")

    def test_no_duplicate_prompts(self) -> None:
        prompts = [g[0].lower() for g in GOLDEN]
        self.assertEqual(len(prompts), len(set(prompts)))


class RoutingRuleTests(unittest.TestCase):
    def test_empty_prompt_is_simple_not_expensive(self) -> None:
        """Routing is not validation: an empty prompt must fail cheaply and loudly, not silently cost VRAM."""
        for prompt in ("", "   ", "\n\t"):
            decision = route(prompt)
            self.assertIs(decision.route, Route.SIMPLE)
            self.assertIs(decision.reason, RouteReason.EMPTY_PROMPT)

    def test_reason_order_reports_the_most_specific_property(self) -> None:
        """Several rules fire on this; the operator should see the most informative one."""
        decision = route("not the largest cube behind the tray")
        self.assertIs(decision.route, Route.VLM)
        self.assertIs(decision.reason, RouteReason.NEGATION)
        signals = decision.signals
        self.assertTrue(signals.has_negation and signals.has_comparative and signals.has_spatial_relation)

    def test_german_outranks_every_other_reason(self) -> None:
        """A German prompt is unusable on the simple path whatever else is true of it."""
        decision = route("nicht der größte Würfel hinter der Kiste")
        self.assertIs(decision.reason, RouteReason.NON_ENGLISH)

    def test_accentless_non_english_is_a_known_limitation(self) -> None:
        """A stated limit, asserted so it cannot rot into a surprise.

        Language detection is vocabulary-based for German (the cell's operating language) plus a
        non-ASCII-letter check for everything else. A prompt in a third language that happens to use no
        accented characters is therefore NOT detected, and grounds badly on the English-trained simple
        path instead of failing loudly. Widening this means a language-ID model -- a weight load in
        order to decide whether to load weights -- or one vocabulary per language. Neither is justified
        before a second operator language exists; when one does, THIS test is the one that changes.
        """
        self.assertIs(route("le cube rouge").route, Route.SIMPLE)
        self.assertFalse(analyse("il cubo rosso").non_english)

    def test_ambiguous_words_do_not_trigger_false_german(self) -> None:
        """A false NON_ENGLISH silently doubles the cost of every English pick -- guard the collisions."""
        for prompt in ("the die", "a can", "the war memorial", "in the bin", "so many cubes"):
            with self.subTest(prompt=prompt):
                self.assertFalse(analyse(prompt).non_english, f"{prompt!r} wrongly read as German")

    def test_attribute_threshold_boundary(self) -> None:
        two = route("small red cube")
        three = route("small red plastic cube")
        self.assertIs(two.route, Route.SIMPLE)
        self.assertEqual(two.signals.attributes, 2)
        self.assertIs(three.route, Route.VLM)
        self.assertIs(three.reason, RouteReason.MULTI_ATTRIBUTE)

    def test_length_threshold_boundary(self) -> None:
        at_limit = " ".join(["cube"] * MAX_SIMPLE_WORDS)
        over = " ".join(["cube"] * (MAX_SIMPLE_WORDS + 1))
        self.assertIs(route(at_limit).route, Route.SIMPLE)
        self.assertIs(route(over).reason, RouteReason.TOO_LONG)

    def test_case_and_punctuation_do_not_change_the_route(self) -> None:
        for variant in ("A RED CUBE", "a red cube.", "  a  red   cube  "):
            with self.subTest(variant=variant):
                self.assertIs(route(variant).route, Route.SIMPLE)

    def test_signals_are_json_safe_for_telemetry(self) -> None:
        payload = route("the broken part").to_dict()
        self.assertEqual(payload["route"], "vlm")
        self.assertEqual(payload["reason"], "state_word")
        self.assertIsInstance(payload["signals"], dict)
        for value in payload["signals"].values():  # type: ignore[union-attr]
            self.assertIsInstance(value, (int, bool))

    def test_router_protocol_is_satisfied(self) -> None:
        router = RuleBasedRouter()
        self.assertEqual(router.route("a red cube"), route("a red cube"))

    def test_decisions_are_deterministic_and_comparable(self) -> None:
        self.assertEqual(route("the largest cube"), route("the largest cube"))


class NormalizerTests(unittest.TestCase):
    """GroundingDINO's caption convention: lowercase, one trailing period."""

    def test_lowercases_and_terminates(self) -> None:
        self.assertEqual(normalize_simple_prompt("A Red Cube"), "a red cube.")

    def test_is_idempotent(self) -> None:
        once = normalize_simple_prompt("A Red Cube")
        self.assertEqual(normalize_simple_prompt(once), once)

    def test_does_not_double_terminate(self) -> None:
        self.assertEqual(normalize_simple_prompt("a red cube."), "a red cube.")
        self.assertEqual(normalize_simple_prompt("which cube?"), "which cube?")

    def test_collapses_whitespace(self) -> None:
        self.assertEqual(normalize_simple_prompt("  a   red \n cube "), "a red cube.")

    def test_empty_stays_empty_rather_than_becoming_a_lone_period(self) -> None:
        """A bare '.' is a caption with one empty phrase: grounds nothing, and looks real in a log."""
        for prompt in ("", "   ", "\n"):
            self.assertEqual(normalize_simple_prompt(prompt), "")

    def test_multi_phrase_captions_survive(self) -> None:
        self.assertEqual(normalize_simple_prompt("Red Cube . Blue Box"), "red cube . blue box.")


if __name__ == "__main__":
    unittest.main()
