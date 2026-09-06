"""The console's text is legible in a hall, and this recomputes that from the stylesheet itself.

WHY A TEST AND NOT A NOTE. A palette is edited by eye, and contrast is the property eyes are worst at
judging — a colour that looks fine on the laptop it was chosen on is the same colour on a panel bolted
next to a cell, under hall lighting, read over somebody's shoulder. This console also puts real weight
on its quietest text tier: `--fg-subtle` carries the DENOMINATOR under every figure on the demo page
("over 2 047 logged attempts"), and a rate whose denominator is hard to read is a rate presented
without its denominator, which is the thing `api/history.py` exists to prevent.

Measured when this was written, on the palette as first shipped:

    --fg-subtle #7d8aa6 on --panel     #ffffff   3.47:1     below AA
    --fg-subtle #7d8aa6 on --panel-alt #f5f3ed   3.12:1     below AA
    --accent    #0b7f96 on --ground    #fbfaf7   4.48:1     below AA, as link text
    --fg-subtle #66738f on --panel     #111a2e   3.64:1     below AA  (dark)

The fix was the smallest lightness step on each hue that clears 4.5:1 everywhere the token lands, so
this file is not defending a look — it is defending a floor, and it reads the numbers out of
`styles.css` rather than duplicating them, so a future palette edit is checked rather than trusted.

WHAT IS DELIBERATELY NOT ASSERTED. Semantic hues (`--ok`, `--warn`, `--alarm`, `--bench`, `--info`)
are NOT checked as text colours, because the stylesheet never uses them as text: a status pill's tint
groups it, its dot hues it, and the word itself stays in `--fg`. That rule is what makes the palette
survive both themes and a reader who cannot separate red from green, and it is asserted here too.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

_CSS = Path(__file__).resolve().parents[1] / "frontend" / "src" / "styles.css"

#: WCAG 2.1 AA for normal-size text. The console's body is 13.5px, so nothing here qualifies for the
#: 3:1 large-text allowance.
_AA = 4.5


def _luminance(hex_colour: str) -> float:
    value = hex_colour.lstrip("#")
    channels = [int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a: str, b: str) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _tokens(block: str) -> dict[str, str]:
    """`--name: #rrggbb;` pairs out of one CSS block. Non-colour tokens are ignored."""
    return {
        name: colour.lower()
        for name, colour in re.findall(r"--([a-z-]+):\s*(#[0-9a-fA-F]{6})\s*;", block)
    }


def _blocks(css: str) -> tuple[dict[str, str], dict[str, str]]:
    """The light palette and the dark one, read from the stylesheet as it ships."""
    light_start = css.index(":root {")
    light = _tokens(css[light_start:css.index("}", light_start)])

    dark_start = css.index('@media (prefers-color-scheme: dark)')
    dark = _tokens(css[dark_start:css.index("\n  }", dark_start)])
    return light, dark


class TheTextIsLegibleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.css = _CSS.read_text(encoding="utf-8")
        cls.light, cls.dark = _blocks(cls.css)

    def _check(self, theme: str, palette: dict[str, str]) -> None:
        grounds = ("ground", "panel", "panel-alt", "inset")
        # Every text tier against every surface it can sit on. `--accent` is in here because it is
        # LINK text, not only a border colour.
        for text in ("fg", "fg-muted", "fg-subtle", "accent"):
            for ground in grounds:
                with self.subTest(theme=theme, text=text, on=ground):
                    ratio = _contrast(palette[text], palette[ground])
                    self.assertGreaterEqual(
                        ratio, _AA,
                        f"{theme}: --{text} {palette[text]} on --{ground} {palette[ground]} "
                        f"is {ratio:.2f}:1, below AA ({_AA}:1) for 13.5px text",
                    )

    def test_light_theme(self) -> None:
        self._check("light", self.light)

    def test_dark_theme(self) -> None:
        self._check("dark", self.dark)

    def test_the_primary_button_label_is_legible_on_its_own_accent(self) -> None:
        for theme, palette in (("light", self.light), ("dark", self.dark)):
            with self.subTest(theme=theme):
                self.assertGreaterEqual(
                    _contrast(palette["accent-fg"], palette["accent"]), _AA,
                    f"{theme}: button.primary's label does not carry on its own fill",
                )

    def test_white_on_the_arming_red_is_legible_in_both_themes(self) -> None:
        """The two buttons that command motion. If any label must be readable, it is these."""
        for theme, palette in (("light", self.light), ("dark", self.dark)):
            with self.subTest(theme=theme):
                self.assertGreaterEqual(_contrast("#ffffff", palette["arm"]), _AA)

    def test_the_dark_stamp_matches_the_media_query_exactly(self) -> None:
        """Two copies of the dark palette exist by necessity; drifting them is the bug they invite.

        The media query serves the un-stamped document (system preference) and
        ``:root[data-theme="dark"]`` serves the explicit toggle. Both must exist -- a colour defined
        in only one applies in only one of the three states -- and both must say the same thing.
        """
        stamp_start = self.css.index(':root[data-theme="dark"] {')
        stamped = _tokens(self.css[stamp_start:self.css.index("\n}", stamp_start)])
        self.assertEqual(stamped, self.dark, "the two dark palettes have drifted apart")

    def test_the_three_text_tiers_stay_distinguishable(self) -> None:
        """Raising a floor must not flatten the hierarchy into one grey."""
        for theme, palette in (("light", self.light), ("dark", self.dark)):
            with self.subTest(theme=theme):
                fg, muted, subtle = (
                    _luminance(palette["fg"]),
                    _luminance(palette["fg-muted"]),
                    _luminance(palette["fg-subtle"]),
                )
                order = [fg, muted, subtle] if theme == "dark" else [subtle, muted, fg]
                self.assertEqual(order, sorted(order, reverse=True),
                                 f"{theme}: fg / fg-muted / fg-subtle are out of order")
                self.assertGreater(abs(muted - subtle), 0.02, "muted and subtle read as one tier")


class ColourNeverCarriesTheTextTests(unittest.TestCase):
    """The rule the whole status system rests on, asserted rather than remembered."""

    def test_no_semantic_hue_is_ever_used_as_a_pill_text_colour(self) -> None:
        css = _CSS.read_text(encoding="utf-8")
        for status in ("ok", "warn", "block", "bench", "info"):
            rule = re.search(rf"\.pill\.{status}\s*\{{(.*?)\}}", css, re.S)
            self.assertIsNotNone(rule, f".pill.{status} has no rule")
            assert rule is not None
            body = rule.group(1)
            self.assertIn("color: var(--fg)", body,
                          f".pill.{status} must keep its text in --fg; the tint does the grouping")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
