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

WHERE THE SEMANTIC HUES ARE TEXT. A status pill, chip, tile or banner never colours its words: its tint
groups it, its dot hues it, and the word stays in `--fg`. That rule is what makes the palette survive both
themes and a reader who cannot separate red from green, and it is asserted here too. A few LINES are coloured
by their status (a warning in the event stream, the inline `.caution`, a refused overlay toggle), so every
semantic hue the stylesheet uses as a text colour is checked like any text tier, on every surface, in every
theme. And text that sits ON a tinted surface (a chip's label, a banner's code) is checked on that tint.
"""

from __future__ import annotations

import colorsys
import re
import unittest
from pathlib import Path

_CSS = Path(__file__).resolve().parents[1] / "frontend" / "src" / "styles.css"
_FRONTEND = Path(__file__).resolve().parents[1] / "frontend"

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


def _stamped(css: str) -> dict[str, str]:
    """The dark palette the HTML entries stamp (``data-theme="dark"``): the console's default."""
    start = css.index(':root[data-theme="dark"] {')
    return _tokens(css[start:css.index("\n}", start)])


def _hue(hex_colour: str) -> float:
    value = hex_colour.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return colorsys.rgb_to_hls(r, g, b)[0] * 360


def _hue_distance(a: str, b: str) -> float:
    d = abs(_hue(a) - _hue(b)) % 360
    return min(d, 360 - d)


def _percentages(block: str) -> dict[str, float]:
    """`--name: 13%;` pairs out of one CSS block, as fractions."""
    return {name: int(value) / 100 for name, value in re.findall(r"--([a-z-]+):\s*(\d+)%\s*;", block)}


def _block_texts(css: str) -> dict[str, str]:
    """The three token blocks' raw text: light, the dark media query, and the stamped dark."""
    light_start = css.index(":root {")
    dark_start = css.index('@media (prefers-color-scheme: dark)')
    stamped_start = css.index(':root[data-theme="dark"] {')
    return {
        "light": css[light_start:css.index("}", light_start)],
        "dark": css[dark_start:css.index("\n  }", dark_start)],
        "stamped dark": css[stamped_start:css.index("\n}", stamped_start)],
    }


def _mix(colour: str, base: str, weight: float) -> str:
    """``color-mix(in srgb, colour weight, base)``, as the browser computes it."""
    a, b = (colour.lstrip("#"), base.lstrip("#"))
    channels = (
        round(int(a[i:i + 2], 16) * weight + int(b[i:i + 2], 16) * (1 - weight)) for i in (0, 2, 4)
    )
    return "#" + "".join(f"{c:02x}" for c in channels)


def _rule(css: str, selector: str) -> str:
    """The body of the rule whose selector list is exactly ``selector`` (the first one); fails loudly if absent."""
    match = re.search(rf"(?:^|[}};]|\*/)\s*{re.escape(selector)}\s*\{{([^}}]*)\}}", css, re.M)
    if match is None:
        raise AssertionError(f"styles.css has no rule for {selector!r}")
    return match.group(1)


def _colour_token(body: str) -> str | None:
    """The token a rule body sets as its text ``color`` (not a background or border colour), if any."""
    match = re.search(r"(?<![\w-])color:\s*var\(--([\w-]+)\)", body)
    return match.group(1) if match else None


def _rem(body: str, prop: str) -> float | None:
    match = re.search(rf"(?<![\w-]){prop}:\s*([\d.]+)rem", body)
    return float(match.group(1)) if match else None


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


class TheControlRoomTokensTests(unittest.TestCase):
    """The tokens commit 2 adds: the halt button, the camera stage, and the logo's lime as the one accent.

    "Halt now" stops the run and latches the arm, and is NOT the emergency stop. Its button is filled with
    ``--halt`` and labelled in ``--halt-fg``; it is pressed in a hurry, so its label must carry, and it must never be
    taken for the arming red (a go button) or the alarm red (the colour of the real e-stop at the cell).
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.css = _CSS.read_text(encoding="utf-8")
        cls.light, cls.dark = _blocks(cls.css)
        cls.stamped = _stamped(cls.css)

    def _themes(self) -> tuple[tuple[str, dict[str, str]], ...]:
        return (("light", self.light), ("dark", self.dark), ("stamped dark", self.stamped))

    def test_every_theme_defines_every_colour_token(self) -> None:
        """A token defined in one block only applies in one of the three states; the halt and the stage included."""
        for name in ("halt", "halt-fg", "stage", "stage-fg", "stage-accent", "brand-neon", "brand-steel",
                     "brand-steel-hi"):
            for theme, palette in self._themes():
                with self.subTest(theme=theme, token=name):
                    self.assertIn(name, palette, f"{theme}: --{name} is not defined")
        self.assertEqual(set(self.light), set(self.dark), "the light and dark palettes name different tokens")

    def test_the_halt_label_is_legible_on_its_fill(self) -> None:
        for theme, palette in self._themes():
            with self.subTest(theme=theme):
                ratio = _contrast(palette["halt-fg"], palette["halt"])
                self.assertGreaterEqual(
                    ratio, _AA, f"{theme}: --halt-fg on --halt is {ratio:.2f}:1, below AA for the halt button's label",
                )

    def test_the_halt_is_neither_the_arming_red_nor_the_alarm_red(self) -> None:
        """At least 15 degrees of hue from both reds, and a DARK label where the arming red's is white: two cues."""
        for theme, palette in self._themes():
            with self.subTest(theme=theme):
                self.assertGreaterEqual(_hue_distance(palette["halt"], palette["arm"]), 15.0)
                self.assertGreaterEqual(_hue_distance(palette["halt"], palette["alarm"]), 15.0)
                self.assertLess(_luminance(palette["halt-fg"]), _luminance(palette["halt"]),
                                f"{theme}: the halt label must be dark on its fill, unlike the arming red's white")

    def test_text_over_the_camera_stage_is_legible_in_both_themes(self) -> None:
        """The stage stays dark in the light theme too (the image is the content); what sits on it must carry."""
        for theme, palette in self._themes():
            for text in ("stage-fg", "stage-accent"):
                with self.subTest(theme=theme, text=text):
                    ratio = _contrast(palette[text], palette["stage"])
                    self.assertGreaterEqual(ratio, _AA, f"{theme}: --{text} on --stage is {ratio:.2f}:1")

    def test_the_one_accent_is_the_logos_lime_in_both_themes(self) -> None:
        """The logo's neon lime sits at a hue of about 72-77 degrees; the light theme's darker lime stays in the family."""
        for theme, palette in self._themes():
            for name in ("accent", "stage-accent"):
                with self.subTest(theme=theme, token=name):
                    hue = _hue(palette[name])
                    self.assertTrue(65.0 <= hue <= 95.0, f"{theme}: --{name} {palette[name]} has hue {hue:.0f}, not lime")
        self.assertGreater(_luminance(self.stamped["accent"]), 0.6, "the dark theme's accent is the NEON lime")


class TheStatusSystemSurvivesTailwindTests(unittest.TestCase):
    """A blocking status keeps the shape of every other status.

    Tailwind v4 generates a utility for every class name it finds in the sources, and ``block`` is a status here: so
    ``.block { display: block }`` is generated, and the utilities layer beats the components layer whatever the
    specificity. ``class="pill block"`` (every blocking preflight row) and ``class="chip block"`` (a halted arm in the
    top bar) then lost their inline-flex layout: no dot, label and value run together. Measured on 2026-10-01 in a
    screenshot of the chip "STEUERUNGangehalten". The fix restates the layout in the utilities layer itself.
    """

    def test_a_blocking_pill_and_chip_keep_their_layout_over_the_block_utility(self) -> None:
        css = _CSS.read_text(encoding="utf-8")
        start = css.index("@layer utilities {")
        utilities = css[start:]
        for selector in (".pill.block", ".chip.block"):
            with self.subTest(selector=selector):
                rule = re.search(rf"[^{{}}]*{re.escape(selector)}[^{{}}]*\{{([^}}]*)\}}", utilities)
                self.assertIsNotNone(rule, f"{selector} is not restated in the utilities layer")
                assert rule is not None
                self.assertIn("display: inline-flex", rule.group(1))


#: The surfaces a status tints (each tone's own rule), and the rules of the words drawn on them; every one of them
#: also carries the body text in ``--fg``. Each tone rule's ``background`` names its hue and what it is mixed over.
_TINTED: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "chip": ((".chip.ok", ".chip.warn", ".chip.block", ".chip.info", ".chip.bench"), (".chip", ".chip .k")),
    "pill": ((".pill.ok", ".pill.warn", ".pill.block", ".pill.bench", ".pill.info"), ()),
    "tile": (
        (".tile.block:not(.zero)", ".tile.warn:not(.zero)", ".tile.bench:not(.zero)", ".tile.ok:not(.zero)"),
        (".tile .t",),
    ),
    "banner": (
        (".banner.ok", ".banner.warn", ".banner.error", ".banner.info"),
        (".banner .code", ".banner .said", "details.payload-toggle summary"),
    ),
    "event gap": ((".ev.gap",), (".ev.gap .seq", ".ev.gap .payload")),
    "dialog band": ((".dialog-motion",), ()),
    "active tab": ((".topnav a.active",), ()),
}

_MIX = re.compile(
    r"background:\s*color-mix\(in srgb,\s*var\(--([\w-]+)\)\s+(?:var\(--([\w-]+)\)|(\d+)%),\s*"
    r"(?:var\(--([\w-]+)\)|transparent)\)"
)

#: Every surface a tint over ``transparent`` can land on.
_SURFACES = ("ground", "panel", "panel-alt", "inset")


class TheWordsOnATintedSurfaceAreLegibleTests(unittest.TestCase):
    """A chip's label, a banner's code, a tile's caption: text that sits ON a status tint is checked on that tint.

    The tint darkens a dark panel and colours a light one, so a tier that clears AA on the plain panel can fall below it
    on the tint: measured on 2026-10-01, ``--fg-subtle`` (the chip's label) was 4.19:1 on the dark theme's ok, warn and
    info chips. Every tone, every theme, every tier the rules draw there.
    """

    def test_every_text_tier_on_every_tinted_surface_in_every_theme(self) -> None:
        css = _CSS.read_text(encoding="utf-8")
        for theme, block in _block_texts(css).items():
            palette, amounts = _tokens(block), _percentages(block)
            for name, (tones, texts) in _TINTED.items():
                tiers = {"fg"} | {token for token in (_colour_token(_rule(css, s)) for s in texts) if token}
                for tone in tones:
                    body = _rule(css, tone)
                    mixed = _MIX.search(body)
                    self.assertIsNotNone(mixed, f"{tone} has no tinted background this test can read")
                    assert mixed is not None
                    hue, amount_token, amount_pct, base = mixed.groups()
                    weight = amounts[amount_token] if amount_token else int(amount_pct) / 100
                    own = _colour_token(body)
                    for ground in ((base,) if base else _SURFACES):
                        surface = _mix(palette[hue], palette[ground], weight)
                        for tier in sorted(tiers | ({own} if own else set())):
                            with self.subTest(theme=theme, surface=name, tone=tone, over=ground, text=tier):
                                ratio = _contrast(palette[tier], surface)
                                self.assertGreaterEqual(
                                    ratio, _AA,
                                    f"{theme}: --{tier} {palette[tier]} on {tone} ({surface}, over --{ground}) is "
                                    f"{ratio:.2f}:1, below AA",
                                )


class TheStatusHuesUsedAsTextAreLegibleTests(unittest.TestCase):
    """A line coloured by its status is read like any line: the hue is a text tier wherever the stylesheet makes it one.

    The light theme's hues were chosen for dots and tints, and as text they fell to 2.90:1 (``--ok`` on ``--inset``).
    The set of hues checked is read from the stylesheet (every ``color: var(--<hue>)``), so a new use is checked too.
    """

    def test_every_status_hue_used_as_text_clears_aa_on_every_surface_in_every_theme(self) -> None:
        css = _CSS.read_text(encoding="utf-8")
        used = sorted(set(re.findall(r"(?<![\w-])color:\s*var\(--(ok|warn|alarm|info|bench)\)", css)))
        for theme, block in _block_texts(css).items():
            palette = _tokens(block)
            for hue in used:
                for ground in _SURFACES:
                    with self.subTest(theme=theme, hue=hue, on=ground):
                        ratio = _contrast(palette[hue], palette[ground])
                        self.assertGreaterEqual(
                            ratio, _AA, f"{theme}: --{hue} {palette[hue]} as text on --{ground} is {ratio:.2f}:1",
                        )


class TheBrandLookTests(unittest.TestCase):
    """The owner's fourth round (2026-10-01, binding): the full look of docs/assets/willy_logo.png and willy_banner.png.
    A black stage, the logo's neon lime as the one accent, the banner's L-shaped HUD corner marks, the Willy mark and the
    WORKAHOLIC WILLY wordmark in the top bar, status chips and numbers in monospace like the banner's STILL GRINDING box,
    and system fonts only."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.css = _CSS.read_text(encoding="utf-8")
        cls.light, cls.dark = _blocks(cls.css)
        cls.stamped = _stamped(cls.css)

    def test_the_wordmark_is_set_in_capitals_and_slanted_like_the_logo(self) -> None:
        mark = _rule(self.css, ".brand-name strong")
        self.assertIn("text-transform: uppercase", mark)
        self.assertIn("font-style: italic", mark)
        self.assertIn("font-weight: 900", mark, "the wordmark is the system's heaviest italic, as heavy as the logo's")
        willy = _rule(self.css, ".brand-name strong em")
        self.assertEqual(_colour_token(willy), "accent", "WILLY is the lime half of the wordmark")
        self.assertNotIn("font-style: normal", willy, "WILLY keeps the wordmark's slant")

    def test_the_status_chips_are_monospace_with_tabular_numbers(self) -> None:
        chip = _rule(self.css, ".chip")
        self.assertIn("font-family: var(--font-mono)", chip)
        self.assertIn("font-variant-numeric: tabular-nums", chip)

    def test_the_dark_theme_is_the_logos_black_stage(self) -> None:
        """The ground and the camera stage are near black, as the logo's and the banner's are; the panels sit just
        above them, and the stage stays black in the light theme too (the image is the content)."""
        for theme, palette in (("dark", self.dark), ("stamped dark", self.stamped)):
            for name in ("ground", "inset", "stage"):
                with self.subTest(theme=theme, token=name):
                    self.assertLess(_luminance(palette[name]), 0.003, f"{theme}: --{name} {palette[name]} is not black")
            with self.subTest(theme=theme, token="panel"):
                self.assertLess(_luminance(palette["panel"]), 0.01)
                self.assertGreater(_luminance(palette["panel"]), _luminance(palette["ground"]),
                                   "a card sits just above the black stage")
        self.assertLess(_luminance(self.light["stage"]), 0.003, "light: the camera stage is black too")

    def test_the_mark_keeps_the_logos_colours_in_every_theme(self) -> None:
        """The Willy mark (icons.tsx ``Logo``) is drawn in the logo's own colours whatever the theme: the neon lime is the
        dark theme's accent, the same in every block, and its lime edges stand out on its dark steel (3:1, the floor
        for a graphic, WCAG 1.4.11)."""
        for name in ("brand-neon", "brand-steel", "brand-steel-hi"):
            with self.subTest(token=name):
                self.assertEqual({self.light[name]}, {self.dark[name], self.stamped[name]},
                                 f"--{name} follows the theme; a logo does not")
        neon = self.stamped["brand-neon"]
        self.assertEqual(neon, self.stamped["accent"], "the mark's lime is the console's one accent")
        self.assertTrue(65.0 <= _hue(neon) <= 95.0, f"--brand-neon {neon} is not the logo's lime")
        for steel in ("brand-steel", "brand-steel-hi"):
            with self.subTest(on=steel):
                self.assertGreaterEqual(_contrast(neon, self.stamped[steel]), 3.0)
        icons = (_FRONTEND / "src" / "icons.tsx").read_text(encoding="utf-8")
        logo = icons[icons.index("export function Logo"):]
        for token in ("--brand-neon", "--brand-steel", "--brand-steel-hi"):
            self.assertIn(token, icons, f"the mark does not draw with {token}")
        self.assertNotIn("currentColor", logo, "the mark follows the text colour, so the theme")

    def test_the_hud_frame_draws_the_banners_four_corner_marks(self) -> None:
        """The banner's four L-shaped brackets, from one pseudo-element: two bars per corner, in the accent (the stage's
        neon lime over a camera frame), never in the way of a click; ``.hud`` is the same frame under its first name."""
        frame = _rule(self.css, ".hud-frame, .hud")
        self.assertIn("--hud-color: var(--accent)", frame)
        self.assertIn("position: relative", frame)
        marks = _rule(self.css, ".hud-frame::after, .hud::after")
        self.assertIn("pointer-events: none", marks)
        self.assertIn("no-repeat", marks)
        bars = re.findall(r"linear-gradient\(var\(--hud-color\) 0 0\)\s+(top|bottom)\s+(left|right)\s*/", marks)
        self.assertEqual(8, len(bars), "four corners, two bars each")
        self.assertEqual({("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")}, set(bars))
        stage = _rule(self.css, ".stage.hud-frame, .stage .hud-frame, .stage.hud, .stage .hud")
        self.assertIn("--hud-color: var(--stage-accent)", stage)

    def test_a_readout_is_the_still_grinding_box_and_its_words_are_checked_tiers(self) -> None:
        """The banner's STILL GRINDING box: monospace with tabular numbers, its title in the accent and its keys in
        ``--fg-muted``, on ``--panel``: tiers the text tests above check on every surface."""
        box = _rule(self.css, ".readout")
        self.assertIn("font-family: var(--font-mono)", box)
        self.assertIn("font-variant-numeric: tabular-nums", box)
        self.assertIn("clip-path: polygon(", box, "the box's cut corners")
        self.assertIn("background: var(--panel)", _rule(self.css, ".readout::before"))
        self.assertEqual("accent", _colour_token(_rule(self.css, ".readout-title")))
        self.assertEqual("fg-muted", _colour_token(_rule(self.css, ".readout-row .k")))
        self.assertEqual("fg", _colour_token(_rule(self.css, ".readout-row .v")))

    def test_only_system_fonts(self) -> None:
        """No font file is loaded, from the stylesheet or the two pages: system-ui for text, ui-monospace for chips and
        numbers, and nothing to license or to fetch."""
        stacks = dict(re.findall(r"--font-(sans|display|mono):\s*([^;]+);", self.css))
        self.assertEqual({"sans", "display", "mono"}, set(stacks))
        self.assertTrue(stacks["sans"].startswith("system-ui"), stacks["sans"])
        self.assertTrue(stacks["display"].startswith("system-ui"), stacks["display"])
        self.assertTrue(stacks["mono"].startswith("ui-monospace"), stacks["mono"])
        for name, text in (("styles.css", self.css),
                           ("index.html", (_FRONTEND / "index.html").read_text(encoding="utf-8")),
                           ("demo.html", (_FRONTEND / "demo.html").read_text(encoding="utf-8"))):
            with self.subTest(file=name):
                self.assertNotIn("@font-face", text)
                self.assertNotRegex(text, r"\.(woff2?|ttf|otf)\b")
                self.assertNotIn("fonts.googleapis", text)
                self.assertNotRegex(text, r'"(Inter|JetBrains Mono)"')


class TheStatusStripNeverHidesAStateTests(unittest.TestCase):
    """The four chips say the cell's state in full words on every screen; a narrow window wraps them, never hides one.

    The strip used to scroll sideways with its scrollbar hidden: at 1100 px with the real cell's words, the run chip
    ("Auftrag · hält an … · Teil 12") was cut off with no hint that it was there.
    """

    def test_the_strip_wraps_its_chips_instead_of_scrolling_them_out_of_sight(self) -> None:
        css = _CSS.read_text(encoding="utf-8")
        strip = _rule(css, ".statusbar")
        self.assertIn("flex-wrap: wrap", strip)
        self.assertNotIn("overflow-x: auto", strip)
        self.assertNotIn("scrollbar-width: none", strip)
        self.assertIn("max-width: 100%", _rule(css, ".chip"), "a chip is never wider than the strip")


class TheTopBarIsEasyToHitTests(unittest.TestCase):
    """Build plan 4.4: hit targets of at least 44 px, which is 2.75rem at the 16 px root every size here is in."""

    def test_every_top_bar_control_is_at_least_44_px(self) -> None:
        css = _CSS.read_text(encoding="utf-8")
        for selector, prop in (
            (".seg button", "min-height"),
            (".iconbtn", "height"),
            (".iconbtn", "width"),
            (".topnav a", "min-height"),
            (".textbtn", "min-height"),
        ):
            with self.subTest(control=selector, size=prop):
                value = _rem(_rule(css, selector), prop)
                self.assertIsNotNone(value, f"{selector} sets no {prop} in rem")
                assert value is not None
                self.assertGreaterEqual(value, 2.75, f"{selector} {prop} is {value}rem, below 44 px")


class TheJawsQuestionIsOnTopTests(unittest.TestCase):
    """A waiting jaws question is the one thing the console shows above everything else (build plan 1.6): the tech
    view's Diagnostics drawer and a confirm dialog never cover it. The question is the global dialog the shell mounts
    (``jaws/JawsDialog.tsx``), whose backdrop is ``.jd-backdrop`` in its own sheet; it replaced the shell's notice."""

    def test_the_jaws_dialog_sits_above_every_other_layer(self) -> None:
        css = _CSS.read_text(encoding="utf-8")
        jaws = (_FRONTEND / "src" / "jaws" / "jaws.css").read_text(encoding="utf-8")

        def layer(sheet: str, selector: str) -> int:
            match = re.search(r"z-index:\s*(\d+)", _rule(sheet, selector))
            self.assertIsNotNone(match, f"{selector} has no z-index")
            assert match is not None
            return int(match.group(1))

        for other in (".topbar", ".drawer-backdrop", ".drawer", ".dialog-backdrop"):
            with self.subTest(under=other):
                self.assertGreater(layer(jaws, ".jd-backdrop"), layer(css, other))


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
