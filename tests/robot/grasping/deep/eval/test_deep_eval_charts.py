"""Four panels that answer "is my model any good", and the reading one of them corrected.

⛔ THE PANEL THAT PAID FOR ITSELF ON ITS FIRST DRAW. Reading the referee's refusal notes alone
("the jaw reads 132.4 mm, past the 85 mm hand this cell models"), the conclusion written into the
handover was "the model asks for 100 to 166 mm on an 85 mm hand". It does not. MEASURED against the
proposals those refusals point at: they asked for 45.4, 38.5 and 47.0 mm, comfortably inside the
hand. The 132 mm is the CONTACT SEPARATION the referee derives by laying the proposed closing axis
through the object and measuring the object there.

So the defect is not a width head that overshoots. It is a width head that DISAGREES WITH ITS OWN
POSE, by about a factor of three, on geometry up to 230 mm wide. A panel showing only the request
reports zero over the aperture while nineteen proposals were refused for exactly that, which is what
the first version of it did.

⚠ EVERY PANEL EXISTS BECAUSE A READING OF THIS PROJECT'S OWN RUNS NEEDED IT. A hold rate with no
label control has no scale; a bare rate merges three defects with three different repairs; a
collapsed head makes a referee judge one pose eight times and report eight trials.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.robot.grasping.deep.eval.charts import (
    ChartInputs,
    summarise_charts,
    write_eval_charts,
)


def _proposal(scene: str, rank: int, *, width: float = 45.0, distance: float = 3.0,
              x: float = 0.0) -> dict:
    return {"scene_id": scene, "instance_id": 0, "rank": rank,
            "position_mm": [x, 0.0, 30.0], "approach": [0.0, 0.0, -1.0],
            "closing_axis": [1.0, 0.0, 0.0], "width_mm": width, "confidence": 0.4,
            "seed_index": rank, "slot": 0, "object_distance_mm": distance}


def _verdict(*, held: bool, note: str = "drift 0.00 mm; jaw asked 40.0 mm reached 40.0 mm") -> dict:
    return {"source": "generator", "held": held, "note": note, "row_index": 0}


def _write(directory: Path, proposals: list[dict], verdicts: list[dict],
           control: list[dict] | None = None) -> ChartInputs:
    p = directory / "p.jsonl"
    v = directory / "v.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in proposals), encoding="utf-8")
    v.write_text("\n".join(json.dumps(r) for r in verdicts), encoding="utf-8")
    c = None
    if control is not None:
        c = directory / "c.jsonl"
        c.write_text("\n".join(json.dumps(r) for r in control), encoding="utf-8")
    return ChartInputs(p, v, control=c, aperture_mm=85.0, gripper="2f85")


class TheDenominatorsAreBothShownTests(unittest.TestCase):
    """⛔ A refusal is a proposal physics never judged. Counting it as a failure and excluding it are
    two different numbers, and this project has a scar from publishing one without the other."""

    def test_a_refusal_is_excluded_from_the_rate_and_counted_separately(self) -> None:
        with TemporaryDirectory() as name:
            directory = Path(name)
            verdicts = ([_verdict(held=True)] * 2 + [_verdict(held=False)] * 6
                        + [_verdict(held=False, note="refused: the jaw reads 132.4 mm, past the "
                                                     "85 mm hand this cell models")] * 2)
            numbers = write_eval_charts(
                _write(directory, [_proposal("s", i) for i in range(10)], verdicts),
                directory / "chart.png")

        self.assertEqual((numbers["held"], numbers["judged"], numbers["refused"]), (2, 8, 2))
        self.assertAlmostEqual(numbers["hold_rate"], 2 / 8, places=5)
        self.assertAlmostEqual(numbers["hold_rate_counting_refusals"], 2 / 10, places=5)


class ThePoseImpliesSomethingElseTests(unittest.TestCase):

    def test_a_request_INSIDE_the_aperture_still_reports_the_implied_separation(self) -> None:
        """⭐ THE CORRECTION ITSELF. Every proposal asks for 45 mm, well under 85, so a panel drawn
        from the requests alone reports nothing wrong. The referee refused two of them for a derived
        separation of 132 and 230 mm."""
        with TemporaryDirectory() as name:
            directory = Path(name)
            verdicts = [
                _verdict(held=False, note="refused: the jaw reads 132.4 mm, past the 85 mm hand"),
                _verdict(held=False, note="refused: the jaw reads 230.0 mm, past the 85 mm hand"),
                _verdict(held=True),
            ]
            numbers = write_eval_charts(
                _write(directory, [_proposal("s", i, width=45.0) for i in range(3)], verdicts),
                directory / "chart.png")

        self.assertEqual(numbers["over_aperture"], 0, "a 45 mm request is inside an 85 mm hand")
        self.assertEqual(numbers["implied_separation_over_aperture"], 2)
        self.assertAlmostEqual(numbers["implied_separation_max_mm"], 230.0, places=1)

    def test_the_summary_NAMES_the_disagreement_rather_than_the_aperture(self) -> None:
        """The wrong reading was "it asks for too much". The line has to say the other thing."""
        lines = summarise_charts({
            "held": 1, "judged": 3, "refused": 2, "hold_rate": 0.33,
            "hold_rate_counting_refusals": 0.2, "control_hold_rate": 0.22,
            "off_object": 0, "slipped": 2, "over_aperture": 0, "aperture_mm": 85.0,
            "implied_separation_over_aperture": 2, "implied_separation_max_mm": 230.0,
            "distinct_share": 1.0, "chart": "x.png"})
        text = " ".join(lines)

        self.assertIn("disagree", text)
        self.assertIn("230", text)


class TheScaleIsShownOrItsAbsenceIsTests(unittest.TestCase):

    def test_with_a_control_the_share_of_the_ceiling_is_stated(self) -> None:
        """⭐ A hold rate alone has no scale. The referee's own ceiling is nowhere near 100 %."""
        lines = summarise_charts({
            "held": 17, "judged": 301, "refused": 19, "hold_rate": 0.0565,
            "hold_rate_counting_refusals": 0.0531, "control_hold_rate": 0.22,
            "off_object": 14, "slipped": 270, "over_aperture": 0, "aperture_mm": 85.0,
            "implied_separation_over_aperture": 0, "implied_separation_max_mm": None,
            "distinct_share": 1.0, "chart": "x.png"})
        text = " ".join(lines)

        self.assertIn("22.0", text)
        self.assertIn("26 %", text, "the share of the ceiling is the number a customer needs")

    def test_WITHOUT_a_control_the_missing_scale_is_said_in_words(self) -> None:
        """⚠ Not left blank. A single bar with no reference invites reading it out of 100."""
        lines = summarise_charts({
            "held": 17, "judged": 301, "refused": 0, "hold_rate": 0.0565,
            "hold_rate_counting_refusals": 0.0565, "control_hold_rate": None,
            "off_object": 0, "slipped": 284, "over_aperture": 0, "aperture_mm": 85.0,
            "implied_separation_over_aperture": 0, "implied_separation_max_mm": None,
            "distinct_share": 1.0, "chart": "x.png"})
        text = " ".join(lines)

        self.assertIn("NO LABEL CONTROL", text)
        self.assertIn("no scale", text)
        self.assertIn("physics-sample", text, "the line does not say how to get one")

    def test_a_missing_control_file_is_not_an_error(self) -> None:
        """A customer who has not run one still gets the other three panels."""
        with TemporaryDirectory() as name:
            directory = Path(name)
            inputs = _write(directory, [_proposal("s", 0)], [_verdict(held=True)])
            inputs.control = directory / "not_here.jsonl"
            numbers = write_eval_charts(inputs, directory / "chart.png")

        self.assertIsNone(numbers["control_hold_rate"])


class TheFailuresAreSeparatedTests(unittest.TestCase):

    def test_aimed_at_nothing_is_not_counted_as_a_slip(self) -> None:
        """⛔ Three defects with three different repairs. A grasp 200 mm from every object did not
        fail, it never had a chance to; MEASURED on this project's first run, 40 % were like that."""
        with TemporaryDirectory() as name:
            directory = Path(name)
            proposals = [_proposal("s", 0, distance=3.0), _proposal("s", 1, distance=200.0, x=90.0)]
            numbers = write_eval_charts(
                _write(directory, proposals, [_verdict(held=False)] * 2),
                directory / "chart.png")

        self.assertEqual(numbers["off_object"], 1)
        self.assertEqual(numbers["slipped"], 1)


class TheOutputIsSafeForTheTerminalTests(unittest.TestCase):

    def test_every_summary_line_is_ASCII(self) -> None:
        """⚠ NOT COSMETIC. This is printed and the terminal is cp1252, where a non-ASCII character
        raises UnicodeEncodeError."""
        lines = summarise_charts({
            "held": 1, "judged": 2, "refused": 1, "hold_rate": 0.5,
            "hold_rate_counting_refusals": 0.33, "control_hold_rate": 0.22,
            "off_object": 0, "slipped": 1, "over_aperture": 0, "aperture_mm": 85.0,
            "implied_separation_over_aperture": 1, "implied_separation_max_mm": 132.4,
            "distinct_share": 0.5, "chart": "x.png"})

        for line in lines:
            with self.subTest(line):
                self.assertTrue(line.isascii(), line)

    def test_the_chart_is_actually_written(self) -> None:
        with TemporaryDirectory() as name:
            directory = Path(name)
            target = directory / "sub" / "chart.png"
            write_eval_charts(
                _write(directory, [_proposal("s", i) for i in range(4)],
                       [_verdict(held=i == 0) for i in range(4)]),
                target)

            self.assertTrue(target.is_file())
            self.assertGreater(target.stat().st_size, 5_000, "the figure is suspiciously small")


if __name__ == "__main__":
    unittest.main()
