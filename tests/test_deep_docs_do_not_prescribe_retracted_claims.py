"""The retrain runbook and the CLI help told a customer to act on claims this project had retracted.

⛔ TWO OF THEM, and both pointed the customer away from a defensible default.

`--slot-mixing film` sat in the ONE training command the runbook gives. Its justification was a
measured slot spread of 9.6e-09 under `affine`. RETRACTED 2026-09-04: the same metric on the actual
trained artifact gives 34.87 deg, MORE than film's 25.81, and the 9.6e-09 came from 64 seeds that
were themselves feature-identical, which was a serve-time seeding defect with its own repair.

`--axis-mode`'s help said `vector` was "eighteen times" better for the same job. That comparison was
CONFOUNDED and was retracted by the commit that wrote it: the approach control regresses `-normal`,
a linear function of an input channel, while the axis control regresses `normalise(cross(normal, z))`,
which is not the same job. A direct comparison puts the DEFAULT ahead, 0.49 deg against 1.17.

⚠ WHY A TEST AND NOT JUST AN EDIT. A retraction that lives only in a handover file does not reach the
person reading `--help`. Both of these were retracted in `.ai-memory/` and left standing in the code
and the docs for the rest of the day, which is how a customer would have acted on them.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

_RUNBOOK = Path("docs/runbooks/train_your_own_generator.md")
_CLI = Path("src/robot/grasping/deep/__main__.py")


def _step_four_command() -> str:
    """The bash block under `## Step 4. Train`, which is what a customer copies."""
    text = _RUNBOOK.read_text(encoding="utf-8")
    after = text[text.index("## Step 4. Train"):]
    return after[after.index("```bash"):after.index("```", after.index("```bash") + 7)]


class TheRunbookTests(unittest.TestCase):

    def test_the_training_command_does_not_pass_slot_mixing(self) -> None:
        """⭐ The command a customer copies. `--slot-mixing` is an ARM being measured, not a fix, and
        prescribing an arm as though it were settled is how a retracted claim reaches production."""
        self.assertNotIn("--slot-mixing", _step_four_command(),
                         "the runbook prescribes an unsettled architecture flag")

    def test_the_training_command_still_carries_what_IS_settled(self) -> None:
        """The control. Stripping the wrong flag must not strip the right ones.

        ⚠ DELIVERED, NOT SPELLED. `--collapse-floor-deg` used to be listed here literally, and this
        subtest failed the moment Step 4 moved to `--recipe v1`, which SETS it. The question the
        control is actually asking is whether a collapsed fold still gets caught, and a recipe that
        sets the value answers it as well as a flag that spells it. So the recipe's own settings count.
        `--artifact-gripper` cannot come from a recipe: no bundle knows which hand the customer owns.
        """
        from src.robot.grasping.deep.train.recipes import settings

        command = _step_four_command()
        for flag in ("--artifact-gripper", "--clouds", "--out"):
            with self.subTest(flag):
                self.assertIn(flag, command)

        self.assertIn("--recipe v1", command, "the command names no recipe")
        self.assertGreater(settings("v1", "full").get("collapse_floor_deg", 0.0), 0.0,
                           "neither the command nor the recipe arms the collapse floor")

    def test_the_smoke_run_comes_FIRST(self) -> None:
        """⚠ Order is the point of two tiers. A customer who runs the full tier first discovers a
        broken corpus after a night instead of after minutes."""
        command = _step_four_command()

        self.assertLess(command.index("--tier smoke"), command.index("--tier full"))

    def test_the_runbook_SAYS_it_was_retracted_rather_than_going_quiet(self) -> None:
        """A reader who saw the old command needs to know why it changed. Deleting a recommendation
        without a word is how the same claim comes back."""
        text = _RUNBOOK.read_text(encoding="utf-8")
        self.assertIn("RETRACTED", text)
        self.assertIn("34.87", text, "the number that replaced the retracted one is not quoted")


class TheCLIHelpTests(unittest.TestCase):
    """`--help` is the surface a customer reaches without reading any document at all."""

    def _help_for(self, flag: str) -> str:
        source = _CLI.read_text(encoding="utf-8")
        start = source.index(f'train_set.add_argument("{flag}"')
        end = source.index("train_set.add_argument(", start + 10)
        return source[start:end]

    def test_the_slot_mixing_help_no_longer_calls_the_default_a_defect(self) -> None:
        help_text = self._help_for("--slot-mixing")
        self.assertNotIn("MEASURED defect", help_text)
        self.assertNotIn("defect", help_text.split("SEED")[0],
                         "the default is called a defect before the sentence that is about seeding")
        self.assertIn("WHICH IS BETTER IS OPEN", help_text)
        self.assertIn("Under affine the slots do", help_text)
        self.assertIn("differ from one another", help_text)

    def test_the_axis_mode_help_no_longer_claims_the_default_is_worse(self) -> None:
        help_text = self._help_for("--axis-mode")
        self.assertNotIn("Eighteen times worse", help_text)
        self.assertNotIn("eighteen times", help_text.lower())
        self.assertIn("puts the director ahead and a matched control agrees", help_text)
        self.assertIn("default stands; the flag exists to keep", help_text)

    def test_the_axis_help_states_the_outcome_and_warns_off_the_controls(self) -> None:
        """The measurement itself belongs to our corpus and does not ship. What a reader can act on
        is the outcome and the reason the two controls do not settle it, and both are here."""
        help_text = self._help_for("--axis-mode")
        self.assertIn("puts the director ahead and a matched control agrees", help_text)
        self.assertIn("A direct comparison", help_text)
        self.assertIn("Do not settle it through the controls", help_text)
        self.assertIn("is not the same job", help_text)

    def test_every_help_string_stays_ASCII(self) -> None:
        """⚠ NOT COSMETIC. This terminal is cp1252 and `print` of a non-ASCII marker raises
        `UnicodeEncodeError`, so a decorated help string turns `--help` into a crash."""
        source = _CLI.read_text(encoding="utf-8")
        offenders = [
            (index, line) for index, line in enumerate(source.splitlines(), start=1)
            if re.search(r'^\s*(help=|")', line) and any(ord(char) > 127 for char in line)
        ]
        self.assertEqual(offenders, [], f"non-ASCII in a help string: {offenders}")


if __name__ == "__main__":
    unittest.main()
