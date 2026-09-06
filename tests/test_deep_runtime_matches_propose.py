"""The cell path and the offline path must be the SAME system.

⛔⛔ FOR FOUR HOURS THEY WERE NOT, AND THE FIX HAD ALREADY BEEN MEASURED. The serve-time seed mixture
was corrected in `propose.py` on 2026-09-04 and NOT in `calculator.py`, which kept the hardcoded
`SeedShares(labelled=0.0, predicted=1.0, random=0.0)`: the exact configuration measured at cosine
**1.0000** between seed features and **25.6 %** distinct proposals, against **71.2 %** once the mixture
came from the artifact.

So `deep propose` and a cell running `calculator: deep` were two different systems under one name, and
every ladder rung, which builds through `build_calculator`, graded the collapsed one. The commit that
fixed it (`ceae488`) touched `propose.py`, `set_loop.py` and a test.

⚠ THIS FILE IS THE GUARD AGAINST THE NEXT ONE. It does not re-test the mixture; it tests that the two
paths agree, which is the property a per-path fix keeps breaking.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from src.robot.grasping.deep.net.set_targets import SeedShares
from src.robot.grasping.deep.eval.propose import serving_shares

_ROOT = Path(__file__).resolve().parents[1]
_CALCULATOR = _ROOT / "src/robot/grasping/deep/calculator.py"
_PROPOSE = _ROOT / "src/robot/grasping/deep/eval/propose.py"


def _shares_argument(path: Path) -> list[str]:
    """What each `sample_seeds(...)` call passes as `shares=`, as source text.

    ⚠ AST, NOT A TEXT SEARCH. My first version grepped for `predicted=1.0` and failed on BOTH files,
    because both explain the defect in a comment. A guard that cannot tell code from prose about code
    is worse than none: it fails on the documentation that exists to prevent the defect.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "sample_seeds"):
            continue
        for keyword in node.keywords:
            if keyword.arg == "shares":
                out.append(ast.unparse(keyword.value))
    return out


class BothPathsTests(unittest.TestCase):

    def test_BOTH_paths_derive_the_mixture_from_the_artifact(self) -> None:
        """⛔ THE DEFECT ITSELF, as the assertion that would have caught it. Every `sample_seeds`
        call in either path must take its mixture from `serving_shares`, never from a literal."""
        for path in (_CALCULATOR, _PROPOSE):
            with self.subTest(path.name):
                passed = _shares_argument(path)
                self.assertTrue(passed, f"{path.name} no longer draws seeds; retire this entry")
                for argument in passed:
                    self.assertIn("serving_shares", argument,
                                  f"{path.name} passes a literal mixture ({argument}) instead of "
                                  f"the one the artifact was trained under")

    def test_NEITHER_path_passes_an_ALL_PREDICTED_literal(self) -> None:
        """The other half, stated positively: no call may name the collapsed configuration."""
        for path in (_CALCULATOR, _PROPOSE):
            with self.subTest(path.name):
                for argument in _shares_argument(path):
                    self.assertNotIn("predicted=1.0", argument)

    def test_the_runtime_KEEPS_the_step_config_it_needs_to_ask(self) -> None:
        """⚠ The calculator could not have derived the mixture even if it wanted to: its loader
        discarded `loaded.step`. A fix that only changed the draw would have had nothing to read."""
        text = _CALCULATOR.read_text(encoding="utf-8")
        self.assertIn("self._step = loaded.step", text)

    def test_a_missing_step_config_falls_back_rather_than_crashing(self) -> None:
        """An artifact written before the step block existed must still serve, and must not silently
        get the collapsed draw either."""
        out = serving_shares(None)
        self.assertEqual(out.labelled, 0.0)
        self.assertLess(out.predicted, 1.0)
        self.assertGreater(out.random, 0.0)

    def test_the_shipped_training_default_serves_at_half_and_half(self) -> None:
        """The concrete number a cell will actually use today."""
        out = serving_shares(SeedShares(labelled=0.5, predicted=0.25, random=0.25))
        self.assertAlmostEqual(out.predicted, 0.5)
        self.assertAlmostEqual(out.random, 0.5)


class GenerativeIsServedTests(unittest.TestCase):
    """⛔⛔ A `--generative` ARTIFACT WAS SERVED BY THE UNTRAINED K-SLOT HEAD.

    Both heads are built on a generative net, but only `self.generative` receives a gradient in a
    generative run, so `self.head` stays at its initialisation. `propose` returned it anyway. The
    shapes are right and the poses look like poses, which is the quietest possible way to be wrong.
    """

    @staticmethod
    def _net(generative: bool):
        import torch

        from src.robot.grasping.deep.net.generative_head import GenerativeHeadConfig
        from src.robot.grasping.deep.net.set_generator import (
            SetGenerator,
            SetGeneratorConfig,
        )

        torch.manual_seed(0)
        base = SetGeneratorConfig()
        return SetGenerator(type(base)(
            backbone=type(base.backbone)(width=32, depth=2, heads=2),
            head=type(base.head)(slots=4, width=32),
            generative=(GenerativeHeadConfig(width=64, depth=2, steps=4, gripper_width=8,
                                             time_width=16) if generative else None)))

    def _predict(self, net):
        import torch

        from src.robot.grasping.deep.net.serialized_backbone import SerializedConfig

        torch.manual_seed(1)
        points = torch.randn(1, 96, 3) * 0.08
        features = torch.randn(1, 96, SerializedConfig().in_features)
        with torch.no_grad():
            encoded = net.encode(points, features)
            return net.propose(encoded, torch.tensor([0, 0]), torch.tensor([3, 9]),
                               torch.zeros(2, 14), points)

    def test_a_generative_net_does_NOT_answer_from_the_K_slot_head(self) -> None:
        import torch

        net = self._net(generative=True)
        prediction = self._predict(net)
        features = net.seed_features(
            net.encode(torch.randn(1, 96, 3) * 0.08,
                       torch.randn(1, 96, net.backbone.config.in_features)),
            torch.tensor([0, 0]), torch.tensor([3, 9]))
        with torch.no_grad():
            slot_answer = net.head(features, torch.zeros(2, 14))
        self.assertFalse(torch.allclose(prediction.approach, slot_answer.approach),
                         "a generative net answered from its untrained K-slot head")

    def test_it_still_returns_the_shape_the_decoder_reads(self) -> None:
        prediction = self._predict(self._net(generative=True))
        self.assertEqual(prediction.approach.shape[:2], (2, 4))
        self.assertEqual(prediction.axis_mode, "vector")

    def test_a_NON_generative_net_is_unchanged(self) -> None:
        """The control, and the default-off guarantee."""
        import torch

        net = self._net(generative=False)
        self.assertIsNone(net.generative)
        prediction = self._predict(net)
        self.assertEqual(prediction.approach.shape[:2], (2, 4))
        self.assertEqual(prediction.axis_mode, "director")
        self.assertTrue(bool(torch.isfinite(prediction.approach).all()))


if __name__ == "__main__":
    unittest.main()
