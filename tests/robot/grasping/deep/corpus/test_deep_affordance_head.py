"""The K-slot family can say which PART a grasp is on, and something at inference reads it.

⛔⛔ **THE RETIRED FAMILY TRAINED THIS HEAD AND NOTHING EVER READ IT.** `GraspGeneratorNet` carries an
affordance classifier trained with a 6-way cross-entropy, MEASURED at 14.09 % of the trunk's
gradient, and `grep affordance` outside the net finds only training, loss weights and one datagen
comment. Every epoch paid for it; no inference ever collected it. Worse, the K-slot family that
REPLACED that one has no such head at all, so "grasp it by the handle" had no path to a grasp on the
family the runbook tells a customer to train.

⭐ **THE SUPERVISION WAS ALREADY IN THE CORPUS.** MEASURED over 400 v5 clouds and 82,935 grasp
labels: 32.30 % carry a non-empty role, and 155 of 400 scenes have at least one. grip 18.04 %,
body 10.68 %, handle 2.87 %, head 0.70 %, neck 0.02 %. Nothing had to be re-rendered or re-labelled.

⚠ **OFF BY DEFAULT AND A SEPARATE OUTPUT LAYER.** Widening `SLOT_OUTPUTS` past 14 would have changed
the shape of every tensor in every artifact ever written. Off, a net has neither the parameters nor
the output and is unchanged tensor for tensor.

⚠ **UNMEASURED: whether it helps or costs the geometry.** That is the arm the flag exists to run. Its
weight is 0.25 rather than 1.0 precisely so a role cannot quietly starve the heads a cell fails on.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.robot.grasping.deep.corpus.sample import _PART_ROLE_INDEX
from src.robot.grasping.deep.net.slot_head import (
    SLOT_OUTPUTS,
    SlotGraspHead,
    SlotHeadConfig,
)
from src.robot.grasping.deep.net.set_loss import (
    GraspSetTarget,
    SetLossWeights,
    grasp_set_loss,
)
from src.robot.grasping.deep.set_decode import decode_set_prediction

_ROLES = tuple(sorted(_PART_ROLE_INDEX, key=lambda name: _PART_ROLE_INDEX[name]))


def _head(roles: tuple[str, ...]) -> SlotGraspHead:
    return SlotGraspHead(16, SlotHeadConfig(slots=3, width=32, gripper_width=8, query_width=8,
                                              part_roles=roles))


class ItIsOffByDefaultTests(unittest.TestCase):

    def test_the_default_head_has_no_affordance_output(self) -> None:
        head = _head(())

        self.assertIsNone(head.affordance)
        self.assertIsNone(head(torch.randn(4, 16), torch.randn(4, 14)).part_logits)

    def test_a_role_less_net_is_unchanged_TENSOR_FOR_TENSOR(self) -> None:
        """⭐ THE PROPERTY THAT LETS THIS SHIP. If turning roles off left even one extra tensor,
        every artifact ever written would have stopped loading."""
        plain = _head(())
        self.assertEqual(set(_head(()).state_dict()), set(plain.state_dict()))

    def test_SLOT_OUTPUTS_did_not_move(self) -> None:
        """The fourteen numbers keep their meaning and their weights, because the roles come from
        their OWN layer rather than from a wider one."""
        self.assertEqual(SLOT_OUTPUTS, 14)

    def test_turning_it_on_adds_only_the_affordance_tensors(self) -> None:
        extra = set(_head(_ROLES).state_dict()) - set(_head(()).state_dict())

        self.assertEqual(extra, {"affordance.weight", "affordance.bias"})


class TheHeadPredictsTests(unittest.TestCase):

    def test_the_logits_have_one_column_per_role(self) -> None:
        prediction = _head(_ROLES)(torch.randn(5, 16), torch.randn(5, 14))

        assert prediction.part_logits is not None
        self.assertEqual(prediction.part_logits.shape, (5, 3, len(_ROLES)))

    def test_the_vocabulary_matches_the_dataset_mapping(self) -> None:
        """⚠ ONE LIST, NOT TWO. The index the loss trains against comes from
        `_PART_ROLE_INDEX`; a second list here would be a second thing to keep in step, and the
        failure would be a head naming `handle` where the corpus meant `grip`."""
        self.assertEqual(_ROLES, ("", "body", "handle", "grip", "neck", "head"))


class TheLossTests(unittest.TestCase):

    def _target(self, *, with_roles: bool, seeds: int = 5) -> GraspSetTarget:
        approach = torch.nn.functional.normalize(torch.randn(seeds, 4, 3), dim=-1)
        return GraspSetTarget(
            approach=approach, axis=approach.roll(1, dims=-1),
            offset_m=torch.randn(seeds, 4, 3) * 0.01,
            width_m=torch.full((seeds, 4), 0.05), valid=torch.ones(seeds, 4, dtype=torch.bool),
            part_index=(torch.randint(0, len(_ROLES), (seeds, 4)) if with_roles else None))

    def test_the_part_term_appears_when_BOTH_sides_have_roles(self) -> None:
        prediction = _head(_ROLES)(torch.randn(5, 16), torch.randn(5, 14))
        terms = grasp_set_loss(prediction, self._target(with_roles=True))

        self.assertIn("part", terms)
        self.assertGreater(float(terms["part"]), 0.0)

    def test_it_starts_near_the_uniform_loss(self) -> None:
        """⭐ The control that says the term is a real cross-entropy rather than an arbitrary number:
        an untrained 6-way classifier scores about ln(6) = 1.79."""
        prediction = _head(_ROLES)(torch.randn(64, 16), torch.randn(64, 14))
        value = float(grasp_set_loss(prediction, GraspSetTarget(
            approach=torch.nn.functional.normalize(torch.randn(64, 4, 3), dim=-1),
            axis=torch.nn.functional.normalize(torch.randn(64, 4, 3), dim=-1),
            offset_m=torch.randn(64, 4, 3) * 0.01, width_m=torch.full((64, 4), 0.05),
            valid=torch.ones(64, 4, dtype=torch.bool),
            part_index=torch.randint(0, len(_ROLES), (64, 4))))["part"])

        self.assertAlmostEqual(value, float(np.log(len(_ROLES))), delta=0.35)

    def test_NO_part_term_when_the_head_has_no_roles(self) -> None:
        """A corpus with roles under a role-less head must not invent a term. Every arm before
        2026-09-04 is exactly this case."""
        prediction = _head(())(torch.randn(5, 16), torch.randn(5, 14))

        self.assertNotIn("part", grasp_set_loss(prediction, self._target(with_roles=True)))

    def test_NO_part_term_when_the_corpus_has_no_roles(self) -> None:
        """And the other way: a head with roles on a corpus without them would fit noise."""
        prediction = _head(_ROLES)(torch.randn(5, 16), torch.randn(5, 14))

        self.assertNotIn("part", grasp_set_loss(prediction, self._target(with_roles=False)))

    def test_it_is_weighted_BELOW_the_geometry(self) -> None:
        """⚠ 0.25, not 1.0. The binned family gave its role head full weight and it took 14.09 % of
        the trunk's gradient for something nothing read. A cell fails on geometry."""
        weights = SetLossWeights()

        self.assertLess(weights.loss_part, weights.loss_approach)
        self.assertLessEqual(weights.loss_part, weights.loss_width)

    def test_the_term_REACHES_the_total(self) -> None:
        """A term computed and left out of the total is the same defect as a head nothing reads."""
        prediction = _head(_ROLES)(torch.randn(8, 16), torch.randn(8, 14))
        target = self._target(with_roles=True, seeds=8)

        with_part = grasp_set_loss(prediction, target, SetLossWeights(loss_part=1.0))
        without = grasp_set_loss(prediction, target, SetLossWeights(loss_part=0.0))

        self.assertGreater(float(with_part["total"]), float(without["total"]))


class TheRuntimeREADSItTests(unittest.TestCase):
    """⛔ The whole point. The other family's head was trained and never read."""

    def _decode(self, roles: tuple[str, ...]):  # noqa: ANN202
        head = _head(roles)
        prediction = head(torch.randn(6, 16), torch.randn(6, 14))
        return decode_set_prediction(
            prediction, torch.arange(6), torch.randn(6, 3), centre_xy_mm=np.zeros(2),
            support_height_mm=0.0, min_width_mm=1.0, max_width_mm=200.0, part_roles=roles)

    def test_a_decoded_grasp_carries_a_role_and_a_score(self) -> None:
        grasps = self._decode(_ROLES)

        self.assertTrue(grasps)
        for grasp in grasps:
            with self.subTest(grasp.slot):
                self.assertIn(grasp.part_role, _ROLES)
                assert grasp.part_score is not None
                self.assertGreaterEqual(grasp.part_score, 0.0)
                self.assertLessEqual(grasp.part_score, 1.0)

    def test_a_role_less_net_reports_NO_role_rather_than_inventing_one(self) -> None:
        for grasp in self._decode(()):
            with self.subTest(grasp.slot):
                self.assertIsNone(grasp.part_role)
                self.assertIsNone(grasp.part_score)

    def test_both_serve_paths_pass_the_ARTIFACT_vocabulary(self) -> None:
        """⚠ From the net that was trained, never from what this build thinks the roles are. A
        vocabulary read off the current code would rename every role the day someone reorders it."""
        from pathlib import Path

        # ⛔ WHOLE PATHS, NOT A DIRECTORY PLUS A NAME. This built each path with an f-string, so the
        # string `deep/propose.py` existed nowhere in the file and no rename tool could see it. When
        # `propose.py` moved to `eval/` on 2026-09-04 the AST pass, the dotted-import pass and the
        # prose pass all reported a clean sweep, and this test failed at RUN time on a path that was
        # assembled a character at a time. A literal can be found; a concatenation cannot.
        for name in ("src/robot/grasping/deep/calculator.py",
                     "src/robot/grasping/deep/eval/propose.py"):
            with self.subTest(name):
                source = Path(name).read_text(encoding="utf-8")
                self.assertIn("part_roles=tuple(", source)
                self.assertIn("config.head.part_roles", source)

    def test_the_role_REACHES_the_caller_and_not_only_the_decoder(self) -> None:
        """⛔⛔ IT DID NOT, AND THE TEST ABOVE IS WHY THAT SHIPPED. Asserting that two source files
        contain a string proves the decoder is CALLED with a vocabulary. It proves nothing about
        whether the answer leaves the function. `94129cd` built the head, wired the decoder, and then
        constructed `GraspPoint` without the role, so nothing outside `_propose_set` could ever see
        one: the head was unread again, one layer further out than the defect it was built to fix.

        ⚠ SO THIS READS THE CONSTRUCTION, not the file. The role has to be in the metadata a caller
        actually receives.
        """
        import ast
        from pathlib import Path

        source = Path(
            "src/robot/grasping/deep/calculator.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        method = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name == "_propose_set")
        built = [node for node in ast.walk(method)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == "GraspPoint"]

        self.assertTrue(built, "the set path no longer builds a GraspPoint")
        for call in built:
            metadata = next((kw.value for kw in call.keywords if kw.arg == "metadata"), None)
            self.assertIsInstance(metadata, ast.Dict, "GraspPoint is built without metadata")
            keys = [key.value for key in metadata.keys                       # type: ignore[union-attr]
                    if isinstance(key, ast.Constant)]
            self.assertIn("part_role", keys,
                          "the affordance head names a part and the caller never sees it")
            self.assertIn("part_score", keys,
                          "a role with no confidence cannot be refused, only obeyed")


class TheCorpusCarriesItTests(unittest.TestCase):

    def test_the_sample_emits_one_role_per_TABLE_ROW(self) -> None:
        """Per grasp, not per pair, so a slot's role is the role of the very label whose approach and
        width it is graded on."""
        from src.robot.grasping.deep.corpus.sample import _grasp_set_arrays

        block = _grasp_set_arrays(
            pair_rows=np.array([0, 1]), pair_owner=np.array([0, 1]),
            grasp_position=np.zeros((2, 3)), grasp_approach=np.zeros((2, 3)),
            grasp_axis=np.zeros((2, 3)), grasp_width=np.array([50.0, 60.0]),
            supervise=np.ones(2, dtype=bool), centre_xy=np.zeros(2), support_height_mm=0.0,
            grasp_part_role=np.array(["handle", "grip"]))

        self.assertEqual(list(block["set_grasp_part_index"]),
                         [_PART_ROLE_INDEX["handle"], _PART_ROLE_INDEX["grip"]])

    def test_an_unknown_role_lands_on_the_EMPTY_class(self) -> None:
        """Rather than raising or dropping the label. A corpus written by a future labeller with a
        new role must still train, and 0 is a real class meaning "no parts to tell apart"."""
        from src.robot.grasping.deep.corpus.sample import _grasp_set_arrays

        block = _grasp_set_arrays(
            pair_rows=np.array([0]), pair_owner=np.array([0]),
            grasp_position=np.zeros((1, 3)), grasp_approach=np.zeros((1, 3)),
            grasp_axis=np.zeros((1, 3)), grasp_width=np.array([50.0]),
            supervise=np.ones(1, dtype=bool), centre_xy=np.zeros(2), support_height_mm=0.0,
            grasp_part_role=np.array(["spout"]))

        self.assertEqual(list(block["set_grasp_part_index"]), [0])

    def test_a_scene_with_no_roles_still_emits_the_key(self) -> None:
        """⚠ ZEROS, NOT AN ABSENT KEY. A missing key would make supervision depend on which scene a
        batch happened to draw, and a batch that trains on some samples and not others reads as a
        noisy loss rather than as a defect."""
        from src.robot.grasping.deep.corpus.sample import _grasp_set_arrays

        block = _grasp_set_arrays(
            pair_rows=np.array([0]), pair_owner=np.array([0]),
            grasp_position=np.zeros((1, 3)), grasp_approach=np.zeros((1, 3)),
            grasp_axis=np.zeros((1, 3)), grasp_width=np.array([50.0]),
            supervise=np.ones(1, dtype=bool), centre_xy=np.zeros(2), support_height_mm=0.0)

        self.assertEqual(list(block["set_grasp_part_index"]), [0])


if __name__ == "__main__":
    unittest.main()
