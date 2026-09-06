"""The evaluation must judge a grasp against the part it holds, not against part 0.

⛔ THE DEFECT, and it made the whole part measurement impossible. `evaluate.py` judged every candidate
against `geometry.objects[i]` -- the PRIMARY solid -- and `obstacles_for` deliberately excludes the
object's own other parts, so a handle was neither target nor obstacle. The LABELLER does the opposite:
it loops every part and pairs each with its siblings (`labels.py:460-470`). Two different rules for the
same object.

⭐ THE CONTROL THAT SETTLED IT, and it is the shape of this whole test file: take the labeller's OWN
ground-truth labels and push them back through the evaluator's reference verdict. A correct grasp must
survive its own label. MEASURED before the fix, over 25 evaluated scenes:

    handle   0/36 valid      grip  25/49      body  76/122

Zero. Not "the stack is bad at handles" -- a grasp on a handle never enters the body, so
`check_jaw_grasp` returns LINE_MISSES_OBJECT whatever the generator does. And since COMPOSITE_KINDS
puts the body first for mug/jug/pan/bucket but the GRIP first for hammer, the per-role numbers were
reporting which slot the family builder used. In the other direction body precision was inflated: a
body grasp whose fingers pass through the object's own handle was never rejected.

The consequence that mattered: a later deep-versus-baseline comparison would have scored BOTH arms at
0.0 on handles, so the difference the run exists to detect was invisible by construction.
"""

from __future__ import annotations

import collections
import unittest

import numpy as np

from datagen.assets.composite import PartRole, build_composite
from datagen.eval.ladder import _verdict_over_parts
from datagen.grasps.labels import SceneGeometry, label_jaw_grasps
from datagen.grasps.shapes import Solid
from datagen.grasps.verdict import JawGrasp, check_jaw_grasp


def _geometry_for(asset) -> SceneGeometry:
    solids = [
        Solid(kind=part.primitive,
              half_extent_mm=np.asarray(part.extent_mm, dtype=float) / 2.0,
              rotation=np.eye(3),
              centre_mm=np.array([450.0, 0.0, asset.extent_mm[2] / 2.0])
              + np.asarray(part.offset_mm, dtype=float),
              instance_id=0, asset_id=asset.asset_id)
        for part in asset.parts
    ]
    return SceneGeometry(scene_id="t", family="sparse", objects={0: solids[0]}, walls=(),
                         parts={0: tuple(solids[1:])},
                         part_roles={(0, i): p.role for i, p in enumerate(asset.parts)})


def _labels_by_role(kind: str, seed: int):
    asset = build_composite(kind, np.random.default_rng(seed), index=0)
    geometry = _geometry_for(asset)
    labels, _ = label_jaw_grasps(geometry, 0)
    return geometry, labels


class TheLabellersOwnGraspsMustSurviveTheEvaluatorTest(unittest.TestCase):
    """A ground-truth grasp that the verdict rejects means the two disagree about the object."""

    def _survival(self, kind: str, seeds=range(4)):
        old, new = collections.Counter(), collections.Counter()
        for seed in seeds:
            geometry, labels = _labels_by_role(kind, seed)
            primary = geometry.objects[0]
            for label in labels:
                grasp = JawGrasp(np.asarray(label.position_mm, dtype=np.float64),
                                 np.asarray(label.approach, dtype=np.float64),
                                 np.asarray(label.closing_axis, dtype=np.float64),
                                 float(label.width_mm))
                role = label.part_role or "(none)"
                # THE OLD RULE: primary solid only, own parts absent from the obstacles.
                if check_jaw_grasp(grasp, primary, [], model=None).ok:
                    old[role] += 1
                verdict, _index, _role = _verdict_over_parts(grasp, geometry, 0, [], model=None)
                if verdict.ok:
                    new[role] += 1
        totals = collections.Counter(
            (label.part_role or "(none)")
            for seed in seeds for label in _labels_by_role(kind, seed)[1])
        return old, new, totals

    def test_a_HANDLE_grasp_is_creditable_now_and_was_not_before(self) -> None:
        """⚠ A BUCKET, NOT A MUG. MEASURED over 8 seeds each: a mug yields 206 body labels and ZERO
        handle labels at this density -- its handle is never jaw-labelled at all -- while a bucket
        yields handle labels and no body ones, and a jug yields both. Writing this test against a mug
        would have asserted a property of an empty set, which is the failure this file exists to catch
        in the measurement it is testing."""
        old, new, totals = self._survival("bucket")
        self.assertGreater(totals[PartRole.HANDLE], 0, "this mug has no handle labels to test with")
        self.assertEqual(old[PartRole.HANDLE], 0,
                         "the old rule was expected to credit ZERO handle grasps")
        self.assertGreater(new[PartRole.HANDLE], 0,
                           "the labeller's own handle grasps still score zero after the fix")

    def test_a_HAMMER_HEAD_grasp_is_creditable_now(self) -> None:
        """The other slot ordering: a hammer's PRIMARY solid is the grip, so the head was the loser."""
        old, new, totals = self._survival("hammer")
        self.assertGreater(totals[PartRole.HEAD], 0)
        self.assertGreater(new[PartRole.HEAD], old[PartRole.HEAD])


class TheRuleMustNotGetLooserTest(unittest.TestCase):
    def test_a_single_solid_object_is_judged_EXACTLY_as_before(self) -> None:
        """Most objects have one part. They must not move at all, or every old number shifts."""
        solid = Solid(kind="box", half_extent_mm=np.array([20.0, 20.0, 40.0]), rotation=np.eye(3),
                      centre_mm=np.array([450.0, 0.0, 40.0]), instance_id=0, asset_id="a")
        geometry = SceneGeometry(scene_id="t", family="sparse", objects={0: solid}, walls=())
        grasp = JawGrasp(np.array([450.0, 0.0, 40.0]), np.array([0.0, 0.0, -1.0]),
                         np.array([1.0, 0.0, 0.0]), 50.0)
        before = check_jaw_grasp(grasp, solid, [], model=None)
        after, index, role = _verdict_over_parts(grasp, geometry, 0, [], model=None)
        self.assertEqual(after.ok, before.ok)
        self.assertEqual(str(after.reason), str(before.reason))
        self.assertEqual((index, role), (0, ""))

    def test_a_grasp_through_the_objects_OWN_handle_is_rejected(self) -> None:
        """The other half of the labeller's rule: siblings are obstacles for the part being judged.

        Without it a body grasp whose fingers pass through the mug's own handle was accepted, which is
        the direction that INFLATED the body figures rather than deflating the handle ones.
        """
        geometry, labels = _labels_by_role("bucket", 1)
        handle = [label for label in labels if label.part_role == PartRole.HANDLE]
        self.assertTrue(handle, "no handle label to build the case from")
        # A grasp centred on the handle, closing along the axis that runs THROUGH the body.
        label = handle[0]
        through = JawGrasp(np.asarray(geometry.objects[0].centre_mm, dtype=np.float64),
                           np.asarray(label.approach, dtype=np.float64),
                           np.asarray(label.closing_axis, dtype=np.float64), 5.0)
        verdict, _index, _role = _verdict_over_parts(through, geometry, 0, [], model=None)
        self.assertFalse(verdict.ok, "a 5 mm jaw closing at the body centre should not be valid")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class BothVerdictPathsMustAgreeTest(unittest.TestCase):
    """⛔ THE FILE HAS TWO PLACES THAT JUDGE A GRASP, and for one commit today they disagreed.

    `_evaluate_view` grades a fresh run; the re-verdict path re-grades stored rows in place. Moving
    only the first to `_verdict_over_parts` meant a re-judged run reported the OLD numbers and a fresh
    run the new ones, out of the same file. That is the same "two rules for one object" defect the
    part fix exists to repair, reintroduced one function away from it.

    This pins it structurally rather than by eye: every `check_jaw_grasp` call in the module must be
    inside the shared helper.
    """

    def test_no_SUCTION_verdict_site_bypasses_the_shared_helper_either(self) -> None:
        """⛔ THE SAME DEFECT, THE OTHER MODALITY, and it was still there after the jaw was fixed.

        `label_suction_grasps` targeted `geometry.objects[i]` with `obstacles_for`, which omits the
        object's own parts, so a cup landing straight through a mug's handle was labelled valid and
        no suction label ever carried a role. MEASURED: all 19,924 suction labels read `part_role:
        ""`. Over 120 composite-bearing scenes the part-aware rule moves suction labels 2,105 to
        2,817, with a hammer's HEAD going from 20 to 162 because the head is not the primary solid,
        and a jug's body losing 56 landings that passed through its own handle.
        """
        import inspect
        import re

        from datagen.eval import ladder as evaluate

        source = inspect.getsource(evaluate)
        shared = inspect.getsource(evaluate._first_part_that_accepts)  # noqa: SLF001 - the rule IS it
        stray = re.findall(r"check_suction_grasp\(", source.replace(shared, ""))
        # One remains by construction: the closure that the shared helper is handed.
        self.assertLessEqual(
            len(stray), 1,
            f"{len(stray)} suction verdicts sit outside the shared part rule")

    def test_the_LABELLER_loops_parts_for_suction_as_it_does_for_the_jaw(self) -> None:
        import inspect

        from datagen.grasps import labels

        source = inspect.getsource(labels.label_suction_grasps)
        self.assertIn("parts_of", source, "the suction labeller still sees one solid")
        self.assertIn("part_role=role", source, "suction labels still carry no part role")

    def test_no_verdict_site_bypasses_the_shared_helper(self) -> None:
        import inspect
        import re

        from datagen.eval import ladder as evaluate

        source = inspect.getsource(evaluate)
        helper = inspect.getsource(evaluate._verdict_over_parts)  # noqa: SLF001 - the rule IS it
        outside = source.replace(helper, "")
        stray = re.findall(r"check_jaw_grasp\(", outside)
        self.assertFalse(
            stray,
            f"{len(stray)} call(s) to check_jaw_grasp sit outside _verdict_over_parts, so a composite "
            f"would be judged against its primary solid there")


class ReverdictMustLiftTheWholeFileTest(unittest.TestCase):
    """⛔ STAMPING ONLY THE RE-JUDGED ROWS WOULD REFUSE THE FILE IT JUST WROTE.

    `reverdict` re-judges candidate rows and passes `no_candidate`, `not_visible` and suction rows
    through untouched. `summarise` refuses a config key that pools two schema generations, because
    rows judged against the primary solid alone and rows judged against every part are two
    experiments. Stamp only the candidates and the two meet inside one file: the refusal fires AFTER
    the rows have been rewritten, which is the worst possible moment.

    A skeptic predicted this exact failure before it was written. The stamp therefore goes on every
    row the pass touches, before the `kind != "jaw"` continue.
    """

    def test_every_row_kind_is_stamped_not_only_the_candidates(self) -> None:
        import inspect

        from datagen.eval import ladder as evaluate

        source = inspect.getsource(evaluate.reverdict)
        stamp = source.index('row["row_schema"] = EVAL_ROW_SCHEMA')
        skip = source.index('if row.get("kind") != "jaw":')
        self.assertLess(stamp, skip,
                        "the stamp sits after the jaw filter, so suction rows keep the old generation")
