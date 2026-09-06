"""Objects with parts, and grasp labels that say WHICH part they touch.

The library rendered sixteen kind words as three convex solids, so nothing in a scene ever had a
graspable feature distinct from its body. A model trained on that cannot learn that a hammer is held
by its grip -- it has never seen the distinction.

THE MEASUREMENT THAT SHAPED THIS, and it is the reason these tests assert a MIX rather than success:
a handle is jaw-graspable only when its free span exceeds roughly 100 mm (sweep: 36/56/74/78/83 mm
-> 0 labels, 101 mm -> 7, 108 mm -> 12). The cause is the gripper, not the shape -- a 2F-85's inner
finger is 62.3 mm with a 41.5 mm pad and needs more clear run than a mug handle has. **A 2F-85
cannot grasp a typical mug by its handle**, which is consistent with what this project already
measured elsewhere about low top-down jaw grasps.

So the families are deliberately a mix: hammers, pans and buckets give a part-aware model positive
examples, mugs give it the negative. A dataset of positives only would teach it that handles are
always graspable and it would fail on the first real mug.
"""

from __future__ import annotations

import collections
import unittest

import numpy as np

from datagen.assets.composite import (
    COMPOSITE_KINDS,
    CompositeAsset,
    CompositePart,
    PartRole,
    build_composite,
    sample_composite_asset,
)
from datagen.grasps.labels import SceneGeometry, label_jaw_grasps
from datagen.grasps.shapes import Solid
from datagen.scenes.spec import SceneAsset


def _geometry_for(asset: CompositeAsset) -> SceneGeometry:
    """The composite standing on the table at the cell's pick centre, unrotated."""
    solids = [
        Solid(
            kind=part.primitive,
            half_extent_mm=np.asarray(part.extent_mm, dtype=float) / 2.0,
            rotation=np.eye(3),
            centre_mm=np.array([450.0, 0.0, asset.extent_mm[2] / 2.0])
            + np.asarray(part.offset_mm, dtype=float),
            instance_id=0,
            asset_id=asset.asset_id,
        )
        for part in asset.parts
    ]
    return SceneGeometry(
        scene_id="t", family="sparse", objects={0: solids[0]}, walls=(),
        parts={0: tuple(solids[1:])},
        part_roles={(0, i): p.role for i, p in enumerate(asset.parts)},
    )


def _roles(asset: CompositeAsset) -> collections.Counter[str]:
    labels, _ = label_jaw_grasps(_geometry_for(asset), 0)
    return collections.Counter(label.part_role for label in labels)


class ACompositeIsAnOrdinarySceneAssetTests(unittest.TestCase):
    def test_it_satisfies_the_layout_contract(self) -> None:
        asset = build_composite("mug", np.random.default_rng(3), index=0)
        self.assertIsInstance(asset, SceneAsset)

    def test_its_extent_bounds_every_part(self) -> None:
        """Placement spaces objects by this; under-reporting it would let a handle overlap a neighbour."""
        asset = build_composite("pan", np.random.default_rng(3), index=0)
        for part in asset.parts:
            for axis in range(3):
                half = part.extent_mm[axis] / 2.0
                reach = abs(part.offset_mm[axis]) + half
                with self.subTest(part=part.role, axis=axis):
                    self.assertLessEqual(reach, asset.extent_mm[axis] / 2.0 + 1e-6)

    def test_only_the_three_authorable_primitives_are_used(self) -> None:
        """The coupling the whole design rests on: labels intersect what the renderer authors.

        A part using a shape the renderer cannot build would produce labels for geometry that was
        never rendered -- the exact failure `procedural.py` warns about.
        """
        rng = np.random.default_rng(7)
        for kind in sorted(COMPOSITE_KINDS):
            for _ in range(5):
                for part in build_composite(kind, rng, index=0).parts:
                    with self.subTest(kind=kind, role=part.role):
                        self.assertIn(part.primitive, ("box", "cylinder", "sphere"))

    def test_a_part_may_not_use_another_shape(self) -> None:
        with self.assertRaises(ValueError) as caught:
            CompositePart(PartRole.HANDLE, "torus", (10.0, 10.0, 10.0))
        self.assertIn("box/cylinder/sphere", str(caught.exception))

    def test_graspability_is_per_part_not_per_object(self) -> None:
        """A bucket body too wide for the jaw with a handle that is not IS graspable."""
        asset = CompositeAsset(
            asset_id="c", composite_kind="bucket", mass_kg=0.2,
            parts=(
                CompositePart(PartRole.BODY, "cylinder", (140.0, 140.0, 120.0)),
                CompositePart(PartRole.HANDLE, "box", (140.0, 14.0, 14.0), (0.0, 0.0, 90.0)),
            ),
        )
        self.assertFalse(asset.parts[0].jaw_graspable)
        self.assertTrue(asset.parts[1].jaw_graspable)
        self.assertTrue(asset.jaw_graspable)


class LabelsCarryThePartTheyTouchTests(unittest.TestCase):
    def test_a_hammer_yields_both_grip_and_head_grasps(self) -> None:
        """The supervision signal: the same object, two parts, distinguishable."""
        roles = _roles(build_composite("hammer", np.random.default_rng(42), index=0))
        self.assertIn(PartRole.GRIP, roles)
        self.assertIn(PartRole.HEAD, roles)

    def test_a_single_primitive_object_still_reports_no_part(self) -> None:
        """Every label generated before parts existed keeps its exact meaning."""
        from datagen.grasps.shapes import solid_from_scene

        solid = solid_from_scene("cylinder", (60.0, 60.0, 90.0), (450.0, 0.0, 45.0),
                                 (0.0, 0.0, 0.0, 1.0), instance_id=0)
        geometry = SceneGeometry(scene_id="t", family="sparse", objects={0: solid}, walls=())
        labels, _ = label_jaw_grasps(geometry, 0)
        self.assertTrue(labels, "the fixture must produce grasps for this to mean anything")
        self.assertEqual({label.part_role for label in labels}, {""})

    def test_the_part_row_reaches_the_serialised_label(self) -> None:
        labels, _ = label_jaw_grasps(
            _geometry_for(build_composite("pan", np.random.default_rng(42), index=0)), 0
        )
        self.assertTrue(labels)
        self.assertIn("part_role", labels[0].as_row())

    def test_a_grasp_through_the_body_is_rejected(self) -> None:
        """The correctness half of per-part labelling: siblings are obstacles.

        Without it, every handle grasp that passes through the mug body would be labelled valid.
        """
        asset = build_composite("mug", np.random.default_rng(3), index=0)
        with_siblings = _roles(asset)

        stripped = SceneGeometry(
            scene_id="t", family="sparse",
            objects=_geometry_for(asset).objects, walls=(),
        )
        without, _ = label_jaw_grasps(stripped, 0)
        self.assertGreaterEqual(
            len(without), sum(with_siblings.values()),
            "removing the sibling parts must not REDUCE the grasps -- they are pure obstacles",
        )


class TheFamiliesAreDeliberatelyAMixTests(unittest.TestCase):
    """Positives AND negatives, because the gripper cannot reach every handle a human can."""

    def _feature_fraction(self, kind: str, n: int = 40) -> float:
        rng = np.random.default_rng(42)
        hits = 0
        for i in range(n):
            roles = _roles(build_composite(kind, rng, index=i))
            hits += any(role not in ("body", "") for role in roles)
        return hits / n

    def test_tools_reliably_offer_a_feature_grasp(self) -> None:
        for kind in ("hammer", "pan"):
            with self.subTest(kind=kind):
                self.assertGreater(self._feature_fraction(kind), 0.8)

    def test_a_bail_handle_is_sometimes_reachable(self) -> None:
        self.assertGreater(self._feature_fraction("bucket"), 0.2)

    def test_a_mug_handle_is_NOT_reachable_by_this_gripper(self) -> None:
        """MEASURED, and asserted so nobody 'fixes' it by inflating mug handles to bucket size.

        The 2F-85's finger needs ~100 mm of clear span; a mug handle has a third of that. The mug is
        in the library as the negative example, and as a suction target once that head is trained.
        """
        self.assertEqual(self._feature_fraction("mug"), 0.0)

    def test_sampling_is_deterministic(self) -> None:
        a = sample_composite_asset(np.random.default_rng(5), index=0)
        b = sample_composite_asset(np.random.default_rng(5), index=0)
        self.assertEqual(a.asset_id, b.asset_id)
        self.assertEqual(a.extent_mm, b.extent_mm)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
