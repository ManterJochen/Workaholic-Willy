"""The mask-completion policy: one implementation, and the measurement that made it a lever.

WHY THIS EXISTS. Both perception sources -- the real-camera one and the Isaac one -- carried the same
twenty lines, copied verbatim: when a SAM2 mask covers under 80 % of the detector's box area, throw
the mask away and use the box. Two copies of a transform that decides the grasp's CLOSING AXIS is one
copy too many, so it moved to `robot/perception/mask_completion.py` and both sources delegate.

WHAT THE MEASUREMENT SAID (`python -m datagen camera-probe`, 809 views of v1_proof):

    fires on predicted (SAM2) masks     78.7 %      median mask/box area 0.696
    fires on GROUND-TRUTH masks         78.8 %      median mask/box area 0.691   <-- the control

Identical on complete masks, so the rule is a NON-RECTANGULARITY test, not an incompleteness test.
And SAM2 genuinely truncates in only 10.3 % of detections -- it fires 7.6x more often than the
condition it was written for.

⚠ AND WHAT IT DID NOT SAY, corrected here because the first reading overstated it. The rule rotates
the closing axis by a median 16.9 deg when it fires, which looked damning. Scored against the ANALYTIC
LABELS -- which state the true closing axis -- it is close to a coin flip:

    policy                n     median err   p75      outside a mu=0.5 cone
    mask (none)          933      13.7 deg   39.6         31.6 %
    axis_aligned_box     933      16.6 deg   41.9         37.5 %      <-- shipped
    oriented_box         933      13.7 deg   40.0         31.1 %

The mask's own axis is already 13.7 deg off the truth, so the rotation lands about as often closer as
further. The consistent cost is ~6 pp more contacts outside the friction cone. Real, and second-order:
the first-order finding in that table is that a 2-D silhouette is a WEAK source for a 3-D closing axis
at all, which is why SFE and fused multi-view geometry exist.

AND THEN THE OUTCOME WAS MEASURED, which is what moved the default. `datagen eval-grasps
--mask-completion` runs the real calculator over all 270 reference scenes; PAIRED over the 1365
object-views evaluated under every policy:

    policy               top-1     coverage
    none                 26.7 %     30.9 %
    axis_aligned_box     19.2 %     25.6 %     -7.5 pp     129 grasps LOST, 26 WON
    oriented_box         26.2 %     30.0 %     -0.5 pp

So the default is now `NONE`. `ORIENTED_BOX` is indistinguishable from it -- 0.5 pp over 1365 is about
seven object-views -- and `NONE` never INVENTS geometry, which filling an L-shaped silhouette to any
box does.

⚠ This changes the Isaac cell too, and those gates are re-run on-box before it is trusted there.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.perception.mask_completion import (
    DEFAULT_MASK_COMPLETION,
    DEFAULT_MIN_FILL_RATIO,
    MaskCompletion,
    complete_mask,
)


class _Det:
    def __init__(self, box) -> None:
        self.box = box


def _principal_axis_deg(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    centred = np.stack([xs - xs.mean(), ys - ys.mean()])
    values, vectors = np.linalg.eigh(centred @ centred.T / max(1, centred.shape[1] - 1))
    axis = vectors[:, int(np.argmax(values))]
    return float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0)


def _diagonal_bar(size: int = 60, half: int = 3) -> np.ndarray:
    """A bar at 45 deg -- the shape the axis-aligned rule cannot represent."""
    mask = np.zeros((size, size), dtype=bool)
    for i in range(8, size - 8):
        mask[max(0, i - half):i + half, max(0, i - half):i + half] = True
    return mask


class TheShippedDefaultTests(unittest.TestCase):
    """The default is ``NONE``, and it is pinned HERE so a flip cannot happen by accident.

    It moved from ``AXIS_ALIGNED_BOX`` on 2026-08-20, by the outcome measurement in the module
    docstring: -7.5 pp of top-1, 129 grasps lost against 26 won over 1365 object-views.
    """

    def test_the_shipped_default_is_none(self) -> None:
        self.assertIs(DEFAULT_MASK_COMPLETION, MaskCompletion.NONE)

    def test_both_perception_sources_take_that_default(self) -> None:
        """One default, two sources -- the drift this module exists to prevent."""
        import inspect

        from src.robot.perception import RealSenseVisionPerceptionSource

        real = inspect.signature(RealSenseVisionPerceptionSource.__init__)
        self.assertIs(real.parameters["mask_completion"].default, DEFAULT_MASK_COMPLETION)

        # The sim source imports Isaac lazily, so its signature is readable without a GPU.
        from src.willy_sim.perception.vision import MultiObjectVisionPerceptionSource

        sim = inspect.signature(MultiObjectVisionPerceptionSource.__init__)
        self.assertIs(sim.parameters["mask_completion"].default, DEFAULT_MASK_COMPLETION)

    def test_at_the_default_a_truncated_mask_is_left_ALONE(self) -> None:
        """The behaviour that changed. The old default replaced this with the detector box."""
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:8, 1:6] = True
        self.assertIs(complete_mask(mask, _Det([1.0, 2.0, 17.0, 8.0])), mask)

    def test_a_near_complete_mask_is_returned_UNTOUCHED(self) -> None:
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:8, 1:17] = True
        out = complete_mask(mask, _Det([1.0, 2.0, 17.0, 8.0]))
        # `is`, not array-equal: callers assert identity to mean untouched, and a copy here is waste.
        self.assertIs(out, mask)

    def test_the_old_default_still_works_when_asked_for_by_name(self) -> None:
        """Removed as a DEFAULT, not as a capability -- a cell that wants it can still set it."""
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:8, 1:6] = True
        out = complete_mask(mask, _Det([1.0, 2.0, 17.0, 8.0]),
                            policy=MaskCompletion.AXIS_ALIGNED_BOX)
        self.assertEqual(int(out.sum()), 6 * 16)

    def test_it_refuses_rather_than_guesses(self) -> None:
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:8, 1:6] = True
        for label, det in (
            ("no detection", None),
            ("no box attribute", object()),
            ("degenerate box", _Det([5.0, 5.0, 5.0, 5.0])),
            ("box off-image", _Det([-9.0, -9.0, -1.0, -1.0])),
        ):
            with self.subTest(case=label):
                self.assertIs(complete_mask(mask, det,
                                            policy=MaskCompletion.AXIS_ALIGNED_BOX), mask)


class PolicyTests(unittest.TestCase):
    def test_none_never_replaces(self) -> None:
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:8, 1:6] = True
        out = complete_mask(mask, _Det([1.0, 2.0, 17.0, 8.0]), policy=MaskCompletion.NONE)
        self.assertIs(out, mask)

    def test_oriented_box_preserves_the_axis_that_the_shipped_rule_destroys(self) -> None:
        """The mechanism, in one shape. A 45-degree bar is what the measurement is made of."""
        mask = _diagonal_bar()
        truth = _principal_axis_deg(mask)
        self.assertAlmostEqual(truth, 45.0, delta=6.0, msg="the fixture is not diagonal")

        ys, xs = np.nonzero(mask)
        det = _Det([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])

        filled = complete_mask(mask, det, policy=MaskCompletion.AXIS_ALIGNED_BOX)
        self.assertIsNot(filled, mask, "the shipped rule must fire on a diagonal bar")
        # An axis-aligned square has no principal axis to speak of; either way it is not 45 deg.
        shipped_error = min(abs(_principal_axis_deg(filled) - truth),
                            180.0 - abs(_principal_axis_deg(filled) - truth))
        self.assertGreater(shipped_error, 20.0)

        oriented = complete_mask(mask, det, policy=MaskCompletion.ORIENTED_BOX)
        oriented_error = min(abs(_principal_axis_deg(oriented) - truth),
                             180.0 - abs(_principal_axis_deg(oriented) - truth))
        self.assertLess(oriented_error, 6.0, "the oriented box must keep the object's orientation")

    def test_oriented_box_gives_up_the_extent_rescue_and_says_so(self) -> None:
        """Pinned because it is the honest cost of the alternative, not an oversight.

        The oriented box is derived FROM the mask, so a mask missing one end of an object yields a box
        missing that end too. It removes the harm; it does not keep the cure.
        """
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:8, 1:6] = True                                   # truncated
        det = _Det([1.0, 2.0, 17.0, 8.0])                       # the detector saw the whole object
        self.assertGreater(
            int(complete_mask(mask, det, policy=MaskCompletion.AXIS_ALIGNED_BOX).sum()),
            int(mask.sum()))
        self.assertEqual(
            int(complete_mask(mask, det, policy=MaskCompletion.ORIENTED_BOX).sum()),
            int(mask.sum()),
        )


class TheThresholdSelectsForShapeTests(unittest.TestCase):
    """The control, in miniature: the rule fires on complete masks that simply are not rectangles."""

    def test_a_complete_disc_is_already_under_the_threshold(self) -> None:
        # A circle fills pi/4 = 0.785 of its bounding box. The shipped threshold is 0.8.
        self.assertLess(np.pi / 4.0, DEFAULT_MIN_FILL_RATIO)

        radius, size = 20, 50
        yy, xx = np.mgrid[:size, :size]
        disc = ((yy - size // 2) ** 2 + (xx - size // 2) ** 2) <= radius ** 2
        ys, xs = np.nonzero(disc)
        det = _Det([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])

        out = complete_mask(disc, det, policy=MaskCompletion.AXIS_ALIGNED_BOX)
        self.assertIsNot(out, disc, "a COMPLETE disc is replaced by its bounding square")
        self.assertGreater(int(out.sum()), int(disc.sum()))

        # And the alternative leaves it alone -- for a measured reason, not by luck. `cv.minAreaRect`
        # fits the convex hull of the pixels, and on a discretised disc it finds a tighter square at
        # 26.6 deg than the axis-aligned extent: 39.35^2 = 1548.8 against 41x41 = 1681. So the ratio
        # is 1257/1548.8 = 0.812, ABOVE the threshold, where the axis-aligned ratio is 0.748, below it.
        # The oriented box does not merely preserve orientation; it is a tighter fit, which is what
        # moves this shape from "replace it" to "keep it".
        self.assertIs(complete_mask(disc, det, policy=MaskCompletion.ORIENTED_BOX), disc)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
