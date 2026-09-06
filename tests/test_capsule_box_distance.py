"""`capsule_box_distance_mm`: the error used to point the unsafe way, always.

⛔ THE DEFECT. The implementation sampled the capsule axis at seventeen points and took the minimum.
A minimum over samples can only OVERESTIMATE the true minimum, so whenever it was wrong it reported
the capsule as further from the box than it really was. That is structural, not bad luck, and it is
the one direction a safety guard must never be wrong in.

Its docstring claimed "sub-millimetre for typical arm-link capsule lengths < 500 mm". Measured
against a dense reference over 4,000 random cases in exactly that range: worst overestimate
**6.066 mm**, and one case reporting `+0.156 mm clear` where the truth was `-0.037 mm penetrating`.

⚠ It was absorbed in practice rather than dangerous, and the honest scope matters. `min_distance_mm`
ships at 10.0, and `robot.yaml` ships `fixtures: []`, so the path is not reached by a default cell at
all. It would have gone live on the first real cell to declare a table or a bin. The schema permits
`min_distance_mm` down to zero, below which the guard would have passed penetrating configurations.

⭑ These tests compare against a DENSE reference rather than against expected values, because a
hand-computed expected value proves only that the author and the code agree. The reference is the
same clamp-to-box distance evaluated at 200,001 points along the segment, which is the thing the old
code was approximating with seventeen.
"""

from __future__ import annotations

import time
import unittest

import numpy as np

from src.robot.safety._capsule import (
    AxisAlignedBox,
    Capsule,
    capsule_box_distance_mm,
)

#: Dense enough that its own residual error is far below anything asserted here.
_REFERENCE_SAMPLES = 200_000


def _reference(capsule: Capsule, box: AxisAlignedBox) -> float:
    """The signed distance, by brute force. Slow, obvious, and not the code under test."""
    t = np.linspace(0.0, 1.0, _REFERENCE_SAMPLES + 1)[:, None]
    points = capsule.p0 + t * (capsule.p1 - capsule.p0)
    clamped = np.clip(points - box.center_mm, -box.half_extents_mm, box.half_extents_mm)
    nearest = box.center_mm + clamped
    return float(np.linalg.norm(points - nearest, axis=1).min()) - capsule.radius_mm


def _random_case(rng: np.random.Generator) -> tuple[Capsule, AxisAlignedBox]:
    """A link-sized capsule and a fixture-sized box, in the range the old docstring named."""
    length = rng.uniform(100.0, 500.0)
    start = rng.uniform(-400.0, 400.0, 3)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    return (
        Capsule(p0=start, p1=start + length * direction, radius_mm=float(rng.uniform(30.0, 80.0))),
        AxisAlignedBox(center_mm=rng.uniform(-300.0, 300.0, 3),
                       half_extents_mm=rng.uniform(50.0, 300.0, 3)),
    )


def _adversarial_case(rng: np.random.Generator) -> tuple[Capsule, AxisAlignedBox]:
    """The worst case for a sampler: the true closest point falls between two of its samples.

    Built by construction rather than by luck. The segment grazes a box face at a parameter near
    1/32, which is exactly halfway between the old implementation's samples at 0 and 1/16.
    """
    box = AxisAlignedBox(center_mm=np.zeros(3), half_extents_mm=np.full(3, 100.0))
    length = rng.uniform(300.0, 500.0)
    t_star = 1.0 / 32.0 + rng.uniform(-0.004, 0.004)
    touch = np.array([100.0 + rng.uniform(0.0, 8.0), rng.uniform(-90.0, 90.0),
                      rng.uniform(-90.0, 90.0)])
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    return (
        Capsule(p0=touch - t_star * length * direction,
                p1=touch + (1.0 - t_star) * length * direction,
                radius_mm=float(rng.uniform(30.0, 80.0))),
        box,
    )


def _seventeen_sample_min(capsule: Capsule, box: AxisAlignedBox) -> float:
    """What the OLD implementation computed: the minimum over seventeen samples of the segment.

    Kept in the test file rather than in the module, because its only job is to define the band in
    which the old code was wrong, so that a test can aim at it deliberately instead of hoping a
    random draw lands there.
    """
    best = float("inf")
    for index in range(17):
        point = capsule.p0 + (index / 16.0) * (capsule.p1 - capsule.p0)
        clamped = np.clip(point - box.center_mm, -box.half_extents_mm, box.half_extents_mm)
        best = min(best, float(np.linalg.norm(point - (box.center_mm + clamped))))
    return best


class ExactnessTests(unittest.TestCase):
    """The distance has to match the reference, and may never come out HIGH."""

    def _sweep(self, make, count: int, tolerance_mm: float = 1e-6) -> None:
        rng = np.random.default_rng(0)
        worst_over = 0.0
        for _ in range(count):
            capsule, box = make(rng)
            got = capsule_box_distance_mm(capsule, box)
            reference = _reference(capsule, box)
            worst_over = max(worst_over, got - reference)
            # ⛔ THE ASYMMETRY IS THE POINT. Coming out LOW is conservative for a guard; coming out
            # HIGH is the failure that lets a penetrating configuration read as clear.
            self.assertLessEqual(
                got - reference, tolerance_mm,
                f"overestimated by {got - reference:.6f} mm: reported {got:.6f}, truth {reference:.6f}")
            self.assertAlmostEqual(got, reference, delta=tolerance_mm)
        self.assertLessEqual(worst_over, tolerance_mm)

    def test_random_link_sized_capsules(self) -> None:
        self._sweep(_random_case, 300)

    def test_the_case_a_sampler_cannot_see(self) -> None:
        """⭑ THIS IS THE REGRESSION TEST. The old seventeen-sample implementation fails it by
        millimetres, because the closest approach is placed between two of its samples on purpose."""
        self._sweep(_adversarial_case, 300)

    def test_it_never_reports_clear_when_the_truth_is_penetrating(self) -> None:
        """The failure that matters, and the radius is CONSTRUCTED rather than drawn.

        ⛔ THE FIRST VERSION OF THIS TEST WAS WORTHLESS AND LOOKED FINE. It drew 600 random cases and
        checked whether any of them reported clear while penetrating. Run against the old sampler it
        PASSED, because that failure needs the capsule radius to fall in the narrow band between the
        true distance and the overestimated one, and a random radius almost never lands there. The
        assertion that mattered most was the one least likely to fire.

        So the band is computed and the radius is placed inside it. For every geometry where a
        seventeen-sample scan overestimates at all, there exists a radius that makes the old code
        report clear on a penetrating capsule, and this builds exactly that capsule. Against the
        current implementation the band is empty to nine decimal places, which is the point.
        """
        rng = np.random.default_rng(7)
        constructed = 0
        for _ in range(400):
            capsule, box = _adversarial_case(rng)
            true_min = _reference(capsule, box) + capsule.radius_mm
            sampled_min = _seventeen_sample_min(capsule, box)
            if sampled_min - true_min < 1e-4:
                continue                       # nothing to exploit on this geometry
            # A radius inside the band: the truth penetrates, the approximation reads clear.
            radius = 0.5 * (true_min + sampled_min)
            probe = Capsule(p0=capsule.p0, p1=capsule.p1, radius_mm=float(radius))
            constructed += 1
            self.assertLess(_reference(probe, box), 0.0, "the construction is wrong, not the code")
            self.assertLess(capsule_box_distance_mm(probe, box), 0.0,
                            "reported clear while the truth was penetrating")
        self.assertGreater(constructed, 20,
                           "no exploitable geometry was constructed, so this test proved nothing")


class BehaviourTests(unittest.TestCase):
    def test_a_capsule_inside_the_box_is_negative(self) -> None:
        box = AxisAlignedBox(center_mm=np.zeros(3), half_extents_mm=np.full(3, 100.0))
        capsule = Capsule(p0=np.array([-10.0, 0.0, 0.0]), p1=np.array([10.0, 0.0, 0.0]),
                          radius_mm=25.0)
        self.assertLess(capsule_box_distance_mm(capsule, box), 0.0)

    def test_penetration_saturates_at_minus_radius(self) -> None:
        """⚠ A DELIBERATE IMPRECISION, pinned so nobody reads the value as a depth. Inside the box
        the clamp yields zero distance, so the result is exactly `-radius_mm` however deep the
        capsule is. For a guard that is safe, because it is already a rejection and it errs toward
        rejecting harder."""
        box = AxisAlignedBox(center_mm=np.zeros(3), half_extents_mm=np.full(3, 100.0))
        shallow = Capsule(p0=np.array([0.0, 0.0, 90.0]), p1=np.array([0.0, 0.0, 95.0]),
                          radius_mm=40.0)
        deep = Capsule(p0=np.array([0.0, 0.0, 0.0]), p1=np.array([0.0, 0.0, 5.0]), radius_mm=40.0)
        self.assertAlmostEqual(capsule_box_distance_mm(shallow, box), -40.0, places=6)
        self.assertAlmostEqual(capsule_box_distance_mm(deep, box), -40.0, places=6)

    def test_a_degenerate_capsule_is_a_sphere(self) -> None:
        box = AxisAlignedBox(center_mm=np.zeros(3), half_extents_mm=np.full(3, 100.0))
        point = np.array([200.0, 0.0, 0.0])
        capsule = Capsule(p0=point, p1=point, radius_mm=30.0)
        self.assertAlmostEqual(capsule_box_distance_mm(capsule, box), 100.0 - 30.0, places=6)

    def test_it_is_symmetric_in_the_segment_endpoints(self) -> None:
        """Swapping p0 and p1 describes the same capsule, so it must give the same answer. A search
        that is not symmetric has a bracketing bug."""
        rng = np.random.default_rng(3)
        for _ in range(100):
            capsule, box = _random_case(rng)
            flipped = Capsule(p0=capsule.p1, p1=capsule.p0, radius_mm=capsule.radius_mm)
            self.assertAlmostEqual(capsule_box_distance_mm(capsule, box),
                                   capsule_box_distance_mm(flipped, box), places=6)


class CostTests(unittest.TestCase):
    def test_the_exact_answer_is_also_the_cheap_one(self) -> None:
        """⭑ Being exact did not cost anything, it SAVED. The sixteen-sample approximation measured
        61.7 us per call; scalar golden-section measures about 12. The bound here is loose enough to
        survive a slower machine and tight enough to catch a return to numpy on three-vectors, which
        measured 448 us for the identical search."""
        rng = np.random.default_rng(1)
        cases = [_random_case(rng) for _ in range(400)]
        for capsule, box in cases[:50]:
            capsule_box_distance_mm(capsule, box)
        started = time.perf_counter()
        for capsule, box in cases:
            capsule_box_distance_mm(capsule, box)
        micros = (time.perf_counter() - started) / len(cases) * 1e6
        self.assertLess(micros, 150.0, f"{micros:.1f} us per call is far above the measured 12")


if __name__ == "__main__":
    unittest.main()
