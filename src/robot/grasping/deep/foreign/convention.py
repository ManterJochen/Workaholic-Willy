"""Which axis is the approach, which is the closing axis, and what the translation refers to.

A foreign dataset does not state this. A 6-DoF grasp is normally published as a 4x4 homogeneous
matrix and a scalar width. The matrix fixes a frame, but nothing in the file says which column of
the rotation the gripper travels along, which column the jaws close along, whether the translation
sits at the grasp centre or at the gripper base, or whether the numbers are metres or millimetres.
Different publications make different choices, and the choice is often recorded only in a loader
script or a figure caption.

Importing under a wrong guess does not fail loudly. It produces a corpus that is geometrically
self-consistent and completely wrong, and a network trains on it to a confident wrong answer. A
correction applied in the wrong frame measures worse than applying no correction at all, and none of
these failures announces itself.

So the convention is measured, not assumed. A parallel-jaw grasp makes a falsifiable geometric claim
about the cloud it was labelled on, and that claim needs no knowledge of the publisher's intentions:

    the two jaw contacts lie on the surface        the jaws close until they touch the object
    material lies between the two jaws             otherwise there is nothing to hold
    the corridor the gripper arrives through is    otherwise the gripper could not have got there
      emptier than the space beyond the grasp

Under the right convention all three hold. Under a wrong one the contacts fly off into free space,
because the stored width is the object extent along the true closing axis and not along whichever
column was picked instead. That is a large, cheap, unambiguous signal.

This module enumerates the plausible conventions, scores each against those three claims, and
returns the winner only if it beats the runner-up by a margin. An ambiguous result is a refusal, not
a coin flip: not knowing is a state an importer can report, and a wrong guess is not.

What this cannot do: it cannot detect a convention that is geometrically indistinguishable from the
right one on the scenes it was given, and it cannot judge whether the labels are any good. A dataset
of physically valid poses on the wrong objects passes. It settles the frame, and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np

__all__ = ["Convention", "ConventionScore", "ConventionProof", "prove_convention"]

#: Finger length on the modelled grippers, so the distance the palm sits back from the grasp centre.
#: The fingers themselves are not probed: they slide past the object they are closing on, and a
#: surface cloud has points exactly where they travel.
_FINGER_MM: Final[float] = 50.0

#: How deep the palm is, and therefore how much space behind the fingers has to be free. A grasp
#: whose palm would be inside the table or inside the object is not a grasp anything could execute.
_PALM_MM: Final[float] = 30.0

#: Candidate displacements of the stored translation from the true grasp centre, along the approach,
#: in millimetres. Covers a translation at the grasp centre (0), at a palm or a wrist mount well
#: behind it, and the same in front for a publisher who measures the other way.
#:
#: The bound is set by what a gripper is: a wrist mount sits a palm plus a finger behind the
#: contact, and 250 mm covers the 209.4 mm base-to-fingertip of the widest hand in the reference
#: dataset with room over. A grid that ends too early fits its own boundary and reports a confident
#: number: one corpus stores the gripper's mounting face about 173 mm behind the annotated point,
#: and a grid ending at 120 puts every imported grasp 5 cm out.
_OFFSET_GRID_MM: Final[tuple[float, ...]] = tuple(float(v) for v in range(-250, 251, 5))

#: Unit hypotheses. A dataset in millimetres and one in metres are both common, and every other check
#: in this module passes on the wrong one because the geometry stays self-consistent.
_SCALES: Final[tuple[float, ...]] = (1.0, 0.001)

#: A contact this far from the cloud is not touching anything. Generous on purpose: real clouds are
#: sparse and partially observed, so a correct convention still misses by a few millimetres.
_CONTACT_TOLERANCE_MM: Final[float] = 15.0

#: Below this share of jaw contacts landing on the surface, no reading of the poses describes the
#: cloud at all. A synthetic control of poses unrelated to their scene lands far below it.
_MIN_TOUCHING: Final[float] = 0.5

#: How much better the winning closing axis must be. Either it puts this much more of its contacts on
#: the surface than the runner-up, or it gets them this much closer. On a synthetic scene with a
#: known answer the right axis separates on both routes, so neither carries the decision alone.
_AXIS_TOUCH_MARGIN: Final[float] = 0.05
_AXIS_CONTACT_MARGIN: Final[float] = 0.25

#: How much more often the winning approach must leave the palm somewhere to be. On the same scene
#: the right approach leaves the palm free far more often than its own reversal, so a gap of 0.10
#: sits well inside that separation and refuses anything closer.
_SIGN_CLEARANCE_MARGIN: Final[float] = 0.10


@dataclass(frozen=True, slots=True)
class Convention:
    """One reading of a published pose matrix.

    `centre_offset_m` is measured rather than enumerated as a category: it is how far along the
    approach the stored translation sits from the true grasp centre. Zero means the publisher stored
    the centre; a negative value of roughly a finger length means they stored a palm or a base.
    """

    approach_column: int
    approach_sign: int
    axis_column: int
    scale: float
    centre_offset_m: float

    def describe(self) -> str:
        letters = "xyz"
        sign = "+" if self.approach_sign > 0 else "-"
        unit = "metres" if self.scale == 1.0 else "millimetres"
        return (f"approach {sign}{letters[self.approach_column]}, "
                f"closing axis {letters[self.axis_column]}, {unit}, "
                f"translation {self.centre_offset_m * 1000.0:+.0f} mm from the centre")


@dataclass(frozen=True, slots=True)
class ConventionScore:
    """What one hypothesis measured. Every field is reported, including for the losers.

    The whole ranking is reported rather than only the winner: a winner that beats its runner-up by
    a hair is a different situation from one that beats it tenfold, and the difference is invisible
    if only the answer survives.
    """

    convention: Convention
    contact_mm: float
    touching: float
    straddle: float
    clearance: float
    score: float


@dataclass(frozen=True, slots=True)
class ConventionProof:
    """The outcome. Either `chosen` is set, or `refusal` says why nothing was."""

    chosen: ConventionScore | None
    ranked: tuple[ConventionScore, ...]
    refusal: str | None

    @property
    def settled(self) -> bool:
        return self.chosen is not None


class _Cloud:
    """The scene, with its nearest-neighbour index built once.

    The search over translation offsets is by far the largest part of the hypothesis space, and it
    pays only for tree queries, never for tree construction. Rebuilding the index inside the
    hypothesis loop instead is 1,176 rebuilds of a 3,000-point tree for a single scene, which does
    not finish inside two minutes.
    """

    __slots__ = ("points", "_tree")

    def __init__(self, points_mm: np.ndarray) -> None:
        self.points = points_mm
        try:
            from scipy.spatial import cKDTree  # noqa: PLC0415

            self._tree = cKDTree(points_mm)
        except ImportError:                                    # pragma: no cover (scipy is required)
            self._tree = None

    def nearest(self, query: np.ndarray) -> np.ndarray:
        """Distance from each query point to the nearest cloud point, in millimetres."""
        if self._tree is not None:
            return np.asarray(self._tree.query(query, k=1)[0], dtype=np.float64)
        gaps = np.linalg.norm(self.points[None, :, :] - query[:, None, :], axis=-1)
        return gaps.min(axis=1)


def _count_in_cylinder(cloud: np.ndarray, base: np.ndarray, direction: np.ndarray,
                       length: float, radius: np.ndarray) -> np.ndarray:
    """How many cloud points lie in each grasp cylinder, swept from `base` along `direction`.

    Vectorised over grasps against the whole cloud. Quadratic in principle, which is why callers
    subsample the grasp table: the discriminating power here comes from having a few hundred grasps
    under every hypothesis, not from having all of them under one.
    """
    delta = cloud[None, :, :] - base[:, None, :]
    along = np.einsum("gpc,gc->gp", delta, direction)
    perpendicular = np.linalg.norm(delta - along[:, :, None] * direction[:, None, :], axis=-1)
    inside = (along >= 0.0) & (along <= length) & (perpendicular <= radius[:, None])
    return inside.sum(axis=1)


def _axes(rotations: np.ndarray, convention: Convention) -> tuple[np.ndarray, np.ndarray]:
    """The unit approach and unit closing axis this hypothesis reads out of the stored rotations."""
    approach = convention.approach_sign * rotations[:, :, convention.approach_column]
    axis = rotations[:, :, convention.axis_column]
    approach = approach / np.linalg.norm(approach, axis=-1, keepdims=True).clip(1e-9)
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True).clip(1e-9)
    return approach, axis


def _best_offset_mm(cloud: _Cloud, translations_mm: np.ndarray, widths_mm: np.ndarray,
                    approach: np.ndarray, axis: np.ndarray) -> float:
    """Where along the approach the stored translation sits, by fitting the jaw contacts.

    One batched query for the whole grid. Every candidate offset moves the pair of contacts by a
    known amount, so all of them can be laid out at once and asked of the index in a single call.
    That turns the largest dimension of the search into the cheapest one.

    This is a fitted free parameter, and a fitted parameter can flatter a hypothesis that deserves
    nothing. It cannot flatter a wrong closing axis, because it slides the contacts along the
    approach while a wrong axis displaces them along the axis, and the two directions are
    perpendicular by construction. That is why the offset is fitted and the axis is enumerated.
    """
    grid = np.asarray(_OFFSET_GRID_MM, dtype=np.float64)
    centres = translations_mm[None, :, :] + grid[:, None, None] * approach[None, :, :]
    half = (widths_mm * 0.5)[None, :, None] * axis[None, :, :]
    query = np.concatenate([centres + half, centres - half], axis=1).reshape(-1, 3)
    gaps = cloud.nearest(query).reshape(len(grid), 2 * len(translations_mm))
    worst = np.maximum(gaps[:, :len(translations_mm)], gaps[:, len(translations_mm):])
    touching = (worst <= _CONTACT_TOLERANCE_MM).mean(axis=1)
    # Ordered by how many contacts land on the surface, and only then by how close they land. The
    # median rather than the mean, so one grasp whose contact is far away cannot choose the offset.
    order = np.lexsort((np.median(worst, axis=1), -touching))
    return float(grid[order[0]])


def _score_one(cloud: _Cloud, rotations: np.ndarray, translations_mm: np.ndarray,
               widths_mm: np.ndarray, convention: Convention) -> ConventionScore:
    """Measure the three geometric claims for one hypothesis. Everything here is in millimetres."""
    approach, axis = _axes(rotations, convention)
    centre = translations_mm + (convention.centre_offset_m * 1000.0) * approach
    half = (widths_mm * 0.5)[:, None]

    # Claim one: both jaw contacts lie on the surface. This is the claim that identifies the closing
    # axis, because the stored width is the object extent along the true axis and along no other.
    gaps = np.stack([cloud.nearest(centre + half * axis),
                     cloud.nearest(centre - half * axis)], axis=1)
    worst = gaps.max(axis=1)
    contact_mm = float(np.median(worst))
    touching = float(np.mean(worst <= _CONTACT_TOLERANCE_MM))

    # Claim two: material lies between the jaws. A grasp whose contacts both happen to sit near the
    # surface but which encloses nothing is a pose beside the object rather than on it.
    straddle = float(np.mean(cloud.nearest(centre) <= np.maximum(widths_mm * 0.5, 10.0)))

    # Claim three: the palm has somewhere to be. This is what fixes the sign, which claims one and
    # two are both blind to.
    #
    # A gripper is not a solid cylinder: it is two thin fingers that slide down past the object and
    # a palm that stops short of it, so the volume that must be free is the palm box behind the
    # fingers, and the fingers themselves are allowed to graze the surface they are closing on.
    # Compared as a solid cylinder of the jaw radius in front of the grasp against one behind it, a
    # top-down grasp on a box reads as blocked by the box's own top face, which is exactly the
    # surface the fingers pass on either side of, and the reversed approach wins.
    #
    # What makes this discriminate is scene content, not the table. On a multi-object scene with the
    # table removed the four readings of the approach still separate, because an object's own body
    # blocks the reverse of most grasps on it. What defeats the claim is an object small next to
    # the gripper with nothing around it: on a single 40 mm cube in isolation the four readings
    # score alike and this module refuses.
    palm_start = centre - (_FINGER_MM + _PALM_MM) * approach
    occupied = _count_in_cylinder(cloud.points, palm_start, approach, _PALM_MM,
                                  np.maximum(widths_mm * 0.5, 5.0))
    clearance = float(np.mean(occupied == 0))

    # The combination. `touching` carries the most weight because it is the claim with the widest
    # separation between a right and a wrong hypothesis, and `contact_mm` enters as a soft term so
    # that two hypotheses which both touch are still ordered by how well.
    score = (0.5 * touching + 0.2 * straddle + 0.3 * clearance) / (1.0 + contact_mm / 50.0)
    return ConventionScore(convention, round(contact_mm, 2), round(touching, 4),
                           round(straddle, 4), round(clearance, 4), round(score, 5))


def prove_convention(points_m: np.ndarray, poses: np.ndarray, widths: np.ndarray, *,
                     sample_grasps: int = 200, seed: int = 0) -> ConventionProof:
    """Work out how to read `poses`, or refuse to.

    `points_m` is the cloud in metres. `poses` is (G,4,4) as published, in whatever unit the
    publisher used. `widths` is (G,) in the same unit. Neither the frame nor the unit is trusted.

    Returns the ranked hypotheses and, if one wins by a margin, the convention to import under.

    A refusal does not mean "use the first one anyway". The correct response to one is to look at
    the dataset by hand, not to proceed with the best of several unsupported readings.
    """
    cloud_mm = np.asarray(points_m, dtype=np.float64) * 1000.0
    poses = np.asarray(poses, dtype=np.float64)
    widths = np.asarray(widths, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must be (G,4,4), got {poses.shape}")
    if len(widths) != len(poses):
        raise ValueError(f"{len(widths)} widths against {len(poses)} poses")
    if not len(poses):
        return ConventionProof(None, (), "the scene carries no grasp to measure")
    if len(cloud_mm) < 32:
        return ConventionProof(None, (), f"the cloud has only {len(cloud_mm)} points")

    rng = np.random.default_rng(seed)
    if len(poses) > sample_grasps:
        pick = rng.choice(len(poses), size=sample_grasps, replace=False)
        poses, widths = poses[pick], widths[pick]
    rotations = poses[:, :3, :3]
    translations = poses[:, :3, 3]

    cloud = _Cloud(cloud_mm)
    scored: list[ConventionScore] = []
    for scale in _SCALES:
        translations_mm = translations * scale * 1000.0
        widths_mm = widths * scale * 1000.0
        # A width that is not a plausible jaw opening rules the unit out before anything is measured,
        # which keeps the ranking from being padded with hypotheses nobody could believe.
        if not (1.0 <= float(np.median(widths_mm)) <= 300.0):
            continue
        for approach_column in range(3):
            for approach_sign in (1, -1):
                for axis_column in range(3):
                    if axis_column == approach_column:
                        continue
                    probe = Convention(approach_column, approach_sign, axis_column, scale, 0.0)
                    approach, axis = _axes(rotations, probe)
                    offset_mm = _best_offset_mm(cloud, translations_mm, widths_mm, approach, axis)
                    scored.append(_score_one(
                        cloud, rotations, translations_mm, widths_mm,
                        Convention(approach_column, approach_sign, axis_column, scale,
                                   offset_mm / 1000.0)))

    if not scored:
        return ConventionProof(None, (), "no unit hypothesis produced a plausible jaw opening; the "
                                         "widths are neither metres nor millimetres of a gripper "
                                         "we could model")
    ranked = tuple(sorted(scored, key=lambda s: s.score, reverse=True))

    # Two decisions, each judged by the claim that can actually answer it.
    #
    # The contact claim settles the closing axis and the unit, because a wrong closing axis puts the
    # contacts in mid-air. The palm claim then settles the approach column and its sign among the
    # readings that survive, because that is the only claim a sign flip changes at all: flipping the
    # sign leaves the contacts exactly where they were, so `touching`, `contact_mm` and `straddle`
    # are identical for a hypothesis and its reversal.
    #
    # One combined score cannot do this. It dilutes a decisive difference in one claim into a
    # narrow gap in a number that measures neither question, and refuses while holding the correct
    # convention in first place. A margin is only meaningful in the units of the decision it guards.

    by_family: dict[tuple[float, int], ConventionScore] = {}
    for candidate in scored:
        key = (candidate.convention.scale, candidate.convention.axis_column)
        held = by_family.get(key)
        if held is None or (candidate.touching, -candidate.contact_mm) > (held.touching,
                                                                         -held.contact_mm):
            by_family[key] = candidate
    families = sorted(by_family.values(), key=lambda s: (s.touching, -s.contact_mm), reverse=True)
    axis_winner = families[0]
    if axis_winner.touching < _MIN_TOUCHING:
        return ConventionProof(None, ranked,
                               f"the best reading puts only {axis_winner.touching:.0%} of jaw "
                               f"contacts on the surface, so no reading of these poses describes "
                               f"this cloud")
    if len(families) > 1:
        rival = families[1]
        touch_gap = axis_winner.touching - rival.touching
        contact_gain = 1.0 - axis_winner.contact_mm / max(rival.contact_mm, 1e-6)
        if touch_gap < _AXIS_TOUCH_MARGIN and contact_gain < _AXIS_CONTACT_MARGIN:
            return ConventionProof(None, ranked,
                                   f"the closing axis is not settled: reading it as "
                                   f"{axis_winner.convention.describe()} puts {axis_winner.touching:.0%} "
                                   f"of contacts on the surface at {axis_winner.contact_mm:.1f} mm, "
                                   f"and reading it as {rival.convention.describe()} manages "
                                   f"{rival.touching:.0%} at {rival.contact_mm:.1f} mm")

    family = sorted((s for s in scored
                     if s.convention.scale == axis_winner.convention.scale
                     and s.convention.axis_column == axis_winner.convention.axis_column),
                    key=lambda s: (s.clearance, s.touching), reverse=True)
    winner = family[0]
    if len(family) > 1 and winner.clearance - family[1].clearance < _SIGN_CLEARANCE_MARGIN:
        return ConventionProof(None, ranked,
                               f"the closing axis is settled but the approach is not: "
                               f"{winner.convention.describe()} leaves the palm free on "
                               f"{winner.clearance:.0%} of grasps and "
                               f"{family[1].convention.describe()} on {family[1].clearance:.0%}. "
                               f"A scene with no support surface cannot separate these, and neither "
                               f"can one whose objects are small next to the gripper")
    return ConventionProof(winner, ranked, None)


def agree(proofs: Sequence[ConventionProof]) -> ConventionProof | None:
    """The single convention every settled proof agrees on, or None.

    One scene is not evidence: a convention that wins on one cloud may have won by an accident of
    that object's symmetry, and a symmetric object is where a wrong axis is cheapest to hide. A
    convention that wins independently on a dozen unrelated scenes has been measured. Callers should
    prove several scenes and pass the results here rather than trusting the first.
    """
    settled = [p for p in proofs if p.chosen is not None]
    if not settled:
        return None
    first = settled[0].chosen
    assert first is not None
    for proof in settled[1:]:
        other = proof.chosen
        assert other is not None
        if (other.convention.approach_column != first.convention.approach_column
                or other.convention.approach_sign != first.convention.approach_sign
                or other.convention.axis_column != first.convention.axis_column
                or other.convention.scale != first.convention.scale):
            return None
    return settled[0]
