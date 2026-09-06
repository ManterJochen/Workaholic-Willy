"""Turning a 6-DoF grasp into what a network can classify, and back again.

Approach is classified into view bins, never regressed as a quaternion, per section 4 of the arc
plan: a regression head averages two valid grasps into an invalid one, because the space of
rotations is not flat and the loss does not know that.

So a grasp here is three numbers and two scalars:

    approach bin      which of `APPROACH_BINS` directions the gripper comes in along
    rotation bin      how far the jaw is turned about that approach, mod pi
    depth, width      the two continuous quantities that genuinely are continuous

Mod pi, and that is not a rounding convenience. A closing axis and its negative describe the same
physical grasp, because the jaw is symmetric, so an encoding that told them apart would ask the net
to learn a distinction that does not exist and would split the probability mass for every grasp in
two.

The approach set is a set of directions, not of axes. This is the one place where sharing code with
`datagen.grasps.labels._fibonacci_hemisphere` would be wrong rather than merely inconvenient. That
function enumerates closing axes, where a direction and its negative are the same thing, so a
half-sphere is the whole space. An approach is a direction: coming down onto an object and coming up
from underneath it are different grasps. The two functions look alike and mean different things;
unifying them would silently halve or double one of the two spaces.

And it is a cap, not a hemisphere, because the reference does not emit a hemisphere. An admissible
label is usually within 90 deg of straight down and some reach beyond it, because a grasp high
enough for the gripper to still start above the table may be approached from slightly below. The
cap is 135 deg and anything outside it is refused rather than snapped to the nearest bin.

`encode` followed by `decode` must return the same grasp to within one bin's angular resolution, on
real labels rather than on constructed ones. A convention error produces a corpus that trains
perfectly and describes nothing, and it is invisible in every loss curve.
"""

from __future__ import annotations

from typing import Final

import numpy as np

__all__ = [
    "APPROACH_BINS",
    "APPROACH_MAX_TILT_DEG",
    "ROTATION_BINS",
    "approach_bin",
    "approach_directions",
    "decode_grasp",
    "encode_grasp",
    "graspability_field",
    "rotation_bin",
    "rotation_frame",
]

#: How far from straight down an approach may tilt and still be inside the bin set.
#:
#: A pure lower hemisphere covers most admissible labels and silently mis-bins the rest, so the cap
#: is 135 deg, which leaves room above the tilt a label reaches, and what lies beyond it is refused
#: rather than snapped to the nearest bin.
#:
#: It is not a hemisphere because the reference does not emit one. An approach may tilt past
#: horizontal whenever the grasp is high enough that the gripper still starts above the table: a
#: 30 deg upward approach onto a tall object is a real grasp, and `check_jaw_grasp` accepts it.
APPROACH_MAX_TILT_DEG: Final[float] = 135.0

#: How many approach directions that cap is cut into. 100 over a cap of this size puts neighbours
#: about 18 deg apart, which is finer than the 20-25 deg a 2F-85 pad tolerates before a contact
#: slides off the surface it was aimed at. So the discretisation is not the binding error.
APPROACH_BINS: Final[int] = 100

#: In-plane rotation bins over [0, pi). 12 bins is 15 deg each. The jaw is symmetric, so pi is the
#: whole space; using 2*pi would ask the net to distinguish a grasp from itself.
ROTATION_BINS: Final[int] = 12

#: How far an approach may sit from its nearest bin before it is called out-of-domain. One bin
#: spacing: further than that and the nearest bin is not a description of the grasp, it is a guess.
_ADMISSIBLE_TOLERANCE_DEG: Final[float] = 20.0


def approach_directions(count: int = APPROACH_BINS,
                        max_tilt_deg: float = APPROACH_MAX_TILT_DEG,
                        *, pole_bin: bool = False) -> np.ndarray:
    """``count`` unit directions spread over a spherical cap about straight down, deterministically.

    Golden-angle spiral, the same construction the label generator uses for axes, evaluated over the
    cap ``z in [-1, -cos(max_tilt)]`` rather than over a hemisphere. Deterministic because the bin
    index is part of every stored target: a corpus encoded against one spiral and decoded against
    another is silently wrong everywhere, and nothing in a loss curve says so.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if not 0.0 < max_tilt_deg <= 180.0:
        raise ValueError(f"max_tilt_deg must be in (0, 180], got {max_tilt_deg}")
    # z runs from -1 (straight down) up to -cos(max_tilt), and the minus sign is the whole geometry.
    # The tilt of a direction d from straight down is `acos(-d_z)`, because straight down is (0,0,-1).
    # So `tilt <= T` is `d_z <= -cos(T)`, not `d_z <= cos(T)`; written the other way, `max_tilt=135`
    # gives a 45 deg cap, with 100 bins crammed 7.4 deg apart into a sixth of the intended domain.
    #
    # Sampling uniformly in z is what makes the spiral equal-area on a sphere; sampling uniformly in
    # the angle would crowd the pole.
    top = float(-np.cos(np.radians(max_tilt_deg)))
    if pole_bin:
        # Without this, the encoding cannot express straight down. The `+ 0.5` below is the
        # cell-centre convention that makes the spiral equal-area, and its cost is that no direction
        # sits at the pole: the nearest is 7.491 deg off and exactly one bin lies inside a 10 deg
        # cone. For a top-down grasping task that is the single most common approach, so every
        # vertical label is encoded as a tilted one and decoded back tilted, and a generator trained
        # on such a corpus concentrates its candidates on that one near-pole bin.
        #
        # With this on, bin 0 is straight down and the remaining `count - 1` keep the equal-area
        # spiral over the rest of the cap. It changes every bin index, so a corpus encoded with it
        # off and decoded with it on is silently wrong everywhere. No config field holds it and no
        # artifact key records it: it is a keyword argument no caller passes, so a corpus encoded
        # with it on cannot be decoded from an artifact alone. Default off: every corpus and model
        # that exists was built without it.
        index = np.arange(count - 1, dtype=np.float64) + 0.5
        z = -1.0 + (index / max(count - 1, 1)) * (top + 1.0)
        radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        theta = np.pi * (1.0 + 5.0**0.5) * index
        rest = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])
        directions = np.vstack([np.array([[0.0, 0.0, -1.0]]), rest])
        return directions / np.linalg.norm(directions, axis=1, keepdims=True)
    index = np.arange(count, dtype=np.float64) + 0.5
    z = -1.0 + (index / count) * (top + 1.0)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = np.pi * (1.0 + 5.0**0.5) * index              # the golden angle
    directions = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def _unit(vectors: np.ndarray) -> np.ndarray:
    array = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.all(norms > 1e-12):
        raise ValueError("a zero-length direction cannot be encoded")
    return array / norms


def approach_bin(approach: np.ndarray, count: int = APPROACH_BINS,
                 max_tilt_deg: float = APPROACH_MAX_TILT_DEG,
                 *, pole_bin: bool = False,
                 ) -> tuple[np.ndarray, np.ndarray]:
    """``(nearest bin, admissible)`` for each approach.

    The second return is the point. `argmax` alone always answers, so an approach outside the cap,
    such as a grasp reached from underneath, is snapped to the nearest bin and becomes a target that
    is 45 deg wrong with nothing anywhere reporting it. The analytic reference does emit such an
    approach. A caller that ignores `admissible` is training on invented labels.
    """
    directions = approach_directions(count, max_tilt_deg, pole_bin=pole_bin)
    similarity = _unit(approach) @ directions.T
    best = np.argmax(similarity, axis=1).astype(np.int64)
    closest = np.take_along_axis(similarity, best[:, None], axis=1).ravel()
    admissible = closest >= float(np.cos(np.radians(_ADMISSIBLE_TOLERANCE_DEG)))
    return best, admissible


def rotation_frame(approach: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A deterministic tangent basis ``(t, b)`` for the plane perpendicular to each approach.

    The reference tangent is built from whichever world axis is least parallel to the approach. Any
    fixed choice is degenerate for some approach; crossing with +z alone fails for a straight-down
    grasp, the single most common one here, and picking the least parallel axis moves the degeneracy
    to where no approach can reach it.
    """
    normals = _unit(approach)
    world = np.eye(3)
    # For each row, the world axis with the smallest |dot|: the most perpendicular one available.
    choice = np.argmin(np.abs(normals @ world.T), axis=1)
    tangent = np.cross(normals, world[choice])
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    binormal = np.cross(normals, tangent)
    return tangent, binormal


def rotation_bin(approach: np.ndarray, closing_axis: np.ndarray,
                 count: int = ROTATION_BINS) -> np.ndarray:
    """Which in-plane rotation bin the closing axis falls in, mod pi.

    The closing axis is projected onto the plane perpendicular to the approach rather than assumed
    to lie in it. Labels are not exactly orthogonal, because they come from contact geometry rather
    than from a constructed frame, and silently trusting orthogonality would fold that error into
    the angle.
    """
    normals = _unit(approach)
    axes = _unit(closing_axis)
    tangent, binormal = rotation_frame(normals)
    planar = axes - normals * np.einsum("ij,ij->i", axes, normals)[:, None]
    lengths = np.linalg.norm(planar, axis=1, keepdims=True)
    # A closing axis parallel to the approach has no in-plane angle at all; `GraspPoint` forbids it
    # as a grasp, so bin 0 is a placeholder the caller is expected to have filtered.
    planar = np.where(lengths > 1e-9, planar / np.maximum(lengths, 1e-12), tangent)
    angle = np.arctan2(np.einsum("ij,ij->i", planar, binormal),
                       np.einsum("ij,ij->i", planar, tangent))
    angle = np.mod(angle, np.pi)                      # the jaw is symmetric: pi is the whole space
    return np.minimum((angle / (np.pi / count)).astype(np.int64), count - 1)


def encode_grasp(approach: np.ndarray, closing_axis: np.ndarray, *,
                 approach_count: int = APPROACH_BINS,
                 rotation_count: int = ROTATION_BINS,
                 max_tilt_deg: float = APPROACH_MAX_TILT_DEG,
                 pole_bin: bool = False,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(approach bin, rotation bin, admissible)`` for each grasp.

    ``admissible`` is carried out of here rather than being applied here, because dropping rows is the
    caller's decision and the count of what was dropped is a number worth reporting: a corpus that
    quietly loses part of its labels to an encoding limit looks exactly like a corpus that is
    smaller.
    """
    bins, admissible = approach_bin(approach, approach_count, max_tilt_deg,
                                    pole_bin=pole_bin)
    # The rotation is measured against the binned approach, not the true one, because that is the
    # frame the net will decode in. Measuring against the true approach would encode an angle the
    # decoder can never reproduce, and the error would look like label noise.
    binned = approach_directions(approach_count, max_tilt_deg, pole_bin=pole_bin)[bins]
    return bins, rotation_bin(binned, closing_axis, rotation_count), admissible


def decode_grasp(approach_index: np.ndarray, rotation_index: np.ndarray, *,
                 approach_count: int = APPROACH_BINS,
                 rotation_count: int = ROTATION_BINS,
                 max_tilt_deg: float = APPROACH_MAX_TILT_DEG,
                 pole_bin: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """``(approach, closing axis)`` for each bin pair: the exact inverse of :func:`encode_grasp`.

    ``pole_bin`` must match what the corpus was encoded with. It changes every bin index, so
    decoding under the other setting is wrong everywhere and silent. No config field holds it and
    no artifact key records it, so the caller has to keep encode and decode in step itself.
    """
    directions = approach_directions(
        approach_count, max_tilt_deg, pole_bin=pole_bin)[np.asarray(approach_index, dtype=np.int64)]
    tangent, binormal = rotation_frame(directions)
    # The bin centre, not its edge: decoding to the edge biases every reconstruction by half a bin in
    # the same direction, which is a systematic error rather than a rounding one.
    angle = (np.asarray(rotation_index, dtype=np.float64) + 0.5) * (np.pi / rotation_count)
    axis = tangent * np.cos(angle)[:, None] + binormal * np.sin(angle)[:, None]
    return directions, axis / np.linalg.norm(axis, axis=1, keepdims=True)


def graspability_field(points_mm: np.ndarray, contact_points_mm: np.ndarray,
                       *, radius_mm: float = 10.0) -> np.ndarray:
    """Per-point graspability: is this bit of surface somewhere a labelled grasp actually touched?

    The supervision for the per-point head, and it is deliberately about contacts rather than about
    grasp centres. A grasp centre floats in the air between the pads; a net trained to light up there
    would be predicting a point that is not on the object at all, and the seeds sampled from that
    field would sit in free space.

    Returns a float array in [0, 1]: 1 where a contact lands within ``radius_mm``, 0 elsewhere. Binary
    rather than a distance falloff because the downstream use is seed sampling, and a soft field would
    mostly sample the fringe of every contact patch.
    """
    from src.robot.grasping.geometry._spatial import RadiusIndex  # noqa: PLC0415

    points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    contacts = np.asarray(contact_points_mm, dtype=np.float64).reshape(-1, 3)
    field = np.zeros(len(points), dtype=np.float32)
    if not len(points) or not len(contacts):
        return field
    index = RadiusIndex(points)
    for contact in contacts:
        near = index.query_radius(contact, radius_mm)
        if near.size:
            field[near] = 1.0
    return field
