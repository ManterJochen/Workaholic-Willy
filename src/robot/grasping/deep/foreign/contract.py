"""What a training sample must be, stated once so two different producers can be held to it.

The training loop does not care where a sample came from. Two producers exist:

    the local generator  scenes rendered here, labelled by the physics labeller
    an imported corpus   a public dataset converted by one of the readers in this package

Both must hand the loop the same nine arrays with the same meanings, and this module is where those
meanings are written down and checked.

The frame is the whole contract, and four ways of getting it wrong are known: an offset applied in
the wrong frame, which measures worse than no correction at all; a rotation taken modulo pi, which
negates the closing axis; a ground-truth depth oracle that reports an object's centre instead of
its surface; a closing axis zeroed in the camera frame rather than the support plane. Each produces
a corpus that trains happily to a confident wrong answer and announces nothing, so a validator that
runs before training is the cheapest place to catch the fifth.

The checks below are about meaning rather than about types. A shape check catches a transposed
array. It does not catch a cloud in millimetres, a translation that refers to the gripper base
instead of the grasp centre, or labels rotated by an augmentation the points were not. Each of those
has a check of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping

import numpy as np

__all__ = ["SAMPLE_KEYS", "SampleProblem", "validate_sample", "describe_sample"]

#: Every array the training loop reads, with what it means. Written out rather than derived from a
#: producer, so that a producer changing its output is a contract break the validator reports, instead
#: of a silent redefinition that the loop absorbs.
SAMPLE_KEYS: Final[dict[str, str]] = {
    "points_m": "(N,3) float32. The cloud in METRES, in the sample own local frame: the xy mean "
                "removed and the support height subtracted from z.",
    "features": "(N,F) float32. Per-point input channels. Channels 0..2 are the unit surface normal.",
    "supervise": "(N,) bool. Which points carry a label and may therefore be scored.",
    "set_pair_point": "(P,) int32. Row into points_m for each (point, grasp) pair.",
    "set_pair_grasp": "(P,) int32. Row into the grasp table for each pair.",
    "set_grasp_position_m": "(G,3) float32. The grasp CENTRE, the midpoint between the two jaw "
                            "contacts, IN THE SAME LOCAL FRAME as points_m.",
    "set_grasp_approach": "(G,3) float32. Unit. The direction the gripper travels to reach the "
                          "grasp, pointing INTO the object. Its sign is meaningful.",
    "set_grasp_axis": "(G,3) float32. Unit. The direction the jaws close along. ANTIPODAL: it and "
                      "its negation are the same grasp, so its sign carries no information.",
    "set_grasp_width_m": "(G,) float32. The jaw opening in METRES.",
}

#: A cloud wider than this is almost certainly in millimetres. A tabletop scene is well under a metre
#: across, and the mistake this catches reads a 127 mm bracket as a 127 metre one.
_MAX_PLAUSIBLE_EXTENT_M: Final[float] = 5.0

#: A parallel jaw that opens wider than this is not a gripper this stack models. The widest profile
#: it carries opens 140 mm, so 300 mm leaves generous room for an unfamiliar gripper while still
#: refusing a value that is really a millimetre count wearing a metre label.
_MAX_PLAUSIBLE_WIDTH_M: Final[float] = 0.3

#: How far a grasp centre may sit from the nearest point of the cloud before the two are presumed to
#: be in different frames. A grasp centre lies between the jaws, so it is at most half a jaw opening
#: from the surface it grips, plus room for a sparse cloud. A frame mismatch misses by far more: a
#: cloud whose support height was never subtracted is off by that whole height.
_MAX_CENTRE_TO_CLOUD_M: Final[float] = 0.25


@dataclass(frozen=True, slots=True)
class SampleProblem:
    """One contract violation, named so a caller can act on it rather than print it.

    `fatal` separates "this sample cannot be trained on" from "this sample is suspicious". Both are
    reported; only a fatal problem refuses the sample.
    """

    key: str
    problem: str
    detail: str
    fatal: bool


def _array(sample: Mapping[str, Any], key: str) -> np.ndarray | None:
    value = sample.get(key)
    return None if value is None else np.asarray(value)


def _unit_problems(vectors: np.ndarray, key: str, tolerance: float = 5e-3) -> list[SampleProblem]:
    """Direction arrays must be unit length. A near-zero row is the specific failure worth naming."""
    if not len(vectors):
        return []
    norms = np.linalg.norm(vectors, axis=-1)
    dead = int(np.count_nonzero(norms < 1e-6))
    if dead:
        return [SampleProblem(key, "zero-length direction",
                              f"{dead} of {len(norms)} rows have no direction at all", True)]
    worst = float(np.abs(norms - 1.0).max())
    if worst > tolerance:
        return [SampleProblem(key, "not unit length",
                              f"worst deviation {worst:.4f}; a head trained against unnormalised "
                              f"targets learns the norm as well as the direction", True)]
    return []


def validate_sample(sample: Mapping[str, Any]) -> list[SampleProblem]:
    """Every way this sample breaks the contract, most serious first.

    An empty list means the sample can be trained on. It does not mean the sample is correct:
    nothing here can tell a well-formed corpus from a well-formed corpus of nonsense. What it
    catches are mismatches of unit, frame or convention, not mismatches of type.
    """
    problems: list[SampleProblem] = []
    for key in SAMPLE_KEYS:
        if key not in sample:
            problems.append(SampleProblem(key, "missing", SAMPLE_KEYS[key], True))
    if problems:
        return problems

    points = _array(sample, "points_m")
    features = _array(sample, "features")
    supervise = _array(sample, "supervise")
    position = _array(sample, "set_grasp_position_m")
    approach = _array(sample, "set_grasp_approach")
    axis = _array(sample, "set_grasp_axis")
    width = _array(sample, "set_grasp_width_m")
    pair_point = _array(sample, "set_pair_point")
    pair_grasp = _array(sample, "set_pair_grasp")
    assert points is not None and features is not None and supervise is not None
    assert position is not None and approach is not None and axis is not None and width is not None
    assert pair_point is not None and pair_grasp is not None

    # ---- shape and alignment ------------------------------------------------------------------
    if points.ndim != 2 or points.shape[1] != 3:
        problems.append(SampleProblem("points_m", "wrong shape", f"got {points.shape}", True))
        return problems
    count = len(points)
    for key, array, expected in (("features", features, count), ("supervise", supervise, count)):
        if len(array) != expected:
            problems.append(SampleProblem(key, "length mismatch",
                                          f"{len(array)} rows against {expected} points", True))
    if features.ndim != 2 or features.shape[1] < 3:
        problems.append(SampleProblem("features", "too few channels",
                                      "channels 0..2 must be the unit surface normal", True))
    grasps = len(position)
    for key, array in (("set_grasp_approach", approach), ("set_grasp_axis", axis),
                       ("set_grasp_width_m", width)):
        if len(array) != grasps:
            problems.append(SampleProblem(key, "length mismatch",
                                          f"{len(array)} rows against {grasps} grasps", True))
    if len(pair_point) != len(pair_grasp):
        problems.append(SampleProblem("set_pair_grasp", "length mismatch",
                                      f"{len(pair_grasp)} against {len(pair_point)} pair points",
                                      True))
    if problems:
        return problems

    # ---- indices ------------------------------------------------------------------------------
    if len(pair_point):
        if int(pair_point.max()) >= count or int(pair_point.min()) < 0:
            problems.append(SampleProblem("set_pair_point", "index out of range",
                                          f"range [{pair_point.min()}, {pair_point.max()}] against "
                                          f"{count} points", True))
        if grasps and (int(pair_grasp.max()) >= grasps or int(pair_grasp.min()) < 0):
            problems.append(SampleProblem("set_pair_grasp", "index out of range",
                                          f"range [{pair_grasp.min()}, {pair_grasp.max()}] against "
                                          f"{grasps} grasps", True))
        elif not grasps:
            problems.append(SampleProblem("set_pair_grasp", "pairs without a grasp table",
                                          f"{len(pair_grasp)} pairs point at an empty table", True))
    # Returning here is not tidiness. Every check below indexes with these pairs, so a validator
    # that carried on would raise IndexError on exactly the malformed input it exists to report.
    if problems:
        return sorted(problems, key=lambda p: (not p.fatal, p.key))

    # ---- units --------------------------------------------------------------------------------
    # The millimetre check. Every other check here passes on a corpus in the wrong unit, and the
    # network trains on it happily: the geometry is self-consistent, only a thousand times too big.
    if count:
        extent = float(np.ptp(points, axis=0).max())
        if extent > _MAX_PLAUSIBLE_EXTENT_M:
            problems.append(SampleProblem("points_m", "probably not metres",
                                          f"the cloud spans {extent:.1f} across its widest axis; "
                                          f"a tabletop scene in metres does not", True))
    if grasps and float(width.max()) > _MAX_PLAUSIBLE_WIDTH_M:
        problems.append(SampleProblem("set_grasp_width_m", "probably not metres",
                                      f"widest opening {float(width.max()):.1f}; the widest jaw we "
                                      f"model opens 0.14", True))
    if grasps and float(width.min()) <= 0.0:
        problems.append(SampleProblem("set_grasp_width_m", "non-positive opening",
                                      f"minimum {float(width.min()):.4f}", True))

    # ---- directions ---------------------------------------------------------------------------
    problems.extend(_unit_problems(approach, "set_grasp_approach"))
    problems.extend(_unit_problems(axis, "set_grasp_axis"))
    problems.extend(_unit_problems(np.asarray(features[:, :3], dtype=np.float64), "features"))
    if grasps and not problems:
        # The jaws close across the approach. A dataset that stores two columns of the same rotation
        # without saying which is which will fail this the moment the wrong pair is chosen, which is
        # exactly the confusion an importer has to resolve before it writes anything.
        skew = float(np.abs(np.einsum("ij,ij->i", approach, axis)).max())
        if skew > 0.05:
            problems.append(SampleProblem("set_grasp_axis", "not perpendicular to the approach",
                                          f"worst |cos| {skew:.3f}; the approach and the closing "
                                          f"axis have probably been swapped or mislabelled", True))

    # ---- frame --------------------------------------------------------------------------------
    # A grasp centre sits between the jaws, so it is near the surface it grips. A table of grasps in
    # a different frame from the cloud is off by a constant per scene, which looks like a systematic
    # bias a head could almost learn around, which is why it survives review.
    if grasps and count and not problems:
        chunk = position[:256]
        gaps = np.linalg.norm(points[None, :, :] - chunk[:, None, :], axis=-1).min(axis=1)
        worst = float(gaps.max())
        if worst > _MAX_CENTRE_TO_CLOUD_M:
            problems.append(SampleProblem("set_grasp_position_m", "probably a different frame",
                                          f"a grasp centre sits {worst:.3f} from the nearest cloud "
                                          f"point; a grasp lies against the surface it grips", True))
        elif float(np.median(gaps)) > _MAX_CENTRE_TO_CLOUD_M / 2.0:
            problems.append(SampleProblem("set_grasp_position_m", "suspiciously far from the cloud",
                                          f"median distance {float(np.median(gaps)):.3f}", False))

    # ---- supervision --------------------------------------------------------------------------
    if count and not bool(np.any(supervise)):
        problems.append(SampleProblem("supervise", "nothing supervised",
                                      "the sample carries no trainable point", False))
    elif len(pair_point) and not bool(np.all(supervise[pair_point])):
        loose = int(np.count_nonzero(~supervise[pair_point]))
        problems.append(SampleProblem("set_pair_point", "pairs on unsupervised points",
                                      f"{loose} pairs name a point the mask excludes, so the loss "
                                      f"and the metric will disagree about the denominator", True))
    return sorted(problems, key=lambda p: (not p.fatal, p.key))


def describe_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    """The numbers a human needs to judge an imported sample at a glance.

    Separate from `validate_sample`: the validator answers whether a sample may be trained on, and
    this answers what it contains. Without it an importer's output reads as a wall of arrays, and a
    corpus of all-black renders passes every automated check.
    """
    points = np.asarray(sample["points_m"], dtype=np.float64)
    width = np.asarray(sample["set_grasp_width_m"], dtype=np.float64)
    supervise = np.asarray(sample["supervise"], dtype=bool)
    pair_point = np.asarray(sample["set_pair_point"])
    out: dict[str, Any] = {
        "points": len(points),
        "grasps": len(width),
        "pairs": len(pair_point),
        "supervised_points": int(np.count_nonzero(supervise)),
        "supervised_share": round(float(np.mean(supervise)), 4) if len(supervise) else 0.0,
        "extent_m": [round(float(v), 4) for v in np.ptp(points, axis=0)] if len(points) else [],
        "width_mm": {"min": round(float(width.min()) * 1000.0, 1),
                     "median": round(float(np.median(width)) * 1000.0, 1),
                     "max": round(float(width.max()) * 1000.0, 1)} if len(width) else {},
    }
    if len(pair_point):
        _, per_point = np.unique(pair_point, return_counts=True)
        # The number that decides whether set prediction is worth anything on this corpus. A local
        # corpus has supervised points that admit more than one approach; an imported corpus with a
        # mean of 1.0 here has already collapsed its own multimodality, and no set head can recover
        # it. Under the K-slot head multimodality caps coverage rather than top-1: a low mean warns
        # about how much a set head has to gain, not about a top-1 ceiling.
        out["labels_per_point"] = {"mean": round(float(per_point.mean()), 3),
                                   "max": int(per_point.max()),
                                   "share_multi": round(float(np.mean(per_point > 1)), 4)}
    return out
