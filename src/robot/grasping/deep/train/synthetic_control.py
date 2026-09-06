"""A target the stack must be able to learn, so a flat curve can be attributed.

When the architecture arm sits near its trivial baselines, two explanations fit every number
equally well:

    the architecture cannot learn a 6-DoF grasp head
    the corpus cannot teach one

Nothing else separates them. More assets, a bigger backbone, a different head: each is a bet on one
of the two, with no evidence for either.

This module removes the ambiguity by replacing the corpus's labels with a target that is a
deterministic function of an input channel. Every difficulty the real target has is deleted:

    one label per seed          no multimodality, so nothing to average
    derived from the normal     which the network receives as feature channels 0..2
    a fixed closing axis        no antipodal ambiguity
    a fixed width               nothing to regress
    offset along the normal     one direction, no two-contact coin flip

A network that cannot fit this is broken, and the fault is in the model, the loss, the optimiser or
the wiring, not in the data. A network that fits it and still fails on the real corpus says the
opposite, and the assets added afterwards are then justified rather than hoped for.

It is a control, not a benchmark. Passing it proves the stack can learn; it proves nothing at all
about grasping. Any number from here belongs beside the word "control" wherever it is quoted.

Two levels, because they fail for different reasons:

`normal`
    approach = -normal, exactly. The answer is a linear function of an input channel at the same
    point, so even a head with no backbone at all should reach it. If this fails, the failure is in
    the head, the loss or the optimiser.

`local`
    approach = -normal, but the width varies with the local point spread, so the answer needs the
    neighbourhood rather than the point. If `normal` passes and `local` fails, the failure is in the
    encoder.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["CONTROL_LEVELS", "synthetic_labels"]

#: The levels, in the order they should be run. See the module docstring for what each isolates.
CONTROL_LEVELS = ("normal", "local")

#: Where the synthetic grasp centre sits, along the inward normal from the seed. 25 mm: inside the
#: range the real corpus shows, so the head's output scaling is exercised the same way, and far
#: enough from zero that predicting nothing scores badly.
_OFFSET_MM = 25.0

#: The synthetic opening. Constant on purpose in the `normal` level: a target with nothing to regress
#: isolates the direction heads from the width head.
_WIDTH_MM = 50.0


def _empty(sample: dict[str, Any]) -> dict[str, Any]:
    """A sample with no labels at all, synthetic or otherwise.

    Returning the original would mix the two targets. A sample with nothing supervised has no
    synthetic answer to give, and handing back the corpus's real grasp table means a batch trains
    partly on the control and partly on the thing the control exists to replace. The run would then
    measure neither, and nothing would say so.
    """
    out = dict(sample)
    out["set_pair_point"] = np.zeros(0, dtype=np.int32)
    out["set_pair_grasp"] = np.zeros(0, dtype=np.int32)
    out["set_grasp_position_m"] = np.zeros((0, 3), dtype=np.float32)
    out["set_grasp_approach"] = np.zeros((0, 3), dtype=np.float32)
    out["set_grasp_axis"] = np.zeros((0, 3), dtype=np.float32)
    out["set_grasp_width_m"] = np.zeros(0, dtype=np.float32)
    return out


def _perpendicular(normals: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to each normal, chosen deterministically.

    Deterministic and continuous both matter here. A rule that picked the axis by a branch on the
    largest component would jump between neighbouring points, so the control would contain a
    discontinuity the real target does not have, turning a clean question into a harder one than
    the thing it is controlling for.
    """
    reference = np.zeros_like(normals)
    reference[:, 2] = 1.0
    # Where the normal is the reference the cross product vanishes, so fall back to another axis.
    axis = np.cross(normals, reference)
    degenerate = np.linalg.norm(axis, axis=-1) < 1e-6
    if degenerate.any():
        other = np.zeros_like(normals)
        other[:, 0] = 1.0
        axis[degenerate] = np.cross(normals[degenerate], other[degenerate])
    return axis / np.clip(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-9, None)


def synthetic_labels(sample: dict[str, Any], level: str = "normal") -> dict[str, Any]:
    """Replace a sample's grasp set with a target derived from its own input channels.

    Returns a new dict; the caller's sample is not modified. Every array the loop reads is rewritten
    so nothing downstream can accidentally mix a real label with a synthetic one:

        set_pair_point / set_pair_grasp     one pair per supervised point
        set_grasp_*                         one grasp per supervised point

    The supervised set is unchanged. Only the answers are synthetic, not which points are asked
    about, so the seed mixture, the label density and the class balance are exactly what the real run
    sees. A control that also made the supervision dense would be answering two questions at once and
    could not attribute either.
    """
    if level not in CONTROL_LEVELS:
        raise ValueError(f"unknown control level {level!r}; choose from {', '.join(CONTROL_LEVELS)}")
    if "supervise" not in sample or "features" not in sample:
        raise ValueError("the sample carries no supervise mask or features")

    supervise = np.asarray(sample["supervise"], dtype=bool)
    points_m = np.asarray(sample["points_m"], dtype=np.float64)
    normals = np.asarray(sample["features"], dtype=np.float64)[:, :3]
    index = np.flatnonzero(supervise)
    if not index.size:
        return _empty(sample)

    unit = normals[index]
    norm = np.linalg.norm(unit, axis=-1, keepdims=True)
    # A point whose normal never resolved carries a zero vector; it cannot define a target and is
    # dropped rather than given an arbitrary one.
    usable = norm[:, 0] > 1e-6
    index, unit, norm = index[usable], unit[usable], norm[usable]
    if not index.size:
        return _empty(sample)
    unit = unit / norm

    approach = -unit
    axis = _perpendicular(unit)
    offset = approach * (_OFFSET_MM / 1000.0)
    if level == "local":
        # The width follows the local spread, so the answer needs the neighbourhood rather than the
        # point. Measured against the whole cloud rather than a ball query: this is a control, and a
        # second radius parameter would be another thing to get wrong.
        spread = float(np.std(points_m[:, 2])) if len(points_m) else 0.0
        width = np.full(index.size, np.clip(0.02 + spread, 0.01, 0.085))
    else:
        width = np.full(index.size, _WIDTH_MM / 1000.0)

    out = dict(sample)
    out["set_pair_point"] = index.astype(np.int32)
    out["set_pair_grasp"] = np.arange(index.size, dtype=np.int32)
    out["set_grasp_position_m"] = (points_m[index] + offset).astype(np.float32)
    out["set_grasp_approach"] = approach.astype(np.float32)
    out["set_grasp_axis"] = axis.astype(np.float32)
    out["set_grasp_width_m"] = width.astype(np.float32)
    return out
