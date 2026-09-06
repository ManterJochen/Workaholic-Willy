"""How a grasp's orientation is written down, and the losses that go with it.

A parallel jaw rotated 180 degrees about its approach axis is the same grasp, so the closing axis
has no meaningful sign and the target is bimodal: `+c` and `-c` are both correct. One regressed
vector settles between the two modes, which is not a valid pose.
`corpus.grasp_encoding.decode_grasp` builds the closing axis as `tangent*cos(a) + binormal*sin(a)`
with the angle in `[0, pi)`, so the line round trips and the sign does not: the decoded axis often
comes back negated. Repairing that sign in the wrong frame is worse than not repairing it at all.

On a 2F-85, a representation that removes the ambiguity scores 79.0 % in simulation and 70.0 % on a
real robot, against 65.5 % / 53.6 % for a plain rotation matrix and 58.3 % / 55.9 % for a quaternion
(arXiv 2410.04826). The rule that follows: either model the multimodality or quotient it out;
regressing a single vector against a symmetric target is the one thing that cannot work. This module
quotients it out, and the generative head models it instead.

Only the sign ambiguity is removed here, not the genuine multi-mode one. A box graspable along x or
along y admits two different closing lines, and a unimodal head averages those too. Nothing in this
file addresses that.

The representation is a director, not a vector. An unsigned direction is a point of the real
projective plane, and the standard smooth embedding of that into a vector space is the symmetric
traceless matrix

    M(v) = v v^T - I/3

`M(v) == M(-v)` by construction, so the ambiguity is gone from the output rather than papered over
in the loss. A loss taking the smaller of the two branch errors (equivalently `1 - |v_hat . v|`) is
sign-invariant, but its output is still a single vector, so neighbouring seeds settle into opposite
branches and points between them settle near the perpendicular saddle.

Two properties earn this over the obvious `1 - (v_hat . v)^2`. First, no vanishing gradient at the
perpendicular: `d/dv (v_hat . v)^2` is zero exactly where the prediction is perpendicular to the
target, and the Frobenius loss in matrix space has no such stationary point. Second, a confidence
measure for free: the prediction is not renormalised, so `||M||_F` may fall below the target norm
`sqrt(2/3)`, a hedging network shrinks toward zero, and the shortfall is a concentration estimate
that costs nothing. Whether it correlates with angular error is a prediction to measure, not a
claim; `director_confidence` exists so the measurement is possible.

The same six numbers are also a Bingham `A` matrix at free concentration, so the Bingham variant is
a different loss on this identical head rather than a different head.

The seed-to-centre offset is predicted in the seed-local cloud frame, a plain 3-vector from the seed
to the grasp centre, and not in the decoded grasp frame `(approach, closing, binormal)`. Under the
180-degree flip the closing axis and the binormal both negate, so an offset in that frame has
components `(d, l, b)` that become `(d, -l, -b)`, and its target is bimodal one level down. The
seed-local frame needs no frame construction, so it carries no sign ambiguity and no hairy-ball
singularity in a canonical tangent. `frame_projection` still reports the decoded-frame components,
because how deep along the approach is the interpretable quantity and a metric may want it, but
nothing is supervised through a frame that can flip.

The cost is that equivariance must be learned rather than granted: the target depends on how the
object sits in the scene. Whether that costs anything is measurable with a rotation jitter and is
not yet measured.
"""

from __future__ import annotations

import math
from typing import Final

import torch

__all__ = [
    "AXIS_MODES",
    "axis_decode",
    "axis_loss",
    "vector_axis_decode",
    "vector_axis_loss",
    "DIRECTOR_NORM",
    "angular_error_deg",
    "approach_loss",
    "decode_frame",
    "director_confidence",
    "director_decode",
    "director_loss",
    "director_matrix",
    "director_params",
    "frame_projection",
]

#: Frobenius norm of `v v^T - I/3` for any unit `v`. The target the prediction's norm is compared
#: against, and the scale a confidence shortfall is measured in.
DIRECTOR_NORM: Final[float] = math.sqrt(2.0 / 3.0)

#: Below this, a vector is treated as having no direction at all. Metres-scale inputs are normalised
#: here, so this is a guard against exact zeros rather than a tolerance anyone should tune.
_EPS: Final[float] = 1e-8


def _unit(vectors: torch.Tensor) -> torch.Tensor:
    """Normalise along the last axis, leaving degenerate rows as zeros rather than as NaN."""
    norm = vectors.norm(dim=-1, keepdim=True)
    return vectors / norm.clamp_min(_EPS)


def director_matrix(axis: torch.Tensor) -> torch.Tensor:
    """`(..., 3)` unit axis to its `(..., 3, 3)` director. Identical for `axis` and `-axis`."""
    unit = _unit(axis)
    outer = unit.unsqueeze(-1) * unit.unsqueeze(-2)
    eye = torch.eye(3, dtype=outer.dtype, device=outer.device)
    return outer - eye / 3.0


def director_params(raw: torch.Tensor) -> torch.Tensor:
    """`(..., 6)` head output to a `(..., 3, 3)` symmetric traceless matrix.

    The six numbers fill the upper triangle in row-major order (xx, xy, xz, yy, yz, zz); the matrix
    is symmetrised and its trace removed. Removing the trace makes the map onto the target space
    exact, so the loss cannot be reduced by moving in a direction the target can never occupy.
    """
    if raw.shape[-1] != 6:
        raise ValueError(f"director parameters are 6 numbers, got {raw.shape[-1]}")
    xx, xy, xz, yy, yz, zz = raw.unbind(-1)
    rows = torch.stack([
        torch.stack([xx, xy, xz], dim=-1),
        torch.stack([xy, yy, yz], dim=-1),
        torch.stack([xz, yz, zz], dim=-1),
    ], dim=-2)
    trace = (xx + yy + zz) / 3.0
    eye = torch.eye(3, dtype=rows.dtype, device=rows.device)
    return rows - trace.unsqueeze(-1).unsqueeze(-1) * eye


def director_loss(raw: torch.Tensor, axis: torch.Tensor, *,
                  reduction: str = "mean") -> torch.Tensor:
    """Squared Frobenius distance between the predicted director and the labelled axis's.

    Sign-invariant in the target by construction, so a label may be handed over with either sign and
    the same number comes back. The corpus therefore keeps whichever sign it holds, with no
    canonicalisation pass.
    """
    difference = director_params(raw) - director_matrix(axis)
    per_item = difference.pow(2).flatten(start_dim=-2).sum(-1)
    if reduction == "none":
        return per_item
    if reduction == "sum":
        return per_item.sum()
    if reduction == "mean":
        return per_item.mean()
    raise ValueError(f"unknown reduction {reduction!r}")


def director_decode(raw: torch.Tensor) -> torch.Tensor:
    """The unsigned axis a predicted director stands for: the eigenvector of its largest eigenvalue.

    Inference only, and deliberately not differentiable in practice. Supervision happens in matrix
    space precisely so nothing has to back-propagate through an eigendecomposition, whose gradient is
    ill-conditioned exactly where two eigenvalues meet, which is where an uncertain prediction sits.

    The returned sign is whatever the decomposition produced. It carries no meaning and no consumer
    may depend on it.
    """
    matrices = director_params(raw)
    # `eigh` needs a symmetric input, which `director_params` guarantees, and returns eigenvalues in
    # ascending order, so the director's own axis is the last column.
    _, vectors = torch.linalg.eigh(matrices.detach().to(torch.float32))
    return vectors[..., -1].to(raw.dtype)


def director_confidence(raw: torch.Tensor) -> torch.Tensor:
    """How concentrated the prediction is, in `[0, 1]`, as `||M||_F / sqrt(2/3)` clamped above.

    A network with nothing to say shrinks its output toward zero, because the Frobenius loss against
    a fixed-norm target is minimised at the mean of the admissible targets, and the mean of two
    directors 90 degrees apart has a smaller norm than either. The shortfall is therefore a hedging
    signal, produced without a normalising constant.

    Whether it tracks angular error is unmeasured. Treat it as a prediction, not a result.
    """
    norm = director_params(raw).flatten(start_dim=-2).norm(dim=-1)
    return (norm / DIRECTOR_NORM).clamp(max=1.0)


def approach_loss(raw: torch.Tensor, approach: torch.Tensor, *,
                  reduction: str = "mean") -> torch.Tensor:
    """`1 - cos` between predicted and labelled approach. Sign-preserving, unlike the closing axis.

    The approach lies in the gripper's symmetry plane and points at the object, so flipping it is a
    different grasp, not the same one. An antipodal loss here would throw away a distinction the
    label carries: the difference between reaching for the object and reaching away.
    """
    cosine = (_unit(raw) * _unit(approach)).sum(-1)
    per_item = 1.0 - cosine
    if reduction == "none":
        return per_item
    if reduction == "sum":
        return per_item.sum()
    if reduction == "mean":
        return per_item.mean()
    raise ValueError(f"unknown reduction {reduction!r}")


def decode_frame(approach: torch.Tensor, axis: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """An orthonormal `(approach, closing, binormal)` from two heads that do not agree exactly.

    The approach is trusted and kept; the closing axis is Gram-Schmidted against it and
    renormalised; the binormal completes the frame. The order matters: the approach comes from a
    sign-preserving cosine loss and is the better determined of the two.

    A closing axis that is nearly parallel to the approach carries no usable component. An arbitrary
    perpendicular is substituted rather than a NaN frame; callers that care read
    `director_confidence`, which is low exactly in that case.
    """
    a = _unit(approach)
    residual = axis - (axis * a).sum(-1, keepdim=True) * a
    degenerate = residual.norm(dim=-1, keepdim=True) < 1e-4
    # A vector guaranteed not to be parallel to `a`: swap two components and negate one. Cheap,
    # branch-free, and correct for every unit input.
    fallback = torch.stack([-a[..., 1], a[..., 0], torch.zeros_like(a[..., 2])], dim=-1)
    still_degenerate = fallback.norm(dim=-1, keepdim=True) < 1e-4
    fallback = torch.where(
        still_degenerate,
        torch.stack([torch.zeros_like(a[..., 0]), -a[..., 2], a[..., 1]], dim=-1),
        fallback)
    residual = torch.where(degenerate, fallback, residual)
    c = _unit(residual)
    return a, c, torch.cross(a, c, dim=-1)


def frame_projection(offset: torch.Tensor, approach: torch.Tensor,
                     axis: torch.Tensor) -> torch.Tensor:
    """A seed-local offset expressed as `(depth, lateral, binormal)` in the decoded grasp frame.

    For reporting, never for supervision. The last two components change sign with the grasp's
    180-degree flip, so a loss on them has the bimodal target this module exists to avoid.
    """
    a, c, b = decode_frame(approach, axis)
    return torch.stack([(offset * a).sum(-1), (offset * c).sum(-1), (offset * b).sum(-1)], dim=-1)


def angular_error_deg(predicted: torch.Tensor, target: torch.Tensor, *,
                      antipodal: bool) -> torch.Tensor:
    """Angle between two directions in degrees, folded to `[0, 90]` when the sign has no meaning.

    `antipodal` must match what the quantity means: `True` for a closing axis, `False` for an
    approach. Reporting a closing axis without folding turns a perfect prediction into a 180-degree
    error on the labels stored with the other sign.
    """
    cosine = (_unit(predicted) * _unit(target)).sum(-1)
    if antipodal:
        cosine = cosine.abs()
    return torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))

#: How a closing axis may be written down. `director` is the default; see
#: `SlotHeadConfig.axis_mode`.
AXIS_MODES: Final[tuple[str, ...]] = ("director", "vector")


def vector_axis_loss(raw: torch.Tensor, axis: torch.Tensor, *,
                     reduction: str = "mean") -> torch.Tensor:
    """`1 - (v_hat . v)^2` on the first three outputs. Manifestly sign-invariant, no matrix.

    Perpendicular is the worst point of this loss rather than an attractor, so mode averaging is
    eliminated without a matrix representation.

    The approach head and the axis head are not the same job, so their error figures do not compare
    the two representations: the approach target is `-normal`, a linear function of an input
    channel, and the axis target is `normalise(cross(normal, z))`, which is not. A run whose
    approach head reports a lower error than its axis head is reporting that difference, not a
    result about either representation.

    This removes the sign ambiguity, not the multi-mode one. A box graspable along x or along y
    still has two distinct closing axes, and a unimodal head averages those whichever representation
    it uses; the K slots are what address that.
    """
    predicted = raw[..., :3]
    predicted = predicted / predicted.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    target = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    per_item = 1.0 - (predicted * target).sum(-1).pow(2)
    if reduction == "none":
        return per_item
    if reduction == "sum":
        return per_item.sum()
    if reduction == "mean":
        return per_item.mean()
    raise ValueError(f"unknown reduction {reduction!r}")


def vector_axis_decode(raw: torch.Tensor) -> torch.Tensor:
    """The unit axis the first three outputs stand for. No eigendecomposition, so it is exact."""
    predicted = raw[..., :3]
    return predicted / predicted.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def axis_loss(raw: torch.Tensor, axis: torch.Tensor, *, mode: str,
              reduction: str = "mean") -> torch.Tensor:
    """The axis term for whichever representation the run selected."""
    if mode == "director":
        return director_loss(raw, axis, reduction=reduction)
    if mode == "vector":
        return vector_axis_loss(raw, axis, reduction=reduction)
    raise ValueError(f"unknown axis mode {mode!r}; choose from {', '.join(AXIS_MODES)}")


def axis_decode(raw: torch.Tensor, *, mode: str) -> torch.Tensor:
    """The unit axis for whichever representation the run selected."""
    if mode == "director":
        return director_decode(raw)
    if mode == "vector":
        return vector_axis_decode(raw)
    raise ValueError(f"unknown axis mode {mode!r}; choose from {', '.join(AXIS_MODES)}")
