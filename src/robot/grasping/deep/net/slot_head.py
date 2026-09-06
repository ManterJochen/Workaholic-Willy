"""The head that turns one seed's features into the several grasps that seed admits.

The parts under it: `rotation.py` for how an orientation is written, `gripper.py` for what hand it
is planning for, `assignment.py` for which prediction answers which label, `set_loss.py` for what a
mismatch costs.

K slots, not one. A supervised point usually reaches more than one label, and those labels usually
differ in their approach, so a head with a single output answers a different question and can never
score above one on the coverage metric. `slots = 1` reproduces the single-output behaviour exactly,
so K is a sweep.

Each slot carries a learned query. Set prediction has a known instability: nothing ties slot `k` to
any particular mode, so under Hungarian matching the slots can thrash, swapping modes between steps
and learning the average of both. A per-slot embedding, concatenated onto the shared feature before
that slot's output layer, is the standard fix and costs `K x query_width` parameters. It lets a slot
specialise and keep its specialisation.

It is a mitigation, not a proof. Whether the slots settle has to show in the training curve, and the
per-slot confidence is the instrument: a head whose slots thrash reports confidences that do not
separate.

The width is not clamped to the gripper's aperture. The head is conditioned on the gripper, so
bounding its predicted opening by that gripper's aperture would be easy and physically correct, and
it would destroy the measurement this seam exists for: with one gripper the conditioning input is
constant and carries no gradient, and the open question is whether a net trained on varied grippers
learns to use it. A hard-coded coupling would make the seam look alive while proving nothing, which
is the inert-switch shape this repository fences everywhere else.

A consumer may clamp at inference. The head regresses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import nn

from src.robot.grasping.deep.net.gripper import GRIPPER_VECTOR_DIM, GripperEncoder
from src.robot.grasping.deep.net.rotation import AXIS_MODES
from src.robot.grasping.deep.net.set_loss import GraspSetPrediction

__all__ = ["SlotGraspHead", "SlotHeadConfig", "SLOT_MIXING", "SLOT_OUTPUTS"]

#: How a slot may differ from its neighbour. The default is first; see
#: `SlotHeadConfig.slot_mixing`.
SLOT_MIXING: Final[tuple[str, ...]] = ("affine", "film", "mlp")

#: Numbers emitted per slot: approach (3), director parameters (6), offset (3), width (1),
#: confidence (1). Spelled here so the slicing in `forward` and any consumer read one constant.
SLOT_OUTPUTS: Final[int] = 14


@dataclass(frozen=True, slots=True)
class SlotHeadConfig:
    """Every shape the head has, in one place a model card can quote."""

    #: How many grasps one seed may propose. Four covers the usual number of distinct approaches at
    #: a point and eight reaches most of the tail. `1` is one grasp per seed, the single-output
    #: behaviour.
    slots: int = 4
    #: Width of the shared trunk the slots read from.
    width: int = 256
    #: Width of the gripper embedding. Small on purpose: the input is fourteen hand-designed numbers,
    #: and the outside measurement is that a small such description beat a large learned one
    #: (0.528 mAUC against 0.349).
    gripper_width: int = 32
    #: Per-slot learned embedding. The guard against slot thrashing; see the module docstring.
    query_width: int = 32
    dropout: float = 0.0

    #: The two metric scales, derived from the corpus rather than searched for. A linear layer's
    #: natural output is order one and these targets are order 0.05, so without them the optimiser
    #: spends its first epochs finding a factor of twenty that arithmetic already knows.
    #:
    #: The two values below are the medians of the offset and of the opening over the corpus a run
    #: trains on; a cell whose objects are much larger or smaller wants its own.
    #:
    #: The outside work this design borrows from derives its translation scale the same way, from
    #: dataset statistics rather than a grid search, and reports the relationship as convex.
    offset_scale_m: float = 0.04

    #: And the width gets an offset, not just a scale, because its distribution is not centred on
    #: zero: a labelled opening usually sits close to the 85 mm aperture, i.e. most grasps are
    #: nearly wide open. A head whose output starts at zero starts 70 mm wrong on every seed and has
    #: to travel there. Starting at the median costs one constant.
    width_centre_m: float = 0.070
    width_scale_m: float = 0.05

    #: Predict a direction and take the length from the width head, instead of regressing a free
    #: 3-vector. Default True. `False` is the free-vector alternative, kept so the two can be
    #: compared.
    #:
    #: The length is solved by geometry rather than statistics: the offset's magnitude is half the
    #: gripper opening, because the seed is a contact point and the label is the grasp centre, half
    #: an opening away. Everything left is direction.
    #:
    #: A free vector cannot get there. The seed is one of two contacts and nothing tells the head
    #: which, so `offset . axis / (width/2)` comes out near `+1` or near `-1` about equally often.
    #: The L2-optimal answer to a target that is `+v` or `-v` with equal probability is their mean,
    #: which is zero, so a free head converges on a near-zero offset and stays there however hard
    #: it is pushed: it is already at its optimum. A unit vector cannot shrink to a mean.
    offset_from_width: bool = True

    #: How a slot is allowed to differ from its neighbour. Which value is best is open.
    #:
    #: `affine` concatenates the per-slot query onto the shared feature and applies one linear
    #: layer. Linear is affine, so the output is `W_s @ shared(x) + (W_q @ q_k + b)`: the
    #: input-dependent term is identical for every slot, and the slot-dependent term cannot see the
    #: input. That structural limit is why the alternatives exist.
    #:
    #: What `affine` freezes is the seed conditioning: two slots do differ, but by the same amount
    #: whatever seed they read. Whether `film` repairs that in a way that reaches grasp quality is
    #: not established, so treat the three values as an open arm. A slot spread read from seeds
    #: that are themselves feature-identical measures the seed mixture, not the slot mixing.
    #:
    #: `affine` stays the default so the arms already measured remain byte-identical, and the two
    #: alternatives are opt-in:
    #:
    #:     film  per-slot scale and shift on the shared feature before the same output layer. The
    #:           slot difference becomes multiplicative in the input. Initialised at scale 1 and
    #:           shift 0, so the model starts numerically identical to `affine` and moves away from
    #:           it only if the gradient asks. The cheaper repair: `2 x K x width` parameters.
    #:     mlp   a hidden layer with a nonlinearity between the concatenation and the outputs, so the
    #:           query and the feature can interact at all. The more general repair, and the more
    #:           expensive: it changes the output block's parameter shapes outright.
    #:
    #: Fixing this is not predicted to move the held-out approach error. The pathway it unblocks is
    #: the shared one, which two ablations show transfers nothing to unseen assets: replacing the
    #: approach output with its per-slot constants costs almost nothing held out while costing much
    #: more on train. What it should move is `coverage` and the meaning of `top1`.
    slot_mixing: str = "affine"

    #: How the closing axis is written down. `director` is the default and stays so, which keeps the
    #: runs already measured byte-identical.
    #:
    #: `vector` predicts a 3-vector with loss `1 - (v_hat . v)^2` and decodes by normalising.
    #: Manifestly sign-invariant, and perpendicular is the worst point of that loss rather than an
    #: attractor, so it cannot average two modes into their bisector.
    #:
    #: The two heads are not the same job, so their error figures do not compare the two
    #: representations: the approach control's target is `-normal`, a linear function of an input
    #: channel, and the axis control's is `normalise(cross(normal, z))`, which is not. A run whose
    #: approach head reports a lower error than its axis head is reporting that difference. The
    #: switch therefore exists to settle the question rather than to encode an answer: only an arm
    #: that holds the target fixed and moves the mode separates them.
    #:
    #: The output block keeps all fourteen numbers under both modes, so the shapes, the parameter
    #: count and the artifact layout do not move. `vector` reads the first three of the six axis
    #: slots and leaves the rest unused, which costs a few hundred parameters and buys a comparison
    #: that is otherwise confounded by a size change.
    axis_mode: str = "director"

    #: Part roles the affordance head classifies, in a fixed order. Empty is off, and off is the
    #: default.
    #:
    #: Without it, "grasp it by the handle" has no path to a grasp on this stack: the K-slot family
    #: carries no affordance head of its own. The binned family's affordance head is trained with a
    #: 6-way cross-entropy that takes a large share of the trunk's gradient, and nothing reads it
    #: at inference.
    #:
    #: The supervision is real and it is already in the corpus: a minority of labels carry a
    #: non-empty role such as `grip`, `body` or `handle`, and the rest carry the empty one.
    #:
    #: A separate output layer rather than a wider one: widening `SLOT_OUTPUTS` past 14 changes the
    #: shape of every tensor in every artifact already written, and an artifact that trains but
    #: cannot be loaded is the failure this avoids. Off, a net has neither the parameters nor the
    #: output; on, it gains one `Linear` and one tensor.
    #:
    #: The order matches the binned family's, so a role means the same thing on both.
    part_roles: tuple[str, ...] = ()


class SlotGraspHead(nn.Module):
    """`(N, F)` seed features plus `(N, 14)` gripper vectors to `K` grasps per seed.

    Seeds arrive already flattened, `N = batch x seeds`, which is the shape `set_loss` consumes.
    Carrying `(B, S, K, ...)` through the loss instead would mean every term reshaping the same way
    and one of them eventually doing it differently.
    """

    def __init__(self, in_features: int, config: SlotHeadConfig | None = None) -> None:
        super().__init__()
        config = config or SlotHeadConfig()
        if config.slots < 1:
            raise ValueError(f"slots must be >= 1, got {config.slots}")
        if in_features < 1:
            raise ValueError(f"in_features must be >= 1, got {in_features}")
        self.config = config
        self.in_features = int(in_features)

        self.gripper = GripperEncoder(config.gripper_width)
        trunk_in = self.in_features + config.gripper_width
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, config.width), nn.ReLU(inplace=True),
            nn.Dropout(config.dropout) if config.dropout else nn.Identity(),
            nn.Linear(config.width, config.width), nn.ReLU(inplace=True),
        )
        if config.axis_mode not in AXIS_MODES:
            raise ValueError(f"unknown axis_mode {config.axis_mode!r}; "
                             f"choose from {', '.join(AXIS_MODES)}")
        if config.slot_mixing not in SLOT_MIXING:
            raise ValueError(f"unknown slot_mixing {config.slot_mixing!r}; "
                             f"choose from {', '.join(SLOT_MIXING)}")
        # One embedding per slot, so a slot can specialise and keep the specialisation.
        self.queries = nn.Parameter(torch.randn(config.slots, config.query_width) * 0.02)
        if config.slot_mixing == "mlp":
            self.output: nn.Module = nn.Sequential(
                nn.Linear(config.width + config.query_width, config.width),
                nn.ReLU(inplace=True),
                nn.Linear(config.width, SLOT_OUTPUTS))
        else:
            self.output = nn.Linear(config.width + config.query_width, SLOT_OUTPUTS)
        # Created last and from zeros, both deliberately. Zeros mean scale 1 and shift 0, so a
        # `film` head starts numerically identical to an `affine` one and can only diverge by
        # learning. Last, because `torch.zeros` draws no randomness but the order of parameter
        # construction fixes the RNG stream for everything built after it, and a repair that
        # perturbed the initialisation of the trunk would be two experiments in one curve.
        self.film = (nn.Parameter(torch.zeros(config.slots, 2, config.width))
                     if config.slot_mixing == "film" else None)
        # Created last, after `film`, for the same reason: inserting it earlier would shift the RNG
        # stream and perturb the initialisation of every tensor below it. Absent when roles are off.
        self.affordance = (nn.Linear(config.width + config.query_width, len(config.part_roles))
                           if config.part_roles else None)

    def forward(self, features: torch.Tensor, gripper: torch.Tensor) -> GraspSetPrediction:
        """`(N, F)` and `(N, 14)` in, a `GraspSetPrediction` of `(N, K, ...)` out.

        The gripper vector is per seed and the caller expands it. Broadcasting a bare `(14,)` here
        would be convenient and would hide the case this seam exists to expose: an input that should
        vary per sample and never does.
        """
        if features.dim() != 2 or features.shape[-1] != self.in_features:
            raise ValueError(f"features must be (N, {self.in_features}), "
                             f"got {tuple(features.shape)}")
        if gripper.shape != (features.shape[0], GRIPPER_VECTOR_DIM):
            raise ValueError(f"gripper must be {(features.shape[0], GRIPPER_VECTOR_DIM)}, "
                             f"got {tuple(gripper.shape)}")

        shared = self.trunk(torch.cat([features, self.gripper(gripper)], dim=-1))
        seeds, slots = shared.shape[0], self.config.slots
        if self.film is None:
            wide = shared.unsqueeze(1).expand(seeds, slots, shared.shape[-1])
        else:
            # The slot's own scale and shift, so what separates two slots depends on what the seed
            # looks like. Under `affine` that separation is a constant the input cannot reach.
            wide = shared.unsqueeze(1) * (1.0 + self.film[:, 0]) + self.film[:, 1]
        per_slot = torch.cat([
            wide,
            self.queries.unsqueeze(0).expand(seeds, slots, self.config.query_width),
        ], dim=-1)
        raw = self.output(per_slot)
        width = self.config.width_centre_m + raw[..., 12] * self.config.width_scale_m
        return GraspSetPrediction(
            approach=raw[..., 0:3],
            axis_params=raw[..., 3:9],
            offset_m=self._offset(raw[..., 9:12], width, raw[..., 0:3]),
            width_m=width,
            confidence=raw[..., 13],
            axis_mode=self.config.axis_mode,
            # From its own layer over the same per-slot feature, so the fourteen numbers above keep
            # their meaning and their weights.
            part_logits=(self.affordance(per_slot) if self.affordance is not None else None),
        )

    def _offset(self, raw: torch.Tensor, width_m: torch.Tensor,
                approach: torch.Tensor) -> torch.Tensor:
        """The seed-to-centre offset, as a direction times half the opening or as a free vector.

        The length is not learned when `offset_from_width` is on: it is `width / 2`, which the width
        head already predicts. The coupling is deliberate; a wrong width moves the offset with it,
        which is what the geometry does too.
        """
        if not self.config.offset_from_width:
            return raw * self.config.offset_scale_m
        # A degenerate raw direction falls back to the approach, not to zero. Dividing by a clamped
        # norm keeps the gradient finite but still emits a zero-length offset when the raw vector is
        # exactly zero, and a zero offset means "the grasp centre is exactly on the contact", which
        # is geometrically wrong rather than merely uninformative. The approach is the principled
        # default: the centre lies ahead of a contact along it. Unreachable in practice, correct
        # when reached.
        norm = raw.norm(dim=-1, keepdim=True)
        fallback = approach / approach.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        direction = torch.where(norm > 1e-6, raw / norm.clamp_min(1e-6), fallback)
        # And if both are degenerate there is no direction to be had; a fixed one keeps the shape
        # valid and the confidence head is what says the slot is empty.
        second = direction.norm(dim=-1, keepdim=True)
        straight_down = torch.zeros_like(direction)
        straight_down[..., 2] = -1.0
        direction = torch.where(second > 1e-6, direction, straight_down)
        return direction * (width_m.unsqueeze(-1) * 0.5)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, slots={self.config.slots}, "
                f"width={self.config.width}, offset_from_width={self.config.offset_from_width}")
