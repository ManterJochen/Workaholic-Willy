"""Turning K slots per seed into grasps a cell can execute.

The inference half of the set head. Training compares predictions against labels in one local frame
and never has to produce a pose anybody acts on; this does, which brings in two things training
never worried about: units and frames.

The frame round trip. `build_sample` hands the network a local cloud: xy with the scene's mean
removed, z with the support height subtracted. It carries `centre_xy_mm` and `support_height_mm`
precisely so the transform can be undone, and a decoder that forgets to undo it emits grasps offset
by a per-scene constant. That is the shape a head can almost learn around in training and that puts
a gripper in the wrong place in a cell. Nothing raises.

Units. The network works in metres because that is what keeps its targets order-one. This
repository is millimetres everywhere else. The conversion happens here, once, and is checkable
against a hand-computed case rather than against itself.

What a slot becomes.

    approach    normalised. Sign-meaningful: it points into the object.
    closing     the director's top eigenvector, then Gram-Schmidted against the approach.
                Its sign is arbitrary and no consumer may depend on it.
    position    seed + offset, in metres, then back to the scene frame, then to millimetres.
    width       as predicted, clamped to what the gripper can actually open to.
    score       the slot's confidence through a sigmoid.

The width is clamped here and is not clamped in the head. The head regresses freely on purpose:
a hard coupling to the gripper's aperture would make the conditioning seam look alive while proving
nothing about whether the net learned to use it. A cell, on the other hand, cannot be handed an
opening its gripper does not have. Two different jobs, two different places.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.robot.grasping.deep.net.rotation import axis_decode, decode_frame
from src.robot.grasping.deep.net.set_loss import GraspSetPrediction

__all__ = ["DecodedGrasp", "decode_set_prediction"]

_M_TO_MM = 1000.0


@dataclass(frozen=True, slots=True)
class DecodedGrasp:
    """One executable grasp, in the frame and units the rest of the stack speaks."""

    position_mm: np.ndarray     # (3,) in the scene frame the cloud arrived in
    approach: np.ndarray        # (3,) unit, points into the object
    axis: np.ndarray            # (3,) unit, closing direction, sign arbitrary
    width_mm: float
    score: float
    seed_index: int
    seed_position_mm: np.ndarray
    slot: int
    confidence: float
    #: Which part of the object this grasp is on, or None when the net has no affordance head.
    #:
    #: The reason the head exists at all: "grasp it by the handle" needs a role attached to a pose,
    #: not to a point, and this is the pose. A role classifier nothing reads at inference still
    #: takes a share of the trunk's gradient every epoch and returns nothing, which is what the
    #: retired binned family did.
    #:
    #: The empty string is a real answer, not a missing one: it means "this object has no parts to
    #: tell apart".
    part_role: str | None = None
    #: The head's confidence in that role, `None` alongside it. Separate from `confidence`, which is
    #: about whether the slot answers a grasp at all.
    part_score: float | None = None


def decode_set_prediction(prediction: GraspSetPrediction, seed_index: torch.Tensor,
                          points_m: torch.Tensor, *, centre_xy_mm: np.ndarray,
                          support_height_mm: float, min_width_mm: float, max_width_mm: float,
                          min_confidence: float = 0.0,
                          part_roles: "tuple[str, ...]" = ()) -> list[DecodedGrasp]:
    """`(S, K, ...)` slots to a flat list of grasps, ordered by confidence.

    Ordered, and the order is the product. A cell takes the first candidate it can reach, so a
    decoder that returned slots in emission order would hand it slot zero of seed zero regardless of
    what the head thought. The confidence head exists to rank these and this is where that ranking
    becomes real.

    `min_confidence` drops slots the head itself says are empty. Zero by default: a threshold chosen
    before the confidence distribution of a trained head is measured is a threshold nobody can
    justify, and this argument is where such a measurement is applied.
    """
    if prediction.approach.dim() != 3:
        raise ValueError(f"expected (S, K, 3) slots, got {tuple(prediction.approach.shape)}")
    if seed_index.numel() != prediction.approach.shape[0]:
        raise ValueError(f"{seed_index.numel()} seed(s) against "
                         f"{prediction.approach.shape[0]} rows of slots")
    if not min_width_mm < max_width_mm:
        raise ValueError(f"min_width_mm {min_width_mm} must be below max_width_mm {max_width_mm}")

    with torch.no_grad():
        seeds, slots = prediction.approach.shape[:2]
        axis = axis_decode(prediction.axis_params, mode=prediction.axis_mode)
        approach, closing, _ = decode_frame(prediction.approach, axis)
        confidence = torch.sigmoid(prediction.confidence)
        # `None` on every net without an affordance head.
        part_logits = prediction.part_logits

        seed_points_m = points_m[seed_index].unsqueeze(1)
        # Local metres to local millimetres, then the sample's own transform undone: the xy mean back
        # on, the support height back on. See the module header for what forgetting this costs.
        centre_mm = torch.as_tensor(np.asarray(centre_xy_mm, dtype=np.float64),
                                    dtype=torch.float64)
        position_mm = (seed_points_m + prediction.offset_m).to(torch.float64) * _M_TO_MM
        position_mm[..., 0] += centre_mm[0]
        position_mm[..., 1] += centre_mm[1]
        position_mm[..., 2] += float(support_height_mm)
        seed_mm = seed_points_m.to(torch.float64).expand(seeds, slots, 3) * _M_TO_MM
        seed_mm = seed_mm.clone()
        seed_mm[..., 0] += centre_mm[0]
        seed_mm[..., 1] += centre_mm[1]
        seed_mm[..., 2] += float(support_height_mm)

        width_mm = (prediction.width_m * _M_TO_MM).clamp(min_width_mm, max_width_mm)

    out: list[DecodedGrasp] = []
    for seed in range(seeds):
        for slot in range(slots):
            value = float(confidence[seed, slot])
            if value < min_confidence:
                continue
            # The role, when the net has one. Softmax over the slot's logits: `argmax` names the
            # part and its probability says how much to believe the name, which a caller filtering
            # on "handle" needs in order to refuse a weak claim rather than act on it.
            role, role_score = None, None
            if part_logits is not None and part_roles:
                probabilities = torch.softmax(part_logits[seed, slot].to(torch.float64), dim=-1)
                best = int(torch.argmax(probabilities))
                if best < len(part_roles):
                    role, role_score = part_roles[best], float(probabilities[best])
            out.append(DecodedGrasp(
                part_role=role, part_score=role_score,
                position_mm=position_mm[seed, slot].numpy(),
                approach=approach[seed, slot].to(torch.float64).numpy(),
                axis=closing[seed, slot].to(torch.float64).numpy(),
                width_mm=float(width_mm[seed, slot]),
                score=value,
                seed_index=int(seed_index[seed]),
                seed_position_mm=seed_mm[seed, slot].numpy(),
                slot=slot,
                confidence=value))
    out.sort(key=lambda grasp: grasp.score, reverse=True)
    return out


def as_metadata(grasp: DecodedGrasp) -> dict[str, Any]:
    """The set family's own per-candidate keys, on a helper nothing calls.

    These are not the protocol's de-facto contract. That contract is the score breakdown
    `protocol.py` names (`total`, `geometric`, `stability`, `reachability`, `feasibility_score`,
    `pose_confidence`, `score_components`), which the success model, the RL candidate rows and the
    shadow aggregator read, and which a generator that omits them silently empties rather than
    crashing. The keys here are the set family's additions: `slot` and `confidence` trace its
    output back to which of its K answers a cell took, and `DeepGraspCalculator` builds its own
    metadata inline.
    """
    return {
        "generator": "deep_set",
        "seed_index": grasp.seed_index,
        "seed_position_mm": tuple(float(v) for v in grasp.seed_position_mm),
        "slot": grasp.slot,
        "slot_confidence": grasp.confidence,
        # Not calibrated, and the key says so wherever this travels. The confidence is a BCE output
        # over "this slot answers a real grasp", never a probability that the grasp holds.
        "confidence_is_calibrated": False,
    }
