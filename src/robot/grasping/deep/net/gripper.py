"""What the network is told about the hand it is planning for.

A generator trained on one gripper learns that gripper. Making the gripper an input is what turns a
model into something a customer can adapt: outside work (arXiv 2606.00998) fine-tunes to a novel
gripper with about 140,000 grasps where training from scratch took two billion. That ratio is why
this seam exists before a second gripper does.

The representation is a swept volume, and a very small one. Their ablation:

    12-number swept volume     0.528 mAUC
    a tsdf of the gripper      0.432
    a learned point encoder    0.349

A hand-designed vector of twelve numbers beats a learned encoder over the gripper's own geometry by
a wide margin. The numbers are, at two opening states, an axis-aligned box approximating the space
the fingers sweep through as they close: three extents and three centre offsets each.

For a simple parallel jaw this vector is nearly degenerate, and that has to be said before any
result is read. Their 32 grippers span six kinematic families, so all twelve numbers move. A single
two-finger jaw moves fewer: the centre offsets across the closing axis and the binormal are zero by
symmetry, and only the closing extent differs between the open and half-open state. Eight of the
twelve are constant here. The published shape is kept anyway so a later cross-gripper model reads
the same field, but the information actually present is closer to six numbers.

On a single-gripper corpus the whole vector would be constant and carry no gradient at all: a
network handed the same fourteen numbers in every sample learns to ignore them, and the seam would
look built while doing nothing. That is the inert-switch shape this repository fences everywhere
else. The corpus is stamped with several jaws instead, `JAW_GEOMETRY` below carries their geometry,
and `eval/gripper_differential.py` measures the seam: the same hand twice must leave the proposed
position where it is, while `narrow_55` against `wide_140` must move it. Whether conditioning makes
the grasps better is a separate measurement.

Two numbers beyond the published twelve close limitations the authors state, that the
representation omits friction and cannot express the contact pattern:

* `pad_span_mm`: the contact patch along the approach, measured 38.0 mm on the 2F-85 and asymmetric
  about the grasp centre. Modelling the pad as a line rather than a rectangle understates the
  opening a grasp needs, and `too_wide` is a large share of all label rejections. A gripper input
  that cannot express the pad cannot explain those rejections.
* `friction_coefficient`: 0.5 on the 2F-85, a 26.6 degree cone. The antipodal verdict is the one
  result that moves with it.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import nn

__all__ = [
    "GRIPPER_VECTOR_DIM",
    "GRIPPER_LENGTH_SCALE_MM",
    "JAW_GEOMETRY",
    "GripperEncoder",
    "gripper_vector",
    "swept_volume_vector",
]

#: Twelve published numbers plus the two added here. Changing this changes a trained artifact's
#: input shape, so it travels on the model config rather than being read from here at load time.
GRIPPER_VECTOR_DIM: Final[int] = 14

#: Every length is divided by this before it reaches the network. Not cosmetic: a first layer
#: receiving raw metric offsets sees them at a far smaller spread than its feature channels, and the
#: learned weights do not compensate. A jaw is order 100 mm, so this puts every extent near 1.
GRIPPER_LENGTH_SCALE_MM: Final[float] = 100.0


def swept_volume_vector(
    *,
    aperture_mm: float,
    min_width_mm: float,
    finger_ahead_mm: float,
    finger_behind_mm: float,
    finger_thickness_mm: float,
    finger_width_mm: float,
    pad_span_mm: float,
    friction_coefficient: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """The `(14,)` description of one parallel jaw, in the local grasp frame.

    Frame, matching `ParallelJawGripperModel`: x closes between the contacts, y is the binormal, z
    is the approach. Lengths arrive in millimetres and leave divided by `GRIPPER_LENGTH_SCALE_MM`.

    Layout, in order: the fully-open state's three extents and three centre offsets, then the
    half-open state's six, then `pad_span` and `friction`.

    The swept box at opening `w` is the union of both fingers' travel from `w` down to closed. Along
    the closing axis a finger's outer face sits at `w/2 + thickness` and its inner face ends at
    `min_width/2`, so the union's bounding extent is `w + 2*thickness`. Along the approach the finger
    reaches `finger_ahead` past the grasp centre and `finger_behind` behind it, which is not
    symmetric on a real jaw: measured 28.72 against 33.37 on the 2F-85. That asymmetry is why the
    centre offset is carried at all.

    Plain keyword floats rather than a geometry object: `net/` describes architecture and imports
    nothing from the geometry or dataset layers, so nothing here can drift into depending on a model
    class that lives on the other side of the stack.
    """
    for name, value in (("aperture_mm", aperture_mm), ("finger_thickness_mm", finger_thickness_mm),
                        ("finger_width_mm", finger_width_mm), ("pad_span_mm", pad_span_mm)):
        if value <= 0.0:
            raise ValueError(f"{name} must be > 0, got {value}")
    if not 0.0 <= min_width_mm < aperture_mm:
        raise ValueError(f"min_width_mm must lie in [0, aperture_mm), got {min_width_mm}")

    scale = GRIPPER_LENGTH_SCALE_MM
    approach_extent = (finger_ahead_mm + finger_behind_mm) / scale
    approach_centre = (finger_ahead_mm - finger_behind_mm) / 2.0 / scale
    numbers: list[float] = []
    for opening_mm in (aperture_mm, aperture_mm / 2.0):
        numbers += [
            (opening_mm + 2.0 * finger_thickness_mm) / scale,   # extent across the closing axis
            finger_width_mm / scale,                            # extent along the binormal
            approach_extent,                                    # extent along the approach
            0.0,                                                # centre, closing axis: symmetric
            0.0,                                                # centre, binormal: symmetric
            approach_centre,                                    # centre, approach: not symmetric
        ]
    numbers += [pad_span_mm / scale, float(friction_coefficient)]
    return torch.tensor(numbers, dtype=dtype)


class GripperEncoder(nn.Module):
    """The gripper vector to an embedding the local head and the scorer both read.

    Three layers, as in the outside work. It is deliberately tiny: the input is fourteen numbers and
    the measurement that motivates the whole design is that a small hand-designed description beat a
    large learned one. Spending capacity here would be answering a question that has been answered.
    """

    def __init__(self, width: int, *, in_features: int = GRIPPER_VECTOR_DIM,
                 hidden: int = 64) -> None:
        super().__init__()
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}")
        self.in_features = int(in_features)
        self.width = int(width)
        self.layers = nn.Sequential(
            nn.Linear(self.in_features, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, self.width),
        )

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        """`(B, 14)` or `(14,)` to `(B, width)`. A single vector is broadcast by the caller, not here.

        Deliberately not broadcasting a bare `(14,)` to a batch: a silent broadcast is how an input
        that should vary per sample becomes one that never does, which is the exact failure this
        module's header warns about.
        """
        if vector.shape[-1] != self.in_features:
            raise ValueError(f"gripper vector is {self.in_features} numbers, "
                             f"got {vector.shape[-1]}")
        return self.layers(vector)


#: The jaws a corpus may be stamped with, by the name the extraction writes into every cloud.
#:
#: These numbers must match the labeller's own model: they describe the gripper the labels came
#: from, so a copy that drifted would train the head on a hand that does not exist. `net/` may not
#: import the labelling layer, which is why this is a copy at all; the same rule as the
#: feature-channel constants next door.
JAW_GEOMETRY: Final[dict[str, dict[str, float]]] = {
    "2f85": {"aperture_mm": 85.0, "min_width_mm": 5.0, "finger_ahead_mm": 28.72,
             "finger_behind_mm": 33.37, "finger_thickness_mm": 31.35, "finger_width_mm": 27.0,
             "pad_span_mm": 38.0, "friction_coefficient": 0.5},
    "narrow_55": {"aperture_mm": 55.0, "min_width_mm": 5.0, "finger_ahead_mm": 18.58,
                  "finger_behind_mm": 21.59, "finger_thickness_mm": 20.29,
                  "finger_width_mm": 17.47, "pad_span_mm": 24.59, "friction_coefficient": 0.5},
    "wide_140": {"aperture_mm": 140.0, "min_width_mm": 5.0, "finger_ahead_mm": 47.30,
                 "finger_behind_mm": 54.96, "finger_thickness_mm": 51.63,
                 "finger_width_mm": 44.47, "pad_span_mm": 62.58, "friction_coefficient": 0.5},
    "slim_pad": {"aperture_mm": 85.0, "min_width_mm": 5.0, "finger_ahead_mm": 28.72,
                 "finger_behind_mm": 33.37, "finger_thickness_mm": 31.35, "finger_width_mm": 27.0,
                 "pad_span_mm": 19.01, "friction_coefficient": 0.5},
}


def gripper_vector(name: str, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The `(14,)` description of a jaw, by the name a cloud is stamped with.

    An unknown name raises, and never falls back to the 2F-85. A corpus labelled for a hand this
    module does not know would otherwise be trained as if it carried the 2F-85, and a wrong
    conditioning input is worse than a constant one: constant teaches the head nothing, wrong teaches
    it a relationship that is not there. Refusing costs one traceback; the alternative costs a run
    nobody can interpret afterwards.
    """
    if name not in JAW_GEOMETRY:
        raise ValueError(f"unknown gripper stamp {name!r}; known: {', '.join(sorted(JAW_GEOMETRY))}. "
                         f"Refusing to guess, because a wrong gripper vector trains a relationship "
                         f"that does not exist.")
    return swept_volume_vector(**JAW_GEOMETRY[name], dtype=dtype)
