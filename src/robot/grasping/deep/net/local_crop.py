"""Stage 3: a ball of points around the seed, in the seed's own frame, scaled by a measured kappa.

What the head sees by default: `SetGenerator.propose` takes its seed feature from `seed_features`,
which gathers `encoded[batch_index, point_index]` and concatenates this crop only when
`SetGeneratorConfig.crop` is set. The default is `crop=None`, so without the flag the head reads
one gathered per-point feature and the gripper vector, with no crop, no local frame and no scale
normalisation. This module is the architecture plan's section 3.3: the ball crop, the seed-local
frame and the measured kappa.

The difference is not cosmetic. The encoder does separate points, but the head is handed one vector
per seed and nothing about the neighbourhood that vector sits in. Whether a crop helps is an arm,
not an assumption, which is why this ships default-off.

Kappa is measured here, not copied. Outside work (`2507.13097`) derives it as the reciprocal of the
mean per-axis range of the crop and reports 3.27 for a household distribution. Measured on v5:

    radius     40 mm   60 mm   85 mm   120 mm
    mean range 0.054   0.081   0.119   0.167   (metres)
    kappa      18.6    12.4    8.4     6.0

Kappa needs no per size band value. Across a wide object-size range the spread at r = 85 mm stays
small, because the radius caps the range: a ball of radius r holds at most 2r of span, so once an
object is larger than the ball the crop stops growing with it. The crop's extent is a property of
the crop. What kappa depends on is the radius, and copying 3.27 would be wrong by 2.6x at
r = 85 mm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import nn

__all__ = ["LocalCropConfig", "LocalCrop", "kappa_for_radius", "MEASURED_RANGE_M"]

#: Measured on v5: crop radius in mm to the mean per-axis range of that crop in metres. Interpolated
#: between, and refused outside, because a kappa outside the measured band is a guess wearing a
#: measurement's name.
MEASURED_RANGE_M: Final[dict[float, float]] = {
    40.0: 0.054, 60.0: 0.081, 85.0: 0.119, 120.0: 0.167,
}


def kappa_for_radius(radius_mm: float) -> float:
    """The scale a crop of this radius is divided by, from the measured table.

    Refuses outside the measured band rather than extrapolating. The relation is close to linear
    over 40 to 120 mm, which is exactly the kind of observation that invites extrapolating to 300 mm
    and getting a number nobody measured.
    """
    radii = sorted(MEASURED_RANGE_M)
    if not radii[0] <= radius_mm <= radii[-1]:
        raise ValueError(
            f"crop radius {radius_mm} mm is outside the measured band {radii[0]} to {radii[-1]} mm. "
            f"Kappa is derived from corpus statistics, and there are none out there. Measure it "
            f"first (the sweep is in the plan's 3.3) or pick a radius inside the band.")
    for low, high in zip(radii, radii[1:]):
        if low <= radius_mm <= high:
            span = high - low
            weight = 0.0 if span == 0 else (radius_mm - low) / span
            mean_range = MEASURED_RANGE_M[low] * (1 - weight) + MEASURED_RANGE_M[high] * weight
            return 1.0 / mean_range
    raise AssertionError("unreachable: the band check above covers every radius")


@dataclass(frozen=True, slots=True)
class LocalCropConfig:
    """Every shape stage 3 has, in one place a model card can quote."""

    #: Ball radius around the seed, in millimetres. 85 is the reference jaw's aperture, so the crop
    #: is "what the hand could reach", which is the only radius with a physical meaning here.
    radius_mm: float = 85.0
    #: How many neighbours are kept. A fixed count keeps the tensor rectangular; points beyond the
    #: radius are masked rather than dropped, so a sparse neighbourhood stays sparse instead of
    #: silently reaching further for filler.
    neighbours: int = 32
    width: int = 64


class LocalCrop(nn.Module):
    """`(S, width)` crop features, one per seed, from the cloud around it.

    A pointnet-shaped encoder, deliberately the simplest one. A crop encoder that is itself a
    transformer would make an arm against the no-crop baseline uninterpretable: two things would have
    changed. Per-point MLP then a masked max-pool is the smallest thing that can carry neighbourhood
    geometry at all.
    """

    def __init__(self, in_features: int, config: LocalCropConfig | None = None) -> None:
        super().__init__()
        self.config = config or LocalCropConfig()
        if self.config.neighbours < 1:
            raise ValueError(f"neighbours must be >= 1, got {self.config.neighbours}")
        if in_features < 1:
            raise ValueError(f"in_features must be >= 1, got {in_features}")
        self.in_features = in_features
        self.kappa = kappa_for_radius(self.config.radius_mm)
        # 3 relative coordinates (already scaled) + the neighbour's own encoded feature.
        self.point = nn.Sequential(
            nn.Linear(3 + in_features, self.config.width), nn.SiLU(),
            nn.Linear(self.config.width, self.config.width))
        # The output starts at zero so a crop-enabled net begins numerically identical to one
        # without it. An arm whose two halves start from different initialisations measures the
        # initialisation. Same reason `film` mixing zero-initialises its own projection.
        self.out = nn.Linear(self.config.width, self.config.width)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, points_m: torch.Tensor, encoded: torch.Tensor,
                batch_index: torch.Tensor, point_index: torch.Tensor) -> torch.Tensor:
        """`points_m` is `(B, P, 3)`, `encoded` is `(B, P, C)`, the seeds are a flat list."""
        if points_m.dim() != 3 or points_m.shape[-1] != 3:
            raise ValueError(f"points_m must be (B, P, 3), got {tuple(points_m.shape)}")
        if encoded.shape[:2] != points_m.shape[:2]:
            raise ValueError(f"encoded {tuple(encoded.shape)} does not match the cloud "
                             f"{tuple(points_m.shape)}")
        seeds = points_m[batch_index, point_index]                      # (S, 3)
        cloud = points_m[batch_index]                                   # (S, P, 3)
        delta = cloud - seeds.unsqueeze(1)
        distance = delta.norm(dim=-1)
        count = min(self.config.neighbours, cloud.shape[1])
        near, index = torch.topk(distance, count, dim=1, largest=False)
        # Masked, not dropped. A seed whose neighbourhood is emptier than `neighbours` must read as
        # sparse; taking the nearest `count` regardless would quietly reach across a gap and describe
        # a different object's surface as this one's.
        inside = near <= (self.config.radius_mm / 1000.0)
        gathered = torch.gather(delta, 1, index.unsqueeze(-1).expand(-1, -1, 3)) * self.kappa
        neighbours = torch.gather(
            encoded[batch_index], 1,
            index.unsqueeze(-1).expand(-1, -1, encoded.shape[-1]))
        hidden = self.point(torch.cat([gathered, neighbours], dim=-1))
        # A very negative fill rather than zero: zero is a legal feature value, and max-pooling over
        # it would let an empty neighbourhood outvote a real but negative one.
        hidden = hidden.masked_fill(~inside.unsqueeze(-1), torch.finfo(hidden.dtype).min)
        pooled = hidden.max(dim=1).values
        # A seed with no neighbour inside the radius pools the fill value; that is not a feature, it
        # is the absence of one, and it becomes zero rather than the dtype's minimum.
        empty = ~inside.any(dim=1, keepdim=True)
        pooled = torch.where(empty, torch.zeros_like(pooled), pooled)
        return self.out(pooled)

    def extra_repr(self) -> str:
        return (f"radius_mm={self.config.radius_mm}, neighbours={self.config.neighbours}, "
                f"kappa={self.kappa:.2f}")
