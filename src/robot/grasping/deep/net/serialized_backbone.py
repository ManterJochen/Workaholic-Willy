"""A point-cloud encoder that scales, written in plain PyTorch.

The strong published encoders lean on a sparse-convolution engine or a scatter extension, both of
which need a CUDA compiler on the machine that installs them. Users train this model on their own
cell, so a dependency needing a build toolchain there is a product defect. Everything here is
`torch` and nothing else. Torch 2.7 ships flash attention inside `scaled_dot_product_attention`,
which is the only fast kernel this design needs.

The idea is borrowed rather than invented. Attention over an unstructured cloud is quadratic and
hopeless at 8,192 points, so the cloud is given an order first: the points are sorted along a
space-filling curve, which puts sequence neighbours next to spatial neighbours, and attention runs
inside fixed windows of that sequence. Cost drops to `O(N x window)`, and the windows change shape
from block to block because the curve changes.

This is a simplification of the published design. That work alternates a Hilbert curve and a
transposed Hilbert curve; this alternates Z-order curves under permuted axes, which is a few lines
instead of a few hundred and keeps the property that matters here, namely that consecutive blocks
group points differently. Whether the difference costs anything is not measured, and no result from
this file may be read as a result about the published encoder.

The capacity measurement this repository already holds concludes that capacity is not the
constraint; the works this design borrows from run at tens of millions of parameters, so that
conclusion has never been tested in the regime the field lives in, and this encoder is what makes
testing it possible. A high-capacity model on a small dataset approximately memorises rather than
generalises, and the corpus's folds are cut over asset groups, which is why the memorisation probe
is an acceptance criterion rather than a footnote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 (the conventional alias)

__all__ = ["SerializedBackbone", "SerializedConfig", "morton_code", "AXIS_ORDERS"]

#: Bits per axis in the Morton code. Ten gives 1,024 cells per axis, so on a 600 mm workspace one cell
#: is 0.59 mm: finer than the 3 mm voxel the cloud is built at, which means the ordering never has to
#: break a tie between two distinct points. Three axes at ten bits is 30 bits, comfortably inside
#: int64.
_MORTON_BITS: Final[int] = 10

#: The axis permutations the blocks cycle through. Six orders from one curve: a Z-order under
#: `(x, y, z)` and one under `(y, z, x)` group the same cloud differently, which is the property the
#: published alternation exists for.
AXIS_ORDERS: Final[tuple[tuple[int, int, int], ...]] = (
    (0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (1, 0, 2), (2, 1, 0),
)


def _spread(value: torch.Tensor) -> torch.Tensor:
    """Insert two zero bits between every bit of a 10-bit integer. The Morton interleave, per axis."""
    value = (value | (value << 16)) & 0x030000FF
    value = (value | (value << 8)) & 0x0300F00F
    value = (value | (value << 4)) & 0x030C30C3
    value = (value | (value << 2)) & 0x09249249
    return value


def morton_code(points: torch.Tensor, order: tuple[int, int, int] = (0, 1, 2)) -> torch.Tensor:
    """`(B, N, 3)` positions to a `(B, N)` Z-order key, quantised per cloud.

    Per cloud, not per corpus: the key only has to order the points of one scene relative to each
    other, and a fixed global grid would waste most of its resolution on empty workspace. A degenerate
    cloud, every point at one place, produces an all-zero key and therefore the identity order, which
    is correct rather than an error.
    """
    if points.dim() != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must be (B, N, 3), got {tuple(points.shape)}")
    low = points.amin(dim=1, keepdim=True)
    span = (points.amax(dim=1, keepdim=True) - low).clamp_min(1e-6)
    scale = float((1 << _MORTON_BITS) - 1)
    cells = ((points - low) / span * scale).round().clamp(0, scale).to(torch.int64)
    first, second, third = (_spread(cells[..., axis]) for axis in order)
    return first | (second << 1) | (third << 2)


@dataclass(frozen=True, slots=True)
class SerializedConfig:
    """Every shape the encoder has, in one place a model card can quote."""

    #: Channels beyond xyz on each input point. Five today: normal (3), is_target, is_object. The
    #: authority is `corpus.sample.FEATURE_NAMES` and this count must match it.
    in_features: int = 5
    width: int = 384
    depth: int = 12
    heads: int = 6
    #: Points per attention window. 512 keeps one window's attention matrix at 262k entries, which the
    #: flash kernel handles without materialising it. Larger windows see further per block; the curve
    #: alternation is what gives reach across blocks, so this is a memory knob rather than a
    #: receptive-field one.
    window: int = 512
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    #: How positions reach the features. `"linear"` projects xyz once at the input. Cheap, and the
    #: serialisation already carries most of the locality: two points close in the sequence are close
    #: in space by construction.
    position: str = "linear"
    axis_orders: tuple[tuple[int, int, int], ...] = field(default=AXIS_ORDERS)


class _WindowAttention(nn.Module):
    """Multi-head self-attention inside fixed windows of an already-ordered sequence."""

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if width % heads:
            raise ValueError(f"width {width} is not divisible by heads {heads}")
        self.heads = heads
        self.qkv = nn.Linear(width, width * 3)
        self.project = nn.Linear(width, width)
        self.dropout = dropout

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """`(W, L, C)` windows and a `(W, L)` validity mask to `(W, L, C)`."""
        windows, length, width = features.shape
        qkv = self.qkv(features).reshape(windows, length, 3, self.heads, width // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        # A padded key must not be attended to. Padded queries are harmless: their outputs are
        # thrown away with the padding, and masking them too would give a row with no admissible key
        # at all, whose softmax is NaN.
        mask = valid[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0)
        return self.project(attended.transpose(1, 2).reshape(windows, length, width))


class _Block(nn.Module):
    """Pre-norm attention plus MLP, both residual. The standard transformer block, no surprises."""

    def __init__(self, config: SerializedConfig) -> None:
        super().__init__()
        hidden = int(config.width * config.mlp_ratio)
        self.norm_attention = nn.LayerNorm(config.width)
        self.attention = _WindowAttention(config.width, config.heads, config.dropout)
        self.norm_mlp = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, hidden), nn.GELU(),
            nn.Dropout(config.dropout) if config.dropout else nn.Identity(),
            nn.Linear(hidden, config.width),
        )

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        features = features + self.attention(self.norm_attention(features), valid)
        return features + self.mlp(self.norm_mlp(features))


class SerializedBackbone(nn.Module):
    """`(B, N, 3)` positions and `(B, N, C)` features to `(B, N, width)` per-point features.

    Per point and at full resolution, deliberately: the head runs at seeds chosen from a graspability
    field over every point, so an encoder that pooled the cloud down would have to interpolate back up
    and the seeds would read a smoothed version of the geometry they were chosen for.
    """

    def __init__(self, config: SerializedConfig | None = None) -> None:
        super().__init__()
        config = config or SerializedConfig()
        if config.depth < 1:
            raise ValueError(f"depth must be >= 1, got {config.depth}")
        if config.window < 1:
            raise ValueError(f"window must be >= 1, got {config.window}")
        if not config.axis_orders:
            raise ValueError("at least one axis order is needed")
        self.config = config
        self.embed = nn.Linear(config.in_features, config.width)
        self.position = nn.Linear(3, config.width) if config.position == "linear" else None
        self.blocks = nn.ModuleList(_Block(config) for _ in range(config.depth))
        self.norm = nn.LayerNorm(config.width)

    @property
    def out_features(self) -> int:
        return self.config.width

    def forward(self, points_m: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if points_m.dim() != 3 or points_m.shape[-1] != 3:
            raise ValueError(f"points must be (B, N, 3), got {tuple(points_m.shape)}")
        if features.shape[:2] != points_m.shape[:2]:
            raise ValueError(f"features {tuple(features.shape)} do not match points "
                             f"{tuple(points_m.shape)}")
        if features.shape[-1] != self.config.in_features:
            raise ValueError(f"expected {self.config.in_features} input features, "
                             f"got {features.shape[-1]}")

        hidden = self.embed(features)
        if self.position is not None:
            hidden = hidden + self.position(points_m)

        batch, count, width = hidden.shape
        window = min(self.config.window, count)
        padded = int(math.ceil(count / window)) * window
        for index, block in enumerate(self.blocks):
            order = self.config.axis_orders[index % len(self.config.axis_orders)]
            # Sort, run, then put back. The residual stream stays in the caller's point order the
            # whole time; only the block's own view is reordered. A backbone that returned a permuted
            # cloud would hand the head features belonging to other points, which is a defect no shape
            # check can see and every metric would blame on the head.
            sort = morton_code(points_m, order).argsort(dim=1)
            gathered = hidden.gather(1, sort.unsqueeze(-1).expand(batch, count, width))

            if padded != count:
                pad = padded - count
                gathered = F.pad(gathered, (0, 0, 0, pad))
                valid = F.pad(torch.ones(batch, count, dtype=torch.bool, device=hidden.device),
                              (0, pad))
            else:
                valid = torch.ones(batch, count, dtype=torch.bool, device=hidden.device)

            windows = gathered.reshape(batch * (padded // window), window, width)
            updated = block(windows, valid.reshape(batch * (padded // window), window))
            updated = updated.reshape(batch, padded, width)[:, :count]

            hidden = hidden.scatter(1, sort.unsqueeze(-1).expand(batch, count, width), updated)
        return self.norm(hidden)

    def extra_repr(self) -> str:
        parameters = sum(p.numel() for p in self.parameters())
        return (f"width={self.config.width}, depth={self.config.depth}, "
                f"heads={self.config.heads}, window={self.config.window}, "
                f"parameters={parameters:,}")
