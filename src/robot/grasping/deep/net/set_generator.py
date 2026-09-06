"""The whole generator: encode the scene, say where to grasp, then say how, K times per place.

The composition of everything under it, and deliberately thin: this only wires the parts, so a
defect here is a wiring defect.

    (B, N, 3) points and (B, N, C) features
        backbone      gives (B, N, W) per-point features
        graspability  gives (B, N) logits: where a grasp is worth attempting
        gather        takes the seeds, a list of (b, n) pairs, to (S, W)
        head          turns (S, W) into K grasps per seed

The seeds are a flat `(batch, point)` pair list, not a `(B, S)` block. How many seeds a sample
yields depends on how many object points it has, and the draw is capped by what is available: a
scene with five candidate points produces five seeds, not sixty padded ones. A padded block would
mean a mask threaded through the head, the loss and both metrics, and one of them eventually reading
it differently; a flat list has no padding to disagree about.

Full resolution throughout. The graspability field is per point and the seeds are chosen from it, so
an encoder that pooled the cloud down would have to interpolate back up and the seeds would read a
smoothed version of the geometry they were selected for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from src.robot.grasping.deep.net.generative_head import (
    GenerativeGraspHead,
    GenerativeHeadConfig,
)
from src.robot.grasping.deep.net.local_crop import LocalCrop, LocalCropConfig
from src.robot.grasping.deep.net.slot_head import SlotGraspHead, SlotHeadConfig
from src.robot.grasping.deep.net.serialized_backbone import SerializedBackbone, SerializedConfig
from src.robot.grasping.deep.net.set_loss import GraspSetPrediction

__all__ = ["SetGenerator", "SetGeneratorConfig"]

# Names in this package are checked against the licence boundary as prose, not as imports: a class
# name whose lowercased form contains a forbidden project's name is refused even when nothing was
# borrowed from that project, because a reader cannot tell a coincidence from a citation.


@dataclass(frozen=True, slots=True)
class SetGeneratorConfig:
    """Every shape the generator has, in one place a model card can quote."""

    backbone: SerializedConfig = field(default_factory=SerializedConfig)
    head: SlotHeadConfig = field(default_factory=SlotHeadConfig)
    #: Hidden width of the graspability head. Small: it answers one number per point, which is a
    #: much easier question than the pose.
    graspability_width: int = 64
    #: `None` is off, and off is the default. Without the crop the head reads one gathered per-point
    #: feature and nothing about the neighbourhood that feature sits in; the architecture plan's
    #: section 3.3 describes a ball crop, a seed-local frame and a measured kappa. Whether closing
    #: that gap helps is unmeasured, so a run that enables the crop says so in its card.
    crop: LocalCropConfig | None = None
    #: `None` is off. When set, the K-slot head is replaced rather than joined: the two are
    #: alternative answers to the same question, and a net carrying both would train one on
    #: gradients the other produced. The comparison is `K = 4` against an arbitrary sample count,
    #: and it is only a comparison if exactly one of them answers.
    generative: GenerativeHeadConfig | None = None


class SetGenerator(nn.Module):
    """Backbone, graspability field, and the K-slot local head, as one module."""

    def __init__(self, config: SetGeneratorConfig | None = None) -> None:
        super().__init__()
        self.config = config or SetGeneratorConfig()
        self.backbone = SerializedBackbone(self.config.backbone)
        width = self.backbone.out_features
        self.graspability = nn.Sequential(
            nn.Linear(width, self.config.graspability_width), nn.ReLU(inplace=True),
            nn.Linear(self.config.graspability_width, 1),
        )
        # The head's input width grows with the crop, so an artifact trained with one cannot be
        # loaded into a net built without it: the state dict does not fit.
        self.crop = LocalCrop(width, self.config.crop) if self.config.crop is not None else None
        head_width = width + (self.config.crop.width if self.config.crop is not None else 0)
        # Exactly one head. `head` stays built so every shape check, artifact reader and decode
        # keeps working; `generative` is what `propose` uses, and the training step asserts that
        # only one of the two receives a gradient.
        self.head = SlotGraspHead(head_width, self.config.head)
        self.generative = (GenerativeGraspHead(head_width, self.config.generative)
                           if self.config.generative is not None else None)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode(self, points_m: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """`(B, N, 3)` and `(B, N, C)` to `(B, N, W)` per-point features."""
        return self.backbone(points_m, features)

    def graspability_logit(self, encoded: torch.Tensor) -> torch.Tensor:
        """`(B, N, W)` to `(B, N)`. Where a grasp is worth attempting, before any pose is proposed."""
        return self.graspability(encoded).squeeze(-1)

    def propose(self, encoded: torch.Tensor, batch_index: torch.Tensor,
                point_index: torch.Tensor, gripper: torch.Tensor,
                points_m: torch.Tensor | None = None) -> GraspSetPrediction:
        """`K` grasps at each `(batch_index, point_index)` seed.

        The gripper vector is per seed. The caller expands it, here as everywhere, because a silent
        broadcast is how an input that should vary per sample becomes one that never does.
        """
        if batch_index.shape != point_index.shape:
            raise ValueError(f"batch_index {tuple(batch_index.shape)} and point_index "
                             f"{tuple(point_index.shape)} must describe the same seeds")
        if batch_index.dim() != 1:
            raise ValueError(f"seeds are a flat list, got {tuple(batch_index.shape)}")
        features = self.seed_features(encoded, batch_index, point_index, points_m)
        if self.generative is not None:
            # A generative artifact must not be served by the K-slot head. Both heads are built,
            # but in a generative run only `self.generative` receives a gradient, so `self.head` is
            # still at its initialisation. Returning it here would serve random weights under the
            # learned generator's name, and the failure is silent: the shapes are right and the
            # poses look like poses.
            #
            # `count` is the slot count so both families answer with the same number of candidates
            # and a comparison is not decided by a budget. The samples are reshaped into the exact
            # prediction the deterministic decode reads.
            from src.robot.grasping.deep.net.generative_loss import (  # noqa: PLC0415
                samples_as_prediction,
            )

            drawn = self.generative.sample(features, gripper, count=self.config.head.slots)
            return samples_as_prediction(drawn)
        return self.head(features, gripper)

    def seed_features(self, encoded: torch.Tensor, batch_index: torch.Tensor,
                      point_index: torch.Tensor,
                      points_m: torch.Tensor | None = None) -> torch.Tensor:
        """What a head is handed, per seed. Shared so the two head families read the same input:
        an arm whose halves see different features measures the features."""
        seed_features = encoded[batch_index, point_index]
        if self.crop is not None:
            if points_m is None:
                raise ValueError(
                    "this net was built with stage 3 enabled, so `propose` needs the cloud it is "
                    "cropping. Pass `points_m`, or build the net with `crop=None`.")
            seed_features = torch.cat(
                [seed_features, self.crop(points_m, encoded, batch_index, point_index)], dim=-1)
        return seed_features

    def forward(self, points_m: torch.Tensor, features: torch.Tensor,
                batch_index: torch.Tensor, point_index: torch.Tensor,
                gripper: torch.Tensor) -> tuple[torch.Tensor, GraspSetPrediction]:
        """Both stages in one call: the field over every point, and the grasps at the seeds."""
        encoded = self.encode(points_m, features)
        return self.graspability_logit(encoded), self.propose(
            encoded, batch_index, point_index, gripper, points_m)

    def extra_repr(self) -> str:
        return f"parameters={self.parameter_count:,}, slots={self.config.head.slots}"
