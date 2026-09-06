"""Stage 4b: a head that models the multimodality instead of quotienting it out.

A supervised point usually reaches more than one label, and often admits several approaches far
enough apart to be distinct grasps rather than one grasp measured twice.

The deterministic head answers that with K slots and a one-to-one matching, which is a bounded
answer: K slots bind at most K of a seed's labels, so its coverage cannot reach 1.0 wherever a seed
carries more, and that bound is a property of K against the data rather than of the model. A
generative head has no such bound. It is asked for a distribution and sampled as many times as a
caller wants.

The rule from the plan: either model the multimodality or quotient it out; regressing a single
value against a symmetric target is the one thing that cannot work. The director representation
quotients out the sign. Nothing else models the several genuinely different grasps a point admits,
and outside work reports the rotation representation becoming nearly irrelevant once a generative
head carries the bimodality natively.

How it works: a denoising head over the pose, conditioned on the seed's feature and the gripper.
Given a noised grasp and a timestep it predicts the noise, and sampling walks that backwards.

Translation and rotation run on separate schedules, which outside work measured as better than one
and which the units make obvious: an offset is metres and an axis is a direction, so one noise
level cannot be right for both. The two are noised independently and the head is told both levels.

It is an arm, not a replacement, and the comparison is the point: it runs against the deterministic
head on identical seeds, identical crops and identical folds. The plan's risk table states the
falsifier: if it beats 4a on seen assets and loses on unseen, it is memorising, and the response is
to keep 4a and wait for a bigger asset bank. A small asset bank against a large parameter count is
the regime that warning is about, and why the probe is an acceptance criterion.

Nothing here is measured. This file has trained nothing, and until an arm runs beside 4a on the
same folds, the only claim it supports is that the shape is buildable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import nn

from src.robot.grasping.deep.net.gripper import GRIPPER_VECTOR_DIM, GripperEncoder

__all__ = ["GenerativeHeadConfig", "GenerativeGraspHead", "POSE_DIM"]

#: What one pose carries while it is being denoised: an approach (3), a closing axis (3), an offset
#: (3) and a width (1).
#:
#: The axis travels as a plain vector here, not as a director. The director exists to make a loss
#: sign-invariant, and this head has no such loss: it predicts noise. Sign invariance falls out of
#: the sampling instead, because a distribution that covers both signs is what a generative head is
#: for. Forcing a quotient here would remove the thing being tested.
POSE_DIM: Final[int] = 10

#: Diffusion steps. Small on purpose: this is an arm to be compared, and a head whose sampling costs
#: a thousand forward passes per seed cannot be run beside a deterministic one on the same budget.
#: Measured elsewhere in this repository, the deterministic head's whole forward is 0.1 s per scene.
_DEFAULT_STEPS: Final[int] = 32


@dataclass(frozen=True, slots=True)
class GenerativeHeadConfig:
    """Every shape the generative head has, in one place a model card can quote."""

    width: int = 256
    depth: int = 4
    gripper_width: int = 32
    #: Timestep embedding width. The head must know how noisy its input is, and separately for the
    #: translation and the rotation; see the module docstring on why one schedule is not enough.
    time_width: int = 32
    steps: int = _DEFAULT_STEPS
    #: How many samples a caller draws per seed by default. The deterministic head emits K = 4; this
    #: matches it so a comparison is like for like rather than a budget difference wearing the name
    #: of a modelling difference.
    samples: int = 4


def _timestep_embedding(steps: torch.Tensor, width: int) -> torch.Tensor:
    """Sinusoidal embedding of a noise level, the standard one, written out rather than imported.

    A learned embedding would also work and is not used, because a sinusoidal one has no parameters
    and therefore cannot be the thing that differs between two arms.
    """
    half = width // 2
    frequencies = torch.exp(
        -torch.arange(half, device=steps.device, dtype=torch.float32)
        * (torch.log(torch.tensor(10000.0, device=steps.device)) / max(half - 1, 1)))
    angles = steps.to(torch.float32).unsqueeze(-1) * frequencies.unsqueeze(0)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def _draw(shape: tuple[int, ...], *, device: torch.device, kind: str = "randn",
          high: int = 0, generator: torch.Generator | None = None) -> torch.Tensor:
    """Random numbers from `generator`, on `device`, whichever device the generator lives on.

    A generator has a device and torch refuses a mismatch. The training loop seeds one CPU
    generator and uses it for every draw in the run while the tensors are on CUDA, and passing the
    two together raises "Expected a 'cuda' device type for generator but found 'cpu'".

    Drawing on the generator's device rather than seeding a CUDA generator keeps the noise identical
    across GPU models and driver versions, so two machines running the same seed produce the same
    run.
    """
    source = torch.device("cpu") if generator is None else generator.device
    if kind == "randn":
        out = torch.randn(shape, device=source, generator=generator)
    else:
        out = torch.randint(0, high, shape, device=source, generator=generator)
    return out.to(device)


class GenerativeGraspHead(nn.Module):
    """`(N, F)` seed features plus a noised pose and two noise levels, to the noise it predicts.

    Seeds arrive flattened, `N = batch x seeds`, the same shape the deterministic head consumes, so a
    comparison does not have to reshape one of the two and risk reshaping it differently.
    """

    def __init__(self, in_features: int, config: GenerativeHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or GenerativeHeadConfig()
        if in_features < 1:
            raise ValueError(f"in_features must be >= 1, got {in_features}")
        if self.config.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.config.steps}")
        self.in_features = in_features
        self.gripper = GripperEncoder(width=self.config.gripper_width)
        # Two embeddings, because the translation and the rotation are noised on separate schedules
        # and the head is told both levels rather than an average of them.
        self.time_translation = nn.Linear(self.config.time_width, self.config.width)
        self.time_rotation = nn.Linear(self.config.time_width, self.config.width)
        width = self.config.width
        self.stem = nn.Linear(in_features + self.config.gripper_width + POSE_DIM, width)
        layers: list[nn.Module] = []
        for _ in range(self.config.depth):
            layers += [nn.Linear(width, width), nn.SiLU()]
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(width, POSE_DIM)

    def forward(self, features: torch.Tensor, gripper: torch.Tensor, noised: torch.Tensor,
                step_translation: torch.Tensor, step_rotation: torch.Tensor) -> torch.Tensor:
        """The noise this head believes is in `noised`, shaped like the pose."""
        if features.dim() != 2 or features.shape[-1] != self.in_features:
            raise ValueError(f"features must be (N, {self.in_features}), "
                             f"got {tuple(features.shape)}")
        if gripper.shape != (features.shape[0], GRIPPER_VECTOR_DIM):
            raise ValueError(f"gripper must be {(features.shape[0], GRIPPER_VECTOR_DIM)}, "
                             f"got {tuple(gripper.shape)}")
        if noised.shape != (features.shape[0], POSE_DIM):
            raise ValueError(f"noised pose must be {(features.shape[0], POSE_DIM)}, "
                             f"got {tuple(noised.shape)}")
        hidden = self.stem(torch.cat([features, self.gripper(gripper), noised], dim=-1))
        # Both levels are added, not concatenated, so a head cannot learn to read one and ignore
        # the other by giving it a zero column. Added, they are indistinguishable in shape and both
        # must be used to recover either.
        hidden = hidden + self.time_translation(
            _timestep_embedding(step_translation, self.config.time_width))
        hidden = hidden + self.time_rotation(
            _timestep_embedding(step_rotation, self.config.time_width))
        return self.out(self.trunk(hidden))

    @torch.no_grad()
    def sample(self, features: torch.Tensor, gripper: torch.Tensor, *, count: int | None = None,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """`(N, S, POSE_DIM)` poses drawn from the head, S samples per seed.

        `count` is a caller's budget rather than an architectural bound: this is the only head in
        the package that can answer with more than a fixed number of modes, and a seed admits
        several distinct approaches.

        The schedule is the simplest one that can work. Sampling quality is a training result, and
        this file has trained nothing.
        """
        count = count or self.config.samples
        seeds = features.shape[0]
        device = features.device
        wide_features = features.repeat_interleave(count, dim=0)
        wide_gripper = gripper.repeat_interleave(count, dim=0)
        pose = _draw((seeds * count, POSE_DIM), device=device, generator=generator)
        for step in reversed(range(self.config.steps)):
            level = torch.full((seeds * count,), float(step), device=device)
            noise = self(wide_features, wide_gripper, pose, level, level)
            alpha = (step + 1) / self.config.steps
            pose = pose - noise * (1.0 / self.config.steps) / max(alpha, 1e-3)
        return pose.view(seeds, count, POSE_DIM)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
