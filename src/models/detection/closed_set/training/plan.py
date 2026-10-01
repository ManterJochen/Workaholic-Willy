"""The run description of a detector training, and the one place it is assembled from a recipe, a tier and the
caller's own choices.

Explicit wins over a recipe by construction: :class:`DetectorPlanOverrides` holds only what the caller chose, every
field defaulting to ``UNSET``, and :func:`build_plan` fills from the recipe and the tier only what is still unset. The
defaults stay on :class:`DetectorPlan`, the object the manifest quotes. Nothing here imports torch.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from src.contracts.options import UNSET, _Unset, chosen
from src.models.detection.closed_set.training.recipes import DEFAULT_BASE_MODEL, settings

__all__ = ["DEFAULT_SEED", "DetectorPlan", "DetectorPlanOverrides", "MULTISCALE_SIZES", "build_plan"]

DEFAULT_SEED = 20260716
#: RT-DETR's multi-scale training sizes in pixels; 640 three times, as its own list weights it.
MULTISCALE_SIZES = (480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800)
_AMP = ("auto", "bf16", "fp16", "off")


@dataclass(frozen=True, slots=True)
class DetectorPlan:
    """Everything a run is, before it runs. The defaults are recipe ``v1`` at the ``full`` tier."""

    base_model: str = DEFAULT_BASE_MODEL
    image_size: int = 640
    epochs: int = 50
    batch: int = 8
    #: Optimizer steps every ``accumulate`` batches: an effective batch of ``batch * accumulate``.
    accumulate: int = 1
    learning_rate: float = 1e-4
    backbone_lr_scale: float = 0.1
    weight_decay: float = 1e-4
    clip_grad_norm: float = 0.1
    #: "auto" is bf16 where the GPU has it, else fp16 with a gradient scaler; a CPU always runs fp32.
    amp: str = "auto"
    ema: bool = True
    ema_decay: float = 0.9999
    ema_warmup: int = 2000
    augment: bool = True
    multiscale: bool = True
    #: The last epochs run without the strong augmentations and multi-scale; None is a tenth of the epochs, at least 1.
    no_aug_epochs: int | None = None
    #: Stop after this many epochs without a better validation mAP; 0 never stops early.
    patience: int = 15
    #: The validation share cut from a dataset that brings no validation split of its own.
    val_fraction: float = 0.15
    seed: int = DEFAULT_SEED
    #: Data-loader worker processes; None chooses from the dataset size and the machine.
    workers: int | None = None
    #: Train on at most this many images (the smoke tier); None is every training image.
    max_train_images: int | None = None
    recipe: str | None = None
    tier: str | None = None

    def stop_augment_epoch(self) -> int:
        """The first epoch (counted from 0) that runs without the strong augmentations and multi-scale."""
        if not (self.augment or self.multiscale):
            return 0
        tail = self.no_aug_epochs if self.no_aug_epochs is not None else max(1, round(self.epochs / 10))
        return max(0, self.epochs - tail) if self.epochs > 1 else self.epochs

    def validated(self) -> "DetectorPlan":
        """This plan, or a ``ValueError`` naming the first setting a run cannot proceed with."""
        checks = [
            (self.epochs >= 1, f"epochs must be at least 1, not {self.epochs}"),
            (self.batch >= 1, f"batch must be at least 1, not {self.batch}"),
            (self.accumulate >= 1, f"accumulate must be at least 1, not {self.accumulate}"),
            (self.image_size >= 64 and self.image_size % 32 == 0,
             f"image_size must be a multiple of 32 and at least 64, not {self.image_size}"),
            (self.learning_rate > 0.0, f"learning_rate must be positive, not {self.learning_rate}"),
            (self.backbone_lr_scale >= 0.0, f"backbone_lr_scale cannot be negative ({self.backbone_lr_scale})"),
            (self.weight_decay >= 0.0, f"weight_decay cannot be negative ({self.weight_decay})"),
            (self.clip_grad_norm >= 0.0, f"clip_grad_norm cannot be negative ({self.clip_grad_norm}); 0 turns it off"),
            (self.amp in _AMP, f"unknown amp {self.amp!r}; choose from {', '.join(_AMP)}"),
            (0.0 < self.ema_decay < 1.0, f"ema_decay must lie between 0 and 1, not {self.ema_decay}"),
            (self.ema_warmup >= 1, f"ema_warmup must be at least 1, not {self.ema_warmup}"),
            (self.no_aug_epochs is None or self.no_aug_epochs >= 0,
             f"no_aug_epochs cannot be negative ({self.no_aug_epochs})"),
            (self.patience >= 0, f"patience cannot be negative ({self.patience}); 0 never stops early"),
            (0.0 <= self.val_fraction < 1.0, f"val_fraction must be at least 0 and below 1, not {self.val_fraction}"),
            (self.workers is None or self.workers >= 0, f"workers cannot be negative ({self.workers})"),
            (self.max_train_images is None or self.max_train_images >= 1,
             f"max_train_images must be at least 1, not {self.max_train_images}"),
        ]
        for ok, message in checks:
            if not ok:
                raise ValueError(message)
        return self

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class DetectorPlanOverrides:
    """What the caller chose explicitly. Every field defaults to ``UNSET``, which a recipe or tier may fill."""

    base_model: str | _Unset = UNSET
    image_size: int | _Unset = UNSET
    epochs: int | _Unset = UNSET
    batch: int | _Unset = UNSET
    accumulate: int | _Unset = UNSET
    learning_rate: float | _Unset = UNSET
    backbone_lr_scale: float | _Unset = UNSET
    weight_decay: float | _Unset = UNSET
    clip_grad_norm: float | _Unset = UNSET
    amp: str | _Unset = UNSET
    ema: bool | _Unset = UNSET
    ema_decay: float | _Unset = UNSET
    ema_warmup: int | _Unset = UNSET
    augment: bool | _Unset = UNSET
    multiscale: bool | _Unset = UNSET
    no_aug_epochs: int | None | _Unset = UNSET
    patience: int | _Unset = UNSET
    val_fraction: float | _Unset = UNSET
    seed: int | _Unset = UNSET
    workers: int | None | _Unset = UNSET
    max_train_images: int | None | _Unset = UNSET

    def forwarded(self) -> dict[str, Any]:
        """Only the fields the caller actually set."""
        return {field.name: value for field in dataclasses.fields(self)
                if chosen(value := getattr(self, field.name))}


def build_plan(*, recipe: str | None = None, tier: str | None = None,
               overrides: DetectorPlanOverrides | None = None,
               base: DetectorPlan | None = None) -> tuple[DetectorPlan, dict[str, Any]]:
    """The plan a run will use, and a record of what the recipe and tier supplied.

    Returns ``(plan, applied)``: ``applied`` maps each setting the recipe or tier filled to its value, and its key
    ``"overridden"`` lists the settings the caller had already chosen. Raises ``ValueError`` for an unknown recipe or
    tier and for a setting no run can proceed with; never ``SystemExit``.
    """
    settled = dict((overrides or DetectorPlanOverrides()).forwarded())
    applied: dict[str, Any] = {}
    overridden: list[str] = []
    for key, value in settings(recipe, tier).items():
        if key in settled:
            overridden.append(f"{key}={settled[key]}")
            continue
        settled[key] = value
        applied[key] = value
    applied["overridden"] = overridden
    plan = dataclasses.replace(base or DetectorPlan(), **settled, recipe=recipe, tier=tier)
    return plan.validated(), applied
