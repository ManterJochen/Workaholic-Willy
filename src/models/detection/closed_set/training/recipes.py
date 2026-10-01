"""Named, versioned detector recipes and the two tiers a customer runs, the same idiom as the grasp generator's.

A recipe is frozen once it ships: ``v1`` always means exactly what it means today, and a better bundle becomes ``v2``
beside it, so a model can say which settings produced it and be rebuilt from them. Its values are the defaults of
:class:`~src.models.detection.closed_set.training.plan.DetectorPlan`, the RT-DETR recipe as its authors and the large
toolkits run it; naming them here is what pins them.
"""

from __future__ import annotations

from typing import Any, Final

__all__ = ["RECIPES", "TIERS", "recipe", "settings", "tier"]

DEFAULT_BASE_MODEL: Final[str] = "PekingU/rtdetr_r50vd"

#: Frozen. A change becomes a new version, never an edit to an old one.
RECIPES: Final[dict[str, dict[str, Any]]] = {
    "v1": {
        # RT-DETR R50-vd, COCO-pretrained, its classification head rebuilt for the dataset's classes.
        "base_model": DEFAULT_BASE_MODEL,
        "image_size": 640,
        # AdamW at 1e-4 with the backbone at a tenth, weight decay 1e-4 off norms and biases, gradients clipped at
        # 0.1: RT-DETR's own optimizer settings.
        "learning_rate": 1e-4,
        "backbone_lr_scale": 0.1,
        "weight_decay": 1e-4,
        "clip_grad_norm": 0.1,
        # An exponential moving average of the weights (decay 0.9999, warm-up 2000 updates), as RT-DETR and YOLO
        # keep it; every evaluation and every exported model is the average.
        "ema": True,
        "ema_decay": 0.9999,
        "ema_warmup": 2000,
        # The strong augmentations and multi-scale batches, off for the last tenth of the epochs.
        "augment": True,
        "multiscale": True,
    },
}

#: What each tier is for. ``smoke`` narrows the run to prove the chain; ``full`` is the model you deploy.
TIERS: Final[dict[str, dict[str, Any]]] = {
    "smoke": {"epochs": 2, "max_train_images": 64, "multiscale": False, "patience": 0,
              "why": "proves the chain closes on your dataset and your machine. It says NOTHING about detection "
                     "quality and is not a model to deploy"},
    "full": {"epochs": 50, "patience": 15,
             "why": "the model you deploy: up to 50 epochs, stopped after 15 without a better validation mAP, the "
                    "best epoch kept"},
}


def recipe(name: str) -> dict[str, Any]:
    """The settings of a named recipe, or a refusal naming the ones that exist."""
    try:
        return dict(RECIPES[name])
    except KeyError:
        raise ValueError(f"unknown detector recipe {name!r}; known: {', '.join(sorted(RECIPES))}. A recipe is frozen "
                         f"once it ships, so a name that does not exist here never existed.") from None


def tier(name: str) -> dict[str, Any]:
    """The settings of a named tier, or a refusal naming the ones that exist."""
    try:
        return {key: value for key, value in TIERS[name].items() if key != "why"}
    except KeyError:
        raise ValueError(f"unknown training tier {name!r}; known: {', '.join(sorted(TIERS))}") from None


def settings(recipe_name: str | None, tier_name: str | None) -> dict[str, Any]:
    """The recipe's settings with the tier's laid over them; the tier narrows what the recipe names."""
    merged: dict[str, Any] = {}
    if recipe_name:
        merged.update(recipe(recipe_name))
    if tier_name:
        merged.update(tier(tier_name))
    return merged
