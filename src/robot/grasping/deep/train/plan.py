"""Assembling a `SetTrainingPlan`: the one place a run description is built, and the only one.

Two rules make a plan wrong or unbuildable if they are not kept in one place:

  * The order. `plan_with_slots` rebuilds `plan.model` and `plan.model.head`, so applying
    `slot_mixing`, `axis_mode` or `crop` before it discards them silently, and the run then reports
    the setting it was asked for without running it. The order lives here, once, so no code caller
    has to know it.

  * Chosen is data, not `sys.argv`. A recipe is applied only to settings the caller did not choose.
    Deciding that from `sys.argv` is wrong outside a shell: in a hosted process the arguments
    belong to the host, `--epochs` never appears, and the recipe then overrides a value the caller
    passed explicitly. `PlanOverrides` states it as data, so the rule holds in both worlds.

Default values do not live here. They stay on `SetTrainingPlan`, which is the object a model card
quotes. `UNSET` means "not chosen", never "zero" or "false": `--refit` is a store_true whose absence
must stay distinguishable from an explicit `refit=False`, or `--tier smoke` could not turn it off.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.contracts.options import UNSET, _Unset, chosen
from src.robot.grasping.deep.corpus.sample import SampleSpec

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.grasping.deep.train.trainer import SetTrainingPlan

__all__ = ["UNSET", "PlanOverrides", "build_plan"]

# Importing this module must not cost torch. `SetTrainingPlan` lives in `train.trainer`, which
# imports torch at module level, so a module-level import here would drag in 1,558 modules and a
# CUDA probe just to construct a `PlanOverrides` or read the `UNSET` sentinel. Measured: this
# module imports torch-free at 770 modules against 1,558 with torch.
#
# The cost is paid by `build_plan` and only by it. That function needs the real defaults and the
# enum vocabularies, which do live behind torch, so calling it loads torch. What stays free is
# everything a caller does before deciding to train: build an overrides object, hand it around,
# check whether a field was chosen.


# The sentinel comes from `contracts/options.py`, it is not declared here. `x is UNSET` compares
# identity, so a second copy of the object is a second answer to the same question.
#
# Why a class rather than `None`: `None` is a legitimate value for several fields below.
# `train_units=None` means every unit, `control=None` means the corpus labels,
# `labelled_unit_share=None` means no rebalancing. Using `None` for both would make "train on
# everything" indistinguishable from "not chosen", and a recipe merge would then overwrite a choice
# the caller made.


@dataclass(frozen=True, slots=True)
class PlanOverrides:
    """The generator training settings you choose explicitly; they outrank the recipe and the tier. Every field you
    leave out stays ``UNSET``, which the recipe or the tier fills; the defaults named are the plan's own.

    ```python
    overrides = PlanOverrides(epochs=24, folds=5, run_folds=5, refit=True)
    ```

    Attributes:
        epochs (int): Epochs per fold; default 12 (default: UNSET).
        folds (int): Asset-disjoint folds the corpus is cut into; default 5 (default: UNSET).
        run_folds (int): How many of them actually run: 1 compares two settings on the same held-out assets, 5 earns a
            number worth quoting; default 1 (default: UNSET).
        batch (int): Samples per batch; the points per sample, not the parameters, are the memory ceiling; default 4
            (default: UNSET).
        learning_rate (float): The learning rate; default 1e-3 (default: UNSET).
        weight_decay (float): The weight decay; default 1e-4 (default: UNSET).
        seed (int): The run's seed; default 0 (default: UNSET).
        train_units (int | None): Units per epoch, ``None`` for all; a cap keeps epochs comparable across corpora;
            default 4000 (default: UNSET).
        eval_units (int): Held-out units each evaluation scores; default 512 (default: UNSET).
        points (int): Points per sample cloud; default 8192 (default: UNSET).
        refit (bool): After the folds, train one more net on every unit and ship that as the artifact; default False
            (default: UNSET).
        labelled_unit_share (float | None): The share of an epoch's units that carry a grasp label; ``None`` the natural
            mix; default 0.5 (default: UNSET).
        collapse_floor_deg (float): Stop a fold whose head proposes one direction for every seed, held-out seed spread
            below this many degrees from epoch three on; 0 is off; default 0.0 (default: UNSET).
        control (str | None): Replace the labels with a target derived from an input channel, ``"normal"`` or
            ``"local"``: a control run, whose numbers are not grasping numbers; ``None`` a real run (default: UNSET).
        target (str): Which direction the head predicts: ``"approach"`` or ``"axis"`` (the closing axis); default
            ``"approach"`` (default: UNSET).
        axis_mode (str): How the closing axis is written: ``"director"`` (six parameters, an eigendecode) or
            ``"vector"``; default ``"director"`` (default: UNSET).
        slots (int): Grasps predicted per seed; default 4 (default: UNSET).
        slot_mixing (str): How a slot may differ from its neighbour: ``"affine"``, ``"film"`` or ``"mlp"``; default
            ``"affine"`` (default: UNSET).
        part_roles (bool): An affordance output, one role per grasp (empty, body, handle, grip, neck, head); default
            False (default: UNSET).
        generative (bool): A denoising head trained on every label instead of the slot head; default False
            (default: UNSET).
        crop_mm (float): A ball of points around each seed, in its own frame, of this radius in millimetres (40 to 120);
            0 is off; default 0 (default: UNSET).
        crop_neighbours (int): Points kept per crop; default 32 (default: UNSET).
        backbone_width (int): The encoder's channel width; default 384, the 21.5M-parameter model (default: UNSET).
        backbone_depth (int): The encoder's block count; default 12 (default: UNSET).
        backbone_heads (int): Attention heads, which must divide the width; default 6 (default: UNSET).
    """

    epochs: int | _Unset = UNSET
    folds: int | _Unset = UNSET
    run_folds: int | _Unset = UNSET
    batch: int | _Unset = UNSET
    learning_rate: float | _Unset = UNSET
    weight_decay: float | _Unset = UNSET
    seed: int | _Unset = UNSET
    train_units: int | None | _Unset = UNSET
    eval_units: int | _Unset = UNSET
    points: int | _Unset = UNSET
    refit: bool | _Unset = UNSET
    labelled_unit_share: float | None | _Unset = UNSET
    collapse_floor_deg: float | _Unset = UNSET
    control: str | None | _Unset = UNSET
    target: str | _Unset = UNSET
    axis_mode: str | _Unset = UNSET
    slots: int | _Unset = UNSET
    slot_mixing: str | _Unset = UNSET
    part_roles: bool | _Unset = UNSET
    generative: bool | _Unset = UNSET
    crop_mm: float | _Unset = UNSET
    crop_neighbours: int | _Unset = UNSET
    backbone_width: int | _Unset = UNSET
    backbone_depth: int | _Unset = UNSET
    backbone_heads: int | _Unset = UNSET

    def forwarded(self) -> dict[str, Any]:
        """Only the settings you set.

        Returns:
            dict[str, Any]: Each set field by name; the ``UNSET`` ones are left out.
        """
        return {field.name: value for field in dataclasses.fields(self)
                if chosen(value := getattr(self, field.name))}


def _validated(name: str, value: str, allowed: "tuple[str, ...] | frozenset[str] | dict[str, Any]",
               what: str) -> str:
    """Refuse an unknown enum value here, naming the ones that exist.

    At construction, not at use. Without this the refusal arrives only after the corpus walk and
    the ceiling, floor and memorisation probes have run: minutes of work to learn that a string was
    misspelled.
    """
    if value not in allowed:
        raise ValueError(f"unknown {what} {value!r}; choose from {', '.join(sorted(allowed))}")
    return value


def build_plan(*, recipe: str | None = None, tier: str | None = None,
               overrides: PlanOverrides | None = None,
               base: SetTrainingPlan | None = None) -> tuple[SetTrainingPlan, dict[str, Any]]:
    """The plan a run will actually use, plus a record of what the recipe did.

    Returns `(plan, applied)` where `applied` maps each setting the recipe or tier supplied to its
    value, and carries the key `"overridden"` listing the settings the caller had already chosen.
    The CLI prints that; a library caller can log it or ignore it.

    Explicit wins over the recipe, by construction rather than by convention: `recipes.settings`
    fills only fields the caller left `UNSET`.

    Raises `ValueError` for anything the run cannot proceed with, and never `SystemExit`.
    """
    from src.robot.grasping.deep.net.local_crop import (  # noqa: PLC0415
        LocalCropConfig,
        kappa_for_radius,
    )
    from src.robot.grasping.deep.net.rotation import AXIS_MODES  # noqa: PLC0415
    from src.robot.grasping.deep.net.slot_head import SLOT_MIXING  # noqa: PLC0415
    from src.robot.grasping.deep.train.recipes import settings  # noqa: PLC0415
    from src.robot.grasping.deep.train.trainer import (  # noqa: PLC0415
        SetTrainingPlan,
        plan_with_slots,
    )
    from src.robot.grasping.deep.train.synthetic_control import (  # noqa: PLC0415
        CONTROL_LEVELS,
    )

    settled = dict((overrides or PlanOverrides()).forwarded())
    applied: dict[str, Any] = {}
    overridden: list[str] = []
    if recipe or tier:
        for key, value in settings(recipe, tier).items():
            if key in settled:
                overridden.append(f"{key}={settled[key]}")
                continue
            settled[key] = value
            applied[key] = value
    applied["overridden"] = overridden

    defaults = base or SetTrainingPlan()

    def pick(name: str, fallback: Any) -> Any:
        return settled.get(name, fallback)

    target = _validated("target", str(pick("target", defaults.step.target)),
                        ("approach", "axis"), "target")
    axis_mode = _validated("axis_mode", str(pick("axis_mode", defaults.model.head.axis_mode)),
                           AXIS_MODES, "axis mode")
    slot_mixing = _validated("slot_mixing",
                             str(pick("slot_mixing", defaults.model.head.slot_mixing)),
                             SLOT_MIXING, "slot mixing")
    control = pick("control", defaults.control)
    if control is not None:
        _validated("control", str(control), CONTROL_LEVELS, "control level")

    # The backbone check carries its suggestion. `serialized_backbone.py` raises the bare form with
    # no hint, and the architecture plan's own 6.5M variant (width 256) trips it.
    width = int(pick("backbone_width", defaults.model.backbone.width))
    heads = int(pick("backbone_heads", defaults.model.backbone.heads))
    depth = int(pick("backbone_depth", defaults.model.backbone.depth))
    if width % heads:
        options = [h for h in range(1, 17) if width % h == 0]
        raise ValueError(f"backbone width {width} is not divisible by {heads} heads; "
                         f"choose a head count from {options}")

    crop = None
    crop_mm = float(pick("crop_mm", 0.0))
    if crop_mm > 0.0:
        # Checked here so a typo fails now rather than after six hours against an unmeasured scale.
        kappa_for_radius(crop_mm)
        crop = LocalCropConfig(radius_mm=crop_mm, neighbours=int(pick("crop_neighbours", 32)))

    plan = dataclasses.replace(
        defaults,
        epochs=int(pick("epochs", defaults.epochs)),
        folds=int(pick("folds", defaults.folds)),
        run_folds=int(pick("run_folds", defaults.run_folds)),
        batch=int(pick("batch", defaults.batch)),
        learning_rate=float(pick("learning_rate", defaults.learning_rate)),
        weight_decay=float(pick("weight_decay", defaults.weight_decay)),
        seed=int(pick("seed", defaults.seed)),
        train_units=pick("train_units", defaults.train_units),
        eval_units=int(pick("eval_units", defaults.eval_units)),
        refit=bool(pick("refit", defaults.refit)),
        recipe=recipe,
        tier=tier,
        collapse_floor_deg=float(pick("collapse_floor_deg", defaults.collapse_floor_deg)),
        labelled_unit_share=pick("labelled_unit_share", defaults.labelled_unit_share),
        control=control,
        sample=dataclasses.replace(defaults.sample, grasp_set=True,
                                   points=int(pick("points", defaults.sample.points))),
        step=dataclasses.replace(defaults.step, target=target))

    # From here the order is load-bearing and is the reason this function exists.
    # `plan_with_slots` rebuilds `plan.model` and `plan.model.head` from scratch, so every head
    # setting has to be applied after it.
    if pick("part_roles", False):
        from src.robot.grasping.deep.corpus.sample import (  # noqa: PLC0415
            _PART_ROLE_INDEX,
        )
        roles = tuple(sorted(_PART_ROLE_INDEX, key=lambda name: _PART_ROLE_INDEX[name]))
        plan = dataclasses.replace(plan, model=dataclasses.replace(
            plan.model, head=dataclasses.replace(plan.model.head, part_roles=roles)))

    if "slots" in settled:
        plan = plan_with_slots(plan, int(settled["slots"]))

    backbone = dataclasses.replace(plan.model.backbone, width=width, depth=depth, heads=heads)
    plan = dataclasses.replace(plan, model=dataclasses.replace(plan.model, backbone=backbone))
    plan = dataclasses.replace(plan, model=dataclasses.replace(
        plan.model, head=dataclasses.replace(plan.model.head, axis_mode=axis_mode,
                                             slot_mixing=slot_mixing)))

    if pick("generative", False):
        from src.robot.grasping.deep.net.generative_head import (  # noqa: PLC0415
            GenerativeHeadConfig,
        )
        plan = dataclasses.replace(plan, model=dataclasses.replace(
            plan.model, generative=GenerativeHeadConfig()))

    if crop is not None:
        plan = dataclasses.replace(plan, model=dataclasses.replace(plan.model, crop=crop))

    return plan, applied


def sample_spec(points: int) -> SampleSpec:
    """The sample spec a set run needs. `grasp_set=True` is not optional and the plan refuses without
    it, so this exists to keep that fact in one place rather than in every caller."""
    return SampleSpec(grasp_set=True, points=points)
