"""Named, versioned training recipes: the proven settings, in one place, stamped into the artifact.

The defaults in `train-set` do not move. Every individual default is what the arms already measured
ran with, so changing one would make an earlier run irreproducible without explicit flags and leave
a reader of an old `epochs.json` unable to tell which world it came from. A recipe is a named
bundle a customer asks for by one flag, and it is written into the artifact, so a model can say
which recipe produced it.

A recipe is frozen once it ships. `v1` always means exactly what it means today; a better bundle
becomes `v2` beside it, so a customer who trained under `v1` can rebuild that model and a cell
serving those weights can say what they were.

A recipe does not contain an unsettled arm. `slot_mixing`, `axis_mode`, `--crop-mm` and
`--generative` are all being measured, and pinning one inside a recipe would hand a customer a coin
flip wearing a version number. Only settings that are measured, or that are a product decision with
its reasoning written down, belong here.

The other half of the recipe lives in `datagen`, because `src` may never import it. The two halves
carry the same version string and must agree.
"""

from __future__ import annotations

from typing import Any, Final

#: Frozen. See the module docstring: a change becomes a new version, never an edit to an old one.
RECIPES: Final[dict[str, dict[str, Any]]] = {
    "v1": {
        # The artifact must see every part the customer owns. Without this it is one fold's net:
        # the units of the folds that were not run never reach the weights a cell serves, and on a
        # customer's corpus that is a share of their own catalogue. The fold pass still earns the
        # numbers; this only decides which net ships. Unmeasured on this architecture: it is a
        # product decision, not a metric one, and it costs a second training run.
        "refit": True,
        # The cheapest instrument in the arc. A head that answers every seed with one direction is
        # invisible in every quality metric (they report it honestly as a low number without being
        # able to say why) until the poses are read hours downstream. One degree of held-out seed
        # spread, checked from epoch three.
        "collapse_floor_deg": 1.0,
    },
}

#: The two tiers a customer runs, and what each one is for. `full` narrows epochs alone; `smoke`
#: narrows three settings (`epochs=2`, `train_units=400`, `refit=False`), so it is a shorter run
#: over a smaller slice of the corpus with the refit off, not the same run stopped early.
TIERS: Final[dict[str, dict[str, Any]]] = {
    # What the fast tier proves: that the chain closes on their corpus and their box. It proves
    # nothing whatever about grasp quality, and the runbook says so in those words. Its value is
    # that a broken corpus surfaces in minutes instead of after a night.
    "smoke": {"epochs": 2, "train_units": 400, "refit": False,
              "why": "proves the chain closes on your corpus and your box. Says NOTHING about "
                     "grasp quality, and is not a model to deploy"},
    # 36 is a floor here, not a ceiling.
    #
    # Single points off a curve mislead: a metric that rises, falls and rises again reads as a peak
    # followed by overfitting at any one epoch, so it has to be read in windows of several epochs.
    #
    # Two traps sit around that. The report's "plateau" verdict compares against an absolute
    # `_REAL_DIFFERENCE`, so a curve whose whole span is inside that band cannot reach it by
    # arithmetic, and the report says so itself under "below resolution". And any arm compared
    # against this one at fewer epochs is a different experiment, not a worse architecture.
    #
    # A customer whose own curve is still climbing at 36 should pass `--epochs` and run more;
    # `deep report` prints their verdict on their own data, which is the instrument this number
    # cannot replace.
    "full": {"epochs": 36,
             "why": "the model you deploy. Hours on a single GPU. Thirty-six because the reference "
                    "arm was still improving at thirty, not because it is a round number"},

}


def recipe(name: str) -> dict[str, Any]:
    """The settings for a named recipe, or a refusal that lists the ones that exist.

    Refuses rather than falling back to the defaults: a customer who typed `v2` before it exists
    would otherwise get `v1`'s behaviour under `v2`'s name, which is the failure a version string is
    supposed to prevent.
    """
    try:
        return dict(RECIPES[name])
    except KeyError:
        raise ValueError(
            f"unknown training recipe {name!r}; known: {', '.join(sorted(RECIPES))}. A recipe is "
            "frozen once it ships, so a name that does not exist here never existed."
        ) from None


def tier(name: str) -> dict[str, Any]:
    """The settings for a named tier, or a refusal that lists the ones that exist."""
    try:
        chosen = dict(TIERS[name])
    except KeyError:
        raise ValueError(
            f"unknown tier {name!r}; known: {', '.join(sorted(TIERS))}"
        ) from None
    chosen.pop("why", None)
    return chosen


def settings(name: str | None, tier_name: str | None = None) -> dict[str, Any]:
    """The effective settings: the recipe, with the tier applied on top.

    The tier wins, and it has to. `smoke` sets `refit=False` precisely because a run that only
    proves the chain closes must not pay for a second training pass, and a recipe that could not be
    narrowed by a tier would make the fast tier slower than the thing it exists to precede.
    """
    merged: dict[str, Any] = {}
    if name:
        merged.update(recipe(name))
    if tier_name:
        merged.update(tier(tier_name))
    return merged


def describe(name: str | None, tier_name: str | None = None) -> str:
    """The settings that actually apply, on one ASCII line, for the run log.

    The merged result, never both layers. Printing the recipe's `refit=True` beside the tier's
    `refit=False` tells a reader two things and lets them pick the wrong one; the whole value of this
    line is that it says what the run will do.

    ASCII on purpose. This is printed, and this project's terminal is cp1252: a decorated marker
    turns a training run's first line into a UnicodeEncodeError.
    """
    label = " ".join(part for part in (f"recipe {name}" if name else "",
                                       f"tier {tier_name}" if tier_name else "") if part)
    body = " ".join(f"{key}={value}" for key, value in sorted(settings(name, tier_name).items()))
    return f"{label}: {body}" if body else f"{label}: (nothing set)"
