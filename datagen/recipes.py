"""The corpus half of a named recipe: the settings a customer's dataset needs to be worth training on.

The training half lives in `src`, and the two may not import each other: `src` never imports
`datagen`, by rule. So a recipe version exists twice, once on each side, and the two sides must know
the same version names. They are one recipe: `v1` here and `v1` there are the corpus and the training
settings of the same bundle.

This exists rather than better defaults because every default in `DatagenConfig` is what the corpora
already on disk were built with, and moving one would silently change what a rebuild produces. A
recipe is a named bundle a customer asks for once, and the reasons travel with it.

A recipe is frozen once it ships. A better bundle becomes `v2` beside `v1`.
"""

from __future__ import annotations

from typing import Any, Final

#: Config overrides, by recipe. Only keys a measurement or a stated product decision justifies.
CORPUS_RECIPES: Final[dict[str, dict[str, Any]]] = {
    "v1": {
        # The largest single lever in the corpus, and the default is the other value.
        #
        # An object placed upright on its own base sits in a stable equilibrium and a solver leaves
        # it there. A tipped object is several times as likely to admit a jaw grasp, so `random`
        # buys several times the jaw labels per scene. It costs the objects that never come to
        # rest, and the trade is a large net gain in graspable objects per scene.
        #
        # It compounds with the engine. MuJoCo is the engine that leaves objects upright, and MuJoCo
        # is what a customer without an NVIDIA box uses, so `upright` plus `mujoco` is the worst cell
        # of the matrix and is exactly what an unspecified config produces.
        #
        # A corpus of standing objects also does not look like a bin-picking cell, where parts are
        # dumped rather than placed.
        "families": {"flat_orientation": "random"},
    },
}

#: What the customer must fill in themselves, because nobody else can know it.
PLACEHOLDERS: Final[dict[str, dict[str, str]]] = {
    "v1": {
        # The screen step 1 produces. Without it the bank is filtered by a box proxy on the mesh
        # hull, and a jaw closes on a line through the object, not on its hull: the proxy calls
        # meshes graspable that the labeller finds nothing on, and refuses meshes that do have
        # grasps.
        "assets.jaw_screen_path": "assets/screens/mine.json",
        "assets.mesh_asset_ids_path": "assets/screens/mine.ids.json",
    },
}

#: The commands, in order, with the flags whose defaults are not what a customer wants.
#: Kept beside the config because half of a corpus recipe is flags on later steps, not config keys.
STEPS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "v1": (
        ("python -m datagen prepare-assets --collection custom --out assets/screens/mine.json",
         "screens which of YOUR parts a jaw can grasp at all, and which a cup can. Writes the file "
         "the config above points at"),
        ("python -m datagen decompose --config CONFIG",
         "convex decomposition up front. Skipping it does not fail; it makes the render pay the "
         "decomposition one mesh at a time inside its own loop"),
        ("python -m datagen build --config CONFIG",
         "renders the scenes. This is the long pole, hours per shard, and it is the only step that "
         "wants a GPU: `render.engine` picks isaac, mujoco or none, and mujoco needs no NVIDIA box. "
         "It resumes, so an interrupted build continues rather than restarting"),
        ("python -m datagen label-grasps --name NAME --density grid",
         "--density grid guarantees the top-down approach per closing axis, which uniform azimuth "
         "sampling reaches only by coincidence: the shipped corpus sits a median 61 degrees off "
         "vertical. NO --kinds here: this step always labels BOTH kinds and the flag does nothing, "
         "which the runbook used to get wrong"),
        ("python -m datagen build-cloud-corpus --name NAME --corpus-out CORPUS --kinds both",
         "one .npz per scene, and the only thing training reads. THIS is where --kinds matters: it "
         "defaults to `jaw`, so the suction labels the step before wrote would be dropped here. They "
         "do not train the pose heads (a cup has no closing axis) but they do train the WHERE stage: "
         "MEASURED on v5, they cut the share of training units with nothing to learn from from "
         "77.3 % to 29.8 %"),
    ),
}


def corpus_recipe(name: str) -> dict[str, Any]:
    """The config overrides for a named recipe, or a refusal that lists the ones that exist."""
    try:
        return {key: dict(value) if isinstance(value, dict) else value
                for key, value in CORPUS_RECIPES[name].items()}
    except KeyError:
        raise ValueError(
            f"unknown corpus recipe {name!r}; known: {', '.join(sorted(CORPUS_RECIPES))}. A recipe "
            "is frozen once it ships, so a name that does not exist here never existed."
        ) from None


def config_for(name: str) -> dict[str, Any]:
    """A ready-to-edit config body: the recipe's overrides plus the paths only the customer knows.

    Deliberately minimal rather than a full dump of every default. A config file that repeats two
    hundred defaults hides the five lines that were chosen, and a customer editing it cannot tell
    which of them matter.
    """
    body = corpus_recipe(name)
    for dotted, value in PLACEHOLDERS.get(name, {}).items():
        section, key = dotted.split(".", 1)
        body.setdefault(section, {})[key] = value
    return body


def steps(name: str) -> tuple[tuple[str, str], ...]:
    """The command sequence for a recipe, each with the reason its flags are not the defaults."""
    if name not in STEPS:
        raise ValueError(f"unknown corpus recipe {name!r}; known: {', '.join(sorted(STEPS))}")
    return STEPS[name]
