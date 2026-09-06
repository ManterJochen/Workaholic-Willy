"""Seed to scene. Pure numpy, no Isaac, no meshes, so the expensive half never hides a cheap bug.

From a dataset seed and a scene index this fixes everything: which assets, where they spawn, how
they are oriented, where the cameras are, and every randomised light and colour value. Physics then
settles those spawn poses and the settled result is recorded separately, so a scene is reproducible
from `(seed, index, code version, physics version)` and needs all four.

Randomisation happens here and not at render time. A renderer that rolls its own dice makes the seed
a half-truth, and two runs of the same dataset then differ in ways nobody can diff.

Four families, four physics regimes:

* sparse: rejection sampling with a real separation margin. Nothing touches; the baseline.
* packed: rejection sampling with the margin removed. Objects touch in a single layer, which is the
  case where segmentation merges two objects into one mask.
* pile: positions inside a tighter radius, staggered upward, released from height with random
  orientation. Occlusion and stacking come from physics, not from the layout.
* bin: inside a KLT footprint, with walls emitted as fixtures so the renderer and the collision
  model get the same walls from one description.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from src.utility.log_cfg import create_logger

from datagen.assets.composite import sample_composite_asset
from datagen.assets.meshes import MeshAssetBank
from datagen.assets.procedural import sample_procedural_asset
from datagen.config import DatagenConfig
from datagen.constants import DATAGEN_LOG_DIR, LAYOUT_LOG_FILE
from datagen.scenes.spec import (
    SceneAsset,
    BinWall,
    CameraMount,
    CameraPlacement,
    DomainKind,
    DomainRandomization,
    ObjectPlacement,
    SceneFamily,
    SceneSpec,
)

_LOG = create_logger("SceneLayout", LAYOUT_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

__all__ = ["layout_scene", "layout_scene_with_assets", "plan_families", "scene_seed"]

# The scene constants live in the config, because a customer's cell is not this one. Everything that
# describes physical furniture or a difficulty knob is a config field: the margins between objects,
# the try budgets, the KLT's footprint and its walls. They are `workspace.bin` and
# `families.placement`, and each default carries the measurement it came from as field
# documentation.
#
# The pile's drop footprint is `config.families.pile_footprint_fraction` and nothing else. A module
# constant beside it would be a second number the renderer never reads, and the two drift apart in
# silence.


#: Independent random streams off one dataset seed. Separate streams mean a knob that consumes
#: randomness in one place cannot shift every other draw, which is what keeps "same seed, same
#: dataset" true between two versions of the code.
_STREAM_SCENE = 0
_STREAM_FAMILY_ORDER = 1


def scene_seed(dataset_seed: int, index: int, *, stream: int = _STREAM_SCENE) -> int:
    """A per-scene seed derived from the dataset seed, so scenes are independent but reproducible.

    Derived rather than sequential: `dataset_seed + index` makes datasets 0 and 1 share every scene
    but the first, which looks like independent data and is not. ``stream`` selects an independent
    sequence from the same dataset seed. Index and stream must both be non-negative, because
    ``SeedSequence`` refuses otherwise.
    """
    if index < 0 or stream < 0:
        raise ValueError(f"scene_seed needs non-negative index/stream, got {index}/{stream}")
    entropy = [int(dataset_seed), int(index), int(stream)]
    return int(np.random.SeedSequence(entropy=entropy).generate_state(1)[0])


def plan_families(config: DatagenConfig) -> tuple[SceneFamily, ...]:
    """Which family each scene index gets, deterministically, in the configured proportions.

    Assigned up front rather than drawn per scene so the mix is exact: with 300 scenes and equal
    weights the plan holds 75 of each, not roughly 75.
    """
    weights = {
        SceneFamily.SPARSE: config.families.sparse_weight,
        SceneFamily.PACKED: config.families.packed_weight,
        SceneFamily.PILE: config.families.pile_weight,
        SceneFamily.BIN: config.families.bin_weight,
    }
    active = {family: weight for family, weight in weights.items() if weight > 0.0}
    total = sum(active.values())
    counts = {family: int(config.scenes * weight / total) for family, weight in active.items()}
    # Hand the rounding remainder to the heaviest families, largest first, so the total is exact.
    remainder = config.scenes - sum(counts.values())
    for family in sorted(active, key=lambda f: (-active[f], f.value))[:remainder]:
        counts[family] += 1
    plan: list[SceneFamily] = []
    for family in sorted(counts, key=lambda f: f.value):
        plan.extend([family] * counts[family])
    # Interleave deterministically so a truncated run still sees every family, rather than 75 sparse
    # scenes and nothing else.
    rng = np.random.default_rng(scene_seed(config.seed, 0, stream=_STREAM_FAMILY_ORDER))
    order = rng.permutation(len(plan))
    return tuple(plan[i] for i in order)



#: Sources that can supply a real mesh, and the config field carrying each one's weight.
_MESH_SOURCE_WEIGHTS: "tuple[tuple[str, str], ...]" = (
    ("gso", "gso_weight"),
    ("ycb", "ycb_weight"),
    # A source is drawable only if it has both an entry here and a weight field on the config.
    # Without both, every step of the path works except the one that puts an object in a scene, and
    # nothing says so.
    ("thingi10k", "thingi10k_weight"),
    ("asos", "asos_weight"),
    ("objaverse", "objaverse_weight"),
    # The customer's own parts, on the same seam as the research banks: `MeshAssetBank` measures a
    # directory and never cared where the directory came from.
    ("custom", "custom_weight"),
)

#: Weights that name a source this generator cannot supply, with the reason. Accepted by the schema
#: (an operator may reasonably ask) and refused here, loudly, once.
#:
#: It is empty: every source the schema carries a weight for can be supplied today. The mechanism
#: stays because the next source may genuinely be unsupplied, and because a collection whose meshes
#: cannot be attributed belongs here rather than in a corpus: a fail-closed CC-BY audit cannot pass
#: rows that name no author.
#:
#: A refusal that outlives its reason is worse than no refusal, because it reads as a considered
#: decision. An entry comes out the moment its reason is met.
_UNSUPPORTED_SOURCE_WEIGHTS: "tuple[tuple[str, str, str], ...]" = ()

#: Warned-once latches. A 10,000-scene run must not emit the same warning ten thousand times, and it
#: must not stay silent about a knob that is not doing what its operator asked either.
_WARNED_EMPTY: "set[str]" = set()
_WARNED_UNSUPPORTED: "set[str]" = set()

#: The measured mesh bank, built on first use. Measuring a library is expensive, so it happens once
#: per process rather than once per scene.
_MESH_BANK: "MeshAssetBank | None" = None
#: What the cached bank was built for. Every filter that shapes the bank belongs in this key:
#: `max_extent_mm`, the asset-id restriction, `max_jaw_span_mm` and the jaw screen. A key that omits
#: a filter serves the previous filter's bank to the next config, and the filter then reads as
#: inert. For a held-out run that is silent contamination: the restriction is ignored and the
#: dataset still looks held-out.
_MESH_BANK_KEY: "tuple[float, frozenset[str] | None, float | None, str, str] | None" = None


def resolve_mesh_asset_ids(config: "DatagenConfig") -> "frozenset[str] | None":
    """The asset-id restriction from config, or `None` for "the whole bank".

    Both forms, unioned, because they answer different questions: an inline `mesh_asset_ids` is
    right for three custom parts and unusable for a held-out set of hundreds; `mesh_asset_ids_path`
    is right for the set and clumsy for three. The file is a JSON list of ids, or `{source: [ids]}`,
    which is the shape `datagen`'s own held-out analysis writes.

    Refuses a missing or unreadable file rather than falling back to the whole bank. Falling back is
    the worst possible behaviour here: the run would succeed, produce a full-bank dataset, and be
    reported as held-out. A restriction that can silently not apply is not a restriction.
    """

    assets = config.assets
    ids: set[str] = set(getattr(assets, "mesh_asset_ids", ()) or ())
    path = str(getattr(assets, "mesh_asset_ids_path", "") or "")
    if path:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(
                f"assets.mesh_asset_ids_path points at {source}, which does not exist. Refusing "
                f"rather than drawing the whole bank: a restriction that silently does not apply "
                f"turns a held-out dataset into a contaminated one with no way to tell.")
        loaded = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            for group in loaded.values():
                ids.update(str(item) for item in group)
        else:
            ids.update(str(item) for item in loaded)
    if not ids:
        return None
    return frozenset(ids)


def _mesh_bank(config: "DatagenConfig") -> "MeshAssetBank":
    global _MESH_BANK, _MESH_BANK_KEY
    # Every filter that shapes the bank goes in the cache key. A bank cached under a key that does
    # not mention a filter is served to the next config that changes it, and the filter then reads
    # as inert. The screen goes in by its content stamp rather than its path: a screen re-run in
    # place under an unchanged filename would otherwise be served from a bank built before it, and
    # the new verdict would read as inert too.
    screen_path = str(getattr(config.assets, "jaw_screen_path", "") or "")
    stamp = ""
    if screen_path:
        try:
            info = Path(screen_path).stat()
            stamp = f"{info.st_mtime_ns}:{info.st_size}"
        except OSError:
            stamp = "missing"
    key = (float(config.assets.max_mesh_extent_mm), resolve_mesh_asset_ids(config),
           getattr(config.assets, "max_jaw_span_mm", None), screen_path, stamp)
    if _MESH_BANK is None or _MESH_BANK_KEY != key:
        _MESH_BANK = MeshAssetBank(max_extent_mm=key[0], only_asset_ids=key[1],
                                   max_jaw_span_mm=key[2],
                                   jaw_screen=load_jaw_screen(screen_path))
        _MESH_BANK_KEY = key
    return _MESH_BANK


def load_jaw_screen(path: str) -> "dict[str, int] | None":
    """`{asset_id: jaw label count}` from a screen file, or `None` when none is configured.

    Only rows with status `ok` reach the mapping. A count stamped on a mesh that could not be loaded
    reads exactly like a measurement, so such a row is left out and the bank falls back to the proxy
    for it rather than treating "not measured" as "not graspable".
    """
    if not path:
        return None
    file = Path(path)
    if not file.is_file():
        _LOG.warning("assets.jaw_screen_path %s does not exist; the bounding-box filter still "
                       "decides, which over-states graspability by about 55 percent on this "
                  "library", path)
        return None
    payload = json.loads(file.read_text(encoding="utf-8"))
    # Both shapes. A screen taken at `default` density and one taken at `grid` are different
    # measurements of the same meshes and the files look identical, so the density now travels
    # inside the file. A bare list is a screen written before that and is still read, because
    # refusing it would strand the screens already on disk; it just cannot say what it is.
    rows = payload["rows"] if isinstance(payload, dict) else payload
    density = str(payload.get("density", "")) if isinstance(payload, dict) else ""
    screen = {str(row["asset_id"]): int(row.get("jaw", 0))
              for row in rows if row.get("status") == "ok" and "asset_id" in row}
    # A partial screen is not a library with few graspable objects. `--want-graspable` stops the
    # screen once it has found enough, so most meshes may never have been looked at, and every
    # consumer here treats a mesh with no row as unmeasured. Warned once rather than left for
    # whoever wonders why the bank shrank.
    if isinstance(payload, dict) and payload.get("partial"):
        _LOG.warning(
            "jaw screen %s is PARTIAL: %s of %s mesh(es) were screened, the rest were never looked "
            "at. The bank below is a subset of this library, not a measurement of it.",
            path, payload.get("screened"), payload.get("of"))
    if not density:
        _LOG.warning(
            "jaw screen %s does not record the density it was taken at, so it cannot be checked "
            "against the density the corpus will be labelled with. Re-run "
            "`python -m datagen screen-meshes`.", path)
    _LOG.info("jaw screen %s (density %s): %d measured mesh(es), %d with at least one grasp",
              path, density or "UNRECORDED", len(screen),
              sum(1 for v in screen.values() if v > 0))
    return screen


def reset_mesh_bank() -> None:
    """Drop the cached bank and the warn-once latches. A generation run never needs this."""
    global _MESH_BANK, _MESH_BANK_KEY
    _MESH_BANK = None
    _MESH_BANK_KEY = None
    _WARNED_EMPTY.clear()
    _WARNED_UNSUPPORTED.clear()
    _WARNED_UNRESTRICTED.clear()


#: Warned-once latch for the restriction/procedural mix below.
_WARNED_UNRESTRICTED: "set[str]" = set()


def _refuse_unrestricted_procedural(config: "DatagenConfig") -> None:
    """A mesh restriction restricts the mesh draw, not the scene. Say so, loudly.

    This is the trap the whole restriction feature falls into: a list of asset ids plus a mesh
    source weight plus `refuse_procedural_fallback`, with `procedural_weight` left at its default of
    1.0, fills much of every scene with procedural objects, none of them in the requested list, and
    warns about none of it. A user building a held-out corpus gets a contaminated one that looks
    exactly right.

    The empty-source guard cannot see this. It fires only when a weighted mesh source is empty, and
    here nothing is empty: procedural at weight 1.0 is a configured choice sitting beside the
    restricted meshes rather than a fallback, which is also why a user who has just written out a
    list of asset ids does not expect it.

    Warn always, refuse only under the flag. Refusing by default would break every config that mixes
    procedural with meshes on purpose, which is the ordinary case.
    """
    assets_cfg = config.assets
    if resolve_mesh_asset_ids(config) is None:
        return
    procedural = max(0.0, float(assets_cfg.procedural_weight))
    composite = max(0.0, float(getattr(assets_cfg, "composite_weight", 0.0)))
    if procedural <= 0.0 and composite <= 0.0:
        return
    detail = (f"assets.mesh_asset_ids restricts the MESH draw to a named subset, but "
              f"procedural_weight={procedural:.3g} and composite_weight={composite:.3g} still put "
              f"authored objects in every scene; and those are NOT in your list.")
    if getattr(assets_cfg, "refuse_procedural_fallback", False):
        raise ValueError(
            f"{detail} Set both to 0.0 for a dataset drawn only from the assets you named, or clear "
            f"assets.refuse_procedural_fallback if the mix is deliberate.")
    key = f"{procedural}:{composite}"
    if key not in _WARNED_UNRESTRICTED:
        _WARNED_UNRESTRICTED.add(key)
        _LOG.warning("%s Set them to 0.0 for a dataset drawn only from the assets you named.", detail)


def sample_scene_asset(
    config: "DatagenConfig", rng: "np.random.Generator", *, index: int
) -> "SceneAsset":
    """Draw one object for a scene, honouring the operator's source weights.

    The weights are relative, not probabilities: `procedural 1.0, ycb 1.0` means half and half, and
    `procedural 0.0, gso 1.0` means real meshes only. Deterministic given ``rng``.
    """
    assets_cfg = config.assets

    for source, field, reason in _UNSUPPORTED_SOURCE_WEIGHTS:
        if float(getattr(assets_cfg, field, 0.0)) > 0.0 and source not in _WARNED_UNSUPPORTED:
            _WARNED_UNSUPPORTED.add(source)
            _LOG.warning(
                "assets.%s is set but %s cannot be supplied: %s. Those draws fall back to the "
                "remaining sources.", field, source, reason,
            )

    _refuse_unrestricted_procedural(config)
    choices: list[tuple[str, float]] = [
        ("procedural", max(0.0, float(assets_cfg.procedural_weight))),
    ]
    # Composites need no bank on disk: they are authored from parameters like the procedural ones,
    # so there is no "fetch them first" failure mode to guard against.
    composite_weight = max(0.0, float(getattr(assets_cfg, "composite_weight", 0.0)))
    if composite_weight > 0.0:
        choices.append(("composite", composite_weight))
    for source, field in _MESH_SOURCE_WEIGHTS:
        weight = max(0.0, float(getattr(assets_cfg, field, 0.0)))
        if weight <= 0.0:
            continue
        if _mesh_bank(config).is_empty(source):
            if getattr(assets_cfg, "refuse_procedural_fallback", False):
                # The contamination path, closed on request. Falling back turns a dataset whose
                # whole point is which assets are in it into one full of procedural objects, and
                # every procedural family is already in the training corpus, so a held-out render
                # would be fully contaminated behind a single warning line.
                raise ValueError(
                    f"assets.{field}={weight:.3g} but no placeable {source} mesh is present, and "
                    f"assets.refuse_procedural_fallback is set. Falling back to procedural would "
                    f"fill this dataset with objects a model has already seen. Check "
                    f"`python -m datagen heldout` and `assets.mesh_asset_ids_path`, or fetch meshes "
                    f"with `python -m datagen.assets.fetch --list`.")
            if source not in _WARNED_EMPTY:
                _WARNED_EMPTY.add(source)
                _LOG.warning(
                    "assets.%s=%.3g but no placeable %s mesh is present; falling back to "
                    "procedural for those draws. Fetch them with "
                    "`python -m datagen.assets --fetch --from <dir>`; `--check` reports the state.",
                    field, weight, source,
                )
            continue
        choices.append((source, weight))

    # Zero-weight entries dropped before anything counts them. `procedural` is appended
    # unconditionally above, so a composite-only or mesh-only mix would otherwise look like two
    # choices with nothing to choose between, and would consume a random number deciding it.
    choices = [(source, weight) for source, weight in choices if weight > 0.0]

    total = sum(weight for _, weight in choices)
    if total <= 0.0 and getattr(assets_cfg, "refuse_procedural_fallback", False):
        # Same reasoning as above, for the other way in: every weighted source came out empty.
        raise ValueError(
            "every weighted asset source is empty and assets.refuse_procedural_fallback is set. A "
            "procedural scene here would silently replace the assets this dataset exists to isolate.")
    if total <= 0.0:
        # Every weight is zero, or every weighted source is empty. Procedural is the only thing
        # that can always be produced, so a scene still happens: an empty scene would look like a
        # generator bug rather than a configuration one.
        return sample_procedural_asset(rng, assets_cfg.procedural_families, index=index)

    if len(choices) == 1:
        # One source: pick it without touching the random stream.
        #
        # A weighted draw consumes an `rng.random()` even when there is nothing to choose between,
        # so drawing here shifts every subsequent draw by one number and yields different assets, in
        # a different order, from the same seed. A dataset whose provenance stamp no longer rebuilds
        # its own manifest cannot be re-labelled: `scene_geometry` looks extents up by asset id, so
        # the objects that survived would be given the sizes of other objects and the missing ones
        # silently skipped. `manifest_for` reports the hash mismatch, and that is the only sign.
        picked = choices[0][0]
    else:
        draw = float(rng.random()) * total
        cumulative = 0.0
        picked = choices[-1][0]
        for source, weight in choices:
            cumulative += weight
            if draw < cumulative:
                picked = source
                break

    if picked == "procedural":
        return sample_procedural_asset(rng, assets_cfg.procedural_families, index=index)
    if picked == "composite":
        return sample_composite_asset(
            rng, assets_cfg.composite_kinds, index=index,
            unique_id=str(getattr(assets_cfg, "composite_id", "slot")) == "unique")

    asset = _mesh_bank(config).draw(rng, picked)
    if asset is None:  # pragma: no cover (is_empty already excluded this above)
        return sample_procedural_asset(rng, assets_cfg.procedural_families, index=index)
    return asset

def _object_counts(config: DatagenConfig, family: SceneFamily, rng: np.random.Generator) -> int:
    low, high = {
        SceneFamily.SPARSE: config.families.sparse_objects,
        SceneFamily.PACKED: config.families.packed_objects,
        SceneFamily.PILE: config.families.pile_objects,
        SceneFamily.BIN: config.families.bin_objects,
    }[family]
    return int(rng.integers(low, high))


def _yaw_quat_xyzw(yaw_rad: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0))


def _random_quat_xyzw(rng: np.random.Generator) -> tuple[float, float, float, float]:
    """Uniform random orientation (Shoemake). Used for the pile, where anything can land any way up."""
    u1, u2, u3 = rng.random(3)
    s1, s2 = math.sqrt(1.0 - u1), math.sqrt(u1)
    return (
        s1 * math.sin(2.0 * math.pi * u2),
        s1 * math.cos(2.0 * math.pi * u2),
        s2 * math.sin(2.0 * math.pi * u3),
        s2 * math.cos(2.0 * math.pi * u3),
    )


def _oriented_half_extents(
    extent_mm: tuple[float, float, float], quaternion_xyzw: tuple[float, float, float, float],
) -> np.ndarray:
    """Half-extents of the axis-aligned box containing this object at this orientation.

    ``abs(R) @ half`` is the standard result: each world axis's reach is the sum of the object's
    half-extents projected onto it, and the absolute value covers every sign of the rotation. Exact
    rather than conservative, which is the point at spawn time: a bounding sphere claims a 9 mm
    blister needs 49 mm of headroom and drops it from ten times the height it needs.
    """
    x, y, z, w = (float(v) for v in quaternion_xyzw)
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    return np.abs(rotation) @ (np.asarray(extent_mm, dtype=np.float64) / 2.0)


def _random_color(rng: np.random.Generator, config: DatagenConfig) -> tuple[float, float, float]:
    """One RGB triple inside the configured range. Explicitly 3-tuple, so the type survives the draw."""
    lo, hi = config.randomization.object_color_min, config.randomization.object_color_max
    return (
        round(float(rng.uniform(lo[0], hi[0])), 4),
        round(float(rng.uniform(lo[1], hi[1])), 4),
        round(float(rng.uniform(lo[2], hi[2])), 4),
    )


def _footprint_radius_mm(asset: SceneAsset) -> float:
    return 0.5 * math.hypot(asset.extent_mm[0], asset.extent_mm[1])


def _reject_sample_xy(
    rng: np.random.Generator,
    placed: list[tuple[float, float, float]],
    radius_mm: float,
    *,
    center: tuple[float, float],
    half_extents: tuple[float, float],
    margin_mm: float,
    tries: int,
) -> tuple[float, float]:
    """A position at least ``margin_mm`` clear of everything placed, or the least-bad one tried.

    Returns the least-bad candidate rather than raising: a dense family in a small workspace
    legitimately runs out of room, and losing the scene costs more than a tight one. ``tries``
    bounds the search, and the caller passes `families.placement.max_place_tries`.
    """
    best: tuple[float, float] | None = None
    best_slack = -math.inf
    for _ in range(tries):
        x = float(rng.uniform(center[0] - half_extents[0], center[0] + half_extents[0]))
        y = float(rng.uniform(center[1] - half_extents[1], center[1] + half_extents[1]))
        slack = math.inf
        for px, py, pr in placed:
            slack = min(slack, math.hypot(x - px, y - py) - (radius_mm + pr + margin_mm))
        if slack >= 0.0:
            return x, y
        if slack > best_slack:
            best, best_slack = (x, y), slack
    return best if best is not None else (center[0], center[1])


def _place_flat(
    rng: np.random.Generator,
    assets: list[SceneAsset],
    config: DatagenConfig,
    *,
    margin_mm: float,
    center: tuple[float, float],
    half_extents: tuple[float, float],
    colors: list[tuple[float, float, float]],
) -> tuple[ObjectPlacement, ...]:
    """Single-layer placement on the table, no stacking. Orientation per `families.flat_orientation`.

    The orientation is the costly part of this function, not the packing. An object spawned on its
    own base sits in a stable equilibrium, so a solver leaves it there and the corpus fills with
    standing objects, while a tipped object is markedly more likely to admit a jaw grasp. See
    `FamilyMixConfig.flat_orientation` for the measurement.
    """
    random_pose = str(getattr(config.families, "flat_orientation", "upright")) == "random"
    placed: list[tuple[float, float, float]] = []
    out: list[ObjectPlacement] = []
    for asset, color in zip(assets, colors):
        radius = _footprint_radius_mm(asset)
        x, y = _reject_sample_xy(
            rng, placed, radius, center=center, half_extents=half_extents,
            margin_mm=margin_mm, tries=config.families.placement.max_place_tries,
        )
        placed.append((x, y, radius))
        # Spawn a hair above the table so the settle seats the object instead of resolving a
        # penetration on the first step.
        #
        # Half the z extent is the right clearance only for an upright object. Rotate the same body
        # and its lowest point can reach half its diagonal below the centre, so the upright figure
        # would spawn it inside the table and the solver's first step would be a penetration to
        # resolve rather than a settle. The diagonal is the bound that holds for every rotation.
        if random_pose:
            clearance = float(np.linalg.norm(np.asarray(asset.extent_mm, dtype=np.float64))) / 2.0
        else:
            clearance = asset.extent_mm[2] / 2.0
        z = config.workspace.table_height_mm + clearance + 2.0
        out.append(ObjectPlacement(
            asset_id=asset.asset_id,
            position_mm=(round(x, 3), round(y, 3), round(z, 3)),
            orientation_xyzw=(_random_quat_xyzw(rng) if random_pose
                              else _yaw_quat_xyzw(float(rng.uniform(0.0, 2.0 * math.pi)))),
            color_rgb=color,
            mass_kg=asset.mass_kg,
        ))
    return tuple(out)


def _place_pile(
    rng: np.random.Generator,
    assets: list[SceneAsset],
    config: DatagenConfig,
    *,
    center: tuple[float, float],
    colors: list[tuple[float, float, float]],
) -> tuple[ObjectPlacement, ...]:
    """Packed into the lowest free spot above the drop footprint, then released. Physics does the rest.

    Two things have to be true at once:

    * Nothing may spawn inside anything else. PhysX answers deep interpenetration with a separating
      impulse, and the scene launches instead of settling.
    * The drop must be short. Stacking each object above the previous one satisfies the first rule
      and builds a tower metres high; objects then land hard enough to slide off the table, and one
      that leaves it falls forever at PhysX's clamped travel per check window.

    Both hold if the clearance is the object's oriented bounding box rather than its bounding
    sphere. The orientation is drawn here, before the position, so the true axis-aligned extent is
    known rather than guessed at: a 9 mm blister lying flat needs 4.5 mm of headroom, while its
    bounding sphere would claim 49 mm and drop it from ten times the height it needs. Disjoint AABBs
    are still a hard guarantee that the shapes inside them are disjoint, so this is tighter and no
    weaker.

    Several candidate ``(x, y)`` are tried per object and the lowest resulting ``z`` wins, so small
    objects fill the gaps beside large ones instead of being stacked on top of them.
    """
    out: list[ObjectPlacement] = []
    table = config.workspace.table_height_mm
    place = config.families.placement
    reach = min(config.workspace.half_extents_mm) * config.families.pile_footprint_fraction
    placed: list[tuple[float, float, float, np.ndarray]] = []  # x, y, z, oriented half-extents

    for asset, color in zip(assets, colors):
        orientation = _random_quat_xyzw(rng)
        half = _oriented_half_extents(asset.extent_mm, orientation)
        best: tuple[float, float, float] | None = None
        for _ in range(place.pile_drop_tries):
            angle, distance = rng.random() * 2.0 * math.pi, math.sqrt(rng.random()) * reach
            x, y = center[0] + distance * math.cos(angle), center[1] + distance * math.sin(angle)
            z = table + place.pile_base_clearance_mm + float(half[2])
            for px, py, pz, phalf in placed:
                overlaps_xy = (
                    abs(x - px) < half[0] + phalf[0] + place.pile_gap_mm
                    and abs(y - py) < half[1] + phalf[1] + place.pile_gap_mm
                )
                if overlaps_xy:
                    z = max(z, pz + float(phalf[2]) + float(half[2]) + place.pile_gap_mm)
            if best is None or z < best[2]:
                best = (x, y, z)
        assert best is not None  # noqa: S101 (pile_drop_tries >= 1 by the schema)
        placed.append((best[0], best[1], best[2], half))
        out.append(ObjectPlacement(
            asset_id=asset.asset_id,
            position_mm=(round(best[0], 3), round(best[1], 3), round(best[2], 3)),
            orientation_xyzw=orientation,
            color_rgb=color,
            mass_kg=asset.mass_kg,
        ))
    return tuple(out)


def _bin_walls(center: tuple[float, float], config: "DatagenConfig") -> tuple[BinWall, ...]:
    """Four walls around the bin footprint, as centre + half-extents (the fixture-box form).

    One description feeds both the rendered prim and the collision fixture: a wall the camera sees
    but the planner does not is worse than no wall at all.
    """
    box = config.workspace.bin
    inner_x, inner_y = box.inner_mm[0] / 2.0, box.inner_mm[1] / 2.0
    t, h = box.wall_thickness_mm / 2.0, box.wall_height_mm / 2.0
    z = config.workspace.table_height_mm + h
    return (
        BinWall("bin_x_pos", (center[0] + inner_x + t, center[1], z), (t, inner_y + 2 * t, h)),
        BinWall("bin_x_neg", (center[0] - inner_x - t, center[1], z), (t, inner_y + 2 * t, h)),
        BinWall("bin_y_pos", (center[0], center[1] + inner_y + t, z), (inner_x + 2 * t, t, h)),
        BinWall("bin_y_neg", (center[0], center[1] - inner_y - t, z), (inner_x + 2 * t, t, h)),
    )


def _cameras(config: DatagenConfig, rng: np.random.Generator) -> tuple[CameraPlacement, ...]:
    """The configured rig, aimed at the workspace centre.

    The oblique pair sits raised and to either side because the arm is this cell's dominant
    occluder: two opposed viewpoints mean it can never hide the scene from both at once. The wrist
    view is drawn on a hemisphere and is the one the renderer may not be able to reach; that failure
    is a recorded outcome, not a silently relocated camera.
    """
    cx, cy = config.workspace.center_mm
    target = (cx, cy, config.workspace.table_height_mm)
    rig = config.camera_rig
    built: list[CameraPlacement] = []

    # A declared rig wins, and its mounts are read rather than guessed. When `cameras` is None the
    # loop below runs instead, so a run without a declared rig is unchanged.
    #
    # The mount is never inferred from the name. Deciding it by comparing the name makes every name
    # outside a fixed list an eye-in-hand camera, so the obvious way to add a fourth fixed camera
    # produces a wrist one that the renderer then has to reach with the arm.
    if rig.cameras is not None:
        for camera in rig.cameras:
            if camera.mount == "wrist" and camera.position_mm is None:
                radius = float(rng.uniform(*rig.wrist_radius_mm))
                elevation = math.radians(float(rng.uniform(*rig.wrist_elevation_deg)))
                azimuth = float(rng.uniform(0.0, 2.0 * math.pi))
                position = (
                    cx + radius * math.cos(elevation) * math.cos(azimuth),
                    cy + radius * math.cos(elevation) * math.sin(azimuth),
                    config.workspace.table_height_mm + radius * math.sin(elevation),
                )
            else:
                # `CameraSpec` refuses a fixed camera with no position, so this is not None here.
                declared = camera.position_mm
                assert declared is not None
                position = (float(declared[0]), float(declared[1]), float(declared[2]))
            aim = target if camera.look_at_mm is None else (
                float(camera.look_at_mm[0]), float(camera.look_at_mm[1]),
                float(camera.look_at_mm[2]))
            built.append(CameraPlacement(
                name=camera.name,
                mount=CameraMount.WRIST if camera.mount == "wrist" else CameraMount.FIXED,
                position_mm=(round(position[0], 3), round(position[1], 3), round(position[2], 3)),
                look_at_mm=aim,
                resolution=camera.resolution or rig.resolution,
                horizontal_fov_deg=camera.horizontal_fov_deg or rig.horizontal_fov_deg,
            ))
        return tuple(built)

    for name in rig.views:
        if name == "overhead":
            position = (cx, cy, config.workspace.table_height_mm + rig.overhead_height_mm)
            mount = CameraMount.FIXED
        elif name in ("oblique_left", "oblique_right"):
            sign = -1.0 if name == "oblique_left" else 1.0
            position = (
                cx - rig.oblique_setback_mm,
                cy + sign * rig.oblique_lateral_mm,
                config.workspace.table_height_mm + rig.oblique_height_mm,
            )
            mount = CameraMount.FIXED
        else:  # wrist: a hemisphere viewpoint the arm has to reach
            radius = float(rng.uniform(*rig.wrist_radius_mm))
            elevation = math.radians(float(rng.uniform(*rig.wrist_elevation_deg)))
            azimuth = float(rng.uniform(0.0, 2.0 * math.pi))
            position = (
                cx + radius * math.cos(elevation) * math.cos(azimuth),
                cy + radius * math.cos(elevation) * math.sin(azimuth),
                config.workspace.table_height_mm + radius * math.sin(elevation),
            )
            mount = CameraMount.WRIST
        built.append(CameraPlacement(
            name=name, mount=mount,
            position_mm=(round(position[0], 3), round(position[1], 3), round(position[2], 3)),
            look_at_mm=target, resolution=rig.resolution,
            horizontal_fov_deg=rig.horizontal_fov_deg,
        ))
    return tuple(built)


def _randomization(config: DatagenConfig, rng: np.random.Generator) -> DomainRandomization:
    ranges = config.randomization
    return DomainRandomization(
        light_elevation_deg=round(float(rng.uniform(*ranges.light_elevation_deg)), 3),
        light_azimuth_deg=round(float(rng.uniform(0.0, 360.0)), 3),
        light_intensity=round(float(rng.uniform(*ranges.light_intensity)), 3),
        dome_intensity=round(float(rng.uniform(*ranges.dome_intensity)), 3),
        table_color_rgb=(
            round(float(rng.uniform(ranges.table_color_min[0], ranges.table_color_max[0])), 4),
            round(float(rng.uniform(ranges.table_color_min[1], ranges.table_color_max[1])), 4),
            round(float(rng.uniform(ranges.table_color_min[2], ranges.table_color_max[2])), 4),
        ),
    )


def layout_scene(config: DatagenConfig, index: int, family: SceneFamily | None = None) -> SceneSpec:
    """The whole scene for ``index``, from the seed alone. Same inputs, byte-identical output."""
    return layout_scene_with_assets(config, index, family)[0]


def layout_scene_with_assets(
    config: DatagenConfig, index: int, family: SceneFamily | None = None,
) -> tuple[SceneSpec, tuple[SceneAsset, ...]]:
    """The scene and the assets it drew, in placement order.

    Both are returned together because the alternative, re-drawing the assets from the same seed
    elsewhere, depends on consuming the random stream in exactly the same order: a contract no
    signature states and nothing checks. Handing the assets back makes that mistake unavailable.
    """
    if not 0 <= index < config.scenes:
        raise ValueError(f"scene index {index} outside 0..{config.scenes - 1}")
    chosen = family if family is not None else plan_families(config)[index]
    rng = np.random.default_rng(scene_seed(config.seed, index))

    count = _object_counts(config, chosen, rng)
    assets = [sample_scene_asset(config, rng, index=i) for i in range(count)]
    colors = [_random_color(rng, config) for _ in assets]

    center = config.workspace.center_mm
    place = config.families.placement
    walls: tuple[BinWall, ...] = ()
    drop_height = 0.0
    if chosen is SceneFamily.SPARSE:
        objects = _place_flat(rng, assets, config, margin_mm=place.sparse_margin_mm, center=center,
                              half_extents=config.workspace.half_extents_mm, colors=colors)
    elif chosen is SceneFamily.PACKED:
        objects = _place_flat(rng, assets, config, margin_mm=place.packed_margin_mm, center=center,
                              half_extents=config.workspace.half_extents_mm, colors=colors)
    elif chosen is SceneFamily.PILE:
        objects = _place_pile(rng, assets, config, center=center, colors=colors)
        drop_height = place.pile_base_clearance_mm
    else:  # BIN: inside the KLT footprint, walls emitted with it
        box = config.workspace.bin
        inner = (box.inner_mm[0] / 2.0 - box.object_inset_mm,
                 box.inner_mm[1] / 2.0 - box.object_inset_mm)
        objects = _place_flat(rng, assets, config, margin_mm=place.packed_margin_mm, center=center,
                              half_extents=inner, colors=colors)
        walls = _bin_walls(center, config)

    spec = SceneSpec(
        scene_id=f"{chosen.value}_{index:06d}",
        seed=scene_seed(config.seed, index),
        family=chosen,
        domain=DomainKind(config.domain),
        objects=objects,
        cameras=_cameras(config, rng),
        randomization=_randomization(config, rng),
        bin_walls=walls,
        drop_height_mm=drop_height,
    )
    return spec, tuple(assets)
