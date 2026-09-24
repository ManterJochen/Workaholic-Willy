"""Configuration loader: reads YAML files, merges them, validates the result.

Public API
----------
:func:`load_config`
    Read the configured YAML tree, merge it and return a fully-validated
    :class:`AppConfig`. The result is cached by ``data_dir`` and active
    profile; :func:`reload_config` invalidates it after file edits.

:func:`reload_config`
    Drop the cache.

Section loaders
    :func:`load_robot_section`, :func:`load_camera_section`, :func:`load_speech_section` and
    :func:`load_perception_section` read and validate one section each, through the same profile
    chain, so a fault in one section leaves the others loadable. They are not cached.

Layout
------
The loader expects (paths are relative to ``data_dir``)::

    app/runtime.yaml          (optional: schema defaults are used otherwise)
    camera/cam.yaml
    camera/stereomatcher.yaml
    camera/hand_eye.yaml       (optional: schema defaults are used otherwise)
    models/*.yaml             (auto-discovered; every top-level key is merged)
    robot/robot.yaml          (optional)

Environment-variable substitution
---------------------------------
Strings in any YAML may reference environment variables as ``${VAR}`` or
``${VAR:-default}``. Substitution happens before parsing, so the result must
be valid YAML.

Examples::

    ip: ${ROBOT_IP:-192.168.1.100}
    model_path: ${MODEL_DIR}/whisper

A referenced variable that is unset and carries no default raises
:class:`ConfigError`. ``${WILLY_PROJECT_ROOT}`` is the one name not substituted: it is the path anchor
below, it reaches the parsed data as written in any YAML style, quoted or not, inside ``{...}`` and
``[...]`` too, and the path rule expands it in a path field. Any other field refuses it.

Paths
-----
A relative path in the tree is read against the tree's folder, ``data_dir``, wherever the tree
lives; ``${WILLY_PROJECT_ROOT}/...`` names a file in the repository; an absolute path is read as
written. Every path field is typed :data:`~src.config.paths.ConfigPath`, and every door below
validates with ``data_dir`` as the validation context, so the validated config holds absolute paths.
:mod:`src.config.paths` states the rule and why it is applied here.

Errors
------
Every load-time failure is raised as :class:`ConfigError`. A schema failure names, per offending key,
the file and line it was written on, which profile layer that file belongs to, and a did-you-mean for
a near-miss key name. A profile chain merges several files into one section, so the data directory
alone does not identify which of them carried the typo.
"""

from __future__ import annotations

import copy
import difflib
import os
import re
from collections.abc import Collection, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

# The one import this package takes out of `src`, and it is free: `src` is a namespace package with
# no `__init__.py` and `contracts` is stdlib-only by rule, so this pulls three modules in about
# 1 ms, with no numpy, no torch and no `utility/__init__`. Anything heavier reaching in here would
# put weight on the boot path of every process that reads config, which is all of them.
from src.contracts import UNSET, Maybe, chosen

from ._merge import _deep_merge
from .paths import CONFIG_FOLDER, PROJECT_ROOT_ANCHOR, PROJECT_ROOT_VARIABLE
from .schema.app import AppConfig, CameraConfig, ModelsConfig, PerceptionModelsConfig
from .schema.models import SpeechToTextConfig
from .schema.robot import RobotConfig

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class ConfigError(RuntimeError):
    """Raised on any configuration load / validation failure."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The YAML tree sits at the repository root beside src/, because it is what an operator edits.
#: This file is src/config/loader.py, so the root is two levels up.
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "config"
_PROFILE_ENV_VAR = "WILLY_PROFILE"

#: ``WILLY_PROFILE`` accepts a comma-separated chain of profile layers applied left-to-right:
#: ``"sim,ur3e"`` merges every ``*.sim.yaml`` first, then every ``*.ur3e.yaml`` on top, and a single
#: name is a one-element chain. Layers chain rather than fork because they are independent
#: dimensions: the sim overlays span four files (``robot``, ``camera/hand_eye``, ``models/object``,
#: ``models/segmenting``) and carry measured values such as the detector ``torch_dtype`` reset that
#: small-object recall depends on, and a "sim cell driving a UR3e" profile would fork all four and
#: duplicate those values per robot, where they drift. Chaining keeps one ``sim`` layer and adds a
#: small per-robot layer stating only what the robot changes.
_PROFILE_SEPARATOR = ","
#: Phase U11 opt-in environment variable naming an overlay YAML. When it points at a readable path,
#: the loader deep-merges that overlay into the ``robot`` section of the raw config before Pydantic
#: validation, and rejects it unless every leaf key path is in the ``runtime_mutable=True``
#: allow-list discovered from ``RobotConfig``. Unset means no overlay is merged.
_ADAPTATION_OVERLAY_ENV_VAR = "WILLY_ADAPTATION_OVERLAY"

#: Reset sentinel for profile overlays, and the only way an overlay can force a field back to
#: ``None``. A ``None`` overlay leaf keeps the base value, so a partial overlay never wipes a field;
#: a profile that must unset one writes this string instead and the loader resolves it to ``None``
#: after the merge. ``models/object.sim.yaml`` uses it to drop the production
#: ``optim.torch_dtype: auto`` so the sim's detector runs its validated unset/fp32-weights recall
#: path.
_OVERLAY_RESET = "__null__"

# The required camera files, not auto-discovered: the loader maps each one to a specific section
# of the AppConfig tree.
_CAMERA_FILES = ("cam.yaml", "stereomatcher.yaml")
_CAMERA_KEYS = ("cameras", "stereomatcher")

# Matches ${VAR} and ${VAR:-default} on a single line; nested references are not supported.
_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:\:-([^}]*))?\}")

#: The refusal every door that asks a tree for its robot block gives when there is none:
#: :func:`load_robot_config` and :attr:`~src.config.tree.LoadedTree.robot`.
_NO_ROBOT_BLOCK: Final = "the loaded config has no `robot` block"

#: Where a refusal says a value given to :meth:`~src.config.tree.LoadedTree.with_values` was
#: written. It has no file and no line, and naming the file line it replaced would put the value
#: above a line that sets another one.
_IN_MEMORY_ORIGIN: Final = "LoadedTree.with_values"

#: One part of a dotted key given in memory: a name, and at most one ``[index]`` after it.
_VALUE_SEGMENT = re.compile(r"(?P<name>[^.\[\]\s]+)(?:\[(?P<index>\d+)\])?")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(
    data_dir: str | Path | None = None,
    *,
    profile: "Maybe[str | None]" = UNSET,
) -> AppConfig:
    """Load + validate the YAML config tree.

    Args:
        data_dir: Override the default ``config`` location, for a fixture
            tree or a per-deployment overlay.
        profile: The overlay chain to apply, e.g. ``"sim"`` or
            ``join_profiles("sim", "ur3e")``. Omit it to read
            :data:`WILLY_PROFILE` from the environment.

    Returns:
        A fully-validated, immutable :class:`AppConfig` instance. Repeated
        calls with the same ``data_dir`` and active profile return the
        same cached object.

    Raises:
        ConfigError: If a required file is missing, YAML parsing fails,
            an environment variable is unresolved, or schema validation
            rejects the merged result.

    ``profile=None`` selects the base tree with no overlays, which is why the default is
    :data:`UNSET` and not ``None``: ``None`` is a value here and cannot also mean "not given". A
    caller passing it gets the unlayered tree even when ``WILLY_PROFILE`` is set in the environment,
    which is what a tool that must not inherit the operator's shell needs. Selecting the chain by
    mutating ``os.environ[WILLY_PROFILE]`` instead leaks process-global state into whatever runs
    next.

    Both doors, the environment and this argument, validate through :func:`_validated_chain`, so a
    typo'd layer is refused here rather than merging as a silent no-op and bringing the cell up with
    another robot's geometry.
    """
    root, chain = _root_and_chain(data_dir, profile)
    return _load_cached(str(root), chain)


def load_robot_config(
    data_dir: str | Path | None = None,
    *,
    profile: "Maybe[str | None]" = UNSET,
) -> "RobotConfig":
    """The ``robot`` block of the tree, or a refusal saying there is none.

    The refusal is a :class:`ConfigError`, not a ``SystemExit``: this is library code, where an exit
    request reads as something other than a fault and is invisible to a caller that catches
    exceptions properly. A CLI wrapping this call must catch ``ConfigError``, which
    :func:`load_config` also raises on a broken YAML tree, or that tree reaches the terminal as a
    traceback instead of as the refusal the CLI means to print.

    ``robot`` is optional on :class:`AppConfig` (``robot: RobotConfig | None = None``), so a tree
    without one is a configuration answer rather than a crash.
    """
    cfg = load_config(data_dir, profile=profile)
    robot = getattr(cfg, "robot", None)
    if robot is None:
        raise ConfigError(_NO_ROBOT_BLOCK)
    return robot


def load_robot_section(
    data_dir: str | Path | None = None,
    *,
    profile: "Maybe[str | None]" = UNSET,
) -> RobotConfig:
    """The ``robot`` section alone: ``robot/robot.yaml`` and its overlays, validated as ``RobotConfig``.

    It equals ``load_config(...).robot`` on a tree that loads, and it still loads when a camera file or a
    model file is broken, which :func:`load_robot_config` does not: that one validates the whole tree
    first. The U11 adaptation overlay is applied here exactly as the whole-tree load applies it, or the
    two doors would describe different cells whenever ``WILLY_ADAPTATION_OVERLAY`` is set. The
    cross-section rule on :class:`AppConfig` (every fused camera is a rig) needs the camera section
    and does not run here; a caller that combines this section with :func:`load_camera_section` runs
    it through :func:`~src.config.schema.app.camera_calibration_conflict`. A schema failure names each
    key with its ``robot.`` prefix and the file and line it was written on, as the whole-tree load does.

    Not cached. A section is cheap to read, and a cached one would outlive the file edit that
    :func:`reload_config` is documented to cover. The other side of that: after a change and before
    :func:`reload_config`, a section already reads the new state while :func:`load_config` still
    returns the tree it cached. That holds for a file edit and equally for the environment the load
    reads, a changed ``WILLY_ADAPTATION_OVERLAY`` or a ``${VAR}`` a file substitutes, because the cache
    is keyed by the data directory and the profile chain only.
    """
    root, chain = _root_and_chain(data_dir, profile)
    path = root / "robot" / "robot.yaml"
    raw: dict[str, Any] = {}
    robot = _load_optional_section(path, "robot", chain)
    if robot is not None:
        raw["robot"] = robot
    # The whole-tree order: the file first, then the overlay, which can supply a robot block the
    # file does not have.
    _apply_env_adaptation_overlay(raw)
    _apply_named_hand(raw, root, chain)
    if raw.get("robot") is None:
        raise ConfigError(
            f"there is no robot section: {path} and its profile overlays are absent or leave the "
            "top-level 'robot:' empty, and no adaptation overlay supplies one"
        )
    return _validate_section(RobotConfig, raw["robot"], ("robot",), root, chain)


def load_camera_section(
    data_dir: str | Path | None = None,
    *,
    profile: "Maybe[str | None]" = UNSET,
) -> CameraConfig:
    """The ``camera`` section alone: ``camera/cam.yaml``, ``stereomatcher.yaml`` and ``hand_eye.yaml``.

    It equals ``load_config(...).camera`` on a tree that loads, and it reads neither the robot nor the
    models. The cross-section rule on :class:`AppConfig` does not run, for the reason given on
    :func:`load_robot_section`; a caller combining both sections runs
    :func:`~src.config.schema.app.camera_calibration_conflict`. Not cached.
    """
    root, chain = _root_and_chain(data_dir, profile)
    return _validate_section(
        CameraConfig, _load_camera_section(root, chain), ("camera",), root, chain,
    )


def load_speech_section(
    data_dir: str | Path | None = None,
    *,
    profile: "Maybe[str | None]" = UNSET,
) -> SpeechToTextConfig:
    """``models.stt`` alone, so speech loads without a camera, a robot or a detector.

    ``models/`` is read file by file under the rule :func:`_load_model_keys` states: a model file that
    cannot be read is tolerated only when the ``stt`` block was found in files that could. Not cached.
    """
    root, chain = _root_and_chain(data_dir, profile)
    found = _load_model_keys(root, chain, ("stt",))
    if "stt" not in found:
        raise ConfigError(
            f"there is no speech section: no file under {root / 'models'} defines a top-level 'stt:' key"
        )
    return _validate_section(SpeechToTextConfig, found["stt"], ("models", "stt"), root, chain)


def load_perception_section(
    data_dir: str | Path | None = None,
    *,
    profile: "Maybe[str | None]" = UNSET,
) -> PerceptionModelsConfig:
    """The seven ``models`` keys the perception stack reads, without ``stt`` or the hand detectors.

    Every field equals the same field of ``load_config(...).models`` on a tree that loads, and the
    result carries exactly what :class:`~src.models.perception_spec.PerceptionSpec` reads. When
    a model file cannot be read and one of the seven was found nowhere, the section refuses: the
    unreadable file might have held it, and a schema default must not stand in for a value nobody
    could read. Not cached.
    """
    root, chain = _root_and_chain(data_dir, profile)
    found = _load_model_keys(root, chain, tuple(PerceptionModelsConfig.model_fields))
    return _validate_section(PerceptionModelsConfig, found, ("models",), root, chain)


def reload_config() -> None:
    """Invalidate the :func:`load_config` cache.

    Call after editing YAML files at runtime, or between loads of two different fixture trees.
    """
    _load_cached.cache_clear()


def active_profile() -> str | None:
    """The raw chain named by :data:`WILLY_PROFILE`, or ``None`` when the variable is unset."""
    raw = os.environ.get(_PROFILE_ENV_VAR, "").strip()
    return raw or None


def profile_layers(profile: str | None) -> tuple[str, ...]:
    """Split a profile value into its ordered overlay layers (``"sim,ur3e"`` -> ``("sim", "ur3e")``).

    Later layers win: each layer's ``*.<layer>.yaml`` overlay is deep-merged on top of the previous
    one. Blank segments are dropped, so ``"sim,"`` and ``" sim , ur3e "`` are accepted.
    """
    if not profile:
        return ()
    return tuple(part for part in (seg.strip() for seg in profile.split(_PROFILE_SEPARATOR)) if part)


def join_profiles(*layers: str) -> str:
    """Build a chain value for :func:`set_active_profile` (``join_profiles("sim", "ur3e")``)."""
    return _PROFILE_SEPARATOR.join(layer for layer in layers if layer)


def set_active_profile(profile: str | None) -> None:
    """Set or clear :data:`WILLY_PROFILE` for subsequent config loads.

    This writes process-global state that outlives the call. A caller selecting a tree for itself
    passes ``profile=`` to :func:`load_config` instead.
    """
    val = (profile or "").strip()
    if val:
        os.environ[_PROFILE_ENV_VAR] = val
    else:
        os.environ.pop(_PROFILE_ENV_VAR, None)


def available_profiles(data_dir: str | Path | None = None) -> list[str]:
    """The sorted profile layer names discovered from ``*.{layer}.yaml`` files under ``data_dir``.

    Layers are composable: any subset can be combined into a chain (``"sim,ur3e"``), so this is the
    menu of building blocks, not the list of valid whole configurations.
    """
    root = Path(data_dir).resolve() if data_dir else _DEFAULT_DATA_DIR
    out: set[str] = set()
    for path in root.rglob("*.yaml"):
        parts = path.stem.split(".")
        if len(parts) < 2:
            continue
        profile = parts[-1].strip()
        if profile:
            out.add(profile)
    return sorted(out)


# ---------------------------------------------------------------------------
# Internal: cached loader
# ---------------------------------------------------------------------------

_Section = TypeVar("_Section", bound=BaseModel)


def _root_and_chain(data_dir: str | Path | None, profile: "Maybe[str | None]") -> tuple[Path, str]:
    """The data root and the validated profile chain, resolved once for every public door.

    :func:`load_config` and the section loaders all come through here, so a chain is validated the same
    way whichever of them a caller used.
    """
    root = Path(data_dir).resolve() if data_dir else _DEFAULT_DATA_DIR
    # `chosen()` rather than `profile is UNSET`, with the positive arm the one that uses the value:
    # `is` narrows nothing for a type checker (`_Unset` is a plain class, not a singleton it can
    # reason about), while `chosen` is a `TypeGuard` and narrows only where it is true. Hence the
    # order of the branches.
    chain = (
        _validated_chain(root, profile, source="profile") if chosen(profile)
        else _active_profile(root)
    )
    return root, chain


def _validate_section(
    model: type[_Section], data: Any, prefix: tuple[str, ...], root: Path, profile: str,
) -> _Section:
    """Validate one section as ``model``; a failure is described with the section's place in the tree."""
    try:
        return model.model_validate(data, context=_path_context(root))
    except ValidationError as exc:
        raise ConfigError(_describe_validation_error(exc, root, profile, prefix=prefix)) from exc


def _apply_named_hand(
    raw: dict[str, Any], root: Path, profile: str, *, in_memory: Collection[str] = (),
) -> None:
    """A named hand supplies its unset widths and envelope, and a contradiction is refused.

    It runs after every merge and overlay, so what a profile leaves unset is what the chain as a
    whole leaves unset, and before validation, so the schema judges the numbers the cell will run.
    Both doors call it: the whole tree and the robot section. The numbers come from the repository
    registry whatever tree is loaded, because that registry is the one authority: a tree's own
    grippers/ may repeat a hand, and the validator and the desk refuse one that differs.

    ``in_memory`` holds the keys given to :meth:`~src.config.tree.LoadedTree.with_values`, spelled
    by :func:`_normal_key`. A contradiction among them is said to be written there, never at the
    file line the value replaced.
    """
    if raw.get("robot") is None:
        return
    from .hand_numbers import apply_named_hand

    def stated_in() -> dict[str, str]:
        try:
            from ._provenance import index_origins

            found = {key: origin.location(root) for key, origin in index_origins(root, profile_layers(profile)).items()
                     if key.startswith("robot.")}
        except Exception:  # noqa: BLE001 (where a number was written is for the sentence; never mask the refusal)
            found = {}
        found.update({key: _IN_MEMORY_ORIGIN for key in in_memory if key.startswith("robot.")})
        return found

    raw["robot"] = apply_named_hand(raw["robot"], stated_in=stated_in, data_dir=root)


def _value_segments(key: str) -> tuple[str | int, ...]:
    """The parts of a dotted key given in memory, an index as an ``int``.

    ``rigs[0].enabled`` gives ``rigs, 0, enabled``. A key that does not parse is a programmer error
    and raises ``ValueError``; it never reaches a tree.
    """
    segments: list[str | int] = []
    for part in key.split("."):
        match = _VALUE_SEGMENT.fullmatch(part)
        if match is None:
            raise ValueError(
                f"{key!r} is not a dotted config key: each part between the dots is a name, with at most one "
                "[index] after it, as in 'camera.cameras.rigs[0].enabled'"
            )
        segments.append(match["name"])
        if match["index"] is not None:
            segments.append(int(match["index"]))
    return tuple(segments)


def _normal_key(key: str) -> str:
    """``key`` as a validation error spells it once its union tags are gone: ``rigs[0]`` becomes ``rigs.0``.

    Not as the side-car index spells it, which keeps the brackets (``rigs[0]``): :func:`_yaml_segments`
    turns an error location into the one spelling both are compared in.
    """
    return ".".join(str(segment) for segment in _value_segments(key))


def _bracketed(segments: Sequence[str | int], origins: "Mapping[str, Any] | None" = None) -> str:
    """``segments`` as the YAML files and the side-car index spell them: ``camera.cameras.rigs[0].fps``.

    An ``int`` is a list position, written ``[0]``, unless the files write it as a map key: a YAML
    ``0:`` under a mapping is indexed ``cameras.0.enabled``, and bracketing it would name a key no file
    has.
    """
    out = ""
    for segment in segments:
        if isinstance(segment, int) and not (origins is not None and _written(f"{out}.{segment}", origins)):
            out = f"{out}[{segment}]"
        else:
            out = f"{out}.{segment}" if out else str(segment)
    return out


def _yaml_segments(loc: Sequence[str | int], origins: Mapping[str, Any]) -> list[str | int]:
    """A validation error's location with the tagged-union levels no YAML file has taken out.

    ⚠ MEASURED 2026-09-21, AND IT IS WHAT AN OPERATOR WAS SHOWN. A camera rig is a tagged union,
    and pydantic puts the tag in the error location as a level of its own:
    ``camera.cameras.rigs.0.rgbd.extrinsics.artifact_path``. The file spells that key
    ``rigs[0].extrinsics.artifact_path``, with no ``rgbd`` level, so the lookup missed and the message
    said "not written in any YAML" about a key that was on line 13 of the operator's own layer. Every
    error inside a camera rig was reported that way.

    ⛔ THE TAG IS FOUND IN THE SCHEMA, NOT IN THE FILES. A first version guessed it from what the YAML
    wrote beside it, and a review measured it dropping real blocks: `rgbd_backend: realsense` beside a
    `realsense` block no file writes made `rigs[0].realsense.laser_power_mw` print as
    `rigs[0].laser_power_mw`, and `pipeline.kind: zero_shot` did the same to `pipeline.zero_shot`. A
    level is dropped here only where :func:`_tag_sites` says a tagged union sits, and only when it is
    one of that union's own tags, so a field that merely shares a name with a value keeps its name.
    One level per position: the part after a dropped tag is a field of the member, even one named
    like a tag (``rigs.0.realsense.realsense`` keeps its second ``realsense``).
    ``origins`` is not read; it stays in the signature so every caller spells the location one way.
    """
    del origins  # decided by position and allowed value alone; see the docstring
    out: list[str | int] = []
    dropped_at = -1
    for part in loc:
        if isinstance(part, str) and out and len(out) != dropped_at and _is_union_tag(out, part):
            dropped_at = len(out)
            continue
        out.append(part)
    return out


def _is_union_tag(prefix: Sequence[str | int], part: str) -> bool:
    for path, tags in _tag_sites():
        if (len(path) == len(prefix) and part in tags
                and all(step == "*" or step == str(given) for step, given in zip(path, prefix))):
            return True
    return False


@lru_cache(maxsize=1)
def _tag_sites() -> "tuple[tuple[tuple[str, ...], frozenset[str]], ...]":
    """Every place in the tree a tagged union sits, and its tags. ``*`` in a path is a list position or map key.

    Read off :class:`AppConfig` once, so a union added to the schema is covered the day it lands. Only a
    union with a string discriminator adds a level to an error location; an optional field or a union
    without one adds none, and is not listed.
    """
    sites: list[tuple[tuple[str, ...], frozenset[str]]] = []
    _sites_in_model(AppConfig, (), sites, set())
    return tuple(sites)


def _sites_in_model(model: type, path: tuple[str, ...], sites: list[Any], seen: set[tuple[type, tuple[str, ...]]]) -> None:
    if (model, path) in seen or len(path) > 16:
        return
    seen.add((model, path))
    for name, field in getattr(model, "model_fields", {}).items():
        _sites_in_type(field.annotation, getattr(field, "discriminator", None),
                       (*path, field.alias or name), sites, seen)


def _sites_in_type(annotation: Any, discriminator: Any, path: tuple[str, ...], sites: list[Any],
                   seen: set[tuple[type, tuple[str, ...]]]) -> None:
    import types
    import typing

    from pydantic.fields import FieldInfo

    origin = typing.get_origin(annotation)
    arguments = typing.get_args(annotation)
    if origin is typing.Annotated:
        for extra in arguments[1:]:
            if isinstance(extra, FieldInfo) and isinstance(extra.discriminator, str):
                discriminator = extra.discriminator
        _sites_in_type(arguments[0], discriminator, path, sites, seen)
    elif origin in (typing.Union, types.UnionType):
        members = [member for member in arguments if isinstance(member, type) and issubclass(member, BaseModel)]
        if isinstance(discriminator, str) and len(members) > 1:
            tags = frozenset(tag for member in members for tag in _literal_values(member, discriminator))
            sites.append((path, tags))
        for member in arguments:
            _sites_in_type(member, None, path, sites, seen)
    elif origin in (list, tuple, set, frozenset):
        for item in arguments:
            if item is not Ellipsis:
                _sites_in_type(item, discriminator, (*path, "*"), sites, seen)
    elif origin is dict:
        if len(arguments) == 2:
            _sites_in_type(arguments[1], discriminator, (*path, "*"), sites, seen)
    elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
        _sites_in_model(annotation, path, sites, seen)


def _literal_values(model: type, field_name: str) -> tuple[str, ...]:
    import typing

    field = getattr(model, "model_fields", {}).get(field_name)
    annotation = getattr(field, "annotation", None)
    while typing.get_origin(annotation) is typing.Annotated:
        annotation = typing.get_args(annotation)[0]
    if typing.get_origin(annotation) is typing.Literal:
        return tuple(str(value) for value in typing.get_args(annotation))
    return ()


def _block_origin(segments: Sequence[str | int], origins: Mapping[str, Any]) -> Any:
    """Where the block ``segments`` names begins, or ``None`` where no file writes it.

    A block written as a key has its own line. An item of a list does not: the index records keys, and
    ``rigs[0]`` is a position, so the item is found at the first line any of its own keys is written on.
    """
    if not segments:
        return None
    key = _bracketed(segments, origins)
    if key in origins:
        return origins[key]
    inside = [origin for other, origin in origins.items() if other.startswith((f"{key}.", f"{key}["))]
    return min(inside, key=lambda origin: (str(origin.file), origin.line)) if inside else None


def _written(key: str, origins: Mapping[str, Any]) -> bool:
    """Whether any file writes ``key`` or anything inside it."""
    return key in origins or any(other.startswith((f"{key}.", f"{key}[")) for other in origins)


def _in_memory_keys(values: Mapping[str, Any]) -> frozenset[str]:
    """The keys of ``values``, spelled by :func:`_normal_key`."""
    return frozenset(_normal_key(key) for key in values)


def _within(dotted: str, keys: Collection[str]) -> bool:
    """Whether ``dotted``, spelled by :func:`_normal_key`, is one of ``keys`` or lies inside one.

    An item of a list given whole lies inside the key of the list.
    """
    return any(dotted == key or dotted.startswith(f"{key}.") for key in keys)


def _holds(dotted: str, keys: Collection[str]) -> bool:
    """Whether ``dotted``, spelled by :func:`_normal_key`, is a block that holds one of ``keys``."""
    return any(key.startswith(f"{dotted}.") for key in keys)


def _apply_values(raw: dict[str, Any], values: Mapping[str, Any]) -> None:
    """Set each dotted key of ``values`` in the merged ``raw`` tree, in order, so a later key wins.

    A block on the way that no file writes, or that a file wrote as a scalar, becomes a mapping, so
    a value of the wrong shape is refused by the schema in its own words. ``[n]`` sets an item of a
    list the files hold; an item the list does not hold is refused here, because no schema rule can
    name it. Each value is copied, so the tree that was given it never shares a list with the raw
    mapping.
    """
    for key, value in values.items():
        segments = _value_segments(key)
        node: Any = raw
        for depth, segment in enumerate(segments):
            if isinstance(segment, int) and not (isinstance(node, list) and segment < len(node)):
                where = ".".join(str(part) for part in segments[:depth])
                held = f"which holds {len(node)} item(s)" if isinstance(node, list) else "which is not a list"
                raise ConfigError(
                    f"{key} names item {segment} of {where}, {held} in this tree. A value given in memory sets an "
                    "item the tree holds and adds none."
                )
            if depth == len(segments) - 1:
                node[segment] = copy.deepcopy(value)
                break
            child = node[segment] if isinstance(segment, int) else node.get(segment)
            if not isinstance(segments[depth + 1], int) and not isinstance(child, dict):
                child = {}
                node[segment] = child
            node = child


@lru_cache(maxsize=8)
def _load_cached(root_str: str, profile: str) -> AppConfig:
    return _assemble(Path(root_str), profile, {})


def _load_with_values(
    data_dir: str | Path | None, *, profile: "Maybe[str | None]", values: Mapping[str, Any],
) -> AppConfig:
    """The whole tree with ``values`` given in memory on top of every layer, validated, not cached.

    The door behind :meth:`~src.config.tree.LoadedTree.with_values`. It is not cached because two
    sets of values under one chain are two trees, and the cache is keyed by the directory and the
    chain only.
    """
    root, chain = _root_and_chain(data_dir, profile)
    return _assemble(root, chain, values)


def _assemble(root: Path, profile: str, values: Mapping[str, Any]) -> AppConfig:
    """Read every file of the tree under ``profile``, set ``values`` on top, fill the named hand, and validate.

    The one pipeline behind :func:`load_config` and :meth:`~src.config.tree.LoadedTree.with_values`,
    so a value given in memory meets exactly the checks a value written in a layer meets. Every file
    is read first, the runtime section with the others, and then the steps that change the merged
    mapping run: the adaptation overlay, the values, the named hand. The runtime file and the named
    hand touch different sections, so their order decides only which refusal a tree broken in both
    places meets first, and that is the runtime file's.
    """
    raw: dict[str, Any] = {
        "camera": _load_camera_section(root, profile),
        "models": _load_models_section(root, profile),
    }

    robot = _load_optional_section(root / "robot" / "robot.yaml", "robot", profile)
    if robot is not None:
        raw["robot"] = robot

    runtime = _load_optional_section(root / "app" / "runtime.yaml", "runtime", profile)
    if runtime is not None:
        raw["runtime"] = runtime

    _apply_env_adaptation_overlay(raw)
    given = _in_memory_keys(values)
    _apply_values(raw, values)
    _apply_named_hand(raw, root, profile, in_memory=given)

    try:
        return AppConfig.model_validate(raw, context=_path_context(root))
    except ValidationError as exc:
        raise ConfigError(_describe_validation_error(exc, root, profile, in_memory=given)) from exc


def _path_context(root: Path) -> dict[str, Any]:
    """The validation context every door validates with: the tree's folder, which relative paths are read against.

    A value given in memory is validated under it too, because it is part of this tree: the same value
    written in a layer would be read against the same folder.
    """
    return {CONFIG_FOLDER: root}


def _describe_validation_error(
    exc: ValidationError, root: Path, profile: str, *, prefix: tuple[str, ...] = (),
    in_memory: Collection[str] = (),
) -> str:
    """Render a validation failure so the reader can go straight to the line that caused it.

    A profile chain merges several files into one section, so naming the data directory alone would
    leave the reader to guess which of them carried the typo. Explaining relaxes nothing: same
    exception, same fail-closed behaviour, strictly more information.

    Explaining is best-effort and must never become a second failure: if the side-car index cannot be
    built, the original pydantic report is still printed in full.

    ``prefix`` is where a section validated on its own sits in the whole tree (``("robot",)`` for
    :func:`load_robot_section`), so its keys are named, and found in the side-car index, exactly as
    the whole-tree load names them.

    ``in_memory`` holds the keys given to :meth:`~src.config.tree.LoadedTree.with_values`, spelled
    by :func:`_normal_key`. A key at or inside one of them is said to be written there, ahead of a
    file line that also writes it, because that line holds another value; a block no file writes
    that holds one is said to be written there too.
    """
    layers = profile_layers(profile)
    header = f"configuration failed schema validation under {root}"
    if prefix:
        header = f"the {'.'.join(prefix)} section failed schema validation under {root}"
    if layers:
        header += f" (profile chain: {' -> '.join(layers)})"
    lines = [header + ":"]
    try:
        from ._provenance import index_origins, nearest_keys
        from .schema._removed import REMOVED_KEYS

        origins = index_origins(root, layers)
    except Exception:  # noqa: BLE001 (never let the explainer mask the real error)
        return f"{header}:\n{exc}"

    for err in exc.errors():
        # One location, two spellings, and neither is pydantic's: `where` is the key as the file writes
        # it and as the side-car index is keyed, `dotted` the same key with a list position as a plain
        # segment, the spelling the in-memory keys and the removed-key table use.
        segments = _yaml_segments((*prefix, *err["loc"]), origins)
        where = _bracketed(segments, origins)
        dotted = ".".join(str(segment) for segment in segments)
        lines.append(f"\n  {where}")
        origin = origins.get(where)
        if _within(dotted, in_memory) or (origin is None and _holds(dotted, in_memory)):
            lines.append(f"      at {_IN_MEMORY_ORIGIN}")
        elif origin is not None:
            lines.append(f"      at {origin.location(root)}")
        elif err["type"] == "missing" and (block := _block_origin(segments[:-1], origins)) is not None:
            # A required key nobody wrote, inside a block somebody did: the block is where to add it.
            lines.append(f"      (required, and the block at {block.location(root)} does not write it)")
        elif (block := _block_origin(segments, origins)) is not None:
            # A rule over a whole block, a list item above all: a rig-level validator, or a tag no member
            # of the union takes. An item has no line of its own in the index, and saying "not written"
            # about a rig written in the operator's own layer sends them looking for a default.
            lines.append(f"      (a rule over the block written at {block.location(root)})")
        else:
            # No origin means the key appears in no YAML: a schema default that failed a
            # cross-field rule, or a required field nobody wrote.
            lines.append("      (not written in any YAML: a default or a cross-field rule)")
        lines.append(f"      {err['msg']}")
        removed = None
        if err["type"] == "extra_forbidden":
            # A `*` segment in a removed key stands for one map key, such as a camera id.
            parts = dotted.split(".")
            removed = REMOVED_KEYS.get(dotted) or next(
                (text for key, text in REMOVED_KEYS.items()
                 if "*" in key and len(key.split(".")) == len(parts)
                 and all(k in ("*", p) for k, p in zip(key.split("."), parts))),
                None,
            )
        if removed is not None:
            lines.append(f"      removed on purpose: {removed}")
        elif err["type"] == "extra_forbidden" and err["loc"]:
            siblings = [
                key.rsplit(".", 1)[-1]
                for key in origins
                if key.rsplit(".", 1)[0] == where.rsplit(".", 1)[0] and key != where
            ]
            for suggestion in nearest_keys(str(err["loc"][-1]), siblings):
                lines.append(f"      did you mean: {suggestion}?")
    lines.append(
        "\nEvery key above is rejected on purpose (unknown keys are never ignored). "
        "`python -m src.config` re-runs this check without booting anything."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal: section loaders
# ---------------------------------------------------------------------------

def _load_camera_section(root: Path, profile: str) -> dict[str, Any]:
    section: dict[str, Any] = {}
    for filename, key in zip(_CAMERA_FILES, _CAMERA_KEYS):
        path = root / "camera" / filename
        data = _load_yaml_with_profile(path, profile, required=True)
        section[key] = _require_top_key(data, path, key)
    hand_eye_path = root / "camera" / "hand_eye.yaml"
    hand_eye = _load_optional_section(hand_eye_path, "hand_eye", profile)
    if hand_eye is not None:
        section["hand_eye"] = hand_eye
    return section


def _load_models_section(root: Path, profile: str) -> dict[str, Any]:
    """Auto-discover every ``models/*.yaml`` and merge their top-level keys.

    Each file contributes one or more named model configs at its top level (``handdetect:`` and
    ``gesturedetect:`` in ``hand.yaml``). A key defined in two files is rejected, so the merge stays
    unambiguous.
    """
    models_dir = root / "models"
    if not models_dir.is_dir():
        raise ConfigError(f"models directory not found: {models_dir}")

    merged: dict[str, Any] = {}
    seen_in: dict[str, Path] = {}
    base_paths = sorted(
        p for p in models_dir.glob("*.yaml") if not _is_profile_overlay_file(p)
    )

    for path in base_paths:
        data = _load_yaml_with_profile(path, profile, required=True)
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected mapping at top level")
        for key, value in data.items():
            if key in merged:
                raise ConfigError(
                    f"duplicate model key {key!r} found in {path} "
                    f"(already defined in {seen_in[key]})"
                )
            merged[key] = value
            seen_in[key] = path

    # Profile-only model files (models/edge.prod.yaml) have no non-profile base counterpart. They
    # are gathered per layer in chain order and merged by stem first, so a later layer refines the
    # same profile-only file (models/foo.sim.yaml -> models/foo.ur3e.yaml) instead of colliding with
    # it. A duplicate model key across two different files still raises.
    base_stems = {p.stem for p in base_paths}
    extra: dict[str, tuple[Path, dict[str, Any]]] = {}
    for layer in profile_layers(profile):
        for overlay_path in sorted(models_dir.glob(f"*.{layer}.yaml")):
            stem = _base_stem_for_overlay(overlay_path, layer)
            if stem in base_stems:
                continue
            data = _load_yaml(overlay_path)
            if not isinstance(data, dict):
                raise ConfigError(f"{overlay_path}: expected mapping at top level")
            previous = extra.get(stem)
            extra[stem] = (
                overlay_path, data if previous is None else _deep_merge(previous[1], data),
            )
    for overlay_path, data in extra.values():
        for key, value in data.items():
            if key in merged:
                raise ConfigError(
                    f"duplicate model key {key!r} found in {overlay_path} "
                    f"(already defined in {seen_in[key]})"
                )
            merged[key] = value
            seen_in[key] = overlay_path

    if not merged:
        raise ConfigError(f"no model configurations found under {models_dir}")
    return merged


def _load_model_keys(root: Path, profile: str, wanted: tuple[str, ...]) -> dict[str, Any]:
    """The ``models/`` keys one section reads, with an unreadable file tolerated where it cannot matter.

    The files are walked as :func:`_load_models_section` walks them, every base file with its overlays
    and then the profile-only files merged by stem, but each one is parsed on its own. A file that does
    not parse, or is not a mapping, is set aside. If a key in ``wanted`` is then found nowhere, the
    section refuses and names every file set aside: an unreadable file might have held that key, and a
    schema default standing in for a value nobody could read is the silent fallback this loader
    refuses everywhere else. A key in ``wanted`` defined twice is refused as the whole-tree load refuses
    it; a duplicate among keys this section does not read is not this section's fault.

    A top-level key the schema does not know is refused when it sits in a file that defines one of
    ``wanted``, or when one of ``wanted`` is missing and the unknown key may be that key spelled
    wrong. An unknown key anywhere else belongs to another part of ``models`` and is left to
    :func:`load_config`, which refuses it. A refused key is named at each file that writes it, an
    overlay or one layer of a profile-only file included, not at the file its data was merged into.

    What a file that could not be read would have added is invisible here, a second definition of
    a wanted key included. :func:`load_config` refuses such a tree, and so does any section missing
    a key; a section whose keys were all found reads the definitions it could see.
    """
    models_dir = root / "models"
    if not models_dir.is_dir():
        raise ConfigError(f"models directory not found: {models_dir}")

    # One entry per base file merged with its overlays, and per profile-only stem merged across its
    # layers. The third item says which file wrote each top-level key: the merged data keeps no file,
    # and a key is reported where it is written. For a base file it is worked out only when needed.
    readable: list[tuple[Path, dict[str, Any], dict[str, list[Path]] | None]] = []
    set_aside: list[str] = []
    base_paths = sorted(p for p in models_dir.glob("*.yaml") if not _is_profile_overlay_file(p))
    for path in base_paths:
        try:
            data = _load_yaml_with_profile(path, profile, required=True)
        except ConfigError as exc:
            set_aside.append(str(exc))
            continue
        if not isinstance(data, dict):
            set_aside.append(f"{path}: expected mapping at top level")
            continue
        readable.append((path, data, None))

    base_stems = {p.stem for p in base_paths}
    extra: dict[str, tuple[Path, dict[str, Any], dict[str, list[Path]]]] = {}
    unreadable_stems: set[str] = set()
    for layer in profile_layers(profile):
        for overlay_path in sorted(models_dir.glob(f"*.{layer}.yaml")):
            stem = _base_stem_for_overlay(overlay_path, layer)
            if stem in base_stems or stem in unreadable_stems:
                continue
            try:
                data = _load_yaml(overlay_path)
            except ConfigError as exc:
                set_aside.append(str(exc))
                unreadable_stems.add(stem)
                extra.pop(stem, None)
                continue
            if not isinstance(data, dict):
                set_aside.append(f"{overlay_path}: expected mapping at top level")
                unreadable_stems.add(stem)
                extra.pop(stem, None)
                continue
            previous = extra.get(stem)
            writers: dict[str, list[Path]] = {} if previous is None else previous[2]
            for key in data:
                writers.setdefault(str(key), []).append(overlay_path)
            extra[stem] = (
                overlay_path, data if previous is None else _deep_merge(previous[1], data), writers,
            )
    readable.extend(extra.values())

    known = set(ModelsConfig.model_fields)
    found: dict[str, Any] = {}
    seen_in: dict[str, Path] = {}
    unknown: list[tuple[Path, str, bool]] = []
    for path, data, written_by in readable:
        defines_a_wanted_key = any(key in data for key in wanted)
        strangers = [str(key) for key in data if key not in known]
        if strangers and written_by is None:
            written_by = _top_key_writers(path, profile)
        for key in strangers:
            unknown.extend(
                (writer, key, defines_a_wanted_key) for writer in (written_by or {}).get(key, [path])
            )
        for key in wanted:
            if key not in data:
                continue
            if key in found:
                raise ConfigError(
                    f"duplicate model key {key!r} found in {path} "
                    f"(already defined in {seen_in[key]})"
                )
            found[key] = data[key]
            seen_in[key] = path

    missing = [key for key in wanted if key not in found]
    blamed = [(path, key) for path, key, in_a_file_read in unknown if in_a_file_read or missing]
    if blamed:
        lines = []
        for path, key in blamed:
            near = difflib.get_close_matches(key, sorted(known), n=1)
            lines.append(f"{path}: {key!r}" + (f" (did you mean {near[0]!r}?)" if near else ""))
        raise ConfigError(
            "a models file carries a top-level key the schema does not know, in a file this section "
            "reads or while one of the section's own keys is missing, so it may be that key spelled "
            "wrong. load_config refuses it too:\n  " + "\n  ".join(lines)
        )
    if set_aside and missing:
        raise ConfigError(
            f"a file under {models_dir} could not be read, and this section needs "
            f"{', '.join(missing)}, which no readable file defines: the unreadable file may have held "
            "it, so no default stands in.\n  " + "\n  ".join(set_aside)
        )
    return found


def _top_key_writers(base: Path, profile: str) -> dict[str, list[Path]]:
    """Which files of one base file's chain write each top-level key: the base, then each layer in order.

    Called only after that chain has loaded, so every file here parses. The files are read again because
    the merge that loaded them keeps no record of which file a key came from.
    """
    writers: dict[str, list[Path]] = {}
    layers = profile_layers(profile)
    for path in (base, *(base.with_name(f"{base.stem}.{layer}{base.suffix}") for layer in layers)):
        if not path.exists():
            continue
        data = _load_yaml(path)
        if isinstance(data, dict):
            for key in data:
                writers.setdefault(str(key), []).append(path)
    return writers


def _load_optional_section(path: Path, key: str, profile: str) -> Any | None:
    """Load ``path`` and return ``data[key]``, or ``None`` when the file is absent."""
    data = _load_yaml_with_profile(path, profile, required=False)
    if data is None:
        return None
    return _require_top_key(data, path, key)


def _require_top_key(data: Any, path: Path, key: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a YAML mapping with top-level '{key}:'")
    if key not in data:
        raise ConfigError(f"{path} must contain a top-level '{key}:' key")
    return data[key]


def _load_yaml_with_profile(path: Path, profile: str, *, required: bool) -> Any | None:
    """Load base YAML and the optional ``.<layer>.yaml`` overlay of every profile layer, deep-merged.

    Overlay semantics:
    - a multi-layer profile (``"sim,ur3e"``) merges its overlays left-to-right, then onto the base
    - dict values merge recursively
    - scalars and lists replace base values
    - a ``None`` (``null``) overlay leaf keeps the base value, so a partial overlay can omit by null
    - the string :data:`_OVERLAY_RESET` (``"__null__"``) resets a field to ``None``, the one way a
      profile overlay can unset a base value
    """
    if required:
        base = _load_yaml(path)
    else:
        base = _load_yaml_optional(path)
    overlay = _load_profile_overlay(path, profile)
    if base is None and overlay is None:
        return None
    if base is None:
        return _resolve_overlay_resets(overlay)
    if overlay is None:
        return base
    return _resolve_overlay_resets(_deep_merge(base, overlay))


def _load_profile_overlay(path: Path, profile: str) -> Any | None:
    """The merged overlay for ``path`` across every layer of ``profile`` (left-to-right, later wins).

    Returns ``None`` when no layer contributes an overlay file for ``path``; the caller then keeps
    the base verbatim. A single-layer profile reduces to loading one ``*.<profile>.yaml``.
    """
    merged: Any | None = None
    for layer in profile_layers(profile):
        overlay_path = path.with_name(f"{path.stem}.{layer}{path.suffix}")
        if not overlay_path.exists():
            continue
        data = _load_yaml(overlay_path)
        merged = data if merged is None else _deep_merge(merged, data)
    return merged


def _resolve_overlay_resets(value: Any) -> Any:
    """Replace every :data:`_OVERLAY_RESET` sentinel with ``None`` (post-merge, profile overlays only).

    Runs on the merged result, so a sentinel that arrived there as an ordinary scalar replace
    becomes ``None`` in the final config. A tree loaded with no profile carries no sentinel, which
    makes this a no-op.
    """
    if isinstance(value, dict):
        return {key: _resolve_overlay_resets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_overlay_resets(item) for item in value]
    return None if value == _OVERLAY_RESET else value


def _validated_chain(root: Path, raw: str | None, *, source: str) -> str:
    """Validate and normalise a profile chain. Every layer of it must contribute a file.

    Fail-closed per layer: a typo'd layer (``sim,ur3``) would otherwise merge as a silent no-op and
    the cell would come up with the other robot's geometry.

    Both doors route here, the environment and the ``load_config(profile=...)`` argument, so neither
    can walk past the check. ``source`` names the thing the operator typed, and does nothing else.
    """
    profile = (raw or "").strip()
    layers = profile_layers(profile)
    if not layers:
        return ""
    for layer in layers:
        if not any(root.rglob(f"*.{layer}.yaml")):
            raise ConfigError(
                f"{source}={profile!r} selects profile layer {layer!r}, but no "
                f"'*.{layer}.yaml' overlay file exists under {root}."
            )
    return _PROFILE_SEPARATOR.join(layers)


def _active_profile(root: Path) -> str:
    """The chain named by :data:`WILLY_PROFILE`, validated through :func:`_validated_chain`."""
    return _validated_chain(root, os.environ.get(_PROFILE_ENV_VAR, ""), source=_PROFILE_ENV_VAR)


def _is_profile_overlay_file(path: Path) -> bool:
    return "." in path.stem


def _base_stem_for_overlay(path: Path, profile: str) -> str:
    suffix = f".{profile}"
    stem = path.stem
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


# ---------------------------------------------------------------------------
# Internal: YAML I/O with env-var substitution and file-scoped errors
# ---------------------------------------------------------------------------

# Byte-order marks, longest first, each carrying its own sentence.
#
# Longest first because the UTF-32 marks begin with the UTF-16 ones (`\xff\xfe` opens both the
# UTF-16-LE and the UTF-32-LE mark); testing the short ones first names the wrong encoding and sends
# the operator to the wrong menu entry.
#
# A sentence per mark rather than a name slotted into one shared template, because the templated
# version was written first and was measured to garble exactly one row: the UTF-8 mark is valid
# UTF-8, so reaching this table with it means the bad byte is somewhere later in the file, a
# different fault (something appended text in another encoding) with a different cure, and the
# shared template would have told that operator to re-save a file whose encoding is already right.
_BOM_DIAGNOSES: tuple[tuple[bytes, str], ...] = (
    (b"\x00\x00\xfe\xff",
     "It begins with a UTF-32 (big-endian) byte-order mark, so it was saved as UTF-32."),
    (b"\xff\xfe\x00\x00",
     "It begins with a UTF-32 (little-endian) byte-order mark, so it was saved as UTF-32."),
    (b"\xfe\xff",
     "It begins with a UTF-16 (big-endian) byte-order mark, so it was saved as UTF-16. "
     "Notepad calls that 'Unicode big endian'."),
    (b"\xff\xfe",
     "It begins with a UTF-16 (little-endian) byte-order mark, so it was saved as UTF-16. "
     "Notepad calls that 'Unicode' and PowerShell '-Encoding Unicode'."),
    (b"\xef\xbb\xbf",
     "It begins with a UTF-8 byte-order mark, so the file starts as UTF-8 and the byte named above "
     "is further into it: something appended or pasted text in another encoding into a UTF-8 file. "
     "Fix that one line rather than the whole file's encoding."),
)


def _not_utf8_refusal(path: Path, exc: UnicodeDecodeError) -> str:
    """Say which file is not UTF-8, what identifies it, and what to do about it.

    The guard below used to be ``except OSError`` alone, and could never fire.
    ``UnicodeDecodeError.__mro__`` is ``(UnicodeDecodeError, UnicodeError, ValueError, Exception,
    BaseException, object)``: no ``OSError`` anywhere in it. Measured cost: an operator who edits a
    YAML in Notepad and saves it as ANSI (one umlaut in a comment is enough) got a raw traceback
    naming a byte offset and no file, out of every tool in the stack, including
    ``python -m src.config``, whose entire job is to validate the tree and print a refusal. The
    same tree broken the ordinary way (bad indentation, still UTF-8) named the file correctly, so
    the machinery existed and was being walked past.

    It names the encoding, it does not use it. Retrying the read in cp1252 would make the tree load,
    and would make it load differently depending on the machine's code page: a cell coming up with
    silently different values on the operator's laptop than on the controller box. Refusing is the
    cheaper failure.
    """
    try:
        head = path.read_bytes()[:4]
    except OSError:  # the read that just succeeded cannot normally fail twice; say less, not more
        head = b""
    diagnosis = (
        "There is no byte-order mark, so the file does not identify its own encoding. A lone byte "
        "above 0x7f like this one is usually a Windows ANSI code page (cp1252), which is what "
        "Notepad's 'ANSI' and PowerShell's '-Encoding Default' write."
    )
    if b"\x00" in head:
        # Measured: a UTF-16-LE file saved without a mark reaches here (its umlaut is `\xf6\x00`,
        # and `\xf6` is an invalid UTF-8 start byte), and the cp1252 sentence above would have been
        # a confident wrong answer: it would send the operator to the one menu entry that cannot
        # help. A NUL inside the first four bytes of a YAML file is never text in any single-byte
        # encoding, so it identifies the wide encodings even with no mark to read.
        diagnosis = (
            "The first bytes contain a NUL, which no single-byte encoding produces, so this is "
            "UTF-16 or UTF-32 saved without a byte-order mark."
        )
    for bom, sentence in _BOM_DIAGNOSES:
        if head.startswith(bom):
            diagnosis = sentence
            break
    offending = f"0x{exc.object[exc.start]:02x}" if exc.start < len(exc.object) else "?"
    return (
        f"{path} is not valid UTF-8: byte {offending} at offset {exc.start} could not be decoded "
        f"({exc.reason}). Config files must be UTF-8.\n"
        f"{diagnosis}\n"
        f"Re-save the file as UTF-8 and run this again (Notepad: Save As, then set Encoding to "
        f"UTF-8; VS Code: click the encoding in the status bar, then 'Save with Encoding').\n"
        f"The loader will not retry in another encoding: a tree that loads differently depending on "
        f"the machine's code page is worse than one that refuses."
    )


def _load_yaml(path: Path) -> Any:
    if not path.exists():
        raise ConfigError(f"required config file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        # A second clause, not a wider one. `except (OSError, UnicodeDecodeError)` would have made
        # the guard fire, and would have printed "could not read <path>: 'utf-8' codec can't decode
        # byte 0xf6 in position 9": true, and still not an instruction. The two failures have
        # different cures (a permission or disk problem against a text editor's encoding menu), so
        # they get different sentences.
        raise ConfigError(_not_utf8_refusal(path, exc)) from exc

    try:
        text = _substitute_env_vars(text, path)
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"env-var substitution failed in {path}: {exc}") from exc

    try:
        # A `yaml.SafeLoader` that differs only in how it builds a string (`_TreeYamlLoader`), so this is
        # as safe as `yaml.safe_load`.
        return yaml.load(text, Loader=_TreeYamlLoader)  # noqa: S506
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML parse error in {path}: {_as_written(exc)}") from exc


def _load_yaml_optional(path: Path) -> Any | None:
    if not path.exists():
        return None
    return _load_yaml(path)


def _substitute_env_vars(text: str, path: Path) -> str:
    """Replace ``${VAR}`` and ``${VAR:-default}`` with the corresponding environment values.

    A variable that is unset and carries no default raises :class:`ConfigError`, naming ``path``.

    ``${WILLY_PROJECT_ROOT}`` is not substituted, set or not: it is the path anchor, and
    :func:`~src.config.paths.resolve_config_path` expands it in a path field. Substituted here, a
    Windows value (``D:\\dev\\...``) would land inside a double-quoted YAML string as escapes and
    stop the file from parsing. Nor is it left as written: unquoted inside ``{...}`` or ``[...]`` its
    own braces are flow indicators, and the file stopped at "while parsing a flow mapping" where any
    other ``${VAR}`` loads (measured 2026-09-24). It becomes :data:`_ANCHOR_STAND_IN`, which is plain
    text in every YAML style, and :class:`_TreeYamlLoader` turns it back into the anchor in every
    string the parse builds. So the path rule still sees the anchor as written, and neither the
    variable's value nor the anchor's braces ever meet the parser. Written with a default, the anchor
    is left as written: a path field refuses it naming the key, and so does every other field
    (:class:`~src.config.schema._base.StrictModel`); inside ``{...}`` it stops the parse instead, which
    refuses it all the same.

    A file that already holds the stand-in's text is refused, naming ``path``: the parse would turn it
    into an anchor nobody wrote.
    """
    if _ANCHOR_STAND_IN in text:
        raise ConfigError(
            f"{path} holds the text {_ANCHOR_STAND_IN}, which the loader reserves: it parses "
            f"{PROJECT_ROOT_ANCHOR} as that text and turns it back afterwards, so the file would read "
            f"an anchor it does not write. Rename whatever holds it"
        )

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name == PROJECT_ROOT_VARIABLE:
            return _ANCHOR_STAND_IN if match.group(2) is None else match.group(0)
        default = match.group(2)
        value = os.environ.get(name)
        if value is not None:
            return value
        if default is not None:
            return default
        raise ConfigError(
            f"{path}: environment variable ${{{name}}} is not set and no "
            f"default was provided (use ${{{name}:-fallback}} to supply one)"
        )

    return _ENV_VAR_RE.sub(repl, text)


#: What ``${WILLY_PROJECT_ROOT}`` is written as while a file is parsed (:func:`_substitute_env_vars`
#: says why). Letters and underscores only, so it is plain text quoted or unquoted, in a block or inside
#: ``{...}`` and ``[...]``, and it cannot resolve to a number, a boolean or a null. As long as the anchor,
#: so a parse error that quotes the line keeps its column once the anchor is put back.
_ANCHOR_STAND_IN: Final = "__" + PROJECT_ROOT_VARIABLE + "_"


class _TreeYamlLoader(yaml.SafeLoader):
    """``yaml.SafeLoader``, except that every string it builds holds the anchor where the text held its stand-in.

    Keys and values alike, since both are built as strings; nothing else about the safe loader changes.
    """


def _construct_tree_string(loader: yaml.SafeLoader, node: Any) -> str:
    return str(loader.construct_scalar(node)).replace(_ANCHOR_STAND_IN, PROJECT_ROOT_ANCHOR)


_TreeYamlLoader.add_constructor("tag:yaml.org,2002:str", _construct_tree_string)


def _as_written(exc: yaml.YAMLError) -> str:
    """A parse error as the operator's file reads: the anchor, not its stand-in, in the quoted line.

    The quoted line is cut from the text the parser held, and cut short (``__WILLY_PROJECT ...``), so
    replacing the stand-in in the finished message misses it. The anchor goes back into that text
    instead, before the message is rendered; the stand-in is as long as the anchor, so the column and
    the caret still point where they did.
    """
    for mark in (getattr(exc, "context_mark", None), getattr(exc, "problem_mark", None)):
        buffer = getattr(mark, "buffer", None)
        if mark is not None and isinstance(buffer, str):
            mark.buffer = buffer.replace(_ANCHOR_STAND_IN, PROJECT_ROOT_ANCHOR)
    return str(exc).replace(_ANCHOR_STAND_IN, PROJECT_ROOT_ANCHOR)


# ---------------------------------------------------------------------------
# Phase U11 adaptation overlay merge
# ---------------------------------------------------------------------------

def _apply_env_adaptation_overlay(raw: dict[str, Any]) -> None:
    """Merge the overlay :data:`_ADAPTATION_OVERLAY_ENV_VAR` names, if it names one.

    The whole-tree load and :func:`load_robot_section` both call this, so a set variable reaches the
    robot section the same way through either door.
    """
    overlay_path_str = os.environ.get(_ADAPTATION_OVERLAY_ENV_VAR, "").strip()
    if overlay_path_str:
        _apply_adaptation_overlay(raw, Path(overlay_path_str))


def _apply_adaptation_overlay(raw: dict[str, Any], overlay_path: Path) -> None:
    """Merge a U11 adaptation overlay YAML into ``raw`` in place.

    The overlay must be a mapping rooted at ``robot:``, and every leaf key path must be in the
    ``runtime_mutable`` allow-list that ``adaptation.discover_runtime_mutable_fields`` derives from
    the schema. An unknown or forbidden key raises :class:`ConfigError`.
    """

    if not overlay_path.exists():
        raise ConfigError(
            f"adaptation overlay not found: {overlay_path} "
            f"(env {_ADAPTATION_OVERLAY_ENV_VAR})"
        )

    # Local imports: reading config must not pull in the grasping replay package on the ordinary
    # path, where the env var is unset and this function never runs.
    from src.config.schema.robot.robot_schema import RobotConfig
    from src.robot.grasping.replay.adaptation import (
        OverlayPathError,
        discover_runtime_mutable_fields,
        validate_overlay_against_allowlist,
    )

    data = _load_yaml(overlay_path)
    if not isinstance(data, dict):
        raise ConfigError(
            f"{overlay_path}: adaptation overlay must be a YAML mapping"
        )
    if set(data.keys()) - {"robot"}:
        raise ConfigError(
            f"{overlay_path}: adaptation overlay may only contain a "
            f"top-level 'robot:' section (saw {sorted(data.keys())!r})"
        )
    if "robot" not in data:
        return  # empty / no-op overlay

    overlay_robot = data["robot"]
    if not isinstance(overlay_robot, dict):
        raise ConfigError(
            f"{overlay_path}: 'robot' section must be a mapping"
        )

    allowed = tuple(
        # Allow-list paths carry a ``robot.`` prefix. The overlay's ``robot:`` key is implicit
        # below it, so strip the prefix before matching.
        spec.dotted_key.removeprefix("robot.")
        for spec in discover_runtime_mutable_fields(RobotConfig())
    )
    forbidden = validate_overlay_against_allowlist(overlay_robot, allowed)
    if forbidden:
        raise ConfigError(
            f"{overlay_path}: adaptation overlay touches forbidden key(s): "
            f"{', '.join(sorted(forbidden))}"
        )

    base_robot = raw.get("robot", {})
    if not isinstance(base_robot, dict):
        raise ConfigError(
            f"{overlay_path}: cannot merge overlay, base 'robot' section "
            f"is not a mapping"
        )
    raw["robot"] = _deep_merge(base_robot, overlay_robot)
    # `OverlayPathError` is imported but not otherwise used here; binding it keeps the
    # unused-symbol warning off the import.
    _ = OverlayPathError
