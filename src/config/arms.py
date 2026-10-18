"""Load an arm from the arm registry.

One YAML file per arm, ``<config tree>/arms/<model>.yaml``, with a top-level ``arm:`` key that validates against
:class:`~src.config.schema.arms.ArmSpec`. ``robot.<vendor>.model`` names the arm a cell drives, and this is
where the stack asks what that arm is: its joints, their envelopes, its home pose, its reach, and what a
simulator needs to stand it up.

It lives in the config layer, the bottom of the dependency stack, for the same reason the gripper registry does:
only there can every reader ask, from the safety guards and the planner through the drivers to datagen and the
sim, without any of them importing another. This module imports nothing above itself and nothing from
``scripts/``.

It describes more arms than a cell may name: each says so in ``admitted``, read by :func:`admitted_arms`,
and ur16e and ur20 say false on purpose (guide 04, section 2.4, says why).

Every refusal is a ``ConfigError`` that names what IS known, as the hand and camera registries do: an unknown
name, a file named after a different arm, and a file that is not ``<model>.yaml`` all refuse rather than pick
one, and a registry with one broken file refuses every lookup, because a registry that is only partly readable
cannot say which arm a name means.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

from pydantic import ValidationError

from src.contracts import UNSET, Maybe, chosen

from ._registry import flat, registry_files, shown
from .loader import _DEFAULT_DATA_DIR, ConfigError, _load_yaml
from .schema.arms import ArmSpec
from .schema.grippers import MODEL_NAME_PATTERN

__all__ = ["NO_DESCRIPTOR", "admitted_arms", "available_arms", "curobo_descriptor_name", "described_home", "load_arm",
           "tree_arm_refusal"]

_TOP_KEY = "arm"

#: What a reading names as the descriptor of a cell that names no hand: no file, and the key to set. It lived in the
#: sim driver until 2026-10-13, although the planner stack, the desk checklist and the doctor are the ones that print
#: it, so modules above the drivers imported a driver to read a sentence.
NO_DESCRIPTOR = "<none: robot.gripper.model is unset>"


def _registry_dir(data_dir: str | Path | None) -> Path:
    root = Path(data_dir).resolve() if data_dir else _DEFAULT_DATA_DIR
    return root / "arms"


def _read(path: Path) -> ArmSpec:
    data = _load_yaml(path)
    if not isinstance(data, dict) or _TOP_KEY not in data:
        raise ConfigError(f"{path} must contain a top-level '{_TOP_KEY}:' key")
    try:
        spec = ArmSpec.model_validate(data[_TOP_KEY])
    except ValidationError as exc:
        raise ConfigError(f"{path} is not a valid arm description:\n{exc}") from exc
    if spec.model != path.stem:
        raise ConfigError(
            f"{path} describes the arm {spec.model!r} but is named {path.stem!r}. A registry file is named "
            f"after the arm it describes, so a name always finds exactly one file."
        )
    return spec


def _read_registry(directory: Path) -> tuple[Path, dict[str, ArmSpec]]:
    if not directory.is_dir():
        raise ConfigError(f"no arm registry at {directory}")
    paths = registry_files(directory, noun="arm")
    if directory.resolve() == _registry_dir(None).resolve():
        return directory, {spec.model: spec for spec in (_read(path) for path in paths)}
    return directory, _read_tree_copies(directory, paths)


def _read_tree_copies(directory: Path, paths: list[Path]) -> dict[str, ArmSpec]:
    """A deployment tree's arm files, all of them read before one refusal names every stale copy (P0f F6-0).

    A deployment tree is normally a copytree of the shipped one, so a description that gains a required key leaves
    every older tree holding copies the schema cannot read, eight of them at once. Refusing at the first one, with
    pydantic's words alone, sent the operator round eight times and never said which file to put there. A file the
    repository also holds is a copy of it by the tree rule, so the remedy is that file; a file the repository does
    not hold has nothing to copy over it and keeps the plain refusal.
    """
    repository = _registry_dir(None)
    specs: dict[str, ArmSpec] = {}
    stale: list[str] = []
    for path in paths:
        try:
            spec = _read(path)
        except ConfigError as exc:
            if not (repository / path.name).is_file():
                raise
            stale.append(f"{path} ({_why(exc)}): copy {shown(repository / path.name)} over it")
            continue
        specs[spec.model] = spec
    if stale:
        raise ConfigError(
            f"{directory} holds arm descriptions this repository can no longer read, most likely copies made before "
            f"a description gained a key: {'; '.join(stale)}. A tree may only repeat the repository's arm files, so "
            f"copy each one over, or delete {directory}, and the tree reads through the repository's registry"
        )
    return specs


def _why(exc: ConfigError) -> str:
    """What made an arm file unreadable, one line per field for a schema refusal and the refusal itself otherwise."""
    cause = exc.__cause__
    if isinstance(cause, ValidationError):
        return "; ".join(f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in cause.errors())
    return str(exc)


@functools.lru_cache(maxsize=4)
def _repository_registry(directory: str) -> tuple[Path, dict[str, ArmSpec]]:
    """The repository's registry, read once per process per directory, because the guard asks on every evaluation.

    Only the repository's is cached, and not an exception; a deployment tree's is read fresh every time (guide 04,
    section 2.4, has the measurement and the reasons).
    """
    return _read_registry(Path(directory))


def _registry(data_dir: str | Path | None) -> tuple[Path, dict[str, ArmSpec]]:
    """Every arm in the registry, keyed by model."""
    if data_dir is None:
        return _repository_registry(str(_registry_dir(None)))
    return _read_registry(_registry_dir(data_dir))


def available_arms(data_dir: str | Path | None = None) -> list[str]:
    """The model name of every arm in the registry, sorted.

    Not the arms a cell may name. This registry describes every arm this repository holds geometry for; see
    the module docstring.
    """
    return sorted(_registry(data_dir)[1])


def admitted_arms(
    data_dir: str | Path | None = None, *, simulated: bool = False, vendor: Maybe[str] = UNSET,
) -> list[str]:
    """The arms a cell may name, sorted; with ``simulated``, only those a simulator here can also stand up.

    The gate since P0d D3f. It replaces ``UR_MODEL_KEYS``, which both model-name validators read until then. Each
    description says ``admitted`` for itself, so the gate cannot disagree with the arm it admits. ``simulated`` adds
    the sim block, because a sim cell stands its arm up from that block, and a real-only arm admitted later must not be
    selectable as ``robot.sim.robot_model``. ``vendor`` keeps only that vendor's descriptions (P0g slice 11): the
    ``ur`` block's validator asks for its own, so a second vendor's arm admitted beside the URs is never a
    ``robot.ur.model``.
    """
    specs = _registry(data_dir)[1]
    return sorted(model for model, spec in specs.items()
                  if spec.admitted and (not simulated or spec.sim is not None)
                  and (not chosen(vendor) or spec.vendor == vendor))


def described_home(model: "str | None") -> tuple[float, ...] | None:
    """The home ``model``'s description states, ``home_rad`` per joint in radians, or ``None`` for an arm nobody describes.

    Since P0e it is the home a UR drives to where the cell declares none (``robot.motion.home_deg``); until then
    that was the hand-kept ``HOME_JOINTS_DEFAULT``, which every UR description repeats: shoulder lift and wrist 1 at
    -pi/2, every other joint at zero.
    """
    if not model:
        return None
    try:
        description = load_arm(model)
    except ConfigError:
        return None
    return tuple(joint.home_rad for joint in description.joints)


def load_arm(name: str, *, data_dir: str | Path | None = None) -> ArmSpec:
    """The arm called ``name``. Raises ``ConfigError`` when there is none."""
    if not isinstance(name, str) or not re.fullmatch(MODEL_NAME_PATTERN, name):
        raise ConfigError(f"{name!r} is not a registry name ({MODEL_NAME_PATTERN})")
    directory, specs = _registry(data_dir)
    if name in specs:
        return specs[name]
    raise ConfigError(f"no arm {name!r} in {directory}. Known arms: {', '.join(sorted(specs)) or 'none'}")


def curobo_descriptor_name(model: str) -> str:
    """The cuRobo robot description this stack builds for ``model``, by name; ``ConfigError`` for an arm nobody describes.

    A function, not a field. It is one spelling of the model with no freedom in it, so hand-typed copies in several
    files would be several chances to write ``willy_ur5.yml`` into ``ur5e.yaml``, a typo no validator could catch. It
    lives here rather than in the sim driver, so the desk checklist and the planner stack ask the config layer instead
    of importing a driver to read a string.

    Case-insensitive, as its predecessor was: ``--model UR5e`` works on the command line. It refuses by description,
    not by simulator (guide 04, section 2.4, says what refusing by simulator cost).
    """
    key = model.strip().lower() if isinstance(model, str) else model
    if not isinstance(key, str) or not re.fullmatch(MODEL_NAME_PATTERN, key):
        raise ConfigError(f"{model!r} is not a registry name ({MODEL_NAME_PATTERN})")
    load_arm(key)
    return f"willy_{key}.yml"


def tree_arm_refusal(name: str, *, data_dir: str | Path | None) -> str | None:
    """Why the registry of the tree at ``data_dir`` cannot stand for the repository's for the arm ``name``, or ``None``.

    The same authority rule the hand and camera registries carry: the repository's registry is the one that was
    measured, and a deployment tree may repeat its file and nothing else. ``None`` when no tree is given, when
    the tree IS the repository's, when the tree holds no arm registry, and when the tree's description equals
    the repository's key by key: a deployment tree is normally a copy of the shipped one (guide 04, section 2.4).
    """
    if data_dir is None:
        return None
    repository_dir = _registry_dir(None)
    tree_dir = _registry_dir(data_dir)
    if tree_dir.resolve() == repository_dir.resolve():
        return None
    why = ("An arm's geometry is read from the repository's registry file, whose numbers its collision bundles, "
           "sphere maps, retract rows and evidence files were all measured through, so a tree may repeat that "
           "file for the arm its cell names and nothing else")
    repository_file = shown(repository_dir / f"{name}.yaml")
    repository_specs = _registry(None)[1]
    tree_spec = None
    if tree_dir.is_dir():
        tree_spec = _registry(data_dir)[1].get(name)
    ours = repository_specs.get(name)
    if ours is None:
        held = (f"{tree_dir / (name + '.yaml')} describes the arm {name}" if tree_spec is not None
                else f"the cell names the arm {name}")
        return (f"{held}, and the repository registry {shown(repository_dir)} does not. {why}: describe the arm "
                f"in {repository_file}, beside the geometry that was measured for it, and run the cell from the "
                f"repository")
    if tree_spec is None:
        if not tree_dir.is_dir():
            # A tree with no arms/ at all reads through the repository's registry, as a tree with no cameras/
            # does: an arm description fills none of a tree's own numbers at load, so there is no tree copy a
            # load could have read instead.
            return None
        return (f"{tree_dir} holds no arm {name}, and the cell names it. {why}: copy {repository_file} into "
                f"{tree_dir}")
    theirs, mine = flat(tree_spec.model_dump(mode="json")), flat(ours.model_dump(mode="json"))
    differing = [f"{key} {theirs.get(key)!r} there and {mine.get(key)!r} in the repository"
                 for key in sorted(set(theirs) | set(mine)) if theirs.get(key) != mine.get(key)]
    if not differing:
        return None
    return (f"{tree_dir / (name + '.yaml')} describes the arm {name} differently from the repository registry "
            f"{repository_file}: {'; '.join(differing)}. {why}: copy {repository_file} over the tree's file, or "
            f"describe a new arm under its own name in the repository")
