"""Load a hand from the gripper registry.

One YAML file per hand, ``<config tree>/grippers/<model>.yaml``, with a top-level ``gripper:`` key that
validates against :class:`~src.config.schema.grippers.GripperSpec`. ``robot.gripper.model`` names the
hand a cell carries. The loader fills a named hand's widths and collision envelope from it
(``hand_numbers.py``), and the self collision guard, the planner, the deep calculator and the sim's
mount take the hand from it.

The repository's registry, ``config/grippers``, is the one authority: a hand's body, sphere map,
retract rows and evidence are committed beside the code and written from it. A deployment tree may
carry its own ``grippers/``, and for the hand its cell names that copy may repeat the repository's
file and nothing else; :func:`tree_hand_refusal` says why a tree cannot.

Every refusal is a ``ConfigError`` that names what is known. A hand that resolves to a default is the
drift the registry exists to prevent, so an unknown name, a file named after a different hand and an
alias claimed by two hands all refuse rather than pick one. A registry with one broken file refuses
every lookup: the alias checks need every file, and a registry that is only partly readable cannot say
which hand a name means.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import ValidationError

from ._registry import flat, registry_files, shown
from .loader import _DEFAULT_DATA_DIR, ConfigError, _load_yaml
from .schema.grippers import MODEL_NAME_PATTERN, GripperSpec

__all__ = ["available_grippers", "load_gripper", "tree_hand_refusal"]

_TOP_KEY = "gripper"


def _registry_dir(data_dir: str | Path | None) -> Path:
    root = Path(data_dir).resolve() if data_dir else _DEFAULT_DATA_DIR
    return root / "grippers"


def _hand_files(directory: Path) -> list[Path]:
    """The files that are hands, found the same way on every platform (:func:`._registry.registry_files`)."""
    return registry_files(directory, noun="hand")


def _read(path: Path) -> GripperSpec:
    data = _load_yaml(path)
    if not isinstance(data, dict) or _TOP_KEY not in data:
        raise ConfigError(f"{path} must contain a top-level '{_TOP_KEY}:' key")
    try:
        spec = GripperSpec.model_validate(data[_TOP_KEY])
    except ValidationError as exc:
        raise ConfigError(f"{path} is not a valid gripper description:\n{exc}") from exc
    if spec.model != path.stem:
        raise ConfigError(
            f"{path} describes the hand {spec.model!r} but is named {path.stem!r}. A registry file is "
            f"named after the hand it describes, so a name always finds exactly one file."
        )
    return spec


def _registry(data_dir: str | Path | None) -> tuple[Path, dict[str, GripperSpec]]:
    """Every hand in the registry, keyed by model, with its aliases checked across all files."""
    directory = _registry_dir(data_dir)
    if not directory.is_dir():
        raise ConfigError(f"no gripper registry at {directory}")
    specs = {spec.model: spec for spec in (_read(path) for path in _hand_files(directory))}
    owners: dict[str, list[str]] = {}
    for spec in specs.values():
        for alias in spec.aliases:
            owners.setdefault(alias, []).append(spec.model)
    for alias, models in sorted(owners.items()):
        if len(models) > 1:
            raise ConfigError(
                f"the alias {alias!r} is claimed by {sorted(models)} in {directory}; an alias names "
                f"exactly one hand"
            )
        if alias in specs:
            raise ConfigError(
                f"the alias {alias!r} of {models[0]!r} is also the model name of another hand in "
                f"{directory}"
            )
    return directory, specs


def available_grippers(data_dir: str | Path | None = None) -> list[str]:
    """The model name of every hand in the registry, sorted. Aliases are not listed."""
    return sorted(_registry(data_dir)[1])


def load_gripper(
    name: str, *, data_dir: str | Path | None = None, aliases: bool = True,
) -> GripperSpec:
    """The hand called ``name``. Raises ``ConfigError`` when there is none.

    With ``aliases`` (the default) a short name such as ``2f85``, the one existing corpora stamp a
    hand with, resolves too; that is the lookup for reading such a corpus. A cell names its hand by the
    model name, so the lookup behind ``robot.gripper.model`` passes ``aliases=False`` and refuses a
    short name there, naming the model it stands for.
    """
    if not isinstance(name, str) or not re.fullmatch(MODEL_NAME_PATTERN, name):
        raise ConfigError(f"{name!r} is not a registry name ({MODEL_NAME_PATTERN})")
    directory, specs = _registry(data_dir)
    if name in specs:
        return specs[name]
    for spec in specs.values():
        if name in spec.aliases:
            if aliases:
                return spec
            raise ConfigError(
                f"{name!r} is an alias of {spec.model!r} in {directory}, and this lookup takes model "
                f"names only: name the hand {spec.model!r}"
            )
    raise ConfigError(
        f"no gripper {name!r} in {directory}. Known hands: {', '.join(sorted(specs)) or 'none'}"
    )


def tree_hand_refusal(name: str, *, data_dir: str | Path | None) -> str | None:
    """Why the registry of the tree at ``data_dir`` cannot stand for the repository's for the hand ``name``, or ``None``.

    A hand's body, sphere map, retract rows and evidence are committed beside the code and written from
    the repository's registry file, so for the hand a cell names a deployment tree may repeat that file
    and nothing else. ``None`` when ``data_dir`` is not given or is the repository's own tree, and when
    the tree's description equals the repository's, aliases included. Otherwise one sentence naming both
    places, what differs or which side holds no such hand, why, and the fix. A registry that cannot be
    read answers with its own refusal.
    """
    if data_dir is None:
        return None
    repository_dir = _registry_dir(None)
    tree_dir = _registry_dir(data_dir)
    if tree_dir.resolve() == repository_dir.resolve():
        return None
    why = ("A hand's body, sphere map, retract rows and evidence are committed beside the code and written from the "
           "repository's registry file, so a tree may repeat that file for the hand its cell names and nothing else")
    repository_file = f"{shown(repository_dir)}/{name}.yaml"
    try:
        _, repository_specs = _registry(None)
    except ConfigError as exc:
        return str(exc)
    if not tree_dir.is_dir():
        tree_spec = None
    else:
        try:
            _, tree_specs = _registry(data_dir)
        except ConfigError as exc:
            return str(exc)
        tree_spec = tree_specs.get(name)
    ours = repository_specs.get(name)
    if ours is None:
        held = (f"{tree_dir / (name + '.yaml')} describes the hand {name}" if tree_spec is not None
                else f"the cell names the hand {name}")
        return (f"{held}, and the repository registry {shown(repository_dir)} does not. {why}: describe the hand in "
                f"{repository_file} with the scripts under scripts/grippers/, which write its body beside the code "
                f"(docs/runbooks/your_own_gripper.md), and run the chain from the repository")
    if tree_spec is None:
        where = tree_dir if tree_dir.is_dir() else tree_dir.parent
        holds = "holds no hand" if tree_dir.is_dir() else "holds no gripper registry, so no hand"
        return (f"{where} {holds} {name}, and the cell names it. {why}: copy {repository_file} into {tree_dir}")
    theirs, mine = flat(tree_spec.model_dump(mode="json")), flat(ours.model_dump(mode="json"))
    differing = [f"{key} {theirs.get(key)!r} there and {mine.get(key)!r} in the repository"
                 for key in sorted(set(theirs) | set(mine)) if theirs.get(key) != mine.get(key)]
    if not differing:
        return None
    return (f"{tree_dir / (name + '.yaml')} describes the hand {name} differently from the repository registry "
            f"{repository_file}: {'; '.join(differing)}. {why}: copy {repository_file} over the tree's file, or "
            f"measure again, correct {repository_file} and rewrite the hand's committed artefacts from it with the "
            f"scripts under scripts/grippers/ and scripts/curobo/")
