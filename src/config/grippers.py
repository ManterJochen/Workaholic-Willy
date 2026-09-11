"""Load a hand from the gripper registry.

One YAML file per hand, ``<config tree>/grippers/<model>.yaml``, with a top-level ``gripper:`` key that
validates against :class:`~src.config.schema.grippers.GripperSpec`. ``robot.gripper.model`` names the
hand a cell carries. Nothing on the pick path reads the registry.

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

from .loader import _DEFAULT_DATA_DIR, ConfigError, _load_yaml
from .schema.grippers import MODEL_NAME_PATTERN, GripperSpec

__all__ = ["available_grippers", "load_gripper"]

_TOP_KEY = "gripper"


def _registry_dir(data_dir: str | Path | None) -> Path:
    root = Path(data_dir).resolve() if data_dir else _DEFAULT_DATA_DIR
    return root / "grippers"


def _hand_files(directory: Path) -> list[Path]:
    """The files that are hands, found the same way on every platform.

    A glob for ``*.yaml`` ignores ``.yml`` everywhere, matches ``.YAML`` only where the filesystem is
    case-insensitive, and reads ``robotiq_2f85.sim.yaml`` as a hand named after a stem no model can
    match. So every file that looks like YAML must be named ``<model>.yaml`` exactly, and any other one
    is refused by name. A file that is not YAML at all, a README for instance, is not a hand.
    """
    hands: list[Path] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".yaml", ".yml"):
            continue
        if path.suffix != ".yaml" or not re.fullmatch(MODEL_NAME_PATTERN, path.stem):
            raise ConfigError(
                f"{path.name} in {directory} is not a hand the registry can read: a hand is one file "
                f"named <model>.yaml, with the suffix in lower case, a registry name "
                f"({MODEL_NAME_PATTERN}) as its stem, and no profile layer"
            )
        hands.append(path)
    return hands


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
