"""Load a camera body from the camera registry.

One YAML file per camera model, ``<config tree>/cameras/<model>.yaml``, with a top-level ``camera:`` key that
validates against :class:`~src.config.schema.cameras.CameraBodySpec`. ``camera.cameras.rigs[<id>].body.model``
names the camera a wrist rig carries, and the planner, the exact mesh guard and the self filter place its housing on
the flange from the rig's calibration.

The repository's registry, ``config/cameras``, is the one authority, as it is for grippers: a housing's numbers were
read off the drawing the file names. A deployment tree may carry its own ``cameras/``, and for a camera its rigs name
that copy may repeat the repository's file and nothing else; :func:`tree_camera_refusal` says why a tree cannot.

Every refusal is a ``ConfigError`` that names what is known: an unknown name lists the known cameras, a file named
after another camera is refused, and a registry with one broken file refuses every lookup, because a registry that is
only partly readable cannot say what a name means.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import ValidationError

from ._registry import flat, registry_files, shown
from .loader import _DEFAULT_DATA_DIR, ConfigError, _load_yaml
from .schema.cameras import CameraBodySpec
from .schema.grippers import MODEL_NAME_PATTERN

__all__ = ["available_cameras", "load_camera", "tree_camera_refusal"]

_TOP_KEY = "camera"


def _registry_dir(data_dir: str | Path | None) -> Path:
    root = Path(data_dir).resolve() if data_dir else _DEFAULT_DATA_DIR
    return root / "cameras"


def _read(path: Path) -> CameraBodySpec:
    data = _load_yaml(path)
    if not isinstance(data, dict) or _TOP_KEY not in data:
        raise ConfigError(f"{path} must contain a top-level '{_TOP_KEY}:' key")
    try:
        spec = CameraBodySpec.model_validate(data[_TOP_KEY])
    except ValidationError as exc:
        raise ConfigError(f"{path} is not a valid camera description:\n{exc}") from exc
    if spec.model != path.stem:
        raise ConfigError(
            f"{path} describes the camera {spec.model!r} but is named {path.stem!r}. A registry file is "
            f"named after the camera it describes, so a name always finds exactly one file."
        )
    return spec


def _registry(data_dir: str | Path | None) -> tuple[Path, dict[str, CameraBodySpec]]:
    """Every camera in the registry, keyed by model."""
    directory = _registry_dir(data_dir)
    if not directory.is_dir():
        raise ConfigError(f"no camera registry at {directory}")
    return directory, {spec.model: spec for spec in (_read(path) for path in registry_files(directory, noun="camera"))}


def available_cameras(data_dir: str | Path | None = None) -> list[str]:
    """The model name of every camera in the registry, sorted."""
    return sorted(_registry(data_dir)[1])


def load_camera(name: str, *, data_dir: str | Path | None = None) -> CameraBodySpec:
    """The camera called ``name``. Raises ``ConfigError`` when there is none."""
    if not isinstance(name, str) or not re.fullmatch(MODEL_NAME_PATTERN, name):
        raise ConfigError(f"{name!r} is not a registry name ({MODEL_NAME_PATTERN})")
    directory, specs = _registry(data_dir)
    if name in specs:
        return specs[name]
    raise ConfigError(
        f"no camera {name!r} in {directory}. Known cameras: {', '.join(sorted(specs)) or 'none'}"
    )


def tree_camera_refusal(name: str, *, data_dir: str | Path | None) -> str | None:
    """Why the registry of the tree at ``data_dir`` cannot stand for the repository's for the camera ``name``, or ``None``.

    A camera body's numbers are read from the repository's registry file, so for a camera its rigs name a deployment
    tree may repeat that file and nothing else. ``None`` when ``data_dir`` is not given or is the repository's own
    tree, when the tree holds no camera registry and the repository describes the camera, and when the tree's
    description equals the repository's. Otherwise one sentence naming both places, what differs or which side holds
    no such camera, why, and the fix. A registry that cannot be read answers with its own refusal.

    A tree without ``cameras/`` is read through the repository's registry, unlike a tree without ``grippers/``: a hand
    fills a tree's numbers at load and a camera body fills none, so there is no tree copy a load could have read
    instead.
    """
    if data_dir is None:
        return None
    repository_dir = _registry_dir(None)
    tree_dir = _registry_dir(data_dir)
    if tree_dir.resolve() == repository_dir.resolve():
        return None
    why = ("A camera body's numbers are read from the repository's registry file, which names the drawing they were "
           "read off, so a tree may repeat that file for the camera its rigs name and nothing else")
    repository_file = f"{shown(repository_dir)}/{name}.yaml"
    try:
        _, repository_specs = _registry(None)
    except ConfigError as exc:
        return str(exc)
    ours = repository_specs.get(name)
    tree_spec = None
    if tree_dir.is_dir():
        try:
            _, tree_specs = _registry(data_dir)
        except ConfigError as exc:
            return str(exc)
        tree_spec = tree_specs.get(name)
    if ours is None:
        held = (f"{tree_dir / (name + '.yaml')} describes the camera {name}" if tree_spec is not None
                else f"a rig names the camera {name}")
        return (f"{held}, and the repository registry {shown(repository_dir)} does not. {why}: describe the camera in "
                f"{repository_file}, every number from its vendor drawing and the drawing's revision in source")
    if tree_spec is None:
        return None
    theirs, mine = flat(tree_spec.model_dump(mode="json")), flat(ours.model_dump(mode="json"))
    differing = [f"{key} {theirs.get(key)!r} there and {mine.get(key)!r} in the repository"
                 for key in sorted(set(theirs) | set(mine)) if theirs.get(key) != mine.get(key)]
    if not differing:
        return None
    return (f"{tree_dir / (name + '.yaml')} describes the camera {name} differently from the repository registry "
            f"{repository_file}: {'; '.join(differing)}. {why}: copy {repository_file} over the tree's file, or add "
            f"the camera to the repository registry")
