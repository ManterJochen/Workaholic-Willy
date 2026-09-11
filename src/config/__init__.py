"""Workaholic-Willy configuration package.

The schema classes are imported eagerly. The YAML-backed loader helpers arrive
through ``__getattr__`` instead, so a schema-only import does not pull in the
optional YAML dependency.
"""

from __future__ import annotations

from .schema import (
    AppConfig,
    CameraConfig,
    ModelsConfig,
    RobotConfig,
    RuntimeConfig,
)

_LOADER_EXPORTS = {
    "ConfigError",
    "load_camera_section",
    "load_config",
    "load_perception_section",
    "load_robot_config",
    "load_robot_section",
    "load_speech_section",
    "reload_config",
}

#: The capability noun of this package. Lazy for the same reason as the loader helpers: it
#: reaches YAML on first use, and a schema-only import must not need the optional dependency.
_TREE_EXPORTS = {"ConfigTree", "LoadedTree", "default_data_dir"}


def __getattr__(name: str):
    if name in _LOADER_EXPORTS:
        from . import loader

        value = getattr(loader, name)
        globals()[name] = value
        return value
    if name in _TREE_EXPORTS:
        from . import tree

        value = getattr(tree, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AppConfig",
    "CameraConfig",
    "ConfigError",
    "ConfigTree",
    "LoadedTree",
    "ModelsConfig",
    "RobotConfig",
    "RuntimeConfig",
    "default_data_dir",
    "load_camera_section",
    "load_config",
    "load_perception_section",
    "load_robot_config",
    "load_robot_section",
    "load_speech_section",
    "reload_config",
]
