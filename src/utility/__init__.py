"""Shared runtime helpers.

The small helpers are imported eagerly. The torch-dependent ones in
:mod:`src.utility.device` are named in ``_DEVICE_EXPORTS`` and resolved on first
attribute access, so importing this package, or a leaf of it such as
:mod:`src.utility.unit_scaling`, never pulls in the optional ML dependencies.
"""

from __future__ import annotations

from .io import atomic_write_text, dump_json, load_json
from .log_cfg import create_logger
from .paths import debug_dir, ensure_dir, logs_dir, project_root, rotate_files
from .timing import timed
from .unit_scaling import unit_scaling
from .vision import bgr_to_rgb, rgb_to_bgr

_DEVICE_EXPORTS = {
    "get_device",
    "is_cuda",
    "move_inputs_to_device",
    "resolve_torch_dtype",
}


def __getattr__(name: str):
    if name in _DEVICE_EXPORTS:
        from . import device

        value = getattr(device, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "atomic_write_text",
    "bgr_to_rgb",
    "create_logger",
    "debug_dir",
    "dump_json",
    "ensure_dir",
    "get_device",
    "is_cuda",
    "load_json",
    "logs_dir",
    "move_inputs_to_device",
    "project_root",
    "resolve_torch_dtype",
    "rgb_to_bgr",
    "rotate_files",
    "timed",
    "unit_scaling",
]
