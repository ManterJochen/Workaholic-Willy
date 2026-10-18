"""Recursive mapping-merge used by the config loader.

Contract: a ``None`` overlay leaf keeps the base value, so a partial overlay can omit by null
without wiping fields. A profile overlay that must reset a base field to ``None`` writes the reset
sentinel defined in ``src/config/loader.py`` instead, so a reset never reaches this merge as
``None``.
"""

from __future__ import annotations

from typing import Any


def _deep_merge(base: Any, overlay: Any) -> Any:
    """Recursively merge ``overlay`` into ``base`` (returns a new value)."""
    if overlay is None:
        return base
    if isinstance(base, dict) and isinstance(overlay, dict):
        out: dict[str, Any] = dict(base)
        for key, value in overlay.items():
            if key in out:
                out[key] = _deep_merge(out[key], value)
            else:
                out[key] = value
        return out
    return overlay
