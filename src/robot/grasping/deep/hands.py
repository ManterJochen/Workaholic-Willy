"""Which numbers a hand's name means, for the network's conditioning vector.

Two sources describe a jaw, and this module makes them one question:

* the gripper registry, ``config/grippers/<model>.yaml``, one file per hand this repository owns, read
  through ``src.config.grippers``;
* ``JAW_GEOMETRY`` in ``net/gripper.py``, the three procedural jaws a corpus was labelled with so the conditioning
  input varies. They are not grippers anybody owns, and they stay out of the registry.

The registry answers first, aliases included. Every corpus and artifact written before the registry is stamped
``2f85``, and that stamp is the Robotiq 2F-85's alias, so it resolves to the 2F-85's file. ``JAW_GEOMETRY`` answers
only a name the registry does not know. A name both sources know with different numbers is refused, because a file
and a copy describing two different jaws under one stamp is the drift the registry exists to end, and a name
neither source knows is refused with every name both do know.

It lives outside ``net/`` because ``net/`` may not import the geometry or dataset layers. Resolving is cached per
name and tree: the trainer asks once per sample, and the registry is a directory of files.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch

from src.config.grippers import available_grippers, load_gripper
from src.robot.grasping.deep.net.gripper import JAW_GEOMETRY, swept_volume_vector

__all__ = ["ResolvedHand", "canonical_hand", "hand_vector", "resolve_hand"]


@dataclass(frozen=True, slots=True)
class ResolvedHand:
    """One hand's name resolved: whose numbers they are, and the eight the conditioning vector takes."""

    #: The registry's model name, or the procedural jaw's own name.
    model: str
    #: ``"registry"`` or ``"procedural"``.
    source: str
    #: The keyword arguments of ``swept_volume_vector``. Read-only, because a resolved hand is shared.
    numbers: Mapping[str, float]


def resolve_hand(name: str, *, data_dir: "str | Path | None" = None) -> ResolvedHand:
    """The hand ``name`` means: the registry's, by model or alias, else a procedural jaw's. Raises ``ValueError``.

    ``data_dir`` is the config tree whose registry answers; ``None`` is the repository's.
    """
    return _resolve(str(name), None if data_dir is None else str(Path(data_dir).resolve()))


def hand_vector(
    name: str, *, data_dir: "str | Path | None" = None, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """The ``(14,)`` conditioning vector of the hand ``name`` means (see :func:`resolve_hand`)."""
    return swept_volume_vector(**resolve_hand(name, data_dir=data_dir).numbers, dtype=dtype)


def canonical_hand(name: str, *, data_dir: "str | Path | None" = None) -> str:
    """The one name of the hand ``name`` means: ``2f85`` is ``robotiq_2f85``."""
    return resolve_hand(name, data_dir=data_dir).model


@functools.lru_cache(maxsize=None)
def _resolve(name: str, data_dir: str | None) -> ResolvedHand:
    specs = [load_gripper(model, data_dir=data_dir) for model in available_grippers(data_dir)]
    for spec in specs:
        if name != spec.model and name not in spec.aliases:
            continue
        jaw = spec.jaw
        numbers = {
            "aperture_mm": jaw.aperture_mm, "min_width_mm": jaw.min_width_mm,
            "finger_ahead_mm": jaw.finger_ahead_mm, "finger_behind_mm": jaw.finger_behind_mm,
            "finger_thickness_mm": jaw.finger_thickness_mm, "finger_width_mm": jaw.finger_width_mm,
            "pad_span_mm": jaw.pad_length_mm, "friction_coefficient": jaw.friction_coefficient,
        }
        for known in (spec.model, *spec.aliases):
            procedural = JAW_GEOMETRY.get(known)
            if procedural is None or dict(procedural) == numbers:
                continue
            differing = sorted(key for key in numbers if procedural.get(key) != numbers[key])
            raise ValueError(
                f"gripper stamp {name!r} is the registry's {spec.model!r}, and JAW_GEOMETRY[{known!r}] describes a "
                f"different jaw under that name ({', '.join(differing)}). Refusing to condition on either: a file "
                f"and its copy have to describe one hand."
            )
        return ResolvedHand(model=spec.model, source="registry", numbers=MappingProxyType(numbers))
    if name in JAW_GEOMETRY:
        return ResolvedHand(model=name, source="procedural", numbers=MappingProxyType(dict(JAW_GEOMETRY[name])))
    raise ValueError(
        f"unknown gripper stamp {name!r}; the gripper registry knows "
        f"{', '.join(spec.model for spec in specs) or 'no hand'} and JAW_GEOMETRY knows "
        f"{', '.join(sorted(JAW_GEOMETRY))}. Refusing to guess, because a wrong gripper vector trains a "
        f"relationship that does not exist."
    )
