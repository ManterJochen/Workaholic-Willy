"""An arm description read by path, as the build scripts need it: ``config/arms/<model>.yaml`` (P0f F7).

Standard library and PyYAML only, and Python 3.10, so the venv, the cuRobo environment (where ``src.config``
cannot be imported) and Isaac read the same numbers from the same file. No schema runs here: the description is
validated where it lives, ``src/config/arms.py``, and ``tests/scripts/arms/test_the_arm_file_reads_as_the_registry_does.py``
holds every answer below to what ``load_arm`` answers, bit for bit. Each answer is the value the file holds, in the
type the file holds it, because the scripts that read it write bytes that are hashed.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "ARMS", "ArmFileError", "chain_yaw_deg", "collision_links", "described", "dh_rows", "flange_link", "joint_names",
    "limits_deg", "planner_limits_deg", "position_limit_clip_rad", "read", "retract_anchor", "retract_anchor_source",
]

#: The repository's arm registry, the one the committed artefacts were measured through.
ARMS = Path(__file__).resolve().parents[2] / "config" / "arms"


class ArmFileError(ValueError):
    """An arm this registry does not describe, or a description a build script cannot read."""


def described() -> list[str]:
    """Every arm the registry holds a file for, sorted."""
    return sorted(path.stem for path in ARMS.glob("*.yaml"))


@functools.lru_cache(maxsize=None)
def read(model: str) -> dict[str, Any]:
    """The ``arm:`` mapping of ``model``'s file, as PyYAML reads it."""
    path = ARMS / f"{model}.yaml"
    if not path.is_file():
        raise ArmFileError(f"no arm {model!r} in {ARMS}; described: {', '.join(described())}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("arm"), dict):
        raise ArmFileError(f"{path} holds no top-level 'arm:' mapping")
    return document["arm"]


def _joints(model: str) -> list[dict[str, Any]]:
    return list(read(model)["joints"])


def joint_names(model: str) -> list[str]:
    """Each joint's name, base to flange, in the order a joint vector arrives in."""
    return [joint["name"] for joint in _joints(model)]


def limits_deg(model: str) -> list[tuple[float, float]]:
    """The envelope the guard enforces per joint, degrees: the arm's own, wider than the planner's where it differs."""
    return [(joint["limit_deg"][0], joint["limit_deg"][1]) for joint in _joints(model)]


def dh_rows(model: str) -> list[tuple[float, float, float]]:
    """``(a, d, alpha)`` per joint, metres and radians, base to flange."""
    return [(joint["a_m"], joint["d_m"], joint["alpha_rad"]) for joint in _joints(model)]


def collision_links(model: str) -> list[tuple[str, str, int]]:
    """``(part, link, frame)`` base to flange: the frame is the row's number, and the flange frame is the last."""
    return [(joint["part"], joint["link"], frame) for frame, joint in enumerate(_joints(model), start=1)]


def chain_yaw_deg(model: str) -> int:
    """The turn about z from the description's root to the frame the DH chain starts in: 0 or 180."""
    return read(model)["base"]["chain_yaw_deg"]


def flange_link(model: str) -> str:
    """The link a hand hangs from, the end of the chain: what a body on the flange is added under."""
    return read(model)["flange"]["link"]


def planner_limits_deg(model: str) -> list[tuple[float, float]]:
    """The envelope the trajectory planner is given per joint, degrees, before its clip."""
    return [(joint["planner_limit_deg"][0], joint["planner_limit_deg"][1]) for joint in _joints(model)]


def position_limit_clip_rad(model: str) -> float:
    """What cuRobo narrows each end of every joint's envelope by when it loads a descriptor."""
    return read(model)["planner"]["position_limit_clip_rad"]


def retract_anchor(model: str) -> "tuple[float, ...] | None":
    """Where the retract search starts, one value per joint, or ``None`` for an arm with no anchor."""
    values: list[Any] = [joint.get("retract_anchor_rad") for joint in _joints(model)]
    return None if all(value is None for value in values) else tuple(values)


def retract_anchor_source(model: str) -> "str | None":
    """Where the anchor came from, as every retract row of the arm records it."""
    return read(model).get("retract_anchor_source")
