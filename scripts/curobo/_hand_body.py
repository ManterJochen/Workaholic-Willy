"""The body link a cell sends for its hand, read from the committed sphere map. Standard library plus PyYAML.

The descriptor check and the fidelity probe run under the cuRobo interpreter, where ``src.robot...`` cannot be
imported, and both have to send exactly the body a cell's planner sends
(``src/robot/safety/planning/body_link.py``). So the map reading and its refusals live here, once, and the placement
arithmetic itself stays in ``_curobo_body_links.hand_body_link``, which this and the cell side both call.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
PLANNING = REPO / "src" / "robot" / "safety" / "planning"

#: The Isaac cell's declared tool frame (robot.sim.yaml), which places the hand model on the identity.
SIM_TOOL_ROTATION_XYZW = (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)

if str(PLANNING) not in sys.path:
    # `_curobo_body_links` imports its two siblings by their plain names when it is loaded outside the package.
    sys.path.insert(0, str(PLANNING))


class HandBodyError(ValueError):
    """The hand cannot be sent as a body link as described."""


def load_by_path(path: Path) -> ModuleType:
    """A standard library module beside the planner, loaded by path: the cuRobo interpreter has no Willy package."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def placement(rotation_xyzw: "tuple[float, ...]" = SIM_TOOL_ROTATION_XYZW) -> Any:
    """Where a declared tool frame puts the hand model, by the one derivation (``_hand_placement``)."""
    return load_by_path(PLANNING / "_hand_placement.py").HandPlacement.from_quaternion_xyzw(tuple(rotation_xyzw))


def hand_body(
    hand: str, *, coupling_mm: "float | None", rotation_xyzw: "tuple[float, ...]" = SIM_TOOL_ROTATION_XYZW,
) -> dict[str, Any]:
    """The body link for ``hand``: its committed map, placed by the declared frame and the plates between.

    The plate rules are the cell's: a mounting face map needs a coupling, and a flange map refuses one, because the map
    already sits where the hand is bolted.
    """
    from _curobo_body_links import hand_body_link  # type: ignore[import-not-found]

    path = PLANNING / "robot" / f"{hand}_gripper_spheres.yml"
    if not path.is_file():
        raise HandBodyError(f"no committed sphere map for {hand!r} at {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    origin = str((document.get("_provenance") or {}).get("origin"))
    spheres = (document.get("collision_spheres") or {}).get("tool0") or []
    if not spheres:
        raise HandBodyError(f"{path.name} holds no spheres under collision_spheres.tool0")
    if origin == "mounting_face" and coupling_mm is None:
        raise HandBodyError(
            f"{path.name} starts at the hand's own mounting face, so the plates between it and the flange are needed: "
            "pass --coupling-mm. 0 is a real answer for a hand bolted straight to the flange, and saying it is the point"
        )
    if origin == "flange" and coupling_mm:
        raise HandBodyError(
            f"{path.name} already sits at the flange, so a coupling of {coupling_mm:g} mm would move the hand away from "
            "where it was measured"
        )
    placed = placement(rotation_xyzw)
    return hand_body_link(
        spheres=spheres, rotation=placed.rotation, approach_in_tool0=placed.approach_in_tool0,
        origin=origin, coupling_mm=coupling_mm or 0.0,
    )
