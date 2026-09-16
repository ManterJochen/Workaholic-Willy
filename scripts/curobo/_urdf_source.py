"""Which description a descriptor is built from, said out loud. Standard library plus PyYAML, loaded by path.

Scanning Isaac's motion policy folder for the model, taking the first file that describes a body, repairing a stray
parenthesis in one of them and grafting a tool frame from a sibling description onto the one generation that has none
is defensible part by part, and together it means two things nobody chose: a box without a simulator cannot build a
descriptor, and ``ur10`` comes that way from an asset whose frames are a different family from the vendor's, with
its links up to 65 mm from where Universal Robots' own description puts them.

So there are two sources, and they promise different things.

``isaac``
    A file on Isaac's disk, whatever it says, with the repairs the builder makes.

``ur``
    Universal Robots' own description, pinned to one upstream commit and rendered here
    (:mod:`_ur_description`). It carries a tool frame, complete inertia and an elbow already inside UR's planning
    limit, so nothing is repaired and nothing is grafted. What it cannot supply is refused rather than transcribed
    from a neighbour: a tool frame written from somewhere else is how a hand ends up pointing where it is not.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _description_check import describes_a_body  # type: ignore[import-not-found]
from _ur_description import _commit, render_urdf  # type: ignore[import-not-found]

__all__ = ["SOURCES", "UrdfChoice", "choose", "isaac_root"]

#: The two answers to "where does this description come from".
SOURCES = ("isaac", "ur")

#: Links a description must carry for the builder to use it without inventing anything.
_REQUIRED_LINKS = ("base_link", "base_link_inertia", "shoulder_link", "upper_arm_link", "forearm_link",
                   "wrist_1_link", "wrist_2_link", "wrist_3_link", "flange", "tool0")


@dataclass(frozen=True)
class UrdfChoice:
    """The description the builder will use, and where it came from."""

    text: str
    #: The file it was read from, or ``None`` where it was rendered rather than read.
    path: "Path | None"
    provenance: dict[str, Any]

    def render(self) -> str:
        where = str(self.path) if self.path is not None else f"rendered from UR {self.provenance['description_commit']}"
        return f"urdf from {self.provenance['urdf_from']}: {where}"


def urdf_sha256(text: str) -> str:
    """A hash of the description itself, line endings normalised: the same robot on either platform."""
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def isaac_root(model: str) -> "Path | None":
    """Isaac's motion policy folder for ``model``, or ``None`` when this box has no simulator."""
    env = os.environ.get("WILLY_ISAAC_MP")
    roots = [Path(env)] if env else []
    roots += [
        Path("D:/isaacsim/isaac-sim-standalone-5.1.0-windows-x86_64/exts/isaacsim.robot_motion.motion_generation"
             "/motion_policy_configs/universal_robots"),
    ]
    for root in roots:
        if (root / model).is_dir():
            return root / model
    return None


def _isaac_choice(model: str, isaac_dir: "Path | None", importer_root: "Path | None",
                  asset_root: "Path | None") -> UrdfChoice:
    folder = isaac_dir if isaac_dir is not None else isaac_root(model)
    if folder is None:
        raise SystemExit(
            f"--urdf-from isaac needs Isaac's motion policy configs for {model}, and this box has none. "
            f"Build from the vendor's own description instead: --urdf-from ur"
        )
    candidates = [folder / f"{model}.urdf", folder / f"{model}_robot.urdf"]
    if importer_root is not None:
        candidates.append(importer_root / f"{model}/urdf/{model}.urdf")
    roots = (asset_root,) if asset_root is not None else ()
    chosen: "Path | None" = None
    for candidate in candidates:
        if not candidate.is_file():
            continue
        usable, why = describes_a_body(candidate, asset_roots=roots)
        print(f"urdf candidate {candidate.name:26s} {'USABLE ' if usable else 'skipped'} {why}")
        if usable and chosen is None:
            chosen = candidate
    if chosen is None:
        raise SystemExit(
            f"no usable description for {model}. Looked at: "
            f"{[str(c) for c in candidates if c.is_file()] or 'nothing on disk'}. A description with no "
            f"geometry cannot supply collision meshes, and one whose meshes are missing cannot either."
        )
    text = chosen.read_text(encoding="utf-8")
    return UrdfChoice(
        text=text,
        path=chosen,
        provenance={"urdf_from": "isaac", "urdf_source": str(chosen), "urdf_sha256": urdf_sha256(text)},
    )


def _ur_choice(model: str) -> UrdfChoice:
    text = render_urdf(model)
    missing = [link for link in _REQUIRED_LINKS if f'<link name="{link}"' not in text]
    if missing:
        raise SystemExit(
            f"the description rendered for {model} declares no {', '.join(missing)}. Under --urdf-from ur nothing "
            f"is transcribed from a neighbouring robot: a frame taken from somewhere else is how a hand ends up "
            f"pointing where it is not. Fix the renderer or the vendored config."
        )
    return UrdfChoice(
        text=text,
        path=None,
        provenance={
            "urdf_from": "ur",
            "description_commit": _commit(),
            "urdf_sha256": urdf_sha256(text),
        },
    )


def choose(
    model: str,
    *,
    source: str = "ur",
    isaac_dir: "Path | None" = None,
    importer_root: "Path | None" = None,
    asset_root: "Path | None" = None,
) -> UrdfChoice:
    """The description for ``model`` from ``source``, with the provenance that says which it was.

    ``isaac_dir``, ``importer_root`` and ``asset_root`` are only read on the ``isaac`` path; the ``ur`` path opens
    nothing outside this repository, which is the whole point of it.
    """
    if source not in SOURCES:
        raise SystemExit(f"--urdf-from {source!r}: not one of {', '.join(SOURCES)}")
    if source == "ur":
        return _ur_choice(model)
    return _isaac_choice(model, isaac_dir, importer_root, asset_root)
