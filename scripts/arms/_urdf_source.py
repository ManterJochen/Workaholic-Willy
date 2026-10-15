"""Which description a descriptor is built from, said out loud. Standard library plus PyYAML, loaded by path.

Scanning Isaac's motion policy folder for the model, taking the first file that describes a body, repairing a stray
parenthesis in one of them and grafting a tool frame from a sibling description onto the one generation that has none
is defensible part by part, and together it means two things nobody chose: a box without a simulator cannot build a
descriptor, and ``ur10`` comes that way from an asset whose frames are a different family from the vendor's, with
its links up to 65 mm from where Universal Robots' own description puts them.

There is one source now, ``ur``: Universal Robots' own description, pinned to one upstream commit and rendered here
(:mod:`_ur_description`). It carries a tool frame, complete inertia and an elbow already inside UR's planning limit, so
nothing is repaired and nothing is grafted. What it cannot supply is refused rather than transcribed from a neighbour:
a tool frame written from somewhere else is how a hand ends up pointing where it is not. The ``isaac`` source, a file
on a simulator's disk with the builder's repairs, was retired in P0f F5; every arm had been built from ``ur`` since
B6.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from _ur_description import _commit, render_urdf  # type: ignore[import-not-found]

__all__ = ["SOURCES", "UrdfChoice", "choose"]

#: The one answer to "where does this description come from".
SOURCES = ("ur",)

#: Links a description must carry for the builder to use it without inventing anything.
_REQUIRED_LINKS = ("base_link", "base_link_inertia", "shoulder_link", "upper_arm_link", "forearm_link",
                   "wrist_1_link", "wrist_2_link", "wrist_3_link", "flange", "tool0")


@dataclass(frozen=True)
class UrdfChoice:
    """The description the builder will use, and where it came from."""

    text: str
    provenance: dict[str, Any]

    def render(self) -> str:
        return f"urdf from {self.provenance['urdf_from']}: rendered from UR {self.provenance['description_commit']}"


def urdf_sha256(text: str) -> str:
    """A hash of the description itself, line endings normalised: the same robot on either platform."""
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


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
        provenance={
            "urdf_from": "ur",
            "description_commit": _commit(),
            "urdf_sha256": urdf_sha256(text),
        },
    )


def choose(model: str, *, source: str = "ur") -> UrdfChoice:
    """The description for ``model`` from ``source``, with the provenance that says which it was. It opens nothing
    outside this repository, which is the whole point of it."""
    if source not in SOURCES:
        retired = " (the isaac source was retired in P0f F5)" if source == "isaac" else ""
        raise SystemExit(f"--urdf-from {source!r}: not one of {', '.join(SOURCES)}{retired}")
    return _ur_choice(model)
