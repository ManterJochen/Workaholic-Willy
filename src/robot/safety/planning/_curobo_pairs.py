"""Which two links a planner's spheres overlap in, and by how much. Standard library plus numpy, on purpose.

cuRobo answers a refused configuration with a number: the self collision term is positive. The question it leaves open
is which two links touched, and neither the client nor the sidecar can read it off that number. cuRobo holds the robot
as one flat array of spheres, so the answer comes from ownership in the descriptor the planner loaded:
``collision_link_names`` in order, the count per link from ``collision_spheres``, the spare slots from
``extra_collision_spheres``, the padding from ``self_collision_buffer``, and the pairs ``self_collision_ignore``
removes.

Pure arithmetic on plain data, imported from both sides of the process boundary like ``_curobo_margin``, so both
interpreters run the same code. What nothing in this module can prove is that cuRobo lays ``robot_spheres`` out in that
order; only a box run naming a pair a probe measured the same way can.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["PairDepth", "SphereLayout", "SphereLayoutError", "deepest_pairs"]


class SphereLayoutError(ValueError):
    """A layout and the planner's own sphere array disagree, so every name past the difference is somebody else's."""


@dataclass(frozen=True)
class PairDepth:
    """Two links and how far their spheres reach into one another, in millimetres."""

    link_a: str
    link_b: str
    depth_mm: float

    def render(self) -> str:
        return f"{self.link_a} and {self.link_b} overlap by {self.depth_mm:.1f} mm"

    def to_dict(self) -> dict[str, Any]:
        return {"link_a": self.link_a, "link_b": self.link_b, "depth_mm": self.depth_mm}


@dataclass(frozen=True)
class SphereLayout:
    """Who owns each sphere slot of a loaded descriptor, what pads it, and which pairs are ignored."""

    #: The link of every sphere slot, in cuRobo's own order.
    owners: tuple[str, ...]
    #: The self collision buffer of each slot's link, in metres. A link with no entry pads 0.
    pads_m: tuple[float, ...]
    #: Unordered link pairs the descriptor takes out of self collision, each stored sorted.
    ignored: frozenset[tuple[str, str]]

    @property
    def slots(self) -> int:
        return len(self.owners)

    @classmethod
    def from_robot_config(cls, config: "dict[str, Any]") -> "SphereLayout | None":
        """The layout of a robot config, or ``None`` where the config does not spell its spheres out.

        A stock cuRobo descriptor points ``collision_spheres`` at a file it resolves as it loads, and by then the
        ownership is gone. That is a diagnostic this config cannot support, and not a reason to stop a planner: the
        caller reports the pair as unnamed and everything else about the refusal stands. The one thing that does refuse
        lives in :func:`deepest_pairs`, where a layout and the planner's array can contradict one another.
        """
        kinematics = (config.get("robot_cfg") or config).get("kinematics")
        if not isinstance(kinematics, dict):
            return None
        spheres = kinematics.get("collision_spheres")
        if not isinstance(spheres, dict):
            return None
        extra = kinematics.get("extra_collision_spheres") or {}
        buffers = kinematics.get("self_collision_buffer") or {}
        owners: list[str] = []
        pads: list[float] = []
        for link in kinematics.get("collision_link_names") or ():
            name = str(link)
            # extra_collision_spheres replaces a link's spheres with that many empty slots, so a link that has both
            # is counted once, by the table that wins in cuRobo.
            count = int(extra[name]) if name in extra else len(spheres.get(name) or ())
            owners.extend([name] * count)
            pads.extend([float(buffers.get(name, 0.0))] * count)
        ignored: set[tuple[str, str]] = set()
        for link, against in (kinematics.get("self_collision_ignore") or {}).items():
            for other in against or ():
                pair = (str(link), str(other))
                ignored.add((min(pair), max(pair)))
        if not owners:
            return None
        return cls(owners=tuple(owners), pads_m=tuple(pads), ignored=frozenset(ignored))


def deepest_pairs(spheres: Any, layout: SphereLayout) -> "tuple[PairDepth | None, ...]":
    """The deepest overlapping link pair per pose, or ``None`` where a pose holds none.

    ``spheres`` is cuRobo's own array, ``(poses, slots, 4)`` in metres: centre and radius. A slot with a radius of 0 or
    less is an empty payload slot and can never collide, a pair inside one link is not a self collision, and an ignored
    pair is not one either. The depth is what cuRobo's own arithmetic leaves over:
    ``r_i + r_j + pad_i + pad_j - |c_i - c_j|``.
    """
    array = np.asarray(spheres, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 4:
        raise SphereLayoutError(f"spheres must be (poses, slots, 4) in metres, got {array.shape}")
    if array.shape[1] != layout.slots:
        raise SphereLayoutError(
            f"this layout describes {layout.slots} sphere slot(s) and the planner handed over {array.shape[1]}: every "
            "name after the first difference would belong to another link"
        )
    owners = np.asarray(layout.owners)
    pads = np.asarray(layout.pads_m, dtype=np.float64)
    upper = np.triu_indices(layout.slots, k=1)
    same_link = owners[upper[0]] == owners[upper[1]]
    ignored = np.asarray([
        (min(a, b), max(a, b)) in layout.ignored for a, b in zip(owners[upper[0]], owners[upper[1]])
    ], dtype=bool) if layout.slots > 1 else np.zeros(0, dtype=bool)
    skip = same_link | ignored

    out: list[PairDepth | None] = []
    for pose in array:
        centres, radii = pose[:, :3], pose[:, 3]
        empty = radii <= 0.0
        reach = radii + pads
        gap = np.linalg.norm(centres[upper[0]] - centres[upper[1]], axis=1)
        depth = (reach[upper[0]] + reach[upper[1]] - gap) * 1000.0
        depth[skip | empty[upper[0]] | empty[upper[1]]] = -np.inf
        best = int(np.argmax(depth)) if depth.size else -1
        if best < 0 or not np.isfinite(depth[best]) or depth[best] <= 0.0:
            out.append(None)
            continue
        first, second = str(owners[upper[0][best]]), str(owners[upper[1][best]])
        out.append(PairDepth(link_a=min(first, second), link_b=max(first, second), depth_mm=float(depth[best])))
    return tuple(out)
