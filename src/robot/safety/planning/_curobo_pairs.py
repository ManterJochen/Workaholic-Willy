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

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["NAMED_WITHIN_MM", "PairDepth", "PairOverlap", "SphereLayout", "SphereLayoutError", "collision_pair_rows",
           "deepest_pairs", "overlapping_pairs", "refused_rows"]

#: How close to touching a pair of links is still named by :func:`overlapping_pairs`, in millimetres. The kernel judges
#: in float32 and this in float64 over the same spheres, which differ by about a millionth of a millimetre here; a pair
#: the kernel reads as just touching must never be one this reads as just apart, so a pair this close is named too.
NAMED_WITHIN_MM = 0.01


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
class PairOverlap:
    """Two links whose spheres overlap in one configuration: how deep with the planner's padding, and without any.

    ``depth_mm`` is the deepest sphere pair of the two links as the planner judged it, ``r_i + r_j + pad_i + pad_j -
    |c_i - c_j|``; ``unpadded_depth_mm`` the same sphere pair with no padding at all, negative where the spheres alone
    are that far apart. The names are sorted, as :class:`PairDepth`'s are.
    """

    link_a: str
    link_b: str
    depth_mm: float
    unpadded_depth_mm: float

    def render(self) -> str:
        return (f"{self.link_a} and {self.link_b} overlap by {self.depth_mm:.1f} mm with the planner's padding "
                f"({self._spheres_alone()})")

    def _spheres_alone(self) -> str:
        if self.unpadded_depth_mm > 0.0:
            return f"{self.unpadded_depth_mm:.1f} mm without it"
        return f"the spheres alone {-self.unpadded_depth_mm:.1f} mm apart"

    def to_row(self) -> list[Any]:
        """``[link_a, link_b, depth_mm, unpadded_depth_mm]``, the row a check_js report carries."""
        return [self.link_a, self.link_b, self.depth_mm, self.unpadded_depth_mm]


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


def collision_pair_rows(distances: Any) -> np.ndarray:
    """The sphere pairs cuRobo's self collision checks, as ``(pairs, 2)`` int16 rows, from its pair distance matrix.

    The rows ``SelfCollisionKinematicsCfg.create_from_sphere_pair_distances`` builds, in its order, from one array
    operation: every ``(i, j)`` with ``i < j`` whose distance is not ``-inf``, except where row ``i``'s largest value
    is ``-inf`` (a sphere that checks against nothing). cuRobo builds them in a Python loop over every pair, once per
    robot load, and the sidecar loads the robot three times: 4.8 s of a 24 s start on the desk at 841 spheres, 169,642
    pairs, against 0.02 s for these (2026-10-07, the same rows on all three loads).
    """
    matrix = np.asarray(distances)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"a sphere pair distance matrix is square, got {matrix.shape}")
    count = int(matrix.shape[0])
    if count > int(np.iinfo(np.int16).max) + 1:
        raise ValueError(f"{count} spheres do not fit cuRobo's int16 pair indices")
    if count < 2:
        return np.zeros((0, 2), dtype=np.int16)
    # A NaN in a row is that row's largest value, as torch.max reads it, so such a row is kept, as cuRobo keeps it.
    checks = ~(matrix.max(axis=1) == -np.inf)
    first, second = np.triu_indices(count, k=1)
    keep = (matrix[first, second] != -np.inf) & checks[first]
    return np.ascontiguousarray(np.stack((first[keep], second[keep]), axis=1).astype(np.int16))


@functools.lru_cache(maxsize=8)
def _pair_mask(owners: "tuple[str, ...]", ignored: "frozenset[tuple[str, str]]") -> "tuple[tuple[np.ndarray, np.ndarray], np.ndarray]":
    """The upper-triangle slot pairs of a layout and which of them are never a self collision: inside one link, or a
    pair the descriptor ignores. Once per layout (``owners`` and ``ignored`` are its whole say on it); the arrays are
    read-only, shared by every call."""
    names = np.asarray(owners)
    upper = np.triu_indices(len(owners), k=1)
    links = sorted(set(owners))
    index = {name: i for i, name in enumerate(links)}
    ids = np.asarray([index[name] for name in owners], dtype=np.int64)
    table = np.zeros((len(links), len(links)), dtype=bool)
    for a, b in ignored:
        # A pair is asked for sorted, as the layout stores it; an entry stored the other way round matches nothing.
        if a in index and b in index and a <= b:
            table[index[a], index[b]] = table[index[b], index[a]] = True
    skip = (names[upper[0]] == names[upper[1]]) | table[ids[upper[0]], ids[upper[1]]]
    for array in (*upper, skip):
        array.setflags(write=False)
    return upper, skip


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
    upper, skip = _pair_mask(layout.owners, layout.ignored)

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


def overlapping_pairs(
    spheres: Any, layout: SphereLayout, *, within_mm: float = NAMED_WITHIN_MM,
) -> "tuple[tuple[PairOverlap, ...], ...]":
    """Every pair of links whose spheres overlap, per pose, deepest first; a pose that holds none gets ``()``.

    Where :func:`deepest_pairs` names the one pair an operator reads first, this names all of them, because a driver
    that leaves a pair to the exact guard has to know every pair the planner's verdict rests on. The pairs are the
    kernel's: two different links the descriptor does not take out of self collision, each sphere counted where its
    padded radius is not negative (the kernel's ``valid_mask`` reads the radius after the offset, so an empty slot at
    -100 m never counts and a zero radius with padding does). A pair is named where its deepest sphere pair reaches
    deeper than ``-within_mm`` with the loaded padding (:data:`NAMED_WITHIN_MM`).

    Two links far apart are passed over on their bounding spheres, which bound every sphere of the link with its
    padding, so the shortcut drops no pair a sphere-by-sphere pass would name.
    """
    array = np.asarray(spheres, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 4:
        raise SphereLayoutError(f"spheres must be (poses, slots, 4) in metres, got {array.shape}")
    if array.shape[1] != layout.slots:
        raise SphereLayoutError(
            f"this layout describes {layout.slots} sphere slot(s) and the planner handed over {array.shape[1]}: every "
            "name after the first difference would belong to another link"
        )
    within_m = float(within_mm) / 1000.0
    owners = np.asarray(layout.owners)
    pads = np.asarray(layout.pads_m, dtype=np.float64)
    links = list(dict.fromkeys(layout.owners))
    slots = {name: np.flatnonzero(owners == name) for name in links}
    candidates = [
        (a, b) for index, a in enumerate(links) for b in links[index + 1:]
        if (min(a, b), max(a, b)) not in layout.ignored
    ]
    out: list[tuple[PairOverlap, ...]] = []
    for pose in array:
        centres, reach = pose[:, :3], pose[:, 3] + pads
        held: dict[str, np.ndarray] = {}
        bound: dict[str, tuple[np.ndarray, float]] = {}
        for name in links:
            index = slots[name][reach[slots[name]] >= 0.0]
            if index.size == 0:
                continue
            middle = centres[index].mean(axis=0)
            held[name] = index
            bound[name] = (middle, float(np.max(np.linalg.norm(centres[index] - middle, axis=1) + reach[index])))
        found: list[PairOverlap] = []
        for a, b in candidates:
            if a not in held or b not in held:
                continue
            (middle_a, radius_a), (middle_b, radius_b) = bound[a], bound[b]
            if float(np.linalg.norm(middle_a - middle_b)) - radius_a - radius_b >= within_m:
                continue
            ia, ib = held[a], held[b]
            gap = np.linalg.norm(centres[ia][:, None, :] - centres[ib][None, :, :], axis=2)
            depth = reach[ia][:, None] + reach[ib][None, :] - gap
            row, column = np.unravel_index(int(np.argmax(depth)), depth.shape)
            deepest = float(depth[row, column])
            if deepest <= -within_m:
                continue
            padding = float(pads[ia[row]] + pads[ib[column]])
            found.append(PairOverlap(link_a=min(a, b), link_b=max(a, b), depth_mm=deepest * 1000.0,
                                     unpadded_depth_mm=(deepest - padding) * 1000.0))
        found.sort(key=lambda pair: (-pair.depth_mm, pair.link_a, pair.link_b))
        out.append(tuple(found))
    return tuple(out)


def refused_rows(
    passes: "Sequence[bool]", bound: Any, self_hit: Any, world_hit: Any,
    spheres_of: "Callable[[list[int]], Any]", layout: "SphereLayout | None", *, name_pairs: bool,
) -> "tuple[list[dict[str, Any]], bool]":
    """Every configuration of one check the planner refused, as the rows a check_js report carries, and whether they
    name their pairs.

    ``passes`` is the verdict each configuration got, the one the reply's ``first_invalid`` was read from, so the
    report and the verdict open on the same sample. ``bound``, ``self_hit`` and ``world_hit`` are the three costs of
    every configuration, indexable and each readable as a float; zero is clear. ``spheres_of(indices)`` gives the
    planner's own spheres of those configurations, ``(len(indices), slots, 4)`` in metres, and is asked once, for the
    configurations the self term refused alone. A row is ``{"index", "bound_ok", "self_ok", "world_ok", "pairs"}``,
    ``pairs`` every overlapping pair of links as ``[link_a, link_b, depth_mm, unpadded_depth_mm]``
    (:func:`overlapping_pairs`), empty where the self term did not refuse or its spheres name none, and ``None`` for
    every row where no names were asked or ``layout`` is ``None``. Plain JSON: the sidecar writes it as it stands.
    """
    count = len(passes)
    if not (len(bound) == len(self_hit) == len(world_hit) == count):
        raise ValueError(f"{count} verdict(s) and {len(bound)}, {len(self_hit)} and {len(world_hit)} term(s): every "
                         "configuration has one verdict and three terms")
    named = bool(name_pairs) and layout is not None
    refused = [index for index, ok in enumerate(passes) if not ok]
    hits = [index for index in refused if float(self_hit[index]) > 0.0]
    overlaps: dict[int, tuple[PairOverlap, ...]] = {}
    if named and hits:
        assert layout is not None  # noqa: S101 (named says so)
        overlaps = dict(zip(hits, overlapping_pairs(np.asarray(spheres_of(hits), dtype=np.float64), layout)))
    rows = [{
        "index": index,
        "bound_ok": float(bound[index]) == 0.0,
        "self_ok": float(self_hit[index]) == 0.0,
        "world_ok": float(world_hit[index]) == 0.0,
        "pairs": [pair.to_row() for pair in overlaps.get(index, ())] if named else None,
    } for index in refused]
    return rows, named
