"""The bench's hard scenes, on the owner's mat: a deep bin's wall and corner, dense clutter, a flat tray, awkward shapes.

Every scene is built on the stand-in camera's static scene (``scripts/ursim/_mat_scene.py``: the owner's mat cast whole,
its plane as the camera read it) and is a list of :class:`Part` with one named target. Bins and trays are fixed parts:
the camera sees their walls and floor, the jaws never grip them and a push never moves them. The target is what the
prompt names; every other part is a neighbour the camera sees and nobody named.

Each scene says whether a pick is possible at all with the owner's Hand-E (``possible``) and, where not, why
(``impossible``): the bench's goal counts the possible ones only (the owner, 2026-10-05: >= 80 % where a grasp is
geometrically possible, a push counting).

Numbers: the Hand-E opens 50 mm, a finger is about 11 mm thick and 20 mm wide, and the guard keeps 3 mm (2026-10-05),
so a gap under about 14 mm leaves no room for a finger beside a part.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ursim"))
import _mat_scene as ms  # noqa: E402

__all__ = ["BenchScene", "CENTRE_XY", "SCENES", "scene_names"]

#: Where the scenes stand on the mat: in LOOK[0]'s view, 150 mm and more inside every mat edge.
CENTRE_XY = (-150.0, -690.0)
#: A deep small-load carrier: inner length, width, depth, wall and floor thickness, millimetres.
KLT_INNER = (380.0, 280.0, 147.0)
KLT_WALL_MM = 5.0
KLT_FLOOR_MM = 3.0
#: A flat tray: inner length, width, depth.
TRAY_INNER = (300.0, 210.0, 70.0)


@dataclass
class BenchScene:
    """One scene: its family, a builder that places its parts on the static scene, the target's name and prompt."""

    name: str
    family: str
    build: Callable[[Any], list[ms.Part]]
    target: str
    prompt: str
    possible: bool = True
    impossible: str = ""
    note: str = ""
    #: Parts on a floor that is not the mat: a push slides them at the height they stand at.
    keep_height_on_push: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)
    #: ``single``: the target alone is grounded and picked once. ``clear``: every part is grounded ("alle Objekte"), the
    #: stack chooses, and the bench picks until nothing movable is left or two picks in a row fail.
    mode: str = "single"
    #: Of the parts a clearing scene holds, the ones no pick can take at all, by name, and why.
    unpickable: dict[str, str] = field(default_factory=dict)


# --- geometry -------------------------------------------------------------------------------------------------------

def _foot(static: Any, xy: Sequence[float], lift_mm: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    z, normal = static.support_at(xy)
    normal = np.asarray(normal, dtype=np.float64)
    return np.array([float(xy[0]), float(xy[1]), float(z)]) + lift_mm * normal, normal


def _box(static: Any, name: str, xy: Sequence[float], size: Sequence[float], *, yaw_deg: float = 0.0,
         lift_mm: float = 0.0, fixed: bool = False) -> ms.Part:
    foot, normal = _foot(static, xy, lift_mm)
    part = ms.box(name, foot, normal, size_mm=size, yaw_rad=math.radians(yaw_deg))
    part.fixed = fixed
    return part


def _cylinder(static: Any, name: str, xy: Sequence[float], radius: float, height: float, *,
              lift_mm: float = 0.0) -> ms.Part:
    foot, normal = _foot(static, xy, lift_mm)
    return ms.cylinder(name, foot, normal, radius_mm=radius, height_mm=height)


def _lying_cylinder(static: Any, name: str, xy: Sequence[float], radius: float, length: float, *,
                    yaw_deg: float = 0.0, lift_mm: float = 0.0) -> ms.Part:
    """A cylinder lying on its side, its middle over ``xy``, its axis along ``yaw_deg`` in the support's plane."""
    foot, normal = _foot(static, xy, lift_mm)
    up = normal / np.linalg.norm(normal)
    along = np.array([math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg)), 0.0])
    along = along - up * float(along @ up)
    along /= np.linalg.norm(along)
    side = np.cross(along, up)
    frame = np.eye(4)
    frame[:3, 0], frame[:3, 1], frame[:3, 2] = up, side, along   # local z: the axis
    frame[:3, 1] = np.cross(frame[:3, 2], frame[:3, 0])
    frame[:3, 3] = foot + radius * up - 0.5 * length * along
    return ms.Part(name, "cylinder", (radius, radius, length), frame)


def _bin(static: Any, prefix: str, centre: Sequence[float], inner: Sequence[float], *, wall: float,
         floor: float) -> list[ms.Part]:
    """A bin's floor and four walls, fixed, standing on the mat round ``centre``."""
    lx, ly, depth = (float(v) for v in inner)
    cx, cy = float(centre[0]), float(centre[1])
    parts = [_box(static, f"{prefix}_floor", (cx, cy), (lx + 2 * wall, ly + 2 * wall, floor), fixed=True)]
    height = depth + floor
    parts.append(_box(static, f"{prefix}_wall_px", (cx + lx / 2 + wall / 2, cy), (wall, ly + 2 * wall, height),
                      fixed=True))
    parts.append(_box(static, f"{prefix}_wall_nx", (cx - lx / 2 - wall / 2, cy), (wall, ly + 2 * wall, height),
                      fixed=True))
    parts.append(_box(static, f"{prefix}_wall_py", (cx, cy + ly / 2 + wall / 2), (lx, wall, height), fixed=True))
    parts.append(_box(static, f"{prefix}_wall_ny", (cx, cy - ly / 2 - wall / 2), (lx, wall, height), fixed=True))
    return parts


# --- the families ---------------------------------------------------------------------------------------------------

CYL_R, CYL_H = 20.0, 50.0   # the probe's 40 mm cylinder


def _bin_wall(gap_x: float, gap_y: float | None = None, neighbour_gap: float | None = None) -> Callable[[Any], list]:
    """The target cylinder in the deep bin, ``gap_x`` from the inner +x wall (and ``gap_y`` from the +y wall for a
    corner), a 40 mm cube ``neighbour_gap`` off its -x side where given."""

    def build(static: Any) -> list[ms.Part]:
        cx, cy = CENTRE_XY
        lx, ly, _ = KLT_INNER
        parts = _bin(static, "klt", CENTRE_XY, KLT_INNER, wall=KLT_WALL_MM, floor=KLT_FLOOR_MM)
        x = cx + lx / 2 - gap_x - CYL_R
        y = cy if gap_y is None else cy + ly / 2 - gap_y - CYL_R
        parts.append(_cylinder(static, "cylinder", (x, y), CYL_R, CYL_H, lift_mm=KLT_FLOOR_MM))
        if neighbour_gap is not None:
            parts.append(_box(static, "cube40", (x - CYL_R - neighbour_gap - 20.0, y), (40.0, 40.0, 40.0),
                              lift_mm=KLT_FLOOR_MM))
        return parts

    return build


def _clutter(neighbours: Sequence[tuple[str, str, float, float]], target: str = "cylinder") -> Callable[[Any], list]:
    """The target at the centre on the mat, each neighbour ``(kind, side, gap, along)`` off the side it names."""
    sizes = {"cube30": (30.0, 30.0, 30.0), "cube40": (40.0, 40.0, 40.0), "block60": (60.0, 60.0, 40.0),
             "tall": (25.0, 25.0, 90.0)}

    def build(static: Any) -> list[ms.Part]:
        cx, cy = CENTRE_XY
        if target == "cylinder":
            parts = [_cylinder(static, "cylinder", (cx, cy), CYL_R, CYL_H)]
            half = CYL_R
        else:
            parts = [_box(static, target, (cx, cy), (30.0, 30.0, 30.0))]
            half = 15.0
        for index, (kind, side, gap, along) in enumerate(neighbours):
            sx, sy, sz = sizes[kind]
            axis = {"-x": (-1, 0), "+x": (1, 0), "-y": (0, -1), "+y": (0, 1)}[side]
            reach = half + gap + (sx if axis[0] else sy) / 2.0
            x = cx + axis[0] * reach + (along if axis[1] else 0.0)
            y = cy + axis[1] * reach + (along if axis[0] else 0.0)
            parts.append(_box(static, f"{kind}_{index}", (x, y), (sx, sy, sz)))
        return parts

    return build


def _tray(target_kind: str) -> Callable[[Any], list]:
    """A flat tray with five parts in it, ``target_kind`` the target among them."""

    def build(static: Any) -> list[ms.Part]:
        cx, cy = CENTRE_XY
        parts = _bin(static, "tray", CENTRE_XY, TRAY_INNER, wall=4.0, floor=3.0)
        lift = 3.0
        layout = [
            ("cylinder", "cyl", (cx - 60.0, cy + 10.0)),
            ("cube30", "box", (cx + 5.0, cy + 15.0)),
            ("cube40", "box", (cx + 70.0, cy - 30.0)),
            ("cyl_small", "cyl", (cx - 5.0, cy - 50.0)),
            ("lying", "lying", (cx + 80.0, cy + 55.0)),
        ]
        for name, kind, xy in layout:
            if kind == "cyl":
                radius = CYL_R if name == "cylinder" else 15.0
                parts.append(_cylinder(static, name, xy, radius, 45.0, lift_mm=lift))
            elif kind == "box":
                size = 30.0 if name == "cube30" else 40.0
                parts.append(_box(static, name, xy, (size, size, size), yaw_deg=20.0, lift_mm=lift))
            else:
                parts.append(_lying_cylinder(static, name, xy, 14.0, 90.0, yaw_deg=10.0, lift_mm=lift))
        return parts

    return build


def _shape(kind: str) -> Callable[[Any], list]:
    def build(static: Any) -> list[ms.Part]:
        cx, cy = CENTRE_XY
        if kind == "lying":
            return [_lying_cylinder(static, "part", (cx, cy), 15.0, 100.0, yaw_deg=0.0)]
        if kind == "lying_diag":
            return [_lying_cylinder(static, "part", (cx, cy), 15.0, 100.0, yaw_deg=45.0)]
        if kind == "tall":
            return [_box(static, "part", (cx, cy), (25.0, 25.0, 100.0), yaw_deg=15.0)]
        if kind == "flat":
            return [_box(static, "part", (cx, cy), (70.0, 40.0, 18.0), yaw_deg=30.0)]
        if kind == "short":
            return [_box(static, "part", (cx, cy), (22.0, 22.0, 22.0))]
        if kind == "wide_cyl":
            return [_cylinder(static, "part", (cx, cy), 23.0, 30.0)]
        if kind == "lying_beside":
            return [_lying_cylinder(static, "part", (cx, cy), 15.0, 100.0, yaw_deg=0.0),
                    _box(static, "cube40", (cx, cy - 15.0 - 15.0 - 20.0), (40.0, 40.0, 40.0))]
        raise ValueError(kind)

    return build


# --- piles: many parts, dense, seeded --------------------------------------------------------------------------------

#: The kinds a pile draws from, with a weight: (shape, sizes). Boxes are x, y, z; cylinders radius, height; lying
#: cylinders radius, length. Every one fits the Hand-E's 50 mm stroke across one side and stands at least 30 mm high.
PILE_KINDS: dict[str, tuple[str, tuple[float, ...], float]] = {
    "cube30": ("box", (30.0, 30.0, 30.0), 1.0),
    "cube40": ("box", (40.0, 40.0, 40.0), 1.0),
    "brick": ("box", (30.0, 50.0, 32.0), 1.0),
    "slab": ("box", (24.0, 60.0, 30.0), 0.7),
    "post": ("box", (25.0, 25.0, 80.0), 0.5),
    "cyl40": ("cylinder", (20.0, 50.0), 1.0),
    "cyl30": ("cylinder", (15.0, 45.0), 1.0),
    "cyl46": ("cylinder", (23.0, 35.0), 0.6),
    "lying": ("lying", (14.0, 90.0), 0.8),
    "lying_thin": ("lying", (12.0, 70.0), 0.6),
}


def _footprint(kind: str, xy: Sequence[float], yaw: float) -> np.ndarray:
    """The part's footprint in BASE XY as a convex polygon (a circle as a 20-gon)."""
    shape, dims, _ = PILE_KINDS[kind]
    c, s = math.cos(yaw), math.sin(yaw)
    if shape == "cylinder":
        r = dims[0]
        a = np.linspace(0.0, 2.0 * math.pi, 20, endpoint=False)
        return np.column_stack([xy[0] + r * np.cos(a), xy[1] + r * np.sin(a)])
    if shape == "lying":
        hx, hy = dims[1] / 2.0, dims[0]
    else:
        hx, hy = dims[0] / 2.0, dims[1] / 2.0
    corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    turn = np.array([[c, -s], [s, c]])
    return corners @ turn.T + np.asarray(xy, dtype=np.float64)


def _overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two convex polygons overlap (separating axes)."""
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        for ex, ey in edges:
            axis = np.array([-ey, ex])
            pa, pb = a @ axis, b @ axis
            if pa.max() <= pb.min() + 1e-9 or pb.max() <= pa.min() + 1e-9:
                return False
    return True


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    """The least distance between two convex polygons, 0 where they overlap."""
    if _overlap(a, b):
        return 0.0

    def point_edges(points: np.ndarray, poly: np.ndarray) -> float:
        p0, p1 = poly, np.roll(poly, -1, axis=0)
        d = p1 - p0
        best = math.inf
        for q in points:
            t = np.clip(np.einsum("ij,ij->i", q - p0, d) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-12), 0.0, 1.0)
            best = min(best, float(np.min(np.linalg.norm(p0 + t[:, None] * d - q, axis=1))))
        return best

    return min(point_edges(a, b), point_edges(b, a))


def _pile(container: str, count: int, seed: int, *, gap: tuple[float, float] = (0.0, 8.0),
          kinds: Sequence[str] | None = None, candidates: int = 80) -> Callable[[Any], list]:
    """``count`` parts drawn from :data:`PILE_KINDS` with seed ``seed``, packed into ``container`` (``bin``, ``tray`` or
    ``mat``) as densely as ``gap`` allows. Each part is tried at ``candidates`` random places and turns that keep 3 to
    15 mm from the walls and a gap drawn from ``gap`` from every part before it (0: touching), and it takes the one that
    comes nearest the parts already there and the container's middle: a pile, not a scatter. A part that finds no place
    is left out, so a scene may hold fewer than ``count``; its note says how many it holds."""

    def build(static: Any) -> list[ms.Part]:
        rng = np.random.default_rng(seed)
        cx, cy = CENTRE_XY
        if container == "bin":
            parts = _bin(static, "klt", CENTRE_XY, KLT_INNER, wall=KLT_WALL_MM, floor=KLT_FLOOR_MM)
            lx, ly, lift = KLT_INNER[0], KLT_INNER[1], KLT_FLOOR_MM
        elif container == "tray":
            parts = _bin(static, "tray", CENTRE_XY, TRAY_INNER, wall=4.0, floor=3.0)
            lx, ly, lift = TRAY_INNER[0], TRAY_INNER[1], 3.0
        else:
            parts, lx, ly, lift = [], 360.0, 260.0, 0.0
        x0, x1, y0, y1 = cx - lx / 2, cx + lx / 2, cy - ly / 2, cy + ly / 2
        names = list(kinds or PILE_KINDS)
        weights = np.array([PILE_KINDS[k][2] for k in names], dtype=np.float64)
        weights /= weights.sum()
        placed: list[np.ndarray] = []
        made = 0
        for _ in range(count * 3):
            if made >= count:
                break
            kind = names[int(rng.choice(len(names), p=weights))]
            want = float(rng.uniform(*gap))
            best: tuple[float, tuple[float, float], float, np.ndarray] | None = None
            for _ in range(candidates):
                yaw = float(rng.uniform(0.0, math.pi))
                xy = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
                poly = _footprint(kind, xy, yaw)
                wall = float(rng.uniform(3.0, 15.0))
                if (poly[:, 0].min() < x0 + wall or poly[:, 0].max() > x1 - wall
                        or poly[:, 1].min() < y0 + wall or poly[:, 1].max() > y1 - wall):
                    continue
                nearest = min((_distance(poly, other) for other in placed), default=math.inf)
                if nearest < want:
                    continue
                # Nearest the pile, and the middle while the pile is empty: the score packs.
                score = (nearest - want if placed else 0.0) + 0.05 * math.hypot(xy[0] - cx, xy[1] - cy)
                if best is None or score < best[0]:
                    best = (score, xy, yaw, poly)
            if best is None:
                continue
            _, xy, yaw, poly = best
            placed.append(poly)
            name = f"{kind}_{made}"
            shape, dims, _ = PILE_KINDS[kind]
            if shape == "box":
                parts.append(_box(static, name, xy, dims, yaw_deg=math.degrees(yaw), lift_mm=lift))
            elif shape == "cylinder":
                parts.append(_cylinder(static, name, xy, dims[0], dims[1], lift_mm=lift))
            else:
                parts.append(_lying_cylinder(static, name, xy, dims[0], dims[1], yaw_deg=math.degrees(yaw),
                                             lift_mm=lift))
            made += 1
        return parts

    return build


def _piles() -> list[BenchScene]:
    out: list[BenchScene] = []
    for count, seeds in ((15, (1, 2, 3)), (22, (4, 5)), (30, (6,))):
        for seed in seeds:
            out.append(BenchScene(f"pile_bin_{count}_s{seed}", "pile", _pile("bin", count, seed), "", "part",
                                  mode="clear", keep_height_on_push=True,
                                  note=f"clear a 147 mm deep bin packed with {count} mixed parts, 0 to 8 mm apart"))
    for count, seeds in ((20, (11, 12)), (30, (13,))):
        for seed in seeds:
            out.append(BenchScene(f"pile_mat_{count}_s{seed}", "pile", _pile("mat", count, seed), "", "part",
                                  mode="clear", note=f"clear {count} mixed parts packed on the mat, 0 to 8 mm apart"))
    out.append(BenchScene("pile_tray_12_s21", "pile", _pile("tray", 12, 21), "", "part", mode="clear",
                          keep_height_on_push=True, note="clear a 70 mm tray packed with 12 mixed parts"))
    out.append(BenchScene("pile_bin_20_touching_s31", "pile", _pile("bin", 20, 31, gap=(0.0, 2.0)), "", "part",
                          mode="clear", keep_height_on_push=True,
                          note="clear the deep bin, 20 parts packed all but touching (0 to 2 mm)"))
    out.append(BenchScene("pile_bin_20_upright_s41", "pile",
                          _pile("bin", 20, 41, gap=(0.0, 5.0), kinds=("cube30", "cube40", "cyl40", "cyl30", "post")),
                          "", "part", mode="clear", keep_height_on_push=True,
                          note="clear the deep bin, 20 upright cubes, cylinders and posts, 0 to 5 mm apart"))
    return out


def _scenes() -> list[BenchScene]:
    out: list[BenchScene] = _piles()
    # Deep bin: the wall, the corner, and a neighbour in the bin.
    for gap in (10.0, 20.0, 35.0, 50.0, 80.0):
        out.append(BenchScene(f"bin_wall_{gap:g}", "bin", _bin_wall(gap), "cylinder", "green cylinder",
                              keep_height_on_push=True, tags=("wall",),
                              note=f"cylinder {gap:g} mm from the +x wall of a 147 mm deep bin"))
    for gap in (15.0, 30.0, 50.0):
        # 15 mm off both walls the cylinder's axis stands 35 mm from each; the Hand-E's housing reaches 37.5 mm from the
        # TCP's axis and keeps the guard's 3 mm from a wall's box the world grows by 8: 48.5 mm. Down inside a 147 mm bin
        # to a 50 mm part it cannot come (2026-10-06). At 30 mm it has 1.5 mm to spare, square with the walls.
        out.append(BenchScene(f"bin_corner_{gap:g}", "bin", _bin_wall(gap, gap), "cylinder", "green cylinder",
                              keep_height_on_push=True, tags=("corner",), possible=gap > 15.0,
                              impossible="" if gap > 15.0 else
                              "the housing, 37.5 mm off the TCP's axis and 11 mm off a wall's box, cannot come down "
                              "into a corner whose part stands 35 mm from either wall",
                              note=f"cylinder {gap:g} mm from two walls of the deep bin"))
    out.append(BenchScene("bin_centre", "bin", _bin_wall(170.0), "cylinder", "green cylinder",
                          keep_height_on_push=True, note="the control: the cylinder in the middle of the deep bin"))
    for gap in (15.0, 30.0):
        out.append(BenchScene(f"bin_wall_35_neighbour_{gap:g}", "bin", _bin_wall(35.0, None, gap), "cylinder",
                              "green cylinder", keep_height_on_push=True,
                              note=f"35 mm from the wall, a 40 mm cube {gap:g} mm off its other side"))
    # Dense clutter on the mat.
    for gap in (0.0, 5.0, 10.0, 20.0, 30.0):
        out.append(BenchScene(f"clutter_one_{gap:g}", "clutter", _clutter([("cube40", "-x", gap, 0.0)]),
                              "cylinder", "green cylinder", note=f"one 40 mm cube {gap:g} mm off its -x side"))
    for gap in (5.0, 15.0, 28.0):
        out.append(BenchScene(f"clutter_L_{gap:g}", "clutter",
                              _clutter([("cube40", "-x", gap, 0.0), ("cube40", "+y", gap, 0.0)]),
                              "cylinder", "green cylinder", note=f"two 40 mm cubes in an L, {gap:g} mm off"))
    for gap in (10.0, 25.0):
        out.append(BenchScene(f"clutter_ring_{gap:g}", "clutter",
                              _clutter([("cube40", side, gap, 0.0) for side in ("-x", "+x", "-y", "+y")]),
                              "cylinder", "green cylinder", note=f"four 40 mm cubes all round, {gap:g} mm off"))
    out.append(BenchScene("clutter_ring_0", "clutter",
                          _clutter([("cube40", side, 0.0, 0.0) for side in ("-x", "+x", "-y", "+y")]),
                          "cylinder", "green cylinder", possible=False,
                          impossible="touching on all four sides: no finger fits and no push has room to start",
                          note="four 40 mm cubes touching it all round"))
    out.append(BenchScene("clutter_block_20", "clutter", _clutter([("block60", "-x", 20.0, 20.0)]), "cylinder",
                          "green cylinder", note="the URSim push scene: a 60 mm block 20 mm off, shifted 20 mm"))
    out.append(BenchScene("clutter_tall_10", "clutter", _clutter([("tall", "-x", 10.0, 0.0), ("tall", "+x", 10.0, 0.0)]),
                          "cylinder", "green cylinder", note="two tall 90 mm posts 10 mm off either side"))
    # Flat tray.
    for kind in ("cylinder", "cube30", "cube40", "cyl_small", "lying"):
        out.append(BenchScene(f"tray_{kind}", "tray", _tray(kind), kind, {
            "cylinder": "green cylinder", "cube30": "small cube", "cube40": "big cube", "cyl_small": "small cylinder",
            "lying": "lying cylinder"}[kind], keep_height_on_push=True, note=f"the {kind} among five parts in a tray"))
    # Shapes.
    out.append(BenchScene("shape_lying", "shapes", _shape("lying"), "part", "lying cylinder",
                          note="a 30 mm cylinder lying along x"))
    out.append(BenchScene("shape_lying_diag", "shapes", _shape("lying_diag"), "part", "lying cylinder",
                          note="the same, lying at 45 deg"))
    out.append(BenchScene("shape_lying_beside", "shapes", _shape("lying_beside"), "part", "lying cylinder",
                          note="lying, a 40 mm cube 20 mm off its side"))
    out.append(BenchScene("shape_tall", "shapes", _shape("tall"), "part", "tall box",
                          note="a 25 x 25 x 100 mm post, turned 15 deg"))
    out.append(BenchScene("shape_flat", "shapes", _shape("flat"), "part", "flat box",
                          note="a 70 x 40 x 18 mm plate, turned 30 deg: grippable across its 40 mm"))
    out.append(BenchScene("shape_wide_cyl", "shapes", _shape("wide_cyl"), "part", "wide cylinder",
                          note="a 46 mm cylinder, 30 mm high: 4 mm under the Hand-E's 50 mm stroke"))
    out.append(BenchScene("shape_short", "shapes", _shape("short"), "part", "small cube", possible=False,
                          impossible="22 mm high, under the Hand-E's least graspable height (about 28 mm)",
                          note="a 22 mm cube"))
    return out


SCENES: dict[str, BenchScene] = {scene.name: scene for scene in _scenes()}


def scene_names(selection: str = "all") -> list[str]:
    """The scenes ``selection`` names: ``all``, a family (``bin``, ``clutter``, ``tray``, ``shapes``) or names, comma
    separated."""
    names: list[str] = []
    for item in [s.strip() for s in selection.split(",") if s.strip()]:
        if item == "all":
            names += list(SCENES)
        elif item in {scene.family for scene in SCENES.values()}:
            names += [n for n, s in SCENES.items() if s.family == item]
        elif item in SCENES:
            names.append(item)
        else:
            raise ValueError(f"{item!r} is no scene or family; known: {sorted({s.family for s in SCENES.values()})}"
                             f" and {sorted(SCENES)}")
    return list(dict.fromkeys(names))
