"""What a coupling plate costs as a body, measured on the arms it sits on.

    .venv/Scripts/python.exe scripts/curobo/probe_plate_body.py --hand robotiq_hande --plate-mm 20

Run it with the project venv. Needs Coal or python-fcl; no simulator, no GPU, no sidecar.

A plate in ``robot.gripper.coupling_plates`` that declares ``cross_section_mm`` is a body between the flange and the
hand's own mounting face. A plate that declares none is a translation of the hand's meshes along the approach, and
nothing occupies the space it stands for. Whether that changes any verdict is a distance: the plate sits closer to the
wrist than the hand does, so if anything the arm carries binds against the hand, the plate binds first.

The plate is judged by the guard's own pair rule: a pair is checked only where the two frames differ by more than
one, so a plate at frame 6 is never checked against wrist_3 or wrist_2, which is right, because it is bolted to
them. Its first real partner is wrist_1.

No shipped cell declares a cross section, which is what this probe puts a number on. So the radius is swept rather
than read, and the sweep is reported against the hand's own widest half extent at the same pose: a plate wider than
the hand it carries is the case that would change a verdict.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exact_self_collision_poses import ARM_LINKS, _hand_bundle, _parts, judged_pose_set  # noqa: E402

from src.robot.safety._fcl_self_collision import (  # noqa: E402
    _HAND_PARTS,
    MeshSelfCollisionBackend,
    _EngineAdapter,
    import_collision_engine,
)
from src.robot.safety._ur_kinematics import ur_link_transforms_mm  # noqa: E402
from src.robot.safety.planning.environment import collision_mesh_bundle  # noqa: E402
from src.robot.safety.planning.robot.retract_table import (  # noqa: E402
    TABLE_PATH,
    RetractMissing,
    read_retract,
)

#: The name the plate carries in the part map, so a report says which body was measured.
PLATE = "plate"
#: The approach axis in the hand bundle's own frame, which is where the plate stacks. The same +Y every committed
#: hand model approaches along; a declared tool frame turns the whole stack, not this axis.
APPROACH_AXIS = 1
#: Swept, because nothing in the repository declares one. The last value is larger than any UR flange.
RADII_MM = (10.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0)


def plate_part(plate_mm: float, radius_mm: float, *, sections: int = 64) -> "tuple[np.ndarray, np.ndarray, int]":
    """A closed cylinder from the flange to the hand's mounting face, in the hand bundle's frame, millimetres.

    Built by hand rather than through trimesh so this probe carries no dependency the guard does not: two rings and
    two centres, wound outwards, which is the same surface the declared box writer produces for a rectangular plate.
    """
    if plate_mm <= 0.0 or radius_mm <= 0.0:
        raise ValueError(f"a plate body needs a thickness and a radius; got {plate_mm} mm by {radius_mm} mm")
    angles = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    circle = np.stack([radius_mm * np.cos(angles), np.zeros(sections), radius_mm * np.sin(angles)], axis=1)
    near, far = circle.copy(), circle.copy()
    far[:, APPROACH_AXIS] = plate_mm
    centres = np.array([[0.0, 0.0, 0.0], [0.0, plate_mm, 0.0]])
    vertices = np.concatenate([near, far, centres])
    low, high, centre_near, centre_far = 0, sections, 2 * sections, 2 * sections + 1

    faces = []
    for i in range(sections):
        j = (i + 1) % sections
        faces.append([low + i, high + i, high + j])
        faces.append([low + i, high + j, low + j])
        faces.append([centre_near, low + j, low + i])
        faces.append([centre_far, high + i, high + j])
    return vertices, np.asarray(faces, dtype=np.int64), 6


class PlateJudge:
    """The arm with its hand and a plate body, and the smallest exact distance the plate reaches."""

    def __init__(self, arm: str, hand: str, plate_mm: float, radius_mm: float) -> None:
        mod, kind = import_collision_engine()
        if mod is None or kind is None:
            raise SystemExit("no collision engine: install Coal or python-fcl in this interpreter")
        self.arm = arm
        self.engine = kind
        self._adapter = _EngineAdapter(mod, kind)
        meshes = {**_parts(collision_mesh_bundle(arm), ARM_LINKS, 0.0),
                  **_parts(_hand_bundle(arm, hand), tuple(_HAND_PARTS), plate_mm)}
        if radius_mm > 0.0:
            meshes[PLATE] = plate_part(plate_mm, radius_mm)
        self._meshes = meshes
        self._backend = MeshSelfCollisionBackend(self._adapter, meshes)
        names = self._backend._names
        frame = self._backend._frame
        self._pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if abs(frame[a] - frame[b]) > 1]

    def _place(self, joints: "list[float]") -> None:
        frames = ur_link_transforms_mm(self.arm, np.asarray(joints, dtype=np.float64))
        if frames is None:
            raise SystemExit(f"no DH chain for {self.arm}")
        for name in self._backend._names:
            t = frames[self._backend._frame[name]]
            self._adapter.set_transform(self._backend._models[name], t[:3, :3], t[:3, 3])

    def nearest(self, joints: "list[float]", *, only: "str | None" = None) -> "tuple[float, str]":
        """The smallest distance over the checked pairs, or over the pairs touching ``only``."""
        self._place(joints)
        best, pair = float("inf"), ""
        for a, b in self._pairs:
            if only is not None and only not in (a, b):
                continue
            d = float(self._adapter.distance(self._backend._models[a], self._backend._models[b]))
            if d < best:
                best, pair = d, f"{a}|{b}"
        return best, pair

    def marginal(self, poses: "list[list[float]]", *, margin_mm: float) -> "tuple[int, int, float, str]":
        """Over ``poses``: how many the plate alone turns from clear into a collision, and the worst it reaches.

        ``(clear_today, turned_by_the_plate, worst_plate_mm, worst_pose_kind)``. A pose the cell already refuses is
        not counted, because a plate cannot make a collision worse in any way a guard reports. This is the number
        that decides whether an unmodelled plate is a false clear or only a body nobody needed.
        """
        clear_today = turned = 0
        worst, where = float("inf"), ""
        for index, pose in enumerate(poses):
            self._place(pose)
            cell, plate = float("inf"), float("inf")
            for a, b in self._pairs:
                d = float(self._adapter.distance(self._backend._models[a], self._backend._models[b]))
                if PLATE in (a, b):
                    plate = min(plate, d)
                else:
                    cell = min(cell, d)
            if cell <= margin_mm:
                continue
            clear_today += 1
            if plate <= margin_mm:
                turned += 1
            if plate < worst:
                worst, where = plate, f"pose {index}"
        return clear_today, turned, worst, where

    def hand_half_extent_mm(self) -> float:
        """How far the hand's own meshes reach from the approach axis, which is what a plate is compared against."""
        worst = 0.0
        for name in _HAND_PARTS:
            vertices = self._meshes[name][0]
            radial = np.delete(vertices, APPROACH_AXIS, axis=1)
            worst = max(worst, float(np.linalg.norm(radial, axis=1).max()))
        return worst


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Measure what a coupling plate would cost as a collision body.")
    parser.add_argument("--hand", default="robotiq_hande", help="the registry name of the hand the plate carries")
    parser.add_argument("--plate-mm", type=float, default=20.0, help="the plate stack's thickness, in millimetres")
    parser.add_argument("--planner-margin-mm", type=float, default=4.0,
                        help="which judged row of the retract table the pose comes from")
    parser.add_argument("--arms", default="ur3,ur3e,ur5,ur5e,ur10,ur10e,ur16e", help="comma separated")
    parser.add_argument("--radii-mm", default=",".join(f"{r:g}" for r in RADII_MM), help="comma separated")
    parser.add_argument("--pose-set", action="store_true",
                        help="judge the whole committed pose set instead of the retract alone")
    parser.add_argument("--random", type=int, default=1000, help="how many seeded draws the pose set holds")
    parser.add_argument("--tool-rotation-xyzw", type=float, nargs=4, default=[-0.7071067811865476, 0.0, 0.0, 0.7071067811865476],
                        help="the declared tool frame whose retract rows are read; the Isaac cell's by default")
    parser.add_argument("--guard-margin-mm", type=float, default=10.0,
                        help="what the exact guard keeps clear, which is what turns a distance into a refusal")
    args = parser.parse_args(argv)

    from src.robot.safety.planning._hand_placement import HandPlacement

    placed = HandPlacement.from_quaternion_xyzw(tuple(args.tool_rotation_xyzw))
    radii = [float(v) for v in args.radii_mm.split(",") if v.strip()]
    print(f"[plate] {args.hand} on a {args.plate_mm:g} mm plate, at each pair's own committed retract "
          f"({Path(TABLE_PATH).name}, planner margin {args.planner_margin_mm:g} mm)")
    print(f"[plate] radii swept: {', '.join(f'{r:g}' for r in radii)} mm. Nothing in the tree declares one.")

    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        try:
            pose = read_retract(arm, args.hand, args.plate_mm, args.planner_margin_mm,
                                placement=f"{placed.approach}{placed.closing}")
        except RetractMissing as refused:
            print(f"  {arm:6s} no judged retract: {str(refused).splitlines()[0]}")
            continue
        bare = PlateJudge(arm, args.hand, args.plate_mm, 0.0)
        clear, pair = bare.nearest(pose)
        reach = bare.hand_half_extent_mm()
        print(f"  {arm:6s} today {clear:8.3f} mm at {pair:24s} hand reaches {reach:6.2f} mm off the axis")
        poses = judged_pose_set(arm, random_n=args.random)[0] if args.pose_set else None
        for radius in radii:
            judge = PlateJudge(arm, args.hand, args.plate_mm, radius)
            plate_clear, plate_pair = judge.nearest(pose, only=PLATE)
            whole, whole_pair = judge.nearest(pose)
            binds = "  <- the plate is now the nearest body" if PLATE in whole_pair.split("|") else ""
            print(f"         r={radius:5.1f} mm: plate {plate_clear:8.3f} mm at {plate_pair:20s} "
                  f"cell {whole:8.3f} mm at {whole_pair}{binds}")
            if poses is not None:
                total, turned, worst, where = judge.marginal(poses, margin_mm=args.guard_margin_mm)
                print(f"                      over {len(poses)} poses: {total} clear at a {args.guard_margin_mm:g} mm "
                      f"guard, {turned} of them turned by the plate, worst plate {worst:.3f} mm at {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
