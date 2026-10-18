"""The world a cell of several arms states, in the base of each arm (owner, 2026-10-15).

The cell's first arm stands at the cell's frame, every other arm where touching put it (``arm_registration``). What a
layer the arms share states of the world stands in the cell's frame, written once; what an arm's own layer states
stands in its own base, measured with that arm (``LoadedCell.where_written``). :func:`arm_tree` is the tree an arm runs
on: the shared world turned into its base, nothing else changed.

* A point and a height: exact.
* A box the guard or the planner keeps out of (``self_collision.fixtures``, the planner's bench): the box about the
  turned box, never smaller.
* A box the arm keeps inside (a bin's interior): the largest box of its shape inside the turned box, never larger.
* A planner's mesh: its centre and its turn about Z, exact.
* A fixed camera whose calibration a shared layer names: calibrated once, against the first arm. Every other arm reads
  it through a file of its own beside its registration, the same camera in its base, its error the camera's and the
  touching's together.

Each of these is exact where the arms stand square to each other; off square, a box only grows or shrinks to the safe
side. Two bases have to stand level to each other within :data:`LEVEL_DEG`, so a bench level in one is level in the
other.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.config.cell import LoadedCell
    from src.config.tree import LoadedTree

__all__ = ["LEVEL_DEG", "arm_tree", "cameras_folder"]

#: How far two bases of one cell may tilt against each other, degrees: 0.5 puts a bench 9 mm off level a metre away.
LEVEL_DEG = 0.5


def cameras_folder(cell: "LoadedCell", arm: str) -> Path:
    """Where the fixed cameras ``arm`` shares with the first arm are written in its own base: beside its registration.

    Args:
        cell (LoadedCell): The cell.
        arm (str): The arm's name.

    Returns:
        Path: ``<cell folder>/<cell name>/<arm>/``; it may not exist.
    """
    from src.robot.execution.arm_registration import registration_file  # noqa: PLC0415

    return registration_file(cell, arm).with_suffix("")


def arm_tree(cell: "LoadedCell", arm: str, in_cell: np.ndarray) -> "LoadedTree":
    """The tree ``arm`` runs on in its cell: the world a layer it shares with the first arm states, in its own base.

    Writes the file of every fixed camera it shares (:func:`cameras_folder`), where that changed. Moves nothing.

    Args:
        cell (LoadedCell): The cell, loaded.
        arm (str): The arm's name.
        in_cell (np.ndarray): 4x4, millimetres: its base in the cell's frame (``arm_registration.base_in_cell``).

    Returns:
        LoadedTree: The first arm's tree as it is; another arm's with its shared world turned into its base.

    Raises:
        ConfigError: The two bases tilt against each other, a block stands in two frames, or the turned tree does not
            load.
    """
    from src.config.loader import ConfigError  # noqa: PLC0415

    tree = cell.arm(arm).tree
    if cell.arm(arm).name == cell.frame_arm.name:
        return tree
    place = _Placement(np.asarray(in_cell, dtype=np.float64))
    if place.tilt_deg > LEVEL_DEG:
        raise ConfigError(f"the arm {arm!r} stands {place.tilt_deg:.2f} degrees tilted against {cell.frame_arm.name!r}, "
                          f"and a cell of several arms needs their bases level to {LEVEL_DEG} degrees: a bench level for "
                          "one would not be for the other. Level the mount, or touch the marks again")
    robot = tree.robot
    values: dict[str, Any] = {}

    def shared(block: str) -> bool:
        return cell.where_written(arm, block) == "cell"

    if shared("robot.safety.self_collision.fixtures"):
        values["robot.safety.self_collision.fixtures"] = [
            {"name": box.name, **place.box_about(box.center_mm, box.half_extents_mm)}
            for box in robot.safety.self_collision.fixtures]
    world = robot.safety.planning_world
    bench = getattr(world, "support_plane", None)
    if bench is not None and shared("robot.safety.planning_world.support_plane"):
        top = place.point((*bench.center_mm, bench.height_mm))
        half = np.abs(place.turn[:2, :2]) @ (np.asarray(bench.extent_mm, dtype=np.float64) / 2.0)
        values["robot.safety.planning_world.support_plane"] = {
            "height_mm": float(top[2]), "center_mm": [float(top[0]), float(top[1])],
            "extent_mm": [float(v) for v in 2.0 * half], "thickness_mm": float(bench.thickness_mm)}
    if shared("robot.safety.planning_world.meshes"):
        values["robot.safety.planning_world.meshes"] = [
            {"name": mesh.name, "path": str(mesh.path), "center_mm": _listed(place.point(mesh.center_mm)),
             "yaw_deg": _wrapped(float(mesh.yaw_deg) - place.yaw_deg), "scale": float(mesh.scale)}
            for mesh in world.meshes]
    if shared("robot.grasping.support"):
        # The parts stand on the bench, which is level (2026-10-17), so a height moves by the base's own height.
        along = float(place.shift[2])
        container = robot.grasping.support.container
        if container.floor_height_mm is not None:
            values["robot.grasping.support.container.floor_height_mm"] = float(container.floor_height_mm) - along
        if container.interior_min_mm is not None and container.interior_max_mm is not None:
            low, high = np.asarray(container.interior_min_mm), np.asarray(container.interior_max_mm)
            inside = place.box_inside((low + high) / 2.0, (high - low) / 2.0)
            centre, half = np.asarray(inside["center_mm"]), np.asarray(inside["half_extents_mm"])
            values["robot.grasping.support.container.interior_min_mm"] = _listed(centre - half)
            values["robot.grasping.support.container.interior_max_mm"] = _listed(centre + half)
    values.update(_shared_cameras(cell, arm, tree, place))
    if not values:
        return tree
    turned = tree.with_values(values)
    if not turned.ok:
        raise ConfigError(f"the world the cell states, in the base of {arm!r}, does not load: {turned.error}")
    return turned


class _Placement:
    """An arm's base in the cell's frame, and what it does to a point, a direction and a box of the cell."""

    def __init__(self, in_cell: np.ndarray) -> None:
        #: 4x4, millimetres: the arm's base in the cell's frame.
        self.matrix = in_cell
        rotation = in_cell[:3, :3]
        #: The cell's directions in the arm's base, and where the arm's base stands in the cell.
        self.turn = rotation.T
        self.shift = in_cell[:3, 3]
        self.tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(rotation[2, 2])))))
        self.yaw_deg = math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))
        # The turn snapped to the nearest quarter turn about Z: the shape a box keeps where the arms stand square.
        quarter = math.radians(90.0 * round(self.yaw_deg / 90.0))
        c, s = round(math.cos(quarter)), round(math.sin(quarter))
        self._square = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def point(self, point: Sequence[float]) -> np.ndarray:
        """A point of the cell, in the arm's base."""
        return self.turn @ (np.asarray(point, dtype=np.float64) - self.shift)

    def box_about(self, center: Sequence[float], half: Sequence[float]) -> dict[str, list[float]]:
        """The box of the arm's axes about a box of the cell's axes: never smaller."""
        return {"center_mm": _listed(self.point(center)),
                "half_extents_mm": _listed(np.abs(self.turn) @ np.asarray(half, dtype=np.float64))}

    def box_inside(self, center: Sequence[float], half: Sequence[float]) -> dict[str, list[float]]:
        """The largest box of the arm's axes, of the box's own shape, inside a box of the cell's axes: never larger.
        Every corner of it, turned back into the cell, stays inside the box (``|R| h' <= h``)."""
        sides = np.asarray(half, dtype=np.float64)
        shape = np.abs(self._square.T) @ sides
        reach = np.abs(self.turn.T) @ shape
        scale = min(1.0, *(float(h / r) for h, r in zip(sides, reach) if r > 0.0))
        return {"center_mm": _listed(self.point(center)), "half_extents_mm": _listed(scale * shape)}


def _shared_cameras(cell: "LoadedCell", arm: str, tree: "LoadedTree", place: _Placement) -> dict[str, Any]:
    """The artifact of every fixed camera whose calibration a shared layer names, in ``arm``'s base: a file of its own,
    written where it changed, and the key that names it."""
    from src.calibration.rig_calibration import RigCalibration  # noqa: PLC0415
    from src.calibration.serialization import extrinsics_to_dict, save_extrinsics  # noqa: PLC0415
    from src.geometry import Frame  # noqa: PLC0415
    from src.geometry.transform import Transform  # noqa: PLC0415
    from src.robot.execution.arm_registration import ArmRegistration, registration_file  # noqa: PLC0415

    values: dict[str, Any] = {}
    registration: ArmRegistration | None = None
    for index, rig in enumerate(tree.app_config.camera.cameras.rigs):
        extrinsics = getattr(rig, "extrinsics", None)
        if extrinsics is None or extrinsics.mounting_mode != "eye_to_hand":
            continue
        if cell.where_written(arm, f"camera.cameras.rigs[{index}].extrinsics") != "cell":
            continue
        registration = registration or ArmRegistration.read(registration_file(cell, arm))
        source = Path(str(extrinsics.artifact_path))
        calibrated = RigCalibration.from_config(rig.rig_id, extrinsics).solve
        assert calibrated is not None  # noqa: S101 (a fixed camera's artifact is its solve)
        to_cell = calibrated.transform.to_matrix()
        to_arm = np.linalg.inv(place.matrix) @ to_cell
        seen = replace(
            calibrated,
            transform=Transform.from_matrix(to_arm, from_frame=Frame.CAMERA, to_frame=Frame.BASE),
            rmse_mm=math.hypot(float(calibrated.rmse_mm), float(registration.rmse_mm)),
            max_error_mm=float(calibrated.max_error_mm) + max(registration.residuals_mm, default=0.0))
        target = cameras_folder(cell, arm) / source.name
        held = (RigCalibration.from_config(rig.rig_id, extrinsics.model_copy(update={"artifact_path": str(target)})).solve
                if target.is_file() else None)
        if held is None or extrinsics_to_dict(held) != extrinsics_to_dict(seen):
            save_extrinsics(target, seen)
        values[f"camera.cameras.rigs[{index}].extrinsics.artifact_path"] = str(target)
    return values


def _listed(values: Sequence[float] | np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=np.float64).reshape(-1)]


def _wrapped(degrees: float) -> float:
    """``degrees`` in (-180, 180]."""
    wrapped = (degrees + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped
