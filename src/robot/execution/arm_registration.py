"""Where an arm of a cell stands in the cell's frame: measured by touching, never written (owner, 2026-10-15).

A person guides each arm's TCP by hand onto the same few marked points, in the same order, and presses Enter at each
(Freedrive, as a pose is taught by hand: the arm moves only while a person moves it and holds where it is left). The
first arm of the cell names the frame. The rigid transform that takes the other arm's points onto the first arm's is
that arm's base in the cell (:class:`~src.calibration.UmeyamaRigid`), and every point's residual says how well the
two touches agree. Nothing is calibrated automatically; nothing moves by itself.

    python -m src.robot.execution.real_cell.register --cell clip --arm werker

The result is a file beside the cell file (``<cell folder>/<cell name>/<arm>.json``), so it lives with the user's own
tree, never in the repository. A cell whose arm has no such file knows where every other arm stands but that one.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from src.robot.execution.hand_guiding import CAPTURE, FINISH, HandGuide, HandGuidingLimits, HandGuidingRefused

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.config.cell import LoadedCell
    from src.robot.core.freedrive import SupportsFreedrive

__all__ = [
    "ArmRegistration",
    "RegistrationRefused",
    "base_in_cell",
    "register_arm",
    "registration_file",
    "solve_registration",
    "touch_points",
]

#: The schema of the file, so a later change of what it holds is refused rather than misread.
_SCHEMA = "willy.arm_registration.v1"
#: The fewest points a registration takes, and how many it asks for unless told otherwise.
MIN_POINTS, DEFAULT_POINTS = 3, 4
#: The touched points must span the cell: the second largest spread of the first arm's points, in millimetres. Points
#: on one line, or a hand's width apart, leave the turn about that line to the noise of one touch.
MIN_SPREAD_MM = 100.0
#: The largest root mean square disagreement of the two touches a registration keeps, millimetres. A TCP set on a
#: marked point by hand agrees to about a millimetre; several millimetres is a point touched in the wrong order, a
#: wrong tool frame on one arm, or a base that moved between the two sessions.
MAX_RMSE_MM = 3.0
#: The box a touched point is watched against: none, see :func:`touch_points`.
_ANYWHERE = SimpleNamespace(**{f"{axis}_{end}": math.inf * (1 if end == "max" else -1)
                               for axis in "xyz" for end in ("min", "max")})


class RegistrationRefused(Exception):
    """A registration that was not kept, in one sentence that says what to do."""


def registration_file(cell: "LoadedCell", arm: str) -> Path:
    """Where the registration of ``arm`` lives: ``<cell folder>/<cell name>/<arm>.json``, beside the cell file.

    Args:
        cell (LoadedCell): The cell.
        arm (str): The arm's name as the cell file writes it.

    Returns:
        Path: The file; it may not exist.
    """
    return cell.path.parent / cell.path.stem / f"{cell.arm(arm).name}.json"


@dataclass(frozen=True)
class ArmRegistration:
    """One arm's base in the cell's frame, and the touches it was solved from.

    Attributes:
        arm (str): The arm registered.
        frame_arm (str): The arm whose base is the cell's frame.
        base_in_cell (np.ndarray): 4x4, millimetres: a point in the arm's base, in the cell's frame.
        rmse_mm (float): The root mean square disagreement of the two touches after the fit.
        residuals_mm (tuple[float, ...]): Each point's disagreement, in the order touched.
        frame_points_mm (tuple[tuple[float, float, float], ...]): The frame arm's TCP at each point, in its base.
        arm_points_mm (tuple[tuple[float, float, float], ...]): The registered arm's TCP at each point, in its base.
        captured_at (str): When it was solved, ISO 8601 UTC.
    """

    arm: str
    frame_arm: str
    base_in_cell: np.ndarray = field(compare=False)
    rmse_mm: float
    residuals_mm: tuple[float, ...]
    frame_points_mm: tuple[tuple[float, float, float], ...]
    arm_points_mm: tuple[tuple[float, float, float], ...]
    captured_at: str

    def render(self) -> str:
        """Where the arm stands, and how well the touches agree, in two lines.

        Returns:
            str: ASCII, no trailing newline.
        """
        x, y, z = (float(v) for v in self.base_in_cell[:3, 3])
        yaw = math.degrees(math.atan2(self.base_in_cell[1, 0], self.base_in_cell[0, 0]))
        worst = max(self.residuals_mm, default=0.0)
        return (f"{self.arm} stands at ({x:.1f}, {y:.1f}, {z:.1f}) mm, turned {yaw:.2f} deg, in {self.frame_arm}'s base\n"
                f"  {len(self.residuals_mm)} points, RMS {self.rmse_mm:.2f} mm, worst {worst:.2f} mm")

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def to_dict(self) -> dict[str, Any]:
        """Plain data, ``json.dumps`` safe: what :meth:`write` writes.

        Returns:
            dict[str, Any]: The schema, the arms, the transform (rows), the fit and the touches.
        """
        return {
            "schema": _SCHEMA, "arm": self.arm, "frame_arm": self.frame_arm,
            "base_in_cell_mm": [[float(v) for v in row] for row in self.base_in_cell],
            "rmse_mm": self.rmse_mm, "residuals_mm": list(self.residuals_mm),
            "frame_points_mm": [list(p) for p in self.frame_points_mm],
            "arm_points_mm": [list(p) for p in self.arm_points_mm],
            "captured_at": self.captured_at,
        }

    def write(self, path: Path) -> Path:
        """Write the registration to ``path``, creating its folder.

        Args:
            path (Path): Usually :func:`registration_file`.

        Returns:
            Path: ``path``.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path) -> "ArmRegistration":
        """The registration in ``path``.

        Args:
            path (Path): A file :meth:`write` wrote.

        Returns:
            ArmRegistration: What it holds.

        Raises:
            RegistrationRefused: The file is not a registration this version reads.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != _SCHEMA:
            raise RegistrationRefused(f"{path} is no arm registration this version reads (schema "
                                      f"{data.get('schema')!r}, wanted {_SCHEMA!r}): register the arm again")
        matrix = np.asarray(data["base_in_cell_mm"], dtype=np.float64)
        return cls(arm=data["arm"], frame_arm=data["frame_arm"], base_in_cell=matrix, rmse_mm=float(data["rmse_mm"]),
                   residuals_mm=tuple(float(v) for v in data["residuals_mm"]),
                   frame_points_mm=tuple(tuple(float(v) for v in p) for p in data["frame_points_mm"]),  # type: ignore[misc]
                   arm_points_mm=tuple(tuple(float(v) for v in p) for p in data["arm_points_mm"]),  # type: ignore[misc]
                   captured_at=data["captured_at"])


def solve_registration(
    frame_points_mm: Sequence[Sequence[float]], arm_points_mm: Sequence[Sequence[float]]
) -> tuple[np.ndarray, float, tuple[float, ...]]:
    """The other arm's base in the frame arm's, from the same points touched by both.

    Args:
        frame_points_mm (Sequence[Sequence[float]]): The frame arm's TCP at each point, in its base.
        arm_points_mm (Sequence[Sequence[float]]): The other arm's TCP at the same points, in the same order, in its base.

    Returns:
        tuple[np.ndarray, float, tuple[float, ...]]: The 4x4 transform (millimetres), the RMS disagreement and each
            point's disagreement.

    Raises:
        RegistrationRefused: Too few points, different counts, points on a line or too close together, or touches that
            disagree by more than :data:`MAX_RMSE_MM`.
    """
    from src.calibration import UmeyamaRigid  # noqa: PLC0415

    frame = np.asarray(frame_points_mm, dtype=np.float64).reshape(-1, 3)
    other = np.asarray(arm_points_mm, dtype=np.float64).reshape(-1, 3)
    if len(frame) != len(other):
        raise RegistrationRefused(f"{len(frame)} points touched with the frame arm and {len(other)} with the other: "
                                  "touch the same points with both")
    if len(frame) < MIN_POINTS:
        raise RegistrationRefused(f"{len(frame)} points touched: a registration needs at least {MIN_POINTS}, spread "
                                  "over the space both arms reach")
    spread = np.linalg.svd(frame - frame.mean(axis=0), compute_uv=False)
    second = float(spread[1]) / math.sqrt(len(frame))
    if second < MIN_SPREAD_MM / 2.0:
        raise RegistrationRefused(f"the points lie nearly on one line or close together (spread {2.0 * second:.0f} mm "
                                  f"across, at least {MIN_SPREAD_MM:.0f} mm wanted): spread them out, not in a row")
    rotation, translation, rmse = UmeyamaRigid().solve(other, frame)
    residuals = np.linalg.norm(other @ rotation.T + translation - frame, axis=1)
    if rmse > MAX_RMSE_MM:
        worst = int(np.argmax(residuals))
        raise RegistrationRefused(
            f"the two arms' touches disagree by {rmse:.1f} mm RMS (at most {MAX_RMSE_MM:.1f} mm kept); point "
            f"{worst + 1} is off by {residuals[worst]:.1f} mm. Touch every point again, in the same order, with the "
            "tool frame each arm declares")
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, translation
    return matrix, float(rmse), tuple(float(v) for v in residuals)


def touch_points(
    arm: "SupportsFreedrive", *, name: str, count: int = DEFAULT_POINTS, guide: HandGuide | None = None,
) -> tuple[tuple[float, float, float], ...]:
    """The TCP of a connected arm at ``count`` points a person guides it onto, in order. Holds the arm on every way out.

    Each point is watched against the arm's cable window, as a taught pose is (said, and Enter does not capture
    outside it), and not against the workspace box: the marks sit where both arms reach, often on the table under the
    box's floor, and nothing is commanded to them.

    Args:
        arm (SupportsFreedrive): A connected arm a person can guide.
        name (str): The arm's name, said in every prompt.
        count (int): How many points (default: 4).
        guide (HandGuide | None): The console side; ``None`` the terminal (default: None).

    Returns:
        tuple[tuple[float, float, float], ...]: The TCP at each point, millimetres, in the arm's base.

    Raises:
        HandGuidingRefused: The person finished before every point was touched.
    """
    guide = guide if guide is not None else HandGuide()
    limits = HandGuidingLimits.of(arm, _ANYWHERE)
    guide.confirm_payload(arm)
    points: list[tuple[float, float, float]] = []
    with arm.freedrive() as session:
        while len(points) < count:
            point = len(points) + 1
            session.free()
            choice = guide.wait(session, limits, banner=f"GUIDE {name.upper()}'S TCP ONTO POINT {point} of {count}, "
                                                        "then Enter")
            session.hold()
            if choice == FINISH:
                raise HandGuidingRefused(f"finished after {len(points)} of {count} points: nothing is kept")
            if choice != CAPTURE:
                guide.console.say(f"point {point} is touched again: every point is needed, in order")
                continue
            xyz = session.sample().tcp_xyz_mm
            points.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
            guide.console.say(f"{name}: point {point} at ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}) mm; the arm holds")
    return tuple(points)


def base_in_cell(cell: "LoadedCell", arm: str, *, read: Callable[[Path], ArmRegistration] = ArmRegistration.read
                 ) -> np.ndarray | None:
    """Where ``arm`` stands in the cell's frame: the identity for the first arm, its registration for any other.

    Args:
        cell (LoadedCell): The cell.
        arm (str): The arm's name.
        read (Callable[[Path], ArmRegistration]): How a registration file is read (default: ArmRegistration.read).

    Returns:
        np.ndarray | None: 4x4, millimetres; ``None`` for an arm that has not been registered yet.
    """
    if cell.arm(arm).name == cell.frame_arm.name:
        return np.eye(4)
    path = registration_file(cell, arm)
    return read(path).base_in_cell if path.is_file() else None


def now_iso() -> str:
    """The time a registration is solved at, ISO 8601 UTC, seconds."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


#: Why a robot built for touching may stand without a wrist camera body that cannot be placed yet: it is never
#: commanded here, only guided by hand (as the fault door builds its arm, P0i i4 review).
_ONLY_GUIDED = ("touching marks moves nothing: the arm is guided by hand and held, and never commanded")


def register_arm(
    cell: "LoadedCell", arm: str, *, count: int = DEFAULT_POINTS,
    guide_for: "Callable[[str], HandGuide] | None" = None, robot_of: "Callable[[Any], Any] | None" = None,
) -> ArmRegistration:
    """Touch the same marks with the cell's first arm and then with ``arm``, solve, and write the file beside the cell.

    Each arm is built alone (no hand), connected under its own lock, guided by hand and held, then disconnected
    before the next; only one arm is connected at a time.

    Args:
        cell (LoadedCell): The cell.
        arm (str): The arm to register; not the first one, which is the frame.
        count (int): How many marks (default: 4).
        guide_for (Callable[[str], HandGuide] | None): The console side for the arm named; ``None`` a terminal for
            each, so each arm's payload is confirmed on its own (default: None).
        robot_of (Callable[[Any], Any] | None): Builds a ``Robot`` from an arm's tree; ``None`` builds it alone
            (default: None).

    Returns:
        ArmRegistration: What was written to :func:`registration_file`.

    Raises:
        RegistrationRefused: ``arm`` is the frame arm, an arm offers no hand guiding, or the touches do not fit.
        HandGuidingRefused: A person finished early or did not confirm a payload.
    """
    from src.robot.core.freedrive import SupportsFreedrive  # noqa: PLC0415

    target = cell.arm(arm)
    frame = cell.frame_arm
    if target.name == frame.name:
        raise RegistrationRefused(f"{target.name} is the cell's frame, so it stands where the cell's positions are "
                                  "written: register another arm against it")
    if robot_of is None:
        from src.robot.execution.robot import Robot  # noqa: PLC0415

        def robot_of(tree: Any) -> Any:
            return Robot.from_tree(tree, gripper=None, unmodelled_wrist_body=_ONLY_GUIDED)
    touched: dict[str, tuple[tuple[float, float, float], ...]] = {}
    for each, how in ((frame, "the marks you choose, in an order you keep"), (target, "the same marks, in that order")):
        robot = robot_of(each.tree)
        if not isinstance(robot.arm, SupportsFreedrive):
            raise RegistrationRefused(f"{each.name} ({type(robot.arm).__name__}) offers no hand guiding: its base is "
                                      "measured by touching, so it needs an arm a person can guide")
        guide = guide_for(each.name) if guide_for is not None else HandGuide()
        guide.console.say(f"{each.name}: touch {count} of {how}, with the TCP its tool frame declares.")
        with robot.connected():
            touched[each.name] = touch_points(robot.arm, name=each.name, count=count, guide=guide)
    matrix, rmse, residuals = solve_registration(touched[frame.name], touched[target.name])
    registration = ArmRegistration(arm=target.name, frame_arm=frame.name, base_in_cell=matrix, rmse_mm=rmse,
                                   residuals_mm=residuals, frame_points_mm=touched[frame.name],
                                   arm_points_mm=touched[target.name], captured_at=now_iso())
    registration.write(registration_file(cell, target.name))
    return registration
