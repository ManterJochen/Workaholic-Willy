"""Doubles for the place tests: what a pick's looks kept, a surface the camera found under a part, and a task run on them.

* :class:`FlatSupport` is a support model (``support_surfaces.SupportModel``'s two questions a place asks) of one level
  surface over a rectangle of BASE XY: the owner's 55 mm mat, say.
* :func:`pick_view` is one of a pick's looks as ``LookedAround.views`` keeps it, ray cast over a :class:`BinScene`
  through the wrist camera of ``tests/_task_fakes.py`` (200 mm behind the TCP, along the tool's axis).
* :class:`LookingService` is ``TaskService`` whose picks leave what their looks came to (``looked_around``): the part's
  cloud, the surface under it and the looks' frames, as a test scripts them per pick.
* :class:`AgainWorld` is the arm's live planner world that can hold a pick's frames again for a place
  (``hold_pick_views_again``), every call on the shared log.
* :func:`run_placing` is ``tests._task_fakes.run`` on a :class:`LookingService`.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from src.contracts import UNSET, chosen
from src.geometry import Pose
from src.robot.core import JointPositions
from tests._task_fakes import (
    _CAMERA_TO_TOOL,
    _LENS,
    GRASP_Z_MM,
    PART_XY,
    POSES,
    BinScene,
    Pick,
    RecordingHooks,
    Ran,
    TaskArm,
    TaskService,
    toggle,
)

__all__ = ["GRASP", "AgainWorld", "FlatSupport", "LookingService", "cube_cloud", "hande_tree", "pick_view",
           "run_placing", "what_the_looks_saw"]


@dataclass(frozen=True)
class FlatSupport:
    """A level surface at ``level_mm`` over BASE XY ``x_mm`` x ``y_mm``: what the pick's looks found the part stood on."""

    level_mm: float
    x_mm: "tuple[float, float]" = (-1000.0, 1000.0)
    y_mm: "tuple[float, float]" = (-1000.0, 1000.0)

    def _covers(self, xy: Any) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        return ((points[:, 0] >= self.x_mm[0]) & (points[:, 0] <= self.x_mm[1]) & (points[:, 1] >= self.y_mm[0])
                & (points[:, 1] <= self.y_mm[1]))

    def local_plane_under(self, xy: Any) -> "np.ndarray | None":
        covered = self._covers(xy)
        if not covered.any():
            return None
        planes = np.full((covered.shape[0], 4), np.nan)
        planes[covered] = (0.0, 0.0, 1.0, self.level_mm)
        return planes

    def height_under(self, foot_xy: Any) -> "float | None":
        return float(self.level_mm) if self._covers(foot_xy).any() else None


def cube_cloud(xy: "tuple[float, float]" = PART_XY, *, bottom_mm: float = 0.0, side_mm: float = 40.0,
               step_mm: float = 4.0) -> np.ndarray:
    """What the looks saw of a cube standing at ``xy`` on ``bottom_mm``: its top and its four sides, BASE mm."""
    half = side_mm / 2.0
    grid = np.arange(-half, half + 1e-9, step_mm)
    gx, gy = np.meshgrid(grid, grid)
    top = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, bottom_mm + side_mm)])
    sides = []
    for z in np.arange(bottom_mm + step_mm / 2.0, bottom_mm + side_mm, step_mm):
        for u in grid:
            sides += [(u, -half, z), (u, half, z), (-half, u, z), (half, u, z)]
    cloud = np.vstack([top, np.asarray(sides, dtype=np.float64)])
    cloud[:, 0] += xy[0]
    cloud[:, 1] += xy[1]
    return cloud


def pick_view(scene: BinScene, tcp: Pose, label: str, *, rig: str = "wrist", stamp: float = 1.0) -> Any:
    """One of a pick's looks as ``LookedAround.views`` keeps it: the frame the wrist camera takes over ``scene`` with the
    tool at ``tcp``, its colour (RGB), its depth and where the camera stood, under the look's ``label``."""
    camera_to_base = np.asarray(tcp.to_matrix(), dtype=np.float64) @ _CAMERA_TO_TOOL
    depth, image, _shows = scene.render(camera_to_base)
    frame = SimpleNamespace(depth_map=depth, intrinsics=_LENS.copy(), rgb=np.ascontiguousarray(image[..., ::-1]),
                            timestamp=stamp, segmentations=())
    return SimpleNamespace(label=label, name=f"{rig}@{label}", frame=frame, camera_to_base=camera_to_base, depth=depth)


def hande_tree(**place: Any) -> Any:
    """The robot tree of the repository's Hand-E profile, its ``robot.place`` set as ``place`` says."""
    from src.config import load_tree
    from src.config.schema.robot.place_schema import RobotPlaceConfig

    loaded = load_tree("hande")
    assert loaded.ok, loaded
    robot = loaded.robot
    return robot.model_copy(update={"place": RobotPlaceConfig(**place)})


class AgainWorld:
    """The arm's live planner world as a place over the rim meets it: it holds a pick's frames, lets them go, and holds
    the last pick's frames again (``again`` False: it has none to hold)."""

    def __init__(self, log: list[Any], *, again: bool = True) -> None:
        self.log = log
        self.again = again
        self.holding = False

    def hold_pick_views(self) -> bool:
        self.log.append(("hold",))
        self.holding = True
        return True

    def forget_pick_views(self) -> None:
        self.log.append(("forget",))
        self.holding = False

    def hold_pick_views_again(self) -> bool:
        self.log.append(("hold_again",))
        self.holding = self.again
        return self.again

    @property
    def holds_pick_views(self) -> bool:
        return self.holding

    def offer_segmentation(self, **_: Any) -> None:
        self.log.append(("offer",))

    def forget_segmentation(self) -> None:
        self.log.append(("forget_offer",))


@dataclass
class _Looked:
    """What one scripted pick's looks came to: the part's cloud, the surface under it, the looks' frames."""

    cloud: "np.ndarray | None" = None
    support: Any = None
    views: tuple[Any, ...] = ()


class LookingService(TaskService):
    """``TaskService`` whose picks leave what their looks came to (``looked_around``): ``looked`` answers it per pick, the
    pick's number from 1, or it is the same for every pick."""

    def __init__(self, *args: Any, looked: "Callable[[int], Any] | Any" = None, **keywords: Any) -> None:
        super().__init__(*args, **keywords)
        self.looked = looked
        self.played = 0

    def _play(self, pick: Pick, *args: Any, **keywords: Any) -> Any:
        report = super()._play(pick, *args, **keywords)
        self.played += 1
        looked = self.looked(self.played) if callable(self.looked) else self.looked
        if looked is not None and pick.kind == "part":
            cloud = looked.cloud if looked.cloud is not None else pick.cloud
            self.looked_around = SimpleNamespace(
                judged=SimpleNamespace(target_cloud_base_mm=cloud, support_model=looked.support),
                views=tuple(looked.views))
        return report


def what_the_looks_saw(cloud: "np.ndarray | None" = None, support: Any = None,
                       views: Iterable[Any] = ()) -> _Looked:
    """What a pick's looks came to, for :class:`LookingService`."""
    return _Looked(cloud=cloud, support=support, views=tuple(views))


def run_placing(picks: Any = ("part",), *, place: Any = None, scope: Any = "once", return_to: str = "home",
                options: Any = None, hooks: "RecordingHooks | None" = None, arm: "TaskArm | None" = None,
                jaws: Any = None, locators: Any = UNSET, poses: "Mapping[str, JointPositions] | None" = None,
                looked_around: Any = None, **service_keywords: Any) -> Ran:
    """``tests._task_fakes.run`` on a :class:`LookingService` whose picks leave ``looked_around``."""
    from src.robot.execution.task import PlaceAt, TaskOptions, TaskPlan, run_task

    log: list[Any] = arm.log if arm is not None else []
    arm = arm if arm is not None else TaskArm(log)
    jaws = jaws if jaws is not None else toggle(log, arm=arm)
    service = LookingService(arm, jaws, picks, looked=looked_around, **service_keywords)
    hooks = hooks if hooks is not None else RecordingHooks()
    recorded = hooks.on_event

    def mark(name: str, data: dict[str, Any]) -> None:
        log.append(("event", name))
        if recorded is not None:
            recorded(name, data)

    hooks.on_event = mark
    plan = TaskPlan(object="red cube", place=place if place is not None else PlaceAt(pose="drop_left"),
                    return_to=return_to, scope=scope, options=options if options is not None else TaskOptions())
    keywords: dict[str, Any] = {} if not chosen(locators) else {"locators": locators}
    report = run_task(service, plan, hooks=hooks, poses=POSES if poses is None else poses, **keywords)
    return Ran(report=report, log=log, arm=arm, jaws=jaws, service=service, hooks=hooks)


#: Where a scripted pick's tool closed: on the 40 mm part at :data:`tests._task_fakes.PART_XY`, 20 mm over the bench.
GRASP = Pose.tool_down(PART_XY[0], PART_XY[1], GRASP_Z_MM, label="approach_01")
