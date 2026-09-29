"""A wrist D415 looking at boxes on a bench, rendered by ray casting: the shared scene of the wrist-look tests.

The owner's cell in miniature. The camera rides the tool through a hand-eye mount like the owner's (60 mm beside the
TCP, 130 mm behind it, tilted 45 degrees outward, the stand-in ``test_the_camera_world_lets_the_owners_pick_finish``
uses), and every frame is rendered from where the tool stands when it is taken and stamped with that tool pose, as a
real wrist cell stamps its frames at the shutter. The image is a D415 colour camera at half resolution, 640 x 360 with
its field of view, so a 40 mm part at 450 mm spans some 40 pixels and a jaw's contact face holds hundreds of points.

A look is a :class:`~src.robot.core.JointPositions` the arm maps to a tool pose (:class:`LookingArm`), which is what a
program's taught joints are to a real arm. The poses come from :func:`camera_looking_at`: a camera ``range_mm`` from
the part, at a bearing round it and an elevation over it, looking at it.

Only the geometry is modelled: depth along each ray to the nearest box or the bench, a mask per box that the ray hit,
nothing else. The masks are exact, so no mask bleeds past its part; the tests that care about bleeding say so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.geometry import Frame, Pose, Transform
from src.robot.core import JointPositions, MotionCommand, MotionResult, MotionStatus
from src.robot.core.errors import CameraWorldUnavailable
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.perception import PerceptionFrame

#: A D415 colour camera at half its 1280 x 720, focal length in pixels.
WIDTH, HEIGHT, FOCAL = 640, 360, 925.0 / 2.0
K = np.array([[FOCAL, 0.0, WIDTH / 2.0], [0.0, FOCAL, HEIGHT / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
#: Where a wrist look stands from the part, about where the owner's D415 works.
RANGE_MM = 450.0
#: The part every test picks: a 40 mm cube on the bench, and where its centre is.
CUBE_LOW, CUBE_HIGH = (-20.0, -720.0, 0.0), (20.0, -680.0, 40.0)
CUBE_CENTRE = (0.0, -700.0, 20.0)


def mount() -> np.ndarray:
    """CAMERA to TOOL: 60 mm along the tool's +x, 130 mm behind the TCP, turned 45 degrees toward +x."""
    half = math.sqrt(0.5)
    matrix = np.eye(4)
    matrix[:3, :3] = [[half, 0.0, half], [0.0, 1.0, 0.0], [-half, 0.0, half]]
    matrix[:3, 3] = (60.0, 0.0, -130.0)
    return matrix


def camera_looking_at(
    target: tuple[float, float, float] = CUBE_CENTRE, *, bearing_deg: float, elevation_deg: float = 45.0,
    range_mm: float = RANGE_MM,
) -> np.ndarray:
    """CAMERA to BASE of a camera ``range_mm`` from ``target``, at ``bearing_deg`` round it (0 on its +x side, 90 on
    its +y side) and ``elevation_deg`` over it, looking at it; the image's x axis is horizontal."""
    b, e = math.radians(bearing_deg), math.radians(elevation_deg)
    centre = np.asarray(target, dtype=np.float64)
    position = centre + range_mm * np.array([math.cos(e) * math.cos(b), math.cos(e) * math.sin(b), math.sin(e)])
    z = (centre - position) / np.linalg.norm(centre - position)
    x = np.cross(z, (0.0, 0.0, 1.0))
    x /= np.linalg.norm(x)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    matrix[:3, 3] = position
    return matrix


def tool_for(camera_to_base: np.ndarray) -> Pose:
    """The tool pose that puts the wrist camera at ``camera_to_base``."""
    return Pose.from_matrix(camera_to_base @ np.linalg.inv(mount()), frame=Frame.BASE)


def camera_to_tool() -> Transform:
    return Transform.from_matrix(mount(), from_frame=Frame.CAMERA, to_frame=Frame.TOOL)


@dataclass(frozen=True)
class Box:
    """An axis-aligned box standing in the scene, and what a detector calls it."""

    low: tuple[float, float, float]
    high: tuple[float, float, float]
    label: str

    def holds(self, points: np.ndarray, margin_mm: float = 1.0) -> np.ndarray:
        """Which of ``points`` (BASE, (N, 3)) lie in this box, ``margin_mm`` round it."""
        low = np.asarray(self.low) - margin_mm
        high = np.asarray(self.high) + margin_mm
        return np.all((points >= low) & (points <= high), axis=1)


CUBE = Box(CUBE_LOW, CUBE_HIGH, "part")


def render(camera_to_base: np.ndarray, boxes: tuple[Box, ...]) -> tuple[np.ndarray, np.ndarray]:
    """The depth along every ray to the nearest box or the bench (0 where it meets nothing), and which box it met
    (-1 the bench, -2 nothing)."""
    cols, rows = np.meshgrid(np.arange(WIDTH, dtype=np.float64), np.arange(HEIGHT, dtype=np.float64))
    rays = np.stack([(cols - K[0, 2]) / FOCAL, (rows - K[1, 2]) / FOCAL, np.ones_like(cols)], axis=-1)
    rays = rays @ camera_to_base[:3, :3].T
    origin = camera_to_base[:3, 3]
    hit = np.full((HEIGHT, WIDTH), -2, dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = -origin[2] / rays[..., 2]
        depth = np.where(np.isfinite(depth) & (depth > 0.0), depth, np.inf)
        hit[np.isfinite(depth)] = -1
        for index, box in enumerate(boxes):
            t_low = (np.asarray(box.low) - origin) / rays
            t_high = (np.asarray(box.high) - origin) / rays
            near = np.nanmax(np.minimum(t_low, t_high), axis=-1)
            far = np.nanmin(np.maximum(t_low, t_high), axis=-1)
            meets = (far >= near) & (near > 0.0) & (near < depth)
            depth = np.where(meets, near, depth)
            hit[meets] = index
    return np.where(np.isfinite(depth), depth, 0.0), hit


@dataclass(eq=False)
class _Segment:
    mask: np.ndarray
    label: str
    score: float = 0.9


@dataclass(eq=False)
class WristCamera:
    """The wrist D415: renders the scene from where the arm's tool stands and stamps the frame with that tool pose.

    ``labels`` renames a box in one frame, by the frame's number (0 for the first frame taken): what a detector that
    calls one part two things does. ``stamp_error_mm`` shifts one frame's stamped tool pose, by frame number, with the
    picture taken where the tool really stood: a hand-eye that no longer places the camera where it is. ``taken`` is
    every tool pose a frame was rendered from.
    """

    arm: Any
    boxes: tuple[Box, ...] = (CUBE,)
    labels: dict[int, dict[str, str]] = field(default_factory=dict)
    stamp_error_mm: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    taken: list[Pose] = field(default_factory=list)
    rig_id: str = "wrist"

    def acquire(self) -> PerceptionFrame:
        number = len(self.taken)
        tool = self.arm.get_tcp_pose()
        self.taken.append(tool)
        depth, hit = render(tool.to_matrix() @ mount(), self.boxes)
        renamed = self.labels.get(number, {})
        segmentations = tuple(
            _Segment(mask=(hit == index), label=renamed.get(box.label, box.label))
            for index, box in enumerate(self.boxes)
            if np.any(hit == index)
        )
        stamped = tool
        if number in self.stamp_error_mm:
            matrix = tool.to_matrix().copy()
            matrix[:3, 3] += np.asarray(self.stamp_error_mm[number], dtype=np.float64)
            stamped = Pose.from_matrix(matrix, frame=Frame.BASE)
        return PerceptionFrame(depth_map=depth, intrinsics=K.copy(), segmentations=segmentations,
                               rgb=np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8), timestamp=float(number),
                               tool_pose=stamped)


def joints_key(joints: JointPositions) -> tuple[float, ...]:
    return tuple(round(value, 1) for value in joints.degrees())


class LookingArm(DummyRobotArm):
    """A desk arm whose joint moves put the tool where a look's table says, and which writes every motion down.

    ``refuse`` holds the looks it refuses the way a real arm's gate refuses a pose outside its cable window;
    ``unvouched`` the looks on whose way the camera world cannot vouch.
    """

    def __init__(self, looks: dict[JointPositions, Pose], *, start: Pose | None = None,
                 refuse: tuple[JointPositions, ...] = (), unvouched: tuple[JointPositions, ...] = ()) -> None:
        super().__init__(initial_pose=start if start is not None else next(iter(looks.values()), None))
        self.table = {joints_key(joints): pose for joints, pose in looks.items()}
        self.refused_keys = {joints_key(joints) for joints in refuse}
        self.unvouched_keys = {joints_key(joints) for joints in unvouched}
        self.motions: list[tuple[str, Any]] = []
        self.connect()

    def move_to_joints(self, joints: JointPositions, **keywords: Any) -> MotionResult:
        key = joints_key(joints)
        self.motions.append(("joints", key))
        if key in self.unvouched_keys:
            raise CameraWorldUnavailable(camera="wrist", verdict="blind", attempts=4,
                                         reason="the wrist camera could not vouch for the cell on the way")
        if key in self.refused_keys:
            return MotionResult.failed(MotionStatus.WORKSPACE_REJECTED, MotionCommand.MOVE_JOINTS,
                                       target_joints=joints, message="outside the cable window")
        result = super().move_to_joints(joints, **keywords)
        self._tcp = self.table[key]
        return result

    def move(self, pose: Any, **keywords: Any) -> MotionResult:
        self.motions.append(("move", tuple(round(float(v), 1) for v in pose.position_mm)))
        return super().move(pose, **keywords)


class LookCalculator:
    """Grasps the part, or finds no candidate, by the frame it is ranking, and keeps what every call was handed.

    ``grasps_on`` are the frame numbers on which the part gets its grasp: centred in the cube, straight down, closing
    along base x, so its jaw faces are the cube's -x (jaw 1) and +x (jaw 2) faces. Every other object, and the part on
    every other frame, gets no candidate with a rescan reason, as the real calculator reports it. On the frames in
    ``uncertain_on`` the part's grasp is valid and carries ``RESCAN_RECOMMENDED``: a grasp the calculator is not sure
    of. ``calls`` keeps, per call, the frame number, the label and the keywords.
    """

    render_debug_images = False

    def __init__(self, camera: WristCamera, *, grasps_on: tuple[int, ...] | None = None,
                 uncertain_on: tuple[int, ...] = (), camera_matrix: np.ndarray | None = None) -> None:
        self.camera = camera
        self.grasps_on = grasps_on
        self.uncertain_on = uncertain_on
        self.calls: list[tuple[int, str, dict[str, Any]]] = []
        if camera_matrix is not None:
            self.camera_matrix = camera_matrix

    def compute_result(self, seg: Any, *_args: Any, **kwargs: Any) -> GraspResult:
        number = len(self.camera.taken) - 1
        label = str(getattr(seg, "label", ""))
        self.calls.append((number, label, dict(kwargs)))
        uncertain = label in ("part", "bolt") and number in self.uncertain_on
        if uncertain or (label in ("part", "bolt") and (self.grasps_on is None or number in self.grasps_on)):
            grasp = GraspPoint(position=np.array(CUBE_CENTRE), approach=np.array([0.0, 0.0, -1.0]),
                               axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=40.0, score=0.9, frame=GraspFrame.BASE,
                               label=label)
            return GraspResult(candidates=(grasp,), top_score=0.9,
                               reasons=(GraspFailureReason.RESCAN_RECOMMENDED,) if uncertain else ())
        return GraspResult(reasons=(GraspFailureReason.NO_CANDIDATES_GENERATED,
                                    GraspFailureReason.RESCAN_RECOMMENDED))

    def calls_on(self, number: int, label: str = "part") -> list[dict[str, Any]]:
        return [kwargs for taken, named, kwargs in self.calls if taken == number and named == label]


def base_points(kwargs: dict[str, Any], key: str) -> np.ndarray:
    """A cloud a call was handed, in BASE: ``geometry_points_base_mm`` as it is, a CAMERA cloud placed by the call's
    own ``camera_to_base``."""
    points = np.asarray(kwargs[key], dtype=np.float64).reshape(-1, 3)
    if key.endswith("_base_mm"):
        return points
    matrix = np.asarray(kwargs["camera_to_base"].to_matrix(), dtype=np.float64)
    return points @ matrix[:3, :3].T + matrix[:3, 3]
