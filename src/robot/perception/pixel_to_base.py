"""A pixel of your own in BASE: one frame of a camera, the point a pixel of it shows, and a TCP pose over that point.

A program that finds what it wants in the camera image itself (a pen stroke on paper, a mark, a marker's corner) needs
no detector. It needs the frame and where each of its pixels lies in the robot's base frame. :func:`capture` takes one
frame as a pick takes its frame, through the ``Locator``'s own source: the warm-ups, the burst of grabs whose median is
the depth, the rig's mixed pixels taken out, and on the wrist the tool pose read before and after the burst, the arm at
rest by its own steady gate first. It keeps the frame whole (:class:`CapturedFrame`): the colour image, the depth
aligned to it, the lens, the TCP the arm read at the shutter and the transforms that place it.

:func:`pixel_to_base` finds one pixel in BASE (:class:`PixelPoint`), and :func:`pixels_to_base` a whole stroke. Never by
the depth of the one pixel: a plane is fit to the depth measured in the window about it, :data:`WINDOW_PX` pixels
across, and the pixel's viewing ray meets that plane, so a thin stroke with no depth of its own stands on the paper it
was drawn on. The plane is the surface the pixel itself measured, where it measured a depth, and it has to hold a
quarter of the window; a pixel with no depth of its own (a thin stroke, a hole) stands on the surface that holds more
than half of what its window measured. Each is found among planes through three of the window's points, a fixed draw
so one frame always gives one answer. Every point is judged by its distance from a plane along its own ray, as the
camera measures it, and the plane chosen is the one whose nearest quarter of the window (or half) lies nearest it, a
least quantile of squares: a bevel across a part's edge holds both faces within the band, and loses to the face that
holds its own points exactly. It is then fit by least squares to the points within :data:`SURFACE_BAND_MM` of it, and
again with the band narrowed to its own points' deviation, which lets go of a strip of the face beyond an edge. Handed
``plane=``, the ray meets that plane instead (the mat by its height, or a point and a normal), and no depth is read.

:func:`tcp_above` proposes a TCP pose over a point: perpendicular to its surface, ``offset_mm`` above it, the tool's +X
along base -Y as the owner's poses stand; or the fingertips there instead of the TCP. Nothing in this module moves the
arm or sends it anything: a program that wants the arm at a proposed pose drives it there through ``Robot.move`` and
its guards, once a person said so.

What cannot be found in BASE raises :class:`PixelRefused` with its reason: a pixel outside the image, too little depth
about it, no one surface there, a ray along the plane, or a plane behind the camera. Never a silent 0.

At import this loads numpy and the geometry alone; :func:`capture` loads the locator when it runs.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from src.contracts import UNSET, Maybe, chosen
from src.geometry import Frame, Pose

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import RobotConfig

__all__ = [
    "LEAST_RAY_ANGLE_DEG",
    "SURFACE_BAND_MM",
    "WINDOW_PX",
    "CapturedFrame",
    "PixelPoint",
    "PixelRefused",
    "capture",
    "pixel_to_base",
    "pixels_to_base",
    "tcp_above",
]

#: How many pixels across, and down, the window is whose depth a pixel is found by: the owner's "about 15 x 15"
#: (2026-10-14). About 7 mm of a surface 450 mm from a D415's colour camera at 1280 x 720, 11 mm at 848 x 480.
WINDOW_PX = 15

#: How far from a plane, along its own ray, a point of the window may lie and still count toward that surface while the
#: surface is found, millimetres of depth. A choice, not a measurement: wide enough for a RealSense's depth noise and
#: its whole millimetres where a wrist camera works, under a metre, narrow enough that a part 10 mm tall and what it
#: stands on are two surfaces. The plane found is then fit to a band that narrows to its own points' deviation.
SURFACE_BAND_MM = 5.0

#: The shallowest angle at which a viewing ray still meets a plane, degrees. Below it the ray runs along the plane, and
#: the point it meets moves by metres for a tenth of a degree of the camera's turn. A choice, not a measurement.
LEAST_RAY_ANGLE_DEG = 1.0

#: The least share of its window a plane is fit to, and that the plane of the surface a pixel measured has to hold:
#: fewer measured pixels than a quarter of the window say where a few pixels are, not where a surface is. A choice, not
#: a measurement.
_LEAST_SHARE = 0.25

#: How many planes through three of the window's points are drawn, in each family: through any three, and through the
#: pixel's own point and two others. At 256 the draws through the pixel's own point miss a surface that holds a quarter
#: of the window with a chance of 7e-8 (0.9375^256), and the draws through any three miss one that holds half of it with
#: a chance of 1e-15 (0.875^256).
_DRAWS = 256

#: The fixed seed of the draws, so one frame always gives one answer.
_SEED = 0

#: Below this, mm squared, three points of a draw span no plane (two of them the same, or the three on a line).
_SPANS_NO_PLANE_MM2 = 1e-6

#: How many times a plane is fit, at most, each time to the points its band then holds.
_REFITS = 8

#: How far the band of a plane being fit reaches, in its points' own deviation from it.
_BAND_PER_DEVIATION = 2.5

#: A normal noise's deviation per median distance of its points from their plane: the points' deviation is their median
#: distance along their rays times this.
_MAD_TO_DEVIATION = 1.4826

#: The narrowest that band narrows to, millimetres: the whole millimetre a RealSense's depth is read in.
_LEAST_BAND_MM = 1.0

#: Points whose second widest spread is below this share of their widest lie along a line, which spans no plane.
_ON_A_LINE = 1e-6

#: Why a pixel was not found in BASE (:class:`PixelRefused`).
RefusalReason = Literal["outside", "no_depth", "no_surface", "parallel", "behind"]


class PixelRefused(ValueError):
    """A pixel that cannot be found in BASE, and why: never a silent 0.

    Attributes:
        reason (str): ``"outside"``, the pixel lies outside the image; ``"no_depth"``, the camera measured less than a
            quarter of the window about it; ``"no_surface"``, the window holds no one surface it stands on;
            ``"parallel"``, its viewing ray runs along the plane; ``"behind"``, the plane lies behind the camera.
        pixel (tuple[float, float]): The pixel, ``(u, v)``: its column and its row.
        index (int | None): Its place in the pixels :func:`pixels_to_base` was handed; ``None`` for one pixel.
    """

    def __init__(self, message: str, *, reason: RefusalReason, pixel: tuple[float, float],
                 index: int | None = None) -> None:
        super().__init__(message)
        self.reason: RefusalReason = reason
        self.pixel = pixel
        self.index = index


def _matrix(value: Any, shape: tuple[int, int], what: str) -> np.ndarray:
    """``value`` as a read-only float64 array of ``shape``, every entry finite."""
    array = np.array(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{what} is a finite {shape[0]} x {shape[1]} matrix, not {np.shape(value)}")
    array.setflags(write=False)
    return array


def _xyz(values: Any) -> str:
    return "({:.1f}, {:.1f}, {:.1f})".format(*(float(v) for v in values))


@dataclass(frozen=True, slots=True, eq=False)
class CapturedFrame:
    """One frame of one camera, kept whole, and where it stood when its shutter opened: what :func:`pixel_to_base` finds
    pixels of.

    :func:`capture` takes one; a frame of your own (a recorded one) is built from its fields. Every ``*_to_*_mm`` is a
    4 x 4 in millimetres that maps a point of the first frame into the second, so on the wrist
    ``camera_to_base_mm == tool_to_base_mm @ camera_to_tool_mm``, and its inverse maps the other way.

    Attributes:
        camera (str): The rig id of the camera that took it.
        captured_at_s (float): Its shutter, host seconds on the ``time.time`` clock.
        mounting (str): ``"eye_in_hand"`` (on the wrist) or ``"eye_to_hand"`` (fixed).
        color (np.ndarray): The colour image, ``(H, W, 3)`` BGR ``uint8``, as ``RGBDFrame.color``.
        depth_mm (np.ndarray): The depth aligned to the colour image, ``(H, W)``: millimetres along the camera's +Z, 0
            where nothing was measured; a pick frame's, the median of its burst of grabs.
        intrinsics (np.ndarray): The colour image's 3 x 3 pinhole matrix, in pixels.
        camera_to_base_mm (np.ndarray): Where the camera stood at the shutter: CAMERA to BASE, 4 x 4.
        tcp_pose (Pose | None): The TCP at the shutter as the arm reads it (``robot.arm.get_tcp_pose``), in BASE;
            ``None`` for a fixed camera (default: None).
        camera_to_tool_mm (np.ndarray | None): The camera's hand-eye calibration, CAMERA to TOOL, 4 x 4, as the rig's
            artifact holds it (``camera.calibration()``); ``None`` for a fixed camera (default: None).
    """

    camera: str
    captured_at_s: float
    mounting: str
    color: np.ndarray
    depth_mm: np.ndarray
    intrinsics: np.ndarray
    camera_to_base_mm: np.ndarray
    tcp_pose: Pose | None = None
    camera_to_tool_mm: np.ndarray | None = None

    def __post_init__(self) -> None:
        depth = np.array(self.depth_mm, dtype=np.float64)
        color = np.array(self.color)
        if depth.ndim != 2 or color.ndim != 3 or color.shape[2] != 3 or color.shape[:2] != depth.shape:
            raise ValueError(f"a frame's colour is (H, W, 3) and its depth (H, W) of the same image, not "
                             f"{np.shape(self.color)} and {np.shape(self.depth_mm)}")
        lens = _matrix(self.intrinsics, (3, 3), "a frame's lens")
        if not (lens[0, 0] > 0.0 and lens[1, 1] > 0.0):
            raise ValueError(f"a frame's lens has positive focal lengths, not fx {lens[0, 0]} and fy {lens[1, 1]}")
        if self.mounting not in ("eye_in_hand", "eye_to_hand"):
            raise ValueError(f"a frame's camera is mounted 'eye_in_hand' or 'eye_to_hand', not {self.mounting!r}")
        if self.tcp_pose is not None and (not isinstance(self.tcp_pose, Pose) or self.tcp_pose.frame is not Frame.BASE):
            raise ValueError("a frame's TCP pose is a Pose in BASE, or None")
        depth.setflags(write=False)
        color.setflags(write=False)
        object.__setattr__(self, "depth_mm", depth)
        object.__setattr__(self, "color", color)
        object.__setattr__(self, "intrinsics", lens)
        object.__setattr__(self, "camera_to_base_mm", _matrix(self.camera_to_base_mm, (4, 4), "CAMERA to BASE"))
        if self.camera_to_tool_mm is not None:
            object.__setattr__(self, "camera_to_tool_mm", _matrix(self.camera_to_tool_mm, (4, 4), "CAMERA to TOOL"))

    @property
    def tool_to_base_mm(self) -> np.ndarray | None:
        """Where the tool stood at the shutter: TOOL to BASE, 4 x 4 millimetres, the TCP pose's matrix; ``None`` for a
        fixed camera."""
        return None if self.tcp_pose is None else self.tcp_pose.to_matrix()

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """The frame as a person reads it: the camera, the image, how much of it has a depth, where it stood.

        Returns:
            str: ASCII, no trailing newline.
        """
        height, width = self.depth_mm.shape
        measured = 100.0 * float(np.count_nonzero(self.depth_mm > 0.0)) / max(1, self.depth_mm.size)
        tool = "" if self.tcp_pose is None else f" and the TCP at {_xyz(self.tcp_pose.position_mm)} mm"
        return (f"camera {self.camera!r} ({self.mounting}) captured at {self.captured_at_s:.3f} s: {width} x {height} "
                f"pixels, {measured:.0f} % of them with a depth; the camera at {_xyz(self.camera_to_base_mm[:3, 3])} mm"
                f"{tool} in BASE")

    def to_dict(self) -> dict[str, Any]:
        """The frame as plain data, without its images.

        Returns:
            dict[str, Any]: ``json.dumps`` safe: the camera, the shutter, the image's size, the lens and the transforms.
        """
        return {
            "camera": self.camera,
            "captured_at_s": self.captured_at_s,
            "mounting": self.mounting,
            "width": int(self.depth_mm.shape[1]),
            "height": int(self.depth_mm.shape[0]),
            "intrinsics": self.intrinsics.tolist(),
            "camera_to_base_mm": self.camera_to_base_mm.tolist(),
            "tcp_pose": None if self.tcp_pose is None else {
                "position_mm": self.tcp_pose.position_mm.tolist(),
                "quaternion_xyzw": self.tcp_pose.quaternion_xyzw.tolist()},
            "camera_to_tool_mm": None if self.camera_to_tool_mm is None else self.camera_to_tool_mm.tolist(),
        }


@dataclass(frozen=True, slots=True, eq=False)
class PixelPoint:
    """One pixel found in BASE: the point it shows, the surface there, and what the point rests on.

    Attributes:
        pixel (tuple[float, float]): The pixel, ``(u, v)``: its column and its row, a pixel's centre at its whole index.
        base_mm (np.ndarray): The point, ``(3,)`` BASE millimetres.
        tool_mm (np.ndarray | None): The same point in the TOOL frame at the shutter, ``(3,)`` millimetres; ``None``
            for a fixed camera.
        normal (np.ndarray): The surface's unit normal there, ``(3,)`` in BASE, toward the camera's side of it.
        source (str): ``"depth"``, the plane fit to the depth about the pixel; ``"plane"``, the plane handed in.
        valid_share (float | None): The share of the window's pixels with a depth measured, 0 to 1; ``None`` for a
            plane handed in.
        spread_mm (float | None): How far the window's points the plane rests on lie from it along their rays, their
            root mean square, millimetres; ``None`` for a plane handed in.
        camera_to_base_mm (np.ndarray): Where the camera stood at the shutter: CAMERA to BASE, 4 x 4 millimetres.
    """

    pixel: tuple[float, float]
    base_mm: np.ndarray
    tool_mm: np.ndarray | None
    normal: np.ndarray
    source: str
    valid_share: float | None
    spread_mm: float | None
    camera_to_base_mm: np.ndarray

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """The point as a person reads it: the pixel, where it lies in BASE, its surface and what it rests on.

        Returns:
            str: ASCII, no trailing newline.
        """
        u, v = self.pixel
        facing = "({:.2f}, {:.2f}, {:.2f})".format(*(float(n) for n in self.normal))
        if self.source == "plane":
            rests = "on the plane handed in"
        else:
            rests = (f"from the depth: {100.0 * float(self.valid_share or 0.0):.0f} % of its window measured, the "
                     f"plane's spread {float(self.spread_mm or 0.0):.2f} mm")
        return f"pixel ({u:g}, {v:g}): {_xyz(self.base_mm)} mm in BASE, its surface facing {facing}, {rests}"

    def to_dict(self) -> dict[str, Any]:
        """The point as plain data.

        Returns:
            dict[str, Any]: ``json.dumps`` safe; every array as a list.
        """
        return {
            "pixel": [float(self.pixel[0]), float(self.pixel[1])],
            "base_mm": self.base_mm.tolist(),
            "tool_mm": None if self.tool_mm is None else self.tool_mm.tolist(),
            "normal": self.normal.tolist(),
            "source": self.source,
            "valid_share": self.valid_share,
            "spread_mm": self.spread_mm,
            "camera_to_base_mm": self.camera_to_base_mm.tolist(),
        }


def capture(camera: Any, robot: Any = None) -> CapturedFrame:
    """One frame of an open camera, kept whole, with the arm at rest: what :func:`pixel_to_base` finds pixels of.

    ```python
    tree = load_tree()
    robot = Robot.from_tree(tree, gripper=None)         # the arm alone: nothing is said to the hand
    with Camera.from_tree(tree) as camera, robot.connected():
        frame = capture(camera, robot)                  # nothing moves
    print(pixel_to_base(frame, 412.0, 233.5))
    ```

    The frame is a pick frame, taken through the ``Locator``'s source with no detector (``Locator.capture``). A camera
    on the wrist is placed by the TCP read just before its burst of grabs and held to the rig's shutter tolerance after
    it, and first the arm's own steady gate is asked whether it stands still (``wait_until_steady`` waiting for nothing:
    on a UR whose ``safety.dwell.steady_signal`` is ``joint_speeds``, every joint slower than 0.01 rad/s on three reads
    in a row). Nothing is commanded and nothing is said to the hand.

    A capture on the wrist needs the arm connected, and ``robot.connected()`` takes the cell's lock first (``CellLock``,
    the one the operator console, ``Cell`` and the command line runner take): while another process holds the cell, the
    console connected among them, the connect raises ``CellBusy`` naming that process, and nothing is captured. The
    console keeps its camera open from its build until it builds again or stops, too, so a capture runs with the
    console stopped.

    Args:
        camera (Any): An open camera owner (:class:`Camera`) of an RGB-D rig that declares its calibration and aligns
            its depth to its colour image.
        robot (Any): The connected :class:`Robot` whose wrist carries the camera, built from the tree
            (``Robot.from_tree``), so its tool frame is known; not read for a fixed camera, which is placed by its own
            calibration and never asks the arm (default: None).

    Returns:
        CapturedFrame: The colour image, the depth aligned to it, the lens, where the camera stood at the shutter, and
            on the wrist the TCP the arm read there and the camera's CAMERA to TOOL.

    Raises:
        LocatorRefused: Before any grab: the rig has no depth or does not align it to its colour; a camera on the wrist
            and no robot, a robot that keeps no robot section, or one not connected; a wrist rig solved against
            another flange to TCP than the cell's; an arm its steady gate says is not at rest.
        RigCalibrationError: The rig declares no calibration (``RigNotCalibrated``), or one that does not load.
        PerceptionFrameMoved: The tool moved beyond the rig's shutter tolerance across every grab.
    """
    from src.robot.perception.locator import Locator, LocatorRefused  # noqa: PLC0415 (the locator loads with a capture)

    rig_id = str(getattr(camera, "rig_id", "camera"))
    if getattr(getattr(camera, "rig", None), "align_depth_to_color", True) is False:
        raise LocatorRefused(
            f"camera {rig_id!r} does not align its depth to its colour image (align_depth_to_color: false), so a pixel "
            "of the colour image is no pixel of the depth: set align_depth_to_color true for the rig")
    if str(camera.calibration().mounting_mode) != "eye_in_hand":
        return Locator.from_parts(camera=camera, backend=None).capture()
    if robot is None:
        raise LocatorRefused(
            f"camera {rig_id!r} is on the wrist, and its frame is placed by the TCP read at its shutter: hand capture "
            "the connected robot that carries it, capture(camera, robot)")
    arm: Any = getattr(robot, "arm", None)
    section: Any = getattr(robot, "robot_config", None)
    if section is None:
        section = getattr(arm, "config", None)
    if section is None:
        raise LocatorRefused(
            f"camera {rig_id!r} is on the wrist and this robot keeps no robot section, so the tool frame its "
            "calibration was solved against is unknown: build it with Robot.from_tree(tree, gripper=None)")
    if getattr(arm, "is_connected", False) is not True:
        raise LocatorRefused(
            f"camera {rig_id!r} is on the wrist and the arm is not connected, so where the tool stands cannot be read: "
            "capture inside `with robot.connected():`")
    locator = Locator.from_parts(
        camera=camera, backend=None, tool_pose=arm.get_tcp_pose,
        attempts=int(section.safety.planning_world.perceived.fresh_frame_attempts),
        tool_frame=section.gripper.tool_frame)
    # The arm's own steady gate, asked to wait for nothing, right before the burst: a look's "already there" asks the
    # same (``execution.looks``).
    steady = getattr(arm, "wait_until_steady", None)
    if not callable(steady) or steady(0.0) is not True:
        raise LocatorRefused(
            f"camera {rig_id!r} is on the wrist and the arm is not at rest (its steady gate says it moves), so no "
            "single tool pose places a frame of it: stand the arm still, then capture")
    return locator.capture()


def pixel_to_base(frame: CapturedFrame, u: float, v: float, *, plane: Any = None,
                  window_px: int = WINDOW_PX) -> PixelPoint:
    """Where one pixel of a frame lies in BASE: its viewing ray met with the surface about it, or with a plane you name.

    ```python
    point = pixel_to_base(frame, 412.0, 233.5)               # the depth about the pixel
    on_mat = pixel_to_base(frame, 412.0, 233.5, plane=50.0)  # no depth: the mat at BASE z 50 mm
    print(point.base_mm, point.normal, point.spread_mm)
    ```

    Never by the depth of the one pixel: the ray through the pixel meets a plane fit to the depth measured in the
    ``window_px`` x ``window_px`` window about it, so a stroke with no depth of its own stands on the paper. The plane
    is the surface the pixel itself measured, where it measured a depth, which holds a quarter of the window at least;
    for a pixel with no depth of its own, the surface that holds more than half of what the window measured. Every
    point is judged by its distance from a plane along its own ray, and the plane is fit by least squares to a band
    that narrows to its points' deviation, so a pixel 1 mm from a part's edge gives the face it shows.

    Args:
        frame (CapturedFrame): The frame, as :func:`capture` takes it.
        u (float): The pixel's column, 0 at the left; a pixel's centre at its whole index, and a fraction between.
        v (float): The pixel's row, 0 at the top.
        plane (Any): ``None`` reads the depth. A number is the horizontal plane at that BASE z, millimetres; a pair,
            ``(point, normal)``, is the plane through that BASE point, millimetres, with that normal. A plane reads no
            depth (default: None).
        window_px (int): How many pixels across, and down, the window is; odd, 3 or more (default: 15).

    Returns:
        PixelPoint: The point in BASE and in the TOOL frame at the shutter, the surface's normal, whether the depth or
            your plane found it, the window's share of measured pixels and the plane's spread, and the camera's pose.

    Raises:
        PixelRefused: The pixel lies outside the image, its window has too little depth or no one surface it stands on,
            its ray runs along the plane, or the plane lies behind the camera; ``reason`` says which.
        ValueError: A pixel that is not two finite numbers, a window that is not odd or under 3, or a plane that is no
            number and no point and normal.
    """
    return _found(frame, (float(u), float(v)), plane=_plane_of(plane), half=_half_window(window_px))


def pixels_to_base(frame: CapturedFrame, pixels: Any, *, plane: Any = None,
                   window_px: int = WINDOW_PX) -> tuple[PixelPoint, ...]:
    """Where many pixels of one frame lie in BASE, in order, each as :func:`pixel_to_base` finds it: a whole stroke.

    ```python
    stroke = pixels_to_base(frame, contour)        # (N, 2) pixels, or an OpenCV contour of (N, 1, 2)
    path_mm = np.array([point.base_mm for point in stroke])
    ```

    Every pixel or none: the first that cannot be found raises, naming its place in the list. A program that keeps
    the others calls :func:`pixel_to_base` per pixel and catches :class:`PixelRefused`.

    Args:
        frame (CapturedFrame): The frame, as :func:`capture` takes it.
        pixels (Any): The pixels, ``(u, v)`` each: a list of pairs, an ``(N, 2)`` array or an OpenCV contour.
        plane (Any): ``None`` reads the depth; a BASE z or a ``(point, normal)`` pair, as :func:`pixel_to_base` takes
            it (default: None).
        window_px (int): How many pixels across, and down, each window is; odd, 3 or more (default: 15).

    Returns:
        tuple[PixelPoint, ...]: One point per pixel, in the order given; ``()`` for no pixels.

    Raises:
        PixelRefused: A pixel that cannot be found in BASE; ``index`` is its place in the list, ``reason`` why.
        ValueError: Pixels that are not pairs of finite numbers, or a window or a plane :func:`pixel_to_base` refuses.
    """
    array = np.asarray(pixels, dtype=np.float64)
    if array.size == 0:
        return ()
    if array.shape[-1] != 2:
        raise ValueError(f"pixels are (u, v) pairs, an (N, 2) array or an (N, 1, 2) contour, not {array.shape}")
    pairs = array.reshape(-1, 2)
    resolved, half = _plane_of(plane), _half_window(window_px)
    return tuple(_found(frame, (float(u), float(v)), plane=resolved, half=half, index=index, count=len(pairs))
                 for index, (u, v) in enumerate(pairs))


def tcp_above(point: Any, normal: Any, offset_mm: float, *, closing_axis: str = "-y",
              fingertips_of: "Maybe[RobotConfig]" = UNSET) -> Pose:
    """A TCP pose proposed over a point: perpendicular to its surface, ``offset_mm`` above it. It only proposes.

    ```python
    point = pixel_to_base(frame, 412.0, 233.5)
    above = tcp_above(point.base_mm, point.normal, 20.0)                          # the TCP 20 mm above
    tips = tcp_above(point.base_mm, point.normal, 5.0, fingertips_of=tree.robot)  # the fingertips 5 mm above
    ```

    The tool's +Z points along the normal, down onto the point, and its +X as near to the closing axis as a pose
    perpendicular to the surface lets it (``Pose.aimed_at``); over a level surface the pose is ``Pose.tool_down`` with
    that axis. Nothing moves and nothing is sent to the arm: a program that wants it there drives it through
    ``robot.move`` and its guards, once a person said so.

    Args:
        point (Any): The point, three BASE millimetres: a :class:`PixelPoint`'s ``base_mm``.
        normal (Any): The surface's normal there, three numbers in BASE, toward the side the tool stands on: a
            :class:`PixelPoint`'s ``normal``. Its length is not read.
        offset_mm (float): How far above the surface the TCP stands, along the normal, millimetres; 0 or more.
        closing_axis (str): The axis the tool's +X lines up with, one of ``Pose``'s closing axes (``"x"``, ``"-y"``,
            ``"radial"``, ...); the owner's poses stand along ``"-y"`` (default: "-y").
        fingertips_of (Maybe[RobotConfig]): The cell's robot section, ``tree.robot``: then the fingertips of the hand
            it names stand ``offset_mm`` above the point, and the TCP as far behind them as they reach past it along the
            approach (the jaw's ``finger_ahead_mm`` past its grasp centre, and that centre past the TCP). Unset stands
            the TCP there: the one ``robot.move`` places and ``robot.arm.get_tcp_pose`` reads, the controller's active
            TCP on a ``polyscope`` tool frame (default: UNSET).

    Returns:
        Pose: The proposed TCP pose, in BASE.

    Raises:
        ValueError: A point or a normal that is not three finite numbers, a normal of no length, an offset below 0 or
            not finite, a closing axis that names none or runs along the normal, or ``fingertips_of`` a section naming
            no hand.
        ConfigError: ``fingertips_of`` names a hand the registry does not hold or cannot place.
    """
    at = _three(point, "the point")
    up = _three(normal, "the normal")
    length = float(np.linalg.norm(up))
    if length <= 0.0:
        raise ValueError("the normal has no length, so it names no direction to stand above the point in")
    up = up / length
    offset = float(offset_mm)
    if not math.isfinite(offset) or offset < 0.0:
        raise ValueError(f"the TCP stands 0 mm or more above the surface, not {offset_mm!r}: below 0 it stands in it")
    reach = _fingertips_past_the_tcp_mm(fingertips_of) if chosen(fingertips_of) else 0.0
    tcp = at + up * (offset + reach)
    stands = "the fingertips" if chosen(fingertips_of) else "the TCP"
    return Pose.aimed_at(float(tcp[0]), float(tcp[1]), float(tcp[2]), target_mm=tcp - up,
                         closing_axis=closing_axis, label=f"{stands} {offset:g} mm above the point")


# ---------------------------------------------------------------------------------------------------------------------
# How a pixel is found
# ---------------------------------------------------------------------------------------------------------------------


def _three(value: Any, what: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{what} is three finite numbers, not {value!r}")
    return array.reshape(3)


def _plane_of(plane: Any) -> "tuple[np.ndarray, np.ndarray] | None":
    """``plane`` as a BASE point and a unit normal, or ``None`` where the depth is read."""
    if plane is None:
        return None
    if isinstance(plane, (int, float, np.integer, np.floating)) and not isinstance(plane, bool):
        z = float(plane)
        if not math.isfinite(z):
            raise ValueError(f"a horizontal plane is a finite BASE z in millimetres, not {plane!r}")
        return np.array([0.0, 0.0, z]), np.array([0.0, 0.0, 1.0])
    try:
        point, normal = plane
    except (TypeError, ValueError):
        raise ValueError(f"a plane is a BASE z in millimetres, or a (point, normal) pair, not {plane!r}") from None
    through, facing = _three(point, "the plane's point"), _three(normal, "the plane's normal")
    length = float(np.linalg.norm(facing))
    if length <= 0.0:
        raise ValueError("the plane's normal has no length, so it names no plane")
    return through, facing / length


def _half_window(window_px: Any) -> int:
    """How far the window reaches each way of its pixel, from how many pixels across it is."""
    whole = isinstance(window_px, (int, np.integer)) and not isinstance(window_px, bool)
    if not whole or window_px < 3 or window_px % 2 == 0:
        raise ValueError(f"a window is an odd number of pixels across, 3 or more, so its pixel stands in its middle; "
                         f"not {window_px!r}")
    return int(window_px) // 2


def _found(frame: CapturedFrame, pixel: tuple[float, float], *, plane: "tuple[np.ndarray, np.ndarray] | None",
           half: int, index: int | None = None, count: int = 1) -> PixelPoint:
    """``pixel`` of ``frame`` found in BASE, on ``plane`` (a BASE point and a unit normal) or, ``None``, on the surface
    its window measured; the ``index``-th of ``count`` pixels where a list was handed."""
    u, v = pixel
    if not (math.isfinite(u) and math.isfinite(v)):
        raise ValueError(f"a pixel is two finite numbers, its column and its row, not {pixel!r}")
    named = f"pixel ({u:g}, {v:g})" if index is None else f"pixel {index} of {count}, ({u:g}, {v:g}),"

    def refused(reason: RefusalReason, why: str) -> PixelRefused:
        return PixelRefused(f"{named} of camera {frame.camera!r} {why}", reason=reason, pixel=pixel, index=index)

    height, width = frame.depth_mm.shape
    col, row = math.floor(u + 0.5), math.floor(v + 0.5)
    if not (0 <= col < width and 0 <= row < height):
        raise refused("outside", f"lies outside its {width} x {height} image: a pixel's column runs from 0 to "
                                 f"{width - 1} and its row from 0 to {height - 1}")
    lens = frame.intrinsics
    ray = np.array([(u - lens[0, 2]) / lens[0, 0], (v - lens[1, 2]) / lens[1, 1], 1.0])
    rotation, origin = frame.camera_to_base_mm[:3, :3], frame.camera_to_base_mm[:3, 3]
    share = spread = None
    if plane is None:
        normal, offset, share, spread = _surface_about(frame, col, row, half, refused)
    else:
        # The plane n . x = n . p0 in BASE, with x = R y + t, is (R^T n) . y = n . (p0 - t) in CAMERA.
        through, facing = plane
        normal, offset = rotation.T @ facing, float(facing @ (through - origin))
    if offset > 0.0:
        # The normal toward the camera's side: the camera stands at CAMERA's origin, where n . y - d is then above 0.
        normal, offset = -normal, -offset
    along = float(normal @ ray)
    angle = math.degrees(math.asin(min(1.0, abs(along) / float(np.linalg.norm(ray)))))
    if angle < LEAST_RAY_ANGLE_DEG:
        raise refused("parallel", f"looks along the plane: its viewing ray meets it at {angle:.3f} deg, under the "
                                  f"{LEAST_RAY_ANGLE_DEG:g} deg a ray meets a plane at")
    distance = offset / along
    if not distance > 0.0:
        raise refused("behind", "looks away from the plane: the plane lies behind the camera along its viewing ray, or "
                                "runs through the camera")
    base = rotation @ (distance * ray) + origin
    tool = frame.tool_to_base_mm
    in_tool = None if tool is None else (np.linalg.inv(tool) @ np.append(base, 1.0))[:3]
    return PixelPoint(pixel=pixel, base_mm=base, tool_mm=in_tool, normal=rotation @ normal,
                      source="depth" if plane is None else "plane", valid_share=share, spread_mm=spread,
                      camera_to_base_mm=frame.camera_to_base_mm)


def _surface_about(frame: CapturedFrame, col: int, row: int, half: int,
                   refused: Callable[[RefusalReason, str], PixelRefused]) -> tuple[np.ndarray, float, float, float]:
    """The plane of the surface the window about ``(col, row)`` shows, in CAMERA (``n . y = d``, ``n`` a unit normal),
    the share of the window measured, and the root mean square of the plane's points off it; ``refused(reason, why)``
    is raised where there is none."""
    height, width = frame.depth_mm.shape
    top, left = max(0, row - half), max(0, col - half)
    patch = frame.depth_mm[top:min(height, row + half + 1), left:min(width, col + half + 1)]
    with np.errstate(invalid="ignore"):
        measured = np.isfinite(patch) & (patch > 0.0)
    least = math.ceil(_LEAST_SHARE * patch.size)
    count = int(np.count_nonzero(measured))
    if count < least:
        raise refused("no_depth", f"has too little depth about it: the camera measured {count} of the {patch.size} "
                                  f"pixels of its window, fewer than the quarter a surface is fit to. Name the plane "
                                  f"it lies on instead (plane=)")
    rows, cols = np.nonzero(measured)
    z = patch[rows, cols]
    rows, cols = rows + top, cols + left
    lens = frame.intrinsics
    points = np.column_stack(((cols - lens[0, 2]) * z / lens[0, 0], (rows - lens[1, 2]) * z / lens[1, 1], z))
    rays = points / z[:, None]
    hit = np.flatnonzero((rows == row) & (cols == col))
    own = int(hit[0]) if hit.size else None
    normals, offsets = _draws(points, own)
    off = _off_along(rays, z, normals, offsets)
    if own is not None:
        # The surface the pixel measured: of the planes whose band holds its own depth, the one whose nearest quarter of
        # the window lies nearest it (a least quantile of squares). A bevel across an edge holds both faces within the
        # band, and only the face holds its own points exactly.
        rank, candidates = least, off[own] <= SURFACE_BAND_MM
    else:
        # The surface more than half of the window measured: the plane whose nearer half lies nearest it.
        rank, candidates = count // 2 + 1, np.ones(off.shape[1], dtype=bool)
    nearest = np.where(candidates, np.partition(off, rank - 1, axis=0)[rank - 1], np.inf)
    if not (nearest.size and float(nearest.min()) <= SURFACE_BAND_MM):
        if own is not None:
            raise refused("no_surface", f"measured a depth that lies on no surface of its window: no plane through it "
                                        f"holds a quarter of the window's {patch.size} pixels within "
                                        f"{SURFACE_BAND_MM:g} mm (an edge, a step or a stray reading)")
        raise refused("no_surface", f"has no depth of its own, and its window holds no one surface: no plane holds "
                                    f"more than half of its {count} measured points within {SURFACE_BAND_MM:g} mm (an "
                                    f"edge or a step)")

    def fit(selected: np.ndarray) -> tuple[np.ndarray, float]:
        plane = _fit(points[selected])
        if plane is None:
            raise refused("no_surface", "has a window whose measured points lie along a line, which spans no plane")
        return plane

    # Fit by least squares to the points the band holds, and again with the band narrowed to the plane's own deviation,
    # until it holds what it held: a strip of the face beyond an edge the wide band took in is let go of.
    on = off[:, int(np.argmin(nearest))] <= SURFACE_BAND_MM
    for _ in range(_REFITS):
        normal, offset = fit(on)
        distance = _off_along(rays, z, normal[None, :], np.array([offset]))[:, 0]
        deviation = _MAD_TO_DEVIATION * float(np.median(distance[on]))
        again = distance <= min(SURFACE_BAND_MM, max(_LEAST_BAND_MM, _BAND_PER_DEVIATION * deviation))
        if np.array_equal(again, on) or int(np.count_nonzero(again)) < 3:
            break
        on = again
    spread = math.sqrt(float(np.mean(distance[on] ** 2)))
    return normal, offset, count / patch.size, spread


def _off_along(rays: np.ndarray, depth: np.ndarray, normals: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """How far each point lies from each plane along its own ray, millimetres of depth, as the camera measures:
    ``(N, H)`` for ``N`` points (their rays at depth 1 and their depths) and ``H`` planes (``n . y = d``); infinite
    where a ray runs along a plane. Not the distance straight to the plane: a plane that runs along the rays at an edge
    would hold a strip of each surface the edge parts."""
    with np.errstate(divide="ignore", invalid="ignore"):
        off = np.abs(depth[:, None] - offsets[None, :] / (rays @ normals.T))
    return np.where(np.isfinite(off), off, np.inf)


def _draws(points: np.ndarray, own: int | None) -> tuple[np.ndarray, np.ndarray]:
    """The planes through three of ``points`` each, a fixed draw, every triple that spans one: their unit normals
    ``(H, 3)`` and offsets ``(H,)``; through any three, and through the pixel's own point ``own`` and two others."""
    draw = np.random.default_rng(_SEED)
    triples = [draw.integers(0, points.shape[0], size=(_DRAWS, 3))]
    if own is not None:
        through_own = draw.integers(0, points.shape[0], size=(_DRAWS, 3))
        through_own[:, 0] = own
        triples.append(through_own)
    picked = np.vstack(triples)
    first = points[picked[:, 0]]
    across = np.cross(points[picked[:, 1]] - first, points[picked[:, 2]] - first)
    lengths = np.linalg.norm(across, axis=1)
    spans = lengths > _SPANS_NO_PLANE_MM2
    normals = across[spans] / lengths[spans, None]
    return normals, np.einsum("ij,ij->i", normals, first[spans])


def _fit(points: np.ndarray) -> "tuple[np.ndarray, float] | None":
    """The least-squares plane of ``points``, its unit normal and offset: the direction they spread least along.
    ``None`` where they lie along a line, which spans no plane."""
    if points.shape[0] < 3:
        return None
    centre = points.mean(axis=0)
    _, spreads, directions = np.linalg.svd(points - centre, full_matrices=False)
    if not spreads[1] > _ON_A_LINE * spreads[0]:
        return None
    normal = directions[-1]
    return normal, float(normal @ centre)


def _fingertips_past_the_tcp_mm(robot_config: Any) -> float:
    """How far the fingertips of the hand ``robot_config`` names reach past its TCP along the approach, millimetres:
    the jaw's ``finger_ahead_mm`` past its grasp centre and that centre past the TCP
    (``planning.hand.hand_past_the_tcp_mm``), as the calculator and the guard hold the hand. A section that names no
    hand is refused: no fingertips, no 0."""
    from src.robot.safety.planning.hand import hand_past_the_tcp_mm, planner_hand  # noqa: PLC0415

    hand = planner_hand(robot_config)
    if not chosen(hand):
        raise ValueError("fingertips_of names a robot section whose robot.gripper.model is unset, so no hand says "
                         "where its fingertips stand: name the hand, or leave fingertips_of out to stand the TCP there")
    return float(hand.jaw.finger_ahead_mm) + float(hand_past_the_tcp_mm(robot_config))
