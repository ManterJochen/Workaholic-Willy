"""A synthetic stereo pair with known geometry, and the rigs that read it, for the stereo-as-RGB-D tests.

Two cameras of one known lens look at a textured plane with a box standing on it and an ArUco marker printed on it.
The right camera stands ``BASELINE_MM`` to the left camera's right and is turned a few degrees, so the pair is not
rectified as rendered and a reader that skips the rectification, or poses the marker with the raw lens, is visibly
wrong. Every image is ray cast at twice the size and shrunk, so the marker's edges are as crisp as a camera's.

The left camera's frame is the scene's: the plane, the box and the marker are placed in it, and ``true_depth`` gives
the depth of every pixel of the left image rectified, along the rectified left camera's axis, the frame the stereo
rig's depth lies in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from src.calibration.stereo.config import CalibrationResult
from src.calibration.stereo.sub_modules.calib_store import StereoCalibrationStore
from src.camera.setup.image_taking.frames import StereoFrame
from src.config.schema.camera import SingleDeviceRigConfig, WebcamPairRigConfig

SIZE = (640, 480)
K = np.array([[600.0, 0.0, 316.0], [0.0, 600.0, 243.0], [0.0, 0.0, 1.0]])
BASELINE_MM = 60.0
#: The right camera's turn, degrees about its own y and x axes: what the rectification has to undo.
TURN_DEG = (3.0, 0.8)
MARKER_ID = 7
MARKER_MM = 60.0
DICTIONARY = "DICT_5X5_100"

#: The plane: a point of it and its normal, toward the camera; the box's footprint centre and size on it.
_PLANE_POINT = np.array([0.0, 20.0, 720.0])
_PLANE_NORMAL = np.array([0.10, -0.18, -1.0]) / np.linalg.norm([0.10, -0.18, -1.0])
_BOX_CENTRE_UV = (-95.0, -10.0)
_BOX_HALF_MM = (45.0, 40.0)
BOX_HEIGHT_MM = 55.0
_MARKER_UV = (35.0, 15.0)  # the marker's top-left corner on the plane, plane millimetres


def _rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    about_y = np.array([[math.cos(y), 0.0, math.sin(y)], [0.0, 1.0, 0.0], [-math.sin(y), 0.0, math.cos(y)]])
    about_x = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(p), -math.sin(p)], [0.0, math.sin(p), math.cos(p)]])
    return about_y @ about_x


#: The right camera's axes in the left camera's frame, and its centre there.
RIGHT_AXES = _rotation(*TURN_DEG)
RIGHT_CENTRE = np.array([BASELINE_MM, 0.0, 0.0])


def _plane_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The plane's own axes: ``e1`` and ``e2`` along it, ``n`` its normal toward the camera. ``e1 x e2`` points into
    the scene, as the image's x by its y does, so the printed marker is seen turned, never mirrored."""
    n = _PLANE_NORMAL
    e1 = np.cross(np.array([0.0, 1.0, 0.0]), n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, n)
    return e1, e2, n


def _texture(size: int, seed: int, sigma: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.uniform(0.0, 255.0, size=(size, size)).astype(np.float32)
    texture = cv.GaussianBlur(noise, (0, 0), sigma)
    texture = (texture - texture.mean()) / (texture.std() + 1e-6) * 45.0 + 128.0
    return np.clip(texture, 20.0, 235.0).astype(np.float32)


_TEXEL_MM = 0.5
_PLANE_TEXTURE = _texture(2400, 11, 2.0)  # 1200 mm across, centred on the plane point: wider than either view
_BOX_TEXTURE = _texture(400, 12, 2.0)
_MARKER = cv.aruco.generateImageMarker(cv.aruco.getPredefinedDictionary(getattr(cv.aruco, DICTIONARY)), MARKER_ID, 210)


def _sample(texture: np.ndarray, u_mm: np.ndarray, v_mm: np.ndarray) -> np.ndarray:
    """The texture at plane millimetres ``u_mm``, ``v_mm``, bilinear, centred on the texture's middle."""
    side = texture.shape[0]
    x = np.clip((np.asarray(u_mm) + side * _TEXEL_MM / 2.0) / _TEXEL_MM, 0.0, side - 1.001)
    y = np.clip((np.asarray(v_mm) + side * _TEXEL_MM / 2.0) / _TEXEL_MM, 0.0, side - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    top = texture[y0, x0] * (1 - fx) + texture[y0, x0 + 1] * fx
    bottom = texture[y0 + 1, x0] * (1 - fx) + texture[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def _hits(origin: np.ndarray, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each ray, the distance to the first surface and the grey level there."""
    e1, e2, n = _plane_axes()
    flat = rays.reshape(-1, 3)
    # The plane.
    den = flat @ n
    t_plane = ((_PLANE_POINT - origin) @ n) / den
    points = origin + flat * t_plane[:, None]
    u = (points - _PLANE_POINT) @ e1
    v = (points - _PLANE_POINT) @ e2
    grey = _sample(_PLANE_TEXTURE, u, v)
    # The marker, printed on the plane with a white quiet zone of 12 mm.
    mu, mv = _MARKER_UV
    in_zone = (u > mu - 12) & (u < mu + MARKER_MM + 12) & (v > mv - 12) & (v < mv + MARKER_MM + 12)
    grey[in_zone] = 240.0
    in_marker = (u >= mu) & (u < mu + MARKER_MM) & (v >= mv) & (v < mv + MARKER_MM)
    cols = np.clip(((u[in_marker] - mu) / MARKER_MM * _MARKER.shape[1]).astype(int), 0, _MARKER.shape[1] - 1)
    rows = np.clip(((v[in_marker] - mv) / MARKER_MM * _MARKER.shape[0]).astype(int), 0, _MARKER.shape[0] - 1)
    grey[in_marker] = np.where(_MARKER[rows, cols] > 127, 240.0, 15.0)
    # The box: a slab test in its own frame, x along e1, y along e2 and z up off the plane, toward the camera.
    centre = _PLANE_POINT + _BOX_CENTRE_UV[0] * e1 + _BOX_CENTRE_UV[1] * e2
    axes = np.stack([e1, e2, n])
    local_origin = axes @ (origin - centre)
    local_rays = flat @ axes.T
    low = np.array([-_BOX_HALF_MM[0], -_BOX_HALF_MM[1], 0.0])
    high = np.array([_BOX_HALF_MM[0], _BOX_HALF_MM[1], BOX_HEIGHT_MM])
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (low - local_origin) / local_rays
        t2 = (high - local_origin) / local_rays
    t_near = np.nanmax(np.minimum(t1, t2), axis=1)
    t_far = np.nanmin(np.maximum(t1, t2), axis=1)
    box = (t_near <= t_far) & (t_near > 0) & (t_near < t_plane)
    inside = local_origin + local_rays[box] * t_near[box, None]
    # Each face is textured by the two coordinates along it.
    on_top = np.abs(inside[:, 2] - BOX_HEIGHT_MM) < 1e-6
    a = np.where(on_top, inside[:, 0], np.where(np.abs(np.abs(inside[:, 0]) - _BOX_HALF_MM[0]) < 1e-6,
                                                 inside[:, 1], inside[:, 0]))
    b = np.where(on_top, inside[:, 1], inside[:, 2] + 100.0)
    grey[box] = _sample(_BOX_TEXTURE, a, b) * np.where(on_top, 1.0, 0.8)
    distance = np.where(box, t_near, t_plane)
    return distance.reshape(rays.shape[:2]), grey.reshape(rays.shape[:2])


def _rays(lens: np.ndarray, axes: np.ndarray, scale: int) -> np.ndarray:
    """Every pixel's ray of a camera of ``lens``, its axes ``axes`` in the scene, rendered ``scale`` times larger."""
    width, height = SIZE[0] * scale, SIZE[1] * scale
    us, vs = np.meshgrid((np.arange(width) + 0.5) / scale - 0.5, (np.arange(height) + 0.5) / scale - 0.5)
    pixels = np.stack([us, vs, np.ones_like(us)], -1)
    return (pixels @ np.linalg.inv(lens).T) @ axes.T


def render() -> tuple[np.ndarray, np.ndarray]:
    """The pair as the two cameras see it, grey ``uint8`` at :data:`SIZE`, rendered at twice the size and shrunk."""
    images = []
    for centre, axes in ((np.zeros(3), np.eye(3)), (RIGHT_CENTRE, RIGHT_AXES)):
        _, grey = _hits(centre, _rays(K, axes, 2))
        images.append(cv.resize(grey, SIZE, interpolation=cv.INTER_AREA).astype(np.uint8))
    return images[0], images[1]


def calibration() -> CalibrationResult:
    """The pair's stereo calibration, rectified by ``cv.stereoRectify`` from the cameras as they were placed."""
    rotation = RIGHT_AXES.T  # a point of the left camera's frame into the right one's
    translation = -RIGHT_AXES.T @ RIGHT_CENTRE
    zero = np.zeros(5)
    rect_l, rect_r, proj_l, proj_r, q, _, _ = cv.stereoRectify(K, zero, K, zero, SIZE, rotation, translation,
                                                               alpha=0.0)
    return CalibrationResult(camL=K, distL=zero, rectL=rect_l, projL=proj_l, camR=K, distR=zero, rectR=rect_r,
                             projR=proj_r, Q=q, frame_size=SIZE)


def true_depth(result: CalibrationResult) -> np.ndarray:
    """The depth of every pixel of the left image rectified, millimetres along the rectified left camera's axis."""
    rect = np.asarray(result.rectL)
    rays = _rays(np.asarray(result.K_rect), rect.T, 1)  # the rectified camera's rays, in the left camera's frame
    distance, _ = _hits(np.zeros(3), rays)
    points = rays * distance[..., None]
    return points @ rect.T[:, 2]


def true_marker(result: CalibrationResult) -> tuple[np.ndarray, np.ndarray]:
    """The marker's centre and its z axis, out of the print toward the camera, in the rectified left camera's frame."""
    e1, e2, n = _plane_axes()
    centre = _PLANE_POINT + (_MARKER_UV[0] + MARKER_MM / 2) * e1 + (_MARKER_UV[1] + MARKER_MM / 2) * e2
    rect = np.asarray(result.rectL)
    return rect @ centre, rect @ n


@dataclass
class PairDouble:
    """A pair device standing in for two webcams: it hands out the same rendered pair at every grab."""

    left: np.ndarray
    right: np.ndarray
    opened: bool = False
    grabs: int = 0

    def open(self) -> None:
        self.opened = True

    def release(self) -> None:
        self.opened = False

    def is_opened(self) -> bool:
        return self.opened

    def grab(self) -> StereoFrame:
        self.grabs += 1
        return StereoFrame(left=self.left, right=self.right)


def webcam_rig(base_dir: Path, *, rig_id: str = "pair", size: tuple[int, int] = SIZE, **extra: Any) -> WebcamPairRigConfig:
    """A webcam pair rig whose calibration files live under ``base_dir``."""
    return WebcamPairRigConfig.model_validate({
        "rig_id": rig_id, "enabled": True, "source": "webcam_pair", "frame_size": list(size), "fps": 30,
        "backend": 0, "max_cam_scan": 2, "cam_left_id": 4, "cam_right_id": 5,
        "calibration_paths": {"base_dir": str(base_dir)}, **extra})


def single_rig(base_dir: Path, *, rig_id: str = "device") -> SingleDeviceRigConfig:
    """A side by side stereo device rig whose calibration files live under ``base_dir``."""
    return SingleDeviceRigConfig.model_validate({
        "rig_id": rig_id, "enabled": True, "source": "single_device", "device_index": 6, "fps": 30, "backend": 0,
        "device_frame_size": [SIZE[0] * 2, SIZE[1]], "per_eye_frame_size": list(SIZE), "layout": "horizontal",
        "calibration_paths": {"base_dir": str(base_dir)}})


def write_calibration(rig: Any, result: CalibrationResult) -> Path:
    """The pair's calibration written where the rig reads it, ``calibration_paths.stereo_map_file``."""
    path = Path(str(rig.calibration_paths.stereo_map_file))
    StereoCalibrationStore().save(str(path), result)
    return path
