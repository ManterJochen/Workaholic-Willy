"""Camera authoring for the Isaac cell, and the CAMERA->BASE / CAMERA->TOOL transforms.

Covers the overhead eye-to-hand camera in its top-down and tilted placements, the eye-in-hand wrist
camera, and the ground-truth fitters that derive each transform from Isaac's own projection. All
``isaacsim.*`` imports are lazy, inside the functions, so the pure-numpy helpers here are usable on a
machine with no Isaac Sim installed.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from src.geometry import Frame, Transform
from src.geometry.matrix import invert_homogeneous
from src.calibration.solver.umeyama_rigid import UmeyamaRigid

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.drivers.sim._isaac_protocols import IsaacCamera


# ===========================================================================
# Field of view: matching the sim optics to the real sensor.
# ===========================================================================
#
# An Isaac ``Camera`` is authored with USD's default lens (focal length and aperture), which is not
# the lens of the camera the cell ships with. That is a silent sim2real gap: a narrower sim lens sees
# a smaller slice of the table, so a scene comfortably in frame in sim can fall outside the real
# camera's image, or, more often, the real camera sees far more clutter than the detector was ever
# exercised on. Stating the sensor's FOV in config and authoring the lens from it closes the gap.
#
# Intel RealSense D435, from the vendor datasheet:
#   * RGB   stream: 69.4 deg x 42.5 deg (+/-3 deg), the stream the detector sees
#   * Depth stream: 87 deg x 58 deg
# The sim models one camera producing both RGB and depth, so it is authored with the RGB FOV: framing
# the detector correctly matters more than framing depth, and the depth the sim reports is dense and
# exact anyway. The real D435's wider, noisier depth is a separate sim2real gap, not this one.
D435_RGB_HFOV_DEG = 69.4
D435_RGB_VFOV_DEG = 42.5


def aperture_for_hfov(focal_length: float, hfov_deg: float) -> float:
    """Camera aperture giving ``hfov_deg`` at ``focal_length`` (``2*f*tan(hfov/2)``).

    Unit-agnostic by construction: aperture and focal length are expressed in the same units, so this
    holds whether the caller reads USD's raw ``focalLength``/``horizontalAperture`` (tenths of a world
    unit) or Isaac's ``Camera.get_focal_length()`` (which scales both by the same factor).
    """
    if focal_length <= 0.0:
        raise ValueError(f"focal_length must be > 0, got {focal_length!r}")
    if not 0.0 < hfov_deg < 180.0:
        raise ValueError(f"hfov_deg must be in (0, 180), got {hfov_deg!r}")
    return 2.0 * float(focal_length) * math.tan(math.radians(hfov_deg) / 2.0)


def vfov_deg_from_hfov(hfov_deg: float, resolution: tuple[int, int]) -> float:
    """Vertical FOV implied by ``hfov_deg`` at the render aspect, assuming square pixels.

    The sim renders 4:3 while a D435 quotes its 69.4 x 42.5 deg pair at 16:9, so the vertical FOV is
    derived from the authored horizontal one and the actual resolution rather than taken from the
    datasheet. Taking it from the datasheet would make the two disagree and stretch the image.
    """
    width, height = int(resolution[0]), int(resolution[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"resolution must be positive, got {resolution!r}")
    return math.degrees(2.0 * math.atan(math.tan(math.radians(hfov_deg) / 2.0) * height / width))


def hfov_deg_from_intrinsics(fx: float, width_px: int) -> float:
    """Horizontal FOV in degrees implied by a pinhole ``fx`` over ``width_px``.

    This is the measured counterpart to :func:`aperture_for_hfov`, used to verify that authoring the
    lens landed. The intrinsics matrix Isaac reports after ``initialize()`` is the ground truth for
    what the renderer does; the authored aperture values are not.
    """
    if fx <= 0.0 or width_px <= 0:
        raise ValueError(f"fx and width_px must be > 0, got {fx!r}, {width_px!r}")
    return math.degrees(2.0 * math.atan(float(width_px) / (2.0 * float(fx))))


def set_horizontal_fov(camera: object, hfov_deg: float, resolution: tuple[int, int]) -> float:
    """Author ``camera``'s lens so its horizontal FOV is ``hfov_deg``, and return the measured result.

    Keeps the focal length and solves for the aperture, then sets the vertical aperture to the same
    scale times the image aspect so pixels stay square (``fx == fy``). Isaac derives both intrinsics
    from the aperture pair, so changing only the horizontal one would silently stretch the image and
    corrupt every depth back-projection.

    The returned horizontal FOV is read back from the camera's own intrinsics, not computed from the
    values just written.
    """
    width, height = int(resolution[0]), int(resolution[1])
    focal = float(camera.get_focal_length())  # type: ignore[attr-defined]
    h_aperture = aperture_for_hfov(focal, hfov_deg)
    camera.set_horizontal_aperture(h_aperture)  # type: ignore[attr-defined]
    camera.set_vertical_aperture(h_aperture * height / width)  # type: ignore[attr-defined]
    k = np.asarray(camera.get_intrinsics_matrix(), dtype=np.float64)  # type: ignore[attr-defined]
    return hfov_deg_from_intrinsics(float(k[0, 0]), width)




# ===========================================================================
# Tilted eye-to-hand placement: where the real cell's cameras sit.
# ===========================================================================
#
# The default sim cell authors its eye-to-hand camera straight down over the workspace. The cell built
# for real hardware does not look like that: its two D435s are mounted off to the side and tilted,
# roughly 70 deg up from the table plane. That difference is not cosmetic:
#   * a tilted view sees object sides, so a silhouette is no longer the object's footprint,
#   * it has real parallax, so depth error projects into the grasp position differently,
#   * and it breaks the top-down CAMERA->BASE shortcut below, which assumes a nadir view.
# Expressing the mounting as (aim, elevation, azimuth, distance) keeps the rig auditable: the angle is
# a number in config, not an implicit consequence of three hand-picked coordinates.


def camera_position_for_elevation(
    aim_mm: tuple[float, float, float],
    *,
    elevation_deg: float,
    azimuth_deg: float,
    distance_mm: float,
) -> tuple[float, float, float]:
    """World position of a camera ``distance_mm`` from ``aim_mm``, at ``elevation_deg`` above the table.

    ``elevation_deg`` is measured from the table plane (90 = straight down/nadir, 0 = horizontal), which
    is how a rig is described on site ("about 70 degrees up from the table edge"). ``azimuth_deg`` is
    measured in the table plane from base +X toward +Y, so a pair of cameras on opposite sides is
    ``azimuth_deg=-90`` / ``+90``.
    """
    if distance_mm <= 0.0:
        raise ValueError(f"distance_mm must be > 0, got {distance_mm!r}")
    el, az = math.radians(elevation_deg), math.radians(azimuth_deg)
    horizontal = float(distance_mm) * math.cos(el)
    return (
        float(aim_mm[0]) + horizontal * math.cos(az),
        float(aim_mm[1]) + horizontal * math.sin(az),
        float(aim_mm[2]) + float(distance_mm) * math.sin(el),
    )


def camera_elevation_deg(
    position_mm: tuple[float, float, float], aim_mm: tuple[float, float, float]
) -> float:
    """Elevation in degrees above the table plane of the view from ``position_mm`` to ``aim_mm``.

    The inverse of :func:`camera_position_for_elevation`, so the tilt an authored rig actually has can
    be read back from its coordinates rather than taken on trust.
    """
    d = np.asarray(position_mm, dtype=np.float64) - np.asarray(aim_mm, dtype=np.float64)
    horizontal = float(math.hypot(d[0], d[1]))
    if horizontal < 1e-9:
        return 90.0  # exactly nadir
    return math.degrees(math.atan2(float(d[2]), horizontal))


def top_down_camera_to_base_matrix(
    cam_pos_mm: np.ndarray,
    *,
    base_pos_mm: np.ndarray | None = None,
) -> np.ndarray:
    """4x4 ``CAMERA(optical) -> BASE`` matrix in mm for an overhead camera looking straight down.

    The grasp calculator back-projects pixels into the CV optical frame (+X right, +Y down, +Z into
    the scene). For a downward camera the depth axis maps optical +Z to base -Z, and the plain
    top-down form of the other two is optical +X to base +X and +Y to base -Y, that is
    ``R = diag(1, -1, -1)``. The rotation returned here is that form yawed 90 degrees, for the reason
    the body comment gives. The translation is the camera position, since base and world coincide.
    The rotation must not be derived from the camera quaternion: Isaac's ``Camera`` sensor optical
    convention differs from a raw ``UsdGeom.Camera``, and this geometric form is exact for the
    authored top-down rig. In-plane image roll is irrelevant for a centred single object.
    """
    pos = np.asarray(cam_pos_mm, dtype=np.float64).reshape(3)
    base = np.zeros(3) if base_pos_mm is None else np.asarray(base_pos_mm, dtype=np.float64).reshape(3)
    mat = np.eye(4, dtype=np.float64)
    # Optical +Z maps to base -Z, so depth runs downward. The image axes map image-X to base +Y and
    # image-Y to base +X, a 90 degree yaw of the plain top-down diag(1,-1,-1). The yaw orients the
    # camera frame so that the grasp calculator's default antipodal axis, which falls along image Y
    # for a flat top-down silhouette, resolves to base +X: the UR5e can reach a top-down grasp that
    # closes along base X, while a base-Y closing hits a wrist singularity. The object centroid sits
    # on the optical axis, so the yaw leaves the grasp position exact and only fixes the closing
    # direction.
    mat[:3, :3] = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    mat[:3, 3] = pos - base
    return mat



# ===========================================================================
# Eye-in-hand wrist camera: opt-in, never added to the overhead-only path.
# ===========================================================================
#
# The wrist camera is parented to /World/UR5e/wrist_3_link so it moves rigidly with the
# arm. Unlike the overhead eye-to-hand rig, which has a static CAMERA->BASE, an eye-in-hand
# camera has a fixed CAMERA->TOOL transform composed with the live TCP pose every frame by
# EyeInHandFrameResolver. That transform is measured, by the ground-truth oracle here and by
# the AX=XB calibration in the calibration runner, rather than derived by hand.
#
# Mount geometry: at the straight-down viewing pose the local wrist_3_link +Y axis points
# along the gripper approach, which is base -Z. The camera is oriented so its optical -Z
# looks along wrist_3 +Y, down the approach at a viewing pose, and offset sideways to see
# past the gripper. The exact offset and orientation are the constants below.

WRIST_LINK_PRIM = "/World/UR5e/wrist_3_link"

WRIST_CAM_PRIM = "/World/UR5e/wrist_3_link/wrist_cam"

WRIST_CAM_RESOLUTION = (640, 480)

# Camera placement is in the wrist_3_link local frame:
#   wrist_3 +Y points along the gripper approach (world -Z, "down" at a downward viewpose),
#   wrist_3 +X is sideways, wrist_3 -Y is "up" (world +Z) at that viewpose.
# A camera looking straight down through the gripper axis sees only the gripper, so it is mounted
# off to the side and aimed straight down, parallel to the gripper and offset about 130 mm sideways:
#   * aiming back at the approach axis (aim X = 0) puts the gripper dead-centre, where it occludes
#     the whole view and ArUco does not detect;
#   * aim X equal to offset X makes the optical axis pure wrist +Y, parallel to the gripper, imaging
#     the workspace about 130 mm to the side of it. The gripper is out of frame, marker views are
#     clean, and during a pick the gripper does not occlude the object while it is perceived; the
#     arm only then moves in to grasp.
# Offset (X sideways, Y along the approach, Z) in mm:
WRIST_CAM_OFFSET_MM = (130.0, 20.0, 0.0)

# Aim point (wrist frame): aim X equals offset X, so the optical axis runs straight down,
# parallel to the gripper.
WRIST_CAM_AIM_TARGET_MM = (130.0, 457.0, 0.0)

# Image-up hint (wrist frame): wrist -Y is about world-up at a downward viewpose, so the image
# comes out upright.
WRIST_CAM_UP_HINT = (0.0, -1.0, 0.0)

# A wrist camera views the workspace from tens of centimetres, and the 1.0 m default near clip
# would clip that whole view to black, so a small near clip is set at mount time.
WRIST_CAM_NEAR_CLIP_M = 0.02


def _rotmat_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    """XYZW quaternion of a 3x3 rotation matrix, taking the numerically stable branch."""
    r = np.asarray(rot, dtype=np.float64)
    t = float(np.trace(r))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)



def look_at_quat_in_link_xyzw(
    cam_pos_mm: tuple[float, float, float],
    target_mm: tuple[float, float, float],
    up_hint: tuple[float, float, float],
) -> np.ndarray:
    """Local camera orientation (XYZW) so a camera at ``cam_pos_mm`` looks at ``target_mm``.

    All vectors are in the same (wrist-link) frame. A USD/Isaac camera looks down its local -Z,
    so the camera +Z points away from the target. Returns the quaternion of the rotation whose
    columns are the camera axes expressed in the link frame (i.e. the prim's local orientation).
    """
    cam = np.asarray(cam_pos_mm, dtype=np.float64).reshape(3)
    tgt = np.asarray(target_mm, dtype=np.float64).reshape(3)
    up = np.asarray(up_hint, dtype=np.float64).reshape(3)
    z = cam - tgt  # camera +Z away from target
    nz = np.linalg.norm(z)
    if nz < 1e-9:
        raise ValueError("look_at: camera position coincides with target")
    z = z / nz
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-9:  # up parallel to z, so pick any non-parallel axis
        x = np.cross(np.array([1.0, 0.0, 0.0]), z)
        if np.linalg.norm(x) < 1e-9:
            x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return _rotmat_to_quat_xyzw(np.column_stack([x, y, z]))



def mount_wrist_camera(
    session: object,
    *,
    prim_path: str = WRIST_CAM_PRIM,
    parent_prim: str = WRIST_LINK_PRIM,
    offset_mm: tuple[float, float, float] = WRIST_CAM_OFFSET_MM,
    aim_target_mm: tuple[float, float, float] = WRIST_CAM_AIM_TARGET_MM,
    up_hint: tuple[float, float, float] = WRIST_CAM_UP_HINT,
    quat_in_link_xyzw: tuple[float, float, float, float] | np.ndarray | None = None,
    resolution: tuple[int, int] = WRIST_CAM_RESOLUTION,
    near_clip_m: float = WRIST_CAM_NEAR_CLIP_M,
    warmup_steps: int = 5,
) -> IsaacCamera:
    """Author an eye-in-hand camera as a child of ``wrist_3_link`` so it follows the arm.

    The camera's transform is set as a local transform relative to the wrist link, a translate and
    an orient xform op, so the wrist carries it rigidly. By default the orientation is a look-at:
    the camera sits at ``offset_mm`` in the wrist frame and aims at ``aim_target_mm`` in the same
    frame, so it views the workspace beside the gripper. Passing an explicit ``quat_in_link_xyzw``
    overrides the look-at. Returns the live ``isaacsim.sensors.camera.Camera`` handle. The
    ``isaacsim.*`` imports are lazy, so this module imports without Isaac Sim installed. Call after
    the robot is referenced and ``world.reset()`` has run, that is after
    :func:`build_combined_scene`.
    """
    import omni.usd  # type: ignore[import-not-found]
    from isaacsim.sensors.camera import Camera  # type: ignore[import-not-found]
    from pxr import Gf, UsdGeom  # type: ignore[import-not-found]

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(parent_prim).IsValid():
        raise RuntimeError(f"mount_wrist_camera: parent prim not found: {parent_prim}")

    camera = Camera(prim_path=prim_path, resolution=resolution)

    # Author the local transform relative to wrist_3_link directly, so the camera follows the wrist.
    # The Camera ctor has already authored xformOp:translate (vec3d) and xformOp:orient (quatd), and
    # re-adding an op with a different precision raises, because AddOrientOp defaults to float while
    # the existing attribute is quatd. Clearing the op order and removing the existing op attributes
    # before adding fresh ones is precision-agnostic and robust to whatever the ctor authored.
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    for _attr in (
        "xformOp:translate", "xformOp:orient", "xformOp:scale",
        "xformOp:rotateXYZ", "xformOp:transform",
    ):
        if prim.HasAttribute(_attr):
            prim.RemoveProperty(_attr)
    if quat_in_link_xyzw is None:
        quat_in_link_xyzw = look_at_quat_in_link_xyzw(offset_mm, aim_target_mm, up_hint)
    ox, oy, oz = (float(v) / 1000.0 for v in offset_mm)
    xform.AddTranslateOp().Set(Gf.Vec3d(ox, oy, oz))
    qx, qy, qz, qw = (float(v) for v in quat_in_link_xyzw)
    xform.AddOrientOp().Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))

    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()
    try:
        camera.set_clipping_range(near_clip_m, 1.0e6)
    except Exception:  # noqa: BLE001 (clipping API is best-effort)
        pass

    app = getattr(session, "app", None)
    for _ in range(max(0, warmup_steps)):
        session.step(render=True)  # type: ignore[attr-defined]
        if app is not None:
            app.update()
    return camera



def camera_to_base_ground_truth(camera: object, *, max_points: int = 400) -> tuple[Transform, float]:
    """Empirical ``CAMERA(CV optical) -> BASE`` for an Isaac camera, fit from Isaac's projection.

    The Isaac ``Camera`` sensor's optical convention differs from a raw USD camera, so a rotation
    derived from the camera quaternion is wrong. Isaac instead projects a set of valid-depth pixels
    to world through ``get_world_points_from_image_coords``; the same pixels are back-projected into
    the CV optical frame with the intrinsics, and :class:`UmeyamaRigid` solves the rigid
    ``CV -> world`` transform. World and base coincide, since the UR base sits at the world origin.
    The fit is convention-agnostic and lands in exactly the frame the grasp calculator back-projects
    into.

    Requires the camera to be seeing geometry, that is to have valid depth. Returns
    ``(Transform, rmse_mm)``.
    """
    depth_m = np.asarray(camera.get_depth(), dtype=np.float64)  # type: ignore[attr-defined]
    k = np.asarray(camera.get_intrinsics_matrix(), dtype=np.float64)  # type: ignore[attr-defined]
    valid = np.isfinite(depth_m) & (depth_m > 1e-4)
    # The ndim check comes first, and it is not defensive padding: it guards the failure the
    # diagnostic below is written for and could not otherwise reach. An unticked annotator returns an
    # empty buffer, which ``np.asarray`` makes 1-D, so ``np.where`` yields a 1-tuple and the unpack
    # raises "not enough values to unpack (expected 2, got 1)": a message about tuple arity, three
    # frames away from the camera, for the one failure this function is best placed to explain.
    if depth_m.ndim == 2:
        ys, xs = np.where(valid)
    else:
        ys = xs = np.zeros(0, dtype=np.int64)
    if ys.size < 3:
        # Say what the camera returned, not only that it was unusable. On its own "camera sees
        # nothing" cannot distinguish an empty buffer, where the annotator is not attached or not
        # ticked and the size is 0, from a buffer full of zeros or NaNs, where the camera is attached
        # but nothing is in frame. Those two need opposite fixes.
        _shape = getattr(depth_m, "shape", None)
        _n = int(depth_m.size)
        _finite = int(np.isfinite(depth_m).sum()) if _n else 0
        _pos = int((np.isfinite(depth_m) & (depth_m > 1e-4)).sum()) if _n else 0
        _rng = (
            f"min={np.nanmin(depth_m):.4f} max={np.nanmax(depth_m):.4f}"
            if _finite else "no finite values"
        )
        raise RuntimeError(
            f"camera_to_base_ground_truth: <3 valid depth pixels. depth shape={_shape} "
            f"ndim={depth_m.ndim} size={_n} finite={_finite} positive={_pos} ({_rng}). "
            f"ndim!=2 or size=0 means the depth annotator produced no frame (attach/tick problem; "
            f"give the camera more warmup steps before reading it); size>0 with positive=0 means the "
            f"camera is attached but nothing is in front of it (aim/clip/geometry problem)."
        )
    if ys.size > max_points:  # even spatial subsample
        idx = np.linspace(0, ys.size - 1, max_points).astype(int)
        ys, xs = ys[idx], xs[idx]
    d = depth_m[ys, xs]
    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    cam_cv_mm = np.stack(
        [(xs - cx) / fx * d, (ys - cy) / fy * d, d], axis=1
    ).astype(np.float64) * 1000.0
    pixels = np.stack([xs, ys], axis=1).astype(np.float64)  # (u=col, v=row)
    world_m = np.asarray(
        camera.get_world_points_from_image_coords(pixels, d), dtype=np.float64  # type: ignore[attr-defined]
    )
    world_mm = world_m[:, :3] * 1000.0
    rot, trans, rmse = UmeyamaRigid().solve(cam_cv_mm, world_mm)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = trans
    return Transform.from_matrix(mat, from_frame=Frame.CAMERA, to_frame=Frame.BASE), float(rmse)



def camera_to_tcp_ground_truth(camera: object, arm: object) -> tuple[Transform, float]:
    """Empirical ``CAMERA -> TOOL`` for an eye-in-hand wrist camera, TOOL being the grasp-centre TCP.

    Computed as ``inv(T_tool_to_base) @ T_cam_to_base``, where ``T_cam_to_base`` comes from
    :func:`camera_to_base_ground_truth` and ``T_tool_to_base`` is the live TCP pose from
    ``arm.get_tcp_pose()``, the grasp-centre frame EyeInHandFrameResolver expects. This is the oracle
    an EyeInHandCalibrator result is checked against. The camera is rigidly mounted to the wrist, so
    the transform is invariant to the arm pose, which evaluating at several poses confirms. Returns
    ``(Transform, fit_rmse_mm)``.
    """
    t_cam_to_base, rmse = camera_to_base_ground_truth(camera)
    t_tool_to_base = np.asarray(arm.get_tcp_pose().to_matrix(), dtype=np.float64)  # type: ignore[attr-defined]
    t_cam_to_tool = invert_homogeneous(t_tool_to_base) @ np.asarray(t_cam_to_base.to_matrix())
    return Transform.from_matrix(t_cam_to_tool, from_frame=Frame.CAMERA, to_frame=Frame.TOOL), rmse


def author_fixed_camera(
    session: object,
    prim_path: str,
    *,
    position_mm: tuple[float, float, float],
    aim_mm: tuple[float, float, float],
    up_hint: tuple[float, float, float] = (0.0, 0.0, 1.0),
    resolution: tuple[int, int] = (640, 480),
    near_clip_m: float = 0.05,
    hfov_deg: float | None = None,
    warmup_steps: int = 12,
) -> "IsaacCamera":
    """Author a fixed world-frame eye-to-hand camera looking at ``aim_mm``, such as the bin centre.

    Built the same way as the wrist camera in :func:`mount_wrist_camera`: create the Camera, then set
    the world xform translate and orient from the geometric look-at of
    :func:`look_at_quat_in_link_xyzw`, in world coordinates here. That avoids the Camera-ctor
    ``orientation`` convention, which a look-at quaternion does not match. A small near clip is
    mandatory: the 1.0 m default clips a closer side view to black. Used by the multi-view and the
    two-side-camera rigs. The ``isaacsim``, ``pxr`` and ``omni`` imports are lazy, so this module
    imports without Isaac Sim installed.
    """
    import omni.usd  # type: ignore[import-not-found]
    from isaacsim.sensors.camera import Camera  # type: ignore[import-not-found]
    from pxr import Gf, UsdGeom  # type: ignore[import-not-found]

    cam = Camera(prim_path=prim_path, resolution=resolution)
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    for _attr in ("xformOp:translate", "xformOp:orient", "xformOp:scale",
                  "xformOp:rotateXYZ", "xformOp:transform"):
        if prim.HasAttribute(_attr):
            prim.RemoveProperty(_attr)
    q_xyzw = look_at_quat_in_link_xyzw(position_mm, aim_mm, up_hint)
    px, py, pz = (float(v) / 1000.0 for v in position_mm)
    xform.AddTranslateOp().Set(Gf.Vec3d(px, py, pz))
    qx, qy, qz, qw = (float(v) for v in q_xyzw)
    xform.AddOrientOp().Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))

    cam.initialize()
    cam.add_distance_to_image_plane_to_frame()
    try:
        cam.set_clipping_range(near_clip_m, 1.0e6)
    except Exception:  # noqa: BLE001 (clipping API is best-effort)
        pass
    # Author the lens from the real sensor's FOV when the caller states one; None keeps the default.
    if hfov_deg is not None:
        set_horizontal_fov(cam, hfov_deg, resolution)
    app = getattr(session, "app", None)
    for _ in range(max(0, warmup_steps)):
        session.step(render=True)  # type: ignore[attr-defined]
        if app is not None:
            app.update()
    return cam
