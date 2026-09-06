"""The camera model of the rig, stated once and written into every scene.

Recording only where each camera is and what it looks at leaves the handedness to the reader, and a
look-at matrix whose X and Y axes are rolled 180 deg from the image convention projects every object
to ``(W - u, H - v)``: a result that is perfectly self-consistent, perfectly wrong, and
indistinguishable from a permuted set of masks.

So the convention is fixed here, once, and the resulting matrices are written into ``scene.json``.
A consumer never has to re-derive them, and a consumer that does can check itself against them.

The CV optical convention, the same one the rest of this repository's perception uses:

* ``+Z`` forward, into the scene, toward the target
* ``+X`` right in the image (``u`` grows with it)
* ``+Y`` down in the image (``v`` grows with it)

so ``u = fx*x/z + cx`` and ``v = fy*y/z + cy`` with no sign corrections anywhere.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "intrinsics_matrix", "look_at_camera_to_base", "project_to_pixels", "unproject_to_base",
]

#: Tried in order when the view direction is parallel to the requested up axis. A true overhead
#: camera is the normal case for this generator, not an edge case, so it needs a defined answer.
_UP_FALLBACKS = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0))


def look_at_camera_to_base(
    eye_mm: Sequence[float] | np.ndarray,
    look_at_mm: Sequence[float] | np.ndarray,
    up_hint: Sequence[float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """4x4 ``CAMERA(CV optical) -> BASE``: +Z at the target, +X image-right, +Y image-down.

    ``up_hint`` is which way is up in the world, not in the image; the image's up is derived from it.
    A camera pointing straight down has no answer from ``up_hint=+Z``, so the fallbacks give it a
    defined one rather than a division by ~0.
    """
    eye = np.asarray(eye_mm, dtype=np.float64).reshape(3)
    target = np.asarray(look_at_mm, dtype=np.float64).reshape(3)
    forward = target - eye
    norm = float(np.linalg.norm(forward))
    if norm < 1e-9:
        raise ValueError("look_at_camera_to_base: camera coincides with its target")
    forward /= norm

    for candidate in (tuple(up_hint), *_UP_FALLBACKS):
        world_up = np.asarray(candidate, dtype=np.float64).reshape(3)
        right = np.cross(forward, world_up)
        if float(np.linalg.norm(right)) > 1e-6:
            break
    else:  # pragma: no cover (three mutually non-parallel hints cannot all fail)
        raise ValueError("look_at_camera_to_base: no usable up axis")
    right /= np.linalg.norm(right)
    # Image-down is the negative of world-up-in-camera, which is what makes v = fy*y/z + cy correct.
    down = np.cross(forward, right)

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack([right, down, forward])
    matrix[:3, 3] = eye
    return matrix


def intrinsics_matrix(resolution: Sequence[int], horizontal_fov_deg: float) -> np.ndarray:
    """Pinhole ``K`` for a square-pixel camera with the given horizontal field of view.

    Square pixels are not an assumption: the renderer authors the lens aperture from this same
    horizontal FOV and refuses to continue if the lens reads back a different one.
    """
    width, height = int(resolution[0]), int(resolution[1])
    if not 0.0 < horizontal_fov_deg < 180.0:
        raise ValueError(f"horizontal_fov_deg must be in (0, 180), got {horizontal_fov_deg}")
    fx = (width / 2.0) / float(np.tan(np.radians(horizontal_fov_deg) / 2.0))
    return np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]])


def project_to_pixels(
    points_mm: np.ndarray, camera_to_base: np.ndarray, intrinsics: np.ndarray,
) -> np.ndarray:
    """Pixels (N x 2) for BASE-frame points (N x 3, mm). Points behind the camera become NaN.

    NaN rather than a mirrored pixel: a point behind the camera has no image position, and a
    plausible wrong number here would be scored as a detection somewhere downstream.
    """
    points = np.atleast_2d(np.asarray(points_mm, dtype=np.float64))
    base_to_camera = np.linalg.inv(np.asarray(camera_to_base, dtype=np.float64))
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    camera = (base_to_camera @ homogeneous.T).T[:, :3]
    k = np.asarray(intrinsics, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = k[0, 0] * camera[:, 0] / camera[:, 2] + k[0, 2]
        v = k[1, 1] * camera[:, 1] / camera[:, 2] + k[1, 2]
    pixels = np.stack([u, v], axis=1)
    pixels[camera[:, 2] <= 0.0] = np.nan
    return pixels


def unproject_to_base(
    depth_mm: np.ndarray, mask: np.ndarray, camera_to_base: np.ndarray, intrinsics: np.ndarray,
) -> np.ndarray:
    """The inverse of :func:`project_to_pixels`: BASE-frame points (N x 3, mm) from masked depth.

    Same convention, deliberately in the same file: a back-projection that disagreed with the forward
    one by a sign would produce a point cloud that is perfectly self-consistent and in the wrong
    place, which is the failure the camera convention was written down to prevent. Projecting and
    unprojecting a point returns the point.
    """
    depth = np.asarray(depth_mm, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool) & (depth > 0.0)
    rows, columns = np.nonzero(selected)
    if rows.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    z = depth[rows, columns]
    camera = np.stack([
        (columns - k[0, 2]) * z / k[0, 0],
        (rows - k[1, 2]) * z / k[1, 1],
        z,
    ], axis=1)
    transform = np.asarray(camera_to_base, dtype=np.float64)
    return (transform[:3, :3] @ camera.T).T + transform[:3, 3]
