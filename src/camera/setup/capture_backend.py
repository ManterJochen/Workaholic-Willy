"""The OpenCV capture backend a rig names, against the platform that opens it (P0h h7, owner decision 5a).

``camera.cameras.rigs[*].backend`` is OpenCV's capture API, the integer ``cv2.VideoCapture`` takes as its second
argument. 700 (DirectShow) and 1400 (Media Foundation) exist only in OpenCV's Windows builds: on Linux
``VideoCapture(i, 700)`` opens nothing, and an operator read "No enabled camera rigs are currently available" with no
word about the backend. The owner chose, on 2026-09-26, a refusal that names the key and what Linux opens a camera
with, over translating the value in silence, so a Windows tree stays as it is and still loads on Linux: the refusal
sits where a camera is built (``camera.orchestration.camera.select_rig`` for the cell, and every OpenCV camera's
constructor for the tools). A RealSense rig opens through its SDK and never reads the key, so it is never refused
for it.
"""

from __future__ import annotations

from typing import Any

from src.utility.interpreters import is_windows

__all__ = ["WINDOWS_ONLY", "CaptureBackendRefused", "capture_backend_refusal", "refuse_a_backend_this_platform_lacks"]

#: The capture backends only OpenCV's Windows builds have, with the names OpenCV gives them.
WINDOWS_ONLY: dict[int, str] = {700: "CAP_DSHOW (DirectShow)", 1400: "CAP_MSMF (Media Foundation)"}
#: What opens a USB camera on Linux instead, as a person writes it into the key.
_LINUX_VALUES = "0 (CAP_ANY, OpenCV chooses) or 200 (CAP_V4L2)"


class CaptureBackendRefused(ValueError):
    """A rig names a capture backend this platform's OpenCV does not have."""


def capture_backend_refusal(rig: Any, *, platform: str | None = None) -> str | None:
    """Why ``platform`` (this one when not given) cannot open ``rig``'s capture backend, or None when it can.

    Only a rig that opens through ``cv2.VideoCapture`` reads the key: a webcam pair, a single stereo device and an
    RGB-D rig on the ``opencv`` backend.
    """
    if getattr(rig, "source", None) == "rgbd" and getattr(rig, "rgbd_backend", "opencv") != "opencv":
        return None
    backend = int(getattr(rig, "backend", 0))
    if backend not in WINDOWS_ONLY or is_windows(platform):
        return None
    return (f"camera.cameras.rigs[{getattr(rig, 'rig_id', '?')!r}].backend is {backend}, {WINDOWS_ONLY[backend]}, "
            f"which OpenCV has on Windows only, so this rig opens no camera here. On Linux set it to {_LINUX_VALUES}.")


def refuse_a_backend_this_platform_lacks(rig: Any) -> None:
    """Raise :class:`CaptureBackendRefused` when this platform cannot open ``rig``'s capture backend."""
    refusal = capture_backend_refusal(rig)
    if refusal is not None:
        raise CaptureBackendRefused(refusal)
