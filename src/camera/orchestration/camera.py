"""One owner per camera rig: the only opener of its device, grabs serialised, every frame stamped.

`Camera` owns one rig's device. Every consumer (the planner world, the pick perception,
calibration, the console viewfinder) takes a handle from that owner rather than opening the device
itself. A second owner of an open device in this process is refused with `CameraBusy` naming the
holder, every grab through any handle of the owner runs under the rig's lock, and a frame the owner
hands out carries the host time read just before the device grab (`captured_at_s`).

Identity is the device, not the rig name:

* a RealSense rig is its serial. A RealSense with no serial is one shared key that collides with
  every open RealSense, because the SDK then binds whichever camera it offers first;
* an OpenCV RGB-D rig and a single-device stereo rig are their `device_index`;
* a webcam pair is both of its ids, and an unset id collides with every open video device.

The registry holds live owners only. An owner that no longer exists holds no claim, so a build that
raised past its release and was dropped does not leave the next build meeting `CameraBusy` for an
object nobody can reach. Releasing is still what gives the device itself back.

A frame to look at is not a frame to measure with. `Camera.peek` hands a camera window
(``src/camera/live_view.py``) a colour image through the device's display path, which leaves every
measuring grab as it would have been without the window: see its docstring for why and how.

`FrameProvider` is the multi-rig catalogue for stereo capture and hand detection, and opens every rig
through an owner of this class, so a catalogue and a `Camera` never both hold one device.

This module imports nothing from `src.robot`: the camera package sits below the robot.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
import weakref
from time import monotonic as _monotonic
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from src.config.schema.camera import (
    RGBDDeviceRigConfig,
    SingleDeviceRigConfig,
    WebcamPairRigConfig,
)
from src.calibration.rig_calibration import RigCalibration
from src.camera.setup.image_taking.rgbd import OpenCvRGBDStreamer, RealSenseRGBDStreamer
from src.camera.setup.image_taking.single import SingleDeviceStreamer
from src.camera.setup.image_taking.webcam import WebcamPairStreamer
from src.contracts import UNSET, Maybe, chosen

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from types import TracebackType

    from src.camera.orchestration.frame_provider import RigHandle
    from src.camera.setup.image_taking.frames import AnyFrame

CameraRigConfig = WebcamPairRigConfig | SingleDeviceRigConfig | RGBDDeviceRigConfig
AnyStreamer = WebcamPairStreamer | SingleDeviceStreamer | OpenCvRGBDStreamer | RealSenseRGBDStreamer
#: A device as the registry knows it: its kind and its id. None is no id, which matches any device
#: of the kind.
DeviceKey = tuple[str, "int | str | None"]
RefusalReason = Literal["unknown", "disabled", "not_rgbd"]

__all__ = [
    "PEEK_QUIET_S",
    "AnyStreamer",
    "Camera",
    "CameraBusy",
    "CameraNotOpen",
    "CameraRefused",
    "CameraRigConfig",
    "PeekFrame",
    "create_streamer",
    "device_keys",
    "select_rig",
]

logger = logging.getLogger(__name__)

#: How long after a measuring grab, and after a notice that the camera moved, the owner hands out no frame to look at
#: (:meth:`Camera.peek`), in seconds; three frame periods where the rig streams slower than ten a second. Longer than
#: the pause the RealSense driver still counts as one burst of back-to-back grabs (``_PAUSE_S``, 0.25 s), so a look
#: never lands inside a burst: between a planning world's notice, its warm-ups and the grab it keeps, or between a pick
#: frame's warm-ups and its grab.
PEEK_QUIET_S = 0.3

_REGISTRY_LOCK = threading.Lock()
#: The open owners, by the device keys they hold. Weak, so an owner that no longer exists holds no
#: claim.
_HOLDERS: weakref.WeakValueDictionary[DeviceKey, Camera] = weakref.WeakValueDictionary()


class CameraBusy(RuntimeError):
    """A second owner asked for a device that another live owner in this process holds."""

    def __init__(self, rig_id: str, holder: str, device: DeviceKey) -> None:
        self.rig_id = rig_id
        self.holder = holder
        self.device = device
        super().__init__(
            f"rig {rig_id!r} cannot open its camera: rig {holder!r} holds the same device "
            f"({_describe(device)}) in this process. Release {holder!r} first, or give the two rigs "
            "distinct device ids."
        )


class CameraRefused(ValueError):
    """The camera section cannot give this rig: it is not configured, it is switched off, or it has
    no depth."""

    def __init__(self, message: str, *, reason: RefusalReason) -> None:
        super().__init__(message)
        self.reason: RefusalReason = reason


class CameraNotOpen(RuntimeError):
    """A grab or a lens read on an owner whose device is not open."""


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class PeekFrame:
    """A colour image taken for a person to look at (:meth:`Camera.peek`), never one to measure with.

    ``color`` is BGR, a copy the device no longer holds: the left eye of a stereo rig, and no depth at all.
    ``captured_at_s`` is the host time read just before the device was asked for it, as a grab's stamp is.
    """

    color: np.ndarray
    captured_at_s: float


def device_keys(rig: CameraRigConfig) -> tuple[DeviceKey, ...]:
    """The devices a rig occupies, as the registry compares them."""
    if isinstance(rig, RGBDDeviceRigConfig):
        if rig.rgbd_backend == "realsense":
            return (("realsense", rig.serial_number or None),)
        return (("video", rig.device_index),)
    if isinstance(rig, SingleDeviceRigConfig):
        return (("video", rig.device_index),)
    if isinstance(rig, WebcamPairRigConfig):
        return (("video", rig.cam_left_id), ("video", rig.cam_right_id))
    raise ValueError(f"Unsupported rig type: {type(rig)!r}")


def create_streamer(rig: CameraRigConfig) -> AnyStreamer:
    """The device streamer a rig names. Constructing one touches no device: every streamer
    ``__init__`` under ``camera.setup.image_taking`` only stores config."""
    if isinstance(rig, WebcamPairRigConfig):
        return WebcamPairStreamer(rig)
    if isinstance(rig, SingleDeviceRigConfig):
        return SingleDeviceStreamer(rig)
    if isinstance(rig, RGBDDeviceRigConfig):
        if rig.rgbd_backend == "realsense":
            return RealSenseRGBDStreamer(rig)
        return OpenCvRGBDStreamer(rig)
    raise ValueError(f"Unsupported rig type: {type(rig)!r}")


def select_rig(camera_cfg: Any, *, rig_id: Maybe[str] = UNSET, open_disabled: Maybe[bool] = UNSET) -> Any:
    """The rig a camera section gives for ``rig_id``, or `CameraRefused` saying why it gives none.

    ``rig_id`` UNSET means the rig the cell runs on, ``camera.cameras.primary_rig_id``. Three
    refusals, in this order: a rig the section does not configure, a rig switched off (lifted only
    by ``open_disabled``, which the bench exerciser passes), and a rig with no depth channel. The
    last is the stereo refusal: the planner world and the pick path take depth from RGB-D rigs only.

    The cell, the calibration CLI and the exerciser all select through this function, so the three
    apply one policy. A refusal for a rig with no depth lists the RGB-D rigs the section has,
    because the reader of that refusal is the one who does not know which they are.
    """
    cameras = getattr(camera_cfg, "cameras", None)
    rigs = list(getattr(cameras, "rigs", []) or [])
    by_key = not chosen(rig_id)
    wanted = getattr(cameras, "primary_rig_id", None) if by_key else rig_id
    rig = next((r for r in rigs if getattr(r, "rig_id", None) == wanted), None)
    if rig is None:
        known = sorted(r.rig_id for r in rigs)
        raise CameraRefused(
            f"camera.cameras.primary_rig_id names {wanted!r}, which is not in camera.cameras.rigs ({known})."
            if by_key else f"no rig {wanted!r} in camera.cameras.rigs; configured: {known}.",
            reason="unknown",
        )
    if not getattr(rig, "enabled", True) and not (chosen(open_disabled) and open_disabled):
        subject = (f"camera.cameras.primary_rig_id names {wanted!r} and that rig has" if by_key
                   else f"rig {wanted!r} has")
        raise CameraRefused(
            f"{subject} `enabled: false`. A cell cannot run on a camera its own config declares off. "
            "Switch it on, or name a rig that is on.",
            reason="disabled",
        )
    source = getattr(rig, "source", None)
    if source != "rgbd":
        depth_rigs = sorted(r.rig_id for r in rigs if getattr(r, "source", None) == "rgbd")
        subject = (f"camera.cameras.primary_rig_id names {wanted!r}, a {source!r} rig" if by_key
                   else f"rig {wanted!r} is a {source!r} rig")
        why = ("A grasp cell needs an RGB-D primary: the grasp is synthesised from its depth." if by_key
               else "Every camera this owner opens gives depth, and stereo rigs wait for their own step.")
        tail = (f"The RGB-D rigs here are {depth_rigs}. Name one of those instead, and prove it standalone "
                "with `python -m src.robot.perception`." if depth_rigs else
                "There is no RGB-D rig in camera.cameras.rigs at all, so this profile cannot build a grasp "
                "cell until one is added.")
        raise CameraRefused(f"{subject}, not an RGB-D device. {why} {tail}", reason="not_rgbd")
    return rig


class Camera:
    """The owner of one rig's device: open it, hand out frames and handles, give it back.

    Build it with :meth:`from_tree` (the usual way), :meth:`from_config` or :meth:`from_rig`; building touches no
    device. A ``with`` block opens and releases it:

    ```python
    with Camera.from_tree(load_tree()) as camera:
        frame = camera.grab()          # colour (BGR uint8) and depth (uint16 millimetres)
    ```

    :meth:`open` claims the device in the process registry, so two owners never share one, and :meth:`release` gives
    both back and never raises. Every grab and every lens read goes through the rig's lock.

    Args:
        rig (CameraRigConfig): The rig, one entry of ``camera.cameras.rigs``.
        streamer (Any): The device streamer behind the owner: the rig's own, or a device double.
    """

    def __init__(self, rig: CameraRigConfig, streamer: Any) -> None:
        self._rig = rig
        self._streamer = streamer
        self._keys = device_keys(rig)
        self._lock = threading.Lock()
        self._open = False
        #: Until when, on the monotonic clock, no frame is handed out to look at: the quiet after the last measuring
        #: grab or notice that the camera moved (:meth:`peek`).
        self._quiet_until = float("-inf")

    @classmethod
    def from_rig(cls, rig: CameraRigConfig, *, streamer: Maybe[Any] = UNSET) -> Camera:
        """The owner of one rig, not yet open.

        Args:
            rig (CameraRigConfig): The rig, one entry of ``camera.cameras.rigs``.
            streamer (Maybe[Any]): The device streamer to put behind the owner, such as a device double; unset is the
                streamer the rig names (default: UNSET).

        Returns:
            Camera: The owner; nothing is opened.
        """
        return cls(rig, streamer if chosen(streamer) else create_streamer(rig))

    @classmethod
    def from_config(cls, camera_cfg: Any, *, rig_id: Maybe[str] = UNSET,
                    open_disabled: Maybe[bool] = UNSET) -> Camera:
        """The owner of the rig a camera section gives, not yet open.

        Args:
            camera_cfg (Any): The camera section, ``tree.app_config.camera``.
            rig_id (Maybe[str]): Which rig of ``camera.cameras.rigs``; unset is the one the cell runs on,
                ``camera.cameras.primary_rig_id`` (default: UNSET).
            open_disabled (Maybe[bool]): ``True`` opens a rig switched off (``enabled: false``), as the bench exerciser
                does; unset refuses it (default: UNSET).

        Returns:
            Camera: The owner; nothing is opened.

        Raises:
            CameraRefused: The section does not configure the rig, the rig is switched off, or it has no depth (the
                message lists the RGB-D rigs it has).
        """
        return cls.from_rig(select_rig(camera_cfg, rig_id=rig_id, open_disabled=open_disabled))

    @classmethod
    def from_tree(cls, tree: Any, *, rig_id: Maybe[str] = UNSET,
                  open_disabled: Maybe[bool] = UNSET) -> Camera:
        """The owner of the rig a loaded tree's camera section gives, not yet open.

        Args:
            tree (Any): A loaded tree, ``load_tree()``.
            rig_id (Maybe[str]): Which rig of ``camera.cameras.rigs``; unset is the one the cell runs on,
                ``camera.cameras.primary_rig_id`` (default: UNSET).
            open_disabled (Maybe[bool]): ``True`` opens a rig switched off (``enabled: false``), as the bench exerciser
                does; unset refuses it (default: UNSET).

        Returns:
            Camera: The owner; nothing is opened.

        Raises:
            ConfigError: The tree did not load.
            CameraRefused: The rig is not configured, switched off, or has no depth.
        """
        return cls.from_config(tree.app_config.camera, rig_id=rig_id, open_disabled=open_disabled)

    @property
    def rig(self) -> CameraRigConfig:
        """The rig this owner holds, as the camera section configures it."""
        return self._rig

    @property
    def rig_id(self) -> str:
        """The rig's id, as ``camera.cameras.rigs`` names it."""
        return self._rig.rig_id

    @property
    def source(self) -> str:
        """The rig's device kind, such as ``"realsense"``."""
        return self._rig.source

    @property
    def enabled(self) -> bool:
        """Whether the rig is switched on in the camera section (``enabled``)."""
        return bool(self._rig.enabled)

    @property
    def calibrated(self) -> bool:
        """Whether the rig DECLARES its calibration, `camera.cameras.rigs[<id>].extrinsics`.

        Declared, not loadable: a block written before the sweep that writes its artifact had run is
        declared and names a file that is not there. `calibration()` is what loads it, and it raises
        `RigCalibrationError` saying which of the two it is; a caller that wants either answer without
        a branch calls that and catches the one exception.
        """
        return getattr(self._rig, "extrinsics", None) is not None

    def calibration(self) -> RigCalibration:
        """The rig's calibration: where the camera sits against the robot, through the one loader.

        Returns:
            RigCalibration: Its mounting (fixed or on the wrist), its transform and where it was solved.

        Raises:
            RigNotCalibrated: The rig declares no calibration; it names the key.
            RigCalibrationError: The declared artifact does not load; it says which key and why.
        """
        return RigCalibration.from_config(self.rig_id, getattr(self._rig, "extrinsics", None))

    @property
    def is_open(self) -> bool:
        """Whether the device is open for this owner."""
        return self._open

    def open(self) -> None:
        """Claim the device for this owner and open it. Idempotent.

        Refused with `CameraBusy`, before the device is touched, while another live owner holds it.
        A device that fails to open gives the claim back and raises what the device raised.
        """
        with self._lock:
            if self._open:
                return
            self._claim()
            try:
                self._streamer.open()
            except BaseException:
                self._unclaim()
                raise
            self._open = True

    def release(self) -> None:
        """Give the device and the claim back. Idempotent, and it never raises: a camera that cannot
        be closed must not stop the thing that was closing it. The failure is logged, because a
        device that would not close is the reason the next open fails."""
        with self._lock:
            if not self._open:
                return
            try:
                self._streamer.release()
            except Exception as exc:  # noqa: BLE001 (teardown reports, it does not propagate)
                logger.warning("Release of rig %s failed: %s", self.rig_id, exc)
            finally:
                self._open = False
                self._unclaim()

    def __enter__(self) -> Camera:
        """Open the device for a ``with`` block and hand the block this owner. Refused as
        :meth:`open` is."""
        self.open()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        """Give the device back as :meth:`release` does, however the block ended. An exception
        propagates.

        The block releases the owner it holds, also one that was already open when the block began.
        """
        self.release()

    def grab(self) -> AnyFrame:
        """One frame for measuring, taken under the rig's lock and stamped with the host time read just before the grab.

        Returns:
            AnyFrame: An ``RGBDFrame`` for an RGB-D rig: ``color`` (BGR ``uint8``), ``depth`` (``uint16`` millimetres)
                and the stamp. Every grab keeps the rig quiet for display frames for a while after it (:meth:`peek`).

        Raises:
            CameraNotOpen: The owner is not open.
        """
        with self._lock:
            self._require_open()
            captured_at_s = time.time()
            try:
                frame = self._streamer.grab()
            finally:
                self._quiet_until = _monotonic() + self._quiet_s()
        return _stamped(frame, captured_at_s)

    def camera_moved(self) -> None:
        """Say that the camera moved since its last grab, so its next frame holds only depth seen from where it is now.

        A RealSense's temporal filter averages each depth pixel over the frames it was handed and fills a hole with a
        depth it saw in them; on a camera the arm carries, those frames were taken at the pose before the move. This
        drops that history. A caller that moved the camera calls it before the next grab: the planning world after
        every move of the arm. Taken under the rig's lock, so it never lands inside a grab, and a no-op on a device
        that keeps no history and on an owner that is not open. The measuring grabs that follow it are due, so no
        frame is handed out to look at for a while (:meth:`peek`).
        """
        with self._lock:
            if not self._open:
                return
            self._quiet_until = _monotonic() + self._quiet_s()
            moved = getattr(self._streamer, "camera_moved", None)
            if callable(moved):
                moved()

    def peek(self) -> PeekFrame | None:
        """A colour image for a person to look at, never one to measure with: what a camera window takes.

        It changes nothing the measuring grabs rely on: a RealSense is read through its display path (the newest colour
        image, no filter, no alignment), and it never waits for the rig. It answers ``None`` while a measuring grab
        holds the lock, for :data:`PEEK_QUIET_S` after every measuring grab and :meth:`camera_moved`, and where the
        device has no new image.

        Returns:
            PeekFrame | None: The image with a stamp of its own, which marks no measuring frame; ``None`` as above.

        Raises:
            CameraNotOpen: The owner is not open.
            RuntimeError: A device that keeps a depth history and offers no display path: looking would change what it
                measures.
        """
        if not self._lock.acquire(blocking=False):
            return None  # a measuring grab holds the rig: it goes first, and the window looks again later
        try:
            self._require_open()
            if _monotonic() < self._quiet_until:
                return None
            display = getattr(self._streamer, "peek", None)
            captured_at_s = time.time()
            if callable(display):
                colour = display()
            elif callable(getattr(self._streamer, "camera_moved", None)):
                raise RuntimeError(
                    f"rig {self.rig_id!r} keeps a history its grabs feed (camera_moved) and offers no display frame "
                    "(peek), so a window would change what it measures: nothing is shown from it")
            else:
                colour = _colour_of(self._streamer.grab())
        finally:
            self._lock.release()
        if colour is None:
            return None
        return PeekFrame(color=np.array(colour, copy=True), captured_at_s=captured_at_s)

    def _quiet_s(self) -> float:
        """How long no frame is handed out to look at after a measuring grab (:data:`PEEK_QUIET_S`)."""
        fps = getattr(self._rig, "fps", 0)
        frames = 3.0 / float(fps) if isinstance(fps, (int, float)) and fps > 0 else 0.0
        return max(PEEK_QUIET_S, frames)

    def get_intrinsics(self) -> np.ndarray | None:
        """The camera matrix the device reports.

        Returns:
            np.ndarray | None: The 3 x 3 pinhole matrix in pixels; ``None`` where the rig has no single one.
        """
        with self._lock:
            self._require_open()
            read = getattr(self._streamer, "get_intrinsics", None)
            return read() if callable(read) else None

    def get_distortion(self) -> np.ndarray | None:
        """The lens distortion coefficients the device reports.

        Returns:
            np.ndarray | None: The coefficients, OpenCV's order; ``None`` where it has none.
        """
        with self._lock:
            self._require_open()
            read = getattr(self._streamer, "get_distortion", None)
            return read() if callable(read) else None

    def handle(self) -> RigHandle:
        """A handle shaped like the streamer its consumer expects, reaching this owner and no other.

        Returns:
            RigHandle: What a perception source or a planner world reads frames through.
        """
        from src.camera.orchestration.frame_provider import RigHandle  # noqa: PLC0415 (it imports this module)

        return RigHandle.of_camera(self)

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """The camera as a person reads it.

        Returns:
            str: Such as ``camera 'EIH_Cam' (realsense, serial ...): open``.
        """
        switched = "" if self.enabled else ", enabled: false"
        state = "open" if self._open else "closed"
        devices = " and ".join(_describe(key) for key in dict.fromkeys(self._keys))
        return f"camera {self.rig_id!r} ({self.source}, {devices}): {state}{switched}"

    def to_dict(self) -> dict[str, Any]:
        """The camera as plain data.

        Returns:
            dict[str, Any]: ``rig_id``, ``source``, ``enabled``, ``open`` and ``devices``.
        """
        return {
            "rig_id": self.rig_id,
            "source": self.source,
            "enabled": self.enabled,
            "open": self._open,
            "devices": [_describe(key) for key in dict.fromkeys(self._keys)],
        }

    def __repr__(self) -> str:
        return f"Camera({self.rig_id!r})"

    def _require_open(self) -> None:
        if not self._open:
            raise CameraNotOpen(f"rig {self.rig_id!r} is not open; call open() before reading it")

    def _claim(self) -> None:
        with _REGISTRY_LOCK:
            for mine in self._keys:
                for held, holder in list(_HOLDERS.items()):
                    if holder is not self and _collides(mine, held):
                        raise CameraBusy(self.rig_id, holder.rig_id, mine)
            for key in self._keys:
                _HOLDERS[key] = self

    def _unclaim(self) -> None:
        with _REGISTRY_LOCK:
            for key in self._keys:
                if _HOLDERS.get(key) is self:
                    del _HOLDERS[key]


def _collides(a: DeviceKey, b: DeviceKey) -> bool:
    return a[0] == b[0] and (a[1] is None or b[1] is None or a[1] == b[1])


def _describe(key: DeviceKey) -> str:
    kind, ident = key
    name = "RealSense" if kind == "realsense" else "video device"
    if ident is None:
        return f"{name} with no {'serial' if kind == 'realsense' else 'id'}, which is any {name}"
    return f"{name} {ident}"


def _stamped(frame: Any, captured_at_s: float) -> Any:
    """The frame with its capture time where its kind carries one, anything else unchanged."""
    if dataclasses.is_dataclass(frame) and not isinstance(frame, type) and hasattr(frame, "captured_at_s"):
        return dataclasses.replace(frame, captured_at_s=captured_at_s)
    return frame


def _colour_of(frame: Any) -> Any:
    """The image of a frame a person looks at: an RGB-D frame's colour, a stereo pair's left eye, or the frame."""
    colour = getattr(frame, "color", None)
    if colour is None:
        colour = getattr(frame, "left", None)
    return frame if colour is None else colour
