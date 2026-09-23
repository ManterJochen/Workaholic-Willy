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
from typing import TYPE_CHECKING, Any, Literal

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

    import numpy as np

    from src.camera.orchestration.frame_provider import RigHandle
    from src.camera.setup.image_taking.frames import AnyFrame

CameraRigConfig = WebcamPairRigConfig | SingleDeviceRigConfig | RGBDDeviceRigConfig
AnyStreamer = WebcamPairStreamer | SingleDeviceStreamer | OpenCvRGBDStreamer | RealSenseRGBDStreamer
#: A device as the registry knows it: its kind and its id. None is no id, which matches any device
#: of the kind.
DeviceKey = tuple[str, "int | str | None"]
RefusalReason = Literal["unknown", "disabled", "not_rgbd"]

__all__ = [
    "AnyStreamer",
    "Camera",
    "CameraBusy",
    "CameraNotOpen",
    "CameraRefused",
    "CameraRigConfig",
    "create_streamer",
    "device_keys",
    "select_rig",
]

logger = logging.getLogger(__name__)

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
    """The owner of one rig's device: open it, hand out handles, give it back.

    Built by `from_config` from a camera section, or by `from_rig` from one rig; building touches no
    device. `open` claims the device in the process registry and opens it, `release` gives both back
    and never raises. Every grab and every lens read goes through the rig's lock. A ``with`` block
    does both:

        with Camera.from_config(app.camera, rig_id="overhead") as camera:
            frame = camera.grab()
    """

    def __init__(self, rig: CameraRigConfig, streamer: Any) -> None:
        self._rig = rig
        self._streamer = streamer
        self._keys = device_keys(rig)
        self._lock = threading.Lock()
        self._open = False

    @classmethod
    def from_rig(cls, rig: CameraRigConfig, *, streamer: Maybe[Any] = UNSET) -> Camera:
        """The owner of ``rig``. ``streamer`` UNSET is the device streamer the rig names; a caller
        passes one to put a device double, or the catalogue's own streamer, behind the owner."""
        return cls(rig, streamer if chosen(streamer) else create_streamer(rig))

    @classmethod
    def from_config(cls, camera_cfg: Any, *, rig_id: Maybe[str] = UNSET,
                    open_disabled: Maybe[bool] = UNSET) -> Camera:
        """The owner of the rig a camera section gives (see `select_rig`), not yet open."""
        return cls.from_rig(select_rig(camera_cfg, rig_id=rig_id, open_disabled=open_disabled))

    @classmethod
    def from_tree(cls, tree: Any, *, rig_id: Maybe[str] = UNSET,
                  open_disabled: Maybe[bool] = UNSET) -> Camera:
        """The owner of the rig a loaded tree's camera section gives, not yet open (see
        :meth:`from_config`).

        A tree that did not load is refused with its own refusal, as ``ConfigError``.
        """
        return cls.from_config(tree.app_config.camera, rig_id=rig_id, open_disabled=open_disabled)

    @property
    def rig(self) -> CameraRigConfig:
        return self._rig

    @property
    def rig_id(self) -> str:
        return self._rig.rig_id

    @property
    def source(self) -> str:
        return self._rig.source

    @property
    def enabled(self) -> bool:
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
        """The rig's calibration, through the one loader. `RigNotCalibrated` names the key for a rig
        that declares none, and `RigCalibrationError` names it for an artifact that does not load."""
        return RigCalibration.from_config(self.rig_id, getattr(self._rig, "extrinsics", None))

    @property
    def is_open(self) -> bool:
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
        """One frame, taken under the rig's lock and stamped with the host time read just before the
        grab."""
        with self._lock:
            self._require_open()
            captured_at_s = time.time()
            frame = self._streamer.grab()
        return _stamped(frame, captured_at_s)

    def camera_moved(self) -> None:
        """Say that the camera moved since its last grab, so its next frame holds only depth seen from where it is now.

        A RealSense's temporal filter averages each depth pixel over the frames it was handed and fills a hole with a
        depth it saw in them; on a camera the arm carries, those frames were taken at the pose before the move. This
        drops that history. A caller that moved the camera calls it before the next grab: the planning world after
        every move of the arm. Taken under the rig's lock, so it never lands inside a grab, and a no-op on a device
        that keeps no history and on an owner that is not open.
        """
        with self._lock:
            if not self._open:
                return
            moved = getattr(self._streamer, "camera_moved", None)
            if callable(moved):
                moved()

    def get_intrinsics(self) -> np.ndarray | None:
        """The camera matrix the device reports, or None where the rig has no single pinhole matrix."""
        with self._lock:
            self._require_open()
            read = getattr(self._streamer, "get_intrinsics", None)
            return read() if callable(read) else None

    def get_distortion(self) -> np.ndarray | None:
        """The lens distortion coefficients the device reports, or None where it has none."""
        with self._lock:
            self._require_open()
            read = getattr(self._streamer, "get_distortion", None)
            return read() if callable(read) else None

    def handle(self) -> RigHandle:
        """A handle shaped like the streamer its consumer expects, reaching this owner and no other."""
        from src.camera.orchestration.frame_provider import RigHandle  # noqa: PLC0415 (it imports this module)

        return RigHandle.of_camera(self)

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        switched = "" if self.enabled else ", enabled: false"
        state = "open" if self._open else "closed"
        devices = " and ".join(_describe(key) for key in dict.fromkeys(self._keys))
        return f"camera {self.rig_id!r} ({self.source}, {devices}): {state}{switched}"

    def to_dict(self) -> dict[str, Any]:
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
