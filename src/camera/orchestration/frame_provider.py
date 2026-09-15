from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:                                    # pragma: no cover (typing only)
    import numpy as np

from src.config.schema.camera import RGBDDeviceRigConfig
from src.calibration.stereo.manager import StereoCam3D
from src.camera.orchestration.camera import (
    AnyStreamer,
    Camera,
    CameraRigConfig,
    create_streamer,
)
from src.camera.setup.image_taking.frames import AnyFrame, StereoFrame

__all__ = [
    "FrameProvider",
    "FrameProviderStateError",
    "RigHandle",
    "UnknownCameraRigError",
]


class UnknownCameraRigError(KeyError):
    """Raised when a requested camera rig ID is not registered."""


class FrameProviderStateError(RuntimeError):
    """Raised when frames are requested before the provider is opened."""


class FrameProvider:
    """The multi-rig catalogue: frames keyed by rig id, over stereo, single-device and RGB-D rigs.

    It owns rig indexing and which rigs it holds. Every rig it opens is held by a
    :class:`~src.camera.orchestration.camera.Camera` owner, so the catalogue and a camera owner
    elsewhere in the process never both open one device, and every grab through the catalogue runs
    under that rig's lock. Acquisition, splitting, cropping, resizing and quality setup stay in the
    streamers under ``camera.setup``, and :meth:`rig` hands one rig to a consumer as a `RigHandle`.
    """

    #: The streamer a rig names, built in the camera owner's module, the one module that constructs
    #: device streamers. A class attribute, so device doubles can stand behind the catalogue.
    _create_streamer = staticmethod(create_streamer)

    def __init__(
        self,
        rigs: list[CameraRigConfig],
        stereo: StereoCam3D | None = None,
    ) -> None:
        if not rigs:
            raise ValueError("At least one rig must be provided.")

        self.logger = logging.getLogger(__name__)
        self._rigs: dict[str, CameraRigConfig] = {}
        self._streamers: dict[str, AnyStreamer] = {}
        self._stereo = stereo
        self._stereo_rig_index: dict[str, int] = {}
        #: Which rigs stream right now. Per rig rather than one flag, so a consumer handed a
        #: single rig gives that one back without closing devices it never held. See
        #: `RigHandle.release`.
        self._open_rigs: set[str] = set()
        #: The owner holding each open rig.
        self._cameras: dict[str, Camera] = {}

        stereo_idx = 0
        for rig in rigs:
            rig_id = rig.rig_id
            if rig_id in self._rigs:
                raise ValueError(f"Duplicate rig_id: {rig_id}")

            self._rigs[rig_id] = rig
            self._streamers[rig_id] = self._create_streamer(rig)

            if not isinstance(rig, RGBDDeviceRigConfig):
                self._stereo_rig_index[rig_id] = stereo_idx
                stereo_idx += 1

        self.logger.info("FrameProvider created with rigs: %s", list(self._rigs))

    @property
    def is_open(self) -> bool:
        """True when every rig is streaming.

        Read from the per-rig set, so a provider whose one rig was handed back reports false
        and the next ``grab`` on it fails instead of reaching a released device.
        """
        return bool(self._rigs) and self._open_rigs == set(self._rigs)

    def open(self) -> None:
        """Open every rig, rolling back the ones this call opened on failure."""
        if self.is_open:
            return
        opened: list[str] = []
        try:
            for rig_id in self._rigs:
                if rig_id in self._open_rigs:
                    continue
                self.open_rig(rig_id)
                opened.append(rig_id)
        except Exception:
            self.logger.exception("Failed to open rig; rolling back opened streamers")
            for rig_id in opened:
                self.release_rig(rig_id)
            raise

    def open_rig(self, rig_id: str) -> None:
        """Open one rig and leave the others untouched. Idempotent.

        The counterpart to `release_rig`, for a consumer that needs a single rig where
        `open()` claims every configured one. No ``__init__`` under
        `camera.setup.image_taking` touches a device, it only stores config, so a provider
        knows every rig while holding open only the ones it was asked for.

        The rig is held by a camera owner, so a device that another owner in this process
        holds refuses here with ``CameraBusy``, naming the holder, and nothing is marked open.
        """
        streamer = self._require_streamer(rig_id)
        if rig_id in self._open_rigs:
            return
        camera = Camera.from_rig(self._rigs[rig_id], streamer=streamer)
        camera.open()
        self._cameras[rig_id] = camera
        self._open_rigs.add(rig_id)

    def release(self) -> None:
        """Release every streamer. Safe to call repeatedly."""
        for rig_id in list(self._open_rigs):
            self.release_rig(rig_id)

    def release_rig(self, rig_id: str) -> None:
        """Release one rig and leave every other one streaming. Idempotent, and it never raises.

        Takes a rig id rather than tearing down the provider, so a consumer gives back the rig
        it was handed and cannot close one it was not. The console's only teardown path ends
        here, through a built service: ``api/lifecycle.release_perception`` ->
        ``perception.close()``.

        Teardown must not be stopped by a camera that will not close, so the rig's owner logs
        a failure rather than raising it. The log is where the reason for the next failed open
        shows up.
        """
        self._require_rig(rig_id)
        if rig_id not in self._open_rigs:
            return
        camera = self._cameras.pop(rig_id, None)
        if camera is not None:
            camera.release()
        self._open_rigs.discard(rig_id)

    def __enter__(self) -> FrameProvider:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        self.release()
        return False

    def grab(self, rig_id: str) -> AnyFrame:
        """Grab a raw frame from ``rig_id`` (a StereoFrame or RGBDFrame per the rig kind), stamped
        with its capture time by the rig's owner."""
        return self.camera(rig_id).grab()

    def grab_rectified(self, rig_id: str) -> StereoFrame:
        """Grab a stereo frame and rectify it with ``StereoCam3D``."""
        self._require_rig(rig_id)
        if rig_id not in self._stereo_rig_index:
            raise ValueError(f"Rig {rig_id!r} is RGB-D and does not support stereo rectification")
        if self._stereo is None:
            raise RuntimeError("StereoCam3D is required for grab_rectified()")

        frame = self.grab(rig_id)
        if not isinstance(frame, StereoFrame):
            raise TypeError(f"Rig {rig_id!r} returned {type(frame).__name__}, expected StereoFrame")
        index = self._stereo_rig_index[rig_id]
        left_rectified, right_rectified = self._stereo.rectify(frame.left, frame.right, rig=index)
        return StereoFrame(left=left_rectified, right=right_rectified, captured_at_s=frame.captured_at_s)

    @property
    def rig_ids(self) -> list[str]:
        """All registered rig identifiers in configuration order."""
        return list(self._rigs)

    def is_rgbd(self, rig_id: str) -> bool:
        """True if ``rig_id`` is an RGB-D device."""
        return isinstance(self._require_rig(rig_id), RGBDDeviceRigConfig)

    def is_stereo(self, rig_id: str) -> bool:
        """True if ``rig_id`` is a stereo rig."""
        self._require_rig(rig_id)
        return not self.is_rgbd(rig_id)

    def get_rig_config(self, rig_id: str) -> CameraRigConfig:
        """Return the original config object for ``rig_id``."""
        return self._require_rig(rig_id)

    def get_stereo_rig_index(self, rig_id: str) -> int:
        """Return the ``StereoCam3D`` rig index for ``rig_id``."""
        self._require_rig(rig_id)
        try:
            return self._stereo_rig_index[rig_id]
        except KeyError as exc:
            raise KeyError(f"Rig {rig_id!r} has no stereo index because it is RGB-D") from exc

    def get_intrinsics(self, rig_id: str) -> "np.ndarray | None":
        """The camera matrix ``rig_id`` reports, or ``None`` if its streamer has none.

        The rig must be open. K is what gives a frame a metric meaning, and both consumers of
        a real camera need it: the perception adapter to unproject, and `GraspCalculator` at
        construction. Serving it here keeps them inside the rig-keyed lifecycle instead of
        opening a `RealSenseRGBDStreamer` of their own. A stereo rig answers ``None``, its
        geometry living in `StereoCam3D` rather than in a single pinhole matrix.
        """
        return self.camera(rig_id).get_intrinsics()

    def open_rig_ids(self) -> frozenset[str]:
        """Which rigs are streaming right now."""
        return frozenset(self._open_rigs)

    def get_distortion(self, rig_id: str) -> "np.ndarray | None":
        """The lens distortion coefficients ``rig_id`` reports, or ``None`` when it has none.

        The rig must be open. Hand-eye calibration needs the coefficients, which is why this
        sits beside `get_intrinsics`. `calibration.rgbd_marker_source.RGBDArucoMarkerSource`,
        which lets a fixed RGB-D camera see the ArUco board, duck-types against ``grab``,
        ``get_intrinsics`` and ``get_distortion``; a rig serving only the first two is
        addressable but cannot be calibrated.
        """
        return self.camera(rig_id).get_distortion()

    def camera(self, rig_id: str) -> Camera:
        """The owner holding ``rig_id`` while it is open. A rig that is not open refuses here
        rather than reaching a device that was never opened or was already given back."""
        self._require_rig(rig_id)
        self._require_open(rig_id)
        return self._cameras[rig_id]

    def rig(self, rig_id: str) -> "RigHandle":
        """A :class:`RigHandle` to one rig, shaped like the streamer its consumer expects.

        The handle answers ``grab()``, ``get_intrinsics()`` and ``release()``, the surface
        `RealSenseVisionPerceptionSource` duck-types against, and reaches this rig and no other.
        """
        self._require_rig(rig_id)
        return RigHandle(self, rig_id)

    def _require_open(self, rig_id: str | None = None) -> None:
        if rig_id is None:
            if not self.is_open:
                raise FrameProviderStateError(
                    "FrameProvider is not open; call open() before grabbing frames")
            return
        if rig_id not in self._open_rigs:
            raise FrameProviderStateError(
                f"Rig {rig_id!r} is not open; call open() before grabbing frames "
                f"(open rigs: {sorted(self._open_rigs) or 'none'})")

    def _require_rig(self, rig_id: str) -> CameraRigConfig:
        try:
            return self._rigs[rig_id]
        except KeyError as exc:
            raise UnknownCameraRigError(f"Unknown rig_id: {rig_id!r}") from exc

    def _require_streamer(self, rig_id: str) -> AnyStreamer:
        self._require_rig(rig_id)
        return self._streamers[rig_id]


class RigHandle:
    """One rig, shaped like the streamer its consumer expects.

    The robot pick path consumes a duck-typed streamer, ``grab()``, ``get_intrinsics()`` and
    ``release()``, and so does `datagen`'s camera probe, which passes a shim and runs with no
    hardware present. A handle serves that surface, so neither learns about camera
    orchestration while every frame goes through the one owner of the rig's device.

    A handle reaches its rig through the catalogue that handed it out (`FrameProvider.rig`) or
    through the owner itself (`Camera.handle`). Either way ``camera`` is that owner, and every
    call runs under its lock. A handle reaches one rig. ``release()`` gives that rig back and
    leaves the others streaming.
    """

    __slots__ = ("_camera", "_provider", "rig_id")

    def __init__(self, provider: FrameProvider | None, rig_id: str, *, camera: Camera | None = None) -> None:
        if (provider is None) == (camera is None):
            raise ValueError("a RigHandle reaches its rig through a provider or through a camera, exactly one")
        self._provider = provider
        self._camera = camera
        self.rig_id = rig_id

    @classmethod
    def of_camera(cls, camera: Camera) -> RigHandle:
        """A handle that reaches ``camera`` directly."""
        return cls(None, camera.rig_id, camera=camera)

    def __repr__(self) -> str:
        return f"RigHandle({self.rig_id!r})"

    @property
    def camera(self) -> Camera:
        """The owner of this rig's device. Through a catalogue, only while the rig is open."""
        if self._provider is not None:
            return self._provider.camera(self.rig_id)
        assert self._camera is not None  # one of the two, checked at construction
        return self._camera

    def grab(self) -> AnyFrame:
        """One frame from this rig."""
        return self.camera.grab()

    def get_intrinsics(self) -> "np.ndarray | None":
        """This rig's camera matrix, or ``None`` where the rig has no single pinhole matrix."""
        return self.camera.get_intrinsics()

    def get_distortion(self) -> "np.ndarray | None":
        """This rig's distortion coefficients, or ``None`` where the rig reports none.

        Completes the surface hand-eye calibration duck-types against, so a handle can be
        calibrated and not only read.
        """
        return self.camera.get_distortion()

    def release(self) -> None:
        """Give this rig back. Idempotent, never raises, and it touches no other rig."""
        if self._provider is not None:
            self._provider.release_rig(self.rig_id)
            return
        assert self._camera is not None  # one of the two, checked at construction
        self._camera.release()

    @property
    def is_open(self) -> bool:
        if self._provider is not None:
            return self.rig_id in self._provider.open_rig_ids()
        assert self._camera is not None  # one of the two, checked at construction
        return self._camera.is_open
