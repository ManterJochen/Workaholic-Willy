"""The perception snapshot value objects the grasping stack shares.

A :class:`PerceptionSource` hands the pick loop a :class:`PerceptionFrame`, which is one
depth, RGB and segmentation snapshot of the bin. They live in the bottom ``types`` tier,
so every downstream module, the calculator, the frame resolver, refinement, verification
and recovery, names them without importing the orchestrator that consumes them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover (a type-only import, so this module stays dependency-light)
    from src.geometry import Pose

import numpy as np


class SegmentationLike(Protocol):
    """Minimal segmentation boundary: only a ``mask`` (HxW bool/uint8 array) is required."""

    mask: np.ndarray


@dataclass(frozen=True, slots=True)
class PerceptionFrame:
    """One snapshot of the bin from the perception system.

    ``segmentations`` need only satisfy :class:`SegmentationLike`, meaning a ``.mask``
    attribute. The orchestrator does not care which model produced them.
    """

    depth_map: np.ndarray
    intrinsics: np.ndarray
    segmentations: tuple[SegmentationLike, ...]
    rgb: np.ndarray | None = None
    timestamp: float | None = None
    #: Where the tool was when this frame was captured, for an eye-in-hand rig.
    #:
    #: `EyeInHandFrameResolver` computes CAMERA to BASE from the TCP, and without this
    #: field it reads that TCP at resolve time, on the grounds that the frame is not
    #: needed for the transform itself. For a camera bolted to the wrist the transform
    #: depends on where the tool was when the shutter opened, and the closed-loop path
    #: moves the arm between capture and resolve by design. Every millimetre the tool
    #: travelled in between lands in the grasp, in a frame nothing checks.
    #:
    #: `None` means the producer did not stamp it, and the resolver then falls back to
    #: reading the arm, so an existing source is unchanged. It is not a default pose:
    #: inventing one would turn a missing measurement into a wrong one.
    #:
    #: This has never run on hardware. It is bucket 3 until an eye-in-hand cell stamps
    #: one.
    tool_pose: "Pose | None" = None
    #: The depth the camera measured, before any grasp-referenced overwrite.
    #:
    #: `depth_map` is the grasp path's view of the scene: a producer may replace the depth
    #: inside a detected mask with one number, the object's top surface plus a penetration,
    #: because that is the depth a jaw is driven to. A consumer that wants the shape of what
    #: is there cannot use it: every detected object is a sheet at its top face and the body
    #: underneath reads as empty space. This field carries the untouched surface for those
    #: consumers, and the planner's world is the first of them.
    #:
    #: `None` means the producer published nothing, not that the two agree. A consumer that
    #: needs the measured surface refuses rather than falling back to `depth_map`, because
    #: falling back would hand a planner a sheet and call it an obstacle.
    surface_depth_map: np.ndarray | None = None


@runtime_checkable
class PerceptionSource(Protocol):
    """Anything that can produce a :class:`PerceptionFrame` on demand."""

    def acquire(self) -> PerceptionFrame: ...


@dataclass(frozen=True, slots=True)
class CameraObservation:
    """The frame of one named camera, from a rig of several.

    ``camera_id`` is the join key, and it is a contract rather than a label: it matches
    an entry in ``robot.grasping.fusion.cameras``, which is where the CAMERA to BASE
    calibration artifact of that camera is declared. A frame whose id has no calibration
    entry cannot be placed in BASE and so cannot be fused, and the orchestrator drops it
    and names the id rather than guessing at a default extrinsic and fusing a cloud into
    the wrong place.
    """

    camera_id: str
    frame: PerceptionFrame


@runtime_checkable
class MultiCameraPerceptionSource(Protocol):
    """A rig of several cameras observed together.

    It is separate from :class:`PerceptionSource` on purpose. The single-camera protocol
    answers what the camera sees now, and this one answers what the cameras see at the
    same moment, and that simultaneity is why multi-view fusion of a bin of moving parts
    is sound at all. An implementation triggers its cameras as close together as the
    hardware allows and stamps the ``timestamp`` of each frame.

    A camera that failed to produce a frame is simply absent from the returned tuple,
    because the protocol has no error channel by design. Which cameras were expected is
    config, in ``fusion.cameras``, so comparing expected against delivered belongs to
    the caller that holds the config, in one place under one policy,
    ``on_camera_unavailable``. An implementation that invented an empty frame instead
    would defeat that check.
    """

    def acquire_all(self) -> tuple[CameraObservation, ...]: ...


@dataclass(frozen=True, slots=True)
class MappedCameraRig:
    """A :class:`MultiCameraPerceptionSource` assembled from named single-camera sources.

    It is the practical bridge for a cell whose cameras are already separate
    :class:`PerceptionSource` objects, which is what the sim runners and the first real
    rigs look like. Observation order follows the insertion order of the mapping, so it
    is deterministic.

    One caveat is why this is an adapter rather than the reference implementation: it
    triggers its cameras sequentially. The multi-camera protocol exists because fusing
    several views of a bin is sound only where the views show the same instant, and a
    loop of ``acquire()`` calls does not guarantee that. It is harmless for a stepped
    simulator or a settled bin and not harmless for parts that are still moving. A rig
    with a hardware trigger implements the protocol directly rather than wrapping itself
    in this.
    """

    sources: Mapping[str, PerceptionSource]

    def acquire_all(self) -> tuple[CameraObservation, ...]:
        return tuple(
            CameraObservation(camera_id=name, frame=source.acquire())
            for name, source in self.sources.items()
        )

    def close(self) -> None:
        """Hand back every device the sources of this rig hold. Duck-typed, idempotent, never raises.

        It exists because a multi-camera cell opens several devices and a teardown that
        walks `orchestrator.perception.close()`, as `api.lifecycle.release_perception`
        does, reaches the primary camera alone. Every additional camera a fused cell
        opened would then stay open across a rebuild, and the second build could not
        start their pipelines.

        It is duck-typed because only some sources own a device: a sim source owns a
        simulator this rig does not, and a source without `close()` is the normal case
        rather than an error, so this does nothing for a caller that predates a real rig.

        Every camera is attempted and the failures are reported once. One camera that
        will not close must not keep the others open, and a teardown failure that leaves
        no trace is the reason the next build cannot open a device. This module carries
        no logger by design, so the report goes to the caller:
        `api.lifecycle.release_perception` catches and logs it, and a process that is
        exiting does not care.
        """
        failures: list[str] = []
        for name, source in self.sources.items():
            closer = getattr(source, "close", None)
            if not callable(closer):
                continue
            try:
                closer()
            except Exception as exc:  # noqa: BLE001 (every camera is tried before anything raises)
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        if failures:
            raise RuntimeError(
                "MappedCameraRig.close: camera(s) refused to release their device -- "
                + "; ".join(failures)
            )
