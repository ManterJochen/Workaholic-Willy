"""Replay a datagen scene as a perception rig the real grasping stack can drive.

The RL trainers read fifteen feature keys, thirteen of which are stack telemetry (fusion,
uncertainty, drift, latencies) that `shadow.py` produces. So rather than hand-build a bridge from
datagen rows to `GraspAttemptRecord`, this drives the actual service over datagen scenes and lets
it emit its own records: a scene on disk, presented as the rig protocol the pick loop already
speaks.

Why datagen imports the robot library and not the other way round. `datagen/eval/ladder.py`
already imports `GraspCalculator` and the grasping support config: datagen is a consumer of the
robot stack, not a dependency of it. Keeping this file here preserves that direction; putting it
under `src/` would make the robot library depend on the data generator.

The simultaneity caveat does not apply here, structurally. `MappedCameraRig` warns that it
triggers its cameras sequentially, which is unsound for a bin whose parts are still moving. A
datagen scene is settled and written to disk, so the views are the same instant by construction
and no amount of sequential reading can change that.

Fidelity is a deliberate choice, not a default. Scenes ship GT masks and predicted masks, clean
depth and noisy depth. The realistic pair is what makes the uncertainty features carry signal
instead of constants, and it is what this module defaults to: a predicted mask can miss an object
entirely, and noisy depth drops pixels. Clean plus GT would hand the stack a scene no camera
produces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.utility.log_cfg import create_logger

from datagen.constants import DATAGEN_LOG_DIR, RL_PERCEPTION_LOG_FILE

__all__ = [
    "DatagenSegmentation",
    "DatagenViewSource",
    "SceneRig",
    "load_scene_rig",
]

logger = create_logger("datagen.rl.perception", RL_PERCEPTION_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

#: Views whose render did not complete are absent, not empty. Same check the evaluator uses.
_RENDERED = ("rendered", "rendered_after_resample")

#: Instance ids are 1-based in the mask PNG: 0 is background, object k is written as k+1.
_MASK_OFFSET = 1


# Not frozen, deliberately: `SegmentationLike` declares `mask` as a mutable attribute, and a
# frozen dataclass field is not assignable, so a frozen version does not structurally satisfy the
# protocol. Matching the protocol the stack declares beats a local immutability preference that
# would need a cast to hide.
@dataclass(slots=True)
class DatagenSegmentation:
    """One object's mask in one view, carrying the identity the pick loop's label gate needs.

    ``label`` is the asset id, which is what an operator prompt would name. The pick loop's hard
    target gate reads ``.label`` (falling back to ``.prim_path``), so a segmentation without one
    can never be an executable target when a label is set; that refusal is named
    ``TARGET_LABEL_NOT_FOUND`` rather than silent, but it is still a refusal.
    """

    mask: np.ndarray
    instance_id: int
    label: str
    #: Fraction of this object visible in this view, from the scene manifest. Ground truth, and the
    #: honest reference for anything the perception stack later estimates about occlusion.
    visibility: float = 1.0
    visible_px: int = 0


@dataclass(frozen=True, slots=True)
class DatagenViewSource:
    """One camera of one scene, as a :class:`PerceptionSource`.

    Deterministic and re-readable: ``acquire()`` re-reads the PNGs each call rather than caching, so a
    loop that acquires twice gets two independent arrays and cannot be corrupted by a caller that
    writes into one of them. The files are small and this is not on a robot's motion path.
    """

    scene_dir: Path
    view_name: str
    intrinsics_mm: np.ndarray
    camera_to_base_mm: np.ndarray
    objects: tuple[dict[str, Any], ...]
    #: ``"pred"`` (what a detector produced) or ``"gt"`` (what the renderer knows).
    mask_source: str = "pred"
    #: ``"noisy"`` (sensor-like, with dropouts) or ``"clean"``.
    depth_source: str = "noisy"
    #: Counts acquisitions so a multi-pick run is visibly not one cached frame.
    frames_served: list[int] = field(default_factory=lambda: [0])

    def acquire(self) -> Any:
        from src.robot.grasping.types.perception import PerceptionFrame

        from PIL import Image

        depth_name = "depth_noisy" if self.depth_source == "noisy" else "depth"
        mask_name = "pred_instances" if self.mask_source == "pred" else "instances"
        depth = np.asarray(
            Image.open(self.scene_dir / f"{self.view_name}_{depth_name}.png"), dtype=np.float64
        )
        instances = np.asarray(Image.open(self.scene_dir / f"{self.view_name}_{mask_name}.png"))
        rgb_path = self.scene_dir / f"{self.view_name}_rgb.png"
        rgb = np.asarray(Image.open(rgb_path).convert("RGB")) if rgb_path.is_file() else None

        segmentations: list[DatagenSegmentation] = []
        for entry in self.objects:
            instance_id = int(entry["instance_id"])
            mask = instances == (instance_id + _MASK_OFFSET)
            if not mask.any():
                # Absent, not empty. With predicted masks this is the common case and it is the
                # point of choosing them: the detector missed the object, so the stack must not be
                # handed a zero-pixel segmentation it would then reject for the wrong reason.
                continue
            segmentations.append(DatagenSegmentation(
                mask=mask,
                instance_id=instance_id,
                label=str(entry.get("asset_id") or f"instance_{instance_id}"),
                visibility=float(entry.get("visibility", 1.0)),
                visible_px=int(entry.get("visible_px", 0)),
            ))
        self.frames_served[0] += 1
        return PerceptionFrame(
            depth_map=depth,
            intrinsics=np.asarray(self.intrinsics_mm, dtype=np.float64),
            segmentations=tuple(segmentations),
            rgb=rgb,
        )


@dataclass(frozen=True, slots=True)
class SceneRig:
    """Everything one datagen scene hands the grasping stack.

    ``primary`` is the single-camera source the pick loop acquires from; ``rig`` and ``resolvers``
    are what multi-camera fusion needs. Handing all three from one loader keeps the camera ids in
    ``rig`` and in ``resolvers`` provably the same set: an id present in one and missing from the
    other is exactly the mismatch the orchestrator drops a camera for.
    """

    scene_id: str
    family: str
    primary_view: str
    primary: DatagenViewSource
    rig: Any
    resolvers: dict[str, Any]
    sources: dict[str, DatagenViewSource]
    payload: dict[str, Any]

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(self.sources)


def _resolver_for(camera_to_base_mm: np.ndarray) -> Any:
    """A fixed CAMERA->BASE resolver from the scene's own extrinsic.

    Built from the matrix in scene.json rather than from a calibration artifact, and that is the
    honest description of what it is: the renderer's ground truth, not a measurement. On a real cell
    this transform is the output of a hand-eye routine and carries its residual; here it is exact by
    construction, so anything downstream that looks good because the extrinsic is perfect is
    measuring the simulator, not the cell.
    """
    from src.geometry import Frame, Transform

    matrix = np.asarray(camera_to_base_mm, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"camera_to_base_mm must be 4x4, got {matrix.shape}")
    from src.robot.grasping.motion.frame_resolver import StaticCameraToBaseResolver

    transform = Transform.from_matrix(matrix, from_frame=Frame.CAMERA, to_frame=Frame.BASE)
    return StaticCameraToBaseResolver(transform=transform)


def load_scene_rig(
    scene_dir: Path | str,
    *,
    primary_view: str | None = None,
    mask_source: str = "pred",
    depth_source: str = "noisy",
) -> SceneRig:
    """Read one scene directory into a rig the grasping stack can drive.

    ``primary_view`` defaults to the first rendered view in the manifest, which keeps the choice
    deterministic and makes it visible in the returned rig rather than hidden in a default argument.
    A scene with no rendered view raises: an empty rig that silently produces no frames would surface
    downstream as a grasping failure, which is the wrong place to learn that a render did not finish.
    """
    from src.robot.grasping.types.perception import MappedCameraRig

    scene_dir = Path(scene_dir)
    payload = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))

    sources: dict[str, DatagenViewSource] = {}
    resolvers: dict[str, Any] = {}
    for view in payload.get("views", ()):
        if str(view.get("outcome")) not in _RENDERED:
            continue
        name = str(view["name"])
        sources[name] = DatagenViewSource(
            scene_dir=scene_dir,
            view_name=name,
            intrinsics_mm=np.asarray(view["intrinsics"], dtype=np.float64),
            camera_to_base_mm=np.asarray(view["camera_to_base_mm"], dtype=np.float64),
            objects=tuple(view.get("objects", ())),
            mask_source=mask_source,
            depth_source=depth_source,
        )
        resolvers[name] = _resolver_for(view["camera_to_base_mm"])

    if not sources:
        raise ValueError(
            f"{scene_dir.name} has no rendered view (every entry's outcome is outside {_RENDERED}). "
            f"A rig built from it would produce no frames at all, which reaches the pick loop as a "
            f"grasping failure and sends the reader to the grasp stack for a rendering problem."
        )

    chosen = primary_view or next(iter(sources))
    if chosen not in sources:
        raise ValueError(
            f"primary_view {chosen!r} is not a rendered view of {scene_dir.name}; "
            f"available: {sorted(sources)}"
        )

    scene_id = str(payload.get("spec", {}).get("scene_id") or scene_dir.name)
    family = str(payload.get("spec", {}).get("family") or "")
    declared = tuple(str(view.get("name", "?")) for view in payload.get("views", ()))
    dropped = tuple(name for name in declared if name not in sources)
    if dropped:
        # A rig with fewer cameras than the scene declares is a quieter scene, not a broken one:
        # fusion still runs, on less evidence, and every downstream number moves with it. The sweeps
        # that consume this rig record only that they got one, so the loss has to be named here.
        logger.warning("%s: %d of %d view(s) never rendered (%s); the rig carries %d camera(s)",
                       scene_id, len(dropped), len(declared), ", ".join(dropped), len(sources))
    # One line per scene, minutes of stack driving apart rather than a tight loop. The primary
    # view is chosen implicitly ("the first rendered one"), so a render outcome that flipped
    # silently moves which camera every single-camera pick was made from; this is the only place
    # that is recorded.
    logger.info("%s (%s): %d camera(s) %s, primary %s, masks=%s, depth=%s",
                scene_id, family, len(sources), sorted(sources), chosen, mask_source, depth_source)

    return SceneRig(
        scene_id=scene_id,
        family=family,
        primary_view=chosen,
        primary=sources[chosen],
        rig=MappedCameraRig(sources=dict(sources)),
        resolvers=resolvers,
        sources=sources,
        payload=payload,
    )
