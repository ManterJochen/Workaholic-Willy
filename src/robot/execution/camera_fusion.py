"""Which cameras a cell fuses into each object's cloud before it grasps, or why it fuses none.

One depth view sees one side of a part, and a parallel jaw closes on two opposite faces. A cell with two or more
calibrated cameras fuses each object's surface across them into one cloud in BASE and plans the grasp on that
(:mod:`src.robot.grasping.multiview`, "geometry fusion"). Measured in simulation on the datagen reference: the best
ranked grasp is graspable for 43.5 % of objects from one view and 55.9 % fused. Never run on hardware: no physical
multi-camera cell has been built with this code.

There is no switch for it (the owner, 2026-10-16): a cell fuses by itself wherever ``camera.cameras.rigs`` declares
two or more enabled rigs with their calibration, the primary among them (:func:`fused_cameras`), and grasps from one
view where it does not. A stereo rig is one of them as an RGB-D rig is: its depth is read through its stereo
calibration (2026-10-17), and its owner refuses to open before that calibration is written. :class:`CameraFusionPlan`
reads the tree, opens nothing, and says which cameras it fuses, or what it lacks to fuse any, so a program asks before a
camera opens or a model loads.

A camera on the wrist is one of the views, not a refusal: its resolver places each of its frames by the arm's pose at
the shutter (``EyeInHandFrameResolver``), and the cell stamps its frames as it stamps the primary's. What is measured in
simulation is a wrist primary with fixed cameras fused onto it (``run_multiview_pick``); the shipped reference is two
fixed cameras, ``config/camera/cam.eth2.yaml``.

Beside :class:`~src.robot.execution.camera_world_wiring.CameraWorldPlan`, which answers the other question a
multi-camera cell raises: which rigs feed the live planner world.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.tree import LoadedTree

__all__ = ["CameraFusionPlan", "fused_cameras", "gives_depth"]

#: The rig sources whose depth is matched from a pair through its stereo calibration (``StereoDepthStreamer``).
_STEREO_SOURCES = frozenset({"webcam_pair", "single_device"})

#: What a fused cell needs, said after what this one is missing, with where the reference cell is written down.
_A_FUSED_CELL = (
    "A cell fuses by itself wherever camera.cameras.rigs declares two or more enabled rigs with their calibration, "
    "the primary among them: the reference is config/camera/cam.eth2.yaml, loaded as the profile chain "
    "<your arm>,eth2"
)


def fused_cameras(rigs: Sequence[Any], primary_rig_id: str) -> tuple[str, ...]:
    """The rig ids a cell fuses, the primary first and then the others in id order, or ``()`` where it fuses none.

    Every enabled rig that declares its calibration takes part, an RGB-D rig or a stereo one whose stereo calibration
    is on disk (:func:`gives_depth`), where two or more do and the primary is one of them: one camera has nothing to
    fuse with, and a grasp is synthesised in the primary's view.

    Args:
        rigs (Sequence[Any]): The camera section's rigs, ``camera.cameras.rigs``.
        primary_rig_id (str): ``camera.cameras.primary_rig_id``.

    Returns:
        tuple[str, ...]: The fused rig ids; empty where the cell grasps from one view.
    """
    primary = str(primary_rig_id or "")
    fusable = sorted(str(rig.rig_id) for rig in rigs if not _why_not_fused(str(rig.rig_id), rig))
    if len(fusable) < 2 or primary not in fusable:
        return ()
    return (primary, *(rig_id for rig_id in fusable if rig_id != primary))


def gives_depth(rig: Any) -> bool:
    """Whether a rig hands out depth: an RGB-D rig its own, a stereo rig once its stereo calibration is written.

    Read as the camera's own refusal reads it (``stereo_depth.stereo_calibration_refusal``), by duck typing and the
    file's presence, because this module is handed to ``Robot`` and its guard forbids the camera package. A stereo rig
    without it is no camera a cell fuses or was meant to fuse: the base profile's webcam pair is one.

    Args:
        rig (Any): One entry of ``camera.cameras.rigs``.

    Returns:
        bool: ``False`` for a ``webcam_pair`` or ``single_device`` rig whose ``calibration_paths.stereo_map_file`` is
            not a file; ``True`` for every other rig.
    """
    if getattr(rig, "source", None) not in _STEREO_SOURCES:
        return True
    written = getattr(getattr(rig, "calibration_paths", None), "stereo_map_file", None)
    return bool(written) and Path(str(written)).is_file()


@dataclass(frozen=True, slots=True)
class CameraFusionPlan:
    """The cameras a cell fuses into each object's cloud before it grasps, primary first, or what it is missing to.
    Reads config and opens nothing, so it answers at a desk.

    Attributes:
        cameras (tuple[str, ...]): The rig ids whose views are fused, the primary first; empty when anything is missing
            (default: ()).
        missing (tuple[str, ...]): What the tree lacks to fuse, one clause each, in the order a person fixes them; empty
            when it can (default: ()).
        on_the_wrist (tuple[str, ...]): The rigs among ``cameras`` on the wrist, placed by the arm's pose at the shutter
            (default: ()).
        primary_rig_id (str): The camera each grasp is synthesised in, the others fused onto it (default: "").
    """

    #: The rigs whose views are fused, the primary first and then the others in id order, the order the cell opens
    #: them in. Empty when anything is missing: a cell that cannot fuse fuses no camera.
    cameras: tuple[str, ...] = ()
    #: What the tree is missing to fuse, one clause each, in the order a person fixes them. Empty when it can.
    missing: tuple[str, ...] = ()
    #: The rigs among ``cameras`` that ride on the wrist: their frames are placed by the arm's pose at the shutter.
    on_the_wrist: tuple[str, ...] = ()
    #: ``camera.cameras.primary_rig_id``: the camera each grasp is synthesised in, the others fused onto it.
    primary_rig_id: str = ""

    @classmethod
    def from_tree(cls, tree: "LoadedTree") -> "CameraFusionPlan":
        """The fusion plan of a loaded tree.

        Args:
            tree (LoadedTree): A loaded tree, ``load_tree()``.

        Returns:
            CameraFusionPlan: The plan; :meth:`refusal` says why it cannot fuse.

        Raises:
            ConfigError: The tree did not load.
        """
        cameras = tree.app_config.camera.cameras
        return cls.from_config(cameras.rigs, primary_rig_id=cameras.primary_rig_id)

    @classmethod
    def from_config(cls, rigs: Sequence[Any], *, primary_rig_id: str) -> "CameraFusionPlan":
        """The fusion plan of the camera section's rigs.

        Args:
            rigs (Sequence[Any]): The camera section's rigs, ``tree.app_config.camera.cameras.rigs``.
            primary_rig_id (str): ``camera.cameras.primary_rig_id``: the camera each grasp is synthesised in.

        Returns:
            CameraFusionPlan: The plan.
        """
        primary = str(primary_rig_id or "")
        by_id = {str(rig.rig_id): rig for rig in rigs}
        fused = fused_cameras(rigs, primary)
        if fused:
            return cls(cameras=fused, on_the_wrist=tuple(cam_id for cam_id in fused if _on_the_wrist(by_id[cam_id])),
                       primary_rig_id=primary)
        fusable = [rig_id for rig_id, rig in sorted(by_id.items()) if not _why_not_fused(rig_id, rig)]
        missing: list[str] = []
        if len(fusable) < 2:
            missing.append(f"camera.cameras.rigs declares {_named(fusable, 'enabled, calibrated rig', by_id)}")
            missing.extend(why for rig_id, rig in sorted(by_id.items()) if (why := _why_not_fused(rig_id, rig)))
        else:
            missing.append(f"the primary rig {primary!r} is not one of them: "
                           f"{_why_not_fused(primary, by_id.get(primary))}")
        return cls(missing=tuple(missing), primary_rig_id=primary)

    def refusal(self) -> str | None:
        """Why this cell cannot fuse its cameras.

        Returns:
            str | None: One sentence naming what is missing and what a fused cell needs; ``None`` where it can fuse.
        """
        if not self.missing:
            return None
        return f"this cell cannot fuse its cameras before a grasp: {'; '.join(self.missing)}. {_A_FUSED_CELL}"

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """The plan as a person reads it.

        Returns:
            str: ASCII, no trailing newline.
        """
        if self.missing:
            return "no camera fusion: " + "; ".join(self.missing)
        labels = [self._label(cam_id) for cam_id in self.cameras]
        named = ", ".join(labels[:-1]) + " and " + labels[-1] if len(labels) > 1 else "".join(labels)
        return f"fusing {named} into each object's cloud before its grasp is planned"

    def to_dict(self) -> dict[str, Any]:
        """The plan as plain data.

        Returns:
            dict[str, Any]: ``cameras``, ``missing``, ``on_the_wrist``, ``primary_rig_id`` and the refusal.
        """
        return {"cameras": list(self.cameras), "missing": list(self.missing), "on_the_wrist": list(self.on_the_wrist),
                "primary_rig_id": self.primary_rig_id, "refusal": self.refusal()}

    def _label(self, cam_id: str) -> str:
        notes = [note for note, holds in (("primary", cam_id == self.primary_rig_id),
                                          ("on the wrist", cam_id in self.on_the_wrist)) if holds]
        return f"{cam_id!r}" + (f" ({', '.join(notes)})" if notes else "")


def _on_the_wrist(rig: Any) -> bool:
    """Whether the arm carries this rig: its calibration says eye in hand, or it declares the body the arm carries."""
    mounting = getattr(getattr(rig, "extrinsics", None), "mounting_mode", None)
    return mounting == "eye_in_hand" or getattr(rig, "body", None) is not None


def _why_not_fused(cam_id: str, rig: Any) -> str:
    """Why a rig gives no fused view, or ``""`` when it does: an enabled rig that declares its calibration and gives
    depth, a stereo rig through its stereo calibration."""
    if rig is None:
        return f"no rig {cam_id!r} in camera.cameras.rigs"
    if not bool(getattr(rig, "enabled", False)):
        return f"rig {cam_id!r} is switched off"
    if getattr(rig, "extrinsics", None) is None:
        return f"rig {cam_id!r} declares no calibration, camera.cameras.rigs[{cam_id!r}].extrinsics"
    if not gives_depth(rig):
        return (f"rig {cam_id!r} gives no depth: its stereo calibration, camera.cameras.rigs[{cam_id!r}]."
                "calibration_paths.stereo_map_file, is not on disk")
    return ""


def _named(ids: Sequence[str], noun: str, rigs: dict[str, Any]) -> str:
    """``no camera``, ``one camera, 'a'`` or ``2 cameras, 'a', 'b'``, each rig on the wrist said to be."""
    names = [f"{cam_id!r}" + (" (on the wrist)" if _on_the_wrist(rigs.get(cam_id)) else "") for cam_id in ids]
    if not names:
        return f"no {noun}"
    if len(names) == 1:
        return f"one {noun}, {names[0]}"
    return f"{len(names)} {noun}s, " + ", ".join(names)
