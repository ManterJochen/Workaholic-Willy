"""Which cameras a cell fuses into each object's cloud before it grasps, or what the tree is missing to fuse any.

One depth view sees one side of a part, and a parallel jaw closes on two opposite faces. A cell with two or more
calibrated cameras fuses each object's surface across them into one cloud in BASE and plans the grasp on that
(:mod:`src.robot.grasping.multiview`, "geometry fusion"). Measured in simulation on the datagen reference: the best
ranked grasp is graspable for 43.5 % of objects from one view and 55.9 % fused. Never run on hardware: no physical
multi-camera cell has been built with this code.

Whether a tree can do that is spread over two sections and two switches, and a cell that cannot does not refuse: it
grasps from one view, with a warning in its log. :class:`CameraFusionPlan` reads the tree, opens nothing, and names
every piece that is missing in one sentence, so a program asks before a camera opens or a model loads:

* ``robot.grasping.fusion.geometry.enabled``, the switch that hands the fused cloud to the grasp generator;
* ``robot.grasping.fusion.enabled``. It arms the shadow voxel grid, which changes no grasp, and it is also the switch
  the other cameras' CAMERA to BASE are built under (``build_config_frame_resolvers`` returns none while it is off),
  so with it off every other camera's view is dropped at the pick. The schema calls the two switches independent; for
  a fused cell they are not;
* two or more cameras in ``robot.grasping.fusion.cameras``, keyed by rig id, the primary among them;
* each of them an enabled RGB-D rig in ``camera.cameras.rigs`` that declares its calibration (``extrinsics``).

A camera on the wrist is one of the views, not a refusal: the builders count an eye-in-hand rig, its resolver places
each of its frames by the arm's pose at the shutter (``EyeInHandFrameResolver``), and the cell stamps its frames as it
stamps the primary's. What is measured in simulation is a wrist primary with fixed cameras fused onto it
(``run_multiview_pick``); the shipped reference is two fixed cameras, ``config/robot/robot.eth2.yaml`` with
``config/camera/cam.eth2.yaml``.

Beside :class:`~src.robot.execution.camera_world_wiring.CameraWorldPlan`, which answers the other question a
multi-camera cell raises: which rigs feed the live planner world.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import RobotConfig
    from src.config.tree import LoadedTree

__all__ = ["CameraFusionPlan"]

#: What a fused cell needs, said after what this one is missing, with where the reference cell is written down.
_A_FUSED_CELL = (
    "A fused cell lists two or more enabled RGB-D rigs that declare their calibration under "
    "robot.grasping.fusion.cameras, by rig id and the primary among them, with robot.grasping.fusion.enabled and "
    "robot.grasping.fusion.geometry.enabled on: the reference is config/robot/robot.eth2.yaml with "
    "config/camera/cam.eth2.yaml, loaded as the profile chain <your arm>,eth2"
)


@dataclass(frozen=True, slots=True)
class CameraFusionPlan:
    """The cameras a cell fuses into each object's cloud before it grasps, primary first, or what it is missing.

    Pure: it reads config and opens nothing, so it answers at a desk as well as at the cell.
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
        """The plan of a loaded tree. A tree that did not load raises its own refusal (``ConfigError``)."""
        cameras = tree.app_config.camera.cameras
        return cls.from_config(tree.robot, cameras.rigs, primary_rig_id=cameras.primary_rig_id)

    @classmethod
    def from_config(
        cls, robot_cfg: "RobotConfig", rigs: Sequence[Any], *, primary_rig_id: str,
    ) -> "CameraFusionPlan":
        """The plan of a robot section and the camera section's rigs, as :class:`CameraWorldPlan` takes them."""
        fusion = getattr(getattr(robot_cfg, "grasping", None), "fusion", None)
        geometry = getattr(fusion, "geometry", None)
        primary = str(primary_rig_id or "")
        by_id = {str(rig.rig_id): rig for rig in rigs}
        listed = [str(cam_id) for cam_id, cam in (getattr(fusion, "cameras", None) or {}).items()
                  if bool(getattr(cam, "enabled", True))]

        missing: list[str] = []
        switches = [key for key, on in (("robot.grasping.fusion.enabled", getattr(fusion, "enabled", False)),
                                        ("robot.grasping.fusion.geometry.enabled", getattr(geometry, "enabled", False)))
                    if not bool(on)]
        if len(switches) == 2:
            missing.append(f"{switches[0]} and {switches[1]} are false")
        elif switches == ["robot.grasping.fusion.geometry.enabled"]:
            missing.append(f"{switches[0]} is false, so no second view reaches the grasp generator")
        elif switches:
            missing.append(f"{switches[0]} is false, and the other cameras' CAMERA to BASE are built only while it "
                           "is on")
        if len(listed) < 2:
            declared = [rig_id for rig_id, rig in by_id.items() if _gives_depth(rig)]
            missing.append(f"robot.grasping.fusion.cameras lists {_named(listed, 'camera', by_id)}, and "
                           f"camera.cameras.rigs declares {_named(declared, 'enabled RGB-D rig', by_id)}")
        if listed and primary not in listed:
            missing.append(f"robot.grasping.fusion.cameras leaves out the primary rig {primary!r}")
        for cam_id in listed:
            refused = _why_not_fused(cam_id, by_id.get(cam_id))
            if refused:
                missing.append(refused)

        if missing:
            return cls(missing=tuple(missing), primary_rig_id=primary)
        fused = (primary, *sorted(cam_id for cam_id in listed if cam_id != primary))
        return cls(cameras=fused, on_the_wrist=tuple(cam_id for cam_id in fused if _on_the_wrist(by_id[cam_id])),
                   primary_rig_id=primary)

    def refusal(self) -> str | None:
        """Why this cell cannot fuse, in one sentence naming what is missing and then what a fused cell needs; else
        ``None``."""
        if not self.missing:
            return None
        return f"this cell cannot fuse its cameras before a grasp: {'; '.join(self.missing)}. {_A_FUSED_CELL}"

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        if self.missing:
            return "no camera fusion: " + "; ".join(self.missing)
        labels = [self._label(cam_id) for cam_id in self.cameras]
        named = ", ".join(labels[:-1]) + " and " + labels[-1] if len(labels) > 1 else "".join(labels)
        return f"fusing {named} into each object's cloud before its grasp is planned"

    def to_dict(self) -> dict[str, Any]:
        """The wire view."""
        return {"cameras": list(self.cameras), "missing": list(self.missing), "on_the_wrist": list(self.on_the_wrist),
                "primary_rig_id": self.primary_rig_id, "refusal": self.refusal()}

    def _label(self, cam_id: str) -> str:
        notes = [note for note, holds in (("primary", cam_id == self.primary_rig_id),
                                          ("on the wrist", cam_id in self.on_the_wrist)) if holds]
        return f"{cam_id!r}" + (f" ({', '.join(notes)})" if notes else "")


def _gives_depth(rig: Any) -> bool:
    """An enabled RGB-D rig: the only kind whose frames carry the depth a surface is built from."""
    return bool(getattr(rig, "enabled", False)) and getattr(rig, "source", None) == "rgbd"


def _on_the_wrist(rig: Any) -> bool:
    """Whether the arm carries this rig: its calibration says eye in hand, or it declares the body the arm carries."""
    mounting = getattr(getattr(rig, "extrinsics", None), "mounting_mode", None)
    return mounting == "eye_in_hand" or getattr(rig, "body", None) is not None


def _why_not_fused(cam_id: str, rig: Any) -> str:
    """Why the rig a fusion camera names cannot give a fused view, or ``""`` when it can."""
    if rig is None:
        return f"fusion camera {cam_id!r} names no rig in camera.cameras.rigs"
    if not bool(getattr(rig, "enabled", False)):
        return f"rig {cam_id!r} is switched off"
    source = getattr(rig, "source", None)
    if source != "rgbd":
        return f"rig {cam_id!r} is a {source} rig, with no depth of its own"
    if getattr(rig, "extrinsics", None) is None:
        return f"rig {cam_id!r} declares no calibration, camera.cameras.rigs[{cam_id!r}].extrinsics"
    return ""


def _named(ids: Sequence[str], noun: str, rigs: dict[str, Any]) -> str:
    """``no camera``, ``one camera, 'a'`` or ``2 cameras, 'a', 'b'``, each rig on the wrist said to be."""
    names = [f"{cam_id!r}" + (" (on the wrist)" if _on_the_wrist(rigs.get(cam_id)) else "") for cam_id in ids]
    if not names:
        return f"no {noun}"
    if len(names) == 1:
        return f"one {noun}, {names[0]}"
    return f"{len(names)} {noun}s, " + ", ".join(names)
