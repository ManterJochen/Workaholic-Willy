"""The tool on an arm's flange as one value: the hand its guard and planner model, its tool frame and its payload.

P0i i5 (plan 3.2, owner decision 5, 2026-09-26). The hand was fixed in four places: the self collision guard resolved
it from ``robot.gripper.model`` when the preflight was built and built its exact-mesh backend from it at the first judged
path, the planner captured the same hand when its client was first built, and the connect read the tool frame and the
payload from the tree. One factory now reads all three from the tree (``MountedTool.from_config``), a planned arm is
built with one, and it reads its tool frame and its payload from nothing else; the guard's and the planner's hand follow
the hand it was built with. ``PlannedArm.mount_tool`` is the one path that changes it, and a change while the arm is
connected is refused: a tool change is not built yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.contracts import Maybe, chosen

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

    from src.config.schema.robot import RobotConfig
    from src.config.schema.robot.safety_schema import PayloadSafetyConfig
    from src.config.schema.robot.tool_frame_schema import ToolFrameConfig

    from .hand import PlannerHand

__all__ = ["MountedTool", "ToolChangeRefused"]


class ToolChangeRefused(RuntimeError):
    """A mount that did not happen, with nothing changed: the arm is connected, and a tool change is not built yet
    (owner, 2026-09-26, decision 5), or a guard of the arm could not judge with the tool mounted (P0i i5 review)."""


@dataclass(frozen=True)
class MountedTool:
    """The hand on the flange, where the controller's tool frame puts its grasp centre, and what it weighs."""

    #: The hand the self collision guard and the planner model (``planner_hand``), ``UNSET`` for a cell that names none
    #: and reads no hand geometry.
    hand: "Maybe[PlannerHand]"
    #: ``robot.gripper.tool_frame``: where the grasp centre sits on the flange, and who composes it.
    tool_frame: "ToolFrameConfig"
    #: ``robot.safety.payload``: the mass and centre of gravity the controller is told at the connect.
    payload: "PayloadSafetyConfig"

    def __post_init__(self) -> None:
        """A hand is placed on the flange by a tool frame, so a tool whose hand another frame placed is not one tool:
        the guard would model the hand along one approach while the arm composes its grasp centre along another."""
        if not chosen(self.hand):
            return
        from .hand import placement_of

        if placement_of(self.tool_frame) != (self.hand.placement, self.hand.placement_refusal):
            raise ValueError(
                f"{self.hand.model} is placed by another tool frame than the one this tool carries "
                f"({self.tool_frame.source}, rotation {tuple(float(v) for v in self.tool_frame.rotation_quat_xyzw)}): "
                "resolve the hand from the frame it hangs on, MountedTool.from_config")

    @classmethod
    def from_config(cls, robot_cfg: "RobotConfig", *, data_dir: "str | Path | None" = None) -> "MountedTool":
        """The tool a robot section describes. The hand is resolved as the guard resolves it and refused as it refuses
        it (``planner_hand``: a hand the registry does not know, a tree that describes it differently)."""
        from .hand import planner_hand

        return cls(hand=planner_hand(robot_cfg, data_dir=data_dir), tool_frame=robot_cfg.gripper.tool_frame,
                   payload=robot_cfg.safety.payload)

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """Describe this to a person, as one line of ASCII, no trailing newline."""
        hand = self.hand.model if chosen(self.hand) else "no hand named"
        tf = self.tool_frame
        payload = f"{self.payload.mass_kg:g} kg at {tuple(round(float(v), 1) for v in self.payload.cog_mm)} mm"
        return f"tool  {hand}, tool frame {tf.source} {tuple(float(v) for v in tf.offset_mm)} mm, {payload}"

    def to_dict(self) -> dict[str, Any]:
        """Plain data, ``json.dumps`` safe: the same readings ``render()`` prints."""
        return {
            "hand": self.hand.model if chosen(self.hand) else None,
            "tool_frame": {"source": self.tool_frame.source,
                           "offset_mm": [float(v) for v in self.tool_frame.offset_mm],
                           "rotation_quat_xyzw": [float(v) for v in self.tool_frame.rotation_quat_xyzw]},
            "payload": {"mass_kg": float(self.payload.mass_kg), "cog_mm": [float(v) for v in self.payload.cog_mm]},
        }
