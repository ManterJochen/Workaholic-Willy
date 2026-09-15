"""Keys that left the configuration schema, and what replaced each.

A tree that still writes one of these keys is refused at load by ``extra='forbid'``, which names
the file, the line and the layer (``loader._describe_validation_error``). This table adds the
sentence that sends the reader to what replaced the key. Only keys removed on purpose belong here;
a typo gets the nearest-key suggestion instead. A ``*`` segment in a key stands for one map key,
such as a camera id.
"""

from __future__ import annotations

from typing import Final

__all__ = ["REMOVED_KEYS"]

#: The dotted key as the whole tree names it, to the sentence that replaces it.
REMOVED_KEYS: Final[dict[str, str]] = {
    "robot.sim.gripper_mount": (
        "the sim mount is derived from the hand, robot.gripper.model, and the arm asset, "
        "robot.sim.robot_model. Name the hand in robot.gripper.model and delete this key."
    ),
    "robot.sim.gripper_variant": (
        "the baked USD variant is derived from the hand, robot.gripper.model, and the arm asset, "
        "robot.sim.robot_model. Name the hand in robot.gripper.model and delete this key."
    ),
    "robot.grasping.deep_generator.gripper": (
        "the deep generator conditions on the cell's hand, robot.gripper.model, which the "
        "calculator factory checks against the artifact's trained hands at build. Name the hand in "
        "robot.gripper.model and delete this key."
    ),
    "robot.safety.self_collision.collision_mesh_variant": (
        "the guard's mesh bundle is derived from the hand, robot.gripper.model, which is the only "
        "name a hand has. Name the hand in robot.gripper.model and delete this key."
    ),
    "robot.safety.self_collision.coupling_mm": (
        "the guard's coupling is the sum of robot.gripper.coupling_plates_mm, the plates between "
        "the flange and the hand's own mounting face. Write the plates there and delete this key."
    ),
    "robot.safety.planning_world.perceived.self_radius_mm": (
        "the self filter fits one capsule per link to the committed arm bundle and takes the "
        "hand's sphere map, padded by perceived.margin_mm. Set the padding there and delete this "
        "key."
    ),
    "robot.safety.planning_world.perceived.tool_radius_mm": (
        "the hand in the self filter is its sphere map and a carried part a capsule sized from "
        "planning_world.payload, padded by perceived.margin_mm. Set the padding there and delete "
        "this key."
    ),
    "robot.grasping.fusion.extrinsics_artifact_path": (
        "a camera's calibration is declared on its rig, camera.cameras.rigs[<id>].extrinsics, with "
        "its mounting_mode and artifact_path, and the primary's resolver is built from there. Move "
        "the path there and delete this key."
    ),
    "robot.grasping.fusion.cameras.*.mounting_mode": (
        "the mounting is declared on the rig, camera.cameras.rigs[<id>].extrinsics.mounting_mode. "
        "Move it there; an entry in fusion.cameras keeps enabled only."
    ),
    "robot.grasping.fusion.cameras.*.extrinsics_artifact_path": (
        "the artifact is declared on the rig, camera.cameras.rigs[<id>].extrinsics.artifact_path. "
        "Move it there; an entry in fusion.cameras keeps enabled only."
    ),
}
