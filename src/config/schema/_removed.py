"""Keys that left the configuration schema, and what replaced each.

A tree that still writes one of these keys is refused at load by ``extra='forbid'``, which names
the file, the line and the layer (``loader._describe_validation_error``). This table adds the
sentence that sends the reader to what replaced the key. Only keys removed on purpose belong here;
a typo gets the nearest-key suggestion instead. A ``*`` segment in a key stands for one map key,
such as a camera id. :func:`removed_key_sentence` is the one lookup: the loader's message, a preset's
validation (``src.robot.grasping.replay.presets``) and ``python -m src.config explain`` all ask it.

A grasp mode is a value, not a key, so ``extra='forbid'`` never sees one that left.
:data:`REMOVED_GRASP_MODES` holds those, each with the sentence that names the mode to use
instead: the schema refuses a tree or a preset that still names one (``default_mode`` and every
mode list under ``robot.grasping``), and ``resolve_grasp_mode`` refuses a caller that passes one.
A recovery action is a value too: :data:`REMOVED_RECOVERY_ACTIONS` holds the one that left, with the
sentence that names the action to use instead, and the schema refuses a ``recovery.allowed_actions`` or
a ``recovery.per_action_budget`` that still names it.
"""

from __future__ import annotations

from typing import Final

__all__ = ["REMOVED_GRASP_MODES", "REMOVED_KEYS", "REMOVED_RECOVERY_ACTIONS", "removed_key_sentence"]

#: Said for each key of the multi-view voxel grid, which left with the commit gate that read it.
_VOXEL_GRID_REMOVED: Final[str] = (
    "the multi-view voxel grid this key configured was removed on 2026-09-28 with the commit gate "
    "that read it, because what it held never reached a grasp, a decision or a measured result, so "
    "delete this key (cameras are fused by robot.grasping.fusion.enabled with fusion.geometry)."
)

#: Said for the post-grasp verification block and for each key it held.
_VERIFICATION_REMOVED: Final[str] = (
    "the post-grasp verification stage this block configured was removed on 2026-09-29, because no "
    "pick path ran it once the two-scan refinement left and the gripper's own hold evidence already "
    "decides every close (the execution policy reads is_object_detected and hold_evidence right after "
    "it, which on a Robotiq is its gOBJ register), so delete this block."
)

#: Said for the dense-recovery block and for each key it held.
_DENSE_RECOVERY_REMOVED: Final[str] = (
    "the dense-recovery block was removed on 2026-09-29 because the recovery policy and strategy it "
    "built were stored on the service and consulted by no pick, so delete this block and configure "
    "recovery under robot.grasping.recovery (enabled, allowed_actions, fixture), the recovery loop a "
    "pick actually runs."
)

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
        "the guard's coupling is the sum of the thicknesses in robot.gripper.coupling_plates, the "
        "plates between the flange and the hand's own mounting face: write the plates there and "
        "delete this key."
    ),
    "robot.gripper.coupling_plates_mm": (
        "a plate is one thing with a thickness and, where it was measured across, a cross section, "
        "so it is now robot.gripper.coupling_plates: a list of {name, thickness_mm, "
        "cross_section_mm}. The thicknesses sum to exactly what this list of millimetres used to "
        "say, and a plate that declares cross_section_mm also becomes collision geometry instead of "
        "an empty gap between the flange and the hand."
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
    "robot.grasping.fusion.max_views": _VOXEL_GRID_REMOVED,
    "robot.grasping.fusion.max_view_age_s": _VOXEL_GRID_REMOVED,
    "robot.grasping.fusion.voxel_size_mm": _VOXEL_GRID_REMOVED,
    "robot.grasping.fusion.roi_extent_mm": _VOXEL_GRID_REMOVED,
    "robot.grasping.fusion.max_voxels": _VOXEL_GRID_REMOVED,
    "robot.grasping.fusion.depth_min_mm": _VOXEL_GRID_REMOVED,
    "robot.grasping.fusion.depth_max_mm": _VOXEL_GRID_REMOVED,
    "robot.grasping.fusion.intrinsics_atol": _VOXEL_GRID_REMOVED,
    "robot.grasping.fusion.commit_policy": (
        "the multi-view commit gate was removed on 2026-09-28 with the voxel grid it read, because "
        "that grid never reached a grasp, a decision or a measured result and no gate replaces it, "
        "so delete this block."
    ),
    "robot.grasping.fusion.active_perception_use_fusion": (
        "the viewpoint planner this key gated had nothing to serve once the commit gate left on "
        "2026-09-28, and the planners were removed on 2026-09-29, so delete this key (a second "
        "view comes from a second camera, fused by robot.grasping.fusion.enabled with "
        "fusion.geometry)."
    ),
    "robot.grasping.decision.max_reobservations": (
        "the decision gate no longer moves the camera to look again (MOVE_CAMERA was removed on "
        "2026-09-29 with the viewpoint planners it needed), so there is no re-observation to "
        "budget: delete this key; the gate decides once per pick, on one frame."
    ),
    "robot.grasping.closed_loop": (
        "the two-scan pre-grasp refinement this block configured was removed on 2026-09-29 with "
        "the closed_loop and dense_autonomous modes that ran it, because no shipped config switched "
        "it on and it never ran on a physical arm, so delete this block (a pick grasps what it "
        "perceived; another view comes from a look pose or a second camera)."
    ),
    "robot.grasping.verification.post_lift_vision_check": (
        "the post-lift vision check was removed on 2026-09-29 with the two-scan refinement whose "
        "target identity it compared against, so delete this key (the execution policy's own "
        "post-close hold check still reads the gripper)."
    ),
    "robot.grasping.verification.vision_displacement_iou_max": (
        "the threshold of the post-lift vision check, which was removed on 2026-09-29 with the "
        "two-scan refinement whose target identity it compared against, so delete this key (the "
        "execution policy's own post-close hold check still reads the gripper)."
    ),
    "robot.grasping.verification": _VERIFICATION_REMOVED,
    "robot.grasping.verification.enabled": _VERIFICATION_REMOVED,
    "robot.grasping.verification.require_object_detected": _VERIFICATION_REMOVED,
    "robot.grasping.verification.width_delta_min_mm": _VERIFICATION_REMOVED,
    "robot.grasping.verification.width_delta_max_mm": _VERIFICATION_REMOVED,
    "robot.grasping.verification.fail_closed": _VERIFICATION_REMOVED,
    "robot.grasping.verification.require_all_conclusive": _VERIFICATION_REMOVED,
    "robot.grasping.dense_recovery": _DENSE_RECOVERY_REMOVED,
    "robot.grasping.dense_recovery.enabled": _DENSE_RECOVERY_REMOVED,
    "robot.grasping.dense_recovery.max_recovery_actions": _DENSE_RECOVERY_REMOVED,
    "robot.grasping.dense_recovery.strategy": _DENSE_RECOVERY_REMOVED,
    "robot.grasping.dense_recovery.allowed_actions": _DENSE_RECOVERY_REMOVED,
}


def removed_key_sentence(dotted: str) -> str | None:
    """The sentence for ``dotted`` when it, or a block that held it, was removed on purpose.

    ``dotted`` is the key as the whole tree names it, ``robot.grasping.closed_loop.enabled`` say, and
    a key inside a removed block answers with the block's sentence. A ``*`` segment in a table key
    matches any one segment. ``None`` for every other key, which then gets the nearest-key suggestion.
    """
    parts = dotted.split(".")
    for end in range(len(parts), 0, -1):
        held = parts[:end]
        exact = REMOVED_KEYS.get(".".join(held))
        if exact is not None:
            return exact
        for key, text in REMOVED_KEYS.items():
            segments = key.split(".")
            if ("*" in segments and len(segments) == len(held)
                    and all(k in ("*", p) for k, p in zip(segments, held))):
                return text
    return None


#: Said for ``closed_loop`` and for the alias it answered to.
_CLOSED_LOOP_MODE_REMOVED: Final[str] = (
    "the closed_loop grasp mode was removed on 2026-09-29 with the two-scan pre-grasp refinement it "
    "ran, which no shipped config switched on and no physical arm ever ran, so name auto instead: "
    "the same sampler, grasping what it perceived without the second look."
)

#: Said for ``dense_autonomous`` and for the alias it answered to.
_DENSE_AUTONOMOUS_MODE_REMOVED: Final[str] = (
    "the dense_autonomous grasp mode was removed on 2026-09-29 with the two-scan pre-grasp "
    "refinement it ran, which no shipped config switched on and no physical arm ever ran, so name "
    "dense_clutter instead: the same dense sampler, which now also allows the nudge_target recovery "
    "that only dense_autonomous allowed."
)

#: A grasp mode that left the runtime, by every spelling it was accepted in, to the sentence that
#: names the mode to use instead.
REMOVED_GRASP_MODES: Final[dict[str, str]] = {
    "closed_loop": _CLOSED_LOOP_MODE_REMOVED,
    "closedloop": _CLOSED_LOOP_MODE_REMOVED,
    "dense_autonomous": _DENSE_AUTONOMOUS_MODE_REMOVED,
    "autonomous": _DENSE_AUTONOMOUS_MODE_REMOVED,
}

#: A recovery action that left the runtime, to the sentence that names the action to use instead. The
#: string stays readable in a record logged before it left (the replay layer and the RL token map keep
#: it); only a config or a preset that asks for it again is refused.
REMOVED_RECOVERY_ACTIONS: Final[dict[str, str]] = {
    "next_viewpoint": (
        "use rescan, which it was merged into on 2026-09-29: with the viewpoint planners gone nothing "
        "moved the camera for it, so it re-perceived from where the camera stood exactly as rescan does."
    ),
}
