"""Keys that left the configuration schema, and what replaced each.

A tree that still writes one of these keys is refused at load by ``extra='forbid'``, which names
the file, the line and the layer (``loader._describe_validation_error``). This table adds the
sentence that sends the reader to what replaced the key. Only keys removed on purpose belong here;
a typo gets the nearest-key suggestion instead. A ``*`` segment in a key stands for one map key,
such as a camera id. :func:`removed_key_sentence` is the one lookup: the loader's message and
``python -m src.config explain`` both ask it.

A grasp mode is a value, not a key, so ``extra='forbid'`` never sees one that left.
:data:`REMOVED_GRASP_MODES` holds those, each with the sentence that names the mode to use
instead: the schema refuses a tree that still names one (``default_mode`` and every mode list
under ``robot.grasping``), and ``resolve_grasp_mode`` refuses a caller that passes one.
A recovery action is a value too: :data:`REMOVED_RECOVERY_ACTIONS` holds those that left, with the
sentence that says what to do instead, and a campaign that still asks for one (``PickRun``'s
``recovery_actions``) is refused with it.
"""

from __future__ import annotations

from typing import Final

__all__ = ["REMOVED_GRASP_MODES", "REMOVED_KEYS", "REMOVED_RECOVERY_ACTIONS", "removed_key_sentence"]

#: Said for each key of the multi-view voxel grid, which left with the commit gate that read it.
_VOXEL_GRID_REMOVED: Final[str] = (
    "the multi-view voxel grid this key configured was removed on 2026-09-28 with the commit gate "
    "that read it, because what it held never reached a grasp, a decision or a measured result, so "
    "delete this key (a cell fuses its calibrated cameras by itself since 2026-10-16)."
)

#: Said for the fusion block and every key it held.
_FUSION_REMOVED: Final[str] = (
    "the fusion block was removed on 2026-10-16 (the owner's cleanup): a cell fuses its cameras by itself wherever "
    "camera.cameras.rigs declares two or more enabled RGB-D rigs with their calibration, at the numbers every fused "
    "cell ran, so delete this block (a camera that is not to take part is switched off on its rig)."
)

#: Said for each key of the support-footprint stage's own search, which no pick asked once the part's body was
#: grasped alone (2026-10-14).
_SFE_SEARCH_REMOVED: Final[str] = (
    "this key sped up the support-footprint stage's own search (its worker processes, its batched builds, its fine "
    "search that waited), which no pick asks since 2026-10-14, when the part's own body became the only grasps the "
    "stage offers, so it reached nothing; it left on 2026-10-16 with that code (the owner's cleanup), so delete this "
    "key."
)

#: Said for the geometry block and for each key it held: which stage proposes grasps, and the numbers it planned with.
_GEOMETRY_REMOVED: Final[str] = (
    "the geometry block was removed on 2026-10-16 (the owner's cleanup): the part's own body over its support is the "
    "only stage that proposes grasps, at the numbers every cell ran (a 2 mm rim off the mask's edge, the faces given "
    "back by the part's visual hull, none pushed out), and the silhouette stage and its depth levers left with the "
    "block, so delete this block."
)

#: Said for each key of the support block that restated a number every cell ran with (the bin stays a key).
_SUPPORT_FIXED: Final[str] = (
    "the support block keeps only the bin since 2026-10-17 (the owner's cleanup): the parts stand on the planner "
    "world's bench, the top of robot.safety.planning_world.support_plane (0 mm where the tree declares none), level, "
    "raised by a part's own lowest point wherever its cloud saw down its side, and the gripper keeps 1 mm over it, "
    "the guard's finger floor, as every cell ran. Delete this key; where height_mm said another table than the "
    "bench, set the bench's height_mm to it (a bin's floor stays robot.grasping.support.container.floor_height_mm)."
)

#: Said for each grasping switch fixed at the value the owner's cell ran, by key: where the behaviour now lives.
_GRASPING_FIXED: Final[dict[str, str]] = {
    "max_attempts": "a pick makes up to 5 attempts, the cell's (MAX_ATTEMPTS in autonomous_grasp/config.py; a service "
                    "built in code may still ask fewer with from_robot_config(max_attempts=...))",
    "scene_obstacles": "the calculator always sees the parts beside the one it grasps as obstacles",
    "side_approaches": "tilted approaches always join the vertical ones, ranked by the room they keep",
    "inner_grasps": "a Robotiq hand always searches a part's bores where no outer grasp holds",
    "first_good_part": "a pick always takes the first part whose grasp scores 0.6 without computing the rest",
    "good_part_score": "the score a part needs to be taken at once is 0.6 (GOOD_PART_SCORE in loop/pick_loop.py)",
    "colour_check": "a part whose pixels contradict the colour its phrase names is always no target",
    "colour_check_clipped": "the colour check always leaves out a pixel the camera clipped in some of its channels",
    "name_neighbours": "the parts beside the named ones are always named, read off the depth",
    "weak_look_trigger": "a weak view of the part always sends a wrist pick on to its next look",
    "hide_own_places": "a task's own bin and drops are always painted out of the image its detector reads",
    "follow_parts": "a task always follows its parts from pick to pick, within 10 mm a pick and 20 mm since its "
                    "grounding, and grounds them again on every trigger",
}

#: Said for each motion switch fixed at the value the owner's cell ran, by its dotted key: where the behaviour now lives.
_MOTION_FIXED: Final[dict[str, str]] = {
    "robot.motion.judge_next_leg": "every arm planned with cuRobo judges the next leg of a pick or a place while the "
                                   "jaws' stroke is waited out, and during the line before it where a halt brakes "
                                   "that line (every UR)",
    "robot.ur.brake_on_halt": "a halt always brakes the move in flight under control (stopJ or stopL), by the thread "
                              "that sent it, and the move answers True only once the arm stands at its target",
}

#: Said for the steady gate's block and for each key it held.
_DWELL_FIXED: Final[str] = (
    "the steady gate left the config on 2026-10-17 at the values the owner's cell ran (the owner's cleanup): every "
    "motion waits for the arm to stand still, at most 5 s (STEADY_TIMEOUT_S in robot/core/robot_arm.py), right before "
    "it is sent on an arm planned with cuRobo and at the start of its verb elsewhere, read off the joint speeds "
    "(isSteady where they cannot be read); the Isaac sim, whose wait steps its own physics, waits for nothing. Delete "
    "this block."
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
    "built were stored on the service and consulted by no pick, so delete this block; the recovery a "
    "pick runs is fixed since 2026-10-16 at the values the owner's cell ran."
)

#: Said for each grasping block that left on 2026-10-16 with its code (the owner's cleanup), measured first on the
#: replay bank of the owner's cell: 479 picks, each judged on the tree the cell ran it with.
_DECISION_REMOVED: Final[str] = (
    "the decision layer this block configured was removed on 2026-10-16 with its code, because switched on it "
    "changed none of the 479 picks of the owner's cell's replay bank and the exact guard judges every grasp before "
    "the arm moves, so delete this block."
)
_FEASIBILITY_REMOVED: Final[str] = (
    "the reachability ranking this block configured was removed on 2026-10-16 with its code, because its default "
    "weight of 0 made it inert, at 0.5 it changed none of the 479 picks of the owner's cell's replay bank, and its "
    "IK-quality signal took a measured UR5e overhead pick from 10 of 10 to 0 of 10, so delete this block."
)
_OCCLUSION_REMOVED: Final[str] = (
    "the corridor analysis this block configured was removed on 2026-10-16 with its code, because switched on it "
    "changed none of the 479 picks of the owner's cell's replay bank and its hard reject was never wired, so delete "
    "this block."
)
_ORDERING_REMOVED: Final[str] = (
    "the clearing order this block configured was removed on 2026-10-16 with its code, because it never acted once "
    "a task names its target, which every task does, so delete this block."
)
_UNCERTAINTY_REMOVED: Final[str] = (
    "the fused-uncertainty layer this block configured was removed on 2026-10-16 with its code, because switched "
    "on it changed none of the 479 picks of the owner's cell's replay bank, so delete this block."
)
_SUCCESS_MODEL_REMOVED: Final[str] = (
    "the learned success estimate this block configured was removed on 2026-10-16 with its code and its artifact, "
    "because in the shadow it changed nothing and it never passed the gate that would have let it rank, so delete "
    "this block."
)
_WATCHDOG_REMOVED: Final[str] = (
    "the drift and out-of-distribution watchdog this block configured was removed on 2026-10-16 with its code, "
    "because it ran in the shadow only and wrote telemetry nothing read, so delete this block."
)
_PERFORMANCE_REMOVED: Final[str] = (
    "the per-stage latency budgets this block configured were removed on 2026-10-16 with their code, because they "
    "changed no grasp and no motion and a pick's wall time stays in its record (attempt_wall_time_s), so delete "
    "this block."
)
_APPROACH_VALIDATION_REMOVED: Final[str] = (
    "the swept-volume check of the approach this block configured was removed on 2026-10-16 with its code, because "
    "switched on it changed none of the 479 picks of the owner's cell's replay bank and the exact guard judges "
    "every sample of every motion anyway, so delete this block."
)
_RL_REMOVED: Final[str] = (
    "the reinforcement-learning layer this block configured was removed on 2026-10-16 with its code, because the "
    "run time refused every mode that would have let it act and its shadow only wrote telemetry, so a cell always "
    "ran the deterministic pipeline it runs now; delete this block."
)

#: Said for the recovery block and for each key it held.
_RECOVERY_REMOVED: Final[str] = (
    "the recovery a pick runs is fixed since 2026-10-16 at the values the owner's cell ran (the owner's cleanup): "
    "always on, a rescan once and a push twice, at most two actions a pick, three tries at a blocker's grasps and the "
    "blocker into the place, a push of 30 mm and 50 mm at most, while container_agitate, its fixture box and "
    "next_target left with their code; whether the parts are critical is now runtime.task.critical_parts in "
    "app/runtime.yaml, and a task may still say otherwise for itself, so delete this block."
)

#: Said for the stereo matcher block, camera/stereomatcher.yaml, and for every key it held.
_STEREOMATCHER_REMOVED: Final[str] = (
    "the stereo matcher block was removed on 2026-10-17 with its file, camera/stereomatcher.yaml (the owner's "
    "decision): a stereo rig's disparity search is derived from its own stereo calibration, its focal length and "
    "baseline, and from where the robot's workspace stands before it, and every other setting of the matcher is fixed "
    "at values measured on the D415's infrared pair, with no WLS fill and no temporal smoothing, because the guard "
    "judges every motion against this depth and a filled hole can hide an obstacle; delete the file."
)

#: Said for GroundingDINO's block and for the keys that chose it: the phrase grounder left with the router and the
#: fallback to it (the owner's cleanup, 2026-10-17).
_GROUNDING_DINO_REMOVED: Final[str] = (
    "GroundingDINO left on 2026-10-17 (the owner's cleanup) with the prompt router and the fallback to it: the VLM "
    "grounds every free-text prompt (models.pipeline.zero_shot.vlm) and RT-DETR answers with fixed classes "
    "(models.pipeline.kind: closed_set, models.rtdetr), as the owner's cell detected with the VLM alone, so delete "
    "this key."
)

#: Said for OneFormer's block and for every key that chose a segmenter.
_ONEFORMER_REMOVED: Final[str] = (
    "OneFormer left on 2026-10-17 (the owner's cleanup): SAM2 cuts every mask on every stack (models.segmenter), as "
    "on the owner's cell, so delete this key."
)

#: Said for the prompt router's block and every key it held.
_ROUTER_REMOVED: Final[str] = (
    "the prompt router left on 2026-10-17 (the owner's cleanup) with GroundingDINO, the phrase grounder it sent the "
    "plain prompts to: the VLM grounds every prompt, so delete this block."
)

#: Said for each key of a model block that said where its weights were.
_MODEL_PATH_REMOVED: Final[str] = (
    "a model's weights are read from its model_id since 2026-10-17 (the owner's cleanup): from "
    "<models.weights_root>/<kind>/<org>--<name>, the folder scripts/model_weights/fetch.py writes, under the "
    "repository's assets/models/hf unless models.weights_root names another; a folder of your own (a DetectorTraining "
    "out_dir) is named by its path in model_id. Delete this key, and set models.weights_root once where the weights "
    "live elsewhere."
)

#: Said for `local`, which every model now reads as the owner's cell set it.
_LOCAL_REMOVED: Final[str] = (
    "every model reads this machine alone since 2026-10-17 (the owner's cleanup), as every block of the owner's cell "
    "said with local: true: the fetched folder, else the Hugging Face cache, and nothing is downloaded at run time "
    "(scripts/model_weights/fetch.py fetches), so delete this key."
)

#: Said for a model block's `optim` and for each key it held.
_OPTIM_REMOVED: Final[str] = (
    "how each model loads is fixed per model since 2026-10-17 (the owner's cleanup) at the values the owner's cell "
    "ran (src/models/_inference.py): RT-DETR torch_dtype auto, eager attention, channels last; SAM2 fp32 weights "
    "under fp16 autocast, sdpa attention, channels last; Whisper torch_dtype auto, sdpa attention; torch.compile off "
    "everywhere. Delete this block."
)

#: Said for each key of the VLM block that set how it answers.
_VLM_ANSWERS_FIXED: Final[str] = (
    "how the VLM answers is fixed since 2026-10-17 (the owner's cleanup) at the values the owner's cell ran "
    "(src/models/vlm/qwen.py): it loads at the first prompt that reaches it, ends an answer where its JSON closes, "
    "proposes 10 tokens a pass from its own answer for a grounding and for a command, and decodes with CUDA graphs "
    "on a card of compute capability 8.0 or newer (decode_graphs auto). Delete this key."
)

#: Said for `on_unavailable`, whose degrade fell back to GroundingDINO.
_ON_UNAVAILABLE_REMOVED: Final[str] = (
    "a cell without its VLM refuses the pick since 2026-10-17 (the owner's cleanup), as the owner's cell did with "
    "on_unavailable: refuse; degrade fell back to GroundingDINO, which left, so delete this key."
)

#: Said for the speech keys that fixed the microphone and the decoder.
_SPEECH_FIXED: Final[str] = (
    "the speech stack runs at the owner's cell's numbers since 2026-10-17 (the owner's cleanup), fixed in "
    "src/models/speech: the microphone opens at 16000 Hz, 8000 frames a block, one channel of int16, every "
    "recording is resampled to 16000 Hz, and Whisper transcribes and never translates. Delete this key; models.stt "
    "keeps model_id and language."
)

#: Said for Silero's path, which the weights catalogue fixes.
_VAD_PATH_REMOVED: Final[str] = (
    "Silero, the voice detector, is read since 2026-10-17 (the owner's cleanup) from where "
    "scripts/model_weights/fetch.py silero-vad writes it, <models.weights_root>/speech/silero-vad-6.2.1/"
    "silero_vad.jit, the path every cell named, so delete this key (and set models.weights_root once where the "
    "weights live elsewhere)."
)

#: Said for the two MediaPipe blocks, which held only the bundle paths and the owner's cell's numbers.
_HAND_FIXED: Final[str] = (
    "the MediaPipe hand and gesture detectors run at the owner's cell's numbers since 2026-10-17 (the owner's "
    "cleanup), fixed in src/models/handdetection/constants.py: two hands at most, a confidence of 0.5 to detect, keep "
    "and track a hand and to report a thumb, the median depth of a 15 px disc about the palm with 20 valid pixels at "
    "least; their bundles are read from <models.weights_root>/mediapipe/, where scripts/model_weights/fetch.py "
    "--mediapipe writes them. Delete this block."
)

def _moved_under_motion(new: str) -> str:
    """Said for a key that moved under ``robot.motion`` (the owner's cleanup, 2026-10-17), naming where it went."""
    return (f"every key of how a cell moves stands under robot.motion since 2026-10-17 (the owner's cleanup): this one "
            f"is now {new}, with the same values. `python -m src.config migrate <tree> --write` moves it, comments and "
            "all.")


#: Said for the home in radians: the home is written in degrees alone, as the pendant shows it.
_HOME_RADIANS_REMOVED: Final[str] = (
    "the home is written in degrees alone since 2026-10-17 (the owner's cleanup), as the pendant shows it: "
    "robot.motion.home_deg, one value per joint. `python -m src.config migrate <tree> --write` converts a list written "
    "on one line; convert a list written over several lines by hand (each joint's radians times 180/pi)."
)

#: Said for the place's five numbers and ``relocate``, fixed at the owner's cell's (B4).
_PLACE_FIXED: Final[str] = (
    "a task sets its parts down at the owner's cell's numbers since 2026-10-17 (the owner's cleanup), fixed in "
    "src/robot/execution/place_target.py: 30 mm under the rim where the jaws open below it, 10 mm inside a box's "
    "opening, 20 mm of air over the rim (a task's own air goes before it), 20 mm between parts side by side and 15 mm "
    "over the rim on a carry; and a bin the check before a drop lost is always looked for again. Delete this key."
)

# ---------------------------------------------------------------------------------------------------------------------
# Safety keys fixed at the cell's values (phase 2, 2026-10-17): each sentence says the value every cell runs since.
# ---------------------------------------------------------------------------------------------------------------------

#: Said for the camera world's block and every key it held.
_PERCEIVED_FIXED: Final[str] = (
    "the camera world is built alike on every cell since 2026-10-17 (the owner's cleanup), at the values the owner's "
    "cell ran: on wherever the planning world is, a frame planned against up to 2000 ms after its shutter, a camera "
    "that cannot vouch for the cell asked 3 more times, every 2nd pixel read, a 10 mm thinning voxel, a 10 mm "
    "distance field for the planner, a 25 mm cluster grid of at least 12 points, every box grown by 6 mm, room for 128 "
    "boxes, every box carried down to the bench, a 5 mm bench band, the supports the parts stand on found on every "
    "build and held 2 mm over their band, and the fingers coming 3 mm past a named part's surface and to 1 mm of what "
    "a support reads; the self filter takes the robot's body out by at least 15 mm. Delete this block."
)

#: Said for ``planning_world.enabled``.
_WORLD_ON_WITH_ITS_BENCH: Final[str] = (
    "the planning world is on wherever the cell declares the bench it stands on since 2026-10-17 (the owner's "
    "cleanup): a tree that declares safety.planning_world.support_plane tells the planner its bench, its meshes and "
    "what its cameras see, and a tree that declares none tells it nothing and its planner keeps the world it booted "
    "with, as an enabled world without a bench was refused before. Delete this key."
)

#: Said for ``planning_world.include_fixtures``.
_FIXTURES_STAY_THE_GUARDS: Final[str] = (
    "the guard's own fixtures stay the guard's since 2026-10-17 (the owner's cleanup), as on the owner's cell: the "
    "planner is told the bench, the meshes and what the cameras see, and the cameras find what the parts stand on. "
    "Delete this key."
)

#: Said for ``planning_world.require_registration``.
_REGISTRATION_REQUIRED: Final[str] = (
    "a world the planner confirms only in part refuses the motion on every cell since 2026-10-17 (the owner's "
    "cleanup), as it did by default: no cell plans on against the planner's boot world. Delete this key."
)

#: Said for ``planning_world.register_in_place``.
_REGISTERED_BY_UPDATE_WORLD: Final[str] = (
    "every world is registered through cuRobo's update_world since 2026-10-17 (the owner's cleanup): the one-call "
    "write of a world of boxes ran only on a planner that holds no distance field, and every cell's camera world "
    "sends one now. Delete this key."
)

#: Said for ``planning_world.hold`` and each switch it held.
_WORLD_HELD: Final[str] = (
    "a pick and a place judge across their standoff, their carry and their drop in the world they already hold, with "
    "no new camera frame, on every cell since 2026-10-17 (the owner's cleanup), as the owner's cell did since "
    "2026-10-09. Delete this block."
)

#: Said for ``planning_world.payload`` and each key it held but ``length_mm``.
_NO_CARRIED_PART_MODELLED: Final[str] = (
    "no carried part is modelled on any cell since 2026-10-17 (the owner's cleanup), as on the owner's cell since "
    "2026-10-05: a closed hand is judged as an empty one, by the exact guard at full stroke, and the planner reserves "
    "no spheres for a part. How far the parts hang past the fingertips is robot.motion.place.part_hang_mm; delete this "
    "block."
)

#: Said for ``planning_world.payload.length_mm``.
_PART_HANG_MOVED: Final[str] = (
    "how far the longest part a cell carries hangs past the fingertips is robot.motion.place.part_hang_mm since 2026-10-17 "
    "(the owner's cleanup), with the same range, null still hanging the part from the TCP, and the console writes it "
    "there: put the value there and delete this block."
)

#: Said for ``planned_motion`` and every key it held.
_PLANNED_MOTION_FIXED: Final[str] = (
    "how a cuRobo cell chooses and judges the path of a move is alike on every cell since 2026-10-17 (the owner's "
    "cleanup), at the values the owner's cell ran: a straight joint line keeps 3 mm from the planner's world before it "
    "runs instead of a plan, no joint of a plan swings more than 45 degrees past its span, the exact mesh guard alone "
    "judges a line and a goal wherever it holds everything the planner would judge them against (mesh first), a joint "
    "plan runs no time-optimal pass after its first, and cuRobo plans on the lean sphere model wherever the arm and "
    "the hand have one. Delete this block."
)

#: Said for ``self_collision.kinematics_base_yaw_deg``.
_BASE_YAW_IN_CODE: Final[str] = (
    "where the arm's DH base stands from the frame poses and fixtures are written in is the arm's own fact since "
    "2026-10-17 (the owner's cleanup): a real arm's controller reports in its DH base, so no turn, as on the owner's "
    "cell, and the Isaac arm's half turn about z is stated in code with its harness (src/willy_sim/guard_numbers.py). "
    "Delete this key."
)

#: Said for the capsule proxy's tool keys.
_CAPSULE_TOOL_FROM_THE_HAND: Final[str] = (
    "the capsule proxy takes the hand from its registry file since 2026-10-17 (the owner's cleanup), "
    "config/grippers/<robot.gripper.model>.yaml (the 2F-85 every arm bundle carries where a guard is built with none): "
    "its fingers as one capsule along the grasp's closing axis at the TCP, as the owner's cell modelled them "
    "(tool_model finger), as long as the open jaw is across its fingers' outer faces and as thick as a finger is "
    "wide. The exact mesh guard, every cell's, never read these keys. Delete this key."
)

#: Said for the capsule proxy's base keys.
_CAPSULE_BASE_FROM_THE_ARM: Final[str] = (
    "the capsule proxy takes the robot's base from the arm's description since 2026-10-17 (the owner's cleanup): a "
    "column from the mounting face up to the shoulder joint config/arms/<model>.yaml places, 80 mm round as every cell "
    "ran, and none for an arm nobody describes. The exact mesh guard, every cell's, never read these keys. Delete this "
    "key."
)

#: Said for ``self_collision.link_radii_mm``.
_CAPSULE_LINKS_FIXED: Final[str] = (
    "every arm link of the capsule proxy is 60 mm round on every cell since 2026-10-17 (the owner's cleanup), as on the "
    "owner's cell, which named no radii. The exact mesh guard, every cell's, never read this key. Delete it."
)

#: Said for ``self_collision.whole_path_judge``.
_WHOLE_PATH_JUDGED: Final[str] = (
    "every path is judged whole first on every cell since 2026-10-17 (the owner's cleanup), as on the owner's cell: "
    "each guard finds over the whole path at once the first sample it might refuse, and every sample from it on is "
    "judged one at a time, with the verdict, the sample it names and its message those of judging every sample in "
    "turn. Delete this key."
)

#: Said for ``safety.limits``.
_WORKSPACE_MARGIN_FIXED: Final[str] = (
    "every cell keeps a commanded pose 20 mm inside every face of robot.motion.workspace since 2026-10-17 (the owner's "
    "cleanup), as the owner's cell did (src/robot/safety/workspace.py, WORKSPACE_MARGIN_MM). Delete this block."
)

#: Said for ``safety.joint_limits``.
_JOINT_LIMITS_FIXED: Final[str] = (
    "every arm's joint limits are its own since 2026-10-17 (the owner's cleanup), as on the owner's cell: the envelope "
    "its description states (config/arms/<model>.yaml), 5 degrees kept inside each end, and on every real arm every "
    "full-turn joint within half a turn of home. The Isaac arm states in code that it keeps its whole envelope "
    "(src/willy_sim/guard_numbers.py). Delete this block."
)

#: Said for ``safety.ik_quality``.
_IK_QUALITY_FIXED: Final[str] = (
    "every real arm's IK solutions are judged alike since 2026-10-17 (the owner's cleanup), at the owner's cell's "
    "numbers: no joint jumps more than 1.0 rad, the Jacobian keeps a smallest singular value of 0.005 and a condition "
    "number of at most 250, no joint comes within 2 degrees of its limit, a line a cuRobo UR judges is solved on the "
    "controller's own kinematics (the controller asked where those cannot vouch for it), and the singularity check "
    "before a move differentiates the arm's own DH chain (the controller's FK where the chain cannot be read). The "
    "Isaac arm states its own numbers in code (src/willy_sim/guard_numbers.py). Delete this block."
)

#: Said for ``safety.motion_continuity``.
_CONTINUITY_FIXED: Final[str] = (
    "two commanded targets in a row may lie at most 45 degrees of any joint, 30 degrees of the TCP's turn and 250 mm "
    "apart on every real arm since 2026-10-17 (the owner's cleanup), the owner's cell's numbers; the Isaac arm states "
    "its own in code (src/willy_sim/guard_numbers.py). Delete this block."
)

#: Said for ``self_collision.enforce``.
_SELF_COLLISION_ALWAYS: Final[str] = (
    "the self collision guard runs on every cell since 2026-10-17 (the owner's cleanup), as on the owner's cell. "
    "Delete this key."
)

#: Said for ``payload.enforce``.
_PAYLOAD_ALWAYS: Final[str] = (
    "the payload guard runs, and a real arm pushes mass_kg and cog_mm to its controller at connect, on every cell "
    "since 2026-10-17 (the owner's cleanup), as on the owner's cell: a mass of 0, the shipped 'not weighed yet' "
    "marker, is refused at connect, so weigh what the flange carries. Delete this key."
)

#: The dotted key as the whole tree names it, to the sentence that replaces it.
REMOVED_KEYS: Final[dict[str, str]] = {
    "camera.stereomatcher": _STEREOMATCHER_REMOVED,
    "robot.grasping.recovery": _RECOVERY_REMOVED,
    "robot.grasping.fusion": _FUSION_REMOVED,
    "robot.grasping.decision": _DECISION_REMOVED,
    "robot.grasping.feasibility": _FEASIBILITY_REMOVED,
    "robot.grasping.occlusion": _OCCLUSION_REMOVED,
    "robot.grasping.ordering": _ORDERING_REMOVED,
    "robot.grasping.uncertainty": _UNCERTAINTY_REMOVED,
    "robot.grasping.success_model": _SUCCESS_MODEL_REMOVED,
    "robot.grasping.watchdog": _WATCHDOG_REMOVED,
    "robot.grasping.performance": _PERFORMANCE_REMOVED,
    "robot.grasping.approach_validation": _APPROACH_VALIDATION_REMOVED,
    "robot.rl": _RL_REMOVED,
    "robot.ur.motion_planner": (
        "how a cell plans its motions is the cell's key rather than one vendor's: it is now robot.motion.planner, with "
        "the same values 'ik' and 'curobo' (2026-10-13). Refused rather than ignored, because a cell that sets 'ik' on "
        "purpose on a box with no cuRobo would otherwise fall back to the 'curobo' default and refuse every motion it "
        "was asked for. Move the key to robot.motion.planner."
    ),
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
    "robot.grasping.fusion.extrinsics_artifact_path": (
        "a camera's calibration is declared on its rig, camera.cameras.rigs[<id>].extrinsics, with "
        "its mounting_mode and artifact_path, and the primary's resolver is built from there. Move "
        "the path there and delete this key."
    ),
    "robot.grasping.fusion.cameras.*.mounting_mode": (
        "the mounting is declared on the rig, camera.cameras.rigs[<id>].extrinsics.mounting_mode. "
        "Move it there; the fusion block itself left on 2026-10-16."
    ),
    "robot.grasping.fusion.cameras.*.extrinsics_artifact_path": (
        "the artifact is declared on the rig, camera.cameras.rigs[<id>].extrinsics.artifact_path. "
        "Move it there; the fusion block itself left on 2026-10-16."
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
        "view comes from a second calibrated camera, which a cell fuses by itself)."
    ),
    "robot.grasping.geometry": _GEOMETRY_REMOVED,
    "robot.grasping.isotropic_radial_closing": (
        "this key turned a near-round silhouette's grasp toward the robot base for the simulated runners (a cell's "
        "calculator never read it); the silhouette stage left on 2026-10-16 (the owner's cleanup), and the part's body "
        "ranks every closing direction it holds, so delete this key."
    ),
    "robot.grasping.workers": _SFE_SEARCH_REMOVED,
    "robot.grasping.batched_builds": _SFE_SEARCH_REMOVED,
    "robot.grasping.fine_pass_waits": _SFE_SEARCH_REMOVED,
    "robot.grasping.gripper_geometry": (
        "the hand's collision envelope is its registry file's since 2026-10-17 (the owner's cleanup): "
        "config/grippers/<robot.gripper.model>.yaml, the 2F-85's (robotiq_2f85.yaml) on a cell that names no hand, "
        "with no margin added, as every cell ran; the block only restated those numbers, so delete it (a hand measured "
        "differently is corrected in its registry file)."
    ),
    **{f"robot.grasping.{key}": (
        f"this switch left on 2026-10-17 at the value the owner's cell ran (the owner's cleanup): {said}, so delete "
        "this key.") for key, said in _GRASPING_FIXED.items()},
    **{key: (
        f"this switch left on 2026-10-17 at the value the owner's cell ran (the owner's cleanup): {said}, so delete "
        "this key.") for key, said in _MOTION_FIXED.items()},
    "robot.safety.dwell": _DWELL_FIXED,
    "robot.safety.dwell.require_steady_before_motion": _DWELL_FIXED,
    "robot.safety.dwell.steady_timeout_s": _DWELL_FIXED,
    "robot.safety.dwell.gate_at": _DWELL_FIXED,
    "robot.safety.dwell.steady_signal": _DWELL_FIXED,
    "robot.grasping.support.height_mm": _SUPPORT_FIXED,
    "robot.grasping.support.normal": _SUPPORT_FIXED,
    "robot.grasping.support.refine_from_target": _SUPPORT_FIXED,
    "robot.grasping.support.min_clearance_mm": _SUPPORT_FIXED,
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
    "robot.safety.self_collision.kinematics_model": (
        "the self collision guard judges the arm the cell names: robot.ur.model on a UR cell, robot.sim.robot_model on "
        "a sim cell, whose Isaac arm reports that model instead of 'isaac-sim' (P0d D5, ported 2026-10-13). The key was "
        "a second name for the arm that two validators held equal to those two, and a third refused it on any other "
        "arm, where it had no right value. Name the arm in its vendor block and delete this line, a null one included."
    ),
    "robot.calibration.orientation_spread_deg": (
        "it shaped the poses a calibration sweep generated, and nothing generates a calibration station any more (the "
        "owner, 2026-09-24): the stations are a file of your own (--fixed-poses), tilted as you write them, or the arm "
        "guided by hand (--freedrive). Delete this key."
    ),
    "robot.calibration.max_attempts_per_pose": (
        "it was the pose generator's retry budget, and nothing generates a calibration station any more (the owner, "
        "2026-09-24): the stations are a file of your own (--fixed-poses) or the arm guided by hand (--freedrive). "
        "Delete this key."
    ),
    "robot.calibration.pose_box_z_frac": (
        "it was the z band the eye-to-hand pose generator drew from, and nothing generates a calibration station any "
        "more (the owner, 2026-09-24): the sim sweep runs the stations declared for its robot model "
        "(willy_sim/calibration/stations/eth_<model>.json, or run_eth_calibrate --stations PATH), and a real cell names "
        "its own (--fixed-poses) or guides the arm by hand (--freedrive). Delete this key."
    ),
    # The perception models (the owner's cleanup, 2026-10-17): the VLM grounds, RT-DETR answers with fixed classes,
    # SAM2 cuts, and every model reads its weights from its model_id under models.weights_root.
    "models.objectdetector": _GROUNDING_DINO_REMOVED,
    "models.detector": _GROUNDING_DINO_REMOVED,
    "models.segmenter_backend": _ONEFORMER_REMOVED,
    "models.oneformer": _ONEFORMER_REMOVED,
    "models.handdetect": _HAND_FIXED,
    "models.gesturedetect": _HAND_FIXED,
    "models.pipeline.router": _ROUTER_REMOVED,
    "models.pipeline.closed_set": _ONEFORMER_REMOVED,
    "models.pipeline.zero_shot.backend": _GROUNDING_DINO_REMOVED,
    "models.pipeline.zero_shot.segmenter": _ONEFORMER_REMOVED,
    "models.pipeline.zero_shot.vlm.model_path": _MODEL_PATH_REMOVED,
    "models.pipeline.zero_shot.vlm.local": _LOCAL_REMOVED,
    "models.pipeline.zero_shot.vlm.preload": _VLM_ANSWERS_FIXED,
    "models.pipeline.zero_shot.vlm.on_unavailable": _ON_UNAVAILABLE_REMOVED,
    "models.pipeline.zero_shot.vlm.stop_at_answer_end": _VLM_ANSWERS_FIXED,
    "models.pipeline.zero_shot.vlm.prompt_lookup_tokens": _VLM_ANSWERS_FIXED,
    "models.pipeline.zero_shot.vlm.text_prompt_lookup_tokens": _VLM_ANSWERS_FIXED,
    "models.pipeline.zero_shot.vlm.decode_graphs": _VLM_ANSWERS_FIXED,
    "models.segmenter.model_path": _MODEL_PATH_REMOVED,
    "models.segmenter.local": _LOCAL_REMOVED,
    "models.segmenter.optim": _OPTIM_REMOVED,
    "models.rtdetr.model_path": _MODEL_PATH_REMOVED,
    "models.rtdetr.local": _LOCAL_REMOVED,
    "models.rtdetr.optim": _OPTIM_REMOVED,
    "models.stt.model_path": _MODEL_PATH_REMOVED,
    "models.stt.vad_model_path": _VAD_PATH_REMOVED,
    "models.stt.local": _LOCAL_REMOVED,
    "models.stt.optim": _OPTIM_REMOVED,
    "models.stt.samplerate": _SPEECH_FIXED,
    "models.stt.blocksize": _SPEECH_FIXED,
    "models.stt.channels": _SPEECH_FIXED,
    "models.stt.dtype": _SPEECH_FIXED,
    "models.stt.task": _SPEECH_FIXED,
    "robot.sim.scene_setup.eih_viewpoints": (
        "it shaped the hemisphere of viewpoints the eye-in-hand sim sweep generated, and nothing generates a "
        "calibration station any more (the owner, 2026-09-24): the sim sweep runs the stations declared for its robot "
        "model (willy_sim/calibration/stations/eih_<model>.json, or run_eih_calibrate --stations PATH). Delete this key."
    ),
    # Every key of how a cell moves, under robot.motion (the owner's R1, 2026-10-16; built 2026-10-17).
    "robot.motion_limits": _moved_under_motion("robot.motion.max_velocity and robot.motion.max_acceleration"),
    "robot.workspace_limits": _moved_under_motion("robot.motion.workspace"),
    "robot.home_joint_positions_deg": _moved_under_motion("robot.motion.home_deg"),
    "robot.home_joint_positions": _HOME_RADIANS_REMOVED,
    "robot.look_joint_positions_deg": _moved_under_motion("robot.motion.look_deg"),
    "robot.natural_closing_axis": _moved_under_motion("robot.motion.natural_closing_axis"),
    "robot.named_poses": _moved_under_motion("robot.motion.named_poses, where the console's pose writer writes"),
    "robot.default_place_pose": _moved_under_motion("robot.motion.default_place_pose"),
    "robot.place": _moved_under_motion("robot.motion.place"),
    "robot.motion.place.below_the_rim_mm": _PLACE_FIXED,
    "robot.motion.place.opening_margin_mm": _PLACE_FIXED,
    "robot.motion.place.rim_air_mm": _PLACE_FIXED,
    "robot.motion.place.spacing_margin_mm": _PLACE_FIXED,
    "robot.motion.place.rim_floor_margin_mm": _PLACE_FIXED,
    "robot.motion.place.relocate": _PLACE_FIXED,
    "robot.safe_pose": (
        "the safe pose was removed on 2026-10-17 (the owner's cleanup): nothing moved to it, and its one reader, the "
        "willy_sim reach doctor, only asked whether it lay inside the arm's reach. Delete this block."
    ),
    #: Safety keys fixed at the cell's values (phase 2, 2026-10-17).
    "robot.safety.planning_world.perceived": _PERCEIVED_FIXED,
    "robot.safety.planning_world.enabled": _WORLD_ON_WITH_ITS_BENCH,
    "robot.safety.planning_world.include_fixtures": _FIXTURES_STAY_THE_GUARDS,
    "robot.safety.planning_world.require_registration": _REGISTRATION_REQUIRED,
    "robot.safety.planning_world.register_in_place": _REGISTERED_BY_UPDATE_WORLD,
    "robot.safety.planning_world.hold": _WORLD_HELD,
    "robot.safety.planning_world.payload": _NO_CARRIED_PART_MODELLED,
    "robot.safety.planning_world.payload.length_mm": _PART_HANG_MOVED,
    "robot.safety.planned_motion": _PLANNED_MOTION_FIXED,
    "robot.safety.self_collision.kinematics_base_yaw_deg": _BASE_YAW_IN_CODE,
    "robot.safety.self_collision.tool_length_mm": _CAPSULE_TOOL_FROM_THE_HAND,
    "robot.safety.self_collision.tool_radius_mm": _CAPSULE_TOOL_FROM_THE_HAND,
    "robot.safety.self_collision.tool_model": _CAPSULE_TOOL_FROM_THE_HAND,
    "robot.safety.self_collision.tool_finger_radius_mm": _CAPSULE_TOOL_FROM_THE_HAND,
    "robot.safety.self_collision.tool_finger_span_mm": _CAPSULE_TOOL_FROM_THE_HAND,
    "robot.safety.self_collision.base_radius_mm": _CAPSULE_BASE_FROM_THE_ARM,
    "robot.safety.self_collision.base_height_mm": _CAPSULE_BASE_FROM_THE_ARM,
    "robot.safety.self_collision.link_radii_mm": _CAPSULE_LINKS_FIXED,
    "robot.safety.self_collision.whole_path_judge": _WHOLE_PATH_JUDGED,
    "robot.safety.limits": _WORKSPACE_MARGIN_FIXED,
    "robot.safety.joint_limits": _JOINT_LIMITS_FIXED,
    "robot.safety.ik_quality": _IK_QUALITY_FIXED,
    "robot.safety.motion_continuity": _CONTINUITY_FIXED,
    "robot.safety.self_collision.enforce": _SELF_COLLISION_ALWAYS,
    "robot.safety.payload.enforce": _PAYLOAD_ALWAYS,
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

#: A recovery action that left the runtime, to the sentence that says what to do instead. The string
#: stays readable in a record logged before it left (the replay layer keeps it); only a campaign that
#: asks for it again is refused.
REMOVED_RECOVERY_ACTIONS: Final[dict[str, str]] = {
    "next_viewpoint": (
        "use rescan, which it was merged into on 2026-09-29: with the viewpoint planners gone nothing "
        "moved the camera for it, so it re-perceived from where the camera stood exactly as rescan does."
    ),
    "next_target": (
        "use rescan: the rescan that skipped the part a pick failed on left on 2026-10-16 (the owner's "
        "cleanup), and a pick takes the first part whose best grasp is good, the push and the clearing of a "
        "blocker beside it."
    ),
    "container_agitate": (
        "it left on 2026-10-16 with its fixture box (the owner's cleanup): no cell declared the container it "
        "needed, and the push moves a boxed-in part within what the camera saw."
    ),
}
