"""Shared constants for the datagen package: today, the log paths.

Centralising them here is the same call the ``models`` package made: the alternative is the
magic-string drift that appears the moment two modules name the same directory by hand.

``datagen`` has no aggregate log and deliberately does not grow one. The robot package keeps a
``robot.log`` because its subsystems interleave inside a single pick and the chronology across them is
the story. A datagen run is the opposite shape: one long pass owns the process, a render or a mask
prediction or an evaluation ladder or a physics sample, and what an operator wants hours later is
that one pass's file, not a merge of five passes that never overlapped. So one file per module, like
``models``, and the file name says which pass wrote it.

Paths are relative to the process working directory, resolved by
:func:`src.utility.log_cfg.create_logger`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
#: One directory for the whole generator. Not under ``logs/``: datagen is a tool that imports
#: ``src`` and is never imported by it, and its logs should be as separable as its code.
DATAGEN_LOG_DIR: Final[str] = "logs/datagen"

#: The CLI itself: the run header (which command, over which dataset) that makes every line in the
#: files below attributable to an invocation.
CLI_LOG_FILE: Final[str] = "cli.log"

#: The licence gate. Its own file on purpose: this is the one log with a legal reason to exist, and it
#: must stay readable without scrolling past a night of render progress.
LICENSING_LOG_FILE: Final[str] = "licensing.log"
#: The dataset gate: what a corpus was judged on, and why it was refused. Its own file because
#: the answer has to be readable months later, when the model trained on it behaves oddly.
CORPUS_GATE_LOG_FILE: Final[str] = "corpus_gate.log"
#: Reading real meshes into closed solids: every refusal (unreadable, boundaries that are not
#: simple loops, closes to a sheet) is one object the bank silently loses without this.
MESH_GEOMETRY_LOG_FILE: Final[str] = "mesh_geometry.log"
#: The dataset's serial number: the git state a stamp was built from, including "git could not say".
PROVENANCE_LOG_FILE: Final[str] = "provenance.log"
#: Which robot a run resolved, and what it could not do with it: no measured wrist mount, no
#: wrist view.
ROBOTS_LOG_FILE: Final[str] = "robots.log"

#: The on-box renderer session and its per-scene outcomes.
RENDER_ISAAC_LOG_FILE: Final[str] = "render_isaac.log"
#: Where the bytes landed, and what a resumed run decided to skip.
WRITER_LOG_FILE: Final[str] = "dataset_writer.log"
#: The build loop, split from the two files above because it records the decisions neither of them
#: sees: how many scenes a resume skipped, every render retry, and the abort when a session is broken
#: rather than unlucky. The retry rate is a measured property of the renderer, and stdout is not
#: something an overnight run keeps.
BUILD_LOG_FILE: Final[str] = "build.log"

#: Scene layout: which asset source each object came from, and every fallback taken when a
#: weighted source could not supply one. Its own file because "why is this scene all procedural
#: when the config asked for real meshes" is answered by one grep, and because those lines are
#: decisions rather than progress: they belong next to each other, not inside a render log.
LAYOUT_LOG_FILE: Final[str] = "scene_layout.log"
#: The on-box arm check. Its own file for a mechanical reason: Isaac's ``close()`` terminates the
#: process, so this command's verdict has to be somewhere that survives the teardown.
VERIFY_ROBOT_LOG_FILE: Final[str] = "verify_robot.log"

#: The post-hoc passes over a written dataset. Separate files because they are separate runs, hours
#: apart, and each is read on its own.
VERIFY_LOG_FILE: Final[str] = "verify.log"
PREVIEW_LOG_FILE: Final[str] = "preview.log"
PROMPTS_BUILD_LOG_FILE: Final[str] = "prompts_build.log"
PARAPHRASE_LOG_FILE: Final[str] = "paraphrase.log"
PARAPHRASE_MODEL_LOG_FILE: Final[str] = "paraphrase_model.log"

#: The grasp tier: labels, the two mask sources, the evaluation ladder, the physics harness.
GRASP_LABELS_LOG_FILE: Final[str] = "grasp_labels.log"
#: Splitting a concave mesh into convex parts so a solver collides the shape the scene
#: contains. Its own file because the entries are the record of a cache: which mesh was
#: decomposed, into how many parts, and under which settings. A run that suddenly slows down
#: is a run whose cache key moved, and this is where that is visible.
CONVEX_DECOMPOSITION_LOG_FILE: Final[str] = "convex_decomposition.log"
MASK_PREDICT_LOG_FILE: Final[str] = "mask_predict.log"
GRASP_EVAL_LOG_FILE: Final[str] = "grasp_eval.log"
SUPPORT_EVAL_LOG_FILE: Final[str] = "support_eval.log"
PHYSICS_SAMPLE_LOG_FILE: Final[str] = "physics_sample.log"
#: The Isaac side of the harness, split from the sampler above: when a hold rate looks wrong, the
#: question is always whether the cell was healthy, and that answer should not be interleaved.
PHYSICS_CELL_LOG_FILE: Final[str] = "physics_cell.log"

#: Driving the real camera adapter over written scenes (the d435 bench).
CAMERA_PROBE_LOG_FILE: Final[str] = "camera_probe.log"

#: The RL chain: the occupancy gate, then physics collection, then the ranker proof.
#: ``rl_perception`` is the scene-to-rig loader in front of all three. Separate from them because a
#: rig built from fewer cameras than the scene declares is a quiet fidelity change, and the sweep
#: that consumes it reports only that it got a rig.
RL_PERCEPTION_LOG_FILE: Final[str] = "rl_perception.log"
RL_OCCUPANCY_LOG_FILE: Final[str] = "rl_occupancy.log"
RL_COLLECT_LOG_FILE: Final[str] = "rl_collect.log"
RL_PROOF_LOG_FILE: Final[str] = "rl_proof.log"

#: Where fetched meshes live. Declared once and imported by `assets/library.py` and
#: `assets/fetch.py` alike: two declarations of one value are held in step by nothing, and a
#: comment is not a mechanism.
#:
#: `constants.py` is the right home because it is stdlib-only, so naming this path costs no numpy,
#: which `library.py` would pull in through the manifest chain.
#:
#: Gitignored: fetched binaries do not belong in a clone, and a different revision than the one
#: last tested may be wanted.
MESH_LIBRARY_DIR: Final[Path] = Path("assets/meshes")
