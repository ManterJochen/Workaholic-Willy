"""Shared constants for the willy_sim package: the log paths.

One name per directory keeps the bootstrap, the scene builder and both perception sources from
drifting apart on a magic string.

Not to be confused with :mod:`src.willy_sim.scene.constants`, which holds the scene's prim paths
and geometry, a different kind of constant owned by the scene subpackage.

This package keeps one log file per subsystem and no shared aggregate, unlike
``robot/constants.py``, which also has a shared ``robot.log``. An Isaac run's chronological
narrative is the runner's own stdout (the boot banner, the per-run ``RUN i: ...`` lines, the
``GATE: p/n`` line), which lands in that run's redirect and so belongs to exactly one run. The
files below answer a different question, what one subsystem did, and are read one subsystem at a
time.
"""

from __future__ import annotations

from typing import Final

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
#: Single directory under which every willy_sim module drops its rotating log file. Resolved
#: relative to the process working directory by :func:`src.utility.log_cfg.create_logger`.
WILLY_SIM_LOG_DIR: Final[str] = "logs/willy_sim"

#: The shared boot prefix (:func:`~src.willy_sim.harness.bootstrap.bootstrap_sim_cell`).
#: Its own file because it is the only place that records which cell came up: robot model, gripper
#: mount, and whether the cuRobo and exact-mesh engines were present. A pick rate measured on a
#: degraded stack describes a different system, and this file is where that is on record.
BOOTSTRAP_LOG_FILE: Final[str] = "bootstrap.log"

#: The Isaac scene builder (:mod:`~src.willy_sim.scene.build`). Separate from the boot log
#: because "the cell started" and "the stage holds these objects, this camera, this extrinsic" fail
#: independently: a scene that renders nothing still boots fine.
SCENE_BUILD_LOG_FILE: Final[str] = "scene_build.log"

#: Ground-truth perception (:mod:`~src.willy_sim.perception.ground_truth`). Its own file
#: because the annotator-buffer retries and the empty-mask substitutions recorded here are silent
#: everywhere else: downstream they arrive as an ordinary ``no_valid_grasp``.
PERCEPTION_GT_LOG_FILE: Final[str] = "perception_ground_truth.log"

#: Real-vision perception (:mod:`~src.willy_sim.perception.vision`). Kept apart from the
#: ground-truth log: comparing a vision run against a ground-truth run is the standard sim
#: diagnosis, and it only works if the two do not interleave in one file.
PERCEPTION_VISION_LOG_FILE: Final[str] = "perception_vision.log"

#: Sim hand-eye calibration (:mod:`~src.willy_sim.calibration.hand_eye`). Its own file
#: because a calibration run is a long sweep of viewpoints whose interesting content is which views
#: the marker source refused, a question nobody asks during a pick.
HAND_EYE_LOG_FILE: Final[str] = "hand_eye.log"

#: The GSO ``.obj`` to USD converter (:mod:`~src.willy_sim.gso_assets`). An offline, one-time
#: on-box job rather than a runtime component: mixing it into the scene log would bury a per-run
#: scene line under a whole asset conversion.
GSO_ASSETS_LOG_FILE: Final[str] = "gso_assets.log"

#: Domain randomization (:mod:`~src.willy_sim.harness.randomizer`). Separate because these
#: lines are read against a dataset rather than against a run: they say what the recorded episodes
#: were actually varied over, and whether the visual axis silently degraded.
RANDOMIZER_LOG_FILE: Final[str] = "randomizer.log"

#: The per-pick artifact seam (:mod:`~src.willy_sim.harness.instrumentation`). Its own file
#: because it is the shared place every runner writes its records / overlays / result JSON through,
#: so it is the one index of what a collection run actually left on disk.
INSTRUMENTATION_LOG_FILE: Final[str] = "instrumentation.log"

#: The parsed ``WILLY_*`` runner knobs (:mod:`~src.willy_sim.harness.env`). Read against a
#: result rather than against a run: every one of these knobs moves a measured number (which scene,
#: which detector, which camera height, which close width), and nothing else on disk says whether a
#: given figure was taken with the documented defaults or with overrides in the shell.
RUNNER_ENV_LOG_FILE: Final[str] = "runner_env.log"

#: The shared safety-guard wiring seam (``run_dense_pick.wire_safety_guards``, which every runner's
#: ``build_service`` calls). Not a runner log: "the guard was asked for" and "the guard is
#: installed" are different facts and can disagree silently. The engine-resolution half is durable
#: in the robot package's ``planning_environment.log``; this is the request half.
SAFETY_GUARDS_LOG_FILE: Final[str] = "safety_guards.log"

# ----------------------------------------------------------------------
# Producers: runners whose output outlives the shell that ran them
# ----------------------------------------------------------------------
# The pick and demo gate runners get no file: their whole content is a per-run narrative that is
# already on stdout, and stdout lands in that run's own redirect, which is what makes a number
# attributable to a run (see the block comment in ``run_m1_pick.run_gate``). The runners below are
# different: they leave an artifact behind, a calibration a whole cell then trusts, a labelled
# dataset, a baseline other numbers are diffed against, and the artifact carries no record of the
# conditions it was produced under.

#: Both calibration runners (:mod:`~src.willy_sim.run_eth_calibrate` and
#: :mod:`~src.willy_sim.run_eih_calibrate`) share one file on purpose: ETH and EIH are the
#: two halves of one calibration session, and checking that the pair agrees means reading them
#: together. The logger name on every line keeps them attributable. Distinct from ``hand_eye.log``,
#: which records which views the marker source refused, a different question from what was solved.
CALIBRATION_RUNS_LOG_FILE: Final[str] = "calibration_runs.log"

#: The mode-matrix driver (:mod:`~src.willy_sim.run_mode_matrix`). The one runner that is
#: above the per-run stdout rule: it spawns each cell as its own subprocess with its own redirect,
#: so the driver's narrative (which cells were launched, which timed out and had their tree killed,
#: which produced no result) exists in no cell's log by construction.
MODE_MATRIX_LOG_FILE: Final[str] = "mode_matrix.log"

#: The physics shake labeller (:mod:`~src.willy_sim.run_shake_label`). Its JSONL is training
#: data, and a label file is only usable while the cell and the shake config that produced it are
#: still recoverable.
SHAKE_LABEL_LOG_FILE: Final[str] = "shake_label.log"

#: The dense-clutter pile baseline (:mod:`~src.willy_sim.run_pile_baseline`). Every later
#: lever is measured as a delta against this report, so the conditions behind the floor number
#: (perception mode, pile count, which opt-in blocks were on) matter as much as the number.
PILE_BASELINE_LOG_FILE: Final[str] = "pile_baseline.log"

#: The multi-view occlusion probe (:mod:`~src.willy_sim.run_occlusion_probe`). Its own file
#: mainly for the degraded paths: when the reach move or the wrist fit fails the probe still prints
#: a full-looking coverage summary with a silently missing pose, which is the "reads as a
#: measurement, is not" shape.
OCCLUSION_PROBE_LOG_FILE: Final[str] = "occlusion_probe.log"
