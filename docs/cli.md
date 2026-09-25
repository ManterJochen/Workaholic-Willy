# The command line

Every capability the [examples](../examples/README.md) show from Python also has a command, except the
few named below that a program holds between calls. Run each from the repository root with the
project's virtual environment active. A command that reads the configuration tree follows
`WILLY_PROFILE`, or the `--profile` it is given. Below, `<your cell>` is your cell's profile, `<rig id>` a
camera rig of your camera section and `<model>` an arm model key.

The topics follow the examples: your cell, then simulation, then offline. A row whose description starts
with "Moves the arm" commands the robot; every other command reads configuration, opens a camera or
computes at a desk.

## Your cell

### The config tree

`python -m src.config` exits 0 when the tree loads or a question was answered (asking about a key that
does not exist is answered too), 1 when the tree does not load, with the file and the key on the line
above, and 2 on bad arguments.

| command | what it does | runbook |
|---|---|---|
| `python -m src.config --profile <your cell>` | validates the tree under a profile; `--print` adds the tree as JSON, `--data <dir>` reads another tree | [cell_bringup](runbooks/cell_bringup.md), [real_cell_first_pick](runbooks/real_cell_first_pick.md), [ur_family_bringup](runbooks/ur_family_bringup.md) |
| `python -m src.config explain <key>` | one key's value, type, bounds and default, and the file and line that set it; a misspelled key gets the key that was meant | [cell_bringup](runbooks/cell_bringup.md) |
| `python -m src.config decisions --section <prefix>` | only what the cell decided: the values under a prefix that differ from their default, each with its file and line | [cell_bringup](runbooks/cell_bringup.md) |
| `python -m src.config where <word>` | searches the schema rather than the files, so it finds keys no YAML writes; it refuses `--profile` and `--data` | |
| `python -m src.config --profile <your cell>,<layer>` | chains one more layer on top; a layer no file carries is refused, never skipped | [cell_bringup](runbooks/cell_bringup.md) |

To try a value without editing your tree, copy the tree and pass `--data <copy>`; a value that contradicts
another file (a width beside a named hand, say) is refused naming both places. A relative path in the
copy is read against the copy's folder, so the shipped `../calibration/...` paths then point beside the
copy ([01-configuration.md section 2](guide/01-configuration.md#2-what-the-loader-reads)). From Python,
`LoadedTree.with_values` changes a value in memory.

### The desk check

`real_cell` exits 0 when the campaign passed, or when `--check`, `--dry-run` or `--start-planner` was
satisfied, 1 when the preflight blocked or the build, the connect or the planner start was refused, 2 when
the cell connected and the campaign did not pass its rule, and 3 when a fault of the cell stopped a pick.

| command | what it does | runbook |
|---|---|---|
| `python -m src.robot.execution.real_cell --check` | every stop-the-cell condition decidable at a desk, each with its fix; touches nothing and exits 1 while anything blocks | [cell_bringup](runbooks/cell_bringup.md), [real_cell_first_pick](runbooks/real_cell_first_pick.md) |
| `python -m src.robot.execution.real_cell --start-planner` | on a planner cell, builds the arm alone and starts its planner the way the first planned move does; no camera, no controller, exit 0 when it started | [cell_bringup](runbooks/cell_bringup.md), [your_own_gripper](runbooks/your_own_gripper.md) |
| `python -m src.robot.drivers.doctor --require <vendor>` | which arm and gripper drivers this machine can build; exit 1 when the named arm vendor is not ready, 2 for a vendor it does not know | [real_cell_first_pick](runbooks/real_cell_first_pick.md) |
| `python -m src.robot.execution.real_cell --dry-run` | builds the whole cell and stops before the connect; it opens the cameras and loads the models, and moves nothing | [real_cell_first_pick](runbooks/real_cell_first_pick.md) |

### Connecting

| command | what it does | runbook |
|---|---|---|
| `python scripts/checks/cell_bringup.py --live` | connects the arm alone under the cell lock, reads its pose back and holds it against the workspace box; no motion. Exit 0 it stands inside its box, 1 it disagrees with its configuration, 2 there is nothing to connect to | [cell_bringup](runbooks/cell_bringup.md), [real_cell_first_pick](runbooks/real_cell_first_pick.md) |
| `python -m src.robot.drivers.ur --read` | the bench for a gripper on the UR controller's digital I/O; never moves the arm. `--read` and `--watch PIN` read pins. Every write (`--set`, `--pulse`, `--measure`, and `--jaws open` or `--jaws closed`, which moves the configured `jaw_io` hand through its driver, a single toggle asking first where its jaws stand) needs `--yes`, or a typed `yes` at a terminal. `--where` is read-only: it prints the arm's joints as a `JointPositions.deg(...)` line to paste as a look pose, and the TCP in mm and degrees | [hande_gripper_bringup](runbooks/hande_gripper_bringup.md) |

### The camera

| command | what it does | runbook |
|---|---|---|
| `python -m src.config explain camera.cameras.primary_rig_id` | which rig the cell opens, where that was set, and the rigs it could name instead | |
| `python -m src.config explain camera.cameras.rigs` | every rig as it validated: its source, whether it is enabled, and `extrinsics`, its declared calibration (`None` is uncalibrated) | |
| `python -m src.robot.execution.real_cell.calibrate --rig <rig id> --freedrive --check` | the refusals a named rig meets (unknown, switched off, no depth), from configuration alone | [real_cell_first_pick](runbooks/real_cell_first_pick.md) |
| `python -m src.robot.perception --rig <rig id> --warmup 10` | opens the rig, grabs, detects and segments, then prints the lens matrix and the depth holes inside every mask; needs the camera and the model weights, no robot | [real_cell_first_pick](runbooks/real_cell_first_pick.md) |
| `python -m src.robot.perception --prompt "<phrase>"` | the same, grounding one prompt on the primary rig | [real_cell_first_pick](runbooks/real_cell_first_pick.md) |

### Calibration

`calibrate` exits 0 when done, 1 when the configuration or the build refused, another process holds the
cell or the connect refused, 2 when it ran and wrote no artifact (the cell keeps its previous calibration)
and 3 when the sweep raised. Every line below is used by
[real_cell_first_pick](runbooks/real_cell_first_pick.md) and [calibration-setup.md](calibration-setup.md).

| command | what it does |
|---|---|
| `python -m src.robot.execution.real_cell.calibrate --rig <rig id> --mode eye_to_hand --freedrive --check` | validates the configuration and the rig; touches no hardware. A run names its stations or is guided by hand: without `--freedrive` or `--fixed-poses` it is refused. The arm carries every wrist camera body the tree declares, named on a `wrist cameras` line; one whose camera is not calibrated yet refuses here unless `--unmodelled-wrist-body "<reason>"` is given for this sweep, and a wrist rig declared with no `body` refuses nothing and is named on one `!!` line (`wrist camera '<id>' is declared on the arm without a body: ...`) |
| `python -m src.robot.execution.real_cell.calibrate --rig <rig id> --mode eye_to_hand --freedrive --dry-run` | builds the arm and opens that one camera, then stops before any motion; an arm that offers no hand guiding is refused here. The build repeats the `wrist cameras` line and the `!!` line of a wrist rig without a body, which is also logged; `--unmodelled-wrist-body "<reason>"` excuses a wrist body that cannot be placed yet, as at `--check` |
| `python -m src.robot.execution.real_cell.calibrate --rig <rig id> --mode eye_to_hand --freedrive --marker-length-mm <measured>` | You move the arm by hand to each pose where the camera sees the board and press Enter; nothing moves by itself. Shows the controller payload for confirmation first, ends at `--samples` (15) or `q`, writes `eth_<rig id>.json`, the stations it counted (`eye_to_hand_<rig id>_stations.json`) and the rig block to paste. Takes `--unmodelled-wrist-body "<reason>"` as `--check` does |
| `python -m src.robot.execution.real_cell.calibrate --rig <rig id> --mode eye_in_hand --fixed-poses <stations.json> --adjust` | Moves the arm to each of your stations, frees it there to be fine-tuned by hand, and asks for your hands off and counts down before each next move; writes `eih_<rig id>.json`. Without `--adjust` the stations run without hands, on any arm. `--unmodelled-wrist-body "<reason>"` sweeps a camera whose body cannot be placed yet; in either mode it also lets the arm move without the body of any other wrist camera the tree declares that is not calibrated yet, which otherwise refuses `--check`. A wrist rig declared with no `body`, the camera this mode calibrates included, needs no reason and is named on one `!!` line; a pick (`Robot.from_tree`, the real cell) still refuses an enabled one |

### Picks

| command | what it does | runbook |
|---|---|---|
| `python -m src.robot.execution.real_cell --runs 3 --prompt "<phrase>"` | Moves the arm: a campaign of picks under one connect; exit 0 only when every attempt succeeded | [real_cell_first_pick](runbooks/real_cell_first_pick.md) |
| `python -m src.robot.grasping.replay --records <file>` | rolls up the KPIs of the record file a run wrote, once the tree sets `grasping.record_log_path` | [real_cell_first_pick](runbooks/real_cell_first_pick.md) |

A spoken command, the locator and a pick at a known pose have no command: they are objects a program
holds between calls, so they are Python only ([examples/real_robot/](../examples/README.md)).

## Simulation

### At a desk

| command | what it does | runbook |
|---|---|---|
| `python -m src.robot.execution.real_cell --rehearse --runs 3 --profile console_dummy` | the whole call path on a dummy arm and a synthetic scene, commanding nothing; the exit codes of a live run | [real_cell_first_pick](runbooks/real_cell_first_pick.md) |
| `python -m src.robot.execution.real_cell --rehearse --dry-run` | builds your cell on a dummy arm and stops, printing which gripper class came out; a `NullGripper` holds nothing | |

A rehearsal without `--profile console_dummy` keeps your hand, and a hand that lives on the real
controller cannot be built on a dummy arm: that cell is refused at the connect, exit 1.

### Isaac Sim

These run under Isaac's own interpreter, `<isaac-sim>/python.bat -m ...` ([isaac-ready.md](isaac-ready.md));
the project environment answers `--help` only. A pick runner's verdict is the `GATE:` line it prints and
the JSON `--result-json` writes, not its exit code. The three pick runners take `--robot-model <model>`
and `--profile <layer>`, which build the chain `sim,<model>,<layer>`.

| command | what it does | runbook |
|---|---|---|
| `python -m src.willy_sim.run_m1_pick --runs 10 --result-json <file>` | ten known-pose picks, headless, scored against the scene's ground truth | [cell_bringup](runbooks/cell_bringup.md) |
| `python -m src.willy_sim.run_m2_pick --runs 5 --prompt "<phrase>"` | real vision in the loop: the detector and the segmenter decide what is picked | [cell_bringup](runbooks/cell_bringup.md) |
| `python -m src.willy_sim.run_eih_pick --runs 10` | the wrist camera rides the arm and perceives again from where it moved | |
| `python -m src.willy_sim.run_eih_demo --fps 18 --out <file>.mp4` | films one wrist camera pick; without `--out` the film lands at `logs/demo/eih_pick_demo.mp4` | |
| `python -m src.willy_sim.run_bin_clearing_demo --out <file>.mp4` | films a bin emptied object by object | |
| `python -m src.willy_sim.run_sorting_demo --concat <a>.mp4,<b>.mp4 --out <file>.mp4` | stitches finished clips into one file; needs no Isaac | |

## Offline

### Config: which hand, which planner

`safety.planning` exits 0 when the cell is fully anchored and 1 when a degraded fallback would run (blind
IK, or the capsule proxy), which is also its answer when the tree did not load. `--doctor` exits 2 when an
operating system policy blocks a binary ([code-integrity.md](code-integrity.md)).

| command | what it does | runbook |
|---|---|---|
| `python -m src.robot.safety.planning --check` | reads the paths the planner and the exact-mesh collision engine need for the arm the configuration names; spawns nothing | [cell_bringup](runbooks/cell_bringup.md), [ur_family_bringup](runbooks/ur_family_bringup.md) |
| `python -m src.robot.safety.planning --doctor` | loads every engine instead: one real distance query, and the planner's own interpreter asked what resolves | [cell_bringup](runbooks/cell_bringup.md), [ur_family_bringup](runbooks/ur_family_bringup.md), [hande_gripper_bringup](runbooks/hande_gripper_bringup.md) |
| `python -m src.robot.safety.planning --check --model <model> --hand <hand>` | the same for another arm or hand than the configuration names; `--profile <layer>` reads an arm's own layer instead | [ur_family_bringup](runbooks/ur_family_bringup.md) |

`robot.safety.self_collision.backend` states an intention and these two state what will run: an arm with
no mesh bundle runs the capsule proxy, and `--check` says so with exit 1.

### Perception

| command | what it does |
|---|---|
| `python -m src.config explain models.pipeline.kind` | an open-vocabulary grounder or a closed-set detector |
| `python -m src.config explain models.detector` | the legacy detector key, read by a tree without the pipeline block and by the bench exerciser |
| `python -m src.config explain models.segmenter_backend` | its segmenter twin |
| `python -m src.config explain models.pipeline.zero_shot.backend` | which model grounds: `grounded_sam` for phrases, `vlm` for the prompts a phrase grounder cannot represent |
| `python -m src.config explain models.pipeline.router.enabled` | whether the per-prompt router is on; it reads the text and loads no model |
| `python -m src.config explain models.pipeline.zero_shot.vlm.on_unavailable` | `refuse` stops, `degrade` grounds with the phrase detector instead |

Routing a prompt has no command; the operator console answers `GET /v1/diagnostics/route?prompt=...`.

### Safety

| command | what it does |
|---|---|
| `python -m src.robot.execution.real_cell --check` | its `fixtures` and `planning world` rows say what the cell declared about the space it moves through |
| `python -m src.config` | after declaring the bench, the tree still loads |

No command judges a trajectory; [gate_the_whole_path.py](../examples/offline/safety/gate_the_whole_path.py)
does it from Python.

### Grasping

| command | what it does |
|---|---|
| `python -m src.config explain robot.gripper.vendor` | which driver actuates the hand; `vacuum` is the suction one |
| `python -m src.config explain robot.grasping.gripper_geometry.kind` | the envelope candidates are filtered against; `suction` changes the shape, not the pick path |
| `python -m src.config explain robot.gripper.vacuum` | the pins, the port and the time the ejector takes to build vacuum |
| `python -m src.config explain robot.grasping.calculator` | `geometric` or `deep` |
| `python -m src.config explain robot.grasping.deep_generator.artifact_path` | where the deep branch reads its weights |
| `python -m src.robot.grasping.deep inspect --artifact <file>` | what a weights file says about itself: kind, version, gripper |
| `python -m src.robot.grasping.calibration --replay <records.jsonl> --out <file>.json` | fits the uncertainty calibration of the grasp score from a replay of pick records |

Nothing on the command line proposes a suction grasp or triggers the deep generator's refusal; the
examples under [offline/grasping/](../examples/README.md) do both from Python.

### Data generation

`datagen` exits 0 when done, 1 on a problem named on the line above and 2 on bad arguments.

| command | what it does | runbook |
|---|---|---|
| `python -m datagen.assets.fetch --list` | every public mesh collection, its size, its licence, and whether that was checked per model | [train_your_own_generator](runbooks/train_your_own_generator.md) |
| `python -m datagen.assets.fetch gso --limit 20` | downloads a collection into the gitignored mesh library; an already-present mesh is skipped, so it resumes | [train_your_own_generator](runbooks/train_your_own_generator.md) |
| `python -m datagen.assets --check` | what the mesh library holds per source, and whether it is enough to render from | [train_your_own_generator](runbooks/train_your_own_generator.md) |
| `python -m datagen.assets --attribution` | the attribution text a CC-BY collection obliges you to ship with a dataset built from it | |
| `python -m datagen.assets --fetch --from <dir> --source custom --license own --attribution-text "<you>"` | imports your own parts; the licence is required, never defaulted | [train_your_own_generator](runbooks/train_your_own_generator.md) |
| `python -m datagen audit` | the licence gate on the manifest the configuration draws, before a render rather than after | |
| `python -m datagen plan --scenes 40 --seed 0` | the family mix and the object counts of a whole dataset | |
| `python -m datagen describe --scenes 40 --index 0` | one scene in full as JSON: assets, poses, cameras, lighting | |
| `python -m datagen init-config --recipe v1 --out <file>.json` | writes a configuration file for a recipe, without rendering | [train_your_own_generator](runbooks/train_your_own_generator.md) |
| `python -m datagen cost --engine none --scenes 8` | prices a request; a request is not a yield, and `none` refuses the pile family | [corpus_v5_build](runbooks/corpus_v5_build.md) |
| `python -m datagen build --engine none --scenes 11 --name <name> --out <dir>` | renders a dataset; `none` needs nothing, `mujoco` a pip wheel, `isaac` a GPU install | [train_your_own_generator](runbooks/train_your_own_generator.md), [corpus_v5_build](runbooks/corpus_v5_build.md) |
| `python -m datagen label-grasps --name <name> --out <dir>` | labels every scene from geometry, on the CPU; `--density grid` gives many more labels | [train_your_own_generator](runbooks/train_your_own_generator.md), [corpus_v5_build](runbooks/corpus_v5_build.md) |
| `python -m datagen physics-sample --name <name> --out <dir> --physics-engine mujoco --per-class 4` | the physics screen over the jaw labels; needs `pip install mujoco`, or Isaac | [train_your_own_generator](runbooks/train_your_own_generator.md) |
| `python -m datagen why-no-jaw --collection <collection> --limit 3` | which meshes a jaw cannot grasp at all, and why | |
| `python -m datagen build-cloud-corpus --name <name> --out <dir> --corpus-out <dir> --kinds both --physics <file>` | extracts the point cloud corpus a generator trains on; `--kinds` decides which grasps exist | [train_your_own_generator](runbooks/train_your_own_generator.md), [corpus_v5_build](runbooks/corpus_v5_build.md) |
| `python -m src.robot.grasping.deep sanity --clouds <dir> --scenes 8` | whether the corpus's ground truth sits where a network can see it | |

### Training

`deep` exits 0 when done, 2 on bad arguments and 3 on a problem.

| command | what it does | runbook |
|---|---|---|
| `python -m src.robot.grasping.deep train-set --tier smoke --recipe v1 --run-folds 1 --clouds <dir> --out <dir>` | fits the generator; the smoke tier is minutes, `--tier full` hours on a GPU | [train_your_own_generator](runbooks/train_your_own_generator.md) |
| `python -m src.robot.grasping.deep report --run <dir>` | the four numbers, the lift among them and the verdict; `--run` is the trainer's output directory, `--curve <file>.png` places the curve | [train_your_own_generator](runbooks/train_your_own_generator.md) |
| `python -m datagen cost --scenes 2000 --engine none --epochs 36 --train-folds 1 --no-refit` | prices the whole chain with training; `--no-refit` prices the single pass a tier alone buys | |
| `python -m src.robot.grasping.deep import-foreign --out <dir> --limit 4 --jobs 8` | imports a public 6-DoF grasp corpus as the scene files the local generator writes; needs the network, and the first run caches the archive index | [train_your_own_generator](runbooks/train_your_own_generator.md) |
| `python -m src.robot.grasping.deep train-set --help` | the recipe and tier vocabulary, on the command that takes them | |

## Checks

A check holds the cell against its own configuration and says what is wrong: exit 0 when it agrees, 1
when it does not, 2 when there is nothing to check. Run each under the cell's profile.

| command | what it does | runbook |
|---|---|---|
| `python scripts/checks/cell_bringup.py --live` | see [Connecting](#connecting) | [cell_bringup](runbooks/cell_bringup.md), [real_cell_first_pick](runbooks/real_cell_first_pick.md) |
| `python scripts/checks/safety_guards.py` | makes every wired guard refuse a violation of its own family | |
| `python scripts/checks/camera_artifacts.py` | opens every calibration artifact the configuration names | |
| `python scripts/checks/grasping_switches.py` | which grasping block is reachable in which grasp mode | |
