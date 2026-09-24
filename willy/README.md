# `willy`: the library in one import

`from willy import ...` gives you every public name of Workaholic-Willy: a cell's config tree, a
robot and its verbs, a whole cell and its pick campaigns, cameras and calibration, grasps, speech,
Isaac Sim, and offline data and training. Importing `willy` loads nothing; a name loads its own module
the first time you use it, and it is exactly the object that module defines, so mypy checks every call
you make through it.

```python
from willy import Pose, Robot, load_tree

robot = Robot.from_tree(load_tree("console_dummy"))   # the desk profile; load_tree() reads WILLY_PROFILE
with robot.connected():
    print(robot.home())                               # every verb returns a report that prints as itself
    print(robot.move(Pose.tool_down(450.0, 100.0, 300.0)))
```

At a real cell every motion is checked against the camera world: hand the robot its cameras
(`Robot.from_tree(tree, cameras=[camera])`), or say why it moves without them, with
`with robot.without_camera_world("why"):` or `decline="why"` on one verb, as `real_robot/03` does.

Install the repository once with `pip install -e . --no-deps` from its root, and this runs from any
directory. The programs that use each name are in [`examples/`](../examples/README.md).

## Choosing the tree

Every noun that describes a cell is built from a loaded config tree, `X.from_tree(tree)`:

- `load_tree()` loads the chain `WILLY_PROFILE` names, so a program run as
  `WILLY_PROFILE=<your cell> python ...` names no robot itself.
- `load_tree("ur5e,eth2")` loads that chain, and `load_tree("console_dummy")` the desk profile.
- `load_tree(None)` loads the base tree, whatever `WILLY_PROFILE` says; `root=` reads another tree.
- A tree that does not load comes back with `ok` false: `print(tree)` gives the refusal with the file
  and the line to fix, and `tree.exit_code` is 1. Any `from_tree` on it raises `ConfigError`.

## The names

Grouped as a program meets them. "Shown in" names the example under `examples/`, or where you meet a
name that no example imports.

### The config tree of a cell

| Name | What it is | Shown in |
|---|---|---|
| `load_tree` | Loads and validates a tree in one call, and returns a `LoadedTree` | `real_robot/01_load_your_cell.py` |
| `LoadedTree` | A validated tree: `ok`, `.robot`, `explain(key)`, `decisions()`, `with_values({...})` | `real_robot/01_load_your_cell.py` |
| `ConfigTree` | Where the YAML lives and which layers apply; `ConfigTree.from_directory(...).load()` | what `load_tree` calls |
| `ConfigError` | The refusal of a tree that did not load, raised when something asks it for a robot | any `from_tree` |
| `load_speech_section` | `models.stt` alone, so speech loads without a camera, a robot or a detector | `real_robot/12_speak_a_command.py` |

### Poses and frames, in millimetres

| Name | What it is | Shown in |
|---|---|---|
| `Pose` | A 6-DoF pose tagged with its frame, in millimetres with an XYZW quaternion; `Pose.tool_down(x, y, z)` | `real_robot/03_connect_and_move.py` |
| `Pose.aimed_at` | The same, pointing the tool's +Z AT a point instead of straight down: a camera or a board turned to face something | `real_robot/08_calibrate_a_fixed_camera_with_fixed_poses.py` |
| `Frame` | The frames a pose can be in: `BASE`, `CAMERA`, `TCP`, `TOOL`, `OBJECT`, `GRASP`, `WORLD`, `MARKER` | `Pose.frame` |

### The robot: an arm and its hand

| Name | What it is | Shown in |
|---|---|---|
| `Robot` | An arm and its hand: `connected()`, `home`, `move`, `move_joints`, `grasp`, `release`, `is_holding`, `pick`, `place`, `safety()`, `without_camera_world(why)` | `real_robot/03_connect_and_move.py` |
| `MotionReport` | What a motion verb asked, how the arm's motions reach it, and what the arm said | `real_robot/03_connect_and_move.py` |
| `MotionOutcome` | How a motion verb ended: `EXECUTED`, `MOTION_REFUSED`, `CAMERA_WORLD_UNAVAILABLE`, `REFUSED` | `MotionReport.outcome` |
| `HandReport` | What `grasp()` or `release()` commanded and measured, and what it did to the carried part model | `real_robot/04_open_and_close_the_hand.py` |
| `HandOutcome` | How a hand verb ended, such as `GRASPED`, `NOTHING_HELD` or `RELEASE_NOT_CONFIRMED` | `HandReport.outcome` |
| `HandlingReport` | What `pick()` or `place()` commanded, what stood behind each motion, and what the hand did | `real_robot/05_pick_and_place_a_known_part.py` |
| `HandlingOutcome` | How a pick or a place ended; `NOTHING_HELD` means the fingers closed on nothing | `HandlingReport.outcome` |
| `HoldEvidence` | What the hand measured after its last command: `HELD`, `EMPTY` or `UNMEASURED` | `real_robot/04_open_and_close_the_hand.py` |
| `JointPositions` | An immutable joint configuration, in radians; `JointPositions.deg(...)` takes degrees, as the pendant shows them | `real_robot/11_pick_with_the_camera.py` |
| `SafetyPreflight` | The ordered fail-closed guards; `SafetyPreflight.from_tree(tree)` builds a cell's own at a desk | `offline/safety/gate_the_whole_path.py` |
| `create_arm` | The arm driver registered for a vendor, built and not connected | `offline/safety/gate_the_whole_path.py` |

### The whole cell and its pick service

| Name | What it is | Shown in |
|---|---|---|
| `Cell` | Cameras, perception, the grasp stack, the arm and the hand: `preflight()`, `build()`, `connected()`; `mode=` picks the grasp mode | `real_robot/11_pick_with_the_camera.py` |
| `AutonomousGraspService` | What `Cell.build()` returns: one attempt per `pick()` (`look=` where it looks from), `put_back(report)`, `set_prompt()`, `enable_record_logging()` | `cell.service` in `real_robot/11_pick_with_the_camera.py` |
| `AutonomousGraspReport` | What one pick did: outcome, mode, the profile in effect, and the layers that actually ran | `simulation/06_grasp_modes_and_what_each_needs.py` |
| `AutonomousGraspOutcome` | How a pick ended: `SUCCEEDED`, `NO_TARGET`, `NO_VALID_GRASP`, `MODE_NOT_AVAILABLE`, ... | `AutonomousGraspReport.outcome` |
| `GraspMode` | `easy`, `auto`, `dense_clutter`, `closed_loop`, `dense_autonomous`; chosen when the service is built | `simulation/06_grasp_modes_and_what_each_needs.py` |
| `PickPrompt` | What every camera grounds, the labels it maps onto and the filter; `service.set_prompt(text)` | `PickRun.from_cell(cell, prompt=...)` |
| `PickRun` | N picks against one cell under one connect: `PickRun.from_cell(cell, runs=...).execute()`; `look=` the joints each pick looks from, `put_back=True` to put each part back | `real_robot/11_pick_with_the_camera.py` |
| `PickAttempt` | One pick of a campaign as `on_attempt` gets it: the outcome, the looks tried, `object_mm` (where the object was seen, BASE), `grasp_pose` (where the tool closed) and the put back | `real_robot/11_pick_with_the_camera.py` |
| `PickRunReport` | What a campaign did, whether it passed its rule, and how the cell came down; `exit_code` | `real_robot/11_pick_with_the_camera.py` |
| `PassRule` | When a campaign passes: a `fraction` of attempts, and `confirm=` for a check of your own | `real_robot/11_pick_with_the_camera.py` |
| `Recording` | Where a campaign appends one attempt record per line: `Recording.to_file(path)` or `Recording.off()` | `real_robot/11_pick_with_the_camera.py` |
| `RecordLog` | A record log read back into KPIs, with the audit that says whether they rest on anything | `simulation/05_measure_a_campaign.py` |
| `GraspMotion` | What a caller may choose about how a pick moves; a field left unset keeps the service's own | `real_robot/11_pick_with_the_camera.py` |
| `PlannerStart` | One cell's planner, started and stopped at a desk with no controller | `cell.start_planner()` in `real_robot/02_check_the_cell_at_a_desk.py` |

### What a build or a connect refuses with

| Name | Raised when | Shown in |
|---|---|---|
| `CameraWorldRequired` | A cell that plans with cuRobo is handed calibrated cameras that build it no live world | `Robot.from_tree(tree, cameras=...)` |
| `WristBodyRequired` | A cell that reads geometry cannot place, or was not told, the body of a camera its arm carries | `Robot.from_tree(tree, cameras=...)` |
| `LockKeyRequired` | An arm that drives a controller is handed in and no lock key derives from it; pass `lock_key=` | `Robot.from_parts(...)` |
| `CellBusy` | Another process already holds the cell; the message names the holder | `connected()` |
| `NoRealGripper` | The tree names a hand this arm cannot drive, and a connect was asked for | `offline/config/which_gripper_gets_built.py` |
| `CellNotBuilt` | A `Cell` step that needs the built cell ran before `build()` | `cell.connected()` |

### Cameras, calibration and where an object is

| Name | What it is | Shown in |
|---|---|---|
| `Camera` | The owner of one rig's device, used as `with Camera.from_tree(tree) as camera:`; `grab()`; `rig_id=` names another | `real_robot/06_open_a_camera.py` |
| `CameraWorldPlan` | Which rigs feed the live planner world and, one line each, why the others do not; opens nothing | `real_robot/14_a_cell_with_several_cameras.py` |
| `RGBDFrame` | Colour (BGR uint8) and depth (uint16 millimetres) from one grab | `real_robot/06_open_a_camera.py` |
| `CameraRefused` | Raised when the camera section cannot give a rig: not configured, switched off, or no depth | `Camera.from_tree(tree)` |
| `RigCalibrationError` | Raised by `camera.calibration()` when the declared calibration does not load, such as a block written before the sweep that writes its artifact; `RigNotCalibrated` is one of these | `real_robot/06_open_a_camera.py` |
| `RigNotCalibrated` | Raised when a rig is asked for its calibration and declares none | `camera.calibration()` |
| `HandEyeCalibration` | One camera calibrated against its robot: `check()` at a desk, `run()` at the cell; the stations are your own (`fixed_poses`) or you guide the arm to each pose by hand (`freedrive`) | `real_robot/07_calibrate_a_fixed_camera.py` |
| `SweepOptions` | What a caller may choose about one calibration sweep: `fixed_poses`, `freedrive`, `adjust` (fine-tune each fixed station by hand), `samples`, the target and the preview; unset takes the tree's value | `real_robot/10_calibrate_a_wrist_camera_with_fixed_poses.py` |
| `print_sweep_progress` | A sweep's `on_event`: one line per pose as it happens, where the arm goes, what the camera saw, and whether the pose counted or why not | `real_robot/07_calibrate_a_fixed_camera.py` |
| `Locator` | An open camera and a perception backend that place what they see in the robot's base frame | `real_robot/13_speak_pick_and_hand_handover.py` |
| `Located` | What one frame located: `objects`, `scene(i, robot)` for grasps, `keep_out(i)` for the planner | `real_robot/13_speak_pick_and_hand_handover.py` |
| `LocatorRefused` | Raised when a locator cannot place what its camera sees, before it grabs or on a frame | `Locator.from_tree(...)`, `locate()` |

### Hands seen by a camera

MediaPipe, optional and standalone: nothing in the grasp pipeline imports it, and each builder needs
its own `.task` file, named by `models.handdetect.model_path` and fetched separately. The finder
answers in the robot's base frame, so a pose built from it is a pose the arm's guards can check;
it needs a FIXED camera, and refuses a wrist rig rather than composing a transform of its own.

| Name | What it is | Shown in |
|---|---|---|
| `build_hand_finder_on_camera` | Where a hand is in MILLIMETRES in the base frame, over a camera you already hold open; `find_hand()` | `real_robot/13_speak_pick_and_hand_handover.py` |
| `build_gesture_recognizer` | Thumbs up or down, with the palm centre from the same pass; `observe(frame_bgr)` | [`src/models/handdetection/`](../src/models/handdetection/README.md) |
| `HandGesture` | What a reading may be: `THUMB_UP`, `THUMB_DOWN`, `OTHER`, `NONE`; the last two are not the same | `build_gesture_recognizer(...).observe(frame)` |
| `build_palm_detector` | Where hands are in a colour frame, in pixels, with no gesture; `observe(frame_bgr)` | [`src/models/handdetection/`](../src/models/handdetection/README.md) |

### Grasps, and the stacks a desk can evaluate without a robot

| Name | What it is | Shown in |
|---|---|---|
| `Scene` | A segmented target cloud on a support surface in base millimetres; `grasps()` ranks jaw grasps | `offline/grasping/grasps_for_a_cloud.py` |
| `synthesize_suction_grasps` | Ranked suction candidates for one segmented object | `offline/grasping/jaw_or_suction.py` |
| `build_calculator` | The grasp generator a tree's `robot.grasping.calculator` asks for, `geometric` or `deep` | `offline/grasping/select_grasp_generator.py` |
| `preflight_calculator` | Checks that selector without building anything, and returns what it chose | `offline/grasping/select_grasp_generator.py` |
| `PerceptionSpec` | The perception stack a tree builds, resolved before a weight loads; `resolve()`, `build()` | `offline/perception/resolve_perception_stack.py` |
| `RuleBasedRouter` | Routes a prompt by its text alone, and loads no model | `offline/perception/route_hard_prompts.py` |
| `MotionStack` | Which robot a cell is and which motion engines this machine holds for it | `offline/config/planner_or_ik.py` |

### Speech: push to talk, and a person confirms

| Name | What it is | Shown in |
|---|---|---|
| `shared_speech` | The process's one speech holder, which every caller shares | `real_robot/12_speak_a_command.py` |
| `TalkButton` | A talk switch pressed and released in software | `real_robot/12_speak_a_command.py` |
| `PushToTalkSource` | The microphone, serving audio only while the talk switch is held | `real_robot/12_speak_a_command.py` |
| `TerminalConfirmer` | Asks the person at the terminal whether the heard words become the prompt | `real_robot/12_speak_a_command.py` |
| `Confirmation` | Whether a person let proposed words become a prompt, and which words | `real_robot/12_speak_a_command.py` |
| `Listener` | A voice stream cut into utterances; `listen()` | [`src/models/speech/`](../src/models/speech/README.md) |

### Isaac Sim, under Isaac's own interpreter

| Name | What it is | Shown in |
|---|---|---|
| `run_gate` | Known-pose picks, each scored against the scene's ground truth, and the gate verdict | `simulation/03_isaac_pick_rate.py` |
| `record_demo` | One wrist-camera pick, filmed as an MP4 | `simulation/04_isaac_record_a_pick.py` |

### Offline: scenes, a dataset, your own parts and a trained generator

| Name | What it is | Shown in |
|---|---|---|
| `DatasetBuild` | Describe a dataset, then render, label and extract it | `offline/datagen/02_choose_an_engine.py` |
| `layout_scene` | One scene of a dataset, decided from the seed alone before anything renders | `offline/datagen/01_plan_scenes.py` |
| `engine_is_available` | Whether an engine can run in this interpreter, and why not | `offline/datagen/02_choose_an_engine.py` |
| `PhysicsSampling` | A labelled dataset and the simulator that grades its grasps | `offline/datagen/03_label_and_shake.py` |
| `import_from_directory` | Copies your own meshes into the library under the licence you declare | `offline/datagen/05_bring_your_own_parts.py` |
| `MeshPreparation` | Your mesh collections, fetched, normalised, screened, decomposed and diagnosed | `offline/datagen/06_fetch_public_parts.py` |
| `available_sources` | Every public mesh collection, its size, its licence and whether that was checked per model | `offline/datagen/06_fetch_public_parts.py` |
| `verify_dataset` | Checks a written dataset against itself, by path; `DatasetBuild.verify()` is the same check | `offline/datagen/08_verify_a_dataset.py` |
| `held_out_assets` | Which placeable assets a corpus never trained on, the denominator a held-out claim needs | `offline/training/03_prove_it_never_saw_the_test_parts.py` |
| `GeneratorTraining` | Fits a grasp generator from a recipe or a plan; `probe()`, `train()`, then `write_report()` | `offline/training/02_train_on_your_own_meshes.py` |
| `PlanOverrides` | The training settings you choose explicitly; they outrank the recipe and the tier | `offline/training/01_recipe_and_tier.py` |
| `PublicCorpus` | A published grasp corpus, read into the scene files this training loop already eats | `offline/training/04_train_on_a_public_corpus.py` |

## How it is kept honest

[`willy/__init__.py`](__init__.py) maps each name to the module that defines it, and `willy.__all__`
is that list. [`tests/test_willy.py`](../tests/test_willy.py) holds that every name is its home
module's own object, that importing `willy` loads nothing, and that every report which renders prints
the same text. CI type-checks `willy` and every example against these signatures, including the
`real_robot` files it never runs. How far each capability is proven is in the root README's
[Status and honest scope](../README.md#status-and-honest-scope).
