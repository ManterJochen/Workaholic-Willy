# Examples

Short programs that call Workaholic-Willy the way your own code would: `from willy import ...`, then the
library's own objects, and `print(report)` for what came back. They sit in three folders, by what has to
be attached for them to run.

| folder | what it is | what it needs |
|---|---|---|
| [`real_robot/`](real_robot/) | the library at your cell, numbered in the order a cell comes up | your cell: the arm, the hand, the cameras |
| [`simulation/`](simulation/) | the same calls on a dummy arm at a desk, and picks in Isaac Sim | nothing, or an Isaac Sim install |
| [`offline/`](offline/) | data generation, training, and the models a desk evaluates | nothing attached |

Install the repository once from its root with `pip install -e . --no-deps` (the requirements first, see
[guide 02](../docs/guide/02-models.md)); after that `from willy import ...` works from any directory.
Every name it gives you is listed, with the file that shows it, in
[`willy/README.md`](../willy/README.md).

## real_robot: your cell

Each file drives the cell your profile names. A profile is the name of your cell's layer in the config
tree, the `*.<profile>.yaml` files ([guide 01](../docs/guide/01-configuration.md)). Set it around the
command; no example sets it for you:

```bash
WILLY_PROFILE=<your cell> python examples/real_robot/01_load_your_cell.py
```

In PowerShell, `$env:WILLY_PROFILE = "<your cell>"` before the command does the same for that session.

Read a file before you run it. Start with 01 and 02, which move nothing. From 03 on the arm moves: poses
are millimetres in the robot's base frame, and the numbers in the files are placeholders to replace with
poses inside your own workspace. Every motion goes through the cell's safety checks, and a file that
moves the arm either hands the robot its cameras or declines the camera world and says why.

| file | what it shows | what moves |
|---|---|---|
| [`01_load_your_cell.py`](real_robot/01_load_your_cell.py) | load your cell's config tree, read one key, change one in memory | nothing |
| [`02_check_the_cell_at_a_desk.py`](real_robot/02_check_the_cell_at_a_desk.py) | the checklist a first connect has to pass, and whether the planner starts | nothing |
| [`03_connect_and_move.py`](real_robot/03_connect_and_move.py) | connect, go home, a move, a straight line, a joint move | the arm |
| [`04_open_and_close_the_hand.py`](real_robot/04_open_and_close_the_hand.py) | open the hand, close it on a part you place, read what the hand measured | the hand |
| [`05_pick_and_place_a_known_part.py`](real_robot/05_pick_and_place_a_known_part.py) | pick at a pose you know, place at another | the arm and the hand |
| [`06_open_a_camera.py`](real_robot/06_open_a_camera.py) | open the cell's camera: one frame, its lens matrix, its declared calibration; runs before the camera is calibrated | nothing |
| [`07_calibrate_a_fixed_camera.py`](real_robot/07_calibrate_a_fixed_camera.py) | where a fixed camera sits in the robot's frame, with you guiding the arm by hand (`freedrive`): the payload confirmed first, Enter captures once the arm stands still, the window red outside the cable window or the box, every counted pose written to a stations file that replays without hands | nothing by itself: you move the arm |
| [`08_calibrate_a_fixed_camera_with_fixed_poses.py`](real_robot/08_calibrate_a_fixed_camera_with_fixed_poses.py) | the same fixed-camera solve from poses you choose, each turning the board to FACE the camera rather than tool down, or from the stations 07 wrote | the arm, in the sweep |
| [`09_calibrate_a_wrist_camera.py`](real_robot/09_calibrate_a_wrist_camera.py) | a camera on the wrist, guided by hand over a board on the table: works wherever the camera sits on the tool, because you aim it by eye | nothing by itself: you move the arm |
| [`10_calibrate_a_wrist_camera_with_fixed_poses.py`](real_robot/10_calibrate_a_wrist_camera_with_fixed_poses.py) | the wrist-camera solve from stations of your own, in order: poses, the joint angles you taught on the pendant (`{"joints_deg": [...]}`, as in the template [`eih_fixed_stations.json`](real_robot/eih_fixed_stations.json)), or the stations 09 wrote; with `adjust` the arm stops at each station and you fine-tune it by hand, then hands off, Enter, and a 3 s countdown before it drives on | the arm, in the sweep, and you at each station |
| [`11_pick_with_the_camera.py`](real_robot/11_pick_with_the_camera.py) | a campaign of camera picks on one part: your own pick motion, look poses declared as joints in degrees and tried in order, the part put back where it was grasped after every lift, where the object and the grasp were printed per attempt, a verdict, a JSONL record and a saved overlay image per attempt (weights: `dino-tiny sam2`) | the arm and the hand |
| [`12_speak_a_command.py`](real_robot/12_speak_a_command.py) | a spoken command, confirmed by a person, becomes the pick prompt (weights: `whisper-turbo silero-vad`, and those of 11) | the arm and the hand |
| [`13_speak_pick_and_hand_handover.py`](real_robot/13_speak_pick_and_hand_handover.py) | speech (12), a camera pick step by step (`Locator`, the part's grasps, `robot.pick`), then the arm brings the part to where the camera sees the person's hand | the arm and the hand |
| [`14_a_cell_with_several_cameras.py`](real_robot/14_a_cell_with_several_cameras.py) | which rigs feed the live planner world and why the others do not, then one owner per device handed to the arm | the arm, and two or more cameras |

## simulation: no cell

`console_dummy` is the desk profile: a dummy arm and a dummy hand that command nothing. The Isaac files
need an Isaac Sim install and run under its own interpreter from the repository root, with the root on its path
so it finds `willy` (`$env:PYTHONPATH = (Get-Location).Path` in PowerShell, `set PYTHONPATH=%CD%` in cmd), then
`<isaac-sim>/python.bat examples/simulation/03_isaac_pick_rate.py` ([docs/isaac-ready.md](../docs/isaac-ready.md));
in any other interpreter they say so and exit.

| file | what it shows | needs |
|---|---|---|
| [`01_rehearse_a_pick.py`](simulation/01_rehearse_a_pick.py) | one pick on a dummy arm and a synthetic scene, with its verdict | nothing |
| [`02_a_robot_at_the_desk.py`](simulation/02_a_robot_at_the_desk.py) | the real_robot calls on the dummy arm: connect, move, the hand, pick and place | nothing |
| [`03_isaac_pick_rate.py`](simulation/03_isaac_pick_rate.py) | a pick rate, scored against the scene's ground truth | Isaac Sim |
| [`04_isaac_record_a_pick.py`](simulation/04_isaac_record_a_pick.py) | one pick, filmed as an MP4 | Isaac Sim |
| [`05_measure_a_campaign.py`](simulation/05_measure_a_campaign.py) | a campaign's record log rolled up into KPIs, and what those numbers rest on | nothing |
| [`06_grasp_modes_and_what_each_needs.py`](simulation/06_grasp_modes_and_what_each_needs.py) | the five grasp modes, what each switches on, and how one that is not wired refuses by name | nothing |

## offline: data, training and models at a desk

Offline means data generation, training and the models a desk evaluates, with no robot and no camera
attached. Run these from the repository root: the model and asset paths in the config tree are relative
to it, and each file names the config tree it reads. A file that needs something a bare machine lacks (model weights, a GPU, a generated corpus, the
network, an optional engine) checks for it, names what is missing in one sentence and exits.

| file | what it shows |
|---|---|
| [`config/which_gripper_gets_built.py`](offline/config/which_gripper_gets_built.py) | which hand your config really builds, and what stands in when it cannot |
| [`config/planner_or_ik.py`](offline/config/planner_or_ik.py) | a collision-free planner or the controller's straight line, and whether this machine has the planner |
| [`perception/resolve_perception_stack.py`](offline/perception/resolve_perception_stack.py) | which detector and segmenter your config builds, before a weight loads |
| [`perception/route_hard_prompts.py`](offline/perception/route_hard_prompts.py) | the prompts a phrase grounder answers wrongly, and where they go instead |
| [`safety/gate_the_whole_path.py`](offline/safety/gate_the_whole_path.py) | every waypoint of a path checked, not only where it ends |
| [`safety/self_collision_backend.py`](offline/safety/self_collision_backend.py) | exact meshes or the capsule proxy for self collision, and which one runs |
| [`grasping/grasps_for_a_cloud.py`](offline/grasping/grasps_for_a_cloud.py) | ranked grasps for an object's points, with no config tree at all |
| [`grasping/jaw_or_suction.py`](offline/grasping/jaw_or_suction.py) | grasps for a jaw and for a suction cup on the same part |
| [`grasping/select_grasp_generator.py`](offline/grasping/select_grasp_generator.py) | the geometric generator or the learned one, which refuses without weights |
| [`datagen/01_plan_scenes.py`](offline/datagen/01_plan_scenes.py) | what each scene holds, decided before anything renders |
| [`datagen/02_choose_an_engine.py`](offline/datagen/02_choose_an_engine.py) | which engine settles and renders the scenes, and which ones this machine can run |
| [`datagen/03_label_and_shake.py`](offline/datagen/03_label_and_shake.py) | grasp labels from geometry, and the physics screen that grades them |
| [`datagen/04_extract_a_corpus.py`](offline/datagen/04_extract_a_corpus.py) | the point cloud corpus a generator trains on |
| [`datagen/05_bring_your_own_parts.py`](offline/datagen/05_bring_your_own_parts.py) | your own parts in the mesh library, under the licence you declare |
| [`datagen/06_fetch_public_parts.py`](offline/datagen/06_fetch_public_parts.py) | the public mesh collections, what they cost to download and what they oblige you to |
| [`datagen/07_estimate_the_cost.py`](offline/datagen/07_estimate_the_cost.py) | hours, gigabytes and refused scenes, per engine, before anybody starts a build |
| [`datagen/08_verify_a_dataset.py`](offline/datagen/08_verify_a_dataset.py) | a written dataset checked against itself: labels, masks and pictures agreeing |
| [`training/01_recipe_and_tier.py`](offline/training/01_recipe_and_tier.py) | what a recipe and a tier resolve to before anything trains |
| [`training/02_train_on_your_own_meshes.py`](offline/training/02_train_on_your_own_meshes.py) | train a grasp generator on your own parts, and read what the run produced |
| [`training/03_prove_it_never_saw_the_test_parts.py`](offline/training/03_prove_it_never_saw_the_test_parts.py) | which placeable assets a corpus never trained on, the denominator a held-out claim needs |
| [`training/04_train_on_a_public_corpus.py`](offline/training/04_train_on_a_public_corpus.py) | train on a published grasp corpus, for a user with no simulator and no cell |
| [`training/05_floor_and_ceiling_before_you_train.py`](offline/training/05_floor_and_ceiling_before_you_train.py) | what a number on your corpus could possibly mean, measured with no weights at all |

## How they are kept working

CI type-checks every `real_robot` file against the library's own signatures and never runs one, because
each drives a real cell. It runs every `simulation` and `offline` file on a machine with nothing attached.

The same capabilities from a shell are in [docs/cli.md](../docs/cli.md), and the procedures that use them
are the [runbooks](../docs/runbooks/): [bringing up a cell](../docs/runbooks/cell_bringup.md) and
[the first pick on a physical arm](../docs/runbooks/real_cell_first_pick.md) first.
