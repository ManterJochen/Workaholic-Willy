# Isaac Sim picks and demos (`src/willy_sim`)

This package runs the real pick service inside NVIDIA Isaac Sim on a UR5e cell: known-pose, real-vision,
eye-in-hand, multi-view and suction picks, hand-eye calibration, and the recorders that film them. It is a test
and demo harness, not a runtime, and a simulator rate proves the software, never the cell.

```python
from willy import record_demo, run_gate

result = run_gate(runs=10, headless=True, mode="easy")    # ten known-pose picks, each scored on the part's rise
print(f"{result.passed} of {result.runs} passed, gate passed: {result.gate_passed}")

film = record_demo(out_path="logs/demo/pick.mp4", fps=18)  # one wrist-camera pick, filmed as an MP4
print(film["out"], film["lift_mm"])
```

Everything here runs under Isaac's own interpreter, from the repository root; the project's virtual environment
cannot import Isaac. [03_isaac_pick_rate.py](../../examples/simulation/03_isaac_pick_rate.py) and
[04_isaac_record_a_pick.py](../../examples/simulation/04_isaac_record_a_pick.py) make these two calls and check
the interpreter first. The runners are the same from a shell:

```bash
<isaac-sim>/python.bat -m src.willy_sim.run_m1_pick --runs 10 --result-json logs/m1.json
<isaac-sim>/python.bat -m src.willy_sim.run_m2_pick --runs 5 --prompt "a red cube"
<isaac-sim>/python.bat -m src.willy_sim.run_eih_demo --fps 18 --out logs/demo/eih.mp4
```

A pick runner's verdict is its `GATE:` line and the JSON `--result-json` writes. No pick runner's exit code is
that verdict: most exit 0 whatever the gate said, so a script reads the JSON. `run_m1_pick --mode` takes
`easy`, `auto` (the decision gate before every grasp) or `closed_loop` (refine and verify each one).
`run_m1_pick`, `run_m2_pick`, `run_eih_pick`, `run_multiview_pick`, `run_attribute_pick` and the two
calibrations take `--robot-model <model>` and `--profile <layer>`, which load the chain `sim,<model>,<layer>`,
and `--hand <name>`, which runs a registry hand in place of the one the tree names.
[isaac-ready.md](../../docs/isaac-ready.md) turns a fresh workstation into one that runs all of this.

## The gate rule

A run passes when `pick()` reports success and, checked separately, the part's world height rose by at least
`robot.sim.scene_setup.gate.lift_threshold_mm`, 50.0 in the sim profile. The service's own outcome is never
enough on its own. A campaign passes when the passing runs reach `int(pass_fraction * runs)`, and at least one,
with `pass_fraction` 0.8. A real cell's `PickRun` defaults to unanimity instead (`PassRule`).
`run_m1_pick` retreats 100.0 mm after the grasp, twice the threshold, so a run that grips at all clears it.

## What it needs, and what it refuses

The sim profile plans with cuRobo and checks self-collision on exact meshes with Coal. Both install into
[ext_deps/](../../ext_deps/README.md) with `scripts/ext_deps/install.ps1`, and
`python -m src.robot.safety.planning --doctor` reports what is installed without starting Isaac.

| Refusal | When | What to do |
|---|---|---|
| the boot, before Isaac starts | cuRobo or Coal cannot be found | install them; `WILLY_ALLOW_DEGRADED_MOTION=1` runs anyway, marked degraded |
| a cuRobo cell with no camera world | the runner states neither a decline nor a live world | pass `camera_world=`: a `CameraWorldDecline`, or `SimCameraWorld("overhead")` |
| a policy pick on an `ik` or `rmpflow` arm | before the jaws open: those planners keep no straight line | plan with `curobo`, the sim profile's default |
| a profile layer that does not claim its robot | `--robot-model ur3e` with a layer that says otherwise | set `robot.sim.robot_model` in that layer |
| a hand with no Isaac mount | `--hand` or the tree names a hand that exists only on a real cell | refused before the boot; pick a hand `grippers.py` mounts |
| a multi-view, fused or industrial pick | no calibrated eye-in-hand artifact for the marker | run `run_eih_calibrate` first; it writes `logs/calibration/<model>/` |

A degraded run prints a banner every time and logs an error, because a rate measured on blind IK or the capsule
proxy describes a different system; `run_m2_pick` and `run_attribute_pick` also stamp `planner_degraded` into
their result JSON.

## The runners

| Runner | Scenario |
|---|---|
| `run_m1_pick` | known-pose pick with ground-truth perception, so a failure is motion or geometry |
| `run_m2_pick` | real vision: detector and segmenter in the loop, planning against a live world from the overhead camera |
| `run_eih_pick` | eye-in-hand: the camera rides the wrist and perceives again from where it moved |
| `run_dense_pick` | dense clutter; `--vision` for real perception, `--mode dense_autonomous` for the full loop |
| `run_fused_pick` | overhead coarse scan, then wrist refine and grasp |
| `run_multiview_pick` | fixed cameras localize, the wrist refines; `--mode eth1`, `eth2`, `eth3` or `sides` |
| `run_industrial_bin_pick` | two side cameras find the prompted part in a tray of mixed parts |
| `run_attribute_pick` | four objects the noun alone cannot tell apart, compared per route |
| `run_eth_calibrate`, `run_eih_calibrate` | eye-to-hand and eye-in-hand calibration through the real `CalibrationRoutine` |
| `run_suction_pick`, `run_industrial_suction_pick` | a suction cup on the wrist: one part, and a wide flat package the jaw cannot span |

`run_m2_pick --decline-camera-world "<why>"` boots the control run without the live world. The film recorders
are `run_eih_demo`, `run_dense_demo`, `run_dense_demo_endgame`, `run_sorting_demo`, `run_bin_clearing_demo`,
`run_klt_combined_demo`, `run_clutter_demo`, `run_expose_pick`, `run_suction_demo` and
`run_industrial_suction_demo`. The probes and matrices are `inspect_wrist_cam`, `run_commit_gate`,
`run_mode_matrix`, `run_occlusion_probe`, `run_pile_baseline`, `run_shake_label`, `run_suction_probe` and
`run_curobo_demo`.

## On a workstation

| Rule | Why |
|---|---|
| one file-redirected `cmd /c "<isaac-sim>/python.bat ... > log 2>&1"` | `cmd` writes UTF-8 where PowerShell redirection writes UTF-16 |
| never a compound command | it hangs the boot |
| one Isaac process at a time | they contend for the GPU and the asset cache |
| export `WILLY_CUROBO_PYTHON` and `WILLY_COAL_PREFIX` before launching | otherwise a cuRobo cell refuses at the boot |

- **Most runners build through `from_components`**, because the Isaac gripper is not in the gripper registry and
  needs the arm's session, so they switch the grasping overlays on in runner code. `run_multiview_pick` builds
  through `from_robot_config` by default (`--boot config`, with `--boot components` as the comparison), which
  is the path a real cell takes, and `run_eih_pick.build_service(service_from_config=True)` does from Python.
- **The overhead camera keeps Isaac's 1.0 m near clip on purpose.** It hides the arm from the instance mask;
  read the clipping range back off the camera before believing a value was applied.
- **The detector runs fp32 weights in the simulator.** The sim model overlays leave the dtype unset, which
  holds recall on small objects in the overhead view.
- **Record logging is opt-in**: `run_m1_pick --record-log <file>` appends one `GraspAttemptRecord` per pick.
- **Logs.** Library modules write under `logs/willy_sim/`. The runners print to stdout, so a run's narrative
  lands in that run's redirect; the ones that leave an artifact (the calibrations, the mode matrix, the shake
  labeller, the pile baseline, the occlusion probe) also log the conditions the artifact cannot carry.

## Status

| Capability | Evidence |
|---|---|
| Known-pose gate, `run_m1_pick` | measured in simulation: run on an Isaac workstation against this tree, and passed |
| Real-vision, eye-in-hand, multi-view and suction picks, and calibration | measured in simulation; rates on your cell are yours to measure |
| Any of it on a physical cell | never touched hardware: contact is a model and the depth is a perfect sensor |

Without Isaac, the mock suite (`tests/test_willy_sim_*.py`) covers the CPU half: ground-truth perception, the
randomizer, the IK service, the gate, the runner knobs, the modes, hand-eye calibration, the config and the scene.

## Files

| Path | Holds |
|---|---|
| [config.py](config.py) | `load_sim_config`, which loads the `config` tree under the `sim` profile, and its helpers |
| [harness/](harness/) | the shared seams below |
| [perception/](perception/) | ground-truth and real-vision perception sources, and the overhead camera as a live world's camera |
| [scene/](scene/) | Isaac scene authoring: the cell, the cameras and their frames, the markers |
| [calibration/](calibration/) | marker-pose sources and viewpoints around the real `CalibrationRoutine` |
| [grippers.py](grippers.py) | which jaw or suction cup Isaac mounts, following `robot.gripper.model` |
| [gso_assets.py](gso_assets.py), [suction_mount.py](suction_mount.py), [shake_label.py](shake_label.py) | scanned-object USDs, the suction anchor, the shake labeller |

| Seam in `harness/` | Gives |
|---|---|
| `bootstrap_sim_cell(...) -> SimCell` | the shared boot: config, reach and camera audits, motion stack probe, arm, scene, gripper |
| `GateResult`, `gate_passed`, `lift_mm_since` | the verdict type the pick runners return, and the gate arithmetic |
| `SimCameraWorld`, `CellCameraWorld` | a live camera world from the overhead camera, or the decline a cell holds |
| `RunnerEnv.from_env(...)` | the `WILLY_*` runner knobs in one place |
| `mode_service_kwargs`, `resolve_demo_mode` | a mode string to a typed `GraspMode` and its sub-policy arguments |
| `reach.py`, `coverage.py`, `cli.py` | the reach and camera checks before the boot, and the arguments the runners share |
| `DomainRandomizer`, `ArmBackedIKService`, `depth_noise` | seeded randomization, the IK service behind the veto seam, noise on perfect depth |

## Details

- [docs/isaac-ready.md](../../docs/isaac-ready.md), the workstation setup, and [docs/cli.md](../../docs/cli.md),
  the commands.
- [robot/drivers/sim](../robot/drivers/sim/README.md), the Isaac driver;
  [robot/execution/autonomous_grasp](../robot/execution/autonomous_grasp/README.md), the pick path the runners
  drive; [robot/grasping](../robot/grasping/README.md), the grasp stack under test.
- [Guide 05](../../docs/guide/05-pick-loop.md), the pick loop in the simulator.
