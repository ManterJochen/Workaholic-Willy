# Isaac Sim harness

The on-box validation and demo cell: it wires the vendor-neutral grasping stack to the Isaac
driver and drives a real `AutonomousGraspService.pick()` inside NVIDIA Isaac Sim.

This is a harness, not a shipped runtime. It sits above both `robot/drivers` and `robot/grasping`,
so it may import from each: its perception sources produce a `grasping.PerceptionFrame`, which
`drivers` is forbidden to import.

## What the simulator needs that this repository does not ship

Three external systems, none of them a dependency of the library:

- **Isaac Sim**, a multi-gigabyte standalone install with its own bundled Python, which stays
  wherever its installer put it. Every runner needs that interpreter, not this repository's
  virtual environment.
- **cuRobo**, the collision-aware motion planner, reached over a stdio sidecar and pointed at by
  `WILLY_CUROBO_PYTHON`.
- **Coal**, the exact-mesh self-collision engine, pointed at by `WILLY_COAL_PREFIX`.

The last two install into [`ext_deps/`](../../ext_deps/README.md) with
`scripts/ext_deps/install.ps1`, which touches nothing outside that directory.

The `sim` profile configures `motion_planner: curobo` and `self_collision.backend: fcl`, so
`bootstrap_sim_cell` probes for both before Isaac starts and refuses to boot when either is
missing. That refusal is the point: blind IK proposes self-colliding branches and the capsule proxy
passes configurations the mesh check rejects, so a missing engine replaces the safety argument
rather than weakening it, and any rate measured that way describes a different system.
`WILLY_ALLOW_DEGRADED_MOTION=1` runs anyway, prints the banner every time, logs an error, and puts
the missing engines on `SimCell.degraded_engines`, which a runner can stamp onto its result JSON as
`planner_degraded`.

`python -m src.robot.safety.planning --doctor` reports which of the two are installed, without
Isaac.

The Isaac imports themselves stay lazy, after the `mock_mode` check, so the package imports on a
machine with no Isaac at all and the mock suite covers the parts that do not need it.

## The gate rule

Each pick runner shares the boot prefix and the gate primitives, runs N picks, prints a gate line
naming how many of the N passed, and exits with a meaningful code.

A run passes when `pick()` reports success and, independently, the object's measured world-Z rose
by at least `robot.sim.gate.lift_threshold_mm`, which the tree sets to 50.0. The service's own
outcome is never sufficient on its own. The campaign passes when the number of passing runs reaches
`int(pass_fraction * runs)`, and at least one, with `pass_fraction` set to 0.8. Those two numbers
are the difference between this gate and the unanimity rule the real-cell runner applies.

The known-pose gate has been run on an Isaac workstation against this tree and passed. It is not a
close call by construction: `run_m1_pick` retreats 100.0 mm after the grasp, twice the threshold the
gate applies, so a run that grips at all clears it and a run that fails is a failure to grip rather
than a marginal lift. Every other runner's numbers are yours to measure on your own cell.

## The runners

| Runner | Scenario |
| --- | --- |
| `run_m1_pick.py` | known-pose pick with ground-truth perception, so a failure is motion or geometry |
| `run_m2_pick.py` | real-vision pick, with the detector and segmenter in the loop |
| `run_eih_pick.py` | eye-in-hand pick: the camera rides the wrist and re-perceives from where it moved |
| `run_dense_pick.py` | dense-clutter pick, with `--vision` and `--mode dense_autonomous` |
| `run_fused_pick.py` | dual-camera fused pick: overhead coarse scan, then wrist refine and grasp |
| `run_multiview_pick.py` | fixed-camera localize, wrist refine, grasp. `--mode` selects the active fixed-camera set (`eth1`, `eth2`, `eth3`, or `sides`, the two-side-camera industrial rig) |
| `run_industrial_bin_pick.py` | two side cameras localize the prompted part in a tray of mixed parts |
| `run_attribute_pick.py` | four objects where the noun alone is never enough, as a route comparison |
| `run_eth_calibrate.py`, `run_eih_calibrate.py` | eye-to-hand and eye-in-hand calibration through the real `CalibrationRoutine` |

Beyond the pick gates: cinematic recorders (`run_dense_demo`, `run_dense_demo_endgame`,
`run_sorting_demo`, `run_bin_clearing_demo`, `run_klt_combined_demo`, `run_clutter_demo`,
`run_eih_demo`, `run_expose_pick`), suction runners (`run_suction_pick`,
`run_industrial_suction_pick`, `run_suction_demo`, `run_suction_probe`), and probes and matrices
(`inspect_wrist_cam`, `run_commit_gate`, `run_mode_matrix`, `run_occlusion_probe`,
`run_pile_baseline`, `run_shake_label`, `run_curobo_demo`).

```bash
# Isaac's bundled interpreter, from the repository root:
<isaac-sim>\python.bat -m src.willy_sim.run_m1_pick --runs 10
```

`scripts/examples/07_sim.py` fronts the first three and checks the interpreter before anything
else, which is the single most common way an hour disappears here.

## Contents

| Path | Role |
| --- | --- |
| `config.py` | `load_sim_config`, which loads the repository `config` tree under the `sim` profile, plus `require_robot`, `sim_driver_config`, `sim_safety_preflight` and `sim_profile_chain` |
| `harness/` | The shared runner seams, below |
| `perception/` | `PerceptionSource` implementations: `ground_truth.py` (single and multi-object, runnable on CPU) and `vision.py` (the real detector and segmenter, on box) |
| `scene/` | Isaac scene authoring: `build.py`, `cameras.py` (overhead and wrist, with their frame transforms), `markers.py`, `constants.py` |
| `calibration/` | `hand_eye.py`, the marker-pose sources and hemisphere viewpoints around the real `CalibrationRoutine`, and `paths.py` |
| `grippers.py` | Which end-effector to mount, jaw or suction, as config-selected data |
| `gso_assets.py` | Loader that converts scanned-object meshes to USD for realistic scenes |
| `suction_mount.py` | Authors an Isaac surface-gripper suction anchor on the wrist |
| `shake_label.py` | Native physics shake-test labeller, held or dropped, via kicks and gravity overload |

The seams in `harness/`:

| Seam | What it gives |
| --- | --- |
| `bootstrap_sim_cell(...) -> SimCell` | The shared boot prefix: load the sim-profile config, audit reach and camera coverage, probe the motion stack, build a fail-closed `IsaacRobotArm`, start the session, author the scene, connect the arm and then the gripper. `SimCell` exposes `arm`, `gripper`, `handles`, `cfg`, `robot`, `sim`, `dwell` and `degraded_engines`. |
| `GateResult` and helpers | `GateResult.to_dict()` emits exactly the key set its runner expects, plus `reset_object_to_home_z0()`, `lift_mm_since()` and `gate_passed()`. The per-pick loop and the scoring stay per-runner, deliberately not shared. |
| `RunnerEnv.from_env(...)` | The 17 `WILLY_*` runner knobs in one place, with their context-aware defaults preserved. |
| `mode_service_kwargs`, `resolve_demo_mode` | Map a mode string to a typed `GraspMode` plus the sub-policy arguments it needs. |
| `reach.py`, `coverage.py` | Geometry checks that run before the Isaac boot: whether a configured point is inside the arm's reach sphere, and whether the scene is inside the camera frame. |
| `cli.py` | The `--robot-model` and `--profile` arguments every runner shares. |
| `DomainRandomizer`, `ArmBackedIKService`, `instrumentation`, `depth_noise` | A seeded randomizer stamped into the record, a singularity-aware IK service backing the grasping veto seam, opt-in per-pick record and image dumps, and synthetic noise on Isaac's noise-free depth. |

## Launch discipline on a workstation

| Rule | Why |
| --- | --- |
| One single file-redirected `cmd /c "...python.bat... > log 2>&1"` | `cmd` writes UTF-8 and PowerShell redirection writes UTF-16 |
| Never a compound command | it hangs the boot |
| One Isaac process at a time | they contend for the GPU and the asset cache |
| Export `WILLY_CUROBO_PYTHON` and `WILLY_COAL_PREFIX` before launching | otherwise the boot refuses, and every runner prints the anchoring status so a degraded run is obvious rather than mistaken for a grasp-quality problem |

## Traps

- **Most runners build through `from_components`, not `from_robot_config`,** because `IsaacGripper`
  is not in the gripper registry and needs the arm's session. That leaves `effective_config=None`
  and silences the config-driven overlays, so those runners re-enable them in runner code through
  `build_effective_config` and `apply_orchestrator_overlays`. Two runners can take the other path
  by handing the live gripper handle to `from_robot_config`, which is the path a real cell takes:
  `run_multiview_pick` defaults to it (`--boot config`, with `--boot components` kept as the
  comparison that must agree), and `run_eih_pick.build_service` takes `service_from_config=True`
  from Python.
- **The overhead camera's near clip is deliberate.** The sim tree leaves `near_clip_m` unset for
  the nadir camera, so it keeps Isaac's 1.0 m default, which hides the arm from the instance mask.
  A clean mask and the object in the depth render are mutually exclusive there, and mutating a
  camera after its annotators are attached kills them. Read the clipping range back off the camera
  before believing a value was applied.
- **The detector runs fp32 weights in the simulator.** The `sim` model overlays leave the dtype
  unset so the weights load fp32 under fp16 autocast, which is what holds recall on small objects
  in the overhead view.
- **On-box only for the runners.** They need a real Isaac install and are excluded from the
  coverage measurement. The CPU-runnable core, meaning the ground-truth perception path, the
  randomizer, the IK service, the gate, the environment knobs, the modes, the hand-eye calibration,
  the config module and the scene module, is what the mock suite covers.
- **Record logging is opt-in**, through `enable_record_logging(path)`, so the offline soak, KPI and
  learning layers get real simulator data only when a run asks for it.
- **Where the logs go.** Each library module writes a rotating file under `logs/willy_sim/`, and
  every filename is declared in `constants.py`. The pick and demo runners deliberately do not: their
  per-run narrative stays on stdout, which lands in that run's redirect and is therefore
  attributable to one run, where a shared rotating file is not. The exceptions are the runners that
  leave an artifact behind, the two calibrations, the mode matrix, the shake labeller, the pile
  baseline and the occlusion probe, which log the conditions their artifact cannot carry.
- **A simulator rate proves the software, never the cell.** Contact friction is a model, the depth
  is a perfect sensor, a gripper that closes cleanly here can slip on a real surface, and a
  simulated camera's near clip hides things no real camera hides.

## See also

- [Workaholic-Willy](../../README.md), the repository overview
- [robot/drivers/sim](../robot/drivers/sim/README.md), the Isaac driver this cell mounts
- [robot/execution/autonomous_grasp](../robot/execution/autonomous_grasp/README.md), the pick path the runners drive
- [robot/grasping](../robot/grasping/README.md), the vendor-neutral grasp stack under test
- [ext_deps](../../ext_deps/README.md), how to install cuRobo and Coal
- `scripts/examples/07_sim.py`, the readiness check and a fronted run
