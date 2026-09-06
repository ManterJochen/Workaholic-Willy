# Make Isaac ready, and run the demos

Turn a fresh NVIDIA workstation into a box that runs the Isaac Sim picks and the cinematic demo
recorders. This is the top-down checklist; the sidecar internals are in
[`ext_deps/README.md`](../ext_deps/README.md), sections on
[Coal](../ext_deps/README.md#coal-the-exact-mesh-self-collision-engine) and on
[cuRobo](../ext_deps/README.md#curobo-the-collision-aware-motion-planner), and the runner catalogue
is in [`src/willy_sim/`](../src/willy_sim/README.md).

## 0. What you need

| | |
|---|---|
| GPU | An NVIDIA RTX card. |
| OS | Windows 10 or 11, or Linux. |
| Isaac Sim | 5.1 standalone, the bundled-Python distribution rather than the pip package. |
| Disk | Isaac and its assets, plus the two sidecar conda environments. |

The library core and the offline suite are torch-free in the sense that matters here: they need none
of this. This page is only for the on-box, full-motion picks and demos.

## 1. Install Isaac Sim 5.1 standalone

Download the standalone package and unpack it. Everything is driven through its bundled Python,
`python.bat` on Windows and `python.sh` on Linux, which already carries `isaacsim`, `omni.*` and a
matching torch. The repository's own virtual environment cannot import `isaacsim` however the
configuration is set, so a runner started with the wrong interpreter fails eight imports deep.
`scripts/examples/sim/70_sim_pick.py` checks that first and refuses with the fix.

Keep the whole stack on one fast drive.

### Assets

Isaac needs its asset pack, which carries the robot USDs and materials; the UR5e and the Robotiq
2F-85 come from it. `robot.sim.assets_root` in
[`config/robot/robot.sim.yaml`](../config/robot/robot.sim.yaml) ships as `null`, which asks Isaac
itself through `get_assets_root_path()` and is correct on a normal install. Set it only where the
assets sit somewhere Isaac does not look, and prefer the environment variable form
`assets_root: ${WILLY_ISAAC_ASSETS_ROOT:-}` over a literal so the file stays portable.

## 2. The two motion sidecars

Motion is planned with cuRobo and self-collision is gated on exact meshes with Coal. Neither can live
inside the Isaac process, because cuRobo wants warp 1.14 while Isaac ships 1.8.2 and one Python
process holds exactly one warp, and because there is no Windows Coal wheel. Each therefore runs in
its own conda environment, located by an environment variable with a default inside `ext_deps/`.

One command installs both, into the paths the code looks in by default, so no environment variable is
needed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\ext_deps\install.ps1
```

Set `WILLY_CUROBO_PYTHON` or `WILLY_COAL_PREFIX` only if you installed them somewhere other than
`ext_deps/`.

Verify both are anchored before a real run:

```bash
python -m src.robot.safety.planning --doctor
#   0 = healthy: both engines LOAD (Coal ran a real query, the sidecar imported cuRobo)
#   1 = degraded: at least one engine is missing
#   2 = an OS application-control policy is blocking a binary; see docs/code-integrity.md

python -m src.robot.safety.planning --check   # the cheap path-only reading, milliseconds
```

**A missing engine does not degrade quietly here.** The `sim` profile configures
`motion_planner: curobo` and `self_collision.backend: fcl`, so `bootstrap_sim_cell` probes for both
before Isaac starts and refuses to boot when either is absent. That refusal is the point: blind IK
proposes self-colliding branches and the capsule proxy passes configurations the mesh check rejects,
so a missing engine replaces the safety argument rather than weakening it, and any rate measured that
way describes a different system. `WILLY_ALLOW_DEGRADED_MOTION=1` runs anyway, prints the banner
every time, logs an error, and puts the missing engines on `SimCell.degraded_engines` so a runner can
stamp them onto its result.

## 3. Launch discipline

The Isaac runners are long-lived GPU processes. Two rules keep them reliable on Windows.

1. **One single, file-redirected command**, always
   `cmd /c "...python.bat -m <module> ... > run.log 2>&1"`. `cmd` gives UTF-8 logs where PowerShell
   redirection writes UTF-16 and corrupts them, and a single command avoids the boot hang that a
   compound command causes.
2. **One Isaac process at a time.** Kill a stale `kit.exe` before the next launch with
   `taskkill /f /im kit.exe`. They contend for the GPU and the asset cache.

## 4. Run a pick

```bash
# a deterministic known-pose pick, the fastest reading of whether the box is ready
cmd /c "<isaac-sim>\python.bat -m src.willy_sim.run_m1_pick --runs 10 > m1.log 2>&1"
```

Each pick runner prints a line of the form `GATE: <n>/<runs> passed -> gate_passed=<bool>` and exits
with a meaningful code. A run passes when `pick()` reports success and, independently, the object's
measured world Z rose by at least `robot.sim.gate.lift_threshold_mm`, which the tree sets to 50.0;
the service's own outcome is never sufficient on its own. The campaign passes when the number of
passing runs reaches `int(pass_fraction * runs)` and is at least one, with `pass_fraction` set to
0.8.

Then the real-vision pick, which downloads the detector and segmenter weights on first run:

```bash
cmd /c "<isaac-sim>\python.bat -m src.willy_sim.run_m2_pick --runs 10 > m2.log 2>&1"
```

The full catalogue, eye-in-hand, multi-view, fused, suction, industrial and the two calibrations, is
in the [willy_sim README](../src/willy_sim/README.md), and `scripts/examples/sim/70_sim_pick.py` fronts the
first three.

A simulator rate proves the software and never the cell. Contact friction is a model, the depth is a
perfect sensor, a gripper that closes cleanly here can slip on a real surface, and a simulated
camera's near clip hides things no real camera hides.

## 5. Record a demo

The clips under [`docs/assets/demo/`](assets/demo/) come from the `run_*_demo` recorders:

```bash
cmd /c "<isaac-sim>\python.bat -m src.willy_sim.run_dense_demo_endgame > demo.log 2>&1"
```

The MP4 lands under `logs/demo/`. Copy the ones you want to keep into `docs/assets/demo/`.

## If your GPU is Blackwell

None of the following is in the official compatibility notes, and each one costs an afternoon.

**Use the standard `Camera`, never `TiledCamera`.** `TiledCamera` hangs at reset on sm_120
([IsaacLab #4951](https://github.com/isaac-sim/IsaacLab/issues/4951)). This is the single most likely
way to lose an afternoon, because a hang looks exactly like a slow first boot.

**The PhysX GPU pipeline can fall back to CPU without saying so**
([IsaacLab #3448](https://github.com/isaac-sim/IsaacLab/issues/3448)). The reported workaround is a
driver downgrade. A silent fallback shows up as a simulation that is correct but far too slow, so
check throughput rather than trusting that the GPU is being used.

**16 GB of VRAM is the stated minimum for Isaac 5.1**, and several Blackwell consumer cards have
exactly that. Keep scenes lean, cameras at 720p and textures modest, or expect to run out. Check the
minimum driver version your Isaac release states, and note that the minimum is not always on the
recommended list, which matters only if cold-boot crashes appear.

**Validate a headless boot and a render before trusting anything else.**

## Five API rules

**`SimulationApp` comes before any other isaacsim import.** `art.initialize()` and `cam.initialize()`
come after `world.reset()`, and have to be redone after every reset.

**Depth means `get_depth()`**, which is `distance_to_image_plane` and therefore true Z. Never
`distance_to_camera`, which is radial and warps the point cloud. The values are stage units, and
those are metres only where the world was built with `World(stage_units_in_meters=1.0)`. Mask out the
`+inf` background.

**`get_rgb()` returns RGB, and both the detector and OpenCV want BGR.** The `[..., ::-1]` swap is
mandatory. Skip it and nothing crashes, the detections are simply worse.

**The first camera frames are black.** Render three to five before reading. A related trap:
`add_default_ground_plane()` adds no light at all, so RGB renders black while depth stays correct
because depth is geometric. Add a DomeLight and a DistantLight and warm up about 30 render steps.

**One `SimulationApp` per process**, so one Isaac session per process, and integration tests are
session-scoped. Never call `world.step()` off the main thread.

On macOS, never import `omni.*` or `isaacsim.*` at module top level. Keep them lazy inside method
bodies, after the mock-mode check, so the package keeps importing on a machine with no Isaac at all.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Gate 0 of N, black camera frames | Camera near clip too large. The sim tree sets 0.05 m for the vision cameras; check `robot.sim.cameras`. The nadir overhead camera leaves `near_clip_m` unset on purpose, keeping Isaac's 1.0 m default, which hides the arm from the instance mask |
| The vision pick finds nothing, or drops the small cube | The detector ran fp16. The `sim` model overlays reset `torch_dtype` to unset so the weights load fp32 under fp16 autocast. Confirm with `python -m src.config --profile sim --print` |
| A `BLIND IK` warning where the planner is `curobo` | The cuRobo environment was not found. Re-check `WILLY_CUROBO_PYTHON` and run `--doctor` |
| The self-collision guard reports unavailable | Neither Coal nor python-fcl is importable. Re-check `WILLY_COAL_PREFIX` and the mesh bundle for this robot model |
| Boot hangs | A compound command. Use a single `cmd /c "..."`, then kill `kit.exe` and retry |
| A segfault at shutdown | The headless `SimulationApp.close()` of Isaac can segfault on teardown. It fires after the run has produced its result, so capture the result before stopping |

## See also

- [`ext_deps/README.md`](../ext_deps/README.md)
- [`src/willy_sim/`](../src/willy_sim/README.md)
- [`src/robot/safety/planning/`](../src/robot/safety/planning/README.md)
- [code integrity](code-integrity.md)
- [calibration setup](calibration-setup.md)
