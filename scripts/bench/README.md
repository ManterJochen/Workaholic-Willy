# Grasp bench (`scripts/bench/`)

**Your cell's own stack, picking in the scenes it fails in, in seconds per pick.** The bench builds the owner's cell from
its config tree and runs whole picks: the UR driver with its judged lines and cuRobo routes, the exact guard, the camera
world, the grasp calculator, rescan, push and clearing, the toggle Hand-E. Only three things stand in:

| Stand-in | What it does | Where |
|---|---|---|
| **Instant controller** | answers what `ur_rtde` would; a judged `moveJ` or `moveL` is taken at once | `instant_ur.py` |
| **Camera** | renders the scene from where the wrist camera stands, the owner's mat cast whole | `scripts/ursim/_mat_scene.py` |
| **Parts** | carried while the jaws hold them, moved as far as a push plans; no physics | `scripts/ursim/_mat_scene.py` |

```bash
# the GPU route (Smart App Control blocks the Windows cuRobo envs): the planner runs in WSL
WILLY_CUROBO_PYTHON=D:/dev/willy_handover/2026-10-02/final/gpu_aurora/wsl_sidecar.cmd \
WILLY_PROBE_CUROBO_CONTENT=D:/dev/Workaholic-Willy/ext_deps/curobo/curobo/content \
WILLY_COAL_PREFIX=D:/dev/Workaholic-Willy/ext_deps/coal_env \
python scripts/bench/run_bench.py --scenes all --work logs/bench/<run>

python scripts/bench/run_bench.py --scenes bin,clutter_L_15 --looks first   # a family, or names
python scripts/bench/run_bench.py --scenes pile --ip 127.0.0.2              # a second bench beside the first
python scripts/bench/run_bench.py --scenes bin --standoff 10                # a short standoff instead of the 80 mm
python scripts/bench/run_bench.py --no-planner --scenes clutter_one_30      # wiring only: no GPU, no result
python scripts/bench/run_bench.py --scenes pile --noise clean               # the render without the D415's depth
```

The pick moves as the console's does: the policy's own 80 mm standoff, unless `--standoff` names another. `--ip` is
the cell's lock key (nothing dials it): two benches at once need two. After every pick the jaws open and the arm forgets
the part, as a place's release does, so the next scene is not judged as if a part were still carried.

The camera reads depth as the cell's D415 does (`--noise real`, the default): noise growing with the depth, most of it
fixed per camera pose, the invalid band on the left, the stereo shadow, blurred small steps and holes, measured on the
cell's frames of 2026-10-07 (`scripts/ursim/_mat_scene.py`, `D415`). `--noise clean` gives the render as it was.

Since 2026-10-07 the bench runs in WSL (Smart App Control refuses `rtde_control` on Windows):
`gpu_aurora/bench_wsl.sh <name> <ip> <tree> "<profile>" <scenes>` beside the handover. The sidecar there loads the
descriptor and the URDF from its own cuRobo content, which must be the repository's built ones, byte for byte: the
evidence names their hashes.

## The scenes

`scenes.py` holds 36 scenes in four families, each on the owner's mat in LOOK[0]'s view:

| Family | Scenes |
|---|---|
| `bin` | a 40 mm cylinder in a 147 mm deep bin, 10 to 80 mm from a wall, 15 to 50 mm into a corner, with a cube beside it |
| `clutter` | the cylinder with one cube 0 to 30 mm off, an L of two, a ring of four, a 60 mm block, two tall posts |
| `tray` | five parts in a 70 mm tray, each the target in turn |
| `shapes` | lying cylinders, a tall post, a flat plate, a wide cylinder, a cube too short for the Hand-E |

A scene says whether a pick is possible at all (`possible`), and why not where it is not: the rate counts the
possible ones only. Bins and trays are fixed parts: seen, never gripped, never pushed.

## What it writes

`<work>/bench_result.json` (every pick: outcome, attempts and their tries, pushes, blockers, the calculator's counts,
the scene's events, and its telling log lines: each try and why it was refused, the box the guard met by its name and
size, the pairs the planner refuses where a carried lift starts) and `<work>/report.html` (the rate per family and one card per scene with its debug pictures:
every detection's mask, box and label, and the ranked grasps as grasp rectangles, the chosen one thicker).

## What it is not

The parts do not fall, slide or tip, and the jaws hold whatever they close on. Whether a grasp holds, or where a pushed
part really ends, is for the physics: MuJoCo (`datagen/grasps/physics_mujoco.py`) or Isaac. A scene the bench picks is
confirmed in URSim (the real controller) and Isaac before it counts on the cell.
