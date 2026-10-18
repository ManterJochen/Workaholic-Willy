# Runbook: putting a Robotiq Hand-E on the cell

**Scope.** A hand this repository never shipped follows [your_own_gripper.md](your_own_gripper.md)
instead. Swapping the end effector, from an unpacked gripper to a cell whose planner, whose
collision guard, whose grasp calculator and whose driver all model the hand that is actually bolted
on. It is the gripper half; the physical-arm procedure is
[real_cell_first_pick.md](real_cell_first_pick.md) and the robot half is
[cell_bringup.md](cell_bringup.md).

**Why it exists.** A cell's gripper answers three different questions, and until 2026-09-08 no
validator named more than one of them:

| where | question it answers |
|---|---|
| `robot.gripper.*` | what the driver may command |
| `config/grippers/robotiq_hande.yaml` (the hand `robot.gripper.model` names) | what shape the calculator sweeps |
| `robot.safety.self_collision` plus the cuRobo descriptor | what the planner and the guard refuse against |

Each was individually valid while the three disagreed. That was reachable and undetectable: set
`max_width_mm: 50` for a Hand-E, leave the rest, and the cell commanded a 50 mm stroke while the
collision filter swept a 2F-85's fingers and the planner refused against a 2F-85's spheres. The
first two are not written in a profile at all: `robot.gripper.model` names the hand, the loader
fills its widths and envelope from `config/grippers/robotiq_hande.yaml`, and a profile that states
one differently is refused at load, naming both files.

**The one line version.** One name sets all three, and three numbers cannot be derived from any
file and have to come off the bench.

---

## Trigger

Any of:

- a Hand-E is being fitted to a cell that ran a different hand
- `python -m src.robot.safety.planning --doctor` reports `gripper geometry` or `gripper sphere map`
- a grasp contacts the object somewhere other than where the plan said, on a cell that recently
  changed grippers
- the planner refuses reachable poses, or accepts poses that collide, after a gripper swap

---

## Diagnose

Ask the artifacts, not your memory. The cuRobo descriptor states that it models no hand:

```bash
python - <<'EOF'
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path("<curobo content>/configs/robot/willy_ur3e.yml").read_text())
print(cfg.get("_provenance", "NO PROVENANCE: built before 2026-09-08"))
EOF
```

A current descriptor says `carries_hand: false`: the hand is added as a body link when the planner
starts. One with no `_provenance` block, or one that names a hand, was built before a descriptor was
per arm; both drivers refuse it by name, and it has to be rebuilt.

Then ask the doctor, which now probes the hand and not only the arm:

```bash
python -m src.robot.safety.planning --doctor
```

Read three lines of its output:

- `collision mesh bundle (<arm>)` is the arm. It was green before this runbook existed even when the
  gripper geometry was missing entirely.
- `gripper bundle (<hand> on <arm>)` is the hand for this arm.
- `gripper sphere map (<hand>)` is what the planner will be given.

---

## Mitigate

### 1. Measure the coupling, once

Flange face to gripper mounting face, in millimetres, along the tool axis. Take the flange-to-TCP
transform in the same session: it is the same setup and the same reference surface, and doing them
apart is how the two end up describing different geometry.

The gripper's own contribution is already measured and does not need re-taking: **mounting face to
grasp centre is 135.75 mm**, off the asset. What you are measuring is the plate.

> **The rotational half is the dangerous half.** An offset wrong by 100 mm is a crash on run 1 and
> you fix it. A rotation wrong by 90 degrees is a cell that logs 10/10 successes while closing the
> jaws along the wrong object axis, and hand-eye calibration cannot catch it: it solves for whatever
> frame `get_tcp_pose()` reports and returns an excellent RMSE either way.

### 2. Select the hand in config

```bash
export WILLY_PROFILE=hande          # a UR5e cell
export WILLY_PROFILE=ur3e,hande     # a UR3e cell
```

That one profile names the hand, and the name brings the stroke and the collision envelope from the
registry file and the bundle the guard reads. It
deliberately leaves `robot.gripper.tool_frame.source` at `undeclared`, and the real UR driver refuses
to connect while it says that. Fill it in from step 1, in your own cell profile rather than in the
gripper one: the plate is a property of your cell and the gripper profile is shared.

### 3. Give the planner the hand

The guard reads config, and so does the planner's hand. **The descriptor is the arm alone** and is built
once per arm, on the box, with the cuRobo environment's interpreter:

```bash
ext_deps/curobo_env/python.exe scripts/arms/descriptor.py ur3e
```

It refuses `--gripper` and `--coupling-mm` by name: the hand is not in the descriptor. When the planner
starts, its sidecar adds the hand this cell names as a fixed link under `tool0`, placed from
`robot.gripper.model`, `robot.gripper.coupling_plates` and the declared tool frame. So the plate from
step 1 goes into the cell profile and nowhere else, and a plate nobody measured leaves the cell refusing
to build rather than modelling the hand one plate too close to the flange.

Check that it plans with the hand added:

```bash
ext_deps/curobo_env/python.exe scripts/arms/check.py willy_ur3e --hand robotiq_hande --coupling-mm <the plate from step 1>
```

### 4. Declare the payload

`robot.safety.payload.mass_kg` and `cog_mm`, and the same numbers on the controller. The shipped tree
declares `mass_kg: 0.0`, which makes the payload guard's envelope check vacuous rather than wrong,
and a heavier hand eats the arm's rated payload with nothing noticing.

### 5. Declare the coupling plate, once, in two places

The plate between the arm flange and the gripper's own mounting face is a bench measurement, and it
is the same number every time:

```
robot.gripper.tool_frame.offset_mm            plate + 135.75 mm along the approach
robot.gripper.coupling_plates                 [{name: adapter, thickness_mm: plate}]
```

Both are read at runtime, and both models of the hand are placed from them: the planner's hand link and
the exact-mesh guard's composed hand bundle. The check in step 3 takes the same number as
`--coupling-mm`, so a rehearsal models the cell being built rather than a bare flange.

`robotiq_hande_hand_meshes.npz` is stamped `gripper__origin = "mounting_face"`, which the bake module
defines as "whatever plate sits between that face and the flange has to be added before the planner sees
it". A Hand-E cell that writes no plate refuses to build: the hand's own arrays start at the mounting
face, and no plate written is no measurement rather than zero. `robot.hande.yaml` writes one 20.0 mm
plate, an assumption to replace with this cell's bench number.

### 6. How far the parts hang past the fingertips

No cell models a part in the jaws since 2026-10-17 (the owner's since 2026-10-05: "mehr Griffe"): a closed
hand is judged as an empty one, by the exact guard at full stroke, and the planner reserves no spheres for a
part, so every committed evidence file is one a cell plans on. Until then
`robot.safety.planning_world.payload` modelled the carried part from its `length_mm`, and a part's box fitted
with spheres reached past it, so a lift close over a surface could be refused; the load refuses that block
now, naming what replaced it.

What stays is the number a place needs: `robot.motion.place.part_hang_mm`, how far the longest part hangs past the
fingertips along the approach. A place sets the part's bottom down where the taught fingertips stood, raised
by it, and keeps it over a box's rim; the pick's looks measure the hang part by part where they read under the
part. It ships `null`, which hangs the part from the TCP (the owner, 2026-10-14); the owner's cell sets 1. This
cell handles several parts, so declare the longest as its worst case: too long lets a part drop from higher
than it needs to, too short sets the longest part down into what it is placed on.

**The operator console's task needs no length.** No task waits on the carried part and the ready bar has no
light for it; a hand that measures a part where none should be still ends the task `part_still_held` where the
arm stands.

---

## Verify

```bash
python -m src.robot.safety.planning --doctor
```

Everything in the gripper block reads `ok` except the Hand-E's `gripper sphere map`, which stays a
warning on purpose: its map starts at the mounting face, so it asks you to check that
`robot.gripper.coupling_plates` holds this cell's plate.

```bash
python -c "
from src.config.loader import load_config
from src.robot.safety._fcl_self_collision import mesh_backend_status
from src.robot.safety.planning.hand import planner_hand
c = load_config(); h = planner_hand(c.robot)
print(mesh_backend_status(c.robot.ur.model, None, h.guard_variant))"
```

`ok`. Anything else means the whole cell is on the capsule proxy, arm included, not just the hand.

Then the arm, from [real_cell_first_pick.md](real_cell_first_pick.md), and one bench check of the
driver: command a width, read it back, and confirm the two agree within a millimetre. A Hand-E switched
over its I/O coupling (`jaw_io`, `single_toggle`) reads no width back: time its stroke both ways
instead and set `close_settle_s` (real_cell_first_pick.md, Diagnose 2). In the operator console the Hand-E's
connect asks where its jaws stand in the browser, with no default: hold the part before you answer "Jetzt
öffnen", which is one change of tool DO0, and count exactly two changes per part during a task
([console_at_the_cell.md](console_at_the_cell.md)).

---

## Rollback

Set `WILLY_PROFILE` back to the cell profile without `hande`. Nothing is rebuilt: the descriptor carries
no hand, so the planner adds whichever hand the profile then names.

The 2F-85's arrays are measured at the flange and need no plate, because they already include wherever
the arm put the hand. A profile naming it writes no `coupling_plates`, and a plate declared on it is
refused rather than applied.

---

## What this cannot do for you

- **One Hand-E has run on a physical arm**: a UR10 (CB3), the hand switched over one tool output
  (`jaw_io`, `single_toggle`), planned with cuRobo. The Robotiq socket driver was measured with a UR, and
  on that UR10 its port 63352 was refused, which is why the Hand-E runs as `jaw_io` there. Every number
  here is measured off the vendor asset and against the committed bundles, and a bench day is a
  different measurement.
- **The push plans the open hand from the registry file**, the housing's 75 mm along the closing axis
  (`palm_thickness_mm`) included, so a tall neighbour stays clear of the palm
  ([real_cell_first_pick.md](real_cell_first_pick.md), Diagnose 9).
- **Custom fingertips change `closed_width_mm`.** The registry file says `0.0`, which is the bare
  gripper: its carriages meet, measured. Fingertips that do not meet make that a real number, and it
  is the single anchor of the driver's count map. They also change the body the planner refuses
  against, so they are a different hand to this stack: describe the hand with its tips as its own
  registry hand. A cell profile that states another `closed_width_mm` for `robotiq_hande` is refused
  at load.
- **The simulator's jaw direction is unconfirmed.** The asset's joint frames say a rising joint value
  opens the jaws; its geometry says that would open the hand to 100 mm, which no Hand-E does. The
  profile follows the geometry. If a sim run drives the jaws the wrong way, reverse
  `ROBOTIQ_HANDE_PROFILE.angle_width_table` and nothing else.
- **The collision envelope under-models one sliver, on purpose.** Between 24.6 and 34.7 mm behind the
  grasp centre the carriage shoulder narrows the passage to 36.7 mm, and a box model cannot say
  "thicker at the back". Covering it would declare the gap 13.3 mm narrower everywhere and refuse
  every object over 36.7 mm even for a shallow grasp. The box is a candidate filter; Coal and cuRobo
  refuse against the vertex-exact bundle, which has the shoulder in it, so the cost is a plan
  rejected later rather than a collision.
