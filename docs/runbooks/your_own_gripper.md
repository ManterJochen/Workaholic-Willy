# Runbook: your own gripper on a UR arm

**Scope.** A parallel jaw gripper this repository never shipped, on a UR arm, from a registry file to
a desk checklist whose only blocking rows are the camera nobody calibrated yet and the camera world it
will give the planner, and a planner that starts. The shipped hands are the Robotiq 2F-85, the
Robotiq Hand-E and the Schunk EGU-50; the Hand-E has its own runbook,
[hande_gripper_bringup.md](hande_gripper_bringup.md). The physical arm is
[real_cell_first_pick.md](real_cell_first_pick.md).

**Why it exists.** A cell models its gripper in five places that have to agree: the registry file
that describes the jaw, the collision body the exact-mesh guard reads, the sphere map the planner
reads, the retract pose the planner starts from, and the evidence file that says the planner and the
guard were measured to agree on that combination. Each one is written by one script from the one
before it, and every refusal on the way names the script that writes what is missing. This runbook
is those scripts in the order their inputs exist.

**How it is kept true.** Every command below is run, in a copy of the tree, by
`scripts/trial/run_runbook.py` against `scripts/trial/customer_hand_trial.json`, and
`tests/test_your_own_gripper_runbook.py` holds every command to the parser of the script it names. A
step marked `trial-skip` in the source says why the trial did not run it.

**The names used below.** Set them once for your cell; the commands are bash, and a PowerShell reader
writes `& $PY`.

| Name | What it is | Example |
|---|---|---|
| `PY` | the project interpreter | `.venv/Scripts/python.exe` |
| `ARM` | the arm, a profile this tree carries | `ur10e` |
| `HAND` | the registry name your hand will have: lower case letters, digits, underscores | `acme_2f` |
| `CELL` | the name of your cell's profile layer | `acme_2f_ur10e` |
| `ORIGIN` | where your numbers start: `flange`, or the hand's own `mounting_face` | `flange` |
| `ROT` | the tool frame rotation your cell declares, four numbers x y z w | `0 0 0 1` on a real UR |
| `TCP_MM` | flange to grasp centre along the approach, plates included | `140` |
| `COUPLING_MM` | the plates between flange and mounting face, summed | `0` |

---

## Trigger

Any of:

- a gripper that is not the 2F-85, the Hand-E or the EGU-50 is going onto a UR arm this tree plans for
- the desk checklist or the doctor names `write_hand_from_dimensions.py`, `write_hand_from_mesh.py`,
  `bake_gripper_variant.py`, `fit_cover_spheres.py`, `choose_ur_retract.py` or `matrix_gate.py` in a
  remedy
- `robot.gripper.model` names a hand and the loader refuses it as unknown

---

## Diagnose

Ask the doctor about the arm and the hand before any file exists. It names the first thing missing
and the script that writes it.

<!-- step: diagnose-doctor -->
```bash
$PY -m src.robot.safety.planning --doctor --model $ARM --hand $HAND
```

Once a cell layer exists, the desk checklist says what still stops the cell, each row with its fix:

<!-- step: diagnose-desk -->
```bash
$PY -m src.robot.execution.real_cell --check --profile $ARM,$CELL
```

---

## Mitigate

### 1. Describe the hand once, in the repository registry

One file per hand, `config/grippers/$HAND.yaml`. Every number is millimetres in the grasp frame: X
closes between the contacts, Y is the binormal, Z is the approach, and the grasp centre is where the
pads meet. The schema refuses numbers that do not describe one jaw (a pad longer than its finger, a
closed width past the aperture). The hand lives in the repository registry and nowhere else: its
body, sphere map, retract rows and evidence are committed beside the code and written from this
file, and a deployment tree that describes the same hand differently is refused by the validator and
at the desk, naming both files.

A cell that names the hand takes its widths and its collision envelope from this file; a profile
that states one differently is refused at load. You may list the `gripper.vendor` drivers that can
actuate the hand under `drivers`; a real UR profile naming another is then refused at load, and
`none` and `dummy` always pass.

<!-- step: registry-file file: config/grippers/$HAND.yaml -->
```yaml
gripper:
  model: $HAND
  kind: parallel_jaw
  source: >-
    $SOURCE
  jaw:
    grasp_centre_mm: $GRASP_CENTRE_MM
    aperture_mm: $APERTURE_MM
    min_width_mm: $MIN_WIDTH_MM
    closed_width_mm: $CLOSED_WIDTH_MM
    finger_ahead_mm: $FINGER_AHEAD_MM
    finger_behind_mm: $FINGER_BEHIND_MM
    finger_length_mm: $FINGER_LENGTH_MM
    finger_thickness_mm: $FINGER_THICKNESS_MM
    finger_width_mm: $FINGER_WIDTH_MM
    finger_pad_overlap_mm: $FINGER_PAD_OVERLAP_MM
    pad_length_mm: $PAD_LENGTH_MM
    pad_ahead_mm: $PAD_AHEAD_MM
    pad_behind_mm: $PAD_BEHIND_MM
    palm_depth_mm: $PALM_DEPTH_MM
    palm_width_mm: $PALM_WIDTH_MM
    palm_thickness_mm: $PALM_THICKNESS_MM
    palm_measured: true
    friction_coefficient: 0.5
    friction_is_default: true
```

<!-- step: registry-validate -->
```bash
$PY -m src.config
```

The OK line lists `$HAND` among the hands.

### 2. Give the hand a body: one of three routes

The exact-mesh guard composes the hand onto the arm from
`src/robot/safety/data/$HAND_hand_meshes.npz`. Each writer records its row in `bundles.json` beside
it, so the bundle says where it came from.

**From its numbers.** An envelope of three boxes built from the registry file. It does not enclose
the body it stands for, so the inflation is yours to state; the two hands that could be measured
asked for 5.73 mm and 9.50 mm. `--measure-only` prints the boxes and writes nothing:

<!-- step: dims-measure -->
```bash
$PY scripts/grippers/write_hand_from_dimensions.py $HAND --inflation-mm $INFLATION_MM --origin $ORIGIN --measure-only
```

<!-- step: dims-write -->
```bash
$PY scripts/grippers/write_hand_from_dimensions.py $HAND --inflation-mm $INFLATION_MM --origin $ORIGIN
```

**From the vendor's mesh files.** One STL or OBJ per part: the housing and each finger, the left
finger on the side of negative closing. Name the vendor's axes as words (`+X` to `-Z`) and the
mounting face along the approach, in the vendor's frame and millimetres; a mirror is refused, because
a mirrored symmetric gripper has identical extents:

<!-- step: mesh-write -->
```bash
$PY scripts/grippers/write_hand_from_mesh.py $HAND --gripper-mesh $MESH_DIR/gripper.stl --lfinger-mesh $MESH_DIR/lfinger.stl --rfinger-mesh $MESH_DIR/rfinger.stl --scale-to-mm $SCALE_TO_MM --closing $CLOSING --approach $APPROACH --binormal $BINORMAL --mount-face-mm $MOUNT_FACE_MM --origin $ORIGIN --write
```

A body from a mesh gives the registry numbers too. Measure them off it and copy them into the file
from step 1:

<!-- step: mesh-measure -->
```bash
$PY scripts/grippers/measure_jaw_from_bundle.py $HAND --centre-mm $GRASP_CENTRE_MM
```

**From a standalone USD.** The baker reads the vendor's stage with `pxr` and first proves its reader
on the committed 2F-85; skipping that control needs a reason, which is recorded in the bundle:

<!-- trial-skip: pxr is refused by the trial box's application control policy (WinError 4551); tests/test_a_usd_bakes_from_the_command_line.py holds the route where pxr imports -->
<!-- step: usd-control -->
```bash
$PY scripts/grippers/bake_gripper_variant.py --check
```

<!-- trial-skip: pxr is refused by the trial box's application control policy (WinError 4551); tests/test_a_usd_bakes_from_the_command_line.py holds the route where pxr imports -->
<!-- step: usd-write -->
```bash
$PY scripts/grippers/bake_gripper_variant.py --usd $USD --hand $HAND --bodies $BODIES --closing $CLOSING --approach $APPROACH --binormal $BINORMAL --mount-face-mm $MOUNT_FACE_MM --origin $ORIGIN --write
```

### 3. Fit the planner's spheres

The planner models the hand as spheres fitted to the body so every surface point is inside one. Read
the count against reach first, then write the map; a body that does not fit says whether the reach
was not tried or did not fit within the cap:

<!-- step: spheres-curve -->
```bash
$PY scripts/curobo/fit_cover_spheres.py --hand $HAND --curve
```

<!-- step: spheres-write -->
```bash
$PY scripts/curobo/fit_cover_spheres.py --hand $HAND --write
```

<!-- step: spheres-ruler -->
```bash
$PY scripts/curobo/measure_sphere_map.py --hand $HAND
```

### 4. Choose the retract, at the tool frame your cell declares

The pose the planner starts from belongs to the arm and the hand on it. The chooser asks the exact
meshes and the planner's spheres for the first pose both accept, on a box with the cuRobo
environment, and merges only the pairs it judged. Pass the rotation your cell declares: a real UR
flange approaches along `+Z`, which is `0 0 0 1`.

<!-- step: retract-choose -->
```bash
$PY scripts/curobo/choose_ur_retract.py $ARM --hand $HAND --tool-rotation-xyzw $ROT
```

<!-- step: retract-check -->
```bash
$PY scripts/curobo/choose_ur_retract.py $ARM --hand $HAND --tool-rotation-xyzw $ROT --check
```

### 5. Measure the combination: the evidence a planner starts on

A planner starts only on a combination of arm, hand, plates, placement and margins that a committed
evidence file measured at rung b1. The gate judges the retract, 1,000 seeded random poses and a
wrist sweep with the planner and the exact guard, and writes the file:

<!-- step: evidence-gate -->
```bash
$PY scripts/curobo/matrix_gate.py --arm $ARM --hand $HAND --coupling-mm $COUPLING_MM --planner-margin-mm 4 --guard-margin-mm 10 --attach 0 --tool-rotation-xyzw $ROT --write
```

### 6. The cell layer

Name the hand, its driver, its plates, the tool frame and the arm, both as `ur.model` and as the
model the guard derives its links from: the layer chains on `$ARM`, it still loads on its own, and
the arm it states is the one the retract table expects your hand's row on. Write no widths and no
`grasping.gripper_geometry`: the hand supplies them. Weigh the whole assembly for the payload; the
numbers below are placeholders a bench replaces.

The layer also states that the cell carries no part, because the evidence above was measured with
`--attach 0`. A cell that carries one declares `safety.planning_world.payload.length_mm` instead,
how far its longest part hangs past the fingertips, and measures its evidence with `--attach 16`;
with neither, the checklist's `carried part` row blocks.

<!-- step: cell-layer file: config/robot/robot.$CELL.yaml -->
```yaml
robot:
  ur:
    model: $ARM
  gripper:
    vendor: jaw_io
    model: $HAND
    coupling_plates: []
    tool_frame:
      source: willy
      offset_mm: [0.0, 0.0, $TCP_MM]
      rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0]
  safety:
    payload:
      mass_kg: $MASS_KG
      cog_mm: [0.0, 0.0, $COG_Z_MM]
    self_collision:
      kinematics_model: $ARM
      planner_margin_mm: 4.0
    planning_world:
      payload:
        enabled: false
```

<!-- step: cell-validate -->
```bash
$PY -m src.config --profile $ARM,$CELL
```

A cell that plans nothing needs no evidence and no retract: the controller's IK solves its moves and
the exact-mesh guard judges them. Its twin layer:

<!-- step: ik-layer file: config/robot/robot.${CELL}_ik.yaml -->
```yaml
robot:
  ur:
    motion_planner: ik
```

---

## Verify

The desk checklist. On a cell with no calibrated camera the two blocking rows are `camera -> base`
and `camera world` (a cuRobo cell refuses every motion with neither a live camera world nor a
decline); the ik twin blocks on `camera -> base` alone. The hand row names your sphere map and
placement, the grasp centre row holds `offset_mm` to your registry number, and the planner margin row
names your evidence file:

<!-- step: verify-desk -->
```bash
$PY -m src.robot.execution.real_cell --check --profile $ARM,$CELL
```

<!-- step: verify-doctor -->
```bash
$PY -m src.robot.safety.planning --doctor --profile $ARM,$CELL
```

The planner starts through the arm's own path, with every refusal that path meets, and stops. No
camera and no controller:

<!-- step: verify-planner-start -->
```bash
$PY -m src.robot.execution.real_cell --start-planner --profile $ARM,$CELL
```

<!-- step: verify-ik-desk -->
```bash
$PY -m src.robot.execution.real_cell --check --profile $ARM,$CELL,${CELL}_ik
```

The first move is a bench step, [real_cell_first_pick.md](real_cell_first_pick.md):

<!-- trial-skip: needs a physical controller in Remote Control -->
<!-- step: verify-first-move -->
```bash
$PY -m src.robot.execution.real_cell --runs 1 --profile $ARM,$CELL
```

---

## Rollback

Remove what the chain wrote, newest first, through version control: the cell layers, the evidence
file under `src/robot/safety/planning/robot/evidence/`, the hand's rows in `ur_retract.yaml`, the
sphere map `$HAND_gripper_spheres.yml`, the bundle and its `bundles.json` row, and the registry file.
Never rewrite a bundle after its retract rows were judged: the table binds the bundle it was judged
on, so a new body means the chooser and the gate run again.

---

## What this cannot do for you

- **Nothing here has run on a physical controller.** The chain up to the planner start is exercised
  in a copy of the tree on the box; the first move has never run.
- **An envelope is not the hand.** A body from numbers is three inflated boxes. Where the hand has a
  mesh, the mesh route models it.
- **The simulator does not mount your hand.** A registry hand with no simulator mount runs on a real
  cell and is refused by name in the simulator; a mount needs the USD asset, the mount rotation and a
  measured `tcp_offset_mm`.
