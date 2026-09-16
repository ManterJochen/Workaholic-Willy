# Reference robot and gripper for the trajectory planner

This folder is the version-controlled home and label for the robot the cuRobo motion planner in
`src/robot/safety/planning/` plans for. Both the simulator driver and the real UR execution path in
`src/robot/drivers/ur/curobo_motion.py` use it. It answers the question of where the geometry a
planner trusts actually came from.

## Robot: Universal Robots UR5e

Kinematics come from the canonical UR5e URDF in Universal Robots' `ur_description`, licensed
BSD-3-Clause upstream. It is rendered from UR's own config, vendored and pinned to one upstream
commit under `scripts/curobo/ur_ros2_description`, by
[`scripts/curobo/build_ur_config.py`](../../../../../scripts/curobo/build_ur_config.py) with
`--urdf-from ur`, the default, with the mesh paths rewritten relative and written into the ignored
planner content directory as `ur5e.urdf`. Rendering from the vendor config is what lets a box without
a simulator build a descriptor at all; where a simulator is present the two descriptions agree for
this arm, and the ready gate over all 18 arm and hand pairs reads identically either way. The UR5e
kinematic parameters are fixed and public. In the bundled Denavit-Hartenberg form:

```
a     = [0, -0.425, -0.3922, 0, 0, 0]              m
d     = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]     m
alpha = [pi/2, 0, 0, pi/2, -pi/2, 0]
```

Joint order on the `tool0` chain is `shoulder_pan`, `shoulder_lift`, `elbow`, `wrist_1`, `wrist_2`,
`wrist_3`, from base outward, with `tool0` as the end-effector frame. That is the order the UR
interface reports and expects, and the order `drivers/ur/curobo_motion.py` remaps against.

## Robot: Universal Robots UR3e

Same standard UR DH family. The UR3e row lives in
[`../../_ur_kinematics.py`](../../_ur_kinematics.py):

```
a = [0, -0.24355, -0.2132, 0, 0, 0]              m
d = [0.15185, 0, 0, 0.13105, 0.08535, 0.0921]    m
```

Its planner descriptor, `willy_ur3e.yml`, is assembled on the box by the same script, one per arm.

Its collision geometry is [`../../data/ur3e_collision_meshes.npz`](../../data/ur3e_collision_meshes.npz),
baked from Universal Robots' own collision STL files by
[`scripts/isaac/bake_ur_meshes_from_urdf.py`](../../../../../scripts/isaac/bake_ur_meshes_from_urdf.py),
through the URDF rendered from UR's pinned config. It uses the same DH-frame recipe as every other
bundle:

```
M_dh = inv(T_dh[frame](q0)) @ inv(R_base) @ urdf_link(q0) @ mesh_origin
```

Four things about that generator are load-bearing:

- The bundle was baked from the simulator's own UR3e collision meshes before the vendor STLs. The
  two agree on this arm to under a thousandth of a millimetre, which is one of the six measurements
  that licensed the change. Four links of the family did not agree, and each difference was a mesh
  origin offset the simulator's URDFs omit and UR declares. `bundles.json` beside the bundles records
  each one's source and a hash over its arrays, so every bundle can be rebuilt from the pinned STLs
  and compared.
- It gates every write. A bake is compared with the bundle already committed for the same model, and
  a difference beyond the ceilings refuses the write unless the caller names what moved. That gate
  must pass before a new model's bundle is trusted, because a wrong frame here corrupts a safety
  guard with no other symptom.
- The link meshes in the simulator asset sit behind instance proxies. A traversal must ask for them
  explicitly or it finds nothing at all.
- The Robotiq 2F-85 `tool0` meshes at DH frame 6 are model-independent and are copied verbatim. The
  URDF joints from `wrist_3` through `flange` to `tool0` are identical across the UR e-series, so the
  transform from DH frame 6 to `tool0` is identity for every model here.

## Robot: Universal Robots UR10e

Same standard UR DH family, and the longest arm with committed geometry here. The row lives in
[`../../_ur_kinematics.py`](../../_ur_kinematics.py):

```
a = [0, -0.6127, -0.57155, 0, 0, 0]                  m
d = [0.1807, 0, 0, 0.17415, 0.11985, 0.11655]        m
```

Its collision geometry is
[`../../data/ur10e_collision_meshes.npz`](../../data/ur10e_collision_meshes.npz), baked by the same
generator under the same DH-frame recipe, so it carries the same 27 keys as the ur5e bundle: six arm
links and the three Robotiq bodies, each as vertices, faces and a frame index.

Two properties of that bundle are worth checking before trusting it, and both hold. The nine Robotiq
arrays are byte-identical to the ur5e bundle's, which is what the generator's `tool0` rule requires:
the transform from DH frame 6 to `tool0` is identity across the e-series, so the tool geometry is
copied rather than recomputed. The arm links are not identical and carry this model's own lengths:
`upper_arm__v` spans 749.1 mm against 545.2 mm on the ur5e, each about 136 mm and 120 mm longer than
its own DH link (612.7 mm and 425.0 mm), which is the joint housing at either end.

With this bundle committed, `self_collision.backend: fcl` and `mesh_dir` at `null` resolve to exact
meshes for a ur10e cell: `mesh_backend_status("ur10e")` no longer answers `no_bundle`, and answers
`ok` wherever an engine is installed. Its arm spheres are committed here as
[`ur10e_arm_spheres.yml`](ur10e_arm_spheres.yml), like every other arm's, and the descriptor is built
on the target box from that file.

## Gripper: Robotiq 2F-85

Collision geometry is committed and vertex-exact in
[`../../data/ur5e_collision_meshes.npz`](../../data/ur5e_collision_meshes.npz), which is the single
source of truth shared by the exact-mesh self-collision guard in `safety/self_collision.py` and by
the planner's collision-sphere fit. The keys `gripper__v`, `lfinger__v` and `rfinger__v` are in the
`tool0` frame in millimetres; every `<arm-link>__v` key is in its own DH frame.

The committed sphere map is [`robotiq_2f85_gripper_spheres.yml`](robotiq_2f85_gripper_spheres.yml),
the 2F-85 `tool0` collision spheres cover-fit from that bundle by
[`scripts/curobo/fit_cover_spheres.py`](../../../../../scripts/curobo/fit_cover_spheres.py). That
generator needs no simulator and no GPU: it reads the committed bundle and writes the map beside
itself, so the gripper exists as a labelled, regenerable file in this repository. The spheres are
frame-correct and directly usable by the planner. The map is named after the hand's registry name, as
every map is, so the guard and the planner find a hand's files from `robot.gripper.model` alone
([`../hand.py`](../hand.py)).

## What a committed sphere map claims

A sphere model makes two errors and only one is safe. A hole is a false clear: the planner believes
part of the robot is not there. Reach past the body is a false collide: picks and workspace lost,
nothing worse. The cover fit makes the safe half a property of the construction, in that every
surface sample ends up inside a sphere or there is no map at all, and what the caller trades is count
against reach.

Measured with one ruler
([`scripts/curobo/_mesh_body.py`](../../../../../scripts/curobo/_mesh_body.py)) against the same
meshes, before and after:

| hand | before (grid fit) | after (cover fit) |
|---|---|---|
| `robotiq_2f85` | 36 spheres, hole 18.5 mm, reach 24.7 mm | 142 spheres, no hole, reach 12.0 mm |
| `robotiq_hande` | 38 spheres, hole 16.5 mm, reach 23.4 mm | 70 spheres, no hole, reach 12.0 mm |
| `schunk_egu50` | 46 spheres, hole 19.0 mm, reach 22.9 mm | 129 spheres, no hole, reach 12.0 mm |

The arm links are fitted the same way, one `{arm}_arm_spheres.yml` per arm, and the descriptor
builder reads them instead of the simulator's Lula family, which reached 0 to 61 mm past its own
links, left gaps up to 45 mm, and on ur10 offered a map belonging to a different link-frame family.
Each link is asked for the reach the pairs that actually bind need, which the retract table records:
`wrist_1|wrist_3` is the nearest exact pair in 25 rows, `forearm|wrist_2` in 11, `wrist_1|gripper` in
8, while shoulder and upper_arm appear in none.

## The hand is not the arm

A sphere map describes an end effector, and the arm bundles are named after arms. The same Robotiq
sits in every committed arm bundle, so one map covers all of them; a different hand needs its own.
[`schunk_egu50_gripper_spheres.yml`](schunk_egu50_gripper_spheres.yml) is the second one.

This mattered more than it looks. `scripts/curobo/build_ur_config.py` carried its own copy of the fit
and always used the Robotiq bundle, under a comment calling it model independent. It is independent
of the arm and not of the hand, so a cell running the Schunk had a safety guard that read the right
bundle through its variant key and a planner that modelled a Robotiq. The on-box script now
reads the committed map for the gripper it is told about:

```
python scripts/curobo/build_ur_config.py ur5e --gripper schunk_egu50
```

A gripper nobody has baked a bundle for is fitted from its own mesh, which is the path a customer
with a vendor STL takes:

```
.venv/Scripts/python.exe -m src.robot.safety.planning.robot.build_gripper_spheres     --mesh vendor/eoat.stl --gripper eoat --scale-to-mm 1000 --cell-mm 34 --rmax-mm 17     --origin mounting_face --out eoat_gripper_spheres.yml
```

Nothing on that path has a default, and that is deliberate. The pairing of `--cell-mm` 34.0 with
`--rmax-mm` 24.0 is this repository's 2F-85 finger cell size against its palm radius cap: a
combination describing no part of any gripper, including the one both halves were measured from. A
default that looks calibrated is worse than one that looks arbitrary. The shipped bundles use 44 mm
cells capped at 24 mm for a palm and 34 mm capped at 17 mm for a finger blade.

Three things the file cannot check and a person has to: the mesh must be in the `tool0` axes,
`--scale-to-mm` must be right, and `--origin` must say where the numbers start. Metres read as
millimetres is a hand a thousand times too small, and it plans happily straight through everything it
should have hit.

## Gripper: Robotiq Hand-E

Its collision geometry is
[`../../data/robotiq_hande_hand_meshes.npz`](../../data/robotiq_hande_hand_meshes.npz), the hand
alone, composed onto the arm when the guard loads. Its arrays were baked by
[`scripts/grippers/bake_gripper_variant.py`](../../../../../scripts/grippers/bake_gripper_variant.py)
from the simulator's `Robotiq/Hand-E/Robotiq_Hand_E_edit.usd`, and its committed sphere map is
[`robotiq_hande_gripper_spheres.yml`](robotiq_hande_gripper_spheres.yml).

Measured off the bundle: stroke `width_mm = 49.99 - 2q` with `q` in `[0, 25]` mm per finger, housing
99.20 mm long and 75.00 mm across, contact patch 20.91 mm with 10.45 mm of it ahead of the grasp
centre, grasp centre 135.75 mm from the gripper's own mounting face.

### One bundle per hand, composed onto the arm

A hand bundle is `{hand}_hand_meshes.npz` and holds the hand alone: the `gripper`, `lfinger` and
`rfinger` meshes at DH frame 6, the origin its numbers start from, and `hand__admitted_arms`, the
arms it was proven on. `environment.compose_collision_meshes(model, hand)` takes the arm's own
bundle, drops its 2F-85 and puts the hand in, so every arm link is the arm's committed bytes whatever
hand it carries.

A gripper used to need one arm-plus-hand file per arm, and a missing one dropped the whole cell to
the capsule proxy, the arm included. Measured: `schunk_egu50` on a ur3e did exactly that, because
only a ur5e file was ever baked for it. It still refuses, now for the right reason and as
`variant_model_mismatch` from the admitted arms record: the Schunk is proven on a ur5e only. A
freshly baked hand records no arm, so it is admitted to one by evidence and never by being written.

### The coupling, and why a map says where it starts

The 2F-85's geometry came out of a composed UR asset, so the arm had already placed it: its numbers
start at the flange. Measured, reading the standalone 2F-85 asset in its own root frame reproduces
the committed bundle to 0.00 mm on all six corners of the palm, which is what says that asset's root
is the flange.

The Hand-E asset is a standalone vendor model with no coupling part in it at all, and its housing
measures 99.20 mm, the published body length. Its numbers therefore start at the gripper's own
mounting face, and whatever plate sits between that face and the flange is not in them. Every sphere
map carries `_provenance.origin`, one of `flange` or `mounting_face`, and the planner refuses to
place a `mounting_face` map without the coupling rather than assuming zero. Assuming zero puts every
sphere one plate too close to the flange, which is optimistic in the one direction a planner must not
be, and the file looks entirely reasonable either way.

It is the same bench measurement `robot.gripper.tool_frame.offset_mm` needs. Take it once.

### The frame check that a size assertion cannot make

The bake script's `--check` reads the standalone 2F-85, places it by the same reasoning as a new
hand, and diffs it against the committed bundle. It runs before every write and a failure stops the
write. That gate exists because the first attempt at the Hand-E frame was an axis swap, which is a
reflection with determinant -1: it mirrors the hand, and a mirrored symmetric gripper has identical
extents. Every assertion anyone would naturally write about a gripper bundle, widths, lengths,
bounding boxes, corner distances, passes on a hand built inside out. Only the determinant and a diff
against a known-good bundle see it.

The grid fit itself lives in [`gripper_spheres.py`](gripper_spheres.py), in one place, so the file
and the generator cannot drift apart.

## The planner robot config, one per arm

The complete cuRobo `robot_cfg` is `willy_{arm}.yml`, for example `willy_ur5e.yml`, named by the
cell's arm alone; only a client built with no name falls back to `WILLY_CUROBO_ROBOT`, whose default
is `ur5e.yml`. It bundles the URDF above, the arm-link spheres from this folder placed into each URDF
link frame, joint limits, `default_q` and `self_collision_ignore`.

The hand is not in it. The gripper spheres in this folder become a fixed link under `tool0` that the
sidecar of the planner adds when it starts, placed from `robot.gripper.model`,
`robot.gripper.coupling_plates_mm` and the declared tool frame (`safety/planning/body_link.py`).
Measured over all 18 arm and hand pairs: the composed robot and the per-hand descriptor it replaced
agree on every sphere position to 1.19e-7 m and on the `check_js` verdict of all 1,482 poses.

`default_q` is chosen, not inherited. It is the retract from [`ur_retract.yaml`](ur_retract.yaml)
beside this file, the first pose outwards from the simulator's own pose that clears every registry
hand by the guard margin plus 3 mm on the exact meshes and stands 10 mm above the table.
`elbow_joint` is clamped to UR's own 180 degree planning limit in the description, and
`position_limit_clip` is written rather than inherited, so the envelope of the planner stays inside
the guard's.

The exact cuRobo schema is produced on the target box by `scripts/curobo/build_ur_config.py` and
written into the ignored content directory under `ext_deps/`. That is where the descriptor physically
lives at runtime; this folder is its committed geometry authority and label. The descriptor records
its arm and the fact that it carries no hand under `_provenance`, and a planner refuses one built for
another arm, and one that models a hand of its own.

## A descriptor build reproduces

A re-run reproduces the URDF half byte for byte, and it reproduces the YAML half as well, because
every sphere in it comes from a committed file: the arm map in this folder and the hand map the
sidecar adds as a body link. With a committed arm fit present, the surface augmentation that samples
the mesh is skipped outright, so nothing in the build draws from a generator.

The augmentation is what used to make a descriptor unreproducible, and it still runs for an arm with
no committed fit, which is the fallback to the simulator's own sphere map. On that path two runs of
the same recipe differ in sphere centres and the sphere count wobbles by a sphere or two on one link,
a different link each time. An arm in that state is one whose map has not been fitted here yet.

## Honesty

Planner collision-awareness is simulation-grade and is not a certified functional-safety stop. A real
cell still needs the vendor safety-rated stop.

The `tool0` gripper spheres here are frame-correct and derived from committed data in this
repository. The complete assembled descriptor and any statement about real planning quality are
on-box measurements, and the engines they need are not present in a plain checkout.

A hand bundle carries no arm, so re-baking an arm bundle cannot unpair a hand from it. What a hand
bundle does carry is the arms it was proven on, and a hand is admitted to an arm by that record
alone: a cell pairing a hand with an arm nobody measured it on is refused rather than guessed at.

## Licences

- UR5e and UR3e URDF, from `ur_description`: BSD-3-Clause, Universal Robots.
- UR arm meshes for every descriptor (ur3, ur3e, ur5, ur5e, ur10, ur10e, ur16e): BSD-3-Clause,
  Universal Robots, from their ROS 2 description at commit
  `89bbe795f38a7ab00fb66fe8831dfff79dc99edf`, the source cuRobo's own `ur_description/LICENSE` names.
  cuRobo ships the ur5e and ur10e meshes; the others are fetched into the ignored cuRobo asset root
  by [`scripts/curobo/fetch_ur_meshes.py`](../../../../../scripts/curobo/fetch_ur_meshes.py) and held
  to [`scripts/curobo/ur_meshes.sha256`](../../../../../scripts/curobo/ur_meshes.sha256), as the
  commit holds them. `install.ps1` runs it before the descriptors. Every arm's collision meshes are
  pinned; the visual meshes are pinned for the arms cuRobo does not ship, because its own ur5e and
  ur10e visuals are another revision.
- Robotiq 2F-85 description: BSD-3-Clause, Robotiq and ROS-Industrial.
- The committed `.npz` mesh bundles and the sphere `.yml` are artifacts of this project, derived from
  those descriptions for collision checking.
