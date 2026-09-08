# Reference robot and gripper for the trajectory planner

This folder is the version-controlled home and label for the robot the cuRobo motion planner in
`src/robot/safety/planning/` plans for. Both the simulator driver and the real UR execution path in
`src/robot/drivers/ur/curobo_motion.py` use it. It answers the question of where the geometry a
planner trusts actually came from.

## Robot: Universal Robots UR5e

Kinematics come from the canonical UR5e URDF in Universal Robots' `ur_description`, licensed
BSD-3-Clause upstream. It is assembled on the target box by
[`scripts/curobo/build_ur_config.py`](../../../../../scripts/curobo/build_ur_config.py) from the
workstation's own UR5e URDF, with the mesh paths rewritten relative, and written into the ignored
planner content directory. The UR5e kinematic parameters are fixed and public. In the bundled
Denavit-Hartenberg form:

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

Its planner descriptor `ur3e.yml` is assembled on the box by the same script.

Its collision geometry is [`../../data/ur3e_collision_meshes.npz`](../../data/ur3e_collision_meshes.npz),
baked from the simulator's UR3e collision meshes by
[`scripts/isaac/bake_ur_collision_meshes.py`](../../../../../scripts/isaac/bake_ur_collision_meshes.py),
which is the generator for this whole artifact family. It uses the same DH-frame recipe as the ur5e
bundle:

```
M_dh = inv(T_dh[frame](q0)) @ inv(R_base) @ world_usd(q0)
```

Three things about that generator are load-bearing:

- It is self-validating. Run it for `ur5e` and it diffs the freshly computed links against the
  committed ur5e bundle and refuses to write when the largest per-vertex deviation reaches the
  0.15 mm gate. That gate must pass before a new model's bundle is trusted, because a wrong frame
  here corrupts a safety guard with no other symptom.
- The link meshes in the source asset sit behind instance proxies. A traversal must ask for them
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
`ok` wherever an engine is installed.
There is no committed planner descriptor for this model: the arm-link sphere fit is produced on the
target box, as described below, and the arm-link surface augmentation stays ur5e only.

## Gripper: Robotiq 2F-85

Collision geometry is committed and vertex-exact in
[`../../data/ur5e_collision_meshes.npz`](../../data/ur5e_collision_meshes.npz), which is the single
source of truth shared by the exact-mesh self-collision guard in `safety/self_collision.py` and by
the planner's collision-sphere fit. The keys `gripper__v`, `lfinger__v` and `rfinger__v` are in the
`tool0` frame in millimetres; every `<arm-link>__v` key is in its own DH frame.

The committed sphere map is [`ur5e_gripper_spheres.yml`](ur5e_gripper_spheres.yml), the 2F-85 `tool0`
collision spheres grid-fit from that bundle by
[`build_gripper_spheres.py`](build_gripper_spheres.py). That generator needs no simulator and no GPU:
it reads the committed bundle and writes the map beside itself, so the gripper exists as a labelled,
regenerable file in this repository. The spheres are frame-correct and directly usable by the planner.

## The hand is not the arm

A sphere map describes an END EFFECTOR, and the bundles are named after arms. The same Robotiq sits
in the ur5e bundle, the ur3e one and the ur10e one, so one map covers all three; a different hand
needs its own. [`schunk_egu50_gripper_spheres.yml`](schunk_egu50_gripper_spheres.yml) is the second
one, 46 spheres against the Robotiq's 36.

This mattered more than it looks. `scripts/curobo/build_ur_config.py` carried its own copy of the fit
and always used the Robotiq bundle, under a comment calling it model independent. It is independent
of the ARM and not of the HAND, so a cell running the Schunk had a safety guard that read the right
bundle through `collision_mesh_variant` and a planner that modelled a Robotiq. The on-box script now
reads the committed map for the gripper it is told about:

```
python scripts/curobo/build_ur_config.py ur5e --gripper schunk_egu50
```

A gripper nobody has baked a bundle for is fitted from its own mesh, which is the path a customer
with a vendor STL takes:

```
.venv/Scripts/python.exe -m src.robot.safety.planning.robot.build_gripper_spheres     --mesh vendor/eoat.stl --gripper eoat --scale-to-mm 1000 --out eoat_gripper_spheres.yml
```

Two things that file cannot check and a person has to: the mesh must already be in the `tool0` frame,
and `--scale-to-mm` must be right. Metres read as millimetres is a hand a thousand times too small,
and it plans happily straight through everything it should have hit.

The fit itself lives in [`gripper_spheres.py`](gripper_spheres.py), in one place, and
`tests/test_gripper_spheres.py` compares every committed map against it so the file and the generator
cannot drift apart.

## The planner robot config

The complete descriptor the planner loads, named by `WILLY_CUROBO_ROBOT` and defaulting to
`ur5e.yml`, bundles the URDF, the `tool0` gripper spheres from this folder, the arm-link spheres,
joint limits, `default_q` and the self-collision ignore set. The arm-link spheres and the exact
schema are produced on the target box by `scripts/curobo/build_ur_config.py`, which needs the
simulator's own kinematic description and the planner environment, and written into the ignored
content directory under `ext_deps/`. That is where the descriptor physically lives at runtime; this
folder is its committed geometry authority and label.

The arm-link surface augmentation is ur5e only. The meshes and the DH chain it uses are
ur5e-specific, so every other model keeps its own model-tuned arm spheres, which are correct for its
link lengths.

## The trap: a regenerated descriptor is not the same file

A re-run reproduces the URDF half byte for byte. It does not reproduce the YAML half. The planner's
sphere fit samples the mesh surface and is not deterministic: two runs of the same recipe on the same
box differ in sphere centres, and the sphere count wobbles by a sphere or two on one link, a
different link each time. Everything else is stable: kinematics, joint limits, the self-collision
ignore set, `default_q`, the joint-space configuration, the link set and the `tool0` spheres.

So a descriptor in the content directory is one draw from a stochastic fit. If a measured planning
result depends on a specific one, keep that file. No re-run will reproduce it.

## Honesty

Planner collision-awareness is simulation-grade and is not a certified functional-safety stop. A real
cell still needs the vendor safety-rated stop.

The `tool0` gripper spheres here are frame-correct and derived from committed data in this
repository. The complete assembled descriptor and any statement about real planning quality are
on-box measurements, and the engines they need are not present in a plain checkout.

No tool in this repository produces the per-gripper variant bundles such as
`schunk_egu50_collision_meshes.npz`. Their arm links are copied from a model bundle and the exact-mesh
backend pairs a variant to its model by demanding one arm link match exactly, so re-baking a model
bundle unpairs every variant that pointed at it and silently drops those cells to the capsule proxy.
The bake refuses to overwrite a bundle other bundles are paired with unless told to force it.

## Licences

- UR5e and UR3e URDF, from `ur_description`: BSD-3-Clause, Universal Robots.
- Robotiq 2F-85 description: BSD-3-Clause, Robotiq and ROS-Industrial.
- The committed `.npz` mesh bundles and the sphere `.yml` are artifacts of this project, derived from
  those descriptions for collision checking.
