# Reference robot + gripper for the cuRobo planner (LABELED)

This folder is the **version-controlled home and label** for the robot the cuRobo motion
planner (`backend/src/robot/safety/planning/`) plans for — used by **both** the Isaac sim
driver and the real-UR execution path (`drivers/ur/curobo_motion.py`). It answers "the robot
and gripper must at least exist as a file, labeled."

## Robot — Universal Robots UR5e

- **Kinematics (URDF):** the canonical UR5e URDF from Universal Robots' `ur_description`
  (upstream license **BSD-3-Clause**). It is assembled on-box by
  [`scripts/curobo/build_ur_config.py`](../../../../../../scripts/curobo/build_ur_config.py) from
  the workstation's Isaac `ur5e.urdf` (identical kinematics to `ur_description`), with the mesh
  paths rewritten relative, and written into the (gitignored) cuRobo content dir as `ur5e.urdf`.
  The UR5e kinematic parameters are fixed and public (DH: a = [0, −0.425, −0.3922, 0, 0, 0] m,
  d = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996] m, α = [π/2, 0, 0, π/2, −π/2, 0]).
- **Joint order (tool0 chain):** `shoulder_pan · shoulder_lift · elbow · wrist_1 · wrist_2 · wrist_3`
  (base → wrist_3). End-effector frame = `tool0`. This is the order `ur_rtde` reports/expects and
  the order `drivers/ur/curobo_motion.py` remaps against.

## Gripper — Robotiq 2F-85

- **Collision geometry:** committed, vertex-exact, in
  [`../../data/ur5e_collision_meshes.npz`](../../data/ur5e_collision_meshes.npz) — the **single
  source of truth** shared by (a) the Coal fail-closed self-collision guard (`safety/self_collision.py`)
  and (b) the cuRobo collision-sphere fit. Keys `gripper__v` / `lfinger__v` / `rfinger__v` are in the
  `tool0` frame (mm); `<arm-link>__v` are per-DH-frame.
- **Committed cuRobo sphere map:** [`ur5e_gripper_spheres.yml`](ur5e_gripper_spheres.yml) — the 2F-85
  tool0 collision spheres, grid-fit from that npz by [`build_gripper_spheres.py`](build_gripper_spheres.py)
  (Isaac-free, repo-derived, regenerable here). Frame-correct and directly cuRobo-usable.

## Robot — Universal Robots UR3e (hardware-validation robot, Sept 2026)

- **Kinematics:** same standard UR DH family; the UR3e row is bundled in
  [`../../_ur_kinematics.py`](../../_ur_kinematics.py) (`a = [0, −0.24355, −0.2132, 0, 0, 0]` m,
  `d = [0.15185, 0, 0, 0.13105, 0.08535, 0.0921]` m). cuRobo config `ur3e.yml` is assembled on-box by
  [`scripts/curobo/build_ur_config.py`](../../../../../../scripts/curobo/build_ur_config.py).
- **Collision geometry:** [`../../data/ur3e_collision_meshes.npz`](../../data/ur3e_collision_meshes.npz),
  baked from the Isaac `ur3e.usd` COLLISION meshes by
  [`scripts/isaac/bake_ur_collision_meshes.py`](../../../../../../scripts/isaac/bake_ur_collision_meshes.py) —
  the (previously uncommitted) generator for this artifact family. Same DH-frame recipe as the ur5e bundle:
  `M_dh = inv(T_dh[frame](q0)) · inv(R_base) · world_usd(q0)`.
  - The generator is **self-validating**: run it for `ur5e` and it diffs against the committed ur5e bundle.
    That gate MUST pass before trusting a new model — measured **max |Δv| = 0.000653 mm** (sub-micron), which
    is what licenses the ur3e bake.
  - The Isaac UR link meshes sit behind **instance proxies**: `Usd.PrimRange(prim,
    Usd.TraverseInstanceProxies())` is required or a traversal finds *nothing*.
  - The Robotiq 2F-85 `tool0` meshes (DH frame 6) are **model-independent** and copied verbatim: the
    `wrist_3 → flange → tool0` URDF joints are byte-identical across the UR e-series, so `X_{tool0←DH6}`
    measures as exactly identity for both ur3e and ur5e.

## The cuRobo robot config (`ur5e.yml`)

The complete cuRobo `robot_cfg` (`ur5e.yml`, referenced by `WILLY_CUROBO_ROBOT`, default `"ur5e.yml"`)
bundles: the URDF above · the tool0 gripper spheres (this folder) · the ARM-link spheres (Lula-tuned +
cuRobo surface-fit, placed into each URDF link frame via the measured-exact constant transform
`X_{link←dh} = inv(T_urdf) · Rz(π) · T_dh`) · joint limits · `default_q` · `self_collision_ignore`.
Those Lula-tuned arm spheres and the exact cuRobo schema are produced **on-box** by
`scripts/curobo/build_ur_config.py` (it needs Isaac's Lula `ur5e_robot_description.yaml`, `trimesh`, and
`curobo.sphere_fit`), and written into the gitignored `ext_deps/curobo/…` content dir. That is where
`ur5e.yml` physically lives at runtime; this folder is its committed geometry authority + label.

### One script, and how the retirement of the ur5e-only one was checked (2026-08-26)

`build_ur_config.py` used to sit beside a ur5e-ONLY predecessor, `build_ur5e_config.py`. The
predecessor is gone; `build_ur_config.py ur5e` is the ur5e recipe now. What was measured before
deleting it, on-box under `ext_deps/curobo_env/python.exe` with Isaac 5.1:

* **The URDF is byte-identical** between the two — 23,820 bytes, sha256 `47586eb625cf…`.
* **The YAML cannot be compared byte-wise at all, and that is the finding.** cuRobo's
  `sphere_fit` samples the mesh surface and is **not deterministic**: the OLD script run twice
  against itself produced **486** differing lines and a different sphere count on one link
  (`forearm_link` 36 vs 34; the run before had reported +82 surface spheres, the next +80).
* So the comparison was made against that noise floor, structurally. **Old-vs-new is
  indistinguishable from old-vs-itself**: same 486 differing lines, everything except sphere
  CENTRES identical (kinematics, joint limits, `self_collision_ignore`, `default_q`, `cspace`,
  link set, the 36 tool0 spheres), and the same ±2-sphere wobble on exactly one link — a
  different link each run.

⚠ WHAT THIS DOES **NOT** SAY: that a regenerated `ur5e.yml` equals the one a previous session
generated. It cannot, and no re-run will. A `ur5e.yml` in the content dir is one draw from a
stochastic fit; if a measured planner result depends on a specific one, keep THAT file rather
than trusting a rebuild to reproduce it. (The runtime `ur5e.yml` overwritten during this check
was restored byte-identically from a backup taken beforehand.)

## Honesty

- Being planner-collision-aware is **simulation-grade**, NOT a certified functional-safety stop; a real
  cell still needs the vendor safety-rated stop (see `.ai-memory/Real_Hardware_Needed.md`).
- The **tool0 gripper spheres here are frame-correct and repo-derived**; the full Lula-tuned `ur5e.yml`
  and real planning quality on hardware are **on-box only** (not exercisable in CI — no Isaac, no cuRobo
  GPU env here).

## Licenses

- UR5e URDF / `ur_description`: BSD-3-Clause (Universal Robots).
- Robotiq 2F-85 description: BSD-3-Clause (Robotiq / ROS-Industrial).
- The committed `.npz` mesh bundle and the sphere `.yml` are Workaholic-Willy artifacts derived from
  those descriptions for collision checking.
