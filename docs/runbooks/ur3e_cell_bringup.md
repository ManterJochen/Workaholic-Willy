# Runbook: bringing up a UR3e cell, and keeping the UR5e one working

**Scope.** Standing a cell up on a robot that is not the UR5e the stack grew up on. This is the
configuration and geometry half, proven in simulation; the physical-arm procedure is
[real_cell_first_pick.md](real_cell_first_pick.md). It is written around a UR3e, and every step
generalises.

**Why it exists.** Almost nothing here is about the arm being smaller. It is about the half dozen
independent places that each name a robot, and what happens when one of them still says `ur5e`. Each
failure below first presented as something else entirely: bad grasping, bad perception, a flaky
planner.

**The one line version.** A UR3e is not a short UR5e. Its reach, the clearance its planner can
absorb and the jaw orientations it can reach all differ, and each of them is a separate
configuration value that has to be measured rather than scaled.

---

## Trigger

Any of:

1. Standing a cell up on a UR model this stack has not run before.
2. `--robot-model <model>` refuses at load, because the profile layer is missing or does not claim
   the robot.
3. A new cell's pick rate is non-deterministic on a static scene, which is the signature of a
   degenerate grasp axis meeting an arm that cannot reach every orientation. See Diagnose 4.
4. The boot prints the motion-stack-incomplete box. See Diagnose 1.
5. Any pick-path change, on any robot: re-run both cells' gates before believing a number.

---

## Diagnose

**1. Is the motion stack actually the configured one?**

The cell routes every motion through cuRobo and the exact-mesh collision engine, Coal or fcl.
Neither has a safe fallback: blind IK proposes self-colliding branches the guard then rejects, which
reads as bad grasping, and the capsule proxy over-rejects legitimate reach-down grasps.
`bootstrap_sim_cell` therefore fails closed and prints a box naming what is missing. The probe is a
filesystem check that runs before the Isaac boot, so an operator who has not pointed at the two
environments learns it in milliseconds rather than after a start-up and a run whose numbers describe
a different motion stack.

```bash
# both engines install into ext_deps/ via scripts/ext_deps/install.ps1
python -m src.robot.safety.planning --check --model ur3e    # exit 0 means fully anchored
python -m src.robot.safety.planning --doctor --model ur3e   # and that they actually load
```

`WILLY_ALLOW_DEGRADED_MOTION=1` proceeds anyway. Numbers from such a run describe a different
system, and they are stamped `degraded_engines` in the records so they cannot be mistaken for the
configured cell's later.

**2. Does the boot banner name the robot you think it does?**

```
cuRobo planner: AVAILABLE  (python=..., robot=ur3e.yml not verified: ...)
exact-mesh collision engine: coal  (ur3e mesh bundle present, ...)
=> fully anchored
```

Both halves are per robot: the mesh bundle ships as `{model}_collision_meshes.npz` and the cuRobo
descriptor is `{model}.yml`, so a present `ur5e` bundle says nothing about a UR3e cell. The reading
carries the model and the key that chose it, rather than leaving it to be assumed. What
`AVAILABLE` still does not mean is that the descriptor was built; that is a separate check, and
[real_cell_first_pick.md](real_cell_first_pick.md) Diagnose 6 is where it lives.

**3. Can the arm reach the scene, and can the camera see it?**

Both audits run before the Isaac boot and print only when something is wrong. The reach audit
measures every configured Cartesian point against the model's working sphere about the shoulder,
which sits `d1` above the base. Measuring from the base origin instead would over-state the envelope
by that height, 152 mm on a UR3e, and wave through poses the arm cannot touch.

Known gap: the audit checks Cartesian points, `scene_setup.object`, `scene_setup.marker`,
`safe_pose`, the worst eye-in-hand viewpoint and the worst workspace corner. `home_joint_positions`
and `park_joint_positions` are joint poses and are not covered. Check those by hand. See Mitigate 3.

**4. Non-deterministic pick rate on a static scene?**

The symptom is the same scene, the same selected score, and a run-to-run mix of successes and
`MotionStatus.TIMEOUT`. The cause is that a near-isotropic silhouette, a cube seen from above, has a
numerically degenerate principal axis, so the chosen jaw orientation flips between runs. That only
becomes a failure on an arm that cannot reach both: a UR3e cannot plan a tangential top-down close
at any azimuth while a radial one plans, and a UR5e absorbs both, which is why it stayed invisible
until a second robot arrived. See Mitigate 4.

**5. Losing picks to self-collision rejections just under the margin?**

`forearm|wrist_2: mesh distance 9.44 mm < 10.000 mm` is not a bad grasp. cuRobo plans against
spheres and the guard checks exact meshes, so without
[`planner_margin_mm`](../../src/config/schema/robot/safety_schema.py) the two disagree by fractions
of a millimetre and the plan loses. See Mitigate 5.

---

## Mitigate

**1. Give the robot its own configuration layer.** Profiles compose, so `WILLY_PROFILE=sim,ur3e`
applies every `*.sim.yaml` overlay and then every `*.ur3e.yaml` on top, of which
`config/robot/robot.ur3e.yaml` is the one that ships. Do not fork the simulator overlays into a
`sim-ur3e` profile: they carry measured values, such as the detector dtype the vision pick depends
on, and a second copy will drift.

**1b. URSim needs Docker inside WSL2, and it is installed from a signed repository.**

Deliberately not `curl https://get.docker.com | sh`. The convenience script is Docker's own and
works, but it pipes an unpinned remote script into a root shell. The repository route installs the
same packages from a GPG-signed source that apt keeps verifying on every future update, and this box
later talks to a robot controller.

```bash
wsl.exe -u root -- bash -lc '
  apt-get update -qq && apt-get install -y -qq ca-certificates curl
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  . /etc/os-release
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
    https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
  service docker start'
```

The WSL session owns `dockerd`. Closing the `wsl.exe` window takes the daemon and every running
container with it, so hold a real, durable `wsl.exe` process open for as long as URSim must run. A
backgrounded process inside a throwaway `wsl -e bash -lc` is not enough, because it dies with the
session it was meant to outlive.

Then `scripts/ursim/ursim.sh {up MODEL|down|status}`, where `MODEL` is `UR5` or `UR3`. The
[`scripts/ursim/` README](../../scripts/ursim/README.md) carries the container's failure modes and
the ordered probes that prove this stack talks to the controller, from the raw SDK up to a pick loop
meeting a protective stop. `config/robot/robot.ursim.yaml` and `robot.ursim_ur3.yaml` are the
matching profile layers.

**2. Build the robot's cuRobo configuration and its collision-mesh bundle.**

```bash
python scripts/curobo/build_ur_config.py ur3e            # -> {curobo content}/configs/robot/ur3e.yml

# the bake runs under Isaac's own interpreter, on the box that has Isaac
python.bat scripts/isaac/bake_ur_collision_meshes.py ur5e            # validate, write nothing
python.bat scripts/isaac/bake_ur_collision_meshes.py ur3e --write    # -> src/robot/safety/data/
```

The bake script is self-validating, which is why the `ur5e` run comes first: it diffs the freshly
computed links against the committed bundle, prints the largest per-vertex deviation and refuses the
write at or above its gate. Reproducing a known-good bundle is what licenses baking a new one,
because a wrong frame would silently corrupt a fail-closed guard. It writes nothing without
`--write`, and it refuses to overwrite a bundle that other bundles are paired with unless `--force`
says so, because a rewritten reference unpairs every variant that pointed at it and drops those
cells to the capsule proxy on one log line.

**3. Re-anchor the scene, and check the joint poses too.** Keep every configured position under
about 85 % of reach. Past that the arm is near straight with almost no orientation freedom left,
which is exactly what a top-down grasp needs. Then check `home_joint_positions` and
`park_joint_positions` by hand, because the audit does not: they are joint poses, and a park pose
inherited from a longer arm puts the TCP outside the shorter arm's sphere while every audit stays
quiet. `config/robot/robot.ur3e.yaml` records the resulting radius and percentage of reach beside
each pose it sets, which is the form to copy.

**4. Let a symmetric object's free yaw go to the reachable axis.** `--radial-closing` on the sim
runners, or `GraspCalculator(isotropic_radial_closing=True)`. Only near-isotropic silhouettes and
near-vertical approaches are touched; an elongated object's axis carries real information and is
left alone. Do not instead lower the standoff: the pick still has to lift, so an unreachable
orientation simply fails at the retreat instead of at the approach.

**5. Tell the planner what the guard will demand, with a measured, per-robot value.**
`safety.self_collision.planner_margin_mm`. Do not derive it from `min_distance_mm`. How much margin
a planner can absorb depends on how tightly its sphere model fits that arm: a UR5e plans at the
10 mm `min_distance_mm` the base configuration sets and ships `planner_margin_mm: 0.0`, while a
UR3e's thinner links read as permanently self-colliding once every sphere is inflated by half the
margin, and it finds no plan at all above roughly 6 mm. `robot.ur3e.yaml` therefore ships
`planner_margin_mm: 4.0`, a clear gap under that ceiling. Deriving the value from the guard's
distance looks obviously right and takes the UR3e cell to zero picks. Sweep the real planner for any
new robot before choosing.

**6. Keep the two robots' telemetry apart.** Records carry `robot_model`, `gripper_mount` and
`degraded_engines` through `cell_identity()`. Confirm a new cell's JSONL carries them before
collecting a dataset, because the contamination is retroactive and invisible.

**7. Protect the calibration artifacts.** The simulator writes them under
`logs/calibration/<robot_model>/`, namespaced per robot so one arm's run cannot overwrite another's
in place; the real cell writes under `calibration/real`, keyed by `rig_id`. Neither directory is
tracked. Copy an artifact aside before moving a camera.

---

## Verify

Run both cells. A change that fixes one robot can cost the other, and only measuring shows it:
`--radial-closing` alone helps the UR3e and costs the UR5e until Mitigate 5 is in place.

```bash
python -m src.willy_sim.run_m1_pick --robot-model ur3e --runs 10 --radial-closing
python -m src.willy_sim.run_m2_pick --robot-model ur3e --runs 10 --radial-closing
python -m src.willy_sim.run_m1_pick --runs 10        # UR5e control
python -m src.willy_sim.run_m2_pick --runs 10        # UR5e control
```

Accept when all four meet the configured gate, `sim.pass_fraction`, which ships at 0.8, and the run
reports zero safety rejections. That gate does not take the service's own success report as
sufficient evidence either: it additionally requires a lift measured from the physics to clear
`sim.lift_threshold_mm`, which ships at 50.0.

Off the box, before any of the above:

```bash
python -m src.config                                     # the production tree
WILLY_PROFILE=sim,ur3e python -m src.config              # the new cell
python -m pytest tests/
python -m src.robot.grasping.replay --soak-report
```

The soak gate is robot-agnostic and synthetic. It proves telemetry and KPI consistency, not grasp
quality, and it will stay green through a completely broken UR3e cell. The on-box
`run_*_pick --runs 10` gate is the only real signal.

---

## Rollback

The bring-up is additive and every step is reversible without touching the UR5e cell.

1. **Drop the robot layer.** Omit `--robot-model`, or set `WILLY_PROFILE=sim`, and the cell is the
   UR5e one, byte-identical. `config/robot/robot.ur3e.yaml` can stay on disk; an unreferenced
   layer is inert.
2. **Drop a behaviour flag.** `--radial-closing` is default off, and `planner_margin_mm: 0.0`
   returns the planner's own configuration untouched. Both revert to the previously validated
   behaviour with no code change.
3. **Restore a mesh bundle.** The bundles are committed, so check the previous
   `{model}_collision_meshes.npz` back out. A cuRobo `{model}.yml` lives in the external cuRobo
   install; re-run the builder rather than hand-editing it.
4. **Restore calibration.** Copy back the artifacts saved in Mitigate 7, then re-verify.
5. **If the motion stack is the problem, do not work around it.** Removing the fail-closed check
   would bring back the silent degradation this runbook exists to prevent. Fix the environment, or
   set `WILLY_ALLOW_DEGRADED_MOTION=1` deliberately and treat the resulting numbers as
   non-comparable. The [safety package README](../../src/robot/safety/README.md) says what a
   degraded guard set actually means.

---

Once the geometry is proven here, [real_cell_first_pick.md](real_cell_first_pick.md) takes the same
cell to metal, from a powered-off arm to one verified, logged pick.

See also: [`ext_deps/README.md`](../../ext_deps/README.md) for cuRobo and Coal, and
[`src/willy_sim/`](../../src/willy_sim/README.md) for the simulator cell itself.
