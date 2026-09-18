# Runbook: bringing up a cell

**Scope.** Standing a cell up on a robot the base tree does not describe, from its configuration layer to
an arm that answers and stands where its configuration says it stands. The five steps are the same for
every robot: profile, desk check, simulator and connect under Mitigate, then check under Verify. What
differs between vendors is the controller simulator and how the planner's geometry is built for an arm,
so each vendor has its own section at the end, and [UR](#ur) is the first. The physical first pick
continues in [real_cell_first_pick.md](real_cell_first_pick.md), the hand in
[your_own_gripper.md](your_own_gripper.md) and [hande_gripper_bringup.md](hande_gripper_bringup.md).

**Why it exists.** Almost nothing here is about the arm itself. It is about the half dozen independent
places that each name a robot (the configuration layer, the kinematics, the collision geometry, the
planner's description, the simulator's asset and the telemetry), and what happens when one of them still
names the previous robot. Each such failure first presents as something else entirely: bad grasping, bad
perception, a flaky planner.

**The one line version.** A new arm is not a scaled old one. Its reach, the clearance its planner can
absorb and the jaw orientations it can reach all differ, and each of them is a separate configuration
value that has to be measured on that arm rather than scaled.

---

## Trigger

Any of:

1. Standing a cell up on a robot, or a robot model, this stack has not run before.
2. A profile refuses at load, because its layer is missing or does not claim the robot.
3. A simulated cell prints the motion-stack-incomplete box. See Diagnose 1.
4. A new cell's pick rate is non-deterministic on a static scene, which is the signature of a degenerate
   grasp axis meeting an arm that cannot reach every orientation. See Diagnose 4.
5. Any pick-path change, on any robot: re-run every cell's gate before believing a number.

---

## Diagnose

**1. Is the motion stack actually the configured one?**

Every motion goes through the planner and the exact-mesh collision engine, both per robot, and neither
has a safe fallback: blind IK proposes self-colliding branches the guard then rejects, which reads as bad
grasping, and the capsule proxy over-rejects legitimate reach-down grasps. A real cell whose planner
cannot start refuses every planned move. A simulated cell fails closed in `bootstrap_sim_cell` and prints
a box naming what is missing; that probe is a filesystem check that runs before the Isaac boot, so the
gap shows in milliseconds rather than after a start-up and a run whose numbers describe a different
motion stack.

```bash
# both engines install into ext_deps/ via scripts/ext_deps/install.ps1
python -m src.robot.safety.planning --check --profile <your cell>    # exit 0 means fully anchored
python -m src.robot.safety.planning --doctor --profile <your cell>   # and that they actually load
```

Both take the hand from the profile, or from `--hand <hand>` when the profile names none. In simulation,
`WILLY_ALLOW_DEGRADED_MOTION=1` proceeds anyway. Numbers from such a run describe a different system, and
they are stamped `degraded_engines` in the records so they cannot be mistaken for the configured cell's
later.

**2. Does the boot banner name the robot you think it does?**

The planner's description and the collision-mesh bundle are both per arm, so a bundle present for
another arm says nothing about this cell. The planner banner and the doctor name the ones they loaded,
with the model and the key that chose it, rather than leaving it to be assumed. A name that is not your
arm is the first thing to fix, before any pick rate means anything. What `AVAILABLE` still does not mean
is that the descriptor was built; that is a separate check, and
[real_cell_first_pick.md](real_cell_first_pick.md) Diagnose 6 is where it lives.

**3. Can the arm reach the scene, and can the camera see it?**

Both audits run before a simulated cell boots and print only when something is wrong. The reach audit
measures every configured Cartesian point against the model's working sphere about the shoulder, which
sits `d1` above the base. Measuring from the base origin instead would over-state the envelope by that
height and wave through poses the arm cannot touch.

Known gap: the audit checks Cartesian points, `scene_setup.object`, `scene_setup.marker`, `safe_pose`,
the worst eye-in-hand viewpoint and the worst workspace corner. `home_joint_positions` and
`park_joint_positions` are joint poses and are not covered. Check those by hand. See Mitigate 1.

**4. Non-deterministic pick rate on a static scene?**

The symptom is the same scene, the same selected score, and a run-to-run mix of successes and
`MotionStatus.TIMEOUT`. The cause is that a near-isotropic silhouette, a cube seen from above, has a
numerically degenerate principal axis, so the chosen jaw orientation flips between runs. That only
becomes a failure on an arm that cannot plan one of the two orientations another arm absorbs.

Declare `robot.grasping.isotropic_radial_closing: true` in your cell's layer, and a symmetric object's
free yaw goes to the radial direction. Only near-isotropic silhouettes and near-vertical approaches are
touched; an elongated object's axis carries real information and is left alone. Do not instead lower the
standoff: the pick still has to lift, so an unreachable orientation simply fails at the retreat instead of
at the approach.

**5. Losing picks to self-collision rejections just under the margin?**

`forearm|wrist_2: mesh distance 9.44 mm < 10.000 mm` is not a bad grasp. The planner plans against
spheres and the guard checks exact meshes, so without
[`planner_margin_mm`](../../src/config/schema/robot/safety_schema.py) the two disagree by fractions of a
millimetre and the plan loses. See Mitigate 2.

---

## Mitigate

### 1. Profile

Give the robot its own configuration layer: a `*.<your cell>.yaml` file beside each file it changes in
`config/`. Then ask the tree what it made of it:

```bash
WILLY_PROFILE=<your cell> python -m src.config                             # the tree loads, or the line names file and key
python -m src.config decisions --section robot --profile <your cell>       # what your layer decided, with file and line
python -m src.config explain robot.workspace_limits --profile <your cell>  # one key, and where it was set
```

Profiles compose, so `WILLY_PROFILE=sim,<your cell>` applies every `*.sim.yaml` overlay and then yours on
top. Do not fork the simulator overlays into a profile of their own: they carry measured values, such as
the detector dtype the vision pick depends on, and a second copy will drift.

Re-anchor the scene, and check the joint poses too. Keep every configured position under about 85 % of
the arm's reach. Past that the arm is near straight with almost no orientation freedom left, which is
exactly what a top-down grasp needs. Then check `home_joint_positions` and `park_joint_positions` by hand,
because the audit does not: they are joint poses, and a pose inherited from a longer arm can put the TCP
outside the shorter arm's sphere while every audit stays quiet.

Protect the calibration artifacts. The simulator writes them under `logs/calibration/<robot_model>/`,
namespaced per robot so one arm's run cannot overwrite another's in place; the real cell writes under
`calibration/real`, keyed by `rig_id`. Neither directory is tracked. Copy an artifact aside before moving
a camera ([calibration-setup.md](../calibration-setup.md)).

From Python: [`examples/real_robot/01_load_your_cell.py`](../../examples/real_robot/01_load_your_cell.py).

### 2. Desk check

```bash
WILLY_PROFILE=<your cell> python -m src.robot.execution.real_cell --check
python -m src.robot.safety.planning --doctor --profile <your cell>
WILLY_PROFILE=<your cell> python -m src.robot.execution.real_cell --start-planner   # a planner cell only
```

`--check` lists every stop-the-cell condition decidable at a desk, each with its fix, and exits 1 while
anything blocks; the fixes go into your layer. The planner and the guard need the arm's geometry, a
planner description and a collision-mesh bundle, built once per arm by the vendor's tooling (see the
vendor section). `--start-planner` then starts the planner the way the first planned move does, with
every refusal that move meets, and needs neither a camera nor a controller: it exits 0 when the planner
started.

Tell the planner what the guard will demand, with a measured, per-robot value:
`safety.self_collision.planner_margin_mm`. Do not derive it from `min_distance_mm`. How much margin a
planner can absorb depends on how tightly its sphere model fits that arm, and deriving the value from the
guard's distance looks obviously right and can take a cell to zero picks. Sweep the real planner for any
new robot before choosing. A planner cell starts its planner only with a committed evidence file measured
at its arm, hand, plates, placement and margin; the checklist's `planner margin` row names the file it
read.

From Python: [`examples/real_robot/02_check_the_cell_at_a_desk.py`](../../examples/real_robot/02_check_the_cell_at_a_desk.py).

### 3. Simulator

Two simulators answer two different questions, and a new cell needs both before it meets metal.

* The vendor's controller simulator runs the real controller software against a simulated arm. It proves
  the call path: the protocol, the safety modes, the I/O registers. It proves nothing about the wiring.
  The vendor section says how to install and start it, and step 4 connects to it.
* Isaac Sim proves the geometry: reach, collision and the pick itself, with the arm's own meshes and the
  planner in the loop. It runs under Isaac's own interpreter ([isaac-ready.md](../isaac-ready.md)).

```bash
python -m src.willy_sim.run_m1_pick --robot-model <model> --runs 10   # known-pose picks
python -m src.willy_sim.run_m2_pick --robot-model <model> --runs 10   # real vision in the loop
python -m src.willy_sim.run_m1_pick --runs 10                         # the base cell, as a control
```

`--robot-model` stacks the arm's layer on the sim profile, the chain `sim,<model>`, and takes the arms
whose vendor section provides an Isaac asset. Run the base cell as well: a change that fixes one robot
can cost another, and only measuring shows it.

### 4. Connect

First to the controller simulator, under the profile the vendor section names for it, then to the real
controller under your cell's profile:

```bash
WILLY_PROFILE=<your cell> python scripts/checks/cell_bringup.py --live
```

It connects the arm alone under the cell lock, reads its pose back and holds that pose against the
workspace box the guard will use. Nothing moves. Exit 0: the arm answered and stands inside its box.
Exit 1: it answered and disagrees with its own configuration. Exit 2: there is nothing to connect to, or
the driver refused the configuration before it opened a socket, and the line above says which.

From Python, [`examples/real_robot/03_connect_and_move.py`](../../examples/real_robot/03_connect_and_move.py)
connects the same way and then moves the arm.

---

## Verify

The fifth step, check: every step above again, in order, on the finished layer.

```bash
WILLY_PROFILE=<your cell> python -m src.config                                # exit 0
python -m src.robot.safety.planning --doctor --profile <your cell>            # exit 0
WILLY_PROFILE=<your cell> python -m src.robot.execution.real_cell --check     # only camera rows left
python -m src.willy_sim.run_m1_pick --robot-model <model> --runs 10           # the gate, in Isaac
WILLY_PROFILE=<your cell> python scripts/checks/cell_bringup.py --live        # exit 0
```

Accept when the tree loads, both engines load for this arm, `--check` blocks only on rows a calibration
fills in, which [real_cell_first_pick.md](real_cell_first_pick.md) does next, and each Isaac run meets
the configured gate with zero safety rejections. That gate is `sim.pass_fraction`, which ships at 0.8,
and it does not take the service's own success report as sufficient evidence either: it additionally
requires a lift measured from the physics to clear `sim.lift_threshold_mm`, which ships at 50.0. Last,
the bring-up check exits 0 against the controller simulator and then against the real controller.

Keep two robots' telemetry apart. A real cell's records carry `robot_vendor` and `robot_model`; a
simulated cell's records also carry `gripper_mount` and `degraded_engines`, stamped by `cell_identity()`.
Confirm a new cell's JSONL carries them before collecting a dataset, because the contamination is
retroactive and invisible.

The soak gate, `python -m src.robot.grasping.replay --soak-report`, is robot-agnostic and synthetic. It
proves telemetry and KPI consistency, not grasp quality, and it will stay green through a completely
broken cell. The Isaac gate and the bring-up check are the signals that describe this cell.

---

## Rollback

The bring-up is additive, and every step is reversible without touching another cell.

1. **Drop the layer.** Stop setting `WILLY_PROFILE`, or drop your layer from the chain, and the cell is
   the base tree again. The `*.<your cell>.yaml` files can stay on disk; an unreferenced layer is inert.
2. **Drop a behaviour key.** `isotropic_radial_closing` is default off. The planner margin has no neutral
   value: a planner cell starts its planner only with a committed evidence file measured at its margin,
   so the planner's rollback is the arm's `motion_planner: ik`. Neither needs a code change.
3. **Restore geometry.** The collision-mesh bundles are committed, so check the previous one back out. A
   planner description lives in the planner's own install; re-run its builder rather than hand-editing
   it.
4. **Restore calibration.** Copy back the artifacts saved in Mitigate 1, then re-verify with
   [calibration-setup.md](../calibration-setup.md).
5. **If the motion stack is the problem, do not work around it.** Removing the fail-closed check would
   bring back the silent degradation this runbook exists to prevent. Fix the environment, or set
   `WILLY_ALLOW_DEGRADED_MOTION=1` in simulation deliberately and treat the resulting numbers as
   non-comparable. The [safety package README](../../src/robot/safety/README.md) says what a degraded
   guard set actually means.

---

## Vendor sections

Each vendor that joins adds one section here with the same five parts, so a cell on any robot reads the
same way: the controller simulator and how to install and start it, its traps, the profile that points a
cell at it, the geometry the planner and the guard need for its arms, and the models it covers.

### UR

**The controller simulator: URSim.** Universal Robots' offline simulator runs the real controller
software against a simulated arm, in Docker inside WSL2. Docker is installed from a signed repository,
deliberately not with `curl https://get.docker.com | sh`. The convenience script is Docker's own and
works, but it pipes an unpinned remote script into a root shell. The repository route installs the same
packages from a GPG-signed source that apt keeps verifying on every future update, and this box later
talks to a robot controller.

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

Then `scripts/ursim/ursim.sh {up MODEL|down|status}`, where `MODEL` is `UR5` or `UR3`, and
`URSIM_FRESH=1` recreates the controller for a clean state. The
[`scripts/ursim/` README](../../scripts/ursim/README.md) carries the container's failure modes and the
ordered probes that prove this stack talks to the controller, from the raw SDK up to a pick loop meeting
a protective stop.

**Its traps.**

* The WSL session owns `dockerd`. Closing the last `wsl.exe` session takes the daemon and every running
  container with it, some 30 to 60 s after the start and with nothing in `docker logs`, so hold a real,
  durable `wsl.exe` process open for as long as URSim must run. A backgrounded process inside a throwaway
  `wsl -e bash -lc` is not enough, because it dies with the session it was meant to outlive:

  ```powershell
  Start-Process wsl.exe -ArgumentList "-d","<distro>","-e","sleep","infinity" -WindowStyle Hidden
  ```

  When diagnosing a dead container, read the journal with `-b`, or you read a previous boot's log.
* Every start comes up in Local mode, and in Local mode the controller never runs an externally sent
  program. Enable Remote Control at `http://localhost:6080/vnc.html` (Settings, System, Remote Control),
  then switch the selector at the top right from Local to Remote, after every start. A real controller
  needs the same, powered, with its brakes released and no pendant program owning it.

**The profile.** `WILLY_PROFILE=ursim` points the driver at the container on localhost and declares a
tool frame and a payload the simulator accepts; they are a simulator's numbers, never a cell's.
`ursim,ursim_ur3` names a UR3e controller (start it with `URSIM_FRESH=1 scripts/ursim/ursim.sh up UR3`),
and `ursim,ursim_curobo` turns the planner on; with no camera there, every motion has to decline the
camera world or it is refused before planning. `config/robot/robot.ursim.yaml`, `robot.ursim_ur3.yaml`
and `robot.ursim_curobo.yaml` are those layers. For a cell of your own, `config/robot/robot.ur3e.yaml`
records the resulting radius and percentage of reach beside each pose it sets, which is the form to copy.

**The geometry.**

```bash
ext_deps/curobo_env/python.exe scripts/curobo/fetch_ur_meshes.py         # the pinned UR arm meshes, once per box
ext_deps/curobo_env/python.exe scripts/curobo/build_ur_config.py <model>  # -> {curobo content}/configs/robot/willy_<model>.yml
python scripts/isaac/bake_ur_meshes_from_urdf.py <model> --write          # -> src/robot/safety/data/
```

cuRobo ships the meshes of ur5e and ur10e only, so without the fetch any other arm's build refuses with
every mesh reference missing from disk. `scripts/ext_deps/install.ps1` runs the fetch and then builds
every UR arm once. The descriptor carries no hand: the planner adds the one this cell names when its
sidecar starts, so a cell that changes hands needs no rebuild.

The bake is self-validating, and that is the gate: it compares the fresh bake with the bundle already
committed for the same model and refuses to write a difference nobody named, because a wrong frame would
silently corrupt a fail-closed guard. Run it without `--write` first. It needs no simulator and no GPU:
the geometry is the collision STL files of Universal Robots, pinned to one upstream commit. The evidence
file a planner starts on is measured by `scripts/curobo/matrix_gate.py`, step 5 of
[your_own_gripper.md](your_own_gripper.md).

The planner margin was measured across this family. In a planner sweep a UR5e planned at the 10 mm
`min_distance_mm` the base configuration sets, while a UR3e's thinner links read as permanently
self-colliding once every sphere is inflated by half the margin, and it found no plan at all above
roughly 6 mm. That sweep predates the refitted sphere map: against the refitted map every UR runs
4.0 mm, and `robot.sim.yaml` carries the measurement.

**The models.** [ur_family_bringup.md](ur_family_bringup.md) lists the UR arms this stack models, what
each has, how to add one, and the trap between a CB-series arm and its e-series namesake, which report
the same model string to a controller.

---

Once the arm answers, [real_cell_first_pick.md](real_cell_first_pick.md) takes the same cell to metal,
from a powered-off arm to one verified, logged pick.

See also: [`ext_deps/README.md`](../../ext_deps/README.md) for cuRobo and Coal,
[`src/willy_sim/`](../../src/willy_sim/README.md) for the simulator cell itself,
[isaac-ready.md](../isaac-ready.md) for Isaac Sim, and [cli.md](../cli.md) for every command above.
