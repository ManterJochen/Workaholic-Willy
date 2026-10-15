# faults: a person acknowledging a fault at the cell

```bash
python -m src.robot.faults acknowledge --profile ursim          # asks at the prompt
python -m src.robot.faults acknowledge --profile ursim --yes    # the person at the cell says yes here
```

The one door to `SupportsRobotStatus.acknowledge_fault` (P0i i4). Recovery is never automatic (owner, 2026-09-19):
nothing in `robot/execution`, `robot/grasping`, `robot/safety` or `api/` calls it, and the console shows a fault and
this command beside it and clears nothing (owner, 2026-09-26). `tests/test_a_fault_has_one_door.py` holds every caller
in this repository's code to this package.

| File | What it holds |
| --- | --- |
| `acknowledge.py` | `FaultAcknowledgement.from_tree(loaded, confirm=...)`, one verb `run()`, a frozen `AcknowledgementReport` (`render()`, `to_dict()`, `exit_code`), and `acknowledge_command(profile)`, the command the console shows |
| `__main__.py` | the command line: `acknowledge`, `--profile`, `--data-dir`, `--yes` |

`from_tree` builds the arm alone, and builds it where a wrist camera's body cannot be placed yet: the arm is never
commanded here. What `run()` does, in order: asks the robot's `acknowledge fault` ability row and no other (an arm that
reads no controller state, or reads it only over a connection, is refused with the row's sentence); takes the cell
lock (held by the console while it is connected, so disconnect it first) and watches the arm alone (`Robot.watching`,
`SupportsWatching`), never connecting it; reads the controller's state (`ArmFault.from_status`); answers `no_fault`
where there is none, and where the controller is ready and only runs no control program, since a watch never starts
one; answers `not_acknowledgeable` with what a person does where no software clears the fault (an emergency button, a
controller powered off); asks the person, with the fault's own sentence, through `confirm`, which has no default; and
only then acknowledges. It closes the watch and gives the lock back on every way out, and never raises for a refusal.

Why it watches: a full connect opens the UR control interface, whose constructor uploads a control script, and a
controller in a protective stop starts no external control program, so a door that connects is refused at the very
stop it is for (measured on aurora's URSim, 2026-09-26, before the watch: 1 of 4). The transport Protocol `WatchesController`
reads the state and releases a stop and never starts a control program (the UR: RTDE receive and the dashboard), the
planned arm offers it as `SupportsWatching` (`PlannedWatching`), and while it watches the transport is not connected,
so every motion verb refuses with nothing sent.

Exit codes: 0 acknowledged, or no fault; 1 refused, declined, not acknowledgeable or not taken; 2 usage.

Measured on URSim, 2026-10-15, on the CB3 UR10 the owner's cell matches (PolyScope 3.15.8): `scripts/ursim/
probe_acknowledge.py` 4 of 4. A real protective stop; the door without a yes is DECLINED (it reads an answer only
from a terminal) and the stop stays; with `--yes` it is ACKNOWLEDGED and the dashboard reads NORMAL; a second run
finds no fault. Never on a physical arm.
