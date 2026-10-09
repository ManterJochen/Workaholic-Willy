# Runbook: the console at the cell

**Scope.** The operator console from the controller simulator to a physical cell: the URSim CB3 checklist
that has to pass before the console may halt a real arm, then the supervised first run at the cell. It
picks up where [real_cell_first_pick.md](real_cell_first_pick.md) ends: the cell already connects, moves,
calibrates and picks from the examples. The cell it is written for is the owner's: a **UR10 (CB3)** planned with
**cuRobo**, a **Hand-E switched as `single_toggle` on tool DO0** (no sensor: every change of the output moves the
jaws), and a **D415 on the wrist** in its robobrain.eye housing. The pages behind it are
[`api/README.md`](../../api/README.md) and [`frontend/README.md`](../../frontend/README.md).

**Where it has run.** On the desk profile, through the browser (the Playwright smoke tests). Against URSim CB3
(a UR10, PolyScope 3.15.8): `probe_halt.py` M0 to M9 on 2026-10-01 and M0 to M5 again on 2026-10-02, and
`probe_console_task.py` C1 to C5 on 2026-10-02. **Not at the cell yet.**

**The one line version.** Measure halt now on URSim first, switch `robot.ur.brake_on_halt` on only after it
passed there and at the cell, and for every first at the cell keep a hand on the emergency stop: "Sofort
anhalten" is not one.

---

## Trigger

Any of:

1. The console runs at a cell for the first time, or after an update of the console, the UR driver's halt path
   or the cell's hand.
2. Before `robot.ur.brake_on_halt: true` goes into a cell's profile.
3. A halt, a stop record, the jaws question or a teach behaves unlike [`api/README.md`](../../api/README.md)
   says.

---

## Diagnose

### 1. What the console needs from the cell's profile

* **The cell's own layer.** The chain ends in it, `--profile ur10,hande,...,cell`, and its robot overlay,
  `robot/robot.cell.yaml` under the config root, exists (one comment line is enough; the loader refuses a layer
  with no file). Git ignores every `*.cell.yaml`, and the console refuses to teach into a layer git does not keep
  out (`no_layer`): `git check-ignore -v config/robot/robot.cell.yaml` names the rule.
* **The carried part.** `safety.planning_world.payload.length_mm`, the longest part's hang past the fingertips.
  Without it every task is refused (`carried_part_not_modelled`) and the ready bar's carried-part light is red:
  between the close and the release only the planner holds the part.
* **The wrist housing at its real size.** A teach screens every pose at once, and an arm that was not handed the
  wrist camera's housing screens nothing (`screen_unavailable`): a pose the housing meets would read clear.
* **`robot.ur.brake_on_halt` stays `false`** until steps 1 to 3 of Mitigate have passed on URSim and step 5 at the
  cell. Off, a halt latches and the move in flight runs to its end; on, it is braked under control.
* `python -m src.config --profile <your cell>,cell` exits 0, and `python -m src.robot.execution.real_cell --check`
  blocks on nothing a person has not decided.

### 2. URSim CB3 comes up differently from the e-Series

```bash
# inside the WSL distro that runs Docker, from the repository root
URSIM_IMAGE=universalrobots/ursim_cb3:latest URSIM_NAME=ursim_cb3 URSIM_FRESH=1 bash scripts/ursim/ursim.sh up UR10
```

Hold a WSL session open first ([`scripts/ursim/README.md`](../../scripts/ursim/README.md)). `.gitattributes`
checks every `*.sh` out with LF, which bash needs, on Windows too; a checkout older than that rule holds a CRLF
copy until `git checkout -- scripts/ursim/ursim.sh`. Then, at the noVNC pendant (http://localhost:6080/vnc.html):

1. a **"Power off"** notice comes first: click **Not now**;
2. power on and release the brakes, from the dashboard (`power on`, `brake release` on port 29999) or the pendant;
3. only then **"Confirm Safety Configuration"** appears: confirm it, or RTDE refuses with "SafetySetup has not been
   confirmed yet".

A CB3 has no Remote/Local switch. **After a protective stop the control script has ended**: the first script
upload after the release can time out ("Failed to start control script, before timeout of 5 seconds"), and the
second works.

### 3. What the probes prove

`scripts/ursim/probe_halt.py` measures "halt now" on the connection, the arm and the planner's executor
(M0 to M9 on 2026-10-01; M0 to M5 again on 2026-10-02, with the same verdicts; M1 gives each day's numbers, and
every other row holds for every run of it):

| Check | What it proves | Measured |
|---|---|---|
| M0 | brakes off: a cross-thread `stop()` does not stop a synchronous move, and the latch lets the move in flight run out and refuses the next | the move arrived anyway, `stop()` returned after 2.3 s; the next move refused, nothing sent |
| M1 | brakes on: the brake's latency | stop sent 0.7 to 8.2 ms after the request and deceleration 18.7 to 37.1 ms after it on 2026-10-01, 0.3 to 5.0 ms and 12.3 to 34.2 ms on 2026-10-02; standing still after 0.26 to 0.75 s by speed on both |
| M2 | the stop point stays on the judged joint line | at most 2.3e-7 rad off the line |
| M3, M3b | 200 watched moveJ and 50 watched moveL, no halt: every one true only at its target | 0 early returns, 0 refused; a watched move no slower than a synchronous one |
| M4 | a 5-waypoint judged path halted in leg 3 sends no later waypoint | cancelled, 3 moveJ sent, the base stopped short of waypoint 3 |
| M5 | the toggle on tool DO0: no output change after a halt, brakes on and off | 0 DO0 edges after the halt; the close refused |
| M6 | a line braked with stopL stays on the line | 0.042 mm off it |
| M7 | a protective stop during a watched move answers false, never true | false; the first reconnect timed out, the second worked |
| M8 | the alternative not taken: the program stopped from the dashboard | still after 175 ms, DO0 unchanged, the control program gone |
| M9 | the latch outlives a disconnect and a connect | a move and a DO write refused until cleared |

`scripts/ursim/probe_console_task.py` drives the console itself (its routes, its runs, the browser's jaws
question) against the UR10, cuRobo planning every motion and the toggle on tool DO0, with DO0's edges read on a
second RTDE connection. Measured on 2026-10-02:

| Check | What it proves | Measured |
|---|---|---|
| C1 | three tasks, a pose place and the return home each | 3 of 3 `finished`, 1 part and exactly 2 DO0 edges each, 31 to 34 s a cycle |
| C2 | halt now in the approach | `halted` where the arm stood, braked in 0.12 s, 0 DO0 edges after; task, Home and Restart refused `halted`; after "the cell is clear" a task `restart_required` and the Restart `jaws_not_confirmed` until the jaws were answered; the Restart's first motion the move home |
| C3 | halt now in the carry | `halted` holding the part; the Restart `part_still_held` until the jaws question was answered, "closed" then "open now", exactly 1 DO0 edge; the Restart counted down 3 s first. Since the review of 2026-10-02 a count that says closed is asked "open now" first and only, and the probe answers that alone: pinned on the doubles, not yet rerun on URSim |
| C4 | a protective stop mid-move | `controller_stopped`; "the cell is clear" and the Restart refused `controller_stopped` until it was released; the control script ended, Disconnect, one Connect refused, the second up; the Restart went home first |
| C5 | Disconnect during the planner start | waited for the start (24 s, `planner_ready`), 0 rad moved, 0 DO0 edges |

**A CB3 URSim is proven a simulator too.** `ur_rtde` reads no serial number below PolyScope 5.6, so the driver
asks the dashboard's own "get serial number" over a connection of its own, and the simulator proof
(`api/telemetry.py`) takes 10 digits ending 9999 (a CB3: `2018309999`) or 11 ending 99999 (an e-Series), on a
loopback address only. The run above read the serial through a scratch shim that did the same before it shipped;
the shipped reading is pinned on a local dashboard double
([`tests/test_ur_cb3_serial.py`](../../tests/test_ur_cb3_serial.py)) and not rerun on URSim yet.

**Not measured on URSim:** a halt with the hand inside a bin (the bed has no camera world, so the path home is
judged against no frame), Disconnect during a jaws question and during a free teach, a pose taught by URSim
freedrive, the teach's largest sample gap while a VLM load or a parse runs, and a Disconnect with brakes off and
a move in flight. Each is a step below.

---

## Mitigate

### The URSim CB3 checklist

1. **Halt now, every check.** With URSim up as Diagnose 2 says:

   ```bash
   python scripts/ursim/probe_halt.py --json halt.json        # M0-M6 (M3b included) and M9
   python scripts/ursim/probe_halt.py --only M7,M8            # the two that stop the controller's program
   ```

   Exit 0 means every check that ran passed. The probe moves the arm and switches tool DO0 without asking, so it
   runs on 127.0.0.1 only and writes its own scratch profile layer (`brake_on_halt: true`, the toggle on tool
   DO0, a declared `payload.length_mm`) into a temporary copy of `config/`.
2. **The console's task.** First move URSim to the probe's home, [-90, -90, -100, -80, 90, 0] deg. Step 1
   leaves wrist 2 at -90 deg, half a turn from that home and outside the joint window the UR10 profile keeps
   about it (`safety.joint_limits.within_half_turn_of_home`): every path would be refused at its first sample,
   C1 would end `failed_in_a_row`, and its stop record would refuse C2 to C4 (`cell_not_cleared`). The probe
   reads where the arm stands before it moves anything, with the console's own joint guard, and refuses to start
   outside that window (exit 2, naming the axis). Jog it there at the pendant's Move tab, or send it there with the
   Python the probes run in (on the simulator only: this move passes no check of this repository):

   ```bash
   python -c "import math, rtde_control; c = rtde_control.RTDEControlInterface('127.0.0.1'); c.moveJ([math.radians(d) for d in (-90, -90, -100, -80, 90, 0)], 0.6, 0.8); c.stopScript()"
   ```

   Then, with the cuRobo environment the planner's sidecar runs in:

   ```bash
   python scripts/ursim/probe_console_task.py --json console_task.json   # C5, then C1-C4
   ```

   Exit 2 where the console proves no simulator (Diagnose 3), and on an arm outside the window. It also
   writes its own layer: the Hand-E as
   `jaw_io` `single_toggle` on tool DO0, `brake_on_halt: true`, `payload.length_mm: 40`, a home inside the
   workspace box, and two taught poses.
3. **The console in the browser, by hand.** Copy the probe's layer (`_LAYER` in `probe_console_task.py`, with
   `{model}` replaced by `ur10`) into a `robot/robot.cell.yaml` and start
   `python -m api --profile ursim,ur10,ursim_curobo,cell`, the arm at that layer's home as step 2 left it; then
   Setup, a task, "Sofort anhalten" during the approach, the stop card, "Zelle ist frei", the jaws question,
   Restart. That layer belongs to the desk that runs URSim, never to the cell PC.
4. **Disconnect at the hard moments**: during a jaws question (it returns at once, the question refused, never
   "open"), and during a free teach where the arm offers freedrive (the arm is held before it is taken down,
   within 10 s).

### The supervised first run at the cell

The owner stands at the emergency stop, the pendant's speed slider turned well down, hands and cables clear.

1. **Pull, declare, restart.** Pull the commit. Declare `safety.planning_world.payload.length_mm` in the cell
   profile, create the cell's own layer if it is missing (Diagnose 1), and restart the planner. The carried-part
   light turns green.
2. **Build and connect.** Build real, read the preview, connect. The Hand-E's jaws question opens in the browser:
   time it, look at the jaws, answer what you see. Where they stand closed, answer "closed", hold the part, then
   "Jetzt öffnen": one change of DO0, and the answer returns within 5 s of the stroke. While a stop from before
   stands (after a restart of the server, say), "Jetzt öffnen" waits for "Zelle ist frei", and a connect whose jaws
   stand closed is refused: say the cell is clear first (Setup offers it once the cell is built), then connect.
   Time the planner start
   (about a minute) until the ready bar is green. Watch the live image's rate and age idle and during a pick:
   LIVE, a frame younger than 2 s.
3. **Teach the place, then a park pose.** Confirm the payload the controller shows (read it again after any tool
   change). For the place pose put the fingertips where the part's bottom should be let go, save, read the verdict.
   Then a park pose. Once, Disconnect while the arm is free: it is held before it comes down. Do not press "Laden"
   or send a command while a person guides the arm: a VLM load (about 6 s) or a parse (about 2 s) shares the
   process, and the teach's largest sample gap under it is still unmeasured.
4. **One task, slowly.** `Einmal`, the default place, at reduced speed. Watch tool DO0 on the pendant's I/O tab:
   exactly 2 changes per part, the close and the release. Measure the release height against the taught fingertip
   height. Then a `Bis leer` task and **"Nach diesem Teil stoppen"**: the part is placed and the arm returns.
5. **Halt now, slowly.** Only once URSim M1 to M5 have passed. "Sofort anhalten" during a slow approach. With
   `brake_on_halt` off the button says "hält vor der nächsten Bewegung": the move in flight ends at its target and
   nothing after it is sent. Then the stop card: "Zelle ist frei", "Backen prüfen" (the jaws question: a count that
   says closed asks "Jetzt öffnen" at once, so hold the part first), Restart; its first motion is the planned move
   home. Once, stop the server while the card stands and start it again: the card is back, uncleared, the build's
   arm is latched and Connect refuses `halted` until "Zelle ist frei"; then the jaws question and Restart. Repeat
   the halt once with the hand inside the bin: jog it clear at the pendant
   first, since the move home is planned against what the camera sees now and the D415 never sees the fingers. Only
   then decide `brake_on_halt`; switched on, repeat this step and check "bremst kontrolliert", and that "Not-Aus
   drücken" never appears on a halt the arm confirmed.
6. **A protective stop** mid-move, provoked the way the cell's safety setup allows: the card says to clear it at
   the pendant, and "Zelle ist frei" is refused until it is. Clear it, Disconnect, Connect (again if the first
   refuses), "Zelle ist frei", the jaws question, Restart.
7. **A bin the camera finds.** A survey alone, then `Einmal` into the bin, 3 parts, `Bis leer`. Move the bin a
   little between parts (it is followed) and then far, or take it away (lost: the part goes back where it was
   gripped, the arm returns, the chat asks). Check the rim and the opening against a ruler, and that no phantom
   obstacle appears at the jaws during the carry.
8. **Commands.** First set Settings → Auftrag → Start to **"Erst die Karte"**: as shipped, Enter starts a task at
   once where the sentence reads clean and the cell is ready (the owner, 2026-10-08), and this step reads
   sentences, it starts nothing. Ten of the owner's own sentences, German and English. Time "Laden", read each
   answer's latency and attempts (`model.latency_ms`, `model.attempts` of `POST /v1/commands/parse`, shown in the
   tech view, where a known sentence reads "bekannter Satz" and a sentence typed again "aus dem Gedächtnis"), watch
   the VRAM with `nvidia-smi` while everything is loaded, and check each card before Start. Set Start back
   afterwards.
9. **Sortieren.** Two bins on the table, two kinds of parts on the mat. Say or type the owner's sentence, "Grüne
   Teile in die gelbe Kiste, rote in die blaue": up to four kinds, each to its own bin or taught pose (two rules may
   share one). The card lists every rule, each kind and its place; Enter starts the whole sort only where every rule
   reads clean, and a rule that needs a look opens the card with its number ("Regel 2: ..."). Before the first pick
   the camera must find every bin, one locate of both per look: a bin it does not find stops the task before anything
   is picked, naming that bin (`target_not_found`). Each part then goes by the rule of the kind its pick went for
   (`task.rule`). Move one bin about 15 cm between two parts: the check before the drop loses it, the task looks for
   it again with the part in the jaws and drops into it where it stands now (`task.target_relocated`); take a bin
   away, and the part goes back where it was gripped and the chat asks. A part no rule claims (another colour, or one
   the detector could not tell apart) stays where it lies and is named at the end (`task.unsorted`). What stops a
   sort before anything moves: a rule that names no kind, one kind in two rules, a cell whose detector grounds no
   phrase (the rehearsal scene), and every refusal of a task of one kind, rule by rule. Not measured yet, so watch for
   it: how well the cell's 8B tells the two colours apart in one call, a bin found again while the part hangs in the
   camera's view, and whether the two bins' labels come back swapped (nothing catches that yet: the yellow bin would
   take the red parts).
10. **The room.** The audience window on the projector (drag it there, F11). Voice output on, "Stimme testen" in
    Settings: note which German voice the cell PC has.
11. **Write the numbers down** (Verify), the date and the commit beside them.

---

## Verify

| Check | Passing means |
|---|---|
| `probe_halt.py` | exit 0 for the default run and for `--only M7,M8` |
| `probe_console_task.py` | exit 0 |
| The jaws question | no default, an answer within 5 s of the stroke, never "open" when nobody answered |
| A task | `finished`, the part where the taught pose says, exactly 2 DO0 changes per part, the arm back home |
| Halt now | the run `halted`, no DO0 change after it, every moving route refused `halted` until "Zelle ist frei" |
| The way back | Restart refused until the cell is clear and the jaws are answered; its first motion the move home |
| A protective stop | "the cell is clear" refused until the pendant released it; the console never clears one |
| Teaching | a refused or unscreened pose not written; the arm held only once still; a Disconnect holds it first |
| The camera place | followed close by, lost far off, a lost bin's part put back |
| Sortieren | every bin found before the first pick, each part in its rule's place, a moved bin found again, a part no rule claims left where it lies |

The measurements to keep: the planner start on the cell PC, the live image's rate and age during a pick, the jaws
question's timing, the release height against the taught one over the first parts, the halt's stop distance at the
cell's speed, the bin's rim and opening against a ruler, the VLM's load time, latency, retries and VRAM, and the
German voice.

---

## Rollback

* **Stop motion**: the physical emergency stop. Nothing in software outranks it, and "Sofort anhalten" is not it.
* **The brake**: `robot.ur.brake_on_halt: false` in the cell profile, and the arm halts before its next motion
  only, as shipped.
* **A pose taught wrong**: teach it again with "replace", or edit `robot/robot.cell.yaml` by hand; the console
  never renames or deletes a pose. The default place: choose another in Setup, or none.
* **A stop record that will not end**: it ends at a Restart's or a Home run's arrival. A restart of the server
  keeps it (`logs/console/stop.<profile chain>.json`, read back uncleared, with a halt nobody cleared), so after a
  restart the stop card is back: "Zelle ist frei", the jaws question, then Restart or Home. A file the server
  cannot read stands as a stop of unknown origin, whose way back is Home. Removing the file with the server
  stopped forgets the stop: only after a person has looked at the arm and the jaws, and only where no way back
  can run.
* **The console itself**: stop the server; the command line and the examples run without it
  ([real_cell_first_pick.md](real_cell_first_pick.md)). Nothing in the shipped tree changes when the console runs:
  it writes only the cell's own layer.

---

## See also

- [`api/README.md`](../../api/README.md): every route, the refusals, halt now, the way back, the jaws question,
  teaching.
- [`frontend/README.md`](../../frontend/README.md): the cockpit, Setup and the audience window.
- [`scripts/ursim/README.md`](../../scripts/ursim/README.md): the probes and the container.
- [Guide 04](../guide/04-robot-and-safety.md), section 8: halt now against the emergency stop.
- [real_cell_first_pick.md](real_cell_first_pick.md) and [cell_bringup.md](cell_bringup.md): the cell before the
  console.
