# `api/`: the operator console

The HTTP and WebSocket surface behind Willy's browser console. Optional in the strict sense: a cell
that never starts this server behaves exactly as it does without it, and nothing under `src/` imports
this package. It is the one tree in this repository that may import a web framework.

```bash
pip install -r requirements.txt

python -m api                             # 127.0.0.1:8000
python -m api --profile console_dummy     # no controller, no camera, no GPU
python -m api --profile ursim,ursim_ur3   # real UR controller software, UR3e kinematics
python -m api --profile ur5e,eth2         # a real bench with two fixed cameras
```

The server binds to `127.0.0.1` and that is not configurable from this entry point. There is no
authentication in front of these endpoints and the machine running them is next to a robot arm, so
making the bind address a flag would turn "expose the cell to the network" into a typo. Remote access
means a reverse proxy with real authentication, decided deliberately.

**The browser console is served by this same process**, from `api/static/`, once
[`frontend/`](../frontend/README.md) has been built (`cd frontend && npm run build`). That directory
is not committed. Absent it, nothing is mounted and this package serves the API alone, which is the
correct behaviour for a headless deployment.

Serving the page from here rather than from a second server is what makes the browser's API base URL
the empty string: there is no host to configure, so no build and no bookmark can aim the console at a
different cell than the one whose server it was opened from.

## What it is for

| | |
|---|---|
| Prepare a cell | the preflight checklist, the config with its provenance, and forms for the handful of values a person measures at a bench |
| Bring the cell up | build, acknowledge what will move, connect (arm before gripper), live telemetry, and the bench diagnostics that move nothing |
| Drive a pick | prompt in, typed or spoken; run lifecycle; the live event stream |
| See what it sees | one image plus the sentence saying whether it is a camera, a drawing or a decision |
| See what happened | runs, logged attempts, the KPI roll-up, and CSV out |
| Show someone | the same data at demo density, at `/demo.html` |

## The rule this package is built around: the console issues tasks, not parameters

A pick can be started, a prompt can be given, a run can be told not to start the next attempt. Nothing
that changes how a motion executes (speed, acceleration, workspace limits, safety toggles) is writable
from a browser, and nothing at all is writable while a run is active, which is a typed `409` and never
a queue.

Three things follow.

**The writable set is an allowlist**, and it is the library's own, in `src/config/edit.py`: the
payload mass and centre of gravity, the three tool-frame keys, a camera rig's serial number, and the
two controller addresses. They are there because they are measurements and site facts rather than
policy.

**Limits and thresholds are read-only here.** Not because a machine could not write the line, but
because in this tree the evidence for a value lives in the comment above it, and a form that writes the
number without showing the comment invites changing a value whose reason nobody remembers.

**There is no emergency-stop endpoint.** A stop that travels over a socket depends on latency, an open
tab and an awake laptop. Offering one would invite relying on it instead of the physical mushroom
button. The one endpoint called `stop` documents itself as not an emergency stop.

## The cell lifecycle

| from | what you call | to |
|---|---|---|
| unbuilt | `POST /v1/cell/build` | built |
| built | `GET /v1/cell/connect-preview` | previewed, holding a single-use token |
| previewed | `POST /v1/cell/connect` with that token | connected: arm, then gripper |
| previewed | any config change | built, and the token is void |
| connected | `POST /v1/cell/disconnect` | built: gripper, then arm |

**Connect is motion on this cell.** A Robotiq activates by sweeping its full finger travel; a vacuum
cup's connect asserts the ejector immediately and drops whatever it is holding. A browser button is
reachable by a stray curl, a replayed request and a reloaded tab, so `POST /v1/cell/connect` accepts
only a token that `connect-preview` issued for this configuration. The token expires after five
minutes, is single-use, and is void the moment any config value changes, because what was acknowledged
is then no longer what would happen. Without one the answer is `428 Precondition Required`.

Three refusals that look like bugs and are not:

**The preview's warnings come from the gripper that was actually built**, not from what the config
asked for. An empty warning list on a null-gripper cell is correct: warning about a finger sweep that
cannot happen teaches an operator to skip the warning that can.

**Connect is a transaction.** Arm first, then gripper, and if the gripper refuses, the arm is
disconnected again. There is no `degraded` state to render, because a cell that came up halfway would
hold a UR controller's single control script while reporting failure.

**A substituted gripper blocks the connect** (`403 no_real_gripper`). A misconfigured end effector, a
Robotiq on a non-UR arm, a vacuum gripper on an arm with no digital I/O, or a vendor with no driver,
yields a working null gripper. Without this refusal the cell would connect, every pick would report
success, and the jaws would close on nothing. The typed reason travels on the gripper object itself.

**One owner per cell.** `CellLock` is a byte range held by an open file handle, keyed on the
controller's address, so the operating system releases it when the process dies. There is no stale lock
and therefore no habit of deleting one. The CLI runner takes the same lock, because a lock only one
side respects protects nothing. It cannot see a second machine on the network. No local lock can, and
the reachability probe in `GET /v1/diagnostics` is the honest complement rather than a substitute.

## Endpoints

Everything is mounted under `/v1`.

| Route | What it answers |
|---|---|
| **Health and configuration** | |
| `GET /v1/health` | the server, and deliberately nothing about the cell |
| `GET /v1/preflight` | the checklist, from the same function the CLI's `--check` calls |
| `GET /v1/config/explain?key=...` | value, type, default, tier, which layer won, and the YAML comment |
| `GET /v1/config/writable` | the guided-write form, described by the library |
| `PATCH /v1/config` | one transaction over a group of keys, all or none |
| **Cell lifecycle** | |
| `GET /v1/cell` | state, arm, gripper, and who holds the cross-process lock |
| `POST /v1/cell/build?rehearse=` | assemble the cell; touches no robot |
| `GET /v1/cell/connect-preview` | what will move, plus the token that acknowledges it |
| `POST /v1/cell/connect` | takes the token; arm then gripper, rolled back together |
| `POST /v1/cell/disconnect` | gripper then arm; idempotent; frees the lock |
| `GET /v1/cell/status` | live telemetry; the receive stream only, unless asked otherwise |
| `GET /v1/diagnostics` | SDKs, motion stack, perception stack, controller reachability; nothing moves |
| `GET /v1/diagnostics/route` | which route a prompt would take, decided from text alone: no GPU, no model, no image |
| **Picking** | |
| `POST /v1/pick` | `{prompt, picks}`, answered `202` with a run id. This moves. |
| `POST /v1/pick/stop` | do not start the next attempt. Not an emergency stop. |
| `GET /v1/runs`, `GET /v1/runs/{id}` | recent runs, and one run |
| `WS /v1/events?run_id=&since_seq=` | replay what you missed, then go live |
| `GET /v1/camera` | what the cell is looking at, and which of three pictures that is |
| `POST /v1/overlay/enable` | opt into the debug render; it costs time per pick |
| `WS /v1/overlay` | each new overlay still, with its age |
| `POST /v1/voice/transcribe` | audio to text. Never starts anything. |
| **History** | |
| `GET /v1/history/kpis` | rolled up with the same function the offline gate uses |
| `GET /v1/history/records` | logged grasp attempts, newest first |
| `GET /v1/history/runs` | this session's last 200 runs, in memory and perishable |
| `GET /v1/history/runs.csv` | one row per run, the report view |
| `GET /v1/history/records.csv` | one row per attempt, the analysis view |

`PATCH` refuses by name: `403 not_writable`, `404 unknown_key`, `422 invalid_value` carrying the
loader's own message, `409 run_active`, `409 cell_connected`. A rejected write leaves every file it
touched byte-identical. It takes a group because the schema has cross-field rules no single write
satisfies: a tool frame that declares an owner while its transform is still identity is rejected, so
the three tool-frame keys only ever validate together. There is no order in which they validate one at
a time.

## Driving a pick, and watching one you were not there for

**Starting returns an id, not a result.** A pick blocks for as long as the arm takes, so a handler that
waited would hold an HTTP connection open across a physical motion, and the browser, a proxy or a
sleeping laptop would give up somewhere in the middle, leaving an operator with no idea whether the arm
was still moving.

**A run survives the browser.** The arm is holding a part, and the correct behaviour when a tab closes
is to finish the pick and put the object down. What makes that survivable for the UI is the sequence
number: a client reconnects with `since_seq` and receives everything after it. If it slept longer than
the ring buffer it gets a `gap` frame saying how many events it will never see, because a short replay
that looked complete would make the UI draw a run that never had those steps.

**Stop is not kill.** `POST /v1/pick/stop` sets a flag the pick loop reads before it begins the next
attempt. A motion already in flight completes. There is nothing stronger here, and a cancelled run ends
with the typed `CANCELLED` outcome rather than a failure, because a stop is a decision and counting it
as a failure would make every press of the button lower the measured pick rate.

**Two audiences, one envelope.** Every event carries `human`, a plain sentence, and `data`, the machine
payload. The person watching a demo reads one and the person diagnosing a failure reads the other.

The library side of this is `PickStage` and the frozen `PickProgress` payload in
[`src/robot/grasping/loop/progress.py`](../src/robot/grasping/loop/progress.py). With no listener
attached the emit is one `is None` test and no payload is constructed, so a cell that never runs this
server pays nothing for the seam.

**Speech goes to the prompt box, not to the arm.** `POST /v1/voice/transcribe` returns text and starts
nothing: a misheard word must not be able to move an arm, so a spoken prompt is confirmed by the same
button, with the same acknowledgement, as a typed one. The recogniser loads on first use, so a console
on a machine without the model still starts and says what is missing instead of failing to boot.

[`audio.py`](audio.py) decodes 16-bit WAV with the standard library, which is what the console itself
uploads, and everything else (webm/opus, mp4, ogg, flac, mp3) through the optional `av` dependency. No
browser records WAV, so that second path is the everyday one. The two failures are answered
differently on purpose: a container this host cannot decode is `501 audio_format_unsupported` and names
the dependency, while bytes that are not usable audio are `422 audio_undecodable`. An operator sent to
their microphone settings for something that is an install has been told the wrong thing.

**A truncated upload is one of those 422s, and it used to be a crash.** A connection dropped
mid-upload delivers a few bytes, and `wave.open()` answers a short read with a bare `EOFError`, which
is not a `wave.Error` and not an `AudioDecodeError`. Measured on a WAV truncated to every length from
0 to 59: lengths 1-7 and 20-35 raised it, and the odd lengths from 45 raised `ValueError` out of
`np.frombuffer` instead. The endpoint reported all of them from its last-resort handler as
`transcription_failed` with the message `EOFError:` and no cause in it. They are now
`audio_undecodable`, and bytes too short to be any container at all (under RIFF's 12-byte header) are
refused as a bad request rather than as a missing decoder.

## The viewfinder: three pictures, never blurred

`GET /v1/camera` is what the console draws as "what the robot sees". It returns one image and one
sentence saying what that image is, because there are three of them and they look identical on a
screen:

| `source` | what it is |
|---|---|
| `camera` | a colour frame off a physical device, taken now |
| `synthetic` | pixels this process drew; a rehearsal cell has no camera at all |
| `overlay` | the grasp render: segmentation, projected gripper, ranked candidates |

There are also four ways to say no picture, each a `200` with a reason and never a `404`, because a
cell with no camera is an ordinary cell and an error code would make an everyday state look like a
fault: `not_built`, `pick_owns_camera`, `no_colour_source`, `encode_failed`.

The kind is declared by the perception source, never inferred from the array, and a source that does
not declare one is described as unknown rather than as a camera.

**An overlay's age is only ever a lower bound, and the payload says which kind it is.** Nothing in the
stack stamps a render time on the image, so all a server can measure is when it first saw those bytes.
The first overlay a process sees may predate it, and the payload then carries `age_is_exact: false` and
a sentence saying "at least this old, possibly much older". Once the server has watched one image
replace another the age is real to within a poll. `WS /v1/overlay` does not make this distinction: it
seeds its clock when the socket is accepted.

**The pick owns the camera, and this endpoint stands down.** The camera package holds no lock and a
pick runs on its own thread, so two grabs on one device pipeline would split the frame stream between
the viewer and the robot. While a run is active the viewfinder never touches the device and serves the
overlay instead, which during a pick is the more informative picture anyway. The exclusion is by run
state and not a mutex, because a correct mutex would have to live inside the camera package on the
pick's own hot path. One frame can still land in the viewer instead of the pick; the pick opens every
acquisition by discarding warm-up frames, so nothing it relies on changes.

**Every camera this server opens, it closes.** Building twice is a normal thing to do: build, read the
refusal, fix a key, build again. Two rules make the second build survive the first. The session state
is checked before anything is acquired, so a rebuild refused with `wrong_state` claims no device. And
adopting a new service closes the one it replaces, because a dropped service with its pipeline still
streaming makes the next start on that device fail on real hardware, and the old service sits in a
reference cycle so refcounting would not collect it either. The app's lifespan releases the session on
the way out, which matters most under `--reload`, where a worker is replaced while the parent lives on.

## History: two sources, never blurred

| | lives in | survives a restart | includes the CLI runner |
|---|---|---|---|
| Runs | memory, and is rich: candidate counts, scores, the motion chain | no | no |
| Records | a JSONL file | yes | yes |

Every response says which it came from. Mixing them would let an operator conclude that something is
stored which is not.

**The console turns record logging on itself.** `grasping.record_log_path` is null in the shipped
config, so a console-driven cell would keep no history at all and a bring-up that went wrong would
leave nothing to diagnose from. The console writes to `logs/console/grasp_records.jsonl` without
touching the YAML. A cell that never runs the console is unaffected, and a configured path always wins.

**A rate over an empty denominator is not a measurement.** The ratio helper returns `0.0` for a zero
denominator, so such a rate would arrive looking exactly like a measurement of zero, reading as perfect
or as broken depending which rate it is, and both are claims nobody made. Four KPIs are therefore
withheld and named in an `unmeasurable` map instead, one always and three decided per record set by
checking the denominator rather than the result:

| KPI | withheld when | because |
|---|---|---|
| `false_positive_grasp_rate` | always | it needs an independent post-grasp re-check this stack does not have |
| `dense_recovery_success_rate` | no record ran in a dense mode | the shipped mode is `auto`, so the denominator is empty by construction |
| `first_attempt_success_rate` | no record carries a recovery action | it then equals `pick_success_rate`, the same number printed twice |
| `median_cycle_time_s` | it is absent | nothing on the production path writes that field |

A recovery rate of zero over two hundred dense attempts is a real and alarming measurement; the same
zero over none of them is not a measurement at all, and only the input tells the two apart.

The duration to quote is `median_attempt_seconds`, which is stamped by the pick itself on every attempt
regardless of config. Everything else comes from `compute_kpis`, the same function
`python -m src.robot.grasping.replay --records` uses. A console with its own arithmetic would eventually
disagree with the offline gate and nobody could say which number was real.

## What the status panel costs

`GET /v1/cell/status` reads the RTDE output stream, which the controller is already broadcasting, so
polling it costs a running motion nothing. Two neighbouring reads behave differently despite
identical-looking signatures:

| read | cost | offered |
|---|---|---|
| TCP pose, joints, wrench | free; the controller already broadcasts it | always |
| robot status | four cheap enum reads plus a dashboard socket round trip every call | opt-in with `?include_controller_state=true`, and the response says which reading you got |
| joint torques | goes through the control interface, the same register space a running pick uses | never, at any price |

There is no counterpart that clears a protective stop, though it sits one method away on the same
Protocol. That is done at the pendant, where the arm is visible, for the same reason there is no
emergency-stop button here.

## One envelope, for every failure

Every error leaves here as `{code, message, detail}`, including the router's own `404` and including a
request-validation failure, which arrives as `code: "bad_request"` rather than as the framework's
default validation shape. `ErrorOut` is declared as the app's default response, so it appears in the
OpenAPI document and a generated client can type the failure path as well as the success path.

## Layout

| file | what it owns |
|---|---|
| `app.py` | the app, the `/v1` mount, the single error envelope, and mounting the built console bundle |
| `cell.py` | the session: config root, profile chain, the built cell, and the run lock everything guards against |
| `lifecycle.py` | the state machine (build, preview, token, connect, disconnect), the rollback, and closing the camera a rebuild replaces |
| `telemetry.py` | what one panel tick may read, and what each read actually costs |
| `schemas.py` | wire types: serialisation only, no judgement of their own |
| `events.py` | the event stream: per-run sequence numbers, a bounded history, catch-up by sequence, and `forget()` for a run that has left the console |
| `runs.py` | a run on its own thread, the sentences the operator reads, and the 200-run retention window |
| `history.py` | reading the record log, the KPI roll-up, and the CSV writers |
| `viewfinder.py` | resolving the honest picture: camera against synthetic against overlay, and the four refusals |
| `audio.py` | decoding an uploaded recording, and the two ways it can refuse |
| `constants.py` | the log directory and the per-module log-file names, in one place, so the set of files this server writes is readable without grepping for it |
| `routers/preflight.py` | `GET /v1/preflight` |
| `routers/config.py` | the config explain, writable and patch routes |
| `routers/cell.py` | the cell lifecycle and `GET /v1/cell/status` |
| `routers/diagnostics.py` | `GET /v1/diagnostics` and the route preview |
| `routers/pick.py` | start, stop, list runs, and the events socket |
| `routers/media.py` | `GET /v1/camera`, the overlay socket and speech to text |
| `routers/history.py` | the history routes |
| `__main__.py` | `python -m api`, bound to 127.0.0.1 and not configurable off it; the port is checked and claimed before the banner claims to serve |

**Each module that does something logs to its own file** under `logs/api/`. Modules that only hold data
or delegate have no logger, because an empty rotating file makes a log directory harder to read, not
easier. Two things are deliberately absent from those files: the connect token, which is what
authorises a motion and would outlive a browser tab in a log, and the polled reads, which log when the
answer changes rather than once per request, so a log stays a record of events rather than a frame
counter.

**Routers stay thin deliberately.** When one starts deciding something, such as what counts as blocking
or whether a value is allowed, that decision belongs under `src/`, where the CLI reaches it too. Four
such decisions live there rather than here:

| decision | lives in |
|---|---|
| what may be written, and how, without destroying the file | [`src/config/edit.py`](../src/config/edit.py) |
| what blocks | [`src/robot/execution/real_cell/preflight.py`](../src/robot/execution/real_cell/preflight.py) |
| who owns a cell, with the CLI runner taking the same lock | [`src/robot/execution/cell_lock.py`](../src/robot/execution/cell_lock.py) |
| how a cell is assembled, with console and CLI making the same one call | [`src/robot/execution/autonomous_grasp/cells.py`](../src/robot/execution/autonomous_grasp/cells.py) |

## Two things that will bite you

**The profile chain is process-global.** Setting the active profile mutates an environment variable, so
the console's own config accessor is the only place that touches it, under a lock, restoring what was
there. A handler that set the chain inline would let one request read another's.

**One process owns the cell.** The CLI runners and this server must not both hold the driver. The
server refuses to build a cell whose driver is already claimed: refusal, not arbitration.

## What has and has not been proved

The endpoints are exercised against a copy of the real config tree, and the preflight this server
renders is asserted equal to the CLI's, row for row. The whole console has been driven end to end
against URSim, which is real UR controller software, through preflight, build, connect-preview,
connect, live telemetry, a pick, history and disconnect.

That leaves the physical half untouched. **No endpoint here has ever run against a physical arm, a
physical gripper or a real camera frame.**

`--profile console_dummy` is the hardware-free cell this console is developed and reviewed against: a
dummy arm and gripper, paired with `POST /v1/cell/build?rehearse=true` for a desk scene instead of a
camera. It exercises the console end to end in seconds and proves nothing whatsoever about grasping,
because a dummy arm accepts every motion and reports success. The vendor field on the cell response and
the simulated flag on the telemetry response are how the UI is required to say so.

## See also

- [`frontend/README.md`](../frontend/README.md), the browser face of every endpoint here
- [`src/config/README.md`](../src/config/README.md), the config system these forms read and write
- [`scripts/ursim/README.md`](../scripts/ursim/README.md), bringing up the controller software this was measured against
