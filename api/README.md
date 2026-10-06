# Operator console server (`api/`)

The HTTP and WebSocket server behind the browser console. It checks a cell, builds and connects it, runs
**tasks** from a typed or spoken command (pick a part, set it down at a taught pose or in a bin the camera
finds, return, once or until nothing is left), shows the live image, and brings the cell back after a stop.
It needs the packages in `requirements.txt` (FastAPI and uvicorn are pinned there) and a config tree, and it
serves on 127.0.0.1 only, with no login. API version **0.2.0**.

Nothing under `src/` imports this package, so a cell that never starts it behaves exactly the same.
It builds a cell with the same calls `Cell` makes and runs a task with the library's own `run_task`, so the
cell the browser drives is the cell your code drives. The pages themselves are in
[`frontend/`](../frontend/README.md).

**Three promises.** Nothing moves but on a request that says it moves (the **Moves** column below), and the
pages send one only on a person's click on a button that names the motion, bar the one exception the owner
chose: a greeting typed in the chat ("Hallo Willy") waves at once where the app config says `direct`, the
shipped default since 2026-10-06 (`runtime.greeting.wave`; `confirm` asks first), two swings of the wrist the
exact guard judges, and never a way back after a stop. Nothing moves on its own after a
stop: a person says the cell is clear, then chooses Restart or Home, and the stop outlives a restart of the
server (`python -m api` keeps it in a file). And there is **no emergency stop endpoint**: "halt now" stops the
run and latches the arm, so nothing after the move in flight is sent, and only with `robot.ur.brake_on_halt`
on (off as shipped) is that move braked under control; the red button at the cell stays the safety stop.

## Try it without hardware

```bash
python -m api --profile console_dummy
```

`console_dummy` is the desk profile: a dummy arm and a dummy hand, no controller, no camera, no GPU, and two
poses of its own, **"Ablage links"** (the default place) and **"Parkposition"**, written by hand because a
dummy arm cannot be guided to teach one. So a task runs as shipped. The server prints the preflight summary
(`0 blocking`) and `serving http://127.0.0.1:8000`. Open http://127.0.0.1:8000/docs for the interactive
OpenAPI page, or http://127.0.0.1:8000 once the frontend is built.

The same flow as the cockpit, from a script, with `httpx` (also pinned in `requirements.txt`):

```python
import time

import httpx

with httpx.Client(base_url="http://127.0.0.1:8000", timeout=60.0) as api:
    api.post("/v1/cell/build", params={"rehearse": True}).raise_for_status()  # a desk scene, no camera
    preview = api.get("/v1/cell/connect-preview").json()                     # what connecting will move
    api.post("/v1/cell/connect", json={"token": preview["token"]}).raise_for_status()
    print(api.get("/v1/cell/readiness").json()["ready"])                     # True: the ready bar is green
    run = api.post("/v1/task", json={"object": "", "place": {"kind": "pose", "pose": None}}).json()
    while api.get(f"/v1/runs/{run['id']}").json()["state"] == "running":     # 202 came back at once
        time.sleep(0.2)
    print(api.get(f"/v1/runs/{run['id']}").json()["stop_code"])              # finished: picked, placed, home
```

`WS /v1/events?run_id=<id>` replays the run's events. A dummy arm accepts every motion and reports success,
so a desk run proves that the console and the library agree and says nothing about grasping.

## Your own cell

1. Start the console on your cell's profile, ending in the cell's **own layer**:
   `python -m api --profile <your cell>,cell`. The layer is `robot/robot.cell.yaml` under the config root
   (one comment line is enough to start: the loader refuses a layer with no file), git-ignored, and it is
   where the poses taught at the cell are written. A profile is the name of your
   cell's layer in the config tree ([guide 01](../docs/guide/01-configuration.md)); without `--profile` the
   console uses `WILLY_PROFILE`, then the base tree.
2. Work down the check in Setup, or `GET /v1/preflight`. Every row carries its fix, and the verdicts are the
   ones `python -m src.robot.execution.real_cell --check` prints.
3. Write the values a person measures at the bench in Settings or with `PATCH /v1/config`
   (see [What you can change here](#what-you-can-change-here)). Everything else is edited in YAML.
4. Build, read the connect preview, connect. A toggle hand asks where its jaws stand **in the browser**
   ([The jaws question](#the-jaws-question-in-the-browser)), and a cuRobo arm starts its planner by itself,
   about a minute, moving nothing.
5. Teach the place pose by hand in Setup ([Poses and teaching](#poses-and-teaching)), declare
   `safety.planning_world.payload.length_mm`, and give the cockpit its first command.

The procedures are the runbooks [bringing up a cell](../docs/runbooks/cell_bringup.md),
[the first pick on a physical arm](../docs/runbooks/real_cell_first_pick.md) and
[the console at the cell](../docs/runbooks/console_at_the_cell.md). The same steps from your own code are in
[`examples/`](../examples/README.md).

> [!WARNING]
> `POST /v1/cell/connect` moves hardware. A Robotiq activates by sweeping its full finger travel; a
> vacuum cup's connect switches the ejector on at once and releases whatever it holds; an I/O jaw may
> open. The connect therefore takes only a token from `GET /v1/cell/connect-preview`, which lists what
> will move. The token is single use, expires after five minutes and is void once any config value
> changes. **There is no emergency stop endpoint.** "Halt now" (`POST /v1/cell/brake`) stops the run and
> latches the arm: nothing after the move in flight is sent, and only with `robot.ur.brake_on_halt` on (off as
> shipped) is that move braked under control. It is not an emergency stop; the physical button at the cell
> stays the safety stop.

## The endpoints

Everything is under `/v1`, and every failure comes back as one envelope, `{code, message, detail}`,
with `code` a string your client can branch on. `GET /v1/codes` serves every code the console answers
with. **Moves** says what a route can do to the arm and the hand.

| Route | What it does | Moves |
|---|---|---|
| `GET /v1/health` | the server is up; says nothing about the cell | no |
| `GET /v1/codes` | every typed code: run kinds, stop codes and their classes, event types, refusals, the ready bar's lights and blockers, the jaws question's stages and choices, the command reader's notes | no |
| `GET /v1/preflight` | the cell checklist, each row with its fix | no |
| `GET /v1/config/explain?key=...` | a value, its type, default, the layer that set it, and its YAML comment | no |
| `GET /v1/config/writable` | the keys this server may write, described by the library | no |
| `PATCH /v1/config` | write a group of keys, all or none | no |
| `GET /v1/diagnostics` | SDKs, motion stack, perception stack, controller reachability | no |
| `GET /v1/diagnostics/route?prompt=` | which perception route a prompt would take, from the text alone | no |
| `GET /v1/cell` | state, arm, hand, the halt latch, the planner, the stop record, the countdown, a jaws question waiting, and who holds the cell lock | no |
| `GET /v1/cell/facts` | what this cell is: cameras and looks, the natural closing axis, push, detector, brake, carried part, motion route | no |
| `GET /v1/cell/readiness` | the ready bar: six lights and the blockers; Start only where `ready` | no |
| `POST /v1/cell/build?rehearse=` | assemble the cell; touches no robot. `rehearse=true` uses a desk scene | no |
| `GET /v1/cell/connect-preview` | what connecting will move, and the token that acknowledges it | no |
| `POST /v1/cell/connect` | `{token}`: arm, then gripper; the jaws question in the browser; then the planner start | **yes** |
| `POST /v1/cell/disconnect` | the ordered teardown, gripper then arm; safe to call twice; frees the lock | latches a moving run's arm |
| `GET /v1/cell/status` | live pose, joints and wrench; `?include_controller_state=true` adds modes | no |
| `POST /v1/cell/planner` | start cuRobo, as its own run; about a minute | no |
| `POST /v1/task` | a task: pick, place, return, `once` or `until_empty`; answers `202` with the run | **yes** |
| `POST /v1/task/stop?run_id=` | stop the task after the part in hand; a Home run before its move is sent, a wave before its next swing | no |
| `POST /v1/task/restart` | `{run_id}`: a new run of the stopped plan; first motion the planned move to the return pose | **yes** |
| `POST /v1/pick` | `{prompt, picks}`, and `push_mm` for a `dense_clutter` push: picks only, nothing placed | **yes** |
| `POST /v1/pick/stop?run_id=` | a pick run starts no next attempt; the motion in flight completes | no |
| `GET /v1/runs`, `GET /v1/runs/{run_id}` | recent runs of every kind, and one run | no |
| `WS /v1/events?run_id=&since_seq=` | replay a run's events from a sequence number, then follow it live; `run_id=cell` is the cell's own stream | no |
| `POST /v1/cell/brake` | **halt now**: the run commands nothing more, the arm latches, and brakes a move in flight where enabled | latches; brakes where enabled |
| `POST /v1/cell/acknowledge` | `{cell_clear, jaws_empty}`: a person's word that the cell is clear | no |
| `POST /v1/cell/home` | `{to}`: a planned move to Home or a taught pose; the second way back after a stop | **yes** |
| `POST /v1/cell/wave` | Willy waves back at a greeting: two judged swings of the wrist | **yes** |
| `GET /v1/cell/jaws` | the hand, and the jaws question waiting for its answer | no |
| `POST /v1/cell/jaws/answer` | `{question_id, choice}`: answer it; `open_now` is ONE change of the output | `open_now`: the jaws |
| `POST /v1/cell/jaws/check` | ask where the jaws stand now; blocks until answered | only after `open_now` |
| `GET /v1/poses` | Home, the taught poses with labels and screens, the default place, whether one can be taught now, the file it goes to | no |
| `PUT /v1/poses/default-place` | `{name}`: the pose a task places at when its command names none; `null` for none | no |
| `GET /v1/teach/payload` | the payload the controller compensates for, to confirm before the arm is freed | no |
| `POST /v1/teach` | teach one pose by hand; answers `202` with the run and its token | **frees the arm for a person** |
| `GET /v1/teach/{run_id}?token=` | where the session stands; every poll is the heartbeat | no |
| `POST /v1/teach/{run_id}/capture?token=` | save the pose where the arm stands, once it is still | holds |
| `POST /v1/teach/{run_id}/cancel?token=` | hold the arm at once and save nothing | holds |
| `POST /v1/commands/parse` | `{text, source, language}`: the VLM reads a sentence into a task card | no |
| `GET /v1/commands/status` | whether the command reader is loaded, and why not | no |
| `POST /v1/commands/warmup` | "Laden": load the command reader now | no |
| `GET /v1/camera/live?rig=&max_width=` | a live display frame of one camera, during a pick too | no |
| `GET /v1/camera/live.mjpeg?rig=&max_width=&fps=` | the same frames as a continuous MJPEG stream, up to 30 a second; a frame only when the camera has a new one | no |
| `GET /v1/runs/{run_id}/overlays/{n}` | the n-th grasp overlay a task captured, as a PNG | no |
| `GET /v1/runs/{run_id}/target/overlay` | the bin a task's camera found, drawn over the frame it was seen in | no |
| `GET /v1/camera` | one picture and one sentence saying what it is | no |
| `POST /v1/overlay/enable` | turn on the grasp overlay render; it costs time on every pick | no |
| `WS /v1/overlay` | each new overlay image, with its age | no |
| `POST /v1/voice/transcribe` | a WAV upload to a text proposal for the prompt box; starts nothing | no |
| `POST /v1/voice/talk` | `{pressed}`: press or release the push to talk switch | no |
| `POST /v1/voice/listen` | one push to talk turn on the cell PC's microphone, to a proposal; starts nothing | no |
| `GET /v1/history/kpis` | the KPI roll-up over the logged attempts | no |
| `GET /v1/history/records` | logged attempts, newest first | no |
| `GET /v1/history/runs` | this session's runs, from memory | no |
| `GET /v1/history/runs.csv`, `GET /v1/history/records.csv` | one row per run, and one row per attempt | no |

`tests/test_api_contract.py` holds this table to the app: every route here exists, and every `/v1` route of
the app is here.

## What you can change here

The console starts tasks; it does not set parameters. Speed, acceleration, workspace limits and safety
switches are read-only here and edited in YAML, next to the comment that says why each value is what
it is. The writable keys are an allowlist in [`src/config/edit.py`](../src/config/edit.py): the
payload mass and centre of gravity, the three tool frame keys, each camera rig's serial number and
whether it is enabled, which rig is primary, and the UR and KUKA controller addresses. A `PATCH`
takes them as a group, because the three tool frame keys only validate together, and a refused write
leaves every file byte-identical. **A taught pose is never a `PATCH`**: `robot.named_poses` and
`robot.default_place_pose` are written by the pose door alone, after a person guided the arm there and the
screen cleared it ([Poses and teaching](#poses-and-teaching)).

## What it refuses

One code, one meaning, one status (`REFUSAL_STATUS` in [`codes.py`](codes.py)): 409 fix the cell, 422 fix the
request, 428 read the preview first, 403 fix the config, 404 no such thing, 501 this server cannot, 502 the
controller refused, 400 an empty patch or an empty recording, 415 a recording that is not WAV.

| Code | Status | When | What to do |
|---|---|---|---|
| `bad_request` | 422 | a body or query that does not match the endpoint, a plan the library cannot build, both jaw faces asked of a cell that turns every grasp off them, a sentence that is no sentence | the message names the field |
| `no_such_run` | 404 | a run id the console does not hold (it keeps the newest 200), or a stop with no run | read `GET /v1/runs` |
| `not_built_yet` | 501 | a route of the console's contract this server has not built; no route of 0.2.0 answers it | the server is older than the page |
| `no_robot_configured` | 409 | the tree and profile configure no robot | start with the profile that has the arm |
| `not_acknowledged`, `stale_token` | 428 | connect without a valid preview token | read the preview again, then connect with its token |
| `no_real_gripper` | 403 | the build could not make the configured gripper and put a null one in its place | fix the gripper section; `GET /v1/cell` has the reason |
| `cell_busy` | 409 | another process holds this controller's cell lock | stop the other program; `GET /v1/cell` names it |
| `not_built`, `wrong_state` | 409 | preview before a build, connect twice, rebuild while connected, "the cell is clear" with nothing built | build first, or disconnect first |
| `driver_refused` | 502 | the controller refused the connect (payload, tool frame, network), or nobody answered the jaws question | the message is the driver's own sentence |
| `build_refused` | 422 | a build failed, for example a camera that did not open | the message names the cause |
| `jaws_seam_missing` | 409 | a hand that asks where its jaws stand, built without the browser's question | rebuild the cell; the console never asks at its terminal |
| `not_connected` | 409 | a task, pick, Home, teach, halt or jaws check before the cell is connected | connect first |
| `run_active` | 409 | a second run of any kind, a config write, a build, a connect, a parse or "the cell is clear" while a run owns the cell | wait for the run, or stop it |
| `halted` | 409 | the arm's halt latch is set: every moving route, a teach, a jaws check and Connect | a person confirms the cell is clear (`POST /v1/cell/acknowledge`) |
| `controller_stopped` | 409 | the controller's own fields say it cannot move: a protective or emergency stop, a power-off, or a status that does not say | clear it at the pendant; the console never does |
| `cell_not_cleared` | 409 | a moving route, a teach or a jaws check while a stop record stands and nobody said the cell is clear since | look at the cell, then "Zelle ist frei" |
| `restart_required` | 409 | a new task or pick while a stop record stands | Restart or Home; their first motion is the planned move to the return pose |
| `needs_person` | 409 | a recovery of an earlier pick stopped where the arm stands | "Zelle ist frei"; no run clears it silently |
| `jaws_question_pending` | 409 | a moving route or a teach while a jaws question, its one change or its stroke is in progress | answer the question first |
| `part_still_held` | 409 | a toggle's count says CLOSED, the planner still models a part, a hand measures one, or the stop record believes one in a hand that measures nothing | hold the part and answer "open now", or empty the hand and say "Backen leer" |
| `jaws_not_confirmed` | 409 | a toggle's count nobody can vouch for, or, after a stop, jaws not answered open since | `POST /v1/cell/jaws/check` |
| `route_refused` | 409 | the arm's place and return go through no planner that judges them: a UR on `ik`, a KUKA | run cuRobo; a part picked there could not be set down |
| `carried_part_not_modelled` | 409 | a task on an arm that models no carried part | declare `safety.planning_world.payload.length_mm` |
| `camera_target_unavailable` | 409 | a camera place on a cell with no camera to find it with (the rehearsal cell), or a wrist camera whose arm holds no frames | place at a taught pose |
| `object_required` | 422 | a task's object is empty on a cell whose detector grounds a phrase | name the part, or tick "everything the camera sees" (`options.pick_anything`) |
| `prompt_not_routable`, `target_not_routable` | 422 | the object, or the camera place's phrase, needs the VLM route and this cell has none | configure the VLM, or set `on_unavailable: degrade` |
| `push_distance_refused` | 422 | a `push_mm` above the cell's `recovery.fixture.max_nudge_mm` or 50 mm, or under 10 mm; no run starts | ask within the ceiling; the message says it |
| `unknown_pose` | 422 | a place, return or Home pose nobody taught; `detail.which` says which | teach it, or name a taught one |
| `no_place_declared` | 422 | a pose place with no pose and no `robot.default_place_pose` | choose a default place, or name the pose |
| `closing_axis_refused` | 422 | an axis that names no direction, or one this cell's motion turns every grasp off | `x`, `-x`, `y`, `-y`, `radial`, `-radial`, `tangential`, `-tangential` |
| `not_a_task`, `not_a_pick` | 409 | `POST /v1/task/stop` on a pick, teach or planner run; `POST /v1/pick/stop` on anything but a pick run | the other stop, or halt now |
| `not_restartable` | 409 | a Restart of a run that is not the stop record's, or a run that keeps no plan | Restart the run the stop card names |
| `jaws_not_open` | 409 | a jaws check that did not end open, or "Backen leer" on a toggle not counted open or a hand that measures a part | look at the jaws, answer again |
| `no_question`, `question_changed`, `choice_not_offered` | 404, 409, 422 | an answer with no question waiting, to a stale question, or a choice it did not offer | read `GET /v1/cell/jaws` again |
| `planner_not_used` | 409 | "start the planner" on an arm with no cuRobo planner | nothing to start |
| `no_overlay` | 404 | an overlay a run did not keep | the run's events name the ones it kept |
| `no_layer` | 422 | a taught pose or a default place under a chain with no layer of the cell's own, or a last layer git does not keep out | end the chain in a git-ignored cell layer, such as `robot/robot.cell.yaml` |
| `no_hand_guiding`, `screen_unavailable` | 409 | a teach on an arm that cannot be guided by hand, or that screens no pose (no planner, no wrist housing) | teach on the cell's UR, with its housing declared |
| `planner_not_ready` | 409 | a teach while cuRobo is not ready: every pose is screened at once | wait for the planner light |
| `part_in_hand` | 409 | a teach while a part may be in the jaws | empty the hand first |
| `payload_changed` | 409 | the controller's payload is not the one the person confirmed | read the payload again and confirm it |
| `name_taken`, `invalid_name`, `invalid_label` | 409, 422, 422 | a pose name in use, or a name or label the pose rule refuses | another name, or `replace: true` |
| `wrong_token`, `not_free` | 403, 409 | a teach poll without its session's token; Save while the arm is not free | the token from `POST /v1/teach` |
| `vlm_not_loaded` | 409 | a command on a cell whose detector is not the VLM, before "Laden"; or another checkpoint is loaded | "Laden" (`POST /v1/commands/warmup`), or rebuild the cell |
| `vlm_unavailable`, `vlm_model_missing` | 501 | the VLM cannot answer here, or its weights are not on this machine | fill in the card by hand; `python scripts/model_weights/fetch.py vlm-4b` |
| `not_writable`, `unknown_key` | 403, 404 | a key outside the allowlist, a taught pose's key, or no such key | edit it in YAML; teach a pose in Setup |
| `invalid_value` | 422 | the loader rejected the written tree; every file is restored | the message is the loader's |
| `empty_patch` | 400 | a `PATCH` with no values | send at least one key |
| `cell_connected` | 409 | the primary rig or a controller address, while connected | disconnect first |
| `no_target` | 409 | no file under the config root backs the key | point the console at the config tree with `--data` |
| `empty_audio` | 400 | a transcription upload with no bytes | record again |
| `audio_format_unsupported` | 415 | an upload that is not WAV | upload 16-bit PCM WAV, which the console records |
| `audio_undecodable`, `audio_too_long` | 422 | a truncated or unusable WAV; longer than Whisper's window | record again, shorter |
| `transcription_failed` | 422 | the speech engine raised on a recording it could read | the message is the engine's; type the command |
| `speech_unavailable`, `speech_model_missing` | 501 | the speech packages or weights are not on this machine | the message names what installs them; or type |
| `talk_not_pressed`, `microphone_ended` | 409 | the switch was not pressed within `timeout_s`, or the microphone stopped | listen again |
| `nothing_recorded` | 422 | the switch came up before any audio | hold the switch while speaking |
| `microphone_unavailable`, `listen_busy` | 501, 409 | no microphone on this host; a listen already running | use the browser's microphone; wait |
| `listen_failed` | 422 | a push to talk turn that failed for another reason | the message names it |

A request whose body or query does not match the endpoint answers `422 bad_request`, and an unknown
path answers `404 http_404`, both in the same envelope. `http_<status>` is the one code the catalog does not
list: it is the server's, not the cell's.

## How a cell comes up

| State | Call | Next state |
|---|---|---|
| `disconnected` (nothing built) | `POST /v1/cell/build` | `built` |
| `built` | `GET /v1/cell/connect-preview` | `built`, holding a token |
| `built`, holding a token | `POST /v1/cell/connect` | `connected`: arm, then gripper |
| `connected` | `POST /v1/cell/disconnect` | `built`: gripper, then arm |

A connect is all or nothing: if the gripper refuses, the arm is disconnected again and the state
stays `built`. A toggle hand asks where its jaws stand **during** the connect, in the browser, and a question
nobody answers rolls the connect back. Once connected, a cuRobo arm whose planner is off starts it as a run of
kind `planner` before Connect answers: it moves nothing, takes about a minute, and holds the run lock so no task
races a half-started planner. The dummy and an `ik` arm start nothing. A built arm whose halt latch is set
refuses Connect (`409 halted`) until a person says the cell is clear, and building again ends no halt: the
console latches the arm of every later build with it.

The preview's warnings come from the gripper that was actually built, so a cell with a null gripper shows
none. One process owns a cell: `CellLock` in
[`src/robot/execution/cell_lock.py`](../src/robot/execution/cell_lock.py) is keyed on the controller
address and held by an open file handle, so the operating system frees it when the process dies. The
console, `Robot`, `Cell` and the command line runner all take it. It cannot see a second machine; the
reachability check in `GET /v1/diagnostics` covers that.

**The ready bar** is `GET /v1/cell/readiness`: it always answers and moves nothing, and Start is on only where
`ready` is true, every light that blocks is ok and no blocker stands.

| Light | Ok when | Its codes |
|---|---|---|
| `robot` | connected, not halted, and the controller's own fields say it can move, read from the receive stream | `connected`, `not_built`, `not_connected`, `halted`, `controller_stopped`, `controller_unreadable` |
| `cameras` | every camera of the cell has a frame younger than 2 s, or the cell runs the rehearsal scene | `live`, `rehearsal`, `no_frame`, `none` |
| `planner` | cuRobo is ready, or the arm moves nothing real (the dummy) | `ready`, `starting`, `off`, `failed`, `unplanned`, `route_refused` |
| `gripper` | a toggle's count is open and no question waits; another hand is connected | `open_confirmed`, `jaws_unknown`, `jaws_closed`, `question_pending`, `connected`, `none` |
| `carried_part` | the arm models a carried part, or carries nothing real | `modelled`, `not_modelled`, `not_applicable` |
| `commands` | informational only, it never blocks | `ready`, `idle`, `loading`, `missing`, `failed`, `not_configured` |

The blockers are `run_active`, `cell_not_cleared`, `restart_required`, `needs_person`, `part_still_held` and
`jaws_question_pending`, each the refusal of the same name said before a button is pressed. Only a controller
whose own fields say it can move lets anything go: a status that does not say refuses as surely as a stopped one.

**The cell's facts** are `GET /v1/cell/facts`, read once after a build and once after a connect: the cameras with
their mounting and the looks, the natural closing axis, the push (`can_push`, `default_mm`, `ceiling_mm`, the
50 mm hard cap where the cell declares no fixture, `critical_parts` and `blocker_into_the_place`, the cell's switches), the detector
(`backend`, the router, the VLM's id, and its `precision`: `fp32 weights, fp16 autocast` where
`optim.torch_dtype` is unset, the configured dtype otherwise, `auto (the checkpoint's own)` for the VLM, both
for a routed stack), the brake (`latches`, `brakes_in_motion`), the carried part (`modelled`, why not,
`length_mm`) and the motion route (`planned`, `unplanned` or `refused`).

**Disconnect, and the server's shutdown, take the cell down in one order:** a jaws question waiting is
cancelled, before anything takes the session lock its connect holds; a teach is held at once and joined (at
most 10 s; it ends `disconnected`); a planner start is joined; the run is abandoned; the arm of an abandoned
moving run is **latched**, and braked where it brakes a move in flight; then the cell comes down. Where a move
was in flight, or the arm braked one, the latch stays: `CellOut.halted.reason` says the cell was disconnected
while a run was moving the arm, and the next Connect waits for "the cell is clear", which is allowed while
`built`. With nothing in flight on an arm that lets a move run to its end, the latch is given back once the
cell is down.

## Tasks and their events

**A command is a task.** `POST /v1/task` takes `TaskIn` and answers `202` with the run (kind `task`) at once;
everything after that arrives on the run's event stream. Start on the cockpit's card is the confirmation:
nothing else starts a task, and a parse never does.

| `TaskIn` field | What it says |
|---|---|
| `object` | the English phrase the detector grounds, at most 80 characters; `""` is anything, which a real cell takes only with `options.pick_anything` |
| `object_said` | the operator's own words for it, for display |
| `place` | `{"kind": "pose", "pose": <name or null>}`, where `null` is `robot.default_place_pose`; or `{"kind": "camera", "phrase": "blue bin", "said": ...}`, a target the camera finds |
| `return_to` | `home`, or a taught pose's name |
| `scope` | `once`, or `until_empty` |
| `options` | `multi_view` (on: the configured looks; off: the first look, no generated view), `both_faces`, `closing_axis`, `push_mm`, `critical_parts` (true: nothing is pushed, and a blocker is cleared instead; unset: the cell's `recovery.critical_parts`), `blocker_into_the_place` (true: a task that names no object takes a blocker as the part its pick takes and sets it down where the parts go; false: it is set aside on the support; unset: the cell's `recovery.blocker_into_the_place`, on), `record_views`, `rim_air_mm` (10 to 50, the cell's 20 when unset; a camera place only), `pick_anything`, `overlay` |
| `command` | `{text, source, language, parsed, edited}`: the sentence it came from, for the record only |

**Refused before anything moves, in this order.** The cell's state first: `not_connected`, `run_active`,
`halted`, `controller_stopped`, `cell_not_cleared`, `restart_required`, `needs_person`,
`jaws_question_pending`, `part_still_held`, `jaws_not_confirmed`, `route_refused`,
`carried_part_not_modelled`. Then the request's own: `camera_target_unavailable`, `object_required`,
`prompt_not_routable`, `target_not_routable`, `push_distance_refused`, `unknown_pose` (the place's, then the
return's), `no_place_declared`, `closing_axis_refused`, `bad_request`. The run registry reads the stop record
once more under its run lock as it starts the run, so a stop that landed between the gates and the start
still refuses it.

**The plan is resolved and echoed.** `RunOut.plan` (`TaskPlanOut`) carries the place pose's label and joints,
the return pose's, the resolved push distance and rim air, `first_motion` (`look` for a new task, `return`
for a Restart) and `countdown`. Its options say whether the operator asked for the push distance
(`push_asked`: an asked distance is taken as asked, the cell's may go longer, to 40 and 50 mm, where 30
frees no direction) and the parts' switch as asked (`critical_parts`, null for the cell's). A Restart runs
exactly this plan again.

**What a task does.** It screens every taught pose it will use before any motion (`task.pose_screened`; an
`ERROR` ends `pose_refused`), finds a camera place's bin first (`task.survey_started`, `task.target_found`),
then picks, places and returns, part after part:

- **A taught place pose says where the part's bottom is let go.** The tool goes there raised by the part's
  hang, the grasp height over the declared support, so every error goes toward more air.
- **A camera place goes into the bin over its rim**: the drop stands at rim + hang + air (20 mm unless
  `rim_air_mm` says 10 to 50); a grasp within 5 degrees of vertical is turned about the vertical along
  `robot.natural_closing_axis`. The bin is checked again before every drop, from the look it was found at;
  moved more than min(100 mm, half its diagonal), or another footprint or rim, it is lost: the part goes back
  where it was gripped, the arm returns, and the task asks.
- **Its own drop area stays out of its picks**: a camera place's bin for both scopes, and 150 mm about a pose
  place's drop for `until_empty`, where the arm's own kinematics say where the pose puts the tool (the dummy's
  do not). A look that sees only those parts is an empty one.
- **Benign ends return, problems stay.** `once` ends when the part is placed and the arm is back.
  `until_empty` ends after 2 empty looks in a row. 3 failed picks in a row end it where the arm stands, and
  100 parts end it `part_limit`.
- **A toggle hand changes DO0 exactly twice per part**: the close at the part, the release at the drop.
- **Nobody is asked on a run's thread.** A hand that would ask is a refusal, and the run ends.

| Class | Stop codes | Run state | What the cockpit shows |
|---|---|---|---|
| done | `finished`, `nothing_left`, `part_limit`, `taught`, `planner_ready` | finished | "Fertig: 7 Teile abgelegt.", then the next-instruction question; the arm is at its return pose |
| operator | `stopped_after_part`, `cancelled` | cancelled | a neutral line, then the next-instruction question |
| ask | `target_not_found`, `target_lost`, `target_unreachable`, `part_does_not_fit`, `pose_refused`, `teach_refused` | finished | an ask card; every option that moves is a new Start |
| teach | `heartbeat_lost`, `teach_time_limit`, `teach_not_saved` | failed | the pose was not saved, and why |
| planner | `planner_failed` | failed | the planner light and "Planer starten" |
| problem | `halted`, `controller_stopped`, `hand_needs_person`, `recovery_needs_person`, `part_still_held`, `return_failed`, `failed_in_a_row`, `detector_failed`, `cell_fault`, `disconnected`, `software_error` | failed | the stop card; a pick, task or Home run leaves the stop record |

`RunOut` carries, beside a pick run's counts: `kind`, `stop_code` and `stop_class` (empty while it runs),
`plan`, `parts_placed`, `holding` (the program believed a part in the jaws at the end; every gate reads the live
hand first, and for a hand that can say nothing itself the stop record keeps the belief, as `part_still_held`,
until "Backen leer"), `restart_of`, `halt_requested`, `step` (the last timeline step) and `refusal`.
**`refusal`** is `{code, status}` where a run ended `cancelled` before anything moved on what its route
would have refused: the library's own backstop inside a task, or `jaws_question_pending` where a moving run
gave way to a jaws check that began as it started. `GET /v1/history/runs.csv` adds the columns `kind`,
`stop_code` and `parts_placed` after the ones a pick run always had.

**The events of a task**, beside every run's `run_started`, `run_finished` and `run_error`:

| Event | Says |
|---|---|
| `task.pose_screened` | a taught pose the task will use, and its verdict, before any motion |
| `task.survey_started`, `task.target_found`, `task.target_missing` | the camera place's bin, found or not, before the first pick |
| `task.part_started` | part n (of 1, or of no bound), its pick |
| `pick.*`, `pick_result` | the pick loop's stages, and each pick's result with the overlay URL it captured |
| `task.nothing_found` | a look that saw nothing to pick, or only parts the task keeps out |
| `task.carry_started`, `task.target_checked`, `task.target_lost` | the carry to the bin's look, the bin checked again, or lost |
| `task.drop_planned`, `task.place_started`, `task.placed`, `task.place_failed`, `task.put_back` | the drop and how it went |
| `task.return_started`, `task.returned`, `task.return_failed` | the way back, with the motion's own note where it took a straight leg |
| `task.part_finished` | the part, whether it was placed, and its seconds |

A task's `pick_result` adds `part` and `pick`, the hold and the stops, the grasp pose and the part's middle,
`both_faces`, `pushes`, `fused_views`, `found_nothing`, `only_excluded`, `detector_failed`,
`hand_eye_warn_mm`, `views_file` and `overlay`, each only where the report says it.

**Stop after this part** is `POST /v1/task/stop?run_id=`. Not a kill and not an emergency stop: the pick in
flight finishes its attempts and pushes, a held part is still placed, the arm returns, and the task ends
`stopped_after_part` (`run_stop_requested` with `scope: after_part`). During the countdown it ends
`cancelled` with nothing moved. A Home run is stopped too, before its one move is sent (`scope:
before_motion`); a move already under way runs to its end. It latches nothing, so no "the cell is clear" is
owed.

**The hands-off countdown.** After a teach, after "Jetzt öffnen" and after "Backen leer", a person's hands
were at the arm, so the next moving run (task, Restart, Home, pick) counts down 3 s before its first motion:
three `run_countdown` events a second apart, `{seconds_left, because: teach | jaws_opened}`, with stop, halt
and a Disconnect read every 100 ms. A run stopped there ends `cancelled` with nothing moved. Only a run that
moved the arm ends the countdown (a pick that gripped, a part placed, an arrival); `CellOut.countdown_due`
says it is due, and Start, Restart and Home say so on their button.

## Halt now (brake)

`POST /v1/cell/brake` is the cockpit's "Sofort anhalten": one click, no dialog, no session lock, so a connect
waiting on a jaws question never holds it up. It is refused only `409 not_connected`, and it answers
`BrakeOut {run_halted, latched, braking, in_motion, run_id, message}`.

- **The run commands nothing more.** The task and Home read the halt after every motion, a pick's cancel check
  between its attempts, a countdown every 100 ms (`run_halt_requested {reason, in_motion, requested_at,
  braking}`; `cell.halted` on the cell stream where no run was active).
- **The arm latches** (the UR and the dummy): every next motion and every output switch is refused before
  anything is sent, the jaws included, until a person says the cell is clear. The latch outlives Disconnect
  and Connect, a rebuild and a restart of the server: the console carries a halt nobody has said the cell is
  clear of to the arm of every later build, and keeps it in its stop file. Only `POST /v1/cell/acknowledge`
  ends it.
- **The move in flight** is the arm's. With `robot.ur.brake_on_halt: true` the UR brakes it under control
  (`stopJ` or `stopL` at max(2.0, the move's own acceleration), from the thread that sent it) and `braking`
  is true. **Off, as shipped**, every move is sent exactly as it always was: the move in flight runs to its
  end, and nothing after it is sent.
- **An arm without a latch** (sim, KUKA) answers `latched: false`: the run commands nothing more, and the
  current motion ends first.

A halt reads `halted` everywhere, in every refusal, light and stop code, and never as a stopped controller,
whose remedy is the pendant. `CellOut.halted` says what became of the move in flight: `brake` is `none`,
`pending`, `braked` (with `brake_s`), `ran_out` or **`unconfirmed`**: nobody saw the arm stand still, so if it
still moves, press the emergency stop. **It is not an emergency stop**: it travels browser, localhost,
threadpool and an 8 ms poll, and the red button does not.

## After a stop: "Zelle ist frei", then Restart or Home

**The stop record.** A pick, task or Home run that ends on a problem code leaves the arm where it stopped and
writes `CellOut.recovery {run_id, kind, stop_code, at, holding, cleared_at}` (`cell.recovery` on the cell
stream). It lives on the console, so it survives Disconnect, a rebuild and a page reload. It is written before
the run lets go of the lock, so nothing new starts in between. A halt with no run, or a teach or planner run
that stopped, writes none: nothing was stopped mid-motion, and the arm's latch alone gates the next motion.

**It outlives a restart of the server.** `python -m api` keeps the stop in **`logs/console/stop.<profile
chain>.json`**, one file per profile chain (`stop.json` with no profile), so a desk rehearsal's stop never gates
the cell's own console: the record, the stopped run it names, and a halt nobody has said the cell is clear of.
The file is replaced whole on every change and removed once nothing stands. The next start reads it back, says
so in its banner, and:

- the record stands again **uncleared**, whatever it said before: a restart is no "the cell is clear";
- the stopped run is known again (`GET /v1/runs/{id}` answers it, Restart replays its plan; its events do not
  come back);
- a kept halt latches the arm of the next build, so Connect answers `409 halted` until "the cell is clear";
- a file that does not read gates as a stop of unknown origin (run `run-unknown`, `software_error`): "the cell
  is clear", then Home.

A server started on a tree named with `--data` keeps no file unless `WILLY_CONSOLE_STOP_FILE` names one, so a
scratch tree's stop never gates the cell; `--reload` hands the file to its worker through the same variable,
and a console a program builds itself keeps nothing on disk. On the way out the server waits up to 10 s for a
run it abandoned, so that run's record is kept.

**While it stands, nothing new starts.** Until a person says the cell is clear, every moving route, a teach and
a jaws check answer `409 cell_not_cleared`. After that, a new task or pick answers `409 restart_required`: the
two ways back are **Restart** and **Home**, and their first motion is the planned move to the return pose.
Arriving there ends the record (`cell.recovery_ended {run_id, by: restart | home}`).

**"Zelle ist frei"** is `POST /v1/cell/acknowledge {cell_clear: true, jaws_empty: false}`: a person's word,
allowed while `built` or `connected`. It clears the arm's halt latch (and the halt the console carries to later
builds), the service's latch of a recovery that needs a person, and stamps the record cleared
(`cell.acknowledged`). It moves nothing, and it **never clears a protective stop**: a stopped controller refuses
it (`409 controller_stopped`), because that is cleared at the pendant, where the arm is visible. Refused too:
`not_built`, `run_active`. **`jaws_empty: true`** is the person's word that the hand holds nothing ("Backen
leer"): taken for a hand that cannot say it itself, refused `409 jaws_not_open` for a toggle whose count is not
open (its way is the jaws question) and for a hand that measures a part. It tells the planner the hand is empty,
and the next motion counts down 3 s.

**Restart** is `POST /v1/task/restart {run_id}`: a new run of the stopped run's plan (`restart_of`), whose first
motion is the planned move to the return pose; a stopped Home run restarts as Home to the same pose. Only the
record's own run restarts (`409 not_restartable` otherwise), and every refusal of a new task holds but
`restart_required`, read against the cell as it is now: a toggle must have been answered open since the stop
(`jaws_not_confirmed`), and no part may be held (`part_still_held`).

**Home** is `POST /v1/cell/home {to: "home" | <pose name>}`, `202` with a run of kind `home`: a planned move,
after the countdown where one is due. Refused `not_connected`, `run_active`, `halted`, `controller_stopped`,
`cell_not_cleared`, `needs_person`, `jaws_question_pending`, `part_still_held` (nothing outside a task carries
a part), `jaws_not_confirmed`, `route_refused` and `unknown_pose`. Its events are `home.started`,
`home.arrived` and `home.refused`, and it ends `finished`, `cancelled`, `part_still_held`, `return_failed`,
`halted`, `controller_stopped` or `disconnected`.

**The wave** is `POST /v1/cell/wave`, `202` with a run of kind `wave`: the second wrist joint swings 15 degrees
either way, twice, and back to where it stood, each swing a straight joint line the exact guard judges against the
camera world before it is sent, after the countdown where one is due. The console starts it when someone greets
Willy in the chat (`CommandOut.greeting`, see Commands). Refused as a new task is, bar a carried part:
`not_connected`, `run_active`, `halted`, `controller_stopped`, `cell_not_cleared`, `restart_required` (a wave is
no way back after a stop), `needs_person`, `jaws_question_pending`, `part_still_held`, `jaws_not_confirmed` and
`route_refused`. Its events are `wave.started`, `wave.done` and `wave.refused`. `POST /v1/task/stop` ends it before
its next swing (`cancelled`, the arm where the last swing left it). A swing the guard refuses before the first one
ran ends it `cancelled` with nothing moved; after one ran, as a refused Home ends (`return_failed`, `halted`,
`controller_stopped`), so a person decides. It also ends `finished`, `part_still_held` or `disconnected`.

The stop card enables Restart and Home only when every gate is green:

| Gate | Read from |
|---|---|
| the controller can move, the halt aside | the ready bar's `robot` light |
| "Zelle ist frei" since the stop | `recovery.cleared_at > recovery.at` |
| a toggle's jaws answered open since the stop | `jaws_confirmed_at > recovery.at`, and `hand.jaws` is `open` |
| no part held, now | no `part_still_held` among the ready bar's blockers: a toggle's count, the planner's model, a measuring hand, and the stop record's belief for a hand that measures nothing, until "Backen leer" |
| no run, no jaws question, the cell connected | `CellOut` |

**The move home is planned against what the camera sees now.** The stopped pick's frames end with it, and a
wrist camera never sees the fingers, so a hand that stands in a bin or among parts is jogged clear at the
pendant first. **After a protective stop on a CB3** the control script has ended: clear the stop at the
pendant, Disconnect and Connect (the first script upload can time out once; connect again), say the cell is
clear, answer the jaws question, then Restart.

## The jaws question in the browser

A toggle hand (the owner's Hand-E on tool DO0, no sensor) counts its own changes from where a person says the
jaws stand. Connect asks, and so does a check (Restart, Setup, the ready bar). **The console asks only in the
browser**: a question at the server's terminal cannot be cancelled, and nobody watching the console sees it.
CLI programs and the examples still ask at their own terminal.

- **No default.** A question waits at most **120 s** for `POST /v1/cell/jaws/answer`; unanswered, cancelled or
  cut by a Disconnect, it is refused, **never "open"**, and a connect is rolled back (`502 driver_refused`,
  saying nobody answered).
- **`GET /v1/cell/jaws`** shows the hand (`HandOut`: kind, driver, output, jaws `open` / `closed` / `unknown`
  / `not_counted`, why unknown) and the waiting question with exactly its choices. `CellOut.jaws_question` true
  with no question waiting means the hand is acting on an answer, or a check is reading the latch and the
  controller: keep every moving button off.
- **The stages.** `where` takes `open` or `closed`; after `closed`, `open_now` takes `open_now` or `abort`.
  The answer returns once the hand moved on (at most 5 s), with the next question or none. Refused `404
  no_question`, `409 question_changed` (a stale id), `422 choice_not_offered`; the question goes on waiting.
- **A count that says closed is never answered "open".** Where the program itself knows the jaws are closed
  (its count says CLOSED, no change failed or went unread since, and DO0 still reads the level that count
  left), a check asks `open_now` first and only: "open" is no answer there, and an abort keeps the count
  closed. Where the count cannot vouch for itself (DO0 switched at the pendant, a change that failed) or says
  open, it asks `where` with both answers.
- **`open_now` is ONE change of the output.** It opens the jaws where the arm stands and drops what they
  hold, so the person holds the part first. It goes out only once the gate let it: no run holding the cell,
  the arm not halted, the controller's own fields saying it can move, and no stop record nobody has said the
  cell is clear of (at Connect too). A stopped controller gets no "open now", with nothing sent.
- **An answer that leaves the jaws open** stamps `jaws_confirmed_at`, tells the planner the hand is empty
  (`arm.detach_payload()`), and after `open_now` makes the next moving run count down.
- **`POST /v1/cell/jaws/check`** always asks, also where the count says open, and blocks until the question
  ended: give it a client timeout of at least 250 s. A hand that is not a toggle answers at once. Refused, in
  order: `not_connected`, `run_active`, `halted`, `controller_stopped`, `cell_not_cleared` (after a stop the
  jaws wait for "the cell is clear", as every motion does; nobody is asked), `jaws_question_pending`,
  `jaws_seam_missing`, and `jaws_not_open` with the hand's own sentence.
- **A check and a run never go on together.** A question is pending from a check's start until its one change
  and stroke are done; every moving route and a teach refuse meanwhile (`jaws_question_pending`), and a moving
  run that started as a check began ends before its first motion (`cancelled`, `refusal:
  jaws_question_pending`). Disconnect ends a waiting question before it takes the session lock.

Events on the cell stream: `cell.jaws_question {question_id, stage, at, where, reason, choices, attempt, of,
why_again, expires_at}`, `cell.jaws_answered {question_id, choice}`, and `cell.jaws_ended {question_id,
outcome: open | refused | no_answer | cancelled, refusal, detached}`.

## Poses and teaching

**`GET /v1/poses`** answers Home (read-only, from `robot.home_joint_positions`), every taught pose (name, label,
joints in degrees, screen, note, when it was taught), the default place, whether a pose can be taught now
(`teachable`, `why_not`, `why_not_code`) and the file a taught pose is written to, said before anyone frees the
arm. **`PUT /v1/poses/default-place {name}`** chooses the place a task uses when its command names none
(`null` for none); refused `run_active`, `unknown_pose`, `no_layer`, `invalid_value`.

**Teach one pose by hand.** `GET /v1/teach/payload` shows what the controller compensates for (a wrong payload
makes a freed arm sink or rise in a person's hands). `POST /v1/teach {name, label, role: place | other,
replace, make_default_place, payload_seen}` answers `202 TeachStartOut {run, token}` and frees the arm. Refused
before anything is freed, in order: `not_connected`, `run_active`, `halted`, `controller_stopped`,
`cell_not_cleared`, `jaws_question_pending`, `part_in_hand`, `jaws_not_confirmed`, `invalid_name`,
`no_hand_guiding`, `screen_unavailable`, `planner_not_ready`, `invalid_label`, `name_taken`, `no_layer` and
`payload_changed`; a NaN or infinite payload is `422 bad_request`.

- **Every poll is the heartbeat.** `GET /v1/teach/{run_id}?token=` says where the session stands (`freeing`,
  `free`, `holding_when_still`, `holding`, `screening`, `saved`, `refused`, `not_saved`, `ended`), the joints,
  the TCP, whether it is outside the workspace, and the time left. Without a poll for **3 s**, and at the
  **5-minute** limit (`teach.time_warning` 30 s before), the arm is held **only once it stands still**, never
  while it moves in a person's hands; the run ends `heartbeat_lost` or `teach_time_limit`, nothing saved.
- **Save** is `POST .../capture`: captured once the arm is still, held, screened at once by the exact guard and
  the planner, and written where the verdict allows; `409 not_free` outside `free`. **Hold** and **Cancel** are
  both `POST .../cancel`, which holds at once and is idempotent; the page also sends it with `sendBeacon` when
  it closes. Halt, a stop request and a Disconnect hold at once too. Shared refusals: `403 wrong_token`, `404
  no_such_run`, `409 not_free`.
- **The verdict decides what is written.** `clear` and `band` are written into the cell's own layer, through the
  pose door alone; an `ERROR` is never written (`teach.refused` with the nearest clear joints, `teach_refused`),
  and neither is a pose the screen could not judge (`teach.not_saved`).
- **A place pose says where the part's bottom is let go**: the fingertips go there, and a task raises every
  part by its hang. The dialog says so.

Teach events: `teach.free`, `teach.say`, `teach.outside`, `teach.inside`, `teach.time_warning`,
`teach.holding_when_still`, `teach.holding`, `teach.screening`, `teach.saved`, `teach.refused`,
`teach.not_saved`. A teach ends `taught`, `teach_refused`, `teach_not_saved`, `heartbeat_lost`,
`teach_time_limit`, `cancelled` or `disconnected`. A jaws question that comes up as a teach starts ends it
`teach_not_saved` with nothing freed. After a teach the next moving run counts down 3 s.

## Commands

`POST /v1/commands/parse {text, source: typed | spoken, language}` hands the whole sentence, German or English,
to the VLM, which answers `CommandOut`: `understood`, `intent` (`task`, `stop`, `none`), the `object` and the
`place` as English phrases with the operator's own words and the route the detector would take, `place_pose`
and `return_to` (taught poses by **name**: a spoken label comes back as its name), `scope`, `count`, `notes`,
the model's id, latency and attempts, and its raw answer. **Reading never starts anything**: no run, no prompt
set, no cell touched. A sentence read as "stop" only points at the stop buttons.

**A greeting** ("Hallo Willy", "Hi Willy, wie geht's?", "Tschüss Willy", "Willy, wink mal!") comes back with
`greeting` set to how the console answers it, the app config's `runtime.greeting.wave` (`config/app/runtime.yaml`):
`direct`, the default (the owner, 2026-10-06), and the console starts the wave at once, `POST /v1/cell/wave`, with
no click; `confirm`, and a dialog like Home's asks first; `off`, and Willy only greets back. `null` for every
other sentence. The model reads the sentence as ever; a greeting is a sentence it read as no command that opens with
a greeting or a farewell or asks to wave, or one that is nothing but a greeting, whatever the model made of it, bar a
stop (`src.models.vlm.command.greets`). A command with a greeting in front stays a task.

Refused: `409 run_active` first (during any run, a teach included), `409 vlm_not_loaded` (a cell whose detector
is not the VLM, before "Laden"; or another checkpoint loaded: rebuild the cell), `501 vlm_unavailable` (also a
model that fails while answering, and a reader refusal the catalog does not know), `501 vlm_model_missing`, and
`422 bad_request`. `GET /v1/commands/status`
says `ready`, `idle`, `loading`, `missing`, `failed` or `not_configured`, with the cause and whether the copy is
shared with detection; `POST /v1/commands/warmup` is "Laden", refused during a run.

**One VLM copy per process** reads commands and, on a cell whose detector is the VLM, detects too. A VLM cell
loads it at its first command; any other cell waits for "Laden". Building a cell that does not detect with it
lets the copy go, unless a person loaded it, and switching checkpoints takes a rebuild. Measured on the
development box's RTX 5080 with the 4B checkpoint: a load of about 6.3 to 7.3 s, 8.93 GB after it, about
2.2 s per command on a free card ([`src/models/vlm/`](../src/models/vlm/README.md)).

## The live image

`GET /v1/camera/live?rig=&max_width=960` answers `LiveFrameOut`: one display frame of the rig asked for (the
primary where omitted), downscaled to `max_width` (32 to 4096) and JPEG-encoded at
`runtime.image_encoding.frame_quality`, with `captured_at`, `age_s`, and the cell's `rigs` (each `wrist`,
`fixed` or `unknown`, and which is primary).

- **Display, never a measurement.** It reads through `Camera.peek`, which never waits and never takes a frame
  a pick relies on, so the image runs during a pick. While a pick grabs, it answers `reason: measuring` and the
  browser keeps its last frame.
- **No picture is an answer, not an error**: `not_built`, `no_camera`, `no_rig`, `measuring`, `encode_failed`.
  The rehearsal cell's drawing is `source: synthetic`, never `camera`.
- **The ready bar's camera light** wants a frame younger than 2 s from every rig, and peeks a stale rig itself.

**Overlays are their own images.** A task keeps the grasp overlay rendered when the grasp was decided, before
the arm moved: at every `pick.executing` and after each pick, and only where it is a new image.
`GET /v1/runs/{run_id}/overlays/{n}` serves them as PNG, and `GET /v1/runs/{run_id}/target/overlay` the bin the
camera found; refused `404 no_such_run`, `404 no_overlay`. The store holds 64 MB and lets the oldest go. The
browser pins an overlay over the stage for 5 s and never draws it over the moving live picture.

## Picks and their events

- `POST /v1/pick` returns a run id at once. A run continues when the browser closes, so the pick in flight
  runs to its end. **A pick run sets nothing down**: what its last pick lifted stays in the jaws
  (`RunOut.holding`), and on a toggle hand the next pick does not start. Picking and placing is a task.
- It refuses, after `not_connected`: `run_active`, `halted`, `cell_not_cleared`, `restart_required`,
  `needs_person` and `jaws_question_pending`. A run never clears a person's latch: one started on a service
  whose earlier pick stopped where the arm stands ends `recovery_needs_person`, with nothing commanded and the
  latch kept.
- The prompt is what the detector grounds for this run. An empty prompt keeps the cell's own phrase.
- Every pick of a run looks from the cell profile's looks (`robot.look_joint_positions_deg`), else, on a
  wrist camera, from home, as a `PickRun` does. A wrist camera fuses its looks until the grasp is safe.
  `both_faces` stays off here: the console runs the fast rule, and the switch belongs to a program
  (`PickRun`, `service.pick`) or a task's Advanced options.
- `push_mm` sets how far a `dense_clutter` push of a wrist camera's pick moves the part in this run:
  taken as asked up to the cell's `recovery.fixture.max_nudge_mm` (50 mm on a cell that declares no
  fixture, which the push does not need), refused above it or under 10 mm
  (`422 push_distance_refused`, no run started), never shortened. Without it a push is the cell's
  `push_distance_mm`, 30 mm unless the cell says otherwise, and longer, 40 then 50 mm up to the ceiling,
  where that frees no direction. The push may rearrange the scene, and runs only where the cell's parts
  are not critical (`recovery.critical_parts`); critical parts are never pushed, and a blocker is cleared
  instead. A run is one campaign: its push budgets and the parts `next_target` skips are its own, and
  `run_started` carries `push_mm` where the run asked for one.
- Nobody is asked anything during a run, where nobody watching the console would see the question. A toggle
  hand whose jaws the program believes closed, or cannot place, ends the run before its next pick, and a
  question inside a pick is a refusal: the hand needs a person. So is a hand nobody can vouch for, at a push
  or before a re-pick drives the looks: a toggle's count, or a width gripper not connected or unreadable.
  The jaws question in the browser brings a toggle's count back to open.
- A run stops, `failed`, on a pick whose controller cannot move, whose hand needs a person, or whose push
  stopped once something may have moved: the arm stays where it stopped, the stop record stands, and a person
  decides. A pick a halt cut short ends `halted`, whatever else its report says. A push's attempt says what
  the push did, in its own words.
- Each `pick_result` event carries `looks` (the looks perceived from; a fixed camera's, the looks it
  moved to) and `looks_fused` (a wrist camera's, the looks the grasp's cloud was fused from), and, only
  where the pick says them, `jaw_faces_seen`, `generated_view_deg`, `hand_eye_gap_mm`, `refused_look` and the
  fields a task's `pick_result` adds.
- `WS /v1/events` sends every event with a sequence number. A client that reconnects with `since_seq`
  gets what it missed. If the per-run buffer dropped some, it first gets a `gap` frame with the count.
  `run_id=cell` is the cell's own stream: the jaws question, a halt with no run, the stop record, "the cell is
  clear" and the planner.
- Each event carries `human`, a sentence for the operator, and `data`, the machine payload. An
  attempt's `data` includes `camera_world` whenever the motions say something about the camera world.
- **Two stops, two meanings.** `POST /v1/pick/stop` ends a pick run before its next attempt; the motion in
  flight completes, and the run ends `cancelled`, which is not counted as a failed pick. It refuses any other
  kind of run (`409 not_a_pick`): a task's stop is `POST /v1/task/stop`, and halt now is `POST /v1/cell/brake`.
  Run states are `running`, `finished`, `cancelled` and `failed`.

## Speech

Speech fills the prompt box and never starts anything: a person reads the card it opens and starts the task
with Start, as for a typed command. `POST /v1/voice/transcribe` takes 16-bit PCM WAV, which is what the console
records in the browser. A voice check runs first, so a recording with no speech comes back as an
empty proposal with its reason. The text stays in the spoken language, and the answer carries the
whole transcript: language, where the language came from, duration, engine, weights, device and
decode time. The speech engine is built from `models.stt` alone, loads on the first request and stays
loaded. `POST /v1/voice/talk` and `POST /v1/voice/listen` are push to talk on the cell PC's own
microphone; no console screen calls them yet.

## The camera view

`GET /v1/camera` returns one image and says which of three it is: `camera` (a frame taken now),
`synthetic` (a rehearsal cell's drawing; it has no camera) or `overlay` (the grasp render). No picture
is a `200` with a reason, never an error: `not_built`, `pick_owns_camera`, `no_colour_source` or
`encode_failed`. While a run is active the view never touches the camera and shows the overlay
instead. An overlay's age is a lower bound until the server has seen one image replace another, and
`age_is_exact` says which. The cockpit's live image is `GET /v1/camera/live`; this route stays for programs.

## History

| Source | Lives in | Survives a restart | Includes the command line runner |
|---|---|---|---|
| Runs | memory, the last 200 | no | no |
| Records | a JSONL file | yes | yes |

The console logs records to `logs/console/grasp_records.jsonl` when `grasping.record_log_path` is
unset, and to the configured path otherwise. The KPIs come from `compute_kpis`, the function
`python -m src.robot.grasping.replay --records` uses, so the console and the offline gate agree.
A rate with an empty denominator is withheld and named in `unmeasurable` rather than shown as zero:

| KPI | Withheld when |
|---|---|
| `false_positive_grasp_rate` | always: it needs a post-grasp re-check this stack does not have |
| `dense_recovery_success_rate` | no record both ran in a dense mode and recorded a recovery action |
| `first_attempt_success_rate` | no record carries a recovery action |
| `median_cycle_time_s` | no record carries a cycle time |

The pick duration to quote is `median_attempt_seconds`, which every attempt stamps. Runs of every kind are
in `GET /v1/history/runs` and the CSV, with their `kind`, `stop_code` and `parts_placed`. The stop record and
the run it names are the exception: `python -m api` keeps them across a restart
([After a stop](#after-a-stop-zelle-ist-frei-then-restart-or-home)).

## The status panel

`GET /v1/cell/status` reads the stream the controller already broadcasts, so polling it costs a
running motion nothing. `?include_controller_state=true` adds robot mode, safety mode and the
controller's message at the price of a dashboard round trip per call; poll it at seconds. Joint
torques are never offered, because they go through the control interface a pick uses, and nothing
here clears a protective stop: that is done at the pendant, where the arm is visible. The ready bar
reads the controller from the receive stream alone (`quick_robot_status`), never the dashboard.

## Files

| File | Holds |
|---|---|
| [`__main__.py`](__main__.py) | `python -m api`: `--profile`, `--port`, `--data`, `--reload`; where the console keeps its stop; checks the port before it announces it |
| [`app.py`](app.py) | the app, the `/v1` mount, the one error envelope, serving the built frontend, and the ordered shutdown |
| [`cell.py`](cell.py) | the session: config root, profile chain, the built cell, the run flag, the stop record and the halt it carries, the stop file, the countdown stamps, and `take_down` |
| [`lifecycle.py`](lifecycle.py) | build, preview, token, connect, disconnect, and the rollback |
| [`codes.py`](codes.py) | every typed code and its one status: run kinds, stop codes and classes, events, refusals, lights, blockers, the jaws question, command notes |
| [`runs.py`](runs.py) | a run on its own thread: the pick run, `start_kind` for the others, the countdown, stop, halt, the record, the 200-run window |
| [`task_run.py`](task_run.py) | the bodies of a task, Home and the planner start; the hooks that put a task's events on its stream |
| [`readiness.py`](readiness.py) | the ready bar, the cell's facts, and the gates every moving route keeps, in their order |
| [`jaws.py`](jaws.py) | the hand as the console reads it, and the jaws question answered in the browser |
| [`teach.py`](teach.py) | teaching one pose by hand from the browser, the heartbeat, and the poses read and chosen |
| [`live.py`](live.py) | the live image: a peek per rig, downscaled and encoded, and each rig's frame age |
| [`overlays.py`](overlays.py) | the bounded store of the overlays a task captured |
| [`events.py`](events.py) | per-run sequence numbers, the bounded buffer, catch-up and the gap, and the cell's own stream |
| [`history.py`](history.py) | reading the record log, the KPI roll-up, the CSV writers |
| [`viewfinder.py`](viewfinder.py) | camera, synthetic or overlay, and the no-picture reasons |
| [`telemetry.py`](telemetry.py) | what one status read may touch, and what it costs |
| [`audio.py`](audio.py) | decoding an uploaded WAV, and its two refusals |
| [`schemas.py`](schemas.py) | the wire types |
| [`constants.py`](constants.py) | the log directory and each module's log file |
| [`routers/`](routers/) | one module per route group: preflight, config, cell, diagnostics, pick, media, history, codes, task, jaws, live, poses, commands |

Each module that acts logs to its own file under `logs/api/`. The connect token is never logged.
Decisions live in the library, where the command line reaches them too: what may be written
([`src/config/edit.py`](../src/config/edit.py)), what blocks
([`src/robot/execution/real_cell/preflight.py`](../src/robot/execution/real_cell/preflight.py)),
who owns a cell ([`src/robot/execution/cell_lock.py`](../src/robot/execution/cell_lock.py)), how
a cell is built ([`src/robot/execution/autonomous_grasp/cells.py`](../src/robot/execution/autonomous_grasp/cells.py))
and what a task does ([`src/robot/execution/task.py`](../src/robot/execution/task.py)).

[`scripts/console/capture_event_log.py`](../scripts/console/capture_event_log.py) drives the console's
scenarios through `TestClient` and writes the event logs the frontend's run model replays
(`frontend/src/test/fixtures/`), every one but `teach_one_pose`, which stays written by hand until a teach
scenario lands. Rerun it after any change to what the console says: the output is byte for byte the same
otherwise (fixed run and question ids, a fixed clock).

## What is proven and what is not

| Capability | Evidence |
|---|---|
| The endpoints, against a copy of the config tree | the `tests/test_api_*.py` files; the preflight is asserted equal to the command line's |
| The contract: every route of this table, every code in the catalog and in `frontend/src/api/codes.ts`, the recovery record | [`tests/test_api_contract.py`](../tests/test_api_contract.py) |
| A task end to end on `console_dummy`: once, until empty, stop after the part, halt and "Backen leer", Home's countdown | the captured event logs and `tests/test_api_task*.py` |
| A stop that outlives a restart of the server: the record read back uncleared, the halt carried to every later build, an unreadable file | [`tests/test_api_a_stop_outlives_a_server_restart.py`](../tests/test_api_a_stop_outlives_a_server_restart.py), and a real restart of `python -m api --profile console_dummy` |
| Halt now on a CB3 controller: the brake, the stop point on the judged leg, 200 moves with no early return, a braked path that sends no later waypoint, no DO0 edge after a halt | measured against real controller software (URSim CB3, `scripts/ursim/probe_halt.py`) |
| The console's task on a CB3 controller: three cycles with 2 DO0 edges each, halt in the approach and in the carry, a protective stop, the jaws question in the browser, Restart home | measured against real controller software (URSim CB3, `scripts/ursim/probe_console_task.py`), with the CB3's serial read by a scratch shim that did what the driver now does; the shipped reading is pinned on a local dashboard double ([`tests/test_ur_cb3_serial.py`](../tests/test_ur_cb3_serial.py)) and not rerun on URSim yet ([the runbook](../docs/runbooks/console_at_the_cell.md#3-what-the-probes-prove)) |
| Preflight, build, connect, telemetry, a pick, history and disconnect, the routes the console had before the task | measured against real controller software (URSim, UR3e, 2026-08-20) |
| The cockpit, Setup and the audience window calling these routes in a browser | run on `console_dummy` (Playwright); not yet against URSim ([`frontend/README.md`](../frontend/README.md)) |
| Any endpoint with a physical arm, gripper or camera frame | never touched hardware |

On `console_dummy` the cell response's `vendor` and the status response's `simulated` flag say that
the arm is not real, and the frontend shows both.

## Where the details live

- [`frontend/README.md`](../frontend/README.md): the browser pages that call these endpoints.
- [`docs/runbooks/console_at_the_cell.md`](../docs/runbooks/console_at_the_cell.md): the URSim checklist and
  the supervised first run at the cell.
- [`src/robot/execution/README.md`](../src/robot/execution/README.md): `run_task`, the task from a program.
- [`src/config/README.md`](../src/config/README.md): the config tree these forms read and write.
- [`scripts/ursim/README.md`](../scripts/ursim/README.md): the controller software this was measured against.
- [`docs/cli.md`](../docs/cli.md): the same capabilities from a shell.
