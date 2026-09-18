# Operator console server (`api/`)

The HTTP and WebSocket server behind the browser console. It checks a cell, builds and connects it,
runs picks from a typed or spoken prompt, and shows what happened. It needs the packages in
`requirements.txt` (FastAPI and uvicorn are pinned there) and a config tree, and it serves on
127.0.0.1 only, with no login.

Nothing under `src/` imports this package, so a cell that never starts it behaves exactly the same.
It builds a cell with the same calls `Cell` makes, so the cell the browser drives is the cell your
code drives. The pages themselves are in [`frontend/`](../frontend/README.md).

## Try it without hardware

```bash
python -m api --profile console_dummy
```

`console_dummy` is the desk profile: a dummy arm and a dummy hand, no controller, no camera, no GPU.
The server prints the preflight summary (`0 blocking`) and `serving http://127.0.0.1:8000`. Open
http://127.0.0.1:8000/docs for the interactive OpenAPI page, or http://127.0.0.1:8000 once the
frontend is built. The same flow from a script, with `httpx` (also pinned in `requirements.txt`):

```python
import httpx

with httpx.Client(base_url="http://127.0.0.1:8000", timeout=60.0) as api:
    print(api.get("/v1/preflight").json()["ok"])                     # True: nothing blocks
    api.post("/v1/cell/build", params={"rehearse": True}).raise_for_status()  # a desk scene, no camera
    preview = api.get("/v1/cell/connect-preview").json()             # what connecting will move
    api.post("/v1/cell/connect", json={"token": preview["token"]}).raise_for_status()
    run = api.post("/v1/pick", json={"prompt": "", "picks": 1}).json()  # 202 and a run id
    print(run["id"], run["state"])                                    # run-..., running
```

`GET /v1/runs/{id}` then reports `finished` with one success, and `WS /v1/events?run_id=<id>`
replays the run's events. A dummy arm accepts every motion and reports success, so a desk run proves
that the console and the library agree and says nothing about grasping.

## Your own cell

1. Start the console on your cell's profile: `python -m api --profile <your cell>`. A profile is the
   name of your cell's layer in the config tree ([guide 01](../docs/guide/01-configuration.md)).
   Without `--profile` the console uses `WILLY_PROFILE`, then the base tree.
2. Work down the Preflight screen or `GET /v1/preflight`. Every row carries its fix, and the verdicts
   are the ones `python -m src.robot.execution.real_cell --check` prints.
3. Write the values a person measures at the bench from the Config screen or `PATCH /v1/config`
   (see [What you can change here](#what-you-can-change-here)). Everything else is edited in YAML.
4. Build, read the connect preview, connect, pick. On a real cell a build opens the cameras and loads
   the models, which takes tens of seconds.

The procedures are the runbooks [bringing up a cell](../docs/runbooks/cell_bringup.md) and
[the first pick on a physical arm](../docs/runbooks/real_cell_first_pick.md). The same steps from your
own code are in [`examples/`](../examples/README.md).

> [!WARNING]
> `POST /v1/cell/connect` moves hardware. A Robotiq activates by sweeping its full finger travel; a
> vacuum cup's connect switches the ejector on at once and releases whatever it holds; an I/O jaw may
> open. The connect therefore takes only a token from `GET /v1/cell/connect-preview`, which lists what
> will move. The token is single use, expires after five minutes and is void once any config value
> changes. There is no emergency stop endpoint: the stop is the physical button at the cell.

## The endpoints

Everything is under `/v1`, and every failure comes back as one envelope, `{code, message, detail}`,
with `code` a string your client can branch on.

| Route | What it does |
|---|---|
| `GET /v1/health` | the server is up; says nothing about the cell |
| `GET /v1/preflight` | the cell checklist, each row with its fix |
| `GET /v1/config/explain?key=...` | a value, its type, default, the layer that set it, and its YAML comment |
| `GET /v1/config/writable` | the keys this server may write, described by the library |
| `PATCH /v1/config` | write a group of keys, all or none |
| `GET /v1/cell` | state, arm, gripper, and which process holds the cell lock |
| `POST /v1/cell/build?rehearse=` | assemble the cell; touches no robot. `rehearse=true` uses a desk scene |
| `GET /v1/cell/connect-preview` | what connecting will move, and the token that acknowledges it |
| `POST /v1/cell/connect` | `{token}`: arm, then gripper; the arm is rolled back if the gripper refuses |
| `POST /v1/cell/disconnect` | gripper, then arm; safe to call twice; frees the lock |
| `GET /v1/cell/status` | live pose, joints and wrench; `?include_controller_state=true` adds modes |
| `GET /v1/diagnostics` | SDKs, motion stack, perception stack, controller reachability; moves nothing |
| `GET /v1/diagnostics/route?prompt=` | which perception route a prompt would take, from the text alone |
| `POST /v1/pick` | `{prompt, picks}`: starts a run, answers `202` with its id. This moves |
| `POST /v1/pick/stop` | do not start the next attempt; the motion in flight completes |
| `GET /v1/runs`, `GET /v1/runs/{id}` | recent runs, and one run |
| `WS /v1/events?run_id=&since_seq=` | replay a run's events from a sequence number, then follow it live |
| `GET /v1/camera` | one picture and one sentence saying what it is |
| `POST /v1/overlay/enable` | turn on the grasp overlay render; it costs time on every pick |
| `WS /v1/overlay` | each new overlay image, with its age |
| `POST /v1/voice/transcribe` | a WAV upload to a text proposal for the prompt box; starts nothing |
| `POST /v1/voice/talk` | `{pressed}`: press or release the push to talk switch |
| `POST /v1/voice/listen` | one push to talk turn on the cell PC's microphone, to a proposal; starts nothing |
| `GET /v1/history/kpis` | the KPI roll-up over the logged attempts |
| `GET /v1/history/records` | logged attempts, newest first |
| `GET /v1/history/runs` | this session's runs, from memory |
| `GET /v1/history/runs.csv`, `GET /v1/history/records.csv` | one row per run, and one row per attempt |

## What you can change here

The console starts tasks; it does not set parameters. Speed, acceleration, workspace limits and safety
switches are read-only here and edited in YAML, next to the comment that says why each value is what
it is. The writable keys are an allowlist in [`src/config/edit.py`](../src/config/edit.py): the
payload mass and centre of gravity, the three tool frame keys, each camera rig's serial number and
whether it is enabled, which rig is primary, and the UR and KUKA controller addresses. A `PATCH`
takes them as a group, because the three tool frame keys only validate together, and a refused write
leaves every file byte-identical.

## What it refuses

| Code | Status | When | What to do |
|---|---|---|---|
| `not_acknowledged`, `stale_token` | 428 | connect without a valid preview token | read the preview again, then connect with its token |
| `no_real_gripper` | 403 | the build could not make the configured gripper and put a null one in its place | fix the gripper section; `GET /v1/cell` has the reason |
| `cell_busy` | 409 | another process holds this controller's cell lock | stop the other program; `GET /v1/cell` names it |
| `not_built`, `wrong_state` | 409 | preview before a build, connect twice, rebuild while connected | build first, or disconnect first |
| `driver_refused` | 502 | the controller refused the connect (payload, tool frame, network) | the message is the driver's own sentence |
| `build_refused` | 422 | a build failed, for example a camera that did not open | the message names the cause |
| `no_robot_configured` | 409 | the tree and profile configure no robot | start with the profile that has the arm |
| `not_connected` | 409 | a pick before the cell is connected | connect first |
| `prompt_not_routable` | 422 | the prompt needs the VLM route and this cell has none | configure the VLM, or set `on_unavailable: degrade` |
| `run_active` | 409 | a second pick, or a config write, while a run owns the cell | wait for the run, or stop it |
| `not_writable`, `unknown_key` | 403, 404 | a key outside the allowlist, or no such key | edit it in YAML |
| `invalid_value` | 422 | the loader rejected the written tree; every file is restored | the message is the loader's |
| `cell_connected` | 409 | the primary rig or a controller address, while connected | disconnect first |
| `no_target` | 409 | no file under the config root backs the key | point the console at the config tree with `--data` |
| `audio_format_unsupported` | 415 | an upload that is not WAV | upload 16-bit PCM WAV, which the console records |
| `audio_undecodable`, `audio_too_long` | 422 | a truncated or unusable WAV; longer than Whisper's window | record again, shorter |
| `speech_unavailable`, `speech_model_missing` | 501 | the speech packages or weights are not on this machine | the message names what installs them; or type |
| `talk_not_pressed`, `microphone_ended` | 409 | the switch was not pressed within `timeout_s`, or the microphone stopped | listen again |
| `nothing_recorded` | 422 | the switch came up before any audio | hold the switch while speaking |
| `microphone_unavailable`, `listen_busy` | 501, 409 | no microphone on this host; a listen already running | use the browser's microphone; wait |

A request whose body or query does not match the endpoint answers `422 bad_request`, and an unknown
path answers `404 http_404`, both in the same envelope.

## How a cell comes up

| State | Call | Next state |
|---|---|---|
| `disconnected` (nothing built) | `POST /v1/cell/build` | `built` |
| `built` | `GET /v1/cell/connect-preview` | `built`, holding a token |
| `built`, holding a token | `POST /v1/cell/connect` | `connected`: arm, then gripper |
| `connected` | `POST /v1/cell/disconnect` | `built`: gripper, then arm |

A connect is all or nothing: if the gripper refuses, the arm is disconnected again and the state
stays `built`. The preview's warnings come from the gripper that was actually built, so a cell with a
null gripper shows none. One process owns a cell: `CellLock` in
[`src/robot/execution/cell_lock.py`](../src/robot/execution/cell_lock.py) is keyed on the controller
address and held by an open file handle, so the operating system frees it when the process dies. The
console, `Robot`, `Cell` and the command line runner all take it. It cannot see a second machine; the
reachability check in `GET /v1/diagnostics` covers that.

## Picks and their events

- `POST /v1/pick` returns a run id at once. A run continues when the browser closes, so an arm holding
  a part finishes the pick and puts it down.
- The prompt is what the detector grounds for this run. An empty prompt keeps the cell's own phrase.
- `WS /v1/events` sends every event with a sequence number. A client that reconnects with `since_seq`
  gets what it missed. If the per-run buffer dropped some, it first gets a `gap` frame with the count.
- Each event carries `human`, a sentence for the operator, and `data`, the machine payload. An
  attempt's `data` includes `camera_world` whenever the motions say something about the camera world.
- `POST /v1/pick/stop` ends the run before its next attempt. The run then ends `cancelled`, which is
  not counted as a failed pick. Run states are `running`, `finished`, `cancelled` and `failed`.

## Speech

Speech fills the prompt box and never starts a pick: a person confirms a spoken prompt with the same
button as a typed one. `POST /v1/voice/transcribe` takes 16-bit PCM WAV, which is what the console
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
`age_is_exact` says which.

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

The pick duration to quote is `median_attempt_seconds`, which every attempt stamps.

## The status panel

`GET /v1/cell/status` reads the stream the controller already broadcasts, so polling it costs a
running motion nothing. `?include_controller_state=true` adds robot mode, safety mode and the
controller's message at the price of a dashboard round trip per call; poll it at seconds. Joint
torques are never offered, because they go through the control interface a pick uses, and nothing
here clears a protective stop: that is done at the pendant, where the arm is visible.

## Files

| File | Holds |
|---|---|
| [`__main__.py`](__main__.py) | `python -m api`: `--profile`, `--port`, `--data`, `--reload`; checks the port before it announces it |
| [`app.py`](app.py) | the app, the `/v1` mount, the one error envelope, and serving the built frontend |
| [`cell.py`](cell.py) | the session: config root, profile chain, the built cell, the run flag |
| [`lifecycle.py`](lifecycle.py) | build, preview, token, connect, disconnect, and the rollback |
| [`runs.py`](runs.py) | a run on its own thread, the operator's sentences, the 200-run window |
| [`events.py`](events.py) | per-run sequence numbers, the bounded buffer, catch-up and the gap |
| [`history.py`](history.py) | reading the record log, the KPI roll-up, the CSV writers |
| [`viewfinder.py`](viewfinder.py) | camera, synthetic or overlay, and the no-picture reasons |
| [`telemetry.py`](telemetry.py) | what one status read may touch, and what it costs |
| [`audio.py`](audio.py) | decoding an uploaded WAV, and its two refusals |
| [`schemas.py`](schemas.py) | the wire types |
| [`constants.py`](constants.py) | the log directory and each module's log file |
| [`routers/`](routers/) | one module per route group: preflight, config, cell, diagnostics, pick, media, history |

Each module that acts logs to its own file under `logs/api/`. The connect token is never logged.
Decisions live in the library, where the command line reaches them too: what may be written
([`src/config/edit.py`](../src/config/edit.py)), what blocks
([`src/robot/execution/real_cell/preflight.py`](../src/robot/execution/real_cell/preflight.py)),
who owns a cell ([`src/robot/execution/cell_lock.py`](../src/robot/execution/cell_lock.py)) and how
a cell is built ([`src/robot/execution/autonomous_grasp/cells.py`](../src/robot/execution/autonomous_grasp/cells.py)).

## What is proven and what is not

| Capability | Evidence |
|---|---|
| The endpoints, against a copy of the config tree | the `tests/test_api_*.py` files; the preflight is asserted equal to the command line's |
| The whole console: preflight, build, connect, telemetry, a pick, history, disconnect | measured against real controller software (URSim) |
| Any endpoint with a physical arm, gripper or camera frame | never touched hardware |

On `console_dummy` the cell response's `vendor` and the status response's `simulated` flag say that
the arm is not real, and the frontend shows both.

## Where the details live

- [`frontend/README.md`](../frontend/README.md): the browser pages that call these endpoints.
- [`src/config/README.md`](../src/config/README.md): the config tree these forms read and write.
- [`scripts/ursim/README.md`](../scripts/ursim/README.md): the controller software this was measured against.
- [`docs/cli.md`](../docs/cli.md): the same capabilities from a shell.
