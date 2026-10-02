# Operator console pages (`frontend/`)

The browser pages of the operator console: the **cockpit** a person gives Willy tasks from and watches them
run in, **Setup**, **History** and **Settings**, and the **audience window** for a projector. They are React
and TypeScript, built with Vite into `api/static/`, and served by the same [`api/`](../api/README.md)
process that drives the cell, so they need Node.js to build and a running `python -m api` to talk to.

The library, every command line and every Python test run without this folder; a server with no
built pages serves the API alone.

## Try it without hardware

```bash
# 1. the server, on the desk profile: a dummy arm and hand, "Ablage links" to place at and "Parkposition"
python -m api --profile console_dummy

# 2a. development: hot reload at http://localhost:5173; Vite forwards /v1 and its WebSockets to the server
cd frontend && npm install && npm run dev

# 2b. or build once into api/static/; the server then serves the pages at http://127.0.0.1:8000
cd frontend && npm install && npm run build
```

In **Setup**, read the check, build (the dummy arm rehearses by default), read the preview and connect. In the
**cockpit**, type a command and press Enter: without a language model on the desk the card opens by hand, so
leave **Greifen** empty (the desk scene takes anything), keep the default place, and press **Start**. The top
bar's cell chip reads `dummy · verbunden · simulierter Arm`, because a dummy arm proves the console and nothing
about grasping. `WILLY_API=http://10.0.0.5:8000 npm run dev` points the development server at a
console on another machine.

| Command | What it does |
|---|---|
| `npm run dev` | development server on port 5173, `/v1` forwarded to the console |
| `npm run build` | type-check, then bundle into `../api/static` |
| `npm test` | every screen, the run model and the catalogs against captured payloads and event logs (Vitest) |
| `npm run lint` | oxlint |
| `npm run api:types` | regenerate `src/api/schema.d.ts` from `logs/openapi.json` |
| `npx playwright install chromium` | once per machine: the browser the smoke tests drive |
| `npm run e2e` | the browser smoke tests on a real `console_dummy` server (Playwright; `npm run build` first) |

## Your own cell

Start the server on your cell's profile, ending in the cell's own layer (`python -m api --profile <your
cell>,cell`), and open http://127.0.0.1:8000. Work in this order:

1. **Setup**: Prüfen, Aufbauen, Vorschau, Verbinden, Bereit. A toggle hand's jaws question opens over the
   page; answer what you see.
2. **Setup, Posen**: teach the default place and a park pose by hand, one at a time.
3. **Cockpit**: a command, typed or spoken; read the card; **Start**.
4. **History** for the session and the logged attempts; **Settings** for language, theme, view, voice output,
   the talk key and the bench values.

The runbooks [bringing up a cell](../docs/runbooks/cell_bringup.md),
[the first pick on a physical arm](../docs/runbooks/real_cell_first_pick.md) and
[the console at the cell](../docs/runbooks/console_at_the_cell.md) are the procedure around these screens.

> [!WARNING]
> **The controls that start motion are filled lime**: Start, the ask card's options that move, Restart, and the
> confirm of Home, which is outlined until then. **What arms something is filled red**: Verbinden, "Arm
> freigeben" in the teach dialog and "Jetzt öffnen" in the jaws question. **"Sofort anhalten" is
> orange**: one click stops the run and latches the arm, so nothing after the move in flight is sent; only
> with `robot.ur.brake_on_halt` on (off as shipped) is that move braked under control. It is **not the
> emergency stop**: the stop for a moving arm is the physical red button at the cell.

## The pages

| Page | Address | What it is |
|---|---|---|
| **Cockpit** | `/` | the live image, the task's steps, the two stops and Home under it; the numbers and the chat beside it; the ready bar or the run header across the top |
| **Einrichten / Setup** | `/setup` | Prüfen → Aufbauen → Vorschau → Verbinden → Bereit, the taught poses and the teach dialog, and what this cell is |
| **Verlauf / History** | `/history` | the session's tasks in numbers and two charts, every run with its kind, its command and how it ended, the logged attempts and their KPIs, CSV |
| **Einstellungen / Settings** | `/settings` | language, theme, demo or tech view, voice output, the talk key; the config explained and its bench values written |
| **Audience window** | `/demo.html` | a read-only mirror of the cell for a projector: no button, no link, always the black stage |

The old addresses redirect, so a bookmark lands where its work now is: `/pick` to the cockpit, `/cell` to
Setup, `/config` to Settings. `POST /v1/pick` stays for programs; no page calls it.

**The top bar is a state surface.** On every screen it shows four chips, fed by one shared cell poll every 2 s
and the cell's own event stream: the **cell** (vendor, state, and what is on the other end: a simulated arm,
URSim, or a controller about whose arm nothing is proven), the **hand** (a toggle's jaws as a COUNT of its
output's changes, "gezählt, kein Sensor", never as sensed; the driver's class in the tech view only), the
**controller** (a halt, then a protective stop
at the pendant, then a person needed, then not connected) and the **run** (what runs, or a stop record: "Stopp
offen: Zelle freigeben", then "Stopp offen: Neustart oder Home"; a run the server forgot when it restarted is
said as no longer known, never as no run). Beside them: the demo or tech view, DE/EN,
the theme, voice output, the audience window, and in the tech view the Diagnostics drawer. A narrow window
wraps the chips and never hides one.

### The cockpit

- **The live image leads**, on the left, about 62 % of the width: polled at 4 Hz, 2 Hz while a pose is taught.
  LIVE for a camera, PROBE for the rehearsal's drawing, KEIN BILD with the reason; the camera's name and the
  frame's age; a button per camera where the cell has more than one. While a pick measures, the last frame
  stays and the badge says "misst …".
- **Under it**: the steps (Schauen → Erkennen → Greifen → Ablegen → Zurück, with "Hände weg" and "Ziel suchen"
  as chips of their own, part n of N or ∞, attempt n of N), the two stops and **Home**.
- **Beside it**: the numbers on top, the success rate leading in the brand's readout frame, then the parts of
  this task and the median time per part (the tech view adds pushes, empty looks and counts per outcome); below
  them the **chat**, with the command box and the microphone at its foot. A pick counts in the rate once its
  part's end is known (a failed grasp at once, a gripped part once it is placed or not), so the rate never dips
  while a part is carried, and reads "erstes Teil in der Hand" while the first one is.
- **Across the top**: the ready bar before a task, the run header during one, the stop in red after a problem.
  The commands light is named for what it is, **Sprachmodell**, and where there is none it says **Karte von
  Hand**.
- **It fills the window under the top bar**, and below 1200 px it stacks (bar, image, steps and stops, numbers,
  chat) with the stop row kept in view.
- **Demo or tech view.** The demo view shows the reader's words only, in big type. The tech view adds the
  server's own English sentence under every line, the data, the route badges and the Diagnostics drawer. The
  grasp render is the generator's debug image (its scores, its candidates), so it is pinned over the stage in
  the tech view only; the demo view shows it as the part card's thumbnail. The bin a camera found is pinned in
  both.

**An overlay is its own image.** A new overlay is pinned over the stage for 5 s, never drawn over the moving
live picture, with a band that says when it was decided and, on a wrist camera, "der Arm hat sich seither
bewegt". It stays as the thumbnail of its part's card.

### Talking to Willy

- **Click the microphone to start, click again to stop.** A foot switch keeps hold to talk: the talk key
  records while it is held, wherever the focus is.
- **The transcript lands in the box**, editable, marked as spoken until it is edited. **Enter only reads the
  sentence**: the VLM fills the **Understood card** (Greifen, Ablegen, Umfang: Einmal or Bis leer, Danach, and
  the Advanced drawer), and nothing moves.
- **Start is the confirmation.** One click, no second dialog, and its label names the first motion: "Start –
  der Roboter fährt zu Blick 1" (configured looks), "nach Home und schaut" (a wrist camera with no looks) or
  "zum ersten Griff" (a fixed camera). A due 3 s countdown ("zuerst 3 s Countdown „Hände weg“") and a camera
  place's fallback (put back, then ask) are lines of their own under it. Start stays off, saying why, until the
  ready bar is green and the card is complete.
- **A refusal is said where the click was.** A refused Start, Restart, Home or ask-card option is said right
  above its buttons and brought into view; the demo view folds the code and the HTTP status under "Details",
  and the tech view shows them open.
- **Without the language model** (`501`, or not loaded) the card opens by hand with the reason, every field
  editable; "Laden" loads the reader where the cell allows it. A sentence read as "stop" stops nothing: the card
  points at the stop buttons.
- **The chat is locked during a run**, the box and the microphone; a teach switches the microphone off too.

The page encodes 16-bit PCM WAV itself (`src/prompt/recordWav.ts`), because no browser records WAV and WAV is
the only format the server decodes; anything else is refused with 415. The browser offers a microphone only
over HTTPS or on localhost: a console opened as `http://<cell PC>:8000` from another machine has none, and the
box says why.

### Stopping

- **"Nach diesem Teil stoppen"** asks the task to end after the part in hand: a held part is still placed and
  the arm returns. For a Home run it stops the move before it is sent; for a pick run, before its next attempt.
  It never stops the arm.
- **"Sofort anhalten"** is one click, no dialog, and is **not the e-stop**: it says "bremst kontrolliert" where
  the arm brakes a move in flight, and "hält vor der nächsten Bewegung" where it only latches, as the owner's
  cell does as shipped. It is enabled whenever the cell is connected, with or without a run: drawn calm (a
  darker orange, no glow) while nothing runs, full orange during a run, and one click whatever else is open,
  the Diagnostics drawer included (the stop row stays above the drawer's shade).
- **"Anhalten nicht bestätigt: Not-Aus drücken"** appears only where the arm brakes and no confirmation came
  within 1.5 s while it moved, and also when the arm reports the brake unconfirmed after the run ended. A
  controller that reports a stop has answered it. A halt reads as braked only where the arm reports it so, and
  the stop card never claims a brake the arm did not report.
- **The stop card** opens in the chat after a problem stop, rebuilt from the cell's stop record, so it survives
  a reload. Its checklist, in order: a stopped controller is cleared at the pendant; **"Zelle ist frei"**; the
  jaws ("Backen prüfen" opens the jaws question for a toggle, and a count that says closed asks "Jetzt öffnen"
  at once; "Backen leer" is offered wherever the server still believes a part in a hand that measures nothing,
  the stopped run's belief included); a hand among parts is jogged clear first; then **Restart** or **Home**,
  each behind its confirm dialog and on only when every gate is green. Restart names the stopped task's own
  return pose and waits until the stopped run's record is read. While the way back runs, the card folds to one
  line.
- **After a restart of the server** the card comes back from the stop the server kept: the stopped run's record
  answers, its events do not, and a stop of unknown origin (a file the server could not read) offers Home only.
  Until the cell is connected no hand is read, so the card says what the stopped run believed it held, never
  "Kein Teil in der Hand" about a hand nobody read.

### The chat

- The operator's lines are bubbles on the right, "gesprochen" where they were spoken; Willy's are on the left:
  one card per part (the overlay thumbnail, the looks fused, the jaw faces seen, the hand-eye gap, the push, the
  hold "nicht gemessen (kein Sensor)" on a hand that measures nothing), the step lines of the part in hand, and
  the Understood, ask and stop cards in the same flow. Looks that found nothing more read as one quiet line,
  "Nichts mehr gefunden (n Blicke).", never as a part that failed.
- After a task Willy reports the end and asks "Was soll ich als Nächstes tun?", and a live end hands the focus to
  the box. Earlier tasks read as one line each, with their part count; a tab that did not see them reads the
  session's ended tasks from `GET /v1/runs`, one line each, with the command as it was said.
- **The reader keeps their place.** Scrolled up to read, the chat stays there however often the cell is polled,
  and "Zu den neuesten Zeilen" brings them back. A new card opens from the start of its turn.
- The conversation lives in `sessionStorage['willy.chat']`, the last 200 entries: it survives a reload of the
  tab and ends with it.

**Voice output** is off by default (the speaker in the top bar, `willy.voiceOut`). On, the browser's own speech
synthesis says the start, the end with the next question, a problem and a question, in the reader's language,
live only and once each, in the cockpit. The teach dialog says its own 30 s warning on the Setup page.

### Setup

- **One step at a time, the one the server says.** A cell that is down opens at Prüfen, whatever it found, and
  goes on at "Weiter: Aufbauen" (remembered for the browser session); the step wears "!" instead of a tick while
  the checklist warns or blocks.
- **Building is real by default** unless the vendor is `dummy`. **Verbinden** is the only red button here, and
  it carries the token of the preview the person read.
- **A latched arm** (a halt, or a Disconnect during a move) is shown above every step and is released only by a
  person: tick "Ich habe hingesehen …", then "Zelle ist frei". **A stop nobody has cleared** ("Ein Stopp steht
  noch", after a restart of the server too) is offered the same word there once the cell is built, before
  Verbinden: a toggle's "Jetzt öffnen" at Connect waits for it, and so does "Backen prüfen" in the ready panel.
- **The cell's facts** say what was built, in words: the cameras and the looks, the push, how "halt now" acts,
  whether the carried part is modelled, and the detector by name (GroundingDINO + SAM, VLM, RT-DETR); its config
  word (`grounded_sam`, `vlm`, `rtdetr`) is in the tech view, and so is the raw telemetry (TCP, quaternion,
  joints).

### Poses and teaching

- The poses panel shows Home read-only, every taught pose by its label with its verdict chip (frei; frei ·
  gerader Start; abgelehnt: Kollisionsprüfung / Planer), the default place as one radio, and "Pose einlernen" or
  why not now ("Planer startet noch").
- **Before anything is freed** the teach dialog shows the file the pose is written to and the payload the
  controller compensates for, with "Nutzlast neu lesen". A place pose says where the fingertips go: where the
  part's bottom is let go. Only the red **"Arm freigeben"** frees the arm.
- **While the arm is being freed** there is no "Abbrechen" and Escape does nothing; an answer that lands after
  the dialog went away is held at once.
- **While it is free** the dialog polls four times a second, and every poll is the heartbeat; a closing tab sends
  the cancel beacon on `pagehide`. The time left is shown, with the 30 s sentence (said aloud once where voice
  output is on). "Speichern" captures once the arm is still; "Halten" holds at once, in the halt's orange; a
  lapse or the 5-minute limit says "Hält, sobald der Arm stillsteht."
- The verdict ends the dialog: a saved pose with its screen, or a refused one with the nearest clear joints.

### The jaws question

It opens over every screen whenever a toggle hand asks where its jaws stand, at Connect or at a check before a
Restart. It offers **exactly the server's choices and never a default**: none is focused or lit, the focus
goes to the dialog itself, and Escape does nothing. **"Jetzt öffnen" is one switch of Tool-DO0**, in the
arming red, and the dialog says to hold the part first. It shows only the question that waits; a flag with no
waiting question means the hand is acting on an answer or a check is under way, and the dialog offers nothing.
The answer returns the next stage, a read that lands after an answer is dropped, and a question can end
`cancelled` when a run takes the cell.

### History

- **The session** counts every run the server keeps (its newest 200); the table lists the newest 50, the rest
  one click away. Rates read as percentages.
- **The rate is the cockpit's**: placed parts over the picks whose end is known; empty looks, and a part a
  person halted, cancelled or disconnected in the jaws, are left out. The runs table's parts column reads
  "abgelegt / gegriffen". The two charts, **Erfolg je Auftrag** and **Median-Zeit pro Teil**
  (from the search to the return), are replayed from the newest 12 tasks' own events through the cockpit's
  reducer, and the tooltip says "x von n Griff/Griffen".
- **Nothing unmeasurable is zeroed**: a KPI with no denominator is listed with its reason, and the demo view
  folds the server's English reason behind "Warum (Server)".

### Settings

Language, theme, the demo or tech view, voice output (off by default) with "Stimme testen", and the talk key: F8
unless another is set, and only F1-F24, Pause or Scroll Lock, keys typing never uses. Each switch writes the key
the rest of the console reads (`willy.lang`, `willy.theme`, `willy.view`, `willy.voiceOut`, `willy.talkKey`),
per browser. Below them the keys this server may write, each in the reader's words with a one-line hint (the
server's own English sentence under "Details (Server)" in the demo view), and a controller address only for the
cell's own vendor.

### The audience window

Opened from the top bar as its own popup: drag it to the projector and press F11. It shows the live image of
the primary camera full-bleed, a new overlay pinned from the moment it arrives until 5 s after it was captured
(allowing up to 2 s between the server's clock and the screen's), the step in big words with the part and where
it goes, the counters, the target chip, a large read-only stop or ask card ("Zelle freigegeben: wartet auf
Neustart oder Home" once a person cleared the cell), and the **STILL GRINDING** card: Teile heute, Laufzeit,
Kaffeepausen 0, Gehaltserhöhungen verlangt 0. **It has no control at all, and it is always dark.** It follows
the run the cell names, the stop record (so a stop shows when the window opened after it), and runs too short
for the cell poll, and names taught poses by their labels.

## What the screens promise

| The screen | Because | Test |
|---|---|---|
| Start names the first motion, the countdown and a camera place's fallback, and stays off until ready | Start is the confirmation; nothing moves on a click that did not say where | yes |
| Enter and a spoken sentence only fill the card; nothing starts but Start | a misheard word must never move an arm | yes |
| "Sofort anhalten" is one click, orange, and says it is not the e-stop, whatever else is open | the red button at the cell is the e-stop | yes |
| "Not-Aus drücken" only where the arm brakes and no confirmation came | a false alarm teaches people to ignore the real one | yes |
| After a stop nothing starts by itself; Restart and Home stay off until every gate is green, and ask first | the arm stands where the problem left it | yes |
| The stop card is rebuilt from the stop record, never from "the newest run", and never says "no part" about a hand nobody read | the newest run can be a planner or a teach run; after a restart no hand is read until the cell is connected | yes |
| The jaws question offers exactly the server's choices, never a default, and cannot be dismissed into one | only a person looking at the jaws can say where they stand | yes |
| A toggle's jaws are said as counted, not sensed | a count is not a measurement | yes |
| A push is said as pushed, with its distance, only where the pick loop says it ran | a push claimed that never ran sends a person looking for a part that never moved | yes |
| A blocker set aside is said in the demo view; a clearing that stopped says the arm stands and a person decides, never that it was set aside | the obstacle may still be in the hand | yes |
| A halt is said as a halt, never as a stopped controller | their remedies differ: a person's word, or the pendant | yes |
| An overlay is pinned as its own image, with when it was decided and that the arm has moved since | a picture over the moving image would read as now | yes |
| The live image keeps its last frame while a pick measures, and names a rehearsal drawing PROBE | blank reads as a fault, a drawing reads as a camera | yes |
| A teach frees the arm only at "Arm freigeben", beats four times a second and cancels on `pagehide` | a freed arm with no dialog is an arm nobody holds | yes |
| A simulated arm says simulated; an unproven controller is never called physical | a simulated reading looks exactly like a physical one | yes |
| A missing measurement says "not offered", a blank controller state "not asked" | a zero reads as a measurement, blank as fine | yes |
| An unmeasurable KPI is listed with its reason; a task without a value says "kein Wert" | hidden reads as fine and zero as bad | yes |
| A refusal is said in the reader's words with its code; an unknown code reads "Abgelehnt (<code>)" | a refusal nobody can read is a refusal nobody acts on | yes |
| A lost event is shown as a gap, and the counters come from the run record after it | a replay that looks complete would draw a run that never was | yes |
| The audience window has no button and no link | a viewer is never one click from a motion | yes |
| The joke lives in the audience window only | the cockpit, Setup and every operator text stay factual | yes |
| Preflight fixes are shown word for word, per row | a checklist that says wrong without saying what to do has only moved the guessing | yes |

"yes" means a Vitest file asserts it (`npm test`); the browser smoke tests drive several of them end to end.

## Where the types come from

Nothing describes a payload by hand. The server's Pydantic schemas produce its OpenAPI document,
`npm run api:types` turns that into `src/api/schema.d.ts` (generated, do not edit), and
`src/api/client.ts` names those types for the screens. `src/api/codes.ts` mirrors `GET /v1/codes`, every code
the console answers with, and the types hold the mirror to the server's unions at compile time. After any
schema change on the server, from the repository root:

```bash
python -c "import json; from api.app import create_app; print(json.dumps(create_app().openapi(), indent=2))" > logs/openapi.json
cd frontend && npm run api:types
```

The event logs the run model replays (`src/test/fixtures/*.json`) are captured from the real console:
`python scripts/console/capture_event_log.py` drives each of its 10 scenarios through the console and writes
them byte for byte the same (fixed run and question ids, a fixed clock). The five `console_dummy_*` logs are the
desk cell's, word for word; the four `task_*` logs and `jaws_question_round_trip` come from a scripted cell for
what the desk cannot show (a toggle hand, a push, a bin the camera found, a halt during the grasp, the jaws
question answered in the browser). **`teach_one_pose` is the one log written by hand**: teaching needs a
hand-guided arm double, so it stays as written until a scenario for it lands. Rerun the script after any change
to what the console says.

## Languages and the look

- **German by default, English on a switch.** The language lives in `localStorage['willy.lang']` and
  `<html lang>` follows it. Each area keeps its own catalog pair, `i18n.de.ts` and `i18n.en.ts` (the shell and
  every code in `src/i18n/`, then `cockpit`, `prompt`, `setup`, `jaws`, `screens`, `demo`); German is the
  source, and English is typed against it, so a missing key fails `tsc`. `useT()` without a provider answers
  English. The browser translates **codes**, never the server's sentences: those stay English and show as
  details, open in the tech view and folded behind "Details (Server)" in the demo view. A code no catalog knows
  reads "Abgelehnt (<code>)".
- **The catalogs are tested**: key parity, no empty text, the same placeholders in both languages (plural forms
  included), the joke only in the audience window, plain German, and no word in capitals for emphasis. The
  cockpit's own test holds its halt wording: a brake claimed only where the arm brakes, a halt never called a
  protective stop, the e-stop named only to say what is not one or to ask for it, and "the robot moves" on every
  control that starts motion.
- **Dark by default**, light on a switch for a bright hall (`willy.theme`; an old `system` choice reads as
  dark). Both pages carry an inline script that stamps the theme and the language before the first paint, and
  the audience window is always dark.
- **The brand** is the look of [the logo](../docs/assets/willy_logo.png) and
  [the banner](../docs/assets/willy_banner.png): a black stage, the logo's **neon lime** as the one accent (a
  darker lime in the light theme, legible as text), the banner's **HUD corner marks** (`.hud-frame`) around the
  live image and the main cards, the **Willy mark** (an inline SVG, also `public/favicon.svg`) with the
  wordmark **WORKAHOLIC WILLY** in the top bar, and chips and numbers in monospace like the banner's STILL
  GRINDING box (`.readout`). **System fonts only**: no font file is loaded.

## How the pages are built

- **No server address anywhere.** In production the pages come from the server itself; in development
  the Vite proxy makes them same-origin too. No build and no bookmark can aim the console at a
  different cell than the one whose server opened it.
- **The server is the authority.** One shared poll of `GET /v1/cell` every 2 s feeds every screen, and the
  cell's own event stream makes it react in the same instant. A run is never kept in the browser: after a
  reload it is rebuilt from its record and its events, replayed from seq 0 through a pure reducer
  (`src/model/runModel.ts`). The server enforces every gate on its own; a disabled button only says so first.
- **Two reds, an orange and the lime, apart by colour and by form.** `--alarm` is only ever a tinted surface
  with a border, for a `block` status; `--arm` is only ever a filled button with white text, for what arms
  something (Verbinden, "Arm freigeben", "Jetzt öffnen"); `--halt` is only ever a filled orange button with a
  dark label, for "Sofort anhalten"; the lime accent fills the controls that start motion, since IEC 60204-1
  never colours a start red. Colour never carries the text.
- **Contrast is a test.** [`tests/test_console_contrast.py`](../tests/test_console_contrast.py) reads the
  numbers out of `styles.css` and checks WCAG AA for every text tier on every surface it can land on, the halt
  label on its fill, the brand's tokens, text on tinted surfaces and the status colours used as text.
- **One file of design tokens.** `src/styles.css` is a Tailwind v4 theme: the palette is declared under
  `@theme inline` over runtime custom properties, every size in rem so a large screen scales, and hit targets
  of at least 44 px. The camera stage is `--stage`, black in both themes because the image is the content, and
  whatever is drawn over it uses `--stage-*`, whose lime stays the neon one. Area styles sit beside their area
  and begin with `@layer theme, base, components, utilities;`. A state class that is also a Tailwind utility
  (`block`, `hidden`, `fixed`, ...) would be overridden from the utilities layer, so the areas name their own
  (`alarm`, `dimmed`).
- **Outcome strings in one place.** `src/lib/outcome.ts` maps an outcome to a pill tone. The server
  spells a success `succeeded`, the literal the KPI roll-up counts.

## Files

| File | Holds |
|---|---|
| `src/App.tsx`, `src/main.tsx` | the shell: the top bar, its chips and switches, the four screens, the jaws question and the confirm host |
| `src/styles.css` | the design tokens, both themes, the brand classes, every shared class |
| `src/icons.tsx` | the Willy mark and the inline SVG icons |
| `src/api/client.ts`, `src/api/events.ts` | the only code that talks to the server, typed from `schema.d.ts`; the event streams, resumed from `since_seq`, with gaps shown |
| `src/api/codes.ts`, `src/api/schema.d.ts` | the catalog of `GET /v1/codes`; generated types, do not edit |
| `src/i18n/` | `I18nProvider`, `useT()`, the shell's catalog and every code's text in both languages |
| `src/model/` | the shared cell poll (`useCell`), the run on screen (`useRun`, `runModel`), the cell stream (`cellModel`), the conversation (`chat`), the preferences (`prefs`) and the confirm host (`confirm`) |
| `src/cockpit/` | the cockpit: stage, ready bar, timeline, stop buttons, stop and ask cards, Understood card and Advanced drawer, chat, numbers, voice output |
| `src/prompt/` | the command box, typed or spoken, the WAV encoder, its catalog |
| `src/speech/speak.ts` | voice output through the browser's speech synthesis |
| `src/setup/` | Setup: the stepper, the ready panel, the poses, the teach dialog, the cell's facts |
| `src/jaws/` | the jaws question, over every screen |
| `src/screens/` | Preflight and Cell (Setup's panels), History with its charts, Settings, Config and Preferences |
| `src/demo/`, `demo.html` | the audience window: its own entry, always dark |
| `src/components/` | status pills, panels, the error banner, the confirm dialog, the Diagnostics drawer, the provenance badge |
| `src/lib/` | the theme, outcome tones, provenance and the async helper |
| `src/test/` | the render helpers and the captured event logs |
| `e2e/`, `playwright.config.ts` | the browser smoke tests and the server they start |
| `public/favicon.svg` | the Willy mark |

`api/static/` and `node_modules/` are not committed; `npm run build` regenerates the bundle.

## Testing

- **`npm test`** runs 21 Vitest files and 414 tests: every screen against captured payloads, the run model
  against the captured event logs, the catalogs, the theme scripts and the client. A test picks a run out of a
  captured log with `runOf`, `eventsOf` and `upTo` (`src/test/render.tsx`), never by its id.
- **`npm run build`, then `npm run e2e`** runs three Playwright specs, four tests, against a real `python -m api
  --profile console_dummy` the run starts itself: `smoke.e2e.ts` connects in Setup, starts a task in the
  cockpit, halts it, clears the cell and restarts; `audience.e2e.ts` mirrors a task, its stop card and the
  restart in the audience window; `halt.e2e.ts` halts a running task with one click while the Diagnostics
  drawer is open, at 1366x768 and 1920x1080, the button inside the window. `e2e/serve.mjs` serves the shipped
  desk profile from a copy of the config in `node_modules/.cache/willy-e2e` (`--data`), so the repository's
  config is never written and no stop is kept between runs. `WILLY_PYTHON` names the interpreter,
  `WILLY_E2E_PORT` the port (8761), and `WILLY_E2E_URL` a console that is already up. Playwright is a dev
  dependency only, and its browser is installed once per machine with `npx playwright install chromium`;
  without it the run stops at launch.
- **Not in CI yet.** `.github/workflows/ci.yml` runs the Python checks only, so run `npm run lint`, `npm test`
  and `npm run build` before a change to these pages lands.

## What is proven and what is not

| Capability | Evidence |
|---|---|
| The screens, the run model and the catalogs keep the rules marked yes above | `npm test` (Vitest) |
| A task from Setup to the cockpit, halted, the cell cleared and restarted, the audience window mirroring it, and one click halting with the Diagnostics drawer open | `npm run e2e` (Playwright, Chromium) against `console_dummy` |
| The server serves the built pages, and an unknown `/v1` path stays a JSON 404 | [`tests/test_api_serves_the_console.py`](../tests/test_api_serves_the_console.py) |
| Contrast and the brand's tokens | [`tests/test_console_contrast.py`](../tests/test_console_contrast.py) |
| Preflight, build, connect, live status, a pick, history and disconnect, through the console's earlier pages | measured against real controller software (URSim, UR3e, 2026-08-20); the cockpit and Setup have replaced those pages since |
| The routes the cockpit calls for a task, halt now, the way back and the jaws question | measured against real controller software (URSim CB3), through the API in-process, not through these pages ([what the probes prove](../docs/runbooks/console_at_the_cell.md#3-what-the-probes-prove)) |
| The cockpit, Setup and the audience window in a browser | run on `console_dummy` (Playwright); not yet against URSim ([the runbook](../docs/runbooks/console_at_the_cell.md), the URSim checklist, step 3) |
| Any screen with a physical arm, gripper or camera | never touched hardware |

The cockpit has not run at the cell yet: the controller-stopped banner live, the toggle's jaws in the stop card,
several cameras, a real overlay, a real VLM reading and the German voice of the cell PC are owed there
([the console at the cell](../docs/runbooks/console_at_the_cell.md)).

## Where the details live

- [`api/README.md`](../api/README.md): every endpoint these pages call, and what each refuses.
- [`docs/runbooks/console_at_the_cell.md`](../docs/runbooks/console_at_the_cell.md): the console on URSim, then at the cell.
- [`scripts/ursim/README.md`](../scripts/ursim/README.md): the controller software the console was measured against.
