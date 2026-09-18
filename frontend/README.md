# Operator console pages (`frontend/`)

The browser pages of the operator console: the console an engineer uses to bring a cell up and run
picks, and a room view for a projector. They are React and TypeScript, built with Vite into
`api/static/`, and served by the same [`api/`](../api/README.md) process that drives the cell, so
they need Node.js to build and a running `python -m api` to talk to.

The library, every command line and every Python test run without this folder; a server with no
built pages serves the API alone.

## Try it without hardware

```bash
# 1. the server, on the desk profile: no controller, no camera, no GPU
python -m api --profile console_dummy

# 2a. development: hot reload at http://localhost:5173; Vite forwards /v1 and its WebSockets to the server
cd frontend && npm install && npm run dev

# 2b. or build once into api/static/; the server then serves the pages at http://127.0.0.1:8000
cd frontend && npm install && npm run build
```

On the Cell screen, build with the desk scene, read the connect preview, connect, then pick on the
Pick screen. The Cell screen and the room view label the arm `simulated arm` and the sidebar names
its vendor, `dummy`, because a dummy arm proves the console and nothing about grasping.
`WILLY_API=http://10.0.0.5:8000 npm run dev` points the development server at a console on another
machine.

| Command | What it does |
|---|---|
| `npm run dev` | development server on port 5173, `/v1` forwarded to the console |
| `npm run build` | type-check, then bundle into `../api/static` |
| `npm test` | render every screen against captured payloads (Vitest) |
| `npm run lint` | oxlint |
| `npm run api:types` | regenerate `src/api/schema.d.ts` from `logs/openapi.json` |

## Your own cell

Start the server on your cell's profile, `python -m api --profile <your cell>`, and open
http://127.0.0.1:8000. Work in this order:

1. **Preflight**: every blocking row with its fix. A blocking row does not refuse the connect; it
   makes the later motions fail, so the banner says the cell is not runnable as configured.
2. **Config**: the values a person measures at the bench (payload, tool frame, camera serials,
   controller address). Everything else stays in YAML.
3. **Cell**: build, read what connecting will move, connect, watch the live status.
4. **Pick**: a typed or spoken prompt, the run's events as they happen, and Stop.
5. **History**: the runs of this session, the logged attempts, the KPIs, and CSV downloads.

The runbooks [bringing up a cell](../docs/runbooks/cell_bringup.md) and
[the first pick on a physical arm](../docs/runbooks/real_cell_first_pick.md) are the procedure around
these screens.

> [!WARNING]
> Two buttons move hardware, and only these two are filled red: Connect (a gripper may sweep or
> release on connect) and Pick. Stop does not stop the arm. It declines the next attempt, and the
> motion in flight completes. The stop for a moving arm is the physical button at the cell.

## Two pages

| Page | What it is | Who opens it |
|---|---|---|
| `index.html`, the console | five screens: Preflight, Cell, Pick, History, Config | whoever brings the cell up |
| `demo.html`, the room view | one prompt, one large narration, the live counts, no navigation | a projector |

They are two separate bundler entries, not two routes of one app. The room view drives a cell that is
already connected and reads its KPIs; it cannot build a cell, connect one or write config, so a page
shown to an audience is never one click from bringing a robot up.

## What the screens promise

| The screen | Because | Test |
|---|---|---|
| A simulated arm says simulated, on both pages | a simulated reading looks exactly like a physical one; only `simulated` tells them apart | yes |
| A missing measurement says "not offered by this driver", never zero | dummy, sim and KUKA offer no force and torque reading, and a zero reads as a measurement | yes |
| A blank controller state says "not asked" | mode and safety cost a dashboard round trip and are opt-in; blank must not read as fine | yes |
| An unmeasurable KPI is listed as unmeasurable | hidden reads as fine and zero reads as bad; the records cannot say either | yes |
| Stop says it does not stop the arm | a button labelled Stop is what someone reaches for instead of the physical button | yes |
| Preflight fixes are shown word for word, per row | a checklist that says wrong without saying what to do has only moved the guessing | yes |
| A lost event is shown as a gap | the server sends a `gap` frame when a client slept past its buffer; the screen draws it | no |
| The camera view names its picture: `live`, `rehearsal`, `overlay` or `held` | the four look alike; the sentence under the picture is the server's | no |
| A held frame is dimmed, not blanked | while a pick owns the camera the last image stays, greyed; blank reads as a fault | no |
| Zero counts are drawn | `0 blocking` is the answer an operator came for | no |

"yes" means `src/screens/screens.test.tsx` asserts it; the rows marked "no" hold in the code and have
no test yet.

## Speech

The microphone button fills the prompt box and nothing else; the operator still presses Pick. While
the text is what the machine heard, the box says so, and it stops saying so once the operator edits
it. Push to talk records while the button is held; a click alone records nothing. A foot switch that
sends a key works the same way: the talk key is F8 unless the `talkKey` prop or
`localStorage['willy.talkKey']` names another, and a key that repeats while held counts once.

The page encodes 16-bit PCM WAV itself (`src/prompt/recordWav.ts`), because no browser records WAV and
WAV is the only format the server decodes; anything else is refused with 415. The browser offers a
microphone only over HTTPS or on localhost. A console opened as `http://<cell PC>:8000` from another
machine has none, and the button is disabled with that sentence.

## Where the types come from

Nothing describes a payload by hand. The server's Pydantic schemas produce its OpenAPI document,
`npm run api:types` turns that into `src/api/schema.d.ts` (generated, do not edit), and
`src/api/client.ts` names those types for the screens. After any schema change on the server, from the
repository root:

```bash
python -c "import json; from api.app import create_app; print(json.dumps(create_app().openapi(), indent=2))" > logs/openapi.json
cd frontend && npm run api:types
```

## How the pages are built

- **No server address anywhere.** In production the pages come from the server itself; in development
  the Vite proxy makes them same-origin too. No build and no bookmark can aim the console at a
  different cell than the one whose server opened it.
- **The server writes every sentence.** An event row shows the server's `human` text word for word, so
  there is one account of what happened.
- **One file of design tokens.** `src/styles.css` is a Tailwind v4 theme: the palette is declared under
  `@theme inline` over runtime custom properties, because one status has one colour.
- **Two reds, apart by colour and by form.** `--alarm` is only ever a tinted surface with a border,
  for every `block` status; `--arm` is only ever a filled button with white text, for Connect and Pick.
  Colour never carries the text: a pill's word stays in the foreground colour in both themes.
- **Contrast is a test.** [`tests/test_console_contrast.py`](../tests/test_console_contrast.py) reads the
  numbers out of `styles.css` and checks WCAG AA for every text tier on every surface it can land on.
- **Three theme states.** `src/lib/theme.ts` cycles system, light and dark. System sets no attribute
  and follows the operating system live; light and dark set `<html data-theme>`.
- **Outcome strings in one place.** `src/lib/outcome.ts` maps an outcome to a pill tone. The server
  spells a success `succeeded`, the literal the KPI roll-up counts.

## Files

| File | Holds |
|---|---|
| `src/styles.css` | the design tokens, both themes, every shared class |
| `src/App.tsx`, `src/main.tsx` | the console shell and its entry; the sidebar always shows vendor, profile and state |
| `src/api/client.ts` | the only code that talks to the server, typed from `schema.d.ts` |
| `src/api/events.ts` | the run WebSocket: resume from `since_seq`, show gaps, back off |
| `src/api/schema.d.ts` | generated from the OpenAPI document; do not edit |
| `src/lib/useAsync.ts` | loading, error and stale states, and the poll helper |
| `src/lib/theme.ts` | system, light or dark, remembered per browser |
| `src/lib/outcome.ts` | outcome string to pill tone |
| `src/lib/provenance.ts` | the label `simulated arm`, `simulator` (URSim) or `controller`; never "physical arm" |
| `src/components/ui.tsx` | status pills, panels, the error banner |
| `src/components/ProvenanceBadge.tsx` | the badge that renders `provenance.ts` |
| `src/components/Viewfinder.tsx` | polls `/v1/camera` and badges what the picture is |
| `src/prompt/` | the shared prompt box, typed or spoken, the WAV encoder, and their tests |
| `src/screens/` | Preflight, Cell, Pick, History, Config, and `screens.test.tsx` |
| `src/demo/` | the room view: its own entry and a layout stylesheet over the same tokens |

`api/static/` and `node_modules/` are not committed; `npm run build` regenerates the bundle.

## What is proven and what is not

| Capability | Evidence |
|---|---|
| The screens render captured payloads and keep the rules marked yes above | `npm test` (Vitest) |
| The server serves the built pages, and an unknown `/v1` path stays a JSON 404 | [`tests/test_api_serves_the_console.py`](../tests/test_api_serves_the_console.py) |
| Preflight, build, connect, live status, a pick, history, disconnect | measured against real controller software (URSim, UR3e) |
| Any screen with a physical arm, gripper or camera | never touched hardware |

## Where the details live

- [`api/README.md`](../api/README.md): every endpoint these pages call, and what each refuses.
- [`scripts/ursim/README.md`](../scripts/ursim/README.md): the controller software the console was measured against.
