/**
 * Every screen renders against a payload the backend actually produced.
 *
 * The fixtures below are not invented: they are trimmed captures from `console_dummy`, the
 * hardware-free profile, taken over real HTTP. That matters because the failure this file exists to
 * catch is the one nothing else can -- a screen that reads `preview.warnings.map(...)` and meets a
 * payload where the field is absent, which type-checks (the field is optional) and then throws in
 * front of an operator.
 *
 * The second thing asserted here is the console's honesty rules, as text. A simulated arm must SAY
 * simulated; an unmeasurable KPI must SAY unmeasurable; Stop must say it does not stop the arm. Those
 * are the sentences a reviewer would quietly "clean up", so they are pinned.
 */

import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { RunOut } from '../api/client'
import { provenanceOf } from '../lib/provenance'
import { EMPTY_RUN, reduce, type RunView } from '../model/runModel'
import onceJson from '../test/fixtures/console_dummy_task_once.json'
import haltedJson from '../test/fixtures/task_halted_then_restart.json'
import twoPartsJson from '../test/fixtures/task_two_parts_nothing_left.json'
import { eventsOf, fakeSockets, renderWith, runOf, stubApi as serveApi, type ApiCall, type Fixture } from '../test/render'
import Cell from './Cell'
import Config from './Config'
import History from './History'
import { statsOf } from './historyStats'
import Preflight from './Preflight'
import Settings from './Settings'

const PREFLIGHT = {
  ok: false,
  n_blocking: 1,
  n_warn: 2,
  n_bench: 1,
  vendor: 'ur',
  profile: 'ur3e',
  checks: [
    { name: 'payload', status: 'ok', detail: 'Declared as 1.5 kg.', fix: '' },
    {
      name: 'calibration',
      status: 'block',
      detail: 'No CAMERA->BASE resolver for this cell.',
      fix: 'Run the eye-to-hand routine and point robot.calibration.path at the artifact.',
    },
    { name: 'tcp', status: 'warn', detail: 'Tool frame declared but never measured.', fix: 'Measure it.' },
    { name: 'estop', status: 'bench', detail: 'Only a person at the cell can confirm this.', fix: '' },
  ],
}

const CELL_CONNECTED = {
  state: 'connected',
  arm: 'DummyRobotArm',
  gripper: 'DummyGripper',
  vendor: 'dummy',
  profile: 'console_dummy',
  gripper_substitution: null,
  lock_holder: null,
  active_run_id: null,
}

const TELEMETRY = {
  state: 'connected',
  connected: true,
  simulated: true,
  vendor: 'dummy',
  model: '',
  tcp_position_mm: [400.0, 0.0, 300.0],
  // Deliberately absent, as the dummy driver leaves them: quaternion, joints, force, torque.
  controller_state_included: false,
}

const ROLLUP = {
  total_attempts: 12,
  kpis: { success_rate: 0.75 },
  unmeasurable: { cycle_time_s: 'no record carries a duration' },
  outcomes: { succeeded: 9, no_valid_grasp: 3 },
  source: 'logs/console/grasp_records.jsonl',
  record_log_path: 'D:/dev/aurora_backend/logs/console/grasp_records.jsonl',
  record_log_exists: true,
}

/** A `ConnectPreviewOut` with `warnings` and `blocking` ABSENT -- the shape that used to throw. */
const PREVIEW_SPARSE = {
  token: 'tok',
  expires_at: '2026-08-19T18:00:00Z',
  arm: 'DummyRobotArm',
  gripper: 'DummyGripper',
}

function stubApi(routes: Record<string, unknown>) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: string) => {
      const url = String(input).split('?')[0]
      const body = routes[url]
      if (body === undefined) {
        return new Response(JSON.stringify({ code: 'http_404', message: 'Not Found', detail: {} }), {
          status: 404,
          headers: { 'content-type': 'application/json' },
        })
      }
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    }),
  )
}

function draw(ui: React.ReactElement) {
  return render(<MemoryRouter>{ui}</MemoryRouter>)
}

beforeEach(() => {
  vi.stubGlobal('WebSocket', class { close() {} } as unknown as typeof WebSocket)
})
afterEach(() => {
  // Explicit, because `globals: false` means testing-library cannot register its own auto-cleanup:
  // without this every screen from every earlier test is still in the document, and a query that
  // should find one element finds four.
  cleanup()
  vi.unstubAllGlobals()
})

describe('Preflight', () => {
  it('shows a blocking row with its fix, and says connect will be refused', async () => {
    stubApi({ '/v1/preflight': PREFLIGHT })
    draw(<Preflight />)
    // NOT "connect will be refused" -- measured 2026-08-20 against a real UR controller, a cell with
    // a blocking checklist item CONNECTED and then refused every motion. The connect refusal lives in
    // the connect preview's own `blocking` list, on the Cell screen.
    await waitFor(() => expect(screen.getByText(/not runnable as configured/)).toBeTruthy())
    expect(screen.getByText(/No CAMERA->BASE resolver/)).toBeTruthy()
    // The fix is rendered verbatim, per row -- not summarised, not linked to a manual.
    expect(screen.getByText(/Run the eye-to-hand routine/)).toBeTruthy()
  })

  it('names the profile chain, because the same tree gives a different verdict under another', async () => {
    stubApi({ '/v1/preflight': PREFLIGHT })
    draw(<Preflight />)
    await waitFor(() => expect(screen.getByText('ur3e')).toBeTruthy())
  })

  it('does not soften a bench row into a pass', async () => {
    stubApi({ '/v1/preflight': PREFLIGHT })
    draw(<Preflight />)
    await waitFor(() => expect(screen.getByText(/only a person standing at the cell/i, { selector: 'div' })).toBeTruthy())
  })
})

describe('Cell', () => {
  it('renders a connect preview whose optional lists are absent', async () => {
    // The regression this file was written for.
    stubApi({ '/v1/cell': CELL_CONNECTED, '/v1/cell/status': TELEMETRY, '/v1/cell/connect-preview': PREVIEW_SPARSE })
    draw(<Cell />)
    await waitFor(() => expect(screen.getByText('DummyRobotArm')).toBeTruthy())
  })

  it('says SIMULATED when the arm is simulated, and says why the numbers are not real', async () => {
    stubApi({ '/v1/cell': CELL_CONNECTED, '/v1/cell/status': { ...TELEMETRY, simulated: true } })
    draw(<Cell />)
    await waitFor(() => expect(screen.getByText(/simulated arm/)).toBeTruthy())
    expect(screen.getByText(/No controller is involved/)).toBeTruthy()
  })

  it('renders a missing measurement as "not offered", never as a zero', async () => {
    stubApi({ '/v1/cell': CELL_CONNECTED, '/v1/cell/status': TELEMETRY })
    draw(<Cell />)
    await waitFor(() => expect(screen.getAllByText(/not offered by this driver/).length).toBeGreaterThan(0))
  })

  it('says a blank controller state means "not asked", not "fine"', async () => {
    stubApi({ '/v1/cell': CELL_CONNECTED, '/v1/cell/status': TELEMETRY })
    draw(<Cell />)
    await waitFor(() => expect(screen.getByText(/a blank here means/i)).toBeTruthy())
  })
})

describe('History', () => {
  it('shows where the numbers came from, above the numbers', async () => {
    stubApi({ '/v1/history/kpis': ROLLUP, '/v1/history/records': [], '/v1/runs': [] })
    draw(<History />)
    // Twice on purpose: the rollup names both its `source` and the file it read.
    await waitFor(() => expect(screen.getAllByText(/grasp_records\.jsonl/).length).toBe(2))
  })

  it('lists an unmeasurable KPI instead of hiding or zeroing it', async () => {
    stubApi({ '/v1/history/kpis': ROLLUP, '/v1/history/records': [], '/v1/runs': [] })
    draw(<History />)
    await waitFor(() => expect(screen.getByText(/no record carries a duration/)).toBeTruthy())
    expect(screen.getByText(/Not measurable from these records/)).toBeTruthy()
  })
})

describe('Config', () => {
  it('offers only the keys the backend declares writable, each with how to measure it', async () => {
    stubApi({
      '/v1/config/writable': [
        { key: 'robot.safety.payload.mass_kg', label: 'Payload mass', measure: 'Weigh the tool on a scale.', unit: 'kg' },
      ],
    })
    draw(<Config />)
    await waitFor(() => expect(screen.getByText('Weigh the tool on a scale.')).toBeTruthy())
    expect(screen.getByText('robot.safety.payload.mass_kg')).toBeTruthy()
  })

  it('has no free-text key field for writing', async () => {
    stubApi({ '/v1/config/writable': [] })
    draw(<Config />)
    await waitFor(() => expect(screen.getByText(/no writable keys/)).toBeTruthy())
  })

  const WRITABLE = [
    {
      key: 'robot.safety.payload.mass_kg',
      label: 'Payload mass',
      measure: 'Weigh the whole assembly on a bench scale: gripper plus coupling/adapter plate, every cable and hose.',
      unit: 'kg',
    },
    { key: 'camera.cameras.rigs[*].serial_number', label: 'Camera serial', measure: 'rs-enumerate-devices -s.', unit: '' },
    { key: 'robot.ur.ip', label: 'UR controller address', measure: 'Read it off the pendant under Settings > Network.', unit: '' },
    { key: 'robot.kuka.controller_ip', label: 'KUKA controller address', measure: 'The controller\'s address on your network.', unit: '' },
    { key: 'robot.brand_new.key', label: 'A key this console has no words for', measure: 'Measure it.', unit: '' },
  ]

  it('names every key it knows in the reader\'s words, with one line on how to get the value, and folds the server\'s English in the demo view', async () => {
    serveApi({ '/v1/cell': { state: 'disconnected', vendor: 'ur' }, '/v1/config/writable': WRITABLE })
    fakeSockets()
    renderWith(<Config />)
    expect(await screen.findByText('Nutzlast: Masse')).toBeTruthy()
    expect(screen.getByText(/Die ganze Baugruppe am Handgelenk wiegen/)).toBeTruthy()
    expect(screen.getByText('Seriennummer einer Kamera')).toBeTruthy()
    expect(screen.queryByText('Payload mass')).toBeNull()
    // The server's own sentence is a detail: folded behind "Details (Server)" in the demo view.
    const said = screen.getByText(/Weigh the whole assembly/)
    const folded = said.closest('details')
    expect(folded).not.toBeNull()
    expect(within(folded!).getByText('Details (Server)')).toBeTruthy()
    // A key the console has no words for keeps the server's, rather than disappearing.
    expect(screen.getByText('A key this console has no words for')).toBeTruthy()
  })

  it('offers a controller address only for the vendor this cell is', async () => {
    serveApi({ '/v1/cell': { state: 'disconnected', vendor: 'ur' }, '/v1/config/writable': WRITABLE })
    fakeSockets()
    renderWith(<Config />)
    expect(await screen.findByText('Adresse der UR-Steuerung')).toBeTruthy()
    await waitFor(() => expect(screen.queryByText(/KUKA/)).toBeNull())
    expect(screen.queryByText('robot.kuka.controller_ip')).toBeNull()
  })

  it('shows the server\'s sentence open in the tech view', async () => {
    localStorage.setItem('willy.view', 'tech')
    try {
      serveApi({ '/v1/cell': { state: 'disconnected', vendor: 'ur' }, '/v1/config/writable': WRITABLE })
      fakeSockets()
      renderWith(<Config />)
      const said = await screen.findByText(/Weigh the whole assembly/)
      expect(said.closest('details')).toBeNull()
    } finally {
      localStorage.removeItem('willy.view')
    }
  })
})

describe('Provenance — what is on the other end', () => {
  it('a simulated DRIVER says simulated arm', () => {
    const p = provenanceOf({ ...TELEMETRY, simulated: true } as never)
    expect(p.kind).toBe('sim-driver')
    expect(p.label).toContain('simulated arm')
  })

  it('a provably simulated CONTROLLER (URSim) says simulator, with its evidence', () => {
    // Measured 2026-08-19: URSim reports `simulated: false` -- it IS real UR controller software --
    // and a panel that stops there renders a Docker container as a robot.
    const p = provenanceOf({
      ...TELEMETRY, simulated: false, vendor: 'ur', model: 'ur3e',
      controller_is_simulator: true, controller_serial: '20195399999', controller_host: '127.0.0.1',
    } as never)
    expect(p.kind).toBe('sim-controller')
    expect(p.label).toContain('simulator')
    expect(p.detail).toContain('20195399999')
    expect(p.detail).toContain('does not exist')
  })

  it('an unproven controller is NEVER called physical', () => {
    // The discipline. `controller_is_simulator: null` means "cannot tell", and the console must not
    // convert that into the one fact nobody measured.
    const p = provenanceOf({
      ...TELEMETRY, simulated: false, vendor: 'ur', model: 'ur5e', controller_is_simulator: null,
    } as never)
    expect(p.kind).toBe('controller')
    expect(p.label).not.toContain('physical')
    expect(p.detail).toContain('cannot prove')
  })

  it('a disconnected cell claims nothing at all', () => {
    expect(provenanceOf({ ...TELEMETRY, connected: false } as never).kind).toBe('unknown')
    expect(provenanceOf(null).kind).toBe('unknown')
  })
})

// ── History: the charts, the runs table, what cannot be measured ───────────────────────────────────────────────────

/** A column's accessible name, its spaces as a reader hears them (Intl writes a no-break space before "%"). */
function label(bar: HTMLElement): string {
  return (bar.getAttribute('aria-label') ?? '').replace(/\s+/g, ' ')
}

const TWO_PARTS = twoPartsJson as unknown as Fixture
const HALTED = haltedJson as unknown as Fixture
const ONCE = onceJson as unknown as Fixture

/** A task's success rate as the cockpit's own model counts it, from the run's record and its events. */
function cockpitRate(fixture: Fixture, runId: string): number | null {
  const start = reduce(EMPTY_RUN, { type: 'run.snapshot', run: fixture.runs[runId] })
  return eventsOf(fixture, runId).reduce((view, event) => reduce(view, event), start).stats.successRate
}

/**
 * The session's runs as the server lists them, newest first: at most `limit` of them, each route with its own default
 * (50, `api/routers/history.py` and `api/routers/pick.py`), and never more than the registry keeps (200).
 */
function listed(runs: readonly RunOut[]) {
  return (call: ApiCall) => runs.slice(0, Math.min(Number(new URLSearchParams(call.query).get('limit') ?? 50), 200))
}

function historyRoutes(runs: RunOut[]) {
  return serveApi({
    '/v1/cell': { state: 'connected', vendor: 'ur', active_run_id: null, needs_person: '', payload_model: 'none', countdown_due: false, jaws_question: false },
    '/v1/cell/readiness': { ready: true, lights: [], blockers: [] },
    '/v1/cell/facts': { wrist_camera: true, hand_eye_warn_mm: 6, route: { route: 'planned', sentence: '' }, rehearsal: false },
    '/v1/history/kpis': ROLLUP,
    '/v1/history/records': [],
    '/v1/history/runs': listed(runs),
    '/v1/runs': listed(runs),
  })
}

describe('History, the session', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it("charts the success per task and the median time per part from each task's own events", async () => {
    const two = runOf(TWO_PARTS)
    const halted = runOf(HALTED, (run) => run.stop_code === 'halted')
    historyRoutes([TWO_PARTS.runs[two], HALTED.runs[halted]])
    const sockets = fakeSockets()
    renderWith(<History />)
    await waitFor(() => expect(sockets.find(two)).toBeTruthy())
    await waitFor(() => expect(sockets.find(halted)).toBeTruthy())
    act(() => {
      sockets.find(two)!.deliver(eventsOf(TWO_PARTS, two))
      sockets.find(halted)!.deliver(eventsOf(HALTED, halted))
    })
    const rate = await screen.findByRole('figure', { name: 'Erfolg je Auftrag' })
    // Two parts placed of four picks, two of them empty looks: 100 %. The halted task is said exactly as the cockpit's
    // own model counts it (a pick a person halted is no failed grasp where the model says so): History is never a
    // second formula. Oldest first, left to right, as a time axis reads.
    const halt = cockpitRate(HALTED, halted)
    await waitFor(() =>
      expect(within(rate).getAllByRole('img').map((bar) => label(bar))).toEqual([
        expect.stringMatching(halt === null ? /kein Wert$/ : new RegExp(`${Math.round(halt * 100)} %$`)),
        expect.stringMatching(/100 %$/),
      ]),
    )
    const time = screen.getByRole('figure', { name: 'Median-Zeit pro Teil' })
    // The median of 21.4 s and 27.9 s; a task that placed nothing has no time per part, said in words, never a zero.
    const labels = within(time).getAllByRole('img').map((bar) => label(bar))
    expect(labels[0]).toMatch(/kein Wert$/)
    expect(labels[1]).toMatch(/24,[67] s$/)
  })

  it('says one pick as one in the tooltip ("1 von 1 Griff"), never "1 von 1 Griffen"', async () => {
    const once = runOf(ONCE)
    historyRoutes([ONCE.runs[once]])
    const sockets = fakeSockets()
    renderWith(<History />)
    await waitFor(() => expect(sockets.find(once)).toBeTruthy())
    act(() => sockets.find(once)!.deliver(eventsOf(ONCE, once)))
    const rate = await screen.findByRole('figure', { name: 'Erfolg je Auftrag' })
    await waitFor(() => expect(label(within(rate).getAllByRole('img')[0])).toMatch(/100 %$/))
    fireEvent.focus(within(rate).getAllByRole('img')[0])
    expect((await within(rate).findByText(/von 1 Griff/)).textContent).toMatch(/1 von 1 Griff$/)
  })

  it('counts the picks the cockpit counts: empty looks aside, and picks a person cut short where the model says so', () => {
    // The model's own count of the picks its rate is of (`rated`) is the one History sums: a part a person halted in
    // the jaws is left out there too, so the session's rate never disagrees with the bars or the cockpit.
    const view = { ...EMPTY_RUN, kind: 'task', stats: { ...EMPTY_RUN.stats, picks: 4, emptyLooks: 1, cut: 1, placed: 2, rated: 2 } }
    expect(statsOf('run-x', view as unknown as RunView, true).fairPicks).toBe(2)
    const halted = { ...EMPTY_RUN, kind: 'task', stats: { ...EMPTY_RUN.stats, picks: 2, emptyLooks: 0, cut: 0, placed: 1, rated: 1 } }
    expect(statsOf('run-h', halted as unknown as RunView, true).fairPicks).toBe(1)
    // A model that keeps no such counts leaves the cut picks and the empty looks out, or only the empty looks.
    const older = { ...EMPTY_RUN, kind: 'task', stats: { ...EMPTY_RUN.stats, picks: 4, emptyLooks: 1, cut: 1, placed: 2, rated: undefined } }
    expect(statsOf('run-o', older as unknown as RunView, true).fairPicks).toBe(2)
    const plain = { ...EMPTY_RUN, kind: 'task', stats: { ...EMPTY_RUN.stats, picks: 4, emptyLooks: 1, placed: 2, rated: undefined } }
    expect(statsOf('run-y', { ...plain, stats: { ...plain.stats, cut: undefined } } as unknown as RunView, true).fairPicks).toBe(3)
  })

  it('lists the runs with their kind, the command as it was said, and how each ended', async () => {
    const two = runOf(TWO_PARTS)
    historyRoutes([TWO_PARTS.runs[two]])
    fakeSockets()
    renderWith(<History />)
    const table = await screen.findByRole('table', { name: 'Läufe dieser Sitzung' })
    const row = within(table).getAllByRole('row')[1]
    expect(row.textContent).toMatch(/Auftrag/)
    expect(row.textContent).toMatch(/Räum alle grünen Würfel auf die Ablage links/)
    expect(within(row).getByText('gesprochen')).toBeTruthy()
    expect(within(row).getByText('Nichts mehr da')).toBeTruthy()
    // Placed of gripped: the two empty looks that ended "until empty" are no part of either number.
    expect(within(table).getByRole('columnheader', { name: 'abgelegt / gegriffen' })).toBeTruthy()
    expect(row.textContent).toMatch(/2 \/ 2/)
    expect(row.textContent).not.toMatch(/2 \/ 4/)
  })

  it("keeps each unmeasurable KPI's name on its own line, apart from its reason", async () => {
    historyRoutes([])
    fakeSockets()
    renderWith(<History />)
    const reason = await screen.findByText(/no record carries a duration/)
    const dd = reason.closest('dd') as HTMLElement
    expect(dd).toBeTruthy()
    expect(dd.previousElementSibling?.tagName).toBe('DT')
    expect(dd.previousElementSibling?.textContent).toBe('cycle_time_s')
  })

  it('names an unmeasurable KPI in German and folds the server\'s English reason in the demo view', async () => {
    serveApi({
      '/v1/cell': { state: 'connected', vendor: 'ur', active_run_id: null },
      '/v1/history/kpis': {
        ...ROLLUP,
        kpis: { pick_success_rate: 0.985 },
        unmeasurable: { median_cycle_time_s: 'Not measurable from these records: no record carries a cycle time.' },
      },
      '/v1/history/records': [],
      '/v1/runs': [],
    })
    fakeSockets()
    renderWith(<History />)
    const reason = await screen.findByText(/no record carries a cycle time/)
    expect(reason.closest('details')?.open).toBe(false)
    expect((reason.closest('dd') as HTMLElement).previousElementSibling?.textContent).toBe('Median-Zykluszeit (s)')
    // A rate reads as the percentage the tiles beside it use, never as a bare fraction.
    const row = screen.getByText('Greiferfolg').closest('tr') as HTMLElement
    expect(row.textContent).toMatch(/99\s%/)
    expect(row.textContent).not.toMatch(/0,985/)
  })

  it('names the source file once when it is the file the numbers were read from', async () => {
    const path = 'D:\\dev\\Workaholic-Willy\\logs\\console\\grasp_records.jsonl'
    serveApi({
      '/v1/cell': { state: 'connected', vendor: 'ur', active_run_id: null },
      '/v1/history/kpis': { ...ROLLUP, source: path, record_log_path: path },
      '/v1/history/records': [],
      '/v1/runs': [],
    })
    fakeSockets()
    renderWith(<History />)
    await screen.findByText(/no record carries a duration/)
    expect(screen.getAllByText(/grasp_records\.jsonl/)).toHaveLength(1)
  })

  it('says where a Home run went and which pose a teach saved, and the time per part by its name', async () => {
    const home = { ...TWO_PARTS.runs[runOf(TWO_PARTS)], id: 'run-home', kind: 'home', plan: null, prompt: '', stop_code: 'finished', stop_class: 'done' }
    const teach = { ...home, id: 'run-teach', kind: 'teach', stop_code: 'taught' }
    historyRoutes([home as RunOut, teach as RunOut])
    const sockets = fakeSockets()
    renderWith(<History />)
    await waitFor(() => expect(sockets.find('run-home')).toBeTruthy())
    await waitFor(() => expect(sockets.find('run-teach')).toBeTruthy())
    const started = (runId: string, data: Record<string, unknown>) => ({
      type: 'run_started', run_id: runId, seq: 1, ts: 1, severity: 'info', human: 'started', step: '', step_index: null, step_total: null, data,
    })
    act(() => {
      sockets.find('run-home')!.deliver([started('run-home', { kind: 'home', to: 'park', label: 'Parkposition' })])
      sockets.find('run-teach')!.deliver([started('run-teach', { kind: 'teach', name: 'drop_left', label: 'Ablage links', role: 'place' })])
    })
    const table = await screen.findByRole('table', { name: 'Läufe dieser Sitzung' })
    await waitFor(() => expect(table.textContent).toMatch(/Fahrt nach Parkposition/))
    expect(table.textContent).toMatch(/Pose „Ablage links“/)
    expect(screen.getByText('Zeit pro Teil')).toBeTruthy()
  })

  it('counts every run the server keeps, not only the newest fifty, and lists the newest fifty first', async () => {
    const base = TWO_PARTS.runs[runOf(TWO_PARTS)]
    const runs = Array.from({ length: 60 }, (_, n) => ({ ...base, id: `run-${n}`, started_at: base.started_at - n * 60 }))
    const calls = historyRoutes(runs as RunOut[])
    fakeSockets()
    renderWith(<History />)
    const tiles = await screen.findByRole('region', { name: 'Diese Sitzung' })
    const tasks = within(tiles).getByText('Aufträge').closest('.hs-tile') as HTMLElement
    await waitFor(() => expect(tasks.textContent).toMatch(/60/))
    const table = screen.getByRole('table', { name: 'Läufe dieser Sitzung' })
    expect(within(table).getAllByRole('row')).toHaveLength(51)
    fireEvent.click(screen.getByRole('button', { name: 'Alle 60 zeigen' }))
    expect(within(table).getAllByRole('row')).toHaveLength(61)
    // The whole window the server keeps, read in one request.
    expect(calls.filter((c) => c.path === '/v1/runs').map((c) => c.query)).toContain('limit=200')
  })

  it('says the success rate and the time per part are of the newest twelve tasks, where the session ran more', async () => {
    const base = TWO_PARTS.runs[runOf(TWO_PARTS)]
    const runs = Array.from({ length: 13 }, (_, n) => ({ ...base, id: `run-${n}`, started_at: base.started_at - n * 60 }))
    historyRoutes(runs as RunOut[])
    fakeSockets()
    renderWith(<History />)
    const tiles = await screen.findByRole('region', { name: 'Diese Sitzung' })
    expect(within(tiles).getAllByText(/die letzten 12 Aufträge/)).toHaveLength(2)
  })
})

// ── Settings: the preferences of this browser ────────────────────────────────────────────────────────────────────

describe('Settings', () => {
  beforeEach(() => {
    localStorage.clear()
    fakeSockets()
    serveApi({
      '/v1/config/writable': [],
      '/v1/cell': { state: 'disconnected', vendor: 'ur', needs_person: '', payload_model: 'none', countdown_due: false, jaws_question: false },
    })
  })

  afterEach(() => {
    document.documentElement.removeAttribute('data-theme')
  })

  it('switches the theme, the view and the voice output for this browser, and the language', async () => {
    renderWith(<Settings />)
    fireEvent.click(within(await screen.findByRole('group', { name: 'Design' })).getByRole('button', { name: 'Hell' }))
    expect(localStorage.getItem('willy.theme')).toBe('light')
    fireEvent.click(within(screen.getByRole('group', { name: 'Ansicht' })).getByRole('button', { name: 'Technik' }))
    expect(localStorage.getItem('willy.view')).toBe('tech')
    fireEvent.click(within(screen.getByRole('group', { name: 'Sprachausgabe' })).getByRole('button', { name: 'An' }))
    expect(localStorage.getItem('willy.voiceOut')).toBe('on')
    fireEvent.click(within(screen.getByRole('group', { name: 'Sprache' })).getByRole('button', { name: 'EN' }))
    expect(await screen.findByRole('heading', { level: 2, name: 'Settings' })).toBeTruthy()
  })

  it('stores the talk key a foot switch sends, and refuses a letter', async () => {
    renderWith(<Settings />)
    fireEvent.click(await screen.findByRole('button', { name: 'Taste festlegen' }))
    fireEvent.keyDown(window, { key: 'a' })
    expect(await screen.findByText(/„a“ geht nicht/)).toBeTruthy()
    expect(localStorage.getItem('willy.talkKey')).toBeNull()
    fireEvent.keyDown(window, { key: 'F9' })
    await waitFor(() => expect(localStorage.getItem('willy.talkKey')).toBe('F9'))
    expect(screen.getByText('Jetzt: F9')).toBeTruthy()
  })

  it('refuses every key typing uses as the talk key (Enter after "Taste festlegen" is the easy accident), and keeps F8', async () => {
    renderWith(<Settings />)
    fireEvent.click(await screen.findByRole('button', { name: 'Taste festlegen' }))
    for (const key of ['Enter', 'Shift', 'Tab', 'Backspace', 'Delete', 'ArrowLeft', 'Control', 'Alt', 'Meta', 'CapsLock', 'Home', 'PageDown', 'Insert', 'ContextMenu', ' ']) {
      fireEvent.keyDown(window, { key })
      expect(localStorage.getItem('willy.talkKey'), key).toBeNull()
    }
    expect(screen.getByText(/„Leertaste“ geht nicht/)).toBeTruthy()
    expect(screen.getByText('Jetzt: F8')).toBeTruthy()
    // Still capturing: the next foot-switch key is taken.
    fireEvent.keyDown(window, { key: 'Pause' })
    await waitFor(() => expect(localStorage.getItem('willy.talkKey')).toBe('Pause'))
  })

  it('says which key was refused, by its name', async () => {
    renderWith(<Settings />)
    fireEvent.click(await screen.findByRole('button', { name: 'Taste festlegen' }))
    fireEvent.keyDown(window, { key: 'Enter' })
    expect(await screen.findByText(/„Enter“ geht nicht/)).toBeTruthy()
    expect(localStorage.getItem('willy.talkKey')).toBeNull()
  })
})

// ── Preflight in German: the check names in the reader's words, the server's sentences as details ──────────────────

const PREFLIGHT_DE = {
  ok: true,
  n_blocking: 0,
  n_warn: 1,
  n_bench: 0,
  vendor: 'dummy',
  profile: '',
  checks: [
    { name: 'fixtures', status: 'warn', detail: 'none declared; there is no table in the collision world', fix: 'declare the bench' },
    { name: 'camera -> base', status: 'ok', detail: 'primary: eye_in_hand from calib.json', fix: '' },
    { name: 'brand new check', status: 'ok', detail: 'something new', fix: '' },
  ],
}

describe('Preflight, as the cell PC shows it', () => {
  beforeEach(() => {
    localStorage.clear()
    fakeSockets()
  })

  it('names every check in German and folds the server\'s English sentences behind "Details" in the demo view', async () => {
    serveApi({ '/v1/preflight': PREFLIGHT_DE, '/v1/cell': { state: 'disconnected', vendor: 'dummy' } })
    renderWith(<Preflight />)
    expect(await screen.findByText('Feste Hindernisse im Kollisionsmodell')).toBeTruthy()
    expect(screen.getByText('Kamera → Basis (Kalibrierung)')).toBeTruthy()
    // A check this console does not know yet keeps the server's name.
    expect(screen.getByText('brand new check')).toBeTruthy()
    const detail = screen.getByText(/there is no table in the collision world/)
    expect(detail.closest('details')?.open).toBe(false)
    expect(screen.getByText('declare the bench').closest('details')?.open).toBe(false)
    // The profile chain, said in German when there is none.
    expect(screen.getByText('(keins)')).toBeTruthy()
  })

  it('shows the server\'s sentences open in the tech view, with the check\'s own name', async () => {
    localStorage.setItem('willy.view', 'tech')
    serveApi({ '/v1/preflight': PREFLIGHT_DE, '/v1/cell': { state: 'disconnected', vendor: 'dummy' } })
    renderWith(<Preflight />)
    const detail = await screen.findByText(/there is no table in the collision world/)
    expect(detail.closest('details')).toBeNull()
    expect(screen.getByText('fixtures')).toBeTruthy()
  })
})
