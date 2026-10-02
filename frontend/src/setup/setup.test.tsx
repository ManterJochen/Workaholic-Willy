/**
 * Setup (build plan 4.3): the stepper Check -> Build -> Preview -> Connect -> Ready, the poses taught by freedrive, and
 * the cell's facts.
 *
 * What is pinned here is what keeps a person safe while the cell comes up: building touches no robot and is real by
 * default where the arm is real; Connect is the one red button, it says the robot moves, and it carries the token of
 * the preview a person read; a latched arm is cleared only by a person's word, after they ticked that they looked; a
 * pose is taught only after the person confirmed the payload, sees where it is written and clicks the button that frees
 * the arm; the session's poll is its heartbeat (four a second), a closing tab sends the cancel beacon, and the dialog
 * says the arm is held only once it stands still.
 */

import { act, cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { fakeSockets, refusal, renderWith, reply, stubApi, type ApiCall, type Route } from '../test/render'
import CellFacts from './CellFacts'
import PosesPanel from './PosesPanel'
import Setup from './Setup'
import { currentStep, stepMarks } from './steps'
import TeachDialog from './TeachDialog'

const HAND_TOGGLE = {
  kind: 'toggle',
  driver: 'JawIOGripper',
  where: 'tool output 0',
  connected: true,
  jaws: 'open',
  why_unknown: '',
  no_sensor: true,
  commands_sent: 2,
}

function cellOf(over: Record<string, unknown> = {}) {
  return {
    state: 'disconnected',
    arm: null,
    gripper: null,
    vendor: 'ur',
    profile: 'ur10,hande,cell',
    gripper_substitution: null,
    lock_holder: null,
    active_run_id: null,
    needs_person: '',
    halted: null,
    planner: { state: 'not_used' },
    payload_model: 'none',
    recovery: null,
    jaws_confirmed_at: null,
    countdown_due: false,
    jaws_question: false,
    hand: { ...HAND_TOGGLE, connected: false, jaws: 'unknown' },
    ...over,
  }
}

const PREFLIGHT = { ok: true, n_blocking: 0, n_warn: 0, n_bench: 0, vendor: 'ur', profile: 'ur10,hande,cell', checks: [] }

const LIGHTS_READY = [
  { id: 'robot', state: 'ok', code: 'connected', message: 'connected', blocks: true },
  { id: 'cameras', state: 'ok', code: 'live', message: 'live', blocks: true },
  { id: 'planner', state: 'ok', code: 'ready', message: 'ready', blocks: true },
  { id: 'gripper', state: 'ok', code: 'open_confirmed', message: 'open', blocks: true },
  { id: 'carried_part', state: 'ok', code: 'modelled', message: 'modelled', blocks: true },
  { id: 'commands', state: 'info', code: 'ready', message: 'ready', blocks: false },
]

const POSES = {
  home: { name: 'home', label: 'Home', joints_deg: [0, -90, 0, -90, 0, 0], source: 'config', screen: null, note: '', taught_at: null },
  poses: [
    {
      name: 'drop_left',
      label: 'Ablage links',
      joints_deg: [-60, -95, -120, -55, 90, 0],
      source: 'taught',
      screen: 'clear',
      note: '',
      taught_at: '2026-10-01T09:12:00+00:00',
    },
    { name: 'park', label: 'Parkposition', joints_deg: [30, -80, -100, -90, 90, 0], source: 'config', screen: 'band', note: 'band', taught_at: null },
  ],
  default_place: 'drop_left',
  teachable: true,
  why_not: '',
  why_not_code: '',
  target_file: 'D:/dev/Workaholic-Willy/config/robot/robot.cell.yaml',
}

function serve(routes: Record<string, Route>): ApiCall[] {
  return stubApi({
    '/v1/cell': cellOf(),
    '/v1/cell/readiness': refusal(501, 'not_built_yet', 'not built yet'),
    '/v1/cell/facts': refusal(501, 'not_built_yet', 'not built yet'),
    '/v1/preflight': PREFLIGHT,
    '/v1/poses': POSES,
    '/v1/camera/live': { rig_id: null, rigs: [], source: 'none', reason: 'not_built', width: 0, height: 0 },
    ...routes,
  })
}

function posted(calls: ApiCall[], path: string): ApiCall[] {
  return calls.filter((c) => c.method !== 'GET' && c.path === path)
}

/** The person read the checklist and goes on: the one click that leaves the check step of a fresh cell. */
async function pastTheCheck(): Promise<void> {
  fireEvent.click(await screen.findByRole('button', { name: 'Weiter: Aufbauen' }))
}

beforeEach(() => {
  localStorage.clear()
  sessionStorage.clear()
  fakeSockets()
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.useRealTimers()
  localStorage.clear()
  sessionStorage.clear()
})

describe('the steps', () => {
  it('follow the cell, and never mark a step the server has not reached', () => {
    const base = { preflightBlocking: 0, preflightWarnings: 0, checkSeen: true, previewRead: false, ready: null }
    expect(currentStep({ ...base, cellState: null })).toBe('check')
    expect(currentStep({ ...base, cellState: 'disconnected' })).toBe('build')
    expect(currentStep({ ...base, cellState: 'built' })).toBe('preview')
    expect(currentStep({ ...base, cellState: 'built', previewRead: true })).toBe('connect')
    expect(currentStep({ ...base, cellState: 'connected' })).toBe('ready')
    expect(stepMarks({ ...base, cellState: 'connected', ready: false }).ready).toBe('current')
    expect(stepMarks({ ...base, cellState: 'connected', ready: true }).ready).toBe('done')
    // A blocking checklist is something to read, not a gate: its step is a warning once passed.
    expect(stepMarks({ ...base, cellState: 'disconnected', preflightBlocking: 2 }).check).toBe('warn')
  })

  it('stay at the check until the person has read it, and mark it with a warning, never a tick, when it warns', () => {
    const fresh = { preflightBlocking: 0, preflightWarnings: 6, checkSeen: false, previewRead: false, ready: null }
    expect(currentStep({ ...fresh, cellState: 'disconnected' })).toBe('check')
    expect(stepMarks({ ...fresh, cellState: 'disconnected' }).check).toBe('current')
    expect(stepMarks({ ...fresh, cellState: 'disconnected' }).build).toBe('todo')
    expect(stepMarks({ ...fresh, cellState: 'disconnected', checkSeen: true }).check).toBe('warn')
    expect(stepMarks({ ...fresh, cellState: 'disconnected', checkSeen: true, preflightWarnings: 0 }).check).toBe('done')
    // A cell another session built is past the check: what the checklist says still marks it.
    expect(stepMarks({ ...fresh, cellState: 'built' }).check).toBe('warn')
  })
})

describe('Setup', () => {
  it('opens at the check with what it found, and goes on to Aufbauen only at the person\'s click', async () => {
    serve({
      '/v1/preflight': {
        ...PREFLIGHT,
        n_warn: 1,
        checks: [{ name: 'fixtures', status: 'warn', detail: 'none declared; there is no table in the collision world', fix: '' }],
      },
    })
    renderWith(<Setup />)
    const stepper = await screen.findByRole('list', { name: 'Schritte' })
    await waitFor(() => expect(within(stepper).getByText('Prüfen').closest('[aria-current="step"]')).toBeTruthy())
    expect(await screen.findByText(/Feste Hindernisse/)).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Zelle aufbauen' })).toBeNull()
    await pastTheCheck()
    await waitFor(() => expect(within(stepper).getByText('Aufbauen').closest('[aria-current="step"]')).toBeTruthy())
    expect(within(stepper).getByText('Prüfen').closest('li')?.className).toMatch(/\bwarn\b/)
    expect(screen.getByRole('button', { name: 'Zelle aufbauen' })).toBeTruthy()
  })

  it('builds a real cell by default where the arm is real, and touches no robot doing it', async () => {
    const calls = serve({ 'POST /v1/cell/build': cellOf({ state: 'built' }) })
    renderWith(<Setup />)
    await pastTheCheck()
    const rehearse = await screen.findByRole('checkbox', { name: /Probe/ })
    await waitFor(() => expect((rehearse as HTMLInputElement).checked).toBe(false))
    const build = screen.getByRole('button', { name: 'Zelle aufbauen' })
    expect(build.className).not.toMatch(/danger/)
    fireEvent.click(build)
    await waitFor(() => expect(posted(calls, '/v1/cell/build')).toHaveLength(1))
    expect(posted(calls, '/v1/cell/build')[0].query).toBe('rehearse=false')
  })

  it('rehearses by default on the dummy arm', async () => {
    const calls = serve({ '/v1/cell': cellOf({ vendor: 'dummy' }), 'POST /v1/cell/build': cellOf({ vendor: 'dummy', state: 'built' }) })
    renderWith(<Setup />)
    await pastTheCheck()
    const rehearse = await screen.findByRole('checkbox', { name: /Probe/ })
    await waitFor(() => expect((rehearse as HTMLInputElement).checked).toBe(true))
    fireEvent.click(screen.getByRole('button', { name: 'Zelle aufbauen' }))
    await waitFor(() => expect(posted(calls, '/v1/cell/build')[0]?.query).toBe('rehearse=true'))
  })

  it('reads the preview first, then connects with its token through the one red button that says the robot moves', async () => {
    const calls = serve({
      '/v1/cell': cellOf({ state: 'built', arm: 'URRobotArm', gripper: 'JawIOGripper' }),
      '/v1/cell/connect-preview': {
        token: 'tok-1',
        expires_at: '2026-10-01T18:00:00Z',
        arm: 'URRobotArm',
        gripper: 'JawIOGripper',
        warnings: [{ subject: 'gripper', what: 'the jaws may move', precaution: 'keep hands clear' }],
        blocking: [],
      },
      'POST /v1/cell/connect': cellOf({ state: 'connected' }),
    })
    renderWith(<Setup />)
    fireEvent.click(await screen.findByRole('button', { name: /Vorschau lesen/ }))
    expect(await screen.findByText('the jaws may move')).toBeTruthy()
    const connect = await screen.findByRole('button', { name: /der Roboter bewegt sich/ })
    expect(connect.className).toMatch(/danger/)
    expect(posted(calls, '/v1/cell/connect')).toHaveLength(0)
    // The token's expiry in the reader's own clock, never the server's UTC stamp; driver classes are the tech view's.
    const local = new Intl.DateTimeFormat('de-DE', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })
    expect(screen.getByText(local.format(new Date('2026-10-01T18:00:00Z')))).toBeTruthy()
    expect(screen.queryByText(/2026-10-01T18:00:00Z/)).toBeNull()
    expect(screen.queryByText('URRobotArm')).toBeNull()
    expect(screen.getByText('Greifer')).toBeTruthy()
    fireEvent.click(connect)
    await waitFor(() => expect(posted(calls, '/v1/cell/connect')).toHaveLength(1))
    expect(posted(calls, '/v1/cell/connect')[0].body).toEqual({ token: 'tok-1' })
  })

  it('clears a latched arm only on a person\'s word, after they ticked that they looked', async () => {
    const calls = serve({
      '/v1/cell': cellOf({
        state: 'built',
        arm: 'URRobotArm',
        halted: { reason: 'the cell was disconnected while a run was moving the arm', requested_at: 1, in_motion: true, braked: false, brake: 'ran_out' },
      }),
      'POST /v1/cell/acknowledge': cellOf({ state: 'built' }),
    })
    renderWith(<Setup />)
    const clear = await screen.findByRole('button', { name: 'Zelle ist frei' })
    expect((clear as HTMLButtonElement).disabled).toBe(true)
    fireEvent.click(screen.getByRole('checkbox', { name: /niemand ist im Arbeitsraum/ }))
    expect((clear as HTMLButtonElement).disabled).toBe(false)
    fireEvent.click(clear)
    await waitFor(() => expect(posted(calls, '/v1/cell/acknowledge')).toHaveLength(1))
    expect(posted(calls, '/v1/cell/acknowledge')[0].body).toEqual({ cell_clear: true, jaws_empty: false })
  })

  it('offers "Zelle ist frei" before Verbinden while a stop nobody has cleared stands (after a restart too)', async () => {
    // The jaws wait for it as every motion does: a toggle's "Jetzt öffnen" at Connect is refused until it is said.
    const stopped = { run_id: 'run-9', kind: 'task', stop_code: 'failed_in_a_row', at: 50, holding: true, cleared_at: null }
    const calls = serve({
      '/v1/cell': cellOf({ state: 'built', arm: 'URRobotArm', recovery: stopped }),
      'POST /v1/cell/acknowledge': cellOf({ state: 'built', recovery: { ...stopped, cleared_at: 51 } }),
    })
    renderWith(<Setup />)
    expect(await screen.findByText('Ein Stopp steht noch')).toBeTruthy()
    const clear = screen.getByRole('button', { name: 'Zelle ist frei' })
    expect((clear as HTMLButtonElement).disabled).toBe(true)
    fireEvent.click(screen.getByRole('checkbox', { name: /niemand ist im Arbeitsraum/ }))
    fireEvent.click(clear)
    await waitFor(() => expect(posted(calls, '/v1/cell/acknowledge')).toHaveLength(1))
    expect(posted(calls, '/v1/cell/acknowledge')[0].body).toEqual({ cell_clear: true, jaws_empty: false })
    cleanup()
    // Nothing built: the person's word needs a built cell, so the panel waits; once said, it is gone.
    serve({ '/v1/cell': cellOf({ state: 'disconnected', recovery: stopped }) })
    renderWith(<Setup />)
    await screen.findByRole('button', { name: 'Weiter: Aufbauen' })
    expect(screen.queryByText('Ein Stopp steht noch')).toBeNull()
    cleanup()
    serve({ '/v1/cell': cellOf({ state: 'built', arm: 'URRobotArm', recovery: { ...stopped, cleared_at: 51 } }) })
    renderWith(<Setup />)
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(screen.queryByText('Ein Stopp steht noch')).toBeNull()
  })

  it('shows the ready lights once connected, and starts a planner that is off (it moves nothing)', async () => {
    const lights = LIGHTS_READY.map((l) => (l.id === 'planner' ? { ...l, state: 'blocked', code: 'off' } : l))
    const calls = serve({
      '/v1/cell': cellOf({ state: 'connected', arm: 'URRobotArm', hand: HAND_TOGGLE, planner: { state: 'off' } }),
      '/v1/cell/readiness': { ready: false, lights, blockers: [] },
      'POST /v1/cell/planner': reply(202, { id: 'run-p', kind: 'planner', state: 'running', prompt: '', requested_picks: 0, started_at: 1, succeeded: 0, attempted: 0, error: '', stop_requested: false, stop_code: '', stop_class: '', parts_placed: 0, holding: false, halt_requested: false, step: '' }),
    })
    renderWith(<Setup />)
    await screen.findByText('Planer')
    expect(screen.getByText('Aus')).toBeTruthy()
    // A blocked light wears a tone of its own: `block` is also Tailwind's `display: block`, whose utilities layer beats
    // the row's grid and ran the light's name into its answer ("PlanerAus").
    const row = screen.getByText('Planer').closest('li') as HTMLElement
    expect(row.classList.contains('block')).toBe(false)
    expect(row.classList.contains('alarm')).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: 'Planer starten' }))
    await waitFor(() => expect(posted(calls, '/v1/cell/planner')).toHaveLength(1))
  })

  it('keeps the raw telemetry (TCP, quaternion, joints) for the tech view: the demo view reads the lights', async () => {
    const telemetry = {
      state: 'connected',
      connected: true,
      simulated: false,
      vendor: 'ur',
      model: 'ur10',
      tcp_position_mm: [400, 0, 300],
      tcp_quaternion_xyzw: [0, 1, 0, 0],
      joint_positions: [0, -1.57, 1.57, -1.57, -1.57, 0],
      controller_state_included: false,
    }
    const routes = {
      '/v1/cell': cellOf({ state: 'connected', arm: 'URRobotArm', hand: HAND_TOGGLE, planner: { state: 'ready' } }),
      '/v1/cell/readiness': { ready: true, lights: LIGHTS_READY, blockers: [] },
      '/v1/cell/status': telemetry,
    }
    serve(routes)
    renderWith(<Setup />)
    await screen.findByText('Planer')
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(screen.queryByText('Live-Telemetrie')).toBeNull()
    expect(screen.queryByText('Quat. xyzw')).toBeNull()
    cleanup()
    localStorage.setItem('willy.view', 'tech')
    serve(routes)
    renderWith(<Setup />)
    expect(await screen.findByText('Live-Telemetrie')).toBeTruthy()
  })

  it('asks where the jaws stand where nobody can vouch for the count', async () => {
    const lights = LIGHTS_READY.map((l) => (l.id === 'gripper' ? { ...l, state: 'blocked', code: 'jaws_unknown' } : l))
    const calls = serve({
      '/v1/cell': cellOf({ state: 'connected', arm: 'URRobotArm', hand: { ...HAND_TOGGLE, jaws: 'unknown' } }),
      '/v1/cell/readiness': { ready: false, lights, blockers: [] },
      'POST /v1/cell/jaws/check': { hand: HAND_TOGGLE, question: null },
    })
    renderWith(<Setup />)
    fireEvent.click(await screen.findByRole('button', { name: 'Backen prüfen' }))
    await waitFor(() => expect(posted(calls, '/v1/cell/jaws/check')).toHaveLength(1))
  })

  it('lets the jaws wait for "Zelle ist frei" while a stop nobody has cleared stands', async () => {
    // The server refuses the check cell_not_cleared then, asking nobody: the button stays off and says why.
    const lights = LIGHTS_READY.map((l) => (l.id === 'gripper' ? { ...l, state: 'blocked', code: 'jaws_closed' } : l))
    const stopped = { run_id: 'run-1', kind: 'task', stop_code: 'halted', at: 100, holding: true, cleared_at: null }
    const calls = serve({
      '/v1/cell': cellOf({ state: 'connected', arm: 'URRobotArm', hand: { ...HAND_TOGGLE, jaws: 'closed' }, recovery: stopped }),
      '/v1/cell/readiness': {
        ready: false,
        lights,
        blockers: [{ code: 'cell_not_cleared', message: 'not cleared' }, { code: 'part_still_held', message: 'held' }],
      },
      'POST /v1/cell/jaws/check': { hand: HAND_TOGGLE, question: null },
    })
    renderWith(<Setup />)
    const check = await screen.findByRole('button', { name: 'Backen prüfen' })
    expect((check as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getByText('erst nach „Zelle ist frei“ (oben, oder auf der Stoppkarte im Cockpit)')).toBeTruthy()
    fireEvent.click(check)
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(posted(calls, '/v1/cell/jaws/check')).toHaveLength(0)
    cleanup()
    // Once a person said the cell is clear, the check is offered again: the toggle's way to "Jetzt öffnen".
    serve({
      '/v1/cell': cellOf({
        state: 'connected',
        arm: 'URRobotArm',
        hand: { ...HAND_TOGGLE, jaws: 'closed' },
        recovery: { ...stopped, cleared_at: 101 },
      }),
      '/v1/cell/readiness': { ready: false, lights, blockers: [{ code: 'restart_required', message: 'restart' }] },
    })
    renderWith(<Setup />)
    await waitFor(async () =>
      expect(((await screen.findByRole('button', { name: 'Backen prüfen' })) as HTMLButtonElement).disabled).toBe(false),
    )
  })
})

describe('the poses', () => {
  it('show Home read-only, the taught poses by label with their verdicts, and the default place', async () => {
    serve({})
    renderWith(<PosesPanel />)
    const table = await screen.findByRole('table', { name: 'Posen' })
    const rows = within(table).getAllByRole('row').slice(1)
    expect(rows.map((r) => within(r).getAllByRole('cell')[0].textContent)).toEqual([
      expect.stringContaining('Home'),
      expect.stringContaining('Ablage links'),
      expect.stringContaining('Parkposition'),
    ])
    expect(within(rows[0]).queryByRole('radio')).toBeNull()
    expect(within(rows[1]).getByText('frei')).toBeTruthy()
    // A short chip; what it means is its title (a planned move begins with a straight leg of at most 10° per joint).
    const band = within(rows[2]).getByText('frei · gerader Start')
    expect(band.closest('[title]')?.getAttribute('title')).toMatch(/höchstens 10° je Gelenk/)
    expect((within(rows[1]).getByRole('radio') as HTMLInputElement).checked).toBe(true)
    // Where a taught pose is written, before any arm is freed.
    expect(screen.getByText(/robot\.cell\.yaml/)).toBeTruthy()
  })

  it('choose the default place through the pose door', async () => {
    const calls = serve({ 'PUT /v1/poses/default-place': { ...POSES, default_place: 'park' } })
    renderWith(<PosesPanel />)
    const table = await screen.findByRole('table', { name: 'Posen' })
    const park = within(table).getAllByRole('radio')[1]
    fireEvent.click(park)
    await waitFor(() => expect(posted(calls, '/v1/poses/default-place')).toHaveLength(1))
    expect(posted(calls, '/v1/poses/default-place')[0].body).toEqual({ name: 'park' })
  })

  it('say why no pose can be taught now: the planner is still starting', async () => {
    serve({ '/v1/poses': { ...POSES, teachable: false, why_not_code: 'planner_not_ready', why_not: 'the planner is starting' } })
    renderWith(<PosesPanel />)
    const teach = await screen.findByRole('button', { name: /Pose einlernen/ })
    expect((teach as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getByText('Planer startet noch')).toBeTruthy()
  })
})

const TEACH_RUN = {
  id: 'run-t',
  prompt: '',
  requested_picks: 0,
  state: 'running',
  started_at: 1,
  succeeded: 0,
  attempted: 0,
  error: '',
  stop_requested: false,
  kind: 'teach',
  stop_code: '',
  stop_class: '',
  parts_placed: 0,
  holding: false,
  halt_requested: false,
  step: '',
}

function teachState(over: Record<string, unknown> = {}) {
  return {
    state: 'free',
    outside: false,
    lines: [],
    joints_deg: [-60.1, -95.2, -120.3, -55.4, 90.5, 0.6],
    tcp_mm: [300.0, -400.0, 150.0],
    verdict: null,
    nearby_deg: null,
    message: '',
    time_left_s: 287.0,
    ...over,
  }
}

describe('the teach dialog', () => {
  let state: Record<string, unknown>
  let beacon: ReturnType<typeof vi.fn>

  function serveTeach(extra: Record<string, Route> = {}): ApiCall[] {
    return serve({
      '/v1/cell': cellOf({ state: 'connected', arm: 'URRobotArm', hand: HAND_TOGGLE }),
      '/v1/teach/payload': { mass_kg: 1.5, cog_mm: [0, 0, 60], readable: true, source: 'the controller, read now' },
      'POST /v1/teach': reply(202, { run: TEACH_RUN, token: 'tok' }),
      'GET /v1/teach/run-t': () => state,
      'POST /v1/teach/run-t/capture': () => state,
      'POST /v1/teach/run-t/cancel': () => ({ ...state, state: 'ended' }),
      '/v1/runs/run-t': { ...TEACH_RUN, state: 'finished', stop_code: 'taught', stop_class: 'done' },
      ...extra,
    })
  }

  async function free(calls: ApiCall[], role: 'place' | 'other' = 'place') {
    fireEvent.change(await screen.findByLabelText(/^Name/), { target: { value: 'drop_right' } })
    fireEvent.change(screen.getByLabelText(/^Bezeichnung/), { target: { value: 'Ablage rechts' } })
    if (role === 'other') fireEvent.click(screen.getByRole('radio', { name: /Andere Pose/ }))
    const go = screen.getByRole('button', { name: /Arm freigeben/ })
    expect((go as HTMLButtonElement).disabled).toBe(true)
    fireEvent.click(await screen.findByRole('checkbox', { name: /Nutzlast/ }))
    expect(posted(calls, '/v1/teach')).toHaveLength(0)
    fireEvent.click(go)
    await waitFor(() => expect(posted(calls, '/v1/teach')).toHaveLength(1))
  }

  beforeEach(() => {
    state = teachState()
    beacon = vi.fn(() => true)
    Object.defineProperty(window.navigator, 'sendBeacon', { configurable: true, writable: true, value: beacon })
  })

  afterEach(() => {
    Reflect.deleteProperty(window.navigator, 'sendBeacon')
  })

  it('shows where the pose is written and the payload before the arm is freed, and frees nothing before the click', async () => {
    const calls = serveTeach()
    renderWith(<TeachDialog poses={POSES as never} onClose={() => undefined} />)
    const dialog = await screen.findByRole('dialog', { name: /Pose einlernen/ })
    expect(within(dialog).getByText(/robot\.cell\.yaml/)).toBeTruthy()
    await waitFor(() => expect(dialog.textContent).toMatch(/1,5 kg/))
    // A place pose: the fingertips go where the part's bottom is let go (build plan item 7).
    expect(dialog.textContent).toMatch(/Fingerspitzen dorthin, wo die Unterseite des Teils losgelassen wird/)
    await free(calls)
    expect(posted(calls, '/v1/teach')[0].body).toEqual({
      name: 'drop_right',
      label: 'Ablage rechts',
      role: 'place',
      replace: false,
      make_default_place: false,
      payload_seen: { mass_kg: 1.5, cog_mm: [0, 0, 60] },
    })
  })

  it('frees nothing for a name already taken until the person ticks that it replaces that pose', async () => {
    const calls = serveTeach()
    renderWith(<TeachDialog poses={POSES as never} onClose={() => undefined} />)
    fireEvent.change(await screen.findByLabelText(/^Name/), { target: { value: 'drop_left' } })
    fireEvent.change(screen.getByLabelText(/^Bezeichnung/), { target: { value: 'Ablage links' } })
    fireEvent.click(await screen.findByRole('checkbox', { name: /Nutzlast/ }))
    const go = screen.getByRole('button', { name: /Arm freigeben/ })
    expect((go as HTMLButtonElement).disabled).toBe(true)
    fireEvent.click(screen.getByRole('checkbox', { name: /„drop_left“ ersetzen/ }))
    expect((go as HTMLButtonElement).disabled).toBe(false)
    fireEvent.click(go)
    await waitFor(() => expect(posted(calls, '/v1/teach')).toHaveLength(1))
    expect(posted(calls, '/v1/teach')[0].body).toMatchObject({ name: 'drop_left', replace: true })
  })

  it('polls the session four times a second as its heartbeat, and sends the cancel beacon when the page goes', async () => {
    const calls = serveTeach()
    renderWith(<TeachDialog poses={POSES as never} onClose={() => undefined} />)
    await free(calls)
    const polls = () => calls.filter((c) => c.method === 'GET' && c.path === '/v1/teach/run-t')
    await waitFor(() => expect(polls().length).toBeGreaterThanOrEqual(4), { timeout: 2000 })
    expect(polls()[0].query).toBe('token=tok')
    act(() => {
      window.dispatchEvent(new Event('pagehide'))
    })
    expect(beacon).toHaveBeenCalledWith('/v1/teach/run-t/cancel?token=tok')
  })

  it('says the arm is held only once it stands still, and warns in the last 30 s', async () => {
    const calls = serveTeach()
    renderWith(<TeachDialog poses={POSES as never} onClose={() => undefined} />)
    await free(calls)
    state = teachState({ time_left_s: 25 })
    expect(await screen.findByText(/Noch 25 s/)).toBeTruthy()
    state = teachState({ state: 'holding_when_still', time_left_s: 3 })
    expect(await screen.findByText(/hält, sobald der Arm stillsteht/i)).toBeTruthy()
  })

  it('says from 30 s on what happens at 0, announced once, and shows no time left once the arm is being held', async () => {
    const calls = serveTeach()
    renderWith(<TeachDialog poses={POSES as never} onClose={() => undefined} />)
    await free(calls)
    state = teachState({ time_left_s: 45 })
    expect(await screen.findByText('noch 45 s zum Speichern')).toBeTruthy()
    expect(screen.queryByRole('alert')).toBeNull()
    state = teachState({ time_left_s: 25 })
    expect(await screen.findByText('Noch 25 s: danach hält der Arm, sobald er stillsteht. Gespeichert wird nichts.')).toBeTruthy()
    const alert = await screen.findByRole('alert')
    expect(alert.textContent).toMatch(/hält der Arm, sobald er stillsteht/)
    state = teachState({ time_left_s: 24 })
    expect(await screen.findByText(/^Noch 24 s/)).toBeTruthy()
    // Said once, when the 30 s were crossed: the countdown under it is not read out every second.
    expect(screen.getAllByRole('alert')).toHaveLength(1)
    state = teachState({ state: 'holding_when_still', time_left_s: 0 })
    expect(await screen.findByText(/Hält, sobald der Arm stillsteht/)).toBeTruthy()
    expect(screen.queryByText(/noch 0 s/i)).toBeNull()
  })

  it("says the 30 s warning aloud once where voice output is on, in the reader's language", async () => {
    const spoken: string[] = []
    class Utterance {
      lang = ''
      text: string
      constructor(text: string) {
        this.text = text
      }
    }
    vi.stubGlobal('SpeechSynthesisUtterance', Utterance)
    vi.stubGlobal('speechSynthesis', { speak: (u: Utterance) => spoken.push(`${u.lang} ${u.text}`), cancel: () => undefined, getVoices: () => [] })
    localStorage.setItem('willy.voiceOut', 'on')
    const calls = serveTeach()
    renderWith(<TeachDialog poses={POSES as never} onClose={() => undefined} />)
    await free(calls)
    state = teachState({ time_left_s: 45 })
    expect(await screen.findByText('noch 45 s zum Speichern')).toBeTruthy()
    expect(spoken).toEqual([])
    state = teachState({ time_left_s: 25 })
    expect(await screen.findByText(/^Noch 25 s/)).toBeTruthy()
    state = teachState({ time_left_s: 24 })
    expect(await screen.findByText(/^Noch 24 s/)).toBeTruthy()
    // Once, when the last 30 s begin: the person's hands are on the arm, the eyes perhaps not on the screen.
    expect(spoken).toEqual(['de-DE Noch höchstens 30 Sekunden zum Speichern: danach hält der Arm, sobald er stillsteht.'])
  })

  it('says nothing aloud where voice output is off, as it is unless switched on', async () => {
    const speak = vi.fn()
    vi.stubGlobal(
      'SpeechSynthesisUtterance',
      class {
        lang = ''
        text: string
        constructor(text: string) {
          this.text = text
        }
      },
    )
    vi.stubGlobal('speechSynthesis', { speak, cancel: () => undefined, getVoices: () => [] })
    const calls = serveTeach()
    renderWith(<TeachDialog poses={POSES as never} onClose={() => undefined} />)
    await free(calls)
    state = teachState({ time_left_s: 25 })
    expect(await screen.findByText(/^Noch 25 s/)).toBeTruthy()
    expect(speak).not.toHaveBeenCalled()
  })

  it('says what is still missing beside the button that frees the arm, the label as required as the name', async () => {
    serveTeach()
    renderWith(<TeachDialog poses={POSES as never} onClose={() => undefined} />)
    const go = await screen.findByRole('button', { name: /Arm freigeben/ })
    expect((go as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getByText('Fehlt noch: Name, Bezeichnung, Nutzlast bestätigen')).toBeTruthy()
    expect(screen.getByText(/^Pflicht/)).toBeTruthy()
    fireEvent.change(screen.getByLabelText(/^Name/), { target: { value: 'drop_right' } })
    expect(screen.getByText('Fehlt noch: Bezeichnung, Nutzlast bestätigen')).toBeTruthy()
  })

  it('offers no way out while the arm is being freed, and holds a session that answers after the dialog went away', async () => {
    let answer: (value: unknown) => void = () => undefined
    const calls = serveTeach({ 'POST /v1/teach': () => new Promise((resolve) => (answer = resolve)) })
    const onClose = vi.fn()
    const view = renderWith(<TeachDialog poses={POSES as never} onClose={onClose} />)
    await free(calls)
    // Freeing is under way: neither "Abbrechen" nor Escape can close the dialog over an arm that is about to be free.
    expect((screen.getByRole('button', { name: 'Abbrechen' }) as HTMLButtonElement).disabled).toBe(true)
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
    expect(onClose).not.toHaveBeenCalled()
    // The page goes away all the same (a navigation), then the server frees the arm: it is held at once, never followed.
    view.unmount()
    await act(async () => {
      answer(reply(202, { run: TEACH_RUN, token: 'tok' }))
      await new Promise((resolve) => setTimeout(resolve, 20))
    })
    await waitFor(() => expect(posted(calls, '/v1/teach/run-t/cancel')).toHaveLength(1))
    expect(posted(calls, '/v1/teach/run-t/cancel')[0].query).toBe('token=tok')
    expect(calls.filter((c) => c.method === 'GET' && c.path === '/v1/teach/run-t')).toHaveLength(0)
  })

  it('draws "Halten" in the orange of a halt, and the joints in the reader\'s numbers', async () => {
    const calls = serveTeach()
    renderWith(<TeachDialog poses={POSES as never} onClose={() => undefined} />)
    await free(calls)
    const hold = await screen.findByRole('button', { name: /Halten/ })
    expect(hold.className).toMatch(/\bhalt\b/)
    expect(await screen.findByText(/-60,1/)).toBeTruthy()
    expect(screen.queryByText(/-60\.1/)).toBeNull()
    expect(screen.getByText(/„Ablage rechts“/)).toBeTruthy()
  })

  it('quotes the pose\'s label in the reader\'s own quotation marks', async () => {
    const calls = serveTeach()
    renderWith(<TeachDialog poses={POSES as never} onClose={() => undefined} />, { lang: 'en' })
    fireEvent.change(await screen.findByLabelText(/^Name/), { target: { value: 'drop_right' } })
    fireEvent.change(screen.getByLabelText(/^Label/), { target: { value: 'Drop right' } })
    fireEvent.click(await screen.findByRole('checkbox', { name: /payload/ }))
    fireEvent.click(screen.getByRole('button', { name: /Free the arm/ }))
    await waitFor(() => expect(posted(calls, '/v1/teach')).toHaveLength(1))
    expect(await screen.findByText('“Drop right”')).toBeTruthy()
    expect(screen.queryByText(/„Drop right“/)).toBeNull()
  })

  it('reads the controller\'s payload again on request, and asks for it to be confirmed again', async () => {
    let reads = 0
    serveTeach({ '/v1/teach/payload': () => ({ mass_kg: ++reads === 1 ? 1.5 : 2.25, cog_mm: [0, 0, 60], readable: true, source: 'controller' }) })
    renderWith(<TeachDialog poses={POSES as never} onClose={() => undefined} />)
    const confirm = await screen.findByRole('checkbox', { name: /Nutzlast/ })
    await waitFor(() => expect(screen.getByRole('dialog').textContent).toMatch(/1,5 kg/))
    fireEvent.click(confirm)
    expect((confirm as HTMLInputElement).checked).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: 'Nutzlast neu lesen' }))
    await waitFor(() => expect(screen.getByRole('dialog').textContent).toMatch(/2,25 kg/))
    expect((screen.getByRole('checkbox', { name: /Nutzlast/ }) as HTMLInputElement).checked).toBe(false)
  })

  it('draws a red banner while the tool is outside the workspace', async () => {
    const calls = serveTeach()
    renderWith(<TeachDialog poses={POSES as never} onClose={() => undefined} />)
    await free(calls)
    state = teachState({ outside: true })
    const alert = await screen.findByRole('alert')
    expect(alert.textContent).toMatch(/außerhalb des Arbeitsraums/)
  })

  it('saves with capture and holds with cancel, each at a click', async () => {
    const calls = serveTeach()
    renderWith(<TeachDialog poses={POSES as never} onClose={() => undefined} />)
    await free(calls)
    fireEvent.click(await screen.findByRole('button', { name: /Speichern/ }))
    await waitFor(() => expect(posted(calls, '/v1/teach/run-t/capture')).toHaveLength(1))
    fireEvent.click(screen.getByRole('button', { name: /Halten/ }))
    await waitFor(() => expect(posted(calls, '/v1/teach/run-t/cancel')).toHaveLength(1))
    expect(posted(calls, '/v1/teach/run-t/cancel')[0].query).toBe('token=tok')
  })

  it('ends on the verdict: a saved pose, with its screen', async () => {
    const calls = serveTeach()
    renderWith(<TeachDialog poses={POSES as never} onClose={() => undefined} />)
    await free(calls)
    state = teachState({ state: 'saved', verdict: 'clear', time_left_s: null })
    expect(await screen.findByText(/Gespeichert: „Ablage rechts“/)).toBeTruthy()
    expect(screen.getByText('frei')).toBeTruthy()
    // The session is over: no beacon goes for it any more.
    act(() => {
      window.dispatchEvent(new Event('pagehide'))
    })
    expect(beacon).not.toHaveBeenCalled()
  })

  it('ends on the verdict: a refused pose, with the nearest clear joints', async () => {
    const calls = serveTeach({ '/v1/runs/run-t': { ...TEACH_RUN, state: 'finished', stop_code: 'teach_refused', stop_class: 'ask' } })
    renderWith(<TeachDialog poses={POSES as never} onClose={() => undefined} />)
    await free(calls)
    state = teachState({ state: 'refused', verdict: 'guard_refused', nearby_deg: [-50, -95, -120, -55, 90, 0], time_left_s: null })
    expect(await screen.findByText(/Nicht gespeichert/)).toBeTruthy()
    // The verdict once, as its chip, in German (the collision check, not "the Guard"), never repeated as a sentence.
    expect(screen.getByText('abgelehnt: Kollisionsprüfung')).toBeTruthy()
    expect(screen.getByRole('dialog').textContent).not.toMatch(/Guard|lehnt die Pose ab/)
    expect(screen.getByText(/-50/)).toBeTruthy()
  })

  it('ends on the verdict: a refused pose with no clear pose nearby says to teach it elsewhere', async () => {
    const calls = serveTeach({ '/v1/runs/run-t': { ...TEACH_RUN, state: 'finished', stop_code: 'teach_refused', stop_class: 'ask' } })
    renderWith(<TeachDialog poses={POSES as never} onClose={() => undefined} />)
    await free(calls)
    state = teachState({ state: 'refused', verdict: 'planner_refused', nearby_deg: null, time_left_s: null })
    expect(await screen.findByText('abgelehnt: Planer')).toBeTruthy()
    expect(screen.getByText(/an anderer Stelle einlernen/)).toBeTruthy()
  })
})

describe('the cell facts', () => {
  it('say what the cell is: the camera, the push, the brake and the carried part', async () => {
    serve({
      '/v1/cell': cellOf({ state: 'connected', arm: 'URRobotArm', gripper: 'JawIOGripper', hand: HAND_TOGGLE, planner: { state: 'ready' } }),
      '/v1/cell/facts': {
        wrist_camera: true,
        cameras: [{ rig_id: 'EIH_Cam', mounting: 'wrist', primary: true }],
        looks: ['look_1', 'look_2', 'look_3'],
        natural_closing_axis: '-y',
        push: { can_push: true, why_not: '', default_mm: 30, ceiling_mm: 50 },
        detector: { backend: 'vlm', router_enabled: false, vlm_model_id: 'Qwen/Qwen3-VL-4B-Instruct', precision: 'auto' },
        hand_eye_warn_mm: 6,
        brake: { latches: true, brakes_in_motion: false },
        payload: { modelled: true, declined_reason: null, length_mm: 40 },
        route: { route: 'planned', sentence: 'cuRobo plans every move' },
        rehearsal: false,
      },
    })
    renderWith(<CellFacts />)
    const facts = await screen.findByRole('region', { name: 'Zelle' })
    await waitFor(() => expect(facts.textContent).toMatch(/EIH_Cam/))
    expect(facts.textContent).toMatch(/Handgelenk/)
    expect(facts.textContent).toMatch(/30 mm, höchstens 50 mm/)
    expect(facts.textContent).toMatch(/modelliert \(40 mm\)/)
    expect(facts.textContent).toMatch(/Tool-DO0/)
  })

  it("name the detector in a person's words, and its config name only in the tech view", async () => {
    const routes = {
      '/v1/cell': cellOf({ state: 'connected', arm: 'URRobotArm', gripper: 'JawIOGripper', hand: HAND_TOGGLE }),
      '/v1/cell/facts': {
        wrist_camera: true,
        cameras: [{ rig_id: 'EIH_Cam', mounting: 'wrist', primary: true }],
        looks: ['look_1'],
        detector: { backend: 'grounded_sam', router_enabled: false, vlm_model_id: null, precision: 'fp32 weights, fp16 autocast' },
        hand_eye_warn_mm: 6,
        route: { route: 'planned', sentence: '' },
        rehearsal: false,
      },
    }
    serve(routes)
    renderWith(<CellFacts />)
    const facts = await screen.findByRole('region', { name: 'Zelle' })
    await waitFor(() => expect(facts.textContent).toMatch(/GroundingDINO \+ SAM/))
    expect(facts.textContent).not.toMatch(/grounded_sam/)
    cleanup()

    localStorage.setItem('willy.view', 'tech')
    serve(routes)
    renderWith(<CellFacts />)
    const tech = await screen.findByRole('region', { name: 'Zelle' })
    await waitFor(() => expect(tech.textContent).toMatch(/GroundingDINO \+ SAM/))
    expect(tech.textContent).toMatch(/grounded_sam/)
  })
})
