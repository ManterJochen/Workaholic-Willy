/**
 * The cockpit (build plan 4.2, owner decisions of the fifth round): the live image, the run strip under it, the
 * statistics and the chat to its right, against a stubbed server and fake sockets.
 *
 * The pins that matter most are safety, and they are pinned in both languages: Start is the confirmation and names the
 * first motion; nothing a person says or types starts anything; "Sofort anhalten" is one click and is not the e-stop;
 * the red "press the e-stop" alarm appears only where the arm brakes a move in flight; Restart and Home ask first; the
 * jaws question is never answered here; nothing starts by itself after a stop.
 *
 * The three pins of the old Pick screen (`screens.test.tsx`, "Pick") live here now, with the cockpit's copy.
 */

import { act, cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { RunEvent } from '../api/events'
import type { Lang } from '../i18n'
import { clearConversation, say } from '../model/chat'
import { ConfirmProvider } from '../model/confirm'
import targetLostJson from '../test/fixtures/task_camera_target_lost.json'
import haltedRestartJson from '../test/fixtures/task_halted_then_restart.json'
import twoPartsJson from '../test/fixtures/task_two_parts_nothing_left.json'
import { api } from '../api/client'
import { eventsOf, fakeSockets, refusal, renderWith, reply, runOf, stubApi, type ApiCall, type FakeSockets, type Fixture, type Route } from '../test/render'
import Cockpit from './Cockpit'
import { firstMotion } from './draft'
import Stage from './Stage'

const twoParts = twoPartsJson as unknown as Fixture
const halted = haltedRestartJson as unknown as Fixture
const targetLost = targetLostJson as unknown as Fixture
const TWO = runOf(twoParts)
const HALTED = runOf(halted, (record) => record.stop_class === 'problem')

const HAND = { kind: 'toggle', driver: 'JawIOGripper', where: 'tool output 0', connected: true, jaws: 'open', why_unknown: '', no_sensor: true, commands_sent: 2 }

const CELL = {
  state: 'connected',
  arm: 'URRobotArm',
  gripper: 'JawIOGripper',
  vendor: 'ur',
  profile: 'ur10_cell',
  active_run_id: null as string | null,
  needs_person: '',
  halted: null,
  payload_model: 'none',
  recovery: null,
  jaws_confirmed_at: null,
  countdown_due: false,
  jaws_question: false,
  hand: HAND,
  planner: { state: 'ready' },
}

const light = (id: string, state: string, code: string) => ({ id, state, code, message: `${id} ${code}`, blocks: id !== 'commands' })
const LIGHTS = [
  light('robot', 'ok', 'connected'),
  light('cameras', 'ok', 'live'),
  light('planner', 'ok', 'ready'),
  light('gripper', 'ok', 'open_confirmed'),
  light('carried_part', 'ok', 'modelled'),
  light('commands', 'info', 'ready'),
]
const READY = { ready: true, lights: LIGHTS, blockers: [] }

const FACTS = {
  wrist_camera: true,
  cameras: [{ rig_id: 'EIH_Cam', mounting: 'wrist', primary: true }],
  looks: ['look_1', 'look_2', 'look_3'],
  natural_closing_axis: '-y',
  push: { can_push: true, why_not: '', default_mm: 30, ceiling_mm: 50 },
  detector: { backend: 'vlm', router_enabled: false, vlm_model_id: 'Qwen/Qwen3-VL-4B-Instruct', precision: 'auto' },
  hand_eye_warn_mm: 6,
  brake: { latches: true, brakes_in_motion: false },
  payload: { modelled: true, declined_reason: null, length_mm: 40 },
  route: { route: 'planned', sentence: '' },
  rehearsal: false,
}

const POSES = {
  home: { name: 'home', label: 'Home', joints_deg: [0, -90, 90, -90, -90, 0], source: 'config', screen: 'clear', note: '', taught_at: null },
  poses: [
    { name: 'drop_left', label: 'Ablage links', joints_deg: [-60, -95, -120, -55, 90, 0], source: 'taught', screen: 'clear', note: '', taught_at: null },
    { name: 'park', label: 'Parkposition', joints_deg: [30, -80, -110, -80, 90, 0], source: 'taught', screen: 'clear', note: '', taught_at: null },
  ],
  default_place: 'drop_left',
  teachable: true,
  why_not: '',
  why_not_code: '',
  target_file: 'config/robot/robot.cell.yaml',
}

const now = () => Date.now() / 1000

function live(extra: Record<string, unknown> = {}) {
  return {
    rig_id: 'EIH_Cam',
    rigs: [{ rig_id: 'EIH_Cam', mounting: 'wrist', primary: true }],
    source: 'camera',
    reason: '',
    image_base64: 'AAAA',
    width: 960,
    height: 540,
    captured_at: now(),
    age_s: 0.2,
    ...extra,
  }
}

const ROUTE = { prompt: 'green cube', route: 'simple', reason: 'english', description: 'simple (english)', runnable: true }

const PARSED = {
  understood: true,
  intent: 'task',
  object: { phrase: 'green cube', said: 'alle grünen Würfel', verified: true, route: ROUTE },
  place: null,
  place_pose: 'drop_left',
  scope: 'until_empty',
  count: null,
  return_to: null,
  notes: [],
  reason: '',
  model: { model_id: 'Qwen/Qwen3-VL-4B-Instruct', latency_ms: 2200, attempts: 1, loaded_now: false },
  raw: '{"object": "green cube"}',
}

const ACCEPTED = { id: 'run-new', prompt: 'green cube', requested_picks: 0, state: 'running', started_at: 1, succeeded: 0, attempted: 0, error: '', stop_requested: false, kind: 'task', stop_code: '', stop_class: '', parts_placed: 0, holding: false, halt_requested: false, step: '' }

/** The server, with every route the cockpit reads answered; a test overrides what it is about. */
function server(over: Record<string, Route> = {}): ApiCall[] {
  return stubApi({
    '/v1/cell': CELL,
    '/v1/cell/readiness': READY,
    '/v1/cell/facts': FACTS,
    '/v1/cell/status': { state: 'connected', connected: true, simulated: false, vendor: 'ur', model: 'ur10', controller_state_included: false },
    '/v1/poses': POSES,
    '/v1/camera/live': () => live(),
    'POST /v1/commands/parse': PARSED,
    'POST /v1/task': reply(202, ACCEPTED),
    'POST /v1/cell/brake': { run_halted: true, latched: true, braking: false, in_motion: true, run_id: null, message: 'halted' },
    'POST /v1/task/stop': ACCEPTED,
    'POST /v1/cell/home': reply(202, { ...ACCEPTED, id: 'run-home', kind: 'home' }),
    'POST /v1/cell/wave': reply(202, { ...ACCEPTED, id: 'run-wave', kind: 'wave', prompt: '' }),
    'POST /v1/task/restart': reply(202, { ...ACCEPTED, id: 'run-restart', restart_of: HALTED }),
    'POST /v1/cell/acknowledge': CELL,
    'POST /v1/cell/jaws/check': { hand: HAND, question: null },
    ...over,
  })
}

const MOVES = ['POST /v1/task', 'POST /v1/task/restart', 'POST /v1/cell/home', 'POST /v1/pick', 'POST /v1/cell/wave']
const moved = (calls: ApiCall[]) => calls.filter((c) => MOVES.includes(`${c.method} ${c.path}`))
const sent = (calls: ApiCall[], key: string) => calls.filter((c) => `${c.method} ${c.path}` === key)

function cockpit(lang: Lang = 'de') {
  return renderWith(
    <ConfirmProvider>
      <Cockpit />
    </ConfirmProvider>,
    { lang },
  )
}

/** Type a sentence and press Enter: the parse, never a start. */
async function command(text: string) {
  const box = await screen.findByRole('textbox', { name: /Befehl an Willy|Command for Willy/ })
  await waitFor(() => expect(box.hasAttribute('disabled')).toBe(false))
  fireEvent.change(box, { target: { value: text } })
  fireEvent.keyDown(box, { key: 'Enter' })
}

/** A live event: stamped now, so the cockpit treats it as happening, not as a replay. */
function ev(runId: string, seq: number, type: string, data: Record<string, unknown> = {}, ts = now()): RunEvent {
  return { type, run_id: runId, seq, ts, severity: 'info', human: `${type} said`, step: '', step_index: null, step_total: null, data }
}

function follow(sockets: FakeSockets, runId: string) {
  return waitFor(() => {
    const socket = sockets.find(runId)
    if (!socket) throw new Error(`no socket for ${runId}`)
    return socket
  })
}

let sockets: FakeSockets

beforeEach(() => {
  localStorage.clear()
  clearConversation()
  sockets = fakeSockets()
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  localStorage.clear()
  clearConversation()
})

describe('the Pick screen\'s pins, now the cockpit\'s, in both languages', () => {
  const COPY = {
    de: {
      stopAfter: /Nach diesem Teil stoppen/, finishes: /legt das Teil noch ab/, halt: /Sofort anhalten/, notEstop: /kein Not-Aus/,
      redButton: /Kein Not-Aus; das ist der rote Taster/, start: /Start – der Roboter fährt zu Blick 1/, down: /Die Zelle ist nicht verbunden/,
    },
    en: {
      stopAfter: /Stop after this part/, finishes: /the part in hand is still placed/, halt: /Halt now/, notEstop: /not the e-stop/,
      redButton: /Not an emergency stop; the red button is/, start: /Start – the robot moves to look 1/, down: /The cell is not connected/,
    },
  } as const

  for (const lang of ['de', 'en'] as const) {
    const copy = COPY[lang]

    it(`Stop after this part lets the part finish, and Halt now is not the emergency stop (${lang})`, async () => {
      server()
      cockpit(lang)
      const stopAfter = await screen.findByRole('button', { name: copy.stopAfter })
      expect(stopAfter.textContent).toMatch(copy.finishes)
      const halt = screen.getByRole('button', { name: copy.halt })
      expect(halt.textContent).toMatch(copy.notEstop)
      expect(halt.getAttribute('title')).toMatch(copy.redButton)
      expect(halt.className).toContain('halt')
    })

    it(`Start says the robot moves, and names its first motion (${lang})`, async () => {
      server()
      cockpit(lang)
      await command('Räum alle grünen Würfel auf die Ablage links')
      const start = await screen.findByRole('button', { name: copy.start })
      // The owner's decision of 2026-10-02: the controls that start motion are the lime accent, never red.
      expect(start.className).toContain('primary')
      expect(start.className).not.toContain('danger')
    })

    it(`The cell is not connected (${lang})`, async () => {
      server({ '/v1/cell': { ...CELL, state: 'disconnected' }, '/v1/cell/readiness': { ready: false, lights: [light('robot', 'blocked', 'not_connected')], blockers: [] } })
      cockpit(lang)
      expect(await screen.findByText(copy.down)).toBeTruthy()
    })
  }
})

describe('Start', () => {
  it('is the confirmation: one click sends the task once, with no dialog, and the run is followed', async () => {
    const calls = server()
    cockpit()
    await command('Räum alle grünen Würfel auf die Ablage links')
    const start = await screen.findByRole('button', { name: /Start – der Roboter fährt zu Blick 1/ })
    await waitFor(() => expect(start.hasAttribute('disabled')).toBe(false))
    fireEvent.click(start)
    await waitFor(() => expect(sent(calls, 'POST /v1/task')).toHaveLength(1))
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(sent(calls, 'POST /v1/task')[0].body).toMatchObject({
      object: 'green cube',
      object_said: 'alle grünen Würfel',
      place: { kind: 'pose', pose: 'drop_left' },
      return_to: 'home',
      scope: 'until_empty',
      options: { multi_view: true, both_faces: false, record_views: false, pick_anything: false },
      command: { text: 'Räum alle grünen Würfel auf die Ablage links', source: 'typed', parsed: true, edited: [] },
    })
    // The 202 answer is followed at once: a run that ends between two cell polls is still drawn.
    await follow(sockets, 'run-new')
    expect(moved(calls)).toHaveLength(1)
  })

  it('is off until the cell is ready, and says why', async () => {
    server({
      '/v1/cell/readiness': { ready: false, blockers: [], lights: [...LIGHTS.filter((l) => l.id !== 'planner'), light('planner', 'wait', 'starting')] },
    })
    cockpit()
    await command('Nimm den grünen Würfel')
    const start = await screen.findByRole('button', { name: /Start – der Roboter fährt zu Blick 1/ })
    expect(start.hasAttribute('disabled')).toBe(true)
    const card = screen.getByRole('region', { name: 'Verstanden' })
    expect(within(card).getByText(/Planer: Startet/)).toBeTruthy()
  })

  for (const [lang, motion, countdown] of [
    ['de', /^Start – der Roboter fährt zu Blick 1/, 'zuerst 3 s Countdown „Hände weg“'],
    ['en', /^Start – the robot moves to look 1/, 'first a 3 s “hands off” countdown'],
  ] as const) {
    it(`names the countdown under its first motion when one is due, in words that read right (${lang})`, async () => {
      server({ '/v1/cell': { ...CELL, countdown_due: true } })
      cockpit(lang)
      await command('Nimm den grünen Würfel')
      const start = await screen.findByRole('button', { name: motion })
      expect(start.textContent).toContain(countdown)
      // "dann der Roboter fährt" is not German: the verb comes second after "dann".
      expect(start.textContent).not.toMatch(/dann der Roboter fährt/)
    })
  }

  it('names the put-back fallback for a target the camera finds, in the reader\'s words', async () => {
    server({
      'POST /v1/commands/parse': { ...PARSED, place_pose: null, place: { phrase: 'blue bin', said: 'in die blaue Kiste', verified: true, route: { ...ROUTE, prompt: 'blue bin' } } },
    })
    cockpit()
    await command('Räum alle grünen Würfel in die blaue Kiste')
    const start = await screen.findByRole('button', { name: /Start – der Roboter fährt zu Blick 1/ })
    expect(start.textContent).toMatch(/Findet er das Ziel nicht, legt er das Teil zurück und fragt/)
    // The detector's English phrase stays in its field: the sentence on the button is German.
    expect(start.textContent).not.toMatch(/blue bin/)
    expect(screen.getByText(/Kamera sucht/)).toBeTruthy()
  })

  it('quotes the sentence it read in the reader\'s language', async () => {
    server()
    cockpit('en')
    await command('Put all green cubes onto the left place')
    const card = await screen.findByRole('region', { name: 'Understood' })
    expect(within(card).getByText('“Put all green cubes onto the left place”')).toBeTruthy()
  })

  it('asks what to pick when the sentence names nothing, and takes "anything" only with the tick', async () => {
    const calls = server({ 'POST /v1/commands/parse': { ...PARSED, object: null, notes: ['object_not_in_sentence'] } })
    cockpit()
    await command('Räum auf')
    const start = await screen.findByRole('button', { name: /Start – der Roboter fährt zu Blick 1/ })
    expect(start.hasAttribute('disabled')).toBe(true)
    expect(screen.getByText('Was soll ich greifen?')).toBeTruthy()
    fireEvent.click(screen.getByRole('checkbox', { name: /alles, was die Kamera sieht, auch Kistenwände/ }))
    await waitFor(() => expect(start.hasAttribute('disabled')).toBe(false))
    fireEvent.click(start)
    await waitFor(() => expect(sent(calls, 'POST /v1/task')).toHaveLength(1))
    expect(sent(calls, 'POST /v1/task')[0].body).toMatchObject({ object: '', options: { pick_anything: true } })
  })

  it('is not offered under a stop record: the stop card offers Restart and Home instead', async () => {
    const record = { run_id: HALTED, kind: 'task', stop_code: 'halted', at: 10, holding: false, cleared_at: null }
    server({
      '/v1/cell': { ...CELL, recovery: record },
      '/v1/cell/readiness': { ready: false, lights: LIGHTS, blockers: [{ code: 'restart_required', message: 'restart' }] },
      [`/v1/runs/${HALTED}`]: halted.runs[HALTED],
    })
    cockpit()
    const card = await screen.findByRole('region', { name: /Angehalten/ })
    expect(await within(card).findByRole('button', { name: /Neustart – geplante Fahrt nach Home/ })).toBeTruthy()
    expect(within(card).getByRole('button', { name: /Home – geplante Fahrt nach Home/ })).toBeTruthy()
    expect(screen.queryByRole('button', { name: /^Start/ })).toBeNull()
    expect(screen.getByRole('textbox', { name: 'Befehl an Willy' }).hasAttribute('disabled')).toBe(true)
  })

  it('says a refusal right above Start, where the person is looking, brings it into view, and folds its code in the demo view', async () => {
    const shown = vi.fn()
    const had = Object.getOwnPropertyDescriptor(Element.prototype, 'scrollIntoView')
    Object.defineProperty(Element.prototype, 'scrollIntoView', { configurable: true, writable: true, value: shown })
    try {
      const calls = server({ 'POST /v1/task': refusal(409, 'camera_target_unavailable', 'this cell has no camera to find a target with') })
      cockpit()
      await command('Räum alle grünen Würfel in die blaue Kiste')
      const card = await screen.findByRole('region', { name: 'Verstanden' })
      const start = await within(card).findByRole('button', { name: /^Start/ })
      await waitFor(() => expect(start.hasAttribute('disabled')).toBe(false))
      fireEvent.click(start)
      await waitFor(() => expect(sent(calls, 'POST /v1/task')).toHaveLength(1))
      const banner = await within(card).findByRole('alert')
      expect(banner.textContent).toMatch(/Diese Zelle kann kein Ziel mit der Kamera suchen\./)
      expect(banner.compareDocumentPosition(start) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
      await waitFor(() => expect(shown.mock.contexts.some((el) => el instanceof Element && el.contains(banner))).toBe(true))
      // No raw code for the audience in the demo view: it is under "Details", for a person who must quote it.
      expect(within(banner).getByText(/camera_target_unavailable · HTTP 409/).closest('details')).not.toBeNull()
      expect(moved(calls)).toHaveLength(1)
    } finally {
      if (had) Object.defineProperty(Element.prototype, 'scrollIntoView', had)
      else delete (Element.prototype as { scrollIntoView?: unknown }).scrollIntoView
    }
  })
})

describe('reading a command', () => {
  it('never starts anything: Enter only asks the reader', async () => {
    const calls = server()
    cockpit()
    await command('Räum alle grünen Würfel auf die Ablage links')
    await screen.findByRole('region', { name: 'Verstanden' })
    expect(sent(calls, 'POST /v1/commands/parse')).toHaveLength(1)
    expect(sent(calls, 'POST /v1/commands/parse')[0].body).toMatchObject({ text: 'Räum alle grünen Würfel auf die Ablage links', source: 'typed', language: 'de' })
    expect(moved(calls)).toHaveLength(0)
    expect(screen.getByText('Verstanden. Prüf kurz und drück auf Start.')).toBeTruthy()
  })

  it('shows the card with what was understood: the phrase, the words said, the place by its label, the scope', async () => {
    server()
    cockpit()
    await command('Räum alle grünen Würfel auf die Ablage links')
    const card = await screen.findByRole('region', { name: 'Verstanden' })
    expect((within(card).getByRole('textbox', { name: 'Greifen' }) as HTMLInputElement).value).toBe('green cube')
    expect(within(card).getByText('deine Worte: „alle grünen Würfel“')).toBeTruthy()
    expect((within(card).getByRole('combobox', { name: 'Ablegen' }) as HTMLSelectElement).selectedOptions[0].textContent).toBe('Ablage links')
    expect(within(card).getByRole('button', { name: 'Bis leer' }).getAttribute('aria-pressed')).toBe('true')
    expect(within(card).getByText('PHRASE')).toBeTruthy()
  })

  it('opens the card by hand when the language model is not there (501), and starts nothing by itself', async () => {
    const calls = server({ 'POST /v1/commands/parse': refusal(501, 'vlm_unavailable', 'no VLM on this cell') })
    cockpit()
    await command('Nimm den grünen Würfel')
    const card = await screen.findByRole('region', { name: 'Auftrag von Hand' })
    expect(within(card).getByText(/Das Sprachmodell ist auf dieser Zelle nicht verfügbar/)).toBeTruthy()
    const start = within(card).getByRole('button', { name: /Start – der Roboter fährt zu Blick 1/ })
    expect(start.hasAttribute('disabled')).toBe(true)
    fireEvent.change(within(card).getByRole('textbox', { name: 'Greifen' }), { target: { value: 'green cube' } })
    await waitFor(() => expect(start.hasAttribute('disabled')).toBe(false))
    expect(moved(calls)).toHaveLength(0)
    fireEvent.click(start)
    await waitFor(() => expect(sent(calls, 'POST /v1/task')).toHaveLength(1))
    expect(sent(calls, 'POST /v1/task')[0].body).toMatchObject({ object: 'green cube', command: { parsed: false, edited: ['object'] } })
  })

  it('answers a stop said in words with where the stop buttons are, and stops nothing', async () => {
    const calls = server({ 'POST /v1/commands/parse': { ...PARSED, intent: 'stop', object: null, place_pose: null } })
    cockpit()
    await command('Stopp!')
    expect(await screen.findByText(/Zum Anhalten die Knöpfe unter dem Bild benutzen/)).toBeTruthy()
    expect(calls.filter((c) => c.method === 'POST' && c.path !== '/v1/commands/parse')).toHaveLength(0)
    expect(screen.queryByRole('region', { name: 'Verstanden' })).toBeNull()
  })

  it('keeps the operator\'s words in a bubble, and says when they were spoken', async () => {
    say({ who: 'operator', kind: 'command', text: 'Räum die gelbe Kiste aus', source: 'spoken' })
    server()
    cockpit()
    const bubble = (await screen.findByText('Räum die gelbe Kiste aus')).closest('.ck-bubble') as HTMLElement
    expect(bubble).toBeTruthy()
    expect(within(bubble).getByText('gesprochen')).toBeTruthy()
  })

  it('is locked while a run is active: the box and the microphone', async () => {
    server({ '/v1/cell': { ...CELL, active_run_id: 'run-busy' }, '/v1/runs/run-busy': { ...ACCEPTED, id: 'run-busy' } })
    cockpit()
    const box = await screen.findByRole('textbox', { name: 'Befehl an Willy' })
    await waitFor(() => expect(box.hasAttribute('disabled')).toBe(true))
    // Says what runs, in the operator's words ("Auftrag"), not the console's ("Lauf").
    await waitFor(() => expect(box.getAttribute('placeholder')).toBe('Gesperrt: Auftrag läuft'))
    expect(screen.getByRole('button', { name: 'Sprechen' }).hasAttribute('disabled')).toBe(true)
  })
})

describe('the first motion Start names', () => {
  const FIXED = { ...FACTS, wrist_camera: false, looks: [], cameras: [{ rig_id: 'Top_Cam', mounting: 'fixed', primary: true }] }
  const CASES = [
    ['the configured looks: look 1 first', FACTS, 'der Roboter fährt zu Blick 1', 'the robot moves to look 1'],
    ['a wrist camera with no looks configured: it looks from Home', { ...FACTS, looks: [] }, 'der Roboter fährt nach Home und schaut', 'the robot moves to Home and looks'],
    ['a fixed camera: the approach to the first grasp', FIXED, 'der Roboter fährt zum ersten Griff', 'the robot moves to its first grasp'],
  ] as const
  for (const [what, facts, de, en] of CASES) {
    for (const lang of ['de', 'en'] as const) {
      it(`${what} (${lang})`, async () => {
        server({ '/v1/cell/facts': facts })
        cockpit(lang)
        await command('Nimm den grünen Würfel')
        const motion = lang === 'de' ? de : en
        const start = await screen.findByRole('button', { name: new RegExp(`^Start – ${motion}`) })
        expect(start).toBeTruthy()
      })
    }
  }

  it('is the configured looks wherever they are, and only "the robot moves" while the cell\'s facts are unknown', () => {
    expect(firstMotion({ wrist_camera: false, looks: ['look_1'] })).toBe('look')
    expect(firstMotion({ wrist_camera: true, looks: [] })).toBe('homeLook')
    expect(firstMotion({ wrist_camera: false, looks: [] })).toBe('grasp')
    expect(firstMotion(null)).toBe('unknown')
  })
})

describe('the stop buttons', () => {
  it('halt is one click: the brake is sent at once, with no dialog', async () => {
    const calls = server({ '/v1/cell': { ...CELL, active_run_id: 'run-busy' }, '/v1/runs/run-busy': { ...ACCEPTED, id: 'run-busy' } })
    cockpit()
    const halt = await screen.findByRole('button', { name: /Sofort anhalten/ })
    fireEvent.click(halt)
    await waitFor(() => expect(sent(calls, 'POST /v1/cell/brake')).toHaveLength(1))
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(moved(calls)).toHaveLength(0)
  })

  it('raises no "press the e-stop" where the arm does not brake, however long the halt waits', async () => {
    server({ '/v1/cell': { ...CELL, active_run_id: 'run-h' }, '/v1/runs/run-h': { ...ACCEPTED, id: 'run-h' } })
    cockpit()
    const socket = await follow(sockets, 'run-h')
    act(() => socket.deliver([
      ev('run-h', 1, 'run_started', { kind: 'task', plan: { object: 'cube', place: { kind: 'pose', pose: 'drop_left' }, return_to: 'home', scope: 'once', first_motion: 'look', countdown: false } }),
      ev('run-h', 2, 'run_halt_requested', { reason: 'the operator pressed halt now', in_motion: true, requested_at: now() - 5, braking: false }),
    ]))
    expect(await screen.findByText('Anhalten angefordert …')).toBeTruthy()
    await new Promise((resolve) => setTimeout(resolve, 400))
    expect(screen.queryByText(/Not-Aus drücken/)).toBeNull()
    expect(screen.getByRole('button', { name: /Sofort anhalten/ }).textContent).toMatch(/hält vor der nächsten Bewegung/)
  })

  it('raises "press the e-stop" where the arm brakes and no confirmation came within 1.5 s', async () => {
    server({
      '/v1/cell': { ...CELL, active_run_id: 'run-h' },
      '/v1/runs/run-h': { ...ACCEPTED, id: 'run-h' },
      '/v1/cell/facts': { ...FACTS, brake: { latches: true, brakes_in_motion: true } },
    })
    cockpit()
    const socket = await follow(sockets, 'run-h')
    act(() => socket.deliver([
      ev('run-h', 1, 'run_started', { kind: 'task', plan: { object: 'cube', place: { kind: 'pose', pose: 'drop_left' }, return_to: 'home', scope: 'once', first_motion: 'look', countdown: false } }),
      ev('run-h', 2, 'run_halt_requested', { reason: 'the operator pressed halt now', in_motion: true, requested_at: now() - 5, braking: true }),
    ]))
    const alarm = await screen.findByRole('alert', {}, { timeout: 2000 })
    expect(alarm.textContent).toMatch(/Anhalten nicht bestätigt: Not-Aus drücken/)
    expect(screen.getByRole('button', { name: /Sofort anhalten/ }).textContent).toMatch(/bremst kontrolliert/)
  })

  it('halt is one click away whenever the cell is connected, with or without a run', async () => {
    const calls = server()
    cockpit()
    const halt = await screen.findByRole('button', { name: /Sofort anhalten/ })
    await waitFor(() => expect(halt.hasAttribute('disabled')).toBe(false))
    fireEvent.click(halt)
    await waitFor(() => expect(sent(calls, 'POST /v1/cell/brake')).toHaveLength(1))
    expect(moved(calls)).toHaveLength(0)
  })

  it('draws the halt calm while nothing runs, and in its full orange while a run is under way; one click either way', async () => {
    const calls = server()
    cockpit()
    const idle = await screen.findByRole('button', { name: /Sofort anhalten/ })
    await waitFor(() => expect(idle.hasAttribute('disabled')).toBe(false))
    expect(idle.classList.contains('halt')).toBe(true)
    expect(idle.classList.contains('calm')).toBe(true)
    fireEvent.click(idle)
    await waitFor(() => expect(sent(calls, 'POST /v1/cell/brake')).toHaveLength(1))
    cleanup()
    server({ '/v1/cell': { ...CELL, active_run_id: 'run-busy' }, '/v1/runs/run-busy': { ...ACCEPTED, id: 'run-busy' } })
    cockpit()
    const busy = await screen.findByRole('button', { name: /Sofort anhalten/ })
    await waitFor(() => expect(busy.classList.contains('calm')).toBe(false))
    expect(busy.classList.contains('halt')).toBe(true)
  })

  it('keeps Home off while the arm is halted, and while a stop waits for "Zelle ist frei", saying why', async () => {
    const latch = { reason: 'the operator pressed halt now', requested_at: now() - 3, in_motion: false, braked: false, brake: 'none' }
    server({ '/v1/cell': { ...CELL, halted: latch } })
    cockpit()
    const strip = await screen.findByRole('region', { name: 'Anhalten und Home' })
    const home = within(strip).getByRole('button', { name: /^Home/ })
    await waitFor(() => expect(home.getAttribute('title')).toMatch(/Der Arm ist angehalten/))
    expect(home.hasAttribute('disabled')).toBe(true)
    cleanup()
    server({
      '/v1/cell': { ...CELL, recovery: { run_id: HALTED, kind: 'task', stop_code: 'failed_in_a_row', at: 10, holding: false, cleared_at: null } },
      '/v1/cell/readiness': { ready: false, lights: LIGHTS, blockers: [{ code: 'cell_not_cleared', message: '' }] },
      [`/v1/runs/${HALTED}`]: halted.runs[HALTED],
    })
    cockpit()
    const again = within(await screen.findByRole('region', { name: 'Anhalten und Home' })).getByRole('button', { name: /^Home/ })
    await waitFor(() => expect(again.getAttribute('title')).toMatch(/Erst bestätigen, dass die Zelle frei ist/))
    expect(again.hasAttribute('disabled')).toBe(true)
  })

  it('keeps "press the e-stop" when the arm\'s "brake not confirmed" comes in after the halted run ended', async () => {
    const stopped = { run_id: 'run-u', kind: 'task', stop_code: 'halted', at: now() - 1, holding: false, cleared_at: null }
    const latch = { reason: 'the operator pressed halt now', requested_at: now() - 4, in_motion: true, braked: false, brake: 'unconfirmed' }
    const plan = { object: 'cube', place: { kind: 'pose', pose: 'drop_left' }, return_to: 'home', scope: 'once', first_motion: 'look', countdown: false }
    server({
      '/v1/cell': { ...CELL, active_run_id: 'run-u', halted: latch, recovery: stopped },
      '/v1/cell/readiness': { ready: false, lights: [light('robot', 'blocked', 'halted'), ...LIGHTS.slice(1)], blockers: [{ code: 'cell_not_cleared', message: '' }] },
      '/v1/cell/facts': { ...FACTS, brake: { latches: true, brakes_in_motion: true } },
      '/v1/runs/run-u': { ...ACCEPTED, id: 'run-u', plan },
    })
    cockpit()
    const socket = await follow(sockets, 'run-u')
    act(() => socket.deliver([
      ev('run-u', 1, 'run_started', { kind: 'task', plan }),
      ev('run-u', 2, 'run_halt_requested', { reason: 'the operator pressed halt now', in_motion: true, requested_at: now() - 4, braking: true }),
      ev('run-u', 3, 'run_error', { stop_code: 'halted', error: 'halted' }),
      ev('run-u', 4, 'run_finished', { ...ACCEPTED, id: 'run-u', state: 'failed', stop_code: 'halted', stop_class: 'problem', plan }),
    ]))
    const alarm = await waitFor(() => {
      const found = screen.getAllByRole('alert').find((el) => /Not-Aus drücken/.test(el.textContent ?? ''))
      if (!found) throw new Error('no alarm')
      return found
    })
    expect(alarm.textContent).toMatch(/Der Arm meldet das Bremsen als nicht bestätigt/)
    // The stop card does not claim the arm stands where it stopped: the arm said it was not seen to stop.
    const card = await screen.findByRole('region', { name: /Angehalten/ })
    expect(card.textContent).not.toMatch(/der Arm steht/)
    expect(card.textContent).toMatch(/hat das Bremsen nicht bestätigt/)
  })

  it('drops "press the e-stop" once the controller says it is stopped: the e-stop was pressed', async () => {
    const stopped = { run_id: 'run-u', kind: 'task', stop_code: 'halted', at: now() - 1, holding: false, cleared_at: null }
    const latch = { reason: 'the operator pressed halt now', requested_at: now() - 4, in_motion: true, braked: false, brake: 'unconfirmed' }
    server({
      '/v1/cell': { ...CELL, halted: latch, recovery: stopped },
      '/v1/cell/readiness': { ready: false, lights: [light('robot', 'blocked', 'controller_stopped'), ...LIGHTS.slice(1)], blockers: [] },
      '/v1/cell/facts': { ...FACTS, brake: { latches: true, brakes_in_motion: true } },
      '/v1/runs/run-u': { ...ACCEPTED, id: 'run-u', state: 'failed', stop_code: 'halted', stop_class: 'problem' },
    })
    cockpit()
    const stage = await screen.findByRole('region', { name: 'Live-Bild' })
    expect(await within(stage).findByText('Steuerung gestoppt')).toBeTruthy()
    await new Promise((resolve) => setTimeout(resolve, 300))
    expect(screen.queryByText(/Not-Aus drücken/)).toBeNull()
  })

  it('reads that the arm brakes from the halt itself when the cell\'s facts could not be read', async () => {
    server({
      '/v1/cell': { ...CELL, active_run_id: 'run-h' },
      '/v1/runs/run-h': { ...ACCEPTED, id: 'run-h' },
      '/v1/cell/facts': refusal(500, 'software_error', 'facts failed'),
    })
    cockpit()
    const socket = await follow(sockets, 'run-h')
    act(() => socket.deliver([
      ev('run-h', 1, 'run_started', { kind: 'task', plan: { object: 'cube', place: { kind: 'pose', pose: 'drop_left' }, return_to: 'home', scope: 'once', first_motion: 'look', countdown: false } }),
      ev('run-h', 2, 'run_halt_requested', { reason: 'the operator pressed halt now', in_motion: true, requested_at: now() - 5, braking: true }),
    ]))
    const alarm = await screen.findByRole('alert', {}, { timeout: 2000 })
    expect(alarm.textContent).toMatch(/Anhalten nicht bestätigt: Not-Aus drücken/)
    expect(screen.getByRole('button', { name: /Sofort anhalten/ }).textContent).toMatch(/bremst kontrolliert/)
  })

  it('stop after this part asks the task to stop, and then says it waits for the part', async () => {
    const calls = server({ '/v1/cell': { ...CELL, active_run_id: 'run-s' }, '/v1/runs/run-s': { ...ACCEPTED, id: 'run-s' } })
    cockpit()
    const socket = await follow(sockets, 'run-s')
    act(() => socket.deliver([
      ev('run-s', 1, 'run_started', { kind: 'task', plan: { object: 'cube', place: { kind: 'pose', pose: 'drop_left' }, return_to: 'home', scope: 'until_empty', first_motion: 'look', countdown: false } }),
      ev('run-s', 2, 'task.part_started', { part: 1, of: null }),
    ]))
    const stop = await screen.findByRole('button', { name: /Nach diesem Teil stoppen/ })
    await waitFor(() => expect(stop.hasAttribute('disabled')).toBe(false))
    fireEvent.click(stop)
    await waitFor(() => expect(sent(calls, 'POST /v1/task/stop')).toHaveLength(1))
    expect(sent(calls, 'POST /v1/task/stop')[0].query).toBe('run_id=run-s')
    act(() => socket.deliver([ev('run-s', 3, 'run_stop_requested', { scope: 'after_part' })]))
    expect(await screen.findByText('stoppt nach diesem Teil')).toBeTruthy()
  })

  it('stops a Home run before its move is sent, and says so', async () => {
    const calls = server({ '/v1/cell': { ...CELL, active_run_id: 'run-home' }, '/v1/runs/run-home': { ...ACCEPTED, id: 'run-home', kind: 'home' } })
    cockpit()
    const socket = await follow(sockets, 'run-home')
    act(() => socket.deliver([ev('run-home', 1, 'run_started', { kind: 'home', to: 'home' }), ev('run-home', 2, 'run_countdown', { seconds_left: 3, because: 'teach' })]))
    const stop = await screen.findByRole('button', { name: /Home-Fahrt stoppen/ })
    expect(stop.textContent).toMatch(/die Bewegung wird nicht gesendet/)
    fireEvent.click(stop)
    await waitFor(() => expect(sent(calls, 'POST /v1/task/stop')).toHaveLength(1))
  })

  it('Home asks first; a cancel sends nothing, a confirm sends one planned move', async () => {
    const calls = server()
    cockpit()
    const strip = await screen.findByRole('region', { name: 'Anhalten und Home' })
    const home = within(strip).getByRole('button', { name: /^Home/ })
    await waitFor(() => expect(home.hasAttribute('disabled')).toBe(false))
    fireEvent.click(home)
    const dialog = await screen.findByRole('dialog', { name: 'Home' })
    expect(dialog.textContent).toMatch(/Erste Bewegung: geplante Fahrt nach Home/)
    fireEvent.click(within(dialog).getByRole('button', { name: 'Abbrechen' }))
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    expect(moved(calls)).toHaveLength(0)
    fireEvent.click(home)
    fireEvent.click(within(await screen.findByRole('dialog', { name: 'Home' })).getByRole('button', { name: 'Home – der Roboter fährt' }))
    await waitFor(() => expect(sent(calls, 'POST /v1/cell/home')).toHaveLength(1))
    expect(sent(calls, 'POST /v1/cell/home')[0].body).toEqual({ to: 'home' })
  })
})

describe('the stop card', () => {
  const RECORD = { run_id: HALTED, kind: 'task', stop_code: 'halted', at: 1790856001, holding: false, cleared_at: null as number | null }
  const STOPPED_CELL = { ...CELL, recovery: RECORD, halted: { reason: 'the operator pressed halt now', requested_at: 1790856000.5, in_motion: true, braked: false } }
  const STOPPED_READY = { ready: false, lights: [light('robot', 'blocked', 'halted'), ...LIGHTS.slice(1)], blockers: [{ code: 'cell_not_cleared', message: '' }, { code: 'restart_required', message: '' }] }

  it('is rebuilt from the cell\'s stop record and its run, and keeps Restart and Home off until the cell is clear', async () => {
    const calls = server({ '/v1/cell': STOPPED_CELL, '/v1/cell/readiness': STOPPED_READY, [`/v1/runs/${HALTED}`]: halted.runs[HALTED] })
    cockpit()
    const card = await screen.findByRole('region', { name: /Angehalten/ })
    expect(within(card).getByText(/keine Bewegung mehr befohlen/)).toBeTruthy()
    expect((await within(card).findByRole('button', { name: /Neustart – geplante Fahrt nach Home/ })).hasAttribute('disabled')).toBe(true)
    expect(within(card).getByRole('button', { name: /Home – geplante Fahrt nach Home/ }).hasAttribute('disabled')).toBe(true)
    fireEvent.click(within(card).getByRole('button', { name: /Zelle ist frei/ }))
    await waitFor(() => expect(sent(calls, 'POST /v1/cell/acknowledge')).toHaveLength(1))
    expect(sent(calls, 'POST /v1/cell/acknowledge')[0].body).toEqual({ cell_clear: true, jaws_empty: false })
    expect(moved(calls)).toHaveLength(0)
  })

  it('asks the cell to check the jaws, and never answers the question itself', async () => {
    const calls = server({
      '/v1/cell': { ...STOPPED_CELL, halted: null, recovery: { ...RECORD, cleared_at: RECORD.at + 5 } },
      '/v1/cell/readiness': { ready: false, lights: LIGHTS, blockers: [{ code: 'restart_required', message: '' }] },
      [`/v1/runs/${HALTED}`]: halted.runs[HALTED],
    })
    cockpit()
    const card = await screen.findByRole('region', { name: /Angehalten/ })
    const check = within(card).getByRole('button', { name: /Backen prüfen/ })
    await waitFor(() => expect(check.hasAttribute('disabled')).toBe(false))
    fireEvent.click(check)
    await waitFor(() => expect(sent(calls, 'POST /v1/cell/jaws/check')).toHaveLength(1))
    expect(calls.filter((c) => c.path === '/v1/cell/jaws/answer')).toHaveLength(0)
    expect((await within(card).findByRole('button', { name: /Neustart – geplante Fahrt nach Home/ })).hasAttribute('disabled')).toBe(true)
  })

  it('offers Home and no Restart after a pick run stopped, and says why in words that fit it', async () => {
    server({
      '/v1/cell': { ...STOPPED_CELL, halted: null, recovery: { ...RECORD, run_id: 'run-pick', kind: 'pick', stop_code: 'controller_stopped' } },
      '/v1/cell/readiness': STOPPED_READY,
      '/v1/runs/run-pick': { ...ACCEPTED, id: 'run-pick', kind: 'pick', state: 'failed', stop_code: 'controller_stopped', stop_class: 'problem' },
    })
    cockpit()
    const card = await screen.findByRole('region', { name: 'Steuerung gestoppt' })
    expect(within(card).getByText('Ein Pick-Lauf startet nicht neu: zurück geht es mit Home.')).toBeTruthy()
    expect(within(card).queryByText(/zuletzt gestoppte/)).toBeNull()
    expect(within(card).queryByRole('button', { name: /Neustart/ })).toBeNull()
    expect(within(card).getByRole('button', { name: /Home – geplante Fahrt nach Home/ })).toBeTruthy()
  })

  it('Restart, once every gate is green, asks first, names the planned move home, and sends one restart', async () => {
    const calls = server({
      '/v1/cell': { ...STOPPED_CELL, halted: null, countdown_due: true, jaws_confirmed_at: RECORD.at + 9, recovery: { ...RECORD, cleared_at: RECORD.at + 5 } },
      '/v1/cell/readiness': { ready: false, lights: LIGHTS, blockers: [{ code: 'restart_required', message: '' }] },
      [`/v1/runs/${HALTED}`]: halted.runs[HALTED],
    })
    cockpit()
    const card = await screen.findByRole('region', { name: /Angehalten/ })
    const restart = await within(card).findByRole('button', { name: /Neustart – geplante Fahrt nach Home/ })
    await waitFor(() => expect(restart.hasAttribute('disabled')).toBe(false))
    // A cancel sends nothing.
    fireEvent.click(restart)
    fireEvent.click(within(await screen.findByRole('dialog', { name: 'Neustart' })).getByRole('button', { name: 'Abbrechen' }))
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(moved(calls)).toHaveLength(0)
    fireEvent.click(restart)
    const dialog = await screen.findByRole('dialog', { name: 'Neustart' })
    expect(dialog.textContent).toMatch(/Erste Bewegung: geplante Fahrt nach Home/)
    expect(dialog.textContent).toMatch(/3 s Countdown/)
    // What follows the move is one more of the dialog's notes, in line with them.
    expect(within(dialog).getAllByRole('listitem').some((item) => /Danach geht der Auftrag weiter/.test(item.textContent ?? ''))).toBe(true)
    expect(moved(calls)).toHaveLength(0)
    fireEvent.click(within(dialog).getByRole('button', { name: 'Neustart – der Roboter fährt' }))
    await waitFor(() => expect(sent(calls, 'POST /v1/task/restart')).toHaveLength(1))
    expect(sent(calls, 'POST /v1/task/restart')[0].body).toEqual({ run_id: HALTED })
    expect(moved(calls)).toHaveLength(1)
  })

  it('names the stopped task\'s own return pose on Restart and in its dialog', async () => {
    const record = halted.runs[HALTED]
    const plan = { ...record.plan!, return_to: 'park', return_label: 'Parkposition' }
    server({
      '/v1/cell': { ...STOPPED_CELL, halted: null, jaws_confirmed_at: RECORD.at + 9, recovery: { ...RECORD, cleared_at: RECORD.at + 5 } },
      '/v1/cell/readiness': { ready: false, lights: LIGHTS, blockers: [{ code: 'restart_required', message: '' }] },
      [`/v1/runs/${HALTED}`]: { ...record, plan },
    })
    cockpit()
    const card = await screen.findByRole('region', { name: /Angehalten/ })
    const restart = await within(card).findByRole('button', { name: /Neustart – geplante Fahrt nach Parkposition/ })
    await waitFor(() => expect(restart.hasAttribute('disabled')).toBe(false))
    fireEvent.click(restart)
    const dialog = await screen.findByRole('dialog', { name: 'Neustart' })
    expect(dialog.textContent).toMatch(/Erste Bewegung: geplante Fahrt nach Parkposition/)
  })

  it('keeps Restart off until the stopped task\'s record is read, says so, and asks again', async () => {
    let answer = false
    server({
      '/v1/cell': { ...STOPPED_CELL, halted: null, jaws_confirmed_at: RECORD.at + 9, recovery: { ...RECORD, cleared_at: RECORD.at + 5 } },
      '/v1/cell/readiness': { ready: false, lights: LIGHTS, blockers: [{ code: 'restart_required', message: '' }] },
      [`/v1/runs/${HALTED}`]: () => (answer ? halted.runs[HALTED] : refusal(503, 'http_503', 'busy')),
    })
    cockpit()
    const card = await screen.findByRole('region', { name: /Angehalten/ })
    expect(await within(card).findByText(/ließ sich nicht lesen/)).toBeTruthy()
    const restart = within(card).getByRole('button', { name: /^Neustart/ })
    expect(restart.hasAttribute('disabled')).toBe(true)
    // Home goes to Home whatever the task was: it needs no record.
    await waitFor(() => expect(within(card).getByRole('button', { name: /Home – geplante Fahrt nach Home/ }).hasAttribute('disabled')).toBe(false))
    answer = true
    await waitFor(() => expect(within(card).getByRole('button', { name: /^Neustart – geplante Fahrt nach Home/ }).hasAttribute('disabled')).toBe(false), { timeout: 4000 })
  })

  it('after a reload under a stop record, draws the stopped run again: its numbers, where it stopped', async () => {
    const record = { ...halted.runs[HALTED], parts_placed: 3, attempted: 4, succeeded: 3, outcomes: ['succeeded', 'succeeded', 'succeeded', 'execution_failed'] }
    server({ '/v1/cell': STOPPED_CELL, '/v1/cell/readiness': STOPPED_READY, [`/v1/runs/${HALTED}`]: record })
    cockpit()
    // The stopped run is followed (its stream replayed from seq 0), so the chat and the numbers are its own.
    const socket = await follow(sockets, HALTED)
    expect(socket.sinceSeq).toBe(0)
    const numbers = screen.getByRole('region', { name: 'Dieser Auftrag in Zahlen' })
    await waitFor(() => expect(within(numbers).getByText('3')).toBeTruthy())
    expect(within(numbers).queryByText('noch kein Auftrag')).toBeNull()
  })

  it('says the cell is not connected under a stop record too, with the way to Setup', async () => {
    const record = { run_id: HALTED, kind: 'task', stop_code: 'disconnected', at: 10, holding: false, cleared_at: null }
    server({
      '/v1/cell': { ...CELL, state: 'built', recovery: record },
      '/v1/cell/readiness': { ready: false, lights: [light('robot', 'blocked', 'not_connected')], blockers: [] },
      [`/v1/runs/${HALTED}`]: halted.runs[HALTED],
    })
    cockpit()
    const top = await screen.findByRole('region', { name: 'Lauf' })
    expect(await within(top).findByText(/Die Zelle ist nicht verbunden/)).toBeTruthy()
    expect(within(top).getByRole('link', { name: 'Zu Einrichten' }).getAttribute('href')).toBe('/setup')
  })

  it('drops the halt banner once a person confirmed the cell clear; the card stays until the way back', async () => {
    server({
      '/v1/cell': { ...STOPPED_CELL, halted: null, recovery: { ...RECORD, cleared_at: RECORD.at + 5 } },
      '/v1/cell/readiness': { ready: false, lights: LIGHTS, blockers: [{ code: 'restart_required', message: '' }] },
      [`/v1/runs/${HALTED}`]: halted.runs[HALTED],
    })
    cockpit()
    await screen.findByRole('region', { name: /Angehalten/ })
    const stage = screen.getByRole('region', { name: 'Live-Bild' })
    await waitFor(() => expect(within(stage).queryByText('Angehalten')).toBeNull())
    cleanup()
    server({ '/v1/cell': STOPPED_CELL, '/v1/cell/readiness': STOPPED_READY, [`/v1/runs/${HALTED}`]: halted.runs[HALTED] })
    cockpit()
    const before = await screen.findByRole('region', { name: 'Live-Bild' })
    expect(await within(before).findByText('Angehalten')).toBeTruthy()
  })

  it('folds to one line while the way back runs, and the top shows that run', async () => {
    server({
      '/v1/cell': { ...STOPPED_CELL, halted: null, active_run_id: 'run-back', recovery: { ...RECORD, cleared_at: RECORD.at + 5 } },
      '/v1/cell/readiness': { ready: false, lights: LIGHTS, blockers: [{ code: 'run_active', message: '' }] },
      [`/v1/runs/${HALTED}`]: halted.runs[HALTED],
      '/v1/runs/run-back': { ...ACCEPTED, id: 'run-back', restart_of: HALTED },
    })
    cockpit()
    const socket = await follow(sockets, 'run-back')
    act(() => socket.deliver([ev('run-back', 1, 'run_started', { kind: 'task', restart_of: HALTED, plan: { object: 'cube', place: { kind: 'pose', pose: 'drop_left' }, return_to: 'home', scope: 'once', first_motion: 'return', countdown: false } })]))
    const card = await screen.findByRole('region', { name: /Angehalten/ })
    expect(await within(card).findByText(/Der Weg zurück läuft/)).toBeTruthy()
    expect(within(card).queryByRole('button')).toBeNull()
    expect(screen.getByRole('region', { name: 'Lauf' })).toBeTruthy()
  })

  // A hand that is no toggle and measures nothing (a gripper driven to a width, no sensor) on an arm that models no
  // part: only the stop record says a part may still be in the jaws, and the server keeps `part_still_held` (its
  // `part_held` rule, `api/readiness.py`) until a person says "Backen leer".
  const WIDTH = { kind: 'width', driver: 'DummyGripper', where: '', connected: true, jaws: 'not_counted', why_unknown: '', no_sensor: true, commands_sent: null }

  it('asks for "Backen leer" while the stopped run may have left a part in a hand that measures nothing, and keeps Restart and Home off until then', async () => {
    let emptied = false
    const record = () => ({ ...RECORD, holding: !emptied, cleared_at: RECORD.at + 5 })
    const cellNow = () => ({ ...STOPPED_CELL, halted: null, hand: WIDTH, payload_model: 'not_applicable', recovery: record() })
    const calls = server({
      '/v1/cell': cellNow,
      '/v1/cell/readiness': () => ({
        ready: false,
        lights: LIGHTS,
        blockers: [
          { code: 'restart_required', message: '' },
          ...(emptied ? [] : [{ code: 'part_still_held', message: 'the stopped run left a part in the jaws, and this hand measures nothing' }]),
        ],
      }),
      [`/v1/runs/${HALTED}`]: halted.runs[HALTED],
      'POST /v1/cell/acknowledge': () => {
        emptied = true
        return cellNow()
      },
    })
    cockpit()
    const card = await screen.findByRole('region', { name: /Angehalten/ })
    const empty = await within(card).findByRole('button', { name: 'Backen leer' })
    expect(within(card).getByText('Hält die Hand ein Teil: die Hand leeren, dann bestätigen.')).toBeTruthy()
    expect(within(card).queryByText('Kein Teil in der Hand.')).toBeNull()
    const restart = await within(card).findByRole('button', { name: /Neustart – geplante Fahrt nach Home/ })
    expect(restart.hasAttribute('disabled')).toBe(true)
    expect(within(card).getByRole('button', { name: /Home – geplante Fahrt nach Home/ }).hasAttribute('disabled')).toBe(true)
    // Home under the stage is off too, and says why: the server would refuse it.
    const strip = screen.getByRole('region', { name: 'Anhalten und Home' })
    await waitFor(() => expect(within(strip).getByRole('button', { name: /^Home/ }).getAttribute('title')).toBe('Ein Teil ist noch im Greifer.'))
    await waitFor(() => expect(empty.hasAttribute('disabled')).toBe(false))
    fireEvent.click(empty)
    await waitFor(() => expect(sent(calls, 'POST /v1/cell/acknowledge')).toHaveLength(1))
    expect(sent(calls, 'POST /v1/cell/acknowledge')[0].body).toEqual({ cell_clear: true, jaws_empty: true })
    // The person's word ends the record's belief: every gate is green, and still nothing has moved.
    await waitFor(() => expect(restart.hasAttribute('disabled')).toBe(false))
    expect(within(card).getByText('Kein Teil in der Hand.')).toBeTruthy()
    expect(moved(calls)).toHaveLength(0)
  })

  it('believes the stop record\'s part where the ready bar cannot be read, or was read before the record', async () => {
    const holdingCell = { ...STOPPED_CELL, halted: null, hand: WIDTH, payload_model: 'not_applicable', recovery: { ...RECORD, holding: true, cleared_at: RECORD.at + 5 } }
    server({ '/v1/cell': holdingCell, '/v1/cell/readiness': refusal(500, 'software_error', 'readiness failed'), [`/v1/runs/${HALTED}`]: halted.runs[HALTED] })
    cockpit()
    const card = await screen.findByRole('region', { name: /Angehalten/ })
    expect(await within(card).findByRole('button', { name: 'Backen leer' })).toBeTruthy()
    expect(within(card).queryByText('Kein Teil in der Hand.')).toBeNull()
    cleanup()
    // A ready bar that names no stop record at all predates it (the server lists cell_not_cleared or restart_required
    // for as long as one stands): it says nothing about the part the record believes held.
    server({ '/v1/cell': holdingCell, '/v1/cell/readiness': READY, [`/v1/runs/${HALTED}`]: halted.runs[HALTED] })
    cockpit()
    const again = await screen.findByRole('region', { name: /Angehalten/ })
    expect(await within(again).findByRole('button', { name: 'Backen leer' })).toBeTruthy()
  })

  it('never says "no part" about a hand nobody read: the cell down after a restart keeps the record\'s part', async () => {
    // After a restart of the server the record comes back uncleared and nothing is built: no hand is read, and the
    // ready bar names no part_still_held (there is no hand to hold one). The record's belief stands until a hand is.
    const none = { kind: 'none', driver: '', where: '', connected: false, jaws: 'not_counted', why_unknown: '', no_sensor: false, commands_sent: null }
    const down = { ...CELL, state: 'disconnected', arm: null, gripper: null, hand: none, payload_model: 'not_applicable', recovery: { ...RECORD, holding: true } }
    const blockers = [{ code: 'cell_not_cleared', message: '' }, { code: 'restart_required', message: '' }]
    server({
      '/v1/cell': down,
      '/v1/cell/readiness': { ready: false, lights: [light('robot', 'blocked', 'not_built')], blockers },
      [`/v1/runs/${HALTED}`]: halted.runs[HALTED],
    })
    cockpit()
    const card = await screen.findByRole('region', { name: /Angehalten/ })
    expect(await within(card).findByText(/wird nach dem Verbinden geprüft/)).toBeTruthy()
    expect(within(card).queryByText('Kein Teil in der Hand.')).toBeNull()
    expect(within(card).queryByRole('button', { name: 'Backen leer' })).toBeNull()
    expect(within(card).getByRole('button', { name: /Zelle ist frei/ }).hasAttribute('disabled')).toBe(true)
    cleanup()
    // The record believed no part: that is what the card says, and still nothing about a hand it cannot read.
    server({
      '/v1/cell': { ...down, recovery: { ...RECORD, holding: false } },
      '/v1/cell/readiness': { ready: false, lights: [light('robot', 'blocked', 'not_built')], blockers },
      [`/v1/runs/${HALTED}`]: halted.runs[HALTED],
    })
    cockpit()
    const calm = await screen.findByRole('region', { name: /Angehalten/ })
    expect(await within(calm).findByText('Kein Teil in der Hand.')).toBeTruthy()
  })

  it('says a refused Restart above the buttons, where the person is looking, and brings it into view', async () => {
    const shown = vi.fn()
    const had = Object.getOwnPropertyDescriptor(Element.prototype, 'scrollIntoView')
    Object.defineProperty(Element.prototype, 'scrollIntoView', { configurable: true, writable: true, value: shown })
    try {
      server({
        '/v1/cell': { ...STOPPED_CELL, halted: null, jaws_confirmed_at: RECORD.at + 9, recovery: { ...RECORD, cleared_at: RECORD.at + 5 } },
        '/v1/cell/readiness': { ready: false, lights: LIGHTS, blockers: [{ code: 'restart_required', message: '' }] },
        [`/v1/runs/${HALTED}`]: halted.runs[HALTED],
        'POST /v1/task/restart': refusal(409, 'part_still_held', 'the gripper measures a part in its jaws'),
      })
      cockpit()
      const card = await screen.findByRole('region', { name: /Angehalten/ })
      const restart = await within(card).findByRole('button', { name: /Neustart – geplante Fahrt nach Home/ })
      await waitFor(() => expect(restart.hasAttribute('disabled')).toBe(false))
      fireEvent.click(restart)
      fireEvent.click(within(await screen.findByRole('dialog', { name: 'Neustart' })).getByRole('button', { name: 'Neustart – der Roboter fährt' }))
      const banner = await within(card).findByRole('alert')
      expect(banner.textContent).toMatch(/Ein Teil ist noch im Greifer\./)
      // Above Restart and Home, not under the card's foot where the chat cuts it off.
      expect(banner.compareDocumentPosition(restart) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
      await waitFor(() => expect(shown.mock.contexts.some((el) => el instanceof Element && el.contains(banner))).toBe(true))
      // The code and the HTTP status are a detail of the demo view: folded, one click away.
      expect(within(banner).getByText(/part_still_held · HTTP 409/).closest('details')).not.toBeNull()
    } finally {
      if (had) Object.defineProperty(Element.prototype, 'scrollIntoView', had)
      else delete (Element.prototype as { scrollIntoView?: unknown }).scrollIntoView
    }
  })

  it('starts nothing by itself after a stop', async () => {
    const calls = server({ '/v1/cell': { ...CELL, active_run_id: HALTED }, [`/v1/runs/${HALTED}`]: { ...halted.runs[HALTED], state: 'running', stop_code: '', stop_class: '' } })
    cockpit()
    const socket = await follow(sockets, HALTED)
    act(() => socket.deliver(eventsOf(halted, HALTED)))
    expect(await screen.findByText('Problem: Angehalten.')).toBeTruthy()
    await new Promise((resolve) => setTimeout(resolve, 300))
    expect(moved(calls)).toHaveLength(0)
    expect(calls.filter((c) => c.path === '/v1/cell/acknowledge')).toHaveLength(0)
  })
})

describe('the chat', () => {
  async function replayTwoParts(view: 'demo' | 'tech' = 'demo', fresh = false) {
    localStorage.setItem('willy.view', view)
    server({ '/v1/cell': { ...CELL, active_run_id: TWO }, [`/v1/runs/${TWO}`]: { ...twoParts.runs[TWO], state: 'running', stop_code: '', stop_class: '', finished_at: null } })
    cockpit()
    const socket = await follow(sockets, TWO)
    const events = eventsOf(twoParts, TWO).map((e) => (fresh ? { ...e, ts: now() } : e))
    act(() => socket.deliver(events))
  }

  it('draws a card per part with what the pick saw: looks fused, faces, the push, a hold nobody measured', async () => {
    await replayTwoParts()
    const first = (await screen.findByText('Teil 1')).closest('.ck-part') as HTMLElement
    expect(within(first).getByText('2 Blicke zusammengeführt')).toBeTruthy()
    expect(within(first).getByText('eine Backenfläche gesehen')).toBeTruthy()
    expect(within(first).getByText('Halten: nicht gemessen (kein Sensor)')).toBeTruthy()
    expect(within(first).getByText(/abgelegt · 21 s/)).toBeTruthy()
    const second = screen.getByText('Teil 2').closest('.ck-part') as HTMLElement
    expect(within(second).getByText('30 mm geschoben')).toBeTruthy()
  })

  it('draws the looks that found nothing more as one quiet line, not as a part that failed', async () => {
    await replayTwoParts()
    expect(await screen.findByText('Nichts mehr gefunden (2 Blicke).')).toBeTruthy()
    expect(screen.queryByText('Teil 3')).toBeNull()
    expect(screen.queryByText('2 Griffe')).toBeNull()
    expect(screen.queryByText('nicht abgelegt')).toBeNull()
  })

  it('counts real grasps only on a part\'s card: a look that saw nothing first is no grasp', async () => {
    server({ '/v1/cell': { ...CELL, active_run_id: 'run-g' }, '/v1/runs/run-g': { ...ACCEPTED, id: 'run-g' } })
    cockpit()
    const socket = await follow(sockets, 'run-g')
    act(() => socket.deliver([
      ev('run-g', 1, 'run_started', { kind: 'task', plan: { object: 'cube', place: { kind: 'pose', pose: 'drop_left' }, return_to: 'home', scope: 'until_empty', first_motion: 'look', countdown: false } }),
      ev('run-g', 2, 'task.part_started', { part: 1, of: null }),
      ev('run-g', 3, 'pick_result', { part: 1, pick: 1, succeeded: false, found_nothing: true, outcome: 'no_target' }),
      ev('run-g', 4, 'task.nothing_found', { part: 1, empty_in_a_row: 1, only_excluded: false }),
      ev('run-g', 5, 'task.part_started', { part: 1, of: null }),
      ev('run-g', 6, 'pick_result', { part: 1, pick: 2, succeeded: true, outcome: 'succeeded', hold_measured: false }),
      ev('run-g', 7, 'task.placed', { outcome: 'executed', no_sensor: true, line_out_refused: false }),
      ev('run-g', 8, 'task.part_finished', { part: 1, placed: true, duration_s: 12 }),
    ]))
    const log = screen.getByRole('log', { name: 'Gesprächsverlauf' })
    const card = (await within(log).findByText('Teil 1')).closest('.ck-part') as HTMLElement
    await waitFor(() => expect(within(card).getByText(/abgelegt · 12 s/)).toBeTruthy())
    expect(within(card).getByText('Halten: nicht gemessen (kein Sensor)')).toBeTruthy()
    // One grasp, one empty look before it: the card names no "2 Griffe".
    expect(within(card).queryByText(/Griff/)).toBeNull()
  })

  it('stays where the operator scrolled to read, while the card and the cell are redrawn', async () => {
    // A window where the card overflows the chat by less than 160 px: 100 px. The operator's bubble at 80 px.
    let writes = 0
    const restore = layoutLike(
      {
        offsetTop: (el) => (el.classList.contains('ck-cards') ? 90 : el.classList.contains('me') ? 80 : 0),
        scrollHeight: (el) => (el.classList.contains('ck-flow') ? 700 : 0),
        clientHeight: (el) => (el.classList.contains('ck-flow') ? 600 : 0),
      },
      () => {
        writes += 1
      },
    )
    try {
      server()
      cockpit()
      await command('Räum alle grünen Würfel auf die Ablage links')
      const card = await screen.findByRole('region', { name: 'Verstanden' })
      const log = screen.getByRole('log', { name: 'Gesprächsverlauf' })
      await waitFor(() => expect(log.scrollTop).toBe(80))
      // The card is edited and the cell is polled: neither is a new line, so the panel leaves the scroll alone (a
      // write, even of the same place, would also stop a finger's fling on a tablet).
      const opened = writes
      fireEvent.change(within(card).getByRole('textbox', { name: 'Greifen' }), { target: { value: 'red cube' } })
      await new Promise((resolve) => setTimeout(resolve, 2300))
      expect(writes).toBe(opened)
      // The operator scrolls up to read their own words again, and the card and the cell are redrawn once more.
      log.scrollTop = 0
      fireEvent.scroll(log)
      fireEvent.change(within(card).getByRole('textbox', { name: 'Greifen' }), { target: { value: 'green cube' } })
      await new Promise((resolve) => setTimeout(resolve, 2300))
      expect(log.scrollTop).toBe(0)
    } finally {
      restore()
    }
  })

  it('keeps the reader\'s place a little above the newest lines while the run goes on', async () => {
    const run = await liveTwoParts()
    const size = { scroll: 2000, client: 500 }
    const { place, scrollTo } = measuredLog(size)
    scrollTo(1500)
    // Up by 100 px, less than a screenful, to read the last card again: that is the reader's place now.
    scrollTo(1400)
    expect(await screen.findByRole('button', { name: 'Zu den neuesten Zeilen' })).toBeTruthy()
    size.scroll = 2600
    run.partTwo()
    await within(screen.getByRole('log', { name: 'Gesprächsverlauf' })).findByText('Teil 2')
    expect(place.top).toBe(1400)
  })

  /** A run that is live, its first part delivered; the rest is the test's to deliver. */
  async function liveTwoParts() {
    server({ '/v1/cell': { ...CELL, active_run_id: TWO }, [`/v1/runs/${TWO}`]: { ...twoParts.runs[TWO], state: 'running', stop_code: '', stop_class: '', finished_at: null } })
    cockpit()
    const socket = await follow(sockets, TWO)
    const events = eventsOf(twoParts, TWO).map((e) => ({ ...e, ts: now() }))
    const second = events.findIndex((e) => e.type === 'task.part_started' && (e.data as { part?: number }).part === 2)
    act(() => socket.deliver(events.slice(0, second)))
    await within(screen.getByRole('log', { name: 'Gesprächsverlauf' })).findByText('Teil 1')
    const placed = events.findIndex((e, index) => index > second && e.type === 'task.placed')
    return {
      rest: () => act(() => socket.deliver(events.slice(second))),
      /** Part 2 up to its grasp, then the rest of the run. */
      partTwo: () => act(() => socket.deliver(events.slice(second, placed))),
      end: () => act(() => socket.deliver(events.slice(placed))),
    }
  }

  /**
   * A layout as a browser would measure it, for every element at once (jsdom lays nothing out): `offsetTop`,
   * `scrollHeight` and `clientHeight` by the rules given, and a `scrollTop` each element keeps as it is set. For a card
   * that opens in the same render as the lines above it, before a test could measure that one element. `wrote` hears
   * every write of the chat's scroll position. Returns the undo.
   */
  function layoutLike(
    rules: Partial<Record<'offsetTop' | 'scrollHeight' | 'clientHeight', (el: HTMLElement) => number>>,
    wrote: () => void = () => undefined,
  ) {
    const proto = HTMLElement.prototype as unknown as Record<string, unknown>
    const props = ['offsetTop', 'scrollHeight', 'clientHeight', 'scrollTop'] as const
    const saved = new Map(props.map((prop) => [prop, Object.getOwnPropertyDescriptor(proto, prop)]))
    const tops = new WeakMap<HTMLElement, number>()
    for (const prop of ['offsetTop', 'scrollHeight', 'clientHeight'] as const) {
      const rule = rules[prop]
      Object.defineProperty(proto, prop, { configurable: true, get(this: HTMLElement) { return rule ? rule(this) : 0 } })
    }
    Object.defineProperty(proto, 'scrollTop', {
      configurable: true,
      get(this: HTMLElement) { return tops.get(this) ?? 0 },
      set(this: HTMLElement, value: number) {
        tops.set(this, value)
        if (this.classList.contains('ck-flow')) wrote()
      },
    })
    return () => {
      for (const [prop, descriptor] of saved) {
        if (descriptor) Object.defineProperty(proto, prop, descriptor)
        else delete proto[prop]
      }
    }
  }

  it('opens a card from the start of its turn: the operator\'s words at the top of the window, the card under them', async () => {
    // The turn and its card are taller than the chat's window: the operator's bubble at 300 px, the card at 420 px,
    // 1400 px of flow in a 500 px window.
    const restore = layoutLike({
      offsetTop: (el) => (el.classList.contains('ck-cards') ? 420 : el.classList.contains('me') ? 300 : 0),
      scrollHeight: (el) => (el.classList.contains('ck-flow') ? 1400 : 0),
      clientHeight: (el) => (el.classList.contains('ck-flow') ? 500 : 0),
    })
    try {
      server()
      cockpit()
      await command('Räum alle grünen Würfel auf die Ablage links')
      await screen.findByRole('region', { name: 'Verstanden' })
      const log = screen.getByRole('log', { name: 'Gesprächsverlauf' })
      // Not the bottom (900 px), where the operator's own words would be scrolled away: they start the view.
      await waitFor(() => expect(log.scrollTop).toBe(300))
    } finally {
      restore()
    }
  })

  /** The log's scroll as a browser would measure it (jsdom lays nothing out): the top is clamped to the content. */
  function measuredLog(size: { scroll: number; client: number }) {
    const log = screen.getByRole('log', { name: 'Gesprächsverlauf' })
    const place = { top: 0 }
    Object.defineProperty(log, 'scrollHeight', { configurable: true, get: () => size.scroll })
    Object.defineProperty(log, 'clientHeight', { configurable: true, get: () => size.client })
    Object.defineProperty(log, 'scrollTop', {
      configurable: true,
      get: () => place.top,
      set: (value: number) => {
        place.top = Math.max(0, Math.min(value, size.scroll - size.client))
      },
    })
    const scrollTo = (top: number) => {
      place.top = top
      fireEvent.scroll(log)
    }
    return { place, scrollTo }
  }

  it('follows the newest lines again once the reader scrolls back down to the end', async () => {
    const run = await liveTwoParts()
    const size = { scroll: 2000, client: 500 }
    const { place, scrollTo } = measuredLog(size)
    // Up to read, then back down with the wheel: 40 px short of the end is the end for a hand on a wheel, while the
    // run keeps adding lines under it.
    scrollTo(600)
    scrollTo(1460)
    size.scroll = 2600
    run.rest()
    await within(screen.getByRole('log', { name: 'Gesprächsverlauf' })).findByText('Teil 2')
    await waitFor(() => expect(place.top).toBe(2100))
  })

  it('offers the way back to the newest lines while the reader is away, and follows them again on a click', async () => {
    const run = await liveTwoParts()
    const size = { scroll: 2000, client: 500 }
    const { place, scrollTo } = measuredLog(size)
    scrollTo(1500)
    expect(screen.queryByRole('button', { name: 'Zu den neuesten Zeilen' })).toBeNull()
    scrollTo(400)
    const back = await screen.findByRole('button', { name: 'Zu den neuesten Zeilen' })
    // Away, the reader keeps their place while the run goes on.
    size.scroll = 2600
    run.partTwo()
    await within(screen.getByRole('log', { name: 'Gesprächsverlauf' })).findByText('Teil 2')
    expect(place.top).toBe(400)
    fireEvent.click(back)
    expect(place.top).toBe(2100)
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Zu den neuesten Zeilen' })).toBeNull())
    // Back at the newest lines, the flow follows them again as the run goes on.
    size.scroll = 3200
    run.end()
    await within(screen.getByRole('log', { name: 'Gesprächsverlauf' })).findByText('Was soll ich als Nächstes tun?')
    await waitFor(() => expect(place.top).toBe(2700))
  })

  for (const [lang, looking, found, missing] of [
    ['de', 'Ich suche das Ziel „in die blaue Kiste“ (2 Blicke).', 'Ziel gefunden (0,8); 3 Teile gesehen.', 'Ziel nicht gefunden: „in die blaue Kiste“.'],
    ['en', 'Looking for the target “into the blue bin” (2 looks).', 'Target found (0.8); 3 parts seen.', 'Target not found: “into the blue bin”.'],
  ] as const) {
    it(`says the camera's search in words that read right around the operator's own, a preposition and all (${lang})`, async () => {
      server({ '/v1/cell': { ...CELL, active_run_id: 'run-s' }, '/v1/runs/run-s': { ...ACCEPTED, id: 'run-s' } })
      cockpit(lang)
      const socket = await follow(sockets, 'run-s')
      const said = lang === 'de' ? 'in die blaue Kiste' : 'into the blue bin'
      const plan = { object: 'red cube', place: { kind: 'camera', phrase: 'blue bin', said }, return_to: 'home', scope: 'once', first_motion: 'look', countdown: false }
      act(() => socket.deliver([
        ev('run-s', 1, 'run_started', { kind: 'task', plan }),
        ev('run-s', 2, 'task.survey_started', { phrase: 'blue bin', looks: ['look_1', 'look_2'] }),
        ev('run-s', 3, 'task.target_found', { target: { label: 'blue bin', score: 0.8, look: 'look_1', overlay: null }, look: 'look_1', parts_seen: 3 }),
        ev('run-s', 4, 'task.target_missing', { phrase: 'blue bin', looks_tried: 2 }),
      ]))
      const log = screen.getByRole('log', { name: lang === 'de' ? 'Gesprächsverlauf' : 'Conversation history' })
      expect(await within(log).findByText(looking)).toBeTruthy()
      expect(within(log).getByText(found)).toBeTruthy()
      expect(within(log).getByText(missing)).toBeTruthy()
    })
  }

  it('says that a Home run stops before its move is sent, and that a pose the task did not screen ahead is judged as it runs', async () => {
    localStorage.setItem('willy.view', 'tech')
    server({ '/v1/cell': { ...CELL, active_run_id: 'run-h' }, '/v1/runs/run-h': { ...ACCEPTED, id: 'run-h', kind: 'home' } })
    cockpit()
    const home = await follow(sockets, 'run-h')
    act(() => home.deliver([
      ev('run-h', 1, 'run_started', { kind: 'home', to: 'home' }),
      ev('run-h', 2, 'run_stop_requested', { scope: 'before_motion' }),
    ]))
    const log = screen.getByRole('log', { name: 'Gesprächsverlauf' })
    expect(await within(log).findByText('Ich stoppe vor der Fahrt; eine schon gesendete Fahrt endet erst.')).toBeTruthy()
    cleanup()
    sockets = fakeSockets()
    server({ '/v1/cell': { ...CELL, active_run_id: 'run-u' }, '/v1/runs/run-u': { ...ACCEPTED, id: 'run-u' } })
    cockpit()
    const task = await follow(sockets, 'run-u')
    act(() => task.deliver([
      ev('run-u', 1, 'run_started', { kind: 'task', plan: { object: 'cube', place: { kind: 'pose', pose: 'drop_left', pose_label: 'Ablage links' }, return_to: 'home', scope: 'once', first_motion: 'look', countdown: false } }),
      ev('run-u', 2, 'task.pose_screened', { pose: 'drop_left', role: 'place', verdict: 'unscreened', detail: '', nearby_deg: null }),
    ]))
    const taskLog = screen.getByRole('log', { name: 'Gesprächsverlauf' })
    expect(await within(taskLog).findByText('Pose „Ablage links“ nicht vorab geprüft: die Fahrt prüft sie, wenn sie läuft.')).toBeTruthy()
    expect(within(taskLog).queryByText(/geprüft: nicht geprüft/)).toBeNull()
  })

  it('says the end and asks for the next instruction, and a live end hands the focus to the input', async () => {
    await replayTwoParts('demo', true)
    expect(await screen.findByText('Fertig, nichts mehr da: 2 Teile abgelegt.')).toBeTruthy()
    expect(screen.getByText('Was soll ich als Nächstes tun?')).toBeTruthy()
    await waitFor(() => expect(document.activeElement).toBe(screen.getByRole('textbox', { name: 'Befehl an Willy' })))
  })

  it('says what to do while the conversation is empty', async () => {
    server()
    cockpit()
    expect(await screen.findByText(/Sag oder tipp, was Willy tun soll/)).toBeTruthy()
  })

  it('shows the session\'s earlier tasks in a tab of its own, one line each, read from the server', async () => {
    // A second tab, or the browser opened again: its own conversation is empty, but the server keeps the session's
    // runs. The chat is the history, so each ended task reads as its one line, the command as it was said.
    const calls = server({ '/v1/runs': [twoParts.runs[TWO], { ...halted.runs[HALTED] }, { ...ACCEPTED, id: 'run-home', kind: 'home', state: 'finished', stop_code: 'finished', stop_class: 'done' }] })
    cockpit()
    const log = screen.getByRole('log', { name: 'Gesprächsverlauf' })
    expect(await within(log).findByText('Räum alle grünen Würfel auf die Ablage links · 2 Teile · Nichts mehr da')).toBeTruthy()
    expect(within(log).getByText(/· Angehalten$/)).toBeTruthy()
    // Tasks only: a Home move is no earlier task.
    expect(within(log).queryByText(/Fahrt nach|Home-Fahrt/)).toBeNull()
    expect(calls.filter((c) => c.path === '/v1/runs')[0].query).toBe('limit=200')
    expect(moved(calls)).toHaveLength(0)
  })

  it('draws the run\'s own end and the next question as Willy\'s words, not as steps', async () => {
    await replayTwoParts()
    const end = (await screen.findByText('Fertig, nichts mehr da: 2 Teile abgelegt.')).closest('.ck-bubble') as HTMLElement
    expect(end.className).toContain('willy')
    expect(within(end).getByText('Willy')).toBeTruthy()
    const next = screen.getByText('Was soll ich als Nächstes tun?').closest('.ck-bubble') as HTMLElement
    expect(next.className).toContain('willy')
    // The question follows the end in the same turn of Willy's: his name is not said twice.
    expect(within(next).queryByText('Willy')).toBeNull()
  })

  it('shows the backend\'s own sentence under the lines in the tech view only', async () => {
    await replayTwoParts('tech')
    expect(await screen.findAllByText('Task run finished: nothing_left.')).toHaveLength(1)
    // The tech view's numbers add the pushes and a count per outcome (plan 4.2, Stats).
    const numbers = screen.getByRole('region', { name: 'Dieser Auftrag in Zahlen' })
    expect(within(numbers).getByText('Geschoben')).toBeTruthy()
    expect(within(numbers).getByText(/^gegriffen 2 · kein Ziel \d$/)).toBeTruthy()
    cleanup()
    sockets = fakeSockets()
    await replayTwoParts('demo')
    await screen.findByText('Fertig, nichts mehr da: 2 Teile abgelegt.')
    expect(screen.queryByText('Task run finished: nothing_left.')).toBeNull()
  })
})

describe('the stage', () => {
  it('shows the live image with its badge, its camera and its age', async () => {
    server()
    cockpit()
    const stage = await screen.findByRole('region', { name: 'Live-Bild' })
    expect(await within(stage).findByRole('img', { name: 'Live-Bild der Kamera EIH_Cam' })).toBeTruthy()
    expect(within(stage).getByText('LIVE')).toBeTruthy()
    expect(within(stage).getByText('EIH_Cam')).toBeTruthy()
    expect(within(stage).getByText('0,2 s')).toBeTruthy()
  })

  it('says PROBE for the rehearsal\'s drawing, and KEIN BILD with the reason when there is none', async () => {
    server({ '/v1/camera/live': () => live({ source: 'synthetic', rig_id: null, rigs: [] }) })
    cockpit()
    const stage = await screen.findByRole('region', { name: 'Live-Bild' })
    expect(await within(stage).findByText('PROBE')).toBeTruthy()
    cleanup()
    server({ '/v1/camera/live': () => live({ source: 'none', reason: 'not_built', image_base64: null, rig_id: null, rigs: [] }) })
    cockpit()
    const blank = await screen.findByRole('region', { name: 'Live-Bild' })
    expect(await within(blank).findByText('KEIN BILD')).toBeTruthy()
    expect(within(blank).getByText('Die Zelle ist nicht aufgebaut.')).toBeTruthy()
  })

  it('keeps the last frame while the cell measures, and says so', async () => {
    let calls = 0
    server({ '/v1/camera/live': () => (++calls < 2 ? live() : live({ reason: 'measuring', image_base64: null })) })
    cockpit()
    const stage = await screen.findByRole('region', { name: 'Live-Bild' })
    expect(await within(stage).findByText('misst …', {}, { timeout: 2000 })).toBeTruthy()
    expect(within(stage).getByRole('img', { name: 'Live-Bild der Kamera EIH_Cam' })).toBeTruthy()
  })

  it('offers the cameras when the cell has more than one, and asks for the one chosen', async () => {
    const rigs = [{ rig_id: 'EIH_Cam', mounting: 'wrist', primary: true }, { rig_id: 'Top_Cam', mounting: 'fixed', primary: false }]
    const asked: string[] = []
    server({ '/v1/camera/live': (call: ApiCall) => { asked.push(call.query); return live({ rigs, rig_id: call.query.includes('rig=Top_Cam') ? 'Top_Cam' : 'EIH_Cam' }) } })
    cockpit()
    const stage = await screen.findByRole('region', { name: 'Live-Bild' })
    const top = await within(stage).findByRole('button', { name: /Top_Cam/ })
    fireEvent.click(top)
    await waitFor(() => expect(asked.some((q) => q.includes('rig=Top_Cam'))).toBe(true))
    await waitFor(() => expect(within(stage).getByRole('button', { name: /Top_Cam/ }).getAttribute('aria-pressed')).toBe('true'))
  })

  async function graspDecided() {
    server({ '/v1/cell': { ...CELL, active_run_id: 'run-o' }, '/v1/runs/run-o': { ...ACCEPTED, id: 'run-o' } })
    cockpit()
    const socket = await follow(sockets, 'run-o')
    act(() => socket.deliver([
      ev('run-o', 1, 'run_started', { kind: 'task', plan: { object: 'cube', place: { kind: 'pose', pose: 'drop_left' }, return_to: 'home', scope: 'once', first_motion: 'look', countdown: false } }),
      ev('run-o', 2, 'task.part_started', { part: 1, of: 1 }),
      ev('run-o', 3, 'pick.executing', { position_mm: [1, 2, 3], overlay: '/v1/runs/run-o/overlays/1' }),
    ]))
    return screen.findByRole('region', { name: 'Live-Bild' })
  }

  it('pins a fresh grasp overlay over the stage as its own image in the tech view, saying when it was decided', async () => {
    localStorage.setItem('willy.view', 'tech')
    const stage = await graspDecided()
    const pinned = await within(stage).findByRole('img', { name: /GRIFF/ })
    expect(pinned.getAttribute('src')).toBe('/v1/runs/run-o/overlays/1')
    expect(within(stage).getByText(/wie entschieden .*; der Arm hat sich seither bewegt/)).toBeTruthy()
  })

  it('keeps the grasp render, a debug image, off the stage in the demo view: it is the part card\'s thumbnail', async () => {
    const stage = await graspDecided()
    expect(await screen.findByRole('img', { name: 'Griff von Teil 1, wie entschieden' })).toBeTruthy()
    await new Promise((resolve) => setTimeout(resolve, 100))
    expect(within(stage).queryByRole('img', { name: /GRIFF/ })).toBeNull()
  })

  it('pins what the camera found as the target over the stage in the demo view too, saying when it was seen', async () => {
    server({ '/v1/cell': { ...CELL, active_run_id: 'run-t' }, '/v1/runs/run-t': { ...ACCEPTED, id: 'run-t' } })
    cockpit()
    const socket = await follow(sockets, 'run-t')
    act(() => socket.deliver([
      ev('run-t', 1, 'run_started', { kind: 'task', plan: { object: 'green cube', place: { kind: 'camera', phrase: 'blue bin', said: 'die blaue Kiste' }, return_to: 'home', scope: 'once', first_motion: 'look', countdown: false } }),
      ev('run-t', 2, 'task.survey_started', { phrase: 'blue bin', looks: ['look_1'] }),
      ev('run-t', 3, 'task.target_found', {
        target: { label: 'blue bin', score: 0.91, centre_mm: [400, -100, 80], rim_mm: 80, footprint_mm: [300, 200], opening_mm: [260, 160], look: 'look_1', seen_at: now(), overlay: '/v1/runs/run-t/target/overlay' },
        look: 'look_1',
        parts_seen: 3,
      }),
    ]))
    const stage = await screen.findByRole('region', { name: 'Live-Bild' })
    const pinned = await within(stage).findByRole('img', { name: /ZIEL/ })
    expect(pinned.getAttribute('src')).toBe('/v1/runs/run-t/target/overlay')
    expect(within(stage).getByText(/wie gesehen .*; der Arm hat sich seither bewegt/)).toBeTruthy()
    // The first grasp, decided a moment later, is the tech view's to pin: the target keeps its moment on the stage.
    act(() => socket.deliver([
      ev('run-t', 4, 'task.part_started', { part: 1, of: 1 }),
      ev('run-t', 5, 'pick.executing', { position_mm: [1, 2, 3], overlay: '/v1/runs/run-t/overlays/1' }),
    ]))
    expect(await screen.findByRole('img', { name: 'Griff von Teil 1, wie entschieden' })).toBeTruthy()
    expect(within(stage).getByRole('img', { name: /ZIEL/ })).toBeTruthy()
    expect(within(stage).queryByRole('img', { name: /GRIFF/ })).toBeNull()
  })

  it('goes back to the cell\'s own camera when the one chosen is gone', async () => {
    const both = [{ rig_id: 'EIH_Cam', mounting: 'wrist', primary: true }, { rig_id: 'Top_Cam', mounting: 'fixed', primary: false }]
    let gone = false
    const asked: string[] = []
    server({
      '/v1/camera/live': (call: ApiCall) => {
        asked.push(call.query)
        const top = call.query.includes('rig=Top_Cam')
        if (top && gone) return live({ source: 'none', reason: 'no_rig', image_base64: null, rig_id: 'Top_Cam', rigs: both.slice(0, 1) })
        return live({ rigs: gone ? both.slice(0, 1) : both, rig_id: top ? 'Top_Cam' : 'EIH_Cam' })
      },
    })
    cockpit()
    const stage = await screen.findByRole('region', { name: 'Live-Bild' })
    fireEvent.click(await within(stage).findByRole('button', { name: /Top_Cam/ }))
    await waitFor(() => expect(asked.some((q) => q.includes('rig=Top_Cam'))).toBe(true))
    gone = true
    const before = asked.length
    await waitFor(() => expect(asked.slice(before).some((q) => !q.includes('rig=Top_Cam'))).toBe(true), { timeout: 2000 })
    expect(await within(stage).findByRole('img', { name: 'Live-Bild der Kamera EIH_Cam' })).toBeTruthy()
  })
})

describe('the stage\'s pace', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  async function countFor(teaching: boolean): Promise<number[]> {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
    const live$ = vi.spyOn(api, 'live').mockImplementation(async () => live() as never)
    renderWith(<Stage teaching={teaching} overlays={[]} wrist banner={null} pinGrasps={false} />, { providers: false })
    await vi.advanceTimersByTimeAsync(1)
    const counts = [live$.mock.calls.length]
    for (let i = 0; i < 2; i += 1) {
      await vi.advanceTimersByTimeAsync(1000)
      counts.push(live$.mock.calls.length)
    }
    cleanup()
    live$.mockRestore()
    vi.useRealTimers()
    return counts
  }

  it('plays the camera as a stream and asks only twice a second for what the badges say', async () => {
    // The first answer arrives through the poll; from then on the stream plays and the poll keeps the badges.
    const [first, after1, after2] = await countFor(false)
    expect(first).toBe(1)
    expect(after2 - after1).toBe(2)
    expect(after1 - first).toBeLessThanOrEqual(3)
    const [t0, , t2] = await countFor(true)
    expect(t0).toBe(1)
    expect(t2 - t0).toBeLessThanOrEqual(6)
  })
})

describe('the run header and the timeline', () => {
  async function perceiving(lang: Lang, object = 'green cube') {
    server({ '/v1/cell': { ...CELL, active_run_id: 'run-p' }, '/v1/runs/run-p': { ...ACCEPTED, id: 'run-p' } })
    cockpit(lang)
    const socket = await follow(sockets, 'run-p')
    act(() => socket.deliver([
      ev('run-p', 1, 'run_started', { kind: 'task', plan: { object, object_said: null, place: { kind: 'pose', pose: 'drop_left', pose_label: 'Ablage links' }, return_to: 'home', scope: 'until_empty', first_motion: 'look', countdown: false, options: { multi_view: true } } }),
      ev('run-p', 2, 'task.part_started', { part: 1, of: null }),
      ev('run-p', 3, 'pick.pick_started', { attempt_total: 5 }),
      ev('run-p', 4, 'pick.attempt_started', { attempt: 0, attempt_total: 5 }),
      ev('run-p', 5, 'pick.perceived', { look: 'look_1', segmentation_count: 3 }),
    ]))
    return screen.findByRole('list', { name: lang === 'de' ? 'Ablauf' : 'Steps' })
  }

  it('fills one step at a time: the look in progress; detecting rides along unfilled', async () => {
    const steps = await perceiving('de')
    await waitFor(() => expect(within(steps).getAllByRole('listitem').filter((item) => item.getAttribute('aria-current') === 'step')).toHaveLength(1))
    const current = within(steps).getAllByRole('listitem').find((item) => item.getAttribute('aria-current') === 'step')
    expect(current?.textContent).toMatch(/Schauen/)
    expect(current?.textContent).toMatch(/Blick 1\/3/)
  })

  it('names the look as a view in English, not "Look Look 1"', async () => {
    const steps = await perceiving('en')
    await waitFor(() => expect(within(steps).getByText('view 1/3')).toBeTruthy())
    expect(within(steps).queryByText(/Look 1/)).toBeNull()
  })

  it('starts the run\'s headline with a capital, whatever word it begins with', async () => {
    await perceiving('de', '')
    const top = await screen.findByRole('region', { name: 'Lauf' })
    expect(await within(top).findByText('Alles, was die Kamera sieht → Ablage links')).toBeTruthy()
  })
})

describe('voice output', () => {
  function installSpeech() {
    const spoken: Array<{ text: string; lang: string }> = []
    class Utterance {
      text: string
      lang = ''
      voice: unknown = null
      rate = 1
      constructor(text: string) {
        this.text = text
      }
    }
    vi.stubGlobal('SpeechSynthesisUtterance', Utterance)
    vi.stubGlobal('speechSynthesis', {
      speak: (u: Utterance) => spoken.push({ text: u.text, lang: u.lang }),
      cancel: () => undefined,
      getVoices: () => [],
    })
    return spoken
  }

  async function finishATask() {
    server({ '/v1/cell': { ...CELL, active_run_id: TWO }, [`/v1/runs/${TWO}`]: { ...twoParts.runs[TWO], state: 'running', stop_code: '', stop_class: '', finished_at: null } })
    cockpit()
    const socket = await follow(sockets, TWO)
    act(() => socket.deliver(eventsOf(twoParts, TWO).map((e) => ({ ...e, ts: now() }))))
    await screen.findByText('Fertig, nichts mehr da: 2 Teile abgelegt.')
  }

  it('is off by default: nothing is said', async () => {
    const spoken = installSpeech()
    await finishATask()
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(spoken).toEqual([])
  })

  it('switched on, says the start, the end and the next question in German', async () => {
    const spoken = installSpeech()
    localStorage.setItem('willy.voiceOut', 'on')
    await finishATask()
    await waitFor(() => expect(spoken.map((s) => s.text)).toEqual(['Ich greife alle grünen Würfel.', 'Fertig, 2 Teile.', 'Was soll ich als Nächstes tun?']))
    expect(spoken.every((s) => s.lang === 'de-DE')).toBe(true)
  })
})

describe('the ready bar', () => {
  it('starts the planner on its light\'s action, which moves nothing, and then shows the planner\'s run', async () => {
    const calls = server({
      '/v1/cell/readiness': { ready: false, blockers: [], lights: [LIGHTS[0], LIGHTS[1], light('planner', 'blocked', 'off'), LIGHTS[3], LIGHTS[4], LIGHTS[5]] },
      'POST /v1/cell/planner': reply(202, { ...ACCEPTED, id: 'run-planner', kind: 'planner' }),
    })
    cockpit()
    const bar = await screen.findByRole('region', { name: 'Bereitschaft der Zelle' })
    // Written as a word, drawn in capitals by the style (the banner's chips): the catalogs never shout.
    expect(within(bar).getByText('Nicht bereit')).toBeTruthy()
    fireEvent.click(await within(bar).findByRole('button', { name: 'Planer starten' }))
    await waitFor(() => expect(sent(calls, 'POST /v1/cell/planner')).toHaveLength(1))
    // The planner's run is followed at once: the top says it, and nothing that moves was sent.
    expect(await screen.findByText('Der Planer startet')).toBeTruthy()
    expect(moved(calls)).toHaveLength(0)
  })

  it('names the language model where it is missing, and says the card is then filled in by hand', async () => {
    server({ '/v1/cell/readiness': { ready: true, blockers: [], lights: [...LIGHTS.slice(0, 5), light('commands', 'info', 'missing')] } })
    cockpit()
    const bar = await screen.findByRole('region', { name: 'Bereitschaft der Zelle' })
    const row = (await within(bar).findByText('Sprachmodell')).closest('li') as HTMLElement
    expect(within(row).getByText('Fehlt')).toBeTruthy()
    expect(within(row).getByText('Karte von Hand')).toBeTruthy()
    expect(within(bar).queryByText('Befehle')).toBeNull()
  })

  it('offers no "Planer starten" on a cell that is not built: the way there is Setup', async () => {
    const calls = server({
      '/v1/cell': { ...CELL, state: 'disconnected' },
      '/v1/cell/readiness': { ready: false, blockers: [], lights: [light('robot', 'blocked', 'not_built'), light('planner', 'blocked', 'off'), light('commands', 'info', 'idle')] },
    })
    cockpit()
    const bar = await screen.findByRole('region', { name: 'Bereitschaft der Zelle' })
    expect(await within(bar).findByRole('link', { name: 'Zu Einrichten' })).toBeTruthy()
    await within(bar).findByText('Aus')
    expect(within(bar).queryByRole('button', { name: 'Planer starten' })).toBeNull()
    expect(calls.filter((c) => c.path === '/v1/cell/planner')).toHaveLength(0)
  })

  it('asks the cell where the jaws stand on the gripper light\'s action, and answers nothing itself', async () => {
    const calls = server({
      '/v1/cell': { ...CELL, hand: { ...HAND, jaws: 'unknown' } },
      '/v1/cell/readiness': { ready: false, blockers: [], lights: [LIGHTS[0], LIGHTS[1], LIGHTS[2], light('gripper', 'blocked', 'jaws_unknown'), LIGHTS[4], LIGHTS[5]] },
    })
    cockpit()
    const bar = await screen.findByRole('region', { name: 'Bereitschaft der Zelle' })
    fireEvent.click(await within(bar).findByRole('button', { name: 'Backen prüfen' }))
    await waitFor(() => expect(sent(calls, 'POST /v1/cell/jaws/check')).toHaveLength(1))
    expect(calls.filter((c) => c.path === '/v1/cell/jaws/answer')).toHaveLength(0)
    expect(moved(calls)).toHaveLength(0)
  })

  it('offers "Zelle ist frei" for an arm a Disconnect left latched, while the cell is only built', async () => {
    const latched = { reason: 'the cell was disconnected while a run was moving the arm', requested_at: 5, in_motion: true, braked: false }
    const calls = server({
      '/v1/cell': { ...CELL, state: 'built', halted: latched },
      '/v1/cell/readiness': { ready: false, lights: [light('robot', 'blocked', 'not_connected')], blockers: [] },
    })
    cockpit()
    const bar = await screen.findByRole('region', { name: 'Bereitschaft der Zelle' })
    fireEvent.click(await within(bar).findByRole('button', { name: 'Zelle ist frei' }))
    await waitFor(() => expect(sent(calls, 'POST /v1/cell/acknowledge')).toHaveLength(1))
    expect(sent(calls, 'POST /v1/cell/acknowledge')[0].body).toEqual({ cell_clear: true, jaws_empty: false })
  })

  it('lets the stop card take "Zelle ist frei" while the cell is only built (acknowledge is allowed then)', async () => {
    const record = { run_id: HALTED, kind: 'task', stop_code: 'disconnected', at: 10, holding: false, cleared_at: null }
    const calls = server({
      '/v1/cell': { ...CELL, state: 'built', recovery: record },
      '/v1/cell/readiness': { ready: false, lights: [light('robot', 'blocked', 'not_connected')], blockers: [] },
      [`/v1/runs/${HALTED}`]: halted.runs[HALTED],
    })
    cockpit()
    const card = await screen.findByRole('region', { name: 'Getrennt' })
    const clear = within(card).getByRole('button', { name: /Zelle ist frei/ })
    await waitFor(() => expect(clear.hasAttribute('disabled')).toBe(false))
    fireEvent.click(clear)
    await waitFor(() => expect(sent(calls, 'POST /v1/cell/acknowledge')).toHaveLength(1))
    // Restart and Home stay off: a built cell is not connected.
    expect((await within(card).findByRole('button', { name: /Neustart – geplante Fahrt nach Home/ })).hasAttribute('disabled')).toBe(true)
  })
})

describe('the ask card', () => {
  const LOST = runOf(targetLost, (record) => record.stop_code === 'target_lost')

  async function lostTheBin(over: Record<string, Route> = {}) {
    const calls = server({
      '/v1/cell': { ...CELL, active_run_id: LOST },
      [`/v1/runs/${LOST}`]: { ...targetLost.runs[LOST], state: 'running', stop_code: '', stop_class: '' },
      ...over,
    })
    cockpit()
    const socket = await follow(sockets, LOST)
    act(() => socket.deliver(eventsOf(targetLost, LOST)))
    const card = await screen.findByRole('region', { name: 'Rückfrage: Ziel verloren' })
    return { calls, card }
  }

  it('names the countdown and the put-back fallback on each option that moves, as Start does', async () => {
    const { card } = await lostTheBin({ '/v1/cell': { ...CELL, active_run_id: LOST, countdown_due: true } })
    const again = within(card).getByRole('button', { name: /Nochmal suchen – der Roboter fährt zu Blick 1/ })
    await waitFor(() => expect(again.textContent).toContain('zuerst 3 s Countdown „Hände weg“'))
    expect(again.textContent).toMatch(/Findet er das Ziel nicht, legt er das Teil zurück und fragt/)
    const pose = within(card).getByRole('button', { name: /Standard-Ablage nehmen – der Roboter fährt zu Blick 1/ })
    expect(pose.textContent).toContain('zuerst 3 s Countdown „Hände weg“')
    // A pose place has no target to lose.
    expect(pose.textContent).not.toMatch(/Findet er das Ziel nicht/)
  })

  it('keeps the options that move off while the cell is not ready; the ones that move nothing stay on', async () => {
    const notReady = { ready: false, blockers: [], lights: [...LIGHTS.filter((l) => l.id !== 'planner'), light('planner', 'wait', 'starting')] }
    const { calls, card } = await lostTheBin({ '/v1/cell/readiness': notReady })
    const again = within(card).getByRole('button', { name: /Nochmal suchen/ })
    await new Promise((resolve) => setTimeout(resolve, 100))
    expect(again.hasAttribute('disabled')).toBe(true)
    expect(within(card).getByRole('button', { name: /Standard-Ablage nehmen/ }).hasAttribute('disabled')).toBe(true)
    expect(within(card).getByRole('button', { name: 'Anderes Ziel' }).hasAttribute('disabled')).toBe(false)
    fireEvent.click(again)
    expect(moved(calls)).toHaveLength(0)
  })

  it('says why, and each option that moves is a Start of its own naming the first motion', async () => {
    const { calls, card } = await lostTheBin()
    expect(within(card).getByText('Das Ziel wurde zu weit verschoben.')).toBeTruthy()
    expect(moved(calls)).toHaveLength(0)
    const again = within(card).getByRole('button', { name: /Nochmal suchen – der Roboter fährt zu Blick 1/ })
    await waitFor(() => expect(again.hasAttribute('disabled')).toBe(false))
    fireEvent.click(again)
    await waitFor(() => expect(sent(calls, 'POST /v1/task')).toHaveLength(1))
    expect(sent(calls, 'POST /v1/task')[0].body).toMatchObject({ object: 'red cube', place: { kind: 'camera', phrase: 'blue bin' }, scope: 'once' })
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('puts the part into the default place on its own Start, opens the card for another target, and ends quietly', async () => {
    const { calls, card } = await lostTheBin()
    fireEvent.click(within(card).getByRole('button', { name: 'Anderes Ziel' }))
    const edited = await screen.findByRole('region', { name: 'Auftrag von Hand' })
    expect((within(edited).getByRole('textbox', { name: 'Greifen' }) as HTMLInputElement).value).toBe('red cube')
    expect(moved(calls)).toHaveLength(0)
    fireEvent.click(within(card).getByRole('button', { name: /Standard-Ablage nehmen – der Roboter fährt zu Blick 1/ }))
    await waitFor(() => expect(sent(calls, 'POST /v1/task')).toHaveLength(1))
    expect(sent(calls, 'POST /v1/task')[0].body).toMatchObject({ place: { kind: 'pose', pose: null } })
  })

  it('closes on "Beenden" and asks for the next instruction, moving nothing', async () => {
    const { calls, card } = await lostTheBin()
    fireEvent.click(within(card).getByRole('button', { name: 'Beenden' }))
    await waitFor(() => expect(screen.queryByRole('region', { name: 'Rückfrage: Ziel verloren' })).toBeNull())
    expect(screen.getByText('Beendet. Was soll ich als Nächstes tun?')).toBeTruthy()
    expect(moved(calls)).toHaveLength(0)
  })
})

describe('the Advanced drawer and the numbers', () => {
  it('takes a push of 10 mm or more only, as the server does, and says the range', async () => {
    const calls = server()
    cockpit()
    await command('Räum alle grünen Würfel auf die Ablage links')
    const card = await screen.findByRole('region', { name: 'Verstanden' })
    fireEvent.click(within(card).getByText('Erweitert'))
    const push = within(card).getByRole('spinbutton', { name: /Schieben/ }) as HTMLInputElement
    expect(push.min).toBe('10')
    expect(within(card).getByText(/10 bis 50 mm/)).toBeTruthy()
    fireEvent.change(push, { target: { value: '5' } })
    fireEvent.blur(push)
    await waitFor(() => expect(push.value).toBe('10'))
    const start = within(card).getByRole('button', { name: /^Start/ })
    await waitFor(() => expect(start.hasAttribute('disabled')).toBe(false))
    fireEvent.click(start)
    await waitFor(() => expect(sent(calls, 'POST /v1/task')).toHaveLength(1))
    expect(sent(calls, 'POST /v1/task')[0].body).toMatchObject({ options: { push_mm: 10 } })
  })

  it('says critical parts are cleared, not pushed, and sends the switch only where it was touched', async () => {
    // The owner's switch (2026-10-03): the cell's own word stands until the operator turns it.
    const calls = server()
    cockpit()
    await command('Räum alle grünen Würfel auf die Ablage links')
    const card = await screen.findByRole('region', { name: 'Verstanden' })
    fireEvent.click(within(card).getByText('Erweitert'))
    const critical = within(card).getByRole('checkbox', { name: /Kritische Teile/ }) as HTMLInputElement
    expect(critical.checked).toBe(false)
    expect(within(card).getByRole('spinbutton', { name: /Schieben/ })).toBeTruthy()
    fireEvent.click(critical)
    await waitFor(() => expect(critical.checked).toBe(true))
    expect(within(card).queryByRole('spinbutton', { name: /Schieben/ })).toBeNull()
    expect(within(card).getByText(/kritische Teile: nur wegräumen/)).toBeTruthy()
    const start = within(card).getByRole('button', { name: /^Start/ })
    await waitFor(() => expect(start.hasAttribute('disabled')).toBe(false))
    fireEvent.click(start)
    await waitFor(() => expect(sent(calls, 'POST /v1/task')).toHaveLength(1))
    expect(sent(calls, 'POST /v1/task')[0].body).toMatchObject({ options: { critical_parts: true, push_mm: null } })
  })

  it('sends where a blocker goes only where it was touched', async () => {
    // The owner's switch (2026-10-06): a blocker goes where the parts go, or is only set aside.
    const calls = server()
    cockpit()
    await command('Räum alle grünen Würfel auf die Ablage links')
    const card = await screen.findByRole('region', { name: 'Verstanden' })
    fireEvent.click(within(card).getByText('Erweitert'))
    const into = within(card).getByRole('checkbox', { name: /Blocker direkt wegpacken/ }) as HTMLInputElement
    expect(into.checked).toBe(true)
    fireEvent.click(into)
    await waitFor(() => expect(into.checked).toBe(false))
    const start = within(card).getByRole('button', { name: /^Start/ })
    await waitFor(() => expect(start.hasAttribute('disabled')).toBe(false))
    fireEvent.click(start)
    await waitFor(() => expect(sent(calls, 'POST /v1/task')).toHaveLength(1))
    expect(sent(calls, 'POST /v1/task')[0].body).toMatchObject({ options: { blocker_into_the_place: false } })
  })

  it("leaves the cell's switch to the cell where nobody touched it", async () => {
    const calls = server()
    cockpit()
    await command('Räum alle grünen Würfel auf die Ablage links')
    const card = await screen.findByRole('region', { name: 'Verstanden' })
    const start = within(card).getByRole('button', { name: /^Start/ })
    await waitFor(() => expect(start.hasAttribute('disabled')).toBe(false))
    fireEvent.click(start)
    await waitFor(() => expect(sent(calls, 'POST /v1/task')).toHaveLength(1))
    expect(sent(calls, 'POST /v1/task')[0].body).toMatchObject({
      options: { critical_parts: null, blocker_into_the_place: null },
    })
  })

  it('brings Start back into the chat\'s view when Advanced opens above it', async () => {
    const shown: Element[] = []
    const scroll = vi.fn(function (this: Element) {
      shown.push(this)
    })
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, writable: true, value: scroll })
    try {
      server()
      cockpit()
      await command('Räum alle grünen Würfel auf die Ablage links')
      const card = await screen.findByRole('region', { name: 'Verstanden' })
      const drawer = within(card).getByText('Erweitert').closest('details') as HTMLDetailsElement
      drawer.open = true
      fireEvent(drawer, new Event('toggle'))
      await waitFor(() => expect(scroll).toHaveBeenCalled())
      expect(scroll).toHaveBeenLastCalledWith(expect.objectContaining({ block: 'nearest' }))
      expect(within(shown.at(-1) as HTMLElement).getByRole('button', { name: /^Start/ })).toBeTruthy()
    } finally {
      delete (HTMLElement.prototype as unknown as Record<string, unknown>).scrollIntoView
    }
  })

  it('reads number first in every tile, the success rate leading', async () => {
    server()
    cockpit()
    const numbers = await screen.findByRole('region', { name: 'Dieser Auftrag in Zahlen' })
    const tiles = Array.from(numbers.children) as HTMLElement[]
    expect(tiles.length).toBeGreaterThanOrEqual(3)
    for (const tile of tiles) expect(tile.firstElementChild?.className).toMatch(/\bbig\b|\bck-stat-num\b/)
    expect(tiles[0].textContent).toMatch(/^–\s*Erfolgsquote/)
  })
})

describe('without a loaded reader', () => {
  it('offers "Laden" on the card, which moves nothing, and reads the sentence again on a click', async () => {
    let loaded = false
    const calls = server({
      'POST /v1/commands/parse': () => (loaded ? PARSED : refusal(409, 'vlm_not_loaded', 'load it first')),
      'POST /v1/commands/warmup': () => {
        loaded = true
        return { state: 'ready', model_id: 'Qwen/Qwen3-VL-4B-Instruct', weights_present: true, cause: '', last_latency_ms: null, shared_with_detection: false }
      },
    })
    cockpit()
    await command('Räum alle grünen Würfel auf die Ablage links')
    const manual = await screen.findByRole('region', { name: 'Auftrag von Hand' })
    expect(within(manual).getByText(/Das Sprachmodell ist nicht geladen/)).toBeTruthy()
    fireEvent.click(within(manual).getByRole('button', { name: 'Laden' }))
    await waitFor(() => expect(sent(calls, 'POST /v1/commands/warmup')).toHaveLength(1))
    expect(await screen.findByText('Das Sprachmodell ist geladen.')).toBeTruthy()
    fireEvent.click(within(screen.getByRole('region', { name: 'Auftrag von Hand' })).getByRole('button', { name: 'Satz nochmal lesen' }))
    await screen.findByRole('region', { name: 'Verstanden' })
    expect(sent(calls, 'POST /v1/commands/parse')).toHaveLength(2)
    expect(moved(calls)).toHaveLength(0)
  })
})

describe('a greeting in the chat (the owner, 2026-10-06: "Sofort winken", the app config may ask first)', () => {
  const GREETED = { ...PARSED, intent: 'none', object: null, place_pose: null, scope: null }

  it('waves at once where the app config says direct, and opens no card', async () => {
    const calls = server({ 'POST /v1/commands/parse': { ...GREETED, greeting: 'direct' } })
    cockpit()
    await command('Hallo Willy!')
    expect(await screen.findByText('Hallo!')).toBeTruthy()
    await waitFor(() => expect(sent(calls, 'POST /v1/cell/wave')).toHaveLength(1))
    expect(sent(calls, 'POST /v1/task')).toHaveLength(0)
    expect(screen.queryByRole('region', { name: 'Verstanden' })).toBeNull()
  })

  it("asks first where the app config says confirm, and waves only on the dialog's button", async () => {
    const calls = server({ 'POST /v1/commands/parse': { ...GREETED, greeting: 'confirm' } })
    cockpit()
    await command('Hallo Willy!')
    expect(await screen.findByText('Hallo! Soll ich winken?')).toBeTruthy()
    const dialog = await screen.findByRole('dialog', { name: 'Winken' })
    expect(sent(calls, 'POST /v1/cell/wave')).toHaveLength(0)
    fireEvent.click(within(dialog).getByRole('button', { name: 'Winken – der Roboter bewegt sich' }))
    await waitFor(() => expect(sent(calls, 'POST /v1/cell/wave')).toHaveLength(1))
  })

  it('sends nothing where the dialog is cancelled', async () => {
    const calls = server({ 'POST /v1/commands/parse': { ...GREETED, greeting: 'confirm' } })
    cockpit()
    await command('Hallo Willy!')
    const dialog = await screen.findByRole('dialog', { name: 'Winken' })
    fireEvent.click(within(dialog).getByRole('button', { name: 'Abbrechen' }))
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Winken' })).toBeNull())
    expect(moved(calls)).toHaveLength(0)
  })

  it('only greets back where the app config says off, and moves nothing', async () => {
    const calls = server({ 'POST /v1/commands/parse': { ...GREETED, greeting: 'off' } })
    cockpit()
    await command('Hallo Willy!')
    expect(await screen.findByText('Hallo! Schön, dass du da bist.')).toBeTruthy()
    expect(moved(calls)).toHaveLength(0)
  })

  it('says a refused wave in the chat, and tries it once', async () => {
    const calls = server({
      'POST /v1/commands/parse': { ...GREETED, greeting: 'direct' },
      'POST /v1/cell/wave': refusal(409, 'restart_required', 'a stop record stands'),
    })
    cockpit()
    await command('Hallo Willy!')
    expect(await screen.findByText(/Winken geht gerade nicht/)).toBeTruthy()
    expect(sent(calls, 'POST /v1/cell/wave')).toHaveLength(1)
  })

  it('reads a sentence with no greeting as ever: a task card, and no wave', async () => {
    const calls = server()
    cockpit()
    await command('Räum alle grünen Würfel auf die Ablage links')
    await screen.findByRole('region', { name: 'Verstanden' })
    expect(sent(calls, 'POST /v1/cell/wave')).toHaveLength(0)
  })
})
