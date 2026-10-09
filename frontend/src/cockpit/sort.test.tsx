/**
 * A sort through the console (the owner, 2026-10-09: "Grüne Teile in die gelbe Kiste, rote in die blaue"): each kind of
 * part goes to its rule's place, four rules at most, the task's own fields the first.
 *
 * The card takes every rule from the reading or from a run's plan, and Start sends them all (`more_rules`); a card of
 * one rule sends exactly the body it always sent. Start stays off where a rule names no part or two rules name one
 * kind. The card shows the further rules as rows a person edits, removes and adds, three at most. Enter starts a clean
 * sort with every rule and opens the card for a note on any rule. The run says every rule at its start, files each
 * gripped part under its rule and counts it there once it is placed, says a bin found again or found nowhere, and the
 * parts no rule took; the run header shows the rules while a sort runs, the rule of the part in hand lit.
 */

import { act, cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { CommandOut, TaskPlanOut } from '../api/client'
import type { RunEvent } from '../api/events'
import { translate } from '../i18n'
import { clearConversation } from '../model/chat'
import { ConfirmProvider } from '../model/confirm'
import { DEFAULT_TASK, type TaskPrefs } from '../model/prefs'
import { currentRule, replay, sortRules, type ChatLine, type RunView } from '../model/runModel'
import { fakeSockets, renderWith, reply, stubApi, type ApiCall, type FakeSockets, type Route } from '../test/render'
import { voiceFor } from './announce'
import AskCard from './AskCard'
import Cockpit from './Cockpit'
import { doubts, draftFromPlan, draftFromReading, draftProblems, newRule, taskOf, taskOfPlan, type Draft } from './draft'
import { COCKPIT } from './i18n'

// ── what the reader and the server answer ───────────────────────────────────────────────────────────────────────

const SENTENCE = 'Grüne Teile in die gelbe Kiste, rote in die blaue'
const SAID = { text: SENTENCE, source: 'typed', language: 'de' } as const

const route = (prompt: string) => ({ prompt, route: 'simple', reason: 'english', description: 'simple (english)', runnable: true })
const GREEN = { phrase: 'green part', said: 'Grüne Teile', verified: true, route: route('green part') }
const YELLOW = { phrase: 'yellow bin', said: 'in die gelbe Kiste', verified: true, route: route('yellow bin') }
const RED = { phrase: 'red part', said: 'rote', verified: true, route: route('red part') }
const BLUE = { phrase: 'blue bin', said: 'in die blaue', verified: true, route: route('blue bin') }

/** The reader's answer to the owner's sentence: two rules, clean. */
function sortReading(over: Partial<CommandOut> = {}): CommandOut {
  return {
    understood: true,
    intent: 'task',
    object: GREEN,
    which: null,
    source: null,
    place: YELLOW,
    place_pose: null,
    rules: [
      { object: GREEN, which: null, source: null, place: YELLOW, place_pose: null, notes: [] },
      { object: RED, which: null, source: null, place: BLUE, place_pose: null, notes: [] },
    ],
    scope: 'until_empty',
    count: null,
    return_to: null,
    notes: [],
    reason: '',
    model: { model_id: 'Qwen/Qwen3-VL-4B-Instruct', latency_ms: 3600, attempts: 1, loaded_now: false, remembered: false },
    raw: '{"intent": "task"}',
    greeting: null,
    startable: true,
    ...over,
  }
}

/** A reading of one kind as the server answers it since 2026-10-09: its one rule listed too. */
const ONE_KIND = sortReading({ rules: [{ object: GREEN, which: null, source: null, place: YELLOW, place_pose: null, notes: [] }] })

const camera = (phrase: string, said: string | null) => ({ kind: 'camera', pose: null, pose_label: null, pose_joints_deg: null, phrase, said })

/** The sort as the server resolved it (`TaskPlanOut`). */
const PLAN = {
  object: 'green part',
  object_said: 'Grüne Teile',
  which: '',
  source: '',
  place: camera('yellow bin', 'in die gelbe Kiste'),
  more_rules: [{ object: 'red part', object_said: 'rote', which: '', source: '', place: camera('blue bin', 'in die blaue') }],
  return_to: 'home',
  return_label: null,
  scope: 'until_empty',
  options: { multi_view: true, every_look: false, both_faces: false, push_asked: false, record_views: true, rim_air_mm: 20, carry: null, pick_anything: false, overlay: true },
  command: { text: SENTENCE, source: 'typed', language: 'de', parsed: true, edited: [] },
  first_motion: 'look',
  countdown: false,
} as unknown as TaskPlanOut

const ENTER: TaskPrefs = { ...DEFAULT_TASK, start: 'enter', scope: 'until_empty' }

// ── the draft: every rule from the reading or the plan, and out with the task ──────────────────────────────────────

describe('the draft of a sort', () => {
  it('takes its further rules from the reading: each kind with its words and route, its place, its notes', () => {
    const card = draftFromReading(sortReading(), SAID, ENTER)
    expect(card.moreRules).toEqual([
      {
        object: 'red part',
        objectSaid: 'rote',
        objectVerified: true,
        objectRoute: route('red part'),
        which: '',
        source: '',
        place: { kind: 'camera', phrase: 'blue bin', said: 'in die blaue' },
        placeVerified: true,
        placeRoute: route('blue bin'),
        notes: [],
      },
    ])
    expect(draftFromReading(ONE_KIND, SAID, ENTER).moreRules).toEqual([])
    expect(draftFromReading({ ...ONE_KIND, rules: undefined } as unknown as CommandOut, SAID, ENTER).moreRules).toEqual([])
  })

  it('takes a further rule\'s taught pose by its name, and the settings\' place where the rule names none', () => {
    const task: TaskPrefs = { ...ENTER, place: { kind: 'pose', pose: 'park' } }
    const out = sortReading({
      rules: [
        { object: GREEN, place: YELLOW, notes: [] },
        { object: RED, place_pose: 'drop_left', notes: [] },
        { object: { ...RED, phrase: 'blue part', said: 'blaue' }, notes: ['place_not_in_sentence'] },
      ],
    })
    const [pose, none] = draftFromReading(out, SAID, task).moreRules
    expect(pose.place).toEqual({ kind: 'pose', pose: 'drop_left' })
    expect(none.place).toEqual({ kind: 'pose', pose: 'park' })
    expect(none.notes).toEqual(['place_not_in_sentence'])
  })

  it('takes its rules from a run\'s plan as they ran, and starts them again with every rule', () => {
    const card = draftFromPlan(PLAN)
    expect(card.moreRules).toEqual([{ ...newRule(), object: 'red part', objectSaid: 'rote', place: { kind: 'camera', phrase: 'blue bin', said: 'in die blaue' } }])
    const again = taskOfPlan(PLAN, 'same')
    expect(again.more_rules).toEqual([{ object: 'red part', object_said: 'rote', which: '', source: '', place: { kind: 'camera', phrase: 'blue bin', said: 'in die blaue' } }])
    expect(again.options).toMatchObject({ rim_air_mm: 20 })
    // The default place moves the first rule's parts alone, and keeps the others' bins (the ask card does not offer it).
    const home = taskOfPlan(PLAN, 'default')
    expect(home.place).toEqual({ kind: 'pose', pose: null })
    expect(home.more_rules).toEqual(again.more_rules)
    expect(home.options).toMatchObject({ rim_air_mm: 20 })
  })

  it('sends every rule after the task\'s own place, without blanks, and a pose rule\'s pose by its name', () => {
    const card = draftFromReading(sortReading(), SAID, ENTER)
    const body = taskOf({
      ...card,
      moreRules: [
        { ...card.moreRules[0], object: '  red part ', place: { kind: 'camera', phrase: ' blue bin  ', said: 'in die blaue' } },
        { ...newRule(), object: 'black part', place: { kind: 'pose', pose: 'drop_left' } },
        { ...newRule(), object: 'white part' },
      ],
    })
    expect(Object.keys(body)).toEqual(['object', 'object_said', 'which', 'source', 'place', 'more_rules', 'return_to', 'scope', 'options', 'command'])
    expect(body.more_rules).toEqual([
      { object: 'red part', object_said: 'rote', which: '', source: '', place: { kind: 'camera', phrase: 'blue bin', said: 'in die blaue' } },
      { object: 'black part', object_said: null, which: '', source: '', place: { kind: 'pose', pose: 'drop_left' } },
      { object: 'white part', object_said: null, which: '', source: '', place: { kind: 'pose', pose: null } },
    ])
  })

  it('sends the rim\'s air and the carry where any rule\'s place is a bin the camera finds', () => {
    const card = draftFromReading(sortReading({ object: GREEN, place: null, place_pose: 'drop_left' }), SAID, ENTER)
    expect(card.place).toEqual({ kind: 'pose', pose: 'drop_left' })
    expect(taskOf({ ...card, options: { ...card.options, rimAirMm: 30, carryOverTheRim: true } }).options).toMatchObject({ rim_air_mm: 30, carry: 'over_the_rim' })
    const poses = { ...card, moreRules: [{ ...card.moreRules[0], place: { kind: 'default' } as const }] }
    expect(taskOf({ ...poses, options: { ...poses.options, rimAirMm: 30 } }).options).toMatchObject({ rim_air_mm: null, carry: null })
  })

  it('sends a card of one rule exactly as before: no more_rules at all', () => {
    for (const out of [ONE_KIND, { ...ONE_KIND, rules: undefined } as unknown as CommandOut]) {
      const body = taskOf(draftFromReading(out, SAID, ENTER))
      expect(Object.keys(body)).toEqual(['object', 'object_said', 'which', 'source', 'place', 'return_to', 'scope', 'options', 'command'])
      expect(JSON.stringify(body)).toBe(
        JSON.stringify({
          object: 'green part',
          object_said: 'Grüne Teile',
          which: '',
          source: '',
          place: { kind: 'camera', phrase: 'yellow bin', said: 'in die gelbe Kiste' },
          return_to: 'home',
          scope: 'until_empty',
          options: {
            multi_view: true,
            every_look: false,
            both_faces: false,
            closing_axis: null,
            push_mm: null,
            critical_parts: null,
            blocker_into_the_place: null,
            rescan: null,
            push: null,
            clear: null,
            record_views: true,
            rim_air_mm: null,
            carry: null,
            pick_anything: false,
            overlay: true,
          },
          command: { text: SENTENCE, source: 'typed', language: 'de', parsed: true, edited: [] },
        }),
      )
    }
    const plan = { ...PLAN, more_rules: [] } as unknown as TaskPlanOut
    expect('more_rules' in taskOfPlan(plan)).toBe(false)
    expect(draftFromPlan({ ...plan, more_rules: undefined } as unknown as TaskPlanOut).moreRules).toEqual([])
  })

  it('keeps Start off where a rule names no part, or two rules name one kind, case and blanks aside', () => {
    const card = draftFromReading(sortReading(), SAID, ENTER)
    const context = { rehearsal: false, poses: null }
    expect(draftProblems(card, context)).toEqual([])
    expect(draftProblems({ ...card, moreRules: [{ ...card.moreRules[0], object: ' ' }] }, context)).toEqual(['ruleObject'])
    // "Anything" sorts nothing: the first rule names its kind too, ticked or not, on the rehearsal cell as well.
    expect(draftProblems({ ...card, object: '', pickAnything: true }, { rehearsal: true, poses: null })).toEqual(['ruleObject'])
    expect(draftProblems({ ...card, moreRules: [{ ...card.moreRules[0], object: ' Green   PART ' }] }, context)).toEqual(['sameKind'])
    // Two rules may share one place.
    expect(draftProblems({ ...card, moreRules: [{ ...card.moreRules[0], place: card.place }] }, context)).toEqual([])
  })

  it('judges every rule\'s place: a bin needs its phrase, a pose must be taught, the default place declared', () => {
    const card = draftFromReading(sortReading(), SAID, ENTER)
    const poses = { home: {}, poses: [{ name: 'drop_left', label: 'Ablage links' }], default_place: null } as never
    const rule = (place: Draft['place']) => ({ ...card, moreRules: [{ ...card.moreRules[0], place }] })
    expect(draftProblems(rule({ kind: 'camera', phrase: ' ', said: null }), { rehearsal: false, poses })).toEqual(['needPhrase'])
    expect(draftProblems(rule({ kind: 'pose', pose: 'gone' }), { rehearsal: false, poses })).toEqual(['unknownPose'])
    expect(draftProblems(rule({ kind: 'default' }), { rehearsal: false, poses })).toEqual(['noDefault'])
    expect(draftProblems(rule({ kind: 'pose', pose: 'drop_left' }), { rehearsal: false, poses })).toEqual([])
  })

  it('says why a sort waits for the card: each further rule\'s doubts with its number, a note once', () => {
    expect(doubts(sortReading())).toEqual([])
    const noted = sortReading({
      notes: ['object_not_in_sentence', 'retried'],
      rules: [
        { object: GREEN, place: YELLOW, notes: [] },
        { object: { ...RED, verified: false }, place: { ...BLUE, verified: false }, notes: ['object_not_in_sentence'] },
      ],
    })
    expect(doubts(noted)).toEqual([
      { key: 'note.retried' },
      { key: 'ck.doubt.rule', params: { n: 2, why: { key: 'note.object_not_in_sentence' } } },
      { key: 'ck.doubt.rule', params: { n: 2, why: { key: 'ck.doubt.object' } } },
      { key: 'ck.doubt.rule', params: { n: 2, why: { key: 'ck.doubt.place' } } },
    ])
    // The card says the further rule's note with that rule, and the reading's own as always.
    const card = draftFromReading(noted, SAID, ENTER)
    expect(card.notes).toEqual(['retried'])
    expect(card.moreRules[0].notes).toEqual(['object_not_in_sentence'])
    expect(translate('de', 'ck.doubt.rule' as never, { n: 2, why: { key: 'ck.doubt.object' } }, COCKPIT)).toBe('Regel 2: das Teil steht so nicht im Satz')
  })

  it('opens the card for a note on any rule, whatever the reading\'s startable says', () => {
    const noted = sortReading({ rules: [{ object: GREEN, place: YELLOW, notes: [] }, { object: null, place: BLUE, notes: [] }] })
    expect(doubts(noted)).toEqual([{ key: 'ck.doubt.rule', params: { n: 2, why: { key: 'ck.doubt.noPart' } } }])
    expect(doubts(sortReading({ startable: false }))).toEqual([{ key: 'ck.doubt.reading' }])
  })
})

// ── the cockpit, against a stubbed server ──────────────────────────────────────────────────────────────────────

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
const READY = {
  ready: true,
  blockers: [],
  lights: [
    light('robot', 'ok', 'connected'),
    light('cameras', 'ok', 'live'),
    light('planner', 'ok', 'ready'),
    light('gripper', 'ok', 'open_confirmed'),
    light('carried_part', 'ok', 'modelled'),
    light('commands', 'info', 'ready'),
  ],
}
const FACTS = {
  wrist_camera: true,
  cameras: [{ rig_id: 'EIH_Cam', mounting: 'wrist', primary: true }],
  looks: ['look_1', 'look_2', 'look_3'],
  natural_closing_axis: '-y',
  push: { can_push: true, why_not: '', default_mm: 30, ceiling_mm: 50 },
  detector: { backend: 'grounded_sam', router_enabled: false, vlm_model_id: null, precision: 'auto' },
  hand_eye_warn_mm: 6,
  brake: { latches: true, brakes_in_motion: false },
  payload: { modelled: true, declined_reason: null, length_mm: 40 },
  route: { route: 'planned', sentence: '' },
  rehearsal: false,
}
const POSES = {
  home: { name: 'home', label: 'Home', joints_deg: [0, -90, 90, -90, -90, 0], source: 'config', screen: 'clear', note: '', taught_at: null },
  poses: [{ name: 'drop_left', label: 'Ablage links', joints_deg: [-60, -95, -120, -55, 90, 0], source: 'taught', screen: 'clear', note: '', taught_at: null }],
  default_place: 'drop_left',
  teachable: true,
  why_not: '',
  why_not_code: '',
  target_file: 'config/robot/robot.cell.yaml',
}
const ACCEPTED = { id: 'run-new', prompt: 'green part', requested_picks: 0, state: 'running', started_at: 1, succeeded: 0, attempted: 0, error: '', stop_requested: false, kind: 'task', stop_code: '', stop_class: '', parts_placed: 0, holding: false, halt_requested: false, step: '' }

function server(over: Record<string, Route> = {}): ApiCall[] {
  return stubApi({
    '/v1/cell': CELL,
    '/v1/cell/readiness': READY,
    '/v1/cell/facts': FACTS,
    '/v1/cell/status': { state: 'connected', connected: true, simulated: false, vendor: 'ur', model: 'ur10', controller_state_included: false },
    '/v1/poses': POSES,
    '/v1/camera/live': { rig_id: 'EIH_Cam', rigs: [], source: 'camera', reason: '', image_base64: '', width: 0, height: 0, captured_at: 0, age_s: null },
    '/v1/diagnostics/route': (call: ApiCall) => route(new URLSearchParams(call.query).get('prompt') ?? ''),
    'POST /v1/commands/parse': sortReading(),
    'POST /v1/task': reply(202, ACCEPTED),
    ...over,
  })
}

const MOVES = ['POST /v1/task', 'POST /v1/task/restart', 'POST /v1/cell/home', 'POST /v1/pick', 'POST /v1/cell/wave']
const moved = (calls: ApiCall[]) => calls.filter((c) => MOVES.includes(`${c.method} ${c.path}`))
const sent = (calls: ApiCall[], key: string) => calls.filter((c) => `${c.method} ${c.path}` === key)

function cockpit() {
  return renderWith(
    <ConfirmProvider>
      <Cockpit />
    </ConfirmProvider>,
  )
}

/** Type a sentence and press Enter. */
async function command(text: string) {
  const box = await screen.findByRole('textbox', { name: 'Befehl an Willy' })
  await waitFor(() => expect(box.hasAttribute('disabled')).toBe(false))
  fireEvent.change(box, { target: { value: text } })
  fireEvent.keyDown(box, { key: 'Enter' })
}

const now = () => Date.now() / 1000

function ev(runId: string, seq: number, type: string, data: Record<string, unknown> = {}, ts = now()): RunEvent {
  return { type, run_id: runId, seq, ts, severity: 'info', human: `${type} said`, step: '', step_index: null, step_total: null, data }
}

let sockets: FakeSockets

beforeEach(() => {
  localStorage.clear()
  localStorage.setItem('willy.taskStart', 'card')
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

describe('the card of a sort', () => {
  async function card(reading: CommandOut = sortReading()) {
    const calls = server({ 'POST /v1/commands/parse': reading })
    cockpit()
    await command(SENTENCE)
    return { calls, card: await screen.findByRole('region', { name: 'Verstanden' }) }
  }

  async function startOf(card: HTMLElement, calls: ApiCall[]) {
    const start = within(card).getByRole('button', { name: /^Start/ })
    await waitFor(() => expect(start.hasAttribute('disabled')).toBe(false))
    fireEvent.click(start)
    await waitFor(() => expect(sent(calls, 'POST /v1/task')).toHaveLength(1))
    return sent(calls, 'POST /v1/task')[0].body as Record<string, unknown>
  }

  it('says every rule it understood, and shows the further ones as rows under "Regeln"', async () => {
    const { card: found } = await card()
    expect(screen.getByText('Verstanden, ich sortiere: Grüne Teile → in die gelbe Kiste · rote → in die blaue. Prüf kurz und drück auf Start.')).toBeTruthy()
    const rules = within(found).getByRole('list', { name: 'Regeln' })
    const rows = within(rules).getAllByRole('listitem')
    expect(rows).toHaveLength(2)
    expect(rows[0].textContent).toMatch(/Grüne Teile → in die gelbe Kiste/)
    expect(rows[0].textContent).toMatch(/Greifen und Ablegen oben/)
    expect((within(rows[1]).getByRole('textbox', { name: 'Teil von Regel 2' }) as HTMLInputElement).value).toBe('red part')
    expect((within(rows[1]).getByRole('combobox', { name: 'Ablage von Regel 2' }) as HTMLSelectElement).value).toBe('camera')
    expect((within(rows[1]).getByRole('textbox', { name: 'Was die Kamera für Regel 2 sucht' }) as HTMLInputElement).value).toBe('blue bin')
    expect(within(rows[1]).getByText('deine Worte: „rote“ → „in die blaue“')).toBeTruthy()
    expect(within(rows[1]).queryByText('bitte prüfen')).toBeNull()
    // The main fields are the first rule, as always; "anything" sorts nothing, so no tick is offered.
    expect((within(found).getByRole('textbox', { name: 'Greifen' }) as HTMLInputElement).value).toBe('green part')
  })

  it('marks "bitte prüfen" on a further rule where the reader did not find its words, and says its note with it', async () => {
    const { card: found } = await card(
      sortReading({
        startable: false,
        notes: ['object_not_in_sentence'],
        rules: [{ object: GREEN, place: YELLOW, notes: [] }, { object: { ...RED, verified: false }, place: { ...BLUE, verified: false }, notes: ['object_not_in_sentence'] }],
      }),
    )
    const row = within(within(found).getByRole('list', { name: 'Regeln' })).getAllByRole('listitem')[1]
    expect(within(row).getAllByText('bitte prüfen')).toHaveLength(2)
    expect(within(found).getByText('Regel 2: Was gegriffen werden soll, steht nicht im Satz.')).toBeTruthy()
  })

  it('sends every rule on Start, a row\'s edit recorded as "rules"', async () => {
    const { calls, card: found } = await card()
    fireEvent.change(within(found).getByRole('textbox', { name: 'Teil von Regel 2' }), { target: { value: 'red cube' } })
    fireEvent.change(within(found).getByRole('combobox', { name: 'Ablage von Regel 2' }), { target: { value: 'pose:drop_left' } })
    const body = await startOf(found, calls)
    expect(body).toMatchObject({
      object: 'green part',
      place: { kind: 'camera', phrase: 'yellow bin', said: 'in die gelbe Kiste' },
      more_rules: [{ object: 'red cube', object_said: 'rote', which: '', source: '', place: { kind: 'pose', pose: 'drop_left' } }],
      scope: 'until_empty',
      command: { text: SENTENCE, parsed: true, edited: ['rules'] },
    })
    expect(moved(calls)).toHaveLength(1)
  })

  it('asks how the detector routes a row\'s kind once a person changed it, and shows it on the row', async () => {
    const calls = server({ '/v1/diagnostics/route': (call: ApiCall) => ({ ...route(new URLSearchParams(call.query).get('prompt') ?? ''), route: 'vlm', reason: 'german' }) })
    cockpit()
    await command(SENTENCE)
    const found = await screen.findByRole('region', { name: 'Verstanden' })
    const row = () => within(within(found).getByRole('list', { name: 'Regeln' })).getAllByRole('listitem')[1]
    expect(within(row()).queryByText('VLM')).toBeNull()
    fireEvent.change(within(row()).getByRole('textbox', { name: 'Teil von Regel 2' }), { target: { value: 'rote Teile' } })
    const asked = () => sent(calls, 'GET /v1/diagnostics/route').map((c) => new URLSearchParams(c.query).get('prompt'))
    await waitFor(() => expect(asked()).toContain('rote Teile'))
    expect(await within(row()).findByText('VLM')).toBeTruthy()
    // The bin's phrase was not touched: its route is the reader's still, and it is not asked again.
    expect(asked()).not.toContain('blue bin')
    expect(moved(calls)).toHaveLength(0)
  })

  it('removes a rule, and a card left with one rule looks and sends as a task of one kind', async () => {
    const { calls, card: found } = await card()
    fireEvent.click(within(found).getByRole('button', { name: 'Regel 2 entfernen' }))
    expect(within(found).queryByRole('list', { name: 'Regeln' })).toBeNull()
    expect(within(found).queryByRole('button', { name: /Regel hinzufügen/ })).toBeNull()
    const body = await startOf(found, calls)
    expect('more_rules' in body).toBe(false)
    expect(body).toMatchObject({ object: 'green part', command: { edited: ['rules'] } })
  })

  it('adds rules up to three beside the first, and keeps Start off until each names its part', async () => {
    const { calls, card: found } = await card()
    const add = within(found).getByRole('button', { name: /Regel hinzufügen/ })
    fireEvent.click(add)
    fireEvent.click(add)
    const rows = within(within(found).getByRole('list', { name: 'Regeln' })).getAllByRole('listitem')
    expect(rows).toHaveLength(4)
    expect(add.hasAttribute('disabled')).toBe(true)
    expect(within(found).getByText('höchstens 3 weitere Regeln')).toBeTruthy()
    // A new rule starts at the default place, with no part: Start waits for its kind.
    expect((within(rows[3]).getByRole('combobox', { name: 'Ablage von Regel 4' }) as HTMLSelectElement).value).toBe('default')
    const start = within(found).getByRole('button', { name: /^Start/ })
    expect(start.hasAttribute('disabled')).toBe(true)
    expect(within(found).getByText('jede Regel braucht ein Teil')).toBeTruthy()
    fireEvent.change(within(rows[2]).getByRole('textbox', { name: 'Teil von Regel 3' }), { target: { value: 'black part' } })
    fireEvent.change(within(rows[3]).getByRole('textbox', { name: 'Teil von Regel 4' }), { target: { value: 'white part' } })
    const body = await startOf(found, calls)
    expect((body.more_rules as unknown[]).length).toBe(3)
    expect(body.more_rules).toEqual([
      expect.objectContaining({ object: 'red part' }),
      { object: 'black part', object_said: null, which: '', source: '', place: { kind: 'pose', pose: null } },
      { object: 'white part', object_said: null, which: '', source: '', place: { kind: 'pose', pose: null } },
    ])
  })

  it('keeps Start off while two rules name one kind, and says why', async () => {
    const { calls, card: found } = await card()
    fireEvent.change(within(found).getByRole('textbox', { name: 'Teil von Regel 2' }), { target: { value: ' Green  Part' } })
    const start = within(found).getByRole('button', { name: /^Start/ })
    await waitFor(() => expect(within(found).getByText('zwei Regeln nennen dieselben Teile')).toBeTruthy())
    expect(start.hasAttribute('disabled')).toBe(true)
    fireEvent.click(start)
    expect(moved(calls)).toHaveLength(0)
  })

  it('looks as it always did for a reading of one kind: no rules, no way to add one', async () => {
    const { card: found } = await card(ONE_KIND)
    expect(screen.getByText('Verstanden. Prüf kurz und drück auf Start.')).toBeTruthy()
    expect(within(found).queryByRole('list', { name: 'Regeln' })).toBeNull()
    expect(within(found).queryByText('Regeln')).toBeNull()
    expect(within(found).queryByRole('button', { name: /Regel hinzufügen/ })).toBeNull()
  })
})

describe('Enter and a sort (the owner, 2026-10-09)', () => {
  beforeEach(() => {
    localStorage.removeItem('willy.taskStart')
  })

  it('starts a clean sort at once with every rule, and says every rule it starts', async () => {
    const calls = server()
    cockpit()
    await command(SENTENCE)
    await waitFor(() => expect(sent(calls, 'POST /v1/task')).toHaveLength(1))
    expect(sent(calls, 'POST /v1/task')[0].body).toMatchObject({
      object: 'green part',
      place: { kind: 'camera', phrase: 'yellow bin' },
      more_rules: [{ object: 'red part', object_said: 'rote', place: { kind: 'camera', phrase: 'blue bin', said: 'in die blaue' } }],
      command: { edited: [] },
    })
    expect(await screen.findByText('Verstanden, ich sortiere: Grüne Teile → in die gelbe Kiste · rote → in die blaue. Ich fange an: der Roboter fährt zu Blick 1.')).toBeTruthy()
    expect(screen.queryByRole('region', { name: 'Verstanden' })).toBeNull()
    expect(moved(calls)).toHaveLength(1)
  })

  it('opens the card for a note on any rule, says which rule, and starts nothing', async () => {
    const noted = sortReading({ notes: ['object_not_in_sentence'], rules: [{ object: GREEN, place: YELLOW, notes: [] }, { object: { ...RED, verified: false }, place: BLUE, notes: ['object_not_in_sentence'] }] })
    const calls = server({ 'POST /v1/commands/parse': noted })
    cockpit()
    await command(SENTENCE)
    expect(await screen.findByRole('region', { name: 'Verstanden' })).toBeTruthy()
    expect(
      screen.getByText('Verstanden, ich soll sortieren: Grüne Teile → in die gelbe Kiste · rote → in die blaue. Bitte die Karte prüfen und auf Start drücken: Regel 2: Was gegriffen werden soll, steht nicht im Satz.'),
    ).toBeTruthy()
    expect(moved(calls)).toHaveLength(0)
  })

  it('opens the card where the server does not call the sort startable', async () => {
    const calls = server({ 'POST /v1/commands/parse': sortReading({ startable: false }) })
    cockpit()
    await command(SENTENCE)
    expect(await screen.findByRole('region', { name: 'Verstanden' })).toBeTruthy()
    expect(screen.getByText(/Bitte die Karte prüfen und auf Start drücken: die Lesung ist nicht eindeutig genug/)).toBeTruthy()
    expect(moved(calls)).toHaveLength(0)
  })

  it('starts a clean reading of one kind as it always did, its reply unchanged', async () => {
    const calls = server({ 'POST /v1/commands/parse': ONE_KIND })
    cockpit()
    await command('Grüne Teile in die gelbe Kiste')
    await waitFor(() => expect(sent(calls, 'POST /v1/task')).toHaveLength(1))
    expect('more_rules' in (sent(calls, 'POST /v1/task')[0].body as object)).toBe(false)
    expect(await screen.findByText('Verstanden, ich fange an: der Roboter fährt zu Blick 1.')).toBeTruthy()
  })
})

// ── the run of a sort ────────────────────────────────────────────────────────────────────────────────────────────

/** A small event of the run `run-s`, for the run model alone. */
function e(seq: number, type: string, data: Record<string, unknown> = {}): RunEvent {
  return { type, run_id: 'run-s', seq, ts: 1000 + seq, severity: 'info', human: type, step: '', step_index: null, step_total: null, data }
}

function lineOf(view: RunView, key: string): ChatLine | undefined {
  return view.chat.find((line) => line.msg.key === key)
}

function linesOf(view: RunView, key: string): ChatLine[] {
  return view.chat.filter((line) => line.msg.key === key)
}

function sayIt(line: ChatLine | undefined, lang: 'de' | 'en' = 'de'): string {
  if (!line) throw new Error('no line')
  return translate(lang, line.msg.key as never, line.msg.params, COCKPIT)
}

const YELLOW_OVERLAY = '/v1/runs/run-s/target/overlay'

const SURVEYED = [
  e(1, 'run_started', { kind: 'task', plan: PLAN }),
  e(2, 'task.survey_started', { phrases: ['yellow bin', 'blue bin'], looks: ['look_1', 'look_2', 'look_3'] }),
  e(3, 'task.target_found', { phrase: 'yellow bin', target: { label: 'yellow bin', score: 0.91, look: 'look_1', overlay: YELLOW_OVERLAY }, look: 'look_1', parts_seen: null }),
  e(4, 'task.target_found', { phrase: 'blue bin', target: { label: 'blue bin', score: 0.88, look: 'look_2', overlay: '/v1/runs/run-s/target/overlay?phrase=blue+bin' }, look: 'look_2', parts_seen: null }),
]

/** Part 1 is red: gripped, filed under the second rule, placed in the blue bin. */
const FIRST_PART = [
  e(5, 'task.part_started', { part: 1, of: null }),
  e(6, 'pick_result', { succeeded: true, part: 1, pick: 1, outcome: 'succeeded' }),
  e(7, 'task.rule', { part: 1, rule: 1, object: 'red part', place: 'target:blue bin', place_label: 'in die blaue' }),
  e(8, 'task.carry_started', { to_look: 'look_2' }),
  e(9, 'task.place_started', { place: 'target:blue bin' }),
  e(10, 'task.placed', { outcome: 'executed' }),
  e(11, 'task.part_finished', { part: 1, placed: true, duration_s: 21 }),
]

/** Part 2 is green: gripped and filed under the first rule. */
const SECOND_PART = [
  e(12, 'task.part_started', { part: 2, of: null }),
  e(13, 'pick_result', { succeeded: true, part: 2, pick: 2, outcome: 'succeeded' }),
  e(14, 'task.rule', { part: 2, rule: 0, object: 'green part', place: 'target:yellow bin', place_label: 'in die gelbe Kiste' }),
]

describe('a sort as it runs', () => {
  it('starts by naming every rule in the operator\'s words, the mode and where it goes after', () => {
    const view = replay(SURVEYED.slice(0, 1))
    const first = view.chat[0]
    expect(first.msg.key).toBe('event.run_started.sort')
    expect(sayIt(first)).toBe('Los, ich sortiere (bis leer): Grüne Teile → in die gelbe Kiste · rote → in die blaue; danach Home.')
    expect(sayIt(first, 'en')).toBe('Starting to sort (until empty): Grüne Teile → in die gelbe Kiste · rote → in die blaue; then Home.')
    expect(voiceFor(first)).toEqual({ key: 'ck.voice.sort' })
  })

  it('looks for every bin in one survey, and names each bin it found', () => {
    const view = replay(SURVEYED)
    expect(sayIt(lineOf(view, 'ck.target.lookingAll'))).toBe('Ich suche die Ziele „in die gelbe Kiste“, „in die blaue“ (3 Blicke).')
    expect(sayIt(lineOf(view, 'ck.target.lookingAll'), 'en')).toBe('Looking for the targets “in die gelbe Kiste”, “in die blaue” (3 looks).')
    expect(linesOf(view, 'ck.target.foundNamed').map((line) => sayIt(line))).toEqual([
      'Ziel „in die gelbe Kiste“ gefunden (0,91).',
      'Ziel „in die blaue“ gefunden (0,88).',
    ])
    expect(view.survey).toMatchObject({ state: 'done', phrases: ['yellow bin', 'blue bin'] })
    expect(view.overlays.map((pin) => pin.kind)).toEqual(['target', 'target'])
  })

  it('keeps a survey failed that missed one bin, and names the bin it missed', () => {
    const view = replay([
      ...SURVEYED.slice(0, 2),
      e(3, 'task.target_missing', { phrase: 'blue bin', looks_tried: ['look_1', 'look_2', 'look_3'] }),
      e(4, 'task.target_found', { phrase: 'yellow bin', target: { label: 'yellow bin', score: 0.91 }, parts_seen: null }),
    ])
    expect(sayIt(lineOf(view, 'ck.target.missing'))).toBe('Ziel nicht gefunden: „in die blaue“.')
    expect(view.survey.state).toBe('failed')
  })

  it('files each gripped part under its rule, says it, and places it where that rule says', () => {
    const view = replay([...SURVEYED, ...FIRST_PART])
    const rule = lineOf(view, 'event.task.rule')
    expect(sayIt(rule)).toBe('Teil 1: rote → in die blaue.')
    expect(sayIt(rule, 'en')).toBe('Part 1: rote → in die blaue.')
    expect(rule?.part).toBe(1)
    expect(rule?.level).toBe('demo')
    expect(view.parts[0].rule).toBe(1)
    expect(sayIt(lineOf(view, 'event.task.place_started'))).toBe('Lege ab: in die blaue.')
  })

  it('counts a part for its rule once it is placed, and lights the part in hand\'s rule until then', () => {
    const counts = (view: RunView) => sortRules(view).map((rule) => rule.placed)
    expect(counts(replay(SURVEYED))).toEqual([0, 0])
    expect(currentRule(replay([...SURVEYED, ...FIRST_PART.slice(0, 2)]))).toBeNull()
    expect(currentRule(replay([...SURVEYED, ...FIRST_PART.slice(0, 3)]))).toBe(1)
    const placed = replay([...SURVEYED, ...FIRST_PART])
    expect(counts(placed)).toEqual([0, 1])
    expect(currentRule(placed)).toBeNull()
    const second = replay([...SURVEYED, ...FIRST_PART, ...SECOND_PART])
    expect(counts(second)).toEqual([0, 1])
    expect(currentRule(second)).toBe(0)
    expect(sortRules(second).map((rule) => [rule.what, rule.where])).toEqual([
      ['Grüne Teile', 'in die gelbe Kiste'],
      ['rote', 'in die blaue'],
    ])
  })

  it('says a bin found again with how far it stood, pins its new picture, and plans the drop anew', () => {
    const view = replay([
      ...SURVEYED,
      ...FIRST_PART,
      ...SECOND_PART,
      e(15, 'task.carry_started', { to_look: 'look_1' }),
      // As the server says it: the check that lost the bin, then the search's answer; a bin found again is no loss.
      e(16, 'task.target_checked', { moved_mm: 150, followed: false, phrase: 'yellow bin', target: { label: 'yellow bin' } }),
      e(17, 'task.target_relocated', {
        phrase: 'yellow bin', found: true, moved_mm: 150.4, by: 'detector', look: 'look_3',
        target: { label: 'yellow bin', score: 0.9, centre_mm: [400, -500, 80], look: 'look_3', overlay: YELLOW_OVERLAY },
        looks_tried: ['look_1', 'look_3'], refused: [], passed_over: [],
      }),
    ])
    expect(sayIt(lineOf(view, 'ck.target.checkedNamed'))).toBe('Ziel „in die gelbe Kiste“ wieder gesehen (150 mm verschoben).')
    expect(lineOf(view, 'ck.target.lostNamed')).toBeUndefined()
    const found = lineOf(view, 'event.task.target_relocated')
    expect(sayIt(found)).toBe('Das Ziel „in die gelbe Kiste“ stand 150 mm weiter; neu gefunden, die Ablage wird neu berechnet.')
    expect(sayIt(found, 'en')).toBe('The target “in die gelbe Kiste” had moved 150 mm; found again, the drop is planned anew.')
    expect(found?.part).toBe(2)
    // Served at the address its first picture had: pinned again under an address of its own, so the new one shows.
    expect(view.overlays.at(-1)).toMatchObject({ kind: 'target', url: `${YELLOW_OVERLAY}?seen=17`, look: 'look_3' })
    expect(view.target).toMatchObject({ label: 'yellow bin', centreMm: [400, -500, 80] })
    expect(view.timeline.find((step) => step.id === 'place')?.state).toBe('active')
    expect(view.lostWhy).toBeNull()
    expect(currentRule(view)).toBe(0)
  })

  it('says a bin found nowhere as a warning, and the ask card names that bin', () => {
    const lost = [
      ...SURVEYED,
      ...FIRST_PART.slice(0, 4),
      e(9, 'task.target_relocated', { phrase: 'blue bin', found: false, moved_mm: null, by: '', look: null, target: null, looks_tried: ['look_1', 'look_2', 'look_3'], refused: [], passed_over: [] }),
      e(10, 'task.target_lost', { look: 'look_2', why: 'not_seen', phrase: 'blue bin' }),
      e(11, 'task.put_back', { outcome: 'executed' }),
      e(12, 'run_finished', { id: 'run-s', kind: 'task', state: 'finished', stop_code: 'target_lost', stop_class: 'ask', parts_placed: 0, holding: false, finished_at: 1012 }),
    ]
    const view = replay(lost)
    const nowhere = lineOf(view, 'event.task.target_relocated.nowhere')
    expect(sayIt(nowhere)).toBe('Das Ziel „in die blaue“ wurde nirgends gefunden.')
    expect(sayIt(nowhere, 'en')).toBe('The target “in die blaue” was found nowhere.')
    expect(nowhere?.tone).toBe('warn')
    expect(sayIt(lineOf(view, 'ck.target.lostNamed'))).toMatch(/^„in die blaue“: /)
    expect(view.askCard).toMatchObject({ stopCode: 'target_lost', nowhere: 'in die blaue', why: 'not_seen' })
  })

  it('names the parts no rule took at the end, as the run\'s own warning', () => {
    const view = replay([
      ...SURVEYED,
      e(5, 'task.part_started', { part: 1, of: null }),
      e(6, 'task.unsorted', { count: 3, labels: ['ambiguous', 'orange part', 'ambiguous'] }),
    ])
    const line = lineOf(view, 'event.task.unsorted')
    expect(sayIt(line)).toBe('3 Teile gehören zu keiner Regel und bleiben liegen (nicht eindeutig, orange part).')
    expect(sayIt(line, 'en')).toBe('3 parts match no rule and stay where they lie (ambiguous, orange part).')
    expect(line).toMatchObject({ part: null, tone: 'warn', level: 'demo' })
    const one = replay([...SURVEYED, e(5, 'task.unsorted', { count: 1, labels: [] })])
    expect(sayIt(lineOf(one, 'event.task.unsorted.unnamed'))).toBe('1 Teil gehört zu keiner Regel und bleibt liegen.')
  })

  it('says a task of one kind as it always did, whatever new data its events carry', () => {
    const plan = { ...PLAN, more_rules: [] }
    const view = replay([
      e(1, 'run_started', { kind: 'task', plan }),
      e(2, 'task.survey_started', { phrase: 'yellow bin', phrases: ['yellow bin'], looks: ['look_1'] }),
      e(3, 'task.target_found', { phrase: 'yellow bin', target: { label: 'yellow bin', score: 0.91 }, parts_seen: null }),
      e(4, 'task.part_started', { part: 1, of: null }),
      e(5, 'task.target_checked', { moved_mm: 12, followed: true, phrase: 'yellow bin', target: { label: 'yellow bin' } }),
      e(6, 'task.place_started', { place: 'target:yellow bin' }),
    ])
    expect(view.chat.map((line) => line.msg.key)).toEqual([
      'event.run_started.task',
      'ck.target.looking',
      'ck.target.found.nocount',
      'event.task.part_started',
      'event.task.target_checked',
      'event.task.place_started',
    ])
    expect(sayIt(view.chat[0])).toBe('Los: Grüne Teile → in die gelbe Kiste → Home, bis leer.')
    expect(sayIt(view.chat[1])).toBe('Ich suche das Ziel „in die gelbe Kiste“ (1 Blick).')
    expect(sortRules(view)).toEqual([])
    expect(currentRule(view)).toBeNull()
  })
})

// ── what the cockpit draws of a running sort, and the ask card ────────────────────────────────────────────────────

describe('the rule strip in the run header', () => {
  async function running(plan: unknown, events: (runId: string) => RunEvent[]) {
    server({ '/v1/cell': { ...CELL, active_run_id: 'run-s' }, '/v1/runs/run-s': { ...ACCEPTED, id: 'run-s' } })
    cockpit()
    const socket = await waitFor(() => {
      const found = sockets.find('run-s')
      if (!found) throw new Error('no socket for run-s')
      return found
    })
    act(() => socket.deliver([ev('run-s', 1, 'run_started', { kind: 'task', plan }), ...events('run-s')]))
    return screen.findByRole('region', { name: 'Lauf' })
  }

  it('shows each rule with the parts it placed, and lights the rule of the part in hand', async () => {
    const top = await running(PLAN, (id) => [
      ev(id, 2, 'task.part_started', { part: 1, of: null }),
      ev(id, 3, 'pick_result', { succeeded: true, part: 1, pick: 1, outcome: 'succeeded' }),
      ev(id, 4, 'task.rule', { part: 1, rule: 1, object: 'red part', place: 'target:blue bin', place_label: 'in die blaue' }),
      ev(id, 5, 'task.placed', { outcome: 'executed' }),
      ev(id, 6, 'task.part_finished', { part: 1, placed: true, duration_s: 21 }),
      ev(id, 7, 'task.part_started', { part: 2, of: null }),
      ev(id, 8, 'pick_result', { succeeded: true, part: 2, pick: 2, outcome: 'succeeded' }),
      ev(id, 9, 'task.rule', { part: 2, rule: 0, object: 'green part', place: 'target:yellow bin', place_label: 'in die gelbe Kiste' }),
    ])
    const strip = await within(top).findByRole('list', { name: 'Regeln dieses Auftrags' })
    expect(within(top).getByText('Sortieren')).toBeTruthy()
    const rules = within(strip).getAllByRole('listitem')
    expect(rules.map((rule) => rule.textContent)).toEqual(['Grüne Teile → in die gelbe Kiste0', 'rote → in die blaue1'])
    expect(rules.map((rule) => rule.getAttribute('aria-current'))).toEqual(['true', null])
    expect(within(rules[1]).getByTitle('1 abgelegt')).toBeTruthy()
  })

  it('is not there for a task of one kind: its part and its place, as always', async () => {
    const plan = { ...PLAN, more_rules: [] }
    const top = await running(plan, (id) => [ev(id, 2, 'task.part_started', { part: 1, of: null })])
    expect(await within(top).findByText('Grüne Teile → in die gelbe Kiste')).toBeTruthy()
    expect(within(top).queryByRole('list', { name: 'Regeln dieses Auftrags' })).toBeNull()
  })
})

describe('the ask card of a sort', () => {
  const ASK = { runId: 'run-s', stopCode: 'target_lost', part: 1, why: 'not_seen', nowhere: 'in die blaue', target: null, error: '' } as const

  it('names the bin found nowhere, searches again with every rule, and offers no default place', () => {
    const start = vi.fn()
    renderWith(
      <AskCard ask={ASK} plan={PLAN} poses={POSES as never} motion="der Roboter fährt zu Blick 1" countdown={false} canStart busy={false} error={null} start={start} edit={() => undefined} end={() => undefined} />,
      { providers: false },
    )
    const card = screen.getByRole('region', { name: 'Rückfrage: Ziel verloren' })
    expect(within(card).getByText('Das Ziel „in die blaue“ wurde nirgends gefunden.')).toBeTruthy()
    expect(within(card).queryByRole('button', { name: /Standard-Ablage nehmen/ })).toBeNull()
    const again = within(card).getByRole('button', { name: /Nochmal suchen – der Roboter fährt zu Blick 1/ })
    expect(again.textContent).toMatch(/Findet er das Ziel nicht, legt er das Teil zurück und fragt/)
    fireEvent.click(again)
    expect(start).toHaveBeenCalledTimes(1)
    expect(start.mock.calls[0][0]).toMatchObject({ object: 'green part', more_rules: [{ object: 'red part', place: { kind: 'camera', phrase: 'blue bin' } }] })
  })
})
