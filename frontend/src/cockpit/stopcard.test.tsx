/**
 * The stop card says where the pick left the arm, and the way out by hand where no planned way back starts from there
 * (2026-10-08): the arm never stays where a refused last try left it, and where its move back did not run either, the
 * task stops for a person (`recovery_needs_person`) and the card names the standoff it stands at. Where Home was then
 * refused because the wrist camera saw no depth from there, or because the planner will not start from there, the card
 * names freedrive at the teach pendant, or Setup's "Teach a pose", whose Cancel saves nothing. No button moves the arm
 * for it, and the card adds none.
 *
 * Against a stubbed server and fake sockets, as the cockpit's own tests run: the stopped run is followed after a
 * reload and its events replayed, so what its pick said reaches the card.
 */

import { act, cleanup, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { RunEvent } from '../api/events'
import type { Lang } from '../i18n'
import { clearConversation } from '../model/chat'
import { ConfirmProvider } from '../model/confirm'
import { fakeSockets, renderWith, stubApi, type ApiCall, type FakeSockets, type Route } from '../test/render'
import Cockpit from './Cockpit'

const HAND = { kind: 'toggle', driver: 'JawIOGripper', where: 'tool output 0', connected: true, jaws: 'open', why_unknown: '', no_sensor: true, commands_sent: 2 }

const light = (id: string, state: string, code: string) => ({ id, state, code, message: `${id} ${code}`, blocks: id !== 'commands' })
const LIGHTS = [
  light('robot', 'ok', 'connected'),
  light('cameras', 'ok', 'live'),
  light('planner', 'ok', 'ready'),
  light('gripper', 'ok', 'open_confirmed'),
  light('carried_part', 'ok', 'modelled'),
  light('commands', 'info', 'ready'),
]

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
  poses: [],
  default_place: null,
  teachable: true,
  why_not: '',
  why_not_code: '',
  target_file: 'config/robot/robot.cell.yaml',
}

const PLAN = {
  object: 'green cube',
  object_said: null,
  place: { kind: 'pose', pose: 'drop_left', pose_label: 'Ablage links' },
  return_to: 'home',
  return_label: null,
  scope: 'once',
  first_motion: 'look',
  countdown: false,
}

const now = () => Date.now() / 1000

/** A stop record of the run ``runId`` and that run as the server keeps it, ended on ``stopCode`` with ``error``. */
function stopped(runId: string, kind: 'task' | 'home', stopCode: string, error: string) {
  const at = now()
  return {
    record: { run_id: runId, kind, stop_code: stopCode, at, holding: false, cleared_at: null as number | null },
    run: {
      id: runId, kind, prompt: kind === 'task' ? 'green cube' : '', requested_picks: 0, state: 'failed', started_at: at - 30,
      finished_at: at, succeeded: 0, attempted: kind === 'task' ? 1 : 0, error, stop_requested: false, stop_code: stopCode,
      stop_class: 'problem', parts_placed: 0, holding: false, halt_requested: false, step: kind === 'task' ? 'grasp' : 'return',
      ...(kind === 'task' ? { plan: PLAN } : {}),
    },
  }
}

function server(record: unknown, run: { id: string }, over: Record<string, Route> = {}): ApiCall[] {
  return stubApi({
    '/v1/cell': { state: 'connected', arm: 'URRobotArm', gripper: 'JawIOGripper', vendor: 'ur', profile: 'ur10_cell', active_run_id: null,
      needs_person: '', halted: null, payload_model: 'none', recovery: record, jaws_confirmed_at: null, countdown_due: false,
      jaws_question: false, hand: HAND, planner: { state: 'ready' } },
    '/v1/cell/readiness': { ready: false, lights: LIGHTS, blockers: [{ code: 'cell_not_cleared', message: '' }] },
    '/v1/cell/facts': FACTS,
    '/v1/cell/status': { state: 'connected', connected: true, simulated: false, vendor: 'ur', model: 'ur10', controller_state_included: false },
    '/v1/poses': POSES,
    '/v1/camera/live': () => ({ rig_id: 'EIH_Cam', rigs: [], source: 'camera', reason: '', image_base64: 'AAAA', width: 960, height: 540, captured_at: now(), age_s: 0.2 }),
    [`/v1/runs/${run.id}`]: run,
    ...over,
  })
}

function cockpit(lang: Lang = 'de') {
  return renderWith(
    <ConfirmProvider>
      <Cockpit />
    </ConfirmProvider>,
    { lang },
  )
}

function ev(runId: string, seq: number, type: string, data: Record<string, unknown> = {}): RunEvent {
  return { type, run_id: runId, seq, ts: now() - 20 + seq, severity: 'info', human: `${type} said`, step: '', step_index: null, step_total: null, data }
}

/** The stopped task's stream, as its pick said it: one part, one pick, its attempt ended where the arm stands. */
function theTaskSaid(runId: string, run: Record<string, unknown>, standsAt: string): RunEvent[] {
  return [
    ev(runId, 1, 'run_started', { kind: 'task', plan: PLAN }),
    ev(runId, 2, 'task.part_started', { part: 1, of: 1, pick: 1 }),
    ev(runId, 3, 'pick.pick_started', { attempt_total: 1 }),
    ev(runId, 4, 'pick.attempt_started', { attempt: 0, attempt_total: 1 }),
    ev(runId, 5, 'pick.executing', { attempt: 0, position_mm: [0, -650, 20] }),
    ev(runId, 6, 'pick.attempt_finished', { attempt: 0, action: 'execution_failed', outcome: 'PickOutcome.EXECUTION_FAILED', stands_at: standsAt }),
    ev(runId, 7, 'run_error', { stop_code: run.stop_code, error: run.error }),
    ev(runId, 8, 'run_finished', run),
  ]
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

async function replayed(runId: string, events: RunEvent[]) {
  const socket = await waitFor(() => {
    const found = sockets.find(runId)
    if (!found) throw new Error(`no socket for ${runId}`)
    return found
  })
  act(() => socket.deliver(events))
}

describe('the stop card says where the pick left the arm', () => {
  const SENTENCE = "the pick's move back to its look did not run, and the arm stands where the pick left it (standoff of grasp 3)"

  for (const [lang, title, words] of [
    ['de', 'Eine Person muss entscheiden', 'Der Arm steht am Anfahrpunkt von Griff 3 über dem Teil und ist nicht zur Blickpose zurückgefahren.'],
    ['en', 'A person must decide', 'The arm stands at the standoff of grasp 3 over the part and did not go back to its look pose.'],
  ] as const) {
    it(`names the standoff of the grasp a refused move back left it at (${lang})`, async () => {
      const { record, run } = stopped('run-stood', 'task', 'recovery_needs_person', SENTENCE)
      const calls = server(record, run)
      cockpit(lang)
      await replayed('run-stood', theTaskSaid('run-stood', run, 'standoff of grasp 3'))
      const card = await screen.findByRole('region', { name: title })
      expect(await within(card).findByText(words)).toBeTruthy()
      // Home was not asked yet, so no way out by hand is named, and nothing moved.
      expect(within(card).queryByText(/Freedrive|freedrive/)).toBeNull()
      expect(calls.filter((c) => c.method === 'POST')).toHaveLength(0)
    })
  }

  it('names the way back the move broke off on, where it stopped once sent', async () => {
    const { record, run } = stopped('run-way', 'task', 'recovery_needs_person', SENTENCE)
    server(record, run)
    cockpit()
    await replayed('run-way', theTaskSaid('run-way', run, 'way back from the standoff of grasp 2'))
    const card = await screen.findByRole('region', { name: 'Eine Person muss entscheiden' })
    expect(await within(card).findByText(/auf dem Weg vom Anfahrpunkt von Griff 2 zurück zur Blickpose/)).toBeTruthy()
  })

  it('says nothing of a standoff where the pick named none', async () => {
    const { record, run } = stopped('run-plain', 'task', 'failed_in_a_row', '3 picks in a row failed, so the task stops where the arm stands')
    server(record, run)
    cockpit()
    await replayed('run-plain', theTaskSaid('run-plain', run, ''))
    const card = await screen.findByRole('region', { name: 'Zu oft in Folge fehlgeschlagen' })
    await waitFor(() => expect(within(card).getByText(/Mehrere Griffe hintereinander/)).toBeTruthy())
    expect(within(card).queryByText(/Anfahrpunkt/)).toBeNull()
  })
})

describe('the stop card names the way out by hand', () => {
  const BLIND = "the move to home was refused (unsupported: camera 'EIH_Cam' could not vouch for the cell after 3 reading(s) (blind): no depth on 100% of EIH_Cam), and the arm stays where it stands; a person decides what happens next"
  const START = 'the move to home was refused (self_collision_rejected: the arm stands where the planner will not start from, so no plan exists from here and nothing was sent; bring the arm out of this configuration before it plans again), and the arm stays where it stands; a person decides what happens next'
  const GOAL = 'the move to home was refused (self_collision_rejected: no configuration of this goal passes the endpoint gate), and the arm stays where it stands; a person decides what happens next'

  for (const [why, error] of [['a blind frame', BLIND], ['a start the planner refuses', START]] as const) {
    it(`where Home was refused for ${why}: freedrive at the pendant, or Setup's "Teach a pose", and no new button`, async () => {
      const { record, run } = stopped('run-home', 'home', 'return_failed', error)
      const calls = server(record, run)
      cockpit()
      const card = await screen.findByRole('region', { name: 'Rückfahrt abgelehnt' })
      const step = await within(card).findByText(/Freedrive am Handbediengerät/)
      expect(step.textContent).toMatch(/Einrichten → „Pose einlernen …“ → „Arm freigeben – von Hand führen“/)
      expect(step.textContent).toMatch(/Abbrechen speichert nichts/)
      expect(within(card).getAllByRole('button').map((b) => b.textContent)).toEqual(
        expect.not.arrayContaining([expect.stringMatching(/Freedrive|freigeben|Pose einlernen/)]))
      expect(calls.filter((c) => c.method === 'POST')).toHaveLength(0)
    })
  }

  it('in the reader\'s language', async () => {
    const { record, run } = stopped('run-home', 'home', 'return_failed', BLIND)
    server(record, run)
    cockpit('en')
    const card = await screen.findByRole('region', { name: 'Return refused' })
    const step = await within(card).findByText(/freedrive at the teach pendant/)
    expect(step.textContent).toMatch(/Setup → “Teach a pose…” → “Free the arm – guide it by hand” \(Cancel saves nothing\)/)
  })

  for (const [why, stopCode, kind, error, title] of [
    ['Home was refused for anything else', 'return_failed', 'home', GOAL, 'Rückfahrt abgelehnt'],
    ['another stop says the same words', 'cell_fault', 'task', BLIND, 'Fehler in der Zelle'],
  ] as const) {
    it(`not where ${why}`, async () => {
      const { record, run } = stopped(`run-${stopCode}`, kind, stopCode, error)
      server(record, run)
      cockpit()
      const card = await screen.findByRole('region', { name: title })
      await waitFor(() => expect(within(card).getByRole('button', { name: /Home – geplante Fahrt nach Home/ })).toBeTruthy())
      expect(within(card).queryByText(/Freedrive am Handbediengerät/)).toBeNull()
    })
  }
})
