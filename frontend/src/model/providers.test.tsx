/**
 * The shared models against a stubbed server: one cell poll, the cell stream, and a run rebuilt after a reload.
 *
 * The reload is the case that matters. A run outlives the tab; a person who reloads the cockpit while the arm is
 * carrying a part must see the same run, step and card as before the reload, rebuilt from the server alone: the
 * active run's id from the cell poll, its record, and its events replayed from seq 0.
 */

import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { useEffect } from 'react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '../i18n'
import { eventsOf, fakeSockets, isType, refusal, renderWith, runOf, stubApi, upTo, type Fixture } from '../test/render'
import haltedRestartJson from '../test/fixtures/task_halted_then_restart.json'
import twoPartsJson from '../test/fixtures/task_two_parts_nothing_left.json'
import jawsJson from '../test/fixtures/jaws_question_round_trip.json'
import { clearConversation, loadConversation } from './chat'
import { usePrefs } from './prefs'
import { CELL_POLL_MS, CellProvider, useCell } from './useCell'
import { RunProvider, useRun } from './useRun'

const twoParts = twoPartsJson as unknown as Fixture
const halted = haltedRestartJson as unknown as Fixture
const jaws = jawsJson as unknown as Fixture

// Runs are chosen by what they are, never by their ids: the run track regenerates these logs with random ids.
const TWO = runOf(twoParts)
const HALTED = runOf(halted, (record) => record.stop_class === 'problem')

/** Hands the test the shared model's `refresh`: "poll now", as an action that changed the cell would. */
let refresh: () => void = () => undefined
function Grab() {
  const again = useCell().refresh
  useEffect(() => {
    refresh = again
  }, [again])
  return null
}

const CELL = {
  state: 'connected',
  arm: 'DummyRobotArm',
  gripper: 'DummyGripper',
  vendor: 'dummy',
  profile: 'console_dummy',
  active_run_id: null,
  needs_person: '',
  payload_model: 'not_applicable',
  countdown_due: false,
  jaws_question: false,
}

const TELEMETRY = { state: 'connected', connected: true, simulated: true, vendor: 'dummy', model: '', controller_state_included: false }

function CellProbe() {
  const { cell, readiness, readinessError, facts, telemetry, stream } = useCell()
  return (
    <ul>
      <li>state:{cell?.state ?? '-'}</li>
      <li>ready:{readiness ? String(readiness.ready) : readinessError?.code ?? '-'}</li>
      <li>facts:{facts ? 'yes' : 'no'}</li>
      <li>telemetry:{telemetry ? 'yes' : 'no'}</li>
      <li>question:{stream.question?.stage ?? 'none'}</li>
    </ul>
  )
}

function RunProbe() {
  const { view } = useRun()
  return (
    <ul>
      <li>run:{view.runId ?? '-'}</li>
      <li>phase:{view.phase}</li>
      <li>placed:{view.stats.placed}</li>
      <li>picks:{view.stats.picks}</li>
      <li>card:{view.stopCard?.stopCode ?? 'none'}</li>
    </ul>
  )
}

beforeEach(() => {
  clearConversation()
  localStorage.clear()
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  clearConversation()
})

describe('the shared cell model', () => {
  it('reads the cell, the ready bar, and, once connected, the facts and the provenance', async () => {
    fakeSockets()
    const calls = stubApi({
      '/v1/cell': CELL,
      '/v1/cell/readiness': { ready: true, lights: [], blockers: [] },
      '/v1/cell/facts': { wrist_camera: true, hand_eye_warn_mm: 6, route: { route: 'unplanned', sentence: '' }, rehearsal: true },
      '/v1/cell/status': TELEMETRY,
    })
    renderWith(<CellProbe />)
    await waitFor(() => expect(screen.getByText('state:connected')).toBeTruthy())
    await waitFor(() => expect(screen.getByText('facts:yes')).toBeTruthy())
    expect(screen.getByText('ready:true')).toBeTruthy()
    await waitFor(() => expect(screen.getByText('telemetry:yes')).toBeTruthy())
    // Reads only: nothing that moves, nothing that writes.
    expect(calls.every((c) => c.method === 'GET')).toBe(true)
    expect(calls.filter((c) => c.path === '/v1/cell/facts')).toHaveLength(1)
  })

  it('asks for the facts again while a read of them failed: the halt\'s words and alarm depend on them', async () => {
    fakeSockets()
    let fail = true
    const calls = stubApi({
      '/v1/cell': CELL,
      '/v1/cell/readiness': { ready: true, lights: [], blockers: [] },
      '/v1/cell/facts': () => (fail ? refusal(500, 'software_error', 'facts failed') : { wrist_camera: true, hand_eye_warn_mm: 6, rehearsal: true }),
      '/v1/cell/status': TELEMETRY,
    })
    render(
      <CellProvider pollMs={20}>
        <CellProbe />
      </CellProvider>,
    )
    await waitFor(() => expect(calls.filter((c) => c.path === '/v1/cell/facts').length).toBeGreaterThanOrEqual(1))
    expect(screen.getByText('facts:no')).toBeTruthy()
    fail = false
    await waitFor(() => expect(screen.getByText('facts:yes')).toBeTruthy(), { timeout: 2000 })
    const asked = calls.filter((c) => c.path === '/v1/cell/facts').length
    expect(asked).toBeGreaterThanOrEqual(2)
    // Read once it answered: no more asking.
    await new Promise((resolve) => setTimeout(resolve, 150))
    expect(calls.filter((c) => c.path === '/v1/cell/facts').length).toBe(asked)
  })

  it('polls the cell every 2 s, once for the whole console', () => {
    // The top bar's chips, the ready bar and the stop card read this one poll (build plan 4.1): every 2 s.
    fakeSockets()
    stubApi({ '/v1/cell': CELL, '/v1/cell/readiness': { ready: false, lights: [], blockers: [] } })
    const timers = vi.spyOn(globalThis, 'setInterval')
    try {
      renderWith(<CellProbe />)
      expect(CELL_POLL_MS).toBe(2000)
      expect(timers.mock.calls.filter(([, ms]) => ms === CELL_POLL_MS)).toHaveLength(1)
    } finally {
      timers.mockRestore()
    }
  })

  it('stops asking a server that has not built the ready bar yet', async () => {
    fakeSockets()
    const calls = stubApi({ '/v1/cell': CELL, '/v1/cell/readiness': refusal(501, 'not_built_yet', 'not built yet') })
    renderWith(
      <>
        <CellProbe />
        <Grab />
      </>,
    )
    await waitFor(() => expect(screen.getByText('ready:not_built_yet')).toBeTruthy())
    const asked = calls.filter((c) => c.path === '/v1/cell/readiness').length
    const polled = calls.filter((c) => c.path === '/v1/cell').length
    // Two more polls, one after the other (two refreshes in one moment are one poll).
    for (const more of [1, 2]) {
      await act(async () => refresh())
      await waitFor(() => expect(calls.filter((c) => c.path === '/v1/cell').length).toBeGreaterThanOrEqual(polled + more))
    }
    expect(calls.filter((c) => c.path === '/v1/cell/readiness').length).toBe(asked)
  })

  it('keeps a slow answer: a poll still out is not thrown away by the next tick', async () => {
    // A cell PC whose answers take longer than the poll interval must still see its cell, not an endless "loading".
    fakeSockets()
    stubApi({
      '/v1/cell': () => new Promise((resolve) => setTimeout(() => resolve(CELL), 60)),
      '/v1/cell/readiness': { ready: false, lights: [], blockers: [] },
    })
    render(
      <CellProvider pollMs={15}>
        <CellProbe />
      </CellProvider>,
    )
    await waitFor(() => expect(screen.getByText('state:connected')).toBeTruthy(), { timeout: 2000 })
  })

  it('opens the jaws question the moment the cell stream says it, from the stream replayed at seq 0', async () => {
    const sockets = fakeSockets()
    stubApi({ '/v1/cell': CELL, '/v1/cell/readiness': { ready: false, lights: [], blockers: [] } })
    renderWith(<CellProbe />)
    await waitFor(() => expect(sockets.find('cell')).toBeTruthy())
    expect(sockets.find('cell')?.sinceSeq).toBe(0)
    act(() => sockets.find('cell')?.deliver(jaws.events.slice(0, 1)))
    expect(screen.getByText('question:where')).toBeTruthy()
    act(() => sockets.find('cell')?.deliver(jaws.events.slice(1)))
    expect(screen.getByText('question:none')).toBeTruthy()
  })
})

describe('the run on screen', () => {
  it('is rebuilt after a reload from the active run id, its record, and its events from seq 0', async () => {
    const sockets = fakeSockets()
    const record = twoParts.runs[TWO]
    stubApi({
      '/v1/cell': { ...CELL, active_run_id: TWO },
      '/v1/cell/readiness': { ready: false, lights: [], blockers: [] },
      [`/v1/runs/${TWO}`]: { ...record, state: 'running', stop_code: '', stop_class: '', finished_at: null },
    })
    renderWith(<RunProbe />)
    await waitFor(() => expect(screen.getByText(`run:${TWO}`)).toBeTruthy())
    await waitFor(() => expect(sockets.find(TWO)).toBeTruthy())
    const socket = sockets.find(TWO)!
    expect(socket.sinceSeq).toBe(0)
    act(() => socket.deliver(eventsOf(twoParts, TWO)))
    expect(screen.getByText('phase:finished')).toBeTruthy()
    expect(screen.getByText('placed:2')).toBeTruthy()
    // The run ended: its stream is closed and its summary is in the conversation.
    expect(socket.closed).toBe(true)
    expect(loadConversation().map((e) => e.id)).toContain(`summary:${TWO}`)
  })

  it('draws the stop card of a run halted while the page was closed', async () => {
    const sockets = fakeSockets()
    const record = halted.runs[HALTED]
    stubApi({
      '/v1/cell': { ...CELL, active_run_id: HALTED },
      '/v1/cell/readiness': { ready: false, lights: [], blockers: [] },
      [`/v1/runs/${HALTED}`]: { ...record, state: 'running', stop_code: '', stop_class: '' },
    })
    renderWith(<RunProbe />)
    await waitFor(() => expect(sockets.find(HALTED)).toBeTruthy())
    act(() => sockets.find(HALTED)!.deliver(eventsOf(halted, HALTED)))
    expect(screen.getByText('card:halted')).toBeTruthy()
    expect(screen.getByText('phase:failed')).toBeTruthy()
  })

  it('takes the counters from the run record after a gap, never from a guess', async () => {
    const sockets = fakeSockets()
    const record = twoParts.runs[TWO]
    const calls = stubApi({
      '/v1/cell': { ...CELL, active_run_id: TWO },
      '/v1/cell/readiness': { ready: false, lights: [], blockers: [] },
      [`/v1/runs/${TWO}`]: record,
    })
    renderWith(<RunProbe />)
    await waitFor(() => expect(sockets.find(TWO)).toBeTruthy())
    const events = eventsOf(twoParts, TWO)
    act(() => sockets.find(TWO)!.deliver([{ type: 'gap', run_id: TWO, dropped: 45, human: 'lost' }, ...events.slice(45, 50)]))
    await waitFor(() => expect(calls.filter((c) => c.path === `/v1/runs/${TWO}`).length).toBe(2))
    await waitFor(() => expect(screen.getByText('picks:4')).toBeTruthy())
    expect(screen.getByText('placed:2')).toBeTruthy()
  })

  /** A followed run, half-way through its first grasp: the cell names it active, and its record says running. */
  async function followHalfway(server: { active: string | null; record: () => unknown }) {
    const sockets = fakeSockets()
    const record = twoParts.runs[TWO]
    const calls = stubApi({
      '/v1/cell': () => ({ ...CELL, active_run_id: server.active }),
      '/v1/cell/readiness': { ready: false, lights: [], blockers: [] },
      [`/v1/runs/${TWO}`]: () => server.record() ?? { ...record, state: 'running', stop_code: '', stop_class: '', finished_at: null },
    })
    renderWith(
      <>
        <RunProbe />
        <Grab />
      </>,
    )
    await waitFor(() => expect(sockets.find(TWO)).toBeTruthy())
    act(() => sockets.find(TWO)!.deliver(upTo(eventsOf(twoParts, TWO), isType('pick.executing'))))
    expect(screen.getByText('phase:running')).toBeTruthy()
    return { socket: sockets.find(TWO)!, calls }
  }

  it('asks the cell again the moment a run says the hand changed: a grasp, a place', async () => {
    // The top bar's hand chip reads the one cell poll, every 2 s; a toggle's count changes at the grasp and at the place,
    // in between two polls. The run's own events say when, so the cell is read again at once (a poll of a minute here,
    // so only those events can ask).
    const sockets = fakeSockets()
    const record = twoParts.runs[TWO]
    const calls = stubApi({
      '/v1/cell': { ...CELL, active_run_id: TWO },
      '/v1/cell/readiness': { ready: false, lights: [], blockers: [] },
      [`/v1/runs/${TWO}`]: { ...record, state: 'running', stop_code: '', stop_class: '', finished_at: null },
    })
    render(
      <MemoryRouter>
        <I18nProvider lang="de">
          <CellProvider pollMs={60_000}>
            <RunProvider>
              <RunProbe />
            </RunProvider>
          </CellProvider>
        </I18nProvider>
      </MemoryRouter>,
    )
    await waitFor(() => expect(sockets.find(TWO)).toBeTruthy())
    const polls = () => calls.filter((c) => c.path === '/v1/cell').length
    const events = eventsOf(twoParts, TWO)
    const grasp = upTo(events, isType('pick_result'))
    act(() => sockets.find(TWO)!.deliver(grasp.slice(0, -1)))
    await waitFor(() => expect(screen.getByText('phase:running')).toBeTruthy())
    await new Promise((resolve) => setTimeout(resolve, 50))
    const before = polls()
    act(() => sockets.find(TWO)!.deliver(grasp.slice(-1)))
    await waitFor(() => expect(polls()).toBeGreaterThan(before))
    const placed = upTo(events, isType('task.placed'))
    const afterGrasp = polls()
    act(() => sockets.find(TWO)!.deliver(placed.slice(grasp.length)))
    await waitFor(() => expect(polls()).toBeGreaterThan(afterGrasp))
  })

  it('stops drawing a run that a restarted server no longer knows, says so, and closes its stream', async () => {
    // The server restarted mid-run: its cell names no run, and the run's record is gone with it (404). The view
    // must not stay "running" with a step in progress forever over an arm nothing reports on.
    const server: { active: string | null; record: () => unknown } = { active: TWO, record: () => undefined }
    const { socket } = await followHalfway(server)
    server.active = null
    server.record = () => refusal(404, 'no_such_run', 'There is no run with that id.')
    await act(async () => refresh())
    await waitFor(() => expect(screen.getByText('phase:lost')).toBeTruthy())
    expect(socket.closed).toBe(true)
  })

  it('reads the record of a run the cell stopped naming while its stream was silent, and keeps its stream', async () => {
    // The stream may simply be late: the cell's poll can see the run end before the run's own last events arrive.
    const server: { active: string | null; record: () => unknown } = { active: TWO, record: () => undefined }
    const { socket, calls } = await followHalfway(server)
    const asked = calls.filter((c) => c.path === `/v1/runs/${TWO}`).length
    server.active = null
    server.record = () => ({ ...halted.runs[HALTED], id: TWO, kind: 'task' })
    await act(async () => refresh())
    await waitFor(() => expect(calls.filter((c) => c.path === `/v1/runs/${TWO}`).length).toBe(asked + 1))
    await waitFor(() => expect(screen.getByText('phase:failed')).toBeTruthy())
    expect(socket.closed).toBe(false)
  })
})

describe('the preferences', () => {
  it('start in the demo view with voice output off, and remember a switch', () => {
    let prefs: ReturnType<typeof usePrefs> | null = null
    function Probe() {
      const now = usePrefs()
      useEffect(() => {
        prefs = now
      }, [now])
      return <p>{`${now.view}/${now.voiceOut ? 'on' : 'off'}/${now.theme}`}</p>
    }
    fakeSockets()
    stubApi({ '/v1/cell': CELL })
    renderWith(<Probe />)
    expect(screen.getByText('demo/off/dark')).toBeTruthy()
    expect(document.documentElement.dataset.view).toBe('demo')
    act(() => {
      prefs!.setView('tech')
      prefs!.setVoiceOut(true)
    })
    expect(screen.getByText('tech/on/dark')).toBeTruthy()
    expect(localStorage.getItem('willy.view')).toBe('tech')
    expect(localStorage.getItem('willy.voiceOut')).toBe('on')
    expect(document.documentElement.dataset.view).toBe('tech')
  })
})
