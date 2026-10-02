/**
 * The audience window (`demo.html`, build plan 4.3, OD 4): a read-only mirror for the large screen.
 *
 * What is pinned: it has no button and no link at all (a viewer is never one click from a motion); it follows the run
 * the cell names, and after a stop it follows the recovery record, so a stopped run's card is on the projector even
 * when the window opened after the stop; a grasp overlay is pinned over the stage only while it is fresh (5 s), with
 * the band that says it shows the grasp as decided and that the arm has moved since; the caption says the step in big
 * words with the target; and the STILL GRINDING card counts today's parts (the one place with a joke in it).
 */

import { act, cleanup, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { RunOut } from '../api/client'
import type { RunEvent } from '../api/events'
import { EMPTY_RUN, reduce, type RunView } from '../model/runModel'
import homeJson from '../test/fixtures/console_dummy_home_counts_down.json'
import onceJson from '../test/fixtures/console_dummy_task_once.json'
import haltedJson from '../test/fixtures/task_halted_then_restart.json'
import { eventsOf, fakeSockets, isType, renderWith, runOf, stubApi, upTo, type Fixture } from '../test/render'
import Audience, { RUNS_MS } from './Audience'
import { captionOf } from './caption'

const HALTED = haltedJson as unknown as Fixture
const STOPPED = runOf(HALTED, (run) => run.stop_code === 'halted')
const RESTART = runOf(HALTED, (run) => run.restart_of !== null && run.restart_of !== undefined)
const ONCE = onceJson as unknown as Fixture
const QUICK = runOf(ONCE)
const HOME = homeJson as unknown as Fixture

const FACTS = {
  wrist_camera: true,
  cameras: [{ rig_id: 'EIH_Cam', mounting: 'wrist', primary: true }],
  looks: ['look_1', 'look_2'],
  hand_eye_warn_mm: 6,
  route: { route: 'planned', sentence: '' },
  rehearsal: false,
}

function cell(over: Record<string, unknown> = {}) {
  return {
    state: 'connected',
    vendor: 'ur',
    arm: 'URRobotArm',
    active_run_id: null,
    needs_person: '',
    halted: null,
    payload_model: 'none',
    recovery: null,
    countdown_due: false,
    jaws_question: false,
    ...over,
  }
}

function serve(routes: Record<string, unknown>) {
  return stubApi({
    '/v1/cell/readiness': { ready: true, lights: [], blockers: [] },
    '/v1/cell/facts': FACTS,
    '/v1/cell/status': { state: 'connected', connected: true, simulated: false, vendor: 'ur', model: 'ur10', controller_state_included: false },
    '/v1/camera/live': { rig_id: 'EIH_Cam', rigs: [], source: 'none', reason: 'no_camera', width: 0, height: 0 },
    '/v1/runs': [],
    ...routes,
  })
}

/** A run's events folded into the view, as the window draws them. */
function viewOf(events: readonly RunEvent[]): RunView {
  return events.reduce((view, event) => reduce(view, event), EMPTY_RUN)
}

/** A running copy of a run's record: the run the cell names while it runs. */
function running(id: string, fixture: Fixture = HALTED): RunOut {
  return { ...fixture.runs[id], state: 'running', finished_at: null, stop_code: '', stop_class: '' }
}

/** The events as if they happened just now: the last one at `now`, the gaps between them kept. */
function now(events: readonly RunEvent[]): RunEvent[] {
  const last = events[events.length - 1]?.ts ?? 0
  const shift = Date.now() / 1000 - last
  return events.map((event) => ({ ...event, ts: event.ts + shift }))
}

beforeEach(() => {
  localStorage.clear()
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.useRealTimers()
  localStorage.clear()
})

describe('the audience window', () => {
  it('shows a stopped run\'s card from the recovery record, read-only, without one button or link', async () => {
    const record = HALTED.runs[STOPPED]
    serve({
      '/v1/cell': cell({ recovery: { run_id: STOPPED, kind: 'task', stop_code: 'halted', at: record.finished_at, holding: false, cleared_at: null } }),
      [`/v1/runs/${STOPPED}`]: record,
    })
    const sockets = fakeSockets()
    renderWith(<Audience />)
    await waitFor(() => expect(sockets.find(STOPPED)).toBeTruthy())
    act(() => sockets.find(STOPPED)!.deliver(eventsOf(HALTED, STOPPED)))
    const card = await screen.findByRole('alert')
    expect(within(card).getByText('Angehalten')).toBeTruthy()
    expect(card.textContent).toMatch(/keine Bewegung mehr befohlen/)
    expect(card.textContent).toMatch(/Bedienplatz/)
    expect(screen.queryAllByRole('button')).toHaveLength(0)
    expect(screen.queryAllByRole('link')).toHaveLength(0)
    // Under a card the live picture dims, and only the picture: `dim` is the console's muted-text utility, which would
    // grey every word on the stage, the card's included.
    const stage = document.querySelector('.aud-stage') as HTMLElement
    expect(stage.classList.contains('dim')).toBe(false)
    expect(stage.classList.contains('dimmed')).toBe(true)
    // The card is the one thing to read: its title once (no big caption repeating it under the card), and no joke
    // beside a problem stop.
    expect(screen.getAllByText('Angehalten')).toHaveLength(1)
    expect(screen.queryByRole('heading', { level: 1 })).toBeNull()
    expect(screen.queryByRole('region', { name: 'STILL GRINDING' })).toBeNull()
  })

  it('says the cell was confirmed clear once a person said so, and that it now waits for Restart or Home', async () => {
    const record = HALTED.runs[STOPPED]
    const at = record.finished_at ?? 0
    serve({
      '/v1/cell': cell({ recovery: { run_id: STOPPED, kind: 'task', stop_code: 'halted', at, holding: false, cleared_at: at + 5 } }),
      [`/v1/runs/${STOPPED}`]: record,
    })
    const sockets = fakeSockets()
    renderWith(<Audience />)
    await waitFor(() => expect(sockets.find(STOPPED)).toBeTruthy())
    act(() => sockets.find(STOPPED)!.deliver(eventsOf(HALTED, STOPPED)))
    const card = await screen.findByRole('alert')
    await waitFor(() => expect(card.textContent).toMatch(/Zelle freigegeben: wartet auf Neustart oder Home/))
    expect(card.textContent).not.toMatch(/die Zelle freigeben/)
  })

  it('pins a fresh grasp overlay over the stage, as decided, and says the arm has moved since', async () => {
    const record: RunOut = { ...HALTED.runs[RESTART], state: 'running', finished_at: null, stop_code: '', stop_class: '' }
    serve({ '/v1/cell': cell({ active_run_id: RESTART }), [`/v1/runs/${RESTART}`]: record })
    const sockets = fakeSockets()
    renderWith(<Audience />)
    await waitFor(() => expect(sockets.find(RESTART)).toBeTruthy())
    act(() => sockets.find(RESTART)!.deliver(now(upTo(eventsOf(HALTED, RESTART), isType('pick.executing')))))
    const pin = await screen.findByRole('img', { name: /wie entschieden/ })
    expect(pin.getAttribute('src')).toBe(`/v1/runs/${RESTART}/overlays/1`)
    expect(screen.getByText(/der Arm hat sich seither bewegt/)).toBeTruthy()
  })

  it("pins a new overlay the moment it arrives, also where the server's clock runs a little ahead of this screen's", async () => {
    serve({ '/v1/cell': cell({ active_run_id: RESTART }), [`/v1/runs/${RESTART}`]: running(RESTART) })
    const sockets = fakeSockets()
    renderWith(<Audience />)
    await waitFor(() => expect(sockets.find(RESTART)).toBeTruthy())
    // The grasp was decided a second "from now" by this screen's clock: the server's clock runs ahead (another PC), or
    // this window's clock has not ticked since (it ticks twice a second). Either way the overlay is new.
    const ahead = now(upTo(eventsOf(HALTED, RESTART), isType('pick.executing'))).map((event) => ({ ...event, ts: event.ts + 1 }))
    act(() => sockets.find(RESTART)!.deliver(ahead))
    // At once, not half a second later, and not only once this screen's clock has caught up with the server's.
    expect(screen.getByRole('img', { name: /wie entschieden/ }).getAttribute('src')).toBe(`/v1/runs/${RESTART}/overlays/1`)
  })

  it('does not pin an overlay that is no longer fresh', async () => {
    const record: RunOut = { ...HALTED.runs[RESTART], state: 'running', finished_at: null, stop_code: '', stop_class: '' }
    serve({ '/v1/cell': cell({ active_run_id: RESTART }), [`/v1/runs/${RESTART}`]: record })
    const sockets = fakeSockets()
    renderWith(<Audience />)
    await waitFor(() => expect(sockets.find(RESTART)).toBeTruthy())
    act(() => sockets.find(RESTART)!.deliver(upTo(eventsOf(HALTED, RESTART), isType('pick.executing'))))
    await screen.findByText(/den grünen Würfel/)
    expect(screen.queryByRole('img', { name: /wie entschieden/ })).toBeNull()
  })

  it('says the step in big words, with the part and where it goes', async () => {
    const record: RunOut = { ...HALTED.runs[RESTART], state: 'running', finished_at: null, stop_code: '', stop_class: '' }
    serve({ '/v1/cell': cell({ active_run_id: RESTART }), [`/v1/runs/${RESTART}`]: record })
    const sockets = fakeSockets()
    renderWith(<Audience />)
    await waitFor(() => expect(sockets.find(RESTART)).toBeTruthy())
    act(() => sockets.find(RESTART)!.deliver(now(upTo(eventsOf(HALTED, RESTART), isType('pick.executing')))))
    const caption = await screen.findByRole('heading', { level: 1 })
    await waitFor(() => expect(caption.textContent).toMatch(/Greift/))
    expect(caption.textContent).toMatch(/den grünen Würfel/)
    expect(screen.getByText(/Ablage links/)).toBeTruthy()
    expect(screen.getByText(/Teil 1/)).toBeTruthy()
    // The pick's first attempt is attempt 1 of its 5 (the server counts from 0, the model from 1).
    expect(screen.getByText('Versuch 1/5')).toBeTruthy()
  })

  it('says in plain German what the robot looks for, and where the part goes without saying "Ablage" twice', async () => {
    serve({ '/v1/cell': cell({ active_run_id: RESTART }), [`/v1/runs/${RESTART}`]: running(RESTART) })
    const sockets = fakeSockets()
    renderWith(<Audience />)
    await waitFor(() => expect(sockets.find(RESTART)).toBeTruthy())
    act(() => sockets.find(RESTART)!.deliver(now(upTo(eventsOf(HALTED, RESTART), isType('pick.perceived')))))
    const caption = await screen.findByRole('heading', { level: 1 })
    // "Sucht" takes the operator's own words ("den grünen Würfel") as they were said; "Schaut: den …" does not.
    await waitFor(() => expect(caption.textContent).toBe('Sucht: den grünen Würfel'))
    expect(screen.getByText('Ziel: Ablage links')).toBeTruthy()
    expect(document.body.textContent).not.toMatch(/Ablage: Ablage/)
    // While it looks, the strip lights the one step the caption names, never two at once.
    const lit = Array.from(document.querySelectorAll('.aud-step.active')).map((step) => step.textContent)
    expect(lit).toEqual(['Schauen'])
  })

  it('looks for "everything the camera sees" in a sentence that reads, before the first step', async () => {
    serve({ '/v1/cell': cell({ active_run_id: QUICK }), [`/v1/runs/${QUICK}`]: running(QUICK, ONCE) })
    const sockets = fakeSockets()
    renderWith(<Audience />)
    await waitFor(() => expect(sockets.find(QUICK)).toBeTruthy())
    act(() => sockets.find(QUICK)!.deliver(now(upTo(eventsOf(ONCE, QUICK), isType('task.pose_screened')))))
    const caption = await screen.findByRole('heading', { level: 1 })
    await waitFor(() => expect(caption.textContent).toBe('Sucht: alles, was die Kamera sieht'))
  })

  it("names a Home run's pose by its label, never by its YAML name, and says it arrived", async () => {
    const id = runOf(HOME)
    serve({
      '/v1/cell': cell({ active_run_id: id }),
      [`/v1/runs/${id}`]: running(id, HOME),
      '/v1/poses': {
        home: { name: 'home', label: 'Home', joints_deg: [0, -90, 0, -90, 0, 0], source: 'config', screen: null, note: '', taught_at: null },
        poses: [{ name: 'park', label: 'Parkposition', joints_deg: [30, -80, -100, -90, 90, 0], source: 'config', screen: 'band', note: '', taught_at: null }],
        default_place: null,
        teachable: false,
        why_not: '',
        why_not_code: '',
        target_file: null,
      },
    })
    const sockets = fakeSockets()
    renderWith(<Audience />)
    await waitFor(() => expect(sockets.find(id)).toBeTruthy())
    act(() => sockets.find(id)!.deliver(now(upTo(eventsOf(HOME, id), isType('home.started')))))
    const caption = await screen.findByRole('heading', { level: 1 })
    await waitFor(() => expect(caption.textContent).toBe('Fährt nach Parkposition'))
    act(() => sockets.find(id)!.deliver(eventsOf(HOME, id)))
    await waitFor(() => expect(caption.textContent).toBe('Angekommen: Parkposition'))
    expect(document.body.textContent).not.toMatch(/\bpark\b/)
  })

  it('draws the hands-off countdown in the warning hue, not in the lime of a running step', () => {
    const counting = viewOf(upTo(eventsOf(HOME, runOf(HOME)), isType('run_countdown')))
    expect(captionOf(counting, true).tone).toBe('warn')
  })

  it('says the count once when a run has ended: in the caption, not again in the counters', async () => {
    serve({ '/v1/cell': cell({ active_run_id: RESTART }), [`/v1/runs/${RESTART}`]: running(RESTART) })
    const sockets = fakeSockets()
    renderWith(<Audience />)
    await waitFor(() => expect(sockets.find(RESTART)).toBeTruthy())
    act(() => sockets.find(RESTART)!.deliver(eventsOf(HALTED, RESTART)))
    const caption = await screen.findByRole('heading', { level: 1 })
    await waitFor(() => expect(caption.textContent).toMatch(/Fertig: 1 Teil abgelegt/))
    expect(document.querySelector('.aud-counters')?.textContent ?? '').not.toMatch(/abgelegt/)
  })

  it('follows a run that began and ended between two reads of the cell, but none from before the window opened', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    const quick = ONCE.runs[QUICK]
    const old: RunOut = { ...HALTED.runs[RESTART], id: 'run-old', started_at: quick.started_at - 100, finished_at: quick.started_at - 90 }
    let reads = 0
    serve({ '/v1/cell': cell(), '/v1/runs': () => (++reads === 1 ? [old] : [quick, old]) })
    const sockets = fakeSockets()
    renderWith(<Audience />)
    const caption = await screen.findByRole('heading', { level: 1 })
    await waitFor(() => expect(reads).toBe(1))
    expect(caption.textContent).toMatch(/Bereit/)
    expect(sockets.find('run-old')).toBeUndefined()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(RUNS_MS)
    })
    await waitFor(() => expect(sockets.find(QUICK)).toBeTruthy())
    act(() => sockets.find(QUICK)!.deliver(eventsOf(ONCE, QUICK)))
    await waitFor(() => expect(caption.textContent).toMatch(/Fertig: 1 Teil abgelegt/))
  })

  it('never counts down on the STILL GRINDING card when a run falls out of the server\'s list, and asks for all it keeps', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    const t = Date.now() / 1000
    const base = HALTED.runs[RESTART]
    const early: RunOut = { ...base, id: 'run-a', started_at: t - 3600, finished_at: t - 3000, parts_placed: 7, state: 'finished' }
    const late: RunOut = { ...base, id: 'run-b', started_at: t - 600, finished_at: t - 300, parts_placed: 3, state: 'finished' }
    let reads = 0
    const calls = serve({ '/v1/cell': cell(), '/v1/runs': () => (++reads === 1 ? [early] : [late]) })
    fakeSockets()
    renderWith(<Audience />)
    const card = await screen.findByRole('region', { name: 'STILL GRINDING' })
    const parts = () => within(card).getByText('Teile heute').closest('.readout-row')?.textContent ?? ''
    await waitFor(() => expect(parts()).toMatch(/7$/))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(RUNS_MS)
    })
    await waitFor(() => expect(reads).toBeGreaterThanOrEqual(2))
    await waitFor(() => expect(parts()).toMatch(/10$/))
    expect(calls.find((call) => call.path === '/v1/runs')?.query).toBe('limit=200')
  })

  it('counts today\'s parts on the STILL GRINDING card, with no coffee break and no raise asked', async () => {
    const t = Date.now() / 1000
    const base = HALTED.runs[RESTART]
    serve({
      '/v1/cell': cell(),
      '/v1/runs': [
        { ...base, id: 'run-today', started_at: t - 3600, finished_at: t - 600, parts_placed: 7, state: 'finished' },
        { ...base, id: 'run-yesterday', started_at: t - 3 * 86400, finished_at: t - 3 * 86400 + 600, parts_placed: 5, state: 'finished' },
      ],
    })
    fakeSockets()
    renderWith(<Audience />)
    const card = await screen.findByRole('region', { name: 'STILL GRINDING' })
    await waitFor(() => expect(within(card).getByText('Teile heute').closest('.readout-row')?.textContent).toMatch(/7$/))
    expect(within(card).getByText('Kaffeepausen').closest('.readout-row')?.textContent).toMatch(/0$/)
    expect(within(card).getByText('Gehaltserhöhungen verlangt').closest('.readout-row')?.textContent).toMatch(/0$/)
  })
})
