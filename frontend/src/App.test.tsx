/**
 * The shell: four screens and their old addresses, a top bar that says what the cell is on every screen, and the
 * switches a person needs at the cell (view, language, theme, voice, the audience window).
 *
 * The chips are the honesty rule of the old sidebar carried into the new bar: what is connected, what the hand is
 * believed to hold (and that a toggle's count is a count, not a sensor), whether the arm is halted, what runs.
 */

import { act, cleanup, fireEvent, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from './App'
import appSource from './App.tsx?raw'
import { clearConversation } from './model/chat'
import { fakeSockets, refusal, renderWith, stubApi } from './test/render'

const CELL = {
  state: 'connected',
  arm: 'URRobotArm',
  gripper: 'JawIOGripper',
  vendor: 'ur',
  profile: 'ur10_cell',
  active_run_id: null,
  needs_person: '',
  halted: null,
  payload_model: 'planner_and_filter',
  recovery: null,
  countdown_due: false,
  jaws_question: false,
  hand: {
    kind: 'toggle',
    driver: 'JawIOGripper',
    where: 'tool output 0',
    connected: true,
    jaws: 'open',
    why_unknown: '',
    no_sensor: true,
    commands_sent: 2,
  },
}

const TELEMETRY = { state: 'connected', connected: true, simulated: false, vendor: 'ur', model: 'ur10', controller_is_simulator: true, controller_state_included: false }

function serve(cell: Record<string, unknown> = CELL, extra: Record<string, unknown> = {}) {
  return stubApi({
    '/v1/cell': cell,
    '/v1/cell/readiness': refusal(501, 'not_built_yet', 'not built yet'),
    '/v1/cell/facts': refusal(501, 'not_built_yet', 'not built yet'),
    '/v1/cell/status': TELEMETRY,
    '/v1/preflight': { ok: true, n_blocking: 0, n_warn: 0, n_bench: 0, vendor: 'ur', profile: 'ur10_cell', checks: [] },
    '/v1/config/writable': [],
    '/v1/history/kpis': { total_attempts: 0, kpis: {}, unmeasurable: {}, outcomes: {}, source: 'logs', record_log_path: 'x', record_log_exists: false },
    '/v1/history/records': [],
    '/v1/history/runs': [],
    ...extra,
  })
}

beforeEach(() => {
  localStorage.clear()
  clearConversation()
  fakeSockets()
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  localStorage.clear()
})

describe('the routes', () => {
  it('open the Cockpit at /, and every screen under its own address', async () => {
    serve()
    renderWith(<App />, { route: '/' })
    expect(screen.getByRole('heading', { level: 2, name: 'Cockpit' })).toBeTruthy()
    cleanup()
    renderWith(<App />, { route: '/setup' })
    expect(screen.getByRole('heading', { level: 2, name: 'Einrichten' })).toBeTruthy()
    cleanup()
    renderWith(<App />, { route: '/settings' })
    expect(screen.getByRole('heading', { level: 2, name: 'Einstellungen' })).toBeTruthy()
  })

  it('send the old addresses to their new screens', () => {
    serve()
    for (const [old, heading] of [['/pick', 'Cockpit'], ['/cell', 'Einrichten'], ['/config', 'Einstellungen']]) {
      renderWith(<App />, { route: old })
      expect(screen.getByRole('heading', { level: 2, name: heading }), old).toBeTruthy()
      cleanup()
    }
  })

  it('say so for an address that is nothing', () => {
    serve()
    renderWith(<App />, { route: '/nowhere' })
    expect(screen.getByText('Diese Seite gibt es nicht')).toBeTruthy()
  })

  it('no longer mount the Pick screen', () => {
    expect(appSource).not.toMatch(/screens\/Pick/)
  })
})

describe('the top bar', () => {
  it('names the four screens in German, with the cockpit first', () => {
    serve()
    renderWith(<App />)
    const nav = screen.getByRole('navigation', { name: 'Hauptnavigation' })
    expect([...nav.querySelectorAll('a')].map((a) => a.textContent)).toEqual(['Cockpit', 'Einrichten', 'Verlauf', 'Einstellungen'])
  })

  it('says what the cell is, and that a toggle\'s jaws are counted, not sensed', async () => {
    serve()
    renderWith(<App />)
    await waitFor(() => expect(screen.getByText(/ur · verbunden/)).toBeTruthy())
    await waitFor(() => expect(screen.getByText(/Simulator \(URSim\)/)).toBeTruthy())
    // The demo view names the output and the count, not the driver's Python class.
    expect(screen.getByText('Tool-DO0 · Backen OFFEN (gezählt, kein Sensor)')).toBeTruthy()
    expect(screen.queryByText(/JawIOGripper/)).toBeNull()
    expect(screen.getByText('kein Lauf')).toBeTruthy()
  })

  it('names the hand\'s driver in the tech view, and a hand that is no toggle by its kind in the demo view', async () => {
    localStorage.setItem('willy.view', 'tech')
    serve()
    renderWith(<App />)
    await waitFor(() => expect(screen.getByText('JawIOGripper · Tool-DO0 · Backen OFFEN (gezählt, kein Sensor)')).toBeTruthy())
    cleanup()
    localStorage.setItem('willy.view', 'demo')
    const jaw = { kind: 'jaw', driver: 'DummyGripper', where: '', connected: true, jaws: 'not_counted', why_unknown: '', no_sensor: true }
    serve({ ...CELL, gripper: 'DummyGripper', hand: jaw })
    renderWith(<App />)
    await waitFor(() => expect(screen.getByText('Backengreifer · kein Sensor')).toBeTruthy())
    expect(screen.queryByText(/DummyGripper/)).toBeNull()
  })

  it('says a halt as a halt, and a standing stop record on the run chip, never as a controller to restart', async () => {
    serve({ ...CELL, halted: { reason: 'the operator pressed halt now', requested_at: 1, in_motion: true, braked: false } })
    renderWith(<App />)
    await waitFor(() => expect(screen.getByText('angehalten')).toBeTruthy())
    expect(screen.getByText('angehalten').closest('.chip')?.className).toContain('block')
    cleanup()
    // The record is the run's business: first the person's "Zelle ist frei", then Restart or Home. On the controller
    // chip, "Neustart nötig" read like a reboot of the controller; that chip says the controller.
    const ready = { ready: true, blockers: [], lights: [{ id: 'robot', state: 'ok', code: 'connected', message: '', blocks: true }] }
    const record = { run_id: 'run-7', kind: 'task', stop_code: 'halted', at: 2, holding: false, cleared_at: null }
    serve({ ...CELL, recovery: record }, { '/v1/cell/readiness': ready })
    renderWith(<App />)
    await waitFor(() => expect(screen.getByTitle('Lauf: Stopp offen: Zelle freigeben')).toBeTruthy())
    expect(screen.getByTitle('Lauf: Stopp offen: Zelle freigeben').className).toContain('block')
    await waitFor(() => expect(screen.getByTitle('Steuerung: bereit')).toBeTruthy())
    expect(screen.queryByText(/Neustart nötig/)).toBeNull()
    cleanup()
    serve({ ...CELL, recovery: { ...record, cleared_at: 3 } }, { '/v1/cell/readiness': ready })
    renderWith(<App />)
    await waitFor(() => expect(screen.getByTitle('Lauf: Stopp offen: Neustart oder Home')).toBeTruthy())
  })

  it('says a latched arm and a standing stop record while the cell is not connected', async () => {
    const latch = { reason: 'the operator pressed halt now', requested_at: 1, in_motion: false, braked: false }
    serve({ ...CELL, state: 'built', halted: latch })
    renderWith(<App />)
    // Connect refuses a latched arm ('halted'): the chip says so before anything else.
    await waitFor(() => expect(screen.getByText('angehalten')).toBeTruthy())
    cleanup()
    // The record outlives a Disconnect: the run chip keeps saying it, and the controller chip says what is true.
    const record = { run_id: 'run-7', kind: 'task', stop_code: 'return_failed', at: 2, holding: false, cleared_at: null }
    serve({ ...CELL, state: 'disconnected', recovery: record })
    renderWith(<App />)
    await waitFor(() => expect(screen.getByTitle('Lauf: Stopp offen: Zelle freigeben')).toBeTruthy())
    expect(screen.getByTitle('Steuerung: nicht verbunden')).toBeTruthy()
  })

  it('says a protective stop on the controller chip while the stop record waits on the run chip; a halt before both', async () => {
    const record = { run_id: 'run-7', kind: 'task', stop_code: 'controller_stopped', at: 2, holding: false, cleared_at: null }
    const stopped = { ready: false, blockers: [], lights: [{ id: 'robot', state: 'blocked', code: 'controller_stopped', message: '', blocks: true }] }
    serve({ ...CELL, recovery: record }, { '/v1/cell/readiness': stopped })
    renderWith(<App />)
    await waitFor(() => expect(screen.getByTitle('Steuerung: gestoppt')).toBeTruthy())
    expect(screen.getByTitle('Lauf: Stopp offen: Zelle freigeben')).toBeTruthy()
    cleanup()
    const latch = { reason: 'the operator pressed halt now', requested_at: 1, in_motion: true, braked: true }
    serve({ ...CELL, recovery: record, halted: latch }, { '/v1/cell/readiness': stopped })
    renderWith(<App />)
    await waitFor(() => expect(screen.getByTitle('Steuerung: angehalten')).toBeTruthy())
    expect(screen.queryByTitle('Steuerung: gestoppt')).toBeNull()
  })

  it('says a pick run stops before its next attempt, never "after the part", which is a task\'s stop', async () => {
    const sockets = fakeSockets()
    const record = { id: 'run-p', kind: 'pick', state: 'running', stop_code: '', stop_class: '', prompt: 'cube', started_at: 1 }
    serve({ ...CELL, active_run_id: 'run-p' }, { '/v1/runs/run-p': record })
    renderWith(<App />)
    await waitFor(() => expect(sockets.find('run-p')).toBeTruthy())
    const event = (seq: number, type: string, data: Record<string, unknown>) => ({
      type, run_id: 'run-p', seq, ts: 1 + seq, severity: 'info', human: type, step: '', step_index: null, step_total: null, data,
    })
    act(() => sockets.find('run-p')!.deliver([event(1, 'run_started', { kind: 'pick', prompt: 'cube' }), event(2, 'run_stop_requested', { scope: 'between_attempts' })]))
    expect(screen.getByText('Pick-Lauf · stoppt vor dem nächsten Versuch')).toBeTruthy()
  })

  it('says a run the restarted server forgot is no longer known, rather than "no run"', async () => {
    const sockets = fakeSockets()
    const server = { active: 'run-t' as string | null, known: true }
    const record = { id: 'run-t', kind: 'task', state: 'running', stop_code: '', stop_class: '', prompt: '', started_at: 1 }
    serve(CELL, {
      '/v1/cell': () => ({ ...CELL, active_run_id: server.active }),
      '/v1/runs/run-t': () => (server.known ? record : refusal(404, 'no_such_run', 'There is no run with that id.')),
    })
    renderWith(<App />)
    await waitFor(() => expect(sockets.find('run-t')).toBeTruthy())
    const started = { type: 'run_started', run_id: 'run-t', seq: 1, ts: 2, severity: 'info', human: '', step: '', step_index: null, step_total: null, data: { kind: 'task' } }
    act(() => sockets.find('run-t')!.deliver([started]))
    expect(screen.getByText('Auftrag · läuft')).toBeTruthy()
    server.active = null
    server.known = false
    await waitFor(() => expect(screen.getByText('Auftrag: nicht mehr bekannt')).toBeTruthy(), { timeout: 4000 })
    expect(screen.getByText('Auftrag: nicht mehr bekannt').closest('.chip')?.className).toContain('warn')
  })

  it('says "no server" when the cell cannot be read, never an empty cell', async () => {
    stubApi({ '/v1/cell': refusal(0, 'http_0', 'no answer') })
    vi.stubGlobal('fetch', vi.fn(async () => {
      throw new TypeError('Failed to fetch')
    }))
    renderWith(<App />)
    await waitFor(() => expect(screen.getByText('kein Server')).toBeTruthy())
  })

  it('switches the language, and the whole shell follows', () => {
    serve()
    renderWith(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'EN' }))
    const nav = screen.getByRole('navigation', { name: 'Main navigation' })
    expect([...nav.querySelectorAll('a')].map((a) => a.textContent)).toEqual(['Cockpit', 'Setup', 'History', 'Settings'])
    expect(localStorage.getItem('willy.lang')).toBe('en')
  })

  it('switches between dark and light, and stamps the document', () => {
    serve()
    renderWith(<App />)
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark')
    fireEvent.click(screen.getByRole('button', { name: 'Zum hellen Design wechseln' }))
    expect(document.documentElement.getAttribute('data-theme')).toBe('light')
    expect(screen.getByRole('button', { name: 'Zum dunklen Design wechseln' })).toBeTruthy()
  })

  it('turns voice output on with one click, off by default', () => {
    serve()
    renderWith(<App />)
    const voice = screen.getByRole('button', { name: 'Sprachausgabe aus' })
    expect(voice.getAttribute('aria-pressed')).toBe('false')
    fireEvent.click(voice)
    expect(screen.getByRole('button', { name: 'Sprachausgabe an' }).getAttribute('aria-pressed')).toBe('true')
    expect(localStorage.getItem('willy.voiceOut')).toBe('on')
  })

  it('opens the audience window as its own popup', () => {
    serve()
    const open = vi.fn()
    vi.stubGlobal('open', open)
    renderWith(<App />)
    fireEvent.click(screen.getByRole('button', { name: /Publikumsfenster/ }))
    expect(open).toHaveBeenCalledWith('/demo.html', 'willy-audience', 'popup')
  })

  it('offers Diagnostics in the tech view only', async () => {
    serve(CELL, { '/v1/diagnostics': refusal(503, 'no_robot_configured', 'none') })
    renderWith(<App />)
    expect(screen.queryByRole('button', { name: 'Diagnose' })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Technik' }))
    fireEvent.click(screen.getByRole('button', { name: 'Diagnose' }))
    expect(screen.getByRole('dialog', { name: 'Diagnose' })).toBeTruthy()
  })
})

describe('the jaws question, over every screen', () => {
  it('opens the global dialog with exactly the question\'s choices, none of them focused, and no way out', async () => {
    serve({ ...CELL, jaws_question: true })
    const sockets = fakeSockets()
    renderWith(<App />)
    await waitFor(() => expect(sockets.find('cell')).toBeTruthy())
    act(() =>
      sockets.find('cell')!.deliver([
        {
          type: 'cell.jaws_question',
          run_id: 'cell',
          seq: 1,
          ts: Date.now() / 1000,
          severity: 'warn',
          human: 'Where do the jaws stand?',
          step: '',
          step_index: null,
          step_total: null,
          data: {
            question_id: 'q-1',
            stage: 'where',
            at: 'connect',
            where: 'tool output 0',
            choices: ['open', 'closed'],
            expires_at: Date.now() / 1000 + 100,
          },
        },
      ]),
    )
    // The shell mounts the dialog (jaws/JawsDialog.tsx): an alertdialog above every screen, not a notice.
    const dialog = screen.getByRole('alertdialog', { name: 'Backenfrage' })
    expect(dialog.textContent).toMatch(/Wo stehen die Backen an Tool-DO0/)
    expect(dialog.textContent).toMatch(/nie „offen“/)
    // Exactly the server's choices, in its order, and no default: the focus is on the dialog, so Enter answers nothing.
    const choices = [...dialog.querySelectorAll('button[data-choice]')].map((button) => button.getAttribute('data-choice'))
    expect(choices).toEqual(['open', 'closed'])
    expect(document.activeElement).toBe(dialog)
    fireEvent.keyDown(dialog, { key: 'Escape' })
    expect(screen.getByRole('alertdialog', { name: 'Backenfrage' })).toBeTruthy()
  })
})
