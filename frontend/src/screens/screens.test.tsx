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

import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { provenanceOf } from '../lib/provenance'
import Cell from './Cell'
import Config from './Config'
import History from './History'
import Pick from './Pick'
import Preflight from './Preflight'

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

describe('Pick', () => {
  it('states that Stop does not stop the arm', async () => {
    stubApi({ '/v1/cell': CELL_CONNECTED })
    draw(<Pick />)
    await waitFor(() => expect(screen.getByText(/does not stop the arm/i)).toBeTruthy())
  })

  it('marks the pick button as moving the robot', async () => {
    stubApi({ '/v1/cell': CELL_CONNECTED })
    draw(<Pick />)
    await waitFor(() => expect(screen.getByText(/this moves the robot/i)).toBeTruthy())
  })

  it('refuses to offer a pick while the cell is down', async () => {
    stubApi({ '/v1/cell': { ...CELL_CONNECTED, state: 'disconnected' } })
    draw(<Pick />)
    await waitFor(() => expect(screen.getByText(/The cell is not connected/)).toBeTruthy())
  })
})

describe('History', () => {
  it('shows where the numbers came from, above the numbers', async () => {
    stubApi({ '/v1/history/kpis': ROLLUP, '/v1/history/records': [], '/v1/history/runs': [] })
    draw(<History />)
    // Twice on purpose: the rollup names both its `source` and the file it read.
    await waitFor(() => expect(screen.getAllByText(/grasp_records\.jsonl/).length).toBe(2))
  })

  it('lists an unmeasurable KPI instead of hiding or zeroing it', async () => {
    stubApi({ '/v1/history/kpis': ROLLUP, '/v1/history/records': [], '/v1/history/runs': [] })
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
