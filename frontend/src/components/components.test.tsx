/**
 * The shared pieces every area builds on: the confirm dialog before a motion, the Diagnostics drawer, and the error
 * banner that says a refusal in the reader's language without hiding the server's own sentence.
 *
 * The confirm dialog is the one place a click becomes motion outside Start (Restart and Home keep a dialog, build plan
 * item 12), so its safety behaviour is pinned: it names the first motion, it says when a countdown and a straight leg
 * come first, Enter or a stray click does not confirm, Escape and the backdrop cancel, and one confirm is one call.
 */

import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ApiError } from '../api/client'
import { I18nProvider } from '../i18n'
import { refusal, stubApi } from '../test/render'
import ConfirmDialog from './ConfirmDialog'
import DiagnosticsDrawer from './DiagnosticsDrawer'
import { Chip, ErrorBanner, Loading } from './ui'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

function homeDialog(props: Partial<Parameters<typeof ConfirmDialog>[0]> = {}) {
  const onConfirm = vi.fn()
  const onCancel = vi.fn()
  render(
    <I18nProvider lang="de">
      <ConfirmDialog
        open
        title="Home anfahren?"
        firstMotion="geplante Fahrt nach Home"
        confirmLabel="Home – der Roboter fährt"
        moves
        onConfirm={onConfirm}
        onCancel={onCancel}
        {...props}
      />
    </I18nProvider>,
  )
  return { onConfirm, onCancel }
}

describe('ConfirmDialog', () => {
  it('names the first motion and the straight leg a planned move may begin with', () => {
    homeDialog()
    const dialog = screen.getByRole('dialog', { name: 'Home anfahren?' })
    expect(dialog.getAttribute('aria-modal')).toBe('true')
    expect(screen.getByText('Erste Bewegung: geplante Fahrt nach Home.')).toBeTruthy()
    expect(screen.getByText(/geraden Stück von höchstens 10° je Gelenk/)).toBeTruthy()
    expect(screen.queryByText(/3 s Countdown/)).toBeNull()
  })

  it('says the countdown when one is due', () => {
    homeDialog({ countdown: true })
    expect(screen.getByText('Hände weg: 3 s Countdown, dann die erste Bewegung.')).toBeTruthy()
  })

  it('draws a confirm that moves in the lime accent, and gives the focus to Cancel, not to the motion', () => {
    homeDialog()
    const confirm = screen.getByRole('button', { name: 'Home – der Roboter fährt' })
    expect(confirm.className).toContain('primary')
    expect(confirm.className).not.toContain('danger')
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Abbrechen' }))
  })

  it('cancels on Escape and on the backdrop, and never confirms on either', () => {
    const { onConfirm, onCancel } = homeDialog()
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
    fireEvent.click(screen.getByTestId('dialog-backdrop'))
    expect(onCancel).toHaveBeenCalledTimes(2)
    expect(onConfirm).not.toHaveBeenCalled()
  })

  it('sends one confirm for a double click', () => {
    const { onConfirm } = homeDialog()
    const confirm = screen.getByRole('button', { name: 'Home – der Roboter fährt' })
    fireEvent.click(confirm)
    fireEvent.click(confirm)
    expect(onConfirm).toHaveBeenCalledTimes(1)
  })

  it('renders nothing while closed', () => {
    homeDialog({ open: false })
    expect(screen.queryByRole('dialog')).toBeNull()
  })
})

describe('DiagnosticsDrawer', () => {
  it('reads the diagnostics when it opens, and says nothing here moves', async () => {
    const calls = stubApi({
      '/v1/diagnostics': {
        vendors: [{ vendor: 'ur', kind: 'arm', registered: true, ready: true, note: 'ur_rtde found', sdks: [] }],
        motion_stack: {
          model: 'ur10', model_source: 'config', fully_anchored: true, curobo_available: true, curobo_python: 'py',
          curobo_caveat: '', mesh_bundle_present: true,
        },
        perception: {
          pipeline_configured: true, kind: 'zero_shot', backend: 'vlm', segmenter: 'sam2', router_enabled: false,
          vlm_model_id: 'Qwen3-VL', vlm_weights_present: true, detail: 'The VLM is configured and its weights are here.',
        },
        reachability: { checked: true, address: '192.168.0.10', port: 30004, reachable: true, latency_ms: 2.5, detail: '' },
      },
    })
    const onClose = vi.fn()
    render(
      <I18nProvider lang="de">
        <DiagnosticsDrawer open onClose={onClose} />
      </I18nProvider>,
    )
    expect(screen.getByRole('dialog', { name: 'Diagnose' })).toBeTruthy()
    expect(screen.getByText(/Von hier aus bewegt sich nichts/)).toBeTruthy()
    await waitFor(() => expect(screen.getByText('ur10')).toBeTruthy())
    expect(screen.getByText('Qwen3-VL')).toBeTruthy()
    expect(screen.getByText('erreichbar in 2,5 ms')).toBeTruthy()
    expect(calls.map((c) => `${c.method} ${c.path}`)).toEqual(['GET /v1/diagnostics'])
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
    expect(onClose).toHaveBeenCalled()
  })

  it('shows a refusal as a banner rather than an empty drawer', async () => {
    // The drawer belongs to the tech view, where a refusal also shows the server's own sentence.
    localStorage.setItem('willy.view', 'tech')
    stubApi({ '/v1/diagnostics': refusal(503, 'no_robot_configured', 'no robot in the config') })
    render(
      <I18nProvider lang="en">
        <DiagnosticsDrawer open onClose={() => undefined} />
      </I18nProvider>,
    )
    await waitFor(() => expect(screen.getByText('No robot is configured.')).toBeTruthy())
    expect(screen.getByText('no robot in the config')).toBeTruthy()
    localStorage.removeItem('willy.view')
  })
})

describe('ErrorBanner', () => {
  it('says a known refusal in the reader\'s language, and its code, which can be quoted', () => {
    const error = new ApiError(409, { code: 'restart_required', message: 'the stop of run-7 stands', detail: {} }, '')
    render(
      <I18nProvider lang="de">
        <ErrorBanner error={error} />
      </I18nProvider>,
    )
    expect(screen.getByText('Nach dem Stopp geht es nur mit Neustart oder Home weiter.')).toBeTruthy()
    // The server's English sentence and the code are details (OD adopted; little jargon in the demo view): the demo
    // view keeps them folded behind "Details", closed until a person opens it, so a generic refusal ("Der Aufbau wurde
    // abgelehnt.") still has its reason one click away, and the code is there to quote.
    const sentence = screen.getByText('the stop of run-7 stands')
    const folded = sentence.closest('details')
    expect(folded).toBeTruthy()
    expect(folded!.open).toBe(false)
    expect(within(folded!).getByText('Details')).toBeTruthy()
    expect(within(folded!).getByText(/restart_required · HTTP 409/)).toBeTruthy()
  })

  it('folds the code of a refusal with no sentence of its own too, in the demo view', () => {
    const error = new ApiError(409, { code: 'restart_required', message: '', detail: {} }, '')
    render(
      <I18nProvider lang="de">
        <ErrorBanner error={error} />
      </I18nProvider>,
    )
    expect(screen.getByText(/restart_required · HTTP 409/).closest('details')).not.toBeNull()
  })

  it('puts the server\'s own sentence and the code under it in the tech view, unfolded', () => {
    localStorage.setItem('willy.view', 'tech')
    const error = new ApiError(409, { code: 'restart_required', message: 'the stop of run-7 stands', detail: {} }, '')
    render(
      <I18nProvider lang="de">
        <ErrorBanner error={error} />
      </I18nProvider>,
    )
    expect(screen.getByText('the stop of run-7 stands').closest('details')).toBeNull()
    expect(screen.getByText(/restart_required · HTTP 409/).closest('details')).toBeNull()
    localStorage.removeItem('willy.view')
  })

  it('says an unknown code as "Refused (<code>)", never as silence', () => {
    const error = new ApiError(409, { code: 'brand_new_refusal', message: 'something new', detail: {} }, '')
    render(
      <I18nProvider lang="de">
        <ErrorBanner error={error} />
      </I18nProvider>,
    )
    expect(screen.getByText('Abgelehnt (brand_new_refusal)')).toBeTruthy()
  })

  it('names a server that does not answer as such', () => {
    const error = new ApiError(0, null, 'No answer from the backend at http://localhost:3000. Is it running?')
    render(<ErrorBanner error={error} />)
    expect(screen.getByText(/No answer from the backend/)).toBeTruthy()
  })
})

describe('the small pieces', () => {
  it('Loading answers in English without a provider, as the screen tests expect', () => {
    render(<Loading what="the cell" />)
    expect(screen.getByText('Reading the cell…')).toBeTruthy()
  })

  it('a chip carries its label and its tone', () => {
    render(<Chip label="Hand" tone="ok">Backen OFFEN</Chip>)
    const chip = screen.getByText('Backen OFFEN').closest('.chip')
    expect(chip?.className).toContain('ok')
    expect(screen.getByText('Hand')).toBeTruthy()
  })
})
