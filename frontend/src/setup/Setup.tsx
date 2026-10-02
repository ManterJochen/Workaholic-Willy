/**
 * Setup (`/setup`, formerly `/cell`): Prüfen -> Aufbauen -> Vorschau -> Verbinden -> Bereit, then the poses taught by
 * freedrive and the cell's facts (build plan 4.3; the owner's approved preview: the stepper across, the step's panel
 * beside the cell's facts, the poses below).
 *
 * **One step at a time, and always the one the server says.** The stepper follows the cell's own state (`steps.ts`):
 * the panel of the step to do now is shown, and a click on another step shows its panel until the cell moves on. The
 * Check and Cell screens' panels are reused as they are (`screens/Preflight`, `screens/Cell`), over the ONE shared
 * cell poll (`useCell`), so this page and the top bar can never disagree about the cell.
 *
 * **The check is never skipped.** A cell that is down opens at the checklist, whatever it found, and goes on to
 * Aufbauen only at the person's "Weiter: Aufbauen" (or a click on a later step); after that the check wears a warning,
 * never a tick, while the checklist warns or blocks (a cell without fixtures, without a planning world).
 *
 * **What moves, and what does not.** Building touches no robot and builds for real by default where the arm is real;
 * the preview a person read issues the token, and Connect is the one red button, which says the robot moves. A latched
 * arm (a halt, or a disconnect during a move) is shown above every step, and is cleared only by a person's word after
 * they ticked that they looked. The jaws question that a toggle hand asks at Connect opens in the global dialog
 * (`jaws/JawsDialog`). Teaching frees the arm only at the teach dialog's own red button.
 *
 * **Bereit reads the lights.** The raw telemetry (the TCP, the quaternion, the joints, what the controller says) is the
 * tech view's; the demo view has little jargon (OD 1), and the top bar's cell chip still says what is on the other end.
 */

import { useCallback, useRef, useState } from 'react'

import { api } from '../api/client'
import { ErrorBanner, ScreenHead } from '../components/ui'
import { useT } from '../i18n'
import { useAsync } from '../lib/useAsync'
import { usePrefs } from '../model/prefs'
import { useCell } from '../model/useCell'
import { rehearseByDefault, useCellActions } from '../screens/cellActions'
import {
  BuildPanel,
  ConnectPanel,
  LatchPanel,
  PreviewPanel,
  StatePanel,
  SubstitutionBanner,
  TelemetryPanel,
} from '../screens/Cell'
import { PreflightView } from '../screens/Preflight'
import CellFacts from './CellFacts'
import { SETUP } from './i18n'
import PosesPanel from './PosesPanel'
import ReadyPanel from './ReadyPanel'
import Stepper from './Stepper'
import { currentStep, readCheckSeen, stepMarks, writeCheckSeen, type SetupState, type StepId } from './steps'
import './setup.css'

export default function Setup() {
  const t = useT(SETUP)
  const tech = usePrefs().view === 'tech'
  const { cell, readiness, refresh } = useCell()
  const preflight = useAsync(() => api.preflight())
  /** A step the person picked, kept only while the cell stays at the step it was picked from, and until an action. */
  const [picked, setPicked] = useState<{ step: StepId; at: StepId } | null>(null)
  /** The person read the checklist and went on, in this browser session. */
  const [checkSeen, setCheckSeen] = useState(readCheckSeen)
  /** The stepper, brought back into view when the long checklist gives way to the next step's panel. */
  const steps = useRef<HTMLDivElement | null>(null)
  // After every action the stepper follows the cell again: a build shows the preview next, a connect the ready lights.
  const changed = useCallback(() => {
    refresh()
    setPicked(null)
  }, [refresh])
  const actions = useCellActions(changed)
  const [rehearse, setRehearse] = useState<boolean | null>(null)

  const state: SetupState = {
    cellState: cell?.state ?? null,
    preflightBlocking: preflight.data?.n_blocking ?? null,
    preflightWarnings: preflight.data?.n_warn ?? null,
    checkSeen,
    previewRead: actions.preview !== null && cell?.state === 'built',
    ready: readiness?.ready ?? null,
  }
  const current = currentStep(state)
  const marks = stepMarks(state)
  const shown = picked && picked.at === current ? picked.step : current

  /** The person has read the checklist: the check is passed for this session, and the stepper follows the cell again. */
  const leaveTheCheck = () => {
    writeCheckSeen()
    setCheckSeen(true)
    setPicked(null)
    steps.current?.scrollIntoView?.({ block: 'start', behavior: 'smooth' })
  }

  const pick = (step: StepId) => {
    // Going on from the checklist that is on screen is reading it: the same as "Weiter: Aufbauen".
    const leaving = shown === 'check' && step !== 'check' && !checkSeen
    if (leaving) {
      writeCheckSeen()
      setCheckSeen(true)
    }
    const now = leaving ? currentStep({ ...state, checkSeen: true }) : current
    setPicked(step === now ? null : { step, at: now })
  }

  const panel = () => {
    if (shown === 'check' || !cell) {
      return (
        <section className="panel su-check">
          <div className="panel-head">
            <h3>{t('su.check.title')}</h3>
          </div>
          <p className="panel-lede">{t('su.check.lede')}</p>
          <PreflightView
            data={preflight.data}
            error={preflight.error}
            loading={preflight.loading}
            reload={preflight.reload}
          />
          {cell?.state === 'disconnected' && preflight.data && (
            <div className="actions su-check-next">
              <button type="button" className="primary big" onClick={leaveTheCheck}>
                {t('su.check.next')}
              </button>
              <span className="small faint">{t('su.check.nextNote')}</span>
            </div>
          )}
        </section>
      )
    }
    switch (shown) {
      case 'build':
        return (
          <>
            <BuildPanel
              cell={cell}
              actions={actions}
              rehearse={rehearse ?? rehearseByDefault(cell)}
              onRehearse={setRehearse}
            />
            <StatePanel cell={cell} />
          </>
        )
      case 'preview':
        return <PreviewPanel cell={cell} actions={actions} />
      case 'connect':
        return (
          <>
            <ConnectPanel cell={cell} actions={actions} />
            {actions.preview && <PreviewPanel cell={cell} actions={actions} />}
          </>
        )
      case 'ready':
        return (
          <>
            <ReadyPanel />
            {tech && <TelemetryPanel connected={cell.state === 'connected'} />}
          </>
        )
    }
  }

  return (
    <div className="page su-page">
      <ScreenHead title={t('nav.setup')} lede={t('screen.setup.lede')} />
      <div ref={steps} className="su-steps-anchor">
        <Stepper marks={marks} shown={shown} onPick={pick} />
      </div>
      {cell && <LatchPanel cell={cell} actions={actions} />}
      {cell && <SubstitutionBanner cell={cell} />}
      {actions.error && <ErrorBanner error={actions.error} />}
      <div className="su-grid">
        <div className="su-step-panel">{cell || shown === 'check' ? panel() : <p className="small faint">{t('su.waiting')}</p>}</div>
        <CellFacts />
      </div>
      <PosesPanel />
    </div>
  )
}
