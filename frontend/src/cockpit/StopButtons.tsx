/**
 * The two stops and Home, under the live image (OD 20; build plan 4.2).
 *
 * **"Nach diesem Teil stoppen"** asks the task to end after the part in hand: a held part is still placed and the arm
 * returns, then the task ends (`POST /v1/task/stop`). It never stops the arm. For a Home run the same button stops the
 * run before its one move is sent; for a pick run, before its next attempt (`POST /v1/pick/stop`).
 *
 * **"Sofort anhalten"** is ONE click, no dialog (`POST /v1/cell/brake`): the run commands nothing more and, where the
 * arm latches, nothing more is sent to it. It is the console's halt, NOT the emergency stop, and says so; it is drawn
 * in the halt orange, never the e-stop's red. Where the arm brakes a move in flight it says "bremst kontrolliert", and
 * only there does a halt the arm has not confirmed within 1.5 s raise the red "Anhalten nicht bestätigt: Not-Aus
 * drücken". Where the arm only latches (the owner's cell as shipped) it says "hält vor der nächsten Bewegung", and no
 * false alarm is raised.
 *
 * Whether the arm brakes is read from the cell's facts, and also from the halt itself: the run's own word that a brake
 * is pending (`run_halt_requested.braking`), or a brake outcome on the latch (only an arm that brakes reports one). A
 * facts read that failed never silences the alarm. The arm's own "brake not confirmed" raises it whenever it comes in,
 * also after the halted run ended, until the latch is cleared; a controller that says it is stopped (the e-stop was
 * pressed, or a protective stop) has answered it, for that halt.
 *
 * **Home** is a planned move, so it asks first (the confirm dialog names it); it is off, saying why, wherever the
 * server would refuse it.
 *
 * **Calm while nothing runs.** With no run and no halt under way nothing moves, so the halt is drawn calm (the same
 * filled orange with its dark label, darker and without its glow; the full orange under the pointer or the focus), as
 * the approved preview showed it quiet at rest, and the live image stays the brightest thing on screen; it is still one
 * click. The moment a run holds the cell it is the full orange.
 * The stop row lies above every layer that is not about safety (the Diagnostics drawer's backdrop included), so the
 * one click is never spent closing something else (`cockpit.css` `.ck-stops`).
 */

import { useState } from 'react'

import { ApiError, api, type CellFactsOut } from '../api/client'
import { ErrorBanner } from '../components/ui'
import { useT } from '../i18n'
import { refusalMsg } from '../i18n/codes'
import { Icon } from '../icons'
import { haltState, type RunView } from '../model/runModel'
import { useCell } from '../model/useCell'
import { useNow } from './hooks'
import { COCKPIT } from './i18n'
import { useMotions } from './motions'
import { armBrakes, controllerStopped, homeRefusal } from './recovery'

export interface StopButtonsProps {
  readonly view: RunView
  /** A run holds the cell (the poll's word, or the view's own). */
  readonly running: boolean
  readonly facts: CellFactsOut | null
}

export default function StopButtons({ view, running, facts }: StopButtonsProps) {
  const t = useT(COCKPIT)
  const { cell, readiness, refresh } = useCell()
  const motions = useMotions()
  const [error, setError] = useState<ApiError | null>(null)
  const [pressedAt, setPressedAt] = useState<number | null>(null)
  const [stopping, setStopping] = useState(false)
  /** The latch (by its request time) whose missing brake confirmation a stopped controller answered. */
  const [answered, setAnswered] = useState<number | null>(null)

  const connected = cell?.state === 'connected'
  const latch = cell?.halted ?? null
  const brakes = armBrakes(facts, view, latch)
  const waiting = view.halt.state === 'requested'
  const now = useNow(250, waiting || pressedAt !== null)
  const state = haltState(view.halt, now, brakes, latch)
  // The controller says it cannot move (the e-stop was pressed, or a protective stop): the arm stands, whatever its
  // brake reported. That answers the alarm for this latch, also once the stop is released at the pendant.
  const controllerDown = controllerStopped(readiness)
  if (latch && controllerDown && answered !== latch.requested_at) setAnswered(latch.requested_at)
  const alarm = state === 'unconfirmed' && !controllerDown && !(latch !== null && answered === latch.requested_at)
  // The arm's own word, rather than the console's 1.5 s window, says the brake was not confirmed.
  const armSaid = latch?.brake === 'unconfirmed'
  // A press whose run event has not come in yet (or a halt with no run): "requested" until the cell or the run says it.
  const pressedOnly = pressedAt !== null && now - pressedAt < 5 && view.halt.state === 'none' && !latch
  // "Angehalten" while the arm's latch stands (until "Zelle ist frei"), or while the halted run is still on its way out.
  const halted = Boolean(latch) || (running && state === 'confirmed')
  const requested = state === 'requested' || pressedOnly

  const runId = view.runId
  const kind = running ? view.kind : null
  const asked =
    (kind === 'task' && view.stopScope === 'after_part') ||
    ((kind === 'home' || kind === 'wave') && view.stopScope === 'before_motion') ||
    (kind === 'pick' && view.stopScope === 'between_attempts')
  const canStop =
    running && runId !== null && (kind === 'task' || kind === 'home' || kind === 'pick' || kind === 'wave') && !asked && !stopping

  const stopTitle = kind === 'home' ? t('ck.stopHome') : kind === 'wave' ? t('ck.stopWave') : kind === 'pick' ? t('ck.stopPick') : t('ck.stopAfter')
  const stopSub =
    kind === 'home'
      ? t(asked ? 'ck.stopHomeAsked' : 'ck.stopHomeSub')
      : kind === 'wave'
        ? t(asked ? 'ck.stopWaveAsked' : 'ck.stopWaveSub')
        : kind === 'pick'
        ? t('ck.stopPickSub')
        : t(asked ? 'ck.stopAfterAsked' : 'ck.stopAfterSub')

  const stop = async () => {
    if (!canStop || runId === null) return
    setStopping(true)
    setError(null)
    try {
      if (kind === 'pick') await api.stopPick(runId)
      else await api.stopTask(runId)
      refresh()
    } catch (err: unknown) {
      setError(err instanceof ApiError ? err : new ApiError(0, null, String(err)))
    } finally {
      setStopping(false)
    }
  }

  const halt = async () => {
    // One click, no dialog, never disabled by a run: the halt is what a person reaches for when something is wrong.
    setPressedAt(Date.now() / 1000)
    setError(null)
    try {
      await api.brake()
      refresh()
    } catch (err: unknown) {
      setError(err instanceof ApiError ? err : new ApiError(0, null, String(err)))
    }
  }

  const homeOff = homeRefusal(cell, readiness, running)
  // Nothing runs and no halt is under way: nothing moves, and the halt is drawn calm (still enabled, still one click).
  const calm = !running && !requested && !alarm

  return (
    <section className="ck-stops" aria-label={t('ck.stops.label')}>
      <div className="ck-stop-row">
        <button
          type="button"
          className="ck-stopbtn ck-stopafter"
          disabled={!canStop}
          onClick={() => void stop()}
          title={t('ck.stopAfterTitle')}
        >
          <span className="ck-stopbtn-title">
            <Icon name="stop" size={17} />
            {stopTitle}
          </span>
          <small>{stopSub}</small>
        </button>

        <button
          type="button"
          className={`halt ck-stopbtn ck-halt${calm ? ' calm' : ''}`}
          disabled={!connected}
          onClick={() => void halt()}
          title={t(brakes ? 'ck.haltTitleBrake' : 'ck.haltTitleLatch')}
        >
          <span className="ck-stopbtn-title">
            <Icon name="halt" size={18} />
            {t('ck.halt')}
          </span>
          <small>{t(brakes ? 'ck.haltSubBrake' : 'ck.haltSubLatch')}</small>
        </button>

        {(requested || halted) && !alarm && (
          <span className={`ck-halt-state${halted ? ' done' : ''}`} aria-live="polite">
            {halted ? t('ck.haltConfirmed') : t('ck.haltRequested')}
          </span>
        )}
        <button
          type="button"
          className="ghost ck-stopbtn ck-home"
          disabled={homeOff !== null || motions.busy}
          onClick={() => void motions.home()}
          title={homeOff ? t.msg(refusalMsg(homeOff)) : t('ck.homeTitle')}
        >
          <span className="ck-stopbtn-title">
            <Icon name="home" size={17} />
            {t('ck.home')}
          </span>
          <small>{t('ck.homeSub')}</small>
        </button>
      </div>

      {alarm && (
        <div className="ck-alarm" role="alert">
          <Icon name="alert" size={26} />
          <div>
            <strong>{t('ck.haltAlarm')}</strong>
            <span>{t(armSaid ? 'ck.haltAlarmSubArm' : 'ck.haltAlarmSub')}</span>
          </div>
        </div>
      )}
      {error && <ErrorBanner error={error} />}
      {motions.error && <ErrorBanner error={motions.error} />}
    </section>
  )
}
