/**
 * Ready? The readiness lights of the cell (build plan 1.5, OD 16), each with what a person can do about it here: start
 * the planner (it moves nothing; about a minute), or ask where the jaws stand (the question opens in the browser). A
 * blocker that belongs to a stop (the cell not cleared, a restart owed, a part still held) is answered on the cockpit's
 * stop card, and this panel sends the person there rather than answering it twice. While a stop record stands that
 * nobody has said the cell is clear of, the jaws wait too: the server refuses the check (`cell_not_cleared`) until
 * "Zelle ist frei", so the button stays off and says so, rather than offering a question nobody will be asked.
 *
 * Nothing here moves the arm. The server enforces every gate on its own; these lights only say what it will answer.
 */

import { useId, useState } from 'react'
import { Link } from 'react-router-dom'

import { ApiError, api, type LightOut } from '../api/client'
import { ErrorBanner } from '../components/ui'
import { useT } from '../i18n'
import { blockerMsg, lightIdMsg, lightMsg } from '../i18n/codes'
import { usePrefs } from '../model/prefs'
import { useCell } from '../model/useCell'
import { useRun } from '../model/useRun'
import { SETUP } from './i18n'

/** A light's tone class. Not `block` for a blocked one: that is also Tailwind's `display: block`, and the utilities
 *  layer beats the row's grid whatever the specificity. */
const TONE: Record<LightOut['state'], string> = { ok: 'ok', wait: 'warn', blocked: 'alarm', info: 'info' }

/** Blockers a stop owns: the cockpit's stop card answers them. */
const STOP_CARD = new Set(['cell_not_cleared', 'restart_required', 'part_still_held', 'needs_person'])

function asApiError(err: unknown): ApiError {
  return err instanceof ApiError ? err : new ApiError(0, null, String(err))
}

export default function ReadyPanel() {
  const t = useT(SETUP)
  const { view } = usePrefs()
  const { cell, readiness, refresh } = useCell()
  const { follow } = useRun()
  const base = useId()
  const [busy, setBusy] = useState<'planner' | 'jaws' | null>(null)
  const [error, setError] = useState<ApiError | null>(null)

  const connected = cell?.state === 'connected'
  const running = Boolean(cell?.active_run_id)
  // A stop nobody has said the cell is clear of: the jaws wait for "Zelle ist frei", as every motion does.
  const record = cell?.recovery
  const uncleared = record != null && !(record.cleared_at != null && record.cleared_at > record.at)

  const startPlanner = async () => {
    setBusy('planner')
    setError(null)
    try {
      follow(await api.startPlanner())
    } catch (err) {
      setError(asApiError(err))
    } finally {
      setBusy(null)
      refresh()
    }
  }

  const checkJaws = async () => {
    setBusy('jaws')
    setError(null)
    try {
      // Blocks until the question ended (up to 120 s per stage); the question itself opens in the jaws dialog.
      await api.checkJaws()
    } catch (err) {
      setError(asApiError(err))
    } finally {
      setBusy(null)
      refresh()
    }
  }

  const action = (light: LightOut) => {
    if (light.id === 'planner' && (light.code === 'off' || light.code === 'failed')) {
      const note = `${base}-planner`
      return (
        <>
          <button
            type="button"
            className="big"
            aria-describedby={note}
            disabled={busy !== null || running}
            onClick={() => void startPlanner()}
          >
            {busy === 'planner' ? t('su.act.planner.busy') : t('su.act.planner')}
          </button>
          <small id={note}>{t('su.act.planner.note')}</small>
        </>
      )
    }
    if (light.id === 'gripper' && (light.code === 'jaws_unknown' || light.code === 'jaws_closed')) {
      const note = `${base}-jaws`
      return (
        <>
          <button
            type="button"
            className="big"
            aria-describedby={note}
            disabled={busy !== null || running || uncleared || cell?.jaws_question === true}
            onClick={() => void checkJaws()}
          >
            {busy === 'jaws' ? t('su.act.jaws.busy') : t('su.act.jaws')}
          </button>
          <small id={note}>{t(uncleared ? 'su.act.jaws.uncleared' : 'su.act.jaws.note')}</small>
        </>
      )
    }
    return null
  }

  const lights = readiness?.lights ?? []
  const blockers = readiness?.blockers ?? []
  const toStopCard = blockers.some((b) => STOP_CARD.has(b.code))

  return (
    <section className="panel su-ready" aria-label={t('su.ready.title')}>
      <div className="panel-head">
        <h3>{t('su.ready.title')}</h3>
        {readiness && (
          <span className={`su-verdict ${readiness.ready ? 'yes' : 'no'}`}>
            {readiness.ready ? t('su.ready.yes') : t('su.ready.no')}
          </span>
        )}
      </div>
      <p className="panel-lede">{t('su.ready.lede')}</p>
      {!connected && <p className="small faint">{t('su.ready.notConnected')}</p>}
      {connected && !readiness && <p className="small faint">{t('su.ready.unknown')}</p>}
      {error && <ErrorBanner error={error} />}

      {lights.length > 0 && (
        <ul className="su-lights">
          {lights.map((light) => (
            <li key={light.id} className={`su-light ${TONE[light.state]}`}>
              <span className="su-dot" aria-hidden="true" />
              <span className="su-light-name">{t.msg(lightIdMsg(light.id))}</span>
              <span className="su-light-code">{t.msg(lightMsg(light.code))}</span>
              {!light.blocks && <span className="su-light-info">{t('su.ready.info')}</span>}
              {view === 'tech' && light.message && <span className="su-light-msg">{light.message}</span>}
              <span className="su-light-act">{action(light)}</span>
            </li>
          ))}
        </ul>
      )}

      {blockers.length > 0 && (
        <div className="su-blockers">
          <h4>{t('su.ready.blockers')}</h4>
          <ul>
            {blockers.map((blocker) => (
              <li key={blocker.code}>
                {t.msg(blockerMsg(blocker.code))}
                {view === 'tech' && blocker.message && <span className="su-light-msg">{blocker.message}</span>}
              </li>
            ))}
          </ul>
          {toStopCard && (
            <Link className="su-link" to="/">
              {t('su.ready.toStopCard')}
            </Link>
          )}
        </div>
      )}

      {readiness?.ready && (
        <div className="actions su-go">
          <Link className="su-link primary" to="/">
            {t('su.ready.cockpit')}
          </Link>
        </div>
      )}
    </section>
  )
}
