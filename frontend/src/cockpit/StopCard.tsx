/**
 * The stop card (OD 9, OD 14; build plan 1.3.5): a moving run stopped on a problem, the arm stands where it stopped,
 * and nothing moves until a person has gone through the card.
 *
 * It is rebuilt from the cell's stop record (`CellOut.recovery`) and the run it names (`GET /v1/runs/{id}`), never from
 * "the newest run", which may be a planner or a teach run; so it survives a reload, and it stays until the record ends
 * (a Restart's or a Home run's arrival at its return pose).
 *
 * **What happened**: the stop's title and sentence, the part and the step (the backend's own sentence in the tech view).
 * **What you do**, a live checklist in this order where it applies: a stopped controller is cleared at the pendant
 * (never here); "Zelle ist frei" (the person's word: `POST /v1/cell/acknowledge`); the jaws: a toggle hand's question
 * is asked by the cell (`POST /v1/cell/jaws/check`) and answered in its own window, never here and never with a
 * default; another hand is emptied and confirmed "Backen leer"; a hand among parts is jogged clear first; then Restart
 * or Home, each behind its confirm dialog, enabled only when every gate is green.
 *
 * Restart replays the stopped run to ITS return pose, so it waits for the stopped run's record (read again until it
 * answers) and names that pose; Home needs no record. Where the arm said its brake was not confirmed, the card does not
 * say the arm stands where it stopped: the arm never said so.
 *
 * Whether a part may still be in the jaws is the server's word (`recovery.partHeld`), the stop record's own belief
 * included: a hand that measures nothing is offered "Backen leer" for as long as the server would refuse the way back
 * for it. A refusal of Restart or Home is said right above the two buttons and brought into view, never only under the
 * card's foot, where the chat may cut it off.
 */

import { useState } from 'react'

import { ApiError, api, type PosesOut } from '../api/client'
import { ErrorBanner } from '../components/ui'
import { useT } from '../i18n'
import { runKindMsg, stopMsg, stopSayMsg, whereMsg } from '../i18n/codes'
import { Icon } from '../icons'
import type { StepId } from '../model/runModel'
import { useCell } from '../model/useCell'
import { poseLabel } from './draft'
import { useBroughtIntoView, type RunRecord } from './hooks'
import { COCKPIT } from './i18n'
import { useMotions } from './motions'
import { allGreen, controllerStopped, isToggle, recoveryGates, type GateId } from './recovery'

export interface StopCardProps {
  /** The stopped run's record as far as it is read (`run` null while it is read, `failed` while a read is refused). */
  readonly record: RunRecord
  /** A Home run's target (`home` or a pose name), where the run on screen is the stopped one: its record lacks it. */
  readonly homeTo: string | null
  /** A run holds the cell (the Restart or the Home move that is the way back): the card folds to one line. */
  readonly running: boolean
  /** Where the run stopped, as the run on screen saw it: the part and the step. */
  readonly part: number | null
  readonly step: StepId | null
  readonly poses: PosesOut | null
  readonly tech: boolean
}

const GATE_KEY: Record<GateId, 'ck.stop.gate.controller' | 'ck.stop.gate.cleared' | 'ck.stop.gate.jaws' | 'ck.stop.gate.empty' | 'ck.stop.gate.idle'> = {
  controller: 'ck.stop.gate.controller',
  cleared: 'ck.stop.gate.cleared',
  jaws: 'ck.stop.gate.jaws',
  empty: 'ck.stop.gate.empty',
  idle: 'ck.stop.gate.idle',
}

export default function StopCard({ record: stoppedRun, homeTo, running, part, step, poses, tech }: StopCardProps) {
  const t = useT(COCKPIT)
  const { cell, readiness, refresh } = useCell()
  const motions = useMotions()
  const [busy, setBusy] = useState<'clear' | 'jaws' | 'empty' | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const refused = useBroughtIntoView<HTMLDivElement>(error, motions.error)
  const record = cell?.recovery
  if (!cell || !record) return null

  const run = stoppedRun.run
  const gates = recoveryGates(cell, readiness)
  const ok = (id: GateId) => gates.find((gate) => gate.id === id)?.ok ?? true
  const green = allGreen(gates)
  const toggle = isToggle(cell)
  const stopped = controllerStopped(readiness)
  const title = t.msg(stopMsg(record.stop_code))
  const restartable = record.kind === 'task' || record.kind === 'home'
  // Where Restart goes first: a task's own return pose, from its record; a Home run's own target, from the run on
  // screen (its record does not carry it). Unknown until read: then Restart waits, and never names a guess.
  const target: string | null = record.kind === 'task' ? (run?.plan ? run.plan.return_to || 'home' : null) : homeTo
  const known = target !== null
  const to = !known
    ? '…'
    : target === 'home'
      ? t('common.home')
      : (record.kind === 'task' && run?.plan?.return_label) || poseLabel(target, poses) || target
  const where = whereMsg(cell.hand?.where)
  const plan = run?.plan ?? null
  const then = plan
    ? t('ck.confirm.restartThen', {
        what: plan.object_said || plan.object || t('common.anything'),
        where: plan.place.kind === 'camera' ? plan.place.said || plan.place.phrase || '—' : plan.place.pose_label || plan.place.pose || t('common.defaultPlace'),
      })
    : null
  // The arm itself said its brake was not confirmed: it never said it stands, so neither does the card.
  const brakeUnconfirmed = record.stop_code === 'halted' && cell.halted?.brake === 'unconfirmed'

  const act = async (what: 'clear' | 'jaws' | 'empty', request: () => Promise<unknown>) => {
    setBusy(what)
    setError(null)
    try {
      await request()
    } catch (err: unknown) {
      setError(err instanceof ApiError ? err : new ApiError(0, null, String(err)))
    } finally {
      setBusy(null)
      refresh()
    }
  }

  const cleared = ok('cleared')
  const idle = ok('idle')
  // No hand is read while the cell is down: which hand it is, and what it holds, is said once it is connected.
  const handRead = cell.state === 'connected'
  const header = (
    <header className="ck-card-head">
      <Icon name="alert" size={22} className="ck-alarm-icon" />
      <h3>{title}</h3>
      <span className="ck-card-meta">
        {part !== null && step !== null
          ? t('ck.run.stoppedAt', { part, at: t(`ck.at.${step}`) })
          : t('ck.run.stoppedRun', { kind: runKindMsg(record.kind) })}
      </span>
    </header>
  )
  if (running) {
    // The way back is under way: nothing to do here until the arm arrives (the record then ends, and the card with it).
    return (
      <section className="ck-card ck-stopcard folded" aria-label={title}>
        {header}
        <p className="ck-quiet">{t('ck.stop.wayBack')}</p>
      </section>
    )
  }
  // Acknowledge is allowed while the cell is BUILT or CONNECTED (a Disconnect may have cut the run), never during a run,
  // and not while a connected controller says it is stopped (the pendant first).
  const canClear = (cell.state === 'built' || cell.state === 'connected') && !cell.active_run_id && !(cell.state === 'connected' && stopped)

  return (
    <section className="ck-card ck-stopcard hud-frame" aria-label={title}>
      {header}

      <p className="ck-say">{brakeUnconfirmed ? t('ck.stop.haltedUnconfirmed') : t.msg(stopSayMsg(record.stop_code))}</p>
      {tech && run?.error && <p className="ck-human">{run.error}</p>}
      {restartable && !known && <p className="ck-quiet">{t(stoppedRun.failed ? 'ck.stop.readFailed' : 'ck.stop.loading')}</p>}

      <h4 className="ck-card-sub">{t('ck.stop.do')}</h4>
      <ol className="ck-checklist">
        {stopped && (
          <li className="ck-check">
            <Icon name="controller" size={16} />
            <span>{t('ck.stop.step.pendant')}</span>
          </li>
        )}
        <li className={`ck-check${cleared ? ' done' : ''}`}>
          <Icon name={cleared ? 'check' : 'eye'} size={16} />
          <span>{cleared ? t('ck.stop.cleared') : t('ck.stop.step.clear')}</span>
          {!cleared && (
            <button
              type="button"
              className="primary big"
              disabled={busy !== null || !canClear}
              title={t('ck.act.clearTitle')}
              onClick={() => void act('clear', () => api.acknowledge(false))}
            >
              {t('ck.act.clear')}
            </button>
          )}
        </li>
        {!handRead ? (
          <li className={`ck-check${ok('empty') ? ' done' : ''}`}>
            <Icon name={ok('empty') ? 'check' : 'gripper'} size={16} />
            <span>{ok('empty') ? t('ck.stop.empty') : t('ck.stop.step.jawsLater')}</span>
          </li>
        ) : toggle ? (
          <li className={`ck-check${ok('jaws') ? ' done' : ''}`}>
            <Icon name={ok('jaws') ? 'check' : 'gripper'} size={16} />
            <span>{ok('jaws') ? t('ck.stop.jawsOpen') : t('ck.stop.step.jawsToggle', { where })}</span>
            {!ok('jaws') && (
              <button
                type="button"
                className="big"
                disabled={busy !== null || !cleared || !idle || stopped}
                title={t('ck.act.jawsTitle')}
                onClick={() => void act('jaws', () => api.checkJaws())}
              >
                {t('ck.act.jaws')}
              </button>
            )}
            {busy === 'jaws' && <span className="ck-quiet">{t('ck.stop.jawsWaiting')}</span>}
          </li>
        ) : (
          <li className={`ck-check${ok('empty') ? ' done' : ''}`}>
            <Icon name={ok('empty') ? 'check' : 'gripper'} size={16} />
            <span>{ok('empty') ? t('ck.stop.empty') : t('ck.stop.step.jawsOther')}</span>
            {!ok('empty') && (
              <button
                type="button"
                className="big"
                disabled={busy !== null || !cleared || !idle}
                onClick={() => void act('empty', () => api.acknowledge(true))}
              >
                {t('ck.stop.emptyButton')}
              </button>
            )}
          </li>
        )}
        <li className="ck-check info">
          <Icon name="info" size={16} />
          <span>{t('ck.stop.step.clearHand')}</span>
        </li>
        <li className="ck-check info">
          <Icon name="restart" size={16} />
          <span>{t('ck.stop.step.back')}</span>
        </li>
      </ol>

      <div className="ck-gates" role="group" aria-label={t('ck.stop.gates')}>
        {gates.map((gate) => (
          <span key={gate.id} className={`ck-gate${gate.ok ? ' ok' : ''}`}>
            <Icon name={gate.ok ? 'check' : 'close'} size={14} />
            {t(GATE_KEY[gate.id])}
          </span>
        ))}
      </div>

      {/* What the server refused, right above the buttons a person just pressed, and brought into view. */}
      {(error || motions.error) && (
        <div className="ck-refusal" ref={refused}>
          {error && <ErrorBanner error={error} />}
          {motions.error && <ErrorBanner error={motions.error} />}
        </div>
      )}

      <div className="ck-card-actions">
        {restartable && (
          <button
            type="button"
            className="primary big ck-go"
            disabled={!green || !known || motions.busy}
            onClick={() => void motions.restart(record.run_id, to, record.kind === 'task' ? then : null)}
          >
            <span>{t('ck.stop.restart', { to })}</span>
            {cell.countdown_due && <small>{t('ck.start.countdown')}</small>}
          </button>
        )}
        <button type="button" className="ghost big ck-go" disabled={!green || motions.busy} onClick={() => void motions.home()}>
          <span>{t('ck.stop.home')}</span>
          {cell.countdown_due && <small>{t('ck.start.countdown')}</small>}
        </button>
      </div>
      {!green && <p className="ck-quiet">{t('ck.stop.gates')}</p>}

      {/* A pick run (a program's `POST /v1/pick`) is not restarted: Home is its way back. */}
      {!restartable && <p className="ck-quiet">{t('ck.stop.pickNoRestart')}</p>}
    </section>
  )
}
