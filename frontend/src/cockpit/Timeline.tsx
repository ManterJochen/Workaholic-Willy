/**
 * The task's steps under the live image (OD 2: look → detect → grasp → place → return), with the hands-off countdown
 * and the camera target's survey as chips of their own, and where the task stands: part n of N (∞ until empty) and the
 * pick's attempt n of N (build plan 1.4's timeline mapping, 4.2).
 *
 * A step is drawn by its state: idle, active (the accent), done (✓), a warning (seen, nothing usable), failed (✕) or
 * skipped. Its sub-chip says the look ("Blick 2/3") or the attempt.
 *
 * One step is filled at a time: while the wrist looks, each look's frame is also searched, so the model has the look
 * AND the detection going at once; the first of them in the plan's order is filled, a later one rides along outlined.
 */

import type { CellFactsOut } from '../api/client'
import { useT } from '../i18n'
import type { Msg } from '../i18n/types'
import { Icon } from '../icons'
import { stopMsg } from '../i18n/codes'
import { hasEnded, lookNote, phaseMsg, plannedLooks, type RunView, type StepView } from '../model/runModel'
import { COCKPIT } from './i18n'

/** The look's sub-chip in the cockpit's words: "Blick 2/3", in English "view 2/3" beside the step "Look". */
function lookWords(note: Msg | null): Msg<string> | null {
  if (note?.key === 'step.lookOf') return { key: 'ck.tl.lookOf', params: note.params }
  if (note?.key === 'step.lookN') return { key: 'ck.tl.lookN', params: note.params }
  return note
}

export default function Timeline({ view, facts }: { view: RunView; facts: CellFactsOut | null }) {
  const t = useT(COCKPIT)
  const running = view.runId !== null && !hasEnded(view.phase) && view.phase !== 'idle'
  const total = plannedLooks(view.plan, facts)
  const lead = view.timeline.find((step) => step.state === 'active')?.id ?? null

  const note = (step: StepView): Msg<string> | null => {
    if (step.id === 'look' && view.current.lookIndex > 0 && step.state === 'active') return lookWords(lookNote(view, total))
    return lookWords(step.note)
  }

  let where: string
  if (view.kind === 'teach' && running) where = t('ck.tl.teach')
  else if (view.kind === 'planner' && running) where = t('ck.tl.planner')
  else if (view.current.part !== null) {
    const parts = t('ck.tl.part', { part: view.current.part, of: view.current.of ?? '∞' })
    const attempt =
      view.current.attempt !== null && view.current.attemptTotal !== null
        ? t('step.attempt', { n: view.current.attempt, total: view.current.attemptTotal })
        : null
    where = attempt ? `${parts} · ${attempt}` : parts
  } else if (view.runId && hasEnded(view.phase) && view.stopCode) where = t.msg(stopMsg(view.stopCode))
  else where = view.runId ? t.msg(phaseMsg(view)) : t('ck.tl.idle')

  return (
    <div className="ck-timeline">
      {view.countdown.state === 'active' && (
        <span className="ck-chip warn">
          <Icon name="alert" size={15} />
          {t('ck.tl.countdown', { seconds: view.countdown.secondsLeft ?? '…' })}
        </span>
      )}
      {view.survey.state !== 'none' && (
        <span className={`ck-chip ${view.survey.state === 'failed' ? 'failed' : view.survey.state === 'done' ? 'done' : 'active'}`}>
          {view.survey.state === 'done' && <Icon name="check" size={15} />}
          {t('step.survey')}
        </span>
      )}
      <ol className="ck-steps" aria-label={t('ck.tl.label')}>
        {view.timeline.map((step, index) => {
          const sub = note(step)
          const along = step.state === 'active' && step.id !== lead
          return (
            <li
              key={step.id}
              className={`ck-step ${step.state}${along ? ' along' : ''}`}
              aria-current={step.state === 'active' && !along ? 'step' : undefined}
            >
              {index > 0 && <Icon name="chevronRight" size={14} className="ck-step-sep" />}
              <span className="ck-step-pill">
                {step.state === 'done' && <Icon name="check" size={15} />}
                {step.state === 'failed' && <Icon name="close" size={15} />}
                {step.state === 'warn' && <Icon name="alert" size={15} />}
                <span>{t(`step.${step.id}`)}</span>
                {sub && step.state !== 'idle' && <span className="ck-step-note">{t.msg(sub)}</span>}
              </span>
            </li>
          )
        })}
      </ol>
      <span className="ck-where mono">{where}</span>
    </div>
  )
}
