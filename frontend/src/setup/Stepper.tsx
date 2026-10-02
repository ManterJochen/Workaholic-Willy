/**
 * The steps to a ready cell, in one row: Prüfen -> Aufbauen -> Vorschau -> Verbinden -> Bereit (build plan 4.3).
 *
 * Each step is marked from what the server said (`steps.ts`): done, done with notes, now, or not yet; the step to do
 * now carries `aria-current="step"`. A click shows that step's panel and moves nothing: the panels hold the buttons,
 * and every button that moves says so.
 */

import { Icon } from '../icons'
import { useT } from '../i18n'
import { SETUP } from './i18n'
import { STEP_ORDER, type StepId, type StepMark } from './steps'

export interface StepperProps {
  marks: Record<StepId, StepMark>
  /** The step whose panel is on screen. */
  shown: StepId
  onPick: (step: StepId) => void
}

export default function Stepper({ marks, shown, onPick }: StepperProps) {
  const t = useT(SETUP)
  return (
    <nav className="su-stepper">
      <ol aria-label={t('su.steps')}>
        {STEP_ORDER.map((id, index) => {
          const mark = marks[id]
          return (
            <li key={id} className={`su-step ${mark}`} aria-current={mark === 'current' ? 'step' : undefined}>
              <button type="button" aria-pressed={shown === id} onClick={() => onPick(id)}>
                <span className="su-num" aria-hidden="true">
                  {mark === 'done' ? <Icon name="check" size={14} /> : mark === 'warn' ? '!' : index + 1}
                </span>
                <span className="su-name">{t(`su.step.${id}`)}</span>
                <span className="sr-only">, {t(`su.mark.${mark}`)}</span>
              </button>
            </li>
          )
        })}
      </ol>
    </nav>
  )
}
