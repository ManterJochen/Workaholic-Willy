/**
 * The ask card (build plan 1.2, "ask" class; 4.2): the task stopped on a question, the arm is back at its return pose,
 * and a person chooses how it goes on: the camera did not find its target, lost it at the drop, cannot reach it, the
 * part does not fit, or the screen refused a pose.
 *
 * Every option that moves starts a NEW task, on a click, and is a Start of its own: its label names the first motion,
 * the 3 s hands-off countdown when one is due, and, for a target the camera finds, what happens if it is not found;
 * it is off while the cell is not ready. "Anderes Ziel" opens the Understood card with the task as it ran, to change;
 * "Beenden" closes the card. Nothing on it moves by itself. A refused option is said right above the options, and
 * brought into view.
 *
 * A sort (the owner, 2026-10-09) asks the same way where one of its bins was found nowhere, and says which: "Nochmal
 * suchen" sorts again with every rule, and "Anderes Ziel" opens the card with every rule as it ran. It offers no
 * "Standard-Ablage nehmen": which rule's parts would go there is no question a button can answer.
 */

import type { ApiError, PosesOut, TaskIn, TaskPlanOut } from '../api/client'
import { ErrorBanner } from '../components/ui'
import { useT } from '../i18n'
import { stopMsg, stopSayMsg } from '../i18n/codes'
import { Icon } from '../icons'
import { isSort, type AskCardView } from '../model/runModel'
import { taskOfPlan } from './draft'
import { useBroughtIntoView } from './hooks'
import { COCKPIT } from './i18n'

const CAMERA_ASKS = new Set(['target_not_found', 'target_lost', 'target_unreachable', 'part_does_not_fit'])
const WHY = new Set(['not_seen', 'moved_too_far', 'footprint_changed'])

export interface AskCardProps {
  readonly ask: AskCardView
  readonly plan: TaskPlanOut | null
  readonly poses: PosesOut | null
  /** The first motion a Start names ("der Roboter fährt zu Blick 1"). */
  readonly motion: string
  /** A 3 s hands-off countdown comes before that motion (`CellOut.countdown_due`). */
  readonly countdown: boolean
  /** Start is allowed now (the cell is ready, nothing runs, no stop record stands). */
  readonly canStart: boolean
  readonly busy: boolean
  readonly error: ApiError | null
  start(task: TaskIn): void
  edit(plan: TaskPlanOut): void
  end(): void
}

export default function AskCard({ ask, plan, poses, motion, countdown, canStart, busy, error, start, edit, end }: AskCardProps) {
  const t = useT(COCKPIT)
  const refused = useBroughtIntoView<HTMLDivElement>(error)
  const title = t.msg(stopMsg(ask.stopCode))
  const sort = isSort(plan)
  const places = plan ? [plan.place, ...(plan.more_rules ?? []).map((rule) => rule.place)] : []
  const camera = CAMERA_ASKS.has(ask.stopCode) && places.some((place) => place?.kind === 'camera')
  const defaultPlace = sort ? null : (poses?.default_place ?? null)
  /** An option that moves, labelled as Start is: the first motion, then the countdown and the camera's fallback. */
  const go = (option: string, task: TaskIn) => (
    <button type="button" className="primary big ck-go" disabled={!canStart || busy} onClick={() => start(task)}>
      <span className="ck-go-title">{t('ck.ask.option', { option, motion })}</span>
      {countdown && <small>{t('ck.start.countdown')}</small>}
      {[task.place, ...(task.more_rules ?? []).map((rule) => rule.place)].some((place) => place.kind === 'camera') && (
        <small>{t('ck.start.camera')}</small>
      )}
    </button>
  )

  return (
    <section className="ck-card ck-askcard hud-frame" aria-label={`${t('stopClass.ask')}: ${title}`}>
      <header className="ck-card-head">
        <Icon name="info" size={22} className="ck-ask-icon" />
        <h3>{title}</h3>
        {ask.part !== null && <span className="ck-card-meta">{t('ck.run.part', { part: ask.part })}</span>}
      </header>
      <p className="ck-say">{t.msg(stopSayMsg(ask.stopCode))}</p>
      {ask.nowhere && <p className="ck-quiet">{t.msg({ key: 'event.task.target_relocated.nowhere', params: { target: ask.nowhere } })}</p>}
      {ask.why && WHY.has(ask.why) && <p className="ck-quiet">{t.msg({ key: `event.task.target_lost.${ask.why}` })}</p>}
      <h4 className="ck-card-sub">{t('ck.ask.question')}</h4>
      {/* What the server refused, right above the options, and brought into view. */}
      {error && (
        <div className="ck-refusal" ref={refused}>
          <ErrorBanner error={error} />
        </div>
      )}
      <div className="ck-card-actions">
        {plan && camera && go(t('ck.ask.again'), taskOfPlan(plan, 'same'))}
        {plan && camera && defaultPlace && go(t('ck.ask.default'), taskOfPlan(plan, 'default'))}
        {plan && (
          <button type="button" className="big" onClick={() => edit(plan)}>
            {t(camera ? 'ck.ask.other' : 'ck.ask.otherPlace')}
          </button>
        )}
        <button type="button" className="ghost big" onClick={end}>
          {t('ck.ask.end')}
        </button>
      </div>
    </section>
  )
}
