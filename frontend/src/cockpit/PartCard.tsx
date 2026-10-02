/**
 * One card per part in the chat (OD 12; build plan 4.2): the grasp's overlay as its thumbnail, and in a few tags what
 * the pick saw and did: the looks it fused, the jaw faces it saw, the hand-eye gap, how far it pushed the part free,
 * and the hold, said as "nicht gemessen (kein Sensor)" on a hand that measures nothing, never as "held".
 *
 * The part in hand shows its step lines as they happen, under "Willy · Teil n", and the step it is on. A finished part
 * folds into its card; the tech view opens its lines (with the backend's own sentences) under it.
 *
 * Looks that found nothing more are how "until empty" ends, not a part that failed: a "part" whose every pick saw
 * nothing matching (or only parts already placed) is said in one quiet line, "Nichts mehr gefunden (2 Blicke).", never
 * as a card "nicht abgelegt". Empty looks are never counted as grasps.
 */

import { useT } from '../i18n'
import { Icon } from '../icons'
import type { ChatLine, PartView, PickView, StepId } from '../model/runModel'
import ChatLineView from './ChatLine'
import { COCKPIT } from './i18n'

export interface PartCardProps {
  readonly part: PartView
  readonly lines: readonly ChatLine[]
  readonly current: boolean
  /** The step the part in hand is on, for its live line ("Greift …"). */
  readonly step: StepId | null
  readonly tech: boolean
}

/** The pick the card speaks for: the one that succeeded, else the last one. */
function pickOf(part: PartView): PickView | null {
  return [...part.picks].reverse().find((pick) => pick.succeeded) ?? part.picks.at(-1) ?? null
}

/** A look that saw nothing to grasp: nothing matching, or only parts the task keeps out (those it placed). */
function empty(pick: PickView): boolean {
  return pick.foundNothing || pick.onlyExcluded
}

export default function PartCard({ part, lines, current, step, tech }: PartCardProps) {
  const t = useT(COCKPIT)
  const shown = lines.filter((line) => tech || line.level === 'demo')

  if (!current && part.placed !== true && part.picks.length > 0 && part.picks.every(empty)) {
    const placedOnly = part.picks.every((p) => p.onlyExcluded)
    return (
      <div className="ck-line ck-part-none">
        <span className="ck-line-text">{t(placedOnly ? 'ck.part.onlyPlaced' : 'ck.part.nothingLeft', { n: part.picks.length })}</span>
        {tech && shown.length > 0 && (
          <details className="ck-part-lines">
            <summary>{t('ck.part.steps')}</summary>
            {shown.map((line) => (
              <ChatLineView key={line.id} line={line} tech={tech} step />
            ))}
          </details>
        )}
      </div>
    )
  }

  const pick = pickOf(part)
  const grasps = part.picks.filter((p) => !empty(p)).length

  const tags: Array<{ text: string; tone?: 'warn' | 'ok' }> = []
  if (pick) {
    const fused = pick.looksFused.length
    if (fused > 1) tags.push({ text: t('ck.part.fused', { n: fused }) })
    else if (pick.looks.length > 0) tags.push({ text: t('ck.part.looks', { n: pick.looks.length }) })
    if (pick.faces && pick.faces.length > 0) {
      const seen = pick.faces.filter(Boolean).length
      tags.push({ text: t(seen >= 2 ? 'ck.part.facesBoth' : seen === 1 ? 'ck.part.facesOne' : 'ck.part.facesNone') })
    }
    if (pick.handEyeMm !== null) {
      const warn = pick.handEyeWarnMm !== null && pick.handEyeMm > pick.handEyeWarnMm
      if (warn) tags.push({ text: t('ck.part.handEyeWarn', { mm: pick.handEyeMm, warn: pick.handEyeWarnMm }), tone: 'warn' })
      else if (tech) tags.push({ text: t('ck.part.handEye', { mm: pick.handEyeMm }) })
    }
  }
  if (part.pushedMm > 0) tags.push({ text: t('ck.part.pushed', { mm: Math.round(part.pushedMm) }), tone: 'warn' })
  if (pick?.succeeded && pick.holdMeasured === false) tags.push({ text: t('ck.part.holdNo') })
  if (pick?.succeeded && pick.holdMeasured === true) tags.push({ text: t('ck.part.holdYes') })
  if (grasps > 1) tags.push({ text: t('ck.part.picks', { n: grasps }) })
  if (part.picks.length > 0 && part.picks.every(empty)) {
    tags.push({ text: t(part.picks.every((p) => p.onlyExcluded) ? 'ck.part.onlyExcluded' : 'ck.part.nothing') })
  }
  if (part.picks.some((p) => p.detectorFailed)) tags.push({ text: t('ck.part.detectorFailed'), tone: 'warn' })

  const status =
    part.placed === true
      ? part.durationS !== null
        ? t('ck.part.placedIn', { time: t.fmt.duration(part.durationS) })
        : t('ck.part.placed')
      : part.placed === false
        ? t('ck.part.notPlaced')
        : t('ck.part.running')
  const tone = part.placed === true ? 'ok' : part.placed === false ? 'off' : 'now'

  return (
    <article className={`ck-part ck-card ${tone}${current ? ' current' : ''}`}>
      <div className="ck-thumb">
        {part.overlay ? (
          <img src={part.overlay} alt={t('ck.part.thumb', { part: part.part })} loading="lazy" />
        ) : (
          <Icon name="cube" size={26} />
        )}
      </div>
      <div className="ck-part-body">
        <header className="ck-part-head">
          <strong>{t('ck.part.title', { part: part.part })}</strong>
          <span className={`ck-part-status ${tone}`}>{status}</span>
        </header>
        {tags.length > 0 && (
          <ul className="ck-tags">
            {tags.map((tag) => (
              <li key={tag.text} className={tag.tone ?? ''}>
                {tag.text}
              </li>
            ))}
          </ul>
        )}
        {tech && pick?.graspMm && (
          <div className="ck-human">
            {t('ck.part.grasp', { x: Math.round(pick.graspMm[0]), y: Math.round(pick.graspMm[1]), z: Math.round(pick.graspMm[2]) })}
            {pick.viewsFile ? ` · ${t('ck.part.views', { file: pick.viewsFile })}` : ''}
          </div>
        )}
        {current ? (
          <div className="ck-steps-live">
            {shown.map((line) => (
              <ChatLineView key={line.id} line={line} tech={tech} step />
            ))}
            {step && <div className="ck-line now">{t(`ck.now.${step}`)}</div>}
          </div>
        ) : (
          tech &&
          shown.length > 0 && (
            <details className="ck-part-lines">
              <summary>{t('ck.part.steps')}</summary>
              {shown.map((line) => (
                <ChatLineView key={line.id} line={line} tech={tech} step />
              ))}
            </details>
          )
        )}
      </div>
    </article>
  )
}
