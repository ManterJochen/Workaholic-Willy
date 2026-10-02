/**
 * The task in numbers, on top of the right column beside the live image (OD 17; owner decisions, fifth round): the
 * success rate leads (the brand's readout box), then the parts placed in this task and the median time per part; the
 * tech view adds the pushes, the empty looks and a count per outcome. Subtle and factual: a number nobody measured is a
 * dash, never a zero.
 *
 * The success rate counts placed parts over the picks whose end is known (`StatsView.rated`): an empty look is how
 * "until empty" ENDS, a part on its way is not decided yet (so the rate never dips while a part is carried), and a pick
 * or a part a halt cut short was stopped by a person; none of them is a failure (`runModel.computeStats`).
 *
 * Every tile reads number first, then its name, then what the number counts: the three numbers stand on one line, the
 * success rate leading in the brand's readout frame and its lime.
 */

import { useT } from '../i18n'
import { outcomeMsg } from '../i18n/codes'
import type { RunView } from '../model/runModel'
import { COCKPIT } from './i18n'

export default function Stats({ view, tech }: { view: RunView; tech: boolean }) {
  const t = useT(COCKPIT)
  const stats = view.stats
  const real = stats.rated
  const task = view.kind !== 'pick'
  const rate = stats.successRate !== null ? t.fmt.percent(stats.successRate) : '–'
  const rateSub =
    view.runId === null
      ? t('ck.stats.rateNone')
      : real <= 0
        ? t(task && view.holding ? 'ck.stats.rateHolding' : 'ck.stats.rateNoReal')
        : task
          ? t('ck.stats.rateOf', { placed: stats.placed, real })
          : t('ck.stats.rateOfPick', { won: stats.succeeded, real })
  const median = stats.medianPartS !== null ? t.fmt.duration(stats.medianPartS) : '–'

  return (
    <section className={`ck-stats${tech ? ' tech' : ''}`} aria-label={t('ck.stats.label')}>
      <div className="readout ck-stat-lead">
        <span className="big">{rate}</span>
        <span className="ck-stat-name">{t('ck.stats.rate')}</span>
        <span className="ck-stat-sub">{rateSub}</span>
      </div>
      <div className="ck-stat">
        <b className="mono ck-stat-num">{view.runId === null ? '0' : stats.placed}</b>
        <span className="ck-stat-name">{t('ck.stats.parts')}</span>
        <span className="ck-stat-sub">{t('ck.stats.partsSub')}</span>
      </div>
      <div className="ck-stat">
        <b className="mono ck-stat-num">{median}</b>
        <span className="ck-stat-name">{t('ck.stats.time')}</span>
        <span className="ck-stat-sub">{t('ck.stats.timeSub')}</span>
      </div>
      {tech && (
        <div className="ck-stat">
          <b className="mono ck-stat-num">{stats.pushes}</b>
          <span className="ck-stat-name">{t('ck.stats.pushes')}</span>
          <span className="ck-stat-sub">{t('ck.stats.pushesSub', { picks: stats.picks, empty: stats.emptyLooks })}</span>
          {Object.keys(stats.outcomes).length > 0 && (
            <span className="ck-stat-outcomes mono">
              {Object.entries(stats.outcomes)
                .map(([outcome, count]) => `${t.msg(outcomeMsg(outcome))} ${count}`)
                .join(' · ')}
            </span>
          )}
        </div>
      )}
    </section>
  )
}
