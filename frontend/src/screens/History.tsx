/**
 * What this cell has done (build plan 4.3): this session's tasks in numbers and two charts, every run of the session
 * with its kind, its command and how it ended, and the grasp attempts from the log that outlives the process.
 *
 * **Two sources, never blurred.** The session's runs live in the server's memory and go with it; the attempt records
 * live in a file and survive restarts. Each panel says which it reads. The rollup states WHERE its numbers came from
 * (`source`, `record_log_path`) above the numbers, because a KPI panel without its source is the fastest way to
 * present a synthetic self-check as a production result.
 *
 * **The whole session, not a page of it.** The runs are read as the server keeps them (its newest 200, `RUNS_KEPT`),
 * so "Diese Sitzung" counts every task of a long day and not only the newest fifty a plain read lists; the table shows
 * the newest fifty, the rest one click away. Where the server has already let older runs go, the tiles say whose runs
 * they count.
 *
 * **The session's numbers are the cockpit's.** The success per task and the median time per part are replayed from
 * each task's own events through the cockpit's reducer (`historyStats.ts`), so an "until empty" task's two closing
 * empty looks never count against it, and no second formula can disagree with the cockpit's statistics. Where the
 * session ran more tasks than are replayed, those two tiles say they are of the newest twelve.
 *
 * **Nothing unmeasurable is zeroed.** A KPI the records cannot support is listed with its reason, its name on its own
 * line above the reason (never a chip squeezed beside it); a task without a value is "kein Wert", never 0.
 *
 * **Whose words.** A KPI is named in the reader's words and a rate reads as a percentage, as the tiles do; the
 * server's English reasons are details: open in the tech view, under "Warum (Server)" in the demo view.
 */

import { useId, useMemo, useState } from 'react'

import { api, type RecordOut, type RollupOut, type RunOut } from '../api/client'
import { isStopCode } from '../api/codes'
import { Caveat, Empty, ErrorBanner, Loading, Panel, ScreenHead, StatusPill } from '../components/ui'
import { useT } from '../i18n'
import { outcomeMsg, refusalMsg, runKindMsg, stopMsg } from '../i18n/codes'
import { outcomeTone, stopTone } from '../lib/outcome'
import { useAsync } from '../lib/useAsync'
import { usePrefs } from '../model/prefs'
import { useCell } from '../model/useCell'
import BarChart, { type Bar } from './BarChart'
import {
  median,
  niceCeiling,
  REPLAY_TASKS,
  replayable,
  useRunInputs,
  useTaskStats,
  type RunInputs,
  type TaskStats,
} from './historyStats'
import { SCREENS, type ScreensT } from './i18n'
import './screens.css'

/** The KPIs this console has words for; another is shown by the server's name. */
const KPI_NAMES = [
  'pick_success_rate',
  'first_attempt_success_rate',
  'dead_loop_rate',
  'safety_rejection_rate',
  'median_cycle_time_s',
  'dense_recovery_success_rate',
  'false_positive_grasp_rate',
  'median_attempt_seconds',
] as const

type KpiName = (typeof KPI_NAMES)[number]

function isKpiName(name: string): name is KpiName {
  return (KPI_NAMES as readonly string[]).includes(name)
}

/** A KPI's name in the reader's words, or the server's name for one this console does not know. */
function kpiName(t: ScreensT, name: string): string {
  return isKpiName(name) ? t(`hs.kpi.${name}`) : name
}

/** A KPI's value: a rate as the percentage the tiles use, any other number as itself. */
function kpiValue(t: ScreensT, name: string, value: unknown): string {
  if (typeof value !== 'number') return String(value)
  return name.endsWith('_rate') && value >= 0 && value <= 1 ? t.fmt.percent(value) : t.fmt.number(value, 3)
}

/** One path, whichever separator it is written with: a source that names the very file read is said once. */
function samePath(a: string, b: string): boolean {
  const norm = (path: string) => path.replace(/\\/g, '/').replace(/\/+$/, '').toLowerCase()
  return norm(a) === norm(b)
}

/** How many logged attempts are listed before "show all". */
const RECORDS_SHOWN = 20

/** Every run the server keeps (`api/runs.py` RETAINED_RUNS): the session is counted whole, not cut at a page of 50. */
const RUNS_KEPT = 200

/** How many runs the table lists before "show all": the newest, as a plain read of the server lists them. */
const RUNS_SHOWN = 50

function clock(t: ScreensT, unixSeconds: number): string {
  return new Intl.DateTimeFormat(t.lang === 'de' ? 'de-DE' : 'en-GB', { hour: '2-digit', minute: '2-digit', hour12: false }).format(
    new Date(unixSeconds * 1000),
  )
}

/** Seconds as a person reads them: two decimals under a second (a desk arm), one under 100 s, none above. */
function seconds(t: ScreensT, value: number): string {
  return `${t.fmt.number(value, value < 1 ? 2 : value < 100 ? 1 : 0)} s`
}

/** What a run was asked to do, in the operator's words where the plan kept them, and where a Home or teach run said. */
function commandWords(t: ScreensT, run: RunOut, inputs: RunInputs | undefined): { text: string; spoken: boolean } {
  const plan = run.plan
  if (run.kind === 'task' && plan) {
    if (plan.command?.text) return { text: plan.command.text, spoken: plan.command.source === 'spoken' }
    const what = plan.object_said || plan.object || t('common.anything')
    const place = plan.place
    const where =
      place.kind === 'pose'
        ? place.pose_label || place.pose || t('common.defaultPlace')
        : place.said || place.phrase || '—'
    return { text: `${what} ${t('hs.runs.to', { where })}`, spoken: false }
  }
  if (run.kind === 'pick') return { text: run.prompt || t('common.anything'), spoken: false }
  if (run.kind === 'home' && inputs?.to) {
    const to = inputs.to === 'home' ? t('common.home') : inputs.label || inputs.to
    return { text: t('hs.runs.home', { to }), spoken: false }
  }
  if (run.kind === 'teach' && (inputs?.label || inputs?.name)) {
    return { text: t('hs.runs.teach', { label: inputs.label || inputs.name }), spoken: false }
  }
  return { text: '—', spoken: false }
}

function ResultPill({ t, run }: { t: ScreensT; run: RunOut }) {
  if (run.state === 'running') return <StatusPill status="info">{t('hs.runs.running')}</StatusPill>
  if (run.refusal) return <StatusPill status="warn">{t.msg(refusalMsg(run.refusal.code))}</StatusPill>
  if (run.stop_code && isStopCode(run.stop_code)) {
    return <StatusPill status={stopTone(run.stop_code)}>{t.msg(stopMsg(run.stop_code))}</StatusPill>
  }
  return <StatusPill status="idle">{run.state}</StatusPill>
}

/**
 * Placed of gripped: a task's parts set down, of the grasps that held one; a pick run places nothing. The picks are
 * never the second number: an "until empty" task ends on two empty looks, which are no part.
 */
function parts(run: RunOut): string {
  if (run.kind === 'task') return `${run.parts_placed ?? 0} / ${run.succeeded ?? 0}`
  if (run.kind === 'pick') return `– / ${run.succeeded ?? 0}`
  return '—'
}

function SessionTiles({ t, runs, stats }: { t: ScreensT; runs: readonly RunOut[]; stats: ReadonlyMap<string, TaskStats> }) {
  const tasks = runs.filter((run) => run.kind === 'task')
  const placed = tasks.reduce((sum, run) => sum + (run.parts_placed ?? 0), 0)
  const replayed = replayable(runs)
    .map((run) => stats.get(run.id))
    .filter((s): s is TaskStats => s !== undefined)
  const real = replayed.reduce((sum, s) => sum + s.fairPicks, 0)
  const won = replayed.reduce((sum, s) => sum + s.placed, 0)
  const rate = real > 0 ? Math.min(1, won / real) : null
  const perPart = median(replayed.flatMap((s) => s.partSeconds))
  // The two tiles from the replays count the newest REPLAY_TASKS tasks only; said where the session ran more.
  const ofLast = tasks.length > REPLAY_TASKS ? ` · ${t('hs.kpi.lastN', { n: REPLAY_TASKS })}` : ''
  // The counts are the session's, unless the server has let its oldest runs go: then they are of the runs it keeps.
  const whose = runs.length >= RUNS_KEPT ? t('hs.session.kept', { n: RUNS_KEPT }) : t('hs.session.title')
  return (
    <section className="hs-tiles" aria-label={t('hs.session.title')}>
      <div className="hs-tile lead">
        <span className="hs-tile-k">{t('hs.kpi.rate')}</span>
        <span className="hs-tile-v">{rate === null ? '–' : t.fmt.percent(rate)}</span>
        <span className="hs-tile-note">
          {t('hs.kpi.rateNote')}
          {ofLast}
        </span>
      </div>
      <div className="hs-tile">
        <span className="hs-tile-k">{t('hs.kpi.placed')}</span>
        <span className="hs-tile-v">{placed}</span>
        <span className="hs-tile-note">{whose}</span>
      </div>
      <div className="hs-tile">
        <span className="hs-tile-k">{t('hs.kpi.tasks')}</span>
        <span className="hs-tile-v">{tasks.length}</span>
        <span className="hs-tile-note">{whose}</span>
      </div>
      <div className="hs-tile">
        <span className="hs-tile-k">{t('hs.kpi.perPart')}</span>
        <span className="hs-tile-v">{perPart === null ? '–' : seconds(t, perPart)}</span>
        <span className="hs-tile-note">
          {t('hs.kpi.perPartNote')}
          {ofLast}
        </span>
      </div>
    </section>
  )
}

function Charts({ t, runs, stats }: { t: ScreensT; runs: readonly RunOut[]; stats: ReadonlyMap<string, TaskStats> }) {
  const tasks = runs.filter((run) => run.kind === 'task')
  // Oldest first, left to right, as a time axis reads; numbered in the order the session ran them.
  const shown = replayable(runs).slice().reverse()
  const ordinal = (run: RunOut) => tasks.length - tasks.findIndex((task) => task.id === run.id)

  // Ticked by the task's number in the session (two tasks can start in one minute); the time is in the tooltip.
  const rateBars: Bar[] = shown.map((run) => {
    const s = stats.get(run.id)
    const n = ordinal(run)
    const when = clock(t, run.started_at)
    const value = s ? s.successRate : null
    const valueText = !s ? t('hs.kpi.pending') : value === null ? t('hs.chart.none') : t.fmt.percent(value)
    const parts = s ? t('hs.chart.parts', { placed: s.placed, picks: s.fairPicks }) : ''
    return {
      key: run.id,
      tick: `#${n}`,
      value,
      valueText,
      detail: [when, parts, s?.incomplete ? t('hs.chart.partial') : ''].filter(Boolean).join(' · '),
      label: t('hs.chart.bar', { n, when, value: valueText }),
    }
  })

  const times = shown.map((run) => stats.get(run.id)?.medianPartS ?? null)
  const longest = Math.max(0, ...times.filter((v): v is number => v !== null))
  const top = niceCeiling(longest > 0 ? longest * 1.1 : 10)
  const timeBars: Bar[] = shown.map((run, index) => {
    const s = stats.get(run.id)
    const n = ordinal(run)
    const when = clock(t, run.started_at)
    const value = times[index]
    const valueText = !s ? t('hs.kpi.pending') : value === null ? t('hs.chart.none') : seconds(t, value)
    return {
      key: run.id,
      tick: `#${n}`,
      value,
      valueText,
      detail: [when, s?.incomplete ? t('hs.chart.partial') : ''].filter(Boolean).join(' · '),
      label: t('hs.chart.bar', { n, when, value: valueText }),
    }
  })

  return (
    <div className="hs-charts">
      <BarChart
        title={t('hs.chart.rate.title')}
        note={t('hs.chart.rate.note', { n: shown.length })}
        bars={rateBars}
        max={1}
        ticks={[0, 0.5, 1]}
        tickText={(v) => t.fmt.percent(v)}
        empty={t('hs.chart.empty')}
      />
      <BarChart
        title={t('hs.chart.time.title')}
        note={t('hs.chart.time.note')}
        bars={timeBars}
        max={top}
        ticks={[0, top / 2, top]}
        tickText={(v) => `${t.fmt.number(v, top < 1 ? 2 : top < 10 ? 1 : 0)} s`}
        empty={t('hs.chart.empty')}
      />
    </div>
  )
}

function RunsTable({ t, runs, tech }: { t: ScreensT; runs: readonly RunOut[]; tech: boolean }) {
  const titleId = useId()
  const inputs = useRunInputs(runs)
  const [all, setAll] = useState(false)
  const listed = all ? runs : runs.slice(0, RUNS_SHOWN)
  return (
    <Panel title={<span id={titleId}>{t('hs.runs.title')}</span>} aside={<a href="/v1/history/runs.csv">runs.csv</a>}>
      {runs.length === 0 ? (
        <Empty>{t('hs.runs.empty')}</Empty>
      ) : (
        <>
          <div className="hs-table">
            <table aria-labelledby={titleId}>
              <thead>
                <tr>
                  <th>{t('hs.runs.when')}</th>
                  <th>{t('hs.runs.kind')}</th>
                  <th>{t('hs.runs.command')}</th>
                  <th>{t('hs.runs.result')}</th>
                  <th>{t('hs.runs.parts')}</th>
                  <th>{t('hs.runs.duration')}</th>
                  {tech && <th>{t('hs.runs.id')}</th>}
                </tr>
              </thead>
              <tbody>
                {listed.map((run) => {
                  const command = commandWords(t, run, inputs.get(run.id))
                  const duration = run.finished_at ? t.fmt.duration(run.finished_at - run.started_at) : '…'
                  return (
                    <tr key={run.id} title={tech && run.error ? run.error : undefined}>
                      <td>{t.fmt.time(run.started_at)}</td>
                      <td className="prose">{run.kind ? t.msg(runKindMsg(run.kind)) : '—'}</td>
                      <td className="prose hs-command">
                        <span>{command.text}</span>
                        {command.spoken && <span className="hs-badge">{t('hs.runs.spoken')}</span>}
                        {run.restart_of && tech && <span className="hs-sub">{t('hs.runs.restart', { run: run.restart_of })}</span>}
                      </td>
                      <td>
                        <ResultPill t={t} run={run} />
                      </td>
                      <td>{parts(run)}</td>
                      <td>{duration}</td>
                      {tech && <td className="hs-id">{run.id}</td>}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          {runs.length > RUNS_SHOWN && (
            <div className="actions hs-more">
              <button type="button" className="ghost" onClick={() => setAll(!all)}>
                {all ? t('hs.list.less', { n: RUNS_SHOWN }) : t('hs.list.more', { n: runs.length })}
              </button>
            </div>
          )}
        </>
      )}
      <Caveat>{t('hs.runs.caveat')}</Caveat>
    </Panel>
  )
}

function Rollup({ t, rollup, tech }: { t: ScreensT; rollup: RollupOut; tech: boolean }) {
  const kpis = (rollup.kpis ?? {}) as Record<string, unknown>
  const unmeasurable = (rollup.unmeasurable ?? {}) as Record<string, string>
  const outcomes = (rollup.outcomes ?? {}) as Record<string, number>
  // The file read is named once where the source IS that file (a console's own log), twice where they differ.
  const path = rollup.record_log_path && !samePath(rollup.record_log_path, rollup.source || '') ? rollup.record_log_path : ''

  return (
    <Panel
      title={t('hs.log.title', { n: rollup.total_attempts })}
      aside={
        rollup.record_log_exists ? (
          <StatusPill status="ok">{t('hs.log.present')}</StatusPill>
        ) : (
          <StatusPill status="warn">{t('hs.log.missing')}</StatusPill>
        )
      }
    >
      <div className="small faint provenance-line hs-source">
        {t('hs.log.source')} <span className="mono">{rollup.source || '—'}</span>
        {path && (
          <>
            {' · '}
            <span className="mono">{path}</span>
          </>
        )}
      </div>

      {Object.keys(kpis).length === 0 ? (
        <Empty>{t('hs.kpis.empty')}</Empty>
      ) : (
        <div className="hs-table">
          <table>
            <thead>
              <tr>
                <th>{t('hs.kpis.name')}</th>
                <th className="hs-num">{t('hs.kpis.value')}</th>
                <th className="hs-barcell" />
              </tr>
            </thead>
            <tbody>
              {Object.entries(kpis).map(([name, value]) => (
                <tr key={name}>
                  <td className="prose">
                    {kpiName(t, name)}
                    {tech && isKpiName(name) && <span className="hs-sub mono">{name}</span>}
                  </td>
                  <td className="hs-num">{kpiValue(t, name, value)}</td>
                  <td className="hs-barcell">
                    {typeof value === 'number' && value >= 0 && value <= 1 && name.endsWith('_rate') && (
                      <div className="bar">
                        <span style={{ width: `${Math.min(100, value * 100)}%` }} />
                      </div>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {Object.keys(unmeasurable).length > 0 && (
        <div className="stack">
          <h3>{t('hs.unmeasurable.title')}</h3>
          <dl className="hs-unmeasurable">
            {Object.entries(unmeasurable).map(([name, why]) => (
              <div key={name} className="hs-unmeasurable-item">
                <dt>
                  {kpiName(t, name)}
                  {tech && isKpiName(name) && <span className="hs-sub mono">{name}</span>}
                </dt>
                <dd>
                  {tech ? (
                    String(why)
                  ) : (
                    <details className="payload-toggle hs-why">
                      <summary>{t('hs.unmeasurable.why')}</summary>
                      <span>{String(why)}</span>
                    </details>
                  )}
                </dd>
              </div>
            ))}
          </dl>
          <Caveat>{t('hs.unmeasurable.caveat')}</Caveat>
        </div>
      )}

      {Object.keys(outcomes).length > 0 && (
        <div className="stack">
          <h3>{t('hs.outcomes')}</h3>
          <div className="actions">
            {Object.entries(outcomes).map(([name, n]) => (
              <StatusPill key={name} status={outcomeTone(name)}>
                {t.msg(outcomeMsg(name))} · {n}
              </StatusPill>
            ))}
          </div>
        </div>
      )}
    </Panel>
  )
}

function Records({ t, records }: { t: ScreensT; records: readonly RecordOut[] }) {
  const [all, setAll] = useState(false)
  const shown = all ? records : records.slice(0, RECORDS_SHOWN)
  return (
    <Panel title={t('hs.records.title')} aside={<a href="/v1/history/records.csv">records.csv</a>}>
      {records.length === 0 ? (
        <Empty>{t('hs.records.empty')}</Empty>
      ) : (
        <>
          <div className="hs-table">
            <table>
              <thead>
                <tr>
                  <th>{t('hs.records.when')}</th>
                  <th>{t('hs.records.attempt')}</th>
                  <th>{t('hs.records.mode')}</th>
                  <th>{t('hs.records.outcome')}</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((record) => (
                  <tr key={record.attempt_id}>
                    <td>{new Date(record.timestamp * 1000).toLocaleString(t.lang === 'de' ? 'de-DE' : 'en-GB')}</td>
                    <td>{record.attempt_id}</td>
                    <td>{record.mode}</td>
                    <td>
                      <StatusPill status={outcomeTone(record.final_outcome)}>{t.msg(outcomeMsg(record.final_outcome))}</StatusPill>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {records.length > RECORDS_SHOWN && (
            <div className="actions hs-more">
              <button type="button" className="ghost" onClick={() => setAll(!all)}>
                {all ? t('hs.list.less', { n: RECORDS_SHOWN }) : t('hs.list.more', { n: records.length })}
              </button>
            </div>
          )}
        </>
      )}
    </Panel>
  )
}

export default function History() {
  const t = useT(SCREENS)
  const { view } = usePrefs()
  const { cell } = useCell()
  const tech = view === 'tech'
  // The session's runs are read again whenever a run starts or ends.
  const active = cell?.active_run_id ?? ''
  const rollup = useAsync(() => api.kpis())
  const records = useAsync(() => api.records(100))
  const runs = useAsync(() => api.runs(RUNS_KEPT), [active])
  const runList = useMemo(() => runs.data ?? [], [runs.data])
  const stats = useTaskStats(runList)

  return (
    <div className="page hs-page">
      <ScreenHead title={t('nav.history')} lede={t('hs.lede')} />

      {runs.error ? (
        <ErrorBanner error={runs.error} onRetry={runs.reload} />
      ) : runs.loading && !runs.data ? (
        <Loading what={t('hs.what.runs')} />
      ) : (
        <>
          <SessionTiles t={t} runs={runList} stats={stats} />
          <Charts t={t} runs={runList} stats={stats} />
          <RunsTable t={t} runs={runList} tech={tech} />
        </>
      )}

      {rollup.error ? (
        <ErrorBanner error={rollup.error} onRetry={rollup.reload} />
      ) : rollup.loading && !rollup.data ? (
        <Loading what={t('hs.what.kpis')} />
      ) : (
        rollup.data && <Rollup t={t} rollup={rollup.data} tech={tech} />
      )}

      {records.error ? (
        <ErrorBanner error={records.error} onRetry={records.reload} />
      ) : records.loading && !records.data ? (
        <Loading what={t('hs.what.records')} />
      ) : (
        <Records t={t} records={records.data ?? []} />
      )}
    </div>
  )
}
