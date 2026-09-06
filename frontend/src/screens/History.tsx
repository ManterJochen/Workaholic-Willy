/**
 * What this cell has actually done, from the logged records rather than from memory.
 *
 * The single most important thing this screen does is state WHERE the numbers came from. The backend
 * returns `source` and `record_log_path` with every rollup, and a KPI panel that omits them is the
 * fastest way to present a synthetic self-check as a production result — which is exactly the
 * confusion this project's soak gate is documented to avoid. So the provenance line is not a footnote
 * here; it sits above the numbers.
 *
 * `unmeasurable` is rendered as prominently as the KPIs themselves. A metric the records cannot
 * support must read as "not measured", never as a blank or a zero.
 */

import { api, type RollupOut } from '../api/client'
import { Caveat, Empty, ErrorBanner, Loading, Panel, StatusPill } from '../components/ui'
import { outcomeLabel, outcomeTone, runTone } from '../lib/outcome'
import { useAsync } from '../lib/useAsync'

export default function History() {
  const rollup = useAsync(() => api.kpis())
  const records = useAsync(() => api.records(100))
  const runs = useAsync(() => api.historyRuns())

  return (
    <>
      <h2>History</h2>
      <p className="lede">
        Rolled up from the structured attempt log this console writes. Every number below is traceable
        to a line in that file.
      </p>

      {rollup.error ? (
        <ErrorBanner error={rollup.error} onRetry={rollup.reload} />
      ) : rollup.loading && !rollup.data ? (
        <Loading what="the KPIs" />
      ) : (
        rollup.data && <Kpis rollup={rollup.data} />
      )}

      <Panel
        title="This session's runs"
        aside={<a href="/v1/history/runs.csv">runs.csv</a>}
      >
        {runs.error ? (
          <ErrorBanner error={runs.error} onRetry={runs.reload} />
        ) : runs.loading && !runs.data ? (
          <Loading what="the runs" />
        ) : !runs.data || runs.data.length === 0 ? (
          <Empty>No runs yet in this process.</Empty>
        ) : (
          <table>
            <thead>
              <tr>
                <th>run</th>
                <th>prompt</th>
                <th>state</th>
                <th>picks</th>
                <th>outcomes</th>
              </tr>
            </thead>
            <tbody>
              {runs.data.map((r) => (
                <tr key={r.id}>
                  <td>{r.id}</td>
                  <td className="prose">{r.prompt || '—'}</td>
                  <td>
                    <StatusPill status={runTone(r.state)}>{r.state}</StatusPill>
                  </td>
                  <td>
                    {r.succeeded ?? 0}/{r.attempted ?? 0}
                  </td>
                  <td className="small">{(r.outcomes ?? []).join(', ') || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <Caveat>
          In-memory, for this process only. The attempt records below outlive it; these do not.
        </Caveat>
      </Panel>

      <Panel
        title="Logged grasp attempts"
        aside={<a href="/v1/history/records.csv">records.csv</a>}
      >
        {records.error ? (
          <ErrorBanner error={records.error} onRetry={records.reload} />
        ) : records.loading && !records.data ? (
          <Loading what="the records" />
        ) : !records.data || records.data.length === 0 ? (
          <Empty>No attempts logged yet.</Empty>
        ) : (
          <table>
            <thead>
              <tr>
                <th>when</th>
                <th>attempt</th>
                <th>mode</th>
                <th>outcome</th>
              </tr>
            </thead>
            <tbody>
              {records.data.map((rec) => (
                <tr key={rec.attempt_id}>
                  <td>{new Date(rec.timestamp * 1000).toLocaleString()}</td>
                  <td>{rec.attempt_id}</td>
                  <td>{rec.mode}</td>
                  <td>
                    <StatusPill status={outcomeTone(rec.final_outcome)}>
                      {outcomeLabel(rec.final_outcome)}
                    </StatusPill>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>
    </>
  )
}

function Kpis({ rollup }: { rollup: RollupOut }) {
  const kpis = (rollup.kpis ?? {}) as Record<string, number>
  const unmeasurable = (rollup.unmeasurable ?? {}) as Record<string, string>
  const outcomes = (rollup.outcomes ?? {}) as Record<string, number>

  return (
    <Panel
      title={`${rollup.total_attempts} logged attempt${rollup.total_attempts === 1 ? '' : 's'}`}
      aside={
        rollup.record_log_exists ? (
          <StatusPill status="ok">log present</StatusPill>
        ) : (
          <StatusPill status="warn">no log file</StatusPill>
        )
      }
    >
      <div className="small faint provenance-line">
        source <span className="mono">{rollup.source ?? 'unknown'}</span>
        {rollup.record_log_path && (
          <>
            {' · '}
            <span className="mono">{rollup.record_log_path}</span>
          </>
        )}
      </div>

      {Object.keys(kpis).length === 0 ? (
        <Empty>Nothing measurable yet.</Empty>
      ) : (
        <table>
          <thead>
            <tr>
              <th>KPI</th>
              <th style={{ width: 110 }}>value</th>
              <th style={{ width: '40%' }} />
            </tr>
          </thead>
          <tbody>
            {Object.entries(kpis).map(([name, value]) => (
              <tr key={name}>
                <td className="prose">{name}</td>
                <td>{typeof value === 'number' ? value.toFixed(3) : String(value)}</td>
                <td>
                  {typeof value === 'number' && value >= 0 && value <= 1 && (
                    <div className="bar">
                      <span style={{ width: `${Math.min(100, value * 100)}%` }} />
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {Object.keys(unmeasurable).length > 0 && (
        <div className="stack">
          <h3>Not measurable from these records</h3>
          {Object.entries(unmeasurable).map(([name, why]) => (
            <div className="row" key={name}>
              <div>
                <StatusPill status="bench">{name}</StatusPill>
              </div>
              <div className="detail">{String(why)}</div>
            </div>
          ))}
          <Caveat>
            Listed rather than hidden: a KPI missing from a dashboard reads as “fine”, and a KPI shown
            as zero reads as “bad”. Both are wrong when the truth is “the records cannot say”.
          </Caveat>
        </div>
      )}

      {Object.keys(outcomes).length > 0 && (
        <div className="stack">
          <h3>Outcomes</h3>
          <div className="actions">
            {Object.entries(outcomes).map(([name, n]) => (
              <StatusPill key={name} status={outcomeTone(name)}>
                {outcomeLabel(name)} · {n}
              </StatusPill>
            ))}
          </div>
        </div>
      )}
    </Panel>
  )
}
