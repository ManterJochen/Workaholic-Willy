/**
 * "Is this cell runnable?" -- the first screen, and the one that must never be skippable.
 *
 * The backend already decided every verdict here (`run_config_preflight`). This screen adds no
 * judgement of its own: it does not re-rank rows, does not hide `ok` ones by default, and above all
 * does not summarise a blocking item into a friendlier word. The `fix` string is rendered verbatim,
 * per row, because a checklist that says "wrong" without saying "do this" has only moved the guessing.
 *
 * The profile chain is in the header for a measured reason: the SAME tree blocks on three items as a
 * UR cell and none as a sim cell, so a verdict without its chain is unreadable.
 *
 * The four tiles are a READOUT, not a filter. Every count the backend returned is on screen at once,
 * including the zeros -- a tile that vanishes when it is empty forces an operator to count the
 * remaining tiles to work out which category is missing, which is the opposite of glanceable.
 */

import { useState } from 'react'

import { api } from '../api/client'
import { Caveat, ErrorBanner, Loading, Panel, StatusPill } from '../components/ui'
import { useAsync } from '../lib/useAsync'

const ORDER = { block: 0, warn: 1, bench: 2, ok: 3 } as const

function Tile({ n, label, tone }: { n: number; label: string; tone: string }) {
  return (
    <div className={`tile ${tone}${n === 0 ? ' zero' : ''}`}>
      <div className="n">{n}</div>
      <div className="t">{label}</div>
    </div>
  )
}

export default function Preflight() {
  const [showOk, setShowOk] = useState(true)
  const { data, error, loading, reload } = useAsync(() => api.preflight())

  if (error) return <ErrorBanner error={error} onRetry={reload} />
  if (loading && !data) return <Loading what="the checklist" />
  if (!data) return null

  const rows = [...data.checks].sort(
    (a, b) => (ORDER[a.status] ?? 9) - (ORDER[b.status] ?? 9),
  )
  const visible = showOk ? rows : rows.filter((r) => r.status !== 'ok')
  const nPassing = rows.filter((r) => r.status === 'ok').length

  return (
    <>
      <h2>Preflight</h2>
      <p className="lede">
        Everything the cell checks before it will accept a connect. Read top to bottom: the list is
        sorted by severity, and nothing here has been softened for presentation.
      </p>

      <div className={`banner ${data.ok ? 'ok' : 'error'}`}>
        <div className="body">
          <div className="verdict-line">
            <StatusPill status={data.ok ? 'ok' : 'block'} />
            <strong>
              {data.ok
                ? 'This cell is runnable.'
                : `${data.n_blocking} blocking item${data.n_blocking === 1 ? '' : 's'}: this cell is not runnable as configured.`}
            </strong>
          </div>
          <div className="small dim">
            vendor <span className="mono">{data.vendor}</span> · profile{' '}
            <span className="mono">{data.profile || '(none)'}</span>
          </div>
        </div>
        <button className="ghost" onClick={reload}>
          Re-check
        </button>
      </div>

      <div className="tiles">
        <Tile n={data.n_blocking} label="blocking" tone="block" />
        <Tile n={data.n_warn} label="warnings" tone="warn" />
        <Tile n={data.n_bench} label="bench" tone="bench" />
        <Tile n={nPassing} label="passing" tone="ok" />
      </div>

      <Panel
        title={`${visible.length} of ${rows.length} checks`}
        aside={
          <label className="check">
            <input type="checkbox" checked={showOk} onChange={(e) => setShowOk(e.target.checked)} />
            show passing
          </label>
        }
      >
        {visible.length === 0 ? (
          <div className="empty">Nothing but passing checks.</div>
        ) : (
          visible.map((check) => (
            <div className="row" key={check.name}>
              <div>
                <StatusPill status={check.status} />
              </div>
              <div>
                <div className="name">{check.name}</div>
                <div className="detail">{check.detail}</div>
                {check.fix && <div className="fix">{check.fix}</div>}
              </div>
            </div>
          ))
        )}
        <Caveat>
          A <span className="mono">bench</span> row is not a failure: it is something this box cannot
          decide, and that only a person standing at the cell can answer.
          <br />
          A <span className="mono">block</span> row does not necessarily stop you CONNECTING — measured
          2026-08-20 against a real UR controller, a cell with a blocking checklist connected fine and
          then refused every motion, which is the failure that reads as a broken robot. What refuses
          the connect itself is the list on the Cell screen's connect preview.
        </Caveat>
      </Panel>
    </>
  )
}
