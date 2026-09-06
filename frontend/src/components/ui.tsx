/**
 * The small pieces every screen shares.
 *
 * They live together because they encode the console's honesty rules, and a rule that is re-decided
 * per screen is a rule that will be decided differently on one of them:
 *
 * * `StatusPill` is the ONLY thing that maps a backend status to a colour;
 * * `ErrorBanner` always shows the backend's `code` next to its message, so an operator can quote it;
 * * `Loading` never renders stale data underneath itself without saying it is stale.
 */

import type { ReactNode } from 'react'

import type { ApiError, Status } from '../api/client'

export function StatusPill({ status, children }: { status: Status | string; children?: ReactNode }) {
  const known = ['ok', 'warn', 'block', 'bench', 'info', 'idle'].includes(status)
  // An unknown status is shown as itself rather than styled as OK. The backend pins these as a
  // literal type precisely so a new one arrives loudly instead of as an unstyled green row.
  return <span className={`pill ${known ? status : 'info'}`}>{children ?? status}</span>
}

export function Panel({
  title,
  aside,
  children,
}: {
  title?: ReactNode
  aside?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="panel">
      {(title || aside) && (
        <div className="panel-head">
          {title && <h3>{title}</h3>}
          {aside && <div className="small dim">{aside}</div>}
        </div>
      )}
      {children}
    </section>
  )
}

export function ErrorBanner({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  const hasDetail = Object.keys(error.detail).length > 0
  return (
    <div className="banner error">
      <div className="body">
        <div>{error.message}</div>
        <div className="code">
          {error.code}
          {error.httpStatus ? ` · HTTP ${error.httpStatus}` : ''}
        </div>
        {hasDetail && (
          <details className="payload-toggle" style={{ marginTop: 6 }}>
            <summary>detail</summary>
            <pre className="payload">{JSON.stringify(error.detail, null, 2)}</pre>
          </details>
        )}
      </div>
      {onRetry && (
        <button className="ghost" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  )
}

export function Loading({ what }: { what: string }) {
  return <div className="spinner">Reading {what}…</div>
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>
}

/**
 * A short sentence that says how sure the console is about what it just drew.
 *
 * The project keeps three honesty buckets -- measured in simulation, proven against a real protocol,
 * never touched real hardware -- and a UI that renders all three identically is the fastest way to
 * turn a sim result into a claim about a factory. Screens use this wherever the answer depends on
 * which bucket the cell is in.
 */
export function Caveat({ children }: { children: ReactNode }) {
  return <div className="caveat">{children}</div>
}

export function KeyValues({ pairs }: { pairs: Array<[string, ReactNode]> }) {
  return (
    <dl className="kv">
      {pairs.map(([k, v]) => (
        <div key={k} style={{ display: 'contents' }}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  )
}

/** Millimetres, always three of them, always the same width. */
export function Vec3({ v, unit = 'mm' }: { v: number[] | null | undefined; unit?: string }) {
  if (!v || v.length < 3) return <span className="faint">—</span>
  return (
    <span className="mono">
      {v.slice(0, 3).map((n) => n.toFixed(1).padStart(8)).join('  ')} {unit}
    </span>
  )
}
