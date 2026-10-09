/**
 * The small pieces every screen shares.
 *
 * They live together because they encode the console's honesty rules, and a rule that is re-decided
 * per screen is a rule that will be decided differently on one of them:
 *
 * * `StatusPill` and `Chip` are the ONLY things that map a backend status to a colour;
 * * `ErrorBanner` says a refusal in the reader's language and always carries its `code`, so an operator can quote it:
 *   the code, the HTTP status and the backend's own English sentence sit under the words in the tech view, and folded
 *   behind "Details" in the demo view, one click away (a generic refusal's only reason is often that sentence; the
 *   demo view shows no raw code to an audience); a code this console does not know reads "Abgelehnt (<code>)";
 * * `Loading` never renders stale data underneath itself without saying it is stale.
 *
 * Their words come from the shared catalog (`useT()`), so under no provider they answer in English, which is
 * what the existing screen tests render.
 */

import type { ReactNode } from 'react'

import type { ApiError, Status } from '../api/client'
import { useT } from '../i18n'
import { refusalMsg } from '../i18n/codes'
import { usePrefs } from '../model/prefs'

export function StatusPill({ status, children }: { status: Status | string; children?: ReactNode }) {
  const known = ['ok', 'warn', 'block', 'bench', 'info', 'idle'].includes(status)
  // An unknown status is shown as itself rather than styled as OK. The backend pins these as a
  // literal type precisely so a new one arrives loudly instead of as an unstyled green row.
  return <span className={`pill ${known ? status : 'info'}`}>{children ?? status}</span>
}

/**
 * A state chip of the top bar (and of anything that says a state in one line): a small label, the value, and a dot
 * in the tone's colour. The words stay in the foreground colour; the tint only groups them.
 */
export function Chip({
  label,
  tone = 'idle',
  live = false,
  title,
  children,
}: {
  label: ReactNode
  tone?: 'ok' | 'warn' | 'block' | 'info' | 'bench' | 'idle'
  live?: boolean
  title?: string
  children: ReactNode
}) {
  return (
    <span className={`chip ${tone}${live ? ' live' : ''}`} title={title}>
      <span className="k">{label}</span>
      <span className="v">{children}</span>
    </span>
  )
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

/** A screen's title and its one-line purpose, with the accent rail. */
export function ScreenHead({ title, lede, children }: { title: ReactNode; lede?: ReactNode; children?: ReactNode }) {
  return (
    <header className="screen-head">
      <h2>{title}</h2>
      {lede && <p className="lede">{lede}</p>}
      {children}
    </header>
  )
}

/** A switch whose options are all visible, the chosen one lit: Demo | Technik, DE | EN. */
export function Segmented<V extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: V
  options: ReadonlyArray<{ value: V; label: ReactNode; title?: string }>
  onChange: (value: V) => void
}) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          aria-pressed={option.value === value}
          title={option.title}
          onClick={() => {
            if (option.value !== value) onChange(option.value)
          }}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}

export function ErrorBanner({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  const t = useT()
  const { view } = usePrefs()
  const hasDetail = Object.keys(error.detail).length > 0
  // No answer at all is its own situation (the server is not running), not a refusal with a code.
  const noAnswer = error.httpStatus === 0
  const said = noAnswer ? t('common.noAnswer', { origin: window.location.origin }) : t.msg(refusalMsg(error.code, error.detail))
  // The server's own English sentence and the code are details (OD adopted; OD 1, little jargon in the demo view):
  // open under the words in the tech view, folded behind "Details" in the demo view, where a generic refusal ("Der
  // Aufbau wurde abgelehnt.") would otherwise lose its only reason. The code is quotable in both, one click away there.
  const tech = view === 'tech'
  const backend = !noAnswer && error.message && error.message !== said ? error.message : ''
  const code = (
    <div className="code">
      {error.code}
      {error.httpStatus ? ` · ${t('common.http', { status: error.httpStatus })}` : ''}
    </div>
  )
  // No answer at all has no code worth a fold: the words say it all ("no answer from the server").
  const foldCode = !tech && !noAnswer
  return (
    <div className="banner error" role="alert">
      <div className="body">
        <div>{said}</div>
        {tech && backend && <div className="said">{backend}</div>}
        {!foldCode && code}
        {((!tech && backend) || hasDetail || foldCode) && (
          <details className="payload-toggle" style={{ marginTop: 6 }}>
            <summary>{t('common.detail')}</summary>
            {!tech && backend && <div className="said">{backend}</div>}
            {foldCode && code}
            {hasDetail && <pre className="payload">{JSON.stringify(error.detail, null, 2)}</pre>}
          </details>
        )}
      </div>
      {onRetry && (
        <button className="ghost" onClick={onRetry}>
          {t('common.retry')}
        </button>
      )}
    </div>
  )
}

export function Loading({ what }: { what: string }) {
  const t = useT()
  return <div className="spinner">{t('common.loading', { what })}</div>
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
