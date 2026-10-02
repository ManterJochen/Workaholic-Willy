/**
 * "Is this cell runnable?" -- the first step of Setup, and the one that must never be skippable.
 *
 * The backend already decided every verdict here (`run_config_preflight`). This view adds no judgement of its own:
 * it does not re-rank rows, does not hide `ok` ones by default, and above all does not summarise a blocking item into
 * a friendlier word. The `fix` string is rendered verbatim, per row, because a checklist that says "wrong" without
 * saying "do this" has only moved the guessing.
 *
 * **Whose words.** A check is named in the reader's language (`pf.check.*`; a check this console does not know yet
 * keeps the server's name), and its status is a pill in that language too. Its detail and fix are the server's own
 * English sentences (OD adopted: details of the tech view): open under the name in the tech view, where the server's
 * own name of the check stands beside it; folded behind "Details (Server)" in the demo view, one click away, never
 * dropped. That is how `ErrorBanner` folds a refusal's own sentence.
 *
 * The profile chain is in the header for a measured reason: the SAME tree blocks on three items as a UR cell and none
 * as a sim cell, so a verdict without its chain is unreadable.
 *
 * The four tiles are a READOUT, not a filter. Every count the backend returned is on screen at once, including the
 * zeros -- a tile that vanishes when it is empty forces an operator to count the remaining tiles to work out which
 * category is missing, which is the opposite of glanceable.
 *
 * `PreflightView` draws a checklist a caller already read (the Setup stepper reads it once for its own step mark);
 * the default export reads it itself.
 */

import { useState } from 'react'

import { api, type ApiError, type PreflightCheckOut, type PreflightOut } from '../api/client'
import { Caveat, ErrorBanner, Loading, Panel, StatusPill } from '../components/ui'
import { useT } from '../i18n'
import { useAsync } from '../lib/useAsync'
import { usePrefs } from '../model/prefs'
import { SCREENS, type ScreensT } from './i18n'

const ORDER = { block: 0, warn: 1, bench: 2, ok: 3 } as const

/** The server's check names (`src/robot/execution/real_cell/preflight.py`), each with its words in the catalog. */
const CHECK_WORDS = {
  vendor: 'pf.check.vendor',
  'tool frame': 'pf.check.toolFrame',
  payload: 'pf.check.payload',
  'camera -> base': 'pf.check.cameraBase',
  'camera world': 'pf.check.cameraWorld',
  'carried part': 'pf.check.carriedPart',
  'controller state': 'pf.check.controllerState',
  'cuRobo environment': 'pf.check.curobo',
  'end-effector wiring': 'pf.check.endEffectorWiring',
  'exact mesh engine': 'pf.check.exactMesh',
  fixtures: 'pf.check.fixtures',
  'grasp centre': 'pf.check.graspCentre',
  'gripper driver': 'pf.check.gripperDriver',
  hand: 'pf.check.hand',
  'jaw travel time': 'pf.check.jawTravel',
  'joint window': 'pf.check.jointWindow',
  looks: 'pf.check.looks',
  'planner margin': 'pf.check.plannerMargin',
  'planner reservation': 'pf.check.plannerReservation',
  'planning world': 'pf.check.planningWorld',
  'record log': 'pf.check.recordLog',
  'self-collision': 'pf.check.selfCollision',
  'wrist camera body': 'pf.check.wristCamera',
} as const

/** A check's name in the reader's words; one this console does not know yet keeps the server's. */
function checkName(t: ScreensT, name: string): string {
  const key = (CHECK_WORDS as Record<string, (typeof CHECK_WORDS)[keyof typeof CHECK_WORDS]>)[name]
  return key ? t(key) : name
}

/** One check: its status and name, then the server's detail and fix (open in the tech view, folded in the demo view). */
function CheckRow({ t, check, tech }: { t: ScreensT; check: PreflightCheckOut; tech: boolean }) {
  const name = checkName(t, check.name)
  const said = (
    <>
      <div className="detail">{check.detail}</div>
      {check.fix && <div className="fix">{check.fix}</div>}
    </>
  )
  return (
    <div className="row">
      <div>
        <StatusPill status={check.status}>{t(`pf.status.${check.status}`)}</StatusPill>
      </div>
      <div>
        <div className="name">{name}</div>
        {tech && name !== check.name && <div className="pf-server-name">{check.name}</div>}
        {tech ? (
          said
        ) : (
          <details className="payload-toggle pf-said">
            <summary>{t('pf.details')}</summary>
            {said}
          </details>
        )}
      </div>
    </div>
  )
}

function Tile({ n, label, tone }: { n: number; label: string; tone: string }) {
  return (
    <div className={`tile ${tone}${n === 0 ? ' zero' : ''}`}>
      <div className="n">{n}</div>
      <div className="t">{label}</div>
    </div>
  )
}

export interface PreflightViewProps {
  data: PreflightOut | null
  error: ApiError | null
  loading: boolean
  reload: () => void
}

export function PreflightView({ data, error, loading, reload }: PreflightViewProps) {
  const t = useT(SCREENS)
  const { view } = usePrefs()
  const [showOk, setShowOk] = useState(true)

  if (error) return <ErrorBanner error={error} onRetry={reload} />
  if (loading && !data) return <Loading what={t('pf.what')} />
  if (!data) return null

  const rows = [...data.checks].sort((a, b) => (ORDER[a.status] ?? 9) - (ORDER[b.status] ?? 9))
  const visible = showOk ? rows : rows.filter((r) => r.status !== 'ok')
  const nPassing = rows.filter((r) => r.status === 'ok').length

  return (
    <>
      <div className={`banner ${data.ok ? 'ok' : 'error'}`}>
        <div className="body">
          <div className="verdict-line">
            <StatusPill status={data.ok ? 'ok' : 'block'}>{t(data.ok ? 'pf.status.ok' : 'pf.status.block')}</StatusPill>
            <strong>{data.ok ? t('pf.verdict.ok') : t('pf.verdict.block', { n: data.n_blocking })}</strong>
          </div>
          <div className="small dim">
            {t('pf.vendor')} <span className="mono">{data.vendor}</span> · {t('pf.profile')}{' '}
            <span className="mono">{data.profile || t('cl.noProfile')}</span>
          </div>
        </div>
        <button className="ghost" onClick={reload}>
          {t('pf.recheck')}
        </button>
      </div>

      <div className="tiles">
        <Tile n={data.n_blocking} label={t('pf.tile.block')} tone="block" />
        <Tile n={data.n_warn} label={t('pf.tile.warn')} tone="warn" />
        <Tile n={data.n_bench} label={t('pf.tile.bench')} tone="bench" />
        <Tile n={nPassing} label={t('pf.tile.ok')} tone="ok" />
      </div>

      <Panel
        title={t('pf.checks', { shown: visible.length, total: rows.length })}
        aside={
          <label className="check">
            <input type="checkbox" checked={showOk} onChange={(e) => setShowOk(e.target.checked)} />
            {t('pf.showPassing')}
          </label>
        }
      >
        {visible.length === 0 ? (
          <div className="empty">{t('pf.empty')}</div>
        ) : (
          visible.map((check, index) => (
            <CheckRow key={`${check.name}-${index}`} t={t} check={check} tech={view === 'tech'} />
          ))
        )}
        <Caveat>
          <div>{t('pf.caveat.bench')}</div>
          <div>{t('pf.caveat.block')}</div>
        </Caveat>
      </Panel>
    </>
  )
}

export default function Preflight() {
  const { data, error, loading, reload } = useAsync(() => api.preflight())
  return <PreflightView data={data} error={error} loading={loading} reload={reload} />
}
