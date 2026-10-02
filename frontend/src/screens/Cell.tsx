/**
 * Build -> preview -> connect -> live telemetry -> disconnect: the cell's own panels.
 *
 * The connect flow is the one place in this console where the UI is deliberately in the way. The backend hands out a
 * token from `/v1/cell/connect-preview` and refuses `/v1/cell/connect` without it; there is no force flag on either
 * side. These panels therefore CANNOT offer a one-click connect, and do not try to: the preview's motion warnings are
 * drawn first, and the button that follows them is the only red one, and it says that the robot moves.
 *
 * Building touches no robot, so its button is not red; it builds for real by default where the arm is real (a
 * rehearsal is chosen, never assumed: build plan 4.3). A latched arm (a halt, or a disconnect during a move) is
 * cleared only by a person's word, after they ticked that they looked; a protective stop is cleared at the pendant,
 * never here.
 *
 * Two backend fields drive most of what is drawn, and both exist because a console that omits them is misleading:
 *
 * * `gripper_substitution` -- a real end-effector could not be built. Connect is refused while it is set, because the
 *   cell would come up, every pick would report success, and the gripper would close on nothing.
 * * `TelemetryOut.simulated` -- a simulated arm produces numbers that look exactly like a real one's. This is the only
 *   field that says which produced the ones on screen, so it is always visible.
 *
 * The Setup stepper shows one panel per step and holds the actions (`useCellActions`), so the preview read in one
 * step is the token the next one sends. The default export stacks every panel over its own read of the cell.
 */

import { useCallback, useEffect, useState } from 'react'

import { api, type CellOut, type TelemetryOut } from '../api/client'
import { ProvenanceBadge, ProvenanceDetail } from '../components/ProvenanceBadge'
import { Caveat, ErrorBanner, KeyValues, Loading, Panel, StatusPill, Vec3 } from '../components/ui'
import { useT } from '../i18n'
import { useAsync, usePoll } from '../lib/useAsync'
import { usePrefs } from '../model/prefs'
import { rehearseByDefault, useCellActions, type CellActions } from './cellActions'
import './screens.css'
import { SCREENS, type ScreensT } from './i18n'

/** A cell that may not be built again or connected now: one moves, one asks, or one is already up. */
function settled(cell: CellOut): boolean {
  return cell.state === 'connected' || cell.state === 'connecting'
}

const STATES = ['disconnected', 'built', 'connecting', 'connected'] as const

/** A cell state in words; one this console does not know is shown as the server said it. */
function cellStateWords(t: ScreensT, state: string): string {
  const known = STATES.find((s) => s === state)
  return known ? t(`cellState.${known}`) : state
}

export function StatePanel({ cell }: { cell: CellOut }) {
  const t = useT(SCREENS)
  const connected = cell.state === 'connected'
  const state = cellStateWords(t, cell.state)
  return (
    <Panel
      title={t('cl.state.title')}
      aside={
        <>
          {t('cl.profile')} <span className="mono">{cell.profile || t('cl.noProfile')}</span>
        </>
      }
    >
      <KeyValues
        pairs={[
          [t('cl.k.state'), <StatusPill key="s" status={connected ? 'ok' : cell.state === 'built' ? 'info' : 'idle'}>{state}</StatusPill>],
          [t('cl.k.vendor'), cell.vendor],
          [t('cl.k.arm'), cell.arm ?? <span key="a" className="faint">{t('cl.notBuilt')}</span>],
          [t('cl.k.gripper'), cell.gripper ?? <span key="g" className="faint">{t('cl.notBuilt')}</span>],
          [t('cl.k.lock'), cell.lock_holder ?? <span key="l" className="faint">{t('cl.nobody')}</span>],
          [t('cl.k.run'), cell.active_run_id ?? <span key="r" className="faint">{t('cl.noRun')}</span>],
        ]}
      />
    </Panel>
  )
}

/** A real end-effector could not be built: said as loudly as anything on the page, and connect is refused. */
export function SubstitutionBanner({ cell }: { cell: CellOut }) {
  const t = useT(SCREENS)
  const substitution = cell.gripper_substitution as Record<string, unknown> | null | undefined
  if (!substitution) return null
  return (
    <div className="banner error">
      <div className="body">
        <strong>{t('cl.substitution.title')}</strong>
        <div className="small dim">{t('cl.substitution.why')}</div>
        <pre className="payload">{JSON.stringify(substitution, null, 2)}</pre>
      </div>
    </div>
  )
}

export function BuildPanel({
  cell,
  actions,
  rehearse,
  onRehearse,
}: {
  cell: CellOut
  actions: CellActions
  rehearse: boolean
  onRehearse: (rehearse: boolean) => void
}) {
  const t = useT(SCREENS)
  const busy = actions.busy !== null
  const up = settled(cell)
  const blocked = busy || up || Boolean(cell.active_run_id) || cell.jaws_question === true
  return (
    <Panel title={t('cl.build.title')}>
      <p className="panel-lede">{t('cl.build.lede')}</p>
      <div className="actions">
        <button
          type="button"
          className="primary big"
          disabled={blocked}
          onClick={() => void actions.build(rehearse)}
        >
          {actions.busy === 'build' ? t('cl.build.busy') : cell.arm ? t('cl.build.again') : t('cl.build.go')}
        </button>
        <label className="check">
          <input type="checkbox" checked={rehearse} disabled={up || busy} onChange={(e) => onRehearse(e.target.checked)} />
          {t('cl.build.rehearse')}
        </label>
      </div>
      {up && <p className="small faint">{t('cl.build.disconnectFirst')}</p>}
      <Caveat>{t('cl.build.caveat')}</Caveat>
    </Panel>
  )
}

/** The token's expiry as the reader's wall clock (`19:46:17`), never the server's UTC stamp; a stamp that does not
 * parse is shown as it came. */
function expiryWords(t: ScreensT, expiresAt: string): string {
  const at = Date.parse(expiresAt)
  return Number.isFinite(at) ? t.fmt.time(at / 1000) : expiresAt
}

/** What a preview warning is about, in the reader's words; a subject this console does not know is shown as sent. */
function subjectWords(t: ScreensT, subject: string): string {
  if (subject === 'gripper') return t('cl.preview.k.gripper')
  if (subject === 'arm') return t('cl.preview.k.arm')
  return subject
}

export function PreviewPanel({ cell, actions }: { cell: CellOut; actions: CellActions }) {
  const t = useT(SCREENS)
  const { view } = usePrefs()
  const preview = actions.preview
  const blocking = preview?.blocking ?? []
  const warnings = preview?.warnings ?? []
  const can = Boolean(cell.arm) && !settled(cell) && actions.busy === null
  // The driver classes are the tech view's; everyone reads when the token runs out, in their own clock.
  const facts: Array<[string, string]> = preview
    ? [
        ...(view === 'tech'
          ? ([
              [t('cl.preview.k.arm'), preview.arm],
              [t('cl.preview.k.gripper'), preview.gripper],
            ] as Array<[string, string]>)
          : []),
        [t('cl.preview.k.expires'), expiryWords(t, preview.expires_at)],
      ]
    : []
  return (
    <Panel title={t('cl.preview.title')}>
      <div className="actions">
        <button type="button" className="big" disabled={!can} onClick={() => void actions.readPreview()}>
          {actions.busy === 'preview' ? t('cl.preview.busy') : preview ? t('cl.preview.again') : t('cl.preview.go')}
        </button>
        {!cell.arm && <span className="small faint">{t('cl.preview.buildFirst')}</span>}
      </div>

      {preview && (
        <div className="stack">
          <KeyValues pairs={facts} />
          {blocking.length > 0 && (
            <div className="banner error stack-item">
              <div className="body">
                <strong>{t('cl.preview.blocking')}</strong>
                <ul className="reasons">
                  {blocking.map((b) => (
                    <li key={b}>{b}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
          {warnings.length > 0 ? (
            <div className="stack-item">
              <h3>{t('cl.preview.moves')}</h3>
              {warnings.map((w) => (
                <div className="row" key={`${w.subject}-${w.what}`}>
                  <div>
                    <StatusPill status="warn">{subjectWords(t, w.subject)}</StatusPill>
                  </div>
                  <div>
                    <div className="detail">{w.what}</div>
                    <div className="fix">{w.precaution}</div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            blocking.length === 0 && <p className="small faint stack-item">{t('cl.preview.nothing')}</p>
          )}
        </div>
      )}
    </Panel>
  )
}

export function ConnectPanel({ cell, actions }: { cell: CellOut; actions: CellActions }) {
  const t = useT(SCREENS)
  const preview = actions.preview
  const connected = cell.state === 'connected'
  const blocked = (preview?.blocking ?? []).length > 0
  const busy = actions.busy !== null
  const toggle = cell.hand?.kind === 'toggle'
  return (
    <Panel title={t('cl.connect.title')}>
      <p className="panel-lede">{t('cl.connect.lede')}</p>
      {toggle && !connected && <p className="setup-note">{t('cl.connect.jaws')}</p>}
      <div className="actions">
        <button
          type="button"
          className="danger big"
          disabled={busy || !preview || blocked || connected || cell.state === 'connecting' || Boolean(cell.halted)}
          onClick={() => void actions.connect()}
        >
          {actions.busy === 'connect' || cell.state === 'connecting' ? t('cl.connect.busy') : t('cl.connect.go')}
        </button>
        <button
          type="button"
          className="ghost big"
          disabled={busy || !(connected || cell.state === 'connecting')}
          onClick={() => void actions.disconnect()}
        >
          {actions.busy === 'disconnect' ? t('cl.disconnect.busy') : t('cl.disconnect')}
        </button>
      </div>
      {!preview && !connected && cell.state !== 'connecting' && (
        <p className="small faint">{t('cl.connect.previewFirst')}</p>
      )}
      {blocked && <p className="small caution">{t('cl.connect.blocked')}</p>}
      {connected && <p className="small">{t('cl.connect.connected')}</p>}
    </Panel>
  )
}

/**
 * The arm is latched (`CellOut.halted`): a halt, or a disconnect during a move. Nothing moves and Connect is refused
 * until a person confirms the cell is clear; this is where they do, after ticking that they looked. It moves nothing.
 *
 * A stop nobody has said the cell is clear of (`CellOut.recovery`, uncleared) is offered the same word here once the
 * cell is built, before Verbinden: the jaws wait for it too, so a toggle's "Jetzt öffnen" at Connect is refused until
 * it is said, and after a restart of the server the record comes back uncleared. The cockpit's stop card offers the
 * same word; Restart and Home stay there.
 */
export function LatchPanel({ cell, actions }: { cell: CellOut; actions: CellActions }) {
  const t = useT(SCREENS)
  const { view } = usePrefs()
  const [looked, setLooked] = useState(false)
  const record = cell.recovery
  const stopStands =
    record != null &&
    !(record.cleared_at != null && record.cleared_at > record.at) &&
    (cell.state === 'built' || cell.state === 'connected')
  if (!cell.halted && !stopStands) return null
  const title = cell.halted ? t('cl.latch.title') : t('cl.stop.title')
  const reason = cell.halted ? cell.halted.reason : record ? `${record.kind} ${record.run_id} · ${record.stop_code}` : ''
  return (
    <section className="banner warn latch" aria-label={title}>
      <div className="body">
        <strong>{title}</strong>
        <p className="latch-body">{cell.halted ? t('cl.latch.body') : t('cl.stop.body')}</p>
        <p className="small dim">{t('cl.latch.pendant')}</p>
        {view === 'tech' && reason && <p className="small mono dim">{t('cl.latch.reason', { reason })}</p>}
        <div className="actions latch-actions">
          <label className="check latch-check">
            <input type="checkbox" checked={looked} onChange={(e) => setLooked(e.target.checked)} />
            {t('cl.latch.check')}
          </label>
          <button
            type="button"
            className="big"
            disabled={!looked || actions.busy !== null || Boolean(cell.active_run_id)}
            onClick={() => void actions.acknowledge().then(() => setLooked(false))}
          >
            {actions.busy === 'acknowledge' ? t('cl.latch.busy') : t('cl.latch.go')}
          </button>
        </div>
      </div>
    </section>
  )
}

export function TelemetryPanel({ connected }: { connected: boolean }) {
  const t = useT(SCREENS)
  const [telemetry, setTelemetry] = useState<TelemetryOut | null>(null)
  const [controllerState, setControllerState] = useState(false)

  const readTelemetry = useCallback(() => {
    api
      .status(controllerState)
      .then(setTelemetry)
      .catch(() => setTelemetry(null))
  }, [controllerState])

  // Once immediately, then on the tick: without the immediate read the panel sat on "Reading the cell..." for a full
  // second after every connect, long enough, watching a robot come up, to read as "the console has lost it".
  useEffect(() => {
    if (connected) readTelemetry()
  }, [connected, readTelemetry])

  usePoll(readTelemetry, 1000, connected)

  // Derived, not cleared from an effect: a disconnected cell HAS no telemetry.
  const live = connected ? telemetry : null
  if (!connected) return null

  const notOffered = <span className="faint">{t('cl.tel.notOffered')}</span>
  return (
    <Panel
      title={t('cl.tel.title')}
      aside={
        <label className="check">
          <input type="checkbox" checked={controllerState} onChange={(e) => setControllerState(e.target.checked)} />
          {t('cl.tel.controller')}
        </label>
      }
    >
      {!live ? (
        <Loading what={t('cl.what')} />
      ) : (
        <>
          <div className="provenance-line">
            <ProvenanceBadge telemetry={live} /> <ProvenanceDetail telemetry={live} />
          </div>
          <KeyValues
            pairs={[
              [t('cl.tel.k.tcp'), <Vec3 key="p" v={live.tcp_position_mm} />],
              [
                t('cl.tel.k.quat'),
                live.tcp_quaternion_xyzw ? (
                  <span key="q" className="mono">
                    {live.tcp_quaternion_xyzw.map((n) => n.toFixed(4)).join('  ')}
                  </span>
                ) : (
                  <span key="qn">{notOffered}</span>
                ),
              ],
              [
                t('cl.tel.k.joints'),
                live.joint_positions ? (
                  <span key="j" className="mono">
                    {live.joint_positions.map((n) => n.toFixed(3)).join('  ')}
                  </span>
                ) : (
                  <span key="jn">{notOffered}</span>
                ),
              ],
              [t('cl.tel.k.force'), <Vec3 key="f" v={live.tcp_force_n} unit="N" />],
              [t('cl.tel.k.torque'), <Vec3 key="t" v={live.tcp_torque_nm} unit="Nm" />],
            ]}
          />
          <div className="stack-item">
            {live.controller_state_included ? (
              <KeyValues
                pairs={[
                  [t('cl.tel.k.mode'), live.robot_mode ?? '—'],
                  [t('cl.tel.k.safety'), live.safety_mode ?? '—'],
                  [
                    t('cl.tel.k.protective'),
                    live.protective_stopped ? <StatusPill key="ps" status="block">{t('cl.tel.stopped')}</StatusPill> : t('cl.tel.no'),
                  ],
                  [
                    t('cl.tel.k.emergency'),
                    live.emergency_stopped ? <StatusPill key="es" status="block">{t('cl.tel.stopped')}</StatusPill> : t('cl.tel.no'),
                  ],
                  [t('cl.tel.k.says'), live.controller_message || '—'],
                ]}
              />
            ) : (
              <span className="small faint">{t('cl.tel.notAsked')}</span>
            )}
          </div>
          <Caveat>{t('cl.tel.caveat')}</Caveat>
        </>
      )}
    </Panel>
  )
}

/** Every panel of the cell, stacked, over the cell as this component reads it. */
export default function Cell() {
  const t = useT(SCREENS)
  const cell = useAsync(() => api.cell())
  const actions = useCellActions(cell.reload)
  const [rehearse, setRehearse] = useState<boolean | null>(null)

  if (cell.error) return <ErrorBanner error={cell.error} onRetry={cell.reload} />
  if (cell.loading && !cell.data) return <Loading what={t('cl.what')} />
  const c = cell.data
  if (!c) return null

  return (
    <>
      <h2>{t('cl.title')}</h2>
      <p className="lede">{t('cl.lede')}</p>
      {actions.error && <ErrorBanner error={actions.error} />}
      <StatePanel cell={c} />
      <SubstitutionBanner cell={c} />
      <LatchPanel cell={c} actions={actions} />
      <BuildPanel cell={c} actions={actions} rehearse={rehearse ?? rehearseByDefault(c)} onRehearse={setRehearse} />
      <PreviewPanel cell={c} actions={actions} />
      <ConnectPanel cell={c} actions={actions} />
      <TelemetryPanel connected={c.state === 'connected'} />
    </>
  )
}
