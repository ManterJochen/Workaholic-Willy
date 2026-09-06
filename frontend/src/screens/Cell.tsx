/**
 * Build → preview → connect → live telemetry → disconnect.
 *
 * The connect flow is the one place in this console where the UI is deliberately in the way. The
 * backend hands out a token from `/v1/cell/connect-preview` and refuses `/v1/cell/connect` without
 * it; there is no force flag on either side. This screen therefore CANNOT offer a one-click connect,
 * and does not try to: the preview's motion warnings are rendered first, and the button that follows
 * them is the only red one on the page.
 *
 * Two backend fields drive most of what is drawn here, and both exist because a console that omits
 * them is actively misleading:
 *
 * * `gripper_substitution` -- a real end-effector could not be built. Connect is refused while it is
 *   set, because the cell would come up, every pick would report success, and the gripper would close
 *   on nothing.
 * * `TelemetryOut.simulated` -- a simulated arm produces numbers that look exactly like a real one's.
 *   This is the only field that says which produced the ones on screen, so it is always visible.
 */

import { useCallback, useEffect, useState } from 'react'

import { ApiError, api, type CellOut, type ConnectPreviewOut, type TelemetryOut } from '../api/client'
import { ProvenanceBadge } from '../components/ProvenanceBadge'
import { provenanceOf } from '../lib/provenance'
import { Caveat, ErrorBanner, KeyValues, Loading, Panel, StatusPill, Vec3 } from '../components/ui'
import { useAsync, usePoll } from '../lib/useAsync'

export default function Cell() {
  const cell = useAsync(() => api.cell())
  const [busy, setBusy] = useState<string | null>(null)
  const [actionError, setActionError] = useState<ApiError | null>(null)
  const [preview, setPreview] = useState<ConnectPreviewOut | null>(null)
  const [rehearse, setRehearse] = useState(true)
  const [telemetry, setTelemetry] = useState<TelemetryOut | null>(null)
  const [controllerState, setControllerState] = useState(false)

  const run = useCallback(
    async (label: string, fn: () => Promise<CellOut | ConnectPreviewOut>) => {
      setBusy(label)
      setActionError(null)
      try {
        const result = await fn()
        if ('token' in result) setPreview(result as ConnectPreviewOut)
        else cell.reload()
      } catch (err) {
        setActionError(err instanceof ApiError ? err : new ApiError(0, null, String(err)))
      } finally {
        setBusy(null)
      }
    },
    [cell],
  )

  const connected = cell.data?.state === 'connected'

  const readTelemetry = useCallback(() => {
    api
      .status(controllerState)
      .then(setTelemetry)
      .catch(() => setTelemetry(null))
  }, [controllerState])

  // Once immediately, then on the tick. Without the immediate read the panel sat on "Reading the
  // cell..." for a full second after every connect and after every toggle of the controller-state
  // switch -- long enough, watching a robot come up, to read as "the console has lost it".
  useEffect(() => {
    if (connected) readTelemetry()
  }, [connected, readTelemetry])

  usePoll(readTelemetry, 1000, connected)

  // Derived, not cleared from an effect: a disconnected cell HAS no telemetry, so the last reading is
  // not stale data to be tidied away later -- it is data that must stop being shown the same render.
  const live = connected ? telemetry : null

  if (cell.error) return <ErrorBanner error={cell.error} onRetry={cell.reload} />
  if (cell.loading && !cell.data) return <Loading what="the cell" />
  const c = cell.data
  if (!c) return null

  const substitution = c.gripper_substitution as Record<string, unknown> | null

  return (
    <>
      <h2>Cell</h2>
      <p className="lede">
        Assembling the cell touches no robot. Connecting does — it releases brakes and, on most
        grippers, sweeps the fingers. Everything between those two sentences is on this page.
      </p>

      {actionError && <ErrorBanner error={actionError} />}

      <Panel
        title="State"
        aside={
          <>
            profile <span className="mono">{c.profile || '(none)'}</span>
          </>
        }
      >
        <KeyValues
          pairs={[
            ['state', <StatusPill key="s" status={connected ? 'ok' : 'idle'}>{c.state}</StatusPill>],
            ['vendor', c.vendor],
            ['arm', c.arm ?? <span key="a" className="faint">not built</span>],
            ['gripper', c.gripper ?? <span key="g" className="faint">not built</span>],
            ['lock holder', c.lock_holder ?? <span key="l" className="faint">nobody</span>],
            ['active run', c.active_run_id ?? <span key="r" className="faint">none</span>],
          ]}
        />
      </Panel>

      {substitution && (
        <div className="banner error">
          <div className="body">
            <strong>A real end-effector could not be built, and connect is refused.</strong>
            <div className="small dim">
              Left alone, the cell would come up, every pick would report success, and the gripper
              would close on nothing.
            </div>
            <pre className="payload">{JSON.stringify(substitution, null, 2)}</pre>
          </div>
        </div>
      )}

      <Panel title="1 · Assemble">
        <div className="actions">
          <button
            className="primary"
            disabled={busy !== null || connected}
            onClick={() => run('build', () => api.build(rehearse))}
          >
            {busy === 'build' ? 'Assembling…' : 'Build the cell'}
          </button>
          <label className="check">
            <input
              type="checkbox"
              checked={rehearse}
              disabled={connected}
              onChange={(e) => setRehearse(e.target.checked)}
            />
            rehearse
          </label>
        </div>
        <Caveat>
          <strong>rehearse</strong> substitutes a desk scene for the camera and the two models, so the
          whole path runs in seconds with no GPU. A real build opens the camera and loads the
          detector and segmenter, which takes tens of seconds — and grounds prompts against what the
          camera actually sees. A rehearsed cell grasps a synthetic box and proves the WIRING, never
          the perception.
        </Caveat>
      </Panel>

      <Panel title="2 · Preview what connecting will do">
        <div className="actions">
          <button
            disabled={busy !== null || !c.arm || connected}
            onClick={() => run('preview', () => api.connectPreview())}
          >
            {busy === 'preview' ? 'Reading…' : 'Show me'}
          </button>
          {!c.arm && <span className="small faint">Build the cell first.</span>}
        </div>

        {preview && (
          <div className="stack">
            <KeyValues
              pairs={[
                ['arm', preview.arm],
                ['gripper', preview.gripper],
                ['token expires', preview.expires_at],
              ]}
            />
            {(preview.blocking ?? []).length > 0 && (
              <div className="banner error stack-item">
                <div className="body">
                  <strong>Connect will be refused, whatever token is presented.</strong>
                  <ul className="reasons">
                    {(preview.blocking ?? []).map((b) => (
                      <li key={b}>{b}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
            {(preview.warnings ?? []).length > 0 && (
              <div className="stack-item">
                <h3>This will physically move:</h3>
                {(preview.warnings ?? []).map((w) => (
                  <div className="row" key={`${w.subject}-${w.what}`}>
                    <div>
                      <StatusPill status="warn">{w.subject}</StatusPill>
                    </div>
                    <div>
                      <div className="detail">{w.what}</div>
                      <div className="fix">{w.precaution}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </Panel>

      <Panel title="3 · Connect">
        <div className="actions">
          <button
            className="danger"
            disabled={busy !== null || !preview || (preview.blocking ?? []).length > 0 || connected}
            onClick={() => preview && run('connect', () => api.connect(preview.token))}
          >
            {busy === 'connect' ? 'Connecting…' : 'Connect — this moves the robot'}
          </button>
          <button
            disabled={busy !== null || !connected}
            onClick={() => run('disconnect', () => api.disconnect())}
          >
            Disconnect
          </button>
          {!preview && !connected && (
            <span className="small faint">Read the preview above first — it issues the token.</span>
          )}
        </div>
      </Panel>

      {connected && (
        <Panel
          title="Live telemetry"
          aside={
            <label className="check">
              <input
                type="checkbox"
                checked={controllerState}
                onChange={(e) => setControllerState(e.target.checked)}
              />
              include controller state
            </label>
          }
        >
          {!live ? (
            <Loading what="the cell" />
          ) : (
            <>
              <div className="provenance-line">
                <ProvenanceBadge telemetry={live} />{' '}
                <span className="small dim">{provenanceOf(live).detail}</span>
              </div>
              <KeyValues
                pairs={[
                  ['TCP', <Vec3 key="p" v={live.tcp_position_mm} />],
                  [
                    'quat xyzw',
                    live.tcp_quaternion_xyzw ? (
                      <span key="q" className="mono">
                        {live.tcp_quaternion_xyzw.map((n) => n.toFixed(4)).join('  ')}
                      </span>
                    ) : (
                      <span key="qn" className="faint">not offered by this driver</span>
                    ),
                  ],
                  [
                    'joints',
                    live.joint_positions ? (
                      <span key="j" className="mono">
                        {live.joint_positions.map((n) => n.toFixed(3)).join('  ')}
                      </span>
                    ) : (
                      <span key="jn" className="faint">not offered by this driver</span>
                    ),
                  ],
                  ['force N', <Vec3 key="f" v={live.tcp_force_n} unit="N" />],
                  ['torque Nm', <Vec3 key="t" v={live.tcp_torque_nm} unit="Nm" />],
                ]}
              />
              <div className="stack-item">
                {live.controller_state_included ? (
                  <KeyValues
                    pairs={[
                      ['robot mode', live.robot_mode ?? '—'],
                      ['safety mode', live.safety_mode ?? '—'],
                      [
                        'protective stop',
                        live.protective_stopped ? (
                          <StatusPill key="ps" status="block">stopped</StatusPill>
                        ) : (
                          'no'
                        ),
                      ],
                      [
                        'emergency stop',
                        live.emergency_stopped ? (
                          <StatusPill key="es" status="block">stopped</StatusPill>
                        ) : (
                          'no'
                        ),
                      ],
                      ['controller says', live.controller_message || '—'],
                    ]}
                  />
                ) : (
                  <span className="small faint">
                    Controller mode and safety state were not read on this tick. They cost a dashboard
                    socket round trip, so they are opt-in — a blank here means “not asked”, never
                    “fine”.
                  </span>
                )}
              </div>
              <Caveat>
                A missing measurement above means this driver offers no such capability (sim, dummy and
                KUKA advertise no force/torque Protocol at all). It is not an error and not a zero.
              </Caveat>
            </>
          )}
        </Panel>
      )}
    </>
  )
}
