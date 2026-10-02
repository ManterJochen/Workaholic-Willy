/**
 * Diagnostics, as a drawer of the tech view (OD 1): reachable from every screen, never in the way of the demo view.
 *
 * It reads `GET /v1/diagnostics` when it opens (which nothing in the console called before) and says what this cell
 * can do: which drivers are registered and ready, whether cuRobo and the collision meshes are there, which perception
 * stack a prompt would touch and whether its weights are on this box, and whether the controller answers on the
 * network. Nothing here moves anything; the drawer says so, and only reads.
 */

import { useEffect, useId, useRef, type KeyboardEvent, type ReactNode } from 'react'

import { api } from '../api/client'
import { Icon } from '../icons'
import { useT } from '../i18n'
import { useAsync } from '../lib/useAsync'
import { ErrorBanner, Loading, StatusPill } from './ui'

export interface DiagnosticsDrawerProps {
  open: boolean
  onClose: () => void
}

function Rows({ pairs }: { pairs: Array<[string, ReactNode]> }) {
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

export default function DiagnosticsDrawer({ open, onClose }: DiagnosticsDrawerProps) {
  const t = useT()
  const titleId = useId()
  const closeRef = useRef<HTMLButtonElement | null>(null)
  const diag = useAsync(() => (open ? api.diagnostics() : Promise.resolve(null)), [open])

  useEffect(() => {
    if (open) closeRef.current?.focus()
  }, [open])

  if (!open) return null

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') {
      event.stopPropagation()
      onClose()
    }
  }

  const yes = (value: boolean | null | undefined) =>
    value === null || value === undefined ? t('common.none') : value ? t('diag.yes') : t('diag.no')
  const d = diag.data

  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <div className="drawer" role="dialog" aria-modal="true" aria-labelledby={titleId} onKeyDown={onKeyDown}>
        <div className="drawer-head">
          <h2 id={titleId}>{t('diag.title')}</h2>
          <div className="actions">
            <button type="button" className="ghost" onClick={diag.reload} disabled={diag.loading}>
              <Icon name="restart" size={16} />
              {t('diag.reload')}
            </button>
            <button ref={closeRef} type="button" className="iconbtn" onClick={onClose} aria-label={t('common.close')}>
              <Icon name="close" />
            </button>
          </div>
        </div>
        <div className="drawer-body">
          <p className="dim" style={{ margin: 0 }}>{t('diag.lede')}</p>
          {diag.error && <ErrorBanner error={diag.error} onRetry={diag.reload} />}
          {!d && !diag.error && <Loading what={t('diag.title')} />}
          {d && (
            <>
              <section className="drawer-section">
                <h3>
                  <Icon name="controller" size={16} />
                  {t('diag.vendors')}
                </h3>
                {(d.vendors ?? []).map((v) => (
                  <div className="vendor-row" key={`${v.vendor}-${v.kind}`}>
                    <div>
                      <div className="mono">
                        {v.vendor} <span className="faint">· {v.kind}</span>
                      </div>
                      {v.note && <div className="small dim">{v.note}</div>}
                    </div>
                    <StatusPill status={v.ready ? 'ok' : v.registered ? 'warn' : 'idle'}>
                      {v.ready ? t('diag.ready') : v.registered ? t('diag.notReady') : t('diag.notRegistered')}
                    </StatusPill>
                  </div>
                ))}
              </section>

              <section className="drawer-section">
                <h3>
                  <Icon name="bolt" size={16} />
                  {t('diag.motion')}
                </h3>
                <Rows
                  pairs={[
                    [t('diag.model'), <span key="model" className="mono">{d.motion_stack.model}</span>],
                    [t('diag.anchored'), yes(d.motion_stack.fully_anchored)],
                    [t('diag.curobo'), yes(d.motion_stack.curobo_available)],
                    [t('diag.collision'), d.motion_stack.collision_engine ?? t('common.none')],
                    [t('diag.meshes'), yes(d.motion_stack.mesh_bundle_present)],
                  ]}
                />
                {d.motion_stack.curobo_caveat && <div className="small dim stack-item">{d.motion_stack.curobo_caveat}</div>}
              </section>

              <section className="drawer-section">
                <h3>
                  <Icon name="camera" size={16} />
                  {t('diag.perception')}
                </h3>
                <Rows
                  pairs={[
                    [t('diag.pipeline'), d.perception.kind],
                    [t('diag.backend'), <span key="backend" className="mono">{d.perception.backend}</span>],
                    [t('diag.segmenter'), <span key="segmenter" className="mono">{d.perception.segmenter}</span>],
                    [t('diag.router'), yes(d.perception.router_enabled)],
                    [t('diag.vlm'), <span key="vlm" className="mono">{d.perception.vlm_model_id ?? t('common.none')}</span>],
                    [t('diag.weights'), yes(d.perception.vlm_weights_present)],
                  ]}
                />
                {d.perception.detail && <div className="small dim stack-item">{d.perception.detail}</div>}
              </section>

              <section className="drawer-section">
                <h3>
                  <Icon name="link" size={16} />
                  {t('diag.network')}
                </h3>
                <Rows
                  pairs={[
                    [
                      t('diag.address'),
                      <span key="address" className="mono">
                        {d.reachability.address ?? t('common.none')}
                        {d.reachability.port ? `:${d.reachability.port}` : ''}
                      </span>,
                    ],
                    [
                      t('chip.controller'),
                      !d.reachability.checked
                        ? t('diag.notChecked')
                        : d.reachability.reachable
                          ? t('diag.reachable', { ms: d.reachability.latency_ms ?? null })
                          : t('diag.unreachable'),
                    ],
                  ]}
                />
                {d.reachability.detail && <div className="small dim stack-item">{d.reachability.detail}</div>}
              </section>
            </>
          )}
        </div>
      </div>
    </>
  )
}
