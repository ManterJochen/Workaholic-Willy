/**
 * The Advanced drawer of the Understood card (OD 12; build plan 4.2): per task, never written to the cell's config,
 * collapsed by default with its settings in one summary line.
 *
 * Multi-view (on), both jaw faces (off), the closing axis (any), the push distance (the cell's, hidden where the cell
 * cannot push), recordings (off), and, for a target the camera finds, the air over its rim (20 mm, 10-50). The tech
 * view adds the overlay switch, as the measuring aid it is. Each setting says in one line what it does.
 */

import type { CellFactsOut } from '../api/client'
import { useT } from '../i18n'
import { CLOSING_AXES, PUSH_MIN_MM, PUSH_MM, RIM_AIR_MM, RIM_AIR_RANGE, type ClosingAxis, type Draft, type DraftOptions } from './draft'
import { COCKPIT } from './i18n'

export interface AdvancedDrawerProps {
  readonly draft: Draft
  readonly facts: CellFactsOut | null
  readonly tech: boolean
  change(options: Partial<DraftOptions>): void
  /** The drawer was opened (it opens downwards, over what is under it). */
  opened?(): void
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value))
}

export default function AdvancedDrawer({ draft, facts, tech, change, opened }: AdvancedDrawerProps) {
  const t = useT(COCKPIT)
  const o = draft.options
  const camera = draft.place.kind === 'camera'
  const canPush = facts?.push?.can_push !== false
  const pushShown = o.pushMm ?? facts?.push?.default_mm ?? PUSH_MM
  const pushMax = Math.max(PUSH_MIN_MM, facts?.push?.ceiling_mm ?? 50)
  const rimShown = o.rimAirMm ?? RIM_AIR_MM
  const onOff = (on: boolean) => t(on ? 'ck.adv.on' : 'ck.adv.off')

  const summary = [
    t('ck.adv.summary.multiView', { state: onOff(o.multiView) }),
    t('ck.adv.summary.bothFaces', { state: onOff(o.bothFaces) }),
    ...(o.closingAxis ? [t('ck.adv.summary.axis', { axis: o.closingAxis })] : []),
    ...(canPush ? [t('ck.adv.summary.push', { mm: pushShown })] : []),
    t('ck.adv.summary.record', { state: onOff(o.recordViews) }),
    ...(camera ? [t('ck.adv.summary.rim', { mm: rimShown })] : []),
  ].join(' · ')

  return (
    <details className="ck-adv" onToggle={(e) => e.currentTarget.open && opened?.()}>
      <summary>
        <span className="ck-adv-title">{t('ck.adv.title')}</span>
        <span className="ck-adv-summary">{summary}</span>
      </summary>
      <div className="ck-adv-grid">
        <label className="ck-switch">
          <input type="checkbox" checked={o.multiView} onChange={(e) => change({ multiView: e.target.checked })} />
          <span className="ck-switch-name">{t('ck.adv.multiView')}</span>
          <span className="ck-help">{t('ck.adv.multiViewHelp')}</span>
        </label>
        <label className="ck-switch">
          <input type="checkbox" checked={o.bothFaces} onChange={(e) => change({ bothFaces: e.target.checked })} />
          <span className="ck-switch-name">{t('ck.adv.bothFaces')}</span>
          <span className="ck-help">{t('ck.adv.bothFacesHelp')}</span>
        </label>
        <label className="ck-field">
          <span className="ck-switch-name">{t('ck.adv.axis')}</span>
          <select
            value={o.closingAxis ?? ''}
            onChange={(e) => change({ closingAxis: e.target.value === '' ? null : (e.target.value as ClosingAxis) })}
          >
            <option value="">—</option>
            {CLOSING_AXES.map((axis) => (
              <option key={axis} value={axis}>
                {axis}
              </option>
            ))}
          </select>
          <span className="ck-help">{t('ck.adv.axisHelp')}</span>
        </label>
        {canPush && (
          <label className="ck-field">
            <span className="ck-switch-name">{t('ck.adv.push')}</span>
            <span className="ck-number">
              <input
                type="number"
                min={PUSH_MIN_MM}
                max={pushMax}
                step={1}
                value={pushShown}
                // Typed freely (a "5" may become "50"), held to the server's range once the field is left: under
                // 10 mm the server refuses the push (`push_distance_refused`), over the cell's ceiling too.
                onChange={(e) => {
                  const mm = Number(e.target.value)
                  change({ pushMm: Number.isFinite(mm) && mm > 0 ? Math.min(Math.round(mm), pushMax) : null })
                }}
                onBlur={() => {
                  if (o.pushMm !== null && o.pushMm < PUSH_MIN_MM) change({ pushMm: clamp(o.pushMm, PUSH_MIN_MM, pushMax) })
                }}
              />
              mm
            </span>
            <span className="ck-help">{t('ck.adv.pushHelp', { max: pushMax })}</span>
          </label>
        )}
        <label className="ck-switch">
          <input type="checkbox" checked={o.recordViews} onChange={(e) => change({ recordViews: e.target.checked })} />
          <span className="ck-switch-name">{t('ck.adv.record')}</span>
          <span className="ck-help">{t('ck.adv.recordHelp')}</span>
        </label>
        {camera && (
          <label className="ck-field">
            <span className="ck-switch-name">{t('ck.adv.rim')}</span>
            <span className="ck-number">
              <input
                type="number"
                min={RIM_AIR_RANGE[0]}
                max={RIM_AIR_RANGE[1]}
                step={1}
                value={rimShown}
                onChange={(e) => {
                  const mm = Number(e.target.value)
                  change({ rimAirMm: Number.isFinite(mm) ? clamp(Math.round(mm), RIM_AIR_RANGE[0], RIM_AIR_RANGE[1]) : null })
                }}
              />
              mm
            </span>
            <span className="ck-help">{t('ck.adv.rimHelp')}</span>
          </label>
        )}
        {tech && (
          <label className="ck-switch">
            <input type="checkbox" checked={o.overlay} onChange={(e) => change({ overlay: e.target.checked })} />
            <span className="ck-switch-name">{t('ck.adv.overlay')}</span>
            <span className="ck-help">{t('ck.adv.overlayHelp')}</span>
          </label>
        )}
      </div>
    </details>
  )
}
