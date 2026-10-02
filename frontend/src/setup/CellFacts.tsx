/**
 * What this cell is, as it was built (`GET /v1/cell/facts`, read once after a build and once after a connect): the
 * arm, the hand and its output, the cameras and the looks, the planner, whether it can push, how "halt now" acts on
 * it, and whether the carried part is modelled (a task refuses to run until it is: build plan item 9).
 *
 * Read only. Each fact says what the cell will do, in words; the server's own sentences behind them (why a push is
 * not possible, why the part is not modelled, the detector's precision) are details of the tech view.
 */

import { useId, type ReactNode } from 'react'

import type { CellFactsOut, CellOut, HandOut, RigOut } from '../api/client'
import { useT } from '../i18n'
import { whereMsg } from '../i18n/codes'
import { usePrefs } from '../model/prefs'
import { useCell } from '../model/useCell'
import { SETUP, type SetupT } from './i18n'

function handWords(t: SetupT, hand: HandOut | undefined): string {
  if (!hand || hand.kind === 'none') return t('hand.none')
  if (hand.kind === 'toggle') return t('fx.hand.toggle', { where: whereMsg(hand.where) })
  const kind = t(`hand.kind.${hand.kind}`)
  return hand.no_sensor ? `${kind} · ${t('hand.noSensor')}` : kind
}

function rigWords(t: SetupT, rig: RigOut): string {
  const mount = t(`fx.mount.${rig.mounting}`)
  return [rig.rig_id, mount, rig.primary ? t('fx.primary') : null].filter(Boolean).join(' · ')
}

function cameraWords(t: SetupT, facts: CellFactsOut): string {
  if (facts.rehearsal) return t('fx.camera.rehearsal')
  const rigs = facts.cameras ?? []
  return rigs.length ? rigs.map((rig) => rigWords(t, rig)).join('; ') : t('fx.camera.none')
}

function plannerWords(t: SetupT, facts: CellFactsOut, cell: CellOut | null): string {
  const route = t(`fx.route.${facts.route.route}`)
  const state = cell?.planner?.state
  return state && state !== 'not_used' ? `${route} · ${t(`fx.planner.${state}`)}` : route
}

function pushWords(t: SetupT, facts: CellFactsOut): string {
  const push = facts.push
  if (!push?.can_push) return t('fx.push.no')
  return t('fx.push.yes', { mm: push.default_mm ?? '—', max: push.ceiling_mm ?? '—' })
}

function payloadWords(t: SetupT, facts: CellFactsOut): string {
  const payload = facts.payload
  if (payload?.modelled) {
    return payload.length_mm != null ? t('fx.payload.modelled', { mm: payload.length_mm }) : t('fx.payload.modelledBare')
  }
  return payload?.declined_reason ? t('fx.payload.not') : t('fx.payload.na')
}

/** The detector backends the server names (`PerceptionStackOut.backend`), each by the name a person knows it by. */
const DETECTORS = {
  grounded_sam: 'fx.detector.grounded_sam',
  vlm: 'fx.detector.vlm',
  rtdetr: 'fx.detector.rtdetr',
} as const

/** A detector backend by its name; one this console does not know yet is shown as the server names it. */
function detectorWords(t: SetupT, backend: string): string {
  const key = (DETECTORS as Record<string, (typeof DETECTORS)[keyof typeof DETECTORS]>)[backend]
  return key ? t(key) : backend
}

function brakeWords(t: SetupT, facts: CellFactsOut): string {
  if (!facts.brake?.latches) return t('fx.brake.none')
  return facts.brake.brakes_in_motion ? t('fx.brake.brakes') : t('fx.brake.latches')
}

export default function CellFacts() {
  const t = useT(SETUP)
  const titleId = useId()
  const { view } = usePrefs()
  const { cell, facts, telemetry } = useCell()
  const tech = view === 'tech'

  const rows: Array<[string, ReactNode, string?]> = []
  if (facts) {
    const model = [telemetry?.vendor ?? cell?.vendor, telemetry?.model].filter(Boolean).join(' ')
    rows.push([t('fx.arm'), model || cell?.arm || '—', tech ? (cell?.arm ?? undefined) : undefined])
    rows.push([t('fx.hand'), handWords(t, cell?.hand), tech ? cell?.hand?.driver || undefined : undefined])
    rows.push([t('fx.camera'), cameraWords(t, facts)])
    const looks = facts.looks ?? []
    rows.push([
      t('fx.looks'),
      looks.length ? t('fx.looks.n', { n: looks.length }) : t('fx.looks.none'),
      tech && looks.length ? looks.join(', ') : undefined,
    ])
    rows.push([t('fx.planner'), plannerWords(t, facts, cell), tech ? facts.route.sentence : undefined])
    rows.push([t('fx.push'), pushWords(t, facts), tech && !facts.push?.can_push ? facts.push?.why_not : undefined])
    rows.push([t('fx.payload'), payloadWords(t, facts), tech ? (facts.payload?.declined_reason ?? undefined) : undefined])
    rows.push([t('fx.brake'), brakeWords(t, facts)])
    const detector = facts.detector
    if (detector) {
      // The backend by its name (its config word is the tech view's), then the model a VLM runs.
      rows.push([
        t('fx.detector'),
        [detector.backend ? detectorWords(t, detector.backend) : null, detector.vlm_model_id].filter(Boolean).join(' · ') || '—',
        tech
          ? [detector.backend, detector.precision, detector.router_enabled ? 'router' : null].filter(Boolean).join(' · ')
          : undefined,
      ])
    }
    if (tech) {
      rows.push([t('fx.handEye'), t('fx.mm', { mm: facts.hand_eye_warn_mm })])
      if (facts.natural_closing_axis) rows.push([t('fx.axis'), facts.natural_closing_axis])
    }
  }

  return (
    <section className="panel su-facts" aria-labelledby={titleId}>
      <div className="panel-head">
        <h3 id={titleId}>{t('fx.title')}</h3>
      </div>
      <p className="panel-lede">{t('fx.lede')}</p>
      {facts ? (
        <dl className="su-facts-list">
          {rows.map(([key, value, detail]) => (
            <div key={key} className="su-fact">
              <dt>{key}</dt>
              <dd>
                {value}
                {detail && <span className="su-fact-detail">{detail}</span>}
              </dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="empty">{t('fx.empty')}</p>
      )}
    </section>
  )
}
