/**
 * What is on the other end of the connection, as one pill, in the reader's language. The RULE it renders lives in
 * `lib/provenance.ts` and is tested there: a simulated arm is never a robot, and a controller is never claimed to
 * drive a real one. `ProvenanceDetail` is the sentence that says why.
 */

import type { TelemetryOut } from '../api/client'
import { useT } from '../i18n'
import { provenanceOf } from '../lib/provenance'
import { SCREENS } from '../screens/i18n'
import { StatusPill } from './ui'

export function ProvenanceBadge({ telemetry }: { telemetry: TelemetryOut | null | undefined }) {
  const t = useT(SCREENS)
  const p = provenanceOf(telemetry)
  const name = t(`pv.label.${p.kind}`)
  return <StatusPill status={p.tone}>{p.where && p.kind !== 'unknown' ? `${name} · ${p.where}` : name}</StatusPill>
}

export function ProvenanceDetail({ telemetry }: { telemetry: TelemetryOut | null | undefined }) {
  const t = useT(SCREENS)
  const p = provenanceOf(telemetry)
  if (p.kind === 'unknown') return null
  const text =
    p.kind === 'sim-controller'
      ? t('pv.detail.sim-controller', { serial: p.serial ?? '?', host: p.host ?? '?' })
      : t(`pv.detail.${p.kind}`)
  return <span className="small dim">{text}</span>
}
