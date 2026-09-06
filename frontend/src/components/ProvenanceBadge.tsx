/** The badge itself. The RULE it renders lives in `lib/provenance.ts` and is tested there. */

import type { TelemetryOut } from '../api/client'
import { provenanceOf } from '../lib/provenance'
import { StatusPill } from './ui'

export function ProvenanceBadge({ telemetry }: { telemetry: TelemetryOut | null | undefined }) {
  const p = provenanceOf(telemetry)
  return <StatusPill status={p.tone}>{p.label}</StatusPill>
}
