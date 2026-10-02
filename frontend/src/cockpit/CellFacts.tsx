/**
 * What this cell is, read once after a build and once after a connect (`GET /v1/cell/facts`), for the tech view
 * (OD 1; build plan 1.10): the cameras and looks, the natural closing axis, whether it can push, what "halt now" does
 * on this arm, the carried part, how its moves are planned, and whether it is the rehearsal cell.
 */

import type { CellFactsOut } from '../api/client'
import { KeyValues } from '../components/ui'
import { useT } from '../i18n'
import { COCKPIT } from './i18n'

export default function CellFacts({ facts }: { facts: CellFactsOut | null }) {
  const t = useT(COCKPIT)
  if (!facts) return null
  const yes = (on: boolean) => t(on ? 'diag.yes' : 'diag.no')
  const push = facts.push
  const brake = facts.brake
  const cameras = (facts.cameras ?? []).map((rig) => `${rig.rig_id} (${rig.mounting}${rig.primary ? ', primary' : ''})`).join(', ')
  return (
    <details className="ck-facts">
      <summary>{t('ck.facts.title')}</summary>
      <KeyValues
        pairs={[
          [t('ck.facts.wrist'), `${yes(facts.wrist_camera)}${cameras ? ` · ${cameras}` : ''}`],
          [t('ck.facts.looks'), (facts.looks ?? []).join(', ') || '—'],
          [t('ck.facts.axis'), facts.natural_closing_axis ?? '—'],
          [
            t('ck.facts.push'),
            push?.can_push ? t('ck.facts.pushYes', { mm: push.default_mm ?? '—', max: push.ceiling_mm ?? '—' }) : t('ck.facts.pushNo', { why: push?.why_not || '—' }),
          ],
          [t('ck.facts.detector'), [facts.detector?.backend, facts.detector?.vlm_model_id, facts.detector?.precision].filter(Boolean).join(' · ') || '—'],
          [
            t('ck.facts.brake'),
            brake?.brakes_in_motion ? t('ck.facts.brakes') : brake?.latches ? t('ck.facts.latches') : t('ck.facts.noLatch'),
          ],
          [
            t('ck.facts.payload'),
            facts.payload?.modelled ? `${yes(true)} · ${facts.payload.length_mm ?? '—'} mm` : facts.payload?.declined_reason || yes(false),
          ],
          [t('ck.facts.route'), `${facts.route.route}${facts.route.sentence ? ` · ${facts.route.sentence}` : ''}`],
          [t('ck.facts.handEye'), `${facts.hand_eye_warn_mm} mm`],
          [t('ck.facts.rehearsal'), yes(facts.rehearsal)],
        ]}
      />
    </details>
  )
}
