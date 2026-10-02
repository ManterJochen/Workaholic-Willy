/**
 * A pose's screen verdict as one short chip (build plan 1.8; the screen of `src/robot/safety/planning/band.py`).
 *
 * Short, because a chip is read at a glance in a table and on the teach dialog's verdict: "frei", "frei · gerader
 * Start", "abgelehnt: Kollisionsprüfung", "abgelehnt: Planer", "nicht geprüft". What a band verdict means (a planned
 * move to or from the pose begins with a straight leg of at most 10° per joint) is the chip's title. A verdict this
 * console does not know reads as refused, never as clear; a pose with no verdict (one from the config) as not screened.
 */

import { StatusPill } from '../components/ui'
import { useT } from '../i18n'
import { SETUP } from './i18n'

export default function VerdictChip({ verdict }: { verdict: string | null | undefined }) {
  const t = useT(SETUP)
  switch (verdict) {
    case 'clear':
      return <StatusPill status="ok">{t('ps.chip.clear')}</StatusPill>
    case 'band':
      return (
        <span className="su-chip-why" title={t('ps.chip.band.title')}>
          <StatusPill status="info">{t('ps.chip.band')}</StatusPill>
        </span>
      )
    case 'guard_refused':
      return <StatusPill status="block">{t('ps.chip.guard_refused')}</StatusPill>
    case 'planner_refused':
      return <StatusPill status="block">{t('ps.chip.planner_refused')}</StatusPill>
    case 'unscreened':
      return <StatusPill status="warn">{t('ps.chip.unscreened')}</StatusPill>
    case null:
    case undefined:
    case '':
      return <StatusPill status="idle">{t('ps.chip.unscreened')}</StatusPill>
    default:
      return <StatusPill status="block">{t('ps.chip.error')}</StatusPill>
  }
}
