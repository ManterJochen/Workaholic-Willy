/**
 * One outcome string, one colour — and the string is `succeeded`, not `success`.
 *
 * This file exists because it wasn't. Three screens each decided for themselves what a good outcome
 * looks like, and the History screen compared against `'success'` while `AutonomousGraspOutcome`
 * spells it `SUCCEEDED = "succeeded"` (and `replay/kpi.py` computes `pick_success_rate` off exactly
 * that literal). Every successful logged attempt therefore rendered as a WARNING pill — the console
 * showing a worse result than the cell actually achieved, on the one screen an operator goes to for
 * the record.
 *
 * The lesson is not "fix the string": it is that a comparison against a backend enum, written out by
 * hand at each call site, is a bug waiting for its third copy. So the mapping lives here, once.
 *
 * The taxonomy is deliberately three-way rather than pass/fail, because two of these categories mean
 * genuinely different things to the person reading them:
 *
 *   ok      the cell did what was asked.
 *   block   the cell REFUSED — safety, a missing frame, a decision gate. Nothing went wrong; a rule
 *           fired. An operator's next move is to change a setting, not to inspect a robot.
 *   warn    the cell tried and did not manage it. That one is about the scene or the grasp.
 */

/** Outcomes where the stack declined to act rather than failing to. `pill` renders these red. */
const REFUSALS = new Set([
  'missing_camera_frame',
  'unsafe_recovery_refused',
  'decision_fail_closed',
  'decision_recover_pending',
  'refinement_diverged',
  'mode_not_available',
])

/** The literal the backend writes for a good pick. Pinned here so no screen re-types it. */
export const SUCCEEDED = 'succeeded'

/** A `GraspAttemptRecord.final_outcome` / `AutonomousGraspReport.outcome` → a status pill tone. */
export function outcomeTone(outcome: string | null | undefined): string {
  if (!outcome) return 'idle'
  const value = String(outcome)
  if (value === SUCCEEDED) return 'ok'
  if (REFUSALS.has(value)) return 'block'
  return 'warn'
}

/** True for the one outcome that counts as a pick. */
export function isSuccess(outcome: string | null | undefined): boolean {
  return String(outcome ?? '') === SUCCEEDED
}

/** `no_valid_grasp` → `no valid grasp`. The enum is for programs; the pill is for people. */
export function outcomeLabel(outcome: string | null | undefined): string {
  return String(outcome ?? '—').replace(/_/g, ' ')
}

/** A run's lifecycle state (`running` | `finished` | `cancelled` | `failed`) → a pill tone. */
export function runTone(state: string | null | undefined): string {
  switch (state) {
    case 'finished':
      return 'ok'
    case 'running':
      return 'info'
    case 'failed':
      return 'block'
    case 'cancelled':
      return 'warn'
    default:
      return 'idle'
  }
}
