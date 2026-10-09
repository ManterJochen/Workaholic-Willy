/**
 * Whether a task may start now, said once for the card's Start and for Enter (the owner, 2026-10-08).
 *
 * `cellOffOf` is the cell's half: not connected, a run, a stop record, a ready bar that is not green, each in the
 * reader's words. The card's Start draws it from the shared cell poll; Enter asks the server afresh (`gate`), because a
 * poll up to two seconds old is not what a start may stand on, and adds what the card would say about itself
 * (`draftProblems`) and one rule of its own: on a cell that detects with the language model, a model that is not
 * loaded yet would load in the task's first look (144 s on the cell, the arm standing at the look), so Enter opens the
 * card and says to load it first.
 *
 * Nothing here starts anything. Whatever it says, `POST /v1/task` checks every gate again on the server and refuses
 * what it must; `gate` only decides whether Enter goes on to ask it, or opens the card.
 */

import { api, type CellFactsOut, type CellOut, type PosesOut, type ReadinessOut } from '../api/client'
import { blockerMsg, lightIdMsg, lightMsg, refusalMsg } from '../i18n/codes'
import type { Msg } from '../i18n/types'
import { draftProblems, PROBLEM_KEY, type Draft } from './draft'

/** Why a task may not start for the cell's sake, in the order the card says it; empty when the cell allows it. */
export function cellOffOf(cell: CellOut | null, readiness: ReadinessOut | null, running: boolean): Msg<string>[] {
  const off: Msg<string>[] = []
  if (cell?.state !== 'connected') off.push(refusalMsg('not_connected'))
  else if (running) off.push(blockerMsg('run_active'))
  else if (cell.recovery) off.push(blockerMsg('restart_required'))
  else if (!readiness) off.push({ key: 'ck.start.readinessUnknown' })
  else if (!readiness.ready) {
    for (const light of readiness.lights ?? []) {
      if (light.blocks && light.state !== 'ok') off.push({ key: 'ck.start.light', params: { id: lightIdMsg(light.id), state: lightMsg(light.code) } })
    }
    for (const blocker of readiness.blockers ?? []) off.push(blockerMsg(blocker.code))
    if (off.length === 0) off.push({ key: 'ck.ready.not' })
  }
  return off
}

/**
 * A cell that detects with the language model, whose model is not loaded: the commands light says `ready` once it is
 * (`api/readiness.py`). A cell that detects otherwise, or whose facts are not known, needs nothing loaded to start.
 */
export function modelToLoad(facts: CellFactsOut | null, readiness: ReadinessOut | null): boolean {
  if (facts?.detector?.backend !== 'vlm') return false
  return (readiness?.lights ?? []).find((light) => light.id === 'commands')?.code !== 'ready'
}

export interface GateAnswer {
  /** Why Enter does not start: empty when it may. */
  readonly why: readonly Msg<string>[]
  /** The 3 s hands-off countdown comes first (`CellOut.countdown_due`), as the server said it just now. */
  readonly countdown: boolean
  /** The cell's facts as read just now (the ones known before where the read failed): the first motion is theirs. */
  readonly facts: CellFactsOut | null
}

/**
 * Whether Enter may start `card` now, asked of the server afresh: the cell, its ready bar and its facts, then the
 * card's own problems against the taught `poses`. A server that does not answer is a reason too. Never throws.
 */
export async function gate(card: Draft, context: { facts: CellFactsOut | null; poses: PosesOut | null }): Promise<GateAnswer> {
  try {
    const [cell, readiness, facts] = await Promise.all([
      api.cell(),
      api.readiness(),
      api.facts().catch(() => context.facts),
    ])
    const why = cellOffOf(cell, readiness, cell.active_run_id != null)
    if (modelToLoad(facts, readiness)) why.push({ key: 'ck.start.loadFirst' })
    for (const problem of draftProblems(card, { rehearsal: facts?.rehearsal === true, poses: context.poses })) {
      why.push({ key: PROBLEM_KEY[problem] })
    }
    return { why, countdown: cell.countdown_due === true, facts }
  } catch {
    return { why: [{ key: 'ck.ready.noServer' }], countdown: false, facts: context.facts }
  }
}
