/**
 * The stop card's gates (build plan 1.3.5): Restart and Home are offered only when every gate is green, each read from
 * the cell as it is NOW (the shared poll and the ready bar).
 *
 * | gate        | green when                                                                                      |
 * |-------------|--------------------------------------------------------------------------------------------------|
 * | controller  | the robot light says the controller can move (a halt aside: "Zelle ist frei" clears that)        |
 * | cleared     | a person confirmed "Zelle ist frei" after the stop (`recovery.cleared_at > recovery.at`)          |
 * | jaws        | a toggle hand's jaws were answered open since the stop (`jaws_confirmed_at > at`, jaws open)     |
 * | empty       | no part may be held now: the server's own word (`part_still_held` among the ready bar's blockers) |
 * | idle        | the cell is connected, no run holds it and no jaws question waits                                 |
 *
 * **The empty gate is the server's rule** (`api/readiness.py` `part_held`), not a copy of half of it: a toggle's count
 * that says closed, a planner that models a part, a hand that measures one, and, while a stop record stands, the
 * record's own belief that a part is in the jaws of a hand that can say nothing (no toggle, no measurement), until a
 * person says "Backen leer". Read from the live hand alone, that last case drew a green gate and enabled Restart and
 * Home, which the server then refused `part_still_held`, on a card that no longer offered "Backen leer": the cockpit
 * was stuck until the server restarted. Where the ready bar cannot be read, or predates the record, the record is
 * believed.
 *
 * The server enforces every one of these itself; the card only says which is still open, and keeps its buttons off.
 */

import type { CellFactsOut, CellOut, HaltStateOut, ReadinessOut } from '../api/client'
import { noPartModelled, type RefusalCode } from '../api/codes'
import type { RunView } from '../model/runModel'

/** Brake outcomes only an arm that brakes a move in flight reports (with brakes off the outcome is `ran_out`). */
const BRAKING_OUTCOMES: ReadonlySet<string> = new Set(['pending', 'braked', 'unconfirmed'])

/**
 * Whether this arm brakes a move in flight (build plan 4.2, the halt's words and its alarm): the cell's facts say so,
 * or the halt itself shows it: the run's own pending brake (`run_halt_requested.braking`), or a brake outcome on the
 * latch. Any one is enough: the alarm must not hang on a single read of the facts, which may have failed.
 */
export function armBrakes(facts: CellFactsOut | null, view: RunView, halted: HaltStateOut | null | undefined): boolean {
  return facts?.brake?.brakes_in_motion === true || view.halt.braking || BRAKING_OUTCOMES.has(halted?.brake ?? '')
}

export type GateId = 'controller' | 'cleared' | 'jaws' | 'empty' | 'idle'

export interface Gate {
  readonly id: GateId
  readonly ok: boolean
}

/** A hand whose jaws are a count of its output's changes (the owner's Hand-E on tool DO0): the jaws question decides. */
export function isToggle(cell: CellOut | null): boolean {
  return cell?.hand?.kind === 'toggle'
}

/** The robot light's code, `null` where the ready bar is not read. */
export function robotCode(readiness: ReadinessOut | null): string | null {
  return readiness?.lights?.find((light) => light.id === 'robot')?.code ?? null
}

/** The controller can move, a halt aside (the latch is the person's "Zelle ist frei" to clear, not the pendant's). */
export function controllerCanMove(readiness: ReadinessOut | null): boolean {
  const code = robotCode(readiness)
  return code === 'connected' || code === 'halted'
}

/** The controller is stopped: a protective or emergency stop, or a powered-off arm, cleared at the pendant. */
export function controllerStopped(readiness: ReadinessOut | null): boolean {
  return robotCode(readiness) === 'controller_stopped'
}

/**
 * Whether a part may be in the jaws now, as the server would refuse the way back for it (`part_still_held`).
 *
 * The live reads first (a toggle's count that says closed, a planner that models a part), then the server's own word:
 * `part_still_held` among the ready bar's blockers, which also counts a measuring hand's hold and the stop record's
 * belief for a hand that can say nothing. Where the ready bar was not read, or was read before the stop record it
 * should know (the poll reads the cell, then the ready bar; while a record stands the server always lists
 * `cell_not_cleared` or `restart_required`), the stop record is believed for every hand that is no toggle (a toggle's
 * jaws question decides): a part nobody can rule out keeps the way back shut, and the card offers "Backen leer" for it.
 * While the cell is not connected no hand is read at all (after a Disconnect, or a restart of the server): the record's
 * belief stands, whatever the hand turns out to be, so the card never says "no part" about a hand nobody read.
 */
export function partHeld(cell: CellOut, readiness: ReadinessOut | null): boolean {
  if (cell.hand?.jaws === 'closed' || !noPartModelled(cell.payload_model)) return true
  if (cell.state !== 'connected') return cell.recovery?.holding === true
  const blockers = readiness?.blockers ?? []
  const said = (code: string) => blockers.some((blocker) => blocker.code === code)
  const knowsTheRecord = cell.recovery == null || said('cell_not_cleared') || said('restart_required')
  if (readiness && knowsTheRecord) return said('part_still_held')
  return cell.recovery?.holding === true && !isToggle(cell)
}

/** The gates of the stop card, in the card's order; the jaws gate only for a toggle hand. */
export function recoveryGates(cell: CellOut, readiness: ReadinessOut | null): Gate[] {
  const record = cell.recovery
  const at = record?.at ?? Number.POSITIVE_INFINITY
  const gates: Gate[] = [
    { id: 'controller', ok: controllerCanMove(readiness) },
    { id: 'cleared', ok: record != null && record.cleared_at != null && record.cleared_at > record.at },
  ]
  if (isToggle(cell)) {
    gates.push({ id: 'jaws', ok: cell.jaws_confirmed_at != null && cell.jaws_confirmed_at > at && cell.hand?.jaws === 'open' })
  }
  gates.push({ id: 'empty', ok: !partHeld(cell, readiness) })
  gates.push({ id: 'idle', ok: cell.state === 'connected' && !cell.active_run_id && !cell.jaws_question })
  return gates
}

export function allGreen(gates: readonly Gate[]): boolean {
  return gates.every((gate) => gate.ok)
}

/**
 * Why Home is refused now, as the server would refuse it (build plan 1.3.6), or `null` where it may be asked: the
 * button stays off and says why. The server enforces every one of these itself.
 */
export function homeRefusal(cell: CellOut | null, readiness: ReadinessOut | null, running: boolean): RefusalCode | null {
  // `running` is the cockpit's word: the poll's active run, unless the stream already saw that run end.
  if (!cell || cell.state !== 'connected') return 'not_connected'
  if (running) return 'run_active'
  if (cell.halted) return 'halted'
  if (controllerStopped(readiness)) return 'controller_stopped'
  const record = cell.recovery
  if (record && !(record.cleared_at != null && record.cleared_at > record.at)) return 'cell_not_cleared'
  if (cell.needs_person) return 'needs_person'
  if (cell.jaws_question) return 'jaws_question_pending'
  if (partHeld(cell, readiness)) return 'part_still_held'
  if (isToggle(cell)) {
    if (cell.hand?.jaws === 'unknown') return 'jaws_not_confirmed'
    if (record && !(cell.jaws_confirmed_at != null && cell.jaws_confirmed_at > record.at)) return 'jaws_not_confirmed'
  }
  return null
}
