/**
 * The chat's one flow (`chatFlow`): the conversation, the run on screen and the cell's own lines, oldest first.
 *
 * Two sources can say something in the same instant: on a Windows clock the cell's "the arm is back at its return
 * pose" (the end of the stop record) and the Restart's first part are stamped alike. The return came first, and the
 * flow must read that way: "zurück", then "Teil 1", never the part card above the line that ended the stop.
 */

import { describe, expect, it } from 'vitest'

import type { RunEvent } from '../api/events'
import { replayCell } from '../model/cellModel'
import { replay } from '../model/runModel'
import backenLeerJson from '../test/fixtures/console_dummy_halted_backen_leer.json'
import { eventsOf, runOf, type Fixture } from '../test/render'
import { chatFlow, type FlowItem } from './chatFlow'

const backenLeer = backenLeerJson as unknown as Fixture

function what(item: FlowItem): string {
  if (item.kind === 'part') return `part ${item.part.part}`
  if (item.kind === 'line') return item.line.msg.key
  return `entry ${item.entry.kind}`
}

describe('the chat flow', () => {
  it('puts the cell\'s line before a part that began in the same instant: the arm came back, then part 1 began', () => {
    const restart = runOf(backenLeer, (record) => record.restart_of !== null && record.restart_of !== undefined)
    const ended = backenLeer.events.find((event) => event.type === 'cell.recovery_ended')
    if (!ended) throw new Error('the fixture ends no stop record')
    // The restart's first part is stamped in the very instant the stop record ended.
    const events: RunEvent[] = eventsOf(backenLeer, restart).map((event) => (event.type === 'task.part_started' ? { ...event, ts: ended.ts } : event))
    const view = replay(events)
    const cell = replayCell(backenLeer.events.filter((event) => event.run_id === 'cell'))
    const flow = chatFlow({ conversation: [], view, cellLines: cell.lines, since: 0, tech: false }).map(what)
    const back = flow.indexOf('event.cell.recovery_ended')
    const part = flow.indexOf('part 1')
    expect(back).toBeGreaterThanOrEqual(0)
    expect(part).toBeGreaterThan(back)
  })
})
