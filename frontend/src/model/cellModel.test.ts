/**
 * The cell's own stream (`run_id=cell`): the jaws question, a brake with no run, the recovery record.
 *
 * The poll of `GET /v1/cell` stays the authority on each of these; the stream is what makes the jaws dialog open the
 * moment a question is asked rather than up to two seconds later. These tests replay the contract's logs and assert
 * the state a page opened at any point would show, because a page opened late replays the cell stream from seq 0.
 */

import { describe, expect, it } from 'vitest'

import type { RunEvent } from '../api/events'
import { COCKPIT } from '../cockpit/i18n'
import { translate } from '../i18n'
import backenLeerJson from '../test/fixtures/console_dummy_halted_backen_leer.json'
import haltedRestartJson from '../test/fixtures/task_halted_then_restart.json'
import jawsJson from '../test/fixtures/jaws_question_round_trip.json'
import { runOf, type Fixture } from '../test/render'
import { CELL_RESET, EMPTY_CELL_STREAM, pendingQuestion, reduceCell, replayCell } from './cellModel'

const jaws = (jawsJson as unknown as Fixture).events.filter((e) => e.run_id === 'cell')
const haltedLog = haltedRestartJson as unknown as Fixture
const halted = haltedLog.events.filter((e) => e.run_id === 'cell')
// Chosen by what they are, never by their ids (the capture names them; a regeneration may not).
const STOPPED = runOf(haltedLog, (record) => record.stop_class === 'problem')
const RESTART = runOf(haltedLog, (record) => Boolean(record.restart_of))

describe('the jaws question round trip', () => {
  it('opens the question the moment it is asked, with its choices and no default', () => {
    const view = replayCell(jaws.slice(0, 1))
    expect(view.question).toMatchObject({ id: 'q-5d1e0c', stage: 'where', at: 'connect', where: 'tool output 0' })
    expect(view.question?.choices).toEqual(['open', 'closed'])
    // The captured log's fixed clock (scripts/console/capture_event_log.py): asked at 1790856000.0, 120 s to answer.
    expect(view.question?.expiresAt).toBe(1790856120)
  })

  it('closes a question once it is answered, and opens the next stage as its own question', () => {
    const answered = replayCell(jaws.slice(0, 2))
    expect(answered.question).toBeNull()
    expect(answered.lastAnswer).toMatchObject({ id: 'q-5d1e0c', choice: 'closed' })
    const openNow = replayCell(jaws.slice(0, 3))
    expect(openNow.question).toMatchObject({ id: 'q-8a2071', stage: 'open_now' })
    expect(openNow.question?.choices).toEqual(['open_now', 'abort'])
  })

  it('ends open, with the planner told the hand is empty', () => {
    const view = replayCell(jaws)
    expect(view.question).toBeNull()
    expect(view.lastEnded).toMatchObject({ id: 'q-8a2071', outcome: 'open', detached: true })
    expect(view.lines.map((l) => l.msg.key)).toEqual([
      'event.cell.jaws_question',
      'event.cell.jaws_answered',
      'event.cell.jaws_question.open_now',
      'event.cell.jaws_answered',
      'event.cell.jaws_ended.open',
    ])
  })

  it('never reads a question that ended without an answer as open', () => {
    const view = reduceCell(replayCell(jaws.slice(0, 1)), {
      ...jaws[4],
      seq: 2,
      data: { question_id: 'q-5d1e0c', outcome: 'no_answer', refusal: 'nobody answered', detached: false },
    })
    expect(view.question).toBeNull()
    expect(view.lastEnded?.outcome).toBe('no_answer')
    expect(view.lines.at(-1)?.msg.key).toBe('event.cell.jaws_ended.no_answer')
  })

  it('stops offering a question once it has expired, even if the stream missed its end', () => {
    const view = replayCell(jaws.slice(0, 1))
    expect(pendingQuestion(view, 1790856100)).not.toBeNull()
    expect(pendingQuestion(view, 1790856130)).toBeNull()
  })
})

describe('the recovery record on the cell stream', () => {
  it('stands from the problem stop, is cleared by a person, and ends with the arrival at the return pose', () => {
    const stopped = replayCell(halted.slice(0, 1))
    expect(stopped.recovery).toMatchObject({ runId: STOPPED, stopCode: 'halted', acknowledgedAt: null })
    const cleared = replayCell(halted.slice(0, 2))
    expect(cleared.acknowledged?.cleared).toEqual(['halted', 'recovery'])
    expect(cleared.recovery?.acknowledgedAt).not.toBeNull()
    const ended = replayCell(halted)
    expect(ended.recovery).toBeNull()
    expect(ended.recoveryEnded).toMatchObject({ runId: STOPPED, by: 'restart', endedBy: RESTART })
  })

  it('starts over on a reset, so a stream replayed from seq 0 after a reconnect is read again', () => {
    const before = replayCell(halted)
    const again = replayCell(halted.slice(0, 1), reduceCell(before, CELL_RESET))
    expect(again.recovery).toMatchObject({ runId: STOPPED })
    expect(again.lines).toHaveLength(1)
  })

  it('ignores the events of runs, and an event it already applied', () => {
    const view = replayCell([...halted, ...halted])
    expect(view.lines).toHaveLength(halted.length)
    const run: RunEvent = { ...halted[0], run_id: STOPPED, type: 'run_started' }
    expect(reduceCell(EMPTY_CELL_STREAM, run)).toBe(EMPTY_CELL_STREAM)
  })

  it('files its lines under no part, with the event\'s data for the tech view', () => {
    const view = replayCell(halted)
    expect(view.lines.every((line) => line.part === null)).toBe(true)
    expect(view.lines[0].data).toMatchObject({ run_id: STOPPED, stop_code: 'halted' })
  })
})

describe('what a person confirmed', () => {
  const backenLeer = (backenLeerJson as unknown as Fixture).events.filter((e) => e.run_id === 'cell')
  const say = (key: string, lang: 'de' | 'en' = 'de') => translate(lang, key as never, undefined, COCKPIT)

  it('says "Zelle ist frei" once, then that the hand is empty after "Backen leer", never the clear cell twice', () => {
    const view = replayCell(backenLeer)
    const keys = view.lines.map((line) => line.msg.key)
    expect(keys.filter((key) => key === 'event.cell.acknowledged')).toHaveLength(1)
    const empty = view.lines.find((line) => line.type === 'cell.acknowledged' && line.data.jaws_emptied === true)
    expect(empty?.msg.key).toBe('ck.cell.jawsEmptied')
    expect(say(empty!.msg.key)).toBe('Bestätigt: kein Teil in der Hand.')
    expect(say(empty!.msg.key, 'en')).toBe('Confirmed: no part in the hand.')
    expect(empty?.level).toBe('demo')
  })

  it('says both where one confirmation clears the cell and empties the hand', () => {
    const stopped = replayCell(backenLeer.slice(0, 1))
    const both = reduceCell(stopped, { ...backenLeer[1], data: { cleared: ['halted', 'recovery'], jaws_emptied: true } })
    expect(both.lines.at(-1)?.msg.key).toBe('ck.cell.clearAndEmpty')
    expect(say('ck.cell.clearAndEmpty')).toBe('Bestätigt: die Zelle ist frei, und kein Teil ist in der Hand.')
  })
})
