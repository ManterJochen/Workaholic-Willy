/**
 * The conversation survives a reload of the tab, stays bounded, and sums up a finished task in one line.
 *
 * It lives in `sessionStorage` (one tab's session, cleared when the tab closes), at most the last 200 entries. The
 * operator's own words are kept as typed or spoken, never translated; Willy's lines are message keys, so a language
 * switch re-renders the whole history in the new language.
 */

import { act, cleanup, render, screen } from '@testing-library/react'
import { createElement } from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { COCKPIT } from '../cockpit/i18n'
import { translate } from '../i18n'
import haltedRestartJson from '../test/fixtures/task_halted_then_restart.json'
import twoPartsJson from '../test/fixtures/task_two_parts_nothing_left.json'
import { runOf, type Fixture } from '../test/render'
import {
  CONVERSATION_KEY,
  CONVERSATION_LIMIT,
  archiveRun,
  clearConversation,
  loadConversation,
  say,
  seedRuns,
  summaryOf,
  useConversation,
} from './chat'

// Chosen by what they are, never by their (randomly drawn) ids: the run track regenerates these logs.
const twoPartsLog = twoPartsJson as unknown as Fixture
const haltedLog = haltedRestartJson as unknown as Fixture
const twoParts = twoPartsLog.runs[runOf(twoPartsLog)]
const halted = haltedLog.runs[runOf(haltedLog, (record) => record.stop_class === 'problem')]

beforeEach(() => {
  clearConversation()
})

afterEach(() => {
  cleanup()
  clearConversation()
})

describe('the conversation', () => {
  it('keeps the operator\'s words as they were said, with where they came from', () => {
    say({ who: 'operator', kind: 'command', text: 'Räum alle grünen Würfel auf die Ablage links', source: 'spoken' })
    const [entry] = loadConversation()
    expect(entry).toMatchObject({ who: 'operator', kind: 'command', text: 'Räum alle grünen Würfel auf die Ablage links', source: 'spoken' })
    expect(entry.id).toBeTruthy()
  })

  it('is kept in this tab\'s session storage and read back after a reload', () => {
    say({ who: 'willy', kind: 'reply', msg: { key: 'chat.next' } })
    const stored = JSON.parse(sessionStorage.getItem(CONVERSATION_KEY) ?? '[]') as unknown[]
    expect(stored).toHaveLength(1)
    expect(loadConversation()[0].msg).toEqual({ key: 'chat.next' })
  })

  it('keeps only the last 200 entries', () => {
    for (let i = 0; i < CONVERSATION_LIMIT + 25; i += 1) say({ who: 'operator', kind: 'command', text: `line ${i}` })
    const entries = loadConversation()
    expect(entries).toHaveLength(CONVERSATION_LIMIT)
    expect(entries[0].text).toBe('line 25')
    expect(entries.at(-1)?.text).toBe(`line ${CONVERSATION_LIMIT + 24}`)
  })

  it('reads a storage that holds nonsense as an empty conversation, not as a crash', () => {
    sessionStorage.setItem(CONVERSATION_KEY, '{not json')
    clearConversation({ keepStorage: true })
    expect(loadConversation()).toEqual([])
    sessionStorage.setItem(CONVERSATION_KEY, JSON.stringify([{ who: 'nobody' }, 7]))
    clearConversation({ keepStorage: true })
    expect(loadConversation()).toEqual([])
  })

  it('drops a stored entry whose line is not a message, or whose words are not words, and keeps the good ones', () => {
    // An older console's format, or a hand-edited session: an entry the chat could not draw must never reach it.
    const good = { id: 'ok', at: 1, who: 'willy', kind: 'reply', msg: { key: 'chat.next' } }
    const bad = [
      { id: 'a', at: 1, who: 'willy', kind: 'reply', msg: {} },
      { id: 'b', at: 1, who: 'willy', kind: 'reply', msg: { key: 7 } },
      { id: 'c', at: 1, who: 'willy', kind: 'reply', msg: { key: 'chat.next', params: 'x' } },
      { id: 'd', at: 1, who: 'willy', kind: 'reply', msg: 'chat.next' },
      { id: 'e', at: 1, who: 'operator', kind: 'command', text: { words: 'nimm' } },
      { id: 'f', at: 1, who: 'operator', kind: 'command', text: 'nimm', source: 'shouted' },
      { id: 'g', at: Number.NaN, who: 'willy', kind: 'reply', msg: { key: 'chat.next' } },
    ]
    sessionStorage.setItem(CONVERSATION_KEY, JSON.stringify([good, ...bad, { ...good, id: 'ok2', runId: 'run-1', tone: 'ok' }]))
    clearConversation({ keepStorage: true })
    expect(loadConversation().map((e) => e.id)).toEqual(['ok', 'ok2'])
  })
})

describe('an earlier task, in one line', () => {
  it('is the command as it was said, how many parts the task placed, and how it ended', () => {
    const entry = summaryOf(twoParts)
    expect(entry).toMatchObject({ who: 'willy', kind: 'summary', runId: twoParts.id, tone: 'ok' })
    expect(entry?.msg).toEqual({
      key: 'ck.summary.task',
      params: { command: 'Räum alle grünen Würfel auf die Ablage links', parts: 2, title: { key: 'stop.nothing_left' } },
    })
    expect(translate('de', 'ck.summary.task' as never, entry?.msg?.params, COCKPIT)).toBe('Räum alle grünen Würfel auf die Ablage links · 2 Teile · Nichts mehr da')
  })

  it('is the run\'s kind and how it ended for a run that places no parts', () => {
    const home = { ...twoParts, kind: 'home' as const, plan: null, prompt: '', stop_code: 'finished' as const, stop_class: 'done' as const }
    expect(summaryOf(home)?.msg).toEqual({ key: 'chat.summary', params: { command: { key: 'runKind.home' }, title: { key: 'stop.finished' } } })
  })

  it('carries the problem tone for a problem stop', () => {
    expect(summaryOf(halted)?.tone).toBe('error')
  })

  it('is not made for a run that has not ended', () => {
    expect(summaryOf({ ...twoParts, state: 'running', stop_code: '', stop_class: '' })).toBeNull()
  })

  it('is archived once per run, however often the run is seen finishing', () => {
    expect(archiveRun(twoParts)).toBe(true)
    expect(archiveRun(twoParts)).toBe(false)
    expect(loadConversation().filter((e) => e.kind === 'summary')).toHaveLength(1)
  })
})

describe('the session\'s earlier tasks, read from the server', () => {
  const home = { ...twoParts, id: 'run-home', kind: 'home' as const, plan: null, stop_code: 'finished' as const, stop_class: 'done' as const }
  const running = { ...twoParts, id: 'run-now', state: 'running', stop_code: '' as const, stop_class: '' as const, finished_at: null }

  it('adds each ended task once, as its one line, and nothing else', () => {
    // The server lists the newest first; a Home move, a run still going and a task already kept add nothing.
    archiveRun(twoParts)
    expect(seedRuns([running, home, halted, twoParts])).toBe(1)
    const summaries = loadConversation().filter((e) => e.kind === 'summary')
    expect(summaries.map((e) => e.runId)).toEqual(
      [twoParts, halted].sort((a, b) => (a.finished_at ?? 0) - (b.finished_at ?? 0)).map((run) => run.id),
    )
    expect(seedRuns([twoParts, halted])).toBe(0)
  })

  it('puts each line in its place by time among the tab\'s own', () => {
    const before = (halted.finished_at ?? halted.started_at) - 10
    say({ who: 'operator', kind: 'command', text: 'früher', source: 'typed', at: before })
    say({ who: 'operator', kind: 'command', text: 'jetzt', source: 'typed', at: (halted.finished_at ?? halted.started_at) + 10 })
    seedRuns([halted])
    expect(loadConversation().map((e) => e.text ?? e.kind)).toEqual(['früher', 'summary', 'jetzt'])
  })
})

describe('useConversation', () => {
  it('re-renders when a line is said', () => {
    function Probe() {
      const entries = useConversation()
      return createElement('p', null, `${entries.length} entries`)
    }
    render(createElement(Probe))
    expect(screen.getByText('0 entries')).toBeTruthy()
    act(() => {
      say({ who: 'operator', kind: 'command', text: 'nimm den Würfel' })
    })
    expect(screen.getByText('1 entries')).toBeTruthy()
  })
})
