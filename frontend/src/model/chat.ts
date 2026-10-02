/**
 * The conversation with Willy, kept for the tab's session.
 *
 * What a person said (typed or spoken, word for word) and what Willy answered, in `sessionStorage['willy.chat']`:
 * it survives a reload of the tab, which is the moment an operator most needs to see what they asked for, and it ends
 * with the tab, so a console left open overnight starts the next morning with an empty page rather than yesterday's
 * orders. At most the last 200 entries are kept.
 *
 * The run in progress is not stored here: its lines are rebuilt by replaying its events (`runModel`). What IS stored
 * of a run is one summary line once it has ended ("Räum alle grünen Würfel … → Nichts mehr da"), which is how earlier
 * tasks read in the history. Willy's lines are message keys, so a language switch re-renders the whole conversation.
 *
 * The server keeps the session's runs, so a tab that did not see them (a second tab, the browser opened again) reads
 * the earlier tasks from it (`seedRuns`, `GET /v1/runs`): each as its one line, in time order among the tab's own.
 */

import { useSyncExternalStore } from 'react'

import type { RunOut } from '../api/client'
import { STOP_CLASS_OF, isStopCode, type StopClass } from '../api/codes'
import { runKindMsg, stopMsg } from '../i18n/codes'
import type { Msg } from '../i18n/types'
import type { Tone } from './runModel'

export const CONVERSATION_KEY = 'willy.chat'
export const CONVERSATION_LIMIT = 200

export type Speaker = 'operator' | 'willy'

export interface ConversationEntry {
  readonly id: string
  /** Unix seconds. */
  readonly at: number
  readonly who: Speaker
  /** `command`: the operator's words; `reply`: Willy's answer; `summary`: an earlier task in one line. */
  readonly kind: 'command' | 'reply' | 'summary'
  /** The operator's own words, never translated. */
  readonly text?: string
  readonly source?: 'typed' | 'spoken'
  /** Willy's line, as a message key. */
  readonly msg?: Msg<string>
  readonly tone?: Tone
  readonly runId?: string
}

export type NewEntry = Omit<ConversationEntry, 'id' | 'at'> & { readonly id?: string; readonly at?: number }

const SPEAKERS: readonly string[] = ['operator', 'willy']
const KINDS: readonly string[] = ['command', 'reply', 'summary']
const SOURCES: readonly string[] = ['typed', 'spoken']
const TONES: readonly string[] = ['info', 'ok', 'warn', 'error']

let entries: readonly ConversationEntry[] | null = null
let counter = 0
const listeners = new Set<() => void>()

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Absent, or a string from `allowed` (any string when no list is given). */
function optional(value: unknown, allowed?: readonly string[]): boolean {
  return value === undefined || (typeof value === 'string' && (!allowed || allowed.includes(value)))
}

/** A message the translator can draw: a string key, and parameters that are an object when present. */
function isMessage(value: unknown): boolean {
  return isRecord(value) && typeof value.key === 'string' && (value.params === undefined || isRecord(value.params))
}

/**
 * An entry the chat can draw. Storage is read back after a reload and may hold an older console's format, or a
 * hand-edited session: an entry with a line that is not a message, or words that are not a string, is dropped here
 * rather than reaching a render that would throw on it.
 */
function valid(value: unknown): value is ConversationEntry {
  if (!isRecord(value)) return false
  const e = value
  return (
    typeof e.id === 'string' &&
    typeof e.at === 'number' &&
    Number.isFinite(e.at) &&
    typeof e.who === 'string' &&
    SPEAKERS.includes(e.who) &&
    typeof e.kind === 'string' &&
    KINDS.includes(e.kind) &&
    optional(e.text) &&
    optional(e.source, SOURCES) &&
    optional(e.tone, TONES) &&
    optional(e.runId) &&
    (e.msg === undefined || isMessage(e.msg))
  )
}

function read(): readonly ConversationEntry[] {
  try {
    const raw = sessionStorage.getItem(CONVERSATION_KEY)
    if (!raw) return []
    const parsed: unknown = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed.filter(valid).slice(-CONVERSATION_LIMIT) : []
  } catch {
    // Nonsense in storage, or no storage at all (a locked-down kiosk): an empty conversation, never a crash.
    return []
  }
}

function write(next: readonly ConversationEntry[]): void {
  try {
    sessionStorage.setItem(CONVERSATION_KEY, JSON.stringify(next))
  } catch {
    /* the conversation still works for this page; it only does not survive a reload */
  }
}

function snapshot(): readonly ConversationEntry[] {
  if (entries === null) entries = read()
  return entries
}

function publish(next: readonly ConversationEntry[]): void {
  entries = next
  write(next)
  for (const listener of listeners) listener()
}

/** The conversation as stored now. */
export function loadConversation(): readonly ConversationEntry[] {
  return snapshot()
}

/** Add one entry (the operator's command, Willy's reply) and keep the last 200. */
export function say(entry: NewEntry): ConversationEntry {
  counter += 1
  const full: ConversationEntry = {
    ...entry,
    id: entry.id ?? `c-${Date.now().toString(36)}-${counter}`,
    at: entry.at ?? Date.now() / 1000,
  }
  publish([...snapshot(), full].slice(-CONVERSATION_LIMIT))
  return full
}

const TONE: Record<StopClass, Tone> = {
  done: 'ok',
  operator: 'info',
  ask: 'warn',
  teach: 'warn',
  planner: 'warn',
  problem: 'error',
}

/**
 * An ended run in one line: the command as it was said (or what was picked), how many parts a task placed, and how it
 * ended ("Räum die gelbe Kiste aus · 12 Teile · Nichts mehr da"). `null` for a run that has not ended, which has no
 * summary yet. The task's line is the cockpit's (`ck.summary.task`): the cockpit is where the conversation is drawn.
 */
export function summaryOf(run: RunOut): ConversationEntry | null {
  if (!isStopCode(run.stop_code)) return null
  const command: string | Msg =
    run.plan?.command?.text || run.plan?.object_said || run.plan?.object || run.prompt || runKindMsg(run.kind ?? 'pick')
  const title = stopMsg(run.stop_code)
  return {
    id: `summary:${run.id}`,
    at: run.finished_at ?? run.started_at,
    who: 'willy',
    kind: 'summary',
    msg:
      run.kind === 'task'
        ? { key: 'ck.summary.task', params: { command, parts: run.parts_placed ?? 0, title } }
        : { key: 'chat.summary', params: { command, title } },
    tone: TONE[STOP_CLASS_OF[run.stop_code]],
    runId: run.id,
  }
}

/** Keep an ended run's summary in the conversation, once per run. Returns whether it was added now. */
export function archiveRun(run: RunOut): boolean {
  const entry = summaryOf(run)
  if (!entry || snapshot().some((e) => e.id === entry.id)) return false
  publish([...snapshot(), entry].slice(-CONVERSATION_LIMIT))
  return true
}

/**
 * The session's earlier tasks as the server lists them (`GET /v1/runs`, newest first), each ended task as its one
 * summary line, so a tab that did not see them still reads the session's history. A line the conversation already
 * holds is not added twice; the lines take their places by time among the tab's own, and the newest 200 are kept.
 * Only tasks: a Home move, a teach or a planner start is no earlier task. Returns how many lines were added.
 */
export function seedRuns(runs: readonly RunOut[]): number {
  const kept = new Set(snapshot().map((e) => e.id))
  const added: ConversationEntry[] = []
  for (const run of runs) {
    if (run.kind !== 'task') continue
    const entry = summaryOf(run)
    if (!entry || kept.has(entry.id)) continue
    kept.add(entry.id)
    added.push(entry)
  }
  if (added.length === 0) return 0
  const merged = [...snapshot(), ...added]
    .map((entry, index) => ({ entry, index }))
    .sort((a, b) => a.entry.at - b.entry.at || a.index - b.index)
    .map(({ entry }) => entry)
  publish(merged.slice(-CONVERSATION_LIMIT))
  return added.length
}

/**
 * Forget the conversation. `keepStorage` only drops what this page holds in memory, so the next read comes from
 * storage again (a reload, as a test sees it).
 */
export function clearConversation(options: { keepStorage?: boolean } = {}): void {
  if (options.keepStorage) {
    entries = null
    return
  }
  try {
    sessionStorage.removeItem(CONVERSATION_KEY)
  } catch {
    /* see write() */
  }
  entries = []
  for (const listener of listeners) listener()
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

/** The conversation, re-rendering on every new entry. */
export function useConversation(): readonly ConversationEntry[] {
  return useSyncExternalStore(subscribe, snapshot, snapshot)
}
