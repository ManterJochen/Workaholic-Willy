/**
 * One path for every prompt, whichever way it was entered.
 *
 * Before this, two screens each kept their own `useState('')` and called `api.pick` directly. That is
 * two copies of the same three decisions -- when may this start, what does the operator see while it
 * is starting, what happened to the last one -- and copies drift. Adding a microphone to one of them
 * would have made it three.
 *
 * ⭑ WHAT THE HISTORY IS FOR, and it is not nostalgia. A spoken prompt is a TRANSCRIPTION: the arm
 * moves on what Whisper heard, which is not necessarily what the operator said. When a run goes
 * wrong, "the operator asked for the wrong thing" and "the machine heard the wrong thing" are
 * different faults and they look identical afterwards -- unless something recorded which of the two
 * routes the text came in by. That is what `source` is, and it is the reason this file exists
 * rather than just a shared component.
 *
 * ⚠ IN-MEMORY, AND DELIBERATELY. It is a session aid, not a record: the durable one is the backend's
 * `GraspAttemptRecord` log, which is the thing a KPI or an audit may read. A second store that
 * survived a reload would look authoritative while carrying only what one browser tab happened to
 * see, and a half-record that looks whole is worse than none.
 */

import { useCallback, useState, useSyncExternalStore } from 'react'

import { ApiError, api, type RunOut } from '../api/client'

/** How the text reached the box. `spoken` means a machine transcribed it. */
export type PromptSource = 'typed' | 'spoken'

export interface PromptEntry {
  readonly id: number
  readonly text: string
  readonly source: PromptSource
  readonly at: number
  /** The run it started, once the backend accepted it. Null while starting, or if it never did. */
  readonly run: RunOut | null
  /** Why it did not start. Null when it did. */
  readonly error: ApiError | null
}

type Listener = (entries: PromptEntry[]) => void

let entries: PromptEntry[] = []
let nextId = 1
const listeners = new Set<Listener>()

/** Cap: this is a session aid, and an unbounded array in a console left open for a shift is a leak. */
const KEEP = 50

function publish(): void {
  const snapshot = entries
  listeners.forEach((listener) => listener(snapshot))
}

/** Record that a prompt was submitted. Returns its id so the outcome can be attached. */
export function submitted(text: string, source: PromptSource): number {
  const id = nextId++
  entries = [{ id, text, source, at: Date.now(), run: null, error: null }, ...entries].slice(0, KEEP)
  publish()
  return id
}

/** Attach what happened to a submitted prompt. */
export function settled(id: number, outcome: { run?: RunOut; error?: ApiError }): void {
  entries = entries.map((entry) =>
    entry.id === id ? { ...entry, run: outcome.run ?? null, error: outcome.error ?? null } : entry,
  )
  publish()
}

/** Test seam: the store is module state, so a test that does not reset it inherits the last one. */
export function resetPromptHistory(): void {
  entries = []
  nextId = 1
  publish()
}

/**
 * ⭑ `useSyncExternalStore`, NOT a subscribe-and-setState effect, and the difference is not stylistic.
 * The effect version has a gap: an entry can land between the first render and the effect that
 * subscribes, so the component shows a snapshot that is already stale and only corrects itself on the
 * NEXT mutation. Re-reading inside the effect closes the gap by starting a second render, which is
 * what the linter objects to and what React added this hook to replace. `entries` is replaced rather
 * than mutated, so the snapshot reference is stable between changes -- which is exactly what this
 * hook requires and what makes it safe here.
 */
export function usePromptHistory(): PromptEntry[] {
  return useSyncExternalStore(
    (onChange) => {
      const listener: Listener = () => onChange()
      listeners.add(listener)
      return () => {
        listeners.delete(listener)
      }
    },
    () => entries,
    () => entries,
  )
}

export interface PromptPipeline {
  text: string
  setText: (text: string, source?: PromptSource) => void
  /** How the CURRENT text got there. Reset to `typed` the moment the operator edits it. */
  source: PromptSource
  starting: boolean
  error: ApiError | null
  /** Start a run with the current text. No-op while one is starting. */
  start: (picks: number) => Promise<RunOut | null>
  clearError: () => void
}

/**
 * The prompt, its provenance, and the one call that starts a run with it.
 *
 * `onStarted` rather than a returned run only: the two screens do different things with it (one
 * narrates from the event stream, the other refreshes its KPIs), and a hook that tried to do both
 * would need to know which screen it is in.
 */
export function usePromptPipeline(onStarted?: (run: RunOut) => void): PromptPipeline {
  const [text, setTextRaw] = useState('')
  const [source, setSource] = useState<PromptSource>('typed')
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<ApiError | null>(null)

  const setText = useCallback((next: string, nextSource: PromptSource = 'typed') => {
    setTextRaw(next)
    // ⚠ EDITING A TRANSCRIPTION MAKES IT TYPED AGAIN, and that is the honest answer rather than a
    // convenience. Once a human has corrected what the machine heard, calling the result "spoken"
    // would blame Whisper for a word the operator chose -- which is exactly the confusion the
    // history exists to prevent.
    setSource(nextSource)
  }, [])

  const start = useCallback(
    async (picks: number): Promise<RunOut | null> => {
      if (starting) return null
      setStarting(true)
      setError(null)
      const id = submitted(text, source)
      try {
        const run = await api.pick(text, picks)
        settled(id, { run })
        onStarted?.(run)
        return run
      } catch (err: unknown) {
        const failure = err instanceof ApiError ? err : new ApiError(0, null, String(err))
        settled(id, { error: failure })
        setError(failure)
        return null
      } finally {
        setStarting(false)
      }
    },
    [text, source, starting, onStarted],
  )

  return { text, setText, source, starting, error, start, clearError: () => setError(null) }
}
