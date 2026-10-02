/**
 * Small readers the cockpit shares: a clock that ticks only while something is waiting for it, the taught poses
 * (`GET /v1/poses`), read again whenever the cell or its poses may have changed, the record of a stopped run, the
 * window's room under the top bar, the session's earlier tasks for the chat, and a card's refusal brought into view.
 * None of them moves anything.
 */

import { useEffect, useLayoutEffect, useRef, useState, type RefObject } from 'react'

import { api, type PosesOut, type RunOut } from '../api/client'
import { seedRuns } from '../model/chat'

/** Wall-clock seconds, refreshed every `intervalMs` while `active`: a countdown, an age, a halt waiting to be confirmed. */
export function useNow(intervalMs: number, active = true): number {
  const [now, setNow] = useState(() => Date.now() / 1000)
  useEffect(() => {
    if (!active) return
    const id = setInterval(() => setNow(Date.now() / 1000), intervalMs)
    return () => clearInterval(id)
  }, [intervalMs, active])
  return now
}

/**
 * The poses a task can name: Home, the taught ones by label, and the default place. `null` while unread or where the
 * server answers none (an older server, no cell configured): the card then offers the default place and the camera.
 */
export function usePoses(key: string): PosesOut | null {
  const [read, setRead] = useState<{ key: string; poses: PosesOut | null } | null>(null)
  useEffect(() => {
    let cancelled = false
    api
      .poses()
      .then((poses) => {
        if (!cancelled) setRead({ key, poses })
      })
      .catch(() => {
        if (!cancelled) setRead({ key, poses: null })
      })
    return () => {
      cancelled = true
    }
  }, [key])
  // What was read for an earlier key stays shown until the new answer is in: the card's choices never blink empty.
  return read?.poses ?? null
}

/**
 * The room the cockpit has in the window, as CSS custom properties on its element: `--ck-fit`, the height from where
 * it begins to the window's foot (less the room the page keeps below it), so it fills the window and never more: the
 * stop strip always ends on screen, and a tall window's height goes to the chat; and `--ck-bar`, the height of the
 * shell's sticky top bar, under which the stop buttons stick in the stacked layout. Measured, because the top bar
 * wraps onto more rows in a narrower window. Set on the element itself: a resize redraws nothing.
 */
export function useWindowFit(ref: RefObject<HTMLElement | null>): void {
  useLayoutEffect(() => {
    const box = ref.current
    if (!box || typeof window === 'undefined') return
    const bar = document.querySelector<HTMLElement>('.topbar')
    const measure = () => {
      const style = getComputedStyle(box)
      const parent = box.parentElement ? getComputedStyle(box.parentElement) : null
      const top = box.getBoundingClientRect().top + window.scrollY
      const below = (parseFloat(parent?.paddingBottom ?? '') || 0) + (parseFloat(style.marginBottom) || 0)
      box.style.setProperty('--ck-fit', `${Math.max(0, Math.floor(window.innerHeight - top - below))}px`)
      box.style.setProperty('--ck-bar', `${bar ? Math.ceil(bar.getBoundingClientRect().height) : 0}px`)
    }
    measure()
    window.addEventListener('resize', measure)
    const Observer = typeof ResizeObserver === 'undefined' ? null : ResizeObserver
    const watch = Observer && bar ? new Observer(measure) : null
    if (watch && bar) watch.observe(bar)
    return () => {
      window.removeEventListener('resize', measure)
      watch?.disconnect()
    }
  }, [ref])
}

/**
 * A card's refusal, brought into view the moment it is said: a click on Start, Restart or Home whose refusal lands
 * under the chat's visible edge changes nothing a person can see, and the click reads as ignored. The refusal is drawn
 * right above the buttons it refuses, and this scrolls the chat just far enough to show it (`block: 'nearest'`). Each
 * new refusal is a new answer, so it is brought into view again.
 */
export function useBroughtIntoView<T extends HTMLElement>(said: unknown, saidToo: unknown = null): RefObject<T | null> {
  const ref = useRef<T | null>(null)
  useEffect(() => {
    if (said || saidToo) ref.current?.scrollIntoView?.({ block: 'nearest', behavior: 'smooth' })
  }, [said, saidToo])
  return ref
}

/** How many of the session's runs the chat reads for its earlier tasks: the registry's whole window. */
export const EARLIER_RUNS = 200

/**
 * The session's earlier tasks, each as its one summary line in the conversation (`chat.seedRuns`), read from the
 * server once the cockpit opens: a second tab, or the browser opened again, shows what the session did instead of an
 * empty chat, because the chat is the history (OD 2). A server that cannot answer leaves the chat as it is.
 */
export function useEarlierTasks(): void {
  useEffect(() => {
    let cancelled = false
    api
      .runs(EARLIER_RUNS)
      .then((runs) => {
        if (!cancelled) seedRuns(runs)
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [])
}

/** How long the stop card waits before it asks again for a record it could not read. */
export const RECORD_RETRY_MS = 1500

/** The record of a stopped run, as far as it is read: `run` once read; `failed` while the last read was refused. */
export interface RunRecord {
  readonly run: RunOut | null
  readonly failed: boolean
}

/**
 * The record of the run a stop names (`GET /v1/runs/{id}`): the stop card's plan, return pose and the backend's
 * sentence. Taken from the run on screen where that is the same run. A read that fails is asked again every
 * `RECORD_RETRY_MS` until it answers: Restart must name the stopped run's own return pose, never a guessed one, so the
 * card keeps Restart off while the record is not read, and says why.
 */
export function useRunRecord(runId: string | null, known: RunOut | null): RunRecord {
  const [read, setRead] = useState<{ id: string; run: RunOut | null; failed: boolean } | null>(null)
  const [attempt, setAttempt] = useState(0)
  const knownId = known?.id ?? null
  const have = read !== null && read.id === runId && read.run !== null
  useEffect(() => {
    if (!runId || knownId === runId || have) return
    let cancelled = false
    let again: ReturnType<typeof setTimeout> | null = null
    api
      .run(runId)
      .then((run) => {
        if (!cancelled) setRead({ id: runId, run, failed: false })
      })
      .catch(() => {
        if (cancelled) return
        setRead({ id: runId, run: null, failed: true })
        again = setTimeout(() => setAttempt((n) => n + 1), RECORD_RETRY_MS)
      })
    return () => {
      cancelled = true
      if (again) clearTimeout(again)
    }
  }, [runId, knownId, have, attempt])
  if (!runId) return { run: null, failed: false }
  if (known && knownId === runId) return { run: known, failed: false }
  if (read?.id !== runId) return { run: null, failed: false }
  return { run: read.run, failed: read.failed }
}
