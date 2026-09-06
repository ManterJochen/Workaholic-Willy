/**
 * Loading a value from the cell, with the three states a console must be able to draw.
 *
 * A screen that models a request as "value or null" cannot tell "not asked yet" from "asked and the
 * cell said nothing", and those mean opposite things during a bring-up. So every load carries an
 * explicit `loading` and a typed `error`, and no screen renders a value while an error is standing.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiError } from '../api/client'

export interface AsyncState<T> {
  data: T | null
  error: ApiError | null
  loading: boolean
  /** Never null after the first successful load: lets a screen show stale data as stale, not as gone. */
  loadedAt: number | null
  reload: () => void
}

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []): AsyncState<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadedAt, setLoadedAt] = useState<number | null>(null)
  const [tick, setTick] = useState(0)
  // Guards against a slow first request landing after a fast second one and overwriting it.
  const generation = useRef(0)

  useEffect(() => {
    const mine = ++generation.current
    setLoading(true)
    fn()
      .then((value) => {
        if (generation.current !== mine) return
        setData(value)
        setError(null)
        setLoadedAt(Date.now())
      })
      .catch((err: unknown) => {
        if (generation.current !== mine) return
        setError(err instanceof ApiError ? err : new ApiError(0, null, String(err)))
      })
      .finally(() => {
        if (generation.current === mine) setLoading(false)
      })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick])

  const reload = useCallback(() => setTick((t) => t + 1), [])
  return { data, error, loading, loadedAt, reload }
}

/**
 * Re-run something on a timer, e.g. the telemetry tick.
 *
 * `enabled` rather than conditional mounting because a poll that stops when the cell disconnects must
 * resume when it comes back, without the screen losing what it already showed.
 */
export function usePoll(fn: () => void, intervalMs: number, enabled: boolean): void {
  const saved = useRef(fn)
  // Written in an effect, not during render: a ref mutated while rendering is read by React's own
  // rules as a value that should have been state, and in Strict Mode's double render it is written
  // twice before anything can use it.
  useEffect(() => {
    saved.current = fn
  }, [fn])
  useEffect(() => {
    if (!enabled) return
    const id = setInterval(() => saved.current(), intervalMs)
    return () => clearInterval(id)
  }, [intervalMs, enabled])
}
