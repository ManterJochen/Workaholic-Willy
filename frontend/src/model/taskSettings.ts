/**
 * The cell's task settings, shared: what every task the console is given takes where its sentence says nothing.
 *
 * Whether Enter starts a clean reading at once, the mode, the place, the tick for "anything", how each pick looks and
 * whether the parts are critical are the cell's, for every browser (the owner, 2026-10-16: `runtime.task` in
 * `app/runtime.yaml`, no longer `localStorage`). The console reads them from `GET /v1/task/settings`; the Settings page
 * ("Auftrag") writes them through `PATCH /v1/task/settings` into the cell's own layer (2026-10-17), the way a taught
 * pose is written, and every screen of this tab sees the answer at once. Another browser reads them when it opens.
 *
 * * `settings` is `null` until the server answered. **Enter then starts nothing**: it only reads, and the card waits
 *   for Start, as "Erst die Karte" does; no default of the browser's stands in for the cell's word. A read that failed
 *   is asked again on the next tick of the cell poll, until it answers; a server that has no such route (`404`, `501`)
 *   is not asked again.
 * * `change(values)` writes the settings named, all or none, and the server's answer becomes `settings`. A refusal
 *   (`run_active`, `no_layer`, `invalid_value`, `unknown_pose`) comes back as the typed `ApiError`, and the settings
 *   stay as they were.
 *
 * Nothing here moves anything or starts a task: the cockpit fills a card from these, and `POST /v1/task` alone starts
 * one. Written without JSX (`.ts`) so the provider and its hook can share one module (see `i18n/index.ts`).
 */

import { createContext, createElement, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'

import { ApiError, api, type TaskSettingsIn, type TaskSettingsOut } from '../api/client'
import { useCell } from './useCell'

/** The cell's task settings, as the server answers them (`runtime.task`, with where a change is written). */
export type TaskSettings = TaskSettingsOut

/** Where a task's part goes when its sentence names no place: the cell's default place, a taught pose, a camera's target. */
export type TaskPlace = TaskSettings['place']

/**
 * How each pick of a task looks, the console's one choice (the owner, 2026-10-08 night): the first look only
 * (`TaskOptionsIn.multi_view` off), the looks when needed (on: until a grasp is safe, and where the cell turns its
 * trigger on, a weak look is not), or every look (`every_look`: the early stop off).
 */
export type LooksChoice = TaskSettings['looks']
export const LOOKS_CHOICES: readonly LooksChoice[] = ['first', 'when_needed', 'every']

export interface TaskSettingsModel {
  /** The cell's settings; `null` until the server answered. */
  readonly settings: TaskSettings | null
  /** The last read failed: `settings` is what was read before it, if anything. */
  readonly error: ApiError | null
  /** Write the settings named into the cell's own layer, all or none; the refusal, where there is one, comes back. */
  readonly change: (values: TaskSettingsIn) => Promise<ApiError | null>
}

const INERT: TaskSettingsModel = { settings: null, error: null, change: async () => null }

const TaskSettingsContext = createContext<TaskSettingsModel | null>(null)

function asApiError(err: unknown): ApiError {
  return err instanceof ApiError ? err : new ApiError(0, null, String(err))
}

/** A refusal that means "this server has no task settings": asking again on every tick would only fill its log. */
function notThere(error: ApiError): boolean {
  return error.httpStatus === 404 || error.httpStatus === 501
}

/** The cell's task settings for every screen under it. Inside `CellProvider`: a failed read is asked again on its tick. */
export function TaskSettingsProvider({ children }: { children?: ReactNode }) {
  const { loadedAt } = useCell()
  const [settings, setSettings] = useState<TaskSettings | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  // Read only in effects and callbacks, never while rendering.
  const alive = useRef(true)
  const inFlight = useRef(false)
  /** The server said it has no task settings: not asked again. */
  const absent = useRef(false)

  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])

  // Read once, and again on the cell poll's next answer while no read has answered: a console opened before its server
  // came up reads the cell's settings as soon as the server is there.
  useEffect(() => {
    if (settings !== null || absent.current || inFlight.current) return
    inFlight.current = true
    api
      .taskSettings()
      .then((answer) => {
        if (!alive.current) return
        setSettings(answer)
        setError(null)
      })
      .catch((err: unknown) => {
        if (!alive.current) return
        const refused = asApiError(err)
        setError(refused)
        absent.current = notThere(refused)
      })
      .finally(() => {
        inFlight.current = false
      })
  }, [settings, loadedAt])

  // An answer that arrives after the console closed sets nothing: React drops a state change of an unmounted provider.
  const change = useCallback(async (values: TaskSettingsIn): Promise<ApiError | null> => {
    try {
      const answer = await api.setTaskSettings(values)
      setSettings(answer)
      setError(null)
      return null
    } catch (err: unknown) {
      return asApiError(err)
    }
  }, [])

  const value = useMemo<TaskSettingsModel>(() => ({ settings, error, change }), [settings, error, change])
  return createElement(TaskSettingsContext.Provider, { value }, children)
}

/** The cell's task settings. Without a provider: none read, and a change that does nothing (a bare component test). */
export function useTaskSettings(): TaskSettingsModel {
  return useContext(TaskSettingsContext) ?? INERT
}
