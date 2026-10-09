/**
 * The person's preferences: the demo or the tech view, voice output, the theme, and how a typed task starts.
 *
 * Each is per browser (`localStorage`), so the cell PC and the projector window remember their own:
 *
 * * `willy.view`: `demo` (big type, localised lines only, no JSON) or `tech` (the backend's sentences, the data,
 *   route badges, the raw event log, the Diagnostics drawer). Demo unless tech was chosen (OD 1). The view is also
 *   stamped on `<html data-view>`, so an area's stylesheet can size its type for it without a prop;
 * * `willy.voiceOut`: the browser's own speech synthesis at the key moments (OD 18); OFF unless switched on;
 * * the theme, owned by `lib/theme.ts` (`willy.theme`, dark by default); held here so the top bar's switch and the
 *   Settings page show the same state;
 * * the task (the owner, 2026-10-08: Enter starts at once, the mode is set beforehand, what and where come from the
 *   sentence, and these settings only where the sentence says nothing), under Settings, "Auftrag":
 *   * `willy.taskStart`: `enter` (the default: Enter reads the sentence and starts a clean reading at once, and the
 *     card opens for every other) or `card` (Enter only reads; Start on the card starts, as before);
 *   * `willy.taskScope`: `once` (the default) or `until_empty`, the mode of a task Enter starts;
 *   * `willy.taskPlace`: where the part goes when the sentence names no place: empty for the cell's default place,
 *     `pose:<name>` for a taught pose, `camera:<phrase>` for a target the camera finds;
 *   * `willy.taskAnything`: `anything` ticks "alles, was die Kamera sieht" on a card whose sentence names no part
 *     (Q7 A+, the tick given beforehand); `ask`, the default, leaves the tick to the person. Such a card never starts
 *     on Enter;
 *   * `willy.taskLooks`: how each pick of a task looks (the owner, 2026-10-08 night: "Multi-View" and "Alle Posen" one
 *     choice): `first` (the first look only, multi-view off), `when_needed` (the default: the looks until a grasp is
 *     safe, as always) or `every` (every look, the early stop off). Every card starts from it; the card may change it.
 *
 * Written without JSX (`.ts`) so the provider and its hook can share one module (see `i18n/index.ts`).
 */

import { createContext, createElement, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'

import { MAX_FIELD_CHARS } from '../api/limits'
import { readTheme, useTheme, type Theme } from '../lib/theme'

export type View = 'demo' | 'tech'

export const VIEW_KEY = 'willy.view'
export const VOICE_OUT_KEY = 'willy.voiceOut'
export const TASK_START_KEY = 'willy.taskStart'
export const TASK_SCOPE_KEY = 'willy.taskScope'
export const TASK_PLACE_KEY = 'willy.taskPlace'
export const TASK_ANYTHING_KEY = 'willy.taskAnything'
export const TASK_LOOKS_KEY = 'willy.taskLooks'

/** Where a task's part goes when the sentence names no place. */
export type TaskPlace =
  | { readonly kind: 'default' }
  | { readonly kind: 'pose'; readonly pose: string }
  | { readonly kind: 'camera'; readonly phrase: string }

/**
 * How each pick of a task looks, the console's one choice (the owner, 2026-10-08 night): the first look only
 * (`TaskOptionsIn.multi_view` off), the looks when needed (on: until a grasp is safe, and where the cell turns its
 * trigger on, a weak look is not), or every look (`every_look`: the early stop off).
 */
export const LOOKS_CHOICES = ['first', 'when_needed', 'every'] as const
export type LooksChoice = (typeof LOOKS_CHOICES)[number]

/** How a typed or spoken task starts, set beforehand (Settings, "Auftrag"). */
export interface TaskPrefs {
  /** `enter`: Enter starts a clean reading at once. `card`: Enter only reads, and Start on the card starts. */
  readonly start: 'enter' | 'card'
  /** The mode of a task Enter starts: the sentence does not set it (the owner, 2026-10-08). */
  readonly scope: 'once' | 'until_empty'
  readonly place: TaskPlace
  /** A sentence that names no part means anything the camera sees: its card comes ticked (Q7 A+). */
  readonly anything: boolean
  /** How each pick of every task looks; a card may change it for its task. */
  readonly looks: LooksChoice
}

export const DEFAULT_TASK: TaskPrefs = {
  start: 'enter',
  scope: 'once',
  place: { kind: 'default' },
  anything: false,
  looks: 'when_needed',
}

export interface Prefs {
  readonly view: View
  readonly setView: (view: View) => void
  /** Voice output on: Willy says the start, the end and every problem aloud. Off by default. */
  readonly voiceOut: boolean
  readonly setVoiceOut: (on: boolean) => void
  readonly theme: Theme
  readonly setTheme: (theme: Theme) => void
  readonly toggleTheme: () => void
  readonly task: TaskPrefs
  /** Change some of the task's settings; each is written at once. */
  readonly setTask: (change: Partial<TaskPrefs>) => void
}

function read(key: string): string | null {
  try {
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

function write(key: string, value: string): void {
  try {
    localStorage.setItem(key, value)
  } catch {
    /* a browser without storage still switches; it only forgets on reload */
  }
}

export function readView(): View {
  return read(VIEW_KEY) === 'tech' ? 'tech' : 'demo'
}

export function readVoiceOut(): boolean {
  return read(VOICE_OUT_KEY) === 'on'
}

/** A stored place as the setting means it; anything it cannot read is the cell's default place. */
export function placeOf(stored: string | null): TaskPlace {
  const pose = stored?.startsWith('pose:') ? stored.slice(5).trim() : ''
  if (pose !== '') return { kind: 'pose', pose }
  const phrase = stored?.startsWith('camera:') ? stored.slice(7).trim().slice(0, MAX_FIELD_CHARS) : ''
  if (phrase !== '') return { kind: 'camera', phrase }
  return { kind: 'default' }
}

/** A place as it is stored: empty for the default place, `pose:<name>`, `camera:<phrase>`. */
export function placeText(place: TaskPlace): string {
  if (place.kind === 'pose') return `pose:${place.pose}`
  if (place.kind === 'camera') return `camera:${place.phrase}`
  return ''
}

/** The task's settings as this browser stored them, each the default where nothing readable is stored. */
export function readTask(): TaskPrefs {
  const looks = read(TASK_LOOKS_KEY)
  return {
    start: read(TASK_START_KEY) === 'card' ? 'card' : 'enter',
    scope: read(TASK_SCOPE_KEY) === 'until_empty' ? 'until_empty' : 'once',
    place: placeOf(read(TASK_PLACE_KEY)),
    anything: read(TASK_ANYTHING_KEY) === 'anything',
    looks: looks === 'first' || looks === 'every' ? looks : 'when_needed',
  }
}

function writeTask(task: TaskPrefs): void {
  write(TASK_START_KEY, task.start)
  write(TASK_SCOPE_KEY, task.scope)
  write(TASK_PLACE_KEY, placeText(task.place))
  write(TASK_ANYTHING_KEY, task.anything ? 'anything' : 'ask')
  write(TASK_LOOKS_KEY, task.looks)
}

const PrefsContext = createContext<Prefs | null>(null)

export function PrefsProvider({ children }: { children?: ReactNode }) {
  const { theme, setTheme, toggle } = useTheme()
  const [view, setViewState] = useState<View>(readView)
  const [voiceOut, setVoiceState] = useState<boolean>(readVoiceOut)
  const [task, setTaskState] = useState<TaskPrefs>(readTask)

  const setView = useCallback((next: View) => {
    setViewState(next)
    write(VIEW_KEY, next)
  }, [])
  const setVoiceOut = useCallback((on: boolean) => {
    setVoiceState(on)
    write(VOICE_OUT_KEY, on ? 'on' : 'off')
    // Switching it off silences what is being said now, not only what comes next.
    if (!on) {
      try {
        window.speechSynthesis?.cancel()
      } catch {
        /* no speech synthesis in this browser: nothing is speaking */
      }
    }
  }, [])

  const setTask = useCallback((change: Partial<TaskPrefs>) => {
    setTaskState((current) => {
      const next = { ...current, ...change }
      writeTask(next)
      return next
    })
  }, [])

  useEffect(() => {
    document.documentElement.dataset.view = view
  }, [view])

  const value = useMemo<Prefs>(
    () => ({ view, setView, voiceOut, setVoiceOut, theme, setTheme, toggleTheme: toggle, task, setTask }),
    [view, setView, voiceOut, setVoiceOut, theme, setTheme, toggle, task, setTask],
  )
  return createElement(PrefsContext.Provider, { value }, children)
}

const NOOP = () => undefined

/** The preferences. Without a provider: the stored values, and switches that do nothing (a bare component test). */
export function usePrefs(): Prefs {
  const prefs = useContext(PrefsContext)
  if (prefs) return prefs
  return {
    view: readView(),
    setView: NOOP,
    voiceOut: readVoiceOut(),
    setVoiceOut: NOOP,
    theme: readTheme(),
    setTheme: NOOP,
    toggleTheme: NOOP,
    task: readTask(),
    setTask: NOOP,
  }
}
