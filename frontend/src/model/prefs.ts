/**
 * The person's preferences: the demo or the tech view, voice output, and the theme.
 *
 * Each is per browser (`localStorage`), so the cell PC and the projector window remember their own:
 *
 * * `willy.view`: `demo` (big type, localised lines only, no JSON) or `tech` (the backend's sentences, the data,
 *   route badges, the raw event log, the Diagnostics drawer). Demo unless tech was chosen (OD 1). The view is also
 *   stamped on `<html data-view>`, so an area's stylesheet can size its type for it without a prop;
 * * `willy.voiceOut`: the browser's own speech synthesis at the key moments (OD 18); OFF unless switched on;
 * * the theme, owned by `lib/theme.ts` (`willy.theme`, dark by default); held here so the top bar's switch and the
 *   Settings page show the same state.
 *
 * Written without JSX (`.ts`) so the provider and its hook can share one module (see `i18n/index.ts`).
 */

import { createContext, createElement, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'

import { readTheme, useTheme, type Theme } from '../lib/theme'

export type View = 'demo' | 'tech'

export const VIEW_KEY = 'willy.view'
export const VOICE_OUT_KEY = 'willy.voiceOut'

export interface Prefs {
  readonly view: View
  readonly setView: (view: View) => void
  /** Voice output on: Willy says the start, the end and every problem aloud. Off by default. */
  readonly voiceOut: boolean
  readonly setVoiceOut: (on: boolean) => void
  readonly theme: Theme
  readonly setTheme: (theme: Theme) => void
  readonly toggleTheme: () => void
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

const PrefsContext = createContext<Prefs | null>(null)

export function PrefsProvider({ children }: { children?: ReactNode }) {
  const { theme, setTheme, toggle } = useTheme()
  const [view, setViewState] = useState<View>(readView)
  const [voiceOut, setVoiceState] = useState<boolean>(readVoiceOut)

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

  useEffect(() => {
    document.documentElement.dataset.view = view
  }, [view])

  const value = useMemo<Prefs>(
    () => ({ view, setView, voiceOut, setVoiceOut, theme, setTheme, toggleTheme: toggle }),
    [view, setView, voiceOut, setVoiceOut, theme, setTheme, toggle],
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
  }
}
