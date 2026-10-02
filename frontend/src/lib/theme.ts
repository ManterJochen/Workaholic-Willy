/**
 * Dark, or light: dark unless a person chose light.
 *
 * The console is a control room next to a cell (OD 3): a dark ground with the logo's neon lime as the one accent, the
 * camera image dominant. Light is a switch for a bright hall, not a mode the operating system picks behind the
 * operator's back, so the choice is two states and the default is dark. A browser that still stores the old third
 * state (`system`) reads it as dark.
 *
 * The choice is written to `<html data-theme>` ALWAYS, light or dark, and mirrored to `localStorage`. The stylesheet
 * keeps its `prefers-color-scheme` block only for a page whose scripts did not run at all.
 *
 * **Half of this lives in the HTML, and has to.** A React effect runs after the browser has painted, so without help
 * the page would show one frame in the wrong colours before this file could stamp it. Both entry documents therefore
 * carry an inline script that reads the SAME key and stamps the same attribute before the first paint (`theme.test.ts`
 * runs those scripts); this module owns every change after that. The key (`willy.theme`) is the contract between them.
 */

import { useCallback, useEffect, useState } from 'react'

export type Theme = 'light' | 'dark'

export const THEME_KEY = 'willy.theme'

/** The stored choice: `light` only when light was chosen; everything else, nothing stored included, is dark. */
export function readTheme(): Theme {
  try {
    return localStorage.getItem(THEME_KEY) === 'light' ? 'light' : 'dark'
  } catch {
    // Private browsing, a file:// origin, a locked-down kiosk profile. Not worth failing over.
    return 'dark'
  }
}

/** Stamp the theme on the document. Always an explicit value: the page never falls back to the OS's choice. */
export function applyTheme(theme: Theme): void {
  document.documentElement.setAttribute('data-theme', theme)
}

function store(theme: Theme): void {
  try {
    localStorage.setItem(THEME_KEY, theme)
  } catch {
    /* see readTheme() */
  }
}

/** The theme and its switch. Each change is stamped on the document and remembered in this browser. */
export function useTheme(): { theme: Theme; setTheme: (theme: Theme) => void; toggle: () => void } {
  const [theme, setTheme] = useState<Theme>(readTheme)

  useEffect(() => {
    applyTheme(theme)
    store(theme)
  }, [theme])

  const toggle = useCallback(() => setTheme((current) => (current === 'dark' ? 'light' : 'dark')), [])

  return { theme, setTheme, toggle }
}
