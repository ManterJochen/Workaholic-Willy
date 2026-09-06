/**
 * Light, dark, or whatever the operating system says.
 *
 * Three states rather than two, and the third one is the default. A console lives on whatever screen
 * is bolted next to the cell -- a bright hall in the morning, the same hall at night -- so following
 * the OS is the only setting that is right without anyone thinking about it. The explicit choices
 * exist because "the OS is wrong for this screen" is a real situation and an operator should not have
 * to change a system setting to fix a panel.
 *
 * The choice is written to `<html data-theme>` and mirrored to `localStorage`. `system` REMOVES the
 * attribute rather than resolving it to a colour, which is what lets the stylesheet keep following
 * `prefers-color-scheme` live -- resolving it here would freeze the page at whatever the OS said when
 * the tab was opened.
 *
 * **Half of this lives in the HTML, and has to.** A React effect runs after the browser has painted,
 * so a saved dark choice would show one white frame before this file could apply it. Both entry
 * documents therefore carry a four-line inline script that reads the SAME key and stamps the same
 * attribute before first paint; this module then owns every change after that. The duplication is
 * deliberate and the key (`willy.theme`) is the contract between them.
 */

import { useCallback, useEffect, useState } from 'react'

export type Theme = 'system' | 'light' | 'dark'

const KEY = 'willy.theme'
const ORDER: Theme[] = ['system', 'light', 'dark']

function read(): Theme {
  try {
    const stored = localStorage.getItem(KEY)
    return ORDER.includes(stored as Theme) ? (stored as Theme) : 'system'
  } catch {
    // Private browsing, a file:// origin, a locked-down kiosk profile. Not worth failing over.
    return 'system'
  }
}

function apply(theme: Theme): void {
  const root = document.documentElement
  if (theme === 'system') root.removeAttribute('data-theme')
  else root.setAttribute('data-theme', theme)
}

/** Read the current theme and cycle it: system -> light -> dark -> system. */
export function useTheme(): { theme: Theme; cycle: () => void } {
  const [theme, setTheme] = useState<Theme>(read)

  useEffect(() => {
    apply(theme)
    try {
      localStorage.setItem(KEY, theme)
    } catch {
      /* see read() */
    }
  }, [theme])

  const cycle = useCallback(() => {
    setTheme((current) => ORDER[(ORDER.indexOf(current) + 1) % ORDER.length])
  }, [])

  return { theme, cycle }
}
