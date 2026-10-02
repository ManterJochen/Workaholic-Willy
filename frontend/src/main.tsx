/**
 * The console entry point.
 *
 * `BrowserRouter` rather than a hash router because the backend serves this SPA and will be given a
 * catch-all that returns `index.html` for any non-`/v1` path -- so a deep link an operator bookmarked
 * ("/setup") survives a reload. A hash router would work without that mount, but it would also put a
 * `#` in every URL an operator is asked to type at a cell.
 *
 * The providers, outermost first (build plan 4.7): the language (German unless English was chosen), the person's
 * preferences (view, voice output, theme), the ONE shared cell poll and the cell stream, and the run on screen,
 * rebuilt from the server after a reload. Every screen reads these; none polls the cell on its own.
 */

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'

import App from './App'
import { I18nProvider } from './i18n'
import { PrefsProvider } from './model/prefs'
import { CellProvider } from './model/useCell'
import { RunProvider } from './model/useRun'
import './styles.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <I18nProvider>
        <PrefsProvider>
          <CellProvider>
            <RunProvider>
              <App />
            </RunProvider>
          </CellProvider>
        </PrefsProvider>
      </I18nProvider>
    </BrowserRouter>
  </StrictMode>,
)
