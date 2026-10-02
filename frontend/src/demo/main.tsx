/**
 * The audience window's entry point (`demo.html`). Separate from the console's on purpose: it has no navigation, no
 * config, no write path and no button, so a viewer is never one click from a motion (see `Audience.tsx`).
 *
 * The providers are the console's own, so the window reads the same cell and draws the same run model: the language
 * (German unless English was chosen), the shared cell poll (every second here: the projector follows a run promptly),
 * and the run on screen. It pulls the console's stylesheet for the tokens and the brand utilities, then its own
 * layout. The window is always the black stage, the brand's look on a projector, whatever theme the console uses.
 */

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import '../styles.css'
import { I18nProvider } from '../i18n'
import { applyTheme } from '../lib/theme'
import { CellProvider } from '../model/useCell'
import { RunProvider } from '../model/useRun'
import Audience from './Audience'
import './demo.css'

/** The audience window's cell poll: a second, so the projector follows a run as it starts and stops. */
const AUDIENCE_POLL_MS = 1000

applyTheme('dark')

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <I18nProvider>
      <CellProvider pollMs={AUDIENCE_POLL_MS}>
        <RunProvider>
          <Audience />
        </RunProvider>
      </CellProvider>
    </I18nProvider>
  </StrictMode>,
)
