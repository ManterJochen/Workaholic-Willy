/**
 * The demo entry point. Separate from the console's on purpose -- see `Demo.tsx`.
 *
 * It pulls the console's stylesheet for the shared tokens (fonts, the status palette) and then
 * overrides the layout entirely: one page, one instruction, one narration, no navigation.
 */

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import '../styles.css'
import Demo from './Demo'
import './demo.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Demo />
  </StrictMode>,
)
