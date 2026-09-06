/**
 * The console entry point.
 *
 * `BrowserRouter` rather than a hash router because the backend serves this SPA and will be given a
 * catch-all that returns `index.html` for any non-`/v1` path -- so a deep link an operator bookmarked
 * ("/pick") survives a reload. A hash router would work without that mount, but it would also put a
 * `#` in every URL an operator is asked to type at a cell.
 */

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'

import App from './App'
import './styles.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
)
