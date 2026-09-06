/**
 * The shell: navigation, and the one thing that must be true on every screen.
 *
 * That one thing is the cell's identity. A console showing a green telemetry panel is indistinguishable
 * from a console showing a green telemetry panel for a SIMULATED arm, and the difference is whether a
 * number on screen describes a machine in the room. So the sidebar carries the vendor, the profile
 * chain and the connection state at all times — not just on the Cell screen, where an operator who is
 * mid-pick is not looking.
 *
 * Which is also why the sidebar is a state surface and not a menu: identity sits at the bottom in a
 * fixed place, in mono, so it can be read without being looked for.
 */

import { NavLink, Route, Routes } from 'react-router-dom'

import { api } from './api/client'
import { StatusPill } from './components/ui'
import { useAsync, usePoll } from './lib/useAsync'
import { useTheme } from './lib/theme'
import Cell from './screens/Cell'
import Config from './screens/Config'
import History from './screens/History'
import Pick from './screens/Pick'
import Preflight from './screens/Preflight'

const SCREENS = [
  { to: '/', label: 'Preflight', element: <Preflight />, end: true },
  { to: '/cell', label: 'Cell', element: <Cell /> },
  { to: '/pick', label: 'Pick', element: <Pick /> },
  { to: '/history', label: 'History', element: <History /> },
  { to: '/config', label: 'Config', element: <Config /> },
]

const THEME_LABEL = { system: 'Auto', light: 'Light', dark: 'Dark' } as const

export default function App() {
  const cell = useAsync(() => api.cell())
  const health = useAsync(() => api.health())
  const { theme, cycle } = useTheme()
  // Slow on purpose. This is an identity readout, not telemetry; the Cell screen polls at 1 Hz when
  // it needs to, and a second fast poll from the shell would double every request for no new fact.
  usePoll(cell.reload, 5000, true)

  const connected = cell.data?.state === 'connected'

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="mark" aria-hidden="true">
            W
          </div>
          <div>
            <h1>Workaholic-Willy</h1>
            <div className="sub">console {health.data?.version ?? ''}</div>
          </div>
        </div>

        <nav className="nav">
          {SCREENS.map((s) => (
            <NavLink key={s.to} to={s.to} end={s.end} className={({ isActive }) => (isActive ? 'active' : '')}>
              {s.label}
              {s.to === '/cell' && (
                <StatusPill status={connected ? 'ok' : 'idle'}>
                  {connected ? 'up' : 'down'}
                </StatusPill>
              )}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-foot">
          {cell.error ? (
            <span className="pill block">no backend</span>
          ) : (
            <>
              <div className="k">vendor</div>
              <div className="v">{cell.data?.vendor ?? '—'}</div>
              <div className="k">profile</div>
              <div className="v">{cell.data?.profile || '(none)'}</div>
              <div className="k">arm</div>
              <div className="v">{cell.data?.arm ?? 'not built'}</div>
            </>
          )}
          <div className="foot-links">
            <a href="/demo.html">Demo view →</a>
            <button
              className="ghost theme-toggle"
              onClick={cycle}
              title="Theme: auto follows the operating system"
            >
              {THEME_LABEL[theme]}
            </button>
          </div>
        </div>
      </aside>

      <main className="main">
        <Routes>
          {SCREENS.map((s) => (
            <Route key={s.to} path={s.to} element={s.element} />
          ))}
          <Route
            path="*"
            element={
              <>
                <h2>No such screen</h2>
                <p className="lede">
                  Nothing is served at this address. Pick one from the left.
                </p>
              </>
            }
          />
        </Routes>
      </main>
    </div>
  )
}
