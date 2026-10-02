/**
 * The shell: a top bar that says what the cell is on every screen, four screens, and the console's global layers.
 *
 * **The top bar is a state surface, not only a menu** (it replaces the old sidebar). An operator watching the arm is
 * not looking for a settings page, so the four chips say, on every screen and in the same place: which cell this is
 * and what is on the other end (a simulated arm, URSim, or a controller about whose arm nothing is proven); what the
 * hand is believed to hold (a toggle's jaws are a COUNT of its output's changes, never a sensor, and the chip says
 * so); whether the arm is halted, stopped at the pendant, or a stop waits for a restart; and what runs. All four read
 * the one shared cell poll (`useCell`), never a poll of their own.
 *
 * **The switches a person needs at the cell sit beside them**: the demo or tech view, German or English, dark or
 * light, voice output, and the audience window for the projector (`demo.html`, its own page in its own window): a
 * read-only mirror of the cell with no control at all (build plan 4.3, `demo/Audience.tsx`), always on the black stage.
 *
 * **Four screens** (build plan 4.1): the Cockpit at `/`, Setup, History, Settings. The old addresses (`/pick`, `/cell`,
 * `/config`) redirect, so a bookmark from the old console lands where its work now is. The Pick screen is gone from
 * the shell; `POST /v1/pick` stays for programs.
 *
 * **Global layers**: the jaws question (it opens over every screen), the one confirm host for Restart and Home, and
 * the Diagnostics drawer of the tech view.
 */

import { useEffect, useState } from 'react'
import { NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom'

import type { ApiError, CellOut, ReadinessOut, TelemetryOut } from './api/client'
import Cockpit from './cockpit/Cockpit'
import DiagnosticsDrawer from './components/DiagnosticsDrawer'
import { Chip, ScreenHead, Segmented } from './components/ui'
import { useLang, useT, type MessageKey, type Msg, type Translate } from './i18n'
import { runKindMsg, stopMsg, whereMsg } from './i18n/codes'
import { Icon, Logo, type IconName } from './icons'
import JawsDialog from './jaws/JawsDialog'
import { stopTone } from './lib/outcome'
import { provenanceOf } from './lib/provenance'
import { ConfirmProvider } from './model/confirm'
import { usePrefs, type View } from './model/prefs'
import { phaseMsg, type RunView } from './model/runModel'
import { useCell } from './model/useCell'
import { useRun } from './model/useRun'
import History from './screens/History'
import Settings from './screens/Settings'
import Setup from './setup/Setup'

const SCREENS: ReadonlyArray<{ to: string; key: MessageKey; icon: IconName; end?: boolean }> = [
  { to: '/', key: 'nav.cockpit', icon: 'cockpit', end: true },
  { to: '/setup', key: 'nav.setup', icon: 'setup' },
  { to: '/history', key: 'nav.history', icon: 'history' },
  { to: '/settings', key: 'nav.settings', icon: 'settings' },
]

type Tone = 'ok' | 'warn' | 'block' | 'info' | 'bench' | 'idle'

interface ChipState {
  tone: Tone
  text: string
  live?: boolean
}

function words(t: Translate<MessageKey>, value: Msg | string): string {
  return typeof value === 'string' ? value : t.msg(value)
}

function cellChip(t: Translate<MessageKey>, cell: CellOut | null, error: ApiError | null, telemetry: TelemetryOut | null): ChipState {
  if (error && (!cell || error.httpStatus === 0)) return { tone: 'block', text: t('chip.noBackend') }
  if (!cell) return { tone: 'idle', text: t('common.none') }
  const known = ['disconnected', 'built', 'connecting', 'connected']
  const state = known.includes(cell.state) ? t(`cellState.${cell.state}` as MessageKey) : cell.state
  const provenance = cell.state === 'connected' && telemetry ? provenanceOf(telemetry).kind : null
  const parts = [cell.vendor, state]
  if (provenance && provenance !== 'unknown') parts.push(t(`prov.${provenance}` as MessageKey))
  let tone: Tone = 'idle'
  if (cell.gripper_substitution) tone = 'block'
  else if (cell.state === 'connected') tone = 'ok'
  else if (cell.state === 'built') tone = 'info'
  else if (cell.state === 'connecting') tone = 'warn'
  return { tone, text: parts.join(' · '), live: cell.state === 'connecting' }
}

/**
 * The hand: a toggle's output and its COUNTED jaws, or another hand's kind. The driver's class name ("JawIOGripper") is
 * a detail of the tech view; the server has no display name for the hand yet (`HandOut` carries none).
 */
function handChip(t: Translate<MessageKey>, cell: CellOut | null, view: View): ChipState {
  if (!cell) return { tone: 'idle', text: t('common.none') }
  const hand = cell.hand
  if (!hand || hand.kind === 'none') return { tone: 'idle', text: t('hand.none') }
  const where = hand.where ? words(t, whereMsg(hand.where)) : null
  const driver = view === 'tech' && hand.driver ? hand.driver : null
  if (hand.kind === 'toggle') {
    // A toggle has no sensor: what the chip says is the program's count of its output's changes.
    const jaws = t(`hand.jaws.${hand.jaws}` as MessageKey)
    const counted = hand.no_sensor ? ` (${t('hand.counted')})` : ''
    const tone: Tone = hand.jaws === 'open' ? 'ok' : hand.jaws === 'closed' ? 'warn' : hand.jaws === 'unknown' ? 'block' : 'idle'
    return { tone, text: [driver, where, `${jaws}${counted}`].filter(Boolean).join(' · ') }
  }
  const named = driver ?? t(`hand.kind.${hand.kind}`)
  const parts = [named, where, hand.no_sensor ? t('hand.noSensor') : null].filter((p): p is string => Boolean(p))
  return { tone: hand.connected ? 'ok' : 'idle', text: parts.join(' · ') }
}

/**
 * What the arm and its controller can do now, in the order a person must act on it, in every state of the cell (a
 * built arm carries its latch too, and Connect refuses on it):
 *
 * 1. the console's own halt: never shown as a stopped controller (build plan item 1);
 * 2. a protective or emergency stop: cleared at the pendant first, the stop card's first step (OD 14);
 * 3. a person the service waits for;
 * 4. not connected; then what the robot light says.
 *
 * A stop record waiting for "Zelle ist frei", then Restart or Home, is the run chip's to say: here, "Neustart nötig"
 * read like a reboot of the controller (the cockpit's review, 2026-10-02), and the controller itself may be ready
 * meanwhile.
 */
function controllerChip(t: Translate<MessageKey>, cell: CellOut | null, readiness: ReadinessOut | null): ChipState {
  if (!cell) return { tone: 'idle', text: t('controller.offline') }
  const connected = cell.state === 'connected'
  const robot = connected ? readiness?.lights?.find((light) => light.id === 'robot') : undefined
  if (cell.halted || robot?.code === 'halted') return { tone: 'block', text: t('controller.halted') }
  if (robot?.code === 'controller_stopped') return { tone: 'block', text: t('controller.stopped') }
  if (cell.needs_person) return { tone: 'block', text: t('controller.needsPerson') }
  if (!connected) return { tone: 'idle', text: t('controller.offline') }
  if (robot?.code === 'controller_unreadable') return { tone: 'warn', text: t('controller.unknown') }
  if (robot?.state === 'ok') return { tone: 'ok', text: t('controller.ready') }
  return { tone: 'idle', text: t('controller.unknown') }
}

function runChip(t: Translate<MessageKey>, cell: CellOut | null, view: RunView): ChipState {
  const active = cell?.active_run_id ?? null
  if (active) {
    if (view.runId !== active || !view.kind) return { tone: 'info', text: t('run.phase.running'), live: true }
    const parts = [t.msg(runKindMsg(view.kind)), t.msg(phaseMsg(view))]
    if (view.current.part !== null) parts.push(t('run.part', { part: view.current.part }))
    return { tone: view.phase === 'halting' ? 'block' : 'info', text: parts.join(' · '), live: true }
  }
  // A stop record outlives its run, a Disconnect and a page reload: what it owes comes first, the person's "Zelle ist
  // frei" (a clearing after the stop), then Restart or Home. The stop card says what happened.
  const record = cell?.recovery
  if (record) {
    const cleared = record.cleared_at != null && record.cleared_at > record.at
    return { tone: 'block', text: t(cleared ? 'run.recovery.cleared' : 'run.recovery.open') }
  }
  if (view.runId && view.kind && view.stopCode) {
    return { tone: stopTone(view.stopCode) as Tone, text: `${t.msg(runKindMsg(view.kind))}: ${t.msg(stopMsg(view.stopCode))}` }
  }
  // A run the server forgot (it restarted) has no stop code to show: say that it is no longer known, not "no run".
  if (view.runId && view.kind && view.phase === 'lost') {
    return { tone: 'warn', text: `${t.msg(runKindMsg(view.kind))}: ${t.msg(phaseMsg(view))}` }
  }
  return { tone: 'idle', text: t('run.none') }
}

function TopBar({ onDiagnostics }: { onDiagnostics: () => void }) {
  const t = useT()
  const { lang, setLang } = useLang()
  const prefs = usePrefs()
  const { cell, error, telemetry, readiness } = useCell()
  const { view } = useRun()

  const chips: Array<[MessageKey, ChipState]> = [
    ['chip.cell', cellChip(t, cell, error, telemetry)],
    ['chip.hand', handChip(t, cell, prefs.view)],
    ['chip.controller', controllerChip(t, cell, readiness)],
    ['chip.run', runChip(t, cell, view)],
  ]

  const openAudience = () => {
    // Its own window, so the operator can drag it to the projector and press F11. It is a read-only mirror (4.3): a
    // viewer there is never one click from a motion.
    window.open('/demo.html', 'willy-audience', 'popup')
  }

  return (
    <header className="topbar">
      <div className="topbar-row">
        {/* The Willy mark and the logo's wordmark, WORKAHOLIC WILLY with WILLY in the lime (owner, 2026-10-01). */}
        <NavLink to="/" className="brand" aria-label={t('app.name')} end>
          <Logo className="brand-mark" />
          <span className="brand-name">
            <strong>
              Workaholic <em>Willy</em>
            </strong>
            <span>{t('app.console')}</span>
          </span>
        </NavLink>

        <nav className="topnav" aria-label={t('nav.label')}>
          {SCREENS.map((screen) => (
            <NavLink key={screen.to} to={screen.to} end={screen.end} className={({ isActive }) => (isActive ? 'active' : '')}>
              <Icon name={screen.icon} size={17} />
              <span>{t(screen.key)}</span>
            </NavLink>
          ))}
        </nav>

        <div className="switches">
          <Segmented
            label={t('view.label')}
            value={prefs.view}
            options={[
              { value: 'demo', label: t('view.demo') },
              { value: 'tech', label: t('view.tech') },
            ]}
            onChange={prefs.setView}
          />
          <Segmented
            label={t('lang.label')}
            value={lang}
            options={[
              { value: 'de', label: t('lang.de'), title: 'Deutsch' },
              { value: 'en', label: t('lang.en'), title: 'English' },
            ]}
            onChange={setLang}
          />
          <button
            type="button"
            className="iconbtn"
            onClick={prefs.toggleTheme}
            aria-label={prefs.theme === 'dark' ? t('theme.toLight') : t('theme.toDark')}
            title={prefs.theme === 'dark' ? t('theme.toLight') : t('theme.toDark')}
          >
            <Icon name={prefs.theme === 'dark' ? 'sun' : 'moon'} />
          </button>
          <button
            type="button"
            className="iconbtn"
            aria-pressed={prefs.voiceOut}
            onClick={() => prefs.setVoiceOut(!prefs.voiceOut)}
            aria-label={prefs.voiceOut ? t('voice.on') : t('voice.off')}
            title={`${prefs.voiceOut ? t('voice.on') : t('voice.off')}. ${t('voice.hint')}`}
          >
            <Icon name={prefs.voiceOut ? 'speaker' : 'speakerOff'} />
          </button>
          <button type="button" className="textbtn ghost" onClick={openAudience} title={t('audience.hint')} aria-label={t('audience.open')}>
            <Icon name="monitor" size={17} />
            <span className="label">{t('audience.open')}</span>
          </button>
          {prefs.view === 'tech' && (
            <button type="button" className="iconbtn" onClick={onDiagnostics} aria-label={t('diag.open')} title={t('diag.open')}>
              <Icon name="pulse" />
            </button>
          )}
        </div>
      </div>

      {/* The status strip: the cell's state in full words, on every screen, never cut to an ellipsis. */}
      <div className="statusbar" role="group" aria-label={t('chip.label')}>
        {chips.map(([label, chip]) => (
          <Chip key={label} label={t(label)} tone={chip.tone} live={chip.live} title={`${t(label)}: ${chip.text}`}>
            {chip.text}
          </Chip>
        ))}
      </div>
    </header>
  )
}

function NotFound() {
  const t = useT()
  return (
    <div className="page">
      <ScreenHead title={t('notFound.title')} lede={t('notFound.body')} />
    </div>
  )
}

export default function App() {
  const t = useT()
  const prefs = usePrefs()
  const location = useLocation()
  const [diagnostics, setDiagnostics] = useState(false)

  // The window's title names the screen, in the reader's language: several console tabs stay tellable apart.
  useEffect(() => {
    const screen = SCREENS.find((s) => (s.end ? location.pathname === s.to : location.pathname.startsWith(s.to)))
    document.title = screen ? `${t(screen.key)} · Workaholic-Willy` : 'Workaholic-Willy'
  }, [location.pathname, t])

  return (
    <ConfirmProvider>
      <div className="shell">
        <a className="skip-link" href="#main">
          {t('app.skip')}
        </a>
        <TopBar onDiagnostics={() => setDiagnostics(true)} />
        <main className="main" id="main" tabIndex={-1}>
          <Routes>
            <Route path="/" element={<Cockpit />} />
            <Route path="/setup" element={<Setup />} />
            <Route path="/history" element={<History />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/pick" element={<Navigate to="/" replace />} />
            <Route path="/cell" element={<Navigate to="/setup" replace />} />
            <Route path="/config" element={<Navigate to="/settings" replace />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </main>
        <JawsDialog />
        <DiagnosticsDrawer open={prefs.view === 'tech' && diagnostics} onClose={() => setDiagnostics(false)} />
      </div>
    </ConfirmProvider>
  )
}
