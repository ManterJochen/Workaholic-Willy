/**
 * The room view: what the robot is looking at, what it is doing, and three numbers that are true.
 *
 * This is a deliberately SMALL surface. It cannot build a cell, cannot connect one and cannot write
 * config. It drives one already-connected cell and shows what happens. Everything it omits, it omits
 * so that a page shown on a projector cannot be the thing that brings a robot up in front of an
 * audience.
 *
 * It DOES read `/v1/history/kpis` — a read, on a poll, for the numbers below. An earlier version of
 * this comment claimed the page "cannot reach the history" while doing exactly that ten seconds apart,
 * which is the kind of stale boundary claim that makes every other one on the page worth less.
 *
 * **It states what it is driving, always**, and it states it in the one place an audience is looking.
 * The badge comes from `lib/provenance.ts`, which distinguishes three things this page must never blur:
 * a simulated DRIVER, a provably simulated CONTROLLER (URSim — real UR software, robot that does not
 * exist), and a real controller stack about whose arm nothing can be proven. The third is labelled
 * "controller", never "physical arm". A demo that lets any of those read as another is not a demo, it
 * is a claim — and this project's discipline is that a sim result stays labelled as a sim result,
 * including when the audience would prefer otherwise.
 *
 * ── THE THREE NUMBERS, AND WHY THESE THREE ────────────────────────────────────────────────────────
 * `api/history.py` computes six KPIs — four rates and two durations. Three are structurally
 * meaningless on a console-driven cell, and the BACKEND now withholds them rather than this page
 * choosing not to draw them:
 *
 *   dense_recovery_success_rate   denominator is dense-mode attempts that also recovered; neither
 *                                 happens on a shipped `auto` cell
 *   first_attempt_success_rate    the same number as the success rate whenever nothing recovered
 *   median_cycle_time_s           reads a key no production writer sets
 *
 * What remains published is `pick_success_rate`, `dead_loop_rate`, `safety_rejection_rate` and
 * `median_attempt_seconds`. This page draws two of those four plus one live counter; the other two
 * belong on the History screen, where an operator has room to read what they mean.
 *
 * What is left, and what this page shows:
 *
 *   picked / attempted     a direct count of completed pick() calls in the run on screen. No rate, no
 *                          derivation, nothing default-off in the path. The hero number.
 *   success rate           over the logged attempts, with the count it is out of — a rate without its
 *                          denominator is a number you cannot argue with, which is the problem.
 *   median attempt         from `attempt_wall_time_s`, which pick() stamps on every attempt whatever
 *                          the config says. The honest answer to "how long does one take?".
 *
 * ── THE PICTURE ───────────────────────────────────────────────────────────────────────────────────
 * `Viewfinder` shows whatever the cell can honestly offer while it is idle — a camera frame on a real
 * cell, a SYNTHETIC scene on a rehearsal one, which has no camera at all — and the grasp overlay while
 * it picks. It labels which, every time. It never invents the missing one: during a pick with the
 * overlay switched off it holds the last frame, dimmed and marked `held`, rather than showing a
 * live-looking image of a scene the robot has since rearranged.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import { api, type CellOut, type RollupOut, type RunOut, type TelemetryOut } from '../api/client'
import { followRun, type RunEvent } from '../api/events'
import { Viewfinder } from '../components/Viewfinder'
import { outcomeLabel, outcomeTone } from '../lib/outcome'
import { provenanceOf } from '../lib/provenance'
import { PromptInput } from '../prompt/PromptInput'
import { usePromptPipeline } from '../prompt/pipeline'
import { usePoll } from '../lib/useAsync'

export default function Demo() {
  const [cell, setCell] = useState<CellOut | null>(null)
  const [telemetry, setTelemetry] = useState<TelemetryOut | null>(null)
  const [kpis, setKpis] = useState<RollupOut | null>(null)
  const [picks, setPicks] = useState(3)
  const [run, setRun] = useState<RunOut | null>(null)
  const [events, setEvents] = useState<RunEvent[]>([])
  const [error, setError] = useState<string | null>(null)
  const stopStream = useRef<(() => void) | null>(null)

  const readCell = useCallback(() => {
    api.cell().then(setCell).catch(() => setCell(null))
  }, [])
  const readKpis = useCallback(() => {
    api.kpis().then(setKpis).catch(() => undefined)
  }, [])

  useEffect(() => {
    readCell()
    readKpis()
  }, [readCell, readKpis])
  usePoll(readCell, 3000, true)
  // Slow: this is a file read on the server and the number moves once per attempt, not per frame.
  usePoll(readKpis, 10000, true)

  const connected = cell?.state === 'connected'
  usePoll(
    () => {
      api.status(false).then(setTelemetry).catch(() => setTelemetry(null))
    },
    1000,
    connected,
  )

  useEffect(() => () => stopStream.current?.(), [])

  const follow = useCallback((started: RunOut) => {
    setError(null)
    setEvents([])
    setRun(started)
    stopStream.current?.()
    stopStream.current = followRun(started.id, {
      onEvent: (event) => {
        setEvents((prev) => [...prev.slice(-40), event])
        if (event.type === 'run_finished' || event.type === 'run_error') {
          api.run(started.id).then(setRun).catch(() => undefined)
          readKpis()
        }
      },
      onState: () => undefined,
    })
  }, [readKpis])

  // The same pipeline the operator console uses -- one prompt path, one provenance record, whichever
  // screen the words were entered on. This page used to keep its own copy of all of it.
  const prompts = usePromptPipeline(follow)
  const prompt = prompts.text
  const start = useCallback(() => void prompts.start(picks), [prompts, picks])

  const latest = events[events.length - 1]
  const running = run?.state === 'running'
  const provenance = provenanceOf(telemetry)

  return (
    <div className="demo">
      <header className="demo-head">
        <div className="demo-brand">
          <div className="demo-mark" aria-hidden="true">
            W
          </div>
          <div>
            <div className="demo-title">Workaholic-Willy</div>
            <div className="demo-sub">vision–language robot grasping</div>
          </div>
        </div>
        <div className="demo-badge">
          {!cell ? (
            <span className="pill block">no backend</span>
          ) : !connected ? (
            <span className="pill warn">cell not connected</span>
          ) : (
            // Never "physical arm". See `lib/provenance.ts`: this console can prove a simulator and
            // cannot prove a real robot, and the page that gets projected is the last place to
            // invent the difference.
            <span className={`pill ${provenance.tone}`}>{provenance.label}</span>
          )}
        </div>
      </header>

      <div className="demo-body">
        <section className="demo-view">
          <Viewfinder />
        </section>

        <section className="demo-side">
          <div className="demo-ask">
            <PromptInput
              value={prompt}
              onChange={prompts.setText}
              spoken={prompts.source === 'spoken'}
              placeholder="Tell it what to pick…"
              canSubmit={connected && !running}
              onSubmit={start}
            />
            <input
              type="number"
              min={1}
              max={20}
              value={picks}
              aria-label="how many"
              onChange={(e) => setPicks(Math.max(1, Number(e.target.value) || 1))}
            />
            <button className="danger" disabled={!connected || running} onClick={() => void start()}>
              {running ? 'Working…' : 'Go'}
            </button>
          </div>

          {/*
            The backend's own contract, said out loud: `PickIn.prompt` defaults to "" and an empty
            prompt means "whatever the perception source already targets". It matters here because the
            cell that needs no hardware -- the rehearsal one -- emits no labels at all BY DESIGN, so a
            typed prompt on it correctly matches nothing and the run honestly refuses. Leaving the hint
            out would make the working path look like the broken one.
          */}
          <div className="demo-hint">
            Leave it empty to pick whatever the cell is already looking at. A name only works if
            perception actually returns that name.
          </div>

          {/* ⚠ BOTH SOURCES. The pipeline reports a refused START (cell busy, prompt unroutable)
              and this page reports everything else. Rendering only its own would have made a refused
              pick look like a pick that simply never happened -- in front of an audience. */}
          {(error || prompts.error) && (
            <div className="banner error demo-error">{error ?? prompts.error?.message}</div>
          )}

          <div className="demo-now">
            {latest ? latest.human : connected ? 'Ready.' : 'Bring the cell up in the console first.'}
          </div>

          <div className="demo-figures">
            <Figure
              value={run ? `${run.succeeded ?? 0}/${run.attempted ?? 0}` : '—'}
              label="picked, this run"
              note={run ? `asked for ${run.requested_picks}` : 'no run yet'}
            />
            <Figure
              value={percent(kpis?.kpis?.['pick_success_rate'])}
              label="success rate"
              note={`over ${kpis?.total_attempts ?? 0} logged attempt${
                kpis?.total_attempts === 1 ? '' : 's'
              }`}
            />
            <Figure
              value={seconds(kpis?.kpis?.['median_attempt_seconds'])}
              label="median attempt"
              note="wall time, every attempt"
            />
          </div>
        </section>
      </div>

      <section className="demo-feed">
        {events.length === 0 ? (
          <div className="demo-line faint">Nothing has happened yet.</div>
        ) : (
          events
            .slice()
            .reverse()
            .map((event, i) => (
              <div className={`demo-line ${event.severity}`} key={`${event.seq}-${i}`}>
                <span className="demo-seq">{event.type === 'gap' ? '—' : event.seq}</span>
                <span>{event.human}</span>
              </div>
            ))
        )}
      </section>

      <footer className="demo-foot">
        <div className="demo-foot-left">
          {telemetry?.tcp_position_mm && (
            <span className="mono">
              TCP {telemetry.tcp_position_mm.map((n) => n.toFixed(0)).join(' / ')} mm
            </span>
          )}
          {run?.outcomes && run.outcomes.length > 0 && (
            <span className="demo-outcomes">
              {run.outcomes.slice(-8).map((outcome, i) => (
                <span key={i} className={`pill ${outcomeTone(outcome)}`}>
                  {outcomeLabel(outcome)}
                </span>
              ))}
            </span>
          )}
        </div>
        <a href="/">full console →</a>
      </footer>
    </div>
  )
}

/** One number, large enough to read from the back of the room, with what it is out of underneath. */
function Figure({ value, label, note }: { value: string; label: string; note: string }) {
  return (
    <div className="demo-figure">
      <div className="demo-figure-value mono">{value}</div>
      <div className="demo-figure-label">{label}</div>
      <div className="demo-figure-note">{note}</div>
    </div>
  )
}

/**
 * A rate as a percentage — and an ABSENT rate as a dash, never as 0 %.
 *
 * The backend withholds a KPI it cannot measure rather than publishing the structural zero its
 * arithmetic produces. Rendering `undefined` as `0 %` here would put the zero back.
 */
function percent(value: unknown): string {
  return typeof value === 'number' ? `${Math.round(value * 100)}%` : '—'
}

function seconds(value: unknown): string {
  return typeof value === 'number' ? `${value.toFixed(1)}s` : '—'
}
