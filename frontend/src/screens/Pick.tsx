/**
 * Type an instruction, watch the cell carry it out.
 *
 * Three things on this screen are deliberate and should survive a redesign:
 *
 * 1. **Stop does not stop the arm.** `/v1/pick/stop` declines to start the NEXT attempt; a motion
 *    already commanded runs to completion. The button says so, in those words. An operator who
 *    believes this button is an e-stop is in danger, and no amount of styling fixes that — only the
 *    sentence does.
 * 2. **The route preview is free.** `/v1/diagnostics/route` is pure text analysis: no GPU, no model,
 *    no image. So the console can answer “which stack will this prompt touch, and can it run here?”
 *    while the operator is still typing, instead of after a failed pick.
 * 3. **The event list renders the backend's own sentences.** `human` is written by the library; the
 *    console never composes its own summary of what happened, because the two would drift and the
 *    prettier one would win.
 * 4. **The viewfinder says which picture it is showing.** While a pick runs the backend refuses to
 *    touch the camera -- the pick owns it -- and serves the grasp overlay instead. Those are different
 *    pictures, and the badge over the frame is what keeps them from reading as the same one.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiError, api, type RoutePreviewOut, type RunOut } from '../api/client'
import { followRun, type RunEvent, type StreamState } from '../api/events'
import { Viewfinder } from '../components/Viewfinder'
import { Caveat, ErrorBanner, Loading, Panel, StatusPill } from '../components/ui'
import { outcomeLabel, outcomeTone, runTone } from '../lib/outcome'
import { useAsync, usePoll } from '../lib/useAsync'
import { PromptInput } from '../prompt/PromptInput'
import { usePromptPipeline } from '../prompt/pipeline'

export default function Pick() {
  const cell = useAsync(() => api.cell())
  const [picks, setPicks] = useState(1)
  const [run, setRun] = useState<RunOut | null>(null)
  const [events, setEvents] = useState<RunEvent[]>([])
  const [stream, setStream] = useState<StreamState>('closed')
  const [error, setError] = useState<ApiError | null>(null)
  const [route, setRoute] = useState<RoutePreviewOut | null>(null)
  const stopStream = useRef<(() => void) | null>(null)
  // Holds `follow`, which is declared below. A plain forward reference would be a use-before-declare;
  // hoisting `follow` up here would drag the whole stream-following block above the render state it
  // reads. The ref is the small seam that keeps both halves where they belong.
  const followRef = useRef<(run: RunOut) => void>(() => undefined)
  const listEnd = useRef<HTMLDivElement | null>(null)

  // ⭑ THE PROMPT, ITS PROVENANCE AND THE START ARE NOT THIS SCREEN'S BUSINESS ANY MORE. Both screens
  // that carry a prompt box used to keep their own copy of the same three decisions, and adding
  // speech to one of them would have made it three copies. `usePromptPipeline` owns them; this
  // screen owns only what is specific to it -- following the event stream.
  //
  // ⚠ Declared HERE rather than beside `start`, because the debounced route preview below reads the
  // prompt and a `const` cannot be read above its own declaration.
  const prompts = usePromptPipeline((started) => followRef.current(started))
  const prompt = prompts.text
  const starting = prompts.starting

  const connected = cell.data?.state === 'connected'
  // Refreshed, not read once. `useAsync` runs its effect a single time, so a cell that was connected
  // (or disconnected) from another tab, or by the CLI, left this screen's red button in the wrong
  // state until a navigation happened to remount it. Slow on purpose -- it is a gate, not telemetry.
  usePoll(cell.reload, 5000, true)

  // The route preview, debounced. Free to call, but not free to call on every keystroke of a long
  // prompt, and a request per character would make the answer flicker while it is being read.
  useEffect(() => {
    if (!prompt.trim()) return
    const id = setTimeout(() => {
      api
        .routePreview(prompt)
        .then(setRoute)
        .catch(() => setRoute(null))
    }, 350)
    return () => clearTimeout(id)
  }, [prompt])

  useEffect(() => () => stopStream.current?.(), [])

  useEffect(() => {
    listEnd.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [events.length])

  // The list is cleared only once the new run EXISTS. Clearing first meant a refused start (the cell
  // busy, the prompt unroutable) wiped the previous run's narration -- and the stream cannot replay
  // it, because `followRun` only ever moves its cursor forward.
  const follow = useCallback((started: RunOut) => {
    setRun(started)
    setEvents([])
    stopStream.current?.()
    stopStream.current = followRun(started.id, {
      onEvent: (event) => {
        setEvents((prev) => [...prev, event])
        if (event.type === 'run_finished' || event.type === 'run_error') {
          api.run(started.id).then(setRun).catch(() => undefined)
        }
      },
      onState: setStream,
    })
  }, [])
  // ⚠ IN AN EFFECT, NOT DURING RENDER, and `usePoll` in `lib/useAsync.ts` says why in its own words:
  // a ref mutated while rendering reads, by React's own rules, as a value that should have been
  // state, and Strict Mode's double render writes it twice before anything can use it.
  useEffect(() => {
    followRef.current = follow
  }, [follow])

  const start = useCallback(() => void prompts.start(picks), [prompts, picks])

  const stop = useCallback(async () => {
    try {
      setRun(await api.stopPick())
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, null, String(err)))
    }
  }, [])

  const running = run?.state === 'running'
  // Derived: an empty box has no route. Deriving beats clearing state from an effect, which would
  // render the previous prompt's verdict for one frame after the box is emptied.
  const shownRoute = prompt.trim() ? route : null

  return (
    <>
      <h2>Pick</h2>
      <p className="lede">
        One instruction, in your own words. The cell grounds it, ranks grasps, checks safety, and
        moves — and narrates every step of that below as it happens.
      </p>

      {(error || prompts.error) && <ErrorBanner error={(error ?? prompts.error)!} />}

      {!connected && (
        <div className="banner warn">
          <div className="body">
            <strong>The cell is not connected.</strong>
            <div className="small dim">
              Assemble and connect it on the Cell screen first. Nothing here will move until you do.
            </div>
          </div>
        </div>
      )}

      <Panel title="Instruction">
        <PromptInput
          value={prompt}
          onChange={prompts.setText}
          spoken={prompts.source === 'spoken'}
          canSubmit={connected && !running && !starting}
          onSubmit={start}
        />
        <div className="actions instruction-row">
          <label className="check">
            picks
            <input
              type="number"
              min={1}
              max={50}
              value={picks}
              className="count"
              onChange={(e) => setPicks(Math.max(1, Number(e.target.value) || 1))}
            />
          </label>
          <button
            className="danger"
            disabled={!connected || running || starting}
            onClick={() => void start()}
          >
            {starting ? 'Starting…' : 'Pick — this moves the robot'}
          </button>
          <button disabled={!running} onClick={() => void stop()}>
            Stop after this attempt
          </button>
        </div>

        {shownRoute && (
          <div className="small dim">
            This prompt would be grounded by the{' '}
            <span className="mono">{shownRoute.route}</span> route ({shownRoute.reason}).{' '}
            {shownRoute.runnable ? (
              <StatusPill status="ok">runnable here</StatusPill>
            ) : (
              <>
                <StatusPill status="block">cannot run on this box</StatusPill>{' '}
                <span>{shownRoute.blocked_reason}</span>
              </>
            )}
            {shownRoute.normalized_prompt && (
              <>
                {' '}
                The grounder will receive: <span className="mono">{shownRoute.normalized_prompt}</span>
              </>
            )}
          </div>
        )}

        <Caveat>
          <strong>Stop</strong> does not stop the arm. It declines to start the next attempt; a motion
          already commanded runs to completion. For an immediate halt, use the physical emergency
          stop — this console has no authority over it, by design.
        </Caveat>
      </Panel>

      <div className="pick-split">
        <Panel title="Viewfinder">
          <Viewfinder showControls />
        </Panel>

        {run ? (
        <Panel
          title={
            <>
              Run <span className="mono">{run.id}</span>
            </>
          }
          aside={
            <>
              <StatusPill status={streamStatus(stream)}>{stream}</StatusPill>{' '}
              <span className="mono">
                {run.succeeded ?? 0}/{run.attempted ?? 0} succeeded
              </span>
            </>
          }
        >
          <div className="run-line">
            <StatusPill status={runTone(run.state)}>{run.state}</StatusPill>{' '}
            <span className="small dim">
              asked for {run.requested_picks} · prompt <span className="mono">{run.prompt || '(none)'}</span>
            </span>
          </div>
          {run.error && (
            <div className="banner error">
              <div className="body">{run.error}</div>
            </div>
          )}

          <div className="stream">
            {events.length === 0 ? (
              <Loading what="the run" />
            ) : (
              events.map((event, i) => (
                <div className={`ev ${event.severity} ${event.type === 'gap' ? 'gap' : ''}`} key={`${event.seq}-${i}`}>
                  <div className="seq">{event.type === 'gap' ? '—' : event.seq}</div>
                  <div className="txt">
                    {event.human}
                    {Object.keys(event.data || {}).length > 0 && (
                      <details className="payload-toggle">
                        <summary>{event.type}</summary>
                        <pre className="payload">{JSON.stringify(event.data, null, 2)}</pre>
                      </details>
                    )}
                  </div>
                </div>
              ))
            )}
            <div ref={listEnd} />
          </div>

          {run.outcomes && run.outcomes.length > 0 && (
            <div className="outcomes">
              <h3>Outcomes</h3>
              <div className="actions">
                {run.outcomes.map((outcome, i) => (
                  <StatusPill key={i} status={outcomeTone(outcome)}>
                    {i + 1}. {outcomeLabel(outcome)}
                  </StatusPill>
                ))}
              </div>
            </div>
          )}
        </Panel>
        ) : (
          <Panel title="Run">
            <div className="empty">
              Nothing is running. Type an instruction above and the cell will narrate every step here
              as it happens.
            </div>
          </Panel>
        )}
      </div>
    </>
  )
}

/** The SOCKET's health, which is not the run's -- a finished run on a closed socket is fine. */
function streamStatus(state: StreamState): string {
  if (state === 'live') return 'ok'
  if (state === 'connecting') return 'info'
  if (state === 'failed') return 'block'
  return 'idle'
}
