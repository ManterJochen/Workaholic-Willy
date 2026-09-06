/**
 * What the robot is looking at — and, in the same glance, which of four things that is.
 *
 * The backend can hand back a colour frame off a device, a SYNTHETIC scene a rehearsal cell drew
 * (there is no camera in that room at all), or the grasp overlay it rendered during a pick. All three
 * are pictures of a workspace and they mean very different things. So the badge is never hidden —
 * `live` · `rehearsal` · `overlay` · `held` — and the sentence under the picture is the backend's own,
 * never one this component composes.
 *
 * **Poll, do not stream.** Every no-picture state here is something an operator has to READ — the cell
 * is not built, a pick owns the camera, this source cannot be peeked at — and a socket would have to
 * express those as the absence of frames. A poll makes each tick a complete answer.
 *
 * **The poll paces itself.** Six a second while there is a live camera frame, once a second for a
 * rehearsal drawing that will never change, and once a second when there is nothing to see. It also
 * schedules the next tick from the END of the last one rather than on an interval, so a slow backend
 * cannot make requests stack up.
 *
 * ── THE FAILURE THIS COMPONENT EXISTS TO PREVENT, AND HOW IT NEARLY SHIPPED ────────────────────────
 * A viewfinder's job is to be trustworthy about liveness. The first cut of this file failed at exactly
 * that: the catch branch set an error but left `frame` untouched, so when the backend died the last
 * good image stayed on screen, undimmed, still badged **live**, forever. Every derived flag now comes
 * from a single `frame` that is CLEARED on failure, which makes the honest rendering the automatic
 * one — `held`, dimmed, with the error where the backend's sentence was.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiError, api, type ViewfinderOut } from '../api/client'

/**
 * How fast to ask, by what came back. A rehearsal scene is a fixed drawing that will never change, so
 * polling it six times a second would burn a request per frame to re-encode identical bytes.
 */
const CADENCE_MS = { camera: 160, synthetic: 1000, overlay: 500, none: 1000 } as const
/** A dead backend gets a long leash rather than a reconnect storm. */
const AFTER_FAILURE_MS = 2000

function cadenceFor(source: string): number {
  return CADENCE_MS[source as keyof typeof CADENCE_MS] ?? CADENCE_MS.none
}

export interface ViewfinderProps {
  /** Rendered under the picture. Off on the demo page, where there is nothing to operate. */
  showControls?: boolean
  /** Stops the poll entirely, e.g. on a screen where the panel is collapsed. */
  paused?: boolean
}

export function Viewfinder({ showControls = false, paused = false }: ViewfinderProps) {
  const [frame, setFrame] = useState<ViewfinderOut | null>(null)
  const [pollError, setPollError] = useState<string | null>(null)
  // The toggle's failures live in their own state. Sharing one with the poll meant a refused
  // POST /v1/overlay/enable was wiped by the next successful tick 160 ms later -- an operator clicked
  // a box, the box sprang back, and the reason why flashed past too fast to read.
  const [actionError, setActionError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  // The last picture that actually arrived. Kept separate from `frame` so a tick answering "no
  // picture" replaces the STATUS at once while the image stays on screen, dimmed and labelled `held`
  // -- a viewfinder that blanks the moment a pick starts is a viewfinder nobody trusts.
  const [held, setHeld] = useState<string | null>(null)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  // The newest frame, readable without re-creating the toggle's callback on every tick. A state
  // updater cannot be used to read current state -- React requires updaters to be pure, and in Strict
  // Mode it calls them twice.
  const latest = useRef<ViewfinderOut | null>(null)

  useEffect(() => {
    if (paused) return
    let stopped = false

    const tick = async () => {
      if (stopped) return
      try {
        const next = await api.camera()
        if (stopped) return
        latest.current = next
        setFrame(next)
        setPollError(null)
        if (next.image_base64) {
          setHeld(`data:${next.media_type || 'image/jpeg'};base64,${next.image_base64}`)
        } else if (next.reason === 'not_built') {
          // The cell was torn down. Holding its last frame would show one cell's workspace while the
          // console is describing another -- the one case where a held image is not merely stale but
          // about the wrong machine.
          setHeld(null)
        }
        timer.current = setTimeout(tick, cadenceFor(next.source))
      } catch (err) {
        if (stopped) return
        setPollError(err instanceof ApiError ? err.message : String(err))
        // Load-bearing: without this the last frame keeps its `live` badge while the backend is gone.
        latest.current = null
        setFrame(null)
        timer.current = setTimeout(tick, AFTER_FAILURE_MS)
      }
    }

    void tick()
    return () => {
      stopped = true
      if (timer.current) clearTimeout(timer.current)
      timer.current = null
    }
  }, [paused])

  const toggleOverlay = useCallback(async () => {
    setBusy(true)
    setActionError(null)
    try {
      // Read the wanted value off the LATEST frame rather than a click-time snapshot, and write the
      // answer back functionally: a poll lands every 160 ms, so a captured copy is already stale by
      // the time the round trip returns.
      const result = await api.enableOverlay(!latest.current?.overlay_enabled)
      setFrame((current) => (current ? { ...current, overlay_enabled: result.enabled } : current))
    } catch (err) {
      setActionError(err instanceof ApiError ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }, [])

  // Everything below is derived from ONE `frame`, which is null whenever the last answer failed.
  const stale = Boolean(held) && !frame?.image_base64
  const badge = badgeFor(frame?.source, Boolean(held), stale)

  return (
    <div className="viewfinder">
      <div className={`vf-stage${stale ? ' stale' : ''}`}>
        {held ? (
          <img src={held} alt="" decoding="async" />
        ) : (
          <div className="vf-blank">
            <span className="vf-blank-mark" aria-hidden="true" />
            <span>{pollError ? 'No answer from the cell.' : 'No picture'}</span>
          </div>
        )}

        <div className="vf-badges">
          <span className={`pill ${badge.tone}`}>{badge.label}</span>
          {frame?.source === 'overlay' && frame.age_s !== null && frame.age_s !== undefined && (
            <span className="pill info">{formatAge(frame.age_s, frame.age_is_exact !== false)}</span>
          )}
        </div>

        {frame?.width ? (
          <div className="vf-dims mono">
            {frame.width}×{frame.height}
          </div>
        ) : null}
      </div>

      <p className="vf-says">{pollError ?? frame?.human ?? 'Asking the cell…'}</p>

      {showControls && (
        <div className="vf-controls">
          <label className="check">
            <input
              type="checkbox"
              checked={Boolean(frame?.overlay_enabled)}
              disabled={busy || !frame || frame.reason === 'not_built'}
              onChange={() => void toggleOverlay()}
            />
            draw the grasp overlay
          </label>
          <span className="small faint">
            Costs render time inside each pick, which is why it is off by default.
          </span>
          {actionError && <span className="small vf-refused">{actionError}</span>}
        </div>
      )}
    </div>
  )
}

/**
 * The one label that says what you are looking at.
 *
 * `rehearsal` gets `bench` — this console's colour for "a simulator produced this" — rather than a
 * green `live`. A rehearsal cell has no camera; the frame is a drawing, and the badge is the fastest
 * place to say so. `held` wins over everything, because a picture whose source has stopped answering
 * is not describing the present whatever it used to be.
 */
function badgeFor(
  source: string | undefined,
  hasImage: boolean,
  stale: boolean,
): { label: string; tone: string } {
  if (stale && hasImage) return { label: 'held', tone: 'warn' }
  switch (source) {
    case 'camera':
      return { label: 'live', tone: 'ok' }
    case 'synthetic':
      return { label: 'rehearsal', tone: 'bench' }
    case 'overlay':
      return { label: 'overlay', tone: 'info' }
    default:
      return hasImage ? { label: 'held', tone: 'warn' } : { label: 'no signal', tone: 'idle' }
  }
}

/**
 * Ages are read at a glance, so they are words, not a float — and sometimes they are only a floor.
 *
 * Nothing in the stack stamps a render time on the overlay, so the backend can date one only from when
 * it watched the image change. The first overlay a server sees may predate it, and the payload says so
 * with `age_is_exact: false`. This renders that as a **≥**, because "just now" over a picture that
 * might be an hour old is the exact failure this whole component is about.
 */
function formatAge(seconds: number, exact: boolean): string {
  const rounded =
    seconds < 1
      ? exact
        ? 'just now'
        : 'age unknown'
      : seconds < 60
        ? `${Math.round(seconds)}s old`
        : seconds < 3600
          ? `${Math.round(seconds / 60)}m old`
          : `${Math.round(seconds / 3600)}h old`
  return exact || seconds < 1 ? rounded : `≥ ${rounded}`
}
