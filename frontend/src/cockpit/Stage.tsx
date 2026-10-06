/**
 * The stage: the live camera image, the largest thing in the cockpit (OD 2, OD 11; build plan 1.7, 4.2).
 *
 * **Live, never a measurement.** `GET /v1/camera/live` reads through `Camera.peek`, which never waits and never takes
 * a frame from a pick. It is polled at 10 Hz, 4 Hz while a pose is taught, each tick scheduled from the end of the last
 * so a slow server never stacks requests. While a pick grabs a frame (`measuring`) the last image stays, and the badge
 * says "misst …"; a cell that is gone (not built, no camera) clears it, because a held image would then show another
 * cell's workspace.
 *
 * **What the badge says, always:** LIVE (a camera), PROBE (the rehearsal cell's drawing: there is no camera at all) or
 * KEIN BILD with the reason; the camera's name and the frame's age beside it. A cell with more than one camera offers
 * them as buttons.
 *
 * **An overlay is its own image** (item 15): what the camera found, as it was seen, is pinned over the stage for 5 s
 * with a band that says when, and, on a wrist camera, that the arm has moved since. It is never drawn over the moving
 * live picture. Only an overlay that is new when it arrives is pinned: a run replayed after a reload pins nothing.
 * The camera's TARGET (the bin a task places into: its mask, outline and middle, `draw_located`) is pinned in both
 * views. The GRASP overlays the run captures today are the grasp generator's DEBUG render (its telemetry header, score
 * bars, up to five candidates), so they are pinned in the tech view only (`pinGrasps`); the demo view keeps the calm
 * live image and shows the render as the part card's thumbnail, until the capture renders a presentation overlay (the
 * chosen grasp, no telemetry).
 *
 * A camera chosen here that the cell no longer has (a rebuild with one camera fewer: `no_rig`) gives way to the
 * cell's own camera at once, rather than leaving the stage on "no such camera" with no button to leave it.
 *
 * **The banners** say what stops the arm, over the dimmed image: the hands-off countdown, a halt, a stopped controller.
 */

import { useEffect, useRef, useState } from 'react'

import { ApiError, api, type LiveFrameOut } from '../api/client'
import { useT } from '../i18n'
import { Icon } from '../icons'
import type { OverlayPin } from '../model/runModel'
import { useNow } from './hooks'
import { COCKPIT } from './i18n'

/** 10 Hz; 4 Hz while a person guides the arm (the teach's own poll is the heartbeat, this is only the picture). */
const POLL_MS = 100
const TEACH_POLL_MS = 250
/** A camera that answers no picture for this long is gone; a shorter gap keeps the last picture on the stage. */
const HOLD_MS = 2000
/** While the stream plays, the stage only asks for what the badges say (camera, age, measuring). */
const STATUS_POLL_MS = 500

/** The camera's live MJPEG stream (`GET /v1/camera/live.mjpeg`), about 30 frames a second. */
function streamUrl(rig: string | null): string {
  return `/v1/camera/live.mjpeg${rig ? `?rig=${encodeURIComponent(rig)}` : ''}`
}
/** A server that does not answer gets a long leash, not a request storm. */
const AFTER_FAILURE_MS = 2000
/** How long a new overlay is pinned over the stage. */
const PIN_S = 5
/** An overlay older than this when it is first seen is a replay (a reloaded page): it is not pinned. */
const FRESH_S = 10

export type StageBanner =
  | { readonly kind: 'countdown'; readonly seconds: number | null }
  | { readonly kind: 'halted' }
  | { readonly kind: 'stopped' }

export interface StageProps {
  /** A pose is being taught: the picture is asked for at half the rate. */
  readonly teaching: boolean
  /** The run's overlays, oldest first (`RunView.overlays`). */
  readonly overlays: readonly OverlayPin[]
  /** The cell's camera rides on the wrist: an overlay's band says the arm has moved since. */
  readonly wrist: boolean
  readonly banner: StageBanner | null
  /** Pin a new GRASP overlay over the stage for a moment too (the tech view: today's are the debug render). */
  readonly pinGrasps: boolean
}

/** What the last answer let the stage show. */
interface Shown {
  readonly frame: LiveFrameOut | null
  readonly offline: boolean
}

export default function Stage({ teaching, overlays, wrist, banner, pinGrasps }: StageProps) {
  const t = useT(COCKPIT)
  const [rig, setRig] = useState<string | null>(null)
  const [shown, setShown] = useState<Shown>({ frame: null, offline: false })
  /** The last picture that arrived: kept through `measuring`, dropped when the cell is gone. */
  const [image, setImage] = useState<string | null>(null)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lastPicture = useRef(0)
  /** The camera's MJPEG stream plays in the stage; a stream the browser could not open falls back to the polled picture. */
  const [streamFailed, setStreamFailed] = useState(false)
  const streaming = !streamFailed && shown.frame?.source === 'camera' && image !== null
  const streamingRef = useRef(false)
  useEffect(() => {
    streamingRef.current = streaming
  }, [streaming])

  useEffect(() => {
    let stopped = false
    const tick = async () => {
      if (stopped) return
      try {
        const frame = await api.live(rig)
        if (stopped) return
        if (rig !== null && frame.reason === 'no_rig') {
          // The camera chosen here is gone: back to the cell's own (the effect asks again with no rig named).
          setRig(null)
          return
        }
        if (frame.image_base64) {
          // Decoded before it is shown: the old picture stays until the new one can be painted, so no blank frame.
          const src = `data:image/jpeg;base64,${frame.image_base64}`
          const next = new Image()
          next.src = src
          try {
            await next.decode()
          } catch {
            /* a picture that does not decode is skipped; the last one stays */
          }
          if (stopped) return
          lastPicture.current = Date.now()
          setImage(src)
          setShown({ frame, offline: false })
        } else {
          // A camera that skips a peek for a moment keeps its last picture; only a lasting gap, or a cell that is not
          // there (not built, no rig), clears it.
          const gone = frame.reason === 'not_built' || frame.reason === 'no_rig'
          const stale = Date.now() - lastPicture.current > HOLD_MS
          if (gone || (stale && frame.reason !== 'measuring')) {
            setImage(null)
            setShown({ frame, offline: false })
          } else {
            // `measuring` is said on the badge; any other short gap keeps what the last picture said.
            setShown((prev) => (frame.reason === 'measuring' || !prev.frame ? { frame, offline: false } : prev))
          }
        }
        timer.current = setTimeout(tick, streamingRef.current ? STATUS_POLL_MS : teaching ? TEACH_POLL_MS : POLL_MS)
      } catch (err: unknown) {
        if (stopped) return
        // A route this server does not have, or no server at all: say so, and ask again slowly.
        const gone = !(err instanceof ApiError) || err.httpStatus === 0 || err.httpStatus === 404
        setShown({ frame: null, offline: gone })
        if (gone) setImage(null)
        timer.current = setTimeout(tick, AFTER_FAILURE_MS)
      }
    }
    void tick()
    return () => {
      stopped = true
      if (timer.current) clearTimeout(timer.current)
      timer.current = null
    }
  }, [rig, teaching])

  const pin = usePin(overlays, pinGrasps)
  const frame = shown.frame
  const rigs = frame?.rigs ?? []
  const current = rig ?? frame?.rig_id ?? null
  const measuring = frame?.reason === 'measuring'
  const badge = shown.offline || !frame
    ? 'none'
    : frame.source === 'camera' || (measuring && image)
      ? 'live'
      : frame.source === 'synthetic'
        ? 'probe'
        : 'none'
  const reason = shown.offline ? 'offline' : frame?.reason && frame.reason !== 'measuring' ? frame.reason : 'none'
  const age = frame?.age_s != null && image ? Math.round(frame.age_s * 10) / 10 : null
  const dim = banner !== null

  return (
    <section className={`ck-stage stage hud-frame${dim ? ' has-banner' : ''}`} aria-label={t('ck.stage.label')}>
      {image ? (
        <img
          className={`ck-frame${dim ? ' dim' : ''}`}
          src={streaming ? streamUrl(current) : image}
          onError={streaming ? () => setStreamFailed(true) : undefined}
          alt={t('ck.stage.alt', { rig: current ?? (frame?.source === 'synthetic' ? t('ck.stage.probe') : '—') })}
          decoding="async"
        />
      ) : (
        <div className="ck-blank">
          <Icon name="camera" size={34} />
          <span>{t(`ck.stage.reason.${reason}` as 'ck.stage.reason.none')}</span>
        </div>
      )}

      <div className="ck-badges">
        <span className={`ck-badge ${badge}`}>
          {badge === 'live' && <span className="ck-dot" aria-hidden="true" />}
          {t(badge === 'live' ? 'ck.stage.live' : badge === 'probe' ? 'ck.stage.probe' : 'ck.stage.none')}
        </span>
        {current && <span className="ck-badge">{current}</span>}
        {age !== null && !measuring && <span className="ck-badge">{t('ck.stage.age', { age })}</span>}
        {measuring && image && <span className="ck-badge warn">{t('ck.stage.measuring')}</span>}
      </div>

      {rigs.length > 1 && (
        <div className="ck-rigs" role="group" aria-label={t('ck.stage.cameras')}>
          {rigs.map((r) => (
            <button
              key={r.rig_id}
              type="button"
              className="ck-rig"
              aria-pressed={r.rig_id === current}
              onClick={() => setRig(r.rig_id)}
            >
              {r.rig_id}
              <span className="ck-rig-kind">{t(r.mounting === 'wrist' ? 'ck.stage.wrist' : 'ck.stage.fixed')}</span>
            </button>
          ))}
        </div>
      )}

      {pin && (
        <figure className="ck-pin">
          <img src={pin.url} alt={`${t(pin.kind === 'grasp' ? 'ck.stage.pinGrasp' : 'ck.stage.pinTarget')}: ${t(pinWhen(pin.kind, false), { time: t.fmt.time(pin.at) })}`} />
          <figcaption>
            <span className="ck-pin-kind">{t(pin.kind === 'grasp' ? 'ck.stage.pinGrasp' : 'ck.stage.pinTarget')}</span>
            <span>{t(pinWhen(pin.kind, wrist), { time: t.fmt.time(pin.at) })}</span>
          </figcaption>
        </figure>
      )}

      {banner && (
        <div className={`ck-banner ${banner.kind}`} role="status">
          <strong>
            {banner.kind === 'countdown'
              ? t('ck.stage.countdown', { seconds: banner.seconds ?? '…' })
              : t(banner.kind === 'halted' ? 'ck.stage.halted' : 'ck.stage.stopped')}
          </strong>
          <span>
            {t(banner.kind === 'countdown' ? 'ck.stage.countdownSub' : banner.kind === 'halted' ? 'ck.stage.haltedSub' : 'ck.stage.stoppedSub')}
          </span>
        </div>
      )}
    </section>
  )
}

/** When the pinned image was made: a grasp "as decided", a target "as seen"; on a wrist camera the arm has moved since. */
function pinWhen(kind: OverlayPin['kind'], wrist: boolean) {
  if (kind === 'grasp') return wrist ? 'ck.stage.pinnedMoved' : 'ck.stage.pinned'
  return wrist ? 'ck.stage.seenMoved' : 'ck.stage.seen'
}

/**
 * The overlay pinned now: the newest one that may be pinned (a target always, a grasp where `grasps`), for `PIN_S`
 * seconds after it arrived, if it was new when it arrived.
 */
function usePin(overlays: readonly OverlayPin[], grasps: boolean): OverlayPin | null {
  const latest = [...overlays].reverse().find((overlay) => overlay.kind === 'target' || grasps) ?? null
  const [pinned, setPinned] = useState<{ pin: OverlayPin; until: number } | null>(null)
  const url = latest?.url ?? null
  useEffect(() => {
    if (!latest || url === null) return
    const nowS = Date.now() / 1000
    if (nowS - latest.at > FRESH_S) return
    setPinned({ pin: latest, until: nowS + PIN_S })
    // Keyed on the overlay's address: a new overlay pins anew, a re-render of the same one does not.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url])
  const now = useNow(500, pinned !== null)
  // A grasp pinned in the tech view goes the moment the view is switched to the demo.
  if (!pinned || now > pinned.until || (pinned.pin.kind === 'grasp' && !grasps)) return null
  return pinned.pin
}
