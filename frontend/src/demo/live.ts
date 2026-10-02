/**
 * The live picture: `GET /v1/camera/live`, polled (build plan 1.7). Display only: the server peeks the camera and
 * never waits, so this is never a measurement and never holds a grab up.
 *
 * **The last picture is kept while the camera measures** (`reason: measuring`: a pick owns the frames for a moment), so
 * the image does not blink at every look; it is dropped when there is no camera to show (`not_built`, `no_camera`,
 * `no_rig`), because a held frame of a cell that is gone is not "the last frame", it is the wrong cell.
 *
 * **The poll paces itself from the end of the last answer**, never on a fixed interval, so a slow server cannot make
 * requests stack up; a failed answer waits longer before the next try and leaves the picture marked as not live.
 *
 * Shared by the audience window (the projector) and the teach dialog (the wrist camera while a person guides the arm).
 */

import { useEffect, useState } from 'react'

import { api, type LiveFrameOut } from '../api/client'

/** After a failed answer: a long leash rather than a reconnect storm. */
const AFTER_FAILURE_MS = 2000

/** Reasons that mean there is no camera to show: a picture kept from before would be another cell's. */
const NO_CAMERA = new Set(['not_built', 'no_camera', 'no_rig'])

export interface LiveOptions {
  /** A rig id; `null` or unset: the server's primary rig. */
  readonly rig?: string | null
  /** The widest picture asked for, in pixels. */
  readonly maxWidth?: number
  /** Milliseconds between the end of one answer and the next request. */
  readonly intervalMs: number
  /** `false` stops the poll (and keeps nothing). */
  readonly enabled?: boolean
}

export interface LiveView {
  /** The last answer, `null` before the first one and after a failure. */
  readonly frame: LiveFrameOut | null
  /** The last picture, as a data URL: kept across `measuring`, dropped where no camera is. */
  readonly src: string | null
  /** The picture's own width and height (of `src`). */
  readonly size: readonly [number, number] | null
  /** The last request failed: the picture, if any, is not live. */
  readonly failed: boolean
}

const NOTHING: LiveView = { frame: null, src: null, size: null, failed: false }

export function useLiveFrame({ rig = null, maxWidth = 960, intervalMs, enabled = true }: LiveOptions): LiveView {
  const [live, setLive] = useState<LiveView>(NOTHING)

  useEffect(() => {
    if (!enabled) return
    let stopped = false
    let timer: ReturnType<typeof setTimeout> | null = null

    const tick = async () => {
      if (stopped) return
      try {
        const frame = await api.live(rig, maxWidth)
        if (stopped) return
        setLive((previous) => {
          if (frame.image_base64) {
            return {
              frame,
              src: `data:image/jpeg;base64,${frame.image_base64}`,
              size: [frame.width, frame.height],
              failed: false,
            }
          }
          if (NO_CAMERA.has(frame.reason)) return { frame, src: null, size: null, failed: false }
          // `measuring` (and an encode that failed once): the last picture stays.
          return { frame, src: previous.src, size: previous.size, failed: false }
        })
        timer = setTimeout(() => void tick(), intervalMs)
      } catch {
        if (stopped) return
        setLive((previous) => ({ ...previous, frame: null, failed: true }))
        timer = setTimeout(() => void tick(), Math.max(intervalMs, AFTER_FAILURE_MS))
      }
    }

    void tick()
    return () => {
      stopped = true
      if (timer) clearTimeout(timer)
      setLive(NOTHING)
    }
  }, [rig, maxWidth, intervalMs, enabled])

  return live
}
