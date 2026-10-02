/**
 * The confirm before a motion that is not Start: Restart and Home (build plan item 12).
 *
 * Start is its own confirmation (its label names the first motion); Restart and Home keep a dialog, because they are
 * reached from a stop, when a person may be standing at the cell. This dialog says, in this order:
 *
 * 1. the first motion, in its own band ("Erste Bewegung: geplante Fahrt nach Home."), before any button;
 * 2. the 3 s hands-off countdown, when one is due (`CellOut.countdown_due`);
 * 3. that a planned move may begin with a straight leg of at most 10 degrees per joint (the planner's band).
 *
 * And it is hard to confirm by accident: the focus starts on Cancel, not on the motion, so Enter does not move the
 * arm; Escape and a click beside the dialog cancel; a second click on the confirm is not a second request. The
 * confirm is drawn in the lime accent, like every control that starts motion.
 */

import { useEffect, useId, useRef, useState, type KeyboardEvent, type ReactNode } from 'react'

import { Icon } from '../icons'
import { useT } from '../i18n'

export interface ConfirmDialogProps {
  open: boolean
  title: string
  /** The first motion, in words ("geplante Fahrt nach Home"). Every dialog that confirms a motion names one. */
  firstMotion?: string
  /** A 3 s hands-off countdown comes before the first motion. */
  countdown?: boolean
  /** The note on the straight leg a planned move may begin with. On by default whenever a first motion is named. */
  straightLeg?: boolean
  /** More lines: a checklist, the put-back fallback. */
  children?: ReactNode
  confirmLabel: string
  /** The confirm starts motion: the dialog carries the alert icon. */
  moves?: boolean
  /** A request is on its way: the confirm waits. */
  busy?: boolean
  onConfirm: () => void
  onCancel: () => void
}

const FOCUSABLE = 'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'

export default function ConfirmDialog({
  open,
  title,
  firstMotion,
  countdown = false,
  straightLeg,
  children,
  confirmLabel,
  moves = false,
  busy = false,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const t = useT()
  const titleId = useId()
  const bodyId = useId()
  const cancelRef = useRef<HTMLButtonElement | null>(null)
  const dialogRef = useRef<HTMLDivElement | null>(null)
  // One confirm per opening: a double click must not send the motion twice. The state disables the button; the ref
  // also holds within one tick, before React has re-rendered.
  const [fired, setFired] = useState(false)
  const firedRef = useRef(false)
  const [openedAt, setOpenedAt] = useState(open)

  if (open !== openedAt) {
    setOpenedAt(open)
    if (open) setFired(false)
  }

  useEffect(() => {
    if (open) {
      firedRef.current = false
      cancelRef.current?.focus()
    }
  }, [open])

  if (!open) return null

  const showLeg = straightLeg ?? Boolean(firstMotion)

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') {
      event.stopPropagation()
      onCancel()
      return
    }
    if (event.key !== 'Tab' || !dialogRef.current) return
    // Keep the focus inside the dialog while it is open.
    const items = Array.from(dialogRef.current.querySelectorAll<HTMLElement>(FOCUSABLE))
    if (items.length === 0) return
    const first = items[0]
    const last = items[items.length - 1]
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault()
      last.focus()
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault()
      first.focus()
    }
  }

  const confirm = () => {
    if (firedRef.current || fired || busy) return
    firedRef.current = true
    setFired(true)
    onConfirm()
  }

  return (
    <div className="dialog-backdrop" data-testid="dialog-backdrop" onClick={onCancel}>
      <div
        ref={dialogRef}
        className="dialog hud"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={bodyId}
        onClick={(event) => event.stopPropagation()}
        onKeyDown={onKeyDown}
      >
        <div className="dialog-head">
          <Icon name={moves ? 'alert' : 'info'} size={22} className="icon" />
          <h2 id={titleId}>{title}</h2>
        </div>
        <div className="dialog-body" id={bodyId}>
          {firstMotion && (
            <div className="dialog-motion">
              <Icon name="bolt" size={18} />
              <span>{t('confirm.firstMotion', { motion: firstMotion })}</span>
            </div>
          )}
          {(countdown || showLeg) && (
            <ul className="dialog-notes">
              {countdown && <li>{t('confirm.countdown')}</li>}
              {showLeg && <li>{t('confirm.straightLeg')}</li>}
            </ul>
          )}
          {children}
        </div>
        <div className="dialog-foot">
          <button ref={cancelRef} type="button" className="ghost big" onClick={onCancel}>
            {t('confirm.cancel')}
          </button>
          <button
            type="button"
            className="primary big"
            disabled={busy || fired}
            onClick={confirm}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
