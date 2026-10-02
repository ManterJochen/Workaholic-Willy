/**
 * The console's one confirm host: `const ok = await confirm({...})` before a Restart or a Home move (build plan
 * item 12; App mounts the host, the cockpit and the stop card ask it).
 *
 * The answer is `true` only for a click on the confirm button of the dialog that asked it. Cancel, Escape, a click
 * beside the dialog, a newer request replacing an open one, an unmounted host (a request open when it goes, or asked
 * after) and a component rendered with no host at all are each `false`, so a caller that moves the arm on `true` moves
 * it only after a person clicked the button that says it moves.
 *
 * **Every answer is decided outside React's state updates**, in a small store the host owns (`ConfirmStore`): a newer
 * request answers the one it replaces `false` at once, and each dialog's buttons are bound to their own request, which
 * they answer only while it is still the open one. The dialog on screen is read from the store with
 * `useSyncExternalStore`, whose updates are never deferred. Resolving inside a `setState` updater is not safe here:
 * React runs an updater again when it rebases a skipped lower-priority update, so a click on the dialog on screen
 * could resolve a request that replaced it a moment earlier and was never drawn (a click on "Home" answering a
 * "Restart" nobody saw).
 *
 * Written without JSX (`.ts`) so the provider and its hook can share one module (see `i18n/index.ts`).
 */

import { createContext, createElement, useContext, useEffect, useState, useSyncExternalStore, type ReactNode } from 'react'

import ConfirmDialog, { type ConfirmDialogProps } from '../components/ConfirmDialog'

/** What the dialog says: the title, the first motion, the countdown and the confirm's own label. */
export type ConfirmOptions = Omit<ConfirmDialogProps, 'open' | 'busy' | 'onConfirm' | 'onCancel'>

export type Confirm = (options: ConfirmOptions) => Promise<boolean>

interface Pending {
  /** Each request gets a fresh dialog: nothing of an earlier question's state carries over. */
  readonly id: number
  readonly options: ConfirmOptions
  readonly resolve: (answer: boolean) => void
}

/**
 * The open request and every answer to it. One per host. `ask` and `answer` run in event handlers and effects, never
 * while React renders; the host only reads `snapshot`.
 */
class ConfirmStore {
  private open: Pending | null = null
  private ids = 0
  /** Between the host's unmount and its next mount: every request answers "no" at once. */
  private closed = false
  private readonly listeners = new Set<() => void>()

  readonly subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  /** The request whose dialog is on screen, or `null`. */
  readonly snapshot = (): Pending | null => this.open

  /** Ask a person. One question at a time: the one this replaces is answered "no" now, before anything is drawn. */
  readonly ask: Confirm = (options) =>
    new Promise<boolean>((resolve) => {
      if (this.closed) {
        resolve(false)
        return
      }
      this.ids += 1
      const replaced = this.open
      this.open = { id: this.ids, options, resolve }
      replaced?.resolve(false)
      this.emit()
    })

  /** Answer `request`, and only while it is the open one: a dialog's button never answers a request it did not ask. */
  readonly answer = (request: Pending, yes: boolean): void => {
    if (this.open !== request) return
    this.open = null
    request.resolve(yes)
    this.emit()
  }

  /** The host went away: what is open answers "no", and so does anything asked until it mounts again. */
  readonly close = (): void => {
    this.closed = true
    const left = this.open
    this.open = null
    left?.resolve(false)
    this.emit()
  }

  readonly reopen = (): void => {
    this.closed = false
  }

  private emit(): void {
    for (const listener of this.listeners) listener()
  }
}

const ConfirmContext = createContext<Confirm | null>(null)

const NEVER: Confirm = async () => false

export function ConfirmProvider({ children }: { children?: ReactNode }) {
  const [store] = useState(() => new ConfirmStore())
  const pending = useSyncExternalStore(store.subscribe, store.snapshot, store.snapshot)

  // A host that goes away leaves nothing waiting (and Strict Mode's rehearsal of an unmount reopens it after).
  useEffect(() => {
    store.reopen()
    return store.close
  }, [store])

  return createElement(
    ConfirmContext.Provider,
    { value: store.ask },
    children,
    pending
      ? createElement(ConfirmDialog, {
          ...pending.options,
          key: pending.id,
          open: true,
          onConfirm: () => store.answer(pending, true),
          onCancel: () => store.answer(pending, false),
        })
      : null,
  )
}

/** Ask a person before a motion. Without a host the answer is always `false`: nothing moves unconfirmed. */
export function useConfirm(): Confirm {
  return useContext(ConfirmContext) ?? NEVER
}
