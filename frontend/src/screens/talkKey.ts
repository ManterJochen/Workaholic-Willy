/**
 * The talk key: the key a foot switch sends, held down to talk (build plan item 21), and stored per browser.
 *
 * **Only keys that typing never uses.** The cockpit listens for the talk key on the whole window and takes it for the
 * microphone (`preventDefault`), so a key that typing or moving around the page needs would stop working there: Enter
 * would no longer send a command and would open the microphone instead, Shift would talk at every capital letter, Tab
 * would stop the keyboard's way through the page. A foot switch sends a function key (F1 to F24) or one of the keys
 * nobody types with (Pause, Scroll Lock), so those are the keys a talk key may be; every other key is refused, with its
 * name, and the default stays F8.
 */

/** The key the cockpit's microphone reads (`prompt/PromptInput.tsx` reads the same key), and its default. */
export const TALK_KEY_STORAGE = 'willy.talkKey'
export const DEFAULT_TALK_KEY = 'F8'

/** F1 to F24: what a foot switch sends, and what nobody types with. */
const FUNCTION_KEY = /^F([1-9]|1[0-9]|2[0-4])$/

/** The two other keys a foot switch may send and typing never uses. */
const QUIET_KEYS = new Set(['Pause', 'ScrollLock'])

/** Whether `key` (a `KeyboardEvent.key`) may be the talk key: a function key, Pause or Scroll Lock, nothing else. */
export function isTalkKey(key: string): boolean {
  return FUNCTION_KEY.test(key) || QUIET_KEYS.has(key)
}

/**
 * The stored talk key as it is stored (the cockpit reads the same value), or the default where none is stored or
 * storage is locked. A stored key that may not be one is returned as it is, so the page can say so rather than pretend.
 */
export function readTalkKey(): string {
  try {
    return localStorage.getItem(TALK_KEY_STORAGE) || DEFAULT_TALK_KEY
  } catch {
    return DEFAULT_TALK_KEY
  }
}

/** Store the talk key, or forget it (`null`: back to the default). A browser without storage keeps the default. */
export function writeTalkKey(key: string | null): void {
  try {
    if (key === null) localStorage.removeItem(TALK_KEY_STORAGE)
    else localStorage.setItem(TALK_KEY_STORAGE, key)
  } catch {
    /* a browser without storage keeps the default */
  }
}
