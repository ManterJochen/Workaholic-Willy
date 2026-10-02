/**
 * Voice output (owner decision 18): the browser's own speech synthesis, no dependency, OFF unless a person switched it
 * on (`willy.voiceOut`, the top bar's speaker button).
 *
 * Short sentences at the key moments only: a task's start ("Ich greife den grünen Würfel"), its end ("Fertig, 7
 * Teile. Was soll ich als Nächstes tun?"), a problem ("Problem: Angehalten"), a question, and the teach's time warning.
 * Which moments those are is the cockpit's (`cockpit/announce.ts`); this file only says a sentence in a language.
 *
 * The voice is picked by the UI language: a voice of that exact locale first (de-DE, en-GB), else any of the language,
 * else the browser's default with the utterance's `lang` set. Which German voices the cell PC has is unknown (the
 * checklist of build plan 6.6 covers it). A browser without speech synthesis says nothing and nothing fails.
 */

import type { Lang } from '../i18n'

/** The locale each UI language speaks in: the same tags `Intl` formats with (`i18n/index.ts`). */
export const VOICE_LOCALE: Record<Lang, string> = { de: 'de-DE', en: 'en-GB' }

/** Whether this browser can speak at all. */
export function speechAvailable(): boolean {
  return (
    typeof window !== 'undefined' &&
    typeof window.speechSynthesis !== 'undefined' &&
    typeof window.SpeechSynthesisUtterance !== 'undefined'
  )
}

/** The voice for `lang`: its exact locale, else the language, else none (the browser then picks by `lang`). */
export function pickVoice(voices: readonly SpeechSynthesisVoice[], lang: Lang): SpeechSynthesisVoice | null {
  const locale = VOICE_LOCALE[lang].toLowerCase()
  const exact = voices.find((voice) => voice.lang.toLowerCase().replace('_', '-') === locale)
  if (exact) return exact
  return voices.find((voice) => voice.lang.toLowerCase().startsWith(lang)) ?? null
}

/**
 * Say `text` in `lang`, queued after whatever is being said. Returns whether it was queued. Never throws: a console
 * whose speech engine fails keeps working, silently.
 */
export function speak(text: string, lang: Lang): boolean {
  if (!speechAvailable() || text.trim() === '') return false
  try {
    const utterance = new window.SpeechSynthesisUtterance(text)
    utterance.lang = VOICE_LOCALE[lang]
    const voice = pickVoice(window.speechSynthesis.getVoices?.() ?? [], lang)
    if (voice) utterance.voice = voice
    window.speechSynthesis.speak(utterance)
    return true
  } catch {
    return false
  }
}

/** Stop speaking now: what is being said, and what is queued. */
export function stopSpeaking(): void {
  try {
    if (speechAvailable()) window.speechSynthesis.cancel()
  } catch {
    /* nothing is speaking that could be stopped */
  }
}
