/**
 * What Willy says aloud, and when (owner decision 18; build plan 4.2 "Voice output").
 *
 * A short sentence for each key moment of the run on screen, in the UI language: the task's start, its end with the
 * next-instruction question, a problem, a question (an ask card), and the teach's time warning. Nothing else: the chat
 * says the rest, and a cell that talks all the time is a cell nobody listens to.
 *
 * Only what HAPPENS is said: a line older than the moment voice output was switched on (or this page was opened) is a
 * replay of a run already over, and is never read out. Each line is said once per page, whoever mounts the announcer.
 */

import { useEffect, useRef } from 'react'

import type { Lang } from '../i18n'
import type { Msg } from '../i18n/types'
import type { ChatLine } from '../model/runModel'
import { speak } from '../speech/speak'

/** What the announcer needs of a translator: a message in words, and the language to say it in. */
export interface Speaker {
  msg(message: Msg<string>): string
  readonly lang: Lang
}

/** A replay must not talk: lines more than this before voice output came on are not said. */
const LIVE_SLACK_S = 3

const DONE = new Set(['event.run_finished.task', 'event.run_finished.nothing_left', 'event.run_finished.part_limit', 'event.run_finished.stopped_after_part'])

/** The sentence for a key moment, or `null` for a line that is not one. */
export function voiceFor(line: ChatLine): Msg<string> | null {
  const params = line.msg.params ?? {}
  switch (line.msg.key) {
    case 'event.run_started.task':
      return { key: 'ck.voice.start', params: { what: params.what } }
    case 'event.run_finished.problem':
      return { key: 'ck.voice.problem', params: { title: params.title } }
    case 'event.run_finished.ask':
      return { key: 'ck.voice.ask', params: { title: params.title } }
    case 'event.teach.time_warning':
      return { key: 'ck.voice.teachWarning', params: { seconds: params.seconds } }
    case 'chat.next':
      return { key: 'chat.next' }
    default:
      return DONE.has(line.msg.key) ? { key: 'ck.voice.done', params: { parts: params.parts } } : null
  }
}

/** Lines said on this page, by id and time: two announcers (or a remount) never say one twice. */
const said = new Set<string>()

/**
 * Speak the key moments among `lines` while `on`. Mount it where the run's lines are drawn; a second mount says nothing
 * twice.
 */
export function useAnnouncer(lines: readonly ChatLine[], on: boolean, t: Speaker): void {
  /** From when lines count as live: this page's start, or the moment voice output came on. */
  const since = useRef<number | null>(null)
  const wasOn = useRef(on)

  useEffect(() => {
    if (since.current === null || (on && !wasOn.current)) since.current = Date.now() / 1000
    wasOn.current = on
  }, [on])

  useEffect(() => {
    if (!on) return
    const from = since.current ?? Date.now() / 1000
    for (const line of lines) {
      if (line.at < from - LIVE_SLACK_S) continue
      const sentence = voiceFor(line)
      if (!sentence) continue
      const key = `${line.id}@${line.at}`
      if (said.has(key)) continue
      said.add(key)
      speak(t.msg(sentence), t.lang)
    }
  }, [lines, on, t])
}
