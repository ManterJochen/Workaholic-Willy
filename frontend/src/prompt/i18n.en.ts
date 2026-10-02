/**
 * The prompt box's words, in English: the switch, and what the box says when no provider wraps it.
 *
 * Typed by the German file: a key missing here, or one German does not have, fails `tsc`.
 */

import type de from './i18n.de'

const en: Record<keyof typeof de, string> = {
  'prompt.field': 'Command for Willy',
  'prompt.placeholder': 'What should Willy do?',
  'prompt.mic.speak': 'Speak',
  'prompt.mic.stop': 'Stop recording',
  'prompt.mic.transcribing': 'Transcribing…',
  'prompt.mic.title':
    'Click to speak, click again to stop. Holding {key} records while held, so a foot switch that sends {key} works too. The text lands in the box; nothing moves until you press Start.',
  'prompt.level': 'Level',
  'prompt.noMic': 'No microphone here: {why} Type the command instead.',
  'prompt.noMic.insecure': 'The microphone needs HTTPS or localhost; this page was loaded over plain HTTP from another machine.',
  'prompt.noMic.unexposed': 'This browser does not expose a microphone to the page.',
  'prompt.noMic.no_web_audio': 'This browser has no Web Audio support.',
  'prompt.micRefused': 'The microphone could not be opened: {why}',
  'prompt.tooShort': 'That was too short to transcribe: speak a little longer before you stop.',
  'prompt.nothingHeard': 'Nothing was recognised in that recording.',
  'prompt.spoken': 'This was transcribed from speech, not typed: read it before you start. The arm moves on what was heard.',
  'prompt.send': 'Send',
  'prompt.sendTitle': 'Enter or Send reads the sentence. Nothing starts until Start on the card.',
}

export default en
