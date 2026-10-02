/**
 * The prompt box's words, in German: the console's default language.
 *
 * This file DEFINES the prompt area's keys; `i18n.en.ts` is typed by it, so an English text that is missing fails
 * `tsc`. Placeholders are `{name}`. Every text here is factual: what the box does, what it heard, and that nothing
 * moves until a person presses Start.
 */

const de = {
  'prompt.field': 'Befehl an Willy',
  'prompt.placeholder': 'Was soll Willy tun?',
  'prompt.mic.speak': 'Sprechen',
  'prompt.mic.stop': 'Aufnahme beenden',
  'prompt.mic.transcribing': 'Wird erkannt …',
  'prompt.mic.title':
    'Klicken zum Sprechen, nochmal klicken zum Beenden. {key} gedrückt halten nimmt auf, solange die Taste gehalten wird: so geht auch ein Fußschalter, der {key} sendet. Der Text landet im Feld; bewegt wird nichts, bis du auf Start drückst.',
  'prompt.level': 'Pegel',
  'prompt.noMic': 'Kein Mikrofon hier: {why} Bitte den Befehl tippen.',
  'prompt.noMic.insecure': 'Das Mikrofon braucht HTTPS oder localhost; diese Seite kam über einfaches HTTP von einem anderen Rechner.',
  'prompt.noMic.unexposed': 'Dieser Browser gibt der Seite kein Mikrofon.',
  'prompt.noMic.no_web_audio': 'Dieser Browser kann kein Web Audio.',
  'prompt.micRefused': 'Das Mikrofon ließ sich nicht öffnen: {why}',
  'prompt.tooShort': 'Zu kurz zum Erkennen: etwas länger sprechen, dann beenden.',
  'prompt.nothingHeard': 'In dieser Aufnahme wurde nichts erkannt.',
  'prompt.spoken': 'Aus Sprache erkannt, nicht getippt: vor dem Start lesen. Der Arm fährt nach dem, was gehört wurde.',
  'prompt.send': 'Senden',
  'prompt.sendTitle': 'Enter oder Senden liest den Satz. Gestartet wird erst mit Start auf der Karte.',
}

export default de
