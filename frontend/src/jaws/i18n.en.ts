/**
 * The jaws dialog's words, in English: the switch, and what the dialog says under no provider. Typed by the German
 * file: a key missing here, or one German does not have, fails `tsc`.
 */

import type de from './i18n.de'

const en: Record<keyof typeof de, string> = {
  'jd.title': 'Jaws question',
  'jd.at.connect': 'while connecting',
  'jd.at.check': 'check before going on',
  'jd.ask.where': 'Where do the jaws on {where} stand?',
  'jd.ask.open_now': 'The jaws are closed. Open them now?',
  'jd.why.where':
    'The hand has no sensor: every change of {where} moves the jaws, and nothing reads them back. Only you can see whether they stand open or closed. Nothing is preselected.',
  'jd.why.open_now': 'Opening releases what the jaws hold. Hold the part before you choose “Open now”.',
  'jd.noAnswer': 'Unanswered, it is refused, never “open”.',
  'jd.choices': 'Answers',
  'jd.label.open': 'Open',
  'jd.label.closed': 'Closed',
  'jd.label.open_now': 'Open now',
  'jd.label.abort': 'Cancel',
  'jd.choice.open': 'from here on Willy counts every change of {where}',
  'jd.choice.closed': 'then the hand asks whether to open them now',
  'jd.choice.open_now': 'one switch of {where}: the jaws open',
  'jd.choice.abort': 'nothing is sent; the jaws count as not confirmed',
  'jd.expires': 'expires in {seconds} s',
  'jd.expired': 'Expired: refused unanswered, never “open”.',
  'jd.attempt': 'attempt {attempt} of {of}',
  'jd.sending': 'Answer sent: the hand acts on it…',
  'jd.busy':
    'The hand is acting on an answer, or a check is reading the controller and the latch. Until then nothing that moves starts.',
  'jd.tech.reason': 'Reason (server): {text}',
  'jd.tech.again': 'Asked again: {text}',
  'jd.tech.id': 'question {id}',
}

export default en
