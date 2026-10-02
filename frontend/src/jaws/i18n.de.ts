/**
 * The jaws dialog's words, in German (the console's default). This file defines the keys; `i18n.en.ts` is typed by
 * it, so an English text that is missing fails `tsc`.
 *
 * Wording rules: the question is asked about the output a person can find on the pendant (`Tool-DO0`), never about a
 * driver class; there is no default, and the dialog says so; "Jetzt öffnen" is one switch of the output that releases
 * what the jaws hold, and that is said before the button, calmly: no word in capitals.
 */

const de = {
  'jd.title': 'Backenfrage',
  'jd.at.connect': 'beim Verbinden',
  'jd.at.check': 'Prüfung vor dem Weiterfahren',
  'jd.ask.where': 'Wo stehen die Backen an {where}?',
  'jd.ask.open_now': 'Die Backen sind zu. Jetzt öffnen?',
  'jd.why.where':
    'Die Hand hat keinen Sensor: Jede Änderung von {where} bewegt die Backen, und nichts meldet zurück. Nur du siehst, ob sie offen oder zu sind. Es gibt keine Vorauswahl.',
  'jd.why.open_now':
    'Das Öffnen lässt los, was die Backen halten. Halte das Teil fest, bevor du „Jetzt öffnen“ wählst.',
  'jd.noAnswer': 'Ohne Antwort wird abgelehnt, nie „offen“.',
  'jd.choices': 'Antworten',
  'jd.label.open': 'Offen',
  'jd.label.closed': 'Geschlossen',
  'jd.label.open_now': 'Jetzt öffnen',
  'jd.label.abort': 'Abbrechen',
  'jd.choice.open': 'ab hier zählt Willy jede Änderung von {where}',
  'jd.choice.closed': 'danach fragt die Hand, ob sie jetzt öffnen soll',
  'jd.choice.open_now': 'ein Schaltvorgang an {where}: die Backen öffnen sich',
  'jd.choice.abort': 'nichts wird gesendet; die Backen gelten als nicht bestätigt',
  'jd.expires': 'läuft ab in {seconds} s',
  'jd.expired': 'Abgelaufen: ohne Antwort abgelehnt, nie „offen“.',
  'jd.attempt': 'Versuch {attempt} von {of}',
  'jd.sending': 'Antwort gesendet: die Hand führt sie aus …',
  'jd.busy':
    'Die Hand führt eine Antwort aus, oder eine Prüfung liest gerade Steuerung und Sperre. Bis dahin startet nichts, was sich bewegt.',
  'jd.tech.reason': 'Grund (Server): {text}',
  'jd.tech.again': 'Erneut gefragt: {text}',
  'jd.tech.id': 'Frage {id}',
}

export default de
