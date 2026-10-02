/**
 * The audience window's words, in German (the console's default). This file defines the keys; `i18n.en.ts` is typed
 * by it, so an English text that is missing fails `tsc`.
 *
 * Wording rules: big, short and factual, read from across a hall: what the robot does now, with what, and where it
 * goes. A caption's verb takes the operator's own words as they were said ("Sucht: den grünen Würfel"), so it is a
 * verb that takes an accusative; where the part goes is "Ziel: …", never a second "Ablage" before a label that
 * already says it. The one place with a joke is the STILL GRINDING card (the owner's brand, 2026-10-01); every stop
 * and every question says what happened plainly, and that a person at the console decides.
 */

const de = {
  'aud.title': 'Publikumsfenster',
  'aud.live.alt': 'Live-Bild der Zelle',
  'aud.live.camera': 'LIVE',
  'aud.live.synthetic': 'PROBE',
  'aud.live.none': 'KEIN BILD',
  'aud.live.measuring': 'misst …',
  'aud.live.stale': 'nicht live',
  'aud.noPicture': 'Kein Bild: die Zelle zeigt gerade keine Kamera.',
  'aud.notConnected': 'nicht verbunden',
  'aud.noServer': 'kein Server',

  'aud.cap.ready': 'Bereit',
  'aud.cap.ready.sub': 'Wartet auf den nächsten Auftrag.',
  'aud.cap.notConnected': 'Die Zelle ist nicht verbunden.',
  'aud.cap.handsOff': 'Hände weg:',
  'aud.cap.survey': 'Sucht das Ziel:',
  'aud.cap.start': 'Sucht:',
  'aud.cap.look': 'Sucht:',
  'aud.cap.detect': 'Erkennt:',
  'aud.cap.grasp': 'Greift:',
  'aud.cap.place': 'Legt ab:',
  'aud.cap.return': 'Fährt zurück:',
  'aud.cap.home': 'Fährt nach',
  'aud.cap.teach': 'Lernt eine Pose:',
  'aud.cap.teach.sub': 'Ein Mensch führt den Arm von Hand.',
  'aud.cap.planner': 'Der Planer startet',
  'aud.cap.planner.sub': 'Etwa eine Minute, bewegt wird nichts.',
  'aud.cap.halting': 'Hält an …',
  'aud.cap.done': 'Fertig:',
  'aud.cap.stopped': 'Gestoppt:',
  'aud.cap.ask': 'Rückfrage:',
  'aud.cap.problem': 'Problem:',
  'aud.cap.lost': 'Dieser Lauf ist nicht mehr bekannt.',
  'aud.cap.placed': '{n|# Teil|# Teile} abgelegt',
  'aud.cap.arrived': 'Angekommen:',

  'aud.sub.part': 'Teil {part}',
  'aud.sub.partOf': 'Teil {part} von {of}',
  'aud.sub.placed': '{n} abgelegt',
  'aud.sub.attempt': 'Versuch {n}/{total}',
  'aud.sub.pose': 'Ziel: {where}',
  'aud.sub.camera': 'Ziel: {where}',
  'aud.sub.once': 'einmal',
  'aud.sub.untilEmpty': 'bis leer',

  'aud.pin.wrist': 'wie entschieden {time}; der Arm hat sich seither bewegt',
  'aud.pin.fixed': 'wie entschieden {time}',
  'aud.pin.target': 'Ziel, wie gefunden {time}',

  'aud.card.where': 'Teil {part} · beim Schritt „{step}“',
  'aud.card.problem': 'Am Bedienplatz: die Zelle freigeben, dann Neustart oder Home. Bis dahin bewegt sich nichts.',
  'aud.card.cleared': 'Zelle freigegeben: wartet auf Neustart oder Home.',
  'aud.card.ask': 'Willy fragt am Bedienplatz, wie es weitergeht.',

  'aud.grind.title': 'STILL GRINDING',
  'aud.grind.parts': 'Teile heute',
  'aud.grind.time': 'Laufzeit',
  'aud.grind.coffee': 'Kaffeepausen',
  'aud.grind.raises': 'Gehaltserhöhungen verlangt',
  'aud.grind.hours': '{h}:{mm} h',
  'aud.grind.minutes': '{m} min',
  'aud.grind.seconds': '{s} s',
}

export default de
