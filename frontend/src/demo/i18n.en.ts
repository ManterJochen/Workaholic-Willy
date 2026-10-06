/**
 * The audience window's words, in English. Typed by the German file: a key missing here, or one German does not have,
 * fails `tsc`.
 */

import type de from './i18n.de'

const en: Record<keyof typeof de, string> = {
  'aud.title': 'Audience window',
  'aud.live.alt': 'The cell’s live image',
  'aud.live.camera': 'LIVE',
  'aud.live.synthetic': 'REHEARSAL',
  'aud.live.none': 'NO PICTURE',
  'aud.live.measuring': 'measuring…',
  'aud.live.stale': 'not live',
  'aud.noPicture': 'No picture: the cell shows no camera right now.',
  'aud.notConnected': 'not connected',
  'aud.noServer': 'no server',

  'aud.cap.ready': 'Ready',
  'aud.cap.ready.sub': 'Waiting for the next task.',
  'aud.cap.notConnected': 'The cell is not connected.',
  'aud.cap.handsOff': 'Hands off:',
  'aud.cap.survey': 'Finding the target:',
  'aud.cap.start': 'Looking for:',
  'aud.cap.look': 'Looking for:',
  'aud.cap.detect': 'Detecting:',
  'aud.cap.grasp': 'Grasping:',
  'aud.cap.place': 'Placing:',
  'aud.cap.return': 'Returning:',
  'aud.cap.home': 'Moving to',
  'aud.cap.teach': 'Teaching a pose:',
  'aud.cap.teach.sub': 'A person guides the arm by hand.',
  'aud.cap.planner': 'The planner starts',
  'aud.cap.planner.sub': 'About a minute; nothing moves.',
  'aud.cap.wave': 'Willy waves',
  'aud.cap.halting': 'Halting…',
  'aud.cap.done': 'Done:',
  'aud.cap.stopped': 'Stopped:',
  'aud.cap.ask': 'A question:',
  'aud.cap.problem': 'Problem:',
  'aud.cap.lost': 'This run is no longer known.',
  'aud.cap.placed': '{n|# part|# parts} placed',
  'aud.cap.arrived': 'Arrived:',

  'aud.sub.part': 'part {part}',
  'aud.sub.partOf': 'part {part} of {of}',
  'aud.sub.placed': '{n} placed',
  'aud.sub.attempt': 'attempt {n}/{total}',
  'aud.sub.pose': 'target: {where}',
  'aud.sub.camera': 'target: {where}',
  'aud.sub.once': 'once',
  'aud.sub.untilEmpty': 'until empty',

  'aud.pin.wrist': 'as decided {time}; the arm has moved since',
  'aud.pin.fixed': 'as decided {time}',
  'aud.pin.target': 'the target, as found {time}',

  'aud.card.where': 'part {part} · at the step “{step}”',
  'aud.card.problem': 'At the console: confirm the cell is clear, then Restart or Home. Until then nothing moves.',
  'aud.card.cleared': 'Cell confirmed clear: waiting for Restart or Home.',
  'aud.card.ask': 'Willy asks at the console how to go on.',

  'aud.grind.title': 'STILL GRINDING',
  'aud.grind.parts': 'Parts today',
  'aud.grind.time': 'Working time',
  'aud.grind.coffee': 'Coffee breaks',
  'aud.grind.raises': 'Raises asked',
  'aud.grind.hours': '{h}:{mm} h',
  'aud.grind.minutes': '{m} min',
  'aud.grind.seconds': '{s} s',
}

export default en
