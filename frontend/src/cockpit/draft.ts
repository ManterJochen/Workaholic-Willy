/**
 * What the Understood card holds (build plan 4.2, OD 7, 8, 12, 22): the reader's answer made editable, and the one
 * request Start sends.
 *
 * Pure: the card's fields, the edits a person made (recorded by field name, for the run's record), what keeps Start
 * off, why a reading needs a look before it starts (`doubts`), and the `TaskIn` that starts it. Nothing here talks to
 * the server; `useCommand` does, and only on a person's Enter (a parse, which moves nothing, and where the settings
 * say so the task of a clean reading) or a person's click on Start (the task, which moves).
 *
 * The place has three forms, as the card offers them: the cell's default place pose (`robot.default_place_pose`), a
 * taught pose by its name (shown by its label), or a target the camera finds ("blue bin"). Scope is once or until
 * empty; after the task the arm returns to Home or to a taught pose. What the sentence does not say comes from the
 * settings (the owner, 2026-10-08): the place where none is named, the tick for "anything" where no part is named,
 * and, where Enter starts, the mode.
 *
 * What to pick may be narrowed twice (2026-10-08): the one part the sentence singles out (`which`, "the gray cube on
 * top of the other one"), which the detector is asked alone, and where the parts lie (`source`, "on the black mat").
 * A reading that singles out one part starts once, whatever mode the settings set: it names one part.
 *
 * The console declutters (the owner, 2026-10-08 night): how each pick looks is one choice, the first look only, when
 * needed or every look, which the settings set and the card may change; every task the console starts keeps its picks'
 * looks on disk (`record_views`, no switch); "Beide Backenflächen" is a program's switch, which a card made from a run's
 * plan keeps as it ran; and "Direkt über die Kante tragen" overrides the cell's carry to a bin for one task.
 *
 * A sort (the owner, 2026-10-09: "Grüne Teile in die gelbe Kiste, rote in die blaue") is several rules, each a kind of
 * part and the place its parts go: the card's own fields are the first rule, `moreRules` the others, at most three
 * (`TaskIn.more_rules`). Two rules may share one place; two rules may not name one kind, and a sort names every kind it
 * takes, so Start stays off until each rule has its own. A card of one rule sends exactly what it always sent.
 */

import type {
  CellFactsOut,
  CommandOut,
  CommandPhraseOut,
  CommandRuleOut,
  PlanPlaceOut,
  PosesOut,
  RoutePreviewOut,
  TaskIn,
  TaskPlanOut,
  TaskRuleIn,
} from '../api/client'
import type { CommandNote } from '../api/codes'
import { listMsg, noteMsg } from '../i18n/codes'
import type { Msg, ParamValue } from '../i18n/types'
import type { LooksChoice, TaskPlace, TaskPrefs } from '../model/prefs'

/** The closing axes a task may keep its grasps to (`TaskOptionsIn.closing_axis`); `null` is any. */
export const CLOSING_AXES = ['x', '-x', 'y', '-y', 'radial', '-radial', 'tangential', '-tangential'] as const
export type ClosingAxis = (typeof CLOSING_AXES)[number]

export type PlaceChoice =
  | { readonly kind: 'default' }
  | { readonly kind: 'pose'; readonly pose: string }
  | { readonly kind: 'camera'; readonly phrase: string; readonly said: string | null }

/** The Advanced drawer (OD 12, plus the rim air of Q3). `null` numbers are the cell's own value. */
export interface DraftOptions {
  /** How each pick looks (the owner, 2026-10-08 night): the first look only (`multi_view` off), when needed (on, the
   *  early stop as always), or every look (`every_look`). */
  readonly looks: LooksChoice
  /** Both jaw contact faces seen before a grip: a program's switch the console no longer offers. A card made from a run's
   *  plan keeps it as the plan ran; every other card has it off. */
  readonly bothFaces: boolean
  readonly closingAxis: ClosingAxis | null
  readonly pushMm: number | null
  /** The owner's switch (2026-10-03): critical parts are never pushed, a blocker is cleared instead. `null` is the
   *  cell's own. */
  readonly criticalParts: boolean | null
  /** The owner's switch (2026-10-06): where the task takes every part, a blocker goes where the parts go ("direkt
   *  weggepackt"), else it is only set aside ("nur umgelegt"). `null` is the cell's own. */
  readonly blockerIntoThePlace: boolean | null
  /** The run's own word on recovery (null: the cell's). */
  readonly rescan: boolean | null
  readonly push: boolean | null
  readonly clear: boolean | null
  /** The owner's switch (2026-10-08 night, "Direkt über die Kante tragen"): a part goes to a bin a wrist camera found
   *  straight over its rim (`over_the_rim`), or via the look it was found from (`via_the_look`). `null` is the cell's
   *  own (`robot.place.carry`). */
  readonly carryOverTheRim: boolean | null
  readonly rimAirMm: number | null
  readonly overlay: boolean
}

export const DEFAULT_OPTIONS: DraftOptions = {
  looks: 'when_needed',
  bothFaces: false,
  closingAxis: null,
  pushMm: null,
  criticalParts: null,
  blockerIntoThePlace: null,
  rescan: null,
  push: null,
  clear: null,
  carryOverTheRim: null,
  rimAirMm: null,
  overlay: true,
}

/** The options a card starts from: the defaults, its picks looking as the settings say. */
function optionsOf(task?: Pick<TaskPrefs, 'looks'>): DraftOptions {
  return task ? { ...DEFAULT_OPTIONS, looks: task.looks } : DEFAULT_OPTIONS
}

/** How a run's plan looked: the first look only, every look, or when needed. A plan kept without the choice looks as
 *  its multi-view says. */
export function looksOfPlan(options: TaskPlanOut['options'] | null | undefined): LooksChoice {
  if (options?.multi_view === false) return 'first'
  return options?.every_look === true ? 'every' : 'when_needed'
}

/** The choice as the task's options carry it (`TaskOptionsIn.multi_view`, `every_look`). */
export function looksOptions(looks: LooksChoice): { multi_view: boolean; every_look: boolean } {
  return { multi_view: looks !== 'first', every_look: looks === 'every' }
}

/** The carry switch as the task's options carry it (`TaskOptionsIn.carry`): `null`, the cell's. */
function carryOf(overTheRim: boolean | null): 'over_the_rim' | 'via_the_look' | null {
  return overTheRim === null ? null : overTheRim ? 'over_the_rim' : 'via_the_look'
}

/** The air over a camera-found bin's rim when the cell says nothing else (Q3): 20 mm, adjustable 10-50. */
export const RIM_AIR_MM = 20
export const RIM_AIR_RANGE = [10, 50] as const
/** The push distance shown when the cell does not say its own. */
export const PUSH_MM = 30
/** The shortest push the server takes (`PickIn.push_mm`: under 10 mm is refused, 422 `push_distance_refused`). */
export const PUSH_MIN_MM = 10

/** The fields a person can change, as the run's record names them (`CommandProvenanceIn.edited`); `rules`, a sort's
 *  further rules, any of them changed, added or removed. */
export type DraftField = 'object' | 'which' | 'source' | 'place' | 'scope' | 'return_to' | 'options' | 'pick_anything' | 'rules'

/**
 * One further rule of a sort, as the card holds it: a kind of part and where each such part goes. The card's own
 * fields are the first rule. Its one part singled out and where its parts lie come from the reading and go out with
 * the task; the card's row edits the kind and the place.
 */
export interface DraftRule {
  /** The English phrase of the kind (`red part`). */
  readonly object: string
  readonly objectSaid: string | null
  /** The reader found the kind in the sentence; "bitte prüfen" otherwise. */
  readonly objectVerified: boolean
  readonly objectRoute: RoutePreviewOut | null
  readonly which: string
  readonly source: string
  readonly place: PlaceChoice
  readonly placeVerified: boolean
  readonly placeRoute: RoutePreviewOut | null
  /** What the reader wants a person to check in this rule. */
  readonly notes: readonly CommandNote[]
}

/** What the person said or typed, kept for the record only: nothing decides what moves from it. */
export interface Said {
  readonly text: string
  readonly source: 'typed' | 'spoken'
  readonly language: string
}

export interface Reading {
  readonly latencyMs: number | null
  readonly attempts: number | null
  readonly modelId: string | null
  readonly raw: string
  readonly reason: string
  /** The known sentences' table answered: no model was asked. */
  readonly known: boolean
  /** Every answer came from the reader's memory of the same question. */
  readonly remembered: boolean
}

/** The `model_id` of a reading the known sentences' table answered (`src/models/vlm/known.py`). */
export const KNOWN_SENTENCE_MODEL = 'known-sentence'

export interface Draft {
  /** A fresh card per reading: an answer that arrives late for an older card never edits a newer one. */
  readonly id: number
  /** `vlm`: the reader filled it. `manual`: no reader answered, a person fills it (OD 22). */
  readonly mode: 'vlm' | 'manual'
  /** Why the card is manual: the refusal's code (`vlm_unavailable`), `''` where the reader answered. */
  readonly manualCode: string
  /** The backend's own sentence for it, for the tech view. */
  readonly manualMessage: string
  /** The English phrase the detector is given. `''` is anything (only with `pickAnything` on a real cell, Q7). */
  readonly object: string
  /** The operator's own words for it, shown under the phrase. */
  readonly objectSaid: string | null
  /** The reader checked the phrase against the sentence; "bitte prüfen" otherwise. */
  readonly objectVerified: boolean
  /** How the detector would route what the picks ground (`groundedOf`). */
  readonly objectRoute: RoutePreviewOut | null
  /** The one part singled out, in English ("the gray cube on top of the other one"): the detector is asked this alone.
   *  `''` is every part of the object's kind. */
  readonly which: string
  /** Where the parts lie, in English with its preposition ("on the black mat"). `''` is wherever the camera sees one. */
  readonly source: string
  readonly place: PlaceChoice
  readonly placeVerified: boolean
  readonly placeRoute: RoutePreviewOut | null
  /** A sort's further rules, the card's own fields being the first (the owner, 2026-10-09); none for one kind. */
  readonly moreRules: readonly DraftRule[]
  readonly scope: 'once' | 'until_empty'
  /** `home` or a taught pose's name. */
  readonly returnTo: string
  /** The operator's tick: an empty phrase means anything the camera sees, bin walls included (Q7 A+). */
  readonly pickAnything: boolean
  readonly options: DraftOptions
  /** What the reader wants a person to check: the reading's own notes and its first rule's; a further rule keeps its
   *  own (`DraftRule.notes`). */
  readonly notes: readonly CommandNote[]
  readonly said: Said
  readonly edited: readonly DraftField[]
  /** How the reader answered, for the tech view; `null` for a card filled by hand. */
  readonly reading: Reading | null
}

let ids = 0

function nextId(): number {
  ids += 1
  return ids
}

/** A place the settings name, as the card offers it. */
export function placeChoiceOf(place: TaskPlace): PlaceChoice {
  if (place.kind === 'pose') return { kind: 'pose', pose: place.pose }
  if (place.kind === 'camera') return { kind: 'camera', phrase: place.phrase, said: null }
  return { kind: 'default' }
}

/** The place a reading names (a taught pose, then a target the camera finds), else the settings', else the default. */
function placeOfReading(pose: string | null | undefined, place: CommandPhraseOut | null | undefined, task?: TaskPrefs): PlaceChoice {
  if (pose) return { kind: 'pose', pose }
  if (place?.phrase) return { kind: 'camera', phrase: place.phrase, said: place.said ?? null }
  return task ? placeChoiceOf(task.place) : { kind: 'default' }
}

/** A sort's further rule as the reader understood it (`CommandOut.rules[1:]`). */
function ruleOfReading(rule: CommandRuleOut, task?: TaskPrefs): DraftRule {
  return {
    object: rule.object?.phrase ?? '',
    objectSaid: rule.object?.said ?? null,
    objectVerified: rule.object?.verified ?? false,
    objectRoute: rule.object?.route ?? null,
    which: rule.which ?? '',
    source: rule.source ?? '',
    place: placeOfReading(rule.place_pose, rule.place, task),
    placeVerified: rule.place ? rule.place.verified : true,
    placeRoute: rule.place?.route ?? null,
    notes: rule.notes ?? [],
  }
}

/**
 * The reading's notes the card says as its own: all of them for a reading of one rule; in a sort, those of the reading
 * and of its first rule, a note only a further rule carries being said with that rule ("Regel 2: ...").
 */
function ownNotes(out: CommandOut): CommandNote[] {
  const notes = out.notes ?? []
  const rules = out.rules ?? []
  if (rules.length < 2) return [...notes]
  const first = rules[0].notes ?? []
  const own = notes.filter((note) => first.includes(note) || !rules.slice(1).some((rule) => (rule.notes ?? []).includes(note)))
  return [...new Set(own)]
}

/**
 * The reader's answer as a card (`POST /v1/commands/parse`). `understood: false` opens it by hand, with the reason.
 *
 * `task` is what the person set beforehand. The sentence's own place wins (a taught pose, then a target the camera
 * finds), and the settings' place stands where it names none; a sentence that names no part comes ticked for
 * "anything" where the settings say so. Where Enter starts (`task.start === 'enter'`) the mode is the settings', not
 * the sentence's (the owner, 2026-10-08); the card of `card` keeps the reader's, as before. A sentence that singles out
 * one part (`which`) is once either way: "until empty" would go on to parts it did not name. How its picks look comes
 * from the settings, either way. A sort's further rules come from the reading's own (`rules[1:]`).
 */
export function draftFromReading(out: CommandOut, said: Said, task?: TaskPrefs): Draft {
  const object = out.object?.phrase ?? ''
  return {
    id: nextId(),
    mode: out.understood ? 'vlm' : 'manual',
    manualCode: '',
    manualMessage: out.understood ? '' : out.reason,
    object,
    objectSaid: out.object?.said ?? null,
    objectVerified: out.object?.verified ?? false,
    objectRoute: out.object?.route ?? null,
    which: out.which ?? '',
    source: out.source ?? '',
    place: placeOfReading(out.place_pose, out.place, task),
    placeVerified: out.place ? out.place.verified : true,
    placeRoute: out.place?.route ?? null,
    moreRules: (out.rules ?? []).slice(1).map((rule) => ruleOfReading(rule, task)),
    scope: out.which ? 'once' : task?.start === 'enter' ? task.scope : (out.scope ?? 'once'),
    returnTo: out.return_to ?? 'home',
    pickAnything: object === '' && task?.anything === true,
    options: optionsOf(task),
    notes: ownNotes(out),
    said,
    edited: [],
    reading: {
      latencyMs: out.model?.latency_ms ?? null,
      attempts: out.model?.attempts ?? null,
      modelId: out.model?.model_id ?? null,
      raw: out.raw,
      reason: out.reason,
      known: out.model?.model_id === KNOWN_SENTENCE_MODEL,
      remembered: out.model?.remembered === true,
    },
  }
}

/** Why one rule needs a look: its notes, a part named by no word or not found in the sentence, a place not found. */
function ruleDoubts(notes: readonly CommandNote[], object: CommandPhraseOut | null | undefined, place: CommandPhraseOut | null | undefined): Msg<string>[] {
  const why: Msg<string>[] = notes.map((note) => noteMsg(note))
  if (!object?.phrase) why.push({ key: 'ck.doubt.noPart' })
  else if (!object.verified) why.push({ key: 'ck.doubt.object' })
  if (place && !place.verified) why.push({ key: 'ck.doubt.place' })
  return why
}

/** A doubt of a sort's further rule, said with the rule's number ("Regel 2: das Teil steht so nicht im Satz"). */
export function ruleDoubt(n: number, why: Msg<string>): Msg<string> {
  return { key: 'ck.doubt.rule', params: { n, why } }
}

/**
 * Why a reading needs a person's look before it starts, in the order the card would say it: its notes, a part named
 * by no word or not found in the sentence, a place not found there; in a sort, each further rule's the same, said
 * with its number. Empty only for a reading the reader itself calls `startable` (R3: an understood task, no note, its
 * part and its place found in the sentence; in a sort, every rule clean); Enter starts only such a reading, and opens
 * the card for every other: a note on any rule opens it, whatever `startable` says.
 */
export function doubts(out: CommandOut): Msg<string>[] {
  const why = ruleDoubts(ownNotes(out), out.object, out.place)
  ;(out.rules ?? []).slice(1).forEach((rule, index) => {
    for (const doubt of ruleDoubts(rule.notes ?? [], rule.object, rule.place)) why.push(ruleDoubt(index + 2, doubt))
  })
  if (why.length === 0 && out.startable !== true) why.push({ key: 'ck.doubt.reading' })
  return why
}

/** A card filled by hand: no reader answered (`vlm_unavailable`, `vlm_not_loaded`, `vlm_model_missing`). Its picks look
 *  as the settings say (`task`), where they are handed. */
export function manualDraft(said: Said, code: string, message: string, task?: Pick<TaskPrefs, 'looks'>): Draft {
  return {
    id: nextId(),
    mode: 'manual',
    manualCode: code,
    manualMessage: message,
    object: '',
    objectSaid: null,
    objectVerified: true,
    objectRoute: null,
    which: '',
    source: '',
    place: { kind: 'default' },
    placeVerified: true,
    placeRoute: null,
    moreRules: [],
    scope: 'once',
    returnTo: 'home',
    pickAnything: false,
    options: optionsOf(task),
    notes: [],
    said,
    edited: [],
    reading: null,
  }
}

/** A plan's resolved place as the card offers it; a rule whose payload carried none, the default place. */
function placeOfPlan(place: PlanPlaceOut | null | undefined): PlaceChoice {
  if (place?.kind === 'camera') return { kind: 'camera', phrase: place.phrase ?? '', said: place.said ?? null }
  return place?.pose ? { kind: 'pose', pose: place.pose } : { kind: 'default' }
}

/** A further rule a person adds on the card ("Regel hinzufügen"): no kind yet, the default place, nothing to check. */
export function newRule(): DraftRule {
  return {
    object: '',
    objectSaid: null,
    objectVerified: true,
    objectRoute: null,
    which: '',
    source: '',
    place: { kind: 'default' },
    placeVerified: true,
    placeRoute: null,
    notes: [],
  }
}

/** A stopped or asking run's plan as a card to edit ("Anderes Ziel"): every field as it ran, the person changes one. */
export function draftFromPlan(plan: TaskPlanOut): Draft {
  const options = plan.options
  const command = plan.command
  return {
    id: nextId(),
    mode: 'manual',
    manualCode: '',
    manualMessage: '',
    object: plan.object,
    objectSaid: plan.object_said ?? null,
    objectVerified: true,
    objectRoute: null,
    which: plan.which ?? '',
    source: plan.source ?? '',
    place: placeOfPlan(plan.place),
    placeVerified: true,
    placeRoute: null,
    moreRules: (plan.more_rules ?? []).map((rule) => ({
      ...newRule(),
      object: rule.object,
      objectSaid: rule.object_said ?? null,
      which: rule.which ?? '',
      source: rule.source ?? '',
      place: placeOfPlan(rule.place),
    })),
    scope: plan.scope,
    returnTo: plan.return_to || 'home',
    pickAnything: options?.pick_anything === true,
    options: {
      looks: looksOfPlan(options),
      bothFaces: options?.both_faces ?? false,
      closingAxis: (options?.closing_axis as ClosingAxis | null | undefined) ?? null,
      pushMm: null,
      criticalParts: options?.critical_parts ?? null,
      blockerIntoThePlace: options?.blocker_into_the_place ?? null,
      rescan: options?.rescan ?? null,
      push: options?.push ?? null,
      clear: options?.clear ?? null,
      carryOverTheRim: options?.carry ? options.carry === 'over_the_rim' : null,
      rimAirMm: null,
      overlay: options?.overlay ?? true,
    },
    notes: [],
    said: { text: command?.text ?? '', source: command?.source ?? 'typed', language: command?.language ?? '' },
    edited: [],
    reading: null,
  }
}

/** One edit of the card, recorded by its field. */
export function edit(draft: Draft, field: DraftField, change: Partial<Draft>): Draft {
  const edited = draft.edited.includes(field) ? draft.edited : [...draft.edited, field]
  return { ...draft, ...change, edited }
}

/**
 * What the picks of the card would ground, as the server's route guard judges it (`api/routers/task.py` `_grounded`,
 * the library's `pick_phrase` without its "each separate"): the one part singled out alone, else the object where the
 * parts lie. The pick's route badge previews this.
 */
export function groundedOf(draft: Pick<Draft, 'object' | 'which' | 'source'>): string {
  return draft.which.trim() || [draft.object.trim(), draft.source.trim()].filter((words) => words !== '').join(' ')
}

/** A place as the task carries it: a target the camera finds, or a pose (`null`: the cell's default place). */
function placeIn(place: PlaceChoice): TaskIn['place'] {
  return place.kind === 'camera'
    ? { kind: 'camera', phrase: place.phrase.trim(), said: place.said }
    : { kind: 'pose', pose: place.kind === 'pose' ? place.pose : null }
}

/** Every place of the card, its own first, then each further rule's. */
export function placesOf(draft: Pick<Draft, 'place' | 'moreRules'>): PlaceChoice[] {
  return [draft.place, ...draft.moreRules.map((rule) => rule.place)]
}

/** Some rule of the card puts its parts where the camera finds them: the rim's air and the carry are said and sent. */
export function cameraPlaced(draft: Pick<Draft, 'place' | 'moreRules'>): boolean {
  return placesOf(draft).some((place) => place.kind === 'camera')
}

/** A further rule as the task carries it (`TaskRuleIn`), without the blanks around its phrases. */
function ruleIn(rule: DraftRule): TaskRuleIn {
  return {
    object: rule.object.trim(),
    object_said: rule.objectSaid,
    which: rule.which.trim(),
    source: rule.source.trim(),
    place: placeIn(rule.place),
  }
}

/** The request a click on Start posts (`POST /v1/task`). Every task the console starts keeps its picks' looks on disk
 *  (the owner, 2026-10-08 night: records and views always on). A sort sends its further rules (`more_rules`); a card of
 *  one rule sends none, so its body is the one it always was. */
export function taskOf(draft: Draft): TaskIn {
  const object = draft.object.trim()
  const camera = cameraPlaced(draft)
  const more = draft.moreRules.map(ruleIn)
  return {
    object,
    object_said: draft.objectSaid,
    which: draft.which.trim(),
    source: draft.source.trim(),
    place: placeIn(draft.place),
    ...(more.length > 0 ? { more_rules: more } : {}),
    return_to: draft.returnTo || 'home',
    scope: draft.scope,
    options: {
      ...looksOptions(draft.options.looks),
      both_faces: draft.options.bothFaces,
      closing_axis: draft.options.closingAxis,
      push_mm: draft.options.pushMm,
      critical_parts: draft.options.criticalParts,
      blocker_into_the_place: draft.options.blockerIntoThePlace,
      rescan: draft.options.rescan,
      push: draft.options.push,
      clear: draft.options.clear,
      record_views: true,
      // A pose place has no rim: the library refuses air for one (`PlaceAt` in src/robot/execution/place_target.py).
      rim_air_mm: camera ? draft.options.rimAirMm : null,
      // The carry is to a bin the camera finds; a pose place takes none.
      carry: camera ? carryOf(draft.options.carryOverTheRim) : null,
      pick_anything: object === '' && draft.pickAnything,
      overlay: draft.options.overlay,
    },
    command: {
      text: draft.said.text,
      source: draft.said.source,
      language: draft.said.language || null,
      parsed: draft.mode === 'vlm',
      edited: [...draft.edited],
    },
  }
}

/**
 * A run's plan as a request again, for an ask card's option: the same task (`same`), or the same task into the default
 * place (`default`). Each is a Start of its own, on a person's click, its picks looking and the part carried as the plan
 * ran, its looks kept on disk as every task the console starts. A sort keeps every further rule as it ran: "Nochmal
 * suchen" sorts again, never its first rule alone (`default` moves the first rule's place only, and the ask card does
 * not offer it for a sort).
 */
export function taskOfPlan(plan: TaskPlanOut, place: 'same' | 'default' = 'same'): TaskIn {
  const options = plan.options
  const own = place === 'same' && plan.place.kind === 'camera'
  const more: TaskRuleIn[] = (plan.more_rules ?? []).map((rule) => ({
    object: rule.object,
    object_said: rule.object_said ?? null,
    which: rule.which ?? '',
    source: rule.source ?? '',
    place: placeIn(placeOfPlan(rule.place)),
  }))
  const camera = own || more.some((rule) => rule.place.kind === 'camera')
  const command = plan.command
  return {
    object: plan.object,
    object_said: plan.object_said ?? null,
    which: plan.which ?? '',
    source: plan.source ?? '',
    place: own
      ? { kind: 'camera', phrase: plan.place.phrase ?? '', said: plan.place.said ?? null }
      : { kind: 'pose', pose: place === 'same' ? (plan.place.pose ?? null) : null },
    ...(more.length > 0 ? { more_rules: more } : {}),
    return_to: plan.return_to || 'home',
    scope: plan.scope,
    options: {
      ...looksOptions(looksOfPlan(options)),
      both_faces: options?.both_faces ?? false,
      closing_axis: options?.closing_axis ?? null,
      push_mm: options?.push_mm ?? null,
      record_views: true,
      rim_air_mm: camera ? (options?.rim_air_mm ?? null) : null,
      carry: camera ? (options?.carry ?? null) : null,
      pick_anything: options?.pick_anything ?? false,
      overlay: options?.overlay ?? true,
    },
    command: command
      ? { text: command.text, source: command.source, language: command.language ?? null, parsed: command.parsed, edited: [...(command.edited ?? []), 'place'] }
      : null,
  }
}

/** Why Start stays off for this card, beyond the cell's readiness. */
export type DraftProblem = 'needObject' | 'ruleObject' | 'sameKind' | 'needPhrase' | 'noDefault' | 'unknownPose'

/** Each problem's words, a key of the cockpit's catalog: on the card under Start, and in the chat where Enter waits. */
export const PROBLEM_KEY = {
  needObject: 'ck.start.needObject',
  ruleObject: 'ck.start.ruleObject',
  sameKind: 'ck.start.sameKind',
  needPhrase: 'ck.start.needPhrase',
  noDefault: 'ck.place.noDefault',
  unknownPose: 'ck.start.unknownPose',
} as const satisfies Record<DraftProblem, string>

/** A kind as two rules are compared by it: case aside, its blanks collapsed ("Red  Part" is "red part"). */
export function kindOf(object: string): string {
  return object.trim().replace(/\s+/g, ' ').toLowerCase()
}

/**
 * What the card itself still lacks. A real cell refuses an empty phrase unless the operator ticked "anything" (Q7 A+;
 * the rehearsal cell picks what it shows). A taught pose that is gone, or a default place nobody declared, is said here
 * before the server would refuse it; where the poses could not be read, the server's refusal says it. A sort names the
 * kind of every rule ("anything" sorts nothing), and no kind twice: a part of it would have two places.
 */
export function draftProblems(draft: Draft, context: { rehearsal: boolean; poses: PosesOut | null }): DraftProblem[] {
  const problems: DraftProblem[] = []
  const sort = draft.moreRules.length > 0
  const kinds = [draft.object, ...draft.moreRules.map((rule) => rule.object)].map(kindOf)
  if (!sort && kinds[0] === '' && !draft.pickAnything && !context.rehearsal) problems.push('needObject')
  if (sort && kinds.includes('')) problems.push('ruleObject')
  const named = kinds.filter((kind) => kind !== '')
  if (new Set(named).size < named.length) problems.push('sameKind')
  const places = placesOf(draft)
  if (places.some((place) => place.kind === 'camera' && place.phrase.trim() === '')) problems.push('needPhrase')
  const poses = context.poses
  if (poses) {
    const taught = new Set((poses.poses ?? []).map((pose) => pose.name))
    if (places.some((place) => place.kind === 'default') && !poses.default_place) problems.push('noDefault')
    const missing =
      places.some((place) => place.kind === 'pose' && !taught.has(place.pose)) || (draft.returnTo !== 'home' && !taught.has(draft.returnTo))
    if (missing) problems.push('unknownPose')
  }
  return problems
}

/** The first motion of a new task, as the library will make it. */
export type FirstMotion = 'look' | 'homeLook' | 'grasp' | 'unknown'

/**
 * The first motion Start names (item 12), the library's own order (`src/robot/execution/task.py` `_looks`): the
 * configured looks come first wherever they are configured (`look`, "zu Blick 1"); a wrist camera with no looks
 * configured looks from Home, so the task begins with the planned move there (`homeLook`); a fixed camera sees the
 * table from where the arm stands, so the task begins with the approach to its first grasp (`grasp`). While the
 * cell's facts are not read, Start only says that the robot moves (`unknown`): a first motion it cannot know is
 * never guessed.
 */
export function firstMotion(facts: Pick<CellFactsOut, 'wrist_camera' | 'looks'> | null): FirstMotion {
  if (!facts) return 'unknown'
  if ((facts.looks ?? []).length > 0) return 'look'
  return facts.wrist_camera ? 'homeLook' : 'grasp'
}

/** The words for a first motion ("der Roboter fährt zu Blick 1"), a key of the cockpit's catalog. */
export const MOTION_KEY = {
  look: 'ck.motion.look',
  homeLook: 'ck.motion.homeLook',
  grasp: 'ck.motion.grasp',
  unknown: 'ck.motion.unknown',
} as const satisfies Record<FirstMotion, string>

/** A pose's label where it has one, else its name; Home by its own word. */
export function poseLabel(name: string, poses: PosesOut | null): string | null {
  if (name === 'home') return null
  const pose = (poses?.poses ?? []).find((p) => p.name === name)
  return pose?.label || name
}

/** Where a place choice puts the part, in the words a person reads: a pose's label, the target as said, the default
 *  place by its label where it is declared. */
export function placeChoiceWords(place: PlaceChoice, poses: PosesOut | null): ParamValue {
  if (place.kind === 'pose') return poseLabel(place.pose, poses) ?? place.pose
  if (place.kind === 'camera') return place.said || place.phrase.trim() || '—'
  const fallback = poses?.default_place ? poseLabel(poses.default_place, poses) : null
  return fallback ?? { key: 'common.defaultPlace' }
}

/**
 * The card's rules in a row, in the operator's words where they were said: "grüne Teile → in die gelbe Kiste · rote →
 * in die blaue". What the chat says it understood of a sort.
 */
export function rulesMsg(draft: Draft, poses: PosesOut | null): ParamValue {
  const rules = [draft, ...draft.moreRules].map((rule) => ({
    key: 'list.rule',
    params: { what: rule.objectSaid || rule.object.trim() || { key: 'common.anything' }, where: placeChoiceWords(rule.place, poses) },
  }))
  return listMsg(rules, 'list.rules')
}
