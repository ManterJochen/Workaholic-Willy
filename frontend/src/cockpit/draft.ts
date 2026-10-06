/**
 * What the Understood card holds (build plan 4.2, OD 7, 8, 12, 22): the reader's answer made editable, and the one
 * request Start sends.
 *
 * Pure: the card's fields, the edits a person made (recorded by field name, for the run's record), what keeps Start
 * off, and the `TaskIn` a click on Start posts. Nothing here talks to the server; `useCommand` does, and only on a
 * person's Enter (a parse, which moves nothing) or a person's click on Start (the task, which moves).
 *
 * The place has three forms, as the card offers them: the cell's default place pose (`robot.default_place_pose`), a
 * taught pose by its name (shown by its label), or a target the camera finds ("blue bin"). Scope is once or until
 * empty; after the task the arm returns to Home or to a taught pose.
 */

import type { CellFactsOut, CommandOut, PosesOut, RoutePreviewOut, TaskIn, TaskPlanOut } from '../api/client'
import type { CommandNote } from '../api/codes'

/** The closing axes a task may keep its grasps to (`TaskOptionsIn.closing_axis`); `null` is any. */
export const CLOSING_AXES = ['x', '-x', 'y', '-y', 'radial', '-radial', 'tangential', '-tangential'] as const
export type ClosingAxis = (typeof CLOSING_AXES)[number]

export type PlaceChoice =
  | { readonly kind: 'default' }
  | { readonly kind: 'pose'; readonly pose: string }
  | { readonly kind: 'camera'; readonly phrase: string; readonly said: string | null }

/** The Advanced drawer (OD 12, plus the rim air of Q3). `null` numbers are the cell's own value. */
export interface DraftOptions {
  readonly multiView: boolean
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
  readonly recordViews: boolean
  readonly rimAirMm: number | null
  readonly overlay: boolean
}

export const DEFAULT_OPTIONS: DraftOptions = {
  multiView: true,
  bothFaces: false,
  closingAxis: null,
  pushMm: null,
  criticalParts: null,
  blockerIntoThePlace: null,
  rescan: null,
  push: null,
  clear: null,
  recordViews: false,
  rimAirMm: null,
  overlay: true,
}

/** The air over a camera-found bin's rim when the cell says nothing else (Q3): 20 mm, adjustable 10-50. */
export const RIM_AIR_MM = 20
export const RIM_AIR_RANGE = [10, 50] as const
/** The push distance shown when the cell does not say its own. */
export const PUSH_MM = 30
/** The shortest push the server takes (`PickIn.push_mm`: under 10 mm is refused, 422 `push_distance_refused`). */
export const PUSH_MIN_MM = 10

/** The fields a person can change, as the run's record names them (`CommandProvenanceIn.edited`). */
export type DraftField = 'object' | 'place' | 'scope' | 'return_to' | 'options' | 'pick_anything'

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
}

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
  readonly objectRoute: RoutePreviewOut | null
  readonly place: PlaceChoice
  readonly placeVerified: boolean
  readonly placeRoute: RoutePreviewOut | null
  readonly scope: 'once' | 'until_empty'
  /** `home` or a taught pose's name. */
  readonly returnTo: string
  /** The operator's tick: an empty phrase means anything the camera sees, bin walls included (Q7 A+). */
  readonly pickAnything: boolean
  readonly options: DraftOptions
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

/** The reader's answer as a card (`POST /v1/commands/parse`). `understood: false` opens it by hand, with the reason. */
export function draftFromReading(out: CommandOut, said: Said): Draft {
  const place: PlaceChoice = out.place_pose
    ? { kind: 'pose', pose: out.place_pose }
    : out.place?.phrase
      ? { kind: 'camera', phrase: out.place.phrase, said: out.place.said ?? null }
      : { kind: 'default' }
  return {
    id: nextId(),
    mode: out.understood ? 'vlm' : 'manual',
    manualCode: '',
    manualMessage: out.understood ? '' : out.reason,
    object: out.object?.phrase ?? '',
    objectSaid: out.object?.said ?? null,
    objectVerified: out.object?.verified ?? false,
    objectRoute: out.object?.route ?? null,
    place,
    placeVerified: out.place ? out.place.verified : true,
    placeRoute: out.place?.route ?? null,
    scope: out.scope ?? 'once',
    returnTo: out.return_to ?? 'home',
    pickAnything: false,
    options: DEFAULT_OPTIONS,
    notes: out.notes ?? [],
    said,
    edited: [],
    reading: {
      latencyMs: out.model?.latency_ms ?? null,
      attempts: out.model?.attempts ?? null,
      modelId: out.model?.model_id ?? null,
      raw: out.raw,
      reason: out.reason,
    },
  }
}

/** A card filled by hand: no reader answered (`vlm_unavailable`, `vlm_not_loaded`, `vlm_model_missing`). */
export function manualDraft(said: Said, code: string, message: string): Draft {
  return {
    id: nextId(),
    mode: 'manual',
    manualCode: code,
    manualMessage: message,
    object: '',
    objectSaid: null,
    objectVerified: true,
    objectRoute: null,
    place: { kind: 'default' },
    placeVerified: true,
    placeRoute: null,
    scope: 'once',
    returnTo: 'home',
    pickAnything: false,
    options: DEFAULT_OPTIONS,
    notes: [],
    said,
    edited: [],
    reading: null,
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
    place:
      plan.place.kind === 'camera'
        ? { kind: 'camera', phrase: plan.place.phrase ?? '', said: plan.place.said ?? null }
        : plan.place.pose
          ? { kind: 'pose', pose: plan.place.pose }
          : { kind: 'default' },
    placeVerified: true,
    placeRoute: null,
    scope: plan.scope,
    returnTo: plan.return_to || 'home',
    pickAnything: options?.pick_anything === true,
    options: {
      multiView: options?.multi_view ?? true,
      bothFaces: options?.both_faces ?? false,
      closingAxis: (options?.closing_axis as ClosingAxis | null | undefined) ?? null,
      pushMm: null,
      criticalParts: options?.critical_parts ?? null,
      blockerIntoThePlace: options?.blocker_into_the_place ?? null,
      rescan: options?.rescan ?? null,
      push: options?.push ?? null,
      clear: options?.clear ?? null,
      recordViews: options?.record_views ?? false,
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

/** The request a click on Start posts (`POST /v1/task`). */
export function taskOf(draft: Draft): TaskIn {
  const object = draft.object.trim()
  const camera = draft.place.kind === 'camera'
  return {
    object,
    object_said: draft.objectSaid,
    place:
      draft.place.kind === 'camera'
        ? { kind: 'camera', phrase: draft.place.phrase.trim(), said: draft.place.said }
        : { kind: 'pose', pose: draft.place.kind === 'pose' ? draft.place.pose : null },
    return_to: draft.returnTo || 'home',
    scope: draft.scope,
    options: {
      multi_view: draft.options.multiView,
      both_faces: draft.options.bothFaces,
      closing_axis: draft.options.closingAxis,
      push_mm: draft.options.pushMm,
      critical_parts: draft.options.criticalParts,
      blocker_into_the_place: draft.options.blockerIntoThePlace,
      rescan: draft.options.rescan,
      push: draft.options.push,
      clear: draft.options.clear,
      record_views: draft.options.recordViews,
      // A pose place has no rim: the library refuses air for one (`PlaceAt` in src/robot/execution/place_target.py).
      rim_air_mm: camera ? draft.options.rimAirMm : null,
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
 * place (`default`). Each is a Start of its own, on a person's click.
 */
export function taskOfPlan(plan: TaskPlanOut, place: 'same' | 'default' = 'same'): TaskIn {
  const options = plan.options
  const camera = place === 'same' && plan.place.kind === 'camera'
  const command = plan.command
  return {
    object: plan.object,
    object_said: plan.object_said ?? null,
    place: camera
      ? { kind: 'camera', phrase: plan.place.phrase ?? '', said: plan.place.said ?? null }
      : { kind: 'pose', pose: place === 'same' ? (plan.place.pose ?? null) : null },
    return_to: plan.return_to || 'home',
    scope: plan.scope,
    options: {
      multi_view: options?.multi_view ?? true,
      both_faces: options?.both_faces ?? false,
      closing_axis: options?.closing_axis ?? null,
      push_mm: options?.push_mm ?? null,
      record_views: options?.record_views ?? false,
      rim_air_mm: camera ? (options?.rim_air_mm ?? null) : null,
      pick_anything: options?.pick_anything ?? false,
      overlay: options?.overlay ?? true,
    },
    command: command
      ? { text: command.text, source: command.source, language: command.language ?? null, parsed: command.parsed, edited: [...(command.edited ?? []), 'place'] }
      : null,
  }
}

/** Why Start stays off for this card, beyond the cell's readiness. */
export type DraftProblem = 'needObject' | 'needPhrase' | 'noDefault' | 'unknownPose'

/**
 * What the card itself still lacks. A real cell refuses an empty phrase unless the operator ticked "anything" (Q7 A+;
 * the rehearsal cell picks what it shows). A taught pose that is gone, or a default place nobody declared, is said here
 * before the server would refuse it; where the poses could not be read, the server's refusal says it.
 */
export function draftProblems(draft: Draft, context: { rehearsal: boolean; poses: PosesOut | null }): DraftProblem[] {
  const problems: DraftProblem[] = []
  if (draft.object.trim() === '' && !draft.pickAnything && !context.rehearsal) problems.push('needObject')
  if (draft.place.kind === 'camera' && draft.place.phrase.trim() === '') problems.push('needPhrase')
  const poses = context.poses
  if (poses) {
    const taught = new Set((poses.poses ?? []).map((pose) => pose.name))
    if (draft.place.kind === 'default' && !poses.default_place) problems.push('noDefault')
    const missing =
      (draft.place.kind === 'pose' && !taught.has(draft.place.pose)) || (draft.returnTo !== 'home' && !taught.has(draft.returnTo))
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
