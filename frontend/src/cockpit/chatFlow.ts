/**
 * The chat window's flow (owner decisions, fifth round): one list, oldest first, the newest at the bottom.
 *
 * Three sources meet in it, ordered by when each thing was said (unix seconds; the server is on the same PC):
 *
 * * the conversation (`model/chat`): the operator's commands, Willy's replies, and one summary line per earlier run.
 *   The summary of the run on screen is left out: its live lines replace it;
 * * the run on screen (`RunView.chat`): its own lines (start, stops, end) one by one, and every part's lines gathered
 *   into ONE card per part, placed where the part began. The card of the part in hand shows its step lines live;
 * * the cell's own lines (the jaws question, "Zelle ist frei", a recovery ended), only those of this session.
 *
 * The demo view reads the `demo` lines; the tech view reads every line. Pure, so the order is tested without a DOM.
 */

import type { ConversationEntry } from '../model/chat'
import { hasEnded, type ChatLine, type PartView, type RunView } from '../model/runModel'

export type FlowItem =
  | { readonly kind: 'entry'; readonly key: string; readonly at: number; readonly entry: ConversationEntry }
  | { readonly kind: 'line'; readonly key: string; readonly at: number; readonly line: ChatLine }
  | {
      readonly kind: 'part'
      readonly key: string
      readonly at: number
      readonly part: PartView
      /** The part's lines, every level: the card shows what the view allows. */
      readonly lines: readonly ChatLine[]
      /** The part in hand: its card shows its steps as they happen. */
      readonly current: boolean
    }

export interface FlowInput {
  readonly conversation: readonly ConversationEntry[]
  readonly view: RunView
  readonly cellLines: readonly ChatLine[]
  /** Cell lines older than this belong to an earlier session of the console, not to this conversation. */
  readonly since: number
  readonly tech: boolean
}

function shown(line: ChatLine, tech: boolean): boolean {
  return tech || line.level === 'demo'
}

/**
 * The chat's items, oldest first. Equal times keep their source order: the conversation, the run's own lines, the
 * cell's lines, then the run's part cards. A part card stands where its part BEGAN, so whatever was said in that same
 * instant came before it: on a Windows clock the Restart's "the arm is back at its return pose" (the cell's line that
 * ends the stop record) and its part 1 are stamped alike, and the return happened first.
 */
export function chatFlow({ conversation, view, cellLines, since, tech }: FlowInput): FlowItem[] {
  const items: FlowItem[] = []
  for (const entry of conversation) {
    if (entry.kind === 'summary' && entry.runId !== undefined && entry.runId === view.runId) continue
    items.push({ kind: 'entry', key: `c:${entry.id}`, at: entry.at, entry })
  }

  const parts = new Map(view.parts.map((part) => [part.part, part]))
  const byPart = new Map<number, ChatLine[]>()
  for (const line of view.chat) {
    if (line.part !== null && parts.has(line.part)) {
      const list = byPart.get(line.part) ?? []
      list.push(line)
      byPart.set(line.part, list)
    } else if (shown(line, tech)) {
      items.push({ kind: 'line', key: `r:${line.id}`, at: line.at, line })
    }
  }

  for (const line of cellLines) {
    if (line.at < since || !shown(line, tech)) continue
    items.push({ kind: 'line', key: `cell:${line.id}`, at: line.at, line })
  }

  const running = view.runId !== null && !hasEnded(view.phase)
  for (const part of view.parts) {
    items.push({
      kind: 'part',
      key: `p:${view.runId}:${part.part}`,
      at: part.startedAt,
      part,
      lines: byPart.get(part.part) ?? [],
      current: running && part.placed === null,
    })
  }

  // A stable sort: the order the sources were added breaks every tie.
  return items.map((item, index) => ({ item, index })).sort((a, b) => a.item.at - b.item.at || a.index - b.index).map(({ item }) => item)
}
