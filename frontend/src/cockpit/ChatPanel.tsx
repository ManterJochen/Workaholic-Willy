/**
 * The chat window, filling the right column under the statistics (owner decisions, fifth round; OD 2, 7, 12).
 *
 * The operator's lines sit on the right as bubbles (with "gesprochen" where they were spoken), Willy's on the left: his
 * replies, one card per part, the step lines of the part in hand, the run's own lines (the start, a stop, the end with
 * "Was soll ich als Nächstes tun?"), and the Understood, ask and stop cards inside the same flow, the newest at the
 * bottom. The input with the microphone sits at the window's foot. Earlier tasks read as one line each.
 *
 * **Scrolling is the reader's.** The flow follows one of three things: the bottom (new lines scroll into view), the
 * card that opened last (it stays in view while lines arrive above it, as a reloaded page replays the stopped run), or
 * nothing, once the reader scrolled away to read: then it stays where the reader put it, however often the cell is
 * polled or a card redrawn, until the reader scrolls back down to the end (near it is the end: a hand on a wheel stops
 * a little short, and a running task keeps adding lines under it) or takes the button "Zu den neuesten Zeilen", which
 * is there whenever the reader is away from the end. A card that opens is shown from the start of its turn: the
 * Understood card with the operator's words above it, a stop or ask card with Willy's line above it, as far as the
 * window allows; when it all fits, that is the bottom. A line cut by the window's top edge fades out instead of showing
 * half a glyph.
 *
 * It only draws: what it shows comes from the conversation, the run on screen and the cell (`chatFlow`), and the cards
 * and the input are handed in by the cockpit.
 */

import { useLayoutEffect, useRef, useState, type ReactNode } from 'react'

import { useT } from '../i18n'
import { Icon, Logo } from '../icons'
import type { ConversationEntry } from '../model/chat'
import type { ChatLine, StepId } from '../model/runModel'
import ChatLineView from './ChatLine'
import type { FlowItem } from './chatFlow'
import { COCKPIT } from './i18n'
import PartCard from './PartCard'

/** Within this many pixels of the bottom the reader is at the bottom, and the flow follows new lines. */
const AT_BOTTOM_PX = 8

/**
 * A reader scrolling DOWN who stops this close to the end is back at the end: a wheel or a finger stops a little
 * short, and a running task adds lines under it while the flow scrolls. The larger of a few lines and a sixth of the
 * window.
 */
function nearTheEnd(box: HTMLElement): number {
  return Math.max(64, box.clientHeight / 6)
}

/** A card in the flow: its key says which card, for which run or reading (`ask:<run>`, `stop:<run>`, `card:<id>`). */
export interface ChatCard {
  readonly key: string
  readonly node: ReactNode
}

export interface ChatPanelProps {
  readonly items: readonly FlowItem[]
  /** The cards after the flow, oldest first: the ask card, the stop card, the Understood card. */
  readonly cards: readonly ChatCard[]
  /** The input at the foot: text, microphone, send. */
  readonly input: ReactNode
  /** What the chat is doing, top right ("Auftrag läuft", "wartet auf dich"). */
  readonly state: string
  /** The step of the part in hand, for its live line. */
  readonly step: StepId | null
  readonly tech: boolean
}

/** What the flow keeps in view while it grows: the bottom, the card opened last, or the reader's own place. */
type Follow = { readonly mode: 'bottom' } | { readonly mode: 'card'; readonly key: string } | { readonly mode: 'reader' }

/** How much the flow holds: a new item, or a new line in a part's card, changes it; a redraw does not. */
function revisionOf(items: readonly FlowItem[]): string {
  let lines = 0
  for (const item of items) lines += item.kind === 'part' ? 1 + item.lines.length : 1
  return `${lines}:${items.length > 0 ? items[items.length - 1].key : ''}`
}

/**
 * Where the turn a card answers begins, as the flow's scroll position: the Understood card from the operator's words
 * just above it, a stop or ask card from Willy's line just above it, a card under another card from its own top.
 */
function turnTop(box: HTMLElement, key: string): number | null {
  const slot = Array.from(box.querySelectorAll<HTMLElement>('[data-card]')).find((el) => el.dataset.card === key)
  if (!slot) return null
  const cards = slot.parentElement
  let anchor: HTMLElement = slot
  if (cards && slot.previousElementSibling === null) {
    const before = cards.previousElementSibling as HTMLElement | null
    const children = Array.from(box.children) as HTMLElement[]
    const mine = children.filter((child) => child.classList.contains('ck-bubble') && child.classList.contains('me')).at(-1)
    if (key.startsWith('card:') && mine && cards.offsetTop - mine.offsetTop <= box.clientHeight / 2) anchor = mine
    else if (before) anchor = before
  }
  const pad = parseFloat(getComputedStyle(box).paddingTop) || 0
  return Math.max(0, anchor.offsetTop - pad)
}

/** Fade the top edge only while something is scrolled under it. */
function markTop(box: HTMLElement): void {
  box.dataset.top = box.scrollTop > 2 ? 'cut' : ''
}

export default function ChatPanel({ items, cards, input, state, step, tech }: ChatPanelProps) {
  const t = useT(COCKPIT)
  const flow = useRef<HTMLDivElement | null>(null)
  const follow = useRef<Follow>({ mode: 'bottom' })
  /** Where the flow was last put, by the reader or by this panel: a scroll event that lands there is not the reader's. */
  const lastTop = useRef(0)
  const lastKeys = useRef<readonly string[]>([])
  /** The reader is away from the end: the way back is offered. A state, so the button comes and goes. */
  const [away, setAway] = useState(false)
  const revision = revisionOf(items)
  const cardKey = cards.map((card) => card.key).join('|')

  /** Back to the newest lines, and follow them from now on: the reader's own choice. */
  const toNewest = () => {
    const box = flow.current
    follow.current = { mode: 'bottom' }
    setAway(false)
    if (!box) return
    box.scrollTop = box.scrollHeight
    lastTop.current = box.scrollTop
    markTop(box)
  }

  useLayoutEffect(() => {
    const box = flow.current
    if (!box) return
    const keys = cardKey === '' ? [] : cardKey.split('|')
    const opened = keys.filter((key) => !lastKeys.current.includes(key))
    lastKeys.current = keys
    const was = follow.current
    if (opened.length > 0) follow.current = { mode: 'card', key: opened[opened.length - 1] }
    else if (was.mode === 'card' && !keys.includes(was.key)) follow.current = { mode: 'bottom' }
    const now = follow.current
    const bottom = box.scrollHeight - box.clientHeight
    if (now.mode === 'bottom') box.scrollTop = box.scrollHeight
    else if (now.mode === 'card') {
      const top = turnTop(box, now.key)
      box.scrollTop = top === null ? bottom : Math.min(bottom, top)
    }
    lastTop.current = box.scrollTop
    markTop(box)
    // A card that opened brings the reader to it: nobody is away any more.
    if (now.mode !== 'reader') setAway(false)
  }, [revision, cardKey])

  return (
    <section className="ck-chat" aria-label={t('ck.chat.title')}>
      <header className="ck-chat-head">
        <h3>{t('ck.chat.title')}</h3>
        <span className="ck-chat-state">{state}</span>
      </header>
      <div className="ck-flow-box">
        <div
          ref={flow}
          className="ck-flow"
          role="log"
          aria-label={t('ck.chat.log')}
          aria-live="polite"
          onScroll={(e) => {
            const box = e.currentTarget
            const top = box.scrollTop
            // A scroll that lands where the panel put the flow is its own (or the browser keeping the place).
            if (Math.abs(top - lastTop.current) > 1) {
              const gap = box.scrollHeight - top - box.clientHeight
              const down = top > lastTop.current
              const atEnd = gap <= AT_BOTTOM_PX || (down && gap <= nearTheEnd(box))
              follow.current = atEnd ? { mode: 'bottom' } : { mode: 'reader' }
              lastTop.current = top
              setAway(!atEnd)
            }
            markTop(box)
          }}
        >
          {items.length === 0 && cards.length === 0 && (
            <div className="ck-empty">
              <Logo className="ck-empty-mark" />
              <p>{t('ck.chat.empty')}</p>
            </div>
          )}
          {items.map((item, index) => {
            // Lines of Willy's in a row are one turn of his: one label for them all.
            const follows = index > 0 && isWilly(items[index - 1])
            if (item.kind === 'entry') return <Entry key={item.key} entry={item.entry} follows={follows} />
            if (item.kind === 'part') {
              return (
                <div key={item.key} className="ck-willy">
                  {item.current && <span className="ck-who">{t('ck.who.part', { part: item.part.part })}</span>}
                  <PartCard part={item.part} lines={item.lines} current={item.current} step={item.current ? step : null} tech={tech} />
                </div>
              )
            }
            return <ChatLineView key={item.key} line={item.line} tech={tech} willy={saysWilly(item.line)} follows={follows} />
          })}
          {cards.length > 0 && (
            <div className="ck-cards">
              {cards.map((card) => (
                <div key={card.key} data-card={card.key}>
                  {card.node}
                </div>
              ))}
            </div>
          )}
        </div>
        {away && (
          <button type="button" className="ck-newest" onClick={toNewest}>
            <Icon name="chevronDown" size={16} />
            {t('ck.chat.newest')}
          </button>
        )}
      </div>
      <footer className="ck-chat-foot">{input}</footer>
    </section>
  )
}

/** The run's own start and end, and his question after it: Willy's words, drawn as his, not as a step. */
function saysWilly(line: ChatLine): boolean {
  if (line.part !== null) return false
  return line.type === 'run_started' || line.type === 'run_finished' || line.msg.key === 'chat.next' || line.msg.key === 'chat.lost'
}

/** An item drawn as Willy's own words: his replies, and the run's start, end and question. */
function isWilly(item: FlowItem): boolean {
  if (item.kind === 'entry') return item.entry.who === 'willy' && item.entry.kind === 'reply'
  return item.kind === 'line' && saysWilly(item.line)
}

/** One entry of the conversation: the operator's words, Willy's reply, or an earlier task in one line. */
function Entry({ entry, follows }: { entry: ConversationEntry; follows: boolean }) {
  const t = useT(COCKPIT)
  if (entry.who === 'operator') {
    return (
      <div className="ck-bubble me">
        <span className="ck-who">
          {t('ck.who.you')}
          {entry.source === 'spoken' && <span className="ck-badge-spoken">{t('ck.who.spoken')}</span>}
        </span>
        <p>{entry.text}</p>
      </div>
    )
  }
  if (entry.kind === 'summary') {
    return (
      <div className={`ck-summary ${entry.tone ?? 'info'}`}>
        <span className="mono">{t.fmt.time(entry.at)}</span>
        <span>{entry.msg ? t.msg(entry.msg) : ''}</span>
      </div>
    )
  }
  return (
    <div className={`ck-bubble willy ${entry.tone ?? 'info'}${follows ? ' follows' : ''}`}>
      {!follows && <span className="ck-who">{t('ck.who.willy')}</span>}
      <p>{entry.msg ? t.msg(entry.msg) : entry.text}</p>
    </div>
  )
}
