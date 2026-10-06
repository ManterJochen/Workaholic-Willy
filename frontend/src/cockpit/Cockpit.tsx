/**
 * The Cockpit (OD 2; build plan 4.2; the owner's fifth round, approved on a preview): the one screen a task is given,
 * watched and stopped from.
 *
 * Left, about 62 % of the width: the live camera image (the brightest thing on screen), and under it the task's step
 * timeline, the two stop buttons and Home. Right, beside the image: the task in numbers on top (the success rate
 * leads), and below it a real chat window filling the column: the operator's lines, Willy's, a card per part, the
 * Understood, ask and stop cards in the flow, and the input with the microphone at its foot. Across the top: the ready
 * bar before a task, the run header during one, the stop after a problem. Below 1200 px it stacks: the bar, the image,
 * the run strip, the numbers, the chat.
 *
 * **Nothing here moves the arm without a person's click on a button that names the motion,** bar one: a greeting. A
 * sentence, typed or spoken, is only read (`POST /v1/commands/parse`); the card it opens starts the task only on a
 * click on Start, whose label names the first motion. Home and Restart ask first. A greeting ("Hallo Willy") opens no
 * card, and Willy waves back as the app config says (`runtime.greeting.wave`): at once, the greeting being the person's
 * act (`direct`, the owner's choice of 2026-10-06), after a dialog like Home's (`confirm`), or not at all (`off`); every
 * swing is judged, and the wave is never a way back after a stop. "Sofort anhalten" is one click and is not the
 * e-stop. After a stop nothing starts by itself: the stop card waits for a person.
 *
 * It reads the console's shared models (the one cell poll, the run on screen, the conversation, the preferences) and
 * polls nothing of the cell itself; only the live image is its own. It fills the window under the top bar and no more
 * (the stop strip always ends on screen); below 1200 px it stacks, and the stop buttons stay in view as the page
 * scrolls. After a reload under a stop record, the stopped run is followed again, so its parts, its numbers and where
 * it stopped are drawn as before the reload.
 */

import { useEffect, useMemo, useRef, useState } from 'react'

import { useLang, useT } from '../i18n'
import { blockerMsg, lightIdMsg, lightMsg, refusalMsg, runKindMsg } from '../i18n/codes'
import { say, useConversation } from '../model/chat'
import { usePrefs } from '../model/prefs'
import { EMPTY_RUN, hasEnded, STEPS, type RunView, type StepId } from '../model/runModel'
import { useCell } from '../model/useCell'
import { useRun } from '../model/useRun'
import { PromptInput } from '../prompt/PromptInput'
import AskCard from './AskCard'
import CellFacts from './CellFacts'
import ChatPanel, { type ChatCard } from './ChatPanel'
import { chatFlow } from './chatFlow'
import { draftFromPlan, firstMotion, MOTION_KEY } from './draft'
import { useEarlierTasks, usePoses, useRunRecord, useWindowFit } from './hooks'
import { COCKPIT } from './i18n'
import ReadyBar from './ReadyBar'
import { controllerStopped } from './recovery'
import Stage, { type StageBanner } from './Stage'
import Stats from './Stats'
import StopButtons from './StopButtons'
import StopCard from './StopCard'
import Timeline from './Timeline'
import UnderstoodCard from './UnderstoodCard'
import { useMotions } from './motions'
import { useCommand } from './useCommand'
import VoiceOut from './VoiceOut'
import './cockpit.css'

/** A next-instruction question older than this when the page drew it is a replay: it does not take the focus. */
const LIVE_S = 5

function stepOf(value: unknown): StepId | null {
  return typeof value === 'string' && (STEPS as readonly string[]).includes(value) ? (value as StepId) : null
}

export default function Cockpit() {
  const t = useT(COCKPIT)
  const { lang } = useLang()
  const prefs = usePrefs()
  const tech = prefs.view === 'tech'
  const { cell, readiness, facts, stream, refresh } = useCell()
  const { view, follow } = useRun()
  const conversation = useConversation()
  const motions = useMotions()
  const command = useCommand({ lang, follow, refresh, greet: motions.wave })
  const [mountedAt] = useState(() => Date.now() / 1000)
  const [text, setText] = useState('')
  const [source, setSource] = useState<'typed' | 'spoken'>('typed')
  const [askClosed, setAskClosed] = useState<string | null>(null)
  const [lastTask, setLastTask] = useState<RunView>(EMPTY_RUN)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const box = useRef<HTMLDivElement | null>(null)
  useWindowFit(box)
  // The session's earlier tasks, one line each, also in a tab that did not see them run: the chat is the history.
  useEarlierTasks()

  // ── what holds the cell ──────────────────────────────────────────────────────────────────────────────────────
  const connected = cell?.state === 'connected'
  const activeId = cell?.active_run_id ?? null
  const viewLive = view.runId !== null && !hasEnded(view.phase) && view.phase !== 'idle'
  // The poll may still name a run whose end the stream already drew: that run holds nothing any more.
  const running = viewLive || (activeId !== null && !(view.runId === activeId && hasEnded(view.phase)))
  const teaching = running && view.kind === 'teach'
  const recovery = cell?.recovery ?? null

  // The numbers keep the last task's (or pick run's) through a Home move, a planner start or a teach: remembered
  // while rendering (React's pattern for a value derived from an earlier render), never through an effect.
  const counted = view.kind === 'task' || view.kind === 'pick'
  if (counted && view !== lastTask) setLastTask(view)
  const statsView = counted ? view : lastTask

  const poses = usePoses(`${cell?.state ?? ''}:${view.kind === 'teach' && hasEnded(view.phase) ? view.runId : ''}`)
  const stoppedRecord = useRunRecord(recovery?.run_id ?? null, view.runId === recovery?.run_id ? view.run : null)
  const stoppedRun = stoppedRecord.run
  const stoppedAt =
    recovery && view.runId === recovery.run_id && view.stopCard
      ? { part: view.stopCard.part, step: view.stopCard.step }
      : { part: stoppedRun && stoppedRun.kind === 'task' ? (stoppedRun.parts_placed ?? 0) + 1 : null, step: stepOf(stoppedRun?.step) }

  // A page opened (or reloaded) under a stop record shows no run yet: follow the stopped one, read again from its
  // record and its events, so the chat, the numbers and the timeline are its own. It only watches; it starts nothing.
  const followStopped = view.runId === null && stoppedRun !== null && stoppedRun.id === recovery?.run_id ? stoppedRun : null
  useEffect(() => {
    if (followStopped) follow(followStopped)
  }, [followStopped, follow])

  // ── the stage's banner: what stops the arm, over the dimmed image ────────────────────────────────────────────
  let banner: StageBanner | null = null
  if (controllerStopped(readiness)) banner = { kind: 'stopped' }
  else if (running && view.countdown.state === 'active') banner = { kind: 'countdown', seconds: view.countdown.secondsLeft }
  else if (cell?.halted && view.phase !== 'halting') banner = { kind: 'halted' }
  else if (recovery?.stop_code === 'halted' && !running && !(recovery.cleared_at != null && recovery.cleared_at > recovery.at)) {
    banner = { kind: 'halted' }
  }

  // ── Start, and why it is off for the cell's sake ─────────────────────────────────────────────────────────────
  const cellOff: string[] = []
  if (!connected) cellOff.push(t.msg(refusalMsg('not_connected')))
  else if (running) cellOff.push(t.msg(blockerMsg('run_active')))
  else if (recovery) cellOff.push(t.msg(blockerMsg('restart_required')))
  else if (!readiness) cellOff.push(t('ck.start.readinessUnknown'))
  else if (!readiness.ready) {
    for (const light of readiness.lights ?? []) {
      if (light.blocks && light.state !== 'ok') cellOff.push(t('ck.start.light', { id: lightIdMsg(light.id), state: lightMsg(light.code) }))
    }
    for (const blocker of readiness.blockers ?? []) cellOff.push(t.msg(blockerMsg(blocker.code)))
    if (cellOff.length === 0) cellOff.push(t('ck.ready.not'))
  }
  const canStart = cellOff.length === 0
  const motion = t(MOTION_KEY[firstMotion(facts)])

  // ── the chat: the conversation, the run on screen, the cell's lines of this session ─────────────────────────
  // The cell's lines from where this tab's own conversation begins: an earlier task read from the server is one line
  // of its own, and does not bring back every cell line since it ran.
  const ownFirst = conversation.find((entry) => entry.kind !== 'summary')
  const since = Math.min(mountedAt, ownFirst ? ownFirst.at : mountedAt)
  const items = useMemo(
    () => chatFlow({ conversation, view, cellLines: stream.lines, since, tech }),
    [conversation, view, stream.lines, since, tech],
  )
  const askOpen = view.askCard !== null && askClosed !== view.askCard.runId && !running

  // The followed run ended: ask the cell now rather than at the next tick, so the ready bar, Home and the top bar's
  // chips leave the run at once.
  const ended = view.runId !== null && hasEnded(view.phase)
  useEffect(() => {
    if (ended) refresh()
  }, [ended, view.runId, refresh])

  // ── the input: read on Enter, never start; locked during a run, a teach, and under a stop record ────────────
  const inputOff = running || recovery !== null
  // What runs, in the operator's words ("Auftrag", "Home-Fahrt"), never the console's ("Lauf").
  const runningKind = running && view.kind !== null && (view.runId === activeId || viewLive) ? view.kind : null
  const offReason = running
    ? teaching
      ? t('ck.chat.blocked.teach')
      : runningKind
        ? t('ck.chat.blocked.run', { kind: runKindMsg(runningKind) })
        : t('ck.chat.blocked.busy')
    : recovery
      ? t('ck.chat.blocked.recovery')
      : undefined
  const submit = () => {
    if (inputOff || !text.trim()) return
    void command.read(text, source)
    setText('')
    setSource('typed')
  }

  // After a task, Willy asks for the next instruction and the input takes the focus (item 20), when that happened
  // now: a run replayed after a reload does not take the focus away from wherever it is.
  const nextLine = view.nextQuestion ? view.chat.find((line) => line.msg.key === 'chat.next') : undefined
  const nextLive = nextLine !== undefined && nextLine.at >= mountedAt - LIVE_S
  useEffect(() => {
    if (nextLive && !inputOff) inputRef.current?.focus()
  }, [nextLive, inputOff, nextLine?.id])

  const chatState =
    command.busy === 'reading'
      ? t('ck.chat.state.reading')
      : running
        ? runningKind
          ? t('ck.chat.state.running', { kind: runKindMsg(runningKind) })
          : t('ck.chat.state.busy')
        : recovery || askOpen || command.draft
          ? t('ck.chat.state.waiting')
          : t('ck.chat.state.idle')

  const endAsk = () => {
    if (!view.askCard) return
    setAskClosed(view.askCard.runId)
    say({ who: 'willy', kind: 'reply', msg: { key: 'ck.ask.ended' } })
    inputRef.current?.focus()
  }

  const showStop = recovery !== null
  const showCard = command.draft !== null && !running && !recovery
  // The cards in the chat's flow, oldest first, each keyed by what it is for: the chat keeps a card that opens in view,
  // from the start of its turn, and a redraw of the same card (an edit, a poll) is not news.
  const cards: ChatCard[] = []
  if (askOpen && view.askCard) {
    cards.push({
      key: `ask:${view.askCard.runId}`,
      node: (
        <AskCard
          ask={view.askCard}
          plan={view.plan}
          poses={poses}
          motion={motion}
          countdown={cell?.countdown_due === true}
          canStart={canStart}
          busy={command.busy !== null}
          error={command.draft ? null : command.error}
          start={(task) => void command.startTask(task)}
          edit={(plan) => command.open(draftFromPlan(plan))}
          end={endAsk}
        />
      ),
    })
  }
  if (showStop && recovery) {
    cards.push({
      key: `stop:${recovery.run_id}`,
      node: (
        <StopCard
          record={stoppedRecord}
          homeTo={view.runId === recovery.run_id && view.kind === 'home' ? (view.to ?? 'home') : null}
          running={running}
          part={stoppedAt.part}
          step={stoppedAt.step}
          poses={poses}
          tech={tech}
        />
      ),
    })
  }
  if (showCard && command.draft) {
    cards.push({
      key: `card:${command.draft.id}`,
      node: (
        <UnderstoodCard
          draft={command.draft}
          poses={poses}
          facts={facts}
          tech={tech}
          motion={motion}
          countdown={cell?.countdown_due === true}
          cellOff={cellOff}
          busy={command.busy}
          error={command.error}
          update={command.update}
          start={() => void command.start()}
          discard={command.discard}
          load={() => void command.load()}
          retry={() => command.lastSaid && void command.read(command.lastSaid.text, command.lastSaid.source)}
        />
      ),
    })
  }

  const input = (
    <PromptInput
      value={text}
      onChange={(next, from) => {
        setText(next)
        setSource(from)
      }}
      canSubmit={!inputOff && text.trim() !== '' && command.busy === null}
      onSubmit={submit}
      spoken={source === 'spoken'}
      disabled={inputOff}
      disabledReason={offReason}
      inputRef={inputRef}
      showSend
      busy={command.busy !== null}
    />
  )

  return (
    <div className="cockpit" ref={box}>
      <h2 className="sr-only">{t('nav.cockpit')}</h2>
      {/* Voice output, off unless switched on: the start, the end, a problem, a question (OD 18). */}
      <VoiceOut />
      <ReadyBar
        mode={running ? 'running' : recovery ? 'stopped' : 'ready'}
        view={view}
        poses={poses}
        stoppedRun={stoppedRun}
        stoppedAt={stoppedAt}
        tech={tech}
        onLoad={() => void command.load()}
      />
      <div className="ck-main">
        <Stage teaching={teaching} overlays={view.overlays} wrist={facts?.wrist_camera === true} banner={banner} pinGrasps={tech} />
        <div className="ck-strip">
          <Timeline view={view} facts={facts} />
          <StopButtons view={view} running={running} facts={facts} />
        </div>
        {tech && <CellFacts facts={facts} />}
      </div>
      <div className="ck-side">
        <Stats view={statsView} tech={tech} />
        <ChatPanel
          items={items}
          cards={cards}
          input={input}
          state={chatState}
          step={running ? view.current.step : null}
          tech={tech}
        />
      </div>
    </div>
  )
}
