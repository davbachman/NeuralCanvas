import type { GraphViewState } from '../domain/types'
import { useCallback, useEffect, useRef, useState } from 'react'
import { deleteRecovery, readRecovery, writeRecovery, sameRecoveryContent, type RecoveryRecord, type RecoveryWorkspace } from '../domain/localRecovery'

export function useLocalRecovery(workspace: RecoveryWorkspace, training: boolean) {
  const [pending, setPending] = useState<RecoveryRecord>()
  const [ready, setReady] = useState(false)
  const [error, setError] = useState<string>()
  const [savedAt, setSavedAt] = useState<Date>()
  const queue = useRef(Promise.resolve())
  const lastSaved = useRef<RecoveryWorkspace | undefined>(undefined)
  const latest = useRef(workspace)
  const navigation = useRef({ active: false, endedAt: 0 })
  const savedTime = useRef(0)
  const onNavigationChange = useCallback((active: boolean, viewport?: GraphViewState['viewport']) => {
    navigation.current = { active, endedAt: Date.now() }
    // Keep the live camera available to pagehide/visibility flushes, without
    // publishing a model update for each pointer event.
    if (viewport) {
      const current = latest.current
      latest.current = { ...current, state: { ...current.state, graph: { ...current.state.graph,
        view: { ...current.state.graph.view, expandedGroupIds: current.state.graph.view?.expandedGroupIds ?? [], viewport },
      } } }
    }
  }, [])
  useEffect(() => { latest.current = workspace }, [workspace])
  useEffect(() => {
    let active = true
    readRecovery().then(record => {
      if (!active) return
      if (record && record.file.state.graph.nodes.length) setPending(record)
      else setReady(true)
    }).catch(reason => { if (active) setError(String(reason instanceof Error ? reason.message : reason)) })
    return () => { active = false }
  }, [])

  const save = useCallback((value?: RecoveryWorkspace) => {
    if (!ready) return Promise.resolve()
    // Serialize writes so an older checkpoint can never finish after a newer one.
    queue.current = queue.current.then(async () => {
      try {
        // A camera-only save during training starts from the latest durable
        // checkpoint when its turn arrives, never from stale canvas weights.
        const base = value ?? lastSaved.current
        if (!base) return
        const viewport = latest.current.state.graph.view?.viewport
        const previous = base.state.graph.view?.viewport
        const snapshot = viewport && (viewport.x !== previous?.x || viewport.y !== previous?.y || viewport.zoom !== previous?.zoom)
          ? { ...base, state: { ...base.state, graph: { ...base.state.graph,
              view: { ...base.state.graph.view, expandedGroupIds: base.state.graph.view?.expandedGroupIds ?? [], viewport },
            } } }
          : base
        if (lastSaved.current === snapshot) return
        await writeRecovery(snapshot)
        lastSaved.current = snapshot
        savedTime.current = Date.now()
        setSavedAt(new Date())
        setError(undefined)
      } catch { setError('Local recovery could not be saved. Browser storage may be full or unavailable. Use File → Save for a backup.') }
    })
    return queue.current
  }, [ready])

  useEffect(() => {
    if (!ready) return
    const flush = () => {
      if (training) void save()
      else if (lastSaved.current !== latest.current) void save(latest.current)
    }
    const timer = window.setInterval(() => {
      // Navigation writes are cheap patches, coalesced until the gesture settles.
      // Bound deferral, and never delay edits/checkpoints or lifecycle flushes.
      const moving = navigation.current.active || Date.now() - navigation.current.endedAt < 500
      if (moving && lastSaved.current && (training || sameRecoveryContent(lastSaved.current, latest.current)) && Date.now() - savedTime.current < 10000) return
      flush()
    }, 3000)
    const hidden = () => { if (document.visibilityState === 'hidden') flush() }
    window.addEventListener('pagehide', flush)
    document.addEventListener('visibilitychange', hidden)
    return () => {
      window.clearInterval(timer)
      window.removeEventListener('pagehide', flush)
      document.removeEventListener('visibilitychange', hidden)
    }
  }, [ready, training, save])

  const accept = () => { setPending(undefined); setReady(true) }
  const discard = async () => {
    try { await deleteRecovery(); setPending(undefined); setError(undefined); setReady(true) }
    catch { setError('The local recovery copy could not be removed. Use File → Save to protect your work.') }
  }
  return { pending, error, savedAt, save, accept, discard, onNavigationChange }
}
