import { useCallback, useEffect, useRef, useState } from 'react'
import { deleteRecovery, readRecovery, writeRecovery, type RecoveryRecord, type RecoveryWorkspace } from '../domain/localRecovery'

export function useLocalRecovery(workspace: RecoveryWorkspace, training: boolean) {
  const [pending, setPending] = useState<RecoveryRecord>()
  const [ready, setReady] = useState(false)
  const [error, setError] = useState<string>()
  const [savedAt, setSavedAt] = useState<Date>()
  const queue = useRef(Promise.resolve())
  const lastSaved = useRef<RecoveryWorkspace | undefined>(undefined)
  const latest = useRef(workspace)
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

  const save = useCallback((value: RecoveryWorkspace) => {
    if (!ready) return Promise.resolve()
    // Serialize writes so an older checkpoint can never finish after a newer one.
    queue.current = queue.current.then(async () => {
      try {
        await writeRecovery(value)
        lastSaved.current = value
        setSavedAt(new Date())
        setError(undefined)
      } catch { setError('Local recovery could not be saved. Browser storage may be full or unavailable. Use File → Save for a backup.') }
    })
    return queue.current
  }, [ready])

  useEffect(() => {
    if (!ready || training) return
    const flush = () => { if (lastSaved.current !== latest.current) void save(latest.current) }
    const timer = window.setInterval(flush, 3000)
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
  return { pending, error, savedAt, save, accept, discard }
}
