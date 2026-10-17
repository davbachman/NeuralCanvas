import { useCallback, useLayoutEffect, useRef } from 'react'

/** Stable event handlers for memoized canvas data, with the latest committed
 * state. Unlike a stale useCallback closure, edits after navigation stay current. */
export function useStableCallback<Args extends unknown[], Result>(callback: (...args: Args) => Result): (...args: Args) => Result {
  const latest = useRef(callback)
  useLayoutEffect(() => { latest.current = callback })
  return useCallback((...args: Args) => latest.current(...args), [])
}
