import { useLayoutEffect, useRef, type RefObject } from 'react'
import { useStableCallback } from '../hooks/useStableCallback'

/** Keep the selection outline one screen pixel wide without restyling the
 * canvas or publishing React state on every flat-canvas zoom tick. */
export function useCanvasSelectionStyle(shell: RefObject<HTMLDivElement | null>, zoom: number): (zoom: number) => void {
  const liveZoom = useRef(zoom)
  const update = useStableCallback((nextZoom: number) => {
    liveZoom.current = nextZoom
    shell.current?.querySelectorAll<HTMLElement>('.react-flow__nodesselection-rect').forEach(outline => {
      outline.style.setProperty('--selection-border-width', `${1 / nextZoom}px`)
    })
  })
  useLayoutEffect(() => { update(zoom) }, [zoom, update])
  useLayoutEffect(() => {
    const root = shell.current
    if (!root) return
    // React Flow can mount an outline without a parent canvas render.
    const observer = new MutationObserver(() => update(liveZoom.current))
    observer.observe(root, { childList: true, subtree: true })
    return () => observer.disconnect()
  }, [shell, update])
  return update
}
