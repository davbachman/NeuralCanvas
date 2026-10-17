import { useMemo } from 'react'
import type { GraphModel, GraphViewState } from '../domain/types'

/** Camera persistence must not invalidate numerical/trace/layout derivations.
 * The small view key also tolerates callers copying unchanged view arrays. */
export function useModelGraph(graph: GraphModel): GraphModel {
  const { nodes, edges, groups, learningRate, training } = graph
  const viewKey = useMemo(() => {
    if (!graph.view) return undefined
    const { viewport: _viewport, ...view } = graph.view
    void _viewport
    return JSON.stringify(view)
  }, [graph.view])
  const view = useMemo(() => viewKey === undefined ? undefined : JSON.parse(viewKey) as GraphViewState, [viewKey])
  return useMemo(() => ({ nodes, edges, groups, learningRate, training, view }), [nodes, edges, groups, learningRate, training, view])
}
