import { showsTraceMetrics, builderCardHeight, builderCardWidth } from './builderGeometry'
import { captureGroupTransforms, compactVisualHierarchy, layoutContinuousScene, sceneGroupId } from './continuousScene'
import { explodeVisualGroup, mergeNodesIntoVisualGroup } from './grouping'
import { captureCanvasLayout } from './deletionLayout'
import type { GraphModel, GraphViewState } from './types'

/** Capture what the user sees before changing ownership. A merge only scales
 * and translates this arrangement; it never assigns new dependency columns. */
export function mergePreservingLayout(graph: GraphModel, nodeIds: string[], displayGraph = graph) {
  const merged = mergeNodesIntoVisualGroup(graph, nodeIds)
  if (!merged.group) return merged
  const layouts: NonNullable<GraphViewState['preservedLayouts']> = {}
  let transforms: GraphViewState['preservedGroupTransforms'] = {}
  if (displayGraph.groups?.length || displayGraph.view?.preservedLayouts) {
    const scene = layoutContinuousScene(compactVisualHierarchy(displayGraph))
    transforms = captureGroupTransforms(scene)
    for (const level of scene.levels) {
      const entries: typeof layouts[string] = {}
      for (const id of level.ids) {
        const rect = id.startsWith('visual-group:') ? scene.groups.get(id.slice(13))! : scene.nodes.get(id)!
        entries[id] = { x: (rect.x - level.x) / level.scale, y: (rect.y - level.y) / level.scale,
          width: rect.width / level.scale, height: rect.height / level.scale, scale: (scene.scales.get(id) ?? 1) / level.scale }
      }
      layouts[level.parentId ?? ''] = entries
    }
  } else {
    layouts[''] = Object.fromEntries(displayGraph.nodes.map(node => [node.id, {
      ...node.position, width: builderCardWidth(node), height: builderCardHeight(node, showsTraceMetrics(displayGraph)), scale: 1,
    }]))
  }
  const bounds = Object.fromEntries(Object.entries(layouts).flatMap(([key, entries]) => {
    const rects = Object.values(entries)
    if (!rects.length) return []
    const x = Math.min(...rects.map(r => r.x)), y = Math.min(...rects.map(r => r.y))
    return [[key, graph.view?.preservedLayoutBounds?.[key] ?? { x, y, width: Math.max(...rects.map(r => r.x + r.width)) - x, height: Math.max(...rects.map(r => r.y + r.height)) - y }]]
  }))
  const group = merged.group
  const parent = layouts[group.parentId ?? '']
  // Display-only hierarchy projections can hide a canonical parent. Fall back
  // to the model level that actually contains the selected blocks.
  const level = parent ?? Object.values(layouts).find(entries => nodeIds.some(id => entries[id]))
  if (!level) return merged
  const selected = new Set(nodeIds)
  const selectedKeys = Object.keys(level).filter(key => selected.has(key) ||
    displayGraph.groups?.some(candidate => key === sceneGroupId(candidate.id) && candidate.nodeIds.every(id => selected.has(id))))
  if (!selectedKeys.length) return merged
  const content = Object.fromEntries(selectedKeys.map(key => [key, level[key]]))
  const rects = Object.values(content)
  const left = Math.min(...rects.map(rect => rect.x)), top = Math.min(...rects.map(rect => rect.y))
  const width = Math.max(...rects.map(rect => rect.x + rect.width)) - left
  const height = Math.max(...rects.map(rect => rect.y + rect.height)) - top
  const scale = 1
  selectedKeys.forEach(key => { delete level[key] })
  level[sceneGroupId(group.id)] = { x: left + (width - 238 * scale) / 2, y: top + (height - 136 * scale) / 2,
    width: 238 * scale, height: 136 * scale, scale }
  layouts[group.id] = content
  bounds[group.id] = { x: left, y: top, width, height }
  delete transforms[group.id]
  return { ...merged, graph: { ...merged.graph, view: { ...merged.graph.view!, preservedLayouts: layouts, preservedLayoutBounds: bounds, preservedGroupTransforms: transforms,
    layoutOffsets: undefined, manualNodePlacements: undefined } } }
}

/** Undo the container's display compression. Keep edits in its local layout,
 * and expand around its current center so moving the group is preserved. */
export function ungroupPreservingLayout(graph: GraphModel, groupId: string) {
  const group = graph.groups?.find(candidate => candidate.id === groupId)
  if (!group || !graph.view?.preservedLayouts?.[groupId]) return explodeVisualGroup(graph, groupId)
  const scene = layoutContinuousScene(compactVisualHierarchy(graph))
  const parent = scene.levels.find(level => level.parentId === group.parentId)
  const contents = scene.levels.find(level => level.parentId === groupId)
  if (!parent || !contents) return explodeVisualGroup(graph, groupId)
  const saved = captureCanvasLayout(graph)
  const layouts = saved.preservedLayouts!
  const bounds = { ...saved.preservedLayoutBounds }
  const transforms = { ...saved.preservedGroupTransforms }
  const frameScale = scene.scales.get(sceneGroupId(groupId))!
  const offset = {
    x: (contents.x + (contents.bounds.x + contents.bounds.width / 2) * contents.scale - parent.x) / parent.scale - (contents.bounds.x + contents.bounds.width / 2),
    y: (contents.y + (contents.bounds.y + contents.bounds.height / 2) * contents.scale - 10 * frameScale - parent.y) / parent.scale - (contents.bounds.y + contents.bounds.height / 2),
  }
  const entries = layouts[group.parentId ?? ''] ?? {}
  delete entries[sceneGroupId(groupId)]
  for (const id of contents.ids) {
    const rect = layouts[groupId][id]
    entries[id] = { ...rect, x: rect.x + offset.x, y: rect.y + offset.y }
  }
  layouts[group.parentId ?? ''] = entries
  delete layouts[groupId]
  delete bounds[groupId]
  delete transforms[groupId]
  const result = explodeVisualGroup({ ...graph, view: saved }, groupId)
  return { ...result, view: { ...result.view!, preservedLayouts: layouts, preservedLayoutBounds: bounds, preservedGroupTransforms: transforms } }
}
