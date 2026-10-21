import { showsTraceMetrics, builderCardHeight, builderCardWidth } from './builderGeometry'
import { CONTAINMENT_SCALE, captureGroupTransforms, compactVisualHierarchy, layoutContinuousScene, sceneGroupId } from './continuousScene'
import { explodeVisualGroup, mergeNodesIntoVisualGroup, visualSelectionItems } from './grouping'
import { captureCanvasLayout } from './deletionLayout'
import type { GraphModel, GraphViewState } from './types'

/** Capture what the user sees before changing ownership. A merge only scales
 * and translates this arrangement; it never assigns new dependency columns. */
export function mergePreservingLayout(graph: GraphModel, nodeIds: string[], displayGraph = graph, groupIds?: string[]) {
  const merged = mergeNodesIntoVisualGroup(graph, nodeIds, groupIds)
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
  const selected = new Set(group.nodeIds)
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

/** Union removes only the selected container wrappers. Nested modules and all
 * computational nodes/edges survive, sharing the new container as their parent. */
export function unionPreservingLayout(graph: GraphModel, nodeIds: string[], groupIds: string[], displayGraph = graph) {
  const selection = visualSelectionItems(graph, nodeIds, groupIds)
  if (!selection.groupIds.length) return mergePreservingLayout(graph, nodeIds, displayGraph, groupIds)
  const merged = mergeNodesIntoVisualGroup(graph, nodeIds, groupIds)
  if (!merged.group) return merged
  const scene = layoutContinuousScene(compactVisualHierarchy(displayGraph))
  const parent = scene.levels.find(level => level.parentId === merged.group!.parentId)
  const levels = selection.groupIds.map(id => scene.levels.find(level => level.parentId === id))
  if (!parent || levels.some(level => !level)) return { graph }
  const ids = [...selection.nodeIds, ...levels.flatMap(level => level!.ids)]
  const scale = parent.scale * CONTAINMENT_SCALE
  const items = ids.map(id => {
    const rect = id.startsWith('visual-group:') ? scene.groups.get(id.slice(13))! : scene.nodes.get(id)!
    const ratio = scale / scene.scales.get(id)!
    return { id, rect: { ...rect, width: rect.width * ratio, height: rect.height * ratio } }
  })
  const left = Math.min(...items.map(item => item.rect.x)), top = Math.min(...items.map(item => item.rect.y))
  const width = Math.max(...items.map(item => item.rect.x + item.rect.width)) - left
  const height = Math.max(...items.map(item => item.rect.y + item.rect.height)) - top
  const frame = { x: left - 16 * parent.scale, y: top - 40 * parent.scale,
    width: Math.max(238 * parent.scale, width + 32 * parent.scale), height: Math.max(136 * parent.scale, height + 56 * parent.scale) }
  const saved = captureCanvasLayout(displayGraph)
  const layouts = saved.preservedLayouts!, bounds = saved.preservedLayoutBounds!, transforms = saved.preservedGroupTransforms!
  const entries = layouts[merged.group.parentId ?? ''] ??= {}
  for (const id of selection.nodeIds) delete entries[id]
  for (const id of selection.groupIds) {
    delete entries[sceneGroupId(id)]
    delete layouts[id]
    delete bounds[id]
    delete transforms[id]
  }
  entries[sceneGroupId(merged.group.id)] = { x: (frame.x - parent.x) / parent.scale, y: (frame.y - parent.y) / parent.scale,
    width: frame.width / parent.scale, height: frame.height / parent.scale, scale: 1 }
  layouts[merged.group.id] = Object.fromEntries(items.map(({ id, rect }) => [id, { x: (rect.x - left) / scale, y: (rect.y - top) / scale,
    width: rect.width / scale, height: rect.height / scale, scale: 1 }]))
  bounds[merged.group.id] = { x: 0, y: 0, width: width / scale, height: height / scale }
  transforms[merged.group.id] = { x: 16, y: 40, scale: CONTAINMENT_SCALE }
  const removed = new Set(selection.groupIds)
  return { group: merged.group, graph: { ...merged.graph,
    groups: merged.graph.groups!.filter(group => !removed.has(group.id)).map(group => removed.has(group.parentId ?? '') ? { ...group, parentId: merged.group!.id } : group),
    view: { ...saved, layoutEdges: merged.graph.view?.layoutEdges, preservedLayouts: layouts, preservedLayoutBounds: bounds, preservedGroupTransforms: transforms,
      expandedGroupIds: saved.expandedGroupIds.filter(id => !removed.has(id)),
      focusedGroupId: removed.has(saved.focusedGroupId ?? '') ? merged.group.id : saved.focusedGroupId,
    },
  } }
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
