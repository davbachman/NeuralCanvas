import { cloneGraph } from './engine'
import { groupAncestors } from './grouping'
import { builderCardHeight, builderCardWidth, showsTraceMetrics } from './builderGeometry'
import { compactVisualHierarchy, layoutContinuousScene, sceneContentBounds, sceneGroupId } from './continuousScene'
import { captureCanvasLayout } from './deletionLayout'
import type { SemanticRect } from './semanticLayout'
import type { GraphEdge, GraphGroup, GraphModel, GraphNode, GraphViewState, NodeType, Position } from './types'

export interface GraphClipboardSelection {
  nodeIds: string[]
  groupId?: string
}

export interface GraphClipboardFragment {
  nodes: GraphNode[]
  edges: GraphEdge[]
  group?: GraphGroup
  groups?: GraphGroup[]
  parentGroupId?: string
  layout?: Pick<GraphViewState, 'preservedLayouts' | 'preservedLayoutBounds' | 'preservedGroupTransforms'>
}

export interface GraphClipboardPasteResult {
  graph: GraphModel
  selection: GraphClipboardSelection
}

export function copyGraphSelection(
  graph: GraphModel,
  selection: GraphClipboardSelection,
): GraphClipboardFragment | undefined {
  if (selection.groupId) {
    return copyGroupSelection(graph, selection.groupId)
  }

  const selectedNodeIds = unique(selection.nodeIds)
  if (selectedNodeIds.length === 0) return undefined

  const selectedNodeIdSet = new Set(selectedNodeIds)
  const nodes = selectedNodeIds
    .map((nodeId) => graph.nodes.find((node) => node.id === nodeId))
    .filter((node): node is GraphNode => Boolean(node))
  if (nodes.length === 0) return undefined

  return captureClipboardLayout(graph, {
    nodes,
    edges: internalEdgesForNodeIds(graph, selectedNodeIdSet),
  })
}

export function pasteGraphClipboard(
  graph: GraphModel,
  fragment: GraphClipboardFragment,
  offset: Position,
): GraphClipboardPasteResult {
  const source = cloneClipboardFragment(fragment)
  const destinationParent = clipboardParent(graph, source)
  const clipboardParentId = destinationParent?.group.id ?? ''
  const existingNodeIds = new Set(graph.nodes.map((node) => node.id))
  const usedNodeIds = new Set(existingNodeIds)
  const usedEdgeIds = new Set(graph.edges.map((edge) => edge.id))
  const usedGroupIds = new Set((graph.groups ?? []).map((group) => group.id))
  const nodeIdMap = new Map<string, string>()

  const pastedNodes = source.nodes.map((node) => {
    const { id, index } = nextNodeId(node.type, usedNodeIds)
    nodeIdMap.set(node.id, id)
    return {
      ...node,
      id,
      label: labelForPastedNode(node, index),
      position: translatePosition(node.position, offset),
    }
  })

  const pastedEdges = source.edges.flatMap((edge) => {
    // A copied neuron keeps its existing inputs, but its output is a new
    // connection point. Reusing external destinations would attach two wires
    // to the same input port and silently change the original computation.
    if (nodeIdMap.has(edge.source) && !nodeIdMap.has(edge.target)) return []
    const sourceId = nodeIdMap.get(edge.source) ?? (existingNodeIds.has(edge.source) ? edge.source : undefined)
    const targetId = nodeIdMap.get(edge.target) ?? (existingNodeIds.has(edge.target) ? edge.target : undefined)
    if (!sourceId || !targetId) return []

    return [
      {
        ...edge,
        id: nextCopyId(edge.id, usedEdgeIds),
        source: sourceId,
        target: targetId,
      },
    ]
  })

  const originalGroups = source.group ? [source.group, ...(source.groups ?? [])] : []
  const pastedGroups = originalGroups.map((group) => createPastedGroup(group, usedGroupIds, nodeIdMap, offset))
  const groupIdMap = new Map(originalGroups.map((group, index) => [group.id, pastedGroups[index].id]))
  pastedGroups.forEach((group, index) => { group.parentId = groupIdMap.get(originalGroups[index].parentId ?? '') })
  const pastedGroup = pastedGroups[0]
  if (pastedGroup) pastedGroup.parentId = destinationParent?.group.id
  const ancestorIds = new Set(destinationParent ? groupAncestors(graph, destinationParent.group.id).map(group => group.id) : [])
  const existingGroups = graph.groups?.map(group => ancestorIds.has(group.id)
    ? { ...group, nodeIds: [...group.nodeIds, ...pastedNodes.map(node => node.id)] } : group)
  let view = graph.view
  if (fragment.layout || graph.groups?.length || graph.view?.preservedLayouts) {
    const saved = graph.groups?.length || graph.view?.preservedLayouts ? captureCanvasLayout(graph) : {
      ...graph.view, expandedGroupIds: graph.view?.expandedGroupIds ?? [],
      preservedLayouts: { '': Object.fromEntries(graph.nodes.map(node => [node.id, nodeRect(graph, node)])) },
    }
    const layouts: NonNullable<GraphViewState['preservedLayouts']> = { ...saved.preservedLayouts, '': { ...saved.preservedLayouts?.[''] } }
    const bounds = { ...saved.preservedLayoutBounds }
    const transforms = { ...saved.preservedGroupTransforms }
    const remap = (id: string) => id.startsWith('visual-group:')
      ? groupIdMap.has(id.slice(13)) ? sceneGroupId(groupIdMap.get(id.slice(13))!) : undefined
      : nodeIdMap.get(id)
    const copiedLayouts = source.layout?.preservedLayouts ?? { '': Object.fromEntries(source.nodes.map(node => [node.id, nodeRect(graph, node)])) }
    for (const [parent, entries] of Object.entries(copiedLayouts)) {
      const destination = parent ? groupIdMap.get(parent) : clipboardParentId
      if (destination === undefined) continue
      const translated = Object.fromEntries(Object.entries(entries).flatMap(([id, rect]) => {
        const key = remap(id)
        const placed = { ...rect, ...(parent ? {} : translatePosition(rect, offset)) }
        if (!parent) {
          const scale = destinationParent?.level.scale ?? 1
          placed.width *= scale / placed.scale
          placed.height *= scale / placed.scale
          placed.scale = scale
        }
        if (!parent && destinationParent) {
          const level = destinationParent.level
          placed.x = (placed.x - level.x) / level.scale
          placed.y = (placed.y - level.y) / level.scale
          placed.width /= level.scale
          placed.height /= level.scale
          placed.scale /= level.scale
        }
        return key ? [[key, placed]] : []
      }))
      layouts[destination] = { ...layouts[destination], ...translated }
      const bound = source.layout?.preservedLayoutBounds?.[parent]
      if (parent && bound) bounds[destination] = { ...bound }
      const transform = source.layout?.preservedGroupTransforms?.[parent]
      if (parent && transform) transforms[destination] = { ...transform }
    }
    view = { ...saved, preservedLayouts: layouts, preservedLayoutBounds: bounds, preservedGroupTransforms: transforms }
  }

  return {
    graph: {
      ...graph,
      nodes: [...graph.nodes, ...pastedNodes],
      edges: [...graph.edges, ...pastedEdges],
      groups: pastedGroup ? [...(existingGroups ?? []), ...pastedGroups] : existingGroups,
      view,
    },
    selection: pastedGroup
      ? { nodeIds: [], groupId: pastedGroup.id }
      : { nodeIds: pastedNodes.map((node) => node.id) },
  }
}

function copyGroupSelection(graph: GraphModel, groupId: string): GraphClipboardFragment | undefined {
  const group = graph.groups?.find((candidate) => candidate.id === groupId)
  if (!group) return undefined

  const selectedNodeIdSet = new Set(group.nodeIds)
  const nodes = group.nodeIds
    .map((nodeId) => graph.nodes.find((node) => node.id === nodeId))
    .filter((node): node is GraphNode => Boolean(node))
  if (nodes.length === 0) return undefined

  return captureClipboardLayout(graph, {
    nodes,
    edges: boundaryAndInternalEdgesForNodeIds(graph, selectedNodeIdSet),
    group,
    groups: (graph.groups ?? []).filter((candidate) => candidate.id !== group.id && groupAncestors(graph, candidate.id).some((ancestor) => ancestor.id === group.id)),
  })
}

function nodeRect(graph: GraphModel, node: GraphNode) {
  return { ...node.position, width: builderCardWidth(node), height: builderCardHeight(node, showsTraceMetrics(graph)), scale: 1 }
}

/** Stored model coordinates can differ from the drawing after grouping or
 * dragging. Copy the visible arrangement and the internal layout of modules. */
function captureClipboardLayout(graph: GraphModel, fragment: GraphClipboardFragment): GraphClipboardFragment {
  if (!graph.groups?.length && !graph.view?.preservedLayouts) return cloneClipboardFragment(fragment)
  const scene = layoutContinuousScene(compactVisualHierarchy(graph))
  const saved = captureCanvasLayout(graph)
  const layouts: NonNullable<GraphViewState['preservedLayouts']> = { '': {} }
  const bounds: NonNullable<GraphViewState['preservedLayoutBounds']> = {}
  const transforms: NonNullable<GraphViewState['preservedGroupTransforms']> = {}
  const ids = fragment.group ? [sceneGroupId(fragment.group.id)] : fragment.nodes.map(node => node.id)
  const parentGroupId = fragment.group ? scene.parents.get(sceneGroupId(fragment.group.id))
    : (graph.groups ?? []).filter(group => fragment.nodes.every(node => group.nodeIds.includes(node.id)) && scene.levels.some(level => level.parentId === group.id))
      .sort((a, b) => groupAncestors(graph, b.id).length - groupAncestors(graph, a.id).length)[0]?.id
  for (const id of ids) {
    const rect = id.startsWith('visual-group:') ? scene.groups.get(id.slice(13)) : scene.nodes.get(id)
    if (rect) layouts[''][id] = { ...rect, scale: scene.scales.get(id) ?? 1 }
  }
  for (const group of fragment.group ? [fragment.group, ...(fragment.groups ?? [])] : []) {
    if (saved.preservedLayouts?.[group.id]) layouts[group.id] = saved.preservedLayouts[group.id]
    if (saved.preservedLayoutBounds?.[group.id]) bounds[group.id] = saved.preservedLayoutBounds[group.id]
    if (saved.preservedGroupTransforms?.[group.id]) transforms[group.id] = saved.preservedGroupTransforms[group.id]
  }
  return cloneClipboardFragment({ ...fragment, parentGroupId, layout: { preservedLayouts: layouts, preservedLayoutBounds: bounds, preservedGroupTransforms: transforms } })
}

/** A copy remains a sibling of its source while that container still exists.
 * Check membership too, since group IDs can be reused after deleting a module. */
function clipboardParent(graph: GraphModel, fragment: GraphClipboardFragment) {
  const group = graph.groups?.find(group => group.id === fragment.parentGroupId && fragment.nodes.every(node => group.nodeIds.includes(node.id)))
  if (!group) return undefined
  const scene = layoutContinuousScene(compactVisualHierarchy(graph))
  const level = scene.levels.find(level => level.parentId === group.id)
  return level ? { group, scene, level } : undefined
}

/** Use a small screen-space cascade, and bring off-screen copies into view.
 * Clamp the selection as a whole so relative positions are never changed. */
export function clipboardPasteOffset(graph: GraphModel, fragment: GraphClipboardFragment, count = 0, viewport?: SemanticRect & { zoom: number }): Position {
  const parent = clipboardParent(graph, fragment), scale = parent?.level.scale ?? 1
  const rects = Object.values(fragment.layout?.preservedLayouts?.[''] ?? Object.fromEntries(fragment.nodes.map(node => [node.id, nodeRect(graph, node)])))
    .map(rect => ({ ...rect, width: rect.width * scale / rect.scale, height: rect.height * scale / rect.scale }))
  const step = 36 * (count % 5 + 1) / (viewport?.zoom ?? 1)
  if (!rects.length) return { x: step, y: step }
  const x = Math.min(...rects.map(rect => rect.x)), y = Math.min(...rects.map(rect => rect.y))
  const width = Math.max(...rects.map(rect => rect.x + rect.width)) - x
  const height = Math.max(...rects.map(rect => rect.y + rect.height)) - y
  let offset = { x: step, y: step }
  if (viewport) {
    const padding = 16 / viewport.zoom
    const visible = x + width > viewport.x && x < viewport.x + viewport.width && y + height > viewport.y && y < viewport.y + viewport.height
    const place = (start: number, size: number, lower: number, extent: number) => Math.max(lower + padding,
      Math.min(visible ? start + step : lower + (extent - size) / 2, lower + extent - size - padding))
    offset = { x: place(x, width, viewport.x, viewport.width) - x, y: place(y, height, viewport.y, viewport.height) - y }
  }
  if (parent) {
    const bounds = sceneContentBounds(parent.scene, parent.group.id)
    offset = {
      x: Math.max(bounds.x, Math.min(x + offset.x, bounds.x + bounds.width - width)) - x,
      y: Math.max(bounds.y, Math.min(y + offset.y, bounds.y + bounds.height - height)) - y,
    }
  }
  return offset
}

function internalEdgesForNodeIds(graph: GraphModel, nodeIds: Set<string>): GraphEdge[] {
  return graph.edges.filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target))
}

function boundaryAndInternalEdgesForNodeIds(graph: GraphModel, nodeIds: Set<string>): GraphEdge[] {
  return graph.edges.filter((edge) => nodeIds.has(edge.source) || nodeIds.has(edge.target))
}

function cloneClipboardFragment(fragment: GraphClipboardFragment): GraphClipboardFragment {
  const clonedGraph = cloneGraph({
    learningRate: 0,
    nodes: fragment.nodes,
    edges: fragment.edges,
    groups: fragment.group ? [fragment.group, ...(fragment.groups ?? [])] : undefined,
  })

  return {
    nodes: clonedGraph.nodes,
    edges: clonedGraph.edges,
    group: clonedGraph.groups?.[0],
    groups: clonedGraph.groups?.slice(1),
    parentGroupId: fragment.parentGroupId,
    layout: fragment.layout ? structuredClone(fragment.layout) : undefined,
  }
}

function translatePosition(position: Position, offset: Position): Position {
  return {
    x: position.x + offset.x,
    y: position.y + offset.y,
  }
}

function createPastedGroup(
  group: GraphGroup,
  usedGroupIds: Set<string>,
  nodeIdMap: Map<string, string>,
  offset: Position,
): GraphGroup {
  const { id, index } = nextIndexedId('group', usedGroupIds)

  return {
    ...group,
    id,
    label: /^Group \d+$/.test(group.label) ? `Group ${index}` : `${group.label} copy`,
    nodeIds: group.nodeIds.map((nodeId) => nodeIdMap.get(nodeId)).filter((nodeId): nodeId is string => Boolean(nodeId)),
    position: translatePosition(group.position, offset),
    dimensions: { ...group.dimensions },
    detail: group.detail ? remapDetail(group.detail, nodeIdMap) : undefined,
  }
}

function remapDetail(detail: Record<string, unknown>, nodeIdMap: Map<string, string>): Record<string, unknown> {
  const remap = (value: unknown): unknown => {
    if (typeof value === 'string') return nodeIdMap.get(value) ?? value
    if (Array.isArray(value)) return value.map(remap)
    if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, child]) => [key, remap(child)]))
    return value
  }
  return remap(detail) as Record<string, unknown>
}

function nextNodeId(type: NodeType, usedIds: Set<string>): { id: string; index: number } {
  return nextIndexedId(idPrefixForType(type), usedIds)
}

function labelForPastedNode(node: GraphNode, index: number): string {
  const labelPrefix = labelPrefixForType(node.type)
  return labelPrefix ? `${labelPrefix}${index}` : node.label
}

function idPrefixForType(type: NodeType): string {
  if (type === 'input') return 'input'
  if (type === 'weight') return 'weight'
  if (type === 'bias') return 'bias'
  if (type === 'target') return 'target'
  if (type === 'activation') return 'activation'
  return type
}

function labelPrefixForType(type: NodeType): string | undefined {
  if (type === 'input') return 'x'
  if (type === 'weight') return 'w'
  if (type === 'bias') return 'b'
  if (type === 'target') return 'y'
  return undefined
}

function nextIndexedId(prefix: string, usedIds: Set<string>): { id: string; index: number } {
  const matcher = new RegExp(`^${escapeRegExp(prefix)}-(\\d+)$`)
  const indexes = Array.from(usedIds)
    .map((id) => id.match(matcher)?.[1])
    .filter((value): value is string => Boolean(value))
    .map(Number)

  let index = Math.max(0, ...indexes) + 1
  let id = `${prefix}-${index}`
  while (usedIds.has(id)) {
    index += 1
    id = `${prefix}-${index}`
  }

  usedIds.add(id)
  return { id, index }
}

function nextCopyId(baseId: string, usedIds: Set<string>): string {
  const copyBase = `${baseId}-copy`
  if (!usedIds.has(copyBase)) {
    usedIds.add(copyBase)
    return copyBase
  }

  let index = 2
  while (usedIds.has(`${copyBase}-${index}`)) {
    index += 1
  }

  const id = `${copyBase}-${index}`
  usedIds.add(id)
  return id
}

function unique(values: string[]): string[] {
  return Array.from(new Set(values))
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}
