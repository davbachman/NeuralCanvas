import { inputArityForNode, outputArityForNode } from './engine'
import { showsTraceMetrics, builderCardHeight, builderCardWidth, builderInputPortY, builderOutputPortY } from './builderGeometry'
import { visualGroupInterface } from './grouping'
import { layoutSemanticGraph, type SemanticLayout, type SemanticRect } from './semanticLayout'
import type { GraphGroup, GraphModel, GraphViewState, Position } from './types'
import type { WireEndpoint } from './wireRouting'

export const sceneGroupId = (id: string) => `visual-group:${id}`
export const CONTAINMENT_SCALE = .25
export interface SceneLevel { parentId?: string; ids: string[]; scale: number; x: number; y: number; bounds: SemanticRect }
export interface ContinuousScene extends SemanticLayout {
  scales: Map<string, number>
  parents: Map<string, string | undefined>
  levels: SceneLevel[]
}
export interface SceneWire { id: string; edgeId: string; parentId?: string; scale: number; route: Position[]; sourceSide: WireEndpoint['side']; targetSide: WireEndpoint['side']; hidden?: boolean }

/** A container with exactly one child and no calculations of its own adds no
 * visual detail. Keep the most specific block and skip those empty levels.
 * This is a display projection: the editable model and its groups stay intact. */
export function compactVisualHierarchy(graph: GraphModel): GraphModel {
  const groups = graph.groups ?? []
  const replacements = new Map<string, string>()
  const byId = new Map(groups.map(group => [group.id, group]))
  for (const group of groups) {
    if (graph.view?.preservedLayouts?.[group.id]) continue
    const children = groups.filter(child => child.parentId === group.id)
    const child = children[0]
    if (children.length === 1 && child.nodeIds.length > 0 && child.nodeIds.length === group.nodeIds.length && child.nodeIds.every(id => group.nodeIds.includes(id))) {
      replacements.set(group.id, child.id)
    }
  }
  if (!replacements.size) return graph
  function resolve(id?: string): string | undefined {
    const seen = new Set<string>()
    while (id && replacements.has(id) && !seen.has(id)) {
      seen.add(id)
      id = replacements.get(id)
    }
    return id
  }
  const offsets = { ...graph.view?.layoutOffsets }
  const visible = groups.filter(group => !replacements.has(group.id)).map(group => {
    let parentId = group.parentId
    const seen = new Set<string>()
    while (parentId && replacements.has(parentId) && !seen.has(parentId)) {
      seen.add(parentId)
      const offset = graph.view?.layoutOffsets?.[sceneGroupId(parentId)]
      if (offset) {
        const current = offsets[sceneGroupId(group.id)] ?? { x: 0, y: 0 }
        offsets[sceneGroupId(group.id)] = { x: current.x + offset.x, y: current.y + offset.y }
      }
      parentId = byId.get(parentId)?.parentId
    }
    return { ...group, parentId }
  })
  return { ...graph, groups: visible, view: graph.view ? { ...graph.view,
    focusedGroupId: resolve(graph.view.focusedGroupId),
    expandedGroupIds: [...new Set(graph.view.expandedGroupIds.map(id => resolve(id)!))],
    layoutOffsets: Object.keys(offsets).length ? offsets : undefined,
  } : undefined }
}

/** Stop at a readable close-up of the smallest operation, rather than letting
 * the camera magnify empty space tens of thousands of times past the graph. */
export function continuousSceneMaxZoom(scene: ContinuousScene): number {
  const scales = [...scene.nodes.keys()].map(id => scene.scales.get(id)!).filter(scale => scale > 0)
  return 2.5 / Math.min(1, ...scales)
}

/** Each card reserves a permanent place for its contents. Camera movement never
 * changes this geometry: a child is simply a smaller drawing inside its parent. */
export function layoutContinuousScene(graph: GraphModel): ContinuousScene {
  const scene: ContinuousScene = { nodes: new Map(), groups: new Map(), scales: new Map(), parents: new Map(), levels: [] }
  const groups = graph.groups ?? []
  const groupById = new Map(groups.map(group => [group.id, group]))
  const ownerByNode = new Map<string, { id: string; depth: number }>()
  for (const group of groups) {
    let depth = 0, parentId = group.parentId
    const seen = new Set([group.id])
    while (parentId && !seen.has(parentId)) {
      seen.add(parentId)
      depth++
      parentId = groupById.get(parentId)?.parentId
    }
    for (const nodeId of group.nodeIds) {
      if (depth >= (ownerByNode.get(nodeId)?.depth ?? -1)) ownerByNode.set(nodeId, { id: group.id, depth })
    }
  }
  const manualPlacements = graph.view?.manualNodePlacements ?? {}
  const groupSizes = new Map<string, { width: number; height: number }>()
  const measure = (group: GraphGroup, seen = new Set<string>()) => {
    if (groupSizes.has(group.id) || seen.has(group.id)) return
    const children = groups.filter(child => child.parentId === group.id)
    for (const child of children) measure(child, new Set([...seen, group.id]))
    const local = layoutSemanticGraph({ ...graph, nodes: graph.nodes.filter(node => group.nodeIds.includes(node.id)),
      groups: children, view: { expandedGroupIds: [], layoutEdges: graph.view?.layoutEdges } }, true, group, groupSizes)
    const rects = [...local.nodes.values(), ...local.groups.values()]
    const width = rects.length ? Math.max(...rects.map(rect => rect.x + rect.width)) - Math.min(...rects.map(rect => rect.x)) : 0
    const height = rects.length ? Math.max(...rects.map(rect => rect.y + rect.height)) - Math.min(...rects.map(rect => rect.y)) : 0
    groupSizes.set(group.id, { width: Math.max(238, width * CONTAINMENT_SCALE + 40),
      height: Math.max(group.detail?.userCreated && group.kind === 'module' ? 136 : 172, height * CONTAINMENT_SCALE + 60) })
  }
  for (const group of groups) measure(group)
  const activeManualPlacement = (id: string) => {
    if (graph.view?.preservedLayouts?.[ownerByNode.get(id)?.id ?? '']?.[id]) return undefined
    const placement = manualPlacements[id]
    // Grouping changes ownership, but an older manual placement can still
    // point at the previous level. Let membership win so every selected block
    // actually appears inside its new group, including imported projects.
    return placement && placement.parentId === ownerByNode.get(id)?.id ? placement : undefined
  }
  function level(parent?: GraphGroup, ancestors = new Set<string>()) {
    if (parent && ancestors.has(parent.id)) return
    const children = groups.filter(group => group.parentId === parent?.id)
    const manualNodes = graph.nodes.filter(node => {
      const placement = activeManualPlacement(node.id)
      return placement && placement.parentId === parent?.id
    })
    const localGraph: GraphModel = { ...graph,
      nodes: graph.nodes.filter(node => (!parent || parent.nodeIds.includes(node.id)) && !activeManualPlacement(node.id)),
      groups: children.map(group => ({ ...group, parentId: parent?.id })),
      view: { expandedGroupIds: [], layoutEdges: graph.view?.layoutEdges },
    }
    const local = layoutSemanticGraph(localGraph, true, parent, groupSizes)
    const preserved = graph.view?.preservedLayouts?.[parent?.id ?? '']
    if (preserved) {
      for (const [id, rect] of local.nodes) {
        const saved = preserved[id]
        // Preserve placement; containment determines scale. Port changes can
        // grow the card, and wire endpoints must use its current dimensions.
        if (saved) local.nodes.set(id, { ...saved, width: rect.width, height: rect.height })
      }
      for (const [id, rect] of local.groups) {
        const saved = preserved[sceneGroupId(id)]
        local.groups.set(id, saved ? { ...saved, width: saved.width / saved.scale, height: saved.height / saved.scale } : rect)
      }
    }
    const rects = [...local.nodes.values(), ...local.groups.values()]
    if (!rects.length && !manualNodes.length) return
    const savedBounds = graph.view?.preservedLayoutBounds?.[parent?.id ?? '']
    const left = savedBounds?.x ?? (rects.length ? Math.min(...rects.map(rect => rect.x)) : 0)
    const top = savedBounds?.y ?? (rects.length ? Math.min(...rects.map(rect => rect.y)) : 0)
    const width = savedBounds?.width ?? (rects.length ? Math.max(...rects.map(rect => rect.x + rect.width)) - left : 176)
    const height = savedBounds?.height ?? (rects.length ? Math.max(...rects.map(rect => rect.y + rect.height)) - top : 112)
    let scale = 1, x = 0, y = 0
    if (parent) {
      const frame = scene.groups.get(parent.id)!, outerScale = scene.scales.get(sceneGroupId(parent.id))!
      const transform = graph.view?.preservedGroupTransforms?.[parent.id]
      scale = outerScale * CONTAINMENT_SCALE
      if (transform) {
        x = frame.x + transform.x * outerScale
        y = frame.y + transform.y * outerScale
      } else {
        const availableHeight = frame.height - 60 * outerScale
        x = frame.x + (frame.width - width * scale) / 2 - left * scale
        y = frame.y + 40 * outerScale + (availableHeight - height * scale) / 2 - top * scale
      }
    }
    const ids: string[] = []
    function place(id: string, rect: SemanticRect, group: boolean) {
      const key = group ? sceneGroupId(id) : id
      const offset = graph.view?.layoutOffsets?.[key]
      const placed = { x: x + rect.x * scale + (offset?.x ?? 0), y: y + rect.y * scale + (offset?.y ?? 0), width: rect.width * scale, height: rect.height * scale }
      if (group) scene.groups.set(id, placed)
      else scene.nodes.set(id, placed)
      scene.scales.set(key, scale)
      scene.parents.set(key, parent?.id)
      ids.push(key)
    }
    for (const [id, rect] of local.groups) place(id, rect, true)
    for (const [id, rect] of local.nodes) place(id, rect, false)
    for (const node of manualNodes) {
      const placement = activeManualPlacement(node.id)!
      const frame = placement.parentId ? scene.groups.get(placement.parentId) : undefined
      const offset = graph.view?.layoutOffsets?.[node.id]
      const nodeScale = scale
      scene.nodes.set(node.id, {
        x: (frame?.x ?? 0) + placement.offset.x + (offset?.x ?? 0),
        y: (frame?.y ?? 0) + placement.offset.y + (offset?.y ?? 0),
        width: builderCardWidth(node) * nodeScale,
        height: builderCardHeight(node, showsTraceMetrics(graph)) * nodeScale,
      })
      scene.scales.set(node.id, nodeScale)
      scene.parents.set(node.id, parent?.id)
      ids.push(node.id)
    }
    scene.levels.push({ parentId: parent?.id, ids, scale, x, y, bounds: { x: left, y: top, width, height } })
    for (const child of children) level(child, new Set([...ancestors, ...(parent ? [parent.id] : [])]))
  }
  level()
  growSceneContainers(scene)
  return scene
}

/** Expand outwards from the contents, deepest frames first. Children keep
 * their positions and scales; only the containment windows change. */
function growSceneContainers(scene: ContinuousScene) {
  for (const level of [...scene.levels].reverse()) {
    if (!level.parentId) continue
    const frame = scene.groups.get(level.parentId)!, scale = scene.scales.get(sceneGroupId(level.parentId))!
    let left = frame.x, top = frame.y, right = frame.x + frame.width, bottom = frame.y + frame.height
    for (const id of level.ids) {
      const rect = id.startsWith('visual-group:') ? scene.groups.get(id.slice(13))! : scene.nodes.get(id)!
      left = Math.min(left, rect.x - 16 * scale)
      top = Math.min(top, rect.y - 40 * scale)
      right = Math.max(right, rect.x + rect.width + 16 * scale)
      bottom = Math.max(bottom, rect.y + rect.height + 16 * scale)
    }
    scene.groups.set(level.parentId, { x: left, y: top, width: right - left, height: bottom - top })
  }
}

/** Live drag geometry uses the same containment rules as saved layouts. */
export function sceneAtPositions(scene: ContinuousScene, positions: Map<string, Position>): ContinuousScene {
  const current = { ...scene,
    nodes: new Map([...scene.nodes].map(([id, rect]) => [id, { ...rect, ...positions.get(id) }])),
    groups: new Map([...scene.groups].map(([id, rect]) => [id, { ...rect, ...positions.get(sceneGroupId(id)) }])),
  }
  growSceneContainers(current)
  return current
}

export function captureGroupTransforms(scene: ContinuousScene): NonNullable<GraphViewState['preservedGroupTransforms']> {
  return Object.fromEntries(scene.levels.flatMap(level => {
    if (!level.parentId) return []
    const frame = scene.groups.get(level.parentId)!, scale = scene.scales.get(sceneGroupId(level.parentId))!
    return [[level.parentId, { x: (level.x - frame.x) / scale, y: (level.y - frame.y) / scale, scale: level.scale / scale }]]
  }))
}

/** Reserve room for incoming/outgoing wires and the frame heading. */
export function sceneContentBounds(scene: ContinuousScene, groupId: string): SemanticRect {
  const frame = scene.groups.get(groupId)!, scale = scene.scales.get(sceneGroupId(groupId))!
  return { x: frame.x + 16 * scale, y: frame.y + 40 * scale, width: frame.width - 32 * scale, height: frame.height - 56 * scale }
}

const smooth = (t: number) => { const value = Math.max(0, Math.min(1, t)); return value * value * (3 - 2 * value) }
/** Use viewport occupancy, not discrete zoom thresholds. Wide and tall cards
 * both reveal before the user has to scroll past their boundary. */
export function cardReveal(rect: SemanticRect, zoom: number, width: number, height: number): number {
  const occupancy = Math.max(rect.width * zoom / Math.max(1, width), rect.height * zoom / Math.max(1, height))
  return smooth((occupancy - .38) / .48)
}

/** Join exact ports at each level without changing the curve style when a
 * project contains groups. Boundary ports share the same world coordinate. */
export function routeContinuousScene(graph: GraphModel, scene: ContinuousScene, positions?: Map<string, Position>): SceneWire[] {
  const groups = new Map((graph.groups ?? []).map(group => [group.id, group]))
  const nodes = new Map(graph.nodes.map(node => [node.id, node]))
  const interfaces = new Map((graph.groups ?? []).map(group => [group.id, visualGroupInterface(graph, group)]))
  const result: SceneWire[] = []
  const rectFor = (id: string) => {
    const rect = id.startsWith('visual-group:') ? scene.groups.get(id.slice(13))! : scene.nodes.get(id)!
    return { ...rect, ...positions?.get(id) }
  }
  function endpoint(id: string, edge: GraphModel['edges'][number], output: boolean): WireEndpoint {
    const rect = rectFor(id), group = id.startsWith('visual-group:') ? groups.get(id.slice(13)) : undefined
    const ports = group ? interfaces.get(group.id)![output ? 'outputs' : 'inputs'] : undefined
    const index = ports ? ports.findIndex(port => port.edgeId === edge.id || 'edgeIds' in port && port.edgeIds?.includes(edge.id)) : output ? edge.sourceSlot ?? 0 : edge.inputSlot ?? 0
    const count = ports?.length ?? (output ? outputArityForNode(nodes.get(id)!) : Math.max(inputArityForNode(nodes.get(id)!), ...graph.edges.filter(item => item.target === id).map(item => (item.inputSlot ?? 0) + 1)))
    const node = nodes.get(id)
    const fraction = node
      ? (output ? builderOutputPortY(node, Math.max(0, index)) : builderInputPortY(node, Math.max(0, index))) / builderCardHeight(node, showsTraceMetrics(graph))
      : (Math.max(0, index) + 1) / (Math.max(1, count) + 1)
    return { x: rect.x + (output ? rect.width : 0), y: rect.y + rect.height * fraction, side: output ? 'right' : 'left' }
  }
  for (const level of scene.levels) {
    const parent = level.parentId ? groups.get(level.parentId) : undefined
    const owner = new Map<string, string>()
    for (const id of level.ids) {
      const group = id.startsWith('visual-group:') ? groups.get(id.slice(13)) : undefined
      for (const nodeId of group?.nodeIds ?? [id]) owner.set(nodeId, id)
    }
    const boundary = parent ? sceneGroupId(parent.id) : undefined
    const reverse = { left: 'right', right: 'left', top: 'bottom', bottom: 'top' } as const
    const sharedInputs = new Map<string, string>()
    const hiddenAliases = new Map<string, string>()
    const wires: { id: string; source: WireEndpoint; target: WireEndpoint }[] = graph.edges.flatMap(edge => {
      const source = owner.get(edge.source), target = owner.get(edge.target)
      if ((!source && !target) || source === target || (!boundary && (!source || !target))) return []
      if (target?.startsWith('visual-group:')) {
        const input = interfaces.get(target.slice(13))?.inputs.find(port => port.edgeId === edge.id || port.edgeIds?.includes(edge.id))
        if (input) {
          const key = `${target}:${input.handleId}`
          const representative = sharedInputs.get(key)
          if (representative) {
            hiddenAliases.set(edge.id, representative)
            return []
          }
          sharedInputs.set(key, edge.id)
        }
      }
      const from = endpoint(source ?? boundary!, edge, Boolean(source)), to = endpoint(target ?? boundary!, edge, !target)
      return [{ id: edge.id, source: { ...from, side: source ? from.side : reverse[from.side] }, target: { ...to, side: target ? to.side : reverse[to.side] } }]
    })
    const levelWires = new Map<string, SceneWire>()
    for (const { id: edgeId, source, target } of wires) {
      const wire: SceneWire = { id: `${edgeId}::${parent?.id ?? 'model'}`, edgeId, parentId: parent?.id, scale: level.scale,
        route: [{ x: source.x, y: source.y }, { x: target.x, y: target.y }], sourceSide: source.side, targetSide: target.side }
      levelWires.set(edgeId, wire)
      result.push(wire)
    }
    // Preserve each edge's continuous path for topology and inspection while
    // drawing only one exterior wire for the shared signal.
    for (const [edgeId, representative] of hiddenAliases) {
      const route = levelWires.get(representative)
      if (route) result.push({ ...route, id: `${edgeId}::${parent?.id ?? 'model'}`, edgeId, hidden: true })
    }
  }
  return result
}
