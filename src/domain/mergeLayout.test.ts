import { describe, expect, it } from 'vitest'
import { mergePreservingLayout, ungroupPreservingLayout } from './mergeLayout'
import { createNode, createStarterGraph } from './examples'
import { placeCanvasNode } from './nodePlacement'
import { layoutContinuousScene, sceneGroupId } from './continuousScene'
import { builderCardWidth, builderCardHeight } from './builderGeometry'
import { cloneGraph, forwardPass } from './engine'
import type { GraphModel } from './types'
import type { SemanticLayout } from './semanticLayout'

const relative = (rect: {x:number;y:number;width:number;height:number}, anchor: {x:number;y:number;width:number}) => ({
  x: (rect.x-anchor.x)/anchor.width, y:(rect.y-anchor.y)/anchor.width,
  width: rect.width/anchor.width, height:rect.height/anchor.width,
})
function sameShape(a: ReturnType<typeof relative>, b: ReturnType<typeof relative>) {
  for (const key of ['x','y','width','height'] as const) expect(a[key]).toBeCloseTo(b[key], 8)
}

describe('merge layout preservation', () => {
  it('shrinks a deliberately hand-arranged selection about its center without moving other blocks', () => {
    const graph = createStarterGraph()
    graph.nodes.find(n => n.id === 'mul')!.position = {x:700,y:100}
    graph.nodes.find(n => n.id === 'add')!.position = {x:380,y:440}
    const ids = ['w','mul','add']
    const rects = new Map(graph.nodes.map(n => [n.id, {...n.position,width:builderCardWidth(n),height:builderCardHeight(n)}]))
    const original = structuredClone(graph)
    const merged = mergePreservingLayout(graph, ids)
    const scene = layoutContinuousScene(merged.graph)
    for (const id of ids) sameShape(relative(scene.nodes.get(id)!,scene.nodes.get('w')!),relative(rects.get(id)!,rects.get('w')!))
    for (const n of graph.nodes.filter(n => !ids.includes(n.id))) expect(scene.nodes.get(n.id)).toMatchObject(rects.get(n.id)!)
    const selected = ids.map(id => rects.get(id)!)
    const frame = scene.groups.get(merged.group!.id)!
    expect(frame.x+frame.width/2).toBeCloseTo((Math.min(...selected.map(r=>r.x))+Math.max(...selected.map(r=>r.x+r.width)))/2)
    expect(frame.y+frame.height/2).toBeCloseTo((Math.min(...selected.map(r=>r.y))+Math.max(...selected.map(r=>r.y+r.height)))/2)
    expect(forwardPass(merged.graph).graph.nodes.find(n=>n.id==='loss')?.value).toEqual(forwardPass(graph).graph.nodes.find(n=>n.id==='loss')?.value)
    expect(graph).toEqual(original)
  })

  it('preserves a nested selection including dragged offsets and clones saved layouts independently', () => {
    const graph = mergePreservingLayout(createStarterGraph(), ['x','w','mul','b','add','pred']).graph
    graph.view!.layoutOffsets = {mul:{x:2,y:1}}
    const before = layoutContinuousScene(graph)
    const merged = mergePreservingLayout(graph,['mul','add'])
    const after = layoutContinuousScene(merged.graph)
    sameShape(relative(after.nodes.get('add')!,after.nodes.get('mul')!),relative(before.nodes.get('add')!,before.nodes.get('mul')!))
    expect(after.parents.get('mul')).toBe(merged.group!.id)
    for (const key of ['x','y','width','height'] as const) expect(after.nodes.get('x')![key]).toBeCloseTo(before.nodes.get('x')![key], 8)
    const ungrouped = layoutContinuousScene(ungroupPreservingLayout(merged.graph, merged.group!.id))
    for (const id of ['mul','add']) for (const key of ['x','y','width','height'] as const) expect(ungrouped.nodes.get(id)![key]).toBeCloseTo(before.nodes.get(id)![key], 8)
    const cloned = cloneGraph(merged.graph)
    cloned.view!.preservedLayouts![''][sceneGroupId('group-1')].x += 99
    expect(cloned.view!.preservedLayouts).not.toEqual(merged.graph.view!.preservedLayouts)
    const reopened = JSON.parse(JSON.stringify(merged.graph))
    expect(layoutContinuousScene(reopened).nodes).toEqual(after.nodes)
  })
})

const geometry = (graph: GraphModel): SemanticLayout => graph.groups?.length || graph.view?.preservedLayouts
  ? layoutContinuousScene(graph)
  : { nodes: new Map(graph.nodes.map(node => [node.id, { ...node.position, width: builderCardWidth(node), height: builderCardHeight(node) }])), groups: new Map() }

it.each(['root', 'other group', 'nested', 'whole module'])('restores block sizes through repeated merge/edit/unmerge cycles at %s', context => {
  let graph = createStarterGraph()
  if (context === 'other group') graph = mergePreservingLayout(graph, ['x', 'w']).graph
  if (context === 'nested') graph = mergePreservingLayout(graph, ['x', 'w', 'mul', 'b', 'add', 'pred']).graph
  if (context === 'whole module') graph = mergePreservingLayout(graph, ['mul', 'add']).graph
  const initial = geometry(graph)
  let mergedSize: { width: number; height: number } | undefined
  for (let cycle = 0; cycle < 8; cycle++) {
    const before = structuredClone(graph)
    const merged = mergePreservingLayout(graph, context === 'whole module' ? ['x', 'w', 'mul', 'b', 'add', 'pred'] : ['mul', 'add'])
    const frame = geometry(merged.graph).groups.get(merged.group!.id)!
    mergedSize ??= frame
    expect(frame.width).toBeCloseTo(mergedSize.width, 8)
    expect(frame.height).toBeCloseTo(mergedSize.height, 8)
    expect(graph).toEqual(before)
    graph = ungroupPreservingLayout(merged.graph, merged.group!.id)
    const after = geometry(graph)
    for (const kind of ['nodes', 'groups'] as const) for (const [id, rect] of initial[kind]) {
      for (const key of ['x', 'y', 'width', 'height'] as const) expect(after[kind].get(id)![key]).toBeCloseTo(rect[key], 8)
    }
    expect(graph.nodes).toEqual(before.nodes)
    expect(graph.edges).toEqual(before.edges)
    expect(graph.view?.preservedLayouts?.[merged.group!.id]).toBeUndefined()
    expect(graph.view?.preservedLayoutBounds?.[merged.group!.id]).toBeUndefined()
    graph.nodes.find(node => node.id === 'mul')!.params.outputName = `edited_${cycle}`
    graph = JSON.parse(JSON.stringify(graph))
  }
})

it('keeps moves and added blocks made between unmerge and remerge after the last group is removed', () => {
  const merged = mergePreservingLayout(createStarterGraph(), ['mul', 'add'])
  let graph = ungroupPreservingLayout(merged.graph, merged.group!.id)
  graph.view!.layoutOffsets = { mul: { x: 125, y: -55 } }
  const added = { ...createNode('weight', 99), position: { x: 750, y: 250 } }
  graph = placeCanvasNode(graph, added)
  const before = geometry(graph)
  const regrouped = mergePreservingLayout(graph, ['mul', 'add', added.id])
  const ungrouped = ungroupPreservingLayout(regrouped.graph, regrouped.group!.id)
  const after = geometry(ungrouped)
  for (const [id, rect] of before.nodes) {
    for (const key of ['x', 'y', 'width', 'height'] as const) expect(after.nodes.get(id)![key]).toBeCloseTo(rect[key], 8)
  }
  expect(ungrouped.nodes).toEqual(graph.nodes)
  expect(ungrouped.edges).toEqual(graph.edges)
})

it('expands edited contents around a moved group without applying drag offsets twice', () => {
  const original = createStarterGraph(), before = geometry(original)
  const merged = mergePreservingLayout(original, ['mul', 'add'])
  const scale = layoutContinuousScene(merged.graph).levels.find(level => level.parentId === merged.group!.id)!.scale
  merged.graph.view!.layoutOffsets = {
    [sceneGroupId(merged.group!.id)]: { x: 90, y: -40 }, mul: { x: 3, y: 4 }, b: { x: 7, y: 0 },
  }
  const next = ungroupPreservingLayout(merged.graph, merged.group!.id), after = geometry(next)
  for (const id of ['mul', 'add']) {
    const rect = after.nodes.get(id)!, previous = before.nodes.get(id)!
    expect(rect.width).toBeCloseTo(previous.width, 8)
    expect(rect.height).toBeCloseTo(previous.height, 8)
    expect(rect.x).toBeCloseTo(previous.x + 90 + (id === 'mul' ? 3 / scale : 0), 8)
    expect(rect.y).toBeCloseTo(previous.y - 40 + (id === 'mul' ? 4 / scale : 0), 8)
  }
  expect(after.nodes.get('b')!.x).toBeCloseTo(before.nodes.get('b')!.x + 7, 8)
  expect(forwardPass(next).loss).toEqual(forwardPass(original).loss)
})
