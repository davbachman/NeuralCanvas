import { describe, expect, it } from 'vitest'
import { encodeStory, textDataset } from './textData'
import { datasetOutputValueForSlot } from './datasets'
import { backwardPass, forwardPass, parameterValues, runTrainingStep, runTrainingStepFast, updateParameters } from './engine'
import { evaluateDataset, trainDataset, withDatasetExample } from './datasetTraining'
import { storyExperimentData, storyExperimentModel } from '../test/storyExperimentModel'

const edited = {facts: ['John is in garden.'], question: 'Where is Mary?'}

describe('structured story experiments', () => {
  it('shares canonical preprocessing and reconstructs positions and masks with frozen vocabulary', () => {
    const data = storyExperimentData(), before = JSON.stringify(data)
    const original = data.documents[0]
    expect(encodeStory({facts: original.facts!, question: original.question!}, data).features).toEqual(textDataset(data).examples![0].features)
    const {features, unknownWords} = encodeStory({facts: ['MARY is in Atlantis.'], question: 'Where is Mary?'}, data)
    expect(features.map(t => t.shape)).toEqual([[3,6],[1,6],[3],[1,3],[3,6,1],[1,6,1]])
    expect(features[0].data.slice(0,6)).toEqual([data.vocabulary.indexOf('mary'),data.vocabulary.indexOf('is'),data.vocabulary.indexOf('in'),0,data.vocabulary.indexOf('.'),1])
    expect(features[0].data.slice(6)).toEqual(Array(12).fill(1))
    expect(features[2].data).toEqual([0,1,2])
    expect(features[3].data).toEqual([1,0,0])
    expect(features[4].data).toEqual([1,1,1,1,1,0,...Array(12).fill(0)])
    expect(features[5].data).toEqual([1,1,1,1,0,0])
    expect(unknownWords).toEqual(['atlantis'])
    expect(JSON.stringify(data)).toBe(before)
  })

  it('rejects too many facts, empty fields and overlong facts/questions without truncating', () => {
    const data = storyExperimentData()
    for (const story of [
      {facts: Array(4).fill('Mary'), question: 'Where?'},
      {facts: [], question: 'Where?'}, {facts: [''], question: 'Where?'},
      {facts: ['Mary'], question: ''},
      {facts: ['Mary '.repeat(7)], question: 'Where?'},
      {facts: ['Mary'], question: 'Where '.repeat(7)},
    ]) expect(() => encodeStory(story, data)).toThrow()
  })

  it.each([
    ['question', {facts: ['Mary is in kitchen.', 'John is in garden.'], question: 'Where is John?'}],
    ['irrelevant fact', {facts: ['Mary is in kitchen.', 'John is in kitchen.'], question: 'Where is Mary?'}],
    ['answer-determining fact', {facts: ['Mary is in garden.', 'John is in garden.'], question: 'Where is Mary?'}],
  ] as const)('uses a changed %s in predictions and attention', (_, story) => {
    const graph = storyExperimentModel()
    const before = forwardPass(graph).graph
    const after = forwardPass({...graph, nodes: graph.nodes.map(n => n.type === 'dataset' ? {...n, params: {...n.params, storyExperiment: {...story, facts: [...story.facts]}}} : n)}).graph
    for (const id of ['output', 'attention']) expect(after.nodes.find(n => n.id === id)!.value).not.toEqual(before.nodes.find(n => n.id === id)!.value)
    expect(datasetOutputValueForSlot(after.nodes[0], 6)).toEqual(datasetOutputValueForSlot(before.nodes[0], 6))
    expect(parameterValues(after)).toEqual(parameterValues(before))
  })

  it('evaluates edited inputs, probabilities and attention without modifying canonical data or parameters', async () => {
    const graph = storyExperimentModel(), original = JSON.stringify(graph)
    const experimental = {...graph, nodes: graph.nodes.map(n => n.type === 'dataset' ? {...n, params: {...n.params, storyExperiment: edited}} : n)}
    const source = experimental.nodes[0]
    const features = encodeStory(edited, source.params.textData!).features
    features.forEach((feature, slot) => expect(datasetOutputValueForSlot(source, slot)).toEqual(feature))
    const explicit = {...graph, nodes: graph.nodes.map(n => n.id === source.id ? {...n, params: {...n.params, datasetValues: [...features, textDataset(n.params.textData!).examples![0].target]}} : n)}
    const actual = forwardPass(experimental), expected = forwardPass(explicit)
    for (const id of ['output', 'attention', 'fact_lookup', 'question_lookup']) expect(actual.graph.nodes.find(n => n.id === id)!.value).toEqual(expected.graph.nodes.find(n => n.id === id)!.value)
    expect(actual.graph.nodes.find(n => n.id === 'attention')!.value!.data).toEqual([1,0,0])
    expect(parameterValues(actual.graph)).toEqual(parameterValues(graph))
    expect(evaluateDataset(experimental, source.id, 'test')).toEqual(evaluateDataset(graph, source.id, 'test'))
    expect(() => runTrainingStep(experimental)).toThrow(/Restore the original/)
    expect(() => runTrainingStepFast(experimental)).toThrow(/Restore the original/)
    expect(() => updateParameters(backwardPass(actual.graph).graph)).toThrow(/Restore the original/)
    expect(withDatasetExample(experimental, source.id, 1).nodes[0].params.storyExperiment).toBeUndefined()
    const trained = await trainDataset(experimental, source.id, 1, {shuffleEachEpoch: false})
    const baseline = await trainDataset(graph, source.id, 1, {shuffleEachEpoch: false})
    expect(parameterValues(trained)).toEqual(parameterValues(baseline))
    expect(trained.nodes[0].params.storyExperiment).toEqual(edited)
    expect(JSON.stringify(graph)).toBe(original)
  })
})
