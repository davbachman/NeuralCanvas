import { afterEach, beforeAll, describe, expect, it } from 'vitest'
import { storyQuestionData, storyQuestionDocuments, storyQuestionModel } from '../test/storyQuestionModel'
import { encodeStory, isTextDatasetData, prepareTextDocuments, textDataset } from './textData'
import { datasetExamplesForNode, datasetOutputCountForNode, datasetOutputLabelForSlot, datasetOutputValueForSlot, datasetTargetSlotForNode } from './datasets'
import { backwardPass, cloneGraph, forwardPass, parameterValues, runTrainingStep, runTrainingStepFast, updateParameters } from './engine'
import { evaluateDataset, trainDataset, withDatasetExample } from './datasetTraining'
import { createProjectStateFile, parseProjectStateFile } from './session'
import { TensorGraph, tf, trainTensorGraph } from './tensorTraining'
import { DEFAULT_TRAINING } from './trainingSettings'
import { tensorAxisLabels } from './tensorLabels'
import { InferenceInspection } from './inferenceInspection'

beforeAll(async () => { await tf.setBackend('cpu'); await tf.ready() })
afterEach(() => expect(tf.memory().numTensors).toBe(0))

describe('whole story and separate question', () => {
  it.each([[32, 8], [3, 5], [7, 4]])('uses declared lengths %i/%i and exact token IDs, masks, target, and output order', (maxLength, questionLength) => {
    const data = storyQuestionData(maxLength, questionLength), before = structuredClone(data)
    const encoded = encodeStory({text: 'MARY Atlantis', question: 'where John ?'}, data)
    expect(encoded.features).toEqual([
      {shape: [maxLength], data: [2, 0, ...Array(maxLength - 2).fill(1)]},
      {shape: [questionLength], data: [6, 3, 7, ...Array(questionLength - 3).fill(1)]},
      {shape: [maxLength, 1], data: [1, 1, ...Array(maxLength - 2).fill(0)]},
      {shape: [questionLength, 1], data: [1, 1, 1, ...Array(questionLength - 3).fill(0)]},
    ])
    expect(encoded.unknownWords).toEqual(['atlantis'])
    const source = storyQuestionModel(data).nodes[0]
    expect(datasetOutputCountForNode(source)).toBe(5)
    expect(datasetTargetSlotForNode(source)).toBe(4)
    expect(Array.from({length: 5}, (_, slot) => datasetOutputLabelForSlot(source, slot))).toEqual(['story token IDs', 'question token IDs', 'story word mask', 'question word mask', 'answer'])
    expect(datasetOutputValueForSlot(source, 4)).toEqual({shape: [1], data: [0]})
    expect(datasetExamplesForNode(source).map(row => row.split)).toEqual(['train', 'train', 'test', 'test'])
    expect(data).toEqual(before)
  })

  it('fits vocabulary on training stories AND questions only and fits classes on training answers', () => {
    const documents = storyQuestionDocuments.map((doc, i) => i === 2 ? {...doc, text: 'heldoutstory Mary', question: 'heldoutquestion ?'} : doc)
    const data = prepareTextDocuments(documents, 'fit.json', {task: 'classification', representation: 'story-question', maxLength: 4, questionLength: 3})
    expect(data.vocabulary.slice(0, 2)).toEqual(['<unk>', '<pad>'])
    expect(data.vocabulary).toContain('where')
    expect(data.vocabulary).toContain('?')
    expect(data.vocabulary).not.toContain('heldoutstory')
    expect(data.vocabulary).not.toContain('heldoutquestion')
    expect(data.classLabels).toEqual(['garden', 'kitchen'])
    expect(data.documents).toEqual(documents)
    expect(textDataset(data).examples![2].features[0].data[0]).toBe(0)
    expect(textDataset(data).examples![2].features[1].data[0]).toBe(0)
  })

  it('enforces story and question limits independently with no truncation', () => {
    const data = storyQuestionData(2, 5)
    expect(encodeStory({text: 'Mary garden', question: 'where where where where ?'}, data).features[1].shape).toEqual([5])
    expect(() => encodeStory({text: 'Mary kitchen garden', question: '?'}, data)).toThrow(/Story has 3 tokens; the limit is 2.*Nothing was truncated/)
    expect(() => encodeStory({text: 'Mary', question: 'where '.repeat(6)}, data)).toThrow(/Question has 6 tokens; the limit is 5.*Nothing was truncated/)
    expect(() => prepareTextDocuments(storyQuestionDocuments, 'missing.json', {task: 'classification', representation: 'story-question', maxLength: 32})).toThrow(/questionLength/)
    expect(() => prepareTextDocuments(storyQuestionDocuments, 'wrong-task.json', {task: 'language', representation: 'story-question', maxLength: 32, questionLength: 8})).toThrow(/classification/)
  })

  it.each([
    ['missing question length', {questionLength: undefined}], ['zero question length', {questionLength: 0}],
    ['fractional question length', {questionLength: 2.5}], ['large question length', {questionLength: 257}],
    ['string question length', {questionLength: '8'}], ['zero story length', {maxLength: 0}], ['large story length', {maxLength: 257}],
    ['sentiment', {task: 'sentiment'}], ['language', {task: 'language'}], ['missing pad', {vocabulary: ['<unk>', 'word']}],
    ['missing unknown', {vocabulary: ['<pad>', '<unk>']}], ['duplicate vocabulary', {vocabulary: ['<unk>', '<pad>', 'mary', 'mary']}],
    ['non-string vocabulary', {vocabulary: ['<unk>', '<pad>', 4]}], ['empty vocabulary token', {vocabulary: ['<unk>', '<pad>', '']}], ['missing classes', {classLabels: undefined}],
    ['one class', {classLabels: ['kitchen']}], ['duplicate classes', {classLabels: ['kitchen', 'kitchen']}], ['empty class', {classLabels: ['', 'garden']}], ['blank class', {classLabels: ['kitchen', 'garden', ' ']}],
  ])('rejects malformed prepared data: %s', (_, patch) => {
    expect(isTextDatasetData({...storyQuestionData(), ...patch})).toBe(false)
  })

  it.each([
    {question: undefined}, {question: ''}, {question: '  '}, {question: 12}, {question: 'where '.repeat(9)},
    {text: ''}, {text: 'Mary '.repeat(33)}, {label: 'unknown answer'}, {split: 'validation'},
  ])('rejects malformed story/question documents: %j', patch => {
    const data = storyQuestionData()
    expect(isTextDatasetData({...data, documents: data.documents.map((doc, i) => i === 2 ? {...doc, ...patch} : doc)})).toBe(false)
  })

  it('round-trips prepared data and temporary experiments in project saves without remapping IDs or classes', () => {
    const graph = storyQuestionModel(), data = graph.nodes[0].params.textData!
    graph.nodes[0].params.storyExperiment = {text: 'John garden', question: 'where Mary ?'}
    const file = createProjectStateFile({graph, visualizationGraph: graph, initialParameterValues: parameterValues(graph), selectedNodeIds: ['text-data'], phase: 'edit', traceSteps: [], traceIndex: 0, epoch: 0, currentLoss: null, display: {showMath: true, showGradient: false, showCode: false, showVisualization: false}})
    const restored = parseProjectStateFile(JSON.stringify(file))
    expect(isTextDatasetData(JSON.parse(JSON.stringify(data)))).toBe(true)
    expect(restored.ok).toBe(true)
    if (!restored.ok) throw Error(restored.error)
    expect(restored.file.state.graph.nodes[0].params.textData).toEqual(data)
    expect(restored.file.state.graph.nodes[0].params.storyExperiment).toEqual(graph.nodes[0].params.storyExperiment)
    expect(parameterValues(restored.file.state.graph)).toEqual(parameterValues(graph))
    expect(cloneGraph(graph).nodes[0].params.storyExperiment).not.toBe(graph.nodes[0].params.storyExperiment)
    const invalid = structuredClone(file)
    invalid.state.graph.nodes[0].params.textData!.questionLength = 0
    expect(parseProjectStateFile(JSON.stringify(invalid)).ok).toBe(false)
    const wrongExperiment = structuredClone(file)
    wrongExperiment.state.graph.nodes[0].params.storyExperiment = {facts: ['Mary'], question: '?'}
    expect(parseProjectStateFile(JSON.stringify(wrongExperiment)).ok).toBe(false)
  })

  it('matches a legacy fact model after removing its input reshapes and slices, with identical parameters', async () => {
    const data = storyQuestionData(), graph = storyQuestionModel(data), original = structuredClone(graph)
    const legacy = storyQuestionModel({...data, representation: 'facts', maxFacts: 1, factWords: 32,
      documents: data.documents.map(doc => ({...doc, text: doc.text + ' ' + doc.question, facts: [doc.text]}))})
    expect(parameterValues(legacy)).toEqual(parameterValues(graph))
    expect(graph.nodes.some(n => ['story_ids', 'short_question', 'question_ids', 'story_mask', 'short_question_mask', 'question_mask'].includes(n.id))).toBe(false)
    for (let row = 0; row < data.documents.length; row++) {
      const actual = forwardPass(withDatasetExample(graph, 'text-data', row)), expected = forwardPass(withDatasetExample(legacy, 'text-data', row))
      for (const id of ['story_lookup', 'question_lookup', 'attention', 'output', 'probabilities']) expect(actual.graph.nodes.find(n => n.id === id)!.value).toEqual(expected.graph.nodes.find(n => n.id === id)!.value)
      expect(actual.loss).toBe(expected.loss)
      expect(parameterValues(actual.graph)).toEqual(parameterValues(graph))
    }
    expect(evaluateDataset(graph, 'text-data', 'test')).toEqual(evaluateDataset(legacy, 'text-data', 'test'))
    expect(graph).toEqual(original)
  })

  it('evaluates temporary inputs and blocks parameter updates, while held-out inference and epochs use originals', async () => {
    const graph = storyQuestionModel(), original = structuredClone(graph), story = {text: 'John garden garden', question: 'where John ?'}
    const edited = {...graph, nodes: graph.nodes.map(n => n.type === 'dataset' ? {...n, params: {...n.params, storyExperiment: story}} : n)}
    const encoded = encodeStory(story, graph.nodes[0].params.textData!)
    encoded.features.forEach((value, slot) => expect(datasetOutputValueForSlot(edited.nodes[0], slot)).toEqual(value))
    expect(datasetOutputValueForSlot(edited.nodes[0], 4)).toEqual(datasetOutputValueForSlot(graph.nodes[0], 4))
    const actual = forwardPass(edited)
    expect(actual.graph.nodes.find(n => n.id === 'probabilities')!.value).not.toEqual(forwardPass(graph).graph.nodes.find(n => n.id === 'probabilities')!.value)
    expect(parameterValues(actual.graph)).toEqual(parameterValues(graph))
    expect(() => runTrainingStep(edited)).toThrow(/Restore the original/)
    expect(() => runTrainingStepFast(edited)).toThrow(/Restore the original/)
    expect(() => updateParameters(backwardPass(actual.graph).graph)).toThrow(/Restore the original/)
    expect(evaluateDataset(edited, 'text-data', 'test')).toEqual(evaluateDataset(graph, 'text-data', 'test'))
    expect(withDatasetExample(edited, 'text-data', 2).nodes[0].params.storyExperiment).toBeUndefined()
    expect(parameterValues(await trainDataset(edited, 'text-data', 1))).toEqual(parameterValues(await trainDataset(graph, 'text-data', 1)))
    expect(graph).toEqual(original)
  })

  it('batches each input at its own length and matches trace losses, gradients, held-out predictions and inspection', async () => {
    const graph = storyQuestionModel(storyQuestionData(7, 4)), original = structuredClone(graph), model = new TensorGraph(graph)
    try {
      const rows = model.examples.slice(0, 2), expected = rows.map((_, i) => backwardPass(forwardPass(withDatasetExample(graph, 'text-data', i)).graph))
      for (const [slot, shape] of [[0, [2, 7]], [1, [2, 4]], [2, [2, 7, 1]], [3, [2, 4, 1]], [4, [2, 1]]] as const) {
        const signal = await model.inspectSignal(rows, 'text-data', slot)
        expect(signal!.value.shape).toEqual(shape)
        expect(signal!.value.data).toEqual(rows.flatMap(row => slot === 4 ? row.target.data : row.features[slot].data))
      }
      const gradients = model.gradients(rows)
      try {
        expect((await gradients.loss.data())[0]).toBeCloseTo((expected[0].loss! + expected[1].loss!) / 2, 5)
        for (const [id, variable] of model.variables) {
          const a = expected[0].graph.nodes.find(n => n.id === id)!.grad!.data, b = expected[1].graph.nodes.find(n => n.id === id)!.grad!.data
          ;(await gradients.grads[variable.name].data()).forEach((value, i) => expect(value, `${id}[${i}]`).toBeCloseTo((a[i] + b[i]) / 2, 5))
        }
      } finally {tf.dispose([gradients.loss, ...Object.values(gradients.grads)])}
      const report = await model.inference(model.examples.filter(row => row.split === 'test'), 2), trace = evaluateDataset(graph, 'text-data', 'test')
      expect(report.rows).toEqual(trace.rows)
      expect(report.loss).toBeCloseTo(trace.loss, 5)
      expect(report.rows.map(row => row.example)).toEqual(['Dataset row 2', 'Dataset row 3'])
      const inspection = new InferenceInspection(graph, 'text-data', 'test', 'cpu', 2)
      for (const row of [2, 3]) {
        const result = await inspection.read('probabilities', 0, row)
        expect(result.status).toBe('available')
        if (result.status !== 'available') throw Error(result.message)
        const single = forwardPass(withDatasetExample(graph, 'text-data', row)).graph.nodes.find(n => n.id === 'probabilities')!.value!
        result.value.data.forEach((value, i) => expect(value).toBeCloseTo(single.data[i], 6))
      }
      expect(tensorAxisLabels(graph, 'question_lookup')[0]).toHaveLength(4)
      expect(tensorAxisLabels(graph, 'story_lookup')[0]).toHaveLength(7)
      expect(tensorAxisLabels(graph, 'embedding')[0]).toEqual(graph.nodes[0].params.textData!.vocabulary)
      expect(graph).toEqual(original)
    } finally {model.dispose()}
  })

  it('trains Tensor minibatches on the training split with frozen preprocessing and ignores temporary experiments', async () => {
    const graph = storyQuestionModel(storyQuestionData(5, 4)), original = structuredClone(graph)
    const edited = {...graph, nodes: graph.nodes.map(n => n.type === 'dataset' ? {...n, params: {...n.params, datasetIndex: 2, storyExperiment: {text: 'John garden', question: 'where John ?'}}} : n)}
    const options = {epochs: 2, batchSize: 2, shuffle: false, settings: {...DEFAULT_TRAINING, engine: 'tensor' as const, backend: 'cpu' as const, optimizer: 'sgd' as const, clipNorm: 0}}
    const actual = await trainTensorGraph(edited, options), baseline = await trainTensorGraph(graph, options)
    expect(actual.completed).toBe(2)
    expect(parameterValues(actual.graph)).toEqual(parameterValues(baseline.graph))
    expect(parameterValues(actual.graph)).not.toEqual(parameterValues(graph))
    expect(actual.graph.nodes[0].params.textData).toEqual(graph.nodes[0].params.textData)
    expect(graph).toEqual(original)
    const changedHeldOut = {...graph, nodes: graph.nodes.map(n => n.type === 'dataset' ? {...n, params: {...n.params, textData: {...n.params.textData!, documents: n.params.textData!.documents.map(doc => doc.split === 'test' ? {...doc, text: 'John garden garden', question: 'where John ?'} : doc)}}} : n)}
    const changed = await trainTensorGraph(changedHeldOut, options)
    expect(parameterValues(changed.graph)).toEqual(parameterValues(baseline.graph))
  })
})
