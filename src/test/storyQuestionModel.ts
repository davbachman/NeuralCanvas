import { prepareTextDocuments } from '../domain/textData'
import { builder } from './curriculumModels'
import type { TextDatasetData } from '../domain/types'

export const storyQuestionDocuments = [
  { text: 'Mary kitchen', question: 'where Mary ?', label: 'kitchen', split: 'train' as const },
  { text: 'John garden', question: 'where John ?', label: 'garden', split: 'train' as const },
  { text: 'Mary garden', question: 'where Mary ?', label: 'garden', split: 'test' as const },
  { text: 'John kitchen', question: 'where John ?', label: 'kitchen', split: 'test' as const },
]

export function storyQuestionData(maxLength = 32, questionLength = 8): TextDatasetData {
  return { ...prepareTextDocuments(storyQuestionDocuments.map(doc => ({...doc})), 'story-question.json', {
    task: 'classification', representation: 'story-question', maxLength, questionLength,
  }), vocabulary: ['<unk>', '<pad>', 'mary', 'john', 'kitchen', 'garden', 'where', '?'], classLabels: ['kitchen', 'garden'] }
}

/** The legacy graph adapts one whole-story fact and a longer padded question
 * using input reshapes/slices. The new graph consumes the five inputs directly. */
export function storyQuestionModel(data = storyQuestionData()) {
  const { graph, add, weight, linear, reshape } = builder(137)
  graph.training = { ...graph.training!, engine: 'trace', backend: 'cpu', optimizer: 'sgd', weightDecay: 0, clipNorm: 0, patience: 0 }
  add('text-data', 'dataset', { dataset: 'custom-text', textData: data, datasetMode: 'sample', datasetIndex: 0 })
  const legacy = data.representation === 'facts'
  const questionLength = data.questionLength ?? data.factWords!
  const source = (slot: number): [string, number] => ['text-data', slot]
  const storyIds = legacy ? reshape('story_ids', add('old_story_ids', 'input', {}, [source(0)]), [data.factWords!]) : source(0)
  const questionIds = legacy ? reshape('question_ids', add('short_question', 'tensor-transform', { transform: 'slice', axis: 1, start: 0, end: questionLength }, [add('old_question_ids', 'input', {}, [source(1)])]), [questionLength]) : source(1)
  const storyMask = legacy ? reshape('story_mask', add('old_story_mask', 'input', {}, [source(4)]), [data.factWords!, 1]) : source(3)
  const questionMask = legacy ? reshape('question_mask', add('short_question_mask', 'tensor-transform', { transform: 'slice', axis: 1, start: 0, end: questionLength }, [add('old_question_mask', 'input', {}, [source(5)])]), [questionLength, 1]) : source(4)
  const table = weight('embedding', [data.vocabulary.length, 4], 'uniform')
  const positions = weight('positions', [legacy ? data.factWords! : data.maxLength, 4], 'uniform')
  const positionVectors = legacy ? positions : add('position_lookup', 'embedding', {}, [positions, source(2)])
  const positioned = add('positioned_story', 'arithmetic', { expression: 'x1 + x2' }, [add('story_lookup', 'embedding', {}, [table, storyIds]), positionVectors])
  const story = add('story_embeddings', 'arithmetic', { expression: 'x1 * x2' }, [positioned, storyMask])
  const question = add('question_embeddings', 'arithmetic', { expression: 'x1 * x2' }, [add('question_lookup', 'embedding', {}, [table, questionIds]), questionMask])
  const query = add('query', 'tensor-transform', { transform: 'mean', axis: 0, keepDims: true }, [question])
  const keys = add('keys', 'tensor-transform', { transform: 'transpose', axes: [1, 0] }, [story])
  const mask = add('word_mask', 'tensor-transform', { transform: 'transpose', axes: [1, 0] }, [storyMask])
  const scores = add('masked_scores', 'arithmetic', { expression: 'x1 + (x2 - 1) * 1000000000' }, [add('scores', 'matmul', {}, [query, keys]), mask])
  const attention = add('attention', 'softmax', {}, [scores])
  const context = add('context', 'arithmetic', { expression: 'x1 + x2' }, [query, add('read_story', 'matmul', {}, [attention, story])])
  const logits = linear('output', context, 4, 2)
  add('probabilities', 'softmax', {}, [logits])
  add('loss', 'loss', { loss: 'cross-entropy' }, [logits, source(legacy ? 6 : 5)])
  return graph
}
