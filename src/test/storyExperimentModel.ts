import { prepareTextDocuments } from '../domain/textData'
import { buildQaModel } from './curriculumModels'

export function storyExperimentData() {
  const documents = [
    {facts: ['Mary is in kitchen.', 'John is in garden.'], question: 'Where is Mary?', label: 'kitchen', split: 'train' as const},
    {facts: ['Mary is in garden.', 'John is in kitchen.'], question: 'Where is John?', label: 'kitchen', split: 'train' as const},
    {facts: ['John is in garden.'], question: 'Where is John?', label: 'garden', split: 'train' as const},
    {facts: ['Mary is in garden.', 'John is in kitchen.'], question: 'Where is Mary?', label: 'garden', split: 'test' as const},
  ].map(doc => ({...doc, text: [...doc.facts, doc.question].join(' ')}))
  return prepareTextDocuments(documents, 'stories.csv', {task: 'classification', representation: 'facts', factWords: 6, maxFacts: 3})
}

export function storyExperimentModel() {
  const graph = buildQaModel(storyExperimentData(), 'memory-1', 137, 4)
  graph.training = {...graph.training!, engine: 'trace', backend: 'cpu', optimizer: 'sgd'}
  return graph
}
