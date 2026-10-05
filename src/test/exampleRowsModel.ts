import { builder } from './curriculumModels'
import { parseCustomCsv } from '../domain/customCsv'

/** Noncontiguous original rows make split-local numbering errors visible. */
export function exampleRowsModel() {
  const { graph, add } = builder()
  const csv = parseCustomCsv('a,b,y,split\n1,10,0,test\n2,20,1,train\n3,30,0,test\n4,40,1,train\n5,50,0,train\n6,60,1,test\n', 'rows.csv')
  add('data', 'dataset', { dataset: 'custom-csv', customCsv: csv, datasetMode: 'batch', datasetSplit: 'train' })
  add('columns', 'concat', { axis: 1 }, [['data', 0], ['data', 1]])
  add('scaled', 'standardize', { standardization: { mean: [0, 0], scale: [1, 10], count: 3 } }, ['columns'])
  add('weights', 'weight', { value: { shape: [2, 2], data: [1, 0, 0, 1] } })
  add('product', 'matmul', {}, ['scaled', 'weights'])
  add('bias', 'bias', { value: { shape: [2], data: [1, 2] } })
  add('biased', 'arithmetic', { expression: 'x1 + x2' }, ['bias', 'product'])
  add('probabilities', 'softmax', {}, ['biased'])
  add('loss', 'loss', { loss: 'cross-entropy' }, ['biased', ['data', 2]])
  return { graph, add }
}
