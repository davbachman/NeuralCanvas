import { createStarterGraph } from '../domain/examples'
import { parseCustomCsv } from '../domain/customCsv'

export function heldOutInspectionModel() {
  const graph = createStarterGraph(true)
  const source = graph.nodes.find(node => node.type === 'dataset')!
  const customCsv = parseCustomCsv('x,target,split\n-1,0,train\n1,1,train\n-2,0,test\n2,1,test\n', 'inspection.csv')
  customCsv.task = 'binary-classification'
  source.params = { dataset: 'custom-csv', customCsv, datasetMode: 'sample', datasetSplit: 'train', datasetIndex: 0 }
  return graph
}
