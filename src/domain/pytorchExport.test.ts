import { describe, expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createModelPreset } from './modelPresets'
import { LESSONS } from '../learning/presets'
import { createEmptyGraph, createNode } from './examples'
import { parseCustomCsv } from './customCsv'
import { datasetExamplesForNode } from './datasets'
import { backwardPass, forwardPass } from './engine'
import { evaluateDataset, withDatasetExample } from './datasetTraining'
import { parameterPenalty } from './regularization'
import { prepareTextDocuments } from './textData'
import { generatePyTorchExport } from './pytorchExport'
import type { GraphModel } from './types'

function customArithmeticGraph(): GraphModel {
  const dataset = createNode('dataset', 1)
  const csv = parseCustomCsv('answer,feature\n2,1\n4,2\n6,3\n8,4\n', 'measurements.csv')
  csv.targetColumn = 0
  dataset.params = { dataset: 'custom-csv', customCsv: csv, datasetMode: 'batch', datasetSplit: 'train' }
  const parameter = createNode('weight', 2)
  parameter.params.value = 0.5
  const arithmetic = createNode('arithmetic', 3)
  arithmetic.params.expression = 'x1 * x2 + 1'
  const target = createNode('target', 4)
  const loss = createNode('loss', 5)
  loss.params.loss = 'mse'
  return { learningRate: 0.01, nodes: [dataset, parameter, arithmetic, target, loss], edges: [
    { id: 'feature', source: dataset.id, sourceSlot: 1, target: arithmetic.id, inputSlot: 0 },
    { id: 'parameter', source: parameter.id, target: arithmetic.id, inputSlot: 1 },
    { id: 'target-column', source: dataset.id, sourceSlot: 0, target: target.id, inputSlot: 0 },
    { id: 'prediction', source: arithmetic.id, target: loss.id, inputSlot: 0 },
    { id: 'loss-target', source: target.id, target: loss.id, inputSlot: 1 },
  ] }
}

function sparseAutoencoderGraph(): GraphModel {
  const graph = createEmptyGraph()
  const node = (id: string, type: GraphModel['nodes'][number]['type'], params: GraphModel['nodes'][number]['params'] = {}) => ({ id, type, label: id, params, position: { x: 0, y: 0 } })
  graph.nodes = [createNode('dataset', 1),
    node('residual', 'weight', { value: { shape: [3, 2], data: [1, -2, 2, 1, 99, -99] }, trainable: false }),
    node('stable', 'dropout', { dropoutRate: 0.5, inferenceOnly: true }),
    node('encoder', 'weight', { value: { shape: [2, 3], data: [0.5, -0.4, 0, 0.2, 0.7, 0] } }),
    node('encode', 'matmul'), node('features', 'activation', { activation: 'relu' }),
    node('decoder', 'weight', { value: { shape: [3, 2], data: [1, 0, 0, 1, 1, 0] }, constraint: 'unit-row-norm' }),
    node('reconstruction', 'matmul'),
    node('valid', 'input', { value: { shape: [3, 1], data: [1, 1, 0] } }),
    node('loss', 'loss', { loss: 'mse', targetMode: 'internal', lossMaskNodeId: 'valid', activationPenaltyNodeId: 'features', activationPenaltyStrength: 0.1 }),
  ]
  const edge = (source: string, target: string, inputSlot: number) => ({ id: `${source}-${target}`, source, target, inputSlot })
  graph.edges = [edge('residual', 'stable', 0), edge('stable', 'encode', 0), edge('encoder', 'encode', 1),
    edge('encode', 'features', 0), edge('features', 'reconstruction', 0), edge('decoder', 'reconstruction', 1),
    edge('reconstruction', 'loss', 0), edge('stable', 'loss', 1)]
  graph.learningRate = 0.01
  return graph
}

describe('PyTorch export', () => {
  it.each(LESSONS.map(lesson => lesson.id))(
    'produces a runnable notebook for the %s preset', (preset) => {
      const exported = generatePyTorchExport(createModelPreset(preset))
      const notebook = JSON.parse(exported.notebook)
      expect(notebook.nbformat).toBe(4)
      expect(notebook.cells.find((cell: { cell_type: string }) => cell.cell_type === 'code').source.join('')).toBe(exported.script)
      expect(exported.script).toContain('class BuilderModel(nn.Module):')
      expect(exported.script).toContain('torch.optim.SGD')
      expect(exported.script).toContain('Test predictions')
    },
  )

  it('keeps dataset examples outside the Python and notebook files', () => {
    const graph = createModelPreset('linear')
    const parameter = graph.nodes.find(node => node.type === 'weight')!
    parameter.params.value = 3.75
    const exported = generatePyTorchExport(graph)
    const script = exported.script
    expect(script).toContain('torch.tensor([3.75], dtype=torch.float64).reshape([])')
    expect(script).toContain('DATASET = load_dataset()')
    expect(script).toContain('json.load(source)')
    expect(script).not.toContain('-3.62')
    expect(exported.datasetFile?.name).toBe('neural-canvas-dataset.json')
    expect(JSON.parse(exported.datasetFile!.content).examples).toHaveLength(20)
    expect(exported.notebook).not.toContain('-3.62')
    expect(script).toContain('TRAIN_EPOCHS = 10')
  })

  it('exports mini-batch and dual-loss reporting settings', () => {
    const script = generatePyTorchExport(createModelPreset('linear'), { batchSize: 4, shuffleEachEpoch: false, epochs: 3, reportEvery: 2 }).script
    expect(script).toContain('BATCH_MODE = EVALUATE_FULL_BATCH or BATCH_SIZE > 1')
    expect(script).toContain('BATCH_SIZE = 4')
    expect(script).toContain('train_loader = DataLoader(')
    expect(script).toContain('SHUFFLE_EACH_EPOCH = False')
    expect(script).toContain('TRAIN_EPOCHS = 3')
    expect(script).toContain('REPORT_EVERY = 2')
    expect(script).toContain('held_out_loss')
    expect(script).toContain("label='Held-out (validation)'")
    expect(() => generatePyTorchExport(createModelPreset('attention'), { batchSize: 4 })).toThrow('tensor-shaped dataset')
  })

  it('exports a hand-built arithmetic graph with custom CSV columns', () => {
    const exported = generatePyTorchExport(customArithmeticGraph())
    expect(exported.script).toContain('Custom CSV · measurements.csv')
    expect(exported.script).toContain('DATASET_FILE = "measurements.csv"')
    expect(exported.script).toContain('csv.reader(source)')
    expect(exported.datasetFile).toBeUndefined()
    expect(exported.script).not.toContain('answer,feature')
    expect(exported.script).toContain('v_dataset_1_s1 = features[0]')
    expect(exported.script).toContain('v_dataset_1_s0 = target')
    expect(exported.script).toContain(' * ')
  })

  it('includes library CSV examples in the model export without requiring a local source file', () => {
    const graph = customArithmeticGraph()
    graph.nodes.find(node => node.type === 'dataset')!.params.datasetLibraryPreset = 'auto-mpg'
    const exported = generatePyTorchExport(graph)
    expect(exported.script).toContain('DATASET_FILE = "neural-canvas-dataset.json"')
    expect(exported.script).not.toContain('csv.reader(source)')
    const data = JSON.parse(exported.datasetFile!.content)
    expect(data.examples).toHaveLength(4)
    expect(data.examples[0]).toMatchObject({ features: [{ data: [1] }], target: { data: [2] } })
    expect(exported.script).toContain('v_dataset_1_s0 = target')
    expect(exported.script).toContain('v_dataset_1_s1 = features[0]')
  })

  it('refuses incomplete builders with an actionable error', () => {
    expect(() => generatePyTorchExport(createEmptyGraph())).toThrow('Dataset block')
    const graph = createModelPreset('linear')
    graph.edges = graph.edges.filter(edge => edge.target !== 'loss' || edge.inputSlot !== 0)
    expect(() => generatePyTorchExport(graph)).toThrow('Finish the graph')
  })

  const python = process.env.PYTORCH_TEST_PYTHON
  const syntaxPython = [python, process.env.PYTHON, 'python3'].find(candidate =>
    candidate && spawnSync(candidate, ['-c', 'import ast'], { encoding: 'utf8' }).status === 0)
  it.skipIf(!syntaxPython)('generates syntactically valid Python for built-in and custom datasets', () => {
    for (const graph of [createModelPreset('linear'), createModelPreset('attention'), customArithmeticGraph()]) {
      const { script } = generatePyTorchExport(graph)
      const result = spawnSync(syntaxPython!, ['-c', 'import ast,sys; ast.parse(sys.stdin.read())'], { input: script, encoding: 'utf8' })
      expect(result.status, result.stderr).toBe(0)
    }
  })
  it.skipIf(!syntaxPython)('loads built-in data from a sibling JSON file and custom data from its CSV', () => {
    for (const graph of [createModelPreset('linear'), customArithmeticGraph()]) {
      const exported = generatePyTorchExport(graph)
      const directory = mkdtempSync(join(tmpdir(), 'backprop-data-'))
      try {
        if (exported.datasetFile) writeFileSync(join(directory, exported.datasetFile.name), exported.datasetFile.content)
        else writeFileSync(join(directory, 'measurements.csv'), 'answer,feature\n2,1\n4,2\n6,3\n8,4\n')
        const loader = exported.script.slice(exported.script.indexOf('DATASET_DIR = '), exported.script.indexOf('def binary_cross_entropy'))
        const program = `import csv, json\nfrom pathlib import Path\n${loader}\nprint(len(DATASET['examples']))\nprint(DATASET['examples'][0]['features'][0]['data'][0])\n`
        const result = spawnSync(syntaxPython!, ['-c', program], { cwd: directory, encoding: 'utf8' })
        expect(result.status, result.stderr).toBe(0)
        expect(result.stdout.trim().split('\n')).toEqual(exported.datasetFile ? ['20', '-2.4'] : ['4', '1.0'])
      } finally { rmSync(directory, { recursive: true, force: true }) }
    }
  })
  function runWithDataset(exported: ReturnType<typeof generatePyTorchExport>, script: string, harness?: string, csv = 'answer,feature\n2,1\n4,2\n6,3\n8,4\n') {
    const directory = mkdtempSync(join(tmpdir(), 'backprop-export-'))
    try {
      writeFileSync(join(directory, 'neural-canvas-model.py'), script)
      if (exported.datasetFile) writeFileSync(join(directory, exported.datasetFile.name), exported.datasetFile.content)
      else writeFileSync(join(directory, 'measurements.csv'), csv)
      return spawnSync(python!, harness ? ['-c', harness] : ['neural-canvas-model.py'], {
        cwd: directory, input: harness ? script : undefined, encoding: 'utf8', timeout: 120_000, env: { ...process.env, MPLBACKEND: 'Agg' },
      })
    } finally { rmSync(directory, { recursive: true, force: true }) }
  }
  it.skipIf(!python)('matches stable binary logits losses in the generated Python program', () => {
    const graph = customArithmeticGraph()
    const source = graph.nodes.find(node => node.type === 'dataset')!
    const csv = 'answer,feature,split\n1,-1000,train\n0,1000,train\n1,0.1,test\n0,-0.1,test\n'
    source.params.customCsv = parseCustomCsv(csv, 'measurements.csv')
    source.params.customCsv.targetColumn = 0
    source.params.customCsv.task = 'binary-classification'
    graph.nodes.find(node => node.type === 'arithmetic')!.params.expression = 'x1 * x2'
    graph.nodes.find(node => node.type === 'loss')!.params.loss = 'binary-cross-entropy-with-logits'
    const exported = generatePyTorchExport(graph, { epochs: 0 })
    const harness = `import sys,json,math,torch
ns={}
exec(sys.stdin.read(),ns)
print('LOGIT_LOSSES=' + json.dumps([ns['initial_train_loss'],ns['initial_held_out_loss']]))
features=torch.tensor([-1000.,1000.,0.],requires_grad=True)
targets=torch.tensor([1.,0.,1.],requires_grad=True)
prediction,loss,actual=ns['model']([features],targets)
loss.backward()
assert math.isclose(float(loss.detach()),(1000.+math.log(2.))/3.,rel_tol=1e-12)
assert math.isclose(float(ns['model'].p_1.grad),2000./3.,rel_tol=1e-12)
assert torch.allclose(features.grad,torch.tensor([-1./6.,1./6.,-1./12.]))
assert targets.grad is None and not actual.requires_grad
print('LOGIT_GRADIENTS_AND_DETACH_OK')
`
    const run = runWithDataset(exported, exported.script, harness, csv)
    expect(run.status, run.stderr).toBe(0)
    const actual = JSON.parse(run.stdout.match(/LOGIT_LOSSES=([^\n]+)/)![1])
    expect(actual[0]).toBeCloseTo(evaluateDataset(graph, source.id, 'train').loss, 9)
    expect(actual[1]).toBeCloseTo(evaluateDataset(graph, source.id, 'test').loss, 9)
    expect(run.stdout).toContain('LOGIT_GRADIENTS_AND_DETACH_OK')
  }, 30_000)

  it.skipIf(!python)('exports sparse reconstruction gradients, frozen values, padding and row constraints', () => {
    const graph = sparseAutoencoderGraph()
    const forward = forwardPass(graph, false)
    const backward = backwardPass(forward.graph).graph
    const exported = generatePyTorchExport(graph, { epochs: 0 })
    const harness = `import sys,json,torch
ns={}
exec(sys.stdin.read(),ns)
model=ns['model']
row=ns['DATASET']['examples'][0]
features,target=ns['model_inputs']([row],False)
model.train()
_,loss,_=model(features,target)
loss.backward()
print('SAE=' + json.dumps({'loss':float(loss.detach()),'data_loss':float(model.data_loss.detach()),'penalty':float(model.activation_penalty.detach()),'gradients':[p.grad.reshape(-1).tolist() if p.grad is not None else None for p in model.parameters()]}))
before=model.p_1.detach().clone()
optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=0.1,weight_decay=0.5)
optimizer.step()
model.project_constraints()
assert torch.equal(model.p_1,before)
assert torch.allclose(torch.linalg.vector_norm(model.p_3,dim=1),torch.ones(3))
zero=torch.zeros((2,3))
ns['project_unit_rows_'](zero)
assert zero.tolist()==[[1.,0.,0.],[1.,0.,0.]]
extreme=torch.tensor([[1e308,1e308],[1e-300,1e-300]])
ns['project_unit_rows_'](extreme)
assert torch.allclose(torch.linalg.vector_norm(extreme,dim=1),torch.ones(2))
x=torch.tensor([[0.,1.],[4.,-4.]],requires_grad=True)
mask=torch.tensor([[1.],[0.]])
ns['activation_l1'](x,mask).backward()
assert x.grad.tolist()==[[0.,1.],[0.,0.]]
assert model.sparsity['tolerance']==0
print('SAE_INVARIANTS_OK')
`
    const run = runWithDataset(exported, exported.script, harness)
    expect(run.status, run.stderr).toBe(0)
    const actual = JSON.parse(run.stdout.match(/SAE=([^\n]+)/)![1])
    expect(actual.loss).toBeCloseTo(forward.loss!, 9)
    expect(actual.loss).toBeCloseTo(actual.data_loss + actual.penalty, 9)
    expect(actual.gradients[0]).toBeNull()
    for (const [index, id] of ['encoder', 'decoder'].entries()) {
      const expected = backward.nodes.find(node => node.id === id)!.grad!.data
      expected.forEach((value, coordinate) => expect(actual.gradients[index + 1][coordinate]).toBeCloseTo(value, 9))
    }
    expect(run.stdout).toContain('SAE_INVARIANTS_OK')
  }, 30_000)

  it.skipIf(!python).each(['mse', 'mae'] as const)('excludes intrinsic padding and reports token-weighted internal %s', loss => {
    const graph = sparseAutoencoderGraph()
    const dataset = graph.nodes.find(node => node.type === 'dataset')!
    dataset.params = { dataset: 'custom-text', datasetMode: 'sample', datasetIndex: 0,
      textData: prepareTextDocuments([
        { text: 'good', label: 'positive', split: 'train' },
        { text: 'good film', label: 'positive', split: 'train' },
        { text: 'bad film', label: 'negative', split: 'test' },
      ], 'padding.csv', { task: 'sentiment', representation: 'tokens', fixedLength: true, maxLength: 3 }) }
    graph.nodes.find(node => node.id === 'loss')!.params.loss = loss
    const exported = generatePyTorchExport(graph, { epochs: 0 })
    const harness = `import sys,json
ns={}
exec(sys.stdin.read(),ns)
ns['evaluate'](ns['model'],ns['train_rows'])
print('MASKED=' + json.dumps(ns['model'].evaluation_metrics))
assert ns['model'].sparsity['tokens']==2
`
    const run = runWithDataset(exported, exported.script, harness)
    expect(run.status, run.stderr).toBe(0)
    const actual = JSON.parse(run.stdout.match(/MASKED=([^\n]+)/)![1])
    expect(actual.data_loss).toBeCloseTo(evaluateDataset(graph, dataset.id, 'train').loss!, 9)
    expect(actual.objective).toBeCloseTo(actual.data_loss + actual.activation_penalty, 9)
    expect(actual.inactive_features).toBe(2)
    expect(actual.sparsity_tolerance).toBe(0)
  }, 30_000)

  it.skipIf(!python)('matches stable ranking and exact indexed masks in exported Python', () => {
    const graph=customArithmeticGraph()
    const scores={...createNode('input',101),params:{value:{shape:[1,5],data:[2,100,1,2,-3]}}}
    const order=createNode('argsort',102),selected=createNode('index-mask',103)
    graph.nodes.push(scores,order,selected)
    graph.edges.push({id:'export-sort',source:scores.id,target:order.id,inputSlot:0},
      {id:'export-mask-values',source:scores.id,target:selected.id,inputSlot:0},
      {id:'export-mask-indices',source:order.id,target:selected.id,inputSlot:1})
    const exported=generatePyTorchExport(graph,{epochs:0})
    expect(exported.script).toContain(' = sort_indices(')
    expect(exported.script).toContain(' = index_mask(')
    const harness="import sys,json,torch\nns={}\nexec(sys.stdin.read(),ns)\nx=torch.tensor([[2.,100.,1.,2.,-3.]],requires_grad=True)\neligible=ns['index_mask'](x,torch.tensor([1.]),exclude=True)\nids=ns['sort_indices'](eligible)[:,:2]\ny=ns['index_mask'](x,ids)\np=torch.softmax(y,dim=-1)\nassert p.tolist()==[[0.5,0.,0.,0.5,0.]]\ny[0,0].backward()\nassert x.grad.tolist()==[[1.,0.,0.,0.,0.]]\nassert ns['sort_indices'](eligible,descending=False).tolist()==[[4,2,0,3,1]]\nprint('INDEXED_SAMPLING_OK')\n"
    const run=runWithDataset(exported,exported.script,harness)
    expect(run.status,run.stderr).toBe(0)
    expect(run.stdout).toContain('INDEXED_SAMPLING_OK')
  },30_000)

  it.skipIf(!python).each([0, 1, 2])('reloads the original CSV with an explicit split in column %s', splitColumn => {
    const rows = [['answer', 'feature'], ['2', '1'], ['4', '2'], ['6', '3'], ['8', '4']]
    const splits = ['split', 'train', 'test', 'train', 'test']
    rows.forEach((row, index) => row.splice(splitColumn, 0, splits[index]))
    const csv = rows.map(row => row.join(',')).join('\n')
    const graph = customArithmeticGraph()
    const source = graph.nodes.find(node => node.type === 'dataset')!
    source.params.customCsv = { ...parseCustomCsv(csv, 'measurements.csv'), targetColumn: 0 }
    const exported = generatePyTorchExport(graph, { epochs: 1 })
    const harness = `import sys,json\nns={}\nexec(sys.stdin.read(),ns)\nprint('ROWS=' + json.dumps([(row['features'][0]['data'][0],row['target']['data'][0],row['split']) for row in ns['DATASET']['examples']]))\n`
    const run = runWithDataset(exported, exported.script, harness, csv)
    expect(run.status, run.stderr).toBe(0)
    expect(JSON.parse(run.stdout.match(/ROWS=([^\n]+)/)![1])).toEqual([[1, 2, 'train'], [2, 4, 'test'], [3, 6, 'train'], [4, 8, 'test']])
    expect(run.stdout).toContain('Epoch 1: train loss=')
  }, 30_000)

  it.skipIf(!python).each(['l1', 'l2'] as const)('reports unpenalized train and held-out loss with %s regularization', regularization => {
    const graph = createModelPreset('linear')
    const source = graph.nodes.find(node => node.type === 'dataset')!
    graph.nodes.find(node => node.type === 'loss')!.params = { loss: 'mse', regularization, regularizationStrength: 10 }
    const exported = generatePyTorchExport(graph, { epochs: 0 })
    const harness = `import sys,json\nns={}\nexec(sys.stdin.read(),ns)\nprint('LOSSES=' + json.dumps([ns['initial_train_loss'],ns['initial_held_out_loss'],float(ns['model'].regularization_penalty().detach())]))\n`
    const run = runWithDataset(exported, exported.script, harness)
    expect(run.status, run.stderr).toBe(0)
    const actual = JSON.parse(run.stdout.match(/LOSSES=([^\n]+)/)![1])
    expect(actual[0]).toBeCloseTo(evaluateDataset(graph, source.id, 'train').loss, 9)
    expect(actual[1]).toBeCloseTo(evaluateDataset(graph, source.id, 'test').loss, 9)
    expect(actual[2]).toBeCloseTo(parameterPenalty(graph), 9)
    expect(actual[2]).toBeGreaterThan(0)
  }, 30_000)

  it.skipIf(!python)('evaluates a fixed prediction with an unused parameter without calling backward', () => {
    const graph = customArithmeticGraph()
    graph.nodes.find(node => node.type === 'weight')!.type = 'input'
    graph.nodes.push(createNode('weight', 99))
    const exported = generatePyTorchExport(graph, { epochs: 1 })
    const run = runWithDataset(exported, exported.script)
    expect(run.status, run.stderr).toBe(0)
    expect(run.stdout).toContain('Epoch 1: train loss=')
    expect(run.stdout).toContain('Test predictions')
  }, 30_000)

  it.skipIf(!python)('runs the exported mini-batch loop and reports both losses', () => {
    const exported = generatePyTorchExport(createModelPreset('linear'), { batchSize: 4, epochs: 2, reportEvery: 1 })
    const run = runWithDataset(exported, exported.script)
    expect(run.status, run.stderr).toBe(0)
    expect(run.stdout).toContain('Epoch 2: train loss=')
    expect(run.stdout).toContain('held-out loss=')
  }, 120_000)

  it.skipIf(!python)('runs every preset in PyTorch and matches builder losses', () => {
    for (const { id: preset } of LESSONS) {
      const graph = createModelPreset(preset)
      const dataset = graph.nodes.find(node => node.type === 'dataset')!
      const exampleIndex = datasetExamplesForNode(dataset).findIndex(example => example.split === 'train')
      const expected = forwardPass(withDatasetExample(graph, dataset.id, exampleIndex)).loss
      const exported = generatePyTorchExport(graph)
      const script = exported.script
      const harness = `import sys, json\nns = {}\nexec(sys.stdin.read().split('\\nmodel = BuilderModel()\\n')[0], ns)\nmodel = ns['BuilderModel']()\nrow = ns['DATASET']['examples'][${exampleIndex}]\nfeatures = [ns['tensor_value'](value) for value in row['features']]\ntarget = ns['tensor_value'](row['target'])\noutput, loss, _ = model(features, target)\nprint('EXPORT_OUTPUT=' + str(output.numel()))\nprint('EXPORT_LOSS=' + str(float(loss)) if loss is not None else 'EXPORT_LOSS=none')\n`
      const run = runWithDataset(exported, script, harness)
      expect(run.status, `${preset}: ${run.stderr}`).toBe(0)
      expect(Number(run.stdout.match(/EXPORT_OUTPUT=([^\n]+)/)?.[1])).toBeGreaterThan(0)
      if (expected !== undefined) {
        const actual = Number(run.stdout.match(/EXPORT_LOSS=([^\n]+)/)?.[1])
        expect(actual, preset).toBeCloseTo(expected, 7)
      }
    }
  }, 240_000)

  it.skipIf(!python)('runs training and inference exports end to end', () => {
    for (const preset of ['linear', 'attention'] as const) {
      const exported = generatePyTorchExport(createModelPreset(preset))
      const script = exported.script.replace('TRAIN_EPOCHS = 10', 'TRAIN_EPOCHS = 1')
      const run = runWithDataset(exported, script)
      expect(run.status, `${preset}: ${run.stderr}`).toBe(0)
      expect(run.stdout).toContain('Test predictions')
      if (preset === 'linear') expect(run.stdout).toContain('Epoch 1: train loss=')
    }
  }, 180_000)

  it.skipIf(!python)('matches a custom CSV batch and arithmetic expression', () => {
    const graph = customArithmeticGraph()
    const expected = forwardPass(graph).loss!
    const exported = generatePyTorchExport(graph)
    const script = exported.script
    const harness = `import sys\nns = {}\nexec(sys.stdin.read().split('\\nmodel = BuilderModel()\\n')[0], ns)\nmodel = ns['BuilderModel']()\nrows = [row for row in ns['DATASET']['examples'] if row['split'] == 'train']\nfeatures, target = ns['model_inputs'](rows)\n_, loss, _ = model(features, target)\nprint('EXPORT_LOSS=' + str(float(loss)))\n`
    const run = runWithDataset(exported, script, harness)
    expect(run.status, run.stderr).toBe(0)
    expect(Number(run.stdout.match(/EXPORT_LOSS=([^\n]+)/)?.[1])).toBeCloseTo(expected, 7)
  }, 120_000)
})
