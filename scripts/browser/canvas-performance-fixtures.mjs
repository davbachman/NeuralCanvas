import { createServer } from 'vite'
import { mkdir, writeFile } from 'node:fs/promises'

// Uses only repository helpers; no coursework/model downloads are needed.
const server = await createServer({ server: { middlewareMode: true } })
try {
  const { canvasPerformanceModel } = await server.ssrLoadModule('/src/test/canvasPerformanceModel.ts')
  const { createProjectStateFile, parseProjectStateFile } = await server.ssrLoadModule('/src/domain/session.ts')
  await mkdir('output', { recursive: true })
  for (const embedding of [false, true]) {
    const graph = canvasPerformanceModel(embedding)
    const state = {
      graph, visualizationGraph: graph, initialParameterValues: {},
      selectedNodeIds: [], phase: 'backward', traceSteps: [], traceIndex: 0, epoch: 1, currentLoss: null,
      display: { showMath: true, showGradient: true, showCode: false, showVisualization: false },
    }
    const text = JSON.stringify(createProjectStateFile(state))
    const parsed = parseProjectStateFile(text)
    if (!parsed.ok) throw Error(parsed.error)
    await writeFile(`output/canvas-${embedding ? 'embedding' : 'onehot'}.json`, text)
  }
} finally {
  await server.close()
}
