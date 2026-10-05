import { createModelPreset } from '../domain/modelPresets'
import { parameterValues } from '../domain/engine'
import type { RecoveryWorkspace } from '../domain/localRecovery'

export function recoveryWorkspace(): RecoveryWorkspace {
  const graph = createModelPreset('linear')
  return {
    state: {
      graph, visualizationGraph: graph, initialParameterValues: parameterValues(graph), selectedNodeIds: [],
      phase: 'edit', traceSteps: [], traceIndex: 0, epoch: 500, currentLoss: 2,
      runSettings: { epochsPerRun: '2000', reportEvery: '500', examplesPerUpdate: '3' },
      display: { showMath: true, showGradient: true, showCode: false, showVisualization: true },
    },
    lossReports: [{ epoch: 0, loss: 8, heldOutLoss: 9 }, { epoch: 500, loss: 2, heldOutLoss: 3 }],
    shuffleEachEpoch: false,
  }
}
