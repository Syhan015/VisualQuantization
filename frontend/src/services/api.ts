import type { ModelMetadata, DiffResult, WeightAnalysis } from '../types/model'

const API_BASE = '/api'

function toWeightAnalysis(raw: Record<string, unknown>): WeightAnalysis {
  return {
    nodeName: raw.node_name as string,
    cosineSimilarity: raw.cosine_similarity as number,
    l2Error: raw.l2_error as number,
    mae: raw.mae as number,
    meanDiff: raw.mean_diff as number,
    stdDiff: raw.std_diff as number,
    distributionA: raw.distribution_a as number[],
    distributionB: raw.distribution_b as number[],
    binEdges: raw.bin_edges as number[] | undefined,
  }
}

export const api = {
  async uploadModel(file: File): Promise<{ id: string }> {
    const formData = new FormData()
    formData.append('file', file)
    const response = await fetch(`${API_BASE}/models/upload`, {
      method: 'POST',
      body: formData,
    })
    if (!response.ok) throw new Error('Failed to upload model')
    return response.json()
  },

  async getModelMetadata(id: string): Promise<ModelMetadata> {
    const response = await fetch(`${API_BASE}/models/${id}`)
    if (!response.ok) throw new Error('Failed to get model metadata')
    return response.json()
  },

  async compareDiff(modelAId: string, modelBId: string, matchMode: 'conservative' | 'aggressive' = 'conservative'): Promise<{ id: string }> {
    const response = await fetch(`${API_BASE}/diff/compare?match_mode=${matchMode}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_a_id: modelAId, model_b_id: modelBId }),
    })
    if (!response.ok) throw new Error('Failed to compare models')
    return response.json()
  },

  async getDiffResult(diffId: string): Promise<DiffResult> {
    const response = await fetch(`${API_BASE}/diff/${diffId}/result`)
    if (!response.ok) throw new Error('Failed to get diff result')
    return response.json()
  },

  async analyzeWeights(
    modelId: string,
    nodeName: string
  ): Promise<WeightAnalysis> {
    const response = await fetch(`${API_BASE}/weights/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: modelId, node_name: nodeName }),
    })
    if (!response.ok) throw new Error('Failed to analyze weights')
    return toWeightAnalysis(await response.json())
  },

  async compareWeights(
    modelAId: string,
    modelBId: string,
    nodeName: string,
    modelBNodeName?: string
  ): Promise<WeightAnalysis> {
    const response = await fetch(`${API_BASE}/weights/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_a_id: modelAId,
        model_b_id: modelBId,
        node_name: nodeName,
        model_b_node_name: modelBNodeName,
      }),
    })
    if (!response.ok) throw new Error('Failed to compare weights')
    return toWeightAnalysis(await response.json())
  },
}
