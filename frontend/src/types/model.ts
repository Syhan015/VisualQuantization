export interface ModelInfo {
  id: string
  name: string
  file: File
  uploadedAt: Date
  uploadedId?: string
}

export interface ModelMetadata {
  id: string
  name: string
  nodeCount: number
  inputShapes: string[]
  outputShapes: string[]
  sizeBytes: number
}

export interface QuantizationInfo {
  is_quantized: boolean
  quant_type?: string
  scale?: number
  zero_point?: number
  axis?: number
}

export interface DiffNode {
  id: string
  name: string
  op_type: string
  diff_type: 'added' | 'removed' | 'modified' | 'unchanged'
  details?: string
  quantization?: QuantizationInfo
  matched_name?: string
  fusion_info?: FusionInfo
}

export interface FusionInfo {
  is_fusion: boolean
  fp32_components?: string[]
  confidence?: number
}

export interface DiffSummary {
  totalNodesA: number
  totalNodesB: number
  addedCount: number
  removedCount: number
  modifiedCount: number
  quantizationDetected: boolean
  quantNodesA: number
  quantNodesB: number
}

export interface DiffResult {
  id: string
  modelAId: string
  modelBId: string
  nodes: DiffNode[]
  summary: DiffSummary
}

export interface WeightAnalysis {
  nodeName: string
  cosineSimilarity: number
  l2Error: number
  mae: number
  meanDiff: number
  stdDiff: number
  distributionA: number[]
  distributionB: number[]
  binEdges?: number[]
}
