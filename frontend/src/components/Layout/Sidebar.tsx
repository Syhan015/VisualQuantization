import { useState } from 'react'
import { useModelStore } from '../../stores/modelStore'
import { useSyncStore } from '../../stores/syncStore'
import { useDiffHighlight } from '../../hooks/useDiffHighlight'
import { api } from '../../services/api'

export function Sidebar() {
  const {
    models, leftModelId, rightModelId, addModel, setLeftModel, setRightModel,
    isComparing, currentDiff, setWeightAnalysis, setCurrentComparedNode,
    weightAnalysis,
  } = useModelStore()
  const { activeSync, setActiveSync } = useSyncStore()
  const { runCompare } = useDiffHighlight()
  const canCompare = leftModelId !== null && rightModelId !== null && !isComparing

  const [weightPanelOpen, setWeightPanelOpen] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [matchMode, setMatchMode] = useState<'conservative' | 'aggressive'>('aggressive')

  const handleFileSelect = (targetPane: 'left' | 'right') => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.onnx'
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return

      const model = {
        id: crypto.randomUUID(),
        name: file.name,
        file: file,
        uploadedAt: new Date(),
      }

      addModel(model)

      if (targetPane === 'left') {
        setLeftModel(model.id)
      } else {
        setRightModel(model.id)
      }
    }
    input.click()
  }

  const handleCompare = () => {
    if (canCompare && leftModelId && rightModelId) {
      runCompare(leftModelId, rightModelId, matchMode)
    }
  }

  const handleNodeClick = async (nodeName: string, modelBNodeName?: string) => {
    if (!nodeName) return  // Guard against empty node name

    const modelA = useModelStore.getState().getModelById(leftModelId)
    const modelB = useModelStore.getState().getModelById(rightModelId)
    const uploadedIdA = modelA?.uploadedId
    const uploadedIdB = modelB?.uploadedId

    if (!uploadedIdA || !uploadedIdB) {
      console.error('Model not uploaded yet')
      return
    }

    setIsAnalyzing(true)
    setCurrentComparedNode(nodeName || modelBNodeName || 'unknown')
    try {
      const result = await api.compareWeights(uploadedIdA, uploadedIdB, nodeName, modelBNodeName)
      setWeightAnalysis(result)
    } catch (err) {
      console.error('Failed to compare weights:', err)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const weightableOps = ['Conv', 'MatMul', 'Gemm', 'Linear', 'QLinearConv', 'QLinearMatMul']
  const modifiedNodes = currentDiff?.nodes.filter(
    n => n.diff_type === 'modified' && weightableOps.includes(n.op_type)
  ) ?? []

  return (
    <aside className="w-64 bg-gray-800 border-r border-gray-700 flex flex-col">
      <div className="p-4 border-b border-gray-700 space-y-2">
        <button
          onClick={() => handleFileSelect('left')}
          className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 rounded text-sm font-medium transition-colors text-left"
        >
          + 加载左侧模型 (FP32)
        </button>
        <button
          onClick={() => handleFileSelect('right')}
          className="w-full py-2 px-4 bg-purple-600 hover:bg-purple-700 rounded text-sm font-medium transition-colors text-left"
        >
          + 加载右侧模型 (INT8)
        </button>
        <button
          onClick={handleCompare}
          disabled={!canCompare}
          className={`w-full py-2 px-4 rounded text-sm font-medium transition-colors text-left ${
            canCompare
              ? 'bg-green-600 hover:bg-green-700'
              : 'bg-gray-600 cursor-not-allowed opacity-50'
          }`}
        >
          {isComparing ? '对比中...' : '▶ 对比模型'}
        </button>
        <div className="flex items-center gap-2 px-2 py-1 bg-gray-800 rounded text-xs">
          <span className="text-gray-400">匹配模式:</span>
          <button
            onClick={() => setMatchMode('conservative')}
            className={`px-2 py-0.5 rounded ${matchMode === 'conservative' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'}`}
          >
            保守
          </button>
          <button
            onClick={() => setMatchMode('aggressive')}
            className={`px-2 py-0.5 rounded ${matchMode === 'aggressive' ? 'bg-orange-600 text-white' : 'bg-gray-700 text-gray-300'}`}
          >
            激进
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        <h2 className="text-xs font-semibold text-gray-400 uppercase mb-2 px-2">
          模型列表
        </h2>
        <div className="space-y-1">
          {models.map((model) => {
            const isLeft = leftModelId === model.id
            const isRight = rightModelId === model.id
            return (
              <div
                key={model.id}
                className="p-2 rounded text-sm truncate cursor-pointer hover:bg-gray-700 text-gray-300"
                title={model.name}
              >
                <div className="flex items-center gap-1">
                  <span className={`w-2 h-2 rounded-full ${isLeft ? 'bg-blue-500' : isRight ? 'bg-purple-500' : 'bg-gray-500'}`} />
                  <span className="truncate">{model.name}</span>
                </div>
                <div className="text-xs text-gray-500 mt-0.5 ml-3">
                  {isLeft && '← 左侧'}
                  {isRight && '→ 右侧'}
                  {isLeft && isRight && '←→ 双侧'}
                </div>
              </div>
            )
          })}
          {models.length === 0 && (
            <p className="text-xs text-gray-500 px-2">暂无模型</p>
          )}
        </div>

        {currentDiff && (
          <div className="mt-4 border-t border-gray-700 pt-4">
            <button
              onClick={() => setWeightPanelOpen(!weightPanelOpen)}
              className="w-full flex items-center justify-between px-2 py-1 text-xs font-semibold text-gray-400 uppercase hover:text-gray-300"
            >
              <span>权重分析</span>
              <span className={`transform transition-transform ${weightPanelOpen ? 'rotate-90' : ''}`}>
                ▶
              </span>
            </button>

            {weightPanelOpen && (
              <div className="mt-2 space-y-1">
                {modifiedNodes.length === 0 ? (
                  <p className="text-xs text-gray-500 px-2">无修改节点</p>
                ) : (
                  modifiedNodes.map((node) => {
                    const displayName = node.name || node.op_type || 'unknown'
                    const isFusion = node.fusion_info?.is_fusion
                    return (
                      <button
                        key={node.id || displayName}
                        onClick={() => handleNodeClick(node.name, node.matched_name)}
                        disabled={!node.name}
                        className={`w-full p-2 rounded text-left text-sm hover:bg-gray-700 text-gray-300 truncate ${
                          !node.name ? 'opacity-50 cursor-not-allowed' : ''
                        }`}
                        title={node.matched_name ? `${node.name || node.op_type} → ${node.matched_name}` : (node.name || node.op_type)}
                      >
                        <div className="flex items-center gap-1">
                          <span className={`w-2 h-2 rounded-full ${isFusion ? 'bg-orange-500' : 'bg-yellow-500'}`} />
                          <span className="truncate">{displayName}</span>
                          {isFusion && <span className="text-xs text-orange-400 ml-1">[fused]</span>}
                        </div>
                        {isFusion && node.fusion_info?.fp32_components && (
                          <div className="text-xs text-gray-500 ml-4 truncate">
                            {node.fusion_info.fp32_components.join(' → ')}
                          </div>
                        )}
                        {!isFusion && node.matched_name && node.matched_name !== node.name && (
                          <div className="text-xs text-gray-500 ml-4 truncate">
                            → {node.matched_name}
                          </div>
                        )}
                      </button>
                    )
                  })
                )}
                {isAnalyzing && (
                  <p className="text-xs text-blue-400 px-2">分析中...</p>
                )}
              </div>
            )}
          </div>
        )}

        {weightAnalysis && (
          <div className="mt-4 border-t border-gray-700 pt-4 px-2">
            <h3 className="text-xs font-semibold text-gray-400 uppercase mb-2">
              指标详情
            </h3>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-400">Cosine:</span>
                <span className="text-white">{weightAnalysis.cosineSimilarity.toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">L2 Error:</span>
                <span className="text-white">{weightAnalysis.l2Error.toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">MAE:</span>
                <span className="text-white">{weightAnalysis.mae.toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Mean Diff:</span>
                <span className="text-white">{weightAnalysis.meanDiff.toFixed(6)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Std Diff:</span>
                <span className="text-white">{weightAnalysis.stdDiff.toFixed(6)}</span>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="p-4 border-t border-gray-700">
        <label className="flex items-center justify-between">
          <span className="text-sm text-gray-300">双栏联动</span>
          <button
            role="switch"
            aria-checked={activeSync}
            onClick={() => setActiveSync(!activeSync)}
            className={`relative w-10 h-5 rounded-full transition-colors ${
              activeSync ? 'bg-blue-600' : 'bg-gray-600'
            }`}
          >
            <span
              className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-transform ${
                activeSync ? 'translate-x-5' : 'translate-x-0'
              }`}
            />
          </button>
        </label>
        <p className="text-xs text-gray-500 mt-1">
          {activeSync ? '两栏同步缩放和平移' : '左右两栏独立操作'}
        </p>
      </div>
    </aside>
  )
}
