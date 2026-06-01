import { useEffect } from 'react'
import { useModelStore } from '../stores/modelStore'
import { api } from '../services/api'

type DiffType = 'added' | 'removed' | 'modified' | 'unchanged'

interface HighlightMap {
  [nodeName: string]: DiffType
}

export function useDiffHighlight() {
  const { currentDiff, setCurrentDiff, setIsComparing, getModelById } = useModelStore()

  useEffect(() => {
    if (!currentDiff) {
      return
    }

    const leftHighlights: HighlightMap = {}
    const rightHighlights: HighlightMap = {}

    for (const node of currentDiff.nodes) {
      if (node.diff_type === 'unchanged') {
        continue
      }

      if (node.diff_type === 'modified') {
        // Modified node: left uses node.name, right uses matched_name (or node.name if same)
        if (node.name) {
          leftHighlights[node.name] = node.diff_type
        }
        // If matched_name is null but names match, use node.name for right side
        const rightName = node.matched_name || (node.name !== '' ? node.name : null)
        if (rightName) {
          rightHighlights[rightName] = node.diff_type
        }
      } else if (node.diff_type === 'removed') {
        // Removed node: only on LEFT (FP32 only)
        if (node.name) {
          leftHighlights[node.name] = node.diff_type
        }
      } else if (node.diff_type === 'added') {
        // Added node: only on RIGHT (INT8 only)
        if (node.name) {
          rightHighlights[node.name] = node.diff_type
        }
        // Always add op_type fallback for unnamed nodes (QuantizeLinear/DequantizeLinear etc.)
        rightHighlights[`__op__${node.op_type}`] = node.diff_type
      }
    }

    const leftIframe = document.getElementById('netron-left') as HTMLIFrameElement | null
    const rightIframe = document.getElementById('netron-right') as HTMLIFrameElement | null

    // Send targeted highlights to each iframe
    leftIframe?.contentWindow?.postMessage({
      type: 'vq-highlight',
      highlights: leftHighlights,
    }, '*')

    rightIframe?.contentWindow?.postMessage({
      type: 'vq-highlight',
      highlights: rightHighlights,
    }, '*')

    return () => {
      leftIframe?.contentWindow?.postMessage({ type: 'vq-clear-highlight' }, '*')
      rightIframe?.contentWindow?.postMessage({ type: 'vq-clear-highlight' }, '*')
    }
  }, [currentDiff])

  const runCompare = async (modelAId: string, modelBId: string, matchMode: 'conservative' | 'aggressive' = 'aggressive') => {
    setIsComparing(true)
    try {
      const modelA = getModelById(modelAId)
      const modelB = getModelById(modelBId)
      const uploadedIdA = modelA?.uploadedId
      const uploadedIdB = modelB?.uploadedId

      if (!uploadedIdA || !uploadedIdB) {
        throw new Error('Model not uploaded yet')
      }

      const { id: diffId } = await api.compareDiff(uploadedIdA, uploadedIdB, matchMode)
      const result = await api.getDiffResult(diffId)
      setCurrentDiff(result)
    } catch (error) {
      console.error('Compare failed:', error)
    } finally {
      setIsComparing(false)
    }
  }

  return { runCompare }
}