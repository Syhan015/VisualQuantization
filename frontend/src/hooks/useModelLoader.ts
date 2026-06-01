import { useCallback } from 'react'
import type { ModelInfo } from '../types/model'

export function useModelLoader() {
  const loadModelFromFile = useCallback((file: File): Promise<ModelInfo> => {
    return new Promise((resolve, reject) => {
      if (!file.name.endsWith('.onnx')) {
        reject(new Error('Only .onnx files are supported'))
        return
      }

      const modelInfo: ModelInfo = {
        id: crypto.randomUUID(),
        name: file.name,
        file: file,
        uploadedAt: new Date(),
      }

      resolve(modelInfo)
    })
  }, [])

  return { loadModelFromFile }
}
