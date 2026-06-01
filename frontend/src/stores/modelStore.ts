import { create } from 'zustand'
import type { ModelInfo, DiffResult, WeightAnalysis } from '../types/model'

interface ModelState {
  models: ModelInfo[]
  leftModelId: string | null
  rightModelId: string | null
  currentDiff: DiffResult | null
  isComparing: boolean
  weightAnalysis: WeightAnalysis | null
  currentComparedNode: string | null

  addModel: (model: ModelInfo) => void
  removeModel: (id: string) => void
  setLeftModel: (id: string | null) => void
  setRightModel: (id: string | null) => void
  getModelById: (id: string | null) => ModelInfo | null
  setCurrentDiff: (diff: DiffResult | null) => void
  clearDiff: () => void
  setIsComparing: (comparing: boolean) => void
  updateModelUploadedId: (id: string, uploadedId: string) => void
  setWeightAnalysis: (analysis: WeightAnalysis | null) => void
  setCurrentComparedNode: (nodeName: string | null) => void
}

export const useModelStore = create<ModelState>((set, get) => ({
  models: [],
  leftModelId: null,
  rightModelId: null,
  currentDiff: null,
  isComparing: false,
  weightAnalysis: null,
  currentComparedNode: null,

  addModel: (model) =>
    set((state) => {
      const newModels = [...state.models, model]
      return {
        models: newModels,
        leftModelId: state.leftModelId ?? model.id,
      }
    }),

  removeModel: (id) =>
    set((state) => {
      const newModels = state.models.filter((m) => m.id !== id)
      return {
        models: newModels,
        leftModelId: state.leftModelId === id ? (newModels[0]?.id ?? null) : state.leftModelId,
        rightModelId: state.rightModelId === id ? null : state.rightModelId,
        currentDiff: state.currentDiff?.modelAId === id || state.currentDiff?.modelBId === id
          ? null
          : state.currentDiff,
        weightAnalysis: state.currentComparedNode ? null : state.weightAnalysis,
      }
    }),

  setLeftModel: (id) => set({ leftModelId: id }),

  setRightModel: (id) => set({ rightModelId: id }),

  getModelById: (id) => {
    if (!id) return null
    return get().models.find((m) => m.id === id) ?? null
  },

  setCurrentDiff: (diff) => set({ currentDiff: diff }),

  clearDiff: () => set({ currentDiff: null, weightAnalysis: null, currentComparedNode: null }),

  setIsComparing: (comparing) => set({ isComparing: comparing }),

  updateModelUploadedId: (id, uploadedId) =>
    set((state) => ({
      models: state.models.map((m) =>
        m.id === id ? { ...m, uploadedId } : m
      ),
    })),

  setWeightAnalysis: (analysis) => set({ weightAnalysis: analysis }),

  setCurrentComparedNode: (nodeName) => set({ currentComparedNode: nodeName }),
}))
