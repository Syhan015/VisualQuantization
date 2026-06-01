import { create } from 'zustand'
import type { ViewState } from '../types/viewState'

interface SyncState {
  leftView: ViewState
  rightView: ViewState
  activeSync: boolean
  sourceId: 'left' | 'right' | null

  setLeftView: (view: ViewState) => void
  setRightView: (view: ViewState) => void
  setActiveSync: (active: boolean) => void
  acquireLock: (side: 'left' | 'right') => boolean
  releaseLock: () => void
}

export const useSyncStore = create<SyncState>((set, get) => ({
  leftView:  { zoom: 1, scrollLeft: 0, scrollTop: 0 },
  rightView: { zoom: 1, scrollLeft: 0, scrollTop: 0 },
  activeSync: true,
  sourceId: null,

  setLeftView: (view) => set({ leftView: view }),

  setRightView: (view) => set({ rightView: view }),

  setActiveSync: (active) => set({ activeSync: active }),

  acquireLock: (side) => {
    if (get().sourceId !== null) return false
    set({ sourceId: side })
    return true
  },

  releaseLock: () => set({ sourceId: null }),
}))
