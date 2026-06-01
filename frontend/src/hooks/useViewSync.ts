import { useEffect, useRef } from 'react'
import { useSyncStore } from '../stores/syncStore'
import type { SyncSide } from '../types/viewState'

type NetronWindow = Window & { __view__?: any }
type ViewState = { zoom: number; scrollLeft: number; scrollTop: number }

interface SyncMessage {
  type: 'vq-scroll' | 'vq-sync-done'
  zoom?: number
  scrollLeft?: number
  scrollTop?: number
}

const HEARTBEAT_INTERVAL = 150 // ms

export function useViewSync(side: SyncSide) {
  const pendingRafRef = useRef<number | null>(null)
  const storeRef = useRef(useSyncStore.getState())
  const heartbeatRef = useRef<number | null>(null)
  const lastSyncedRef = useRef<ViewState>({ zoom: 1, scrollLeft: 0, scrollTop: 0 })
  const iframeReadyRef = useRef<boolean>(false)

  useEffect(() => {
    const unsubscribe = useSyncStore.subscribe((state) => {
      storeRef.current = state
    })
    storeRef.current = useSyncStore.getState()

    const releaseLock = () => {
      storeRef.current.releaseLock()
    }

    // Get view state from Netron iframe
    const getIframeView = (iframeId: string): ViewState | null => {
      const iframe = document.getElementById(iframeId) as HTMLIFrameElement | null
      if (!iframe) {
        console.log(`[getIframeView] iframe not found: ${iframeId} (DOM may not be ready yet)`)
        return null
      }
      if (!iframe.contentWindow) {
        console.log(`[getIframeView] iframe.contentWindow null: ${iframeId}`)
        return null
      }
      const win = iframe.contentWindow as NetronWindow
      if (!win.__view__?._target) {
        console.log(`[getIframeView] __view__._target not found on ${iframeId} (Netron not loaded yet)`)
        return null
      }
      const view = win.__view__._target
      console.log(`[getIframeView] _target keys on ${iframeId}:`, Object.keys(view || {}).join(', '))
      console.log(`[getIframeView] view type:`, typeof view, ', view?.getViewState:', typeof view?.getViewState)
      console.log(`[getIframeView] _target._zoom:`, view?._zoom, ', _scrollLeft:', view?._scrollLeft, ', _scrollTop:', view?._scrollTop)
      if (typeof view?.getViewState === 'function') {
        const state = view.getViewState()
        if (!state) return null
        return {
          zoom: state.zoom ?? 1,
          scrollLeft: state.scrollLeft ?? 0,
          scrollTop: state.scrollTop ?? 0,
        }
      }
      // Fallback: read state directly from _target properties
      console.log(`[getIframeView] falling back to direct property read`)
      return {
        zoom: view?._zoom ?? 1,
        scrollLeft: view?._scrollLeft ?? 0,
        scrollTop: view?._scrollTop ?? 0,
      }
    }

    // Check if iframe __view__ is available
    const isIframeReady = (iframeId: string): boolean => {
      const iframe = document.getElementById(iframeId) as HTMLIFrameElement | null
      if (!iframe) {
        console.log(`[isIframeReady] iframe not found: ${iframeId}`)
        return false
      }
      if (!iframe.contentWindow) {
        console.log(`[isIframeReady] iframe.contentWindow null: ${iframeId}`)
        return false
      }
      const win = iframe.contentWindow as NetronWindow
      if (!win.__view__?.['_target']) {
        console.log(`[isIframeReady] __view__._target not ready: ${iframeId}`)
        return false
      }
      // Check for either getViewState method or direct _zoom/_scrollLeft/_scrollTop properties
      const view = win.__view__._target
      const hasMethod = typeof view?.getViewState === 'function'
      const hasProps = view?._zoom !== undefined && view?._scrollLeft !== undefined
      if (!hasMethod && !hasProps) {
        console.log(`[isIframeReady] neither getViewState nor _zoom/_scrollLeft found: ${iframeId}`)
        return false
      }
      return true
    }

    // Set view state on Netron iframe
    const setIframeView = (iframeId: string, view: ViewState): boolean => {
      const iframe = document.getElementById(iframeId) as HTMLIFrameElement | null
      if (!iframe) {
        console.log(`[setIframeView] iframe not found: ${iframeId}`)
        return false
      }
      if (!iframe.contentWindow) {
        console.log(`[setIframeView] iframe.contentWindow null: ${iframeId}`)
        return false
      }
      const win = iframe.contentWindow as NetronWindow
      if (!win.__view__?.['_target']) {
        console.log(`[setIframeView] __view__._target not found: ${iframeId}`)
        return false
      }
      const target = win.__view__._target
      if (typeof target.syncFromExternal === 'function') {
        console.log(`[setIframeView] calling syncFromExternal on ${iframeId}`, view)
        target.syncFromExternal(view)
        return true
      }
      // Fallback: set properties directly
      console.log(`[setIframeView] syncFromExternal not found, setting _zoom/_scrollLeft/_scrollTop directly`)
      target._zoom = view.zoom
      target._scrollLeft = view.scrollLeft
      target._scrollTop = view.scrollTop
      return true
    }

    // Immediate sync attempt when iframe becomes ready
    const tryInitialSync = () => {
      if (side !== 'left') return
      const store = storeRef.current
      if (!store.activeSync) return
      if (!isIframeReady('netron-left') || !isIframeReady('netron-right')) return

      const leftView = getIframeView('netron-left')
      if (leftView) {
        setIframeView('netron-right', leftView)
        lastSyncedRef.current = { ...leftView }
      }
    }

    // Heartbeat: only left side runs this to avoid conflicts
    const heartbeat = () => {
      if (side !== 'left') return

      console.log('[Heartbeat] firing...')

      const store = storeRef.current
      if (!store.activeSync) return

      // Get current views
      const leftView = getIframeView('netron-left')
      const rightView = getIframeView('netron-right')

      console.log('[Heartbeat] leftView:', leftView, 'rightView:', rightView)

      if (!leftView || !rightView) return

      // Update store
      store.setLeftView(leftView)
      store.setRightView(rightView)

      // Check if already in sync
      const threshold = 0.001
      if (
        Math.abs(leftView.zoom - rightView.zoom) < threshold &&
        Math.abs(leftView.scrollLeft - rightView.scrollLeft) < 1 &&
        Math.abs(leftView.scrollTop - rightView.scrollTop) < 1
      ) {
        return
      }

      // Check if significantly different from last sync
      const last = lastSyncedRef.current
      if (
        Math.abs(leftView.zoom - last.zoom) < threshold &&
        Math.abs(leftView.scrollLeft - last.scrollLeft) < 1 &&
        Math.abs(leftView.scrollTop - last.scrollTop) < 1
      ) {
        return
      }

      // Sync right to match left
      if (setIframeView('netron-right', leftView)) {
        lastSyncedRef.current = { ...leftView }
      }
    }

    // Start heartbeat only on left side
    if (side === 'left') {
      heartbeatRef.current = window.setInterval(heartbeat, HEARTBEAT_INTERVAL)
    }

    // Poll for iframe ready state, then do initial sync
    const pollForIframeReady = () => {
      if (iframeReadyRef.current) return
      if (isIframeReady('netron-left') && isIframeReady('netron-right')) {
        iframeReadyRef.current = true
        tryInitialSync()
      }
    }
    const pollInterval = window.setInterval(pollForIframeReady, 100)

    // Original message handler for when Netron DOES send events
    const handler = (e: MessageEvent) => {
      const data = e.data as SyncMessage
      if (!data?.type?.startsWith('vq-')) return

      if (data.type === 'vq-sync-done') {
        releaseLock()
        return
      }

      if (data.type !== 'vq-scroll') return

      const state: ViewState = {
        zoom: data.zoom ?? 1,
        scrollLeft: data.scrollLeft ?? 0,
        scrollTop: data.scrollTop ?? 0,
      }

      const store = storeRef.current
      if (!store.activeSync) return

      if (!store.acquireLock(side)) return

      if (side === 'left') {
        store.setLeftView(state)
      } else {
        store.setRightView(state)
      }

      if (pendingRafRef.current !== null) {
        cancelAnimationFrame(pendingRafRef.current)
      }

      pendingRafRef.current = requestAnimationFrame(() => {
        pendingRafRef.current = null

        const current = storeRef.current
        if (!current.activeSync) {
          releaseLock()
          return
        }

        const srcView = side === 'left' ? current.leftView : current.rightView
        const dstView = side === 'left' ? current.rightView : current.leftView

        // Skip if already in sync
        const threshold = 0.001
        if (
          Math.abs(srcView.zoom - dstView.zoom) < threshold &&
          Math.abs(srcView.scrollLeft - dstView.scrollLeft) < 1 &&
          Math.abs(srcView.scrollTop - dstView.scrollTop) < 1
        ) {
          releaseLock()
          return
        }

        const otherSide: SyncSide = side === 'left' ? 'right' : 'left'
        const otherIframeId = otherSide === 'left' ? 'netron-left' : 'netron-right'
        const otherIframe = document.getElementById(otherIframeId) as HTMLIFrameElement | null
        const otherWindow = otherIframe?.contentWindow as NetronWindow | undefined

        const target = otherWindow?.__view__?._target
        if (typeof target?.syncFromExternal === 'function') {
          target.syncFromExternal(srcView)
        } else {
          releaseLock()
        }
      })
    }

    window.addEventListener('message', handler)

    return () => {
      if (heartbeatRef.current !== null) {
        clearInterval(heartbeatRef.current)
      }
      clearInterval(pollInterval)
      window.removeEventListener('message', handler)
      unsubscribe()
    }
  }, [side])
}