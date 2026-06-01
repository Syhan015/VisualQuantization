import { useEffect, useRef, useCallback } from 'react'

interface NetronViewer {
  start: (file: File, container: HTMLElement | string) => unknown
  stop: (viewer: unknown) => void
}

declare global {
  interface Window {
    netron?: NetronViewer
  }
}

export function useNetron(containerId: string, modelFile: File | null) {
  const viewerRef = useRef<unknown>(null)

  const loadNetronScript = useCallback((): Promise<void> => {
    return new Promise((resolve, reject) => {
      if (window.netron) {
        resolve()
        return
      }

      const script = document.createElement('script')
      script.src = '/netron/netron.bundle.js'
      script.async = true
      script.onload = () => resolve()
      script.onerror = () => reject(new Error('Failed to load Netron'))
      document.body.appendChild(script)
    })
  }, [])

  useEffect(() => {
    if (!modelFile) return

    let currentViewer: unknown = null

    const initAndRender = async () => {
      try {
        await loadNetronScript()

        if (!containerId) return

        const container =
          typeof containerId === 'string'
            ? document.getElementById(containerId)
            : containerId

        if (!container) return

        // Stop previous viewer if exists
        if (viewerRef.current && window.netron) {
          window.netron.stop(viewerRef.current)
        }

        // Clear container
        container.innerHTML = ''

        // Start new viewer
        if (window.netron) {
          currentViewer = window.netron.start(modelFile, container)
          viewerRef.current = currentViewer
        }
      } catch (error) {
        console.error('Failed to initialize Netron:', error)
      }
    }

    initAndRender()

    return () => {
      if (currentViewer && window.netron) {
        window.netron.stop(currentViewer)
      }
    }
  }, [containerId, modelFile, loadNetronScript])

  return { viewerRef }
}
