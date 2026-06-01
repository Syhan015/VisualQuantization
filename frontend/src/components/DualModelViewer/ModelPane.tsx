import { useEffect, useRef, useState } from 'react'
import { api } from '../../services/api'
import { useViewSync } from '../../hooks/useViewSync'
import { useModelStore } from '../../stores/modelStore'
import type { ModelInfo } from '../../types/model'
import './ModelPane.css'

interface ModelPaneProps {
  model: ModelInfo | null
  paneId: 'left' | 'right'
  label: string
}

export function ModelPane({ model, paneId, label }: ModelPaneProps) {
  const iframeRef = useRef<HTMLIFrameElement>(null)
  const [remoteUrl, setRemoteUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const updateModelUploadedId = useModelStore((s) => s.updateModelUploadedId)

  useViewSync(paneId)

  useEffect(() => {
    if (!model?.file) {
      setRemoteUrl(null)
      return
    }

    const loadModel = async () => {
      try {
        setError(null)
        const response = await api.uploadModel(model.file)
        updateModelUploadedId(model.id, response.id)
        const modelUrl = `http://localhost:9000/api/models/${response.id}/download`
        setRemoteUrl(modelUrl)
      } catch (err) {
        console.error('Failed to upload model:', err)
        setError('Failed to load model')
      }
    }

    loadModel()
  }, [model?.file])

  useEffect(() => {
    if (!iframeRef.current || !remoteUrl) return
    iframeRef.current.src = `/netron/index.html#${remoteUrl}`
  }, [remoteUrl])

  const iframeId = paneId === 'left' ? 'netron-left' : 'netron-right'

  if (error) {
    return (
      <div className="model-pane h-full w-full bg-gray-900 flex flex-col items-center justify-center">
        <span className="pane-label">{label}</span>
        <div className="text-red-500 mt-2">{error}</div>
      </div>
    )
  }

  if (!remoteUrl) {
    return (
      <div className="model-pane h-full w-full bg-gray-900 flex flex-col items-center justify-center">
        <span className="pane-label">{label}</span>
        <div className="text-gray-400 mt-2">Loading model...</div>
      </div>
    )
  }

  return (
    <div className="model-pane h-full w-full bg-gray-900 flex flex-col">
      <div className="pane-label-bar">
        <span className="pane-label">{label}</span>
      </div>
      <div className="flex-1 relative">
        <iframe
          ref={iframeRef}
          id={iframeId}
          title={`Netron Viewer ${label}`}
          className="w-full h-full border-0"
          allow="accelerometer; camera; encrypted-media; geolocation; gyroscope; microphone; midi; clipboard-read; clipboard-write"
        />
      </div>
    </div>
  )
}
