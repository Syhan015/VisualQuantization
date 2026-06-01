import { useEffect, useRef, useState } from 'react'
import type { ModelInfo } from '../../types/model'
import { api } from '../../services/api'
import './ModelViewer.css'

interface ModelViewerProps {
  model: ModelInfo
}

export function ModelViewer({ model }: ModelViewerProps) {
  const iframeRef = useRef<HTMLIFrameElement>(null)
  const [remoteUrl, setRemoteUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!model.file) return

    const loadModel = async () => {
      try {
        setError(null)
        // Upload to backend first
        const response = await api.uploadModel(model.file)
        // Create URL that Netron can fetch via XHR
        const modelUrl = `http://localhost:9000/api/models/${response.id}/download`
        setRemoteUrl(modelUrl)
      } catch (err) {
        console.error('Failed to upload model:', err)
        setError('Failed to load model')
      }
    }

    loadModel()
  }, [model.file])

  useEffect(() => {
    if (!iframeRef.current || !remoteUrl) return

    // Set the iframe src with the model URL as hash
    // Netron will parse the hash and fetch the model
    iframeRef.current.src = `/netron/index.html#${remoteUrl}`
  }, [remoteUrl])

  if (error) {
    return (
      <div className="model-viewer h-full w-full bg-gray-900 flex items-center justify-center">
        <div className="text-red-500">{error}</div>
      </div>
    )
  }

  if (!remoteUrl) {
    return (
      <div className="model-viewer h-full w-full bg-gray-900 flex items-center justify-center">
        <div className="text-gray-400">Loading model...</div>
      </div>
    )
  }

  return (
    <div className="model-viewer h-full w-full bg-gray-900">
      <iframe
        ref={iframeRef}
        title="Netron Viewer"
        className="w-full h-full border-0"
        allow="accelerometer; camera; encrypted-media; geolocation; gyroscope; microphone; midi; clipboard-read; clipboard-write"
      />
    </div>
  )
}
