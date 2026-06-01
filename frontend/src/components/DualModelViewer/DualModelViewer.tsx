import { ModelPane } from './ModelPane'
import { useSyncStore } from '../../stores/syncStore'
import type { ModelInfo } from '../../types/model'
import './DualModelViewer.css'

interface DualModelViewerProps {
  leftModel: ModelInfo | null
  rightModel: ModelInfo | null
}

export function DualModelViewer({ leftModel, rightModel }: DualModelViewerProps) {
  const { activeSync } = useSyncStore()

  return (
    <div className={`dual-viewer h-full w-full ${activeSync ? 'dual-viewer--linked' : 'dual-viewer--unlinked'}`}>
      <div className="dual-viewer__pane dual-viewer__pane--left">
        <ModelPane model={leftModel} paneId="left" label="FP32" />
      </div>
      <div className="dual-viewer__divider" />
      <div className="dual-viewer__pane dual-viewer__pane--right">
        <ModelPane model={rightModel} paneId="right" label="INT8" />
      </div>
    </div>
  )
}
