import { DualModelViewer } from '../DualModelViewer/DualModelViewer'
import { WeightChart } from '../WeightChart/WeightChart'
import { useModelStore } from '../../stores/modelStore'

export function MainContent() {
  const { leftModelId, rightModelId, getModelById, currentComparedNode } = useModelStore()

  const leftModel = getModelById(leftModelId)
  const rightModel = getModelById(rightModelId)

  const hasAnyModel = leftModel || rightModel

  return (
    <main className="flex-1 flex flex-col overflow-hidden relative">
      <div className="flex-1 relative">
        {hasAnyModel ? (
          <DualModelViewer leftModel={leftModel} rightModel={rightModel} />
        ) : (
          <div className="flex items-center justify-center h-full text-gray-500">
            <p>请从侧边栏选择或加载模型</p>
          </div>
        )}
      </div>
      {currentComparedNode && <WeightChart />}
    </main>
  )
}
