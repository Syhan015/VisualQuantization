import { useEffect, useRef, useState } from 'react'
import * as echarts from 'echarts'
import { useModelStore } from '../../stores/modelStore'
import './WeightChart.css'

export function WeightChart() {
  const { weightAnalysis, currentComparedNode, setCurrentComparedNode } = useModelStore()
  const chartRef = useRef<HTMLDivElement>(null)
  const chartInstanceRef = useRef<echarts.ECharts | null>(null)
  const [isExpanded, setIsExpanded] = useState(false)

  useEffect(() => {
    if (!chartRef.current || !weightAnalysis) return

    if (!chartInstanceRef.current) {
      chartInstanceRef.current = echarts.init(chartRef.current)
    }

    const bins = weightAnalysis.distributionA.length
    // Use bin_edges from backend to show actual weight value ranges
    // bin_edges has N+1 values for N bins - generate range labels like [-0.12, -0.08)
    let xAxisData: string[]
    if (weightAnalysis.binEdges && weightAnalysis.binEdges.length > 0) {
      const edges = weightAnalysis.binEdges
      xAxisData = []
      for (let i = 0; i < edges.length - 1; i++) {
        const low = edges[i]
        const high = edges[i + 1]
        const lowStr = low.toFixed(3)
        const highStr = high.toFixed(3)
        xAxisData.push(`[${lowStr}, ${highStr})`)
      }
    } else {
      xAxisData = Array.from({ length: bins }, (_, i) => i.toString())
    }

    const option: echarts.EChartsOption = {
      title: {
        text: `权重分布: ${weightAnalysis.nodeName}`,
        textStyle: { color: '#fff', fontSize: isExpanded ? 22 : 14 },
        left: 'center',
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        backgroundColor: 'rgba(50, 50, 50, 0.95)',
        textStyle: { color: '#fff', fontSize: isExpanded ? 16 : 12 },
        position: 'bottom',
        confine: true,
      },
      legend: {
        data: ['FP32 (原模型)', 'INT8 (量化后)'],
        textStyle: { color: '#fff', fontSize: isExpanded ? 16 : 12 },
        top: 30,
      },
      xAxis: {
        type: 'category',
        data: xAxisData,
        axisLabel: {
          color: '#fff',
          fontSize: isExpanded ? 14 : 9,
          rotate: 45,
          interval: 4,
          formatter: (value: string) => value
        },
        name: '权重值',
        nameTextStyle: { color: '#888', fontSize: isExpanded ? 16 : 12 },
      },
      yAxis: {
        type: 'value',
        axisLabel: { color: '#fff', fontSize: isExpanded ? 14 : 9 },
        name: '概率',
        nameTextStyle: { color: '#888', fontSize: isExpanded ? 16 : 12 },
      },
      series: [
        {
          name: 'FP32 (原模型)',
          type: 'bar',
          data: weightAnalysis.distributionA,
          itemStyle: { color: 'rgba(59, 130, 246, 0.6)' },
        },
        {
          name: 'INT8 (量化后)',
          type: 'bar',
          data: weightAnalysis.distributionB,
          itemStyle: { color: 'rgba(168, 85, 247, 0.6)' },
        },
      ],
      backgroundColor: 'transparent',
      grid: {
        left: 50,
        right: 20,
        top: 70,
        bottom: 80,
      },
    }

    chartInstanceRef.current.setOption(option)

    const handleResize = () => chartInstanceRef.current?.resize()
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
    }
  }, [weightAnalysis, isExpanded])

  useEffect(() => {
    return () => {
      chartInstanceRef.current?.dispose()
      chartInstanceRef.current = null
    }
  }, [])

  if (!currentComparedNode || !weightAnalysis) {
    return null
  }

  return (
    <div
      className="weight-chart-panel absolute bg-gray-900 bg-opacity-95 rounded-lg shadow-2xl border border-gray-700 overflow-hidden transition-all duration-300 z-50"
      style={
        isExpanded
          ? { top: '64px', left: '256px', right: '0', bottom: '0', overflow: 'hidden' }
          : { bottom: '16px', right: '16px', width: '384px' }
      }
    >
      <div className="flex items-center justify-between p-3 border-b border-gray-700">
        <h3 className={`font-medium text-white ${isExpanded ? 'text-lg' : 'text-sm'}`}>
          {isExpanded ? '权重对比分析 (放大模式)' : '权重对比分析'}
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={() => {
              setIsExpanded(!isExpanded)
              setTimeout(() => chartInstanceRef.current?.resize(), 350)
            }}
            className="text-gray-400 hover:text-white text-lg leading-none p-1"
            title={isExpanded ? '缩小' : '放大'}
          >
            {isExpanded ? '⊡' : '□'}
          </button>
          <button
            onClick={() => setCurrentComparedNode(null)}
            className="text-gray-400 hover:text-white text-lg leading-none"
          >
            ×
          </button>
        </div>
      </div>

      <div className="p-3 space-y-2 border-b border-gray-700">
        <div className={`grid gap-2 text-center ${isExpanded ? 'grid-cols-5' : 'grid-cols-5'}`}>
          <div className="bg-gray-800 rounded p-1.5">
            <div className={`${isExpanded ? 'text-sm' : 'text-xs'} text-gray-300`}>Cosine</div>
            <div className={`font-mono text-blue-400 ${isExpanded ? 'text-base' : 'text-sm'}`}>{weightAnalysis.cosineSimilarity.toFixed(4)}</div>
          </div>
          <div className="bg-gray-800 rounded p-1.5">
            <div className={`${isExpanded ? 'text-sm' : 'text-xs'} text-gray-300`}>L2</div>
            <div className={`font-mono text-green-400 ${isExpanded ? 'text-base' : 'text-sm'}`}>{weightAnalysis.l2Error.toFixed(4)}</div>
          </div>
          <div className="bg-gray-800 rounded p-1.5">
            <div className={`${isExpanded ? 'text-sm' : 'text-xs'} text-gray-300`}>MAE</div>
            <div className={`font-mono text-yellow-400 ${isExpanded ? 'text-base' : 'text-sm'}`}>{weightAnalysis.mae.toFixed(4)}</div>
          </div>
          <div className="bg-gray-800 rounded p-1.5">
            <div className={`${isExpanded ? 'text-sm' : 'text-xs'} text-gray-300`}>Mean</div>
            <div className={`font-mono text-purple-400 ${isExpanded ? 'text-base' : 'text-sm'}`}>{weightAnalysis.meanDiff.toFixed(6)}</div>
          </div>
          <div className="bg-gray-800 rounded p-1.5">
            <div className={`${isExpanded ? 'text-sm' : 'text-xs'} text-gray-300`}>Std</div>
            <div className={`font-mono text-pink-400 ${isExpanded ? 'text-base' : 'text-sm'}`}>{weightAnalysis.stdDiff.toFixed(6)}</div>
          </div>
        </div>
      </div>

      <div ref={chartRef} className="w-full" style={{ height: isExpanded ? 'calc(100vh - 64px - 120px)' : '240px' }} />
    </div>
  )
}