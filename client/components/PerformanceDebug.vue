<template>
  <!-- Template remains unchanged -->
  <div v-if="showDebug" class="performance-debug">
    <div class="metrics-panel">
      <h3>Performance Metrics</h3>
      <div class="metric">
        <span>FPS:</span>
        <span :class="{ warning: fps < 30, critical: fps < 15 }">
          {{ fps.toFixed(1) }}
        </span>
      </div>
      <div class="metric">
        <span>Updates/s:</span>
        <span :class="{ warning: updateRate > 100, critical: updateRate > 200 }">
          {{ updateRate.toFixed(1) }}
        </span>
      </div>
      <div class="metric">
        <span>Avg Update Time:</span>
        <span :class="{ warning: avgUpdateTime > 16, critical: avgUpdateTime > 32 }">
          {{ avgUpdateTime.toFixed(2) }}ms
        </span>
      </div>
      <div class="metric">
        <span>Node Count:</span>
        <span>{{ nodeCount }}</span>
      </div>
      <div class="metric">
        <span>Pending Updates:</span>
        <span :class="{ warning: pendingUpdates > 1000, critical: pendingUpdates > 5000 }">
          {{ pendingUpdates }}
        </span>
      </div>
      <div class="metric">
        <span>Batch Size:</span>
        <span>{{ batchSize }}</span>
      </div>
      <div v-if="hasMemoryAPI" class="memory-section">
        <h4>Memory Usage</h4>
        <div class="memory-bar">
          <div 
            class="memory-used"
            :style="{ width: memoryUsagePercent + '%' }"
            :class="{ warning: memoryUsagePercent > 70, critical: memoryUsagePercent > 90 }"
          >
            {{ formatMemory(memoryUsed) }}
          </div>
        </div>
        <div class="memory-labels">
          <span>Used: {{ formatMemory(memoryUsed) }}</span>
          <span>Total: {{ formatMemory(memoryTotal) }}</span>
        </div>
      </div>
    </div>

    <!-- Performance Warnings -->
    <div v-if="warnings.length > 0" class="warnings-panel">
      <h4>Performance Warnings</h4>
      <div v-for="warning in warnings" :key="warning.timestamp" class="warning">
        <div class="warning-header">
          <span class="warning-type">{{ warning.type }}</span>
          <span class="warning-time">{{ formatTime(warning.timestamp) }}</span>
        </div>
        <div class="warning-message">{{ warning.message }}</div>
        <div v-if="warning.details" class="warning-details">
          <pre>{{ JSON.stringify(warning.details, null, 2) }}</pre>
        </div>
      </div>
    </div>

    <div class="chart-container" ref="chartContainer">
      <!-- Performance charts will be rendered here -->
    </div>

    <div class="controls">
      <div class="control-group">
        <label>Frame Rate Limit:</label>
        <input 
          type="range" 
          :min="1" 
          :max="144" 
          v-model.number="frameRateLimit"
          @input="updateFrameRateLimit"
        >
        <span>{{ frameRateLimit }}fps</span>
      </div>
      <div class="control-group">
        <label>Batch Size:</label>
        <input 
          type="range" 
          :min="1" 
          :max="1000" 
          v-model.number="batchSize"
          @input="updateBatchSize"
        >
        <span>{{ batchSize }}</span>
      </div>
      <div class="control-group">
        <button @click="exportData" class="export-button">
          Export Performance Data
        </button>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, computed, onMounted, onBeforeUnmount } from '@vue/runtime-core'
import { storeToRefs } from 'pinia'
import { usePerformanceMonitor } from '../stores/performanceMonitor'
import { useBinaryUpdateStore } from '../stores/binaryUpdate'
import { performanceOptimizer } from '../services/performanceOptimizer'
import type { ExtendedPerformance } from '../types/performance'
import type { 
  PerformanceChart, 
  ChartDataset, 
  ChartDataPoint,
  ChartOptions 
} from '../types/chart'
import {
  Chart,
  type ChartData,
  type ChartConfiguration,
  type ChartDataset as ChartJSDataset,
  type ChartTypeRegistry,
  type ScriptableContext
} from 'chart.js/auto'

interface Warning {
  type: string;
  message: string;
  details?: any;
  timestamp: number;
}

interface ChartPoint {
  x: number;
  y: number;
}

interface LineChartData extends ChartData {
  labels: number[];
  datasets: Array<{
    label: string;
    data: ChartPoint[];
    borderColor: string;
    tension: number;
  }>;
}

export default defineComponent({
  name: 'PerformanceDebug',
  
  props: {
    showDebug: {
      type: Boolean,
      default: false
    }
  },

  setup() {
    const performanceMonitor = usePerformanceMonitor()
    const binaryUpdateStore = useBinaryUpdateStore()
    const { getBatchSize } = storeToRefs(binaryUpdateStore)
    
    const chartContainer = ref<HTMLElement | null>(null)
    const chart = ref<Chart | null>(null)
    const frameRateLimit = ref(60)
    const warnings = ref<Warning[]>([])

    // Performance metrics
    const fps = computed(() => performanceMonitor.currentFPS)
    const updateRate = computed(() => performanceMonitor.memoryStats.updateCount)
    const avgUpdateTime = computed(() => performanceMonitor.averageUpdateTime)
    const nodeCount = computed(() => performanceMonitor.memoryStats.nodeCount)
    const pendingUpdates = computed(() => binaryUpdateStore.pendingUpdateCount)
    const batchSize = computed({
      get: () => getBatchSize.value,
      set: (value: number) => binaryUpdateStore.setBatchSize(value)
    })

    // Memory metrics
    const hasMemoryAPI = computed(() => performanceMonitor.isFullySupported)
    const memoryUsed = computed(() => performanceMonitor.memoryStats.heapUsed)
    const memoryTotal = computed(() => performanceMonitor.memoryStats.heapTotal)
    const memoryUsagePercent = computed(() => 
      (memoryUsed.value / memoryTotal.value) * 100 || 0
    )

    // Format memory size
    const formatMemory = (bytes: number): string => {
      const mb = bytes / 1024 / 1024
      return `${mb.toFixed(1)} MB`
    }

    // Format timestamp
    const formatTime = (timestamp: number): string => {
      return new Date(timestamp).toLocaleTimeString()
    }

    // Update settings
    const updateFrameRateLimit = () => {
      binaryUpdateStore.setFrameRateLimit(frameRateLimit.value)
    }

    const updateBatchSize = () => {
      binaryUpdateStore.setBatchSize(batchSize.value)
    }

    // Handle performance warnings
    const handleWarning = (event: CustomEvent) => {
      const warning = {
        ...event.detail,
        timestamp: Date.now()
      }
      warnings.value.push(warning)
      // Keep last 10 warnings
      if (warnings.value.length > 10) {
        warnings.value.shift()
      }
    }

    // Export performance data
    const exportData = () => {
      const data = performanceOptimizer.exportData()
      const blob = new Blob([data], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `performance-data-${Date.now()}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    }

    // Initialize performance chart
    const initChart = async () => {
      if (!chartContainer.value) return

      const ctx = document.createElement('canvas')
      chartContainer.value.appendChild(ctx)

      const config: ChartConfiguration = {
        type: 'line',
        data: {
          labels: [],
          datasets: [
            {
              label: 'FPS',
              data: [],
              borderColor: 'rgb(75, 192, 192)',
              tension: 0.1
            },
            {
              label: 'Update Time (ms)',
              data: [],
              borderColor: 'rgb(255, 99, 132)',
              tension: 0.1
            }
          ]
        },
        options: {
          responsive: true,
          animation: false,
          scales: {
            x: {
              type: 'time',
              time: {
                unit: 'second'
              }
            },
            y: {
              beginAtZero: true
            }
          }
        }
      }

      chart.value = new Chart(ctx, config)
    }

    // Update chart data
    const updateChart = () => {
      if (!chart.value) return

      const now = Date.now()
      const chartData = chart.value.data as LineChartData
      
      chartData.labels.push(now)

      chartData.datasets[0].data.push({
        x: now,
        y: fps.value
      })
      chartData.datasets[1].data.push({
        x: now,
        y: avgUpdateTime.value
      })

      // Keep last 60 seconds of data
      const cutoff = now - 60000
      chartData.labels = chartData.labels.filter(t => t >= cutoff)
      chartData.datasets.forEach(dataset => {
        dataset.data = dataset.data.filter(d => d.x >= cutoff)
      })

      chart.value.update()
    }

    // Animation frame for chart updates
    let animationFrame: number | null = null
    const updateLoop = () => {
      updateChart()
      animationFrame = requestAnimationFrame(updateLoop)
    }

    onMounted(async () => {
      await initChart()
      updateLoop()
      performanceOptimizer.start()
      window.addEventListener('performance-warning', handleWarning as EventListener)
    })

    onBeforeUnmount(() => {
      if (animationFrame !== null) {
        cancelAnimationFrame(animationFrame)
      }
      if (chart.value) {
        chart.value.destroy()
      }
      performanceOptimizer.stop()
      window.removeEventListener('performance-warning', handleWarning as EventListener)
    })

    return {
      chartContainer,
      fps,
      updateRate,
      avgUpdateTime,
      nodeCount,
      pendingUpdates,
      hasMemoryAPI,
      memoryUsed,
      memoryTotal,
      memoryUsagePercent,
      frameRateLimit,
      batchSize,
      warnings,
      formatMemory,
      formatTime,
      updateFrameRateLimit,
      updateBatchSize,
      exportData
    }
  }
})
</script>

<style scoped>
.performance-debug {
  position: fixed;
  top: 10px;
  left: 10px;
  background: rgba(0, 0, 0, 0.8);
  color: white;
  padding: 10px;
  border-radius: 4px;
  font-family: monospace;
  z-index: 1000;
  max-width: 300px;
  pointer-events: auto;
}

.metrics-panel {
  margin-bottom: 10px;
}

.metrics-panel h3 {
  margin: 0 0 10px 0;
  font-size: 14px;
}

.metric {
  display: flex;
  justify-content: space-between;
  margin: 5px 0;
  font-size: 12px;
}

.warning {
  color: #ffaa00;
}

.critical {
  color: #ff4444;
}

.memory-section {
  margin-top: 10px;
}

.memory-section h4 {
  margin: 0 0 5px 0;
  font-size: 12px;
}

.memory-bar {
  width: 100%;
  height: 20px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 2px;
  overflow: hidden;
}

.memory-used {
  height: 100%;
  background: #44ff44;
  transition: width 0.3s ease;
  text-align: right;
  padding-right: 5px;
  font-size: 10px;
  line-height: 20px;
}

.memory-labels {
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  margin-top: 2px;
}

.warnings-panel {
  margin: 10px 0;
  padding: 10px;
  background: rgba(255, 0, 0, 0.1);
  border-radius: 4px;
}

.warnings-panel h4 {
  margin: 0 0 10px 0;
  font-size: 12px;
}

.warning {
  margin-bottom: 10px;
  padding: 5px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 2px;
}

.warning-header {
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  margin-bottom: 5px;
}

.warning-type {
  color: #ffaa00;
  font-weight: bold;
}

.warning-time {
  color: #999;
}

.warning-message {
  font-size: 11px;
  margin-bottom: 5px;
}

.warning-details {
  font-size: 10px;
  color: #999;
}

.warning-details pre {
  margin: 0;
  white-space: pre-wrap;
}

.chart-container {
  width: 100%;
  height: 150px;
  margin: 10px 0;
}

.controls {
  margin-top: 10px;
}

.control-group {
  display: flex;
  align-items: center;
  margin: 5px 0;
  font-size: 12px;
}

.control-group label {
  flex: 0 0 100px;
}

.control-group input {
  flex: 1;
  margin: 0 10px;
}

.control-group span {
  flex: 0 0 50px;
  text-align: right;
}

.export-button {
  width: 100%;
  padding: 5px;
  background: #444;
  border: none;
  border-radius: 2px;
  color: white;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.2s;
}

.export-button:hover {
  background: #666;
}
</style>
