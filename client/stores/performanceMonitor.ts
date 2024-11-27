import { defineStore } from 'pinia'
import type { PerformanceMetrics } from '../types/core'
import type { ExtendedPerformance, MemoryInfo } from '../types/performance'

interface PerformanceState {
  metrics: PerformanceMetrics
  frameCount: number
  lastFrameTime: number
  frameRateLimit: number
  batchSize: number
  memoryUsage: {
    heapUsed: number
    heapTotal: number
    nodeCount: number
    updateCount: number
  }
  updateTimes: number[]
}

/**
 * Check if browser supports performance.memory
 */
const hasMemoryAPI = (): boolean => {
  return !!(window.performance && (window.performance as ExtendedPerformance).memory)
}

/**
 * Safely get memory info
 */
const getMemoryInfo = (): MemoryInfo | null => {
  if (!hasMemoryAPI()) return null
  return (window.performance as ExtendedPerformance).memory || null
}

/**
 * Type for batch processor function
 */
export type BatchProcessor<T> = (item: T, index: number, array: T[]) => void;

/**
 * Store for monitoring and optimizing performance
 */
export const usePerformanceMonitor = defineStore('performanceMonitor', {
  state: (): PerformanceState => ({
    metrics: {
      fps: 0,
      drawCalls: 0,
      triangles: 0,
      points: 0
    },
    frameCount: 0,
    lastFrameTime: performance.now(),
    frameRateLimit: 60,
    batchSize: 100,
    memoryUsage: {
      heapUsed: 0,
      heapTotal: 0,
      nodeCount: 0,
      updateCount: 0
    },
    updateTimes: []
  }),

  getters: {
    averageUpdateTime: (state): number => {
      if (state.updateTimes.length === 0) return 0
      const sum = state.updateTimes.reduce((a, b) => a + b, 0)
      return sum / state.updateTimes.length
    },

    currentFPS: (state): number => state.metrics.fps,

    shouldProcessFrame: (state): boolean => {
      const now = performance.now()
      const timeSinceLastFrame = now - state.lastFrameTime
      return timeSinceLastFrame >= (1000 / state.frameRateLimit)
    },

    memoryStats: (state): typeof state.memoryUsage => state.memoryUsage,

    isFullySupported(): boolean {
      return hasMemoryAPI()
    }
  },

  actions: {
    updateMetrics(metrics: Partial<PerformanceMetrics>) {
      this.metrics = { ...this.metrics, ...metrics }
    },

    trackFrame() {
      const now = performance.now()
      const delta = now - this.lastFrameTime

      this.frameCount++
      if (delta >= 1000) {
        this.metrics.fps = (this.frameCount * 1000) / delta
        this.frameCount = 0
        this.lastFrameTime = now

        const memoryInfo = getMemoryInfo()
        if (memoryInfo) {
          this.memoryUsage.heapUsed = memoryInfo.usedJSHeapSize
          this.memoryUsage.heapTotal = memoryInfo.totalJSHeapSize
        }

        if (process.env.NODE_ENV === 'development') {
          console.debug('Performance metrics:', {
            fps: this.metrics.fps.toFixed(1),
            memory: memoryInfo ? {
              used: (memoryInfo.usedJSHeapSize / 1024 / 1024).toFixed(1) + 'MB',
              total: (memoryInfo.totalJSHeapSize / 1024 / 1024).toFixed(1) + 'MB'
            } : 'Not available',
            updates: {
              count: this.memoryUsage.updateCount,
              avgTime: this.averageUpdateTime.toFixed(2) + 'ms'
            },
            nodes: this.memoryUsage.nodeCount
          })
        }
      }
    },

    trackUpdate(updateTime: number) {
      this.updateTimes.push(updateTime)
      if (this.updateTimes.length > 60) {
        this.updateTimes.shift()
      }
      this.memoryUsage.updateCount++

      if (process.env.NODE_ENV === 'development' && updateTime > 16.67) {
        console.warn(`Slow update detected: ${updateTime.toFixed(2)}ms`)
      }
    },

    setNodeCount(count: number) {
      this.memoryUsage.nodeCount = count
    },

    setFrameRateLimit(fps: number) {
      this.frameRateLimit = Math.max(1, Math.min(144, fps))
      if (process.env.NODE_ENV === 'development') {
        console.debug(`Frame rate limit set to ${this.frameRateLimit}fps`)
      }
    },

    setBatchSize(size: number) {
      this.batchSize = Math.max(1, Math.min(1000, size))
      if (process.env.NODE_ENV === 'development') {
        console.debug(`Batch size set to ${this.batchSize}`)
      }
    },

    processBatch<T>(items: T[], processor: BatchProcessor<T>): void {
      const startTime = performance.now()

      for (let i = 0; i < items.length; i += this.batchSize) {
        const batch = items.slice(i, i + this.batchSize)
        batch.forEach((item, index, array) => processor(item, index, array))

        if (i + this.batchSize < items.length) {
          requestAnimationFrame(() => {
            this.processBatch(
              items.slice(i + this.batchSize),
              processor
            )
          })
          break
        }
      }

      const endTime = performance.now()
      this.trackUpdate(endTime - startTime)
    },

    reset() {
      this.metrics = {
        fps: 0,
        drawCalls: 0,
        triangles: 0,
        points: 0
      }
      this.frameCount = 0
      this.lastFrameTime = performance.now()
      this.updateTimes = []
      this.memoryUsage = {
        heapUsed: 0,
        heapTotal: 0,
        nodeCount: 0,
        updateCount: 0
      }
    }
  }
})
