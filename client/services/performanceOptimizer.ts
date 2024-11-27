import { usePerformanceMonitor } from '../stores/performanceMonitor'
import { useBinaryUpdateStore } from '../stores/binaryUpdate'
import { storeToRefs } from 'pinia'

interface PerformanceThresholds {
  minFps: number
  maxUpdateTime: number
  maxMemoryUsage: number
  maxBatchSize: number
  minBatchSize: number
}

interface PerformanceRegression {
  metric: string
  previousValue: number
  currentValue: number
  threshold: number
  timestamp: number
}

interface MemorySnapshot {
  timestamp: number
  heapUsed: number
  nodeCount: number
  updateCount: number
}

/**
 * Service for optimizing performance and detecting issues
 * Handles:
 * - Automatic batch size optimization
 * - Memory leak detection
 * - Performance regression alerts
 * - Performance data export
 */
export class PerformanceOptimizer {
  private static instance: PerformanceOptimizer
  private performanceMonitor = usePerformanceMonitor()
  private binaryUpdateStore = useBinaryUpdateStore()
  private binaryStoreRefs = storeToRefs(useBinaryUpdateStore())
  
  private thresholds: PerformanceThresholds = {
    minFps: 30,
    maxUpdateTime: 16, // ms (targeting 60fps)
    maxMemoryUsage: 0.9, // 90% of available heap
    maxBatchSize: 1000,
    minBatchSize: 10
  }

  private memorySnapshots: MemorySnapshot[] = []
  private regressions: PerformanceRegression[] = []
  private optimizationInterval: number | null = null
  private snapshotInterval: number | null = null

  private constructor() {
    // Private constructor for singleton
  }

  public static getInstance(): PerformanceOptimizer {
    if (!PerformanceOptimizer.instance) {
      PerformanceOptimizer.instance = new PerformanceOptimizer()
    }
    return PerformanceOptimizer.instance
  }

  /**
   * Start performance optimization and monitoring
   */
  public start(): void {
    this.startOptimization()
    this.startMemoryMonitoring()
  }

  /**
   * Stop performance optimization and monitoring
   */
  public stop(): void {
    if (this.optimizationInterval !== null) {
      window.clearInterval(this.optimizationInterval)
      this.optimizationInterval = null
    }
    if (this.snapshotInterval !== null) {
      window.clearInterval(this.snapshotInterval)
      this.snapshotInterval = null
    }
  }

  /**
   * Start automatic batch size optimization
   */
  private startOptimization(): void {
    this.optimizationInterval = window.setInterval(() => {
      const fps = this.performanceMonitor.currentFPS
      const updateTime = this.performanceMonitor.averageUpdateTime
      const currentBatchSize = this.binaryStoreRefs.getBatchSize.value

      // Adjust batch size based on performance
      if (fps < this.thresholds.minFps || updateTime > this.thresholds.maxUpdateTime) {
        // Reduce batch size if performance is poor
        const newBatchSize = Math.max(
          this.thresholds.minBatchSize,
          Math.floor(currentBatchSize * 0.8)
        )
        this.binaryUpdateStore.setBatchSize(newBatchSize)
        
        console.debug('Reduced batch size due to performance:', {
          fps,
          updateTime,
          oldBatchSize: currentBatchSize,
          newBatchSize
        })
      } else if (fps > this.thresholds.minFps * 1.2 && updateTime < this.thresholds.maxUpdateTime * 0.8) {
        // Increase batch size if performance is good
        const newBatchSize = Math.min(
          this.thresholds.maxBatchSize,
          Math.ceil(currentBatchSize * 1.2)
        )
        this.binaryUpdateStore.setBatchSize(newBatchSize)
        
        console.debug('Increased batch size due to good performance:', {
          fps,
          updateTime,
          oldBatchSize: currentBatchSize,
          newBatchSize
        })
      }

      // Check for performance regressions
      this.checkPerformanceRegressions()
    }, 5000) // Check every 5 seconds
  }

  /**
   * Start memory leak detection
   */
  private startMemoryMonitoring(): void {
    this.snapshotInterval = window.setInterval(() => {
      const snapshot: MemorySnapshot = {
        timestamp: Date.now(),
        heapUsed: this.performanceMonitor.memoryStats.heapUsed,
        nodeCount: this.performanceMonitor.memoryStats.nodeCount,
        updateCount: this.performanceMonitor.memoryStats.updateCount
      }

      this.memorySnapshots.push(snapshot)

      // Keep last 60 snapshots (5 minutes with 5-second interval)
      if (this.memorySnapshots.length > 60) {
        this.memorySnapshots.shift()
      }

      // Check for memory leaks
      this.checkMemoryLeaks()
    }, 5000) // Take snapshot every 5 seconds
  }

  /**
   * Check for memory leaks by analyzing memory growth patterns
   */
  private checkMemoryLeaks(): void {
    if (this.memorySnapshots.length < 12) return // Need at least 1 minute of data

    // Calculate memory growth rate over last minute
    const recentSnapshots = this.memorySnapshots.slice(-12)
    const memoryGrowthRate = (recentSnapshots[11].heapUsed - recentSnapshots[0].heapUsed) / 
      (recentSnapshots[11].timestamp - recentSnapshots[0].timestamp)

    // Check if memory is growing consistently
    if (memoryGrowthRate > 1024 * 1024) { // More than 1MB/s growth
      const warning = {
        type: 'memory_leak',
        message: 'Possible memory leak detected',
        details: {
          growthRate: `${(memoryGrowthRate / (1024 * 1024)).toFixed(2)} MB/s`,
          currentHeap: `${(recentSnapshots[11].heapUsed / (1024 * 1024)).toFixed(2)} MB`,
          nodeCount: recentSnapshots[11].nodeCount
        }
      }

      console.warn(warning)
      this.dispatchWarning(warning)
    }
  }

  /**
   * Check for performance regressions
   */
  private checkPerformanceRegressions(): void {
    const currentMetrics = {
      fps: this.performanceMonitor.currentFPS,
      updateTime: this.performanceMonitor.averageUpdateTime,
      memoryUsed: this.performanceMonitor.memoryStats.heapUsed
    }

    // Compare with previous metrics
    if (this.regressions.length > 0) {
      const previousRegression = this.regressions[this.regressions.length - 1]
      const timeSinceLastRegression = Date.now() - previousRegression.timestamp

      // Only check if enough time has passed since last regression
      if (timeSinceLastRegression > 60000) { // 1 minute
        Object.entries(currentMetrics).forEach(([metric, value]) => {
          const previousValue = (previousRegression as any)[metric]
          if (previousValue) {
            const threshold = this.getThresholdForMetric(metric)
            const regression = this.checkRegression(metric, previousValue, value, threshold)
            if (regression) {
              this.regressions.push(regression)
              this.dispatchWarning({
                type: 'performance_regression',
                message: `Performance regression detected in ${metric}`,
                details: regression
              })
            }
          }
        })
      }
    }

    // Keep last 60 regressions
    if (this.regressions.length > 60) {
      this.regressions.shift()
    }
  }

  /**
   * Get threshold for specific metric
   */
  private getThresholdForMetric(metric: string): number {
    switch (metric) {
      case 'fps':
        return this.thresholds.minFps
      case 'updateTime':
        return this.thresholds.maxUpdateTime
      case 'memoryUsed':
        return this.thresholds.maxMemoryUsage
      default:
        return 0
    }
  }

  /**
   * Check for regression in specific metric
   */
  private checkRegression(
    metric: string,
    previousValue: number,
    currentValue: number,
    threshold: number
  ): PerformanceRegression | null {
    const regressionThreshold = 0.2 // 20% change
    const change = Math.abs(currentValue - previousValue) / previousValue

    if (change > regressionThreshold) {
      return {
        metric,
        previousValue,
        currentValue,
        threshold,
        timestamp: Date.now()
      }
    }

    return null
  }

  /**
   * Dispatch warning event
   */
  private dispatchWarning(warning: any): void {
    window.dispatchEvent(new CustomEvent('performance-warning', {
      detail: warning
    }))
  }

  /**
   * Export performance data for analysis
   */
  public exportData(): string {
    const data = {
      snapshots: this.memorySnapshots,
      regressions: this.regressions,
      currentMetrics: {
        fps: this.performanceMonitor.currentFPS,
        updateTime: this.performanceMonitor.averageUpdateTime,
        memory: this.performanceMonitor.memoryStats,
        batchSize: this.binaryStoreRefs.getBatchSize.value
      },
      thresholds: this.thresholds
    }

    return JSON.stringify(data, null, 2)
  }

  /**
   * Update performance thresholds
   */
  public updateThresholds(thresholds: Partial<PerformanceThresholds>): void {
    this.thresholds = {
      ...this.thresholds,
      ...thresholds
    }
  }
}

// Export singleton instance
export const performanceOptimizer = PerformanceOptimizer.getInstance()
