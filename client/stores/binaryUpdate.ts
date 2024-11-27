import { defineStore } from 'pinia'
import { usePerformanceMonitor } from './performanceMonitor'
import type { PositionUpdate } from '../types/websocket'

interface BinaryUpdateState {
  positions: Map<string, PositionUpdate>
  lastUpdateTime: number
  isInitialLayout: boolean
  timeStep: number
  pendingUpdates: PositionUpdate[]
  batchSize: number
}

/**
 * Store for handling transient binary position/velocity updates
 * These updates are frequent and superseded by full mesh updates
 * Includes performance optimization and batching
 */
export const useBinaryUpdateStore = defineStore('binaryUpdate', {
  state: (): BinaryUpdateState => ({
    positions: new Map(),
    lastUpdateTime: 0,
    isInitialLayout: false,
    timeStep: 0,
    pendingUpdates: [],
    batchSize: 100 // Default batch size
  }),

  getters: {
    /**
     * Get latest position update for a node
     */
    getNodePosition: (state) => (nodeId: string): PositionUpdate | undefined => {
      return state.positions.get(nodeId)
    },

    /**
     * Get all current position updates
     */
    getAllPositions: (state): PositionUpdate[] => {
      return Array.from(state.positions.values())
    },

    /**
     * Check if this is initial layout data
     */
    isInitial: (state): boolean => state.isInitialLayout,

    /**
     * Get current simulation timestep
     */
    getCurrentTimeStep: (state): number => state.timeStep,

    /**
     * Get number of pending updates
     */
    pendingUpdateCount: (state): number => state.pendingUpdates.length,

    /**
     * Get current batch size
     */
    getBatchSize: (state): number => state.batchSize
  },

  actions: {
    /**
     * Generate a unique ID for position updates that don't have one
     */
    generateUpdateId(index: number): string {
      return `pos_${this.lastUpdateTime}_${index}`
    },

    /**
     * Process a single position update
     */
    processUpdate(pos: PositionUpdate, index: number): void {
      const updateId = pos.id ?? this.generateUpdateId(index)
      const updatedPosition: PositionUpdate = {
        id: updateId,
        x: pos.x,
        y: pos.y,
        z: pos.z,
        vx: pos.vx,
        vy: pos.vy,
        vz: pos.vz
      }
      this.positions.set(updateId, updatedPosition)
    },

    /**
     * Update positions from binary WebSocket message
     * Uses performance monitoring and batched processing
     */
    updatePositions(positions: PositionUpdate[], isInitial: boolean, timeStep: number) {
      const performanceMonitor = usePerformanceMonitor()

      // Skip update if frame rate limit reached
      if (!performanceMonitor.shouldProcessFrame) {
        this.pendingUpdates.push(...positions)
        return
      }

      // Clear previous positions if this is initial layout
      if (isInitial) {
        this.positions.clear()
        this.pendingUpdates = []
      }

      // Process any pending updates first
      if (this.pendingUpdates.length > 0) {
        performanceMonitor.processBatch(
          this.pendingUpdates,
          (pos: PositionUpdate, index: number) => this.processUpdate(pos, index)
        )
        this.pendingUpdates = []
      }

      // Process new updates in batches
      performanceMonitor.processBatch(
        positions,
        (pos: PositionUpdate, index: number) => this.processUpdate(pos, index)
      )

      // Update state
      this.lastUpdateTime = Date.now()
      this.isInitialLayout = isInitial
      this.timeStep = timeStep

      // Update performance metrics
      performanceMonitor.setNodeCount(this.positions.size)
      performanceMonitor.trackFrame()

      // Log update in development
      if (process.env.NODE_ENV === 'development') {
        console.debug('Binary update processed:', {
          positions: positions.length,
          pending: this.pendingUpdates.length,
          total: this.positions.size,
          isInitial,
          timeStep,
          batchSize: this.batchSize
        })
      }
    },

    /**
     * Process pending updates
     * Called by animation frame when below frame rate limit
     */
    processPendingUpdates() {
      const performanceMonitor = usePerformanceMonitor()
      
      if (this.pendingUpdates.length > 0 && performanceMonitor.shouldProcessFrame) {
        const startTime = performance.now()

        performanceMonitor.processBatch(
          this.pendingUpdates,
          (pos: PositionUpdate, index: number) => this.processUpdate(pos, index)
        )
        this.pendingUpdates = []
        
        const endTime = performance.now()
        performanceMonitor.trackUpdate(endTime - startTime)
      }
    },

    /**
     * Set frame rate limit for updates
     */
    setFrameRateLimit(fps: number) {
      const performanceMonitor = usePerformanceMonitor()
      performanceMonitor.setFrameRateLimit(fps)
    },

    /**
     * Set batch size for processing updates
     */
    setBatchSize(size: number) {
      this.batchSize = Math.max(1, Math.min(1000, size)) // Clamp between 1-1000
      if (process.env.NODE_ENV === 'development') {
        console.debug(`Batch size set to ${this.batchSize}`)
      }
    },

    /**
     * Clear all position data and metrics
     * Called when receiving full mesh update or on cleanup
     */
    clear() {
      this.positions.clear()
      this.pendingUpdates = []
      this.lastUpdateTime = 0
      this.isInitialLayout = false
      this.timeStep = 0

      // Reset performance metrics
      const performanceMonitor = usePerformanceMonitor()
      performanceMonitor.reset()
    }
  }
})
