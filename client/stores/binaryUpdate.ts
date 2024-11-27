import { defineStore } from 'pinia'
import type { PositionUpdate } from '../types/websocket'

interface BinaryUpdateState {
  positions: Map<string, PositionUpdate>
  lastUpdateTime: number
  isInitialLayout: boolean
  pendingUpdates: PositionUpdate[]
  batchSize: number
}

/**
 * Store for handling transient binary position/velocity updates
 * These updates are frequent and superseded by full mesh updates
 */
export const useBinaryUpdateStore = defineStore('binaryUpdate', {
  state: (): BinaryUpdateState => ({
    positions: new Map(),
    lastUpdateTime: 0,
    isInitialLayout: false,
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
     * Process a single position update
     */
    processUpdate(pos: PositionUpdate): void {
      if (!pos.id) {
        console.warn('Received position update without node ID')
        return
      }

      // Store update with node ID
      this.positions.set(pos.id, {
        id: pos.id,
        x: pos.x,
        y: pos.y,
        z: pos.z,
        vx: pos.vx,
        vy: pos.vy,
        vz: pos.vz
      })

      if (process.env.NODE_ENV === 'development') {
        console.debug('Position update:', {
          id: pos.id,
          position: [pos.x, pos.y, pos.z],
          velocity: [pos.vx, pos.vy, pos.vz]
        })
      }
    },

    /**
     * Update positions from binary WebSocket message
     */
    updatePositions(positions: PositionUpdate[], isInitial: boolean) {
      // Clear previous positions if this is initial layout
      if (isInitial) {
        this.positions.clear()
        this.pendingUpdates = []
      }

      // Process any pending updates first
      if (this.pendingUpdates.length > 0) {
        this.pendingUpdates.forEach(pos => this.processUpdate(pos))
        this.pendingUpdates = []
      }

      // Process new updates
      positions.forEach(pos => this.processUpdate(pos))

      // Update state
      this.lastUpdateTime = Date.now()
      this.isInitialLayout = isInitial

      // Log update in development
      if (process.env.NODE_ENV === 'development') {
        console.debug('Binary update processed:', {
          positions: positions.length,
          total: this.positions.size,
          isInitial,
          sample: positions.slice(0, 3).map(p => ({ id: p.id, pos: [p.x, p.y, p.z] }))
        })
      }
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
     * Clear all position data
     * Called when receiving full mesh update or on cleanup
     */
    clear() {
      this.positions.clear()
      this.pendingUpdates = []
      this.lastUpdateTime = 0
      this.isInitialLayout = false
    }
  }
})
