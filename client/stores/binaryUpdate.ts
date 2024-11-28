import { defineStore } from 'pinia'
import type { PositionUpdate } from '../types/websocket'
import { POSITION_SCALE, VELOCITY_SCALE } from '../constants/websocket'

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

      // Enhanced debug logging for position updates
      console.debug('Processing position update:', {
        nodeId: pos.id,
        oldPosition: this.positions.get(pos.id),
        newPosition: {
          x: pos.x,
          y: pos.y,
          z: pos.z,
          vx: pos.vx,
          vy: pos.vy,
          vz: pos.vz
        },
        timestamp: new Date().toISOString()
      })

      // Store update with node ID - no scaling needed as values are already scaled
      this.positions.set(pos.id, {
        id: pos.id,
        x: pos.x,
        y: pos.y,
        z: pos.z,
        vx: pos.vx,
        vy: pos.vy,
        vz: pos.vz
      })
    },

    /**
     * Update positions from binary WebSocket message
     */
    updatePositions(positions: PositionUpdate[], isInitial: boolean) {
      // Enhanced debug logging for batch updates
      console.debug('Starting batch position update:', {
        updateCount: positions.length,
        isInitial,
        currentPositionsCount: this.positions.size,
        pendingUpdatesCount: this.pendingUpdates.length,
        timestamp: new Date().toISOString()
      })

      // Clear previous positions if this is initial layout
      if (isInitial) {
        console.debug('Clearing previous positions for initial layout')
        this.positions.clear()
        this.pendingUpdates = []
      }

      // Process any pending updates first
      if (this.pendingUpdates.length > 0) {
        console.debug(`Processing ${this.pendingUpdates.length} pending updates`)
        this.pendingUpdates.forEach(pos => this.processUpdate(pos))
        this.pendingUpdates = []
      }

      // Process new updates
      positions.forEach(pos => this.processUpdate(pos))

      // Update state
      this.lastUpdateTime = Date.now()
      this.isInitialLayout = isInitial

      // Enhanced debug logging for update completion
      console.debug('Batch position update completed:', {
        finalPositionsCount: this.positions.size,
        isInitial,
        samplePositions: Array.from(this.positions.entries())
          .slice(0, 3)
          .map(([id, pos]) => ({
            id,
            position: {
              x: pos.x,
              y: pos.y,
              z: pos.z
            },
            velocity: {
              vx: pos.vx,
              vy: pos.vy,
              vz: pos.vz
            }
          })),
        timestamp: new Date().toISOString()
      })
    },

    /**
     * Set batch size for processing updates
     */
    setBatchSize(size: number) {
      const oldSize = this.batchSize
      this.batchSize = Math.max(1, Math.min(1000, size)) // Clamp between 1-1000
      console.debug('Batch size updated:', {
        oldSize,
        newSize: this.batchSize,
        timestamp: new Date().toISOString()
      })
    },

    /**
     * Clear all position data
     * Called when receiving full mesh update or on cleanup
     */
    clear() {
      console.debug('Clearing binary update store:', {
        clearedPositions: this.positions.size,
        clearedPending: this.pendingUpdates.length,
        timestamp: new Date().toISOString()
      })
      this.positions.clear()
      this.pendingUpdates = []
      this.lastUpdateTime = 0
      this.isInitialLayout = false
    }
  }
})
