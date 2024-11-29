import { defineStore } from 'pinia'
import type { Position, PositionUpdate } from '../types/websocket'

interface BinaryUpdateState {
  // Use TypedArrays for better performance with binary data
  positions: Float32Array  // [x,y,z] for each node
  velocities: Float32Array // [vx,vy,vz] for each node
  nodeIds: string[]       // Map index to node ID
  idToIndex: Map<string, number>  // Map node ID to index
  nodeCount: number
  lastUpdateTime: number
  isInitialLayout: boolean
}

/**
 * Store for handling binary position/velocity updates
 * Optimized for high-frequency updates in force-directed graph
 */
export const useBinaryUpdateStore = defineStore('binaryUpdate', {
  state: (): BinaryUpdateState => ({
    positions: new Float32Array(0),
    velocities: new Float32Array(0),
    nodeIds: [],
    idToIndex: new Map(),
    nodeCount: 0,
    lastUpdateTime: 0,
    isInitialLayout: false
  }),

  getters: {
    /**
     * Get position for node by ID
     */
    getNodePosition: (state) => (id: string): [number, number, number] | undefined => {
      const index = state.idToIndex.get(id);
      if (index !== undefined) {
        const baseIndex = index * 3;
        return [
          state.positions[baseIndex],
          state.positions[baseIndex + 1],
          state.positions[baseIndex + 2]
        ];
      }
      return undefined;
    },

    /**
     * Get velocity for node by ID
     */
    getNodeVelocity: (state) => (id: string): [number, number, number] | undefined => {
      const index = state.idToIndex.get(id);
      if (index !== undefined) {
        const baseIndex = index * 3;
        return [
          state.velocities[baseIndex],
          state.velocities[baseIndex + 1],
          state.velocities[baseIndex + 2]
        ];
      }
      return undefined;
    },

    /**
     * Get all positions as Float32Array
     */
    getAllPositions: (state): Float32Array => state.positions,

    /**
     * Get all velocities as Float32Array
     */
    getAllVelocities: (state): Float32Array => state.velocities,

    /**
     * Check if this is initial layout data
     */
    isInitial: (state): boolean => state.isInitialLayout
  },

  actions: {
    /**
     * Update position for a single node
     */
    updatePosition(id: string, position: Position) {
      let index = this.idToIndex.get(id);
      
      // If node doesn't exist, add it
      if (index === undefined) {
        index = this.nodeCount;
        this.nodeCount++;
        
        // Resize arrays if needed
        const newPositions = new Float32Array(this.nodeCount * 3);
        const newVelocities = new Float32Array(this.nodeCount * 3);
        newPositions.set(this.positions);
        newVelocities.set(this.velocities);
        this.positions = newPositions;
        this.velocities = newVelocities;
        
        // Update mappings
        this.nodeIds.push(id);
        this.idToIndex.set(id, index);
      }

      // Update position and velocity
      const baseIndex = index * 3;
      this.positions[baseIndex] = position.x;
      this.positions[baseIndex + 1] = position.y;
      this.positions[baseIndex + 2] = position.z;
      this.velocities[baseIndex] = position.vx;
      this.velocities[baseIndex + 1] = position.vy;
      this.velocities[baseIndex + 2] = position.vz;

      this.lastUpdateTime = Date.now();
    },

    /**
     * Update multiple positions
     */
    updatePositions(positions: Position[], isInitial: boolean = false) {
      if (isInitial) {
        // Reset state for initial layout
        this.clear();
        
        // Pre-allocate arrays
        this.positions = new Float32Array(positions.length * 3);
        this.velocities = new Float32Array(positions.length * 3);
        this.nodeIds = new Array(positions.length);
        this.nodeCount = positions.length;
      }

      // Update all positions
      positions.forEach((pos) => {
        this.updatePosition(pos.id, pos);
      });

      this.isInitialLayout = isInitial;
      this.lastUpdateTime = Date.now();

      // Debug logging
      if (positions.length > 0) {
        console.debug('Positions updated:', {
          count: positions.length,
          isInitial,
          sample: {
            id: positions[0].id,
            position: [positions[0].x, positions[0].y, positions[0].z],
            velocity: [positions[0].vx, positions[0].vy, positions[0].vz]
          },
          timestamp: new Date().toISOString()
        });
      }
    },

    /**
     * Update from binary data
     */
    updateFromBinary(data: PositionUpdate) {
      this.updatePositions(data.positions, data.isInitialLayout);
    },

    /**
     * Clear all position data
     */
    clear() {
      this.positions = new Float32Array(0);
      this.velocities = new Float32Array(0);
      this.nodeIds = [];
      this.idToIndex.clear();
      this.nodeCount = 0;
      this.lastUpdateTime = 0;
      this.isInitialLayout = false;
    }
  }
})
