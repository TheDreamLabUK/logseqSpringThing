import { defineStore } from 'pinia'
import type { BinaryMessage } from '../types/websocket'

interface BinaryUpdateState {
  // Use TypedArrays for better performance with binary data
  positions: Float32Array  // [x,y,z] for each node
  velocities: Float32Array // [vx,vy,vz] for each node
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
    nodeCount: 0,
    lastUpdateTime: 0,
    isInitialLayout: false
  }),

  getters: {
    /**
     * Get position for node by index
     */
    getNodePosition: (state) => (index: number): [number, number, number] | undefined => {
      if (index >= 0 && index < state.nodeCount) {
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
     * Get velocity for node by index
     */
    getNodeVelocity: (state) => (index: number): [number, number, number] | undefined => {
      if (index >= 0 && index < state.nodeCount) {
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
    updateNodePosition(
      index: number,
      x: number, y: number, z: number,
      vx: number, vy: number, vz: number
    ): void {
      if (index >= 0 && index < this.nodeCount) {
        const posIndex = index * 3;
        const velIndex = index * 3;

        // Update position
        this.positions[posIndex] = x;
        this.positions[posIndex + 1] = y;
        this.positions[posIndex + 2] = z;

        // Update velocity
        this.velocities[velIndex] = vx;
        this.velocities[velIndex + 1] = vy;
        this.velocities[velIndex + 2] = vz;

        this.lastUpdateTime = Date.now();
      }
    },

    /**
     * Update from binary data
     */
    updateFromBinary(message: BinaryMessage): void {
      const dataView = new Float32Array(message.data);
      const totalFloats = dataView.length - 1; // Subtract 1 for isInitialLayout flag
      const nodeCount = totalFloats / 6; // 6 floats per node (x,y,z,vx,vy,vz)

      // Resize arrays if needed
      if (this.nodeCount !== nodeCount) {
        this.positions = new Float32Array(nodeCount * 3);
        this.velocities = new Float32Array(nodeCount * 3);
        this.nodeCount = nodeCount;
      }

      // Process position and velocity data directly from binary
      let srcOffset = 1; // Skip isInitialLayout flag
      for (let i = 0; i < nodeCount; i++) {
        const posOffset = i * 3;
        const velOffset = i * 3;

        // Copy positions
        this.positions[posOffset] = dataView[srcOffset];
        this.positions[posOffset + 1] = dataView[srcOffset + 1];
        this.positions[posOffset + 2] = dataView[srcOffset + 2];

        // Copy velocities
        this.velocities[velOffset] = dataView[srcOffset + 3];
        this.velocities[velOffset + 1] = dataView[srcOffset + 4];
        this.velocities[velOffset + 2] = dataView[srcOffset + 5];

        srcOffset += 6;
      }

      this.isInitialLayout = message.isInitialLayout;
      this.lastUpdateTime = Date.now();

      // Debug logging
      if (nodeCount > 0) {
        console.debug('Binary update processed:', {
          nodeCount,
          isInitial: this.isInitialLayout,
          sample: {
            position: [
              this.positions[0],
              this.positions[1],
              this.positions[2]
            ],
            velocity: [
              this.velocities[0],
              this.velocities[1],
              this.velocities[2]
            ]
          },
          timestamp: new Date().toISOString()
        });
      }
    },

    /**
     * Clear all position data
     */
    clear(): void {
      this.positions = new Float32Array(0);
      this.velocities = new Float32Array(0);
      this.nodeCount = 0;
      this.lastUpdateTime = 0;
      this.isInitialLayout = false;
    }
  }
})
