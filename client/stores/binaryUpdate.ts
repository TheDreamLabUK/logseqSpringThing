import { defineStore } from 'pinia'
import type { BinaryMessage } from '../types/websocket'
import {
  BINARY_UPDATE_HEADER_SIZE,
  BINARY_UPDATE_NODE_SIZE,
  FLOAT32_SIZE,
  MAX_VALID_POSITION,
  MIN_VALID_POSITION,
  MAX_VALID_VELOCITY,
  MIN_VALID_VELOCITY,
  ENABLE_BINARY_DEBUG,
  ENABLE_POSITION_VALIDATION
} from '../constants/websocket'

interface BinaryUpdateState {
  // Use TypedArrays for better performance with binary data
  positions: Float32Array  // [x,y,z] for each node
  velocities: Float32Array // [vx,vy,vz] for each node
  nodeCount: number
  firstUpdateTime: number  // Track when updates started
  lastUpdateTime: number   // Track most recent update
  isInitialLayout: boolean
  invalidUpdates: number   // Track number of invalid updates for monitoring
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
    firstUpdateTime: 0,
    lastUpdateTime: 0,
    isInitialLayout: false,
    invalidUpdates: 0
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
    isInitial: (state): boolean => state.isInitialLayout,

    /**
     * Get percentage of invalid updates
     */
    invalidUpdateRate: (state): number => {
      if (state.firstUpdateTime === 0) return 0;
      const timeSpan = (state.lastUpdateTime - state.firstUpdateTime) / 1000; // seconds
      const totalUpdates = Math.max(1, state.nodeCount * timeSpan);
      return (state.invalidUpdates / totalUpdates) * 100;
    },

    /**
     * Get update frequency in updates per second
     */
    updateFrequency: (state): number => {
      if (state.firstUpdateTime === 0) return 0;
      const timeSpan = (state.lastUpdateTime - state.firstUpdateTime) / 1000; // seconds
      return timeSpan > 0 ? state.nodeCount / timeSpan : 0;
    }
  },

  actions: {
    /**
     * Internal: Validate a position value
     */
    _validatePosition(value: number): boolean {
      return value >= MIN_VALID_POSITION && value <= MAX_VALID_POSITION;
    },

    /**
     * Internal: Validate a velocity value
     */
    _validateVelocity(value: number): boolean {
      return value >= MIN_VALID_VELOCITY && value <= MAX_VALID_VELOCITY;
    },

    /**
     * Internal: Clamp a position value to valid range
     */
    _clampPosition(value: number): number {
      return Math.max(MIN_VALID_POSITION, Math.min(MAX_VALID_POSITION, value));
    },

    /**
     * Internal: Clamp a velocity value to valid range
     */
    _clampVelocity(value: number): number {
      return Math.max(MIN_VALID_VELOCITY, Math.min(MAX_VALID_VELOCITY, value));
    },

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

        if (ENABLE_POSITION_VALIDATION) {
          // Validate position values
          const positionsValid = [x, y, z].every(v => this._validatePosition(v));
          const velocitiesValid = [vx, vy, vz].every(v => this._validateVelocity(v));

          if (!positionsValid || !velocitiesValid) {
            this.invalidUpdates++;
            console.warn('Invalid position/velocity values detected:', {
              index,
              position: [x, y, z],
              velocity: [vx, vy, vz]
            });

            // Clamp values to valid ranges
            x = this._clampPosition(x);
            y = this._clampPosition(y);
            z = this._clampPosition(z);
            vx = this._clampVelocity(vx);
            vy = this._clampVelocity(vy);
            vz = this._clampVelocity(vz);
          }
        }

        // Update position
        this.positions[posIndex] = x;
        this.positions[posIndex + 1] = y;
        this.positions[posIndex + 2] = z;

        // Update velocity
        this.velocities[velIndex] = vx;
        this.velocities[velIndex + 1] = vy;
        this.velocities[velIndex + 2] = vz;

        // Update timing
        const now = Date.now();
        if (this.firstUpdateTime === 0) {
          this.firstUpdateTime = now;
        }
        this.lastUpdateTime = now;
      }
    },

    /**
     * Update from binary data
     */
    updateFromBinary(message: BinaryMessage): void {
      const dataView = new Float32Array(message.data);
      const totalFloats = dataView.length - 1; // Subtract 1 for isInitialLayout flag
      const nodeCount = totalFloats / 6; // 6 floats per node (x,y,z,vx,vy,vz)

      // Validate buffer size
      const expectedSize = BINARY_UPDATE_HEADER_SIZE + (nodeCount * BINARY_UPDATE_NODE_SIZE);
      if (message.data.byteLength !== expectedSize) {
        console.error('Invalid binary message size:', {
          received: message.data.byteLength,
          expected: expectedSize,
          nodeCount,
          timestamp: new Date().toISOString()
        });
        return;
      }

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

        let x = dataView[srcOffset];
        let y = dataView[srcOffset + 1];
        let z = dataView[srcOffset + 2];
        let vx = dataView[srcOffset + 3];
        let vy = dataView[srcOffset + 4];
        let vz = dataView[srcOffset + 5];

        if (ENABLE_POSITION_VALIDATION) {
          // Validate and clamp values
          const positionsValid = [x, y, z].every(v => this._validatePosition(v));
          const velocitiesValid = [vx, vy, vz].every(v => this._validateVelocity(v));

          if (!positionsValid || !velocitiesValid) {
            this.invalidUpdates++;
            if (ENABLE_BINARY_DEBUG) {
              console.warn('Invalid values in binary update:', {
                index: i,
                position: [x, y, z],
                velocity: [vx, vy, vz],
                timestamp: new Date().toISOString()
              });
            }

            // Clamp values
            x = this._clampPosition(x);
            y = this._clampPosition(y);
            z = this._clampPosition(z);
            vx = this._clampVelocity(vx);
            vy = this._clampVelocity(vy);
            vz = this._clampVelocity(vz);
          }
        }

        // Copy positions
        this.positions[posOffset] = x;
        this.positions[posOffset + 1] = y;
        this.positions[posOffset + 2] = z;

        // Copy velocities
        this.velocities[velOffset] = vx;
        this.velocities[velOffset + 1] = vy;
        this.velocities[velOffset + 2] = vz;

        srcOffset += 6;
      }

      this.isInitialLayout = message.isInitialLayout;
      
      // Update timing
      const now = Date.now();
      if (this.firstUpdateTime === 0) {
        this.firstUpdateTime = now;
      }
      this.lastUpdateTime = now;

      // Debug logging
      if (ENABLE_BINARY_DEBUG && nodeCount > 0) {
        console.debug('Binary update processed:', {
          nodeCount,
          isInitial: this.isInitialLayout,
          invalidRate: this.invalidUpdateRate,
          updateFrequency: this.updateFrequency,
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
      this.firstUpdateTime = 0;
      this.lastUpdateTime = 0;
      this.isInitialLayout = false;
      this.invalidUpdates = 0;
    }
  }
})
