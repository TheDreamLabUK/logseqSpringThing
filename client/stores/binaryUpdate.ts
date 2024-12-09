import { defineStore } from 'pinia'
import type { BinaryMessage } from '../types/websocket'
import { logError, logWarn, logData } from '../utils/debug_log'
import {
  BINARY_UPDATE_NODE_SIZE,
  FLOAT32_SIZE,
  MAX_VALID_POSITION,
  MIN_VALID_POSITION,
  MAX_VALID_VELOCITY,
  MIN_VALID_VELOCITY,
  ENABLE_POSITION_VALIDATION,
  UPDATE_THROTTLE_MS,
  POSITION_CHANGE_THRESHOLD,
  VELOCITY_CHANGE_THRESHOLD
} from '../constants/websocket'

// Maximum array size to prevent memory issues
const MAX_ARRAY_SIZE = 1000000

interface BinaryUpdateState {
  positions: Float32Array
  velocities: Float32Array
  previousPositions: Float32Array
  previousVelocities: Float32Array
  nodeCount: number
  lastUpdateTime: number
  invalidUpdates: number
  pendingUpdate: boolean
  lastThrottledUpdate: number
  changedNodes: Set<number>
  positionChangeThreshold: number
  velocityChangeThreshold: number
}

export const useBinaryUpdateStore = defineStore('binaryUpdate', {
  state: (): BinaryUpdateState => ({
    positions: new Float32Array(0),
    velocities: new Float32Array(0),
    previousPositions: new Float32Array(0),
    previousVelocities: new Float32Array(0),
    nodeCount: 0,
    lastUpdateTime: 0,
    invalidUpdates: 0,
    pendingUpdate: false,
    lastThrottledUpdate: 0,
    changedNodes: new Set(),
    positionChangeThreshold: POSITION_CHANGE_THRESHOLD,
    velocityChangeThreshold: VELOCITY_CHANGE_THRESHOLD
  }),

  getters: {
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

    getAllPositions: (state): Float32Array => state.positions,
    getAllVelocities: (state): Float32Array => state.velocities,
    getChangedNodes: (state): Set<number> => state.changedNodes
  },

  actions: {
    _validateValue(value: number, min: number, max: number): boolean {
      return !isNaN(value) && isFinite(value) && value >= min && value <= max;
    },

    _clampValue(value: number, min: number, max: number): number {
      if (isNaN(value) || !isFinite(value)) return 0;
      return Math.max(min, Math.min(max, value));
    },

    _hasSignificantChange(
      newValue1: number,
      newValue2: number,
      newValue3: number,
      oldValue1: number,
      oldValue2: number,
      oldValue3: number,
      threshold: number
    ): boolean {
      return Math.abs(newValue1 - oldValue1) > threshold ||
             Math.abs(newValue2 - oldValue2) > threshold ||
             Math.abs(newValue3 - oldValue3) > threshold;
    },

    _validateBuffer(buffer: ArrayBuffer): boolean {
      if (buffer.byteLength % FLOAT32_SIZE !== 0) {
        logError('Buffer not aligned to float32:', {
          byteLength: buffer.byteLength,
          alignment: FLOAT32_SIZE
        });
        return false;
      }

      const floatCount = buffer.byteLength / FLOAT32_SIZE;
      if (floatCount > MAX_ARRAY_SIZE) {
        logError('Buffer exceeds maximum size:', {
          floatCount,
          maxSize: MAX_ARRAY_SIZE
        });
        return false;
      }

      return true;
    },

    updateFromBinary(message: BinaryMessage): void {
      const now = performance.now();
      
      // Throttle updates to 5 FPS
      if (now - this.lastThrottledUpdate < UPDATE_THROTTLE_MS) {
        this.pendingUpdate = true;
        return;
      }

      if (!this._validateBuffer(message.data)) {
        return;
      }

      const dataView = new DataView(message.data);
      const nodeCount = message.data.byteLength / BINARY_UPDATE_NODE_SIZE;

      if (nodeCount > MAX_ARRAY_SIZE / 6) {
        logError('Excessive node count:', {
          nodeCount,
          maxNodes: MAX_ARRAY_SIZE / 6
        });
        return;
      }

      // Resize arrays if needed
      if (this.nodeCount !== nodeCount) {
        try {
          // Create new arrays with exact size needed
          const newPositions = new Float32Array(nodeCount * 3);
          const newVelocities = new Float32Array(nodeCount * 3);
          
          // Copy existing data if any
          if (this.nodeCount > 0) {
            newPositions.set(this.positions.subarray(0, Math.min(this.positions.length, newPositions.length)));
            newVelocities.set(this.velocities.subarray(0, Math.min(this.velocities.length, newVelocities.length)));
          }
          
          this.positions = newPositions;
          this.velocities = newVelocities;
          this.previousPositions = new Float32Array(this.positions);
          this.previousVelocities = new Float32Array(this.velocities);
          this.nodeCount = nodeCount;
        } catch (error) {
          logError('Failed to allocate arrays:', {
            nodeCount,
            error: error instanceof Error ? error.message : String(error)
          });
          return;
        }
      } else {
        // Store current values as previous
        this.previousPositions.set(this.positions);
        this.previousVelocities.set(this.velocities);
      }

      this.changedNodes.clear();

      // Process data directly from DataView
      let byteOffset = 0;
      const positionStride = 3;
      const velocityStride = 3;

      for (let i = 0; i < nodeCount; i++) {
        const positionIndex = i * positionStride;
        const velocityIndex = i * velocityStride;

        // Read values directly from DataView
        const x = dataView.getFloat32(byteOffset, true);
        const y = dataView.getFloat32(byteOffset + 4, true);
        const z = dataView.getFloat32(byteOffset + 8, true);
        const vx = dataView.getFloat32(byteOffset + 12, true);
        const vy = dataView.getFloat32(byteOffset + 16, true);
        const vz = dataView.getFloat32(byteOffset + 20, true);
        byteOffset += 24;

        let finalX = x;
        let finalY = y;
        let finalZ = z;
        let finalVX = vx;
        let finalVY = vy;
        let finalVZ = vz;

        if (ENABLE_POSITION_VALIDATION) {
          // Validate and clamp values in-place
          if (!this._validateValue(x, MIN_VALID_POSITION, MAX_VALID_POSITION) ||
              !this._validateValue(y, MIN_VALID_POSITION, MAX_VALID_POSITION) ||
              !this._validateValue(z, MIN_VALID_POSITION, MAX_VALID_POSITION) ||
              !this._validateValue(vx, MIN_VALID_VELOCITY, MAX_VALID_VELOCITY) ||
              !this._validateValue(vy, MIN_VALID_VELOCITY, MAX_VALID_VELOCITY) ||
              !this._validateValue(vz, MIN_VALID_VELOCITY, MAX_VALID_VELOCITY)) {
            this.invalidUpdates++;
            finalX = this._clampValue(x, MIN_VALID_POSITION, MAX_VALID_POSITION);
            finalY = this._clampValue(y, MIN_VALID_POSITION, MAX_VALID_POSITION);
            finalZ = this._clampValue(z, MIN_VALID_POSITION, MAX_VALID_POSITION);
            finalVX = this._clampValue(vx, MIN_VALID_VELOCITY, MAX_VALID_VELOCITY);
            finalVY = this._clampValue(vy, MIN_VALID_VELOCITY, MAX_VALID_VELOCITY);
            finalVZ = this._clampValue(vz, MIN_VALID_VELOCITY, MAX_VALID_VELOCITY);
          }
        }

        // Check for significant changes using direct value comparison
        const hasPositionChange = this._hasSignificantChange(
          finalX, finalY, finalZ,
          this.previousPositions[positionIndex],
          this.previousPositions[positionIndex + 1],
          this.previousPositions[positionIndex + 2],
          this.positionChangeThreshold
        );

        const hasVelocityChange = this._hasSignificantChange(
          finalVX, finalVY, finalVZ,
          this.previousVelocities[velocityIndex],
          this.previousVelocities[velocityIndex + 1],
          this.previousVelocities[velocityIndex + 2],
          this.velocityChangeThreshold
        );

        if (hasPositionChange || hasVelocityChange) {
          // Update values directly in the typed arrays
          this.positions[positionIndex] = finalX;
          this.positions[positionIndex + 1] = finalY;
          this.positions[positionIndex + 2] = finalZ;
          this.velocities[velocityIndex] = finalVX;
          this.velocities[velocityIndex + 1] = finalVY;
          this.velocities[velocityIndex + 2] = finalVZ;

          // Mark node as changed
          this.changedNodes.add(i);
        }
      }

      this.lastUpdateTime = now;
      this.lastThrottledUpdate = now;
      this.pendingUpdate = false;

      logData('Binary update processed:', {
        nodeCount,
        changedNodes: this.changedNodes.size,
        timeSinceLastUpdate: now - this.lastUpdateTime,
        timestamp: new Date().toISOString()
      });
    },

    clear(): void {
      this.positions = new Float32Array(0);
      this.velocities = new Float32Array(0);
      this.previousPositions = new Float32Array(0);
      this.previousVelocities = new Float32Array(0);
      this.nodeCount = 0;
      this.lastUpdateTime = 0;
      this.invalidUpdates = 0;
      this.pendingUpdate = false;
      this.lastThrottledUpdate = 0;
      this.changedNodes.clear();
      this.positionChangeThreshold = POSITION_CHANGE_THRESHOLD;
      this.velocityChangeThreshold = VELOCITY_CHANGE_THRESHOLD;
      
      logData('Binary update store cleared');
    }
  }
})
