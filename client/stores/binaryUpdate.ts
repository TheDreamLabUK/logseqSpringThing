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
      newValues: Float32Array,
      oldValues: Float32Array,
      offset: number,
      threshold: number
    ): boolean {
      return Math.abs(newValues[offset] - oldValues[offset]) > threshold ||
             Math.abs(newValues[offset + 1] - oldValues[offset + 1]) > threshold ||
             Math.abs(newValues[offset + 2] - oldValues[offset + 2]) > threshold;
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
          this.positions = new Float32Array(nodeCount * 3);
          this.velocities = new Float32Array(nodeCount * 3);
          this.previousPositions = new Float32Array(nodeCount * 3);
          this.previousVelocities = new Float32Array(nodeCount * 3);
          this.nodeCount = nodeCount;
        } catch (error) {
          logError('Failed to allocate arrays:', {
            nodeCount,
            error: error instanceof Error ? error.message : String(error)
          });
          return;
        }
      }

      // Store current values as previous
      this.previousPositions.set(this.positions);
      this.previousVelocities.set(this.velocities);
      this.changedNodes.clear();

      // Process position and velocity data using DataView for direct access
      let offset = 0;
      for (let i = 0; i < nodeCount; i++) {
        const posOffset = i * 3;
        const velOffset = i * 3;

        // Read values directly from DataView
        let x = dataView.getFloat32(offset, true);
        let y = dataView.getFloat32(offset + 4, true);
        let z = dataView.getFloat32(offset + 8, true);
        let vx = dataView.getFloat32(offset + 12, true);
        let vy = dataView.getFloat32(offset + 16, true);
        let vz = dataView.getFloat32(offset + 20, true);
        offset += 24;

        if (ENABLE_POSITION_VALIDATION) {
          // Validate and clamp values
          if (!this._validateValue(x, MIN_VALID_POSITION, MAX_VALID_POSITION) ||
              !this._validateValue(y, MIN_VALID_POSITION, MAX_VALID_POSITION) ||
              !this._validateValue(z, MIN_VALID_POSITION, MAX_VALID_POSITION) ||
              !this._validateValue(vx, MIN_VALID_VELOCITY, MAX_VALID_VELOCITY) ||
              !this._validateValue(vy, MIN_VALID_VELOCITY, MAX_VALID_VELOCITY) ||
              !this._validateValue(vz, MIN_VALID_VELOCITY, MAX_VALID_VELOCITY)) {
            this.invalidUpdates++;
            x = this._clampValue(x, MIN_VALID_POSITION, MAX_VALID_POSITION);
            y = this._clampValue(y, MIN_VALID_POSITION, MAX_VALID_POSITION);
            z = this._clampValue(z, MIN_VALID_POSITION, MAX_VALID_POSITION);
            vx = this._clampValue(vx, MIN_VALID_VELOCITY, MAX_VALID_VELOCITY);
            vy = this._clampValue(vy, MIN_VALID_VELOCITY, MAX_VALID_VELOCITY);
            vz = this._clampValue(vz, MIN_VALID_VELOCITY, MAX_VALID_VELOCITY);
          }
        }

        // Check if position or velocity has changed significantly
        const hasPositionChange = this._hasSignificantChange(
          new Float32Array([x, y, z]),
          this.previousPositions,
          posOffset,
          this.positionChangeThreshold
        );

        const hasVelocityChange = this._hasSignificantChange(
          new Float32Array([vx, vy, vz]),
          this.previousVelocities,
          velOffset,
          this.velocityChangeThreshold
        );

        if (hasPositionChange || hasVelocityChange) {
          // Update positions
          this.positions[posOffset] = x;
          this.positions[posOffset + 1] = y;
          this.positions[posOffset + 2] = z;

          // Update velocities
          this.velocities[velOffset] = vx;
          this.velocities[velOffset + 1] = vy;
          this.velocities[velOffset + 2] = vz;

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
