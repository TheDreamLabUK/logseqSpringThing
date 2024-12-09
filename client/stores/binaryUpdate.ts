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
    _validatePosition(value: number): boolean {
      return !isNaN(value) && isFinite(value) && 
             value >= MIN_VALID_POSITION && value <= MAX_VALID_POSITION;
    },

    _validateVelocity(value: number): boolean {
      return !isNaN(value) && isFinite(value) &&
             value >= MIN_VALID_VELOCITY && value <= MAX_VALID_VELOCITY;
    },

    _clampPosition(value: number): number {
      if (isNaN(value) || !isFinite(value)) return 0;
      return Math.max(MIN_VALID_POSITION, Math.min(MAX_VALID_POSITION, value));
    },

    _clampVelocity(value: number): number {
      if (isNaN(value) || !isFinite(value)) return 0;
      return Math.max(MIN_VALID_VELOCITY, Math.min(MAX_VALID_VELOCITY, value));
    },

    _hasSignificantChange(
      newPos: [number, number, number],
      oldPos: [number, number, number],
      newVel: [number, number, number],
      oldVel: [number, number, number]
    ): boolean {
      return (
        Math.abs(newPos[0] - oldPos[0]) > this.positionChangeThreshold ||
        Math.abs(newPos[1] - oldPos[1]) > this.positionChangeThreshold ||
        Math.abs(newPos[2] - oldPos[2]) > this.positionChangeThreshold ||
        Math.abs(newVel[0] - oldVel[0]) > this.velocityChangeThreshold ||
        Math.abs(newVel[1] - oldVel[1]) > this.velocityChangeThreshold ||
        Math.abs(newVel[2] - oldVel[2]) > this.velocityChangeThreshold
      )
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

      try {
        new Float32Array(buffer);
      } catch (error) {
        logError('Failed to create Float32Array view:', error);
        return false;
      }

      return true;
    },

    updateNodePosition(
      index: number,
      x: number, y: number, z: number,
      vx: number, vy: number, vz: number
    ): void {
      const now = Date.now();
      
      // Throttle updates to 5 FPS
      if (now - this.lastThrottledUpdate < UPDATE_THROTTLE_MS) {
        this.pendingUpdate = true;
        return;
      }

      if (index >= 0 && index < this.nodeCount) {
        const posIndex = index * 3;
        const velIndex = index * 3;

        const oldPos: [number, number, number] = [
          this.previousPositions[posIndex],
          this.previousPositions[posIndex + 1],
          this.previousPositions[posIndex + 2]
        ];
        const oldVel: [number, number, number] = [
          this.previousVelocities[velIndex],
          this.previousVelocities[velIndex + 1],
          this.previousVelocities[velIndex + 2]
        ];

        if (ENABLE_POSITION_VALIDATION) {
          const positionsValid = [x, y, z].every(v => this._validatePosition(v));
          const velocitiesValid = [vx, vy, vz].every(v => this._validateVelocity(v));

          if (!positionsValid || !velocitiesValid) {
            this.invalidUpdates++;
            logWarn('Invalid position/velocity values detected:', {
              index,
              position: [x, y, z],
              velocity: [vx, vy, vz]
            });

            x = this._clampPosition(x);
            y = this._clampPosition(y);
            z = this._clampPosition(z);
            vx = this._clampVelocity(vx);
            vy = this._clampVelocity(vy);
            vz = this._clampVelocity(vz);
          }
        }

        const newPos: [number, number, number] = [x, y, z];
        const newVel: [number, number, number] = [vx, vy, vz];

        if (this._hasSignificantChange(newPos, oldPos, newVel, oldVel)) {
          this.positions[posIndex] = x;
          this.positions[posIndex + 1] = y;
          this.positions[posIndex + 2] = z;

          this.velocities[velIndex] = vx;
          this.velocities[velIndex + 1] = vy;
          this.velocities[velIndex + 2] = vz;

          this.changedNodes.add(index);
        }

        this.lastUpdateTime = now;
        this.lastThrottledUpdate = now;
        this.pendingUpdate = false;
      }
    },

    updateFromBinary(message: BinaryMessage): void {
      const now = Date.now();
      
      // Throttle updates to 5 FPS
      if (now - this.lastThrottledUpdate < UPDATE_THROTTLE_MS) {
        this.pendingUpdate = true;
        return;
      }

      if (!this._validateBuffer(message.data)) {
        return;
      }

      const dataView = new Float32Array(message.data);
      const nodeCount = dataView.length / 6;

      if (nodeCount * BINARY_UPDATE_NODE_SIZE !== message.data.byteLength) {
        logError('Binary message size mismatch:', {
          received: message.data.byteLength,
          expected: nodeCount * BINARY_UPDATE_NODE_SIZE,
          nodeCount
        });
        return;
      }

      if (nodeCount > MAX_ARRAY_SIZE / 6) {
        logError('Excessive node count:', {
          nodeCount,
          maxNodes: MAX_ARRAY_SIZE / 6
        });
        return;
      }

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

      this.previousPositions.set(this.positions);
      this.previousVelocities.set(this.velocities);
      this.changedNodes.clear();

      for (let i = 0; i < nodeCount; i++) {
        const srcOffset = i * 6;
        const posOffset = i * 3;
        const velOffset = i * 3;

        let x = dataView[srcOffset];
        let y = dataView[srcOffset + 1];
        let z = dataView[srcOffset + 2];
        let vx = dataView[srcOffset + 3];
        let vy = dataView[srcOffset + 4];
        let vz = dataView[srcOffset + 5];

        if (ENABLE_POSITION_VALIDATION) {
          const positionsValid = [x, y, z].every(v => this._validatePosition(v));
          const velocitiesValid = [vx, vy, vz].every(v => this._validateVelocity(v));

          if (!positionsValid || !velocitiesValid) {
            this.invalidUpdates++;
            logWarn('Invalid values in binary update:', {
              index: i,
              position: [x, y, z],
              velocity: [vx, vy, vz]
            });

            x = this._clampPosition(x);
            y = this._clampPosition(y);
            z = this._clampPosition(z);
            vx = this._clampVelocity(vx);
            vy = this._clampVelocity(vy);
            vz = this._clampVelocity(vz);
          }
        }

        const oldPos: [number, number, number] = [
          this.previousPositions[posOffset],
          this.previousPositions[posOffset + 1],
          this.previousPositions[posOffset + 2]
        ];
        const oldVel: [number, number, number] = [
          this.previousVelocities[velOffset],
          this.previousVelocities[velOffset + 1],
          this.previousVelocities[velOffset + 2]
        ];
        const newPos: [number, number, number] = [x, y, z];
        const newVel: [number, number, number] = [vx, vy, vz];

        if (this._hasSignificantChange(newPos, oldPos, newVel, oldVel)) {
          this.positions[posOffset] = x;
          this.positions[posOffset + 1] = y;
          this.positions[posOffset + 2] = z;

          this.velocities[velOffset] = vx;
          this.velocities[velOffset + 1] = vy;
          this.velocities[velOffset + 2] = vz;

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
