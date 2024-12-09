import { defineStore } from 'pinia'
import type { BinaryMessage } from '../types/websocket'
import { logError, logWarn, logData, logBinaryHeader, logPerformance } from '../utils/debug_log'
import {
  BINARY_UPDATE_NODE_SIZE,
  FLOAT32_SIZE,
  MAX_VALID_POSITION,
  MIN_VALID_POSITION,
  MAX_VALID_VELOCITY,
  MIN_VALID_VELOCITY,
  ENABLE_BINARY_DEBUG,
  ENABLE_POSITION_VALIDATION
} from '../constants/websocket'

// Enhanced change detection thresholds with dynamic adjustment
const BASE_POSITION_CHANGE_THRESHOLD = 0.01
const BASE_VELOCITY_CHANGE_THRESHOLD = 0.001
const THRESHOLD_ADJUSTMENT_FACTOR = 1.5 // Increase threshold when high update frequency detected

// Update throttling configuration
const UPDATE_THROTTLE_MS = 16 // ~60fps
const MAX_UPDATES_PER_SECOND = 120 // Prevent excessive updates

// Memory monitoring thresholds
const MEMORY_WARNING_THRESHOLD = 100 * 1024 * 1024 // 100MB
const MAX_ARRAY_SIZE = 1000000 // Prevent allocation of extremely large arrays

interface BinaryUpdateState {
  positions: Float32Array
  velocities: Float32Array
  previousPositions: Float32Array
  previousVelocities: Float32Array
  nodeCount: number
  firstUpdateTime: number
  lastUpdateTime: number
  invalidUpdates: number
  pendingUpdate: boolean
  lastThrottledUpdate: number
  changedNodes: Set<number>
  updateCount: number // Track total number of updates
  memoryUsage: number // Track approximate memory usage
  positionChangeThreshold: number // Dynamic threshold
  velocityChangeThreshold: number // Dynamic threshold
  processingTimes: number[] // Track update processing times
}

/**
 * Store for handling binary position/velocity updates
 * Optimized for high-frequency updates in force-directed graph
 */
export const useBinaryUpdateStore = defineStore('binaryUpdate', {
  state: (): BinaryUpdateState => ({
    positions: new Float32Array(0),
    velocities: new Float32Array(0),
    previousPositions: new Float32Array(0),
    previousVelocities: new Float32Array(0),
    nodeCount: 0,
    firstUpdateTime: 0,
    lastUpdateTime: 0,
    invalidUpdates: 0,
    pendingUpdate: false,
    lastThrottledUpdate: 0,
    changedNodes: new Set(),
    updateCount: 0,
    memoryUsage: 0,
    positionChangeThreshold: BASE_POSITION_CHANGE_THRESHOLD,
    velocityChangeThreshold: BASE_VELOCITY_CHANGE_THRESHOLD,
    processingTimes: []
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
    getChangedNodes: (state): Set<number> => state.changedNodes,

    invalidUpdateRate: (state): number => {
      if (state.firstUpdateTime === 0) return 0;
      const timeSpan = (state.lastUpdateTime - state.firstUpdateTime) / 1000;
      const totalUpdates = Math.max(1, state.updateCount);
      return (state.invalidUpdates / totalUpdates) * 100;
    },

    updateFrequency: (state): number => {
      if (state.firstUpdateTime === 0) return 0;
      const timeSpan = (state.lastUpdateTime - state.firstUpdateTime) / 1000;
      return timeSpan > 0 ? state.updateCount / timeSpan : 0;
    },

    averageProcessingTime: (state): number => {
      if (state.processingTimes.length === 0) return 0;
      const sum = state.processingTimes.reduce((a, b) => a + b, 0);
      return sum / state.processingTimes.length;
    },

    memoryUsageMB: (state): number => {
      return state.memoryUsage / (1024 * 1024);
    }
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

    _updateMemoryUsage(): void {
      // Calculate approximate memory usage
      const arrayBytes = (this.nodeCount * 3 * 4) * 4; // 4 Float32Arrays, 4 bytes per float
      const setBytes = this.changedNodes.size * 4; // Approximate Set memory usage
      this.memoryUsage = arrayBytes + setBytes;

      if (this.memoryUsage > MEMORY_WARNING_THRESHOLD) {
        logWarn('High memory usage detected:', {
          usageMB: this.memoryUsageMB,
          nodeCount: this.nodeCount,
          changedNodesCount: this.changedNodes.size
        });
      }
    },

    _adjustThresholds(): void {
      const frequency = this.updateFrequency;
      if (frequency > MAX_UPDATES_PER_SECOND) {
        this.positionChangeThreshold = BASE_POSITION_CHANGE_THRESHOLD * THRESHOLD_ADJUSTMENT_FACTOR;
        this.velocityChangeThreshold = BASE_VELOCITY_CHANGE_THRESHOLD * THRESHOLD_ADJUSTMENT_FACTOR;
        logData('Thresholds adjusted for high frequency:', {
          frequency,
          positionThreshold: this.positionChangeThreshold,
          velocityThreshold: this.velocityChangeThreshold
        });
      } else {
        this.positionChangeThreshold = BASE_POSITION_CHANGE_THRESHOLD;
        this.velocityChangeThreshold = BASE_VELOCITY_CHANGE_THRESHOLD;
      }
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
      // Check buffer alignment
      if (buffer.byteLength % FLOAT32_SIZE !== 0) {
        logError('Buffer not aligned to float32:', {
          byteLength: buffer.byteLength,
          alignment: FLOAT32_SIZE
        });
        return false;
      }

      // Check for reasonable size
      const floatCount = buffer.byteLength / FLOAT32_SIZE;
      if (floatCount > MAX_ARRAY_SIZE) {
        logError('Buffer exceeds maximum size:', {
          floatCount,
          maxSize: MAX_ARRAY_SIZE
        });
        return false;
      }

      // Validate data view creation
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
      const startTime = performance.now();

      if (index >= 0 && index < this.nodeCount) {
        const posIndex = index * 3;
        const velIndex = index * 3;

        // Get previous values for change detection
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
              velocity: [vx, vy, vz],
              timestamp: new Date().toISOString()
            });

            x = this._clampPosition(x);
            y = this._clampPosition(y);
            z = this._clampPosition(z);
            vx = this._clampVelocity(vx);
            vy = this._clampVelocity(vy);
            vz = this._clampVelocity(vz);
          }
        }

        // Check if the change is significant
        const newPos: [number, number, number] = [x, y, z];
        const newVel: [number, number, number] = [vx, vy, vz];

        if (this._hasSignificantChange(newPos, oldPos, newVel, oldVel)) {
          // Update position
          this.positions[posIndex] = x;
          this.positions[posIndex + 1] = y;
          this.positions[posIndex + 2] = z;

          // Update velocity
          this.velocities[velIndex] = vx;
          this.velocities[velIndex + 1] = vy;
          this.velocities[velIndex + 2] = vz;

          // Mark node as changed
          this.changedNodes.add(index);
        }

        // Update timing and counters
        const now = Date.now();
        if (this.firstUpdateTime === 0) {
          this.firstUpdateTime = now;
        }
        this.lastUpdateTime = now;
        this.updateCount++;

        // Track processing time
        const processingTime = performance.now() - startTime;
        this.processingTimes.push(processingTime);
        if (this.processingTimes.length > 100) {
          this.processingTimes.shift(); // Keep only last 100 measurements
        }

        this._updateMemoryUsage();
        this._adjustThresholds();
      }
    },

    updateFromBinary(message: BinaryMessage): void {
      const startTime = performance.now();
      const now = Date.now();
      
      // Throttle updates
      if (now - this.lastThrottledUpdate < UPDATE_THROTTLE_MS) {
        this.pendingUpdate = true;
        return;
      }

      // Validate buffer
      if (!this._validateBuffer(message.data)) {
        return;
      }

      const dataView = new Float32Array(message.data);
      const nodeCount = dataView.length / 6;

      // Validate node count
      if (nodeCount * BINARY_UPDATE_NODE_SIZE !== message.data.byteLength) {
        logError('Binary message size mismatch:', {
          received: message.data.byteLength,
          expected: nodeCount * BINARY_UPDATE_NODE_SIZE,
          nodeCount,
          timestamp: new Date().toISOString()
        });
        return;
      }

      // Check for reasonable node count
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

      // Log binary header for debugging
      if (ENABLE_BINARY_DEBUG) {
        logBinaryHeader(message.data);
      }

      // Process position and velocity data
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
              velocity: [vx, vy, vz],
              timestamp: new Date().toISOString()
            });

            x = this._clampPosition(x);
            y = this._clampPosition(y);
            z = this._clampPosition(z);
            vx = this._clampVelocity(vx);
            vy = this._clampVelocity(vy);
            vz = this._clampVelocity(vz);
          }
        }

        // Check if the change is significant
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
          // Copy positions
          this.positions[posOffset] = x;
          this.positions[posOffset + 1] = y;
          this.positions[posOffset + 2] = z;

          // Copy velocities
          this.velocities[velOffset] = vx;
          this.velocities[velOffset + 1] = vy;
          this.velocities[velOffset + 2] = vz;

          // Mark node as changed
          this.changedNodes.add(i);
        }
      }

      // Update timing and state
      if (this.firstUpdateTime === 0) {
        this.firstUpdateTime = now;
      }
      this.lastUpdateTime = now;
      this.lastThrottledUpdate = now;
      this.pendingUpdate = false;
      this.updateCount++;

      // Track processing time
      const processingTime = performance.now() - startTime;
      this.processingTimes.push(processingTime);
      if (this.processingTimes.length > 100) {
        this.processingTimes.shift(); // Keep only last 100 measurements
      }

      // Update memory usage and thresholds
      this._updateMemoryUsage();
      this._adjustThresholds();

      logPerformance('Binary update processed:', {
        nodeCount,
        changedNodes: this.changedNodes.size,
        invalidRate: this.invalidUpdateRate,
        updateFrequency: this.updateFrequency,
        processingTime,
        averageProcessingTime: this.averageProcessingTime,
        memoryUsageMB: this.memoryUsageMB,
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
    },

    clear(): void {
      this.positions = new Float32Array(0);
      this.velocities = new Float32Array(0);
      this.previousPositions = new Float32Array(0);
      this.previousVelocities = new Float32Array(0);
      this.nodeCount = 0;
      this.firstUpdateTime = 0;
      this.lastUpdateTime = 0;
      this.invalidUpdates = 0;
      this.pendingUpdate = false;
      this.lastThrottledUpdate = 0;
      this.changedNodes.clear();
      this.updateCount = 0;
      this.memoryUsage = 0;
      this.positionChangeThreshold = BASE_POSITION_CHANGE_THRESHOLD;
      this.velocityChangeThreshold = BASE_VELOCITY_CHANGE_THRESHOLD;
      this.processingTimes = [];
      
      logData('Binary update store cleared');
    }
  }
})
