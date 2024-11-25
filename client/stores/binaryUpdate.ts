import { defineStore } from 'pinia';
import type { BinaryUpdateState } from '@/types/stores';
import type { PositionUpdate } from '@/types/websocket';
import { useVisualizationStore } from './visualization';

export const useBinaryUpdateStore = defineStore('binaryUpdate', {
  state: (): BinaryUpdateState => ({
    positions: null,
    velocities: null,
    isProcessing: false,
    updateCount: 0,
    lastUpdateTime: 0
  }),

  getters: {
    hasPositions: (state) => state.positions !== null,
    getUpdateRate: (state) => {
      const now = Date.now();
      const timeDiff = now - state.lastUpdateTime;
      return timeDiff > 0 ? (state.updateCount / timeDiff) * 1000 : 0;
    }
  },

  actions: {
    processPositionUpdate(buffer: ArrayBuffer) {
      this.isProcessing = true;
      try {
        const positions = new Float32Array(buffer);
        
        // Extract header (first float32)
        const header = positions[0];
        const isInitialLayout = header >= 1.0;
        const timeStep = header % 1.0;
        
        // Process position data (skip header)
        const positionUpdates: PositionUpdate[] = [];
        for (let i = 1; i < positions.length; i += 6) {
          if (i + 5 < positions.length) {
            positionUpdates.push({
              x: positions[i],
              y: positions[i + 1],
              z: positions[i + 2],
              vx: positions[i + 3],
              vy: positions[i + 4],
              vz: positions[i + 5]
            });
          }
        }

        // Update visualization store with new positions
        const visualizationStore = useVisualizationStore();
        const nodes = visualizationStore.nodes;
        
        // Update node positions
        positionUpdates.forEach((update, index) => {
          if (index < nodes.length) {
            nodes[index].position = [update.x, update.y, update.z];
          }
        });

        // Store binary data
        this.positions = new Float32Array(positionUpdates.flatMap(update => [
          update.x, update.y, update.z
        ]));
        this.velocities = new Float32Array(positionUpdates.flatMap(update => [
          update.vx, update.vy, update.vz
        ]));

        // Update metrics
        this.updateCount++;
        this.lastUpdateTime = Date.now();

        // Emit update event for GPU-based renderers
        window.dispatchEvent(new CustomEvent('gpuPositionsUpdated', {
          detail: {
            positions: this.positions,
            velocities: this.velocities,
            isInitialLayout,
            timeStep
          }
        }));

      } catch (error) {
        console.error('Error processing binary update:', error);
        throw error;
      } finally {
        this.isProcessing = false;
      }
    },

    getPositionBuffer(): ArrayBuffer | null {
      return this.positions?.buffer || null;
    },

    getVelocityBuffer(): ArrayBuffer | null {
      return this.velocities?.buffer || null;
    },

    reset() {
      this.positions = null;
      this.velocities = null;
      this.isProcessing = false;
      this.updateCount = 0;
      this.lastUpdateTime = 0;
    }
  }
});
