import { defineStore } from 'pinia';
import { useVisualizationStore } from './visualization';

// No state needed since updates are transient
export const useBinaryUpdateStore = defineStore('binaryUpdate', {
  actions: {
    processBinaryMessage(buffer: ArrayBuffer) {
      const view = new DataView(buffer);
      let offset = 0;

      // Read header
      const isInitialLayout = Boolean(view.getInt32(offset));
      offset += 4;
      const timeStep = view.getFloat64(offset);
      offset += 8;

      // Read number of nodes
      const nodeCount = view.getInt32(offset);
      offset += 4;

      // Get visualization store for direct updates
      const visualizationStore = useVisualizationStore();
      
      // Process positions and velocities directly
      const updates: Array<[string, [number, number, number]]> = [];
      for (let i = 0; i < nodeCount; i++) {
        // Read position
        const x = view.getFloat64(offset);
        offset += 8;
        const y = view.getFloat64(offset);
        offset += 8;
        const z = view.getFloat64(offset);
        offset += 8;

        // Skip velocity data since it's not stored
        offset += 24; // 3 * 8 bytes for vx, vy, vz

        // Add to updates array
        if (visualizationStore.nodes[i]) {
          updates.push([visualizationStore.nodes[i].id, [x, y, z]]);
        }
      }

      // Batch update positions
      visualizationStore.updateNodePositions(updates);
    },

    createBinaryMessage(
      nodePositions: Array<[string, [number, number, number]]>,
      isInitialLayout: boolean = false,
      timeStep: number = performance.now()
    ): ArrayBuffer {
      const nodeCount = nodePositions.length;

      // Calculate buffer size:
      // 4 bytes (isInitialLayout) + 8 bytes (timeStep) + 4 bytes (nodeCount) +
      // nodeCount * (3 * 8 bytes for position + 3 * 8 bytes for velocity)
      const bufferSize = 4 + 8 + 4 + nodeCount * (6 * 8);
      const buffer = new ArrayBuffer(bufferSize);
      const view = new DataView(buffer);
      let offset = 0;

      // Write header
      view.setInt32(offset, isInitialLayout ? 1 : 0);
      offset += 4;
      view.setFloat64(offset, timeStep);
      offset += 8;

      // Write node count
      view.setInt32(offset, nodeCount);
      offset += 4;

      // Write positions and zero velocities
      for (const [_, position] of nodePositions) {
        // Write position
        const [x, y, z] = position;
        view.setFloat64(offset, x);
        offset += 8;
        view.setFloat64(offset, y);
        offset += 8;
        view.setFloat64(offset, z);
        offset += 8;

        // Write zero velocities since they're not used
        view.setFloat64(offset, 0);
        offset += 8;
        view.setFloat64(offset, 0);
        offset += 8;
        view.setFloat64(offset, 0);
        offset += 8;
      }

      return buffer;
    }
  }
});
