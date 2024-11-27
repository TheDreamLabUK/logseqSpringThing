import { ref, computed, onBeforeUnmount } from 'vue';
import { Vector3 } from 'three';
import { useSettingsStore } from '../stores/settings';
import type { Node, Edge, CoreState, InitializationOptions } from '../types/core';
import type { PositionUpdate } from '../types/websocket';
import { POSITION_SCALE, VELOCITY_SCALE } from '../constants/websocket';

/**
 * Visualization system composable that handles:
 * - State management for vue-threejs
 * - Position updates from WebSocket
 * - Node/Edge data transformation
 * - Performance optimizations
 */
export function useVisualization() {
  // Store for visualization settings
  const settingsStore = useSettingsStore();
  
  // Core visualization state
  const state = ref<CoreState>({
    renderer: null,
    camera: null,
    scene: null,
    canvas: null,
    isInitialized: false,
    isXRSupported: false,
    isWebGL2: false,
    isGPUMode: false,
    fps: 0,
    lastFrameTime: 0
  });

  // Node positions for efficient updates
  const nodePositions = new Map<string, Vector3>();
  const nodeVelocities = new Map<string, Vector3>();
  
  // Computed settings from store
  const settings = computed(() => settingsStore.getVisualizationSettings);

  /**
   * Initialize visualization system
   */
  const initialize = async (options: InitializationOptions) => {
    if (state.value.isInitialized) return;

    try {
      console.log('Initializing visualization system...');
      
      // Set up WebGL2 context
      const context = options.canvas.getContext('webgl2');
      state.value.isWebGL2 = !!context;
      state.value.canvas = options.canvas;
      state.value.isInitialized = true;

      console.log('Visualization system initialized:', {
        webgl2: state.value.isWebGL2,
        canvas: state.value.canvas?.width,
        height: state.value.canvas?.height
      });

    } catch (error) {
      console.error('Failed to initialize visualization:', error);
      throw error;
    }
  };

  /**
   * Handle binary position updates from WebSocket
   */
  const updatePositions = (positions: PositionUpdate[], isInitialLayout: boolean) => {
    if (!state.value.isInitialized) return;

    // Enable GPU mode on first position update
    if (!state.value.isGPUMode) {
      state.value.isGPUMode = true;
      console.log('Switching to GPU-accelerated mode');
    }

    // Update node positions and velocities
    positions.forEach((pos) => {
      // Dequantize position values
      const position = new Vector3(
        pos.x / POSITION_SCALE,
        pos.y / POSITION_SCALE,
        pos.z / POSITION_SCALE
      );
      nodePositions.set(pos.id, position);

      // Dequantize velocity values
      const velocity = new Vector3(
        pos.vx / VELOCITY_SCALE,
        pos.vy / VELOCITY_SCALE,
        pos.vz / VELOCITY_SCALE
      );
      nodeVelocities.set(pos.id, velocity);
    });

    // Log update stats
    console.debug('Position update:', {
      count: positions.length,
      isInitial: isInitialLayout,
      sample: positions[0] ? {
        id: positions[0].id,
        position: nodePositions.get(positions[0].id)?.toArray(),
        velocity: nodeVelocities.get(positions[0].id)?.toArray()
      } : null
    });
  };

  /**
   * Update node data
   */
  const updateNodes = (nodes: Node[]) => {
    if (!state.value.isInitialized) return;

    console.log('Updating nodes:', {
      count: nodes.length,
      sample: nodes[0] ? {
        id: nodes[0].id,
        position: nodes[0].position,
        size: nodes[0].size
      } : null
    });

    // Update stored positions
    nodes.forEach(node => {
      if (node.position) {
        nodePositions.set(node.id, new Vector3(...node.position));
      }
    });
  };

  /**
   * Get node position
   */
  const getNodePosition = (nodeId: string): Vector3 | undefined => {
    return nodePositions.get(nodeId);
  };

  /**
   * Get node velocity
   */
  const getNodeVelocity = (nodeId: string): Vector3 | undefined => {
    return nodeVelocities.get(nodeId);
  };

  /**
   * Clean up visualization system
   */
  const dispose = () => {
    console.log('Disposing visualization system...');
    
    // Clear stored data
    nodePositions.clear();
    nodeVelocities.clear();

    // Reset state
    state.value = {
      renderer: null,
      camera: null,
      scene: null,
      canvas: null,
      isInitialized: false,
      isXRSupported: false,
      isWebGL2: false,
      isGPUMode: false,
      fps: 0,
      lastFrameTime: 0
    };

    console.log('Visualization system disposed');
  };

  // Clean up on component unmount
  onBeforeUnmount(() => {
    dispose();
  });

  return {
    state,
    initialize,
    updateNodes,
    updatePositions,
    getNodePosition,
    getNodeVelocity,
    dispose
  };
}
