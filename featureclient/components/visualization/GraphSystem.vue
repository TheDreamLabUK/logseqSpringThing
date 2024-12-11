<template>
  <div v-if="isReady && graphData">
    <!-- Graph Content -->
    <primitive :object="graphGroup" />
    <div v-if="process.env.NODE_ENV === 'development'" class="debug-info">
      Nodes: {{ graphData.nodes.length }} | Edges: {{ graphData.edges.length }}
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, computed, watch, onMounted, inject } from 'vue';
import { Vector3, Vector2, Plane, Raycaster } from 'three';
import { useGraphSystem } from '../../composables/useGraphSystem';
import { useWebSocketStore } from '../../stores/websocket';
import { useBinaryUpdateStore } from '../../stores/binaryUpdate';
import { useVisualizationStore } from '../../stores/visualization';
import { useVisualization } from '../../composables/useVisualization';
import { usePlatform } from '../../composables/usePlatform';
import type { VisualizationConfig } from '../../types/components';
import type { GraphNode, GraphEdge, GraphData, CoreState } from '../../types/core';

export default defineComponent({
  name: 'GraphSystem',

  props: {
    visualSettings: {
      type: Object as () => VisualizationConfig,
      required: true
    }
  },

  setup(props) {
    // Get visualization state and composable
    const visualizationState = inject<{ value: CoreState }>('visualizationState');
    const visualization = useVisualization();
    
    const isReady = computed(() => {
      const ready = visualizationState?.value.scene != null && 
             visualizationState?.value.isInitialized === true;
      console.debug('Graph system ready state:', {
        hasScene: visualizationState?.value.scene != null,
        isInitialized: visualizationState?.value.isInitialized,
        ready,
        timestamp: new Date().toISOString()
      });
      return ready;
    });

    // Get platform info
    const { 
      getPlatformInfo, 
      enableVR, 
      enableAR, 
      disableXR,
      isXRActive,
      isVRActive,
      isARActive 
    } = usePlatform();
    
    const platformInfo = getPlatformInfo();

    // Initialize graph system
    const {
      graphGroup,
      hoveredNode,
      getNodePosition,
      getNodeScale,
      getNodeColor,
      getEdgePoints,
      getEdgeColor,
      getEdgeWidth,
      handleNodeClick,
      handleNodeHover,
      updateGraphData,
      updateNodePosition
    } = useGraphSystem();

    const websocketStore = useWebSocketStore();
    const binaryUpdateStore = useBinaryUpdateStore();
    const visualizationStore = useVisualizationStore();
    
    // Drag state
    const isDragging = ref(false);
    const draggedNode = ref<GraphNode | null>(null);
    const dragStartPosition = ref<Vector3 | null>(null);
    const dragPlane = ref<Plane | null>(null);
    const dragIntersection = new Vector3();
    const lastUpdateTime = ref(0);
    const UPDATE_INTERVAL = 200; // 5 FPS

    // Graph data from store with enhanced debug logging
    const graphData = computed<GraphData>(() => {
      const data = visualizationStore.getGraphData || { nodes: [], edges: [], metadata: {} };
      console.debug('Graph data computed:', {
        nodes: data.nodes.length,
        edges: data.edges.length,
        hasMetadata: Object.keys(data.metadata || {}).length > 0,
        sampleNodes: data.nodes.slice(0, 3).map(n => ({
          id: n.id,
          hasPosition: !!n.position,
          edgeCount: n.edges?.length || 0
        })),
        timestamp: new Date().toISOString()
      });
      return data;
    });

    // Watch for ready state and request initial data
    watch(isReady, (ready) => {
      if (ready && !websocketStore.initialDataRequested) {
        console.debug('Graph system ready, requesting initial data');
        websocketStore.requestInitialData();
      }
    });

    // Watch for graph data changes with enhanced logging
    watch(() => graphData.value, (newData) => {
      if (newData && newData.nodes.length > 0) {
        console.debug('Graph data changed:', {
          nodes: newData.nodes.length,
          edges: newData.edges.length,
          sampleNode: newData.nodes[0] ? {
            id: newData.nodes[0].id,
            hasPosition: !!newData.nodes[0].position,
            edgeCount: newData.nodes[0].edges?.length || 0
          } : null,
          timestamp: new Date().toISOString()
        });
        
        // Use mergeGraphData to preserve client-side positions
        visualizationStore.mergeGraphData(newData);
        
        // Update graph system with merged data
        const graphDataToUpdate = visualizationStore.getGraphData || { nodes: [], edges: [], metadata: {} };
        updateGraphData(graphDataToUpdate);
      }
    }, { deep: true });

    // Watch for binary updates with enhanced logging
    watch(() => binaryUpdateStore.getAllPositions, (positions) => {
      const nodeCount = positions.length / 3;
      if (nodeCount > 0) {
        console.debug('Processing binary position update:', {
          nodeCount,
          timestamp: new Date().toISOString()
        });
        
        // Get changed nodes from binary store
        const changedNodes = binaryUpdateStore.getChangedNodes;
        
        // Update nodes based on interaction mode
        changedNodes.forEach(index => {
          const node = graphData.value.nodes[index];
          if (!node) return;

          // Only update non-interacted nodes when in local mode
          if (websocketStore.service?.isNodeInteracted(node.id)) {
            return;
          }

          const position = binaryUpdateStore.getNodePosition(index);
          const velocity = binaryUpdateStore.getNodeVelocity(index);
          
          if (position && velocity) {
            // Create Vector3 objects for position and velocity
            const pos = new Vector3(position[0], position[1], position[2]);
            const vel = new Vector3(velocity[0], velocity[1], velocity[2]);
            
            // Update node position in graph system
            updateNodePosition(node.id, pos, vel);

            // Update position and velocity while preserving other metadata
            visualizationStore.updateNode(node.id, {
              position,
              velocity
            });
          }
        });

        // Trigger graph update
        if (visualizationState?.value.scene) {
          visualizationState.value.scene.userData.needsRender = true;
        }
      }
    }, { deep: true });

    // Create binary update for server
    const createBinaryUpdate = (node: GraphNode, position: Vector3): ArrayBuffer => {
      const nodeIndex = graphData.value.nodes.findIndex(n => n.id === node.id);
      if (nodeIndex === -1) return new ArrayBuffer(0);

      const buffer = new ArrayBuffer(24); // 6 float32s (x,y,z,vx,vy,vz)
      const view = new Float32Array(buffer);

      // Set position
      view[0] = position.x;
      view[1] = position.y;
      view[2] = position.z;

      // Set velocity to 0
      view[3] = 0;
      view[4] = 0;
      view[5] = 0;

      return buffer;
    };

    // Drag handlers with enhanced logging and server sync
    const onDragStart = (event: PointerEvent, node: GraphNode) => {
      console.debug('Starting node drag:', {
        nodeId: node.id,
        initialPosition: getNodePosition(node),
        timestamp: new Date().toISOString()
      });

      // Start interaction mode and track the dragged node
      websocketStore.service?.startInteractionMode();
      websocketStore.service?.addInteractedNode(node.id);

      isDragging.value = true;
      draggedNode.value = node;

      const camera = visualizationState?.value.camera;
      if (!camera) return;

      const normal = new Vector3(0, 0, 1);
      normal.applyQuaternion(camera.quaternion);
      dragPlane.value = new Plane(normal, 0);
      dragStartPosition.value = getNodePosition(node).clone();

      if (visualizationState?.value.scene) {
        visualizationState.value.scene.userData.needsRender = true;
      }
    };

    const onDragMove = (event: PointerEvent) => {
      if (!isDragging.value || !draggedNode.value || !dragPlane.value) return;

      const camera = visualizationState?.value.camera;
      if (!camera) return;

      const target = event.currentTarget as HTMLElement;
      const rect = target.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      const raycaster = new Raycaster();
      const pointer = new Vector2(x, y);
      raycaster.setFromCamera(pointer, camera);

      if (raycaster.ray.intersectPlane(dragPlane.value, dragIntersection)) {
        const node = draggedNode.value;
        updateNodePosition(node.id, dragIntersection);

        // Send position update to server at 5 FPS
        const now = performance.now();
        if (now - lastUpdateTime.value >= UPDATE_INTERVAL) {
          const binaryUpdate = createBinaryUpdate(node, dragIntersection);
          if (binaryUpdate.byteLength > 0) {
            websocketStore.sendBinary(binaryUpdate);
          }
          lastUpdateTime.value = now;
        }

        if (visualizationState?.value.scene) {
          visualizationState.value.scene.userData.needsRender = true;
        }
      }
    };

    const onDragEnd = () => {
      if (!isDragging.value || !draggedNode.value || !dragStartPosition.value) return;

      const finalPosition = getNodePosition(draggedNode.value);

      console.debug('Node drag ended:', {
        nodeId: draggedNode.value.id,
        startPosition: dragStartPosition.value.toArray(),
        finalPosition: finalPosition.toArray(),
        timestamp: new Date().toISOString()
      });

      // Send final position to server
      const binaryUpdate = createBinaryUpdate(draggedNode.value, finalPosition);
      if (binaryUpdate.byteLength > 0) {
        websocketStore.sendBinary(binaryUpdate);
      }

      // End interaction mode
      websocketStore.service?.endInteractionMode();

      isDragging.value = false;
      draggedNode.value = null;
      dragStartPosition.value = null;
      dragPlane.value = null;

      if (visualizationState?.value.scene) {
        visualizationState.value.scene.userData.needsRender = true;
      }
    };

    // XR handlers
    const handleEnableAR = async () => {
      try {
        if (isARActive()) {
          await disableXR();
        } else {
          await enableAR();
        }
      } catch (err) {
        console.error('Failed to toggle AR:', err);
      }
    };

    const handleEnableVR = async () => {
      try {
        if (isVRActive()) {
          await disableXR();
        } else {
          await enableVR();
        }
      } catch (err) {
        console.error('Failed to toggle VR:', err);
      }
    };

    // Update graph data when component mounts
    onMounted(() => {
      console.debug('GraphSystem mounted, initialization state:', {
        isReady: isReady.value,
        hasGraphData: !!graphData.value,
        initialDataRequested: websocketStore.initialDataRequested,
        timestamp: new Date().toISOString()
      });
      
      if (graphData.value) {
        console.debug('Initial graph data update:', {
          nodes: graphData.value.nodes.length,
          edges: graphData.value.edges.length,
          timestamp: new Date().toISOString()
        });
        updateGraphData(graphData.value);
      }
    });

    return {
      isReady,
      platformInfo,
      isXRActive,
      isVRActive,
      isARActive,
      enableAR: handleEnableAR,
      enableVR: handleEnableVR,
      graphGroup,
      hoveredNode,
      isDragging,
      graphData,
      onDragStart,
      onDragMove,
      onDragEnd,
      process: {
        env: {
          NODE_ENV: process.env.NODE_ENV
        }
      }
    };
  }
});
</script>

<style scoped>
.debug-info {
  position: fixed;
  top: 40px;
  left: 10px;
  background: rgba(0, 0, 0, 0.7);
  color: #fff;
  padding: 5px 10px;
  border-radius: 4px;
  font-family: monospace;
  font-size: 12px;
  z-index: 1000;
}

.xr-controls {
  position: fixed;
  bottom: 20px;
  right: 20px;
  display: flex;
  gap: 10px;
  z-index: 1000;
}

.ar-button,
.vr-button {
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
  transition: background-color 0.2s;
}

.ar-button {
  background: #4CAF50;
  color: white;
}

.ar-button:hover:not(:disabled) {
  background: #45a049;
}

.vr-button {
  background: #2196F3;
  color: white;
}

.vr-button:hover:not(:disabled) {
  background: #1976D2;
}

.ar-button:disabled,
.vr-button:disabled {
  background-color: #6c757d;
  cursor: not-allowed;
  opacity: 0.65;
}
</style>
