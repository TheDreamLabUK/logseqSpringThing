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
import { usePlatform } from '../../composables/usePlatform';
import type { VisualizationConfig } from '../../types/components';
import type { GraphNode, GraphEdge, GraphData, CoreState } from '../../types/core';
import type { PositionUpdate } from '../../types/websocket';

export default defineComponent({
  name: 'GraphSystem',

  props: {
    visualSettings: {
      type: Object as () => VisualizationConfig,
      required: true
    }
  },

  setup(props) {
    // Get visualization state
    const visualizationState = inject<{ value: CoreState }>('visualizationState');
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

    // Watch for binary updates with enhanced logging
    watch(() => binaryUpdateStore.positions, (positions) => {
      const positionEntries = Array.from(positions.entries());
      if (positionEntries.length > 0) {
        console.debug('Processing binary position update:', {
          updateCount: positionEntries.length,
          sample: positionEntries.slice(0, 3).map(([id, pos]) => ({
            id,
            position: [pos.x, pos.y, pos.z],
            velocity: [pos.vx, pos.vy, pos.vz]
          })),
          timestamp: new Date().toISOString()
        });
        
        // Update node positions using updateNodePosition
        let updatedCount = 0;
        positionEntries.forEach(([id, pos]) => {
          const node = graphData.value.nodes.find(n => n.id === id);
          if (node) {
            // Create Vector3 objects for position and velocity
            const position = new Vector3(pos.x, pos.y, pos.z);
            const velocity = new Vector3(pos.vx, pos.vy, pos.vz);
            
            // Update both the graph system and the node data
            updateNodePosition(id, position, velocity);
            node.position = [pos.x, pos.y, pos.z];
            node.velocity = [pos.vx, pos.vy, pos.vz];
            
            updatedCount++;
          }
        });

        console.debug('Position updates applied:', {
          totalUpdates: positionEntries.length,
          successfulUpdates: updatedCount,
          timestamp: new Date().toISOString()
        });

        // Trigger graph update
        if (visualizationState?.value.scene) {
          visualizationState.value.scene.userData.needsRender = true;
        }
      }
    }, { deep: true });

    // Drag handlers with enhanced logging
    const onDragStart = (event: PointerEvent, node: GraphNode) => {
      console.debug('Starting node drag:', {
        nodeId: node.id,
        initialPosition: getNodePosition(node),
        timestamp: new Date().toISOString()
      });

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
        const position = getNodePosition(node);
        position.copy(dragIntersection);

        // Update position with zero velocity during drag
        updateNodePosition(node.id, position, new Vector3(0, 0, 0));

        console.debug('Node drag update:', {
          nodeId: node.id,
          newPosition: [position.x, position.y, position.z],
          timestamp: new Date().toISOString()
        });

        if (websocketStore.service) {
          websocketStore.service.send({
            type: 'updateNodePosition',
            nodeId: node.id,
            position: [position.x, position.y, position.z]
          });
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

      if (websocketStore.service) {
        websocketStore.service.send({
          type: 'updateNodePosition',
          nodeId: draggedNode.value.id,
          position: [finalPosition.x, finalPosition.y, finalPosition.z]
        });
      }

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
        updateGraphData(newData);
      }
    }, { deep: true });

    // Update graph data when component mounts
    onMounted(() => {
      console.debug('GraphSystem mounted');
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
