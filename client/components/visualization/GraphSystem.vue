<template>
  <div v-if="isReady && graphData">
    <!-- Graph Content -->
    <primitive :object="graphGroup">
      <!-- Nodes and edges are now managed by useVisualization -->
    </primitive>
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
      return visualizationState?.value.scene != null && 
             visualizationState?.value.isInitialized === true;
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
      updateGraphData
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

    // Graph data from store
    const graphData = computed<GraphData>(() => {
      return visualizationStore.getGraphData || { nodes: [], edges: [], metadata: {} };
    });

    // Drag handlers
    const onDragStart = (event: PointerEvent, node: GraphNode) => {
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

    // Watch for graph data changes
    watch(() => graphData.value, (newData) => {
      if (newData && newData.nodes.length > 0) {
        updateGraphData(newData);
      }
    }, { deep: true });

    // Update graph data when component mounts
    onMounted(() => {
      if (graphData.value) {
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
      onDragEnd
    };
  }
});
</script>

<style scoped>
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
