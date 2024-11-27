<template>
  <div v-if="isReady && graphData">
    <!-- Graph Content -->
    <primitive :object="graphGroup">
      <!-- Nodes -->
      <primitive :object="nodesGroup">
        <template v-for="node in graphData.nodes" :key="node.id">
          <Mesh
            :position="getNodePosition(node)"
            :scale="getNodeScaleVector(node)"
            @click="handleNodeClick(node)"
            @pointerenter="handleNodeHover(node)"
            @pointerleave="handleNodeHover(null)"
            @pointerdown="onDragStart($event, node)"
            @pointermove="onDragMove"
            @pointerup="onDragEnd"
          >
            <SphereGeometry :args="[1, 32, 32]" />
            <MeshStandardMaterial
              :color="getNodeColor(node)"
              :metalness="visualSettings.material.node_material_metalness"
              :roughness="visualSettings.material.node_material_roughness"
              :opacity="visualSettings.material.node_material_opacity"
              :transparent="true"
              :emissive="getNodeColor(node)"
              :emissiveIntensity="getNodeEmissiveIntensity(node)"
            />
          </Mesh>
        </template>
      </primitive>

      <!-- Edges -->
      <primitive :object="edgesGroup">
        <template v-for="edge in graphData.edges" :key="`${edge.source}-${edge.target}`">
          <Line
            :points="getEdgePoints(edge)"
            :color="getEdgeColor(edge)"
            :linewidth="getEdgeWidth(edge)"
            :opacity="visualSettings.edge_opacity"
            :transparent="true"
          />
        </template>
      </primitive>
    </primitive>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, computed, watch, onMounted, inject } from 'vue';
import { Vector3, Vector2, Plane, Raycaster } from 'three';
import * as THREE from 'three';
import {
  Mesh,
  SphereGeometry,
  MeshStandardMaterial,
  Line,
  Html
} from '../three';
import { useGraphSystem } from '../../composables/useGraphSystem';
import { useWebSocketStore } from '../../stores/websocket';
import { useBinaryUpdateStore } from '../../stores/binaryUpdate';
import { useVisualizationStore } from '../../stores/visualization';
import { usePlatform } from '../../composables/usePlatform';
import type { VisualizationConfig } from '../../types/components';
import type { GraphNode, GraphEdge, GraphData, CoreState } from '../../types/core';

export default defineComponent({
  name: 'GraphSystem',
  components: {
    Mesh,
    SphereGeometry,
    MeshStandardMaterial,
    Line,
    Html
  },

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
      nodesGroup,
      edgesGroup,
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

    // Node helpers
    const getNodeScaleVector = (node: GraphNode) => {
      const scale = getNodeScale(node);
      return new Vector3(scale, scale, scale);
    };

    const getNodeEmissiveIntensity = (node: GraphNode) => {
      const { node_emissive_min_intensity, node_emissive_max_intensity } = props.visualSettings.material;
      return node_emissive_min_intensity + (node.weight || 0) * (node_emissive_max_intensity - node_emissive_min_intensity);
    };

    // Drag handlers
    const onDragStart = (event: PointerEvent, node: GraphNode) => {
      isDragging.value = true;
      draggedNode.value = node;

      // Get the camera from visualization state
      const camera = visualizationState?.value.camera;
      if (!camera) return;

      // Create a drag plane perpendicular to the camera
      const normal = new Vector3(0, 0, 1);
      normal.applyQuaternion(camera.quaternion);
      dragPlane.value = new Plane(normal, 0);

      // Store initial position
      dragStartPosition.value = getNodePosition(node).clone();

      // Mark scene for update
      if (visualizationState?.value.scene) {
        visualizationState.value.scene.userData.needsRender = true;
      }
    };

    const onDragMove = (event: PointerEvent) => {
      if (!isDragging.value || !draggedNode.value || !dragPlane.value) return;

      // Get the camera from visualization state
      const camera = visualizationState?.value.camera;
      if (!camera) return;

      const target = event.currentTarget as HTMLElement;
      const rect = target.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      // Create raycaster with Vector2
      const raycaster = new Raycaster();
      const pointer = new Vector2(x, y);
      raycaster.setFromCamera(pointer, camera);

      // Find intersection with drag plane
      if (raycaster.ray.intersectPlane(dragPlane.value, dragIntersection)) {
        // Update node position
        const node = draggedNode.value;
        const position = getNodePosition(node);
        position.copy(dragIntersection);

        // Update position in store if needed
        if (websocketStore.service) {
          websocketStore.service.send({
            type: 'updateNodePosition',
            nodeId: node.id,
            position: [position.x, position.y, position.z]
          });
        }

        // Mark scene for update
        if (visualizationState?.value.scene) {
          visualizationState.value.scene.userData.needsRender = true;
        }
      }
    };

    const onDragEnd = () => {
      if (!isDragging.value || !draggedNode.value || !dragStartPosition.value) return;

      // Get final position
      const finalPosition = getNodePosition(draggedNode.value);

      // Update position in store if needed
      if (websocketStore.service) {
        websocketStore.service.send({
          type: 'updateNodePosition',
          nodeId: draggedNode.value.id,
          position: [finalPosition.x, finalPosition.y, finalPosition.z]
        });
      }

      // Reset drag state
      isDragging.value = false;
      draggedNode.value = null;
      dragStartPosition.value = null;
      dragPlane.value = null;

      // Mark scene for update
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
      updateGraphData(graphData.value);
    });

    return {
      isReady,
      platformInfo,
      getNodeScaleVector,
      isXRActive,
      isVRActive,
      isARActive,
      enableAR: handleEnableAR,
      enableVR: handleEnableVR,
      graphGroup,
      nodesGroup,
      edgesGroup,
      hoveredNode,
      isDragging,
      graphData,
      getNodePosition,
      getNodeScale,
      getNodeColor,
      getNodeEmissiveIntensity,
      getEdgePoints,
      getEdgeColor,
      getEdgeWidth,
      handleNodeClick,
      handleNodeHover,
      onDragStart,
      onDragMove,
      onDragEnd
    };
  }
});
</script>

<style scoped>
.node-label {
  background: v-bind('visualSettings.label_background_color');
  color: v-bind('visualSettings.label_text_color');
  padding: v-bind('visualSettings.label_padding + "px"');
  border-radius: 4px;
  font-size: v-bind('visualSettings.label_font_size + "px"');
  font-family: v-bind('visualSettings.label_font_family');
  white-space: nowrap;
  pointer-events: none;
  transition: transform 0.2s;
}

.node-label.is-hovered {
  transform: scale(1.1);
  background: rgba(0, 0, 0, 0.9);
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
