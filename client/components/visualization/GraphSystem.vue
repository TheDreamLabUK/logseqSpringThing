<template>
  <div v-if="isReady">
    <!-- Graph Content -->
    <primitive :object="graphGroup">
      <!-- Nodes -->
      <primitive :object="nodesGroup">
        <template v-for="node in graphData.nodes" :key="node.id">
          <Mesh
            :position="getNodePosition(node)"
            :scale="nodeScale(node)"
            @click="handleNodeClick(node)"
            @pointerenter="handleNodeHover(node)"
            @pointerleave="handleNodeHover(null)"
            @pointerdown="handleDragStart(node)"
            @pointermove="handleDragMove"
            @pointerup="handleDragEnd"
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
          
          <!-- Node Label -->
          <Html
            v-if="node.label"
            :position="nodeLabelPosition(node)"
            :occlude="true"
            :center="true"
            :sprite="true"
            :style="{
              fontSize: `${visualSettings.label_font_size}px`,
              fontFamily: visualSettings.label_font_family,
              padding: `${visualSettings.label_padding}px`,
              backgroundColor: visualSettings.label_background_color,
              color: visualSettings.label_text_color
            }"
          >
            <div class="node-label" :class="{ 'is-hovered': hoveredNode === node.id }">
              {{ node.label }}
            </div>
          </Html>
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

    <!-- AR Button -->
    <button 
      v-if="platformInfo.isQuest && platformInfo.hasXRSupport"
      class="ar-button"
      @click="enableAR"
    >
      Enter AR
    </button>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, computed, watch, onMounted, inject } from 'vue';
import { Vector3 } from 'three';
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
import type { GraphNode, GraphEdge, GraphData, CoreState } from '../../types/core';
import type { VisualizationConfig } from '../../types/components';

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
    const { getPlatformInfo, enableVR } = usePlatform();
    const platformInfo = getPlatformInfo();

    const {
      graphGroup,
      nodesGroup,
      edgesGroup,
      hoveredNode,
      getNodePosition: getBaseNodePosition,
      getNodeScale,
      getNodeColor,
      getEdgePoints: getBaseEdgePoints,
      getEdgeColor,
      getEdgeWidth,
      handleNodeClick,
      handleNodeHover,
      updateGraphData
    } = useGraphSystem();

    const websocketStore = useWebSocketStore();
    const binaryUpdateStore = useBinaryUpdateStore();
    const visualizationStore = useVisualizationStore();
    
    const isDragging = ref(false);
    const draggedNode = ref<GraphNode | null>(null);
    const dragStartPosition = ref<Vector3 | null>(null);

    // Graph data from store
    const graphData = computed<GraphData>(() => {
      return visualizationStore.getGraphData || { nodes: [], edges: [], metadata: {} };
    });

    // Watch for binary position updates
    watch(() => binaryUpdateStore.getAllPositions, (positions) => {
      if (!isDragging.value) {
        positions.forEach(pos => {
          const node = graphData.value.nodes.find(n => n.id === pos.id);
          if (node) {
            node.position = [pos.x, pos.y, pos.z];
            node.velocity = [pos.vx, pos.vy, pos.vz];
          }
        });
      }
    }, { deep: true });

    // Position getters
    const getNodePosition = (node: GraphNode) => {
      if (isDragging.value && draggedNode.value?.id === node.id) {
        return getBaseNodePosition(node);
      }

      const binaryPos = binaryUpdateStore.getNodePosition(node.id);
      if (binaryPos) {
        return new Vector3(binaryPos.x, binaryPos.y, binaryPos.z);
      }

      return getBaseNodePosition(node);
    };

    const getEdgePoints = (edge: GraphEdge) => {
      return getBaseEdgePoints(edge);
    };

    // Node helpers
    const nodeScale = (node: GraphNode) => {
      const scale = getNodeScale(node);
      return { x: scale, y: scale, z: scale };
    };

    const nodeLabelPosition = (node: GraphNode) => {
      const pos = getNodePosition(node);
      return new Vector3(
        pos.x,
        pos.y + props.visualSettings.label_vertical_offset,
        pos.z
      );
    };

    const getNodeEmissiveIntensity = (node: GraphNode) => {
      const { node_emissive_min_intensity, node_emissive_max_intensity } = props.visualSettings.material;
      return node_emissive_min_intensity + (node.weight || 0) * (node_emissive_max_intensity - node_emissive_min_intensity);
    };

    // Drag handlers
    const handleDragStart = (node: GraphNode) => {
      isDragging.value = true;
      draggedNode.value = node;
      dragStartPosition.value = getNodePosition(node).clone();
    };

    const handleDragMove = (event: PointerEvent) => {
      if (!isDragging.value || !draggedNode.value) return;

      const newPosition = getNodePosition(draggedNode.value).clone();
      newPosition.x += event.movementX * 0.1;
      newPosition.y -= event.movementY * 0.1;

      if (websocketStore.service) {
        websocketStore.service.send({
          type: 'updateNodePosition',
          nodeId: draggedNode.value.id,
          position: [newPosition.x, newPosition.y, newPosition.z]
        });
      }
    };

    const handleDragEnd = () => {
      if (isDragging.value && draggedNode.value && dragStartPosition.value) {
        const finalPosition = getNodePosition(draggedNode.value);
        if (websocketStore.service) {
          websocketStore.service.send({
            type: 'updateNodePosition',
            nodeId: draggedNode.value.id,
            position: [finalPosition.x, finalPosition.y, finalPosition.z]
          });
        }
      }
      isDragging.value = false;
      draggedNode.value = null;
      dragStartPosition.value = null;
    };

    // AR support
    const enableAR = async () => {
      try {
        await enableVR();
      } catch (err) {
        console.error('Failed to enter AR:', err);
      }
    };

    // Update graph data when component mounts
    onMounted(() => {
      updateGraphData(graphData.value);
    });

    return {
      // Add isReady to the returned object
      isReady,
      // Platform
      platformInfo,
      // Groups
      graphGroup,
      nodesGroup,
      edgesGroup,
      // State
      hoveredNode,
      isDragging,
      graphData,
      // Helpers
      getNodePosition,
      nodeScale,
      getNodeColor,
      getNodeEmissiveIntensity,
      nodeLabelPosition,
      getEdgePoints,
      getEdgeColor,
      getEdgeWidth,
      // Event handlers
      handleNodeClick,
      handleNodeHover,
      handleDragStart,
      handleDragMove,
      handleDragEnd,
      enableAR
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

.ar-button {
  position: fixed;
  bottom: 20px;
  right: 20px;
  padding: 10px 20px;
  background: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
  z-index: 1000;
}

.ar-button:hover {
  background: #45a049;
}
</style>
