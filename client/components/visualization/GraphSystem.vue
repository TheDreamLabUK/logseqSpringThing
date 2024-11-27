<template>
  <Group ref="graphGroup">
    <!-- Nodes -->
    <Group ref="nodesGroup">
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
    </Group>

    <!-- Edges -->
    <Group ref="edgesGroup">
      <template v-for="edge in graphData.edges" :key="`${edge.source}-${edge.target}`">
        <Line
          :points="getEdgePoints(edge)"
          :color="getEdgeColor(edge)"
          :linewidth="getEdgeWidth(edge)"
          :opacity="visualSettings.edge_opacity"
          :transparent="true"
        />
      </template>
    </Group>

    <!-- Scene Fog -->
    <Scene>
      <component
        :is="'fog'"
        :args="[0x000000, 50, 200]"
        :density="visualSettings.fog_density"
      />
    </Scene>
  </Group>
</template>

<script lang="ts">
import { defineComponent, ref, computed, watch, onMounted } from 'vue';
import { Vector3 } from 'three';
import {
  Group,
  Mesh,
  SphereGeometry,
  MeshStandardMaterial,
  Line,
  Html,
  Scene
} from '../three';
import { useGraphSystem } from '../../composables/useGraphSystem';
import { useWebSocketStore } from '../../stores/websocket';
import { useBinaryUpdateStore } from '../../stores/binaryUpdate';
import { useVisualizationStore } from '../../stores/visualization';
import type { GraphNode, GraphEdge, GraphData } from '../../types/core';
import type { VisualizationConfig } from '../../types/components';

export default defineComponent({
  name: 'GraphSystem',

  components: {
    Group,
    Mesh,
    SphereGeometry,
    MeshStandardMaterial,
    Line,
    Html,
    Scene
  },

  props: {
    visualSettings: {
      type: Object as () => VisualizationConfig,
      required: true
    }
  },

  setup(props) {
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

    // Use the transformed graph data from the visualization store
    const graphData = computed<GraphData>(() => {
      const data = visualizationStore.getGraphData || { nodes: [], edges: [], metadata: {} };
      console.debug('Graph data updated:', {
        nodes: data.nodes.length,
        edges: data.edges.length,
        sampleNode: data.nodes[0] ? {
          id: data.nodes[0].id,
          position: data.nodes[0].position,
          edges: data.nodes[0].edges.length
        } : null,
        sampleEdge: data.edges[0] ? {
          source: data.edges[0].source,
          target: data.edges[0].target,
          weight: data.edges[0].weight
        } : null
      });
      return data;
    });

    // Watch for binary position updates from server
    watch(() => binaryUpdateStore.getAllPositions, (positions) => {
      if (!isDragging.value) { // Don't apply server updates while dragging
        console.debug('Binary position update:', {
          count: positions.length,
          sample: positions[0] ? {
            id: positions[0].id,
            position: [positions[0].x, positions[0].y, positions[0].z],
            velocity: [positions[0].vx, positions[0].vy, positions[0].vz]
          } : null
        });
        positions.forEach(pos => {
          const node = graphData.value.nodes.find(n => n.id === pos.id);
          if (node) {
            node.position = [pos.x, pos.y, pos.z];
            node.velocity = [pos.vx, pos.vy, pos.vz];
          }
        });
      }
    }, { deep: true });

    // Enhanced node position getter that considers binary updates
    const getNodePosition = (node: GraphNode) => {
      if (isDragging.value && draggedNode.value?.id === node.id) {
        return getBaseNodePosition(node);
      }

      const binaryPos = binaryUpdateStore.getNodePosition(node.id);
      if (binaryPos) {
        return new Vector3(binaryPos.x, binaryPos.y, binaryPos.z);
      }

      const pos = getBaseNodePosition(node);
      console.debug('Node position:', {
        id: node.id,
        position: pos.toArray(),
        source: binaryPos ? 'binary' : 'base'
      });
      return pos;
    };

    // Enhanced edge points getter that considers binary updates
    const getEdgePoints = (edge: GraphEdge) => {
      const points = getBaseEdgePoints(edge);
      console.debug('Edge points:', {
        id: `${edge.source}-${edge.target}`,
        points: points.map(p => p.toArray())
      });
      return points;
    };

    // Node scale helper
    const nodeScale = (node: GraphNode) => {
      const scale = getNodeScale(node);
      return { x: scale, y: scale, z: scale };
    };

    // Node label position helper
    const nodeLabelPosition = (node: GraphNode) => {
      const pos = getNodePosition(node);
      return new Vector3(
        pos.x,
        pos.y + props.visualSettings.label_vertical_offset,
        pos.z
      );
    };

    // Node emissive intensity helper
    const getNodeEmissiveIntensity = (node: GraphNode) => {
      const { node_emissive_min_intensity, node_emissive_max_intensity } = props.visualSettings.material;
      return node_emissive_min_intensity + (node.weight || 0) * (node_emissive_max_intensity - node_emissive_min_intensity);
    };

    onMounted(() => {
      console.log('GraphSystem mounted:', {
        settings: props.visualSettings,
        initialData: {
          nodes: graphData.value.nodes.length,
          edges: graphData.value.edges.length
        }
      });
    });

    // Drag handlers
    const handleDragStart = (node: GraphNode) => {
      isDragging.value = true;
      draggedNode.value = node;
      dragStartPosition.value = getNodePosition(node).clone();
      console.debug('Drag start:', {
        id: node.id,
        position: dragStartPosition.value.toArray()
      });
    };

    const handleDragMove = (event: PointerEvent) => {
      if (!isDragging.value || !draggedNode.value) return;

      // Update node position based on drag
      const newPosition = getNodePosition(draggedNode.value).clone();
      newPosition.x += event.movementX * 0.1;
      newPosition.y -= event.movementY * 0.1;

      console.debug('Drag move:', {
        id: draggedNode.value.id,
        position: newPosition.toArray(),
        movement: [event.movementX, event.movementY]
      });

      // Send position update to server
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
        // Send final position to server
        const finalPosition = getNodePosition(draggedNode.value);
        console.debug('Drag end:', {
          id: draggedNode.value.id,
          startPosition: dragStartPosition.value.toArray(),
          finalPosition: finalPosition.toArray()
        });

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

    return {
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
      handleDragEnd
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
</style>
