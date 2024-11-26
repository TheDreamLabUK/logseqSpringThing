<template>
  <Group ref="graphGroup">
    <!-- Nodes -->
    <Group ref="nodesGroup">
      <template v-for="node in nodes" :key="node.id">
        <Mesh
          :position="nodePosition(node)"
          :scale="nodeScale(node)"
          @click="handleNodeClick(node)"
          @pointerenter="handleNodeHover(node, true)"
          @pointerleave="handleNodeHover(node, false)"
        >
          <SphereGeometry :args="[1, 32, 32]" />
          <MeshStandardMaterial
            :color="nodeColor(node)"
            :metalness="visualSettings.material.node_material_metalness"
            :roughness="visualSettings.material.node_material_roughness"
            :opacity="visualSettings.material.node_material_opacity"
            :transparent="true"
          />
        </Mesh>
        
        <!-- Node Label -->
        <Html
          v-if="node.label"
          :position="nodeLabelPosition(node)"
          :occlude="true"
          :center="true"
          :sprite="true"
        >
          <div class="node-label" :class="{ 'is-hovered': hoveredNode === node.id }">
            {{ node.label }}
          </div>
        </Html>
      </template>
    </Group>

    <!-- Edges -->
    <Group ref="edgesGroup">
      <template v-for="edge in edges" :key="`${edge.source}-${edge.target}`">
        <Line
          :points="edgePoints(edge)"
          :color="edgeColor(edge)"
          :linewidth="edgeWidth(edge)"
          :opacity="visualSettings.edge_opacity"
          :transparent="true"
        />
      </template>
    </Group>

    <!-- Force Simulation -->
    <movement-system
      v-if="visualSettings.physics.force_directed_iterations > 0"
      :iterations="visualSettings.physics.force_directed_iterations"
      :spring-strength="visualSettings.physics.force_directed_spring"
      :repulsion="visualSettings.physics.force_directed_repulsion"
      :damping="visualSettings.physics.force_directed_damping"
    >
      <mass-object
        v-for="node in nodes"
        :key="node.id"
        :position="nodePosition(node)"
        :mass="1"
      />
    </movement-system>
  </Group>
</template>

<script lang="ts">
import { defineComponent, ref, computed } from 'vue';
import { Vector3 } from 'three';
import {
  Group,
  Mesh,
  SphereGeometry,
  MeshStandardMaterial,
  Line,
  Html
} from 'vue-threejs';
import type { Node as GraphNode, Edge as GraphEdge } from '../../types/core';
import type { VisualizationConfig } from '../../types/components';

export default defineComponent({
  name: 'GraphSystem',

  components: {
    Group,
    Mesh,
    SphereGeometry,
    MeshStandardMaterial,
    Line,
    Html
  },

  props: {
    nodes: {
      type: Array as () => GraphNode[],
      required: true
    },
    edges: {
      type: Array as () => GraphEdge[],
      required: true
    },
    visualSettings: {
      type: Object as () => VisualizationConfig,
      required: true
    }
  },

  setup(props) {
    const graphGroup = ref(null);
    const nodesGroup = ref(null);
    const edgesGroup = ref(null);
    const hoveredNode = ref<string | null>(null);

    // Node position helper
    const nodePosition = (node: GraphNode) => {
      if (node.position) {
        return { x: node.position[0], y: node.position[1], z: node.position[2] };
      }
      return { x: 0, y: 0, z: 0 };
    };

    // Node scale helper
    const nodeScale = (node: GraphNode) => {
      const baseSize = node.size || 1;
      const minSize = props.visualSettings.min_node_size;
      const maxSize = props.visualSettings.max_node_size;
      const scale = minSize + (baseSize * (maxSize - minSize));
      return { x: scale, y: scale, z: scale };
    };

    // Node color helper
    const nodeColor = (node: GraphNode) => {
      if (node.color) return node.color;
      return props.visualSettings.node_color;
    };

    // Node label position helper
    const nodeLabelPosition = (node: GraphNode) => {
      const pos = nodePosition(node);
      return new Vector3(
        pos.x,
        pos.y + props.visualSettings.label_vertical_offset,
        pos.z
      );
    };

    // Edge points helper
    const edgePoints = computed(() => (edge: GraphEdge) => {
      const sourceNode = props.nodes.find(n => n.id === edge.source);
      const targetNode = props.nodes.find(n => n.id === edge.target);
      
      if (!sourceNode || !targetNode) return [];

      const source = nodePosition(sourceNode);
      const target = nodePosition(targetNode);
      
      return [
        new Vector3(source.x, source.y, source.z),
        new Vector3(target.x, target.y, target.z)
      ];
    });

    // Edge color helper
    const edgeColor = (edge: GraphEdge) => {
      if (edge.color) return edge.color;
      return props.visualSettings.edge_color;
    };

    // Edge width helper
    const edgeWidth = (edge: GraphEdge) => {
      const baseWidth = edge.width || 1;
      const minWidth = props.visualSettings.edge_min_width;
      const maxWidth = props.visualSettings.edge_max_width;
      return minWidth + (baseWidth * (maxWidth - minWidth));
    };

    // Event handlers
    const handleNodeClick = (node: GraphNode) => {
      console.log('Node clicked:', node);
    };

    const handleNodeHover = (node: GraphNode, isHovered: boolean) => {
      hoveredNode.value = isHovered ? node.id : null;
    };

    return {
      graphGroup,
      nodesGroup,
      edgesGroup,
      hoveredNode,
      nodePosition,
      nodeScale,
      nodeColor,
      nodeLabelPosition,
      edgePoints,
      edgeColor,
      edgeWidth,
      handleNodeClick,
      handleNodeHover
    };
  }
});
</script>

<style scoped>
.node-label {
  background: rgba(0, 0, 0, 0.8);
  color: white;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  white-space: nowrap;
  pointer-events: none;
  transition: transform 0.2s;
}

.node-label.is-hovered {
  transform: scale(1.1);
  background: rgba(0, 0, 0, 0.9);
}
</style>
