import { ref, computed, inject, onMounted } from 'vue';
import { Scene, Group } from 'three';
import { useVisualizationStore } from '../stores/visualization';
import { useBinaryUpdateStore } from '../stores/binaryUpdate';
import type { GraphNode, GraphEdge } from '../types/core';
import type { VisualizationConfig } from '../types/components';
import type { CoreState } from '../types/core';

export function useGraphSystem() {
  const visualizationStore = useVisualizationStore();
  const binaryStore = useBinaryUpdateStore();
  
  // Get scene from visualization state
  const visualizationState = inject<{ value: CoreState }>('visualizationState');
  
  // Create Three.js groups
  const graphGroup = new Group();
  const nodesGroup = new Group();
  const edgesGroup = new Group();

  // Add groups to scene hierarchy
  graphGroup.add(nodesGroup);
  graphGroup.add(edgesGroup);

  // State
  const hoveredNode = ref<number | null>(null);
  const nodeCount = ref(0);

  // Direct access to binary data
  const getNodePosition = (index: number): [number, number, number] => {
    const positions = binaryStore.getAllPositions;
    const offset = index * 3;
    return [
      positions[offset],
      positions[offset + 1],
      positions[offset + 2]
    ];
  };

  const getNodeVelocity = (index: number): [number, number, number] => {
    const velocities = binaryStore.getAllVelocities;
    const offset = index * 3;
    return [
      velocities[offset],
      velocities[offset + 1],
      velocities[offset + 2]
    ];
  };

  const updateNodePosition = (
    index: number,
    x: number, y: number, z: number,
    vx: number, vy: number, vz: number
  ) => {
    const positions = binaryStore.getAllPositions;
    const velocities = binaryStore.getAllVelocities;
    const offset = index * 3;

    // Update positions
    positions[offset] = x;
    positions[offset + 1] = y;
    positions[offset + 2] = z;

    // Update velocities
    velocities[offset] = vx;
    velocities[offset + 1] = vy;
    velocities[offset + 2] = vz;

    if (visualizationState?.value.scene) {
      visualizationState.value.scene.userData.needsRender = true;
    }
  };

  const getNodeScale = (node: GraphNode): number => {
    const baseSize = node.size || 1;
    const minSize = settings.value.min_node_size;
    const maxSize = settings.value.max_node_size;
    return minSize + (baseSize * (maxSize - minSize));
  };

  const getNodeColor = (node: GraphNode, index: number): string => {
    return index === hoveredNode.value
      ? settings.value.node_color_core
      : (node.color || settings.value.node_color);
  };

  // Edge helpers using direct array access
  const getEdgePoints = (sourceIndex: number, targetIndex: number): [[number, number, number], [number, number, number]] => {
    return [
      getNodePosition(sourceIndex),
      getNodePosition(targetIndex)
    ];
  };

  const getEdgeColor = (edge: GraphEdge): string => {
    return edge.color || settings.value.edge_color;
  };

  const getEdgeWidth = (edge: GraphEdge): number => {
    const baseWidth = edge.weight || 1;
    const minWidth = settings.value.edge_min_width;
    const maxWidth = settings.value.edge_max_width;
    return minWidth + (baseWidth * (maxWidth - minWidth));
  };

  // Event handlers with direct array access
  const handleNodeClick = (index: number) => {
    const position = getNodePosition(index);
    console.debug('Node clicked:', { index, position });
  };

  const handleNodeHover = (index: number | null) => {
    hoveredNode.value = index;
    if (visualizationState?.value.scene) {
      visualizationState.value.scene.userData.needsRender = true;
    }
  };

  // Graph data management
  const updateGraphData = (graphData: { nodes: GraphNode[]; edges: GraphEdge[] }) => {
    nodeCount.value = graphData.nodes.length;
    
    // Binary store will handle the actual position/velocity arrays
    binaryStore.updateFromBinary(
      new Float32Array(nodeCount.value * 6 + 1).buffer, // +1 for header
      true
    );

    // Mark scene for update
    if (visualizationState?.value.scene) {
      visualizationState.value.scene.userData.needsRender = true;
      visualizationState.value.scene.userData.lastUpdate = performance.now();
    }
  };

  // Get settings from store
  const settings = computed<VisualizationConfig>(() => {
    return visualizationStore.getVisualizationSettings;
  });

  // Initialize scene when available
  onMounted(() => {
    if (visualizationState?.value.scene) {
      visualizationState.value.scene.add(graphGroup);
      visualizationState.value.scene.userData.graphGroup = graphGroup;
      visualizationState.value.scene.userData.nodesGroup = nodesGroup;
      visualizationState.value.scene.userData.edgesGroup = edgesGroup;
    }
  });

  return {
    // Groups
    graphGroup,
    nodesGroup,
    edgesGroup,
    
    // State
    hoveredNode,
    nodeCount,
    
    // Node helpers
    getNodePosition,
    getNodeVelocity,
    updateNodePosition,
    getNodeScale,
    getNodeColor,
    
    // Edge helpers
    getEdgePoints,
    getEdgeColor,
    getEdgeWidth,
    
    // Event handlers
    handleNodeClick,
    handleNodeHover,
    
    // Data management
    updateGraphData
  };
}
