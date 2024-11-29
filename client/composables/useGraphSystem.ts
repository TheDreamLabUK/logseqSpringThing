import { ref, computed, inject, onMounted } from 'vue';
import { Scene, Group, Vector3 } from 'three';
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
  const hoveredNode = ref<string | null>(null);
  const nodeCount = ref(0);

  // Direct access to binary data
  const getNodePosition = (node: GraphNode | string): Vector3 => {
    const id = typeof node === 'object' ? node.id : node;
    const position = binaryStore.getNodePosition(id);
    if (position) {
      return new Vector3(position[0], position[1], position[2]);
    }
    return new Vector3();
  };

  const getNodeVelocity = (node: GraphNode | string): Vector3 => {
    const id = typeof node === 'object' ? node.id : node;
    const velocity = binaryStore.getNodeVelocity(id);
    if (velocity) {
      return new Vector3(velocity[0], velocity[1], velocity[2]);
    }
    return new Vector3();
  };

  const updateNodePosition = (
    id: string,
    position: Vector3,
    velocity: Vector3
  ) => {
    binaryStore.updatePosition(id, {
      id,
      x: position.x,
      y: position.y,
      z: position.z,
      vx: velocity.x,
      vy: velocity.y,
      vz: velocity.z
    });

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

  const getNodeColor = (node: GraphNode): string => {
    return node.id === hoveredNode.value
      ? settings.value.node_color_core
      : (node.color || settings.value.node_color);
  };

  // Edge helpers using direct access
  const getEdgePoints = (source: GraphNode, target: GraphNode): [Vector3, Vector3] => {
    return [
      getNodePosition(source),
      getNodePosition(target)
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

  // Event handlers
  const handleNodeClick = (node: GraphNode) => {
    const position = getNodePosition(node);
    console.debug('Node clicked:', { id: node.id, position });
  };

  const handleNodeHover = (node: GraphNode | null) => {
    hoveredNode.value = node?.id || null;
    if (visualizationState?.value.scene) {
      visualizationState.value.scene.userData.needsRender = true;
    }
  };

  // Graph data management
  const updateGraphData = (graphData: { nodes: GraphNode[]; edges: GraphEdge[] }) => {
    nodeCount.value = graphData.nodes.length;
    
    // Initialize positions for all nodes
    const positions = graphData.nodes.map(node => ({
      id: node.id,
      x: node.position?.[0] || 0,
      y: node.position?.[1] || 0,
      z: node.position?.[2] || 0,
      vx: node.velocity?.[0] || 0,
      vy: node.velocity?.[1] || 0,
      vz: node.velocity?.[2] || 0
    }));
    
    binaryStore.updatePositions(positions, true);

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
