import { ref, computed, inject, onMounted, watch } from 'vue';
import { Vector3, Scene, Group } from 'three';
import { useVisualizationStore } from '../stores/visualization';
import type { GraphNode, GraphEdge } from '../types/core';
import type { VisualizationConfig } from '../types/components';
import type { CoreState } from '../types/core';

export function useGraphSystem() {
  const visualizationStore = useVisualizationStore();
  
  // Get scene from visualization state
  const visualizationState = inject<{ value: CoreState }>('visualizationState');
  const scene = visualizationState?.value.scene;

  if (!scene) {
    throw new Error('Scene not provided to GraphSystem');
  }

  // Create Three.js groups
  const graphGroup = new Group();
  const nodesGroup = new Group();
  const edgesGroup = new Group();

  // Add groups to scene hierarchy
  graphGroup.add(nodesGroup);
  graphGroup.add(edgesGroup);
  scene.add(graphGroup);

  // Initialize scene userData if needed
  scene.userData = scene.userData || {};
  scene.userData.graphGroup = graphGroup;
  scene.userData.nodesGroup = nodesGroup;
  scene.userData.edgesGroup = edgesGroup;

  console.debug('Graph system groups created:', {
    graphGroup: graphGroup.id,
    nodesGroup: nodesGroup.id,
    edgesGroup: edgesGroup.id,
    sceneChildren: scene.children.length
  });

  // State
  const hoveredNode = ref<string | null>(null);

  // Node position management with caching
  const nodePositions = new Map<string, Vector3>();
  const nodeVelocities = new Map<string, Vector3>();
  const positionCache = new Map<string, { position: Vector3; timestamp: number }>();

  // Watch for graph data changes
  watch(() => visualizationStore.getGraphData, (newData) => {
    if (newData) {
      // Mark scene for update
      scene.userData.needsRender = true;
      scene.userData.lastUpdate = performance.now();

      // Update graph data
      updateGraphData(newData);
    }
  }, { deep: true });

  // Get settings from store
  const settings = computed<VisualizationConfig>(() => {
    return visualizationStore.getVisualizationSettings;
  });

  // Node helpers
  const getNodePosition = (node: GraphNode): Vector3 => {
    // Check cache first
    const cached = positionCache.get(node.id);
    const now = Date.now();
    if (cached && now - cached.timestamp < 1000) {
      return cached.position;
    }

    if (!nodePositions.has(node.id)) {
      const position = node.position 
        ? new Vector3(...node.position)
        : new Vector3(
            Math.random() * 100 - 50,
            Math.random() * 100 - 50,
            Math.random() * 100 - 50
          );
      nodePositions.set(node.id, position);
      scene.userData.needsRender = true;
    }

    const position = nodePositions.get(node.id)!;
    positionCache.set(node.id, { position: position.clone(), timestamp: now });
    return position;
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

  // Edge helpers
  const getEdgePoints = (edge: GraphEdge): [Vector3, Vector3] => {
    const sourceNode = edge.sourceNode;
    const targetNode = edge.targetNode;
    
    if (!sourceNode || !targetNode) {
      console.warn('Edge missing nodes:', {
        edge: `${edge.source}-${edge.target}`,
        hasSource: !!sourceNode,
        hasTarget: !!targetNode
      });
      return [new Vector3(), new Vector3()];
    }

    return [getNodePosition(sourceNode), getNodePosition(targetNode)];
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
    console.debug('Node clicked:', {
      id: node.id,
      position: getNodePosition(node).toArray(),
      scale: getNodeScale(node),
      color: getNodeColor(node)
    });
  };

  const handleNodeHover = (node: GraphNode | null) => {
    hoveredNode.value = node?.id || null;
    scene.userData.needsRender = true;
  };

  // Graph data management
  const updateGraphData = (graphData: { nodes: GraphNode[]; edges: GraphEdge[] }) => {
    console.debug('Updating graph data:', {
      nodes: graphData.nodes.length,
      edges: graphData.edges.length
    });

    // Initialize positions for new nodes
    graphData.nodes.forEach(node => {
      if (!nodePositions.has(node.id)) {
        getNodePosition(node);
      }
    });

    // Clean up removed nodes
    const nodeIds = new Set(graphData.nodes.map(n => n.id));
    nodePositions.forEach((_, id) => {
      if (!nodeIds.has(id)) {
        nodePositions.delete(id);
        nodeVelocities.delete(id);
        positionCache.delete(id);
      }
    });

    // Mark scene for update
    scene.userData.needsRender = true;
    scene.userData.lastUpdate = performance.now();
  };

  // Clean up on unmount
  onMounted(() => {
    console.debug('Graph system mounted:', {
      groups: {
        graph: graphGroup.id,
        nodes: nodesGroup.id,
        edges: edgesGroup.id
      },
      sceneChildren: scene.children.length
    });
  });

  return {
    // Groups
    graphGroup,
    nodesGroup,
    edgesGroup,
    
    // State
    hoveredNode,
    
    // Node helpers
    getNodePosition,
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
