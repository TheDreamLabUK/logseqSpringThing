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
  
  // Create Three.js groups
  const graphGroup = new Group();
  const nodesGroup = new Group();
  const edgesGroup = new Group();

  // Add groups to scene hierarchy
  graphGroup.add(nodesGroup);
  graphGroup.add(edgesGroup);

  // Initialize scene when available
  watch(() => visualizationState?.value.scene, (scene) => {
    if (scene) {
      console.debug('Initializing graph system with scene:', {
        sceneId: scene.id,
        timestamp: new Date().toISOString()
      });

      // Add groups to scene
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
        sceneChildren: scene.children.length,
        timestamp: new Date().toISOString()
      });
    }
  }, { immediate: true });

  // State
  const hoveredNode = ref<string | null>(null);

  // Node position management with caching and persistence
  const nodePositions = new Map<string, Vector3>();
  const nodeVelocities = new Map<string, Vector3>();
  const positionCache = new Map<string, { position: Vector3; timestamp: number }>();
  const persistentPositions = new Map<string, Vector3>();

  // Node helpers with enhanced logging
  const getNodePosition = (node: GraphNode): Vector3 => {
    // First check persistent positions
    if (persistentPositions.has(node.id)) {
      return persistentPositions.get(node.id)!;
    }

    // Then check cache
    const cached = positionCache.get(node.id);
    const now = Date.now();
    if (cached && now - cached.timestamp < 1000) {
      return cached.position;
    }

    if (!nodePositions.has(node.id)) {
      console.debug('Creating new node position:', {
        nodeId: node.id,
        hasDefinedPosition: !!node.position,
        timestamp: new Date().toISOString()
      });

      const position = node.position 
        ? new Vector3(...node.position)
        : new Vector3(
            Math.random() * 100 - 50,
            Math.random() * 100 - 50,
            Math.random() * 100 - 50
          );
      nodePositions.set(node.id, position);
      // Store in persistent positions
      persistentPositions.set(node.id, position.clone());
      
      if (visualizationState?.value.scene) {
        visualizationState.value.scene.userData.needsRender = true;
      }
    }

    const position = nodePositions.get(node.id)!;
    positionCache.set(node.id, { position: position.clone(), timestamp: now });
    return position;
  };

  const updateNodePosition = (nodeId: string, position: Vector3, velocity: Vector3) => {
    // Update both current and persistent positions
    nodePositions.set(nodeId, position.clone());
    persistentPositions.set(nodeId, position.clone());
    nodeVelocities.set(nodeId, velocity.clone());
    
    // Clear cache entry to force position refresh
    positionCache.delete(nodeId);
    
    if (visualizationState?.value.scene) {
      visualizationState.value.scene.userData.needsRender = true;
    }
  };

  const getNodeScale = (node: GraphNode): number => {
    const baseSize = node.size || 1;
    const minSize = settings.value.min_node_size;
    const maxSize = settings.value.max_node_size;
    const scale = minSize + (baseSize * (maxSize - minSize));

    console.debug('Node scale calculated:', {
      nodeId: node.id,
      baseSize,
      minSize,
      maxSize,
      finalScale: scale,
      timestamp: new Date().toISOString()
    });

    return scale;
  };

  const getNodeColor = (node: GraphNode): string => {
    const color = node.id === hoveredNode.value
      ? settings.value.node_color_core
      : (node.color || settings.value.node_color);

    console.debug('Node color determined:', {
      nodeId: node.id,
      isHovered: node.id === hoveredNode.value,
      hasCustomColor: !!node.color,
      finalColor: color,
      timestamp: new Date().toISOString()
    });

    return color;
  };

  // Edge helpers with enhanced logging
  const getEdgePoints = (edge: GraphEdge): [Vector3, Vector3] => {
    const sourceNode = edge.sourceNode;
    const targetNode = edge.targetNode;
    
    if (!sourceNode || !targetNode) {
      console.warn('Edge missing nodes:', {
        edge: `${edge.source}-${edge.target}`,
        hasSource: !!sourceNode,
        hasTarget: !!targetNode,
        timestamp: new Date().toISOString()
      });
      return [new Vector3(), new Vector3()];
    }

    const sourcePos = getNodePosition(sourceNode);
    const targetPos = getNodePosition(targetNode);

    console.debug('Edge points calculated:', {
      edgeId: `${edge.source}-${edge.target}`,
      sourcePosition: sourcePos.toArray(),
      targetPosition: targetPos.toArray(),
      timestamp: new Date().toISOString()
    });

    return [sourcePos, targetPos];
  };

  const getEdgeColor = (edge: GraphEdge): string => {
    const color = edge.color || settings.value.edge_color;

    console.debug('Edge color determined:', {
      edgeId: `${edge.source}-${edge.target}`,
      hasCustomColor: !!edge.color,
      finalColor: color,
      timestamp: new Date().toISOString()
    });

    return color;
  };

  const getEdgeWidth = (edge: GraphEdge): number => {
    const baseWidth = edge.weight || 1;
    const minWidth = settings.value.edge_min_width;
    const maxWidth = settings.value.edge_max_width;
    const width = minWidth + (baseWidth * (maxWidth - minWidth));

    console.debug('Edge width calculated:', {
      edgeId: `${edge.source}-${edge.target}`,
      baseWidth,
      minWidth,
      maxWidth,
      finalWidth: width,
      timestamp: new Date().toISOString()
    });

    return width;
  };

  // Event handlers with enhanced logging
  const handleNodeClick = (node: GraphNode) => {
    console.debug('Node clicked:', {
      id: node.id,
      position: getNodePosition(node).toArray(),
      scale: getNodeScale(node),
      color: getNodeColor(node),
      timestamp: new Date().toISOString()
    });
  };

  const handleNodeHover = (node: GraphNode | null) => {
    console.debug('Node hover state changed:', {
      previousHovered: hoveredNode.value,
      newHovered: node?.id || null,
      timestamp: new Date().toISOString()
    });

    hoveredNode.value = node?.id || null;
    if (visualizationState?.value.scene) {
      visualizationState.value.scene.userData.needsRender = true;
    }
  };

  // Graph data management with enhanced logging
  const updateGraphData = (graphData: { nodes: GraphNode[]; edges: GraphEdge[] }) => {
    console.debug('Starting graph data update:', {
      nodeCount: graphData.nodes.length,
      edgeCount: graphData.edges.length,
      existingPositions: nodePositions.size,
      timestamp: new Date().toISOString()
    });

    // Initialize positions for new nodes while preserving existing positions
    let newNodeCount = 0;
    graphData.nodes.forEach(node => {
      if (!persistentPositions.has(node.id)) {
        getNodePosition(node);
        newNodeCount++;
      }
    });

    // Clean up removed nodes
    const nodeIds = new Set(graphData.nodes.map(n => n.id));
    let removedNodeCount = 0;
    nodePositions.forEach((_, id) => {
      if (!nodeIds.has(id)) {
        nodePositions.delete(id);
        nodeVelocities.delete(id);
        positionCache.delete(id);
        persistentPositions.delete(id);
        removedNodeCount++;
      }
    });

    console.debug('Graph data update completed:', {
      totalNodes: graphData.nodes.length,
      newNodes: newNodeCount,
      removedNodes: removedNodeCount,
      finalPositionCount: nodePositions.size,
      timestamp: new Date().toISOString()
    });

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

  // Clean up on unmount
  onMounted(() => {
    if (visualizationState?.value.scene) {
      console.debug('Graph system mounted:', {
        groups: {
          graph: graphGroup.id,
          nodes: nodesGroup.id,
          edges: edgesGroup.id
        },
        sceneChildren: visualizationState.value.scene.children.length,
        timestamp: new Date().toISOString()
      });
    }
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
