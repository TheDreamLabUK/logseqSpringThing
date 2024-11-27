import { ref, computed, onBeforeUnmount } from 'vue';
import { Vector3 } from 'three';
import { useVisualizationStore } from '../stores/visualization';
import type { GraphNode, GraphEdge } from '../types/core';
import type { VisualizationConfig } from '../types/components';

export interface GraphMetrics {
  positionUpdates: number;
  cacheHits: number;
  cacheMisses: number;
}

export function useGraphSystem() {
  const visualizationStore = useVisualizationStore();

  // Refs for Three.js groups
  const graphGroup = ref(null);
  const nodesGroup = ref(null);
  const edgesGroup = ref(null);

  // State
  const hoveredNode = ref<string | null>(null);
  const lastUpdateTime = ref(Date.now());
  const updateCount = ref(0);

  // Node position management with caching
  const nodePositions = new Map<string, Vector3>();
  const nodeVelocities = new Map<string, Vector3>();
  const positionCache = new Map<string, { position: Vector3; timestamp: number }>();

  // Get settings from store
  const settings = computed<VisualizationConfig>(() => visualizationStore.getVisualizationSettings);

  // Performance metrics
  const metrics = ref<GraphMetrics>({
    positionUpdates: 0,
    cacheHits: 0,
    cacheMisses: 0
  });

  // Node helpers
  const getNodePosition = (node: GraphNode): Vector3 => {
    // Check cache first
    const cached = positionCache.get(node.id);
    const now = Date.now();
    if (cached && now - cached.timestamp < 1000) { // 1 second cache
      metrics.value.cacheHits++;
      return cached.position;
    }
    metrics.value.cacheMisses++;

    if (!nodePositions.has(node.id)) {
      const position = node.position 
        ? new Vector3(...node.position)
        : new Vector3(
            Math.random() * 100 - 50,
            Math.random() * 100 - 50,
            Math.random() * 100 - 50
          );
      nodePositions.set(node.id, position);
      console.debug('Created position for node:', {
        id: node.id,
        position: position.toArray()
      });
    }

    const position = nodePositions.get(node.id)!;
    // Update cache
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
    if (node.id === hoveredNode.value) {
      return settings.value.node_color_core;
    }
    return node.color || settings.value.node_color;
  };

  // Edge helpers with caching
  const edgePointsCache = new Map<string, { points: [Vector3, Vector3]; timestamp: number }>();

  const getEdgePoints = (edge: GraphEdge): [Vector3, Vector3] => {
    const cacheKey = `${edge.source}-${edge.target}`;
    const cached = edgePointsCache.get(cacheKey);
    const now = Date.now();
    if (cached && now - cached.timestamp < 1000) {
      metrics.value.cacheHits++;
      return cached.points;
    }
    metrics.value.cacheMisses++;

    const sourceNode = edge.sourceNode;
    const targetNode = edge.targetNode;
    
    if (!sourceNode || !targetNode) {
      console.warn('Edge missing nodes:', edge);
      return [new Vector3(), new Vector3()];
    }

    const points: [Vector3, Vector3] = [getNodePosition(sourceNode), getNodePosition(targetNode)];
    edgePointsCache.set(cacheKey, { points: [points[0].clone(), points[1].clone()], timestamp: now });
    return points;
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
      position: node.position,
      edges: node.edges.length,
      metrics: {
        cacheHits: metrics.value.cacheHits,
        cacheMisses: metrics.value.cacheMisses,
        hitRate: metrics.value.cacheHits / (metrics.value.cacheHits + metrics.value.cacheMisses)
      }
    });
  };

  const handleNodeHover = (node: GraphNode | null) => {
    hoveredNode.value = node?.id || null;
    if (node) {
      console.debug('Node hover:', {
        id: node.id,
        edges: node.edges.length,
        position: nodePositions.get(node.id)?.toArray()
      });
    }
  };

  // Graph data management
  const updateGraphData = (graphData: { nodes: GraphNode[]; edges: GraphEdge[] }) => {
    const now = Date.now();
    const timeSinceLastUpdate = now - lastUpdateTime.value;
    updateCount.value++;
    metrics.value.positionUpdates++;

    console.debug('Updating graph data:', {
      nodes: graphData.nodes.length,
      edges: graphData.edges.length,
      updateInterval: timeSinceLastUpdate,
      updateCount: updateCount.value,
      metrics: {
        cacheHits: metrics.value.cacheHits,
        cacheMisses: metrics.value.cacheMisses,
        hitRate: metrics.value.cacheHits / (metrics.value.cacheHits + metrics.value.cacheMisses)
      },
      sample: graphData.nodes[0] ? {
        id: graphData.nodes[0].id,
        edges: graphData.nodes[0].edges.length,
        position: graphData.nodes[0].position
      } : null
    });

    // Initialize positions for new nodes
    graphData.nodes.forEach(node => {
      if (!nodePositions.has(node.id)) {
        getNodePosition(node); // This will create a new position
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

    // Clean up old edge cache entries
    const edgeIds = new Set(graphData.edges.map(e => `${e.source}-${e.target}`));
    edgePointsCache.forEach((_, key) => {
      if (!edgeIds.has(key)) {
        edgePointsCache.delete(key);
      }
    });

    lastUpdateTime.value = now;
  };

  // Cleanup
  onBeforeUnmount(() => {
    nodePositions.clear();
    nodeVelocities.clear();
    positionCache.clear();
    edgePointsCache.clear();
    metrics.value = {
      positionUpdates: 0,
      cacheHits: 0,
      cacheMisses: 0
    };
  });

  return {
    // Groups
    graphGroup,
    nodesGroup,
    edgesGroup,
    
    // State
    hoveredNode,
    metrics,
    lastUpdateTime,
    updateCount,
    
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
