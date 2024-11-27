import { ref, computed } from 'vue';
import { Vector3, Scene, WebGLRenderer, PerspectiveCamera } from 'three';
import { useVisualizationStore } from '../stores/visualization';
import type { GraphNode, GraphEdge } from '../types/core';
import type { VisualizationConfig } from '../types/components';

export function useGraphSystem() {
  const visualizationStore = useVisualizationStore();

  // Refs for Three.js groups
  const graphGroup = ref(null);
  const nodesGroup = ref(null);
  const edgesGroup = ref(null);

  // Three.js core components
  const scene = ref<Scene | null>(null);
  const renderer = ref<WebGLRenderer | null>(null);
  const camera = ref<PerspectiveCamera | null>(null);

  // State
  const hoveredNode = ref<string | null>(null);

  // Node position management with caching
  const nodePositions = new Map<string, Vector3>();
  const nodeVelocities = new Map<string, Vector3>();
  const positionCache = new Map<string, { position: Vector3; timestamp: number }>();

  // Get settings from store
  const settings = computed<VisualizationConfig>(() => {
    const config = visualizationStore.getVisualizationSettings;
    console.debug('Visualization settings updated:', {
      material: {
        metalness: config.material.node_material_metalness,
        roughness: config.material.node_material_roughness,
        opacity: config.material.node_material_opacity
      },
      nodeColors: {
        base: config.node_color,
        core: config.node_color_core
      },
      sizes: {
        min: config.min_node_size,
        max: config.max_node_size
      }
    });
    return config;
  });

  // Node helpers
  const getNodePosition = (node: GraphNode): Vector3 => {
    // Check cache first
    const cached = positionCache.get(node.id);
    const now = Date.now();
    if (cached && now - cached.timestamp < 1000) { // 1 second cache
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
      console.debug('Created position for node:', {
        id: node.id,
        position: position.toArray(),
        source: node.position ? 'data' : 'random'
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
    const scale = minSize + (baseSize * (maxSize - minSize));
    console.debug('Node scale:', {
      id: node.id,
      baseSize,
      minSize,
      maxSize,
      finalScale: scale
    });
    return scale;
  };

  const getNodeColor = (node: GraphNode): string => {
    const color = node.id === hoveredNode.value
      ? settings.value.node_color_core
      : (node.color || settings.value.node_color);
    console.debug('Node color:', {
      id: node.id,
      isHovered: node.id === hoveredNode.value,
      color,
      defaultColor: settings.value.node_color
    });
    return color;
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

    const points: [Vector3, Vector3] = [getNodePosition(sourceNode), getNodePosition(targetNode)];
    console.debug('Edge points:', {
      edge: `${edge.source}-${edge.target}`,
      source: points[0].toArray(),
      target: points[1].toArray()
    });
    return points;
  };

  const getEdgeColor = (edge: GraphEdge): string => {
    const color = edge.color || settings.value.edge_color;
    console.debug('Edge color:', {
      edge: `${edge.source}-${edge.target}`,
      color,
      defaultColor: settings.value.edge_color
    });
    return color;
  };

  const getEdgeWidth = (edge: GraphEdge): number => {
    const baseWidth = edge.weight || 1;
    const minWidth = settings.value.edge_min_width;
    const maxWidth = settings.value.edge_max_width;
    const width = minWidth + (baseWidth * (maxWidth - minWidth));
    console.debug('Edge width:', {
      edge: `${edge.source}-${edge.target}`,
      baseWidth,
      minWidth,
      maxWidth,
      finalWidth: width
    });
    return width;
  };

  // Event handlers
  const handleNodeClick = (node: GraphNode) => {
    console.debug('Node clicked:', {
      id: node.id,
      position: node.position,
      edges: node.edges.length,
      color: getNodeColor(node),
      scale: getNodeScale(node)
    });
  };

  const handleNodeHover = (node: GraphNode | null) => {
    hoveredNode.value = node?.id || null;
    if (node) {
      console.debug('Node hover:', {
        id: node.id,
        edges: node.edges.length,
        position: nodePositions.get(node.id)?.toArray(),
        color: getNodeColor(node)
      });
    }
  };

  // Graph data management
  const updateGraphData = (graphData: { nodes: GraphNode[]; edges: GraphEdge[] }) => {
    console.debug('Updating graph data:', {
      nodes: {
        count: graphData.nodes.length,
        sample: graphData.nodes[0] ? {
          id: graphData.nodes[0].id,
          edges: graphData.nodes[0].edges.length,
          position: graphData.nodes[0].position,
          color: getNodeColor(graphData.nodes[0]),
          scale: getNodeScale(graphData.nodes[0])
        } : null
      },
      edges: {
        count: graphData.edges.length,
        sample: graphData.edges[0] ? {
          source: graphData.edges[0].source,
          target: graphData.edges[0].target,
          color: getEdgeColor(graphData.edges[0]),
          width: getEdgeWidth(graphData.edges[0])
        } : null
      },
      cacheStats: {
        positions: nodePositions.size,
        velocities: nodeVelocities.size,
        positionCache: positionCache.size
      }
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
        console.debug('Removing node data:', {
          id,
          hadPosition: nodePositions.delete(id),
          hadVelocity: nodeVelocities.delete(id),
          hadCache: positionCache.delete(id)
        });
      }
    });
  };

  return {
    // Groups
    graphGroup,
    nodesGroup,
    edgesGroup,
    
    // Core components
    scene,
    renderer,
    camera,
    
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
