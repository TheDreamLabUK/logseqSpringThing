import { ref, computed, watch, onMounted, onBeforeUnmount } from 'vue';
import { Vector3, Group } from 'three';
import { useSettingsStore } from '@/stores/settings';
import { usePlatform } from './usePlatform';
import type { GraphNode, GraphEdge } from '@/types/core';
import type { VisualizationConfig } from '@/types/components';

export function useGraphSystem() {
  const settingsStore = useSettingsStore();
  const { getPlatformInfo } = usePlatform();

  // Refs for Three.js objects
  const graphGroup = ref<Group | null>(null);
  const nodesGroup = ref<Group | null>(null);
  const edgesGroup = ref<Group | null>(null);

  // State
  const nodes = ref<GraphNode[]>([]);
  const edges = ref<GraphEdge[]>([]);
  const selectedNode = ref<string | null>(null);
  const hoveredNode = ref<string | null>(null);
  const isSimulating = ref(false);

  // Settings
  const settings = computed(() => settingsStore.getVisualizationSettings);
  const platformInfo = computed(() => getPlatformInfo());

  // Node position management
  const nodePositions = new Map<string, Vector3>();
  const nodeVelocities = new Map<string, Vector3>();

  // Node helpers
  const getNodePosition = (node: GraphNode): Vector3 => {
    if (!nodePositions.has(node.id)) {
      const position = node.position 
        ? new Vector3(...node.position)
        : new Vector3(
            Math.random() * 100 - 50,
            Math.random() * 100 - 50,
            Math.random() * 100 - 50
          );
      nodePositions.set(node.id, position);
    }
    return nodePositions.get(node.id)!;
  };

  const getNodeScale = (node: GraphNode): number => {
    const baseSize = node.size || 1;
    const minSize = settings.value.min_node_size;
    const maxSize = settings.value.max_node_size;
    return minSize + (baseSize * (maxSize - minSize));
  };

  const getNodeColor = (node: GraphNode): string => {
    if (node.id === selectedNode.value) {
      return settings.value.node_color_core;
    }
    if (node.id === hoveredNode.value) {
      return settings.value.node_color_recent;
    }
    return node.color || settings.value.node_color;
  };

  // Edge helpers
  const getEdgePoints = (edge: GraphEdge): [Vector3, Vector3] => {
    const sourceNode = nodes.value.find(n => n.id === edge.source);
    const targetNode = nodes.value.find(n => n.id === edge.target);
    
    if (!sourceNode || !targetNode) {
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

  // Physics simulation
  const applyForces = () => {
    if (!isSimulating.value) return;

    const physics = settings.value.physics;
    
    // Reset forces
    nodes.value.forEach(node => {
      nodeVelocities.set(node.id, new Vector3());
    });

    // Apply spring forces (edges)
    edges.value.forEach(edge => {
      const source = nodes.value.find(n => n.id === edge.source);
      const target = nodes.value.find(n => n.id === edge.target);
      if (!source || !target) return;

      const sourcePos = getNodePosition(source);
      const targetPos = getNodePosition(target);
      const direction = targetPos.clone().sub(sourcePos);
      const distance = direction.length();
      
      // Spring force
      const force = direction.normalize().multiplyScalar(
        (distance - physics.force_directed_spring) * physics.force_directed_attraction
      );

      nodeVelocities.get(source.id)?.add(force);
      nodeVelocities.get(target.id)?.sub(force);
    });

    // Apply repulsion forces
    nodes.value.forEach((node1, i) => {
      nodes.value.slice(i + 1).forEach(node2 => {
        const pos1 = getNodePosition(node1);
        const pos2 = getNodePosition(node2);
        const direction = pos2.clone().sub(pos1);
        const distance = direction.length();
        
        if (distance > 0) {
          const force = direction.normalize().multiplyScalar(
            -physics.force_directed_repulsion / (distance * distance)
          );

          nodeVelocities.get(node1.id)?.add(force);
          nodeVelocities.get(node2.id)?.sub(force);
        }
      });
    });

    // Update positions
    nodes.value.forEach(node => {
      const position = getNodePosition(node);
      const velocity = nodeVelocities.get(node.id);
      if (velocity) {
        position.add(velocity.multiplyScalar(physics.force_directed_damping));
      }
    });
  };

  // Animation loop
  let animationFrameId: number;
  const animate = () => {
    applyForces();
    animationFrameId = requestAnimationFrame(animate);
  };

  // Event handlers
  const handleNodeClick = (node: GraphNode) => {
    selectedNode.value = node.id;
  };

  const handleNodeHover = (node: GraphNode | null) => {
    hoveredNode.value = node?.id || null;
  };

  // Graph data management
  const updateGraphData = (graphData: { nodes: GraphNode[]; edges: GraphEdge[] }) => {
    nodes.value = graphData.nodes;
    edges.value = graphData.edges;

    // Initialize positions for new nodes
    nodes.value.forEach(node => {
      if (!nodePositions.has(node.id)) {
        getNodePosition(node); // This will create a new position
      }
    });
  };

  // Lifecycle
  onMounted(() => {
    if (isSimulating.value) {
      animate();
    }
  });

  onBeforeUnmount(() => {
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
    }
  });

  // Watch for simulation state changes
  watch(() => settings.value.physics.force_directed_iterations, (newValue) => {
    isSimulating.value = newValue > 0;
    if (isSimulating.value && !animationFrameId) {
      animate();
    }
  });

  return {
    // State
    nodes,
    edges,
    selectedNode,
    hoveredNode,
    isSimulating,
    
    // Groups
    graphGroup,
    nodesGroup,
    edgesGroup,
    
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
