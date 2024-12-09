import { ref, computed, onBeforeUnmount } from 'vue';
import * as THREE from 'three';
import { useVisualizationStore } from '../stores/visualization';
import { usePlatform } from './usePlatform';
import { initXRInteraction } from '../xr/xrInteraction';
import type { GraphNode, GraphEdge, GraphData } from '../types/core';
import type { Scene, PerspectiveCamera, WebGLRenderer } from 'three';

// Performance tuning constants
const MIN_UPDATE_INTERVAL = 200 // 5 FPS
const BATCH_UPDATE_SIZE = 1000 // Number of nodes to update per frame
const FRAME_BUDGET_MS = 200 // Target 5 FPS
const PERFORMANCE_SAMPLE_SIZE = 60 // Number of frames to average for performance metrics

// Geometry constants
const NODE_SEGMENTS = 32
const NODE_RINGS = 32
const MIN_NODE_SIZE = 0.02
const MAX_NODE_SIZE = 0.1

// Material constants
const DEFAULT_NODE_COLOR = 0x666666
const HOVERED_NODE_COLOR = 0x00ff00
const SELECTED_NODE_COLOR = 0xff0000
const DEFAULT_EDGE_COLOR = 0x666666
const MIN_EDGE_WIDTH = 0.005
const MAX_EDGE_WIDTH = 0.02

export function useGraphSystem() {
  const visualizationStore = useVisualizationStore();
  const platform = usePlatform();
  
  // Three.js objects
  const graphGroup = new THREE.Group();
  const nodeGeometry = new THREE.SphereGeometry(1, NODE_SEGMENTS, NODE_RINGS);
  const nodeMaterial = new THREE.MeshPhysicalMaterial({
    color: DEFAULT_NODE_COLOR,
    metalness: 0.3,
    roughness: 0.7,
    emissive: DEFAULT_NODE_COLOR,
    emissiveIntensity: 0.2
  });
  
  // State
  const nodes = ref<Map<string, THREE.Mesh>>(new Map());
  const edges = ref<Map<string, THREE.Line>>(new Map());
  const hoveredNode = ref<string | null>(null);
  const selectedNode = ref<string | null>(null);
  const nodeCount = ref(0);
  const edgeCount = ref(0);
  const lastUpdateTime = ref(0);
  const performanceMetrics = ref({
    updateTime: 0,
    frameTime: 0,
    samples: [] as number[]
  });

  // XR interaction handler
  let xrInteractionHandler: any = null;

  // Node position and scale helpers
  const getNodePosition = (node: GraphNode): THREE.Vector3 => {
    const position = node.position || [0, 0, 0];
    return new THREE.Vector3(position[0], position[1], position[2]);
  };

  const getNodeScale = (node: GraphNode): number => {
    const size = node.size || 1;
    return THREE.MathUtils.clamp(size * MIN_NODE_SIZE, MIN_NODE_SIZE, MAX_NODE_SIZE);
  };

  const getNodeColor = (node: GraphNode): THREE.Color => {
    if (node.id === selectedNode.value) {
      return new THREE.Color(SELECTED_NODE_COLOR);
    }
    if (node.id === hoveredNode.value) {
      return new THREE.Color(HOVERED_NODE_COLOR);
    }
    return new THREE.Color(node.color || DEFAULT_NODE_COLOR);
  };

  // Edge helpers
  const getEdgePoints = (edge: GraphEdge): [THREE.Vector3, THREE.Vector3] => {
    const sourcePos = getNodePosition(edge.sourceNode);
    const targetPos = getNodePosition(edge.targetNode);
    return [sourcePos, targetPos];
  };

  const getEdgeColor = (edge: GraphEdge): THREE.Color => {
    return new THREE.Color(edge.color || DEFAULT_EDGE_COLOR);
  };

  const getEdgeWidth = (edge: GraphEdge): number => {
    const weight = edge.weight || 1;
    return THREE.MathUtils.clamp(
      weight * MIN_EDGE_WIDTH,
      MIN_EDGE_WIDTH,
      MAX_EDGE_WIDTH
    );
  };

  // Event handlers
  const handleNodeClick = (nodeId: string) => {
    selectedNode.value = nodeId;
    visualizationStore.updateNode(nodeId, { userData: { ...visualizationStore.getNodeById(nodeId)?.userData, selected: true } });

    // Provide haptic feedback in XR mode
    if (platform.isXRActive()) {
      platform.vibrate('right', 0.5, 50);
    }
  };

  const handleNodeHover = (nodeId: string | null) => {
    hoveredNode.value = nodeId;
    if (nodeId) {
      visualizationStore.updateNode(nodeId, { userData: { ...visualizationStore.getNodeById(nodeId)?.userData, hovered: true } });
    }
  };

  // Update methods
  const updateNodePosition = (nodeId: string, position: THREE.Vector3, velocity?: THREE.Vector3) => {
    const now = performance.now();
    if (now - lastUpdateTime.value < MIN_UPDATE_INTERVAL) {
      return;
    }

    const mesh = nodes.value.get(nodeId);
    if (mesh) {
      mesh.position.copy(position);
      if (velocity) {
        mesh.userData.velocity = velocity;
      }

      // Send position update to server at 5 FPS
      visualizationStore.updateNode(nodeId, {
        position: [position.x, position.y, position.z],
        velocity: velocity ? [velocity.x, velocity.y, velocity.z] : undefined
      });

      lastUpdateTime.value = now;
    }
  };

  const updateGraphData = (data: GraphData) => {
    const now = performance.now();
    if (now - lastUpdateTime.value < MIN_UPDATE_INTERVAL) {
      return;
    }

    // Update nodes
    const currentNodeIds = new Set<string>();
    data.nodes.forEach((node, index) => {
      currentNodeIds.add(node.id);
      let mesh = nodes.value.get(node.id);

      if (!mesh) {
        // Create new node
        mesh = new THREE.Mesh(nodeGeometry, nodeMaterial.clone());
        mesh.userData = { id: node.id, type: 'node', data: node };
        nodes.value.set(node.id, mesh);
        graphGroup.add(mesh);

        // Make node interactable in XR
        if (xrInteractionHandler) {
          xrInteractionHandler.makeInteractable(mesh);
        }
      }

      // Update position and scale
      mesh.position.copy(getNodePosition(node));
      mesh.scale.setScalar(getNodeScale(node));

      // Update material
      const material = mesh.material as THREE.MeshPhysicalMaterial;
      const color = getNodeColor(node);
      material.color.copy(color);
      material.emissive.copy(color);

      // Process in batches to maintain performance
      if (index % BATCH_UPDATE_SIZE === 0) {
        if (performance.now() - now > FRAME_BUDGET_MS) {
          requestAnimationFrame(() => updateGraphData(data));
          return;
        }
      }
    });

    // Remove old nodes
    nodes.value.forEach((mesh, id) => {
      if (!currentNodeIds.has(id)) {
        if (xrInteractionHandler) {
          xrInteractionHandler.removeInteractable(mesh);
        }
        graphGroup.remove(mesh);
        nodes.value.delete(id);
      }
    });

    // Update edges
    const currentEdgeIds = new Set<string>();
    data.edges.forEach(edge => {
      const edgeId = `${edge.source}-${edge.target}`;
      currentEdgeIds.add(edgeId);
      let line = edges.value.get(edgeId);

      if (!line) {
        // Create new edge
        const geometry = new THREE.BufferGeometry();
        const material = new THREE.LineBasicMaterial({
          color: getEdgeColor(edge),
          linewidth: getEdgeWidth(edge)
        });
        line = new THREE.Line(geometry, material);
        line.userData = { id: edgeId, type: 'edge', data: edge };
        edges.value.set(edgeId, line);
        graphGroup.add(line);
      }

      // Update position
      const [start, end] = getEdgePoints(edge);
      const positions = new Float32Array([
        start.x, start.y, start.z,
        end.x, end.y, end.z
      ]);
      line.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    });

    // Remove old edges
    edges.value.forEach((line, id) => {
      if (!currentEdgeIds.has(id)) {
        graphGroup.remove(line);
        edges.value.delete(id);
      }
    });

    // Update counts
    nodeCount.value = nodes.value.size;
    edgeCount.value = edges.value.size;

    // Update performance metrics
    const updateTime = performance.now() - now;
    performanceMetrics.value.updateTime = updateTime;
    performanceMetrics.value.samples.push(updateTime);
    if (performanceMetrics.value.samples.length > PERFORMANCE_SAMPLE_SIZE) {
      performanceMetrics.value.samples.shift();
    }
    performanceMetrics.value.frameTime = performanceMetrics.value.samples.reduce((a, b) => a + b, 0) / 
                                       performanceMetrics.value.samples.length;

    lastUpdateTime.value = now;
  };

  // Initialize XR interaction if available
  const initializeXR = (scene: Scene, camera: PerspectiveCamera, renderer: WebGLRenderer) => {
    if (platform.hasXRSupport()) {
      // Initialize XR interaction handler with position update callback
      xrInteractionHandler = initXRInteraction(
        scene, 
        camera, 
        renderer,
        (nodeId: string, position: THREE.Vector3) => {
          updateNodePosition(nodeId, position);
        }
      );
      
      // Make existing nodes interactable
      nodes.value.forEach(mesh => {
        xrInteractionHandler.makeInteractable(mesh);
      });
    }
  };

  // Cleanup
  onBeforeUnmount(() => {
    if (xrInteractionHandler) {
      xrInteractionHandler.cleanup();
    }

    nodes.value.forEach(mesh => {
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    });
    edges.value.forEach(line => {
      line.geometry.dispose();
      (line.material as THREE.Material).dispose();
    });
    nodes.value.clear();
    edges.value.clear();
  });

  return {
    graphGroup,
    hoveredNode: computed(() => hoveredNode.value),
    selectedNode: computed(() => selectedNode.value),
    nodeCount: computed(() => nodeCount.value),
    edgeCount: computed(() => edgeCount.value),
    performanceMetrics: computed(() => performanceMetrics.value),
    getNodePosition,
    getNodeScale,
    getNodeColor,
    getEdgePoints,
    getEdgeColor,
    getEdgeWidth,
    handleNodeClick,
    handleNodeHover,
    updateNodePosition,
    updateGraphData,
    initializeXR
  };
}
